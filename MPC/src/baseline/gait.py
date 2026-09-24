"""Gait 스케줄 + Raibert 착지점 + swing 궤적/임피던스 (명세 5, 6절).

역할 구분:
  - Gait        : 시간 -> 발별 stance/swing (고정 타이밍, Di Carlo 방식)
  - raibert_target : 다음 착지점  p = hip + v·T_st/2 + k(v − v_cmd)
  - SwingController: 든 발을 착지점까지 아치 궤적으로 옮기는 임피던스 (MPC 무관)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco

from mpc_srb import FOOT_SITES, rz


# ---------------------------------------------------------------------------
# 접촉 스케줄
# ---------------------------------------------------------------------------
# ★ 보행 스케줄 켜기/끄기 (Q&A 9/24 Q8)
#   1 = 켬 (기본): 스케줄대로 발을 번갈아 든다 — --vx 0 이면 제자리 스텝, --vx 0.5 면 전진
#   0 = 끔       : 양발이 항상 stance — 가만히 서 있기. --vx 0 과 같이 쓴다
#                  (vx ≠ 0 이면 발은 붙어 있는데 참조만 앞으로 가서 몸이 기운다)
#   끄면 phase() 가 늘 0 을 돌려준다 = 시작 전(t < t_start) 상태가 끝없이 이어진다.
#   in_stance / swing_phase / contact_table / time_to_touchdown 이 모두 phase() 를 거치므로
#   MPC 접촉표·스윙 제어·착지점 계획이 전부 "양발 디딤" 으로 따라온다.
#   이 값을 바꾸거나, 09_walk / 23_walk_affine / 24_walk_split 에 --nogait 를 주면 그 실행만 0 이 된다.
GAIT_ON = 1


@dataclass
class Gait:
    """고정 타이밍 보행 스케줄. 발 i 의 위상 φ_i = ((t−t_start)/T + offset_i) mod 1,
    φ < stance_frac 이면 stance, 아니면 swing.  offsets=(0, 0.5) → 좌우 교대,
    φ=0 근처는 양발 지지(double support)가 자연히 생긴다 (stance_frac > 0.5)."""
    T: float = 0.8                 # 보행 주기 [s]
    stance_frac: float = 0.75      # stance 비율 (swing = 0.25 T = 0.2 s)
    offsets: tuple = (0.0, 0.5)    # (왼발, 오른발) 위상 오프셋
    t_start: float = 0.5           # 이 시각 전에는 둘 다 stance (서 있기)

    def phase(self, t: float, i: int) -> float:
        if not GAIT_ON or t < self.t_start:      # 스케줄 끔 = 영원히 시작 전 (양발 stance)
            return 0.0
        return ((t - self.t_start) / self.T + self.offsets[i]) % 1.0

    def in_stance(self, t: float, i: int) -> bool:
        return self.phase(t, i) < self.stance_frac

    def swing_phase(self, t: float, i: int) -> float:
        """swing 진행도 s ∈ [0,1). stance 면 0."""
        ph = self.phase(t, i)
        if ph < self.stance_frac:
            return 0.0
        return (ph - self.stance_frac) / (1.0 - self.stance_frac)

    @property
    def T_swing(self) -> float:
        return (1.0 - self.stance_frac) * self.T

    @property
    def T_stance(self) -> float:
        return self.stance_frac * self.T

    def contact_table(self, t: float, dt: float, N: int) -> np.ndarray:
        """(N,2) bool — 지평 스텝 k(구간 중점 기준)의 접촉 여부. MPC 용."""
        tbl = np.zeros((N, 2), dtype=bool)
        for k in range(N):
            tk = t + (k + 0.5) * dt
            for i in range(2):
                tbl[k, i] = self.in_stance(tk, i)
        return tbl

    def time_to_touchdown(self, t: float, i: int) -> float:
        """발 i 가 다음에 착지하는 시각까지 남은 시간 (지금 stance 면 0)."""
        if self.in_stance(t, i):
            return 0.0
        return (1.0 - self.phase(t, i)) * self.T


# ---------------------------------------------------------------------------
# Raibert 착지점
# ---------------------------------------------------------------------------
def raibert_target(p_com, v_com, yaw, side_offset, v_cmd, T_stance, z_com,
                   anchor_xy=None, k_anchor=0.15, max_step=0.30, min_y_sep=0.06,
                   z_ground=0.0331, cap_y_max=None, lip_exact=False, ff_scale=1.0):
    """착지점 (world, capture point 기반).

        p = CoM + R·offset + v_cmd·T_st/2 + (v − v_cmd)/ω₀ + k_a(anchor − CoM)
        ω₀ = √(g/z_com)  (선형 역진자 고유진동수, 여기선 ≈3.8 rad/s → 1/ω₀ ≈ 0.26 s)

    역사: v·T_st/2 + k·v (Raibert 원형, 속도계수 0.48) 는 이 로봇/주기에서
    과잉 스텝 → 진동 성장으로 9.5 s 만에 넘어짐 (Step5 디버깅 3차).
    캡처 계수 1/ω₀=0.26 으로 교체. anchor 는 발디딤 표류 방지용 약한 위치 되당김.
    """
    w0 = np.sqrt(9.81 / max(z_com - z_ground, 0.3))   # LIP 높이 = CoM−지면 (리포트 §3)
    R = rz(yaw)[:2, :2]
    hip = p_com[:2] + R @ side_offset[:2]
    v = v_com[:2]
    # 주의: 이 게인은 민감하다. 1/ω₀ 그대로가 최선이었고 +0.06(9s)·+0.22(9.5s)
    # 모두 악화, β_sway·Q_vy 동시 변경도 악화 (Step6 튜닝 기록 — MPC_NOTES 참고).
    # 중립항·게인의 LIP 스텝간 정확해 (H-LIP). 기존 T_st/2 는 "stance 동안 몸이
    # 이동하는 거리의 절반" 이라는 기하 직관인데, LIP 은 지수적으로 가속·감속하므로
    # 실제 '속도 유지' 착지거리는 δ = (v/ω₀)·tanh(ω₀T/2) 로 더 짧다.
    #   T_st=0.60 → 0.300 vs 0.211 (기존이 42% 과대)
    #   T_st=0.456→ 0.228 vs 0.182 (25% 과대)
    # 과대한 중립항 = 매 스텝 발을 너무 앞에 놓음 = 제동 → 고속에서 속도 추종 실패
    # (0.8 명령에 0.17 실측). 게인도 1/ω₀ → 1/(ω₀·tanh(ω₀T)) 로 (−2~−5% 보정).
    if lip_exact:
        ff_coef = np.tanh(w0 * T_stance / 2.0) / w0
        k_cap = 1.0 / (w0 * np.tanh(w0 * T_stance))
    else:
        ff_coef = T_stance / 2.0
        k_cap = 1.0 / w0
    # 착지 거리 배율 (Q&A 9/21 Q11). reference MIT 는 착지 순간 CoM 대비 발 거리를
    # 속도와 무관하게 ~6 cm 로 묶는데 우리는 발 중앙 기준 12~13 cm (0.5~0.6 m/s).
    # 전진 중립항만 줄이고 capture 게인(속도 피드백)은 그대로 둔다.
    ff_coef *= ff_scale
    cap = (v - v_cmd[:2]) * k_cap
    if cap_y_max is not None:
        # 횡(몸 y) 기여만 상한 — capture 는 '정지'의 답이라 주기 보행의 횡
        # 리미트사이클에선 매 착지 바깥으로 과보행한다 (Q&A Q8: +6.6 cm →
        # 보폭 129 %). 게인을 줄이는 대신 상한을 씌워 평상시 보폭은 좁히고
        # 큰 외란(상한 초과)에만 발이 나가게 한다 (reference 의 '고정 오프셋'
        # 과 우리 '속도 피드백'의 절충).
        cap_b = R.T @ cap
        cap_b[1] = np.clip(cap_b[1], -cap_y_max, cap_y_max)
        cap = R @ cap_b
    p = hip + v_cmd[:2] * ff_coef + cap
    if anchor_xy is not None:
        # 부호 주의: +k·(CoM−ref) = 밀린 쪽으로 '더' 내딛기 (위치 복원, 사용자
        # convex_mpc.m 의 k_y=0.3 과 같은 부호). 반대 부호(중앙으로 당김)는
        # CoM 이 발 바깥에 놓여 위치적으로 불안정 — 23 s 느린 발산의 용의자였다.
        p += k_anchor * (p_com[:2] - np.asarray(anchor_xy, float))

    # 스텝 길이 제한
    d = p - hip
    n = np.linalg.norm(d)
    if n > max_step:
        p = hip + d / n * max_step

    # 교차 방지: 몸 좌표 y 로 변환해 부호 유지
    p_body = R.T @ (p - p_com[:2])
    sign = np.sign(side_offset[1])
    if sign * p_body[1] < min_y_sep:
        p_body[1] = sign * min_y_sep
        p = p_com[:2] + R @ p_body
    return np.array([p[0], p[1], z_ground])


# ---------------------------------------------------------------------------
# Swing 궤적 + 임피던스
# ---------------------------------------------------------------------------
@dataclass
class SwingController:
    """swing 발 하나의 궤적 생성 + 임피던스 토크.

    궤적 (명세 6절 간소화 옵션): xy = smoothstep 보간, z = 시작→끝 보간 + h·sin(πs).
    임피던스: F = Kp(p_des−p) + Kd(v_des−v),  발 수평 유지용 자세 스프링/댐퍼 추가.
    """
    h_swing: float = 0.05
    # kp=350 (1.3 Hz) 은 0.2 s swing 에 너무 무름 — 발이 궤적을 못 쫓아 착지가
    # 한 박자 늦고, '스케줄상 stance 인데 공중' 갭이 매 스텝 roll 킥을 만들어
    # 발산했다 (Step5 디버깅 1차). 다리 ~5 kg 기준 3.5 Hz 로 상향.
    kp: float = 2500.0
    kd: float = 80.0
    kp_ori: float = 60.0
    kd_ori: float = 5.0
    f_max: float = 200.0          # 반작용 제한 (lam_swing 시 400 — 09_walk 참고)
    # 착지 수직속도 실험 (Q&A Q3): 기존 sin 아치는 s=1 에서 v_z = −h·π/T
    # = −0.785 m/s 로 발을 내리꽂는다. True 면 z 를 2단으로:
    # 전반 smoothstep 상승(z_mid), 후반 Hermite 하강 — 끝 속도를 −v_td 로 지정.
    # v_td=0 (정확히 0 으로 내려놓기)은 접촉 확립이 한계상황이 돼 타이밍이
    # 조금만 어긋나면 '스케줄상 stance 인데 발은 공중' 갭이 생겼다 (제자리
    # pitch 0.91→3.6° 후퇴 실측). 논문(Residual MPC 식 15)처럼 작은 v_td 로
    # '지그시 누르며' 착지 — 충격은 1/8, 접촉은 보장.
    soft_land: bool = False
    # 착지 하강속도 [m/s]. 0 = 정확히 내려놓기 (기본). 0.1 로 '지그시 누르기'를
    # 시험했으나 제자리 pitch 는 안 돌아오고(3.3° vs 3.6°) 0.3 완주가 4.6 s
    # 전도로 무너짐 — 하중 이전 지연은 v_td 가 아니라 접촉 인지(램프/조기접촉)
    # 계층의 몫으로 판명. 파라미터는 그 작업 대비로 보존.
    v_td: float = 0.0
    # 5 차 (최소 저크) 보간 (Q&A 9/22 Q13). 3 차 smoothstep 3s²−2s³ 는 가속도가 s=0·1 에서 최대라
    # 이륙 순간 발에 최대 가속도가 계단처럼 걸린다 (0.5 m/s 에서 수평 ~20 m/s²). 5 차
    # 10s³−15s⁴+6s⁵ 는 양 끝 속도·가속도가 모두 0. xy / z 따로 켠다.
    quintic_xy: bool = False
    quintic_z: bool = False
    p_liftoff: np.ndarray = field(default_factory=lambda: np.zeros(3))
    active: bool = False

    def start(self, p_foot_now: np.ndarray):
        self.p_liftoff = p_foot_now.copy()
        self.active = True

    def stop(self):
        self.active = False

    @staticmethod
    def _prof(s, quintic):
        """보간 곡선 (값, 1 계, 2 계 도함수). 3 차 smoothstep 또는 5 차 최소 저크."""
        if quintic:
            return (s ** 3 * (10.0 - 15.0 * s + 6.0 * s * s),
                    30.0 * s * s * (1.0 - s) ** 2,
                    60.0 * s * (1.0 - s) * (1.0 - 2.0 * s))
        return s * s * (3.0 - 2.0 * s), 6.0 * s * (1.0 - s), 6.0 - 12.0 * s

    def target(self, s: float, p_land: np.ndarray, T_swing: float):
        """swing 진행도 s ∈ [0,1] 에서 (p_des, v_des)."""
        s = float(np.clip(s, 0.0, 1.0))
        p0, p1 = self.p_liftoff, p_land
        sm, dsm, _ = self._prof(s, self.quintic_xy)   # 3 차 smoothstep (기본) 또는 5 차
        p_des = (p0 + (p1 - p0) * sm).copy()
        v_des = ((p1 - p0) * dsm / T_swing).copy()
        if self.soft_land:
            # z 채널을 xy 보간과 분리. 전반: smoothstep 상승 (양 끝 v=0).
            # 후반: Hermite 하강 — 시작 v=0, 끝 v=−v_td (지그시 누르는 착지).
            z0, z1 = p0[2], p1[2]
            zm = max(z0, z1) + self.h_swing
            T2 = 0.5 * T_swing
            if s <= 0.5:
                u = 2.0 * s
                smu, dsmu, _ = self._prof(u, self.quintic_z)
                p_des[2] = z0 + (zm - z0) * smu
                v_des[2] = (zm - z0) * dsmu / T2
            elif self.quintic_z and self.v_td == 0.0:
                u = 2.0 * s - 1.0
                smu, dsmu, _ = self._prof(u, True)
                p_des[2] = zm + (z1 - zm) * smu
                v_des[2] = (z1 - zm) * dsmu / T2
            else:
                u = 2.0 * s - 1.0
                h00 = 2 * u ** 3 - 3 * u ** 2 + 1
                h01 = -2 * u ** 3 + 3 * u ** 2
                h11 = u ** 3 - u ** 2
                p_des[2] = h00 * zm + h01 * z1 + h11 * T2 * (-self.v_td)
                dh00 = 6 * u ** 2 - 6 * u
                dh11 = 3 * u ** 2 - 2 * u
                v_des[2] = (dh00 * zm - dh00 * z1
                            + dh11 * T2 * (-self.v_td)) / T2
        else:
            p_des[2] += self.h_swing * np.sin(np.pi * s)
            v_des[2] += self.h_swing * np.pi * np.cos(np.pi * s) / T_swing
        return p_des, v_des

    def target_acc(self, s: float, p_land: np.ndarray, T_swing: float):
        """궤적의 해석적 가속도 (2라운드 §5 역동역학 항용).
        양극성(전반 +, 후반 −)이 자동으로 담긴다 — smoothstep 2계도함수 6−12s."""
        s = float(np.clip(s, 0.0, 1.0))
        a = (p_land - self.p_liftoff) * self._prof(s, self.quintic_xy)[2] / (T_swing * T_swing)
        a = a.copy()
        if self.soft_land:
            z0, z1 = self.p_liftoff[2], p_land[2]
            zm = max(z0, z1) + self.h_swing
            T2 = 0.5 * T_swing
            if s <= 0.5:
                u = 2.0 * s
                a[2] = (zm - z0) * self._prof(u, self.quintic_z)[2] / (T2 * T2)
            elif self.quintic_z and self.v_td == 0.0:
                u = 2.0 * s - 1.0
                a[2] = (z1 - zm) * self._prof(u, True)[2] / (T2 * T2)
            else:
                u = 2.0 * s - 1.0
                a[2] = ((12 * u - 6) * (zm - z1)
                        + (6 * u - 2) * T2 * (-self.v_td)) / (T2 * T2)
        else:
            a[2] += -self.h_swing * np.pi ** 2 * np.sin(np.pi * s) / (T_swing * T_swing)
        return a

    def wrench(self, m, d, site_name: str, s: float, p_land: np.ndarray,
               T_swing: float, yaw_des: float | None = None,
               kp_axis: np.ndarray | None = None,
               kd_axis: np.ndarray | None = None):
        """이 발에 가할 (F(3), moment(3), jacp, jacr). 토크는 +Jᵀ 로 인가.

        yaw_des: 착지 시 발이 가리킬 방향. 안 주면 yaw 자유 (직진용).
        회전 보행에서 이게 없으면 발이 몸을 따라 돌지 않아 hip yaw 가 감기며
        결국 넘어진다 (18.6 — 회전 지속 실패의 메커니즘).
        kp_axis: 축별 kp (3,). 주면 상수 self.kp 대신 사용 — Λ(q) 스케줄링용
        (kp = ω_n²·diag(Λ), reference OperationalSpaceDynamics 방식. Q&A Q3:
        상수 2500 은 수직 방향 등가 대역폭이 15.6 rad/s 뿐이라 발이 궤적을
        절반밖에 못 쫓았다 — 실측 추종률 48 %).
        """
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)

        p_foot = d.site_xpos[sid]
        v_foot = jacp @ d.qvel
        p_des, v_des = self.target(s, p_land, T_swing)
        kp_eff = self.kp if kp_axis is None else kp_axis
        kd_eff = self.kd if kd_axis is None else kd_axis
        F = kp_eff * (p_des - p_foot) + kd_eff * (v_des - v_foot)
        # 반작용 제한: 다리를 너무 세게 던지면 상체가 반대로 돈다 (Step5 디버깅 2차).
        # 단 전진 보행은 swing 이동거리가 길어 ~250 N 이 필요 — 80 N 은 발을 묶어서
        # 걸려 넘어졌다 (Step6 디버깅 1차). 상체 PD 가 반작용을 받아주므로 200 으로.
        # (lam_swing 실험에선 f_max=400 — 고강성에서 200 은 상시 포화라 kp 가 무의미해짐)
        nF = np.linalg.norm(F)
        if nF > self.f_max:
            F *= self.f_max / nF

        # 발바닥 수평 유지: 발 z축을 world z 로 돌리는 스프링 + 각속도 댐핑
        R_f = d.site_xmat[sid].reshape(3, 3)
        z_f = R_f[:, 2]
        axis = np.cross(z_f, np.array([0.0, 0.0, 1.0]))   # 회전축 (부호 포함)
        w_foot = jacr @ d.qvel
        mom = self.kp_ori * axis - self.kd_ori * w_foot
        if yaw_des is not None:
            yaw_f = np.arctan2(R_f[1, 0], R_f[0, 0])
            e_yaw = (yaw_des - yaw_f + np.pi) % (2 * np.pi) - np.pi
            mom[2] += self.kp_ori * e_yaw
        return F, mom, jacp, jacr
