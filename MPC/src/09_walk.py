"""Step 5-6 — 보행: gait 스케줄 + Raibert + swing 임피던스 + gait MPC.

구조 (매 사이클):
  [100 Hz] 접촉 스케줄 + 참조 + 발 위치 계획 조립 → solve_gait → stance wrench
           swing 발이면 Raibert 착지점 갱신
  [500 Hz] τ = qfrc_bias − Σ_stance JᵀW + Σ_swing Jᵀ(임피던스 F, m)

사용:
  .venv/Scripts/python.exe MPC/src/09_walk.py               # 제자리 스텝 (Step 5)
  .venv/Scripts/python.exe MPC/src/09_walk.py --vx 0.3      # 전진 (Step 6)
  .venv/Scripts/python.exe MPC/src/09_walk.py --view --vx 0.3
"""
from __future__ import annotations

import sys

import numpy as np
import mujoco

import g1_model
import mpc_srb
from mpc_qp import WrenchMPC, actuated_dofs
from mpc_log import MPCLog
from gait import Gait, SwingController, raibert_target

np.set_printoptions(precision=3, suppress=True, linewidth=160)

HORIZON = 16
DT_MPC = 0.05          # 지평 N*dt = 0.8 s = 보행주기 1개 (꽉 채우기).
                       # 0.64 s(80%)보다 0.2 m/s 생존 9.2->13.4 s 개선.
                       # 매트랩 quadruped 도 stride 100% 커버 + dt 0.05 였음.
DECIM = 5              # MPC 100 Hz
RAMP_T0, RAMP_T1 = 2.0, 3.0   # v_cmd 램프 구간


KP_YAWHOLD, KD_YAWHOLD = 100.0, 10.0   # stance 발 yaw 유지 PD (reference 값)


class WalkController:
    def __init__(self, m, d, vx_cmd=0.0, kp_up=60.0, swing_id=False,
                 wz_cmd=0.0, yaw_hold=True,
                 soft_land=False, lam_swing=False, wn_swing=100.0,
                 zeta_swing=0.5, cap_y=None, cycle=0.8, stance_frac=0.75,
                 lip_exact=False, q_py=None, td_mode="cont", gate_ff=True, gate_sy=True, swing_h=0.05):
        self.m = m
        self.adof = actuated_dofs(m)
        # SRB 재료 훅 — 서브클래스(11_walk_srb_upper 등)가 '무엇을 강체로 볼
        # 것인가'만 바꿔치기할 수 있게 params/state 추출을 메서드로 분리.
        self.params = self.make_srb_params(m, d)
        # 보행 전용 가중치: 횡 속도 감쇠 강화 (서 있기 기본값은 mpc_qp.Q_DEFAULT)
        from mpc_qp import Q_DEFAULT
        # 횡 위치 가중치: 이중지지를 줄이면(긴 스윙) 횡 균형을 MPC 가 더 많이
        # 져야 한다 — reference 는 py/px = 5~14 (우리 기본 1). 24절.
        _q = None
        if q_py is not None:
            _q = Q_DEFAULT.copy(); _q[4] = q_py
        self.mpc = WrenchMPC(self.params, horizon=HORIZON, dt=DT_MPC, q_diag=_q,
                     psi0=mpc_srb.quat_to_euler_zyx(d.qpos[3:7])[2])
        # gait 타이밍: 스윙 peak 가속도 a=6L/T_sw² 이라 T_sw 가 짧으면 반작용이
        # 제곱으로 커진다. 기본 0.2 s 는 reference(G1 0.3 s) 대비 2.6배 격렬 —
        # 0.4 m/s 붕괴의 상류 원인 (MPC_NOTES 24절).
        self.gait = Gait(T=cycle, stance_frac=stance_frac, t_start=0.5)
        self.vx_cmd = vx_cmd
        self.wz_cmd = wz_cmd    # 요 각속도 명령 [rad/s]

        x0 = self.get_state(d, 0.0)
        self.yaw0 = x0[2]
        self._yaw_unwrap = x0[2]   # 측정 yaw 언랩 상태 (±180° 랩 대비)
        self.com0 = x0[3:6].copy()
        feet = mpc_srb.get_foot_positions(m, d)
        R2 = mpc_srb.rz(self.yaw0)[:2, :2]
        self.side_offset = [np.append(R2.T @ (feet[i, :2] - x0[3:5]), 0.0)
                            for i in range(2)]          # CoM 기준 발 중립 오프셋
        self.z_ground = feet[0, 2]
        self.sw = [SwingController(), SwingController()]
        # 흔들림 개선 실험 두 갈래 (Q&A Q3, MPC_NOTES 23절):
        #  soft_land: 착지 수직속도 0 (z 2단 smoothstep — 기존 sin 아치는 −0.785 m/s)
        #  lam_swing: 스윙 kp 를 ω_n²·Λ(q) 로 자세별 스케줄 (수직 등가 대역폭
        #             15.6 → 100 rad/s). 고강성이라 힘 상한도 200→400 N.
        self.lam_swing = lam_swing
        self.wn_swing = wn_swing
        self.zeta_swing = zeta_swing
        self.cap_y = cap_y      # capture 횡 기여 상한 [m] (None=무제한, Q&A Q8 §5)
        self.lip_exact = lip_exact   # 착지 중립항·게인을 LIP 정확해로 (24절)
        self.td_mode = td_mode       # 착지 목표 갱신: cont|lp|fz (Q&A 9/16)
        # gate_yaw=False 면 제약 frame(발 실측 yaw)·스윙 yaw 정렬을 직진에서도
        # 켠다. 원래 wz≠0 게이트였는데, 0.5 m/s 직진에서 stance 발이 최대 20°
        # 돌아가 있는 게 측정됐다 — 제약 frame 오차 20° 면 MPC 가 '발 안'이라
        # 믿는 CoP 가 실제로는 옆으로 1.6 cm 밖이다 (반폭 2.5 cm). Q&A 9/16.
        self.gate_ff = gate_ff   # 제약 frame 을 발 실측 yaw 로 (모델 정직화, 공짜)
        self.gate_sy = gate_sy   # 스윙 발 yaw 정렬 (물리 교정, 토크 비용 있음)
        for sc in self.sw:
            sc.soft_land = soft_land
            sc.h_swing = swing_h
            if lam_swing:
                sc.f_max = 400.0
        self.was_stance = [True, True]
        self.p_land = [feet[0].copy(), feet[1].copy()]
        self.wr = np.zeros((2, 6))
        self.last_info = {}
        # 상체(허리+팔) 자세 PD: 중력보상만 받으면 둥둥 떠서 다리 반작용에
        # 휘둘린다. 고정해야 로봇이 SRB 가정에 가까워짐 (Step5 디버깅 2차).
        self.q_ref = d.qpos[7:].copy()
        self.upper_act = [a for a in range(m.nu)
                          if m.jnt_dofadr[m.actuator_trnid[a, 0]] >= 18]
        self.kp_up, self.kd_up = kp_up, kp_up / 15.0
        # 채널② 보상 (2라운드 §5-6): τ += M(q)·q̈_swing,des — swing 다리의 계획된
        # 가속을 역동역학으로 미리 감당. M 이 밀집이라 stance·base 관절에도 보상이
        # 들어가 '실현 wrench ≠ 명령 wrench' 누수를 막는다. J̇q̇ 항은 생략(근사).
        self.swing_id = swing_id
        self.leg_dofs = [list(range(6, 12)), list(range(12, 18))]
        self._Mfull = np.zeros((m.nv, m.nv))
        # stance 발 yaw 유지 (reference 분석 8절 #1): 착지 순간 yaw 를 기록하고
        # stance 중 world-z 모멘트 PD 로 유지 — 회전 시 발이 미끄러지며 감기는
        # 것을 hip_yaw 가 능동적으로 막는다. MPC wrench 밖의 추가 채널.
        self.yaw_hold = yaw_hold
        self.yaw_td = []
        for sname in mpc_srb.FOOT_SITES:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, sname)
            Rf = d.site_xmat[sid].reshape(3, 3)
            self.yaw_td.append(float(np.arctan2(Rf[1, 0], Rf[0, 0])))
        self._st_prev = [True, True]

    # ------------------------------------------------ SRB 재료 훅 (교체 지점)
    def make_srb_params(self, m, d):
        """SRB 파라미터 (질량·관성·발 형상). 기본: 전신."""
        return mpc_srb.make_params(m, d)

    def get_state(self, d, t=None):
        """SRB 상태 x(13). 기본: Θ=pelvis, p·v=전신 CoM, ω=전신 각운동량.
        t: 게이트 시각 — 상체 SRB 가 '매달린 swing 다리' 보정에 쓴다."""
        return mpc_srb.get_state(self.m, d, I_body=self.params.I_body)

    def ramp(self, t):
        return float(np.clip((t - RAMP_T0) / (RAMP_T1 - RAMP_T0), 0.0, 1.0))

    def v_cmd(self, t):
        """몸(yaw) 기준 명령을 world 로 회전해서 반환."""
        r = self.ramp(t)
        v_body = np.array([self.vx_cmd * r, 0.0, 0.0])
        return mpc_srb.rz(self.yaw_ref(t)) @ v_body

    def wz(self, t):
        return self.wz_cmd * self.ramp(t)

    def yaw_ref(self, t):
        """명령 요각속도를 적분한 참조 yaw (램프 구간 해석적 적분)."""
        if abs(self.wz_cmd) < 1e-12:
            return self.yaw0
        T0, T1 = RAMP_T0, RAMP_T1
        if t <= T0:
            integ = 0.0
        elif t <= T1:
            integ = 0.5 * (t - T0) ** 2 / (T1 - T0)
        else:
            integ = 0.5 * (T1 - T0) + (t - T1)
        return self.yaw0 + self.wz_cmd * integ

    # -------------------------------------------------- 100 Hz: MPC
    def update_mpc(self, d, t):
        m = self.m
        x0 = self.get_state(d, t)
        # 측정 yaw 언랩: atan2 는 ±180° 에서 랩되는데 yaw_ref 는 무한 적분이라
        # 랩 순간 MPC 가 −360° 오차를 보고 폭주한다 (0.3 rad/s 가 매번
        # t≈13 s(=ref 180° 도달)에 무너진 원인 — reference 의 yaw unwrap 정책).
        self._yaw_unwrap += (x0[2] - self._yaw_unwrap + np.pi) % (2 * np.pi) - np.pi
        x0[2] = self._yaw_unwrap
        feet = mpc_srb.get_foot_positions(m, d)
        vc = self.v_cmd(t)

        # swing 시작/종료 관리 + Raibert 착지점 갱신
        for i in range(2):
            st = self.gait.in_stance(t, i)
            if self.was_stance[i] and not st:
                self.sw[i].start(feet[i])
            if not st:
                # capture point 배치. y 만 참조에 약하게 앵커 (표류 방지)
                # 회전 시: 공칭 오프셋을 '착지 시점' 참조 yaw 로 회전 (reference
                # SwingFootPlanner 방식) — 발이 미리 돌아간 위치에 딛어야
                # 다음 stance 가 회전을 실어나른다. 직진은 기존(측정 yaw) 유지.
                if abs(self.wz_cmd) > 1e-12:
                    yaw_pl = self.yaw_ref(t + self.gait.time_to_touchdown(t, i))
                else:
                    yaw_pl = x0[2]
                # y 앵커(초기 y₀)는 직선 경로 전제 — 전진+회전(원호)에선 경로를
                # 벗어난 지점으로 끌어당겨 발산시킨다 (±34 cm 실측). 조합에선 해제.
                combo = abs(self.wz_cmd) > 1e-12 and abs(self.vx_cmd) > 1e-9
                raw = raibert_target(
                    x0[3:6], x0[9:12], yaw_pl, self.side_offset[i], vc,
                    self.gait.T_stance, z_com=x0[5], z_ground=self.z_ground,
                    anchor_xy=None if combo else [x0[3], self.com0[1]],
                    cap_y_max=self.cap_y, lip_exact=self.lip_exact)
                # 착지 목표 갱신 정책 (Q&A 9/16 Q1): capture 항이 '순간 v_y' 를
                # 봐서 목표가 한 스윙에 14 cm 쓸려다닌다 (안쪽→바깥쪽) — 발이
                # 모였다 튀는 모션·착지오차 4~5 cm·반작용 낭비의 원인.
                #   cont: 매 틱 그대로 (기존)
                #   lp  : 1차 저역통과 τ=0.1 s — 급변만 제거
                #   fz  : s≥0.5 부터 동결 — 후반에 발이 수렴할 시간 확보
                s_ph = self.gait.swing_phase(t, i)
                just_lifted = self.was_stance[i]
                if just_lifted or self.td_mode == "cont":
                    self.p_land[i] = raw
                elif self.td_mode == "lp":
                    a = 0.095        # 1-exp(-0.01/0.1): 갱신주기 10 ms, τ=0.1 s
                    self.p_land[i] = self.p_land[i] + a * (raw - self.p_land[i])
                elif self.td_mode == "fz":
                    if s_ph < 0.5:
                        self.p_land[i] = raw
            self.was_stance[i] = st

        # 지평 재료 조립
        N = HORIZON
        contact = self.gait.contact_table(t, DT_MPC, N)
        # stage-0 접촉은 '지금' 기준 (중점 t+25ms 는 착지 25ms 전부터 그 발에
        # W>0 을 명령해 torque() 의 swing 처리와 어긋남 — 검증 리포트 §5-(d))
        for i in range(2):
            contact[0, i] = self.gait.in_stance(t, i)
        X_ref = np.zeros((N, mpc_srb.NX))
        foot_traj = np.zeros((N, 2, 3))
        u_ref = np.zeros((N, mpc_srb.NU))
        # 공칭 수직력: x0[12] 는 보통 g 지만, 상체 SRB 는 '매달린 swing 다리'
        # 를 유효 중력으로 반영하므로 (11_walk_srb_upper) u_ref 도 따라간다.
        Mg = self.params.mass * x0[12]
        for i in range(2):
            became_swing = not self.was_stance[i]
            for k in range(N):
                if not contact[k, i]:
                    became_swing = True
                foot_traj[k, i] = self.p_land[i] if became_swing else feet[i]
        BETA_SWAY = 0.5      # 체중이동 참조 강도 (0=항상 중앙, 1=stance 발 위)
        # 전진+회전 조합: 참조 경로가 직선이 아니라 원호 — 회전하는 명령 속도를
        # 지평에 걸쳐 적분 (reference 의 arc reference). 직진/제자리는 기존 유지.
        combo = abs(self.wz_cmd) > 1e-12 and abs(self.vx_cmd) > 1e-9
        arc = np.zeros((N, 2))
        if combo:
            vb = np.array([self.vx_cmd * self.ramp(t), 0.0])
            p_arc = np.zeros(2)
            for k in range(N):
                Rk = mpc_srb.rz(self.yaw_ref(t + (k + 1) * DT_MPC))[:2, :2]
                p_arc = p_arc + Rk @ vb * DT_MPC
                arc[k] = p_arc
        for k in range(N):
            X_ref[k, 2] = self.yaw_ref(t + (k + 1) * DT_MPC)
            X_ref[k, 8] = self.wz(t)          # 요 각속도 참조
            if combo:
                X_ref[k, 3] = x0[3] + arc[k, 0]                  # 원호 추종
            elif abs(self.vx_cmd) > 1e-9:
                X_ref[k, 3] = x0[3] + vc[0] * (k + 1) * DT_MPC   # 속도 추종 모드
            else:
                X_ref[k, 3] = self.com0[0]                       # 제자리 모드
            # sway 참조: 지평 k 의 지지 중심 쪽으로 CoM y 를 미리 이동시킨다.
            # 항상 중앙(0)이면 MPC 가 체중이동 없이 발목으로만 버티다 진동이
            # 누적된다 (Step5 디버깅 4차). preview 가 있어야 가능한 정석 방식.
            # 중앙 = 경로 y: 직선/제자리는 y₀ 고정, 조합은 원호 위의 y.
            st = [i for i in range(2) if contact[k, i]]
            cy = (x0[4] + arc[k, 1]) if combo else self.com0[1]
            y_sup = np.mean([foot_traj[k, i, 1] for i in st]) if st else cy
            X_ref[k, 4] = cy + BETA_SWAY * (y_sup - cy)
            X_ref[k, 5] = self.com0[2]
        # sway 참조를 속도 제한 램프로 스무딩 (스텝 함수 그대로면 전환 순간
        # 참조 점프를 MPC 가 쫓아 옆으로 차버린다 — Step5 디버깅 5차)
        y_prev = x0[4]
        for k in range(N):
            dy = np.clip(X_ref[k, 4] - y_prev, -0.4 * DT_MPC, 0.4 * DT_MPC)
            X_ref[k, 4] = y_prev = y_prev + dy
        for k in range(N):
            if combo:   # 속도 참조도 지평 내 heading 을 따라 회전
                Rk = mpc_srb.rz(self.yaw_ref(t + (k + 1) * DT_MPC))[:2, :2]
                X_ref[k, 9:11] = Rk @ np.array([self.vx_cmd * self.ramp(t), 0.0])
                X_ref[k, 11] = 0.0
            else:
                X_ref[k, 9:12] = vc
            X_ref[k, 8] = self.wz(t)
            X_ref[k, 12] = mpc_srb.GRAV
            st_k = [i for i in range(2) if contact[k, i]]
            for i in st_k:
                u_ref[k, 6 * i + 2] = Mg / max(len(st_k), 1)

        # 선형화 yaw: 회전 명령이 있으면 지평 평균 yaw (Di Carlo 방식), 없으면 yaw0
        psi_lin = float(np.mean(X_ref[:, 2])) if abs(self.wz_cmd) > 1e-12 else self.yaw0
        # 발별 실제 yaw (제약 frame 용). 직진에선 발 yaw 잔진동이 제약 박스를
        # 흔들어 marginal 한 0.3 m/s 를 떨어뜨려서, 회전 명령이 있을 때만 사용.
        if (not self.gate_ff) or abs(self.wz_cmd) > 1e-12:
            psi_feet = []
            for sname in mpc_srb.FOOT_SITES:
                sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, sname)
                Rf = d.site_xmat[sid].reshape(3, 3)
                psi_feet.append(float(np.arctan2(Rf[1, 0], Rf[0, 0])))
        else:
            psi_feet = None
        u0, _, info = self.mpc.solve_gait(x0, X_ref, psi_lin,
                                          foot_traj, contact, u_ref,
                                          psi_feet=psi_feet)
        self.wr = u0.reshape(2, 6)
        self.last_info = info
        return x0, X_ref[0], contact[0]

    # -------------------------------------------------- 500 Hz: 토크
    def torque(self, d, t):
        m = self.m
        tau_c = np.zeros(m.nv)
        for i, s_name in enumerate(mpc_srb.FOOT_SITES):
            st = self.gait.in_stance(t, i)
            if st:
                sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s_name)
                jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
                mujoco.mj_jacSite(m, d, jacp, jacr, sid)
                tau_c -= jacp.T @ self.wr[i, :3] + jacr.T @ self.wr[i, 3:]
                # 직진에선 착지 yaw 산포에 홀드 토크가 얹혀 marginal 0.3 을
                # 떨어뜨림 (12.9s 전도 실측) — 다른 회전 기구와 같이 wz 게이트.
                if self.yaw_hold and abs(self.wz_cmd) > 1e-12:
                    Rf = d.site_xmat[sid].reshape(3, 3)
                    psi_f = float(np.arctan2(Rf[1, 0], Rf[0, 0]))
                    if not self._st_prev[i]:          # 방금 착지 -> yaw 고정점 기록
                        self.yaw_td[i] = psi_f
                    e = (self.yaw_td[i] - psi_f + np.pi) % (2 * np.pi) - np.pi
                    wz_f = float(jacr[2] @ d.qvel)
                    mz_hold = KP_YAWHOLD * e - KD_YAWHOLD * wz_f
                    tau_c += jacr.T @ np.array([0.0, 0.0, mz_hold])
            else:
                s = self.gait.swing_phase(t, i)
                if (not self.gate_sy) or abs(self.wz_cmd) > 1e-12:
                    # 착지 '시점'의 참조 yaw 를 겨냥 (현재 yaw 면 반 스텝 늦다)
                    yd = self.yaw_ref(t + self.gait.time_to_touchdown(t, i))
                    # 회전 안쪽 발 toe-in 바이어스 (reference SwingYawTarget:
                    # 100°/(rad/s), 최대 20°) — 다음 stance 의 회전 여유 선확보.
                    bias = min(np.radians(100.0) * abs(self.wz_cmd),
                               np.radians(20.0))
                    if self.wz_cmd > 0 and i == 0:      # +wz: 왼발이 안쪽
                        yd += bias
                    elif self.wz_cmd < 0 and i == 1:    # -wz: 오른발이 안쪽
                        yd -= bias
                else:
                    yd = None
                kp_axis = kd_axis = None
                if self.lam_swing:
                    # 겉보기 관성 Λ = (J M⁻¹ Jᵀ)⁻¹, kp = ω_n²·diag(Λ) —
                    # 자세가 변해도 발끝 추종 대역폭이 ω_n 으로 일정 (reference
                    # OperationalSpaceDynamics). 상수 kp=2500 은 수직 등가
                    # 대역폭이 15.6 rad/s 뿐 (Λ_z≈10 kg) → 추종률 48 % 실측.
                    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s_name)
                    jp = np.zeros((3, m.nv))
                    mujoco.mj_jacSite(m, d, jp, None, sid)
                    dofs = self.leg_dofs[i]
                    J = jp[:, dofs]
                    mujoco.mj_fullM(m, d, self._Mfull)
                    Ml = self._Mfull[np.ix_(dofs, dofs)]
                    lam_inv = J @ np.linalg.solve(Ml, J.T)
                    lam_inv[np.diag_indices(3)] += 1e-9
                    lam = np.diag(np.linalg.inv(lam_inv))
                    kp_axis = self.wn_swing ** 2 * lam
                    # kd 도 Λ 스케줄 (kd = 2ζ·ω_n·Λ). 1차 시도에서 kd=80 고정
                    # 으로 뒀더니 ζ≈0.04 저감쇠 → swing 이 명령 5 cm 를 22 cm
                    # 로 오버슛하며 11 s 전도. reference 는 Λ(a−J̇q̇) 피드포워드
                    # 가 주역이라 저감쇠로 버티지만 우리 ff 는 더 거친 근사.
                    kd_axis = 2.0 * self.zeta_swing * self.wn_swing * lam
                F, mom, jacp, jacr = self.sw[i].wrench(
                    m, d, s_name, s, self.p_land[i], self.gait.T_swing,
                    yaw_des=yd, kp_axis=kp_axis, kd_axis=kd_axis)
                tau_c += jacp.T @ F + jacr.T @ mom
                if self.swing_id:
                    a_ff = self.sw[i].target_acc(s, self.p_land[i],
                                                 self.gait.T_swing)
                    dofs = self.leg_dofs[i]
                    J6 = np.vstack([jacp, jacr])[:, dofs]
                    qdd, *_ = np.linalg.lstsq(
                        J6, np.concatenate([a_ff, np.zeros(3)]), rcond=None)
                    nq = np.linalg.norm(qdd)
                    if nq > 300.0:                    # 특이자세 보호
                        qdd *= 300.0 / nq
                    qacc = np.zeros(m.nv)
                    qacc[dofs] = qdd
                    mujoco.mj_fullM(m, d, self._Mfull)   # 3.12: (m, d, dst)
                    tau_c += self._Mfull @ qacc
            self._st_prev[i] = st
        tau = d.qfrc_bias[self.adof] + tau_c[self.adof]
        for a in self.upper_act:
            jid = self.m.actuator_trnid[a, 0]
            qa = self.m.jnt_qposadr[jid]
            dofa = self.m.jnt_dofadr[jid]
            tau[a] += self.kp_up * (self.q_ref[qa - 7] - d.qpos[qa])                       - self.kd_up * d.qvel[dofa]
        return tau


# ===========================================================================
LEG_BODY_KEYS = ("hip_pitch_link", "hip_roll_link", "hip_yaw_link",
                 "knee_link", "ankle_pitch_link", "ankle_roll_link")


def scale_leg_mass(m, d, scale):
    """다리 링크 질량·관성 스케일링 — massless-leg 가설 격리 실험용
    (검증 리포트 §5-1: 벽이 사라지면 swing 반작용 확정)."""
    import mujoco as _mj
    changed = []
    for b in range(m.nbody):
        name = _mj.mj_id2name(m, _mj.mjtObj.mjOBJ_BODY, b) or ""
        if any(k in name for k in LEG_BODY_KEYS):
            m.body_mass[b] *= scale
            m.body_inertia[b] *= scale
            changed.append(name)
    _mj.mj_setConst(m, d)
    return changed


def headless(vx=0.0, seconds=12.0, legmass=1.0, kp_up=60.0, swing_id=False,
             wz=0.0, yaw_hold=True, ctor=None, variant="",
             soft_land=False, lam_swing=False, wn_swing=100.0,
             zeta_swing=0.5, cap_y=None, cycle=0.8, stance_frac=0.75,
             lip_exact=False, q_py=None, td_mode="cont",
             gate_ff=True, gate_sy=True, swing_h=0.05):
    """ctor: WalkController 서브클래스 주입 (예: 11_walk_srb_upper).
    variant: 로그 파일명 접두어 — baseline 로그와 섞이지 않게."""
    m, d = g1_model.load_torque()
    g1_model.set_crouch(m, d)
    if abs(legmass - 1.0) > 1e-9:
        n = len(scale_leg_mass(m, d, legmass))
        import mujoco as _mj
        print(f"  [실험] 다리 링크 {n}개 질량 x{legmass} -> 총질량 "
              f"{_mj.mj_getTotalmass(m):.2f} kg")
        g1_model.set_crouch(m, d)   # 질량 변경 후 자세 재설정
    ctl = (ctor or WalkController)(m, d, vx_cmd=vx, kp_up=kp_up,
                                   swing_id=swing_id, wz_cmd=wz,
                                   yaw_hold=yaw_hold, soft_land=soft_land,
                                   lam_swing=lam_swing, wn_swing=wn_swing,
                                   zeta_swing=zeta_swing, cap_y=cap_y,
                                   cycle=cycle, stance_frac=stance_frac,
                                   lip_exact=lip_exact, q_py=q_py,
                                   td_mode=td_mode, gate_ff=gate_ff,
                                   gate_sy=gate_sy, swing_h=swing_h)
    tag = "inplace" if abs(vx) < 1e-9 else "vx" + f"{vx:g}".replace(".", "p")
    if abs(wz) > 1e-12:
        tag += "_wz" + f"{wz:g}".replace(".", "p").replace("-", "m")
    if variant:
        tag = variant + "_" + tag
    log = MPCLog(f"logs/walk_{tag}",
                 note=f"Step5/6: gait MPC N={HORIZON}, T={ctl.gait.T}, vx_cmd={vx}")

    z0 = d.qpos[2]
    dt = m.opt.timestep
    fell = None
    swing_apex = [0.0, 0.0]
    n_swings = [0, 0]
    prev_st = [True, True]
    vx_hist, py_hist = [], []

    print(f"  t[s]  pelvis_z   com_x    com_y   vx     yaw[°] yaw_e[°]  |mz|  접촉  solve_ms")
    next_rep = 0.0
    x0 = None
    for k in range(int(seconds / dt)):
        t = k * dt
        mujoco.mj_forward(m, d)
        if k % DECIM == 0:
            x0, xr0, c0 = ctl.update_mpc(d, t)
            log.add(t=t, x=x0, x_ref=xr0, u=ctl.wr.reshape(-1),
                    tau=ctl.torque(d, t), contact=c0.astype(float),
                    foot_z=[mpc_srb.get_foot_positions(m, d)[i, 2] for i in range(2)],
                    solve_ms=ctl.last_info["solve_ms"],
                    violation=ctl.last_info["violation"], ncon=d.ncon)
            if t > RAMP_T1 + 1.0:
                vx_hist.append(x0[9]); py_hist.append(x0[4])
        d.ctrl[:] = ctl.torque(d, t)
        mujoco.mj_step(m, d)

        feet_z = mpc_srb.get_foot_positions(m, d)[:, 2]
        for i in range(2):
            st = ctl.gait.in_stance(t, i)
            if not st:
                swing_apex[i] = max(swing_apex[i], feet_z[i] - ctl.z_ground)
            if (not prev_st[i]) and st:
                n_swings[i] += 1
            prev_st[i] = st

        if t >= next_rep - 1e-9:
            next_rep += 1.0
            cc = "".join("S" if ctl.gait.in_stance(t, i) else "w" for i in range(2))
            yaw_now = np.degrees(x0[2])
            yaw_err = np.degrees(x0[2] - ctl.yaw_ref(t))
            print(f"  {t:5.1f}  {d.qpos[2]:8.4f}  {d.subtree_com[0][0]:7.3f}"
                  f"  {d.subtree_com[0][1]:7.3f}  {x0[9]:5.2f}  {yaw_now:7.1f} {yaw_err:7.1f}"
                  f"  {np.abs(ctl.wr[:,5]).max():5.2f}  {cc}  {ctl.last_info['solve_ms']:7.1f}")
        if fell is None and d.qpos[2] < z0 - 0.25:
            fell = t
            break

    print()
    ok = fell is None
    if fell:
        print(f"  쓰러짐 @ {fell:.2f}s")
    else:
        print(f"  {seconds:.0f}초 생존 ✓  pelvis z {z0:.4f} -> {d.qpos[2]:.4f}")
    print(f"  완료한 swing 횟수: 왼 {n_swings[0]}, 오른 {n_swings[1]}"
          f"   swing 최고높이: 왼 {swing_apex[0]*100:.1f} cm, 오른 {swing_apex[1]*100:.1f} cm")
    if abs(ctl.wz_cmd) > 1e-12:
        yaw_end = ctl._yaw_unwrap        # 언랩 (quat 직접 읽으면 ±180° 랩)
        t_end = min(seconds, fell if fell else seconds)
        wz_avg = (yaw_end - ctl.yaw0) / max(t_end - RAMP_T0 * 0.5, 1e-6)
        print(f"  yaw: {np.degrees(ctl.yaw0):.1f}° -> {np.degrees(yaw_end):.1f}°"
              f"  (참조 {np.degrees(ctl.yaw_ref(t_end)):.1f}°,"
              f" 오차 {np.degrees(yaw_end - ctl.yaw_ref(t_end)):+.1f}°)"
              f"  평균 wz ≈ {wz_avg:.3f} rad/s (명령 {ctl.wz_cmd})")
    if vx_hist:
        print(f"  정착 후 평균 vx = {np.mean(vx_hist):.3f} m/s (명령 {vx})"
              f"   CoM y 진폭 = ±{(np.max(py_hist)-np.min(py_hist))/2*100:.1f} cm")
    sms = np.stack(log.rows["solve_ms"])
    print(f"  QP: 평균 {sms.mean():.1f} ms, 최대 {sms.max():.1f} ms")
    log.save()
    return ok


def view(vx=0.0, kp_up=60.0, swing_id=False, wz=0.0, yaw_hold=True, ctor=None,
         soft_land=False, lam_swing=False, wn_swing=100.0, zeta_swing=0.5,
         cap_y=None, cycle=0.8, stance_frac=0.75, lip_exact=False, q_py=None,
         td_mode="cont", gate_ff=True, gate_sy=True, swing_h=0.05):
    import mujoco.viewer
    m, d = g1_model.load_torque()
    g1_model.set_crouch(m, d)
    ctl = (ctor or WalkController)(m, d, vx_cmd=vx, kp_up=kp_up,
                                   swing_id=swing_id, wz_cmd=wz,
                                   yaw_hold=yaw_hold, soft_land=soft_land,
                                   lam_swing=lam_swing, wn_swing=wn_swing,
                                   zeta_swing=zeta_swing, cap_y=cap_y,
                                   cycle=cycle, stance_frac=stance_frac,
                                   lip_exact=lip_exact, q_py=q_py,
                                   td_mode=td_mode, gate_ff=gate_ff,
                                   gate_sy=gate_sy, swing_h=swing_h)
    print(f"뷰어: gait MPC (vx_cmd={vx}, uppd={kp_up}, swingid={swing_id}). 창을 닫으면 종료.")
    k = 0
    t0 = 0.0
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t = t0 + k * m.opt.timestep
            mujoco.mj_forward(m, d)
            if k % DECIM == 0:
                ctl.update_mpc(d, t)
            d.ctrl[:] = ctl.torque(d, t)
            mujoco.mj_step(m, d)
            v.sync()
            k += 1


if __name__ == "__main__":
    vx = 0.0
    if "--vx" in sys.argv:
        vx = float(sys.argv[sys.argv.index("--vx") + 1])
    secs = 12.0
    if "--seconds" in sys.argv:
        secs = float(sys.argv[sys.argv.index("--seconds") + 1])
    legm = 1.0
    if "--legmass" in sys.argv:
        legm = float(sys.argv[sys.argv.index("--legmass") + 1])
    kpu = 60.0
    if "--uppd" in sys.argv:
        kpu = float(sys.argv[sys.argv.index("--uppd") + 1])
    sid = "--swingid" in sys.argv
    wzc = 0.0
    if "--wz" in sys.argv:
        wzc = float(sys.argv[sys.argv.index("--wz") + 1])
    yh = "--noyawhold" not in sys.argv
    sl = "--softland" in sys.argv
    lam = "--lamswing" in sys.argv
    wns = float(sys.argv[sys.argv.index("--wn") + 1]) if "--wn" in sys.argv else 100.0
    zts = float(sys.argv[sys.argv.index("--zeta") + 1]) if "--zeta" in sys.argv else 0.5
    cpy = float(sys.argv[sys.argv.index("--capy") + 1]) if "--capy" in sys.argv else None
    cyc = float(sys.argv[sys.argv.index("--cycle") + 1]) if "--cycle" in sys.argv else 0.8
    sfr = float(sys.argv[sys.argv.index("--sf") + 1]) if "--sf" in sys.argv else 0.75
    lipx = "--lip" in sys.argv
    qpy = float(sys.argv[sys.argv.index("--qpy") + 1]) if "--qpy" in sys.argv else None
    tdm = sys.argv[sys.argv.index("--tdmode") + 1] if "--tdmode" in sys.argv else "cont"
    _ng = "--nogate" in sys.argv
    gff = not (_ng or "--footframe" in sys.argv)   # 제약 frame 항상 켜기
    gsy = not (_ng or "--swingyaw" in sys.argv)    # 스윙 yaw 정렬 항상 켜기
    swh = float(sys.argv[sys.argv.index("--swingh") + 1]) if "--swingh" in sys.argv else 0.05
    exp_tags = [t for t, on in (("sl", sl), ("lam", lam),
                                ("capy", cpy is not None),
                                (f"sf{sfr:g}".replace(".", "p"),
                                 abs(sfr - 0.75) > 1e-9 or abs(cyc - 0.8) > 1e-9)) if on]
    exp_variant = "-".join(exp_tags)
    if "--decim" in sys.argv:
        # MPC 재풀이 주기 실험용 (500/DECIM Hz). 지평(DT_MPC, HORIZON)은 그대로 —
        # '재풀이를 얼마나 자주 하느냐'만 분리해서 보기 위함. residual RL 병렬화
        # 예산 산정에 쓰인다 (Q&A Q5).
        DECIM = int(sys.argv[sys.argv.index("--decim") + 1])
        print(f"  [실험] MPC 재풀이 {500/DECIM:.0f} Hz (DECIM={DECIM})")
    if "--view" in sys.argv:
        view(vx, kp_up=kpu, swing_id=sid, wz=wzc, yaw_hold=yh,
             soft_land=sl, lam_swing=lam, wn_swing=wns, zeta_swing=zts,
             cap_y=cpy, cycle=cyc, stance_frac=sfr, lip_exact=lipx, q_py=qpy,
             td_mode=tdm, gate_ff=gff, gate_sy=gsy, swing_h=swh)
    else:
        ok = headless(vx=vx, seconds=secs, legmass=legm,
                      kp_up=kpu, swing_id=sid, wz=wzc, yaw_hold=yh,
                      variant=exp_variant,
                      soft_land=sl, lam_swing=lam, wn_swing=wns,
                      zeta_swing=zts, cap_y=cpy, cycle=cyc, stance_frac=sfr,
                      lip_exact=lipx, q_py=qpy, td_mode=tdm,
                      gate_ff=gff, gate_sy=gsy, swing_h=swh)
        step = "STEP 5 (제자리 스텝)" if abs(vx) < 1e-9 else f"STEP 6 (전진 {vx} m/s)"
        print()
        print(step + (" 통과 ✓" if ok else " 실패"))
