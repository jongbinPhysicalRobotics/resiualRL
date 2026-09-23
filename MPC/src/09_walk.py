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

import os
import sys

# 작은 행렬(~200×200)에 BLAS 다중 스레드를 쓰면 스레드 깨우기 비용으로 MPC 풀이가 가끔
# 수십~수백 ms 튄다 (0.7 m/s 실측: p99 154 ms → 35 ms, 배속 0.33 → 0.41). numpy 를
# 부르기 전에 1 스레드로 고정 (Q&A 9/22 Q6). 이미 설정돼 있으면 그대로 둔다.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
                 lip_exact=False, q_py=None, td_mode="cont", gate_ff=True, gate_sy=True, swing_h=0.05,
                 jdot=False, side_w=None, td_scale=1.0, td_dx=0.0,
                 cop_margin=1.0, du_f=0.0, du_m=0.0, q_over=None, wz_pelvis=0.0,
                 wx_pelvis=0.0, wy_pelvis=0.0, td_sink=0.0, swing_prof=0,
                 lo_ramp=0.0, early_td=False,
                 liftoff_fix=True):
        self.m = m
        # ω_z 출처 (Q&A 9/22): 0 = 전신 각운동량 (기존), 1 = 골반 yaw 각속도. 사이는 섞기.
        # 전신 ω_z 를 0 으로 몰면 스윙 다리의 yaw 운동량을 골반이 반대로 돌아 상쇄한다
        # (골반 yaw 진동 16° vs 배포 RL 2°). yaw 축만 바꾸고 roll/pitch 는 그대로.
        self.wz_pelvis = float(wz_pelvis)
        # roll·pitch 도 같은 방식으로 섞기 (Q&A 9/22 Q2). 몸 yaw frame 에서 축별로.
        self.w_pel = np.array([wx_pelvis, wy_pelvis, wz_pelvis], float)
        self._pel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.adof = actuated_dofs(m)
        # SRB 재료 훅 — 서브클래스(11_walk_srb_upper 등)가 '무엇을 강체로 볼
        # 것인가'만 바꿔치기할 수 있게 params/state 추출을 메서드로 분리.
        self.params = self.make_srb_params(m, d)
        # CoP 여유 (Q&A 9/22): 발 사각형을 '발 중앙' 기준으로 cop_margin 배 축소.
        # 명령 CoP 가 발 가장자리에 꽂히면 반대쪽 하중이 0 이 되어 발이 뒤꿈치로 선다
        # (9/21 Q8) — 그 상태에선 비틀림 마찰이 거의 없어 발이 돌아간다 (9/22).
        self.cop_margin = cop_margin
        if cop_margin < 1.0:
            p_ = self.params
            X_, xc_ = (p_.l_t + p_.l_h) / 2.0, (p_.l_t - p_.l_h) / 2.0
            p_.l_t, p_.l_h, p_.w = xc_ + cop_margin * X_, cop_margin * X_ - xc_, cop_margin * p_.w
        # 보행 전용 가중치: 횡 속도 감쇠 강화 (서 있기 기본값은 mpc_qp.Q_DEFAULT)
        from mpc_qp import Q_DEFAULT
        # 횡 위치 가중치: 이중지지를 줄이면(긴 스윙) 횡 균형을 MPC 가 더 많이
        # 져야 한다 — reference 는 py/px = 5~14 (우리 기본 1). 24절.
        _q = None
        if q_py is not None:
            _q = Q_DEFAULT.copy(); _q[4] = q_py
        if q_over:                                    # {상태 인덱스: 가중치} (Q&A 9/22, 예: {2: yaw})
            _q = (Q_DEFAULT.copy() if _q is None else _q)
            for _i, _v in q_over.items():
                _q[int(_i)] = float(_v)
        _du = None
        if du_f > 0 or du_m > 0:                      # Δu 벌점 (Q&A 9/22)
            _du = np.tile([du_f] * 3 + [du_m] * 3, 2)
        self.mpc = WrenchMPC(self.params, horizon=HORIZON, dt=DT_MPC, q_diag=_q,
                     psi0=mpc_srb.quat_to_euler_zyx(d.qpos[3:7])[2], du_w=_du)
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
        # ★ 보폭 (Q&A 9/17 Q2·Q3, 9/18, SUMMARY §5). 위 side_offset 의 y 는 G1 'stand'
        # 키프레임(hip_roll=0 → 발이 엉덩이 바로 아래)에서 물려받은 ±11.8 cm — 걷기용으로
        # 고른 값이 아니다. 착지 보폭 27 cm 의 87 % 가 이것. 배포 RL 정책(9/17)은 같은
        # 기하에서 출발해 15.6 cm(0.5 m/s) 로 걷고, 보폭↔횡 CoP 이용률 상관이 +0.951.
        #   side_w=None  : 기하 기본값 유지 (23.7 cm)
        #   side_w=float : 고정 보폭 [m]  (예: 0.176 → 오프셋 ±8.8 cm)
        #   side_w="auto": w(v) = 0.1933 − 0.0433·v_ramped, [0.14, 0.20] 클램프 (RL 회귀 R²=0.94)
        # 실측 (9/21 Q7, 13_sidew_ab.py): **13 cm 가 최적점** — 120 s 전 항목 통과,
        #   0.6 추종 82→91 %, 0.7 추종 40→67 % (뒤로 젖힘 −12.3→−6°), y 진폭 0.3/0.5 에서 −35 %.
        #   11~12 는 roll σ 가 다시 튄다 (min_y_sep 하한 근처). auto 는 RL 계수라 우리 최적보다 넓다.
        # 기전: 횡 LIP 에서 |y−p| ∝ 보폭 → Fy RMS 40→17 N, v_y RMS −74 % →
        #   capture 항(v_y/ω₀)이 착지 목표를 쓸어내는 폭 7.9→1.8 cm. '보정을 억제'(td_mode,
        #   실패)가 아니라 '보정의 필요를 줄인' 것. ⚠ 롤 CoP 이용률은 안 줄었다 (예측 기각).
        self.side_w = side_w
        self._side_geom = [o.copy() for o in self.side_offset]
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
        # 착지 거리 (Q&A 9/21 Q10·Q11): td_scale = 전진 중립항 배율,
        # td_dx = 착지 기준점 x 이동 [m] (−0.035 = 발목 대신 발 중앙을 목표에 놓기, reference 방식)
        self.td_scale = td_scale
        self.td_dx = td_dx
        # 스윙 착지 목표를 땅보다 낮게 [m] (Q&A 9/22 Q12). 목표가 딱 지면이면 속도 0 으로 다가가다
        # 추종 지연만큼 늦게 닿는다 (예정 착지 순간 접촉 0 %, 평균 8~9 ms 지연, 그 사이 공중 발에
        # ~180 N 명령 — Q11). reference 는 약 1 cm 아래를 겨냥한다. MPC 의 미래 발 위치는 그대로.
        self.td_sink = float(td_sink)
        # 이륙 전 하중 내림 [s] (Q&A 9/22 Q13): 떠날 발의 Fz 상한을 이륙 전 lo_ramp 동안 1 → 0 으로.
        # 뒷발 이륙 순간(+60 ms)이 한 스텝에서 가장 큰 몸통 흔들림 (골반 pitch 각속도 −3.3 rad/s,
        # 합력 72 %) — 하중이 새 발로 넘어가기 전에 떨어지기 때문이라는 가설 (Q12).
        self.lo_ramp = float(lo_ramp)
        # 조기 접촉 처리 (Q&A 9/22 Q13, reference ContactManager 의 early contact): 스윙 후반(s ≥ 0.5)에
        # 발이 실제로 닿으면(발 합력 > 20 N) 예정 착지를 기다리지 않고 바로 스탠스로 — 스윙 제어기가
        # 땅속 목표로 발을 계속 누르지 않게 (착지 목표 낮추기 단독이 실패한 이유, Q12).
        self.early_td = bool(early_td)
        # ★ 이륙 순간 스윙 시작점 버그 수정 (Q&A 9/22 Q13). 스윙 시작점 p_liftoff 는 update_mpc (100 Hz)
        # 에서만 기록됐는데, torque (500 Hz) 는 스케줄이 바뀌는 틱에 바로 스윙 제어를 시작한다. 그 사이
        # 틱(들)에서 스윙 제어기가 '한 주기 전 이륙 위치' (실제 발에서 ~40 cm) 를 목표로 발을 400 N
        # (상한) 으로 당겼다 — 뒷발 이륙 +60 ms 의 골반 pitch 튐(−3.3 rad/s) 후보. 수정: torque 에서 스윙이
        # 시작되는 바로 그 틱에 시작점을 기록. False 면 예전 동작 (비교용).
        self.liftoff_fix = bool(liftoff_fix)
        self._sw_t0 = [-1.0, -1.0]
        self.early = [False, False]
        self._fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{q}_ankle_roll_link")
                    for q in ("left", "right")]
        self._f6 = np.zeros(6)
        # gate_yaw=False 면 제약 frame(발 실측 yaw)·스윙 yaw 정렬을 직진에서도
        # 켠다. 원래 wz≠0 게이트였는데, 0.5 m/s 직진에서 stance 발이 최대 20°
        # 돌아가 있는 게 측정됐다 — 제약 frame 오차 20° 면 MPC 가 '발 안'이라
        # 믿는 CoP 가 실제로는 옆으로 1.6 cm 밖이다 (반폭 2.5 cm). Q&A 9/16.
        self.gate_ff = gate_ff   # 제약 frame 을 발 실측 yaw 로 (모델 정직화, 공짜)
        self.gate_sy = gate_sy   # 스윙 발 yaw 정렬 (물리 교정, 토크 비용 있음)
        for sc in self.sw:
            sc.soft_land = soft_land
            sc.h_swing = swing_h
            # 스윙 보간 (Q&A 9/22 Q13): 0 = 3 차 smoothstep, 1 = 수평만 5 차, 2 = 수평·높이 모두 5 차
            sc.quintic_xy = int(swing_prof) >= 1
            sc.quintic_z = int(swing_prof) >= 2
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
        # 가속을 역동역학으로 미리 감당. J̇q̇ 항은 생략 (아래).
        #  ※ 옛 주석의 "M 이 밀집이라 stance 관절에도 보상이 들어간다" 는 틀렸다 —
        #    두 다리는 서로 다른 가지라 M[왼다리,오른다리] = 0. 실제로 흘러가는 곳은
        #    베이스 행이고, 그건 모터가 없어 버려진다 (9/23 Q4).
        self.swing_id = swing_id
        # J̇q̇ 보상 (9/18 Q2, 9/21 Q6, 9/23 Q4). p̈ = J q̈ + J̇q̇ 이므로 정확한
        # 역기구학은 q̈ = J⁺(a_ff − J̇q̇) 인데 원래 J̇q̇ 를 생략했다. 항은 작지 않다
        # — a_ff 의 34 % (평균 3.5, 최대 6.4 m/s². 옛 기록 69 % 는 이륙 버그가
        # 있던 때 측정이라 과대). 식 자체는 이쪽이 옳다 (Cheetah 3 식 (2), 9/23 Q3).
        # ★ 그래도 **기본 꺼짐** — 이륙 버그(9/22 Q13)를 잡고 재시험해도 기각.
        #   이유는 '횡 흔들림'(9/21 추정)이 아니라 **하중 인계**였다:
        #   M q̈_ff 의 베이스 행(42.8 N·m)이 스윙다리 행(14.0)의 3 배인데 모터가
        #   없어 버려진다 → 요구의 3/4 를 골반·접촉이 떠안는다. 켜면 ‖q̈_ff‖ +13 %,
        #   버려지는 베이스 요구 +18 % → 착지 후 합력 최저 54 → 16 % (0.7),
        #   56 → 26 % (0.6). 정작 목표였던 스윙 추종오차도 3/3 속도 악화 (+17~21 %).
        #   0.3·0.5 는 영향 거의 없음 (하중 여유가 있다).
        # → 해결은 MPC 쪽 (다리 각운동량 아핀항, 9/22 Q9). `--jdot` 으로 켤 수 있다.
        self.jdot = jdot
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
        x = mpc_srb.get_state(self.m, d, I_body=self.params.I_body)
        a = self.w_pel
        if a.any():
            w_p = d.xmat[self._pel_id].reshape(3, 3) @ d.qvel[3:6]     # 골반 ω (world)
            Rz_ = mpc_srb.rz(x[2])
            wb = Rz_.T @ x[6:9]; wpb = Rz_.T @ w_p                     # 몸 yaw frame
            x[6:9] = Rz_ @ ((1.0 - a) * wb + a * wpb)
        return x

    def ramp(self, t):
        return float(np.clip((t - RAMP_T0) / (RAMP_T1 - RAMP_T0), 0.0, 1.0))

    def v_cmd(self, t):
        """몸(yaw) 기준 명령을 world 로 회전해서 반환."""
        r = self.ramp(t)
        v_body = np.array([self.vx_cmd * r, 0.0, 0.0])
        return mpc_srb.rz(self.yaw_ref(t)) @ v_body

    def wz(self, t):
        return self.wz_cmd * self.ramp(t)

    def side_width(self, t):
        """공칭 보폭 [m] (좌우 발 간격). None 이면 기하 기본값."""
        if self.side_w is None:
            return None
        if isinstance(self.side_w, str):          # "auto" — 속도 함수
            v = abs(self.vx_cmd) * self.ramp(t)
            return float(np.clip(0.1933 - 0.0433 * v, 0.14, 0.20))
        return float(self.side_w)

    def side_offset_at(self, t):
        """발 i 의 공칭 오프셋 (몸 yaw frame, 3-벡터). y 만 보폭 정책으로 덮어쓴다."""
        w = self.side_width(t)
        if w is None:
            return self._side_geom
        out = []
        for i in range(2):
            o = self._side_geom[i].copy()
            o[1] = np.sign(self._side_geom[i][1]) * 0.5 * w
            out.append(o)
        return out

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
                if not (self.liftoff_fix and 0.0 <= t - self._sw_t0[i] < 0.02):
                    self.sw[i].start(feet[i])
                    self._sw_t0[i] = t
            if not st and not self.early[i]:
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
                off = np.array(self.side_offset_at(t)[i], float)
                off[0] += self.td_dx
                raw = raibert_target(
                    x0[3:6], x0[9:12], yaw_pl, off, vc,
                    self.gait.T_stance, z_com=x0[5], z_ground=self.z_ground,
                    anchor_xy=None if combo else [x0[3], self.com0[1]],
                    cap_y_max=self.cap_y, lip_exact=self.lip_exact,
                    ff_scale=self.td_scale)
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
            contact[0, i] = self.gait.in_stance(t, i) or self.early[i]
            if self.early[i]:                    # 조기 접촉: 예정 착지까지 남은 지평도 '닿음' (Q13)
                t_td = t + self.gait.time_to_touchdown(t, i)
                for k in range(1, N):
                    if t + (k + 0.5) * DT_MPC < t_td:
                        contact[k, i] = True
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
        fz_scale = None
        if self.lo_ramp > 0.0:                  # 이륙 전 하중 내림 (Q13)
            fz_scale = np.ones((N, 2))
            for k in range(N):
                tk = t if k == 0 else t + (k + 0.5) * DT_MPC     # contact 표와 같은 시각
                for i in range(2):
                    if contact[k, i] and self.gait.in_stance(tk, i):
                        ttl = (self.gait.stance_frac - self.gait.phase(tk, i)) * self.gait.T
                        fz_scale[k, i] = float(np.clip(ttl / self.lo_ramp, 0.0, 1.0))
        u0, _, info = self.mpc.solve_gait(x0, X_ref, psi_lin,
                                          foot_traj, contact, u_ref,
                                          psi_feet=psi_feet, fz_scale=fz_scale)
        self.wr = u0.reshape(2, 6)
        self.last_info = info
        return x0, X_ref[0], contact[0]

    def _foot_fz(self, d, i):
        """발 i 의 실측 수직 접촉력 합 [N] (조기 접촉 판정용)."""
        fz = 0.0
        for c in range(d.ncon):
            con = d.contact[c]
            if self.m.geom_bodyid[con.geom1] == self._fb[i] or self.m.geom_bodyid[con.geom2] == self._fb[i]:
                mujoco.mj_contactForce(self.m, d, c, self._f6)
                fz += abs(self._f6[0])
        return fz

    # -------------------------------------------------- 500 Hz: 토크
    def torque(self, d, t):
        m = self.m
        tau_c = np.zeros(m.nv)
        for i, s_name in enumerate(mpc_srb.FOOT_SITES):
            st = self.gait.in_stance(t, i)
            if st:
                self.early[i] = False            # 예정 스탠스가 시작되면 조기 접촉 표시 해제
            elif self.early_td and not self.early[i] and self.gait.swing_phase(t, i) >= 0.5:
                if self._foot_fz(d, i) > 20.0:   # 스윙 후반에 실제로 닿았다 → 바로 스탠스 (Q13)
                    self.early[i] = True
                    sid_e = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s_name)
                    p_e = d.site_xpos[sid_e].copy(); p_e[2] = self.z_ground
                    self.p_land[i] = p_e         # 착지점을 닿은 곳으로 고정
            st = st or self.early[i]
            if (self.liftoff_fix and not st and self._st_prev[i]
                    and not (0.0 <= t - self._sw_t0[i] < 0.02)):
                sid_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s_name)
                self.sw[i].start(d.site_xpos[sid_l].copy())   # 스윙 시작 틱에 시작점 기록 (Q13)
                self._sw_t0[i] = t
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
                p_tgt = self.p_land[i]
                if self.td_sink:
                    p_tgt = p_tgt - np.array([0.0, 0.0, self.td_sink])
                F, mom, jacp, jacr = self.sw[i].wrench(
                    m, d, s_name, s, p_tgt, self.gait.T_swing,
                    yaw_des=yd, kp_axis=kp_axis, kd_axis=kd_axis)
                tau_c += jacp.T @ F + jacr.T @ mom
                if self.swing_id:
                    a_ff = self.sw[i].target_acc(s, p_tgt, self.gait.T_swing)
                    dofs = self.leg_dofs[i]
                    J6 = np.vstack([jacp, jacr])[:, dofs]
                    rhs = np.concatenate([a_ff, np.zeros(3)])
                    if self.jdot:
                        # p̈ = J q̈ + J̇ q̇  →  J q̈ = a_ff − J̇ q̇
                        sid_j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s_name)
                        jdp = np.zeros((3, m.nv)); jdr = np.zeros((3, m.nv))
                        mujoco.mj_jacDot(m, d, jdp, jdr, d.site_xpos[sid_j],
                                         m.site_bodyid[sid_j])
                        rhs -= np.vstack([jdp, jdr])[:, dofs] @ d.qvel[dofs]
                    qdd, *_ = np.linalg.lstsq(J6, rhs, rcond=None)
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
             gate_ff=True, gate_sy=True, swing_h=0.05, jdot=False, side_w=None,
             td_scale=1.0, td_dx=0.0, cop_margin=1.0, du_f=0.0, du_m=0.0, wz_pelvis=0.0,
             wx_pelvis=0.0, wy_pelvis=0.0, td_sink=0.0, swing_prof=0,
             lo_ramp=0.0, early_td=False,
             liftoff_fix=True):
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
                                   gate_sy=gate_sy, swing_h=swing_h,
                                   jdot=jdot, side_w=side_w,
                                   td_scale=td_scale, td_dx=td_dx,
                                   cop_margin=cop_margin, du_f=du_f, du_m=du_m,
                                   wz_pelvis=wz_pelvis, wx_pelvis=wx_pelvis,
                                   wy_pelvis=wy_pelvis, td_sink=td_sink,
                                   swing_prof=swing_prof, lo_ramp=lo_ramp,
                                   early_td=early_td, liftoff_fix=liftoff_fix)
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

    _w = ctl.side_width(RAMP_T1 + 1.0)
    print(f"  공칭 보폭: {'기하 기본 ' + f'{200*abs(ctl._side_geom[0][1]):.1f}' if _w is None else f'{100*_w:.1f}'} cm"
          f"{'  (auto: w(v))' if isinstance(side_w, str) else ''}")
    print(f"  t[s]  pelvis_z   com_x    com_y   vx     yaw[°] yaw_e[°]  |mz|  접촉  solve_ms")
    next_rep = 0.0
    x0 = None
    for k in range(int(seconds / dt)):
        t = k * dt
        mujoco.mj_forward(m, d)
        if k % DECIM == 0:
            x0, xr0, c0 = ctl.update_mpc(d, t)
            # 스윙 참조궤적(발 site 기준)·실제 발 위치·착지점 — 추종 분석용 (Q&A 9/21 Q3)
            feet_now = mpc_srb.get_foot_positions(m, d)
            sw_s = np.zeros(2)
            f_ref = np.full((2, 3), np.nan)        # stance 중엔 NaN (플롯에서 끊김)
            for i in range(2):
                if not ctl.gait.in_stance(t, i):
                    sw_s[i] = ctl.gait.swing_phase(t, i)
                    f_ref[i] = ctl.sw[i].target(sw_s[i], ctl.p_land[i],
                                                ctl.gait.T_swing)[0]
            log.add(t=t, x=x0, x_ref=xr0, u=ctl.wr.reshape(-1),
                    tau=ctl.torque(d, t), contact=c0.astype(float),
                    foot_z=feet_now[:, 2],
                    swing_s=sw_s, foot_ref=f_ref.reshape(-1),
                    foot_pos=feet_now.reshape(-1),
                    p_land=np.asarray(ctl.p_land).reshape(-1),
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
    sms = np.stack(log.rows["solve_ms"]); tt = np.stack(log.rows["t"])
    med = float(np.median(sms)); thr = max(3.0 * med, 20.0)
    print(f"  QP: 평균 {sms.mean():.1f} ms, 중앙값 {med:.1f}, p99 {np.percentile(sms, 99):.1f},"
          f" 최대 {sms.max():.1f} ms")
    spike = np.flatnonzero(sms > thr)
    if len(spike):
        head = ", ".join(f"{tt[j]:.2f}s({sms[j]:.0f})" for j in spike[:8])
        print(f"      튀는 지점 {len(spike)}회 (>{thr:.0f} ms): {head}"
              f"{' ...' if len(spike) > 8 else ''}")
    log.save()
    return ok


def _draw_swing(v, ctl, t, n_seg=24):
    """뷰어에 스윙 참조궤적(선) + 현재 목표(작은 구) + 착지점(빨간 구) 을 그린다.
    기준점은 발 site(발목 원점) — 스윙 제어기가 실제로 추종하는 점 그대로
    (발바닥 중심은 발이 수평일 때 여기서 앞 3.5 cm·아래 3.5 cm 상수 오프셋. Q&A 9/21 Q4).
    왼발 초록, 오른발 파랑."""
    scn = v.user_scn
    scn.ngeom = 0

    def sphere(p, r, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0.0, 0.0]),
                            np.asarray(p, float), np.eye(3).flatten(),
                            np.asarray(rgba, np.float32))
        scn.ngeom += 1

    def line(a, b, rgba, w=3.0):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, w,
                             np.asarray(a, float), np.asarray(b, float))
        scn.ngeom += 1

    T_sw = ctl.gait.T_swing
    for i in range(2):
        if ctl.gait.in_stance(t, i):
            continue
        col = (0.2, 0.9, 0.3, 0.9) if i == 0 else (0.3, 0.5, 1.0, 0.9)
        pts = [ctl.sw[i].target(s, ctl.p_land[i], T_sw)[0]
               for s in np.linspace(0.0, 1.0, n_seg + 1)]
        for a, b in zip(pts[:-1], pts[1:]):
            line(a, b, col)
        s_now = ctl.gait.swing_phase(t, i)
        sphere(ctl.sw[i].target(s_now, ctl.p_land[i], T_sw)[0], 0.012, col)
        sphere(ctl.p_land[i], 0.015, (1.0, 0.3, 0.2, 0.9))

from viewer_hud import ViewerHUD      # 뷰어 글자 표시 (Q&A 9/22 Q5)


def view(vx=0.0, kp_up=60.0, swing_id=False, wz=0.0, yaw_hold=True, ctor=None,
         soft_land=False, lam_swing=False, wn_swing=100.0, zeta_swing=0.5,
         cap_y=None, cycle=0.8, stance_frac=0.75, lip_exact=False, q_py=None,
         td_mode="cont", gate_ff=True, gate_sy=True, swing_h=0.05, jdot=False, side_w=None, follow=True, realtime=True, max_sim=None, timelog=None,
         sync_every=17, draw_at_sync=True, boost=True, lite=False,
         td_scale=1.0, td_dx=0.0, cop_margin=1.0, du_f=0.0, du_m=0.0, wz_pelvis=0.0,
         wx_pelvis=0.0, wy_pelvis=0.0, td_sink=0.0, swing_prof=0,
         lo_ramp=0.0, early_td=False,
         liftoff_fix=True):
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
                                   gate_sy=gate_sy, swing_h=swing_h,
                                   jdot=jdot, side_w=side_w,
                                   td_scale=td_scale, td_dx=td_dx,
                                   cop_margin=cop_margin, du_f=du_f, du_m=du_m,
                                   wz_pelvis=wz_pelvis, wx_pelvis=wx_pelvis,
                                   wy_pelvis=wy_pelvis, td_sink=td_sink,
                                   swing_prof=swing_prof, lo_ramp=lo_ramp,
                                   early_td=early_td, liftoff_fix=liftoff_fix)
    print(f"뷰어: gait MPC (vx_cmd={vx}, uppd={kp_up}, swingid={swing_id}). 창을 닫으면 종료.")
    k = 0
    t0 = 0.0
    import time as _time
    # 뷰어 기본값 (Q&A 9/22 Q7): 화면 갱신 ~30 Hz, 궤적 그리기는 갱신 때만, Windows 부스트.
    # 부스트 없이는 뷰어가 떠 있는 동안 Windows 가 계산 스레드를 느리게 돌려 (같은 고정 계산이
    # 3 → 8~9 ms) sim 시간이 실제 시간에 계속 뒤처졌다 (배속 0.4~0.55). 부스트 켜면 1.0.
    SYNC_EVERY = sync_every                     # 500 Hz / 17 ≈ 29 Hz 화면 갱신
    if lite:
        # 렌더링 가볍게 (Q&A 9/22 Q7): 바닥 반사·그림자가 GPU 부하의 큰 몫이고, 노트북은 CPU 와
        # 내장 GPU 가 전력 한도를 나눠 써서 렌더링이 무거우면 MPC 계산 클럭이 떨어진다.
        m.light_castshadow[:] = 0
        m.mat_reflectance[:] = 0.0
    if boost:
        from viewer_hud import boost_process
        print(f"  프로세스 부스트 (우선순위 높음 + 스로틀링 끔): {boost_process()}")
    with mujoco.viewer.launch_passive(m, d) as v:
        hud = ViewerHUD(m, vx)
        wall0 = _time.perf_counter()
        if follow:
            # 트래킹 카메라: 골반을 따라가되 마우스 회전·줌은 그대로 된다 (--nofollow 로 끔)
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            v.cam.distance, v.cam.elevation, v.cam.azimuth = 3.0, -15.0, 120.0
        # 뷰어 시간 기록 (Q&A 9/22 Q7): sim 1 s 마다 실제 걸린 시간과 구간별 몫
        pc = _time.perf_counter
        spw = int(round(1.0 / m.opt.timestep))
        acc = dict(mpc=0.0, draw=0.0, torque=0.0, hud=0.0, step=0.0, sync=0.0, sleep=0.0)
        tl, w_prev = [], pc()
        while v.is_running():
            t = t0 + k * m.opt.timestep
            a_ = pc(); mujoco.mj_forward(m, d); acc["step"] += pc() - a_
            if k % DECIM == 0:
                a_ = pc(); ctl.update_mpc(d, t); acc["mpc"] += pc() - a_
                if not draw_at_sync:
                    a_ = pc()
                    with v.lock():
                        _draw_swing(v, ctl, t)  # 스윙 참조궤적 오버레이 (Q&A 9/21)
                    acc["draw"] += pc() - a_
            a_ = pc(); d.ctrl[:] = ctl.torque(d, t); acc["torque"] += pc() - a_
            a_ = pc(); hud.update(v, d, t, k); acc["hud"] += pc() - a_   # sim/실제 시간·속도 (Q5)
            a_ = pc(); mujoco.mj_step(m, d); acc["step"] += pc() - a_
            k += 1
            # 화면 갱신은 ~60 Hz 면 충분하다 — 500 Hz 스텝마다 sync 하면 그리기가 계산을
            # 잡아먹는다. 계산이 실시간보다 빠르면 기다려서 sim 시간 = 실제 시간 (Q&A 9/22 Q6).
            if k % SYNC_EVERY == 0:
                if draw_at_sync:                # 궤적은 화면 갱신 때만 그려도 충분 (잠금 횟수↓)
                    a_ = pc()
                    with v.lock():
                        _draw_swing(v, ctl, t)
                    acc["draw"] += pc() - a_
                a_ = pc(); v.sync(); acc["sync"] += pc() - a_
                if realtime:
                    ahead = k * m.opt.timestep - (pc() - wall0)
                    if ahead > 0:
                        a_ = pc(); _time.sleep(ahead); acc["sleep"] += pc() - a_
            if k % spw == 0:
                now = pc()
                row = [k * m.opt.timestep, now - w_prev, now - wall0] + [1000 * x for x in acc.values()]
                if timelog:
                    from viewer_hud import cpu_bench
                    row.append(cpu_bench())     # CPU 속도 지표 (창 시간에서 뺌)
                    wall0 += pc() - now
                tl.append(row)
                for key in acc:
                    acc[key] = 0.0
                w_prev = pc()
            if max_sim is not None and k * m.opt.timestep >= max_sim:
                break
    if timelog:
        np.savez(timelog, tl=np.array(tl), cols=np.array(["t", "wall", "wall_total"] + list(acc.keys()) + ["bench"]),
                 vx=vx, realtime=realtime)
        print(f"  뷰어 시간 기록 저장: {timelog}")


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
    jd = "--jdot" in sys.argv           # J̇q̇ 보상 (기본 꺼짐 — 아래 주석)
    fol = "--nofollow" not in sys.argv   # 뷰어 카메라가 로봇을 따라감 (기본 켜짐)
    sw_ = None
    if "--sidew" in sys.argv:            # 보폭 [cm] 또는 auto (Q&A 9/21)
        _a = sys.argv[sys.argv.index("--sidew") + 1]
        sw_ = "auto" if _a == "auto" else float(_a) / 100.0
    tds = float(sys.argv[sys.argv.index("--tdscale") + 1]) if "--tdscale" in sys.argv else 1.0
    tdx = float(sys.argv[sys.argv.index("--tddx") + 1]) / 100.0 if "--tddx" in sys.argv else 0.0   # [cm]
    cpm = float(sys.argv[sys.argv.index("--copm") + 1]) if "--copm" in sys.argv else 1.0    # CoP 여유
    duf = float(sys.argv[sys.argv.index("--duf") + 1]) if "--duf" in sys.argv else 0.0     # Δu 벌점 (힘)
    dum = float(sys.argv[sys.argv.index("--dum") + 1]) if "--dum" in sys.argv else 0.0     # Δu 벌점 (모멘트)
    wzp = float(sys.argv[sys.argv.index("--wzpel") + 1]) if "--wzpel" in sys.argv else 0.0   # ω_z 골반 비율
    wxp = float(sys.argv[sys.argv.index("--wxpel") + 1]) if "--wxpel" in sys.argv else 0.0   # ω_x (roll)
    wyp = float(sys.argv[sys.argv.index("--wypel") + 1]) if "--wypel" in sys.argv else 0.0   # ω_y (pitch)
    tsk = float(sys.argv[sys.argv.index("--tdsink") + 1]) / 1000.0 if "--tdsink" in sys.argv else 0.0  # [mm]
    spf = 0                              # 스윙 보간: --quintic xy (수평만 5 차) / --quintic all (높이도)
    lrp = float(sys.argv[sys.argv.index("--loramp") + 1]) / 1000.0 if "--loramp" in sys.argv else 0.0  # [ms]
    etd = "--earlytd" in sys.argv       # 조기 접촉 처리 (Q13)
    lfx = "--oldliftoff" not in sys.argv  # 이륙 시작점 버그 수정 (기본 켬, --oldliftoff 면 예전 동작)
    if "--quintic" in sys.argv:
        spf = 2 if sys.argv[sys.argv.index("--quintic") + 1] == "all" else 1
    rt = "--fast" not in sys.argv       # 뷰어: 실시간 맞춤 (기본). --fast 면 계산되는 만큼 빨리
    vsec = float(sys.argv[sys.argv.index("--vsec") + 1]) if "--vsec" in sys.argv else None   # 뷰어 자동 종료 [sim s]
    vlog = sys.argv[sys.argv.index("--timelog") + 1] if "--timelog" in sys.argv else None     # 뷰어 시간 기록 파일
    vsyn = int(sys.argv[sys.argv.index("--syncevery") + 1]) if "--syncevery" in sys.argv else 17  # 화면 갱신 = 500/N Hz
    dsyn = "--drawmpc" not in sys.argv  # 스윙 궤적 오버레이: 기본은 화면 갱신 때만 (--drawmpc 면 MPC 마다)
    bst = "--noboost" not in sys.argv   # Windows 우선순위 높음 + 전원 스로틀링 끔 (기본 켬)
    lit = "--lite" in sys.argv          # 렌더링 가볍게 (그림자·바닥 반사 끔)
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
             td_mode=tdm, gate_ff=gff, gate_sy=gsy, swing_h=swh, jdot=jd, side_w=sw_, follow=fol, realtime=rt, max_sim=vsec, timelog=vlog,
             sync_every=vsyn, draw_at_sync=dsyn, boost=bst, lite=lit,
             td_scale=tds, td_dx=tdx, cop_margin=cpm, du_f=duf, du_m=dum, wz_pelvis=wzp,
             wx_pelvis=wxp, wy_pelvis=wyp, td_sink=tsk, swing_prof=spf, lo_ramp=lrp, early_td=etd, liftoff_fix=lfx)
    else:
        ok = headless(vx=vx, seconds=secs, legmass=legm,
                      kp_up=kpu, swing_id=sid, wz=wzc, yaw_hold=yh,
                      variant=exp_variant,
                      soft_land=sl, lam_swing=lam, wn_swing=wns,
                      zeta_swing=zts, cap_y=cpy, cycle=cyc, stance_frac=sfr,
                      lip_exact=lipx, q_py=qpy, td_mode=tdm,
                      gate_ff=gff, gate_sy=gsy, swing_h=swh, jdot=jd, side_w=sw_,
                      td_scale=tds, td_dx=tdx, cop_margin=cpm, du_f=duf, du_m=dum,
                      wz_pelvis=wzp, wx_pelvis=wxp, wy_pelvis=wyp, td_sink=tsk,
                      swing_prof=spf, lo_ramp=lrp, early_td=etd, liftoff_fix=lfx)
        step = "STEP 5 (제자리 스텝)" if abs(vx) < 1e-9 else f"STEP 6 (전진 {vx} m/s)"
        print()
        print(step + (" 통과 ✓" if ok else " 실패"))
