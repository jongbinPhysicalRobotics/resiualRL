"""다리/팔 토크 법칙 — reference C++ 그대로 (07_port_design.md §3.8).

    stance_torque            OperationalSpaceDynamics.cpp:87-94 (τ = Jvᵀ F + Jwᵀ M, bias 없음)        spec 04 §10.1
    stance_yaw_hold_moment_z My_Controller.cpp:86-117 (computeStanceYawHoldMomentWorld, z 성분)        spec 04 §10.1
    walking_stance_torque    My_Controller.cpp:1315-1350 (F = −αu_F, M = −αu_M, M.z += α·hold)        spec 05 §9.1
    apparent_inertia         OperationalSpaceDynamics.cpp:34-46 (Λ = (Jv M⁻¹ Jvᵀ + 1e-9 I)⁻¹)
    swing_kp                 OperationalSpaceDynamics.cpp:49-58 (Kp = diag(ω_n² ⊙ diag(Λ)))
    swing_attitude_torque    SwingAttitudeControl.h:9-84 (roll 0/0 → 생략, pitch 300/18, yaw 305/18)
    swing_torque             OperationalSpaceDynamics.cpp:61-84 + LegController.cpp:262-291
                             τ = Jvᵀ(Kp e + Kd ė) + Jvᵀ Λ (a_des − J̇v q̇) + bias + τ_attitude
    standing_torque          My_Controller.cpp:1254-1295 (6×10 결합 Jacobian, 부호 반전, ramp/hold 없음) spec 05 §9.2
    arm_pd                   ArmController.cpp:204-221 (kp 100, kd 5, qd_des 0, 중력보상 없음)          spec 04 §11

다리 동역학 dyn (mit_model.LegDyn): Jv, Jw (3,5) 발 SITE, world 축, 다리 관절 열만 (고정 base 의미 — Jw·q̇_leg
는 torso 에 대한 발 상대 각속도); JvDot_qd (3,); M (5,5); bias (5,) (중력 + 코리올리, base 속도 0).
u = MPC 해 [F_L, F_R, M_L, M_R] = 지면이 몸에 주는 wrench → 다리는 음수를 민다.
"""
from __future__ import annotations

import numpy as np

LEFT, RIGHT = 0, 1
_EZ = np.array([0.0, 0.0, 1.0])


def _cfg(cfg, path, default=None):
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            try:
                cur = getattr(cur, key)
            except AttributeError:
                return default
    return default if cur is None else cur


# ---------------------------------------------------------------------------
# Stance
# ---------------------------------------------------------------------------
def stance_torque(dyn, F_W, M_W) -> np.ndarray:
    """computeStanceLegJointTorque (OperationalSpaceDynamics.cpp:87-94): τ = Jvᵀ F + Jwᵀ M.
    F_W, M_W = 발이 지면에 가하는 힘/모멘트 (= −α·u, 호출자가 부호/ramp 적용). bias/PD 없음 (LegController.cpp:302-315 주석 처리)."""
    return np.asarray(dyn.Jv).T @ np.asarray(F_W, dtype=float) + np.asarray(dyn.Jw).T @ np.asarray(M_W, dtype=float)


def stance_yaw_hold_moment_z(state, leg: int, dyn, touchdown_yaw: float, cfg, gains=None) -> float:
    """computeStanceYawHoldMomentWorld (My_Controller.cpp:86-117) 의 z 성분 (x,y 는 항상 0).

    x̂_proj = 수평 투영·정규화한 발 SITE x축, d̂ = [cos ψ_td, sin ψ_td, 0]
    err = atan2(ẑ·(x̂_proj × d̂), x̂_proj·d̂),  rate = ẑ·(Jw q̇_leg)  (torso 제외 상대 각속도)
    m_z = kp·err − kd·rate   (kp = stance_yaw_kp 20, kd = stance_yaw_kd 4)
    0 을 돌려주는 경우: kp ≤ 0 이고 kd ≤ 0, ψ_td 비유한, x̂_proj 비유한 또는 ‖x̂_proj‖ ≤ 1e-9.
    """
    kp = float(_cfg(cfg, "swing.stance_yaw_kp", 0.0))
    kd = float(_cfg(cfg, "swing.stance_yaw_kd", 0.0))
    if gains is not None:                        # 변형 shw (Q&A 9/30 Q10): 호출자가 회전 명령에 따라 조정한 이득
        kp, kd = float(gains[0]), float(gains[1])
    if not (kp > 0.0 or kd > 0.0) or not np.isfinite(touchdown_yaw):
        return 0.0
    Jw = np.asarray(dyn.Jw)
    qd = np.asarray(state.qd_leg[leg], dtype=float)
    if Jw.shape != (3, qd.size) or qd.size == 0:
        return 0.0
    foot_x = np.asarray(state.foot_R_W[leg], dtype=float)[:, 0]
    proj = foot_x - foot_x.dot(_EZ) * _EZ
    nrm = np.linalg.norm(proj)
    if not np.all(np.isfinite(proj)) or nrm <= 1e-9:
        return 0.0
    proj = proj / nrm
    d = np.array([np.cos(touchdown_yaw), np.sin(touchdown_yaw), 0.0])
    err = np.arctan2(_EZ.dot(np.cross(proj, d)), proj.dot(d))
    rate = _EZ.dot(Jw @ qd)
    if int(_cfg(cfg, "swing.stance_yaw_rate_absolute", 0) or 0):   # 변형 shabs (Q&A 9/30 Q13): 발의 절대 yaw 각속도로 감쇠.
        rate += float(np.asarray(state.torso_angvel_W, dtype=float)[2])   # reference 는 몸통 기준 상대 속도 → 제자리 회전 중엔
    return float(kp * err - kd * rate)                                    # 가만히 있는 발도 −ψ̇ 로 보여 발을 몸통 따라 돌리려 한다


def project_line_wrench(F_g, M_g, yaw: float, C, l: float, mode: int = 1):
    """변형 lproj (Q&A 9/30 Q11): 지면 → 발 wrench (월드) 를 선 접촉 가능 집합에 투영.
    발 yaw 좌표에서 line 제약 C·w ≤ 0 (행 0~3 마찰, 5 Fz ≥ 0, 6~11 CoP · 두 점 y 마찰 — 행 4 Fz 상한은 뺌) 과 Mx = 0 을
    만족하는 가장 가까운 w. 거리 가중치 힘 1 · 모멘트 1/l² (두 점 힘으로 쟀을 때 같은 크기 — 유도). 이미 만족하면 그대로.
    반환 (F_g', M_g', 투영 거리 [N 환산]).
    mode 2 (lproj=2): 투영 대신 차례로 자르기 — Fz ≥ 0, |Fx|·|Fy| ≤ µFz, |My| ≤ l·Fz, Mx = 0, 마지막으로 Mz 를
    두 점 y 마찰 (행 8~11) 이 허용하는 구간으로. 힘은 마찰을 넘을 때만 바뀌고 넘치는 비틀림은 Mz 에서만 깎는다."""
    c, s = np.cos(yaw), np.sin(yaw)
    Rt = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    w = np.concatenate([Rt @ np.asarray(F_g, dtype=float), Rt @ np.asarray(M_g, dtype=float)])
    if mode == 2:
        mu = -float(np.asarray(C)[0, 2])                     # 행 0: Fx − µFz ≤ 0
        x = w.copy()
        x[2] = max(x[2], 0.0)
        x[0] = np.clip(x[0], -mu * x[2], mu * x[2])
        x[1] = np.clip(x[1], -mu * x[2], mu * x[2])
        if float(np.asarray(C)[0, 1]) != 0.0:                # 변형 fcone=1 (Q&A 10/6 Q97): 마름모 |Fx| + |Fy| ≤ µFz 로 같은 비율로 줄임
            s1 = abs(x[0]) + abs(x[1])
            if s1 > mu * x[2]:
                x[0:2] *= mu * x[2] / s1
        x[3] = 0.0
        # CoP 행 6 (my ≤ c₆·l·Fz) · 7 (−my ≤ c₇·l·Fz) 의 계수 (cmar 가 있으면 (1 − 2ε)·l). 대칭 템플릿이면 전과 같음,
        # 변형 toeoff (Q&A 10/6 Q30) 의 template_toe 는 행 7 만 풀려 있다.
        x[4] = np.clip(x[4], float(np.asarray(C)[7, 2]) * x[2], -float(np.asarray(C)[6, 2]) * x[2])
        lo = max(-mu * (l * x[2] - x[4]) - l * x[1], l * x[1] - mu * (l * x[2] + x[4]))
        hi = min(mu * (l * x[2] - x[4]) - l * x[1], l * x[1] + mu * (l * x[2] + x[4]))
        x[5] = np.clip(x[5], lo, hi)
        d = np.array([1.0, 1.0, 1.0, 1.0 / l ** 2, 1.0 / l ** 2, 1.0 / l ** 2])
        return Rt.T @ x[:3], Rt.T @ x[3:], float(np.sqrt(np.sum(d * (x - w) ** 2)))
    rows = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11]
    Cr = np.asarray(C, dtype=float)[rows]
    if np.all(Cr @ w <= 1e-9 * max(1.0, abs(w[2]))) and abs(w[3]) <= 1e-9:
        return np.asarray(F_g, dtype=float), np.asarray(M_g, dtype=float), 0.0
    import quadprog
    d = np.array([1.0, 1.0, 1.0, 1.0 / l ** 2, 1.0 / l ** 2, 1.0 / l ** 2])
    G = np.diag(d)
    Aq = np.vstack([np.eye(6)[3:4], -Cr]).T               # quadprog: Aqᵀ x ≥ b, 첫 열 등식 (Mx = 0)
    b = np.zeros(1 + len(rows))
    x = quadprog.solve_qp(G, d * w, Aq, b, meq=1)[0]
    dist = float(np.sqrt(np.sum(d * (x - w) ** 2)))
    return Rt.T @ x[:3], Rt.T @ x[3:], dist


def stance_foot_damping(dyn, state, leg: int, kd: float, mode: int = 1) -> np.ndarray:
    """변형 sjd (Q&A 9/30 Q13): 딛는 발이 제자리에 있도록 하는 관절 감쇠 — 논문 식 (21) 의 K_d(q̇_MPC − q̇) 항을
    SRB 틀에서 옮긴 것. 논문 MPC 의 접촉 조건 J·q̇·φ = 0 (디딤 발은 움직이지 않는다) 을 만족하는 관절 속도를 목표로 삼는다:
      q̇_des = q̇ − J5⁻¹·V5,  τ = kd·(q̇_des − q̇) = −kd·J5⁻¹·V5
    V5 = [발 site 절대 선속도 (3), 발 y · z 축 둘레 절대 각속도 (2)]. 발 x 축 (roll) 은 뺀다 — 선 접촉은 그 축으로
    구를 수 있고 (Mx = 0), 막으려면 접촉이 낼 수 없는 roll 모멘트가 필요하다. 발이 가만히 있으면 0 → MPC wrench 를 건드리지 않는다.
    mode 2 (sjdm=2): 미끄럼 성분만 — 수평 선속도 2 + yaw 각속도 1 에만 감쇠 (수직 · pitch 는 목표 = 현재, 즉 감쇠 0).
    걷기의 발끝 차기 (발 pitch 회전) 와 착지 내려앉기를 막지 않으려는 것."""
    R = np.asarray(state.foot_R_W[leg], dtype=float)
    ey, ez = R[:, 1], R[:, 2]
    Jv, Jw = np.asarray(dyn.Jv), np.asarray(dyn.Jw)
    qd = np.asarray(state.qd_leg[leg], dtype=float)
    w_f = np.asarray(state.torso_angvel_W, dtype=float) + Jw @ qd
    J5 = np.vstack([Jv, ey @ Jw, ez @ Jw])
    V5 = np.concatenate([np.asarray(state.foot_vel_W[leg], dtype=float), [ey.dot(w_f), ez.dot(w_f)]])
    if mode >= 2:
        V5 = V5.copy(); V5[2] = 0.0; V5[3] = 0.0                     # 수직 · pitch 는 그대로 둔다
    if mode == 5:                                                   # 수평 미끄럼만 (yaw 는 yaw hold 에 맡긴다), 감쇠 최소제곱
        V5[4] = 0.0
        dq = J5.T @ np.linalg.solve(J5 @ J5.T + 0.05 ** 2 * np.eye(5), V5)
        return -kd * dq
    if mode == 3:                                                   # 감쇠 최소제곱 (λ = 0.05) — 무릎이 펴져 J5 가 특이점 근처일 때 과대 토크 방지
        dq = J5.T @ np.linalg.solve(J5 @ J5.T + 0.05 ** 2 * np.eye(5), V5)
        return -kd * dq
    if mode == 4:                                                   # Jᵀ 방식: 발에 감쇠 wrench −kd·V (N·s/m, yaw 는 N·m·s/rad) 를 관절로
        return -kd * (J5.T @ V5)
    dq = np.linalg.lstsq(J5, V5, rcond=1e-3)[0]
    return -kd * dq


def walking_stance_torque(dyn, state, leg: int, u0, alpha: float, touchdown_yaw: float, cfg,
                          yaw_hold_gains=None, line_proj=None) -> np.ndarray:
    """writeLegCommands stance 가지 (My_Controller.cpp:1315-1350) + stance_torque.
    F = −α u0[F_leg], M = −α u0[M_leg]; enable_stance_foot_yaw_hold 이고 α > 0 이면 M.z += α·m_z.
    line_proj = (C, l, 발 yaw, mode) 이면 (변형 lproj) yaw hold 를 더한 최종 wrench 를 선 접촉 가능 집합에 투영한 뒤 토크로."""
    u0 = np.asarray(u0, dtype=float)
    if leg == LEFT:
        F = -alpha * u0[0:3]
        M = -alpha * u0[6:9]
    else:
        F = -alpha * u0[3:6]
        M = -alpha * u0[9:12]
    if line_proj is not None and alpha > 0.0 and int(line_proj[3]) == 3:
        # lproj=3 (Q&A 9/30 Q13): MPC 해 u0 만 선 접촉 집합으로 자른다 (ADMM 고정 반복의 잔여 위반 보정 — OSQP polishing 과 같은 역할).
        Fg, Mg, _ = project_line_wrench(-F, -M, line_proj[2], line_proj[0], line_proj[1], 2)
        F, M = -Fg, -Mg
    two_pt_only = int(_cfg(cfg, "swing.stance_yaw_hold_mode", 1) or 1) == 2      # 변형 yawhold=2 (Q&A 9/29 Q5)
    two_pt = int(np.asarray(getattr(state, "foot_contact_count", np.full(2, 2)))[leg]) >= 2
    if bool(_cfg(cfg, "swing.enable_stance_foot_yaw_hold", False)) and alpha > 0.0 and (two_pt or not two_pt_only):
        M = M.copy()
        M[2] += alpha * stance_yaw_hold_moment_z(state, leg, dyn, touchdown_yaw, cfg, yaw_hold_gains)
    kd_p = float(_cfg(cfg, "swing.stance_foot_pitch_damping", 0.0) or 0.0)   # 변형 fpd (Q&A 9/30 Q12, 기본 0)
    if kd_p > 0.0 and alpha > 0.0:
        # 딛는 발이 발끝 ↔ 뒤꿈치로 흔들리는 것 (발 y 축 둘레 절대 각속도) 을 감쇠. yaw hold 처럼 MPC 밖에서 모멘트를 더한다.
        # M 은 발이 지면을 미는 모멘트 (τ = Jᵀ[F; M] — 가상일). 관절이 발에 −kd·ω_y 를 주면 발이 지면을 미는 모멘트도 −kd·ω_y.
        y_f = np.asarray(state.foot_R_W[leg], dtype=float)[:, 1]
        w_f = np.asarray(state.torso_angvel_W, dtype=float) + np.asarray(dyn.Jw) @ np.asarray(state.qd_leg[leg], dtype=float)
        M = M - alpha * kd_p * float(y_f.dot(w_f)) * y_f
    if line_proj is not None and alpha > 0.0 and int(line_proj[3]) in (1, 2):
        Fg, Mg, _ = project_line_wrench(-F, -M, line_proj[2], line_proj[0], line_proj[1], int(line_proj[3]))
        F, M = -Fg, -Mg
    hc = float(_cfg(cfg, "swing.stance_yaw_hold_torsion_clip", 0.0) or 0.0)
    if hc > 0.0 and line_proj is not None and alpha > 0.0:
        # 변형 holdclip (Q&A 10/6 Q50): yaw hold 를 더한 최종 비틀림을 선 접촉 두 점 옆 마찰 (행 8 ~ 11) 이 허용하는 구간으로.
        # 힘 · CoP 는 그대로 (lproj=3 에서 이미 잘림) — Mz 만. 구간 [lo, hi] 는 project_line_wrench mode 2 와 같은 식,
        # hc = 1 이면 물리 한계 그대로, hc > 1 이면 구간 가운데를 기준으로 폭을 hc 배.
        C_, l_ = np.asarray(line_proj[0], dtype=float), float(line_proj[1])
        mu_ = -float(C_[0, 2])
        c_, s_ = np.cos(float(line_proj[2])), np.sin(float(line_proj[2]))
        Rt_ = np.array([[c_, s_, 0.0], [-s_, c_, 0.0], [0.0, 0.0, 1.0]])
        Fg, Mg = Rt_ @ (-F), Rt_ @ (-M)               # 지면 → 로봇, 발 yaw 좌표 (z 축은 월드와 같음)
        fz_, fy_, my_ = max(float(Fg[2]), 0.0), float(Fg[1]), float(Mg[1])
        lo = max(-mu_ * (l_ * fz_ - my_) - l_ * fy_, l_ * fy_ - mu_ * (l_ * fz_ + my_))
        hi = min(mu_ * (l_ * fz_ - my_) - l_ * fy_, l_ * fy_ + mu_ * (l_ * fz_ + my_))
        mid, half = 0.5 * (lo + hi), max(0.0, 0.5 * (hi - lo)) * hc   # 구간이 한 점 (Fy 가 마찰 한계) 이면 반올림으로
        M = M.copy()                                                     # lo > hi 가 될 수 있다 → 가운데 점으로
        M[2] = -float(np.clip(Mg[2], mid - half, mid + half))
    tau = stance_torque(dyn, F, M)
    if bool(_cfg(cfg, "swing.walk_stance_bias", False)):   # 변형 walkbias (Q&A 9/29 Q7): 입각 다리 자중 · 코리올리 보상
        tau = tau + np.asarray(dyn.bias, dtype=float)
    return tau


# ---------------------------------------------------------------------------
# Swing
# ---------------------------------------------------------------------------
def apparent_inertia(Jv, M) -> np.ndarray:
    """computeApparentInertia (OperationalSpaceDynamics.cpp:34-46): Λ = (Jv M⁻¹ Jvᵀ + 1e-9 I)⁻¹ (LDLT → solve)."""
    Jv = np.asarray(Jv, dtype=float)
    lam_inv = Jv @ np.linalg.solve(np.asarray(M, dtype=float), Jv.T)
    lam_inv[np.diag_indices(3)] += 1e-9
    return np.linalg.solve(lam_inv, np.eye(3))


def swing_kp(dyn, cfg) -> np.ndarray:
    """computeSwingCartesianKp (OperationalSpaceDynamics.cpp:49-58): Kp = diag(ω_n² ⊙ diag(Λ)), ω_n = [151,151,110].
    Λ 의 대각만 쓴다 (비대각 무시)."""
    wn = np.asarray(_cfg(cfg, "swing.natural_frequency"), dtype=float)
    lam = apparent_inertia(dyn.Jv, dyn.M)
    return np.diag(wn * wn * np.diag(lam))


def swing_attitude_torque(state, leg: int, dyn, touchdown_yaw: float, cfg, yaw_des=None, yaw_rate_des: float = 0.0,
                          torso_wz: float = 0.0, pitch_bias: float = 0.0, pitch_rate_ref: float = 0.0) -> np.ndarray:
    """computeSwingAttitudeLevelTorque (SwingAttitudeControl.h:9-84).

    x̂,ŷ,ẑ = 정규화한 발 SITE 축 (R_WF 열), ω = Jw q̇_leg (torso 제외), up = e_z
    roll : e = atan2(x̂·(ẑ×up), ẑ·up), rate = x̂·ω, moment += (kp e − kd rate) x̂   (MIT: 0/0 → 생략)
    pitch: e = atan2(ŷ·(ẑ×up), ẑ·up), rate = ŷ·ω, moment += (300 e − 18 rate) ŷ
    yaw  : x̂_proj = normalize(x̂ − (x̂·up)up), d̂ = [cos ψ_td, sin ψ_td, 0]
           e = atan2(up·(x̂_proj×d̂), x̂_proj·d̂), rate = up·ω, moment += (305 e − 18 rate) up
    τ = Jwᵀ moment.  항이 켜지는 조건: kp > 0 또는 kd > 0. 축이 비유한/‖·‖≤1e-9 이면 C++ 처럼 예외.
    """
    Jw = np.asarray(dyn.Jw, dtype=float)
    dof = Jw.shape[1]
    rkp, rkd = float(_cfg(cfg, "swing.roll_kp", 0.0)), float(_cfg(cfg, "swing.roll_kd", 0.0))
    pkp, pkd = float(_cfg(cfg, "swing.pitch_kp", 0.0)), float(_cfg(cfg, "swing.pitch_kd", 0.0))
    ykp, ykd = float(_cfg(cfg, "swing.yaw_kp", 0.0)), float(_cfg(cfg, "swing.yaw_kd", 0.0))
    roll_on = rkp > 0.0 or rkd > 0.0
    pitch_on = pkp > 0.0 or pkd > 0.0
    yaw_on = ykp > 0.0 or ykd > 0.0
    if not (roll_on or pitch_on or yaw_on):
        return np.zeros(dof)
    qd = np.asarray(state.qd_leg[leg], dtype=float)
    if Jw.shape[0] != 3 or qd.size != dof:
        raise RuntimeError("Swing attitude control received inconsistent leg angular data")
    R = np.asarray(state.foot_R_W[leg], dtype=float)
    fx, fy, fz = R[:, 0].copy(), R[:, 1].copy(), R[:, 2].copy()
    for a in (fx, fy, fz):
        if not np.all(np.isfinite(a)) or np.linalg.norm(a) <= 1e-9:
            raise RuntimeError("Swing attitude control received invalid foot frame axes")
    fx /= np.linalg.norm(fx)
    fy /= np.linalg.norm(fy)
    fz /= np.linalg.norm(fz)
    omega = Jw @ qd
    moment = np.zeros(3)
    if roll_on:
        e = np.arctan2(fx.dot(np.cross(fz, _EZ)), fz.dot(_EZ))
        moment += (rkp * e - rkd * fx.dot(omega)) * fx
    ankle_tau = None
    if pitch_on:
        e = np.arctan2(fy.dot(np.cross(fz, _EZ)), fz.dot(_EZ))
        # 변형 pitchtraj (Q&A 10/6 Q28): 보정각 목표 pitch_bias · 각속도 목표 pitch_rate_ref (기본 0 → 기존식)
        m_pitch = (pkp * (e - pitch_bias) - pkd * (fy.dot(omega) - pitch_rate_ref)) * fy
        if int(_cfg(cfg, "swing.attitude_ankle_only", 0) or 0):
            # 변형 attank (Q&A 10/6 Q3): 발 pitch 자세 모멘트를 발목 관절에만 (Jwᵀ 로 보내면 축이 평행한
            # hip pitch · knee · ankle 세 관절에 같은 크기로 걸려 hip pitch 를 포화시켰다 — 10/6 Q2)
            ankle_tau = float(Jw[:, dof - 1].dot(m_pitch))
        else:
            moment += m_pitch
    if yaw_on:
        proj = fx - fx.dot(_EZ) * _EZ
        nrm = np.linalg.norm(proj)
        if nrm <= 1e-9:
            raise RuntimeError("Swing attitude control received degenerate yaw axis")
        proj = proj / nrm
        target = touchdown_yaw if yaw_des is None else float(yaw_des)   # 변형 yawtraj: 보간 목표 (Q&A 9/29 Q1)
        d = np.array([np.cos(target), np.sin(target), 0.0])
        e = np.arctan2(_EZ.dot(np.cross(proj, d)), proj.dot(d))
        # ω 는 다리 관절에 의한 (몸통 기준) 회전 → 목표 월드 yaw 속도에서 몸통 yaw 속도를 뺀 값과 비교
        rate_ref = 0.0 if (yaw_des is None or yaw_rate_des is None) else float(yaw_rate_des) - float(torso_wz)
        moment += (ykp * e - ykd * (_EZ.dot(omega) - rate_ref)) * _EZ
    tau = Jw.T @ moment
    if ankle_tau is not None:
        tau[dof - 1] += ankle_tau
    return tau


def swing_torque(dyn, state, leg: int, p_des, v_des, a_des, touchdown_yaw: float, cfg,
                 yaw_des=None, yaw_rate_des: float = 0.0, torso_wz: float = 0.0,
                 pitch_bias: float = 0.0, pitch_rate_ref: float = 0.0) -> np.ndarray:
    """SwingFoot 모드 (LegController.cpp:262-291 → OperationalSpaceDynamics.cpp:61-84) + 자세 토크 (MC:1361).

    F_fb = 0 (forceFeedForward) + Kp (p_des − p) + Kd (v_des − v)     p, v = 발 SITE 위치/속도 (state, world)
    τ    = Jvᵀ F_fb + Jvᵀ Λ (a_des − J̇v q̇) + bias + τ_attitude
    """
    Jv = np.asarray(dyn.Jv, dtype=float)
    kp = swing_kp(dyn, cfg)
    kd = np.diag(np.asarray(_cfg(cfg, "swing.kd_diag"), dtype=float))
    zeta = _cfg(cfg, "swing.kd_zeta", None)                   # 변형 kdz (Q&A 9/29 Q5): K_d = 2ζ·diag(Λ)·ω_n
    if zeta:
        wn = np.asarray(_cfg(cfg, "swing.natural_frequency"), dtype=float)
        kd = np.diag(2.0 * float(zeta) * np.diag(apparent_inertia(Jv, dyn.M)) * wn)
    p = np.asarray(state.foot_pos_W[leg], dtype=float)
    v = np.asarray(state.foot_vel_W[leg], dtype=float)
    f_fb = kp @ (np.asarray(p_des, dtype=float) - p) + kd @ (np.asarray(v_des, dtype=float) - v)
    a_res = np.asarray(a_des, dtype=float) - np.asarray(dyn.JvDot_qd, dtype=float)
    lam = apparent_inertia(Jv, dyn.M)
    tau = Jv.T @ f_fb + Jv.T @ (lam @ a_res) + np.asarray(dyn.bias, dtype=float)
    return tau + swing_attitude_torque(state, leg, dyn, touchdown_yaw, cfg, yaw_des, yaw_rate_des, torso_wz,
                                       pitch_bias, pitch_rate_ref)


def limit_swing_torque(tau_des, dyn, lo, hi, mode: int, w_att: float = 0.01, eps: float = 1e-4) -> np.ndarray:
    """변형 swlim (Q&A 9/30 Q4): 스윙 토크가 관절 한계를 넘을 때 관절별로 따로 자르지 않고 발 가속 방향을 지킨다.

    mode 1 (scale): τ = h + s·(τ_des − h), s = 모든 관절이 [lo, hi] 안에 드는 가장 큰 값 (0..1). h = 중력·코리올리.
    mode 2 (qp)   : x = Δq̈ 를 변수로  min ‖J_v x‖² + w_att‖J_ω x‖² + eps‖x‖²  s.t.  lo ≤ τ_des + M x ≤ hi,  τ = τ_des + M x
                    (다리만의 동역학 τ = M q̈ + h 에서 발 위치 가속 오차를 가장 작게, 발 자세는 작은 가중치).
    한계 안이면 두 방법 모두 τ_des 그대로 (s = 1, x = 0)."""
    tau_des = np.asarray(tau_des, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if np.all(tau_des <= hi) and np.all(tau_des >= lo):
        return tau_des
    if int(mode) == 1:
        h = np.asarray(dyn.bias, dtype=float)
        d = tau_des - h
        s = 1.0
        for j in range(tau_des.size):
            if tau_des[j] > hi[j] and d[j] > 1e-12:
                s = min(s, (hi[j] - h[j]) / d[j])
            elif tau_des[j] < lo[j] and d[j] < -1e-12:
                s = min(s, (lo[j] - h[j]) / d[j])
        s = min(max(s, 0.0), 1.0)
        return h + s * d
    import quadprog
    Jv = np.asarray(dyn.Jv, dtype=float)
    Jw = np.asarray(dyn.Jw, dtype=float)
    M = np.asarray(dyn.M, dtype=float)
    n = tau_des.size
    G = Jv.T @ Jv + w_att * (Jw.T @ Jw) + eps * np.eye(n)
    # quadprog: min ½xᵀGx − aᵀx  s.t. Cᵀx ≥ b.   M x ≤ hi − τ_des  →  −M x ≥ τ_des − hi ;  M x ≥ lo − τ_des
    C = np.hstack([-M.T, M.T])
    b = np.concatenate([tau_des - hi, lo - tau_des])
    try:
        x = quadprog.solve_qp(G, np.zeros(n), C, b, 0)[0]
    except ValueError:                           # 풀 수 없으면 (중력만으로도 한계 밖 등) 관절별 자르기로
        return np.clip(tau_des, lo, hi)
    return tau_des + M @ x


# ---------------------------------------------------------------------------
# Standing / arms
# ---------------------------------------------------------------------------
def standing_torque(Jv, Jw, u) -> np.ndarray:
    """writeStandingLegCommands (My_Controller.cpp:1254-1295).
    Jv, Jw (6,10): 행 [L 발 3; R 발 3], 열 [L 다리 5 | R 다리 5].
    τ(10) = Jvᵀ(−[F_L;F_R]) + Jwᵀ(−[M_L;M_R]) → (2,5). ramp·yaw hold·bias·PD 없음."""
    u = np.asarray(u, dtype=float)
    f = -np.concatenate([u[0:3], u[3:6]])
    m = -np.concatenate([u[6:9], u[9:12]])
    tau = np.asarray(Jv, dtype=float).T @ f + np.asarray(Jw, dtype=float).T @ m
    return tau.reshape(2, 5)


def arm_pd(state, q_des, cfg, qd_des=None) -> np.ndarray:
    """ArmController.cpp:204-221: τ = kp (q_des − q) + kd (0 − q̇), kp/kd = joint_tracking.arm (100, 5).
    qd_des: 변형 (팔 흔들기, Q&A 9/28) 의 목표 속도 — None 이면 0 (reference)."""
    kp = np.asarray(_cfg(cfg, "joint_tracking.arm.kp"), dtype=float)
    kd = np.asarray(_cfg(cfg, "joint_tracking.arm.kd"), dtype=float)
    q = np.asarray(state.q_arm, dtype=float).reshape(2, 4)
    qd = np.asarray(state.qd_arm, dtype=float).reshape(2, 4)
    qd_ref = 0.0 if qd_des is None else np.asarray(qd_des, dtype=float).reshape(2, 4)
    return kp[None] * (np.asarray(q_des, dtype=float).reshape(2, 4) - q) + kd[None] * (qd_ref - qd)
