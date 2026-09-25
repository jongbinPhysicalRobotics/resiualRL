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


def stance_yaw_hold_moment_z(state, leg: int, dyn, touchdown_yaw: float, cfg) -> float:
    """computeStanceYawHoldMomentWorld (My_Controller.cpp:86-117) 의 z 성분 (x,y 는 항상 0).

    x̂_proj = 수평 투영·정규화한 발 SITE x축, d̂ = [cos ψ_td, sin ψ_td, 0]
    err = atan2(ẑ·(x̂_proj × d̂), x̂_proj·d̂),  rate = ẑ·(Jw q̇_leg)  (torso 제외 상대 각속도)
    m_z = kp·err − kd·rate   (kp = stance_yaw_kp 20, kd = stance_yaw_kd 4)
    0 을 돌려주는 경우: kp ≤ 0 이고 kd ≤ 0, ψ_td 비유한, x̂_proj 비유한 또는 ‖x̂_proj‖ ≤ 1e-9.
    """
    kp = float(_cfg(cfg, "swing.stance_yaw_kp", 0.0))
    kd = float(_cfg(cfg, "swing.stance_yaw_kd", 0.0))
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
    return float(kp * err - kd * rate)


def walking_stance_torque(dyn, state, leg: int, u0, alpha: float, touchdown_yaw: float, cfg) -> np.ndarray:
    """writeLegCommands stance 가지 (My_Controller.cpp:1315-1350) + stance_torque.
    F = −α u0[F_leg], M = −α u0[M_leg]; enable_stance_foot_yaw_hold 이고 α > 0 이면 M.z += α·m_z."""
    u0 = np.asarray(u0, dtype=float)
    if leg == LEFT:
        F = -alpha * u0[0:3]
        M = -alpha * u0[6:9]
    else:
        F = -alpha * u0[3:6]
        M = -alpha * u0[9:12]
    if bool(_cfg(cfg, "swing.enable_stance_foot_yaw_hold", False)) and alpha > 0.0:
        M = M.copy()
        M[2] += alpha * stance_yaw_hold_moment_z(state, leg, dyn, touchdown_yaw, cfg)
    return stance_torque(dyn, F, M)


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


def swing_attitude_torque(state, leg: int, dyn, touchdown_yaw: float, cfg) -> np.ndarray:
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
    if pitch_on:
        e = np.arctan2(fy.dot(np.cross(fz, _EZ)), fz.dot(_EZ))
        moment += (pkp * e - pkd * fy.dot(omega)) * fy
    if yaw_on:
        proj = fx - fx.dot(_EZ) * _EZ
        nrm = np.linalg.norm(proj)
        if nrm <= 1e-9:
            raise RuntimeError("Swing attitude control received degenerate yaw axis")
        proj = proj / nrm
        d = np.array([np.cos(touchdown_yaw), np.sin(touchdown_yaw), 0.0])
        e = np.arctan2(_EZ.dot(np.cross(proj, d)), proj.dot(d))
        moment += (ykp * e - ykd * _EZ.dot(omega)) * _EZ
    return Jw.T @ moment


def swing_torque(dyn, state, leg: int, p_des, v_des, a_des, touchdown_yaw: float, cfg) -> np.ndarray:
    """SwingFoot 모드 (LegController.cpp:262-291 → OperationalSpaceDynamics.cpp:61-84) + 자세 토크 (MC:1361).

    F_fb = 0 (forceFeedForward) + Kp (p_des − p) + Kd (v_des − v)     p, v = 발 SITE 위치/속도 (state, world)
    τ    = Jvᵀ F_fb + Jvᵀ Λ (a_des − J̇v q̇) + bias + τ_attitude
    """
    Jv = np.asarray(dyn.Jv, dtype=float)
    kp = swing_kp(dyn, cfg)
    kd = np.diag(np.asarray(_cfg(cfg, "swing.kd_diag"), dtype=float))
    p = np.asarray(state.foot_pos_W[leg], dtype=float)
    v = np.asarray(state.foot_vel_W[leg], dtype=float)
    f_fb = kp @ (np.asarray(p_des, dtype=float) - p) + kd @ (np.asarray(v_des, dtype=float) - v)
    a_res = np.asarray(a_des, dtype=float) - np.asarray(dyn.JvDot_qd, dtype=float)
    lam = apparent_inertia(Jv, dyn.M)
    tau = Jv.T @ f_fb + Jv.T @ (lam @ a_res) + np.asarray(dyn.bias, dtype=float)
    return tau + swing_attitude_torque(state, leg, dyn, touchdown_yaw, cfg)


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


def arm_pd(state, q_des, cfg) -> np.ndarray:
    """ArmController.cpp:204-221: τ = kp (q_des − q) + kd (0 − q̇), kp/kd = joint_tracking.arm (100, 5)."""
    kp = np.asarray(_cfg(cfg, "joint_tracking.arm.kp"), dtype=float)
    kd = np.asarray(_cfg(cfg, "joint_tracking.arm.kd"), dtype=float)
    q = np.asarray(state.q_arm, dtype=float).reshape(2, 4)
    qd = np.asarray(state.qd_arm, dtype=float).reshape(2, 4)
    return kp[None] * (np.asarray(q_des, dtype=float).reshape(2, 4) - q) + kd[None] * (0.0 - qd)
