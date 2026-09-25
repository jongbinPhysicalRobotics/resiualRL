"""Owner A 단위 점검 (07_port_design.md §4 [A]). PASS/FAIL 줄 출력, 실패 시 exit 1.

  - config 파생값
  - 축소 몸체 at "stand" = spec 01 §4 숫자, yaw 회전 불변성, x0 at spawn
  - torso 속도 의미 (mjOBJ_BODY = 몸통 COM 점의 속도)
  - yaw unwrap: ±π 를 여러 번 넘는 합성 시퀀스, 시간 역행 리셋
  - CommandFilter: 첫 호출 = raw, dt = 0 snap, 95 % 상승 시간 ≈ tau·ln 20 ≈ 3 tau, 즉시 정지, clamp, zero-motion,
    높이 통과 / roll·pitch 오프셋 필터
  - CommandSchedule: constant / step (첫 점 값이 그 전에도) / reference 램프
  - FSM: StandingSettle → Walking at 1.0 s (MuJoCo 식 시간 누적, 틱당 2 번 update), interactive braking 경로
  - write_torque: 이름 기반 scatter + ctrlrange clamp
  - standing_jacobians 블록 구조, D4 (01_check_model.check_d4) + 대조군

실행:  PYTHONIOENCODING=utf-8 ../../../.venv/Scripts/python.exe 02_unit_checks_A.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import command as cmdm  # noqa: E402
import config  # noqa: E402
import fsm as fsmm  # noqa: E402
import mit_model as mm  # noqa: E402
from mit_model import LEFT, RIGHT  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


def quat_yaw(a):
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


# ----------------------------------------------------------------------------------------------------------------
def t_config(cfg):
    ok = (cfg.dt == 0.002 and abs(cfg.dt_mpc - 0.02) < 1e-15 and cfg.N == 25 and abs(cfg.stance_frac - 0.66) < 1e-12
          and cfg.gravity == -9.81)
    check("config derived", ok, f"dt {cfg.dt} dt_mpc {cfg.dt_mpc} N {cfg.N} stance_frac {cfg.stance_frac} g {cfg.gravity}")
    hs = [cfg.half_stance_offset(v) for v in (0.0, 0.60, 0.6000001, 0.65, 0.66)]
    check("config half_stance_offset rule", hs == [0.37, 0.37, 0.28, 0.28, 0.26], f"{hs}")
    ok = (cfg.timing.cycle == 0.5 and list(cfg.mpc.walking.state_weight_diag[:3]) == [50000, 9000, 500]
          and cfg.contact_manager.contact_ramp_duration == 0.01 and cfg.swing.natural_frequency.shape == (3,)
          and cfg.swing.nominal_foot_offsets_B.shape == (2, 3) and cfg.port.spawn_keyframe == "stand"
          and cfg.port.solver == "osqp" and cfg.port.armature is True and cfg.port.skip_leg_pd_init is True)
    check("config nested access + port section", ok, f"port {cfg.port.to_dict()}")


def t_reduced_body(model):
    model.reset("stand")
    s = model.read_state()
    r = s.reduced
    spec_c = np.array([0.015306, 0.002857, 0.116574])
    spec_I = np.array([[0.519367, 0.001639, 0.005789], [0.001639, 0.168406, -0.001019],
                       [0.005789, -0.001019, 0.402526]])
    dm, dc, dI = abs(r.mass - 14.227468), np.abs(r.com_offset_B - spec_c).max(), np.abs(r.inertia_B - spec_I).max()
    check("reduced body at stand = spec 01 §4", dm < 1e-6 and dc < 1e-6 and dI < 1e-6,
          f"mass {r.mass:.6f} (d {dm:.1e})  com {np.round(r.com_offset_B, 6)} (d {dc:.1e})  I max d {dI:.1e}")
    check("reduced body excludes legs (11 bodies)", len(model.reduced_bodies) == 11 and
          not any(model.in_leg_subtree[model.reduced_bodies]), f"{len(model.reduced_bodies)} bodies")
    x0 = s.x0()
    exp = np.r_[0, 0, 0, spec_c + [0, 0, 0.679472], np.zeros(6), -9.81]
    check("x0 at spawn", np.abs(x0 - exp).max() < 1e-6, f"x0 {np.round(x0, 6)}")
    check("feet at spawn (site z = 0, under whole CoM x)",
          np.abs(s.foot_pos_W[:, 2]).max() < 1e-6 and np.abs(s.foot_pos_W[:, 0] - model.whole_com_W()[0]).max() < 1e-5
          and s.foot_contact.all(),
          f"site L {np.round(s.foot_pos_W[LEFT], 6)} R {np.round(s.foot_pos_W[RIGHT], 6)} contact {s.foot_contact}")

    # yaw 를 돌려도 yaw 프레임 값은 불변, x0 의 com 은 Rz(yaw) 로 돈다
    m, d = model.m, model.d
    yaw = 2.5
    d.qpos[3:7] = quat_yaw(yaw)
    mujoco.mj_forward(m, d)
    model.reset_yaw_unwrap()
    s2 = model.read_state()
    dcy = np.abs(s2.reduced.com_offset_B - r.com_offset_B).max()
    dIy = np.abs(s2.reduced.inertia_B - r.inertia_B).max()
    dcw = np.abs(s2.com_W() - s2.reduced.com_W).max()
    check("reduced body yaw-frame invariance (yaw 2.5)", dcy < 1e-12 and dIy < 1e-12 and dcw < 1e-12,
          f"d com {dcy:.1e}  d I {dIy:.1e}  |com_W(x0) − Σm xipos/M| {dcw:.1e}")

    # 기울어진 몸통: com_offset 은 yaw 프레임 (roll/pitch 는 반영됨)
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([0.1, -0.2, 0.7]), "XYZ")
    d.qpos[3:7] = q
    mujoco.mj_forward(m, d)
    model.reset_yaw_unwrap()
    s3 = model.read_state()
    Rw = mm.Rz(s3.yaw_wrapped)
    check("com offset = Rz(yaw)ᵀ (com_W − torso origin) when tilted",
          np.abs(Rw @ s3.reduced.com_offset_B + s3.torso_pos_W - s3.reduced.com_W).max() < 1e-12,
          f"roll {s3.roll:.4f} pitch {s3.pitch:.4f} yaw {s3.yaw_wrapped:.4f}")
    model.reset("stand")


def t_velocity_semantics(model):
    m, d = model.m, model.d
    model.reset("stand")
    rng = np.random.default_rng(3)
    d.qvel[:] = rng.normal(size=m.nv)
    mujoco.mj_forward(m, d)
    s = model.read_state()
    b = model.base_body
    w_W = d.xmat[b].reshape(3, 3) @ d.qvel[3:6]              # free joint 각속도는 몸통 로컬 축
    v_com = d.qvel[0:3] + np.cross(w_W, d.xipos[b] - d.xpos[b])
    check("torso_angvel_W = world ω", np.abs(s.torso_angvel_W - w_W).max() < 1e-12, "")
    check("torso_vel_W = velocity of torso COM (mjOBJ_BODY), not origin",
          np.abs(s.torso_vel_W - v_com).max() < 1e-12 and np.abs(s.torso_vel_W - d.qvel[0:3]).max() > 1e-3,
          f"v {np.round(s.torso_vel_W, 4)}  origin {np.round(d.qvel[0:3], 4)}")
    x0 = s.x0()
    off = mm.Rz(s.yaw_unwrapped) @ s.reduced.com_offset_B
    check("x0 v_com = v_torso + ω × Rz(ψ) com_offset",
          np.abs(x0[9:12] - (s.torso_vel_W + np.cross(s.torso_angvel_W, off))).max() < 1e-14, "")
    model.reset("stand")


def t_yaw_unwrap(model):
    m, d = model.m, model.d
    model.reset("stand")
    # 합성 시퀀스: +0.35 rad/tick 로 3 바퀴, 그다음 −0.5 rad/tick 로 5 바퀴 되돌기 (±π 를 여러 번 넘음)
    seq = np.r_[np.arange(0, 3 * 2 * np.pi, 0.35), 3 * 2 * np.pi - np.arange(0, 5 * 2 * np.pi, 0.5)]
    err = 0.0
    t = 0.0
    crosses = 0
    prev_w = None
    for k, a in enumerate(seq):
        d.qpos[3:7] = quat_yaw(a)
        d.time = t
        mujoco.mj_forward(m, d)
        s = model.read_state()
        err = max(err, abs(s.yaw_unwrapped - a))
        if prev_w is not None and abs(s.yaw_wrapped - prev_w) > np.pi:
            crosses += 1
        prev_w = s.yaw_wrapped
        t += 0.002
    check("yaw unwrap across ±pi (synthetic)", err < 1e-9 and crosses >= 6,
          f"{len(seq)} samples, {crosses} wrap crossings, max |unwrapped − true| {err:.2e}")
    # 시간 역행 → wrapped 로 리셋 (StateEstimator.cpp:32)
    d.time = t - 1.0
    mujoco.mj_forward(m, d)
    s = model.read_state()
    check("yaw unwrap resets when time goes backwards", abs(s.yaw_unwrapped - s.yaw_wrapped) < 1e-15,
          f"unwrapped {s.yaw_unwrapped:.4f} wrapped {s.yaw_wrapped:.4f}")
    model.reset("stand")
    s = model.read_state()
    check("reset() resets yaw unwrap", s.yaw_unwrapped == 0.0, "")


def t_filter(cfg):
    f = cmdm.CommandFilter(cfg)
    raw = cmdm.UserCommand(x_dot=0.6, y_dot=0.3, psi_dot=1.3)
    out = f.update(raw, 0.002)
    check("filter first call: filtered = raw", (out.x_dot, out.y_dot, out.psi_dot) == (0.6, 0.3, 1.3), f"{out}")

    f.reset()
    f.update(cmdm.UserCommand(), 0.0)                        # 첫 호출 (0)
    dt = 0.002
    t95 = {}
    for fld, v, tau in (("x_dot", 0.6, 0.92), ("y_dot", 0.3, 0.8), ("psi_dot", 1.3, 0.70)):
        f.reset()
        f.update(cmdm.UserCommand(), 0.0)
        r = cmdm.UserCommand(**{fld: v})
        n = 0
        o = None
        while True:
            n += 1
            o = f.update(r, dt)
            if getattr(o, fld) >= 0.95 * v:
                break
        t95[fld] = (n * dt, tau * np.log(20.0), tau)
    ok = all(abs(tt - th) <= dt + 1e-12 for tt, th, _ in t95.values())
    check("filter 95% rise time = tau·ln20 (≈3 tau)", ok,
          "  ".join(f"{k}: {tt:.3f}s vs {th:.3f}s (3tau {3 * tau:.2f})" for k, (tt, th, tau) in t95.items()))

    f.reset()
    f.update(cmdm.UserCommand(), 0.0)
    r = cmdm.UserCommand(x_dot=0.6)
    a = -np.expm1(-dt / 0.92)
    o1 = f.update(r, dt)
    check("filter one tick = alpha·raw", abs(o1.x_dot - a * 0.6) < 1e-15, f"{o1.x_dot:.6e} alpha {a:.6e}")
    for _ in range(100):
        f.update(r, dt)
    mid = f.filtered.x_dot
    o = f.update(cmdm.UserCommand(x_dot=0.45, psi_dot=0.8, standing_roll_offset_rad=0.1), 0.0)
    check("filter dt=0 snap (filtered = raw)", o.x_dot == 0.45 and o.psi_dot == 0.8 and o.standing_roll_offset_rad == 0.1,
          f"before {mid:.4f} → {o.x_dot}, psi {o.psi_dot}, roll {o.standing_roll_offset_rad}")
    o = f.update(cmdm.UserCommand(body_height_offset_m=0.02), dt)
    check("filter instant zero when x,y,psi raw all 0; height passes", o.x_dot == 0 and o.y_dot == 0 and o.psi_dot == 0
          and o.body_height_offset_m == 0.02, f"{o}")
    o = f.update(cmdm.UserCommand(x_dot=5.0, y_dot=-3.0, psi_dot=9.0), 0.0)
    check("filter clamp to 0.7/0.5/2.0", (o.x_dot, o.y_dot, o.psi_dot) == (0.7, -0.5, 2.0), f"{o}")
    o = f.update(cmdm.UserCommand(x_dot=0.6, standing_pitch_offset_rad=0.2), dt, zero_motion=True)
    ap = -np.expm1(-dt / 0.70)
    check("filter zero_motion → 0; pitch offset low-passed with tau 0.70",
          o.x_dot == 0 and abs(o.standing_pitch_offset_rad - ap * 0.2) < 1e-15,
          f"x {o.x_dot} pitch {o.standing_pitch_offset_rad:.3e}")
    check("alpha rule (dt<=0 or tau<=0 → 1)", cmdm.low_pass_blend_alpha(0.5, 0.0) == 1.0
          and cmdm.low_pass_blend_alpha(0.0, 0.002) == 1.0 and cmdm.low_pass_blend_alpha(0.5, -1) == 1.0, "")


def t_schedule():
    s = cmdm.CommandSchedule.constant(0.6, 0.0, 1.3)
    c0, c9 = s(0.0), s(99.0)
    check("schedule constant from t=0", c0.x_dot == 0.6 and c0.psi_dot == 1.3 and c9.x_dot == 0.6, f"{c0}")
    s = cmdm.CommandSchedule({"x_dot": [(1.0, 0.2), (3.0, 0.6)]})
    v = [s(t).x_dot for t in (0.0, 0.999, 1.0, 2.9, 3.0, 10.0)]
    check("schedule step (first value applies before its time)", v == [0.2, 0.2, 0.2, 0.2, 0.6, 0.6], f"{v}")
    s = cmdm.CommandSchedule.step_at(1.0, y_dot=0.3)
    check("schedule step_at", s(0.998).y_dot == 0.0 and s(1.0).y_dot == 0.3, "")
    s = cmdm.CommandSchedule({"x_dot": {"final": -0.6, "rate": 0.3, "start": 1.0}})
    v = [s(t).x_dot for t in (0.5, 1.0, 2.0, 3.0, 5.0)]
    check("schedule reference ramp", np.allclose(v, [0, 0, -0.3, -0.6, -0.6]), f"{v}")
    s = cmdm.CommandSchedule({"psi_dot": {"points": [(0, 0), (2, 1.3)], "interp": "linear"}})
    check("schedule linear", abs(s(1.0).psi_dot - 0.65) < 1e-12 and s(5).psi_dot == 1.3, "")


def t_fsm(cfg):
    dt = 0.002
    fsm = fsmm.LocomotionFSM.from_cfg(cfg, start_time=0.0)
    check("FSM initial state StandingSettle (walking, settle 1.0)",
          fsm.state == fsmm.LocomotionState.STANDING_SETTLE and fsm.mode == "standing", f"{fsm.state}")
    t = 0.0
    first_walk = None
    resets = 0
    second_noop = True
    for k in range(1000):
        if k == 0:
            outs = [fsm.update(t)]                              # 첫 컨트롤러 틱: 1 번 (prepareController 생략)
        else:
            outs = [fsm.update(t), fsm.update(t)]               # prepareController + runController
        resets += sum(o.reset_gait_clock for o in outs)
        if len(outs) == 2 and outs[1].just_transitioned:
            second_noop = False
        if first_walk is None and outs[-1].mode == "walking":
            first_walk = (k, t, outs[0].reset_swing_state, outs[0].swing_leg_dynamics)
        t += dt                                                 # MuJoCo mj_step: d.time += timestep
    k, tw, rs, sw = first_walk
    check("FSM StandingSettle → Walking at 1.0 s", abs(tw - 1.0) <= dt + 1e-12 and rs and sw and resets == 1
          and second_noop, f"tick {k}, t = {tw!r}, reset_swing {rs}, swing_dyn {sw}, resets {resets}")
    check("FSM toggle ignored for mode walking", fsm.request_toggle() is False and fsm.mode == "walking", "")

    # interactive: braking 경로 (틱당 두 번 update → settle tick 이 2 씩 증가, hold 3 은 2 틱에 도달)
    f2 = fsmm.LocomotionFSM("interactive", 0.0, 0.01, 0.01, 1.0, 3, 3.0, 5, start_time=0.0)
    o = f2.update(0.0)
    check("FSM interactive, settle 0 → Standing", f2.state == fsmm.LocomotionState.STANDING, f"{f2.state}")
    f2.request_toggle()
    o = f2.update(0.002)
    ok = f2.state == fsmm.LocomotionState.WALKING and o.reset_gait_clock
    f2.request_toggle()
    t = 0.004
    o = f2.update(t)
    ok &= f2.state == fsmm.LocomotionState.BRAKING_TO_STANDING and o.zero_motion_command and not o.reset_gait_clock
    t_ready = None
    while t < 10.0:
        t += dt
        f2.update(t, 0.0, 0.0, 0.0)
        f2.update(t, 0.0, 0.0, 0.0)
        if f2.braking_ready and t_ready is None:
            t_ready = t
        if f2.state == fsmm.LocomotionState.STANDING:
            break
    ok &= t_ready is not None and abs(t_ready - (0.004 + 1.0)) < 2 * dt + 1e-9 and abs(t - (t_ready + 3.0)) < 2 * dt
    check("FSM interactive braking: window 1.0 s, ready, timeout 3.0 s → Standing", ok,
          f"ready at {t_ready}, standing at {t:.3f}")


def t_write_torque(model):
    model.reset("stand")
    tau_leg = np.array([[1, 2, 3, 1000, 5], [-6, -7, -8, -9, -1000.0]])
    tau_arm = np.array([[11, 12, 13, 100], [15, 16, 17, 18.0]])
    raw = model.write_torque(tau_leg, tau_arm)
    m, d = model.m, model.d
    a_lk = m.actuator("a09_left_knee").id
    a_ra = m.actuator("a05_right_ankle").id
    a_le = m.actuator("a18_left_elbow").id
    a_rsp = m.actuator("a11_right_shoulder_pitch").id
    ok = (d.ctrl[a_lk] == 144 and raw[a_lk] == 1000 and d.ctrl[a_ra] == -68 and d.ctrl[a_le] == 55
          and d.ctrl[a_rsp] == 15 and d.ctrl[m.actuator("a06_left_hip_yaw").id] == 1)
    check("write_torque scatter by name + ctrlrange clamp", ok,
          f"L knee {d.ctrl[a_lk]} R ankle {d.ctrl[a_ra]} L elbow {d.ctrl[a_le]} R sh_pitch {d.ctrl[a_rsp]}")
    d.ctrl[:] = 0


def t_standing_jac(model):
    model.reset("stand")
    Jv, Jw = model.standing_jacobians()
    ok = Jv.shape == (6, 10) and Jw.shape == (6, 10) and np.all(Jv[0:3, 5:] == 0) and np.all(Jv[3:, :5] == 0)
    dL = model.leg_dynamics(LEFT)
    ok &= np.array_equal(Jv[0:3, 0:5], dL.Jv) and np.array_equal(Jw[3:6, 5:10], model.leg_dynamics(RIGHT).Jw)
    # 발 Jacobian 이 맞는지: 작은 관절 변화에 대한 site 이동 (유한 차분)
    m, d = model.m, model.d
    eps = 1e-7
    q0 = d.qpos.copy()
    p0 = d.site_xpos[model.foot_site[LEFT]].copy()
    fd = np.zeros((3, 5))
    for j in range(5):
        d.qpos[:] = q0
        d.qpos[model.leg_qadr[LEFT, j]] += eps
        mujoco.mj_kinematics(m, d)
        fd[:, j] = (d.site_xpos[model.foot_site[LEFT]] - p0) / eps
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    e = np.abs(fd - dL.Jv).max()
    check("standing_jacobians 6x10 block structure + finite-difference Jv", ok and e < 1e-5, f"|FD − Jv| {e:.1e}")
    check("leg M symmetric PD, includes armature", np.allclose(dL.M, dL.M.T) and np.all(np.linalg.eigvalsh(dL.M) > 0)
          and dL.M[3, 3] > m.dof_armature[model.leg_dof[LEFT, 3]], f"diag {np.round(dL.M.diagonal(), 4)}")


def t_d4():
    cm = importlib.import_module("01_check_model")
    w = cm.check_d4(3, seed=0, armature=True, verbose=False)
    check("D4 full-model scratch == MjSpec fixed-base leg model (3 random configs)",
          all(v < 1e-9 for v in w.values()), "  ".join(f"{k} {v:.1e}" for k, v in w.items()))
    nc = cm.negative_control()
    check("D4 negative control (base velocity not zeroed differs)", nc["bias"] > 1e-3,
          f"|dbias| {nc['bias']:.3f}  |dJvDot qd| {nc['JvDot_qd']:.3f}")


def t_markers(cfg):
    model = mm.MitModel(emulate_debug_markers=True)
    model.reset("stand")
    s0 = model.read_state()
    model.debug_marker_fn = lambda: {"debug_left_touchdown_target": (np.array([0.1, 0.1, -0.005]), np.array([1, 0, 0, 0.]))}
    s1 = model.read_state()
    check("U1 debug-marker emulation adds 0.2141 kg", abs(s0.reduced.mass - 14.227468 - 0.2141) < 1e-3,
          f"mass {s0.reduced.mass:.4f}; Iyy {s0.reduced.inertia_B[1, 1]:.4f} → {s1.reduced.inertia_B[1, 1]:.4f} "
          f"after moving L touchdown marker to the ground")


def main():
    np.set_printoptions(precision=6, suppress=True)
    cfg = config.load()
    model = mm.MitModel(armature=cfg.port.armature)
    t_config(cfg)
    t_reduced_body(model)
    t_velocity_semantics(model)
    t_yaw_unwrap(model)
    t_filter(cfg)
    t_schedule()
    t_fsm(cfg)
    t_write_torque(model)
    t_standing_jac(model)
    t_d4()
    t_markers(cfg)
    print(f"\n{len(FAILS)} FAIL" + (f": {FAILS}" if FAILS else " — all PASS"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
