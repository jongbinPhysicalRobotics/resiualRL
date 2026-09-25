"""통합 스모크 테스트 — 서기 (Standing) 2.0 s, controller.py 없이 모듈을 직접 이어 붙인다.

07_port_design.md §2 루프 순서 (spec 05 §2, MC:1374-1429 중 Standing 에 해당하는 단계만):
    state = read_state()            (mj_forward 없음 — 1-tick stale 운동학)
    clock.sync(t); x0 = state.x0()
    dt = max(0, t - last_control_time); filtered = filter.update(raw, dt)
    body_target.update(x0, dt, 'standing', foot_pos, filtered)
    nominal = 측정 발 (z = -0.005)  → contact_manager.update → managed_foot_positions
    MPC tick (tick 0, 이후 7 tick 마다): foot yaw, horizon (모두 stance, override 없음),
        seed = x0 [0:3]=euler, [3:6]=nominal, [5]=nominal_height → build_reference → ConvexMPC.solve('standing')
    tau_leg = standing_torque(6x10 J, u)    (ramp/bias/PD 없음, MC:1254-1295)
    tau_arm = arm PD → [0,0,0,-1.65]
    write_torque → mj_step
시작: keyframe 'stand' + mj_forward 한 번 (D2).

실행: python 03_integration_smoke.py [--seconds 2.0] [--solver osqp|quadprog] [--noarmature]
통과 기준: 2 s 동안 넘어지지 않음 (base z 변화 < 3 cm, |roll|,|pitch| < 0.05 rad), MPC fallback 0.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time

import mujoco
import numpy as np

import config
from command import CommandFilter, UserCommand
from contact_manager import ContactManager
from gait import GaitScheduler, HorizonClock, stance_table
from leg_control import arm_pd, standing_torque
from mit_model import LEFT, RIGHT, MitModel
from mpc import ConvexMPC, build_reference, mpc_foot_yaw
from planner import BodyTarget, planar_speed, standing_foot_targets, swing_foot_yaw_target_world, \
    swing_foot_yaw_psi_offset


def run(seconds=2.0, solver="osqp", armature=True, verbose=True):
    cfg = config.load()
    model = MitModel(armature=armature)
    model.reset(str(cfg.port.spawn_keyframe))
    m, d = model.m, model.d
    t = float(d.time)

    # --- initializeController (MC:393-474), 모드는 Standing 고정 ---
    clock = HorizonClock(t, cfg)
    gait = GaitScheduler(cfg, clock)
    gait.set_mode("standing")
    cm = ContactManager(cfg)
    filt = CommandFilter(cfg)
    body = BodyTarget()
    mpc = ConvexMPC(cfg, solver=solver, verbose_errors=True)
    raw_cmd = UserCommand()                                    # 명령 0 (서기)
    q_arm_des = np.tile(np.asarray(cfg.initial_pose.arm_joint_offsets, dtype=float), (2, 1))

    state0 = model.read_state()
    cm.reset(state0, gait, "standing", t)
    filtered = filt.update(raw_cmd, 0.0)
    speed = planar_speed(filtered.x_dot, filtered.y_dot)
    base_yaw = swing_foot_yaw_target_world(state0.yaw_unwrapped, filtered.psi_dot, speed, cfg)
    touchdown_yaw = np.array([base_yaw + swing_foot_yaw_psi_offset(leg, filtered.psi_dot) for leg in (LEFT, RIGHT)])
    last_control_time = t
    iteration, last_mpc_iteration = 0, 0
    n_between = cfg.iterations_between_solve
    u_hold = np.zeros(12)
    # 'reset' 이 read_state 의 yaw unwrap 을 이미 한 번 소비 — 같은 시각 재호출은 무해 (dt = 0)

    base_z0 = float(d.qpos[2])
    n_steps = int(round(seconds / m.opt.timestep))
    log = dict(t=[], z=[], roll=[], pitch=[], yaw=[], com=[], err=[], fz=[], tau=[])
    infos = []
    fell = False
    t_wall = time.perf_counter()
    for step in range(n_steps):
        t = float(d.time)
        state = model.read_state()
        clock.sync(t)
        x0 = state.x0(cfg.gravity)
        dt = max(0.0, t - last_control_time)
        filtered = filt.update(raw_cmd, dt)
        body.update(x0, dt, "standing", state.foot_pos_W, filtered, cfg, com_offset_B=state.reduced.com_offset_B)
        nominal = standing_foot_targets(state.foot_pos_W)
        cm.update(state, gait, "standing", nominal, t)
        desired = cm.managed_foot_positions(nominal)
        last_control_time = t                                  # updateSwingTrajectories (standing: 전부 stance)

        if iteration == 0 or iteration - last_mpc_iteration >= n_between:
            active = np.array([gait.contact(LEFT, t), gait.contact(RIGHT, t)])   # standing: gait.c = True
            foot_yaw = np.array([mpc_foot_yaw(state, leg, bool(active[leg]), float(touchdown_yaw[leg]))
                                 for leg in (LEFT, RIGHT)])
            steps = gait.horizon_steps(clock, None)            # override 는 walking 에서만 (MC:953-956)
            seed = x0.copy()
            seed[0:3] = body.euler_W
            seed[3:6] = body.nominal_position_W
            seed[5] = body.nominal_height_W
            ref = build_reference(filtered, seed, desired, clock, cfg)
            u_hold, info = mpc.solve(x0, ref, steps, foot_yaw, state.reduced, "standing", active=active)
            info["x0"] = x0.copy()
            info["xref0"] = ref.X_ref[0].copy()
            info["t"] = t
            info["stance_all"] = bool(stance_table(steps).all())
            infos.append(info)
            last_mpc_iteration = iteration

        Jv, Jw = model.standing_jacobians()
        tau_leg = standing_torque(Jv, Jw, u_hold)
        tau_arm = arm_pd(state, q_arm_des, cfg)
        model.write_torque(tau_leg, tau_arm)

        roll, pitch = state.roll, state.pitch
        log["t"].append(t)
        log["z"].append(float(d.qpos[2]))
        log["roll"].append(roll)
        log["pitch"].append(pitch)
        log["yaw"].append(state.yaw_unwrapped)
        log["com"].append(x0[3:6].copy())
        log["err"].append(x0[0:12] - np.r_[body.euler_W, body.position_W, 0, 0, 0, 0, 0, 0])
        log["fz"].append(state.foot_normal_force.copy())
        log["tau"].append(np.abs(tau_leg).max())

        mujoco.mj_step(m, d)
        iteration += 1
        if not np.all(np.isfinite(d.qpos)) or abs(float(d.qpos[2]) - base_z0) > 0.15:
            fell = True
            break
    wall = time.perf_counter() - t_wall

    z = np.array(log["z"])
    roll = np.array(log["roll"])
    pitch = np.array(log["pitch"])
    err = np.array(log["err"])
    fz = np.array(log["fz"])
    solve_ms = np.array([i["total_ms"] for i in infos])
    osqp_ms = np.array([i["solve_ms"] for i in infos])
    n_fb = sum(i["fallback"] for i in infos)
    n_cold = sum(i.get("cold", False) for i in infos)
    iters = np.array([i["iters"] for i in infos])
    x0_ref = np.array([np.abs(i["x0"][:12] - i["xref0"][:12]) for i in infos])
    res = dict(
        fell=fell, sim_t=float(d.time), wall=wall, ticks=len(z), n_mpc=len(infos),
        dz_final=float(z[-1] - base_z0), dz_maxabs=float(np.abs(z - base_z0).max()),
        roll_maxabs=float(np.abs(roll).max()), pitch_maxabs=float(np.abs(pitch).max()),
        roll_final=float(roll[-1]), pitch_final=float(pitch[-1]),
        yaw_final=float(log["yaw"][-1]),
        com_err_final=err[-1, 3:6], com_err_max=np.abs(err[:, 3:6]).max(0),
        x0_ref_max=x0_ref.max(0), x0_ref_last=x0_ref[-1],
        qp_median_ms=float(np.median(solve_ms)), osqp_median_ms=float(np.median(osqp_ms)),
        qp_p90_ms=float(np.percentile(solve_ms, 90)), qp_iters_median=float(np.median(iters)),
        fallbacks=int(n_fb), colds=int(n_cold), all_stance=all(i["stance_all"] for i in infos),
        fz_foot_last=fz[-1], fz_sum_mean_last_half=float(fz[len(fz) // 2:].sum(1).mean()),
        u_last=u_hold, tau_max=float(np.max(log["tau"])),
        mg_total=float(model.total_mass() * 9.81), mg_reduced=float(state.reduced.mass * 9.81),
    )
    if verbose:
        np.set_printoptions(precision=4, suppress=True, linewidth=140)
        print(f"[smoke] standing {seconds:.2f} s  solver {solver}  armature {armature}  "
              f"ticks {res['ticks']}  MPC solves {res['n_mpc']} (cold {res['colds']}, fallback {res['fallbacks']})  "
              f"wall {wall:.1f} s")
        print(f"  base z drift: final {res['dz_final']*1e3:+.2f} mm, max |dz| {res['dz_maxabs']*1e3:.2f} mm")
        print(f"  roll : final {res['roll_final']:+.5f} rad, max |roll| {res['roll_maxabs']:.5f}")
        print(f"  pitch: final {res['pitch_final']:+.5f} rad, max |pitch| {res['pitch_maxabs']:.5f}")
        print(f"  yaw  : final {res['yaw_final']:+.5f} rad")
        print(f"  COM - target (x,y,z) final {res['com_err_final']*1e3} mm, max |.| {res['com_err_max']*1e3} mm")
        print(f"  max |x0 - X_ref[0]| per state [r p y px py pz wx wy wz vx vy vz]:\n    {res['x0_ref_max']}")
        print(f"  last |x0 - X_ref[0]|:\n    {res['x0_ref_last']}")
        print(f"  QP: median total {res['qp_median_ms']:.2f} ms (OSQP {res['osqp_median_ms']:.2f} ms, "
              f"p90 {res['qp_p90_ms']:.2f}), median iters {res['qp_iters_median']:.0f}, failures {res['fallbacks']}")
        print(f"  horizon all stance: {res['all_stance']}")
        print(f"  u0 last: F_L {u_hold[0:3]}  F_R {u_hold[3:6]}  M_L {u_hold[6:9]}  M_R {u_hold[9:12]}")
        print(f"  MPC Fz sum {u_hold[2] + u_hold[5]:.2f} N vs reduced m g {res['mg_reduced']:.2f} N; "
              f"measured foot normal force (2nd half mean) {res['fz_sum_mean_last_half']:.2f} N "
              f"vs total m g {res['mg_total']:.2f} N; max |tau_leg| {res['tau_max']:.1f} N m")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--solver", default="osqp", choices=["osqp", "quadprog"])
    ap.add_argument("--noarmature", action="store_true")
    a = ap.parse_args()
    r = run(a.seconds, a.solver, not a.noarmature)
    ok = (not r["fell"] and r["dz_maxabs"] < 0.03 and r["roll_maxabs"] < 0.05 and r["pitch_maxabs"] < 0.05
          and r["fallbacks"] == 0 and r["all_stance"])
    print(("PASS" if ok else "FAIL") + f"  standing {a.seconds:.1f} s holds  (max|dz| {r['dz_maxabs']*1e3:.2f} mm, "
          f"max|roll| {r['roll_maxabs']:.4f}, max|pitch| {r['pitch_maxabs']:.4f}, fallbacks {r['fallbacks']})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
