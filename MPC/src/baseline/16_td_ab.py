"""착지 거리 A/B — 발을 CoM 에 더 가깝게 딛으면 발 들림·속도 벽이 풀리나? (Q&A 9/21 Q11)

변형: td_scale (전진 중립항 배율), td_dx (착지 기준점 x 이동, −3.5 cm = 발 중앙 기준).
계측 (settle 이후):
  속도·추종·pitch·roll σ, 착지 거리 rx (발 site − CoM, 몸 yaw frame),
  발가락 들림 mm·발가락 하중 0 ms·명령 CoP 뒤꿈치 포화 %,
  토크 한계 초과 — 명령 토크(클리핑 전)/한계, stance 다리와 swing 다리 분리 (9/17 Q5 재측정).

사용:
  .venv/Scripts/python.exe MPC/src/baseline/16_td_ab.py --vx 0.6 0.7 --var 1:0 0.75:0 0.5:0 1:-3.5 --seconds 40
  (--var 은 td_scale:td_dx[cm])
"""
from __future__ import annotations
import sys, argparse, importlib, itertools
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_model, mpc_srb
walk = importlib.import_module("09_walk")


def run(vx, td_scale, td_dx, seconds=40.0, settle=8.0, side_w=0.13, ramp_end=None):
    if ramp_end is not None:                  # v_cmd 램프 끝 시각 (기본 3.0 s = 1 s 동안 가속)
        walk.RAMP_T1 = ramp_end
        settle = max(settle, ramp_end + 5.0)
    m, d = g1_model.load_torque(); g1_model.set_crouch(m, d)
    ctl = walk.WalkController(m, d, vx_cmd=vx, kp_up=300.0, swing_id=True, soft_land=True,
                              lam_swing=True, wn_swing=30.0, zeta_swing=0.7, stance_frac=0.57,
                              q_py=300, gate_sy=False, side_w=side_w,
                              td_scale=td_scale, td_dx=td_dx)
    P = ctl.params; h, lh = P.h_sole, P.l_h
    Tst = ctl.gait.T_stance
    fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link") for s in ("left", "right")]
    toe, heel = [set(), set()], [set(), set()]
    for i, b in enumerate(fb):
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                (toe if m.geom_pos[g][0] > 0 else heel)[i].add(g)
    rad = m.geom_size[next(iter(toe[0]))][0]
    sids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in mpc_srb.FOOT_SITES]
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    leg_act = [[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{s}_{j}_joint")
                for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
               for s in ("left", "right")]
    lim = [m.actuator_ctrlrange[a, 1] for a in leg_act]
    dt = m.opt.timestep
    vs, pit, rol, rx, xc_heel = [], [], [], [], []
    tq = {"st": [], "sw": []}                     # 샘플별 다리 최대 |τ|/한계
    tq_joint = {"st": np.zeros(6), "sw": np.zeros(6)}
    zt_hist = [[], []]; toe0 = [[], []]; tds = [[], []]; prev = [True, True]; f6 = np.zeros(6); fell = None
    for k in range(int(seconds / dt)):
        t = k * dt
        mujoco.mj_forward(m, d)
        if k % walk.DECIM == 0:
            ctl.update_mpc(d, t)
        tau = ctl.torque(d, t)
        on = t > settle
        psi = ctl._yaw_unwrap; c, s = np.cos(psi), np.sin(psi)
        if on:
            x0 = ctl.get_state(d, t); Rp = d.xmat[pel].reshape(3, 3)
            vs.append(c * x0[9] + s * x0[10])
            pit.append(-np.arcsin(np.clip(Rp[2, 0], -1, 1))); rol.append(np.arctan2(Rp[2, 1], Rp[2, 2]))
        for i in range(2):
            st = ctl.gait.in_stance(t, i)
            if st and not prev[i] and on:
                tds[i].append(len(zt_hist[i]))
                r = d.site_xpos[sids[i]] - d.subtree_com[0]; rx.append(c * r[0] + s * r[1])
            prev[i] = st
            if not on:
                continue
            ratio = np.abs(tau[leg_act[i]]) / np.array(lim[i])
            key = "st" if st else "sw"
            tq[key].append(ratio.max()); tq_joint[key] = np.maximum(tq_joint[key], ratio)
            ft = fh = 0.0
            for ci in range(d.ncon):
                con = d.contact[ci]
                for g in (con.geom1, con.geom2):
                    if g in toe[i] or g in heel[i]:
                        mujoco.mj_contactForce(m, d, ci, f6)
                        fz = abs((con.frame.reshape(3, 3).T @ f6[:3])[2])
                        if g in toe[i]: ft += fz
                        else: fh += fz
            zt_hist[i].append((min(d.geom_xpos[g][2] for g in toe[i]) - rad, ft, fh))
            if st:
                Fx, Fy, Fz, mx, my, mz = ctl.wr[i]
                if Fz > 20:
                    xc = -((-s * mx + c * my) + h * (c * Fx + s * Fy)) / Fz
                    xc_heel.append(xc <= -0.95 * lh)
        d.ctrl[:] = tau; mujoco.mj_step(m, d)
        if d.qpos[2] < 0.5:
            fell = t; break
    lift, t0 = [], []
    n30, nst = int(0.03 / dt), int(Tst / dt)
    for i in range(2):
        Z = np.array(zt_hist[i]) if zt_hist[i] else np.zeros((0, 3))
        for k0 in tds[i]:
            seg = Z[k0 + n30:k0 + nst]
            if len(seg) < nst - n30:
                continue
            lift.append(1000 * seg[:, 0].max())
            t0.append(1000 * dt * np.sum((seg[:, 1] < 5) & (seg[:, 2] > 50)))
    f = lambda a, fn=np.mean: float(fn(a)) if len(a) else float("nan")
    return dict(vx=vx, s=td_scale, dx=td_dx, fell=fell, v=f(vs), pitch=np.degrees(f(pit)),
                roll=np.degrees(f(rol, np.std)), rx=100 * f(rx), lift=f(lift), toe0=f(t0),
                heel=100 * f(xc_heel),
                st_over=100 * f(np.array(tq["st"]) > 1.0), st_max=f(tq["st"], np.max),
                sw_over=100 * f(np.array(tq["sw"]) > 1.0), sw_max=f(tq["sw"], np.max),
                st_joint=tq_joint["st"].tolist(), sw_joint=tq_joint["sw"].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, nargs="+", default=[0.6, 0.7])
    ap.add_argument("--var", nargs="+", default=["1:0", "0.75:0", "0.5:0", "1:-3.5"])
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--ramp", type=float, default=None, help="v_cmd 램프 끝 시각 [s] (기본 3.0)")
    a = ap.parse_args()
    var = [(float(x.split(":")[0]), float(x.split(":")[1]) / 100.0) for x in a.var]
    jobs = [(vx, sc, dx, a.seconds, 8.0, 0.13, a.ramp) for vx, (sc, dx) in itertools.product(a.vx, var)]
    with Pool(min(a.procs, len(jobs))) as pool:
        res = pool.starmap(run, jobs)
    print(f"착지 거리 A/B — {a.seconds:.0f} s, 권장 구성 + 보폭 13 cm\n")
    print(f"{'vx':>4s} {'배율':>4s} {'dx':>5s} | {'결과':>6s} {'실측':>6s} {'추종':>4s} {'pitch':>6s} {'rollσ':>5s} | "
          f"{'착지rx':>6s} | {'들림':>6s} {'발가락0':>6s} {'뒤꿈치포화':>6s} | {'stance 초과':>9s} {'최대':>5s} | {'swing 초과':>9s} {'최대':>5s}")
    for r in res:
        res_s = f"{a.seconds:.0f}s OK" if r["fell"] is None else f"{r['fell']:.1f}s✗"
        print(f"{r['vx']:4.1f} {r['s']:4.2f} {100*r['dx']:+5.1f} | {res_s:>6s} {r['v']:6.3f} {100*r['v']/r['vx']:3.0f}% "
              f"{r['pitch']:+5.1f}° {r['roll']:4.2f}° | {r['rx']:5.1f}cm | {r['lift']:5.1f}mm {r['toe0']:4.0f}ms {r['heel']:5.1f}% | "
              f"{r['st_over']:8.2f}% {r['st_max']:4.2f}x | {r['sw_over']:8.2f}% {r['sw_max']:4.2f}x", flush=True)
    print("\n관절별 최대 |τ|/한계 (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)")
    for r in res:
        print(f"{r['vx']:4.1f} {r['s']:4.2f} {100*r['dx']:+5.1f} | stance " + " ".join(f"{x:4.2f}" for x in r["st_joint"])
              + " | swing " + " ".join(f"{x:4.2f}" for x in r["sw_joint"]))


if __name__ == "__main__":
    main()
