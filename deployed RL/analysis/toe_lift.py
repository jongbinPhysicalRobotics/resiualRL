"""배포 RL 정책의 착지 직후 발가락/뒤꿈치 하중·발 pitch·발가락 높이·실현 CoP (Q&A 9/21 Q9).

MPC 의 MPC/src/baseline/14_toe_lift.py 와 같은 양을 잰다. 착지 = 40 ms 이상 공중 뒤 첫 접촉.

사용:
  .venv/Scripts/python.exe "deployed RL/analysis/toe_lift.py" --vx 1.0 --seconds 120
"""
import sys, os, json, argparse
from pathlib import Path
import numpy as np, mujoco, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_g1_policy as R

ap = argparse.ArgumentParser()
ap.add_argument("--vx", type=float, default=0.5)
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--settle", type=float, default=6.0)
a = ap.parse_args()

m, d, policy, c = R.load()
nA, nO = c["num_actions"], c["num_obs"]
kps = np.array(c["kps"], np.float32); kds = np.array(c["kds"], np.float32)
dflt = np.array(c["default_angles"], np.float32)
cmd = np.array([a.vx, 0, 0], np.float32); cmd_scale = np.array(c["cmd_scale"], np.float32)
dt = c["simulation_dt"]; DEC = c["control_decimation"]
action = np.zeros(nA, np.float32); target = dflt.copy(); obs = np.zeros(nO, np.float32)
mujoco.mj_forward(m, d)
fb = R.foot_bodies(m); PEL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
R0 = d.xmat[PEL].reshape(3, 3); YAW_REF = np.arctan2(R0[1, 0], R0[0, 0])
toe, heel = [set(), set()], [set(), set()]
for i, b in enumerate(fb):
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            (toe if m.geom_pos[g][0] > 0 else heel)[i].add(g)
rad = m.geom_size[list(toe[0])[0]][0]
rec = [[], []]; f6 = np.zeros(6); counter = 0; fell = None; vs = []
for k in range(int(a.seconds / dt)):
    t = k * dt
    d.ctrl[:] = (target - d.qpos[7:]) * kps + (0.0 - d.qvel[6:]) * kds
    mujoco.mj_step(m, d); counter += 1
    if counter % DEC == 0:
        Rp = d.xmat[PEL].reshape(3, 3)
        e = (np.arctan2(Rp[1, 0], Rp[0, 0]) - YAW_REF + np.pi) % (2 * np.pi) - np.pi
        cmd[2] = float(np.clip(-e, -0.5, 0.5))
        ph = (counter * dt) % 0.8 / 0.8
        obs[:3] = d.qvel[3:6] * c["ang_vel_scale"]; obs[3:6] = R.gravity_orientation(d.qpos[3:7])
        obs[6:9] = cmd * cmd_scale
        obs[9:9+nA] = (d.qpos[7:] - dflt) * c["dof_pos_scale"]; obs[9+nA:9+2*nA] = d.qvel[6:] * c["dof_vel_scale"]
        obs[9+2*nA:9+3*nA] = action; obs[9+3*nA:9+3*nA+2] = [np.sin(2*np.pi*ph), np.cos(2*np.pi*ph)]
        action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
        target = action * c["action_scale"] + dflt
    if t > a.settle:
        mujoco.mj_subtreeVel(m, d)
        Rp = d.xmat[PEL].reshape(3, 3); yw = np.arctan2(Rp[1, 0], Rp[0, 0])
        vs.append(d.subtree_linvel[0][0] * np.cos(yw) + d.subtree_linvel[0][1] * np.sin(yw))
    for i in range(2):
        ft = fh = 0.0
        for ci in range(d.ncon):
            con = d.contact[ci]
            for g in (con.geom1, con.geom2):
                if g in toe[i] or g in heel[i]:
                    mujoco.mj_contactForce(m, d, ci, f6)
                    fz = abs((con.frame.reshape(3, 3).T @ f6[:3])[2])
                    if g in toe[i]: ft += fz
                    else: fh += fz
        Rf = d.xmat[fb[i]].reshape(3, 3)
        _, cp = R.foot_wrench(m, d, fb[i])
        rec[i].append((t, ft, fh, np.degrees(-np.arcsin(np.clip(Rf[2, 0], -1, 1))),
                       min(d.geom_xpos[g][2] for g in toe[i]) - rad,
                       min(d.geom_xpos[g][2] for g in heel[i]) - rad,
                       cp[0] if cp is not None else np.nan))
    if d.qpos[2] < 0.5:
        fell = t; break

grid = np.arange(-0.02, 0.40, 0.004)
Pall = []; steps = []
for i in range(2):
    A = np.array(rec[i]); T = A[:, 0]; tot = A[:, 1] + A[:, 2]
    air = tot < 1.0
    # 착지 = 40 ms 이상 공중 뒤 첫 접촉
    k = 0
    while k < len(T):
        if T[k] > a.settle and not air[k] and k >= 20 and air[k-20:k].all():
            k_end = k
            while k_end < len(T) and not (air[k_end:k_end+10].all() if k_end + 10 <= len(T) else True):
                k_end += 1
            if k_end >= len(T) - 1: break
            idx = np.clip(np.searchsorted(T, T[k] + grid), 0, len(T) - 1)
            seg = A[idx, 1:].copy(); seg[T[idx] > T[k_end], :] = np.nan
            Pall.append(seg)
            k30 = k + 15
            s = A[k30:k_end]
            steps.append(dict(toe0=1000*dt*np.sum((s[:, 1] < 5) & (s[:, 2] > 50)),
                              heel0=1000*dt*np.sum((s[:, 2] < 5) & (s[:, 1] > 50)),
                              lift=1000*np.max(s[:, 4]) if len(s) else np.nan,
                              pitch_td=A[k, 3], pitch_min=np.min(A[k:k_end, 3]),
                              first=("heel" if A[k, 2] > 0 and A[k, 1] == 0 else "toe" if A[k, 1] > 0 and A[k, 2] == 0 else "flat"),
                              stance=1000*(T[k_end]-T[k])))
            k = k_end
        k += 1
P = np.array(Pall)
print(f"RL vx {a.vx}: {'OK' if fell is None else f'fell {fell:.1f}'}  실측 {np.mean(vs):.3f} m/s, 착지 {len(steps)}회", flush=True)
print(f"{'대비':>7s} | {'발가락Fz':>7s} {'뒤꿈치Fz':>7s} | {'발 pitch':>8s} {'발가락높이':>8s} {'뒤꿈치높이':>8s} | {'실현 CoP x':>9s}")
for j, g in enumerate(grid):
    ms = round(g * 1000)
    if ms % 24 != 0 and ms not in (4, 8, 12, 16, 20, 40, 56, 64, 72):
        continue
    r = np.nanmean(P[:, j, :], axis=0)
    print(f"{ms:+6d}ms | {r[0]:6.0f}N {r[1]:6.0f}N | {r[2]:+7.2f}° {r[3]*1000:7.1f}mm {r[4]*1000:7.1f}mm | {r[5]*100:+7.1f}cm")
S = steps
print(f"\n착지 첫 접촉: " + ", ".join(f"{f} {100*np.mean([s['first']==f for s in S]):.0f}%" for f in ("heel", "flat", "toe")))
for key in ("toe0", "heel0", "lift", "pitch_td", "pitch_min", "stance"):
    print(f"  {key:10s} 평균 {np.nanmean([s[key] for s in S]):7.2f}   최대 {np.nanmax([s[key] for s in S]):7.2f}")
