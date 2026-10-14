"""디딤 구간 (두 점 / 발끝만) 별 발 yaw 각속도 vs 몸통 yaw 각속도 — 상관 · 기울기, hip yaw 관절 속도 (Q&A 10/6 Q71).
사용: toe_follow.py vx variant
"""
import sys, importlib.util
import numpy as np
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/diag")
import mujoco
spec = importlib.util.spec_from_file_location("walk", ROOT + "/10_walk.py")
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
from configs import resolve
vx, var = float(sys.argv[1]), sys.argv[2]
kw = W.parse_variant(resolve(var)); prof = kw.pop("profile", "ramp")
cfg, model, ctl, walk_t = W._setup(vx, 0, 0, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
m, d = model.m, model.d
f6 = np.zeros(6); w6 = np.zeros(6); rows = []
for _ in range(int(round((walk_t + 8.0) / m.opt.timestep))):
    tl, ta = ctl.tick(); model.write_torque(tl, ta); mujoco.mj_step(m, d)
    if W._fell(d): break
    if d.time < walk_t + 3.0: continue
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, model.base_body, w6, 0); wt = w6[2]
    for leg in (0, 1):
        fb = model.foot_body[leg]; R = d.xmat[fb].reshape(3, 3); p = d.xpos[fb]
        fh = ft = 0.0
        for c in range(d.ncon):
            con = d.contact[c]
            if model._geom_foot[con.geom1] != leg and model._geom_foot[con.geom2] != leg: continue
            mujoco.mj_contactForce(m, d, c, f6)
            if (R.T @ (con.pos - p))[0] < 0.03: fh += f6[0]
            else: ft += f6[0]
        ph = 1 if (ft > 5 and fh < 1) else (0 if (ft > 5 and fh > 5) else -1)
        if ph < 0: continue
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, fb, w6, 0)
        rows.append([ph, w6[2], wt, d.qvel[model.leg_dof[leg][0]]])
A = np.array(rows)
print(f"[vx {vx} {var}]")
for ph, nm in ((0, "두 점"), (1, "발끝만")):
    s = A[:, 0] == ph; wf, wt, qy = A[s, 1], A[s, 2], A[s, 3]
    k = np.polyfit(wt, wf, 1)[0]
    print(f"   {nm:4s}: 발 ω_z |·| {np.degrees(np.mean(abs(wf))):5.1f} °/s · 몸통 ω_z |·| {np.degrees(np.mean(abs(wt))):5.1f} °/s · hip yaw 관절 속도 |·| {np.degrees(np.mean(abs(qy))):5.1f} °/s"
          f" | 발 ~ 몸통 상관 {np.corrcoef(wf, wt)[0,1]:+.2f}, 기울기 {k:+.2f}")
