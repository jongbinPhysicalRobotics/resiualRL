"""디딤 중 발 heading 변화가 언제 쌓이나 (Q&A 10/6 Q65). 걷기 + 3 s 부터 5 s, 틱마다 |Δheading| 을
(두 점 / 발끝만 (닿은 발끝 접촉 수) / 뒤꿈치만) × (마찰 사용 ≥ 0.95) 칸으로 나눠 합한다.
사용: toe_when.py vx variant [noslip_iterations]
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
ns = int(sys.argv[3]) if len(sys.argv) > 3 else 0
kw = W.parse_variant(resolve(var)); prof = kw.pop("profile", "ramp")
cfg, model, ctl, walk_t = W._setup(vx, 0, 0, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
m, d = model.m, model.d
m.opt.noslip_iterations = ns
f6 = np.zeros(6); prev = {0: None, 1: None}; acc = {}; cnt = {}; nst = 0
for _ in range(int(round((walk_t + 8.0) / m.opt.timestep))):
    tl, ta = ctl.tick(); model.write_torque(tl, ta); mujoco.mj_step(m, d)
    if W._fell(d): break
    if d.time < walk_t + 3.0: continue
    for leg in (0, 1):
        fb = model.foot_body[leg]; R = d.xmat[fb].reshape(3, 3); p = d.xpos[fb]
        fh = ft = 0.0; use = 0.0; ntoe = 0
        for c in range(d.ncon):
            con = d.contact[c]
            if model._geom_foot[con.geom1] != leg and model._geom_foot[con.geom2] != leg: continue
            mujoco.mj_contactForce(m, d, c, f6)
            if f6[0] <= 0: continue
            if (R.T @ (con.pos - p))[0] < 0.03: fh += f6[0]
            else:
                ft += f6[0]; use = max(use, np.linalg.norm(f6[1:3]) / (con.friction[0] * f6[0]))
                ntoe += f6[0] > 5
        head = np.arctan2(R[1, 0], R[0, 0])
        if prev[leg] is not None and (fh > 1 or ft > 1):
            dh = abs(np.degrees((head - prev[leg] + np.pi) % (2 * np.pi) - np.pi))
            ph = "두 점" if (fh > 5 and ft > 5) else ("발끝만" if ft > 5 else ("뒤꿈치만" if fh > 5 else "약함"))
            key = (ph, f"발끝 접촉 {ntoe}" if ph == "발끝만" else "", "마찰포화" if use >= 0.95 else "")
            acc[key] = acc.get(key, 0.0) + dh; cnt[key] = cnt.get(key, 0) + 1
        prev[leg] = head
tot = sum(acc.values())
print(f"[vx {vx} {var.split(',')[-1]} noslip {ns}] 디딤 중 heading |Δ| 합 {tot:.0f}° (8 s 중 5 s, 두 발). 칸별 몫 (틱 비율):")
for k in sorted(acc, key=lambda k: -acc[k])[:7]:
    print(f"   {' · '.join(x for x in k if x):22s} {100*acc[k]/tot:5.1f} %  (틱 {100*cnt[k]/sum(cnt.values()):4.1f} %)")
