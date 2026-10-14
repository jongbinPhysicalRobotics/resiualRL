"""디딤 발이 전신 CoM 둘레 수직축 모멘트를 어떻게 만드나 (Q&A 10/6 Q74). 디딤 앞 · 뒤 반, 발마다:
  발 접촉력 (지면 → 발) 의 앞뒤 Fx · 옆 Fy, 발 site 의 CoM 기준 앞뒤 x · 옆 y (몸 yaw 좌표), −y·Fx 와 x·Fy (Mz = x·Fy − y·Fx).
사용: foot_yaw_moment.py vx variant
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
rows = []
for _ in range(int(round((walk_t + 8.0) / m.opt.timestep))):
    tl, ta = ctl.tick(); model.write_torque(tl, ta); mujoco.mj_step(m, d)
    if W._fell(d): break
    s = ctl.state
    if s.t < walk_t + 3.0: continue
    c = d.subtree_com[0]; yaw = s.yaw_unwrapped; hv = np.array([np.cos(yaw), np.sin(yaw), 0]); lv = np.array([-np.sin(yaw), np.cos(yaw), 0])
    F = np.asarray(s.foot_force_W)
    for leg in (0, 1):
        if not ctl._active(leg, s.t) or F[leg][2] < 20: continue
        ph = ctl.gait.phase(leg, s.t, ctl.clock) * ctl.clock.cycle / ctl.gait.stance
        r = d.site_xpos[model.foot_site[leg]] - c
        fx, fy, x, y = F[leg] @ hv, F[leg] @ lv, r @ hv, r @ lv
        rows.append([leg, ph, fx, y, -y * fx, x * fy, x, fy, F[leg][2]])
A = np.array(rows)
print(f"[vx {vx}] 디딤 발 (전신 CoM 기준) 옆 거리 y: 왼 {1e3*np.median(A[A[:,0]==0,3]):+.0f} · 오른 {1e3*np.median(A[A[:,0]==1,3]):+.0f} mm")
for lo, hi, nm in ((0, 0.5, "디딤 앞 반 (착지 쪽)"), (0.5, 1.0, "디딤 뒤 반 (이륙 쪽)")):
    for leg in (0, 1):
        k = (A[:, 0] == leg) & (A[:, 1] >= lo) & (A[:, 1] < hi)
        print(f"   {nm} {'왼' if leg == 0 else '오른'}발: Fx {A[k,2].mean():+4.0f} N · y {1e3*A[k,3].mean():+4.0f} mm → −y·Fx {A[k,4].mean():+.2f} | Fy {A[k,7].mean():+4.0f} N · x {1e3*A[k,6].mean():+4.0f} mm → x·Fy {A[k,5].mean():+.2f} N·m | Fz {A[k,8].mean():4.0f} N")
