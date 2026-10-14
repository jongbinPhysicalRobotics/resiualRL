"""팔 관절 토크 / 한계 (최대 · p95 · 포화 틱 비율, 관절별 p95) 와 몸통 ω_z 표준편차 — 12 s headless, 걷기 + 4 s 부터 (Q&A 10/6 Q75).
사용: arm_torque.py vx variant
"""
import sys, importlib.util
import numpy as np
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/diag")
spec = importlib.util.spec_from_file_location("walk", ROOT + "/10_walk.py")
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
from configs import resolve
vx, var = float(sys.argv[1]), sys.argv[2]
kw = W.parse_variant(resolve(var))
r = W.headless(vx, 0, 0, 12.0, verbose=False, keep_record=True, **kw)
R = r["record"]; t = R["t"]; w = (R["walking"] > 0.5)
wt = t[np.flatnonzero(w)[0]]; k = t > wt + 4.0
cfg, model, ctl, walk_t = W._setup(0, 0, 0, "osqp", True, "ramp", False, False, 1.0, True, False)
lim = model.ctrl_hi[model.arm_act.ravel()]
ratio = np.abs(R["tau_arm"][k]) / lim
names = ["어깨pitch", "어깨abad", "어깨yaw", "팔꿈치"]
print(f"[vx {vx} {var.split(',')[-1]}] {'✓' if r['pass'] else '✗ %.1f s' % r['survived']}  팔 토크/한계 최대 {ratio.max():.2f} · p95 {np.percentile(ratio, 95):.2f} · 포화 틱 {100*(ratio >= 1).any(axis=1).mean():.1f} %"
      f" | 관절별 p95: " + " ".join(f"{names[j]} {np.percentile(ratio[:, [j, j + 4]], 95):.2f}" for j in range(4))
      + f" | 몸통 ω_z 표준편차 {np.degrees(np.std(R['x0'][k, 8])):.1f} °/s")
