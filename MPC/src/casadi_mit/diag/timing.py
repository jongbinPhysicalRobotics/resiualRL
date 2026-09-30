"""풀이 시간 단독 측정: timing.py vx wz seconds variant [variant ...]"""
import sys, time, importlib.util
import numpy as np
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent          # casadi_mit
sys.path.insert(0, str(_ROOT))
import mujoco
spec = importlib.util.spec_from_file_location("walk", str(_ROOT / "10_walk.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
vx, wz, sec = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
for var in sys.argv[4:]:
    kw = W.parse_variant(var); prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(vx, 0.0, wz, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
    m, d = model.m, model.d
    n = int(round((walk_t + sec) / m.opt.timestep))
    last, rec = None, []
    t0 = time.perf_counter()
    for _ in range(n):
        tl, ta = ctl.tick()
        inf = ctl.last_info
        if inf is not None and inf.get("iteration") != last:
            last = inf.get("iteration")
            if d.time > walk_t + 2.0:
                rec.append([inf.get("build_ms", np.nan), inf.get("solve_ms", np.nan), inf.get("total_ms", np.nan), inf.get("iters", 0)])
        model.write_torque(tl, ta); mujoco.mj_step(m, d)
        if W._fell(d):
            break
    wall = time.perf_counter() - t0
    r = np.array(rec)
    tag = ",".join(k for k in var.split(",") if k.split("=")[0] in ("admm", "nbs", "rho", "hor"))
    print(f"{tag:<32s} 풀이 {len(r)/max(d.time-walk_t-2,1e-9):5.1f} 회/s  build {np.nanmedian(r[:,0]):5.2f} ms  solve {np.nanmedian(r[:,1]):5.2f} ms (p99 {np.nanpercentile(r[:,1],99):5.2f})  total {np.nanmedian(r[:,2]):5.2f} ms  iters {np.median(r[:,3]):.0f}/{r[:,3].max():.0f}  MPC 부하 {np.nansum(r[:,2])/1e3/max(d.time-walk_t-2,1e-9)*100:5.1f} %  fell {W._fell(d)}")
