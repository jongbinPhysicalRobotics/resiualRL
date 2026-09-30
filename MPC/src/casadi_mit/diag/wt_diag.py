"""걸으며 돌기 전도 과정 기록. 사용: wt_diag.py vx wz seconds variant out.npz"""
import sys, importlib.util
import numpy as np
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent          # casadi_mit
sys.path.insert(0, str(_ROOT))
import mujoco
spec = importlib.util.spec_from_file_location("walk", str(_ROOT / "10_walk.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)


def run(vx, wz, seconds, var, out):
    kw = W.parse_variant(var); prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(vx, 0.0, wz, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
    m, d = model.m, model.d
    rows = []
    n = int(round((walk_t + seconds) / m.opt.timestep))
    fell = -1.0
    for _ in range(n):
        tl, ta = ctl.tick()
        s = ctl.state
        psi = s.yaw_unwrapped; c, sn = np.cos(psi), np.sin(psi)
        v = ctl.x0[9:11]
        rel = s.foot_pos_W - ctl.x0[3:6]
        tgt = ctl.planner.touchdown_targets - ctl.x0[3:6]
        f = ctl.filtered
        rows.append([s.t, float(ctl.mode == "walking"), psi, float(ctl.body.euler_W[2]), s.roll, s.pitch,
                     c * v[0] + sn * v[1], -sn * v[0] + c * v[1], float(s.torso_angvel_W[2]),
                     *[c * rel[l, 0] + sn * rel[l, 1] for l in (0, 1)], *[-sn * rel[l, 0] + c * rel[l, 1] for l in (0, 1)],
                     *[-sn * tgt[l, 0] + c * tgt[l, 1] for l in (0, 1)],
                     *[float(ctl._active(l, s.t)) for l in (0, 1)], float(f.x_dot), float(f.psi_dot), ctl.lat_err])
        model.write_torque(tl, ta)
        mujoco.mj_step(m, d)
        if W._fell(d):
            fell = d.time; break
    np.savez(out, rows=np.array(rows), fell=fell)
    print(out, "fell", fell)


if __name__ == "__main__":
    run(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5])
