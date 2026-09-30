"""옆 밀기 뒤 회복 과정 기록. 사용: push_diag.py vx Fx Fy t0 dur seconds variant out.npz"""
import sys, importlib.util
import numpy as np
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent          # casadi_mit
sys.path.insert(0, str(_ROOT))
import mujoco
spec = importlib.util.spec_from_file_location("walk", str(_ROOT / "10_walk.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)


def run(vx, Fx, Fy, t0, dur, seconds, var, out):
    kw = W.parse_variant(var); prof = kw.pop("profile", "ramp"); kw.pop("push", None)
    cfg, model, ctl, walk_t = W._setup(vx, 0.0, 0.0, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
    m, d = model.m, model.d
    rows = []
    n = int(round((walk_t + seconds) / m.opt.timestep))
    fell = -1.0
    for _ in range(n):
        tl, ta = ctl.tick()
        s = ctl.state
        psi = float(ctl.body.euler_W[2]); c, sn = np.cos(psi), np.sin(psi)
        v = ctl.x0[9:11]
        rel = s.foot_pos_W - ctl.x0[3:6]
        tgt = ctl.planner.touchdown_targets - ctl.x0[3:6]
        u = np.asarray(ctl.u_hold, dtype=float)
        rows.append([s.t - walk_t, s.roll, s.pitch, c * v[0] + sn * v[1], -sn * v[0] + c * v[1],
                     *[-sn * rel[l, 0] + c * rel[l, 1] for l in (0, 1)], *[-sn * tgt[l, 0] + c * tgt[l, 1] for l in (0, 1)],
                     *[float(ctl._active(l, s.t)) for l in (0, 1)], -sn * u[0] + c * u[1], -sn * u[3] + c * u[4], u[2], u[5],
                     u[6] * c + u[7] * sn, u[9] * c + u[10] * sn, ctl.lat_err])
        model.write_torque(tl, ta)
        tr = d.time - walk_t - t0
        if 0.0 <= tr < dur:
            d.xfrc_applied[model.base_body, 0:3] = (c * Fx - sn * Fy, sn * Fx + c * Fy, 0.0)
        else:
            d.xfrc_applied[model.base_body, :] = 0.0
        mujoco.mj_step(m, d)
        if W._fell(d):
            fell = d.time - walk_t; break
    np.savez(out, rows=np.array(rows), fell=fell)
    print(out, "fell", fell)


if __name__ == "__main__":
    a = sys.argv
    run(float(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[5]), float(a[6]), a[7], a[8])
