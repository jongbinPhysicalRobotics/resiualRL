"""걸으며 돌기 MPC wrench 기록. 사용: wt_force.py vx wz seconds variant out.npz"""
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
        psi = float(ctl.body.euler_W[2]); c, sn = np.cos(psi), np.sin(psi)
        R = np.array([[c, sn, 0], [-sn, c, 0], [0, 0, 1]])        # 월드 → 목표 yaw (heading) 기준
        u = np.asarray(ctl.u_hold, dtype=float)
        FL, FR, ML, MR = R @ u[0:3], R @ u[3:6], R @ u[6:9], R @ u[9:12]
        rel = (s.foot_pos_W - ctl.x0[3:6]) @ R.T
        v = R @ ctl.x0[9:12]
        info = ctl.last_info or {}
        rows.append([s.t, float(ctl.mode == "walking"), s.yaw_unwrapped - psi, s.roll, s.pitch, *v,
                     float(s.torso_angvel_W[2]), *FL, *FR, *ML, *MR, *rel[0], *rel[1],
                     *[float(ctl._active(l, s.t)) for l in (0, 1)], float(info.get("prim_res", np.nan)),
                     float(ctl.x0[2])])
        model.write_torque(tl, ta)
        mujoco.mj_step(m, d)
        if W._fell(d):
            fell = d.time; break
    np.savez(out, rows=np.array(rows), fell=fell)
    print(out, "fell", fell)


if __name__ == "__main__":
    run(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5])
