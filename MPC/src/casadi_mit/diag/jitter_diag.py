"""딛는 발 떨림 진단 (9/30 Q12) — 발끝 · 뒤꿈치 접촉점의 미끄럼 속도, 접촉점 켜짐/꺼짐, 발 각속도, MPC 명령 · 발목 토크를 매 틱 기록.

사용: jitter_diag.py vx vy wz seconds variant out.npz
열 (다리마다 21 개, 앞에 t, walk):
  act, n_con, fz_뒤, fz_앞, 발 site xy (2), 발 각속도 월드 (3), 뒤 · 앞 접촉점 미끄럼 속도 xy (4, 그 점에 힘이 없으면 nan),
  MPC Fy · My · Mz (발 yaw 좌표, 3), 다리 관절 토크 (5)  → 21
시뮬 순서는 10_walk.headless 와 같다 (기록만).
"""
import sys, importlib.util
import numpy as np
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent          # casadi_mit
sys.path.insert(0, str(_ROOT))
import mujoco
spec = importlib.util.spec_from_file_location("walk", str(_ROOT / "10_walk.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)


def run(vx, vy, wz, seconds, var, out):
    sys.path.insert(0, str(_ROOT / "diag")); from configs import resolve
    kw = W.parse_variant(resolve(var)); prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(vx, vy, wz, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
    m, d = model.m, model.d
    f6 = np.zeros(6); v6 = np.zeros(6)
    rows = []
    n = int(round((walk_t + seconds) / m.opt.timestep))
    fell = -1.0
    for _ in range(n):
        tl, ta = ctl.tick()
        s = ctl.state
        u = np.asarray(ctl.u_hold, dtype=float)
        row = [s.t, float(ctl.mode == "walking")]
        for leg in (0, 1):
            Rf = s.foot_R_W[leg]; p0 = s.foot_pos_W[leg]
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, int(model.foot_site[leg]), v6, 0)
            w, v = v6[:3].copy(), v6[3:].copy()
            fz = np.zeros(2); vpt = np.full((2, 2), np.nan); nc = 0
            for c in range(d.ncon):
                con = d.contact[c]
                if model._geom_foot[con.geom1] != leg and model._geom_foot[con.geom2] != leg:
                    continue
                nc += 1
                mujoco.mj_contactForce(m, d, c, f6)
                r = con.pos - p0
                k = 1 if Rf[:, 0].dot(r) > 0 else 0              # 0 뒤꿈치 · 1 발끝
                fz[k] += abs(f6[0])
                vpt[k] = (v + np.cross(w, r))[:2]
            yaw = np.arctan2(Rf[1, 0], Rf[0, 0]); c_, s_ = np.cos(yaw), np.sin(yaw)
            RT = np.array([[c_, s_, 0], [-s_, c_, 0], [0, 0, 1]])
            Fm = RT @ u[3 * leg:3 * leg + 3]; Mm = RT @ u[6 + 3 * leg:9 + 3 * leg]
            row += [float(ctl._active(leg, s.t)), float(nc), *fz, *p0[:2], *w, *vpt.reshape(-1),
                    Fm[1], Mm[1], Mm[2], *np.asarray(tl[leg], dtype=float)]
        rows.append(row)
        model.write_torque(tl, ta)
        mujoco.mj_step(m, d)
        if W._fell(d):
            fell = d.time; break
    np.savez(out, rows=np.array(rows), fell=fell, walk_t=walk_t, dt=m.opt.timestep)
    print(out, "fell", fell)


if __name__ == "__main__":
    run(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6])
