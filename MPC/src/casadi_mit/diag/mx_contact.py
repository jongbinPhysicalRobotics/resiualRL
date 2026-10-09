"""실제 지면 → 디딤 발 렌치 (발 site 기준, 발 좌표) 의 Mx 크기 (Q&A 10/6 Q30). 걷기 시작 + 2 s 부터, 디딤 Fz > 50 N 틱.
접촉마다 mj_contactForce (접촉 좌표) → 월드, 발이 받는 힘으로 부호를 맞춰 site 둘레 모멘트를 합한다.
사용: mx_contact.py vx vy wz seconds variant
"""
import sys, importlib.util
import numpy as np
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent          # casadi_mit
sys.path.insert(0, str(_ROOT))
import mujoco
spec = importlib.util.spec_from_file_location("walk", str(_ROOT / "10_walk.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)


def run(vx, vy, wz, seconds, var):
    kw = W.parse_variant(var); prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(vx, vy, wz, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
    m, d = model.m, model.d
    f6 = np.zeros(6); rows = []
    fell = -1.0
    for _ in range(int(round((walk_t + seconds) / m.opt.timestep))):
        tl, ta = ctl.tick(); s = ctl.state
        if s.t > walk_t + 2.0:
            for leg in (0, 1):
                if not ctl._active(leg, s.t):
                    continue
                Rf = np.asarray(s.foot_R_W[leg]); p0 = np.asarray(s.foot_pos_W[leg])
                F = np.zeros(3); M = np.zeros(3); nc = 0
                for c in range(d.ncon):
                    con = d.contact[c]
                    g1, g2 = model._geom_foot[con.geom1], model._geom_foot[con.geom2]
                    if g1 != leg and g2 != leg:
                        continue
                    nc += 1
                    mujoco.mj_contactForce(m, d, c, f6)
                    fw = con.frame.reshape(3, 3).T @ f6[:3]       # 법선은 geom1 → geom2
                    if g2 != leg:
                        fw = -fw
                    F += fw; M += np.cross(con.pos - p0, fw)
                if nc:
                    Ff, Mf = Rf.T @ F, Rf.T @ M
                    rows.append([*Ff, *Mf, np.arctan2(Rf[2, 1], Rf[2, 2])])
        model.write_torque(tl, ta); mujoco.mj_step(m, d)
        if W._fell(d):
            fell = d.time; break
    r = np.array(rows); r = r[r[:, 2] > 50]
    q = lambda a: np.round(np.percentile(np.abs(a), [50, 95, 99]), 3)
    print(f"vx {vx} vy {vy} wz {wz}  fell {fell:.1f}  틱 {len(r)}   (p50 / p95 / p99)")
    print("  실제 |Mx| [N·m]        ", q(r[:, 3]), "   |My| [N·m]", q(r[:, 4]))
    print("  |Mx|/Fz [mm]           ", q(1e3 * r[:, 3] / r[:, 2]), "   |My|/Fz [mm]", q(1e3 * r[:, 4] / r[:, 2]))
    print("  |Fy| [N]               ", q(r[:, 1]), "   발 roll [°]", q(np.degrees(r[:, 6])))


if __name__ == "__main__":
    run(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
