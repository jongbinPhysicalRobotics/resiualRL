"""선 접촉 진단 — MPC 가 명령한 디딤 발 wrench 와 실제 MuJoCo 접촉 wrench 를 발 좌표 (발 yaw) 로 기록하고,
OA-MPC line 제약 (CoP 행 6·7, 두 점 y 마찰 행 8~11) 의 여유를 함께 남긴다 (9/30 Q11).

사용: line_diag.py vx vy wz seconds variant out.npz
열 (다리 l = 0 왼, 1 오른): t, walk, yaw_err, yaw_rate, prim_res, 그리고 다리마다
  act, Fx Fy Fz Mx My Mz (MPC, 발 좌표), Fx Fy Fz Mx My Mz (실제 접촉, 발 site 기준, 발 좌표), n_con
line 여유는 분석 쪽에서 계산한다 (여기선 원자료만).
"""
import sys, importlib.util
import numpy as np
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent          # casadi_mit
sys.path.insert(0, str(_ROOT))
import mujoco
spec = importlib.util.spec_from_file_location("walk", str(_ROOT / "10_walk.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)


def _rz_T(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])   # 월드 → 발 yaw 좌표


def run(vx, vy, wz, seconds, var, out):
    kw = W.parse_variant(var); prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(vx, vy, wz, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
    m, d = model.m, model.d
    f6 = np.zeros(6)
    rows = []
    n = int(round((walk_t + seconds) / m.opt.timestep))
    fell = -1.0
    for _ in range(n):
        tl, ta = ctl.tick()
        s = ctl.state
        info = ctl.last_info or {}
        u = np.asarray(ctl.u_hold, dtype=float)
        fy_mpc = np.asarray(info.get("foot_yaw", [np.nan, np.nan]), dtype=float).reshape(-1, 2)[0]
        row = [s.t, float(ctl.mode == "walking"), s.yaw_unwrapped - float(ctl.body.euler_W[2]),
               float(s.torso_angvel_W[2]), float(info.get("prim_res", np.nan))]
        for leg in (0, 1):
            Rf = s.foot_R_W[leg]
            yaw = fy_mpc[leg] if np.isfinite(fy_mpc[leg]) else np.arctan2(Rf[1, 0], Rf[0, 0])
            RT = _rz_T(yaw)
            F = u[3 * leg:3 * leg + 3]; M = u[6 + 3 * leg:9 + 3 * leg]
            Fc = np.zeros(3); Mc = np.zeros(3); nc = 0
            for c in range(d.ncon):
                con = d.contact[c]
                g1, g2 = model._geom_foot[con.geom1], model._geom_foot[con.geom2]
                if g1 != leg and g2 != leg:
                    continue
                mujoco.mj_contactForce(m, d, c, f6)
                fW = con.frame.reshape(3, 3).T @ f6[:3]      # geom1 → geom2 방향 (geom2 가 받는 힘)
                if g1 == leg:
                    fW = -fW
                Fc += fW; Mc += np.cross(con.pos - s.foot_pos_W[leg], fW); nc += 1
            row += [float(ctl._active(leg, s.t)), *(RT @ F), *(RT @ M), *(RT @ Fc), *(RT @ Mc), float(nc)]
        rows.append(row)
        model.write_torque(tl, ta)
        mujoco.mj_step(m, d)
        if W._fell(d):
            fell = d.time; break
    np.savez(out, rows=np.array(rows), fell=fell, walk_t=walk_t)
    print(out, "fell", fell)


if __name__ == "__main__":
    run(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6])
