"""고정 반복 ADMM 해 품질 vs 반복 수. 걷기 중 QP 를 저장해 오프라인으로 k 회 반복 해를 정확 해와 비교.
사용: iter_diag.py vx wz seconds variant out.pkl"""
import sys, importlib.util, pickle
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
    tl, ta = ctl.tick(); model.write_torque(tl, ta); mujoco.mj_step(m, d)
    mpc = ctl.mpc
    orig = mpc._solve_fixed
    samples = []
    cnt = [0]
    last_y = [None]

    def patched(Px, q, Ax, l, u, sig, stance, mass, info):
        warm = mpc._warm.copy() if mpc._has_prev else None
        y_prev = last_y[0]
        U = orig(Px, q, Ax, l, u, sig, stance, mass, info)
        cnt[0] += 1
        if d.time > walk_t + 3.0 and cnt[0] % 20 == 0 and len(samples) < 60:
            samples.append(dict(t=d.time, Px=Px.copy(), q=q.copy(), Ax=Ax.copy(), l=l.copy(), u=u.copy(),
                                warm=warm, y=None if y_prev is None else y_prev.copy(), U=U.copy()))
        return U

    mpc._solve_fixed = patched
    # 쌍대 y 캡처: osqp solve 결과를 가로챈다
    s0 = mpc._solver
    n = int(round((walk_t + seconds) / m.opt.timestep))
    for _ in range(n):
        if mpc._solver is not None and not getattr(mpc._solver, "_wrapped", False):
            sv = mpc._solver; osolve = sv.solve

            def wsolve(*a, _o=osolve, **k):
                r = _o(*a, **k); last_y[0] = None if r.y is None else np.array(r.y, copy=True); return r
            sv.solve = wsolve; sv._wrapped = True
        tl, ta = ctl.tick(); model.write_torque(tl, ta); mujoco.mj_step(m, d)
        if W._fell(d):
            break
    pickle.dump(dict(samples=samples, P_idx=(mpc.P_pattern.indices, mpc.P_pattern.indptr, mpc.P_pattern.shape),
                     A_idx=(mpc.A_pattern.indices, mpc.A_pattern.indptr, mpc.A_pattern.shape),
                     settings=mpc.settings), open(out, "wb"))
    print(out, len(samples), "fell" if W._fell(d) else "ok", d.time)


if __name__ == "__main__":
    run(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5])
