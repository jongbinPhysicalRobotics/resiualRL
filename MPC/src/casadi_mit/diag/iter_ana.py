"""iter_diag 표본으로: k 회 고정 반복 (같은 warm x, y) 해 vs 정확 해."""
import sys, pickle
import numpy as np, scipy.sparse as sp, osqp
D = pickle.load(open(sys.argv[1], "rb"))
Pi, Pp, Ps = D["P_idx"]; Ai, Ap, As = D["A_idx"]; st = dict(D["settings"])
KS = [25, 50, 75, 100, 150, 200, "conv"]
err = {k: [] for k in KS}; vio = {k: [] for k in KS}; gap = {k: [] for k in KS}; err_in = []; its = []; Jstar = []
for s in D["samples"]:
    P = sp.csc_matrix((s["Px"], Pi, Pp), shape=Ps); A = sp.csc_matrix((s["Ax"], Ai, Ap), shape=As)
    ex = osqp.OSQP(); ex.setup(P, s["q"], A, s["l"], s["u"], verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=200000,
                                polishing=True, rho=1e-5, adaptive_rho=True, scaling=10)
    xs = ex.solve().x
    u0s = xs[:12]
    J = lambda x: 0.5 * x @ (P @ x) + s["q"] @ x   # P 는 상삼각만 저장 → 대칭화 필요
    Pf = P + sp.triu(P, 1).T
    J = lambda x: 0.5 * x @ (Pf @ x) + s["q"] @ x
    Js = J(xs); Jstar.append(Js)
    for k in KS:
        so = osqp.OSQP()
        if k == "conv":
            so.setup(P, s["q"], A, s["l"], s["u"], verbose=False, warm_starting=True, polishing=False, max_iter=200,
                     adaptive_rho=1, adaptive_rho_interval=50, check_dualgap=1, eps_abs=1e-3, eps_rel=1e-3,
                     eps_prim_inf=1e-4, eps_dual_inf=1e-4, sigma=1e-6, alpha=1.6, scaling=10, check_termination=25,
                     scaled_termination=0, delta=1e-6, adaptive_rho_tolerance=5.0, adaptive_rho_fraction=0.4, rho=1e-5)
        else:
            kw = {k_: v for k_, v in st.items() if k_ in ("sigma", "alpha", "scaling", "delta", "rho")}
            so.setup(P, s["q"], A, s["l"], s["u"], verbose=False, warm_starting=True, polishing=False, max_iter=k,
                     adaptive_rho=False, check_termination=0, eps_abs=1e-3, eps_rel=1e-3, **kw)
        if s["warm"] is not None:
            if s["y"] is not None:
                so.warm_start(x=s["warm"], y=s["y"])
            else:
                so.warm_start(x=s["warm"])
        res = so.solve(raise_error=False); x = res.x
        if k == "conv":
            its.append(res.info.iter)
        gap[k].append(J(x) - Js)
        Ax_ = A @ x
        v = max(0.0, float(np.max(np.maximum(s["l"] - Ax_, Ax_ - s["u"]))))
        err[k].append(np.linalg.norm(x[:12] - u0s)); vio[k].append(v)
    err_in.append(np.linalg.norm(s["U"][:12] - u0s))
print("n", len(D["samples"]), "rho", st.get("rho"), "alpha", st.get("alpha"))
print(f"  in-sim (admm 사용값)  |u0-u0*| med {np.median(err_in):6.2f} p90 {np.percentile(err_in,90):6.2f}")
for k in KS:
    e = np.array(err[k]); v = np.array(vio[k])
    g = np.array(gap[k])
    print(f"  k={str(k):>4s}  |u0-u0*| med {np.median(e):6.2f} p90 {np.percentile(e,90):6.2f}   위반 med {np.median(v):5.2f} p90 {np.percentile(v,90):5.2f}   목적 차 med {np.median(g):9.3g} p90 {np.percentile(g,90):9.3g}")
print("  conv iters", np.median(its), np.max(its), " |J*| med", "%.3g" % np.median(np.abs(Jstar)))
