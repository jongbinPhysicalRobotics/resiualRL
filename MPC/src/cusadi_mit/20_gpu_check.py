"""일괄 MPC (batch_mpc.py) 를 CPU 판 (CasadiCondensedMPC + OSQP 1.1.3) 과 대조 · 시간 재기 (Q&A 11/9).

1) 기록: CPU 판 A42 로 닫힌 고리를 돌리며 매 풀이의 입력 · OSQP 가 받은 QP · 해를 모은다 (경우마다 따로).
2) 재생: 같은 입력을 경우 수 B 만큼의 배치로 시간 순서대로 넣는다 (환경마다 warm start 상태가 이어진다).
     [조립] BatchQP 의 P · q · A · u  vs  CPU 가 OSQP 에 넘긴 Px · q · Ax · u
     [풀이] CPU 의 QP 를 그대로 BatchOSQP 에 넣은 해  vs  OSQP 해 (Ū)       — 풀이기만의 차이
     [전체] 입력 → BatchMPC 해 U  vs  CPU info["U"]                          — 조립 + 풀이
3) 시간: --bench 1,64,512,4096 — 기록한 입력을 환경 수만큼 복제해 BatchMPC.solve 한 번의 시간 (조립 / 풀이 나눠서).

예:
  노트북 (CPU, 검증만):   .venv/bin/python MPC/src/cusadi_mit/20_gpu_check.py
  실험실 (GPU):           python MPC/src/cusadi_mit/20_gpu_check.py --device cuda --dtype float64 --backend cusadi --linsys chol
                          python MPC/src/cusadi_mit/20_gpu_check.py --device cuda --dtype float32 --backend cusadi --linsys cudss --bench 1,64,512,2048,4096
  기록 저장 / 다시 쓰기:  --save MPC/logs/gpu/golden_A42.npz  ·  --load MPC/logs/gpu/golden_A42.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "diag"))

import numpy as np
import torch

import mpc as mpc_mod
import mpc_casadi
from configs import resolve
from batch_mpc import BatchMPC, BatchOSQP, from_global, NC_STEP

# 경우: (이름, vx, vy, wz, stand, push (Fx, Fy, t0, dur) 또는 None)
CASES = [
    ("stand", 0.0, 0.0, 0.0, True, None),
    ("stand_push40", 0.0, 0.0, 0.0, True, (0.0, 40.0, 0.5, 0.2)),
    ("walk0.5", 0.5, 0.0, 0.0, False, None),
    ("walk1.0", 1.0, 0.0, 0.0, False, None),
    ("lat0.5", 0.0, 0.5, 0.0, False, None),
    ("turn2.5", 0.0, 0.0, 2.5, False, None),
    ("walk0.5_push50", 0.5, 0.0, 0.0, False, (0.0, 50.0, 1.0, 0.2)),
]
KEYS = ("x0", "X_ref", "psi", "r", "stance", "min_scale", "foot_yaw", "mass", "inertia", "walking",
        "U", "Px", "q", "Ax", "l", "u", "Ubar")


def _engine():
    spec = importlib.util.spec_from_file_location("w10", str(HERE / "10_walk.py"))
    eng = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eng)
    return eng


def record(cfg_name: str, seconds: float, cases):
    """CPU 닫힌 고리를 돌리며 풀이마다 기록. → (경우별 dict of arrays, 마지막 MPC 객체)."""
    eng = _engine()
    kw = eng.parse_variant(resolve(cfg_name))
    kw.pop("stand", None)
    orig_solve = mpc_casadi.CasadiCondensedMPC.solve
    orig_fixed = mpc_mod.ConvexMPC._solve_fixed
    store = {}
    holder = {}

    def solve_wrap(self, x0, ref, steps, foot_yaw, reduced, mode, active=None, mz_ref=None):
        holder["mpc"] = self
        self._cap = {}
        u0, info = orig_solve(self, x0, ref, steps, foot_yaw, reduced, mode, active=active, mz_ref=mz_ref)
        if info.get("fallback") or "Px" not in self._cap:
            return u0, info
        fy = np.asarray(foot_yaw, dtype=float)
        rec = dict(x0=np.asarray(x0, float), X_ref=np.asarray(ref.X_ref, float), psi=np.asarray(ref.psi, float),
                   r=np.asarray(ref.r, float),
                   stance=np.array([np.asarray(s.stance, bool) for s in steps]).reshape(self.N, 2),
                   min_scale=np.array([np.asarray(s.min_scale, float) for s in steps]).reshape(self.N, 2),
                   foot_yaw=(fy if fy.ndim == 2 else np.tile(fy.reshape(1, 2), (self.N, 1))),
                   mass=float(reduced.mass), inertia=np.asarray(reduced.inertia_B, float).reshape(3, 3),
                   walking=(self._mode_now == "walking"), U=np.asarray(info["U"], float).copy(), **self._cap)
        for k_, v_ in rec.items():
            store[holder["case"]].setdefault(k_, []).append(v_)
        return u0, info

    def fixed_wrap(self, Px, q, Ax, l, u, sig, stance, mass, info):
        U = orig_fixed(self, Px, q, Ax, l, u, sig, stance, mass, info)
        self._cap = dict(Px=np.array(Px, float), q=np.array(q, float), Ax=np.array(Ax, float),
                         l=np.array(l, float), u=np.array(u, float), Ubar=np.array(U, float))
        return U

    mpc_casadi.CasadiCondensedMPC.solve = solve_wrap
    mpc_mod.ConvexMPC._solve_fixed = fixed_wrap
    try:
        for name, vx, vy, wz, stand, push in cases:
            holder["case"] = name
            store[name] = {}
            t0 = time.perf_counter()
            r = eng.headless(vx, vy, wz, seconds, verbose=False, stand=stand, push=push, **kw)
            n = len(store[name].get("U", []))
            print(f"  기록 {name:16s} 풀이 {n:4d} 회  {'넘어짐' if r['fell'] else 'ok'}  ({time.perf_counter() - t0:.0f} s)", flush=True)
    finally:
        mpc_casadi.CasadiCondensedMPC.solve = orig_solve
        mpc_mod.ConvexMPC._solve_fixed = orig_fixed
    out = {name: {k_: np.array(v_) for k_, v_ in d.items()} for name, d in store.items()}
    return out, holder["mpc"]


def save(path, data, cfg_name, seconds):
    flat = {"_cases": np.array(list(data.keys())), "_cfg": np.array(cfg_name), "_seconds": np.array(seconds)}
    for name, d in data.items():
        for k_, v_ in d.items():
            flat[f"{name}/{k_}"] = v_
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)
    print(f"  저장: {path}")


def load(path):
    z = np.load(path, allow_pickle=False)
    names = list(z["_cases"])
    data = {n: {k_: z[f"{n}/{k_}"] for k_ in KEYS if f"{n}/{k_}" in z.files} for n in names}
    return data, str(z["_cfg"]), float(z["_seconds"])


def cpu_template(cfg_name):
    """기록 없이 BatchMPC 를 만들 CPU MPC 객체 (같은 설정) — 짧은 서기 1 회로 컨트롤러를 만든다."""
    eng = _engine()
    kw = eng.parse_variant(resolve(cfg_name))
    kw.pop("stand", None)
    keys = ("solver", "armature", "profile", "markers", "friction_scale", "fsm_double_update", "osqp_settings", "fixes",
            "x_dot_max", "k_half", "foot_model", "mpc_mu", "stand_yaw_hold", "stand_bias", "side_swap", "raibert_gain",
            "ypos_gain", "ypath", "arm_swing", "cop_half", "w_moment", "q_state", "sim_cone", "yaw_hold", "yaw_lead",
            "point_fmin", "k_half_y", "swing_over", "xpath", "raw_over", "w_force_xy", "sym_com", "sym_inertia",
            "hip_fix", "foot_geom", "sim_mu", "mass_rand")
    sk = {k_: kw[k_] for k_ in keys if k_ in kw}
    sk.setdefault("solver", "osqp"); sk.setdefault("armature", True); sk.setdefault("profile", "ramp")
    sk.setdefault("markers", False); sk.setdefault("friction_scale", 1.0); sk.setdefault("fsm_double_update", True)
    solver = sk.pop("solver"); arm = sk.pop("armature"); prof = sk.pop("profile"); mk = sk.pop("markers")
    fs = sk.pop("friction_scale"); fsm2 = sk.pop("fsm_double_update")
    _, _, ctl, _ = eng._setup(0.0, 0.0, 0.0, solver, arm, prof, mk, True, fs, fsm2, False, **sk)
    ctl.tick()                                                # MPC 객체는 첫 tick 의 _initialize 에서 만들어진다
    return ctl.mpc


def _stack(data, names, key, t):
    return np.stack([data[n][key][t] for n in names])


def _rel(a, b):
    return float(np.max(np.abs(a - b))), float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12))


def replay(data, cpu, names, device, dtype, backend, linsys):
    T = min(len(data[n]["U"]) for n in names)
    B = len(names)
    bm = BatchMPC.from_cpu(cpu, B, device=device, dtype=dtype, backend=backend, linsys=linsys)
    s = cpu.settings
    solo = BatchOSQP(B, cpu.N, sigma=s["sigma"], alpha=s["alpha"], scaling=s["scaling"], rho_walk=s["rho"],
                     rho_stand=(cpu.rho_stand or s["rho"]), iters_walk=cpu.fixed_iter,
                     iters_stand=(cpu.fixed_iter_stand or cpu.fixed_iter), init_mode=cpu.init_mode,
                     warm_shift=cpu.warm_shift, shifted=cpu.use_shifted_warm_start, gravity=cpu.gravity,
                     device=device, dtype=dtype, linsys=linsys, a_mask=bm.qp.A_mask)
    P_rows, P_cols, gather = cpu._P_rows, cpu._P_cols, cpu._A_gather
    n, nv = cpu.N, cpu.nv
    err = {k_: np.zeros((B, 2)) for k_ in ("P", "q", "A", "u", "solve", "U", "u0")}
    t_build = t_solve = 0.0
    tt = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
    sync = (lambda: torch.cuda.synchronize()) if str(device).startswith("cuda") else (lambda: None)
    for t in range(T):
        g = lambda k_: _stack(data, names, k_, t)
        inp = [g(k_) for k_ in ("x0", "X_ref", "psi", "r", "stance", "min_scale", "foot_yaw", "mass", "inertia", "walking")]
        inp[3] = inp[3].reshape(B, n, 2, 3)
        sync(); t0 = time.perf_counter()
        P, q, A, l, u = bm.qp.build(*inp)
        sync(); t1 = time.perf_counter()
        X, ok = bm.osqp.solve(P, q, A, l, u, inp[9], stance=tt(inp[4]).bool(), mass=tt(inp[7]))
        sync(); t2 = time.perf_counter()
        t_build += t1 - t0; t_solve += t2 - t1
        U = (X * bm.Sv[None] if bm.var_scale else X).reshape(B, n, 12).cpu().double().numpy()
        Pn, qn = P.cpu().double().numpy(), q.cpu().double().numpy()
        An = A.cpu().double().numpy().reshape(B, -1)[:, gather]
        un = from_global(torch.as_tensor(g("u")), n).numpy()
        for b in range(B):
            for k_, mine, ref in (("P", Pn[b][P_rows, P_cols], data[names[b]]["Px"][t]), ("q", qn[b], data[names[b]]["q"][t]),
                                  ("A", An[b], data[names[b]]["Ax"][t]),
                                  ("u", u.cpu().double().numpy()[b], un[b]),
                                  ("U", U[b], data[names[b]]["U"][t]), ("u0", U[b][0], data[names[b]]["U"][t][0])):
                e_ = _rel(mine, ref)
                err[k_][b] = np.maximum(err[k_][b], e_)
        # 풀이기만: CPU 가 OSQP 에 넘긴 QP 를 그대로
        Pd = np.zeros((B, nv, nv))
        Ad = np.zeros((B, n * NC_STEP * 12))
        for b in range(B):
            Pd[b][P_rows, P_cols] = data[names[b]]["Px"][t]
            Pd[b][P_cols, P_rows] = data[names[b]]["Px"][t]
            Ad[b][gather] = data[names[b]]["Ax"][t]
        Xs, _ = solo.solve(tt(Pd), tt(g("q")), tt(Ad.reshape(B, n, NC_STEP, 12)),
                           from_global(tt(np.maximum(g("l"), -1e30)), n), from_global(tt(g("u")), n),
                           inp[9], stance=tt(inp[4]).bool(), mass=tt(inp[7]))
        Xs = Xs.cpu().double().numpy()
        for b in range(B):
            err["solve"][b] = np.maximum(err["solve"][b], _rel(Xs[b], data[names[b]]["Ubar"][t]))
    print(f"\n  재생 {T} 풀이 × {B} 경우  ({device}, {str(dtype).split('.')[-1]}, backend {backend}, linsys {linsys})")
    print(f"  풀이 1 회 평균: 조립 {t_build / T * 1e3:.2f} ms · ADMM {t_solve / T * 1e3:.2f} ms  (B = {B}, 첫 호출 포함)")
    print(f"  {'경우':16s} " + " ".join(f"{k_:>17s}" for k_ in ("P", "q", "A", "u", "풀이(같은 QP)", "U 전체", "u0")))
    print(f"  {'':16s} " + " ".join(f"{'최대|Δ| / 상대':>17s}" for _ in range(7)))
    for b, nm in enumerate(names):
        print(f"  {nm:16s} " + " ".join(f"{err[k_][b, 0]:8.1e}/{err[k_][b, 1]:7.1e}"
                                         for k_ in ("P", "q", "A", "u", "solve", "U", "u0")))
    return err


def bench(data, cpu, names, sizes, device, dtype, backend, linsys, reps=10):
    print(f"\n  시간 ({device}, {str(dtype).split('.')[-1]}, backend {backend}, linsys {linsys}) — 풀이 1 회 (B 환경 동시)")
    base = names[0] if "walk0.5" not in names else "walk0.5"
    d = data[base]
    t_mid = len(d["U"]) // 2
    n = cpu.N
    sync = (lambda: torch.cuda.synchronize()) if str(device).startswith("cuda") else (lambda: None)
    for B in sizes:
        bm = BatchMPC.from_cpu(cpu, B, device=device, dtype=dtype, backend=backend, linsys=linsys)
        rep = lambda k_: np.repeat(d[k_][t_mid][None], B, axis=0)
        inp = [rep(k_) for k_ in ("x0", "X_ref", "psi", "r", "stance", "min_scale", "foot_yaw")] + \
              [np.full(B, d["mass"][t_mid]), rep("inertia"), np.full(B, bool(d["walking"][t_mid]))]
        inp[3] = inp[3].reshape(B, n, 2, 3)
        inp = [torch.as_tensor(a, device=device) for a in inp]
        for _ in range(2):                                    # 예열 (커널 · 분해 준비)
            P, q, A, l, u = bm.qp.build(*inp)
            bm.osqp.solve(P, q, A, l, u, inp[9], stance=inp[4].bool(), mass=inp[7])
        tb = ts = 0.0
        for _ in range(reps):
            sync(); t0 = time.perf_counter()
            P, q, A, l, u = bm.qp.build(*inp)
            sync(); t1 = time.perf_counter()
            bm.osqp.solve(P, q, A, l, u, inp[9], stance=inp[4].bool(), mass=inp[7])
            sync(); t2 = time.perf_counter()
            tb += t1 - t0; ts += t2 - t1
        mem = (torch.cuda.max_memory_allocated() / 2**20) if str(device).startswith("cuda") else float("nan")
        print(f"  B {B:5d}:  조립 {tb / reps * 1e3:8.2f} ms   ADMM {ts / reps * 1e3:8.2f} ms   합 {(tb + ts) / reps * 1e3:8.2f} ms"
              f"   ({(tb + ts) / reps / B * 1e6:7.1f} µs/환경)   GPU 메모리 최대 {mem:7.0f} MiB", flush=True)
        del bm
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="A42")
    ap.add_argument("--seconds", type=float, default=2.0, help="경우마다 걷는 시간 (서기 경우는 시뮬 시간)")
    ap.add_argument("--cases", default="all", help="쉼표로 경우 이름 (기본 전부): " + ",".join(c[0] for c in CASES))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--backend", default="casadi", choices=["casadi", "cusadi"])
    ap.add_argument("--linsys", default="chol", choices=["chol", "cudss", "kktcpu"])
    ap.add_argument("--save", default=None)
    ap.add_argument("--load", default=None)
    ap.add_argument("--bench", default=None, help="환경 수 목록, 예 1,64,512,4096")
    ap.add_argument("--noreplay", action="store_true")
    a = ap.parse_args()
    dtype = torch.float64 if a.dtype == "float64" else torch.float32
    if a.load:
        data, cfg_name, sec = load(a.load)
        print(f"  불러옴: {a.load} (cfg {cfg_name}, {sec} s, 경우 {list(data)})")
        cpu = cpu_template(cfg_name)
    else:
        cases = CASES if a.cases == "all" else [c for c in CASES if c[0] in a.cases.split(",")]
        print(f"  CPU 기록: cfg {a.cfg}, 경우 {len(cases)} 개 × {a.seconds} s")
        data, cpu = record(a.cfg, a.seconds, cases)
        cfg_name = a.cfg
        if a.save:
            save(a.save, data, cfg_name, a.seconds)
    names = [n for n in data if len(data[n].get("U", [])) > 0]
    if a.load and a.cases != "all":
        names = [n for n in names if n in a.cases.split(",")]
    if not a.noreplay:
        replay(data, cpu, names, a.device, dtype, a.backend, a.linsys)
    if a.bench:
        bench(data, cpu, names, [int(v) for v in a.bench.split(",")], a.device, dtype, a.backend, a.linsys)


if __name__ == "__main__":
    main()
