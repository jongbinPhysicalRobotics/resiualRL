"""실시간 여유 로그 — 120 s 동안 어디서 계산이 튀고, 시간이 갈수록 느려지는지 (Q&A 9/22 Q7).

기록 (창 없이, 09_walk 와 같은 루프):
  MPC 호출마다 : sim 시각, solve_gait 전체 ms, quadprog ms, quadprog 반복 수, 변수·제약 수, 지지 발 수
  sim 1 s 마다 : 실제 걸린 시간 (→ 배속), 고정 계산 벤치 ms (CPU 클럭이 떨어지면 이게 는다),
                 torque()·MuJoCo 누적 ms
  GC           : 파이썬 가비지 컬렉션 정지 시간·세대

사용:
  .venv/Scripts/python.exe MPC/src/20_timing_log.py --vx 0.7 --seconds 120 --out MPC/logs/timing_0p7.npz
  .venv/Scripts/python.exe MPC/src/20_timing_log.py --report MPC/logs/timing_0p7.npz
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, argparse, importlib, time, gc
from pathlib import Path
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_model, mpc_qp
walk = importlib.import_module("09_walk")

KW = dict(kp_up=300.0, swing_id=True, soft_land=True, lam_swing=True, wn_swing=30.0, zeta_swing=0.7,
          stance_frac=0.57, q_py=300, gate_sy=False, side_w=0.13, td_scale=1.25, cop_margin=0.9,
          wz_pelvis=1.0, wx_pelvis=0.5)


def bench():
    """고정 작업 (MPC 와 비슷한 크기의 numpy + 파이썬 루프). CPU 속도 지표."""
    t0 = time.perf_counter()
    A = np.arange(200 * 108, dtype=float).reshape(200, 108) % 7.0
    for _ in range(20):
        (A.T * 1.1) @ A
    s = 0
    for i in range(20000):
        s += i
    return 1000 * (time.perf_counter() - t0)


def record(vx, seconds, out):
    m, d = g1_model.load_torque(); g1_model.set_crouch(m, d)
    ctl = walk.WalkController(m, d, vx_cmd=vx, **KW)
    calls = []                                   # (t, solve_ms, qp_ms, iters, n_var, n_con, n_stance)
    cur = {}
    qp_orig = mpc_qp.quadprog.solve_qp

    def qp_wrap(G, a, C, b, meq=0):
        t0 = time.perf_counter()
        r = qp_orig(G, a, C, b, meq)
        cur["qp"] = 1000 * (time.perf_counter() - t0)
        cur["it"] = int(r[3][0]); cur["nv"] = G.shape[0]; cur["nc"] = C.shape[1]
        return r
    mpc_qp.quadprog.solve_qp = qp_wrap
    sg = ctl.mpc.solve_gait

    def sg_wrap(*a, **k):
        t0 = time.perf_counter(); r = sg(*a, **k)
        cur["sg"] = 1000 * (time.perf_counter() - t0); return r
    ctl.mpc.solve_gait = sg_wrap

    gcev = []; gcs = {}
    def gc_cb(phase, info):
        if phase == "start": gcs["t"] = time.perf_counter()
        elif "t" in gcs: gcev.append((sim_t[0], info["generation"], 1000 * (time.perf_counter() - gcs["t"])))
    gc.callbacks.append(gc_cb)
    sim_t = [0.0]

    dt = m.opt.timestep; spw = int(round(1.0 / dt))
    win = []                                     # (t_end, wall_s, bench_ms, torque_ms, mj_ms, mpc_ms)
    w_prev = time.perf_counter(); acc_tq = acc_mj = acc_mpc = 0.0
    wall0 = time.perf_counter(); fell = None
    for k in range(int(seconds / dt)):
        t = k * dt; sim_t[0] = t
        a0 = time.perf_counter(); mujoco.mj_forward(m, d); acc_mj += time.perf_counter() - a0
        if k % walk.DECIM == 0:
            a0 = time.perf_counter(); ctl.update_mpc(d, t); acc_mpc += time.perf_counter() - a0
            calls.append((t, cur.get("sg", np.nan), cur.get("qp", np.nan), cur.get("it", -1),
                          cur.get("nv", 0), cur.get("nc", 0),
                          sum(ctl.gait.in_stance(t, i) for i in range(2))))
        a0 = time.perf_counter(); tau = ctl.torque(d, t); acc_tq += time.perf_counter() - a0
        d.ctrl[:] = tau
        a0 = time.perf_counter(); mujoco.mj_step(m, d); acc_mj += time.perf_counter() - a0
        if (k + 1) % spw == 0:
            now = time.perf_counter()
            b = bench()
            win.append((t + dt, now - w_prev, b, 1000 * acc_tq, 1000 * acc_mj, 1000 * acc_mpc))
            acc_tq = acc_mj = acc_mpc = 0.0
            w_prev = time.perf_counter()          # 벤치 시간은 창에서 뺀다
        if d.qpos[2] < 0.5:
            fell = t; break
    gc.callbacks.remove(gc_cb); mpc_qp.quadprog.solve_qp = qp_orig
    np.savez(out, calls=np.array(calls), win=np.array(win), gc=np.array(gcev) if gcev else np.zeros((0, 3)),
             vx=vx, fell=-1.0 if fell is None else fell, wall=time.perf_counter() - wall0)
    print(f"저장: {out}", flush=True)


def report(path):
    z = np.load(path)
    C, W, G = z["calls"], z["win"], z["gc"]
    t, sg, qp, it, nv, nc, ns = C.T
    build = sg - qp
    print(f"\n=== {Path(path).name}: vx {float(z['vx'])}, {'완주' if z['fell'] < 0 else f'전도 {float(z['fell']):.1f}s'}, "
          f"MPC 호출 {len(C)} 회 ===")
    q = lambda a: f"평균 {np.mean(a):5.2f} · 중앙 {np.median(a):5.2f} · p99 {np.percentile(a, 99):6.2f} · 최대 {np.max(a):6.1f}"
    print(f"  solve_gait ms : {q(sg)}")
    print(f"  └ quadprog ms : {q(qp)}")
    print(f"  └ 조립 ms     : {q(build)}")
    print(f"  quadprog 반복 : 평균 {it.mean():.1f}, 최대 {it.max():.0f}")
    thr = max(3 * np.median(sg), 10.0)
    sp = sg > thr
    print(f"  튄 호출 (> {thr:.1f} ms): {sp.sum()} 회 ({100 * sp.mean():.2f} %)")
    if sp.any():
        print(f"    그때 quadprog 비중 {np.nanmean(qp[sp] / sg[sp]) * 100:.0f} %, 반복 평균 {it[sp].mean():.1f} (평소 {it[~sp].mean():.1f})")
        for s_ in np.flatnonzero(sp)[:8]:
            print(f"    t={t[s_]:6.2f}s  전체 {sg[s_]:6.1f}  qp {qp[s_]:6.1f}  반복 {it[s_]:3.0f}  변수 {nv[s_]:3.0f}  제약 {nc[s_]:3.0f}  지지발 {ns[s_]:.0f}")
    # 시간에 따른 변화 (20 s 구간)
    print("  구간별     | 배속 (창 없이) | 벤치 ms | solve_gait 중앙 · p99 | quadprog 중앙 | 조립 중앙 | torque ms/s | GC ms")
    for a in range(0, int(W[-1, 0]), 20):
        wm = (W[:, 0] > a) & (W[:, 0] <= a + 20)
        cm = (t >= a) & (t < a + 20)
        gm = (G[:, 0] >= a) & (G[:, 0] < a + 20) if len(G) else np.zeros(0, bool)
        if not wm.any():
            continue
        rtf = wm.sum() / W[wm, 1].sum()
        print(f"  {a:3d}~{a + 20:3d} s | {rtf:13.2f} x | {np.median(W[wm, 2]):6.2f} | "
              f"{np.median(sg[cm]):8.2f} · {np.percentile(sg[cm], 99):6.2f} | {np.median(qp[cm]):12.2f} | "
              f"{np.median(build[cm]):8.2f} | {np.mean(W[wm, 3]):10.0f} | {G[gm, 2].sum() if len(G) else 0:5.1f}")
    # 지지 발 수별
    for n_ in (1, 2):
        mm = ns == n_
        if mm.any():
            print(f"  지금 지지발 {n_}개: solve_gait 중앙 {np.median(sg[mm]):.2f}, p99 {np.percentile(sg[mm], 99):.2f} ms (호출 {mm.sum()})")
    if len(G):
        print(f"  GC: {len(G)} 회, 총 {G[:, 2].sum():.1f} ms, 최대 {G[:, 2].max():.1f} ms (gen2 {int((G[:, 1] == 2).sum())} 회)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.7)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", nargs="*", default=None)
    a = ap.parse_args()
    if a.report is not None:
        for p in a.report:
            report(p)
    else:
        record(a.vx, a.seconds, a.out or f"timing_{a.vx:g}.npz")
