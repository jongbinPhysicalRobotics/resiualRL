"""요소별 힘·토크 리포트 — casadi_mit (Q&A 9/26 Q5·Q6).

walk 모드 (headless 기록 사용):
  1) 관절별 토크 p50/p95/max vs 한계 — 그 다리의 디딤/스윙 구분
  2) MPC wrench 성분별 (하중 실린 디딤 발): Fz, |F_xy|, Mx/My/Mz
  3) **접촉 없는데 힘 명령** (active=1, 측정 Fn < 5 N, 명령 Fz > 30 N) — G1 9/22 Q11 유형
  4) 스윙 발 wrench ≠ 0 (D 제약상 0 이어야 함; fallback 틱 제외)
  5) 토크 포화 틱의 접촉 상태 (포화가 '접촉 없는 힘' 때문인지)

stand 모드 (자체 루프, GAIT_ON=0 서기 진단 — 발이 안으로 미끄러지는 현상):
  발 site·뒤꿈치·발끝 위치, 발 yaw, hip yaw 관절, 명령 wrench (Fy·Mz), 접촉점별 마찰 사용률
  (|F_접선|/F_법선, 1.0 이면 미끄러지는 중) 을 기록해 드리프트의 원인 (병진 vs 회전, 명령 vs 잔떨림) 을 가른다.

실행 (ROOT):
    .venv/bin/python MPC/src/casadi_mit/12_force_report.py walk --vx 0.6 --seconds 20
    .venv/bin/python MPC/src/casadi_mit/12_force_report.py stand --seconds 60
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib

import mujoco
import numpy as np

walk = importlib.import_module("10_walk")

JN = ["hip_yaw", "hip_abad", "hip_pitch", "knee", "ankle"]
UC = ["FxL", "FyL", "FzL", "FxR", "FyR", "FzR", "MxL", "MyL", "MzL", "MxR", "MyR", "MzR"]


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


# ---------------------------------------------------------------------------
def run_walk(a):
    r = walk.headless(a.vx, a.vy, a.wz, seconds=a.seconds, profile="ramp", fixes=("yawref",),
                      verbose=False, keep_record=True)
    R = r.get("record", r.get("rec"))
    m = walk.MitModel(armature=True)                       # 한계 조회용 (기록과 같은 모델)
    lim = m.ctrl_hi[m.leg_act.ravel()]
    t = R["t"]
    win = t > (r.get("walk_t", 1.0) or 1.0) + 5.0
    act = R["active"] > 0.5
    fb = R["fallback"] > 0.5 if "fallback" in R else np.zeros(len(t), bool)
    tau = R["tau"]                                          # (n,10) = [L 5 | R 5]
    u = R["u"]
    fn = R["fn"]
    print(f"\n[walk] vx {a.vx} vy {a.vy} wz {a.wz}, {r['survived']:.1f}/{a.seconds:.0f} s, "
          f"창 {t[win][0]:.1f}~{t[-1]:.1f} s ({win.sum()} 틱)  체중 {9.81 * 24.89:.0f} N")

    print("\n1) 관절별 |토크| [N·m] (그 다리 디딤 / 스윙)   한계")
    for leg, side in ((0, "L"), (1, "R")):
        for j in range(5):
            col = np.abs(tau[:, 5 * leg + j])
            st = col[win & act[:, leg]]
            sw = col[win & ~act[:, leg]]
            print(f"  {side} {JN[j]:9s} 디딤 p50 {q(st,50):6.1f} p95 {q(st,95):6.1f} max {st.max() if len(st) else 0:7.1f} | "
                  f"스윙 p50 {q(sw,50):6.1f} p95 {q(sw,95):6.1f} max {sw.max() if len(sw) else 0:7.1f} | {lim[5*leg+j]:5.0f}")

    print("\n2) MPC wrench (하중 실린 디딤 발, Fz > 20 N)")
    for leg, (ix, iy, iz, mx, my, mz) in ((0, (0, 1, 2, 6, 7, 8)), (1, (3, 4, 5, 9, 10, 11))):
        mk = win & act[:, leg] & (u[:, iz] > 20.0)
        if not mk.any():
            continue
        fxy = np.hypot(u[mk, ix], u[mk, iy])
        print(f"  {'LR'[leg]}: Fz p50 {q(u[mk, iz],50):5.0f} p95 {q(u[mk, iz],95):5.0f} max {u[mk, iz].max():6.0f} N | "
              f"|Fxy| p95 {q(fxy,95):5.1f} max {fxy.max():6.1f} | "
              f"|Mx| p95 {q(np.abs(u[mk, mx]),95):4.1f}  |My| p95 {q(np.abs(u[mk, my]),95):4.1f}  "
              f"|Mz| p95 {q(np.abs(u[mk, mz]),95):4.1f} N·m")

    print("\n3) 접촉 없는데 힘 명령 (active=1, 측정 Fn < 5 N, 명령 Fz > 30 N)")
    for leg, iz in ((0, 2), (1, 5)):
        mk = win & act[:, leg] & (fn[:, leg] < 5.0) & (u[:, iz] > 30.0)
        na = (win & act[:, leg]).sum()
        if mk.any():
            sat = (np.abs(tau[mk][:, 5 * leg:5 * leg + 5]) >= lim[5 * leg:5 * leg + 5][None]).any(1)
            print(f"  {'LR'[leg]}: {mk.sum()} 틱 ({100 * mk.sum() / max(na, 1):.2f} % of 디딤)  "
                  f"명령 Fz max {u[mk, iz].max():.0f} N  그중 토크 포화 {100 * sat.mean():.0f} %")
        else:
            print(f"  {'LR'[leg]}: 없음")

    print("\n4) 스윙(비활성) 발 wrench — OSQP 등식 잔차 vs 이륙 직후 stale u_hold (fallback 제외)")
    dt_tick = float(np.median(np.diff(t)))
    for leg, cols in ((0, (0, 1, 2, 6, 7, 8)), (1, (3, 4, 5, 9, 10, 11))):
        mk = win & ~act[:, leg] & ~fb
        mag = np.abs(u[np.ix_(mk, cols)]).max(1) if mk.any() else np.array([])
        # 이륙 (active 1→0) 후 경과 시간
        age = np.full(len(t), np.nan)
        t_off = None
        for k_ in range(1, len(t)):
            if not act[k_, leg] and act[k_ - 1, leg]:
                t_off = t[k_]
            if not act[k_, leg] and t_off is not None:
                age[k_] = t[k_] - t_off
        ag = age[mk]
        big = mag > 1.0
        line = (f"  {'LR'[leg]}: p50 {q(mag,50):.1e}  p99 {q(mag,99):.1e}  max {mag.max() if len(mag) else 0:.1f} N | "
                f"> 1 N: {big.sum()} 틱 ({100 * big.mean() if len(mag) else 0:.1f} %)")
        if big.any():
            line += f", 그때 이륙 후 경과 중앙 {1e3 * np.nanmedian(ag[big]):.0f} ms (max {1e3 * np.nanmax(ag[big]):.0f})"
        print(line + f"   [MPC 재풀이 간격 {1e3 * 7 * dt_tick:.0f} ms — 그 안이면 stale]")

    print("\n5) 토크 포화 틱 (한 관절이라도 |τ| ≥ 한계) 의 접촉 상태")
    ratio = np.abs(tau[win]) / lim[None]
    sat_any = ratio.max(1) >= 1.0
    idx = np.where(win)[0][sat_any]
    if len(idx):
        jmax = np.argmax(ratio[sat_any], axis=1)
        names, counts = np.unique(jmax, return_counts=True)
        top = ", ".join(f"{'LR'[n // 5]} {JN[n % 5]}×{c}" for n, c in sorted(zip(names, counts), key=lambda x: -x[1])[:4])
        legmax = jmax // 5
        nofn = fn[idx, legmax] < 5.0
        nact = ~act[idx, legmax]
        print(f"  {len(idx)} 틱 ({100 * len(idx) / win.sum():.2f} %)  관절: {top}")
        print(f"  포화 관절 다리의 상태: 측정 Fn < 5 N {100 * nofn.mean():.0f} %  |  스윙 (active=0) {100 * nact.mean():.0f} %")
    else:
        print("  없음")


# ---------------------------------------------------------------------------
def run_stand(a):
    cfg, model, ctl, walk_t = walk._setup(0.0, 0.0, 0.0, "osqp", True, "const", False, True,
                                          1.0, True, False, fixes=())
    m, d = model.m, model.d
    n = int(round(a.seconds / m.opt.timestep))
    every = 5                                               # 10 ms
    rows = []
    slip = [[], []]
    foot_geoms = set(np.concatenate(model.foot_geoms)) if hasattr(model, "foot_geoms") else None
    f6 = np.zeros(6)
    for k in range(n):
        tau_leg, tau_arm = ctl.tick()
        model.write_torque(tau_leg, tau_arm)
        mujoco.mj_step(m, d)
        if walk._fell(d):
            print(f"넘어짐 t = {d.time:.2f} s")
            break
        if k % every:
            continue
        s = ctl.state
        rec = dict(t=d.time, u=np.array(ctl.u_hold), fn=np.array(s.foot_normal_force))
        for leg in (0, 1):
            fbdy = model.foot_body[leg]
            Rb = d.xmat[fbdy].reshape(3, 3)
            yaw = np.arctan2(Rb[1, 0], Rb[0, 0])
            site = d.site_xpos[model.foot_site[leg]]
            rec[f"yaw{leg}"] = yaw
            rec[f"site{leg}"] = site.copy()
            rec[f"heel{leg}"] = site - 0.075 * Rb[:, 0]
            rec[f"toe{leg}"] = site + 0.075 * Rb[:, 0]
            rec[f"hipyaw{leg}"] = float(d.qpos[model.leg_qadr[leg, 0]])
        # 접촉점별 마찰 사용률
        for ci in range(d.ncon):
            c = d.contact[ci]
            for gi, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                bid = m.geom_bodyid[gi]
                for leg in (0, 1):
                    if bid == model.foot_body[leg]:
                        mujoco.mj_contactForce(m, d, ci, f6)
                        if f6[0] > 1.0:
                            slip[leg].append(np.hypot(f6[1], f6[2]) / f6[0])
        rows.append(rec)
    T = {k_: np.array([r[k_] for r in rows]) for k_ in rows[0]}
    t = T["t"]
    print(f"\n[stand] {t[-1]:.1f} s  (표본 {len(t)}, 10 ms 간격)")
    print("\n발 위치 드리프트 (시작 1 s 평균 → 마지막 1 s 평균) [mm]")
    a0, a1 = t < t[0] + 1.0, t > t[-1] - 1.0
    for leg in (0, 1):
        for name in ("heel", "toe", "site"):
            p0, p1 = T[f"{name}{leg}"][a0].mean(0), T[f"{name}{leg}"][a1].mean(0)
            dxy = 1e3 * (p1 - p0)
            print(f"  {'LR'[leg]} {name:4s}  Δx {dxy[0]:+7.2f}  Δy {dxy[1]:+7.2f}")
    hh0 = np.abs(T["heel0"][a0, 1] - T["heel1"][a0, 1]).mean()
    hh1 = np.abs(T["heel0"][a1, 1] - T["heel1"][a1, 1]).mean()
    tt0 = np.abs(T["toe0"][a0, 1] - T["toe1"][a0, 1]).mean()
    tt1 = np.abs(T["toe0"][a1, 1] - T["toe1"][a1, 1]).mean()
    print(f"\n  뒤꿈치 사이 y 거리: {1e3 * hh0:.1f} → {1e3 * hh1:.1f} mm  (Δ {1e3 * (hh1 - hh0):+.2f})")
    print(f"  발끝   사이 y 거리: {1e3 * tt0:.1f} → {1e3 * tt1:.1f} mm  (Δ {1e3 * (tt1 - tt0):+.2f})")
    for leg in (0, 1):
        dyaw = np.degrees(T[f"yaw{leg}"][a1].mean() - T[f"yaw{leg}"][a0].mean())
        dhip = np.degrees(T[f"hipyaw{leg}"][a1].mean() - T[f"hipyaw{leg}"][a0].mean())
        print(f"  {'LR'[leg]} 발 yaw Δ {dyaw:+.3f}°   hip_yaw 관절 Δ {dhip:+.3f}°")
    print("\n명령 wrench (MPC, 시간 평균 ± σ)")
    for leg, (ix, iy, iz, mx, my, mz) in ((0, (0, 1, 2, 6, 7, 8)), (1, (3, 4, 5, 9, 10, 11))):
        U = T["u"]
        print(f"  {'LR'[leg]}: Fx {U[:, ix].mean():+6.2f}±{U[:, ix].std():.2f}  Fy {U[:, iy].mean():+6.2f}±{U[:, iy].std():.2f}  "
              f"Fz {U[:, iz].mean():6.1f}  Mx {U[:, mx].mean():+5.2f}  My {U[:, my].mean():+5.2f}  Mz {U[:, mz].mean():+5.2f}")
    for leg in (0, 1):
        sl = np.array(slip[leg])
        if len(sl):
            print(f"  {'LR'[leg]} 접촉점 마찰 사용률 |Ft|/Fn: p50 {q(sl,50):.3f}  p95 {q(sl,95):.3f}  max {sl.max():.3f}  "
                  f"(≥0.95: {100 * (sl > 0.95).mean():.1f} % — 미끄러지는 중)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("walk")
    s.add_argument("--vx", type=float, default=0.6)
    s.add_argument("--vy", type=float, default=0.0)
    s.add_argument("--wz", type=float, default=0.0)
    s.add_argument("--seconds", type=float, default=20.0)
    s = sub.add_parser("stand")
    s.add_argument("--seconds", type=float, default=60.0)
    a = ap.parse_args()
    run_walk(a) if a.cmd == "walk" else run_stand(a)


if __name__ == "__main__":
    main()
