"""MIT 걸음 품질 표 — (명령 × 변형 문자열) 행렬을 process Pool 로 돌리고 표로 찍는다.

지표 (10_walk.compute_metrics, walking 마지막 100 s 창, 짧으면 가능한 창):
    surv       walking 시작 후 버틴 시간 [s]  (120 s 가 기준)
    trk%       명령 축별 달성/명령 평균 (heading frame 평균 속도, yaw rate 는 yaw 변화/시간)
    vx vy wz   달성값
    rollσ pitchσ [rad]
    yaw95      몸통 yaw 진동: yaw − 0.5 s(한 cycle) 이동평균 의 95% 폭 [rad] (명령 yaw 램프/드리프트 제거)
    apex       스윙 발 site 최고 높이 평균 [mm] (목표 55 = −5 + 60)
    tdErr      실제 착지 (estimated contact 상승) − 명목 착지 (scheduled 상승) 평균±σ [ms]
    early      early contact 가 난 스윙 비율
    slip       stance 발 xy 이동 (착지 +10 ms → 이륙) 평균 [mm]
    τ/lim      다리 토크 (clamp 전) / ctrlrange 최대, sat = 한 관절이라도 한계 넘은 틱 비율
    fb cold ms MPC fallback 수, cold start 비율, solve 중앙값 [ms]

실행 (ROOT 에서):
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe MPC/src/mit/11_gait_quality.py \
        --cmd 0.6,0,0 0,0.3,0 0,0,1.3 --var base profile=ramp noarm solver=quadprog markers mu=0.8 --seconds 120
변형 문자열: 'k=v,k=v' (10_walk.parse_variant: solver, arm/noarm, profile, markers, mu, fsm2).
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import importlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np


def _job(args):
    cmd, var, seconds, log = args
    walk = importlib.import_module("10_walk")
    kw = walk.parse_variant(var)
    t0 = time.perf_counter()
    try:
        r = walk.headless(cmd[0], cmd[1], cmd[2], seconds, log=log, variant=var.replace(",", "+").replace("=", ""),
                          verbose=False, **kw)
    except Exception as exc:
        return dict(cmd=cmd, var=var, error=f"{type(exc).__name__}: {exc}", wall=time.perf_counter() - t0)
    r = {k: v for k, v in r.items() if k not in ("record",)}
    r["cmd"] = cmd
    r["var"] = var
    return r


def _f(v, fmt):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        return format(v, fmt)
    except (TypeError, ValueError):
        return str(v)


COLS = [("cmd", 14, None), ("var", 18, None), ("surv", 6, ".1f"), ("trk%", 6, ".0f"), ("vx", 6, "+.2f"),
        ("vy", 6, "+.2f"), ("wz", 6, "+.2f"), ("rollσ", 6, ".3f"), ("pitchσ", 6, ".3f"), ("yaw95", 6, ".3f"),
        ("apex", 5, ".0f"), ("tdErr", 11, None), ("early", 5, ".2f"), ("slip", 5, ".1f"), ("τ/lim", 5, ".2f"),
        ("sat", 5, ".3f"), ("fb", 3, "d"), ("cold", 5, ".2f"), ("ms", 5, ".1f")]


def row(r) -> list[str]:
    c = r["cmd"]
    if r.get("error") and "survived" not in r:
        return [f"{c[0]:g},{c[1]:g},{c[2]:g}", r["var"], "ERR " + r["error"][:60]]
    return [f"{c[0]:g},{c[1]:g},{c[2]:g}", r["var"], _f(r["survived"], ".1f"), _f(r.get("track"), ".0f"),
            _f(r["vx"], "+.2f"), _f(r["vy"], "+.2f"), _f(r["wz"], "+.2f"), _f(r["roll_sd"], ".3f"),
            _f(r["pitch_sd"], ".3f"), _f(r.get("yaw_osc95"), ".3f"), _f(r["apex_mm"], ".0f"),
            f"{_f(r['td_err_ms'], '+.0f')}±{_f(r['td_err_sd_ms'], '.0f')}", _f(r["early_rate"], ".2f"),
            _f(r["slip_mm"], ".1f"), _f(r.get("tau_ratio_max"), ".2f"), _f(r.get("tau_sat_frac"), ".3f"),
            _f(r["fallbacks"], "d"), _f(r["cold_rate"], ".2f"), _f(r["solve_ms_med"], ".1f")]


def print_table(results, markdown=False):
    hdr = [c[0] for c in COLS]
    if markdown:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join("---" for _ in hdr) + "|")
        for r in results:
            print("| " + " | ".join(row(r)) + " |")
        return
    w = [c[1] for c in COLS]
    print("  ".join(h.rjust(n) for h, n in zip(hdr, w)))
    for r in results:
        cells = row(r)
        print("  ".join(s.rjust(n) for s, n in zip(cells, w + [0] * (len(cells) - len(w)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", nargs="+", default=["0.6,0,0", "0,0.3,0", "0,0,1.3"], help="vx,vy,wz")
    ap.add_argument("--var", nargs="+", default=["base"])
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--procs", type=int, default=18)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--md", action="store_true", help="markdown table")
    ap.add_argument("--json", default=None, help="save results json here")
    a = ap.parse_args()
    cmds = [tuple(float(x) for x in c.split(",")) for c in a.cmd]
    jobs = [(c, v, a.seconds, a.log) for v in a.var for c in cmds]
    t0 = time.perf_counter()
    with Pool(min(a.procs, len(jobs))) as pool:
        results = pool.map(_job, jobs, chunksize=1)
    print(f"[11_gait_quality] {len(jobs)} runs, {a.seconds:.0f} s walking each, wall {time.perf_counter() - t0:.0f} s")
    print_table(results, markdown=a.md)
    if a.json:
        Path(a.json).write_text(json.dumps(results, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
                                           indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
