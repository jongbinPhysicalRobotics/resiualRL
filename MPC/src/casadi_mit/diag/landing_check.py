"""착지 · 스윙 추종 점검 (Q&A 10/6 Q63, 스윙 감쇠 kdz 부작용). 명령 4 개 × 설정들, 20 s, 걷기 + 4 s 부터:
  착지 오차 (실제 발 − 6 ms 전 목표, 수평) · 착지 순간 발 속도 (z · 수평) · 착지 뒤 30 ms 최대 수직력 · 스윙 중 발 − 궤적 목표 거리.
사용: landing_check.py A38 A39 ...   (diag/configs.resolve 문법)
"""
import os, sys, json
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "1"
import importlib.util
import numpy as np
from multiprocessing import Pool
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/diag")
from configs import resolve

def job(a):
    lab, cfg, vx, vy, wz = a
    spec = importlib.util.spec_from_file_location("w", ROOT + "/10_walk.py")
    W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
    kw = W.parse_variant(resolve(cfg))
    r = W.headless(vx, vy, wz, 20.0, verbose=False, keep_record=True, **kw)
    R = r["record"]; t = R["t"]; act = R["active"] > 0.5; dt = float(np.median(np.diff(t)))
    walk_t = float(t[np.flatnonzero(R["walking"] > 0.5)[0]])
    err, vtd, fpk, trk = [], [], [], []
    for leg in (0, 1):
        a_ = act[:, leg]
        for i in np.flatnonzero(a_[1:] & ~a_[:-1]) + 1:
            if t[i] < walk_t + 4.0 or i < 5 or i + int(0.03 / dt) >= len(t):
                continue
            err.append(np.linalg.norm(R["foot"][i, leg, :2] - R["desired"][i - 3, leg, :2]))
            vtd.append((R["foot"][i, leg] - R["foot"][i - 2, leg]) / (2 * dt))
            fpk.append(R["fn"][i:i + int(0.03 / dt), leg].max())
        sw = ~a_ & (t > walk_t + 4.0)
        d = R["foot"][sw, leg] - R["pdes"][sw, leg]
        trk.append(np.linalg.norm(d[np.all(np.isfinite(d), axis=1)], axis=1))
    vtd = np.array(vtd); trk = np.concatenate(trk)
    return dict(lab=lab, cmd=f"{vx},{vy},{wz}", ok=bool(r["pass"]), err=1e3 * np.median(err), err95=1e3 * np.percentile(err, 95),
                vz=np.median(vtd[:, 2]), vxy=np.median(np.linalg.norm(vtd[:, :2], axis=1)), fpk=np.median(fpk), fpk95=np.percentile(fpk, 95),
                trk=1e3 * np.median(trk), trk95=1e3 * np.percentile(trk, 95))

if __name__ == "__main__":
    cmds = [(1.0, 0, 0), (-1.0, 0, 0), (0, -0.5, 0), (0.6, 0, 1.0)]
    jobs = [(lab, lab, *c) for c in cmds for lab in sys.argv[1:]]
    with Pool(8) as p:
        res = p.map(job, jobs)
    print("명령        설정  착지오차 중앙/p95 mm   착지 발속도 z / 수평 m/s   착지 30 ms 최대 Fz 중앙/p95 N   스윙 추종 오차 중앙/p95 mm")
    for x in res:
        print(f"{x['cmd']:11s} {x['lab']}  {x['err']:5.1f} / {x['err95']:5.1f}        {x['vz']:+.2f} / {x['vxy']:.2f}              {x['fpk']:4.0f} / {x['fpk95']:4.0f}                {x['trk']:5.1f} / {x['trk95']:5.1f}   {'✓' if x['ok'] else '✗'}")
