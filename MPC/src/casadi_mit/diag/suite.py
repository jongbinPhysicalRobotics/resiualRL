"""기준표 · A/B 묶음 실행기 (Q&A 10/6 Q49). 작업 줄: 'label|vx,vy,wz|stand(0/1)|seconds|설정' (설정은 diag/configs.resolve 문법).
사용:
  suite.py run jobs.txt out.jsonl procs            작업 파일을 Pool 로 돌려 한 줄씩 jsonl 에 쓴다
  suite.py jobs CONFIG LABEL [std|ext|all] > f     표준 작업 목록을 만든다
       std = 주행 22 (120 s) + 서기 외란 30 · 40 N 8 개 (12 s) + 걷기 옆 외란 ±40 · ±50 N × 위상 8 (20 s)  — Q43 기준표와 같음
       ext = 서기 외란 50 N 4 개 + 걷기 앞뒤 외란 ±40 · ±50 N × 위상 8  (교수님 0 ~ 50 N 지시의 빈칸, Q49)
  suite.py table out.jsonl LABEL1 LABEL2 ...       비교표
외란: 몸통 CoM 에 진행 방향 기준 수평력 0.2 s. 서기는 3 s, 걷기 (0.6 m/s) 는 걷기 시작 뒤 10 s + 위상 k × (주기 0.55 / 8).
"""
import os, sys, json
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "1"
import importlib.util
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from configs import resolve

MAIN = [("vx0.6", "0.6,0,0", ""), ("vx0.7", "0.7,0,0", ""), ("vx0.8", "0.8,0,0", ""), ("vx0.9", "0.9,0,0", ""), ("vx1.0", "1.0,0,0", ""),
        ("vx-1.0", "-1.0,0,0", ""), ("vy0.3", "0,0.3,0", ""), ("vy+0.5", "0,0.5,0", ""), ("vy-0.5", "0,-0.5,0", ""), ("vy0.8", "0,0.8,0", ""),
        ("wz1.3L0", "0,0,1.3", ",yawlead=0"), ("wz1.3L1", "0,0,1.3", ",yawlead=1"), ("wz1.5L0", "0,0,1.5", ",yawlead=0"),
        ("wz1.5L1", "0,0,1.5", ",yawlead=1"), ("wz2.5L0", "0,0,2.5", ",yawlead=0"), ("wz2.5L.5", "0,0,2.5", ",yawlead=0.5"),
        ("wz2.5L1", "0,0,2.5", ",yawlead=1"), ("0.3+0.5", "0.3,0,0.5", ""), ("0.6+0.5", "0.6,0,0.5", ""), ("0.7+0.5", "0.7,0,0.5", ""),
        ("0.3+1.0", "0.3,0,1.0", ""), ("0.6+1.0", "0.6,0,1.0", "")]
KEEP = ("pass", "survived", "vx", "vy", "wz", "track", "slip_mm", "apex_mm", "tau_ratio_max", "tau_sat_frac", "yaw_osc95",
        "load_RL", "pitch_sd", "roll_sd", "fric_cone_p95")


def make_jobs(cfg, label, which="std"):
    out = []
    if which in ("std", "all"):
        out += [f"{label}:{n}|{c}|0|120|{cfg}{e}" for n, c, e in MAIN]
        for f in (30, 40):
            for nm, fx, fy in (("f", f, 0), ("b", -f, 0), ("l", 0, f), ("r", 0, -f)):
                out.append(f"{label}:stand_{nm}{f}|0,0,0|1|12|{cfg},push={fx}:{fy}:3:0.2")
        for f in (40, -40, 50, -50):
            for k in range(8):
                out.append(f"{label}:wp{f:+d}_{k}|0.6,0,0|0|20|{cfg},push=0:{f}:{10.0 + k * 0.06875:.4f}:0.2")
    if which in ("ext", "all"):
        for nm, fx, fy in (("f", 50, 0), ("b", -50, 0), ("l", 0, 50), ("r", 0, -50)):
            out.append(f"{label}:stand_{nm}50|0,0,0|1|12|{cfg},push={fx}:{fy}:3:0.2")
        for f in (40, -40, 50, -50):
            for k in range(8):
                out.append(f"{label}:wpx{f:+d}_{k}|0.6,0,0|0|20|{cfg},push={f}:0:{10.0 + k * 0.06875:.4f}:0.2")
    return out


def job(line):
    spec = importlib.util.spec_from_file_location("w10", str(_HERE.parent / "10_walk.py"))
    eng = importlib.util.module_from_spec(spec); spec.loader.exec_module(eng)
    label, cmd, stand, sec, cfg = line.split("|", 4)
    vx, vy, wz = (float(x) for x in cmd.split(","))
    kw = eng.parse_variant(resolve(cfg)); kw["stand"] = stand == "1"
    r = eng.headless(vx, vy, wz, float(sec), verbose=False, **kw)
    keep = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in r.items() if k in KEEP}
    keep.update(label=label, cmd=cmd, cfg=cfg)
    return keep


def table(path, labels):
    D = {}
    for l in open(path):
        r = json.loads(l); D[tuple(r["label"].split(":", 1))] = r
    conds = [n for n, _, _ in MAIN]
    print(f"{'명령':9s}" + "".join(f" | {v:^32s}" for v in labels))
    for c in conds:
        if not any((v, c) in D for v in labels):
            continue
        line = f"{c:9s}"
        for v in labels:
            r = D.get((v, c))
            if r is None:
                line += f" | {'-':^32s}"; continue
            st = "✓" if r["pass"] else f"✗{r['survived']:.0f}s"
            line += f" | {st:>5s} trk{r['track']:4.0f} slp{r['slip_mm']:5.1f} sat{100 * r['tau_sat_frac']:5.1f}%"
        print(line)
    for v in labels:
        main = [D.get((v, c)) for c in conds]
        groups = defaultdict(lambda: [0, 0])
        for (vv, c), r in D.items():
            if vv != v or c in conds or not (c.startswith("wp") or c.startswith("stand_")):   # 표준 밖 라벨 (고속 등) 은 건너뜀
                continue
            g = c.split("_")[0] if c.startswith("wp") else ("stand_" + c.split("_")[1][1:])
            groups[g][1] += 1; groups[g][0] += int(bool(r["pass"]))
        stands = " ".join(f"{c.split('_')[1]}{'✓' if D[(vv, c)]['pass'] else '✗'}" for (vv, c) in sorted(D) if vv == v and c.startswith("stand"))
        print(f"{v:6s}: 주행 {sum(bool(r and r['pass']) for r in main)}/{sum(r is not None for r in main)} | 서기 {stands} | "
              + " ".join(f"{g} {a}/{b}" for g, (a, b) in sorted(groups.items()) if g.startswith("wp")))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        lines = [l.strip() for l in open(sys.argv[2]) if l.strip()]
        with Pool(int(sys.argv[4])) as pool, open(sys.argv[3], "a") as fo:
            for res in pool.imap_unordered(job, lines, chunksize=1):
                fo.write(json.dumps(res, ensure_ascii=False) + "\n"); fo.flush()
    elif cmd == "jobs":
        print("\n".join(make_jobs(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "std")))
    elif cmd == "table":
        table(sys.argv[2], sys.argv[3:])
