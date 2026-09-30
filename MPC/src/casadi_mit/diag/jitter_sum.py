"""jitter_diag 결과 여러 개를 한 줄씩 (두 다리 평균): slip, 접촉점 미끄럼 경로, 켜짐 전환, 발 ω 고주파 roll · pitch (9/30 Q15).
사용: jitter_sum.py a.npz [b.npz ...] [--skip 3]"""
import sys
import numpy as np

args = sys.argv[1:]
skip = 3.0
if "--skip" in args:
    i = args.index("--skip"); skip = float(args[i + 1]); del args[i:i + 2]


def hp(x, k=10):
    ker = np.ones(k) / k
    return x - np.apply_along_axis(lambda c: np.convolve(c, ker, mode="same"), 0, x)


print(f"{'파일':28s} slip[mm] 경로[mm/걸음] 켜짐전환[회/s] ω고주파 roll·pitch[rad/s]")
for f in args:
    z = np.load(f)
    R = z["rows"]; wt = float(z["walk_t"]); dt = float(z["dt"])
    t = R[:, 0]
    win = (t > wt + skip) & (R[:, 1] > 0.5)
    slips, paths, tog, dur, hfs = [], [], 0, 0.0, []
    for leg in (0, 1):
        o = 2 + 21 * leg
        act = (R[:, o] > 0.5) & win
        starts = [e + 1 for e in np.flatnonzero(np.diff(act.astype(int))) if act[e + 1]]
        for s0 in starts:
            e = s0
            while e < len(act) and act[e]:
                e += 1
            a = s0 + 5
            if e - a < 20:
                continue
            slips.append(1e3 * np.linalg.norm(R[e - 1, o + 4:o + 6] - R[a, o + 4:o + 6]))
            p = 0.0
            for k in (0, 1):
                on = R[a:e, o + 2 + k] > 5.0
                vv = R[a:e, o + 9 + 2 * k:o + 11 + 2 * k]
                vv = np.where(on[:, None] & np.isfinite(vv), vv, 0.0)
                p += 1e3 * dt * np.sum(np.linalg.norm(vv, axis=1))
            paths.append(p)
            on2 = R[a:e, o + 2:o + 4] > 5.0
            tog += np.sum(np.abs(np.diff(on2.astype(int), axis=0))); dur += (e - a) * dt
        st = act & (R[:, o + 3] + R[:, o + 2] > 5.0)
        hfs.append(np.sqrt(np.mean(hp(R[:, o + 6:o + 8])[st] ** 2, 0)))
    hf = np.mean(hfs, 0)
    print(f"{f.split('/')[-1]:28s} {np.mean(slips):6.1f}   {np.mean(paths):6.1f}       {tog / dur:5.1f}        {hf[0]:.2f} · {hf[1]:.2f}")
