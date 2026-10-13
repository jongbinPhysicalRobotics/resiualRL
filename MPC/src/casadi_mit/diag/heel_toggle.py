"""jitter_diag.py 결과 → 디딤마다 뒤꿈치 · 발끝 켜짐 전환 (걸음당), 발끝만 딛는 비율, slip, 뒤꿈치 전환이 일어난 디딤 구간 (Q&A 10/6 Q30 ~ Q39 의 표).
디딤 = controller 디딤 (active) 의 착지 + 10 ms ~ 이륙, 걷기 시작 + 3 s 이후. 접촉점 켜짐 = 수직력 > 5 N.
사용: heel_toggle.py a.npz [b.npz ...]
"""
import sys
import numpy as np

TH = 5.0
print(f"{'파일':34s} 넘어짐  뒤꿈치/걸음 발끝/걸음  발끝만[%]  slip[mm]  뒤꿈치 전환 위치 (디딤 10 % 칸별 걸음당)")
for f in sys.argv[1:]:
    z = np.load(f); R = z["rows"]; dt = float(z["dt"]); wt = float(z["walk_t"]); fell = float(z["fell"])
    t = R[:, 0]
    nh = nt = ns = 0; toe_only = tot = 0; slip = []; bins = np.zeros(10)
    for leg in (0, 1):
        b = 2 + 21 * leg
        act = R[:, b] > 0.5; fh = R[:, b + 2] > TH; ft = R[:, b + 3] > TH; xy = R[:, b + 4:b + 6]
        i = 1
        while i < len(t):
            if act[i] and not act[i - 1] and t[i] > wt + 3.0:
                j = i
                while j < len(t) and act[j]:
                    j += 1
                if j >= len(t):
                    break
                a = i + int(round(0.010 / dt))
                if j - a > 10:
                    dh = np.flatnonzero(np.diff(fh[a:j].astype(int)) != 0)
                    dtt = np.flatnonzero(np.diff(ft[a:j].astype(int)) != 0)
                    nh += len(dh); nt += len(dtt); ns += 1
                    for k in dh:
                        bins[min(int(10 * k / (j - a)), 9)] += 1
                    on = fh[a:j] | ft[a:j]
                    toe_only += np.sum(ft[a:j] & ~fh[a:j]); tot += np.sum(on)
                    slip.append(1e3 * np.linalg.norm(xy[j - 1] - xy[a]))
                i = j
            i += 1
    ns = max(ns, 1)
    print(f"{f.split('/')[-1]:34s} {fell:6.1f}  {nh / ns:9.2f} {nt / ns:9.2f}  {100 * toe_only / max(tot, 1):8.0f}  "
          f"{np.mean(slip) if slip else float('nan'):7.1f}   " + " ".join(f"{v / ns:.1f}" for v in bins))
