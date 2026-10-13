"""디딤마다 발끝 · 뒤꿈치 접촉 순서와 CoP 이동 (Q&A 10/6 Q58). jitter_diag.py 결과 (.npz) 를 읽는다.
실제 접촉 구간 (발끝 · 뒤꿈치 중 하나라도 수직력 > 5 N 인 연속 구간, 착지 ~ 마지막 이륙) 마다:
  착지: 뒤꿈치 · 발끝 처음 닿은 시각 차 (발끝 − 뒤꿈치, + = 뒤꿈치 먼저) [ms]
  이륙: 마지막으로 떨어진 시각 차 (발끝 − 뒤꿈치, + = 발끝이 나중 = 발끝으로 떼기) [ms]
  디딤 중 발끝만 · 뒤꿈치만 닿은 비율 [%], 디딤 10 등분 칸별 실제 CoP x/l (두 점 힘, + 발끝)
걷기 시작 + 3 s 이후, 20 ms 보다 짧은 접촉 (되튐) 은 뺀다.
사용: contact_seq.py a.npz [b.npz ...]
"""
import sys
import numpy as np

TH = 5.0
for f in sys.argv[1:]:
    z = np.load(f); R = z["rows"]; dt = float(z["dt"]); wt = float(z["walk_t"]); fell = float(z["fell"])
    t = R[:, 0]
    td, lo, toe_only, heel_only, cop = [], [], [], [], []
    for leg in (0, 1):
        b = 2 + 21 * leg
        fh, ft = R[:, b + 2], R[:, b + 3]
        on = (fh > TH) | (ft > TH)
        i = 1
        while i < len(t):
            if on[i] and not on[i - 1] and t[i] > wt + 3.0:
                j = i
                while j < len(t) and (on[j] or np.any(on[j:j + int(0.01 / dt)])):   # 10 ms 안 되튐은 한 구간으로
                    j += 1
                if j >= len(t):
                    break
                if (j - i) * dt > 0.10:
                    seg_h, seg_t = fh[i:j] > TH, ft[i:j] > TH
                    ih = np.flatnonzero(seg_h); it = np.flatnonzero(seg_t)
                    if len(ih) and len(it):
                        td.append((it[0] - ih[0]) * dt * 1e3)
                        lo.append((it[-1] - ih[-1]) * dt * 1e3)
                    toe_only.append(np.mean(seg_t & ~seg_h)); heel_only.append(np.mean(seg_h & ~seg_t))
                    c = (ft[i:j] - fh[i:j]) / np.maximum(ft[i:j] + fh[i:j], 1e-6)
                    n = j - i
                    cop.append([np.mean(c[int(k * n / 10):max(int((k + 1) * n / 10), int(k * n / 10) + 1)]) for k in range(10)])
                i = j
            i += 1
    td, lo = np.array(td), np.array(lo); C = np.array(cop)
    print(f"[{f.split('/')[-1]}] {'넘어짐 %.1f s' % fell if fell > 0 else '버팀'}  디딤 {len(C)}")
    print(f"   착지 (발끝 − 뒤꿈치) 중앙 {np.median(td):+.0f} ms (p10 {np.percentile(td, 10):+.0f} · p90 {np.percentile(td, 90):+.0f})"
          f"   이륙 (발끝 − 뒤꿈치) 중앙 {np.median(lo):+.0f} ms (p10 {np.percentile(lo, 10):+.0f} · p90 {np.percentile(lo, 90):+.0f})")
    print(f"   발끝만 {100 * np.mean(toe_only):.0f} % · 뒤꿈치만 {100 * np.mean(heel_only):.0f} %   실제 CoP x/l (디딤 0 → 100 %): "
          + " ".join(f"{v:+.2f}" for v in C.mean(0)))
