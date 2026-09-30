"""line_diag.py 결과 요약 — 디딤 구간에서 MPC 명령 / 실제 접촉의 CoP · 비틀림 사용률, line 행 활성 비율 (9/30 Q11).

사용: line_ana.py a.npz [b.npz ...] [--skip 3] [--last 1.5] [--l 0.065] [--lphys 0.075]
  위반 v = max(|Mz + l·Fy| − µ(l·Fz − My), |l·Fy − Mz| − µ(l·Fz + My)) / (l·Fz)
       (발끝 · 뒤꿈치 점 y 마찰 — line 행 8~11). 0 이하면 두 점 모델로 낼 수 있는 wrench, 양수면 넘침.
  MPC 는 l = foot_half_length, 실제는 원기둥 반길이 lphys. CoP 끝 = |−My/Fz| > 0.98·l (발끝 / 뒤꿈치).
"""
import sys
import numpy as np

args = sys.argv[1:]
opt = {"--skip": 3.0, "--last": 1.5, "--l": 0.065, "--lphys": 0.075, "--mu": 1.0}
files = []
i = 0
while i < len(args):
    if args[i] in opt:
        opt[args[i]] = float(args[i + 1]); i += 2
    else:
        files.append(args[i]); i += 1


def use(F, M, l, mu):
    Fy, Fz, My, Mz = F[:, 1], F[:, 2], M[:, 1], M[:, 2]
    Fzp = np.maximum(Fz, 1e-6)
    v = np.maximum(np.abs(Mz + l * Fy) - mu * (l * Fz - My), np.abs(l * Fy - Mz) - mu * (l * Fz + My)) / (l * Fzp)
    return v, -My / Fzp, M[:, 0] / Fzp


def block(R, sel, tag, l, lp, mu):
    out = []
    for leg in (0, 1):
        o = 5 + 14 * leg
        st = sel & (R[:, o] > 0.5) & (R[:, o + 3] > 20.0)
        Fm, Mm = R[st, o + 1:o + 4], R[st, o + 4:o + 7]
        Fa, Ma = R[st, o + 7:o + 10], R[st, o + 10:o + 13]
        ok = Fa[:, 2] > 20.0                                   # 실제 접촉이 실린 틱만 (착지·이륙 과도 제외)
        vm, copm, _ = use(Fm, Mm, l, mu)
        va, copa, mxa = use(Fa[ok], Ma[ok], lp, mu)
        out.append(f"  {tag} {'LR'[leg]} n{st.sum():5d} | MPC: 위반 p95 {np.percentile(vm, 95):+.2f} >0.05 {np.mean(vm > 0.05) * 100:3.0f}%  "
                   f"CoP 발끝 {np.mean(copm > 0.98 * l) * 100:3.0f}% 뒤 {np.mean(copm < -0.98 * l) * 100:3.0f}%  "
                   f"|Mz|/Fz p95 {np.percentile(np.abs(Mm[:, 2]) / Fm[:, 2], 95):.3f}  |Fy|/Fz p50 {np.median(np.abs(Fm[:, 1]) / Fm[:, 2]):.2f} p95 {np.percentile(np.abs(Fm[:, 1]) / Fm[:, 2], 95):.2f}"
                   f" | 실제: 위반 p95 {np.percentile(va, 95):+.2f}  CoP 발끝 {np.mean(copa > 0.98 * lp) * 100:3.0f}% 뒤 {np.mean(copa < -0.98 * lp) * 100:3.0f}%  "
                   f"|Mz|/Fz p95 {np.percentile(np.abs(Ma[ok, 2]) / Fa[ok, 2], 95):.3f}  |Fy|/Fz p95 {np.percentile(np.abs(Fa[ok, 1]) / Fa[ok, 2], 95):.2f}  |Mx|/Fz p95 {np.percentile(np.abs(mxa), 95):.4f}")
    return out


for f in files:
    z = np.load(f)
    R = z["rows"]; fell = float(z["fell"]); wt = float(z["walk_t"])
    t = R[:, 0]
    print(f"{f}  fell {fell:.2f} (걷기 시작 {wt:.2f}, 걸은 시간 {t[-1] - wt:.1f} s)  prim_res p95 {np.nanpercentile(R[t > wt, 4], 95):.2e}")
    steady = (t > wt + opt["--skip"]) & (R[:, 1] > 0.5)
    if fell > 0:
        steady &= t < fell - opt["--last"] - 0.5
    for s in block(R, steady, "정상", opt["--l"], opt["--lphys"], opt["--mu"]):
        print(s)
    if fell > 0:
        last = t > fell - opt["--last"]
        for s in block(R, last, f"마지막 {opt['--last']:.1f}s", opt["--l"], opt["--lphys"], opt["--mu"]):
            print(s)
        print(f"  마지막 yaw 오차 {R[last, 2].min():+.3f} ~ {R[last, 2].max():+.3f} rad, yaw rate {R[last, 3].min():+.2f} ~ {R[last, 3].max():+.2f}")
    print(f"  정상 yaw 오차 p95 |.| {np.nanpercentile(np.abs(R[steady, 2]), 95):.3f} rad")
