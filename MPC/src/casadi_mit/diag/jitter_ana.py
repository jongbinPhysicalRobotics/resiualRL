"""jitter_diag.py 결과 요약 (9/30 Q12). 사용: jitter_ana.py a.npz [b.npz ...] [--skip 3]

딛는 구간 (controller stance, 착지 + 10 ms ~ 이륙) 마다:
  slip       발 site xy 순이동 [mm] (11_gait_quality 의 slip 과 같은 정의)
  경로 뒤/앞  그 점에 수직력 > 5 N 인 동안 접촉점 미끄럼 속도 |v| 를 적분 [mm] — 떨며 왔다 갔다 한 거리까지 포함
  순이동 뒤/앞 같은 동안 v 벡터 적분의 크기 [mm]
  켜짐 전환  뒤 · 앞 점의 (수직력 > 5 N) 켜짐/꺼짐 바뀐 횟수 / 딛는 시간 [회/s]
떨림 (딛는 틱 전체):
  ω 고주파   발 각속도 (발 yaw 좌표 x · y, 월드 z) − 20 ms 이동평균 의 RMS [rad/s]  (약 25 Hz 이상 성분)
  τ 고주파   다리 관절 토크 − 20 ms 이동평균 의 RMS [N·m] (관절 0 ~ 4)
  명령 변화  MPC Mz 명령이 바뀐 틱에서 |ΔMz| 의 RMS [N·m]
"""
import sys
import numpy as np

args = sys.argv[1:]
skip = 3.0
if "--skip" in args:
    i = args.index("--skip"); skip = float(args[i + 1]); del args[i:i + 2]


def hp(x, k=10):
    ker = np.ones(k) / k
    lo = np.apply_along_axis(lambda c: np.convolve(c, ker, mode="same"), 0, x)
    return x - lo


for f in args:
    z = np.load(f)
    R = z["rows"]; wt = float(z["walk_t"]); dt = float(z["dt"]); fell = float(z["fell"])
    t = R[:, 0]
    win = (t > wt + skip) & (R[:, 1] > 0.5)
    print(f"{f.split('/')[-1]}  fell {fell:.1f}")
    for leg in (0, 1):
        o = 2 + 21 * leg
        act = (R[:, o] > 0.5) & win
        edges = np.flatnonzero(np.diff(act.astype(int)))
        starts = [e + 1 for e in edges if act[e + 1]]
        slips, path, net, tog, dur, dyaw = [], [[], []], [[], []], [], [], []
        for s0 in starts:
            e = s0
            while e < len(act) and act[e]:
                e += 1
            a = s0 + 5
            if e - a < 20:
                continue
            slips.append(1e3 * np.linalg.norm(R[e - 1, o + 4:o + 6] - R[a, o + 4:o + 6]))
            for k in (0, 1):
                on = R[a:e, o + 2 + k] > 5.0
                vv = R[a:e, o + 9 + 2 * k:o + 11 + 2 * k]
                vv = np.where(on[:, None] & np.isfinite(vv), vv, 0.0)
                path[k].append(1e3 * dt * np.sum(np.linalg.norm(vv, axis=1)))
                net[k].append(1e3 * dt * np.linalg.norm(vv.sum(0)))
            on2 = R[a:e, o + 2:o + 4] > 5.0
            tog.append(np.sum(np.abs(np.diff(on2.astype(int), axis=0))))
            dur.append((e - a) * dt)
        st = act & (R[:, o + 3] + R[:, o + 2] > 5.0)
        w = R[:, o + 6:o + 9]
        whp = hp(w)[st]
        tau_hp = hp(R[:, o + 16:o + 21])[st]
        mz = R[st, o + 15]; dmz = np.diff(mz); dmz = dmz[np.abs(dmz) > 1e-9]
        print(f"  {'LR'[leg]} 걸음 {len(slips):3d} | slip {np.mean(slips):5.1f} mm | 경로 뒤/앞 {np.mean(path[0]):5.1f}/{np.mean(path[1]):5.1f}"
              f"  순이동 {np.mean(net[0]):5.1f}/{np.mean(net[1]):5.1f} mm | 켜짐 전환 {np.sum(tog) / np.sum(dur):5.1f} 회/s"
              f" | ω고주파 x/y/z {np.sqrt(np.mean(whp ** 2, 0)).round(2)} | τ고주파 {np.sqrt(np.mean(tau_hp ** 2, 0)).round(1)}"
              f" | ΔMz rms {np.sqrt(np.mean(dmz ** 2)) if len(dmz) else 0:.2f}")
