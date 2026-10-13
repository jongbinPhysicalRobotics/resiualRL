"""고속 직진에서 무엇이 막나 (Q&A 10/6 Q46). headless 기록 (keep_record) 으로:
  - 1 s 평균 실측 전진 속도 (명령 대비)
  - 관절별 토크 포화 비율 — 디딤 / 스윙, 전체 창과 넘어지기 전 2 s
  - 넘어지기 전 1 s: roll · pitch · 높이, 두 발 옆 간격 최소값 (발 꼬임), MPC Fz 최대 · 최소
  - 착지 순간 (active 0→1): 실제 발 − 6 ms 전 목표 착지점 (앞뒤 · 옆), 목표 착지점의 hip 기준 거리 (다리 길이 0.5455 m 대비)
사용: speed_fail.py vx seconds variant
"""
import sys, importlib.util
import numpy as np
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent          # casadi_mit
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "diag"))
from configs import resolve
spec = importlib.util.spec_from_file_location("walk", str(_ROOT / "10_walk.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
NAMES = ["hip_yaw", "hip_abad", "hip_pitch", "knee", "ankle"]
LEG_LEN = 0.267 + 0.2785


def run(vx, seconds, var):
    kw = W.parse_variant(resolve(var))
    r = W.headless(vx, 0.0, 0.0, seconds, verbose=False, keep_record=True, **kw)
    R = r["record"]
    pos = dict(solver="osqp", armature=True, profile="const", markers=False, stand=False, friction_scale=1.0, fsm_double_update=True)
    kw2 = W.parse_variant(resolve(var))
    for k in list(kw2):
        if k in pos:
            pos[k] = kw2.pop(k)
    _, model, _, walk_t = W._setup(0, 0, 0, *pos.values(), False, **kw2)
    lim = model.ctrl_hi[model.leg_act.ravel()]
    t = R["t"]; act = R["active"] > 0.5; ratio = np.abs(R["tau"]) / lim[None, :]
    t_end = float(t[-1])
    surv = float(r["survived"])
    print(f"vx {vx}: {'완주' if r['pass'] else f'넘어짐 {surv:.1f} s'}  추종 {r['track']:.0f} %  slip {r['slip_mm']:.1f} mm")
    # 1. 실측 속도 (몸 yaw 기준 전진, 1 s 평균)
    v = R["x0"][:, 9:11]; yaw = R["rpy"][:, 2]
    vf = v[:, 0] * np.cos(yaw) + v[:, 1] * np.sin(yaw)
    secs = np.arange(walk_t, t_end, 1.0)
    print("  1 s 평균 실측 전진 속도:", " ".join(f"{vf[(t >= a) & (t < a + 1)].mean():.2f}" for a in secs[:min(len(secs), 25)]))
    print("  명령 (필터 뒤):          ", " ".join(f"{R['cmd'][(t >= a) & (t < a + 1), 0].mean():.2f}" for a in secs[:min(len(secs), 25)]))
    # 2. 관절 포화
    for nm, win in (("걷기 4 s 뒤 전체", t > walk_t + 4.0), ("넘어지기 전 2 s", t > t_end - 2.0)):
        line = []
        for leg in (0, 1):
            for j in range(5):
                c = leg * 5 + j
                st = win & act[:, leg]; sw = win & ~act[:, leg]
                s_st = (ratio[st, c] >= 1).mean() if st.any() else 0.0
                s_sw = (ratio[sw, c] >= 1).mean() if sw.any() else 0.0
                if s_st > 0.01 or s_sw > 0.01:
                    line.append(f"{'LR'[leg]}.{NAMES[j]} 디딤 {100*s_st:.0f}% 스윙 {100*s_sw:.0f}%")
        print(f"  포화 ({nm}): " + (" · ".join(line) if line else "없음 (≤ 1 %)"))
    # 3. 넘어지기 전 1 s
    w1 = t > t_end - 1.0
    rpy = R["rpy"][w1]; z = R["base"][w1, 2]
    lat = []
    for i in np.flatnonzero(w1):
        yv = np.array([-np.sin(yaw[i]), np.cos(yaw[i])])
        lat.append((R["foot"][i, 0, :2] - R["foot"][i, 1, :2]).dot(yv))
    fz = np.maximum(R["u"][w1, 2], R["u"][w1, 5])
    print(f"  넘어지기 전 1 s: roll {np.degrees(rpy[:,0]).min():+.0f} ~ {np.degrees(rpy[:,0]).max():+.0f}°, pitch {np.degrees(rpy[:,1]).min():+.0f} ~ {np.degrees(rpy[:,1]).max():+.0f}°,"
          f" 몸통 z {z.min():.3f} ~ {z.max():.3f} m, 두 발 옆 간격 (왼 − 오른) 최소 {1e3*min(lat):.0f} mm, MPC Fz 최대 {fz.max():.0f} N")
    # 4. 착지 오차 · 착지점 거리
    rows = []
    for leg in (0, 1):
        a = act[:, leg]
        for i in np.flatnonzero(a[1:] & ~a[:-1]) + 1:
            if t[i] < walk_t + 4.0:
                continue
            yv = np.array([-np.sin(yaw[i]), np.cos(yaw[i])]); hv = np.array([np.cos(yaw[i]), np.sin(yaw[i])])
            e = R["foot"][i, leg, :2] - R["desired"][i - 3, leg, :2]   # 착지 틱에는 목표가 실제 발 위치로 바뀌므로 6 ms 전 목표와 비교
            hip = R["base"][i] - np.array([0.0, 0.0, 0.132])   # 근사: hip pitch 축 ≈ 몸통 원점 아래 0.132 m (hip_yaw · abad 오프셋 합)
            dpos = R["desired"][i - 3, leg] - hip
            reach = float(np.hypot(dpos[:2].dot(hv), dpos[2]))      # 시상면 hip → 목표 착지점 거리
            run_ = np.flatnonzero(np.diff(np.r_[1, a[:i].astype(int)]) == -1)   # 이륙 틱 (1 → 0) 들
            t_sw = t[i] - t[run_[-1]] if len(run_) else np.nan
            foot_fwd = (R["foot"][i, leg, :2] - R["base"][i, :2]).dot(hv)
            rows.append([t[i], e.dot(hv), e.dot(yv), dpos[:2].dot(hv), reach, leg, t_sw, foot_fwd, R["rpy"][i, 1]])
    if rows:
        A = np.array(rows)
        last = A[A[:, 0] > t_end - 2.0]
        print(f"  착지 (전체 {len(A)} 걸음): 실제 − 목표 앞뒤 {1e3*np.median(A[:,1]):+.0f} mm (p95 |·| {1e3*np.percentile(np.abs(A[:,1]),95):.0f}),"
              f" 옆 {1e3*np.median(A[:,2]):+.0f} mm · 목표 착지점 몸통 앞 {1e3*np.median(A[:,3]):+.0f} mm, 다리 뻗음 {np.median(A[:,4]):.3f} m (다리 길이 {LEG_LEN:.3f})")
        if len(last):
            print(f"  착지 (넘어지기 전 2 s, {len(last)} 걸음): 앞뒤 오차 " + " ".join(f"{1e3*x:+.0f}" for x in last[:, 1]) +
                  " mm · 목표 앞 " + " ".join(f"{1e3*x:+.0f}" for x in last[:, 3]) + " mm")
            print("    다리 · 스윙 시간 [ms] · 실제 발 몸통 앞 [mm] · 착지 순간 pitch [°]: " +
                  "  ".join(f"{'LR'[int(r_[5])]} {1e3*r_[6]:.0f} {1e3*r_[7]:+.0f} {np.degrees(r_[8]):+.0f}" for r_ in last))


if __name__ == "__main__":
    run(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
