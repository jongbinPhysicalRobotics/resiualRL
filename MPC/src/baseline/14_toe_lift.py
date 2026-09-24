"""착지 직후 발 앞부분이 들리는 현상 계측 — 발가락/뒤꿈치 접촉구를 따로 본다.

스케줄 착지 시각을 t=0 으로 여러 스텝을 정렬해 평균 (Q&A 9/21 Q8).
  - 발가락 Fz / 뒤꿈치 Fz      : 한쪽 하중이 0 이 되는가
  - 발 pitch, 발가락·뒤꿈치 높이 : 언제 얼마나 들리는가
  - 명령 CoP x = −(my + h·Fx)/Fz : 한계(+12 / −5 cm)에 꽂히는가, 점프하는가

사용:
  .venv/Scripts/python.exe MPC/src/baseline/14_toe_lift.py --vx 0.5 --seconds 20
"""
from __future__ import annotations
import sys, argparse, importlib
from pathlib import Path
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_model, mpc_srb
walk = importlib.import_module("09_walk")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--settle", type=float, default=6.0)
    ap.add_argument("--sidew", type=float, default=13.0, help="공칭 보폭 [cm]")
    a = ap.parse_args()

    m, d = g1_model.load_torque(); g1_model.set_crouch(m, d)
    ctl = walk.WalkController(m, d, vx_cmd=a.vx, kp_up=300.0, swing_id=True, soft_land=True,
                              lam_swing=True, wn_swing=30.0, zeta_swing=0.7, stance_frac=0.57,
                              q_py=300, gate_sy=False, side_w=a.sidew / 100.0)
    h = ctl.params.h_sole
    fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link")
          for s in ("left", "right")]
    toe, heel = [[], []], [[], []]
    for i, b in enumerate(fb):
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == b and m.geom_group[g] == 3:
                (toe if m.geom_pos[g][0] > 0 else heel)[i].append(g)
    sids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in mpc_srb.FOOT_SITES]

    dt = m.opt.timestep
    rec = [[], []]; tds = [[], []]; prev = [True, True]; f6 = np.zeros(6)
    for k in range(int(a.seconds / dt)):
        t = k * dt
        mujoco.mj_forward(m, d)
        if k % walk.DECIM == 0:
            ctl.update_mpc(d, t)
        for i in range(2):
            st = ctl.gait.in_stance(t, i)
            if st and not prev[i] and t > a.settle:
                tds[i].append(t)
            prev[i] = st
            ft = fh = 0.0
            for c in range(d.ncon):
                con = d.contact[c]
                for g in (con.geom1, con.geom2):
                    if g in toe[i] or g in heel[i]:
                        mujoco.mj_contactForce(m, d, c, f6)
                        fz = abs((con.frame.reshape(3, 3).T @ f6[:3])[2])
                        if g in toe[i]:
                            ft += fz
                        else:
                            fh += fz
            R = d.site_xmat[sids[i]].reshape(3, 3)
            pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))            # + = 발끝이 아래로
            zt = min(d.geom_xpos[g][2] for g in toe[i]) - 0.005     # 접촉구 바닥 높이
            zh = min(d.geom_xpos[g][2] for g in heel[i]) - 0.005
            Fx, Fy, Fz, mx, my, mz = ctl.wr[i]
            xc = -(my + h * Fx) / Fz if Fz > 20 else np.nan          # 명령 CoP x (+ = 앞)
            rec[i].append((t, ft, fh, np.degrees(pitch), zt, zh, xc, my))
        d.ctrl[:] = ctl.torque(d, t); mujoco.mj_step(m, d)
        if d.qpos[2] < 0.5:
            print(f"  ⚠ {t:.1f}s 전도"); break

    grid = np.arange(-0.02, 0.30, 0.004)
    P = []
    for i in range(2):
        A = np.array(rec[i]); T = A[:, 0]
        for td in tds[i]:
            idx = np.clip(np.searchsorted(T, td + grid), 0, len(T) - 1)
            P.append(A[idx, 1:])
    P = np.array(P)
    lt, lh = ctl.params.l_t, ctl.params.l_h
    print(f"착지 {len(P)}회 평균 — vx={a.vx}, 보폭 {a.sidew:.0f} cm.  "
          f"CoP 한계: 발가락 +{100*lt:.0f} cm / 뒤꿈치 −{100*lh:.0f} cm\n")
    print(f"{'대비':>7s} | {'발가락Fz':>8s} {'뒤꿈치Fz':>8s} | {'발 pitch':>8s} {'발가락높이':>9s} "
          f"{'뒤꿈치높이':>9s} | {'명령 CoP x':>10s} {'명령 my':>8s}")
    for j, g in enumerate(grid):
        ms = round(g * 1000)
        if ms % 12 != 0 and not (0 <= ms <= 100 and ms % 8 == 0):
            continue
        r = np.nanmean(P[:, j, :], axis=0)
        flag = "  ← 발가락 하중 0" if (g > 0.02 and r[0] < 5 and r[1] > 50) else (
               "  ← 뒤꿈치 하중 0" if (g > 0.02 and r[1] < 5 and r[0] > 50) else "")
        print(f"{ms:+6d}ms | {r[0]:7.0f}N {r[1]:7.0f}N | {r[2]:+7.2f}° {r[3]*1000:7.1f}mm "
              f"{r[4]*1000:7.1f}mm | {r[5]*100:+8.1f}cm {r[6]:+7.1f}{flag}")

    late = grid > 0.03
    zt = P[:, :, 3] * 1000
    lift = zt[:, late].max(axis=1)
    xc = P[:, :, 5] * 100
    jump = np.nanmax(np.abs(np.diff(xc, axis=1)), axis=1)
    print(f"\n발가락 최대 들림: 평균 {lift.mean():.1f} mm, 최대 {lift.max():.1f} mm   "
          f"(가장 높은 시각 +{1000*grid[late][np.argmax(np.nanmean(zt[:, late], axis=0))]:.0f} ms)")
    print(f"발 pitch 최소: 평균 {P[:, :, 2].min(axis=1).mean():+.1f}°")
    print(f"명령 CoP x 의 최대 점프(4 ms 격자): 평균 {np.nanmean(jump):.1f} cm")
    print(f"발가락 하중 0 구간: 평균 {1000*0.004*np.mean(np.sum((P[:, late, 0] < 5) & (P[:, late, 1] > 50), axis=1)):.0f} ms / 스텝")


if __name__ == "__main__":
    main()
