"""속도별 발 앞 들림 — 속도 벽과 같이 자라나? (Q&A 9/21 Q9)

각 속도를 병렬로 돌려 스텝(착지)마다:
  발가락 하중 0 구간 ms, 발가락 최대 들림 mm, 명령 CoP 뒤꿈치/발가락 포화 %,
  착지 앞거리 rx (발 − CoM, 몸 yaw frame), 이 발의 제동 충격량,
  다음 착지까지 반주기의 속도 변화 dv·평균 속도·pitch
를 기록하고, 속도 간 요약 + 한 속도 안의 스텝 단위 상관을 출력.

사용:
  .venv/Scripts/python.exe MPC/src/baseline/15_toe_speed.py --vx 0.3 0.5 0.6 0.7 --seconds 120
"""
from __future__ import annotations
import sys, argparse, importlib
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_model, mpc_srb
walk = importlib.import_module("09_walk")


def run(vx, seconds=120.0, settle=8.0, side_w=0.13):
    m, d = g1_model.load_torque(); g1_model.set_crouch(m, d)
    ctl = walk.WalkController(m, d, vx_cmd=vx, kp_up=300.0, swing_id=True, soft_land=True,
                              lam_swing=True, wn_swing=30.0, zeta_swing=0.7, stance_frac=0.57,
                              q_py=300, gate_sy=False, side_w=side_w)
    P = ctl.params; h, lh, lt = P.h_sole, P.l_h, P.l_t
    T, Tst = ctl.gait.T, ctl.gait.T_stance
    fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link") for s in ("left", "right")]
    toe, heel = [set(), set()], [set(), set()]
    for i, b in enumerate(fb):
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                (toe if m.geom_pos[g][0] > 0 else heel)[i].add(g)
    rad = m.geom_size[next(iter(toe[0]))][0]
    sids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in mpc_srb.FOOT_SITES]
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    dt = m.opt.timestep
    tt, vv, pp = [], [], []
    F = [dict(ft=[], fh=[], zt=[], xc=[], fx=[], rx=[]) for _ in range(2)]
    tds = [[], []]; prev = [True, True]; f6 = np.zeros(6); fell = None
    for k in range(int(seconds / dt)):
        t = k * dt
        mujoco.mj_forward(m, d)
        if k % walk.DECIM == 0:
            ctl.update_mpc(d, t)
        psi = ctl._yaw_unwrap; c, s = np.cos(psi), np.sin(psi)
        x0 = ctl.get_state(d, t); Rp = d.xmat[pel].reshape(3, 3)
        tt.append(t); vv.append(c * x0[9] + s * x0[10]); pp.append(-np.arcsin(np.clip(Rp[2, 0], -1, 1)))
        for i in range(2):
            st = ctl.gait.in_stance(t, i)
            if st and not prev[i] and t > settle:
                tds[i].append(t)
            prev[i] = st
            ft = fh = 0.0; Fw = np.zeros(3)
            for ci in range(d.ncon):
                con = d.contact[ci]
                for g, sg in ((con.geom1, -1.0), (con.geom2, 1.0)):
                    if g in toe[i] or g in heel[i]:
                        mujoco.mj_contactForce(m, d, ci, f6)
                        fw = sg * (con.frame.reshape(3, 3).T @ f6[:3])
                        if g in toe[i]: ft += fw[2]
                        else: fh += fw[2]
                        Fw += fw
            Fx, Fy, Fz, mx, my, mz = ctl.wr[i]
            Fxb, myb = c * Fx + s * Fy, -s * mx + c * my           # 제약은 몸 yaw frame
            F[i]["ft"].append(ft); F[i]["fh"].append(fh)
            F[i]["zt"].append(min(d.geom_xpos[g][2] for g in toe[i]) - rad)
            F[i]["xc"].append(-(myb + h * Fxb) / Fz if Fz > 20 else np.nan)
            F[i]["fx"].append(c * Fw[0] + s * Fw[1])
            r = d.site_xpos[sids[i]] - d.subtree_com[0]; F[i]["rx"].append(c * r[0] + s * r[1])
        d.ctrl[:] = ctl.torque(d, t); mujoco.mj_step(m, d)
        if d.qpos[2] < 0.5:
            fell = t; break
    tt, vv, pp = np.array(tt), np.array(vv), np.degrees(np.array(pp))
    F = [{k: np.array(v) for k, v in f.items()} for f in F]
    steps = []
    for i in range(2):
        for td in tds[i]:
            if td + T > tt[-1]:
                continue
            k0, k30, kst, kh = np.searchsorted(tt, [td, td + 0.03, td + Tst, td + T / 2])
            ft, fh, xc = F[i]["ft"][k30:kst], F[i]["fh"][k30:kst], F[i]["xc"][k0:kst]
            steps.append(dict(
                toe0=1000 * dt * np.sum((ft < 5) & (fh > 50)),
                lift=1000 * np.max(F[i]["zt"][k30:kst]),
                heelsat=100 * np.nanmean(xc <= -0.95 * lh), toesat=100 * np.nanmean(xc >= 0.95 * lt),
                brake=float(np.sum(np.minimum(F[i]["fx"][k0:kst], 0)) * dt),
                rx=100 * F[i]["rx"][k0],
                dv=float(vv[kh] - vv[k0]), v=float(np.mean(vv[k0:kh])), pitch=float(np.mean(pp[k0:kh]))))
    return dict(vx=vx, fell=fell, v=float(np.mean(vv[tt > settle])),
                pitch=float(np.mean(pp[tt > settle])), steps=steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, nargs="+", default=[0.3, 0.5, 0.6, 0.7])
    ap.add_argument("--seconds", type=float, default=120.0)
    a = ap.parse_args()
    with Pool(len(a.vx)) as pool:
        res = pool.starmap(run, [(v, a.seconds) for v in a.vx])

    print(f"속도별 발 앞 들림 — {a.seconds:.0f} s, 권장 구성 + 보폭 13 cm\n")
    print(f"{'vx':>4s} | {'결과':>6s} {'실측':>6s} {'추종':>4s} {'pitch':>6s} | {'착지 앞거리':>8s} | "
          f"{'발가락0':>7s} {'들림':>7s} | {'뒤꿈치포화':>8s} {'발가락포화':>8s} | {'제동충격량':>9s}")
    for o in res:
        g = lambda k: np.array([s[k] for s in o["steps"]])
        r = "OK" if o["fell"] is None else f"{o['fell']:.0f}s✗"
        print(f"{o['vx']:4.1f} | {r:>6s} {o['v']:6.3f} {100*o['v']/o['vx']:3.0f}% {o['pitch']:+5.1f}° | "
              f"{g('rx').mean():7.1f}cm | {g('toe0').mean():5.0f}ms {g('lift').mean():5.1f}mm | "
              f"{g('heelsat').mean():7.1f}% {g('toesat').mean():7.1f}% | {g('brake').mean():+7.2f}Ns")

    print("\n한 속도 안: 발가락 0 구간 짧은 1/3 vs 긴 1/3 의 반주기 평균 속도 (들림이 그 스텝의 속도를 깎나?)")
    for o in res:
        g = lambda k: np.array([s[k] for s in o["steps"]])
        t0 = g("toe0")
        if t0.std() < 1e-9:
            print(f"{o['vx']:4.1f} | 들림 없음"); continue
        lo, hi = t0 <= np.quantile(t0, 1/3), t0 >= np.quantile(t0, 2/3)
        rr = np.corrcoef(t0, g("dv"))[0, 1]
        print(f"{o['vx']:4.1f} | 짧음 {t0[lo].mean():4.0f} ms → v {g('v')[lo].mean():.3f} | "
              f"김 {t0[hi].mean():4.0f} ms → v {g('v')[hi].mean():.3f} | r(발가락0, dv) {rr:+.2f}")


if __name__ == "__main__":
    main()
