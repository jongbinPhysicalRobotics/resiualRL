"""발이 스케줄대로 땅에 닿고 떨어지는가 — 현재 권장 구성 (Q&A 9/22 Q11).

스텝(스케줄 착지)마다:
  착지 예정 순간  : 실제 접촉 여부, 발 최저점 높이 [mm], MPC 명령 Fz [N]
  첫 접촉 지연    : 실제 첫 접촉 − 예정 착지 [ms] (음수 = 일찍 닿음)
  공중 명령 충격량: 예정 착지 후 실제 접촉 전까지 명령 Fz 의 적분 [N·s]
  튐              : 스탠스 중 접촉이 끊겼다 다시 붙은 횟수, 끊긴 총시간 [ms]
  덮음 비율       : 스케줄 스탠스 동안 실제 접촉(발 합력 > 20 N) 비율 [%]
  이륙            : 실제 접촉 끝 − 예정 이륙 [ms] (음수 = 일찍 떨어짐, 양수 = 끌림)
  합력 최저       : 착지 후 30 ~ 150 ms 양발 합력 최저 [체중 %]

사용:
  .venv/Scripts/python.exe MPC/src/21_contact_check.py --vx 0.5 0.7
  .venv/Scripts/python.exe MPC/src/21_contact_check.py --vx 0.5 --var "soft_land=0"   (sin 아치와 비교)
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, argparse, importlib, itertools
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_model
walk = importlib.import_module("09_walk")

REC = dict(kp_up=300.0, swing_id=True, soft_land=True, lam_swing=True, wn_swing=30.0, zeta_swing=0.7,
           stance_frac=0.57, q_py=300, gate_sy=False, side_w=0.13, td_scale=1.25, cop_margin=0.9,
           wz_pelvis=1.0, wx_pelvis=0.5)


def parse(var):
    kw = dict(REC)
    for item in filter(None, var.split(",")):
        k, v = item.split("=")
        kw[k] = float(v) if k not in ("soft_land", "swing_id", "lam_swing") else bool(float(v))
    return kw


def run(vx, var, seconds=40.0, settle=8.0):
    m, d = g1_model.load_torque(); g1_model.set_crouch(m, d)
    ctl = walk.WalkController(m, d, vx_cmd=vx, **parse(var))
    dt = m.opt.timestep
    fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link") for s in ("left", "right")]
    sph = [[g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
           for b in fb]
    rad = m.geom_size[sph[0][0]][0]
    W = mujoco.mj_getTotalmass(m) * 9.81
    f6 = np.zeros(6)
    T, ST, FZ, FC, ZL = [], [], [], [], []
    fell = None
    for k in range(int(seconds / dt)):
        t = k * dt
        mujoco.mj_forward(m, d)
        if k % walk.DECIM == 0:
            ctl.update_mpc(d, t)
        if t > settle:
            fz = [0.0, 0.0]
            for ci in range(d.ncon):
                con = d.contact[ci]
                for i in range(2):
                    if m.geom_bodyid[con.geom1] == fb[i] or m.geom_bodyid[con.geom2] == fb[i]:
                        mujoco.mj_contactForce(m, d, ci, f6); fz[i] += abs(f6[0])
            T.append(t); ST.append([ctl.gait.in_stance(t, i) for i in range(2)]); FZ.append(fz)
            FC.append([ctl.wr[i][2] if ctl.gait.in_stance(t, i) else 0.0 for i in range(2)])
            ZL.append([min(d.geom_xpos[g][2] for g in sph[i]) - rad for i in range(2)])
        d.ctrl[:] = ctl.torque(d, t); mujoco.mj_step(m, d)
        if d.qpos[2] < 0.5:
            fell = t; break
    if len(T) < 1000:                                    # 정착 전에 넘어짐
        return dict(vx=vx, var=var or "권장 구성", fell=fell, at_td=np.zeros(0))
    T, ST, FZ, FC, ZL = map(np.array, (T, ST, FZ, FC, ZL))
    on = FZ > 20.0
    tot = FZ.sum(axis=1) / W
    n150, n30 = int(0.15 / dt), int(0.03 / dt)
    R = dict(at_td=[], z_td=[], fc_td=[], delay=[], imp=[], bounce=[], gap=[], cover=[], lo=[], hole=[])
    for i in range(2):
        s = ST[:, i].astype(int)
        tds = np.flatnonzero(np.diff(s) == 1) + 1
        los = np.flatnonzero(np.diff(s) == -1) + 1
        for k0 in tds:
            nl = los[los > k0]
            if not len(nl) or nl[0] + n150 >= len(T):
                continue
            k1 = nl[0]                                   # 예정 이륙
            R["at_td"].append(on[k0, i]); R["z_td"].append(1000 * ZL[k0, i]); R["fc_td"].append(FC[k0, i])
            # 첫 접촉: 예정 착지 앞 60 ms 부터 찾는다 (일찍 닿는 경우)
            k_pre = max(0, k0 - int(0.06 / dt))
            first = np.flatnonzero(on[k_pre:k1, i])
            if len(first):
                kc = k_pre + first[0]
                R["delay"].append(1000 * (kc - k0) * dt)
                R["imp"].append(FC[k0:max(k0, kc), i].sum() * dt)
                seg = on[kc:k1, i].astype(int)
                R["bounce"].append(int(np.sum(np.diff(seg) == 1)))
                R["gap"].append(1000 * dt * np.sum(seg == 0))
            else:
                R["delay"].append(np.nan); R["imp"].append(FC[k0:k1, i].sum() * dt)
                R["bounce"].append(0); R["gap"].append(1000 * (k1 - k0) * dt)
            R["cover"].append(100 * on[k0:k1, i].mean())
            # 실제 접촉 끝: 예정 이륙 전후 ±150 ms 안에서 마지막으로 붙어 있던 시각
            w0, w1 = max(k0, k1 - n150), min(len(T), k1 + n150)
            last = np.flatnonzero(on[w0:w1, i])
            R["lo"].append(1000 * ((w0 + last[-1] + 1) - k1) * dt if len(last) else np.nan)
            R["hole"].append(100 * tot[k0 + n30:k0 + n150].min())
    out = {k: np.array(v, float) for k, v in R.items()}
    out.update(vx=vx, var=var or "권장 구성", fell=fell)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, nargs="+", default=[0.5, 0.7])
    ap.add_argument("--var", nargs="+", default=[""])
    ap.add_argument("--seconds", type=float, default=40.0)
    a = ap.parse_args()
    jobs = [(vx, var, a.seconds) for vx, var in itertools.product(a.vx, a.var)]
    with Pool(len(jobs)) as pool:
        res = pool.starmap(run, jobs)
    for r in res:
        n = len(r["at_td"])
        if n == 0:
            print(f"\n=== vx {r['vx']}  {r['var']}  — {r['fell']:.1f} s 에 전도 (정착 전) ==="); continue
        print(f"\n=== vx {r['vx']}  {r['var']}  ({'완주' if r['fell'] is None else '전도'}, 스텝 {n}) ===")
        d_ = r["delay"][np.isfinite(r["delay"])]
        print(f"  착지 예정 순간 실제 접촉   : {100 * r['at_td'].mean():5.1f} %   (발 최저점 높이 평균 {np.mean(r['z_td']):+.1f} mm, "
              f"명령 Fz 평균 {np.mean(r['fc_td']):.0f} N)")
        print(f"  첫 접촉 지연              : 평균 {np.mean(d_):+5.1f} ms · 중앙 {np.median(d_):+5.1f} · 최대 {np.max(d_):+5.1f}   "
              f"(일찍 닿음 {100 * np.mean(d_ < 0):.0f} %, 끝내 안 닿음 {int(np.sum(~np.isfinite(r['delay'])))} 회)")
        print(f"  공중 발에 명령한 충격량    : 평균 {np.mean(r['imp']):.2f} N·s · 최대 {np.max(r['imp']):.2f}")
        print(f"  스탠스 중 튐 (끊겼다 다시)  : 스텝당 {np.mean(r['bounce']):.2f} 회, 끊긴 시간 {np.mean(r['gap']):.1f} ms   "
              f"(튐 있는 스텝 {100 * np.mean(r['bounce'] > 0):.0f} %)")
        print(f"  스탠스 덮음 비율           : 평균 {np.mean(r['cover']):5.1f} % · 최저 {np.min(r['cover']):5.1f} %")
        lo = r["lo"][np.isfinite(r["lo"])]
        print(f"  실제 이륙 − 예정 이륙      : 평균 {np.mean(lo):+5.1f} ms · 범위 {np.min(lo):+.0f} ~ {np.max(lo):+.0f}")
        print(f"  착지 후 30~150 ms 합력 최저: 평균 {np.mean(r['hole']):.0f} % · 최악 {np.min(r['hole']):.0f} % (체중 대비)")


if __name__ == "__main__":
    main()
