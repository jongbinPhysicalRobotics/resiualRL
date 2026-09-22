"""걸음 품질 A/B — 속도만이 아니라 '보기 좋게' 걷는지 (Q&A 9/22).

변형 문자열: "tds=1.25,copm=0.8,duf=1e-5,dum=1e-3"  (WalkController 인자 약어)
  tds = td_scale, copm = cop_margin, duf/dum = Δu 벌점 (힘/모멘트), sw = 보폭 [cm],
  qN = Q 대각 N 번 (0 roll 1 pitch 2 yaw ... 8 wz), wzp/wxp/wyp = ω_z/ω_x/ω_y 골반 비율 (0~1)

지표 (정착 8 s 이후):
  추종 %, 몸 pitch, roll σ
  발 튐    : 발가락 들림 mm (스텝별 최대의 평균), 발가락 하중 0 ms/스텝, 명령 CoP x 최대 점프 cm/스텝,
             착지 후 30~150 ms 합력 최저 (체중 %)
  무릎     : 두 무릎 간격 최소 cm, 디딤 무릎 안쪽 방향 95 % [deg], 스윙 hip_roll 모음 95 % [deg]
  yaw      : 디딤 발 yaw 미끄러짐 °/스탠스, 골반 yaw 진동 폭 (95 %) °
  토크     : 스윙 한계 초과 %, stance 최대 배

사용:
  .venv/Scripts/python.exe MPC/src/18_gait_quality.py --vx 0.5 0.7 --var "tds=1.25" "tds=1.25,dum=1e-3"
"""
from __future__ import annotations
import sys, argparse, importlib, itertools
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_model, mpc_srb
walk = importlib.import_module("09_walk")
kg = importlib.import_module("17_knee_geometry")

KEYS = {"tds": "td_scale", "copm": "cop_margin", "duf": "du_f", "dum": "du_m", "wzp": "wz_pelvis",
        "wxp": "wx_pelvis", "wyp": "wy_pelvis"}


def parse(var):
    kw = dict(side_w=0.13)
    for item in filter(None, var.split(",")):
        k, v = item.split("=")
        if k == "sw":
            kw["side_w"] = None if v == "geom" else float(v) / 100.0
        elif k.startswith("q"):                   # q2=100 → Q[2] (yaw) = 100
            kw.setdefault("q_over", {})[int(k[1:])] = float(v)
        else:
            kw[KEYS[k]] = float(v)
    return kw


def run(vx, var, seconds=40.0, settle=8.0):
    kw = parse(var)
    m, d = g1_model.load_torque(); g1_model.set_crouch(m, d)
    ctl = walk.WalkController(m, d, vx_cmd=vx, kp_up=300.0, swing_id=True, soft_land=True,
                              lam_swing=True, wn_swing=30.0, zeta_swing=0.7, stance_frac=0.57,
                              q_py=300, gate_sy=False, **kw)
    h = ctl.params.h_sole
    Tst, dt = ctl.gait.T_stance, m.opt.timestep
    G = kg.Geo(m)
    fb = G.ank
    toe, heel = [set(), set()], [set(), set()]
    for i, b in enumerate(fb):
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                (toe if m.geom_pos[g][0] > 0 else heel)[i].add(g)
    rad = m.geom_size[next(iter(toe[0]))][0]
    leg_act = [[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{s}_{j}_joint")
                for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
               for s in ("left", "right")]
    lim = [m.actuator_ctrlrange[a, 1] for a in leg_act]
    W = mujoco.mj_getTotalmass(m) * 9.81
    L = dict(t=[], v=[], pit=[], rol=[], py=[], sep=[], grf=[])
    F = [dict(zt=[], ft=[], fh=[], fy=[], st=[], xc=[], kin=[], hr=[]) for _ in range(2)]
    tq_sw, tq_st = [], []
    f6 = np.zeros(6); fell = None
    for k in range(int(seconds / dt)):
        t = k * dt
        mujoco.mj_forward(m, d)
        upd = k % walk.DECIM == 0
        if upd:
            ctl.update_mpc(d, t)
        tau = ctl.torque(d, t)
        if t > settle:
            psi = ctl._yaw_unwrap; c, s = np.cos(psi), np.sin(psi)
            x0 = ctl.get_state(d, t); Rp = d.xmat[G.pel].reshape(3, 3)
            L["t"].append(t); L["v"].append(c * x0[9] + s * x0[10])
            L["pit"].append(-np.arcsin(np.clip(Rp[2, 0], -1, 1))); L["rol"].append(np.arctan2(Rp[2, 1], Rp[2, 2]))
            geo, sep = G.sample(m, d); L["sep"].append(sep); L["py"].append(G.last_py)
            tot = 0.0
            for i in range(2):
                st = ctl.gait.in_stance(t, i)
                ft = fh = 0.0
                for ci in range(d.ncon):
                    con = d.contact[ci]
                    for g in (con.geom1, con.geom2):
                        if g in toe[i] or g in heel[i]:
                            mujoco.mj_contactForce(m, d, ci, f6)
                            fz = abs((con.frame.reshape(3, 3).T @ f6[:3])[2])
                            if g in toe[i]: ft += fz
                            else: fh += fz
                tot += ft + fh
                Fx, Fy, Fz, mx, my, mz = ctl.wr[i]
                xc = -((-s * mx + c * my) + h * (c * Fx + s * Fy)) / Fz if (st and Fz > 100 and upd) else np.nan   # 작은 Fz 에선 CoP 가 잡음
                f = F[i]
                f["zt"].append(min(d.geom_xpos[g][2] for g in toe[i]) - rad)
                f["ft"].append(ft); f["fh"].append(fh); f["fy"].append(G.last_fy[i]); f["st"].append(st)
                f["xc"].append(xc); f["kin"].append(geo[i][1]); f["hr"].append(geo[i][2])
                r = np.abs(tau[leg_act[i]]) / np.array(lim[i])
                (tq_st if st else tq_sw).append(r.max())
            L["grf"].append(tot / W)
        d.ctrl[:] = tau; mujoco.mj_step(m, d)
        if d.qpos[2] < 0.5:
            fell = t; break
    out = dict(vx=vx, var=var, fell=fell)
    if len(L["t"]) < 1000:
        return out
    tt = np.array(L["t"]); v = np.array(L["v"])
    out.update(v=v.mean(), pitch=np.degrees(np.mean(L["pit"])), roll=np.degrees(np.std(L["rol"])),
               pitch_sd=np.degrees(np.std(L["pit"])))
    py = np.unwrap(np.array(L["py"])); n = int(0.8 / dt)
    head = np.convolve(py, np.ones(n) / n, mode="same")
    dp = np.degrees(py - head)[n:-n]
    out["pel_yaw"] = np.percentile(dp, 97.5) - np.percentile(dp, 2.5)
    sep = 100 * np.array(L["sep"]); out["sep_min"] = sep.min(); out["sep_p5"] = np.percentile(sep, 5)
    grf = np.array(L["grf"])
    lift, t0, jump, slip, hole, kin_st, hr_sw = [], [], [], [], [], [], []
    n30, n150, nst = int(0.03 / dt), int(0.15 / dt), int(Tst / dt)
    for i in range(2):
        f = {k2: np.array(v2) for k2, v2 in F[i].items()}
        st = f["st"].astype(int)
        tds = np.flatnonzero(np.diff(st) == 1) + 1
        kin_st.append(f["kin"][f["st"]]); hr_sw.append(f["hr"][~f["st"]])
        for k0 in tds:
            if k0 + nst >= len(st):
                continue
            seg = slice(k0 + n30, k0 + nst)
            lift.append(1000 * f["zt"][seg].max())
            t0.append(1000 * dt * np.sum((f["ft"][seg] < 5) & (f["fh"][seg] > 50)))
            xc = f["xc"][k0:k0 + nst]; xc = xc[np.isfinite(xc)]
            if len(xc) > 2:
                jump.append(100 * np.max(np.abs(np.diff(xc))))
            fy = np.degrees(np.unwrap(f["fy"][k0:k0 + nst]))
            slip.append(abs(fy[-1] - fy[n30]))
            hole.append(100 * grf[k0 + n30:k0 + n150].min())
    out.update(lift=np.mean(lift), toe0=np.mean(t0), jump=np.mean(jump), slip=np.mean(slip),
               slip95=np.percentile(slip, 95), hole=np.mean(hole),
               kin95=np.percentile(np.concatenate(kin_st), 95), hr95=np.percentile(np.concatenate(hr_sw), 95),
               sw_over=100 * np.mean(np.array(tq_sw) > 1.0), st_max=np.max(tq_st))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, nargs="+", default=[0.5, 0.7])
    ap.add_argument("--var", nargs="+", default=["tds=1.25"])
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    jobs = [(vx, var, a.seconds) for vx, var in itertools.product(a.vx, a.var)]
    with Pool(min(a.procs, len(jobs))) as pool:
        res = pool.starmap(run, jobs)
    print(f"걸음 품질 A/B — {a.seconds:.0f} s, 권장 구성 (+ 보폭 13 cm 기본)\n")
    print(f"{'vx':>4s} {'변형':<28s} | {'추종':>4s} {'pitch':>6s} {'pitchσ':>6s} {'rollσ':>5s} | "
          f"{'들림':>5s} {'발가락0':>6s} {'CoP점프':>6s} {'합력최저':>6s} | "
          f"{'무릎간격':>6s} {'디딤무릎':>6s} {'스윙모음':>6s} | {'발yaw미끄럼':>10s} {'골반yaw':>6s} | {'스윙초과':>6s}")
    for r in res:
        if r.get("v") is None:
            print(f"{r['vx']:4.1f} {r['var']:<28s} | {r['fell']:.1f} s 전도"); continue
        tag = "" if r["fell"] is None else f" ✗{r['fell']:.0f}s"
        print(f"{r['vx']:4.1f} {r['var'] + tag:<28s} | {100*r['v']/r['vx']:3.0f}% {r['pitch']:+5.1f}° {r['pitch_sd']:5.2f}° {r['roll']:4.2f}° | "
              f"{r['lift']:4.0f}mm {r['toe0']:4.0f}ms {r['jump']:5.1f}cm {r['hole']:5.0f}% | "
              f"{r['sep_min']:5.1f}cm {r['kin95']:+5.1f}° {r['hr95']:+5.1f}° | "
              f"{r['slip']:4.1f}/{r['slip95']:4.1f}° {r['pel_yaw']:5.1f}° | {r['sw_over']:5.2f}%", flush=True)


if __name__ == "__main__":
    main()
