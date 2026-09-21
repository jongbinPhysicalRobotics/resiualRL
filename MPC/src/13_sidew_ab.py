"""보폭 정책 A/B — 공칭 보폭을 바꿨을 때 실제 착지 보폭·횡 CoP·속도가 어떻게 되나.

사용:
  .venv/Scripts/python.exe MPC/src/13_sidew_ab.py --vx 0.3 0.5 0.6 --sidew geom 17.6 15.0 auto --seconds 30
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_model, mpc_srb
import importlib
walk = importlib.import_module("09_walk")


def foot_fz_cop(m, d, bid, origin, R, h):
    F = np.zeros(3); Mo = np.zeros(3); f6 = np.zeros(6)
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
        if bid not in (b1, b2):
            continue
        mujoco.mj_contactForce(m, d, c, f6)
        fw = con.frame.reshape(3, 3).T @ f6[:3]
        if b1 == bid:
            fw = -fw
        F += fw; Mo += np.cross(con.pos - origin, fw)
    if F[2] < 20.0:
        return None
    Fl = R.T @ F; Ml = R.T @ Mo
    cop_y = (Ml[0] - h * Fl[1]) / Fl[2]
    return abs(cop_y) / 0.025          # 롤 CoP 이용률 (1.0 = 반폭)


def run(vx, side_w, seconds, settle=8.0):
    m, d = g1_model.load_torque(); g1_model.set_crouch(m, d)
    ctl = walk.WalkController(m, d, vx_cmd=vx, kp_up=300.0, swing_id=True,
                              soft_land=True, lam_swing=True, wn_swing=30.0,
                              zeta_swing=0.7, stance_frac=0.57, q_py=300,
                              gate_sy=False, side_w=side_w)
    fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link")
          for s in ("left", "right")]
    h = ctl.params.h_sole
    dt = m.opt.timestep
    vs=[]; py=[]; rol=[]; pit=[]; cop=[]; ccop=[]; lat=[]; wid=[]; last={}; prev=[True,True]; fell=None
    w_ = ctl.params.w
    for k in range(int(seconds / dt)):
        t = k * dt
        mujoco.mj_forward(m, d)
        if k % walk.DECIM == 0:
            ctl.update_mpc(d, t)
            if t > settle:
                # ★ 제약은 몸 yaw frame 에서 걸린다 (C·Rzᵀ W). 월드 frame mx 를 그대로 쓰면
                #   my(±30 N·m)가 yaw 몇 도만으로도 롤 예산(8 N·m)에 새어 들어온다 → 회전 후 계산.
                psi = ctl._yaw_unwrap
                cps, sps = np.cos(psi), np.sin(psi)
                for i in range(2):
                    Fx, Fy, Fz, mx, my, mz = ctl.wr[i]
                    if Fz > 20.0:
                        Fy_b = -sps * Fx + cps * Fy
                        mx_b = cps * mx + sps * my
                        ccop.append(abs(mx_b - h * Fy_b) / (w_ * Fz))
        d.ctrl[:] = ctl.torque(d, t); mujoco.mj_step(m, d)
        if t > settle:
            x0 = ctl.get_state(d, t); vs.append(x0[9]); py.append(x0[4])
            Rp = d.xmat[1].reshape(3, 3)
            rol.append(np.arctan2(Rp[2, 1], Rp[2, 2])); pit.append(-np.arcsin(np.clip(Rp[2, 0], -1, 1)))
            yaw = np.arctan2(Rp[1, 0], Rp[0, 0]); cy, sy = np.cos(yaw), np.sin(yaw)
            com = d.subtree_com[0]
            for i, bid in enumerate(fb):
                st = ctl.gait.in_stance(t, i)
                Rf = d.xmat[bid].reshape(3, 3)
                u = foot_fz_cop(m, d, bid, d.xpos[bid], Rf, h)
                if u is not None:
                    cop.append(u)
                if st and not prev[i]:                       # 착지
                    r = d.xpos[bid] - com
                    ly = -sy * r[0] + cy * r[1]
                    lat.append(abs(ly)); last[i] = ly
                    if 0 in last and 1 in last:
                        wid.append(abs(last[0] - last[1]))
                prev[i] = st
        if d.qpos[2] < 0.5:
            fell = t; break
    cop = np.array(cop); ccop = np.array(ccop)
    return dict(fell=fell, ccop=ccop.mean() if len(ccop) else 0.0,
                ccop_sat=100*np.mean(ccop > 0.9) if len(ccop) else 0.0, v=np.mean(vs) if vs else 0.0,
                yamp=100 * (max(py) - min(py)) / 2 if py else 0.0,
                roll=np.degrees(np.std(rol)) if rol else 0.0,
                pitch=np.degrees(np.mean(pit)) if pit else 0.0,
                cop=cop.mean() if len(cop) else 0.0,
                cop_sat=100 * np.mean(cop > 0.9) if len(cop) else 0.0,
                lat=100 * np.mean(lat) if lat else 0.0,
                wid=100 * np.mean(wid) if wid else 0.0,
                nom=ctl.side_width(walk.RAMP_T1 + 1.0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, nargs="+", default=[0.3, 0.5, 0.6])
    ap.add_argument("--sidew", nargs="+", default=["geom", "17.6", "15.0", "auto"])
    ap.add_argument("--seconds", type=float, default=30.0)
    a = ap.parse_args()
    print(f"보폭 정책 A/B — {a.seconds:.0f} s, 권장 구성 (--swingid --softland --lamswing 30/0.7 --sf 0.57 --qpy 300 --swingyaw)\n")
    print(f"{'vx':>4s} {'공칭보폭':>8s} | {'결과':>7s} {'실측v':>6s} {'추종':>5s} | {'착지보폭':>8s} {'횡오프':>6s} | {'롤CoP실측':>8s} {'>0.9':>5s} {'롤CoP명령':>8s} {'>0.9':>5s} | {'y진폭':>6s} {'rollσ':>6s} {'pitch':>6s}")
    for vx in a.vx:
        for sw in a.sidew:
            side_w = None if sw == "geom" else ("auto" if sw == "auto" else float(sw) / 100.0)
            r = run(vx, side_w, a.seconds)
            res = f"{a.seconds:.0f}s OK" if r["fell"] is None else f"{r['fell']:.1f}s ✗"
            nom = "기하 23.7" if r["nom"] is None else f"{100*r['nom']:.1f}"
            print(f"{vx:4.1f} {nom:>8s} | {res:>7s} {r['v']:6.3f} {100*r['v']/vx:4.0f}% | "
                  f"{r['wid']:7.1f} {r['lat']:6.1f} | {r['cop']:8.2f} {r['cop_sat']:4.0f}% {r['ccop']:8.2f} {r['ccop_sat']:4.0f}% | "
                  f"{r['yamp']:6.1f} {r['roll']:5.2f}° {r['pitch']:+5.1f}°", flush=True)
        print()
