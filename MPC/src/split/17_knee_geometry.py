"""무릎이 안쪽으로 접히는 현상 계측 — MPC 와 배포 RL 을 같은 기하 지표로 (Q&A 9/22 Q1).

지표 (몸 yaw frame, 다리별, stance / swing 분리):
  valgus  : 무릎이 엉덩이–발목 직선에서 몸 안쪽으로 벗어난 거리 [cm] (+ = 안쪽)
  kin     : 무릎이 향하는 방향의 안쪽 회전 [deg] (+ = 안쪽, knee 관절축으로 계산)
  sep     : 두 무릎 사이 횡거리 [cm] (작을수록 다리가 모임, 음수 = 교차)
  hip_roll / hip_yaw : 관절각 [deg] (+ = 모음 / 안쪽 회전, 좌우 부호 통일)

사용:
  .venv/Scripts/python.exe MPC/src/split/17_knee_geometry.py --mpc 0.5:1.0:13 0.7:1.25:13 --rl 0.5 0.7
  (--mpc 은 vx:td_scale:보폭cm, 보폭 geom = 기하 기본)
"""
from __future__ import annotations
import sys, argparse, importlib
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import mujoco

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from paths import ROOT as _ROOT                                   # 폴더 깊이와 무관 (9/24 src 분리)
sys.path.insert(0, str(_ROOT / "deployed RL/analysis"))
SIDES = ("left", "right")


class Geo:
    def __init__(self, m):
        b = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        j = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.pel = b("pelvis")
        self.hip = [b(f"{s}_hip_pitch_link") for s in SIDES]
        self.knee = [b(f"{s}_knee_link") for s in SIDES]
        self.ank = [b(f"{s}_ankle_roll_link") for s in SIDES]
        self.kj = [j(f"{s}_knee_joint") for s in SIDES]
        self.qa = {k: [m.jnt_qposadr[j(f"{s}_{k}_joint")] for s in SIDES] for k in ("hip_roll", "hip_yaw")}
        self.sgn = [1.0, -1.0]                      # left = +y
        self.last_fy = [0.0, 0.0]; self.last_py = 0.0

    def sample(self, m, d):
        Rp = d.xmat[self.pel].reshape(3, 3)
        yaw = np.arctan2(Rp[1, 0], Rp[0, 0]); c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])        # world → 몸 yaw
        out = []
        ky = []
        for i in range(2):
            h, k, a = (Rz @ d.xpos[x] for x in (self.hip[i], self.knee[i], self.ank[i]))
            # 정면(y–z) 평면에서 엉덩이–발목 직선 대비 무릎 횡편차
            u = (k[2] - h[2]) / (a[2] - h[2]) if abs(a[2] - h[2]) > 1e-6 else 0.5
            y_line = h[1] + u * (a[1] - h[1])
            valgus = -self.sgn[i] * (k[1] - y_line)
            # 무릎이 향하는 방향: 관절축 × 위 (수평 투영) 의 yaw
            ax = Rz @ (d.xmat[self.knee[i]].reshape(3, 3) @ m.jnt_axis[self.kj[i]])
            fwd = np.cross(ax, [0, 0, 1.0])
            kin = -self.sgn[i] * np.degrees(np.arctan2(fwd[1], fwd[0]))
            hr = self.sgn[i] * -np.degrees(d.qpos[self.qa["hip_roll"][i]])
            hy = -self.sgn[i] * np.degrees(d.qpos[self.qa["hip_yaw"][i]])
            Rf = d.xmat[self.ank[i]].reshape(3, 3)
            fy = np.arctan2(Rf[1, 0], Rf[0, 0])
            out.append((valgus, kin, hr, hy)); ky.append(k[1]); self.last_fy[i] = fy
        self.last_py = yaw
        return out, ky[0] - ky[1]


def summarize(rows):
    rows = np.array(rows) if rows else np.zeros((1, 4))
    return dict(valgus=(100 * rows[:, 0].mean(), 100 * np.percentile(rows[:, 0], 95)),
                kin=(rows[:, 1].mean(), np.percentile(rows[:, 1], 95)),
                hr=(rows[:, 2].mean(), np.percentile(rows[:, 2], 95)),
                hy=(rows[:, 3].mean(), np.percentile(rows[:, 3], 95)))


def run_mpc(vx, tds, sidew, seconds=30.0, settle=8.0):
    import g1_model
    walk = importlib.import_module("09_walk")
    m, d = g1_model.load_torque(); g1_model.set_crouch(m, d)
    ctl = walk.WalkController(m, d, vx_cmd=vx, kp_up=300.0, swing_id=True, soft_land=True,
                              lam_swing=True, wn_swing=30.0, zeta_swing=0.7, stance_frac=0.57,
                              q_py=300, gate_sy=False, side_w=sidew, td_scale=tds)
    G = Geo(m); rec = {"st": [], "sw": []}; sep = []; vs = []; fell = None
    for k in range(int(seconds / m.opt.timestep)):
        t = k * m.opt.timestep
        mujoco.mj_forward(m, d)
        if k % walk.DECIM == 0:
            ctl.update_mpc(d, t)
        if t > settle:
            g, sp = G.sample(m, d); sep.append(sp)
            for i in range(2):
                rec["st" if ctl.gait.in_stance(t, i) else "sw"].append(g[i])
            x0 = ctl.get_state(d, t); psi = ctl._yaw_unwrap
            vs.append(np.cos(psi) * x0[9] + np.sin(psi) * x0[10])
        d.ctrl[:] = ctl.torque(d, t); mujoco.mj_step(m, d)
        if d.qpos[2] < 0.5:
            fell = t; break
    w = "geom" if sidew is None else f"{100*sidew:.0f}"
    return dict(name=f"MPC vx {vx} ×{tds} 보폭 {w}", v=float(np.mean(vs)) if vs else 0.0, fell=fell,
                st=summarize(rec["st"]), sw=summarize(rec["sw"]),
                sep=(100 * np.mean(sep), 100 * np.min(sep)))


def run_rl(vx, seconds=30.0, settle=8.0):
    import torch, run_g1_policy as R
    m, d, policy, c = R.load()
    nA = c["num_actions"]
    kps = np.array(c["kps"], np.float32); kds = np.array(c["kds"], np.float32)
    dflt = np.array(c["default_angles"], np.float32)
    cmd = np.array([vx, 0, 0], np.float32); cs = np.array(c["cmd_scale"], np.float32)
    action = np.zeros(nA, np.float32); target = dflt.copy(); obs = np.zeros(c["num_obs"], np.float32)
    dt = c["simulation_dt"]; DEC = c["control_decimation"]
    G = Geo(m); rec = {"st": [], "sw": []}; sep = []; vs = []; f6 = np.zeros(6)
    mujoco.mj_forward(m, d)
    R0 = d.xmat[G.pel].reshape(3, 3); yref = np.arctan2(R0[1, 0], R0[0, 0])
    for k in range(int(seconds / dt)):
        t = k * dt
        d.ctrl[:] = (target - d.qpos[7:]) * kps + (0.0 - d.qvel[6:]) * kds
        mujoco.mj_step(m, d)
        if (k + 1) % DEC == 0:
            Rp = d.xmat[G.pel].reshape(3, 3)
            e = (np.arctan2(Rp[1, 0], Rp[0, 0]) - yref + np.pi) % (2 * np.pi) - np.pi
            cmd[2] = float(np.clip(-e, -0.5, 0.5))
            ph = ((k + 1) * dt) % 0.8 / 0.8
            obs[:3] = d.qvel[3:6] * c["ang_vel_scale"]; obs[3:6] = R.gravity_orientation(d.qpos[3:7])
            obs[6:9] = cmd * cs
            obs[9:9+nA] = (d.qpos[7:] - dflt) * c["dof_pos_scale"]; obs[9+nA:9+2*nA] = d.qvel[6:] * c["dof_vel_scale"]
            obs[9+2*nA:9+3*nA] = action; obs[9+3*nA:9+3*nA+2] = [np.sin(2*np.pi*ph), np.cos(2*np.pi*ph)]
            action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target = action * c["action_scale"] + dflt
        if t > settle:
            g, sp = G.sample(m, d); sep.append(sp)
            for i in range(2):
                F, _ = R.foot_wrench(m, d, G.ank[i])
                rec["st" if F[2] > 20.0 else "sw"].append(g[i])
            mujoco.mj_subtreeVel(m, d)
            Rp = d.xmat[G.pel].reshape(3, 3); yw = np.arctan2(Rp[1, 0], Rp[0, 0])
            vs.append(d.subtree_linvel[0][0] * np.cos(yw) + d.subtree_linvel[0][1] * np.sin(yw))
    return dict(name=f"RL  vx {vx}", v=float(np.mean(vs)), fell=None,
                st=summarize(rec["st"]), sw=summarize(rec["sw"]),
                sep=(100 * np.mean(sep), 100 * np.min(sep)))


def job(spec):
    kind, args = spec
    return run_mpc(*args) if kind == "mpc" else run_rl(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpc", nargs="*", default=["0.5:1.0:13", "0.5:1.25:13", "0.7:1.25:13", "0.5:1.0:geom"])
    ap.add_argument("--rl", type=float, nargs="*", default=[0.5, 0.7])
    ap.add_argument("--seconds", type=float, default=30.0)
    a = ap.parse_args()
    specs = []
    for s in a.mpc:
        vx, tds, w = s.split(":")
        specs.append(("mpc", (float(vx), float(tds), None if w == "geom" else float(w) / 100.0, a.seconds)))
    specs += [("rl", (vx, a.seconds)) for vx in a.rl]
    with Pool(len(specs)) as pool:
        res = pool.map(job, specs)
    print(f"무릎 기하 — {a.seconds:.0f} s (정착 8 s 이후). 값 = 평균 / 95 백분위\n")
    print(f"{'조건':<26s} {'실측':>6s} | {'무릎 간격':>11s} | "
          f"{'스윙 무릎안쪽':>12s} {'스윙 무릎방향':>12s} {'스윙 hip_roll':>12s} {'스윙 hip_yaw':>12s} | "
          f"{'스탠스 무릎안쪽':>12s} {'스탠스 방향':>11s}")
    for r in res:
        f = lambda t: f"{t[0]:+5.1f}/{t[1]:+5.1f}"
        tag = "" if r["fell"] is None else f" ✗{r['fell']:.0f}s"
        print(f"{r['name'] + tag:<26s} {r['v']:6.3f} | {r['sep'][0]:5.1f}/{r['sep'][1]:5.1f}cm | "
              f"{f(r['sw']['valgus'])}cm {f(r['sw']['kin'])}° {f(r['sw']['hr'])}° {f(r['sw']['hy'])}° | "
              f"{f(r['st']['valgus'])}cm {f(r['st']['kin'])}°", flush=True)


if __name__ == "__main__":
    main()
