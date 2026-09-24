"""이중지지 순간의 발–골반 배치와 실제 지지 시간 — MPC 와 배포 RL 비교 (Q&A 9/22 Q4).

사용자 관찰: "앞발이 착지하고 뒷발이 스윙하기 직전을 보면, RL 은 지지발이 더 뒤에 있고
골반이 두 발 사이 대칭인데 우리는 비대칭이다. RL 이 지지 시간이 더 긴 느낌."

접촉 기준 (발 합력 > 20 N, 10 ms 이상 유지) 으로 이벤트를 잡는다 — 스케줄이 아니라 실제 접촉.
  앞발 착지 순간 (이중지지 시작), 뒷발 이륙 순간 (이중지지 끝) 에
    앞발 / 뒷발 위치 (CoM 기준, 진행 방향 [cm]),
    골반·CoM 이 두 발 사이 어디인가 (0 = 뒷발, 0.5 = 한가운데, 1 = 앞발)
  실제 지지 시간 / 스윙 시간 / 이중지지 시간, 착지~이륙 동안 발이 CoM 기준 어디서 어디까지 가나.
발 위치 = 접촉구 4 개의 중심 (발바닥 중앙). 두 모델의 발은 같다.

사용:
  .venv/Scripts/python.exe MPC/src/baseline/19_stance_geometry.py --mpc 0.5 0.7 --rl 0.5 0.7
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

MPC_KW = dict(kp_up=300.0, swing_id=True, soft_land=True, lam_swing=True, wn_swing=30.0,
              zeta_swing=0.7, stance_frac=0.57, q_py=300, gate_sy=False, side_w=0.13,
              td_scale=1.25, cop_margin=0.9, wz_pelvis=1.0, wx_pelvis=0.5)


class Rec:
    def __init__(self, m):
        b = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.pel = b("pelvis")
        self.fb = [b(f"{s}_ankle_roll_link") for s in ("left", "right")]
        self.sph = [[g for g in range(m.ngeom) if m.geom_bodyid[g] == fb and
                     m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE] for fb in self.fb]
        self.t, self.yaw, self.com, self.pelp, self.foot, self.fz = [], [], [], [], [], []
        self.f6 = np.zeros(6)

    def sample(self, m, d, t):
        Rp = d.xmat[self.pel].reshape(3, 3)
        self.t.append(t); self.yaw.append(np.arctan2(Rp[1, 0], Rp[0, 0]))
        self.com.append(d.subtree_com[0][:2].copy()); self.pelp.append(d.xpos[self.pel][:2].copy())
        self.foot.append([np.mean([d.geom_xpos[g][:2] for g in sp], axis=0) for sp in self.sph])
        fz = [0.0, 0.0]
        for ci in range(d.ncon):
            con = d.contact[ci]
            for i in range(2):
                if m.geom_bodyid[con.geom1] == self.fb[i] or m.geom_bodyid[con.geom2] == self.fb[i]:
                    mujoco.mj_contactForce(m, d, ci, self.f6)
                    fz[i] += abs(self.f6[0])
        self.fz.append(fz)

    def analyze(self, dt):
        t = np.array(self.t); yaw = np.unwrap(np.array(self.yaw))
        n = int(0.8 / dt); head = np.convolve(yaw, np.ones(n) / n, mode="same")
        c, s = np.cos(head), np.sin(head)
        fwd = lambda P: c * P[:, 0] + s * P[:, 1]                    # 진행 방향 성분
        com = fwd(np.array(self.com)); pel = fwd(np.array(self.pelp))
        F = np.array(self.foot); foot = np.stack([fwd(F[:, i]) for i in range(2)], axis=1)
        on = np.array(self.fz) > 20.0
        k10 = max(1, int(0.01 / dt))
        for i in range(2):                                           # 10 ms 미만 깜빡임 제거
            x = on[:, i].astype(int); dx = np.diff(np.r_[0, x, 0])
            st, en = np.flatnonzero(dx == 1), np.flatnonzero(dx == -1)
            for a_, b_ in zip(st, en):
                if b_ - a_ < k10: on[a_:b_, i] = False
            x = on[:, i].astype(int); dx = np.diff(np.r_[0, x, 0])
            st, en = np.flatnonzero(dx == 1), np.flatnonzero(dx == -1)
            for a_, b_ in zip(en[:-1], st[1:]):
                if b_ - a_ < k10: on[a_:b_, i] = True
        ev = {i: (np.flatnonzero(np.diff(on[:, i].astype(int)) == 1) + 1,
                  np.flatnonzero(np.diff(on[:, i].astype(int)) == -1) + 1) for i in range(2)}
        lo_n, rel = n, len(t) - n
        rows_td, rows_lo, stance, swing, ds, stroke = [], [], [], [], [], []
        for i in range(2):
            tds, los = ev[i]; j = 1 - i
            for k in tds:
                if not (lo_n < k < rel) or not on[k, j]:
                    continue
                # 앞발 i 착지: 뒷발 j 가 아직 붙어 있음. 이 이중지지가 끝나는 j 의 이륙
                nxt = [q for q in ev[j][1] if q > k]
                if not nxt: continue
                q = nxt[0]
                ds.append((q - k) * dt)
                for kk, rows in ((k, rows_td), (q, rows_lo)):
                    front, rear = foot[kk, i], foot[kk, j]
                    span = front - rear
                    rows.append((100 * (front - com[kk]), 100 * (com[kk] - rear), 100 * span,
                                 (com[kk] - rear) / span, (pel[kk] - rear) / span))
            for k in tds:
                nl = [q for q in los if q > k]
                if not nl or not (lo_n < k < rel): continue
                q = nl[0]; stance.append((q - k) * dt)
                stroke.append((100 * (foot[k, i] - com[k]), 100 * (foot[q, i] - com[q])))
                nt = [p for p in tds if p > q]
                if nt: swing.append((nt[0] - q) * dt)
        v = np.gradient(com, t)
        A = lambda r: np.array(r)
        return dict(v=float(np.mean(v[lo_n:rel])), td=A(rows_td).mean(0), lo=A(rows_lo).mean(0),
                    stance=1000 * np.mean(stance), swing=1000 * np.mean(swing), ds=1000 * np.mean(ds),
                    stroke=A(stroke).mean(0))


def run_mpc(vx, seconds=40.0, settle=8.0):
    import g1_model
    walk = importlib.import_module("09_walk")
    m, d = g1_model.load_torque(); g1_model.set_crouch(m, d)
    ctl = walk.WalkController(m, d, vx_cmd=vx, **MPC_KW)
    R = Rec(m); dt = m.opt.timestep
    for k in range(int(seconds / dt)):
        t = k * dt
        mujoco.mj_forward(m, d)
        if k % walk.DECIM == 0:
            ctl.update_mpc(d, t)
        if t > settle:
            R.sample(m, d, t)
        d.ctrl[:] = ctl.torque(d, t); mujoco.mj_step(m, d)
    return f"MPC 명령 {vx}", R.analyze(dt)


def run_rl(vx, seconds=40.0, settle=8.0):
    import torch, run_g1_policy as P
    m, d, policy, c = P.load(); nA = c["num_actions"]
    kps = np.array(c["kps"], np.float32); kds = np.array(c["kds"], np.float32)
    dflt = np.array(c["default_angles"], np.float32)
    cmd = np.array([vx, 0, 0], np.float32); cs = np.array(c["cmd_scale"], np.float32)
    action = np.zeros(nA, np.float32); target = dflt.copy(); obs = np.zeros(c["num_obs"], np.float32)
    dt = c["simulation_dt"]; DEC = c["control_decimation"]
    R = Rec(m); mujoco.mj_forward(m, d)
    R0 = d.xmat[R.pel].reshape(3, 3); yref = np.arctan2(R0[1, 0], R0[0, 0])
    for k in range(int(seconds / dt)):
        t = k * dt
        d.ctrl[:] = (target - d.qpos[7:]) * kps + (0.0 - d.qvel[6:]) * kds
        mujoco.mj_step(m, d)
        if (k + 1) % DEC == 0:
            Rp = d.xmat[R.pel].reshape(3, 3)
            e = (np.arctan2(Rp[1, 0], Rp[0, 0]) - yref + np.pi) % (2 * np.pi) - np.pi
            cmd[2] = float(np.clip(-e, -0.5, 0.5)); ph = ((k + 1) * dt) % 0.8 / 0.8
            obs[:3] = d.qvel[3:6] * c["ang_vel_scale"]; obs[3:6] = P.gravity_orientation(d.qpos[3:7])
            obs[6:9] = cmd * cs
            obs[9:9+nA] = (d.qpos[7:] - dflt) * c["dof_pos_scale"]; obs[9+nA:9+2*nA] = d.qvel[6:] * c["dof_vel_scale"]
            obs[9+2*nA:9+3*nA] = action; obs[9+3*nA:9+3*nA+2] = [np.sin(2*np.pi*ph), np.cos(2*np.pi*ph)]
            action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target = action * c["action_scale"] + dflt
        if t > settle:
            mujoco.mj_forward(m, d)
            R.sample(m, d, t)
    return f"RL  명령 {vx}", R.analyze(dt)


def job(spec):
    return run_mpc(spec[1]) if spec[0] == "mpc" else run_rl(spec[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpc", type=float, nargs="*", default=[0.5, 0.7])
    ap.add_argument("--rl", type=float, nargs="*", default=[0.5, 0.7])
    a = ap.parse_args()
    specs = [("mpc", v) for v in a.mpc] + [("rl", v) for v in a.rl]
    with Pool(len(specs)) as pool:
        res = pool.map(job, specs)
    print("이중지지 순간의 배치 — 접촉 기준, 발 = 발바닥 중앙, 진행 방향 [cm]. 위치 비율: 0 뒷발 · 0.5 가운데 · 1 앞발\n")
    print(f"{'조건':<12s} {'실측':>6s} | {'지지':>5s} {'스윙':>5s} {'이중지지':>6s} | "
          f"{'[앞발 착지] 앞발':>13s} {'뒷발':>6s} {'보폭':>5s} {'CoM':>5s} {'골반':>5s} | "
          f"{'[뒷발 이륙] 앞발':>13s} {'뒷발':>6s} {'CoM':>5s} {'골반':>5s} | {'지지발 이동: 착지→이륙':>18s}")
    for name, r in res:
        td, lo, st = r["td"], r["lo"], r["stroke"]
        print(f"{name:<12s} {r['v']:6.3f} | {r['stance']:4.0f}ms {r['swing']:4.0f}ms {r['ds']:5.0f}ms | "
              f"{td[0]:+11.1f} {-td[1]:+6.1f} {td[2]:5.1f} {td[3]:5.2f} {td[4]:5.2f} | "
              f"{lo[0]:+11.1f} {-lo[1]:+6.1f} {lo[3]:5.2f} {lo[4]:5.2f} | {st[0]:+7.1f} → {st[1]:+6.1f} cm", flush=True)


if __name__ == "__main__":
    main()
