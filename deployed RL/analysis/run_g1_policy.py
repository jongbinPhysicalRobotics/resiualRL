"""unitree_rl_gym G1 사전학습 정책을 헤드리스로 돌리며 계측.

우리 MPC 와 같은 양을 재서 비교하는 게 목적 (deployed RL/README.md 참고).
그들의 deploy_mujoco.py 의 관측·제어 규약을 그대로 복제하되 뷰어 없이 돌린다.

사용:
  .venv/Scripts/python.exe "deployed RL/analysis/run_g1_policy.py" --vx 1.0 --seconds 30
"""
from __future__ import annotations
import argparse, io, sys
from pathlib import Path

import numpy as np
import mujoco
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent / "unitree_rl_gym"
CFG = REPO / "deploy/deploy_mujoco/configs/g1.yaml"


def gravity_orientation(q):
    qw, qx, qy, qz = q
    return np.array([2 * (-qz * qx + qw * qy),
                     -2 * (qz * qy + qw * qx),
                     1 - 2 * (qw * qw + qz * qz)])


def load():
    c = yaml.safe_load(open(CFG))
    xml = Path(c["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(REPO)))
    pol = c["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(REPO))
    # MuJoCo 의 C 파일열기가 경로의 한글(사용자명 '백종빈')을 못 읽는다 →
    # XML 디렉터리로 chdir 한 뒤 순수 ASCII 상대경로로 로드 (메시도 같은 폴더 기준).
    import os
    cwd = os.getcwd()
    try:
        os.chdir(xml.parent)
        m = mujoco.MjModel.from_xml_path(xml.name)
    finally:
        os.chdir(cwd)
    m.opt.timestep = c["simulation_dt"]
    # torch.jit.load 도 한글 경로를 못 연다 → 바이트로 읽어서 전달
    net = torch.jit.load(io.BytesIO(Path(pol).read_bytes()))
    return m, mujoco.MjData(m), net, c


def composite_inertia(m, d):
    """crouch 자세 기준 CoM 합성관성 (우리 mpc_srb.make_params 와 같은 방식)."""
    com = d.subtree_com[0].copy()
    I = np.zeros((3, 3))
    for b in range(1, m.nbody):
        R = d.ximat[b].reshape(3, 3)
        r = d.xipos[b] - com
        mb = m.body_mass[b]
        I += R @ np.diag(m.body_inertia[b]) @ R.T + mb * (r @ r * np.eye(3) - np.outer(r, r))
    return I


def foot_bodies(m):
    out = []
    for side in ("left", "right"):
        out.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link"))
    return out


def foot_wrench(m, d, bid):
    """발 하나의 접촉력 합 + 지면 CoP (발 frame x,y, 발목 원점 기준)."""
    F = np.zeros(3); Mo = np.zeros(3); f6 = np.zeros(6)
    origin = d.xpos[bid]
    R = d.xmat[bid].reshape(3, 3)
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
        if bid not in (b1, b2):
            continue
        mujoco.mj_contactForce(m, d, c, f6)
        frame = con.frame.reshape(3, 3)
        fw = frame.T @ f6[:3]                      # world 힘 (geom1→geom2 규약)
        if b1 == bid:
            fw = -fw
        r = con.pos - origin
        F += fw; Mo += np.cross(r, fw)
    if F[2] < 1.0:
        return F, None
    # 발 frame 으로 회전해서 CoP (접촉면은 발목 아래 h=0.035)
    Fl = R.T @ F; Ml = R.T @ Mo
    h = 0.035
    cop_x = (-Ml[1] - h * Fl[0]) / Fl[2]
    cop_y = (Ml[0] - h * Fl[1]) / Fl[2]
    return F, (cop_x, cop_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--settle", type=float, default=6.0, help="이 시각 이후만 통계")
    ap.add_argument("--hold", type=float, default=1.0, help="방향 유지 게인 (0=끄기)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    m, d, policy, c = load()
    nA, nO = c["num_actions"], c["num_obs"]
    kps = np.array(c["kps"], np.float32); kds = np.array(c["kds"], np.float32)
    dflt = np.array(c["default_angles"], np.float32)
    cmd = np.array([a.vx, a.vy, a.wz], np.float32)
    cmd_scale = np.array(c["cmd_scale"], np.float32)
    dt = c["simulation_dt"]; DEC = c["control_decimation"]

    action = np.zeros(nA, np.float32); target = dflt.copy(); obs = np.zeros(nO, np.float32)
    mujoco.mj_forward(m, d)
    I_body = composite_inertia(m, d)
    fb = foot_bodies(m)
    PELVIS = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    _R0 = d.xmat[PELVIS].reshape(3, 3); YAW_REF = np.arctan2(_R0[1,0], _R0[0,0])

    # 로그
    vs=[]; pit=[]; rol=[]; wp=[]; wl=[]; cop=[]; tau_rms=[]; contact=[]; yaws=[]
    fell = None; counter = 0
    n = int(a.seconds / dt)
    for k in range(n):
        t = k * dt
        tau = (target - d.qpos[7:]) * kps + (0.0 - d.qvel[6:]) * kds
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        counter += 1
        if counter % DEC == 0:
            if a.hold > 0.0 and abs(a.wz) < 1e-9:
                Rp = d.xmat[PELVIS].reshape(3, 3)
                e = (np.arctan2(Rp[1,0], Rp[0,0]) - YAW_REF + np.pi) % (2*np.pi) - np.pi
                cmd[2] = float(np.clip(-a.hold * e, -0.5, 0.5))
            qj = (d.qpos[7:] - dflt) * c["dof_pos_scale"]
            dqj = d.qvel[6:] * c["dof_vel_scale"]
            om = d.qvel[3:6] * c["ang_vel_scale"]
            ph = (counter * dt) % 0.8 / 0.8
            obs[:3] = om; obs[3:6] = gravity_orientation(d.qpos[3:7])
            obs[6:9] = cmd * cmd_scale
            obs[9:9+nA] = qj; obs[9+nA:9+2*nA] = dqj; obs[9+2*nA:9+3*nA] = action
            obs[9+3*nA:9+3*nA+2] = [np.sin(2*np.pi*ph), np.cos(2*np.pi*ph)]
            action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target = action * c["action_scale"] + dflt

        if t > a.settle:
            mujoco.mj_subtreeVel(m, d)
            R = d.xmat[PELVIS].reshape(3, 3)
            # 월드 x 가 아니라 '몸이 향한 방향'의 전진속도 (요 표류가 있으면 월드 x 는 과소평가)
            yaw0 = np.arctan2(R[1, 0], R[0, 0])
            v = d.subtree_linvel[0]
            vs.append(v[0] * np.cos(yaw0) + v[1] * np.sin(yaw0))
            yaws.append(yaw0)
            # roll/pitch (ZYX)
            rol.append(np.arctan2(R[2,1], R[2,2])); pit.append(-np.arcsin(np.clip(R[2,0],-1,1)))
            wp.append(R @ d.qvel[3:6])                      # pelvis ω (world)
            yaw = np.arctan2(R[1,0], R[0,0])
            cy, sy = np.cos(yaw), np.sin(yaw)
            Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
            wl.append(np.linalg.solve(Rz @ I_body @ Rz.T, d.subtree_angmom[0]))
            tau_rms.append(np.linalg.norm(d.ctrl))
            nc = 0
            for bid in fb:
                F, cp = foot_wrench(m, d, bid)
                if cp is not None:
                    nc += 1
                    cop.append((cp[0], cp[1], F[2]))
            contact.append(nc)
        if d.qpos[2] < 0.5:
            fell = t; break

    ok = fell is None
    vs = np.array(vs); wp = np.array(wp); wl = np.array(wl)
    cop = np.array(cop) if cop else np.zeros((1,3))
    res = dict(cmd=a.vx, ok=ok, fell=fell,
               vx=float(vs.mean()) if len(vs) else 0.0,
               vx_std=float(vs.std()) if len(vs) else 0.0,
               pitch=float(np.degrees(np.mean(pit))) if pit else 0.0,
               pitch_std=float(np.degrees(np.std(pit))) if pit else 0.0,
               roll_std=float(np.degrees(np.std(rol))) if rol else 0.0,
               wp_y=float(np.sqrt(np.mean(wp[:,1]**2))) if len(wp) else 0.0,
               wl_y=float(np.sqrt(np.mean(wl[:,1]**2))) if len(wl) else 0.0,
               corr_y=float(np.corrcoef(wp[:,1], wl[:,1])[0,1]) if len(wp) > 2 else 0.0,
               cop_x=float(np.mean(np.clip(np.where(cop[:,0]>0, cop[:,0]/0.12, -cop[:,0]/0.05),0,3))),
               cop_y=float(np.mean(np.abs(cop[:,1])/0.025)),
               cop_y_sat=float(np.mean(np.abs(cop[:,1])/0.025 > 0.9)),
               cop_x_sat=float(np.mean(np.where(cop[:,0]>0, cop[:,0]/0.12, -cop[:,0]/0.05) > 0.9)),
               ds=float(np.mean(np.array(contact) == 2)) if contact else 0.0,
               ss=float(np.mean(np.array(contact) == 1)) if contact else 0.0,
               fly=float(np.mean(np.array(contact) == 0)) if contact else 0.0,
               tau=float(np.mean(tau_rms)) if tau_rms else 0.0,
               yaw_drift=float(np.degrees(yaws[-1]-yaws[0])) if len(yaws)>1 else 0.0)
    if not a.quiet:
        r = f"{a.seconds:.0f}s OK" if ok else f"{fell:.1f}s 전도"
        print(f"cmd {a.vx:4.2f} | {r:>9s} | 실측 {res['vx']:5.2f} ({100*res['vx']/max(a.vx,1e-9):3.0f}%) "
              f"| pitch {res['pitch']:+5.1f}±{res['pitch_std']:.1f}° | CoP x {res['cop_x']:.2f} y {res['cop_y']:.2f} "
              f"| 이중지지 {100*res['ds']:.0f}% 비행 {100*res['fly']:.0f}% | 요표류 {res['yaw_drift']:+.0f}°")
    return res


if __name__ == "__main__":
    main()
