"""사전학습 정책 상세 계측 — 우리 Q8/gait 질문에 대응하는 양을 뽑는다."""
import sys, io, argparse
from pathlib import Path
import numpy as np, mujoco, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_g1_policy import load, gravity_orientation, composite_inertia, foot_bodies, foot_wrench


def run(vx, seconds=30.0, settle=8.0):
    m, d, policy, c = load()
    nA, nO = c["num_actions"], c["num_obs"]
    kps = np.array(c["kps"], np.float32); kds = np.array(c["kds"], np.float32)
    dflt = np.array(c["default_angles"], np.float32)
    cmd = np.array([vx, 0, 0], np.float32); cs = np.array(c["cmd_scale"], np.float32)
    dt = c["simulation_dt"]; DEC = c["control_decimation"]
    action = np.zeros(nA, np.float32); target = dflt.copy(); obs = np.zeros(nO, np.float32)
    mujoco.mj_forward(m, d)
    I_body = composite_inertia(m, d); fb = foot_bodies(m)
    PEL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    wp=[]; wl=[]; pit=[]; vs=[]; tau=[]
    st_prev=[False,False]; st_start=[0.0,0.0]; sw_start=[0.0,0.0]
    stance_dur=[]; swing_dur=[]; td_x=[]; foot_x_prev={}
    counter=0
    for k in range(int(seconds/dt)):
        t = k*dt
        d.ctrl[:] = (target - d.qpos[7:])*kps + (0.0 - d.qvel[6:])*kds
        mujoco.mj_step(m, d); counter += 1
        if counter % DEC == 0:
            qj=(d.qpos[7:]-dflt)*c["dof_pos_scale"]; dqj=d.qvel[6:]*c["dof_vel_scale"]
            obs[:3]=d.qvel[3:6]*c["ang_vel_scale"]; obs[3:6]=gravity_orientation(d.qpos[3:7])
            obs[6:9]=cmd*cs; obs[9:9+nA]=qj; obs[9+nA:9+2*nA]=dqj; obs[9+2*nA:9+3*nA]=action
            ph=(counter*dt)%0.8/0.8
            obs[9+3*nA:9+3*nA+2]=[np.sin(2*np.pi*ph), np.cos(2*np.pi*ph)]
            action=policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target=action*c["action_scale"]+dflt
        if t > settle:
            mujoco.mj_subtreeVel(m, d)
            R=d.xmat[PEL].reshape(3,3); yaw=np.arctan2(R[1,0],R[0,0])
            cy,sy=np.cos(yaw),np.sin(yaw); Rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
            wp.append(R@d.qvel[3:6])
            wl.append(np.linalg.solve(Rz@I_body@Rz.T, d.subtree_angmom[0]))
            pit.append(-np.arcsin(np.clip(R[2,0],-1,1)))
            vs.append(d.subtree_linvel[0][0]); tau.append(np.abs(d.ctrl).max())
            for i,bid in enumerate(fb):
                F,_=foot_wrench(m,d,bid); on = F[2] > 20
                if on and not st_prev[i]:
                    st_start[i]=t
                    if sw_start[i]: swing_dur.append(t-sw_start[i])
                    td_x.append(d.xpos[bid][0]-d.subtree_com[0][0])
                if (not on) and st_prev[i]:
                    sw_start[i]=t
                    if st_start[i]: stance_dur.append(t-st_start[i])
                st_prev[i]=on
        if d.qpos[2] < 0.5: return None
    wp=np.array(wp); wl=np.array(wl)
    return dict(vx=np.mean(vs), pit=np.degrees(np.mean(pit)), pit_std=np.degrees(np.std(pit)),
        wp=[np.sqrt(np.mean(wp[:,i]**2)) for i in range(3)],
        wl=[np.sqrt(np.mean(wl[:,i]**2)) for i in range(3)],
        corr=[np.corrcoef(wp[:,i],wl[:,i])[0,1] for i in range(3)],
        stance=np.mean(stance_dur) if stance_dur else 0, swing=np.mean(swing_dur) if swing_dur else 0,
        tau=np.mean(tau), td_x=np.mean(td_x) if td_x else 0)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("speeds", nargs="*", type=float, default=[0.5,1.0,1.5])
    a=ap.parse_args()
    for vx in a.speeds:
        r=run(vx)
        if r is None: print(f"cmd {vx}: 전도"); continue
        T=r["stance"]+r["swing"]
        print(f"\n━━ cmd {vx} m/s  (실측 {r['vx']:.2f}) ━━")
        print(f"  pelvis pitch      {r['pit']:+.1f} ± {r['pit_std']:.2f}°")
        print(f"  gait  stance {r['stance']*1000:.0f}ms  swing {r['swing']*1000:.0f}ms  "
              f"주기 {T*1000:.0f}ms  (stance_frac {r['stance']/max(T,1e-9):.2f})")
        print(f"  착지점 x (CoM 기준) {r['td_x']*100:+.1f} cm    최대토크 {r['tau']:.0f} Nm")
        print(f"  {'축':>6s} {'ω(pelvis)':>10s} {'ω(전신L)':>10s} {'상관':>7s}")
        for i,ax in enumerate(("roll","pitch","yaw")):
            print(f"  {ax:>6s} {r['wp'][i]:10.3f} {r['wl'][i]:10.3f} {r['corr'][i]:7.2f}")
