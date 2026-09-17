"""속도 정밀 스윕 — 보폭·gait 타이밍·지지구간 패턴을 한 번에 뽑는다.

사용: .venv/Scripts/python.exe "deployed RL/analysis/sweep.py" --lo 0.1 --hi 1.2 --step 0.05
"""
import sys, argparse, json
from pathlib import Path
import numpy as np, mujoco, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_g1_policy import load, gravity_orientation, composite_inertia, foot_bodies, foot_wrench


def one(m, d, policy, c, vx, seconds, settle):
    mujoco.mj_resetData(m, d)
    nA, nO = c["num_actions"], c["num_obs"]
    kps=np.array(c["kps"],np.float32); kds=np.array(c["kds"],np.float32)
    dflt=np.array(c["default_angles"],np.float32); cs=np.array(c["cmd_scale"],np.float32)
    dt=c["simulation_dt"]; DEC=c["control_decimation"]
    act=np.zeros(nA,np.float32); tgt=dflt.copy(); obs=np.zeros(nO,np.float32)
    mujoco.mj_forward(m,d)
    fb=foot_bodies(m); PEL=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"pelvis")
    I_body=composite_inertia(m,d)

    vs=[];pit=[];copx=[];copy_=[];nc=[];tau=[]
    prev=[False,False]; st0=[0.,0.]; sw0=[0.,0.]
    stance=[];swing=[];lat=[];wid=[];fore=[];last={}; last_td={}
    cnt=0
    for k in range(int(seconds/dt)):
        t=k*dt
        d.ctrl[:]=(tgt-d.qpos[7:])*kps+(0.0-d.qvel[6:])*kds
        mujoco.mj_step(m,d); cnt+=1
        if cnt%DEC==0:
            obs[:3]=d.qvel[3:6]*c["ang_vel_scale"]; obs[3:6]=gravity_orientation(d.qpos[3:7])
            obs[6:9]=np.array([vx,0,0],np.float32)*cs
            obs[9:9+nA]=(d.qpos[7:]-dflt)*c["dof_pos_scale"]
            obs[9+nA:9+2*nA]=d.qvel[6:]*c["dof_vel_scale"]; obs[9+2*nA:9+3*nA]=act
            ph=(cnt*dt)%0.8/0.8; obs[9+3*nA:9+3*nA+2]=[np.sin(2*np.pi*ph),np.cos(2*np.pi*ph)]
            act=policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            tgt=act*c["action_scale"]+dflt
        if t>settle:
            R=d.xmat[PEL].reshape(3,3); yaw=np.arctan2(R[1,0],R[0,0])
            cy,sy=np.cos(yaw),np.sin(yaw); com=d.subtree_com[0]
            mujoco.mj_subtreeVel(m,d); v=d.subtree_linvel[0]
            vs.append(v[0]*cy+v[1]*sy); pit.append(-np.arcsin(np.clip(R[2,0],-1,1)))
            tau.append(np.abs(d.ctrl).max())
            n=0
            for i,bid in enumerate(fb):
                F,cp=foot_wrench(m,d,bid); on=F[2]>20
                if cp is not None:
                    n+=1
                    copx.append(cp[0]/0.12 if cp[0]>0 else -cp[0]/0.05)
                    copy_.append(abs(cp[1])/0.025)
                if on and not prev[i]:
                    st0[i]=t
                    if sw0[i]: swing.append(t-sw0[i])
                    r=d.xpos[bid]-com
                    lx=cy*r[0]+sy*r[1]; ly=-sy*r[0]+cy*r[1]
                    lat.append(abs(ly)); last[i]=ly
                    if 0 in last and 1 in last: wid.append(abs(last[0]-last[1]))
                    if i in last_td:
                        fore.append(np.linalg.norm(d.xpos[bid][:2]-last_td[i][:2]))
                    last_td[i]=d.xpos[bid][:2].copy()
                if (not on) and prev[i]:
                    sw0[i]=t
                    if st0[i]: stance.append(t-st0[i])
                prev[i]=on
            nc.append(n)
        if d.qpos[2]<0.5: return None
    nc=np.array(nc)
    T=(np.mean(stance)+np.mean(swing)) if stance and swing else 0
    return dict(cmd=vx, vx=float(np.mean(vs)),
        pitch=float(np.degrees(np.mean(pit))), pitch_sd=float(np.degrees(np.std(pit))),
        width=float(np.mean(wid)*100) if wid else 0, lat=float(np.mean(lat)*100) if lat else 0,
        stride=float(np.mean(fore)*100) if fore else 0,
        stance=float(np.mean(stance)*1000) if stance else 0,
        swing=float(np.mean(swing)*1000) if swing else 0,
        T=float(T*1000), sf=float(np.mean(stance)/T) if T else 0,
        freq=float(1.0/T) if T else 0,
        ds=float(np.mean(nc==2)*100), ss=float(np.mean(nc==1)*100), fly=float(np.mean(nc==0)*100),
        copx=float(np.mean(np.clip(copx,0,3))), copy=float(np.mean(copy_)),
        tau=float(np.mean(tau)))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--lo",type=float,default=0.1); ap.add_argument("--hi",type=float,default=1.2)
    ap.add_argument("--step",type=float,default=0.05); ap.add_argument("--seconds",type=float,default=25.0)
    ap.add_argument("--settle",type=float,default=8.0)
    ap.add_argument("--out",type=str,default="")
    a=ap.parse_args()
    m,d,policy,c=load()
    hdr=f"{'cmd':>5s} {'실측':>6s} {'보폭':>6s} {'횡오프':>6s} {'stride':>7s} {'주기':>6s} {'stance':>7s} {'swing':>6s} {'sf':>5s} {'Hz':>5s} {'DS%':>5s} {'fly%':>5s} {'CoPx':>5s} {'CoPy':>5s} {'pitch':>11s} {'τmax':>5s}"
    print(hdr); print("-"*len(hdr))
    rows=[]
    v=a.lo
    while v <= a.hi+1e-9:
        r=one(m,d,policy,c,round(v,3),a.seconds,a.settle)
        if r is None:
            print(f"{v:5.2f}  전도")
        else:
            rows.append(r)
            print(f"{r['cmd']:5.2f} {r['vx']:6.2f} {r['width']:6.1f} {r['lat']:6.1f} {r['stride']:7.1f} "
                  f"{r['T']:6.0f} {r['stance']:7.0f} {r['swing']:6.0f} {r['sf']:5.2f} {r['freq']:5.2f} "
                  f"{r['ds']:5.1f} {r['fly']:5.1f} {r['copx']:5.2f} {r['copy']:5.2f} "
                  f"{r['pitch']:+6.1f}±{r['pitch_sd']:4.2f} {r['tau']:5.0f}")
        v += a.step
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"\n저장: {a.out}")
