"""서 있기·걷기에서 발목 중점 / 발바닥 중심 중점 / 전신 CoM / 골반 / 실측 CoP 의 xy 비교 (Q&A 9/24 Q9)."""
import sys, importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import mujoco
import g1_model, mpc_srb, gait

walk = importlib.import_module("09_walk")
KW = dict(kp_up=300.0, swing_id=True, soft_land=True, lam_swing=True, wn_swing=30.0, zeta_swing=0.7,
          stance_frac=0.57, q_py=300, gate_sy=False, side_w=0.13, td_scale=1.25, cop_margin=0.9,
          wz_pelvis=1.0, wx_pelvis=0.5)


def run(gait_on, vx, seconds, settle):
    gait.GAIT_ON = gait_on
    m, d = g1_model.load_torque(); g1_model.set_crouch(m, d)
    ctl = walk.WalkController(m, d, vx_cmd=vx, **KW)
    p = ctl.params
    xc = (p.l_t - p.l_h) / 2.0                        # 발바닥 사각형 중심의 발목 대비 앞 오프셋
    sids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in mpc_srb.FOOT_SITES]
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{q}_ankle_roll_link") for q in ("left", "right")]
    f6 = np.zeros(6)
    R = []
    dt = m.opt.timestep
    for k in range(int(seconds / dt)):
        t = k * dt
        mujoco.mj_forward(m, d)
        if k % walk.DECIM == 0:
            ctl.update_mpc(d, t)
        if t > settle and k % 5 == 0:
            ank = np.array([d.site_xpos[s] for s in sids])
            ctr = np.array([d.site_xpos[s] + d.site_xmat[s].reshape(3, 3) @ np.array([xc, 0.0, -p.h_sole]) for s in sids])
            # 실측 CoP (지면 접촉력 가중 평균)
            fz_sum, cop, fz_foot = 0.0, np.zeros(3), np.zeros(2)
            for c in range(d.ncon):
                con = d.contact[c]
                mujoco.mj_contactForce(m, d, c, f6)
                fw = con.frame.reshape(3, 3).T @ f6[:3]
                fz = abs(fw[2])
                fz_sum += fz; cop += fz * con.pos
                for i in range(2):
                    if fb[i] in (m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]):
                        fz_foot[i] += fz
            cop = cop / fz_sum if fz_sum > 1e-6 else np.full(3, np.nan)
            st = [ctl.gait.in_stance(t, i) for i in range(2)]
            yaw = mpc_srb.quat_to_euler_zyx(d.qpos[3:7])[2]
            R.append(dict(t=t, ank=ank, ctr=ctr, com=d.subtree_com[0].copy(), pel=d.xpos[pel].copy(),
                          cop=cop, st=st, yaw=yaw, fz=fz_foot))
        d.ctrl[:] = ctl.torque(d, t)
        mujoco.mj_step(m, d)
    return R, xc


def body(v, yaw):                                    # world xy → 몸 yaw frame (x 앞, y 왼쪽)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])


def stand():
    R, xc = run(0, 0.0, 20.0, 5.0)
    r = R[-1]
    mid_a = r["ank"].mean(0); mid_c = r["ctr"].mean(0)
    print(f"■ 서 있기 (보행 스케줄 끔, 20 s 끝 시점).  발바닥 중심 = 발목 + 앞 {100*xc:.1f} cm\n")
    print(f"  {'':22s} {'x [cm]':>8s} {'y [cm]':>8s}   발목 중점 대비 x / y [cm]")
    rows = [("왼 발목", r["ank"][0]), ("오른 발목", r["ank"][1]), ("발목 중점", mid_a),
            ("발바닥 중심 중점", mid_c), ("전신 CoM", r["com"]), ("골반 원점", r["pel"]), ("실측 CoP", r["cop"])]
    for name, v in rows:
        dv = v - mid_a
        print(f"  {name:22s} {100*v[0]:8.2f} {100*v[1]:8.2f}   {100*dv[0]:+7.2f} / {100*dv[1]:+6.2f}")
    print(f"  양발 하중: 왼 {r['fz'][0]:.0f} N, 오른 {r['fz'][1]:.0f} N")
    com = np.array([q["com"] for q in R]); print(f"  5~20 s 동안 CoM xy 변동 (max−min): x {100*np.ptp(com[:,0]):.3f} cm, y {100*np.ptp(com[:,1]):.3f} cm")


def walking(vx):
    R, xc = run(1, vx, 40.0, 8.0)
    D = {k: [] for k in ("ds", "ssL", "ssR")}
    allrows = []
    for r in R:
        mid_a = r["ank"].mean(0)
        dcom = body(r["com"] - mid_a, r["yaw"])
        dcop = body(r["cop"] - mid_a, r["yaw"])
        key = "ds" if all(r["st"]) else ("ssL" if r["st"][0] else "ssR")
        D[key].append(np.r_[dcom, dcop])
        allrows.append(np.r_[dcom, dcop])
        # 한발 지지 때는 디딤 발목 기준도
    A = np.array(allrows)
    print(f"\n■ 걷기 vx {vx} m/s (8~40 s).  기준 = 양 발목 중점, 몸 방향 좌표 (x 앞, y 왼쪽) [cm]\n")
    print(f"  {'구간':16s} {'비율':>5s} | {'CoM−중점 x':>11s} {'CoM−중점 y':>11s} | {'CoP−중점 x':>11s} {'CoP−중점 y':>11s}")
    for key, name in (("ds", "양발 지지"), ("ssL", "왼발만 디딤"), ("ssR", "오른발만 디딤")):
        X = np.array(D[key])
        mu = X.mean(0)
        print(f"  {name:16s} {100*len(X)/len(A):4.0f}% | {100*mu[0]:+11.1f} {100*mu[1]:+11.1f} | {100*mu[2]:+11.1f} {100*mu[3]:+11.1f}")
    mu = A.mean(0); lo = np.percentile(A, 5, axis=0); hi = np.percentile(A, 95, axis=0)
    print(f"  {'전체 평균':16s} {'':5s} | {100*mu[0]:+11.1f} {100*mu[1]:+11.1f} | {100*mu[2]:+11.1f} {100*mu[3]:+11.1f}")
    print(f"  {'5~95 % 범위':16s} {'':5s} | {100*lo[0]:+5.1f}~{100*hi[0]:+4.1f} {100*lo[1]:+5.1f}~{100*hi[1]:+4.1f} |"
          f" {100*lo[2]:+5.1f}~{100*hi[2]:+4.1f} {100*lo[3]:+5.1f}~{100*hi[3]:+4.1f}")
    # 한발 지지 때 CoM 과 디딤 발바닥 중심의 y 거리
    ss = []
    for r in R:
        if all(r["st"]) or not any(r["st"]):
            continue
        i = 0 if r["st"][0] else 1
        sgn = 1.0 if i == 0 else -1.0                 # 왼발 = +y 쪽
        dy = body(r["com"] - r["ctr"][i], r["yaw"])[1] * sgn     # + 면 디딤 발보다 바깥, − 면 안쪽(몸 가운데 쪽)
        ss.append(dy)
    ss = np.array(ss)
    w = np.mean([abs(body(r["ank"][0] - r["ank"][1], r["yaw"])[1]) for r in R])
    print(f"\n  양 발목 좌우 간격 평균 {100*w:.1f} cm → 발목 중점에서 각 발목까지 {50*w:.1f} cm")
    print(f"  한발 지지 때 CoM 이 디딤 발바닥 중심보다 몸 가운데 쪽으로 평균 {-100*ss.mean():.1f} cm (5~95 %: {-100*np.percentile(ss,95):.1f} ~ {-100*np.percentile(ss,5):.1f})")


if __name__ == "__main__":
    stand()
    walking(0.5)
