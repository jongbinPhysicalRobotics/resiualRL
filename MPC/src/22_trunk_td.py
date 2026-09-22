"""착지에 정렬한 골반 흔들림 — 몸통 흔들림 중 착지·이륙 몫 (Q&A 9/22 Q12).

예정 착지 시각에 스텝을 정렬해 평균한 파형: 골반 pitch·roll 각속도, 골반 수직 가속, 양발 합력,
새 발·뒷발 접촉 비율 (0 ~ 150 ms, 10 ms 간격). 권장 구성, 0.5 / 0.7 m/s, 40 s.

사용:
  .venv/Scripts/python.exe MPC/src/22_trunk_td.py
"""
import os, sys, importlib
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
from pathlib import Path
from multiprocessing import Pool
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_model
walk = importlib.import_module("09_walk")
REC = dict(kp_up=300.0, swing_id=True, soft_land=True, lam_swing=True, wn_swing=30.0, zeta_swing=0.7, stance_frac=0.57,
           q_py=300, gate_sy=False, side_w=0.13, td_scale=1.25, cop_margin=0.9, wz_pelvis=1.0, wx_pelvis=0.5)
def run(vx):
    m, d = g1_model.load_torque(); g1_model.set_crouch(m, d)
    ctl = walk.WalkController(m, d, vx_cmd=vx, **REC)
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"); dt = m.opt.timestep
    P, R, AZ, TD, ON, GRF = [], [], [], [], [], []; prev = [True, True]; f6 = np.zeros(6)
    fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'{q}_ankle_roll_link') for q in ('left', 'right')]
    Wt = mujoco.mj_getTotalmass(m) * 9.81; newi = []
    for k in range(int(40 / dt)):
        t = k * dt; mujoco.mj_forward(m, d)
        if k % walk.DECIM == 0: ctl.update_mpc(d, t)
        if t > 8:
            Rp = d.xmat[pel].reshape(3, 3); yaw = np.arctan2(Rp[1, 0], Rp[0, 0])
            w = Rp @ d.qvel[3:6]; c, s = np.cos(yaw), np.sin(yaw)
            P.append(-s * w[0] + c * w[1]); R.append(c * w[0] + s * w[1])     # 몸 yaw frame pitch·roll rate
            AZ.append(d.qacc[2])
            fz = [0.0, 0.0]
            for ci in range(d.ncon):
                con = d.contact[ci]
                for i in range(2):
                    if m.geom_bodyid[con.geom1] == fb[i] or m.geom_bodyid[con.geom2] == fb[i]:
                        mujoco.mj_contactForce(m, d, ci, f6); fz[i] += abs(f6[0])
            ON.append([f > 20 for f in fz]); GRF.append(sum(fz) / Wt)
            for i in range(2):
                st = ctl.gait.in_stance(t, i)
                if st and not prev[i]: TD.append(len(P) - 1); newi.append(i)
                prev[i] = st
        d.ctrl[:] = ctl.torque(d, t); mujoco.mj_step(m, d)
    P, R, AZ, ON, GRF = map(np.array, (P, R, AZ, ON, GRF)); n = int(0.4 / dt)
    keep = [(k, i) for k, i in zip(TD, newi) if k + n < len(P)]
    idx = np.array([k for k, _ in keep])
    NEW = np.stack([ON[k:k + n, i] for k, i in keep]).mean(0); OLD = np.stack([ON[k:k + n, 1 - i] for k, i in keep]).mean(0)
    seg = lambda X: np.stack([X[k:k + n] for k in idx])
    out = {}
    for name, X in (("pitch", P), ("roll", R)):
        S = seg(X); mean = S.mean(0)                                          # 스텝 평균 파형 (착지 정렬)
        early = slice(0, int(0.08 / dt))                                      # 착지 후 0~80 ms
        out[name] = dict(rms_all=np.sqrt(np.mean(X ** 2)), rms_mean=np.sqrt(np.mean(mean ** 2)),
                         peak_early=np.abs(mean[early]).max(), peak_rest=np.abs(mean[early.stop:]).max(),
                         t_peak=1000 * dt * np.argmax(np.abs(mean)), resid=np.sqrt(np.mean((S - mean) ** 2)))
    A = seg(AZ).mean(0)
    out["az"] = (A[:int(0.08 / dt)].min(), A[:int(0.08 / dt)].max(), np.abs(A[int(0.08 / dt):]).max())
    out['prof'] = [(1000 * j * dt, seg(P).mean(0)[j], seg(AZ).mean(0)[j], seg(GRF).mean(0)[j], NEW[j], OLD[j]) for j in range(0, int(0.16 / dt), int(0.01 / dt))]
    return vx, out
if __name__ == "__main__":
    with Pool(2) as p: res = p.map(run, [0.5, 0.7])
    for vx, o in res:
        print(f"\nvx {vx} — 골반 각속도, 착지 시각 정렬 (스텝 = 0.4 s)")
        for ax in ("pitch", "roll"):
            r = o[ax]
            print(f"  {ax:5s}: 전체 RMS {r['rms_all']:.3f} rad/s | 매 스텝 반복되는 파형 RMS {r['rms_mean']:.3f} (스텝마다 다른 몫 {r['resid']:.3f}) | "
                  f"파형 최대: 착지 후 0~80 ms {r['peak_early']:.3f} vs 나머지 {r['peak_rest']:.3f} (최대 시각 +{r['t_peak']:.0f} ms)")
        a0, a1, a2 = o["az"]
        print('   착지 후 | 골반 pitch rate | 골반 수직가속 | 양발 합력 | 새 발 접촉 | 반대발 접촉')
        for tm, pr, az, gr, nw, od in o['prof']:
            print(f'   {tm:+5.0f}ms | {pr:+8.2f} rad/s | {az:+7.1f} m/s² | {100*gr:5.0f} % | {100*nw:5.0f} % | {100*od:5.0f} %')
        print(f"  골반 수직 가속도 평균 파형: 착지 후 0~80 ms {a0:+.1f} ~ {a1:+.1f} m/s², 나머지 구간 최대 |{a2:.1f}| m/s²")
