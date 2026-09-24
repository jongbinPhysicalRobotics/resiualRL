"""착지에 정렬한 골반 흔들림 — 몸통 흔들림 중 착지·이륙 몫 (Q&A 9/22 Q12·Q13).

예정 착지 시각에 스텝을 정렬해 평균한 파형으로:
  골반 pitch 각속도의 최대 (착지 후 0~80 ms = 뒷발 이륙 포함 / 나머지), 그 시각,
  골반 수직 가속 최저 (0~80 ms), 양발 합력 최저 (착지 후 30~150 ms), 골반 pitch 각속도 전체 RMS.
--detail 이면 0 ~ 150 ms 파형을 10 ms 간격으로.

사용:
  .venv/Scripts/python.exe MPC/src/baseline/22_trunk_td.py --vx 0.5 0.7 --var "" "swing_prof=1" --detail
  (--var 은 권장 구성 위에 덮어쓸 WalkController 인자, 예: "swing_prof=2,lo_ramp=0.03")
"""
import os, sys, argparse, importlib, itertools
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
from multiprocessing import Pool
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_model
walk = importlib.import_module("09_walk")

REC = dict(kp_up=300.0, swing_id=True, soft_land=True, lam_swing=True, wn_swing=30.0, zeta_swing=0.7, stance_frac=0.57,
           q_py=300, gate_sy=False, side_w=0.13, td_scale=1.25, cop_margin=0.9, wz_pelvis=1.0, wx_pelvis=0.5)
BOOL = ("soft_land", "swing_id", "lam_swing", "early_td", "liftoff_fix")


def parse(var):
    kw = dict(REC)
    for item in filter(None, var.split(",")):
        k, v = item.split("=")
        kw[k] = bool(float(v)) if k in BOOL else float(v)
    return kw


def run(vx, var, seconds=40.0, settle=8.0):
    m, d = g1_model.load_torque(); g1_model.set_crouch(m, d)
    ctl = walk.WalkController(m, d, vx_cmd=vx, **parse(var))
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"); dt = m.opt.timestep
    fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{q}_ankle_roll_link") for q in ("left", "right")]
    Wt = mujoco.mj_getTotalmass(m) * 9.81; f6 = np.zeros(6)
    P, AZ, GRF, ON, TD, NEWI, VS, CMD = [], [], [], [], [], [], [], []
    prev = [True, True]; fell = None
    for k in range(int(seconds / dt)):
        t = k * dt; mujoco.mj_forward(m, d)
        if k % walk.DECIM == 0:
            ctl.update_mpc(d, t)
        if t > settle:
            Rp = d.xmat[pel].reshape(3, 3); yaw = np.arctan2(Rp[1, 0], Rp[0, 0])
            w = Rp @ d.qvel[3:6]; c, s = np.cos(yaw), np.sin(yaw)
            P.append(-s * w[0] + c * w[1]); AZ.append(d.qacc[2])
            fz = [0.0, 0.0]
            for ci in range(d.ncon):
                con = d.contact[ci]
                for i in range(2):
                    if m.geom_bodyid[con.geom1] == fb[i] or m.geom_bodyid[con.geom2] == fb[i]:
                        mujoco.mj_contactForce(m, d, ci, f6); fz[i] += abs(f6[0])
            ON.append([f > 20 for f in fz]); GRF.append(sum(fz) / Wt)
            # MPC 명령: 발별 pitch 모멘트 (몸 yaw frame) 와 Fz — 원인 좁히기용 (Q13)
            CMD.append([[-s * ctl.wr[j][3] + c * ctl.wr[j][4], ctl.wr[j][2]] for j in range(2)])
            x0 = ctl.get_state(d, t); psi = ctl._yaw_unwrap
            VS.append(np.cos(psi) * x0[9] + np.sin(psi) * x0[10])
            for i in range(2):
                st = ctl.gait.in_stance(t, i)
                if st and not prev[i]:
                    TD.append(len(P) - 1); NEWI.append(i)
                prev[i] = st
        d.ctrl[:] = ctl.torque(d, t); mujoco.mj_step(m, d)
        if d.qpos[2] < 0.5:
            fell = t; break
    if len(P) < 1000:
        return dict(vx=vx, var=var or "권장 구성", fell=fell)
    P, AZ, GRF, ON, CMD = map(np.array, (P, AZ, GRF, ON, CMD)); n = int(0.4 / dt)
    keep = [(k, i) for k, i in zip(TD, NEWI) if k + n < len(P)]
    seg = lambda X: np.stack([X[k:k + n] for k, _ in keep])
    Pm, Am, Gm = seg(P).mean(0), seg(AZ).mean(0), seg(GRF).mean(0)
    NEW = np.stack([ON[k:k + n, i] for k, i in keep]).mean(0)
    OLD = np.stack([ON[k:k + n, 1 - i] for k, i in keep]).mean(0)
    MYN = np.stack([CMD[k:k + n, i, 0] for k, i in keep]).mean(0)      # 새 발 명령 pitch 모멘트
    FZO = np.stack([CMD[k:k + n, 1 - i, 1] for k, i in keep]).mean(0)  # 뒷발 명령 Fz
    e80, e30, e150 = int(0.08 / dt), int(0.03 / dt), int(0.15 / dt)
    prof = [(1000 * j * dt, Pm[j], Am[j], Gm[j], NEW[j], OLD[j], MYN[j], FZO[j]) for j in range(0, e150 + 1, int(0.01 / dt))]
    return dict(vx=vx, var=var or "권장 구성", fell=fell, v=float(np.mean(VS)),
                pk_early=np.abs(Pm[:e80]).max(), pk_rest=np.abs(Pm[e80:]).max(),
                t_pk=1000 * dt * np.argmax(np.abs(Pm)), az_min=Am[:e80].min(), grf_min=Gm[e30:e150].min(),
                p_rms=np.sqrt(np.mean(P ** 2)), prof=prof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, nargs="+", default=[0.5, 0.7])
    ap.add_argument("--var", nargs="+", default=[""])
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    jobs = [(vx, var, a.seconds) for vx, var in itertools.product(a.vx, a.var)]
    with Pool(min(16, len(jobs))) as pool:
        res = pool.starmap(run, jobs)
    print("착지 정렬 골반 파형 — 스텝 평균. pitch 각속도 최대 = |평균 파형| 의 최대\n")
    print(f"{'vx':>4s} {'변형':<26s} | {'추종':>4s} | {'pitch각속도 최대: 0~80ms · 나머지 (시각)':>34s} | {'수직가속 최저':>10s} | {'합력 최저':>7s} | {'pitch각속도 RMS':>12s}")
    for r in res:
        if "pk_early" not in r:
            print(f"{r['vx']:4.1f} {r['var']:<26s} | {r['fell']:.1f} s 전도"); continue
        tag = "" if r["fell"] is None else f" ✗{r['fell']:.0f}s"
        print(f"{r['vx']:4.1f} {r['var'] + tag:<26s} | {100 * r['v'] / r['vx']:3.0f}% | "
              f"{r['pk_early']:7.2f} · {r['pk_rest']:5.2f} rad/s (+{r['t_pk']:3.0f} ms) | {r['az_min']:+7.1f} m/s² | "
              f"{100 * r['grf_min']:5.0f} % | {r['p_rms']:8.3f} rad/s", flush=True)
    if a.detail:
        for r in res:
            if "prof" not in r:
                continue
            print(f"\n  vx {r['vx']}  {r['var']}")
            print("   착지 후 | 골반 pitch 각속도 | 골반 수직가속 | 양발 합력 | 새 발 접촉 | 뒷발 접촉 | 새 발 명령 my | 뒷발 명령 Fz")
            for tm, pr, az, gr, nw, od, myn, fzo in r["prof"]:
                print(f"   {tm:+5.0f}ms | {pr:+8.2f} rad/s | {az:+7.1f} m/s² | {100 * gr:5.0f} % | {100 * nw:5.0f} % | {100 * od:5.0f} % | {myn:+7.1f} N·m | {fzo:6.0f} N")


if __name__ == "__main__":
    main()
