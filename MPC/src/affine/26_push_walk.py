"""걷는 중 밀기 시험 — 구성별로 버티는 최대 힘 (Q&A 9/25 Q1: 참조 성형이 외란에 약하게 만드나).

0.5 m/s 로 걷다가 t_push 에 골반에 F [N] 을 0.1 s. 민 뒤 8 s 를 버티면 통과.
사용: GAIT_ON=1 .venv/Scripts/python.exe MPC/src/affine/26_push_walk.py
구성·방향·힘·시점은 아래 CFGS / DIRS / FORCES / T_PUSH 를 고쳐서. 변형 문자열은 18_gait_quality 와 같다.
"""
import sys, importlib, itertools
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_model                                   # noqa: E402
walk = importlib.import_module("09_walk")
qual = importlib.import_module("18_gait_quality")

BASE = "tds=1.25,copm=0.9,ctl=affine"
CFGS = {
    "H (참조 성형)":   BASE + ",wzp=1,wxp=0,refshape=1",
    "H (성형 없음)":   BASE + ",wzp=1,wxp=0",
    "R (섞기)":        BASE + ",wzp=1,wxp=0.5",
}
DIRS = {"앞 +x": (1.0, 0.0, 0.0), "뒤 −x": (-1.0, 0.0, 0.0), "옆 +y": (0.0, 1.0, 0.0)}
FORCES = [50, 100, 150, 200, 250, 300]
T_PUSH = [20.0, 20.4]          # 걸음 주기 0.8 s 의 반 차이 — 지지 발이 서로 반대
VX = 0.5


def run(cfg, dname, F, t_push):
    kw = qual.parse(CFGS[cfg])
    Ctor = qual.make_ctor(kw)
    m, d = g1_model.load_torque(); g1_model.set_crouch(m, d)
    ctl = Ctor(m, d, vx_cmd=VX, kp_up=300.0, swing_id=True, soft_land=True, lam_swing=True,
               wn_swing=30.0, zeta_swing=0.7, stance_frac=0.57, q_py=300, gate_sy=False, **kw)
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    f = F * np.array(DIRS[dname])
    dt = m.opt.timestep
    fell = None
    for k in range(int((t_push + 8.0) / dt)):
        t = k * dt
        mujoco.mj_forward(m, d)
        if k % walk.DECIM == 0:
            ctl.update_mpc(d, t)
        tau = ctl.torque(d, t)
        d.xfrc_applied[:] = 0.0
        if t_push <= t < t_push + 0.1:
            d.xfrc_applied[pel, :3] = f
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        if d.qpos[2] < 0.5:
            fell = t; break
    return cfg, dname, F, t_push, fell


if __name__ == "__main__":
    jobs = list(itertools.product(CFGS, DIRS, FORCES, T_PUSH))
    with Pool(16) as pool:
        res = pool.starmap(run, jobs)
    ok = {(c, dn, F, tp): fell is None for c, dn, F, tp, fell in res}
    print(f"걷는 중 밀기 (vx {VX}, 골반에 0.1 s, 민 뒤 8 s 버티면 통과) — 버틴 최대 힘 [N]  (시점 {T_PUSH[0]} s / {T_PUSH[1]} s)\n")
    print(f"{'구성':<16s} " + " ".join(f"{dn:>14s}" for dn in DIRS))
    for c in CFGS:
        cells = []
        for dn in DIRS:
            vals = []
            for tp in T_PUSH:
                best = 0
                for F in FORCES:
                    if ok[(c, dn, F, tp)]:
                        best = F
                    else:
                        break                     # 처음 넘어진 힘에서 멈춤 (그 위는 세지 않음)
                vals.append(best)
            cells.append(f"{vals[0]:>5d} / {vals[1]:<5d}")
        print(f"{c:<16s} " + " ".join(f"{x:>14s}" for x in cells))
    print("\n전체 결과 (✓ 버팀, ✗ 넘어진 시각)")
    for c, dn, F, tp, fell in res:
        if fell is not None:
            print(f"  {c:<14s} {dn:<6s} {F:4d} N @ {tp:.1f} s  ✗ {fell:.1f} s")
