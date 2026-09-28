"""입각 발 진단 — 뒤꿈치가 바깥으로 도는가, CoP 가 발 끝에 붙는가, 실제로 몇 점으로 서 있는가 (Q&A 9/28 Q1).

    GAIT_ON=1 .venv/bin/python MPC/src/casadi_mit/13_foot_diag.py --vx 0.65 --seconds 16 \
        --var profile=ramp,fix=slide+yawref,rai=0.2,ypath=0.3 [--profile]

지표 (걷기 시작 + skip 초 이후, 다리별):
  뒤꿈치 회전   입각 (착지 → 이륙) 동안 발 yaw 변화. + = 뒤꿈치가 몸 바깥으로 (왼발 −yaw, 오른발 +yaw)
  CoP 포화      MPC 명령 CoP = −My/Fz 가 ±0.98·foot_half_length 에 붙은 입각 시간 비율 (앞끝 / 뒤끝)
  한 점 접촉    MuJoCo 접촉력의 90 % 이상이 발 중심 앞 (앞꿈치만) / 뒤 (뒤꿈치만) 에 걸린 입각 시간 비율
  |Mz| p95      MPC 가 디딤 발에 요구한 yaw 모멘트
--profile: 입각 10 구간별 평균 (뒤꿈치 회전 누적, 뒤꿈치/앞꿈치 수직력, 발 pitch, MPC Mz·My)
시뮬레이션은 10_walk.headless 와 같은 순서 (tick → write_torque → mj_step) — 계측이 시뮬을 바꾸지 않는다.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("walk10", HERE / "10_walk.py")
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)

D = np.degrees
SGN = (-1.0, 1.0)          # heel-out 부호: 왼발 −yaw, 오른발 +yaw


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def record(vx, vy, wz, seconds, var):
    kw = W.parse_variant(var)
    prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(vx, vy, wz, kw.pop("solver", "osqp"), kw.pop("armature", True), prof,
                                       False, False, kw.pop("friction_scale", 1.0), True, False, **kw)
    m, d = model.m, model.d
    n = int(round((walk_t + seconds) / m.opt.timestep))
    f6 = np.zeros(6)
    R = {k: [] for k in ("t", "walk", "fyaw", "fpitch", "act", "fz_ht", "u")}
    fell = None
    for _ in range(n):
        tau_leg, tau_arm = ctl.tick()
        s = ctl.state
        ht = np.zeros((2, 2))                                # [다리, 뒤꿈치/앞꿈치] 수직력
        for c in range(d.ncon):
            con = d.contact[c]
            for leg in (0, 1):
                if model._geom_foot[con.geom1] == leg or model._geom_foot[con.geom2] == leg:
                    mujoco.mj_contactForce(m, d, c, f6)
                    fz = abs((con.frame.reshape(3, 3).T @ f6[:3])[2])
                    xl = s.foot_R_W[leg][:, 0].dot(con.pos - s.foot_pos_W[leg])
                    ht[leg, 1 if xl > 0 else 0] += fz
        Rf = s.foot_R_W
        R["t"].append(s.t)
        R["walk"].append(ctl.mode == "walking")
        R["fyaw"].append([np.arctan2(Rf[l][1, 0], Rf[l][0, 0]) for l in (0, 1)])
        R["fpitch"].append([np.arcsin(np.clip(-Rf[l][2, 0], -1, 1)) for l in (0, 1)])   # + = 앞꿈치 아래
        R["act"].append([bool(ctl._active(l, s.t)) for l in (0, 1)])
        R["fz_ht"].append(ht)
        R["u"].append(ctl.u_hold.copy() if ctl.u_hold is not None else np.full(12, np.nan))
        model.write_torque(tau_leg, tau_arm)
        mujoco.mj_step(m, d)
        if W._fell(d):
            fell = float(d.time)
            break
    out = {k: np.asarray(v) for k, v in R.items()}
    out["fell"] = fell
    out["a"] = float(cfg.mpc.foot_half_length)
    return out


def _stances(t, act, leg, t0):
    a = act[:, leg].astype(int)
    on = np.where(np.diff(a) == 1)[0] + 1
    off = np.where(np.diff(a) == -1)[0] + 1
    for i in on[t[on] > t0]:
        j = off[off > i]
        if len(j):
            yield i, int(j[0])


def summarize(R, skip=6.0, profile=False, nb=10):
    t, act, fy, fp, ht, u = R["t"], R["act"], R["fyaw"], R["fpitch"], R["fz_ht"], R["u"]
    tw = t[np.argmax(R["walk"])]
    sel = t > tw + skip
    a = R["a"]
    Fz, My, Mz = u[:, [2, 5]], u[:, [7, 10]], u[:, [8, 11]]
    print(f"  걷기 시작 {tw:.1f} s, 끝 {t[-1]:.1f} s" + (f"  (전도 {R['fell']:.1f} s)" if R["fell"] else ""))
    for leg in (0, 1):
        st = sel & act[:, leg] & (Fz[:, leg] > 20)
        cop = -My[st, leg] / Fz[st, leg]
        tot = ht[st, leg].sum(1) + 1e-9
        rot = [SGN[leg] * D(_wrap(fy[j - 1, leg] - fy[i, leg])) for i, j in _stances(t, act, leg, tw + skip)]
        print(f"  {'LR'[leg]}: 뒤꿈치 회전 {np.mean(rot):+5.2f}° ±{np.std(rot):.2f} ({len(rot)} 걸음) | "
              f"CoP 포화 앞끝 {np.mean(cop > 0.98 * a) * 100:4.1f} % 뒤끝 {np.mean(cop < -0.98 * a) * 100:4.1f} % | "
              f"한 점 접촉 앞꿈치만 {np.mean(ht[st, leg, 0] < 0.1 * tot) * 100:4.1f} % 뒤꿈치만 "
              f"{np.mean(ht[st, leg, 1] < 0.1 * tot) * 100:4.1f} % | |Mz| p95 {np.percentile(np.abs(Mz[st, leg]), 95):.2f} N·m")
        if profile:
            P = []
            for i, j in _stances(t, act, leg, tw + skip):
                idx = np.linspace(i, j - 1, nb + 1).astype(int)
                P.append([[SGN[leg] * D(_wrap(fy[idx[k + 1], leg] - fy[i, leg])),
                           ht[idx[k]:idx[k + 1] + 1, leg, 0].mean(), ht[idx[k]:idx[k + 1] + 1, leg, 1].mean(),
                           D(fp[idx[k]:idx[k + 1] + 1, leg].mean()),
                           Mz[idx[k]:idx[k + 1] + 1, leg].mean(), My[idx[k]:idx[k + 1] + 1, leg].mean()]
                          for k in range(nb)])
            print("     입각 %   뒤꿈치회전°  뒤꿈치Fz  앞꿈치Fz  발pitch°  MPC Mz   MPC My")
            for k, r in enumerate(np.mean(P, 0)):
                print(f"     {k * 10:3d}-{k * 10 + 10:3d}  {r[0]:+7.2f}    {r[1]:7.1f}  {r[2]:7.1f}   {r[3]:+6.2f}   {r[4]:+6.2f}  {r[5]:+7.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vx", type=float, default=0.65)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--var", nargs="+", default=["profile=ramp,fix=slide+yawref,rai=0.2,ypath=0.3"])
    ap.add_argument("--skip", type=float, default=6.0, help="걷기 시작 뒤 버리는 시간 [s]")
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()
    for var in a.var:
        print(f"== vx {a.vx} vy {a.vy} wz {a.wz}  {var}")
        summarize(record(a.vx, a.vy, a.wz, a.seconds, var), a.skip, a.profile)


if __name__ == "__main__":
    main()
