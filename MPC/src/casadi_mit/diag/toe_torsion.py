"""발끝만 구간 (뒤꿈치 힘 < 1 N) 의 명령 비틀림 분해 (MPC 해 · yaw hold 가 더한 몫 · 최종) vs 한 점 조건 l·Fy, 어긋남과 발 yaw 각속도의 상관 (Q&A 10/6 Q71).
사용: toe_torsion.py vx variant
"""
import sys, importlib.util
import numpy as np
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/diag")
import mujoco
spec = importlib.util.spec_from_file_location("walk", ROOT + "/10_walk.py")
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
import leg_control as LC
import controller as CT
from configs import resolve
from gait import cfg_get
vx, var = float(sys.argv[1]), sys.argv[2]
kw = W.parse_variant(resolve(var)); prof = kw.pop("profile", "ramp")
cfg, model, ctl, walk_t = W._setup(vx, 0, 0, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
m, d = model.m, model.d; l = float(cfg_get(cfg, "mpc.foot_half_length"))
cur = {}; rec = {0: {}, 1: {}}
o_w, o_h, o_s = CT.walking_stance_torque, LC.stance_yaw_hold_moment_z, LC.stance_torque
def ww(dyn, state, leg, u0, alpha, td, cfg_, yaw_hold_gains=None, line_proj=None):
    cur["leg"] = leg; rec[leg] = {"u0": np.asarray(u0, float).copy(), "alpha": alpha}
    return o_w(dyn, state, leg, u0, alpha, td, cfg_, yaw_hold_gains=yaw_hold_gains, line_proj=line_proj)
def wh(*a, **k):
    v = o_h(*a, **k); rec[cur["leg"]]["hold"] = float(v); return v
def ws(dyn, F, M):
    rec[cur.get("leg", 0)]["FM"] = (-np.asarray(F, float), -np.asarray(M, float)); return o_s(dyn, F, M)
CT.walking_stance_torque, LC.stance_yaw_hold_moment_z, LC.stance_torque = ww, wh, ws
f6 = np.zeros(6); w6 = np.zeros(6); rows = []
for _ in range(int(round((walk_t + 8.0) / m.opt.timestep))):
    rec = {0: {}, 1: {}}
    tl, ta = ctl.tick(); model.write_torque(tl, ta); mujoco.mj_step(m, d)
    if W._fell(d): break
    if d.time < walk_t + 3.0: continue
    for leg in (0, 1):
        r = rec[leg]
        if "FM" not in r: continue
        fb = model.foot_body[leg]; R = d.xmat[fb].reshape(3, 3); p = d.xpos[fb]
        fh = ft = 0.0
        for c in range(d.ncon):
            con = d.contact[c]
            if model._geom_foot[con.geom1] != leg and model._geom_foot[con.geom2] != leg: continue
            mujoco.mj_contactForce(m, d, c, f6)
            if (R.T @ (con.pos - p))[0] < 0.03: fh += f6[0]
            else: ft += f6[0]
        if not (ft > 5 and fh < 1): continue
        yaw = np.arctan2(R[1, 0], R[0, 0]); c_, s_ = np.cos(yaw), np.sin(yaw); Rt = np.array([[c_, s_, 0], [-s_, c_, 0], [0, 0, 1]])
        Fg, Mg = Rt @ r["FM"][0], Rt @ r["FM"][1]                     # 최종 (지면 → 로봇, 발 yaw 좌표)
        u0 = r["u0"]; Mmpc = (Rt @ u0[6 + 3 * leg:9 + 3 * leg])[2] * r["alpha"]   # MPC 해 Mz (지면 → 로봇)
        hold = -r["alpha"] * r.get("hold", 0.0)                       # yaw hold 가 더한 Mz (지면 → 로봇)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, fb, w6, 0)
        rows.append([Mmpc, hold, Mg[2], l * Fg[1], Mg[2] - l * Fg[1], w6[2]])
A = np.array(rows)
ex, wz = A[:, 4], A[:, 5]
print(f"[vx {vx}] 발끝만 틱 {len(A)} (뒤꿈치 힘 < 1 N)")
print(f"   지면 → 로봇 Mz [N·m] 평균 |·|: MPC 해 {np.mean(abs(A[:,0])):.2f} · yaw hold 가 더한 몫 {np.mean(abs(A[:,1])):.2f} · 최종 {np.mean(abs(A[:,2])):.2f}  |  한 점이 낼 수 있는 l·Fy {np.mean(abs(A[:,3])):.2f}")
print(f"   어긋남 (최종 Mz − l·Fy) 평균 |·| {np.mean(abs(ex)):.2f} N·m")
print(f"   발 yaw 각속도와 어긋남의 상관 {np.corrcoef(ex, wz)[0,1]:+.2f}, 부호 일치 비율 {np.mean(np.sign(ex) == np.sign(wz)):.2f}")
