"""MPC 가 왜 CoP 를 발끝으로 계획하나 (Q&A 10/6 Q79). 걷기 + 3 s 부터 seconds 동안, 매 틱 적용 중인 MPC 해 (u_hold = 첫 칸) 로:
MPC 몸 (축소 몸체) CoM c 둘레 pitch 모멘트 (몸 진행 방향 좌표, + = 앞으로 숙이는 쪽) 를 발마다 세 항으로 나눈다 (유도):
  수직력 몫   −x·Fz          (x = 발 site − c 의 앞뒤, Fz 수직력)
  앞뒤 힘 몫   z·Fx           (z = 발 site − c 의 높이, 음수)
  발 모멘트 몫 My             (= CoP 를 site 에서 앞뒤로 옮긴 몫, CoP x = −My/Fz, 발 yaw 좌표)
디딤 발 (일정상 디딤) 의 경과 10 % 칸마다 평균, 그리고 그때 다른 발 Fz (두 발 지지 여부), 두 발 합 · 몸통 pitch · pitch 속도.
그리고 MPC 모델 밖의 pitch 외란: 다리 (두 다리 묶음) 각운동량 L_y (전신 CoM 둘레, 진행 방향 y) 의 변화율을 뒤집은 값 −dL_다리/dt.
  위몸 (몸통 + 팔) 의 실제 식은 dL_위몸/dt = M_바닥 − dL_다리/dt 인데 MPC (SRB) 는 dL/dt = M_바닥 으로 본다 (유도) → −dL_다리/dt 가 위몸에 걸리는 모르는 모멘트.
사용: cop_cause.py vx variant [seconds]
"""
import sys, importlib.util
import numpy as np
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "diag"))
import mujoco
spec = importlib.util.spec_from_file_location("walk", str(_ROOT / "10_walk.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
from configs import resolve
from gait import cfg_get


def run(vx, var, seconds=4.0):
    kw = W.parse_variant(resolve(var)); prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(vx, 0, 0, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
    m, d = model.m, model.d
    l = float(cfg_get(cfg, "mpc.foot_half_length"))
    rows = []
    leg_b = [b for b in range(1, m.nbody) if any(k in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")
                                                  for k in ("hip", "upper_leg", "lower_leg", "foot"))]
    w6 = np.zeros(6); Lprev = None; dL = 0.0
    for _ in range(int(round((walk_t + 3.0 + seconds) / m.opt.timestep))):
        tl, ta = ctl.tick(); model.write_torque(tl, ta); mujoco.mj_step(m, d)
        if W._fell(d):
            print(f"  넘어짐 {d.time:.2f} s"); break
        s = ctl.state
        if s.t < walk_t + 3.0:
            continue
        yaw_ = s.yaw_unwrapped; ey = np.array([-np.sin(yaw_), np.cos(yaw_), 0.0]); cw = d.subtree_com[0]
        Lleg = 0.0
        for b in leg_b:
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, w6, 0)
            Rb = d.ximat[b].reshape(3, 3)
            Lleg += ey @ (Rb @ np.diag(m.body_inertia[b]) @ Rb.T @ w6[:3] + m.body_mass[b] * np.cross(d.xipos[b] - cw, w6[3:]))
        dL = (Lleg - Lprev) / m.opt.timestep if Lprev is not None else 0.0
        Lprev = Lleg
        u = np.asarray(ctl.u_hold, dtype=float); x0 = np.asarray(ctl.x0, dtype=float)
        c = x0[3:6]; yaw = s.yaw_unwrapped
        Rh = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])   # 월드 → 진행 방향
        fz_all = [u[2], u[5]]
        for leg in (0, 1):
            if not ctl._active(leg, s.t) or u[2 + 3 * leg] < 20:
                continue
            r = Rh @ (np.asarray(s.foot_pos_W[leg]) - c)
            F = Rh @ u[3 * leg:3 * leg + 3]; M = Rh @ u[6 + 3 * leg:9 + 3 * leg]
            ph = ctl.gait.phase(leg, s.t, ctl.clock) * ctl.clock.cycle / ctl.gait.stance
            other = fz_all[1 - leg]
            rows.append([ph, r[0], r[2], F[0], F[2], -r[0] * F[2], r[2] * F[0], M[1], -M[1] / F[2] / l,
                         other, x0[1], (Rh @ x0[6:9])[1], -dL])
    R = np.array(rows)
    print(f"[vx {vx} {var}]  MPC 계획 (첫 칸), 발 하나 기준, pitch 모멘트 [N·m] (+ = 앞으로 숙이게)")
    print("   디딤  x[mm] z[mm]  Fx[N]  Fz[N] | 수직력 −x·Fz  앞뒤힘 z·Fx  발모멘트 My | CoP x/l | 다른 발 Fz | pitch[°] pitch속도[°/s] | −dL다리/dt")
    for lo in np.arange(0.0, 1.0, 0.1):
        k = (R[:, 0] >= lo) & (R[:, 0] < lo + 0.1)
        if k.any():
            Z = R[k].mean(0)
            print(f"   {lo:.1f} {1e3*Z[1]:+6.0f} {1e3*Z[2]:5.0f} {Z[3]:+6.1f} {Z[4]:5.0f} |   {Z[5]:+6.2f}      {Z[6]:+6.2f}      {Z[7]:+6.2f}   | {Z[8]:+.2f}  |  {Z[9]:5.0f}   | {np.degrees(Z[10]):+5.2f}   {np.degrees(Z[11]):+6.1f}       | {Z[12]:+6.1f}")
    k = slice(None)
    cc = np.corrcoef(R[:, 12], R[:, 7])[0, 1]
    print(f"   상관 (틱): −dL다리/dt ~ 디딤 발 My {cc:+.2f} · ~ 몸통 pitch 속도 {np.corrcoef(R[:, 12], R[:, 11])[0, 1]:+.2f}")


if __name__ == "__main__":
    run(float(sys.argv[1]), sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 4.0)
