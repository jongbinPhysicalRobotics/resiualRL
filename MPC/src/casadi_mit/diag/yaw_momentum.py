"""몸통 yaw 흔들림의 출처 (Q&A 10/6 Q73). 걷기 + 3 s 부터 seconds 동안:
  전신 CoM 둘레 수직축 각운동량 L_z 를 몸통 · 왼다리 · 오른다리 · 팔 몫으로 (L = Σ I_w ω + m (x − c) × v, 바디 CoM 기준)
  바닥이 준 수직축 모멘트 (CoM 둘레, Σ (p − c) × f 의 z) — 전신 L_z 의 변화율과 같아야 한다 (검산)
  팔 몫을 "몸통과 함께 강체로 돈 몫" (팔 관절 속도 0 으로 둔 L) 과 "팔 관절 (흔들기) 몫" 으로 나눔
  바닥 모멘트를 "발 합력 × 발 위치" ((발 site − c) × F_발) 와 "발 비틀림" (나머지 = 발 안 두 점 짝힘) 으로 나눔
  흔들림 크기 = 각 몫에서 0.55 s (한 주기) 이동평균을 뺀 값의 표준편차, 몸통 ω_z 표준편차, 몫끼리 상관
사용: yaw_momentum.py vx variant [seconds]
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


def groups(m, base):
    """바디 → 0 몸통 · 1 왼다리 · 2 오른다리 · 3 팔 (몸통 아래 subtree 로 나눔)."""
    g = np.full(m.nbody, -1)
    for b in range(1, m.nbody):
        root = b
        while m.body_parentid[root] != base and root != base:
            root = m.body_parentid[root]
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, root) or ""
        if b == base:
            g[b] = 0
        elif "left_hip" in nm:
            g[b] = 1
        elif "right_hip" in nm:
            g[b] = 2
        elif "shoulder" in nm:
            g[b] = 3
    return g


def run(vx, var, seconds=5.0):
    kw = W.parse_variant(resolve(var)); prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(vx, 0, 0, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
    m, d = model.m, model.d
    G = groups(m, model.base_body)
    w6 = np.zeros(6); f6 = np.zeros(6); rows = []
    d2 = mujoco.MjData(m)
    arm_bodies = [b for b in range(m.nbody) if G[b] == 3]
    arm_dof = np.asarray(model.arm_dof).ravel()
    for _ in range(int(round((walk_t + 3.0 + seconds) / m.opt.timestep))):
        tl, ta = ctl.tick(); model.write_torque(tl, ta); mujoco.mj_step(m, d)
        if W._fell(d):
            print(f"  넘어짐 {d.time:.2f} s"); break
        if d.time < walk_t + 3.0:
            continue
        c = d.subtree_com[0].copy()
        L = np.zeros(4)
        for b in range(1, m.nbody):
            if G[b] < 0:
                continue
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, w6, 0)
            Rb = d.ximat[b].reshape(3, 3)
            Iw = Rb @ np.diag(m.body_inertia[b]) @ Rb.T
            L[G[b]] += (Iw @ w6[:3])[2] + m.body_mass[b] * np.cross(d.xipos[b] - c, w6[3:])[2]
        # 팔이 몸통과 강체로 돈 몫: 팔 관절 속도를 0 으로 둔 복사본에서 팔 L_z
        d2.qpos[:] = d.qpos; d2.qvel[:] = d.qvel; d2.qvel[arm_dof] = 0.0
        mujoco.mj_kinematics(m, d2); mujoco.mj_comPos(m, d2); mujoco.mj_comVel(m, d2)
        La_rigid = 0.0
        for b in arm_bodies:
            mujoco.mj_objectVelocity(m, d2, mujoco.mjtObj.mjOBJ_BODY, b, w6, 0)
            Rb = d2.ximat[b].reshape(3, 3)
            La_rigid += (Rb @ np.diag(m.body_inertia[b]) @ Rb.T @ w6[:3])[2] + m.body_mass[b] * np.cross(d2.xipos[b] - c, w6[3:])[2]
        Mg = Mlev = 0.0
        Ff = np.zeros((2, 3))
        for i in range(d.ncon):
            con = d.contact[i]
            mujoco.mj_contactForce(m, d, i, f6)
            f = con.frame.reshape(3, 3).T @ f6[:3]                 # geom1 → geom2 에 준 힘
            sgn = 1.0 if m.geom_bodyid[con.geom2] != 0 else -1.0  # 로봇이 받은 힘 (바닥 body 0)
            Mg += sgn * np.cross(con.pos - c, f)[2]
            lg = model._geom_foot[con.geom1] if model._geom_foot[con.geom1] >= 0 else model._geom_foot[con.geom2]
            if lg >= 0:
                Ff[lg] += sgn * f
        for lg in (0, 1):
            Mlev += np.cross(d.site_xpos[model.foot_site[lg]] - c, Ff[lg])[2]
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, model.base_body, w6, 0)
        rows.append([d.time, *L, Mg, w6[2], La_rigid, Mlev])
    R = np.array(rows); t = R[:, 0]; dt = float(np.median(np.diff(t)))
    n = max(1, int(round(0.55 / dt)))
    hp = lambda x: x - np.convolve(x, np.ones(n) / n, mode="same")          # 한 주기 이동평균 제거
    k = slice(n, len(t) - n)
    Lt, Ll, Lr, La, Mg, wz = (R[:, i] for i in range(1, 7))
    Ltot = Lt + Ll + Lr + La
    dL = np.gradient(Ltot, dt)
    sd = lambda x: float(np.std(hp(x)[k]))
    cc = lambda a, b: float(np.corrcoef(hp(a)[k], hp(b)[k])[0, 1])
    print(f"[vx {vx} {var}] 각운동량 L_z 흔들림 표준편차 [kg·m²/s]: 몸통 {sd(Lt):.3f} · 왼다리 {sd(Ll):.3f} · 오른다리 {sd(Lr):.3f} · 다리 합 {sd(Ll+Lr):.3f}"
          f" · 팔 {sd(La):.3f} · 전신 {sd(Ltot):.3f}")
    print(f"   몸통 ω_z 표준편차 {np.degrees(sd(wz)):.1f} °/s  | 상관: 몸통~다리합 {cc(Lt, Ll+Lr):+.2f} · 몸통~팔 {cc(Lt, La):+.2f} · 다리합~팔 {cc(Ll+Lr, La):+.2f}"
          f" · 몸통~전신 {cc(Lt, Ltot):+.2f}")
    Lar, Mlev = R[:, 7], R[:, 8]
    print(f"   팔 몫 = 몸통과 강체로 돈 몫 {sd(Lar):.3f} + 팔 흔들기 몫 {sd(La - Lar):.3f} (팔 흔들기 ~ 다리합 상관 {cc(La - Lar, Ll + Lr):+.2f})"
          f" · 위몸 (몸통 + 팔) {sd(Lt + La):.3f}")
    print(f"   바닥 수직축 모멘트 (CoM 둘레) 표준편차 {sd(Mg):.2f} N·m = 발 합력 × 발 위치 {sd(Mlev):.2f} + 발 비틀림 {sd(Mg - Mlev):.2f}"
          f" · 검산 dL/dt 와 상관 {cc(Mg, dL):+.2f}")


if __name__ == "__main__":
    run(float(sys.argv[1]), sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 5.0)
