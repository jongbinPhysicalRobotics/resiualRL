"""GRF -> 관절 토크 변환 검증.

정적 평형(qvel=0, qacc=0)에서의 운동방정식:

    M qddot + h(q, qdot) = S^T tau + sum_c J_c^T f_c      (S: 액추에이터 선택 행렬)

qddot = qdot = 0 이면 h 는 중력항 g(q) 만 남고:

    g(q) = S^T tau + sum_c J_c^T f_c

- 베이스 6행 (비구동): g[0:6] = (sum J_c^T f_c)[0:6]
      -> 접촉력이 만족해야 하는 wrench 균형. convex MPC 가 푸는 식이 바로 이것.
- 구동 29행:          tau = g[6:] - (sum J_c^T f_c)[6:]
      -> 접촉력이 이미 만들어주는 토크를 뺀 나머지를 액추에이터가 낸다.

흔히 쓰는 tau = -J^T f 는 위 식에서 g[6:] (다리 링크 자체 중력)를 무시한 근사다.
둘 다 계산해서 차이를 본다.

사용: .venv/Scripts/python.exe MPC/src/baseline/03_grf_to_tau.py
"""
from __future__ import annotations

import numpy as np
import mujoco

import g1_model

np.set_printoptions(precision=3, suppress=True, linewidth=160)

GRAV = 9.81
FEET = ("left_foot", "right_foot")


def site_jacobians(m, d, site_names=FEET):
    """각 발 site 의 (jacp, jacr) 를 world frame 으로 계산. 각각 3 x nv."""
    out = {}
    for name in site_names:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        out[name] = (jacp, jacr)
    return out


def contact_point_jacobians(m, d):
    """발바닥 접촉 구(class="foot", group 3) 각각의 위치 자코비안.

    MuJoCo 가 실제로 접촉을 푸는 지점들이라 MPC 의 contact point 와 대응된다.
    Returns: list of (geom_id, world_pos, jacp(3 x nv), side)
    """
    pts = []
    for g in range(m.ngeom):
        if m.geom_group[g] != 3:
            continue
        bid = m.geom_bodyid[g]
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        if "ankle_roll" not in bname:
            continue
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, None, d.geom_xpos[g], bid)
        side = "left" if bname.startswith("left") else "right"
        pts.append((g, d.geom_xpos[g].copy(), jacp, side))
    return pts


def actuated_dofs(m):
    """액추에이터가 붙은 dof 인덱스 (여기선 6..34)."""
    return np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])


def jname_of_dof(m, dof):
    for j in range(m.njnt):
        if m.jnt_dofadr[j] == dof:
            return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    return "dof%d" % dof


def quasi_static_tau(m, d, adof):
    """현재 q 에서 접촉력을 lstsq 로 배분하고 정적 평형 토크를 계산."""
    mujoco.mj_forward(m, d)
    pts = contact_point_jacobians(m, d)
    if not pts:
        return np.zeros(m.nu)
    A = np.hstack([p[2][:, 0:6].T for p in pts])
    f, *_ = np.linalg.lstsq(A, d.qfrc_bias[0:6], rcond=None)
    tau_full = np.zeros(m.nv)
    for i, p in enumerate(pts):
        tau_full += p[2].T @ f[3 * i:3 * i + 3]
    return d.qfrc_bias[adof] - tau_full[adof]


def main():
    m, d = g1_model.load_torque()
    g1_model.set_crouch(m, d)

    mass = mujoco.mj_getTotalmass(m)
    W = mass * GRAV
    adof = actuated_dofs(m)

    print("=" * 78)
    print("0) 자세 / 기본량")
    print("=" * 78)
    print("  총질량 %.3f kg,  무게 W = %.2f N,  발당 W/2 = %.2f N" % (mass, W, W / 2))
    print("  pelvis z = %.4f m" % d.qpos[2])
    print("  CoM      = %s" % d.subtree_com[0])
    for f in FEET:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f)
        print("  %-11s = %s" % (f, d.site_xpos[sid]))

    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("1) 발 site 자코비안  (v_site = J qdot, world frame)")
    print("=" * 78)
    J = site_jacobians(m, d)
    for f in FEET:
        jacp, _ = J[f]
        nz = np.where(np.abs(jacp).max(axis=0) > 1e-9)[0]
        print("  %s: jacp shape %s, 0 이 아닌 dof = %s" % (f, jacp.shape, nz.tolist()))
    print()
    print("  왼발 jacp 의 다리 관절 열 (수직힘이 곱해질 z 행이 핵심):")
    lp = J["left_foot"][0]
    cols = list(range(6, 12))
    labels = [jname_of_dof(m, c).replace("left_", "").replace("_joint", "") for c in cols]
    print("        " + "".join("%14s" % s for s in labels))
    for i, ax in enumerate("xyz"):
        print("    %s  " % ax + "".join("%14.5f" % lp[i, c] for c in cols))

    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("2) 순진한 배분: 발마다 수직력 W/2  ->  tau = J^T f")
    print("=" * 78)
    f_naive = np.array([0.0, 0.0, W / 2])

    tau_grf_full = np.zeros(m.nv)
    for f in FEET:
        tau_grf_full += J[f][0].T @ f_naive

    g_full = d.qfrc_bias.copy()            # qdot=0 이므로 순수 중력항
    tau_gravcomp = g_full[adof] - tau_grf_full[adof]
    tau_grf_only = -tau_grf_full[adof]     # 흔한 근사: tau = -J^T f

    print("  %-26s%10s%10s%12s%10s" % ("joint", "g(q)", "(J^T f)", "g-J^Tf", "-J^T f"))
    leg_rows = [i for i, dof in enumerate(adof) if dof < 18]
    for i in leg_rows:
        dof = adof[i]
        print("  %-26s%10.3f%10.3f%12.3f%10.3f"
              % (jname_of_dof(m, dof), g_full[dof], tau_grf_full[dof],
                 tau_gravcomp[i], tau_grf_only[i]))
    upper = np.abs(tau_gravcomp[len(leg_rows):]).max()
    print("  ... 상체 관절 tau 최대 |값| = %.3f N.m" % upper)
    print()
    diff = np.abs(tau_gravcomp[leg_rows] - tau_grf_only[leg_rows]).max()
    print("  두 방식 차이 (다리): max |dtau| = %.3f N.m" % diff)
    print("  -> 이 차이가 '다리 링크 자체의 중력'. massless-leg 가정의 오차.")

    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("3) 베이스 6행 잔차: 이 접촉력이 정말 몸을 지탱하나?")
    print("=" * 78)
    res = g_full[0:6] - tau_grf_full[0:6]
    print("  g[0:6]        = %s" % g_full[0:6])
    print("  (J^T f)[0:6]  = %s" % tau_grf_full[0:6])
    print("  잔차          = %s" % res)
    print("  힘 잔차 norm  = %.4f N" % np.linalg.norm(res[0:3]))
    print("  모멘트 잔차   = %.4f N.m   <- 0 이 아니면 넘어진다" % np.linalg.norm(res[3:6]))

    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("4) 접촉점 8개로 힘 재배분 (least squares) — MPC 가 푸는 문제의 축소판")
    print("=" * 78)
    pts = contact_point_jacobians(m, d)
    print("  접촉점 %d 개 (발당 4개)" % len(pts))
    A = np.hstack([p[2][:, 0:6].T for p in pts])       # 6 x (3*npts)
    b = g_full[0:6]
    f_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
    print("  A shape %s, rank %d" % (A.shape, np.linalg.matrix_rank(A)))
    print("  잔차 |A f - b| = %.3e   <- 0 이면 완전 균형" % np.linalg.norm(A @ f_ls - b))
    fz_l = sum(f_ls[3 * i + 2] for i, p in enumerate(pts) if p[3] == "left")
    fz_r = sum(f_ls[3 * i + 2] for i, p in enumerate(pts) if p[3] == "right")
    print("  수직력 합: 왼발 %.2f N, 오른발 %.2f N, 합 %.2f N (W=%.2f)"
          % (fz_l, fz_r, fz_l + fz_r, W))
    fz_min = min(f_ls[3 * i + 2] for i in range(len(pts)))
    print("  접촉점별 최소 수직력 = %.2f N  (음수면 '땅이 당긴다' = 물리적으로 불가)" % fz_min)

    tau_ls_full = np.zeros(m.nv)
    for i, p in enumerate(pts):
        tau_ls_full += p[2].T @ f_ls[3 * i:3 * i + 3]
    tau_ls = g_full[adof] - tau_ls_full[adof]
    print()
    print("  %-26s%14s%14s" % ("joint", "tau(재배분)", "tau(순진)"))
    for i in leg_rows:
        print("  %-26s%14.3f%14.3f" % (jname_of_dof(m, adof[i]), tau_ls[i], tau_gravcomp[i]))

    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("5) 실제로 서 있나? (매 스텝 tau 재계산, 피드백 없음)")
    print("=" * 78)
    modes = (("tau = 0 (무제어)", "zero"),
             ("tau = -J^T f (massless leg)", "grf"),
             ("tau = g - J^T f (full)", "full"))
    for label, mode in modes:
        g1_model.set_crouch(m, d)
        z0 = d.qpos[2]
        fell_at = None
        for k in range(3000):                       # 6 초
            if mode == "zero":
                d.ctrl[:] = 0.0
            elif mode == "grf":
                mujoco.mj_forward(m, d)
                t = np.zeros(m.nv)
                for f in FEET:
                    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f)
                    jp = np.zeros((3, m.nv))
                    mujoco.mj_jacSite(m, d, jp, None, sid)
                    t += jp.T @ np.array([0.0, 0.0, W / 2])
                d.ctrl[:] = -t[adof]
            else:
                d.ctrl[:] = quasi_static_tau(m, d, adof)
            mujoco.mj_step(m, d)
            if fell_at is None and d.qpos[2] < z0 - 0.15:
                fell_at = k * m.opt.timestep
        verdict = ("쓰러짐 @ %.2fs" % fell_at) if fell_at else "버팀 OK"
        print("  %-30s 6초 후 pelvis z = %.4f (시작 %.4f)   %s"
              % (label, d.qpos[2], z0, verdict))

    # 마지막(full) 상태에서 실제 접촉력 확인
    print()
    print("  마지막 상태의 실제 MuJoCo 접촉력:")
    tot = np.zeros(3)
    for c in range(d.ncon):
        frc = np.zeros(6)
        mujoco.mj_contactForce(m, d, c, frc)
        R = d.contact[c].frame.reshape(3, 3)
        tot += R.T @ frc[0:3]
    print("    접촉 %d 개, 합력(world) = %s N   (기대: [0, 0, %.1f])" % (d.ncon, tot, W))


if __name__ == "__main__":
    main()
