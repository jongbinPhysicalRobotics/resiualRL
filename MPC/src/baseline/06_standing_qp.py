"""Step 2 — 1스텝 wrench QP 로 서 있기 (lstsq 교체).

검증 항목 (명세 10절):
  (a) 회귀: 같은 상태에서 QP 해 wrench ≈ lstsq 해를 발별 wrench 로 합친 것
  (b) QP wrench 로 10초 서 있기 성공 (pelvis z 유지)
  (c) 실제 접촉력 합 = [0,0,327] N
  (d) 제약 위반 0

사용: .venv/Scripts/python.exe MPC/src/baseline/06_standing_qp.py
"""
from __future__ import annotations

from importlib import import_module

import numpy as np
import mujoco

import g1_model
import mpc_srb
from mpc_qp import WrenchMPC, wrench_to_tau, actuated_dofs
from mpc_log import MPCLog

m3 = import_module("03_grf_to_tau")

np.set_printoptions(precision=3, suppress=True, linewidth=160)


def lstsq_foot_wrench(m, d):
    """기존 lstsq 해(접촉점 8개 힘)를 발별 site wrench (2,6) 로 합친다 — 회귀 기준."""
    pts = m3.contact_point_jacobians(m, d)
    A = np.hstack([p[2][:, 0:6].T for p in pts])
    f, *_ = np.linalg.lstsq(A, d.qfrc_bias[0:6], rcond=None)
    W = np.zeros((2, 6))
    for k, (side, site) in enumerate((("left", "left_foot"), ("right", "right_foot"))):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
        r_site = d.site_xpos[sid]
        for i, p in enumerate(pts):
            if p[3] != side:
                continue
            fj = f[3 * i:3 * i + 3]
            W[k, :3] += fj
            W[k, 3:] += np.cross(p[1] - r_site, fj)
    return W


def main():
    m, d = g1_model.load_torque()
    g1_model.set_crouch(m, d)
    adof = actuated_dofs(m)
    params = mpc_srb.make_params(m, d)
    Wg = params.mass * mpc_srb.GRAV

    print("=" * 78)
    print("SRB 파라미터 (모델 추출)")
    print("=" * 78)
    print(f"  M = {params.mass:.3f} kg, W = {Wg:.2f} N")
    print(f"  I_body diag = {np.diag(params.I_body)}")
    print(f"  발: l_t={params.l_t}, l_h={params.l_h}, w={params.w}, mu={params.mu}")

    mpc = WrenchMPC(params, horizon=1, dt=0.02)
    print(f"  QP 솔버: {mpc.solver_name} (active-set)")

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("(a) 회귀: N=1 QP vs lstsq (같은 crouch 상태)")
    print("=" * 78)
    x0 = mpc_srb.get_state(m, d)
    x_ref = x0.copy()
    x_ref[6:12] = 0.0                     # ω, v 목표 0 (자세·위치는 현재 유지)
    feet = mpc_srb.get_foot_positions(m, d)
    r_feet = feet - x0[3:6]
    u_ff = mpc.gravity_u_ref()          # 중력 피드포워드 (u_ref=0 이면 R이 중력과 싸움)
    u0, _, info = mpc.solve(x0, x_ref, psi=x0[2], r_feet=r_feet, u_ref=u_ff)
    W_qp = u0.reshape(2, 6)
    W_ls = lstsq_foot_wrench(m, d)

    print(f"  {'':10s}{'Fx':>8s}{'Fy':>8s}{'Fz':>9s}{'mx':>8s}{'my':>8s}{'mz':>8s}")
    for i, name in enumerate(("L(QP)", "L(lstsq)", "R(QP)", "R(lstsq)")):
        w = (W_qp, W_ls)[i % 2][i // 2]
        print("  %-10s" % name + "".join(f"{v:8.2f}" if j != 2 else f"{v:9.2f}"
                                         for j, v in enumerate(w)))
    dmax = np.abs(W_qp - W_ls).max()
    print(f"\n  max |QP − lstsq| = {dmax:.3f}   (힘 N / 모멘트 N·m)")
    print(f"  QP: cost={info['cost']:.4f}, 제약 위반 max = {info['violation']:.2e}, "
          f"{info['solve_ms']:.2f} ms")

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("(b) N=1 QP 컨트롤러로 10초 서 있기 (50 Hz MPC, 500 Hz 토크)")
    print("=" * 78)
    g1_model.set_crouch(m, d)
    log = MPCLog("logs/standing_qp_n1", note="Step2: N=1 wrench QP standing")
    z0 = d.qpos[2]
    dt = m.opt.timestep
    decim = 10                            # 0.002*10 = 0.02 s = 50 Hz
    wr = np.zeros((2, 6))
    x_ref = None
    for k in range(int(10.0 / dt)):
        mujoco.mj_forward(m, d)
        if k % decim == 0:
            x0 = mpc_srb.get_state(m, d)
            if x_ref is None:             # 시작 상태를 참조로 고정
                x_ref = x0.copy()
                x_ref[6:12] = 0.0
            feet = mpc_srb.get_foot_positions(m, d)
            u0, _, info = mpc.solve(x0, x_ref, psi=x0[2], r_feet=feet - x0[3:6],
                                    u_ref=u_ff)
            wr = u0.reshape(2, 6)
            fz_c = sum(_contact_fz(m, d, c) for c in range(d.ncon))
            log.add(t=k * dt, x=x0, x_ref=x_ref, u=u0,
                    tau=wrench_to_tau(m, d, adof, wr),
                    solve_ms=info["solve_ms"], violation=info["violation"],
                    fz_contact=fz_c, ncon=d.ncon)
        d.ctrl[:] = wrench_to_tau(m, d, adof, wr)
        mujoco.mj_step(m, d)

    print(f"  10초 후 pelvis z = {d.qpos[2]:.4f} (시작 {z0:.4f}, drift {d.qpos[2]-z0:+.4f})")
    tot = np.zeros(3)
    for c in range(d.ncon):
        frc = np.zeros(6)
        mujoco.mj_contactForce(m, d, c, frc)
        tot += d.contact[c].frame.reshape(3, 3).T @ frc[0:3]
    print(f"  실제 접촉력 합 = {tot} N (기대 [0,0,{Wg:.1f}])")
    sms = np.stack(log.rows["solve_ms"])
    print(f"  QP 풀이시간: 평균 {sms.mean():.2f} ms, 최대 {sms.max():.2f} ms")
    vio = np.stack(log.rows["violation"]).max()
    print(f"  제약 위반 max (전 구간) = {vio:.2e}")
    log.save()

    stand = abs(d.qpos[2] - z0) < 0.05
    print()
    print("STEP 2 " + ("통과 ✓" if stand and vio < 1e-6 else "실패"))


def _contact_fz(m, d, c) -> float:
    frc = np.zeros(6)
    mujoco.mj_contactForce(m, d, c, frc)
    return float((d.contact[c].frame.reshape(3, 3).T @ frc[0:3])[2])


if __name__ == "__main__":
    main()
