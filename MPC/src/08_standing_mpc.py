"""Step 4 — 시간지평 N=10 condensed wrench MPC 로 서 있기 + 외란.

N≥2 부터 u 가 미래의 p, Θ 에 영향 → 위치·자세 피드백이 생긴다.
(N=1 은 Euler 1스텝이라 속도만 제어 가능 — Step 2 는 사실상 wrench 균형 + 댐핑)

검증 (명세 10절 Step 4):
  - 서 있기 유지 + N=1 (Step 2) 결과와 초기 u0 근사 일치
  - 04_stand.py 의 순수 피드포워드가 넘어졌던 40N/0.1s 밀기에서 회복하는지

사용:
  .venv/Scripts/python.exe MPC/src/08_standing_mpc.py           # 헤드리스 20초+push, 로그 저장
  .venv/Scripts/python.exe MPC/src/08_standing_mpc.py --view    # 뷰어 (Ctrl+우클릭 드래그로 밀어보기)
"""
from __future__ import annotations

import sys

import numpy as np
import mujoco

import g1_model
import mpc_srb
from mpc_qp import WrenchMPC, wrench_to_tau, actuated_dofs
from mpc_log import MPCLog

np.set_printoptions(precision=3, suppress=True, linewidth=160)

HORIZON = 10
DT_MPC = 0.02          # 지평 내부 스텝 (0.2 s preview)
DECIM = 5              # MPC 재계산 주기: 0.002*5 = 100 Hz (50Hz는 진동 발산 — Step4 디버깅)


def make_controller(m, d):
    params = mpc_srb.make_params(m, d)
    mpc = WrenchMPC(params, horizon=HORIZON, dt=DT_MPC)
    u_ff = mpc.gravity_u_ref()
    x_ref = mpc_srb.get_state(m, d)
    x_ref[6:12] = 0.0                   # ω, v 참조 0. Θ, p 는 시작 상태 유지
    return mpc, u_ff, x_ref


def headless(push_N=40.0):
    m, d = g1_model.load_torque()
    g1_model.set_crouch(m, d)
    adof = actuated_dofs(m)
    mpc, u_ff, x_ref = make_controller(m, d)
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    # --- N=10 vs N=1 초기해 비교 (수렴 확인) ---
    mujoco.mj_forward(m, d)
    x0 = mpc_srb.get_state(m, d, I_body=mpc.p.I_body)
    r_feet = mpc_srb.get_foot_positions(m, d) - x0[3:6]
    u10, _, _ = mpc.solve(x0, x_ref, psi=x0[2], r_feet=r_feet, u_ref=u_ff)
    mpc1 = WrenchMPC(mpc.p, horizon=1, dt=DT_MPC)
    u1, _, _ = mpc1.solve(x0, x_ref, psi=x0[2], r_feet=r_feet, u_ref=u_ff)
    print("초기 u0 비교 (N=10 vs N=1):")
    print("  N=10:", u10.reshape(2, 6))
    print("  N=1 :", u1.reshape(2, 6))
    print(f"  max |차이| = {np.abs(u10 - u1).max():.3f}\n")

    # --- 20초 시뮬 + 40N 밀기 2회 (04 피드포워드가 넘어졌던 외란) ---
    log = MPCLog("logs/standing_mpc_n10",
                 note=f"Step4: N={HORIZON} condensed MPC, push 40N@8s,14s")
    z0 = d.qpos[2]
    dt = m.opt.timestep
    wr = np.zeros((2, 6))
    push_events = ((8.0, push_N), (14.0, push_N))
    push_dur = 0.1
    fell = None

    print("  t[s]  pelvis_z   com_x    pitch[deg]  |u_my|   solve_ms  event")
    next_rep = 0.0
    for k in range(int(20.0 / dt)):
        t = k * dt
        mujoco.mj_forward(m, d)
        if k % DECIM == 0:
            x0 = mpc_srb.get_state(m, d, I_body=mpc.p.I_body)
            feet = mpc_srb.get_foot_positions(m, d)
            u0, _, info = mpc.solve(x0, x_ref, psi=x0[2],
                                    r_feet=feet - x0[3:6], u_ref=u_ff)
            wr = u0.reshape(2, 6)
            log.add(t=t, x=x0, x_ref=x_ref, u=u0,
                    tau=wrench_to_tau(m, d, adof, wr),
                    solve_ms=info["solve_ms"], violation=info["violation"],
                    fz_contact=_contact_sum(m, d)[2], ncon=d.ncon)

        d.ctrl[:] = wrench_to_tau(m, d, adof, wr)

        d.xfrc_applied[:] = 0.0
        event = ""
        for pt, pN in push_events:
            if pt <= t < pt + push_dur:
                d.xfrc_applied[pelvis, 0] = pN
                event = f"<- push {pN:.0f}N +x"
        mujoco.mj_step(m, d)

        if t >= next_rep - 1e-9 or event:
            if t >= next_rep - 1e-9:
                next_rep += 2.0
            if event and t * 1000 % 100 > dt * 1000:    # push 중 한 번만 출력
                event_show = event
            print(f"  {t:5.1f}  {d.qpos[2]:8.4f}  {d.subtree_com[0][0]:7.4f}"
                  f"  {np.degrees(mpc_srb.quat_to_euler_zyx(d.qpos[3:7])[1]):9.3f}"
                  f"  {np.abs(wr[:, 4]).max():7.2f}  {info['solve_ms']:7.2f}   {event}")
        if fell is None and d.qpos[2] < z0 - 0.2:
            fell = t
            break

    print()
    if fell:
        print(f"  쓰러짐 @ {fell:.2f}s")
    else:
        tot = _contact_sum(m, d)
        print(f"  20초 생존 ✓  pelvis z {z0:.4f} -> {d.qpos[2]:.4f} "
              f"(drift {d.qpos[2]-z0:+.4f})")
        print(f"  실제 접촉력 합 = {tot} N")
    sms = np.stack(log.rows["solve_ms"])
    vio = np.stack(log.rows["violation"]).max()
    print(f"  QP(N={HORIZON}): 평균 {sms.mean():.2f} ms, 최대 {sms.max():.2f} ms, "
          f"제약 위반 max = {vio:.2e}")
    log.save()
    print()
    print("STEP 4 " + ("통과 ✓" if not fell and vio < 1e-6 else "실패"))


def _contact_sum(m, d):
    tot = np.zeros(3)
    for c in range(d.ncon):
        frc = np.zeros(6)
        mujoco.mj_contactForce(m, d, c, frc)
        tot += d.contact[c].frame.reshape(3, 3).T @ frc[0:3]
    return tot


def view():
    import mujoco.viewer

    m, d = g1_model.load_torque()
    g1_model.set_crouch(m, d)
    adof = actuated_dofs(m)
    mpc, u_ff, x_ref = make_controller(m, d)

    print(f"뷰어: N={HORIZON} wrench MPC. Ctrl+우클릭 드래그로 밀어보세요.")
    wr = np.zeros((2, 6))
    k = 0
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            mujoco.mj_forward(m, d)
            if k % DECIM == 0:
                x0 = mpc_srb.get_state(m, d)
                feet = mpc_srb.get_foot_positions(m, d)
                u0, _, _ = mpc.solve(x0, x_ref, psi=x0[2],
                                     r_feet=feet - x0[3:6], u_ref=u_ff)
                wr = u0.reshape(2, 6)
            d.ctrl[:] = wrench_to_tau(m, d, adof, wr)
            mujoco.mj_step(m, d)
            v.sync()
            k += 1


if __name__ == "__main__":
    if "--view" in sys.argv:
        view()
    else:
        pn = 40.0
        if "--push" in sys.argv:
            pn = float(sys.argv[sys.argv.index("--push") + 1])
        headless(push_N=pn)
