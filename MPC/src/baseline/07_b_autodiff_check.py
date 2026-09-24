"""Step 3 — CasADi 자동미분으로 손으로 채운 A_c, B_c 검증 (명세 2절).

연속 동역학 f(x,u)를 CasADi 심볼릭으로 그대로 적고,
    A_sym = ∂f/∂x,   B_sym = ∂f/∂u
를 자동미분으로 뽑아 mpc_srb.continuous_AB 의 손 채움과 비교한다.
f 가 (x 선형·u 선형이라) 자코비안이 상수 → 한 점에서 평가하면 충분.

주의: CasADi 는 한글 경로에서 QP 플러그인 DLL 을 못 읽지만(솔버로는 사용 불가),
심볼릭/자동미분은 코어 기능이라 정상 동작한다.

사용: .venv/Scripts/python.exe MPC/src/baseline/07_b_autodiff_check.py
"""
from __future__ import annotations

import numpy as np
import casadi as ca

import g1_model
import mpc_srb

np.set_printoptions(precision=4, suppress=True, linewidth=160)


def main():
    # 실제 crouch 상태에서 파라미터·발 위치를 가져와 같은 조건으로 비교
    m, d = g1_model.load_torque()
    g1_model.set_crouch(m, d)
    params = mpc_srb.make_params(m, d)
    x0 = mpc_srb.get_state(m, d)
    psi = x0[2]
    r_feet = mpc_srb.get_foot_positions(m, d) - x0[3:6]

    # ---- CasADi 심볼릭 동역학 (명세 2절 식 그대로) -----------------------
    x = ca.SX.sym("x", 13)          # [Θ(3), p(3), ω(3), v(3), g]
    u = ca.SX.sym("u", 12)          # [F_L, m_L, F_R, m_R]

    Rz = ca.SX.zeros(3, 3)
    c, s = ca.cos(psi), ca.sin(psi)
    Rz[0, 0], Rz[0, 1], Rz[1, 0], Rz[1, 1], Rz[2, 2] = c, -s, s, c, 1

    I_w = Rz @ ca.DM(params.I_body) @ Rz.T
    I_w_inv = ca.inv(I_w)

    omega = x[6:9]
    v = x[9:12]
    g_state = x[12]

    torque_sum = ca.SX.zeros(3)
    force_sum = ca.SX.zeros(3)
    for i in range(2):
        F_i = u[6 * i:6 * i + 3]
        m_i = u[6 * i + 3:6 * i + 6]
        r_i = ca.DM(r_feet[i])
        torque_sum += ca.cross(r_i, F_i) + m_i
        force_sum += F_i

    f = ca.vertcat(
        Rz.T @ omega,                                   # Θ̇
        v,                                              # ṗ
        I_w_inv @ torque_sum,                           # ω̇
        force_sum / params.mass - ca.vertcat(0, 0, g_state),  # v̇  (g는 상태!)
        0,                                              # ġ
    )

    A_sym = ca.Function("A", [x, u], [ca.jacobian(f, x)])(x0, np.zeros(12))
    B_sym = ca.Function("B", [x, u], [ca.jacobian(f, u)])(x0, np.zeros(12))
    A_sym = np.array(A_sym)
    B_sym = np.array(B_sym)

    # ---- 손 채움 --------------------------------------------------------
    A_hand, B_hand = mpc_srb.continuous_AB(params, psi, r_feet)

    dA = np.abs(A_sym - A_hand).max()
    dB = np.abs(B_sym - B_hand).max()
    print("=" * 70)
    print("자동미분 vs 손 채움")
    print("=" * 70)
    print(f"  max |A_sym − A_hand| = {dA:.3e}")
    print(f"  max |B_sym − B_hand| = {dB:.3e}")

    if dA > 1e-10 or dB > 1e-10:
        print("\n  불일치 위치 (B):")
        bad = np.argwhere(np.abs(B_sym - B_hand) > 1e-10)
        for r_, c_ in bad[:10]:
            print(f"    B[{r_},{c_}]: sym={B_sym[r_, c_]:.6f}, hand={B_hand[r_, c_]:.6f}")

    # 핵심 블록 눈으로 확인 (왼발)
    print("\n  B 왼발 블록 (ω̇ 행 6:9):")
    print("    힘 3열 (I_w⁻¹[r]×):")
    print(np.array2string(B_hand[6:9, 0:3], prefix="    "))
    print("    모멘트 3열 (I_w⁻¹, wrench 신규):")
    print(np.array2string(B_hand[6:9, 3:6], prefix="    "))

    ok = dA < 1e-10 and dB < 1e-10
    print()
    print("STEP 3 " + ("통과 ✓" if ok else "실패"))


if __name__ == "__main__":
    main()
