"""CoP 제약 h 결합항 검증 (검증 리포트 §1 의 요구사항).

방법: 발바닥 접촉점 4개에 무작위 점힘(마찰콘 안, 수평력 포함!)을 뿌리고
  (a) 점힘에서 직접 계산한 지면 CoP 가 발 안인가   <- 물리적 정답
  (b) 점힘을 발목 site wrench 로 합쳐 제약 행 7-10 을 통과하는가
둘이 모든 샘플에서 일치해야 한다. Fx,Fy ≠ 0 이므로 h 항이 실제로 시험된다
(서 있기 Step 1 은 Fx≈0 이라 이 항을 검증할 수 없었다).

추가로 h=0 인 옛 제약이 얼마나 오판하는지 센다.
yaw=0.7 rad 회전 케이스(world frame 제약)도 검증.

사용: .venv/Scripts/python.exe MPC/src/10_constraint_check.py
"""
from __future__ import annotations

import numpy as np

import g1_model
import mpc_srb
from mpc_qp import foot_constraints

rng = np.random.default_rng(0)


def sample_case(p, psi=0.0):
    """접촉점 4개 무작위 점힘 (발 frame) -> (world wrench W(6), 물리 정답 bool)."""
    pts = np.array([[p.l_t, p.w, -p.h_sole], [p.l_t, -p.w, -p.h_sole],
                    [-p.l_h, p.w, -p.h_sole], [-p.l_h, -p.w, -p.h_sole]])
    F = np.zeros(3)
    m_site = np.zeros(3)
    # 음수 fz·마찰 초과도 허용해 '물리적으로 불가능한' wrench 도 생성한다
    # (제약이 reject 하는 방향도 시험해야 완전한 검증)
    fz_pts = rng.uniform(-40.0, 120.0, 4)
    for j in range(4):
        s_t = rng.uniform(-1.4, 1.4, 2)
        f = np.array([s_t[0] * p.mu * abs(fz_pts[j]),
                      s_t[1] * p.mu * abs(fz_pts[j]),
                      fz_pts[j]])
        F += f
        m_site += np.cross(pts[j], f)

    Fz = F[2]
    if Fz < p.fz_min or Fz > p.fz_max:
        return None
    # 물리 정답: 지면 CoP (점힘의 fz 가중 평균) + 합력 마찰
    cop_x = (pts[:, 0] * fz_pts).sum() / Fz
    cop_y = (pts[:, 1] * fz_pts).sum() / Fz
    ok_phys = (-p.l_h - 1e-9 <= cop_x <= p.l_t + 1e-9
               and abs(cop_y) <= p.w + 1e-9
               and abs(F[0]) <= p.mu * Fz + 1e-9
               and abs(F[1]) <= p.mu * Fz + 1e-9)

    W = np.concatenate([F, m_site])
    if abs(psi) > 1e-12:
        R = mpc_srb.rz(psi)
        W = np.concatenate([R @ W[:3], R @ W[3:]])   # world 로 회전
    return W, ok_phys


def run(p, psi, n=20000):
    C, dvec = foot_constraints(p, psi=psi)
    C_old, _ = foot_constraints(
        mpc_srb.SRBParams(**{**p.__dict__, "h_sole": 0.0}), psi=psi)
    C, dvec, C_old = C[:10], dvec[:10], C_old[:10]   # mz 4행 제외 (선모델은 사각보다 보수적 — 별도 검증)
    agree = agree_old = n_valid = n_in = 0
    for _ in range(n):
        case = sample_case(p, psi)
        if case is None:
            continue
        W, ok_phys = case
        n_valid += 1
        n_in += ok_phys
        ok_new = bool((C @ W <= dvec + 1e-7).all())
        ok_old = bool((C_old @ W <= dvec + 1e-7).all())
        agree += (ok_new == ok_phys)
        agree_old += (ok_old == ok_phys)
    return n_valid, n_in, agree, agree_old


def run_lp_full(p, psi, n=2500):
    """전체 18행 vs LP 피지빌리티 (4점 사각 접촉, 점별 사각뿔 마찰) — mz 포함 정답."""
    from scipy.optimize import linprog
    mu, h = p.mu, p.h_sole
    X, Y, xc = (p.l_t + p.l_h) / 2, p.w, (p.l_t - p.l_h) / 2
    corners = np.array([[X, Y], [X, -Y], [-X, Y], [-X, -Y]])
    C, dvec = foot_constraints(p, psi=psi)
    R = mpc_srb.rz(psi)

    def lp_feasible(fw_c):
        Aeq = np.zeros((6, 12)); beq = fw_c
        for i, (cx, cy) in enumerate(corners):
            j = 3 * i
            Aeq[0, j] = Aeq[1, j+1] = Aeq[2, j+2] = 1
            Aeq[3, j+2] = cy
            Aeq[4, j+2] = -cx
            Aeq[5, j] = -cy; Aeq[5, j+1] = cx
        Aub, bub = [], []
        for i in range(4):
            j = 3 * i
            for k, sgn in ((0, 1), (0, -1), (1, 1), (1, -1)):
                row = np.zeros(12); row[j+k] = sgn; row[j+2] = -mu
                Aub.append(row); bub.append(0)
            row = np.zeros(12); row[j+2] = -1; Aub.append(row); bub.append(0)
        r = linprog(np.zeros(12), A_ub=np.array(Aub), b_ub=np.array(bub),
                    A_eq=Aeq, b_eq=beq, bounds=[(None, None)] * 12, method="highs")
        return r.status == 0

    agree = feas = 0
    for _ in range(n):
        fz = rng.uniform(p.fz_min + 1, 200)
        fw_c = np.array([rng.uniform(-1.2, 1.2) * mu * fz,
                         rng.uniform(-1.2, 1.2) * mu * fz, fz,
                         rng.uniform(-1.3, 1.3) * Y * fz,
                         rng.uniform(-1.3, 1.3) * X * fz,
                         rng.uniform(-1.5, 1.5) * mu * (X + Y) * fz])
        ok_lp = lp_feasible(fw_c)
        # 중심 wrench -> 발목 wrench -> world 회전
        fx, fy, fzv, tx, ty, tz = fw_c
        W_ankle = np.array([fx, fy, fzv,
                            tx + h * fy,
                            ty - h * fx - xc * fzv,
                            tz + xc * fy])
        W = np.concatenate([R @ W_ankle[:3], R @ W_ankle[3:]])
        ok_c = bool((C @ W <= dvec + 1e-7).all())
        agree += (ok_c == ok_lp); feas += ok_lp
    return n, feas, agree


def main():
    m, d = g1_model.load_torque()
    g1_model.set_crouch(m, d)
    p = mpc_srb.make_params(m, d)
    print(f"h_sole = {p.h_sole:.4f} m (모델 추출: 구 중심 0.03 + 반지름 0.005)")
    print(f"l_t={p.l_t}, l_h={p.l_h}, w={p.w}, mu={p.mu}")
    print()

    print("--- 전체 18행 (mz 8행 포함) vs LP 피지빌리티 ---")
    ok_all = True
    for psi in (0.0, 0.7):
        nv, ni, ag = run_lp_full(p, psi)
        print(f"yaw={psi:.1f}: 샘플 {nv} (feasible {ni})  일치율 {ag}/{nv} ({100*ag/nv:.2f} %)")
        ok_all &= (ag == nv)
    print()
    print("--- 사각 4점 모델 (h 결합항, 행 1-10) ---")
    for psi in (0.0, 0.7):
        nv, ni, ag, ag_old = run(p, psi)
        print(f"yaw = {psi:.1f} rad — 유효 샘플 {nv} (물리적 통과 {ni})")
        print(f"  새 제약(h 항 포함)  일치율: {ag}/{nv}  ({100*ag/nv:.2f} %)")
        print(f"  옛 제약(h=0)      일치율: {ag_old}/{nv}  ({100*ag_old/nv:.2f} %)"
              f"   <- 오판 {nv-ag_old}건")
        ok_all &= (ag == nv)
    print()
    print("검증 " + ("통과 ✓ — 새 제약은 물리 정답과 완전 일치" if ok_all else "실패"))


if __name__ == "__main__":
    main()
