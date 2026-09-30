"""CasADi 형식 condensed MPC — 이식 (mpc.py) 과 같은 QP 를 기호식 + 자동미분으로 만든다 (Q&A 9/26 Q1 계획 1·2 단계).

왜: MIT 재작업은 "모델을 기호식으로 한 번 적고 미분·구조를 도구가 만드는" CasADi 형식으로 가기로 함
(나중에 Residual MPC 로 넘어갈 때 그대로 이어짐). 첫 단계로 reference 의 condensed QP 를 CasADi 로
다시 만들고 기존 이식과 수치로 대조한다 (검증 스크립트: 15_casadi_check.py).

무엇이 기호식인가 (mpc.py 의 손 채움을 대체):
  1. 연속 동역학 f(x, u; ψ_k, r_k, m, I_B) 를 그대로 적고  A_c = ∂f/∂x, B_c = ∂f/∂u 를 **자동미분**으로.
     (mpc.continuous_matrices 의 손 채움 — I_k⁻¹[r]×, Rᵀ 블록 — 이 유도와 맞는지가 검증 포인트)
  2. ZOH 절단 이산화 (reference 식 그대로: A_d = I + dtA + ½dt²A², B_d = dtB + ½dt²AB + dt³/6 A²B)
  3. 비용 회전 Q_k = T_kᵀ Q T_k (bodyYawStateCost) 와 발 yaw 제약 회전 C_W = C_F·blkdiag(Rᵀ,Rᵀ)
숫자 조립 (구조적 — Q&A 9/25 Q11 에서 C++ 과 0 ~ 9e-17 로 이미 검증된 부분은 재사용):
  lift (condensing), H/q 곱, 고정 sparsity 패턴, OSQP 층 (shifted warm start·cold reset·fallback) 은
  ConvexMPC 를 상속해서 그대로 쓴다. 즉 이 클래스는 "공식을 만드는 부분"만 바꾼다.

foot_model 플래그 (기본 "ref" = 기존 동작):
  "ref"  : reference 그대로 — CoP 폭 b=0.01 허용, |Mz| ≤ 0.0657µFz, roll 등식 없음.
  "mx0"  : reference 제약 그대로 + mx^F = 0 등식만 (reference no_roll_moment 와 같음 — Q&A 9/27 Q1: mx 효과만 분리)
  "line" : OA-MPC 선 접촉 (Ding et al. Humanoids 2022 §III-F, Q&A 9/26 논의). 발이 선(발끝~뒤꿈치)으로
           닿는 우리 MIT 모델에 물리적으로 맞는 제약:
             mx = 0 (발 좌표계, D 행 등식 — reference no_roll_moment 자리)
             CoP: −lt·Fz ≤ my ≤ lh·Fz  (lt = lh = foot_half_length)
             비틀림 (발끝·뒤꿈치 두 점 마찰에서 유도, 식 (12)):
               |mz + lh·Fy| ≤ µ(lh·Fz − my),  |lt·Fy − mz| ≤ µ(lt·Fz + my)
           → reference 의 CoP 폭 b 행과 0.0657 비틀림 행을 대체. Fx 마찰·Fz 범위는 동일.

사용:
    from mpc_casadi import CasadiCondensedMPC
    mpc = CasadiCondensedMPC(cfg)                      # 이식 ConvexMPC 와 같은 인터페이스
    mpc = CasadiCondensedMPC(cfg, foot_model="line")   # OA-MPC 선 접촉
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import casadi as ca

from mpc import (ConvexMPC, NX, NU, N_INEQ, N_EQ, LEFT_MAP, RIGHT_MAP,
                 lift, resolved_foot_yaw, yaw_state_cost, _cfg)


# 기호식 빌더
def _rz(psi):
    c, s = ca.cos(psi), ca.sin(psi)
    return ca.vertcat(ca.horzcat(c, -s, 0), ca.horzcat(s, c, 0), ca.horzcat(0, 0, 1))


def build_step_function(dt: float) -> ca.Function:
    """(psi, r(6), m, I_B(9)) → (A_d 13×13, B_d 13×12).  동역학을 적고 A_c/B_c 는 자동미분."""
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    psi = ca.SX.sym("psi")
    r = ca.SX.sym("r", 6)                       # [r_L; r_R] (참조 몸 위치 → 디딤점, 월드)
    m = ca.SX.sym("m")
    I9 = ca.SX.sym("I9", 9)                     # reduced.inertia_B (yaw frame), 행 우선
    I_B = ca.reshape(I9, 3, 3).T                # numpy .reshape(-1) 이 행 우선이라 전치로 복원
    R = _rz(psi)
    Ik_inv = R @ ca.solve(I_B, ca.SX.eye(3)) @ R.T
    # u 순서 (LEFT_MAP/RIGHT_MAP): [F_L(0:3), F_R(3:6), M_L(6:9), M_R(9:12)]
    FL, FR, ML, MR = u[0:3], u[3:6], u[6:9], u[9:12]
    f = ca.vertcat(
        R.T @ x[6:9],                                                   # Θ̇ = Rz(ψ_ref)ᵀ ω
        x[9:12],                                                        # ṗ = v
        Ik_inv @ (ca.cross(r[0:3], FL) + ca.cross(r[3:6], FR) + ML + MR),  # ω̇ = I_k⁻¹ Σ(r×F + M)
        (FL + FR) / m + ca.vertcat(0, 0, x[12]),                        # v̇ = ΣF/m + g·ẑ (g = 13번째 상태)
        ca.SX(0),                                                       # ġ = 0
    )
    A_c = ca.jacobian(f, x)                     # ← 자동미분 (손 채움 대체)
    B_c = ca.jacobian(f, u)
    A2 = A_c @ A_c
    A_d = ca.SX.eye(NX) + dt * A_c + (0.5 * dt * dt) * A2
    B_d = dt * B_c + (0.5 * dt * dt) * (A_c @ B_c) + (dt ** 3 / 6.0) * (A2 @ B_c)
    return ca.Function("mpc_step", [psi, r, m, I9], [ca.densify(A_d), ca.densify(B_d)])


def build_qk_function() -> ca.Function:
    """(psi, Q_diag(13)) → Q_k = T_kᵀ diag(Q) T_k  (bodyYawStateCost)."""
    psi = ca.SX.sym("psi")
    Qd = ca.SX.sym("Qd", NX)
    RT = _rz(psi).T
    T = ca.SX.eye(NX)
    T[3:6, 3:6] = RT
    T[6:9, 6:9] = RT
    T[9:12, 9:12] = RT
    return ca.Function("qk", [psi, Qd], [ca.densify(T.T @ ca.diag(Qd) @ T)])


def build_rot_function(template: np.ndarray) -> ca.Function:
    """(yaw) → C_W = C_F · blkdiag(Rᵀ, Rᵀ)  (yawRotatedFootConstraintValue)."""
    yaw = ca.SX.sym("yaw")
    RT = _rz(yaw).T
    T = ca.DM(template)
    CW = ca.horzcat(T[:, 0:3] @ RT, T[:, 3:6] @ RT)
    return ca.Function("rot", [yaw], [ca.densify(CW)])


def line_foot_template(cfg) -> np.ndarray:
    """OA-MPC 선 접촉 12×6 (발 좌표계 [Fx,Fy,Fz,Mx,My,Mz], C·w ≤ ub).  lt = lh = foot_half_length."""
    mu = float(_cfg(cfg, "mpc.friction_coefficient"))
    a = float(_cfg(cfg, "mpc.foot_half_length"))
    lt = lh = a
    # 변형 cmar=ε (Q&A 9/30 Q12, 기본 0): 두 점이 각각 Fz 의 ε 이상을 받치게 (평발 유지) → CoP 행만 (1 − 2ε)·l.
    # fz_뒤 = (my + l·Fz)/2l ≥ ε·Fz ⟺ −my ≤ (1 − 2ε)·l·Fz. 비틀림 행의 지렛대 l 은 그대로 (실제 두 접촉점).
    c = 1.0 - 2.0 * float(_cfg(cfg, "mpc.cop_margin_frac", 0.0) or 0.0)
    return np.array([
        [1.0, 0.0, -mu, 0.0, 0.0, 0.0],             # 0   Fx − µFz ≤ 0
        [-1.0, 0.0, -mu, 0.0, 0.0, 0.0],            # 1  −Fx − µFz ≤ 0
        [0.0, 1.0, -mu, 0.0, 0.0, 0.0],             # 2   Fy − µFz ≤ 0   (합력 — 점별은 8~11이 정확)
        [0.0, -1.0, -mu, 0.0, 0.0, 0.0],            # 3  −Fy − µFz ≤ 0
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],             # 4   Fz ≤ Fmax
        [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],            # 5  −Fz ≤ −minScale·Fmin
        [0.0, 0.0, -c * lh, 0.0, 1.0, 0.0],         # 6   my ≤ c·lh·Fz   (CoP 뒤꿈치 쪽, c = 1 − 2ε)
        [0.0, 0.0, -c * lt, 0.0, -1.0, 0.0],        # 7  −my ≤ c·lt·Fz  (CoP 발끝 쪽)
        [0.0, lh, -mu * lh, 0.0, mu, 1.0],          # 8   mz + lh·Fy ≤ µ(lh·Fz − my)   (발끝 점 y 마찰)
        [0.0, -lh, -mu * lh, 0.0, mu, -1.0],        # 9  −(mz + lh·Fy) ≤ µ(lh·Fz − my)
        [0.0, lt, -mu * lt, 0.0, -mu, -1.0],        # 10  lt·Fy − mz ≤ µ(lt·Fz + my)   (뒤꿈치 점 y 마찰)
        [0.0, -lt, -mu * lt, 0.0, -mu, 1.0],        # 11 −(lt·Fy − mz) ≤ µ(lt·Fz + my)
    ])


_REF_UNIT_COLS = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [2], [2],
                  [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 5], [2, 5]]
# 월드 좌표 열 패턴 (발 yaw 로 회전한 뒤 기준). ⚠ 9/28 Q10 수정: 전에는 발 좌표 열 ([2,4], [1,2,4,5]) 만 적어
# yaw ≠ 0 에서 Fx · Mx 성분 (회전으로 섞여 들어오는 값) 이 OSQP 행렬에서 빠졌다 — quadprog (dense) 만 맞았다.
_LINE_UNIT_COLS = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [2], [2],
                   [2, 3, 4], [2, 3, 4], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]


def row_level_pattern(steps: int, unit_cols):
    """mpc.row_level_constraint_pattern 의 일반화 (unit_cols 를 인자로). D 열 예약은 동일 (행 6/9 에 3열)."""
    ineq_rows = N_INEQ * steps
    trip = []
    for k in range(steps):
        u0 = NU * k
        c0 = N_INEQ * k
        d0 = ineq_rows + N_EQ * k
        for r in range(12):
            for lc in unit_cols[r]:
                trip.append((c0 + r, u0 + LEFT_MAP[lc]))
        for r in range(12):
            for lc in unit_cols[r]:
                trip.append((c0 + 12 + r, u0 + RIGHT_MAP[lc]))
        for i in range(6):
            trip.append((d0 + i, u0 + i))
        trip += [(d0 + 6, u0 + 6), (d0 + 6, u0 + 7), (d0 + 6, u0 + 8), (d0 + 7, u0 + 7), (d0 + 8, u0 + 8)]
        trip += [(d0 + 9, u0 + 9), (d0 + 9, u0 + 10), (d0 + 9, u0 + 11), (d0 + 10, u0 + 10), (d0 + 11, u0 + 11)]
    trip = np.array(trip, dtype=np.int64)
    order = np.lexsort((trip[:, 0], trip[:, 1]))
    trip = trip[order]
    rows, cols = trip[:, 0], trip[:, 1]
    nvars = NU * steps
    counts = np.bincount(cols, minlength=nvars)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    A = sp.csc_matrix((np.zeros(len(rows)), rows, indptr), shape=((N_INEQ + N_EQ) * steps, nvars))
    return A, rows, cols


# ---------------------------------------------------------------------------
class CasadiCondensedMPC(ConvexMPC):
    """ConvexMPC 와 같은 인터페이스. 공식 (A_c/B_c 자동미분·이산화·Q_k·제약 회전) 만 CasADi 로 바꾼다."""

    def __init__(self, cfg, solver: str = "osqp", osqp_settings: dict | None = None,
                 verbose_errors: bool = True, foot_model: str = "ref"):
        super().__init__(cfg, solver=solver, osqp_settings=osqp_settings, verbose_errors=verbose_errors)
        if foot_model not in ("ref", "line", "mx0"):
            raise ValueError(f"foot_model must be 'ref', 'line' or 'mx0', got {foot_model!r}")
        self.foot_model = foot_model
        # 변형 (Q&A 9/28 Q12): 발 두 점 (뒤꿈치 · 앞꿈치) 각각의 최소 수직력 [N]. 0 = 기존 (CoP 한계만).
        # 두 점 분해 Fz_heel = (my + l_t·Fz)/(l_h + l_t), Fz_toe = (l_h·Fz − my)/(l_h + l_t) (OA-MPC) 에서
        # CoP 두 행의 우변을 0 → −(l_h + l_t)·f_min 로. ref 발은 l_h = l_t = foot_half_length (행 8, 9), line 은 행 6, 7.
        self.point_fmin = float(_cfg(cfg, "mpc.point_force_min", 0.0) or 0.0)
        self._cop_rows = (6, 7) if foot_model == "line" else (8, 9)
        self._cop_span = 2.0 * float(_cfg(cfg, "mpc.foot_half_length"))
        self._step_map = build_step_function(self.dt).map(self.N)
        self._qk_map = build_qk_function().map(self.N)
        if foot_model == "line":
            self.template = line_foot_template(cfg)
            self.A_pattern, A_rows, A_cols = row_level_pattern(self.N, _LINE_UNIT_COLS)
            k_of_col = A_cols // NU
            local_col = A_cols % NU
            ineq_rows = N_INEQ * self.N
            local_row = np.where(A_rows < ineq_rows, A_rows - N_INEQ * k_of_col,
                                 N_INEQ + (A_rows - ineq_rows - N_EQ * k_of_col))
            self._A_gather = (k_of_col * (N_INEQ + N_EQ) + local_row) * NU + local_col
        self._rot_fn = build_rot_function(self.template)

    # ------------------------------------------------------------------ 공식 (기호식)
    def formulate(self, ref, reduced):
        n = self.N
        psi = np.asarray(ref.psi, dtype=float).reshape(1, n)
        r6 = np.asarray(ref.r, dtype=float).reshape(n, 6).T                 # (6, N)
        m_col = np.full((1, n), float(reduced.mass))
        I9 = np.tile(np.asarray(reduced.inertia_B, dtype=float).reshape(-1, 1), (1, n))
        Ad_all, Bd_all = self._step_map(psi, r6, m_col, I9)
        A_d = np.array(Ad_all).reshape(NX, n, NX).transpose(1, 0, 2)
        B_d = np.array(Bd_all).reshape(NX, n, NU).transpose(1, 0, 2)
        A_qp, B_qp = lift(A_d, B_d)                                         # condensing (구조적 — 재사용)
        return A_qp, B_qp, A_d, B_d

    def build_cost(self, A_qp, B_qp, x0, X_ref_flat, mode):
        n = self.N
        psi = X_ref_flat[2::NX].reshape(1, n)
        Qd = np.tile(self.Q_diag[mode].reshape(-1, 1), (1, n))
        Qk = np.array(self._qk_map(psi, Qd)).reshape(NX, n, NX).transpose(1, 0, 2)
        e = A_qp @ x0 - X_ref_flat
        B3 = B_qp.reshape(n, NX, self.nv)
        Wb = (Qk @ B3).reshape(NX * n, self.nv)
        we = (Qk @ e.reshape(n, NX, 1)).reshape(-1)
        H = 2.0 * (B_qp.T @ Wb)
        idx = np.arange(self.nv)
        H[idx, idx] += 2.0 * np.tile(self.R_diag[mode], n)
        H = 0.5 * (H + H.T)
        q = 2.0 * (B_qp.T @ we)
        return H, q

    def _rot_steps(self, yaws):
        """스텝별 yaw (N,) → (N,12,6) 회전된 제약 블록. 모두 같으면 한 번만 계산 (reference 경로와 같은 값)."""
        if np.all(yaws == yaws[0]):
            return np.array(self._rot_fn(yaws[0]))[None].repeat(len(yaws), 0)
        return np.stack([np.array(self._rot_fn(v)) for v in yaws])

    def build_constraints(self, steps, foot_yaw):
        n = self.N
        stance = np.array([np.asarray(s.stance, dtype=bool) for s in steps]).reshape(n, 2)
        min_scale = np.clip(np.array([np.asarray(s.min_scale, dtype=float) for s in steps]).reshape(n, 2), 0.0, 1.0)
        fy = np.asarray(foot_yaw, dtype=float)
        if fy.ndim == 1:                                     # reference: 다리당 yaw 하나
            fy = np.tile(fy.reshape(1, 2), (n, 1))
        yawL = np.array([resolved_foot_yaw(v) for v in fy[:, 0]])   # (N,) — 변형 foot2 는 스텝별 (Q&A 9/28 Q10)
        yawR = np.array([resolved_foot_yaw(v) for v in fy[:, 1]])
        CWL = self._rot_steps(yawL)
        CWR = self._rot_steps(yawR)
        blk = np.zeros((n, N_INEQ + N_EQ, NU))
        sL = stance[:, 0].astype(float)[:, None, None]
        sR = stance[:, 1].astype(float)[:, None, None]
        blk[:, 0:12, LEFT_MAP] = CWL * sL
        blk[:, 12:24, RIGHT_MAP] = CWR * sR
        swL = ~stance[:, 0]
        swR = ~stance[:, 1]
        for i in (0, 1, 2, 6, 7, 8):                        # D: 스윙 발 변수 = 0
            blk[swL, N_INEQ + i, i] = 1.0
        for i in (3, 4, 5, 9, 10, 11):
            blk[swR, N_INEQ + i, i] = 1.0
        if self.foot_model in ("line", "mx0"):              # D: 디딤 발 mx^F = x̂_F·M_W = 0 (no_roll_moment 자리)
            blk[stance[:, 0], N_INEQ + 6, 6] = np.cos(yawL[stance[:, 0]])
            blk[stance[:, 0], N_INEQ + 6, 7] = np.sin(yawL[stance[:, 0]])
            blk[stance[:, 1], N_INEQ + 9, 9] = np.cos(yawR[stance[:, 1]])
            blk[stance[:, 1], N_INEQ + 9, 10] = np.sin(yawR[stance[:, 1]])
        ub_ineq = np.zeros((n, N_INEQ))
        ub_ineq[stance[:, 0], 4] = self.fz_max
        ub_ineq[:, 5] = np.where(stance[:, 0], -min_scale[:, 0] * self.fz_min, 0.0)
        ub_ineq[stance[:, 1], 16] = self.fz_max
        ub_ineq[:, 17] = np.where(stance[:, 1], -min_scale[:, 1] * self.fz_min, 0.0)
        if self.point_fmin > 0.0:                                # 변형: 두 점 최소 수직력 (Q&A 9/28 Q12)
            for side, off in ((0, 0), (1, 12)):
                rhs = np.where(stance[:, side], -self._cop_span * min_scale[:, side] * self.point_fmin, 0.0)
                ub_ineq[:, off + self._cop_rows[0]] = rhs
                ub_ineq[:, off + self._cop_rows[1]] = rhs
        l = np.concatenate([np.full(N_INEQ * n, -np.inf), np.zeros(N_EQ * n)])
        u = np.concatenate([ub_ineq.reshape(-1), np.zeros(N_EQ * n)])
        sig = tuple(int(v) for v in stance.reshape(-1))
        return blk, l, u, sig, stance
