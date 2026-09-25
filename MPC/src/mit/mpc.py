"""Reference convex MPC (SRB, Di Carlo 틀) — reference C++ 를 그대로 옮긴 것 (07_port_design.md §3.7).

구성 (C++ 대응):
    build_reference   ReferenceTrajectory.cpp:8-83 + BodyMotionReference.cpp (yaw mode 'always')   spec 03 §4.3 / 02 §6.3
    mpc_foot_yaw      My_Controller.cpp:930-946 (mpcFootYawForSide)                                spec 05 §8.1 / 02 §5.5
    ConvexMPC         MPCFormulation.cpp:39-107 (A_c/B_c, ZOH, lifted A_qp/B_qp)                   spec 02 §7
                      ConvexMPC.cpp:675-685, 835-869 (yaw 회전 cost, H, q)                          spec 02 §8
                      ConvexMPC.cpp:128-268, 406-576 (행 단위 고정 패턴, C_F 회전, 스윙 등식)       spec 02 §9
                      GaitScheduler.cpp:134-193 (C_bound: Fz 상한 / −min_scale·Fz_min)             spec 02 §5.3
                      ConvexMPC.cpp:880-1108 (OSQP cold/warm, 재시도, shifted warm start)          spec 02 §10
                      My_Controller.cpp:1016-1052 (실패 시 Fz = m|g|/nStance fallback)              spec 05 §8.4

규약 (07 §2): 다리 0 = Left, 1 = Right.  u_k = [F_L, F_R, M_L, M_R] (world, 지면이 몸에 주는 wrench,
모멘트는 발 site 기준).  x = [roll, pitch, yaw_unwrapped, com(3), omega_W(3), v_com(3), g].
X (lifted) 의 행 블록 k = x_{k+1} (u_k 적용 후) ↔ X_ref 행 k.

D5: python osqp 1.1.3 에 reference 의 명시 설정 (verbose F, warm_start T, polish F, max_iter 200,
adaptive_rho T) + spec 02 §10.1 의 1.x 보정 (adaptive_rho_interval 100, check_dualgap False) 을 준다.
solver="quadprog" 는 비교용 정확 QP (warm start 없음).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

try:
    import osqp
except ImportError:          # quadprog 만 쓰는 경우
    osqp = None

# u 인덱스 (ConvexMPC.cpp:192-193)
LEFT_MAP = np.array([0, 1, 2, 6, 7, 8])
RIGHT_MAP = np.array([3, 4, 5, 9, 10, 11])
NX, NU = 13, 12
N_INEQ, N_EQ = 24, 12          # per step


# ---------------------------------------------------------------------------
# 작은 도움 함수
# ---------------------------------------------------------------------------
def _cfg(cfg, path, default=None):
    """cfg.a.b.c 접근 (없으면 default). config.Cfg / SimpleNamespace / dict 모두 허용."""
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            try:
                cur = getattr(cur, key)
            except AttributeError:
                return default
    return default if cur is None else cur


def _horizon(cfg):
    """(N, dt_mpc). dt = horizon / steps 를 C++ 와 같은 식으로 (ControllerConfig.cpp:645-647)."""
    n = int(_cfg(cfg, "timing.horizon_steps"))
    dt = float(_cfg(cfg, "timing.horizon")) / float(n)
    return n, dt


def _mode_str(mode) -> str:
    if not isinstance(mode, str):
        val = getattr(mode, "value", None)
        mode = val if isinstance(val, str) else getattr(mode, "name", str(mode))
    m = str(mode).strip().lower()
    if m not in ("walking", "standing"):
        raise ValueError(f"unknown locomotion mode {mode!r}")
    return m


def wrap_to_pi(a: float) -> float:
    """AngleUtils.h:8-10"""
    return float(np.arctan2(np.sin(a), np.cos(a)))


def lift_angle_near(target_wrapped: float, current_unwrapped: float) -> float:
    """AngleUtils.h:16-18  current + wrapToPi(target − current)"""
    return current_unwrapped + wrap_to_pi(target_wrapped - current_unwrapped)


def Rz(a: float) -> np.ndarray:
    """MatrixUtils.h:11-21"""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def Rz_batch(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R


def skew_batch(v: np.ndarray) -> np.ndarray:
    """MatrixUtils.h:23-30  skew(v) w = v × w.  v (n,3) → (n,3,3)"""
    S = np.zeros((len(v), 3, 3))
    S[:, 0, 1] = -v[:, 2]
    S[:, 0, 2] = v[:, 1]
    S[:, 1, 0] = v[:, 2]
    S[:, 1, 2] = -v[:, 0]
    S[:, 2, 0] = -v[:, 1]
    S[:, 2, 1] = v[:, 0]
    return S


# ---------------------------------------------------------------------------
# Reference trajectory  (ReferenceTrajectory.cpp:8-83, spec 03 §4.3)
# ---------------------------------------------------------------------------
@dataclass
class RefOut:
    X_ref: np.ndarray      # (N,13)  행 k ↔ x_{k+1}
    psi: np.ndarray        # (N,)    Rz(psi_k) (unwrapped)
    r: np.ndarray          # (N,2,3) r[k,leg] = desired_foot[leg] − p_ref[k]  (world)
    tk: np.ndarray         # (N,)    t0 + k·dt  (cycle 원점 기준, HorizonClock.h:32-34)


def build_reference(filtered_cmd, seed_x, desired_feet, clock, cfg) -> RefOut:
    """ReferenceTrajectory::build (ReferenceTrajectory.cpp:8-83) with yaw_integration_mode 'always'.

    seed_x: 호출자가 모드별로 만든 seed (My_Controller.cpp:966-978, spec 05 §8.2).
        walking : x0 에 roll/pitch = bodyTarget.euler_W[0:2], z = nominalHeight_W
        standing: [0:3] = euler_W, [3:6] = nominalPosition_W, [5] = nominalHeight_W
    filtered_cmd: x_dot, y_dot, psi_dot, body_height_offset_m (필터된 명령, MC:610-654)
    desired_feet: (2,3) 이번 tick 의 목표 발 위치 (horizon 전체에 상수, RT:65-66)
    clock: .tk(k) → t0 + k·dt  (없으면 tk = k·dt)
    """
    mode = str(_cfg(cfg, "reference_trajectory.yaw_integration_mode", "always")).lower()
    if mode != "always":
        # single_support/double_support 는 gait.bothFeetStance(tk − dt/2) gate 필요 (BMR:7-24). MIT 는 always.
        raise NotImplementedError(f"yaw_integration_mode {mode!r} not ported (MIT uses 'always')")
    N, dt = _horizon(cfg)
    seed = np.asarray(seed_x, dtype=float)
    feet = np.asarray(desired_feet, dtype=float).reshape(2, 3)

    psi_dot = float(getattr(filtered_cmd, "psi_dot", 0.0))
    h_off = float(getattr(filtered_cmd, "body_height_offset_m", 0.0))
    ux = float(getattr(filtered_cmd, "x_dot", 0.0))
    uy = float(getattr(filtered_cmd, "y_dot", 0.0))

    psi0 = seed[2]
    gravity = seed[12]
    p_ref = seed[3:6].copy()
    p_ref[2] += h_off                                   # RT:32-33
    euler_ref = seed[0:3].copy()
    psi_ref = psi0

    X = np.zeros((N, NX))
    PSI = np.zeros(N)
    TK = np.zeros(N)
    r = np.zeros((N, 2, 3))
    for k in range(N):
        tk = clock.tk(k) if clock is not None else float(k) * dt
        if k > 0 and dt > 0.0:                          # BMR advanceYaw: dt>0 && shouldAdvanceYaw (always)
            psi_ref = psi_ref + psi_dot * dt
        psi_k = psi_ref
        c, s = np.cos(psi_k), np.sin(psi_k)
        v_ref = np.array([c * ux - s * uy, s * ux + c * uy, 0.0])   # Rz(psi_k)·[x_dot, y_dot, 0]  (BMR:58-60)
        if k > 0:
            if dt > 0.0:                                # BMR advancePlanarPosition
                p_ref = p_ref + np.array([c * ux - s * uy, s * ux + c * uy, 0.0]) * dt
            p_ref[2] = seed[5] + h_off                  # RT:61
        TK[k] = tk
        PSI[k] = psi_k
        r[k, 0] = feet[0] - p_ref                       # RT:65
        r[k, 1] = feet[1] - p_ref                       # RT:66
        X[k, 0:3] = euler_ref
        X[k, 2] = psi_k
        X[k, 3:6] = p_ref
        X[k, 8] = psi_dot                               # yawRate (always) — RT:77-80
        X[k, 9:12] = v_ref
        X[k, 12] = gravity
    return RefOut(X_ref=X, psi=PSI, r=r, tk=TK)


def mpc_foot_yaw(state, leg: int, active: bool, touchdown_yaw: float) -> float:
    """mpcFootYawForSide (My_Controller.cpp:930-946).

    active 이면 측정된 발 SITE x축의 yaw 를 touchdown yaw 근처로 lift, 아니면 touchdown yaw.
    footYawFromXAxisWorld (MC:165-176) 가 throw 하는 경우 (비유한 / 수평 성분 ≤ 1e-9) → touchdown yaw.
    """
    if active:
        x_axis = np.asarray(state.foot_R_W[leg])[:, 0]
        if np.all(np.isfinite(x_axis)) and np.hypot(x_axis[0], x_axis[1]) > 1e-9:
            measured = float(np.arctan2(x_axis[1], x_axis[0]))
            return lift_angle_near(measured, touchdown_yaw)
    return touchdown_yaw


# ---------------------------------------------------------------------------
# Formulation (MPCFormulation.cpp:39-107) — 배치 numpy
# ---------------------------------------------------------------------------
def continuous_matrices(psi: np.ndarray, r: np.ndarray, inertia_B: np.ndarray, mass: float):
    """A_c (N,13,13), B_c (N,13,12), I_k (N,3,3).  MPCFormulation.cpp:63-90.

    A_c[0:3,6:9] = R_k^T,  A_c[3:6,9:12] = I,  A_c[9:12,12] = e_z
    B_c[6:9,0:3] = I_k^-1 [r_L]x, [6:9,3:6] = I_k^-1 [r_R]x, [6:9,6:9] = [6:9,9:12] = I_k^-1,
    B_c[9:12,0:3] = B_c[9:12,3:6] = I/m,  I_k^-1 = R_k I_B^-1 R_k^T
    """
    psi = np.asarray(psi, dtype=float)
    n = len(psi)
    R = Rz_batch(psi)
    RT = np.transpose(R, (0, 2, 1))
    I_B = np.asarray(inertia_B, dtype=float)
    I_inv = np.linalg.inv(I_B)
    I_k = R @ I_B @ RT
    I_k_inv = R @ I_inv @ RT
    A_c = np.zeros((n, NX, NX))
    A_c[:, 0:3, 6:9] = RT
    A_c[:, 3, 9] = A_c[:, 4, 10] = A_c[:, 5, 11] = 1.0
    A_c[:, 11, 12] = 1.0
    B_c = np.zeros((n, NX, NU))
    B_c[:, 6:9, 0:3] = I_k_inv @ skew_batch(r[:, 0])
    B_c[:, 6:9, 3:6] = I_k_inv @ skew_batch(r[:, 1])
    B_c[:, 6:9, 6:9] = I_k_inv
    B_c[:, 6:9, 9:12] = I_k_inv
    eye_m = np.eye(3) / mass
    B_c[:, 9:12, 0:3] = eye_m
    B_c[:, 9:12, 3:6] = eye_m
    return A_c, B_c, I_k


def discretize_zoh(A_c: np.ndarray, B_c: np.ndarray, dt: float):
    """discretizeZOH (MPCFormulation.cpp:10-26), 배치.
    A_d = I + dt A + ½dt² A²;  B_d = dt B + ½dt² A B + dt³/6 A² B  (C++ 식 그대로, 3차항 포함 — 값은 0)."""
    A_sq = A_c @ A_c
    A_d = np.eye(NX)[None] + dt * A_c + (0.5 * dt * dt) * A_sq
    B_d = dt * B_c + (0.5 * dt * dt) * (A_c @ B_c) + (dt * dt * dt / 6.0) * (A_sq @ B_c)
    return A_d, B_d


def lift(A_d: np.ndarray, B_d: np.ndarray):
    """A_qp (13N,13), B_qp (13N,12N)  (MPCFormulation.cpp:92-106), C++ 곱 순서 그대로.

    A_qp 블록 k = A_d[k] (A_d[k−1] (… A_d[0]))  (prefix = A_d[k]·prefix)
    B_qp 블록 (k,j) = A_d[k]·(A_d[k−1]·(…(A_d[j+1]·B_d[j])))  (liftedInput = A_d[k+1]·liftedInput)
    대각선 (offset o = k−j) 단위로 배치 곱: L_o[j] = A_d[j+o] @ L_{o−1}[j]  — 같은 곱 순서.
    """
    n = A_d.shape[0]
    A_qp = np.zeros((NX * n, NX))
    prefix = np.eye(NX)
    for k in range(n):
        prefix = A_d[k] @ prefix
        A_qp[NX * k:NX * (k + 1)] = prefix
    B_qp = np.zeros((NX * n, NU * n))
    B4 = B_qp.reshape(n, NX, n, NU)                     # view
    L = B_d.copy()
    for o in range(n):
        m = n - o
        js = np.arange(m)
        B4[js + o, :, js, :] = L[:m]
        if o + 1 < n:
            L = A_d[o + 1:n] @ L[:m - 1]
    return A_qp, B_qp


def yaw_state_cost(Q_diag: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
    """bodyYawStateCost (ConvexMPC.cpp:675-685): T = I13, T[3:6,3:6]=T[6:9,6:9]=T[9:12,9:12]=Rz(psi)^T;
    Q_k = T^T Q T.  배치 (N,13,13)."""
    n = len(psi_ref)
    R_BW = np.transpose(Rz_batch(psi_ref), (0, 2, 1))
    T = np.broadcast_to(np.eye(NX), (n, NX, NX)).copy()
    T[:, 3:6, 3:6] = R_BW
    T[:, 6:9, 6:9] = R_BW
    T[:, 9:12, 9:12] = R_BW
    Q = np.diag(np.asarray(Q_diag, dtype=float))
    return np.transpose(T, (0, 2, 1)) @ Q @ T


def foot_constraint_template(cfg) -> np.ndarray:
    """localFootConstraintTemplateValue (ConvexMPC.cpp:406-437), 12×6, 발 로컬 wrench [Fx,Fy,Fz,Mx,My,Mz]."""
    mu = float(_cfg(cfg, "mpc.friction_coefficient"))
    a = float(_cfg(cfg, "mpc.foot_half_length"))
    b = float(_cfg(cfg, "mpc.foot_half_width"))
    mu_t = float(_cfg(cfg, "mpc.torsional_friction_scale")) * mu
    return np.array([
        [1.0, 0.0, -mu, 0.0, 0.0, 0.0],     # 0   Fx − µFz ≤ 0
        [-1.0, 0.0, -mu, 0.0, 0.0, 0.0],    # 1  −Fx − µFz ≤ 0
        [0.0, 1.0, -mu, 0.0, 0.0, 0.0],     # 2   Fy − µFz ≤ 0
        [0.0, -1.0, -mu, 0.0, 0.0, 0.0],    # 3  −Fy − µFz ≤ 0
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],     # 4   Fz ≤ Fmax
        [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],    # 5  −Fz ≤ −minScale·Fmin
        [0.0, 0.0, -b, 1.0, 0.0, 0.0],      # 6   Mx − b Fz ≤ 0   (half WIDTH)
        [0.0, 0.0, -b, -1.0, 0.0, 0.0],     # 7  −Mx − b Fz ≤ 0
        [0.0, 0.0, -a, 0.0, 1.0, 0.0],      # 8   My − a Fz ≤ 0   (half LENGTH)
        [0.0, 0.0, -a, 0.0, -1.0, 0.0],     # 9  −My − a Fz ≤ 0
        [0.0, 0.0, -mu_t, 0.0, 0.0, 1.0],   # 10  Mz − µ_t Fz ≤ 0
        [0.0, 0.0, -mu_t, 0.0, 0.0, -1.0],  # 11 −Mz − µ_t Fz ≤ 0
    ])


def yaw_rotated_block(template: np.ndarray, yaw: float) -> np.ndarray:
    """yawRotatedFootConstraintValue (ConvexMPC.cpp:439-464) = C_F · blkdiag(Rz(yaw)^T, Rz(yaw)^T)."""
    c, s = np.cos(yaw), np.sin(yaw)
    T = template
    out = np.empty((12, 6))
    out[:, 0] = c * T[:, 0] - s * T[:, 1]
    out[:, 1] = s * T[:, 0] + c * T[:, 1]
    out[:, 2] = T[:, 2]
    out[:, 3] = c * T[:, 3] - s * T[:, 4]
    out[:, 4] = s * T[:, 3] + c * T[:, 4]
    out[:, 5] = T[:, 5]
    return out


def resolved_foot_yaw(yaw: float) -> float:
    """resolvedFootYaw (GaitScheduler.cpp:25-30): NaN → yaw of the stored foot x-axis (기본 UnitX → 0;
    NoRollMoment 가 아니면 저장된 축은 항상 UnitX)."""
    return float(yaw) if np.isfinite(yaw) else 0.0


# ---------------------------------------------------------------------------
# 고정 sparsity 패턴 (ConvexMPC.cpp:50-64, 128-268)
# ---------------------------------------------------------------------------
_C_UNIT_COLS = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [2], [2],
                [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 5], [2, 5]]


def upper_triangular_pattern(n: int):
    """makeUpperTriangularPattern: 열 우선, row ≤ col 전부 (명시적 0 포함). → (P csc, rows, cols)."""
    cols = np.repeat(np.arange(n), np.arange(1, n + 1))
    rows = np.concatenate([np.arange(c + 1) for c in range(n)])
    indptr = np.concatenate([[0], np.cumsum(np.arange(1, n + 1))])
    P = sp.csc_matrix((np.zeros(len(rows)), rows.astype(np.int64), indptr.astype(np.int64)), shape=(n, n))
    return P, rows, cols


def row_level_constraint_pattern(steps: int):
    """makeRowLevelConstraintPattern: step 당 C 60 + D 16 = 76 entries. → (A csc, rows, cols) CSC 저장 순서."""
    ineq_rows = N_INEQ * steps
    trip = []
    for k in range(steps):
        u0 = NU * k
        c0 = N_INEQ * k
        d0 = ineq_rows + N_EQ * k
        for r in range(12):
            for lc in _C_UNIT_COLS[r]:
                trip.append((c0 + r, u0 + LEFT_MAP[lc]))
        for r in range(12):
            for lc in _C_UNIT_COLS[r]:
                trip.append((c0 + 12 + r, u0 + RIGHT_MAP[lc]))
        for i in range(6):
            trip.append((d0 + i, u0 + i))
        trip += [(d0 + 6, u0 + 6), (d0 + 6, u0 + 7), (d0 + 6, u0 + 8), (d0 + 7, u0 + 7), (d0 + 8, u0 + 8)]
        trip += [(d0 + 9, u0 + 9), (d0 + 9, u0 + 10), (d0 + 9, u0 + 11), (d0 + 10, u0 + 10), (d0 + 11, u0 + 11)]
    trip = np.array(trip, dtype=np.int64)
    order = np.lexsort((trip[:, 0], trip[:, 1]))        # 열 우선, 열 안에서 행 오름차순 (Eigen compressed)
    trip = trip[order]
    rows, cols = trip[:, 0], trip[:, 1]
    nvars = NU * steps
    counts = np.bincount(cols, minlength=nvars)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    A = sp.csc_matrix((np.zeros(len(rows)), rows, indptr), shape=((N_INEQ + N_EQ) * steps, nvars))
    return A, rows, cols


# ---------------------------------------------------------------------------
# ConvexMPC
# ---------------------------------------------------------------------------
_OSQP_OK = (1, 2)      # OSQP_SOLVED, OSQP_SOLVED_INACCURATE  (ConvexMPC.cpp:658-661)


class MpcSolveError(RuntimeError):
    pass


class ConvexMPC:
    """ConvexMPC + MPCFormulation (reference). 한 객체 = 한 OSQP workspace (warm start 상태 유지).

    solve(x0, ref, steps, foot_yaw, reduced, mode, active=None) → (u0 (12,), info)
      steps   : 길이 N, 각 항목 .stance (2,) bool, .min_scale (2,) (override 반영된 것, GaitScheduler)
      foot_yaw: (2,) 제약 회전용 발 yaw (mpc_foot_yaw), horizon 전체 공통
      reduced : .mass, .inertia_B (yaw frame, 매 tick 갱신값)
      mode    : 'walking' | 'standing' (Q/R 선택, ConvexMPC.cpp:808-816)
      active  : (2,) bool — 실패 시 fallback Fz 를 줄 다리 (activeContactForSide). None → steps[0].stance
    info: status, status_val, iters, prim_res, dual_res, solve_ms (OSQP setup/update + solve), build_ms,
          total_ms, cold, retry, fallback, sig_changed, U (N,12; fallback 이면 u0 반복), obj, error
    주의: OSQP (eps 1e-3, max_iter 200) 해는 정확 QP 해와 u0 가 수십 % 다를 수 있다 (목적함수 차는 1e-3 이하) —
    reference 도 같은 설정이므로 그대로 둔다 (02_unit_checks_C.py §4 참고).
    성능: 멀티스레드 BLAS 는 300×325 곱에서 오히려 느려질 수 있다 → OPENBLAS_NUM_THREADS=1 권장.
    """

    def __init__(self, cfg, solver: str = "osqp", osqp_settings: dict | None = None, verbose_errors: bool = True):
        self.cfg = cfg
        self.solver_name = str(solver).lower()
        if self.solver_name not in ("osqp", "quadprog"):
            raise ValueError(f"solver must be 'osqp' or 'quadprog', got {solver!r}")
        if self.solver_name == "osqp" and osqp is None:
            raise ImportError("osqp not installed")
        self.N, self.dt = _horizon(cfg)
        self.nv = NU * self.N
        self.nc = (N_INEQ + N_EQ) * self.N
        self.gravity = float(_cfg(cfg, "model.gravity", -9.81))
        self.use_shifted_warm_start = bool(_cfg(cfg, "mpc.use_shifted_warm_start", True))
        self.fz_max = float(_cfg(cfg, "mpc.normal_force_max"))
        self.fz_min = float(_cfg(cfg, "mpc.normal_force_min"))
        for m in ("walking", "standing"):
            if str(_cfg(cfg, f"mpc.{m}.contact_wrench_model", "full_wrench")) != "full_wrench":
                # no_roll_moment 는 D 행 6/9 에 발 x축이 들어간다 (ConvexMPC.cpp:554-568) — MIT 는 full_wrench.
                raise NotImplementedError("contact_wrench_model no_roll_moment not ported (MIT: full_wrench)")
        self.Q_diag = {m: np.asarray(_cfg(cfg, f"mpc.{m}.state_weight_diag"), dtype=float) for m in ("walking", "standing")}
        self.R_diag = {m: np.asarray(_cfg(cfg, f"mpc.{m}.input_weight_diag"), dtype=float) for m in ("walking", "standing")}
        self.template = foot_constraint_template(cfg)

        # 고정 패턴 (명시적 0 포함, nnz 고정 → update(Px, Ax) 가능)
        self.P_pattern, self._P_rows, self._P_cols = upper_triangular_pattern(self.nv)
        self.A_pattern, A_rows, A_cols = row_level_constraint_pattern(self.N)
        # A 값 = step 별 dense 블록 (N,36,12) 에서 gather
        k_of_col = A_cols // NU
        local_col = A_cols % NU
        ineq_rows = N_INEQ * self.N
        local_row = np.where(A_rows < ineq_rows, A_rows - N_INEQ * k_of_col,
                             N_INEQ + (A_rows - ineq_rows - N_EQ * k_of_col))
        self._A_gather = (k_of_col * (N_INEQ + N_EQ) + local_row) * NU + local_col

        # reference 명시 5개 (ConvexMPC.cpp:1024-1028) + 1.x 보정 (spec 02 §10.1) + 0.6 기본값 명시.
        # rho 는 넘기지 않는다: 1.1.3 기본 0.1 과 같고, 넘기면 setup 뒤 update_rho 로 KKT 를 한 번 더 분해한다.
        self.settings = dict(verbose=False, warm_starting=True, polishing=False, max_iter=200,
                             adaptive_rho=1, adaptive_rho_interval=100, check_dualgap=0,
                             eps_abs=1e-3, eps_rel=1e-3, eps_prim_inf=1e-4, eps_dual_inf=1e-4,
                             sigma=1e-6, alpha=1.6, scaling=10, check_termination=25,
                             scaled_termination=0, delta=1e-6, adaptive_rho_tolerance=5.0,
                             adaptive_rho_fraction=0.4)
        if osqp_settings:
            self.settings.update(osqp_settings)
        self.verbose_errors = verbose_errors
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        """solver workspace 와 warm start 를 버린다 (새 ConvexMPC 와 같음)."""
        self._solver = None
        self._initialized = False
        self._has_sig = False
        self._sig = None
        self._skip_warm = False
        self._has_prev = False
        self._warm = np.zeros(self.nv)
        self._last_solution = np.zeros(self.nv)
        # 데이터 (마지막 build)
        self._Px = None
        self._q = None
        self._Ax = None
        self._l = None
        self._u = None
        self.stats = dict(solves=0, cold=0, sig_changes=0, retries=0, failures=0)

    # ------------------------------------------------------------------ build
    def formulate(self, ref: RefOut, reduced):
        """MPCFormulation::build → (A_qp, B_qp, A_d, B_d)."""
        A_c, B_c, _ = continuous_matrices(ref.psi, ref.r, reduced.inertia_B, float(reduced.mass))
        A_d, B_d = discretize_zoh(A_c, B_c, self.dt)
        A_qp, B_qp = lift(A_d, B_d)
        return A_qp, B_qp, A_d, B_d

    def build_cost(self, A_qp, B_qp, x0, X_ref_flat, mode):
        """ConvexMPC::buildQP cost part (ConvexMPC.cpp:835-869) → dense H (nv,nv), q (nv,)."""
        n = self.N
        Qk = yaw_state_cost(self.Q_diag[mode], X_ref_flat[2::NX])          # psiRef = X_ref[13k+2]
        e = A_qp @ x0 - X_ref_flat                                          # _stateError
        B3 = B_qp.reshape(n, NX, self.nv)
        Wb = (Qk @ B3).reshape(NX * n, self.nv)
        we = (Qk @ e.reshape(n, NX, 1)).reshape(-1)
        H = 2.0 * (B_qp.T @ Wb)
        idx = np.arange(self.nv)
        H[idx, idx] += 2.0 * np.tile(self.R_diag[mode], n)                  # block += 2R (R 는 대각)
        H = 0.5 * (H + H.T)
        q = 2.0 * (B_qp.T @ we)
        return H, q

    def build_constraints(self, steps, foot_yaw):
        """(A dense blocks (N,36,12), l (nc,), u (nc,), signature tuple)."""
        n = self.N
        stance = np.array([np.asarray(s.stance, dtype=bool) for s in steps]).reshape(n, 2)
        min_scale = np.clip(np.array([np.asarray(s.min_scale, dtype=float) for s in steps]).reshape(n, 2), 0.0, 1.0)
        yawL = resolved_foot_yaw(foot_yaw[0])
        yawR = resolved_foot_yaw(foot_yaw[1])
        CWL = yaw_rotated_block(self.template, yawL)
        CWR = yaw_rotated_block(self.template, yawR)
        blk = np.zeros((n, N_INEQ + N_EQ, NU))
        sL = stance[:, 0].astype(float)[:, None, None]
        sR = stance[:, 1].astype(float)[:, None, None]
        blk[:, 0:12, LEFT_MAP] = CWL[None] * sL
        blk[:, 12:24, RIGHT_MAP] = CWR[None] * sR
        swL = ~stance[:, 0]
        swR = ~stance[:, 1]
        for i in (0, 1, 2, 6, 7, 8):                      # D: 스윙 발 변수 = 0 (ConvexMPC.cpp:544-575)
            blk[swL, N_INEQ + i, i] = 1.0
        for i in (3, 4, 5, 9, 10, 11):
            blk[swR, N_INEQ + i, i] = 1.0
        ub_ineq = np.zeros((n, N_INEQ))                   # C_bound (GaitScheduler.cpp:149-180)
        ub_ineq[stance[:, 0], 4] = self.fz_max
        ub_ineq[:, 5] = np.where(stance[:, 0], -min_scale[:, 0] * self.fz_min, 0.0)
        ub_ineq[stance[:, 1], 16] = self.fz_max
        ub_ineq[:, 17] = np.where(stance[:, 1], -min_scale[:, 1] * self.fz_min, 0.0)
        l = np.concatenate([np.full(N_INEQ * n, -np.inf), np.zeros(N_EQ * n)])
        u = np.concatenate([ub_ineq.reshape(-1), np.zeros(N_EQ * n)])
        sig = tuple(int(v) for v in stance.reshape(-1))   # [L0,R0,L1,R1,...] (ConvexMPC.cpp:1071-1084)
        return blk, l, u, sig, stance

    # ------------------------------------------------------------------ solve
    def solve(self, x0, ref: RefOut, steps, foot_yaw, reduced, mode, active=None):
        t_start = time.perf_counter()
        mode = _mode_str(mode)
        info = dict(status="", status_val=0, iters=0, solve_ms=0.0, build_ms=0.0, total_ms=0.0,
                    cold=False, retry=False, fallback=False, sig_changed=False, U=None, obj=np.nan, error=None)
        x0 = np.asarray(x0, dtype=float)
        stance0 = None
        t_solve = None
        try:
            if len(steps) != self.N:
                raise MpcSolveError(f"steps length {len(steps)} != N {self.N}")
            if float(reduced.mass) <= 0.0:
                raise MpcSolveError("MPCFormulation requires positive bodyMass")
            X_ref_flat = np.asarray(ref.X_ref, dtype=float).reshape(-1)
            A_qp, B_qp, _, _ = self.formulate(ref, reduced)
            H, q = self.build_cost(A_qp, B_qp, x0, X_ref_flat, mode)
            blk, l, u, sig, stance = self.build_constraints(steps, foot_yaw)
            stance0 = stance[0]
            info["build_ms"] = (time.perf_counter() - t_start) * 1e3
            t_solve = time.perf_counter()
            if self.solver_name == "osqp":
                Px = H[self._P_rows, self._P_cols]
                Ax = blk.reshape(-1)[self._A_gather]
                U = self._solve_osqp(Px, q, Ax, l, u, sig, info)
            else:
                U = self._solve_quadprog(H, q, blk, u, sig, info)
            info["solve_ms"] = (time.perf_counter() - t_solve) * 1e3
            info["obj"] = float(0.5 * U @ H @ U + q @ U)
            u0 = U[:NU].copy()
            info["U"] = U.reshape(self.N, NU)
        except Exception as exc:              # MC:1016-1052 catch (std::exception) → fallback
            if t_solve is not None:
                info["solve_ms"] = (time.perf_counter() - t_solve) * 1e3
            self.stats["failures"] += 1
            info["error"] = f"{type(exc).__name__}: {exc}"
            if self.verbose_errors:
                print(f"[MPC] solve failed: {info['error']}", file=sys.stderr)
            if active is None:
                active = stance0 if stance0 is not None else np.array(
                    [bool(steps[0].stance[0]), bool(steps[0].stance[1])])
            u0 = self.fallback_wrench(active, float(reduced.mass))
            info["fallback"] = True
            info["U"] = np.tile(u0, (self.N, 1))
        self.stats["solves"] += 1
        info["total_ms"] = (time.perf_counter() - t_start) * 1e3
        return u0, info

    def fallback_wrench(self, active, mass: float) -> np.ndarray:
        """MC:1030-1051: u = 0; 활성 접촉 다리마다 Fz = m|g| / nStance (index 2 = L, 5 = R), 모멘트 0."""
        u0 = np.zeros(NU)
        active = np.asarray(active, dtype=bool)
        n_st = int(active.sum())
        if n_st > 0:
            fz = mass * abs(self.gravity) / float(n_st)
            if active[0]:
                u0[2] = fz
            if active[1]:
                u0[5] = fz
        return u0

    # ---- OSQP (ConvexMPC.cpp:871-1108)
    def _init_solver(self, Px, q, Ax, l, u):
        """initializeSolver: 새 workspace (x = y = 0), 현재 데이터로 setup."""
        self._solver = None
        self._initialized = False
        P = sp.csc_matrix((Px, self.P_pattern.indices, self.P_pattern.indptr), shape=self.P_pattern.shape)
        A = sp.csc_matrix((Ax, self.A_pattern.indices, self.A_pattern.indptr), shape=self.A_pattern.shape)
        s = osqp.OSQP()
        s.setup(P, q, A, l, u, **self.settings)
        self._solver = s
        self._initialized = True

    def _solve_once(self, use_warm: bool, info):
        """solveCurrentProblem (ConvexMPC.cpp:915-948). primal 만 warm start (dual y 는 workspace 에 남은 값)."""
        if use_warm and self._has_prev and self._initialized and not self._skip_warm \
                and np.all(np.isfinite(self._warm)):
            self._solver.warm_start(x=self._warm)
        res = self._solver.solve(raise_error=False)
        info["iters"] = int(res.info.iter)
        info["status"] = str(res.info.status)
        info["status_val"] = int(res.info.status_val)
        info["prim_res"] = float(res.info.prim_res)
        info["dual_res"] = float(res.info.dual_res)
        return res

    def _solve_osqp(self, Px, q, Ax, l, u, sig, info):
        # buildQP 후반 (ConvexMPC.cpp:871-906)
        sig_changed = self._has_sig and sig != self._sig
        cold = (not self._initialized) or sig_changed
        info["sig_changed"] = sig_changed
        info["cold"] = cold
        if sig_changed:
            self.stats["sig_changes"] += 1
        if cold:
            self.stats["cold"] += 1
            self._init_solver(Px, q, Ax, l, u)
        else:
            self._solver.update(q=q, l=l, u=u, Px=Px, Ax=Ax)
        self._Px, self._q, self._Ax, self._l, self._u = Px, q, Ax, l, u
        self._sig = sig
        self._has_sig = True
        self._skip_warm = sig_changed

        # solve (ConvexMPC.cpp:950-993)
        res = self._solve_once(not self._skip_warm, info)
        if res.info.status_val not in _OSQP_OK:
            first = info["status"]
            self.stats["retries"] += 1
            info["retry"] = True
            self._init_solver(Px, q, Ax, l, u)                 # fresh re-init (x = y = 0), no warm start
            res = self._solve_once(False, info)
            if res.info.status_val not in _OSQP_OK:
                raise MpcSolveError(f"OSQP did not converge: first_status={first} retry_status={info['status']}")
        U = np.array(res.x, dtype=float, copy=True)
        if U.shape != (self.nv,) or not np.all(np.isfinite(U)):
            raise MpcSolveError("OSQP returned an invalid solution")
        self._last_solution = U.copy()
        self._update_warm_start()
        self._has_prev = True
        self._skip_warm = False
        return U

    def _update_warm_start(self):
        """updateWarmStart (ConvexMPC.cpp:1094-1108): 한 step (12) 당겨오고 마지막 step 반복."""
        if not self.use_shifted_warm_start:
            self._warm = self._last_solution.copy()
            return
        w = np.empty(self.nv)
        w[:self.nv - NU] = self._last_solution[NU:]
        w[self.nv - NU:] = self._last_solution[self.nv - NU:]
        self._warm = w

    # ---- quadprog (비교용 정확 QP)
    def _solve_quadprog(self, H, q, blk, u, sig, info):
        import quadprog
        n = self.N
        rows_eq, rows_in, b_in = [], [], []
        ub_ineq = u[:N_INEQ * n].reshape(n, N_INEQ)
        for k in range(n):
            for r in range(N_INEQ):
                row = blk[k, r]
                if np.any(row != 0.0):
                    full = np.zeros(self.nv)
                    full[NU * k:NU * (k + 1)] = row
                    rows_in.append(-full)                 # C u ≤ b  →  −C u ≥ −b
                    b_in.append(-ub_ineq[k, r])
            for r in range(N_EQ):
                row = blk[k, N_INEQ + r]
                if np.any(row != 0.0):
                    full = np.zeros(self.nv)
                    full[NU * k:NU * (k + 1)] = row
                    rows_eq.append(full)
        C = np.array(rows_eq + rows_in).T
        b = np.concatenate([np.zeros(len(rows_eq)), np.array(b_in)])
        sol = quadprog.solve_qp(H, -q, C, b, meq=len(rows_eq))
        U = np.asarray(sol[0], dtype=float)
        info["status"] = "quadprog_solved"
        info["status_val"] = 1
        info["iters"] = int(sol[3][0])
        info["sig_changed"] = self._has_sig and sig != self._sig
        self._sig = sig
        self._has_sig = True
        self._last_solution = U.copy()
        self._update_warm_start()
        self._has_prev = True
        return U
