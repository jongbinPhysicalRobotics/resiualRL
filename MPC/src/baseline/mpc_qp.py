from __future__ import annotations

import time

import numpy as np
import mujoco
import quadprog

from mpc_srb import (NX, NU, NU_PER_FOOT, N_FEET, FOOT_SITES, SRBParams,
                     continuous_AB, discretize)


#제약조건
def foot_constraints(p: SRBParams, psi: float = 0.0) -> tuple[np.ndarray, np.ndarray]:

    mu, w, lt, lh, h = p.mu, p.w, p.l_t, p.l_h, p.h_sole
    C = np.array([
        [1,  0, -mu,  0,  0, 0],    # Fx ≤ μFz
        [-1, 0, -mu,  0,  0, 0],    # −Fx ≤ μFz
        [0,  1, -mu,  0,  0, 0],    # Fy ≤ μFz
        [0, -1, -mu,  0,  0, 0],    # −Fy ≤ μFz
        [0,  0,  1,   0,  0, 0],    # Fz ≤ Fz_max
        [0,  0, -1,   0,  0, 0],    # −Fz ≤ −Fz_min
        [0, -h, -w,   1,  0, 0],    # mx − h·Fy ≤ w·Fz    (롤 CoP)
        [0,  h, -w,  -1,  0, 0],    # −mx + h·Fy ≤ w·Fz
        [h,  0, -lh,  0,  1, 0],    # my + h·Fx ≤ l_h·Fz  (피치 CoP, 뒤꿈치)
        [-h, 0, -lt,  0, -1, 0],    # −my − h·Fx ≤ l_t·Fz (피치 CoP, 발가락)
    ], dtype=float)

    Xr, Yr, xc = (lt + lh) / 2.0, w, (lt - lh) / 2.0
    fx_r = np.array([1., 0, 0, 0, 0, 0]); fy_r = np.array([0., 1, 0, 0, 0, 0])
    fz_r = np.array([0., 0, 1, 0, 0, 0])
    tx_r = np.array([0., -h, 0, 1, 0, 0])
    ty_r = np.array([h, 0., xc, 0, 1, 0])
    tz_r = np.array([0., -xc, 0, 0, 0, 1])
    mz_rows = []
    for s1 in (1., -1.):
        for s2 in (1., -1.):
            mz_rows.append(tz_r + s1 * (Yr * fx_r + mu * tx_r)
                           + s2 * (Xr * fy_r + mu * ty_r) - mu * (Xr + Yr) * fz_r)
            mz_rows.append(-tz_r + s1 * (Yr * fx_r - mu * tx_r)
                           + s2 * (Xr * fy_r - mu * ty_r) - mu * (Xr + Yr) * fz_r)
    C = np.vstack([C, np.array(mz_rows)])
    if abs(psi) > 1e-12:
        Rz_ = np.zeros((6, 6))
        from mpc_srb import rz
        Rz_[:3, :3] = Rz_[3:, 3:] = rz(psi)
        C = C @ Rz_.T                # 제약은 발(yaw) frame: C·(Rzᵀ W) ≤ d
    d = np.array([0, 0, 0, 0, p.fz_max, -p.fz_min, 0, 0, 0, 0]
                 + [0.0] * 8, dtype=float)           # mz 8행 (전부 Fz 비례)
    return C, d


# mpc 본체
# COST
Q_DEFAULT = np.array([10, 10, 20,   50, 50, 300,   5, 5, 5,   20, 20, 50,   0],
                     dtype=float)

R_FORCE = 1e-5
R_MOMENT = 1e-5  
R_DEFAULT = np.tile(np.array([R_FORCE] * 3 + [R_MOMENT] * 3), N_FEET)


class WrenchMPC:


    def __init__(self, params: SRBParams, horizon: int = 10, dt: float = 0.02,
                 q_diag=None, r_diag=None, psi0: float = 0.0, du_w=None):
        self.p = params
        self.du_w = None if du_w is None else np.asarray(du_w, float)
        self.u_prev = None
        self.N = horizon
        self.dt = dt
        self.Q = np.diag(Q_DEFAULT if q_diag is None else np.asarray(q_diag, float))
        self.R = np.diag(R_DEFAULT if r_diag is None else np.asarray(r_diag, float))
        self.Q_bar = np.kron(np.eye(self.N), self.Q)
        self.R_bar = np.kron(np.eye(self.N), self.R)
        self.q_bar = np.tile(np.diag(self.Q), self.N)
        self.r_bar = np.tile(np.diag(self.R), self.N)

        Cf, df = foot_constraints(params, psi=psi0)
        self.Cf, self.df = Cf, df                       # gait 모드에서 재사용
        C_step = np.kron(np.eye(N_FEET), Cf)            # 20 x 12
        d_step = np.tile(df, N_FEET)
        self.C_all = np.kron(np.eye(self.N), C_step)    # 20N x 12N
        self.d_all = np.tile(d_step, self.N)

        self.nU = NU * self.N
        self.nC = self.C_all.shape[0]
        self.solver_name = "quadprog"
        self.last = {}                                  # 디버그용 (H, g, U, ...)

    def _solve_qp(self, H, g):
        """min ½uᵀHu + gᵀu  s.t.  C_all·u ≤ d_all.  해 u 반환."""
        try:
            # quadprog: min ½xᵀGx − aᵀx  s.t. Cᵀx ≥ b   →  G=H, a=−g, C=−C_allᵀ, b=−d_all
            u, *_ = quadprog.solve_qp(H, -g, -self.C_all.T, -self.d_all)
            self.solver_name = "quadprog"
            return u
        except Exception:               # noqa: BLE001 — 수치 실패 시 OSQP 폴백
            import scipy.sparse as sp
            import osqp
            prob = osqp.OSQP()
            prob.setup(P=sp.csc_matrix(H), q=g, A=sp.csc_matrix(self.C_all),
                       l=-np.inf * np.ones(self.nC), u=self.d_all,
                       verbose=False, eps_abs=1e-8, eps_rel=1e-8)
            res = prob.solve()
            self.solver_name = "osqp"
            return np.asarray(res.x)

    def gravity_u_ref(self) -> np.ndarray:
        """중력 피드포워드 u_ref: 두 발이 Mg/2 씩 수직력, 모멘트 0."""
        u = np.zeros(NU)
        u[2] = u[8] = self.p.mass * 9.81 / N_FEET
        return u

    def solve(self, x0: np.ndarray, x_ref: np.ndarray, psi: float,
              r_feet: np.ndarray, u_ref: np.ndarray | None = None):

        t0 = time.perf_counter()
        Ac, Bc = continuous_AB(self.p, psi, r_feet)
        Ad, Bd = discretize(Ac, Bc, self.dt)

        # A_qp = [Ad; Ad²; …; Ad^N],  B_qp 하삼각 블록
        N = self.N
        A_qp = np.zeros((NX * N, NX))
        B_qp = np.zeros((NX * N, NU * N))
        A_pow = np.eye(NX)
        for k in range(N):
            A_pow = Ad @ A_pow                       # Ad^{k+1}
            A_qp[NX * k:NX * (k + 1)] = A_pow
        for k in range(N):          # 블록 (k, j): Ad^{k-j} Bd
            blk = Bd.copy()
            for j in range(k, -1, -1):
                B_qp[NX * k:NX * (k + 1), NU * j:NU * (j + 1)] = blk
                blk = Ad @ blk
        # 위 루프는 (k,j)에 Ad^{k-j}Bd 를 채운다 (j=k 일 때 Bd)

        x_ref = np.asarray(x_ref, float)
        X_ref = np.tile(x_ref, N) if x_ref.ndim == 1 else x_ref.reshape(-1)

        H = 2.0 * (B_qp.T @ self.Q_bar @ B_qp + self.R_bar)
        H = 0.5 * (H + H.T)                          # 대칭화 (수치)
        g = 2.0 * B_qp.T @ self.Q_bar @ (A_qp @ x0 - X_ref)
        if u_ref is not None:                        # ||u−u_ref||_R 항
            g -= 2.0 * self.R_bar @ np.tile(u_ref, N)

        U = self._solve_qp(H, g)
        u0 = U[:NU]

        info = {
            "solve_ms": (time.perf_counter() - t0) * 1e3,
            "cost": float(0.5 * U @ H @ U + g @ U),
            "violation": float((self.C_all @ U - self.d_all).max()),
            "solver": self.solver_name,
        }
        self.last = {"H": H, "g": g, "U": U, "Ad": Ad, "Bd": Bd,
                     "A_qp": A_qp, "B_qp": B_qp}
        return u0, U, info


    # ------------------------------------------------------------------
    # 보행 모드 (Step 5-6): 접촉 스케줄 + 시변 B + swing 발 W=0 등식 제약
    # ------------------------------------------------------------------
    def solve_gait(self, x0: np.ndarray, X_ref: np.ndarray, psi: float,
                   foot_pos_traj: np.ndarray, contact: np.ndarray,
                   u_ref_traj: np.ndarray, psi_feet=None, fz_scale=None):
        """보행용 풀기.

        X_ref        : (N,13) 스텝별 참조
        foot_pos_traj: (N,2,3) 지평 스텝별 발 위치 계획 (stance=현재값,
                       착지 후=Raibert 목표). swing 구간 값은 무시됨.
        contact      : (N,2) bool — 접촉 스케줄 (Gait.contact_table)
        u_ref_traj   : (N,12) — 스텝별 중력 ff (stance 발끼리 Mg/n 분배)

        swing 발: 해당 6변수 = 0 등식 제약 + B 열 0 (동역학에서도 제거).
        """
        t0 = time.perf_counter()
        N = self.N
        Ac, _ = continuous_AB(self.p, psi, foot_pos_traj[0] - x0[3:6])
        Ad, _ = discretize(Ac, np.zeros((NX, NU)), self.dt)

        # 스텝별 B_d. r_i: k=0 은 '현재 상태' CoM 기준 (Di Carlo / 사용자 MATLAB 의
        # i=1 = x_now 방식 — 검증 리포트 §2), k≥1 은 참조 CoM 기준 (미리 계산).
        Bd = []
        for k in range(N):
            com_k = x0[3:6] if k == 0 else X_ref[k, 3:6]
            r_feet = foot_pos_traj[k] - com_k
            _, Bc = continuous_AB(self.p, psi, r_feet)
            for i in range(N_FEET):
                if not contact[k, i]:
                    Bc[:, NU_PER_FOOT * i:NU_PER_FOOT * (i + 1)] = 0.0
            Bd.append(discretize(Ac, Bc, self.dt)[1])

        A_qp = np.zeros((NX * N, NX))
        B_qp = np.zeros((NX * N, NU * N))
        A_pow = np.eye(NX)
        for k in range(N):
            A_pow = Ad @ A_pow
            A_qp[NX * k:NX * (k + 1)] = A_pow
        for j in range(N):
            blk = Bd[j]
            for k in range(j, N):
                B_qp[NX * k:NX * (k + 1), NU * j:NU * (j + 1)] = blk
                blk = Ad @ blk

        # ★ 스윙 발 변수는 QP 에서 아예 뺀다 (Q&A 9/22 Q6). 예전엔 변수를 두고 등식 u=0
        #   으로 묶었다 — 같은 문제라 해가 같고 (벤치 차이 0), 변수 192 → ~108 개로 줄어
        #   quadprog 가 약 3 배 빠르다. 스윙 열은 위에서 Bc 를 0 으로 만들어 B_qp 에서도 0.
        nU = NU * N
        free_mask = np.repeat(np.asarray(contact, bool).reshape(-1), NU_PER_FOOT)
        free = np.flatnonzero(free_mask)
        eq_idx = np.flatnonzero(~free_mask)
        B_r = B_qp[:, free]
        q, r = self.q_bar, self.r_bar[free]
        H = 2.0 * ((B_r.T * q) @ B_r)
        H[np.diag_indices_from(H)] += 2.0 * r
        g = 2.0 * B_r.T @ (q * (A_qp @ x0 - X_ref.reshape(-1))) \
            - 2.0 * r * u_ref_traj.reshape(-1)[free]
        if self.du_w is not None and np.any(self.du_w > 0):
            D = np.eye(nU) - np.eye(nU, k=-NU)          # 행 k: u_k − u_{k−1}
            e = np.zeros(nU)
            if self.u_prev is not None:
                e[:NU] = self.u_prev                    # u_0 − u_prev
            else:
                D[:NU, :] = 0.0
            D_r = D[:, free]
            DW = D_r.T * np.tile(self.du_w, N)          # D_rᵀ diag(W)
            H = H + 2.0 * DW @ D_r
            g = g - 2.0 * DW @ e
        H = 0.5 * (H + H.T)

        # 제약: stance 발 -> 부등식,  swing 발 -> W=0 등식 6개.
        # frame 주의: 제약은 '그 발이 실제로 놓인 방향' 기준이어야 한다.
        # stance 발은 착지 당시 방향으로 땅에 박혀 있어 몸 yaw 와 다르다 —
        # 몸 yaw 로 돌렸더니 오히려 악화(15s 완주 → 12.4s 전도, 18.6).
        # psi_feet: 발별 실제 site yaw (컨트롤러가 측정해서 전달).
        if psi_feet is None:
            psi_feet = (psi, psi)
        Cf_i, df_i = zip(*(foot_constraints(self.p, psi=pf) for pf in psi_feet))
        blocks = [(k, i) for k in range(N) for i in range(N_FEET) if contact[k, i]]
        nr = Cf_i[0].shape[0]
        C_in = np.zeros((nr * len(blocks), nU))
        for b_, (k, i) in enumerate(blocks):
            base = NU * k + NU_PER_FOOT * i
            C_in[nr * b_:nr * (b_ + 1), base:base + NU_PER_FOOT] = Cf_i[i]
        # 발별·스텝별 Fz 상한 배율 (Q&A 9/22 Q13, 이륙 전 하중 내림). 배율 < 1 이면 하한은 0 으로
        # (상한이 하한보다 작아지는 모순 방지). 배율 0 이면 그 발 wrench 전체가 0 이 된다 (마찰·CoP 행).
        d_list = []
        for (k, i) in blocks:
            dd = df_i[i]
            if fz_scale is not None and fz_scale[k, i] < 1.0:
                dd = dd.copy()
                dd[4] = self.p.fz_max * max(float(fz_scale[k, i]), 0.0)
                dd[5] = 0.0
            d_list.append(dd)
        d_in = np.concatenate(d_list)

        u_r = self._solve_qp_ineq(H, g, C_in[:, free], d_in)
        U = np.zeros(nU)
        U[free] = u_r
        u0 = U[:NU]
        self.u_prev = u0.copy()
        info = {
            "solve_ms": (time.perf_counter() - t0) * 1e3,
            "violation": float((C_in @ U - d_in).max()),
            "eq_violation": float(np.abs(U[eq_idx]).max()) if len(eq_idx) else 0.0,
            "solver": self.solver_name,
        }
        self.last = {"U": U, "Bd": Bd, "contact": contact}
        return u0, U, info

    def _solve_qp_ineq(self, H, g, C, d):
        """min ½uᵀHu+gᵀu  s.t. C·u ≤ d  (등식 없음 — 스윙 변수는 이미 제거됨)."""
        try:
            u, *_ = quadprog.solve_qp(H, -g, -C.T, -d, 0)
            self.solver_name = "quadprog"
            return u
        except Exception:               # noqa: BLE001
            import scipy.sparse as sp
            import osqp
            prob = osqp.OSQP()
            prob.setup(P=sp.csc_matrix(H), q=g, A=sp.csc_matrix(C),
                       l=-np.inf * np.ones(len(d)), u=d,
                       verbose=False, eps_abs=1e-7, eps_rel=1e-7)
            res = prob.solve()
            self.solver_name = "osqp"
            return np.asarray(res.x)

    def _solve_qp_eq(self, H, g, eq_idx, C_in, d_in):
        """min ½uᵀHu+gᵀu  s.t. u[eq_idx]=0, C_in·u ≤ d_in."""
        n = H.shape[0]
        n_eq = len(eq_idx)
        E = np.zeros((n_eq, n))
        E[np.arange(n_eq), eq_idx] = 1.0
        try:
            # quadprog: Cᵀx ≥ b, 앞 meq 개는 등식.  C 는 (n, m)
            Cq = np.hstack([E.T, -C_in.T])
            bq = np.concatenate([np.zeros(n_eq), -d_in])
            u, *_ = quadprog.solve_qp(H, -g, Cq, bq, n_eq)
            self.solver_name = "quadprog"
            return u
        except Exception:               # noqa: BLE001
            import scipy.sparse as sp
            import osqp
            A = sp.vstack([sp.csc_matrix(E), sp.csc_matrix(C_in)]).tocsc()
            lo = np.concatenate([np.zeros(n_eq), -np.inf * np.ones(len(d_in))])
            hi = np.concatenate([np.zeros(n_eq), d_in])
            prob = osqp.OSQP()
            prob.setup(P=sp.csc_matrix(H), q=g, A=A, l=lo, u=hi,
                       verbose=False, eps_abs=1e-7, eps_rel=1e-7)
            res = prob.solve()
            self.solver_name = "osqp"
            return np.asarray(res.x)


# ---------------------------------------------------------------------------
# 토크 변환 (층1 마지막) — 이미 검증된 파이프라인, 6D wrench 로 확장
# ---------------------------------------------------------------------------
def wrench_to_tau(m, d, adof: np.ndarray, wrenches: np.ndarray) -> np.ndarray:
    """τ = qfrc_bias − Σ J_iᵀ W_i.   wrenches: (2,6) 발별 [F(3), m(3)] world.

    J_i 는 6×nv (jacp 위, jacr 아래). mj_forward 가 불린 상태를 가정.
    """
    tau_c = np.zeros(m.nv)
    for i, s in enumerate(FOOT_SITES):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        tau_c += jacp.T @ wrenches[i, :3] + jacr.T @ wrenches[i, 3:]
    return d.qfrc_bias[adof] - tau_c[adof]


def actuated_dofs(m) -> np.ndarray:
    return np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
