"""일괄 (batch) MPC — casadi_mit 의 A42 condensed QP + 고정 반복 OSQP 를 환경 B 개에 대해 한 번에 (torch, CPU / GPU).

목표 (Q&A 11/9): CPU 판 (mpc_casadi.CasadiCondensedMPC + OSQP 1.1.3 고정 25 회) 과 **같은 해**를 B 개 환경에 대해
GPU 에서 동시에. 알고리즘은 바꾸지 않는다 — 같은 QP, 같은 Ruiz 10 회, 같은 ρ 벡터, 같은 warm start, 같은 25 회.

구성
  SxBatch     칸별 기호식 (mpc_step · qk · rot) 을 (환경 × 칸) 일괄로 계산.
              backend "casadi" = ca.Function.map (CPU, 노트북 검증용) / "cusadi" = CusADi CUDA 커널 (실험실 GPU)
  BatchQP     condensed QP 조립 (lift · H · q · 제약 블록 · u · fref · msc) — mpc.py / mpc_casadi.py 와 같은 식, 같은 곱 순서
  BatchOSQP   OSQP 1.1.3 고정 반복 경로의 재현 (codegen_src/src/{scaling,auxil,osqp_api}.c 대조):
                update 마다 Ruiz (scaling 회, 비용 배율 = P 열 노름 평균 vs ‖q‖∞, 한계 1e-4 / 1e4)
                ρ 벡터 (등식 행 10³ρ̄, 부등식 ρ̄), σ, α 완화
                warm start: x = warm · D⁻¹ (새 배율), z = Âx, y = 지난 풀이의 (스케일된) y 그대로
              선형계 "chol" = 축약형 K = P̂ + σI + ÂᵀρÂ 배치 Cholesky (torch, CPU·GPU)
                     "cudss" = 전체 KKT [P̂+σI, Âᵀ; Â, −ρ⁻¹] 배치 LDL (cuDSS, GPU — Residual MPC 논문 방식)
                     "kktcpu" = cudss 와 같은 패턴 · 식을 dense LU 로 (검증용 대역, 느림)
  BatchMPC    위 셋을 묶은 것. from_cpu(mpc_cpu, ...) 로 CPU 객체의 설정을 그대로 읽는다 (값을 따로 적지 않는다).

배치 입력 (solve):
  x0 (B,13) · X_ref (B,N,13) · psi (B,N) · r (B,N,2,3) · stance (B,N,2) bool · min_scale (B,N,2) · foot_yaw (B,2) 또는 (B,N,2)
  mass (B,) · inertia (B,3,3) · walking (B,) bool
출력: U (B,N,12) 원래 단위 (CPU solve 의 info["U"] 와 같은 것) — u0 = U[:, 0].

지원 범위 (A42 가 쓰는 것): foot=line, fref (frefm 없음), msc, init warm / nominal / zero (dual keep), 고정 반복.
그 밖의 변형 (toeoff · htoe · hstk · pfmin · fmoff · hmpc · wdu · ydual shift · reference OSQP 경로) 은 켜져 있으면 예외.
"""
from __future__ import annotations

import numpy as np
import torch
import casadi as ca

from mpc import NX, NU, N_INEQ, N_EQ, LEFT_MAP, RIGHT_MAP
from mpc_casadi import build_step_function, build_qk_function

NC_STEP = N_INEQ + N_EQ                    # 36 행 / 칸 (부등식 24 + 등식 12)
OSQP_MIN_SCALING, OSQP_MAX_SCALING = 1e-4, 1e4
OSQP_RHO_MIN, OSQP_RHO_MAX = 1e-6, 1e6
OSQP_RHO_EQ_OVER_RHO_INEQ = 1e3
OSQP_INFTY = 1e30


# ---------------------------------------------------------------------------
# 칸별 기호식의 일괄 계산
# ---------------------------------------------------------------------------
def _renamed(fn: ca.Function, name: str) -> ca.Function:
    """같은 SX 식을 다른 이름으로 (CusADi 는 함수 이름이 커널 이름이라 겹치면 안 된다)."""
    sx = fn.sx_in()
    return ca.Function(name, sx, fn.call(sx))


class SxBatch:
    """ca.Function (입력 = 열 벡터, 출력 = densify 된 행렬) 을 M 개 일괄로.  __call__(*inputs (M, n_i)) → [(M, r_j, c_j)]."""

    def __init__(self, fn: ca.Function, backend: str, device, dtype, batch: int | None = None):
        self.fn, self.backend, self.device, self.dtype = fn, backend, device, dtype
        self.out_shape = [(fn.size1_out(j), fn.size2_out(j)) for j in range(fn.n_out())]
        for j in range(fn.n_out()):
            if fn.nnz_out(j) != fn.size1_out(j) * fn.size2_out(j):
                raise ValueError(f"{fn.name()}: 출력 {j} 가 dense 가 아님 (densify 필요)")
        self._maps = {}
        self._cusadi = None
        if backend == "cusadi":
            if batch is None:
                raise ValueError("cusadi backend 는 고정 batch (M) 가 필요")
            from cusadi_lite import parallelize
            prec = "double" if dtype == torch.float64 else "float"
            # dynamic_batching: 환경 수가 바뀌어도 다시 컴파일하지 않음 (메모리 배치는 같은 [변수][환경] — 곱셈 하나 더)
            self._cusadi = parallelize([fn], batch_size=batch, precision=prec, dynamic_batching=True)[fn.name()]
        elif backend != "casadi":
            raise ValueError(f"backend must be casadi / cusadi, got {backend!r}")

    def __call__(self, *inputs):
        M = inputs[0].shape[0]
        if self.backend == "casadi":
            fm = self._maps.get(M)
            if fm is None:
                fm = self._maps[M] = self.fn.map(M)
            args = [np.asarray(t.detach().to("cpu", torch.float64).numpy()).T for t in inputs]   # (n_i, M)
            outs = fm(*args)
            if not isinstance(outs, (list, tuple)):
                outs = [outs]
            res = []
            for (r, c), o in zip(self.out_shape, outs):
                a = np.array(o).reshape(r, M, c).transpose(1, 0, 2)                              # (M, r, c)
                res.append(torch.from_numpy(np.ascontiguousarray(a)).to(self.device, self.dtype))
            return res
        outs = self._cusadi.evaluate([t.to(self.device, self.dtype).contiguous() for t in inputs])
        # CusADi 출력 = (M, nnz) 의 열 우선 nonzeros → (M, c, r) → 전치. evaluate 는 출력 텐서를 재사용하므로 복사.
        return [o.reshape(M, c, r).transpose(1, 2).clone() for (r, c), o in zip(self.out_shape, outs)]


# ---------------------------------------------------------------------------
# condensed QP 조립
# ---------------------------------------------------------------------------
def lift_batch(A_d: torch.Tensor, B_d: torch.Tensor):
    """mpc.lift 의 배치판 (같은 곱 순서). A_d (B,N,13,13), B_d (B,N,13,12) → A_qp (B,13N,13), B_qp (B,13N,12N)."""
    Bsz, n = A_d.shape[:2]
    A_qp = torch.empty(Bsz, n, NX, NX, dtype=A_d.dtype, device=A_d.device)
    prefix = torch.eye(NX, dtype=A_d.dtype, device=A_d.device).expand(Bsz, NX, NX)
    for k in range(n):
        prefix = A_d[:, k] @ prefix
        A_qp[:, k] = prefix
    blk = torch.zeros(Bsz, n, n, NX, NU, dtype=A_d.dtype, device=A_d.device)   # blk[:, k, j] = B_qp 블록 (k, j)
    L = B_d
    for o in range(n):
        m = n - o
        js = torch.arange(m, device=A_d.device)
        blk[:, js + o, js] = L[:, :m]
        if o + 1 < n:
            L = A_d[:, o + 1:n] @ L[:, :m - 1]
    return A_qp.reshape(Bsz, NX * n, NX), blk.permute(0, 1, 3, 2, 4).reshape(Bsz, NX * n, NU * n)


class BatchQP:
    """CasadiCondensedMPC.solve 의 QP 부분 (formulate · build_cost · build_constraints · fref · msc) 을 배치로."""

    def __init__(self, cpu, batch: int, device, dtype, backend: str):
        self.N, self.nv = cpu.N, cpu.nv
        self.B, self.device, self.dtype = batch, device, dtype
        M = batch * cpu.N
        mk = lambda fn, name: SxBatch(_renamed(fn, name), backend, device, dtype, batch=M)
        self.f_step = mk(build_step_function(cpu.dt), "cmpc_step")
        self.f_qk = mk(build_qk_function(), "cmpc_qk")
        self.f_rot_stand = mk(cpu._rot_fn, "cmpc_rot_stand")
        self.f_rot_walk = mk(cpu._rot_fn_walk, "cmpc_rot_walk")
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=float), dtype=dtype, device=device)
        self.Q = {m: t(cpu.Q_diag[m]) for m in ("walking", "standing")}
        self.R = {m: t(np.tile(cpu.R_diag[m], cpu.N)) for m in ("walking", "standing")}
        self.fz_max, self.fz_min, self.g_abs = cpu.fz_max, cpu.fz_min, abs(cpu.gravity)
        self.fref_w = cpu.force_ref_w
        self.var_scale = cpu.var_scale
        self.Sv = t(np.tile(cpu._msc12, cpu.N))
        self.msc12 = t(cpu._msc12)
        mask = np.zeros(cpu.N * NC_STEP * NU)
        mask[cpu._A_gather] = 1.0                                   # OSQP 가 보는 A = 고정 패턴의 원소만
        self.A_mask = t(mask.reshape(cpu.N, NC_STEP, NU))
        self.eq_mask = torch.zeros(NC_STEP, dtype=torch.bool, device=device)
        self.eq_mask[N_INEQ:] = True
        self.LM = torch.as_tensor(LEFT_MAP, device=device)
        self.RM = torch.as_tensor(RIGHT_MAP, device=device)

    def build(self, x0, X_ref, psi, r, stance, min_scale, foot_yaw, mass, inertia, walking):
        """→ (P (B,nv,nv) 대칭, q (B,nv), A (B,N,36,12), l (B,N,36), u (B,N,36)). msc 가 켜져 있으면 스케일된 변수 Ū 기준."""
        B, n, nv, dt, dev = x0.shape[0], self.N, self.nv, self.dtype, self.device
        f = lambda a: torch.as_tensor(a, dtype=dt, device=dev)
        x0, X_ref, psi, r, min_scale, mass, inertia = map(f, (x0, X_ref, psi, r, min_scale, mass, inertia))
        stance = torch.as_tensor(stance, dtype=torch.bool, device=dev)
        walking = torch.as_tensor(walking, dtype=torch.bool, device=dev)
        foot_yaw = f(foot_yaw)
        if foot_yaw.dim() == 2:                                     # 다리당 yaw 하나 (reference) → 칸마다 같은 값
            foot_yaw = foot_yaw[:, None, :].expand(B, n, 2)
        foot_yaw = torch.where(torch.isfinite(foot_yaw), foot_yaw, torch.zeros_like(foot_yaw))   # resolved_foot_yaw
        min_scale = min_scale.clamp(0.0, 1.0)
        # 1) 칸별 A_d, B_d (기호식 mpc_step: 동역학 → 자동미분 → ZOH)
        I9 = inertia.reshape(B, 1, 9).expand(B, n, 9)               # 행 우선 (mpc_casadi.formulate 와 같은 순서)
        A_d, B_d = self.f_step(psi.reshape(-1, 1), r.reshape(B * n, 6),
                               mass.reshape(B, 1).expand(B, n).reshape(-1, 1), I9.reshape(-1, 9))
        A_qp, B_qp = lift_batch(A_d.reshape(B, n, NX, NX), B_d.reshape(B, n, NX, NU))
        # 2) 비용 (build_cost): Q_k = T_kᵀ diag(Q) T_k, ψ_k = X_ref[k, 2]
        Qd = torch.where(walking[:, None], self.Q["walking"][None], self.Q["standing"][None])   # (B,13)
        (Qk,) = self.f_qk(X_ref[:, :, 2].reshape(-1, 1), Qd[:, None, :].expand(B, n, NX).reshape(-1, NX))
        Qk = Qk.reshape(B, n, NX, NX)
        Xf = X_ref.reshape(B, NX * n)
        e = (A_qp @ x0[:, :, None]).squeeze(-1) - Xf
        Wb = (Qk @ B_qp.reshape(B, n, NX, nv)).reshape(B, NX * n, nv)
        we = (Qk @ e.reshape(B, n, NX, 1)).reshape(B, NX * n)
        BT = B_qp.transpose(1, 2)
        H = 2.0 * (BT @ Wb)
        Rt = torch.where(walking[:, None], self.R["walking"][None], self.R["standing"][None])
        H = H + torch.diag_embed(2.0 * Rt)
        H = 0.5 * (H + H.transpose(1, 2))
        q = 2.0 * (BT @ we[:, :, None]).squeeze(-1)
        # 3) 변형 fref (Q&A 9/29 Q5): w·(Fz − m|g|/n_디딤)² — 디딤 칸의 Fz 대각 · 선형항
        if self.fref_w > 0.0:
            n_st = stance.sum(-1).clamp(min=1).to(dt)                 # (B,N)
            fz_ref = mass[:, None] * self.g_abs / n_st
            add = torch.zeros(B, n, NU, dtype=dt, device=dev)
            add[:, :, 2] = stance[:, :, 0].to(dt)
            add[:, :, 5] = stance[:, :, 1].to(dt)
            add = add.reshape(B, nv)
            H = H + torch.diag_embed(2.0 * self.fref_w * add)
            q = q - 2.0 * self.fref_w * add * fz_ref.repeat_interleave(NU, dim=1)
        # 4) 제약 블록 (build_constraints, foot=line)
        yL, yR = foot_yaw[:, :, 0], foot_yaw[:, :, 1]
        rot = lambda fn_, y: fn_(y.reshape(-1, 1))[0].reshape(B, n, 12, 6)
        wsel = walking[:, None, None, None]
        CWL = torch.where(wsel, rot(self.f_rot_walk, yL), rot(self.f_rot_stand, yL))
        CWR = torch.where(wsel, rot(self.f_rot_walk, yR), rot(self.f_rot_stand, yR))
        stL, stR = stance[:, :, 0], stance[:, :, 1]
        blk = torch.zeros(B, n, NC_STEP, NU, dtype=dt, device=dev)
        blk[:, :, 0:12, self.LM] = CWL * stL[:, :, None, None].to(dt)
        blk[:, :, 12:24, self.RM] = CWR * stR[:, :, None, None].to(dt)
        swL, swR = (~stL).to(dt), (~stR).to(dt)
        for i in (0, 1, 2, 6, 7, 8):                                # 스윙 발 변수 = 0
            blk[:, :, N_INEQ + i, i] = swL
        for i in (3, 4, 5, 9, 10, 11):
            blk[:, :, N_INEQ + i, i] = swR
        cL, sL_, cR, sR_ = torch.cos(yL), torch.sin(yL), torch.cos(yR), torch.sin(yR)
        blk[:, :, N_INEQ + 6, 6] = torch.where(stL, cL, blk[:, :, N_INEQ + 6, 6])   # 디딤 발 mx^F = 0 (선 접촉)
        blk[:, :, N_INEQ + 6, 7] = torch.where(stL, sL_, blk[:, :, N_INEQ + 6, 7])
        blk[:, :, N_INEQ + 9, 9] = torch.where(stR, cR, blk[:, :, N_INEQ + 9, 9])
        blk[:, :, N_INEQ + 9, 10] = torch.where(stR, sR_, blk[:, :, N_INEQ + 9, 10])
        blk = blk * self.A_mask[None]
        ub = torch.zeros(B, n, NC_STEP, dtype=dt, device=dev)
        zero = torch.zeros((), dtype=dt, device=dev)
        ub[:, :, 4] = torch.where(stL, torch.full((), self.fz_max, dtype=dt, device=dev), zero)
        ub[:, :, 5] = torch.where(stL, -min_scale[:, :, 0] * self.fz_min, zero)
        ub[:, :, 16] = torch.where(stR, torch.full((), self.fz_max, dtype=dt, device=dev), zero)
        ub[:, :, 17] = torch.where(stR, -min_scale[:, :, 1] * self.fz_min, zero)
        lb = torch.where(self.eq_mask, zero, torch.full((), -OSQP_INFTY, dtype=dt, device=dev)).expand(B, n, NC_STEP)
        # 5) 변형 msc (Q&A 9/30 Q16): U = S·Ū
        if self.var_scale:
            H = H * (self.Sv[:, None] * self.Sv[None, :])[None]
            q = q * self.Sv[None]
            blk = blk * self.msc12[None, None, None, :]
        return H, q, blk, lb.contiguous(), ub


# ---------------------------------------------------------------------------
# OSQP 고정 반복 경로의 배치 재현
# ---------------------------------------------------------------------------
def _limit(v):
    """limit_scaling_vector / _scalar (scaling.c): < 1e-4 → 1, > 1e4 → 1e4."""
    v = torch.where(v < OSQP_MIN_SCALING, torch.ones_like(v), v)
    return torch.where(v > OSQP_MAX_SCALING, torch.full_like(v, OSQP_MAX_SCALING), v)


def ruiz(P, q, A, scaling: int):
    """scale_data (scaling.c) — 매 update 마다 원래 데이터에서 다시 계산. A 는 칸 블록 (B,N,36,12).
    → (P̂, q̂, Â, D (B,nv), E (B,N,36), c (B,))."""
    B, n = A.shape[:2]
    D = torch.ones(B, n * NU, dtype=P.dtype, device=P.device)
    E = torch.ones(B, n, NC_STEP, dtype=P.dtype, device=P.device)
    c = torch.ones(B, dtype=P.dtype, device=P.device)
    # ⚠ OSQP 1.1.3 의 P 열 노름은 저장된 위 삼각 (i ≤ j) 만 본다 — OSQPMatrix_col_norm_inf → csc_col_norm_inf (대칭판 아님,
    #   대칭판은 row_norm_inf 에만 있다). 대칭 열 전체로 재면 열 최댓값이 대각 아래에 있는 칸 (착지 순간 등) 에서 배율이 달라진다.
    up = torch.triu(torch.ones(P.shape[1], P.shape[2], dtype=torch.bool, device=P.device))
    colnorm_P = lambda M: torch.where(up, M.abs(), torch.zeros((), dtype=M.dtype, device=M.device)).amax(dim=1)
    for _ in range(scaling):
        Dt = torch.maximum(colnorm_P(P), A.abs().amax(dim=2).reshape(B, n * NU))          # KKT 열 노름
        Et = A.abs().amax(dim=3)                                                           # A 행 노름
        Dt = 1.0 / torch.sqrt(_limit(Dt))
        Et = 1.0 / torch.sqrt(_limit(Et))
        P = (Dt[:, :, None] * P) * Dt[:, None, :]
        A = (Et[..., None] * A) * Dt.reshape(B, n, 1, NU)
        q = q * Dt
        D = D * Dt
        E = E * Et
        c_t = colnorm_P(P).sum(dim=1) / P.shape[1]                  # P 열 노름 (위 삼각) 의 평균
        nq = _limit(q.abs().amax(dim=1))
        c_t = 1.0 / _limit(torch.maximum(c_t, nq))
        P = P * c_t[:, None, None]
        q = q * c_t[:, None]
        c = c * c_t
    return P, q, A, D, E, c


def to_global(v):
    """칸 블록 (B,N,36) → OSQP 행 순서 [부등식 N×24 | 등식 N×12] (B, 36N)."""
    B = v.shape[0]
    return torch.cat([v[:, :, :N_INEQ].reshape(B, -1), v[:, :, N_INEQ:].reshape(B, -1)], dim=1)


def from_global(g, n):
    B = g.shape[0]
    return torch.cat([g[:, :N_INEQ * n].reshape(B, n, N_INEQ), g[:, N_INEQ * n:].reshape(B, n, N_EQ)], dim=2)


class BatchOSQP:
    """mpc.ConvexMPC._solve_fixed (OSQP 1.1.3, 고정 반복, workspace 유지) 의 배치판. 환경마다 상태 (y, warm, has_prev) 를 따로 갖는다."""

    def __init__(self, B, N, *, sigma, alpha, scaling, rho_walk, rho_stand, iters_walk, iters_stand, init_mode,
                 warm_shift, shifted, gravity, device, dtype, linsys="chol", a_mask=None):
        self.B, self.N, self.nv = B, N, NU * N
        self.sigma, self.alpha, self.scaling = float(sigma), float(alpha), int(scaling)
        clamp = lambda r_: min(max(float(r_), OSQP_RHO_MIN), OSQP_RHO_MAX)
        self.rho_walk, self.rho_stand = clamp(rho_walk), clamp(rho_stand)
        self.iters_walk, self.iters_stand = int(iters_walk), int(iters_stand)
        if init_mode not in ("warm", "nominal", "zero"):
            raise NotImplementedError(f"init={init_mode!r} (batch 는 warm / nominal / zero)")
        self.init_mode, self.warm_shift, self.shifted = init_mode, float(warm_shift), bool(shifted)
        self.g_abs = abs(float(gravity))
        self.device, self.dtype, self.linsys = device, dtype, linsys
        self.eq_mask = torch.zeros(NC_STEP, dtype=torch.bool, device=device)
        self.eq_mask[N_INEQ:] = True
        self._kkt = None
        if linsys in ("cudss", "kktcpu"):
            from cudss_kkt import CudssKKT, DenseKKT
            if a_mask is None:
                raise ValueError(f"linsys={linsys} 는 A 고정 패턴 (a_mask, BatchQP.A_mask) 이 필요")
            self._kkt = (CudssKKT if linsys == "cudss" else DenseKKT)(B, N, a_mask, device, dtype)
        elif linsys != "chol":
            raise ValueError(f"linsys must be chol / cudss / kktcpu, got {linsys!r}")
        self.reset()

    def reset(self, env_ids=None):
        """환경 상태를 버린다 (새 OSQP workspace 와 같음: x = y = z = 0, 이전 해 없음)."""
        z = lambda *s: torch.zeros(*s, dtype=self.dtype, device=self.device)
        if env_ids is None:
            self.y = z(self.B, self.N, NC_STEP)                     # 스케일된 쌍대 (OSQP workspace 의 y)
            self.warm = z(self.B, self.nv)                          # 다음 풀이의 x 출발점 (Ū 단위, 원래 배율)
            self.last = z(self.B, self.nv)
            self.has_prev = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        else:
            self.y[env_ids] = 0.0
            self.warm[env_ids] = 0.0
            self.last[env_ids] = 0.0
            self.has_prev[env_ids] = False

    def _nominal(self, stance, mass):
        X = torch.zeros(self.B, self.N, NU, dtype=self.dtype, device=self.device)
        n_st = stance.sum(-1)
        fz = torch.where(n_st > 0, mass[:, None] * self.g_abs / n_st.clamp(min=1).to(self.dtype), torch.zeros_like(X[..., 0]))
        X[:, :, 2] = torch.where(stance[:, :, 0], fz, torch.zeros_like(fz))
        X[:, :, 5] = torch.where(stance[:, :, 1], fz, torch.zeros_like(fz))
        return X.reshape(self.B, self.nv)

    def solve(self, P, q, A, l, u, walking, stance=None, mass=None):
        """→ X (B,nv) 원래 배율 (OSQP res.x, 즉 Ū 단위). 상태 (y, warm) 갱신."""
        B, n, dt = self.B, self.N, self.dtype
        walking = torch.as_tensor(walking, dtype=torch.bool, device=self.device)
        Ps, qs, As, D, E, c = ruiz(P, q, A, self.scaling)
        ls, us = E * l, E * u
        rho_bar = torch.where(walking, torch.full((B,), self.rho_walk, dtype=dt, device=self.device),
                              torch.full((B,), self.rho_stand, dtype=dt, device=self.device))
        rho = torch.where(self.eq_mask, OSQP_RHO_EQ_OVER_RHO_INEQ * rho_bar[:, None, None], rho_bar[:, None, None])
        rho = rho.expand(B, n, NC_STEP)
        rho_inv = 1.0 / rho
        # 출발점 (osqp_warm_start): x = x_orig · D⁻¹, z = Â x, y = 스케일된 값
        if self.init_mode == "warm":
            x_orig = torch.where(self.has_prev[:, None], self.warm, torch.zeros_like(self.warm))
            y = torch.where(self.has_prev[:, None, None], self.y, torch.zeros_like(self.y))
        else:
            x_orig = self._nominal(torch.as_tensor(stance, dtype=torch.bool, device=self.device),
                                   torch.as_tensor(mass, dtype=dt, device=self.device)) \
                if self.init_mode == "nominal" else torch.zeros_like(self.warm)
            y = torch.zeros_like(self.y)
        x = x_orig * (1.0 / D)
        Ax = lambda v: (As @ v.reshape(B, n, NU, 1)).squeeze(-1)                   # (B,N,36)
        ATy = lambda w: (As.transpose(2, 3) @ w[..., None]).squeeze(-1).reshape(B, self.nv)
        z = Ax(x)
        # 분해 (setup / update 마다 1 회)
        if self.linsys == "chol":
            K = Ps + self.sigma * torch.eye(self.nv, dtype=dt, device=self.device)[None]
            ATrA = As.transpose(2, 3) @ (rho[..., None] * As)                       # (B,N,12,12) 블록 대각
            idx = torch.arange(n, device=self.device)
            K4 = K.reshape(B, n, NU, n, NU)
            K4[:, idx, :, idx, :] += ATrA.permute(1, 0, 2, 3)
            L = torch.linalg.cholesky(K4.reshape(B, self.nv, self.nv))
        else:
            self._kkt.factor(Ps, As, rho_inv, self.sigma)
        iters = torch.where(walking, torch.full((B,), self.iters_walk, device=self.device),
                            torch.full((B,), self.iters_stand, device=self.device))
        a = self.alpha
        for it in range(int(iters.max())):
            act = (it < iters)
            if self.linsys == "chol":
                rhs = self.sigma * x - qs + ATy(rho * z - y)
                xt = torch.cholesky_solve(rhs[:, :, None], L).squeeze(-1)
                zt = Ax(xt)                                         # z̃ = Â x̃ (KKT 둘째 줄과 같은 값)
            else:
                xt, nu = self._kkt.solve(self.sigma * x - qs, z - rho_inv * y)
                zt = z - rho_inv * y + rho_inv * nu                 # qdldl_interface 의 z̃ 계산
            x_new = a * xt + (1.0 - a) * x
            zr = a * zt + (1.0 - a) * z
            z_new = torch.minimum(torch.maximum(zr + rho_inv * y, ls), us)        # update_z: [l, u] 투영
            y_new = y + rho * (zr - z_new)                                         # update_y
            x = torch.where(act[:, None], x_new, x)
            z = torch.where(act[:, None, None], z_new, z)
            y = torch.where(act[:, None, None], y_new, y)
        X = D * x                                                   # unscale_solution
        ok = torch.isfinite(X).all(dim=1)
        # 상태 갱신 (_solve_fixed: _last_solution · _update_warm_start · has_prev). 비유한 환경은 상태를 버린다.
        self.last = torch.where(ok[:, None], X, self.last)
        self.y = torch.where(ok[:, None, None], y, torch.zeros_like(y))
        if self.shifted:
            w = torch.cat([self.last[:, NU:], self.last[:, -NU:]], dim=1)
            if self.warm_shift != 1.0:
                w = (1.0 - self.warm_shift) * self.last + self.warm_shift * w
        else:
            w = self.last.clone()
        self.warm = w
        self.has_prev = ok.clone()
        self.info = dict(D=D, E=E, c=c, y_scaled=y, ok=ok)
        return X, ok


class BatchMPC:
    """BatchQP + BatchOSQP.  from_cpu(CasadiCondensedMPC, B, ...) 로 만든다."""

    def __init__(self, qp: BatchQP, osqp: BatchOSQP, cpu):
        self.qp, self.osqp = qp, osqp
        self.N, self.nv, self.B = qp.N, qp.nv, qp.B
        self.var_scale, self.Sv = qp.var_scale, qp.Sv
        self.fallback_g = abs(cpu.gravity)

    @staticmethod
    def check_supported(cpu):
        bad = []
        if cpu.foot_model != "line":
            bad.append(f"foot_model={cpu.foot_model}")
        if cpu.fixed_iter <= 0:
            bad.append("fixed_iter=0 (reference OSQP 경로)")
        for name, v in (("toe_off_frac", cpu.toe_off_frac), ("toe_sense_mode", cpu.toe_sense_mode),
                        ("heel_off_frac", cpu.heel_off_frac), ("heel_strike_s1", cpu.heel_strike_s1),
                        ("point_fmin", cpu.point_fmin), ("fric_offset", cpu.fric_offset),
                        ("hold_w", cpu.hold_w), ("du_w", cpu.du_w)):
            if float(v) != 0.0:
                bad.append(f"{name}={v}")
        if cpu.force_ref_total:
            bad.append("frefm")
        if cpu.dual_init != "keep":
            bad.append(f"dual_init={cpu.dual_init}")
        if cpu.init_mode == "hold":
            bad.append("init=hold")
        if bad:
            raise NotImplementedError("BatchMPC 가 지원하지 않는 변형: " + ", ".join(bad))

    @classmethod
    def from_cpu(cls, cpu, B: int, device="cpu", dtype=torch.float64, backend="casadi", linsys="chol"):
        cls.check_supported(cpu)
        s = cpu.settings
        qp = BatchQP(cpu, B, device, dtype, backend)
        osqp = BatchOSQP(B, cpu.N, sigma=s["sigma"], alpha=s["alpha"], scaling=s["scaling"],
                         rho_walk=s["rho"], rho_stand=(cpu.rho_stand or s["rho"]),
                         iters_walk=cpu.fixed_iter, iters_stand=(cpu.fixed_iter_stand or cpu.fixed_iter),
                         init_mode=cpu.init_mode, warm_shift=cpu.warm_shift, shifted=cpu.use_shifted_warm_start,
                         gravity=cpu.gravity, device=device, dtype=dtype, linsys=linsys, a_mask=qp.A_mask)
        return cls(qp, osqp, cpu)

    def solve(self, x0, X_ref, psi, r, stance, min_scale, foot_yaw, mass, inertia, walking):
        """→ U (B,N,12) 원래 단위, ok (B,) — ok 가 False 인 환경은 fallback (디딤 발 Fz = m|g|/n, 첫 칸 기준) 을 넣는다."""
        P, q, A, l, u = self.qp.build(x0, X_ref, psi, r, stance, min_scale, foot_yaw, mass, inertia, walking)
        st = torch.as_tensor(stance, dtype=torch.bool, device=self.qp.device)
        m = torch.as_tensor(mass, dtype=self.qp.dtype, device=self.qp.device)
        X, ok = self.osqp.solve(P, q, A, l, u, walking, stance=st, mass=m)
        U = X * self.Sv[None] if self.var_scale else X
        if not bool(ok.all()):
            fb = torch.zeros(self.B, NU, dtype=U.dtype, device=U.device)
            n0 = st[:, 0].sum(-1).clamp(min=1).to(U.dtype)
            fz = m * self.fallback_g / n0
            fb[:, 2] = torch.where(st[:, 0, 0], fz, torch.zeros_like(fz))
            fb[:, 5] = torch.where(st[:, 0, 1], fz, torch.zeros_like(fz))
            U = torch.where(ok[:, None], U, fb.repeat(1, self.N))
        return U.reshape(self.B, self.N, NU), ok
