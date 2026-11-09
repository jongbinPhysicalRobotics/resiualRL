"""전체 KKT [P̂+σI, Âᵀ; Â, −diag(ρ⁻¹)] 의 배치 LDL (cuDSS) — Residual MPC 논문 (Alg. 1, cuDSS_fac / cuDSS_solve) 방식.

batch_mpc.BatchOSQP(linsys="cudss") 가 쓴다. OSQP 의 QDLDL 과 같은 꼴의 계 (x̃, ν) 를 풀고, z̃ = z − ρ⁻¹y + ρ⁻¹ν (qdldl_interface 와 같은 식).
희소 패턴 (고정): 위 삼각 CSC — 열 j < nv: P 위 삼각 전부 (명시적 0 포함, P 는 condensed 라 꽉 참),
                                열 nv + g: A 행 g 의 고정 패턴 원소 (cusadi_mit _LINE 패턴) + 대각 −1/ρ_g.
CSC 위 삼각 = CSR 아래 삼각 → cuDSS 에 MVIEW_LOWER 로 (CusADi base.py 와 같은 넘김).
행 순서 g = OSQP 순서 [부등식 N×24 | 등식 N×12] (batch_mpc.to_global).
"""
from __future__ import annotations

import numpy as np
import torch

from mpc import NU, N_INEQ, N_EQ

NC_STEP = N_INEQ + N_EQ


def kkt_pattern(N: int, a_mask):
    """→ (colptr, rows, src): 위 삼각 CSC 패턴과, 값 원천 벡터 [P̂+σI (nv²) | Â 칸 블록 (N·36·12) | −ρ⁻¹ (N·36)] 에서의 위치."""
    nv, nc = NU * N, NC_STEP * N
    mask = a_mask.to("cpu").numpy().astype(bool)                # (N, 36, 12)
    offA, offR = nv * nv, nv * nv + N * NC_STEP * NU
    rows, src = [], []                                           # CSC 위 삼각, 열 순서대로
    colptr = [0]
    for j in range(nv):                                          # P 위 삼각 (i ≤ j)
        rows += list(range(j + 1))
        src += [i * nv + j for i in range(j + 1)]
        colptr.append(len(rows))
    for g in range(nc):                                          # 제약 열: Âᵀ 의 g 열 = Â 의 g 행
        if g < N_INEQ * N:
            k, r = divmod(g, N_INEQ)
        else:
            k, r = divmod(g - N_INEQ * N, N_EQ)
            r += N_INEQ
        for cc in np.flatnonzero(mask[k, r]):
            rows.append(k * NU + int(cc))
            src.append(offA + (k * NC_STEP + r) * NU + int(cc))
        rows.append(nv + g)                                      # 대각 −1/ρ
        src.append(offR + k * NC_STEP + r)
        colptr.append(len(rows))
    return colptr, rows, src


class CudssKKT:
    def __init__(self, B: int, N: int, a_mask: torch.Tensor, device, dtype):
        from cusadi_lite import load_cudss
        cudss = load_cudss()
        self.B, self.N, self.nv, self.nc = B, N, NU * N, NC_STEP * N
        nv, nc = self.nv, self.nc
        colptr, rows, src = kkt_pattern(N, a_mask)
        self.nnz = len(rows)
        self.n = nv + nc
        self.src = torch.as_tensor(np.array(src, dtype=np.int64), device=device)
        i32 = lambda a: torch.as_tensor(np.array(a, dtype=np.int32), device=device)
        self.Ap = i32(colptr)[None].repeat(B, 1).contiguous()
        self.Ai = i32(rows)[None].repeat(B, 1).contiguous()
        self.Ax = torch.rand(B, self.nnz, dtype=dtype, device=device) + 1.0      # 분석 단계용 값 (패턴만 쓰임)
        self.b = torch.zeros(B, self.n, dtype=dtype, device=device)
        self.x = torch.zeros(B, self.n, dtype=dtype, device=device)
        ptr = lambda T: torch.as_tensor([int(T[i].data_ptr()) for i in range(B)], dtype=torch.int64, device=device)
        self._ptrs = (ptr(self.Ap), ptr(self.Ai), ptr(self.Ax), ptr(self.x), ptr(self.b))   # 수명 유지
        hostv = lambda v: torch.full((B,), int(v), dtype=torch.int32)
        self._host = (hostv(self.n), hostv(self.n), hostv(self.nnz), hostv(1))
        self.iface = (cudss.cudssInterface_double if dtype == torch.float64 else cudss.cudssInterface_float)(B)
        self.iface.loadPointers(*self._host, *self._ptrs)
        self.iface.createMatrices()
        self.iface.factorizeSymbolic()
        self.eye_sig = None

    def factor(self, Ps, As, rho_inv, sigma):
        """Ps (B,nv,nv) · As (B,N,36,12) · rho_inv (B,N,36) → 수치 분해 (반복 동안 재사용)."""
        B = self.B
        Pp = Ps + sigma * torch.eye(self.nv, dtype=Ps.dtype, device=Ps.device)[None]
        srcv = torch.cat([Pp.reshape(B, -1), As.reshape(B, -1), (-rho_inv).reshape(B, -1)], dim=1)
        self.Ax.copy_(srcv[:, self.src])
        self.iface.factorizeNumeric()

    def solve(self, bx, bz):
        """bx (B,nv) · bz (B,N,36) (칸 블록) → x̃ (B,nv), ν (B,N,36)."""
        from batch_mpc import to_global, from_global
        self.b[:, :self.nv].copy_(bx)
        self.b[:, self.nv:].copy_(to_global(bz))
        self.iface.solveLinearSystem()
        sol = self.x.clone()
        return sol[:, :self.nv], from_global(sol[:, self.nv:], self.N)


class DenseKKT(CudssKKT):
    """cuDSS 경로의 검증용 대역 (linsys="kktcpu"): 같은 패턴 · 같은 값 배치로 전체 KKT 를 만들어 dense LU 로 푼다 (CPU 가능, 느림)."""

    def __init__(self, B: int, N: int, a_mask: torch.Tensor, device, dtype):
        self.B, self.N, self.nv, self.nc = B, N, NU * N, NC_STEP * N
        colptr, rows, src = kkt_pattern(N, a_mask)
        self.n = self.nv + self.nc
        cols = np.repeat(np.arange(self.n), np.diff(colptr))
        self.r_idx = torch.as_tensor(np.array(rows), device=device)
        self.c_idx = torch.as_tensor(cols, device=device)
        self.src = torch.as_tensor(np.array(src, dtype=np.int64), device=device)
        self.dtype, self.device = dtype, device

    def factor(self, Ps, As, rho_inv, sigma):
        B = self.B
        Pp = Ps + sigma * torch.eye(self.nv, dtype=Ps.dtype, device=Ps.device)[None]
        vals = torch.cat([Pp.reshape(B, -1), As.reshape(B, -1), (-rho_inv).reshape(B, -1)], dim=1)[:, self.src]
        K = torch.zeros(B, self.n, self.n, dtype=self.dtype, device=self.device)
        K[:, self.r_idx, self.c_idx] = vals                              # 위 삼각
        K[:, self.c_idx, self.r_idx] = vals                              # 대칭
        self.LU = torch.linalg.lu_factor(K)

    def solve(self, bx, bz):
        from batch_mpc import to_global, from_global
        b = torch.cat([bx, to_global(bz)], dim=1)
        sol = torch.linalg.lu_solve(*self.LU, b[:, :, None]).squeeze(-1)
        return sol[:, :self.nv], from_global(sol[:, self.nv:], self.N)
