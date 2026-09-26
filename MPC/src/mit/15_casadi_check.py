"""CasADi 형식 MPC (mpc_casadi.py) 검증 — 이식 (mpc.py) 과 같은 입력에서 행렬·해 차이 (Q&A 9/26).

shadow  : 이식 컨트롤러로 걷게 하면서 매 MPC 풀이마다 같은 입력을 CasADi 판에도 넣어
          A_d/B_d (자동미분 vs 손 채움), A_qp/B_qp, H, q, 제약 blk/l/u, u0 (각자 OSQP workspace) 의
          최대 차이를 모은다. 디딤 발 |mx^F| (foot_model="ref" 가 요청하는 roll 모멘트) 도 기록.
closed  : controller.ConvexMPC 를 CasADi 판으로 바꿔치기해서 120 s 헤드리스 (10_walk.headless 그대로).
          foot_model ref ↔ line A/B 용.

실행 (ROOT 에서):
    .venv/bin/python MPC/src/mit/15_casadi_check.py shadow --vx 0.6 --seconds 10
    .venv/bin/python MPC/src/mit/15_casadi_check.py closed --foot ref --vx 0.6 --seconds 120 [--refosqp]
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import importlib

walk = importlib.import_module("10_walk")
import controller as ctrl_mod
import mpc as mpc_mod
import mpc_casadi

REF_OSQP = {"adaptive_rho_interval": 50, "check_dualgap": 1}   # reference 실효 설정 (Q&A 9/25 Q11)


# ---------------------------------------------------------------------------
class ShadowMPC(mpc_mod.ConvexMPC):
    """이식 그대로 동작하면서, 매 solve 마다 CasADi 판을 같은 입력으로 병행 평가해 차이를 모은다."""

    def __init__(self, cfg, **kw):
        super().__init__(cfg, **kw)
        self.twin = mpc_casadi.CasadiCondensedMPC(cfg, **kw)
        self.diffs = {k: 0.0 for k in ("A_d", "B_d", "A_qp", "B_qp", "H", "q", "blk", "u_bound", "u0")}
        self.rel_H = 0.0
        self.n_cmp = 0
        self.mx_stance = []            # 이식 u0 의 디딤 발 roll 모멘트 |mx^F| (foot_model="ref" 낭비 측정)
        self.t_port = self.t_twin = 0.0

    def solve(self, x0, ref, steps, foot_yaw, reduced, mode, active=None):
        mode_s = mpc_mod._mode_str(mode)
        x0a = np.asarray(x0, dtype=float)
        X_ref_flat = np.asarray(ref.X_ref, dtype=float).reshape(-1)
        try:
            t0 = time.perf_counter()
            A_qp, B_qp, A_d, B_d = super().formulate(ref, reduced)
            H, q = super().build_cost(A_qp, B_qp, x0a, X_ref_flat, mode_s)
            blk, l, u, sig, stance = super().build_constraints(steps, foot_yaw)
            self.t_port += time.perf_counter() - t0
            t0 = time.perf_counter()
            A_qp2, B_qp2, A_d2, B_d2 = self.twin.formulate(ref, reduced)
            H2, q2 = self.twin.build_cost(A_qp2, B_qp2, x0a, X_ref_flat, mode_s)
            blk2, l2, u2, sig2, _ = self.twin.build_constraints(steps, foot_yaw)
            self.t_twin += time.perf_counter() - t0
            d = self.diffs
            d["A_d"] = max(d["A_d"], float(np.max(np.abs(A_d - A_d2))))
            d["B_d"] = max(d["B_d"], float(np.max(np.abs(B_d - B_d2))))
            d["A_qp"] = max(d["A_qp"], float(np.max(np.abs(A_qp - A_qp2))))
            d["B_qp"] = max(d["B_qp"], float(np.max(np.abs(B_qp - B_qp2))))
            d["H"] = max(d["H"], float(np.max(np.abs(H - H2))))
            self.rel_H = max(self.rel_H, float(np.max(np.abs(H - H2)) / max(np.max(np.abs(H)), 1e-30)))
            d["q"] = max(d["q"], float(np.max(np.abs(q - q2))))
            d["blk"] = max(d["blk"], float(np.max(np.abs(blk - blk2))))
            d["u_bound"] = max(d["u_bound"], float(np.max(np.abs(u - u2))))
            assert sig == sig2
            self.n_cmp += 1
        except Exception as exc:                       # 비교 실패는 걷기를 막지 않는다
            print(f"[shadow] compare failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        u0_twin, _ = self.twin.solve(x0, ref, steps, foot_yaw, reduced, mode, active=active)
        u0, info = super().solve(x0, ref, steps, foot_yaw, reduced, mode, active=active)
        self.diffs["u0"] = max(self.diffs["u0"], float(np.max(np.abs(u0 - u0_twin))))
        if not info.get("fallback"):
            st0 = np.asarray(steps[0].stance, dtype=bool)
            for leg, sl in ((0, slice(6, 9)), (1, slice(9, 12))):
                if st0[leg]:
                    yaw = mpc_mod.resolved_foot_yaw(foot_yaw[leg])
                    c, s = np.cos(yaw), np.sin(yaw)
                    self.mx_stance.append(abs(c * u0[sl][0] + s * u0[sl][1]))   # x̂_F · M_W
        return u0, info


def run_shadow(a):
    ctrl_mod.ConvexMPC = ShadowMPC
    kw = dict(solver="osqp", armature=True, profile="ramp", markers=False, stand=False,
              friction_scale=1.0, fsm_double_update=True, fixes=("yawref",),
              osqp_settings=REF_OSQP if a.refosqp else None)
    cfg, model, ctl, walk_t = walk._setup(a.vx, 0.0, 0.0, kw["solver"], True, "ramp", False, False,
                                          1.0, True, False, osqp_settings=kw["osqp_settings"], fixes=("yawref",))
    import mujoco
    m, d = model.m, model.d
    n = int(round((walk_t + a.seconds) / m.opt.timestep))
    for k in range(n):
        tau_leg, tau_arm = ctl.tick()
        model.write_torque(tau_leg, tau_arm)
        mujoco.mj_step(m, d)
        if walk._fell(d):
            print(f"넘어짐 t = {d.time:.2f} s")
            break
    sh: ShadowMPC = ctl.mpc
    print(f"\nshadow — vx {a.vx}, sim {d.time:.1f} s, MPC 풀이 {sh.stats['solves']} 회 (비교 {sh.n_cmp} 회), "
          f"osqp {'ref-eff' if a.refosqp else 'port'}")
    print(f"  {'항목':8s} 최대 |차이|")
    for k, v in sh.diffs.items():
        print(f"  {k:8s} {v:.3e}")
    print(f"  H 상대     {sh.rel_H:.3e}")
    mx = np.array(sh.mx_stance)
    if len(mx):
        print(f"  이식(ref 모델) u0 디딤 발 |mx^F|: 평균 {mx.mean():.3f}  p95 {np.percentile(mx, 95):.3f}  "
              f"최대 {mx.max():.3f} N·m   (line 모델이면 0 이어야 할 양)")
    print(f"  공식 조립 시간: 이식 {1e3 * sh.t_port / max(sh.n_cmp, 1):.2f} ms/solve, "
          f"CasADi {1e3 * sh.t_twin / max(sh.n_cmp, 1):.2f} ms/solve")


def run_closed(a):
    fm = a.foot

    def factory(cfg, **kw):
        return mpc_casadi.CasadiCondensedMPC(cfg, foot_model=fm, **kw)

    ctrl_mod.ConvexMPC = factory
    r = walk.headless(a.vx, 0.0, 0.0, seconds=a.seconds, profile="ramp", fixes=("yawref",),
                      osqp_settings=REF_OSQP if a.refosqp else None, verbose=False)
    print(f"\nclosed — CasADi foot_model={fm}, vx {a.vx}, osqp {'ref-eff' if a.refosqp else 'port'}")
    walk.print_summary(r)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shadow")
    s.add_argument("--vx", type=float, default=0.6)
    s.add_argument("--seconds", type=float, default=10.0)
    s.add_argument("--refosqp", action="store_true")
    s = sub.add_parser("closed")
    s.add_argument("--foot", choices=["ref", "line"], default="ref")
    s.add_argument("--vx", type=float, default=0.6)
    s.add_argument("--seconds", type=float, default=120.0)
    s.add_argument("--refosqp", action="store_true")
    a = ap.parse_args()
    if a.cmd == "shadow":
        run_shadow(a)
    else:
        run_closed(a)


if __name__ == "__main__":
    main()
