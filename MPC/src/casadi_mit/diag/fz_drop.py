"""한 발 지지 중 MPC 수직력이 빠지는 원인 (Q&A 10/6 Q36). 시뮬은 그대로 두고, 매 MPC 풀이 직후 같은 QP 를 다시 만든다:
  - 정확해 (quadprog) 첫 칸 — 적용된 ADMM 25 회 해와 비교
  - 상태 오차를 하나씩 지운 정확해 (x0 의 그 성분을 X_ref[0] 값으로) → 그 오차가 첫 칸 Fz 를 얼마나 바꾸나
  - 첫 칸 pitch 토크 분해 (MPC 가 보는 지렛대 ref.r[0]): Fz 지렛대 · Fx 지렛대 · 발 모멘트 My
디딤 경과 (접촉표 위상) 칸별로 한 발 지지 (다른 발 디딤 아님) 인 풀이만 모은다. 옆 성분은 디딤 발 쪽을 + 로 맞춰 평균.
사용: fz_drop.py vx vy wz seconds variant
"""
import sys, importlib.util
import numpy as np
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent          # casadi_mit
sys.path.insert(0, str(_ROOT))
import mujoco
import quadprog
spec = importlib.util.spec_from_file_location("walk", str(_ROOT / "10_walk.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
from mpc import NX, NU, N_INEQ, N_EQ

# 상태 [roll, pitch, yaw, x, y, z, ωx, ωy, ωz, vx, vy, vz, g]
ABL = {"pitch": [1], "ωy": [7], "z": [5], "vz": [11], "v 앞": "fwd", "v 옆": "lat", "roll·ωx": [0, 6], "yaw·ωz": [2, 8], "xy": [3, 4]}


def exact(mpc, x0, ref, steps, foot_yaw, reduced, mode):
    A_qp, B_qp, _, _ = mpc.formulate(ref, reduced)
    X_ref_flat = np.asarray(ref.X_ref, dtype=float).reshape(-1)
    H, q = mpc.build_cost(A_qp, B_qp, x0, X_ref_flat, mode)
    blk, l, u, sig, stance = mpc.build_constraints(steps, foot_yaw)
    if mpc.force_ref_w > 0.0:
        st = np.asarray(stance, dtype=bool).reshape(mpc.N, 2)
        n_st = np.maximum(st.sum(1), 1)
        fz_ref = float(reduced.mass) * abs(mpc.gravity) / n_st
        H = H.copy(); q = q.copy()
        for k in range(mpc.N):
            for leg, iz in ((0, 2), (1, 5)):
                if st[k, leg]:
                    i = k * NU + iz
                    H[i, i] += 2.0 * mpc.force_ref_w
                    q[i] -= 2.0 * mpc.force_ref_w * fz_ref[k]
    n = mpc.N; nv = mpc.nv
    ub = u[:N_INEQ * n].reshape(n, N_INEQ)
    eq, ineq, b_in = [], [], []
    for k in range(n):
        for r in range(N_INEQ):
            row = blk[k, r]
            if np.any(row != 0.0):
                f = np.zeros(nv); f[NU * k:NU * (k + 1)] = row; ineq.append(-f); b_in.append(-ub[k, r])
        for r in range(N_EQ):
            row = blk[k, N_INEQ + r]
            if np.any(row != 0.0):
                f = np.zeros(nv); f[NU * k:NU * (k + 1)] = row; eq.append(f)
    C = np.array(eq + ineq).T
    b = np.concatenate([np.zeros(len(eq)), np.array(b_in)])
    U = quadprog.solve_qp(H, -q, C, b, meq=len(eq))[0]
    return U[:NU]


def run(vx, vy, wz, seconds, var):
    kw = W.parse_variant(var); prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(vx, vy, wz, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
    m, d = model.m, model.d
    rows = []
    box = {}

    def hooked(x0, ref, steps, foot_yaw, reduced, mode, active=None, mz_ref=None):
        mpc = box["mpc"]
        u0, info = box["orig"](x0, ref, steps, foot_yaw, reduced, mode, active=active, mz_ref=mz_ref)
        t = ctl.state.t
        if not (t > walk_t + 3.0 and str(mode).lower().endswith("walking") and active is not None):
            return u0, info
        act = np.asarray(active, dtype=bool)
        if act.sum() != 1:                                   # 한 발 지지만
            return u0, info
        leg = int(np.flatnonzero(act)[0])
        ph = ctl.gait.phase(leg, t, ctl.clock) * ctl.clock.cycle / ctl.gait.stance
        x0 = np.asarray(x0, dtype=float); xr = np.asarray(ref.X_ref, dtype=float)[0]
        try:
            ue = exact(mpc, x0, ref, steps, foot_yaw, reduced, mode)
            dfz = []
            c_, s_ = np.cos(xr[2]), np.sin(xr[2]); hh = np.array([c_, s_]); yy = np.array([-s_, c_])
            for idx in ABL.values():
                xm = x0.copy()
                if isinstance(idx, str):                           # 진행 방향 앞 / 옆 속도 성분만 참조로
                    e = (hh if idx == "fwd" else yy)
                    xm[9:11] = x0[9:11] + e * (xr[9:11] - x0[9:11]).dot(e)
                else:
                    xm[idx] = xr[idx]
                dfz.append(exact(mpc, xm, ref, steps, foot_yaw, reduced, mode)[2 + 3 * leg] - ue[2 + 3 * leg])
            xm = x0.copy(); xm[:12] = xr[:12]
            fz_noerr = exact(mpc, xm, ref, steps, foot_yaw, reduced, mode)[2 + 3 * leg]
        except Exception:
            return u0, info
        psi = xr[2]; c, s = np.cos(psi), np.sin(psi)
        h = np.array([c, s, 0.0]); yv = np.array([-s, c, 0.0])     # 진행 방향 · 그 y 축 (pitch 축)
        r = np.asarray(ref.r, dtype=float)[0, leg]                  # MPC 가 보는 몸통 → 발
        F = ue[3 * leg:3 * leg + 3]; M = ue[6 + 3 * leg:9 + 3 * leg]
        rh = r.dot(h); rz = r[2]
        tau_fz = -rh * F[2]                                          # (r × F)·y 의 Fz 몫: −r_h·Fz (발이 뒤면 +, 앞으로 숙임)
        tau_fx = rz * F.dot(h)                                       # Fx 몫: r_z·F_h (발이 아래, 미는 힘 + 면 −, 뒤로 젖힘)
        tau_m = M.dot(yv)
        cop = -tau_m / max(F[2], 1e-6)                               # 진행 방향 CoP [m] (+ 발끝)
        sg = 1.0 if leg == 0 else -1.0                               # 옆 성분은 디딤 발 쪽을 + 로 (왼발 +y, 오른발 −y)
        ry = sg * r.dot(yv); fy = sg * F.dot(yv)
        roll_fz = ry * F[2]                                          # (r × F)·h 의 Fz 몫: r_y·Fz (디딤 발 쪽 부호)
        roll_fy = -rz * fy                                           # Fy 몫: −r_z·F_y
        rows.append([ph, u0[2 + 3 * leg], ue[2 + 3 * leg], fz_noerr, *dfz,
                     x0[1] - xr[1], x0[7] - xr[7], x0[5] - xr[5], x0[11] - xr[11],
                     x0[9:11].dot(h[:2]) - xr[9:11].dot(h[:2]), sg * (x0[9:11].dot(yv[:2]) - xr[9:11].dot(yv[:2])),
                     tau_fz, tau_fx, tau_m, cop / 0.075, rh, ry, fy, roll_fz, roll_fy])
        return u0, info

    for _ in range(int(round((walk_t + 3.0 + seconds) / m.opt.timestep))):
        if "mpc" not in box and getattr(ctl, "mpc", None) is not None:   # controller 는 첫 틱에 MPC 를 만든다
            box["mpc"] = ctl.mpc; box["orig"] = ctl.mpc.solve; ctl.mpc.solve = hooked
        tl, ta = ctl.tick()
        model.write_torque(tl, ta); mujoco.mj_step(m, d)
        if W._fell(d):
            print("fell", d.time); break
    R = np.array(rows)
    print(f"vx {vx} vy {vy} wz {wz}  한 발 지지 풀이 {len(R)}  (MPC 질량 {float(ctl.state.reduced.mass):.1f} kg → mg {float(ctl.state.reduced.mass) * 9.81:.0f} N)")
    names = ["적용 Fz (ADMM)", "정확해 Fz", "오차 0 이면 Fz"] + [f"Δ {k} 지우면" for k in ABL] + \
            ["pitch 오차 [°]", "ωy 오차 [°/s]", "z 오차 [mm]", "vz 오차 [mm/s]", "v앞 오차 [m/s]",
             "v옆 오차 (디딤쪽+)", "τ Fz 지렛대 [N·m]", "τ Fx 지렛대 [N·m]", "τ 발 My [N·m]", "CoP x/l", "발 앞뒤 r_h [mm]",
             "발 옆 r_y (디딤쪽+)", "Fy (디딤쪽+) [N]", "roll τ Fz 몫 [N·m]", "roll τ Fy 몫 [N·m]"]
    scale = [1, 1, 1] + [1] * len(ABL) + [180 / np.pi, 180 / np.pi, 1e3, 1e3, 1, 1, 1, 1, 1, 1, 1e3, 1e3, 1, 1, 1]
    bins = np.arange(0.0, 1.0, 0.1)
    print(f"  {'디딤 칸 →':22s}" + "".join(f"{b:7.1f}" for b in bins))
    for j, (nm, sc) in enumerate(zip(names, scale)):
        vals = []
        for b in bins:
            k = (R[:, 0] >= b) & (R[:, 0] < b + 0.1)
            vals.append(np.mean(R[k, 1 + j]) * sc if k.any() else np.nan)
        print(f"  {nm:22s}" + "".join(f"{v:7.1f}" if np.isfinite(v) else "      -" for v in vals))
    cnt = [int(((R[:, 0] >= b) & (R[:, 0] < b + 0.1)).sum()) for b in bins]
    print(f"  {'풀이 수':22s}" + "".join(f"{c_:7d}" for c_ in cnt))


if __name__ == "__main__":
    run(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
