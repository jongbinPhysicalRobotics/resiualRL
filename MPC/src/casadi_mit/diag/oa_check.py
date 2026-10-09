"""OA-MPC 선 접촉 제약을 지키나 (Q&A 10/6 Q47). 매 MPC 풀이 직후, 디딤 발마다 발 yaw 좌표 렌치 w = [F; M] 로:
  (가) MPC 해 그대로 (ADMM 고정 반복의 첫 칸 u0)
  (나) 실제로 다리에 보내는 렌치 = 선 접촉 집합으로 자른 값 (lproj=3 · lprojs=1 과 같은 project_line_wrench mode 2)
  를 비교한다.
  - Mx = x̂_F·M (발 x 축 둘레 모멘트) — OA-MPC 등식 m_x^F = 0
  - 부등식 C·w ≤ 0 (행 0~3 마찰, 6~7 CoP (1 − 2ε)·l, 8~11 두 점 y 마찰 · 비틀림), Fz ≥ 0, Fz ≤ 상한
  - 지평 전체 (22 칸) 의 디딤 Mx 도 따로 (적용되지는 않음)
  (다) 최종 렌치 — 매 틱 토크 계산에 실제로 들어가는 값 (걷기: stance_torque 입력 = 자른 뒤 + 디딤 발 yaw hold Mz,
       서기: standing_torque 입력 = 자른 뒤 + 서기 yaw hold). 지면 → 로봇 부호로 바꿔 같은 제약으로 잰다.
사용: oa_check.py vx vy wz seconds stand(0/1) variant [push_Fy_N]   (서기 외란: 3 s 에 옆으로 0.2 s, 몸통 CoM)
"""
import sys, importlib.util
import numpy as np
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent          # casadi_mit
sys.path.insert(0, str(_ROOT))
import mujoco
spec = importlib.util.spec_from_file_location("walk", str(_ROOT / "10_walk.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
from leg_control import project_line_wrench
from mpc import resolved_foot_yaw
from gait import cfg_get
import leg_control as LC
import controller as CT

ROWS = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11]
ROW_NAME = {0: "마찰 x", 1: "마찰 x", 2: "마찰 y", 3: "마찰 y", 6: "CoP 뒤", 7: "CoP 앞", 8: "점 마찰·비틀림", 9: "점 마찰·비틀림",
            10: "점 마찰·비틀림", 11: "점 마찰·비틀림"}


def foot_frame(F, M, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    Rt = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.concatenate([Rt @ F, Rt @ M])


def run(vx, vy, wz, seconds, stand, var, push_fy=0.0):
    kw = W.parse_variant(var); prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(vx, vy, wz, kw.pop("solver", "osqp"), True, prof, False, bool(stand), 1.0, True, False, **kw)
    m, d = model.m, model.d
    l = float(cfg_get(cfg, "mpc.foot_half_length"))
    t_from = 3.0 if stand else walk_t + 3.0
    box = {}; raw, app, hor, fin = [], [], [], []
    cur = {}
    o_wst, o_st, o_std = CT.walking_stance_torque, LC.stance_torque, CT.standing_torque

    def w_wst(dyn, state, leg, u0, alpha, td_yaw, cfg_, yaw_hold_gains=None, line_proj=None):
        cur["leg"], cur["lp"] = leg, line_proj
        return o_wst(dyn, state, leg, u0, alpha, td_yaw, cfg_, yaw_hold_gains=yaw_hold_gains, line_proj=line_proj)

    def w_st(dyn, F_W, M_W):                       # 걷기 디딤: 발이 지면을 미는 렌치 → 지면 → 로봇 = 부호 반대
        if d.time >= t_from and cur.get("lp") is not None:
            C_, l_, yaw_, _ = cur["lp"]
            ww = foot_frame(-np.asarray(F_W, float), -np.asarray(M_W, float), yaw_)
            fin.append([ww[3], ww[2], *(np.asarray(C_) @ ww)[ROWS], -ww[2], -1.0])
        return o_st(dyn, F_W, M_W)

    def w_std(Jv, Jw, u):                          # 서기: u 는 지면 → 로봇 (자른 뒤 + yaw hold)
        if d.time >= t_from and "mpc" in box:
            fy_ = np.asarray((ctl.last_info or {}).get("foot_yaw", [0.0, 0.0]), dtype=float).reshape(-1, 2)[0]
            for leg in (0, 1):
                ww = foot_frame(np.asarray(u[3 * leg:3 * leg + 3]), np.asarray(u[6 + 3 * leg:9 + 3 * leg]), resolved_foot_yaw(float(fy_[leg])))
                fin.append([ww[3], ww[2], *(box["mpc"].template @ ww)[ROWS], -ww[2], -1.0])
        return o_std(Jv, Jw, u)

    CT.walking_stance_torque, LC.stance_torque, CT.standing_torque = w_wst, w_st, w_std

    def hooked(x0, ref, steps, foot_yaw, reduced, mode, active=None, mz_ref=None):
        mpc = box["mpc"]
        u0, info = box["orig"](x0, ref, steps, foot_yaw, reduced, mode, active=active, mz_ref=mz_ref)
        if d.time < t_from or info.get("fallback"):
            return u0, info
        walking = str(mode).lower().endswith("walking")
        C = getattr(mpc, "template_walk", mpc.template) if walking else mpc.template
        fyaw = np.asarray(foot_yaw, dtype=float)
        st0 = np.asarray(steps[0].stance, dtype=bool)
        U = np.asarray(info["U"], dtype=float)
        for leg in (0, 1):
            if not st0[leg]:
                continue
            yv = fyaw[leg] if fyaw.ndim == 1 else fyaw[0, leg]
            yaw = resolved_foot_yaw(float(yv))
            F, M = u0[3 * leg:3 * leg + 3], u0[6 + 3 * leg:9 + 3 * leg]
            w = foot_frame(F, M, yaw)
            Fg, Mg, _ = project_line_wrench(F, M, yaw, C, l, 2)
            wp = foot_frame(Fg, Mg, yaw)
            for arr, ww in ((raw, w), (app, wp)):
                g = (C @ ww)[ROWS]
                arr.append([ww[3], ww[2], *g, -ww[2], ww[2] - mpc.fz_max])
            st = np.array([np.asarray(s.stance, dtype=bool)[leg] for s in steps])
            for k in np.flatnonzero(st):
                yk = fyaw[leg] if fyaw.ndim == 1 else fyaw[k, leg]
                wk = foot_frame(U[k, 3 * leg:3 * leg + 3], U[k, 6 + 3 * leg:9 + 3 * leg], resolved_foot_yaw(float(yk)))
                hor.append([wk[3], wk[2]])
        return u0, info

    for _ in range(int(round((t_from + seconds) / m.opt.timestep))):
        if "mpc" not in box and getattr(ctl, "mpc", None) is not None:
            box["mpc"] = ctl.mpc; box["orig"] = ctl.mpc.solve; ctl.mpc.solve = hooked
        tl, ta = ctl.tick(); model.write_torque(tl, ta)
        if push_fy and stand:
            d.xfrc_applied[model.base_body, 1] = push_fy if 3.0 <= d.time < 3.2 else 0.0
        mujoco.mj_step(m, d)
        if W._fell(d):
            print("fell", d.time); break
    name = f"{'서기' if stand else ''} vx {vx} vy {vy} wz {wz}"
    for tag, arr in (("(가) MPC 해 그대로 (ADMM 첫 칸)", raw), ("(나) 자른 뒤", app), ("(다) 최종 렌치 (yaw hold 포함, 매 틱)", fin)):
        A = np.array(arr)
        mx = np.abs(A[:, 0]); fz = np.maximum(A[:, 1], 1.0)
        g = A[:, 2:]                                   # 부등식 값 (> 0 이면 위반)
        tol = 1e-6 * (1.0 + np.abs(A[:, 1:2]))
        viol = g > tol
        worst = {}
        for j, r_ in enumerate(ROWS + ["Fz≥0", "Fz≤상한"]):
            if viol[:, j].any():
                key = ROW_NAME.get(r_, r_) if isinstance(r_, int) else r_
                worst[key] = max(worst.get(key, 0.0), float(g[viol[:, j], j].max()))
        print(f"[{name}] {tag}: 표본 {len(A)}")
        print(f"   |Mx| p50 {np.median(mx):.2e} · p95 {np.percentile(mx, 95):.2e} · 최대 {mx.max():.2e} N·m  "
              f"(CoP 환산 |Mx|/Fz p95 {1e3*np.percentile(mx/fz, 95):.2f} mm)")
        print(f"   부등식 위반 비율 {100*viol.any(1).mean():.1f} %  최대 위반 " + (", ".join(f"{k} {v:.2f}" for k, v in worst.items()) if worst else "없음"))
    H = np.array(hor)
    print(f"   (참고) 지평 22 칸 디딤 |Mx| p95 {np.percentile(np.abs(H[:, 0]), 95):.2e} · 최대 {np.abs(H[:, 0]).max():.2e} N·m")


if __name__ == "__main__":
    run(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6],
        float(sys.argv[7]) if len(sys.argv) > 7 else 0.0)
