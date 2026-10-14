"""디딤 구간별 MPC CoP vs 실제 (Q&A 10/6 Q30). 걷기 시작 + 3 s 부터 seconds 동안, 디딤 경과 10 % 칸마다:
  풀림      변형 toeoff 로 첫 칸이 풀린 비율 (mpc.late0)
  MPC CoP   x/l = −My/(Fz·l) (발 yaw 좌표, + 발끝 · − 뒤꿈치), 발끝 쪽 ≥ 0.79 · 뒤꿈치 쪽 ≤ −0.5 비율
  실제      뒤꿈치 · 발끝 접촉점 (수직력 > 5 N) 닿음 비율, 두 점 힘으로 잰 CoP x/l, 발목 각 [rad] (한계 ±0.80, |q| ≥ 0.79 비율)
  뒤꿈치 힘 몫 (칸 안 수직력 합 비, 10/14 Q67) — 실제 · MPC 계획 (1 − CoP/l)/2
사용: cop_phase.py vx vy wz seconds variant
시뮬 순서는 10_walk.headless 와 같다 (기록만).
"""
import sys, importlib.util
import numpy as np
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent          # casadi_mit
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "diag"))
import mujoco
from configs import resolve
spec = importlib.util.spec_from_file_location("walk", str(_ROOT / "10_walk.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)

ANKLE = ("a10_left_ankle", "a05_right_ankle")


def run(vx, vy, wz, seconds, var):
    kw = W.parse_variant(resolve(var)); prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(vx, vy, wz, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
    m, d = model.m, model.d
    from gait import cfg_get
    l = float(cfg_get(cfg, "mpc.foot_half_length"))
    qadr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in ANKLE]
    f6 = np.zeros(6); rows = []
    for _ in range(int(round((walk_t + 3.0 + seconds) / m.opt.timestep))):
        tl, ta = ctl.tick(); s = ctl.state; u = np.asarray(ctl.u_hold, dtype=float)
        if s.t > walk_t + 3.0:
            for leg in (0, 1):
                if not ctl._active(leg, s.t):
                    continue
                Rf = np.asarray(s.foot_R_W[leg]); p0 = np.asarray(s.foot_pos_W[leg])
                yaw = np.arctan2(Rf[1, 0], Rf[0, 0]); c_, s_ = np.cos(yaw), np.sin(yaw)
                My = -s_ * u[6 + 3 * leg] + c_ * u[7 + 3 * leg]; Fz = u[2 + 3 * leg]
                fz2 = np.zeros(2)
                for c in range(d.ncon):
                    con = d.contact[c]
                    if model._geom_foot[con.geom1] != leg and model._geom_foot[con.geom2] != leg:
                        continue
                    mujoco.mj_contactForce(m, d, c, f6)
                    fz2[1 if Rf[:, 0].dot(con.pos - p0) > 0 else 0] += abs(f6[0])
                ph = ctl.gait.phase(leg, s.t, ctl.clock) * ctl.clock.cycle / ctl.gait.stance
                rows.append([ph, float(getattr(ctl.mpc, "late0", np.zeros(2))[leg]), -My / max(Fz, 1e-6) / l, Fz,
                             float(fz2[0] > 5), float(fz2[1] > 5), (fz2[1] - fz2[0]) / max(fz2.sum(), 1e-6), d.qpos[qadr[leg]], fz2[0], fz2[1]])
        model.write_torque(tl, ta); mujoco.mj_step(m, d)
        if W._fell(d):
            print("fell", d.time); break
    r = np.array(rows); r = r[r[:, 3] > 30]
    print(f"vx {vx} vy {vy} wz {wz}  ({var[-60:]})")
    for lo in np.arange(0.0, 1.0, 0.1):
        k = (r[:, 0] >= lo) & (r[:, 0] < lo + 0.1)
        if k.any():
            print(f"디딤 {lo:.1f}  풀림 {r[k, 1].mean():.2f}  MPC CoP x/l {r[k, 2].mean():+.2f} (≥0.79 {np.mean(r[k, 2] >= 0.79):.2f},"
                  f" ≤−0.5 {np.mean(r[k, 2] <= -0.5):.2f})  | 실제 뒤꿈치 {r[k, 4].mean():.2f} 발끝 {r[k, 5].mean():.2f}"
                  f" CoP x/l {r[k, 6].mean():+.2f}  발목 {r[k, 7].mean():+.2f} rad (한계 닿음 {np.mean(np.abs(r[k, 7]) >= 0.79):.2f})"
                  f"  뒤꿈치 힘 몫 실제 {r[k, 8].sum() / max(r[k, 8].sum() + r[k, 9].sum(), 1e-6):.2f} · MPC {np.mean((1 - np.clip(r[k, 2], -1, 1)) / 2):.2f}")
    print(f"디딤 전체 뒤꿈치 수직 충격량 몫: 실제 {r[:, 8].sum() / max(r[:, 8].sum() + r[:, 9].sum(), 1e-6):.2f} · MPC 계획 {np.mean((1 - np.clip(r[:, 2], -1, 1)) / 2):.2f}"
          f"  (뒤꿈치 닿은 틱 {r[:, 4].mean():.2f}, 발목 한계 닿은 틱 {np.mean(np.abs(r[:, 7]) >= 0.79):.2f})")


if __name__ == "__main__":
    run(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
