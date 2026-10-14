"""서기 발 힘 분포 (Q&A 10/6 Q66). 서기 seconds 동안 (외란 선택), 시간 창마다 발별:
  뒤꿈치 몫 (발 x < 0.03 접촉의 수직력 / 발 전체)  · 바깥 몫 (상자 발: 다른 발에서 먼 쪽 모서리)
  실제 CoP (발 좌표, site 기준 [mm], x + 발끝 · y + 바깥)  vs  MPC 계획 CoP (u 의 모멘트 / Fz, 같은 좌표)
  발 roll [°] (+ = 바깥 모서리가 위, − = 바깥 모서리가 아래)
그리고 전신 CoM · 축소 몸체 (몸통 + 팔) CoM 의 위치 (두 발 site 가운데 기준, 몸 yaw 좌표 [mm]),
두 발 옆 간격 [mm] · 몸통 높이 [mm] · hip abad 각 (바깥으로 벌림 +) [°].
사용: stand_load.py variant [Fx:Fy:t0:dur] [seconds]
"""
import sys, importlib.util
import numpy as np
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "diag"))
import mujoco
spec = importlib.util.spec_from_file_location("walk", str(_ROOT / "10_walk.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
from configs import resolve

SITE_X = 0.03


def run(var, push=None, seconds=12.0):
    kw = W.parse_variant(resolve(var)); prof = kw.pop("profile", "ramp"); kw.pop("push", None)
    cfg, model, ctl, walk_t = W._setup(0, 0, 0, kw.pop("solver", "osqp"), True, prof, False, True, 1.0, True, False, **kw)
    m, d = model.m, model.d
    f6 = np.zeros(6); rows = []
    for _ in range(int(round(seconds / m.opt.timestep))):
        tl, ta = ctl.tick(); model.write_torque(tl, ta)
        if push is not None and W._push_active(push, 0.0, d.time):
            d.xfrc_applied[model.base_body, 0:3] = W._push_world(push[0], push[1], ctl.state.yaw_unwrapped)
        else:
            d.xfrc_applied[model.base_body, :] = 0.0
        mujoco.mj_step(m, d)
        if W._fell(d):
            print(f"  넘어짐 {d.time:.2f} s"); break
        s = ctl.state; u = np.asarray(ctl.u_hold)
        yaw = s.yaw_unwrapped; Ryaw = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        sites = [np.asarray(s.foot_pos_W[l]) for l in (0, 1)]
        mid = 0.5 * (sites[0] + sites[1])
        com = d.subtree_com[0]
        rb = s.reduced.com_W if hasattr(s.reduced, "com_W") else np.full(3, np.nan)
        qab = [d.qpos[model.leg_qadr[l][1]] for l in (0, 1)]          # hip abad (축 +x: 왼다리 + · 오른다리 − 가 바깥 벌림)
        row = [d.time, *(Ryaw.T @ (com[:2] - mid[:2])), *(Ryaw.T @ (np.asarray(rb)[:2] - mid[:2])),
               (Ryaw.T @ (sites[0][:2] - sites[1][:2]))[1], d.qpos[2], np.degrees(qab[0]), -np.degrees(qab[1])]
        for leg in (0, 1):
            fb = model.foot_body[leg]; R = d.xmat[fb].reshape(3, 3); p = d.xpos[fb]
            out_sign = 1.0 if leg == 0 else -1.0             # 왼발 (0) 바깥 = 발 +y, 오른발 바깥 = 발 −y
            fz = heel = outer = cx = cy = 0.0
            for c in range(d.ncon):
                con = d.contact[c]
                if model._geom_foot[con.geom1] != leg and model._geom_foot[con.geom2] != leg:
                    continue
                mujoco.mj_contactForce(m, d, c, f6)
                q = R.T @ (con.pos - p); fn = f6[0]
                fz += fn; heel += fn * (q[0] < SITE_X); outer += fn * (out_sign * q[1] > 1e-3)
                cx += fn * (q[0] - SITE_X); cy += fn * out_sign * q[1]
            Fl = R.T @ u[3 * leg:3 * leg + 3]; Ml = R.T @ u[6 + 3 * leg:9 + 3 * leg]
            roll = np.arctan2(R[2, 1], R[2, 2]) * out_sign
            row += [fz, heel / max(fz, 1e-6), outer / max(fz, 1e-6), cx / max(fz, 1e-6), cy / max(fz, 1e-6),
                    -Ml[1] / max(Fl[2], 1e-6), out_sign * Ml[0] / max(Fl[2], 1e-6), np.degrees(roll)]
        rows.append(row)
    R_ = np.array(rows); t = R_[:, 0]
    wins = [(1.5, 3.0, "외란 전")]
    if push is not None:
        t0 = push[2]
        wins += [(t0, t0 + 0.5, "외란 + 0.5 s"), (t0 + 2.0, t0 + 4.0, "외란 뒤 2 ~ 4 s"), (t0 + 6.0, seconds, f"외란 뒤 6 s ~")]
    else:
        wins += [(6.0, seconds, "6 s ~")]
    print(f"[{var}{'' if push is None else '  push ' + ':'.join(f'{x:g}' for x in push)}]")
    for a, b, nm in wins:
        w = (t >= a) & (t < b)
        if not w.any():
            continue
        Z = R_[w].mean(0)
        line = (f"  {nm:14s} CoM x {1e3*Z[1]:+3.0f} y {1e3*Z[2]:+3.0f} · 발 간격 {1e3*Z[5]:5.1f} · 높이 {1e3*Z[6]:5.1f} · abad L {Z[7]:+.2f} R {Z[8]:+.2f}° |")
        for leg in (0, 1):
            b_ = 9 + 8 * leg
            fz, he, ou, cx, cy, px, py, rl = Z[b_:b_ + 8]
            line += (f" {'LR'[leg]} 뒤꿈치 {100*he:2.0f}% 바깥 {100*ou:2.0f}% CoP x {1e3*cx:+3.0f} y {1e3*cy:+3.0f} (MPC {1e3*px:+3.0f} {1e3*py:+2.0f})"
                     f" roll {rl:+.2f}° |")
        print(line)


if __name__ == "__main__":
    pu = tuple(float(x) for x in sys.argv[2].split(":")) if len(sys.argv) > 2 and sys.argv[2] not in ("-", "") else None
    run(sys.argv[1], pu, float(sys.argv[3]) if len(sys.argv) > 3 else 12.0)
