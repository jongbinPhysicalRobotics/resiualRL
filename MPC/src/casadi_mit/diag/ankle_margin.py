"""발목 한계 여유 (Q&A 10/6 Q68). 걷기 + 3 s 부터 seconds 동안, 일정상 디딤 칸 (경과 10 % 칸마다):
  dx · dz      엉덩이 pitch 관절 − 발목 관절 (몸 yaw 좌표, dx + = 엉덩이가 발목보다 앞, dz = 높이 차) [mm]
  무릎 · 발목   관절각 [rad] (발목 한계 ±0.80), 발 pitch (+ = 발끝 아래) [°], 뒤꿈치 닿음 비율
  IK 발목      같은 dx · dz 에서 발이 평평할 때의 발목 각 (2 링크 IK, 허벅지 0.267 · 정강이 0.2785, 무릎 앞으로 굽음)
  dx_lim       같은 dz 에서 평평한 발 발목 각이 −0.80 이 되는 dx — dx 가 이보다 커지면 (엉덩이가 더 앞) 뒤꿈치가 든다
그리고 높이: MPC 높이 참조 (축소 몸체 CoM z) − 실제, 착지 · 뒤꿈치 듦 순간의 dx.
사용: ankle_margin.py vx variant [seconds]
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

L1, L2 = 0.267, 0.2785          # 허벅지 (hip pitch → knee) · 정강이 (knee → ankle), mit_humanoid.xml body pos
LIM = 0.80


def flat_ankle(dx, dz):
    """평평한 발일 때 발목 각 = −(정강이 앞 기울기). 엉덩이 H − 발목 A = (dx, dz), 무릎은 A→H 선보다 앞."""
    r = np.hypot(dx, dz)
    if r >= L1 + L2:
        r = L1 + L2 - 1e-9
    alpha = np.arctan2(dx, dz)                                   # A→H 선의 앞 기울기
    gamma = np.arccos(np.clip((L2 ** 2 + r ** 2 - L1 ** 2) / (2 * L2 * r), -1, 1))   # A 에서 A→H 와 정강이 사이
    return -(alpha + gamma)


def dx_limit(dz, lim=LIM):
    lo, hi = -0.3, 0.3                                           # flat_ankle 은 dx 에 대해 감소
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if flat_ankle(mid, dz) > -lim:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run(vx, var, seconds=4.0):
    kw = W.parse_variant(resolve(var)); prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(vx, 0, 0, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
    m, d = model.m, model.d
    jh = [model.leg_jnt[l][2] for l in (0, 1)]; jk = [model.leg_jnt[l][3] for l in (0, 1)]; ja = [model.leg_jnt[l][4] for l in (0, 1)]
    f6 = np.zeros(6); rows = []; hz = []
    for _ in range(int(round((walk_t + 3.0 + seconds) / m.opt.timestep))):
        tl, ta = ctl.tick(); model.write_torque(tl, ta); mujoco.mj_step(m, d)
        if W._fell(d):
            print(f"  넘어짐 {d.time:.2f} s"); break
        s = ctl.state
        if s.t < walk_t + 3.0:
            continue
        hz.append([ctl.body.nominal_height_W - float((ctl.wb_dz or 0.0)), float(ctl.x0[5]), d.qpos[2]])
        yaw = s.yaw_unwrapped; hv = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        for leg in (0, 1):
            if not ctl._active(leg, s.t):
                continue
            H, A = d.xanchor[jh[leg]], d.xanchor[ja[leg]]
            Rf = d.xmat[model.foot_body[leg]].reshape(3, 3); p0 = d.xpos[model.foot_body[leg]]
            fh = 0.0
            for c in range(d.ncon):
                con = d.contact[c]
                if model._geom_foot[con.geom1] != leg and model._geom_foot[con.geom2] != leg:
                    continue
                if (Rf.T @ (con.pos - p0))[0] < 0.03:
                    mujoco.mj_contactForce(m, d, c, f6); fh += f6[0]
            ph = ctl.gait.phase(leg, s.t, ctl.clock) * ctl.clock.cycle / ctl.gait.stance
            dx, dz = float((H - A) @ hv), float(H[2] - A[2])
            rows.append([ph, dx, dz, d.qpos[m.jnt_qposadr[jk[leg]]], d.qpos[m.jnt_qposadr[ja[leg]]],
                         np.degrees(np.arcsin(-Rf[2, 0])), float(fh > 5), flat_ankle(dx, dz), dx_limit(dz)])
    R = np.array(rows); Hz = np.array(hz)
    print(f"[vx {vx} {var}]  높이: MPC 참조 (축소 몸체 CoM z) {1e3*Hz[:,0].mean():.1f} · 실제 {1e3*Hz[:,1].mean():.1f} mm (처짐 {1e3*(Hz[:,0]-Hz[:,1]).mean():+.1f}), 몸통 z {1e3*Hz[:,2].mean():.1f}")
    print("   디딤   dx    dz   | 무릎   발목 (IK 평평)  발 pitch  뒤꿈치 | dx_lim  여유 dx_lim−dx")
    for lo in np.arange(0.0, 1.0, 0.1):
        k = (R[:, 0] >= lo) & (R[:, 0] < lo + 0.1)
        if k.any():
            Z = R[k].mean(0)
            print(f"   {lo:.1f}  {1e3*Z[1]:+5.0f} {1e3*Z[2]:5.0f} | {Z[3]:.2f}  {Z[4]:+.2f} ({Z[7]:+.2f})   {Z[5]:+5.1f}°   {Z[6]:.2f}  | {1e3*Z[8]:+4.0f}   {1e3*(Z[8]-Z[1]):+5.0f} mm")


if __name__ == "__main__":
    run(float(sys.argv[1]), sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 4.0)
