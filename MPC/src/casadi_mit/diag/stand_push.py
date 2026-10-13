"""서기 외란에서 어떻게 넘어지나 (Q&A 10/6 Q55). 서기 (FSM 서기 고정), 3 s 에 몸통 CoM 에 수평력 0.2 s.
50 ms 마다: 몸통 pitch · roll [°], 몸 (MPC CoM) 의 두 발 중점 대비 앞 · 옆 [mm] · 속도, 몸 높이,
두 발 평균 MPC CoP x/l (발 yaw 좌표, + 발끝) · 실제 CoP x/l (뒤꿈치 · 발끝 접촉점 힘), 뒤꿈치 · 발끝 닿음,
발목 · 무릎 · hip pitch 각 평균 [rad], MPC Fz 합.
사용: stand_push.py Fx Fy variant [t_end]
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
from gait import cfg_get


def run(fx, fy, var, t_end=6.0):
    kw = W.parse_variant(resolve(var)); prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(0, 0, 0, kw.pop("solver", "osqp"), True, prof, False, True, 1.0, True, False, **kw)
    m, d = model.m, model.d
    l = float(cfg_get(cfg, "mpc.foot_half_length"))
    J = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in
         ("a10_left_ankle", "a05_right_ankle", "a09_left_knee", "a04_right_knee", "a08_left_hip_pitch", "a03_right_hip_pitch")}
    f6 = np.zeros(6); nxt = 2.9
    print(f"서기 외란 Fx {fx} Fy {fy} N (3.0 ~ 3.2 s), {var[-40:]}")
    print("  t     pitch°  roll°  CoM앞mm CoM옆mm  v앞    z     MPC CoP  실제 CoP  뒤꿈치 발끝  발목    무릎   hip    ΣFz")
    for _ in range(int(round(t_end / m.opt.timestep))):
        tl, ta = ctl.tick(); model.write_torque(tl, ta)
        d.xfrc_applied[model.base_body, 0:2] = (fx, fy) if 3.0 <= d.time < 3.2 else (0.0, 0.0)
        mujoco.mj_step(m, d)
        if W._fell(d):
            print(f"  넘어짐 {d.time:.2f} s"); break
        if d.time + 1e-9 < nxt:
            continue
        nxt += 0.05
        s = ctl.state; u = np.asarray(ctl.u_hold); x0 = ctl.x0
        yaw = s.yaw_unwrapped; hv = np.array([np.cos(yaw), np.sin(yaw)]); yv = np.array([-np.sin(yaw), np.cos(yaw)])
        mid = 0.5 * (np.asarray(s.foot_pos_W[0][:2]) + np.asarray(s.foot_pos_W[1][:2]))
        cop_m, cop_r, heel, toe = [], [], [], []
        for leg in (0, 1):
            Rf = np.asarray(s.foot_R_W[leg]); fyaw = np.arctan2(Rf[1, 0], Rf[0, 0])
            My = -np.sin(fyaw) * u[6 + 3 * leg] + np.cos(fyaw) * u[7 + 3 * leg]; Fz = u[2 + 3 * leg]
            cop_m.append(-My / max(Fz, 1e-6) / l)
            p0 = np.asarray(s.foot_pos_W[leg]); fh = ft = 0.0
            for c in range(d.ncon):
                con = d.contact[c]
                if model._geom_foot[con.geom1] != leg and model._geom_foot[con.geom2] != leg:
                    continue
                mujoco.mj_contactForce(m, d, c, f6)
                if Rf[:, 0].dot(con.pos - p0) > 0:
                    ft += abs(f6[0])
                else:
                    fh += abs(f6[0])
            cop_r.append((ft - fh) / max(ft + fh, 1e-6)); heel.append(fh > 5); toe.append(ft > 5)
        q = lambda a, b: 0.5 * (d.qpos[J[a]] + d.qpos[J[b]])
        print(f"  {d.time:4.2f} {np.degrees(s.pitch):+6.1f} {np.degrees(s.roll):+6.1f} {1e3 * (x0[3:5] - mid).dot(hv):+7.0f} {1e3 * (x0[3:5] - mid).dot(yv):+7.0f}"
              f" {x0[9:11].dot(hv):+5.2f} {x0[5]:.3f}  {np.mean(cop_m):+6.2f}   {np.mean(cop_r):+6.2f}    {sum(heel)}/2   {sum(toe)}/2"
              f"  {q('a10_left_ankle', 'a05_right_ankle'):+.2f}  {q('a09_left_knee', 'a04_right_knee'):+.2f}  {q('a08_left_hip_pitch', 'a03_right_hip_pitch'):+.2f}  {u[2] + u[5]:5.0f}")


if __name__ == "__main__":
    run(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 6.0)
