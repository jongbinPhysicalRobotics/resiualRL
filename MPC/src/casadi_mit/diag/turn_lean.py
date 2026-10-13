"""제자리 회전 때 몸이 회전 방향으로 치우쳐 보이나 (Q&A 10/6 Q52). 걷기 시작 + 4 s 부터 seconds 동안:
  - hip yaw 관절각 평균 (왼 · 오른, + = 반시계 방향 회전과 같은 쪽)
  - 몸통 roll · pitch 평균 [°] (roll + = 오른쪽으로 기울어짐 — 반시계 회전의 바깥? 부호는 MuJoCo: x 축 둘레 + 회전은 왼쪽이 올라감)
  - 몸 (MPC 기준 CoM, x0) 이 두 발 중점보다 몸 좌표로 앞 · 옆 (왼 +) 으로 얼마나 있나 [mm]
  - 제자리 이탈: 몸 xy 위치의 시작점 대비 최대 거리, 평균 위치의 반경 [mm]
  - 무릎 (정강이 원점) 의 몸 좌표 옆 위치 평균 (다리가 한쪽으로 쏠려 보이는지) [mm]
사용: turn_lean.py vx vy wz seconds variant
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


def run(vx, vy, wz, seconds, var):
    kw = W.parse_variant(resolve(var)); prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(vx, vy, wz, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
    m, d = model.m, model.d
    jn = ["a06_left_hip_yaw", "a01_right_hip_yaw"]
    qadr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in jn]
    knee = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_lower_leg", "right_lower_leg")]
    rows = []; fell = -1.0
    for _ in range(int(round((walk_t + 4.0 + seconds) / m.opt.timestep))):
        tl, ta = ctl.tick(); model.write_torque(tl, ta); mujoco.mj_step(m, d)
        if W._fell(d):
            fell = d.time; break
        s = ctl.state
        if s.t < walk_t + 4.0:
            continue
        yaw = s.yaw_unwrapped; c, sn = np.cos(yaw), np.sin(yaw)
        hv, yv = np.array([c, sn]), np.array([-sn, c])
        com = np.asarray(ctl.x0[3:5]); mid = 0.5 * (s.foot_pos_W[0][:2] + s.foot_pos_W[1][:2])
        base = np.asarray(s.torso_pos_W[:2])
        kn = [(d.xpos[b][:2] - base).dot(yv) for b in knee]
        rows.append([d.qpos[qadr[0]], d.qpos[qadr[1]], s.roll, s.pitch, (com - mid).dot(hv), (com - mid).dot(yv),
                     base[0], base[1], kn[0], kn[1]])
    R = np.array(rows)
    p0 = R[0, 6:8]; dist = np.linalg.norm(R[:, 6:8] - p0, axis=1); cen = R[:, 6:8].mean(0)
    print(f"[{var[-24:]}] vx {vx} wz {wz}: {'넘어짐 %.1f s' % fell if fell > 0 else '버팀'} (창 {len(R) * m.opt.timestep:.1f} s)")
    print(f"   hip yaw 평균 왼 {np.degrees(R[:, 0].mean()):+.1f}° · 오른 {np.degrees(R[:, 1].mean()):+.1f}°  (둘 평균 {np.degrees(R[:, :2].mean()):+.1f}°)")
    print(f"   몸통 roll 평균 {np.degrees(R[:, 2].mean()):+.1f}° (σ {np.degrees(R[:, 2].std()):.1f})  pitch 평균 {np.degrees(R[:, 3].mean()):+.1f}°")
    print(f"   CoM − 두 발 중점 (몸 좌표): 앞 {1e3 * R[:, 4].mean():+.0f} mm · 옆 {1e3 * R[:, 5].mean():+.0f} mm (σ {1e3 * R[:, 5].std():.0f})")
    print(f"   무릎 옆 위치 (몸통 원점 기준, 몸 좌표): 왼 {1e3 * R[:, 8].mean():+.0f} · 오른 {1e3 * R[:, 9].mean():+.0f} mm  (명목 ±82 mm 근처)")
    print(f"   제자리 이탈: 시작점에서 최대 {1e3 * dist.max():.0f} mm, 위치 평균의 시작점 대비 {1e3 * np.linalg.norm(cen - p0):.0f} mm")


if __name__ == "__main__":
    run(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
