"""배포된 G1 RL 정책을 뷰어로 보기.

저장소의 deploy_mujoco.py 와 같은 내용이지만
 (1) 한글 경로 우회 (MuJoCo/torch 가 '백종빈' 을 못 읽음)
 (2) 명령속도를 인자로 받음
 (3) legged_gym 패키지 설치 불필요
가 다르다.

사용 (최상위 폴더에서):
  .venv/Scripts/python.exe "deployed RL/analysis/view_g1_policy.py" --vx 1.0
  .venv/Scripts/python.exe "deployed RL/analysis/view_g1_policy.py" --vx 1.0 --wz 0.3 --seconds 60
"""
import sys, time, argparse
from pathlib import Path
import numpy as np, mujoco, mujoco.viewer, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_g1_policy import load, gravity_orientation

ap = argparse.ArgumentParser()
ap.add_argument("--vx", type=float, default=1.0, help="전진 명령 [m/s]")
ap.add_argument("--vy", type=float, default=0.0, help="횡 명령 [m/s]")
ap.add_argument("--wz", type=float, default=0.0, help="요 명령 [rad/s]")
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--hold", type=float, default=1.0,
                help="방향 유지 게인 (0=끄기). 정책 관측에 yaw 가 없어 스스로는 직진 못 함")
a = ap.parse_args()

m, d, policy, c = load()
nA, nO = c["num_actions"], c["num_obs"]
kps = np.array(c["kps"], np.float32); kds = np.array(c["kds"], np.float32)
dflt = np.array(c["default_angles"], np.float32)
cs = np.array(c["cmd_scale"], np.float32)
cmd = np.array([a.vx, a.vy, a.wz], np.float32)
dt = c["simulation_dt"]; DEC = c["control_decimation"]

action = np.zeros(nA, np.float32); target = dflt.copy(); obs = np.zeros(nO, np.float32)
PEL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
print(f"명령: vx={a.vx} vy={a.vy} wz={a.wz}  방향유지 k={a.hold}   (뷰어 닫으면 종료)")

cnt = 0
mujoco.mj_forward(m, d)
_R0 = d.xmat[PEL].reshape(3, 3)
yaw0 = np.arctan2(_R0[1, 0], _R0[0, 0])
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MPC/src/baseline"))
from viewer_hud import ViewerHUD      # MPC 뷰어와 같은 표시 (sim/실제 시간·속도)
with mujoco.viewer.launch_passive(m, d) as viewer:
    hud = ViewerHUD(m, a.vx)
    start = time.time()
    while viewer.is_running() and time.time() - start < a.seconds:
        t0 = time.time()
        d.ctrl[:] = (target - d.qpos[7:]) * kps + (0.0 - d.qvel[6:]) * kds
        mujoco.mj_step(m, d); cnt += 1
        if cnt % DEC == 0:
            if a.hold > 0.0 and abs(a.wz) < 1e-9:                      # 방향 유지 외부 루프
                R = d.xmat[PEL].reshape(3, 3)
                yaw = np.arctan2(R[1, 0], R[0, 0])
                err = (yaw - yaw0 + np.pi) % (2 * np.pi) - np.pi
                cmd[2] = float(np.clip(-a.hold * err, -0.5, 0.5))
            obs[:3] = d.qvel[3:6] * c["ang_vel_scale"]
            obs[3:6] = gravity_orientation(d.qpos[3:7])
            obs[6:9] = cmd * cs
            obs[9:9+nA] = (d.qpos[7:] - dflt) * c["dof_pos_scale"]
            obs[9+nA:9+2*nA] = d.qvel[6:] * c["dof_vel_scale"]
            obs[9+2*nA:9+3*nA] = action
            ph = (cnt * dt) % 0.8 / 0.8
            obs[9+3*nA:9+3*nA+2] = [np.sin(2*np.pi*ph), np.cos(2*np.pi*ph)]
            action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target = action * c["action_scale"] + dflt
            # 카메라가 로봇을 따라가게
            viewer.cam.lookat[:] = d.xpos[PEL]
        hud.update(viewer, d, cnt * dt, cnt)
        viewer.sync()
        rest = dt - (time.time() - t0)
        if rest > 0:
            time.sleep(rest)
print("종료")
