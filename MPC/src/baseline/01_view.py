"""① 그냥 띄워보기 — 제어 없음. 픽 쓰러지는 게 정상.

사용:
  .venv/Scripts/python.exe MPC/src/baseline/01_view.py            # keyframe 'stand' 자세
  .venv/Scripts/python.exe MPC/src/baseline/01_view.py crouch     # 굽힌 자세
"""
import sys

import mujoco
import mujoco.viewer

import g1_model

m, d = g1_model.load()

if len(sys.argv) > 1 and sys.argv[1] == "crouch":
    g1_model.set_crouch(m, d)
else:
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(m, d, kid)

# position 액추에이터에 ctrl=0 을 주면 '모든 관절 0도'로 끌어당기므로,
# 제어 없음을 제대로 보려면 액추에이터를 토크 모드로 바꾸고 ctrl=0 (= 토크 0).
g1_model.to_torque_actuators(m)
d.ctrl[:] = 0.0

print("제어 없음 (모든 토크 0). 쓰러지는 게 정상입니다. 창을 닫으면 종료.")
mujoco.viewer.launch(m, d)
