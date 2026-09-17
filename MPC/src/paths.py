"""프로젝트 경로 모음.

폴더 재편(2026-09-16) 이후 구조:
    residual RL/              ← ROOT (mujoco_menagerie 등 공용 자산)
    ├── MPC/                  ← MPC_ROOT (logs/ 가 여기)
    │   └── src/              ← SRC (이 파일)
    ├── pure RL/  MPC + RL/  deployed RL/  reference/
    └── mujoco_menagerie/
로봇 모델은 5개 작업 폴더가 공유하므로 최상위에 둔다.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent        # .../MPC/src
MPC_ROOT = SRC.parent                        # .../MPC
ROOT = MPC_ROOT.parent                       # .../residual RL  (프로젝트 최상위)

MENAGERIE = ROOT / "mujoco_menagerie"
G1_DIR = MENAGERIE / "unitree_g1"
SCENE_XML = G1_DIR / "scene.xml"
G1_XML = G1_DIR / "g1.xml"
# 토크 제어용으로 우리가 생성하는 씬 (src/make_torque_scene.py 가 만듦)
TORQUE_SCENE_XML = SRC / "g1_torque_scene.xml"
