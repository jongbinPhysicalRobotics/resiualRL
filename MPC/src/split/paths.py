"""프로젝트 경로 모음.

구조 (2026-09-24 src 하위 폴더 분리 이후):
    residual RL/              ← ROOT (mujoco_menagerie 등 공용 자산)
    ├── MPC/                  ← MPC_ROOT (logs/ 가 여기)
    │   └── src/
    │       ├── baseline/     ← 기존 컨트롤러 (09_walk) + 분석 스크립트
    │       ├── affine/       ← 아핀항 MPC (23)
    │       └── split/        ← MPC·스윙 분리 (24, 보류)
    ├── pure RL/  MPC + RL/  deployed RL/  reference/
    └── mujoco_menagerie/
폴더마다 이 파일의 복사본이 있다. 깊이에 묶이지 않도록 위로 올라가며 'MPC' 폴더를 찾는다.
로봇 모델은 5개 작업 폴더가 공유하므로 최상위에 둔다.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent                        # 이 파일이 있는 폴더 (예: .../MPC/src/baseline)
MPC_ROOT = next(p for p in SRC.parents if p.name == "MPC")   # .../MPC
ROOT = MPC_ROOT.parent                                       # .../residual RL  (프로젝트 최상위)

MENAGERIE = ROOT / "mujoco_menagerie"
G1_DIR = MENAGERIE / "unitree_g1"
SCENE_XML = G1_DIR / "scene.xml"
G1_XML = G1_DIR / "g1.xml"
# 토크 제어용으로 우리가 생성하는 씬 (src/make_torque_scene.py 가 만듦)
TORQUE_SCENE_XML = SRC / "g1_torque_scene.xml"
