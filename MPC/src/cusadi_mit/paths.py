"""프로젝트 경로 모음 (MIT 휴머노이드 포팅 폴더).

구조:
    residual RL/              ← ROOT
    ├── MPC/                  ← MPC_ROOT (logs/ 가 여기)
    │   ├── models/mit_humanoid/   ← MIT_DIR (scene.xml, mit_humanoid.xml, meshes_v3/)
    │   └── src/mit/          ← SRC (이 폴더)
    └── reference/            ← C++ reference (읽기 전용)
폴더마다 이 파일의 복사본이 있다 (06 §1.2 패턴). 깊이에 묶이지 않도록 위로 올라가며 'MPC' 폴더를 찾는다.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent                        # .../MPC/src/mit
MPC_ROOT = next(p for p in SRC.parents if p.name == "MPC")   # .../MPC
ROOT = MPC_ROOT.parent                                       # .../residual RL

MIT_DIR = MPC_ROOT / "models" / "mit_humanoid"               # 우리 복사본 (D1)
MIT_SCENE_XML = MIT_DIR / "scene.xml"
MIT_XML = MIT_DIR / "mit_humanoid.xml"
LOG_DIR = MPC_ROOT / "logs"
CONFIG_YAML = SRC / "mit_controller.yaml"
REFERENCE_DIR = ROOT / "reference"
