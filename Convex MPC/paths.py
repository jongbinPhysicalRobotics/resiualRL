"""경로 모음 — 이 폴더 하나로 완결 (복사해서 다른 곳에 둬도 그대로 동작).

    Convex MPC/              ← SRC (이 폴더)
    ├── models/mit_humanoid/ ← MIT_DIR (scene.xml, mit_humanoid.xml, meshes_v3/)
    ├── logs/                ← LOG_DIR (--log 일 때만 생김)
    └── mit_controller.yaml  ← CONFIG_YAML

원본 MPC/src/casadi_mit/paths.py 는 위로 올라가며 'MPC' 폴더를 찾지만, 여기는 폴더 안만 본다.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC

MIT_DIR = SRC / "models" / "mit_humanoid"
MIT_SCENE_XML = MIT_DIR / "scene.xml"
MIT_XML = MIT_DIR / "mit_humanoid.xml"
LOG_DIR = SRC / "logs"
CONFIG_YAML = SRC / "mit_controller.yaml"
