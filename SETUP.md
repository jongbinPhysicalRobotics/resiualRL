# 새 PC 에서 이 저장소 복원하기

저장소에는 **우리가 직접 쓴 것만** 들어 있다.
가상환경과 외부 저장소는 아래 순서로 새로 만든다.

> **2026-09-25 부터 기본 실행 환경은 Ubuntu** (reference C++ 가 Linux/macOS 전용 — termios·POSIX 공유 메모리).
> 아래 "Ubuntu" 절을 먼저 따르고, 그 뒤의 Windows 절은 예전 기록으로 남긴다.

---

## Ubuntu (24.04 기준)

```bash
# 0) 시스템 패키지 — 파이썬 venv + reference C++ 빌드 (GLFW 창 포함)
sudo apt update && sudo apt install -y python3-venv python3-dev git build-essential cmake \
    curl zip unzip tar pkg-config xorg-dev libxinerama-dev libxcursor-dev libglu1-mesa-dev

# 1) 저장소 + 외부 저장소 3 개 (경로에 한글이 없게 — 예: ~/work/residual-RL)
git clone https://github.com/jongbinPhysicalRobotics/resiualRL.git ~/work/residual-RL && cd ~/work/residual-RL
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
git clone --depth 1 https://github.com/ispaik06/convex-mpc-biped.git reference
git clone --depth 1 https://github.com/unitreerobotics/unitree_rl_gym.git "deployed RL/unitree_rl_gym"

# 2) 파이썬 — Windows 에서 쓰던 버전 그대로 (requirements.txt). Ubuntu 24.04 기본은 3.12:
#    고정 버전이 3.12 용 휠이 없으면 python3.13 (deadsnakes PPA) 으로 venv 를 만든다.
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # GPU 머신이면 cu 빌드

# 3) 동작 확인 (Windows 의 .venv/Scripts/python.exe 대신 .venv/bin/python, PYTHONIOENCODING 은 필요 없다)
.venv/bin/python MPC/src/mit/02_unit_checks_A.py
.venv/bin/python MPC/src/mit/10_walk.py --vx 0.6 --profile ramp --fix yawref --seconds 20
GAIT_ON=1 .venv/bin/python MPC/src/affine/18_gait_quality.py --seconds 120 --vx 0.5 0.7 \
    --var "tds=1.25,copm=0.9,wzp=1,wxp=0,ctl=affine,refshape=1"
```

### reference (C++) 빌드 — MIT 이식 검증용 (Q&A 9/25 Q4)

```bash
# vcpkg (Eigen, glfw3, nlohmann-json, osqp, osqp-eigen, yaml-cpp 를 받아 온다)
mkdir -p ~/.local && cd ~/.local && git clone https://github.com/microsoft/vcpkg.git && ./vcpkg/bootstrap-vcpkg.sh
echo 'export VCPKG_ROOT="$HOME/.local/vcpkg"' >> ~/.bashrc && source ~/.bashrc
# MuJoCo 소스 빌드 → ~/.local/mujoco  (reference CMakePresets 가 $HOME/.local/mujoco/lib/cmake/mujoco 를 찾는다)
cd ~ && git clone https://github.com/google-deepmind/mujoco.git && cd mujoco
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$HOME/.local/mujoco" && cmake --build build -j && cmake --install build
# reference
cd ~/work/residual-RL/reference && cmake --preset dev && cmake --build --preset dev -j
./build/apps/main g y          # G1 + 뷰어. 키는 뷰어 창이 아니라 이 터미널에 (w/s a/d q/e, space)
```

### 옮긴 뒤 확인할 것
- **수치 재기준**: MuJoCo·BLAS 빌드가 달라 부동소수점이 조금씩 다르고, 보행은 혼돈적이라 생존 시간 같은 수치가 바뀔 수 있다.
  Q&A 의 표를 인용하기 전에 대표 표 몇 개 (위 3) 의 18 번 표, `MPC/src/mit/11_gait_quality.py` 세 목표) 를 Ubuntu 에서 다시 돌려 기준을 새로 잡는다.
- **Claude Code 작업 규칙**: 저장소 최상위 `CLAUDE.md` 에 옮겨 두었다 (Windows 쪽 메모리는 그 PC 에만 있다).
- GPU 가 있는 머신이면 `nvidia-smi` 로 확인 — IsaacGym/MJX 계열은 Windows 노트북에선 못 썼다.

---

## Windows (2026-09-25 까지 쓰던 환경 — 기록)

> ## ⚠ 폴더 위치: **영문 경로**에 둘 것
> MuJoCo 와 torch 의 C 레벨 파일 열기가 **경로의 한글을 못 읽는다.**
> `C:\Users\백종빈\...` 에서는 XML·정책 로드가 실패해서 우회 코드를 넣어야 했다.
> `C:\work\residual-RL` 같은 영문 경로면 그 문제가 아예 없다.
> (IsaacLab·IsaacGym 계열은 특히 경로에 예민하다.)

---

## 1. 저장소 받기

```bash
git clone <이 저장소 URL> residual-RL
cd residual-RL
```

## 2. 외부 저장소 3개 clone

```bash
# 로봇 모델 (MPC·RL 전부가 쓴다 — 필수)
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git

# 대조 구현 (C++ convex MPC) — 분석용, 없어도 코드는 돈다
git clone --depth 1 https://github.com/ispaik06/convex-mpc-biped.git reference

# 배포된 G1 RL 정책 — deployed RL 분석용
git clone --depth 1 https://github.com/unitreerobotics/unitree_rl_gym.git "deployed RL/unitree_rl_gym"
```

## 3. 가상환경

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install mujoco numpy scipy quadprog matplotlib pyyaml
```

**torch** 는 용도에 따라 다르게 깐다:

```bash
# (a) 배포 정책 실행/계측만 → CPU 로 충분
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# (b) 강화학습 학습 (랩 GPU 머신) → CUDA 빌드
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## 4. 동작 확인

```bash
# MPC — 제자리 스텝 (6초)
.venv/Scripts/python.exe MPC/src/baseline/09_walk.py --seconds 6 --uppd 300 --swingid --softland \
    --lamswing --wn 30 --zeta 0.7 --sf 0.57 --qpy 300 --swingyaw

# 제약 검증 (LP 대조)
.venv/Scripts/python.exe MPC/src/baseline/10_constraint_check.py

# 배포 정책 계측
.venv/Scripts/python.exe "deployed RL/analysis/run_g1_policy.py" --vx 1.0 --seconds 30
```

---

## 저장소에 **없는** 것과 이유

| | 용량 | 왜 제외했나 |
|---|---|---|
| `.venv/` | 1,155 MB | 절대경로가 박혀 다른 PC 에서 깨짐. CPU/CUDA 빌드도 달라야 함 |
| `MPC/logs/*.npz,csv` | 429 MB | 실험 로그 — 다시 돌리면 나온다. **PNG 플롯은 포함** |
| `reference/` | 249 MB | 남의 git 저장소 |
| `deployed RL/unitree_rl_gym/` | 144 MB | 남의 git 저장소 (정책 `motion.pt` 포함) |
| `mujoco_menagerie/` | 38 MB | 남의 git 저장소 |
| `.vscode/` | — | 한글 절대경로가 들어감 |

## 영문 경로로 옮긴 뒤 정리하면 좋을 것

`deployed RL/analysis/run_g1_policy.py` 의 두 우회는 한글 경로 전용이라 **지워도 된다**:
- `load()` 의 `os.chdir` (MuJoCo XML 로드용)
- `torch.jit.load(io.BytesIO(...))` (정책 로드용)

지워도 동작은 같으니 급하지 않으면 그대로 둬도 무방하다.
