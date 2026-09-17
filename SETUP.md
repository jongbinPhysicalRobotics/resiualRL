# 새 PC 에서 이 저장소 복원하기

저장소에는 **우리가 직접 쓴 것(1.6 MB)만** 들어 있다.
가상환경과 외부 저장소는 아래 순서로 새로 만든다.

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
.venv/Scripts/python.exe MPC/src/09_walk.py --seconds 6 --uppd 300 --swingid --softland \
    --lamswing --wn 30 --zeta 0.7 --sf 0.57 --qpy 300 --swingyaw

# 제약 검증 (LP 대조)
.venv/Scripts/python.exe MPC/src/10_constraint_check.py

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
