# 실험실 작업 인계 — 노트북 Claude → 실험실 Claude (2026-11-09)

이 문서는 실험실 컴퓨터 (RTX 5080) 에서 일하는 Claude Code 를 위한 것이다. 사용자는 학부 졸업 논문
(MIT 휴머노이드의 Di Carlo 식 convex SRB MPC + residual RL) 을 노트북에서 해 오다가, MPC 의 GPU 병렬화를 위해 이 컴퓨터로 옮겼다.

## 0. 먼저 읽을 것 (이 순서로)

1. [CLAUDE.md](../../../CLAUDE.md) — 작업 규칙.
2. 아래 §1 — 노트북의 Claude 메모리에만 있던 규칙. **이 컴퓨터의 네 메모리에 저장해 둘 것.**
3. [Q&A/2026-10-06.md](../../../Q&A/2026-10-06.md) 의 **Q110** (지금 상태). 배경은 Q104 ~ Q109 (Ruiz · ρ · warm start · CusADi 계획).
4. [cusadi_mit/README.md](README.md) — 설치 · 실행 순서.

## 1. 규칙 (노트북 메모리에서 옮김)

- **답은 한국어로만.** 영어 문장 금지 (사용자가 여러 번 요구).
- **git push 는 사용자가 요청할 때만.** 커밋은 Q&A 항목 하나마다. 기록은 `Q&A/<날짜>.md` 에 Q 번호로 쌓고 `Q&A/README.md` 색인에 한 줄.
  지금 이어 쓰는 파일은 `Q&A/2026-10-06.md` (Q110 까지, 항목 머리에 실제 날짜 `(11/9)` 를 적는 관례).
- 주장마다 근거 종류 (코드 사실 / 측정 / 추론 / 근거 없는 추측). 틀린 말은 원래 항목에 "⚠ 정정 (Qn)" 을 달고 새 항목에서 바로잡는다.
- 영상 녹화는 사용자가 요청할 때만. 대신 뷰어 명령을 준다 (`GAIT_ON=1 PYGLFW_LIBRARY_VARIANT=x11 … 10_walk.py --view …`).
- MPC 원칙 (모두 사용자 결정):
  - SRB convex MPC 틀 유지, 선형화 · 볼록만 — 비선형 최적화 (NLP) 금지. MIT 에는 affine · WBC 항 금지.
  - MPC 중력값 조작 금지. MPC 제약 변경 금지 (보충은 다른 곳에서). 발은 line foot.
  - 풀이는 고정 반복 ADMM (Residual MPC 식) — polishing · 적응 ρ · 종료 판정을 넣지 않는다.
  - 걸음 주기를 늘리는 쪽으로는 가도 스윙 높이를 낮추지 않는다 (나중에 장애물).
  - 서기 외란 회복에서 발을 내딛지 않는다 (서기 = 제자리 버팀).
- 새 기능은 플래그로, 기본값 = 기존 동작. 채택은 120 s 완주 + 걸음 품질표 + 사용자 렌더 확인.
- 시험 범위: vx 1.0 m/s · wz 2.5 rad/s 까지, 서기 · 직진 걷기에 몸통 외란 0 ~ 50 N, 발 떨림도 본다.
- CPU 부하: 시뮬 일괄 실행은 4 프로세스 이하 · `nice -n 19` · `OPENBLAS_NUM_THREADS=1` (사용자가 따로 허락하면 늘림).
- **이 컴퓨터는 공용 계정이다.** 다른 사람의 conda 환경 (`env_isaaclab` · `parc*` · `beyond_mimic_env` · `~/Desktop/ahn/...`) 과 `~/IsaacLab` 은
  읽기만 하고 바꾸지 않는다. 우리 환경은 `cmpc` (`env_isaaclab` 복제본). sudo 는 쓰지 않는다. 큰 GPU 실험 전에는 `nvidia-smi` 로 다른 사용자를 확인.
- 현재 기준 설정은 **A42** ([diag/configs.py](diag/configs.py)). 바꾸지 않는다.

## 2. 지금 상태 (Q110 요약)

- `MPC/src/cusadi_mit/` = `casadi_mit` 복사본 + GPU 일괄 MPC. A42 를 환경 B 개 동시에 푼다.
  - [batch_mpc.py](batch_mpc.py): torch QP 조립 + OSQP 1.1.3 고정 반복 재현. 선택지 `backend` casadi / cusadi, `linsys` chol / cudss / kktcpu, `dtype`.
  - [cudss_kkt.py](cudss_kkt.py), [cusadi_lite/](cusadi_lite/) (CusADi wip 의 코드 생성 · cuDSS 연결부, GPU 감지 · pip cuDSS 로 수정).
  - [20_gpu_check.py](20_gpu_check.py): CPU 판 기록 → 배치 재생 대조 · 시간 재기.
- 노트북 CPU float64 결과: CPU 판 OSQP 와 최대 2.6e-8 N (7 경우 · 1 900 풀이), kktcpu 7.7e-8 N.
- 재현에서 중요한 OSQP 1.1.3 세부 (코드 사실): Ruiz 의 P 열 노름은 **위 삼각만** (`csc_col_norm_inf`), 비용 배율은 열 노름 **평균** vs ‖q‖∞,
  warm start 는 x 만 새 배율로 · z = Âx · **y 는 스케일된 값 그대로 유지**. 이 규칙들을 "맞추려고" 바꾸지 말 것.
- **GPU 에서는 아직 한 번도 안 돌렸다.** CusADi 컴파일 · cuDSS 링크 · sm_120 이 처음 확인할 것들이다.
- 이 컴퓨터: Ubuntu 24.04 · Ryzen 7 9800X3D · RTX 5080 16 GB (드라이버 595.84) · nvcc 12.9 · cuDSS 없음 (pip 로 설치) ·
  Isaac Lab 2.3.2 + Isaac Sim 5.1 (`env_isaaclab`, torch 2.7.0+cu128). 시뮬레이터는 Isaac Lab 으로 정했다 (닫힌 고리는 다음 단계).

## 3. 할 일 (순서대로 — 단계마다 결과 표를 사용자에게 보여 주고, 막히면 멈추고 물어본다)

공통: `PY=~/miniconda3/envs/cmpc/bin/python`, 저장소 루트에서, `export OPENBLAS_NUM_THREADS=1`, `C=MPC/src/cusadi_mit/20_gpu_check.py`.

**A. 환경 확인** (사용자가 이미 만들었을 수 있다 — 있으면 확인만)
```bash
conda env list | grep cmpc || conda create -n cmpc --clone env_isaaclab -y
$PY -m pip install "nvidia-cudss-cu12==0.7.1.6" ninja
$PY -m pip list --format=freeze > /tmp/cmpc_pins.txt
$PY -m pip install -c /tmp/cmpc_pins.txt casadi==3.8.0 osqp==1.1.3 quadprog PyYAML glfw
$PY -c "import sys, numpy, scipy, casadi, osqp, torch, mujoco; print(sys.version.split()[0], numpy.__version__, scipy.__version__, casadi.__version__, osqp.__version__, torch.__version__, torch.version.cuda, mujoco.__version__, torch.cuda.get_device_name(0))"
```
- pip 가 기존 판과 충돌한다고 하면 억지로 바꾸지 말고 오류를 보여 주고 상의한다 (Isaac Sim 이 쓰는 numpy 등).

**B. CPU 재현** — 노트북과 같은 자릿수 (U 전체 ≤ 1e-7 N) 인지
```bash
$PY $C --save MPC/logs/gpu/golden_A42.npz          # 기록 약 20 초 + CPU 재생. 243 MB, git 에 안 올림 (.gitignore)
```

**C. GPU float64** — 하나씩 (앞 단계가 통과해야 다음)
```bash
$PY $C --load MPC/logs/gpu/golden_A42.npz --device cuda --dtype float64 --backend casadi --linsys chol
$PY $C --load MPC/logs/gpu/golden_A42.npz --device cuda --dtype float64 --backend cusadi --linsys chol    # CusADi 커널 컴파일
$PY $C --load MPC/logs/gpu/golden_A42.npz --device cuda --dtype float64 --backend cusadi --linsys cudss   # cuDSS (논문 방식)
```
- 통과 기준: "U 전체" 최대 |Δ| ≤ 1e-6 N (노트북 2.6e-8 — GPU 연산 순서 차이 여유). 넘으면 B 처럼 tick 별로 어디서 튀는지 찾는다
  (Q110 §6 의 방법: 기록 순서대로 OSQP 를 다시 돌려 반복 수 k 별 대조).

**D. GPU float32** — 오차 기록만 (판정은 닫힌 고리에서, 사용자와 상의)
```bash
$PY $C --load MPC/logs/gpu/golden_A42.npz --device cuda --dtype float32 --backend cusadi --linsys cudss
$PY $C --load MPC/logs/gpu/golden_A42.npz --device cuda --dtype float32 --backend cusadi --linsys chol
```

**E. 시간** — 환경 수별 표 (조립 · ADMM · 합 · µs/환경 · GPU 메모리)
```bash
for L in cudss chol; do $PY $C --load MPC/logs/gpu/golden_A42.npz --noreplay --device cuda --dtype float32 --backend cusadi --linsys $L --bench 1,64,256,1024,2048,4096; done
$PY $C --load MPC/logs/gpu/golden_A42.npz --noreplay --device cuda --dtype float64 --backend cusadi --linsys chol --bench 1,64,256,1024
```

**F. 기록** — 결과를 Q111 로 (`Q&A/2026-10-06.md` 끝, 색인 행 12 끝에 이어 붙임) + 커밋. 푸시는 사용자 요청 시.
- 담을 것: 패키지 판 (A 의 출력), B ~ D 의 표, E 의 시간표, 생긴 문제와 고친 것 (코드 수정은 무엇을 왜).

## 4. 막힐 때 (예상 — 대부분 추론)

- **torch 확장 컴파일이 CUDA 판 불일치로 멈춤:** 시스템 nvcc 12.9 vs torch 12.8. 주 판이 같으면 경고만 내고 지나가는 것으로 안다 (추론).
  안 되면 `CUDA_HOME=/usr/local/cuda-12.9` 를 명시해 본다. conda 에 CUDA 툴킷을 따로 까는 것은 사용자와 상의 후.
- **sm_120 미지원 오류:** nvcc 12.8 이상이어야 한다 (12.9 라 될 것, 추론). [cusadi_lite/utils/jit_kernels.py](cusadi_lite/utils/jit_kernels.py) 의 `-gencode` 를 본다.
- **cuDSS:** `cudss.h` 를 못 찾음 → `nvidia-cudss-cu12` 설치 확인 (`python -c "import nvidia.cudss as c; print(c.__path__)"`).
  링크는 `libcudss.so` 가 없으면 `-l:libcudss.so.0` 식으로 실제 파일 이름을 쓴다 (`find_cudss`). 실행 중 `.so` 를 못 찾으면 rpath / `LD_LIBRARY_PATH`.
  "CUDSS call ended unsuccessfully … details: …" 가 나오면 어느 단계 (createMatrices / analysis / factorization / solve) 인지 그대로 보고.
- **float32 에서 `torch.linalg.cholesky` 실패 (양정치 아님):** 고치지 말고 보고 → 사용자와 상의.
- **casadi 가 numpy 판과 안 맞음:** 오류를 보고. casadi 판을 바꿔야 하면 상의 (CusADi 는 연산 번호를 casadi 상수 이름으로 쓴다 — 코드 사실 — 판이 달라도 되지만 `instruction_*` API 는 확인).
- 하지 말 것: 알고리즘 설정 (A42 · Ruiz 규칙 · ρ · warm start · 25 회) 을 결과에 맞춰 바꾸기, 다른 사람 환경 수정, sudo, 묻지 않은 푸시.

## 5. 그다음 (사용자와 정할 것)

- 닫힌 고리 (가): MuJoCo CPU 여러 개 + GPU MPC 일괄 — `controller.tick` 을 MPC 앞 · 뒤로 나눠 B 개 환경의 MPC 를 한 번에 부르는 구조가 필요.
- 닫힌 고리 (나): Isaac Lab 에 MIT 휴머노이드 (URDF `mit_humanoid_mjcf/urdf_source/humanoid_full_sf.urdf`) 들이기 + 컨트롤러 나머지 (FSM · 걸음표 · 스윙 · 다리 토크) torch 배치화.
- float32 채택 여부, 속도 최적화 (H = 2BᵀQB 의 블록 삼각 구조, CUDA graph).
- CLAUDE.md "옮긴 직후 대표 표를 다시 돌려 수치 기준을 새로 잡는다" — CPU suite 기준표를 이 컴퓨터에서 다시 잡을지.
