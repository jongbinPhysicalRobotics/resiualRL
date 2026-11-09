# cusadi_mit — GPU 일괄 MPC (A42 를 환경 B 개 동시에)

`casadi_mit` 의 복사본 + GPU 일괄 풀이기. 알고리즘은 A42 그대로다 (condensed SRB QP · Ruiz 10 · ρ 5e-5 / 1e-5 · 고정 25 회 ADMM ·
반 칸 warm start). 경위 · 측정은 [Q&A 10/6 Q110](../../../Q&A/2026-10-06.md).
실험실 컴퓨터의 Claude 에게 넘기는 작업 지시 · 규칙은 [LAB_HANDOFF.md](LAB_HANDOFF.md).

| 파일 | 무엇 |
|---|---|
| [batch_mpc.py](batch_mpc.py) | `BatchMPC.from_cpu(cpu_mpc, B, device, dtype, backend, linsys)` — QP 조립 (torch) + OSQP 1.1.3 고정 반복 재현 |
| [cudss_kkt.py](cudss_kkt.py) | 전체 KKT 배치 LDL — `CudssKKT` (cuDSS, GPU) · `DenseKKT` (같은 패턴의 CPU 대역, 검증용) |
| [cusadi_lite/](cusadi_lite/) | CusADi (wip `4003888`, MIT) 의 코드 생성 · cuDSS 연결부만. 수정점은 [`__init__.py`](cusadi_lite/__init__.py) · [`utils/jit_kernels.py`](cusadi_lite/utils/jit_kernels.py) 머리말 |
| [20_gpu_check.py](20_gpu_check.py) | CPU 판과 대조 (기록 → 재생) · 시간 재기 |
| 나머지 | `casadi_mit` 복사본 (2026-11-09, yaml 은 커밋된 판) — CPU 판 기준 · 닫힌 고리용 |

선택지:
- `backend`: `casadi` (칸별 기호식을 CasADi `.map` 으로, CPU) / `cusadi` (같은 SX 식의 CUDA 커널, GPU)
- `linsys`: `chol` (축약형 K = P̂ + σI + ÂᵀρÂ 배치 Cholesky, torch) / `cudss` (전체 KKT LDL, Residual MPC 방식) / `kktcpu` (cudss 와 같은 식의 dense LU, 검증용)
- `dtype`: `float64` (검증) / `float32` (속도 — RTX 소비자용은 FP64 가 매우 느리다)

## 실험실 컴퓨터 설치 (sudo 없이, 원본 환경은 건드리지 않음)

```bash
# 1. Isaac Lab 환경을 복제해 우리 전용 환경으로 (원본 env_isaaclab 은 그대로)
conda create -n cmpc --clone env_isaaclab -y
conda activate cmpc
# 2. cuDSS (CUDA 12 판 — torch 가 cu128 이고 시스템 nvcc 가 12.9) + 빌드 도구
python -m pip install "nvidia-cudss-cu12==0.7.1.6" ninja
# 3. 우리 코드가 쓰는 패키지 — 이미 있는 패키지 판은 고정 (Isaac Sim 이 쓰는 numpy 등을 바꾸지 않게)
python -m pip list --format=freeze > /tmp/cmpc_pins.txt
python -m pip install -c /tmp/cmpc_pins.txt casadi==3.8.0 osqp==1.1.3 quadprog PyYAML glfw
# 4. 코드
git clone https://github.com/jongbinPhysicalRobotics/resiualRL.git ~/jongbin/residualRL   # 이미 있으면 git pull
cd ~/jongbin/residualRL
```

## 실행 순서 (실험실)

```bash
export OPENBLAS_NUM_THREADS=1
C=MPC/src/cusadi_mit/20_gpu_check.py
# (a) CPU 판 기록 + CPU 재생 — 노트북 결과 (최대 2.6e-8 N) 와 같은 자릿수인지. 기록은 약 20 초, 243 MB (git 에 안 올림)
python $C --save MPC/logs/gpu/golden_A42.npz
# (b) GPU — torch 만 (칸별 기호식은 CPU CasADi)
python $C --load MPC/logs/gpu/golden_A42.npz --device cuda --dtype float64 --backend casadi --linsys chol
# (c) + CusADi 커널 (처음엔 컴파일 수십 초)
python $C --load MPC/logs/gpu/golden_A42.npz --device cuda --dtype float64 --backend cusadi --linsys chol
# (d) + cuDSS (Residual MPC 방식)
python $C --load MPC/logs/gpu/golden_A42.npz --device cuda --dtype float64 --backend cusadi --linsys cudss
# (e) float32 — 오차가 얼마나 커지는지
python $C --load MPC/logs/gpu/golden_A42.npz --device cuda --dtype float32 --backend cusadi --linsys cudss
python $C --load MPC/logs/gpu/golden_A42.npz --device cuda --dtype float32 --backend cusadi --linsys chol
# (f) 시간 — 환경 수별 (재생 없이)
python $C --load MPC/logs/gpu/golden_A42.npz --noreplay --device cuda --dtype float32 --backend cusadi --linsys cudss --bench 1,64,256,1024,2048,4096
python $C --load MPC/logs/gpu/golden_A42.npz --noreplay --device cuda --dtype float32 --backend cusadi --linsys chol  --bench 1,64,256,1024,2048,4096
python $C --load MPC/logs/gpu/golden_A42.npz --noreplay --device cuda --dtype float64 --backend cusadi --linsys chol  --bench 1,64,256,1024
```

표의 열: `P · q · A · u` = QP 조립 차이, `풀이(같은 QP)` = CPU 가 OSQP 에 넘긴 QP 를 그대로 풀었을 때의 차이 (풀이기만),
`U 전체` · `u0` = 입력 → 해 전체의 차이. 각 칸은 `최대|Δ| [N 또는 N·m] / 상대 (최대|Δ| / 최대|U|)`.

## 아직 안 된 것

- GPU 에서 한 번도 안 돌렸다 (노트북에 GPU 없음) — CusADi 컴파일 · cuDSS 링크 · sm_120 은 실험실에서 처음 확인.
- 닫힌 고리: Isaac Lab 에 MIT 휴머노이드 들이기 + 컨트롤러 나머지 (FSM · 걸음표 · 스윙 · 다리 토크) 배치화는 다음 단계.
- 비유한 해가 나온 환경: CPU 판은 예외 → fallback 하고 workspace 를 유지, 배치판은 그 환경의 ADMM 상태를 버린다 (근사).
