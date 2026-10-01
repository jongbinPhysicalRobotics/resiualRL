# Convex MPC — MIT Humanoid (MuJoCo)

Di Carlo 식 SRB convex MPC 에 OA-MPC 선 접촉 발 제약을 더한 보행 제어기입니다. 로봇은 MIT Humanoid (MuJoCo) 입니다.
설정은 **A30** (2026-10-01 발표 설정, Q&A 9/30 Q23) 으로 고정되어 있어서 **속도 명령과 gait (걷기 / 서기) 만** 고르면 됩니다.

이 폴더는 복사해서 다른 곳에 두어도 그대로 돌아갑니다. 로봇 모델 (`models/mit_humanoid/`) 과 설정 파일이 모두 폴더 안에 있습니다.
코드는 [MPC/src/casadi_mit](../MPC/src/casadi_mit/) 의 복사본이고, 디버깅과 실험은 원본 폴더에서 합니다.
이 폴더를 고쳐도 원본에는 반영되지 않고, 원본을 고쳐도 이 폴더에는 반영되지 않습니다.

## 설치

Python 3.12 이상이 필요합니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

이 저장소 안이라면 최상위 `.venv` 를 그대로 써도 됩니다 (`../.venv/bin/python`).

## 실행

| 하고 싶은 것 | 명령 |
|---|---|
| 전진 1.0 m/s | `python convex_mpc.py --vx 1.0` |
| 뒤로 1.0 m/s | `python convex_mpc.py --vx -1.0` |
| 옆 걸음 0.5 m/s | `python convex_mpc.py --vy 0.5` |
| 제자리 회전 2.5 rad/s | `python convex_mpc.py --wz 2.5` |
| 걸으며 돌기 | `python convex_mpc.py --vx 0.6 --wz 1.0` |
| 제자리 서기 | `python convex_mpc.py --gait stand` |
| 화면 없이 120 s 판정 | `python convex_mpc.py --vx 1.0 --headless` |

`python` 자리에는 가상환경의 파이썬을 씁니다 (예: `.venv/bin/python`). 어느 폴더에서 실행해도 됩니다.

- 걷기는 시작 후 1 s 동안 서서 자세를 잡고, 이어서 2 s 동안 명령 속도까지 올라갑니다.
- 뷰어는 기본 120 s 동안 돌고, 넘어지면 멈춥니다.
- Ubuntu Wayland 에서는 뷰어용 `PYGLFW_LIBRARY_VARIANT=x11` 이 자동으로 붙습니다.

### 뷰어 조작

| 키 | 동작 |
|---|---|
| ↑ ↓ ← → | 몸통을 앞 · 뒤 · 왼 · 오른쪽으로 0.2 s 밀기 (몸 진행 방향 기준, 빨간 화살표가 표시됨) |
| PageUp / PageDown | 미는 힘 ±10 N (기본 30 N) |

화면의 선과 구는 MPC 가 쓰는 참조입니다. 초록 · 파랑 선은 왼발 · 오른발 스윙 궤적, 빨간 구는 착지 목표, 하늘색 선은 몸 CoM 참조 지평 (0.44 s) 입니다.
`--clean` 을 붙이면 이 표시를 끕니다.

### 옵션

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--vx` `--vy` `--wz` | 0 | 전진 [m/s] · 옆 (왼쪽 +) [m/s] · 회전 (반시계 +) [rad/s] |
| `--gait` | walk | `walk` 걷기 / `stand` 제자리 서기 |
| `--seconds` | 120 | 걷기 (서기) 시간 [s]. 완주 판정 기준이 120 s 입니다 |
| `--headless` | — | 뷰어 없이 실행하고 PASS / FAIL 과 지표를 출력합니다 (종료 코드 0 = PASS) |
| `--clean` | — | 뷰어의 참조 표시 끄기 |
| `--yawlead` | 0.5 | 회전 선행 0 / 0.5 / 1 |
| `--push` | — | 스크립트 외란 `Fx:Fy:t0:dur` [N, N, s, s]. 진행 방향 기준이고, 걷기 시작 (서기면 0 s) 뒤 t0 에 몸통 CoM 에 가합니다. 예: `--push 0:40:5:0.2` |
| `--var` | — | 고급 옵션입니다. A30 위에 변형 키를 덮어씁니다 (`k=v,k=v`, 문법은 `engine.parse_variant`) |

명령 상한은 전진 · 후진 1.0 m/s, 옆 1.0 m/s, 회전 2.5 rad/s 입니다. 이보다 큰 명령은 명령 필터가 잘라냅니다.

## 무엇이 들어 있나

- **MPC**: SRB 모델로 상태 13 개 · 입력 12 개 (두 발의 힘 3 개와 모멘트 3 개) 를 씁니다.
  지평은 22 스텝 × 0.02 s = 0.44 s 이고, 100 Hz 로 볼록 QP 를 한 번씩 풉니다 (OSQP, 걷기 ADMM 25 회 · 서기 50 회).
- **발 제약 (OA-MPC 선 접촉)**: Mx = 0, CoP |My| ≤ c·l·Fz, 비틀림 · 마찰 제약을 둡니다. 발 반길이 l = 0.075 m 입니다.
  MPC 해를 이 집합으로 한 번 더 잘라서 실제로 가하는 힘은 걷기 · 서기 모두 제약을 정확히 지킵니다.
- **걸음**: 주기 0.55 s, 스윙 0.22 s 이고, 착지점은 capture point 식 (k_half 0.41, 명령에 따라 0.35 ~ 0.47) 입니다.
- 저수준: 스윙 다리 임피던스, 딛는 발 yaw hold (관절 되먹임), 팔 반대 흔들기.
- A30 의 변형 키 전체는 [convex_mpc.py](convex_mpc.py) 의 `A30` 문자열에 있습니다. 키마다의 설명은 [원본 README](../MPC/src/casadi_mit/README.md) 를 보세요.

## 시험 결과와 알려진 한계 (A30, 120 s, 측정 — Q&A 9/30 Q22 · Q23)

A30 은 A29 에서 앞발 벌림만 끈 설정입니다. 벌림은 회전 명령에만 걸리므로 (코드 사실) 직진 · 옆 걸음 · 외란 수치는 A29 의 측정값입니다.

| 명령 | 결과 |
|---|---|
| 전진 0.7 ~ 1.0 · 후진 −1.0 | 완주. 실측 속도는 1.0 명령에 0.99 m/s, −1.0 명령에 −0.97 m/s |
| 옆 ±0.5 · 0.8 | 완주 |
| 제자리 회전 1.3 ~ 2.5 rad/s (선행 0.5 · 1) | 완주 |
| 걸으며 돌기 0.3 · 0.6 · 0.7 + 0.5, 0.3 · 0.6 + 1.0 | 완주 (slip 4 ~ 14 mm) |
| **제자리 2.5 rad/s, 선행 0** | **11.2 s 에 넘어짐** |
| **빠르게 걸으며 돌기 0.8 · 1.0 m/s + 1.0 rad/s** | **넘어짐.** 걸음마다 yaw 흔들림이 비대칭이라 바깥 스윙 hip abad 가 포화됩니다 |
| 외란 (걷기, 몸통 0.2 s) | 앞뒤 50 N 까지 버팁니다. 옆 40 N 은 5/8, 50 N 은 2/8 버팁니다 |
| 외란 (서기) | 30 N 은 모든 방향에서 버팁니다. 40 N 앞 · 뒤는 넘어집니다 (선 발 CoP 범위 ±6.4 cm, 발 디딤 회복 없음) |

- 회전할 때 무릎이 가까워집니다 (무릎 중심 간격 최소 약 7.5 cm, 정강이 반지름 3.5 cm).
  예전에 쓰던 앞발 벌림 (`psioff`) 을 일부러 껐습니다. residual RL 이 무릎 충돌을 없애는지를 관찰 포인트로 남기기 위해서입니다.
- 이 폴더의 결과가 원본과 같은지 확인했습니다 (측정, 2026-10-01).
  전진 1.0, 제자리 2.5, 옆 0.8, 서기를 12 s 씩 돌렸을 때 상태 · 토크 · 힘 기록이 원본 [10_walk.py](../MPC/src/casadi_mit/10_walk.py) 와 비트 단위로 같았습니다.
  다른 것은 풀이 시간 (벽시계) 뿐이었습니다. 폴더를 저장소 밖으로 복사해서 돌렸을 때도 같았습니다.

## 파일

| 파일 | 역할 |
|---|---|
| `convex_mpc.py` | 실행기 (A30 고정, 속도 · gait 만 받음) |
| `engine.py` | 시뮬 루프, 뷰어, 헤드리스 판정, 변형 키 해석 (원본 `10_walk.py`) |
| `controller.py` | 컨트롤러 한 틱 (FSM → 걸음 계획 → MPC → 다리 토크) |
| `mpc.py`, `mpc_casadi.py` | SRB condensed MPC (CasADi 로 식 생성, OSQP 로 풀이), 선 접촉 제약 |
| `planner.py`, `gait.py`, `contact_manager.py`, `fsm.py`, `command.py` | 착지점 · 스윙 궤적, 걸음 시간표, 접촉 관리, 상태기계, 명령 필터 |
| `leg_control.py`, `mit_model.py` | 다리 토크 (스윙 임피던스 · 힘→토크 · yaw hold · 해 자르기), MuJoCo 모델 래퍼 |
| `mit_controller.yaml`, `config.py` | 기본 설정 (A30 은 실행기에서 이 위에 덮어씀) |
| `viewer_hud.py`, `paths.py` | 뷰어 HUD, 폴더 안 경로 |
| `models/mit_humanoid/` | 로봇 모델 (scene.xml, mit_humanoid.xml, meshes_v3). 출처 · 라이선스는 그 안의 README 와 LICENSE_source |
