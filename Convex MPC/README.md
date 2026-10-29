# Convex MPC — MIT Humanoid (MuJoCo)

Di Carlo 식 SRB convex MPC 에 OA-MPC 선 접촉 발 제약을 더한 보행 제어기입니다. 로봇은 MIT Humanoid (MuJoCo) 입니다.
설정은 **A41** (MPC 최종 설정, 2026-10-29 이식, Q&A 10/6 Q76 · Q86 · Q88) 이고, 숫자는 전부 [mit_controller.yaml](mit_controller.yaml) 에 있습니다.
A41 은 10/1 발표 설정 A30 에서 10 가지를 바꾼 것입니다 (아래 "A30 에서 바뀐 것").
실행할 때는 **속도 명령과 gait (걷기 / 서기) 만** 고르면 됩니다.

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
| 전진 1.5 m/s (최고 속도) | `python convex_mpc.py --vx 1.5` |
| 제자리 회전 2.5 rad/s | `python convex_mpc.py --wz 2.5` |
| 제자리 회전 3.0 rad/s (최고) | `python convex_mpc.py --wz 3.0` |
| 걸으며 돌기 | `python convex_mpc.py --vx 0.6 --wz 1.0` |
| 제자리 서기 | `python convex_mpc.py --gait stand` |
| 화면 없이 120 s 판정 | `python convex_mpc.py --vx 1.0 --headless` |
| 영상으로 저장 (MP4) | `python convex_mpc.py --vx 1.0 --seconds 15 --record vx10.mp4` |

`python` 자리에는 가상환경의 파이썬을 씁니다 (예: `.venv/bin/python`). 어느 폴더에서 실행해도 됩니다.

- 걷기는 시작 후 1 s 동안 서서 자세를 잡고, 이어서 2 s 동안 명령 속도까지 올라갑니다.
- 뷰어는 기본 120 s 동안 돌고, 넘어지면 멈춥니다.
- Ubuntu Wayland 에서는 뷰어용 `PYGLFW_LIBRARY_VARIANT=x11` 이 자동으로 붙습니다.

### 영상 저장 (`--record`)

뷰어를 띄우지 않고 MP4 파일로 바로 저장합니다. 화면 녹화보다 이 방법을 권합니다.

- 프레임을 시뮬 시간 기준으로 찍습니다 (기본 50 fps = 10 틱마다). 그래서 계산이 실시간보다 느려도 영상은 실제 속도로 재생됩니다.
- 로봇의 움직임은 `--headless` 판정과 끝자리까지 같습니다 (측정, 2026-10-01). 화면 그리기는 제어에 영향을 주지 않습니다.
- 카메라는 뷰어와 같은 추적 카메라입니다. 왼쪽 위에 시간, 명령 속도, 실측 속도 (전신 CoM, 0.5 s 평균), 외란 크기가 표시됩니다.
- 넘어지면 1 s 더 찍고 멈춥니다. H.264 MP4 라 PowerPoint 에서 바로 열립니다.

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--record 파일.mp4` | — | 저장할 파일. `--seconds` 가 길이입니다 (걷기는 앞에 서기 1 s 가 붙습니다) |
| `--size` | 1280x720 | 해상도. 예: `--size 1920x1080` |
| `--fps` | 50 | 프레임률 |
| `--nohud` | — | 글자 끄기 |
| `--clean` | — | 참조 표시 (스윙 궤적 · 착지 목표 · CoM 참조) 끄기 |

예시입니다.

```bash
python convex_mpc.py --vx 1.0 --seconds 15 --record vx10.mp4
python convex_mpc.py --wz 2.5 --seconds 15 --record wz25.mp4 --clean
python convex_mpc.py --gait stand --seconds 8 --push 30:0:3:0.2 --record stand_push.mp4
python convex_mpc.py --vx 0.5 --seconds 15 --push 0:30:6:0.2 --record walk_push.mp4
```

15 s 영상은 1280x720 기준 약 1 분 걸립니다. 외란 영상은 먼저 `--headless` 로 같은 명령이 버티는지 확인하고 찍는 것이 좋습니다.
걷는 중 옆 40 N 은 미는 시점에 따라 결과가 갈립니다. vx 0.7 에서 걷기 시작 3 · 6 · 10 s 뒤에 밀면 ✗ · ✓ · ✗ 였습니다 (측정).

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
| `--yawlead` | 1 | 회전 선행 0 / 0.5 / 1 (0 이면 제자리 2.5 rad/s 가 넘어집니다) |
| `--push` | — | 스크립트 외란 `Fx:Fy:t0:dur` [N, N, s, s]. 진행 방향 기준이고, 걷기 시작 (서기면 0 s) 뒤 t0 에 몸통 CoM 에 가합니다. 예: `--push 0:40:5:0.2`. 음수로 시작하면 `=` 로 붙여 씁니다: `--push=-30:0:5:0.2` |
| `--var` | — | 고급 옵션입니다. yaml 위에 변형 키를 덮어씁니다 (`k=v,k=v`, 문법은 `engine.parse_variant`) |

명령 상한은 전진 · 후진 1.5 m/s, 옆 1.0 m/s, 회전 3.0 rad/s 입니다. 이보다 큰 명령은 명령 필터가 잘라냅니다.
상한 안이라도 넘어지는 명령이 있습니다 (후진 −1.2, 옆 +0.7 · −0.8 등 — 아래 "보행 한계").

## 무엇이 들어 있나

- **MPC**: SRB 모델로 상태 13 개 · 입력 12 개 (두 발의 힘 3 개와 모멘트 3 개) 를 씁니다.
  지평은 22 스텝 × 0.02 s = 0.44 s 이고, 100 Hz 로 볼록 QP 를 한 번씩 풉니다 (OSQP, 걷기 · 서기 모두 ADMM 25 회).
  ρ 는 걷기 5e-5 · 서기 1e-5 이고, warm start 는 지난 해를 반 칸 (10 ms) 당긴 값입니다.
- **발 제약 (OA-MPC 선 접촉)**: Mx = 0, CoP |My| ≤ c·l·Fz, 비틀림 · 마찰 제약을 둡니다. 발 반길이 l = 0.075 m 입니다.
  MPC 해를 이 집합으로 한 번 더 잘라서 실제로 가하는 힘은 걷기 · 서기 모두 제약을 정확히 지킵니다.
- **걸음**: 주기 0.55 s, 스윙 0.22 s 이고, 착지점은 capture point 식 (k_half 0.41, 명령에 따라 0.35 ~ 0.47) 입니다.
- 저수준: 스윙 다리 임피던스 (감쇠비 0.5, 발 pitch 를 이륙 기울기에서 수평으로 옮기는 궤적), 딛는 발 yaw hold (관절 되먹임,
  더한 비틀림도 선 접촉 한계로 자름), 팔 반대 흔들기 (1.0 rad/m).
- A41 의 숫자는 전부 [mit_controller.yaml](mit_controller.yaml) 에 있습니다. 기본값에서 바꾼 줄에는 `# A30 <키> (원래 값)`,
  A30 → A41 에서 바꾼 줄에는 `# A41 <키> (A30 값)` 주석이 달려 있습니다.
  yaml 설정 경로가 없는 값 (profile · fix · foot · Raibert 이득 · ypath · 팔 흔들기 · OSQP 반복 수 · ρ · warm start) 은 맨 아래 `launcher:` 절에 있고, [convex_mpc.py](convex_mpc.py) 가 읽어서 넘깁니다.
  원본 폴더 (casadi_mit) 는 같은 값을 `--var` 문자열로 덮어씁니다. 원본 yaml 에는 바꾸기 전 기본값이 그대로 있습니다.

## A30 에서 바뀐 것 (A30 → A41)

| 단계 | 바꾼 것 (yaml 키) | 효과 (측정) |
|---|---|---|
| A33 | warm start 반 칸 (`mpc.warm_shift_steps` 0.5) · 스윙 감쇠비 1.2 → 0.8 · 스윙 발 pitch 궤적 (`swing.pitch_trajectory_fraction` 0.7) | 1.0 m/s 스윙 토크 포화 12.8 → 1.5 %, 걷기 옆 외란 16 → 21/32 |
| A34 | 옆 위치 가중치 1009300 → 100000 | 걷기 옆 외란 21 → 30/32 |
| A35 | pitch 가중치 9000 → 3000 | 뒤꿈치 되튐 사라짐, 최고 속도 1.10 → 1.26 m/s |
| A36 | 착지 yaw 선행 0.5 → 1 | 회전 때 발이 몸 회전에 뒤처지던 것 해소 |
| A37 | yaw hold 비틀림 제한 (`swing.stance_yaw_hold_torsion_clip` 1) | 다리에 보내는 렌치까지 OA-MPC 제약 위반 0 % (걸으며 돌 때 slip +3 ~ 6 mm) |
| A38 | 서기 ADMM 50 → 25 회 + 서기 전용 ρ 1e-5 (`launcher.osqp.rho_stand`) | 서기 외란 7/12 (정확해와 같음) |
| A39 | 스윙 감쇠비 0.8 → 0.5 | 1.4 m/s 120 s 완주 (A38 은 1.3 까지) |
| A41 | 팔 반대 흔들기 0.3 → 1.0 rad/m | 1.5 m/s 완주, 위몸 yaw 흔들림 1/3, 직진 slip −30 ~ 40 % |

A40 (스윙 높이 기준을 발끝으로) 은 보류해서 넣지 않았습니다. 10/1 발표 설정 (A30) 이 필요하면 원본 폴더에서 `--var A30` 으로 돌리거나 git 기록의 이 폴더를 씁니다.

## 보행 한계 (A41, 120 s 완주 기준, 측정 — Q&A 10/6 Q86)

한계는 시험한 값 가운데 통과한 가장 큰 값이고, 괄호 안은 넘어진 다음 단계입니다. 그 사이 값은 시험하지 않았습니다.

| 명령 | 한계 | 실측 |
|---|---|---|
| 전진 | 1.5 m/s (1.55 ✗) | 1.43 m/s |
| 후진 | −1.1 m/s (−1.2 ✗) | −1.07 m/s |
| 옆 | 왼 +0.65 (+0.7 ✗) · 오른 −0.7 (−0.8 ✗) m/s | 명령 그대로 |
| 제자리 회전 | 3.0 rad/s (3.25 ✗) | 3.00 rad/s |

| 전진 [m/s] | 함께 낼 수 있는 최대 회전 [rad/s] | 함께 낼 수 있는 최대 옆 속도 [m/s] |
|---|---|---|
| 0.6 | 2.5 (3.0 ✗) | 0.65 (그 위는 시험 안 함) |
| 1.0 | 1.25 (1.5 ✗) | 0.5 (0.65 ✗) |
| 1.3 | 1.0 (1.25 ✗) | 0.5 (0.65 ✗) |
| 1.5 | 0 (0.25 ✗) | 0.2 (0.3 ✗) |

- 외란 (몸통 수평력 0.2 s, Q&A 10/6 Q76): 서기 30 N 4/4 · 40 N 3/4 (앞 ✗) · 50 N 0/4 (발 디딤 회복 없음).
  걷기 0.6 m/s 옆 ±40 · ±50 N 31/32, 앞뒤 32/32. 1.3 m/s 30 N 16/16, 1.4 m/s 30 N 13/16.
- 다리끼리 충돌 (Q&A 10/6 Q87): 제자리 회전 2.5 · 3.0 rad/s 에서만 두 허벅지가 보행 주기마다 한 번꼴로 닿습니다 (30 s 에 55 회).
  직진 · 후진 · 옆 · 느린 회전 · 걸으며 돌기는 0 회입니다. 앞발 벌림 (`psioff`) 은 일부러 꺼 두었습니다 — residual RL 관찰 포인트.
- MPC 한 번 풀이는 평균 7.0 ms, 최대 8.7 ms 입니다 (Python 단일 스레드, Intel Core Ultra 5 125H).

## 원본과 같은지 확인 (측정, 2026-10-29)

전진 1.0 · 1.5, 제자리 3.0, 옆 0.5, 0.6 + 2.5, 후진 −1.1, 서기 + 앞 30 N, 0.6 + 옆 40 N 을 8 s 씩 돌렸습니다.
이 폴더와 원본 [10_walk.py](../MPC/src/casadi_mit/10_walk.py) (`--var A41,xmax=1.5,wmax=3.0`) 의 상태 · MPC 해 · 토크 · 발 위치 기록 64 개가 비트 단위로 같았습니다.

실행기 (`convex_mpc.py --headless`) 로 120 s 판정도 돌렸습니다. 8 개 모두 완주했고, 수치는 원본으로 잰 보행 한계 표 (Q86) 와 같습니다.

| 명령 | 결과 | 실측 | slip [mm/걸음] |
|---|---|---|---|
| `--vx 1.0` | 120 s 완주 | 1.01 m/s (101 %) | 8.3 |
| `--vx 1.5` | 120 s 완주 | 1.43 m/s (95 %) | 19.0 |
| `--vx -1.1` | 120 s 완주 | −1.07 m/s (98 %) | 37.4 |
| `--vy 0.65` · `--vy -0.7` | 120 s 완주 | +0.65 · −0.70 m/s | 4.9 · 6.2 |
| `--wz 3.0` | 120 s 완주 | 3.00 rad/s | 14.8 |
| `--vx 0.6 --wz 2.5` | 120 s 완주 | 0.53 m/s (94 %) · 2.50 rad/s | 16.4 |
| `--gait stand` | 120 s 완주 | — | — |

## 파일

| 파일 | 역할 |
|---|---|
| `convex_mpc.py` | 실행기 (속도 · gait 만 받음, yaml 의 `launcher:` 절을 읽음) |
| `engine.py` | 시뮬 루프, 뷰어, 헤드리스 판정, 변형 키 해석 (원본 `10_walk.py`) |
| `controller.py` | 컨트롤러 한 틱 (FSM → 걸음 계획 → MPC → 다리 토크) |
| `mpc.py`, `mpc_casadi.py` | SRB condensed MPC (CasADi 로 식 생성, OSQP 로 풀이), 선 접촉 제약 |
| `planner.py`, `gait.py`, `contact_manager.py`, `fsm.py`, `command.py` | 착지점 · 스윙 궤적, 걸음 시간표, 접촉 관리, 상태기계, 명령 필터 |
| `leg_control.py`, `mit_model.py` | 다리 토크 (스윙 임피던스 · 힘→토크 · yaw hold · 해 자르기), MuJoCo 모델 래퍼 |
| `mit_controller.yaml`, `config.py` | **A41 설정 전부** (바꾼 줄마다 원래 값 주석) 와 로더 |
| `viewer_hud.py`, `paths.py` | 뷰어 HUD, 폴더 안 경로 |
| `models/mit_humanoid/` | 로봇 모델 (scene.xml, mit_humanoid.xml, meshes_v3). 출처 · 라이선스는 그 안의 README 와 LICENSE_source |
| `NOTICE.md` | 컨트롤러 구조의 출처와 라이선스 |
