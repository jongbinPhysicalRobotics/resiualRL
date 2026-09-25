# MPC/src/mit — reference 컨트롤러의 MIT 휴머노이드 Python 이식 (2026-09-25)

reference (`reference/`, ispaik06/convex-mpc-biped, C++) 의 convex MPC 보행 컨트롤러를 **그대로** Python 으로 옮기고,
MIT 휴머노이드 모델 (`MPC/models/mit_humanoid/`) 에서 reference 데모 수치 (전진 0.6 m/s · 횡 0.3 m/s · 제자리 회전 1.3 rad/s)
를 120 s 로 재현해 보는 폴더. 이 폴더 파일만으로 실행된다 (G1 폴더들과 코드 공유 없음). 경위·결과는 [Q&A 9/25 Q2](../../../Q&A/2026-09-25.md).

## 실행 (Git Bash, 프로젝트 루트)

```bash
# 보기 (뷰어) — 지금 세 목표를 120 s 넘기는 구성: 명령 램프 + 방향 유지 (yawref)
.venv/Scripts/python.exe MPC/src/mit/10_walk.py --view --vx 0.6 --profile ramp --fix yawref
.venv/Scripts/python.exe MPC/src/mit/10_walk.py --view --vy 0.3 --profile ramp --fix yawref
.venv/Scripts/python.exe MPC/src/mit/10_walk.py --view --wz 1.3 --profile ramp --fix yawref
# 충실한 이식 그대로 (reference 헤드리스 기본: 명령이 처음부터 있음) — 세 목표 모두 수 초 안에 넘어진다
.venv/Scripts/python.exe MPC/src/mit/10_walk.py --view --vx 0.6
# 헤드리스 1 개 (요약 출력, --log 면 MPC/logs/mit_walk_*.npz)
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe MPC/src/mit/10_walk.py --vx 0.6 --profile ramp --fix yawref --seconds 120
# 표 (Pool, 18 코어)
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe MPC/src/mit/11_gait_quality.py --cmd 0.6,0,0 0,0.3,0 0,0,1.3 \
    --var base "profile=ramp,fix=yawref" --seconds 120 --md
```
변형 키 (`--var "k=v,k=v"`): `profile=const|step|ramp`, `fix=slide+yawref+yawanchor`, `solver=quadprog`, `noarm`,
`markers`, `mu=0.8`, `rhoint=50`, `maxit=4000`, `eps=1e-4`, `fsm2=0`. 검사: `01_check_model.py`, `02_unit_checks_{A,B,C}.py`, `03_integration_smoke.py`.

## 파일

| 파일 | 무엇 (reference 대응) |
|---|---|
| `mit_model.py` | 모델 로더 (한글 경로 VFS), 이름→인덱스, 상태 읽기 (cheater state, yaw unwrap), **축소 몸체** (몸통+팔 14.23 kg, 매 틱), 다리 J·M·bias·J̇q̇ (`setupRobotParams`, `MujocoCheaterStateReader`, `LegSwingDynamicsProvider`) |
| `config.py` · `mit_controller.yaml` | reference MIT yaml 그대로 + `port:` 절 |
| `command.py` · `fsm.py` | 명령 필터 (dt = 0 스냅 포함) · StandingSettle → Walking FSM |
| `gait.py` | HorizonClock (cycle 원점), GaitScheduler (지평 접촉표) |
| `planner.py` | body target, 착지점 계획, 스윙 궤적 (cubic smoothstep) |
| `contact_manager.py` | 조기 접촉 · 램프 · 지평 step 0 override |
| `mpc.py` | 참조 궤적, SRB 식, 제약 (마찰·CoP·비틀림, 발 yaw 회전), OSQP (shifted warm start, cold 재설정, fallback) |
| `leg_control.py` | stance Jᵀ wrench, 스윙 OSC (Kp = ω²·diag Λ), 발 자세, stance yaw hold, 서기 6×10 Jacobian, 팔 PD |
| `controller.py` | `runController` 17 단계 순서 그대로 + 변형 `fixes` |
| `10_walk.py` · `11_gait_quality.py` | 실행기 (헤드리스 / 뷰어) · 변형 표 |
| `specs/` | 이식의 근거 문서 — reference C++ 에서 뽑아 적대적 검증을 거친 명세 01~06, 설계 07 |

## reference 와 다른 점

**허용한 편차 (D1~D5, `specs/07_port_design.md`)**: D1 모델 = 공개 URDF 변환본 (+ 발 site, armature, dt 0.002) ·
D2 시작 = 키프레임 `stand` 에서 바로 서기 MPC (reference 의 2 s 관절 PD 초기화는 이 모델에서 넘어짐) ·
D3 헤드리스 명령 주입 (y·yaw 포함) · D4 다리 동역학을 전신 모델 블록으로 (별도 고정 기반 모델과 4e-15 일치) ·
D5 python osqp 1.1.3 (reference 는 OsqpEigen / osqp 0.6.x).

**모르는 것 (U1~U2)**: reference 의 MIT MJCF 는 비공개 — 디버그 마커 질량 (G1·H1 씬엔 0.21 kg), 관절 armature/damping, 접촉 파라미터.

**특이점 고침 (변형, 기본 꺼짐)** — `fixes`:
- `yawref` — walking 에서 MPC yaw 참조를 측정 yaw 대신 body target yaw (명령 적분) 로. **이것 하나로 세 목표가 120 s 를 넘는다.**
  reference 는 walking 에서 yaw 를 측정값으로 매번 다시 잡아 방향 되먹임이 없다 (서기 모드는 body target 을 쓴다).
  헤드리스에선 yaw 가 흘러가고 착지점 계획 (명령 적분 yaw) 과 몸 방향이 갈라져 넘어진다 (측정).
  reference 데모는 GUI 세션이고 횡·회전은 공개 코드상 키보드로만 낼 수 있다 (추론). 사람이 방향을 바로잡았는지는 모른다 (Q&A 9/25 Q3).
- `slide` — 지평 접촉표를 cycle 원점이 아니라 지금부터. 혼자 쓰면 접촉 패턴이 매 풀이 바뀌어 cold start (비율 0.9 이상) 라 OSQP 200 회 상한에서 나빠진다.
  정확한 풀이 (`maxit=4000` 또는 `solver=quadprog`) 와 같이 쓸 때만 조금 낫다.
- `yawanchor` — 계획 yaw 를 측정 yaw 로 (방향 유지 없이 흘러가는 대로). 전진·회전은 넘어진다.

## 결과 (120 s, 2026-09-25)

| 구성 | 전진 0.6 | 횡 0.3 | 회전 1.3 |
|---|---|---|---|
| 충실한 이식 (명령 처음부터, reference 헤드리스 기본) | ✗ 0.2 s | ✗ 0.2 s | ✗ 2.5 s |
| 충실한 이식 + 명령 램프 2 s | ✗ 22 s | ✗ 13 s | ✗ 5 s |
| **램프 + `yawref`** | **120 s, 88 %** (0.53 m/s) | **120 s, 70 %** (0.21 m/s) | **120 s, 100 %** (1.30 rad/s) |
| 램프 + `slide+yawref` + OSQP 4000 회 | 120 s, 92 % | 120 s, 79 % | 120 s, 100 % |

서기 10 s · 제자리 걸음 120 s 는 충실한 이식 그대로 통과. 회전에선 스윙 다리 hip abad (34 N·m) 가 스윙 시간의 34~62 % 포화
(reference 와 같은 ctrlrange 에서 clamp). 걸음 모양은 렌더로 확인할 것.
