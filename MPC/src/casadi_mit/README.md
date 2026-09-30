# MPC/src/casadi_mit — MIT 재작업 (CasADi 형식, 2026-09-26)

MIT 휴머노이드 재작업의 본 폴더. mit 폴더 (reference C++ 의 충실한 이식) 에서 출발하되,
**MPC 공식을 CasADi 기호식 + 자동미분으로** 만든다 (`mpc_casadi.py`, mit 이식과 행렬 1e-11 · 폐루프 동일 — Q&A 9/26 Q2).
앞으로의 수정 (제약 교체·항 추가·Residual MPC 이행) 은 이 폴더에서 한다. 이 폴더 파일만으로 실행된다.

## mit 폴더와 다른 점

| | mit (이식 — 동결) | **casadi_mit (재작업)** |
|---|---|---|
| MPC 공식 | 손 채움 numpy | **동역학 f(x,u) 기호식 + `ca.jacobian`** (condensing·OSQP 층은 공통) |
| OSQP 기본값 | 100 / 0 (명세 추론값 — 기록 보존) | **reference 실효 50 / 1** (9/25 Q11) |
| 기본 명령 | const, fix 없음 (reference 그대로) | **ramp + yawref** (안정 구성 — `--fix ""` 로 끔) |
| 발 접촉 모델 | ref 고정 | ref 기본 + `--foot line` (OA-MPC 선 접촉, **검증 전** — 9/26 Q2 §4) |
| GAIT_ON | 없음 (`--stand` 플래그만) | **`gait.py` 파일 값 1 (걷기), 환경변수 우선** — 0 = 제자리 서기 (FSM standing) |

## 실행 (프로젝트 루트, HiDPI Wayland 뷰어는 앞에 `PYGLFW_LIBRARY_VARIANT=x11`)

```bash
# 걷기 (기본: 0.6 m/s 램프 + yawref) — 뷰어
PYGLFW_LIBRARY_VARIANT=x11 .venv/bin/python MPC/src/casadi_mit/10_walk.py --view --vx 0.6
# 서 있기
GAIT_ON=0 PYGLFW_LIBRARY_VARIANT=x11 .venv/bin/python MPC/src/casadi_mit/10_walk.py --view
# 헤드리스 120 s 판정
.venv/bin/python MPC/src/casadi_mit/10_walk.py --vx 0.6 --seconds 120
# 변형 키를 뷰어 · 헤드리스에 바로 (11_gait_quality 와 같은 문법, 플래그보다 우선) — 9/28
GAIT_ON=1 PYGLFW_LIBRARY_VARIANT=x11 .venv/bin/python MPC/src/casadi_mit/10_walk.py --view --wz 1.3 --rai 0.2 --ypath 0.3 \
    --var yawlead=0,rho=1e-5
# 입각 발 진단 (뒤꿈치 회전 · CoP 포화 · 한 점 접촉 — 9/28 Q1)
GAIT_ON=1 .venv/bin/python MPC/src/casadi_mit/13_foot_diag.py --vx 0.65 --profile
# 표 (세 목표)
.venv/bin/python MPC/src/casadi_mit/11_gait_quality.py --cmd 0.6,0,0 0,0.3,0 0,0,1.3 \
    --var "profile=ramp,fix=yawref" --seconds 120
```

변형 키 (`--var "k=v,k=v"`): mit 폴더와 동일 + `foot=line|mx0`, `khalf=0.37`, `xmax=1.0`,
`rai=0.2` (착지 측정속도, 9/26 Q8), `ypath=0.3` (옆 경로선, 9/27 Q2), `fix=slide+yawref` (9/27 Q3),
`armsw=k` (팔 반대 흔들기 rad/m), `cop=a` (MPC CoP 앞뒤 한계), `wmom=w` (발 모멘트 입력 가중치),
`qpitch/qyaw/qroll=w` (걷기 자세 가중치) — 9/28 Q1. 모두 기본값 = 기존 동작.
풀이기 (9/28 Q5): `admm=N` (논문식 고정 반복 ADMM, ρ 고정 · 판정 / 재시도 없음), `init=nominal|warm|hold|zero` (hold = 당기지 않는 warm — 9/28 Q8 에서 warm 보다 나쁨), `rho=ρ` (OSQP ρ, 기본 0.1 — 이 문제엔 1e-5 가 맞다).
회전 (9/28 Q6): `yawlead=s` (착지 yaw 선행 배율, 기본 1.0 — 0 이면 정확해도 회전 완주), `yawhold=0|1`, `cone=pyramidal|elliptic` (시뮬 마찰 원뿔).
검증 이력·비교 도구는 mit 폴더 (`15_casadi_check.py`, `12_ref_diff.py`) 에 있다.

## 기준 수치 (측정, 이 폴더 기본 설정)

- 걷기 0.6 m/s 20 s: 추종 85.2 %, fallback 0 (120 s 는 CasADi 판으로 84.5 % — 9/26 Q2 §2)
- 서기 (GAIT_ON=0) 10 s: vx 0.001 m/s, roll σ 0.0002 rad

## 현재 후보 설정 (9/28 Q8, 기본값 아님 — 렌더 확인 대기)

`--rai 0.2 --ypath 0.3 --var rho=1e-5,init=warm,admm=25,khalf=0.37,yawlead=0.5` — 0.6 / 0.65 / 0.7 전진 · 옆 0.3 · 회전 1.3 모두 120 s 완주 (실측 0.69 m/s), 풀이 고정 25 회.

후보 B2 (9/28 Q12, 걸음 품질): `--fix slide+yawref --foot line --rai 0.2 --ypath 0.3 --var rho=1e-5,init=warm,admm=25,khalf=0.2,khalfy=0.37,yawhold=0,cop=0.05,yawlead=0` — 한 점 접촉 11 %, slip 2 mm, 걸으며 돌기 0.3+0.5 완주. 새 변형: `pmin=f` (두 점 최소 수직력), `khalfy=k` (옆 방향 k_half).

스윙 발 yaw (9/29 Q3): `yawtraj=1.0,yawff=2` 를 두 후보에 더하면 회전 선행 무관 · 걸으며 돌기 0.3+0.5 까지 완주. `yawkp`, `yawkd` 도 변형 키.

**후보 A4 (9/29 Q8, 최신)**: `--var profile=ramp,fix=yawref,ypath=0.3,rho=1e-5,init=warm,admm=50,khalf=0.37,yawtraj=1.0,yawff=2,psioff=0,rai=0.2,vdir=0.7,yawlead=0.5`
— 옆 0.3 · 회전 1.3 (선행 0/0.5/1) · 1.5 · 걸으며 돌기 0.3~0.6+0.5 · 0.3+1.0 · 전진 0.65~0.7 모두 120 s 완주 (고정 50 회 — 잔차 학습 쪽 후보).
⚠ 0.75 이상은 `xmax=1.0` 을 줘야 한다 (yaml x_dot_max 0.7 로 잘림 — Q&A 9/29 Q9).

**후보 A27 (10/1 Q21, 최신 · 긴 걸음 주 후보)**: A24 + `cmarw=0.1,psioff=1.5` — 걷기에서만 두 점 최소 하중 0.1 (서기는 0.075 로 서기 외란 유지),
앞발 벌림 1.5 로 제자리 회전 선행 0 여유. 기본 18 + 옆 ±0.5 · 옆 0.8 · 후진 −1.0 120 s 완주, 1.0 실측 0.98, 1.0 떨림 16 회/s (A24 29). 1.0 + 1.0 ✗.

**후보 A24 (10/1 Q20)**: A23 + `ksx=0.35,ksy=0.47` — 곧게 빨리 걸을 땐 전진 k_half 를 0.35 쪽, 옆으로 걸을 땐 옆 k 를 0.47 쪽,
회전 명령이 있으면 0.41. 기본 18 + 옆 ±0.5 · 옆 0.8 · 후진 −1.0 완주, 전진 1.0 실측 0.98 · slip 22, 옆 0.8 slip 36. 곧게 빨리 걸을 때 떨림이 A17 수준.

**후보 A23 (9/30 Q18 · Q19, 긴 걸음)**: A20 에서 `tcycle=0.55,tswing=0.22,khalf=0.41,psioff=1` — 스윙 0.17 → 0.22 s (이중 지지 0.11 s 남김),
제자리 회전 선행 0 은 앞발 벌림 (psioff=1) 으로. 기본 18 + 옆 ±0.5 · 옆 0.8 · 후진 −1.0 120 s 완주 (1.0 + 회전 1.0 만 ✗). 전진 0.8 · 0.9 slip 31 · 40 mm (A20 의 3 ~ 4 배).
외란 · 떨림은 아직. 확장 명령 시험엔 `xmax=1.3,ymax=1.0`.

**변형 A22 (9/30 Q15)**: A20 + `uint=0.5` — 풀이 사이 u0 → u1 보간 (기울기 0.5). 떨림 · 직진 속도 개선 (1.0 실측 0.93, 켜짐 전환 18 → 13 회/s), 18/18.
옆 ±0.5 ✗ · 옆 외란 약함 → 전진 · 회전 위주 시연용. `uintm=2` (모멘트만 보간) 은 옆 걸음은 살지만 이득도 사라짐.

**후보 A20 (9/30 Q14, 최신 · 주 후보)**: A18 에서 `shabs=1,admm=50,lproj=3,minw=0.1` — MPC 해가 OA-MPC 선 접촉 제약을 정확히 만족 (ADMM 50 + 해 자르기,
위반 틱 0 %), yaw hold 감쇠는 발의 절대 yaw 각속도로, 옆 걸음 발 겹침 방지. 18/18 + 확장 8 개 (후진 −1.0, 옆 ±0.5, 조합) 120 s 완주.
전진 1.0 실측 0.89, 풀이 7.1 ms (100 Hz 의 71 %). yaw hold 는 MPC 밖 (논문 식 (21) 의 관절 PD 자리). 옆 50 N 외란 3/8.

**후보 A18 (9/30 Q12)**: A17 에서 `cmar=0.075` — 선 접촉 두 점이 각각 수직력의 7.5 % 이상 (CoP 행만 (1 − 2ε)·l). 발 떨림 · slip 감소
(전진 1.0: 켜짐 전환 29 → 23 회/s, slip 18 → 14 mm, 실측 0.91), 18/18 완주. 걷기 옆 외란 40 N 5/8 · 50 N 4/8 (A17 보다 한 칸 나쁨).
새 키: `cmar` (채택), `wdu` (입력 변화 비용 — 기각), `fpd` (딛는 발 pitch 감쇠 — 보류). 진단 `diag/jitter_diag.py` · `jitter_ana.py`.

**후보 A17 (9/30 Q11)**: A16 에서 `foot=line,cop=0.075,qyaw=750` — MPC 발 제약을 OA-MPC 선 접촉 그대로 (Mx = 0 등식, CoP, 두 점 y 마찰에서
유도한 비틀림 4 행). 발 반길이 0.075 = 시뮬 발의 실제 두 접촉점 (측정). 18/18 완주, 전진 1.0 실측 0.89, 외란 A16 과 비슷 (옆 40 N 6/8).
A16 까지는 reference 발 (`foot=ref`, |Mx| ≤ 0.01·Fz 허용) 이었다. 새 키 `lproj=1|2` (최종 wrench 투영 · 자르기) 는 시험만 하고 채택 안 함.

**후보 A16 (9/30 Q10)**: A14 에서 `armsw=0.3,shkp=60,shkd=12,shw=2.0,wmax=2.5` — 팔을 예전 크기로 되돌리고, 딛는 발 yaw hold 를
60 / 12 로 세게 하되 회전 명령 |ψ̇| 이 2.0 에 가까워질수록 20 / 4 로 푼다. 18/18 완주 (걷기 14 + 회전 1.3 · 2.5 선행 0/0.5/1), 전진 1.0 실측 0.94.
새 키: `shkp` · `shkd` (딛는 발 yaw hold 이득), `shw` (회전에 따라 hold 풀기), `wmax` · `ymax` (명령 상한).

**후보 A14 (9/30 Q8)**: A13 에서 `kdz=1.2` — 스윙 감쇠비 1.2 로 고속 좌우 쏠림 해결 (1.0 m/s R/L 0.96, vy 0.02).
15/15 완주, 전진 1.0 명령 실측 0.94 m/s, 걷기 외란 앞뒤 50 N · 옆 40 N 전 위상. slip 16 ~ 18 mm.
회전 상한 풀기 `wmax=10` (yaml psi_dot_max 2.0): 제자리 3.0 rad/s (선행 ≥ 0.5), 걸으며 돌기 0.6+1.0 · 0.5+1.5 완주 (9/30 Q9). 옆 상한은 `ymax`.

**후보 A13 (9/30 Q6)**: A11 + `swwn=0.46,kdz=0.7,qyaw=2000` — 스윙 PD 를 부드럽게 (ω 0.46 배) + 감쇠비 0.7, MPC yaw 가중치 2000.
전진 1.0 명령 실측 0.96 m/s, 15/15 완주 (14 목표 + 1.0), 걷기 외란 옆 40 N 8/8 · 50 N 6/8. slip 15 mm, 고속 하중비 0.77 ~ 0.81.
새 키: `swwn` (스윙 고유진동수 배율), `swlim` (토크 한계 반영 스윙, 1 비율 · 2 QP — 역효과), `swprof`, `rampT` 프로파일.

**후보 A11 (9/30 Q1)**: A10 에서 `armsw=1.0` — 팔로 스윙 다리 yaw 반동을 더 상쇄해 앞꿈치 축 회전 (slip) 을 절반으로 (0.8: 15.1 → 7.7 mm). 14/14 완주, 외란 A10 이상.

**후보 A10 (9/29 Q12)**: A8 + `rvwb=1,replan=1,rpfz=0.08` — 착지점 되먹임 속도 = 전신 CoM 속도, 스윙 중 매 틱 착지점 재계산 (끝 80 ms 고정).
14/14 완주, 직진 0.6 외란 앞뒤 50 N · 옆 30 N 전 위상 (0.2 s). slip 은 A8 의 약 2 배. 새 키: `minw` (최소 보폭 폭, 역효과), `wf` (수평력 입력 가중).

**후보 A8 (9/29 Q11)**: A7 에서 `fix=half+yawref,khalf=0.3` — 접촉표 원점을 반 주기마다 옮겨 곧은 걸음 좌우 비대칭 (vy) 해소, 14/14 완주.
외란 시험 키: `push=Fx:Fy:t0:dur` (몸통 CoM, 몸 yaw 기준 N, 측정 창 시작 뒤 t0 초, dur 초), `pushrep=T` (T 초마다 반복), `stand=1` (서기).
뷰어 (`--view`) 외란: 같은 `--var push=…` 스크립트 + 키보드 ↑ 앞 · ↓ 뒤 · ← 왼 · → 오른 (0.2 s, 기본 30 N), PageUp/PageDown ±10 N.
미는 동안 몸통에 빨간 화살표 (1 cm/N), HUD 에 크기 · 상태.

**후보 A7 (9/29 Q10, 고정 25 회 · 10 ms — 잔차 학습 쪽)**: `--var profile=ramp,fix=yawref,ypath=0.3,init=warm,khalf=0.37,yawtraj=1.0,yawff=2,psioff=0,rai=0.2,vdir=0.7,admm=25,nbs=5,rho=5e-5,hor=0.44,fref=1e-3,armsw=0.3,xmax=1.0,yawlead=0.5`
— A6 와 같은 14 개 목표 전부 120 s 완주 (전진 0.8 실측 0.73, 걸으며 돌기 0.7+0.5). `nbs` = 몇 틱마다 MPC 를 푸나 (기본 7), `tors` = 비틀림 계수.

**후보 A6 (9/29 Q9, 수렴 OSQP)**: `--var profile=ramp,fix=yawref,ypath=0.3,rho=1e-5,init=warm,khalf=0.37,yawtraj=1.0,yawff=2,psioff=0,rai=0.2,vdir=0.7,xmax=1.0,hor=0.44,fref=1e-3,armsw=0.3,yawlead=0.5`
(수렴 OSQP) — 위 목표 전부 + 걸으며 돌기 0.65 · 0.7+0.5 + 전진 0.75 · 0.8 (실측 0.74) 120 s 완주. 한계: 0.85, 0.6+0.75, 0.5+1.0. 렌더 확인 전.
9/29 변형 키: `psioff` (앞장서는 발 벌림 배율), `vdir` (착지 방향만 걸러진 실측 쪽으로), `vmeas`, `rais`/`raisx` (한 주기 평균 Raibert), `raiy`, `cent`, `raii`, `srbm`, `walkbias`, `wbcom`, `swh`, `tdz`, `fref`, `tswing`, `tcycle`, `hor`, `kdz` — 결과는 Q&A 9/29.

## 알려진 문제 (이어받은 것)

속도 벽 0.6 m/s (9/25 Q10·9/26 Q1 — k_half 고정으로도 벽 유지). 9/28 Q1: slide+rai+ypath 로 0.65 까지 완주, 0.675 에서 발 pitch
구름으로 전도 — MPC 가 CoP 를 입각의 절반 이상 발 끝에 붙임 (다리 운동량 미모델 가설, 아핀항 후보),
매 걸음 early contact ~20 ms (설계상 −5 mm 착지 목표), O1 (주기 원점 접촉표).
