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
— 옆 0.3 · 회전 1.3 (선행 0/0.5/1) · 1.5 · 걸으며 돌기 0.3~0.6+0.5 · 0.3+1.0 · 전진 0.65~0.75 모두 120 s 완주 (실측 천장 0.65).
`admm=50` 을 빼면 (수렴 OSQP) 전진 추종이 더 낫다 (0.75 → 실측 0.68). 한계: 0.65+0.5 · 0.6+0.75 (과속 전도).
9/29 변형 키: `psioff` (앞장서는 발 벌림 배율), `vdir` (착지 방향만 걸러진 실측 쪽으로), `vmeas`, `rais`/`raisx` (한 주기 평균 Raibert), `raiy`, `cent`, `raii`, `srbm`, `walkbias`, `tdz`, `fref`, `tswing`, `tcycle`, `hor`, `kdz` — 결과는 Q&A 9/29.

## 알려진 문제 (이어받은 것)

속도 벽 0.6 m/s (9/25 Q10·9/26 Q1 — k_half 고정으로도 벽 유지). 9/28 Q1: slide+rai+ypath 로 0.65 까지 완주, 0.675 에서 발 pitch
구름으로 전도 — MPC 가 CoP 를 입각의 절반 이상 발 끝에 붙임 (다리 운동량 미모델 가설, 아핀항 후보),
매 걸음 early contact ~20 ms (설계상 −5 mm 착지 목표), O1 (주기 원점 접촉표).
