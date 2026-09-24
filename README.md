# convex MPC 기반 residual RL — Unitree G1 / MuJoCo

## 폴더 구조 (2026-09-16 재편)

```
residual RL/                  ← 최상위
├── MPC/           지금까지 해온 convex MPC 전부 (src/, logs/, 문서)
├── pure RL/       향후 순수 E2E 강화학습 (대조군)
├── MPC + RL/      향후 residual 정책 (최종 목표 구조)
├── reference/     ispaik06/convex-mpc-biped git clone (C++ 대조 구현)
├── deployed RL/   공개된 사전학습 정책 — 참고용 계측 대상
├── Q&A/           날짜별 공부 로그 (전 영역 공통)
├── mujoco_menagerie/   G1 모델 (5개 폴더가 공유)
└── .venv/
```

| 폴더 | 들어갈 것 | 현재 |
|---|---|---|
| **[MPC/](MPC/)** | SRB wrench MPC 구현·실험·기록 | **진행 중** — 0.5 m/s 120초. 목표 1 m/s |
| [pure RL/](pure%20RL/) | E2E RL 베이스라인 | 미착수 (랩 GPU 머신에서) |
| [MPC + RL/](MPC%20+%20RL/) | residual 정책 | 미착수 — 1 m/s 달성 후 |
| [reference/](reference/) | 남의 구현 (clone) | 분석 완료 → [REFERENCE_ANALYSIS.md](REFERENCE_ANALYSIS.md) |
| [deployed RL/](deployed%20RL/) | 사전학습 정책 | 착수 예정 |

**코드를 처음 본다면** → [MPC/CODE_MAP.md](MPC/CODE_MAP.md) (읽는 순서 안내)

실행은 **최상위에서**: `.venv/Scripts/python.exe MPC/src/baseline/09_walk.py --view --vx 0.5 ...`

---


> 이 문서는 **작업 기록 + 공부 자료** 겸용이다. 아래 "개념 정리" 절은 다른 창
> (Claude 본 채팅 등)에 그대로 붙여넣어도 말이 되도록 자기완결적으로 쓴다.
> 설명에 나오는 숫자는 전부 이 저장소에서 실제로 돌려 나온 값이다.

## 현재 단계

**걷는다** (2026-09-10). gait 스케줄 + capture point 착지 + swing 임피던스 + gait
MPC(시변 B, swing 발 W=0 등식 제약)로:
- 제자리 스텝 12 s (발당 14스텝, y 진동 유계)
- **전진 0.15 m/s, 40 s 완주** — 발당 49스텝, ~5.8 m, 97% 추종. `src/09_walk.py --view --vx 0.15`
- 마지막 열쇠는 **매트랩 quadruped 코드(convex_mpc.m)에서 온 착지점 위치 피드백
  부호** (+k·(CoM−ref), 밀린 쪽으로 더 딛기) — MPC_NOTES 12-(7)
- 한계 실측 → 인과 확정 → **결정론적으로 돌파**: swing 반작용(massless-leg 위반)이
  0.2 m/s 벽의 원인임을 격리 실험으로 확정한 뒤 (다리 질량 실측 **43%** — Di Carlo 가
  'SRB 가 합리적'이라 한 10% 의 4배), 적대적 검증 2라운드가 제안한
  **WBC-lite** (`--swingid`: τ += M(q)·q̈_swing,des) + 상체 PD 강화로
  **0.3 m/s 완주 — 벽이 2배 이동.** 이것이 residual RL이 이겨야 할 정직한 baseline.
  질량 스케일 곡선(4점, 단조)도 확보. MPC_NOTES 17절.
- 적대적 검증에서 **CoP 제약의 h 결합항 누락 버그** 발견·수정 (수평력이 발목~발바닥
  3.5 cm 팔길이로 CoP를 옮기는 항 — 2만 샘플 수치검증 100% 일치). 리포트 원문
  [VERIFY_REPORT.md](MPC/VERIFY_REPORT.md), 반영 기록 [MPC_NOTES.md](MPC/MPC_NOTES.md) 16절.

그 전 단계: wrench convex MPC(N=10) 서 있기 — 40 N 외란을 관절 PD 없이 회복
(80 N까지, 120 N은 스텝 필요 영역). 구현 상세·검증·**디버깅 기록 10건**은
**[MPC_NOTES.md](MPC/MPC_NOTES.md)**.

이전 단계 기록 — GRF → 관절 토크 변환 파이프라인(F → τ) 검증:
답을 아는 힘(발당 몸무게 절반의 수직력)을 넣어서 관절 토크가 제대로 나오는지,
그 토크로 실제로 서 있는지를 확인. (아래 내용)

## 세팅

- Python 3.13, venv는 `.venv/` (프로젝트 폴더 안)
- MuJoCo 3.12.0, numpy 2.5.3
- 모델: `mujoco_menagerie/unitree_g1/` (sparse clone — G1만 받음)

```bash
.venv/Scripts/python.exe MPC/src/baseline/02_inspect.py     # 모델 구조 덤프
.venv/Scripts/python.exe MPC/src/baseline/01_view.py crouch # 무제어 — 쓰러지는 게 정상
.venv/Scripts/python.exe MPC/src/baseline/03_grf_to_tau.py  # F -> tau 검증 (핵심)
.venv/Scripts/python.exe MPC/src/baseline/04_stand.py       # 30초 + 외란 테스트
.venv/Scripts/python.exe MPC/src/baseline/04_stand.py --view --pd  # 뷰어로 서 있는 모습
```

## 파일

| 파일 | 역할 |
|---|---|
| [src/paths.py](MPC/src/baseline/paths.py) | 경로 상수 |
| [src/g1_model.py](MPC/src/baseline/g1_model.py) | 모델 로더, position→motor 변환, crouch 자세 |
| [src/02_inspect.py](MPC/src/baseline/02_inspect.py) | body/joint/dof/actuator/site 덤프 |
| [src/01_view.py](MPC/src/baseline/01_view.py) | 무제어 뷰어 |
| [src/03_grf_to_tau.py](MPC/src/baseline/03_grf_to_tau.py) | GRF→토크 변환 유도 + 검증 |
| [src/04_stand.py](MPC/src/baseline/04_stand.py) | 준정적 서 있기 컨트롤러 (lstsq) |
| [src/mpc_srb.py](MPC/src/baseline/mpc_srb.py) | SRB 상태방정식 (A_c, B_c), 파라미터·상태 추출 |
| [src/mpc_qp.py](MPC/src/baseline/mpc_qp.py) | 제약 + condensed QP + quadprog 솔버 + wrench→τ |
| [src/mpc_log.py](MPC/src/baseline/mpc_log.py) / [src/plot_log.py](MPC/src/baseline/plot_log.py) | x/u 로거 (npz+csv) / 플롯 |
| [src/05_sign_check.py](MPC/src/baseline/05_sign_check.py) ~ [src/08_standing_mpc.py](MPC/src/baseline/08_standing_mpc.py) | MPC Step 1~4 검증 스크립트 ([MPC_NOTES.md](MPC/MPC_NOTES.md)) |
| [src/gait.py](MPC/src/baseline/gait.py) / [src/09_walk.py](MPC/src/baseline/09_walk.py) | gait·착지점·swing / 보행 컨트롤러 (Step 5~6) |
| [src/10_constraint_check.py](MPC/src/baseline/10_constraint_check.py) | CoP 제약 h 결합항 수치 검증 |
| **[PHYSICS_MAP.md](MPC/PHYSICS_MAP.md)** | **물리 → 구현 전체 지도**: 모든 근사·휴리스틱·미해결 22건 |

## 걸렸던 것 두 개 (다음에 또 만날 것)

**1. 한글 경로.** MuJoCo의 C++ 파일 로더가 `C:\Users\백종빈\...` 를 못 연다
(`ParseXML: Error opening file`). Python이 파일을 읽어 assets 딕셔너리로 넘기는
방식으로 우회했다 — `g1_model.load()` 참고. MuJoCo VFS는 디렉터리를 무시하고
**basename(대소문자 무시)** 으로 찾으므로 `assets/foo.STL` 과 `foo.STL` 을 둘 다
키로 넣으면 `Repeated file name` 에러가 난다. basename만 넣을 것.
(8.3 단축경로는 이 시스템에서 꺼져 있어서 안 됨.)

**2. Menagerie G1은 position 액추에이터.** `<position kp="500" dampratio="1">` 이라
`d.ctrl` 이 *목표 각도*다. 토크로 쓰려면 gain/bias를 motor와 같게 바꿔야 한다
(`g1_model.to_torque_actuators`). 이걸 안 바꾸고 `ctrl=0` 을 주면 "무제어"가 아니라
"모든 관절 0도로 끌어당기기"가 된다.

## 모델 사실 (측정값)

- `nq=36, nv=35, nu=29` — floating base(dof 0–5) + 29 구동관절
- **총질량 33.341 kg → 무게 327.08 N → 발당 163.54 N**
- 다리 dof: 왼발 6–11, 오른발 12–17 (hip_pitch/roll/yaw, knee, ankle_pitch/roll)
- 발 site: `left_foot` / `right_foot` — `*_ankle_roll_link` 의 원점 (local pos 0,0,0)
- 발바닥 접촉: 발당 구(sphere) 4개, 반지름 0.005, local z=-0.03,
  x=-0.05/+0.12, y=±0.025~0.03 → 지지다각형 약 17cm × 6cm
- keyframe `stand` 는 **무릎이 완전히 펴져 있다**(다리 관절 전부 0).
  수직 GRF에 대한 무릎 자코비안이 거의 0인 특이자세라 F→τ 실험에 부적합.
  → `g1_model.set_crouch()` 로 hip_pitch=-0.30, knee=0.60, ankle_pitch=-0.30
  (합이 0이라 발바닥 수평 유지), 골반 높이는 발이 바닥에 닿도록 자동 조정 (z≈0.7634)

## 핵심 결과

정적 평형(q̇=0, q̈=0)의 운동방정식:

```
g(q) = Sᵀτ + Σ Jᵀf          (S: 액추에이터 선택 행렬)
```

- 베이스 6행(비구동): `g[0:6] = (Σ Jᵀf)[0:6]` ← **접촉력 wrench 균형. convex MPC가 푸는 식.**
- 구동 29행: `τ = g[6:] − (Σ Jᵀf)[6:]`

### 검증 결과 (crouch 자세)

| 항목 | 값 |
|---|---|
| 발당 수직력 W/2 로 계산한 무릎 토크 | −13.48 N·m (좌우 동일) |
| `τ = −Jᵀf` (massless leg) 와의 차이 | max 4.56 N·m ← **다리 링크 자체 중력** |
| W/2 균등 배분의 베이스 잔차 | 힘 0.0000 N, **모멘트 4.22 N·m** |
| 접촉점 8개로 lstsq 재배분 후 잔차 | 2.3e-13 (완전 균형) |
| 재배분 후 접촉점별 최소 수직력 | +30.23 N (전부 양수 = 물리적으로 성립) |

### 6초 시뮬레이션

| 제어 | 결과 |
|---|---|
| `τ = 0` | 0.19초에 쓰러짐 |
| `τ = −Jᵀf` (W/2 균등, massless leg) | 1.27초에 쓰러짐 |
| `τ = g − Jᵀf` (접촉점 lstsq 배분) | **버팀.** pelvis z 0.7634 → 0.7636 |

실제 MuJoCo 접촉력 합 = `[0, 0, 327.077] N` — 계산한 W와 정확히 일치. ✔

### 30초 + 외란 (골반에 40 N, 0.1초 밀기)

| 제어 | 결과 |
|---|---|
| 준정적 피드포워드만 | 외란 전까지 완벽히 정지, push 후 12.25초에 쓰러짐 |
| 준정적 + 관절 PD (kp=30, kd=2) | 30초 생존, push 2회 모두 회복, drift −0.0002 m |

## 여기서 배운 것 (다음 단계로 이어지는 것)

1. **`τ = Jᵀf` 만으로는 부족하다.** massless-leg 가정의 오차가 무릎에서 최대 4.6 N·m,
   힙 피치에서는 −2.79 vs +1.77 로 **부호까지 바뀐다**. convex MPC(SRB)는 다리를
   질량 없다고 보므로, 실제 로봇에는 별도의 중력보상 항이 필요하다.
   → residual RL이 메울 gap 중 하나가 바로 이것.

2. **힘 배분이 곧 균형이다.** W/2 균등 배분은 수직력은 맞지만 CoM이 발 중심에서
   x로 2.3cm 앞에 있어 4.22 N·m 의 피치 모멘트가 남고, 그래서 넘어진다.
   접촉점별로 힘을 재배분하면 잔차가 0이 된다. **이 재배분이 convex MPC가 하는 일**
   (여기선 lstsq 한 줄, MPC에서는 마찰콘·단일방향·시간지평 제약이 붙은 QP).

3. **층1 / 층2 구분이 코드에서 그대로 보인다.**
   - 층1 (재료): 자코비안 `mj_jac`, 중력 `qfrc_bias`, 베이스 wrench 균형식 조립
   - 층2 (솔버): 지금은 `np.linalg.lstsq` 한 줄 → 나중에 qpOASES / OSQP / 자작 ADMM
   `04_stand.py` 의 `QuasiStaticStand.__call__` 에서 lstsq 줄만 갈아끼우면 된다.

4. **피드포워드는 외란에 무방비.** 준정적 해는 정확하지만 열린 루프라 한번 밀리면
   못 돌아온다. MPC의 시간지평(preview)과 피드백이 왜 필요한지의 실물 증거.

---

# 개념 정리 — 대화 기록

질문이 나올 때마다 아래에 누적한다. 각 항목은 앞 내용을 안 읽어도 이해되게 쓴다.

## Q. "접촉점 8개로 재배분했다"는 게 무슨 말인가

### 1) 접촉점 8개가 뭐냐

G1의 발바닥은 면(面)이 아니라 **작은 구(sphere) 4개**로 모델링돼 있다.
`mujoco_menagerie/unitree_g1/g1.xml` 의 `left_ankle_roll_link` 안:

```xml
<geom class="foot" pos="-0.05  0.025 -0.03"/>   <!-- 뒤꿈치 안쪽 -->
<geom class="foot" pos="-0.05 -0.025 -0.03"/>   <!-- 뒤꿈치 바깥 -->
<geom class="foot" pos=" 0.12  0.03  -0.03"/>   <!-- 발가락 안쪽 -->
<geom class="foot" pos=" 0.12 -0.03  -0.03"/>   <!-- 발가락 바깥 -->
```

발당 4개 × 두 발 = **8개**. MuJoCo가 실제로 접촉을 푸는 지점이 정확히 이 8개고
(시뮬레이션 중 `d.ncon == 8`), convex MPC의 contact point와 1:1 대응된다.

```
        발 하나 (위에서 본 그림)
        y
        ↑
   ┌────────────────┐
   │ ●            ● │   ← 앞(발가락) x=+0.12
   │                │
   │ ●            ● │   ← 뒤(뒤꿈치) x=-0.05
   └────────────────┘ → x
```

### 2) "재배분"이 무슨 뜻이냐

**순진한 방식**은 힘 작용점을 발 site(발목 원점) 딱 2개로 잡고 각각 `[0, 0, W/2]` 를
꽂았다. 미지수 0개 — 그냥 정한 값이다.

**재배분**은 그 8개 점 각각에 붙는 3차원 힘을 **미지수로 두고 푼 것**이다:

- 미지수: 8점 × 3축 = **24개**
- 식: 베이스 6행 wrench 균형 `g[0:6] = Σ Jᵢᵀ fᵢ` → **6개**

24 > 6 이라 해가 무수히 많은 **부정(underdetermined)** 문제고, `np.linalg.lstsq` 가
그중 크기가 가장 작은 해(minimum-norm)를 골라준 것.
"몸을 정확히 지탱하는 힘 조합 중 하나를 8개 점에 나눠 배정했다" = 재배분.

### 3) 왜 이게 넘어짐/버팀을 갈랐나 — 숫자

lstsq가 배분한 실제 값 (crouch 자세, W = 327.08 N):

| 위치 | 각 점의 fz |
|---|---|
| 뒤꿈치 4점 (world x = −0.039) | **51.5 N** |
| 발가락 4점 (world x = +0.131) | **30.3 N** |
| 합 | 327.08 N = W ✔ |

뒤에 더 싣고 앞에 덜 실었다. 그 결과 **압력중심(CoP)** 이:

```
재배분 후 CoP = (0.0237, 0.0001)
CoM 수평위치  = (0.0237, 0.0001)     ← 정확히 일치
```

반면 순진한 방식은 힘이 발목 원점(x = 0.0108)에만 걸리니 CoP도 거기 고정:

```
순진한 CoP  = (0.0108, 0.0000)
CoM x       = 0.0237
어긋남      = 0.0129 m
모멘트 오차 = 327.08 N × 0.0129 m = 4.22 N·m   ← 베이스 잔차의 정체
```

**CoM이 발목보다 1.3 cm 앞에 있는데 힘은 발목 밑에서만 밀어 올리니, 몸이 앞으로
넘어가는 모멘트가 남는다.** 재배분은 뒤꿈치에 힘을 더 실어 CoP를 CoM 밑으로 끌고 온 것.
사람이 앞으로 기울면 발뒤꿈치에 체중이 실리는 것과 같다.

### 4) 이게 곧 MPC가 하는 일

지금 푼 건 **한 시점의, 제약 없는** 힘 배분이다. convex MPC는 같은 미지수(접촉점별
힘)에 세 가지를 더 얹는다:

| | 지금 (lstsq) | convex MPC |
|---|---|---|
| 단일방향 `fz ≥ 0` | 없음 (우연히 다 양수) | 제약으로 명시 |
| 마찰콘 `‖f_xy‖ ≤ μ fz` | 없음 | 제약으로 명시 |
| 시간지평 | 1스텝 | N스텝 preview |
| 풀이 | `lstsq` 한 줄 | QP 솔버 |

이번엔 최소 fz가 +30.23 N 으로 전부 양수라 물리적으로 성립했지만 그건 **운**이다.
CoM이 지지다각형 가장자리로 가면 lstsq는 태연히 음수 fz(= 땅이 발을 잡아당김)를 뱉는다.
그 순간부터 QP가 필요해지고, 그게 **층2 교체**의 시작점이다 —
[src/04_stand.py](MPC/src/baseline/04_stand.py) `QuasiStaticStand.__call__` 의 `lstsq` 한 줄 자리.

## Q. MPC 가중치는 어떻게 정했나

값 ([src/mpc_qp.py](MPC/src/baseline/mpc_qp.py) 상단, x = [Θ, p, ω, v, g] 순):

```
Q = diag[ 10, 10, 20,   50, 50, 300,   5, 5, 5,   20, 20, 50,   0 ]
          roll pitch yaw  px  py  pz    ω(감쇠)     vx  vy  vz    g(반드시 0)
R = 1e-5 균일 (힘/모멘트 동일),  단 u_ref = 중력 피드포워드(발당 Mg/2) 기준
```

정한 논리 세 가지 (전부 실패에서 배움, 상세는 MPC_NOTES.md 6절):

1. **pz가 제일 무겁다(300)** — 무게 지탱이 최우선. g 상태는 상수라 가중 0.
2. **R은 균일 + 중력 피드포워드 기준.** R_moment를 힘의 100배로 두면 QP가 발목
   모멘트 대신 "싼" Fx로 피치를 만들어 로봇이 밀려간다. u_ref=0이면 R이 중력과
   싸워 Fz가 Mg의 78%밖에 안 나온다. 둘 다 실제로 넘어져서 발견.
3. **자세 가중은 대역폭으로 정한다.** LQR 근사 ω_n ≈ (q_θ/r_eff)^¼/√I 로 환산해서
   업데이트 주기·실현 지연이 감당할 대역폭(~2-3 Hz)에 맞춘다. Q_Θ=300(~6 Hz)은
   발산했고 10(~3 Hz)은 안정. 가중치 10배 = 대역폭 1.8배(4제곱근)라는 점이 중요.

현재 값은 "튜닝된 최적"이 아니라 **안정 영역의 첫 값** — 튜닝 여지는 많다.

## Q. 지금 전체 시스템이 어떻게 돌아가나 — 제어 한 사이클 해부 (2026-09-11)

두 개의 루프가 돈다. 층1/층2로 보면 100 Hz 상자의 "QP 조립"까지가 층1, quadprog
한 줄이 층2.

```
[100 Hz]  상태 x(13) 추출 → gait 시계 → 착지점 → 참조 → QP 조립 → quadprog
          → 발별 wrench W_L, W_R (각 6: 힘3+발목모멘트3)
[500 Hz]  τ = qfrc_bias − JᵀW(stance) + 임피던스(swing) + M·q̈_swing(WBC-lite) + 상체 PD
```

**100 Hz 단계별** ([src/09_walk.py](MPC/src/baseline/09_walk.py) `update_mpc`):
1. **상태**: x = [Θ, CoM위치, ω, CoM속도, g]. p·v는 골반 아닌 **전신 CoM**,
   ω는 **각운동량 역산**(ω=I⁻¹L) — 골반 각속도를 쓰면 내부 진동에 MPC가 과반응(발산 이력).
2. **gait 시계**: 주기 0.8 s, stance 75%, 좌우 위상 0.5 차. 미래 0.8 s 접촉표(16×2)를 MPC에 전달.
3. **착지점**: 엉덩이밑 + v_cmd·T_st/2 + (v−v_cmd)/ω₀ (capture) + 0.15(CoM−기준)
   — 마지막 +부호 위치항("밀린 쪽으로 더 딛기", 매트랩 k_y에서 옴)이 횡 안정의 열쇠.
4. **참조**: px는 v_cmd 적분, pz 고정, **py는 지지 중심으로 미리 이동(sway)** — preview 필수 기능.
5. **QP**: 미지수 = 16스텝×12 = 192. 동역학 x_{k+1}=Ax+B_k u (B_k 시변), 제약은
   stance 발 10행(마찰콘+Fz+CoP, h 결합항 포함) / swing 발 W=0 등식("그 발로 못 민다"를
   미리 알림 → 체중이동이 저절로 나옴), 비용 ‖x−ref‖²_Q+‖u−중력ff‖²_R.
6. **quadprog** (~5 ms) → 답 192개 중 **첫 12개만 쓰고** 다음 사이클에 다시 푼다 (=MPC).

**500 Hz** (`torque`): qfrc_bias가 SRB가 무시한 다리 중력을 메우고(첫날 τ=g−Jᵀf 검증
그대로), JᵀW가 wrench→토크 번역, swing은 임피던스, M·q̈_des가 다리 던지는 반작용을
역동역학으로 선보상(WBC-lite), 상체 PD가 허리·팔을 고정.

**적대적 검증 2라운드가 바꾼 것**: ① CoP 제약의 h 결합항 버그(발목↔발바닥 3.5 cm —
가속 시 뒤꿈치 권한 과대평가→후방 전도) 발견·수정, 수치검증 100% ② 다리질량 1/3
격리 실험으로 "0.2 벽 = swing 반작용" 인과 확정 ③ WBC-lite 처방 → 원래 질량으로
0.3 m/s 완주. 스코어: 기본 0.15 / +WBC-lite+상체PD **0.30 m/s**.

## Q. 다리 질량을 원상태로 두면 무슨 문제가 생기나 (질량 1/3 실험의 의미)

먼저: 질량 1/3은 **진단 실험**이었지 걷게 하는 조치가 아니다. 최종 상태는
**원래 질량 그대로** WBC-lite 보상(`--swingid`)으로 0.3 m/s를 걷는다.

원래 질량 + 보상 없음일 때의 사고 경로 (전부 로그 실측):
1. 0.3 m/s 보행에서 swing 발은 몸 기준 ~0.35 m 를 0.2 s 에 가야 함
   → 다리 ~5 kg × ~50 m/s² = **~250 N** 으로 던져야 한다.
2. 작용-반작용으로 그 힘이 몸통을 반대로 민다 → CoM 속도 0.3 → −0.1 m/s 붕괴.
3. MPC는 이걸 모른다 — 두 채널로 새기 때문:
   - **토크 실현**: stance 식 τ = g − JᵀW 는 준정적 가정. 실제론 M(q)q̈ 항이 있고
     M 이 밀집이라 swing 가속이 stance 관절로 새어 들어와 "실현 wrench ≠ 명령".
   - **자세 측정**: 다리를 던지면 골반이 반대로 도는데(고양이 회전) 각운동량 기반
     ω 는 이 내부 재배향에 0 을 냄 — Θ와 ω 가 "같은 강체"가 아닌 모순 상태.
4. 뒤로 기우는데 G1 뒤꿈치는 5 cm(발가락 12 cm 와 비대칭) → 후방 CoP 여유 부족
   → 후방 전도.

**WBC-lite 가 막는 방법**: swing 궤적은 계획된 것이라 q̈_des 를 미리 안다.
τ += M(q)·q̈_swing,des 를 넣으면 M 의 밀집 구조가 stance 관절 몫 보상까지 자동
배분 → 토크 실현 어긋남(MPC_NOTES 의 채널 ②)이 막힘 → 원래 질량으로 0.3 완주.

**남는 것**: 보상해도 0.35+ 는 전도 — q̈ 계획≠실제, J̇q̇ 생략, 고정 타이밍,
자세 불일치(상체 PD 밴드에이드). 이 잔여분이 residual RL 이 배울 대상.

## Q. OA-MPC 논문(Ding 2022) Table I 가중치를 어떻게 읽어야 하나

전제 둘: ① 절대값은 무의미, Q/R **비율**만 의미. ② 그 논문은 MPC 아래 full-body
TSC(식 17)가 실현을 담당하는 **2층 구조** — 가중치가 그 전제로 튜닝된 값이라
단층인 우리에게 그대로 이식하면 안 된다.

Table I 의 설계 결정 5개:
1. **Θ: roll 750 ≫ pitch 75** — 라인 풋이라 mx=0(발목 roll 권한 없음) → 권한 없는
   축일수록 비용을 무겁게 걸어 스텝으로 잡게 함. G1도 roll 권한이 약해(반폭 2.5 cm)
   같은 논리 부분 적용 후보.
2. **ṗ: vy 5000 ≫ vx 500** — 횡 속도 감쇠 10배. 우리가 limit cycle 로 고생한 그
   채널을 처음부터 10배로 누른 것. (우리는 vy=vx=20 균등)
3. **R: Fz·m 이 Fxy 의 10배** — 수직력은 체중 근처에 잠잠하게, 수평력은 싸게 풀어
   일꾼으로. 모멘트 10배는 우리 실패값(100배)과 현재값(1배)의 정중앙.
4. **δc: 1e5/1e6 (y 가 10배)** — 착지점은 참조 신뢰 + 이탈만 벌점, 횡은 크로스오버
   때문에 보수적.
5. **항목별 decay γ (m 은 0.5로 최속)** — 지평 끝 선형화 불신을 채널별 속도로 반영.
   어차피 실행 안 될 먼 미래의 발목 계획은 자유롭게. 우리는 decay 없음 — 도입 후보.

R_Fx 기준으로 환산하면 그들의 Q 는 우리보다 1~2차수 무름:
**"몸은 흔들리게 두고 스텝으로 잡는다"(라인 풋+δc 인루프) vs 우리 "발목 wrench 로
꽉 잡는다"(G1 6-DoF 발목)** — 둘 다 자기 로봇에 정합적. 단 그들의 Fig.5(토크 외란
내성)는 큰 외란의 최종 해결사가 발목이 아니라 스텝임을 보여줌 — 우리 120 N 한계,
0.35 m/s 벽과 연결.

주의: 표의 Θ̇ 행이 p_c 와 동일 숫자 — 조판 오류(행 밀림) 가능성, 인용 금지.
추후 실험 후보(우선순위): Q_vy 단독 인상 → decay γ → R_m=10×R_F 재시도 →
roll/pitch 비대칭 → R_Fz 분리.

## Q. "대역폭 추정"이 뭐고 코스트와 무슨 관계인가

**대역폭** = 피드백 루프가 오차를 따라잡으려 드는 속도 (Hz). 높으면 빳빳하고
빠름, 낮으면 무름. 비유: 반사신경 느린 사람이 막대 세우기를 세게 반응하면
좌우로 점점 크게 흔들다 떨어뜨림 — Q_Θ=300 발산이 정확히 그것.

**코스트와의 관계**: MPC 이차비용 ≈ LQR. Q 크다="오차 비쌈"→세게 교정→빠른 루프,
R 크다="힘 비쌈"→살살→느린 루프. 즉 Q/R 비율이 컨트롤러의 반응 속도 다이얼.
피치축(I·θ̈=M, 비용 qθ²+rM²)에서 LQR 해는 가상 스프링 k₁=√(q/r) 이고

    ω_n = (q/r)^(1/4) / √I

우리 값 대입(I_yy=3.18, r≈5e-6): Q_Θ=300 → ~8 Hz, Q_Θ=10 → ~3 Hz.

**왜 8 Hz는 터지나**: 루프 지연(MPC 10 ms 주기 + wrench 실현 등 ~25 ms)의 위상
손실 = ω×T. 8 Hz는 ~70° 잃어 여유 없음 → 갱신마다 과잉교정 → 진동 증폭.
3 Hz는 ~30° → 생존. "지연이 감당 못 하는 강성은 쥐면 안 된다."

**실전 교훈**: ① 가중치 만지기 전에 Q/R을 Hz로 번역해 지연과 대조할 것.
② 4제곱근이라 가중치 10배 = 대역폭 1.8배 — 선형으로 생각하면 "체감 없다가
갑자기 터짐". 가중치는 로그 스케일(×3, ×10, ×30)로 움직일 것.

---

## Q. MIT humanoid 는 다리가 가벼워서 '다리 무시'가 되는 건가 (2026-09-15)

아니다 — **비율은 G1 과 사실상 같다**. 공개 URDF([mit-biomimetics/fld](https://github.com/mit-biomimetics/fld))
와 우리 G1 모델로 직접 계산:

- 양다리/총질량: **MIT 43.7 % (10.61/24.25 kg) vs G1 43.1 % (14.37/33.34 kg)**
- 차이는 **분포**: 무릎 아래가 MIT 0.63 kg vs **G1 2.61 kg (4.2배)**.
  swing 때 가장 빨리 움직이는 원위 질량이 반작용을 지배하므로, 같은 43% 라도
  MIT 쪽이 "massless leg" 근사에 훨씬 유리하다 (구동기를 힙에 모은 설계).
- 그래도 결론은 긍정적: reference 저장소(ispaik06/convex-mpc-biped)는 **같은
  G1 을 상체만 SRB 로** 0.6 m/s / 1.3 rad/s 로 걷게 한다. 조건은 다리를 빼는
  대신 **swing 다리가 자기 중력·관성을 스스로 보상**하는 것 (OSC + bias 보상).
  "모델에서 뺀 다리"와 "방치된 다리"는 다르다. 상세: MPC_NOTES 20절.

## Q. 회전이 0.2 rad/s 에서 막혔던 진짜 이유 (2026-09-15)

세 겹이었다 (MPC_NOTES 19절):
1. **yaw 랩 버그** — 측정 yaw(atan2, ±180° 랩) vs 참조(무한 적분). 참조가 180° 를
   넘는 순간 MPC 가 −360° 오차를 보고 폭주. 0.3 이 매번 13 s 대에 무너진 범인
   (0.3 × 13 s ≈ 180°). 언랩 5줄로 수정.
2. **회전 공급 기구 부재** — 발을 착지 시점 yaw 로 미리 돌려 딛지 않으면 몸이 돌
   수단이 없다 (stance 발 고정 + swing 정렬만으론 부족, A/B 로 확인).
3. **stance 발 미끄럼** — yaw 유지 모멘트(hip_yaw PD)로 고정. 단 2번과 짝일 때만 유효.

셋 다 넣은 결과: 제자리 회전 **1.3 rad/s 완주** (이전 0.2), 전진+회전 조합
(0.15+0.5, 0.3+0.2) 최초 완주. 직진 경로는 전부 wz 게이트라 불변.
⚠ 위는 20초 기준. **120초 내구 기준(뷰어 관찰 후 상향)으로는 회전 0.5·조합
0.15+0.5 까지 견고, 1.3/직진 0.3 은 ~30초 버스트** — MPC_NOTES 22절.

## Q. "상체만 SRB" 실험 — 왜 우리한테는 안 됐나 (2026-09-15)

reference 처럼 다리를 SRB 에서 빼는 실험 ([src/11_walk_srb_upper.py](MPC/src/baseline/11_walk_srb_upper.py),
09_walk 에 훅 2개만 뚫어 diff 가 곧 개념 차이가 되게 함). 결과는 **전 영역 패배**
(직진 0.3: baseline 20 s ✓ vs 상체 변형 4~13 s; 회전 1.3: 완주 vs 8 s). 배운 것 셋:

1. **병진은 뉴턴이 정확하다**: M_tot·v̇ = ΣF − M_tot·g 는 다리 포함 근사 없음.
   "다리 무시"는 회전(오일러) 방정식에만 의미가 있다. 병진까지 상체로 바꾸면
   (v1) 정확한 식을 일부러 틀리게 만드는 것 — 1/M_ub 과민 응답으로 램프 전도.
2. **다리는 wrench 에 매달린다**: 우리 스탠스 사상 τ = qfrc_bias − JᵀW 는
   접촉력 = W 를 정확히 실현하므로 (Step 1 검증의 이면), stance 다리 무게도
   구조가 아니라 W 가 진다 → 상체 모델의 유효 중력은 g·M_tot/M_ub.
   (reference 는 bias 를 안 넣어서 그쪽 W 는 상체 몫만 짐 — 사상 차이가 모델을 바꾼다)
3. **baseline 의 '무거운 모델'은 필터였다**: 전신 L 기반 ω + 전신 I 는 다리
   스윙 반작용을 L-공간에서 암묵 흡수한다 (다리를 차도 전신 L 은 거의 그대로).
   Θ/ω 불일치는 그 흡수의 대가였고, 치를 만한 값이었다. 상체 SRB 로 가려면
   reference 의 전제(위치 가중치 ~840배 + Λ-스케줄 고강성 스윙 OSC 로 반작용
   자체를 축소)부터 이식해야 한다 — 모델은 패키지와 분리 불가. 상세: MPC_NOTES 21절.

## Q. 랩장 피드백 — "버틀넥"과 "직립 특이점"이 뭔가 (2026-09-15)

**버틀넥(병목)**: 직렬 시스템의 성능은 가장 약한 고리 '하나'가 정한다.
병목이 아닌 곳을 개선하면 성능이 안 오르므로, 지금 무엇이 제한하는지
**격리 실험으로 확인하고 그것부터** 치는 게 연구 방법론. 병목은 옮겨 다닌다 —
회전 0.2→1.3 rad/s 는 병목 4개(mz 무제약 → 제약 frame → 발 사전회전 부재 →
yaw 랩 버그)를 순서대로 제거한 결과 (MPC_NOTES 18.5~19절이 그 기록).
현재 병목 (증거 포함): 직진 0.4+ = 스윙 반작용(--legmass 격리, 21절에서 모델
교체로도 못 피함 → 스윙 대역폭 ω_n 30 vs reference 100 rad/s), 몸통 흔들림 =
착지 충격, roll = 발 반폭 2.5 cm(하드웨어).

**직립 특이점**: 무릎이 완전히 펴지면 다리 야코비안 J(q) 랭크가 떨어지는
특이자세 — 다리 축 방향으로 발을 못 움직인다(1차 근사).
- 망가지는 것: ① J⁻¹/lstsq 폭발 (우리 WBC-lite 의 qdd 300 클램프가 이 보호)
  ② 높이 '조절' 권한 소실 — 버티는 건 토크 0으로 공짜지만 ∂길이/∂무릎각=0 이라
  Fz 를 못 만든다 ③ 착지 충격 흡수 불가.
- 휴머노이드 특성: 로봇팔은 특이점이 작업영역 가장자리인데, 휴머노이드는
  '직립'이라는 기본 자세가 바로 특이점 — 한가운데 지뢰. 그래서 bent-knee 가
  표준 처방 (대가: 무릎이 체중을 모멘트로 버텨 토크·발열 증가).
- 우리 상태: **이미 반영** — g1_model.set_crouch(knee=0.60 rad≈34°)가 day-1
  부터 이 이유로 존재 (docstring 에 "keyframe 'stand' 는 특이자세" 명시).
  관찰 포인트: 보행 중 pelvis z 가 0.763→0.79 로 떠오르는 표류 = 무릎이
  펴지는 방향 = 특이점 마진 잠식. 이상 거동 시 무릎각 로그 볼 것.

## 다음

- [x] 접촉력에 마찰콘 + 단일방향(fz ≥ 0) 제약 → lstsq를 QP로 — 2026-09-10, [MPC_NOTES.md](MPC/MPC_NOTES.md)
- [x] SRB 모델로 convex MPC 조립, 시간지평 도입 (N=10 wrench MPC. CasADi는 자동미분 검증용,
      솔버는 한글 경로 DLL 문제로 quadprog) — 2026-09-10
- [x] gait 스케줄 + capture point 착지 + swing 궤적 → 제자리 스텝 → 전진 0.15 m/s — 2026-09-10
- [ ] 횡 진동 장기 성장(시정수 ~20 s) 해결, 보행 속도 확장 (MPC_NOTES 12-(6), 13절)
- [ ] quadprog vs OSQP 비교 → 병렬화 시 ADMM 분해 타당성 판단
- [ ] residual PPO (학습 단계에서 IsaacGym)
