# G1 Wrench MPC — 검증용 번들 v2 (적대적 검증 1라운드 반영 후)

> 1라운드 리포트의 지적을 전부 반영한 최신 상태. 이번 라운드에서 봐줬으면 하는 것:
> ① MPC_NOTES 16.1 — h 결합항 수정과 수치 검증(10_constraint_check.py)이 완전한가
> ② 16.3 — 격리 실험(다리질량 1/3, 힘상한 400)이 massless-leg 인과 확정으로 충분한가
> ③ **16.5 — 1라운드 (b)에 대한 반론**: 'd항을 SRB에 주입'은 병진 채널엔 해당 없음
>    (p·v는 전신 CoM이라 CoM 동역학은 내부 운동과 무관하게 정확). 반작용이 새는 곳은
>    Θ 운동학 불일치와 토크 실현(준정적 가정)이며 feedforward는 실현 계층에 설계해야
>    한다는 주장 — 이 반론이 맞는가? 틀렸다면 어디가 틀렸는가?
> ④ 다음 단계인 'swing 반작용 보상 (실현 계층, 양극성)' 설계 방향에 대한 의견



---

# FILE: MPC_NOTES.md

# G1 Wrench MPC 구현 노트 (공부용)

> 2026-09-10 구현. 이 문서는 **다른 창에 붙여넣어도 읽히는 자기완결 문서**다.
> 모든 숫자는 이 저장소에서 실제 실행해 나온 값. 코드는 `src/` 참조.
> 이전 단계(GRF→토크, lstsq 서 있기)는 [README.md](../README.md) 참조.

## 0. 무엇을 만들었나 — 한 문장

서 있기 컨트롤러의 힘 배분을 `np.linalg.lstsq` 한 줄에서 **시간지평 N=10짜리
제약 있는 wrench QP(convex MPC)** 로 교체했고, 그 결과 순수 피드포워드가
넘어졌던 40 N 외란을 **관절 PD 없이** 회복하게 됐다 (80 N까지 버팀).

```
[매 10ms]  MuJoCo 상태 → x(13) 추출 → QP 풀기 → 발별 wrench W_L, W_R (각 6)
[매 2ms]   τ = qfrc_bias − Σ J_iᵀ W_i  → 29개 관절 토크 인가
```

층1(재료: 상태방정식·제약·참조 조립)과 층2(솔버) 구분 그대로:
- 층1: [src/mpc_srb.py](src/mpc_srb.py) (동역학) + [src/mpc_qp.py](src/mpc_qp.py) 의 행렬 조립
- 층2: `WrenchMPC._solve_qp` 의 quadprog 호출 — **여기 한 줄이 교체 지점**

## 1. 파일 맵

| 파일 | 역할 |
|---|---|
| [src/mpc_srb.py](src/mpc_srb.py) | SRB 상태 정의, 파라미터 추출, A_c/B_c 조립, 이산화, 상태 추정 |
| [src/mpc_qp.py](src/mpc_qp.py) | 제약 C_foot(10행), condensed QP 조립, quadprog 솔버, wrench→τ |
| [src/mpc_log.py](src/mpc_log.py) | x/u/τ/솔버통계 로거 → `logs/*.npz` + `.csv` |
| [src/plot_log.py](src/plot_log.py) | 로그 플롯 → PNG |
| [src/05_sign_check.py](src/05_sign_check.py) | Step 1: 부호 규약 검증 |
| [src/06_standing_qp.py](src/06_standing_qp.py) | Step 2: N=1 QP 서 있기 + lstsq 회귀 비교 |
| [src/07_b_autodiff_check.py](src/07_b_autodiff_check.py) | Step 3: CasADi 자동미분으로 A, B 검증 |
| [src/08_standing_mpc.py](src/08_standing_mpc.py) | Step 4: N=10 MPC + 외란 (`--view`, `--push F`) |

## 2. 모델 요약

상태 (13): `x = [roll, pitch, yaw, px, py, pz, wx, wy, wz, vx, vy, vz, g]`
— Θ는 ZYX 오일러, p·v는 **CoM**(pelvis 아님), ω는 world, g=9.81은 Di Carlo 트릭
(상수를 상태에 넣어 아핀 항을 선형으로 유지).

입력 (12): `u = [W_L, W_R]`, `W = [Fx, Fy, Fz, mx, my, mz]` — **발 site(발목 원점)에
작용하는 wrench, world frame**. quadruped(발4×힘3)와 차원이 12로 같지만 내용이 다름:
발 2개 × (힘3 + **발목 모멘트3**). 모멘트가 생기는 이유는 발이 면접촉이라서.

연속 동역학 → [src/mpc_srb.py](src/mpc_srb.py) `continuous_AB`:
```
Θ̇ = Rz(ψ)ᵀ ω                       ← Di Carlo 식 (12), 작은 roll/pitch 근사
ṗ = v
ω̇ = I_w⁻¹ Σᵢ (rᵢ×Fᵢ + mᵢ)          ← rᵢ = 발 site − CoM (CoM 기준!)
v̇ = (1/M) ΣᵢFᵢ − g ẑ
ġ = 0
```
B_c에서 quadruped 대비 새 블록은 **ω̇ 행의 모멘트 3열 = I_w⁻¹** 뿐.
이산화는 Euler: `A_d = I + A_c·dt`, `B_d = B_c·dt`, dt = 0.02 s.

파라미터 (전부 모델에서 추출, `make_params`):
- M = 33.341 kg, I_body diag ≈ [3.549, 3.175, 0.542] kg·m²
- 발: l_t = 0.12 (발가락), l_h = 0.05 (뒤꿈치), **w = 0.025** (좌우 반폭)
  - ⚠ 명세는 w=0.03이었지만 XML 실측은 뒤꿈치 y=±0.025 → 보수적으로 0.025 사용

## 3. 제약 (발당 10행, `foot_constraints`)

```
마찰 피라미드:  ±Fx ≤ μFz,  ±Fy ≤ μFz          (μ=0.6)
수직력:         10 ≤ Fz ≤ 500 N
롤 CoP:         |mx| ≤ w·Fz                       (G1은 ankle-roll 있어서 범위 제약.
                                                   MIT Humanoid는 mx=0 강제였음)
피치 CoP:       −l_t·Fz ≤ my ≤ l_h·Fz            (비대칭! 앞 0.12 / 뒤 0.05)
```
전부 `(계수)·Fz` 와의 1차 결합이라 **선형** → QP에 그대로 들어간다.
mz 제약은 초기 단계라 생략 (R 벌점만).

## 4. QP (condensed, Di Carlo 식 27–32)

```
X = A_qp x0 + B_qp U          A_qp = [A_d; A_d²; …],  B_qp 하삼각 블록
min ½UᵀHU + gᵀU               H = 2(B_qpᵀQ̄B_qp + R̄)
                              g = 2B_qpᵀQ̄(A_qp x0 − X_ref) − 2R̄·U_ref
s.t. C U ≤ d                  (3절 제약 × 2발 × N)
```
- N=10, dt=0.02 → **0.2 s preview**. MPC 재계산 100 Hz, 토크는 500 Hz로 재계산
  (wrench는 유지하되 J, qfrc_bias는 매 스텝 갱신).
- 솔버: **quadprog** (Goldfarb–Idnani active-set). 평균 6 ms/solve (N=10, 120변수,
  200제약, 밀집). N=1이면 0.4 ms.
- ⚠ CasADi의 qpOASES 등 conic 플러그인은 **한글 경로에서 DLL 로딩 실패**(WIN32 126)
  — CasADi는 자동미분(Step 3)에만 사용. MuJoCo 로딩(assets dict 우회)과 같은 계열 문제.

## 5. 검증 결과 (명세 10절 순서)

### Step 1 — 부호 규약 ✓
lstsq 서 있기 해로 my를 두 방식(외적 직접 / −Fz·x_CoP)으로 계산:
**둘 다 −2.116 N·m (왼발)** 로 일치, 범위 [−19.6, +8.2] 안.
- ⚠ 명세의 예상값 −3.87은 **world 원점 기준** 모멘트(−163.5×0.0237)였다.
  제약의 l_t/l_h는 **발목 기준**이므로 발목 기준 x_CoP = 0.0237−0.0108 = 0.0129,
  my = −2.11이 올바른 통과값. 부호(음수=CoP가 발목 앞)는 두 기준 동일.
  → 명세 11절이 경고한 '기준점 혼동'의 실제 사례.

### Step 2 — N=1 QP 서 있기 ✓
- 회귀: QP vs lstsq 발별 wrench — Fz 163.53 vs 163.65, my −1.77 vs −2.12 (근사 일치)
- 10초 서 있기: drift **+0.0002 m**, 실접촉력 합 [0.009, 0, 327.078] N, 위반 0
- N=1의 정체: Euler 1스텝에서 u는 속도(ω,v)에만 영향 → 위치·자세 피드백 **없음**.
  사실상 "wrench 균형 + 속도 댐핑". 위치 피드백은 N≥2부터 생긴다.

### Step 3 — 자동미분 대조 ✓
CasADi로 f(x,u)를 심볼릭으로 적고 jacobian(f,x), jacobian(f,u) 평가:
**max|A_sym−A_hand| = 0, max|B_sym−B_hand| = 2.2e-16** (기계 정밀도).

### Step 4 — N=10 MPC + 외란 ✓
- 초기해 N=10 vs N=1: Fz 동일, N=10이 my=−2.113(lstsq와 일치), Fx≈0으로 더 정확
  (지평이 길어지면 "Fx로 잠깐 때우기"가 미래 위치 오차로 벌점받기 때문)
- 20초 생존, drift +0.0007 m, 위반 0
- **40 N/0.1s 밀기 2회 회복** — pitch 최대 0.35°, CoM 오차 4 mm, my −2→−15 N·m
  스윙 후 복귀 ([logs/standing_mpc_n10_20260910_184000.png](logs/standing_mpc_n10_20260910_184000.png))
- 외란 여유: **80 N 버팀, 120 N 쓰러짐**. 120 N×0.1 s → Δv≈0.36 m/s는 발 길이
  17 cm의 ankle strategy(CoP 이동) 한계 밖 — 그 너머는 **스텝을 밟아야 하는 영역**
  = Step 5(gait)가 필요한 이유의 실물 증거.

## 6. 디버깅 기록 — 4번 터졌고, 전부 교훈이 됐다

이 절이 이 문서의 핵심이다. 명세대로 짰는데 안 됐던 것들.

### (1) u_ref=0이면 R이 중력과 싸운다 — Fz가 256 N밖에 안 나옴
첫 실행에서 QP가 Fz=127.96/발 (합 256 ≠ 327)을 뱉고 5.5 cm/s로 가라앉았다.
원인: 비용 `‖u‖²_R`은 u=0을 선호 → 중력 지탱에 필요한 힘과 타협한다.
해석적으로 F* = Mg·[2Q_v(dt/M)²] / [2Q_v(dt/M)² + R_F] = 0.783·Mg = 256 N — 출력과
정확히 일치. **해법: u_ref = 중력 피드포워드(발당 Mg/2)를 기준으로 정규화**
(`‖u−u_ref‖²_R`). 명세의 "u_ref=0으로 시작"은 R이 충분히 작을 때만 무해하다.

### (2) R_moment ≫ R_force 면 QP가 모멘트 대신 Fx로 피치를 만든다
R_m=1e-3 (힘의 100배)로 두자 QP가 my=−2.1 대신 my=−0.1 + **Fx=+2.7 N**을 선택
(모멘트가 비싸니 "싼" 수평력으로 피치 균형). 순수 전진력이 남아 로봇이 밀려가다
넘어짐. **해법: R_m = R_F = 1e-5.** 명세의 "힘/모멘트 가중 분리"는 방향이 반대였다
— 모멘트는 CoP 제약이 물리 한계를 지켜주므로 벌점은 가볍게 두는 게 맞다.

### (3) 자세 가중이 크면 이산 루프가 발산한다 — 대역폭 계산
Q_Θ=300, Q_ω=5로 두자 서자마자 50 Hz 업데이트 주기마다 my가 부호 교대하며
증폭(−2.19/−1.78/−2.26/…) → 0.38 s에 bang-bang 발산. LQR 근사로 추정하면 이
가중은 자세 루프 대역폭 ~6 Hz인데, 업데이트 지연 + wrench 실현 지연에서 위상여유가
안 남는다. **해법: Q_Θ=10, Q_ω=5로 낮춰 ~2 Hz 대역폭 + MPC 100 Hz.**
가중치 튜닝은 "적당히"가 아니라 **대역폭 추정**으로 접근할 것:
`ω_n ≈ (q_θ/r_eff)^(1/4)/√I`.

### (4) pelvis 각속도 ≠ SRB 각속도 — 제일 중요한 발견
(3)까지 고쳐도 밀기만 하면 넘어졌다. 로그를 보니 push 직후 10 ms 만에
wy가 −0.06 → **+0.85 rad/s** 점프 — 강체 회전이면 292 N·m가 필요한데 당시 토크는
14 N·m. 즉 **측정이 이상한 것**: ω를 pelvis(free joint qvel)에서 읽고 있었는데,
외란이 "다리 서스펜션 위에 얹힌 골반"의 내부 진동 모드를 때리면 골반만 고주파로
흔들린다. SRB 모델은 이걸 몸 전체 회전으로 착각 → 과반응 → 진동 증폭 → 발산.
단서: CoM 속도(vx)는 처음부터 전신 값(subtree_linvel)이라 매끈했다.
**해법: SRB 정의 그대로 ω = I_w⁻¹·L (전신 각운동량, MuJoCo subtree_angmom).**
내부 모드가 평균화되어 사라진다. → `get_state(m, d, I_body=...)`
(적대적 검증 §4의 재기술: 이건 "ω가 못 본다"기보다 **"Θ(pelvis)와 ω(L)가 같은
강체가 아니다"** — 상태 안에 운동학 불일치가 내재한다. 걷는 동안 상시 작동하는
SRB의 구조적 결손이고, 상체 PD는 관측을 고친 게 아니라 로봇을 SRB에 강제한 것.)

교훈 요약: **모델–측정–비용이 한 세트다.** SRB로 예측하려면 상태도 SRB 정의로
측정해야 하고(4), 비용은 물리(중력 ff(1), 제약이 지켜주는 변수(2), 실현 지연(3))를
반영해야 한다. 그리고 이걸 전부 **로그가 찾아줬다** — x, u 로그 없었으면 (4)는
못 잡았다.

## 7. 명세에서 달라진 것 총정리

| 항목 | 명세 | 실제 | 이유 |
|---|---|---|---|
| w (롤 반폭) | 0.03 | **0.025** | XML 실측 (뒤꿈치 y=±0.025) |
| Step1 my 기대값 | −3.87 | **−2.11** | 명세 값은 world 원점 기준 (기준점 혼동) |
| u_ref | 0 | **중력 ff** | R이 중력과 싸움 (디버깅 1) |
| R 모멘트 가중 | 힘과 분리(크게) | **힘과 동일 1e-5** | Fx 우회 부작용 (디버깅 2) |
| Q_Θ/Q_ω | (명세엔 없음) | 10/5, ~2 Hz | 이산 루프 안정성 (디버깅 3) |
| ω 측정 | (명세엔 없음) | **I⁻¹L 전신 각운동량** | 내부 모드 오염 (디버깅 4) |
| 솔버 | qpOASES(CasADi) | **quadprog** | 한글 경로 DLL 문제. active-set인 건 동일 |
| MPC 주기 | (명세엔 없음) | 100 Hz | 50 Hz는 위상여유 부족 |

## 8. 로그 & 플롯 사용법

```bash
# 실행하면 logs/ 에 npz+csv 자동 저장 (MPC 스텝마다 t, x, x_ref, u, tau, solve_ms, ...)
.venv/Scripts/python.exe MPC/src/08_standing_mpc.py            # 헤드리스 20s+push40N
.venv/Scripts/python.exe MPC/src/08_standing_mpc.py --push 80  # 외란 크기 변경
.venv/Scripts/python.exe MPC/src/08_standing_mpc.py --view     # 뷰어 (Ctrl+우클릭 드래그=밀기)

.venv/Scripts/python.exe MPC/src/plot_log.py                   # 최근 로그 → PNG
.venv/Scripts/python.exe MPC/src/plot_log.py MPC/logs/xxx.npz      # 지정 로그
```
csv는 엑셀에서 바로 열린다 (열: t, x 13개, ref 13개, u 12개, solve_ms, violation).

튜닝 지점 (사용자 몫):
- [src/mpc_qp.py](src/mpc_qp.py) 상단 `Q_DEFAULT`, `R_FORCE`, `R_MOMENT`
- [src/08_standing_mpc.py](src/08_standing_mpc.py) 상단 `HORIZON`, `DT_MPC`, `DECIM`
- `make_params(mu=, fz_min=, fz_max=)`

---

# Step 5–6: 보행 (2026-09-10 오후)

## 10. 무엇이 추가됐나

| 파일 | 역할 |
|---|---|
| [src/gait.py](src/gait.py) | `Gait`(고정 타이밍 스케줄), `raibert_target`(capture point 착지점), `SwingController`(궤적+임피던스) |
| [src/mpc_qp.py](src/mpc_qp.py) `solve_gait` | 접촉 스케줄 반영: **시변 B_d[k]** + swing 발 **W=0 등식 제약** (quadprog meq) |
| [src/09_walk.py](src/09_walk.py) | `WalkController` + Step 5/6 검증 (`--vx`, `--view`, `--seconds`) |

구조 (매 사이클):
```
[100 Hz] 접촉 스케줄(N,2) + sway 참조 + 발 위치 계획(N,2,3) 조립 → solve_gait
[500 Hz] τ = qfrc_bias − Σ_stance JᵀW + Σ_swing Jᵀ(임피던스) + 상체 PD
```
- gait: T=0.8 s, stance 75% (swing 0.2 s, double support 자연 발생), 좌우 위상 0.5 차
- MPC: N=16, dt=0.04 (지평 0.64 s = 주기의 80%), 평균 4.8 ms/solve
- swing 발은 지평에서 B 열을 0으로 + 해당 u 6개 = 0 등식 → "그 발로는 못 민다"를
  미리 알림 → **발 떼기 전에 반대발로 무게가 옮겨지는 것이 QP에서 저절로 나온다**

## 11. 결과

| 시험 | 결과 |
|---|---|
| Step 5 제자리 스텝 12 s | **통과** — 발당 14 swing, apex 6.8 cm, CoM y 진동 유계 |
| Step 6 전진 0.15 m/s | **통과** — 평균 vx 0.146 (97% 추종) |
| 0.15 m/s **40 s 내구** | **통과** — 발당 49스텝, ~5.8 m 전진, y ±6.7 cm 유계 (12-(7)의 부호 수정 후) |
| 0.2 / 0.25 / 0.3 m/s | 9.2 / 4.8 / 4.1 s 에 넘어짐 — **속도 한계, 아래 12-(6)** |

플롯: [logs/walk_vx0p15_20260910_193623.png](logs/walk_vx0p15_20260910_193623.png)
— Fz 좌우 교대 구형파, vx 0.15 정착, sway 참조 추종, 발목 모멘트 교대.

## 12. 디버깅 기록 (보행 편) — 6번 터졌다

**(1) swing 임피던스가 무르면 착지가 한 박자 늦는다.**
kp=350(고유진동수 1.3 Hz)으로는 0.2 s swing 궤적을 못 쫓아 발이 정점도 못 찍고
공중에 뜬 채 '스케줄상 착지 시각'을 맞는다 → MPC는 접지됐다고 믿고 300 N을 기대
→ 그쪽으로 툭 낙하 → 매 스텝 roll 킥 누적 → 발산. **kp=2500 (3.5 Hz)** 로 해결.

**(2) 다리를 던지면 상체가 반대로 돈다 (고양이 회전).**
swing 반작용으로 골반 pitch가 0.2 s 만에 +21.9° — 그런데 각운동량 기반 ω는 ≈0.
내부 운동은 L에 안 잡히므로 **MPC의 ω는 이 회전을 정의상 못 본다** (Θ 피드백만
남음). 대책 둘: ⓐ **상체(허리+팔 17관절) 자세 PD** — 중력보상만 받던 상체를 고정해
로봇을 SRB 가정에 가깝게 만듦. ⓑ swing 힘 상한. roll 은 발볼 반폭 2.5 cm 라 발목 권한이 ~8 N·m 뿐이라 특히 취약했다.
(정정: 초판에 'I_xx=0.54 로 작아서'라 썼는데 0.542 는 I_zz(요)다. I_xx=3.55.
롤 취약의 원인은 관성이 아니라 발목 권한 — 적대적 검증에서 잡힌 오기.)

**(3) 착지점을 현재 CoM 기준으로만 잡으면 발디딤이 표류를 못 잡는다.**
CoM이 옆으로 밀리면 발도 따라가서 아무것도 위치를 고정하지 않음 (y −0.155 까지
표류). 또 Raibert 원형 `v·T_st/2 + k·v` 는 속도계수 0.48 — 역진자 정답(capture
point) `1/ω₀ = 1/√(g/h) = 0.26` 의 2배라 **매 스텝 과잉 스텝 → 에너지 주입**.
→ [src/gait.py](src/gait.py) `raibert_target` 을 capture point 기반으로 재작성:
`p = CoM + offset + v_cmd·T_st/2 + (v−v_cmd)/ω₀ + k_a(anchor−CoM)` (y만 약한 앵커).

**(4) py 참조가 항상 중앙이면 MPC가 체중이동 없이 발목으로만 버틴다.**
지평의 접촉 스케줄로 지지 중심을 계산해 **sway 참조** (β=0.5) 를 넣음 — preview 가
있어야만 가능한 정석 체중이동. 이것까지 넣고 Step 5 통과.

**(5) 단, sway 참조를 스텝 함수로 넣으면 안 된다.**
접촉 전환 순간 참조 점프(0.06 m/0.04 s = vy_ref 1.5 m/s!)를 MPC가 쫓아 옆으로
차버림 → 1.5 s 만에 넘어짐. **속도 제한 램프(0.4 m/s)로 스무딩** 후 해결.
vy_ref 는 0 유지 (미분 넣으면 같은 문제).

**(6) 속도 한계 = massless-leg 가정이 깨지는 지점. ★ 논문감**
0.3 m/s 에서: swing 다리가 0.35 m 를 0.2 s 에 가야 해서 임피던스가 ~200 N —
그 반작용이 몸통(유효 ~28 kg)을 **−5 m/s² 로 제동** → vx 0.3→−0.1 붕괴 → 몸이
뒤로 기움 → G1 뒤꿈치가 5 cm 뿐이라 (my ≤ +0.05·Fz) 발목으로 못 잡고 후방 전도.
SRB(다리 무질량) MPC는 이 반작용을 원천적으로 모른다. **다리 질량 ~30%인 휴머노이드
에서 convex MPC의 속도 한계가 실체로 나타난 것 — residual RL이 메울 gap 그 자체.**
당장의 우회는 감속(0.15 m/s)뿐.
(이 진단은 처음엔 로그 해석뿐이었으나 §16의 격리 실험 — 다리질량 1/3 → 0.3 m/s
완주, swing 힘 상한 400 N → 무효 — 으로 인과가 확정됐다.)

**(7) 착지점 '위치' 피드백의 부호 — 사용자의 quadruped MATLAB 코드가 정답을 줬다. ★**
(1)~(6)까지 고쳐도 횡 진동이 시정수 ~20 s 로 서서히 자라 23 s 에 넘어졌다.
사용자의 convex_mpc.m `get_foot_trajectory` 에 있던 한 줄:
```matlab
p_ref_adj(2) = p_ref(2) + 0.3*y_com + 0.2*vy     % k_y=0.3 (위치!), k_vy=0.2
```
CoM 이 밀린 쪽으로 발을 **더** 내딛는 +부호 위치 피드백이다 — CoM 이 발 안쪽에
놓여 중력이 되돌린다. 반면 우리 anchor 는 발디딤을 중앙 쪽으로 당기는 **반대
부호**였다 (CoM 이 발 바깥 → 위치적으로 불안정). 부호를 뒤집고 게인 스윕:

| k_pos (0.15 m/s, 25 s 시험) | 결과 |
|---|---|
| 0.0 (위치항 없음) | 8.1 s 전도 |
| **+0.15** | **25 s 완주** → 40 s 시험도 완주 (49스텝, 5.8 m) |
| +0.3 (quadruped 값 그대로) | 15.6 s 전도 (biped 에는 과다) |
| −0.3 (이전 anchor) | 23 s 전도 |

교훈: 발디딤에는 **속도 피드백(capture, 1/ω₀)과 위치 피드백(+k_pos) 둘 다**
필요하고, 위치항의 부호는 "밀린 쪽으로 더 딛기"다. → [src/gait.py](src/gait.py)

## 13. 튜닝 민감도 기록 (같은 날 실측 — 건드릴 때 참고)

| 변경 | 생존시간 (0.15 m/s, 25 s 시험) |
|---|---|
| **확정 구성 (capture 1/ω₀ + k_pos=+0.15, β=0.5)** | **25 s+ (40 s 완주)** |
| k_pos = 0 | 8.1 s |
| k_pos = +0.3 | 15.6 s |
| k_pos = −0.3 (부호 반대, 이전 anchor) | 23 s |
| capture 계수 +0.06 | 9.3 s ↓ |
| β=0.65 + Q_vy=40 동시 | 6.1 s ↓ |
| (참고) 속도계수 0.48 (Raibert 원형) | 9.5 s ↓ |

속도 게인은 1/ω₀ 정확값, 위치 게인은 +0.15 가 최선 — 둘 다 산 모양으로 민감하다.

**지평 길이 스윕 (vx=0.2, 20 s 시험):**

| N × dt = 지평 | 생존 | 비고 |
|---|---|---|
| 16 × 0.04 = 0.64 s (주기의 80%) | 9.2 s | 이전 기본값 |
| 20 × 0.04 = 0.80 s | 6.0 s | 같은 지평인데 악화 — dt 조합 민감 |
| **16 × 0.05 = 0.80 s (주기 100%)** | **13.4 s** | **확정값.** 매트랩도 stride 100% + dt 0.05 |
| 24 × 0.05 = 1.20 s | 11.9 s | 더 길다고 좋아지지 않음. QP 16.5 ms |

→ **한 주기를 꽉 채우는 것까지는 이득, 그 이상은 무의미.** 0.15 m/s 40 s 내구는
확정값에서도 완주. 지평이 0.2 m/s 벽을 못 넘는 것은 원인이 preview 부족이 아니라
모델 오차(12-(6) swing 반작용)라는 방증.

## 15. convex_mpc.m (사용자의 quadruped MATLAB) 대조

층1 이식이 실제로 1:1 임을 확인: `get_A_dynamics`↔`continuous_AB`(A, 동일),
`get_B_dynamics`↔B(발당 3열→6열), `build_qp_matrices`↔`solve_gait` 조립(동일),
`get_dynamic_contact`↔`contact_table`+swing 등식, `get_force_constraints_seq`
(발당 6행)↔`foot_constraints`(10행, 모멘트 4행 추가), quadprog↔quadprog.

**주의해서 읽을 점**: MATLAB 의 플랜트(`run_nonlinear_physics`)는 **SRB 자체**다
— 다리·접촉 솔버 없음, 발은 그림용, 땅은 z 클램프. 모델=플랜트라서 MuJoCo 에서
우리가 맞은 문제(고양이 회전, swing 반작용 제동, 늦은 착지)가 원리적으로 발생
불가능한 세계이고, 1.5 m/s trot 이 쉽게 됐던 이유다. G1 의 난이도는 코드가 아니라
플랜트가 진짜 다물체라는 데서 온다. (그 와중에 발디딤 위치 피드백 부호(12-(7))라는
결정적 힌트를 줬다.) 사소한 것: T_end=3.0 인데 외란이 t≥5.0 이라 외란은 죽은 코드,
r_ref_history 미기록.

## 14. 다음

- [x] 횡 진동 장기 성장 해결 — 착지점 위치 피드백 부호 수정 (12-(7)), 40 s 완주
- [ ] 보행 속도 확장 — swing 반작용 feedforward 또는 residual RL (12-(6))
- [ ] mz 제약 (Caron CWC), Bezier swing 정밀화
- [ ] OSQP와 해 비교 (병렬화 ADMM 타당성 판단 — 층2 연구)
- [ ] 그 다음: residual RL

---

# 16. 적대적 검증 반영 (2026-09-10 저녁, claude.ai 검증 리포트)

VERIFY_BUNDLE.md 를 claude.ai(Fable 5.1)에 넘겨 받은 적대적 검증의 반영 기록.
리포트 원문은 [VERIFY_REPORT.md](VERIFY_REPORT.md).

## 16.1 ★ 확정 버그: CoP 제약의 h 결합항 누락 → 수정 + 완전 검증

wrench 는 발 site(발목)에서 정의되는데 CoP 조건은 발바닥(3.5 cm 아래)의 wrench 에
대한 것 — 수평력이 h 팔길이로 CoP 를 옮기는 항이 빠져 있었다:

```
m_ground,y = my + h·Fx,   m_ground,x = mx − h·Fy      (h = h_sole = 0.035, 모델 추출)
```

- 크기: |Fx|=μFz 에서 h·μ = 0.021·Fz — **롤 마진(0.025·Fz)의 84%**, 뒤꿈치 마진의 42%
- 증상 연결: 가속 중(Fx>0) MPC 가 뒤꿈치 CoP 권한을 과대평가 → **후방 전도** —
  0.25/0.3 m/s 실패 양상과 정확히 일치
- 서 있기(Fx≈0)에선 항이 0 이라 Step 1 이 못 잡았다 — **Step 1 은 이 항을 검증할 수
  없는 테스트였다** (검증 리포트의 정확한 지적)
- yaw≠0 잠복 버그(제약이 world x=앞 가정)도 같이 수정: `C·(Rz(ψ)ᵀW) ≤ d`

**검증** ([src/10_constraint_check.py](src/10_constraint_check.py)): 접촉점 4개에
무작위 점힘(음수 fz·마찰 초과 포함 → 위반 케이스도 생성)을 뿌려, 점힘에서 직접 계산한
지면 CoP(경로 1)와 site wrench 로 조립한 제약(경로 2)을 대조:

| | 일치율 (2만 샘플 × yaw 0/0.7) |
|---|---|
| 새 제약 (h 항 포함) | **100.00 %** |
| 옛 제약 (h=0) | 91.2 % — **오판 ~1,700건/2만** |

## 16.2 수정 후 재측정 — h 버그의 기여분

| 시험 | 수정 전 | 수정 후 |
|---|---|---|
| 0.15 m/s 40 s | 완주, y ±8.2 cm | 완주, **y ±3.7 cm (반감)** |
| 0.2 m/s | 13.4 s | **16.0 s** |
| 0.25 m/s | 4.8 s | 6.5 s |
| 0.3 m/s | 4.1 s | 6.4 s |

→ h 결함은 **기여 요인이었지만 벽 자체는 아니었다.** (리포트의 "얼마나 h 탓인지
갈려" 질문의 답: 부분 기여.)

## 16.3 ★ 격리 실험 — massless-leg 가설 인과 확정

리포트 §5: "속도 벽 = swing 반작용"으로 단정하려면 대안 가설을 배제하라.

| 실험 | 결과 | 판정 |
|---|---|---|
| **다리 링크 12개 질량 ×1/3** (`--legmass 0.33`), 0.3 m/s | **20 s 완주** (원래 질량은 6.4 s 전도, 같은 게인) | **massless-leg 확정** |
| 다리 ×1/3, 0.5 m/s | 4.6 s 전도 | 벽이 사라진 게 아니라 밀림 — 질량에 비례 |
| swing 힘 상한 200→400 N, 원래 질량 0.3 | 5.6 s (변화 없음), 0.2 는 오히려 악화 | **"늦은 착지" 대안가설 (a) 배제** — 그 가설이면 개선됐어야 함 |

이로써 "0.2 m/s 벽 = swing 다리 관성 반작용 = SRB massless-leg 가정 위반"이
로그 해석이 아니라 **통제 실험으로 확정**됐다. 논문에 쓸 수 있는 형태.

## 16.4 반영한 사소 수정 일괄

- **이산화**: Euler → 정확한 ZOH `A_d = I + A·dt + ½A²dt²` (A 멱영이라 닫힌 형태)
- **k=0 선형화**: 지평 첫 스텝 r 을 참조가 아닌 **현재 상태** CoM 기준으로 (Di Carlo /
  사용자 MATLAB 의 i=1=x_now 방식)
- **stage-0 접촉**: `contact[0] = in_stance(t)` (중점 t+25ms 는 착지 25 ms 전부터
  허공에 W>0 명령 — torque() 와 불일치)
- **I_body 프레임**: 추출 시 Rz(ψ₀)ᵀ 로 body frame 저장, `get_state` 가 현재 yaw 로
  회전해 사용 (yaw₀≠0 이중회전 잠복 버그)
- **LIP 높이**: ω₀ = √(g/(z_com − z_ground)) — CoM 절대높이가 아니라 지면 기준
- **노트 오기**: "롤은 I_xx=0.54 로 작아서" → 0.542 는 I_zz. 롤 취약 원인은 발목 권한
- 재기술: "ω가 못 본다" → **"Θ와 ω가 같은 강체가 아니다"** (상태 내 운동학 불일치)

전부 반영 후 **서 있기 Step 1–4, 보행 Step 5–6 전체 회귀 통과.**

## 16.5 리포트에 대한 반론 한 가지 (기록)

리포트 §5-(b)의 "swing 반작용을 d 항으로 SRB 동역학에 주입"은 **병진 채널에는
해당 없다**: 우리 상태 p·v 는 전신 CoM 이고, CoM 병진 `v̇ = ΣF_ext/M − g` 는 내부
운동과 무관하게 정확하다 (CoM 은 외력으로만 움직인다). 회전 채널도 CoM 기준
`dL/dt = Σr×F + m` 은 정확. swing 반작용이 실제로 새는 곳은:
1. **Θ(pelvis) 운동학** — Θ̇ ≠ Rz(ψ)ᵀ·(I⁻¹L) (§16.4 재기술의 그 불일치)
2. **토크 실현** — τ = qfrc_bias − JᵀW 의 준정적 가정이 swing 가속 중 깨져
   실현된 wrench ≠ 명령 wrench (0.3 m/s 로그에서 Fx 명령 +50 인데 vx 하락이 증거)

따라서 feedforward 는 SRB 의 d 항이 아니라 **실현 계층**(τ 에 swing 관성 보상항,
사실상 WBC-lite) 또는 Θ 운동학 보정에 넣어야 한다. 리포트 (c)의 "양극성(가속→감속)
패턴" 지적은 이 채널에서도 유효. → 다음 과제.

## 16.6 남은 과제 (리포트 우선순위 반영)

- [ ] swing 반작용 보상 baseline — 실현 계층에 (16.5 참조). **residual RL 의 정직한
      비교 대상**: 이것 없이 "RL이 gap 메움"이라 하면 심사 첫 질문이 "feedforward
      한 줄이면 되는 것 아니냐"가 된다 (리포트의 정확한 지적).
- [ ] gait 타이밍 스윕 (고속에서 T·swing_frac) — 리포트 (e)
- [ ] 임피던스 반작용 vs SRB 예측 GRF 나란히 로깅 — 모델 오차 직접 수치화, 리포트 (3)
- [ ] side_offset x +2 cm (공칭 발 위치 후방 편향) 검토 — 리포트 관찰 ②



---

# FILE: src/mpc_srb.py

```python
"""SRB(single rigid body) 모델 — 층1 재료 (상태방정식).

상태 x ∈ R13 : [roll, pitch, yaw, px, py, pz, wx, wy, wz, vx, vy, vz, g]
                Θ(3)         p(3, CoM)    ω(3, world) ṗ(3, CoM)   중력상수
입력 u ∈ R12 : [W_L(6), W_R(6)],  W = [Fx, Fy, Fz, mx, my, mz]
                발 site(발목 원점)에 작용하는 wrench (world frame)

연속 동역학 (Di Carlo convex MPC + wrench 확장):
    Θ̇ = Rz(ψ)ᵀ ω                        (작은 roll/pitch 근사, 논문 식 12)
    ṗ = v
    ω̇ = I_w⁻¹ Σ_i (r_i × F_i + m_i)      r_i = 발_i site − CoM (world, CoM 기준!)
    v̇ = (1/M) Σ_i F_i − g ẑ
    ġ = 0
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

NX = 13
NU_PER_FOOT = 6
N_FEET = 2
NU = NU_PER_FOOT * N_FEET
GRAV = 9.81
FOOT_SITES = ("left_foot", "right_foot")

X_LABELS = ["roll", "pitch", "yaw", "px", "py", "pz",
            "wx", "wy", "wz", "vx", "vy", "vz", "g"]
U_LABELS = [f"{s}_{c}" for s in ("L", "R")
            for c in ("Fx", "Fy", "Fz", "mx", "my", "mz")]


def rz(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def skew(v) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def quat_to_euler_zyx(quat) -> tuple[float, float, float]:
    """MuJoCo quat (w,x,y,z) -> (roll, pitch, yaw), R = Rz(yaw)Ry(pitch)Rx(roll)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    R = R.reshape(3, 3)
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


@dataclass
class SRBParams:
    """SRB 파라미터 — 전부 모델에서 추출 (하드코딩 금지)."""
    mass: float
    I_body: np.ndarray        # 3x3, CoM 기준 합성 관성 (body frame — yaw₀ 제거됨)
    l_t: float                # 발가락까지 (발목 기준, +x)
    l_h: float                # 뒤꿈치까지 (발목 기준, -x)
    w: float                  # 좌우 반폭 (보수적: 접촉점 |y| 최솟값 = 0.025)
    h_sole: float             # 발 site 에서 발바닥(접촉면)까지 수직거리 (+0.035)
    mu: float = 0.6
    fz_min: float = 10.0
    fz_max: float = 500.0


def make_params(m, d, mu=0.6, fz_min=10.0, fz_max=500.0) -> SRBParams:
    """현재 자세(base가 수직인 crouch)에서 SRB 파라미터 추출."""
    mass = mujoco.mj_getTotalmass(m)

    # CoM 기준 합성 관성 — world 에서 합성 후 yaw₀ 를 제거해 body frame 으로 저장.
    # (world 값을 그대로 두면 continuous_AB 의 Rz(ψ) 와 이중 회전 — 검증 리포트 §2.
    #  yaw₀=0 인 지금은 동일하지만 요 회전 대비.)
    com = d.subtree_com[0].copy()
    inertia = np.zeros((3, 3))
    for b in range(1, m.nbody):
        R = d.ximat[b].reshape(3, 3)
        I_b = R @ np.diag(m.body_inertia[b]) @ R.T
        r = d.xipos[b] - com
        mb = m.body_mass[b]
        inertia += I_b + mb * (r @ r * np.eye(3) - np.outer(r, r))
    yaw0 = quat_to_euler_zyx(d.qpos[3:7])[2]
    R0 = rz(yaw0)
    inertia = R0.T @ inertia @ R0

    # 발 형상: ankle_roll body 의 접촉 geom local 위치에서 추출
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    geoms = [g for g in range(m.ngeom)
             if m.geom_bodyid[g] == bid and m.geom_group[g] == 3]
    pos = np.array([m.geom_pos[g] for g in geoms])
    l_t = float(pos[:, 0].max())
    l_h = float(-pos[:, 0].min())
    w = float(np.abs(pos[:, 1]).min())
    # site(발목 원점) -> 발바닥 접촉면 거리: 구 중심 z(-0.03) - 반지름(0.005)
    h_sole = float(-(pos[:, 2].min() - m.geom_size[geoms[0], 0]))

    return SRBParams(mass=mass, I_body=inertia, l_t=l_t, l_h=l_h, w=w,
                     h_sole=h_sole, mu=mu, fz_min=fz_min, fz_max=fz_max)


# ---------------------------------------------------------------------------
# 상태 추출
# ---------------------------------------------------------------------------
def get_state(m, d, I_body: np.ndarray | None = None) -> np.ndarray:
    """MuJoCo data -> SRB 상태 x(13). mj_forward 가 이미 불린 상태를 가정.

    I_body(body frame) 를 주면 ω 를 전신 각운동량으로 계산:
        ω = (Rz(ψ) I_body Rz(ψ)ᵀ)⁻¹ L      (SRB 정의 그대로, 현재 yaw 반영)
    ⚠ 안 주면 pelvis 각속도를 쓰는데, 외란 시 '다리 서스펜션 위 골반'의 내부
    진동 모드가 그대로 들어와 MPC가 과반응한다 (Step 4 디버깅에서 발산 원인).
    CoM 속도는 처음부터 전신(subtree_linvel)이라 이 문제가 없었다.
    주의(검증 리포트 §4): 이 ω 는 내부 재배향(고양이 회전)에 0 을 낸다 — Θ(pelvis)
    와 ω(L)가 '같은 강체'가 아니라는 상태 불일치가 SRB 의 구조적 한계.
    """
    roll, pitch, yaw = quat_to_euler_zyx(d.qpos[3:7])
    p = d.subtree_com[0]

    mujoco.mj_subtreeVel(m, d)
    if I_body is not None:
        Rz_ = rz(yaw)
        I_w = Rz_ @ I_body @ Rz_.T
        omega_world = np.linalg.solve(I_w, d.subtree_angmom[0])
    else:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        omega_world = R.reshape(3, 3) @ d.qvel[3:6]   # pelvis ω (body->world)

    v_com = d.subtree_linvel[0]

    x = np.empty(NX)
    x[0:3] = (roll, pitch, yaw)
    x[3:6] = p
    x[6:9] = omega_world
    x[9:12] = v_com
    x[12] = GRAV
    return x


def get_foot_positions(m, d) -> np.ndarray:
    """(2,3) 발 site world 위치."""
    out = np.zeros((N_FEET, 3))
    for i, s in enumerate(FOOT_SITES):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)
        out[i] = d.site_xpos[sid]
    return out


# ---------------------------------------------------------------------------
# 동역학 행렬
# ---------------------------------------------------------------------------
def continuous_AB(params: SRBParams, psi: float,
                  r_feet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A_c(13x13), B_c(13x12).  r_feet: (2,3) = 발 site − CoM (world frame)."""
    A = np.zeros((NX, NX))
    A[0:3, 6:9] = rz(psi).T          # Θ̇ = Rz(ψ)ᵀ ω
    A[3:6, 9:12] = np.eye(3)         # ṗ = v
    A[11, 12] = -1.0                 # v̇_z ← −g (중력상수 상태)

    Rz_ = rz(psi)
    I_w = Rz_ @ params.I_body @ Rz_.T
    I_w_inv = np.linalg.inv(I_w)

    B = np.zeros((NX, NU))
    for i in range(N_FEET):
        c = NU_PER_FOOT * i
        B[6:9, c:c + 3] = I_w_inv @ skew(r_feet[i])   # ω̇ ← r×F
        B[6:9, c + 3:c + 6] = I_w_inv                  # ω̇ ← m  (wrench 신규)
        B[9:12, c:c + 3] = np.eye(3) / params.mass     # v̇ ← F
    return A, B


def discretize(Ac: np.ndarray, Bc: np.ndarray, dt: float):
    """정확한 ZOH (A_c 가 멱영: A³=0 이라 닫힌 형태 — 검증 리포트 §2).

        A_d = I + A·dt + ½A²·dt²
        B_d = (I·dt + ½A·dt²) B

    Euler(I + A·dt) 대비 추가되는 항은 ½g·dt²(자유낙하) 등 — dt=0.05 에서
    스텝당 1.2 cm. 평형에선 상쇄되지만 과도상태 preview 정확도에 기여.
    """
    A2 = Ac @ Ac
    Ad = np.eye(NX) + Ac * dt + 0.5 * A2 * dt * dt
    Bd = (np.eye(NX) * dt + 0.5 * Ac * dt * dt) @ Bc
    return Ad, Bd

```


---

# FILE: src/mpc_qp.py

```python
"""Wrench MPC — 제약 조립(층1) + condensed QP + 솔버(층2).

QP (Di Carlo condensed formulation):
    X = A_qp x0 + B_qp U
    min_U  ½ Uᵀ H U + gᵀ U
      H = 2 (B_qpᵀ Q̄ B_qp + R̄)
      g = 2 B_qpᵀ Q̄ (A_qp x0 − X_ref)
    s.t.  C U ≤ d      (마찰콘 + Fz 범위 + CoP 모멘트, 발당 10행 × 2발 × N)

솔버: quadprog (Goldfarb–Idnani active-set, 고정밀 — 명세 9절 "quadprog 계열")
      → 실패 시 OSQP (ADMM) 폴백.
      ⚠ CasADi conic(qpOASES 등)은 한글 경로에서 플러그인 DLL 로딩 실패(WIN32 126)로
        사용 불가. CasADi 는 심볼릭 자동미분(Step 3 검증)에만 사용한다.
"""
from __future__ import annotations

import time

import numpy as np
import mujoco
import quadprog

from mpc_srb import (NX, NU, NU_PER_FOOT, N_FEET, FOOT_SITES, SRBParams,
                     continuous_AB, discretize)


# ---------------------------------------------------------------------------
# 제약 (발 하나, 명세 3절 C_foot 10행)
# ---------------------------------------------------------------------------
def foot_constraints(p: SRBParams, psi: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """C_foot(10x6), d_foot(10):  C·W ≤ d,  W = [Fx,Fy,Fz,mx,my,mz] (world frame).

    CoP 제약의 h 결합항 (검증 리포트 §1에서 잡은 버그):
      wrench W 는 발 site(발목 원점)에서 정의되지만 CoP 조건은 발바닥 접촉면
      (site 아래 h_sole=0.035 m)의 wrench 에 대한 것. 지면 wrench 로 옮기면
        m_ground,y = my + h·Fx,   m_ground,x = mx − h·Fy
      이므로 수평력이 h 팔길이로 CoP 를 이동시킨다. 크기: |Fx|=μFz 에서
      h·μ = 0.021·Fz — 롤 마진(0.025·Fz)의 84%, 뒤꿈치 마진(0.05·Fz)의 42%.
      이 항이 없으면 가속 중(Fx>0) MPC 가 뒤꿈치 권한을 과대평가 → 후방 전도.
      서 있기(Fx≈0)에선 0 이라 Step 1 검증에 안 잡혔던 것.

    psi: 발/제약은 yaw frame 에서 정의되므로 world wrench 를 Rzᵀ 로 돌려 적용
      (yaw≈0 이면 항등 — 요 회전 대비 잠복 버그 수정, 리포트 §1).
    """
    mu, w, lt, lh, h = p.mu, p.w, p.l_t, p.l_h, p.h_sole
    C = np.array([
        [1,  0, -mu,  0,  0, 0],    # Fx ≤ μFz
        [-1, 0, -mu,  0,  0, 0],    # −Fx ≤ μFz
        [0,  1, -mu,  0,  0, 0],    # Fy ≤ μFz
        [0, -1, -mu,  0,  0, 0],    # −Fy ≤ μFz
        [0,  0,  1,   0,  0, 0],    # Fz ≤ Fz_max
        [0,  0, -1,   0,  0, 0],    # −Fz ≤ −Fz_min
        [0, -h, -w,   1,  0, 0],    # mx − h·Fy ≤ w·Fz    (롤 CoP)
        [0,  h, -w,  -1,  0, 0],    # −mx + h·Fy ≤ w·Fz
        [h,  0, -lh,  0,  1, 0],    # my + h·Fx ≤ l_h·Fz  (피치 CoP, 뒤꿈치)
        [-h, 0, -lt,  0, -1, 0],    # −my − h·Fx ≤ l_t·Fz (피치 CoP, 발가락)
    ], dtype=float)
    if abs(psi) > 1e-12:
        Rz_ = np.zeros((6, 6))
        from mpc_srb import rz
        Rz_[:3, :3] = Rz_[3:, 3:] = rz(psi)
        C = C @ Rz_.T                # 제약은 발(yaw) frame: C·(Rzᵀ W) ≤ d
    d = np.array([0, 0, 0, 0, p.fz_max, -p.fz_min, 0, 0, 0, 0], dtype=float)
    return C, d


# ---------------------------------------------------------------------------
# MPC 본체
# ---------------------------------------------------------------------------
# 기본 가중치 (임의 초기값 — 사용자가 튜닝 예정. Di Carlo Table I 스케일 참고)
#   x = [roll,pitch,yaw, px,py,pz, wx,wy,wz, vx,vy,vz, g]
# Step4 디버깅: Q_Θ=300/Q_ω=5 는 자세 루프가 과소감쇠 → 50Hz 업데이트에서 진동 발산.
# 자세 강성을 낮추고 각속도 감쇠를 올려 안정화. (튜닝 여지 큼 — 사용자 몫)
# LQR 근사 추정으로 자세 루프 대역폭 ~2Hz 가 되게 낮춤 (Q_Θ=100 은 ~6Hz → 100Hz
# 샘플링/실현 지연에서 위상여유 부족, push 후 bang-bang 발산 — Step4 디버깅 2차)
Q_DEFAULT = np.array([10, 10, 20,   50, 50, 300,   5, 5, 5,   20, 20, 50,   0],
                     dtype=float)
# R: 힘(N)과 모멘트(N·m) 스케일 분리 (명세 7절 — 균일 금지)
R_FORCE = 1e-5
R_MOMENT = 1e-5   # 힘과 동일. 1e-3 으로 크게 두면 QP가 my 대신 Fx로 피치를 만들어 전진해버림 (Step2 디버깅)
R_DEFAULT = np.tile(np.array([R_FORCE] * 3 + [R_MOMENT] * 3), N_FEET)


class WrenchMPC:
    """시간지평 N 의 wrench MPC. N=1 이면 1스텝 QP (Step 2)."""

    def __init__(self, params: SRBParams, horizon: int = 10, dt: float = 0.02,
                 q_diag=None, r_diag=None, psi0: float = 0.0):
        self.p = params
        self.N = horizon
        self.dt = dt
        self.Q = np.diag(Q_DEFAULT if q_diag is None else np.asarray(q_diag, float))
        self.R = np.diag(R_DEFAULT if r_diag is None else np.asarray(r_diag, float))
        self.Q_bar = np.kron(np.eye(self.N), self.Q)
        self.R_bar = np.kron(np.eye(self.N), self.R)

        Cf, df = foot_constraints(params, psi=psi0)
        self.Cf, self.df = Cf, df                       # gait 모드에서 재사용
        C_step = np.kron(np.eye(N_FEET), Cf)            # 20 x 12
        d_step = np.tile(df, N_FEET)
        self.C_all = np.kron(np.eye(self.N), C_step)    # 20N x 12N
        self.d_all = np.tile(d_step, self.N)

        self.nU = NU * self.N
        self.nC = self.C_all.shape[0]
        self.solver_name = "quadprog"
        self.last = {}                                  # 디버그용 (H, g, U, ...)

    def _solve_qp(self, H, g):
        """min ½uᵀHu + gᵀu  s.t.  C_all·u ≤ d_all.  해 u 반환."""
        try:
            # quadprog: min ½xᵀGx − aᵀx  s.t. Cᵀx ≥ b   →  G=H, a=−g, C=−C_allᵀ, b=−d_all
            u, *_ = quadprog.solve_qp(H, -g, -self.C_all.T, -self.d_all)
            self.solver_name = "quadprog"
            return u
        except Exception:               # noqa: BLE001 — 수치 실패 시 OSQP 폴백
            import scipy.sparse as sp
            import osqp
            prob = osqp.OSQP()
            prob.setup(P=sp.csc_matrix(H), q=g, A=sp.csc_matrix(self.C_all),
                       l=-np.inf * np.ones(self.nC), u=self.d_all,
                       verbose=False, eps_abs=1e-8, eps_rel=1e-8)
            res = prob.solve()
            self.solver_name = "osqp"
            return np.asarray(res.x)

    def gravity_u_ref(self) -> np.ndarray:
        """중력 피드포워드 u_ref: 두 발이 Mg/2 씩 수직력, 모멘트 0."""
        u = np.zeros(NU)
        u[2] = u[8] = self.p.mass * 9.81 / N_FEET
        return u

    def solve(self, x0: np.ndarray, x_ref: np.ndarray, psi: float,
              r_feet: np.ndarray, u_ref: np.ndarray | None = None):
        """한 번 풀기.

        x0    : 현재 상태 (13,)
        x_ref : 참조 — (13,) 이면 전 구간 동일, (N,13) 이면 스텝별
        psi   : yaw (선형화 기준)
        r_feet: (2,3) 발 site − CoM (world). 지평 내내 고정 (서 있기 가정)
        u_ref : 입력 정규화 기준 (12,). None 이면 0.
                ⚠ u_ref=0 이면 R 이 중력과 싸워 Fz 가 Mg 보다 작아진다
                (N=1, R_F=1e-5 에서 22% 부족 → 가라앉음, Step 2 디버깅에서 확인).
                중력 ff(gravity_u_ref)를 주면 R 은 '기준에서 벗어남'만 벌점.

        Returns: (u0(12,), U(12N,), info dict)
        """
        t0 = time.perf_counter()
        Ac, Bc = continuous_AB(self.p, psi, r_feet)
        Ad, Bd = discretize(Ac, Bc, self.dt)

        # A_qp = [Ad; Ad²; …; Ad^N],  B_qp 하삼각 블록
        N = self.N
        A_qp = np.zeros((NX * N, NX))
        B_qp = np.zeros((NX * N, NU * N))
        A_pow = np.eye(NX)
        for k in range(N):
            A_pow = Ad @ A_pow                       # Ad^{k+1}
            A_qp[NX * k:NX * (k + 1)] = A_pow
        for k in range(N):          # 블록 (k, j): Ad^{k-j} Bd
            blk = Bd.copy()
            for j in range(k, -1, -1):
                B_qp[NX * k:NX * (k + 1), NU * j:NU * (j + 1)] = blk
                blk = Ad @ blk
        # 위 루프는 (k,j)에 Ad^{k-j}Bd 를 채운다 (j=k 일 때 Bd)

        x_ref = np.asarray(x_ref, float)
        X_ref = np.tile(x_ref, N) if x_ref.ndim == 1 else x_ref.reshape(-1)

        H = 2.0 * (B_qp.T @ self.Q_bar @ B_qp + self.R_bar)
        H = 0.5 * (H + H.T)                          # 대칭화 (수치)
        g = 2.0 * B_qp.T @ self.Q_bar @ (A_qp @ x0 - X_ref)
        if u_ref is not None:                        # ||u−u_ref||_R 항
            g -= 2.0 * self.R_bar @ np.tile(u_ref, N)

        U = self._solve_qp(H, g)
        u0 = U[:NU]

        info = {
            "solve_ms": (time.perf_counter() - t0) * 1e3,
            "cost": float(0.5 * U @ H @ U + g @ U),
            "violation": float((self.C_all @ U - self.d_all).max()),
            "solver": self.solver_name,
        }
        self.last = {"H": H, "g": g, "U": U, "Ad": Ad, "Bd": Bd,
                     "A_qp": A_qp, "B_qp": B_qp}
        return u0, U, info


    # ------------------------------------------------------------------
    # 보행 모드 (Step 5-6): 접촉 스케줄 + 시변 B + swing 발 W=0 등식 제약
    # ------------------------------------------------------------------
    def solve_gait(self, x0: np.ndarray, X_ref: np.ndarray, psi: float,
                   foot_pos_traj: np.ndarray, contact: np.ndarray,
                   u_ref_traj: np.ndarray):
        """보행용 풀기.

        X_ref        : (N,13) 스텝별 참조
        foot_pos_traj: (N,2,3) 지평 스텝별 발 위치 계획 (stance=현재값,
                       착지 후=Raibert 목표). swing 구간 값은 무시됨.
        contact      : (N,2) bool — 접촉 스케줄 (Gait.contact_table)
        u_ref_traj   : (N,12) — 스텝별 중력 ff (stance 발끼리 Mg/n 분배)

        swing 발: 해당 6변수 = 0 등식 제약 + B 열 0 (동역학에서도 제거).
        """
        t0 = time.perf_counter()
        N = self.N
        Ac, _ = continuous_AB(self.p, psi, foot_pos_traj[0] - x0[3:6])
        Ad, _ = discretize(Ac, np.zeros((NX, NU)), self.dt)

        # 스텝별 B_d. r_i: k=0 은 '현재 상태' CoM 기준 (Di Carlo / 사용자 MATLAB 의
        # i=1 = x_now 방식 — 검증 리포트 §2), k≥1 은 참조 CoM 기준 (미리 계산).
        Bd = []
        for k in range(N):
            com_k = x0[3:6] if k == 0 else X_ref[k, 3:6]
            r_feet = foot_pos_traj[k] - com_k
            _, Bc = continuous_AB(self.p, psi, r_feet)
            for i in range(N_FEET):
                if not contact[k, i]:
                    Bc[:, NU_PER_FOOT * i:NU_PER_FOOT * (i + 1)] = 0.0
            Bd.append(discretize(Ac, Bc, self.dt)[1])

        A_qp = np.zeros((NX * N, NX))
        B_qp = np.zeros((NX * N, NU * N))
        A_pow = np.eye(NX)
        for k in range(N):
            A_pow = Ad @ A_pow
            A_qp[NX * k:NX * (k + 1)] = A_pow
        for j in range(N):
            blk = Bd[j]
            for k in range(j, N):
                B_qp[NX * k:NX * (k + 1), NU * j:NU * (j + 1)] = blk
                blk = Ad @ blk

        H = 2.0 * (B_qp.T @ self.Q_bar @ B_qp + self.R_bar)
        H = 0.5 * (H + H.T)
        g = 2.0 * B_qp.T @ self.Q_bar @ (A_qp @ x0 - X_ref.reshape(-1)) \
            - 2.0 * self.R_bar @ u_ref_traj.reshape(-1)

        # 제약: stance 발 -> 10 부등식,  swing 발 -> W=0 등식 6개
        eq_idx = []                       # 0 으로 고정할 변수 인덱스
        Ci_rows, di = [], []
        for k in range(N):
            for i in range(N_FEET):
                base = NU * k + NU_PER_FOOT * i
                if contact[k, i]:
                    row = np.zeros((10, NU * N))
                    row[:, base:base + NU_PER_FOOT] = self.Cf
                    Ci_rows.append(row)
                    di.append(self.df)
                else:
                    eq_idx.extend(range(base, base + NU_PER_FOOT))
        C_in = np.vstack(Ci_rows)
        d_in = np.concatenate(di)

        U = self._solve_qp_eq(H, g, eq_idx, C_in, d_in)
        u0 = U[:NU]
        info = {
            "solve_ms": (time.perf_counter() - t0) * 1e3,
            "violation": float((C_in @ U - d_in).max()),
            "eq_violation": float(np.abs(U[eq_idx]).max()) if eq_idx else 0.0,
            "solver": self.solver_name,
        }
        self.last = {"U": U, "Bd": Bd, "contact": contact}
        return u0, U, info

    def _solve_qp_eq(self, H, g, eq_idx, C_in, d_in):
        """min ½uᵀHu+gᵀu  s.t. u[eq_idx]=0, C_in·u ≤ d_in."""
        n = H.shape[0]
        n_eq = len(eq_idx)
        E = np.zeros((n_eq, n))
        E[np.arange(n_eq), eq_idx] = 1.0
        try:
            # quadprog: Cᵀx ≥ b, 앞 meq 개는 등식.  C 는 (n, m)
            Cq = np.hstack([E.T, -C_in.T])
            bq = np.concatenate([np.zeros(n_eq), -d_in])
            u, *_ = quadprog.solve_qp(H, -g, Cq, bq, n_eq)
            self.solver_name = "quadprog"
            return u
        except Exception:               # noqa: BLE001
            import scipy.sparse as sp
            import osqp
            A = sp.vstack([sp.csc_matrix(E), sp.csc_matrix(C_in)]).tocsc()
            lo = np.concatenate([np.zeros(n_eq), -np.inf * np.ones(len(d_in))])
            hi = np.concatenate([np.zeros(n_eq), d_in])
            prob = osqp.OSQP()
            prob.setup(P=sp.csc_matrix(H), q=g, A=A, l=lo, u=hi,
                       verbose=False, eps_abs=1e-7, eps_rel=1e-7)
            res = prob.solve()
            self.solver_name = "osqp"
            return np.asarray(res.x)


# ---------------------------------------------------------------------------
# 토크 변환 (층1 마지막) — 이미 검증된 파이프라인, 6D wrench 로 확장
# ---------------------------------------------------------------------------
def wrench_to_tau(m, d, adof: np.ndarray, wrenches: np.ndarray) -> np.ndarray:
    """τ = qfrc_bias − Σ J_iᵀ W_i.   wrenches: (2,6) 발별 [F(3), m(3)] world.

    J_i 는 6×nv (jacp 위, jacr 아래). mj_forward 가 불린 상태를 가정.
    """
    tau_c = np.zeros(m.nv)
    for i, s in enumerate(FOOT_SITES):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        tau_c += jacp.T @ wrenches[i, :3] + jacr.T @ wrenches[i, 3:]
    return d.qfrc_bias[adof] - tau_c[adof]


def actuated_dofs(m) -> np.ndarray:
    return np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])

```


---

# FILE: src/gait.py

```python
"""Gait 스케줄 + Raibert 착지점 + swing 궤적/임피던스 (명세 5, 6절).

역할 구분:
  - Gait        : 시간 -> 발별 stance/swing (고정 타이밍, Di Carlo 방식)
  - raibert_target : 다음 착지점  p = hip + v·T_st/2 + k(v − v_cmd)
  - SwingController: 든 발을 착지점까지 아치 궤적으로 옮기는 임피던스 (MPC 무관)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco

from mpc_srb import FOOT_SITES, rz


# ---------------------------------------------------------------------------
# 접촉 스케줄
# ---------------------------------------------------------------------------
@dataclass
class Gait:
    """고정 타이밍 보행 스케줄. 발 i 의 위상 φ_i = ((t−t_start)/T + offset_i) mod 1,
    φ < stance_frac 이면 stance, 아니면 swing.  offsets=(0, 0.5) → 좌우 교대,
    φ=0 근처는 양발 지지(double support)가 자연히 생긴다 (stance_frac > 0.5)."""
    T: float = 0.8                 # 보행 주기 [s]
    stance_frac: float = 0.75      # stance 비율 (swing = 0.25 T = 0.2 s)
    offsets: tuple = (0.0, 0.5)    # (왼발, 오른발) 위상 오프셋
    t_start: float = 0.5           # 이 시각 전에는 둘 다 stance (서 있기)

    def phase(self, t: float, i: int) -> float:
        if t < self.t_start:
            return 0.0
        return ((t - self.t_start) / self.T + self.offsets[i]) % 1.0

    def in_stance(self, t: float, i: int) -> bool:
        return self.phase(t, i) < self.stance_frac

    def swing_phase(self, t: float, i: int) -> float:
        """swing 진행도 s ∈ [0,1). stance 면 0."""
        ph = self.phase(t, i)
        if ph < self.stance_frac:
            return 0.0
        return (ph - self.stance_frac) / (1.0 - self.stance_frac)

    @property
    def T_swing(self) -> float:
        return (1.0 - self.stance_frac) * self.T

    @property
    def T_stance(self) -> float:
        return self.stance_frac * self.T

    def contact_table(self, t: float, dt: float, N: int) -> np.ndarray:
        """(N,2) bool — 지평 스텝 k(구간 중점 기준)의 접촉 여부. MPC 용."""
        tbl = np.zeros((N, 2), dtype=bool)
        for k in range(N):
            tk = t + (k + 0.5) * dt
            for i in range(2):
                tbl[k, i] = self.in_stance(tk, i)
        return tbl

    def time_to_touchdown(self, t: float, i: int) -> float:
        """발 i 가 다음에 착지하는 시각까지 남은 시간 (지금 stance 면 0)."""
        if self.in_stance(t, i):
            return 0.0
        return (1.0 - self.phase(t, i)) * self.T


# ---------------------------------------------------------------------------
# Raibert 착지점
# ---------------------------------------------------------------------------
def raibert_target(p_com, v_com, yaw, side_offset, v_cmd, T_stance, z_com,
                   anchor_xy=None, k_anchor=0.15, max_step=0.30, min_y_sep=0.06,
                   z_ground=0.0331):
    """착지점 (world, capture point 기반).

        p = CoM + R·offset + v_cmd·T_st/2 + (v − v_cmd)/ω₀ + k_a(anchor − CoM)
        ω₀ = √(g/z_com)  (선형 역진자 고유진동수, 여기선 ≈3.8 rad/s → 1/ω₀ ≈ 0.26 s)

    역사: v·T_st/2 + k·v (Raibert 원형, 속도계수 0.48) 는 이 로봇/주기에서
    과잉 스텝 → 진동 성장으로 9.5 s 만에 넘어짐 (Step5 디버깅 3차).
    캡처 계수 1/ω₀=0.26 으로 교체. anchor 는 발디딤 표류 방지용 약한 위치 되당김.
    """
    w0 = np.sqrt(9.81 / max(z_com - z_ground, 0.3))   # LIP 높이 = CoM−지면 (리포트 §3)
    R = rz(yaw)[:2, :2]
    hip = p_com[:2] + R @ side_offset[:2]
    v = v_com[:2]
    # 주의: 이 게인은 민감하다. 1/ω₀ 그대로가 최선이었고 +0.06(9s)·+0.22(9.5s)
    # 모두 악화, β_sway·Q_vy 동시 변경도 악화 (Step6 튜닝 기록 — MPC_NOTES 참고).
    p = hip + v_cmd[:2] * (T_stance / 2.0) + (v - v_cmd[:2]) / w0
    if anchor_xy is not None:
        # 부호 주의: +k·(CoM−ref) = 밀린 쪽으로 '더' 내딛기 (위치 복원, 사용자
        # convex_mpc.m 의 k_y=0.3 과 같은 부호). 반대 부호(중앙으로 당김)는
        # CoM 이 발 바깥에 놓여 위치적으로 불안정 — 23 s 느린 발산의 용의자였다.
        p += k_anchor * (p_com[:2] - np.asarray(anchor_xy, float))

    # 스텝 길이 제한
    d = p - hip
    n = np.linalg.norm(d)
    if n > max_step:
        p = hip + d / n * max_step

    # 교차 방지: 몸 좌표 y 로 변환해 부호 유지
    p_body = R.T @ (p - p_com[:2])
    sign = np.sign(side_offset[1])
    if sign * p_body[1] < min_y_sep:
        p_body[1] = sign * min_y_sep
        p = p_com[:2] + R @ p_body
    return np.array([p[0], p[1], z_ground])


# ---------------------------------------------------------------------------
# Swing 궤적 + 임피던스
# ---------------------------------------------------------------------------
@dataclass
class SwingController:
    """swing 발 하나의 궤적 생성 + 임피던스 토크.

    궤적 (명세 6절 간소화 옵션): xy = smoothstep 보간, z = 시작→끝 보간 + h·sin(πs).
    임피던스: F = Kp(p_des−p) + Kd(v_des−v),  발 수평 유지용 자세 스프링/댐퍼 추가.
    """
    h_swing: float = 0.05
    # kp=350 (1.3 Hz) 은 0.2 s swing 에 너무 무름 — 발이 궤적을 못 쫓아 착지가
    # 한 박자 늦고, '스케줄상 stance 인데 공중' 갭이 매 스텝 roll 킥을 만들어
    # 발산했다 (Step5 디버깅 1차). 다리 ~5 kg 기준 3.5 Hz 로 상향.
    kp: float = 2500.0
    kd: float = 80.0
    kp_ori: float = 60.0
    kd_ori: float = 5.0
    p_liftoff: np.ndarray = field(default_factory=lambda: np.zeros(3))
    active: bool = False

    def start(self, p_foot_now: np.ndarray):
        self.p_liftoff = p_foot_now.copy()
        self.active = True

    def stop(self):
        self.active = False

    def target(self, s: float, p_land: np.ndarray, T_swing: float):
        """swing 진행도 s ∈ [0,1] 에서 (p_des, v_des)."""
        s = float(np.clip(s, 0.0, 1.0))
        p0, p1 = self.p_liftoff, p_land
        sm = s * s * (3.0 - 2.0 * s)             # smoothstep
        dsm = 6.0 * s * (1.0 - s)                # d(sm)/ds
        p_des = p0 + (p1 - p0) * sm
        v_des = (p1 - p0) * dsm / T_swing
        p_des = p_des.copy()
        p_des[2] += self.h_swing * np.sin(np.pi * s)
        v_des = v_des.copy()
        v_des[2] += self.h_swing * np.pi * np.cos(np.pi * s) / T_swing
        return p_des, v_des

    def wrench(self, m, d, site_name: str, s: float, p_land: np.ndarray,
               T_swing: float):
        """이 발에 가할 (F(3), moment(3), jacp, jacr). 토크는 +Jᵀ 로 인가."""
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)

        p_foot = d.site_xpos[sid]
        v_foot = jacp @ d.qvel
        p_des, v_des = self.target(s, p_land, T_swing)
        F = self.kp * (p_des - p_foot) + self.kd * (v_des - v_foot)
        # 반작용 제한: 다리를 너무 세게 던지면 상체가 반대로 돈다 (Step5 디버깅 2차).
        # 단 전진 보행은 swing 이동거리가 길어 ~250 N 이 필요 — 80 N 은 발을 묶어서
        # 걸려 넘어졌다 (Step6 디버깅 1차). 상체 PD 가 반작용을 받아주므로 200 으로.
        nF = np.linalg.norm(F)
        if nF > 200.0:
            F *= 200.0 / nF

        # 발바닥 수평 유지: 발 z축을 world z 로 돌리는 스프링 + 각속도 댐핑
        R_f = d.site_xmat[sid].reshape(3, 3)
        z_f = R_f[:, 2]
        axis = np.cross(z_f, np.array([0.0, 0.0, 1.0]))   # 회전축 (부호 포함)
        w_foot = jacr @ d.qvel
        mom = self.kp_ori * axis - self.kd_ori * w_foot
        return F, mom, jacp, jacr

```


---

# FILE: src/09_walk.py

```python
"""Step 5-6 — 보행: gait 스케줄 + Raibert + swing 임피던스 + gait MPC.

구조 (매 사이클):
  [100 Hz] 접촉 스케줄 + 참조 + 발 위치 계획 조립 → solve_gait → stance wrench
           swing 발이면 Raibert 착지점 갱신
  [500 Hz] τ = qfrc_bias − Σ_stance JᵀW + Σ_swing Jᵀ(임피던스 F, m)

사용:
  .venv/Scripts/python.exe MPC/src/09_walk.py               # 제자리 스텝 (Step 5)
  .venv/Scripts/python.exe MPC/src/09_walk.py --vx 0.3      # 전진 (Step 6)
  .venv/Scripts/python.exe MPC/src/09_walk.py --view --vx 0.3
"""
from __future__ import annotations

import sys

import numpy as np
import mujoco

import g1_model
import mpc_srb
from mpc_qp import WrenchMPC, actuated_dofs
from mpc_log import MPCLog
from gait import Gait, SwingController, raibert_target

np.set_printoptions(precision=3, suppress=True, linewidth=160)

HORIZON = 16
DT_MPC = 0.05          # 지평 N*dt = 0.8 s = 보행주기 1개 (꽉 채우기).
                       # 0.64 s(80%)보다 0.2 m/s 생존 9.2->13.4 s 개선.
                       # 매트랩 quadruped 도 stride 100% 커버 + dt 0.05 였음.
DECIM = 5              # MPC 100 Hz
RAMP_T0, RAMP_T1 = 2.0, 3.0   # v_cmd 램프 구간


class WalkController:
    def __init__(self, m, d, vx_cmd=0.0):
        self.m = m
        self.adof = actuated_dofs(m)
        self.params = mpc_srb.make_params(m, d)
        # 보행 전용 가중치: 횡 속도 감쇠 강화 (서 있기 기본값은 mpc_qp.Q_DEFAULT)
        from mpc_qp import Q_DEFAULT
        self.mpc = WrenchMPC(self.params, horizon=HORIZON, dt=DT_MPC,
                     psi0=mpc_srb.quat_to_euler_zyx(d.qpos[3:7])[2])
        self.gait = Gait(T=0.8, stance_frac=0.75, t_start=0.5)
        self.vx_cmd = vx_cmd

        x0 = mpc_srb.get_state(m, d, I_body=self.params.I_body)
        self.yaw0 = x0[2]
        self.com0 = x0[3:6].copy()
        feet = mpc_srb.get_foot_positions(m, d)
        R2 = mpc_srb.rz(self.yaw0)[:2, :2]
        self.side_offset = [np.append(R2.T @ (feet[i, :2] - x0[3:5]), 0.0)
                            for i in range(2)]          # CoM 기준 발 중립 오프셋
        self.z_ground = feet[0, 2]
        self.sw = [SwingController(), SwingController()]
        self.was_stance = [True, True]
        self.p_land = [feet[0].copy(), feet[1].copy()]
        self.wr = np.zeros((2, 6))
        self.last_info = {}
        # 상체(허리+팔) 자세 PD: 중력보상만 받으면 둥둥 떠서 다리 반작용에
        # 휘둘린다. 고정해야 로봇이 SRB 가정에 가까워짐 (Step5 디버깅 2차).
        self.q_ref = d.qpos[7:].copy()
        self.upper_act = [a for a in range(m.nu)
                          if m.jnt_dofadr[m.actuator_trnid[a, 0]] >= 18]
        self.kp_up, self.kd_up = 60.0, 4.0

    def v_cmd(self, t):
        ramp = np.clip((t - RAMP_T0) / (RAMP_T1 - RAMP_T0), 0.0, 1.0)
        return np.array([self.vx_cmd * ramp, 0.0, 0.0])

    # -------------------------------------------------- 100 Hz: MPC
    def update_mpc(self, d, t):
        m = self.m
        x0 = mpc_srb.get_state(m, d, I_body=self.params.I_body)
        feet = mpc_srb.get_foot_positions(m, d)
        vc = self.v_cmd(t)

        # swing 시작/종료 관리 + Raibert 착지점 갱신
        for i in range(2):
            st = self.gait.in_stance(t, i)
            if self.was_stance[i] and not st:
                self.sw[i].start(feet[i])
            if not st:
                # capture point 배치. y 만 참조에 약하게 앵커 (표류 방지)
                self.p_land[i] = raibert_target(
                    x0[3:6], x0[9:12], x0[2], self.side_offset[i], vc,
                    self.gait.T_stance, z_com=x0[5], z_ground=self.z_ground,
                    anchor_xy=[x0[3], self.com0[1]])
            self.was_stance[i] = st

        # 지평 재료 조립
        N = HORIZON
        contact = self.gait.contact_table(t, DT_MPC, N)
        # stage-0 접촉은 '지금' 기준 (중점 t+25ms 는 착지 25ms 전부터 그 발에
        # W>0 을 명령해 torque() 의 swing 처리와 어긋남 — 검증 리포트 §5-(d))
        for i in range(2):
            contact[0, i] = self.gait.in_stance(t, i)
        X_ref = np.zeros((N, mpc_srb.NX))
        foot_traj = np.zeros((N, 2, 3))
        u_ref = np.zeros((N, mpc_srb.NU))
        Mg = self.params.mass * mpc_srb.GRAV
        for i in range(2):
            became_swing = not self.was_stance[i]
            for k in range(N):
                if not contact[k, i]:
                    became_swing = True
                foot_traj[k, i] = self.p_land[i] if became_swing else feet[i]
        BETA_SWAY = 0.5      # 체중이동 참조 강도 (0=항상 중앙, 1=stance 발 위)
        for k in range(N):
            X_ref[k, 2] = self.yaw0
            if abs(self.vx_cmd) > 1e-9:
                X_ref[k, 3] = x0[3] + vc[0] * (k + 1) * DT_MPC   # 속도 추종 모드
            else:
                X_ref[k, 3] = self.com0[0]                       # 제자리 모드
            # sway 참조: 지평 k 의 지지 중심 쪽으로 CoM y 를 미리 이동시킨다.
            # 항상 중앙(0)이면 MPC 가 체중이동 없이 발목으로만 버티다 진동이
            # 누적된다 (Step5 디버깅 4차). preview 가 있어야 가능한 정석 방식.
            st = [i for i in range(2) if contact[k, i]]
            y_sup = np.mean([foot_traj[k, i, 1] for i in st]) if st else self.com0[1]
            X_ref[k, 4] = self.com0[1] + BETA_SWAY * (y_sup - self.com0[1])
            X_ref[k, 5] = self.com0[2]
        # sway 참조를 속도 제한 램프로 스무딩 (스텝 함수 그대로면 전환 순간
        # 참조 점프를 MPC 가 쫓아 옆으로 차버린다 — Step5 디버깅 5차)
        y_prev = x0[4]
        for k in range(N):
            dy = np.clip(X_ref[k, 4] - y_prev, -0.4 * DT_MPC, 0.4 * DT_MPC)
            X_ref[k, 4] = y_prev = y_prev + dy
        for k in range(N):
            X_ref[k, 9:12] = vc
            X_ref[k, 12] = mpc_srb.GRAV
            st_k = [i for i in range(2) if contact[k, i]]
            for i in st_k:
                u_ref[k, 6 * i + 2] = Mg / max(len(st_k), 1)

        u0, _, info = self.mpc.solve_gait(x0, X_ref, self.yaw0,
                                          foot_traj, contact, u_ref)
        self.wr = u0.reshape(2, 6)
        self.last_info = info
        return x0, X_ref[0], contact[0]

    # -------------------------------------------------- 500 Hz: 토크
    def torque(self, d, t):
        m = self.m
        tau_c = np.zeros(m.nv)
        for i, s_name in enumerate(mpc_srb.FOOT_SITES):
            if self.gait.in_stance(t, i):
                sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s_name)
                jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
                mujoco.mj_jacSite(m, d, jacp, jacr, sid)
                tau_c -= jacp.T @ self.wr[i, :3] + jacr.T @ self.wr[i, 3:]
            else:
                s = self.gait.swing_phase(t, i)
                F, mom, jacp, jacr = self.sw[i].wrench(
                    m, d, s_name, s, self.p_land[i], self.gait.T_swing)
                tau_c += jacp.T @ F + jacr.T @ mom
        tau = d.qfrc_bias[self.adof] + tau_c[self.adof]
        for a in self.upper_act:
            jid = self.m.actuator_trnid[a, 0]
            qa = self.m.jnt_qposadr[jid]
            dofa = self.m.jnt_dofadr[jid]
            tau[a] += self.kp_up * (self.q_ref[qa - 7] - d.qpos[qa])                       - self.kd_up * d.qvel[dofa]
        return tau


# ===========================================================================
LEG_BODY_KEYS = ("hip_pitch_link", "hip_roll_link", "hip_yaw_link",
                 "knee_link", "ankle_pitch_link", "ankle_roll_link")


def scale_leg_mass(m, d, scale):
    """다리 링크 질량·관성 스케일링 — massless-leg 가설 격리 실험용
    (검증 리포트 §5-1: 벽이 사라지면 swing 반작용 확정)."""
    import mujoco as _mj
    changed = []
    for b in range(m.nbody):
        name = _mj.mj_id2name(m, _mj.mjtObj.mjOBJ_BODY, b) or ""
        if any(k in name for k in LEG_BODY_KEYS):
            m.body_mass[b] *= scale
            m.body_inertia[b] *= scale
            changed.append(name)
    _mj.mj_setConst(m, d)
    return changed


def headless(vx=0.0, seconds=12.0, legmass=1.0):
    m, d = g1_model.load_torque()
    g1_model.set_crouch(m, d)
    if abs(legmass - 1.0) > 1e-9:
        n = len(scale_leg_mass(m, d, legmass))
        import mujoco as _mj
        print(f"  [실험] 다리 링크 {n}개 질량 x{legmass} -> 총질량 "
              f"{_mj.mj_getTotalmass(m):.2f} kg")
        g1_model.set_crouch(m, d)   # 질량 변경 후 자세 재설정
    ctl = WalkController(m, d, vx_cmd=vx)
    tag = "inplace" if abs(vx) < 1e-9 else "vx" + f"{vx:g}".replace(".", "p")
    log = MPCLog(f"logs/walk_{tag}",
                 note=f"Step5/6: gait MPC N={HORIZON}, T={ctl.gait.T}, vx_cmd={vx}")

    z0 = d.qpos[2]
    dt = m.opt.timestep
    fell = None
    swing_apex = [0.0, 0.0]
    n_swings = [0, 0]
    prev_st = [True, True]
    vx_hist, py_hist = [], []

    print(f"  t[s]  pelvis_z   com_x    com_y   vx     접촉  |my|max  solve_ms")
    next_rep = 0.0
    x0 = None
    for k in range(int(seconds / dt)):
        t = k * dt
        mujoco.mj_forward(m, d)
        if k % DECIM == 0:
            x0, xr0, c0 = ctl.update_mpc(d, t)
            log.add(t=t, x=x0, x_ref=xr0, u=ctl.wr.reshape(-1),
                    tau=ctl.torque(d, t), contact=c0.astype(float),
                    foot_z=[mpc_srb.get_foot_positions(m, d)[i, 2] for i in range(2)],
                    solve_ms=ctl.last_info["solve_ms"],
                    violation=ctl.last_info["violation"], ncon=d.ncon)
            if t > RAMP_T1 + 1.0:
                vx_hist.append(x0[9]); py_hist.append(x0[4])
        d.ctrl[:] = ctl.torque(d, t)
        mujoco.mj_step(m, d)

        feet_z = mpc_srb.get_foot_positions(m, d)[:, 2]
        for i in range(2):
            st = ctl.gait.in_stance(t, i)
            if not st:
                swing_apex[i] = max(swing_apex[i], feet_z[i] - ctl.z_ground)
            if (not prev_st[i]) and st:
                n_swings[i] += 1
            prev_st[i] = st

        if t >= next_rep - 1e-9:
            next_rep += 1.0
            cc = "".join("S" if ctl.gait.in_stance(t, i) else "w" for i in range(2))
            print(f"  {t:5.1f}  {d.qpos[2]:8.4f}  {d.subtree_com[0][0]:7.3f}"
                  f"  {d.subtree_com[0][1]:7.3f}  {x0[9]:5.2f}   {cc}"
                  f"   {np.abs(ctl.wr[:,4]).max():6.2f}  {ctl.last_info['solve_ms']:7.1f}")
        if fell is None and d.qpos[2] < z0 - 0.25:
            fell = t
            break

    print()
    ok = fell is None
    if fell:
        print(f"  쓰러짐 @ {fell:.2f}s")
    else:
        print(f"  {seconds:.0f}초 생존 ✓  pelvis z {z0:.4f} -> {d.qpos[2]:.4f}")
    print(f"  완료한 swing 횟수: 왼 {n_swings[0]}, 오른 {n_swings[1]}"
          f"   swing 최고높이: 왼 {swing_apex[0]*100:.1f} cm, 오른 {swing_apex[1]*100:.1f} cm")
    if vx_hist:
        print(f"  정착 후 평균 vx = {np.mean(vx_hist):.3f} m/s (명령 {vx})"
              f"   CoM y 진폭 = ±{(np.max(py_hist)-np.min(py_hist))/2*100:.1f} cm")
    sms = np.stack(log.rows["solve_ms"])
    print(f"  QP: 평균 {sms.mean():.1f} ms, 최대 {sms.max():.1f} ms")
    log.save()
    return ok


def view(vx=0.0):
    import mujoco.viewer
    m, d = g1_model.load_torque()
    g1_model.set_crouch(m, d)
    ctl = WalkController(m, d, vx_cmd=vx)
    print(f"뷰어: gait MPC (vx_cmd={vx}). 창을 닫으면 종료.")
    k = 0
    t0 = 0.0
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t = t0 + k * m.opt.timestep
            mujoco.mj_forward(m, d)
            if k % DECIM == 0:
                ctl.update_mpc(d, t)
            d.ctrl[:] = ctl.torque(d, t)
            mujoco.mj_step(m, d)
            v.sync()
            k += 1


if __name__ == "__main__":
    vx = 0.0
    if "--vx" in sys.argv:
        vx = float(sys.argv[sys.argv.index("--vx") + 1])
    secs = 12.0
    if "--seconds" in sys.argv:
        secs = float(sys.argv[sys.argv.index("--seconds") + 1])
    legm = 1.0
    if "--legmass" in sys.argv:
        legm = float(sys.argv[sys.argv.index("--legmass") + 1])
    if "--view" in sys.argv:
        view(vx)
    else:
        ok = headless(vx=vx, seconds=secs, legmass=legm)
        step = "STEP 5 (제자리 스텝)" if abs(vx) < 1e-9 else f"STEP 6 (전진 {vx} m/s)"
        print()
        print(step + (" 통과 ✓" if ok else " 실패"))

```


---

# FILE: src/10_constraint_check.py

```python
"""CoP 제약 h 결합항 검증 (검증 리포트 §1 의 요구사항).

방법: 발바닥 접촉점 4개에 무작위 점힘(마찰콘 안, 수평력 포함!)을 뿌리고
  (a) 점힘에서 직접 계산한 지면 CoP 가 발 안인가   <- 물리적 정답
  (b) 점힘을 발목 site wrench 로 합쳐 제약 행 7-10 을 통과하는가
둘이 모든 샘플에서 일치해야 한다. Fx,Fy ≠ 0 이므로 h 항이 실제로 시험된다
(서 있기 Step 1 은 Fx≈0 이라 이 항을 검증할 수 없었다).

추가로 h=0 인 옛 제약이 얼마나 오판하는지 센다.
yaw=0.7 rad 회전 케이스(world frame 제약)도 검증.

사용: .venv/Scripts/python.exe MPC/src/10_constraint_check.py
"""
from __future__ import annotations

import numpy as np

import g1_model
import mpc_srb
from mpc_qp import foot_constraints

rng = np.random.default_rng(0)


def sample_case(p, psi=0.0):
    """접촉점 4개 무작위 점힘 (발 frame) -> (world wrench W(6), 물리 정답 bool)."""
    pts = np.array([[p.l_t, p.w, -p.h_sole], [p.l_t, -p.w, -p.h_sole],
                    [-p.l_h, p.w, -p.h_sole], [-p.l_h, -p.w, -p.h_sole]])
    F = np.zeros(3)
    m_site = np.zeros(3)
    # 음수 fz·마찰 초과도 허용해 '물리적으로 불가능한' wrench 도 생성한다
    # (제약이 reject 하는 방향도 시험해야 완전한 검증)
    fz_pts = rng.uniform(-40.0, 120.0, 4)
    for j in range(4):
        s_t = rng.uniform(-1.4, 1.4, 2)
        f = np.array([s_t[0] * p.mu * abs(fz_pts[j]),
                      s_t[1] * p.mu * abs(fz_pts[j]),
                      fz_pts[j]])
        F += f
        m_site += np.cross(pts[j], f)

    Fz = F[2]
    if Fz < p.fz_min or Fz > p.fz_max:
        return None
    # 물리 정답: 지면 CoP (점힘의 fz 가중 평균) + 합력 마찰
    cop_x = (pts[:, 0] * fz_pts).sum() / Fz
    cop_y = (pts[:, 1] * fz_pts).sum() / Fz
    ok_phys = (-p.l_h - 1e-9 <= cop_x <= p.l_t + 1e-9
               and abs(cop_y) <= p.w + 1e-9
               and abs(F[0]) <= p.mu * Fz + 1e-9
               and abs(F[1]) <= p.mu * Fz + 1e-9)

    W = np.concatenate([F, m_site])
    if abs(psi) > 1e-12:
        R = mpc_srb.rz(psi)
        W = np.concatenate([R @ W[:3], R @ W[3:]])   # world 로 회전
    return W, ok_phys


def run(p, psi, n=20000):
    C, dvec = foot_constraints(p, psi=psi)
    C_old, _ = foot_constraints(
        mpc_srb.SRBParams(**{**p.__dict__, "h_sole": 0.0}), psi=psi)
    agree = agree_old = n_valid = n_in = 0
    for _ in range(n):
        case = sample_case(p, psi)
        if case is None:
            continue
        W, ok_phys = case
        n_valid += 1
        n_in += ok_phys
        ok_new = bool((C @ W <= dvec + 1e-7).all())
        ok_old = bool((C_old @ W <= dvec + 1e-7).all())
        agree += (ok_new == ok_phys)
        agree_old += (ok_old == ok_phys)
    return n_valid, n_in, agree, agree_old


def main():
    m, d = g1_model.load_torque()
    g1_model.set_crouch(m, d)
    p = mpc_srb.make_params(m, d)
    print(f"h_sole = {p.h_sole:.4f} m (모델 추출: 구 중심 0.03 + 반지름 0.005)")
    print(f"l_t={p.l_t}, l_h={p.l_h}, w={p.w}, mu={p.mu}")
    print()

    ok_all = True
    for psi in (0.0, 0.7):
        nv, ni, ag, ag_old = run(p, psi)
        print(f"yaw = {psi:.1f} rad — 유효 샘플 {nv} (물리적 통과 {ni})")
        print(f"  새 제약(h 항 포함)  일치율: {ag}/{nv}  ({100*ag/nv:.2f} %)")
        print(f"  옛 제약(h=0)      일치율: {ag_old}/{nv}  ({100*ag_old/nv:.2f} %)"
              f"   <- 오판 {nv-ag_old}건")
        ok_all &= (ag == nv)
    print()
    print("검증 " + ("통과 ✓ — 새 제약은 물리 정답과 완전 일치" if ok_all else "실패"))


if __name__ == "__main__":
    main()

```