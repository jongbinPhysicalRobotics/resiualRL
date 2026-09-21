# 코드 읽기 지도 — 문서를 읽은 다음 코드를 볼 때

> **갱신 2026-09-21.** 9/15~9/18 Q&A 를 읽은 뒤 코드를 보는 순서로 재편했다.
> 각 함수 옆에 **"어느 Q&A 가 이걸 설명하는가"** 를 붙여서, 읽은 내용이
> 코드 어디에 박혀 있는지 바로 찾을 수 있게 했다.
>
> 전체 흐름은 [Q&A/SUMMARY_0915-0918.md](../Q&A/SUMMARY_0915-0918.md) 를 먼저.

---

## 0. 한 장 요약 — 데이터가 흐르는 길

```
[500 Hz 물리 루프]  09_walk.py : headless() / view()
        │
        ├─ 매 5틱(=100 Hz) ─► WalkController.update_mpc()
        │                      ① mpc_srb.get_state()      상태 x(13)
        │                      ② gait.Gait                누가 딛고 있나
        │                      ③ gait.raibert_target()    어디에 딛나  ★보폭
        │                      ④ 참조 X_ref 조립          어디로 갈 건가
        │                      └► mpc_qp.solve_gait()     QP → wrench W(12)
        │
        └─ 매 틱(500 Hz) ───► WalkController.torque()
                               stance: τ −= JᵀW
                               swing : τ += Jᵀ(SwingController.wrench)  ★임피던스
                               swingid: τ += M q̈_swing
                               전체  : τ += qfrc_bias
                                            │
                                            └──► d.ctrl[:] = τ → mj_step
```

**파일 23개 중 컨트롤러는 5개**다. 나머지는 검증 스크립트·분석 도구·유틸.

---

## 1. 핵심 5개 — 이 순서로

| 순서 | 파일 | 줄 | 무엇 |
|---|---|---|---|
| **①** | [src/mpc_srb.py](src/mpc_srb.py) | 190 | **모델** — 상태 x(13), 연속 A·B, 이산화 |
| **②** | [src/gait.py](src/gait.py) | 277 | **계획** — 접촉 스케줄, 착지점, 스윙 궤적·임피던스 |
| **③** | [src/mpc_qp.py](src/mpc_qp.py) | 345 | **최적화** — 제약 18행, condensed QP |
| **④** | [src/09_walk.py](src/09_walk.py) | 661 | **조립** — 셋을 엮어 토크로 |
| ⑤ | [src/g1_model.py](src/g1_model.py) | 114 | 모델 로딩, 토크 액추에이터 변환, `set_crouch` |

---

## 2. 함수별 — 읽은 Q&A 와 짝지어서

### ① `mpc_srb.py` — 가장 짧고 개념이 선명하다. 여기부터.

| 위치 | 무엇 | 관련 Q&A |
|---|---|---|
| `make_params()` L71 | 질량·관성·발 형상을 **모델에서 추출** (하드코딩 없음) | — |
| **`get_state()` L108** | MuJoCo → x(13). **ω 를 전신 각운동량으로 뽑는다** | **[9/16 Q7·Q8](../Q&A/2026-09-16.md)** — 왜 이 조합인지. `I_body=None` 분기가 ablation 장비 |
| `continuous_AB()` L155 | **이 4줄이 SRB 동역학 전부** | 요약 §8 |
| `discretize()` L176 | A 가 멱영(A³=0)이라 ZOH 가 **정확** (근사 아님) | — |

> 읽는 요령: L113-117 주석이 **Θ(pelvis)와 ω(전신 L)가 '같은 강체'가 아니다**라는
> 구조적 한계를 이미 적어두고 있다. 9/16 Q8 이 그걸 숫자로 잰 것.

### ② `gait.py` — "언제 / 어디에 / 어떻게 딛나"

| 위치 | 무엇 | 관련 Q&A |
|---|---|---|
| `Gait` L22 | 시간 → stance/swing. `stance_frac` 이 여기 | 요약 §4 (sf 0.75→0.57) |
| `contact_table()` L54 | 지평 N 스텝의 접촉 여부 → MPC 로 | — |
| **`raibert_target()` L73** | **착지점 공식. ★보폭 문제의 현장** | **[9/15 Q4·Q8](../Q&A/2026-09-15.md)**, 요약 §5 |
| ↳ `hip = p_com + R @ side_offset` L87 | **공칭 오프셋 11.8 cm — 27 cm 의 87 %** | 9/18. **여기를 `w(v)/2` 로 바꾸는 게 1순위** |
| ↳ `cap = (v − v_cmd)·k_cap` L104 | capture 항. `cap_y_max` 가 `--capy` | 9/15 Q8 §6 |
| ↳ `lip_exact` L98 | LIP 정확해 — 유도는 맞는데 **실험은 나빠짐** | 요약 §11 ① |
| ↳ `min_y_sep` L127 | 교차 방지 하한 6 cm | — |
| **`SwingController.target()` L178** | **스윙 z 궤적. `soft_land` 분기가 여기** | **[9/16 Q3](../Q&A/2026-09-16.md)** (sin vs 2단 vs 베지어) |
| `target_acc()` L213 | `--swingid` 가 쓰는 계획 가속도 | **[9/18 Q1·Q2](../Q&A/2026-09-18.md)** |
| **`wrench()` L234** | 작업공간 임피던스 → (F, moment) | 요약 §3. **토크 초과 100 % 가 여기서** (9/18 Q2) |

### ③ `mpc_qp.py` — "무엇이 물리적으로 가능한가"

| 위치 | 무엇 | 관련 Q&A |
|---|---|---|
| `Q_DEFAULT` / `R_DEFAULT` L20 | 가중치. 주석에 튜닝 이력 | — |
| **`foot_constraints()` L30** | **제약 18행 전부** | **[9/18 Q4](../Q&A/2026-09-18.md)** — 18행 전수 해설 |
| ↳ 마찰 4행 L49-52 | 원뿔 → 사각뿔 근사 (**대각 41 % 낙관, 미해결**) | 9/18 Q4 (a) |
| ↳ CoP 4행 L55-58 | `h·Fx` 결합항이 왜 있는지 주석에 | 9/18 Q4 (c) |
| ↳ **Caron mz 8행 L60-81** | 절댓값 2개 × 부호 2 × 상하한 2 | **[9/18 Q3](../Q&A/2026-09-18.md)** — 논문 원문 대조 |
| `solve_gait()` L218 | A_qp/B_qp 조립 → H, g → quadprog | — |
| `_solve_qp_eq()` L299 | swing 발 W=0 등식 제약 | — |

### ④ `09_walk.py` — 제일 길지만 구조는 단순

| 위치 | 무엇 | 관련 Q&A |
|---|---|---|
| `__init__` L40 | **플래그가 다 모여 있음** — 실험 스위치 목록 | §5 표 참고 |
| ↳ `side_offset` L71 | **초기 자세에서 스냅샷, 이후 불변** ← 보폭의 출처 | 9/18, 요약 §5 |
| `update_mpc()` L160 | (100 Hz) 착지점 갱신 → 참조 조립 → QP | — |
| ↳ yaw 언랩 L166 | ±180° 랩 버그 수정 | MPC_NOTES 19절 |
| ↳ 스윙 yaw 목표 | `gate_sy` = `--swingyaw` | **[9/16 Q2](../Q&A/2026-09-16.md)**, 요약 §7 |
| **`torque()` L299** | **Jᵀ 사상.** stance/swing/swingid 세 채널 | **[9/15 Q1](../Q&A/2026-09-15.md)** 이 이 함수 해설 |
| ↳ `lam_swing` 블록 L335 | Λ = (J M⁻¹ Jᵀ)⁻¹ 계산 | 요약 §3 |
| ↳ `swing_id` 블록 L361 | `τ += M q̈_swing` | **9/18 Q1** |
| `headless()` L405 | 러너 + 로깅 | 9/21 Q3 (로그 키 추가) |
| `_draw_swing()` L529 | 뷰어 스윙궤적 오버레이 | 9/21 Q3·Q4 |

---

## 3. 코드보다 먼저 읽을 것

| 문서 | 왜 |
|---|---|
| **[Q&A/SUMMARY_0915-0918.md](../Q&A/SUMMARY_0915-0918.md)** | 전체 흐름. **이미 읽었다면 통과** |
| [Q&A/2026-09-15.md](../Q&A/2026-09-15.md) **Q1** | `τ = qfrc_bias − JᵀW` 유도. **이거 없이 `torque()` 보면 암호다** |
| [Q&A/2026-09-18.md](../Q&A/2026-09-18.md) **Q4** | 제약 18행. `foot_constraints` 읽기 전에 |
| [PHYSICS_MAP.md](PHYSICS_MAP.md) | 어떤 물리를 어떻게 근사했나 (코드의 '왜') |

---

## 4. 손으로 돌려볼 것 — 개념 확인용

역사적 순서 = 난이도 순서. 각각 독립 실행.

| 파일 | 무엇을 확인하나 |
|---|---|
| `01_view.py` / `02_inspect.py` | 모델 띄우기 / 관절·질량 덤프 |
| **`03_grf_to_tau.py`** | **GRF → 토크 사상 검증.** 9/15 Q1 을 눈으로 |
| `04_stand.py` | MPC 없이 서 있기 |
| **`05_sign_check.py`** | **부호 검증** — 우리가 제일 많이 틀렸던 것 |
| `06` / `07` / `08` | 1스텝 QP / B 자동미분 대조 / 지평 MPC 로 서 있기 |
| **`10_constraint_check.py`** | **제약 18행이 옳은지 LP 로 검증.** 9/18 Q4 §5 |
| `11_walk_srb_upper.py` | 상체 SRB 실험 (부정적 결과). **훅 2개만 재정의**한 구조 |
| **`12_contact_analysis.py`** | **착지~stance 의 명령 vs 실측.** 9/21 Q5 — 하중 인계 구멍 |

**추천 3개**: `03` → `05` → `10`. 컨트롤러의 세 축(사상·부호·제약)이 잡힌다.
**보폭 작업 전이라면** `12` 도 (지금 상태의 기준선이 된다).

```bash
.venv/Scripts/python.exe MPC/src/03_grf_to_tau.py
.venv/Scripts/python.exe MPC/src/05_sign_check.py
.venv/Scripts/python.exe MPC/src/10_constraint_check.py
.venv/Scripts/python.exe MPC/src/12_contact_analysis.py --vx 0.5 --plot
```

---

## 5. 현재 권장 실행 구성

```bash
.venv/Scripts/python.exe MPC/src/09_walk.py --view --vx 0.5 \
    --uppd 300 --swingid --softland --lamswing --wn 30 --zeta 0.7 \
    --sf 0.57 --qpy 300 --swingyaw
```

플래그가 많은데 **전부 실험으로 얻은 것**이고 각각 근거가 있다:

| 플래그 | 무엇 | 근거 |
|---|---|---|
| `--uppd 300` | 상체 PD 강화 | Step5 디버깅 |
| `--swingid` | 스윙 역동역학 보상 `M q̈` | [9/18 Q1](../Q&A/2026-09-18.md) |
| `--softland` | 착지·이륙 수직속도 0 | [9/16 Q3](../Q&A/2026-09-16.md) |
| `--lamswing --wn 30 --zeta 0.7` | 스윙 kp = ω_n²Λ(q) | 요약 §3 |
| `--sf 0.57` | 스윙 0.2 → 0.344 s | **요약 §4 (속도 벽 돌파)** |
| `--qpy 300` | 횡 위치 가중치 | 요약 §1 |
| `--swingyaw` | 스윙 발 yaw 정렬 상시 | [9/16 Q2](../Q&A/2026-09-16.md) |

> ⚠ **기본값은 아직 옛날 값**이다 (`soft_land=False` 등). 플래그 없이 돌리면
> 9/14 이전 컨트롤러가 나온다 — 비교 실험용으로 일부러 남겨둔 것.
>
> ⚠ **`--footframe` 은 켜면 안 된다** (제약 frame). 요약 §7 참고.

---

## 6. 분석 도구 (MPC 밖)

| 파일 | 역할 |
|---|---|
| `src/mpc_log.py` / `src/plot_log.py` | 로그 저장 / 6×2 플롯 (상태·입력·토크·솔브타임·스윙추종) |
| [`../deployed RL/analysis/run_g1_policy.py`](../deployed%20RL/analysis/run_g1_policy.py) | 배포 RL 정책 헤드리스 계측 |
| [`../deployed RL/analysis/sweep.py`](../deployed%20RL/analysis/sweep.py) | 속도 정밀 스윕 (보폭·sf·주기 회귀의 출처) |
| [`../deployed RL/analysis/detail.py`](../deployed%20RL/analysis/detail.py) | gait 타이밍, ω(pelvis) vs ω(전신 L) |
| [`../deployed RL/analysis/view_g1_policy.py`](../deployed%20RL/analysis/view_g1_policy.py) | 배포 정책 뷰어 (방향 유지 포함) |

---

## 7. 지금 고칠 곳 — 문서에서 읽은 "다음 할 일"이 코드 어디인가

| 우선순위 | 할 일 | 코드 위치 |
|---|---|---|
| **1** | **보폭 `w(v)`** | [gait.py L87](src/gait.py#L87) `hip = p_com + R @ side_offset` / [09_walk.py L71](src/09_walk.py#L71) `side_offset` 스냅샷 |
| **2** | **`Fz,min` 램프** | [mpc_qp.py L54](src/mpc_qp.py#L54) `−Fz ≤ −fz_min` 행 + `SRBParams.fz_min` |
| 3 | `sf(v)` 속도 함수 | [gait.py L22](src/gait.py#L22) `Gait.stance_frac` |
| 4 | 스윙 토크 클램프 | [gait.py L234](src/gait.py#L234) `SwingController.wrench()` 반환 직전 |
| 5 | `J̇q̇` 보상 | [09_walk.py L361](src/09_walk.py#L361) `swing_id` 블록 (`mj_jacDot` 사용) |
