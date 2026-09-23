# reference/ (ispaik06/convex-mpc-biped) 코드 분석

> 2026-09-15. https://github.com/ispaik06/convex-mpc-biped 를 `reference/` 에 clone 후 분석.
> C++17 + MuJoCo + OSQP + Eigen. G1(23dof)/H1/MIT Humanoid 지원.
> 데모 성능: **전진 0.6 m/s, 횡보행 0.3 m/s, 제자리 회전 1.3 rad/s** — 우리 병목(회전 0.2)의 6.5배.
> 상세 유도 문서가 `reference/docs/` 에 있음 (아래 스터디 포인트 참고).

---

## 0. 한 줄 요약

수식(13-상태 wrench SRB, condensed QP)은 우리와 **거의 동일**하다. 성능 차이는 수식이 아니라
**(1) SRB 를 상체만으로 정의**, (2) **회전을 참조·발디딤·제약·자세제어 전부에 일관되게 주입**,
(3) **스윙 다리를 관성 스케줄링된 작업공간 제어로 고대역폭 추종**, (4) **접촉 이벤트 관리(램프·조기접촉)**,
(5) **제약은 일부러 느슨하게** — 이 다섯 가지 설계 결정에서 나온다.

---

## 1. 아키텍처 개요

```
SimulationRunner (500 Hz, mj_step dt=0.002, implicitfast)
 ├─ updateReducedBodyMassPropertiesFromData()   ← 매 틱 상체 SRB 질량/관성 갱신
 ├─ StateEstimator (Cheater = sim 진값, yaw unwrap)
 ├─ MyController::runController()               ← 매 틱 (500 Hz)
 │   ├─ MPC solve: 5틱마다 (100 Hz), OSQP, shifted warm start
 │   │    horizon N=25, dt=0.02 (0.5 s preview)
 │   ├─ SwingFootPlanner  → 착지목표 (스윙 시작 시 1회 확정)
 │   ├─ ContactManager    → 실측 접촉력 기반 조기접촉/램프/지면탐색
 │   └─ writeLegCommands()
 │        stance: τ = Jᵥᵀ(−F) + J_ωᵀ(−M) + [발 yaw 유지 모멘트]
 │        swing : 작업공간 OSC (Λ 스케줄링 kp) + 발 자세 3축 PD
 └─ ArmController: 관절 PD 로 고정 자세
```

우리 구조(09_walk.py, 500 Hz torque / **100 Hz MPC**)와 계층도 주기도 사실상 같다.
⚠ 2026-09-15 정정: 이 문서 초판에서 우리 MPC 를 20 Hz 로 적었는데 **오류**였다
(DECIM=5 × 500 Hz = 100 Hz). 실제 차이는 재풀이 주기가 아니라 **지평 이산화**
(dt 0.05 vs 0.02) 와 **지평 길이**(0.8 s vs 0.5 s) 다.

---

## 2. 가장 큰 설계 차이: SRB = 상체만 (다리 제외)

`setupRobotParams.cpp` 의 `updateReducedBodyMassPropertiesFromData()`:

- **다리 서브트리에 속한 모든 body 를 제외**하고 질량·CoM·관성을 합성한다.
- 매 제어 틱마다 현재 자세(팔 위치)로 재계산.
- 상태도 일관되게 **torso 강체 프록시**: Θ = torso quat, ω = torso 각속도,
  CoM = torso 원점 + (고정 오프셋을 yaw 회전) — `subtree_com` 도 각운동량 ω 도 안 쓴다.

이게 뭘 해결하나:

| 우리 문제 (PHYSICS_MAP/VERIFY_REPORT) | reference 의 답 |
|---|---|
| 다리 질량 43.1% → massless-leg 위반이 0.3 m/s 벽의 원인 (--legmass 로 격리 입증) | 다리를 SRB 에서 **아예 제외**. 스윙 다리 동역학은 스윙 컨트롤러가 bias(중력·코리올리) 보상으로 스스로 처리 |
| "Θ와 ω가 같은 강체가 아니다" 구조적 불일치 (Θ=pelvis, ω=전신 L) | Θ·ω 둘 다 torso — **같은 강체**로 일관. 모델(상체)과 상태(상체)가 정합 |

즉 우리가 "전신을 정확히 재는" 쪽으로 갔다면, reference 는 "모델과 상태의 **정합성**"을
택했다. 다리가 흔드는 반작용은 상체 SRB 입장에서 외란이지만, 100 Hz MPC + 아래 5절의
고대역폭 스윙 제어가 그 외란 자체를 작게 만든다.

⚠ 정직한 관찰: stance 다리 자중 보상(bias)은 **주석 처리**되어 있다 (`LegController.cpp`
computeStanceLegTorque). 상체 질량만큼만 wrench 로 지탱하고 stance 다리 무게는 구조로
흘려보내는 셈 — 엄밀하진 않지만 위치 가중치가 커서 MPC 피드백이 잔차를 흡수한다.

---

## 3. MPC 정식화 — 우리와 1:1 비교

| 항목 | 우리 (src/mpc_qp.py) | reference (MPCFormulation/ConvexMPC) |
|---|---|---|
| 상태/입력 | x∈R13, u∈R12 (wrench) 동일 | 동일 (입력 순서만 [F_L,F_R,M_L,M_R]) |
| 이산화 | 정확 ZOH (A 멱영) | 동일 + dt³/6 항 (구조상 0이라 무해) |
| **선형화점** | k=0 현재 상태 (ψ 고정) | **매 스텝 ψ_k = 참조 yaw(k)**, I_w(ψ_k), r_k = 계획 발위치 − p_ref(k) |
| 참조 궤적 | 직선 (world 고정) | **원호**: p_ref 를 Rz(ψ_k)·v_cmd 로 적분, ψ_k 는 horizon 내내 적분 |
| dt × N | 0.05 × 16 = 0.8 s | **0.02 × 25 = 0.5 s** (지평 이산화가 2.5배 촘촘) |
| solve 주기 | **100 Hz** (DECIM=5 × 500 Hz) — 동일 | **100 Hz** (5 sim틱마다) |
| solver | quadprog (dense) | OSQP (sparse) + shifted warm start, max_iter 200 |
| 마찰 μ | 0.6 (sim 0.8보다 보수) | **1.0 (sim 0.8보다 느슨!)** |
| CoP 박스 | 실측 발 형상 (l_t=0.12, l_h=0.05, w=0.025) + h_sole 결합항 | **half_len 0.15 / half_wid 0.35 — 실물의 1.8×/12×**, h 결합항 없음 (site 가 발바닥 위 3 cm 인데 무시) |
| mz 제약 | Caron 정확식 8행 (LP 100% 일치) | **\|Mz\| ≤ 0.08·μ·Fz 단순 박스 2행** |
| 스윙 발 | W=0 등식 (동일) | W=0 등식 (동일). line-foot 용 no_roll_moment 모드 별도 |
| 제약 frame | 발별 실측 yaw (wz≠0 일 때만) | **발별 실측 발 x축 yaw, 항상 적용** — 우리 18.6절 발견과 동일 결론 |

가중치 (G1 walking):

```
Q = [roll 900, pitch 900, yaw 900 | px 3000, py 42000, pz 50000 | ω 10,10,10 | v 10,10,90 | g 1]
R = 1e-7 (사실상 자유)
```

- **위치 지배** 설계: py/pz 가 각도의 ~50배. 속도 가중치는 미미. 입력 정규화는 0에 가깝다.
- 해석: "wrench 는 얼마든 써도 좋으니 위치를 데드비트로 잡아라" + 물리 실현성은
  (느슨한) 제약과 sim 접촉이 알아서 자르게 둔다.
- 우리 접근(정확한 CWC 로 실현 가능한 wrench 만 계획)과 정반대 철학. reference 는
  μ_MPC=1.0 > μ_sim=0.8 — **MPC 가 sim 보다 낙관적**이어도 100 Hz 재계획이 수습한다.

---

## 4. 회전 1.3 rad/s 가 되는 이유 — 회전의 전 계층 주입

우리 18.7절에서 "회전이 발디딤/sway 계획에 반영 안 됨 + hip_yaw 미사용"을 다음 병목으로
지목했는데, reference 는 정확히 그 지점들을 전부 구현해 놨다:

1. **yaw 참조 적분기** (`_bodyTarget.euler_W[2]`): 측정 yaw 가 아니라 ψ̇ 명령을 적분한
   **독립 참조**가 존재. MPC 참조·발디딤·스윙 yaw 목표가 전부 이걸 본다.
   (몸이 덜 돌았어도 참조는 계속 전진 → 정상상태 추종오차가 누적되지 않음)
2. **horizon 원호 참조** (3절): ψ_k, p_ref(k) 가 스텝마다 회전. 1.3 rad/s × 0.5 s = 0.65 rad
   — k=0 고정 선형화면 horizon 끝에서 37° 틀린다. 우리 0.5 rad/s 목표 × 0.8 s = 0.4 rad 도 마찬가지.
3. **착지 발 사전 회전** (`SwingFootPlanner`):
   - 보폭 방향: 미리보기 중간 yaw (ψ₀ + ½ψ̇·T_prev) 로 회전 — 원호의 현
   - 공칭 발 오프셋: **착지 순간 yaw** (ψ₀ + ψ̇·잔여스윙시간) 로 회전 — 발이 미리 돌아서 착지
4. **스윙 발 yaw 리드** (`SwingYawTarget.h`): 스윙 발 yaw 목표 = 몸 yaw 참조
   + ψ̇·T_prev (lead_scale=1.0) + **회전 안쪽 발에 추가 toe-in 바이어스** 100°/(rad/s), 최대 20°.
   1.3 rad/s 면 안쪽 발이 20° 먼저 돌아 딛는다.
5. **스윙 발 3축 자세 PD** (`SwingAttitudeControl.h`): roll/pitch 수평 유지 + yaw 목표 추종을
   J_ωᵀ 모멘트로. kp = 200/250/100. (우리는 yaw 스프링 1축만, 그것도 wz≠0 게이트)
6. **★ stance 발 yaw 유지 모멘트** (`computeStanceYawHoldMomentWorld`): stance 중 MPC wrench 에
   **world-z 모멘트 PD (kp=100, kd=10) 를 추가** — 발이 착지 yaw 를 유지하도록 hip_yaw 가
   능동적으로 버틴다. 우리가 "hip_yaw 미사용"이라 지목한 바로 그 조인트가 여기서 일한다.
   MPC 밖에서 더하므로 mz 제약과는 별개 채널 (엄밀히는 제약 위반 가능 — 이것도 느슨한 철학).
7. **제약 frame = 발별 실측 yaw, 상시 적용**: 우리 18.6 발견과 동일. 우리는 직진 회귀 때문에
   wz≠0 게이트를 달았는데, reference 가 상시로 문제없는 건 아래 5·6절(스윙 대역폭·접촉 관리)이
   발 yaw 흔들림 자체를 잡아주기 때문으로 보인다.

---

## 5. 스윙 다리 제어 — 관성 스케줄링 작업공간 제어

`OperationalSpaceDynamics.cpp` — 우리 WBC-lite 의 완성형:

```
Λ(q) = (J M⁻¹ Jᵀ)⁻¹                    (겉보기 관성, 다리 질량행렬 사용)
kp   = ω_n² · diag(Λ)                  (ω_n = 100 rad/s 고정! 자세가 바뀌어도 대역폭 일정)
                                       ※ 2026-09-23 확인: 축별 벡터다 —
                                         G1 config [100,100,100], MIT 휴머노이드 [151,151,110]
                                         kd 는 Λ 스케줄이 아니라 고정 [20,20,23] → ζ = kd/(2ω_nΛ)
                                         = 0.01 ~ 0.08 (거의 무감쇠). 우리는 ω_n 30 · ζ 0.7 (Q&A 9/23 Q9)
τ    = Jᵀ[kp e + kd ė] + Jᵀ Λ (a_des − J̇q̇) + bias(중력·코리올리)
```

| 항목 | 우리 (gait.py SwingController) | reference |
|---|---|---|
| 피드백 | 고정 kp=2500 → ω_n ≈ 30 rad/s | **Λ 스케줄링, ω_n = 100 rad/s 일정** |
| 피드포워드 | M q̈_des (WBC-lite, --swingid) | Λ(a_des − J̇q̇) — 코리올리까지 |
| 다리 중력 | 보상 없음 (PD 가 떠받침) | **bias 로 정확 보상** |
| 힘 상한 | 200 N 클램프 | 없음 (대역폭으로 승부) |
| 발 자세 | yaw 스프링 1축 (조건부) | 3축 PD 상시 |

검증 리포트 §5(b) "known disturbance 처방을 먼저" — reference 는 스윙을 '외란'이 아니라
'자기 책임'으로 만들어 반작용의 크기 자체를 줄였다.

---

## 6. 스윙 궤적과 접촉 관리 — 몸통 흔들림 대책이 이미 들어있다

**궤적** (`SwingFootTrajectory.cpp`): xy 는 smoothstep(3s²−2s³), z 는 2단
(올라가서 zMid, 내려와서 target) smoothstep. smoothstep 은 양 끝 미분이 0이라
**착지 수직속도가 구조적으로 0** — 우리가 17.5절에서 계획만 해둔 "착지 속도 0" 이
여기선 기본값이다. 착지 목표는 **스윙 시작 시 1회 확정 후 동결** (우리처럼 매 틱 재계획
안 함 → 착지 직전 목표 흔들림 없음).

**접촉 관리** (`ContactManager.cpp`, 우리에게 없는 계층):

- 실측 접촉력 기반 판정: on 10 N / off 0.5 N + 확인 틱 (스케줄이 아니라 **실제 접촉**)
- **조기 접촉**: 예정보다 일찍 닿으면 즉시 stance 로 승격, 닿은 위치에 목표 동결
  → 늦은 스윙이 지면을 계속 누르는 사고 방지
- **wrench 램프**: 착지 후 0.02 s 동안 α를 0→1 — 충격 토크 계단 제거 (몸통 흔들림 직결)
- 접촉 소실 시 지면 탐색(0.4 m/s 하강), stance 중 접촉 상실 감지

**게이트 타이밍**: cycle 0.7 s, stance 0.4 (57%), swing 0.3, 이중지지 0.1 s (14%).
우리(0.8 / 75% / 0.2, 이중지지 25%)보다 스윙이 길고 이중지지가 짧다 — 스윙을 느긋하게
(가속도↓ = 반작용↓), 대신 단일지지 균형은 MPC 100 Hz 가 감당.

**발디딤 계획에 capture point 가 없다**: 착지목표 = CoM + Rz·v_cmd·(T_st/2) + 공칭 오프셋.

> **정정 (9/21 Q10)**: 목표는 스윙 **시작 시** 계산해 고정하므로 기준 CoM 은 **이륙 순간** 값이다.
> 착지 순간 CoM 대비 발 위치 = `v·((0.5+k)·T_st − T_sw)`. G1 설정(k=0, T_st 0.4, T_sw 0.3)이면 **−0.1·v, 발 중앙이 CoM 뒤**.
> MIT 는 k 를 속도에 따라 0.37 → 0.28 → 0.26 으로 줄여 0.5~0.7 m/s 에서 약 6 cm 로 묶는다.
> 자세한 계산과 우리 실측 비교는 [Q&A 9/21 Q10](Q&A/2026-09-21.md).
속도 오차 피드백 항이 **없다** (정지 시에만 capture gain 0.2). 속도 안정화를 전부
MPC(py 42000)에 맡기는 설계. 우리는 반대로 발디딤 피드백(1/ω₀ + k_pos)이 주력.
→ 주기가 같은데도(둘 다 100 Hz) 설계가 갈린 것이므로, 차이를 만든 건 주기가 아니라
**가중치 철학**(그들 위치 극단 우세)이다. Q&A Q4 참고.

---

## 7. 기타 관찰

- **cheater state** (sim 진값) — 우리와 같은 전제. 추정기는 TODO 로 비어 있음.
- G1 모델: 우리와 같은 계보의 4구 접촉 발 (x −0.05~0.12, y ±0.025/0.03, sim μ=0.8).
  site `foot_contact_site` pos (0.035, 0, −0.03) — 우리 x_c=0.035, h≈0.035 와 일치.
  23dof (손목 고정) + 팔은 관절 PD 고정.
- 명령 필터: 1차 저역 (v: τ=0.2 s, ψ̇: τ=0.3 s), ψ̇ 상한 1.0 rad/s.
- 시작 시퀀스: 다리 PD 초기화 2 s → **standing MPC 2.5 s 정착** → 보행 개시.
- standing 모드는 별도 가중치 세트 (yaw 10 으로 낮춤 등).
- yaw unwrap 정책 문서화 (docs/yaw_wrapped_unwrapped_policy_for_mpc.md, 1314줄) —
  ±π 랩 사고를 상태·참조 모두 unwrapped 로 통일해서 회피.

---

## 8. 우리 컨트롤러에 이식할 후보 (우선순위)

| # | 항목 | 예상 효과 | 비용 |
|---|---|---|---|
| 1 | **stance 발 yaw 유지 모멘트** (world-z PD → J_ωᵀ, torque() 에 추가) | 회전 0.3+ 의 "발 감김" 해소 — 최소 변경 최대 효과 후보 | 낮음 (~15줄) |
| 2 | **착지 발 사전 회전 + 스윙 yaw 리드/toe-in** (raibert_target, yaw_des 에 ψ̇ 항) | 전진+회전 조합의 발-몸 정렬 | 낮음 |
| 3 | **horizon 원호 참조 + ψ_k 별 선형화** (solve_gait 에 이미 시변 Bd 있음 — ψ_k 만 추가) | 회전 preview 정확도 (0.4 rad 오차 제거) | 중간 |
| 4 | **z 궤적 착지속도 0** (스윙 z 를 2단 smoothstep 으로) | 몸통 흔들림 (계획해둔 항목) | 낮음 |
| 5 | **착지 wrench 램프 0.02 s + 조기접촉 승격** | 충격 계단 제거 — 흔들림 | 중간 |
| 6 | **스윙 kp 를 Λ(q)·ω_n² 로 스케줄링** (M 은 WBC-lite 가 이미 계산) + 다리 중력 bias 보상 | 스윙 추종 대역폭 30→상향, 반작용 감소 | 낮음 |
| 7 | **상체만 SRB 실험** (make_params 에서 다리 서브트리 제외 + 상태를 pelvis 일관 쌍으로) | 43% 다리질량 문제의 원리적 우회 — 논문 비교 실험 가치 | 중간 |
| 8 | 지평 이산화 촘촘하게 (dt 0.05 → 0.02) — 주기는 이미 같음 | preview 정확도 | 중간 (QP 비용 ↑) |

주의: 1·2·5는 우리 "정직한 물리" 서사와 충돌하지 않는다 (제어 채널 추가일 뿐).
반면 "제약 느슨하게 + μ 낙관" (reference 철학) 은 우리 mz/Caron 검증 서사와 정면 충돌 —
채택하지 말고 **비교 대상**으로 논문에 쓰는 게 낫다: "정확 CWC(우리) vs 완화 제약 +
고주기 재계획(reference)".

---

## 9. 스터디 포인트 (reference/docs/)

| 문서 | 내용 |
|---|---|
| `friction_cop_constraint_c_matrix_build.md` (1429줄) | C 행렬 유도 전 과정 — 우리 MPC_NOTES 5절과 대조하며 읽기 |
| `mpc_frame_convention.md` | frame 규약 (우리 18.6 frame 분리 발견과 대조) |
| `swing_foot_touchdown_planner.md` | 착지 계획 유도 |
| `gait_scheduler_and_contact_management.md` | 접촉 관리 상태기계 |
| `yaw_wrapped_unwrapped_policy_for_mpc.md` | yaw ±π 랩 정책 |
| `reference_trajectory.md` | 원호 참조 유도 |

코드 진입점 추천 순서: `config/unitree_robots/g1/my_controller.yaml` →
`My_Controller/src/My_Controller.cpp` (runController/writeLegCommands) →
`MPCFormulation.cpp` → `ConvexMPC.cpp` (제약) → `SwingFootPlanner.cpp` →
`common/src/Dynamics/OperationalSpaceDynamics.cpp` → `ContactManager.cpp`.
