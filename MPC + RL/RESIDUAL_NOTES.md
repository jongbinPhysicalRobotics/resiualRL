# Residual MPC 논문 이식 노트 — 무엇을 그대로 쓰고 무엇을 바꾸나

> Jeon, Lee, Hong, Kim, *"Residual MPC: Blending Reinforcement Learning with
> GPU-Parallelized Model Predictive Control"*, IEEE T-RO 2026.
> DOI 10.1109/TRO.2026.3721669 · 영상 youtu.be/watch?v=L2NrPD4yiMs
>
> 2026-09-16 작성. **residual 단계에 들어갈 때 이 문서부터 볼 것.**

---

## 0. 먼저 — 그들과 우리의 구조 차이

| | 논문 | 우리 |
|---|---|---|
| 로봇 | MIT Humanoid **25 kg** | G1 **33.4 kg** |
| 다리 관절 | **5/다리 (ankle roll 없음)** | 6/다리 |
| 발 접촉 | **선접촉** (4점: 좌우 toe/heel) | 면접촉 (2 site) |
| MPC 모델 | **kinodynamic (24 DoF)** — q, q̇, F 전부 최적화 | **SRB (13 상태)** — wrench만 |
| MPC 출력 | q_MPC, q̇_MPC, τ_RNEA | wrench W(12) |
| 지평 | T = 12 | N = 16 |
| 주기 | 0.8 s, φ=0.5 전환 → **stance 0.4 / swing 0.4, 이중지지 0 %** | 0.8 s, sf 0.57 → stance 0.456 / swing 0.344, 이중지지 14 % |
| 스윙 높이 | 0.075 m (험지 0.15) | 0.05 m |
| 관절 PD | Kp 50 Nm/rad, Kd 3 Nms/rad (**전 관절**) | 상체만 300/20, 다리는 작업공간 임피던스 |
| 팔 | **MPC 결정변수** (box 제약만) | **PD 고정 (얼어 있음)** |

> ⚠ **선접촉인데도 더 빠르다.** MIT Humanoid 는 ankle roll 이 없어 발이 선으로 닿고
> (부록 B), 횡 CoP 권한이 우리보다 **적다**. 그런데도 1 m/s 이상을 간다 —
> "발이 작아서 못 간다"가 아니라는 기존 결론을 다시 뒷받침한다.

### MPC 단독 성능 (우리 목표 눈금)

Fig. 10 에서 residual 이 MPC 대비 **vx 약 78 % 향상**이고 residual 이 2.25 m/s 를
낸다(Fig. 1) → **그들의 kinodynamic MPC 단독은 대략 1.2~1.3 m/s.**
우리 목표 1 m/s 는 그 범위 안이다. **단 그들은 SRB 가 아니라 kinodynamic** 이라는
점은 정직하게 감안해야 한다 (Table I: 선행 SRB 연구들은 14~18 DoF).

---

## 1. ★ 리워드 10개 전수 분류 (Table II)

**논문이 의도적으로 최소한으로 설계했다.** §IV-A 원문:
> "The set of rewards given to the system is **intentionally minimal**. We avoid
> giving overly specific rewards such as **foot guidance, air time, or
> contact-scheduling terms**..."

즉 보행을 빚어내는 항이 애초에 없다 — MPC 가 그 역할을 하므로. **그래서 이식성이 좋다.**

| # | 리워드 | 가중치 | 함수 | **판정** |
|---|---|---|---|---|
| 1 | Lin. vel. tracking | 10.0 | exp(−‖(c_xy−v_xy)/(1+\|c_xy\|)‖²/σ) | **그대로** — `(1+\|c\|)` 정규화라 속도·로봇 무관 |
| 2 | Ang. vel. tracking | 5.0 | exp(−‖c_ω−ω_z‖²/σ) | **그대로** |
| 3 | 1st order action rate | −1e-3 | ‖(a_t−a_{t−1})/Δt‖² | 스케일 — a 단위가 행동공간에 달림 |
| 4 | 2nd order action rate | −1e-4 | ‖(a_t−2a_{t−1}+a_{t−2})/Δt‖² | 스케일 |
| 5 | **Torques** | −1e-4 | ‖τ‖² | **스케일 필수** — 아래 |
| 6 | Orientation | 1.0 | exp(−‖g_xy‖²/σ) | **그대로** — 중력벡터 body xy, 무차원 |
| 7 | Height | 1.0 | exp(−‖c_z−p_z‖²/σ) | **그대로** — c_z 는 명령이라 로봇 무관 (σ만 확인) |
| 8 | Joint regularization | 1.0 | (1/n_j)Σ‖q_j−q̂_j‖² | 그대로 — **q̂ 만 우리 crouch 자세로**. 1/n_j 정규화가 이미 있어 관절수 차이 흡수 |
| 9 | Self-collision | −1.0 | 1_collision | 그대로 — **충돌쌍 정의만 G1 것으로** |
| 10 | Termination | −100 | 1_terminate | **그대로** — 속도·자세·높이 이탈 |

> ### 결론: **10개 중 6개 그대로, 3개 스케일, 1개 기준자세만. 버릴 건 없다.**
> "리워드 그대로 써도 되지 않을까" 하는 직관이 **맞았다.**

**#5 Torques 를 왜 반드시 다시 재나**: 25 kg·10관절 → 33.4 kg·12관절이라
‖τ‖² 가 통째로 커진다. 가중치를 그대로 두면 토크 페널티가 과도해져 정책이
움츠러든다. **우리 정상보행 τ 의 RMS 를 재서 (τ_ours/τ_theirs)² 로 나눠주면 된다** (미측정).

**#3·#4 action rate**: a 는 관절 setpoint(rad)라 로봇 간 비슷하지만, 실효 크기는
λ·Kp 에 걸린다. Kp 를 우리 값으로 바꾸면 같이 조정.

---

## 1-B. ★ **순수 E2E RL 에 이 리워드를 그대로 쓰면? — 논문이 이미 해봤다**

§V 원문:
> "For consistency, the **end-to-end RL baseline is trained with the same reward**
> used for the residual controller."

**같은 리워드 표로 E2E 를 학습시킨 대조군이 논문 안에 있다.** 결과는 두 얼굴이다.

### (a) 숫자상으로는 "된다"

- Fig. 7: E2E 가 residual 과 **거의 같은 점근 리워드**에 도달 (residual 이 조금 더 빠르고 높음)
- Fig. 10: E2E 의 **추종 가능 속도 영역이 residual 보다 오히려 넓다**

### (b) ★ 그런데 걸음이 가짜다

Fig. 8(B) 원문:
> "the end-to-end policy learns to **rapidly oscillate the ankle joint** of the
> humanoid to **'glide' across the floor, exploiting the physics of the
> environment.** While this behavior may be feasible in the IsaacGym simulation,
> it is unlikely that this would transfer safely to hardware."

Fig. 10 에 대해서도:
> "The velocities achievable by the end-to-end policy are likely attained by
> **exploiting numeric irregularities in the contact dynamics of the simulator.**"

Fig. 8(C) 정량 비교 — **같은 토크·action rate 페널티를 받고도** E2E 가 전부 나쁘다:

| 지표 | residual | E2E |
|---|---|---|
| 관절속도 | 중앙값·분산 작음 | 둘 다 **큼** (~5 rad/s 까지) |
| 관절토크 | 작음 | **큼** (~12 Nm 까지) |
| 기계적 파워 | 작음 | **큼** (~20 W) |
| 수직 GRF | ~200–350 N | **~600 N 까지** |

Fig. 9 (UMAP): E2E 는 MPC 와 **완전히 다른 상태분포**로 수렴한다.

### (c) 그래서 결론

> **리워드가 "좋은 보행"과 "미끄러지기"를 구분하지 못한다.**
> 리워드 값은 같은데 거동은 전혀 다르다 — 이게 논문의 핵심 논지다
> (MPC prior 가 리워드 엔지니어링을 대신한다).

**§1 의 "그대로 써도 된다"는 residual 일 때만 성립한다.** 순수 E2E 로 쓰려면
논문이 **일부러 뺀** 항들을 도로 넣어야 한다 (§IV-A 가 뺐다고 명시한 것):
- **foot guidance** (발 궤적 유도)
- **air time** (체공 시간)
- **contact scheduling** (접촉 스케줄)

논문도 이걸 인정한다:
> "purely learned policies are often **sensitive to reward design and may benefit
> from more tailored shaping** than what is used here"

*(이하는 논문 밖 일반 지식)* 실제 E2E 구현들은 여기에 더해 발 미끄러짐 페널티,
좌우 대칭, 관절 한계, 발 clearance 등을 더 넣는다 — unitree_rl_gym 의 G1 config 가
그런 형태다. **즉 E2E 리워드는 이 표가 아니라 그쪽을 봐야 한다.**

### (d) 참고 데이터 목적이라면 — 이게 더 중요하다

E2E 를 **"잘 걷는 대조군에서 계측하기"** 용도로 쓰려던 것이었으니
(ω_pelvis, 팔 사용, gait 타이밍 측정), **리워드가 부실한 E2E 는 참고자료로 최악**이다.
미끄러지는 정책을 재봐야 우리 MPC 에 시사하는 바가 없다.
→ **직접 학습하지 말고 shaping 이 제대로 된 공개 정책(unitree_rl_gym G1)을 쓸 것.**

### (e) 발표자료용 E2E 베이스라인 — **하되, 비교쌍이 있어야 의미가 있다**

논문이 이 리워드로 E2E 를 돌린 목적 자체가 **통제된 대조군**이다:
> 같은 리워드 · 같은 환경 · 같은 게인 → **MPC prior 유무만 다르게.**

그러니 G1 에서 재현하면 발표에 쓸 수 있는 그림이 나온다. **'미끄러지는 실패'는
버그가 아니라 결과물**이다 — 우리 논지(MPC prior 가 리워드 엔지니어링을 대신한다)를
직접 뒷받침한다.

단 조건이 있다:

1. **삼자 비교(MPC / residual / E2E)가 갖춰져야 한다.** E2E 만 돌려 놓으면
   "리워드 부실한 정책이 이상하게 걷는다"에 그친다. **residual 이 생긴 뒤**가 제자리
2. **미끄러짐 재현은 장담 못 한다.** 그건 **IsaacGym 접촉 동역학을 착취한 것**이고,
   G1 은 **ankle roll 이 있어**(MIT 는 없음) 발 자유도·접촉이 다르다.
   다른 형태의 실패가 나올 가능성이 높다 — "어떤 실패가 나오는지"를 보고하면 된다
3. **⚠ 이 PC 로는 학습 불가** — NVIDIA GPU 없음(Intel Arc 내장뿐).
   IsaacGym/IsaacLab(CUDA 필수), MJX(Windows JAX GPU 미지원), CusADi/cuDSS 전부 불가.
   **랩 GPU 머신이나 클라우드 필요.** 반면 **사전학습 정책 실행은 onnxruntime CPU 로
   이 PC 에서 가능** — 참고 데이터 계측은 여기서 해도 된다
4. **출처 명시**: 리워드 표(Table II)와 "E2E 에 같은 리워드를 쓰는 대조군 설계"는
   **둘 다 Jeon et al. 2026** 의 것이다. 발표에 그대로 밝힐 것

---

## 2. ★★ 블렌딩 — **우리 SRB MPC 와 호환된다** (핵심 발견)

논문은 3가지를 비교했다 (§IV-B):

| | 식 | 우리 적용 가능? |
|---|---|---|
| ① joint action, **joint** blending | τ = Kp(q_cmd+λa−q) + Kd(q̇_cmd−q̇) + τ_cmd (22) | **✗ 불가** — q_cmd 가 필요한데 SRB 는 관절 궤적을 안 낸다 |
| ② joint action, **torque** blending | τ_res = Kp(a + **q̂** − q) − Kd q̇ (23)<br>τ = τ_MPC + λ·τ_res (24) | **✓ 가능** |
| ③ torque action, torque blending | τ = τ_MPC + λa (25) | 가능하나 **논문 실험에서 명확히 열등** (Fig. 6) |

**②가 되는 이유**: 식 (23)이 `q̂`(**기본 관절자세**)를 기준으로 쓰고 `q_MPC` 를
쓰지 않는다. 그래서 **우리 MPC 가 관절 궤적을 못 내도 그대로 성립한다.**
필요한 건 `q̂, Kp, Kd, q, q̇` 뿐이고 `τ_MPC` 는 우리 것을 그냥 넣으면 된다.

그리고 **논문이 최종 채택한 것도 ②** 다. 이유가 우리에게도 유리하다:
> "in the event of a diverging MPC solution, the joint-joint strategy would output
> actions relative to potentially infeasible q_MPC setpoints."

### 우리 토크식이 그들 것과 어떻게 대응되나

그들 (식 20-21):
```
τ_RNEA = M(q_MPC)q̈_MPC + h(q_MPC,q̇_MPC) − J(q_MPC)ᵀF_MPC
τ_MPC  = Kp(q_MPC − q) + Kd(q̇_MPC − q̇) + τ_RNEA
```
우리 ([09_walk.py](../MPC/src/09_walk.py) `torque()`):
```
τ = qfrc_bias − JᵀW           (+ M q̈_swing  ← --swingid)
```
- `qfrc_bias` = `h`, `−JᵀW` = `−JᵀF`, `--swingid` = `M q̈` 항 → **τ_RNEA 와 같은 구조**
- **없는 것**: `Kp(q_MPC−q) + Kd(q̇_MPC−q̇)` 관절 피드백. SRB 라 관절 참조가 없다.
  (스윙 다리는 작업공간 임피던스로 유사 역할을 하지만 stance 다리엔 없다)
- λ = **0.1**

---

## 3. 관측 벡터 (§IV)

```
o = [p, θ, q_j, ω, v, q̇_j, φ, V_MPC] ∈ R⁵⁴
```
- **`V_MPC` 는 MPC 의 최적 비용값** — 출력 토크·예측상태가 아니다. 그들이 원래
  토크·예측을 넣었다가 **sim2sim/sim2real 이 나빠져서 값만 남겼다.** 우리도 quadprog
  목적함수 값을 그대로 쓰면 된다 (**공짜, 그대로 이식**)
- `φ` 는 접촉 위상 (그들 4점, 우리 2점)
- 출력 `a ∈ R¹⁰` = **다리만**
- 차원은 우리 관절수에 맞춰 다시 세면 됨

---

## 4. 학습 설정 (§IV-C, §V)

| 항목 | 값 |
|---|---|
| 알고리즘 | PPO (clipped + GAE), 하이퍼파라미터는 [58] 그대로 |
| 병렬 환경 | **2048** (MPC 의 VRAM 제약) |
| PPO iter 당 스텝 | 24 |
| 시뮬 | IsaacGym |
| 도메인 랜덤화 | **지면 마찰 [0.6, 1.0], 링크 질량 ±0.25 kg — 이게 전부** |
| 정책망 | MLP 3층 × 256 ELU |
| 초기화 | **zero-init** (residual 이 처음엔 MPC 를 안 건드리게) |
| 학습 시간 | PPO 1000 iter 에 3~4 시간 (E2E 는 30분, **8배 느림**) |

> 도메인 랜덤화가 이렇게 적은 게 인상적이다 — MPC prior 가 그 역할을 대신한다는 게 논문 주장.

---

## 5. 팔 — 내가 앞서 한 말을 정정해야 한다

논문 §IV 원문:
> "we chose to limit our scope to adapting the **leg torques only**... Although the
> residual networks were tested with arm actions as well, we observed only
> **minimal deviations** from the MPC controller's predictions for the upper body.
> Additionally, simple **box constraints on the arm joints** were sufficient to
> prevent self-collision from the arms through the MPC horizon."

**이건 "팔이 중요하지 않다"가 아니다.** 그들의 kinodynamic MPC 는 q 에 **팔 관절이
이미 들어 있어서 MPC 가 팔을 계획한다.** residual 이 더 보탤 게 없었던 것.

**우리는 팔이 PD 로 얼어 있어 아무도 계획하지 않는다.** 상황이 다르다.
→ **팔은 residual 이 아니라 MPC 쪽에 넣는 게 논문과도 일치하는 방향이다.**

그리고 논문이 팔 전용으로 인용한 후속 논문이 있다:
> [55] H. J. Lee, S. H. Jeon, S. Kim, **"Learning humanoid arm motion via
> *centroidal momentum regularized* multi-agent reinforcement learning"**, RA-L 2025.

**"centroidal momentum regularized"** — 9/16 Q8 에서 우리가 측정한 바로 그 문제
(다리 L 을 상체가 상쇄하느라 pelvis 가 ±4° 흔들림)를 정면으로 다루는 논문이다.
**팔 작업 시작 전에 반드시 읽을 것.**

---

## 6. 오늘 실험과 직접 겹치는 사실들

### (a) 스윙 높이 0.15 → **MPC 단독은 실패** (Fig. 16)
> 험지용으로 0.075 → 0.15 로 올리자 **MPC 단독은 종료(terminate)**, residual 은 적응.

우리 9/16 **Q5 와 정확히 일치** (0.15 에서 0.5 m/s 15.4 s, 0.6 m/s 7.6 s 전도).
우리 MPC 가 0.10 까지 버틴 건 나쁘지 않은 축.

### (b) 착지 — residual 은 **heel-toe 2단 착지**를 스스로 배운다 (Fig. 19)
> "The **touchdown velocity** of the residual policy returns to zero in **two
> discernible stages**, roughly when the phase is 0.48 and 0.52... In comparison,
> the average **MPC touchdown velocity is a single sharp peak**, and the adherence
> to a **strict contact schedule** makes it difficult to adapt to unexpected contacts."

우리 9/16 **Q3 과 같은 가족의 문제**다 (스케줄 착지 시각 접촉률 0 %, 발이 8.5 ms 늦음).
논문도 **MPC 의 엄격한 접촉 스케줄**을 원인으로 지목한다.
단 그들의 해법은 residual — 우리는 **MPC 층의 접촉 인지**로 풀어야 한다
(Fz,min 램프 / 실측 접촉 / N_lock).

### (c) residual 은 **적대적(antagonistic)** 이고 접촉 전환에서 가장 활발 (Fig. 18)
코사인 유사도 −0.6 ~ −1.0, 접촉 전환 근처에서 −1 에 가까움. 발목을 크게 쓴다.
→ **residual 은 MPC 를 증폭하는 게 아니라 반대로 당긴다.** 나중에 λ 를 볼 때 기억할 것.

### (d) 타이밍 — **우리 Q6 결론을 한정해야 한다**
그들: 주기 0.8 s, **stance 0.4 / swing 0.4, 이중지지 0 %** 로 1.2 m/s 이상.
우리 Q6: swing 0.43 s 로 늘렸더니 0.6 에서 추종 82 → 16 % 로 붕괴.

**같은 0.8 s 주기에 그들 스윙(0.4 s)이 우리(0.344 s)보다 길다.** 즉
**"스윙이 길면 횡이 지수 발산해서 안 된다"는 건 보편 법칙이 아니라 우리 구현의 한계다.**
Q6 의 측정 자체는 유효하지만 결론은 "우리 현재 구성에서는"으로 한정해야 한다.
차이의 후보: kinodynamic(관절 한계·전신 예측) vs SRB, 팔 활용, 질량 25 vs 33 kg.

---

## 7. residual 착수 시 체크리스트

1. [ ] 리워드 10항 이식 — 6개 그대로, #3·#4·#5 스케일, #8 q̂ 교체, #9 충돌쌍 정의
2. [ ] `‖τ‖` RMS 측정 → #5 가중치 재계산
3. [ ] 블렌딩 ② (식 23-24), λ=0.1, zero-init
4. [ ] 관측에 `V_MPC` (quadprog 목적함수 값) 추가
5. [ ] **그 전에**: 1 m/s 를 MPC 단독으로 (→ [[mpc-alone-to-1ms]])
6. [ ] 팔은 residual 이 아니라 **MPC 입력**으로. [55] 먼저 읽기
