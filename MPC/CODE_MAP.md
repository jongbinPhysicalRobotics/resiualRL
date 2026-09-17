# 코드 읽기 지도 — 로봇 구현을 이해하려면 어디를 보나

> 파일 18개 중 **실제 컨트롤러는 5개**뿐이다. 나머지는 검증 스크립트(01~08, 10)와
> 유틸(로그·플롯·경로). 아래 순서대로 읽으면 "상태 → 계획 → 최적화 → 토크" 가 이어진다.

---

## 0. 한 장 요약 — 데이터가 흐르는 길

```
                      ┌─────────────────────────────────────────┐
[500 Hz 물리 루프]     │  09_walk.py : headless() / view()        │  ← 여기서 시작
                      └──────────────┬──────────────────────────┘
                                     │
   매 5틱(=100 Hz) ─────────────────► WalkController.update_mpc()
                                     │   ① mpc_srb.get_state()      상태 x(13)
                                     │   ② gait.Gait                누가 딛고 있나
                                     │   ③ gait.raibert_target()    어디에 딛나
                                     │   ④ 참조 X_ref 조립          어디로 갈 건가
                                     │   └► mpc_qp.solve_gait()     QP → wrench W(12)
                                     │
   매 틱(500 Hz) ───────────────────► WalkController.torque()
                                         stance: τ −= JᵀW
                                         swing : τ += Jᵀ(gait.SwingController.wrench)
                                         전체  : τ += qfrc_bias
                                     │
                                     └──► d.ctrl[:] = τ  →  mj_step
```

---

## 1. 반드시 봐야 할 5개 (이게 컨트롤러 전부다)

| 순서 | 파일 | 줄 수 | 무엇 |
|---|---|---|---|
| **①** | [src/mpc_srb.py](src/mpc_srb.py) | 190 | **모델** — 상태 x(13), 입력 u(12), 연속 A·B, 이산화 |
| **②** | [src/gait.py](src/gait.py) | 277 | **계획** — 접촉 스케줄, 착지점(capture), 스윙 궤적·임피던스 |
| **③** | [src/mpc_qp.py](src/mpc_qp.py) | 345 | **최적화** — 제약 행렬, condensed QP, `solve_gait` |
| **④** | [src/09_walk.py](src/09_walk.py) | 599 | **조립** — 위 셋을 엮고 토크로 내보내는 본체 |
| ⑤ | [src/g1_model.py](src/g1_model.py) | 114 | 모델 로딩, 토크 액추에이터 변환, crouch 자세 |

### 각 파일에서 볼 함수 (읽는 순서)

**① `mpc_srb.py`** — 가장 짧고 개념이 선명하다. 여기부터.
- `SRBParams`, `make_params()` — 질량·관성·발 형상을 **모델에서 추출** (하드코딩 없음)
- `get_state()` — MuJoCo → x(13). ω 를 전신 각운동량으로 뽑는 이유가 주석에
- `continuous_AB()` — **이 4줄이 SRB 동역학 전부**
- `discretize()` — A 가 멱영이라 정확한 ZOH 가 닫힌 형태

**② `gait.py`** — "언제/어디에 딛나"
- `Gait` — 시간 → stance/swing (고정 타이밍). `contact_table()` 이 MPC 로 감
- `raibert_target()` — **착지점 공식**. capture point 항이 여기 (Q&A 9/15 Q4, Q8)
- `SwingController.target()` — 스윙 궤적 (z 2단, `soft_land`)
- `SwingController.wrench()` — 작업공간 임피던스 → (F, moment)

**③ `mpc_qp.py`** — "무엇이 물리적으로 가능한가"
- `Q_DEFAULT`, `R_DEFAULT` — 가중치 (주석에 튜닝 이력)
- `foot_constraints()` — **제약 18행**. 마찰 4 + Fz 2 + CoP 4 + Caron mz 8
- `WrenchMPC.solve_gait()` — A_qp/B_qp 조립 → H, g → quadprog

**④ `09_walk.py`** — 여기가 제일 길지만 구조는 단순
- `WalkController.__init__` — 플래그가 다 모여 있음 (실험 스위치)
- `update_mpc()` (100 Hz) — 착지점 갱신 → 참조 조립 → QP 호출
- `torque()` (500 Hz) — **Jᵀ 사상** (Q&A 9/15 Q1 이 이 함수 해설)
- `headless()` / `view()` — 러너

---

## 2. 읽기 전에 볼 문서 (코드보다 먼저)

| 문서 | 왜 |
|---|---|
| [Q&A/2026-09-15.md](../Q&A/2026-09-15.md) **Q1** | `τ = qfrc_bias − JᵀW` 유도. **이거 먼저 읽으면 ④가 쉬워진다** |
| [Q&A/2026-09-15.md](../Q&A/2026-09-15.md) **Q4** | capture point vs Raibert — ②의 `raibert_target` 해설 |
| [MPC_NOTES.md](MPC_NOTES.md) 1~5절 | 파일 맵, 모델 요약, 제약, QP 구조 |
| [PHYSICS_MAP.md](PHYSICS_MAP.md) | 어떤 물리를 어떻게 근사했나 (코드의 '왜') |

---

## 3. 검증 스크립트 (개념 확인용 — 하나씩 돌려보면 이해가 빠르다)

역사적 순서 = 난이도 순서다. 각각 독립 실행 가능.

| 파일 | 무엇을 확인하나 |
|---|---|
| `01_view.py` | 모델 띄우기 |
| `02_inspect.py` | 관절·링크·질량 덤프 |
| **`03_grf_to_tau.py`** | **GRF → 토크 사상 검증.** Q&A Q1 을 눈으로 확인하는 스크립트 |
| `04_stand.py` | MPC 없이 서 있기 |
| **`05_sign_check.py`** | **부호 검증** — 우리가 제일 많이 틀렸던 것 |
| `06_standing_qp.py` | 1스텝 QP (N=1) |
| `07_b_autodiff_check.py` | B 행렬 자동미분 대조 |
| `08_standing_mpc.py` | 지평 있는 MPC로 서 있기 |
| **`10_constraint_check.py`** | **제약 18행이 옳은지 LP 로 검증** (Caron mz 포함) |

**추천**: `03` → `05` → `10` 셋만 돌려봐도 컨트롤러의 핵심 세 축(사상·부호·제약)이 잡힌다.

---

## 4. 나머지

| 파일 | 역할 |
|---|---|
| `11_walk_srb_upper.py` | 상체 SRB 실험 (부정적 결과, MPC_NOTES 21절). **훅 2개만 재정의**한 구조라 ①④ 이해에 도움 |
| `mpc_log.py` / `plot_log.py` | 로그 저장·플롯 |
| `paths.py` | 경로 상수 |
| `reference/` | ispaik06 C++ 구현 (대조군). [REFERENCE_ANALYSIS.md](../REFERENCE_ANALYSIS.md) 참고 |

---

## 5. 현재 권장 실행 구성

```bash
.venv/Scripts/python.exe MPC/src/09_walk.py --view --vx 0.5 \
    --uppd 300 --swingid --softland --lamswing --wn 30 --zeta 0.7 \
    --sf 0.57 --qpy 300 --swingyaw
```

플래그가 많은데, **전부 실험으로 얻은 것**이고 각각 코드 주석에 근거가 달려 있다:

| 플래그 | 무엇 | 근거 |
|---|---|---|
| `--uppd 300` | 상체 PD 강화 | Step5 디버깅 |
| `--swingid` | 스윙 역동역학 보상 | 검증 2라운드 §5 |
| `--softland` | 착지 수직속도 0 | MPC_NOTES 23절 |
| `--lamswing --wn 30 --zeta 0.7` | 스윙 kp = ω_n²Λ(q) | 23절 |
| `--sf 0.57` | 스윙 0.2→0.34 s | **24절 (속도 벽 돌파)** |
| `--qpy 300` | 횡 위치 가중치 | **24절** |
| `--swingyaw` | 스윙 발 yaw 정렬 상시 | Q&A 9/16 Q2 |

> 기본값이 아직 옛날 값인 것들이 있다 (`soft_land=False` 등) — 플래그 없이 돌리면
> 9/14 이전 컨트롤러가 나온다. 비교 실험용으로 일부러 남겨둔 것.
