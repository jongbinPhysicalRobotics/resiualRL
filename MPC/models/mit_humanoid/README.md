# MIT Humanoid — 우리 복사본 (2026-09-25)

원본: `../../../mit_humanoid_mjcf/` (클로드 챗에서 URDF → MJCF 변환, MuJoCo 3.14 importer).
URDF 출처: hojae-io/LearningHumanoidArmMotion-RAL2025-Code `resources/mit_humanoid/urdf/humanoid_full_sf.urdf` (commit d176a14, MIT License — `LICENSE_source`).

**변환 검증 (9/25)**: 원본 MJCF ↔ MuJoCo URDF importer ↔ URDF 원문 숫자를 21 바디·18 관절·32 geom 전부 대조 —
질량·CoM·관절 축/범위/위치 차이 0, 관성·자세 최대 4e-7 / 9e-7 (쿼터니언 소수 6 자리 반올림). **변환은 정확하다.**
총 24.889 kg, 다리 42.8 %, reference 가 쓰는 "축소 몸체" (몸통 + 팔) 14.227 kg.

## 원본 대비 바꾼 것 (`mit_humanoid.xml`)

| 무엇 | 값 | 근거 |
|---|---|---|
| timestep | 0.001 → **0.002** | reference `config/simulation.yaml` (컨트롤러 틱 = 물리 틱, MPC 7 틱마다) |
| 발 site | `left/right_foot_contact_site` at `(0.03, 0, -0.04)` (발 좌표계, 회전 없음) | reference `MitHumanoidSpec.cpp` 가 쓰는 이름. 발 원기둥 (r 0.01, 반길이 0.075, 중심 (0.03,0,-0.03)) 의 **발바닥 선 중앙** — CoP 한계 ±0.065 가 선 안 (±0.075) 에 들어온다 |
| armature | URDF `rotor_inertia` (hip_yaw/abad 0.01188, hip_pitch 0.0198, knee 0.0792, ankle 0.04752, 어깨 0.01188, 팔꿈치 0.0304) | 모터 반사 관성 (URDF 에 값이 있지만 MJCF 변환에서 빠졌던 것). reference 의 MIT MJCF 는 비공개라 그쪽 값은 모름 — 민감도는 로더의 `armature=False` 로 확인 |
| keyframe `stand` | hip_pitch −0.467747, knee 0.996975, ankle −0.529228, base z **0.679472** | 발 평평 + 발 site 가 전신 CoM 바로 아래 + base z = reference yaml `base_position_W` 0.679472 로 역산. reference 의 yaml 관절값 `[0,0,-0.735,1.2,-0.70]` 을 이 모델에 그대로 넣으면 발가락이 13.5° 들리고 CoM 이 뒤꿈치 끝에 있어 관절 PD 만으로 1 s 안에 넘어진다 — reference 의 (비공개) 모델은 관절 0 점이 다른 것으로 본다 |
| keyframe `init_yaml` | reference yaml 관절값 그대로, base z 0.657128 (발 닿는 높이) | 비교용 |

`scene.xml` 은 원본 그대로 (바닥 마찰 0.8, 로봇 geom 1.0 → MuJoCo 쌍 규칙상 실효 1.0 = reference MPC μ 1.0).
이름은 원본 그대로 둔다 (`base`, `a01_right_hip_yaw` …) — reference 이름과의 대응표는 `MPC/src/mit/mit_model.py`.
