# MIT Humanoid — 우리 복사본 (2026-09-25)

원본: `../../../mit_humanoid_mjcf/` (클로드 챗에서 URDF → MJCF 변환, MuJoCo 3.14 importer).
URDF 출처: hojae-io/LearningHumanoidArmMotion-RAL2025-Code `resources/mit_humanoid/urdf/humanoid_full_sf.urdf` (commit d176a14, MIT License — `LICENSE_source`).

**변환 검증 (9/25)**: 원본 MJCF ↔ MuJoCo URDF importer ↔ URDF 원문 숫자를 21 바디·18 관절·32 geom 전부 대조 —
질량·CoM·관절 축/범위/위치 차이 0, 관성·자세 최대 4e-7 / 9e-7 (쿼터니언 소수 6 자리 반올림). **변환은 정확하다.**
총 24.889 kg, 다리 42.8 %, MPC 가 쓰는 "축소 몸체" (몸통 + 팔) 14.227 kg.

## 원본 대비 바꾼 것 (`mit_humanoid.xml`)

| 무엇 | 값 | 근거 |
|---|---|---|
| timestep | 0.001 → **0.002** | 컨트롤러 틱 = 물리 틱 2 ms, MPC 는 5 틱마다 (100 Hz) |
| 발 site | `left/right_foot_contact_site` at `(0.03, 0, -0.04)` (발 좌표계, 회전 없음) | 컨트롤러가 쓰는 발 접촉점. 발 원기둥 (r 0.01, 반길이 0.075, 중심 (0.03,0,-0.03)) 의 **발바닥 선 중앙** — 선 접촉 반길이 l = 0.075 의 원점 |
| armature | URDF `rotor_inertia` (hip_yaw/abad 0.01188, hip_pitch 0.0198, knee 0.0792, ankle 0.04752, 어깨 0.01188, 팔꿈치 0.0304) | 모터 반사 관성 (URDF 에 값이 있지만 MJCF 변환에서 빠졌던 것). 민감도는 로더의 `armature=False` 로 확인 |
| keyframe `stand` | hip_pitch −0.467747, knee 0.996975, ankle −0.529228, base z **0.679472** | 발 평평 + 발 site 가 전신 CoM 바로 아래 + base z = yaml `base_position_W` 0.679472 로 역산. yaml 관절값 `[0,0,-0.735,1.2,-0.70]` 을 그대로 넣으면 발가락이 13.5° 들리고 CoM 이 뒤꿈치 끝에 있어 관절 PD 만으로 1 s 안에 넘어진다 |
| keyframe `init_yaml` | yaml `leg_joint_offsets` 관절값 그대로, base z 0.657128 (발 닿는 높이) | 비교용 |
| **몸통 (base) 좌우 대칭화 (10/9)** | CoM y 0.004771 → **0**, ixy 0.001419 → **0**, iyz −0.001672 → **0** (MJCF 는 `fullinertia` 로). 질량 8.52 · ixx 0.172699 · iyy 0.105949 · izz 0.091906 · ixz 0.004023 그대로 | 좌우 대칭 몸통이면 0 이어야 하는 성분. 원래 값 때문에 관성 주축이 roll −6.9° 돌아 있었다 (뷰어 관성 박스가 기울어 보임). 사용자 결정 (Q&A 10/6 Q38 · Q40 · Q42) |
| **왼 hip_yaw ixz 부호 (10/9)** | URDF 313 줄 `ixz="0.0005578"` → **`-0.0005578`** (오른 hip_yaw 81 줄과 같게). MJCF quat → 오른쪽의 거울상 (w, −x, y, −z), 대각 같음 | URDF 의 좌우 거울상 오류 — 다른 9 쌍은 모두 정확한 거울상이고 이것만 ixz 부호가 반대였다. 링크가 뒤 · 아래로 뻗으므로 ixz < 0 (오른쪽) 이 맞는 값으로 봄 (Q&A 10/6 Q40 · Q42) |

`scene.xml` 은 원본 그대로 (바닥 마찰 0.8, 로봇 geom 1.0 → MuJoCo 쌍 규칙상 실효 1.0 = MPC μ 1.0).
이름은 원본 그대로 둔다 (`base`, `a01_right_hip_yaw` …).

**10/9 주의**: `urdf_source/humanoid_full_sf.urdf` 도 위 두 줄 (몸통 · 왼 hip_yaw) 을 고친 판이다 (원본과 다름). 원래 값은 표와 MJCF 주석에 있고,
고치기 전 파일은 git 이력 (커밋 593a27a 이전) 과 원 저장소 (hojae-io/LearningHumanoidArmMotion-RAL2025-Code, d176a14) 에 있다.
MIT License 는 수정 · 재배포를 허용한다 (저작권 · 허가 문구는 `LICENSE_source` 에 그대로). 처음 변환본 `mit_humanoid_mjcf/` (저장소 루트) 는 기록용으로 고치지 않았다.
