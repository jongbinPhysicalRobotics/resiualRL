# MPC/src — 폴더 안내 (2026-09-24 정리)

컨트롤러 갈래마다 폴더를 나눴다. **각 폴더는 자기 안의 파일만으로 실행된다** (numpy · mujoco 같은 라이브러리 제외).
공용 모듈(`09_walk`, `gait`, `mpc_qp`, `mpc_srb` …)은 폴더마다 **복사본**이 있다.

| 폴더 | 무엇 | 주 실행 파일 | 상태 |
|---|---|---|---|
| [baseline/](baseline/) | 기존 컨트롤러 + 지금까지의 모든 검증·분석 스크립트 (01 ~ 22, 25) | `09_walk.py` | 현재 권장 구성 |
| [affine/](affine/) | **아핀항 MPC** — 다리 각운동량을 MPC 에 알려 줌 (Q&A 9/23 Q11, 9/24 Q1·Q30·Q32). 현재 방향: 세 축 모두 전신 ω + `--refshape`. 회전 명령엔 `--swayrot --turnfix` (Q34) | `23_walk_affine.py` | 채택 후보 (렌더 확인 대기) |
| [split/](split/) | MPC·스윙 분리 — reference 식 스윙 + 상체 SRB (Q&A 9/23 Q11, 9/24 Q2) | `24_walk_split.py` | 보류 |

```bash
F="--uppd 300 --swingid --softland --lamswing --wn 30 --zeta 0.7 --sf 0.57 --qpy 300 --swingyaw --sidew 13 --tdscale 1.25 --copm 0.9 --wzpel 1 --wxpel 0.5"
.venv/Scripts/python.exe MPC/src/baseline/09_walk.py       --view --vx 0.5 $F     # 기존
.venv/Scripts/python.exe MPC/src/affine/23_walk_affine.py  --view --vx 0.5 $F     # 아핀항
.venv/Scripts/python.exe MPC/src/split/24_walk_split.py    --view --vx 0.5 $F     # 분리
# 걸음 품질 비교표는 비교하려는 컨트롤러가 있는 폴더의 18 번으로 (그 폴더엔 기준 09_walk 도 있다)
.venv/Scripts/python.exe MPC/src/affine/18_gait_quality.py --vx 0.5 0.7 --var "tds=1.25,copm=0.9,wzp=1,wxp=0.5" "tds=1.25,copm=0.9,wzp=1,wxp=0.5,ctl=affine"
```

**주의 — 복사본이라 한 폴더를 고쳐도 다른 폴더엔 안 들어간다.** 예: `gait.py` 의 `GAIT_ON` (보행 켜기/끄기, Q&A 9/24 Q8) 은
세 폴더에 따로 있다. 지금 값은 셋 다 `0` (서 있기) — 걷게 하려면 쓰는 폴더의 `gait.py` 를 `1` 로, 또는 파일은 두고 실행마다 `--nogait` 로 끈다.

| 파일 | baseline | affine | split |
|---|---|---|---|
| `09_walk.py` `gait.py` `mpc_qp.py` `mpc_srb.py` `g1_model.py` `mpc_log.py` `paths.py` `viewer_hud.py` | ○ 원본 | 복사 | 복사 |
| `17_knee_geometry.py` `18_gait_quality.py` (판정표) | ○ 원본 | 복사 | 복사 |
| `11_walk_srb_upper.py` | ○ 원본 | — | 복사 (분리가 상속) |
| `walk_cli.py` | — | ○ | 복사 |
| `23_walk_affine.py` `leg_momentum.py` | — | ○ | — |
| `26_push_walk.py` (걷는 중 밀기 시험) | — | ○ | — |
| `24_walk_split.py` | — | — | ○ |

로그는 폴더와 무관하게 `MPC/logs/` 에 쌓인다 (`paths.MPC_ROOT` 가 위로 올라가며 `MPC` 폴더를 찾는다).
코드 읽는 순서는 [../CODE_MAP.md](../CODE_MAP.md).
