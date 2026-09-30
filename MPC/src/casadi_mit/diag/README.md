# diag — 디버깅용 진단 스크립트 (9/28 ~ 9/30 에 쓴 것을 모음)

모두 시뮬을 바꾸지 않고 **기록만** 한다 (tick → write_torque → mj_step 순서는 10_walk.headless 와 같다).
변형 문자열 (`--var` 와 같은 `k=v,k=v`) 을 그대로 받는다. 실행은 저장소 루트에서 `GAIT_ON=1 .venv/bin/python MPC/src/casadi_mit/diag/<파일> …`.
어떤 상황에서 무엇을 썼는지는 [../../../DEBUG_JOURNAL.md](../../../DEBUG_JOURNAL.md) §3 에 있다.

| 파일 | 하는 일 | 사용 예 | 처음 쓴 곳 |
|---|---|---|---|
| `wt_diag.py` | 매 틱 몸 기준 앞 · 옆 속도, yaw − 목표, roll · pitch, 발 옆 위치, 착지 목표, 경로 옆 오차 기록 (.npz) | `wt_diag.py 0.8 0 40 <var> out.npz` | 9/29 Q7 (게걸음), 9/29 Q11 (비대칭) |
| `wt_force.py` | 위 + MPC 가 명령한 두 발 힘 · 모멘트 (몸 기준), ADMM 잔차 | `wt_force.py 0.6 0.5 20 <var> out.npz` | 9/29 Q8 (전도 해부) |
| `push_diag.py` | 옆 밀기 뒤 회복 과정 (발 옆 위치 · 목표 · 옆 힘 · 수직력) | `push_diag.py 0.6 0 -20 10 0.2 14 <var> out.npz` | 9/29 Q12 (발 꼬임) |
| `iter_diag.py` + `iter_ana.py` | 걷는 중 QP 를 저장해 두고, 반복 수별 해를 정확해와 비교 (u0 오차 · 제약 위반 · 목적함수 차) | `iter_diag.py 0.6 0.5 8 <var> qp.pkl` → `iter_ana.py qp.pkl` | 9/29 Q8 |
| `timing.py` | 풀이 1 회 시간 (조립 · 풀이 · p99 · 반복 수) 을 단독으로 잰다 | `timing.py 0.6 0.5 10 <var1> <var2> …` | 9/29 Q10 |
| `tab.py` | 11_gait_quality JSON → 한 줄 요약 표 | `tab.py out.json "<공통 접두어>"` | — |
| `push_tab.py` | 외란 격자 JSON → 방향 × 크기 표 (위상 여러 개면 성공 수) | `push_tab.py push.json` | 9/29 Q11 |
| `gait_tab.py` | 보행 주기 격자 JSON → 주기 · 스윙 · k_half 별 표 | `gait_tab.py gait.json` | 9/30 Q3 |
| `line_diag.py` + `line_ana.py` | 디딤 발 MPC 명령 wrench 와 실제 MuJoCo 접촉 wrench 를 발 좌표로 기록 → CoP 끝 비율 · 옆 힘 · 두 점 (선 접촉) 비틀림 제약 넘침 요약 | `line_diag.py 1.0 0 0 15 <var> out.npz` → `line_ana.py out.npz --l 0.075` | 9/30 Q11 |
| `lat_tab.py` | 옆 외란 선별 JSON → 변형 × 방향 성공 수 (접두어 P 는 파일 안에서 고쳐 쓴다) | `lat_tab.py lat.json lat_n.json` | 9/29 Q12 |

저장소 안에 원래 있던 진단 도구: `11_gait_quality.py` (병렬 A/B 러너 · 걸음 품질 표),
`12_force_report.py` (관절 토크 · wrench 요소별), `13_foot_diag.py` (뒤꿈치 회전 · CoP 포화 · 한 점 접촉 · 입각 구간별 프로파일).
