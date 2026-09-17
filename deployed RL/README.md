# deployed RL — 공개된 사전학습 정책 (참고용 계측 대상)

학습하지 않고 **이미 잘 걷는 정책을 가져와 계측**하는 곳.

## ★ 결과 → [ANALYSIS.md](ANALYSIS.md)

> **G1 은 같은 다리·발·토크한계로 1.29 m/s 를 120초 간다. 팔은 아예 없어도 된다.**
> 우리 0.5 m/s 는 로봇의 한계가 아니라 **우리 구현의 한계**다.
> 격차: **보폭 27 vs 16 cm**, **주기 800 vs 524 ms**, pelvis pitch 표준편차 3.9 vs 0.36°.

## 구성

| | |
|---|---|
| `unitree_rl_gym/` | [공식 저장소](https://github.com/unitreerobotics/unitree_rl_gym) clone. 정책은 `deploy/pre_train/g1/motion.pt` (12-dof 다리 전용) |
| `analysis/run_g1_policy.py` | 헤드리스 계측 러너 — 속도·pitch·CoP 이용률·지지구간 |
| `analysis/detail.py` | 상세 — gait 타이밍, ω(pelvis) vs ω(전신 L), 착지점 |
| [ANALYSIS.md](ANALYSIS.md) | 전체 측정 결과와 해석 |

## 실행

```bash
.venv/Scripts/python.exe "deployed RL/analysis/run_g1_policy.py" --vx 1.5 --seconds 120
.venv/Scripts/python.exe "deployed RL/analysis/detail.py" 0.5 1.0 1.5
```

원본 뷰어 실행은 `deploy/deploy_mujoco/deploy_mujoco.py g1.yaml` (repo 의 README 참고).

**환경**: `torch`(CPU) + `pyyaml` 필요 — 설치 완료. **GPU 불필요** (추론만).

> ⚠ MuJoCo/torch 가 경로의 한글(`백종빈`)을 못 읽어서, 러너는 XML 디렉터리로 `chdir`
> 하고 정책은 `BytesIO` 로 읽는다. 다른 PC 로 옮기면 이 우회는 불필요하다.

## ⚠ 해석 주의

RL 정책은 MPC 가 못 하는 것을 쓴다 — **접촉 타이밍을 자유롭게 바꾸고**(0.8 s 위상 시계를
무시하고 524 ms 로 딛는다), 관절 수준에서 반응한다.
**"RL 이 X 를 한다" ≠ "고정 스케줄 MPC 도 X 를 할 수 있다"** — **물리적 가능성의 하한**으로 읽을 것.
