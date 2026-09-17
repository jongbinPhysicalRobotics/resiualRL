# pure RL — 순수 end-to-end 강화학습

**현재 비어 있음.** 향후 E2E RL 베이스라인을 여기에 둔다.

## 목적

발표·논문의 **통제된 대조군**: 같은 리워드 · 같은 환경 · 같은 게인으로
MPC prior 유무만 다르게 비교한다 (MPC / [MPC + RL](../MPC%20+%20RL/) / pure RL).

## 착수 전에 읽을 것

- [MPC + RL/RESIDUAL_NOTES.md](../MPC%20+%20RL/RESIDUAL_NOTES.md) **§1-B**
  — 논문의 최소 리워드를 E2E 에 그대로 쓰면 **발목을 떨며 미끄러지는 정책**이 나온다.
  E2E 에는 foot guidance / air time / contact scheduling 등 shaping 이 추가로 필요하다
- [deployed RL/](../deployed%20RL/) — 이미 학습된 공개 정책. 참고 계측은 여기서

## 환경 주의

**이 PC 에는 NVIDIA GPU 가 없다** (Intel Arc 내장뿐) → IsaacGym/IsaacLab·MJX 불가.
**학습은 랩 GPU 머신(RTX 5090)에서** 진행한다. 이 PC 에서는 코드 작성·분석만.
