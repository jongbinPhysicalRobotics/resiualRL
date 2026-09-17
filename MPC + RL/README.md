# MPC + RL — residual 정책 (최종 목표 구조)

MPC 출력에 학습된 보정을 **토크 수준에서 블렌딩**한다: `τ = τ_MPC + λ·τ_residual`

## 문서

- **[RESIDUAL_NOTES.md](RESIDUAL_NOTES.md)** — Residual MPC 논문(Jeon et al., T-RO 2026)
  이식 노트. **residual 단계에 들어갈 때 이 문서부터.**
  리워드 10항 전수 분류 · 블렌딩 3종 호환성 · 관측/학습 설정 · 착수 체크리스트

## 현재 상태

**미착수.** 선행 조건: **1 m/s 를 MPC 단독으로 달성** ([../MPC/](../MPC/)).
residual 은 그 이후의 주제다.

## 핵심 발견 (RESIDUAL_NOTES §2)

논문이 채택한 **joint-torque 블렌딩**(식 23-24)은 `q̂`(기본 자세)를 기준으로 쓰고
`q_MPC` 를 쓰지 않는다 → **우리 SRB MPC 가 관절 궤적을 못 내도 그대로 성립한다.**
