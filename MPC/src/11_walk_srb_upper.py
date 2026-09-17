"""SRB '상체' 실험 — v2: 회전만 상체로 (hybrid reduced-body SRB).

═══════════════════════════════════════════════════════════════════════
★ 실험 최종 판정 (2026-09-15, 결과 표는 MPC_NOTES 21절)
═══════════════════════════════════════════════════════════════════════
우리 컨트롤러 구성(부드러운 가중치 + 발디딤 피드백 주도)에서는 **모든 상체
변형이 baseline(전신 L 기반 ω + 전신 I)에 진다**:
  직진 0.3 — baseline 20s ✓ vs 상체 변형 4.3~13.5 s 전도
  회전 1.3 — baseline 완주(−0.8°) vs 상체(angmom_ub) 8.1 s 전도
기전: 다리(43%)의 스윙 반작용 모멘트가 상체-회전 모델에선 전부 '비모델
외란'이 된다. baseline 의 전신 L·전신 I 는 그 반작용을 L-공간에서 암묵적으로
흡수하는 필터였다 — '무거운 모델'이 우연이 아니라 강건성의 근원.
reference 가 상체 SRB 로 성공하는 건 모델 단독이 아니라 패키지(위치 가중치
~840배, Λ-스케줄 고강성 스윙 OSC = 반작용 자체를 축소) 덕분 — 상체 SRB 를
쓰려면 그 전제부터 이식해야 한다는 종속성이 이 실험의 결론.
아래 설계·실패 기록은 그 여정 전체다 (파일은 절단 실험 장비로 보존).

═══════════════════════════════════════════════════════════════════════
v2 설계 — baseline(09_walk) 과의 차이는 단 두 가지
═══════════════════════════════════════════════════════════════════════
  1) I_body ← 상체(다리 서브트리 제외) 합성 관성      [make_srb_params]
  2) ω      ← pelvis 각속도 (전신 각운동량 기반 대신)  [get_state]
  병진(p, v = 전신 CoM, 질량 = 전신, 중력 = g)은 baseline 그대로.

왜 이 분해인가 — 뉴턴/오일러가 갈라지는 지점:
  병진:  M_tot·v̇_CoM = ΣF − M_tot·g   ← 다리 포함 '정확' (근사가 아예 없음)
  회전:  I·ω̇ = Σ r×F + m              ← 여기가 massless-leg 근사의 자리
  즉 "다리를 모델에서 뺀다"는 회전 방정식에만 의미가 있다. 병진까지 상체로
  바꾸면 정확했던 식을 일부러 틀리게 만드는 것 (아래 v1 실패 기록).

이 두 줄이 고치는 구조 문제 (검증 리포트 §4, MPC_NOTES 20절):
  - Θ(pelvis)·ω(전신 L) 가 '같은 강체'가 아니던 불일치 → 둘 다 pelvis 로 정합
  - 전신 합성 관성(I_zz≈0.54)에 다리 배치가 섞여 있던 유령 관성
    → 상체 관성(I_zz≈0.28)으로: MPC 가 '실제로 돌릴 몸통'만 돌린다.
    다리의 회전 반작용은 모델 밖 → swing 보상(--swingid)의 몫 (명시적 분업).

═══════════════════════════════════════════════════════════════════════
v1 실패 기록 — 병진까지 상체로 했을 때 배운 것 (공부 포인트, 순서대로)
═══════════════════════════════════════════════════════════════════════
v1 설계: 질량 18.97 kg(상체), p·v = 상체 CoM (subtree 빼기 항등식:
  M_ub·p_ub = M_tot·subtree_com[0] − Σ_다리 M_leg·subtree_com[hip]), ω = pelvis.

실패 1 — 골반이 0.76→0.58 m 로 주저앉음 (제자리 스텝은 생존):
  가설 1: swing 다리(7.2 kg)가 골반에 '매달린' 무게가 모델 밖
    → g_eff = g·(1 + 0.5·m_leg/M_ub) 보정 → 0.64 m, 절반만 회복.
  로그 진단: 명령 ΣFz 가 319 N(≈전신 무게!)으로 수렴. 진범은 스탠스 사상 —
    τ = qfrc_bias − JᵀW 는 정역학에서 '접촉력 = W' 를 정확히 실현한다
    (Step 1 검증 성질). 접촉력이 W 로 고정이면 stance 다리 무게도 구조가
    아니라 W 가 진다 → 유효 중력은 g·M_tot/M_ub (스윙만이 아니라 다리 전체).
    이 보정으로 높이 완벽 유지 (0.763→0.774), 제자리 12 s 통과.
  (참고: reference 는 stance 에 bias 를 안 넣어 접촉력이 W+다리무게로
   자기증강 → 그쪽 W 는 상체 몫만 짐. bias 유무가 두 설계의 갈림길.)

실패 2 — 전진 램프에서 즉시 전도 (0.3: 4.4 s, vx +0.21→−0.53 요동):
  B 의 수평 응답을 1/M_ub(=1/18.97)로 모델 → 실제로는 LIP 시간스케일에서
  다리도 함께 가속하므로 1/M_tot 에 가깝다. 모델이 1.76배 과민 → 속도 루프
  게인 뻥튀기 → 램프 과도에서 발산. 제자리(수평 과도 없음)만 생존한 이유.
  → 결론: 병진은 전신으로 되돌린다 = v2. (v1 의 g_eff 보정도 병진을 전신으로
    되돌리면 자연 소멸 — M_tot 질량에 중력 g 가 원래 맞다.)

═══════════════════════════════════════════════════════════════════════
구현 노트
═══════════════════════════════════════════════════════════════════════
- 상체 관성: 다리 서브트리(hip_pitch 이하)에 속하지 않는 body 들을 상체
  CoM 기준 평행축 합성, yaw₀ 제거해 body frame 저장 (make_params 와 동일 절차).
  팔이 관절 PD 로 고정이라 사실상 상수 → init 1회 계산.
  근사 주의: 오일러 식의 모멘트 팔 r = 발 − 전신CoM(상태 p)인데 I 는 상체
  CoM 기준 — SRB 수준에서 표준적인 절충 (Di Carlo 도 body 관성 + 모델 CoM 팔).
- ω = pelvis: mpc_srb.get_state 의 I_body=None 분기가 정확히 이것 (전신판이
  각운동량 ω 를 쓴 이유였던 '다리 서스펜션 진동'은 다리가 회전 모델 밖으로
  나가면서 상태에 안 들어온다 — 대신 pelvis 강체와 Θ 가 정합).
- 훅 구조: 09_walk.WalkController 의 make_srb_params / get_state 두 메서드만
  재정의. 게이트·raibert·swing·yaw hold·언랩·토크 사상·러너 전부 상속 —
  이 파일과 09_walk 의 diff 가 곧 개념 차이다.

돌리기:
  .venv/Scripts/python.exe MPC/src/11_walk_srb_upper.py --vx 0.3 --uppd 300 --swingid
  옵션은 09_walk 와 동일 (--vx --wz --seconds --uppd --swingid --noyawhold --view)
로그: logs/walk_upper_* (baseline 은 logs/walk_*)
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
import mujoco

import mpc_srb

walk = importlib.import_module("09_walk")   # 숫자로 시작하는 모듈명은 importlib 로


def leg_root_body_ids(m) -> list[int]:
    """다리 서브트리의 루트 body(hip_pitch link) id 2개. G1 은 hip_pitch 가
    다리 최상단 링크라 이 서브트리가 '다리 전체'와 일치한다."""
    return [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_hip_pitch_link")
            for s in ("left", "right")]


def upper_body_inertia(m, d) -> tuple[np.ndarray, float, np.ndarray]:
    """상체(다리 제외) 합성 관성 (body frame, yaw₀ 제거), 질량, CoM."""
    legs = leg_root_body_ids(m)
    in_leg = np.zeros(m.nbody, dtype=bool)
    for b in range(1, m.nbody):
        p = b
        while p != 0:
            if p in legs:
                in_leg[b] = True
                break
            p = m.body_parentid[p]
    ub = [b for b in range(1, m.nbody) if not in_leg[b]]

    mass = float(sum(m.body_mass[b] for b in ub))
    com = sum(m.body_mass[b] * d.xipos[b] for b in ub) / mass
    inertia = np.zeros((3, 3))                # 평행축 합성 (make_params 동일)
    for b in ub:
        R = d.ximat[b].reshape(3, 3)
        I_b = R @ np.diag(m.body_inertia[b]) @ R.T
        r = d.xipos[b] - com
        inertia += I_b + m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
    yaw0 = mpc_srb.quat_to_euler_zyx(d.qpos[3:7])[2]
    R0 = mpc_srb.rz(yaw0)
    return R0.T @ inertia @ R0, mass, com


class UpperSRBController(walk.WalkController):
    """v2: 회전만 상체 — 훅 2개 재정의 (파일 상단 docstring 참고).

    절단(ablation) 플래그 — I 와 ω 중 무엇이 성능을 바꾸는지 분리:
      i_src: 'upper'(상체 관성) | 'full'(전신 관성)
      w_src: 'pelvis'(pelvis 각속도) | 'angmom'(전신 각운동량 ω — baseline 방식)
    v2 기본 = upper + pelvis. baseline = full + angmom.
    """
    I_SRC = "upper"
    W_SRC = "pelvis"

    # 훅 1: 관성만 상체로 교체 (질량·발 형상·마찰은 전신판 그대로)
    def make_srb_params(self, m, d):
        base = mpc_srb.make_params(m, d)
        self._I_full = base.I_body.copy()          # angmom ω 추출용 (전신 관성)
        I_ub, m_ub, com_ub = upper_body_inertia(m, d)
        if self.I_SRC == "upper":
            base.I_body = I_ub
        print(f"  [upper-SRB v2] I={self.I_SRC}, ω={self.W_SRC}:"
              f" I_zz {self._I_full[2, 2]:.3f} → {base.I_body[2, 2]:.3f} kg·m²"
              f" (I_xx {self._I_full[0, 0]:.3f} → {base.I_body[0, 0]:.3f}),"
              f" 상체 {m_ub:.2f}/{base.mass:.2f} kg — 병진은 전신 유지")
        return base

    # 훅 2: ω 소스 선택.
    #   pelvis    = pelvis 각속도 (I_body=None 분기가 정확히 그 정의)
    #   angmom    = 전신 L 기반 (baseline 방식 — I_full 로 추출해야 자기일관)
    #   angmom_ub = ★ '상체의 L 로 상체의 ω': ω = I_ub⁻¹·L_ub.
    #       L 기반이라 pelvis 서스펜션 진동이 걸러지고(baseline 이 살아남은
    #       이유), 대상이 상체라 Θ(pelvis)·모델 I(상체)와도 정합 — 절단 실험
    #       에서 '일관 쌍'만 버틴 관찰의 논리적 종점.
    #       L_ub = L_tot − Σ_다리 [L_leg + m_leg·(c_leg − C)×v_leg]
    #       (subtree_angmom 은 각 서브트리 자기 CoM 기준 → 전신 CoM C 로
    #        모멘텀 전달항 m·(c−C)×v 를 더해 옮긴 뒤 뺀다)
    def get_state(self, d, t=None):
        if self.W_SRC == "pelvis":
            return mpc_srb.get_state(self.m, d)
        if self.W_SRC == "angmom":
            return mpc_srb.get_state(self.m, d, I_body=self._I_full)

        x = mpc_srb.get_state(self.m, d)          # Θ, p, v 채우기 (ω만 교체)
        mujoco.mj_subtreeVel(self.m, d)
        C = d.subtree_com[0]
        L = d.subtree_angmom[0].copy()
        for b in leg_root_body_ids(self.m):
            m_leg = float(self.m.body_subtreemass[b])
            r = d.subtree_com[b] - C
            L -= d.subtree_angmom[b] + m_leg * np.cross(r, d.subtree_linvel[b])
        Rz_ = mpc_srb.rz(x[2])
        I_w = Rz_ @ self.params.I_body @ Rz_.T    # 모델 관성(상체)과 같은 것
        x[6:9] = np.linalg.solve(I_w, L)
        return x


if __name__ == "__main__":
    vx = float(sys.argv[sys.argv.index("--vx") + 1]) if "--vx" in sys.argv else 0.0
    secs = float(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else 12.0
    kpu = float(sys.argv[sys.argv.index("--uppd") + 1]) if "--uppd" in sys.argv else 60.0
    wzc = float(sys.argv[sys.argv.index("--wz") + 1]) if "--wz" in sys.argv else 0.0
    sid = "--swingid" in sys.argv
    yh = "--noyawhold" not in sys.argv
    if "--isrc" in sys.argv:
        UpperSRBController.I_SRC = sys.argv[sys.argv.index("--isrc") + 1]
    if "--wsrc" in sys.argv:
        UpperSRBController.W_SRC = sys.argv[sys.argv.index("--wsrc") + 1]
    if "--view" in sys.argv:
        walk.view(vx, kp_up=kpu, swing_id=sid, wz=wzc, yaw_hold=yh,
                  ctor=UpperSRBController)
    else:
        ok = walk.headless(vx=vx, seconds=secs, kp_up=kpu, swing_id=sid,
                           wz=wzc, yaw_hold=yh,
                           ctor=UpperSRBController, variant="upper")
        print()
        print("상체 SRB(v2) " + ("통과 ✓" if ok else "실패"))
