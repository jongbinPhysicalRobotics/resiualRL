"""다리 각운동량 — 실측(MuJoCo) 과 예측(계획된 스윙에서 질점 근사).  9/23 Q8 아핀항의 재료.

왜 필요한가 (Q&A 9/16 Q8, 9/22 Q9, 9/23 Q4~Q8):
  전신 ω 로 SRB 를 쓰면 Θ̇ = Rᵀω 가 틀린다 — 다리가 앞뒤로 휘둘리면 상체가 반대로 돌아
  전신 L 은 작은데 골반은 크게 흔들린다 (상관 0.14).  맞는 식은
      ω_골반 ≈ I_상체⁻¹ (L_전신 − L_다리)
  이고 L_다리(t) 는 스윙 궤적이 지평 전체에 걸쳐 계획돼 있으니 미리 계산할 수 있다.

정의 (전부 전신 CoM C 기준, world):
  L_다리 = Σ_다리 [ angmom_sub + m_sub (c_sub − C) × (v_sub − v_C) ]      ← 상대속도 (기본)
  절대속도판 (v_sub) 은 상체 궤도항 m_ub (c_ub − C) × v_C 가 전진 속도에 비례하는 상수 오프셋으로
  남아 Θ̇ 에 가짜 일정 피치율을 만든다.  --check 가 두 정의의 상관을 같이 찍는다.
"""
from __future__ import annotations

import numpy as np
import mujoco

import mpc_srb


# ---- 11_walk_srb_upper.py 에서 옮겨 온 두 함수 (affine 폴더가 11 없이 돌도록, 9/24) ----
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


def subtree_bodies(m, root: int) -> list[int]:
    out = []
    for b in range(1, m.nbody):
        p = b
        while p != 0:
            if p == root:
                out.append(b)
                break
            p = m.body_parentid[p]
    return out


def measured_leg_L(m, d, relative: bool = True) -> np.ndarray:
    """두 다리 각운동량 (전신 CoM 기준, world). mj_subtreeVel 을 안에서 부른다."""
    mujoco.mj_subtreeVel(m, d)
    C = d.subtree_com[0]
    vC = d.subtree_linvel[0] if relative else np.zeros(3)
    L = np.zeros(3)
    for b in leg_root_body_ids(m):
        m_leg = float(m.body_subtreemass[b])
        r = d.subtree_com[b] - C
        L += d.subtree_angmom[b] + m_leg * np.cross(r, d.subtree_linvel[b] - vC)
    return L


def pelvis_omega_world(m, d) -> np.ndarray:
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return d.xmat[pel].reshape(3, 3) @ d.qvel[3:6]


class LegPointModel:
    """다리 하나를 '엉덩이→발' 직선 위의 질점들로 근사.

    각 링크의 질량을 그대로 두고, 링크 CoM 을 엉덩이→발 축에 투영한 비율 β 만 기억한다
    (초기 crouch 자세에서 1 회).  예측 때는 엉덩이·발의 위치/속도만 있으면
        r_j = p_hip + β_j (p_foot − p_hip),   v_j = v_hip + β_j (v_foot − v_hip)
    로 링크 위치·속도를 만들고  L = Σ m_j (r_j − C) × (v_j − v_C).
    무릎이 굽으면 실제 CoM 은 직선에서 벗어나지만, 스윙 반작용의 크기·부호·타이밍은
    발 속도가 지배한다 — 얼마나 맞는지는 23_walk_affine.py --check 가 잰다.
    """

    def __init__(self, m, d, leg: int):
        self.root = leg_root_body_ids(m)[leg]
        self.bodies = subtree_bodies(m, self.root)
        self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, mpc_srb.FOOT_SITES[leg])
        p_hip, p_foot = d.xpos[self.root].copy(), d.site_xpos[self.sid].copy()
        axis = p_foot - p_hip
        Ln = float(np.linalg.norm(axis))
        u = axis / Ln
        self.mass = np.array([m.body_mass[b] for b in self.bodies])
        self.beta = np.array([float((d.xipos[b] - p_hip) @ u) / Ln for b in self.bodies])
        self.m_total = float(self.mass.sum())

    def hip_now(self, d) -> np.ndarray:
        return d.xpos[self.root].copy()

    def L(self, C, vC, p_hip, v_hip, p_foot, v_foot) -> np.ndarray:
        dp, dv = p_foot - p_hip, v_foot - v_hip
        r = p_hip[None, :] + self.beta[:, None] * dp[None, :] - C[None, :]
        v = v_hip[None, :] + self.beta[:, None] * dv[None, :] - vC[None, :]
        return (self.mass[:, None] * np.cross(r, v)).sum(0)


def predict_leg_L_now(models: list[LegPointModel], m, d) -> np.ndarray:
    """질점 모델을 '지금 실제' 엉덩이·발 위치/속도로 평가 (예측식 자체의 정확도 검사용)."""
    mujoco.mj_subtreeVel(m, d)
    C, vC = d.subtree_com[0], d.subtree_linvel[0]
    L = np.zeros(3)
    for pm in models:
        jp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, None, pm.sid)
        jh = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jh, None, pm.root)
        L += pm.L(C, vC, d.xpos[pm.root], jh @ d.qvel, d.site_xpos[pm.sid], jp @ d.qvel)
    return L
