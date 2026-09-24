"""SRB(single rigid body) 모델 — 층1 재료 (상태방정식).

상태 x ∈ R13 : [roll, pitch, yaw, px, py, pz, wx, wy, wz, vx, vy, vz, g]
                Θ(3)         p(3, CoM)    ω(3, world) ṗ(3, CoM)   중력상수
입력 u ∈ R12 : [W_L(6), W_R(6)],  W = [Fx, Fy, Fz, mx, my, mz]
                발 site(발목 원점)에 작용하는 wrench (world frame)

연속 동역학 (Di Carlo convex MPC + wrench 확장):
    Θ̇ = Rz(ψ)ᵀ ω                        (작은 roll/pitch 근사, 논문 식 12)
    ṗ = v
    ω̇ = I_w⁻¹ Σ_i (r_i × F_i + m_i)      r_i = 발_i site − CoM (world, CoM 기준!)
    v̇ = (1/M) Σ_i F_i − g ẑ
    ġ = 0
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

NX = 13
NU_PER_FOOT = 6
N_FEET = 2
NU = NU_PER_FOOT * N_FEET
GRAV = 9.81
FOOT_SITES = ("left_foot", "right_foot")

X_LABELS = ["roll", "pitch", "yaw", "px", "py", "pz",
            "wx", "wy", "wz", "vx", "vy", "vz", "g"]
U_LABELS = [f"{s}_{c}" for s in ("L", "R")
            for c in ("Fx", "Fy", "Fz", "mx", "my", "mz")]


def rz(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def skew(v) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def quat_to_euler_zyx(quat) -> tuple[float, float, float]:
    """MuJoCo quat (w,x,y,z) -> (roll, pitch, yaw), R = Rz(yaw)Ry(pitch)Rx(roll)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    R = R.reshape(3, 3)
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


@dataclass
class SRBParams:
    """SRB 파라미터 — 전부 모델에서 추출 (하드코딩 금지)."""
    mass: float
    I_body: np.ndarray        # 3x3, CoM 기준 합성 관성 (body frame — yaw₀ 제거됨)
    l_t: float                # 발가락까지 (발목 기준, +x)
    l_h: float                # 뒤꿈치까지 (발목 기준, -x)
    w: float                  # 좌우 반폭 (보수적: 접촉점 |y| 최솟값 = 0.025)
    h_sole: float             # 발 site 에서 발바닥(접촉면)까지 수직거리 (+0.035)
    mu: float = 0.6
    fz_min: float = 10.0
    fz_max: float = 500.0


def make_params(m, d, mu=0.6, fz_min=10.0, fz_max=500.0) -> SRBParams:
    """현재 자세(base가 수직인 crouch)에서 SRB 파라미터 추출."""
    mass = mujoco.mj_getTotalmass(m)

    # CoM 기준 합성 관성 — world 에서 합성 후 yaw₀ 를 제거해 body frame 으로 저장.
    # (world 값을 그대로 두면 continuous_AB 의 Rz(ψ) 와 이중 회전 — 검증 리포트 §2.
    #  yaw₀=0 인 지금은 동일하지만 요 회전 대비.)
    com = d.subtree_com[0].copy()
    inertia = np.zeros((3, 3))
    for b in range(1, m.nbody):
        R = d.ximat[b].reshape(3, 3)
        I_b = R @ np.diag(m.body_inertia[b]) @ R.T
        r = d.xipos[b] - com
        mb = m.body_mass[b]
        inertia += I_b + mb * (r @ r * np.eye(3) - np.outer(r, r))
    yaw0 = quat_to_euler_zyx(d.qpos[3:7])[2]
    R0 = rz(yaw0)
    inertia = R0.T @ inertia @ R0

    # 발 형상: ankle_roll body 의 접촉 geom local 위치에서 추출
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    geoms = [g for g in range(m.ngeom)
             if m.geom_bodyid[g] == bid and m.geom_group[g] == 3]
    pos = np.array([m.geom_pos[g] for g in geoms])
    l_t = float(pos[:, 0].max())
    l_h = float(-pos[:, 0].min())
    w = float(np.abs(pos[:, 1]).min())
    # site(발목 원점) -> 발바닥 접촉면 거리: 구 중심 z(-0.03) - 반지름(0.005)
    h_sole = float(-(pos[:, 2].min() - m.geom_size[geoms[0], 0]))

    return SRBParams(mass=mass, I_body=inertia, l_t=l_t, l_h=l_h, w=w,
                     h_sole=h_sole, mu=mu, fz_min=fz_min, fz_max=fz_max)


# ---------------------------------------------------------------------------
# 상태 추출
# ---------------------------------------------------------------------------
def get_state(m, d, I_body: np.ndarray | None = None) -> np.ndarray:
    """MuJoCo data -> SRB 상태 x(13). mj_forward 가 이미 불린 상태를 가정.

    I_body(body frame) 를 주면 ω 를 전신 각운동량으로 계산:
        ω = (Rz(ψ) I_body Rz(ψ)ᵀ)⁻¹ L      (SRB 정의 그대로, 현재 yaw 반영)
    ⚠ 안 주면 pelvis 각속도를 쓰는데, 외란 시 '다리 서스펜션 위 골반'의 내부
    진동 모드가 그대로 들어와 MPC가 과반응한다 (Step 4 디버깅에서 발산 원인).
    CoM 속도는 처음부터 전신(subtree_linvel)이라 이 문제가 없었다.
    주의(검증 리포트 §4): 이 ω 는 내부 재배향(고양이 회전)에 0 을 낸다 — Θ(pelvis)
    와 ω(L)가 '같은 강체'가 아니라는 상태 불일치가 SRB 의 구조적 한계.
    """
    roll, pitch, yaw = quat_to_euler_zyx(d.qpos[3:7])
    p = d.subtree_com[0]

    mujoco.mj_subtreeVel(m, d)
    if I_body is not None:
        Rz_ = rz(yaw)
        I_w = Rz_ @ I_body @ Rz_.T
        omega_world = np.linalg.solve(I_w, d.subtree_angmom[0])
    else:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        omega_world = R.reshape(3, 3) @ d.qvel[3:6]   # pelvis ω (body->world)

    v_com = d.subtree_linvel[0]

    x = np.empty(NX)
    x[0:3] = (roll, pitch, yaw)
    x[3:6] = p
    x[6:9] = omega_world
    x[9:12] = v_com
    x[12] = GRAV
    return x


def get_foot_positions(m, d) -> np.ndarray:
    """(2,3) 발 site world 위치."""
    out = np.zeros((N_FEET, 3))
    for i, s in enumerate(FOOT_SITES):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)
        out[i] = d.site_xpos[sid]
    return out


# ---------------------------------------------------------------------------
# 동역학 행렬
# ---------------------------------------------------------------------------
def continuous_AB(params: SRBParams, psi: float,
                  r_feet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A_c(13x13), B_c(13x12).  r_feet: (2,3) = 발 site − CoM (world frame)."""
    A = np.zeros((NX, NX))
    A[0:3, 6:9] = rz(psi).T          # Θ̇ = Rz(ψ)ᵀ ω
    A[3:6, 9:12] = np.eye(3)         # ṗ = v
    A[11, 12] = -1.0                 # v̇_z ← −g (중력상수 상태)

    Rz_ = rz(psi)
    I_w = Rz_ @ params.I_body @ Rz_.T
    I_w_inv = np.linalg.inv(I_w)

    B = np.zeros((NX, NU))
    for i in range(N_FEET):
        c = NU_PER_FOOT * i
        B[6:9, c:c + 3] = I_w_inv @ skew(r_feet[i])   # ω̇ ← r×F
        B[6:9, c + 3:c + 6] = I_w_inv                  # ω̇ ← m  (wrench 신규)
        B[9:12, c:c + 3] = np.eye(3) / params.mass     # v̇ ← F
    return A, B


def discretize(Ac: np.ndarray, Bc: np.ndarray, dt: float):
    """정확한 ZOH (A_c 가 멱영: A³=0 이라 닫힌 형태 — 검증 리포트 §2).

        A_d = I + A·dt + ½A²·dt²
        B_d = (I·dt + ½A·dt²) B

    B_d 의 (1/6)A²dt³·B 항은 절단이 아니라 **정확히 0** — A²B = 0 이기 때문
    (B 는 ω·v 행만 채우고, A² 의 열은 Θ·p 만 읽는다). 즉 위 식이 정확한 ZOH.
    Euler(I + A·dt) 대비 추가되는 항은 ½g·dt²(자유낙하) 등 — dt=0.05 에서
    스텝당 1.2 cm. 평형에선 상쇄되지만 과도상태 preview 정확도에 기여.
    """
    A2 = Ac @ Ac
    Ad = np.eye(NX) + Ac * dt + 0.5 * A2 * dt * dt
    Bd = (np.eye(NX) * dt + 0.5 * Ac * dt * dt) @ Bc
    return Ad, Bd
