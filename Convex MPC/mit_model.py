"""MIT 휴머노이드 모델 래퍼 — 로드, 이름→인덱스, 상태 읽기, 축소 몸체, 다리 동역학 (07_port_design.md §3.3).

reference (ispaik06/convex-mpc-biped) 의 MuJoCo 쪽 계층을 그대로 옮긴다:
  - 축소 몸체 (reduced body / SRB):  setupRobotParams.cpp:397-491  updateReducedBodyMassPropertiesFromData
  - 상태 읽기 (cheater state):       MujocoCheaterStateReader.cpp:50-89, 326-385
  - yaw unwrap (state estimator):     StateEstimator.cpp:25-53, AngleUtils.h:6-14
  - x0 (13):                          My_Controller.cpp:63-75, 315-332, 656-675
  - 다리 동역학 (aux 모델 대신 D4):  LegSwingDynamicsProvider.cpp:603-743
  - 토크 적용 (ctrlrange clamp):      SimulationRunner.cpp:803-818, RobotModel.cpp:196-221

관례 (07 §2): 다리/팔 인덱스 0 = Left, 1 = Right. 우리 MJCF 는 qpos 가 오른쪽 먼저라서 모든 인덱스는 이름으로 만든다.
다리 관절 순서 [hip_yaw, hip_abad, hip_pitch, knee, ankle], 팔 [shoulder_pitch, shoulder_abad, shoulder_yaw, elbow].

루프 순서 (07 §2, spec 04 §4.4): tick 은 mj_step 이 남긴 d 를 mj_forward 없이 읽는다 →
xpos/xquat/xipos/site/cvel/contact 는 한 틱 전 (pre-integration), qpos/qvel/time 은 현재 값. read_state 는 mj_forward 를
부르지 않는다. mj_forward 는 reset() 에서 한 번만.

mj_objectVelocity 의미 (확인함, mujoco 3.12): mjOBJ_BODY 는 [ω(3), v(3)] 를 월드 축으로 주는데 v 는 바디 **COM (xipos)**
의 선속도다 (mjOBJ_XBODY 가 바디 원점 xpos). reference 는 mjOBJ_BODY 를 쓴다 (MujocoCheaterStateReader.cpp:133-149,
151-164) → torso_vel_W = 몸통 링크 COM 의 속도. spec 04 §4.1 의 "body frame origin" 설명은 틀렸고, 코드 그대로
BODY 를 쓴다 (x0 의 v_com = v_torso + ω × offset 에서 offset 은 몸통 **원점** 기준 — reference 의 quirk 그대로).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

import paths

LEFT, RIGHT = 0, 1
SIDES = ("left", "right")

# spec 01 §6 이름 대응 (reference MitHumanoidSpec.cpp → 우리 MJCF). actuator 이름 == joint 이름.
BASE_BODY = "base"                                            # ref "torso"
LEG_JOINTS = (
    ("a06_left_hip_yaw", "a07_left_hip_abad", "a08_left_hip_pitch", "a09_left_knee", "a10_left_ankle"),
    ("a01_right_hip_yaw", "a02_right_hip_abad", "a03_right_hip_pitch", "a04_right_knee", "a05_right_ankle"),
)
ARM_JOINTS = (
    ("a15_left_shoulder_pitch", "a16_left_shoulder_abad", "a17_left_shoulder_yaw", "a18_left_elbow"),
    ("a11_right_shoulder_pitch", "a12_right_shoulder_abad", "a13_right_shoulder_yaw", "a14_right_elbow"),
)
FOOT_BODY = ("left_foot", "right_foot")                       # ref "left_foot_link"/"right_foot_link"
FOOT_SITE = ("left_foot_contact_site", "right_foot_contact_site")
HAND_BODY = ("left_lower_arm", "right_lower_arm")             # ref "left_forearm_link" (읽기만, 미사용)

# U1: reference G1/H1 씬의 디버그 마커 mocap 바디 (g1/scene_23dof.xml:21-39). 기본 밀도 → 질량이 생김.
DEBUG_MARKER_NAMES = ("debug_reduced_body_com", "debug_body_target",
                      "debug_left_touchdown_target", "debug_right_touchdown_target")
_DEBUG_MARKER_XML = """<mujoco><worldbody>
<!-- g1/scene_23dof.xml:20-40 의 geom 그대로 (rgba 생략 — 질량 무관) -->
<body name="debug_reduced_body_com" mocap="true" pos="0 0 0">
  <geom type="sphere" size="0.018" contype="0" conaffinity="0" group="4"/>
  <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.004" contype="0" conaffinity="0" group="4"/>
  <geom type="capsule" fromto="0 0 0 0 0.08 0" size="0.004" contype="0" conaffinity="0" group="4"/>
  <geom type="capsule" fromto="0 0 0 0 0 0.08" size="0.004" contype="0" conaffinity="0" group="4"/>
</body>
<body name="debug_body_target" mocap="true" pos="0 0 0">
  <geom type="sphere" size="0.018" contype="0" conaffinity="0" group="4"/>
  <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.004" contype="0" conaffinity="0" group="4"/>
  <geom type="capsule" fromto="0 0 0 0 0.08 0" size="0.004" contype="0" conaffinity="0" group="4"/>
  <geom type="capsule" fromto="0 0 0 0 0 0.08" size="0.004" contype="0" conaffinity="0" group="4"/>
</body>
<body name="debug_left_touchdown_target" mocap="true" pos="0 0 0">
  <geom type="sphere" size="0.025" contype="0" conaffinity="0" group="4"/>
  <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.004" contype="0" conaffinity="0" group="4"/>
</body>
<body name="debug_right_touchdown_target" mocap="true" pos="0 0 0">
  <geom type="sphere" size="0.025" contype="0" conaffinity="0" group="4"/>
  <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.004" contype="0" conaffinity="0" group="4"/>
</body>
</worldbody></mujoco>"""


# ----------------------------------------------------------------------------------------------------------------
# 로딩 (한글 경로 → VFS, 06 §5.1)
# ----------------------------------------------------------------------------------------------------------------
def collect_assets(folder=None) -> dict:
    """모델 폴더의 모든 파일 {basename: bytes}. MuJoCo 는 한글 경로를 못 열어서 VFS 로 넘긴다."""
    folder = paths.MIT_DIR if folder is None else folder
    return {f.name: f.read_bytes() for f in folder.rglob("*") if f.is_file()}


def load_model(armature: bool = True, timestep: float = 0.002, integrator: str = "implicitfast",
               assets: dict | None = None) -> mujoco.MjModel:
    """scene.xml (mit_humanoid.xml include) 을 VFS 로 컴파일.

    reference 는 로드 후 timestep/integrator 를 simulation.yaml 값으로 덮어쓴다 (SimulationConfig.cpp:96-104).
    armature=False: dof_armature 를 0 으로 (D1 민감도 옵션)."""
    assets = collect_assets() if assets is None else assets
    xml = assets["scene.xml"].decode("utf-8")
    m = mujoco.MjModel.from_xml_string(xml, assets)
    m.opt.timestep = float(timestep)
    m.opt.integrator = {"implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                        "euler": mujoco.mjtIntegrator.mjINT_EULER,
                        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
                        "rk4": mujoco.mjtIntegrator.mjINT_RK4}[integrator]
    if not armature:
        m.dof_armature[:] = 0.0
    return m


# ----------------------------------------------------------------------------------------------------------------
# 작은 수학 도우미
# ----------------------------------------------------------------------------------------------------------------
def Rz(a: float) -> np.ndarray:
    """ref: MatrixUtils.h:11-21."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap_to_pi(a: float) -> float:
    """ref: AngleUtils.h:6-8  atan2(sin a, cos a)."""
    return float(np.arctan2(np.sin(a), np.cos(a)))


def quat_to_roll_pitch(q) -> tuple[float, float]:
    """ref: My_Controller.cpp:63-75 quaternionToRollPitch (q = w,x,y,z, 먼저 정규화)."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = q / np.linalg.norm(q)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


def quat_to_mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


# ----------------------------------------------------------------------------------------------------------------
# 데이터 구조
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class ReducedBody:
    """축소 몸체 (몸통 + 팔, 다리 subtree 제외). spec 04 §3.6 / setupRobotParams.cpp:488-490.

    mass         : Σ m_b                                              (bodyMass)
    com_offset_B : Rz(ψ)ᵀ (com_W − torso_origin_W)                    (bodyComLocation, yaw frame, 몸통 원점 기준)
    inertia_B    : Rz(ψ)ᵀ I_W Rz(ψ), 축소 몸체 COM 기준                (bodyInertia, 3x3 전체)
    ψ = atan2(xmat[1,0], xmat[0,0]) — wrapped yaw (xmat 에서)."""
    mass: float
    com_offset_B: np.ndarray
    inertia_B: np.ndarray
    com_W: np.ndarray = field(default_factory=lambda: np.zeros(3))   # 참고용 (Σ m xipos / M)


@dataclass
class LegDyn:
    """한 다리의 고정-베이스 동역학 (D4; aux 모델 LegSwingDynamicsProvider.cpp:632-687 과 같은 값).

    Jv, Jw   : (3,5) 발 site 의 병진/회전 Jacobian (월드 축, 다리 dof 열만)
    JvDot_qd : (3,)  JvDot @ qd_leg (베이스 속도 0)
    JvDot    : (3,5) 참고용
    M        : (5,5) 다리 블록 질량 행렬 (armature 포함)
    bias     : (5,)  qfrc_bias 다리 성분 (중력 + 코리올리, 몸통 정지)"""
    Jv: np.ndarray
    Jw: np.ndarray
    JvDot_qd: np.ndarray
    M: np.ndarray
    bias: np.ndarray
    JvDot: np.ndarray = field(default_factory=lambda: np.zeros((3, 5)))


@dataclass
class RobotState:
    """한 틱의 측정 상태 (spec 04 §4-6, spec 05 §0). 다리/팔 첫 인덱스 0 = Left."""
    t: float
    torso_pos_W: np.ndarray        # (3,)  d.xpos[base]                (stale)
    torso_quat_W: np.ndarray       # (4,)  d.xquat[base] (w,x,y,z)     (stale)
    R_WT: np.ndarray               # (3,3) 쿼터니언에서 만든 회전행렬
    roll: float
    pitch: float
    yaw_wrapped: float
    yaw_unwrapped: float
    torso_vel_W: np.ndarray        # (3,)  mj_objectVelocity(BODY, local=0)[3:6]  — 몸통 COM 점의 속도, 월드
    torso_angvel_W: np.ndarray     # (3,)  mj_objectVelocity(BODY, local=0)[0:3]  — 월드 축
    q_leg: np.ndarray              # (2,5)  qpos (fresh)
    qd_leg: np.ndarray             # (2,5)  qvel (fresh)
    q_arm: np.ndarray              # (2,4)
    qd_arm: np.ndarray             # (2,4)
    foot_pos_W: np.ndarray         # (2,3)  site_xpos
    foot_vel_W: np.ndarray         # (2,3)  mj_objectVelocity(SITE, local=0)[3:6]
    foot_R_W: np.ndarray           # (2,3,3) site_xmat
    foot_contact: np.ndarray       # (2,) bool   접촉 개수 > 0
    foot_normal_force: np.ndarray  # (2,) float  Σ |mj_contactForce[0]|
    reduced: ReducedBody
    yaw_rate_W: float = 0.0                                                   # 추정기 부산물 (미사용)
    foot_force_W: np.ndarray = field(default_factory=lambda: np.zeros((2, 3)))  # 발에 작용하는 접촉력 합 (미사용)
    foot_contact_count: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=int))

    # --- ref: My_Controller.cpp:315-332 ---
    def com_offset_W(self) -> np.ndarray:
        """Rz(yaw_unwrapped) @ bodyComLocation."""
        return Rz(self.yaw_unwrapped) @ self.reduced.com_offset_B

    def com_W(self) -> np.ndarray:
        """reducedBodyComWorld = torsoPos_W + Rz(ψ_unwrapped) bodyComLocation."""
        return self.torso_pos_W + self.com_offset_W()

    def com_vel_W(self) -> np.ndarray:
        """reducedBodyComVelocityWorld = torsoLinVel_W + ω_W × offset_W (팔 관절 속도 무시, 강체 가정)."""
        return self.torso_vel_W + np.cross(self.torso_angvel_W, self.com_offset_W())

    def x0(self, g: float = -9.81) -> np.ndarray:
        """ref: My_Controller.cpp:656-675 buildCurrentMpcState.
        [roll, pitch, yaw_unwrapped, com_W(3), ω_torso_W(3), v_com_W(3), g]."""
        x = np.empty(13)
        x[0] = self.roll
        x[1] = self.pitch
        x[2] = self.yaw_unwrapped
        off = self.com_offset_W()
        x[3:6] = self.torso_pos_W + off
        x[6:9] = self.torso_angvel_W
        x[9:12] = self.torso_vel_W + np.cross(self.torso_angvel_W, off)
        x[12] = g
        return x


# ----------------------------------------------------------------------------------------------------------------
# MitModel
# ----------------------------------------------------------------------------------------------------------------
class MitModel:
    """MuJoCo 모델 + 데이터 + reference 바인딩.

    속성: m, d; leg_qadr/leg_dof/leg_act (2,5); arm_qadr/arm_dof/arm_act (2,4); base_body; foot_body (2,);
          foot_site (2,); foot_geoms (list[list[int]], 충돌 geom); hip_location_B (2,3) (= body_pos[hip_yaw]);
          reduced_bodies (축소 몸체에 들어가는 body id 배열)."""

    def __init__(self, armature: bool = True, timestep: float = 0.002, integrator: str = "implicitfast",
                 emulate_debug_markers: bool = False):
        self.assets = collect_assets()
        self.armature = bool(armature)
        self.m = load_model(armature, timestep, integrator, self.assets)
        self.d = mujoco.MjData(self.m)
        m = self.m

        def jid(name):
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if i < 0:
                raise KeyError(f"joint {name!r} not found")
            return i

        def aid(name):
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if i < 0:
                raise KeyError(f"actuator {name!r} not found")
            return i

        # ref: setupRobotParams.cpp:82-105 fillJointGroup (이름 → qposadr/dofadr/actuator id)
        self.leg_jnt = np.array([[jid(n) for n in LEG_JOINTS[s]] for s in (LEFT, RIGHT)], dtype=int)
        self.arm_jnt = np.array([[jid(n) for n in ARM_JOINTS[s]] for s in (LEFT, RIGHT)], dtype=int)
        self.leg_qadr = m.jnt_qposadr[self.leg_jnt].astype(int)
        self.leg_dof = m.jnt_dofadr[self.leg_jnt].astype(int)
        self.arm_qadr = m.jnt_qposadr[self.arm_jnt].astype(int)
        self.arm_dof = m.jnt_dofadr[self.arm_jnt].astype(int)
        self.leg_act = np.array([[aid(n) for n in LEG_JOINTS[s]] for s in (LEFT, RIGHT)], dtype=int)
        self.arm_act = np.array([[aid(n) for n in ARM_JOINTS[s]] for s in (LEFT, RIGHT)], dtype=int)

        self.base_body = m.body(BASE_BODY).id
        self.foot_body = np.array([m.body(n).id for n in FOOT_BODY], dtype=int)
        self.foot_site = np.array([m.site(n).id for n in FOOT_SITE], dtype=int)
        self.hand_body = np.array([m.body(n).id for n in HAND_BODY], dtype=int)
        # ref: setupRobotParams.cpp:37-51 collisionGeomIdsForBody (contype|conaffinity != 0 or group 3)
        self.foot_geoms = [[g for g in range(m.ngeom) if m.geom_bodyid[g] == fb and
                            (m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0 or m.geom_group[g] == 3)]
                           for fb in self.foot_body]
        # 접촉 판정용: geom 이 발 바디 소속이거나 충돌 geom 목록에 있으면 발 (MujocoCheaterStateReader.cpp:50-61)
        self._geom_foot = np.full(m.ngeom, -1, dtype=int)
        for leg in (LEFT, RIGHT):
            self._geom_foot[m.geom_bodyid == self.foot_body[leg]] = leg
            self._geom_foot[self.foot_geoms[leg]] = leg

        # 다리 subtree 루트 = 첫 다리 관절의 body (hip_yaw), ref: setupRobotParams.cpp:132-140,183
        self.leg_root_body = m.jnt_bodyid[self.leg_jnt[:, 0]].astype(int)
        self.hip_location_B = np.array([m.body_pos[b].copy() for b in self.leg_root_body])  # :142-152,197
        # 축소 몸체 바디: id 1..nbody-1, 다리 subtree 아님, mass > 0 (setupRobotParams.cpp:414-431)
        in_leg = np.zeros(m.nbody, dtype=bool)
        for b in range(1, m.nbody):
            c = b
            while c > 0:
                if c in self.leg_root_body:
                    in_leg[b] = True
                    break
                c = m.body_parentid[c]
        self.in_leg_subtree = in_leg
        self.reduced_bodies = np.array([b for b in range(1, m.nbody)
                                        if not in_leg[b] and m.body_mass[b] > 0.0], dtype=int)
        self._rb_mass = m.body_mass[self.reduced_bodies].copy()
        self._rb_inertia = m.body_inertia[self.reduced_bodies].copy()     # (k,3) 주관성
        self.leg_body_ids = [np.array([b for b in range(m.nbody) if in_leg[b] and
                                       self._is_desc(b, self.leg_root_body[s])], dtype=int) for s in (LEFT, RIGHT)]

        self.ctrl_lo = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[:, 1].copy()
        self.ctrl_limited = m.actuator_ctrllimited.astype(bool).copy()

        # U1: 디버그 마커 흉내 (기본 끔). debug_marker_fn() -> {name: (pos(3), quat(4))} 또는 None.
        # reference 에선 tick k 끝에 mocap 을 옮기고 tick k+1 에 xipos 가 따라온다 → 콜백은 "지난 틱에 옮긴 자세" 를 준다.
        self.emulate_debug_markers = bool(emulate_debug_markers)
        self.debug_marker_fn = None
        self._marker_pose = {n: (np.zeros(3), np.array([1.0, 0, 0, 0])) for n in DEBUG_MARKER_NAMES}
        self._marker_props = self._compile_marker_props()

        # 다리 동역학용 scratch (D4)
        self._dyn = mujoco.MjData(m)
        self._dyn_key = None
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))
        self._jacdp = np.zeros((3, m.nv))
        self._Mfull = np.zeros((m.nv, m.nv))
        self._leg_cache: list = [None, None]

        # yaw unwrap 상태 (StateEstimator.cpp:25-53)
        self._has_yaw = False
        self._last_yaw_wrapped = 0.0
        self._last_yaw_unwrapped = 0.0
        self._last_yaw_time = 0.0

        self._v6 = np.zeros(6)
        self._f6 = np.zeros(6)

    def _is_desc(self, b, root):
        c = b
        while c > 0:
            if c == root:
                return True
            c = self.m.body_parentid[c]
        return False

    def _compile_marker_props(self):
        mm = mujoco.MjModel.from_xml_string(_DEBUG_MARKER_XML)
        props = {}
        for n in DEBUG_MARKER_NAMES:
            b = mm.body(n).id
            props[n] = (float(mm.body_mass[b]), mm.body_ipos[b].copy(), mm.body_iquat[b].copy(),
                        mm.body_inertia[b].copy())
        return props

    # ------------------------------------------------------------------------------------------------------------
    def reset(self, key: str = "stand"):
        """mj_resetDataKeyframe + mj_forward (07 §2: 키프레임 리셋 뒤 한 번만). yaw unwrap 초기화."""
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key(key).id)
        mujoco.mj_forward(self.m, self.d)
        self.reset_yaw_unwrap()
        self._dyn_key = None

    def reset_yaw_unwrap(self):
        self._has_yaw = False

    def _update_yaw(self, yaw_wrapped: float, t: float) -> tuple[float, float]:
        """ref: StateEstimator.cpp:29-52. 첫 호출 / 시간 비유한 / 시간 역행 → unwrapped = wrapped."""
        if (not self._has_yaw) or (not np.isfinite(t)) or t < self._last_yaw_time - 1e-9:
            yu, rate = yaw_wrapped, 0.0
        else:
            yu = self._last_yaw_unwrapped + wrap_to_pi(yaw_wrapped - self._last_yaw_wrapped)
            dt = t - self._last_yaw_time
            rate = (yu - self._last_yaw_unwrapped) / dt if dt > 0.0 else 0.0
        self._last_yaw_wrapped = yaw_wrapped
        self._last_yaw_unwrapped = yu
        self._last_yaw_time = t
        self._has_yaw = True
        return yu, rate

    # ------------------------------------------------------------------------------------------------------------
    def reduced_body(self) -> ReducedBody:
        """ref: setupRobotParams.cpp:397-491 (현재 d.xipos/ximat/xmat/xpos, mj_forward 없음)."""
        d = self.d
        ids = self.reduced_bodies
        mass = self._rb_mass
        xipos = d.xipos[ids]                                   # (k,3)
        ximat = d.ximat[ids].reshape(-1, 3, 3)                 # (k,3,3)
        inert = self._rb_inertia
        if self.emulate_debug_markers:
            if self.debug_marker_fn is not None:
                upd = self.debug_marker_fn()
                if upd:
                    for n, (p, q) in upd.items():
                        self._marker_pose[n] = (np.asarray(p, float).copy(), np.asarray(q, float).copy())
            mk_m, mk_p, mk_R, mk_I = [], [], [], []
            for n in DEBUG_MARKER_NAMES:
                mb, ipos, iquat, I = self._marker_props[n]
                p, q = self._marker_pose[n]
                R = quat_to_mat(q)
                mk_m.append(mb)
                mk_p.append(p + R @ ipos)
                mk_R.append(R @ quat_to_mat(iquat))
                mk_I.append(I)
            mass = np.concatenate([mass, mk_m])
            xipos = np.vstack([xipos, mk_p])
            ximat = np.concatenate([ximat, np.array(mk_R)])
            inert = np.vstack([inert, mk_I])
        M = float(mass.sum())
        com_W = (mass[:, None] * xipos).sum(0) / M
        # R diag(I) Rᵀ  +  m (|o|² 1 − o oᵀ)
        I_rot = np.einsum("kij,kj,klj->kil", ximat, inert, ximat)
        o = xipos - com_W
        oo = np.einsum("ki,ki->k", o, o)
        I_W = I_rot.sum(0) + np.einsum("k,kij->ij", mass, oo[:, None, None] * np.eye(3)[None] -
                                       o[:, :, None] * o[:, None, :])
        R_WT = d.xmat[self.base_body].reshape(3, 3)
        psi = np.arctan2(R_WT[1, 0], R_WT[0, 0])              # yawFromRotationMatrix (wrapped)
        Rwb = Rz(psi)
        return ReducedBody(mass=M,
                           com_offset_B=Rwb.T @ (com_W - d.xpos[self.base_body]),
                           inertia_B=Rwb.T @ I_W @ Rwb,
                           com_W=com_W)

    def _foot_contacts(self):
        """ref: MujocoCheaterStateReader.cpp:63-89 readFootContactInfo."""
        m, d = self.m, self.d
        contact = np.zeros(2, dtype=bool)
        count = np.zeros(2, dtype=int)
        normal = np.zeros(2)
        force = np.zeros((2, 3))
        n = d.ncon
        if n == 0:
            return contact, count, normal, force
        g1 = d.contact.geom1[:n]
        g2 = d.contact.geom2[:n]
        f1 = self._geom_foot[g1]
        f2 = self._geom_foot[g2]
        for i in np.nonzero((f1 >= 0) | (f2 >= 0))[0]:
            mujoco.mj_contactForce(m, d, int(i), self._f6)
            frame = d.contact.frame[i].reshape(3, 3)
            F_on_g2 = frame.T @ self._f6[:3]
            for leg in (LEFT, RIGHT):
                is1, is2 = f1[i] == leg, f2[i] == leg
                if not (is1 or is2):
                    continue
                contact[leg] = True
                count[leg] += 1
                force[leg] += (1.0 if is2 else -1.0) * F_on_g2
                normal[leg] += abs(self._f6[0])
        return contact, count, normal, force

    def read_state(self) -> RobotState:
        """현재 d 를 그대로 읽는다 (mj_forward 없음). 축소 몸체는 매 호출 갱신 (SimulationRunner.cpp:409-412)."""
        m, d = self.m, self.d
        b = self.base_body
        reduced = self.reduced_body()                           # 1) updateReducedBodyMassPropertiesFromData
        t = float(d.time)
        pos = d.xpos[b].copy()
        quat = d.xquat[b].copy()
        R_WT = quat_to_mat(quat)                               # Eigen toRotationMatrix (StateEstimator.cpp:27)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, self._v6, 0)
        angvel = self._v6[0:3].copy()
        vel = self._v6[3:6].copy()
        roll, pitch = quat_to_roll_pitch(quat)
        yaw_w = float(np.arctan2(R_WT[1, 0], R_WT[0, 0]))
        yaw_u, yaw_rate = self._update_yaw(yaw_w, t)           # 3) StateEstimator (cheater)

        foot_pos = d.site_xpos[self.foot_site].copy()
        foot_R = d.site_xmat[self.foot_site].reshape(2, 3, 3).copy()
        foot_vel = np.zeros((2, 3))
        for leg in (LEFT, RIGHT):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, int(self.foot_site[leg]), self._v6, 0)
            foot_vel[leg] = self._v6[3:6]
        contact, count, normal, force = self._foot_contacts()

        return RobotState(
            t=t, torso_pos_W=pos, torso_quat_W=quat, R_WT=R_WT, roll=roll, pitch=pitch,
            yaw_wrapped=yaw_w, yaw_unwrapped=yaw_u, torso_vel_W=vel, torso_angvel_W=angvel,
            q_leg=d.qpos[self.leg_qadr].copy(), qd_leg=d.qvel[self.leg_dof].copy(),
            q_arm=d.qpos[self.arm_qadr].copy(), qd_arm=d.qvel[self.arm_dof].copy(),
            foot_pos_W=foot_pos, foot_vel_W=foot_vel, foot_R_W=foot_R,
            foot_contact=contact, foot_normal_force=normal, reduced=reduced,
            yaw_rate_W=yaw_rate, foot_force_W=force, foot_contact_count=count)

    # ------------------------------------------------------------------------------------------------------------
    # D4: 다리 동역학 (aux 모델 대신 전체 모델 scratch)
    # ------------------------------------------------------------------------------------------------------------
    def _prepare_dynamics(self):
        """scratch MjData: 몸통 자세 = d.xpos/xquat[base] (stale, aux assignBasePose 와 같음), 다리 q/qd = d.qpos/qvel
        (fresh), 베이스 속도 0, 팔 속도 0 → mj_forward. ref: LegSwingDynamicsProvider.cpp:611-630, 697-719.

        두 다리 속도를 한 번에 넣는다: 다리 L dof 의 qfrc_bias / JvDot 는 베이스와 다리 L 의 운동에만 의존
        (베이스 속도 0 이면 다리 R 의 속도는 L 의 subtree 에 들어오지 않는다) → 다리별 aux 모델과 같은 값.
        01_check_model.py 가 진짜 고정-베이스 단일 다리 모델과 1e-9 이내로 같음을 보인다.
        결과는 (qpos, qvel, 몸통 자세, time) 이 같으면 캐시."""
        m, d, s = self.m, self.d, self._dyn
        b = self.base_body
        key = np.concatenate([[d.time], d.qpos[self.leg_qadr].ravel(), d.qvel[self.leg_dof].ravel(),
                              d.xpos[b], d.xquat[b]])
        if self._dyn_key is not None and np.array_equal(key, self._dyn_key):
            return
        s.qpos[:] = d.qpos                                     # 팔 q 는 결과에 무관 (aux 에선 팔 삭제)
        s.qpos[0:3] = d.xpos[b]
        s.qpos[3:7] = d.xquat[b]
        s.qvel[:] = 0.0
        s.qvel[self.leg_dof.ravel()] = d.qvel[self.leg_dof.ravel()]
        s.time = d.time
        mujoco.mj_forward(m, s)
        mujoco.mj_fullM(m, s, self._Mfull)                     # mujoco 3.12: mj_fullM(m, d, dst)
        self._dyn_key = key
        self._leg_cache = [None, None]

    def leg_dynamics(self, leg: int) -> LegDyn:
        """D4. Jv, Jw = mj_jacSite 다리 열; M = mj_fullM 다리 블록 (armature 포함); bias = qfrc_bias 다리 성분;
        JvDot = mj_jacDot(point = site_xpos, body = site_bodyid) 다리 열 (ref: LegSwingDynamicsProvider.cpp:201-220),
        모두 scratch (베이스 속도 0) 에서."""
        self._prepare_dynamics()
        if self._leg_cache[leg] is not None:
            return self._leg_cache[leg]
        m, s = self.m, self._dyn
        site = int(self.foot_site[leg])
        cols = self.leg_dof[leg]
        mujoco.mj_jacSite(m, s, self._jacp, self._jacr, site)
        mujoco.mj_jacDot(m, s, self._jacdp, None, s.site_xpos[site].copy(), int(m.site_bodyid[site]))
        Jv = self._jacp[:, cols].copy()
        Jw = self._jacr[:, cols].copy()
        JvDot = self._jacdp[:, cols].copy()
        qd = s.qvel[cols]
        out = LegDyn(Jv=Jv, Jw=Jw, JvDot_qd=JvDot @ qd, M=self._Mfull[np.ix_(cols, cols)].copy(),
                     bias=s.qfrc_bias[cols].copy(), JvDot=JvDot)
        self._leg_cache[leg] = out
        return out

    def standing_jacobians(self) -> tuple[np.ndarray, np.ndarray]:
        """ref: LegSwingDynamicsProvider.cpp:691-743 — 6x10, 행 [L 발 3; R 발 3], 열 [L 다리 5 | R 다리 5]."""
        self._prepare_dynamics()
        Jv = np.zeros((6, 10))
        Jw = np.zeros((6, 10))
        for leg in (LEFT, RIGHT):
            dyn = self.leg_dynamics(leg)
            Jv[3 * leg:3 * leg + 3, 5 * leg:5 * leg + 5] = dyn.Jv
            Jw[3 * leg:3 * leg + 3, 5 * leg:5 * leg + 5] = dyn.Jw
        return Jv, Jw

    # ------------------------------------------------------------------------------------------------------------
    def write_torque(self, tau_leg, tau_arm) -> np.ndarray:
        """actuator id 로 scatter, ctrllimited 면 ctrlrange 로 clamp, d.ctrl 에 쓴다. 나머지 actuator 는 0.
        ref: RobotRunner.cpp:154-158, SimulationRunner.cpp:803-818. 반환: clamp 전 tau (nu,)."""
        tau = np.zeros(self.m.nu)
        tau[self.leg_act.ravel()] = np.asarray(tau_leg, dtype=float).reshape(-1)
        tau[self.arm_act.ravel()] = np.asarray(tau_arm, dtype=float).reshape(-1)
        ctrl = np.where(self.ctrl_limited, np.clip(tau, self.ctrl_lo, self.ctrl_hi), tau)
        self.d.ctrl[:] = ctrl
        return tau

    # 편의
    def total_mass(self) -> float:
        return float(self.m.body_mass.sum())

    def whole_com_W(self) -> np.ndarray:
        return self.d.subtree_com[0].copy()
