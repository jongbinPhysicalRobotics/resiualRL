"""G1 모델 로더.

한글이 포함된 경로에서 MuJoCo(C++)가 파일을 직접 열지 못하므로,
XML/메시를 Python이 읽어 assets 딕셔너리(가상 파일시스템)로 넘긴다.
"""
from __future__ import annotations

import mujoco

from paths import G1_DIR


def _collect_assets() -> dict[str, bytes]:
    """unitree_g1 폴더의 모든 파일을 {상대경로: bytes} 로 모은다.

    """
    assets: dict[str, bytes] = {}
    for f in G1_DIR.rglob("*"):
        if not f.is_file():
            continue
        # MuJoCo VFS 는 디렉터리를 무시하고 파일명(대소문자 무시)으로 찾으므로
        # basename 만 키로 쓴다. (이 폴더엔 basename 충돌이 없음을 확인함)
        assets[f.name] = f.read_bytes()
    return assets


def load(xml_text: str | None = None, scene: str = "scene.xml"):
    """모델 로드. xml_text 를 주면 그 문자열을, 아니면 scene 파일을 쓴다.

    Returns: (MjModel, MjData)
    """
    assets = _collect_assets()
    if xml_text is None:
        xml_text = (G1_DIR / scene).read_text(encoding="utf-8")
    m = mujoco.MjModel.from_xml_string(xml_text, assets)
    return m, mujoco.MjData(m)


# --------------------------------------------------------------------------
# 토크 제어용 모델
# --------------------------------------------------------------------------
def to_torque_actuators(m) -> None:
    """position 액추에이터를 motor(직접 토크)로 바꾼다 (in-place).

    Menagerie G1 은 <position kp=500 dampratio=1> 로 되어 있어서 ctrl 이 '목표 각도'다.
    우리는 ctrl 을 '토크'로 쓰고 싶으므로 gain/bias 를 motor 와 동일하게 만든다.

      position : gaintype=FIXED(gainprm[0]=kp), biastype=AFFINE(biasprm=[0,-kp,-kv])
      motor    : gaintype=FIXED(gainprm[0]=1),  biastype=NONE

    ctrlrange 는 관절의 actuatorfrcrange(N·m)로 바꿔준다.
    """
    for a in range(m.nu):
        m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_gainprm[a, :] = 0.0
        m.actuator_gainprm[a, 0] = 1.0
        m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_NONE
        m.actuator_biasprm[a, :] = 0.0

        jid = m.actuator_trnid[a, 0]
        if m.jnt_actfrclimited[jid]:
            m.actuator_ctrlrange[a] = m.jnt_actfrcrange[jid]
        else:
            m.actuator_ctrllimited[a] = 0


def load_torque(scene: str = "scene.xml"):
    """토크 제어 G1 을 로드한다."""
    m, d = load(scene=scene)
    to_torque_actuators(m)
    return m, d


# --------------------------------------------------------------------------
# 자세
# --------------------------------------------------------------------------
def leg_joint_names() -> list[str]:
    return [f"{side}_{j}_joint"
            for side in ("left", "right")
            for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]


def set_crouch(m, d, hip_pitch=-0.30, knee=0.60, ankle_pitch=-0.30, height=None):
    """무릎을 굽힌 기본 스탠스. keyframe 'stand' 는 무릎이 완전히 펴져 있어
    (특이자세) 수직 GRF 에 대한 무릎 토크가 0 에 가깝다. 실험용으론 굽힌 자세가 낫다.

    hip_pitch + knee + ankle_pitch = 0 이면 발바닥이 수평을 유지한다.
    height=None 이면 발이 바닥에 정확히 닿도록 자동으로 골반 높이를 맞춘다.
    """
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(m, d, kid)
    else:
        mujoco.mj_resetData(m, d)

    for side in ("left", "right"):
        for jname, val in ((f"{side}_hip_pitch_joint", hip_pitch),
                           (f"{side}_knee_joint", knee),
                           (f"{side}_ankle_pitch_joint", ankle_pitch)):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            d.qpos[m.jnt_qposadr[jid]] = val

    if height is None:
        # 발바닥 접촉구의 최저점이 z=0 이 되도록 골반을 내린다.
        mujoco.mj_forward(m, d)
        lowest = min(d.geom_xpos[g, 2] - m.geom_size[g, 0]
                     for g in range(m.ngeom) if m.geom_group[g] == 3)
        d.qpos[2] -= lowest
    else:
        d.qpos[2] = height

    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    return d
