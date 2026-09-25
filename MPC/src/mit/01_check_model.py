"""모델 점검 + D4 증명.

(1) 질량, keyframe "stand" 의 축소 몸체 (spec 01 §4 의 팔 자세 값과 비교), 발 site 위치, spawn 시 x0 출력.
(2) D4: MitModel.leg_dynamics (전체 모델 scratch, 베이스 속도 0) == reference 식 보조 모델
    (mujoco.MjSpec 로 mit_humanoid.xml 에서 actuator/key/free joint/다른 다리/양 팔 삭제 → 몸통이 world 에 고정된
    단일 다리 모델, LegSwingDynamicsProvider.cpp:325-434) 을 무작위 자세·속도 3 개에서 비교.
    reference 와 같은 stale/fresh 조합을 흉내낸다: main d 를 mj_forward 한 뒤 qpos/qvel 을 바꿔 (= mj_step 적분)
    몸통 자세는 d.xpos/xquat (stale), 다리 q/qd 는 새 값 → aux 에는 body_pos/body_quat = stale 몸통 자세,
    qpos/qvel = 새 다리 값 (assignBasePose :73-85, :611-630). standing 6x10 Jacobian 도 두 다리 aux 와 비교.

실행:  PYTHONIOENCODING=utf-8 ../../../.venv/Scripts/python.exe 01_check_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mit_model as mm  # noqa: E402
from mit_model import LEFT, RIGHT  # noqa: E402

# spec 01 §4 (팔 [0,0,0,-1.65], yaw 0) — 다리는 들어가지 않으므로 stand / init_yaml 모두 같은 값
SPEC_MASS = 14.227468
SPEC_TOTAL = 24.888550
SPEC_COM_OFFSET = np.array([0.015306, 0.002857, 0.116574])
SPEC_INERTIA = np.array([[0.519367, 0.001639, 0.005789],
                         [0.001639, 0.168406, -0.001019],
                         [0.005789, -0.001019, 0.402526]])


# ----------------------------------------------------------------------------------------------------------------
# reference 식 보조 모델 (MjSpec)
# ----------------------------------------------------------------------------------------------------------------
def build_aux_model(assets: dict, keep_legs=(LEFT,), armature: bool = True) -> mujoco.MjModel:
    """mit_humanoid.xml (로봇만, 바닥 없음) → actuator/sensor/key 삭제, free joint 삭제, 남기지 않는 다리의 hip_yaw
    subtree 와 양 팔 root subtree 삭제. XML <option> 은 그대로 (reference 도 aux 는 configure 안 함, spec 04 §7.1-5)."""
    spec = mujoco.MjSpec.from_string(assets["mit_humanoid.xml"].decode("utf-8"), assets=assets)
    for a in list(spec.actuators):
        spec.delete(a)
    for s in list(spec.sensors):
        spec.delete(s)
    for k in list(spec.keys):
        spec.delete(k)
    free = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]
    for j in free:
        spec.delete(j)
    for leg in (LEFT, RIGHT):
        if leg not in keep_legs:
            spec.delete(spec.body(mm.LEG_JOINTS[leg][0].split("_", 1)[1]))   # "left_hip_yaw" 등 (hip_yaw body)
    for side in ("left", "right"):
        spec.delete(spec.body(f"{side}_shoulder"))
    aux = spec.compile()
    if not armature:
        aux.dof_armature[:] = 0.0
    return aux


class AuxLeg:
    """한 aux 모델 + 인덱스 (LegSwingDynamicsProvider.cpp:387-426)."""

    def __init__(self, aux: mujoco.MjModel, legs):
        self.m = aux
        self.d = mujoco.MjData(aux)
        self.torso = aux.body(mm.BASE_BODY).id
        self.legs = tuple(legs)
        self.qadr = {l: np.array([aux.jnt_qposadr[aux.joint(n).id] for n in mm.LEG_JOINTS[l]]) for l in legs}
        self.dof = {l: np.array([aux.jnt_dofadr[aux.joint(n).id] for n in mm.LEG_JOINTS[l]]) for l in legs}
        self.site = {l: aux.site(mm.FOOT_SITE[l]).id for l in legs}

    def evaluate(self, torso_pos, torso_quat, q_leg, qd_leg):
        m, d = self.m, self.d
        m.body_pos[self.torso] = torso_pos                   # assignBasePose :73-85
        m.body_quat[self.torso] = torso_quat
        for l in self.legs:
            d.qpos[self.qadr[l]] = q_leg[l]
            d.qvel[self.dof[l]] = qd_leg[l]
        mujoco.mj_forward(m, d)
        out = {}
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); jdp = np.zeros((3, m.nv))
        Mf = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, d, Mf)
        for l in self.legs:
            s = self.site[l]
            mujoco.mj_jacSite(m, d, jp, jr, s)
            mujoco.mj_jacDot(m, d, jdp, None, d.site_xpos[s].copy(), int(m.site_bodyid[s]))
            c = self.dof[l]
            out[l] = dict(Jv=jp[:, c].copy(), Jw=jr[:, c].copy(), JvDot_qd=jdp[:, c] @ d.qvel[c],
                          M=Mf[np.ix_(c, c)].copy(), bias=d.qfrc_bias[c].copy(),
                          site=d.site_xpos[s].copy())
        return out


def _random_state(model: mm.MitModel, rng):
    """무작위 몸통 자세 + 관절 (범위 안) + 모든 dof 속도 (베이스 포함)."""
    m, d = model.m, model.d
    q = d.qpos
    q[0:3] = rng.uniform([-1, -1, 0.5], [1, 1, 0.9])
    quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
    q[3:7] = quat
    for j in range(1, m.njnt):
        lo, hi = m.jnt_range[j]
        q[m.jnt_qposadr[j]] = rng.uniform(lo, hi) if m.jnt_limited[j] else rng.uniform(-1, 1)
    d.qvel[:] = rng.normal(scale=2.0, size=m.nv)


def check_d4(n_configs: int = 3, seed: int = 0, armature: bool = True, verbose: bool = True) -> dict:
    """D4 증명. 반환 {"J":…, "M":…, "bias":…, "JvDot_qd":…, "standing_J":…} 최대 절대 차이."""
    model = mm.MitModel(armature=armature)
    model.reset("stand")
    aux1 = {l: AuxLeg(build_aux_model(model.assets, (l,), armature), (l,)) for l in (LEFT, RIGHT)}
    aux2 = AuxLeg(build_aux_model(model.assets, (LEFT, RIGHT), armature), (LEFT, RIGHT))
    rng = np.random.default_rng(seed)
    worst = dict(J=0.0, M=0.0, bias=0.0, JvDot_qd=0.0, standing_J=0.0, site=0.0)
    scale = dict(J=0.0, M=0.0, bias=0.0, JvDot_qd=0.0)
    m, d = model.m, model.d
    for k in range(n_configs):
        _random_state(model, rng)
        mujoco.mj_forward(m, d)                               # 이 시점 kinematics = "이전 틱"
        # mj_step 적분 흉내: qpos/qvel 을 바꾸고 kinematics 는 그대로 둔다 (stale 몸통 자세 + fresh 다리)
        d.qpos[0:3] += rng.normal(scale=0.01, size=3)
        dq = rng.normal(scale=0.02, size=4); d.qpos[3:7] += dq; d.qpos[3:7] /= np.linalg.norm(d.qpos[3:7])
        d.qpos[model.leg_qadr.ravel()] += rng.normal(scale=0.02, size=10)
        d.qvel[:] += rng.normal(scale=0.5, size=m.nv)
        tpos, tquat = d.xpos[model.base_body].copy(), d.xquat[model.base_body].copy()
        q_leg = d.qpos[model.leg_qadr].copy()
        qd_leg = d.qvel[model.leg_dof].copy()
        Jv6, Jw6 = model.standing_jacobians()
        ref2 = aux2.evaluate(tpos, tquat, q_leg, qd_leg)
        for leg in (LEFT, RIGHT):
            ours = model.leg_dynamics(leg)
            ref = aux1[leg].evaluate(tpos, tquat, q_leg, qd_leg)[leg]
            dJ = max(np.abs(ours.Jv - ref["Jv"]).max(), np.abs(ours.Jw - ref["Jw"]).max())
            dM = np.abs(ours.M - ref["M"]).max()
            db = np.abs(ours.bias - ref["bias"]).max()
            dj = np.abs(ours.JvDot_qd - ref["JvDot_qd"]).max()
            ds = np.abs(model._dyn.site_xpos[model.foot_site[leg]] - ref["site"]).max()
            worst["J"] = max(worst["J"], dJ); worst["M"] = max(worst["M"], dM)
            worst["bias"] = max(worst["bias"], db); worst["JvDot_qd"] = max(worst["JvDot_qd"], dj)
            worst["site"] = max(worst["site"], ds)
            scale["J"] = max(scale["J"], np.abs(ref["Jv"]).max()); scale["M"] = max(scale["M"], np.abs(ref["M"]).max())
            scale["bias"] = max(scale["bias"], np.abs(ref["bias"]).max())
            scale["JvDot_qd"] = max(scale["JvDot_qd"], np.abs(ref["JvDot_qd"]).max())
            # standing (두 다리 aux): 행 [L;R], 열 [L5|R5]
            dS = max(np.abs(Jv6[3 * leg:3 * leg + 3, 5 * leg:5 * leg + 5] - ref2[leg]["Jv"]).max(),
                     np.abs(Jw6[3 * leg:3 * leg + 3, 5 * leg:5 * leg + 5] - ref2[leg]["Jw"]).max())
            worst["standing_J"] = max(worst["standing_J"], dS)
            if verbose:
                print(f"  config {k} leg {'LR'[leg]}: |dJ| {dJ:.2e}  |dM| {dM:.2e}  |dbias| {db:.2e}  "
                      f"|dJvDot qd| {dj:.2e}  |dsite| {ds:.2e}  |dJ standing| {dS:.2e}")
        # 다른 다리 블록이 0 인지
        off = max(np.abs(Jv6[0:3, 5:10]).max(), np.abs(Jv6[3:6, 0:5]).max())
        worst["standing_J"] = max(worst["standing_J"], off)
    if verbose:
        print(f"  magnitudes: |J| {scale['J']:.3f}  |M| {scale['M']:.3f}  |bias| {scale['bias']:.3f}  "
              f"|JvDot qd| {scale['JvDot_qd']:.3f}")
    return worst


def negative_control(seed: int = 2) -> dict:
    """대조군: 베이스 속도를 0 으로 **안 한** 전체 모델 (qvel 전부 그대로 mj_forward) 의 bias / JvDot·qd 는 aux 와 크게
    달라야 한다 → check_d4 의 비교가 무의미하지 않음을 보인다."""
    model = mm.MitModel()
    model.reset("stand")
    aux = AuxLeg(build_aux_model(model.assets, (LEFT,)), (LEFT,))
    rng = np.random.default_rng(seed)
    _random_state(model, rng)
    m, d = model.m, model.d
    mujoco.mj_forward(m, d)
    b = model.base_body
    ref = aux.evaluate(d.xpos[b].copy(), d.xquat[b].copy(), d.qpos[model.leg_qadr].copy(),
                       d.qvel[model.leg_dof].copy())[LEFT]
    c = model.leg_dof[LEFT]
    jdp = np.zeros((3, m.nv))
    s = int(model.foot_site[LEFT])
    mujoco.mj_jacDot(m, d, jdp, None, d.site_xpos[s].copy(), int(m.site_bodyid[s]))
    return dict(bias=float(np.abs(d.qfrc_bias[c] - ref["bias"]).max()),
                JvDot_qd=float(np.abs(jdp[:, c] @ d.qvel[c] - ref["JvDot_qd"]).max()))


# ----------------------------------------------------------------------------------------------------------------
def report_model(armature: bool = True) -> dict:
    np.set_printoptions(precision=6, suppress=True)
    model = mm.MitModel(armature=armature)
    m = model.m
    print(f"[model] nq {m.nq} nv {m.nv} nu {m.nu} nbody {m.nbody}  timestep {m.opt.timestep}  "
          f"integrator {mujoco.mjtIntegrator(m.opt.integrator).name}  armature {armature}")
    print(f"[mass] total {model.total_mass():.6f} (spec {SPEC_TOTAL})   reduced-body bodies "
          f"{[m.body(b).name for b in model.reduced_bodies]}")
    print(f"[index] leg_qadr L {model.leg_qadr[0]} R {model.leg_qadr[1]}  leg_dof L {model.leg_dof[0]} R {model.leg_dof[1]}"
          f"  leg_act L {model.leg_act[0]} R {model.leg_act[1]}  arm_act L {model.arm_act[0]} R {model.arm_act[1]}")
    print(f"[index] foot_geoms {model.foot_geoms}  hip_location_B L {model.hip_location_B[0]} R {model.hip_location_B[1]}")
    res = {}
    for key in ("stand", "init_yaml"):
        model.reset(key)
        s = model.read_state()
        r = s.reduced
        dm = abs(r.mass - SPEC_MASS)
        dc = np.abs(r.com_offset_B - SPEC_COM_OFFSET).max()
        dI = np.abs(r.inertia_B - SPEC_INERTIA).max()
        res[key] = dict(dm=dm, dc=dc, dI=dI, state=s)
        print(f"\n[{key}] base z {s.torso_pos_W[2]:.6f}")
        print(f"  reduced mass {r.mass:.6f}  |d| {dm:.1e}")
        print(f"  com_offset_B {r.com_offset_B}  spec {SPEC_COM_OFFSET}  max|d| {dc:.1e}")
        print(f"  inertia_B\n{r.inertia_B}\n  max|d| vs spec {dI:.1e}")
        print(f"  foot site L {s.foot_pos_W[LEFT]}  R {s.foot_pos_W[RIGHT]}")
        print(f"  site − base L {s.foot_pos_W[LEFT] - s.torso_pos_W}  R {s.foot_pos_W[RIGHT] - s.torso_pos_W}")
        print(f"  whole-body CoM {model.whole_com_W()}  contact {s.foot_contact}  normal force {s.foot_normal_force}")
        print(f"  x0 {s.x0()}")
    return res


if __name__ == "__main__":
    report_model(armature=True)
    print("\n[D4] full-model scratch vs MjSpec fixed-base aux models (armature on)")
    w = check_d4(3, seed=0, armature=True)
    print("  max abs diff:", {k: f"{v:.2e}" for k, v in w.items()})
    print("\n[D4] armature off")
    w2 = check_d4(3, seed=1, armature=False)
    print("  max abs diff:", {k: f"{v:.2e}" for k, v in w2.items()})
    nc = negative_control()
    print(f"\n[D4 대조군] 베이스 속도 그대로인 전체 모델 vs aux: |dbias| {nc['bias']:.3e}  |dJvDot qd| {nc['JvDot_qd']:.3e}"
          "  (커야 정상 — 베이스 속도 0 이 필요함을 보임)")
    ok = all(v < 1e-9 for v in list(w.values()) + list(w2.values())) and nc["bias"] > 1e-3
    print("\nD4", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
