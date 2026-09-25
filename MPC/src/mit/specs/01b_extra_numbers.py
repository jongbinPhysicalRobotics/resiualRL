"""Extra numbers for the MJCF verification report (run after 01_compare_mjcf_urdf.py).

- SRB numbers at the two candidate keyframes (yaml ankle -0.70 / flat ankle -0.465), feet touching
- effective contact friction and number of cylinder-plane contact points
- a proposed keyframe qpos string, verified by loading a patched model with sites + keyframe
"""
from pathlib import Path
import math
import numpy as np
import mujoco

np.set_printoptions(precision=6, suppress=True, linewidth=150)
MJCF_DIR = Path(__file__).resolve().parents[4] / "mit_humanoid_mjcf"   # specs → mit → src → MPC → 최상위
assets = {f.name: f.read_bytes() for f in MJCF_DIR.rglob("*") if f.is_file()}
robot_xml = (MJCF_DIR / "mit_humanoid.xml").read_text(encoding="utf-8")
scene_xml = (MJCF_DIR / "scene.xml").read_text(encoding="utf-8")

# --- patch: sites at the ground-contact line centre, timestep 0.002, keyframes ---
SITE = '<site name="{n}" pos="0.03 0 -0.04" size="0.005" group="1"/>'
patched = robot_xml
for side in ("left", "right"):
    anchor = f'<body name="{side}_foot" pos="0 0 -0.2785">'
    assert anchor in patched
    # insert the site right after the foot body's opening tag
    patched = patched.replace(anchor, anchor + "\n                " + SITE.format(n=f"{side}_foot_contact_site"))
patched = patched.replace('<option timestep="0.001" integrator="implicitfast"/>',
                          '<option timestep="0.002" integrator="implicitfast"/>')
assets["mit_humanoid.xml"] = patched.encode("utf-8")
m = mujoco.MjModel.from_xml_string(scene_xml, assets)
d = mujoco.MjData(m)
print("patched model: nsite =", m.nsite, "timestep =", m.opt.timestep)

jadr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): int(m.jnt_qposadr[j]) for j in range(m.njnt)}
LEG = ["hip_yaw", "hip_abad", "hip_pitch", "knee", "ankle"]
ARM = ["shoulder_pitch", "shoulder_abad", "shoulder_yaw", "elbow"]
PRE = {"right": (["a01", "a02", "a03", "a04", "a05"], ["a11", "a12", "a13", "a14"]),
       "left": (["a06", "a07", "a08", "a09", "a10"], ["a15", "a16", "a17", "a18"])}


def qpos_for(base_z, leg, arm=(0.0, 0.0, 0.0, -1.65)):
    q = np.zeros(m.nq)
    q[0:3] = [0, 0, base_z]
    q[3] = 1.0
    for side, (lp, ap) in PRE.items():
        for p, jn, v in zip(lp, LEG, leg):
            q[jadr[f"{p}_{side}_{jn}"]] = v
        for p, jn, v in zip(ap, ARM, arm):
            q[jadr[f"{p}_{side}_{jn}"]] = v
    return q


def lowest_cyl_z(g):
    R = d.geom_xmat[g].reshape(3, 3)
    a = R[:, 2]
    r, h = m.geom_size[g][0], m.geom_size[g][1]
    return d.geom_xpos[g][2] - h * abs(a[2]) - r * math.sqrt(max(0.0, 1 - a[2] ** 2))


foot_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] and
              mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) in ("left_foot", "right_foot")]
base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
leg_roots = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_hip_yaw", "right_hip_yaw")]


def in_leg(b):
    while b > 0:
        if b in leg_roots:
            return True
        b = m.body_parentid[b]
    return False


def composite(bodies):
    M = sum(float(m.body_mass[b]) for b in bodies)
    com = sum(float(m.body_mass[b]) * d.xipos[b] for b in bodies) / M
    I = np.zeros((3, 3))
    for b in bodies:
        Ri = d.ximat[b].reshape(3, 3)
        o = d.xipos[b] - com
        I += Ri @ np.diag(m.body_inertia[b]) @ Ri.T + float(m.body_mass[b]) * ((o @ o) * np.eye(3) - np.outer(o, o))
    return M, com, I


all_b = list(range(1, m.nbody))
up_b = [b for b in all_b if not in_leg(b) and m.body_mass[b] > 0]
legs_b = [b for b in all_b if in_leg(b)]

for label, leg in (("yaml pose  (ankle -0.70)", [0, 0, -0.735, 1.2, -0.70]),
                   ("flat foot  (ankle -0.465)", [0, 0, -0.735, 1.2, -0.465])):
    d.qpos[:] = qpos_for(0.0, leg)
    mujoco.mj_forward(m, d)
    zmin = min(lowest_cyl_z(g) for g in foot_geoms)
    bz = -zmin
    d.qpos[:] = qpos_for(bz, leg)
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    Ma, ca, Ia = composite(all_b)
    Mu, cu, Iu = composite(up_b)
    Ml, cl, Il = composite(legs_b)
    print(f"\n===== {label}: base z (feet touching) = {bz:.6f} =====")
    print("qpos =", np.array2string(d.qpos, precision=6, separator=' ', max_line_width=400))
    print(f"whole-body CoM_W = {ca}  (height {ca[2]:.6f}; CoM - base = {ca - d.xpos[base]})")
    print("I_whole about whole CoM, yaw-aligned (=world, yaw 0):\n", Ia, "\n  principal:", np.linalg.eigvalsh(Ia))
    print(f"upper body (torso+arms): mass {Mu:.6f}, bodyComLocation (from base origin, yaw frame) = {cu - d.xpos[base]}, upper CoM_W = {cu}")
    print("I_upper about upper CoM, yaw-aligned:\n", Iu, "\n  principal:", np.linalg.eigvalsh(Iu))
    print(f"legs only: mass {Ml:.6f}, CoM_W = {cl}")
    for side in ("left", "right"):
        s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot_contact_site")
        sm = d.site_xmat[s].reshape(3, 3)
        print(f"  {side}_foot_contact_site: xpos_W = {d.site_xpos[s]}   site-z above floor = {d.site_xpos[s][2]:+.6f}   "
              f"r_site - upperCoM = {d.site_xpos[s] - cu}   site pitch(deg) = {math.degrees(math.atan2(-sm[2,0], math.hypot(sm[0,0], sm[1,0]))):+.3f}")
    # push 1 mm into the floor to see contact points and friction
    d.qpos[2] = bz - 0.001
    mujoco.mj_forward(m, d)
    print(f"  contacts at 1 mm penetration: ncon={d.ncon}")
    for i in range(d.ncon):
        c = d.contact[i]
        print(f"     {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)} - body {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2])}: "
              f"pos={c.pos} dist={c.dist:+.4f} mu={c.friction[0]} (torsional {c.friction[3]}, rolling {c.friction[4]}) dim={c.dim}")

# free-fall settle test of the yaml pose with PD on joints -> how does the tilted foot settle? (short, 1 s)
print("\n===== 1 s settle test, joints held by PD at the yaml offsets (kp/kd = yaml joint_tracking) =====")
kp_leg = np.array([70, 50, 100, 100, 200.0]); kd_leg = np.array([15, 10, 10, 10, 30.0])
kp_arm = np.array([100, 100, 100, 100.0]); kd_arm = np.array([5, 5, 5, 5.0])
for label, leg in (("yaml ankle -0.70", [0, 0, -0.735, 1.2, -0.70]), ("flat ankle -0.465", [0, 0, -0.735, 1.2, -0.465])):
    d.qpos[:] = qpos_for(0.0, leg); mujoco.mj_forward(m, d)
    bz = -min(lowest_cyl_z(g) for g in foot_geoms)
    q0 = qpos_for(bz + 0.002, leg)
    d.qpos[:] = q0; d.qvel[:] = 0; d.time = 0
    act = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a): a for a in range(m.nu)}
    kp = np.zeros(m.nu); kd = np.zeros(m.nu); qt = np.zeros(m.nu)
    for side, (lp, ap) in PRE.items():
        for k, (p, jn) in enumerate(zip(lp, LEG)):
            a = act[f"{p}_{side}_{jn}"]; kp[a], kd[a], qt[a] = kp_leg[k], kd_leg[k], leg[k]
        for k, (p, jn) in enumerate(zip(ap, ARM)):
            a = act[f"{p}_{side}_{jn}"]; kp[a], kd[a], qt[a] = kp_arm[k], kd_arm[k], (0, 0, 0, -1.65)[k]
    while d.time < 1.0:
        qj = d.qpos[7:]; vj = d.qvel[6:]
        d.ctrl[:] = np.clip(kp * (qt - qj) - kd * vj, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        mujoco.mj_step(m, d)
    R = d.xmat[base].reshape(3, 3)
    pitch = math.degrees(math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0])))
    print(f"  {label}: after 1 s base pos = {d.xpos[base]}, base pitch = {pitch:+.2f} deg, ncon = {d.ncon}, "
          f"ankle q = {d.qpos[jadr['a10_left_ankle']]:+.3f}, knee q = {d.qpos[jadr['a09_left_knee']]:+.3f}")
print("done")
