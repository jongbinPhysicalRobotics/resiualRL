"""Numerical verification of mit_humanoid_mjcf/mit_humanoid.xml against its source URDF.

Run:  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe 01_compare_mjcf_urdf.py

Everything is loaded through MjModel.from_xml_string(text, assets) because the
project path contains Korean characters (MuJoCo cannot open such paths on Windows).

Sections:
  1. MJCF vs URDF (three-way: MJCF model, MuJoCo URDF import, raw URDF XML numbers)
  2. Hand-added parts (freejoint, actuators, timestep, contacts, hand bodies, mesh mirroring)
  3. Foot geometry / initial pose / SRB numbers
  4. Cross-check against the pbrs-humanoid fixed-arms URDF
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=150)

ROOT = Path(__file__).resolve().parents[4]              # specs → mit → src → MPC → 프로젝트 최상위
MJCF_DIR = ROOT / "mit_humanoid_mjcf"                    # 받은 원본 변환본 (저장소에 포함)
URDF_PATH = MJCF_DIR / "urdf_source" / "humanoid_full_sf.urdf"
# 4 절 교차 확인용 — se-hwan/pbrs-humanoid (dev) resources/robots/mit_humanoid/mit_humanoid_fixed_arms.urdf.
# 저장소에 없다. 받아서 MJCF_DIR 에 두면 4 절이 돈다 (없으면 건너뜀).
OTHER_URDF = MJCF_DIR / "mit_humanoid_fixed_arms.urdf"

# reference/config/mit_humanoid/my_controller.yaml : initial_pose
YAML_BASE_Z = 0.679472
YAML_BASE_Z_ALT = 0.625972          # value in the yaml comment "0.625972 / 0.679472"
LEG_OFFSETS = [0.0, 0.0, -0.735, 1.2, -0.70]      # [hip_yaw, hip_abad, hip_pitch, knee, ankle]
ARM_OFFSETS = [0.0, 0.0, 0.0, -1.65]              # [shoulder_pitch, shoulder_abad, shoulder_yaw, elbow]
# reference/config/mit_humanoid/my_controller.yaml : mpc
FOOT_HALF_LENGTH = 0.065
FOOT_HALF_WIDTH = 0.01


def collect_assets(folder: Path) -> dict[str, bytes]:
    assets = {}
    for f in folder.rglob("*"):
        if f.is_file():
            assets[f.name] = f.read_bytes()   # MuJoCo VFS looks files up by basename
    return assets


def quat2mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def rpy2mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def name(m, objtype, i):
    return mujoco.mj_id2name(m, objtype, i)


def body_full_inertia(m, b):
    """Inertia about body CoM expressed in the BODY frame (not the inertial frame)."""
    Ri = quat2mat(m.body_iquat[b])
    return Ri @ np.diag(m.body_inertia[b]) @ Ri.T


def rot_err(Ra, Rb):
    """angle (rad) between two rotation matrices."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


# ---------------------------------------------------------------------------
# 0. load both models
# ---------------------------------------------------------------------------
assets = collect_assets(MJCF_DIR)
scene_xml = (MJCF_DIR / "scene.xml").read_text(encoding="utf-8")
m_mj = mujoco.MjModel.from_xml_string(scene_xml, assets)
d_mj = mujoco.MjData(m_mj)

urdf_text = URDF_PATH.read_text(encoding="utf-8")
# inject the <mujoco> block right after <robot ...>
urdf_mj = re.sub(r'(<robot[^>]*>)',
                 r'\1\n  <mujoco><compiler balanceinertia="true" fusestatic="false" '
                 r'discardvisual="false" strippath="true"/></mujoco>',
                 urdf_text, count=1)
m_ur = mujoco.MjModel.from_xml_string(urdf_mj, assets)
d_ur = mujoco.MjData(m_ur)

# also a URDF import WITHOUT balanceinertia to see whether balancing changed anything
urdf_nb = urdf_mj.replace('balanceinertia="true"', 'balanceinertia="false"')
try:
    m_nb = mujoco.MjModel.from_xml_string(urdf_nb, assets)
    print("URDF import with balanceinertia=false: OK (no triangle-inequality violation)")
except Exception as e:  # noqa: BLE001
    m_nb = None
    print("URDF import with balanceinertia=false FAILED:", e)

print(f"mujoco {mujoco.__version__}")
print(f"MJCF : nbody={m_mj.nbody} njnt={m_mj.njnt} nq={m_mj.nq} nv={m_mj.nv} nu={m_mj.nu} ngeom={m_mj.ngeom} nsite={m_mj.nsite} nkey={m_mj.nkey}")
print(f"URDF : nbody={m_ur.nbody} njnt={m_ur.njnt} nq={m_ur.nq} nv={m_ur.nv} nu={m_ur.nu} ngeom={m_ur.ngeom}")

# ---------------------------------------------------------------------------
# 1. per-body comparison
# ---------------------------------------------------------------------------
print("\n=== 1. BODY COMPARISON (MJCF vs MuJoCo-URDF-import) ===")
mj_bodies = {name(m_mj, mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(1, m_mj.nbody)}
ur_bodies = {name(m_ur, mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(1, m_ur.nbody)}
print("bodies only in MJCF:", sorted(set(mj_bodies) - set(ur_bodies)))
print("bodies only in URDF:", sorted(set(ur_bodies) - set(mj_bodies)))

worst = {"mass": 0, "ipos": 0, "I_diag": 0, "I_full": 0, "pos": 0, "quat(rad)": 0}
worst_who = {k: "" for k in worst}
rows = []
for bn, b in mj_bodies.items():
    u = ur_bodies[bn]
    dm = abs(m_mj.body_mass[b] - m_ur.body_mass[u])
    dip = np.abs(m_mj.body_ipos[b] - m_ur.body_ipos[u]).max()
    dId = np.abs(np.sort(m_mj.body_inertia[b]) - np.sort(m_ur.body_inertia[u])).max()
    dIf = np.abs(body_full_inertia(m_mj, b) - body_full_inertia(m_ur, u)).max()
    if bn == "base":
        dpos = np.abs(m_mj.body_pos[b] - np.array([0, 0, 0.7483])).max()   # MJCF base is lifted (freejoint start)
    else:
        dpos = np.abs(m_mj.body_pos[b] - m_ur.body_pos[u]).max()
    dq = rot_err(quat2mat(m_mj.body_quat[b]), quat2mat(m_ur.body_quat[u]))
    vals = {"mass": dm, "ipos": dip, "I_diag": dId, "I_full": dIf, "pos": dpos, "quat(rad)": dq}
    rows.append((bn, vals))
    for k, v in vals.items():
        if v > worst[k]:
            worst[k], worst_who[k] = v, bn
print(f"{'body':22s} {'mass':>10s} {'ipos':>10s} {'I_diag':>10s} {'I_full':>10s} {'pos':>10s} {'quat':>10s}")
for bn, v in rows:
    print(f"{bn:22s} {v['mass']:10.2e} {v['ipos']:10.2e} {v['I_diag']:10.2e} {v['I_full']:10.2e} {v['pos']:10.2e} {v['quat(rad)']:10.2e}")
print("MAX:", {k: f"{v:.2e} ({worst_who[k]})" for k, v in worst.items()})
tol = {"mass": 1e-6, "ipos": 1e-6, "I_diag": 1e-4, "I_full": 1e-4, "pos": 1e-6, "quat(rad)": 1e-6}
viol = [(bn, k, v[k]) for bn, v in rows for k in v if v[k] > tol[k]]
print("VIOLATIONS (mass/geom > 1e-6, inertia > 1e-4):", viol if viol else "none")

# --- 1b. raw URDF XML numbers vs MJCF (independent of MuJoCo's URDF importer) ---
print("\n=== 1b. RAW URDF <inertial> vs MJCF (inertia about CoM, link frame) ===")
root = ET.fromstring(urdf_text)
urdf_links = {}
for link in root.findall("link"):
    inert = link.find("inertial")
    if inert is None:
        continue
    o = inert.find("origin")
    xyz = np.array([float(x) for x in o.get("xyz").split()])
    rpy = np.array([float(x) for x in o.get("rpy", "0 0 0").split()])
    mass = float(inert.find("mass").get("value"))
    I = inert.find("inertia")
    ixx, ixy, ixz = float(I.get("ixx")), float(I.get("ixy")), float(I.get("ixz"))
    iyy, iyz, izz = float(I.get("iyy")), float(I.get("iyz")), float(I.get("izz"))
    Im = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    R = rpy2mat(*rpy)
    urdf_links[link.get("name")] = (mass, xyz, R @ Im @ R.T)
worst_raw = 0.0
for bn, b in mj_bodies.items():
    mass, xyz, Ifull = urdf_links[bn]
    dm = abs(mass - m_mj.body_mass[b])
    dc = np.abs(xyz - m_mj.body_ipos[b]).max()
    dI = np.abs(Ifull - body_full_inertia(m_mj, b)).max()
    ev = np.linalg.eigvalsh(Ifull)
    tri_ok = (ev[0] + ev[1] >= ev[2] - 1e-12)
    worst_raw = max(worst_raw, dI)
    flag = "" if (dm < 1e-9 and dc < 1e-9 and dI < 1e-7) else "   <-- differs"
    print(f"{bn:22s} dmass={dm:8.1e} dcom={dc:8.1e} dI_full={dI:8.1e} eig={ev} triangle_ok={tri_ok}{flag}")
print(f"max |I_urdf_raw - I_mjcf| = {worst_raw:.2e} (URDF values quoted to 1e-7, so this is rounding of the diagonalisation)")

# --- 1c. joints ---
print("\n=== 1c. JOINT COMPARISON ===")
mj_j = {name(m_mj, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m_mj.njnt)}
ur_j = {name(m_ur, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m_ur.njnt)}
print("joints only in MJCF:", sorted(set(mj_j) - set(ur_j)), " only in URDF:", sorted(set(ur_j) - set(mj_j)))
urdf_joint_xml = {j.get("name"): j for j in root.findall("joint")}
wj = 0.0
for jn, j in mj_j.items():
    if jn not in ur_j:
        continue
    u = ur_j[jn]
    same_type = m_mj.jnt_type[j] == m_ur.jnt_type[u]
    da = np.abs(m_mj.jnt_axis[j] - m_ur.jnt_axis[u]).max()
    dr = np.abs(m_mj.jnt_range[j] - m_ur.jnt_range[u]).max()
    dp = np.abs(m_mj.jnt_pos[j] - m_ur.jnt_pos[u]).max()
    lim = urdf_joint_xml[jn].find("limit")
    eff = float(lim.get("effort"))
    lo, hi = float(lim.get("lower")), float(lim.get("upper"))
    dr_raw = max(abs(m_mj.jnt_range[j][0] - lo), abs(m_mj.jnt_range[j][1] - hi))
    body_of = name(m_mj, mujoco.mjtObj.mjOBJ_BODY, m_mj.jnt_bodyid[j])
    wj = max(wj, da, dr, dp, dr_raw)
    print(f"{jn:26s} body={body_of:18s} type_same={same_type} axis={m_mj.jnt_axis[j]} range={m_mj.jnt_range[j]} "
          f"d_axis={da:.1e} d_range={dr:.1e} d_range_raw={dr_raw:.1e} d_pos={dp:.1e} effort={eff}")
print(f"max joint diff = {wj:.2e}")

# --- 1d. kinematics at qpos0 (relative to base) ---
print("\n=== 1d. WORLD KINEMATICS at qpos0 (body/geom frames relative to base) ===")
mujoco.mj_forward(m_mj, d_mj)
mujoco.mj_forward(m_ur, d_ur)
bb_mj = mj_bodies["base"]
bb_ur = ur_bodies["base"]
wk = 0.0
for bn, b in mj_bodies.items():
    u = ur_bodies[bn]
    p_mj = d_mj.xpos[b] - d_mj.xpos[bb_mj]
    p_ur = d_ur.xpos[u] - d_ur.xpos[bb_ur]
    dpos = np.abs(p_mj - p_ur).max()
    dq = rot_err(d_mj.xmat[b].reshape(3, 3), d_ur.xmat[u].reshape(3, 3))
    wk = max(wk, dpos, dq)
    if bn in ("left_foot", "right_foot", "left_hand", "right_hand", "left_lower_arm", "right_lower_arm"):
        print(f"{bn:16s} pos_rel_base MJCF={p_mj} URDF={p_ur} dpos={dpos:.1e} drot={dq:.1e}")
print(f"max body-frame kinematic diff (all bodies) = {wk:.2e}")

# geoms: match per body by (type, size) after sorting
print("\n--- geoms per body (collision geoms: contype/conaffinity != 0; visual meshes: contype=conaffinity=0) ---")
wg = 0.0
n_geom_mismatch = 0
for bn, b in mj_bodies.items():
    u = ur_bodies[bn]
    g_mj = [g for g in range(m_mj.ngeom) if m_mj.geom_bodyid[g] == b]
    g_ur = [g for g in range(m_ur.ngeom) if m_ur.geom_bodyid[g] == u]
    key_mj = sorted(g_mj, key=lambda g: (int(m_mj.geom_type[g]), int(m_mj.geom_contype[g]), tuple(m_mj.geom_pos[g])))
    key_ur = sorted(g_ur, key=lambda g: (int(m_ur.geom_type[g]), int(m_ur.geom_contype[g]), tuple(m_ur.geom_pos[g])))
    if len(key_mj) != len(key_ur):
        print(f"{bn}: geom count differs MJCF={len(key_mj)} URDF={len(key_ur)}")
        n_geom_mismatch += 1
        continue
    for gm, gu in zip(key_mj, key_ur):
        dsz = np.abs(m_mj.geom_size[gm] - m_ur.geom_size[gu]).max()
        dp = np.abs((d_mj.geom_xpos[gm] - d_mj.xpos[bb_mj]) - (d_ur.geom_xpos[gu] - d_ur.xpos[bb_ur])).max()
        dr = rot_err(d_mj.geom_xmat[gm].reshape(3, 3), d_ur.geom_xmat[gu].reshape(3, 3))
        same = (m_mj.geom_type[gm] == m_ur.geom_type[gu]) and (m_mj.geom_contype[gm] == m_ur.geom_contype[gu])
        # mesh vertex comparison (checks the mirroring scale): compare bounding boxes of the mesh data
        dmesh = 0.0
        if m_mj.geom_type[gm] == mujoco.mjtGeom.mjGEOM_MESH:
            mid_mj, mid_ur = m_mj.geom_dataid[gm], m_ur.geom_dataid[gu]
            v_mj = m_mj.mesh_vert[m_mj.mesh_vertadr[mid_mj]: m_mj.mesh_vertadr[mid_mj] + m_mj.mesh_vertnum[mid_mj]]
            v_ur = m_ur.mesh_vert[m_ur.mesh_vertadr[mid_ur]: m_ur.mesh_vertadr[mid_ur] + m_ur.mesh_vertnum[mid_ur]]
            if v_mj.shape == v_ur.shape:
                dmesh = np.abs(v_mj - v_ur).max()
            else:
                dmesh = float("nan")
            # mesh pos offset (MuJoCo re-centres meshes; mesh_pos must match too)
            dmesh = max(dmesh, np.abs(m_mj.mesh_pos[mid_mj] - m_ur.mesh_pos[mid_ur]).max(),
                        rot_err(quat2mat(m_mj.mesh_quat[mid_mj]), quat2mat(m_ur.mesh_quat[mid_ur])))
        wg = max(wg, dsz, dp, dr, 0.0 if math.isnan(dmesh) else dmesh)
        if not same or dsz > 1e-6 or dp > 1e-6 or dr > 1e-6 or (not math.isnan(dmesh) and dmesh > 1e-6):
            n_geom_mismatch += 1
        print(f"{bn:18s} type={int(m_mj.geom_type[gm])} contype={int(m_mj.geom_contype[gm])} size={m_mj.geom_size[gm]} "
              f"d_size={dsz:.1e} d_pos={dp:.1e} d_rot={dr:.1e} d_mesh={dmesh:.1e} same_type={same}")
print(f"max geom diff = {wg:.2e}; mismatching geoms = {n_geom_mismatch}")

# ---------------------------------------------------------------------------
# 2. hand-added parts
# ---------------------------------------------------------------------------
print("\n=== 2. HAND-ADDED PARTS ===")
print("timestep", m_mj.opt.timestep, " integrator", int(m_mj.opt.integrator), "(mjINT_IMPLICITFAST =",
      int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST), ")")
print("gravity", m_mj.opt.gravity)
fj = [j for j in range(m_mj.njnt) if m_mj.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
print("free joints:", [(name(m_mj, mujoco.mjtObj.mjOBJ_JOINT, j), "body=" + name(m_mj, mujoco.mjtObj.mjOBJ_BODY, m_mj.jnt_bodyid[j]),
                        "qposadr", int(m_mj.jnt_qposadr[j])) for j in fj])
print("qpos order (joint -> qposadr):")
for j in range(m_mj.njnt):
    print(f"   {name(m_mj, mujoco.mjtObj.mjOBJ_JOINT, j):26s} qposadr={int(m_mj.jnt_qposadr[j]):2d} dofadr={int(m_mj.jnt_dofadr[j]):2d}")
print("actuators:")
bad_act = []
for a in range(m_mj.nu):
    jid = m_mj.actuator_trnid[a][0]
    jn = name(m_mj, mujoco.mjtObj.mjOBJ_JOINT, jid)
    eff = float(urdf_joint_xml[jn].find("limit").get("effort"))
    cr = m_mj.actuator_ctrlrange[a]
    ok = (abs(cr[0] + eff) < 1e-12 and abs(cr[1] - eff) < 1e-12 and m_mj.actuator_ctrllimited[a]
          and m_mj.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED and abs(m_mj.actuator_gainprm[a][0] - 1) < 1e-12
          and m_mj.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_NONE)
    if not ok:
        bad_act.append(name(m_mj, mujoco.mjtObj.mjOBJ_ACTUATOR, a))
    print(f"   {name(m_mj, mujoco.mjtObj.mjOBJ_ACTUATOR, a):26s} joint={jn:26s} ctrlrange={cr} urdf_effort={eff} "
          f"actfrcrange={m_mj.jnt_actfrcrange[jid]} gear={m_mj.actuator_gear[a][0]} ok={ok}")
print("actuator problems:", bad_act if bad_act else "none")
print("joint damping   :", m_mj.dof_damping[6:])
print("joint armature  :", m_mj.dof_armature[6:])
print("joint frictionls:", m_mj.dof_frictionloss[6:])
print("URDF rotor_inertia per joint (comment + <dynamics rotor_inertia>):")
for jn, jx in urdf_joint_xml.items():
    dyn = jx.find("dynamics")
    if dyn is not None:
        print(f"   {jn:26s} damping={dyn.get('damping')} friction={dyn.get('friction')} rotor_inertia={dyn.get('rotor_inertia')}")
print("geom solref/solimp (first collision geom):")
for g in range(m_mj.ngeom):
    if m_mj.geom_contype[g] != 0:
        print(f"   geom {name(m_mj, mujoco.mjtObj.mjOBJ_GEOM, g)} body={name(m_mj, mujoco.mjtObj.mjOBJ_BODY, m_mj.geom_bodyid[g])} "
              f"solref={m_mj.geom_solref[g]} solimp={m_mj.geom_solimp[g]} friction={m_mj.geom_friction[g]} "
              f"priority={m_mj.geom_priority[g]} condim={m_mj.geom_condim[g]} margin={m_mj.geom_margin[g]}")
        break
floor = mujoco.mj_name2id(m_mj, mujoco.mjtObj.mjOBJ_GEOM, "floor")
print(f"floor friction={m_mj.geom_friction[floor]} priority={m_mj.geom_priority[floor]} condim={m_mj.geom_condim[floor]} "
      f"solref={m_mj.geom_solref[floor]} solimp={m_mj.geom_solimp[floor]}")
print("collision geoms (contype!=0):")
for g in range(m_mj.ngeom):
    if m_mj.geom_contype[g] != 0:
        print(f"   body={name(m_mj, mujoco.mjtObj.mjOBJ_BODY, m_mj.geom_bodyid[g]):18s} type={int(m_mj.geom_type[g])} "
              f"size={m_mj.geom_size[g]} pos={m_mj.geom_pos[g]} quat={m_mj.geom_quat[g]} friction={m_mj.geom_friction[g]}")
# hand bodies
for hn in ("left_hand", "right_hand"):
    b = mj_bodies[hn]
    print(f"{hn}: mass={m_mj.body_mass[b]} inertia={m_mj.body_inertia[b]} ipos={m_mj.body_ipos[b]} "
          f"njnt={m_mj.body_jntnum[b]} (MuJoCo accepts zero inertia for a body with no joints)")
    # world position of that point mass at qpos0
    print(f"   point mass world pos at qpos0 = {d_mj.xipos[b]}  (hand body origin {d_mj.xpos[b]})")
# mesh scale check: mesh names and whether the right-side mesh is mirrored
print("mesh mirroring check (right-side visual meshes use scale y=-1; compare vertex y sign to left):")
for ln, rn in (("left_hip_yaw1", "left_hip_yaw"), ("left_leg_upper1", "left_leg_upper"), ("left_foot1", "left_foot"),
               ("left_shoulder11", "left_shoulder1")):
    li = mujoco.mj_name2id(m_mj, mujoco.mjtObj.mjOBJ_MESH, ln)
    ri = mujoco.mj_name2id(m_mj, mujoco.mjtObj.mjOBJ_MESH, rn)
    vl = m_mj.mesh_vert[m_mj.mesh_vertadr[li]: m_mj.mesh_vertadr[li] + m_mj.mesh_vertnum[li]]
    vr = m_mj.mesh_vert[m_mj.mesh_vertadr[ri]: m_mj.mesh_vertadr[ri] + m_mj.mesh_vertnum[ri]]
    # mesh_pos: centring offset MuJoCo applied; recover original vertices
    ol = vl @ quat2mat(m_mj.mesh_quat[li]).T + m_mj.mesh_pos[li]
    orr = vr @ quat2mat(m_mj.mesh_quat[ri]).T + m_mj.mesh_pos[ri]
    print(f"   {ln:16s} bbox={ol.min(0)}..{ol.max(0)}   {rn:16s} bbox={orr.min(0)}..{orr.max(0)}")

# ---------------------------------------------------------------------------
# 3. foot geometry / initial pose / SRB
# ---------------------------------------------------------------------------
print("\n=== 3. FOOT GEOMETRY, INITIAL POSE, SRB ===")
foot_geom = {}
for g in range(m_mj.ngeom):
    bn = name(m_mj, mujoco.mjtObj.mjOBJ_BODY, m_mj.geom_bodyid[g])
    if bn in ("left_foot", "right_foot") and m_mj.geom_contype[g] != 0:
        foot_geom[bn] = g
for bn, g in foot_geom.items():
    R = quat2mat(m_mj.geom_quat[g])
    print(f"{bn} collision cylinder: pos={m_mj.geom_pos[g]} axis(in foot frame)={R[:, 2]} radius={m_mj.geom_size[g][0]} half_len={m_mj.geom_size[g][1]}")


def cylinder_lowest_z(m, d, g):
    """lowest surface point of a cylinder geom (world z)."""
    R = d.geom_xmat[g].reshape(3, 3)
    a = R[:, 2]
    r, h = m.geom_size[g][0], m.geom_size[g][1]
    c = d.geom_xpos[g]
    return c[2] - h * abs(a[2]) - r * math.sqrt(max(0.0, 1.0 - a[2] ** 2))


def set_pose(m, d, base_z, leg=LEG_OFFSETS, arm=ARM_OFFSETS):
    d.qpos[:] = 0
    d.qpos[0:3] = [0, 0, base_z]
    d.qpos[3:7] = [1, 0, 0, 0]
    order = {}
    for j in range(m.njnt):
        order[name(m, mujoco.mjtObj.mjOBJ_JOINT, j)] = int(m.jnt_qposadr[j])
    for side, pre in (("right", ["a01", "a02", "a03", "a04", "a05"]), ("left", ["a06", "a07", "a08", "a09", "a10"])):
        for k, (p, jn) in enumerate(zip(pre, ["hip_yaw", "hip_abad", "hip_pitch", "knee", "ankle"])):
            d.qpos[order[f"{p}_{side}_{jn}"]] = leg[k]
    for side, pre in (("right", ["a11", "a12", "a13", "a14"]), ("left", ["a15", "a16", "a17", "a18"])):
        for k, (p, jn) in enumerate(zip(pre, ["shoulder_pitch", "shoulder_abad", "shoulder_yaw", "elbow"])):
            d.qpos[order[f"{p}_{side}_{jn}"]] = arm[k]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)


def foot_report(label, m, d):
    bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    print(f"--- {label}: base z = {d.xpos[bb][2]:.6f}")
    out = {}
    for bn, g in foot_geom.items():
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)
        R = d.geom_xmat[g].reshape(3, 3)
        a = R[:, 2]
        zmin = cylinder_lowest_z(m, d, g)
        Rf = d.xmat[b].reshape(3, 3)
        pitch = math.degrees(math.atan2(-Rf[2, 0], math.sqrt(Rf[0, 0] ** 2 + Rf[1, 0] ** 2)))
        # candidate site positions in foot frame
        cands = {"(0.03,0,-0.03) axis centre": np.array([0.03, 0, -0.03]),
                 "(0.03,0,-0.04) axis centre - r": np.array([0.03, 0, -0.04]),
                 "(0,0,0) foot origin": np.zeros(3)}
        print(f"   {bn}: origin_W={d.xpos[b]} cyl_centre_W={d.geom_xpos[g]} axis_W={a} foot_pitch_deg={pitch:+.3f} "
              f"lowest_z={zmin:+.6f}")
        for cn, cp in cands.items():
            pw = d.xpos[b] + Rf @ cp
            print(f"       site cand {cn:32s} world={pw}  base_z - site_z = {d.xpos[bb][2] - pw[2]:.6f}")
        out[bn] = zmin
    return out


mujoco.mj_forward(m_mj, d_mj)
foot_report("qpos0 (all joints 0, base z from XML 0.7483)", m_mj, d_mj)
set_pose(m_mj, d_mj, YAML_BASE_Z)
z_at_yaml = foot_report("yaml initial_pose, base z 0.679472", m_mj, d_mj)
set_pose(m_mj, d_mj, YAML_BASE_Z_ALT)
foot_report("yaml comment alt, base z 0.625972", m_mj, d_mj)
# base z so that the lowest cylinder point is exactly on the floor
set_pose(m_mj, d_mj, 0.0)
zmin0 = min(cylinder_lowest_z(m_mj, d_mj, g) for g in foot_geom.values())
base_z_touch = -zmin0
print(f"\nbase z with cylinder surface exactly at z=0 (initial pose) = {base_z_touch:.6f}")
set_pose(m_mj, d_mj, base_z_touch)
foot_report("initial pose, feet exactly touching", m_mj, d_mj)
# contact check via MuJoCo itself
for dz in (0.0, -0.001, 0.005):
    set_pose(m_mj, d_mj, base_z_touch + dz)
    cons = [(name(m_mj, mujoco.mjtObj.mjOBJ_GEOM, c.geom1), name(m_mj, mujoco.mjtObj.mjOBJ_BODY, m_mj.geom_bodyid[c.geom2]), round(float(c.dist), 6))
            for c in d_mj.contact[: d_mj.ncon]]
    print(f"   MuJoCo contacts at base z = {base_z_touch + dz:.6f}: ncon={d_mj.ncon} {cons}")

# pose with the foot flat (sole parallel to ground): ankle = -(hip_pitch + knee)
flat_ankle = -(LEG_OFFSETS[2] + LEG_OFFSETS[3])
leg_flat = LEG_OFFSETS[:4] + [flat_ankle]
set_pose(m_mj, d_mj, 0.0, leg=leg_flat)
zmin_flat = min(cylinder_lowest_z(m_mj, d_mj, g) for g in foot_geom.values())
set_pose(m_mj, d_mj, -zmin_flat, leg=leg_flat)
print(f"\nFor reference: ankle that makes the foot flat = {flat_ankle:.3f} rad (yaml uses -0.70); base z for touching = {-zmin_flat:.6f}")
foot_report("foot-flat variant", m_mj, d_mj)

# ---------- SRB numbers at the yaml initial pose (base z irrelevant except CoM height) ----------
set_pose(m_mj, d_mj, YAML_BASE_Z)
torso = mj_bodies["base"]
leg_roots = [mj_bodies["left_hip_yaw"], mj_bodies["right_hip_yaw"]]


def is_desc(m, b, anc):
    while b > 0:
        if b == anc:
            return True
        b = m.body_parentid[b]
    return b == anc


def in_leg(b):
    return any(is_desc(m_mj, b, r) for r in leg_roots)


total_mass = float(m_mj.body_mass[1:].sum())
leg_mass = float(sum(m_mj.body_mass[b] for b in range(1, m_mj.nbody) if in_leg(b)))
upper_mass = total_mass - leg_mass
print(f"\ntotal mass = {total_mass:.6f} kg; legs = {leg_mass:.6f} kg ({100 * leg_mass / total_mass:.2f} %), one leg = {leg_mass / 2:.6f}; "
      f"torso+arms (reference bodyMass) = {upper_mass:.6f} kg")
print("per-body masses:", {name(m_mj, mujoco.mjtObj.mjOBJ_BODY, b): round(float(m_mj.body_mass[b]), 6) for b in range(1, m_mj.nbody)})


def composite(m, d, bodies):
    M = sum(float(m.body_mass[b]) for b in bodies)
    com = sum(float(m.body_mass[b]) * d.xipos[b] for b in bodies) / M
    I = np.zeros((3, 3))
    for b in bodies:
        Ri = d.ximat[b].reshape(3, 3)
        Ib = Ri @ np.diag(m.body_inertia[b]) @ Ri.T
        o = d.xipos[b] - com
        I += Ib + float(m.body_mass[b]) * ((o @ o) * np.eye(3) - np.outer(o, o))
    return M, com, I


all_bodies = list(range(1, m_mj.nbody))
upper_bodies = [b for b in all_bodies if not in_leg(b) and m_mj.body_mass[b] > 0]
M_all, com_all, I_all = composite(m_mj, d_mj, all_bodies)
M_up, com_up, I_up = composite(m_mj, d_mj, upper_bodies)
print(f"\nWHOLE BODY at yaml initial pose (base z {YAML_BASE_Z}): mass={M_all:.6f}  CoM_W={com_all}  (CoM height {com_all[2]:.6f}, "
      f"CoM - base = {com_all - d_mj.xpos[torso]})")
print("  I_whole about CoM, world(=yaw-aligned, yaw=0) frame:\n", I_all)
print(f"UPPER BODY (torso+arms; reference bodyMass/bodyInertia/bodyComLocation): mass={M_up:.6f}")
print(f"  bodyComLocation (yaw frame, from torso origin) = {com_up - d_mj.xpos[torso]}")
print("  bodyInertia about upper CoM, yaw-aligned frame:\n", I_up)
print("  principal moments whole:", np.linalg.eigvalsh(I_all), " upper:", np.linalg.eigvalsh(I_up))
# reference also has the q=0 body-frame version (fillReducedBodyMassProperties)
d0 = mujoco.MjData(m_mj)
mujoco.mj_forward(m_mj, d0)
M0, com0, I0 = composite(m_mj, d0, upper_bodies)
print(f"UPPER BODY at q=0 (fillReducedBodyMassProperties): mass={M0:.6f} com_B={com0 - d0.xpos[torso]}\n  I_B=\n", I0)
# torso-only numbers (fillBodyMassProperties, unused path)
print(f"torso-only body_inertia diag = {m_mj.body_inertia[torso]} ipos = {m_mj.body_ipos[torso]}")
# hip locations (firstJointLocationFromBase = body_pos of hip_yaw body)
print("hipLocationFromBody: left =", m_mj.body_pos[mj_bodies["left_hip_yaw"]], " right =", m_mj.body_pos[mj_bodies["right_hip_yaw"]])
# whole-body subtree_com from MuJoCo as a cross check
set_pose(m_mj, d_mj, YAML_BASE_Z)
print("MuJoCo subtree_com[base] =", d_mj.subtree_com[torso], " (should equal CoM_W above)")
# foot positions relative to base / CoM at init pose
for bn in ("left_foot", "right_foot"):
    b = mj_bodies[bn]
    site = d_mj.xpos[b] + d_mj.xmat[b].reshape(3, 3) @ np.array([0.03, 0, -0.04])
    print(f"{bn}: site(0.03,0,-0.04)_W = {site}; r = site - upperCoM = {site - com_up}; site - base = {site - d_mj.xpos[torso]}")

# ---------------------------------------------------------------------------
# 4. cross-check with pbrs-humanoid fixed-arms URDF
# ---------------------------------------------------------------------------
print("\n=== 4. CROSS-CHECK vs mit_humanoid_fixed_arms.urdf (pbrs-humanoid) ===")
if not OTHER_URDF.exists():
    print(f"skip: {OTHER_URDF} 없음 (위 주석의 URL 에서 받으면 된다)")
    raise SystemExit(0)
other = ET.fromstring(OTHER_URDF.read_text(encoding="utf-8"))
other_mass = {}
for link in other.findall("link"):
    inert = link.find("inertial")
    if inert is not None:
        other_mass[link.get("name")] = float(inert.find("mass").get("value"))
print("other total mass =", round(sum(other_mass.values()), 6))
for bn in mj_bodies:
    om = other_mass.get(bn)
    ours = float(m_mj.body_mass[mj_bodies[bn]])
    print(f"   {bn:18s} ours={ours:9.6f} other={om if om is not None else float('nan'):9.6f} diff={(ours - om) if om is not None else float('nan'):+.6f}")
print("other links not in ours:", sorted(set(other_mass) - set(mj_bodies)))
other_joints = {j.get("name"): j for j in other.findall("joint")}
for jn in ("left_hip_yaw", "left_hip_abad", "left_hip_pitch", "left_knee", "left_ankle", "right_hip_yaw", "right_hip_abad"):
    j = other_joints[jn]
    print(f"   other joint {jn:16s} origin xyz={j.find('origin').get('xyz'):28s} rpy={j.find('origin').get('rpy'):22s} axis={j.find('axis').get('xyz')} type={j.get('type')}")
print("done")
