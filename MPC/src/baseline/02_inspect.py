"""G1 모델 구조 덤프: body / joint / dof / actuator / foot site.
사용: .venv/Scripts/python.exe MPC/src/baseline/02_inspect.py
"""
import numpy as np
import mujoco

import g1_model
from paths import SCENE_XML

np.set_printoptions(precision=4, suppress=True, linewidth=140)

m, d = g1_model.load()

print(f"model      : {SCENE_XML}")
print(f"nq={m.nq}  nv={m.nv}  nu={m.nu}  nbody={m.nbody}  ngeom={m.ngeom}  nsite={m.nsite}")
print(f"timestep   : {m.opt.timestep}   integrator={m.opt.integrator}")
print(f"total mass : {mujoco.mj_getTotalmass(m):.3f} kg   -> weight = {mujoco.mj_getTotalmass(m)*9.81:.1f} N")
print()

print("=== JOINTS (name, type, qpos_adr, dof_adr, range) ===")
jtype = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
for j in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    rng = m.jnt_range[j] if m.jnt_limited[j] else np.array([-np.inf, np.inf])
    print(f"  [{j:2d}] {name:28s} {jtype[m.jnt_type[j]]:5s} qpos={m.jnt_qposadr[j]:2d} dof={m.jnt_dofadr[j]:2d} "
          f"range=[{rng[0]:7.3f},{rng[1]:7.3f}]  body={mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j])}")
print()

print("=== ACTUATORS (name, joint, trntype, gaintype, biastype, ctrlrange, forcerange) ===")
for a in range(m.nu):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    jid = m.actuator_trnid[a, 0]
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    print(f"  [{a:2d}] {name:28s} joint={jname:28s} gain={m.actuator_gaintype[a]} bias={m.actuator_biastype[a]} "
          f"ctrl={m.actuator_ctrlrange[a]} frc={m.actuator_forcerange[a]}")
print()

print("=== SITES ===")
for s in range(m.nsite):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s)
    bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.site_bodyid[s])
    print(f"  [{s:2d}] {name:22s} body={bname:24s} local_pos={m.site_pos[s]}")
print()

# keyframe 'stand' 로 세팅 후 발 site 위치 확인
kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
if kid >= 0:
    mujoco.mj_resetDataKeyframe(m, d, kid)
else:
    mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)

print("=== keyframe 'stand' 자세에서 ===")
print(f"  base(pelvis) pos = {d.qpos[0:3]}   quat = {d.qpos[3:7]}")
print(f"  CoM (subtree of world body 0) = {d.subtree_com[0]}")
for fs in ("left_foot", "right_foot"):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, fs)
    print(f"  site {fs:12s} world pos = {d.site_xpos[sid]}")

print()
print("=== 다리 관절만 (leg dof) ===")
for j in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    if name and ("hip" in name or "knee" in name or "ankle" in name):
        print(f"  {name:28s} dof={m.jnt_dofadr[j]:2d}  qpos_adr={m.jnt_qposadr[j]:2d}")
