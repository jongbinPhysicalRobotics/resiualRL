# 04 — Reference low-level layer: robot params, state reading, leg/arm torque control, initialization

Scope: everything between MuJoCo and the MPC/gait layer of `ispaik06/convex-mpc-biped` (local copy:
`C:\Users\백종빈\Desktop\4-2\residual RL\reference`, abbreviated `ref/` below), specialised to
`RobotType::MIT_HUMANOID`. The MPC QP, gait scheduler, swing-foot planner and contact manager are
out of scope except where they hand values to this layer (they are referenced by file:line where
the boundary is).

All line numbers are for the files as they exist in `ref/` today. "Code wins" over `ref/docs`
and over the header defaults; every place where they differ is listed in section 13.

---------------------------------------------------------------------------------------------------

## 0. Conventions used in this document

Frames
- `W` : MuJoCo world frame. z up. Gravity `model.gravity = -9.81` (yaml `model.gravity`,
  `ref/config/mit_humanoid/my_controller.yaml:14`, read at `ref/My_Controller/src/ControllerConfig.cpp:230`).
- `T` : torso body frame = MuJoCo frame of the base body named `"torso"` in the reference spec
  (`ref/sim/src/models/MitHumanoidSpec.cpp:6`). Rotation `R_WT = data.xmat[torso]` (row-major 3x3),
  origin `data.xpos[torso]` (body frame origin, NOT the body COM, NOT `subtree_com`).
- `B` : "reduced-body" / yaw-aligned frame: `R_WB = Rz(psi)` where `psi = atan2(R_WT[1,0], R_WT[0,0])`
  (`ref/sim/src/setupRobotParams.cpp:327-330,484-486`). Origin is irrelevant for rotations; for
  `bodyComLocation` the origin is the torso body origin (`setupRobotParams.cpp:490`).
- `F` : foot end-effector frame = the MuJoCo **site** `left_foot_contact_site` / `right_foot_contact_site`
  (`MitHumanoidSpec.cpp:10,19`), because `model.foot_end_effector_source: site`
  (yaml line 13). `R_WF = data.site_xmat[site]` (`ref/sim/src/MujocoCheaterStateReader.cpp:292-305`).
- `Rz(a)` = `[[cos a, -sin a, 0],[sin a, cos a, 0],[0,0,1]]` (`ref/common/include/Utilities/MatrixUtils.h:11-21`).
- Quaternions are `(w, x, y, z)` both in MuJoCo (`xquat`, `body_quat`, `body_iquat`) and in the
  Eigen constructor used by the reference (`MujocoCheaterStateReader.cpp:114-118`).

Vector orderings
- Leg joint order (per leg, 5 DOF): `[hip_yaw, hip_abad, hip_pitch, knee, ankle]`
  (`MitHumanoidSpec.cpp:12-16, 22-26`; yaml comment line 123).
- Arm joint order (per arm, 4 DOF): `[shoulder_pitch, shoulder_abad, shoulder_yaw, elbow]`
  (`MitHumanoidSpec.cpp:34-37, 43-46`; yaml comment line 127).
- Leg index: `legs[0] = Left`, `legs[1] = Right` (order of the spec list, `MitHumanoidSpec.cpp:8,18`);
  same for arms. Everything indexed "per leg" in this document uses that order.
- MPC wrench vector `u ∈ R^12 = [F_L(3), F_R(3), M_L(3), M_R(3)]`, world frame, **ground reaction
  acting on the robot** (`ref/My_Controller/src/MPCFormulation.cpp:61-62`, `My_Controller.cpp:1319-1332`).
- MPC state `x ∈ R^13 = [roll, pitch, yaw, p_x, p_y, p_z, ω_x, ω_y, ω_z, v_x, v_y, v_z, g]`
  (`My_Controller.cpp:656-675`).
- Full torque vector `tau ∈ R^{nu}` indexed by **MuJoCo actuator id** (`ref/common/src/Robot/RobotModel.cpp:196-207`).

Units: SI (m, s, rad, N, N·m, kg). Angles in rad.

---------------------------------------------------------------------------------------------------

## 1. Configuration values consumed by this layer (MIT yaml)

File: `ref/config/mit_humanoid/my_controller.yaml`. Defaults live in
`ref/My_Controller/include/MyController/ControllerConfig.h` and `ref/robot/include/robot/InitialPoseConfig.h`;
the yaml overrides all of them (parsing: `ControllerConfig.cpp:199-597`, `InitialPoseConfig.cpp:48-69`,
`JointTrackingConfig.cpp:62-78`).

| key | value (MIT yaml) | yaml line | header default | used at |
|---|---|---|---|---|
| `model.xml_path` | `models/mit_humanoid/scene.xml` | 11 | — | `SimulationRunner.cpp:235,257` |
| `model.auxiliary_xml_path` | `models/mit_humanoid/mit_humanoid.xml` | 12 | = xml_path | `LegSwingDynamicsProvider.cpp:745-748` |
| `model.foot_end_effector_source` | `site` | 13 | — | everywhere `FootEndEffectorSource::Site` |
| `model.gravity` | `-9.81` | 14 | -9.81 | `x0[12]` (`My_Controller.cpp:673`), fallback force (`:1038`) |
| `mpc.iterations_between_solve` | `7` | 24 | 10 | `My_Controller.cpp:438,918-919` |
| `swing.natural_frequency` | `[151, 151, 110]` | 41 | `[10,10,10]` | `My_Controller.cpp:435,1353-1356` |
| `swing.kd_diag` | `[25, 25, 25]` | 42 | `[15,15,18]` | `My_Controller.cpp:436,1360` |
| `swing.enable_stance_foot_yaw_hold` | `true` | 51 | false | `My_Controller.cpp:1338` |
| `swing.roll_kp / roll_kd` | `0 / 0` | 64-65 | 0/0 | `My_Controller.cpp:1365-1366` |
| `swing.pitch_kp / pitch_kd` | `300 / 18` | 66-67 | 0/0 | `:1367-1368` |
| `swing.yaw_kp / yaw_kd` | `305 / 18` | 68-69 | 0/0 | `:1369-1370` |
| `swing.stance_yaw_kp / stance_yaw_kd` | `20 / 4` | 70-71 | 0/0 | `:1345-1346` |
| `startup.post_init_standing_settle_time` | `1.0` | 107 | 2.0 | `My_Controller.cpp:426`, `SimulationRunner.cpp:271` |
| `joint_tracking.leg.kp` | `[70, 50, 100, 100, 200]` | 124 | (required) | `RobotRunner.cpp:168-172` |
| `joint_tracking.leg.kd` | `[15, 10, 10, 10, 30]` | 125 | (required) | same |
| `joint_tracking.arm.kp` | `[100, 100, 100, 100]` | 128 | (required) | `RobotRunner.cpp:178-182` |
| `joint_tracking.arm.kd` | `[5, 5, 5, 5]` | 129 | (required) | same |
| `initial_pose.base_position_W` | `[0, 0, 0.679472]` | 133 | none | `My_Controller.cpp:683-689` (target seed only) |
| `initial_pose.base_rpy_W` | `[0, 0, 0]` | 134 | none | same |
| `initial_pose.leg_initialization_time` | `2.0` s | 135 | 2.0 | `RobotRunner.cpp:66-67` |
| `initial_pose.arm_initialization_time` | `1.0` s | 136 | 1.0 | `RobotRunner.cpp:68-69` |
| `initial_pose.leg_joint_offsets` | `[0, 0, -0.735, 1.2, -0.70]` | 137 | `[0,0,-0.65,0.80,-0.35]` | `LegPosInitializer.cpp:89-92` |
| `initial_pose.arm_joint_offsets` | `[0, 0, 0, -1.65]` | 138 | `[0,0,0,-0.65]` | `ArmPosInitializer.cpp:89-92` |
| `requested_locomotion_mode` | `walking` | 1 | walking | `LocomotionFSM.cpp:72-88` |

Simulation config `ref/config/simulation.yaml`: `physics_timestep_sec: 0.002`,
`physics_integrator: implicitfast`, `viewer_sync_hz: 60`. These **overwrite** `model.opt.timestep`
and `model.opt.integrator` of the loaded XML (`ref/sim/src/SimulationConfig.cpp:96-104`, called at
`SimulationRunner.cpp:268`). So the controller tick is `dt = 0.002 s` (500 Hz) regardless of the XML.

Note: `InitialPoseConfig` (robot/) and `ControllerConfig.initialPose` (My_Controller/) parse the same
yaml keys twice; the initializers use the former, the body-target seed uses the latter. Same values.

---------------------------------------------------------------------------------------------------

## 2. Model loading and what the simulation starts from

`SimulationRunner::init()` (`ref/sim/src/SimulationRunner.cpp:224-278`):
1. `mj_loadXML(scene.xml)` (`:257`), `mj_makeData` (`:262`).
2. `configureSimulationModel(model)` → timestep 0.002, integrator implicitfast (`:268`).
3. `mj_forward(model, data)` (`:269`).
4. No keyframe is ever loaded by the runner (`mj_resetDataKeyframe` is only used by
   `test/gait_swing_hold`). **Initial state = `model.qpos0`** (all joints at their XML zero, base at the
   XML body position). The yaml `initial_pose.base_position_W` is *not* applied to the simulation
   (section 9.5). The robot is dropped from the XML base height and the joints are driven by PD to the
   offsets during the first 2 s.

`RobotParams` are built on the **first control tick**, not in `init()` (`SimulationRunner.cpp:386-407`).

---------------------------------------------------------------------------------------------------

## 3. RobotParams (`ref/common/include/Robot/RobotParams.h`, built in `ref/sim/src/setupRobotParams.cpp`)

### 3.1 Struct (RobotParams.h:25-86)

```
JointGroupParams: q_idx[], qd_idx[], actuator_idx[], motorTauMax[], damping[] (unused), dryFriction[] (unused)
LegParams       : side, joints, hipLocationFromBody (Vec3, torso frame), jointLocation_offsets (unused)
ArmParams       : side, joints
FixedJointParams: name, q_idx, qd_idx, actuator_idx, qDefault, kp, kd      # MIT: none (MitHumanoidSpec.cpp:49 "{}")
RobotParams     : roboType, nq, nv, nu, bodyMass, bodyInertia (3x3, frame B), bodyComLocation (Vec3, frame B,
                  torso-origin -> reduced-body COM), default_qpos (= model.qpos0, size nq), legs[], arms[], fixedJoints[]
```

### 3.2 Joint groups (`setupRobotParams.cpp:82-105`)

For each `(joint_name, actuator_name)` pair of the spec, in spec order:
```
jointId    = mj_name2id(JOINT, joint_name)      # throws if missing (requireId, :18-26)
actuatorId = mj_name2id(ACTUATOR, actuator_name)
q_idx.append(model.jnt_qposadr[jointId])
qd_idx.append(model.jnt_dofadr[jointId])
actuator_idx.append(actuatorId)
motorTauMax[i] = max(|ctrlrange[actuatorId,0]|, |ctrlrange[actuatorId,1]|)   # :100-103, NOT used by the controller
```
`motorTauMax` is only validated (`RobotModel.cpp:65-67`), never applied. Torque limiting is done by
MuJoCo `ctrlrange` clamping only (section 12).

### 3.3 Feet / hands bindings (`setupRobotParams.cpp:175-210`)

```
foot.rootBodyId = model.jnt_bodyid[first leg joint]           # body carrying hip_yaw joint  (:132-140,183)
foot.bodyId     = mj_name2id(BODY, endBody)                    # "left_foot_link"/"right_foot_link"
foot.siteId     = mj_name2id(SITE, endSite)                    # "left_foot_contact_site"/... (required for Site mode, :189-191)
foot.collisionGeomIds = [g for g in geoms if geom_bodyid[g]==foot.bodyId and
                          (geom_contype[g]!=0 or geom_conaffinity[g]!=0 or geom_group[g]==3)]   # :37-51
leg.hipLocationFromBody = model.body_pos[foot.rootBodyId]      # :142-152, 197. Only position (parent-frame
                                                               # offset of the hip_yaw body from the torso), no rotation.
hand.rootBodyId = body of first arm joint; hand.bodyId = "left_forearm_link"/...; hand.siteId = -1 (endSite "")
```
`hipLocationFromBody` is not used by anything in this layer (it is consumed by the swing planner spec).

### 3.4 Reduced body ("SRB") mass properties — which bodies

Both the start-up and the per-tick version loop over **every body id `b` in `1 .. nbody-1` of the
loaded SCENE model** (`setupRobotParams.cpp:414-431, 439-473`) and keep `b` iff
- `body_mass[b] > 0`, and
- `b` is not in a leg subtree (`isInLegSubtree`, `setupRobotParams.cpp:154-173`): not `foot.rootBodyId`
  (the hip_yaw body) of either leg and not a descendant of one.

This is **not** "the torso subtree": *any* other massive body in the scene passes the filter. The
reference scenes that are available (`ref/models/unitree_robots/g1/scene_23dof.xml:21-39`, `h1/scene.xml`)
contain 4 mocap debug-marker bodies (`debug_reduced_body_com`, `debug_body_target`,
`debug_left_touchdown_target`, `debug_right_touchdown_target`, children of world) whose geoms have no
`density` attribute, so they get default-density mass: 0.0373 + 0.0373 + 0.0697 + 0.0697 = **0.214 kg**
(compiled with mujoco 3.12). They would be counted, and `updateDebugVisualization`
(`SimulationRunner.cpp:748-801`, step 11 of section 8) moves them every tick via `mocap_pos` to the
reduced-body COM, the body target and the ground-level touchdown targets (`My_Controller.cpp:1190-1252`);
their `xipos` follows one tick late (next `mj_step`'s kinematics). If the (undistributed) MIT scene.xml
follows the same pattern, the reference `bodyMass/bodyComLocation/bodyInertia` include ≈0.21 kg of marker
mass; the touchdown markers sit ≈0.75 m below the COM, adding roughly `2·0.0697·0.75² ≈ 0.08 kg·m²` to
Ixx/Iyy (≈ +40 % of Iyy 0.178) and moving the COM down by a few mm. Whether the MIT scene has them is
unknown (15.10). **The port must decide explicitly.** Default: the converted `scene.xml` has no extra
bodies, so the reduced body = torso + 8 arm bodies (`*_shoulder, *_shoulder_2, *_upper_arm, *_lower_arm`)
+ 2 hand bodies (`*_hand`, 0.01 kg each) = 14.2275 kg (14.2). If markers are added to the Python scene
for visualisation, give them mass 0 (`density="0"` / `mass="0"` on their geoms) or exclude them
explicitly from the filter. `bodyMass` is **not** the total robot mass.

`fillBodyMassProperties` (`:61-80`, torso-only inertia) is dead code — never called.

### 3.5 Start-up value (`fillReducedBodyMassProperties`, `setupRobotParams.cpp:212-325`), computed once at `q = 0`

Computed in the torso-root frame using only `model.body_pos/body_quat` (i.e. joint angles = 0):
```
for each included body b:
    offset = 0; rot = I; c = b
    while c > 0 and c != torso:                       # walk up to torso, compose fixed transforms at q=0
        offset = body_pos[c] + R(body_quat[c]) @ offset
        rot    = R(body_quat[c]) @ rot
        c = body_parentid[c]
    com_b   = offset + rot @ body_ipos[b]
    M      += mass_b ;  com += mass_b * com_b
com /= M
for each included body b:
    R_i   = rot_b @ R(body_iquat[b])
    I_b   = R_i @ diag(body_inertia[b]) @ R_i.T
    o     = com_b - com
    I    += I_b + mass_b * ((o·o) I3 - o o^T)          # parallel axis about the reduced-body COM
bodyMass = M ; bodyComLocation = com ; bodyInertia = I
```
This value is only alive until the first tick's runtime update (section 3.6) overwrites it, which
happens *before* the controller or the initializers ever read it (`SimulationRunner.cpp:409` runs
right after `setupRobotParams` at `:388-392`). It is therefore never observable by the controller; an
implementation may skip it and directly use 3.6.

### 3.6 Runtime value — `updateReducedBodyMassPropertiesFromData` (`setupRobotParams.cpp:397-491`), **every control tick**

Called at `SimulationRunner.cpp:409`, before `fillCheaterState`, every physics step (500 Hz).
```
M = 0; com_W = 0
for b in included bodies (same filter, :414-421):
    M += body_mass[b]; com_W += body_mass[b] * data.xipos[b]           # xipos = body COM in world
com_W /= M
I_W = 0
for b in included bodies:
    R = data.ximat[b] (3x3 row-major)                                     # inertial frame orientation in world
    I_b_W = R @ diag(body_inertia[b]) @ R.T
    o = data.xipos[b] - com_W
    I_W += I_b_W + body_mass[b] * ((o·o) I3 - o o^T)                       # about com_W, world axes
R_WT = data.xmat[torso]; psi = atan2(R_WT[1,0], R_WT[0,0])              # :475-484
R_WB = Rz(psi); R_BW = R_WB.T
bodyMass        = M
bodyInertia     = R_BW @ I_W @ R_WB                                       # about the reduced-body COM, YAW-ALIGNED frame B
bodyComLocation = R_BW @ (com_W - data.xpos[torso])                       # torso body ORIGIN -> reduced-body COM, in B
```
Points to note:
- Inertia is about the reduced-body COM (upper-body COM), **not** about the torso origin and not
  about the whole-robot COM. It is expressed in `B` (world rotated by −yaw), so roll/pitch of the
  torso and the current arm configuration are baked into the tensor each tick. It is a full 3x3
  (off-diagonals kept).
- Consumption: `MPCFormulation.cpp:50-53,68-70` uses `I_k = Rz(psi_k) @ bodyInertia @ Rz(psi_k)^T`
  per horizon step and `mass = bodyMass`.
- `data.xipos/ximat/xmat/xpos` here are whatever `mj_step` left in `data` (see the staleness note in 4.4).

---------------------------------------------------------------------------------------------------

## 4. State reading — `fillCheaterState` (`ref/sim/src/MujocoCheaterStateReader.cpp:326-385`)

Output struct `CheaterState<double>` (`ref/common/include/SimulationIO.h:123-158`), copied verbatim into
`StateEstimate` by the estimator (`StateEstimator.h:55-66`).

### 4.1 Torso quantities (`:338-344`)

| field | MuJoCo source | notes |
|---|---|---|
| `time` | `data.time` | |
| `torsoPos_W` | `data.xpos[torso]` (`:20-30`) | body frame origin |
| `torsoQuat_W` | `data.xquat[torso]` as `(w,x,y,z)` (`:109-119`) | |
| `torsoLinVel_W` | `mj_objectVelocity(m,d,mjOBJ_BODY,torso,v6,flg_local=0)`, take `v6[3:6]` (`:133-149,166-168`) | linear velocity of the **body frame origin**, world axes (MuJoCo shifts `cvel` from the subtree COM to `xpos`) |
| `torsoAngVel_W` | same call, take `v6[0:3]` (`:151-164`) | torso angular velocity, **world axes** |
| `torsoLinAcc_W`, `torsoAngAcc_W` | `mj_objectAcceleration(..., flg_local=0)` (`:196-224`) | read but **never used** by the controller (grep confirms); may be stale since `cacc` is only refreshed by `mj_rnePostConstraint`. Do not implement. |

### 4.2 Per-leg quantities (`:346-369`), leg order Left, Right

```
q            = data.qpos[q_idx]                     (copyIndexed :279-286)
qd           = data.qvel[qd_idx]
tauEstimate  = data.actuator_force[actuator_idx]    # torque MuJoCo applied on the previous step; unused by controller
footPos_W    = data.site_xpos[siteId]               (readFootEndEffectorPosition :226-240, Site branch)
footVel_W    = mj_objectVelocity(m,d,mjOBJ_SITE,siteId,v6,0)[3:6]   (:242-260 → :133-149)
footEndPos_W = footPos_W ; footEndVel_W = footVel_W  (aliases)
R_WF         = data.site_xmat[siteId] (3x3 row-major)                (:288-323)
contact, contactForce_W, contactNormalForce  = readFootContactInfo   (:63-89), see 4.3
hasContactForce = true ; hasFootFrame = true ; hasFootJacobians = false ; hasLegDynamics = false
```
`Jv_W, JvDot_W, Jw_W, massMatrix, bias` are left as they were (zeros) and are filled later by the
dynamics provider (section 7).

### 4.3 Foot contact detection (`readFootContactInfo`, `:50-89`)

```
info = {contact: False, contactCount: 0, normalForce: 0, force_W: 0}
for i in range(data.ncon):
    c = data.contact[i]
    foot1 = belongs(c.geom1); foot2 = belongs(c.geom2)      # belongs: geom in foot.collisionGeomIds OR geom_bodyid == foot.bodyId  (:50-61)
    if not (foot1 or foot2): continue
    f6 = mj_contactForce(m, d, i)                             # [normal, tangent1, tangent2, torques...] in contact frame
    frame = c.frame.reshape(3,3)                              # rows = contact-frame axes in world (row 0 = normal, geom1 -> geom2)
    F_W_on_geom2 = frame.T @ f6[0:3]
    sign = +1 if foot2 else -1
    info.contact = True; info.contactCount += 1
    info.force_W += sign * F_W_on_geom2                       # = force ON THE FOOT, world (for a foot on the floor: +z ≈ GRF)
    info.normalForce += abs(f6[0])
```
Only `contact` and `contactNormalForce` are consumed downstream (ContactManager, out of scope).

### 4.4 Timing / staleness (important for an exact port)

Loop (`SimulationRunner.cpp:337-343`): `runRobotControl(); mj_step(model, data);`. `mj_step` runs
`mj_forward` on the *pre-step* state and then integrates `qpos/qvel`. It does not recompute
kinematics after integrating. Consequently, at control tick k+1 the reader sees:
- `qpos, qvel, time` : state after step k (fresh),
- `xpos, xquat, xmat, xipos, ximat` (torso pose, reduced-body mass properties of 3.6, `R_WB`/psi),
  `site_xpos, site_xmat` (`footPos_W`, `R_WF`), `cvel`/`subtree_com` (→ `mj_objectVelocity`: torso
  lin/ang velocity, `footVel_W`), contact list + `efc_force` (contact flag, `contactForce`,
  `normalForce`), `actuator_force`, and mass matrix (`mj_fullM`) / `qfrc_bias` / Jacobians of the **main** `data` :
  computed from the state **before** step k (one tick = 2 ms old).
The yaw estimator (5) pairs the fresh `time` with the stale quaternion. On the very first tick
everything is consistent (`init()` ran `mj_forward`, section 2).
A Python port that calls `mj_step` and then reads `data` without `mj_forward` reproduces this exactly;
calling `mj_forward` (or `mj_step1/mj_step2`) before reading is more consistent but deviates from the
reference by one tick. The auxiliary dynamics models (section 7) get the fresh leg `q, qd` but the
stale torso pose.

---------------------------------------------------------------------------------------------------

## 5. State estimator (cheater mode) — yaw unwrap (`ref/common/src/Estimator/StateEstimator.cpp:25-53`)

```
state = copy of cheater_state (StateEstimator.h:55-66; standingFeet zeroed)
R_WT = R(torsoQuat_W)
yaw_wrapped = atan2(R_WT[1,0], R_WT[0,0])
if first call or time not finite or time < last_time - 1e-9:
    yaw_unwrapped = yaw_wrapped ; yawRate_W = 0
else:
    yaw_unwrapped = last_unwrapped + wrapToPi(yaw_wrapped - last_wrapped)      # unwrapAngle, AngleUtils.h:10-14
    yawRate_W = (yaw_unwrapped - last_unwrapped) / (time - last_time)  if dt > 0 else 0
psi = yaw_W_unwrapped                                                            # legacy alias, :47
store last_wrapped, last_unwrapped, last_time
wrapToPi(a) = atan2(sin a, cos a)                                                # AngleUtils.h:6-8
```
`yawRate_W` is not used by this layer. `psi`/`yaw_W_unwrapped` (identical) are used everywhere
`Rz(psi)` appears below and as `x0[2]`.

---------------------------------------------------------------------------------------------------

## 6. Derived body quantities and the 13-state `x0` fed to the MPC

Helpers (`ref/My_Controller/src/My_Controller.cpp:315-332`; identical copies in `SimulationRunner.cpp:49-63`,
`RobotRunner.cpp:13-16`, `SwingFootPlanner.cpp:14`):
```
offset_W  = Rz(yaw_unwrapped) @ bodyComLocation                     # = com_W - torsoPos_W exactly (since 3.6 used the same Rz)
com_W     = torsoPos_W + offset_W                                    # reduced-body (torso+arms) COM, world
comVel_W  = torsoLinVel_W + torsoAngVel_W × offset_W                 # rigid-body assumption: COM fixed to torso (arm joint rates ignored)
```
`quaternionToRollPitch(q)` (`:63-75`), q normalised first:
```
roll  = atan2(2(w x + y z), 1 - 2(x² + y²))
pitch = asin(clamp(2(w y - z x), -1, 1))
```
`buildCurrentMpcState` (`:656-675`):
```
x0[0]     = roll(torsoQuat_W)          # torso ZYX-Euler roll
x0[1]     = pitch(torsoQuat_W)         # torso ZYX-Euler pitch
x0[2]     = yaw_W_unwrapped            # unwrapped, NOT wrapped
x0[3:6]   = com_W                      # reduced-body COM position, world
x0[6:9]   = torsoAngVel_W              # torso angular velocity, WORLD axes (comment: "approximate reduced-body as rigid body")
x0[9:12]  = comVel_W                   # reduced-body COM velocity, world
x0[12]    = model.gravity = -9.81
```
Sanity vs docs: `ref/docs/mpc_frame_convention.md:35-41` says the same (orientation, COM pos in W,
torso ω in W, COM vel in W, g).

Body-target seeding at first `runController` (`updateBodyTarget`, `:677-697`), used by the MPC
reference (out of scope) but it is where `initial_pose.base_position_W` enters:
```
if initialPose.hasBasePose:            # true for MIT yaml
    nominalPosition_W = base_position_W + Rz(base_rpy_W[2]) @ bodyComLocation    # (:356-361) → z ≈ 0.679472 + bodyComLocation.z
    euler_W           = base_rpy_W
else:
    nominalPosition_W = x0[3:6]; euler_W = (0, 0, x0[2])
nominalHeight_W = nominalPosition_W[2]
```
So the MPC height target is `0.679472 + bodyComLocation.z` (bodyComLocation from the tick when the
target is seeded, i.e. the first tick after leg initialization).

---------------------------------------------------------------------------------------------------

## 7. Leg dynamics provider (`ref/sim/src/LegSwingDynamicsProvider.cpp`)

Mode used by the runner: `Lazy` (`SimulationRunner.cpp:398`) → per tick it computes exactly what
`LegDynamicsRequest{swingLegDynamics, standingFootJacobians}` asks (`:589-601`):
- `swingLegDynamics = (locomotion mode == Walking)`, `standingFootJacobians = (mode == Standing)`
  (`LocomotionFSM.cpp:285-286`); before `MyController` is initialised the request is
  `standingFootJacobians = true` because walking is requested with settle time > 0
  (`My_Controller.cpp:488-504`).
- `stateEstimate.standingFeet.zero()` every update (`:596`) — the combined Jacobian is only valid on
  ticks where it was requested.

### 7.1 Auxiliary models (one per leg + one "standing" model)

Built with the MuJoCo spec API from `auxiliary_xml_path` (= `mit_humanoid.xml`, robot only, no floor)
(`:325-434` per-leg, `:436-562` standing):
1. delete the **named** actuators, sensors, keyframes (`:342-344`, `:457-459`; `deleteElementsByType`
   `:29-47` enumerates names and skips unnamed elements, so unnamed actuators/sensors/keys would
   survive — no effect on M, bias, J; the converted MJCF names all 18 motors and has no sensors/keys.
   A Python `MjSpec` port may delete by type);
2. delete the free joint (`:346-349`, `:461-464`; its name is found by type in the full model,
   `freeJointName :55-62`, called at `:316`) → the torso becomes a **fixed** child of world;
3. per-leg model: delete the *other* leg's hip_yaw root body subtree (`:351-361`) and both arm root
   body subtrees (`:363-366`) → torso + one 5-joint leg, `nv_aux = 5`;
   standing model: delete both arm root bodies only (`:466-469`), keep both legs, `nv_aux = 10`;
4. `mj_compile`, `mj_makeData`; look up torso body, foot body, foot site, foot collision geoms, and the
   leg joints' `jnt_qposadr / jnt_dofadr` in the aux model (`:387-426`, `:489-556`); `qvelIndex` is in
   spec joint order.
   In the standing model `combinedQvelIndex` = Left leg dofs then Right leg dofs (params leg order),
   `footRowLegIndices = {leftLegIndex, rightLegIndex}` (`:499-502`).
5. The aux models **never** go through `configureSimulationModel` (only the main model does,
   `SimulationRunner.cpp:268`): they keep the XML `<option>` unchanged (converted MJCF: timestep 0.001 —
   irrelevant, only `mj_forward` is used — and the XML gravity, which enters `qfrc_bias`).

Per tick (`updateSwingLegDynamics :603-689`, `updateStandingFootJacobians :691-743`):
```
aux.model.body_pos[torso]  = torsoPos_W ;  aux.model.body_quat[torso] = torsoQuat_W (w,x,y,z)    # assignBasePose :73-85
                                              # = main data.xpos/xquat[torso] as read (STALE, 4.4)
aux.data.qpos[leg q idx] = leg q ; aux.data.qvel[leg qd idx] = leg qd                             # = main data.qpos/qvel (FRESH); no other dofs exist
mj_forward(aux.model, aux.data)
```
Base velocity in the aux model is identically zero (it has no base DOFs).

### 7.2 Swing dynamics outputs (per leg, Site branch `:632-652`, then `:679-687`)

```
mj_jacSite(aux, jacp(3×nv_aux), jacr(3×nv_aux), siteId)
Jv_W    = jacp[:, qvelIndex]      (3×5)   translational Jacobian of the site, world
Jw_W    = jacr[:, qvelIndex]      (3×5)   rotational Jacobian, world
mj_jacDot(aux, jacDotp, None, point = site_xpos, body = site_bodyid)
JvDot_W = jacDotp[:, qvelIndex]   (3×5)   time derivative of Jv_W (base velocity = 0 in aux)
mj_fullM(aux) → massMatrix = M_full[qvelIndex][:, qvelIndex]   (5×5)   # fixed-base leg-only joint-space inertia:
                                                                          #   includes dof_armature, EXCLUDES damping and the implicitfast damping term
bias    = aux.data.qfrc_bias[qvelIndex]   (5)                      # Coriolis + centrifugal + GRAVITY of the leg with a stationary torso
                                                                   #   (MuJoCo qfrc_bias includes gravity — XML gravity of the aux model; excludes damping/friction)
hasFootFrame = hasFootJacobians = hasLegDynamics = true
```
`Jw_W` is the **leg-columns-only** (3×5) rotational Jacobian of a fixed-base model, so `Jw_W @ qd_leg`
is the foot angular velocity **relative to the torso**, expressed in world axes (torso ω is NOT
included). Section 10 uses exactly this product for all foot-rate damping terms; never replace it by the
full-model 3×nv Jacobian times the full `qvel`.

Equivalences for a Python port using the *full* model instead of aux models — these hold only if the
full model is evaluated with **the same torso pose and the same leg q** as the aux model, i.e. torso pose
= stale `data.xpos/xquat[torso]` (as read, 4.4) and leg `q, qd` = fresh `data.qpos/qvel`:
- `Jv_W, Jw_W` columns for leg dofs are then identical (kinematic).
- `massMatrix` = leg-dof block of the full-model mass matrix — identical, and in fact independent of the
  torso pose (composite-rigid-body block of a subtree does not depend on the base being free).
- `JvDot_W` and `bias` additionally require zero base velocity: in the aux model the torso is
  stationary, so they contain no torso-velocity coupling.
- Neither the main `data` after `mj_step` (its mass matrix via `mj_fullM`, `qfrc_bias`, `mj_jacSite`, `site_xpos` all belong to
  the **previous** q) nor an `mj_forward` on the fresh full state (fresh torso pose) reproduces the aux
  mix exactly; only J, JvDot and bias are affected (M is torso-pose invariant).
Exact reproduction: build the aux models (python `mujoco.MjSpec` supports the same deletions), or use a
scratch `MjData` of the full model with root pose forced to the stale torso `xpos/xquat`,
`qvel[0:6] = 0`, leg `qpos/qvel` fresh, then `mj_forward`. Do not read `mj_fullM(m, d)` / `d.qfrc_bias` /
`mj_jacSite` of the main data after `mj_step`.

### 7.3 Standing combined Jacobians (`:691-743`, `:256-295`)

```
standingFeet.Jv_W, Jw_W : 6 × 10, zero-initialised
row block 0 (rows 0..2) = Left foot site jacobian (columns = combinedQvelIndex, i.e. [L(5), R(5)])
row block 1 (rows 3..5) = Right foot site jacobian
standingFeet.hasFootJacobians = true
```
(The other leg's columns in each row block are zero automatically.)

---------------------------------------------------------------------------------------------------

## 8. Per-tick order (one physics step = one control tick, 500 Hz)

`SimulationRunner::runRobotControl` (`SimulationRunner.cpp:385-443`) followed by `mj_step`:
```
1  if first tick: setupRobotParams (3.2-3.5); resize states; create provider(Lazy);
                  RobotRunner.init(params, model.opt.timestep, &userCommand)   (RobotRunner.cpp:48-71:
                     RobotModel, LegController, ArmController, LegPosInitializer(2.0 s), ArmPosInitializer(1.0 s),
                     joint tracking gains from yaml)
2  updateReducedBodyMassPropertiesFromData(model, data, bindings, params)      (3.6)           :409
3  fillCheaterState(...)                                                         (4)             :411
4  stateEstimator.update(cheater → stateEstimate)                                (5)             :412
5  userCommand = keyboard / headless schedule                                                    :413-415
6  robotRunner.prepareController(stateEstimate)                                                  :416
      → if _legInitializationComplete (flag from the PREVIOUS tick's run()):
           MyController.prepareController() = syncLocomotionFSM() = FSM.update(t, …)  (RobotRunner.cpp:73-82,
           My_Controller.cpp:476-486)   — SYNC #1
7  provider.update(stateEstimate, robotRunner.legDynamicsRequest())              (7)             :420
      (request = the one produced by sync #1; before MyController is initialised: the pre-init
       request {standingFootJacobians}, My_Controller.cpp:488-504)
8  robotRunner.run(stateEstimate, robotCommand)                                  (9-12)          :427
      → MyController.runController() calls syncLocomotionFSM() AGAIN (My_Controller.cpp:1386)
         with the same t — SYNC #2 — then horizonClock.sync, x0, …
9  dashboard publish                                                                             :428-438
10 applyFixedJointCommands (none for MIT)                                                        :439
11 updateDebugVisualization (mocap markers only)                                                 :440
12 applyRobotCommand: data.ctrl[i] = clamp(tau[i], ctrlrange) if ctrllimited else tau[i]         :441, :803-818
13 writeHeadlessTelemetry (optional CSV)                                                         :442
14 mj_step(model, data)                                                                          :341
```

Double FSM sync (after initialisation the FSM is updated **twice per tick at the same time stamp**):
- Sync #1 (step 6) runs before the provider, sync #2 (inside `runController`) after it. In walking mode
  sync #2 is normally a no-op (no transition can fire twice at the same t; `justTransitioned=false`, so
  no second clock/swing reset).
- A transition must be detected in sync #1 so that the provider computes the new mode's data on the
  same tick: e.g. on the StandingSettle→Walking tick sync #1 switches the request to
  `swingLegDynamics`, the provider fills the per-leg Jacobians/dynamics, and `writeLegCommands` can run.
  A port that syncs only inside `runController` would lack per-leg Jacobians on that tick (throw).
- In `BrakingToStanding` (interactive mode only) the settle samples and settle ticks are pushed twice
  per tick (`LocomotionFSM.cpp:161-268`): the duplicated samples are identical (same state), so the
  time-windowed means are unchanged, but `_brakingSettleTicks` advances by 2 per tick, halving the
  effective hold time. Not reached in the walking-only demos.
- On the init tick (first tick where `legInit` becomes true, 9.1) step 6 is skipped (the flag is only
  set inside `run()`); the provider uses the pre-init request `{standingFootJacobians}`, and
  `runController` first runs `initializeController` (FSM created in `StandingSettle`, mode Standing,
  request `standingFootJacobians`) and then syncs once.

`RobotRunner::run` (`RobotRunner.cpp:105-152`):
```
setupStep(state)                       # :188-261: zero all leg/arm commands+data (mode ← JointPd, gains ← 0, ff ← 0),
                                       #   copy q, qd, tauEstimate; if hasFootJacobians → setLegCartesianData(footPos_W, footVel_W, Jv, JvDot, Jw);
                                       #   if hasLegDynamics → setLegDynamicsData(M, bias)
armInit = armPosInitializer.IsInitialized(armController)      # always called, every tick (writes arm qDes) :113
apply arm joint gains (kp 100, kd 5 diag) to every arm command                                       :115-117
legInit = legPosInitializer.IsInitialized(legController)      # always called, every tick (writes leg qDes) :120
if not legInit:  for each leg: apply leg joint gains; mode = JointPd                                  :127-138
else:            MyController.runController()                 # overwrites leg commands (section 10)   :141
legController.setEnabled(true); armController.setEnabled(true)
composeCommand: tau = 0(nu); legController.updateCommand(tau); armController.updateCommand(tau)      :154-158
```
Note: after leg init the initializer keeps writing `qDes` into the leg commands every tick, but
`runController` overwrites `mode` (and, for JointPd it never falls back), so those `qDes` are inert.
The **JointPd fallback is never used after initialization**: `LegController::zeroCommand` resets every
leg to `JointPd` with zero gains each tick (`LegController.cpp:131-136,35-50`); a leg whose mode is not
rewritten by `writeLegCommands` would get zero torque (cannot happen for Left/Right).

---------------------------------------------------------------------------------------------------

## 9. Initialization sequence

### 9.1 Timeline (sim time from `t = 0`, dt = 0.002)

| phase | duration | legs | arms | who |
|---|---|---|---|---|
| A. Joint PD init | ticks with `data.time ∈ {0, 0.002, …, 1.996}` s (**999 ticks**) | JointPd spline → `qpos0 + leg_joint_offsets` | JointPd spline (1.0 s) → `qpos0 + arm_joint_offsets`, then hold | `RobotRunner.run`, initializers |
| B. StandingSettle | from the tick at `t = 1.998` s for 1.0 s | `JointTorque` from standing MPC via combined 6x10 Jacobian (9.6) | same PD hold | `MyController`, `LocomotionFSM` state `StandingSettle` |
| C. Walking | from the first tick with `t − 1.998 ≥ 1.0` (≈ 2.998 s) | StanceWrench / SwingFoot (10) | same PD hold | `MyController` |

- Tick numbering: the first control tick runs at `data.time = 0` (control runs before `mj_step`,
  `SimulationRunner.cpp:337-343`), and `IsInitialized` does `curr_time += dt` **before** returning
  `curr_time >= T_end` (`LegPosInitializer.cpp:36-57`). With `dt = double(float(0.002))` the 1000th call
  returns true (`curr_time = 1000·float(0.002) ≈ 2.0000001 ≥ 2.0`); that call is the tick at
  `data.time = 1.998` s. On that same tick `RobotRunner::run` skips the PD branch and calls
  `runController()` (`RobotRunner.cpp:119-142`). So leg joint PD acts on 999 ticks (t = 0 … 1.996).
- `MyController::initializeController()` runs on that **first tick where `legInit` is true**
  (t = 1.998; `RobotRunner.cpp:141` → `My_Controller.cpp:1374-1376, 393-474`). On that tick it creates
  `HorizonClock` and `LocomotionFSM` at t = 1.998 (`:411, 424-433`), solves MPC #0 (`_iteration == 0`)
  and writes standing `JointTorque` commands. The 1.0 s settle therefore counts from 1.998 s:
  `StandingSettle → Walking` fires on the first tick with `t − 1.998 ≥ 1.0` (`LocomotionFSM.cpp:230-234`),
  i.e. t ≈ 2.998 s (± one tick, depending on the rounding of the accumulated `data.time`).
- FSM: `initialState = StandingSettle` (walking requested, settle 1.0 > 0; `LocomotionFSM.cpp:72-77`);
  in `update` (`:230-234`) when `time - stateStart >= 1.0` → `Walking` (`:236-240`), with
  `resetGaitClock = resetSwingState = true` (`:280-281`) → `HorizonClock.reset(time)`, swing state reset,
  contact manager reset (`My_Controller.cpp:555-577, 534-553`).
- Dynamics request per phase: A and B → `standingFootJacobians`; C → `swingLegDynamics`.
- Arm init completion flag (`_armInitializationComplete`) is informational only.

### 9.2 Joint PD law (`LegController::computeJointPdTorque`, `LegController.cpp:250-259`; arms identical `ArmController.cpp:204-221`)

```
tau_leg = tauFeedForward(=0) + diag(kp) @ (qDes - q) + diag(kd) @ (qdDes(=0) - qd)
kp_leg = [70, 50, 100, 100, 200], kd_leg = [15, 10, 10, 10, 30]     (per leg, joint order of section 0)
kp_arm = [100,100,100,100],        kd_arm = [5,5,5,5]
```
No gravity compensation, no feed-forward, no Cartesian term (`hasCartesianData` is never set for arms).

### 9.3 Joint trajectory spline (`LegPosInitializer.cpp:27-99`, `ArmPosInitializer.cpp` identical, `BSplineBasicDynamic.h`)

Type `BS_BasicDyn<T, DEGREE=3, NUM_MIDDLE=1, CONST_LEVEL_INI=2, CONST_LEVEL_FIN=2>` (`LegPosInitializer.h:28`),
one spline over all leg joints stacked `[L(5), R(5)]` (dim 10; arms dim 8).
```
On first IsInitialized call (spline not yet built), using the CURRENT q of that tick (t = 0, i.e. qpos0):
  ini[0:dim] = q_now ; ini[dim:3dim] = 0 (vel, acc)
  fin[0:dim] = default_qpos[q_idx] (= model.qpos0 at those addresses) ; fin[dim:3dim] = 0
  mid = fin[0:dim]
  applyConfiguredJointOffsets (InitialPoseConfig.h:19-36): for each limb, for joint j < min(dof, len(offsets)):
        fin[flat+j] += offsets[j] ; mid[flat+j] = fin[flat+j]
  SetParam(ini, fin, [mid], T_end)              # T_end = 2.0 (legs) / 1.0 (arms)
Every call (including the first): curr_time += dt ; jpos = getCurvePoint(curr_time) (clamped to [0, T_end]);
  for each leg/arm command: tauFeedForward = 0, qDes = jpos slice, qdDes = 0
  return curr_time >= T_end
```
Resulting clamped cubic B-spline (dim-wise): knots `[0,0,0,0, T/4, T/2, 3T/4, T,T,T,T]`
(`_CalcKnot :111-124`: 11 knots, 3 interior), 7 control points
`[q0, q0, q0, qf, qf, qf, qf]` (zero velocity and acceleration at both ends make cp1=cp2=q0 and
cp4=cp5=qf, `_CalcConstrainedCPoints :297-345`; the middle point is `qf`, `_CalcCPoints :347-353`).
Python: `scipy.interpolate.BSpline(t=knots, c=cps, k=3)(clamp(u,0,T))` reproduces `getCurvePoint`.
`dt` is passed through a `float` parameter (`LegPosInitializer.h:13`): `_dt = double(float(0.002)) = 0.0020000000949949026`,
so `curr_time` reaches ≥ 2.0 exactly at call 1000 = the tick at `data.time = 1.998` (9.1). (With exact
double 0.002 the accumulated sum may be 1.9999999999999998 at call 1000 and complete one tick later —
irrelevant in practice, but this is why.) Likewise the arm spline reaches its end value on call 500
(t = 0.998).

Final targets for MIT: legs `[0, 0, -0.735, 1.2, -0.70]` (both legs, added to qpos0 which is 0 for all
joints in a model whose joint zeros are the XML zeros), arms `[0, 0, 0, -1.65]`.

### 9.4 What the controller does with the state during phase A

Nothing: for t = 0 … 1.996 `MyController` is neither prepared nor run — no MPC, no FSM, no balance
control (`RobotRunner.cpp:73-82, 105-152`). The free-floating robot starts at `qpos0` (XML base height)
and is held only by the leg joint PD (kp `[70,50,100,100,200]`, kd `[15,10,10,10,30]`) along the 2 s
spline, with no gravity compensation. The provider still computes the standing 6×10 Jacobians (pre-init
request, `My_Controller.cpp:488-504`), but nothing uses them. The MPC target seed (6) is taken on the
first tick of phase B (t = 1.998) from `initial_pose.base_position_W` and the `bodyComLocation` of that tick.

### 9.5 Initial base pose

Not set by the controller. `base_position_W = [0,0,0.679472]` only seeds `nominalPosition_W`/`nominalHeight_W`
(section 6). The yaml comment "Floating-base pose from the scene keyframe" refers to a keyframe in the
reference's (unavailable) scene XML that the runner never loads; presumably that keyframe had the base
at z = 0.679472 with the legs at the offsets above and the feet flat on the floor (see section 14.5 —
the converted MJCF does not satisfy this).

### 9.6 Standing-mode leg command (phase B, and Standing mode in general) — `writeStandingLegCommands` (`My_Controller.cpp:1254-1295`)

```
require standingFeet.hasFootJacobians and shapes 6×10
u = _stanceWrenchWorld (12)                         # latest MPC solution, [F_L, F_R, M_L, M_R], world, GRF on robot
footForces_W  = [-u[0:3]; -u[3:6]]  (6)             # force the feet must apply on the ground
footMoments_W = [-u[6:9]; -u[9:12]] (6)
tau_all_legs (10) = standingFeet.Jv_W.T @ footForces_W + standingFeet.Jw_W.T @ footMoments_W
for leg in (L, R): command.mode = JointTorque ; command.tauFeedForward = tau_all_legs[offset:offset+5]
                   forceFeedForward_W = momentFeedForward_W = 0
```
`JointTorque` mode returns `tauFeedForward` unchanged (`LegController.cpp:327-331`). No contact ramp,
no yaw hold, no gravity/bias compensation, no joint PD. The MPC is solved with the same cadence as in
walking (`maybeUpdateMpc`, every 7 ticks); between solves the last wrench is reused.

---------------------------------------------------------------------------------------------------

## 10. Walking leg commands — `writeLegCommands` (`My_Controller.cpp:1297-1372`) and torque laws

Per tick, for each leg (Left, Right), `isStance = contactManager.activeContact(side)` (walking mode,
`:738-746`).

### 10.1 Stance leg (`:1315-1350`, torque law `computeStanceLegJointTorque`, `OperationalSpaceDynamics.cpp:87-94`)

```
mode = StanceWrench
alpha = contactManager.contactRampAlpha(side)  ∈ [0,1]                              (:748-753)
Left : F_cmd = -alpha * u[0:3] ; M_cmd = -alpha * u[6:9]                           (:1321-1326)
Right: F_cmd = -alpha * u[3:6] ; M_cmd = -alpha * u[9:12]                          (:1327-1332)
if enable_stance_foot_yaw_hold and alpha > 0:                                       (:1338-1348)
    M_cmd.z += alpha * yawHoldMoment_W.z          # ONLY the z component is added (:1347)
tau_leg (5) = Jv_W.T @ F_cmd + Jw_W.T @ M_cmd     # LegController.cpp:294-317 → OperationalSpaceDynamics.cpp:93
```
- `Jv_W, Jw_W` are the per-leg **site** Jacobians of section 7.2 (3×5 each, foot contact site, world
  axes). The sign: MPC gives the ground reaction on the robot; the leg is commanded to push the ground
  with the negative (comment `:1319`).
- Nothing else is added: bias/gravity compensation and `tauFeedForward` are commented out
  (`LegController.cpp:302-304, 312-315`). `hasDynamicsData` is not required for stance.
- **5-DOF mapping**: the 6-D wrench is mapped by plain `J^T` (no least squares, no pseudo-inverse, no
  projection). The roll moment `M_x` still enters through `Jw_W.T` (hip_abad and hip_yaw axes have
  world-x components when the leg is not exactly sagittal); whatever `J^T` cannot realise is simply lost.
  With `contact_wrench_model: full_wrench` (yaml lines 26, 33) the MPC is free to command `M_x`; the
  torsional/roll limits are the MPC's business (foot half-width 0.01 m keeps `M_x` tiny).

Stance yaw hold (`computeStanceYawHoldMomentWorld`, `:86-117`; kp 20, kd 4, desiredYaw = `_legRuntime[leg].touchdownYaw_W`):
```
if not (kp>0 or kd>0) or desiredYaw not finite or not hasFootFrame or Jw shape bad: return 0
footX_W  = R_WF[:,0]; footXProj = footX_W - (footX_W·ẑ) ẑ ; return 0 if |footXProj| ≤ 1e-9 ; normalise
desiredX = (cos ψ_d, sin ψ_d, 0)
yawErr   = atan2( ẑ·(footXProj × desiredX), footXProj·desiredX )
yawRate  = ẑ·(Jw_W @ qd)          # Jw_W = aux 3×5 leg-columns Jacobian (7.2): foot yaw rate RELATIVE to the torso, world axes
moment_W = (kp*yawErr - kd*yawRate) * ẑ
```
The yaw hold modifies only `M_cmd.z` (`+= alpha·m_z`, only when enabled and `alpha > 0`), so its torque
contribution is `alpha · m_z · Jw_W[2,:]ᵀ` (third row of `Jw_W` only), with
`m_z = 20·yawErr − 4·ẑ·(Jw_W qd)`. The error uses the absolute (stale, 4.4) `R_WF`; the damping uses the
torso-relative rate.

`touchdownYaw_W[leg]` is per-leg hidden state (`_legRuntime[leg]`), read by both the stance yaw hold and
the swing attitude yaw (10.2):
- **seeded at controller init** (`initializeRuntimeObjects`, `My_Controller.cpp:450-457`) to
  `swingFootYawTargetWorld(side) = base + psiOffset(side, psi_dot)` (10.3) evaluated at t_init = 1.998 —
  a plain sum, **no** `liftAngleNear`, no wrap;
- **re-latched** to `liftAngleNear(base + psiOffset, base)` (10.3) (a) on the first swing tick after
  stance or whenever the swing trajectory is inactive (`:853-854`), and (b) on entering ground-search
  mode or when search mode has no active trajectory (`:831-832`);
- kept unchanged otherwise, in particular through the following stance, and **not** reset by the
  Standing→Walking transition's `resetSwingState` (`:534-553`). A leg that is in stance when walking
  starts therefore uses the init-time value in its stance yaw hold until its first swing.

### 10.2 Swing leg (`:1352-1371`; torque law `computeSwingLegTorque` `LegController.cpp:262-291` → `OperationalSpaceDynamics.cpp:34-84`)

Requires `hasFootData` and `hasDynamicsData` (throws otherwise).
```
mode        = SwingFoot
Λ           = inv( Jv_W @ inv(M) @ Jv_W.T + 1e-9 I3 )            # computeApparentInertia :34-46 (LDLT solves), M = 5×5 leg-only, Jv 3×5
kpCartesian = diag( ω_n² ⊙ diag(Λ) )    ω_n = [151,151,110]        # computeSwingCartesianKp :49-58 → ONLY the diagonal of Λ is used for Kp
kdCartesian = diag(25, 25, 25)                                     # constant, not scaled by Λ           (:436, :1360)
pDes_W, vDes_W, aDes_W = swingTrajectory.position/velocity/acceleration()   (:1357-1359, planner spec)
forceFeedForward_W = 0 (zeroCommand)

F_fb   = forceFeedForward_W + kpCartesian @ (pDes_W - p_W) + kdCartesian @ (vDes_W - v_W)        # :76-77, p_W = footPos_W (site), v_W = footVel_W
a_res  = aDes_W - JvDot_W @ qd                                                                    # :78
tau_osc = Jv_W.T @ F_fb + Jv_W.T @ (Λ @ a_res) + bias                                             # :81-83, full Λ here; bias = gravity+Coriolis of the leg (fixed base)
tau_leg = tau_osc + tauFeedForward                                                                # LegController.cpp:289
tauFeedForward = attitude torque (below), accumulated with "+=" onto the zeroed command (:1361)
```
Swing attitude torque (`SwingAttitudeControl.h:9-84`; roll disabled because kp=kd=0 → skipped `:20,52`):
```
require hasFootFrame, hasFootData, Jw 3×dof
footX, footY, footZ = normalised columns of R_WF ; up = ẑ
ω_foot = Jw_W @ qd      # aux 3×5 leg-columns Jacobian × leg qd: foot ω RELATIVE to the torso, world axes (torso ω NOT included)
pitch: e = atan2( footY·(footZ × up), footZ·up ) ; rate = footY·ω_foot ; moment += (300 e - 18 rate) * footY
yaw  : footXProj = normalise(footX - (footX·up) up) (throw if degenerate); desiredX = (cos ψ_d, sin ψ_d, 0)
       e = atan2( up·(footXProj × desiredX), footXProj·desiredX ) ; rate = up·ω_foot ; moment += (305 e - 18 rate) * up
tau_att = Jw_W.T @ moment
ψ_d = _legRuntime[leg].touchdownYaw_W
```
Both pitch and yaw errors are "level the foot" (pitch) and "point the foot x-axis at ψ_d" (yaw).

### 10.3 Swing/stance yaw target ψ_d (needed by 10.1/10.2; `My_Controller.cpp:891-909`, `SwingYawTarget.h:11-34`)

```
base(psi_dot, cmd) = yaw_W_unwrapped + swing_foot_yaw_lead_scale(1.0) * psi_dot * previewTime
    previewTime = max(0, (0.5 + halfStanceOffset(|[x_dot,y_dot]|)) * stanceTime(0.33))
    halfStanceOffset = 0.26 if speed > 0.65 ; 0.28 if speed > 0.60 ; else 0.37                  (:35-45, yaml 45-49)
psiOffset(side, psi_dot) = clamp(100°/(rad/s) * |psi_dot|, 0, 20°) in rad, applied +to Left if psi_dot>0, −to Right if psi_dot<0, else 0
touchdownYaw_W = liftAngleNear(base + psiOffset, base)  = base + wrapToPi(psiOffset)             # AngleUtils.h:16-18 (swing/search re-latch)
touchdownYaw_W(init) = base + psiOffset                                                          # controller-init seed, no liftAngleNear (10.1)
```
(`psi_dot`, `x_dot`, `y_dot` are the low-pass filtered, clamped user commands `_filteredUserCommand`.)

### 10.4 MPC failure fallback (`maybeUpdateMpc` catch, `:1016-1052`)

If the MPC throws: `u = 0`, then for every leg currently in stance `u[Fz slot] = bodyMass * 9.81 / nStance`
(slot 2 for Left, 5 for Right). Moments stay 0. This `u` is used until the next successful solve.

### 10.5 MPC cadence and wrench latching

`maybeUpdateMpc` runs when `iteration == 0` or `iteration - lastMpcIteration >= 7` (`:918-919`,
`iterations_between_solve`), i.e. every 14 ms. `_stanceWrenchWorld = solution[0:12]`
(`ConvexMPC.cpp:991`) is the first horizon step and is held constant between solves.

---------------------------------------------------------------------------------------------------

## 11. Arm control (`ArmController.cpp:204-221, 224-233`)

Every tick, for both arms, in every phase:
```
tau_arm (4) = tauFeedForward(=0) + diag([100]*4) @ (qDes - q) + diag([5]*4) @ (0 - qd)
qDes = ArmPosInitializer spline output (1.0 s spline from q(t=0) to qpos0[arm] + [0,0,0,-1.65]);
       from call 500 (t = 0.998) on, the spline is clamped at its end, so
       qDes = qpos0[arm] + [0, 0, 0, -1.65] forever
```
No gravity compensation, no Cartesian/hand term (`hasCartesianData` never set), no dependence on
locomotion. Arms are never commanded by `MyController` (it never touches `_armController` commands;
it only passes it to the debug logger).

---------------------------------------------------------------------------------------------------

## 12. Torque composition, clamping, ordering

- `RobotCommand.tau` has size `nu` and is **indexed by MuJoCo actuator id** (`RobotModel.cpp:196-221`,
  `RobotRunner.cpp:154-158`). Leg/arm torques are scattered via `actuator_idx` (looked up by actuator
  name from the spec). Any actuator not in the spec gets 0 (fixed joints: none for MIT).
- `applyRobotCommand` (`SimulationRunner.cpp:803-818`):
  `data.ctrl[i] = clamp(tau[i], ctrlrange[i]) if actuator_ctrllimited[i] else tau[i]`. Throws if
  `tau.size() != nu`.
- No other limiting anywhere (no `motorTauMax`, no rate limit, no low-pass). MuJoCo itself also clamps
  `ctrl` to `ctrlrange` for `ctrllimited` actuators, and, for joints with `actfrclimited`, clamps the
  summed actuator generalized force **`qfrc_actuator`** of that dof to `jnt_actfrcrange`
  (`actuator_force` itself is NOT clamped by it — verified with mujoco 3.12: ctrl 5, actuatorfrcrange ±1
  → `actuator_force = 5`, `qfrc_actuator = 1`). For the converted model (`ctrlrange` = `actuatorfrcrange`,
  gear 1) all three limits coincide, so the explicit clamp is redundant with the model's own limits.
- Effort limits used by the reference are whatever its MJCF declares; for the converted MJCF see 14.3.

---------------------------------------------------------------------------------------------------

## 13. Docs / header defaults vs code — discrepancies (code wins)

| topic | doc / default says | code does |
|---|---|---|
| MPC cadence | `ref/README.md:404` "iterations_between_solve = 10" | yaml `7` (`my_controller.yaml:24`) |
| leg joint offsets | header default `[0,0,-0.65,0.80,-0.35]` (`InitialPoseConfig.h:10`, `ControllerConfig.h:143`) | yaml `[0,0,-0.735,1.2,-0.70]` |
| arm joint offsets | header `[0,0,0,-0.65]` | yaml `[0,0,0,-1.65]` |
| swing ω_n / kd | header `[10,10,10]` / `[15,15,18]` (`ControllerConfig.h:64-65`) | yaml `[151,151,110]` / `[25,25,25]` |
| settle time | header 2.0 (`ControllerConfig.h:130`) | yaml 1.0 |
| "base pose from scene keyframe" (yaml:132, ControllerConfig.h:147) | implies the sim starts there | runner never loads a keyframe; the value only seeds the MPC target |
| `bodyInertia` comment "reduced-body frame B, yaw aligned" (`RobotParams.h:78`) | — | matches code (3.6) |
| `x0[6:9]` "torso angular velocity in world coordinates" (`docs/mpc_frame_convention.md:39`) | — | matches code |
| `hasLegDynamics` required for stance (`LegController.cpp:302-304`) | (was required) | commented out — stance needs only Jacobians |
| bias compensation in stance (`LegController.cpp:312-315`) | (was added) | commented out — none |
| `LegController` fallback "joint PD" | struct default `mode = JointPd` | only used during phase A; never as a runtime fallback |

---------------------------------------------------------------------------------------------------

## 14. MIT Humanoid specifics and the converted MJCF (`C:\Users\백종빈\Desktop\4-2\residual RL\mit_humanoid_mjcf`)

The reference's own MIT MJCF/URDF is intentionally not distributed (`ref/README.md:552,680`; `ref/models/mit_humanoid/`
contains only `.gitkeep`). The converted model was probed with the reference algorithms
(scratch script `probe_mit.py` next to this file); numbers below come from that run (mujoco 3.12).

### 14.1 Name mapping (reference spec → converted MJCF)

| reference (`MitHumanoidSpec.cpp`) | converted `mit_humanoid.xml` |
|---|---|
| base body `torso` | `base` (line 27) |
| free joint | `floating_base` (line 28) — same name; `freeJointName()` finds it by type anyway |
| leg end body `left_foot_link` / `right_foot_link` | `left_foot` / `right_foot` |
| foot site `left_foot_contact_site` / `right_foot_contact_site` | **absent** (`nsite = 0`) — must be added (14.4) |
| `left_hip_yaw_joint`, act `left_hip_yaw` | joint = actuator = `a06_left_hip_yaw` |
| `left_hip_abad_joint` | `a07_left_hip_abad` (axis x) |
| `left_hip_pitch_joint` | `a08_left_hip_pitch` (axis y) |
| `left_knee_joint` | `a09_left_knee` (axis y, range 0..2.2) |
| `left_ankle_joint` | `a10_left_ankle` (axis y) |
| right leg | `a01_right_hip_yaw … a05_right_ankle` |
| `left_shoulder_pitch_joint … left_elbow_joint` | `a15_left_shoulder_pitch, a16_left_shoulder_abad, a17_left_shoulder_yaw, a18_left_elbow` |
| right arm | `a11 … a14` |
| arm end body `left_forearm_link` / `right_forearm_link` | `left_lower_arm` / `right_lower_arm` (hand body `left_hand` exists below it, 0.01 kg) |
| leg root bodies (hip_yaw) | `left_hip_yaw` (id 7), `right_hip_yaw` (id 2) |
| arm root bodies | `left_shoulder` (17), `right_shoulder` (12) |

Converted MJCF indices: `nq=25, nv=24, nu=18, nbody=22`. `qpos[0:7]` free joint; leg/arm hinge
`jnt_qposadr = 7 + jointIndex`, `jnt_dofadr = 6 + jointIndex` in XML order (right leg 7-11 / dof 6-10,
left leg 12-16 / 11-15, right arm 17-20 / 16-19, left arm 21-24 / 20-23). Actuator ids follow the same
XML order (a01→0 … a18→17), so `tau` ordering for MuJoCo is `[R leg(5), L leg(5), R arm(4), L arm(4)]`
while the controller's internal leg order is `[L, R]` — the scatter by `actuator_idx` handles that.

### 14.2 Reduced-body values (converted MJCF, reference algorithm, at q = 0 = qpos0)

```
total mass          24.8886 kg   (legs 10.6611 kg)
bodyMass            14.2275 kg   (torso 8.52 + 8 arm bodies + 2 hands = 5.7075; converted scene has no debug-marker bodies, 3.4)
bodyComLocation_B   [0.012151, 0.002857, 0.113790] m   (torso origin → upper-body COM)
bodyInertia_B       [[0.538649, 0.001511, 0.001121],
                     [0.001511, 0.177667,-0.001133],
                     [0.001121,-0.001133, 0.392504]]  kg·m²   (about upper-body COM, yaw-aligned; arms straight down)
hipLocationFromBody L = [-0.00565, +0.082, -0.05735], R = [-0.00565, -0.082, -0.05735]
```
These change every tick with arm pose (elbow −1.65 rad) and torso tilt; the values above are the
q = 0 seed only.

### 14.3 Actuator/effort limits in the converted MJCF (`ctrlrange` = `actuatorfrcrange`, both present)

`hip_yaw 34, hip_abad 34, hip_pitch 72, knee 144, ankle 68, shoulder_* 34, elbow 55` N·m; `ctrllimited = 1`
for all 18 → `applyRobotCommand` clamps to these. Whether the reference MJCF used the same limits is unknown.

### 14.4 Foot geometry and the missing contact site

- Foot collision geom: one cylinder per foot, radius 0.01, half-length 0.075 along the foot **x** axis
  (`geom quat 0.707 0 0.707 0`), centre at `(0.03, 0, -0.03)` in the foot body frame (`mit_humanoid.xml:54,82`).
  Lowest surface at foot-frame z = −0.04 when the foot is level.
- MPC foot geometry in yaml: `foot_half_length 0.065`, `foot_half_width 0.01` (`my_controller.yaml:18-19`)
  — consistent with a line foot of ≈0.13–0.15 m along x and negligible width, so the reference model
  is likely the same "sf" (single-foot-cylinder) URDF.
- The port must add `<site name="left_foot_contact_site" pos="?" quat="1 0 0 0"/>` (and right) inside
  the foot bodies. `R_WF` must give the foot x-axis (used for yaw and pitch levelling), so the site
  orientation should be identity in the foot body frame. Position candidates: cylinder centre
  `(0.03, 0, -0.03)` or cylinder bottom `(0.03, 0, -0.04)`. The choice shifts `footPos_W`, the swing
  target height (planner's touchdown z) and the MPC moment arm by ≤1 cm; reference placement unknown (15.1).
- `collisionGeomIds` for the foot = the cylinder only (visual mesh geoms have contype/conaffinity 0, group 1).

### 14.5 Joint-zero convention and standing height (probe results)

In the converted model the three pitch-type body rotations of the leg chain (hip_yaw body −10°, hip_abad
+25°, upper_leg −15° about y) sum to 0, so at `q = 0` the foot is level and the leg is straight. With
the yaml offsets `[0,0,-0.735,1.2,-0.70]` the net pitch is `-0.735 + 1.2 - 0.70 = -0.235 rad`:
the foot is **pitched 13.5° toe-up** relative to the torso; the flat-foot ankle value would be `-0.465`.
Standing base heights if the cylinder touches the floor (torso level, converted geometry):
```
[0,0,-0.735,1.2,-0.70]  : foot pitch -0.235 rad, base z ≈ 0.657 (heel end touching)
[0,0,-0.735,1.2,-0.465] : foot level,            base z ≈ 0.648
[0,0,-0.65, 0.80,-0.35] : foot pitch -0.200,     base z ≈ 0.697
```
The yaml expects `base_position_W.z = 0.679472` (comment also mentions `0.625972`). Neither matches
the converted geometry with the yaml offsets, and the yaml ankle offset does not level the foot.
Therefore the reference's MJCF most likely has a different joint-zero convention (e.g. URDF joint
`rpy` origins folded differently) or a different foot site definition. This must be resolved before
expecting the reference numbers (15.2).

### 14.6 Sample swing-OSC magnitudes (converted model, straight leg, foot body origin instead of site)

```
M_leg (5×5, left, q=0)  diag ≈ [0.0302, 0.1106, 0.1803, 0.0335, 0.00099]   (armature 0 in converted XML)
Λ (foot origin)          ≈ [[0.440, 0.024, 0.274],[0.024, 0.525, 0.154],[0.274, 0.154, 1.968]]
Kp = ω_n² diag(Λ)        ≈ [10025, 11975, 23810] N/m     (Kd = 25 N·s/m)
```
This is the order of magnitude the reference gains were tuned for; joint damping/armature/frictionloss
are unknown in the converted model (README: "NOT from source") and change `M`, `bias` and hence `Kp`.

### 14.7 Python (mujoco 3.12) API notes for an exact port

- Load: `mujoco.MjModel.from_xml_string(scene_xml_text, assets)` with `assets = {basename: bytes}`
  for every file in the folder (Korean path workaround; pattern in `MPC/src/affine/g1_model.py`).
- Override `m.opt.timestep = 0.002`, `m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST`.
- `mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, id, res6, 0)` → `res6 = [ω(3), v(3)]`, world.
- `mujoco.mj_contactForce(m, d, i, res6)`; `d.contact[i].frame` is a flat 9-vector, rows = axes.
- `mujoco.mj_jacSite(m, d, jacp(3×nv), jacr(3×nv), site)`; `mujoco.mj_jacDot(m, d, jacp_dot, None, point3, body)`.
- Mass matrix: `M = np.zeros((nv,nv)); mujoco.mj_fullM(m, d, M)` (3.12 signature takes `d`, `d.qM` no longer exists).
- Aux models: `spec = mujoco.MjSpec.from_string(xml, assets)` of `mit_humanoid.xml` (robot only); delete
  actuators/sensors/keys, the free joint, the other leg root body and the arm root bodies
  (`spec.delete(elem)`), `aux = spec.compile()`; do NOT apply the 0.002/implicitfast override to the aux
  models (XML `<option>` kept, 7.1 step 5);
  then per tick set `aux.body_pos[torso] / aux.body_quat[torso]`, `aux_data.qpos/qvel` for the leg,
  `mujoco.mj_forward(aux, aux_data)`, read `qfrc_bias`, `mj_fullM`, `mj_jacSite`, `mj_jacDot`.
- `data.actuator_force` is the previous step's applied torque (only for logging).

---------------------------------------------------------------------------------------------------

## 15. Open questions / decisions needed

1. **Foot contact site placement** in the converted MJCF (14.4): centre vs bottom of the foot cylinder;
   orientation assumed identity. Affects foot z used by the swing planner/contact manager and the MPC
   moment arm.
2. **Joint-zero / standing-pose mismatch** (14.5): yaml offsets give a 13.5° toe-up foot and base
   z ≈ 0.657 in the converted model vs `0.679472` expected. Decide whether to (a) keep the yaml offsets and
   change `base_position_W` to what the converted model gives, (b) change the ankle offset to `-0.465`
   (flat foot) and re-tune height, or (c) re-derive the converted model's joint origins from the URDF
   so that the yaml values become consistent. The MPC height target = `base_z + bodyComLocation.z`.
3. **Joint damping / armature / frictionloss** and contact `solref/solimp` are absent from the
   converted model (README). They enter `M`, `bias` (swing OSC), contact force thresholds
   (`contact_force_on_threshold 36 N`) and the passive behaviour. Reference values unknown.
4. **Effort limits**: converted `ctrlrange` = URDF effort; the reference's MJCF limits are unknown.
   With knee 144 N·m and kp_z ≈ 24 kN/m the swing law can saturate; check `ctrl` clamping in logs.
5. **Staleness policy** (4.4): reproduce `mj_step`-then-read (1-tick-old kinematics, exact) or use
   `mj_step1/mj_step2` (consistent). Recommend exact first, then evaluate.
6. **Aux-model vs full-model** for `J`, `JvDot_W` and `bias` (7.2): exact requires the stale torso pose
   with fresh leg q/qd and zero base velocity. Recommend building aux models with `MjSpec` to remove
   the ambiguity.
7. **Headless command schedule**: the runner can only script `x_dot` headlessly
   (`SimulationRunner.cpp:454-537`, env `CONVEXMPC_HEADLESS_*`); the lateral 0.3 m/s and turn 1.3 rad/s
   demonstrations were presumably keyboard-driven in the GUI. The Python runner needs its own command
   schedule for y_dot and psi_dot, applied through the same low-pass/clamp filter
   (`updateFilteredUserCommand`, `My_Controller.cpp:610-654`, taus 0.92/0.8/0.70, maxima 0.7/0.5/2.0).
8. **Arm end body**: reference spec `left_forearm_link`; converted has `left_lower_arm` + a 0.01 kg
   `left_hand`. Hand position/velocity are read but unused, so either works; the hand mass is included
   in the reduced body either way (not in a leg subtree).
9. **`mj_objectAcceleration`** fields are read but unused; skip them in the port.
10. **Debug-marker bodies in the reduced body** (3.4): the reference G1/H1 scenes carry 4 massive mocap
    marker bodies (0.214 kg total) that pass the reduced-body filter; whether the undistributed MIT
    scene had them is unknown. Default for the port: none (converted scene); any visualisation markers
    added to the Python scene must be massless or explicitly excluded. Optional experiment: add the four
    marker masses and move them as the reference does, to bound the effect (≈ +0.08 kg·m² on Ixx/Iyy).

---------------------------------------------------------------------------------------------------

## Verification log (9/25)

1. Reduced-body filter includes any massive scene body (debug markers) — **applied** (§3.4 rewritten, §14.2, §15.10). Verified `setupRobotParams.cpp:414-431,439-473` (only `isInLegSubtree` and `mass<=0` excluded); compiled `g1/scene_23dof.xml` with mujoco 3.12: marker masses 0.0373/0.0373/0.0697/0.0697 = 0.2141 kg, parent = world, mocap; `SimulationRunner.cpp:748-801` moves them every tick. Converted MIT scene has 22 bodies, no markers.
2. Phase A is 999 ticks, controller starts at t = 1.998, walking at ≈2.998 — **applied** (§9.1 table + bullets, §9.3, §9.4, §11). Verified `LegPosInitializer.cpp:36-57` (`+= dt` before the `>=` test), `RobotRunner.cpp:119-142`, control-before-`mj_step` loop `SimulationRunner.cpp:337-343`.
3. FSM synced twice per tick after init — **applied** (§8 steps 6-8 + "Double FSM sync" note). Verified `My_Controller.cpp:476-486` and `:1386` (`syncLocomotionFSM` in `runController`), `RobotRunner.cpp:73-82`. Corrected one detail: braking settle means use a time window so duplicated identical samples leave the means unchanged; only the hold-tick counter doubles.
4. touchdownYaw_W seeded at init (plain sum), not reset by resetSwingState — **applied** (§10.1 state description, §10.3). Verified `My_Controller.cpp:450-457` (`swingFootYawTargetWorld(side)` = base + psiOffset), `:534-553` (no touchdownYaw reset), `:826-858` (search-mode latch `:831-832`, swing latch `:853-854`).
5. Jw_W @ qd is the torso-relative foot ω — **applied** (§7.2 note, §10.1 yawRate comment, §10.2 ω_foot comment). Verified aux model has no base dofs (`LegSwingDynamicsProvider.cpp:346-349, 603-651`), `My_Controller.cpp:114-115`, `SwingAttitudeControl.h:49`.
6. Aux deletion is by name, aux keeps XML `<option>`, M has armature not damping — **applied** (§7.1 steps 1-5, §7.2 comments, §14.7). Verified `LegSwingDynamicsProvider.cpp:29-47` (`collectElementNames` skips unnamed) and that only the main model is configured (`SimulationRunner.cpp:268`). Converted MJCF: all 18 motors named, no sensors/keys, XML timestep 0.001, gravity −9.81.
7. Full-model equivalence needs stale torso pose + fresh leg q — **applied** (§7.2 equivalence block rewritten, §4.4, §15.6). Verified `LegSwingDynamicsProvider.cpp:613-630`. Wording adapted to mujoco 3.12 (no `d.qM`; "mass matrix via `mj_fullM`").
8. actuatorfrcrange clamps qfrc_actuator, not actuator_force — **applied** (§12). Verified with mujoco 3.12: ctrl 5, actuatorfrcrange ±1 → actuator_force 5, qfrc_actuator 1.
9. OPEN Q1 (swing bias, Kp diag) — **applied** (already stated in §10.2; confirmed `OperationalSpaceDynamics.cpp:49-58,76-83`, `LegController.cpp:302-315`; §7.2 bias comment now says "stationary torso").
10. OPEN Q2 (stance plain Jᵀ, yaw hold z only) — **applied** (§10.1: explicit torque contribution α·m_z·Jw_W[2,:]ᵀ added; `OperationalSpaceDynamics.cpp:87-94`).
11. OPEN Q3 (staleness list) — **applied** (§4.4 list extended: xipos/ximat, subtree_com, efc_force, main-data mass matrix/bias/Jacobians; fresh time vs stale quaternion; first tick consistent).
12. OPEN Q4 (aux model contents, torso pose, armature) — **applied** (§7.1 steps 3-5, per-tick stale/fresh comments, §7.2 M comment).
13. OPEN Q5 (init phase inactive, arms) — **applied** (§9.4 expanded, §11 spline wording incl. end at call 500 / t = 0.998).
14. OPEN Q6 (torque clamping) — **applied** (§12 already stated ctrlrange-only; MuJoCo-side limits corrected per finding 8).

## Confirmed by verifier

- yaml values and line numbers in §1 table (xml paths 11-12, site 13, gravity 14, iterations 7 @24, ω_n [151,151,110] @41, kd 25 @42, yaw hold true @51, roll 0/0, pitch 300/18, yaw 305/18, stance yaw 20/4 @64-71, settle 1.0 @107, joint gains @124-129, initial_pose @133-138) and header defaults (ControllerConfig.h:43,58,64-65,76,130,143-144; InitialPoseConfig.h)
- simulation.yaml timestep 0.002 / implicitfast overwrite model.opt via configureSimulationModel (SimulationConfig.cpp:96-104, SimulationRunner.cpp:268)
- No keyframe loaded; RobotParams built on first control tick; start-up mass properties overwritten before any read
- fillJointGroup q_idx/qd_idx/actuator_idx, motorTauMax = max|ctrlrange| never applied
- collision geom filter (contype/conaffinity !=0 or group 3), hipLocationFromBody = body_pos of hip_yaw body
- fillReducedBodyMassProperties algorithm (§3.5) and updateReducedBodyMassPropertiesFromData (§3.6): xipos/ximat sums, parallel axis about reduced COM, bodyInertia = R_BWᵀ… = Rz(ψ)ᵀ I_W Rz(ψ), bodyComLocation = Rz(ψ)ᵀ(com_W − xpos_torso), ψ = atan2(R10,R00)
- Converted MJCF reduced-body numbers 14.2275 kg, com [0.01215,0.00286,0.11379], inertia matrix — reproduced with mujoco 3.12
- fillCheaterState: torso xpos/xquat, mj_objectVelocity flg_local=0 split [ω(3),v(3)], site pos/vel/xmat for feet, actuator_force as tauEstimate, flags hasFootJacobians/hasLegDynamics=false
- readFootContactInfo: frameᵀ f[0:3] = force on geom2, sign +1 if foot is geom2 else −1, normalForce += |f0|
- StateEstimator yaw unwrap/reset conditions and psi = yaw_W_unwrapped; standingFeet zeroed in copyFrom
- x0 = [roll, pitch (ZYX from normalized quat), yaw_unwrapped, com_W, torso ω_W, v_torso + ω×offset, −9.81]; offset = Rz(yaw_unwrapped)·bodyComLocation
- Body-target seed: base_position_W + Rz(base_rpy[2])·bodyComLocation when hasBasePose
- Dynamics request: pre-init → standingFootJacobians (walking + settle>0); after init swingLegDynamics = mode==Walking, standing = mode==Standing (LocomotionFSM.cpp:285-286)
- Aux swing outputs: mj_jacSite columns, mj_jacDot at site_xpos on site_bodyid, mj_fullM block, qfrc_bias block; standing 6×10 Jacobians rows [L;R], columns [L5,R5]
- RobotRunner::run order: setupStep zeroes commands/data, arm initializer+gains every tick, leg initializer every tick, PD branch only before legInit, runController after; composeCommand scatter by actuator_idx
- Spline: 11 knots [0,0,0,0,T/4,T/2,3T/4,T,T,T,T], CPs [q0,q0,q0,qf,qf,qf,qf], clamped evaluation, curr_time += dt before evaluation, float dt
- Joint PD law τ = ff + Kp(qDes−q) + Kd(0−qd), no gravity comp; arms identical, hasCartesianData never set
- writeStandingLegCommands: τ(10) = Jvᵀ(−[F_L;F_R]) + Jwᵀ(−[M_L;M_R]), JointTorque passthrough, no ramp/yaw hold/bias
- Walking stance: StanceWrench, F = −α u_F, M = −α u_M, yaw hold z only when enabled and α>0; J = aux site Jacobians
- Swing: Λ = (Jv M⁻¹ Jvᵀ + 1e-9 I)⁻¹ via LDLT; Kp = ω²·diag(Λ); Kd const; τ = Jvᵀ F_fb + Jvᵀ Λ(a − JvDot qd) + bias + attitude
- SwingAttitude pitch/yaw error and rate formulas, roll skipped when gains 0, throw on degenerate axes
- Swing yaw target: base = yaw_unwrapped + 1.0·psi_dot·max(0,(0.5+offset)·0.33), offsets 0.26/0.28/0.37 by speed thresholds 0.65/0.60, psiOffset clamp(100|ψ̇|,0,20)° sign rule, liftAngleNear
- MPC cadence: iteration==0 or diff ≥ 7; fallback u=0 plus Fz = bodyMass·|g|/nStance for stance legs
- applyRobotCommand clamp and size check; motorTauMax unused
- MuJoCo 3.12 python signatures: mj_fullM(m,d,M) and no d.qM; mj_objectVelocity res=[rot,lin]; mj_contactForce in contact frame
- Converted MJCF: nq 25, nv 24, nu 18, nbody 22, body ids (right_hip_yaw 2, left_hip_yaw 7, shoulders 12/17), ctrlrange 34/34/72/144/68/34/55, armature and damping 0, foot cylinder r 0.01 half-length 0.075 at (0.03,0,−0.03)
- Kp sample numbers in 14.6 are consistent with ω²·diag(Λ) (151²·0.440 ≈ 10.0k, 151²·0.525 ≈ 12.0k, 110²·1.968 ≈ 23.8k)
