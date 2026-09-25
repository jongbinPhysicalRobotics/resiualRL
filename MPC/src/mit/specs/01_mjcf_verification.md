# 01 — MIT Humanoid MJCF verification and required edits for the reference-controller port

Scope: numerical verification of `mit_humanoid_mjcf/mit_humanoid.xml` (+ `scene.xml`) against its
source URDF `mit_humanoid_mjcf/urdf_source/humanoid_full_sf.urdf`, plus every model-side change the
Python port of `reference/` (ispaik06/convex-mpc-biped) needs. All numbers below were produced by
`01_compare_mjcf_urdf.py` and `01b_extra_numbers.py` (same folder; outputs in `01_compare_output.txt`,
`01b_extra_output.txt`). MuJoCo 3.12.0, numpy 2.5.3.

Paths (all under `C:\Users\백종빈\Desktop\4-2\residual RL\`):
- MJCF: `mit_humanoid_mjcf\mit_humanoid.xml`, `mit_humanoid_mjcf\scene.xml`, meshes `mit_humanoid_mjcf\meshes_v3\*.stl`
- URDF: `mit_humanoid_mjcf\urdf_source\humanoid_full_sf.urdf`
- Reference: `reference\sim\src\models\MitHumanoidSpec.cpp`, `reference\sim\src\setupRobotParams.cpp`,
  `reference\sim\src\MujocoCheaterStateReader.cpp`, `reference\config\mit_humanoid\my_controller.yaml`,
  `reference\config\simulation.yaml`, `reference\My_Controller\src\GaitScheduler.cpp`
- Loading rule (Korean path): `mujoco.MjModel.from_xml_string(scene_xml_text, assets)` with
  `assets = {basename: bytes}` for every file in the model folder (pattern: `MPC\src\affine\g1_model.py`).
  The `<include file="mit_humanoid.xml"/>` in scene.xml and the mesh files are all resolved from that dict.

---

## 0. Verdict

**The conversion is numerically correct.** Three-way comparison (MJCF model ↔ MuJoCo's own URDF
importer ↔ raw numbers parsed from the URDF text) over all 21 bodies, 18 joints, 32 geoms:

| quantity (max abs diff over all bodies/joints/geoms) | value | tolerance | worst item |
|---|---|---|---|
| body mass | 0 | 1e-6 | – |
| body CoM (`ipos`, link frame) | 0 | 1e-6 | – |
| principal inertia (`body_inertia`, sorted) | 4.2e-7 | 1e-4 | base |
| full inertia tensor about CoM in link frame (`R_iquat·diag·R_iquatᵀ`) vs raw URDF `ixx…izz` | 4.2e-7 | 1e-4 | base |
| body pos relative to parent | 0 | 1e-6 | – |
| body quat relative to parent (rotation angle) | 8.8e-7 rad | 1e-6 | left/right_hip_abad |
| joint type / axis / range / pos / URDF effort vs ctrlrange | 0 | – | – |
| world pose of every body at qpos0 (relative to base) | 8.7e-7 m / rad | 1e-6 | feet |
| every geom (type, size, world pose, mesh vertices, mesh centring) | 8.6e-7 | 1e-6 | foot cylinder |
| bodies/joints/geoms missing or extra | none (MJCF adds only `floating_base` freejoint) | | |

The residual 4e-7 / 9e-7 comes from the MJCF quaternions being written with 6 decimals
(e.g. `quat="0.976296 0 0.216439 0"` for rpy pitch 0.436332). Purely cosmetic; regenerate with more
digits if you want exact zeros. `balanceinertia="true"` had **no effect**: the URDF also imports with
`balanceinertia="false"` (no link violates the triangle inequality). No body/joint exceeds tolerance.

Facts about the hand-added parts (Section 2): freejoint OK; 18 `<motor>` with `ctrlrange = ±URDF effort` OK;
timestep 0.001 in XML must become **0.002** for the port; joint damping/armature/frictionloss are all 0
(URDF says 0, but gives rotor inertias — see recommendations); floor μ 0.8 vs geom μ 1.0 → effective **1.0**
(MuJoCo takes the max) = yaml `friction_coefficient 1.0`; the two 0.01 kg zero-inertia hand bodies load fine.

What must be added for the port (Section 8): two foot contact **sites** at `(0.03, 0, -0.04)` in the foot
frames, a keyframe, timestep 0.002, and a name map (we keep the MJCF names).

---

## 1. Model-vs-URDF comparison (how it was done)

Script `01_compare_mjcf_urdf.py`:

1. `m_mj = from_xml_string(scene.xml, assets)` (nbody 22, njnt 19, nq 25, nv 24, nu 18, ngeom 33, nsite 0, nkey 0).
2. `m_ur = from_xml_string(urdf_text_with_<mujoco><compiler balanceinertia="true" fusestatic="false" discardvisual="false" strippath="true"/></mujoco>, assets)`
   (nbody 22, njnt 18, nq 18, ngeom 32 — no floor, no freejoint). STL bytes are supplied under their basenames,
   so the URDF's `meshes_v3/xxx.stl` references resolve.
3. Raw parse of `<inertial>` with `xml.etree`: `I_link = R(rpy)·[[ixx,ixy,ixz],[ixy,iyy,iyz],[ixz,iyz,izz]]·R(rpy)ᵀ` (all rpy are 0).
4. Per body: mass, `ipos`, sorted `body_inertia`, full tensor, `body_pos`, `body_quat` (as rotation angle).
   Per joint: type, axis, range, `jnt_pos`, URDF `effort`. `mj_forward` at qpos0 on both → `xpos/xmat` of
   every body and `geom_xpos/xmat/size/type` of every geom relative to the base; mesh vertex arrays and
   MuJoCo's mesh centring (`mesh_pos/mesh_quat`) compared for every visual mesh.

Per-body table (all diffs): see `01_compare_output.txt` section 1. Highlights:

| body | mass diff | ipos diff | I_full diff | pos diff | quat diff (rad) |
|---|---|---|---|---|---|
| base | 0 | 0 | 4.2e-7 | 0 (MJCF base sits at z 0.7483, expected) | 0 |
| left/right_hip_abad | 0 | 0 | 4.9e-9 | 0 | 8.8e-7 |
| left/right_upper_leg | 0 | 0 | 4.1e-8 | 0 | 3.7e-8 |
| all others | 0 | 0 | ≤ 4.9e-9 | 0 | 0 |

Joint table: every joint identical (axis, range, pos), effort = ctrlrange:

| joint (MJCF name) | child body | axis | range [rad] | effort [N·m] | URDF velocity [rad/s] | URDF rotor_inertia [kg·m²] |
|---|---|---|---|---|---|---|
| a01_right_hip_yaw / a06_left_hip_yaw | *_hip_yaw | 0 0 1 | ±1 | 34 | 48 | 0.01188 |
| a02_right_hip_abad / a07_left_hip_abad | *_hip_abad | 1 0 0 | ±1.2 | 34 | 48 | 0.01188 |
| a03_right_hip_pitch / a08_left_hip_pitch | *_upper_leg | 0 1 0 | ±1.5 | 72 | 40 | 0.0198 |
| a04_right_knee / a09_left_knee | *_lower_leg | 0 1 0 | 0 … 2.2 | 144 | 20 | 0.0792 |
| a05_right_ankle / a10_left_ankle | *_foot | 0 1 0 | ±0.8 | 68 | 24 | 0.04752 |
| a11_right_shoulder_pitch / a15_left_shoulder_pitch | *_shoulder | 0 1 0 | ±3.14 | 34 | 50 | 0.01188 |
| a12_right_shoulder_abad | right_shoulder_2 | 1 0 0 | −3.14 … 0.1 | 34 | 50 | 0.01188 |
| a16_left_shoulder_abad | left_shoulder_2 | 1 0 0 | −0.1 … 3.14 | 34 | 50 | 0.01188 |
| a13_right_shoulder_yaw / a17_left_shoulder_yaw | *_upper_arm | 0 0 1 | ±3.14 | 34 | 50 | 0.01188 |
| a14_right_elbow / a18_left_elbow | *_lower_arm | 0 1 0 | −2.3 … 1.7 | 55 | 50 | 0.0304 |

Fixed frame offsets in the leg chain (URDF joint origins, reproduced exactly in MJCF `body pos/quat`):
`hip_yaw` body at `(-0.00565, ±0.082, -0.05735)`, pitch −10° (rpy 0 −0.174533 0);
`hip_abad` body at `(-0.06435, 0, -0.07499)`, pitch +25° (0.436332);
`upper_leg` body at `(0.071, ∓0.0018375, 0)`, pitch −15° (−0.261799);
`lower_leg` at `(0, 0, -0.267)`; `foot` at `(0, 0, -0.2785)`. Net fixed pitch = 0, so at q = 0 the
leg is straight down: foot origin at `(0.012581, ±0.080162, -0.706251)` relative to base.
Arms: `shoulder` at `(0.01346, ±0.17608, 0.24657)`, `shoulder_2` at `(0, ±0.0575, 0)`,
`upper_arm` at `(0, 0, -0.1025)`, `lower_arm` at `(0, 0, -0.1455)`, `hand` (fixed) at `(0, 0, -0.27)`.

qpos layout of the MJCF (tree order; **right leg first**, unlike the reference spec which lists Left first):

```
qpos[0:3]  base xyz      qpos[3:7] base quat (w x y z)
qpos[7..11]  a01..a05 right leg  [hip_yaw, hip_abad, hip_pitch, knee, ankle]      dof 6..10
qpos[12..16] a06..a10 left leg   [hip_yaw, hip_abad, hip_pitch, knee, ankle]      dof 11..15
qpos[17..20] a11..a14 right arm  [shoulder_pitch, shoulder_abad, shoulder_yaw, elbow]  dof 16..19
qpos[21..24] a15..a18 left arm   [shoulder_pitch, shoulder_abad, shoulder_yaw, elbow]  dof 20..23
actuator i drives joint i+1 (same names), gear 1, gaintype FIXED(1), biastype NONE  → ctrl = joint torque [N·m]
```
The reference never assumes an order: it looks up `jnt_qposadr/jnt_dofadr` by name
(`setupRobotParams.cpp:83-105 fillJointGroup`). The port must do the same.

---

## 2. Hand-added parts audit

| item | MJCF as delivered | reference expectation | status |
|---|---|---|---|
| free joint | `<freejoint name="floating_base"/>` on body `base`, base pos `0 0 0.7483` (straight legs, feet 2.0 mm above floor) | floating base with `nq = 7 + 18` | OK |
| actuators | 18 `<motor>` named like the joints, `ctrlrange = ±effort` (34/34/72/144/68 legs, 34/34/34/55 arms), `actuatorfrcrange` identical | `motorTauMax = max(|lo|,|hi|)` of `actuator_ctrlrange` (`setupRobotParams.cpp:99-104`) | OK |
| timestep / integrator | `0.001`, `implicitfast` | `simulation.yaml`: `physics_timestep_sec 0.002`, `implicitfast`; the reference **overwrites** `model->opt.timestep/integrator` at load (`SimulationConfig.cpp:102-103`), so its XML value is irrelevant; the controller gets `model->opt.timestep` as its tick (`SimulationRunner.cpp:399`) | **change to 0.002** |
| joint damping / armature / frictionloss | all 0 | URDF: `damping 0 friction 0`, but `rotor_inertia` given per joint (table above; header says they are added in the IsaacGym config, "not in this URDF") | see recommendation R1 |
| contact solref / solimp | MuJoCo defaults `solref 0.02 1`, `solimp 0.9 0.95 0.001 0.5 2`, `margin 0` | reference G1/H1 XMLs also use defaults | OK (fact); tune only if needed |
| friction | floor `0.8 0.005 0.0001` (`condim 3`, priority 0); every robot geom default `1 0.005 0.0001` (priority 0) | MuJoCo pair rule: equal priority → element-wise **max** → μ = 1.0, torsional 0.005, rolling 0.0001 (verified: `contact.friction = [1.0, 1.0, 0.005, 0.0001, 0.0001]`); yaml `mpc.friction_coefficient: 1.0` (line 17) | consistent, but the MPC uses the *full* physical limit with zero margin. R2 |
| hand bodies | `mass 0.01`, `diaginertia 0 0 0`, `ipos 0 0 -0.27`, no joint | MuJoCo only requires positive inertia for bodies **with dofs**; loads without warning | OK. Note: the point mass sits 0.27 m below the hand origin = 0.54 m below the elbow (URDF quirk, `<inertial origin xyz="0 0 -0.27">` inside a link already at −0.27). Effect on upper-body inertia ≈ 2·0.01·(0.54²−0.27²) ≈ 0.004 kg·m² (< 1 % of I_xx). Keep (URDF fidelity) or set `pos="0 0 0"` — either is fine. |
| visual meshes | `contype=conaffinity=0`, `group 1`, `density 0` → no physics | – | cosmetic only |
| mesh mirroring | right-side meshes use `scale="0.001 -0.001 0.001"` (as in URDF); verified bbox y-range flips sign (e.g. `left_hip_yaw1` y ∈ [−0.0506, 0.0504] vs `left_hip_yaw` y ∈ [−0.0504, 0.0506]); vertex arrays identical to the URDF import (d_mesh = 0) | – | correct. `left_forearm.stl` is *not* mirrored for the right arm in the URDF either (both `+0.001`), so the right forearm is visually a left forearm — cosmetic, inherited from the URDF. |
| foot cylinder quat | `0.707109 0 0.707105 0` (from URDF rpy `0 1.57079 0`, not π/2) → axis tilt 6.3e-6 rad from +x | – | negligible (0.5 µm over the half-length). May be written as `0.7071068 0 0.7071068 0` in the copy (R5). |

Collision geoms (contype 1): base box `size 0.075 0.15 0.175` at `(0.023, 0, 0.08)`; thigh cylinder
r 0.035 h 0.09 at `(0,0,-0.16)`; shank cylinder r 0.035 h 0.08 at `(0,0,-0.15)`; foot cylinder r 0.01
h 0.075 at `(0.03, 0, -0.03)` axis along foot x; upper-arm cylinder r 0.05 h 0.075 at `(0,-0.01,-0.035)`;
forearm cylinder r 0.025 h 0.075 at `(-0.01,0,-0.15)`; hand disc r 0.025 h 0.0075 axis y. All match the URDF.
Note the reference's `collisionGeomIdsForBody` treats a geom as collision if `contype != 0 || conaffinity != 0 || group == 3`
(`setupRobotParams.cpp:37-49`) — with our model that picks exactly the one foot cylinder per foot, so
`foot_end_effector_source: collision_geom_center` would return the cylinder centre `(0.03,0,-0.03)` (1 cm
above the ground line). The yaml uses `site` (line 13), see Section 3.

---

## 3. Foot geometry, initial pose, contact site

Foot collision = one cylinder, radius r = 0.01, half-length h = 0.075, centre `c_F = (0.03, 0, -0.03)` in the
foot frame, axis = foot x. Lowest point of the cylinder in world:
```
z_min = c_z,W − h·|a_z| − r·sqrt(1 − a_z²)        a = world direction of the cylinder axis
```
(verified against MuJoCo's own contact detection: `dist = 0` exactly at the predicted base height).

### 3.1 Poses

Yaml initial pose (`my_controller.yaml:133-138`): `base_position_W [0,0,0.679472]` (comment: `0.625972 / 0.679472`),
`base_rpy_W 0`, `leg_joint_offsets [0, 0, -0.735, 1.2, -0.70]`, `arm_joint_offsets [0, 0, 0, -1.65]`.

Foot pitch relative to the base = hip_pitch + knee + ankle = −0.735 + 1.2 − 0.70 = **−0.235 rad = −13.47° (toe up)**
(all three joints are +y axes; the fixed −10°/+25°/−15° offsets sum to zero). So at the yaml joint offsets the
cylinder is *not* parallel to the ground: it touches at its rear (heel) end only → **1 contact point per foot**.
A flat foot at that hip/knee needs ankle = −(−0.735 + 1.2) = **−0.465 rad**.

| pose | base z | foot pitch | lowest cylinder point z | MuJoCo contacts (1 mm penetration) |
|---|---|---|---|---|
| qpos0 (all 0), XML base z 0.7483 | 0.748300 | 0° | +0.002048 (in the air by 2 mm) | – |
| yaml pose, yaml base z 0.679472 | 0.679472 | −13.47° | **+0.022344 (feet 22 mm in the air)** | 0 |
| yaml pose, alt comment 0.625972 | 0.625972 | −13.47° | −0.031156 (31 mm below the floor) | – |
| yaml pose, feet exactly touching | **0.657128** | −13.47° | 0 | 1 per foot, at x = 0.0323 (heel end) |
| flat variant (ankle −0.465), touching | **0.647750** | 0° | 0 | **2 per foot**, at x = 0.0217 and 0.1717 (cylinder ends, span 0.15) |

Neither yaml value (0.679472 / 0.625972) is reproducible from this model at these joint angles with any
plausible site location (for `base_z − site_z = 0.679472` the site would have to be at z ≈ −0.081 in the foot
frame, i.e. 4 cm below the sole). Those numbers were produced by the reference's `scene_posture_clearance`
tool on *their* `models/mit_humanoid/scene.xml` keyframe `nominal_stance` (`test/scene_posture/README.md:15-31`,
`posture_clearance.cpp:16-19,159-171`), which is not in the repo and evidently held a different leg pose.
Note that for G1 the yaml `base_position_W` (0.75) also differs from the G1 keyframe z (0.772571), so the yaml
value is a *reduced-body height target seed*, not the physical spawn height (used only in
`My_Controller.cpp:683-689` → `bodyTarget = basePosition_W + Rz(yaw)·bodyComLocation`).

### 3.2 Where the contact site must be

Reference usage of the site (`foot_end_effector_source: site`):
- foot position = `site_xpos` (`MujocoCheaterStateReader.cpp:226-234`), foot velocity = site linear velocity (`:243-251`),
  foot rotation = `site_xmat` (`:288-304`, used for foot yaw), swing Jacobians `mj_jacSite` (`LegSwingDynamicsProvider.cpp:172,273`).
- `scene_posture_clearance` defines the spawn height as "body z that places the sites on the ground plane z = 0"
  (`test/scene_posture/README.md:10`), i.e. by construction the reference's sites are **ground-level points of the sole**;
  G1/H1 use `pos="0.035 0 -0.03"` (`g1_23dof.xml:122,188`, `h1.xml:82,118`) = sole level for those feet.
- The MPC's CoP/moment limits are ±`foot_half_length` (0.065) and ±`foot_half_width` (0.01) about the site,
  and yaw-moment ≤ `torsional_friction_scale·μ·Fz` = 0.0657 Fz (`GaitScheduler.cpp:83-96`). With a *line* foot
  the natural moment origin is the centre of the contact line.

Therefore the site goes at the **centre of the cylinder axis projected onto the ground** = axis centre minus r in −z:
```
left_foot_contact_site / right_foot_contact_site :  pos = (0.03, 0, -0.04)   in the foot frame, no rotation
```
Justification by numbers: cylinder centre `(0.03,0,-0.03)`, r = 0.01 → ground line at z = −0.04 spanning
x ∈ [−0.045, +0.105] in the foot frame. With the foot flat this site is exactly at z = 0 (verified:
`site z = +0.000000` at base z 0.647750) and half-way between MuJoCo's two contact points (0.0217, 0.1717 world →
centre 0.0967 = site x). `foot_half_length 0.065` < physical 0.075 leaves a 1 cm CoP margin at each end;
`foot_half_width 0.01` = the cylinder radius (a line foot has no physical roll-moment capacity — the yaml keeps
`swing.roll_kp 0` (line 64) and uses `contact_wrench_model full_wrench` (line 26), so M_x is bounded to ±0.01 Fz);
yaw moment: two contacts at ±0.075 with μ = 1 can physically resist ≈ 0.075 Fz ≥ MPC bound 0.0657 Fz (12 % margin).
**Caveat (fact):** the floor and foot use `condim="3"`, so MuJoCo models **no torsional friction** (the 0.005
torsional coefficient is ignored). The yaw moment therefore exists physically only when the foot touches with
**two** points (flat foot, `init_flat`: contacts at x = 0.0217 and 0.1717). In the toe-up yaml pose (`init`) each foot
touches at a single point (x = 0.0323, heel end), so it can transmit **no yaw moment and no pitch moment** until the
foot rolls flat; the MPC's `full_wrench` limits (±0.065 Fz pitch, ±0.0657 Fz yaw) are not physically available there.
(`01b_extra_output.txt` labels `friction[3]` as "torsional"; that index is actually roll — the effective pair friction
is `[1.0, 1.0, 0.005, 0.0001, 0.0001]`, of which only index 0/1 act with condim 3.)
The site must have **identity orientation** in the foot frame because `site_xmat` is read as the foot rotation.
`site_xmat` pitch at the yaml pose = −13.465° (equals the foot body).

Alternative `(0.03,0,-0.03)` (= `collision_geom_center`) would put the reference point 1 cm above the sole:
swing apex (`swing.height 0.06`, yaml:39) and `stance_contact_loss_foot_height 0.060` (yaml:98) would be off by 1 cm.
Not recommended.

### 3.3 Foot/site positions at the initial pose (world, base at x = y = 0, yaw 0)

| | foot origin | site (0.03,0,−0.04) | site − base |
|---|---|---|---|
| yaml pose (ankle −0.70), base z 0.657128 | (0.066742, ±0.080162, 0.049379) | (0.105231, ±0.080162, 0.017464) | (0.105231, ±0.080162, −0.639664) |
| flat (ankle −0.465), base z 0.647750 | (0.066742, ±0.080162, 0.040000) | (0.096742, ±0.080162, 0.000000) | (0.096742, ±0.080162, −0.647750) |

The feet are ~0.10 m **in front of the base origin** at this pose, while the whole-body CoM is only 0.030 m in
front of it (Section 4). With the flat foot the support line spans x ∈ [0.022, 0.172] and the MPC CoP window
(site ± 0.065) is [0.032, 0.162]: the static whole-body CoM (x = 0.030) sits at the heel edge. The standing MPC
will have to shift the body forward a few cm (its target is `base_x + bodyComLocation_x = 0.0153` for the
14.2 kg reduced body). This is inherent to the yaml pose on this model — see open questions.

### 3.4 Passive settle test (joint PD only, no MPC) — `01b_extra_output.txt` lines 47-49

Patched model (sites + timestep 0.002), spawned at the keyframe, 1 s with only joint PD at the yaml offsets
(legs kp [70,50,100,100,200], kd [15,10,10,10,30]; arms kp 100, kd 5 — the yaml `joint_tracking` gains;
torques clipped to `ctrlrange`):

| start | after 1 s: base pos | base pitch | contacts | ankle q | knee q |
|---|---|---|---|---|---|
| `init` (ankle −0.70, z 0.657128) | (0.080, 0.020, 0.053) | −89.9° | 0 | +0.423 | 1.243 |
| `init_flat` (ankle −0.465, z 0.647750) | (−0.042, −0.073, 0.053) | −89.7° | 0 | −0.638 | 1.269 |

Both keyframes **fall over within 1 s** under joint PD alone (base z 0.053 = lying on the floor). This is expected and
is *not* a model defect: the static CoM (x = 0.030) is at/behind the heel edge of the support (Section 3.3), a line
foot has no roll capacity, and nothing balances. It means the port **cannot** use a "PD-only settle" phase before the
MPC starts; balance (standing MPC / reference `leg_initialization_time` logic) must be active from t = 0, or the
robot must be held (e.g. a weld/fixed base) during initialization. See open question 7.

---

## 4. SRB / reduced-body numbers

**Important fact:** the reference's SRB uses the **torso + arms only** ("reduced body"), not the whole robot.
`buildRobotParamsFromSpec` calls `fillReducedBodyMassProperties` (`setupRobotParams.cpp:370`, function at
`:213-326`): it sums every body that is *not* in a leg subtree (leg root = body of the first leg joint, i.e.
`left_hip_yaw`/`right_hip_yaw`) and sets `bodyMass`, `bodyComLocation` (torso-root frame, q = 0) and `bodyInertia`
(about the reduced CoM, parallel-axis). Every control tick the sim then recomputes the same quantities from the
*current* `xipos/ximat` and rotates them into the yaw-aligned frame: `updateReducedBodyMassPropertiesFromData`
(`:398-490`, called at `SimulationRunner.cpp:409`):
```
M_up      = Σ_{b ∉ legs, m_b>0} m_b
c_up,W    = Σ m_b·xipos_b / M_up
I_up,W    = Σ [ ximat_b·diag(body_inertia_b)·ximat_bᵀ + m_b·(|d_b|²·1 − d_b d_bᵀ) ],   d_b = xipos_b − c_up,W
psi       = atan2(R_WT[1,0], R_WT[0,0])     (torso yaw),  R_WB = Rz(psi)
bodyMass = M_up ;  bodyInertia = R_WBᵀ·I_up,W·R_WB ;  bodyComLocation = R_WBᵀ·(c_up,W − torsoPos_W)
```
`fillBodyMassProperties` (total mass, torso-only inertia, `:62-81`) exists but is **never called**. The MPC uses
`bodyMass` and `bodyInertia` directly (`MPCFormulation.cpp:45-56`) and the standing feed-forward force is
`bodyMass·|g| / nStance` (`My_Controller.cpp:1037-1040`); standing joint torque is `Jᵀ(−F)` with no leg-gravity
term (`My_Controller.cpp:1281-1283`). The port must reproduce this (14.23 kg), even though the legs weigh 43 %.

Numbers (MJCF, yaw = 0 so yaw-aligned frame = world axes; base at x = y = 0):

| quantity | value |
|---|---|
| total mass | **24.888550 kg** (README says 24.89) |
| legs (2 × hip_yaw+hip_abad+upper+lower+foot) | 10.661082 kg = **42.84 %** (5.330541 kg per leg) |
| torso + arms = reference `bodyMass` | **14.227468 kg** (base 8.52 + 2 × 2.853734) |
| per body | base 8.52; hip_yaw 0.84563; hip_abad 1.20868; upper_leg 2.64093; lower_leg 0.35435; foot 0.280951; shoulder 0.788506; shoulder_2 0.80125; upper_arm 0.905588; lower_arm 0.34839; hand 0.01 |
| `hipLocationFromBody` (= `body_pos` of hip_yaw, `setupRobotParams.cpp:143-153,197`) | left (−0.00565, 0.082, −0.05735), right (−0.00565, −0.082, −0.05735) |
| torso-only (`fillBodyMassProperties`, unused) | `body_inertia` diag (0.172925, 0.106132, 0.091496), `ipos` (0.009896, 0.004771, 0.100522) |

Reduced body at **q = 0** (`fillReducedBodyMassProperties`, torso-root frame):
```
bodyMass        = 14.227468
bodyComLocation = [0.012151, 0.002857, 0.113790]
bodyInertia     = [[0.538649, 0.001511, 0.001121],
                   [0.001511, 0.177667,-0.001133],
                   [0.001121,-0.001133, 0.392504]]
```
Reduced body at the **yaml initial pose** (arms 0,0,0,−1.65; legs do not enter; identical for both keyframes):
```
bodyMass        = 14.227468
bodyComLocation = [0.015306, 0.002857, 0.116574]          (yaw frame, from base origin)
bodyInertia     = [[0.519367, 0.001639, 0.005789],
                   [0.001639, 0.168406,-0.001019],
                   [0.005789,-0.001019, 0.402526]]         principal (0.168394, 0.402245, 0.519660)
upper CoM_W     = base + bodyComLocation  → z = 0.773702 (base 0.657128) / 0.764323 (base 0.647750)
r_site − upperCoM (yaml pose, tilted foot): left (0.089925, 0.077305, −0.756238), right (0.089925, −0.083020, −0.756238)
r_site − upperCoM (flat variant):            left (0.081436, 0.077305, −0.764323), right (0.081436, −0.083020, −0.764323)
```
(the y-asymmetry is the base CoM offset y = +0.004771 in the URDF.)

Whole body at the yaml pose (for our own SRB variants / sanity checks, not used by the reference):
```
CoM − base      = [0.030034, 0.001633, −0.032627]   (yaml ankle −0.70)
CoM height      = 0.624501 (base 0.657128, feet touching) ; 0.646845 if base 0.679472 ; flat variant 0.614996 (base 0.647750)
I_whole about whole CoM, world axes (yaml pose):
                  [[1.547715, 0.002238, 0.145137],
                   [0.002238, 1.177608,−0.007084],
                   [0.145137,−0.007084, 0.575994]]         principal (0.554694, 1.177691, 1.568933)
I_whole flat variant: [[1.551292,0.002235,0.144850],[0.002235,1.181142,−0.007089],[0.144850,−0.007089,0.575952]]
legs-only CoM_W = (0.049689, 0, 0.425389)  (yaml pose, base 0.657128)
```
Cross-check: `d.subtree_com[base]` equals the summed CoM to 1e-12.

---

## 5. Cross-check against the pbrs-humanoid URDF (`scratchpad\mit_humanoid_fixed_arms.urdf`, arms locked)

Total 24.267612 kg vs ours 24.888550 (+0.620938). Per link:

| link | ours | pbrs | diff |
|---|---|---|---|
| base | 8.520000 | 7.954054 | **+0.565946** |
| hip_yaw | 0.845630 | 0.842752 | +0.002878 |
| hip_abad | 1.208680 | 1.199631 | +0.009049 |
| upper_leg | 2.640930 | 2.634789 | +0.006141 |
| lower_leg | 0.354350 | 0.346291 | +0.008059 |
| foot | 0.280951 | 0.279583 | +0.001368 |
| shoulder / shoulder_2 / upper_arm / lower_arm / hand | 0.788506 / 0.801250 / 0.905588 / 0.348390 / 0.01 | identical (0.801249) | 0 |

Leg links match to within 1 % (0.0275 kg per leg, sum of the five small updates); the arm links are byte-identical.
The whole difference is the torso (+0.566 kg) plus 2 × 0.0275 kg of leg updates = 0.621 kg. Explanation: our URDF
header says "Updated as of April 2023 with values from Robot-Software", the pbrs file is the older 2022 export;
the torso mass/CoM were re-measured (battery/electronics). The two files also use **different base frames**: pbrs
puts `hip_yaw` at `(0, ±0.082, 0)`, ours at `(-0.00565, ±0.082, -0.05735)`; pbrs base CoM z 0.151714 − 0.05735 =
0.0944 ≈ ours 0.1005, consistent with a 5.7 cm frame shift plus a re-measured CoM. The pbrs leg geometry is also
an older generation (hip_pitch origin `(0.08837, ∓0.00284, -0.01385)`, knee `(-0.01306, 0, -0.24916)` vs ours
`(0.071, ∓0.0018375, 0)` / `(0, 0, -0.267)`; hip_yaw axis `0 0 −1` on the left, hip_abad `−1 0 0` on the right;
foot Iyy artificially 0.08 "to account for reflected motor inertia"; box foot). So only the masses are comparable,
and they agree.

---

## 6. Name mapping (reference `MitHumanoidSpec.cpp` → our MJCF)

Decision: **keep the MJCF names**; the Python port carries this map instead of renaming the model. Sites are added
with the reference names (they do not exist yet, so no conflict).

| role | reference name (`MitHumanoidSpec.cpp` line) | our MJCF name |
|---|---|---|
| base body | `torso` (:6) | `base` |
| left / right foot body | `left_foot_link` / `right_foot_link` (:9, :19) | `left_foot` / `right_foot` |
| left / right foot site | `left_foot_contact_site` / `right_foot_contact_site` (:10, :20) | **same names, to be added** |
| left / right hand body (arm end) | `left_forearm_link` / `right_forearm_link` (:31, :40), hand site `""` | `left_lower_arm` / `right_lower_arm` |
| leg joints, order [hip_yaw, hip_abad, hip_pitch, knee, ankle] (:12-16, :22-26) | `left_hip_yaw_joint … left_ankle_joint`, actuators `left_hip_yaw … left_ankle` | `a06_left_hip_yaw, a07_left_hip_abad, a08_left_hip_pitch, a09_left_knee, a10_left_ankle`; right: `a01…a05` (actuator names = joint names) |
| arm joints, order [shoulder_pitch, shoulder_abad, shoulder_yaw, elbow] (:34-37, :43-46) | `left_shoulder_pitch_joint … left_elbow_joint` / actuators without `_joint` | `a15_left_shoulder_pitch, a16_left_shoulder_abad, a17_left_shoulder_yaw, a18_left_elbow`; right: `a11…a14` |
| fixed joints | none (:49 `{}`) | none |
| leg order in spec | Left, Right | (our tree order is right-first; irrelevant when indexing by name) |

Suggested Python constant:
```python
MIT_SPEC = dict(
    base="base",
    legs={"left":  dict(foot_body="left_foot",  foot_site="left_foot_contact_site",
                        joints=["a06_left_hip_yaw","a07_left_hip_abad","a08_left_hip_pitch","a09_left_knee","a10_left_ankle"]),
          "right": dict(foot_body="right_foot", foot_site="right_foot_contact_site",
                        joints=["a01_right_hip_yaw","a02_right_hip_abad","a03_right_hip_pitch","a04_right_knee","a05_right_ankle"])},
    arms={"left":  dict(hand_body="left_lower_arm",
                        joints=["a15_left_shoulder_pitch","a16_left_shoulder_abad","a17_left_shoulder_yaw","a18_left_elbow"]),
          "right": dict(hand_body="right_lower_arm",
                        joints=["a11_right_shoulder_pitch","a12_right_shoulder_abad","a13_right_shoulder_yaw","a14_right_elbow"])},
)   # actuator name == joint name for all 18
```

---

## 7. Keyframe(s)

qpos order as in Section 1. Two candidates (both verified by loading the patched model and `mj_forward`):

```xml
<keyframe>
  <!-- exactly the yaml initial_pose joints; base z chosen so the (toe-up) cylinder touches z=0 at its heel end -->
  <key name="init"
       qpos="0 0 0.657128  1 0 0 0
             0 0 -0.735 1.2 -0.70
             0 0 -0.735 1.2 -0.70
             0 0 0 -1.65
             0 0 0 -1.65"/>
  <!-- same hip/knee, ankle corrected so the sole line is flat on z=0 (2 contact points per foot) -->
  <key name="init_flat"
       qpos="0 0 0.647750  1 0 0 0
             0 0 -0.735 1.2 -0.465
             0 0 -0.735 1.2 -0.465
             0 0 0 -1.65
             0 0 0 -1.65"/>
</keyframe>
```
(The reference's key is called `nominal_stance`; name ours `init`/`init_flat` or add `nominal_stance` as an alias —
nothing in the walking controller reads the key name; only the test tools do.) If you prefer a spawn clearance,
add +0.002 m to z (the delivered XML uses 2 mm at q = 0). All joint values are inside the ranges
(left shoulder_abad range −0.1…3.14, right −3.14…0.1: 0 is legal; elbow −1.65 ∈ [−2.3, 1.7]).

---

## 8. Exact edit list for the copy at `MPC\models\mit_humanoid\`

Copy `mit_humanoid_mjcf\{mit_humanoid.xml, scene.xml, meshes_v3\*}` (the URDF folder and README may be copied
for provenance; they are not loaded). Then apply, in `mit_humanoid.xml`:

E1 (**fact / required**) — timestep 0.002 (reference `simulation.yaml`, forced at `SimulationConfig.cpp:102-103`):
```xml
<option timestep="0.002" integrator="implicitfast"/>
```
E2 (**required**) — foot contact sites (Section 3.2), inserted as the first child of each foot body:
```xml
<body name="right_foot" pos="0 0 -0.2785">
  <site name="right_foot_contact_site" pos="0.03 0 -0.04" size="0.005" group="1"/>
  ...
<body name="left_foot" pos="0 0 -0.2785">
  <site name="left_foot_contact_site" pos="0.03 0 -0.04" size="0.005" group="1"/>
  ...
```
E3 (**required**) — keyframe block from Section 7 (put it in `mit_humanoid.xml` after `</actuator>` or in `scene.xml`
after the floor; both load). The `base` body's `pos="0 0 0.7483"` may stay (only used when no key is applied).

E4 (**keep**) — the 18 `<motor>` actuators exactly as they are (ctrlrange = effort is what the reference reads).

E5 (**fact**) — leave `scene.xml` floor `friction="0.8 0.005 0.0001" condim="3"`: effective μ with the robot geoms is
1.0 by MuJoCo's max rule, equal to yaml `friction_coefficient 1.0`. If you want the sim's μ to be *visibly* what the
MPC assumes, set the floor to `friction="1 0.005 0.0001"` (no behavioural change) — or, to put a margin between MPC
and physics as the reference G1 scene does (`g1_23dof.xml:104-121`, foot geoms `priority="1" friction="0.8 …"` while its
yaml still says 1.0), give the foot cylinders `priority="1" friction="0.8 0.005 0.0001"` (**recommendation, changes physics**).

R1 (**recommendation**) — reflected rotor inertia as MuJoCo `armature` (values are facts from the URDF `<dynamics rotor_inertia>`;
adding them is our choice; the reference's own MIT XML is unknown; the reference G1 XML uses `damping 0.05 armature 0.01 frictionloss 0.2`):
```xml
<!-- legs -->  a0{1,6}_hip_yaw armature="0.01188"  a0{2,7}_hip_abad armature="0.01188"  a0{3,8}_hip_pitch armature="0.0198"
               a0{4,9}_knee armature="0.0792"     a{05,10}_ankle armature="0.04752"
<!-- arms -->  shoulder_pitch/abad/yaw armature="0.01188"   elbow armature="0.0304"
```
Keep `damping="0" frictionloss="0"` (URDF facts); treat a small damping (0.05–0.5 N·m·s/rad) as a tuning knob only if the
2 ms implicitfast sim shows joint chatter. Run the port once with and once without armature to see the sensitivity.

R2 (**recommendation**) — decide the initial ankle (Section 3.1). `init` reproduces the yaml joints literally (toe-up
13.5°, heel-point contact); `init_flat` is the physically sensible standing pose for a line foot. The leg PD target
during `leg_initialization_time 2.0` (yaml:135) is the yaml offsets, so if `init_flat` is used the PD will try to
dorsiflex the ankle by 0.235 rad (ankle kp 200 → up to 47 N·m) while standing MPC runs; check which one settles.

R3 (**recommendation**) — `initial_pose.base_position_W` for our model: the yaml 0.679472 puts the feet 22 mm in the
air and makes the standing height target `0.679472 + 0.116574 = 0.796046` (upper-body CoM); with our keyframes the
initial upper CoM is at 0.773702 (`init`) / 0.764323 (`init_flat`), i.e. the MPC will lift the body 2–3 cm after
start. Either keep 0.679472 (reference value, legs straighten a bit: knee ≈ 1.05 rad) or set it to the keyframe base z.

R4 (**optional, cosmetic**) — hand bodies: `<inertial pos="0 0 0" mass="0.01" diaginertia="0 0 0"/>` removes the
0.54 m point-mass quirk (changes `bodyInertia` by < 1 %). Default: keep as-is for URDF fidelity.

R5 (**optional, cosmetic**) — write the foot cylinder quat as `0.7071068 0 0.7071068 0` and regenerate body/inertial
quats with ≥ 8 digits to drive the 1e-7 residuals to zero.

Nothing else in the model needs to change: masses, inertias, joint limits, collision shapes and the arm chain are
exactly the URDF, and the reference reads everything else (`qpos0`, `body_pos`, `body_ipos/iquat/inertia`,
`actuator_ctrlrange`, `jnt_qposadr/dofadr`, `site_xpos/xmat`, `geom_xpos`) from the compiled model by name.

---

## 9. Open questions / decisions for the implementer

1. **Ankle at start** (`init` vs `init_flat`, R2) and the **base height target** (R3). The yaml pose is inconsistent
   with a flat line foot on this model; the reference's own MJCF (with keyframe `nominal_stance` giving 0.679472) is
   not available, so their actual standing geometry cannot be reproduced exactly.
2. **Reduced-body mass 14.23 kg vs total 24.89 kg.** This is what the reference does (`setupRobotParams.cpp:322,488`;
   MPC gravity force `14.23·9.81 = 139.6 N`, not 244 N), with standing torque `Jᵀ(−F)` and no explicit leg-gravity
   term. Whether the walking path compensates leg gravity (`LegSwingDynamicsProvider.cpp:247-254` copies `qfrc_bias`
   for the leg dofs) must be settled by the controller spec; the SRB numbers above are given for both definitions.
3. **CoM vs support line at the yaml pose**: whole-body CoM x = 0.030 m, MPC CoP window x ∈ [0.032, 0.162]
   (flat) — the initial pose is at the edge of static feasibility; expect the body to move forward ~5 cm at start.
   If that is undesirable, a different hip_pitch/knee pair (feet under the hips) is a tuning decision, not a model fix.
4. **Friction margin**: MPC μ = physical μ = 1.0 (E5). The reference G1 setup has physical 0.8 < MPC 1.0 and still
   walks; H1 uses 0.8/0.8. Decide whether to introduce a margin for MIT.
5. **Joint armature/damping** (R1): unknown in the reference's MIT model; affects the 2 ms implicitfast stability and
   the leg swing dynamics (`mj_fullM` mass matrix, `LegSwingDynamicsProvider.cpp:227`). Run both.
6. `nominal_foot_offsets_B = (0, ±0.0759996, 0)` (yaml:53-55) vs our feet at y = ±0.080162 (hip 0.082 minus the
   0.0018375 abad offset): the reference tuned 0.076 on its own model; verify what the offset is relative to in the
   swing planner spec before copying it.
7. **Start-up without balance falls** (Section 3.4): joint PD alone at either keyframe tips the robot over in < 1 s.
   How the reference sequences `leg_initialization_time 2.0` (PD ramp vs. MPC active) must be taken from the
   orchestration spec (05); if the reference relies on something its (missing) MIT XML provided (e.g. a different
   stance geometry with the CoM over the feet), a hold/weld during initialization or a stance with feet under the
   CoM is a decision for the implementer.
8. **Single-point contact in `init`** (Section 3.2 caveat): condim 3 + toe-up foot → no yaw/pitch moment per foot
   at start; prefer `init_flat` or accept that the first stance must roll the foot flat.
