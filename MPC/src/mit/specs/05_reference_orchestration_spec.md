# 05 — Reference controller orchestration spec (MIT Humanoid)

Scope: the per-tick flow of `MyController::runController()`, the `LocomotionFSM`, the
`ContactManager`, the user-command pipeline, the standing mode, and every yaml key of
`config/mit_humanoid/my_controller.yaml` with the place it is consumed.
The MPC QP itself (ConvexMPC/MPCFormulation), the swing-foot planner geometry and the
reference trajectory are covered only as far as this orchestration touches them.

All paths below are relative to `C:\Users\백종빈\Desktop\4-2\residual RL\reference\`.
Abbreviations: `MC` = `My_Controller/src/My_Controller.cpp`, `FSM` = `My_Controller/src/LocomotionFSM.cpp`,
`CM` = `My_Controller/src/ContactManager.cpp`, `CC` = `My_Controller/src/ControllerConfig.cpp`,
`CCH` = `My_Controller/include/MyController/ControllerConfig.h`, `GS` = `My_Controller/src/GaitScheduler.cpp`,
`HC` = `My_Controller/include/MyController/HorizonClock.h`, `SR` = `sim/src/SimulationRunner.cpp`,
`RR` = `robot/src/RobotRunner.cpp`, `KB` = `common/src/Utilities/KeyboardCommand.cpp`.

"Code wins" rule: where `docs/gait_scheduler_and_contact_management.md` differs from the code the
code is described and the discrepancy is listed in section 13.

---

## 0. Conventions used throughout

### 0.1 Frames
| symbol | meaning | source |
|---|---|---|
| `W` | MuJoCo world frame, z up, gravity `model.gravity = -9.81` along -z | `CCH:43`, yaml `model.gravity` |
| `T` | torso (base) body frame; `R_WT = torsoQuat_W` | `common/include/StateEstimator/StateEstimator.h:17` |
| yaw frame `Rz(psi)` | rotation about world z by the **unwrapped** torso yaw `psi = yaw_W_unwrapped` | `common/src/Estimator/StateEstimator.cpp:29-47` |
| `B` (body / "SRB" frame) | yaw-aligned frame: `Rz(psi)` applied to torso position; roll/pitch ignored | `MC:315-318` |
| `F` | foot end-effector frame, `R_WF = site_xmat` of the foot contact site | `sim/src/MujocoCheaterStateReader.cpp:288-305` |

`Rz(psi)` is the standard z-rotation (`common/include/Utilities/MatrixUtils.h:12`).
Yaw unwrap: `yaw_unwrapped(t) = yaw_unwrapped(t-1) + wrapToPi(yaw_wrapped(t) - yaw_wrapped(t-1))`,
`yaw_wrapped = atan2(R_WT(1,0), R_WT(0,0))` (`StateEstimator.cpp:29-47`, `common/include/Utilities/AngleUtils.h:6-18`).
Roll/pitch used by the controller come from the quaternion (`MC:63-75`):
```
roll  = atan2(2(wx+yz), 1-2(x²+y²))
pitch = asin(clamp(2(wy-zx), -1, 1))
```

### 0.2 Vector orderings
MPC state `x0 ∈ R^13` (`MC:656-675`):
```
[0] roll, [1] pitch, [2] yaw_W_unwrapped,
[3..5] reduced-body COM position in W,
[6..8] torsoAngVel_W   (WORLD-frame angular velocity of the torso, not body frame),
[9..11] reduced-body COM velocity in W,
[12] gravity (= -9.81)
```
MPC input / stance wrench `u ∈ R^12` = `_stanceWrenchWorld` (`MC:1276-1279, 1322-1331`, docs §4):
```
[0..2] F_left_W, [3..5] F_right_W, [6..8] M_left_W, [9..11] M_right_W
```
These are ground-reaction wrenches **on the body**. The leg command applies the negative
(`MC:1319-1331`). Leg index order is `robotParams.legs` order = MIT spec order: leg 0 = Left, leg 1 = Right
(`sim/src/models/MitHumanoidSpec.cpp:8-27`). Leg joint order (5-DOF):
`[hip_yaw, hip_abad, hip_pitch, knee, ankle]`; arm joint order (4-DOF)
`[shoulder_pitch, shoulder_abad, shoulder_yaw, elbow]` (same file; yaml `joint_tracking` comments).

`UserCommand` (`common/include/Utilities/UserCommand.h:4-13`):
```
x_dot [m/s, body/yaw frame +x forward], y_dot [m/s, +y left], psi_dot [rad/s, +z CCW],
body_height_offset_m, standing_roll_offset_rad, standing_pitch_offset_rad,
standing_mpc_debug_log_request (counter), locomotion_mode_toggle_request (counter)
```

### 0.3 Timing constants (MIT yaml + code)
| quantity | value | source |
|---|---|---|
| physics dt | 0.002 s (yaml `physics_timestep_sec`, overrides the MJCF `<option timestep>`) | `config/simulation.yaml:1`, `sim/src/SimulationConfig.cpp:96-104` |
| control tick | one `runRobotControl()` per `mj_step` → 500 Hz | `SR:337-343` |
| `cycleTime()` | 0.5 s | yaml `timing.cycle`, `CC:625` |
| `swingTime()` / `stanceTime()` | 0.17 / 0.33 s (must sum to cycle, `CC:494`) | `CC:629-635` |
| `horizonTime()` / `horizonSteps()` | 0.5 s / 25 | `CC:637-643` |
| `dtMpc()` | `horizon/horizonSteps` = 0.02 s | `CC:645-647` |
| MPC solve period | every `iterations_between_solve` = 7 ticks = 14 ms (~71 Hz) | `MC:438, 918-919` |
| integrator | `implicitfast` | `config/simulation.yaml:2` |

---

## 1. Process-level flow (what surrounds `runController`)

### 1.1 Start-up (`SR:224-278`, `apps/main_helper.cpp:18-73`)
1. `main m` → `SimulationRunner(RobotType::MIT_HUMANOID, ctrl, headless)`; `init()`; `run()`.
2. `init()`: `setActiveRobotType`; keyboard limits ← `user_command_filter.{x,y,psi}_dot_max` (`SR:231-233`);
   model = `mj_loadXML(model.xml_path)`; `configureSimulationModel` sets `opt.timestep = 0.002`, integrator;
   `mj_forward`; `_keyboardInputEnableTime = data->time + startup.post_init_standing_settle_time` (= 1.0 s) (`SR:270-271`).
3. `run()` → `runPhysicsLoop`: `while(!stop) { runRobotControl(); mj_step(); ++_iterations; ... }` (`SR:337-343`).
   Headless: optional real-time throttle, stop at `CONVEXMPC_HEADLESS_STOP_TIME` (`SR:316-324, 353-355`).

### 1.2 Every physics step: `runRobotControl()` (`SR:385-443`)
```
1. first call only: setupRobotParams (masses, indices, bindings), resize states,
   LegSwingDynamicsProvider(Lazy), robotRunner.init(&params, timestep, &_userCommand)   SR:386-407
2. updateReducedBodyMassPropertiesFromData(model, data)  -> params.bodyMass, bodyInertia, bodyComLocation  SR:409
3. fillCheaterState(model, data)  -> torso pose/vel/acc, q, qd, actuator_force, foot pos/vel/R_WF,
   contact flag + contactForce_W + contactNormalForce                                   SR:411
4. StateEstimator.update -> copy + yaw unwrap (psi, yaw_W_unwrapped, yawRate_W)         SR:412
5. maybeStartKeyboardCommand(time)  (starts at t >= 1.0 s)                              SR:413
6. _userCommand = keyboard.getUserCommand()                                             SR:414
7. applyHeadlessUserCommandSchedule(time)   (env-var driven overrides, headless only)   SR:415
8. robotRunner.prepareController(state)  -> if leg init complete: MyController::prepareController()
   -> syncLocomotionFSM()   [FSM update #1 of this tick]                                SR:416, RR:73-82, MC:476-486
9. legSwingDynamicsProvider.update(state, legDynamicsRequest())   -> per-leg Jv,Jw,JvDot,M,bias
   (walking) or combined 6xN standing Jacobians (standing)                              SR:420
10. robotRunner.run(state, command)        (section 1.3)                                SR:427
11. dashboard publish, applyFixedJointCommands (MIT: none), updateDebugVisualization,
    applyRobotCommand (clamp tau to actuator ctrlrange -> data->ctrl), telemetry       SR:428-442
```
Reduced-body mass properties (`sim/src/setupRobotParams.cpp:398-491`): **every tick**,
`bodyMass` = sum of masses of all bodies NOT in a leg subtree (torso + arms; legs excluded),
`bodyComLocation = Rz(psi)^T (com_upper_W - torsoPos_W)` (yaw-frame offset from torso origin),
`bodyInertia = Rz(psi)^T I_upper_W Rz(psi)` (parallel-axis about the upper-body COM, yaw frame).
`psi` here is `atan2(R_WT(1,0), R_WT(0,0))` (wrapped) — used only to build the offset/inertia.
The loop is over **every** body with id ≥ 1, `body_mass > 0` and not in a leg subtree, so any other
massive body in the scene XML is included too (e.g. the G1 `scene_23dof.xml` debug-marker mocap bodies,
`models/unitree_robots/g1/scene_23dof.xml:21-39`, have geoms with default density, ≈0.04–0.07 kg each,
positioned at the marker targets). See O3.

**State timing (no `mj_forward` before reading).** Steps 2–3 read `mjData` directly after the previous
`mj_step`; neither `fillCheaterState` nor `updateReducedBodyMassPropertiesFromData` calls
`mj_forward`/`mj_kinematics` (`SR:338-341, 409-412`; the only `mj_forward` calls in `sim/src` are at init
`SR:269` and on the auxiliary models `LegSwingDynamicsProvider.cpp:630, 719`). `mj_step` computes
kinematics/velocities/contacts and then integrates, so:
* **pre-integration** (state of the previous tick): `xpos/xquat/xmat/xipos/ximat`, `site_xpos/site_xmat`
  (torso pose, foot position, `R_WF`), `mj_objectVelocity` (torso lin/ang velocity, site foot velocity,
  cvel-based), `mj_objectAcceleration`, `data->contact` / `mj_contactForce` (contact flags and normal force),
  the reduced-body mass properties (`xipos/ximat/xmat/xpos`), and `actuator_force` (`tauEstimate`,
  the force applied during the last step);
* **post-integration** (current): `qpos/qvel` (leg and arm `q`, `qd`) and `data->time`.
The auxiliary leg models then combine the current leg `q/qd` with the one-step-old torso pose
(`LegSwingDynamicsProvider.cpp:611-630`). For parity the Python loop must read the same `mjData`
fields directly after `mj_step` without calling `mj_forward`/`mj_kinematics` (calling them shifts
kinematics and contacts by one 2 ms tick).

Contact normal force (`sim/src/MujocoCheaterStateReader.cpp:63-89`): for every `data->contact[i]`
where geom1 or geom2 belongs to the foot body (any geom whose `geom_bodyid == foot body`, or listed
collision geom), `normalForce += |mj_contactForce(...)[0]|`; `contact = (count>0)`;
`hasContactForce = true` always in simulation (`:363`).

### 1.3 `RobotRunner::run` (`RR:105-152`)
```
setupStep: zero leg/arm commands+data; copy q, qd, tauEstimate; if hasFootJacobians copy
   (footPos_W, footVel_W, Jv_W, JvDot_W, Jw_W); if hasLegDynamics copy (massMatrix, bias)   RR:188-261
arm: ArmPosInitializer.IsInitialized() EVERY tick (spline clamps at end -> holds final pose),
   apply joint_tracking.arm gains                                                            RR:113-117
leg: LegPosInitializer.IsInitialized() ; if not initialized: JointPd with joint_tracking.leg gains
     else: robot_ctrl->runController()                                                       RR:119-142
enable controllers; composeCommand: tau = 0; legController.updateCommand(tau); arm...      RR:144-158
```
Leg PD init spline: from current q to `qpos0 + initial_pose.leg_joint_offsets`
(`[0,0,-0.735,1.2,-0.70]` per leg, applied per limb, `robot/include/robot/InitialPoseConfig.h:20-36`,
`robot/src/LegPosInitializer.cpp:61-99`) over `leg_initialization_time = 2.0 s`; done when
`_curr_time >= 2.0` (`LegPosInitializer.cpp:36-57`). Arms: `qpos0 + [0,0,0,-1.65]` over 1.0 s.
Exact timing: control runs **before** `mj_step`, starting at `data->time = 0` (`SR:337-341`), so control
call #n happens at `t = (n-1)*0.002`. The initializers take `float dt` (`LegPosInitializer.cpp:9-13`) and
add `float(0.002) = 0.0020000000949949` to `_curr_time` **before** the `>= end_time` test (`:36, 57`).
The leg test first passes on call #1000, i.e. at `t = 1.998 s`, and `runController` runs on that same call
(`RR:120-141`). Arms finish on call #500 (`t = 0.998 s`).
So **`runController` first executes at t = 1.998 s** (`HorizonClock.t0` and the FSM start time are 1.998)
and is the only thing producing leg torques after that. (Checked numerically.)

### 1.4 MIT time-line with the yaml as shipped
| sim time | event |
|---|---|
| 0 – 1.998 s | leg PD initialisation (JointPd, control calls #1–#999), arms PD (spline done at call #500, t = 0.998 s, then hold) |
| 1.0 s | keyboard input enabled (`SR:270-271, 445-452`); headless `x_dot` schedule starts by default (sec 3.1) |
| 1.998 s | first `runController` (call #1000): `initializeController` → FSM `StandingSettle` (mode **Standing**), `HorizonClock.t0 = 1.998`, MPC solved at `_iteration==0` |
| ≈3.000 s | settle elapsed (`time - 1.998 >= 1.0` first holds 501 ticks later because of floating-point accumulation of `data->time`) → FSM `Walking` (in `prepareController`, FSM update #1): gait clock reset (`t0 = time`), swing state + contact manager reset; gait begins (both feet stance for the first 0.08 s, then left swings first) |

---

## 2. `runController()` tick order (500 Hz) — `MC:1374-1429`

```
 0. if !_initialized: initializeController()                                           MC:1375-1377  (once)
 1. syncLocomotionFSM()          [FSM update #2 of this tick; see 1.2 step 8]          MC:1386
 2. _horizonClock->sync(time)    (advance t0 by whole cycles so 0 <= time-t0 < cycle)  MC:1387, HC:18-26
 3. x0 = buildCurrentMpcState()  (13-vector, sec 0.2)                                 MC:1389
 4. dt = max(0, time - _lastControlTime)   (0 on the first tick and on reset ticks, sec 3.2)  MC:1390
 5. updateFilteredUserCommand(dt)         (sec 3.2)                                    MC:1391
 6. updateBodyTarget(x0, dt)              (sec 3.3)                                    MC:1392
 7. swingFootPlanner.setBodyYawTargetWorld(_bodyTarget.euler_W[2])                     MC:1393-1395
 8. nominalDesiredFootPositions =
       Standing: {footPos_W(L) with z=-0.005, footPos_W(R) with z=-0.005}              MC:1396-1404
       Walking : swingFootPlanner.desiredFootPositions()  (also calls horizonClock.sync) MC:1405
 9. contactManager.update(state, params, gait, mode, nominalDesiredFootPositions)      MC:1406-1412  (sec 6)
10. desiredFootPositions = contactManager.managedFootPositions(nominal)                MC:1413-1416
11. updateSwingTrajectories(desiredFootPositions)  (per-leg swing curves; sets _lastControlTime=time;
    registers braking touchdowns)                                                      MC:1418  (sec 7)
12. updateTouchdownDebugTarget(...)   (debug markers only)                             MC:1419
13. updateStandingMpcDebugRequest()   (debug logging only)                             MC:1420
14. maybeUpdateMpc(x0, desiredFootPositions)  -> only when (_iteration==0) or
    (_iteration - _lastMpcIteration >= 7): constraints, reference, formulation, QP solve,
    _stanceWrenchWorld = first-step wrench                                             MC:1421  (sec 8)
15. writeLegCommands()  (every tick; uses held _stanceWrenchWorld and ramp alpha)      MC:1424  (sec 9)
16. maybeWriteStandingMpcDebugLog(...)  (debug only)                                   MC:1426
17. ++_iteration                                                                       MC:1428
```
Arm commands are not touched by `runController`; they come from `RobotRunner::run` (sec 1.3).
Torque is composed after `runController` returns (`RR:146-149`).

### 2.1 Every tick vs MPC tick
| computed every tick (500 Hz) | computed only on MPC ticks (every 7th tick, plus tick 0) |
|---|---|
| FSM (twice), horizon clock sync, x0, command filter, body target, swing planner targets, contact manager, swing trajectories, leg commands (torques) | foot local x-axes → gait scheduler (only if `no_roll_moment`), contact horizon override, `buildConstraintMatrices`, reference seed & `ReferenceTrajectory.build`, `MPCFormulation.build`, `ConvexMPC.updateInput/solve`, `_stanceWrenchWorld` |

**Held between solves**: `_stanceWrenchWorld` (12-vector) is the **first horizon step** of the last
solution (`ConvexMPC.cpp:991`, `MC:1010`) and is held **constant** (no interpolation, no re-use of
later horizon steps) until the next solve. Per tick it is multiplied by the contact ramp alpha and
the yaw-hold moment is added (sec 9). If the solve throws, a gravity-split fallback replaces it (sec 8.4).
`_lastMpcIteration = _iteration` is set after each attempt, success or failure (`MC:1054`).

Consequences of the tick order an implementer must reproduce (`MC:918-921, 1010, 1054, 1313-1331, 1421-1428`):
* The solve (step 14) runs **before** `writeLegCommands` (step 15) on the same tick, so a new solution is
  applied on the tick it is computed. On an MPC tick the step-0 override is built from the
  `activeContact`/alpha computed in step 9 of **this** tick.
* Solves happen at `_iteration = 0, 7, 14, …`. `_iteration` is **never reset** on FSM transitions, so the
  solve phase relative to the gait clock is arbitrary (whatever `_iteration mod 7` is at walking start).
* Between solves the held first-step wrench is used unchanged, per foot, multiplied by that foot's
  **current** alpha. A foot that becomes active between solves (scheduled touchdown or early contact) was
  inactive in the last step-0 override, so its held `F`/`M` segment is zero (non-stance wrench forced to 0):
  it carries **zero wrench until the next solve** (up to 6 more ticks, 12 ms), even once alpha reaches 1
  (only the yaw-hold moment, sec 9.1, acts). A foot that lifts off between solves goes to `SwingFoot` and its
  share is simply dropped; the other foot's held share is **not** redistributed.

### 2.2 `initializeController` / `initializeRuntimeObjects` (`MC:393-474`)
Creates `HorizonClock(time)`, `GaitScheduler`, `ContactManager(cfg.contactManager)`, `SwingFootPlanner`
(pointer to `_filteredUserCommand`), `MPCFormulation`, `ConvexMPC`, `LocomotionFSM(...)` (sec 4),
sets foot end-effector source, swing gains (`naturalFrequency`, `kdDiag`, `height`),
`_iterationsBetweenMpc = max(iterations_between_solve,1)`; applies FSM initial output
(mode, dynamics request, `gaitScheduler.setLocomotionMode`), `contactManager.reset(...)`,
`updateFilteredUserCommand(0)`, per-leg runtime `{wasInStance = c(side,time), wasSearchMode=false,
touchdownYaw_W = swingFootYawTargetWorld(side)}`, zero wrench, `_iteration=_lastMpcIteration=0`,
`_lastControlTime=time`, `_bodyTarget = {}` (uninitialised), `_zeroMotionCommand=false`.

`legDynamicsRequest()` before init (`MC:488-504`): walking requested with settle ≤ 0 → swing dynamics;
otherwise standing Jacobians. After init it is the FSM output (`swingLegDynamics = mode==Walking`,
`standingFootJacobians = mode==Standing`, `FSM:285-286`).

---

## 3. Command pipeline

### 3.1 Keyboard producer (`KB`, `common/include/Utilities/KeyboardCommand.h:39-45`)
Only the filtering/limits matter for the port:
* step sizes: `_linearStep = 0.05 m/s` (w/s → x_dot, a/d → y_dot), `_yawStep = 0.1 rad/s` (q/e),
  `_heightStep = 0.01 m` (arrow up/down), orientation step `2°` (i/k pitch, j/l roll, no limit).
* limits: `_xLimit/_yLimit/_yawLimit` ← yaml `x_dot_max 0.7 / y_dot_max 0.5 / psi_dot_max 2.0`
  (`SR:231-233`), `_heightLimit = 0.8` (hard-coded). `applyLimitedDelta` clamps to `[-limit, limit]`
  (`KB:106-129`, `clampWithOptionalLimit` `KB:39-44`, infinite limit = no clamp).
* `sanitizeCommand`: any |value| < 1e-12 → 0 (`KB:21-27, 46-53`).
* `'t'`: **clears x_dot,y_dot,psi_dot,height,roll,pitch** and increments `locomotion_mode_toggle_request` (`KB:337-350`).
* space: clears all motion values, keeps the two counters (`KB:355-366`).
* `'L'` increments `standing_mpc_debug_log_request` (`KB:328-335`).
* When "standing controls" are active (`legDynamicsRequest().standingFootJacobians`, `SR:31-33, 417-419`),
  w/s/a/d/q/e are ignored (`KB:368-386`). **This restriction applies only to keyboard keys.**
* Headless auto-walk (`SR:454-537`): env `CONVEXMPC_HEADLESS_AUTO_WALK`, `..._X_DOT_FINAL`, `..._X_DOT_RATE`
  (ramp rate m/s²; `x_dot = sign(final)*min(|final|, rate*(t-start))`, or `final` directly if rate ≤ 0),
  `..._X_DOT_START_TIME` (default = keyboard enable time 1.0 s),
  `..._WALK_TOGGLE_TIME` (default 1.2 s; sets `toggle_request = 1`), `..._X_DOT_PROFILE="t:v,t:v"`
  (piecewise constant; the first point's value applies even before its time).
  x_dot only; y_dot/psi_dot cannot be scheduled headless (must be added for the lateral/turn tests).
  The schedule writes `_userCommand.x_dot` **regardless of FSM mode** (`SR:466-473, 516-524`), and the
  controller filters and uses it in Standing mode too (`MC:610-621, 980`). With the default start time
  (1.0 s) `x_dot` is already nonzero when the controller starts (1.998 s) and during `StandingSettle`
  (≈2.0–3.0 s): the filtered command is set to the raw value on the first controller tick (sec 3.2), and the
  standing MPC reference carries that velocity (sec 10). For parity with the reference demo either replicate
  this or set the start time ≥ walking start (≈3.0 s); the port must state which one it uses.

### 3.2 Controller-side filter `updateFilteredUserCommand(dt)` (`MC:610-654`)
```
raw = clampUserCommand(*_userCommand)             # x,y,psi clamped to ±max   CC:649-656
if _zeroMotionCommand: raw.x_dot = raw.y_dot = raw.psi_dot = 0          # FSM BrakingToStanding
if first call: filtered = raw; return
prev = filtered; filtered = raw                    # non-filtered fields pass through
if raw.x_dot == 0 and raw.y_dot == 0 and raw.psi_dot == 0:
    filtered.{x,y,psi}_dot = 0                     # instant stop, no low-pass
else:
    a(tau) = clamp(1 - exp(-dt/tau), 0, 1)  (tau<=0 or dt<=0 -> 1)        MC:28-33
    filtered.x_dot   = prev.x_dot   + a(x_dot_tau=0.92) * (raw.x_dot   - prev.x_dot)
    filtered.y_dot   = prev.y_dot   + a(y_dot_tau=0.80) * (raw.y_dot   - prev.y_dot)
    filtered.psi_dot = prev.psi_dot + a(psi_dot_tau=0.70)*(raw.psi_dot - prev.psi_dot)
filtered.body_height_offset_m = raw.body_height_offset_m                 # no filter
filtered.standing_roll_offset_rad  = prev + a(0.70)*(raw - prev)
filtered.standing_pitch_offset_rad = prev + a(0.70)*(raw - prev)
filtered = clampUserCommand(filtered)
```
With dt = 0.002 s and tau = 0.92 s, a ≈ 0.00217/tick; a 0.6 m/s step reaches 95 % after ≈ 2.8 s.

**Ticks with dt = 0 (no low-pass).** `dt` is exactly 0, so `a = 1` and `filtered = raw` for x/y/psi_dot
and the roll/pitch offsets, on:
(a) the first controller tick (`initializeRuntimeObjects` sets `_lastControlTime = time`, `MC:470`; the
    first-call pass-through `updateFilteredUserCommand(0)` also ran at init, `MC:448`);
(b) every tick on which the FSM output has `resetSwingState` — every transition except into
    `BrakingToStanding`, **including `StandingSettle → Walking` at walking start**. The transition happens in
    `prepareController` (FSM update #1, `SR:416` before `robotRunner.run` at `SR:427`), and
    `resetSwingState()` sets `_lastControlTime = time` (`MC:552`) before `runController` computes `dt`
    (`MC:1390-1391`). Whatever raw command is present on the walking-start tick passes through unfiltered.
On these ticks the body-target yaw is not advanced (`advanceYaw` needs `dt > 0`) and the swing
trajectories get `dt = 0`. The port must reproduce this: set `last_control_time = t` inside the
FSM-transition handler, before the command filter runs.
Consumers of `_filteredUserCommand`: swing planner (by pointer), reference trajectory (copy, `MC:980`),
body-target yaw integration, swing-foot yaw target, stop-recenter logic, dashboard.

### 3.3 Body target `updateBodyTarget(x0, dt)` (`MC:677-736`)
State `_bodyTarget = {nominalPosition_W, position_W, nominalHeight_W, euler_W, eulerSeed_W, initialized}` (`My_Controller.h:41-50`).
First call (`MC:682-697`): MIT yaml has `initial_pose.base_position_W = [0,0,0.679472]`, `base_rpy_W = [0,0,0]`
→ `nominalPosition_W = basePos + Rz(0)*bodyComLocation` (`MC:356-361`), `euler_W = [0,0,0]`,
`eulerSeed_W = euler_W`, `nominalHeight_W = nominalPosition_W.z`. (Without a base pose it would seed from x0.)
The target is **never re-seeded** on FSM transitions (`seedBodyTargetFromCurrentState` call is commented out, `MC:571`).
Every tick:
```
Standing:  nominalPosition_W.xy = mean of foot end-effector xy over legs       MC:711-715, 338-354
           euler_W[2] += psi_dot*dt   (advanceYaw, mode 'always' for MIT)       MC:716-721
Walking:   nominalPosition_W.xy = x0[3:5]  (current COM xy)                     MC:728
           euler_W[2] += psi_dot*dt                                             MC:729-734
both:      euler_W[0] = eulerSeed[0] + standing_roll_offset (filtered)
           euler_W[1] = eulerSeed[1] + standing_pitch_offset
           nominalPosition_W.z = nominalHeight_W + body_height_offset_m
           position_W = nominalPosition_W                                       MC:704-709
```
`advanceYaw` (`My_Controller/src/BodyMotionReference.cpp:26-36`): returns `yaw + psi_dot*dt` if `dt>0`
and `shouldAdvanceYaw` (mode `always` → true; `single_support` → not both-feet-stance; `double_support` → both stance).
Note: `euler_W[2]` is an open-loop integral of the yaw command starting at the initial-pose yaw (0);
it is NOT the MPC yaw reference in walking (that is anchored to `x0[2]`, sec 8.2) but it IS the yaw the
swing planner uses for foot placement (`setBodyYawTargetWorld`, `MC:1394`, `SwingFootPlanner.cpp:70-78,171-173`).

### 3.4 Swing-foot yaw target (`MC:891-909`, `common/include/Dynamics/SwingYawTarget.h:11-34`)
```
previewTime = max(0, (0.5 + halfStanceOffset(|[x_dot,y_dot]|)) * stanceTime)
    halfStanceOffset = 0.26 if speed>0.65, 0.28 if speed>0.60, else 0.37        MC:35-45
swingFootYawTargetWorld()     = yaw_W_unwrapped + swing_foot_yaw_lead_scale(1.0)*psi_dot*previewTime
swingFootYawPsiOffset(side, psi_dot) = clamp(100 deg/rad * |psi_dot|, 0, 20 deg) in rad,
    applied +for Left when psi_dot>0, -for Right when psi_dot<0, else 0
swingFootYawTargetWorld(side) = base + psiOffset(side)
swingFootYawTargetWorldWithPsiOffset = liftAngleNear(base + offset, base)
```
The per-leg `touchdownYaw_W` is latched at swing start (sec 7) and used by the swing attitude
controller, the stance yaw-hold, and the MPC foot-yaw argument.

---

## 4. `LocomotionFSM` (`FSM`, `My_Controller/include/MyController/LocomotionFSM.h`)

### 4.1 Types
`LocomotionMode ∈ {Walking, Standing, Interactive}` (yaml `requested_locomotion_mode`, parse `CC:117-136`:
`walking|walk`, `standing|stand`, `interactive|general`; missing → Walking).
`LocomotionState ∈ {StandingSettle, Standing, Walking, BrakingToStanding}`.
`modeForState`: StandingSettle/Standing → Standing; Walking/BrakingToStanding → Walking (`FSM:90-101`).

Constructor args (`MC:424-433`, `FSM:48-70`): `requestedMode`, `startup.post_init_standing_settle_time`,
`transition.braking_settle_speed_threshold`, `braking_settle_yaw_rate_threshold`, `braking_settle_average_window`,
`braking_settle_hold_ticks`, `braking_timeout_seconds`, `braking_touchdown_count`, `startTime = state.time`.
`_targetMode = Walking if requested==Walking else Standing`.

Initial state (`FSM:72-88`):
```
Walking requested:      settle>0 -> StandingSettle, else Walking
Interactive requested:  settle>0 -> StandingSettle, else Standing
Standing requested:     Standing
```
MIT: `walking`, settle 1.0 → `StandingSettle`.

### 4.2 `update(time, vx_B, vy_B, wz_B)` (`FSM:215-269`), called from `syncLocomotionFSM` (`MC:579-604`)
Inputs (`MC:593-601`): `R_TW = torsoQuat_W^T`; `v_B = R_TW * reducedBodyComVelocityWorld` (full torso rotation,
not yaw only), `w_B = R_TW * torsoAngVel_W`; passes `v_B.x, v_B.y, w_B.z`.
Before `update`, pending toggle requests are forwarded: `while (_lastToggle < cmd.locomotion_mode_toggle_request) { fsm.requestToggle(); ++_lastToggle; }` (`MC:584-591`).
```
justTransitioned = false
if state==StandingSettle and time - stateStart >= settle:            transitionTo(stateFromMode(target)); jt=true
if target==Walking:
    if state in {Standing, BrakingToStanding}:                        transitionTo(Walking); jt=true
else:  # target Standing
    if state==Walking:                                                transitionTo(BrakingToStanding); jt=true
    elif state==BrakingToStanding:
        if not brakingReady:
            if brakingSettleAveragesReady(...): settleTicks += 1
                 if settleTicks >= hold_ticks(3): brakingReady=true; readyStart=time; touchdownCount=0
            else: settleTicks = 0
        elif touchdownCount >= braking_touchdown_count(5) or time - readyStart >= braking_timeout(3.0):
            transitionTo(Standing); jt=true
return makeOutput(jt)
```
`transitionTo(s)` (`FSM:107-127`): **returns immediately if `s == current state`** (no-op); otherwise sets
state and `stateStart=time`, zeroes settle ticks / ready / touchdown count and **clears the sample deque on
every transition**; entering Braking additionally sets `readyStart = settleStart = time`.
(`justTransitioned` is still set true by the caller even though the call itself may be a no-op; in practice
every call site is guarded so the state differs.)

`brakingSettleAveragesReady` (`FSM:161-213`): push `(time, vx, vy, wz)`; drop samples older than
`time - window(1.0 s)` (keep ≥1); if `time - settleStart < window` → false (still filling);
else ready iff `|mean vx| ≤ 0.01 && |mean vy| ≤ 0.01 && |mean wz| ≤ 0.01`.

`requestToggle` (`FSM:129-148`): **only in Interactive mode**; ignored while state is StandingSettle,
BrakingToStanding, or target≠current mode; otherwise flips `_targetMode`. For the MIT yaml
(`walking`) toggles are no-ops, so braking never occurs; the headless `toggle_request=1` is harmless.

`registerBrakingTouchdown` (`FSM:150-154`): `++touchdownCount` only if state==Braking and ready.
Called from `updateSwingTrajectories` when a leg becomes active-contact after not being in stance (`MC:809-813`).

### 4.3 Output (`FSM:275-289`) and how the controller applies it (`MC:555-577`)
```
mode = modeForState(state)
resetGaitClock  = jt and state != BrakingToStanding      -> horizonClock.reset(time)
resetSwingState = jt and state != BrakingToStanding      -> resetSwingState()  (MC:534-553):
        each leg: swingTrajectory.deactivate(), wasInStance = c(side,time), wasSearchMode=false;
        swingFootPlanner.reset(); contactManager.reset(state, params, gait, mode); _lastControlTime=time
acceptVelocityCommand = mode==Walking and state != Braking   (not used by MC)
zeroMotionCommand = state == BrakingToStanding                -> _zeroMotionCommand (sec 3.2)
dynamicsRequest.swingLegDynamics = mode==Walking; standingFootJacobians = mode==Standing
```
Also every call: `_locomotionMode = mode; _legDynamicsRequest; gaitScheduler.setLocomotionMode(mode)`.

**Double update per tick**: `syncLocomotionFSM` runs in `prepareController` (`SR:416` → `RR:79-81` → `MC:485`)
and again at the top of `runController` (`MC:1386`), both with the same `time`. Consequences:
`_brakingSettleTicks` increments twice per tick (hold_ticks 3 is reached after 2 controller ticks),
the settle deque holds duplicate samples (means unaffected), transitions happen in the first call.
On the very first controller tick `prepareController` returns early (`_initialized` false), so only one update.

### 4.4 What changes between Standing and Walking (summary; details in the referenced sections)
| item | Standing (mode) | Walking (mode) |
|---|---|---|
| gait `c(side,t)` | always true, `p=0` (`GS:98-120`) | periodic (sec 5) |
| contact manager | forced active=1, alpha=1, no early/late (`CM:245-264`) | full logic (sec 6) |
| active contact used by MC | `gaitScheduler.c` (`MC:738-746`) | `contactManager.activeContact` |
| ramp alpha | 1.0 (`MC:748-753`) | `contactManager.contactRampAlpha` |
| desired foot positions | current foot pos, z=-0.005 (`MC:1396-1404`) | swing planner + contact manager |
| MPC weights | `mpc.standing.state/input_weight_diag` | `mpc.walking.*` (`CC:659-704`, `ConvexMPC.cpp:808-816`) |
| horizon override | none (`MC:953-956`) | `contactManager.buildHorizonOverride()` |
| reference seed | full target pose (roll,pitch,yaw,x,y,z) (`MC:967-970`) | roll,pitch,z from target; x,y,yaw from x0 (`MC:971-978`) |
| leg command | `JointTorque` via combined 6×10 Jacobians, no ramp/yaw-hold (`MC:1254-1295`) | per-leg `StanceWrench` / `SwingFoot` (`MC:1308-1371`) |
| leg dynamics request | combined standing Jacobians | per-leg Jacobians + M, bias |
| keyboard | standing controls (no velocity keys) | walking controls |

---

## 5. `GaitScheduler` + `HorizonClock` (nominal schedule)

`HorizonClock` (`HC`): `t0` set at construction/`reset(t)`; `sync(t)`: `while (t - t0 >= cycle) t0 += cycle`;
`tk(k) = t0 + k*dtMpc`. **`tk(0) = t0` is the current cycle origin, not the current time.**
Sync is called in `runController` (`MC:1387`) and in `SwingFootPlanner::desiredFootPositions` (`SwingFootPlanner.cpp:283`);
reset on FSM transitions except into Braking (`MC:564-566`).

Phase and stance (`GS:98-120`), Standing mode short-circuits to `p=0, c=true`:
```
phi(Left) = 0.5, phi(Right) = 0
p(side,t) = fmod((t - t0)/cycle + phi, 1.0)
c(side,t) = 0 <= p < stance/cycle    (= 0.66 for MIT)
bothFeetStance(t) = c(L,t) and c(R,t)
```
MIT per cycle (0.5 s) measured from t0: Right stance [0,0.33), swing [0.33,0.5);
Left stance [0,0.08)∪[0.25,0.5), swing [0.08,0.25). Double support [0,0.08) and [0.25,0.33).

`buildConstraintMatrices(override, leftFootYaw_W, rightFootYaw_W)` (`GS:134-193`) — MPC ticks only:
for `k = 0..24`: `tk = t0 + k*0.02`; `leftStance = c(L,tk)`, `rightStance = c(R,tk)`, min-scales 1.0;
if `override && k < override.steps.size() && steps[k].enabled` → replace stance flags and min-scales
(clamped [0,1]) with the override; then `C_bound[24k+4] = normal_force_max (1500)`,
`C_bound[24k+5] = -minScale*normal_force_min (5)` for a stance left foot, indices `+16/+17` for right;
non-stance feet keep zero bounds (their wrench is forced to zero by the equality rows in ConvexMPC).
`constraintSteps[k]` records stance flags, min scales, resolved foot yaws (NaN → yaw from the stored
foot x-axis) and foot x-axes. Because `tk` starts at `t0`, the horizon (exactly one cycle for MIT)
always shows the **cycle-aligned** contact pattern regardless of the current phase; only the first
`contact_lock_steps` (=1) step is corrected to the measured contact by the override (sec 6.6).
MIT pattern (before the step-0 override): Left stance `k ∈ {0..3, 13..24}`, Right stance `k ∈ {0..16}`.
**Floating-point boundary at Left k = 4**: nominally `p = 0.5 + 0.04·4 = 0.66` = stance fraction (swing),
but `p` is computed as `((t0 + 4·0.02) − t0)/0.5 + 0.5`, which rounds below 0.66 for many `t0` values, making
Left k = 4 **stance**. With the default time-line (t0 = 3.000 + 0.5n) this happens for 223 of the first
400 cycles, from t0 ≈ 16 s on (checked numerically). The port must compute `tk = t0 + k*dtMpc` and
`p = fmod((tk - t0)/cycle + phi, 1)` in exactly this order in float64 to reproduce it.
See open question O1.

---

## 6. `ContactManager` (full) — `CM`, `My_Controller/include/MyController/ContactManager.h`

### 6.1 Parameters (MIT yaml values; defaults in `CCH:109-123`)
| param | MIT | default | meaning |
|---|---|---|---|
| `contactForceOnThreshold` | 36.0 N | 20 | normal force ≥ → candidate ON |
| `contactForceOffThreshold` | 0.5 N | 5 | normal force ≤ → candidate OFF |
| `contactOnConfirmTicks` | 1 | 2 | consecutive ticks to confirm ON (`max(.,1)`) |
| `contactOffConfirmTicks` | 4 | 2 | consecutive ticks to confirm OFF |
| `contactRampDuration` | 0.01 s | 0.08 | ramp of `contactRampAlpha` 0→1 after a leg becomes active |
| `contactLockSteps` | 1 | 3 | number of leading horizon steps overridden with measured contact |
| `lateContactTimeout` | 0.20 s | 0.20 | search time after which `recoveryFailure` is flagged (flag only) |
| `groundSearchVelocity` | 0.40 m/s | 0.08 | downward target speed in search mode |
| `groundSearchMaxDepth` | 0.15 m | 0.04 | max search depth |
| `groundSearchTrackingTime` | 0.08 s | 0.08 | swing-trajectory duration used while in search mode |
| `stanceContactLossFootHeight` | 0.060 m | 0.025 | foot z above which a scheduled-stance foot with no contact counts as "lost stance" (late-contact trigger) |
| `enableEarlyContactHandling` | true | true | |
| `enableLateContactHandling` | **false** | true | all late-contact/search logic disabled for MIT |
Validation (`CC:572-594`): on ≥ off ≥ 0, confirm ticks > 0, ramp ≥ 0, lock ≥ 0, tracking time > 0, others ≥ 0.

### 6.2 Per-leg state
Public (`ContactManager.h:12-30`): `side, scheduledContact, estimatedContact, activeContact, earlyContact,
lateContact, liftoffHold (always false in code), searchModeActive, recoveryFailure, contactRampAlpha,
lateContactTime, liftoffHoldTime (always 0), contactNormalForce, frozenTouchdownPosition_W, commandedFootTarget_W`.
Private (`:62-76`): `initialized, rawContactEstimate, previousActiveContact, previousScheduledContact,
previousLateContact, releasedContactDuringSwing, contactOnTicks, contactOffTicks, contactRampTime, searchDepth`.
Manager: `_lastUpdateTime, _hasLastUpdateTime`.

### 6.3 `reset(state, params, gait, mode)` (`CM:150-198`) — at init and on FSM transitions (not Braking)
```
_lastUpdateTime = time
per leg: scheduled = c(side,time); estimated = (Fn >= on_thr) if hasContactForce else contact bit
         active = true if mode==Standing else scheduled
         early=late=liftoffHold=search=fail=false; alpha = 1 if active else 0; times = 0
         frozenTouchdown = commandedTarget = footPos_W
         initialized=true; raw=estimated; prevActive=active; prevScheduled=scheduled; prevLate=false
         releasedDuringSwing = !scheduled && !estimated; on/off ticks=0
         contactRampTime = rampDuration if active else 0; searchDepth=0
```

### 6.4 Estimated contact with hysteresis `updateEstimatedContact` (`CM:89-148`)
```
Fn = contactNormalForce if (hasContactForce and finite) else 0
if not initialized: estimated = (Fn >= on_thr) [or contact bit]; raw=estimated; ticks=0; return
if no force signal: count contact-bit ticks (on/off)
elif estimated:     if Fn <= off_thr(0.5): offTicks++ ; onTicks=0   else offTicks=0
else:               if Fn >= on_thr(36):   onTicks++  ; offTicks=0  else onTicks=0
if !estimated and onTicks  >= max(onConfirm,1)=1: estimated=true ; ticks=0
elif estimated and offTicks >= max(offConfirm,1)=4: estimated=false; ticks=0
raw = (Fn >= on_thr)
```
MIT: ON after 1 tick ≥ 36 N; OFF after 4 consecutive ticks ≤ 0.5 N (8 ms). Between 0.5 and 36 N the
estimate holds its previous value.

### 6.5 `update(state, params, gait, mode, nominalDesired)` (`CM:200-359`) — every tick, step 9 of sec 2
```
dt = time - _lastUpdateTime (0 if first or time went backwards); _lastUpdateTime = time
per leg (in params.legs order):
  scheduled = c(side,time); contactNormalForce = Fn; estimated = updateEstimatedContact(...)
  nominalTarget = nominal desired position for this side
  if !initialized:   # lazy init (CM:231-243); NOT the same as reset(); unreachable in practice
                     # because reset() is called at init (MC:443)
      initialized=true; prevActive = prevScheduled = scheduled; prevLate=false
      releasedDuringSwing = !scheduled && !estimated
      rampTime = rampDuration if scheduled else 0; searchDepth = 0
      frozen = footPos_W if scheduled else nominalTarget
      # active/alpha/early/... are NOT set here; the normal logic below sets them
  if mode == Standing:
      active=true, early=late=hold=search=fail=false, alpha=1, times=0,
      frozen = commanded = footPos_W, prevActive=prevScheduled=true, prevLate=false,
      releasedDuringSwing=false, rampTime=rampDuration, searchDepth=0; continue
  # ---- walking ----
  if scheduled: releasedDuringSwing=false
  elif !estimated: releasedDuringSwing=true
  early = enableEarly && !scheduled && estimated && releasedDuringSwing
  scheduledTouchdown   = scheduled && !prevScheduled
  continuingLate       = scheduled && prevLate && !estimated
  lostEstablishedStance= scheduled && !estimated && !scheduledTouchdown && !prevLate
                         && footPos_W.z > stanceContactLossFootHeight(0.060)
  late = enableLate(false for MIT) && !estimated && (scheduledTouchdown || continuingLate || lostEstablishedStance)
  liftoffHold=false; liftoffHoldTime=0
  if early && !prevActive: frozen = footPos_W                    # capture actual touchdown point once
  if late:
      if !prevLate: frozen = nominalTarget; lateTime=0; searchDepth=0
      else:         lateTime += dt; searchDepth += groundSearchVelocity*dt
      searchDepth = clamp(searchDepth, 0, groundSearchMaxDepth)
      search=true; fail = (timeout>0 && lateTime > timeout)
      commanded = frozen - [0,0,searchDepth]
  else:
      lateTime=0; search=false; fail=false; searchDepth=0
      commanded = footPos_W if hold else (frozen if early else nominalTarget)
  active = false if late else (true if hold else (true if early else scheduled))
  if !late && !hold && !early && active: frozen = footPos_W           # stance: track actual foot
  if active: rampTime = 0 if !prevActive else rampTime+dt
             alpha = 1 if rampDuration<=0 else clamp(rampTime/rampDuration, 0, 1)
  else:      rampTime=0; alpha=0
  prevActive=active; prevScheduled=scheduled; prevLate=late
```
Ramp for MIT (`rampDuration = 0.01`, dt = 0.002): alpha sequence on the ticks after activation =
0.0, 0.2, 0.4, 0.6, 0.8, 1.0 (first active tick has `rampTime = 0` → alpha 0 → **zero stance wrench on the first tick**).
`activeContact` at a scheduled touchdown therefore starts at the scheduled instant with alpha 0.

**What the ramp scales (complete list)** (`CM:195, 261, 340-353, 430-433`, `GS:165-176`, `MC:748-753, 1317-1347`):
* `alpha = clamp(rampTime/0.01, 0, 1)`, `rampTime = 0` on the first active tick, `+= dt` after that.
  The ramp applies to scheduled and early activations alike.
* Leg command: alpha scales force FF, moment FF and the stance yaw-hold moment; the yaw hold is skipped
  entirely when `alpha == 0`.
* MPC: alpha only scales `Fz_min` of horizon step 0 (the override): `Fz ≥ alpha·5 N`. `Fz_max` (1500),
  friction, CoP and torsion rows are **not** scaled; steps 1–24 use the full 5 N.
* On `reset()` / FSM transitions, active legs start at alpha = 1 (`rampTime` preset to the ramp duration;
  the same-tick `update` has `dt = 0` and `prevActive = active`, so alpha stays 1). Standing mode always uses
  alpha = 1.
* **Liftoff: no ramp.** On the first inactive tick alpha = 0, `rampTime = 0`, and the leg switches to
  `SwingFoot` immediately.
* First activation tick: the leg is in `StanceWrench` with `F = M = 0` and no yaw hold, and `StanceWrench`
  adds no bias/gravity compensation, no `tauFeedForward` and no PD (`LegController.cpp:294-317`), so
  `tau_leg = 0` exactly for 2 ms. This does not happen on reset/transition ticks (alpha = 1). See O6.

### 6.6 Outputs and where they go
* `activeContact(side)` → `MC::activeContactForSide` (walking) → leg mode stance/swing (`MC:1313-1316`),
  swing trajectory deactivation (`MC:805-818`), FSM touchdown counting, MPC failure fallback, foot-yaw for MPC.
* `contactRampAlpha(side)` → multiplies the commanded stance force+moment and the yaw-hold moment (`MC:1317-1347`).
* `searchModeActive(side)` → swing trajectory uses search-mode timing (`MC:824-846`).
* `managedFootPositions(nominal)` (`CM:394-414`): replaces the side's desired position with
  `commandedFootTarget_W` **only if** `search || early || liftoffHold`; otherwise nominal passes through.
  Result feeds the swing trajectory final position (`MC:821`) and the MPC reference lever arms
  `r_left/r_right = desired - p_ref` (`ReferenceTrajectory.cpp:65-66`).
  For a stance foot, the nominal (planner) value is the planner's cached touchdown target latched at the
  start of that foot's last swing (z = −0.005) (`SwingFootPlanner.cpp:334-343`). Exceptions:
  (a) after any planner `reset()` (every walking start) the first stance target of each foot is
  `currentFootTouchdownTarget = footPos_W` measured at the first walking tick (measured z, not −0.005),
  latched until that foot swings (`SwingFootPlanner.cpp:149-151, 337-340`);
  (b) only while `earlyContact == 1` is the frozen measured touchdown point substituted; when scheduled
  contact begins, `early` goes false and the planner's cached target passes through again, so the MPC lever
  arm and swing target jump from the actual landing point back to the planned point
  (`CM:272-274, 321-324, 399-401`);
  (c) the planner decides stance/swing with gait `c(side,t)` (scheduled), not `activeContact`.
  The measured stance-foot position is never used otherwise. See O5.
* `buildHorizonOverride()` (`CM:416-446`): `steps.resize(max(contactLockSteps,0)=1)`; each step
  `{enabled=true, leftContact=active_L, rightContact=active_R, leftNormalForceMinScale = alpha_L if active_L else 0,
  right… }` → `GaitScheduler::buildConstraintMatrices` (sec 5). Only in walking (`MC:953-956`).
  Effect: horizon step 0 uses the measured/managed contact set; `Fz_min` for a freshly-touched foot is
  `alpha*5 N` (0 on the first tick); a scheduled-stance foot that is in search (late) is removed from
  step 0 (zero wrench).

### 6.7 Early contact walk-through (MIT; enabled)
Swing foot (scheduled 0). Once `estimated` drops to 0 during swing, `releasedDuringSwing=1`. When the
foot hits ground early (`Fn ≥ 36 N` for 1 tick): `early=1`, `active=1`, `frozen = footPos_W` (this tick),
`commanded = frozen`. Then:
1. `managedFootPositions` → desired position for that side = frozen actual foot position.
2. `updateSwingTrajectories`: `isStance` → `swingTrajectory.deactivate()`, `wasInStance=true` (sec 7).
3. `writeLegCommands`: mode `StanceWrench` with `alpha` ramping 0→1 over 5 ticks; force/moment
   = `-alpha * _stanceWrenchWorld` segment (held from the last solve — which still assumed swing for
   this foot, so its commanded wrench is whatever the QP had at step 0 for it: zero if step 0 was swing).
4. Next MPC tick: override step 0 marks the foot as stance with `Fz_min = alpha*5`; the reference lever
   arm uses the frozen actual position; steps k≥1 keep the nominal cycle-aligned schedule. The contact
   signature changes → OSQP cold start, no warm start (`ConvexMPC.cpp:880-906`).
5. Early contact persists until `scheduled` turns 1 (then plain stance, `frozen` tracks the foot) or
   `estimated` drops for 4 ticks (then `early=0`, `active=scheduled=0` → back to swing; the swing
   trajectory is re-initialised from the current foot position because `wasInStance` is true).
   The horizon/gait clock are **not** shifted by early contact.

### 6.8 Late contact / ground search (disabled for MIT; description for completeness)
Triggered when `enableLate` and no estimated contact at (a) the scheduled touchdown tick, (b) continuing
from a previous late tick, or (c) an established stance foot whose z > 0.060 m loses contact.
On entry `frozen = nominalTarget`; each following tick the commanded target descends at 0.40 m/s to at most
0.15 m below `frozen`; `active=false` (foot removed from step 0 of the MPC and controlled as swing),
`searchModeActive=true` → the swing trajectory (sec 7) becomes a 0-height, 0.08 s tracking curve toward
the descending target, re-targeted every tick. `recoveryFailure` after 0.20 s is a flag only (nothing acts on it).
Search ends when `estimated` becomes 1 (then `late=0`, `active=scheduled`) — there is no timeout-based exit.
`liftoffHold` is never set (`CM:289`), so the `liftoffHold` branches are dead code.

---

## 7. Swing trajectories `updateSwingTrajectories(desired)` (`MC:792-868`) and the trajectory model

Per leg (`LegRuntimeState`: `swingTrajectory, touchdownYaw_W, wasInStance, wasSearchMode`, `My_Controller.h:52-59`):
```
dt = max(0, time - _lastControlTime); minRemaining = swing.min_remaining_time (0.001)
isStance = activeContactForSide(side,time)
if isStance:
    if FSM state==BrakingToStanding and !wasInStance: fsm.registerBrakingTouchdown()
    traj.deactivate(); wasInStance=true; wasSearchMode=false; continue
p_now = footPos_W; target = desired[side]; fallbackYaw = swingFootYawTargetWorld(); psi_dot = filtered
if contactManager.searchModeActive(side):
    T = max(ground_search_tracking_time(0.08), minRemaining)
    if !wasSearchMode or !traj.active(): touchdownYaw = yawWithPsiOffset(psi_dot, side, fallbackYaw)
                                         traj.reset(p_now, target, height=0.0, T)
    else: traj.setFinalPosition(target); traj.advance(dt)
    wasInStance=false; wasSearchMode=true; continue
wasSearchMode=false
T = max(remainingSwingTime(side,time), minRemaining)
    remainingSwingTime = clamp(cycle*(1 - p(side,time)), 0, swingTime)      MC:59-61
if wasInStance or !traj.active():
    touchdownYaw = yawWithPsiOffset(psi_dot, side, fallbackYaw)
    traj.reset(p_now, target, swing.height(0.06), T)
else: traj.setFinalPosition(target); traj.advance(dt)
wasInStance=false
_lastControlTime = time
```
`SwingFootTrajectory` (`My_Controller/src/SwingFootTrajectory.cpp`): `reset(pInit,pFinal,h,T)` sets
`remaining=T, active=true`; `advance(dt)`: `remaining = max(0, remaining-dt)`, deactivates at 0;
`s = 1 - remaining/T`, cubic blend `b(s)=3s²-2s³`; xy = lerp(pInit,pFinal,b(s)); z: for s≤0.5 blend
`pInit.z → pFinal.z + h` with `u=2s`, for s>0.5 blend `pFinal.z + h → pFinal.z` with `u=2s-1`;
velocities/accelerations are the analytic derivatives with `ds/dt = 1/T` (`:79-127`).
`setFinalPosition` is called every tick, but in ordinary walking the planner target is latched at swing
start (`wasInStance` or invalid cache) and held for the rest of the swing (`SwingFootPlanner.cpp:345-359`).
It changes mid-swing only when stop-recenter has just activated, while a turn-stop frame is valid
(`_turnStopFrameValid`, re-computed every tick then), or when the contact manager substitutes a commanded
target (search/early). When `traj` finishes before touchdown (remaining=0, inactive) the next tick
re-initialises it from the current foot position with `T = minRemaining = 1 ms` (effectively a
position hold on the target). `desired[side].z = -0.005` (planner constant, `SwingFootPlanner.cpp:10`),
so the foot is driven slightly below the ground plane at touchdown.

---

## 8. MPC update `maybeUpdateMpc(x0, desired)` (`MC:911-1055`) — MPC ticks only

### 8.1 Setup
```
mpcFootYaw(side) = if activeContact(side): liftAngleNear(atan2(R_WF.col(0).y, R_WF.col(0).x), touchdownYaw_W[leg])
                   else touchdownYaw_W[leg]                                              MC:930-946
if contactWrenchModel(mode)==NoRollMoment: gaitScheduler.setFootLocalXAxesWorld(R_WF_L.col(0), R_WF_R.col(0))
   (MIT: full_wrench for both modes -> skipped)                                          MC:948-952
override = contactManager.buildHorizonOverride() if walking else none                   MC:953-956
gaitScheduler.buildConstraintMatrices(override, mpcFootYaw(L), mpcFootYaw(R))            MC:958-964
```
### 8.2 Reference seed (`MC:966-978`)
```
seed = x0
Standing: seed[0:3] = _bodyTarget.euler_W; seed[3:6] = _bodyTarget.nominalPosition_W; seed[5] = nominalHeight_W
Walking : seed[0] = euler_W[0]; seed[1] = euler_W[1]; seed[5] = nominalHeight_W   (x,y,yaw,velocities from x0)
```
`ReferenceTrajectory(&filteredCommandCopy, seed, desired, horizonClock, gaitScheduler).build(out)`
(`ReferenceTrajectory.cpp:8-83`): `p_ref = seed[3:6]`, `p_ref.z += body_height_offset`; for k>0
`psi_ref += psi_dot*dt` (mode `always`), `p_ref += Rz(psi_ref)*[x_dot,y_dot,0]*dt`, z reset to `seed[5]+offset`;
`X_ref[k] = [euler(seed roll,pitch), psi_ref, p_ref, 0,0, psi_dot, Rz(psi_ref)*[x_dot,y_dot,0], g]`;
`r_left[k] = desired.left - p_ref[k]`, `r_right[k]` likewise (**constant foot positions over the horizon**);
`psi[k]`, `tk[k]` also stored.
### 8.3 Formulation and solve
`MPCFormulation.build(ref, out)` (`MPCFormulation.cpp:39-107`): per step `R_k=Rz(psi_k)`,
`I_k = R_k I_body R_k^T` with `I_body = params.bodyInertia` (yaw frame, updated every tick), mass =
`params.bodyMass` (upper body only), ZOH-ish 3rd-order discretisation, lifted `A_qp (325×13)`, `B_qp (325×300)`.
`ConvexMPC.updateInput(gait, formulation, ref, x0, mode)` then `solve()`; weights `getL(mode)/getK(mode)`
(`CC:668-704`), rotated into the reference-yaw frame per step (`ConvexMPC.cpp:675-685, 847`).
Cold start when the stance signature (`leftStance,rightStance` per step) changes (`ConvexMPC.cpp:880-906`);
warm start otherwise, shifted by one step when `use_shifted_warm_start` (`:1094-1108`).
`_stanceWrenchWorld = solution[0:12]` (`MC:1010`, `ConvexMPC.cpp:991`).
### 8.4 Failure fallback (`MC:1016-1052`)
Any exception (OSQP not solved twice, dimension errors, non-finite foot axes) → log summary,
`_stanceWrenchWorld = 0`, then for each active-contact leg `Fz = bodyMass*|g| / nStanceLegs`
(index 2 for Left, 5 for Right), all other components 0. Controller continues; `_lastMpcIteration` updated.

---

## 9. Leg commands `writeLegCommands()` (`MC:1297-1372`) and torque (`common/src/Controllers/LegController.cpp`)

### 9.1 Walking
```
per leg: isStance = activeContactForSide(side, time)
stance:  mode = StanceWrench; alpha = contactRampAlpha(side)
         forceFF_W  = -alpha * F_side (from _stanceWrenchWorld)
         momentFF_W = -alpha * M_side
         if enable_stance_foot_yaw_hold (true) and alpha>0:
             M_hold = (stance_yaw_kp(20)*e_yaw - stance_yaw_kd(4)*yawRate) * e_z,   MC:86-117
             e_yaw = atan2(e_z·(x̂_proj × d̂), x̂_proj·d̂), x̂_proj = horizontal projection of R_WF.col(0),
             d̂ = [cos(touchdownYaw_W), sin(touchdownYaw_W), 0], yawRate = e_z·(Jw_W qd)
             momentFF_W.z += alpha * M_hold.z
swing:   mode = SwingFoot
         kpCartesian = diag(naturalFrequency²) * diag(Λ), Λ = (Jv M^-1 Jv^T + 1e-9 I)^-1     OperationalSpaceDynamics.cpp:34-58
         pDes/vDes/aDes = swingTrajectory.position/velocity/acceleration
         kdCartesian = diag(kd_diag) = diag(25,25,25)
         tauFeedForward += computeSwingAttitudeLevelTorque(legState, legData, touchdownYaw_W,
              roll_kp 0, roll_kd 0, pitch_kp 300, pitch_kd 18, yaw_kp 305, yaw_kd 18)   SwingAttitudeControl.h:9-84
              = Jw_W^T * [ (pitchKp*e_pitch - pitchKd*rate_pitch) ŷ_F + (yawKp*e_yaw - yawKd*rate_yaw) e_z ]
```
Swing attitude terms (`SwingAttitudeControl.h:37-80`): x̂_F, ŷ_F, ẑ_F = normalized columns of `R_WF`
(full-model site frame), `ω = Jw_W * qd_leg` (aux-model Jacobian, torso rate excluded), `e_z = [0,0,1]`:
```
e_pitch    = atan2( ŷ_F·(ẑ_F × e_z), ẑ_F·e_z ),   rate_pitch = ŷ_F·ω
x̂proj      = normalize(x̂_F - (x̂_F·e_z) e_z),  d̂ = [cos ψ_td, sin ψ_td, 0]   (ψ_td = touchdownYaw_W)
e_yaw      = atan2( e_z·(x̂proj × d̂), x̂proj·d̂ ),  rate_yaw = e_z·ω
(roll term: e_roll = atan2(x̂_F·(ẑ_F × e_z), ẑ_F·e_z), rate = x̂_F·ω, axis x̂_F — skipped, roll_kp = roll_kd = 0)
```
A term is enabled iff `kp > 0 or kd > 0`; if none is enabled the torque is 0. The function throws if a foot
axis is non-finite or has norm ≤ 1e-9, or if `|x̂proj| ≤ 1e-9` (an exception here propagates out of
`runController`).
Torque (`LegController.cpp:320-339`):
* `StanceWrench` → `tau = Jv_W^T forceFF_W + Jw_W^T momentFF_W` (`:294-317`, `OperationalSpaceDynamics.cpp:87-94`);
  **no gravity/bias compensation, no joint PD** (commented out).
* `SwingFoot` → `tau = Jv^T (F_ff + kp(pDes-p) + kd(vDes-v)) + Jv^T Λ (aDes - JvDot qd) + bias + tauFeedForward`
  (`:262-291`, `OperationalSpaceDynamics.cpp:61-84`); `F_ff = 0` for swing.
* Jacobians/M/bias come from the per-leg auxiliary fixed-base model (base body placed at the torso pose,
  other leg and arms deleted; `sim/src/LegSwingDynamicsProvider.cpp:325-434, 603-689`): `Jv_W, Jw_W` are
  the 3×5 site Jacobian columns of that leg's joints, `M` the 5×5 joint-space mass matrix, `bias = qfrc_bias`
  (Coriolis + gravity) of the leg-only model. Both stance and swing legs get this data in walking.
  The aux model deletes the free joint, the other leg and the arms, places the torso at the measured
  (pre-integration, sec 1.2) pose and uses the current leg `q/qd` with **zero base velocity**
  (`LegSwingDynamicsProvider.cpp:336-356, 611-630, 683-687`). Hence:
  - `M` equals the full model's `qM` sub-block of that leg's dofs at the same (torso pose, leg q), because
    that block depends only on the leg subtree;
  - `bias` = gravity + Coriolis from leg `qd` only, and `JvDot` has no base-motion terms — both **differ**
    from full-model `qfrc_bias` / `mj_jacDot` with the real base twist;
  - foot angular rates in the swing attitude and stance yaw hold are `Jw_W*qd_leg`: foot rate relative to
    the torso, torso angular velocity excluded;
  - `p_W`, `v_W` used by the swing PD are the absolute world values from the full model
    (`RR:223-236`, cheater state).
  Python equivalent (see O7): evaluate a separate `MjData` (or aux model) at `qpos = [torso pose read from
  xpos/xquat, current leg qpos]`, base `qvel = 0`, leg `qvel = qd`, `mj_forward`, then take the leg-dof
  columns of `mj_jacSite`, the leg block of `qM`, `qfrc_bias[leg dofs]` and `mj_jacDot`. (Using the full
  model's own post-`mj_step` data instead gives pre-integration leg q and the real base twist in bias —
  not parity.)

### 9.2 Standing `writeStandingLegCommands()` (`MC:1254-1295`)
Requires `state.standingFeet.hasFootJacobians` with `Jv_W, Jw_W ∈ R^{6×10}` (rows 0-2 left foot, 3-5 right
foot; columns = all leg joints, left then right; from the two-leg fixed-base auxiliary model,
`LegSwingDynamicsProvider.cpp:436-562, 691-743`).
```
f = -[F_L; F_R] (6), m = -[M_L; M_R] (6)
tau_all = Jv_W^T f + Jw_W^T m          (10-vector)
per leg: mode = JointTorque; tauFeedForward = tau_all[offset:offset+5]; force/moment FF = 0
```
No ramp, no yaw hold, no PD, no gravity compensation.

### 9.3 Final torque
`RobotRunner::composeCommand`: `tau = 0(nu)`; leg torques written into actuator slots; arm PD torques
(`kp 100, kd 5` per joint, `qDes` = init spline end, `qdDes = 0`) written; then `applyRobotCommand`
clamps each actuator to its `ctrlrange` **only if `actuator_ctrllimited[i]`**; otherwise tau is written
unclamped (`SR:803-818`).

---

## 10. Standing mode (used at start-up settle and after braking)

* Contact: `gaitScheduler.c = true` for both feet at all horizon steps → all 25 steps double stance,
  constant contact signature; `activeContactForSide` = true; ramp alpha = 1.
* Weights: `mpc.standing.state_weight_diag = [50000, 80000, 500, 50000, 90000, 50000, 10,10,10, 10,10,5, 1]`,
  `input_weight_diag = 1e-4 × 12` (`CC:255-258`, yaml lines 36-38).
* Reference: seed = target pose. Position xy = mean of the two foot end-effector xy positions (every tick),
  z = `nominalHeight_W` (= initial-pose seed z + yaw-frame COM offset z) + `body_height_offset`; roll/pitch =
  `eulerSeed (0,0) + filtered offsets`; yaw = open-loop integral of `psi_dot` from the initial-pose yaw (0).
  Velocities in `X_ref` = `Rz(psi)*[x_dot,y_dot,0]` and `psi_dot`, and `p_ref` advances by
  `Rz(psi_ref)*[x_dot,y_dot,0]*dtMpc` over the horizon (`ReferenceTrajectory.cpp:20-23, 51-60`). They are zero
  in standing only if the command is zero: the keyboard ignores velocity keys while standing, but the headless
  schedule (and any port-side injector) writes `x_dot` regardless of mode (sec 3.1). With the default
  `X_DOT_START_TIME` (1.0 s) `x_dot` is already nonzero during `StandingSettle`, so the standing MPC reference
  carries that velocity. The port must either replicate this or start `x_dot` at ≥ walking start (≈3.0 s),
  and state which.
  `r_left/right = footPos_W(z→-0.005) - p_ref`.
* Leg command: combined-Jacobian `JointTorque` (sec 9.2).
* Start-up: `post_init_standing_settle_time = 1.0` → FSM starts in `StandingSettle` (mode Standing) for 1.0 s
  after the PD init completes; also sets keyboard enable time (1.0 s sim time) and the pre-init
  `legDynamicsRequest` (standing Jacobians).
* Braking→Standing: same standing behaviour; the body target is not re-seeded, so the position reference
  is the current mean foot xy and the *original* nominal height.

---

## 11. Config key table — `config/mit_humanoid/my_controller.yaml`
(default from `CCH`; "consumer" = where the value is read at run time)

| key | MIT value | default | loaded at | consumer |
|---|---|---|---|---|
| `requested_locomotion_mode` (alias key `locomotion_mode`, used only if the first is absent) | walking | Walking (both keys missing) | `CC:212-216` | FSM ctor `MC:425`; pre-init request `MC:495`; `getL()/getK()` no-arg (unused by MC) |
| `timing.cycle` | 0.5 | 1.0 | `CC:219` | `cycleTime()` → gait phase `GS:109`, clock sync `HC:23-24`, remaining swing `MC:60`, planner `SwingFootPlanner.cpp:248` |
| `timing.swing` | 0.17 | 0.4 | `CC:220` | clamp of remaining swing time `MC:60`, planner `:251` |
| `timing.stance` | 0.33 | 0.6 | `CC:221` | stance fraction `GS:118`; preview time `MC:902`, planner `:177` |
| `timing.horizon` | 0.5 | 0.5 | `CC:222` | `dtMpc()` `CC:646` |
| `timing.horizon_steps` | 25 | 15 | `CC:223` | horizon loops `GS:142`, `ReferenceTrajectory.cpp:28`, MPC sizes, `getL/getK` |
| `model.xml_path` | models/mit_humanoid/scene.xml | "" | `CC:226`, `RobotConfig.cpp:64` | `SR:235,257` |
| `model.auxiliary_xml_path` | models/mit_humanoid/mit_humanoid.xml | = `model.xml_path` (runtime loader `RobotConfig.cpp:65-69`; the `ControllerConfig` copy defaults to "" but is only echoed by the debug logger) | `RobotConfig.cpp:65-69`, `CC:227` | leg-dynamics auxiliary models `LegSwingDynamicsProvider.cpp:745-748` |
| `model.foot_end_effector_source` | site | **required** (both loaders throw if missing/invalid; values `site` \| `collision_geom_center`) | `CC:138-142, 228-229`, `RobotConfig.cpp:28-32` | foot position/velocity/rotation reading (`MujocoCheaterStateReader.cpp:226-323`), Jacobians |
| `model.gravity` | -9.81 | -9.81 | `CC:230` | `x0[12]` `MC:673`; fallback `MC:1038` |
| `mpc.friction_coefficient` | 1.0 | 0.1 | `CC:233` | constraint template `GS:83-95` / ConvexMPC C rows |
| `mpc.foot_half_length` | 0.065 | 0.065 | `CC:234` | `GS:92-93` (pitch-moment bound) |
| `mpc.foot_half_width` | 0.01 | 0.01 | `CC:235` | `GS:90-91` (roll-moment bound) |
| `mpc.torsional_friction_scale` | 0.0657 | 0.0657 | `CC:236` | `GS:94-95` |
| `mpc.normal_force_max` | 1500.0 | 200 | `CC:237` | `GS:170,175` |
| `mpc.normal_force_min` | 5.0 | 10 | `CC:238` | `GS:171,176` (scaled by ramp alpha in override) |
| `mpc.use_shifted_warm_start` | true | true | `CC:239` | `ConvexMPC.cpp:923, 1099` |
| `mpc.iterations_between_solve` | 7 | 10 | `CC:240` | `MC:438, 919` |
| `mpc.contact_wrench_model` (top) | (absent) | full_wrench | `CC:241-242` | default for the two mode keys |
| `mpc.walking.contact_wrench_model` | full_wrench | inherits | `CC:243-246` | `contactWrenchModel(mode)` → `MC:948`, `GS:144-145` |
| `mpc.standing.contact_wrench_model` | full_wrench | inherits | `CC:247-250` | same |
| `mpc.walking` / `mpc.standing` (the maps themselves) | present | **required** (`readModeWeights` throws if either is missing or not a map, `CC:76-94`); entries inside are optional | `CC:251-258` | — |
| `mpc.walking.state_weight_diag` | [50000, 9000, 500, 200000, 1009300, 110000, 10,10,10, 100,70,10, 1] | I (if present, length must be 13 for state / 12 for input, else throws) | `CC:251-254` | `getL(Walking)` `CC:668-685`; `ConvexMPC::stateWeightForMode` `:808-811` |
| `mpc.walking.input_weight_diag` | 1e-3 ×12 | I | same | `getK(Walking)`; `ConvexMPC.cpp:813-816` |
| `mpc.standing.state_weight_diag` | [50000, 80000, 500, 50000, 90000, 50000, 10,10,10, 10,10,5, 1] | I | `CC:255-258` | `getL(Standing)` |
| `mpc.standing.input_weight_diag` | 1e-4 ×12 | I | same | `getK(Standing)` |
| `swing.natural_frequency` | [151,151,110] | [10,10,10] | `CC:261-264` | `_swingNaturalFrequency` → `computeSwingCartesianKp` `MC:1353-1356` |
| `swing.kd_diag` | [25,25,25] | [15,15,18] | `CC:265-267` | `_swingKd` → `kdCartesian` `MC:1360` |
| `swing.height` | 0.06 | 0.06 | `CC:268` | `traj.reset(..., height, ...)` `MC:857` |
| `swing.min_remaining_time` | 0.001 | 1e-3 | `CC:269` | `MC:801, 828-829, 849-850` |
| `swing.body_velocity_half_stance_offset` | 0.37 | 0 | `CC:270-272` | preview time `MC:35-45, 902`; planner `SwingFootPlanner.cpp:158-169` |
| `swing.mid_speed_body_velocity_half_stance_offset` | 0.28 | 0 | `CC:273-275` | same (speed > 0.60) |
| `swing.body_velocity_half_stance_offset_switch_speed` | 0.60 | inf | `CC:276-278` | same |
| `swing.high_speed_body_velocity_half_stance_offset` | 0.26 | 0 | `CC:279-281` | same (speed > 0.65) |
| `swing.high_speed_body_velocity_half_stance_offset_switch_speed` | 0.65 | inf | `CC:282-284` | same |
| `swing.swing_foot_yaw_lead_scale` (alias `touchdown_yaw_lead_scale`) | 1.0 | 1.0 | `CC:285-292` | `MC:899-903` |
| `swing.enable_stance_foot_yaw_hold` | true | false | `CC:293-295` | `MC:1338` |
| `swing.turn_tangential_lead_scale` | 1.0 | — | **not read by any loader** | none (dead key) |
| `swing.nominal_foot_offsets_B` | [[0,0.075999602,0],[0,-0.075999602,0]] | (derived from feet) | `CC:296-299` | planner `SwingFootPlanner.cpp:128-137` |
| `swing.stop_braking_offset_B` | (commented out) | none | `CC:300-304` | planner `:211-212` |
| `swing.stop_capture_point_gain` | 0.2 | 1.0 | `CC:305` | planner `:217` |
| `swing.stop_capture_point_max_offset` | 0.08 | 0.20 | `CC:306-308` | planner `:220-224` |
| `swing.stop_velocity_deadband` | 0.02 | 0.02 | `CC:309` | planner `:183-184, 302-304` |
| `swing.stop_braking_latch_clear_ticks` | 5 | 5 | `CC:310-312` | planner `:199` |
| `swing.roll_kp/roll_kd` | 0 / 0 | 0 | `CC:313-314` | swing attitude `MC:1365-1366` (disabled: 5-DOF leg) |
| `swing.pitch_kp/pitch_kd` | 300 / 18 | 0 | `CC:315-316` | `MC:1367-1368` |
| `swing.yaw_kp/yaw_kd` | 305 / 18 | 0 | `CC:317-318` | `MC:1369-1370` |
| `swing.stance_yaw_kp/stance_yaw_kd` | 20 / 4 | 0 | `CC:319-320` | `MC:1345-1346` |
| `user_command_filter.x_dot_tau` | 0.92 | 0 | `CC:323` | `MC:634` |
| `user_command_filter.y_dot_tau` | 0.8 | 0 | `CC:324` | `MC:637` |
| `user_command_filter.psi_dot_tau` | 0.70 | 0 | `CC:325` | `MC:640` |
| `user_command_filter.standing_roll_offset_tau` | 0.70 | 0 | `CC:326-328` | `MC:646` |
| `user_command_filter.standing_pitch_offset_tau` | 0.70 | 0 | `CC:329-331` | `MC:650` |
| `user_command_filter.x_dot_max` | 0.7 | inf | `CC:332` | `clampUserCommand` `CC:652`; keyboard limit `SR:231` |
| `user_command_filter.y_dot_max` | 0.5 | inf | `CC:333` | `CC:653`; `SR:232` |
| `user_command_filter.psi_dot_max` | 2.0 | inf | `CC:334` | `CC:654`; `SR:233` |
| `reference_trajectory.yaw_integration_mode` | always | single_support | `CC:337-339` | `MC:702-703` (body target yaw), `ReferenceTrajectory.cpp:29-30, 43-48, 76-79` |
| `contact_manager.contact_force_on_threshold` | 36.0 | 20 | `CC:342-344` | `CM:98, 124, 145, 170` |
| `contact_manager.contact_force_off_threshold` | 0.5 | 5 | `CC:345-347` | `CM:117` |
| `contact_manager.contact_on_confirm_ticks` | 1 | 2 | `CC:348-350` | `CM:133` |
| `contact_manager.contact_off_confirm_ticks` | 4 | 2 | `CC:351-353` | `CM:138` |
| `contact_manager.contact_ramp_duration` | 0.01 | 0.08 | `CC:354-356` | `CM:195, 239, 261, 347-349` |
| `contact_manager.contact_lock_steps` | 1 | 3 | `CC:357-359` | `CM:418` |
| `contact_manager.late_contact_timeout` | 0.20 | 0.20 | `CC:360-362` | `CM:310-311` (flag only) |
| `contact_manager.ground_search_velocity` | 0.40 | 0.08 | `CC:363-365` | `CM:303` |
| `contact_manager.ground_search_max_depth` | 0.15 | 0.04 | `CC:366-368` | `CM:307` |
| `contact_manager.ground_search_tracking_time` | 0.08 | 0.08 | `CC:369-371` | `MC:828` |
| `contact_manager.stance_contact_loss_foot_height` | 0.060 | 0.025 | `CC:372-374` | `CM:283` |
| `contact_manager.enable_early_contact_handling` | true | true | `CC:375-377` | `CM:273` |
| `contact_manager.enable_late_contact_handling` | false | true | `CC:378-380` | `CM:285` |
| `logging.standing_mpc_debug_trigger_times` | [] | [] | `CC:383-387` | `MC:1108-1119` (debug log trigger) |
| `startup.post_init_standing_settle_time` (alias `standing_settle_time`) | 1.0 | 2.0 | `CC:389-397` | FSM `MC:426`; keyboard enable `SR:271`; pre-init request `MC:496` |
| `locomotion_transition.braking_settle_speed_threshold` | 0.01 | 0.05 | `CC:400-402` | `FSM:198-199` |
| `locomotion_transition.braking_settle_yaw_rate_threshold` | 0.01 | 0.15 | `CC:403-405` | `FSM:200` |
| `locomotion_transition.braking_settle_average_window` | 1.0 | 0.5 | `CC:406-408` | `FSM:166-173` |
| `locomotion_transition.braking_settle_hold_ticks` | 3 | 3 | `CC:409-411` | `FSM:252` |
| `locomotion_transition.braking_timeout_seconds` | 3.0 | 2.0 | `CC:412-414` | `FSM:261` |
| `locomotion_transition.braking_touchdown_count` | 5 | 2 | `CC:415-417` | `FSM:260` |
| `joint_tracking.leg.kp/kd` | [70,50,100,100,200] / [15,10,10,10,30] | (required) | `robot/src/JointTrackingConfig.cpp:74-76` | leg PD init `RR:129-132` |
| `joint_tracking.arm.kp/kd` | [100]*4 / [5]*4 | (required) | same | arm PD every tick `RR:115-117` |
| `initial_pose.base_position_W` | [0,0,0.679472] | none | `CC:422-434` | body target seed `MC:684-689` |
| `initial_pose.base_rpy_W` | [0,0,0] | none | same | `MC:689` |
| `initial_pose.leg_initialization_time` | 2.0 | 2.0 | `CC:435-437`, `InitialPoseConfig.cpp:63-64` | `RR:66-67` |
| `initial_pose.arm_initialization_time` | 1.0 | 1.0 | same | `RR:68-69` |
| `initial_pose.leg_joint_offsets` | [0,0,-0.735,1.2,-0.70] | [0,0,-0.65,0.80,-0.35] | `CC:420`, `InitialPoseConfig.cpp:61` | `LegPosInitializer.cpp:89-92` (added to `qpos0` per leg) |
| `initial_pose.arm_joint_offsets` | [0,0,0,-1.65] | [0,0,0,-0.65] | same | `ArmPosInitializer.cpp:89-92` |
| `gait_swing_hold_test.*` | scene_test.xml / copied_state | — | `CC:479-488` | test tooling only |
| `config/simulation.yaml: physics_timestep_sec / physics_integrator / viewer_sync_hz` | 0.002 / implicitfast / 60 | required | `SimulationConfig.cpp:45-88` | `SR:268` (`configureSimulationModel`), `SR:326-328` |

---

## 12. Telemetry / debug hooks (skipped; hidden state check)
`StandingMpcDebugLogger` snapshots (`MC:1148-1170`): state estimate, params, leg controller, arm controller,
desired foot positions, x0, reference output, formulation output, full horizon wrench, iteration, mode,
FSM state, `t0`, filtered command, contact-manager leg states, per-leg `touchdownYaw_W`. It reveals no
state beyond what is listed in this document. `_standingMpcDebugLog*` members affect nothing else.
`collectDebugVisualization` (`MC:1190-1252`) only writes mocap markers.

---

## 13. Docs vs code discrepancies (`docs/gait_scheduler_and_contact_management.md`)
| doc statement | code |
|---|---|
| §2 tip: "checked-in default stance/cycle = 0.6, double support 0.2 Tc" | MIT yaml: stance/cycle = 0.66, double support per cycle = 2·0.33−0.5 = 0.16 s (two 0.08 s phases). |
| §5.2 hysteresis "n_k ≥ F_on for N_on ticks" | ON check only counts while `estimated==false`, OFF only while `estimated==true`; counters reset when the force is in the dead band (`CM:116-130`). |
| §5.4 late contact "stops when contact is re-established or the search limit is reached" | search never exits on the depth limit or the timeout; only `estimated==1` ends it (`CM:296-325`). Disabled for MIT anyway. |
| §5.6 yaw-hold formula written as `alpha*(kp e - kd rate)` | code multiplies the moment by alpha once in `MC:1347` — consistent; but docs omit that it is skipped when `alpha == 0` (`MC:1338`). |
| §6.1 piecewise "search / early / otherwise" | identical, plus the (dead) `liftoffHold` branch (`CM:399`). |
| §7 "Ask ContactManager for the managed foot positions" during the MPC update | managed positions are computed every tick in `runController` (`MC:1413-1416`), not inside `maybeUpdateMpc`. |
| §9 line numbers (e.g. `GaitScheduler::p` at L50, `maybeUpdateMpc` at L937) | stale; actual: `GS:98`, `MC:911`. |
| §4.2 "`D` … cold-starts OSQP" | signature is built from `constraintSteps` stance flags (`ConvexMPC.cpp:1071-1084`), equivalent. |
| yaml comment on `mpc.standing`: "Keep the same baseline for now" | values differ from walking (see table). |

---

## 14. MIT-specific facts and differences for the Python port
* 5-DOF legs `[hip_yaw, hip_abad, hip_pitch, knee, ankle]`, no ankle roll → `swing.roll_kp = roll_kd = 0`
  (yaml comment line 63); swing attitude control acts on pitch and yaw only; the line-contact foot cannot
  produce a roll moment about the foot x-axis, yet `mpc.foot_half_width = 0.01` allows `|Mx| ≤ 0.01·Fz`
  and both modes use `full_wrench` (so the MPC may command a small Mx that the foot cannot deliver).
* Reference expects MJCF names (`MitHumanoidSpec.cpp`): base body `torso`, foot bodies `left/right_foot_link`,
  foot sites `left/right_foot_contact_site`, joints `<side>_hip_yaw_joint` …, actuators `<side>_hip_yaw` …,
  arms `<side>_forearm_link` with joints `<side>_shoulder_pitch_joint` … `<side>_elbow_joint`, no fixed joints.
  The converted MJCF (`mit_humanoid_mjcf/mit_humanoid.xml`) uses `base`, `left_foot`/`right_foot`,
  `a06_left_hip_yaw` …, `left_hand`, and defines **no sites**; the Python port must build its own name map
  and add a foot contact site (or use `collision_geom_center`) — the reference's own MIT MJCF is absent.
* `physics_timestep_sec = 0.002` overrides the converted MJCF `<option timestep="0.001">`.
* Initial base height for the target seed: `0.679472` (yaml) vs converted MJCF `base pos z = 0.7483`.
* `bodyMass` used by the MPC = torso + arms (+ any other non-leg massive body in the scene) only (legs
  excluded); with a 24.89 kg total the MPC weight is well below the robot weight — replicated as-is by the
  reference (see O3).
* Contact normal force = sum of |normal| over all MuJoCo contacts touching the foot body (line-contact
  cylinder foot → typically 2 contact points).
* Headless runs only schedule `x_dot`; lateral 0.3 m/s and turn 1.3 rad/s tests need an equivalent injector.

## 15. Open questions
* **O1** Horizon anchoring — **answered (confirmed, replicate for parity).** Horizon step k uses
  `c(side, t0 + 0.02k)`, `t0` = current cycle origin (not now) (`HC:18-34`, `GS:98-120, 152-177`). Resulting
  MIT pattern: Left stance `k ∈ {0..3, 13..24}`, Right stance `k ∈ {0..16}`, plus Left k = 4 as stance
  whenever floating-point rounding puts `p` just below 0.66 (sec 5; frequent after t0 ≈ 16 s). Step 0 is then
  overwritten by the measured `activeContact` with min-scale alpha (`contact_lock_steps = 1`, `CM:416-445`).
  Related facts: the shifted warm start shifts by one step (`ConvexMPC.cpp:1094-1108`), which does not match a
  horizon that only moves at cycle boundaries; the stance signature (→ cold start) changes essentially only
  when the step-0 active set changes or at a cycle wrap; `ReferenceTrajectory` also uses `tk(k)`, but only for
  yaw-integration gating, which has no effect in mode `always` (`ReferenceTrajectory.cpp:39-49`).
  A time-sliding horizon can be evaluated later as a variant.
* **O2** FSM updated twice per tick (prepareController + runController) — replicate (affects only braking
  settle tick counting) or call once?
* **O3** — **answered (confirmed).** MPC mass = upper body (torso + arms + any other non-leg massive bodies
  in the scene), recomputed every tick (`setupRobotParams.cpp:414-431, 488-490`), with `Fz ∈ [minScale·5, 1500]`
  N per stance foot. There is **no standing gravity feed-forward**: `bodyMass·|g|/nActive` (upper-body mass;
  mg/2 per foot in standing) is used only as the fallback wrench when the MPC throws (sec 8.4). The port's scene
  must not add massive non-robot bodies, or it must replicate the reference's marker bodies (the MIT
  `scene.xml` is absent from the repo; the G1 scene has four mocap marker bodies with default-density geoms,
  ≈0.2 kg in total, counted in `bodyMass` at the marker positions). Leg link weight is not compensated in
  stance (`StanceWrench` has no bias term, `LegController.cpp:309-315`). The demo numbers may depend on this.
* **O4** `_bodyTarget.euler_W[2]` (planner yaw) is an open-loop integral from yaw 0 and never re-seeded; in
  long turn tests it drifts from the true yaw. Replicate or anchor to `yaw_W_unwrapped`?
* **O5** — **answered (confirmed, keep for parity).** The MPC lever arm for a stance foot is
  `desired − p_ref`, where `desired` = the planner's cached touchdown target latched at the start of that
  foot's last swing (z = −0.005). Exceptions: (a) the first stance after a planner reset (walking start) uses
  the measured `footPos_W` at that tick, measured z included; (b) only during early contact (not yet
  scheduled) is the frozen measured touchdown point used; at the scheduled touchdown it reverts to the planned
  target; (c) stance/swing for the planner comes from gait `c()`, not `activeContact`. The measured
  stance-foot position is never used otherwise (sec 6.6).
* **O6** — **answered (confirmed, replicate for parity).** On the first tick of any activation (scheduled
  touchdown or early contact) alpha = 0 and the leg is in `StanceWrench` with `F = M = 0` and no yaw hold, so
  `tau_leg = 0` exactly for 2 ms (no bias compensation, no PD, no tauFeedForward;
  `CM:340-349`, `MC:1315-1349`, `LegController.cpp:294-317`). This does not happen on reset/transition ticks,
  where alpha = 1 (`CM:195`). See sec 6.5 "What the ramp scales".
* **O7** — **answered.** `M` from the aux model is identical to the full model's leg-dof block of `qM` at the
  same (torso pose, leg q) because it depends only on the leg subtree; only `bias` and `JvDot` differ (aux
  model: zero base velocity). Python equivalent for parity: `Jv/Jw` = leg-dof columns of `mj_jacSite`,
  `M` = leg-dof block of `qM`, `bias` = `qfrc_bias[leg dofs]`, `JvDot` from `mj_jacDot`, all evaluated on a
  separate `MjData` with `qpos = [torso pose read from xpos/xquat (pre-integration), current leg qpos]`,
  base `qvel = 0`, leg `qvel = qd` (then `mj_forward`) — not the full model's own `qfrc_bias`. Foot angular
  rates for the attitude and yaw hold are `Jw_W*qd_leg` (torso rate excluded). `p_W` and `v_W` for the
  swing PD are absolute world values from the full model (sec 9.1).
* **O8** The converted MJCF lacks contact sites and uses different names; foot-site placement (the point
  driven to z = −0.005 at touchdown) must be chosen to match the reference's `*_foot_contact_site`.

## Verification log (9/25)
1. §3.2 dt = 0 on the first tick and on every resetSwingState transition tick (incl. walking start) — **applied** (§3.2 new paragraph, §2 step 4); verified `MC:28-33, 448, 470, 552, 1390-1391`, `SR:416, 427`.
2. Headless x_dot is written regardless of FSM mode, so standing settle carries a nonzero reference velocity — **applied** (§3.1, §10); verified `SR:466-473, 516-524`, `ReferenceTrajectory.cpp:20-23, 51-60`. Also added the ramp formula and the rule that the profile's first value applies before its time.
3. First runController at t = 1.998 s (call #1000), walking at ≈3.000 s (501 ticks), arms done at 0.998 s — **applied** (§1.3, §1.4); verified `LegPosInitializer.cpp:9-13, 36, 57`, `RR:120-141`, `SR:337-341`, re-checked numerically (float(0.002) = 0.0020000000949949).
4. No mj_forward before reading state; poses/velocities/contacts pre-integration, q/qd/time post-integration — **applied** (§1.2 "State timing"); verified no mj_forward/mj_kinematics in the read path (grep of `sim/src`), `MujocoCheaterStateReader.cpp:27, 114, 128, 142, 157, 338-351`.
5. O5 exceptions (first stance after planner reset uses measured footPos_W; early-contact substitution reverts at scheduled touchdown; planner uses c()) — **applied** (§6.6, O5); verified `SwingFootPlanner.cpp:29-44, 149-151, 334-343`, `CM:321-324, 394-414`.
6. §7 "moving target" was wrong: planner target is latched at swing start and changes only on stop-recenter activation / turn-stop frame / CM substitution — **applied** (§7); verified `SwingFootPlanner.cpp:345-359`.
7. Tick-order consequences (solve before leg command, _iteration never reset, newly active foot gets zero held wrench until next solve, no redistribution) — **applied** (§2.1); verified `MC:918-921, 1010, 1054, 1313-1331, 1421-1428`; `applyLocomotionOutput` (`MC:555-577`) never touches `_iteration`.
8. Complete ramp answer (what alpha scales, MPC Fz_min of step 0 only, reset alpha = 1, no liftoff ramp) — **applied** (§6.5 "What the ramp scales"); verified `CM:185-198, 340-353, 430-433`, `GS:165-176`, `MC:748-753, 1317-1347`.
9. O6: tau_leg exactly 0 on the first activation tick; not on reset ticks — **applied** (§6.5, O6); verified `LegController.cpp:294-317` (bias/tauFF commented out).
10. O7: aux M identical to the full qM leg block; only bias/JvDot differ (zero base velocity); foot rates exclude torso rate; p_W/v_W from full model — **applied with amendment** (§9.1, O7): for exact parity the Python evaluation must use the pre-integration torso pose with the current leg q on a separate MjData (base qvel = 0); reading the full model's own post-mj_step data gives pre-integration leg q (see item 4). Verified `LegSwingDynamicsProvider.cpp:336-356, 611-687`, `RR:222-235`.
11. O3: no standing feed-forward (fallback only); bodyMass includes other massive scene bodies; leg weight uncompensated in stance — **applied with amendment** (§1.2, §14, O3): marker masses corrected to ≈0.04–0.07 kg each (touchdown markers have r = 0.025 spheres; ≈0.2 kg total in the G1 scene), and noted that the MIT scene.xml is absent, so whether it has markers is unknown. Verified `setupRobotParams.cpp:414-431`, `models/unitree_robots/g1/scene_23dof.xml:21-39`, `MC:1016-1051`.
12. O1: cycle-aligned pattern L {0..3, 13..24}, R {0..16} — **applied with correction** (§5, O1): the finding's "checked numerically incl. fp boundaries" is wrong at Left k = 4, where `((t0+0.08)-t0)/0.5+0.5` rounds below 0.66 for 223 of the first 400 cycles (t0 ≥ ≈16 s), making k = 4 stance. Verified `HC:18-34`, `GS:98-120, 152-177`, `ConvexMPC.cpp:880-906, 1070-1108`, numerical check.
13. §9.1 e_pitch / rate_pitch undefined — **applied** (§9.1 attitude block); verified `SwingAttitudeControl.h:18-80`.
14. §11: foot_end_effector_source required, auxiliary_xml_path default = xml_path, locomotion_mode alias, mpc.walking/standing maps required — **applied** (§11 rows); verified `CC:58-94, 117-142, 212-229, 251-258`, `RobotConfig.cpp:28-32, 57-72`.
15. transitionTo is a no-op on the same state and clears the deque on every transition; ctrlrange clamp only if ctrllimited; lazy init differs from reset — **applied** (§4.2, §9.3, §6.5); verified `FSM:107-127`, `SR:803-818`, `CM:231-243`.

## Confirmed by verifier
* x0 ordering [roll,pitch,yaw_unwrapped, COM_W, torsoAngVel_W (world), COM vel_W, g=-9.81] (MC:656-675); COM = torsoPos + Rz(yaw)*bodyComLocation, COM vel = v + ω×offset with full ω
* Stance wrench vector layout [F_L, F_R, M_L, M_R], ground-on-body, leg command applies -alpha*segment (MC:1319-1331); standing uses -[F;F], -[M;M] via combined 6x10 J^T (MC:1274-1283)
* C_bound indices: left Fz max/min rows 4/5, right 16/17; bound = -minScale*Fmin => Fz >= minScale*Fmin; non-stance feet get zero bounds (GS:169-177)
* Gait: phi_L = 0.5, phi_R = 0; p = fmod((t-t0)/cycle + phi, 1); c = 0 <= p < 0.66; Standing short-circuits p = 0, c = true; left swings first [0.08, 0.25) after walking start
* HorizonClock sync `while t-t0 >= cycle: t0 += cycle`, tk = t0 + k*dtMpc, dtMpc = 0.5/25 = 0.02; reset on transitions except into Braking
* MPC schedule: solve when _iteration==0 or _iteration-_lastMpcIteration >= 7; _lastMpcIteration set after success or failure; _stanceWrenchWorld = solution[0:12]
* runController step order as listed in §2 (MC:1374-1429); FSM updated twice per tick (prepareController + runController) except on the first controller tick; transitions happen in the first call, before legSwingDynamicsProvider.update
* Command filter: clamp, zero-motion override, first-call pass-through, instant stop when all three raw velocities == 0, low-pass with taus 0.92/0.8/0.70, height offset unfiltered, roll/pitch offsets low-passed with 0.70, final clamp (MC:610-654, CC:649-656)
* Body target: seeded once from initial_pose (0,0,0.679472)+Rz(0)*bodyComLocation, never re-seeded; Standing xy = mean foot xy; Walking xy = x0 COM xy; yaw advanced with psi_dot*dt (mode always); roll/pitch = seed + offsets; z = nominalHeight + height offset
* MPC reference seed: Standing uses the full target pose (euler_W, nominalPosition, z = nominalHeight); Walking uses roll/pitch/z from the target and x, y, yaw, velocities from x0 (MC:966-978); ReferenceTrajectory rollout formulas and constant foot positions over the horizon
* FSM: initial state logic, modeForState, requestToggle only in Interactive (MIT walking => toggles no-op, braking never happens), braking settle averaging/hold ticks/touchdown/timeout logic, output flags (resetGaitClock/resetSwingState excluded for Braking, zeroMotionCommand)
* ContactManager hysteresis (ON after 1 tick >= 36 N, OFF after 4 ticks <= 0.5 N, dead band holds and resets the opposite counter), reset(), walking update logic (early/late/frozen/commanded/active/ramp), managedFootPositions substitution only if search|early|hold, buildHorizonOverride with lock steps = 1
* Late contact disabled for MIT; liftoffHold never set (dead branches)
* Swing trajectory update logic incl. search branch, remainingSwingTime = clamp(cycle*(1-p), 0, swing), minRemaining 1 ms, yaw latch with psi offset (100 deg/rad, clamp 20 deg, +Left for psi_dot>0, -Right for psi_dot<0)
* swingFootYawTargetWorld = yaw_unwrapped + 1.0*psi_dot*(0.5+offset)*0.33 with offsets 0.26 (>0.65), 0.28 (>0.60), 0.37
* mpcFootYaw: measured foot yaw lifted near touchdownYaw if active, else touchdownYaw; NoRollMoment path skipped (full_wrench both modes); override only in walking
* Failure fallback: zero wrench, then Fz = bodyMass*|g|/nActive on index 2 (L) / 5 (R)
* Swing Kp = diag(wn^2)*diag(Λ), Λ = (Jv M^-1 Jv^T + 1e-9 I)^-1; Kd = diag(25) (not Λ-scaled); swing tau = Jv^T(F + Kp e + Kd ė) + Jv^T Λ(aDes - JvDot qd) + bias + tauFF; stance tau = Jv^T F + Jw^T M with no bias/PD
* Stance yaw-hold formula (kp 20, kd 4, horizontal projection of foot x-axis, yawRate = e_z·Jw qd), added as alpha*Mz and skipped when alpha == 0
* Reduced-body mass properties every tick: legs excluded, COM offset and parallel-axis inertia expressed in the yaw frame using wrapped yaw from xmat
* Keyboard step sizes/limits (0.05, 0.1 rad/s, 0.01 m, 2 deg; limits from yaml 0.7/0.5/2.0, height 0.8), 't'/'T' clears motion and increments toggle, space clears motion and keeps counters, 'L' debug counter, sanitize 1e-12
* Headless schedule env vars and defaults (start = 1.0 s, toggle = 1.2 s, x_dot only, profile parsing)
* SimulationConfig: physics_timestep 0.002 overrides the MJCF timestep, implicitfast; all three simulation.yaml keys required
* Leg init spline to qpos0 + [0,0,-0.735,1.2,-0.70] per leg over 2.0 s; arms qpos0 + [0,0,0,-1.65] over 1.0 s; arm PD (100/5) every tick; leg PD gains only during init
* All MIT yaml numeric values in the §11 table (weights, contact-manager params, swing params, filter taus/limits, transition params) match the yaml literally; turn_tangential_lead_scale is read by no loader
* Contact normal force = sum of |mj_contactForce[0]| over contacts touching the foot body or its collision geoms; hasContactForce always true
