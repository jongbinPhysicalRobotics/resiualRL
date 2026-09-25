# 03 — Reference trajectory, touchdown planning, swing trajectory, command filter, gait phase (MIT Humanoid)

Spec of the reference C++ (`ispaik06/convex-mpc-biped`, local copy `C:\Users\백종빈\Desktop\4-2\residual RL\reference`) for a Python port.
All paths below are relative to that `reference` folder. `file:line` = the exact line in the local copy.
**The code wins over `docs/`**; every place where the docs differ is listed in §9.

Nothing in this document requires reading the C++; every constant, branch and formula is reproduced here.

---

## 0. Conventions, glossary, per-tick call order

### 0.1 Frames

| Symbol | Meaning |
|---|---|
| `W` | MuJoCo world frame, z up. |
| `B` (yaw-aligned / "body-yaw") | Frame obtained by rotating `W` by yaw only: `R_WB = Rz(psi)`. Used for commands (`x_dot`,`y_dot`), `nominal_foot_offsets_B`, `bodyComLocation`, `bodyInertia`. Never contains roll/pitch. |
| `T` | Torso body frame (full rotation). Used only inside the LocomotionFSM speed check (`My_Controller.cpp:593-601`). |

```
Rz(psi) = [[cos psi, -sin psi, 0],
           [sin psi,  cos psi, 0],
           [0,        0,       1]]          # common/include/Utilities/MatrixUtils.h:11-21
wrapToPi(a)                 = atan2(sin a, cos a)                  # AngleUtils.h:6-8
unwrapAngle(now, prevW, prevU) = prevU + wrapToPi(now - prevW)      # AngleUtils.h:10-14
liftAngleNear(targetW, curU)   = curU + wrapToPi(targetW - curU)    # AngleUtils.h:16-18
```

Yaw everywhere in the planner/reference is the **unwrapped** yaw `yaw_W_unwrapped` produced by the state estimator (`common/src/Estimator/StateEstimator.cpp:27-56`):
```
yawWrappedNow = atan2(R_WT[1,0], R_WT[0,0])        # R_WT = torso quaternion -> rotation matrix
first tick (or time went backwards): yaw_W_unwrapped = yawWrappedNow, yawRate_W = 0
else: yaw_W_unwrapped = unwrapAngle(yawWrappedNow, lastWrapped, lastUnwrapped)
      yawRate_W = (yaw_W_unwrapped - lastUnwrapped) / dt
state.psi = yaw_W_unwrapped   (legacy alias)
```
The reference-yaw recurrence is **not wrapped** in the code (docs say wrapped — see §9).

### 0.2 Reduced-body state (13) and inputs (12)

```
x = [roll, pitch, yaw, px, py, pz, wx, wy, wz, vx, vy, vz, g]      # index 0..12
     0     1      2    3   4   5   6   7   8   9   10  11  12
u = [Fx_L, Fy_L, Fz_L, Fx_R, Fy_R, Fz_R, Mx_L, My_L, Mz_L, Mx_R, My_R, Mz_R]
```
- `px,py,pz` and `vx,vy,vz` are **world** positions/velocities of the *reduced-body COM* (see 0.3), `wx,wy,wz` = torso angular velocity in **world** (`My_Controller.cpp:661-674`), `g = model.gravity = -9.81` (`my_controller.yaml:14`).
- roll/pitch: `quaternionToRollPitch` (`My_Controller.cpp:63-75`), with `q = (q_w, q_x, q_y, q_z)` = the **normalized** torso quaternion (these `q_*` are quaternion components, not the angular velocities `wx,wy,wz` above):
  `roll = atan2(2(q_w q_x + q_y q_z), 1 − 2(q_x² + q_y²))`, `pitch = asin(clamp(2(q_w q_y − q_z q_x), −1, 1))`.
- `x0` (current MPC state) = `buildCurrentMpcState()` `My_Controller.cpp:656-675`:
  ```
  x0[0:2] = rollPitch(torsoQuat_W)
  x0[2]   = yaw_W_unwrapped
  x0[3:6] = reducedBodyComWorld          (0.3)
  x0[6:9] = torsoAngVel_W
  x0[9:12]= reducedBodyComVelocityWorld  (0.3)
  x0[12]  = gravity (-9.81)
  ```

### 0.3 The "body point": reduced-body COM

`RobotParams.bodyComLocation` (`common/include/Robot/RobotParams.h:79`) = vector torso-root → COM of (torso + arms, i.e. **everything not in a leg subtree**), expressed in the **yaw-aligned** frame B. In the sim it is **recomputed every control tick** from live MuJoCo data (`sim/src/SimulationRunner.cpp:409` → `sim/src/setupRobotParams.cpp:397-491`):
```
upperBodyCom_W  = Σ m_i xipos_i / Σ m_i        over bodies not under a leg root (left/right hip-yaw body)
bodyMass        = Σ m_i                          (upper body only, legs excluded!)
psi             = atan2(xmat_torso[1,0], xmat_torso[0,0])
bodyComLocation = Rz(psi)^T (upperBodyCom_W - torsoPos_W)
bodyInertia     = Rz(psi)^T [Σ (R_i I_i R_iᵀ + m_i (|d|² 1 - d dᵀ))] Rz(psi)      (d = com_i - upperBodyCom_W)
```
Helper functions used by the planner (`SwingFootPlanner.cpp:12-26`, identical copies at `My_Controller.cpp:315-332`):
```
reducedBodyOffsetWorld      = Rz(yaw_W_unwrapped) @ bodyComLocation
reducedBodyComWorld         = torsoPos_W + reducedBodyOffsetWorld
reducedBodyComVelocityWorld = torsoLinVel_W + cross(torsoAngVel_W, reducedBodyOffsetWorld)
```
`torsoPos_W`/`torsoQuat_W`/vel = MuJoCo body `"torso"` (`sim/src/models/MitHumanoidSpec.cpp:6`; in the project model `MPC/models/mit_humanoid/mit_humanoid.xml` this body is named `"base"`, leg roots `left_hip_yaw`/`right_hip_yaw`), *body frame origin* (xpos/xquat), not its COM.

`r_left`, `r_right` in the reference trajectory are foot position **minus the reference COM position `p_ref[k]`** (§4.4), i.e. lever arms about the reduced-body COM, later used as `I_k^{-1} skew(r) F` in the SRB B-matrix (`MPCFormulation.cpp:64-79`).

### 0.4 Legs, feet, sides

- `robotParams.legs[0]` = Left, `legs[1]` = Right (`MitHumanoidSpec.cpp:8,18`). Joint order per leg: `[hip_yaw, hip_abad, hip_pitch, knee, ankle]`.
- Foot end-effector position `footPos_W` = MuJoCo **site** `left_foot_contact_site` / `right_foot_contact_site` (`MitHumanoidSpec.cpp:10,20`, `model.foot_end_effector_source: site` `my_controller.yaml:13`); foot frame `R_WF` = that site's rotation; foot body for contact detection = `left_foot_link` / `right_foot_link`.
- Contact normal force `contactNormalForce` = Σ |contactForceLocal[0]| over all MuJoCo contacts touching the foot body/geoms (`sim/src/MujocoCheaterStateReader.cpp:63-89`); `hasContactForce = true` in sim (`:363`).
- Gait side enum: `Side::Left`, `Side::Right` (`RobotParams.h:23`).

### 0.5 Config values used here (MIT yaml, `config/mit_humanoid/my_controller.yaml`)

```yaml
timing:      cycle 0.5   swing 0.17   stance 0.33   horizon 0.5   horizon_steps 25      # :3-8
mpc:         iterations_between_solve 7                                                  # :24
swing:       height 0.06   min_remaining_time 0.001                                      # :43-44
             body_velocity_half_stance_offset 0.37                                        # :45
             mid_speed_body_velocity_half_stance_offset 0.28                              # :46
             body_velocity_half_stance_offset_switch_speed 0.60                           # :47
             high_speed_body_velocity_half_stance_offset 0.26                             # :48
             high_speed_body_velocity_half_stance_offset_switch_speed 0.65                # :49
             swing_foot_yaw_lead_scale 1.0   enable_stance_foot_yaw_hold true             # :50-51
             turn_tangential_lead_scale 1.0        # NOT READ BY THE CODE (see §9)        # :52
             nominal_foot_offsets_B [[0, 0.075999602, 0], [0, -0.075999602, 0]]           # :53-55
             stop_braking_offset_B  (absent -> hasStopBrakingOffset=false)                # :58
             stop_capture_point_gain 0.2   stop_capture_point_max_offset 0.08             # :59-60
             stop_velocity_deadband 0.02   stop_braking_latch_clear_ticks 5               # :61-62
             roll_kp 0 roll_kd 0 pitch_kp 300 pitch_kd 18 yaw_kp 305 yaw_kd 18           # :64-69
             stance_yaw_kp 20 stance_yaw_kd 4                                             # :70-71
user_command_filter: x_dot_tau 0.92  y_dot_tau 0.8  psi_dot_tau 0.70                     # :74-76
             standing_roll_offset_tau 0.70  standing_pitch_offset_tau 0.70               # :77-78
             x_dot_max 0.7  y_dot_max 0.5  psi_dot_max 2.0                               # :79-81
reference_trajectory: yaw_integration_mode always                                        # :85
contact_manager: contact_force_on_threshold 36  off 0.5  on_confirm 1  off_confirm 4    # :88-91
             contact_ramp_duration 0.01  contact_lock_steps 1  late_contact_timeout 0.20 # :92-94
             ground_search_velocity 0.40  max_depth 0.15  tracking_time 0.08             # :95-97
             stance_contact_loss_foot_height 0.060                                        # :98
             enable_early_contact_handling true   enable_late_contact_handling false      # :99-100
startup:     post_init_standing_settle_time 1.0                                          # :107
initial_pose: base_position_W [0,0,0.679472]  base_rpy_W [0,0,0]                          # :133-134
             leg_initialization_time 2.0   arm_initialization_time 1.0                   # :135-136
             leg_joint_offsets [0,0,-0.735,1.2,-0.70]  arm_joint_offsets [0,0,0,-1.65]   # :137-138
requested_locomotion_mode: walking                                                       # :1
simulation.yaml: physics_timestep_sec 0.002                                              # config/simulation.yaml:1
```
Derived: `dtMpc = horizon/horizon_steps = 0.02 s` (`ControllerConfig.cpp:645-647`); control tick `dt_ctrl = 0.002 s` (controller runs **every** `mj_step`, `SimulationRunner.cpp:337-342`); MPC re-solved every 7 ticks = 0.014 s (`My_Controller.cpp:918-919`); config check `cycle == swing + stance` (`ControllerConfig.cpp:494`).

Config parser defaults that apply if a key is absent (`ControllerConfig.h:31-37, 60-107`; keys are read with `readScalarIfPresent`, so a missing key keeps the default). MIT sets every key used here, but a port that builds its own config must use these defaults:
- `timing`: cycle 1.0, swing 0.4, stance 0.6, horizon 0.5, horizon_steps 15.
- `swing`: height 0.06, min_remaining_time 1e-3, the three half-stance offsets 0.0, both switch speeds `+inf`, swing_foot_yaw_lead_scale 1.0 (alias key `touchdown_yaw_lead_scale`, read only when `swing_foot_yaw_lead_scale` is absent, `ControllerConfig.cpp:285-292`), enable_stance_foot_yaw_hold **false**, `hasStopBrakingOffset=false` (true only if `stop_braking_offset_B` is present), stop_capture_point_gain **1.0**, stop_capture_point_max_offset **0.20**, stop_velocity_deadband 0.02, stop_braking_latch_clear_ticks 5, all roll/pitch/yaw/stance-yaw gains 0.
- `user_command_filter`: all five taus 0.0 (⇒ `alpha = 1`, i.e. no filtering), x/y/psi_dot_max `+inf` (no clamp).
- `reference_trajectory.yaw_integration_mode`: default `SingleSupport` (MIT overrides to `always`). Accepted strings (`ControllerConfig.cpp:176-197`): `single_support`/`single`/`stance_single` → SingleSupport; `double_support`/`double`/`both_feet` → DoubleSupport; `always`/`continuous` → Always; anything else throws.

### 0.6 Per-tick call order (`SimulationRunner.cpp:409-423` → `MyController::runController`, `My_Controller.cpp:1374-1429`)

Before `runController`, in the same tick, `SimulationRunner::runRobotControl` does (`SimulationRunner.cpp:409-423`, `RobotRunner.cpp:73-81, 118-142`):
```
a  updateReducedBodyMassPropertiesFromData()   # bodyComLocation / bodyMass / bodyInertia (§0.3)
b  fillCheaterState(); stateEstimator.update()  # state.time = MuJoCo data.time
c  userCommand = keyboard; applyHeadlessUserCommandSchedule(t)   # §1.4
d  robotRunner.prepareController()  -> controller.prepareController() = syncLocomotionFSM()
       # only if _legInitializationComplete (from the PREVIOUS run) AND the controller is initialised
e  robotRunner.run() -> (leg initialiser until 2.0 s) else controller.runController()
```
So the FSM is updated **twice per tick**: the mode transition, `horizonClock.reset` and `resetSwingState()` (§1.4) take effect in the **prepare** call (d). The second call inside `runController` (step 1 below) at the same `t` then sees no transition (in StandingSettle/Walking it is idempotent; only the BrakingToStanding settle average would get two samples per tick — out of scope). On the very first controller tick (≈2.0 s) step d is skipped, because `_legInitializationComplete` only becomes true inside `run()` of that tick; initialisation then happens inside `runController` (`initializeController()`), followed by the normal sequence below on that same tick.

```
0  if not initialized: initializeController()  # first controller tick only (§1.4)
1  syncLocomotionFSM()                       # mode/state machine (§1.5); second call this tick (see above)
2  horizonClock.sync(t)                      # §1.1
3  x0 = buildCurrentMpcState()               # §0.2
4  dt = max(0, t - lastControlTime)          # lastControlTime is updated at the END of updateSwingTrajectories (step 9)
5  updateFilteredUserCommand(dt)             # §2
6  updateBodyTarget(x0, dt)                  # §3
7  swingFootPlanner.setBodyYawTargetWorld(bodyTarget.euler_W[2])
8  nominal = (Standing) ? {footPos_W with z=-0.005 per foot}          # My_Controller.cpp:1396-1405
                        : swingFootPlanner.desiredFootPositions()     # §5
   contactManager.update(..., nominal)                                # §6
   desired = contactManager.managedFootPositions(nominal)             # §6
9  updateSwingTrajectories(desired)          # §7  (sets lastControlTime = t)
10 updateTouchdownDebugTarget(desired)       # debug only
11 maybeUpdateMpc(x0, desired)               # every 7 ticks: constraints, ReferenceTrajectory (§4), QP
12 writeLegCommands()                        # stance wrench / swing PD (§7.5)
```
Ticks are 0.002 s apart; `dt` in steps 4-6 equals 0.002 except on two ticks where it is exactly 0: (i) the first controller tick (≈2.0 s; `initializeRuntimeObjects` sets `lastControlTime = t`), and (ii) the StandingSettle→Walking transition tick (≈3.0 s; `resetSwingState()` in the prepare call sets `lastControlTime = t`). With `dt = 0` the command filter is bypassed (`alpha = 1`, §2) and `advanceYaw` does not advance.

---

## 1. Phase / timing / clock / start-up

### 1.1 HorizonClock (`My_Controller/include/MyController/HorizonClock.h:9-38`)

```
t0            : cycle origin (s). reset(t0) sets it.                          # :14-16
sync(t):  while (t - t0 >= cycle): t0 += cycle                                # :18-26  (never moves backwards)
tk(k)  = t0 + k * dtMpc                                                       # :32-34
```
**Important quirk:** horizon step `k` lives at `t0 + k·dt`, i.e. the horizon starts at the **beginning of the current gait cycle**, not at the current time. Because `horizon == cycle == 0.5 s`, `tk` covers exactly one full cycle. Gait constraints (`GaitScheduler.cpp:153-156`) and the yaw gate (§4.2) are evaluated at these `tk`; the reference *state* is nevertheless rolled out from `x0` (current state) as if step 0 were "now". Nothing else consumes or shifts `tk`: its only consumers are `GaitScheduler::buildConstraintMatrices` (`GaitScheduler.cpp:153-156`) and the reference build (`ReferenceTrajectory.cpp:40-41, 63`, yaw-gate sample time, irrelevant under `always`); `MPCFormulation` uses only `psi[k]` (`MPCFormulation.cpp:64`). Example: 0.10 s into a cycle (Left in swing), step 1 uses the pattern at cycle time 0.02 s (double support, Left stance) instead of the pattern at 0.12 s (Left swing). Only step 0 is corrected to the measured active contact (`contact_lock_steps 1`, §1.3). Reproduce as-is behind a flag (O1, confirmed).

**Floating point:** reproduce the arithmetic literally. `t` = MuJoCo `data.time` (accumulated by `mj_step` as `time += 0.002`, *not* `n·0.002`; e.g. after 1500 steps from 0 it is `2.999999999999891`); `t0` = `data.time` at the transition tick, afterwards changed **only** by `t0 += 0.5` in `sync`; `tk = t0 + k*(0.5/25)`. Do not use integer tick counters or rounded phases: several boundaries are hit exactly (see §1.2).

`sync` is called at the top of every tick (`My_Controller.cpp:1387`) and again inside `desiredFootPositions()` (`SwingFootPlanner.cpp:283`, `:80-86`).

### 1.2 Gait phase and stance predicate (`GaitScheduler.cpp:98-124`)

```
p(side, t):                                                     # :98-110
    if mode == Standing: return 0.0
    phi = 0.5 if side == Left else 0.0
    return fmod((t - t0)/cycle + phi, 1.0)                     # C fmod; argument is >= 0 after sync
c(side, t):                                                     # :112-120
    if mode == Standing: return True
    return 0.0 <= p(side,t) < stance/cycle                      # stance fraction 0.66
bothFeetStance(t) = c(Left,t) and c(Right,t)                    # :122-124
```
The phase is purely clock-driven (no contact events, no adaptation). With MIT values (`x = (t - t0)/0.5`, `[..)` intervals):

| cycle fraction x | time in cycle | Right | Left |
|---|---|---|---|
| [0.00, 0.16) | [0.000, 0.080) s | stance | stance (**double support**) |
| [0.16, 0.50) | [0.080, 0.250) s | stance | **swing** |
| [0.50, 0.66) | [0.250, 0.330) s | stance | stance (**double support**) |
| [0.66, 1.00) | [0.330, 0.500) s | **swing** | stance |

Each swing is the last 0.34 of that leg's own phase (`p ∈ [0.66,1)`), so *touchdown is at the phase wrap*. Double support = `2·stance − cycle = 0.16 s` per cycle (0.08 s twice).

**Boundary sensitivity (float):** horizon step `k = 4` samples exactly the Left lift-off boundary (`tk(4) − t0 = 0.08 s` ⇒ `p_L = 0.66 = stance/cycle`), so `c(Left, tk(4))` depends on the bits of `t0`: `t0 = 3.0` or `2.999999999999891` → `p_L = 0.6600000000000001` (swing); `t0 = 3.999999999999891` → `p_L = 0.6599999999999993` (stance) (computed). Control ticks at the phase boundaries (t0+0.08, 0.25, 0.33, 0.5) and the `sync` test `t − t0 >= 0.5` are equally sensitive to the accumulated `data.time`. Evaluate literally `p = fmod((t − t0)/0.5 + phi, 1.0)`, `c = 0 <= p < 0.33/0.5` (with `0.33/0.5` computed in double, = `0.66`), exactly as in §8.

Remaining swing time (`My_Controller.cpp:59-61`, also inline in `SwingFootPlanner.cpp:247-251`):
```
remainingSwingTime(side, t) = clamp(cycle * (1 - p(side,t)), 0, swing)      # ∈ (0, 0.17]
```

### 1.3 Horizon contact schedule (what the MPC sees) (`GaitScheduler.cpp:134-193`)

For `k = 0..24`: `leftStance = c(Left, tk(k))`, `rightStance = c(Right, tk(k))`, overwritten for the first `contact_lock_steps = 1` steps by the ContactManager override (`ContactManager.cpp:416-446`): `leftContact = activeContact(Left)`, `leftNormalForceMinScale = activeContact ? contactRampAlpha : 0` (same for right). Bounds (QP rows `C·u ≤ C_bound`, lower bound `−inf`, `ConvexMPC.cpp:886-887`): stance → `Fz ≤ normal_force_max`, `Fz ≥ scale·normal_force_min` (`Ck_bound[4]=max, [5]=−scale·min` for Left, `[16],[17]` for Right); every other bound entry is 0. Swing → both Fz entries are 0 as well, so the rows give `Fz ≤ 0` and `−Fz ≤ 0` ⇒ `Fz = 0`, and then the friction rows (`±Fx − μFz ≤ 0`, `±Fy − μFz ≤ 0`), CoP rows (`±Mx − w·Fz ≤ 0`, `±My − l·Fz ≤ 0`) and torsion rows force `Fx = Fy = Mx = My = Mz = 0` (`GaitScheduler.cpp:83-96, 154-177`). The inequality block alone therefore already enforces a zero 6-D wrench for a swing foot; any zero-wrench equality in the QP (02 spec) enforces the same thing redundantly.

### 1.4 Start-up timeline (sim, `requested_locomotion_mode: walking`)

| sim time | what happens | source |
|---|---|---|
| t = 0 | MuJoCo model loaded; `_keyboardInputEnableTime = 0 + post_init_standing_settle_time = 1.0` | `SimulationRunner.cpp:270-271` |
| 0 → 2.0 s | **Leg PD initialiser**: every tick a B-spline (`BS_BasicDyn<T,3,1,2,2>`) from the *current* joint angles (`ini`) to `fin = default_qpos(MJCF qpos0) + leg_joint_offsets` (offsets added per joint index for each leg, `InitialPoseConfig.h:20-36`; midpoint = fin) over `end_time = leg_initialization_time = 2.0 s`; legs in `JointPd` mode with `joint_tracking.leg kp/kd`; controller **not** run. Arms same with 1.0 s. Complete when `_curr_time >= 2.0` (`LegPosInitializer.cpp:26-55`, `RobotRunner.cpp:119-142`). | |
| ≈2.0 s | first `runController()` → `initializeRuntimeObjects()`: `HorizonClock(t)`, `GaitScheduler`, `ContactManager`, `SwingFootPlanner`, `LocomotionFSM(startTime=t)` → initial state **StandingSettle** (because walking requested and settle time > 0, `LocomotionFSM.cpp:72-88`), mode = **Standing**; `updateFilteredUserCommand(0.0)` (first call ⇒ `filtered := raw`, no smoothing); `_legRuntime[leg].wasInStance = c(side,t)`, `touchdownYaw_W = swingFootYawTargetWorld(side)` (= `yaw_W_unwrapped + 1.0·psi_dot_f·Tp + bias(side, psi_dot_f)`, §7.4; **never reset later** except by a swing reset); `_bodyTarget = {}` (uninitialised); `lastControlTime = t`. Then the normal tick sequence runs on the same tick with `dt = 0`. | `My_Controller.cpp:401-474` |
| 2.0 → 3.0 s | Standing MPC (both feet stance, `p = 0`, `c = true`); body target = average foot xy, yaw integrated from psi_dot (§3). The standing MPC reference is also built from the **filtered command** (§4.1/§4.3 have no mode check): `v_ref = Rz(psi_k)[x_dot, y_dot, 0]` and `p_ref` advances from the average-foot xy, so a non-zero `x_dot` already moves the standing reference. | `My_Controller.cpp:711-724, 966-989` |
| ≈3.0 s | FSM `StandingSettle → Walking` (`time - stateStart >= 1.0`, `LocomotionFSM.cpp:230-234`) — happens in the **prepare** call (§0.6), output `resetGaitClock = resetSwingState = true` → `horizonClock.reset(t)` (t0 := t, so the walking cycle starts here with **Right stance/Left stance double support**), `resetSwingState()` (swing trajectories deactivated, `wasInStance = c(side,t)`, `swingFootPlanner.reset()`, `contactManager.reset()`, `lastControlTime = t`; `touchdownYaw_W` is **not** touched). `seedBodyTargetFromCurrentState()` is **commented out** (`:571`). Because `lastControlTime = t`, `runController` on this tick has `dt = 0` ⇒ `filtered := raw` exactly (§2). | `My_Controller.cpp:555-577, 534-553` |
| ≥3.0 s | Walking. | |

**Commands and the filter at start-up.** The filter only smooths changes made *after* controller init, and it is bypassed (`filtered := raw`) on two ticks: the first controller tick (≈2.0 s) and the walking-transition tick (≈3.0 s, `dt = 0` ⇒ `alpha = 1`). Headless demo runs (only when `CONVEXMPC_HEADLESS_AUTO_WALK` is set and not `0/false/FALSE`) drive **only `x_dot`** (`SimulationRunner.cpp:452-537`): either `CONVEXMPC_HEADLESS_X_DOT_PROFILE` (piecewise-constant `(time, value)` list), or `CONVEXMPC_HEADLESS_X_DOT_FINAL` from `CONVEXMPC_HEADLESS_X_DOT_START_TIME` (default `_keyboardInputEnableTime = 1.0 s`) with an optional ramp `rate·(t − start)` (`CONVEXMPC_HEADLESS_X_DOT_RATE`, default 0 = **step**). In the default setup raw `x_dot` = FINAL (clamped to ±0.7) from t = 1.0 s, so `filtered x_dot` equals the final command from t≈2.0 s: the standing phase already carries it in its reference, and **walking starts with the full commanded velocity (no ramp)**. With a later start or a ramp, the transition tick still snaps `filtered` to whatever raw is at ≈3.0 s. A smooth start therefore needs START_TIME after the transition tick (e.g. ≥ 3.01 s; the transition tick's `data.time` may be `2.99999…`) and/or a RATE. `y_dot` and `psi_dot` demos come only from the keyboard (steps 0.05 m/s and 0.1 rad/s per key press, `KeyboardCommand.h:39-40`, clamped by the yaml maxima; keyboard starts at t ≥ 1.0 s); keyboard changes are a staircase that the filter smooths. The port must reproduce both bypass ticks.

### 1.5 LocomotionFSM outputs relevant here (`LocomotionFSM.cpp:275-289`)

`mode = Walking` for states Walking/BrakingToStanding, `Standing` for StandingSettle/Standing. `zeroMotionCommand = (state == BrakingToStanding)` forces the raw command to zero in §2. In the non-interactive `walking` config the FSM never leaves Walking after 3.0 s (toggle requests are ignored unless `requested_locomotion_mode: interactive`, `LocomotionFSM.cpp:129-133`). Braking-to-standing details are out of scope for the 120 s tests.

---

## 2. User command filter (`My_Controller.cpp:28-33, 610-654`)

```
lowPassBlendAlpha(tau, dt):                                # :28-33
    if not (tau > 0) or not (dt > 0): return 1.0
    return clamp(-expm1(-dt/tau), 0, 1)                    # = 1 - exp(-dt/tau)
```
**`tau` is a time constant in seconds**, not a per-tick alpha. With `dt = 0.002`: `alpha_x = 1-exp(-0.002/0.92) = 2.1716e-3`, `alpha_y = 2.4969e-3`, `alpha_psi = 2.8531e-3` per tick (63 % rise in 0.92 / 0.80 / 0.70 s).

```
updateFilteredUserCommand(dt):                             # :610-654
    raw = clampUserCommand(userCommand or {})              # clamp x_dot∈[-0.7,0.7], y_dot∈[-0.5,0.5], psi_dot∈[-2,2]  (ControllerConfig.cpp:649-656; inf limit = no clamp)
    if zeroMotionCommand: raw.x_dot = raw.y_dot = raw.psi_dot = 0
    if not initialized: filtered = raw; initialized = True; return        # first call (dt = 0)
    prev = filtered
    filtered = raw                                          # copies the integer request counters etc.
    if raw.x_dot == 0 and raw.y_dot == 0 and raw.psi_dot == 0:            # exact zero snap
        filtered.x_dot = filtered.y_dot = filtered.psi_dot = 0
    else:
        filtered.x_dot   = prev.x_dot   + alpha(x_dot_tau,  dt) * (raw.x_dot   - prev.x_dot)
        filtered.y_dot   = prev.y_dot   + alpha(y_dot_tau,  dt) * (raw.y_dot   - prev.y_dot)
        filtered.psi_dot = prev.psi_dot + alpha(psi_dot_tau,dt) * (raw.psi_dot - prev.psi_dot)
    filtered.body_height_offset_m = raw.body_height_offset_m              # NOT filtered
    filtered.standing_roll_offset_rad  = prev + alpha(standing_roll_offset_tau, dt)*(raw - prev)
    filtered.standing_pitch_offset_rad = prev + alpha(standing_pitch_offset_tau,dt)*(raw - prev)
    filtered = clampUserCommand(filtered)
```
Bypass ticks: the first call (controller init, ≈2.0 s) copies `raw`; on the walking-transition tick (≈3.0 s) `dt = 0` (§0.6) ⇒ every `alpha = 1` ⇒ `filtered = raw` exactly. The initialised flag is never reset. See §1.4 for what this means for the demo commands.
UserCommand fields (`common/include/Utilities/UserCommand.h:4-13`): `x_dot, y_dot [m/s, body-yaw frame B], psi_dot [rad/s], body_height_offset_m, standing_roll_offset_rad, standing_pitch_offset_rad` (+ two request counters). The filtered command (`_filteredUserCommand`) is what the planner, reference trajectory and swing-yaw use (`SwingFootPlanner` holds a pointer to it, `My_Controller.cpp:414-419`).

---

## 3. Body target (`MyController::updateBodyTarget`, `My_Controller.cpp:677-736`) — feeds §4 and §5

```
if not bodyTarget.initialized:                                    # first tick after init
    if initial_pose has base pose (MIT: yes):
        nominalPosition_W = base_position_W + Rz(base_rpy_W.z) * bodyComLocation   # :356-361, :685-689
        euler_W           = base_rpy_W                            # [0,0,0]
    else: nominalPosition_W = x0[3:6]; euler_W = [0,0,x0[2]]
    eulerSeed_W    = euler_W
    nominalHeight_W = nominalPosition_W.z                          # = 0.679472 + bodyComLocation.z (evaluated at that tick)
    initialized = True
h   = filtered.body_height_offset_m; rOff = filtered.standing_roll_offset_rad; pOff = filtered.standing_pitch_offset_rad
applyPoseOffsets():  euler_W[0] = eulerSeed_W[0] + rOff ; euler_W[1] = eulerSeed_W[1] + pOff
                     nominalPosition_W.z = nominalHeight_W + h ; position_W = nominalPosition_W
if mode == Standing:
    nominalPosition_W.xy = mean of footPos_W.xy over legs          # :338-354
    euler_W[2] = advanceYaw(gait, euler_W[2], filtered.psi_dot, dt, t, yawMode)   # §4.2
    applyPoseOffsets(); return
# Walking
nominalPosition_W.xy = x0[3:5]                                     # anchored to estimated COM
euler_W[2] = advanceYaw(gait, euler_W[2], filtered.psi_dot, dt, t, yawMode)       # with 'always': += psi_dot*dt every tick
applyPoseOffsets()
```
Consequences: `bodyTarget.euler_W[2]` = `base_rpy_W.z` (yaml **0**, *not* the measured yaw) + `Σ psi_dot_f·dt` over every tick since controller init, in Standing **and** Walking (`dt = 0` on the init and transition ticks), a **pure open-loop integral of the filtered yaw-rate command**, never re-anchored (`_bodyTarget` is reset only at init, `My_Controller.cpp:471`; `seedBodyTargetFromCurrentState` is commented out, `:571`). It is used **only** by the swing-foot planner (`setBodyYawTargetWorld`, §5.3) as `yaw0`; the one exception is a turn-dominant stop, where the planner uses the measured `yaw_W_unwrapped` instead (`turnStopFrameValid`, §5.5). The MPC yaw reference instead starts from the *measured* `x0[2]` (§4.1), and the foot-yaw target `touchdownYaw_W` (§7.4) also uses the *measured* yaw — so a body lagging the commanded turn gets footholds placed in the commanded frame but feet oriented to measured yaw + lead. Roll/pitch reference = `0 + standing offsets` (0 unless keys i/k/j/l pressed).

`nominalHeight_W` = `base_position_W.z` (yaml 0.679472, **not** the measured torso z) + `(Rz(base_rpy_W.z = 0)·bodyComLocation).z`, where `bodyComLocation` is the value computed on that same first controller tick (≈2.0 s) = `Rz(ψ)ᵀ(upperBodyCom_W − torsoPos_W)` (§0.3). It does not depend on leg joints, but does depend on the arm joints and torso roll/pitch at that tick. Fixed for the whole run. Project model `MPC/models/mit_humanoid/mit_humanoid.xml` (keyframe `stand` or `init_yaml`, arms `[0,0,0,−1.65]`, torso upright; computed): `bodyComLocation = [0.01531, 0.00286, 0.11657]` m, upper-body mass 14.2275 kg ⇒ **`nominalHeight_W ≈ 0.79605 m`**.

---

## 4. Reference trajectory over the horizon (`ReferenceTrajectory.cpp:8-83`, `BodyMotionReference.cpp`)

### 4.1 Seed (`My_Controller.cpp:966-978`, only when the MPC is (re)solved)

```
seed = x0.copy()
if mode == Standing:
    seed[0:3] = bodyTarget.euler_W ; seed[3:6] = bodyTarget.nominalPosition_W ; seed[5] = bodyTarget.nominalHeight_W
else:  # Walking
    seed[0] = bodyTarget.euler_W[0]        # roll  ref (0 + offset)
    seed[1] = bodyTarget.euler_W[1]        # pitch ref (0 + offset)
    seed[5] = bodyTarget.nominalHeight_W   # nominal COM height (WITHOUT height offset; added in build)
    # seed[2] (yaw), seed[3:5] (xy), seed[6:12] (rates) stay = x0  (measured)
cmd = filtered user command
ReferenceTrajectory(&cmd, seed, desiredFootPositions, horizonClock, gaitScheduler).build(out)
```

### 4.2 Yaw gate helpers (`BodyMotionReference.cpp:7-43`)

```
isDoubleSupport(t) = gait.bothFeetStance(t)
shouldAdvanceYaw(t, mode):  SingleSupport -> not isDoubleSupport(t)
                            DoubleSupport -> isDoubleSupport(t)
                            Always        -> True            # MIT config
advanceYaw(cur, psiDot, dt, t, mode) = cur + psiDot*dt   if dt > 0 and shouldAdvanceYaw(t,mode) else cur   # NO wrap
yawRate(psiDot, t, mode)             = psiDot            if shouldAdvanceYaw(t,mode) else 0
advancePlanarPosition(p_W, yaw, cmd_B(2), dt) = p_W + Rz(yaw) @ [cmd_B.x, cmd_B.y, 0] * dt   (if dt>0)
worldVelocity(cmd_B(3), yaw)                  = Rz(yaw) @ cmd_B
```
MIT uses `always`, so yaw is integrated on every horizon step and `wz_ref = psi_dot` on every step. (`single_support`/`double_support` exist but are unused for MIT; docs describe `single_support`.)

### 4.3 Build (`ReferenceTrajectory.cpp:8-83`) — output arrays `X_ref (13·25)`, `r_left (3×25)`, `r_right (3×25)`, `psi (25)`, `tk (25)`

```
psi_dot   = cmd.psi_dot ; h = cmd.body_height_offset_m
u_B       = [cmd.x_dot, cmd.y_dot, 0]                      # z command is always 0 (:20-23)
psi0 = seed[2]; g = seed[12]; dt = 0.02; N = 25; mode = always
p_ref  = seed[3:6].copy(); p_ref.z += h                     # :32-33
euler  = seed[0:3].copy()
psi_ref = psi0
for k in range(N):
    tk = t0 + k*dt
    ts = tk if k == 0 else tk - 0.5*dt                        # yaw-gate sample time (irrelevant for 'always')
    if k > 0: psi_ref = advanceYaw(psi_ref, psi_dot, dt, ts, mode)      # psi_k = psi0 + k*psi_dot*dt
    psi_k  = psi_ref
    v_ref_W = Rz(psi_k) @ u_B                                 # uses the ALREADY advanced yaw
    if k > 0:
        p_ref = advancePlanarPosition(p_ref, psi_k, u_B[:2], dt)        # p_k = p_{k-1} + Rz(psi_k) u_B dt  (from previous REFERENCE, not from x0 each step)
        p_ref.z = seed[5] + h                                 # constant height
    out.tk[k] = tk ; out.psi[k] = psi_k
    out.r_left[:,k]  = left_des_W  - p_ref                   # same foot point for every k (:65-66)
    out.r_right[:,k] = right_des_W - p_ref
    X_ref[13k:13k+13] = [euler.roll, euler.pitch, psi_k, p_ref.x, p_ref.y, p_ref.z,
                         0, 0, yawRate(psi_dot, ts, mode),   # = psi_dot for 'always'
                         v_ref_W.x, v_ref_W.y, v_ref_W.z(=0), g]
```
Per-state summary at step k (walking):

| idx | state | reference |
|---|---|---|
| 0,1 | roll, pitch | `0 + standing offsets` (constant over horizon) |
| 2 | yaw | `x0.yaw + k·psi_dot·0.02` (unwrapped, not wrapped) |
| 3,4 | px,py | `x0.xy + Σ_{j=1..k} Rz(psi_j)[x_dot,y_dot]·0.02` (x0 = measured COM; integration uses previous reference sample, yaw at the *new* sample) |
| 5 | pz | `nominalHeight_W + body_height_offset_m` (constant; not the measured height) |
| 6,7 | wx, wy | 0 |
| 8 | wz | `psi_dot` (every step, mode always) |
| 9,10 | vx, vy | `Rz(psi_k)[x_dot, y_dot]` |
| 11 | vz | 0 |
| 12 | g | −9.81 |

### 4.4 `r_left / r_right` per step

`left_des_W`, `right_des_W` = `desiredFootPositions` computed **this tick** (§5, then §6 override), the *same* 3-vector for all 25 steps; `r = foot − p_ref[k]`:
- stance leg: the cached touchdown target of that leg (the planned touchdown point when it last swung; only on the very first stance after `reset()` it is seeded with the *measured* `footPos_W`, §5.6). Its z is `−0.005` (or the measured z on first seeding).
- swing leg: the planned touchdown target (z = −0.005).
- ContactManager may replace either with the frozen measured foot position (early contact) or the search target (late contact, disabled for MIT) (§6).
So `r` is **not** the measured foot position during normal stance; it is the planned foothold. `r.z ≈ −0.005 − (nominalHeight + h)`.

---

## 5. Swing-foot touchdown planner (`SwingFootPlanner.cpp`) — called once per control tick in Walking

### 5.1 Persistent state (`SwingFootPlanner.h:55-69`, `reset()` `:29-44`)

```
touchdownTargets[2] (W), touchdownTargetValid[2]=False, nominalFootOffsets_B[2], nominalFootOffsetValid[2]=False,
wasInStance[2]=True, bodyYawTarget_W, bodyYawTargetValid=False,
stopRecenterClearTicks=0, stopRecenterWasActive=False,
previousPlanarCommand_B=0, previousYawRateCommand=0, previousCommandValid=False,
turnStopCenter_W=0, turnStopYaw_W=0, turnStopFrameValid=False
kSwingFootTargetZ = -0.005                                     # :10
```
`reset()` is called at every FSM transition with `resetSwingState` (i.e. at ≈3.0 s). `seedTouchdownTargets()` exists (`:46-68`) but is **never called by the controller** (only by a test).

### 5.2 Nominal foot offsets (`ensureNominalFootOffsets`, `:109-147`)

MIT yaml provides `nominal_foot_offsets_B`, so: `nominalFootOffsets_B = [[0, +0.075999602, 0], [0, −0.075999602, 0]]` (index 0 = Left, 1 = Right), done once. (Fallback when absent: `offset_B = Rz(yaw0)^T (footPos_W − footCenter_W)` with x and z zeroed.)

### 5.3 Helpers (`:153-233`)

```
planarCmd_B = [f.x_dot, f.y_dot]                                                   # :153-156 (f = filtered cmd)
selectedHalfStanceOffset(planarCmd):                                               # :158-169
    s = |planarCmd|
    if s > 0.65: return 0.26
    if s > 0.60: return 0.28
    return 0.37                                                                     # strict '>' : s == 0.60 -> 0.37
bodyYawTargetWorld() = bodyYawTarget_W if valid else yaw_W_unwrapped               # :171-173  (valid every tick in practice: bodyTarget.euler_W[2] of §3)
touchdownPreviewTime(planarCmd) = max(0, (0.5 + selectedHalfStanceOffset) * stance) # :175-178
      -> 0.2871 s (≤0.60 m/s), 0.2574 s (0.60,0.65], 0.2508 s (>0.65)
stopRecenterRequested(planarCmd) = |planarCmd| <= 0.02 and |f.psi_dot| <= 0.02     # :180-185
stopRecenterActive(planarCmd):                                                     # :187-205  (stateful latch)
    if requested: clearTicks = 0; return True
    if not stopRecenterWasActive: clearTicks = 0; return False
    if clearTicks < 5: clearTicks += 1; return True                                # stays active 5 more ticks after the command leaves the deadband
    return False
computeStopStanceCenterWorld():                                                    # :207-233
    yaw = turnStopYaw_W if turnStopFrameValid else bodyYawTargetWorld()
    if hasStopBrakingOffset: off_B = stop_braking_offset_B                         # MIT: no
    else:
        v_B = Rz(yaw)^T @ reducedBodyComVelocityWorld
        off_B = [0.2*v_B.x, 0.2*v_B.y, 0]                                          # capture-point-like braking term, gain 0.2
        n = |off_B.xy|; if 0.08 > 0 and n > 0.08: off_B.xy *= 0.08/n              # max 0.08 m
    center = turnStopCenter_W if turnStopFrameValid else reducedBodyComWorld + Rz(yaw) @ off_B
    center.z = -0.005 ; return center
```

### 5.4 Touchdown target formula (`touchdownTargetWorldBodyVelocityHalfStance`, `:235-275`)

```
target(leg, planarCmd, stopRecenter):
    v_B      = [planarCmd.x, planarCmd.y, 0]
    Tp       = touchdownPreviewTime(planarCmd)
    psi_dot  = f.psi_dot
    yaw0     = turnStopYaw_W if turnStopFrameValid else bodyYawTargetWorld()
    yawTrans = yaw0 + 0.5*psi_dot*Tp
    Trem     = clamp(cycle*(1 - p(side, t)), 0, swing)               # remaining swing time of THIS leg
    yawTd    = yaw0 + psi_dot*Trem
    step_W   = Rz(yawTrans) @ v_B * Tp                                # Raibert-style "half-stance" preview: v * (0.5+k)*T_stance
    center_W = computeStopStanceCenterWorld() if stopRecenter else reducedBodyComWorld
    off_B    = nominalFootOffsets_B[leg]
    planned_B = Rz(yaw0)^T @ step_W + Rz(yawTd - yaw0) @ off_B
    if stopRecenter: planned_B = Rz(yawTd - yaw0) @ off_B             # no step preview when stopping
    # lateral crossing guard
    if off_B.y > 0: planned_B.y = max(planned_B.y, 0)
    elif off_B.y < 0: planned_B.y = min(planned_B.y, 0)
    target = center_W + Rz(yaw0) @ planned_B
    target.z = -0.005
    return target
```
Terms, in the words of the task:
- **nominal_foot_offsets_B**: `off_B` (±0.076 m lateral, 0 fore-aft), rotated by the yaw the body is expected to have at touchdown (`yawTd`).
- **body_velocity_half_stance_offset (0.37 / 0.28 / 0.26)**: `Tp = (0.5 + k)·T_stance`; the foot lands `v_cmd·Tp` ahead of the **current estimated reduced-body COM** (no velocity-error feedback term, no capture point during walking).
- **capture-point term (gain 0.2, max 0.08, deadband 0.02)**: only in the stop-recenter branch, from *measured* COM velocity (§5.3).
- **stop braking latch**: `stop_braking_latch_clear_ticks = 5` (§5.3).
- **turning**: the translation preview is rotated by `yaw0 + ½ψ̇·Tp`; the lateral offset by `yaw0 + ψ̇·T_rem`. **`turn_tangential_lead_scale` is not used anywhere in the code** (yaml has it; parser ignores it — the docs say the "old extra tangential lead" was removed). **`swing_foot_yaw_lead_scale`** is used only in the foot *yaw* target (§7.4), not in the position.
- **stance_foot_yaw_hold**: not a planner term; it is a stance moment (§7.5).
- **min_remaining_time**: not used by the planner; used by the swing trajectory (§7.2).

### 5.5 Stop / turn-stop bookkeeping at the start of `desiredFootPositions()` (`:277-320`)

```
syncHorizonClock(); ensureCache(); ensureNominalFootOffsets()
planarCmd = planarCmd_B ; psiCmd = f.psi_dot
stop      = stopRecenterActive(planarCmd)
stopJustActivated = stop and not stopRecenterWasActive
if stopJustActivated and previousCommandValid:
    R = max over legs |nominalFootOffsets_B.xy| (= 0.076)
    prevTangential = |previousYawRateCommand| * R
    wasTurnDominant = |previousYawRateCommand| > 0.02 and |previousPlanarCommand_B| <= prevTangential + 0.02
    if wasTurnDominant:
        turnStopCenter_W = reducedBodyComWorld (z=-0.005); turnStopYaw_W = yaw_W_unwrapped; turnStopFrameValid = True
elif not stop:
    turnStopCenter_W = 0; turnStopYaw_W = 0; turnStopFrameValid = False
if stop and turnStopFrameValid:                              # follow the estimated frame while stopping after a turn
    turnStopCenter_W = reducedBodyComWorld (z=-0.005); turnStopYaw_W = yaw_W_unwrapped
```
(At the end of the call: `stopRecenterWasActive = stop; previousPlanarCommand_B = planarCmd; previousYawRateCommand = psiCmd; previousCommandValid = True`, `:365-368`.)

### 5.6 Per-leg target selection — when is the target (re)computed? (`:322-360`)

```
for side in (Left, Right):
    leg = index of side ; isStance = gait.c(side, t)          # NOMINAL schedule, not measured contact
    if isStance:
        if not touchdownTargetValid[leg]:                     # only right after reset(): seed from measured foot
            touchdownTargets[leg] = footPos_W[leg]; valid = True
        wasInStance[leg] = True
        return touchdownTargets[leg]                          # frozen: the last planned touchdown stays the "foot position" used in r_left/r_right
    was = wasInStance[leg]; wasInStance[leg] = False
    update = was or not valid[leg] or stopJustActivated or turnStopFrameValid
    if update:
        touchdownTargets[leg] = target(leg, planarCmd, stop); valid = True
    return touchdownTargets[leg]
```
So during ordinary walking the touchdown target is computed **once, on the first tick of the nominal swing** (`p` crosses 0.66; planner `wasInStance` follows the *nominal* `c`, independent of the driver's flag in §7.3), and then **frozen for the whole swing AND the following stance** (the stance branch returns the same cached value, used in `r`), except: (a) the tick a stop-recenter edge happens (`stopJustActivated`), (b) every tick while `turnStopFrameValid` (turn-dominant stop), (c) `!touchdownTargetValid` (only after construction/`reset()`). At constant command in straight walking none of (a)-(c) holds. `traj.setFinalPosition(target)` is called every swing tick (§7.3) but with the same value. The only other change is the ContactManager early-contact override (§6), which replaces the desired position with the frozen *measured* foot position until the schedule reaches stance; after that the planner's cached target is returned again. `Trem` at that first swing tick ≈ 0.17 s, so `yawTd ≈ yaw0 + 0.17·ψ̇`.

Note the target uses the COM/yaw **at swing start**; the body then keeps moving for 0.17 s, so with `Tp = 0.287 s` the foot lands ≈ `v·(0.287 − 0.17) ≈ 0.117·v` ahead of the COM at touchdown (≈ 7 cm at 0.6 m/s). This is the implicit Raibert balance of the reference.

### 5.7 Output

`DesiredFootPositions{left_des_W, right_des_W}` → ContactManager (§6) → swing trajectory final position (§7) and `r_left/r_right` (§4.4).

---

## 6. ContactManager: what it does to the planner output and to "is this leg in swing?" (`ContactManager.cpp`)

The nominal schedule `c(side,t)` decides *planning*; the **active contact** decides swing/stance *execution* (`activeContactForSide`, `My_Controller.cpp:738-746`: walking → `contactManager.activeContact(side)`, standing → `c(side,t)`). Per leg per tick (`update`, `:200-359`), MIT values:

```
scheduled = c(side, t)
estimated: hysteresis on contactNormalForce: turn ON when force >= 36 N for 1 tick, OFF when force <= 0.5 N for 4 consecutive ticks   (:89-148)
releasedContactDuringSwing: set False when scheduled; set True when (not scheduled and not estimated)
earlyContact = enable_early(true) and not scheduled and estimated and releasedContactDuringSwing
lateContact  = enable_late(false) and ... -> always False for MIT
if earlyContact and not previousActiveContact: frozenTouchdownPosition_W = footPos_W
commandedFootTarget_W = frozen if earlyContact else nominalTarget (search branch unreachable)
activeContact = True if earlyContact else scheduled
if activeContact and not earlyContact: frozenTouchdownPosition_W = footPos_W  (bookkeeping only)
contactRampAlpha = clamp(rampTime/0.01, 0, 1), rampTime restarts at 0 on each activation; 0 when inactive
    # rampTime += dt_cm on later active ticks, dt_cm = ContactManager's own (time − _lastUpdateTime)   (:340-353)
```
Ramp tick by tick: on the **first** active tick `rampTime = 0` ⇒ `alpha = 0` exactly — zero stance force/moment feed-forward (`−alpha·F`), **no** stance yaw hold (it requires `alpha > 0`), and MPC step-0 `Fz_min` scale 0 (`buildHorizonOverride`, §1.3). Then `alpha = 0.2, 0.4, 0.6, 0.8`, and `1.0` from the **6th** active tick on (0.002 s ticks, 0.01 s ramp; with accumulated `data.time` differences the 6th value can be `0.99999…`, reaching exactly 1 on the 7th — compute it literally). `reset()` sets `_lastUpdateTime = t`, so the first `update()` after it (same control tick, §0.6) has `dt_cm = 0`; legs active at reset start with `rampTime = 0.01` (alpha 1).
`managedFootPositions(nominal)` (`:394-414`) replaces a side's target with `commandedFootTarget_W` **only** while `searchModeActive or earlyContact or liftoffHold` — for MIT, only during early contact (measured foot position frozen at first detection). Otherwise the planner's target passes through unchanged.

Effect on swing: a swing leg that touches down early (foot force ≥ 36 N after having been unloaded) becomes stance immediately (wrench mode, ramp 10 ms) and its `r` uses the frozen measured position until the nominal schedule catches up. A leg that is scheduled stance but has no force stays "stance" (late handling disabled).

**Early-contact bounce:** if, after an early touchdown, the foot unloads again before the scheduled stance (force ≤ 0.5 N for 4 ticks ⇒ `estimated = false`), then `releasedContactDuringSwing = true`, `earlyContact = false`, `activeContact = scheduled = false` — the leg returns to swing. The driver (§7.3) sees `wasInStance = true` and starts a **new** smoothstep swing from the current foot position to the (unchanged) planner target, `T = max(remaining nominal swing, 1 ms)`, apex `pFinal.z + 0.06`, and re-latches `touchdownYaw_W`. Early contact can re-trigger on the next ≥ 36 N contact (new frozen position latched when `earlyContact and not previousActiveContact`).

`reset()` (`:150-198`) at ≈3.0 s: `activeContact = scheduled`, ramp alpha 1 for active legs, `releasedContactDuringSwing = not scheduled and not estimated`.

---

## 7. Swing trajectory (`SwingFootTrajectory.cpp`, driven by `updateSwingTrajectories` `My_Controller.cpp:792-868`)

### 7.1 Curve type

**Not** Bézier/B-spline. `common/include/Utilities/BezierCurve.h` and `BSplineBasic.h` are **not used** by the swing code (only `BSplineBasicDynamic.h` is used by the start-up joint initialisers). The swing curve is a cubic smoothstep blend:
```
smoothBlend(s)     = 3s² − 2s³           # :129-131
smoothBlendDot(s)  = 6s − 6s²            # :133-135
smoothBlendDdot(s) = 6 − 12s             # :137-139
```

### 7.2 Object (`SwingFootTrajectory.h`, `.cpp:6-127`)

```
reset(pInit, pFinal, height, swingTime):   swingTime must be > 0    # :6-21
    pInit, pFinal, H = height, T = swingTime, remaining = T, active = True; updateOutputs()
setFinalPosition(pFinal): pFinal = new; if active: updateOutputs()   # :23-28   (target can be moved mid-swing; blend restarts from pInit, no jump filtering)
advance(dt): if not active: return                                   # :30-40
    remaining = max(0, remaining − max(0,dt)); updateOutputs(); if remaining <= 0: active = False
phase() = 1 − remaining/T   (1 if T<=0)                              # :58-63
deactivate(): remaining = 0; active = False; updateOutputs()         # :73-77

updateOutputs():                                                     # :79-127
    s = clamp(phase, 0, 1); ds = 1/T; d2s = 0
    b, bd, bdd = smoothBlend(s), smoothBlendDot(s), smoothBlendDdot(s)
    d = pFinal − pInit
    p.x = (1−b) pInit.x + b pFinal.x ; p.y likewise                  # XY: one smoothstep over the whole swing
    v.x = bd d.x ds ; v.y = bd d.y ds
    a.x = d.x (bdd ds² + bd d2s) ; a.y likewise
    zMid = pFinal.z + H                                              # apex = final z + 0.06 (NOT relative to pInit.z)
    if s <= 0.5:  u = 2s ; du = 2 ds ; dz = zMid − pInit.z          # up half: pInit.z -> zMid
        p.z = (1−B(u)) pInit.z + B(u) zMid ; v.z = Bd(u) dz du ; a.z = dz Bdd(u) du²
    else:         u = 2s − 1 ; du = 2 ds ; dz = pFinal.z − zMid     # down half: zMid -> pFinal.z
        p.z = (1−B(u)) zMid + B(u) pFinal.z ; v.z = Bd(u) dz du ; a.z = dz Bdd(u) du²
```
Lift-off and touchdown velocity are **zero** in all three axes (`Bd(0)=Bd(1)=0`); acceleration at the ends is `±6·d/T²` (XY) and `±6·dz·(2/T)²` (Z). Apex at `s = 0.5` with `v.z = 0`. `pFinal.z = −0.005` ⇒ apex height `+0.055` m in W; the down-half targets 5 mm below the floor. The up-half rises by `0.055 − pInit.z` (`pInit.z` = measured site z at the (re)reset tick). The apex is at `s = 0.5` of the trajectory's *own* `T` (the remaining nominal swing at reset, ≈0.17 s), not at the nominal mid-swing if the reset happened late. `pFinal` can change mid-swing via `setFinalPosition`, and `zMid` moves with it (`SwingFootTrajectory.cpp:102-126`).

### 7.3 Driver: when reset, when advanced, what if time runs out (`My_Controller.cpp:792-868`)

```
t = state.time ; dt = max(0, t − lastControlTime) ; minRem = 0.001
for each leg:
    isStance = activeContactForSide(side, t)                     # §6 (active, not nominal)
    if isStance:
        (BrakingToStanding bookkeeping) ; traj.deactivate() ; wasInStance = True ; wasSearchMode = False ; continue
    pFoot = footPos_W[leg] ; target = desired[side] ; fallbackYaw = swingFootYawTargetWorld() (§7.4) ; psi_dot = f.psi_dot
    if contactManager.searchModeActive(side):                    # never for MIT (late handling off)
        ... reset(pFoot, target, 0.0, max(0.08, minRem)) / setFinalPosition+advance ...
        continue
    wasSearchMode = False
    timeRemaining = max(remainingSwingTime(side, t), minRem)      # clamp(cycle(1−p),0,swing), floor 1 ms
    if wasInStance or not traj.active():
        touchdownYaw_W[leg] = swingFootYawTargetWorldWithPsiOffset(psi_dot, side, fallbackYaw)   # §7.4, latched for the swing AND the following stance
        traj.reset(pFoot, target, 0.06, timeRemaining)           # starts from the MEASURED foot position
    else:
        traj.setFinalPosition(target) ; traj.advance(dt)         # target re-sent every tick (frozen by §5.6 anyway)
    wasInStance = False
lastControlTime = t
```
- A swing starts on the first tick where `activeContact` is false; duration = the remaining *nominal* swing (≈0.17 s minus the offset between control ticks and the phase boundary).
- **Running out of time**: `advance` reaches `remaining = 0` ⇒ `active = False` and the outputs sit at `pFinal` with zero velocity. If the leg is still not in active contact on the next tick (only possible for at most one tick, since at the phase wrap `scheduled` becomes true and `activeContact = scheduled`), `not traj.active()` triggers a **new reset** from the current foot position with `timeRemaining = max(≈0, 0.001) = 1 ms` (and `touchdownYaw_W` is re-latched). On that re-reset tick itself `reset()` evaluates the outputs at phase 0: `pDes = pInit` = the current measured foot position, `vDes = 0`, but `aDes = [6·d.x/T², 6·d.y/T², 6·dz·(2/T)²]` with `T = 1 ms`, `d = pFinal − pInit`, `dz = pFinal.z + 0.06 − pInit.z` (§7.2 end accelerations) — a large one-tick feed-forward spike that the port must reproduce (`aDes` enters the operational-space swing law, `OperationalSpaceDynamics.cpp:78`). Only from the next tick does `advance(0.002)` run, which snaps to `pFinal` (v = 0, a = −6·d/T² end value). Because `enable_late_contact_handling=false`, the leg becomes stance at the phase wrap whether or not the foot has touched the ground. `min_remaining_time` only guards `swingTime > 0` in `reset`.
- Early contact ends the swing early (§6): `traj.deactivate()`. If the foot then unloads again before the scheduled stance (early-contact bounce, §6), the next non-stance tick has `wasInStance = true`, so a fresh swing is reset from the current foot position to the unchanged planner target with `T = max(remaining nominal swing, 1 ms)`, and `touchdownYaw_W` is re-latched.
- On every `reset` tick (normal swing start, 1 ms re-reset, post-bounce restart) the commanded output is `p = pInit`, `v = 0`, `a = +6·d/T²` (XY) / `+6·dz·(2/T)²` (Z) — `a` is **not** zero at phase 0 (`smoothBlendDdot(0) = 6`); advancing starts on the following tick.

### 7.4 Swing-foot yaw target (`SwingYawTarget.h`, `My_Controller.cpp:891-909`)

```
swingFootYawTargetWorld() = yaw_W_unwrapped + swing_foot_yaw_lead_scale(1.0) * f.psi_dot * Tp     # Tp = touchdownPreviewTime(planar filtered cmd), §5.3
swingFootYawPsiOffset(side, psi_dot):                                                             # SwingYawTarget.h:11-27
    mag = clamp(100 deg/(rad/s) * |psi_dot|, 0, 20 deg)  [rad]
    +mag if psi_dot > 0 and side == Left ; −mag if psi_dot < 0 and side == Right ; else 0          # the inner foot of the turn is biased
swingFootYawTargetWorldWithPsiOffset(psi_dot, side, fallback) = liftAngleNear(fallback + bias, fallback)   # SwingYawTarget.h:29-34
```
Lifecycle of `legRuntime[leg].touchdownYaw_W`:
- **Initialised once** at controller init (≈2.0 s, `My_Controller.cpp:450-457`) to `swingFootYawTargetWorld(side)` = `yaw_W_unwrapped + 1.0·psi_dot_f·Tp + bias(side, psi_dot_f)` with the then-filtered command.
- **Not** reset at the walking transition (`resetSwingState`, `:534-553`, does not touch it). The Right leg stays in stance until `t0 + 0.33 s` (and the Left until `t0 + 0.08 s`) and uses this stale init value for the stance yaw hold and the MPC foot yaw until its first swing.
- **Overwritten on every** `traj.reset(...)` in the driver (§7.3: `wasInStance or not traj.active()`), with `liftAngleNear(fallback + bias, fallback)` — i.e. at nominal swing start, but also on the 1 ms re-reset when the trajectory expires before the phase wrap, and on a new swing after an early-contact bounce (`My_Controller.cpp:852-858`).

It is reused during the following stance for the yaw hold (§7.5) and as the MPC foot-yaw for constraint rotation (`mpcFootYawForSide`, `My_Controller.cpp:930-946`, used in `GaitScheduler.cpp:146-147, 187-188`):
- stance vs swing is decided by `activeContactForSide` (ContactManager active contact, **including early contact**; in Standing = `c = true`), not by the nominal schedule;
- stance → `liftAngleNear(atan2(R_WF[1,0], R_WF[0,0]), touchdownYaw_W)`, where `R_WF[:,0]` is the foot **site** x-axis; if that throws (no foot frame, non-finite axis, or `hypot(x,y) ≤ 1e-9`) it falls back to `touchdownYaw_W`;
- swing → `touchdownYaw_W`;
- the one per-leg value is applied to **all 25 horizon steps**, including future steps where a currently-swinging leg is predicted in stance; it is recomputed only on MPC-solve ticks.

E.g. ψ̇ = 1.3 rad/s: lead `1.3·0.2871 = 0.373 rad`, left-foot bias `+20°` (saturated).

### 7.5 How the swing/stance outputs are used (`writeLegCommands`, `My_Controller.cpp:1297-1372`) — for completeness

- Swing leg: `mode = SwingFoot`, `pDes_W/vDes_W/aDes_W = traj.position()/velocity()/acceleration()`, `kpCartesian = computeSwingCartesianKp(Jv, M, natural_frequency [151,151,110])`, `kdCartesian = diag(25,25,25)`, plus `computeSwingAttitudeLevelTorque(roll_kp=0, pitch 300/18, yaw 305/18, desiredYaw = touchdownYaw_W)` (`SwingAttitudeControl.h:9-84`: foot-pitch levelled to world-up, foot yaw to `touchdownYaw_W`, roll disabled for the 5-DOF leg).
- Stance leg: `mode = StanceWrench`, `forceFF_W = −alpha·F_mpc`, `momentFF_W = −alpha·M_mpc`; if `enable_stance_foot_yaw_hold` and `alpha > 0`: `momentFF_W.z += alpha·(20·wrap(touchdownYaw_W − footYaw) − 4·ω_foot,z)` (`My_Controller.cpp:86-117, 1338-1348`).

---

## 8. Compact Python reference of the pure functions (mirrors the C++; use with the state layout above)

```python
import numpy as np
def Rz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
wrap = lambda a: np.arctan2(np.sin(a), np.cos(a))
lift_near = lambda tw, cu: cu + wrap(tw - cu)

CYCLE, SWING, STANCE, HORIZON, N = 0.5, 0.17, 0.33, 0.5, 25
DT_MPC = HORIZON / N
class HorizonClock:
    def __init__(s, t0=0.0): s.t0 = t0
    def reset(s, t0): s.t0 = t0
    def sync(s, t):
        while t - s.t0 >= CYCLE: s.t0 += CYCLE
    def tk(s, k): return s.t0 + k * DT_MPC

def phase(clock, side, t, standing=False):          # side: 'L' or 'R'
    if standing: return 0.0
    return np.fmod((t - clock.t0) / CYCLE + (0.5 if side == 'L' else 0.0), 1.0)
def stance(clock, side, t, standing=False):
    return True if standing else (0.0 <= phase(clock, side, t) < STANCE / CYCLE)
def remaining_swing(clock, side, t):
    return float(np.clip(CYCLE * (1.0 - phase(clock, side, t)), 0.0, SWING))

def lp_alpha(tau, dt):
    return 1.0 if not (tau > 0 and dt > 0) else float(np.clip(-np.expm1(-dt / tau), 0, 1))

def half_stance_offset(planar):                     # planar = filtered [x_dot, y_dot]
    s = np.linalg.norm(planar)
    return 0.26 if s > 0.65 else (0.28 if s > 0.60 else 0.37)
def preview_time(planar): return max(0.0, (0.5 + half_stance_offset(planar)) * STANCE)

def build_reference(seed, cmd, clock, foot_L, foot_R, psi_dot_gate=lambda t: True):
    """seed: 13-vector (§4.1); cmd: dict x_dot,y_dot,psi_dot,h; returns X_ref(25,13), rL(25,3), rR(25,3), psi(25), tk(25)"""
    uB = np.array([cmd['x_dot'], cmd['y_dot'], 0.0]); h = cmd['h']; psid = cmd['psi_dot']
    p = seed[3:6].copy(); p[2] += h; euler = seed[0:3].copy(); psi = seed[2]
    X = np.zeros((N, 13)); rL = np.zeros((N, 3)); rR = np.zeros((N, 3)); PSI = np.zeros(N); TK = np.zeros(N)
    for k in range(N):
        tk = clock.tk(k); ts = tk if k == 0 else tk - 0.5 * DT_MPC
        gate = psi_dot_gate(ts)                       # 'always' -> True
        if k > 0 and gate: psi += psid * DT_MPC
        v = Rz(psi) @ uB
        if k > 0:
            p = p + Rz(psi) @ np.array([uB[0], uB[1], 0.0]) * DT_MPC; p[2] = seed[5] + h
        TK[k] = tk; PSI[k] = psi; rL[k] = foot_L - p; rR[k] = foot_R - p
        X[k] = [euler[0], euler[1], psi, p[0], p[1], p[2], 0, 0, psid if gate else 0.0, v[0], v[1], v[2], seed[12]]
    return X, rL, rR, PSI, TK

def smooth(s): return 3*s*s - 2*s*s*s
def smooth_d(s): return 6*s - 6*s*s
def smooth_dd(s): return 6 - 12*s
def swing_eval(p0, p1, H, T, remaining):
    s = float(np.clip(1 - remaining / T, 0, 1)); ds = 1 / T; d = p1 - p0
    b, bd, bdd = smooth(s), smooth_d(s), smooth_dd(s)
    p = p0.copy(); v = np.zeros(3); a = np.zeros(3)
    p[:2] = (1-b)*p0[:2] + b*p1[:2]; v[:2] = bd*d[:2]*ds; a[:2] = d[:2]*bdd*ds*ds
    zmid = p1[2] + H
    if s <= 0.5: u, du, dz = 2*s, 2*ds, zmid - p0[2]; z0, z1 = p0[2], zmid
    else:        u, du, dz = 2*s-1, 2*ds, p1[2] - zmid; z0, z1 = zmid, p1[2]
    p[2] = (1-smooth(u))*z0 + smooth(u)*z1; v[2] = smooth_d(u)*dz*du; a[2] = dz*smooth_dd(u)*du*du
    return p, v, a

def swing_yaw_bias(side, psi_dot):
    mag = float(np.clip(100.0 * abs(psi_dot), 0.0, 20.0)) * np.pi / 180
    if psi_dot > 0 and side == 'L': return mag
    if psi_dot < 0 and side == 'R': return -mag
    return 0.0
```
The stateful pieces (§2 filter, §3 body target, §5.5-5.6 planner latching, §6 contact manager, §7.3 driver) must be ported as classes following the pseudo-code in those sections; their tick order is §0.6.

---

## 9. Docs vs code (code wins)

| Topic | `docs/*.md` says | Code does |
|---|---|---|
| Yaw reference wrap | `psi_k = wrap(psi_{k-1} + g_k ψ̇ Δt)` (`reference_trajectory.md:118-128`) | No wrap: `currentYaw + psiDot*dt` (`BodyMotionReference.cpp:35`). Unwrapped yaw throughout. |
| Yaw gate | Yaw frozen in double support (`reference_trajectory.md:137-141`, mode `single_support`) | MIT yaml `always` → integrated every step; `wz_ref = ψ̇` every step. |
| Planar propagation yaw | `p_{k+1} = p_k + Rz(ψ_k) u Δt` — the *old* sample's yaw (`reference_trajectory.md:151-160`) | `p_k = p_{k−1} + Rz(ψ_k) u Δt` with `ψ_k` **already advanced** (new yaw; `advanceYaw` at `ReferenceTrajectory.cpp:43-49` before `advancePlanarPosition` at `:55-60`). Difference = one yaw step (`ψ̇Δt = 0.026 rad` at 1.3 rad/s). §4.3 follows the code. |
| Velocity reference z | `[ẋ, ẏ, ż]` rotated (`reference_trajectory.md:171-179`) | `u_des_B.z = 0` always (`ReferenceTrajectory.cpp:20-23`). |
| Horizon time base | `t_k = t_0 + kΔt`, "t0 synchronized cycle origin" (`gait_scheduler…md:42-48`) — stated but the consequence (horizon not starting at "now") is not discussed | Same formula; step 0 is the cycle start, see §1.1 / O1. |
| Planner docs line refs | `SwingFootPlanner.cpp#L235` etc. | Match the current file (`:235`, `:347`). |
| `turn_tangential_lead_scale` | yaml key present (`my_controller.yaml:52`); docs: "without the old extra tangential lead" | Key is never parsed (`ControllerConfig.cpp:260-320` has no such read) → no effect. |
| Late contact search | Fully described (`gait_scheduler…md:259-295`) | Disabled for MIT (`enable_late_contact_handling: false`). |
| Contact ramp | "softens near touchdown" | 10 ms ramp (`contact_ramp_duration 0.01`): alpha = 0 on the first active tick (no feed-forward, no yaw hold, Fz_min scale 0), then 0.2 … 0.8, and 1.0 from the 6th active tick (§6). |
| Stance-leg foot position in `r` | "current foot for stance" is *not* claimed by the docs; docs (`swing_foot_touchdown_planner.md:32-33`) say cached target, seeded from measurement only when no cache | Confirmed (§5.6): `r` uses the cached *planned* touchdown point during stance. |
| `seedTouchdownTargets` | not documented | Exists but unused by the controller. |
| Docs code refs `GaitScheduler::p #L50` etc. | stale line numbers | actual: `p :98`, `c :112`, `bothFeetStance :122`, `buildConstraintMatrices :134`. |

---

## 10. Open questions / decisions for the Python port

- **O1 Horizon origin.** `tk(k) = t0 + k·dt` with `t0` = cycle start, so the contact pattern at step k is the pattern at *cycle time* k·dt, not at *now + k·dt*, while the state reference is rolled from `x0`. Only the first `contact_lock_steps = 1` step is corrected to the measured contact. Port exactly (needed to "reproduce the reference numbers") or fix (`tk = t + k·dt`)? Recommend: port exactly first, add a flag. **Confirmed (9/25):** `tk(k) = t0 + k·0.02` (`HorizonClock.h:32-34`); nothing else consumes or shifts `tk` (only the gait constraints and the yaw gate; `MPCFormulation` uses `psi[k]` only) — see §1.1.
- **O2 `bodyComLocation` / `bodyMass` = upper body only (legs excluded), recomputed each tick from MuJoCo `xipos`.** The Python `g1_model.py` pattern may compute a different reduced body. Decide whether MIT port follows the reference (upper-body SRB, mass ≈ torso+arms) — the planner's COM, `x0` and `r` all depend on it.
- **O3 Foot site (answered 9/25).** Reference expects sites `left_foot_contact_site`/`right_foot_contact_site`, bodies `left_foot_link`/`right_foot_link` and torso body `torso`. The reference MJCF is **not public**: `reference/models/mit_humanoid/` contains only `.gitignore`/`.gitkeep`, so the reference's site pose cannot be read. `mit_humanoid_mjcf/mit_humanoid.xml` has no sites, but the project copy **`MPC/models/mit_humanoid/mit_humanoid.xml` now defines both sites** (`:53`, `:83`): `pos = (0.03, 0, −0.04)` in the foot body frame, the centre of the sole line (cylinder axis minus radius). Use that model. In it, the bodies are `left_foot`/`right_foot` and the torso body is named **`base`** (map these names in the port). With keyframe `stand` (`:173`) the site z is `−5.4e−9` m (computed; site xy = `(0.02236, ±0.08016)`), i.e. on the floor. The −0.005 target (`SwingFootPlanner.cpp:10`; standing target `My_Controller.cpp:1398`) is therefore a 5 mm commanded penetration, and the swing apex is at world z = −0.005 + 0.06 = **0.055 m**. `stance_contact_loss_foot_height (0.06)` is measured on this site z too (only used by late-contact handling, disabled for MIT). `foot_half_length 0.065`, `foot_half_width 0.01` suggest a line foot of 13 cm × 2 cm.
- **O4 `nominalHeight_W` (confirmed 9/25).** = `base_position_W.z` (yaml 0.679472, *not* the measured torso z) + `(Rz(0)·bodyComLocation).z`, with `bodyComLocation` computed on the first controller tick (≈2.0 s; `SimulationRunner.cpp:409` recomputes it every tick before prepare/run; `My_Controller.cpp:356-361, 682-696`). It does not depend on leg joints, only on arm joints and torso roll/pitch at that tick. Project model (arms `[0,0,0,−1.65]`, torso upright): `bodyComLocation = [0.01531, 0.00286, 0.11657]` m, upper-body mass 14.2275 kg ⇒ **`nominalHeight_W ≈ 0.79605 m`** (§3). Fixed for the whole run. The yaml comment lists an alternative base z `0.625972`.
- **O5 Body-target yaw drift.** `bodyTarget.euler_W[2]` (planner yaw0) is an open-loop integral of the filtered ψ̇ from t≈2 s; the MPC yaw reference uses the measured yaw. Over a 120 s turn at 1.3 rad/s the two can diverge if the body lags the command. Port as-is; note as a candidate fix. **Confirmed (9/25):** the integral starts at yaml `base_rpy_W.z = 0` (not the measured yaw), runs in Standing and Walking (`dt = 0` on the init and transition ticks), and is never re-anchored; the only exception is a turn-dominant stop (`turnStopFrameValid` ⇒ measured yaw). The foot-yaw target and the MPC yaw reference use the *measured* yaw (§3, §7.4).
- **O6 Tick rate.** Reference: controller at 500 Hz (every 0.002 s physics step), MPC at 1/7 of that (≈71 Hz), `dt_mpc = 0.02`. The filter alphas, the 5-tick stop latch, contact confirm ticks (1/4) and the 10 ms ramp are all tick-based; changing the control rate changes behaviour.
- **O7 Foot-yaw for MPC constraints** uses `touchdownYaw_W` (a commanded yaw), not the measured foot yaw, for a swing leg; for stance `liftAngleNear(measured, touchdownYaw_W)`. Keep if friction-cone rotation is ported. **Confirmed (9/25)** with details in §7.4: stance/swing by active contact (incl. early contact), measured yaw from the foot-site x-axis with fallback to `touchdownYaw_W` on throw, one value for all 25 steps, recomputed only on MPC-solve ticks.
- **O8 Search-mode / late contact / braking-to-standing / interactive toggling** are not needed for the three 120 s tests (late handling off; mode fixed to walking). Port stubs only.
- **O9 Contact force source.** Early-contact detection needs the summed MuJoCo contact normal force on the foot geoms (≥ 36 N on, ≤ 0.5 N for 4 ticks off). The converted MJCF's foot collision geoms must exist and be attached to the foot bodies for this to work.
- **O10 `swing.height` semantic.** Apex = `pFinal.z + 0.06`, independent of the lift-off height; if the swing starts from a foot that is (numerically) above/below the ground the profile is asymmetric. Fine for the port, but do not "improve" it if reproducing numbers. **Confirmed (9/25):** `zMid = pFinal.z + 0.06 = 0.055 m` (`SwingFootTrajectory.cpp:108, 120`); apex at `s = 0.5` of the trajectory's own `T`; `zMid` follows `setFinalPosition` (§7.2).

---

## Verification log (9/25)

1. §1.4 "filter smooths the staircase" — **applied** (§1.4 table + paragraph, §2, §0.6). Checked: init copies raw (`My_Controller.cpp:448, 618-621`), transition tick dt = 0 via `resetSwingState` in prepare (`:552`, `:1390`) ⇒ alpha = 1 (`:28-31`), headless default start = 1.0 s, rate 0 = step, x_dot only (`SimulationRunner.cpp:459-523`), reference built from the filtered command in both modes (`:966-989`). Refined: with a later start/ramp the transition tick still snaps to the raw value at ≈3.0 s.
2. §0.6 call order — **applied** (§0.6). `prepareController` runs `syncLocomotionFSM` before `run()` (`SimulationRunner.cpp:409-423`, `RobotRunner.cpp:73-81, 118-142`); skipped on the first controller tick. Idempotence of the 2nd call checked against `LocomotionFSM::update` (no transition ⇒ no resets); noted the BrakingToStanding double-sampling exception.
3. §7.4 touchdownYaw latch — **applied** (§7.4, §1.4). Init at `:450-457`, untouched by `resetSwingState` (`:534-553`), overwritten on every `traj.reset` (`:852-858`).
4. §7.3 re-reset / bounce — **applied with correction** (§7.3, §6). Reset tick outputs p = pInit and v = 0 as claimed, but **a is not 0**: `smoothBlendDdot(0) = 6` ⇒ a = 6·d/T² (XY), 6·dz·(2/T)² (Z) (`SwingFootTrajectory.cpp:6-21, 79-139`), a large spike at T = 1 ms; `aDes` is used in `OperationalSpaceDynamics.cpp:78`. Bounce path confirmed (`ContactManager.cpp:266-274, 327-338`, `My_Controller.cpp:852-862`).
5. §1.2/§1.3 float boundaries — **applied** (§1.1, §1.2). Re-computed: t0 = 3.0 / 2.999999999999891 → p_L = 0.6600000000000001 (swing); 3.999999999999891 → 0.6599999999999993 (stance); 1500 accumulated steps give 2.999999999999891.
6. §9 planar propagation yaw row — **applied** (§9). `docs/reference_trajectory.md:151-160` vs `ReferenceTrajectory.cpp:42-61`.
7. §0.5 parser defaults — **applied** (§0.5). All values checked in `ControllerConfig.h:31-107`, aliases in `ControllerConfig.cpp:176-197, 285-292`.
8. §1.3 swing "handled by D" — **applied** (§1.3). `C_bound` zero for swing, rows `C·u ≤ bound` with lower −inf (`ConvexMPC.cpp:886-887`, `GaitScheduler.cpp:83-96, 154-177`) ⇒ full zero wrench.
9. Contact ramp tick count — **applied with refinement** (§6, §9). alpha 0 on the first active tick confirmed (`ContactManager.cpp:340-353`). "dt 0 on the tick after reset" is corrected to "the first update after reset (same control tick)". A float check shows the 6th tick is often 0.99999… (5358 of 8215 sampled windows), so it must be computed literally.
10. §0.2 roll notation — **applied** (§0.2). `My_Controller.cpp:63-75`.
11. O1 — **applied** (§1.1, §10 O1). Consumers of tk checked by grep (`GaitScheduler.cpp:153`, `ReferenceTrajectory.cpp:40-41, 63`, `MPCFormulation.cpp:64`). The finding's example (0.30 s → 0.02 s vs 0.32 s) was replaced, because both instants are double support; used 0.10 s (Left swing) → 0.02 s (DS) instead.
12. O3 — **applied** (§10 O3, §0.3). Reference model dir holds only `.gitignore`/`.gitkeep`; project model sites at `:53, :83`; site z at `stand` recomputed = −5.4e−9 m. Added that the torso body is named `base` in the project model.
13. O4 — **applied** (§3, §10 O4). Recomputed with MuJoCo on `MPC/models/mit_humanoid/scene.xml`: bodyComLocation [0.01531, 0.00286, 0.11657], M 14.2275 kg, nominalHeight 0.79605 m (same for `stand` and `init_yaml` keyframes).
14. O5 — **applied** (§3, §10 O5). `My_Controller.cpp:471, 571, 682-696, 716-734`; `SwingFootPlanner.cpp:171-173, 244`.
15. O7 — **applied** (§7.4, §10 O7). `My_Controller.cpp:165-176, 363-377, 930-963`; `GaitScheduler.cpp:25-30, 146-147, 187-188`.
16. O10 — **applied** (§7.2, §10 O10). `SwingFootTrajectory.cpp:102-126`.
17. Target frozen mid-swing — **applied** (§5.6). `SwingFootPlanner.cpp:334-359`; ContactManager override `:394-414`.

## Confirmed by verifier

- HorizonClock: reset(t0), sync while(t−t0>=cycle) t0+=cycle (never backwards), tk=t0+k·dtMpc — HorizonClock.h:14-34
- dtMpc = horizon/horizon_steps = 0.5/25 = 0.02; cycle=swing+stance check — ControllerConfig.cpp:494, 645-647
- Gait phase p = fmod((t−t0)/cycle + phi, 1) with phi=0.5 for Left, 0 for Right; p=0 and c=true in Standing; c = 0<=p<stance/cycle (0.66) — GaitScheduler.cpp:98-120
- Phase table (DS [0,0.08), Left swing [0.08,0.25), DS [0.25,0.33), Right swing [0.33,0.5)) and swing length 0.17 s
- remainingSwingTime = clamp(cycle·(1−p), 0, swing) — My_Controller.cpp:59-61, SwingFootPlanner.cpp:247-251
- lowPassBlendAlpha = clamp(−expm1(−dt/tau),0,1), 1 if tau<=0 or dt<=0; alphas 2.1716e-3/2.4969e-3/2.8531e-3 at 2 ms
- Filter: zeroMotion zeroing, exact-zero snap on raw, body_height_offset unfiltered, roll/pitch offsets filtered, clamp before and after — My_Controller.cpp:610-654
- Body target init from yaml base pose, Standing xy = mean foot xy, Walking xy = x0 xy, yaw advanceYaw every tick with 'always', applyPoseOffsets — My_Controller.cpp:677-736
- Reference seed: Standing overrides euler/pos/height; Walking overrides only roll, pitch, seed[5]=nominalHeight — My_Controller.cpp:966-978
- ReferenceTrajectory.build: p_ref=seed[3:6] with z+=h, yaw advanced before position, v_ref=Rz(psi_k)[x_dot,y_dot,0], z reset to seed[5]+h each k>0, r = foot_des − p_ref (same foot for all k), X_ref[6:8]=0, X_ref[8]=psi_dot (always), X_ref[12]=seed[12], yaw sample time tk−0.5dt for k>0 — ReferenceTrajectory.cpp:8-83
- advanceYaw has no wrap; advancePlanarPosition/worldVelocity = Rz(yaw)·cmd — BodyMotionReference.cpp:29-43
- yaw_integration_mode parsing and default SingleSupport; MIT 'always' — ControllerConfig.cpp:176-197, yaml:85
- Planner helpers: strict '>' speed switches (0.37/0.28/0.26), Tp=(0.5+k)·0.33 → 0.2871/0.2574/0.2508 s; at exactly 0.6 m/s k=0.37 (norm is exactly 0.6 in double, filter approaches from below)
- stopRecenterRequested/Active latch (5 extra ticks) and computeStopStanceCenterWorld capture-point offset with gain 0.2, max 0.08 — SwingFootPlanner.cpp:180-233
- Touchdown formula incl. yawTrans = yaw0+½ψ̇Tp, yawTd = yaw0+ψ̇·Trem, planned_B, stop override, lateral crossing guard, target.z=−0.005 — SwingFootPlanner.cpp:235-275
- Turn-stop bookkeeping and end-of-call previous-command latches — SwingFootPlanner.cpp:283-320, 365-368
- Planner stance seeding from measured footPos only when target invalid (after reset); ensureSwingTouchdownCache sets wasInStance=true — SwingFootPlanner.cpp:88-107, 336-343
- nominal_foot_offsets_B from yaml [0,±0.075999602,0], fallback inference zeroes x and z — SwingFootPlanner.cpp:109-147
- turn_tangential_lead_scale is never parsed — ControllerConfig.cpp:255-320
- seedTouchdownTargets is not called by the controller
- Swing trajectory: cubic smoothstep XY over whole swing, Z two-half smoothstep with zMid=pFinal.z+H, ds=1/T, zero end velocities, end accelerations ±6d/T² — SwingFootTrajectory.cpp:79-139
- Swing driver: stance → deactivate; reset(pFoot, target, 0.06, max(Trem,0.001)) when wasInStance or !active, else setFinalPosition+advance(dt); lastControlTime=t at end — My_Controller.cpp:792-868
- Swing yaw: base = measured yaw_W_unwrapped + 1.0·ψ̇·Tp; bias 100°/(rad/s)·|ψ̇| clamped at 20°, +Left for ψ̇>0, −Right for ψ̇<0; liftAngleNear(fallback+bias, fallback) — SwingYawTarget.h:11-34, My_Controller.cpp:891-909
- Stance wrench mapping −alpha·F (segments 0/3 forces, 6/9 moments) and yaw hold alpha·(20·err − 4·ω_z) only when enable && alpha>0 — My_Controller.cpp:1315-1348
- ContactManager hysteresis (on ≥36 N ×1 tick, off ≤0.5 N ×4 ticks), releasedContactDuringSwing, earlyContact, activeContact, managedFootPositions override only for search/early/liftoffHold, reset() activeContact=scheduled — ContactManager.cpp:89-414
- Horizon override for contact_lock_steps=1 with scale = activeContact?alpha:0; stance bounds Fz≤max, Fz≥scale·min — ContactManager.cpp:416-446, GaitScheduler.cpp:159-177
- MPC resolves when iteration==0 or iteration−last>=7 — My_Controller.cpp:918-921
- Standing nominal foot targets = measured foot pos with z=−0.005 — My_Controller.cpp:1396-1405
- StandingSettle→Walking after 1.0 s with resetGaitClock/resetSwingState; seedBodyTargetFromCurrentState commented out — LocomotionFSM.cpp:230-234, 275-289; My_Controller.cpp:555-577
- Reduced-body COM helpers and bodyComLocation/bodyInertia (upper body only, yaw-aligned frame) — SwingFootPlanner.cpp:12-26, setupRobotParams.cpp:397-488
- yaml numeric values listed in §0.5 match my_controller.yaml lines 1-138 and simulation.yaml:1
