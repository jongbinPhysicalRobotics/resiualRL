# Reference convex MPC — exact specification for a Python re-implementation (MIT Humanoid)

Source repo: `C:\Users\백종빈\Desktop\4-2\residual RL\reference` (ispaik06/convex-mpc-biped).
All `file:line` references below are relative to that folder. **When docs and code differ, the code wins**; discrepancies are listed in §14.

Abbreviations used throughout:

| symbol | file |
|---|---|
| `MPCF` | `My_Controller/src/MPCFormulation.cpp` |
| `CMPC` | `My_Controller/src/ConvexMPC.cpp` |
| `GS` | `My_Controller/src/GaitScheduler.cpp` |
| `RT` | `My_Controller/src/ReferenceTrajectory.cpp` |
| `BMR` | `My_Controller/src/BodyMotionReference.cpp` |
| `MC` | `My_Controller/src/My_Controller.cpp` |
| `CC` | `My_Controller/src/ControllerConfig.cpp` / `.h` = `My_Controller/include/MyController/ControllerConfig.h` |
| `HC` | `My_Controller/include/MyController/HorizonClock.h` |
| `SRP` | `sim/src/setupRobotParams.cpp` |
| `CSR` | `sim/src/MujocoCheaterStateReader.cpp` |
| `CM` | `My_Controller/src/ContactManager.cpp` |
| `SFP` | `My_Controller/src/SwingFootPlanner.cpp` |
| `MU` | `common/include/Utilities/MatrixUtils.h` |
| `AU` | `common/include/Utilities/AngleUtils.h` |
| `YAML` | `config/mit_humanoid/my_controller.yaml` |

---

## 0. One-paragraph summary

Every physics step (500 Hz, `config/simulation.yaml:1` → `physics_timestep_sec: 0.002`) the controller runs; every `iterations_between_solve = 7` ticks (`YAML:24`, i.e. every 14 ms ≈ 71 Hz — **not** the 50 Hz / 10 ticks the README claims) it rebuilds and solves one QP over a horizon of `N = 25` steps × `dt = 0.02 s` (`YAML:7-8`). Decision variables are the stacked per-step ground-reaction wrenches `u_k = [F_L, F_R, M_L, M_R] ∈ R^12` (world frame, moments about each foot's end-effector site), `U ∈ R^{300}`. Dynamics: a single-rigid-body (SRB) model of *torso + arms only* (legs excluded), linearised with `Rz(psi_k)` from the reference yaw, discretised by a truncated matrix exponential (ZOH with quadratic/cubic terms), lifted to `X = A_qp x0 + B_qp U`. Cost: `Σ_k (x_k − x_ref,k)^T Q_k (x_k − x_ref,k) + u_k^T R u_k` with `Q_k = T_k^T Q T_k` (planar blocks rotated into the reference-yaw frame), no terminal weight. Constraints per step per stance foot: 12 yaw-rotated inequality rows (friction pyramid µ = 1.0, `5 ≤ Fz ≤ 1500`, CoP box `0.065 × 0.01`, torsional `|Mz| ≤ 0.0657·µ·Fz`); swing feet get 6 equality rows `u = 0` (variables are **kept**, not removed). OSQP (via osqp-eigen), max_iter 200, no polish, adaptive rho, shifted warm start. Output = first 12 entries `u_0`; the leg controller applies `τ = Jv^T(−α F) + Jw^T(−α M)` with the contact-ramp `α`.

---

## 1. Timing, cadence, horizon clock

### 1.1 Config values (`YAML:3-8`, `YAML:16-24`, `CC.h:31-37`, `CC.h:46-61`)

| key | MIT value | meaning |
|---|---|---|
| `timing.cycle` | 0.5 s | gait cycle `T_c` |
| `timing.swing` | 0.17 s | `T_sw` |
| `timing.stance` | 0.33 s | `T_st` (check `swing+stance == cycle` within 1e-9, `CC.cpp:494-496`) |
| `timing.horizon` | 0.5 s | `T_h` |
| `timing.horizon_steps` | 25 | `N` |
| `dtMpc()` | `T_h / N = 0.02 s` | `CC.cpp:645-647` |
| `mpc.iterations_between_solve` | 7 | control ticks between QP solves (`CC.cpp:240`, `MC:438`, `MC:918-919`) |
| physics / control tick | 0.002 s | `config/simulation.yaml:1`; controller runs **every** `mj_step` (`sim/src/SimulationRunner.cpp:337-343`) |

"Iterations" = calls of `MyController::runController()` = physics steps at 500 Hz (`MC:1428` increments `_iteration` once per tick). The QP is solved on tick 0 and whenever `_iteration − _lastMpcIteration ≥ 7` (`MC:918-919`), i.e. every 7th tick = **14 ms**. Between solves the last `u_0` is held and re-applied every tick (`MC:1010`, `MC:1315-1349`).

Counter details (O4): `_iteration` and `_lastMpcIteration` are set to 0 at controller init (`MC:460-461`), which happens on the first controller tick after the 2 s leg-PD initialisation (`robot/src/RobotRunner.cpp` `run()` only calls the controller once leg init is complete). The counter is **not reset** at mode transitions (Standing→Walking), so solve ticks are **not aligned to `t0`**. `_lastMpcIteration = _iteration` is executed after the try/catch (`MC:1054`), i.e. **also when the solve throws** — the fallback wrench (§10.3) is then held for 7 ticks. Config: `iterationsBetweenSolve` default 10 if the yaml key is absent (`CC.h:58`), clamped to ≥ 1 (`MC:438`). The 0.002 s tick is forced into `model->opt.timestep` (`sim/src/SimulationConfig.cpp:102`).

> README (`README.md:404`) says "`iterations_between_solve = 10` → 50 Hz". The checked-in YAML says 7. **Code loads YAML → 7 is what runs.** (The G1/H1 yamls were not checked; MIT is 7.)

### 1.2 HorizonClock (`HC:18-34`)

```
sync(t):   while (t - t0 >= cycle): t0 += cycle        # HC:23-25  (t0 is the start of the CURRENT gait cycle)
tk(k)  =   t0 + k * dtMpc()                              # HC:32-34
```

`t0` is initialised to the controller start time (`MC:411`) and reset to the current time whenever the locomotion FSM asks for a gait-clock reset (`MC:564-566`, e.g. standing→walking transition). `sync` is called every tick (`MC:1387`, also `SFP:85`).

**Critical quirk — the horizon grid is anchored at the cycle origin, not at "now".** `tk(k) = t0 + k·dt` with `t0 ≤ t < t0 + 0.5`. Every consumer of the horizon time (`GS:153`, `RT:40`) evaluates step `k` at `t0 + k·dt`, **not** at `t + k·dt`. Because `T_h = T_c`, the 25 samples always cover exactly one full cycle, but rotated relative to the current time by `(t − t0) ∈ [0, 0.5)`. Only step `k=0` is replaced, by the contact manager's `activeContact` evaluated at the **current** time `t` (§5.4, `contact_lock_steps = 1`) — for MIT this is the nominal schedule `c(side, t)` plus early-contact forcing, **not** a force-threshold contact detector. The docs (`docs/gait_scheduler_and_contact_management.md:42-48`) describe the same formula, so docs and code agree — but this is a genuine modelling choice you must copy to reproduce the reference. Steps 1..24 are "the same every solve" only up to a floating-point knife edge at left `k=4` (§5.2). (O1 resolved, §16.)

---

## 2. Frames, reduced-body (SRB) parameters

### 2.1 Frames

| frame | definition |
|---|---|
| `W` | MuJoCo world frame, z up, gravity `−9.81` on z (`YAML:14`) |
| `T` | torso body frame (`MitHumanoidSpec.cpp:6`: base body named `"torso"`) |
| `B` (yaw-aligned / "reduced-body") | origin at reduced-body COM, axes = `Rz(psi)` of world (yaw only, no roll/pitch). `psi` = torso yaw `atan2(R_WT(1,0), R_WT(0,0))` (`SRP:328-330`, `StateEstimator.cpp:30-31`) |
| `F` (foot frame) | foot end-effector **site** frame `left_foot_contact_site` / `right_foot_contact_site` (`MitHumanoidSpec.cpp:10,19`; `YAML:13 foot_end_effector_source: site`), `R_WF = site_xmat` (`CSR:292-305`), position = `site_xpos` (`CSR:121-131, 356`). Foot yaw `psi_f = atan2(R_WF(1,0), R_WF(0,0))` (`MC:165-176`) |

Rotation / skew helpers (`MU:11-30`):

```
Rz(psi) = [[c, -s, 0],
           [s,  c, 0],
           [0,  0, 1]]                      c = cos psi, s = sin psi
skew(v) = [[   0, -v.z,  v.y],
           [ v.z,    0, -v.x],
           [-v.y,  v.x,    0]]              skew(v) w = v × w
```

### 2.2 Reduced-body mass properties (recomputed EVERY control tick)

`RobotParams` fields (`common/include/Robot/RobotParams.h:77-79`): `bodyMass`, `bodyInertia` (3×3, frame B, about reduced COM), `bodyComLocation` (torso-root → reduced COM, expressed in B axes).

Set once at start-up from `q = 0` (`SRP:212-325`), then **overwritten every tick** by `updateReducedBodyMassPropertiesFromData` (`sim/src/SimulationRunner.cpp:409`, called before `fillCheaterState` on every tick; implementation `SRP:397-491`):

```
bodies considered: every body b ≥ 1 with mass > 0 that is NOT in a leg subtree
                   (leg subtree root = body of the first leg joint, i.e. hip_yaw link; SRP:165-173, 183)
                   → torso + both arms (+ hands). Legs are excluded from the SRB.
m_B      = Σ_b m_b                                                        (SRP:429)
c_W      = Σ_b m_b xipos_b / m_B                                          (SRP:430-436)   xipos = MuJoCo body COM in world
I_W      = Σ_b [ ximat_b diag(body_inertia_b) ximat_b^T
                 + m_b (|o_b|^2 I3 − o_b o_b^T) ],  o_b = xipos_b − c_W   (SRP:449-473)
psi      = atan2(xmat_torso(1,0), xmat_torso(0,0))                        (SRP:484)
bodyMass        = m_B
bodyInertia     = Rz(psi)^T I_W Rz(psi)              (frame B)            (SRP:489)
bodyComLocation = Rz(psi)^T (c_W − xpos_torso)       (frame B)            (SRP:490)
```

For the local MIT MJCF (`mit_humanoid_mjcf/mit_humanoid.xml`, 24.89 kg total) this gives `bodyMass = 14.227 kg`; with the yaml initial pose (`YAML:137-138`) applied: `bodyComLocation ≈ [0.0153, 0.0029, 0.1166]`, `bodyInertia ≈ diag(0.519, 0.168, 0.403)` with small off-diagonals (see §13). *These numbers depend on arm pose and are re-evaluated each tick — the Python port must do the same (or accept a deviation).*

O2 (resolved): `updateReducedBodyMassPropertiesFromData` runs every 2 ms tick **before** `fillCheaterState` (`sim/src/SimulationRunner.cpp:409-411`), and `MPCFormulation` holds a pointer to the same `RobotParams` (`MC:420`), so every solve and every `x0` use the current tick's `m`, `I_B` (torso-yaw frame) and `bodyComLocation` (torso + arms only, from `xipos/ximat/body_inertia`). **Exception:** the walking z-reference `nominalHeight_W = base_position_W.z + (Rz(base_euler.z)·bodyComLocation).z` is evaluated **once**, on the first controller tick after leg init, with that tick's `bodyComLocation`, and then held forever (`MC:682-697`; re-seeding at transitions is commented out, `MC:571`). See §6.1.

### 2.3 Reduced-body COM & velocity used for x0 (`MC:315-332`)

```
offset_W  = Rz(yaw_unwrapped) * bodyComLocation                     (MC:317)
com_W     = torsoPos_W + offset_W                                   (MC:322)
comVel_W  = torsoLinVel_W + torsoAngVel_W × offset_W                (MC:331)   full torso ω, not yaw-only
```

`torsoPos_W = xpos[torso]` (`CSR:339`), `torsoQuat_W = xquat[torso]` (`CSR:340`), `torsoLinVel_W`, `torsoAngVel_W` from `mj_objectVelocity(mjOBJ_BODY, torso, flg_local=0)` → **world-frame** angular/linear velocity of the torso body-frame origin (`CSR:133-164`).

**One-step kinematic lag (fidelity).** The runner calls `mj_forward` only once at load (`SimulationRunner.cpp:269`); inside the loop it calls `runRobotControl()` right after the previous `mj_step` (`SimulationRunner.cpp:337-343`). So `xpos, xquat, xipos, ximat, site_xpos, site_xmat, cvel` (hence `mj_objectVelocity`) and contact forces describe the **pre-integration state of the previous step**, while `data->time` and leg `q/qd` read from `qpos/qvel` are current (`CSR:338-360`). Consequently `x0`, the reduced-body properties (§2.2), foot site position/yaw and contact normal forces are one physics step (2 ms) stale. Leg Jacobians `Jv_W, Jw_W` are computed on separate auxiliary per-leg models by `mj_forward` with the torso pose taken from the (stale) `xpos/xquat` and the **current** leg `q, qd` (`sim/src/LegSwingDynamicsProvider.cpp:603-640`). To reproduce: read body/site kinematics and velocities from `mjData` exactly as `mj_step` left them — do **not** call `mj_forward` before building `x0`; `t = data.time`; leg `q/qd` from `qpos/qvel`.

---

## 3. State vector `x ∈ R^13` (`MC:656-675`, `docs/mpc_frame_convention.md:14-41`)

| idx | symbol | value in `x0` | frame |
|---|---|---|---|
| 0 | φ roll | `atan2(2(wx+yz), 1−2(x²+y²))` of `torsoQuat_W` (`MC:63-75`) | Euler ZYX of torso |
| 1 | θ pitch | `asin(clamp(2(wy−zx),−1,1))` | |
| 2 | ψ yaw | `yaw_W_unwrapped` (**continuous, unwrapped**; `StateEstimator.cpp:29-47`, `AU:10-14`) | W |
| 3-5 | p | reduced-body COM position `com_W` | W |
| 6-8 | ω | `torsoAngVel_W` (torso angular velocity, world frame) | W |
| 9-11 | v | `comVel_W` | W |
| 12 | g | `model.gravity = −9.81` (`YAML:14`, `MC:673`) — constant slot | — |

Yaw unwrap (`StateEstimator.cpp:39-40`, `AU:10-14`): `yaw_u[k] = yaw_u[k−1] + wrapToPi(yaw_w[k] − yaw_w[k−1])`, `wrapToPi(a)=atan2(sin a, cos a)`. First sample: `yaw_u = yaw_w`.

Yaw policy actually used (`docs/yaw_wrapped_unwrapped_policy_for_mpc.md` §19 is implemented): `x0[2]` = unwrapped; `X_ref` yaw = unwrapped seed + integrated `psi_dot·dt` (never wrapped, `RT:43-49`); `Rz(psi_k)` in dynamics uses that unwrapped `psi_k` (`MPCF:64-68`); constraint rotation uses stance-foot yaw lifted near the touchdown yaw (`MC:938-940`, `AU:16-18`): `liftAngleNear(target_w, cur_u) = cur_u + wrapToPi(target_w − cur_u)`.

---

## 4. Input vector `u_k ∈ R^12` (`MPCF:61-62`, `CMPC:184-193`)

```
u_k = [ F_L(0:3), F_R(3:6), M_L(6:9), M_R(9:12) ]
      = [Fx_L, Fy_L, Fz_L, Fx_R, Fy_R, Fz_R, Mx_L, My_L, Mz_L, Mx_R, My_R, Mz_R]
leftMap  (6-D foot wrench [Fx,Fy,Fz,Mx,My,Mz] → u index) = [0,1,2,6,7,8]      CMPC:192
rightMap                                                  = [3,4,5,9,10,11]    CMPC:193
```

* Frame: **world** (`W`) for both forces and moments.
* Sign: `u` is the **ground reaction wrench acting on the robot** (MPC solves for GRF; `MC:1319` comment). The leg command negates it (§11).
* Moment reference point: the foot end-effector (site) — the lever-arm `r × F` about the COM is added inside `B_c` (`MPCF:78-79`), so `M` is the *pure contact moment*, exactly what the CoP constraint needs (`docs/friction_cop…md:1358-1385`).
* Stacked: `U = [u_0; …; u_{N−1}] ∈ R^{12N} = R^{300}`.

> Differs from our G1 code (`[F_L, m_L, F_R, m_R]`). Keep the reference order **inside** the port, or add an explicit permutation at the boundary and re-order `R`, constraints and the warm-start shift accordingly.

---

## 5. Gait schedule and contact override

### 5.1 Phase and stance (`GS:98-120`)

```
phi_L = 0.5, phi_R = 0.0                                       (GS:107-108)  ← right foot leads (phase 0 at t0)
p(side, t) = fmod((t − t0)/cycle + phi_side, 1.0)              (GS:109)
c(side, t) = (0 ≤ p < stance/cycle)      stance/cycle = 0.66   (GS:117-119)
bothFeetStance(t) = c(L,t) && c(R,t)                            (GS:122-124)
Standing mode: p ≡ 0, c ≡ true                                  (GS:99-101, 113-115)
```

Right foot: stance for `(t−t0) mod 0.5 ∈ [0, 0.33)`, swing `[0.33, 0.5)`.
Left foot: stance for `(t−t0) mod 0.5 ∈ [0, 0.08) ∪ [0.25, 0.5)`, swing `[0.08, 0.25)`.
Double support: `[0, 0.08)` and `[0.25, 0.33)` → `2·T_st − T_c = 0.16 s` per cycle (32 %). Swing = 0.17 s each.

Remaining swing time used by the swing planner (not the MPC): `clamp(cycle·(1 − p(side,t)), 0, swing)` (`MC:59-61`).

### 5.2 Horizon schedule (`GS:134-193`)

For `k = 0..N−1`: `tk = t0 + k·dt` (**cycle-anchored**, §1.2), `leftStance = c(L, tk)`, `rightStance = c(R, tk)`, `minScale_L = minScale_R = 1`. With `t0` as origin (schedule is the same every solve, independent of `t` — except the knife edge below):

```
k :   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
L :   1  1  1  1  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1
R :   1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  0  0  0  0  0  0  0  0
```

(Step 0 of this table is then replaced by the contact override, §5.4.)

**Floating-point knife edge at left `k=4`.** There `p_L = fmod(((t0+4·0.02) − t0)/0.5 + 0.5, 1)` is nominally exactly `0.66 = stance/cycle`, so the result depends on the rounding of `(t0 + 0.08) − t0`, i.e. on the magnitude/bits of `t0`. Verified numerically (Python doubles, same expression order as `GS:109,117-119`, `HC:32-34`): with `t` accumulated as `+= 0.002` per tick, `t0` reset at ≈3.002 s and advanced by `+= 0.5` in `sync`, left `k=4` is **swing** for `t0 < 16.002 s` and **stance** from `t0 = 16.002 s` on (then L stance k=0..4, 13..24). Other `t0` values (e.g. 7.998, 100.25) also give stance. When it flips, the contact signature changes → one extra cold OSQP init (§9.6). **Do not hard-code this table:** evaluate `c(side, t0 + k·dtMpc)` every solve with the identical double expressions — `dtMpc = horizon/steps` (0.5/25), `tk = t0 + k·dtMpc`, `p = fmod((tk − t0)/cycle + phi, 1.0)`, `stanceFraction = stance/cycle` computed, not a literal — and use MuJoCo's own `data.time` so `t0` has the same bits.

### 5.3 Per-step bound vector `C_bound ∈ R^{24N}` (`GS:149-180`)

`Ck_bound = 0_24`, then if `leftStance`: `Ck_bound[4] = normalForceMax (1500)`, `Ck_bound[5] = −minScale_L · normalForceMin (5)`; if `rightStance`: `Ck_bound[16] = 1500`, `Ck_bound[17] = −minScale_R · 5`. Swing feet keep all-zero bounds (their C rows are also zero → `0 ≤ 0`).

Also stored per step (`GaitConstraintStep`, `GaitScheduler.h:11-20`): stance flags, min-scales, `leftFootYaw_W`, `rightFootYaw_W` (a single yaw per foot for the *whole* horizon), `leftFootXAxis_W`, `rightFootXAxis_W` (only for NoRollMoment).

### 5.4 Contact-manager override (walking only) (`MC:953-956`, `CM:416-446`, `GS:159-167`)

`buildHorizonOverride()` returns `contact_lock_steps` (= **1**, `YAML:93`) enabled steps, all identical:
`leftContact = activeContact_L`, `rightContact = activeContact_R`, `minScale_side = activeContact ? contactRampAlpha_side : 0`. In `GS:159-167` these replace step `k < 1` (i.e. only `k=0`) of the nominal schedule (scale clamped to [0,1]). So **step 0 = `activeContact` at the current time `t`**; steps 1–24 follow the cycle-anchored nominal table (§5.2).

**`activeContact` is NOT a force-threshold contact detector for MIT.** `ContactManager::update` (`CM:200-357`, called every tick in walking before the swing update and the MPC) sets `scheduledContact = gaitScheduler.c(side, t_now)` (`CM:222`, the stance flag at the **current** time, same `t0`) and, with MIT's flags `enable_early_contact_handling: true`, `enable_late_contact_handling: false` (`YAML:99-100`):

```
estimatedContact : debounced force flag — becomes true after normalForce ≥ 36 N for 1 tick (on_confirm 1),
                   false after normalForce ≤ 0.5 N for 4 consecutive ticks (off_confirm 4)       (CM:89-147, YAML:88-91)
                   normalForce = MuJoCo contact normal force on the foot (stale by one step, §2.3)
releasedContactDuringSwing: false while scheduledContact; set true on a scheduled-swing tick with !estimatedContact (CM:266-270)
earlyContact  = !scheduledContact && estimatedContact && releasedContactDuringSwing                       (CM:272-274)
lateContact   = false (disabled), liftoffHold = false                                                     (CM:284-290)
activeContact = earlyContact ? true : scheduledContact                                                    (CM:327-338)
```

So `activeContact = c(side, t_now) OR earlyContact`; a missing touchdown never delays stance. Consequences: step 0 is the nominal schedule at `t`, and the contact signature (§9.6) flips at every `c(side, t_now)` transition.

`contactRampAlpha` (`CM:340-353`): when `activeContact` becomes true (previous tick false) `rampTime = 0`; on later active ticks `rampTime += dt` (dt = time since last update = 0.002); `α = clamp(rampTime / 0.01, 0, 1)` (`contact_ramp_duration`, `YAML:92`) → α = 0, 0.2, 0.4, 0.6, 0.8, 1.0 on the onset tick and the next 5 ticks. On the **onset tick α = 0 exactly**: if a solve happens then, the step-0 Fz lower bound is 0 and that leg's stance wrench is scaled by 0 (§11). Inactive: `rampTime = 0, α = 0`. At `reset()` (every Standing→Walking transition, via `resetSwingState`, `MC:549-552`) and at first initialisation: `activeContact = scheduledContact` (Standing: true), `rampTime = 0.01` (α = 1) if active, else 0 (`CM:150-197`, `:231-240`). In Standing mode: `activeContact ≡ true`, `α ≡ 1` (`CM:245-262`), and `activeContactForSide` uses the scheduler (`c ≡ true`) (`MC:738-746`).

### 5.5 Foot yaw used for constraint rotation (`MC:930-946`, `GS:25-30`)

```
if activeContact(side):      # walking: ContactManager (§5.4); standing: c ≡ true
    try:  yaw = liftAngleNear(atan2(R_WF(1,0), R_WF(0,0)), touchdownYaw_W[side])
              = touchdownYaw_W[side] + wrapToPi(measured − touchdownYaw_W[side])   (measured foot SITE xmat yaw)
    except (planar norm of site x-axis ≤ 1e-9 → footYawFromXAxisWorld throws): yaw = touchdownYaw_W[side]
else:
    yaw = touchdownYaw_W[side]
```

`touchdownYaw_W[side]` is a **per-leg member, latched once per swing**, not recomputed every tick (O8):

* latched in `updateSwingTrajectories` (`MC:852-854`) on a tick where the leg is **not** `activeContact` and (`wasInStance` or the swing trajectory is inactive) — i.e. the first non-contact tick of each swing — and on entering search mode (`MC:830-832`, never for MIT since late contact is off). This runs **before** the MPC on the same tick (`MC:1418-1421`). It is held through the rest of the swing and the following stance.
* also initialised at controller init (`MC:455-456`) with the same formula; `resetSwingState` does not change it.
* formula (`MC:891-909`, `common/include/Dynamics/SwingYawTarget.h:11-34`), all with the **filtered** command (§6.2) of that tick:

```
off(|v|)   = 0.26 if |v| > 0.65 ; 0.28 if |v| > 0.60 ; else 0.37      |v| = ‖(x_dot, y_dot)‖ (YAML:45-49)
fallback   = yaw_W_unwrapped + swing_foot_yaw_lead_scale(1.0) · psi_dot · max(0, (0.5 + off)·T_stance)   T_stance = 0.33
bias       = +min(100·|psi_dot|, 20)° for LEFT  only if psi_dot > 0
             −min(100·|psi_dot|, 20)° for RIGHT only if psi_dot < 0 ;  else 0        (1.3 rad/s → 20° = 0.349 rad)
touchdownYaw_W = liftAngleNear(fallback + bias, fallback) = fallback + bias     (|bias| ≤ 20°)
```

Passed as `leftFootYaw_W`, `rightFootYaw_W` to `buildConstraintMatrices`; if NaN it falls back to yaw of the stored foot x-axis (`GS:25-30`). One yaw per foot for all 25 steps. The same `touchdownYaw_W` is the target of the stance yaw hold (§11).

---

## 6. Reference trajectory `X_ref ∈ R^{13N}`, `psi_k`, `r_left/r_right` (`RT:8-83`, `MC:966-989`)

### 6.1 Seed (`MC:966-978`)

```
seed = x0
Walking:  seed[0] = bodyTarget.euler_W[0]  (roll target, = seed roll 0 + standing_roll_offset;  MC:704-707)
          seed[1] = bodyTarget.euler_W[1]  (pitch target)
          seed[5] = bodyTarget.nominalHeight_W   (COM height target = height of reduced COM at initial pose:
                     base_position_W[2] + (Rz(0)·bodyComLocation)[2], MC:684-689, 356-361; YAML:133 → 0.679472 + com_z;
                     LATCHED ONCE on the first controller tick after leg init with that tick's bodyComLocation, never re-seeded — §2.2)
          seed[2:5 other], velocities untouched → yaw, x, y anchored to x0 (receding)
Standing: seed[0:3] = bodyTarget.euler_W ; seed[3:6] = bodyTarget.nominalPosition_W ; seed[5] = nominalHeight_W
```

`bodyTarget.euler_W[2]` (yaw target) is integrated every tick from `psi_dot` but **is not used by the walking reference** (walking uses `x0` yaw); it feeds the swing planner only.

### 6.2 Command (`MC:610-654`, `RT:16-23`)

`u_des_B = [x_dot, y_dot, 0]`, `psi_dot_des`, `body_height_offset_m` from the **filtered** user command (`updateFilteredUserCommand`, `MC:610-654`, run every tick before the body-target update):

```
raw = clamp(user command)          |x_dot| ≤ 0.7, |y_dot| ≤ 0.5, |psi_dot| ≤ 2.0   (YAML:79-81, CC.cpp:649-656)
if zeroMotionCommand (FSM state BrakingToStanding): raw.x_dot = raw.y_dot = raw.psi_dot = 0      (MC:613-617)
if first controller tick: filtered = raw ; return                                                (MC:618-622)
if raw.x_dot == raw.y_dot == raw.psi_dot == 0: filtered x_dot = y_dot = psi_dot = 0 exactly      (MC:628-632)
else: for each of x_dot, y_dot, psi_dot:  f += α(τ, dt)·(raw − f)   τ_x = 0.92, τ_y = 0.8, τ_psi = 0.70  (YAML:74-76)
filtered.body_height_offset_m = raw.body_height_offset_m        (NOT filtered, MC:643)
standing roll/pitch offsets: low-passed with τ = 0.70 (YAML:77-78)
filtered = clamp(filtered)                                       (clamped again after filtering)

α(τ, dt) = 1 − exp(−dt/τ), but α = 1 if dt ≤ 0 or τ ≤ 0        (MC:28-33)
dt = max(0, t − _lastControlTime)                                (MC:1390)
```

`_lastControlTime` is set to `t` at the end of `updateSwingTrajectories` every tick (`MC:867`), at controller init (`MC:470`) **and in `resetSwingState()` (`MC:552`)**, which the FSM triggers on every transition except into BrakingToStanding (`LocomotionFSM.cpp:280-281`, `MC:567-569`) — in particular on **Standing→Walking**. That reset happens in `syncLocomotionFSM()` before `dt` is computed on the same tick, so on the transition tick `dt = 0 → α = 1` and filtered `x_dot/y_dot/psi_dot` **jump to the raw clamped command** (no τ ramp). If the demo command is already non-zero when walking starts, the reference gets a step. (The body-target yaw integration on that tick also sees `dt = 0`.)

### 6.3 Loop (`RT:39-82`) — with `yaw_integration_mode: always` (`YAML:85`, `BMR:12-24` → `shouldAdvanceYaw ≡ true`)

```
psi_ref = seed[2];  p_ref = seed[3:6];  p_ref[2] += h_off;  euler_ref = seed[0:3]
for k in 0..N−1:
    tk = t0 + k*dt
    if k > 0: psi_ref += psi_dot_des * dt                                     (RT:42-49, BMR:35)
    psi_k   = psi_ref
    v_ref_W = Rz(psi_k) @ u_des_B                                            (RT:52, BMR:58-60)
    if k > 0: p_ref[0:2] += (Rz(psi_k) @ u_des_B)[0:2] * dt ; p_ref[2] = seed[5] + h_off   (RT:54-61, BMR:45-56)
    psi[k] = psi_k ; tk[k] = tk
    r_left[:,k]  = left_des_W  − p_ref                                       (RT:65)
    r_right[:,k] = right_des_W − p_ref                                       (RT:66)
    x_ref_k = [euler_ref[0], euler_ref[1], psi_k, p_ref, 0, 0, psi_dot_des, v_ref_W, g]   (RT:68-81)
```

Notes: `x_ref[8] = psi_dot_des` (yaw-rate reference), `x_ref[6:8] = 0`, `x_ref[12] = x0[12] = g`. Yaw is never wrapped. With modes `single_support`/`double_support` the yaw advance at step k is gated by `bothFeetStance(tk − dt/2)` (`RT:41`, `BMR:7-24`) — not used for MIT.

### 6.4 `r_left`, `r_right` — which points, which frame

`r_side[k] = (desired foot position, W) − (reference COM position at step k, W)`, **world frame**, from **foot end-effector site** to **reduced-body COM (reference, not measured)**. The foot position is constant over the horizon (no per-step foot plan); only `p_ref` moves. Desired foot positions (`MC:1396-1416`, `SFP:277-370`):

* stance foot: the touchdown target frozen when it entered stance (initially seeded from the current site position, `SFP:336-343`, `SFP:149-151`) — i.e. **not** the live measured position; in practice ≈ actual foot site position on the ground;
* swing foot: planned touchdown target with `z = −0.005` (`SFP:10`, `SFP:273`), Raibert-style (`SFP:235-275`, outside this spec);
* contact manager may override with frozen/search targets (`CM:394-414`);
* standing mode: current site position with `z = −0.005` (`MC:1396-1400`).

So `r` for a swing foot is the *future* foothold — the lever arm used when that foot's contact starts later in the horizon.

---

## 7. Dynamics: `A_c`, `B_c`, ZOH, lifted matrices (`MPCF:39-107`)

Per step `k` (`MPCF:63-90`), with `m = bodyMass`, `I_B = bodyInertia`, `R_k = Rz(psi_k)`, `I_k = R_k I_B R_k^T`, `I_k^{-1} = R_k I_B^{-1} R_k^T`, `I3` identity:

```
A_c_k (13×13) = 0 except
    A_c[0:3, 6:9]  = R_k^T          (Euler-rate ≈ Rz^T ω : small-angle roll/pitch)     MPCF:73
    A_c[3:6, 9:12] = I3                                                                MPCF:74
    A_c[9:12, 12]  = [0, 0, 1]^T    (v̇_z += g, g = −9.81 lives in x[12])              MPCF:75

B_c_k (13×12) = 0 except
    B_c[6:9, 0:3]  = I_k^{-1} skew(r_left_k)      (ω̇ from F_L:  I^{-1} (r_L × F_L))    MPCF:78
    B_c[6:9, 3:6]  = I_k^{-1} skew(r_right_k)                                          MPCF:79
    B_c[6:9, 6:9]  = I_k^{-1}                      (ω̇ from M_L)                        MPCF:80
    B_c[6:9, 9:12] = I_k^{-1}                                                          MPCF:81
    B_c[9:12, 0:3] = I3 / m                                                            MPCF:82
    B_c[9:12, 3:6] = I3 / m                                                            MPCF:83
```

As block literals (rows/cols grouped 3,3,3,3,1 for A; 3,3,3,3 for B):

```
        [ 0   0   R_k^T  0    0 ]            [   0          0        0       0    ]
        [ 0   0   0      I3   0 ]            [   0          0        0       0    ]
A_c =   [ 0   0   0      0    0 ]     B_c =  [ I^-1[r_L]x  I^-1[r_R]x  I^-1   I^-1 ]
        [ 0   0   0      0   e_z ]           [  I3/m       I3/m       0       0    ]
        [ 0   0   0      0    0 ]            [   0          0        0       0    ]
```

(no gyroscopic term `ω × Iω`; no `[r]x` in rows 0-2; gravity via the constant state.)

ZOH (`MPCF:10-26`):

```
A_d = I + dt A_c + ½ dt² A_c²
B_d = dt B_c + ½ dt² A_c B_c + (dt³/6) A_c² B_c
```

Structure facts (verified numerically, §13): `A_c² B_c ≡ 0` (row 12 of `B_c` is zero), so **the cubic term is identically zero**; `A_c²` only contributes `A_d[3:6,12] = ½ dt² e_z = 2e-4·e_z` (gravity → position). Non-zero blocks of `A_d`: `I`, `A_d[0:3,6:9] = dt R_k^T`, `A_d[3:6,9:12] = dt I3`, `A_d[9:12,12] = dt e_z`, `A_d[3:6,12] = ½dt² e_z`. Non-zero of `B_d`: rows 6:9 = `dt B_c[6:9,:]`, rows 9:12 = `dt B_c[9:12,:]`, rows 0:3 = `½dt² R_k^T B_c[6:9,:]`, rows 3:6 = `½dt² B_c[9:12,:]`.

Lifted (`MPCF:92-106`), `A_qp ∈ R^{13N×13}`, `B_qp ∈ R^{13N×12N}`:

```
A_qp[13k:13k+13, :]  = A_d[k] A_d[k−1] … A_d[0]                         (prefix product, MPCF:92-96)
B_qp[13k:, 12j:]     = A_d[k] A_d[k−1] … A_d[j+1] B_d[j]   for k ≥ j      (MPCF:98-106; k=j → B_d[j])
```

So `x_{k+1} = A_d[k] x_k + B_d[k] u_k` and row block `k` of `X` is `x_{k+1}` (the state **after** applying `u_k`); `x0` itself is not in `X`. The reference row `k` (`x_ref_k`, built with `psi_ref` advanced `k` times) is compared against `x_{k+1}`. Also stored: `inertia_W[k] = I_k` (`MPCF:87`, unused by the QP).

---

## 8. Cost (`CMPC:675-685`, `CMPC:835-869`, `CC.cpp:58-94`)

Weights (diagonal, `YAML:28-30` walking; `YAML:36-38` standing; loaded by `fillDiagonal`, `CC.cpp:58-74`):

```
Q_walk = diag([50000, 9000, 500,   200000, 1009300, 110000,   10, 10, 10,   100, 70, 10,   1])
         #      roll  pitch yaw    px      py       pz        wx  wy  wz    vx   vy  vz   g
R_walk = 1e-3 * I12
Q_stand = diag([50000, 80000, 500,  50000, 90000, 50000,  10, 10, 10,  10, 10, 5,  1])
R_stand = 1e-4 * I12
```

The gravity slot has weight 1 (`x[12]` and `x_ref[12]` are both `g` and `A_qp` keeps it constant, so its error is exactly zero — harmless; keep it).

Per-step yaw transform (`CMPC:675-685`), `psiRef_k = X_ref[13k+2]`:

```
R_BW = Rz(psiRef_k)^T
T_k  = I13 with blocks [3:6,3:6] = [6:9,6:9] = [9:12,9:12] = R_BW      (orientation block NOT rotated)
Q_k  = T_k^T Q T_k
```

QP build (`CMPC:835-869`):

```
e      = A_qp x0 − X_ref                                   (13N)             CMPC:837-839
for k:  Wb[13k:13k+13,:] = Q_k B_qp[13k:13k+13,:] ;  we[13k:] = Q_k e[13k:]   CMPC:844-853
H      = 2 (B_qp^T Wb) ;  H[12k:,12k:] += 2R  ;  H = ½(H + H^T)               CMPC:858-863
q      = 2 B_qp^T we                                                         CMPC:868
```

OSQP objective `½ U^T H U + q^T U` = `Σ_k (x_{k+1}−x_ref,k)^T Q_k (…) + u_k^T R u_k + const`. **No terminal weight, no state-cost on `x0`.** Only the upper triangle of `H` is passed (`CMPC:50-64`, `270-282`).

Mode selection: `Q,R = standing set if locomotionMode == Standing else walking set` (`CMPC:808-816`; `Interactive` uses walking). Note the asymmetric rule for `contact_wrench_model`: `walking model if mode == Walking else standing model` (`CC.cpp:718-722`; `Interactive` → standing model). For MIT both are `full_wrench`, so no difference.

Config defaults if keys are missing (`CC.h:31-37, 40-44, 46-61`): state/input weights = Identity (13×13 / 12×12) if `*_weight_diag` absent, `iterations_between_solve` 10, `use_shifted_warm_start` true, timing `(cycle 1.0, swing 0.4, stance 0.6, horizon 0.5, horizon_steps 15)`, gravity −9.81. The MIT yaml sets all of them.

---

## 9. Constraints

### 9.1 Layout (`CMPC:82-126`, `128-268`, `717-731`)

Rows `0 … 24N−1` are inequalities (`C U ≤ C_bound`, lower = −∞); rows `24N … 36N−1` are equalities (`D U = 0`, lower = upper = 0). Per step: 24 inequality rows (left 0-11, right 12-23) and 12 equality rows. Total constraints `36N = 900`, variables `12N = 300`. **Live code builds every coefficient directly with `directConstraintValue` (`CMPC:494-576`) into a fixed sparsity pattern (`CMPC:128-268`); the dense `GaitScheduler::C/D` members and `fillYawRotatedFootConstraintBlock` (`GS:32-56`) are dead code.** The numeric result is identical to the docs' description; use whichever is convenient in Python (dense is fine).

### 9.2 Foot-local template `C_F` (12×6, wrench order `[Fx,Fy,Fz,Mx,My,Mz]`) (`GS:83-95`, `CMPC:406-437`)

With MIT values `µ = 1.0`, `a = footHalfLength = 0.065`, `b = footHalfWidth = 0.01`, `µ_t = torsionalFrictionScale·µ = 0.0657·1.0 = 0.0657` (`YAML:17-20`):

```
row  0:  [ 1,  0, -1.0,     0,  0,  0 ]   Fx_F − µFz ≤ 0
row  1:  [-1,  0, -1.0,     0,  0,  0 ]  −Fx_F − µFz ≤ 0
row  2:  [ 0,  1, -1.0,     0,  0,  0 ]   Fy_F − µFz ≤ 0
row  3:  [ 0, -1, -1.0,     0,  0,  0 ]  −Fy_F − µFz ≤ 0
row  4:  [ 0,  0,  1,       0,  0,  0 ]   Fz ≤ 1500                (bound = normalForceMax)
row  5:  [ 0,  0, -1,       0,  0,  0 ]  −Fz ≤ −minScale·5         (bound = −minScale·normalForceMin)
row  6:  [ 0,  0, -0.01,    1,  0,  0 ]   Mx_F − b Fz ≤ 0         (CoP  y_cop = +Mx/Fz ≤ +b)
row  7:  [ 0,  0, -0.01,   -1,  0,  0 ]  −Mx_F − b Fz ≤ 0         (y_cop ≥ −b)
row  8:  [ 0,  0, -0.065,   0,  1,  0 ]   My_F − a Fz ≤ 0         (CoP  x_cop = −My/Fz ≥ −a)
row  9:  [ 0,  0, -0.065,   0, -1,  0 ]  −My_F − a Fz ≤ 0         (x_cop ≤ +a)
row 10:  [ 0,  0, -0.0657,  0,  0,  1 ]   Mz − µ_t Fz ≤ 0
row 11:  [ 0,  0, -0.0657,  0,  0, -1 ]  −Mz − µ_t Fz ≤ 0
```

Sign convention of the CoP box: with the contact moment `M` about the foot site and `Fz > 0`, `x_cop = −My/Fz`, `y_cop = +Mx/Fz` (`docs/friction_cop…md:379-387`). Rows 6-7 bound `|Mx_F| ≤ b·Fz` (roll moment ↔ **half-width 0.01 m**, lateral CoP), rows 8-9 bound `|My_F| ≤ a·Fz` (pitch moment ↔ **half-length 0.065 m**, fore-aft CoP). The box is symmetric, so the sign of the CoP definition does not matter for the feasible set; only `a↔b` assignment matters. Torsional friction: `|Mz| ≤ 0.0657 · Fz` (i.e. `torsional_friction_scale` multiplies `µ`; with µ=1 the effective torsional coefficient is 0.0657 m).

Config defaults if a key is missing (`CC.h:46-52`): µ 0.1, a 0.065, b 0.01, scale 0.0657, Fmax 200, Fmin 10 — MIT yaml overrides all six.

### 9.3 Yaw rotation into world-frame variables (`CMPC:439-464`; docs §8.2)

For a foot with yaw `ψ_f` (`c = cos ψ_f`, `s = sin ψ_f`), each template row `[fx,fy,fz,mx,my,mz]` becomes the world-variable row

```
[ c·fx − s·fy,  s·fx + c·fy,  fz,  c·mx − s·my,  s·mx + c·my,  mz ]
= row · blkdiag(Rz(ψ_f)^T, Rz(ψ_f)^T)       (i.e. C_W = C_F · blkdiag(R_WF^T, R_WF^T))
```

Explicit `C_W` for MIT values:

```
[  c    s  -1      0    0   0 ]
[ -c   -s  -1      0    0   0 ]
[ -s    c  -1      0    0   0 ]
[  s   -c  -1      0    0   0 ]
[  0    0   1      0    0   0 ]
[  0    0  -1      0    0   0 ]
[  0    0  -0.01   c    s   0 ]
[  0    0  -0.01  -c   -s   0 ]
[  0    0  -0.065 -s    c   0 ]
[  0    0  -0.065  s   -c   0 ]
[  0    0  -0.0657 0    0   1 ]
[  0    0  -0.0657 0    0  -1 ]
```

Placed into `u_k` columns via `leftMap`/`rightMap`; left rows `24k+0..11`, right rows `24k+12..23`. **If the foot is swing at step k, all 12 rows are zero** (`CMPC:518-520, 527-529`) and their bound is 0.

### 9.4 Equality rows `D` (12 per step, rows `24N + 12k + i`) (`CMPC:544-575`)

```
i = 0,1,2  : left force  :  D[i, i]   = 1 if !leftStance  else 0     → F_L = 0 in swing
i = 3,4,5  : right force :  D[i, i]   = 1 if !rightStance else 0     → F_R = 0
i = 6      : left Mx     :  if !leftStance: D[6,6] = 1
                            elif NoRollMoment: D[6, 6:9] = leftFootXAxis_W  (x̂_F,W · M_L = 0)
                            else: 0
i = 7,8    : left My,Mz  :  D[i, i]   = 1 if !leftStance
i = 9      : right Mx    :  same as 6 with right, cols 9:12
i = 10,11  : right My,Mz :  D[i, i]   = 1 if !rightStance
```

Lower = upper = 0. Swing-foot variables are therefore **kept in the QP and pinned to zero by equalities** (no variable removal). For a stance foot with `full_wrench` (MIT: `YAML:26,33`) the D rows are all-zero (`0 = 0`).

`contact_wrench_model`: `full_wrench` (MIT) → no extra rows; `no_roll_moment` → the foot-local x-axis (`R_WF.col(0)`, `MC:363-377`, `MC:948-952`) dotted with the world moment must vanish (`GS:144-145`, `CMPC:554-556, 566-568`). The sparse pattern reserves 3 columns for rows 6 and 9 for this (`CMPC:243-257`).

### 9.5 Bounds vector assembly (`CMPC:885-889`)

```
lower[0:24N]   = −INF ;  upper[0:24N]   = C_bound (§5.3)
lower[24N:36N] = 0    ;  upper[24N:36N] = 0
```

### 9.6 Contact signature (`CMPC:1071-1084`)

`sig = [L_0, R_0, L_1, R_1, …]` (stance flags). If it changed since the last solve → OSQP is **re-initialised (cold)** and the warm start is skipped for that solve (`CMPC:880-906`). Because step 0 = `activeContact` at the current `t` = `c(side, t) OR earlyContact` (§5.4) while steps 1..24 are the fixed cycle-anchored table, the signature changes (and a cold init happens at the next solve) whenever:

* `c(side, t_now)` flips — **4 times per 0.5 s cycle**, at `t − t0 = 0.08` (L lift-off), `0.25` (L touchdown), `0.33` (R lift-off), `0.5/0` (R touchdown) → ≈ 8 cold inits per second in steady walking;
* an early-contact event switches on or off;
* the mode changes (Standing all-ones ↔ Walking table), e.g. at the Standing→Walking transition;
* the left `k=4` knife edge flips with `t0` (§5.2).

It does **not** depend on a force-threshold contact detector.

---

## 10. Solver: OSQP settings, warm start, retry, failure fallback

### 10.1 Settings (`CMPC:1015-1043`)

| setting | value | note |
|---|---|---|
| verbose | false | |
| warm_start | true | OSQP internal reuse of previous x,y across `solve()` calls on the same workspace |
| polish | false | |
| max_iter | 200 | |
| adaptive_rho | true | |
| everything else | **not set → library defaults** (`osqp_set_default_settings`) | osqp via vcpkg baseline `4bc07e3e` (`vcpkg.json` pins no osqp version); the code uses the `c_float` API, which implies OSQP 0.6.x. 0.6.x defaults: eps_abs = eps_rel = 1e-3, eps_prim_inf = eps_dual_inf = 1e-4, rho = 0.1, sigma = 1e-6, alpha = 1.6, scaling = 10, check_termination = 25, scaled_termination = 0, delta = 1e-6, adaptive_rho_tolerance = 5, adaptive_rho_fraction = 0.4, **adaptive_rho_interval = 0 (auto)**. |

The only explicit settings are the five above (`CMPC:1024-1028`, O5). **adaptive_rho_interval = 0 (auto)** in OSQP 0.6: if built with `PROFILING` (the 0.6 CMake default), the interval is chosen from wall-clock time during the first solve on a workspace (first iteration count at which elapsed time > `adaptive_rho_fraction × setup_time`, rounded to a multiple of `check_termination`, at least `check_termination`) and written back into the workspace settings — the reference rho schedule is therefore timing-dependent and not bit-reproducible; without `PROFILING` it is `4 × check_termination = 100`. Only `Solved`/`SolvedInaccurate` are accepted (`CMPC:658-661`); `MaxIterReached` counts as failure (§10.3).

Python port: prefer python `osqp` 0.6.x. On `osqp` 1.x (the local `.venv` has 1.1.3) the names and defaults differ — pass `verbose=False, warm_starting=True, polishing=False, max_iter=200, adaptive_rho=1 (iterations), adaptive_rho_interval=100` (deterministic stand-in for the timing rule; 1.1.3 otherwise auto-sets 50), and **`check_dualgap=False`** (exists in 1.1.3, default 1; it adds a termination criterion that changes how often `max_iter` is hit and hence how often the cold retry / fallback run). Keep eps/rho/sigma/alpha/scaling/check_termination at the 0.6 values above (1.1.3 defaults are the same). Treat any status other than solved / solved-inaccurate as failure.

`H` is passed as upper-triangular CSC (`CMPC:50-64`), constraint matrix as the fixed 76-entry-per-step pattern (`CMPC:128-268`); on non-cold solves only values are updated (`updateHessianMatrix/updateGradient/updateLinearConstraintsMatrix/updateBounds`, `CMPC:1045-1061`). **Both patterns are FIXED and contain explicit zeros**: `P` = the full upper triangle, all `300·301/2 = 45150` entries (`makeUpperTriangularPattern`, `CMPC:50-64`, values written by `fillUpperTriangularValues`, `CMPC:270-282`); `A` = the §9.1 row-level pattern, 76 entries per step, with swing-foot rows filled with 0.0 (`CMPC:128-268`, `:578-597`). In Python: build `P_pattern` and `A_pattern` **once** as CSC with explicit zeros (never via `scipy.sparse.triu(dense)`, `csc_matrix(dense)` or `eliminate_zeros()`, which drop exact zeros and change `nnz`); every solve, fill the value arrays in the same CSC order (P: column-major over `row ≤ col`; A: iterate the stored pattern) and pass them as `Px=…, Ax=…` **without** index arrays. `setup(P, q, A, l, u, **settings)` on the first solve, on a contact-signature change and on a failed-solve retry; otherwise `update(Px=…, q=…, Ax=…, l=…, u=…)` on the same persistent solver object.

### 10.2 Warm start (`CMPC:920-938`, `1086-1108`; `YAML:23 use_shifted_warm_start: true`)

After every successful solve (`CMPC:985-993`):

```
if use_shifted_warm_start:
    warm[0 : 12(N−1)] = sol[12 : 12N]        # drop u_0, shift everything one step earlier
    warm[12(N−1) : 12N] = sol[12(N−1) : 12N] # repeat the last step
else:
    warm = sol
```

Before the next solve, if `hasPreviousSolution && solverInitialized && !skipWarmStartForCurrentSolve && all finite` (`CMPC:1086-1092`) → `setPrimalVariable(warm)` (`CMPC:932`). Only the primal is set explicitly (`osqp_warm_start_x`: sets `x`, and `z = A x`), **but the dual `y` is implicitly warm-started**: `setWarmStart(true)` (`CMPC:1025`) and the workspace persists across non-cold solves (`CMPC:891-898`), so the solve starts from the shifted primal plus the **unshifted** `y` of the previous solve. Only a cold re-init (first solve, contact-signature change, failed-solve retry) resets `x` and `y` to 0, and that solve gets no primal warm start. Python: keep one persistent `osqp` object, call `update(...)` then `warm_start(x=warm)` **without** `y`; never re-create the solver or zero `y` except where the reference re-inits.

Note the shift is by one MPC step (20 ms) although solves happen every 14 ms (7 ticks) — the shift is not time-consistent; copy it anyway (O7). `updateWarmStart` runs only after a successful solve (`CMPC:985-993`); if the solve throws, the previous `warm` vector is kept unchanged (and the solver's internal x, y are those of the failed cold retry).

### 10.3 Status handling and retry (`CMPC:955-978`)

Accept `Solved` or `SolvedInaccurate` (`CMPC:658-661`; `MaxIterReached`, infeasible etc. are failures). Otherwise: re-initialise OSQP from scratch (fresh `setup`, x = y = 0), solve once **cold** (no primal warm start); if that also fails → throw. The exception is caught in `MC:1016-1052`: `_stanceWrenchWorld = 0` then, for each *active-contact* foot, `Fz = bodyMass·|g| / nStance` (forces only, no moments) is used as the fallback wrench until the next solve.

### 10.4 Output (`CMPC:990-991`, `MC:1010`)

`optimalWrench = U*[0:12]` (= `u_0`) → `_stanceWrenchWorld` (held constant between solves). `optimalWrenchHorizon = U*` (debug only).

---

## 11. Post-processing → leg controller (`MC:1297-1372`, `LegController.cpp:293-317`, `OperationalSpaceDynamics.cpp:87-94`)

Every tick (500 Hz), for each leg:

```
if activeContact(side):                                    # walking: contact manager; standing: gait c()
    α = contactRampAlpha(side)                             # walking: §5.4 (0 on the onset tick, then 0.2/tick); standing: 1
    F_cmd_W = −α · u_0[F_side]                             # MC:1322-1331: foot pushes the ground → negate GRF
    M_cmd_W = −α · u_0[M_side]
    if enable_stance_foot_yaw_hold (YAML:51 true) and α > 0:   # walking only (MC:1338-1348, 86-117)
        x̂_proj = normalize(R_WF[:,0] − (R_WF[:,0]·ẑ) ẑ)   # foot SITE x-axis projected on the horizontal plane
        x̂_des  = [cos ψ_td, sin ψ_td, 0]                    # ψ_td = touchdownYaw_W[side] (latched, §5.5)
        yawErr = atan2(ẑ·(x̂_proj × x̂_des), x̂_proj·x̂_des)
        yawRate = ẑ·(Jw_W q̇_leg)                          # 5 LEG joints only; excludes floating-base/torso rotation
        M_cmd_W.z += α · (kp·yawErr − kd·yawRate)          # kp = stance_yaw_kp = 20, kd = stance_yaw_kd = 4 (YAML:70-71)
        # term = 0 if kp ≤ 0 and kd ≤ 0, if ψ_td non-finite, or if ‖x̂_proj‖ ≤ 1e-9 before normalising
    τ_leg = Jv_W^T F_cmd_W + Jw_W^T M_cmd_W               # OperationalSpaceDynamics.cpp:93 (no gravity/bias comp)
else:
    swing-foot Cartesian PD + attitude torque (outside this spec; MC:1352-1370)
```

`Jv_W`, `Jw_W` = 3×5 world-frame translational/rotational Jacobians of the foot site w.r.t. the 5 leg joints (from `LegSwingDynamicsProvider`, not read here). No torque limits are applied in the controller (MuJoCo `ctrlrange` clamps: ±34/34/72/144/68 N·m for hip_yaw/abad/pitch/knee/ankle). Standing mode instead uses one combined 6×10 two-foot Jacobian with the same negation (`MC:1254-1295`) and **never** applies the stance yaw hold. The leg Jacobians come from auxiliary fixed-base leg models evaluated with the (one-step stale) torso pose and the current leg `q, qd` (§2.3).

There is **no ramping of the MPC solution itself** beyond `α` (contact ramp, 10 ms) — no interpolation between consecutive solves.

---

## 12. Locomotion modes relevant to reproduction (`MC:393-474`, `LocomotionFSM.cpp`, `YAML:1`, `YAML:107`)

`requested_locomotion_mode: walking`; start-up: leg PD initialisation (2 s, `YAML:135`), then **Standing** MPC for `post_init_standing_settle_time = 1.0 s` (`YAML:107`), then gait clock reset + Walking. In Standing: scheduler `c ≡ true` (both feet always stance, all 25 steps), `Q_stand/R_stand`, reference = fixed body target (average foot xy, nominal height, yaw integrated), foot targets = current sites, no contact override, no ramp. Walking as specified above.

---

## 13. Worked numeric example (local MIT MJCF; script `worked_example.py` next to this file, output `worked_example_output.txt`)

Inputs: `dt = 0.02`, `m = 14.2275 kg`, `I_B` from §2.2 at the yaml initial pose, `psi_k = 0.3 rad`, `p_ref = [0, 0, 0.62]`, `left_des_W = [0.02, 0.076, 0]`, `right_des_W = [−0.05, −0.076, 0]` → `r_L = [0.02, 0.076, −0.62]`, `r_R = [−0.05, −0.076, −0.62]`.

```
Rz(0.3)^T = [[ 0.955336, 0.295520, 0], [-0.295520, 0.955336, 0], [0,0,1]]
I_k^{-1}  = [[ 2.28689, -1.148424, -0.03103], [-1.148424, 5.577205, 0.006429], [-0.03103, 0.006429, 2.484747]]
B_c[6:9,0:3] = I_k^{-1} skew(r_L) = [[ 0.714381, 1.417251, 0.196772],
                                     [-3.458356,-0.711894,-0.198824],
                                     [-0.192827, 0.030456,-0.002487]]
B_c[9:12,0:3] = I3 / m = 0.070287 · I3
A_d[0:3,6:9] = dt Rz^T = [[0.019107, 0.005910, 0], [-0.005910, 0.019107, 0], [0, 0, 0.02]]
A_d[3:6,12]  = [0, 0, 0.0002]   (½dt² g-column) ;  A_d[9:12,12] = [0, 0, 0.02]
B_d[6:9,0:3] = dt·B_c[6:9,0:3] = [[0.014288, 0.028345, 0.003935], [-0.069167,-0.014238,-0.003976], [-0.003857, 0.000609,-0.00005]]
B_d[0:3,0:3] = ½dt² Rz^T I_k^{-1} skew(r_L) = [[-0.000068, 0.000229, 0.000026], [-0.000703,-0.000220,-0.000050], [-0.000039, 0.000006, 0]]
B_d[9:12,0:3] = 0.001406 · I3 ;  B_d[3:6,0:3] = 0.000014 · I3
‖A_c² B_c‖ = 0  → cubic ZOH term vanishes.
```

Constraint block for a foot with yaw 0.3 (`c = 0.955336, s = 0.295520`): rows exactly as §9.3 with those `c, s`, e.g. row 0 = `[0.955336, 0.295520, −1, 0, 0, 0]`, row 8 = `[0, 0, −0.065, −0.295520, 0.955336, 0]`. Verified `C_W == C_F · blkdiag(Rz^T, Rz^T)`.

Schedule table with `t0 = 0` is the one in §5.2. At `t = 0.137 s`: `p_L = 0.774` (left in swing, 0.113 s of swing remaining), `p_R = 0.274` (right stance) — but the horizon grid still starts at `t0 = 0` where both feet are in stance; step 0 is then replaced by the override with `activeContact` at `t` = (L=0, R=1) (L=1 only if an early contact is active), with `minScale_R = contactRampAlpha_R`.

---

## 14. Docs vs code discrepancies (code wins)

| # | topic | docs say | code does |
|---|---|---|---|
| D1 | solve cadence | README:404 "`iterations_between_solve = 10` → 50 Hz" | YAML:24 → 7 ticks = 14 ms ≈ 71 Hz… i.e. **every 7 physics steps** (`MC:918-919`) |
| D2 | where `C`, `D` are built | `docs/gait_scheduler…md:107-138`, `docs/friction_cop…md:909-1034`: `GaitScheduler` fills dense `C`, `D` via `fillYawRotatedFootConstraintBlock` | `GS:134-193` fills only `C_bound` + per-step flags; coefficients are generated in `CMPC:494-576` (`directConstraintValue`). Numerically identical. |
| D3 | yaw reference | `docs/yaw…md:327-331` uses `t = (k+1)·dt` and `psi_u[0] + psi_dot·t` | `RT:42-49`: `psi_ref` is advanced `k` times (k=0 → seed yaw), i.e. `t = k·dt`; gating by `yaw_integration_mode` (`always` for MIT) |
| D4 | horizon time base | docs §2 (`gait_scheduler…md:42-48`) say `t_k = t0 + kΔt` "t0 = synchronized cycle origin" — consistent with code | (no discrepancy, but easily misread as "current time") |
| D5 | cost frame | `docs/mpc_frame_convention.md:43-52` first says the cost is world-fixed, then §5 recommends the yaw transform | code implements the transform (`CMPC:675-685`) — docs §1 is stale |
| D6 | "shifted" warm start semantics | not documented | shift by one MPC step regardless of 14 ms solve spacing (`CMPC:1104-1107`) |
| D7 | `docs/friction_cop…md` §11 says `D U = d` | `d ≡ 0` (`CMPC:888-889`) |
| D8 | README says "MPC solve … 50 Hz" and "Reference trajectory 50 Hz" (`README.md:404-406`) | both at the QP cadence, 7 ticks |

---

## 15. Python implementation checklist (order of operations per tick, `MC:1374-1429`)

```
every tick (dt_ctrl = 0.002):
  0. read mjData as mj_step left it — NO mj_forward (one-step kinematic lag)   (§2.3)
  1. params ← updateReducedBodyMassProperties(mjData)            (§2.2)
  2. state ← cheater state (+ yaw unwrap)                          (§3)
  3. FSM sync (on transition: gait-clock reset t0 = t, resetSwingState → _lastControlTime = t,
     contactManager.reset); horizonClock.sync(t)                   (§1.2, §5.4, §6.2)
  4. x0 ← buildCurrentMpcState()                                   (§3)
  5. dt = t − _lastControlTime; filtered command update (α=1 if dt ≤ 0); body target update   (§6.1-6.2)
  6. nominal desired foot positions; contactManager.update (activeContact = c(t) OR early, ramp α);
     managed foot positions                                        (§5.4, §6.4)
  7. swing trajectories update (latches touchdownYaw_W on first swing tick; sets _lastControlTime = t)   (§5.5)
  8. if iteration == 0 or iteration − last ≥ 7:
        override ← contactManager.buildHorizonOverride()           (§5.4)
        footYaw_L/R ← §5.5
        schedule/C_bound ← GaitScheduler.buildConstraintMatrices(override, yawL, yawR)   (§5.2-5.3)
        seed ← §6.1 ; X_ref, psi, r_L, r_R ← ReferenceTrajectory   (§6.3)
        A_c,B_c,A_d,B_d,A_qp,B_qp ← MPCFormulation                 (§7)
        H, q, A_cons, l, u ← ConvexMPC.buildQP                      (§8, §9)
        cold re-setup if contact signature changed, else update    (§9.6, §10.1)
        solve (warm start §10.2, retry §10.3) → u_0                 (§10.4)
        on exception: fallback wrench (§10.3)
        last = iteration                                            (also after a failure)
  9. leg commands from u_0 (negated, ramped)                        (§11)
  10. iteration += 1
```

Dimensions to assert: `A_qp (325×13)`, `B_qp (325×300)`, `X_ref (325)`, `H (300×300)`, `A_cons (900×300)`, `C_bound (600)`.

---

## 16. Former open questions — answers (where each now lives)

| # | question | answer (section) |
|---|---|---|
| O1 | horizon grid anchored at cycle start; interaction with step-0 override | Confirmed: `tk = t0 + k·dt`, `t0` = start of current cycle (init at controller start `MC:411`, reset to `t` at Standing→Walking `MC:564-566`, advanced by whole 0.5 s cycles in `sync`). Steps 1..24 = §5.2 table (subject to the left `k=4` floating-point knife edge, which flips to stance for `t0 ≳ 16 s` in a realistic run). Step 0 = `activeContact` at the current `t` = `c(side,t) OR earlyContact`, `minScale = contactRampAlpha` (§1.2, §5.2, §5.4). |
| O2 | reduced-body m / I / COM recomputed every tick? | Yes, every 2 ms tick; but `nominalHeight_W` is latched once at the first controller tick (§2.2, §6.1). |
| O4 | 0.002 s physics vs 7 ticks | 7 controller ticks = 7 MuJoCo steps = 14 ms; counter not reset at transitions; `last` updated on failure too; default 10 (§1.1). |
| O5 | which OSQP settings are explicit | verbose=false, warm_start=true, polish=false, max_iter=200, adaptive_rho=true; rest = 0.6.x defaults incl. adaptive_rho_interval = 0 (auto/timing-based) (§10.1). |
| O7 | shifted warm start vs 7-tick spacing | Confirmed: shift by one 12-block after every successful solve regardless of 14 ms spacing; kept unchanged on failure; dual implicitly warm (§10.2). |
| O8 | foot yaw when not in contact / definition of touchdown yaw | `touchdownYaw_W` latched on the first swing tick, lead + one-sided ±20° psi bias; measured yaw lifted near it while in contact (§5.5). |

## Verification log (9/25)

1. §5.4 / §9.6 activeContact is `c(side, t_now)` + early contact, not a force detector; signature flips 4×/cycle — **applied** (verified `CM:222, 266-274, 284-290, 327-338, 416-446`, `YAML:99-100`; rewrote §1.2, §5.4, §9.6, §13).
2. §6.2 filter α = 1 at dt ≤ 0, transition tick snaps to raw, height offset unfiltered, zeroMotionCommand, first-tick init, double clamp — **applied** (verified `MC:28-33, 552, 567-569, 610-654, 867, 1390`, `LocomotionFSM.cpp:280-281`). Note: the reset happens in `syncLocomotionFSM()` (called from both `prepareController` and `runController`), before `dt` is computed on that tick.
3. §10.2 dual y implicitly warm-started (setWarmStart(true) + persistent workspace) — **applied** (`CMPC:891-898, 932, 1025`).
4. §10.1 OSQP defaults / adaptive_rho_interval auto / python 1.x differences — **applied**; `check_dualgap` verified to exist in the local python osqp 1.1.3 (default 1; adaptive_rho_interval auto-sets to 50; max_iter default 4000). The PROFILING timing rule is from OSQP 0.6 source knowledge (source not available locally); vcpkg.json confirmed to pin no osqp version.
5. §10.1 fixed sparsity patterns with explicit zeros; never eliminate zeros — **applied** (`CMPC:50-64, 128-268, 270-282`).
6. §5.5 / O8 touchdownYaw_W latched once per swing, one-sided psi bias, measured-yaw exception fallback — **applied** (`MC:455-456, 830-854, 891-909, 930-946`, `SwingYawTarget.h:11-34`, `YAML:45-50`).
7. O1 — **applied with correction**: the claim that left `k=4` "always evaluates to swing" is **wrong**. Python doubles with the exact C++ expressions give stance at `t0 = 100.25` and `7.998`, and, emulating `t += 0.002`, reset ≈ 3.002 and `t0 += 0.5`, from `t0 = 16.002 s` on. §5.2 now documents the knife edge and requires evaluating `c()` live with identical arithmetic instead of a hard-coded table.
8. O2 reduced-body per tick; nominalHeight_W latched once — **applied** (`SimulationRunner.cpp:409-411`, `MC:420, 571, 682-697`).
9. O4 counter semantics, default 10, clamp ≥ 1 — **applied** (`MC:438, 460-461, 918-919, 1054`, `CC.h:58`).
10. O5 explicit OSQP settings — **applied** (`CMPC:1024-1028`), merged into §10.1 / §16.
11. O7 shift after every successful solve, kept on failure — **applied** (`CMPC:985-993, 1086-1108`).
12. §2.3 one-step kinematic lag — **applied with correction**: body/site kinematics, `cvel` and contact forces are one step stale as claimed, but the leg Jacobians are **not** taken from the main `mjData`; they are computed by `mj_forward` on auxiliary per-leg models from the stale torso pose plus the **current** leg `q, qd` (`LegSwingDynamicsProvider.cpp:603-640`).
13. §11 yaw-hold details (Jw·q̇_leg, projected site x-axis, α > 0, walking only, zero-return cases) — **applied** (`MC:86-117, 1338-1348`; the standing path never calls it).
14. §5.4 contactRampAlpha onset α = 0 then +0.2/tick, reset/init 0.01, standing 1 — **applied** (`CM:179, 195, 238-239, 252, 340-353`).
15. §8/§9.2 config defaults and Interactive asymmetry — **applied** (`CC.h:31-61`, `CC.cpp:718-722`, `CMPC:808-816`).

## Confirmed by verifier

- §1: N=25, dt = horizon/steps = 0.02, cycle 0.5, swing 0.17, stance 0.33, iterations_between_solve 7 (YAML:3-8, 24; CC.cpp:645-647); solve on _iteration==0 or every 7 ticks (MC:918-919); u_0 held between solves (MC:1010)
- §1.2 HorizonClock sync/tk formulas (HC:18-34); t0 init at MC:411, reset at MC:564-566
- §5.1 phases phi_L=0.5, phi_R=0, fmod, stance if 0≤p<stance/cycle; Standing p≡0, c≡true (GS:98-120)
- §5.2 schedule table recomputed numerically (L stance k=0..3,13..24; R stance k=0..16)
- §5.3 C_bound entries [4]=Fmax, [5]=−minScale·Fmin, [16],[17] for right; min scale clamped to [0,1] (GS:149-180)
- §4 input order [F_L,F_R,M_L,M_R], leftMap [0,1,2,6,7,8], rightMap [3,4,5,9,10,11] (CMPC:184-193); u is the ground reaction on the robot and the leg command negates it (MC:1319-1331)
- §7 A_c/B_c blocks: A_c[0:3,6:9]=Rz^T, A_c[3:6,9:12]=I, A_c[9:12,12]=e_z; B_c rows 6:9 = I_k^{-1}[r]x and I_k^{-1}, rows 9:12 = I/m; I_k^{-1}=R I_B^{-1} R^T (MPCF:63-90)
- §7 ZOH A_d=I+dtA+½dt²A², B_d=dtB+½dt²AB+dt³/6 A²B; A_c²B_c≡0 verified analytically; lifted A_qp prefix product and B_qp = A_d[k]…A_d[j+1]B_d[j]; row block k = x_{k+1} (MPCF:10-26, 92-106)
- §13 worked numbers recomputed: B_c[6:9,0:3], B_d[0:3,0:3], 1/m·dt = 0.001406, ½dt²/m = 1.4e-5, C_W rows with c,s of 0.3 rad
- §8 Q/R walking and standing diagonals match YAML:28-38 literally; Q_k = T^T Q T with R_BW blocks at [3:6],[6:9],[9:12] and orientation block unrotated (CMPC:675-685); H = 2B^T Q_k B + 2R per block, symmetrised, upper triangle only; q = 2B^T Q_k (A_qp x0 − X_ref) (CMPC:835-869); no terminal weight
- §9.2 foot-local template rows incl. a=half-length on My rows 8-9, b=half-width on Mx rows 6-7, torsional µ_t = scale·µ (CMPC:406-437, GS:83-95); CoP signs x_cop=−My/Fz, y_cop=Mx/Fz consistent with r×F
- §9.3 yaw rotation = C_F·blkdiag(Rz^T,Rz^T) (CMPC:439-464); swing foot rows all zero with zero bound (CMPC:518-529)
- §9.4 D rows: identity on swing-foot force/moment vars, NoRollMoment x-axis rows 6/9, zero for full_wrench stance; bounds l=u=0; ineq lower −INF (CMPC:544-575, 885-889); pattern 60+16=76 entries/step
- §9.6 contact signature [L0,R0,L1,R1,…]; change → cold initializeSolver and no warm start that solve (CMPC:880-906, 1071-1084)
- §10.3 accept Solved/SolvedInaccurate only; otherwise fresh re-init + cold solve, then throw; fallback in MC: zero wrench, Fz = bodyMass·|g|/nActiveStance on active feet, forces only (CMPC:955-978, MC:1016-1052)
- §10.2 shift formula exactly as CMPC:1104-1107
- §6.3 reference loop: psi advanced k times (k=0 → x0 yaw), v_ref = Rz(psi_k)u_B, p_ref advanced from k=1 using psi_k, p_ref.z = seed z + h_off, x_ref[6:8]=0, x_ref[8]=psi_dot (mode always), x_ref[12]=g; r_side = des_foot − p_ref (RT:25-82, BMR)
- §6.1 seed: walking roll/pitch from bodyTarget.euler_W (seed 0 + offsets), z = nominalHeight_W, yaw/xy from x0; standing uses euler_W, nominalPosition, nominalHeight (MC:966-978, 699-736)
- §3 x0 layout, roll/pitch formulas, unwrapped yaw, world torso ω, COM = torsoPos + Rz(yaw_u)·bodyComLocation, comVel = v + ω×offset (MC:63-75, 315-332, 656-675); yaw unwrap and first-sample rule (StateEstimator.cpp:29-47)
- §2.2 reduced-body formula: non-leg bodies with mass>0, COM, parallel-axis inertia, rotated by Rz(torso yaw)^T (SRP:397-491)
- §11 leg torque τ = Jv^T(−αF) + Jw^T(−αM) with no bias term (MC:1315-1331, OperationalSpaceDynamics.cpp:87-94, LegController.cpp:293-317); standing uses combined Jacobian with negation and no α
- MatrixUtils Rz and skew definitions (MU:11-30); RobotParams fields (RobotParams.h:77-79)
- §14 D1 README 50 Hz vs YAML 7 ticks; D5 cost transform implemented despite docs §1
