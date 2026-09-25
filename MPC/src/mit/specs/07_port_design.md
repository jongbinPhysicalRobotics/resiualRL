# 07 — Python port design: reference convex-MPC controller on MIT Humanoid

Target folder: `C:\Users\백종빈\Desktop\4-2\residual RL\MPC\src\mit\` (self-contained: imports only files in this
folder + numpy/scipy/mujoco/osqp/quadprog/yaml). Model: `MPC\models\mit_humanoid\` (already created — do not edit
unless a bug is found; report it instead).

Specs to implement FROM (the reference C++ is the ground truth; specs cite file:line):
- `01_mjcf_verification.md` — model facts, SRB numbers, name map
- `02_reference_mpc_spec.md` — MPC formulation, constraints, OSQP, warm start
- `03_reference_planning_spec.md` — gait phase, horizon clock, reference trajectory, swing planner, swing trajectory, command filter
- `04_reference_lowlevel_spec.md` — reduced body, state reading, stance/swing torque, OSC, attitude, init
- `05_reference_orchestration_spec.md` — tick order, FSM, contact manager, standing mode, config table
- `06_our_python_conventions.md` — our folder/CLI/logging conventions

## 0. Goal and acceptance

Reproduce the reference's MIT demos in our Python/MuJoCo stack, **each sustained 120 s** (project rule):
forward `x_dot = 0.6 m/s`, lateral `y_dot = 0.3 m/s`, in-place turn `psi_dot = 1.3 rad/s`.
Acceptance per test: no fall for 120 s after walking starts; mean achieved velocity (heading frame, after the
command filter has converged) within ±15 % of the command; report pitch/roll σ, foot slip, QP time.

## 1. Fidelity policy

Port the reference **as it is**, including its quirks (horizon anchored at cycle start, open-loop planner yaw,
planned-not-measured stance lever arms, zero wrench on the first ramp tick, reduced-body mass 14.23 kg, …).
Do not "improve" anything in the base implementation. Improvements go behind flags, default off.

### Documented deviations (the only allowed ones in the base implementation)

| id | deviation | reason |
|---|---|---|
| D1 | Model = our copy `MPC/models/mit_humanoid` (URDF-exact converted MJCF + foot sites at (0.03,0,−0.04) + URDF rotor inertia as `armature` + dt 0.002 + keyframes). Loader option `armature=False` for sensitivity. | the reference's MIT MJCF is not public |
| D2 | **Start-up**: spawn at keyframe `stand` (feet flat, site under whole-body CoM, base z 0.679472 = yaml `base_position_W`). Skip the 2 s leg PD initialisation. The controller initialises at t = 0: FSM `StandingSettle` for `post_init_standing_settle_time` = 1.0 s, then `Walking`. Arms: PD (kp 100, kd 5) to `[0,0,0,−1.65]` from t = 0 (they already start there). | yaml joint offsets are for a different joint-zero convention; on this model joint PD alone (even at `stand`, even with gravity comp) falls within 1.5 s — measured |
| D3 | Headless command injector for x_dot, y_dot **and** psi_dot (the reference headless path only schedules x_dot). Default profile mirrors the reference headless default: the raw command is present **from the first controller tick** (reference: raw x_dot = final from t = 1.0 s while the controller only starts at 2.0 s), so the filter's first-call rule (`filtered = raw`) and the dt = 0 snap at the Standing→Walking tick both apply exactly as in the reference (verified findings, spec 03/05). Optional profiles: ramp, or raw step at walking start. Everything goes through the reference filter (tau 0.92/0.8/0.70 s, clamp 0.7/0.5/2.0). | lateral/turn demos were keyboard-driven |
| D4 | Leg dynamics from the full model instead of separate auxiliary MuJoCo models: `Jv, Jw` = `mj_jacSite` columns of the leg DOFs; `M` = leg block of `mj_fullM`; `bias` and `JvDot·qd` computed with **base and all non-leg DOF velocities set to zero** (fixed-base-leg semantics) on a scratch `MjData`. Must be verified numerically equal to a true fixed-base single-leg model built with `mujoco.MjSpec` (test in `01_check_model.py`). | simpler, same numbers |
| D5 | QP solver: python `osqp` 1.1.3 configured like the reference (max_iter 200, polish off, warm start on, adaptive rho, other settings = the reference's explicit values else library defaults — see spec 02), shifted warm start, cold re-setup when the stance signature changes, fallback wrench on failure. Flag `solver="quadprog"` for an exact-QP comparison. | same algorithm; binding differs |

Known unknowns (not deviations — we cannot know them; expose as options, default = our model as is):
- U1 the reference's MIT scene may contain mocap debug-marker bodies with default-density mass (G1/H1 scenes have 4, 0.214 kg)
  that its reduced-body loop includes (spec 04, verified major) — would add ≈0.21 kg and up to ≈+40 % Iyy. Option
  `emulate_debug_markers=False` in `MitModel.read_state` (adds the four point masses at the positions the reference moves
  them to: reduced COM, body target, two touchdown targets — needs controller data, so implement as an injected callback).
- U2 joint damping/armature/frictionloss and contact solref/solimp of their MJCF.

Everything else (gains, weights, timings, thresholds) = MIT yaml values, loaded from `MPC/src/mit/mit_controller.yaml`
(a copy of `reference/config/mit_humanoid/my_controller.yaml` + a `port:` section for D2/D3 settings).

## 2. Conventions (all modules)

- Leg index **0 = Left, 1 = Right** (reference order). Arm index same. Build every index array by joint/body/site
  **name** (our MJCF is right-first in qpos; never assume order).
- Leg joint order `[hip_yaw, hip_abad, hip_pitch, knee, ankle]`; arm `[shoulder_pitch, shoulder_abad, shoulder_yaw, elbow]`.
- MPC input / stance wrench `u ∈ R^12 = [F_L(3), F_R(3), M_L(3), M_R(3)]`, **ground-on-body**, world frame, moments about the foot site.
  Leg torque uses the negative (`−alpha·F`, `−alpha·M`).
- MPC state `x ∈ R^13 = [roll, pitch, yaw_unwrapped, com_W(3), omega_torso_W(3), v_com_W(3), g=−9.81]`.
- Physics dt 0.002 s; controller once per physics step; MPC every 7 controller ticks (and tick 0).
- **Loop order** (matches the reference's 1-tick-stale kinematics): `tau = controller.tick(t)` using the current
  `MjData` **without** calling `mj_forward`, then `mj_step`. Call `mj_forward` once after keyframe reset only.
- Time `t = d.time`. Numpy float64 everywhere. No per-tick Python object churn in hot loops beyond what is needed.
- Korean path: load with `mujoco.MjModel.from_xml_string(scene_text, assets)` where `assets = {basename: bytes}`.

## 3. Modules and interfaces

File ownership in brackets. Interfaces are contracts: other modules call exactly these names. If an implementer
must change a signature, they note it at the top of their file under `# INTERFACE CHANGE:` and in their report.

### 3.1 `paths.py` [A]
`SRC, MPC_ROOT, ROOT, MIT_DIR (= MPC_ROOT/"models"/"mit_humanoid"), LOG_DIR (= MPC_ROOT/"logs")` — same
parent-walk pattern as `MPC/src/affine/paths.py`.

### 3.2 `config.py` + `mit_controller.yaml` [A]
`cfg = config.load(path=None)` → nested attribute access mirroring yaml keys exactly
(`cfg.timing.cycle`, `cfg.mpc.walking.state_weight_diag`, `cfg.swing.natural_frequency`, `cfg.contact_manager.contact_ramp_duration`, …)
plus derived: `cfg.dt` (0.002), `cfg.dt_mpc` (0.02), `cfg.N` (25), `cfg.stance_frac` (0.66),
`cfg.half_stance_offset(speed)` (0.37 / 0.28 / 0.26 rule), `cfg.gravity` (−9.81).
`port:` section: `spawn_keyframe: stand`, `skip_leg_pd_init: true`, `armature: true`, `solver: osqp`.

### 3.3 `mit_model.py` [A]
```python
LEFT, RIGHT = 0, 1
@dataclass class ReducedBody: mass: float; com_offset_B: (3,); inertia_B: (3,3)   # yaw-aligned frame, spec 04 §1
@dataclass class LegDyn: Jv: (3,5); Jw: (3,5); JvDot_qd: (3,); M: (5,5); bias: (5,)
@dataclass class RobotState:
    t; torso_pos_W (3); torso_quat_W (4, wxyz); R_WT (3,3); roll; pitch; yaw_wrapped; yaw_unwrapped
    torso_vel_W (3); torso_angvel_W (3)             # mj_objectVelocity(BODY, local=0) semantics, spec 04 §2
    q_leg: (2,5); qd_leg: (2,5); q_arm: (2,4); qd_arm: (2,4)
    foot_pos_W: (2,3); foot_vel_W: (2,3); foot_R_W: (2,3,3)   # site
    foot_contact: (2,) bool; foot_normal_force: (2,) float   # sum |normal| of contacts on the foot body geoms
    reduced: ReducedBody
    def x0(self, g=-9.81) -> (13,)                          # spec 05 §0.2 / 04 §3
class MitModel:
    def __init__(self, armature=True)                       # loads scene.xml via VFS; builds name→index maps
    m, d
    leg_qadr: (2,5) int; leg_dof: (2,5) int; leg_act: (2,5) int; arm_qadr/arm_dof/arm_act: (2,4) int
    base_body; foot_body: (2,); foot_site: (2,); foot_geoms: list[list[int]]
    def reset(self, key="stand")                            # mj_resetDataKeyframe + mj_forward; resets yaw unwrap
    def read_state(self) -> RobotState                      # uses current d (no mj_forward); updates reduced body every call
    def leg_dynamics(self, leg: int) -> LegDyn              # D4
    def standing_jacobians(self) -> tuple[(6,10), (6,10)]   # rows [L foot; R foot], cols [L leg 5 | R leg 5]
    def write_torque(self, tau_leg: (2,5), tau_arm: (2,4))  # scatter to actuators by index, clamp to ctrlrange, d.ctrl
```
Also `fsm.py` [A] (LocomotionFSM per spec 05 §4, including the double update per tick if cheap; braking logic
may be ported but is inert for mode `walking`) and `command.py` [A]:
```python
@dataclass class UserCommand: x_dot=0; y_dot=0; psi_dot=0; body_height_offset_m=0; standing_roll_offset_rad=0; standing_pitch_offset_rad=0
class CommandFilter: __init__(cfg); reset(); update(raw: UserCommand, dt, zero_motion=False) -> UserCommand   # spec 05 §3.2
class CommandSchedule: __init__(profile)  # e.g. {"x_dot": [(t0, v0), (t1, v1)...]} step/ramp; __call__(t) -> UserCommand (raw)
```
And `01_check_model.py` [A]: prints model/state numbers vs spec 01 (masses, reduced body at `stand`, site positions,
x0 at spawn) and **proves D4** (full-model leg dynamics == MjSpec fixed-base leg model: max |ΔJ|, |ΔM|, |Δbias|, |ΔJvDot qd|
at 3 random configurations with random velocities).

### 3.4 `gait.py` [B]
```python
class HorizonClock: __init__(t, cfg); reset(t); sync(t); tk(k) -> float          # t0 = cycle origin (spec 03/05)
class GaitScheduler:
    __init__(cfg); set_mode(mode: str)                      # "walking" | "standing"
    phase(leg, t) -> float; contact(leg, t) -> bool; both_stance(t) -> bool
    remaining_swing_time(leg, t) -> float                   # clamp(cycle*(1-p),0,swing)
    horizon_steps(clock, override=None) -> list[StepContact] # len N
@dataclass class StepContact: stance: (2,) bool; min_scale: (2,) float
@dataclass class OverrideStep: enabled: bool; stance: (2,) bool; min_scale: (2,) float
```
### 3.5 `planner.py` [B]
```python
class BodyTarget:   # spec 05 §3.3
    update(x0, dt, mode, foot_pos_W, filtered_cmd, cfg); nominal_position_W; position_W; nominal_height_W; euler_W; initialized
def half_stance_preview_time(speed, cfg) -> float
def swing_foot_yaw_target_world(yaw_unwrapped, psi_dot, speed, cfg) -> float
def swing_foot_yaw_psi_offset(leg, psi_dot) -> float
def yaw_with_psi_offset(psi_dot, leg, base_yaw) -> float    # liftAngleNear(base+offset, base)
def lift_angle_near(a, ref) -> float
class SwingFootPlanner:   # spec 03 §2 (touchdown target, frozen per swing, stop branch, latch)
    __init__(cfg); reset(); set_body_yaw_target(yaw)
    desired_foot_positions(state, x0, clock, gait, filtered_cmd, t) -> (2,3)
class SwingFootTrajectory:   # spec 03 §3, 05 §7 (cubic smoothstep, z two half-blends)
    reset(p_init, p_final, height, T); set_final_position(p); advance(dt); deactivate(); active: bool
    position() -> (3,); velocity() -> (3,); acceleration() -> (3,)
```
### 3.6 `contact_manager.py` [B]
```python
class ContactManager:   # spec 05 §6 (full, incl. early contact; late contact ported but disabled by cfg)
    __init__(cfg); reset(state, gait, mode, t); update(state, gait, mode, nominal_desired (2,3), t)
    managed_foot_positions(nominal (2,3)) -> (2,3); build_horizon_override() -> list[OverrideStep]
    active_contact: (2,) bool; ramp_alpha: (2,) float; early: (2,) bool; search_mode: (2,) bool; estimated: (2,) bool
```
### 3.7 `mpc.py` [C]
```python
@dataclass class RefOut: X_ref: (N,13); psi: (N,); r: (N,2,3); tk: (N,)
def build_reference(filtered_cmd, seed_x (13,), desired_feet (2,3), clock, cfg) -> RefOut      # spec 03 §1 / 05 §8.2
def mpc_foot_yaw(state, leg, active, touchdown_yaw) -> float                                     # spec 05 §8.1
class ConvexMPC:
    __init__(cfg, solver="osqp")
    solve(x0 (13,), ref: RefOut, steps: list[StepContact], foot_yaw (2,), reduced: ReducedBody, mode) -> (u0 (12,), info: dict)
    # info: status, iters, solve_ms, cold (bool), fallback (bool), U (N,12)
    # builds A_c/B_c per step, ZOH, lifted A_qp/B_qp, yaw-rotated cost, C_F rows rotated by foot yaw, swing eq rows,
    # OSQP warm/cold logic, shifted warm start, fallback Fz = mass*|g|/nStance on active legs (caller supplies which)
```
### 3.8 `leg_control.py` [C]
```python
def stance_torque(dyn: LegDyn, F_W (3,), M_W (3,)) -> (5,)            # Jv^T F + Jw^T M (no bias)
def stance_yaw_hold_moment_z(state, leg, dyn, touchdown_yaw, cfg) -> float
def swing_kp(dyn, cfg) -> (3,3)                                        # diag(wn^2) * diag(Lambda)
def swing_attitude_torque(state, leg, dyn, touchdown_yaw, cfg) -> (5,) # roll off, pitch 300/18, yaw 305/18
def swing_torque(dyn, state, leg, p_des, v_des, a_des, touchdown_yaw, cfg) -> (5,)
def standing_torque(Jv (6,10), Jw (6,10), u (12,)) -> (2,5)            # Jv^T(-F) + Jw^T(-M)
def arm_pd(state, q_des (2,4), cfg) -> (2,4)
```
### 3.9 `controller.py`, `10_walk.py`, `11_gait_quality.py` [integration phase]
`MitController(model, cfg).tick() -> (tau_leg (2,5), tau_arm (2,4))` implementing spec 05 §2's 17-step order;
`10_walk.py`: `headless(vx, vy, wz, seconds, **flags)` / `view(...)`, CLI `--vx --vy --wz --seconds --view --solver --noarmature --log`,
prints a summary; logs via an `MPCLog`-like npz/csv into `MPC/logs/mit_*`. `11_gait_quality.py`: MIT metrics
(tracking, pitch/roll σ, foot slip, swing apex, touchdown timing error, early-contact rate, QP stats) over variant
strings with a process Pool (18 cores available; set BLAS threads to 1 per process).

## 4. Unit checks each implementer must provide (`02_unit_checks.py`, one section per owner)
- [A] reduced body at `stand` equals spec 01 §4 numbers (mass 14.227468, com_offset ≈ [0.0153,0.0029,0.1166] for the
  arm pose, inertia ≈ …) ; x0 at spawn; yaw unwrap across ±π; D4 equivalence.
- [B] phase/contact tables for one cycle match spec 03 (left swing [0.08,0.25), right [0.33,0.5)); horizon steps from
  the cycle origin; swing trajectory endpoints/apex/zero velocities + finite-difference derivative check; planner target
  for a hand-computed case (x_dot 0.6, psi_dot 0) = spec formula; contact manager early-contact sequence and ramp 0,0.2,…,1.
- [C] spec 02 worked example (if present) reproduced to 1e-9; standing QP at `stand` gives Fz_L+Fz_R ≈ mass·9.81 and
  nearly equal split; constraint satisfaction of the solution; OSQP vs quadprog agree within OSQP tolerance; stance
  torque sign check (pushing down on the ground → knee torque sign as expected); swing Kp values ≈ spec 04 sample.
