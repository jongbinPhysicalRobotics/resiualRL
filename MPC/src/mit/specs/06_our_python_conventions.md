# 06 — Our existing Python MPC codebase: conventions, reuse map, and G1-specific traps

Purpose: let an implementer write `MPC/src/mit/` (a new, self-contained folder) in the same
conventions as `MPC/src/baseline/`, reusing what is robot-agnostic and replacing what is
G1-specific. Everything below is from the code as of 2026-09-25 (paths relative to
`C:\Users\백종빈\Desktop\4-2\residual RL`). Line numbers refer to `MPC/src/baseline/*.py`
unless another folder is named. The `affine/` and `split/` folders hold copies of the same
modules; line numbers there differ (e.g. `affine/mpc_qp.py` has ~60 extra lines).

Key rule of the port (from the task): the MIT folder must reproduce the **reference C++
controller**, so wherever this document says "ours does X, reference does Y", the port does Y.
This document is about *conventions and plumbing*; the reference's equations are in the other
spec files in this folder.

---

## 1. Folder / file conventions

### 1.1 Layout and self-containment

```
residual RL/                      ROOT   (paths.py:19)
├── MPC/                          MPC_ROOT (paths.py:18)  — logs/ lives here
│   ├── CODE_MAP.md, MPC_NOTES.md, PHYSICS_MAP.md, VERIFY_BUNDLE.md
│   ├── logs/                     all npz/csv/png logs, every folder writes here
│   └── src/
│       ├── README.md             folder table (src/README.md:6-11)
│       ├── baseline/             original controller + analysis scripts 01..22, 25
│       ├── affine/               affine-term MPC (23), copies of common modules
│       ├── split/                MPC/swing split (24), on hold
│       └── mit/                  <-- NEW: MIT humanoid port (to be created)
├── mujoco_menagerie/unitree_g1/  G1 model (paths.py:21-24)
├── mit_humanoid_mjcf/            converted MIT model (mit_humanoid.xml, scene.xml, meshes_v3/)
├── reference/                    C++ reference (read-only)
└── Q&A/                          dated study log; README.md is the index
```

Rules (src/README.md:3-4, 21-22):
- **Each folder runs from its own files only** (plus numpy/mujoco/scipy/quadprog). Shared modules
  (`paths.py`, `g1_model.py`, `mpc_srb.py`, `mpc_qp.py`, `gait.py`, `mpc_log.py`, `viewer_hud.py`,
  `17_knee_geometry.py`, `18_gait_quality.py`, `walk_cli.py`) exist as **copies** in each folder.
  A fix in one folder does not propagate — this is accepted and documented.
- Therefore `MPC/src/mit/` gets its own copies (or MIT rewrites) of every module it imports;
  it must not `import` from `../baseline`.
- Scripts add their own folder to `sys.path` and import siblings by bare name:
  `sys.path.insert(0, str(Path(__file__).resolve().parent))` (18_gait_quality.py:25,
  17_knee_geometry.py:20-21, plot_log.py:21). Runner scripts (`09_walk.py`) rely on being
  executed from their folder path (python puts the script dir on sys.path).
- Numbered scripts (`09_walk.py`, `18_gait_quality.py`) start with digits, so they are imported
  with `importlib.import_module("09_walk")` (18_gait_quality.py:27-28, 17_knee_geometry.py:75,
  23_walk_affine.py). Keep the numbering scheme: `NN_name.py`, next free numbers are 27+ (26 is
  `affine/26_push_walk.py`). Suggested for MIT: reuse the *roles* (e.g. `09_walk.py` = main
  runner, `17`/`18` = gait-quality tools) but any new numbers are fine as long as the folder README
  lists them.

### 1.2 `paths.py` pattern (paths.py:15-26)

```python
from pathlib import Path
SRC = Path(__file__).resolve().parent                       # this folder (MPC/src/<name>)
MPC_ROOT = next(p for p in SRC.parents if p.name == "MPC")  # walks up until a folder named MPC
ROOT = MPC_ROOT.parent                                      # residual RL/
MENAGERIE = ROOT / "mujoco_menagerie"; G1_DIR = MENAGERIE / "unitree_g1"
SCENE_XML = G1_DIR / "scene.xml"; G1_XML = G1_DIR / "g1.xml"
TORQUE_SCENE_XML = SRC / "g1_torque_scene.xml"   # a generated per-folder XML lives beside the code
```

For MIT add `MIT_DIR = ROOT / "mit_humanoid_mjcf"`, `MIT_SCENE_XML = MIT_DIR / "scene.xml"`,
`MIT_XML = MIT_DIR / "mit_humanoid.xml"`. Keep `MPC_ROOT` discovery unchanged so logs land in
`MPC/logs/` (mpc_log.py:30-31).

### 1.3 Logs (`mpc_log.py`)

- `MPCLog(prefix, note)`; `prefix` relative to `MPC_ROOT` (`"logs/walk_<tag>"`), absolute allowed
  (mpc_log.py:29-32). `add(**kw)` appends any scalar/vector per call; the key set must be identical
  every call (mpc_log.py:36-39). `save()` writes `<prefix>_YYYYmmdd_HHMMSS.npz` (compressed) and `.csv`
  (utf-8-sig, opens in Excel) and prints the relative path (mpc_log.py:41-72).
- npz keys (mpc_log.py:10-13, written by 09_walk.py:651-658): `t, x(13), x_ref(13), u(12), tau(nu),
  contact(2), foot_z(2), swing_s(2), foot_ref(6, NaN in stance), foot_pos(6), p_land(6), solve_ms,
  violation, ncon` plus `x_labels, u_labels, note`. csv columns: `t, x_<label>*13, ref_<label>*13,
  u_<label>*12, solve_ms, violation` (mpc_log.py:54-58). `X_LABELS`/`U_LABELS` come from
  `mpc_srb` (mpc_srb.py:29-32) — for MIT, if the input ordering is changed to the reference's
  `[F_L, F_R, M_L, M_R]`, change `U_LABELS` in the MIT copy so the csv header stays truthful.
- Log tag naming (09_walk.py:615-621): `walk_{variant}_{inplace|vx0p5}{_wz0p3}` where `variant`
  is a caller-supplied prefix (`"affine_full"`, experiment tags like `sl-lam-sf0p57`,
  09_walk.py:923-927). For MIT use a distinct prefix, e.g. `mit_walk_...`, so `plot_log.py` picks it
  up and it does not mix with G1 logs (MPC/logs has ~550 files already).
- `plot_log.py` (no args = newest npz in MPC/logs, plot_log.py:31-35) makes a 6x2 PNG next to the
  npz. **G1-specific**: torque limits `LIM = [88,139,88,139,50,50]*2` and `tau[:, :12]` (plot_log.py:101,
  115). The MIT copy must use MIT limits from `m.actuator_ctrlrange` and the first 10 actuators.

### 1.4 Globals in the runner (09_walk.py:35-43)

```python
HORIZON = 16          # MPC steps
DT_MPC  = 0.05        # s; N*dt = 0.8 s = one gait cycle
DECIM   = 5           # MPC every 5 physics ticks -> 100 Hz at dt_sim = 2 ms
RAMP_T0, RAMP_T1 = 2.0, 3.0   # v_cmd ramps 0->1 between these times (09_walk.py:235-236)
KP_YAWHOLD, KD_YAWHOLD = 100.0, 10.0
```

Other scripts override them as module attributes: `walk.DECIM = A["decim"]`
(23_walk_affine.py:486-488), `walk.HORIZON, walk.DT_MPC = 25, 0.02` (18_gait_quality.py:74-75,
split/24_walk_split.py:48,203 define `REF_HORIZON, REF_DT = 25, 0.02`). Analysis loops read
`walk.DECIM` (18_gait_quality.py:114, 17_knee_geometry.py:84). Keep this: the MIT runner should
expose `HORIZON`, `DT_MPC`, `DECIM` as module globals, with reference values as defaults
(reference MIT config: `horizon_steps: 25`, `horizon: 0.5` -> dt 0.02, `iterations_between_solve: 7`
at `physics_timestep_sec: 0.002` — see reference/config/mit_humanoid/my_controller.yaml:3-8,20 and
reference/config/simulation.yaml:1).

Physics dt: the G1 scene has 2 ms and the loops assume `m.opt.timestep` everywhere
(`int(seconds / dt)`, `k % DECIM`). The converted MIT XML says `timestep="0.001"`
(mit_humanoid.xml:3) while the reference sim uses 0.002. Set it explicitly in the MIT loader
(`m.opt.timestep = 0.002`) so `DECIM` keeps its meaning.

Single-thread BLAS (09_walk.py:18-22) — must be at the top of the MIT runner before numpy import:
```python
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
```
(Q&A 9/22 Q6: multi-threaded BLAS on ~200x200 matrices caused 100+ ms QP spikes.)

### 1.5 `GAIT_ON` switch (gait.py:21-32)

```python
GAIT_ON = 1                                   # 1 = walk schedule on, 0 = both feet always stance
import os as _os
if _os.environ.get("GAIT_ON") is not None:    # env var wins over file value (per run)
    GAIT_ON = int(_os.environ["GAIT_ON"])
```
`Gait.phase()` returns 0 whenever `not GAIT_ON or t < t_start` (gait.py:45-48), which makes
`in_stance`, `swing_phase`, `contact_table`, `time_to_touchdown` all report double support. CLI
`--nogait` sets `gait.GAIT_ON = 0` at runtime (09_walk.py:928-933, walk_cli.py:17-20). src/README.md:21-22
warns the file value is per folder copy and says all three are `0`; the code as read today differs:
`baseline/gait.py:29 = 1`, `affine/gait.py:29 = 0`, `split/gait.py:29 = 1` (the code wins — the README
is stale). Consequence: running anything in `affine/` without `GAIT_ON=1` in the environment stands
still. The MIT copy of `gait.py` (or its replacement gait scheduler) must keep the same file value +
env override so experiment scripts (`GAIT_ON=1 python ...`) work unchanged; ship it with `GAIT_ON = 1`.

### 1.6 CLI flag style

Two styles coexist:
- **Runners** (`09_walk.py:863-963`, `23_walk_affine.py:471-499`): hand-parsed `sys.argv`, no argparse.
  Boolean = `"--flag" in sys.argv`; valued = `float(sys.argv[sys.argv.index("--flag")+1])`.
  `affine/walk_cli.py` packages this: `_get(argv, key, default, cast)` (walk_cli.py:11-12) and
  `parse()` returning `dict(vx, seconds, legmass, decim, view, common={controller kwargs},
  view_kw={viewer kwargs})` (walk_cli.py:15-82). Sub-runners do `A = walk_cli.parse()`, add their
  own flags, then `walk.view(A["vx"], ctor=ctor, **A["common"], **A["view_kw"])` or
  `walk.headless(vx=..., seconds=..., ctor=ctor, variant=..., **A["common"])` (23_walk_affine.py:472-499).
  Flag names are short and unit-suffixed by convention: `--sidew 13` [cm], `--tddx` [cm],
  `--tdsink` [mm], `--loramp` [ms], `--sf 0.57` (stance fraction), `--cycle 0.8` [s], `--wn 30`, `--zeta 0.7`,
  `--vx`, `--wz`, `--seconds`, `--view`, `--nogait`, `--decim`. Viewer flags: `--nofollow --fast --vsec S
  --timelog F --syncevery N --drawmpc --noboost --lite` (walk_cli.py:64-73).
- **Analysis tools** (`17`, `18`, `20`, `21`): `argparse` with `--vx` (nargs+), `--var` (nargs+ variant
  strings), `--seconds`, `--procs`, run in a `multiprocessing.Pool` (18_gait_quality.py:193-214).

For MIT: write a `walk_cli.py` in the MIT folder with the MIT flag set (many G1 flags such as
`--wzpel`, `--copm`, `--sidew`, `--tdscale` are G1 tuning knobs that the reference does not have;
do not carry them into the MIT port unless the reference config has a counterpart). Keep `--vx`,
`--vy` (new: reference has lateral 0.3), `--wz`, `--seconds`, `--view`, `--nogait`, `--decim`, and the
viewer flags.

### 1.7 `headless()` / `view()` entry points (09_walk.py:579-714, 760-860)

Both: load model -> set initial pose -> `ctl = (ctor or WalkController)(m, d, vx_cmd=..., **kw)` ->
loop. `ctor` injects a subclass (used by 11/23/24) — keep this hook.

Headless loop (09_walk.py:637-683):
```python
for k in range(int(seconds / dt)):
    t = k * dt
    mujoco.mj_forward(m, d)
    if k % DECIM == 0:
        x0, xr0, c0 = ctl.update_mpc(d, t)      # returns (state, x_ref[0], contact[0])
        log.add(...)                            # see 1.3
    d.ctrl[:] = ctl.torque(d, t)
    mujoco.mj_step(m, d)
    ... per-second console row (t, pelvis z, com xy, vx, yaw, yaw err, |mz|, "Sw" contact string, solve_ms)
    if fell is None and d.qpos[2] < z0 - 0.25: fell = t; break
```
Prints: survival line, swing count/apex, yaw summary when `wz != 0`, settled mean vx and CoM-y
amplitude (after `RAMP_T1 + 1` s), QP solve-time stats (mean/median/p99/max + spike list), then
`log.save()`; returns `ok` (bool). `__main__` prints `"... 통과 ✓" / "실패"`.

Viewer loop (09_walk.py:804-856): `mujoco.viewer.launch_passive`, tracking camera on body
`"pelvis"` (09_walk.py:810), `ViewerHUD(m, vx)` (viewer_hud.py:14, also body `"pelvis"` at :20),
sync every `SYNC_EVERY=17` ticks (~29 Hz), real-time sleep, optional Windows process boost
(viewer_hud.py:66-92), `_draw_swing()` overlay of swing reference line + target sphere + landing
sphere (09_walk.py:717-755), optional per-second timing rows saved via `--timelog`.
Everything here is reusable if the body name is parameterised (`"base"` for MIT).

### 1.8 Console encoding / run commands

Run with the venv python from ROOT; when piping output set `PYTHONIOENCODING=utf-8` (cp949 console
crashes on the Korean/✓ summaries). Example (src/README.md:12-18):
```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe MPC/src/mit/09_walk.py --vx 0.6 --seconds 120
```

### 1.9 Project judgement rules (Q&A/README.md rows, memory)

- **Endurance = 120 s**, not 20/40 s (Q&A 9/24 Q1; 40 s A/B is a screening step only).
- **Judge gait shape, not only tracking %** — use the `18_gait_quality` table (pitch, roll σ, toe lift,
  CoP jump, knee separation, foot-yaw slip, pelvis-yaw oscillation, swing-torque overrun) and render.
- Settle window: metrics computed after `settle = 8 s` (18_gait_quality.py:84, 17_knee_geometry.py:73).
- v_cmd ramps 2 -> 3 s (09_walk.py:40); headless stats use t > 4 s (09_walk.py:659).
- Q&A entries are appended per date to `Q&A/<date>.md` and indexed in `Q&A/README.md`.

---

## 2. Reuse map — robot-agnostic vs G1-specific, module by module

### 2.1 `paths.py` — reusable pattern (replace G1 constants with MIT constants).

### 2.2 `g1_model.py`

| Function | Lines | Reusable? |
|---|---|---|
| `_collect_assets()` | 13-24 | **Yes** — generic VFS collector (`{basename: bytes}` for every file under a dir). Rename to take a directory arg. |
| `load(xml_text=None, scene="scene.xml")` | 27-36 | **Yes** with `MIT_DIR`. |
| `to_torque_actuators(m)` | 42-64 | **Yes, generic** (see §5). MIT XML already uses `<motor>`, so it is a no-op there but harmless. |
| `load_torque()` | 67-71 | Yes. |
| `leg_joint_names()` | 77-80 | **G1**: `f"{side}_{j}_joint"`, 6 joints per leg. MIT: `a01_right_hip_yaw ... a10_left_ankle` (5 per leg, right first). |
| `set_crouch(m, d, hip_pitch, knee, ankle_pitch, height)` | 83-114 | **G1**: keyframe `"stand"` (:90), joint names `{side}_hip_pitch_joint` etc. (:97-99), and ground-fitting via **geom group 3 spheres** `d.geom_xpos[g,2] - m.geom_size[g,0]` (:106-107). MIT has no keyframe, no group-3 geoms, and cylinder feet; write `set_initial_pose()` using the reference `initial_pose.leg_joint_offsets = [0, 0, -0.735, 1.2, -0.70]` (hip_yaw, hip_abad, hip_pitch, knee, ankle) and `arm_joint_offsets = [0, 0, 0, -1.65]`, base z from `initial_pose.base_position_W[2] = 0.679472` (my_controller.yaml:141-147) or fit-to-ground using the foot cylinder (lowest point = cylinder axis z − radius, see §4). |

### 2.3 `mpc_srb.py`

| Item | Lines | Reusable? |
|---|---|---|
| `NX=13, NU_PER_FOOT=6, N_FEET=2, NU=12, GRAV=9.81` | 22-26 | Yes. |
| `FOOT_SITES = ("left_foot", "right_foot")` | 27 | **G1 site names**; MIT MJCF has **no sites at all** (`m.nsite == 0`). See §4/§7. |
| `X_LABELS`, `U_LABELS` | 29-32 | Yes (adjust `U_LABELS` if input order changes). |
| `rz(psi)`, `skew(v)`, `quat_to_euler_zyx(quat)` | 35-54 | **Yes, generic.** ZYX Euler: roll=atan2(R21,R22), pitch=−asin(R20), yaw=atan2(R10,R00). |
| `SRBParams` dataclass | 57-68 | Structure reusable; MIT values differ (`w`, `l_t`, `l_h`, `h_sole`, `mu`, `fz_min/max` must come from the reference yaml: `foot_half_length 0.065, foot_half_width 0.01, friction 1.0, normal_force_min 5, max 1500`). |
| `make_params(m, d, ...)` | 71-102 | Inertia part (:78-88) is generic *whole-body* composite inertia about whole-body CoM, yaw removed — **but the reference uses upper-body (legs excluded) mass/inertia recomputed every control tick** (Q&A 9/24 Q12; reference/sim/src/setupRobotParams.cpp:398). Foot-shape part (:90-99) is **G1**: body `"left_ankle_roll_link"`, group-3 sphere geoms, `h_sole = -(min z - radius)`. |
| `get_state(m, d, I_body)` | 108-140 | **Design decision — see §3.** Do not reuse as-is. |
| `get_foot_positions(m, d)` | 143-149 | Generic given site names; MIT needs sites or a body+offset substitute. |
| `continuous_AB(params, psi, r_feet)` | 155-173 | **Yes** — same equations as the reference (reference/My_Controller/src/MPCFormulation.cpp:69-83) except: (a) input block order (ours `[F_L, M_L, F_R, M_R]`, reference `[F_L, F_R, M_L, M_R]`, MPCFormulation.cpp:62-63, 76-81); (b) gravity sign convention: ours `A[11,12] = -1` with state `g = +9.81` (:161, :139); reference `A_c.block<3,1>(9,12) = (0,0,1)` with state `g = -9.81` (MPCFormulation.cpp:73, my_controller.yaml:14 `gravity: -9.81`). Numerically identical; pick one and be consistent in `X_ref[:,12]` and `u_ref`. |
| `discretize(Ac, Bc, dt)` | 176-190 | **Yes.** `A_d = I + A dt + ½A² dt²`, `B_d = (I dt + ½A dt²) B`. The reference adds `(dt³/6) A² B` (MPCFormulation.cpp:19-26), which is exactly zero for this A/B (our docstring :182-183), so both are the same exact ZOH. |

### 2.4 `mpc_qp.py`

| Item | Lines | Reusable? |
|---|---|---|
| `foot_constraints(p, psi)` | 14-51 | **Structure yes (18 rows/foot)**: 4 friction pyramid, 2 Fz bounds, 4 CoP (with `h·F` coupling), 8 Caron yaw-moment rows; rotated into the foot-yaw frame by `C @ Rz(psi)ᵀ` when `psi != 0` (:44-48). For MIT with `w = 0.01` (line foot) the roll-CoP rows almost pin `mx ≈ h·Fy`; the reference has a `contact_wrench_model: full_wrench | no_roll_moment` switch (my_controller.yaml:22) and `torsional_friction_scale 0.0657` — the reference's exact row set is in reference/docs/friction_cop_constraint_c_matrix_build.md and must take precedence over ours (docs vs code: check ConvexMPC.cpp, the code wins). |
| `Q_DEFAULT`, `R_DEFAULT` | 56-61 | **G1 tuning**; MIT uses `state_weight_diag`/`input_weight_diag` from the yaml (walking: `[50000, 9000, 500, 200000, 1009300, 110000, 10,10,10, 100,70,10, 1]`, R = 1e-3 all; standing set differs; my_controller.yaml:24-34). |
| `WrenchMPC.__init__` | 67-91 | Reusable skeleton (`Q_bar`, `R_bar` via `kron`, per-foot `C` blocks). `du_w` (Δu penalty) is our experiment, not in reference. |
| `_solve_qp(H, g)` | 93-109 | quadprog with OSQP fallback (see §6). |
| `gravity_u_ref()` | 111-115 | Generic (Mg/2 on Fz slots; slot indices depend on input order). |
| `solve(...)` (standing, fixed contact) | 117-159 | Generic. |
| `solve_gait(x0, X_ref, psi, foot_pos_traj, contact, u_ref_traj, psi_feet, fz_scale)` | 165-271 | **Mostly reusable**: per-step `B_d[k]` with `r_i = foot_k − CoM_k` (k=0 uses measured CoM, k≥1 the reference CoM, :185-193); swing columns zeroed and **removed from the QP** via `free_mask` (:210-219, 3x faster); constraints per (k, foot) in contact only (:238-257). Differences from reference: reference uses **one A_d per step with `psi_k`** from the reference trajectory (LTV A, MPCFormulation.cpp:64-70,86-90) whereas ours uses a single `psi` for `A` (:180-181) and per-step `B`. Reference input order differs (above). `fz_scale`/`lo_ramp` are our experiment (off by default). |
| `_solve_qp_ineq(H, g, C, d)` | 273-288 | **Yes** — the solver wrapper actually used by `solve_gait`. |
| `_solve_qp_eq(...)` | 290-314 | Legacy (swing vars as equality rows); unused by `solve_gait` now. |
| `wrench_to_tau(m, d, adof, wrenches)` | 320-332 | Generic given site names: `τ = qfrc_bias[adof] − Σ (jacpᵀ F + jacrᵀ m)[adof]`. Note the reference's stance torque has **no** `qfrc_bias` gravity/Coriolis compensation of the legs (Q&A 9/24 Q12 last bullet); follow the reference spec for the port. |
| `actuated_dofs(m)` | 335-336 | **Yes, generic**: `[m.jnt_dofadr[m.actuator_trnid[a,0]] for a in range(m.nu)]`. |

### 2.5 `gait.py`

| Item | Lines | Reusable? |
|---|---|---|
| `GAIT_ON` + env override | 21-32 | **Yes, keep verbatim.** |
| `Gait(T, stance_frac, offsets=(0, 0.5), t_start)` | 35-81 | **Yes, generic** fixed-timing scheduler: `phase_i = ((t−t_start)/T + offset_i) mod 1`, stance if `phase < stance_frac`. MIT reference timing: `cycle 0.5, swing 0.17, stance 0.33` (my_controller.yaml:3-6) -> `T=0.5, stance_frac=0.66`. `contact_table(t, dt, N)` samples at interval midpoints `t + (k+0.5) dt` (:68-75); 09_walk overrides row 0 with "now" (09_walk.py:341-344). The reference's GaitScheduler/ContactManager semantics (early contact, ramp, lock steps) are in the reference spec; ours has no contact manager (`--earlytd` experiment rejected). |
| `raibert_target(...)` | 87-151 | Generic math but **our own heuristic** (capture-point gain `1/ω0`, `anchor`, `min_y_sep`, `ff_scale`, default `z_ground=0.0331` is the G1 site height :89). The reference touchdown planner (SwingFootPlanner: `body_velocity_half_stance_offset 0.37/0.28/0.26` with speed switches, nominal offsets ±0.076 m, capture-point braking only at stop; my_controller.yaml:40-58) is different — implement the reference's, not this. |
| `SwingController` | 157-315 | Trajectory: xy smoothstep, z two-stage smoothstep (`soft_land=True`) or sine arch; impedance `F = kp(p_des−p) + kd(v_des−v)` with optional per-axis `kp_axis/kd_axis` (Λ scheduling), `f_max` clamp, **foot-orientation spring** `kp_ori·(z_f × ẑ) − kd_ori·ω_f` + yaw term (:305-314). For MIT: the trajectory shape must follow the reference SwingFootTrajectory; the orientation spring produces a **roll moment that a 5-DOF leg cannot realise** (no ankle roll) — the reference sets `roll_kp = roll_kd = 0`, `pitch_kp 300/kd 18`, `yaw_kp 305/kd 18` (my_controller.yaml:59-66). Reuse the class only as a code shape; gains/trajectory from the reference. |

### 2.6 `09_walk.py` (controller assembly) — reusable structure, G1-specific details listed in §4.

Reusable: the class shape (`__init__` -> `make_srb_params()` hook :219-221, `get_state()` hook :223-233,
`update_mpc()` at 100 Hz :282-437, `torque()` at 500 Hz :450-556), yaw unwrap
`self._yaw_unwrap += ((x0[2] − unwrap + π) mod 2π) − π` (:288-289; reference has the same policy,
reference/docs/yaw_wrapped_unwrapped_policy_for_mpc.md), `yaw_ref(t)` analytic ramp integral (:268-279),
arc reference for combined vx+wz (:365-373), sway reference `y_ref = c_y + β(y_support − c_y)`, β=0.5
with 0.4 m/s rate limit (:362-397), Λ = (J M⁻¹ Jᵀ)⁻¹ block for swing gains (:501-521), swing
inverse-dynamics feedforward `τ += M q̈` with `q̈ = lstsq(J6, [a_ff; 0])` and `‖q̈‖ ≤ 300` clamp (:529-548),
upper-body joint PD (:551-555), `headless()/view()`. Many of these are **our** heuristics (sway β,
arc, anchor); the reference's BodyMotionReference/ReferenceTrajectory define the MIT versions.

### 2.7 `viewer_hud.py` — `ViewerHUD` (speed = CoM velocity projected on pelvis yaw, 0.8 s average,
real-time factor, tracking %) reusable if the body name (`"pelvis"`, viewer_hud.py:20) and the
averaging window (0.8 s = G1 cycle; MIT cycle 0.5 s) are parameters. `cpu_bench()` and
`boost_process()` are fully generic.

### 2.8 `17_knee_geometry.py` / `18_gait_quality.py` — see §4; the metric definitions are reusable,
the model bindings are not.

### 2.9 `walk_cli.py` (affine) — parsing helpers reusable; flag set is G1-specific (§1.6).

---

## 3. State definition: ours vs the reference (design decision — the port follows the reference)

### 3.1 Ours (`mpc_srb.get_state`, mpc_srb.py:108-140; wrapped by `WalkController.get_state`, 09_walk.py:223-233)

```
x = [roll, pitch, yaw, px, py, pz, wx, wy, wz, vx, vy, vz, g]      (mpc_srb.py:3-4)
roll, pitch, yaw : ZYX Euler of the PELVIS quaternion d.qpos[3:7]          (:119)
p                : WHOLE-BODY CoM  d.subtree_com[0]                          (:120)
ω (world)        : ω = (Rz(ψ) I_body Rz(ψ)ᵀ)⁻¹ · L_whole-body                (:122-126)
                   L = d.subtree_angmom[0] after mj_subtreeVel; I_body = whole-body
                   composite inertia about CoM, computed ONCE at the crouch pose (make_params:78-88)
                   (fallback when I_body is None: pelvis ω = R_pelvis @ d.qvel[3:6], :127-130)
v (world)        : WHOLE-BODY CoM velocity d.subtree_linvel[0]               (:132)
g                : +9.81                                                     (:139)
```
Then 09_walk mixes per-axis pelvis angular velocity into ω in the yaw frame with weights
`w_pel = (wx_pelvis, wy_pelvis, wz_pelvis)` (09_walk.py:227-233; recommended G1 config uses
`--wzpel 1 --wxpel 0.5`). Mass in the SRB = total mass (`mj_getTotalmass`, mpc_srb.py:73).
Rationale and its known cost (Θ from pelvis vs ω from whole-body L are not the same rigid body)
are in Q&A 9/16 Q7-Q8, 9/24 Q14, Q31.

### 3.2 Reference (`MyController::buildCurrentMpcState`, reference/My_Controller/src/My_Controller.cpp:655-674)

```
x0[0:2] = roll, pitch from torsoQuat_W                                    (:661, :667-668)
x0[2]   = yaw_W_unwrapped                                                 (:669)
x0[3:6] = reducedBodyComWorld = torsoPos_W + Rz(yaw_unwrapped) · bodyComLocation   (:316-323, :670)
x0[6:9] = torsoAngVel_W   — the TORSO BODY angular velocity in world      (:671, comment "approximate reduced-body as rigid body")
x0[9:12]= torsoLinVel_W + torsoAngVel_W × (Rz(yaw)·bodyComLocation)       (:325-331, :672)
x0[12]  = config model.gravity = −9.81                                    (:673; my_controller.yaml:14)
```
where the state estimate is read from MuJoCo (cheater state, reference/sim/src/MujocoCheaterStateReader.cpp:339-342):
`torsoPos_W = d.xpos[torso]`, `torsoQuat_W = d.xquat[torso]`, `torsoLinVel_W`/`torsoAngVel_W` from
`mj_objectVelocity(m, d, mjOBJ_BODY, torso, vel, flg_local=0)` (angular = `vel[0:3]`, world frame,
:151-164). `bodyComLocation` = upper-body (legs removed) CoM offset from the torso origin in the
yaw-aligned body frame, and `bodyMass`, `bodyInertia` = **upper-body** mass and inertia about that
CoM in the yaw frame, **recomputed every control tick** from `d.xipos/d.ximat`
(setupRobotParams.cpp:398 `updateReducedBodyMassPropertiesFromData`, called in
SimulationRunner.cpp:409; summarised in Q&A 9/24 Q12). The MPC uses the values at solve time for
the whole horizon and rotates them by each step's reference yaw (MPCFormulation.cpp:66-68).
Torso body name for MIT is `"torso"` in the reference spec (reference/sim/src/models/MitHumanoidSpec.cpp:6);
in our converted MJCF the same body is `"base"` (mit_humanoid.xml:27).

### 3.3 Decision for the port

Implement §3.2 exactly. In MuJoCo-Python terms:

```python
tid = mj_name2id(m, mjOBJ_BODY, "base")
vel = np.zeros(6); mujoco.mj_objectVelocity(m, d, mjOBJ_BODY, tid, vel, 0)   # [ω_W(3), v_W(3)]
omega_W, v_torso_W = vel[:3], vel[3:]
roll, pitch, yaw = quat_to_euler_zyx(d.xquat[tid])        # reuse mpc_srb.quat_to_euler_zyx
# upper-body (non-leg subtree) mass/CoM/inertia from body_mass, xipos, ximat, body_inertia:
#   loop bodies not in either leg subtree; I_W = Σ (R I_b Rᵀ + m (rᵀr I − r rᵀ)) about upper CoM;
#   I_body = Rz(yaw)ᵀ I_W Rz(yaw); bodyComLocation = Rz(yaw)ᵀ (upperCoM_W − d.xpos[tid])
p = d.xpos[tid] + Rz(yaw_unwrapped) @ bodyComLocation
v = v_torso_W + np.cross(omega_W, Rz(yaw_unwrapped) @ bodyComLocation)
x0 = [roll, pitch, yaw_unwrapped, *p, *omega_W, *v, -9.81]
```
Do **not** port `I_body`-normalised angular momentum, `subtree_com`, `subtree_linvel`, or the
`w_pel` mixing. Our `make_params` inertia loop (mpc_srb.py:79-88) is the right code shape for the
upper-body inertia if the body loop is restricted to non-leg bodies and the reference point is the
upper-body CoM. Which bodies are "legs": reference removes the leg subtrees rooted at the hip-yaw
bodies (`right_hip_yaw`, `left_hip_yaw` in our MJCF, each containing hip_abad, upper_leg,
lower_leg, foot). Verify against setupRobotParams.cpp when implementing (it is the reference spec's job
to list the exact subtree rule).

---

## 4. G1-specific assumptions that break on MIT (exact list)

MIT converted-model facts used below (measured by loading `mit_humanoid_mjcf/scene.xml` with
mujoco 3.12): `nq 25, nv 24, nu 18, nbody 22, ngeom 33, nsite 0, nkey 0`, total mass **24.889 kg**,
`timestep 0.001`, `integrator implicitfast`, all `dof_damping/armature/frictionloss = 0`.
Bodies: `world, base, right_hip_yaw, right_hip_abad, right_upper_leg, right_lower_leg, right_foot,
left_hip_yaw, left_hip_abad, left_upper_leg, left_lower_leg, left_foot, right_shoulder, right_shoulder_2,
right_upper_arm, right_lower_arm, right_hand, left_shoulder, ..., left_hand`.
Joints/actuators (same names, `<motor>` with `ctrlrange` = URDF effort): `a01_right_hip_yaw (34 N·m),
a02_right_hip_abad (34), a03_right_hip_pitch (72), a04_right_knee (144), a05_right_ankle (68),
a06..a10 left leg (same limits), a11..a14 right arm (34,34,34,55), a15..a18 left arm`. Actuated dofs are
6..23 in that order (**right leg = dof 6-10, left leg = dof 11-15, arms 16-23**). Foot collision:
one cylinder per foot, `size = [r=0.01, half_len=0.075]`, `pos = (0.03, 0, −0.03)` in the foot body,
axis rotated to the foot x-axis (mit_humanoid.xml:54,82) -> line contact from x = −0.045 to +0.105 and
z = −0.04 (axis −0.03 minus radius) in the foot body frame. Geom groups present: 0 (collision) and 1
(visual meshes); **no group 3**. Feet with all joints at 0: foot body origin z = 0.042 with base at 0.7483.

### 4.1 `09_walk.py`

| Line | Assumption | MIT impact |
|---|---|---|
| 64, 810 (and viewer_hud.py:20) | body `"pelvis"` | MIT body is `"base"` (reference calls it `"torso"`). |
| 104-107, 290, 417-420, 453, 470, 506, 536 | `mpc_srb.FOOT_SITES` = sites `left_foot`/`right_foot`; `mj_jacSite`, `d.site_xpos/site_xmat` | **MIT MJCF has no sites.** Reference expects `"{side}_foot_contact_site"` (MitHumanoidSpec.cpp:10,20) and `foot_end_effector_source: site` (my_controller.yaml:12); site position in the reference's private MJCF is unknown. Options: (a) inject `<site name="left_foot_contact_site" pos="..."/>` into the XML string before `from_xml_string` (the MIT folder owns a modified copy of `mit_humanoid.xml`, like `TORQUE_SCENE_XML`), or (b) use `mj_jac` at a body-fixed point (`readPointLinearVelocity` pattern, MujocoCheaterStateReader.cpp:170-190). Whichever is chosen, `FOOT_SITES` must be replaced by a `(body, local_point)` or site tuple in the MIT copy of `mpc_srb.py`. |
| 159-160, 439-447 | `_fb = "{side}_ankle_roll_link"` for measured foot Fz | MIT: `"{side}_foot"`. |
| 183-184 | `upper_act = actuators with dofadr >= 18` (G1: 6 base + 12 leg dofs) | MIT: `>= 16` (6 + 10); better: build from actuator names `a11..a18`. |
| 205 | `leg_dofs = [range(6,12), range(12,18)]` with index 0 = left | MIT dof order is **right first**: right = 6..10, left = 11..15, 5 each. Build by name so `i=0` stays "left" if `FOOT_SITES`-style ordering `(left, right)` is kept — or adopt the reference's `(Left, Right)` ordering explicitly (MitHumanoidSpec.cpp lists Left first) and map actuator ids by name. |
| 205, 532-548 | 6-DOF leg: `J6 = [jacp; jacr][:, dofs]` is 6x6 square, `lstsq` for `q̈` | MIT: 5 dofs -> 6x5 system; the orientation rows (especially roll) are not realisable. The reference solves the swing feedforward on the **3-D position** task with `Λ = (J M⁻¹ Jᵀ)⁻¹` (3x3) plus separate pitch/yaw joint-space PD (my_controller.yaml:59-66, Q&A 9/23 Q10) — follow that. |
| 501-521 | Λ from `J = jacp[:, dofs]` (3x6) | Works with 3x5 too (Λ still 3x3). Keep. |
| 182, 551-555 | upper PD holds **all** `d.qpos[7:]` joints not in legs at their initial values | MIT: arms only; reference uses `joint_tracking.arm kp 100 / kd 5` and `arm_joint_offsets [0,0,0,−1.65]` (my_controller.yaml:126-129, 146). |
| 591-592, 770-771 | `g1_model.load_torque()`, `g1_model.set_crouch()` | Replace with MIT loader + initial pose (§2.2). |
| 560-576 | `scale_leg_mass` matches G1 link names | Drop or rename (`hip_yaw, hip_abad, upper_leg, lower_leg, foot`). |
| 43, 476-484 | stance foot yaw-hold PD 100/10 | Reference MIT: `stance_yaw_kp 20, kd 4` (my_controller.yaml:65-66). |
| 35-39 | `HORIZON 16, DT_MPC 0.05, DECIM 5` | Reference MIT: 25 steps × 0.02 s, solve every 7 ticks of 2 ms. |
| 96 | `Gait(T=0.8, stance_frac=0.75)`, `--sf 0.57` recommended | Reference: `T 0.5, swing 0.17` -> `stance_frac 0.66`. |
| 681 | fall test `d.qpos[2] < z0 − 0.25` | Fine (base z ≈ 0.68 standing). 18_gait_quality uses an absolute `< 0.5` (18:152) — keep relative. |

### 4.2 `mpc_srb.make_params` (mpc_srb.py:90-99)

`"left_ankle_roll_link"` + `geom_group == 3` spheres -> `l_t = max x`, `l_h = −min x`, `w = min|y|`,
`h_sole = −(min z − r)`. MIT: no such geoms. Use the reference yaml values (`foot_half_length 0.065`,
`foot_half_width 0.01`) about the foot end-effector point; `h_sole` = vertical distance from that point
to the contact line (from the cylinder: axis at z = −0.03, radius 0.01 in the foot body -> sole at
z = −0.04 relative to the foot body origin; adjust for wherever the site is put). Open question §7.

### 4.3 `18_gait_quality.py`

| Line | Assumption | MIT impact |
|---|---|---|
| 26-28 | imports `g1_model`, `mpc_srb`, `09_walk`, `17_knee_geometry` from the same folder | MIT folder supplies its own `mit_model`, `mpc_srb`, runner and knee-geometry module with the same attribute API. |
| 30-48 | variant-string keys `tds, copm, duf, dum, wzp, wxp, wyp, sw, sink, qN, ctl, affmode, early_td, liftoff_fix` -> `WalkController` kwargs; default `side_w=0.13` | All G1 tuning kwargs. MIT: define its own `KEYS` (e.g. `qN`, gains, `hz`) and no `side_w` default. Keep the `"k=v,k=v"` format and `qN=` (Q diagonal override) since the table printer relies on it. |
| 62-81 | `ctl=affine|split` module lookup | MIT: drop or map to MIT controller variants. |
| 87-90 | `g1_model.load_torque(); set_crouch()`; `Ctor(m, d, vx_cmd=vx, kp_up=300, swing_id=True, soft_land=True, lam_swing=True, wn_swing=30, zeta_swing=0.7, stance_frac=0.57, q_py=300, gate_sy=False, **kw)` | The whole "recommended G1 configuration" baked in. MIT: default kwargs = reference config. |
| 91, 138 | `h = ctl.params.h_sole`; commanded CoP `xc = −((−s·mx + c·my) + h(c·Fx + s·Fy)) / Fz` in the yaw frame | Formula generic, but relies on `ctl.params` (SRBParams) and `ctl.wr` (2x6 `[F, m]` per foot, left first) and `ctl._yaw_unwrap`. Keep these attribute names in the MIT controller so the tool ports 1:1. |
| 93-94 | `kg.Geo(m)`; `G.ank` = ankle-roll bodies | see 4.4. |
| 95-100 | toe/heel = **sphere geoms** on the ankle body split by `geom_pos[g][0] > 0`; `rad = sphere radius` | **MIT has one cylinder per foot** -> no geom split possible. Replace with contact-point split: for each contact whose geom belongs to the foot body, transform `con.pos` into the foot body frame and classify by local x relative to the cylinder centre (x > 0.03 toe, else heel); "toe lift" (18:140 `min geom_xpos z − rad`) becomes the height of the cylinder's front end: `p_front = xpos_foot + R_foot @ (0.03+0.075, 0, −0.03)`, minus radius. |
| 101 | `mpc_srb.FOOT_SITES` sites for swing tracking error | MIT: foot end-effector point (§4.1). |
| 102-105 | `leg_act` by actuator name `f"{s}_{j}_joint"`, 6 names, sides `("left","right")`; `lim = actuator_ctrlrange[:,1]` | MIT names: `f"a{n:02d}_{side}_{j}"` with `j ∈ (hip_yaw, hip_abad, hip_pitch, knee, ankle)`, right = a01-a05, left = a06-a10. `ctrlrange` is set (34/34/72/144/68). |
| 114 | `walk.DECIM` | keep as module global in the MIT runner. |
| 116-117, 126, 147-150 | `ctl.update_mpc(d,t)`, `ctl.torque(d,t)`, `ctl.gait.in_stance/swing_phase/T_swing/T_stance`, `ctl.sw[i].target(s, p_land, T_swing)`, `ctl.p_land[i]` | Controller API to preserve (or adapt the tool). |
| 120 | `d.xmat[G.pel]` pelvis for body pitch/roll | `"base"`. |
| 152 | `d.qpos[2] < 0.5` fall | base z ≈ 0.68 -> ok but prefer `z0 − 0.25`. |
| 161 | pelvis-yaw oscillation uses a 0.8 s moving average (one G1 cycle) | MIT cycle 0.5 s. |
| 168 | `nst = int(Tst/dt)`, 30/150 ms windows after touchdown | generic. |
| 203-214 | table header/printer | generic. |

### 4.4 `17_knee_geometry.py` (`Geo`, 17:27-62)

Bindings: bodies `pelvis`, `{s}_hip_pitch_link`, `{s}_knee_link`, `{s}_ankle_roll_link`; joints
`{s}_knee_joint`, `{s}_hip_roll_joint`, `{s}_hip_yaw_joint`; `sgn = [+1, −1]` (left = +y); knee-facing
direction = `(R_knee @ jnt_axis[knee]) × ẑ` yaw (17:53-55). MIT equivalents: `base`,
`{s}_upper_leg` (hip-pitch body), `{s}_lower_leg` (knee body), `{s}_foot`; joints `a04/a09_{s}_knee`,
`a02/a07_{s}_hip_abad` (roll analogue; **sign convention must be re-checked**: MIT `hip_abad` axis
is `1 0 0` in a body already rotated by quat about y, mit_humanoid.xml:36-38), `a01/a06_{s}_hip_yaw`.
What `Geo` needs from any model: (1) three bodies per leg (hip, knee, ankle/foot) for the frontal-plane
valgus line, (2) the knee joint axis for knee-facing direction, (3) hip roll/yaw joint qpos addresses,
(4) the foot body rotation for foot yaw (`last_fy`), (5) the pelvis/base rotation for body yaw
(`last_py`). `run_rl()` (17:101-138) is G1-deployed-RL only — drop for MIT.

### 4.5 `gait.py`, `viewer_hud.py`, `plot_log.py`

- `raibert_target` default `z_ground = 0.0331` (gait.py:89) = G1 foot-site height above ground; always pass explicitly.
- `SwingController.wrench` orientation spring assumes a 6-DOF foot (gait.py:305-314) — 5-DOF MIT: no roll authority (§2.5).
- `ViewerHUD` body `"pelvis"` (viewer_hud.py:20), 0.8 s window.
- `plot_log.py` `LIM` and `tau[:, :12]` (plot_log.py:101,115).

---

## 5. MuJoCo loading with Korean paths, and the torque-actuator conversion

### 5.1 Loading (g1_model.py:13-36) — MuJoCo's C loader cannot open paths containing Korean characters, so the XML and every mesh are read by Python and handed over as a VFS dict keyed by **basename** (MuJoCo's VFS ignores directories and is case-insensitive; the folder must have no basename collisions — true for `mit_humanoid_mjcf/`):

```python
from pathlib import Path
import mujoco
MIT_DIR = ROOT / "mit_humanoid_mjcf"

def _collect_assets(folder: Path) -> dict[str, bytes]:
    return {f.name: f.read_bytes() for f in folder.rglob("*") if f.is_file()}

def load(xml_text: str | None = None, scene: str = "scene.xml"):
    assets = _collect_assets(MIT_DIR)                       # includes mit_humanoid.xml, scene.xml, *.stl
    if xml_text is None:
        xml_text = (MIT_DIR / scene).read_text(encoding="utf-8")
    m = mujoco.MjModel.from_xml_string(xml_text, assets)    # <include file="mit_humanoid.xml"/> and
    return m, mujoco.MjData(m)                              # meshdir="meshes_v3/" resolve through the VFS
```
Verified in this session: `scene.xml` loads this way (mujoco 3.12.0) and reports the numbers in §4.
`scene.xml` uses `<include file="mit_humanoid.xml"/>` (scene.xml:2) and the model uses
`meshdir="meshes_v3/"` (mit_humanoid.xml:2) — both resolve because the VFS lookup is by basename.
If the MIT folder needs to **modify** the XML (add foot sites, set `timestep`, add a keyframe), do it
on `xml_text` (string edit or `xml.etree`) before `from_xml_string`, or keep a modified copy
`MPC/src/mit/mit_torque_scene.xml` and pass its text with the original folder's assets (the
`TORQUE_SCENE_XML` pattern, paths.py:26). Never write into `mit_humanoid_mjcf/` from the controller.

### 5.2 Torque actuators (g1_model.py:42-64)

```python
def to_torque_actuators(m):
    for a in range(m.nu):
        m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_gainprm[a, :] = 0.0; m.actuator_gainprm[a, 0] = 1.0     # force = 1.0 * ctrl
        m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_NONE
        m.actuator_biasprm[a, :] = 0.0
        jid = m.actuator_trnid[a, 0]
        if m.jnt_actfrclimited[jid]:
            m.actuator_ctrlrange[a] = m.jnt_actfrcrange[jid]              # ctrl [N·m] clamped to joint limit
        else:
            m.actuator_ctrllimited[a] = 0
```
G1 (Menagerie) ships `<position kp=500>` actuators, hence the conversion. The MIT MJCF already
declares `<motor ... ctrlrange="-34 34">` etc. (mit_humanoid.xml:144-161) and joints carry
`actuatorfrcrange` (e.g. :34), so `gaintype == FIXED` and `biastype == NONE` already hold
(verified: `{0}`, `{0}`); calling the function is a harmless no-op that keeps the loader API
identical (`load_torque()`).

Torque application (09_walk.py:550, mpc_qp.py:320-336): `tau` is an `nu`-vector in **actuator order**,
computed from an `nv`-vector via `adof = actuated_dofs(m)`; `d.ctrl[:] = tau`. MuJoCo clamps `ctrl`
to `ctrlrange`, so joint torque saturation is enforced by the simulator (18_gait_quality measures
`|tau| / ctrlrange[:,1]` before clamping to report overrun, 18:143-144).

### 5.3 Model-side unknowns in the converted MIT file (mit_humanoid_mjcf/README.md)

Joint damping/armature/frictionloss and contact `solref/solimp` are **not** from the source URDF
and are currently 0/default; the reference's MJCF is private. Physics dt must be forced to 0.002.
These are open questions (§7), not conventions.

---

## 6. QP solver interface

Primary: **quadprog** (`mpc_qp._solve_qp_ineq`, mpc_qp.py:273-288), called by `solve_gait` with the
reduced (stance-only) variables:

```python
# problem:  min ½ uᵀ H u + gᵀ u   s.t.  C u ≤ d          (H symmetric positive definite)
# quadprog: min ½ xᵀ G x − aᵀ x   s.t.  Cqᵀ x ≥ b, first meq rows equality
u, *_ = quadprog.solve_qp(H, -g, -C.T, -d, 0)        # G=H, a=−g, Cq=−Cᵀ, b=−d, meq=0
```
- `H = 2 (B_rᵀ Q̄ B_r + R̄_r)` built with diagonal Q/R as vectors (`q_bar`, `r_bar`, mpc_qp.py:78-79,
  215-217), symmetrised (:231). `R > 0` keeps `H` positive definite, which quadprog requires (it
  factorises `G` by Cholesky; a singular `H` raises).
- `g = 2 B_rᵀ (q ⊙ (A_qp x0 − X_ref)) − 2 r ⊙ u_ref` (:218-219).
- Constraint rows only for `(k, foot)` in contact (:241-257); swing variables are dropped from the
  problem (`free` index, :210-219, :259-262), so no equality rows are needed.
- Info returned: `solve_ms`, `violation = max(C U − d)`, `eq_violation`, `solver` (:264-269).
- Fallback: on any exception, **OSQP** is used with `P = csc(H), q = g, A = csc(C), l = −inf, u = d,
  eps_abs = eps_rel = 1e-7, verbose False` and `res.x` is returned (:279-288; same pattern in
  `_solve_qp` :100-109 and `_solve_qp_eq` :303-314). `osqp 1.1.3` is installed and importable;
  the fallback path is wired in all three folders' `mpc_qp.py` but is only reached on quadprog failure
  (no script uses OSQP as the primary solver; `20_timing_log.py` wraps `quadprog.solve_qp` to time it,
  20:50). Whether the fallback has ever fired is not verified in this session (`info["solver"]` would
  say `"osqp"`).
- No warm start; a fresh dense solve every MPC call (~4 ms median for 108 variables at 100 Hz after the
  swing-variable removal, Q&A 9/22 Q6).

Reference: **OsqpEigen** (OSQP) with warm start (shifted previous solution, `use_shifted_warm_start:
true`, my_controller.yaml:18), `polish false`, cold re-initialisation only when the contact signature
changes (reference/My_Controller/src/ConvexMPC.cpp:891-1026). Both formulations are the same dense
condensed QP; for the port, quadprog (dense, no warm start) is acceptable at N=25 (300 variables when
both feet are in stance, ~150-200 typical — measure `solve_ms`), and OSQP via the existing fallback code
is the drop-in if solve time or infeasibility becomes an issue. If OSQP is used as primary, the yaml's
`normal_force_min: 5` comment ("could this cause MPC solve issues?") hints the reference sometimes hits
infeasibility; quadprog raises `ValueError("constraints are inconsistent")` in that case — catch it and
fall back or reuse the previous `u0` like the reference's failure path (check ConvexMPC.cpp for what it
does on a failed status; the reference spec should say).

---

## 7. MIT-specific facts collected here (for the implementer)

- Reference robot spec for MIT (reference/sim/src/models/MitHumanoidSpec.cpp): torso body `"torso"`; per
  leg foot body `"{side}_foot_link"`, foot site `"{side}_foot_contact_site"`; 5 leg joints in order
  `hip_yaw, hip_abad, hip_pitch, knee, ankle` (joint name `"{side}_{j}_joint"`, actuator `"{side}_{j}"`);
  arms `shoulder_pitch, shoulder_abad, shoulder_yaw, elbow`, arm end body `"{side}_forearm_link"`.
  **None of these names exist in our converted MJCF** (ours: `base`, `{side}_foot`, `a01_right_hip_yaw`…,
  no sites). The MIT folder needs a name-mapping table (or an XML rename pass) in its `mit_model.py`.
- Reference config values (my_controller.yaml): gait `cycle 0.5, swing 0.17, stance 0.33`; MPC
  `25 × 0.02 s`, solve every 7 sim ticks (2 ms) ≈ 14 ms; `friction 1.0`, `foot_half_length 0.065`,
  `foot_half_width 0.01`, `torsional_friction_scale 0.0657`, `Fz ∈ [5, 1500]`; walking
  `Q = [50000, 9000, 500, 200000, 1009300, 110000, 10,10,10, 100,70,10, 1]`, `R = 1e-3`; swing
  `ω_n = [151, 151, 110]`, `kd = [25,25,25]`, `height 0.06`; foot PD `roll 0/0, pitch 300/18, yaw 305/18`,
  stance yaw hold `20/4`; leg joint PD (init/hold) `kp [70,50,100,100,200], kd [15,10,10,10,30]`; arm
  `kp 100, kd 5`; command limits `x_dot_max 0.7, y_dot_max 0.5, psi_dot_max 2.0` with first-order filters
  `tau 0.92/0.8/0.70`; initial pose base z `0.679472`, leg offsets `[0,0,−0.735,1.2,−0.70]`, arm
  `[0,0,0,−1.65]`; `post_init_standing_settle_time 1.0`, `leg_initialization_time 2.0`.
- Converted model: mass 24.889 kg (URDF `humanoid_full_sf.urdf`, "sf" = single-foot line contact);
  leg mass fraction 43.7 % but swing inertia about hip pitch ≈ 0.2 kg·m² per leg, ratio to torso pitch
  inertia 0.49 (G1: 1.74-1.88) — Q&A 9/24 Q29 — which is why the reference's upper-body SRB with no
  leg-momentum correction works on MIT (Q&A 9/24 Q3, Q12).
- 5-DOF legs (no ankle roll): lateral balance must come from hip_abad + CoP along the foot line only;
  the SRB wrench's `mx` is nearly unconstrained in the model only through `w = 0.01`.

---

## 8. Suggested MIT folder skeleton (same conventions)

```
MPC/src/mit/
  README.md            what each file is, run commands, current status
  paths.py             copy + MIT_DIR/MIT_SCENE_XML
  mit_model.py         load/load_torque (VFS), name map, set_initial_pose, upper-body mass props (per tick)
  mpc_srb.py           copy; FOOT_* replaced by end-effector spec; get_state per §3.3; input order per reference
  mpc_qp.py            copy; constraints per reference doc/code; Q/R from yaml
  gait.py              GAIT_ON kept; Gait(T=0.5, sf=0.66); swing trajectory per reference; no raibert_target
  swing_planner.py     reference SwingFootPlanner (touchdown) — new
  contact_manager.py   reference ContactManager (early contact, ramp) — new
  mpc_log.py, viewer_hud.py, walk_cli.py, plot_log.py   copies with body name / limits parameterised
  09_walk.py           MitWalkController + headless()/view(); HORIZON=25, DT_MPC=0.02, DECIM=7 (dt 2 ms)
  17_knee_geometry.py  Geo with MIT bindings
  18_gait_quality.py   MIT KEYS, cylinder-foot toe/heel split, a01..a10 actuator names
```
Deliverable runs (120 s each, logs in MPC/logs with `mit_` prefix): `--vx 0.6`, `--vy 0.3`, `--wz 1.3`.
