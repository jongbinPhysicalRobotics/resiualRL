"""MitController — reference MyController::runController 한 틱 (spec 05 §2) 을 그대로 조립한다.

ref: My_Controller/src/My_Controller.cpp
    initializeRuntimeObjects  :400-474     prepareController        :476-486 (FSM update #1)
    resetSwingState           :534-553     applyLocomotionOutput    :555-577
    syncLocomotionFSM         :579-604     updateFilteredUserCommand:610-654
    updateBodyTarget          :677-736     activeContactForSide     :738-753
    updateSwingTrajectories   :792-868     swingFootYawTargetWorld  :891-909
    maybeUpdateMpc            :911-1055    writeStandingLegCommands :1254-1295
    writeLegCommands          :1297-1372   runController            :1374-1429
    RobotRunner::run (arm PD every tick, RR:113-117)

틱 순서 (spec 05 §2; 07 §2 루프: tick() 은 mj_step 이 남긴 d 를 mj_forward 없이 읽는다):
    state = read_state()                         (SR:409-412: 축소 몸체 + cheater state + yaw unwrap)
    raw   = schedule(t)                          (SR:413-415)
    [초기화 틱]  initializeController            (MC:1375-1377)
    [그 외]      FSM update #1  (prepareController, SR:416)
     1. FSM update #2 (runController 머리)       (첫 틱엔 이것 하나뿐)
     2. horizonClock.sync(t)
     3. x0
     4. dt = max(0, t - last_control_time)      (첫 틱 / resetSwingState 틱 → 0)
     5. command filter (dt)
     6. body target (x0, dt)
     7. planner.set_body_yaw_target(euler_W[2])
     8. nominal desired feet: standing = 측정 발 (z −0.005) / walking = swing planner
     9. contact manager update
    10. managed foot positions
    11. swing trajectories (touchdownYaw latch, reset/advance, search 경로; last_control_time = t)
    14. MPC (iteration == 0 또는 7 틱마다) — 풀이는 같은 틱의 다리 명령에 바로 쓰인다
    15. 다리 명령 (walking: stance wrench × alpha + yaw hold / swing OSC + 자세; standing: 6×10 Jacobian)
    17. ++iteration
    arm PD (kp 100, kd 5 → qpos0 + arm_joint_offsets)
(12, 13, 16 은 디버그 전용이라 생략. U1 마커 흉내는 flag.)

플래그 (기본값 = 충실한 포팅):
    solver           None → cfg.port.solver ("osqp")  | "quadprog"
    requested_mode   None → cfg.requested_locomotion_mode ("walking") | "standing" (서기 bring-up: FSM 이 Standing 에 머묾)
    emulate_markers  False (U1: reference 씬의 디버그 마커 mocap 질량을 축소 몸체에 흉내; model 이 emulate_debug_markers=True 여야 함)
    fsm_double_update True (O2 그대로)
    friction_scale   1.0 (변형: mpc.friction_coefficient 에 곱함 — 기본 아님)
    osqp_settings    None (변형: OSQP 설정 덮어쓰기, 예 {"adaptive_rho_interval": 50})
    fixes            () — reference 특이점 고침 (변형, 기본 없음 = 충실한 포팅). 9/25 충실한 포팅이 세 목표를 못 넘어서 시험:
        "slide"      지평 접촉표를 cycle 원점이 아니라 지금부터 (tk = t + k·dt) — O1
        "yawref"     walking 에서 MPC yaw 참조를 측정 yaw 가 아니라 body target yaw (명령 적분) 로 — 방향 유지 (O4)
        "yawanchor"  swing planner 의 몸 yaw 를 명령 적분 대신 측정 yaw 로 — 흘러간 방향을 따라감 (O4 의 반대 선택)
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from command import CommandFilter, UserCommand
from contact_manager import ContactManager
from fsm import LocomotionFSM, LocomotionState
from gait import GaitScheduler, HorizonClock
from leg_control import arm_pd, standing_torque, swing_torque, walking_stance_torque
from mit_model import LEFT, RIGHT, Rz
from mpc import ConvexMPC, build_reference, mpc_foot_yaw
from planner import (BodyTarget, SwingFootPlanner, SwingFootTrajectory, planar_speed, standing_foot_targets,
                     swing_foot_yaw_psi_offset, swing_foot_yaw_target_world, yaw_with_psi_offset)

LEGS = (LEFT, RIGHT)


@dataclass
class LegRuntime:
    """ref: My_Controller.h:52-59 LegRuntimeState."""
    traj: SwingFootTrajectory = field(default_factory=SwingFootTrajectory)
    touchdown_yaw: float = 0.0
    was_in_stance: bool = True
    was_search: bool = False


def _rpy_to_quat(r, p, y) -> np.ndarray:
    """ref: My_Controller.cpp:77-84 (q_yaw * q_pitch * q_roll), wxyz."""
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    q = np.array([cy * cp * cr + sy * sp * sr,
                  cy * cp * sr - sy * sp * cr,
                  cy * sp * cr + sy * cp * sr,
                  sy * cp * cr - cy * sp * sr])
    return q / np.linalg.norm(q)


class MitController:
    def __init__(self, model, cfg, schedule, *, solver: str | None = None, requested_mode: str | None = None,
                 emulate_markers: bool = False, fsm_double_update: bool = True, friction_scale: float = 1.0,
                 osqp_settings: dict | None = None, verbose: bool = False, fixes=()):
        self.fixes = set(fixes or ())
        unknown = self.fixes - {"slide", "yawref", "yawanchor"}
        if unknown:
            raise ValueError(f"unknown fixes {sorted(unknown)}")
        self.model = model
        self.cfg = cfg
        self.schedule = schedule
        self.verbose = verbose
        self.fsm_double_update = bool(fsm_double_update)
        self.requested_mode = requested_mode if requested_mode is not None else cfg.requested_locomotion_mode
        if friction_scale != 1.0:
            cfg = copy.deepcopy(cfg)
            cfg.mpc.friction_coefficient = float(cfg.mpc.friction_coefficient) * float(friction_scale)
            self.cfg = cfg
        self.solver_name = str(solver if solver is not None else cfg.port.solver)
        self.osqp_settings = dict(osqp_settings or {})             # 변형 전용 (기본 = reference 설정, mpc.py)
        self.g = cfg.gravity
        self.n_between = cfg.iterations_between_solve
        self.min_remaining = float(cfg.swing.min_remaining_time)
        self.swing_height = float(cfg.swing.height)
        self.search_T = float(cfg.contact_manager.ground_search_tracking_time)
        m = model.m
        self.q_arm_des = (m.qpos0[model.arm_qadr] +
                          np.asarray(cfg.initial_pose.arm_joint_offsets, dtype=float)[None, :])   # ArmPosInitializer end
        self.emulate_markers = bool(emulate_markers)
        if self.emulate_markers:
            if not model.emulate_debug_markers:
                raise ValueError("emulate_markers needs MitModel(emulate_debug_markers=True)")
            self._marker_pose = None
            model.debug_marker_fn = lambda: self._marker_pose
        self.initialized = False
        # 진단 / 로그 (C++ 에 없음)
        self.last_info = None
        self.solve_log = []          # MPC 틱마다 dict
        self.tick_info = {}

    # ------------------------------------------------------------------------------------------------------------
    # 초기화 (MC:400-474)
    # ------------------------------------------------------------------------------------------------------------
    def _initialize(self, state, raw: UserCommand):
        cfg = self.cfg
        t = state.t
        self.clock = HorizonClock(t, cfg)
        self.gait = GaitScheduler(cfg, self.clock)
        self.cm = ContactManager(cfg)
        self.planner = SwingFootPlanner(cfg)
        self.mpc = ConvexMPC(cfg, solver=self.solver_name, osqp_settings=self.osqp_settings or None,
                             verbose_errors=self.verbose)
        tr = cfg.locomotion_transition
        self.fsm = LocomotionFSM(self.requested_mode, cfg.startup.post_init_standing_settle_time,
                                 tr.braking_settle_speed_threshold, tr.braking_settle_yaw_rate_threshold,
                                 tr.braking_settle_average_window, tr.braking_settle_hold_ticks,
                                 tr.braking_timeout_seconds, tr.braking_touchdown_count, t)
        self.filter = CommandFilter(cfg)
        out = self.fsm.output()
        self.mode = out.mode
        self.gait.set_mode(self.mode)
        self.cm.reset(state, self.gait, self.mode, t)
        self.zero_motion = False                                   # _zeroMotionCommand 초기값 (MC:472)
        self.filtered = self.filter.update(raw, 0.0, self.zero_motion)   # updateFilteredUserCommand(0) (MC:448)
        self.legs = [LegRuntime() for _ in LEGS]
        for leg in LEGS:
            rt = self.legs[leg]
            rt.was_in_stance = bool(self.gait.contact(leg, t))
            rt.was_search = False
            rt.touchdown_yaw = self._swing_yaw_target(state) + swing_foot_yaw_psi_offset(leg, self.filtered.psi_dot)
        self.u_hold = np.zeros(12)                                 # _stanceWrenchWorld
        self.iteration = 0
        self.last_mpc_iteration = 0
        self.last_toggle = 0
        self.last_control_time = t
        self.body = BodyTarget()
        self.initialized = True
        self.walk_start_time = None if self.mode == "standing" else t
        self.transitions = [(t, out.state.value)]

    # ------------------------------------------------------------------------------------------------------------
    # FSM (MC:534-604)
    # ------------------------------------------------------------------------------------------------------------
    def _reset_swing_state(self, state, t):
        """ref: MC:534-553."""
        for leg in LEGS:
            rt = self.legs[leg]
            rt.traj.deactivate()
            rt.was_in_stance = bool(self.gait.contact(leg, t))
            rt.was_search = False
        self.planner.reset()
        self.cm.reset(state, self.gait, self.mode, t)
        self.last_control_time = t

    def _sync_fsm(self, state, raw: UserCommand):
        """syncLocomotionFSM + applyLocomotionOutput (MC:555-604)."""
        t = state.t
        req = int(raw.locomotion_mode_toggle_request)
        while self.last_toggle < req:
            self.fsm.request_toggle()
            self.last_toggle += 1
        R_TW = state.R_WT.T
        v_B = R_TW @ state.com_vel_W()
        w_B = R_TW @ state.torso_angvel_W
        out = self.fsm.update(t, float(v_B[0]), float(v_B[1]), float(w_B[2]))
        self.mode = out.mode
        self.gait.set_mode(self.mode)
        if out.reset_gait_clock:
            self.clock.reset(t)
        if out.reset_swing_state:
            self._reset_swing_state(state, t)
        if out.just_transitioned:
            self.transitions.append((t, out.state.value))
            if out.state == LocomotionState.WALKING and self.walk_start_time is None:
                self.walk_start_time = t
            if self.verbose:
                print(f"[LocomotionFSM] switched to {out.state.value} ({out.mode}) at t={t:.3f}")
        self.zero_motion = out.zero_motion_command
        self.fsm_out = out

    # ------------------------------------------------------------------------------------------------------------
    def _active(self, leg: int, t: float) -> bool:
        """activeContactForSide (MC:738-746)."""
        if self.mode == "walking":
            return bool(self.cm.active_contact[leg])
        return bool(self.gait.contact(leg, t))

    def _alpha(self, leg: int) -> float:
        """contactRampAlphaForSide (MC:748-753)."""
        if self.mode == "walking":
            return float(self.cm.ramp_alpha[leg])
        return 1.0

    def _swing_yaw_target(self, state) -> float:
        """swingFootYawTargetWorld() (MC:891-904) — 측정 yaw + lead·psi_dot·preview."""
        f = self.filtered
        return swing_foot_yaw_target_world(state.yaw_unwrapped, f.psi_dot, planar_speed(f.x_dot, f.y_dot), self.cfg)

    # ------------------------------------------------------------------------------------------------------------
    # swing trajectories (MC:792-868, spec 05 §7)
    # ------------------------------------------------------------------------------------------------------------
    def _update_swing_trajectories(self, state, desired, t):
        dt = max(0.0, t - self.last_control_time)
        psi_dot = self.filtered.psi_dot
        for leg in LEGS:
            rt = self.legs[leg]
            if self._active(leg, t):
                if self.fsm.state == LocomotionState.BRAKING_TO_STANDING and not rt.was_in_stance:
                    self.fsm.register_braking_touchdown()
                rt.traj.deactivate()
                rt.was_in_stance = True
                rt.was_search = False
                continue
            p_now = state.foot_pos_W[leg]
            target = desired[leg]
            fallback_yaw = self._swing_yaw_target(state)
            if self.cm.search_mode[leg]:
                T = max(self.search_T, self.min_remaining)
                if not rt.was_search or not rt.traj.active:
                    rt.touchdown_yaw = yaw_with_psi_offset(psi_dot, leg, fallback_yaw)
                    rt.traj.reset(p_now, target, 0.0, T)
                else:
                    rt.traj.set_final_position(target)
                    rt.traj.advance(dt)
                rt.was_in_stance = False
                rt.was_search = True
                continue
            rt.was_search = False
            T = max(self.gait.remaining_swing_time(leg, t), self.min_remaining)
            if rt.was_in_stance or not rt.traj.active:
                rt.touchdown_yaw = yaw_with_psi_offset(psi_dot, leg, fallback_yaw)
                rt.traj.reset(p_now, target, self.swing_height, T)
            else:
                rt.traj.set_final_position(target)
                rt.traj.advance(dt)
            rt.was_in_stance = False
        self.last_control_time = t

    # ------------------------------------------------------------------------------------------------------------
    # MPC (MC:911-1055)
    # ------------------------------------------------------------------------------------------------------------
    def _maybe_update_mpc(self, state, x0, desired, t):
        if not (self.iteration == 0 or (self.iteration - self.last_mpc_iteration) >= self.n_between):
            return None
        active = np.array([self._active(leg, t) for leg in LEGS])
        info = None
        try:
            foot_yaw = np.array([mpc_foot_yaw(state, leg, bool(active[leg]), self.legs[leg].touchdown_yaw)
                                 for leg in LEGS])
            override = self.cm.build_horizon_override() if self.mode == "walking" else None
            steps = self.gait.horizon_steps(self.clock, override, t_now=t if "slide" in self.fixes else None)
            seed = x0.copy()
            if self.mode == "standing":
                seed[0:3] = self.body.euler_W
                seed[3:6] = self.body.nominal_position_W
                seed[5] = self.body.nominal_height_W
            else:
                seed[0] = self.body.euler_W[0]
                seed[1] = self.body.euler_W[1]
                seed[5] = self.body.nominal_height_W
                if "yawref" in self.fixes:                         # 변형: 방향 유지 (reference 는 x0 yaw)
                    seed[2] = self.body.euler_W[2]
            ref = build_reference(self.filtered.copy(), seed, desired, self.clock, self.cfg)
            u0, info = self.mpc.solve(x0, ref, steps, foot_yaw, state.reduced, self.mode, active=active)
            info["x_ref0"] = ref.X_ref[0].copy()
            info["stance0"] = np.asarray(steps[0].stance, dtype=bool).copy()
            info["stance_tab"] = np.array([s.stance for s in steps], dtype=bool)
            info["foot_yaw"] = foot_yaw
        except Exception as exc:                                   # MC:1016-1052 (setup 단계 예외 포함)
            if self.verbose:
                print(f"[MPC] maybeUpdateMpc failed at iteration {self.iteration}, t={t:.3f}: {exc}")
            u0 = self.mpc.fallback_wrench(active, float(state.reduced.mass))
            info = dict(status="exception", fallback=True, cold=False, solve_ms=np.nan, total_ms=np.nan,
                        iters=0, error=str(exc), x_ref0=np.full(13, np.nan), stance0=active.copy())
        self.u_hold = np.asarray(u0, dtype=float).copy()
        self.last_mpc_iteration = self.iteration
        info["t"] = t
        info["iteration"] = self.iteration
        info["mode"] = self.mode
        self.last_info = info
        return info

    # ------------------------------------------------------------------------------------------------------------
    # 다리 명령 (MC:1254-1372)
    # ------------------------------------------------------------------------------------------------------------
    def _leg_commands(self, state, t) -> np.ndarray:
        cfg = self.cfg
        tau = np.zeros((2, 5))
        if self.mode == "standing":
            Jv, Jw = self.model.standing_jacobians()
            return standing_torque(Jv, Jw, self.u_hold)
        for leg in LEGS:
            rt = self.legs[leg]
            dyn = self.model.leg_dynamics(leg)
            if self._active(leg, t):
                tau[leg] = walking_stance_torque(dyn, state, leg, self.u_hold, self._alpha(leg), rt.touchdown_yaw, cfg)
            else:
                tau[leg] = swing_torque(dyn, state, leg, rt.traj.position(), rt.traj.velocity(),
                                        rt.traj.acceleration(), rt.touchdown_yaw, cfg)
        return tau

    # ------------------------------------------------------------------------------------------------------------
    def tick(self):
        """한 컨트롤 틱. 반환 (tau_leg (2,5), tau_arm (2,4)) — clamp 전 (write_torque 가 ctrlrange 로 자름)."""
        model, cfg = self.model, self.cfg
        state = model.read_state()
        t = state.t
        raw = self.schedule(t)
        self.raw = raw
        if not self.initialized:
            self._initialize(state, raw)                           # prepareController 는 init 전이면 no-op
        elif self.fsm_double_update:
            self._sync_fsm(state, raw)                             # FSM update #1 (prepareController, SR:416)
        # ---- runController (MC:1374-1429)
        self._sync_fsm(state, raw)                                 # 1
        self.clock.sync(t)                                         # 2
        x0 = state.x0(self.g)                                      # 3
        dt = max(0.0, t - self.last_control_time)                  # 4
        self.filtered = self.filter.update(raw, dt, self.zero_motion)   # 5
        self.body.update(x0, dt, self.mode, state.foot_pos_W, self.filtered, cfg,       # 6
                         com_offset_B=state.reduced.com_offset_B, gait=self.gait, t=t)
        if self.body.initialized:                                  # 7
            self.planner.set_body_yaw_target(state.yaw_unwrapped if "yawanchor" in self.fixes else self.body.euler_W[2])
        if self.mode == "standing":                                # 8
            nominal = standing_foot_targets(state.foot_pos_W)
        else:
            nominal = self.planner.desired_foot_positions(state, x0, self.clock, self.gait, self.filtered, t)
        self.cm.update(state, self.gait, self.mode, nominal, t)   # 9
        desired = self.cm.managed_foot_positions(nominal)          # 10
        self._update_swing_trajectories(state, desired, t)         # 11
        info = self._maybe_update_mpc(state, x0, desired, t)       # 14
        tau_leg = self._leg_commands(state, t)                     # 15
        self.iteration += 1                                        # 17
        tau_arm = arm_pd(state, self.q_arm_des, cfg)               # RobotRunner arm PD (RR:113-117)

        if self.emulate_markers:                                   # SR:440 updateDebugVisualization (다음 틱에 반영)
            q_torso = state.torso_quat_W.copy()
            self._marker_pose = {
                "debug_reduced_body_com": (x0[3:6].copy(), q_torso),
                "debug_body_target": (self.body.position_W.copy(), _rpy_to_quat(*self.body.euler_W)),
                "debug_left_touchdown_target": (desired[LEFT].copy(), _rpy_to_quat(0, 0, self.legs[LEFT].touchdown_yaw)),
                "debug_right_touchdown_target": (desired[RIGHT].copy(), _rpy_to_quat(0, 0, self.legs[RIGHT].touchdown_yaw)),
            }

        # 진단 (읽기 전용)
        self.state = state
        self.x0 = x0
        self.desired = desired
        self.nominal = nominal
        self.tick_info = dict(t=t, dt=dt, info=info)
        return tau_leg, tau_arm

    # 편의 ---------------------------------------------------------------------------------------------------
    def swing_pdes(self) -> np.ndarray:
        """(2,3) 스윙 궤적 목표 (stance 다리는 NaN)."""
        out = np.full((2, 3), np.nan)
        for leg in LEGS:
            if not self._active(leg, self.state.t):
                out[leg] = self.legs[leg].traj.position()
        return out

    def heading_R(self) -> np.ndarray:
        return Rz(self.state.yaw_unwrapped)
