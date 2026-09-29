
"""""
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
"""""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from command import CommandFilter, UserCommand
from contact_manager import ContactManager
from fsm import LocomotionFSM, LocomotionState
from gait import GaitScheduler, HorizonClock, cfg_get
from leg_control import arm_pd, standing_torque, stance_yaw_hold_moment_z, swing_torque, walking_stance_torque
from mit_model import LEFT, RIGHT, Rz
from mpc import build_reference, mpc_foot_yaw
from mpc_casadi import CasadiCondensedMPC
from planner import (BodyTarget, SwingFootPlanner, SwingFootTrajectory, half_stance_preview_time, planar_speed,
                     standing_foot_targets,
                     swing_foot_yaw_psi_offset, swing_foot_yaw_target_world, yaw_with_psi_offset)

LEGS = (LEFT, RIGHT)


@dataclass
class LegRuntime:
    """ref: My_Controller.h:52-59 LegRuntimeState."""
    traj: SwingFootTrajectory = field(default_factory=SwingFootTrajectory)
    touchdown_yaw: float = 0.0
    was_in_stance: bool = True
    was_search: bool = False
    yaw_lo: float = math.nan      # 변형 yawtraj (Q&A 9/29 Q1): 이륙 순간 발 yaw · 시각 · yaw 궤적 길이
    t_lo: float = math.nan
    T_yaw: float = math.nan


def _rpy_to_quat(r, p, y) -> np.ndarray:
    "roll pitch yaw to quaternion (x, y, z, w)"
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    q = np.array([cy * cp * cr + sy * sp * sr,
                  cy * cp * sr - sy * sp * cr,
                  cy * sp * cr + sy * cp * sr,
                  sy * cp * cr - cy * sp * sr])
    return q / np.linalg.norm(q)


class _StandDyn:
    def __init__(self, Jw):
        self.Jw = Jw


class MitController:
    def __init__(self, model, cfg, schedule, *, solver: str | None = None, requested_mode: str | None = None,
                 emulate_markers: bool = False, fsm_double_update: bool = True, friction_scale: float = 1.0,
                 osqp_settings: dict | None = None, verbose: bool = False, fixes=(), foot_model: str = "ref",
                 stand_yaw_hold: bool = True, stand_bias: bool = True, side_swap: bool = False,
                 raibert_gain: float = 0.0, ypos_gain: float = 0.0, ypath: float = 0.0, arm_swing: float = 0.0,
                 xpath: float = 0.0):
        self.fixes = set(fixes or ())
        self.foot_model = str(foot_model)
        self.raibert_gain = float(raibert_gain)
        self.ypos_gain = float(ypos_gain)            # 착지 옆 위치 되먹임 (9/27 Q2)
        self.ypath = float(ypath)                    # MPC 참조 옆 위치를 경로선 쪽으로 (0 = reference 재고정, 1 = 경로선) (Q&A 9/27 Q2)
        self.arm_swing = float(arm_swing)            # 팔 반대 흔들기 k [rad/m] (0 = reference 고정 팔) (Q&A 9/28):
                                                     # 어깨 pitch = ±k·(x_R − x_L) — 스윙 다리의 수직축 각운동량을 팔이 상쇄
        self.xpath = float(xpath)                    # 변형 (Q&A 9/29 Q4): MPC 참조 앞뒤 위치를 경로선 쪽으로 (뒤처진 만큼, ±5 cm)
        self.lon_err = 0.0
        self.lat_err = 0.0
        self.path_W = None                           # 명령 경로선 (걷기 시작 CoM 에서 명령 속도·목표 yaw 로 적분)
        self.wb_dz = None                            # 변형 wbcom (Q&A 9/29 Q9): 몸통 CoM − 전신 CoM 높이 (첫 틱)
        self.side_swap = bool(side_swap)             # 걸음 위상 좌우 맞바꿈 (Q&A 9/26 Q7: 하중 쏠림은 위상 0 다리를 따라간다)
        self.stand_bias = bool(stand_bias)           # 서기에서 다리 자중 (qfrc_bias) 보상 — Q&A 9/26 Q6:
                                                     # 순수 Jᵀ wrench 는 다리 자중을 안 보상해 발이 평행하게 안으로 크리프 (60 s 에 65 mm)
        self.stand_yaw_hold = bool(stand_yaw_hold)   # 서기에서도 발 yaw holdQ&A 9/26 Q6:
                                                     # 서기는 관절 PD 가 없어 hip yaw 가 자유 내부 자유도 → 60 s 에 발 뒤꿈치가 140 mm 모임)
        unknown = self.fixes - {"slide", "yawref", "yawanchor", "foot2"}
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

    #초기화
    def _initialize(self, state, raw: UserCommand):
        cfg = self.cfg
        t = state.t
        self.clock = HorizonClock(t, cfg)
        self.gait = GaitScheduler(cfg, self.clock, side_swap=self.side_swap)
        self.cm = ContactManager(cfg)
        self.planner = SwingFootPlanner(cfg, raibert_gain=self.raibert_gain, ypos_gain=self.ypos_gain)
        import planner as _planner_mod                             # 변형 psioff (Q&A 9/29 Q6)
        _planner_mod.PSI_OFFSET_SCALE = float(cfg_get(cfg, "swing.psi_offset_scale", 1.0))
        self.mpc = CasadiCondensedMPC(cfg, solver=self.solver_name, osqp_settings=self.osqp_settings or None,
                                      verbose_errors=self.verbose, foot_model=self.foot_model)
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
        self.last_ref = None
        self.iteration = 0
        self.last_mpc_iteration = 0
        self.last_toggle = 0
        self.last_control_time = t
        self.body = BodyTarget()
        self.initialized = True
        self.walk_start_time = None if self.mode == "standing" else t
        self.transitions = [(t, out.state.value)]

    #FSM
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


    #swing trajectory
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
                    self._mark_liftoff_yaw(rt, state, leg, t, T)
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
                self._mark_liftoff_yaw(rt, state, leg, t, T)
            else:
                rt.traj.set_final_position(target)
                rt.traj.advance(dt)
            rt.was_in_stance = False
        self.last_control_time = t


    # MPC
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
            dz = self.wb_dz or 0.0                                 # 변형 wbcom: 전신 CoM 은 몸통 CoM 보다 dz 낮다
            if self.mode == "standing":
                seed[0:3] = self.body.euler_W
                seed[3:6] = self.body.nominal_position_W
                seed[5] = self.body.nominal_height_W - dz
            else:
                seed[0] = self.body.euler_W[0]
                seed[1] = self.body.euler_W[1]
                seed[5] = self.body.nominal_height_W - dz
                if "yawref" in self.fixes:                         # 변형: 방향 유지 (reference 는 x0 yaw)
                    seed[2] = self.body.euler_W[2]
                if self.xpath > 0.0:                               # 변형: 앞뒤 위치 참조를 경로선 쪽으로 (Q&A 9/29 Q4)
                    psi_t = float(self.body.euler_W[2])
                    pull = min(max(-self.xpath * self.lon_err, -0.05), 0.05)   # 뒤처지면 (+) 앞으로, 최대 5 cm
                    seed[3] += pull * np.cos(psi_t)
                    seed[4] += pull * np.sin(psi_t)
                if self.ypath > 0.0:                               # 변형: 옆 위치 참조를 경로선 쪽으로 (Q&A 9/27 Q2)
                    psi_t = float(self.body.euler_W[2])
                    seed[3] -= self.ypath * self.lat_err * (-np.sin(psi_t))
                    seed[4] -= self.ypath * self.lat_err * np.cos(psi_t)
            ref = build_reference(self.filtered.copy(), seed, desired, self.clock, self.cfg)
            if "foot2" in self.fixes and self.mode == "walking":      # 변형: 지평 안 두 번째 착지 (Q&A 9/28 Q10)
                foot_yaw = self._second_footholds(ref, steps, foot_yaw)
            self.last_ref = ref                                # 뷰어 오버레이용 (10_walk._draw_refs)
            red = state.reduced
            ms = float(cfg_get(self.cfg, "mpc.srb_mass_scale", 1.0) or 1.0)   # 변형 srbm (Q&A 9/29 Q7): MPC 질량만 배율
            if ms != 1.0:
                import dataclasses as _dc
                red = _dc.replace(red, mass=float(red.mass) * ms)
            if cfg_get(self.cfg, "mpc.whole_body_com", 0.0):      # 변형 wbcom: 질량 = 전체 질량
                import dataclasses as _dc
                red = _dc.replace(red, mass=float(self.model.total_mass()))
            u0, info = self.mpc.solve(x0, ref, steps, foot_yaw, red, self.mode, active=active)
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

    def _mark_liftoff_yaw(self, rt, state, leg, t, T):
        """변형 yawtraj: 스윙 시작 순간의 발 yaw · 시각 · 궤적 길이 (스윙 시간 × 비율) 를 기억."""
        frac = cfg_get(self.cfg, "swing.yaw_trajectory_fraction", None)
        if not frac:
            return
        R = state.foot_R_W[leg]
        rt.yaw_lo = float(np.arctan2(R[1, 0], R[0, 0]))
        rt.t_lo = float(t)
        rt.T_yaw = max(1e-3, float(frac) * float(T))

    def _swing_yaw_ref(self, rt, t):
        """변형 yawtraj (Q&A 9/29 Q1): 착지 yaw 를 한 번에 목표로 주는 대신 최소 저크 곡선으로 보간.
        s = (t − t_lo)/T_yaw ∈ [0,1], ψ_des = ψ_lo + Δ·(10s³ − 15s⁴ + 6s⁵), ψ̇_des = Δ·(30s² − 60s³ + 30s⁴)/T_yaw,
        Δ = wrap(ψ_td − ψ_lo). 꺼져 있으면 (None, 0) → reference (목표 = ψ_td, 속도 목표 0)."""
        if not cfg_get(self.cfg, "swing.yaw_trajectory_fraction", None) or not np.isfinite(rt.yaw_lo):
            return None, 0.0
        s = min(max((float(t) - rt.t_lo) / rt.T_yaw, 0.0), 1.0)
        ff = int(cfg_get(self.cfg, "swing.yaw_trajectory_ff", 1))     # yawff: 0 위치만, 1 속도 목표 − 실측 몸통 ω_z, 2 − 명령 ψ̇
        dpsi = (rt.touchdown_yaw - rt.yaw_lo + np.pi) % (2.0 * np.pi) - np.pi
        yaw_des = rt.yaw_lo + dpsi * (10 * s**3 - 15 * s**4 + 6 * s**5)
        if not ff:
            return yaw_des, None                    # None → 속도항은 reference 그대로 (다리 상대 속도 감쇠)
        return yaw_des, (dpsi * (30 * s**2 - 60 * s**3 + 30 * s**4) / rt.T_yaw if s < 1.0 else 0.0)

    def _second_footholds(self, ref, steps, foot_yaw):
        """변형 'foot2' (Q&A 9/28 Q10): reference 는 다리당 발 위치 하나를 지평 전체에 쓴다 (RT:65-66).
        지평 (0.5 s) = 한 주기라 디딤 발은 지평 안에서 들렸다가 다시 착지하는데, 그 입각도 지금 위치 (몸 뒤) 로 계산된다.
        각 다리의 입각 구간을 세어 두 번째 이후 구간에 착지점 규칙으로 예측한 위치 · yaw 를 넣는다:
          이륙 스텝 k_lo 의 참조 CoM p, yaw ψ 에서  p + Rz(ψ + ½ψ̇Tp)·v_B·Tp + Rz(ψ + ψ̇·T_sw)·off_B   (planner.touchdown_target 와 같은 식,
          rai · ypos 되먹임은 미래 측정값을 모르므로 뺀다),  yaw = ψ + lead·ψ̇·Tp + psi_offset.
        ref.r 를 제자리에서 바꾸고 (N,2) 발 yaw 를 돌려준다."""
        cfg = self.cfg
        S = np.array([s.stance for s in steps], dtype=bool)
        n = S.shape[0]
        f = self.filtered
        v_B = np.array([float(f.x_dot), float(f.y_dot), 0.0])
        psi_dot = float(f.psi_dot)
        Tp = half_stance_preview_time(planar_speed(f.x_dot, f.y_dot), cfg)
        T_sw = float(cfg.timing.swing)
        yaw_tab = np.tile(np.asarray(foot_yaw, dtype=float), (n, 1))
        for leg in LEGS:
            seg = 0 if S[0, leg] else -1          # 입각 구간 번호: 지금 입각이면 0 (지금 위치), 스윙이면 첫 착지가 0 (계획 착지점)
            k_lo = None
            for k in range(1, n):
                if S[k - 1, leg] and not S[k, leg]:
                    k_lo = k
                if S[k, leg] and not S[k - 1, leg]:
                    seg += 1
                if seg >= 1 and S[k, leg] and k_lo is not None:
                    if not S[k - 1, leg]:        # 새 입각 시작 → 예측 착지점 계산
                        p = ref.X_ref[max(k_lo - 1, 0), 3:6]
                        psi = float(ref.psi[max(k_lo - 1, 0)])
                        off = self.planner.nominal_offsets_B[leg]
                        ky = cfg_get(cfg, "swing.lateral_half_stance_offset", None)
                        Tp_y = Tp if ky is None else max(0.0, (0.5 + float(ky)) * float(cfg.timing.stance))
                        step = Rz(psi + 0.5 * psi_dot * Tp) @ np.array([v_B[0] * Tp, v_B[1] * Tp_y, 0.0])
                        foot = p + step + Rz(psi + psi_dot * T_sw) @ off
                        foot[2] = ref.r[0, leg, 2] + ref.X_ref[0, 5]              # 지면 높이 = 지금 목표 발 z
                        yaw2 = yaw_with_psi_offset(psi_dot, leg, swing_foot_yaw_target_world(
                            psi, psi_dot, planar_speed(f.x_dot, f.y_dot), cfg))
                    ref.r[k, leg] = foot - ref.X_ref[k, 3:6]
                    yaw_tab[k, leg] = yaw2
        return yaw_tab

    # ------------------------------------------------------------------------------------------------------------
    # 다리 명령 (MC:1254-1372)
    # ------------------------------------------------------------------------------------------------------------
    def _leg_commands(self, state, t) -> np.ndarray:
        cfg = self.cfg
        tau = np.zeros((2, 5))
        if self.mode == "standing":
            Jv, Jw = self.model.standing_jacobians()
            u_st = self.u_hold
            # 서기 수정 (Q6) 은 '서기가 요청된' 경우만 — 걷기 전 1 s StandingSettle 에는 안 건다 (Q&A 9/27 Q1:
            # 걸면 걷기 시작 자세가 바뀌어 경계에 있는 0.6 m/s 가 20 s 에 전도)
            stand_fix = self.requested_mode == "standing"
            if self.stand_yaw_hold and stand_fix:      # 걷기의 stance yaw hold (7.6) 를 서기에도 (Q&A 9/26 Q6)
                u_st = np.asarray(u_st, dtype=float).copy()
                Jw_a = np.asarray(Jw)
                for leg, col, sl in ((LEFT, 8, slice(0, 5)), (RIGHT, 11, slice(5, 10))):
                    shim = _StandDyn(Jw_a[3 * leg:3 * leg + 3, sl])
                    # standing_torque 는 m = −u 로 적용 → 적용 모멘트에 +m_z 를 더하려면 u 에서 뺀다
                    u_st[col] -= stance_yaw_hold_moment_z(state, leg, shim, self.legs[leg].touchdown_yaw, cfg)
            tau = standing_torque(Jv, Jw, u_st)
            if self.stand_bias and stand_fix:
                for leg in LEGS:
                    tau[leg] += self.model.leg_dynamics(leg).bias
            return tau
        for leg in LEGS:
            rt = self.legs[leg]
            dyn = self.model.leg_dynamics(leg)
            if self._active(leg, t):
                tau[leg] = walking_stance_torque(dyn, state, leg, self.u_hold, self._alpha(leg), rt.touchdown_yaw, cfg)
            else:
                yaw_des, yaw_rate_des = self._swing_yaw_ref(rt, t)
                ff_mode = int(cfg_get(cfg, "swing.yaw_trajectory_ff", 1))
                torso_ref = float(self.filtered.psi_dot) if ff_mode == 2 else float(state.torso_angvel_W[2])
                tau[leg] = swing_torque(dyn, state, leg, rt.traj.position(), rt.traj.velocity(),
                                        rt.traj.acceleration(), rt.touchdown_yaw, cfg,
                                        yaw_des=yaw_des, yaw_rate_des=yaw_rate_des,
                                        torso_wz=torso_ref)   # yawff=2: 몸통 회전은 명령 ψ̇ 로 (Q&A 9/29 Q1)
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
        if cfg_get(cfg, "mpc.whole_body_com", 0.0):               # 변형 wbcom (Q&A 9/29 Q9): 병진 상태 = 전신 CoM (다리 43 % 포함)
            import mujoco as _mj                                   # — m·a = ΣF + mg 가 정확. 자세 · 각속도 · 관성은 몸통 그대로 (근사)
            _mj.mj_subtreeVel(model.m, model.d)
            if self.wb_dz is None:                                 # 높이 참조 (몸통 CoM 기준) 를 옮길 상수: 첫 틱의 높이 차
                self.wb_dz = float(x0[5] - model.d.subtree_com[0][2])
            x0 = x0.copy()
            x0[3:6] = model.d.subtree_com[0]
            x0[9:12] = model.d.subtree_linvel[0]
        dt = max(0.0, t - self.last_control_time)                  # 4
        self.filtered = self.filter.update(raw, dt, self.zero_motion)   # 5
        self.body.update(x0, dt, self.mode, state.foot_pos_W, self.filtered, cfg,       # 6
                         com_offset_B=state.reduced.com_offset_B, gait=self.gait, t=t)
        if self.body.initialized:                                  # 7
            self.planner.set_body_yaw_target(state.yaw_unwrapped if "yawanchor" in self.fixes else self.body.euler_W[2])
        if self.mode == "standing":                                # 8
            nominal = standing_foot_targets(state.foot_pos_W)
            self.path_W = None
        else:
            # 명령 경로선 + CoM 옆 이탈 (Q&A 9/27 Q2, ypos 용): 경로 += Rz(ψ_목표)·[ẋ, ẏ]·dt
            psi_t = float(self.body.euler_W[2])
            c_, s_ = np.cos(psi_t), np.sin(psi_t)
            if self.path_W is None:
                self.path_W = x0[3:5].copy()
            else:
                vx_, vy_ = float(self.filtered.x_dot), float(self.filtered.y_dot)
                self.path_W = self.path_W + dt * np.array([c_ * vx_ - s_ * vy_, s_ * vx_ + c_ * vy_])
            dxy = x0[3:5] - self.path_W
            self.lat_err = float(-s_ * dxy[0] + c_ * dxy[1])      # heading 왼쪽 성분 (+ = 경로선 왼쪽)
            self.lon_err = float(c_ * dxy[0] + s_ * dxy[1])       # heading 앞 성분 (+ = 경로선보다 앞)
            self.planner.lat_err = self.lat_err
            nominal = self.planner.desired_foot_positions(state, x0, self.clock, self.gait, self.filtered, t)
        self.cm.update(state, self.gait, self.mode, nominal, t)   # 9
        desired = self.cm.managed_foot_positions(nominal)          # 10
        self._update_swing_trajectories(state, desired, t)         # 11
        info = self._maybe_update_mpc(state, x0, desired, t)       # 14
        tau_leg = self._leg_commands(state, t)                     # 15
        self.iteration += 1                                        # 17
        q_arm_des, qd_arm_des = self.q_arm_des, None
        if self.arm_swing != 0.0 and self.mode == "walking":     # 변형: 팔 반대 흔들기 (Q&A 9/28, k<0 = 같은 쪽)
            ex = np.array([np.cos(state.yaw_unwrapped), np.sin(state.yaw_unwrapped), 0.0])
            dx = float(ex.dot(state.foot_pos_W[RIGHT] - state.foot_pos_W[LEFT]))    # 오른발이 앞 = +
            dxd = float(ex.dot(state.foot_vel_W[RIGHT] - state.foot_vel_W[LEFT]))
            q_arm_des = self.q_arm_des.copy()
            qd_arm_des = np.zeros_like(q_arm_des)
            for leg, sgn in ((RIGHT, 1.0), (LEFT, -1.0)):          # 어깨 pitch + = 팔 뒤로 (축 +y)
                q_arm_des[leg, 0] += sgn * float(np.clip(self.arm_swing * dx, -0.6, 0.6))
                qd_arm_des[leg, 0] = sgn * self.arm_swing * dxd
        tau_arm = arm_pd(state, q_arm_des, cfg, qd_arm_des)       # RobotRunner arm PD (RR:113-117)

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
