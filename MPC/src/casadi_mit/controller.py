
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
        "half"       지평 접촉표 원점을 반 주기 (한 걸음) 마다 옮김 — O1 의 좌우 비대칭 완화 (Q&A 9/29 Q11)
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
from leg_control import (arm_pd, limit_swing_torque, standing_torque, stance_yaw_hold_moment_z, swing_torque, stance_foot_damping,
                         walking_stance_torque)
from mit_model import LEFT, RIGHT, Rz
from mpc import build_reference, mpc_foot_yaw, resolved_foot_yaw
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
    pit_lo: float = math.nan      # 변형 pitchtraj (Q&A 10/6 Q28): 이륙 순간 발 pitch 보정각 · 시각 · 궤적 길이
    t_plo: float = math.nan
    T_pit: float = math.nan


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
        unknown = self.fixes - {"slide", "yawref", "yawanchor", "foot2", "half"}
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
        self.planner = SwingFootPlanner(cfg, raibert_gain=self.raibert_gain, ypos_gain=self.ypos_gain,   # 변형 raimax (Q&A 10/6 Q100): 상한 기본 0.10
                                        raibert_max=float(cfg_get(cfg, "swing.raibert_max", 0.10) or 0.10))
        import planner as _planner_mod                             # 변형 psioff (Q&A 9/29 Q6)
        _planner_mod.PSI_OFFSET_SCALE = float(cfg_get(cfg, "swing.psi_offset_scale", 1.0))
        _planner_mod.K_SCHED_X = None                              # 변형 ksx · ksy (Q&A 10/1 Q20) — 매 틱 _k_schedule 이 설정
        _planner_mod.K_SCHED_Y = None
        self.mpc = CasadiCondensedMPC(cfg, solver=self.solver_name, osqp_settings=self.osqp_settings or None,
                                      verbose_errors=self.verbose, foot_model=self.foot_model)
        self.mpc.total_mass = float(self.model.total_mass())        # 변형 frefm (Q&A 10/6 Q5)
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
        for rt_ in self.legs:                                  # 변형 swprof (Q&A 9/30 Q2): 저가속 수평 스윙 blend
            rt_.traj.xy_profile = float(cfg_get(self.cfg, "swing.xy_profile_ramp", 0.0) or 0.0)
            rt_.traj.blend_order = int(float(cfg_get(self.cfg, "swing.blend_order", 3) or 3))   # 변형 swq (Q&A 10/6 Q3)
            rt_.traj.xy_accel_frac = float(cfg_get(self.cfg, "swing.xy_accel_fraction", 0.0) or 0.0)   # 변형 swasym (Q&A 10/6 Q85)
        for leg in LEGS:
            rt = self.legs[leg]
            rt.was_in_stance = bool(self.gait.contact(leg, t))
            rt.was_search = False
            rt.touchdown_yaw = self._swing_yaw_target(state) + swing_foot_yaw_psi_offset(leg, self.filtered.psi_dot)
        self.u_hold = np.zeros(12)                                 # _stanceWrenchWorld
        self._toe_latch = np.zeros(2, dtype=bool)                  # 변형 toesense (Q&A 10/6 Q83): 이번 디딤에서 발끝 모드로 바뀐 다리
        self._heel_seen = np.zeros(2, dtype=bool)                  #   이번 디딤에서 뒤꿈치가 하중을 받은 적이 있나
        self._toe_td_t = np.full(2, -1.0)                          #   이번 디딤 시작 (active 가 켜진) 시각
        self.heel_off_dbg = [(0.0, 0.0, np.nan, np.nan), (0.0, 0.0, np.nan, np.nan)]   # 변형 htoe: (켜짐, 하한 쪽, φ_ref, φ)
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
                self._mark_liftoff_yaw(rt, state, leg, t, T)
                p0 = p_now
                if self._toe_traj_on() and np.isfinite(rt.pit_lo):   # 변형 toetraj: 높이 궤적을 발끝에서 시작
                    p0 = np.asarray(p_now, dtype=float) - np.array([0.0, 0.0, self._toe_drop(-rt.pit_lo)])
                rt.traj.reset(p0, target, self.swing_height, T)
            else:
                rt.traj.set_final_position(target)
                rt.traj.advance(dt)
            rt.was_in_stance = False
        self.last_control_time = t


    # MPC
    def _lateral_sway_velocity(self, seed, desired, t):
        """변형 latsway (Q&A 10/6 Q37): 옆 역진자 (LIPM) 주기 궤도의 옆 속도를 지평 칸마다 (진행 방향 좌표, 왼쪽 +).
        한 걸음 T = cycle/2 를 '두 발 지지 가운데 → 다음 두 발 지지 가운데' 로 보고, 그 사이 디딤 발 (옆 ±w) 둘레 역진자:
          y(τ) = s·(w − A·cosh(ω(τ − T/2))),  A = w / cosh(ωT/2)  → 걸음 경계에서 가운데 (y = 0) 를 지나는 대칭 궤도 (유도)
          v(τ) = s·A·ω·sinh(ω(T/2 − τ))       (τ = 0 에 디딤 발 쪽으로, τ = T 에 반대쪽으로)
        s = 디딤 발 쪽 (왼 +1 · 오른 −1), ω = √(g/z0), z0 = MPC 높이 참조, w = 두 목표 발 (desired) 옆 거리의 절반.
        시각은 실제 시각 t + k·dt (접촉표 원점과 무관). 크기 배율 = mpc.lateral_sway_scale."""
        sc = float(cfg_get(self.cfg, "mpc.lateral_sway_scale", 0.0) or 0.0)
        cyc, st_T = float(self.clock.cycle), float(self.gait.stance)
        T = 0.5 * cyc
        ds = max(st_T - T, 0.0)                                      # 한 걸음의 두 발 지지 시간
        z0 = max(float(seed[5]), 0.2)
        om = np.sqrt(abs(float(self.mpc.gravity)) / z0)
        psi = float(seed[2]); yv = np.array([-np.sin(psi), np.cos(psi), 0.0])
        d_ = np.asarray(desired, dtype=float).reshape(2, 3)
        w = float(np.clip(0.5 * abs((d_[LEFT] - d_[RIGHT]).dot(yv)), 0.02, 0.2))
        A = w / np.cosh(0.5 * om * T)
        out = np.zeros(self.mpc.N)
        for k in range(self.mpc.N):
            tk = t + k * float(self.clock.dt_mpc)
            for leg, sg in ((LEFT, 1.0), (RIGHT, -1.0)):
                tau = (self.gait.phase(leg, tk, self.clock) * cyc - 0.5 * ds) % cyc
                if tau < T:
                    out[k] = sc * sg * A * om * np.sinh(om * (0.5 * T - tau))
                    break
        return out

    def _update_toe_mode(self, state, active):
        """변형 toesense (Q&A 10/6 Q83): 디딤 발을 '발끝 모드' 로 바꿀지 실제 상태로 정하고, 이륙할 때까지 유지 (latch).
        mode 1 = 발목 각이 한계 근처 (q_ankle ≤ −toe_sense_ankle, 발목 한계 −0.80 rad 쪽 = 정강이가 발 위로 앞으로 기욺, Q68),
        mode 2 = 뒤꿈치가 하중을 받다가 (≥ 20 N) 떨어짐 (< 5 N), mode 3 = 둘 중 하나. 걷기에서만.
        두 모드 모두 '착지 후 60 ms 지남 + 이번 디딤에서 뒤꿈치가 20 N 이상 받은 적 있음' 일 때만 — 첫 시험에서 착지 순간
        발목이 이미 −0.77 이라 착지부터 발끝에 묶여 뒤로 넘어졌다 (Q83).
        발끝 모드인 발은 MPC 가 지금 디딤 구간을 template_toe (뒤꿈치 최소 하중 풀기 · toepin 이면 CoP 를 발끝에 묶음) 로 푼다."""
        mode = int(getattr(self.mpc, "toe_sense_mode", 0))
        thr = float(cfg_get(self.cfg, "mpc.toe_sense_ankle", 0.76) or 0.76)
        heel = np.asarray(getattr(state, "foot_heel_force", np.zeros(2)), dtype=float)
        q = np.asarray(state.q_leg, dtype=float).reshape(2, 5)
        t = float(state.t)
        for leg in LEGS:
            if self.mode != "walking" or not active[leg]:
                self._toe_latch[leg] = False
                self._heel_seen[leg] = False
                self._toe_td_t[leg] = -1.0
                continue
            if self._toe_td_t[leg] < 0.0:
                self._toe_td_t[leg] = t
            if heel[leg] >= 20.0:
                self._heel_seen[leg] = True
            ready = bool(self._heel_seen[leg]) and (t - self._toe_td_t[leg]) >= 0.06
            by_ankle = q[leg, 4] <= -thr
            by_heel = heel[leg] < 5.0
            if ready and ((mode in (1, 3) and by_ankle) or (mode in (2, 3) and by_heel)):
                self._toe_latch[leg] = True
        return self._toe_latch.copy()

    def _tag_stance_fraction(self, steps, t, t_tab, n_lock):
        """변형 toeoff (Q&A 10/6 Q30): 칸마다 디딤 경과 비율 s ∈ [0, 1] 을 steps[k].stance_frac (2,) 에 붙인다.
        접촉표와 같은 시각을 쓴다 — override 칸 (지금 접촉) 은 지금 t, 나머지는 표의 tk.
        스케줄상 스윙인데 딛고 있으면: 스윙 후반 = 일찍 착지 → 0, 스윙 전반 = 늦은 이륙 → 1."""
        cyc, st_T, sw_T = float(self.clock.cycle), float(self.gait.stance), float(self.gait.swing)
        for k, s in enumerate(steps):
            tk = t if k < n_lock else (self.clock.tk(k) if t_tab is None else float(t_tab) + k * float(self.clock.dt_mpc))
            fr = np.zeros(2)
            for leg in LEGS:
                p = self.gait.phase(leg, tk, self.clock) * cyc            # 주기 안 시각 [s], 디딤 [0, st_T)
                fr[leg] = p / st_T if p < st_T else (0.0 if p >= st_T + 0.5 * sw_T else 1.0)
            s.stance_frac = fr

    def _maybe_update_mpc(self, state, x0, desired, t):
        if not (self.iteration == 0 or (self.iteration - self.last_mpc_iteration) >= self.n_between):
            return None
        active = np.array([self._active(leg, t) for leg in LEGS])
        info = None
        try:
            foot_yaw = np.array([mpc_foot_yaw(state, leg, bool(active[leg]), self.legs[leg].touchdown_yaw)
                                 for leg in LEGS])
            override = self.cm.build_horizon_override() if self.mode == "walking" else None
            t_tab = t if "slide" in self.fixes else None
            if "half" in self.fixes and t_tab is None:             # 변형 half (Q&A 9/29 Q11): 접촉표 원점을 반 주기 (한 걸음) 마다 옮김
                half = 0.5 * float(self.clock.cycle)               # — O1 (주기 원점) 은 두 다리의 어긋남이 달라 좌우 비대칭 (9/26 Q7)
                t_tab = self.clock.t0 + (half if t - self.clock.t0 >= half else 0.0)
            steps = self.gait.horizon_steps(self.clock, override, t_now=t_tab)
            if self.mode == "walking" and (cfg_get(self.cfg, "mpc.toe_off_fraction", None)
                                           or cfg_get(self.cfg, "mpc.heel_off_fraction", None)
                                           or cfg_get(self.cfg, "swing.heel_strike_angle", None)):
                self._tag_stance_fraction(steps, t, t_tab, len(override) if override else 0)
            self.mpc.heel_off_on = self._heel_off_on()             # 변형 htoe (Q&A 10/6 Q90): 전진 명령일 때만
            self.mpc.heel_off_s2 = self._heel_off_s2()
            seed = x0.copy()
            dz = self.wb_dz or 0.0                                 # 변형 wbcom: 전신 CoM 은 몸통 CoM 보다 dz 낮다
            dz -= float(cfg_get(self.cfg, "mpc.height_ref_offset", 0.0) or 0.0)   # 변형 zoff (Q&A 10/6 Q6): 높이 참조 + zoff [m]
            if self.mode == "standing":
                seed[0:3] = self.body.euler_W
                seed[3:6] = self.body.nominal_position_W
                seed[5] = self.body.nominal_height_W - dz
            else:
                seed[0] = self.body.euler_W[0]
                seed[1] = self.body.euler_W[1]
                pk = float(cfg_get(self.cfg, "mpc.pitch_ref_per_speed", 0.0) or 0.0)   # 변형 pitchref (Q&A 10/6 Q51): 앞으로 걸을 때
                if pk:                                             # 자연스러운 앞 숙임을 참조로 (pitch + = 앞으로 숙임) — 명령 전진 속도 × pk [rad]
                    seed[1] += pk * float(self.filtered.x_dot)
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
            vlat = None
            if self.mode == "walking" and float(cfg_get(self.cfg, "mpc.lateral_sway_scale", 0.0) or 0.0) > 0.0:
                vlat = self._lateral_sway_velocity(seed, desired, t)   # 변형 latsway (Q&A 10/6 Q37)
            ref = build_reference(self.filtered.copy(), seed, desired, self.clock, self.cfg, vlat_add=vlat)
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
            mz_ref = None
            if self.mode == "walking" and float(cfg_get(self.cfg, "mpc.hold_in_mpc_weight", 0.0) or 0.0) > 0.0:
                mz_ref = self._hold_mz_ref(state, t, active)       # 변형 hmpc (Q&A 9/30 Q14)
            if getattr(self.mpc, "toe_sense_mode", 0):
                self.mpc.toe_now = self._update_toe_mode(state, active)   # 변형 toesense (Q&A 10/6 Q83)
            u0, info = self.mpc.solve(x0, ref, steps, foot_yaw, red, self.mode, active=active, mz_ref=mz_ref)
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
        U_ = info.get("U") if isinstance(info, dict) else None             # 변형 uint (Q&A 9/30 Q13): 다음 스텝 입력 (보간용)
        self._u_next = np.asarray(U_[1], dtype=float).copy() if U_ is not None and len(U_) > 1 else self.u_hold.copy()
        st_tab = info.get("stance_tab") if isinstance(info, dict) else None
        if int(float(cfg_get(self.cfg, "mpc.input_interpolation_mode", 0) or 0)) == 2:
            self._u_next[0:6] = self.u_hold[0:6]                   # uintm=2: 힘은 계단 그대로, 모멘트 (CoP · 비틀림) 만 보간
        if st_tab is not None and len(st_tab) > 1 and int(float(cfg_get(self.cfg, "mpc.input_interpolation_mode", 0) or 0)) == 1:
            # uintm=1 (Q&A 9/30 Q15): 스텝 0 → 1 사이 디딤이 바뀌는 다리는 보간하지 않는다 (체중 이동을 앞당기지 않게)
            for leg, cols in ((0, [0, 1, 2, 6, 7, 8]), (1, [3, 4, 5, 9, 10, 11])):
                if bool(st_tab[0][leg]) != bool(st_tab[1][leg]):
                    self._u_next[cols] = self.u_hold[cols]
        self._t_solve = t
        self.last_mpc_iteration = self.iteration
        info["t"] = t
        info["iteration"] = self.iteration
        info["mode"] = self.mode
        self.last_info = info
        return info

    def _k_schedule(self):
        """변형 ksx · ksy (Q&A 10/1 Q20): 착지 미리보기 k_half 를 명령에 따라 — 곧게 빨리 걸을 땐 전진 k 를 k_빠름 쪽으로,
        옆으로 걸을 땐 옆 k 를 k_옆 쪽으로. 회전 명령이 있으면 (|ψ̇| ≥ w) 둘 다 기본 k0 (걸으며 돌기가 k 에 민감 — 9/30 Q19).
        k = k0 + (k_목표 − k0)·s,  s_x = min(1, |v_x|/0.8)·(1 − t),  s_y = min(1, |v_y|/0.5)·(1 − t),  t = min(1, |ψ̇|/w)."""
        cfg = self.cfg
        kx_f = cfg_get(cfg, "swing.k_sched_fast_x", None)
        ky_l = cfg_get(cfg, "swing.k_sched_lat_y", None)
        if kx_f is None and ky_l is None:
            return
        import planner as _pm
        f = self.filtered
        k0 = float(cfg_get(cfg, "swing.body_velocity_half_stance_offset", 0.37))
        w = float(cfg_get(cfg, "swing.k_sched_turn_rate", 0.5) or 0.5)
        turn = min(1.0, abs(float(f.psi_dot)) / w)
        if kx_f is not None:
            _pm.K_SCHED_X = k0 + (float(kx_f) - k0) * min(1.0, abs(float(f.x_dot)) / 0.8) * (1.0 - turn)
        if ky_l is not None:
            _pm.K_SCHED_Y = k0 + (float(ky_l) - k0) * min(1.0, abs(float(f.y_dot)) / 0.5) * (1.0 - turn)

    def _yaw_hold_gains(self):
        """변형 shw (Q&A 9/30 Q10): 회전 명령 |ψ̇| 에 따라 hold 이득을 reference (20 / 4) 쪽으로. 꺼져 있으면 None (yaml 이득)."""
        cfg = self.cfg
        w_fade = float(cfg_get(cfg, "swing.stance_yaw_fade_rate", 0.0) or 0.0)
        if w_fade <= 0.0:
            return None
        f_ = max(0.0, 1.0 - abs(float(self.filtered.psi_dot)) / w_fade)
        kp_s = float(cfg_get(cfg, "swing.stance_yaw_kp", 20.0))
        kd_s = float(cfg_get(cfg, "swing.stance_yaw_kd", 4.0))
        return (20.0 + f_ * (kp_s - 20.0), 4.0 + f_ * (kd_s - 4.0))

    def _hold_mz_ref(self, state, t, active):
        """변형 hmpc (Q&A 9/30 Q14): 딛는 발 yaw hold 가 원하는 모멘트를 MPC 의 Mz 참조로 (지면 반력 기준 = −m_z, 월드 z).
        MPC 밖에서 더하던 hold 를 MPC 비용으로 옮긴 것 — 해가 선 접촉 제약 안에 머문다. 안 딛는 다리는 nan."""
        out = np.full(2, np.nan)
        gains = self._yaw_hold_gains()
        for leg in LEGS:
            if active[leg]:
                m = stance_yaw_hold_moment_z(state, leg, self.model.leg_dynamics(leg), self.legs[leg].touchdown_yaw,
                                             self.cfg, gains)
                # 착지 램프 α 를 곱하고 (MPC 밖 hold 와 같은 크기), 비틀림 한계 μ·l·m·g 로 자른다 — 착지 순간의
                # 큰 yaw 오차 · 스윙 각속도가 지평 전체 참조로 들어가지 않게.
                lim = float(cfg_get(self.cfg, "mpc.friction_coefficient", 1.0)) * float(cfg_get(self.cfg, "mpc.foot_half_length")) \
                    * float(state.reduced.mass) * 9.81
                out[leg] = float(np.clip(-self._alpha(leg) * m, -lim, lim))
        return out

    def _mark_liftoff_yaw(self, rt, state, leg, t, T):
        """변형 yawtraj: 스윙 시작 순간의 발 yaw · 시각 · 궤적 길이 (스윙 시간 × 비율) 를 기억."""
        pfrac = cfg_get(self.cfg, "swing.pitch_trajectory_fraction", None)   # 변형 pitchtraj (Q&A 10/6 Q28)
        if pfrac:
            Rp = np.asarray(state.foot_R_W[leg], dtype=float)
            fy_, fz_ = Rp[:, 1] / np.linalg.norm(Rp[:, 1]), Rp[:, 2] / np.linalg.norm(Rp[:, 2])
            ez = np.array([0.0, 0.0, 1.0])
            rt.pit_lo = float(np.arctan2(fy_.dot(np.cross(fz_, ez)), fz_.dot(ez)))   # swing_attitude_torque 의 pitch 보정각과 같은 정의
            rt.t_plo = float(t)
            rt.T_pit = max(1e-3, float(pfrac) * float(T))
        frac = cfg_get(self.cfg, "swing.yaw_trajectory_fraction", None)
        if not frac:
            return
        R = state.foot_R_W[leg]
        rt.yaw_lo = float(np.arctan2(R[1, 0], R[0, 0]))
        rt.t_lo = float(t)
        rt.T_yaw = max(1e-3, float(frac) * float(T))

    def _swing_pitch_ref(self, rt, t):
        """변형 pitchtraj (Q&A 10/6 Q28): 발 pitch 를 이륙 순간 기울기에서 수평으로 최소 저크로 옮긴다.
        보정각 목표 e_ref = e_lo·(1 − b(s)), b = 10s³ − 15s⁴ + 6s⁵, s = (t − t_lo)/T_pit. 피치 각속도 목표 = e_lo·b'(s)/T_pit
        (보정각이 줄어드는 쪽으로 발이 돈다). 꺼져 있으면 (0, 0) → 기존 (목표 수평, 속도 목표 0)."""
        if not cfg_get(self.cfg, "swing.pitch_trajectory_fraction", None) or not np.isfinite(rt.pit_lo):
            return 0.0, 0.0
        s = min(max((float(t) - rt.t_plo) / rt.T_pit, 0.0), 1.0)
        # 변형 hstk (Q&A 10/6 Q90): 목표를 수평 대신 발끝 위 θ_hs 로 (보정각 + = 발끝을 내려야 수평) — e_ref = e_lo·(1 − b) + θ_hs·b
        e_end = self._heel_strike_angle() if self._heel_off_on() else 0.0
        bias = rt.pit_lo * (1.0 - (10 * s**3 - 15 * s**4 + 6 * s**5)) + e_end * (10 * s**3 - 15 * s**4 + 6 * s**5)
        rate = (rt.pit_lo - e_end) * (30 * s**2 - 60 * s**3 + 30 * s**4) / rt.T_pit if s < 1.0 else 0.0
        return bias, rate

    def _heel_off_on(self) -> bool:
        """변형 htoe · hstk (Q&A 10/6 Q89 · Q90): 걷기 + 명령 전진 속도 ≥ htoev (기본 0.3 m/s) 일 때만 의도한 뒤꿈치 듦 · 뒤꿈치 착지를
        켠다. 후진 · 옆 · 제자리는 지금 (A41) 그대로."""
        on = float(cfg_get(self.cfg, "mpc.heel_off_fraction", 0.0) or 0.0) > 0.0 or self._heel_strike_angle() > 0.0
        v_min = float(cfg_get(self.cfg, "mpc.heel_off_min_speed", 0.3) or 0.3)
        return self.mode == "walking" and on and float(self.filtered.x_dot) >= v_min

    def _heel_strike_angle(self) -> float:
        """변형 hstk: 착지 때 발끝을 드는 각 θ_hs [rad] (0 = 끔)."""
        return float(cfg_get(self.cfg, "swing.heel_strike_angle", 0.0) or 0.0)

    def _heel_off_s2(self) -> float:
        """변형 htoe: 뒤꿈치 드는 디딤 경과 s2 = htoe − htoes·max(명령 전진 속도 − 0.6, 0), 0.2 아래로는 안 내린다.
        빠를수록 엉덩이가 발목 앞으로 일찍 나가 평평한 발목이 한계에 일찍 닿는다 (Q&A 10/6 Q68 · Q89)."""
        s2 = float(cfg_get(self.cfg, "mpc.heel_off_fraction", 0.0) or 0.0)
        k = float(cfg_get(self.cfg, "mpc.heel_off_speed_slope", 0.0) or 0.0)
        return max(0.2, s2 - k * max(float(self.filtered.x_dot) - 0.6, 0.0)) if s2 > 0.0 else 0.0

    def _stance_frac_now(self, leg, t) -> float:
        """지금 시각의 일정상 디딤 경과 비율 (_tag_stance_fraction 의 지금 접촉 칸과 같은 규칙)."""
        st_T, sw_T = float(self.gait.stance), float(self.gait.swing)
        p = self.gait.phase(leg, t, self.clock) * float(self.clock.cycle)
        return p / st_T if p < st_T else (0.0 if p >= st_T + 0.5 * sw_T else 1.0)

    def _heel_off_ankle_torque(self, state, leg, dyn, t) -> float:
        """변형 htoe · hstk: 발끝 한 점 구간 (디딤 경과 ≥ s2) · 뒤꿈치 한 점 구간 (< s1) 의 발목 토크 — MPC 렌치 (Jᵀ[F; M]) 위에 더한다.
        뒤꿈치 구간은 φ_ref = −θ_hs·(1 − b(경과/s1)) (발끝 위에서 평평하게 내림), 나머지는 아래 (발끝 구간) 와 같다.
        발 pitch φ (+ = 발끝 아래 = 뒤꿈치 듦) 를 궤적 φ_traj = φ_end·b(s), s = (경과 − s2)/(1 − s2) (최소 저크) 로 따라가게 하되,
        발목 q 가 하한 q_floor (기본 −0.75, 관절 한계 −0.80) 아래로 가지 않게 한다. 평평한 발의 발목 = −(정강이 앞 기울기) 라
        φ = q + θ_정강이 → 하한을 지키는 φ 는 φ − q + q_floor. 두 목표 중 큰 쪽을 쓴다 (하한 쪽이면 발목 각 PD).
        발목 + 토크 = 발끝 아래로 돌림 (관절 축 +y, 코드 사실). 구간 밖 · 꺼짐이면 0. self.heel_off_dbg[leg] 에 (켜짐, 하한 쪽, φ_ref, φ)."""
        self.heel_off_dbg[leg] = (0.0, 0.0, np.nan, np.nan)
        if not self._heel_off_on():
            return 0.0
        s2 = self._heel_off_s2()
        th_hs = self._heel_strike_angle()
        s1 = float(cfg_get(self.cfg, "mpc.heel_strike_fraction", 0.1) or 0.1) if th_hs > 0.0 else 0.0
        fr = self._stance_frac_now(leg, t)
        heel_phase = s1 > 0.0 and fr < s1                     # 변형 hstk: 뒤꿈치 한 점 구간 — 발끝 위 θ_hs 에서 평평하게 내림
        if not heel_phase and not (s2 > 0.0 and fr >= s2):
            return 0.0
        R = np.asarray(state.foot_R_W[leg], dtype=float)
        fy_, fz_ = R[:, 1] / np.linalg.norm(R[:, 1]), R[:, 2] / np.linalg.norm(R[:, 2])
        ez = np.array([0.0, 0.0, 1.0])
        phi = -float(np.arctan2(fy_.dot(np.cross(fz_, ez)), fz_.dot(ez)))   # = −(swing_attitude_torque 의 pitch 보정각)
        w_f = np.asarray(state.torso_angvel_W, dtype=float) + np.asarray(dyn.Jw) @ np.asarray(state.qd_leg[leg], dtype=float)
        phid = float(fy_.dot(w_f))
        q_a = float(np.asarray(state.q_leg, dtype=float).reshape(2, 5)[leg, 4])
        qd_a = float(np.asarray(state.qd_leg[leg], dtype=float)[4])
        q_fl = float(cfg_get(self.cfg, "swing.heel_off_ankle_floor", -0.75))
        ph_end = float(cfg_get(self.cfg, "swing.heel_off_pitch_end", 0.2))
        ph_end += float(cfg_get(self.cfg, "swing.heel_off_pitch_speed", 0.0) or 0.0) * (float(self.filtered.x_dot) - 1.0)   # htoepk
        kp = float(cfg_get(self.cfg, "swing.heel_off_kp", 300.0))
        kd = float(cfg_get(self.cfg, "swing.heel_off_kd", 6.0))
        if heel_phase:                                         # φ_ref = −θ_hs·(1 − b(fr/s1))
            sv = min(max(fr / s1, 0.0), 1.0)
            T_ph = max(s1 * float(self.gait.stance), 1e-3)
            ph_ref = -th_hs * (1.0 - (10 * sv**3 - 15 * sv**4 + 6 * sv**5))
            ph_rate = th_hs * (30 * sv**2 - 60 * sv**3 + 30 * sv**4) / T_ph if sv < 1.0 else 0.0
        else:
            sv = min(max((fr - s2) / max(1.0 - s2, 1e-6), 0.0), 1.0)
            T_ph = max((1.0 - s2) * float(self.gait.stance), 1e-3)
            ph_ref = ph_end * (10 * sv**3 - 15 * sv**4 + 6 * sv**5)
            ph_rate = ph_end * (30 * sv**2 - 60 * sv**3 + 30 * sv**4) / T_ph if sv < 1.0 else 0.0
        if ph_ref - phi >= q_fl - q_a:                         # 궤적 쪽
            e_p, e_d, floor = ph_ref - phi, ph_rate - phid, 0.0
        else:                                                  # 발목 하한 쪽
            e_p, e_d, floor = q_fl - q_a, -qd_a, 1.0
        self.heel_off_dbg[leg] = (1.0, floor, phi + e_p, phi)
        return kp * e_p + kd * e_d

    def _toe_traj_on(self) -> bool:
        return bool(float(cfg_get(self.cfg, "swing.toe_height_trajectory", 0) or 0)) and \
            bool(cfg_get(self.cfg, "swing.pitch_trajectory_fraction", None))

    def _toe_drop(self, th) -> float:
        """발끝 내림 각 th (+ = 발끝 아래) 일 때 발끝이 발 site 보다 낮은 높이 l·sin th (발끝 위면 0)."""
        return float(cfg_get(self.cfg, "mpc.foot_half_length")) * math.sin(max(float(th), 0.0))

    def _toe_traj_offset(self, rt, t):
        """변형 toetraj (Q&A 10/6 Q61): 스윙 높이 궤적을 발 site 대신 발끝에 건다. 궤적은 이륙 때 발끝 높이에서 시작하고
        (reset 에서 p_init.z −= l·sin θ₀), site 목표 = 궤적 + l·sin θ_ref(t). θ_ref = −(pitchtraj 보정각 목표) 는 이륙 기울기에서
        수평으로 줄어드는 발끝 내림 각 → 이륙 순간 site 목표 = 실제 site (튐 없음), θ_ref = 0 이 된 뒤 (스윙 f 이후) 는 기존과 같다.
        반환 (Δp, Δv, Δa) z 성분 (속도 · 가속은 θ_ref 의 미분으로)."""
        if not self._toe_traj_on() or not np.isfinite(rt.pit_lo):
            return 0.0, 0.0, 0.0
        s = min(max((float(t) - rt.t_plo) / rt.T_pit, 0.0), 1.0)
        th0 = -rt.pit_lo
        if th0 <= 0.0:
            return 0.0, 0.0, 0.0
        l = float(cfg_get(self.cfg, "mpc.foot_half_length"))
        th = th0 * (1.0 - (10 * s**3 - 15 * s**4 + 6 * s**5))
        if s >= 1.0:
            return 0.0, 0.0, 0.0
        thd = -th0 * (30 * s**2 - 60 * s**3 + 30 * s**4) / rt.T_pit
        thdd = -th0 * (60 * s - 180 * s**2 + 120 * s**3) / rt.T_pit**2
        return l * math.sin(th), l * math.cos(th) * thd, l * (math.cos(th) * thdd - math.sin(th) * thd**2)

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
            if self.foot_model == "line" and int(float(cfg_get(cfg, "mpc.line_wrench_projection_stand", 0) or 0)):
                # 변형 lprojs (Q&A 10/1 Q22): 서기에서도 MPC 해를 선 접촉 집합으로 자른다 (lproj=3 은 걷기 디딤 발 경로에만 있었다)
                from leg_control import project_line_wrench
                u_st = np.asarray(u_st, dtype=float).copy()
                fy_ = np.asarray((self.last_info or {}).get("foot_yaw", [np.nan, np.nan]), dtype=float).reshape(-1, 2)[0]
                l_ = float(cfg_get(cfg, "mpc.foot_half_length"))
                for leg, fc, mc in ((LEFT, slice(0, 3), slice(6, 9)), (RIGHT, slice(3, 6), slice(9, 12))):
                    Fg, Mg, _ = project_line_wrench(u_st[fc], u_st[mc], resolved_foot_yaw(fy_[leg]), self.mpc.template, l_, 2)
                    u_st[fc], u_st[mc] = Fg, Mg
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
                gains = None
                w_fade = float(cfg_get(cfg, "swing.stance_yaw_fade_rate", 0.0) or 0.0)   # 변형 shw (Q&A 9/30 Q10):
                if w_fade > 0.0:                                   # 회전 명령 |ψ̇| 이 클수록 yaw hold 이득을 reference 값 (20 / 4) 쪽으로
                    f_ = max(0.0, 1.0 - abs(float(self.filtered.psi_dot)) / w_fade)
                    kp_s = float(cfg_get(cfg, "swing.stance_yaw_kp", 20.0))
                    kd_s = float(cfg_get(cfg, "swing.stance_yaw_kd", 4.0))
                    gains = (20.0 + f_ * (kp_s - 20.0), 4.0 + f_ * (kd_s - 4.0))
                lp = None
                lp_mode = int(float(cfg_get(cfg, "mpc.line_wrench_projection", 0) or 0))   # 1 투영 · 2 자르기
                if self.foot_model == "line" and lp_mode:
                    fy_ = np.asarray((self.last_info or {}).get("foot_yaw", [np.nan, np.nan]), dtype=float).reshape(-1, 2)[0]
                    tmpl = getattr(self.mpc, "template_walk", self.mpc.template)
                    if bool(getattr(self.mpc, "late0", np.zeros(2, dtype=bool))[leg]):   # 변형 toeoff: 첫 칸이 풀린 다리
                        tmpl = self.mpc.template_toe
                    elif bool(getattr(self.mpc, "early0", np.zeros(2, dtype=bool))[leg]):   # 변형 hstk: 첫 칸이 뒤꿈치 한 점
                        tmpl = self.mpc.template_heel
                    lp = (tmpl, float(cfg_get(cfg, "mpc.foot_half_length")), resolved_foot_yaw(fy_[leg]), lp_mode)
                u_use = self.u_hold
                if float(cfg_get(cfg, "mpc.input_interpolation", 0) or 0) and getattr(self, "_u_next", None) is not None:
                    # 변형 uint (Q&A 9/30 Q13): 풀이 사이 (10 ms) 에 u0 → u1 (다음 MPC 스텝) 로 선형 보간 — 계단 명령을 없앤다.
                    # 두 해의 볼록 조합이라 (같은 접촉 · 발 yaw 면) 선 접촉 제약을 그대로 만족한다.
                    a_ = min(max((t - self._t_solve) / float(self.mpc.dt), 0.0), 1.0) \
                        * float(cfg_get(cfg, "mpc.input_interpolation", 0) or 0)       # uint=s: 기울기 배율 (1 = u1 까지)
                    u_use = self.u_hold + a_ * (self._u_next - self.u_hold)
                tau[leg] = walking_stance_torque(dyn, state, leg, u_use, self._alpha(leg), rt.touchdown_yaw, cfg,
                                                 yaw_hold_gains=gains, line_proj=lp)   # 변형 lproj (Q&A 9/30 Q11)
                if float(cfg_get(cfg, "mpc.heel_off_fraction", 0.0) or 0.0) > 0.0 or self._heel_strike_angle() > 0.0:   # 변형 htoe · hstk
                    tau[leg][4] += self._alpha(leg) * self._heel_off_ankle_torque(state, leg, dyn, t)
                kd_s = float(cfg_get(cfg, "swing.stance_joint_damping", 0.0) or 0.0)   # 변형 sjd (Q&A 9/30 Q13)
                if kd_s > 0.0 and bool(np.asarray(state.foot_contact)[leg]):   # 실제로 닿아 있을 때만 (예정 착지 전 공중 발은 제외)
                    tau[leg] = tau[leg] + self._alpha(leg) * stance_foot_damping(
                        dyn, state, leg, kd_s, int(float(cfg_get(cfg, "swing.stance_joint_damping_mode", 1) or 1)))
            else:
                yaw_des, yaw_rate_des = self._swing_yaw_ref(rt, t)
                ff_mode = int(cfg_get(cfg, "swing.yaw_trajectory_ff", 1))
                torso_ref = float(self.filtered.psi_dot) if ff_mode == 2 else float(state.torso_angvel_W[2])
                p_bias, p_rate = self._swing_pitch_ref(rt, t)       # 변형 pitchtraj (Q&A 10/6 Q28)
                pz_, vz_, az_ = self._toe_traj_offset(rt, t)          # 변형 toetraj (Q&A 10/6 Q61), 꺼져 있으면 0
                ez_ = np.array([0.0, 0.0, 1.0])
                tau[leg] = swing_torque(dyn, state, leg, rt.traj.position() + pz_ * ez_, rt.traj.velocity() + vz_ * ez_,
                                        rt.traj.acceleration() + az_ * ez_, rt.touchdown_yaw, cfg,
                                        yaw_des=yaw_des, yaw_rate_des=yaw_rate_des,
                                        torso_wz=torso_ref,   # yawff=2: 몸통 회전은 명령 ψ̇ 로 (Q&A 9/29 Q1)
                                        pitch_bias=p_bias, pitch_rate_ref=p_rate)
                lim_mode = int(cfg_get(cfg, "swing.torque_limit_mode", 0) or 0)   # 변형 swlim (Q&A 9/30 Q4)
                if lim_mode:
                    ids = self.model.leg_act[leg]
                    tau[leg] = limit_swing_torque(tau[leg], dyn, self.model.ctrl_lo[ids], self.model.ctrl_hi[ids], lim_mode)
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
            if cfg_get(cfg, "swing.raibert_vel_wholebody", 0.0):  # 변형 rvwb (Q&A 9/29 Q12): 전신 CoM 속도
                import mujoco as _mj
                _mj.mj_subtreeVel(model.m, model.d)
                self.planner.v_wb = np.array(model.d.subtree_linvel[0], dtype=float, copy=True)
            self._k_schedule()                                     # 변형 ksx · ksy (Q&A 10/1 Q20)
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
                out[leg] = self.legs[leg].traj.position() + np.array([0.0, 0.0, self._toe_traj_offset(self.legs[leg], self.state.t)[0]])
        return out

    def heading_R(self) -> np.ndarray:
        return Rz(self.state.yaw_unwrapped)
