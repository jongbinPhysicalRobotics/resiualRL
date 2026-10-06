"""몸 목표 (body target) + 스윙 발 착지점 계획 + 스윙 궤적 (참조 C++ 그대로 포팅, 버릇 포함).

ref: My_Controller/src/My_Controller.cpp:677-736 (updateBodyTarget), :338-361 (평균 발 xy, base pose 시드),
     :35-45 / :891-909 (half-stance offset, swingFootYawTargetWorld)
     common/include/Dynamics/SwingYawTarget.h:11-34 (psi offset, liftAngleNear)
     common/include/Utilities/AngleUtils.h:6-18
     My_Controller/src/BodyMotionReference.cpp:7-36 (shouldAdvanceYaw / advanceYaw)
     My_Controller/src/SwingFootPlanner.cpp:1-370 (착지점, stop/brake, latch, turn-stop)
     My_Controller/src/SwingFootTrajectory.cpp:6-139 (cubic smoothstep, z 두 반쪽 blend)
spec: 03_reference_planning_spec.md §3, §5, §7.1-7.4;  05_reference_orchestration_spec.md §3.3, §3.4, §7

버릇(그대로 둠, spec 03 §3/§5.6/§7.2):
  - bodyTarget yaw = yaml base_rpy_W.z(0) + Σ psi_dot_f*dt 의 개루프 적분, 재시드 없음 (planner 의 yaw0).
  - 착지점은 swing 첫 tick 에 한 번 계산되어 swing + 다음 stance 동안 고정 (stance 의 r 도 이 값).
  - 스윙 apex = pFinal.z + h (pInit.z 기준 아님), 목표 z = -0.005 (5 mm 파고듦).

# INTERFACE CHANGE: BodyTarget.update(..., cfg, *, com_offset_B=None, gait=None, t=None)
#   첫 호출 시드 nominal = base_position_W + Rz(base_rpy_W.z)·bodyComLocation (My_Controller.cpp:356-361, 685-689)
#   에 reduced-body com_offset_B (= RobotState.reduced.com_offset_B, 그 tick 값) 가 필요해서 keyword 로 추가.
#   gait/t 는 yaw_integration_mode 가 single/double_support 일 때만 쓰임 (MIT 'always' 에서는 무관).
# 참고: SwingFootPlanner.desired_foot_positions 는 reducedBodyComWorld / ComVelocityWorld 로 x0[3:6], x0[9:12] 를 쓴다
#   (C++ SwingFootPlanner.cpp:12-26 과 My_Controller.cpp:315-332 는 같은 식, 같은 tick 의 같은 state).
"""
from __future__ import annotations

import math

import numpy as np

from gait import LEFT, RIGHT, _normalize_mode, cfg_get, timing_params

SWING_FOOT_TARGET_Z = -0.005          # kSwingFootTargetZ, SwingFootPlanner.cpp:10
_DEG_TO_RAD = 3.141592653589793238462643383279502884 / 180.0   # SwingYawTarget.h:14


# ---------------------------------------------------------------------------
# 각도 / 회전 도우미 (AngleUtils.h, MatrixUtils.h:11-21)
# ---------------------------------------------------------------------------
def wrap_to_pi(a: float) -> float:
    """atan2(sin a, cos a).  ref: AngleUtils.h:6-8."""
    return math.atan2(math.sin(a), math.cos(a))


def lift_angle_near(a: float, ref: float) -> float:
    """ref + wrapToPi(a - ref).  ref: AngleUtils.h:16-18 (liftAngleNear(targetWrapped, currentUnwrapped))."""
    return ref + wrap_to_pi(a - ref)


def rz(psi: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rz_mul(psi: float, v) -> np.ndarray:
    """Rz(psi) @ v, Eigen 계수 순서 (r_i0 v0 + r_i1 v1 + r_i2 v2)."""
    c, s = math.cos(psi), math.sin(psi)
    v0, v1, v2 = float(v[0]), float(v[1]), float(v[2])
    return np.array([c * v0 + (-s) * v1 + 0.0 * v2,
                     s * v0 + c * v1 + 0.0 * v2,
                     0.0 * v0 + 0.0 * v1 + 1.0 * v2])


def _rzT_mul(psi: float, v) -> np.ndarray:
    """Rz(psi)^T @ v."""
    c, s = math.cos(psi), math.sin(psi)
    v0, v1, v2 = float(v[0]), float(v[1]), float(v[2])
    return np.array([c * v0 + s * v1 + 0.0 * v2,
                     (-s) * v0 + c * v1 + 0.0 * v2,
                     0.0 * v0 + 0.0 * v1 + 1.0 * v2])


def _norm2(x: float, y: float) -> float:
    """Eigen Vec2::norm() = sqrt(x*x + y*y)."""
    return math.sqrt(x * x + y * y)


def planar_speed(x_dot: float, y_dot: float) -> float:
    """|[x_dot, y_dot]| — half-stance offset 선택용 (Vec2::norm)."""
    return _norm2(float(x_dot), float(y_dot))


# ---------------------------------------------------------------------------
# half-stance offset / preview time / swing foot yaw
# ---------------------------------------------------------------------------
K_SCHED_X = None   # 변형 ksx (Q&A 10/1 Q20): 명령에 따라 바꾼 전진 k_half — 컨트롤러가 매 틱 설정. None = cfg 값
K_SCHED_Y = None   # 변형 ksy: 옆 성분 k_half (swing.lateral_half_stance_offset 대신)


def selected_half_stance_offset(speed: float, cfg) -> float:
    """strict '>' 비교: s > 0.65 → 0.26, s > 0.60 → 0.28, else 0.37 (s == 0.60 → 0.37).

    ref: SwingFootPlanner.cpp:158-169, My_Controller.cpp:35-45. 기본값 ControllerConfig.h:66-72 (0, inf).
    """
    if K_SCHED_X is not None:
        return float(K_SCHED_X)
    hi_sw = float(cfg_get(cfg, "swing.high_speed_body_velocity_half_stance_offset_switch_speed", math.inf))
    mid_sw = float(cfg_get(cfg, "swing.body_velocity_half_stance_offset_switch_speed", math.inf))
    if speed > hi_sw:
        return float(cfg_get(cfg, "swing.high_speed_body_velocity_half_stance_offset", 0.0))
    if speed > mid_sw:
        return float(cfg_get(cfg, "swing.mid_speed_body_velocity_half_stance_offset", 0.0))
    return float(cfg_get(cfg, "swing.body_velocity_half_stance_offset", 0.0))


def half_stance_preview_time(speed: float, cfg) -> float:
    """Tp = max(0, (0.5 + offset) * stance) → 0.2871 / 0.2574 / 0.2508 s.  ref: SwingFootPlanner.cpp:175-178."""
    stance = timing_params(cfg)[2]
    stance_fraction = 0.5 + selected_half_stance_offset(speed, cfg)
    return max(0.0, stance_fraction * stance)


def _yaw_lead_scale(cfg) -> float:
    """swing_foot_yaw_lead_scale (별칭 touchdown_yaw_lead_scale, ControllerConfig.cpp:285-292), 기본 1.0."""
    v = cfg_get(cfg, "swing.swing_foot_yaw_lead_scale", None)
    if v is None:
        v = cfg_get(cfg, "swing.touchdown_yaw_lead_scale", 1.0)
    return float(v)


def swing_foot_yaw_target_world(yaw_unwrapped: float, psi_dot: float, speed: float, cfg) -> float:
    """yaw_W_unwrapped + lead_scale * psi_dot * Tp  (측정 yaw 기준).  ref: My_Controller.cpp:895-904."""
    preview = max(0.0, (0.5 + selected_half_stance_offset(speed, cfg)) * timing_params(cfg)[2])
    return float(yaw_unwrapped) + _yaw_lead_scale(cfg) * float(psi_dot) * preview


PSI_OFFSET_SCALE = 1.0      # 변형 psioff (Q&A 9/29 Q6): 앞장서는 발 벌림 배율 (1 = reference, 0 = 끔). 컨트롤러가 cfg 로 설정


def swing_foot_yaw_psi_offset(leg: int, psi_dot: float) -> float:
    """clamp(100 deg/(rad/s)·|psi_dot|, 0, 20 deg) [rad]; +Left (psi_dot>0), −Right (psi_dot<0), else 0.

    ref: SwingYawTarget.h:11-27.
    """
    psi_dot = float(psi_dot)
    mag_deg = PSI_OFFSET_SCALE * min(max(100.0 * abs(psi_dot), 0.0), 20.0)
    if psi_dot > 0.0 and leg == LEFT:
        return mag_deg * _DEG_TO_RAD
    if psi_dot < 0.0 and leg == RIGHT:
        return -mag_deg * _DEG_TO_RAD
    return 0.0


def yaw_with_psi_offset(psi_dot: float, leg: int, base_yaw: float) -> float:
    """liftAngleNear(base + bias, base).  ref: SwingYawTarget.h:29-34."""
    bias = swing_foot_yaw_psi_offset(leg, psi_dot)
    return lift_angle_near(float(base_yaw) + bias, float(base_yaw))


# ---------------------------------------------------------------------------
# yaw 적분 모드 (BodyMotionReference.cpp:7-36, ControllerConfig.cpp:176-197)
# ---------------------------------------------------------------------------
def parse_yaw_integration_mode(s) -> str:
    """'single_support' | 'double_support' | 'always'. 없으면 기본 single_support (C++ 기본)."""
    if s is None:
        return "single_support"
    m = str(s)
    if m in ("single_support", "single", "stance_single"):
        return "single_support"
    if m in ("double_support", "double", "both_feet"):
        return "double_support"
    if m in ("always", "continuous"):
        return "always"
    raise ValueError(f"Invalid reference_trajectory.yaw_integration_mode: {s!r}")


def should_advance_yaw(gait, sample_time, mode: str) -> bool:
    """ref: BodyMotionReference.cpp:7-24 (gait None → isDoubleSupport false)."""
    if mode == "always":
        return True
    double = gait is not None and sample_time is not None and gait.both_stance(sample_time)
    if mode == "single_support":
        return not double
    if mode == "double_support":
        return double
    return False


def advance_yaw(gait, current_yaw: float, psi_dot: float, dt: float, sample_time, mode: str) -> float:
    """cur + psi_dot*dt if dt > 0 and shouldAdvanceYaw else cur (wrap 없음).  ref: BodyMotionReference.cpp:26-36."""
    if not (dt > 0.0) or not should_advance_yaw(gait, sample_time, mode):
        return current_yaw
    return current_yaw + psi_dot * dt


def _cmd(filtered_cmd, name: str) -> float:
    if filtered_cmd is None:
        return 0.0
    return float(getattr(filtered_cmd, name, 0.0))


# ---------------------------------------------------------------------------
# BodyTarget  (My_Controller.cpp:677-736, spec 05 §3.3, spec 03 §3)
# ---------------------------------------------------------------------------
class BodyTarget:
    """_bodyTarget = {nominalPosition_W, position_W, nominalHeight_W, euler_W, eulerSeed_W, initialized}.

    첫 호출: yaml initial_pose 가 있으면 nominal = base_position_W + Rz(base_rpy_W.z)·com_offset_B,
             euler = base_rpy_W (없으면 x0 로 시드). 이후 재시드 없음 (My_Controller.cpp:571 주석 처리).
    Standing: xy = 측정 발 xy 평균;  Walking: xy = x0[3:5];  둘 다 yaw += psi_dot*dt (mode 'always').
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """_bodyTarget = {} (My_Controller.cpp:471)."""
        self.nominal_position_W = np.zeros(3)
        self.position_W = np.zeros(3)
        self.nominal_height_W = 0.0
        self.euler_W = np.zeros(3)
        self.euler_seed_W = np.zeros(3)
        self.initialized = False

    def update(self, x0, dt, mode, foot_pos_W, filtered_cmd, cfg, *,
               com_offset_B=None, gait=None, t=None) -> None:
        x0 = np.asarray(x0, dtype=float)
        dt = float(dt)
        mode = _normalize_mode(mode)
        if not self.initialized:
            base_pos = cfg_get(cfg, "initial_pose.base_position_W", None)
            base_rpy = cfg_get(cfg, "initial_pose.base_rpy_W", None)
            if (base_pos is None) != (base_rpy is None):
                raise ValueError("initial_pose.base_position_W and base_rpy_W must be provided together")
            if base_pos is not None:
                if com_offset_B is None:
                    raise ValueError("BodyTarget first update needs com_offset_B (reduced.com_offset_B) "
                                     "to seed from initial_pose (My_Controller.cpp:356-361)")
                base_pos = np.asarray(base_pos, dtype=float).reshape(3)
                base_rpy = np.asarray(base_rpy, dtype=float).reshape(3)
                self.nominal_position_W = base_pos + _rz_mul(float(base_rpy[2]), com_offset_B)
                self.euler_W = base_rpy.copy()
            else:
                self.nominal_position_W = x0[3:6].copy()
                self.euler_W = np.array([0.0, 0.0, float(x0[2])])
            self.euler_seed_W = self.euler_W.copy()
            self.nominal_height_W = float(self.nominal_position_W[2])
            self.initialized = True

        h = _cmd(filtered_cmd, "body_height_offset_m")
        r_off = _cmd(filtered_cmd, "standing_roll_offset_rad")
        p_off = _cmd(filtered_cmd, "standing_pitch_offset_rad")
        psi_dot = _cmd(filtered_cmd, "psi_dot")
        yaw_mode = parse_yaw_integration_mode(cfg_get(cfg, "reference_trajectory.yaw_integration_mode", None))

        if mode == "standing":
            fp = np.asarray(foot_pos_W, dtype=float)
            # averageFootEndEffectorXY: 합을 다리 수로 나눔 (My_Controller.cpp:338-354)
            self.nominal_position_W[0] = (0.0 + fp[0, 0] + fp[1, 0]) / 2.0
            self.nominal_position_W[1] = (0.0 + fp[0, 1] + fp[1, 1]) / 2.0
        else:
            self.nominal_position_W[0] = float(x0[3])
            self.nominal_position_W[1] = float(x0[4])
        self.euler_W[2] = advance_yaw(gait, float(self.euler_W[2]), psi_dot, dt, t, yaw_mode)
        # applyPoseOffsets (My_Controller.cpp:704-709)
        self.euler_W[0] = self.euler_seed_W[0] + r_off
        self.euler_W[1] = self.euler_seed_W[1] + p_off
        self.nominal_position_W[2] = self.nominal_height_W + h
        self.position_W = self.nominal_position_W.copy()


# ---------------------------------------------------------------------------
# SwingFootPlanner  (SwingFootPlanner.cpp)
# ---------------------------------------------------------------------------
class SwingFootPlanner:
    """착지점 계획. 매 control tick (walking) 한 번 desired_foot_positions() 호출.

    진단용 속성 (C++ 에 없음, 읽기만): stop_active, stop_just_activated, last_updated (2,) bool.
    """

    def __init__(self, cfg, raibert_gain: float = 0.0, raibert_max: float = 0.10,
                 ypos_gain: float = 0.0, ypos_max: float = 0.05):
        self.cfg = cfg
        self.ypos_gain = float(ypos_gain)            # 착지점 옆 '위치' 되먹임 Δy = k·e_y (MATLAB convex_mpc.m 0.3·y_com, MPC_NOTES §12-7)
        self.ypos_max = float(ypos_max)
        self.lat_err = 0.0                           # 컨트롤러가 매 틱 넣어 줌: CoM 의 경로선 기준 옆 이탈 [m] (+ = 왼쪽)
        self.raibert_gain = float(raibert_gain)      # 측정 속도 되먹임 Δp = K(v_meas − v_cmd) (Q&A 9/26 Q8, 기본 0 = reference)
        self.raibert_max = float(raibert_max)
        self.raibert_int_gain = float(cfg_get(cfg, "swing.raibert_int_gain", 0.0) or 0.0)   # 변형 raii (Q&A 9/29 Q5)
        self.v_int = np.zeros(2)                     # ∫(v_meas − v_cmd) dt (월드 xy)
        self.v_int_t = None
        self.v_filt = None                           # 변형 vdir (Q&A 9/29 Q7): 실측 속도 1 차 저역통과 (τ 0.25 s)
        self.v_filt_t = None
        self.v_wb = None                             # 변형 rvwb: 컨트롤러가 매 틱 넣는 전신 CoM 속도 (월드)
        self.stride_gain = (float(cfg_get(cfg, "swing.raibert_stride_gain_x", 0.0) or 0.0),   # 변형 raisx · rais (Q&A 9/29 Q8)
                            float(cfg_get(cfg, "swing.raibert_stride_gain_y", 0.0) or 0.0))
        self.cycle, self.swing_time, self.stance_time, _, _, _ = timing_params(cfg)
        offs = cfg_get(cfg, "swing.nominal_foot_offsets_B", None)
        if offs is not None and len(offs) > 0:
            offs = np.asarray(offs, dtype=float)
            if offs.shape != (2, 3):
                raise ValueError("swing.nominal_foot_offsets_B must contain one 3-vector per leg")
            self._cfg_offsets = offs
        else:
            self._cfg_offsets = None
        sbo = cfg_get(cfg, "swing.stop_braking_offset_B", None)
        self.has_stop_braking_offset = sbo is not None
        self.stop_braking_offset_B = np.asarray(sbo, dtype=float).reshape(3) if sbo is not None else np.zeros(3)
        self.stop_cp_gain = float(cfg_get(cfg, "swing.stop_capture_point_gain", 1.0))
        self.stop_cp_max = float(cfg_get(cfg, "swing.stop_capture_point_max_offset", 0.20))
        self.stop_deadband = float(cfg_get(cfg, "swing.stop_velocity_deadband", 0.02))
        self.stop_latch_ticks = int(cfg_get(cfg, "swing.stop_braking_latch_clear_ticks", 5))
        self.body_yaw_target_W = 0.0
        self.reset()

    # --- state ---------------------------------------------------------------
    def reset(self) -> None:
        """ref: SwingFootPlanner.cpp:29-44 (+ ensureSwingTouchdownCache :88-107 가 다시 채우는 값)."""
        self.touchdown_targets = np.zeros((2, 3))
        self.v_int = np.zeros(2)                     # 변형 raii 적분 초기화
        self.v_int_t = None
        self.v_filt = None
        self.v_filt_t = None
        self.v_hist = []                             # 변형 rais: (t, v_앞, v_옆) 목표 yaw 기준, 한 주기
        self.v_stride = None                         # 한 주기 평균 (좌우 흔들림 제거) — 없으면 None
        self.touchdown_valid = [False, False]
        self.nominal_offsets_B = np.zeros((2, 3))
        self.nominal_offset_valid = [False, False]
        self.was_in_stance = [True, True]
        self.body_yaw_target_valid = False          # _bodyYawTarget_W 값은 유지 (C++ 동일)
        self.stop_clear_ticks = 0
        self.stop_was_active = False
        self.prev_planar_cmd = (0.0, 0.0)
        self.prev_yaw_rate_cmd = 0.0
        self.prev_cmd_valid = False
        self.turn_stop_center_W = np.zeros(3)
        self.turn_stop_yaw_W = 0.0
        self.turn_stop_frame_valid = False
        # 진단
        self.stop_active = False
        self.stop_just_activated = False
        self.last_updated = [False, False]

    def set_body_yaw_target(self, yaw: float) -> None:
        """ref: SwingFootPlanner.cpp:70-78."""
        yaw = float(yaw)
        if not math.isfinite(yaw):
            self.body_yaw_target_valid = False
            return
        self.body_yaw_target_W = yaw
        self.body_yaw_target_valid = True

    def body_yaw_target_world(self, state) -> float:
        """ref: :171-173."""
        return self.body_yaw_target_W if self.body_yaw_target_valid else float(state.yaw_unwrapped)

    def _ensure_nominal_offsets(self, state) -> None:
        """ref: :109-147. MIT 는 yaml 값 [[0,+0.076,0],[0,-0.076,0]] 을 한 번 복사."""
        if all(self.nominal_offset_valid):
            return
        if self._cfg_offsets is not None:
            self.nominal_offsets_B = self._cfg_offsets.copy()
            self.nominal_offset_valid = [True, True]
            return
        fp = np.asarray(state.foot_pos_W, dtype=float)
        center = (np.zeros(3) + fp[0] + fp[1]) / 2.0
        yaw = self.body_yaw_target_world(state)
        for leg in (LEFT, RIGHT):
            off = _rzT_mul(yaw, fp[leg] - center)
            off[0] = 0.0
            off[2] = 0.0
            self.nominal_offsets_B[leg] = off
            self.nominal_offset_valid[leg] = True

    # --- stop / brake --------------------------------------------------------
    def _stop_recenter_requested(self, px: float, py: float, psi_dot: float) -> bool:
        """ref: :180-185."""
        return _norm2(px, py) <= self.stop_deadband and abs(psi_dot) <= self.stop_deadband

    def _stop_recenter_active(self, px: float, py: float, psi_dot: float) -> bool:
        """상태 있는 latch: 명령이 deadband 를 벗어난 뒤에도 5 tick 더 active.  ref: :187-205."""
        if self._stop_recenter_requested(px, py, psi_dot):
            self.stop_clear_ticks = 0
            return True
        if not self.stop_was_active:
            self.stop_clear_ticks = 0
            return False
        if self.stop_clear_ticks < self.stop_latch_ticks:
            self.stop_clear_ticks += 1
            return True
        return False

    def _stop_stance_center_world(self, state, x0) -> np.ndarray:
        """capture-point 비슷한 제동 항 (gain 0.2, max 0.08).  ref: :207-233."""
        yaw = self.turn_stop_yaw_W if self.turn_stop_frame_valid else self.body_yaw_target_world(state)
        if self.has_stop_braking_offset:
            off_B = self.stop_braking_offset_B.copy()
        else:
            v_B = _rzT_mul(yaw, x0[9:12])
            off_B = np.array([self.stop_cp_gain * v_B[0], self.stop_cp_gain * v_B[1], 0.0])
            n = _norm2(off_B[0], off_B[1])
            if self.stop_cp_max > 0.0 and n > self.stop_cp_max:
                sc = self.stop_cp_max / n
                off_B[0] *= sc
                off_B[1] *= sc
        if self.turn_stop_frame_valid:
            center = self.turn_stop_center_W.copy()
        else:
            center = np.asarray(x0[3:6], dtype=float) + _rz_mul(yaw, off_B)
        center[2] = SWING_FOOT_TARGET_Z
        return center

    # --- 착지점 식 -----------------------------------------------------------
    def touchdown_target(self, leg, px, py, psi_dot, stop_recenter, state, x0, gait, clock, t) -> np.ndarray:
        """touchdownTargetWorldBodyVelocityHalfStance.  ref: SwingFootPlanner.cpp:235-275.

        planned_B = Rz(yaw0)^T·Rz(yaw0 + ½ψ̇Tp)·v_B·Tp + Rz(ψ̇·Trem)·off_B,  target = center + Rz(yaw0)·planned_B, z=-0.005.
        """
        v_B = (px, py, 0.0)
        Tp = half_stance_preview_time(_norm2(px, py), self.cfg)
        elapsed = 0.0                                  # 변형 replan (Q&A 9/29 Q12): 스윙 중 재계획이면 지난 스윙 시간만큼
        if cfg_get(self.cfg, "swing.replan", 0.0):     # 미리보기를 줄인다 (중심 = 지금 CoM 이라 이륙 때 계획과 같은 기하)
            elapsed = max(0.0, self.swing_time - gait.remaining_swing_time(leg, t, clock))
            Tp = max(0.0, Tp - elapsed)
        am = float(cfg_get(self.cfg, "swing.step_vmeas_blend", 0.0) or 0.0)   # 변형 vmeas (Q&A 9/29 Q7): 주항 속도에 실측 섞기
        if am > 0.0:                               # (MIT Cheetah: 실측 속도 × T_p 가 주항). 몸 기준 = 목표 yaw 기준
            y0_ = self.turn_stop_yaw_W if self.turn_stop_frame_valid else self.body_yaw_target_world(state)
            vm = _rzT_mul(y0_, np.array([x0[9], x0[10], 0.0]))
            v_B = ((1 - am) * px + am * vm[0], (1 - am) * py + am * vm[1], 0.0)
        ad = float(cfg_get(self.cfg, "swing.step_vdir_blend", 0.0) or 0.0)   # 변형 vdir (Q&A 9/29 Q7)
        if ad > 0.0 and self.v_filt is not None and _norm2(px, py) > 0.1 and float(np.hypot(*self.v_filt)) > 0.1:
            y0_ = self.turn_stop_yaw_W if self.turn_stop_frame_valid else self.body_yaw_target_world(state)
            vf_B = _rzT_mul(y0_, np.array([self.v_filt[0], self.v_filt[1], 0.0]))
            a_cmd = math.atan2(py, px)
            d_ang = (math.atan2(vf_B[1], vf_B[0]) - a_cmd + math.pi) % (2 * math.pi) - math.pi
            ang = a_cmd + ad * d_ang                  # 방향만 실측 진행 방향 쪽으로, 크기는 명령
            spd = _norm2(px, py)
            v_B = (spd * math.cos(ang), spd * math.sin(ang), 0.0)
        yaw0 = self.turn_stop_yaw_W if self.turn_stop_frame_valid else self.body_yaw_target_world(state)
        yaw_trans = yaw0 + 0.5 * psi_dot * Tp
        t_rem = gait.remaining_swing_time(leg, t, clock)
        yaw_td = yaw0 + psi_dot * t_rem
        ky = cfg_get(self.cfg, "swing.lateral_half_stance_offset", None)   # 변형 khalfy (Q&A 9/28 Q12)
        if K_SCHED_Y is not None:                  # 변형 ksy (Q&A 10/1 Q20): khalfy 와 달리 vdir · vmeas 로 섞은 v_B 를 쓴다
            Tp_y = max(0.0, (0.5 + float(K_SCHED_Y)) * timing_params(self.cfg)[2] - elapsed)
            step_W = _rz_mul(yaw_trans, (v_B[0] * Tp, v_B[1] * Tp_y, 0.0))
        elif ky is None:
            step_W = _rz_mul(yaw_trans, v_B) * Tp
        else:                                  # 옆 성분만 T_p,y = (0.5 + k_y)·T_st  (⚠ 명령 px · py 를 그대로 — vdir 섞기가 빠진다, Q&A 10/1 Q20)
            Tp_y = max(0.0, (0.5 + float(ky)) * timing_params(self.cfg)[2] - elapsed)
            step_W = _rz_mul(yaw_trans, (px * Tp, py * Tp_y, 0.0))
        center = self._stop_stance_center_world(state, x0) if stop_recenter \
            else np.asarray(x0[3:6], dtype=float).copy()
        off_B = self.nominal_offsets_B[leg]
        planned_B = _rzT_mul(yaw0, step_W) + _rz_mul(yaw_td - yaw0, off_B)
        if stop_recenter:
            planned_B = _rz_mul(yaw_td - yaw0, off_B)
        if self.raibert_gain > 0.0:                # 측정 속도 되먹임 (Q&A 9/26 Q8): reference 는 명령 속도만 써서
            v_cmd_W = _rz_mul(yaw_trans, (px, py, 0.0))   # 명령 속도 (vmeas 섞기와 무관). Δp = K(v_meas − v_cmd), 클램프
            verr = np.asarray(x0[9:12], dtype=float) - v_cmd_W
            ky = cfg_get(self.cfg, "swing.raibert_gain_y", None)       # 변형 raiy (Q&A 9/29 Q6): 옆 (진행 기준) 이득 따로
            if ky is None:
                corr = self.raibert_gain * verr
            else:
                vb = _rzT_mul(yaw0, np.array([verr[0], verr[1], 0.0]))
                corr = _rz_mul(yaw0, np.array([self.raibert_gain * vb[0], float(ky) * vb[1], 0.0]))
            n_c = _norm2(corr[0], corr[1])
            if n_c > self.raibert_max:
                corr = corr * (self.raibert_max / n_c)
            corr_B = _rzT_mul(yaw0, np.array([corr[0], corr[1], 0.0]))
            planned_B = planned_B + corr_B
        kc = float(cfg_get(self.cfg, "swing.centrifugal_gain", 0.0) or 0.0)   # 변형 cent (Q&A 9/29 Q6)
        if kc != 0.0:                              # MIT Cheetah 착지 원심 항 (음수 = 반대 부호 시험): Δp = k·½√(h/g)·[v_y ψ̇, −v_x ψ̇] (몸 기준 실측 v, 명령 ψ̇)
            vbm = _rzT_mul(yaw0, np.array([x0[9], x0[10], 0.0]))
            s_h = 0.5 * math.sqrt(max(float(x0[5]), 0.1) / 9.81)
            planned_B = planned_B + kc * s_h * float(psi_dot) * np.array([vbm[1], -vbm[0], 0.0])
        if self.raibert_int_gain > 0.0:           # 변형 raii (Q&A 9/29 Q5): Δp += K_i ∫(v_meas − v_cmd) dt
            ci = self.raibert_int_gain * self.v_int
            planned_B = planned_B + _rzT_mul(yaw0, np.array([ci[0], ci[1], 0.0]))
        if self.v_stride is not None and not stop_recenter:   # 변형 rais · raisx (Q&A 9/29 Q8): 한 주기 평균 속도 오차로 Raibert
            for i_, (k_, vc_) in enumerate(zip(self.stride_gain, (px, py))):   # — 순간 속도 (raiy) 는 걸음 흔들림에 과반응했다
                if k_:
                    planned_B[i_] += min(max(k_ * (self.v_stride[i_] - vc_), -0.08), 0.08)
        if self.ypos_gain > 0.0:                  # 옆 위치 되먹임 (Q&A 9/27 Q2): 밀린 쪽으로 더 내딛기 (+부호)
            planned_B[1] += min(max(self.ypos_gain * self.lat_err, -self.ypos_max), self.ypos_max)
        # 좌우 교차 방지 (lateral crossing guard)
        lat = float(off_B[1])
        if lat > 0.0:
            planned_B[1] = max(planned_B[1], 0.0)
        elif lat < 0.0:
            planned_B[1] = min(planned_B[1], 0.0)
        w_min = float(cfg_get(self.cfg, "swing.min_step_width", 0.0) or 0.0)   # 변형 minw (Q&A 9/29 Q12): 딛고 있는 발에서
        if w_min > 0.0 and lat != 0.0 and not stop_recenter:                   # 옆으로 최소 w_min (몸 중심 기준 교차 방지만으론
            other = 1 - leg                                                     # 밀린 뒤 두 발이 겹쳤다 — 발 꼬임)
            o_B = _rzT_mul(yaw0, np.asarray(state.foot_pos_W[other], dtype=float) - center)
            if lat > 0.0:
                planned_B[1] = max(planned_B[1], float(o_B[1]) + w_min)
            else:
                planned_B[1] = min(planned_B[1], float(o_B[1]) - w_min)
        target = center + _rz_mul(yaw0, planned_B)
        target[2] = float(cfg_get(self.cfg, "swing.touchdown_z", SWING_FOOT_TARGET_Z))   # 변형 tdz (Q&A 9/29 Q5)
        return target

    # --- 매 tick -------------------------------------------------------------
    def desired_foot_positions(self, state, x0, clock, gait, filtered_cmd, t) -> np.ndarray:
        """(2,3) [Left, Right] 목표.  ref: SwingFootPlanner.cpp:277-370.

        stance(명목 스케줄 c) 인 발: 캐시된 착지점 (reset 직후 첫 stance 만 측정 발 위치로 시드).
        swing 인 발: swing 첫 tick / 캐시 무효 / stop 막 켜짐 / turn-stop 유효 일 때만 재계산, 아니면 고정.
        """
        t = float(t)
        clock.sync(t)                                    # syncHorizonClock (:283)
        if cfg_get(self.cfg, "swing.raibert_vel_wholebody", 0.0) and self.v_wb is not None:
            x0 = np.array(x0, dtype=float, copy=True)    # 변형 rvwb (Q&A 9/29 Q12): 착지점 되먹임 속도 = 전신 CoM 속도
            x0[9:12] = self.v_wb                         # (몸통 속도엔 스윙 다리 반동이 섞인다). 착지 기준점 (CoM 위치) 은 그대로
        if cfg_get(self.cfg, "swing.step_vdir_blend", None):   # 변형 vdir: 실측 속도 저역통과
            vm = np.asarray(x0[9:11], dtype=float)
            if self.v_filt is None:
                self.v_filt = vm.copy()
            else:
                a_ = min(1.0, max(0.0, t - self.v_filt_t) / 0.25)
                self.v_filt = self.v_filt + a_ * (vm - self.v_filt)
            self.v_filt_t = t
        if self.stride_gain[0] or self.stride_gain[1]:   # 변형 rais: 한 주기 이동평균 (걸음마다 좌우 흔들림은 주기 성분이라 0)
            vh = _rzT_mul(self.body_yaw_target_world(state), np.array([x0[9], x0[10], 0.0]))
            self.v_hist.append((t, float(vh[0]), float(vh[1])))
            while self.v_hist and t - self.v_hist[0][0] > self.cycle:
                self.v_hist.pop(0)
            if t - self.v_hist[0][0] >= 0.95 * self.cycle:
                a_ = np.asarray(self.v_hist)
                self.v_stride = (float(a_[:, 1].mean()), float(a_[:, 2].mean()))
        if self.raibert_int_gain > 0.0:                  # 변형 raii: 속도 오차 적분 (클램프 → K_i·∫ ≤ 5 cm)
            if self.v_int_t is not None:
                yaw_c = self.body_yaw_target_world(state)
                vc = _rz_mul(yaw_c, (_cmd(filtered_cmd, "x_dot"), _cmd(filtered_cmd, "y_dot"), 0.0))
                self.v_int += (np.asarray(x0[9:11], dtype=float) - vc[:2]) * max(0.0, t - self.v_int_t)
                lim = 0.05 / self.raibert_int_gain
                nrm = float(np.hypot(*self.v_int))
                if nrm > lim:
                    self.v_int *= lim / nrm
            self.v_int_t = t
        self._ensure_nominal_offsets(state)
        x0 = np.asarray(x0, dtype=float)
        px, py = _cmd(filtered_cmd, "x_dot"), _cmd(filtered_cmd, "y_dot")
        psi_cmd = _cmd(filtered_cmd, "psi_dot")
        stop = self._stop_recenter_active(px, py, psi_cmd)
        just = stop and not self.stop_was_active
        if just and self.prev_cmd_valid:
            radius = 0.0
            for off in self.nominal_offsets_B:
                radius = max(radius, _norm2(float(off[0]), float(off[1])))
            prev_tan = abs(self.prev_yaw_rate_cmd) * radius
            was_turn_dominant = (abs(self.prev_yaw_rate_cmd) > self.stop_deadband and
                                 _norm2(*self.prev_planar_cmd) <= prev_tan + self.stop_deadband)
            if was_turn_dominant:
                self.turn_stop_center_W = x0[3:6].copy()
                self.turn_stop_center_W[2] = SWING_FOOT_TARGET_Z
                self.turn_stop_yaw_W = float(state.yaw_unwrapped)
                self.turn_stop_frame_valid = True
        elif not stop:
            self.turn_stop_center_W = np.zeros(3)
            self.turn_stop_yaw_W = 0.0
            self.turn_stop_frame_valid = False
        if stop and self.turn_stop_frame_valid:
            self.turn_stop_center_W = x0[3:6].copy()
            self.turn_stop_center_W[2] = SWING_FOOT_TARGET_Z
            self.turn_stop_yaw_W = float(state.yaw_unwrapped)

        foot_pos = np.asarray(state.foot_pos_W, dtype=float)
        out = np.zeros((2, 3))
        for leg in (LEFT, RIGHT):
            self.last_updated[leg] = False
            if gait.contact(leg, t, clock):
                if not self.touchdown_valid[leg]:
                    self.touchdown_targets[leg] = foot_pos[leg].copy()   # currentFootTouchdownTarget (:149-151)
                    self.touchdown_valid[leg] = True
                self.was_in_stance[leg] = True
                out[leg] = self.touchdown_targets[leg]
                continue
            was = self.was_in_stance[leg]
            self.was_in_stance[leg] = False
            rp = float(cfg_get(self.cfg, "swing.replan", 0.0) or 0.0)            # 변형 replan: 스윙 중 매 틱 재계산 (끝 50 ms 고정)
            late = gait.remaining_swing_time(leg, t, clock) <= float(cfg_get(self.cfg, "swing.replan_freeze", 0.05))
            if was or not self.touchdown_valid[leg] or just or self.turn_stop_frame_valid or (rp > 0.0 and not late):
                self.touchdown_targets[leg] = self.touchdown_target(
                    leg, px, py, psi_cmd, stop, state, x0, gait, clock, t)
                self.touchdown_valid[leg] = True
                self.last_updated[leg] = True
            out[leg] = self.touchdown_targets[leg]

        self.stop_active = stop
        self.stop_just_activated = just
        self.stop_was_active = stop
        self.prev_planar_cmd = (px, py)
        self.prev_yaw_rate_cmd = psi_cmd
        self.prev_cmd_valid = True
        return out


def standing_foot_targets(foot_pos_W) -> np.ndarray:
    """Standing 모드의 명목 발 목표 = 측정 발 위치, z = -0.005.  ref: My_Controller.cpp:1396-1405."""
    out = np.array(foot_pos_W, dtype=float).reshape(2, 3).copy()
    out[:, 2] = -0.005
    return out


# ---------------------------------------------------------------------------
# SwingFootTrajectory  (SwingFootTrajectory.cpp:6-139)
# ---------------------------------------------------------------------------
def _blend(s: float) -> float:
    return 3.0 * s * s - 2.0 * s * s * s


def _blend_dot(s: float) -> float:
    return 6.0 * s - 6.0 * s * s


def _blend_ddot(s: float) -> float:
    return 6.0 - 12.0 * s


def _blend3(s: float):
    """3 차 smoothstep (b, b', b'') — 양 끝 속도 0, 가속 ±6 (기본)."""
    return _blend(s), _blend_dot(s), _blend_ddot(s)


def _blend5(s: float):
    """변형 swq=5 (Q&A 10/6 Q3): 5 차 (최소 저크) b = 10s³ − 15s⁴ + 6s⁵ — 양 끝 속도 · 가속 0, 최대 가속 5.77 (s = 0.21)."""
    s2 = s * s
    return (s2 * s * (10.0 - 15.0 * s + 6.0 * s2), 30.0 * s2 * (1.0 - 2.0 * s + s2), 60.0 * s * (1.0 - 3.0 * s + 2.0 * s2))


_PROFILE_CACHE = {}


def _jerk_limited_profile(r: float):
    """변형 swprof (Q&A 9/30 Q2): 최대 가속을 줄인 수평 스윙 blend. 가속이 0 → A (구간 r) → A 유지 → −A (가운데 2r 동안 선형 전환)
    → −A 유지 → 0 (끝 r). 시작 · 끝의 속도 · 가속 0, b(1) = 1 이 되게 A 를 수치로 맞춘다. r → 0 이면 뱅뱅 (최대 가속 4),
    cubic smoothstep 은 6. 반환: (s 격자, b, b', b'') 표."""
    key = round(float(r), 4)
    if key in _PROFILE_CACHE:
        return _PROFILE_CACHE[key]
    n = 2001
    s = np.linspace(0.0, 1.0, n)
    a = np.empty(n)
    for i, x in enumerate(s):
        if x < r:
            a[i] = x / r
        elif x < 0.5 - r:
            a[i] = 1.0
        elif x < 0.5 + r:
            a[i] = 1.0 - (x - (0.5 - r)) / r
        elif x < 1.0 - r:
            a[i] = -1.0
        else:
            a[i] = -1.0 + (x - (1.0 - r)) / r
    ds = s[1] - s[0]
    v = np.concatenate(([0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * ds)))
    b = np.concatenate(([0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * ds)))
    A = 1.0 / b[-1]
    out = (s, b * A, v * A, a * A)
    _PROFILE_CACHE[key] = out
    return out


class SwingFootTrajectory:
    """cubic smoothstep (Bézier 아님). XY 는 전체 swing 에 한 번, Z 는 pInit.z→pFinal.z+h→pFinal.z 두 반쪽.

    reset tick 출력: p = pInit, v = 0, a = +6·d/T² (XY) / 6·dz·(2/T)² (Z) — a 는 0 이 아님 (spec 03 §7.3).
    position()/velocity()/acceleration() 는 매 갱신마다 새 배열 — 받은 쪽에서 in-place 수정하지 말 것.
    """

    def __init__(self):
        self.p_init = np.zeros(3)
        self.p_final = np.zeros(3)
        self._p = np.zeros(3)
        self._v = np.zeros(3)
        self._a = np.zeros(3)
        self.height = 0.0
        self.T = 0.0
        self.remaining = 0.0
        self.active = False
        self.xy_profile = 0.0                      # 변형 swprof: 0 = cubic smoothstep (reference), r > 0 = 저가속 blend
        self.blend_order = 3                       # 변형 swq (Q&A 10/6 Q3): 3 = cubic smoothstep (기본), 5 = 5 차 최소 저크

    def reset(self, p_init, p_final, height: float, T: float) -> None:
        """ref: :6-21."""
        T = float(T)
        if T <= 0.0:
            raise ValueError("SwingFootTrajectory requires positive swingTime")
        self.p_init = np.array(p_init, dtype=float).reshape(3)
        self.p_final = np.array(p_final, dtype=float).reshape(3)
        self.height = float(height)
        self.T = T
        self.remaining = T
        self.active = True
        self._update_outputs()

    def set_final_position(self, p) -> None:
        """ref: :23-28 — 목표 이동, blend 는 pInit 부터 다시 (점프 필터 없음), zMid 도 따라 움직임."""
        self.p_final = np.array(p, dtype=float).reshape(3)
        if self.active:
            self._update_outputs()

    def advance(self, dt: float) -> None:
        """ref: :30-40."""
        if not self.active:
            return
        self.remaining = max(0.0, self.remaining - max(0.0, float(dt)))
        self._update_outputs()
        if self.remaining <= 0.0:
            self.active = False

    def deactivate(self) -> None:
        """ref: :73-77."""
        self.remaining = 0.0
        self.active = False
        self._update_outputs()

    def remaining_time(self) -> float:
        return self.remaining

    def phase(self) -> float:
        """1 - remaining/T (T<=0 이면 1).  ref: :58-63."""
        if self.T <= 0.0:
            return 1.0
        return 1.0 - (self.remaining / self.T)

    def finished(self) -> bool:
        return self.remaining <= 0.0

    def position(self) -> np.ndarray:
        return self._p

    def velocity(self) -> np.ndarray:
        return self._v

    def acceleration(self) -> np.ndarray:
        return self._a

    def _update_outputs(self) -> None:
        """ref: :79-127."""
        s = min(max(self.phase(), 0.0), 1.0)
        ds = (1.0 / self.T) if self.T > 0.0 else 0.0
        d2s = 0.0
        if self.xy_profile > 0.0:                  # 변형 swprof (Q&A 9/30 Q2): 최대 가속 낮은 수평 blend
            sg, B, Bd, Bdd = _jerk_limited_profile(self.xy_profile)
            b, bd, bdd = float(np.interp(s, sg, B)), float(np.interp(s, sg, Bd)), float(np.interp(s, sg, Bdd))
        else:
            b, bd, bdd = _blend5(s) if self.blend_order == 5 else (_blend(s), _blend_dot(s), _blend_ddot(s))
        pi0, pi1, pi2 = float(self.p_init[0]), float(self.p_init[1]), float(self.p_init[2])
        pf0, pf1, pf2 = float(self.p_final[0]), float(self.p_final[1]), float(self.p_final[2])
        d0, d1 = pf0 - pi0, pf1 - pi1
        px = (1.0 - b) * pi0 + b * pf0
        py = (1.0 - b) * pi1 + b * pf1
        vx = bd * d0 * ds
        vy = bd * d1 * ds
        ax = d0 * (bdd * ds * ds + bd * d2s)
        ay = d1 * (bdd * ds * ds + bd * d2s)
        z_mid = pf2 + self.height
        du = 2.0 * ds
        if s <= 0.5:
            u = 2.0 * s
            bz, bzd, bzdd = _blend5(u) if self.blend_order == 5 else (_blend(u), _blend_dot(u), _blend_ddot(u))
            dz = z_mid - pi2
            pz = (1.0 - bz) * pi2 + bz * z_mid
        else:
            u = 2.0 * s - 1.0
            bz, bzd, bzdd = _blend5(u) if self.blend_order == 5 else (_blend(u), _blend_dot(u), _blend_ddot(u))
            dz = pf2 - z_mid
            pz = (1.0 - bz) * z_mid + bz * pf2
        vz = bzd * dz * du
        az = dz * bzdd * du * du
        self._p = np.array([px, py, pz])
        self._v = np.array([vx, vy, vz])
        self._a = np.array([ax, ay, az])
