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
def selected_half_stance_offset(speed: float, cfg) -> float:
    """strict '>' 비교: s > 0.65 → 0.26, s > 0.60 → 0.28, else 0.37 (s == 0.60 → 0.37).

    ref: SwingFootPlanner.cpp:158-169, My_Controller.cpp:35-45. 기본값 ControllerConfig.h:66-72 (0, inf).
    """
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


def swing_foot_yaw_psi_offset(leg: int, psi_dot: float) -> float:
    """clamp(100 deg/(rad/s)·|psi_dot|, 0, 20 deg) [rad]; +Left (psi_dot>0), −Right (psi_dot<0), else 0.

    ref: SwingYawTarget.h:11-27.
    """
    psi_dot = float(psi_dot)
    mag_deg = min(max(100.0 * abs(psi_dot), 0.0), 20.0)
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

    def __init__(self, cfg):
        self.cfg = cfg
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
        yaw0 = self.turn_stop_yaw_W if self.turn_stop_frame_valid else self.body_yaw_target_world(state)
        yaw_trans = yaw0 + 0.5 * psi_dot * Tp
        t_rem = gait.remaining_swing_time(leg, t, clock)
        yaw_td = yaw0 + psi_dot * t_rem
        step_W = _rz_mul(yaw_trans, v_B) * Tp
        center = self._stop_stance_center_world(state, x0) if stop_recenter \
            else np.asarray(x0[3:6], dtype=float).copy()
        off_B = self.nominal_offsets_B[leg]
        planned_B = _rzT_mul(yaw0, step_W) + _rz_mul(yaw_td - yaw0, off_B)
        if stop_recenter:
            planned_B = _rz_mul(yaw_td - yaw0, off_B)
        # 좌우 교차 방지 (lateral crossing guard)
        lat = float(off_B[1])
        if lat > 0.0:
            planned_B[1] = max(planned_B[1], 0.0)
        elif lat < 0.0:
            planned_B[1] = min(planned_B[1], 0.0)
        target = center + _rz_mul(yaw0, planned_B)
        target[2] = SWING_FOOT_TARGET_Z
        return target

    # --- 매 tick -------------------------------------------------------------
    def desired_foot_positions(self, state, x0, clock, gait, filtered_cmd, t) -> np.ndarray:
        """(2,3) [Left, Right] 목표.  ref: SwingFootPlanner.cpp:277-370.

        stance(명목 스케줄 c) 인 발: 캐시된 착지점 (reset 직후 첫 stance 만 측정 발 위치로 시드).
        swing 인 발: swing 첫 tick / 캐시 무효 / stop 막 켜짐 / turn-stop 유효 일 때만 재계산, 아니면 고정.
        """
        t = float(t)
        clock.sync(t)                                    # syncHorizonClock (:283)
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
            if was or not self.touchdown_valid[leg] or just or self.turn_stop_frame_valid:
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
        b, bd, bdd = _blend(s), _blend_dot(s), _blend_ddot(s)
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
            bz, bzd, bzdd = _blend(u), _blend_dot(u), _blend_ddot(u)
            dz = z_mid - pi2
            pz = (1.0 - bz) * pi2 + bz * z_mid
        else:
            u = 2.0 * s - 1.0
            bz, bzd, bzdd = _blend(u), _blend_dot(u), _blend_ddot(u)
            dz = pf2 - z_mid
            pz = (1.0 - bz) * z_mid + bz * pf2
        vz = bzd * dz * du
        az = dz * bzdd * du * du
        self._p = np.array([px, py, pz])
        self._v = np.array([vx, vy, vz])
        self._a = np.array([ax, ay, az])
