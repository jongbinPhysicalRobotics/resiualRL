"""LocomotionFSM — reference My_Controller/src/LocomotionFSM.cpp 1:1 이식 (spec 05 §4).

상태: StandingSettle, Standing, Walking, BrakingToStanding.  모드: walking / standing / interactive.
MIT yaml (requested_locomotion_mode: walking, settle 1.0 s) 에선 StandingSettle → (1.0 s 후) Walking 만 일어난다.
toggle 은 interactive 에서만 의미가 있고, braking 로직도 이식해 두었지만 walking 모드에선 도달 불가.

틱당 두 번 update (spec 05 §4.3, My_Controller.cpp:485 / :1386) 는 호출자 몫이다:
  컨트롤러 초기화 틱엔 1 번, 그 뒤엔 prepareController 에서 1 번 + runController 에서 1 번 (같은 time).
전이는 첫 호출에서 일어나고 두 번째는 no-op (braking 의 settle tick 카운터만 두 번 증가 — reference quirk).

사용:
    fsm = LocomotionFSM.from_cfg(cfg, start_time=t)
    out = fsm.update(t, vx_B, vy_B, wz_B)     # v_B = R_WTᵀ v_com_W, w_B = R_WTᵀ ω_W (My_Controller.cpp:593-601)
    out.mode, out.reset_gait_clock, out.reset_swing_state, out.zero_motion_command, ...
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np


class LocomotionMode(Enum):
    WALKING = "walking"
    STANDING = "standing"
    INTERACTIVE = "interactive"


class LocomotionState(Enum):
    STANDING_SETTLE = "standing_settle"
    STANDING = "standing"
    WALKING = "walking"
    BRAKING_TO_STANDING = "braking_to_standing"


def parse_mode(s) -> LocomotionMode:
    """ref: ControllerConfig.cpp:117-136 (walking|walk, standing|stand, interactive|general; 없으면 walking)."""
    if isinstance(s, LocomotionMode):
        return s
    if s is None:
        return LocomotionMode.WALKING
    k = str(s).strip().lower()
    if k in ("walking", "walk"):
        return LocomotionMode.WALKING
    if k in ("standing", "stand"):
        return LocomotionMode.STANDING
    if k in ("interactive", "general"):
        return LocomotionMode.INTERACTIVE
    raise ValueError(f"unknown locomotion mode {s!r}")


def _state_from_mode(mode: LocomotionMode) -> LocomotionState:
    """ref: LocomotionFSM.cpp:35-45."""
    if mode == LocomotionMode.WALKING:
        return LocomotionState.WALKING
    return LocomotionState.STANDING


def mode_for_state(state: LocomotionState) -> str:
    """ref: LocomotionFSM.cpp:90-101. 반환은 문자열 "walking" | "standing" (gait.set_mode 등과 같은 표기)."""
    if state in (LocomotionState.STANDING_SETTLE, LocomotionState.STANDING):
        return "standing"
    return "walking"


@dataclass
class FSMOutput:
    """ref: LocomotionFSM.h LocomotionFSMOutput, makeOutput LocomotionFSM.cpp:275-289."""
    state: LocomotionState
    mode: str                          # "walking" | "standing"
    just_transitioned: bool
    reset_gait_clock: bool             # → horizonClock.reset(time)
    reset_swing_state: bool            # → resetSwingState() (swing/planner/contact manager reset, last_control_time = t)
    accept_velocity_command: bool      # (MC 에서 미사용)
    zero_motion_command: bool          # → CommandFilter.update(..., zero_motion=True)
    swing_leg_dynamics: bool           # dynamicsRequest.swingLegDynamics   = mode == walking
    standing_foot_jacobians: bool      # dynamicsRequest.standingFootJacobians = mode == standing


class LocomotionFSM:
    """ref: LocomotionFSM.cpp:48-289."""

    def __init__(self, requested_mode, post_init_standing_settle_time: float,
                 braking_settle_speed_threshold: float, braking_settle_yaw_rate_threshold: float,
                 braking_settle_average_window: float, braking_settle_hold_ticks: int,
                 braking_timeout_seconds: float, braking_touchdown_count: int, start_time: float):
        self.requested_mode = parse_mode(requested_mode)
        self.target_mode = (LocomotionMode.WALKING if self.requested_mode == LocomotionMode.WALKING
                            else LocomotionMode.STANDING)
        self.settle_time = max(0.0, float(post_init_standing_settle_time))
        self.braking_speed_thr = max(0.0, float(braking_settle_speed_threshold))
        self.braking_yaw_rate_thr = max(0.0, float(braking_settle_yaw_rate_threshold))
        self.braking_window = max(0.0, float(braking_settle_average_window))
        self.braking_hold_ticks_thr = max(0, int(braking_settle_hold_ticks))
        self.braking_timeout = max(0.0, float(braking_timeout_seconds))
        self.braking_touchdown_thr = max(0, int(braking_touchdown_count))
        self.braking_settle_ticks = 0
        self.braking_ready = False
        self.braking_ready_start = 0.0
        self.braking_touchdowns = 0
        self.braking_settle_start = 0.0
        self._samples: deque = deque()
        self.state_start = float(start_time)
        # 초기 상태는 clamp 전 settle 값으로 판단 (LocomotionFSM.cpp:70 initialState(requestedMode, postInitStandingSettleTime))
        self.state = self._initial_state(self.requested_mode, float(post_init_standing_settle_time))

    @classmethod
    def from_cfg(cls, cfg, start_time: float) -> "LocomotionFSM":
        """ref: My_Controller.cpp:424-433 (생성자 인자)."""
        tr = cfg.locomotion_transition
        return cls(cfg.requested_locomotion_mode, cfg.startup.post_init_standing_settle_time,
                   tr.braking_settle_speed_threshold, tr.braking_settle_yaw_rate_threshold,
                   tr.braking_settle_average_window, tr.braking_settle_hold_ticks,
                   tr.braking_timeout_seconds, tr.braking_touchdown_count, start_time)

    @staticmethod
    def _initial_state(mode: LocomotionMode, settle: float) -> LocomotionState:
        """ref: LocomotionFSM.cpp:72-88."""
        if mode == LocomotionMode.WALKING and settle > 0.0:
            return LocomotionState.STANDING_SETTLE
        if mode == LocomotionMode.WALKING:
            return LocomotionState.WALKING
        if mode == LocomotionMode.INTERACTIVE and settle > 0.0:
            return LocomotionState.STANDING_SETTLE
        return _state_from_mode(mode)

    @property
    def mode(self) -> str:
        return mode_for_state(self.state)

    def _is_interactive(self) -> bool:
        return self.requested_mode == LocomotionMode.INTERACTIVE

    def _transition_to(self, nxt: LocomotionState, time: float):
        """ref: LocomotionFSM.cpp:107-127 (같은 상태면 no-op)."""
        if self.state == nxt:
            return
        self.state = nxt
        self.state_start = time
        if nxt == LocomotionState.BRAKING_TO_STANDING:
            self.braking_settle_ticks = 0
            self.braking_ready = False
            self.braking_ready_start = time
            self.braking_settle_start = time
            self.braking_touchdowns = 0
        else:
            self.braking_settle_ticks = 0
            self.braking_ready = False
            self.braking_touchdowns = 0
        self._samples.clear()

    def request_toggle(self) -> bool:
        """ref: LocomotionFSM.cpp:129-148 (interactive 에서만)."""
        if not self._is_interactive():
            return False
        current_mode = (LocomotionMode.WALKING if self.mode == "walking" else LocomotionMode.STANDING)
        if (self.state in (LocomotionState.STANDING_SETTLE, LocomotionState.BRAKING_TO_STANDING)
                or self.target_mode != current_mode):
            return False
        self.target_mode = (LocomotionMode.STANDING if self.target_mode == LocomotionMode.WALKING
                            else LocomotionMode.WALKING)
        return True

    def register_braking_touchdown(self):
        """ref: LocomotionFSM.cpp:150-154."""
        if self.state == LocomotionState.BRAKING_TO_STANDING and self.braking_ready:
            self.braking_touchdowns += 1

    def _braking_settle_averages_ready(self, time, vx, vy, wz) -> bool:
        """ref: LocomotionFSM.cpp:161-213."""
        self._samples.append((time, vx, vy, wz))
        window_start = time - self.braking_window
        while len(self._samples) > 1 and self._samples[0][0] < window_start:
            self._samples.popleft()
        elapsed = time - self.braking_settle_start
        if self.braking_window > 0.0 and elapsed + 1e-9 < self.braking_window:
            return False
        arr = np.asarray(self._samples)
        mvx, mvy, mwz = arr[:, 1].mean(), arr[:, 2].mean(), arr[:, 3].mean()
        return (abs(mvx) <= self.braking_speed_thr and abs(mvy) <= self.braking_speed_thr
                and abs(mwz) <= self.braking_yaw_rate_thr)

    def update(self, time: float, vx_B: float = 0.0, vy_B: float = 0.0, wz_B: float = 0.0) -> FSMOutput:
        """ref: LocomotionFSM.cpp:215-269."""
        if not np.isfinite(time):
            raise RuntimeError("LocomotionFSM::update received non-finite time")
        if not (np.isfinite(vx_B) and np.isfinite(vy_B)):
            raise RuntimeError("LocomotionFSM::update received invalid COM velocity")
        if not np.isfinite(wz_B):
            raise RuntimeError("LocomotionFSM::update received invalid yaw rate")

        jt = False
        if self.state == LocomotionState.STANDING_SETTLE and time - self.state_start >= self.settle_time:
            self._transition_to(_state_from_mode(self.target_mode), time)
            jt = True

        if self.target_mode == LocomotionMode.WALKING:
            if self.state in (LocomotionState.STANDING, LocomotionState.BRAKING_TO_STANDING):
                self._transition_to(LocomotionState.WALKING, time)
                jt = True
        else:
            if self.state == LocomotionState.WALKING:
                self._transition_to(LocomotionState.BRAKING_TO_STANDING, time)
                jt = True
            elif self.state == LocomotionState.BRAKING_TO_STANDING:
                if not self.braking_ready:
                    if self._braking_settle_averages_ready(time, vx_B, vy_B, wz_B):
                        self.braking_settle_ticks += 1
                        if self.braking_settle_ticks >= self.braking_hold_ticks_thr:
                            self.braking_ready = True
                            self.braking_ready_start = time
                            self.braking_touchdowns = 0
                    else:
                        self.braking_settle_ticks = 0
                elif (self.braking_touchdowns >= self.braking_touchdown_thr
                      or time - self.braking_ready_start >= self.braking_timeout):
                    self._transition_to(LocomotionState.STANDING, time)
                    jt = True
        return self._make_output(jt)

    def output(self) -> FSMOutput:
        """ref: LocomotionFSM.cpp:271-273 (just_transitioned = False)."""
        return self._make_output(False)

    def _make_output(self, jt: bool) -> FSMOutput:
        mode = mode_for_state(self.state)
        braking = self.state == LocomotionState.BRAKING_TO_STANDING
        return FSMOutput(state=self.state, mode=mode, just_transitioned=jt,
                         reset_gait_clock=jt and not braking, reset_swing_state=jt and not braking,
                         accept_velocity_command=(mode == "walking") and not braking,
                         zero_motion_command=braking,
                         swing_leg_dynamics=mode == "walking",
                         standing_foot_jacobians=mode == "standing")


def pre_init_dynamics_request(cfg) -> tuple[bool, bool]:
    """컨트롤러 초기화 전 요청 (My_Controller.cpp:488-504): walking 요청 + settle ≤ 0 → swing, 아니면 standing Jacobian.
    반환 (swing_leg_dynamics, standing_foot_jacobians)."""
    walking = parse_mode(cfg.requested_locomotion_mode) == LocomotionMode.WALKING
    if walking and float(cfg.startup.post_init_standing_settle_time) <= 0.0:
        return True, False
    return False, True
