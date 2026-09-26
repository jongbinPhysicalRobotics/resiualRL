"""ContactManager — 명목 스케줄 + 힘 히스테리시스로 "실제로 디딘 발" (active contact) 을 정한다.

ref: My_Controller/src/ContactManager.cpp:1-446, include/MyController/ContactManager.h:12-91,
     include/MyController/ContactSchedule.h, ControllerConfig.h:109-123 (기본값)
spec: 05_reference_orchestration_spec.md §6 (전체), 03 §6, 02 §5.4

MIT 값: on 36 N ×1 tick, off 0.5 N ×4 tick, ramp 0.01 s, lock_steps 1, early 켬, late 끔.
  - activeContact = earlyContact ? True : scheduled   (late 는 cfg 플래그로 꺼져 있음; 코드 경로는 포팅)
  - 첫 active tick 은 rampTime = 0 → alpha = 0 정확히. 그 뒤 += dt (자체 dt = t - last update).
  - managed_foot_positions: search / early / liftoffHold 일 때만 commandedFootTarget 으로 교체.
  - build_horizon_override: lock_steps 개, stance = active, min_scale = active ? alpha : 0.

state 는 duck typing: state.foot_pos_W (2,3), state.foot_normal_force (2,), state.foot_contact (2,),
  선택 state.has_contact_force (기본 True — sim 은 항상 힘 신호가 있음, MujocoCheaterStateReader.cpp:363).
leg 순서 0 = Left, 1 = Right.
"""
from __future__ import annotations

import math

import numpy as np

from gait import LEFT, RIGHT, OverrideStep, _normalize_mode, cfg_get


class ContactManager:
    def __init__(self, cfg):
        g = lambda k, d: cfg_get(cfg, "contact_manager." + k, d)   # noqa: E731
        # 기본값: ControllerConfig.h:109-123
        self.on_thr = float(g("contact_force_on_threshold", 20.0))
        self.off_thr = float(g("contact_force_off_threshold", 5.0))
        self.on_confirm = int(g("contact_on_confirm_ticks", 2))
        self.off_confirm = int(g("contact_off_confirm_ticks", 2))
        self.ramp_duration = float(g("contact_ramp_duration", 0.08))
        self.lock_steps = int(g("contact_lock_steps", 3))
        self.late_timeout = float(g("late_contact_timeout", 0.20))
        self.search_velocity = float(g("ground_search_velocity", 0.08))
        self.search_max_depth = float(g("ground_search_max_depth", 0.04))
        self.search_tracking_time = float(g("ground_search_tracking_time", 0.08))
        self.stance_loss_height = float(g("stance_contact_loss_foot_height", 0.025))
        self.enable_early = bool(g("enable_early_contact_handling", True))
        self.enable_late = bool(g("enable_late_contact_handling", True))

        # public per-leg state (ContactManagerLegState 기본값, ContactManager.h:12-30)
        self.scheduled = np.ones(2, dtype=bool)
        self.estimated = np.ones(2, dtype=bool)
        self.active_contact = np.ones(2, dtype=bool)
        self.early = np.zeros(2, dtype=bool)
        self.late = np.zeros(2, dtype=bool)
        self.liftoff_hold = np.zeros(2, dtype=bool)       # 코드상 항상 False (CM:289)
        self.search_mode = np.zeros(2, dtype=bool)
        self.recovery_failure = np.zeros(2, dtype=bool)
        self.ramp_alpha = np.ones(2)
        self.late_contact_time = np.zeros(2)
        self.liftoff_hold_time = np.zeros(2)
        self.normal_force = np.zeros(2)
        self.frozen_touchdown_W = np.zeros((2, 3))
        self.commanded_target_W = np.zeros((2, 3))
        # private (RuntimeLegState, ContactManager.h:62-76)
        self.initialized = np.zeros(2, dtype=bool)
        self.raw_estimate = np.ones(2, dtype=bool)
        self.prev_active = np.ones(2, dtype=bool)
        self.prev_scheduled = np.ones(2, dtype=bool)
        self.prev_late = np.zeros(2, dtype=bool)
        self.released_during_swing = np.zeros(2, dtype=bool)
        self.on_ticks = [0, 0]
        self.off_ticks = [0, 0]
        self.ramp_time = np.zeros(2)
        self.search_depth = np.zeros(2)
        # _legs 가 아직 없음 (ensureRuntimeLayout 전) → buildHorizonOverride 기본 True/1.0
        self._layout_ready = False
        self.last_update_time = 0.0
        self.has_last_update_time = False
        self.last_dt = 0.0                                 # 진단용

    # ------------------------------------------------------------------
    def _ensure_layout(self) -> None:
        """ensureRuntimeLayout (CM:36-56): 처음 만들 때만 _hasLastUpdateTime = false."""
        if not self._layout_ready:
            self._layout_ready = True
            self.has_last_update_time = False

    @staticmethod
    def _force_signal(state, leg):
        has_force = bool(getattr(state, "has_contact_force", True))
        fn = float(np.asarray(state.foot_normal_force)[leg])
        bit = bool(np.asarray(getattr(state, "foot_contact", np.zeros(2, dtype=bool)))[leg])
        return has_force, fn, bit

    def _update_estimated(self, leg: int, has_force: bool, fn_raw: float, bit: bool) -> bool:
        """히스테리시스.  ref: CM:89-148."""
        has_signal = has_force and math.isfinite(fn_raw)
        fn = fn_raw if has_signal else 0.0
        if not self.initialized[leg]:
            self.estimated[leg] = (fn >= self.on_thr) if has_signal else bit
            self.raw_estimate[leg] = self.estimated[leg]
            self.on_ticks[leg] = 0
            self.off_ticks[leg] = 0
            return bool(self.estimated[leg])

        if not has_signal:
            if bit:
                self.on_ticks[leg] += 1
                self.off_ticks[leg] = 0
            else:
                self.off_ticks[leg] += 1
                self.on_ticks[leg] = 0
        elif self.estimated[leg]:
            if fn <= self.off_thr:
                self.off_ticks[leg] += 1
                self.on_ticks[leg] = 0
            else:
                self.off_ticks[leg] = 0
        else:
            if fn >= self.on_thr:
                self.on_ticks[leg] += 1
                self.off_ticks[leg] = 0
            else:
                self.on_ticks[leg] = 0

        if (not self.estimated[leg]) and self.on_ticks[leg] >= max(self.on_confirm, 1):
            self.estimated[leg] = True
            self.on_ticks[leg] = 0
            self.off_ticks[leg] = 0
        elif self.estimated[leg] and self.off_ticks[leg] >= max(self.off_confirm, 1):
            self.estimated[leg] = False
            self.on_ticks[leg] = 0
            self.off_ticks[leg] = 0

        self.raw_estimate[leg] = (fn >= self.on_thr) if has_signal else bit
        return bool(self.estimated[leg])

    # ------------------------------------------------------------------
    def reset(self, state, gait, mode, t) -> None:
        """초기화 / FSM 전이 (Braking 제외) 때.  ref: CM:150-198."""
        self._ensure_layout()
        mode = _normalize_mode(mode)
        t = float(t)
        self.last_update_time = t
        self.has_last_update_time = True
        fpos = np.asarray(state.foot_pos_W, dtype=float)
        for leg in (LEFT, RIGHT):
            has_force, fn, bit = self._force_signal(state, leg)
            sched = bool(gait.contact(leg, t))
            self.scheduled[leg] = sched
            est = (fn >= self.on_thr) if has_force else bit
            self.estimated[leg] = est
            act = True if mode == "standing" else sched
            self.active_contact[leg] = act
            self.early[leg] = False
            self.late[leg] = False
            self.liftoff_hold[leg] = False
            self.search_mode[leg] = False
            self.recovery_failure[leg] = False
            self.ramp_alpha[leg] = 1.0 if act else 0.0
            self.late_contact_time[leg] = 0.0
            self.liftoff_hold_time[leg] = 0.0
            self.normal_force[leg] = fn if has_force else 0.0
            self.frozen_touchdown_W[leg] = fpos[leg]
            self.commanded_target_W[leg] = fpos[leg]

            self.initialized[leg] = True
            self.raw_estimate[leg] = est
            self.prev_active[leg] = act
            self.prev_scheduled[leg] = sched
            self.prev_late[leg] = False
            self.released_during_swing[leg] = (not sched) and (not est)
            self.on_ticks[leg] = 0
            self.off_ticks[leg] = 0
            self.ramp_time[leg] = self.ramp_duration if act else 0.0
            self.search_depth[leg] = 0.0

    def update(self, state, gait, mode, nominal_desired, t) -> None:
        """매 tick (spec 05 §2 step 9).  ref: CM:200-359."""
        self._ensure_layout()
        mode = _normalize_mode(mode)
        t = float(t)
        if self.has_last_update_time and t >= self.last_update_time:
            dt = t - self.last_update_time
        else:
            dt = 0.0
        self.last_update_time = t
        self.has_last_update_time = True
        self.last_dt = dt

        fpos = np.asarray(state.foot_pos_W, dtype=float)
        nominal = np.asarray(nominal_desired, dtype=float)
        for leg in (LEFT, RIGHT):
            has_force, fn, bit = self._force_signal(state, leg)
            sched = bool(gait.contact(leg, t))
            self.scheduled[leg] = sched
            self.normal_force[leg] = fn if has_force else 0.0
            est = self._update_estimated(leg, has_force, fn, bit)
            self.estimated[leg] = est
            nominal_target = nominal[leg].copy()
            foot = fpos[leg]

            if not self.initialized[leg]:
                # lazy init (CM:231-243) — reset() 이 먼저 불리므로 실제로는 도달 안 함
                self.initialized[leg] = True
                self.prev_active[leg] = sched
                self.prev_scheduled[leg] = sched
                self.prev_late[leg] = False
                self.released_during_swing[leg] = (not sched) and (not est)
                self.ramp_time[leg] = self.ramp_duration if sched else 0.0
                self.search_depth[leg] = 0.0
                self.frozen_touchdown_W[leg] = foot if sched else nominal_target

            if mode == "standing":
                # standing short-circuit (CM:245-264)
                self.active_contact[leg] = True
                self.early[leg] = False
                self.late[leg] = False
                self.liftoff_hold[leg] = False
                self.search_mode[leg] = False
                self.recovery_failure[leg] = False
                self.ramp_alpha[leg] = 1.0
                self.late_contact_time[leg] = 0.0
                self.liftoff_hold_time[leg] = 0.0
                self.frozen_touchdown_W[leg] = foot
                self.commanded_target_W[leg] = foot
                self.prev_active[leg] = True
                self.prev_scheduled[leg] = True
                self.prev_late[leg] = False
                self.released_during_swing[leg] = False
                self.ramp_time[leg] = self.ramp_duration
                self.search_depth[leg] = 0.0
                continue

            # ---- walking ----
            if sched:
                self.released_during_swing[leg] = False
            elif not est:
                self.released_during_swing[leg] = True

            early = self.enable_early and (not sched) and est and bool(self.released_during_swing[leg])
            self.early[leg] = early

            scheduled_touchdown = sched and not self.prev_scheduled[leg]
            continuing_late = sched and bool(self.prev_late[leg]) and not est
            lost_stance = (sched and not est and not scheduled_touchdown and not self.prev_late[leg]
                           and float(foot[2]) > self.stance_loss_height)
            late = self.enable_late and (not est) and (scheduled_touchdown or continuing_late or lost_stance)
            self.late[leg] = late
            self.liftoff_hold[leg] = False
            self.liftoff_hold_time[leg] = 0.0

            if early and not self.prev_active[leg]:
                self.frozen_touchdown_W[leg] = foot           # 실제 착지점 한 번 캡처

            if late:
                if not self.prev_late[leg]:
                    self.frozen_touchdown_W[leg] = nominal_target
                    self.late_contact_time[leg] = 0.0
                    self.search_depth[leg] = 0.0
                else:
                    self.late_contact_time[leg] += dt
                    self.search_depth[leg] += self.search_velocity * dt
                self.search_depth[leg] = min(max(float(self.search_depth[leg]), 0.0), self.search_max_depth)
                self.search_mode[leg] = True
                self.recovery_failure[leg] = (self.late_timeout > 0.0 and
                                              self.late_contact_time[leg] > self.late_timeout)
                target = self.frozen_touchdown_W[leg].copy()
                target[2] -= self.search_depth[leg]
                self.commanded_target_W[leg] = target
            else:
                self.late_contact_time[leg] = 0.0
                self.search_mode[leg] = False
                self.recovery_failure[leg] = False
                self.search_depth[leg] = 0.0
                if self.liftoff_hold[leg]:
                    self.commanded_target_W[leg] = foot
                elif early:
                    self.commanded_target_W[leg] = self.frozen_touchdown_W[leg]
                else:
                    self.commanded_target_W[leg] = nominal_target

            if late:
                act = False
            elif self.liftoff_hold[leg]:
                act = True
            elif early:
                act = True
            else:
                act = sched
                if act:
                    self.frozen_touchdown_W[leg] = foot       # stance: 실제 발 추적 (bookkeeping)
            self.active_contact[leg] = act

            if act:
                if not self.prev_active[leg]:
                    self.ramp_time[leg] = 0.0
                else:
                    self.ramp_time[leg] += dt
                if self.ramp_duration <= 0.0:
                    self.ramp_alpha[leg] = 1.0
                else:
                    self.ramp_alpha[leg] = min(max(self.ramp_time[leg] / self.ramp_duration, 0.0), 1.0)
            else:
                self.ramp_time[leg] = 0.0
                self.ramp_alpha[leg] = 0.0

            self.prev_active[leg] = act
            self.prev_scheduled[leg] = sched
            self.prev_late[leg] = late

    # ------------------------------------------------------------------
    def managed_foot_positions(self, nominal) -> np.ndarray:
        """search / early / liftoffHold 인 발만 commandedFootTarget 으로 교체.  ref: CM:394-414."""
        out = np.array(nominal, dtype=float).reshape(2, 3).copy()
        if not self._layout_ready:
            return out
        for leg in (LEFT, RIGHT):
            if not self.search_mode[leg] and not self.early[leg] and not self.liftoff_hold[leg]:
                continue
            out[leg] = self.commanded_target_W[leg]
        return out

    def build_horizon_override(self) -> list[OverrideStep]:
        """step 0..lock_steps-1: stance = active, min_scale = active ? alpha : 0.  ref: CM:416-446."""
        n = max(self.lock_steps, 0)
        if self._layout_ready:
            stance = np.array([bool(self.active_contact[LEFT]), bool(self.active_contact[RIGHT])])
            scale = np.array([float(self.ramp_alpha[l]) if self.active_contact[l] else 0.0 for l in (LEFT, RIGHT)])
        else:
            stance = np.ones(2, dtype=bool)
            scale = np.ones(2)
        return [OverrideStep(enabled=True, stance=stance.copy(), min_scale=scale.copy()) for _ in range(n)]

    # --- 편의 accessor (C++ activeContact(side) 등) ------------------------
    def active(self, leg: int) -> bool:
        return bool(self.active_contact[leg])

    def alpha(self, leg: int) -> float:
        return float(self.ramp_alpha[leg])

    def search_mode_active(self, leg: int) -> bool:
        return bool(self.search_mode[leg])
