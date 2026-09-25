"""사용자 명령: UserCommand, CommandFilter (reference 저역통과), CommandSchedule (headless 주입기, D3).

CommandFilter = My_Controller.cpp:610-654 updateFilteredUserCommand + ControllerConfig.cpp:649-656 clampUserCommand
               + My_Controller.cpp:28-33 lowPassBlendAlpha (spec 05 §3.2):
    raw = clamp(raw)                                   # x,y,psi 만 ±max (max 가 무한이면 그대로)
    zero_motion → raw.x,y,psi = 0                      # FSM BrakingToStanding
    첫 호출: filtered = raw (그대로 반환)
    prev = filtered; filtered = raw                    # 카운터 등 나머지 필드는 통과
    raw.x == raw.y == raw.psi == 0 → filtered.x,y,psi = 0 (즉시 정지, 필터 없음)
    아니면 a(tau) = clamp(-expm1(-dt/tau), 0, 1), tau<=0 또는 dt<=0 이면 1 → prev + a (raw − prev)
    body_height_offset_m = raw (필터 없음)
    standing_roll/pitch_offset = prev + a(0.70) (raw − prev)
    filtered = clamp(filtered)
dt = 0 인 틱 (컨트롤러 첫 틱, resetSwingState 가 있는 모든 FSM 전이 틱 = Standing→Walking 포함) 에선 a = 1 이라
filtered = raw (snap). dt 는 호출자가 max(0, t − last_control_time) 로 준다 (My_Controller.cpp:1390).

CommandSchedule (D3): reference headless 는 x_dot 만 스케줄할 수 있어서 y_dot / psi_dot 까지 넣는 주입기를 둔다.
기본 (07 D3) = 첫 컨트롤러 틱부터 raw 명령이 있는 "constant" (reference 기본: raw x_dot 가 t = 1.0 s 부터, 컨트롤러는 1.998 s 시작).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, fields

import numpy as np

MOTION_FIELDS = ("x_dot", "y_dot", "psi_dot")


@dataclass
class UserCommand:
    """ref: common/include/Utilities/UserCommand.h:4-13.
    x_dot [m/s, yaw 프레임 +x 앞], y_dot [m/s, +y 왼쪽], psi_dot [rad/s, +z 반시계]."""
    x_dot: float = 0.0
    y_dot: float = 0.0
    psi_dot: float = 0.0
    body_height_offset_m: float = 0.0
    standing_roll_offset_rad: float = 0.0
    standing_pitch_offset_rad: float = 0.0
    standing_mpc_debug_log_request: int = 0          # 카운터 (디버그 전용)
    locomotion_mode_toggle_request: int = 0          # 카운터 (FSM toggle, interactive 에서만 의미)

    def copy(self) -> "UserCommand":
        return copy.copy(self)

    def planar_speed(self) -> float:
        return float(np.hypot(self.x_dot, self.y_dot))


def low_pass_blend_alpha(tau: float, dt: float) -> float:
    """ref: My_Controller.cpp:28-33.  !(tau>0) || !(dt>0) → 1; else clamp(-expm1(-dt/tau), 0, 1)."""
    if not (tau > 0.0) or not (dt > 0.0):
        return 1.0
    return float(np.clip(-np.expm1(-dt / tau), 0.0, 1.0))


def _clamp_opt(v: float, max_abs: float) -> float:
    """ref: ControllerConfig.cpp:51-56 clampWithOptionalLimit."""
    if not np.isfinite(max_abs):
        return v
    return float(min(max(v, -max_abs), max_abs))


class CommandFilter:
    """ref: My_Controller.cpp:610-654."""

    def __init__(self, cfg):
        f = cfg.user_command_filter
        self.x_tau = float(f.x_dot_tau)
        self.y_tau = float(f.y_dot_tau)
        self.psi_tau = float(f.psi_dot_tau)
        self.roll_tau = float(f.standing_roll_offset_tau)
        self.pitch_tau = float(f.standing_pitch_offset_tau)
        self.x_max = float(f.x_dot_max)
        self.y_max = float(f.y_dot_max)
        self.psi_max = float(f.psi_dot_max)
        self.reset()

    def reset(self):
        self.filtered = UserCommand()
        self.initialized = False

    def clamp(self, c: UserCommand) -> UserCommand:
        """ref: ControllerConfig.cpp:649-656 clampUserCommand (x, y, psi 만)."""
        out = c.copy()
        out.x_dot = _clamp_opt(out.x_dot, self.x_max)
        out.y_dot = _clamp_opt(out.y_dot, self.y_max)
        out.psi_dot = _clamp_opt(out.psi_dot, self.psi_max)
        return out

    def update(self, raw: UserCommand | None, dt: float, zero_motion: bool = False) -> UserCommand:
        """한 틱 갱신. 반환 = filtered 의 복사본. raw=None 이면 UserCommand{} (reference 의 nullptr 처리)."""
        r = self.clamp(raw if raw is not None else UserCommand())
        if zero_motion:
            r.x_dot = r.y_dot = r.psi_dot = 0.0
        if not self.initialized:
            self.filtered = r
            self.initialized = True
            return self.filtered.copy()

        prev = self.filtered
        new = r.copy()
        if r.x_dot == 0.0 and r.y_dot == 0.0 and r.psi_dot == 0.0:
            new.x_dot = new.y_dot = new.psi_dot = 0.0
        else:
            new.x_dot = prev.x_dot + low_pass_blend_alpha(self.x_tau, dt) * (r.x_dot - prev.x_dot)
            new.y_dot = prev.y_dot + low_pass_blend_alpha(self.y_tau, dt) * (r.y_dot - prev.y_dot)
            new.psi_dot = prev.psi_dot + low_pass_blend_alpha(self.psi_tau, dt) * (r.psi_dot - prev.psi_dot)
        new.body_height_offset_m = r.body_height_offset_m
        new.standing_roll_offset_rad = (prev.standing_roll_offset_rad + low_pass_blend_alpha(self.roll_tau, dt)
                                        * (r.standing_roll_offset_rad - prev.standing_roll_offset_rad))
        new.standing_pitch_offset_rad = (prev.standing_pitch_offset_rad + low_pass_blend_alpha(self.pitch_tau, dt)
                                         * (r.standing_pitch_offset_rad - prev.standing_pitch_offset_rad))
        self.filtered = self.clamp(new)
        return self.filtered.copy()


# ----------------------------------------------------------------------------------------------------------------
# 스케줄 (D3)
# ----------------------------------------------------------------------------------------------------------------
_FIELD_NAMES = tuple(f.name for f in fields(UserCommand))


class CommandSchedule:
    """raw 명령 스케줄. profile = {필드: 명세}. 명세 종류:

      숫자 v                                → constant: t = 0 부터 v (07 D3 기본, reference headless 기본 의미)
      [(t0, v0), (t1, v1), ...]             → step (piecewise constant): 시각 t_i 부터 v_i, **첫 점의 값은 그 시각 전에도**
                                              (reference CONVEXMPC_HEADLESS_X_DOT_PROFILE, SimulationRunner.cpp:503-513)
      {"points": [...], "interp": "linear"} → 점 사이 선형 보간, 양 끝 밖은 끝값 유지
      {"final": v, "rate": r, "start": t0}  → reference 램프 (SimulationRunner.cpp:514-522):
                                              t < t0 → 0 (reference 는 이전 값 유지 = 0), t ≥ t0 →
                                              copysign(min(|v|, r (t − t0)), v); r <= 0 이면 바로 v
    명시 안 한 필드는 0. 예: CommandSchedule({"x_dot": 0.6}); CommandSchedule({"psi_dot": [(0, 0), (1.0, 1.3)]}).
    """

    def __init__(self, profile: dict | None = None):
        profile = dict(profile or {})
        for k in profile:
            if k not in _FIELD_NAMES:
                raise KeyError(f"unknown UserCommand field {k!r}")
        self.profile = {k: self._normalize(v) for k, v in profile.items()}

    @staticmethod
    def _normalize(spec):
        if isinstance(spec, (int, float, np.floating, np.integer)):
            return ("constant", float(spec))
        if isinstance(spec, dict):
            if "final" in spec:
                return ("ramp", float(spec["final"]), float(spec.get("rate", 0.0)), float(spec.get("start", 0.0)))
            pts = [(float(t), float(v)) for t, v in spec["points"]]
            interp = spec.get("interp", "step")
            if interp not in ("step", "linear"):
                raise ValueError(f"interp must be step|linear, got {interp!r}")
            return (interp, sorted(pts))
        pts = [(float(t), float(v)) for t, v in spec]
        if not pts:
            raise ValueError("empty profile")
        return ("step", sorted(pts))

    @staticmethod
    def _eval(spec, t: float) -> float:
        kind = spec[0]
        if kind == "constant":
            return spec[1]
        if kind == "ramp":
            final, rate, start = spec[1], spec[2], spec[3]
            if t < start:
                return 0.0
            if rate > 0.0:
                return float(np.copysign(min(abs(final), rate * (t - start)), final))
            return final
        pts = spec[1]
        if kind == "step":
            v = pts[0][1]
            for tp, vp in pts:
                if t + 1e-12 < tp:
                    break
                v = vp
            return v
        ts = np.array([p[0] for p in pts])
        vs = np.array([p[1] for p in pts])
        return float(np.interp(t, ts, vs))

    def __call__(self, t: float) -> UserCommand:
        c = UserCommand()
        for k, spec in self.profile.items():
            val = self._eval(spec, t)
            if k in ("standing_mpc_debug_log_request", "locomotion_mode_toggle_request"):
                val = int(val)
            setattr(c, k, val)
        return c

    # 편의 생성자 ------------------------------------------------------------------------------------------------
    @classmethod
    def constant(cls, x_dot=0.0, y_dot=0.0, psi_dot=0.0) -> "CommandSchedule":
        """t = 0 부터 일정 (07 D3 기본)."""
        return cls({"x_dot": x_dot, "y_dot": y_dot, "psi_dot": psi_dot})

    @classmethod
    def step_at(cls, T: float, x_dot=0.0, y_dot=0.0, psi_dot=0.0) -> "CommandSchedule":
        """t < T 에선 0, t ≥ T 부터 값 (예: T = walking 시작)."""
        return cls({k: [(0.0, 0.0), (T, v)] for k, v in (("x_dot", x_dot), ("y_dot", y_dot), ("psi_dot", psi_dot))})

    @classmethod
    def ramp(cls, start: float, rate: float, x_dot=0.0, y_dot=0.0, psi_dot=0.0) -> "CommandSchedule":
        """reference 램프 공식 (rate: 필드 단위/s) 을 세 필드에 각각."""
        return cls({k: {"final": v, "rate": rate, "start": start}
                    for k, v in (("x_dot", x_dot), ("y_dot", y_dot), ("psi_dot", psi_dot))})
