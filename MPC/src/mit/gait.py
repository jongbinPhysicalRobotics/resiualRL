"""Gait 위상 스케줄 + horizon clock (참조 C++ 그대로 포팅).

ref: My_Controller/include/MyController/HorizonClock.h:9-38
     My_Controller/src/GaitScheduler.cpp:98-124 (p, c, bothFeetStance), :134-193 (horizon 접촉표)
     My_Controller/src/My_Controller.cpp:59-61 (remainingSwingTime)
     My_Controller/include/MyController/ContactSchedule.h (override 구조)
spec: 03_reference_planning_spec.md §1.1-1.3, 05_reference_orchestration_spec.md §5, 02 §5.2-5.4

주의 (spec 03 §1.2, 02 §5.2):
  - horizon step k 의 시각은 tk = t0 + k*dt_mpc, t0 = "현재 cycle 시작" (지금 시각이 아님).
  - 위상/접촉은 double 식을 C++ 과 같은 순서로 그대로 계산한다. k=4 에서 p_L 이 정확히
    stance/cycle(=0.66) 근처에 걸리는 knife-edge 가 있어서 t0 의 비트에 따라 stance/swing 이
    갈린다 → 표를 하드코딩하지 말고 매 solve 마다 살아있는 식으로 평가.
  - 모든 스칼라 연산은 Python float (IEEE double) + math.fmod (= C fmod).

# INTERFACE CHANGE: GaitScheduler.__init__(cfg, clock=None) + attribute `clock` / set_clock(clock).
#   C++ GaitScheduler 는 HorizonClock 포인터를 들고 p() 안에서 t0 를 읽는다 (GaitScheduler.cpp:103-109).
#   design 3.4 의 phase(leg, t) 에는 clock 인자가 없으므로, 걷기 모드에서 phase/contact 를 부르기 전에
#   clock 을 붙여야 한다 (없으면 C++ 처럼 RuntimeError). horizon_steps(clock, ...) 는 넘겨받은 clock 을 쓴다.
#   phase/contact/both_stance/remaining_swing_time 은 선택 인자 clock=None 도 받는다 (추가 인자, 기존 호출 불변).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

LEFT, RIGHT = 0, 1

_MISSING = object()


# ---------------------------------------------------------------------------
# config 접근 도우미 (owner A 의 config 객체 / dict / SimpleNamespace 모두 허용)
# ---------------------------------------------------------------------------
def cfg_get(cfg, path: str, default=_MISSING):
    """점 경로로 cfg 값을 읽는다. 없으면 default (C++ ControllerConfig.h 기본값을 호출부가 넘김).

    yaml 키가 없을 때 C++ 은 readScalarIfPresent 로 기본값을 유지한다 (spec 03 §0.5).
    """
    node = cfg
    for key in path.split("."):
        try:
            if isinstance(node, dict):
                node = node[key]
            else:
                try:
                    node = getattr(node, key)
                except AttributeError:
                    node = node[key]
        except (AttributeError, KeyError, TypeError, IndexError):
            if default is _MISSING:
                raise KeyError(f"config key missing: {path}") from None
            return default
    if node is None and default is not _MISSING:
        return default
    return node


def timing_params(cfg):
    """(cycle, swing, stance, horizon, horizon_steps, dt_mpc).

    ref: ControllerConfig.cpp:625-647 — dtMpc = horizon / double(horizonSteps) (리터럴 0.02 아님).
    기본값: ControllerConfig.h:31-37 (cycle 1.0, swing 0.4, stance 0.6, horizon 0.5, steps 15).
    """
    cycle = float(cfg_get(cfg, "timing.cycle", 1.0))
    swing = float(cfg_get(cfg, "timing.swing", 0.4))
    stance = float(cfg_get(cfg, "timing.stance", 0.6))
    horizon = float(cfg_get(cfg, "timing.horizon", 0.5))
    steps = int(cfg_get(cfg, "timing.horizon_steps", 15))
    dt_mpc = horizon / float(steps)
    return cycle, swing, stance, horizon, steps, dt_mpc


# ---------------------------------------------------------------------------
# HorizonClock  (HorizonClock.h:9-38)
# ---------------------------------------------------------------------------
class HorizonClock:
    """t0 = 현재 gait cycle 의 원점.  sync 는 cycle 단위로만 앞으로 (뒤로 안 감).

    ref: HorizonClock.h:14-16 reset, :18-26 sync (while t - t0 >= cycle: t0 += cycle), :32-34 tk.
    """

    def __init__(self, t: float, cfg):
        self.cycle, _, _, _, self.N, self.dt_mpc = timing_params(cfg)
        self.t0 = float(t)

    def reset(self, t: float) -> None:
        self.t0 = float(t)

    def sync(self, t: float) -> None:
        t = float(t)
        if not math.isfinite(t):
            raise RuntimeError("HorizonClock::sync received non-finite time")
        while t - self.t0 >= self.cycle:
            self.t0 += self.cycle

    def tk(self, k: int) -> float:
        # _t0 + static_cast<double>(k) * dtMpc()
        return self.t0 + float(k) * self.dt_mpc


# ---------------------------------------------------------------------------
# Horizon 접촉표 항목 / override  (GaitScheduler.h GaitConstraintStep, ContactSchedule.h)
# ---------------------------------------------------------------------------
@dataclass
class StepContact:
    """horizon step 하나의 접촉 (MPC 가 보는 것). stance[leg], min_scale[leg] (Fz ≥ min_scale*normal_force_min)."""
    stance: np.ndarray = field(default_factory=lambda: np.ones(2, dtype=bool))
    min_scale: np.ndarray = field(default_factory=lambda: np.ones(2))


@dataclass
class OverrideStep:
    """ContactScheduleOverrideStep (ContactSchedule.h:6-12). 기본값도 C++ 과 같게: enabled False, 접촉 True, scale 1."""
    enabled: bool = False
    stance: np.ndarray = field(default_factory=lambda: np.ones(2, dtype=bool))
    min_scale: np.ndarray = field(default_factory=lambda: np.ones(2))


def _normalize_mode(mode) -> str:
    """'walking' | 'standing' (대소문자 무관, Enum 이면 .name/.value)."""
    if not isinstance(mode, str):
        val = getattr(mode, "value", None)
        mode = val if isinstance(val, str) else getattr(mode, "name", str(mode))
    m = str(mode).strip().lower()
    if m not in ("walking", "standing"):
        raise ValueError(f"unknown locomotion mode: {mode!r} (expected 'walking' or 'standing')")
    return m


# ---------------------------------------------------------------------------
# GaitScheduler  (GaitScheduler.cpp:98-193)
# ---------------------------------------------------------------------------
class GaitScheduler:
    """시계 기반 고정 위상 스케줄. 접촉 이벤트로 적응하지 않는다.

    p(side,t) = fmod((t - t0)/cycle + phi, 1),  phi_L = 0.5, phi_R = 0     (GaitScheduler.cpp:98-110)
    c(side,t) = 0 <= p < stance/cycle                                       (:112-120)
    Standing 모드: p ≡ 0, c ≡ True.
    MIT (cycle 0.5): Left swing [0.08,0.25), Right swing [0.33,0.5) (cycle 원점 기준).
    """

    def __init__(self, cfg, clock: HorizonClock | None = None):
        self.cycle, self.swing, self.stance, _, self.N, self.dt_mpc = timing_params(cfg)
        self.clock = clock
        # C++ 은 LocomotionMode 기본값이 Walking 이 아니라 생성 후 setLocomotionMode 로 정해진다.
        # 초기화 직후 FSM 출력이 적용되므로 기본값은 'walking' 으로 두되 controller 가 set_mode 해야 한다.
        self.mode = "walking"

    def set_clock(self, clock: HorizonClock) -> None:
        self.clock = clock

    def set_mode(self, mode) -> None:
        self.mode = _normalize_mode(mode)

    @property
    def standing(self) -> bool:
        return self.mode == "standing"

    # --- 위상 / 접촉 -------------------------------------------------------
    def phase(self, leg: int, t: float, clock: HorizonClock | None = None) -> float:
        """ref: GaitScheduler.cpp:98-110."""
        if self.mode == "standing":
            return 0.0
        clk = clock if clock is not None else self.clock
        if clk is None:
            raise RuntimeError("GaitScheduler::p requires initialized HorizonClock")
        phi = 0.0
        if leg == LEFT:
            phi = 0.5
        return math.fmod((float(t) - clk.t0) / self.cycle + phi, 1.0)

    def contact(self, leg: int, t: float, clock: HorizonClock | None = None) -> bool:
        """ref: GaitScheduler.cpp:112-120. stanceFraction = stance/cycle 를 매번 double 로 계산."""
        if self.mode == "standing":
            return True
        p = self.phase(leg, t, clock)
        stance_fraction = self.stance / self.cycle
        return 0.0 <= p < stance_fraction

    def both_stance(self, t: float, clock: HorizonClock | None = None) -> bool:
        """ref: GaitScheduler.cpp:122-124."""
        return self.contact(LEFT, t, clock) and self.contact(RIGHT, t, clock)

    def remaining_swing_time(self, leg: int, t: float, clock: HorizonClock | None = None) -> float:
        """clamp(cycle*(1-p), 0, swing).  ref: My_Controller.cpp:59-61, SwingFootPlanner.cpp:247-251."""
        v = self.cycle * (1.0 - self.phase(leg, t, clock))
        # std::clamp(v, 0, swing)
        if v < 0.0:
            return 0.0
        if self.swing < v:
            return self.swing
        return v

    # --- horizon 접촉표 ----------------------------------------------------
    def horizon_steps(self, clock: HorizonClock | None = None, override=None, t_now: float | None = None) -> list[StepContact]:
        """N 개의 StepContact.  ref: GaitScheduler.cpp:134-193 (buildConstraintMatrices 의 stance/scale 부분).

        tk = t0 + k*dt_mpc (cycle 원점 기준!), stance = c(side, tk), min_scale = 1.
        override (list[OverrideStep] 또는 .steps 를 가진 객체) 의 k < len 이고 enabled 이면
        stance/min_scale 을 교체, min_scale 은 [0,1] 로 clamp (GaitScheduler.cpp:159-167).

        t_now (변형 'slide', 기본 None = reference 그대로): tk = t_now + k*dt_mpc — 지금부터 미끄러지는 지평.
        reference 는 cycle 원점부터라, 예컨대 오른발 스윙 중에 MPC 가 k≈4~12 에 '왼발 스윙' 을 예정한다 (O1).
        """
        clk = clock if clock is not None else self.clock
        if clk is None:
            raise RuntimeError("GaitScheduler::buildConstraintMatrices requires initialized HorizonClock")
        steps = override
        if steps is not None and not isinstance(steps, (list, tuple)):
            steps = getattr(steps, "steps", steps)
        n_over = len(steps) if steps is not None else 0
        out: list[StepContact] = []
        for k in range(self.N):
            tk = clk.tk(k) if t_now is None else float(t_now) + float(k) * self.dt_mpc
            left = self.contact(LEFT, tk, clk)
            right = self.contact(RIGHT, tk, clk)
            ls, rs = 1.0, 1.0
            if steps is not None and k < n_over and bool(steps[k].enabled):
                st = steps[k]
                left = bool(st.stance[LEFT])
                right = bool(st.stance[RIGHT])
                ls = min(max(float(st.min_scale[LEFT]), 0.0), 1.0)
                rs = min(max(float(st.min_scale[RIGHT]), 0.0), 1.0)
            out.append(StepContact(stance=np.array([left, right], dtype=bool),
                                   min_scale=np.array([ls, rs], dtype=float)))
        return out


def stance_table(steps: list[StepContact]) -> np.ndarray:
    """(N,2) bool 표 — 로그/출력용."""
    return np.array([s.stance for s in steps], dtype=bool)
