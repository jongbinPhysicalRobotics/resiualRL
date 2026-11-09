"""mit_controller.yaml 로더 — 중첩 속성 접근 + 파생값 (07_port_design.md §3.2).

    import config
    cfg = config.load()                       # 기본: 이 폴더의 mit_controller.yaml
    cfg.timing.cycle                          # 0.5      (yaml 키 이름 그대로)
    cfg.mpc.walking.state_weight_diag         # np.ndarray (13,)
    cfg.contact_manager.contact_ramp_duration # 0.01
    cfg.port.solver                           # "osqp"
    cfg.dt, cfg.dt_mpc, cfg.N, cfg.stance_frac, cfg.gravity, cfg.half_stance_offset(speed)

규칙
- 숫자 리스트는 float64 np.ndarray 로 바꾼다 (중첩 리스트 → 2-D 배열, 예: swing.nominal_foot_offsets_B (2,3)).
  빈 리스트 (logging.standing_mpc_debug_trigger_times) 는 빈 배열.
- 키는 yaml 그대로. 없는 키 접근은 AttributeError (reference 의 기본값을 여기서 흉내내지 않는다 —
  MIT yaml 은 컨트롤러가 읽는 키를 모두 갖고 있다, spec 05 §11).
- 파생값 (ref 위치):
    dt            = port.physics_dt (0.002)                      config/simulation.yaml, SimulationConfig.cpp:96-104
    dt_mpc        = timing.horizon / timing.horizon_steps (0.02)  ControllerConfig.cpp:645-647
    N             = timing.horizon_steps (25)                     ControllerConfig.cpp:641-643
    stance_frac   = timing.stance / timing.cycle (0.66)           GaitScheduler.cpp:118
    gravity       = model.gravity (-9.81)                         ControllerConfig.cpp:230
    half_stance_offset(speed): speed > 0.65 → 0.26, speed > 0.60 → 0.28, else 0.37
                                                                  My_Controller.cpp:35-45 (선택 순서 그대로: high 먼저)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

import paths


def _convert(value):
    """yaml 값 → Cfg / np.ndarray / 스칼라."""
    if isinstance(value, dict):
        return Cfg(value)
    if isinstance(value, list):
        if len(value) == 0:
            return np.zeros(0)
        try:
            return np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return [_convert(v) for v in value]
    return value


class Cfg:
    """dict → 속성 접근. cfg.a.b, cfg["a"], "a" in cfg, cfg.get("a", default), cfg.to_dict()."""

    def __init__(self, d: dict):
        object.__setattr__(self, "_raw", dict(d))
        object.__setattr__(self, "_data", {str(k): _convert(v) for k, v in d.items()})

    def __getattr__(self, name):
        data = object.__getattribute__(self, "_data")
        try:
            return data[name]
        except KeyError:
            raise AttributeError(f"config key not found: {name!r} (have {sorted(data)})") from None

    def __setattr__(self, name, value):          # 실험용 덮어쓰기 허용 (예: cfg.port.solver = "quadprog")
        self._data[name] = value

    def __getitem__(self, name):
        return self._data[name]

    def __contains__(self, name):
        return name in self._data

    def get(self, name, default=None):
        return self._data.get(name, default)

    def keys(self):
        return self._data.keys()

    def to_dict(self):
        return dict(self._raw)

    def __repr__(self):
        return f"Cfg({', '.join(self._data)})"


class MitConfig(Cfg):
    """최상위 설정 + 파생값."""

    @property
    def dt(self) -> float:
        return float(self.port.physics_dt)

    @property
    def dt_mpc(self) -> float:
        return float(self.timing.horizon) / float(self.timing.horizon_steps)

    @property
    def N(self) -> int:
        return int(self.timing.horizon_steps)

    @property
    def stance_frac(self) -> float:
        return float(self.timing.stance) / float(self.timing.cycle)

    @property
    def gravity(self) -> float:
        return float(self.model.gravity)

    @property
    def iterations_between_solve(self) -> int:
        # ref: My_Controller.cpp:438  max(iterations_between_solve, 1)
        return max(int(self.mpc.iterations_between_solve), 1)

    def half_stance_offset(self, speed: float) -> float:
        """ref: My_Controller.cpp:35-45 selectedBodyVelocityHalfStanceOffset. speed = |[x_dot, y_dot]|."""
        s = self.swing
        if speed > float(s.high_speed_body_velocity_half_stance_offset_switch_speed):
            return float(s.high_speed_body_velocity_half_stance_offset)
        if speed > float(s.body_velocity_half_stance_offset_switch_speed):
            return float(s.mid_speed_body_velocity_half_stance_offset)
        return float(s.body_velocity_half_stance_offset)


def load(path: str | Path | None = None) -> MitConfig:
    """yaml 을 읽어 MitConfig 반환. path=None → paths.CONFIG_YAML (이 폴더의 mit_controller.yaml)."""
    p = Path(path) if path is not None else paths.CONFIG_YAML
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg = MitConfig(raw)
    # 기본 검증 (ref: ControllerConfig.cpp:494 swing+stance == cycle)
    if abs(float(cfg.timing.swing) + float(cfg.timing.stance) - float(cfg.timing.cycle)) > 1e-9:
        raise ValueError("timing.swing + timing.stance must equal timing.cycle")
    if len(cfg.mpc.walking.state_weight_diag) != 13 or len(cfg.mpc.standing.state_weight_diag) != 13:
        raise ValueError("state_weight_diag must have 13 entries")
    if len(cfg.mpc.walking.input_weight_diag) != 12 or len(cfg.mpc.standing.input_weight_diag) != 12:
        raise ValueError("input_weight_diag must have 12 entries")
    return cfg


if __name__ == "__main__":
    c = load()
    print("dt", c.dt, "dt_mpc", c.dt_mpc, "N", c.N, "stance_frac", c.stance_frac, "g", c.gravity)
    print("half_stance_offset(0.5/0.62/0.7)", c.half_stance_offset(0.5), c.half_stance_offset(0.62),
          c.half_stance_offset(0.7))
    print("walking Q", c.mpc.walking.state_weight_diag)
    print("port", c.port.to_dict())
