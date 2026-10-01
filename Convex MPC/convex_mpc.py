"""Convex MPC — MIT Humanoid (MuJoCo) 걷기 실행기 (발표용).

Di Carlo SRB convex MPC (상태 13 · 입력 12 · 지평 22 × 0.02 s = 0.44 s · 100 Hz) + OA-MPC 선 접촉 발 제약.
설정은 A30 (Q&A 9/30 Q23, 2026-10-01 발표 설정) — 숫자는 전부 mit_controller.yaml 에 있다 (바꾼 줄마다 원래 값 주석).
여기서는 속도 명령과 gait (걷기 / 서기) 만 고른다.
엔진 (engine.py 와 나머지 모듈) 은 MPC/src/casadi_mit 의 복사본이다. 디버깅 · 실험은 그쪽에서 한다.

    python convex_mpc.py --vx 1.0                 전진 1.0 m/s (뷰어)
    python convex_mpc.py --vx 0.6 --wz 1.0        걸으며 돌기
    python convex_mpc.py --wz 2.5                 제자리 회전 2.5 rad/s
    python convex_mpc.py --vy 0.5                 옆 걸음
    python convex_mpc.py --vx -1.0                뒤로 걷기
    python convex_mpc.py --gait stand             제자리 서기
    python convex_mpc.py --vx 1.0 --headless      화면 없이 120 s 판정 (PASS / FAIL)

뷰어에서: ↑ ↓ ← → = 몸통을 앞 · 뒤 · 왼 · 오른쪽으로 0.2 s 밀기, PageUp / PageDown = 미는 힘 ±10 N.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):   # numpy 보다 먼저 (engine.py 와 같음)
    os.environ[_v] = "1"
if sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY"):  # HiDPI Wayland 뷰어 (Q&A 9/25 Q7)
    os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import engine

# 설정은 전부 mit_controller.yaml (A30) 에 있다. yaml 경로가 있는 값은 engine 이 그대로 읽고,
# 생성 인자 (profile · fix · foot · rai · ypath · armsw · OSQP) 는 yaml 의 launcher: 절을 여기서 읽어 넘긴다.
_MERGE = ("osqp_settings", "raw_over", "swing_over", "q_state")   # --var 가 dict 로 주는 인자 — 덮지 않고 합친다


def launcher_kwargs() -> dict:
    """mit_controller.yaml 의 launcher: 절 → engine.headless / engine.view 인자."""
    L = config.load().launcher.to_dict()
    return dict(profile=str(L["profile"]), fixes=tuple(str(x) for x in L["fixes"]), foot_model=str(L["foot_model"]),
                raibert_gain=float(L["raibert_gain"]), ypath=float(L["ypath"]), arm_swing=float(L["arm_swing"]),
                osqp_settings=dict(L["osqp"]))


def build_kwargs(gait: str, yawlead: float | None = None, push: str | None = None, extra: str | None = None) -> dict:
    """yaml (A30) 위에 실행기 옵션을 얹는다. yawlead=None 이면 yaml 값 (swing_foot_yaw_lead_scale)."""
    kw = launcher_kwargs()
    if yawlead is not None:
        kw["yaw_lead"] = float(yawlead)
    if push:
        kw.update(engine.parse_variant(f"push={push}"))
    for k, v in engine.parse_variant(extra).items():
        kw[k] = {**kw[k], **v} if k in _MERGE and isinstance(kw.get(k), dict) else v
    kw["stand"] = gait == "stand"
    return kw


def main():
    ap = argparse.ArgumentParser(description="Convex MPC (MIT Humanoid, A30 설정) — 속도와 gait 만 고른다.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 2)[2])
    ap.add_argument("--vx", type=float, default=0.0, help="전진 속도 [m/s] (뒤로는 음수, 시험 범위 −1.0 ~ 1.0)")
    ap.add_argument("--vy", type=float, default=0.0, help="옆 속도 [m/s] (왼쪽 +, 시험 범위 ±0.8)")
    ap.add_argument("--wz", type=float, default=0.0, help="회전 속도 [rad/s] (반시계 +, 시험 범위 ±2.5)")
    ap.add_argument("--gait", default="walk", choices=["walk", "stand"], help="walk = 걷기 (기본) / stand = 제자리 서기")
    ap.add_argument("--seconds", type=float, default=120.0, help="걷기 (서기) 시간 [s] (기본 120 = 완주 판정 기준)")
    ap.add_argument("--headless", action="store_true", help="뷰어 없이 돌리고 PASS / FAIL 과 지표를 출력")
    ap.add_argument("--clean", action="store_true", help="뷰어의 참조 오버레이 (스윙 궤적 · 착지 목표 · CoM 참조) 끄기")
    ap.add_argument("--yawlead", type=float, default=None, choices=[0.0, 0.5, 1.0],
                    help="회전 선행 (기본 = yaml 0.5). 0 이면 제자리 2.5 rad/s 가 넘어진다 (A30 의 알려진 한계)")
    ap.add_argument("--push", default=None, metavar="Fx:Fy:t0:dur",
                    help="스크립트 외란 [N, N, s, s] — 진행 방향 기준, 걷기 시작 (서기면 0) 뒤 t0 에 몸통 CoM 에. "
                         "음수로 시작하면 --push=-30:0:5:0.2 처럼 = 로 붙인다")
    ap.add_argument("--var", default=None, help="고급: yaml 위에 덮어쓸 변형 키 'k=v,k=v' (engine.parse_variant 문법)")
    a = ap.parse_args()

    cfg = config.load()
    f = cfg.user_command_filter
    if a.gait == "stand" and (a.vx or a.vy or a.wz):
        print("  [stand] 서기에서는 속도 명령을 쓰지 않는다 — 0 으로 둔다")
        a.vx = a.vy = a.wz = 0.0
    for k, lim in (("vx", f.x_dot_max), ("vy", f.y_dot_max), ("wz", f.psi_dot_max)):   # 명령 필터 상한 (yaml)
        if abs(getattr(a, k)) > float(lim) + 1e-9:
            print(f"  [주의] --{k} {getattr(a, k):+g} 는 명령 필터 상한 ±{float(lim):g} 로 잘린다")

    kw = build_kwargs(a.gait, a.yawlead, a.push, a.var)
    lead = a.yawlead if a.yawlead is not None else float(cfg.swing.swing_foot_yaw_lead_scale)
    print(f"  Convex MPC (A30, yawlead {lead:g})  gait {a.gait}  cmd vx {a.vx:+.2f} vy {a.vy:+.2f} wz {a.wz:+.2f}"
          f"  {a.seconds:g} s")
    if a.headless:
        r = engine.headless(a.vx, a.vy, a.wz, a.seconds, variant="A30", **kw)
        sys.exit(0 if r["pass"] else 1)
    if a.clean:
        engine._draw_refs = lambda v, ctl, n_seg=16: setattr(v.user_scn, "ngeom", 0)
    engine.view(a.vx, a.vy, a.wz, a.seconds, **kw)


if __name__ == "__main__":
    main()
