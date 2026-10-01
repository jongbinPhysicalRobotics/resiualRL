"""Convex MPC — MIT Humanoid (MuJoCo) 걷기 실행기 (발표용).

Di Carlo SRB convex MPC (상태 13 · 입력 12 · 지평 22 × 0.02 s = 0.44 s · 100 Hz) + OA-MPC 선 접촉 발 제약.
설정은 A30 (Q&A 9/30 Q23, 2026-10-01 발표 설정) 로 고정 — 여기서는 속도 명령과 gait (걷기 / 서기) 만 고른다.
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

import engine

# A30 = Q&A 9/30 Q23 (casadi_mit/README.md 후보 표) + ymax=1.0 (옆 0.8 m/s 를 명령 필터가 자르지 않게).
# 키 뜻은 engine.parse_variant 와 MPC/src/casadi_mit/README.md 참고.
A30 = ("profile=ramp,fix=half+yawref,ypath=0.3,init=warm,yawtraj=1.0,yawff=2,rai=0.2,vdir=0.7,nbs=5,rho=5e-5,"
       "hor=0.44,fref=1e-3,xmax=1.0,ymax=1.0,rvwb=1,replan=1,rpfz=0.08,kdz=1.2,wmax=2.5,armsw=0.3,shkp=60,shkd=12,"
       "shw=2.0,qyaw=750,foot=line,cop=0.075,cmar=0.075,cmarw=0.1,shabs=1,admm=25,admms=50,msc=1,lproj=3,lprojs=1,"
       "minw=0.1,swwn=0.46,tcycle=0.55,tswing=0.22,khalf=0.41,psioff=0,ksx=0.35,ksy=0.47")

LIMITS = {"vx": 1.0, "vy": 1.0, "wz": 2.5}       # 명령 필터 상한 (xmax · ymax · wmax) — 넘는 명령은 잘린다


def build_kwargs(gait: str, yawlead: float, push: str | None = None, extra: str | None = None) -> dict:
    s = A30 + f",yawlead={yawlead:g}"
    if push:
        s += f",push={push}"
    if extra:
        s += "," + extra
    kw = engine.parse_variant(s)
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
    ap.add_argument("--yawlead", type=float, default=0.5, choices=[0.0, 0.5, 1.0],
                    help="회전 선행 (기본 0.5). 0 이면 제자리 2.5 rad/s 가 넘어진다 (A30 의 알려진 한계)")
    ap.add_argument("--push", default=None, metavar="Fx:Fy:t0:dur",
                    help="스크립트 외란 [N, N, s, s] — 진행 방향 기준, 걷기 시작 (서기면 0) 뒤 t0 에 몸통 CoM 에")
    ap.add_argument("--var", default=None, help="고급: A30 위에 덮어쓸 변형 키 'k=v,k=v' (engine.parse_variant 문법)")
    a = ap.parse_args()

    if a.gait == "stand" and (a.vx or a.vy or a.wz):
        print("  [stand] 서기에서는 속도 명령을 쓰지 않는다 — 0 으로 둔다")
        a.vx = a.vy = a.wz = 0.0
    for k, lim in LIMITS.items():
        if abs(getattr(a, k)) > lim + 1e-9:
            print(f"  [주의] --{k} {getattr(a, k):+g} 는 명령 필터 상한 ±{lim:g} 로 잘린다")

    kw = build_kwargs(a.gait, a.yawlead, a.push, a.var)
    print(f"  Convex MPC (A30, yawlead {a.yawlead:g})  gait {a.gait}  cmd vx {a.vx:+.2f} vy {a.vy:+.2f} wz {a.wz:+.2f}"
          f"  {a.seconds:g} s")
    if a.headless:
        r = engine.headless(a.vx, a.vy, a.wz, a.seconds, variant="A30", **kw)
        sys.exit(0 if r["pass"] else 1)
    if a.clean:
        engine._draw_refs = lambda v, ctl, n_seg=16: setattr(v.user_scn, "ngeom", 0)
    engine.view(a.vx, a.vy, a.wz, a.seconds, **kw)


if __name__ == "__main__":
    main()
