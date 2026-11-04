"""Convex MPC — MIT Humanoid (MuJoCo) 걷기 실행기 (발표용).

Di Carlo SRB convex MPC (상태 13 · 입력 12 · 지평 22 × 0.02 s = 0.44 s · 100 Hz) + OA-MPC 선 접촉 발 제약.
설정은 A42 (Q&A 10/6 Q102, MPC 최종 설정 — A30 발표 설정에서 10 가지를 바꾼 A41 + 옆 착지 앞먹임 ksy 0.75) — 숫자는 전부 mit_controller.yaml 에 있다 (바꾼 줄마다 원래 값 주석).
여기서는 속도 명령과 gait (걷기 / 서기) 만 고른다.
엔진 (engine.py 와 나머지 모듈) 은 MPC/src/casadi_mit 의 복사본이다. 디버깅 · 실험은 그쪽에서 한다.

    python convex_mpc.py --vx 1.0                 전진 1.0 m/s (뷰어)
    python convex_mpc.py --vx 0.6 --wz 1.0        걸으며 돌기
    python convex_mpc.py --wz 2.5                 제자리 회전 2.5 rad/s
    python convex_mpc.py --vy 0.5                 옆 걸음
    python convex_mpc.py --vx -1.0                뒤로 걷기
    python convex_mpc.py --gait stand             제자리 서기
    python convex_mpc.py --vx 1.0 --headless      화면 없이 120 s 판정 (PASS / FAIL)
    python convex_mpc.py --vx 1.0 --seconds 15 --record vx10.mp4    MP4 로 저장 (재생 속도 = 실제)

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

# 설정은 전부 mit_controller.yaml (A42) 에 있다. yaml 경로가 있는 값은 engine 이 그대로 읽고,
# 생성 인자 (profile · fix · foot · rai · ypath · armsw · OSQP) 는 yaml 의 launcher: 절을 여기서 읽어 넘긴다.
_MERGE = ("osqp_settings", "raw_over", "swing_over", "q_state")   # --var 가 dict 로 주는 인자 — 덮지 않고 합친다


def launcher_kwargs() -> dict:
    """mit_controller.yaml 의 launcher: 절 → engine.headless / engine.view 인자."""
    L = config.load().launcher.to_dict()
    return dict(profile=str(L["profile"]), fixes=tuple(str(x) for x in L["fixes"]), foot_model=str(L["foot_model"]),
                raibert_gain=float(L["raibert_gain"]), ypath=float(L["ypath"]), arm_swing=float(L["arm_swing"]),
                osqp_settings=dict(L["osqp"]))


def build_kwargs(gait: str, yawlead: float | None = None, push: str | None = None, extra: str | None = None) -> dict:
    """yaml (A42) 위에 실행기 옵션을 얹는다. yawlead=None 이면 yaml 값 (swing_foot_yaw_lead_scale)."""
    kw = launcher_kwargs()
    if yawlead is not None:
        kw["yaw_lead"] = float(yawlead)
    if push and "," in push:                                     # 외란 여러 개 (--record 전용): 'Fx:Fy:t0:dur,Fx:Fy:t0:dur'
        kw["pushes"] = [engine.parse_variant(f"push={p_}")["push"] for p_ in push.split(",") if p_.strip()]
    elif push:
        kw.update(engine.parse_variant(f"push={push}"))
    for k, v in engine.parse_variant(extra).items():
        kw[k] = {**kw[k], **v} if k in _MERGE and isinstance(kw.get(k), dict) else v
    kw["stand"] = gait == "stand"
    return kw


class _AppendScene:
    """engine._draw_refs 는 뷰어의 빈 user_scn 을 비우고 (ngeom = 0) 채운다. 렌더러 장면에서는 로봇 geom 뒤에 덧붙이게 한다."""

    def __init__(self, scn):
        self._s, self._base = scn, scn.ngeom

    geoms = property(lambda self: self._s.geoms)
    maxgeom = property(lambda self: self._s.maxgeom)

    @property
    def ngeom(self):
        return self._s.ngeom

    @ngeom.setter
    def ngeom(self, v):
        self._s.ngeom = max(int(v), self._base)


def record(vx, vy, wz, seconds, out, *, fps=50, size=(1280, 720), clean=False, hud=True, **kw):
    """뷰어 없이 MP4 로 저장. 프레임은 시뮬 시간 기준이라 재생 속도가 실제와 같다 (계산이 느려도).
    시뮬 루프 · 외란은 engine.headless 와 같은 순서 — 화면 그리기는 제어에 영향을 주지 않는다."""
    import types

    import imageio.v2 as iio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    push, push_repeat = kw.pop("push", None), float(kw.pop("push_repeat", 0.0) or 0.0)
    pushes = list(kw.pop("pushes", None) or ([push] if push is not None else []))   # (Fx, Fy, t0, dur) 목록
    pos = dict(solver="osqp", armature=True, profile="const", markers=False, stand=False, friction_scale=1.0,
               fsm_double_update=True)
    for k in list(kw):
        if k in pos:
            pos[k] = kw.pop(k)
    cfg, model, ctl, walk_t = engine._setup(vx, vy, wz, pos["solver"], pos["armature"], pos["profile"], pos["markers"],
                                            pos["stand"], pos["friction_scale"], pos["fsm_double_update"], False, **kw)
    m, d = model.m, model.d
    stand = pos["stand"]
    t_meas0 = 0.0 if stand else walk_t
    total = t_meas0 + seconds

    W, H = size
    m.vis.global_.offwidth, m.vis.global_.offheight = max(W, m.vis.global_.offwidth), max(H, m.vis.global_.offheight)
    renderer = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()                                     # 뷰어와 같은 추적 카메라
    cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, model.base_body
    cam.distance, cam.elevation, cam.azimuth = 2.5, -15.0, 135.0
    every = max(1, int(round(1.0 / (fps * m.opt.timestep))))     # 50 fps = 10 틱마다
    fps_real = 1.0 / (every * m.opt.timestep)
    font = ImageFont.load_default(size=max(14, H // 32))
    buf = np.zeros((max(1, int(round(0.5 * fps_real))), 3))     # 실측 속도 0.5 s 평균
    writer = iio.get_writer(out, fps=fps_real, codec="libx264", quality=8, macro_block_size=2)
    print(f"  녹화 → {out}  ({W}×{H}, {fps_real:g} fps, 시뮬 {total:g} s)", flush=True)

    n_sim = int(round(total / m.opt.timestep))                  # engine.headless 와 같은 틱 수 (시간 누적 오차로 한 틱 더 돌지 않게)
    n_after = int(round(1.0 / m.opt.timestep))                  # 넘어진 뒤 더 찍는 틱 수 (1 s)
    k, n_frame, t_fell, k_fell, F_B = 0, 0, None, None, None
    while k < n_sim and (k_fell is None or k < k_fell + n_after):
        if t_fell is None:                                       # 시뮬 한 틱 (engine.headless 와 같은 순서)
            tau_leg, tau_arm = ctl.tick()
            model.write_torque(tau_leg, tau_arm)
            F_B = None
            if pushes:
                for p_ in pushes:                                # 하나면 engine.headless 와 같다 (반복 주기 포함)
                    if engine._push_active(p_, push_repeat, d.time - t_meas0):
                        F_B = (p_[0], p_[1])
                        break
                if F_B is not None:
                    d.xfrc_applied[model.base_body, 0:3] = engine._push_world(F_B[0], F_B[1], ctl.state.yaw_unwrapped)
                else:
                    d.xfrc_applied[model.base_body, :] = 0.0
            mujoco.mj_step(m, d)
            if engine._fell(d):
                t_fell, k_fell = float(d.time), k
                print(f"  넘어짐 t = {t_fell:.2f} s — 1 s 더 찍고 끝냄", flush=True)
        else:                                                    # 넘어진 뒤: 토크 없이 물리만 (화면용)
            d.ctrl[:] = 0.0
            mujoco.mj_step(m, d)
        k += 1
        if k % every == 0:                                       # 프레임 (틱 뒤에 그림 — 컨트롤러가 첫 틱에 초기화된다)
            renderer.update_scene(d, cam)
            scn = renderer.scene
            if not clean:
                engine._draw_refs(types.SimpleNamespace(user_scn=_AppendScene(scn)), ctl)
            if F_B is not None and scn.ngeom < scn.maxgeom:      # 외란 화살표 (1 cm/N, 빨강) — 뷰어와 같음
                p0 = d.xipos[model.base_body].copy()
                Fw = engine._push_world(F_B[0], F_B[1], ctl.state.yaw_unwrapped)
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                                    np.eye(3).flatten(), np.array([1.0, 0.1, 0.1, 1.0], np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.02, p0 - 0.01 * Fw, p0)
                scn.ngeom += 1
            img = renderer.render()
            if hud:
                mujoco.mj_subtreeVel(m, d)                       # 전신 CoM 속도 (컨트롤러는 읽기 전에 스스로 다시 계산)
                yaw = ctl.state.yaw_unwrapped
                c_, s_ = np.cos(yaw), np.sin(yaw)
                v = d.subtree_linvel[0]
                buf[n_frame % len(buf)] = (c_ * v[0] + s_ * v[1], -s_ * v[0] + c_ * v[1], d.cvel[model.base_body][2])
                avg = buf[:min(n_frame + 1, len(buf))].mean(axis=0)
                f = ctl.filtered
                lines = [f"t {d.time:5.1f} s   {'stand' if stand else 'walk'}",
                         f"cmd   vx {f.x_dot:+.2f}  vy {f.y_dot:+.2f} m/s  wz {f.psi_dot:+.2f} rad/s",
                         f"meas  vx {avg[0]:+.2f}  vy {avg[1]:+.2f} m/s  wz {avg[2]:+.2f} rad/s"]
                if F_B is not None:
                    ang = np.degrees(np.arctan2(F_B[1], F_B[0]))   # 몸 진행 방향 기준: 0 앞 · 90 왼 · ±180 뒤 · −90 오른
                    side = ("front", "left", "back", "right")[int(((ang + 45.0) % 360.0) // 90.0)]
                    lines.append(f"push  {np.hypot(*F_B):.0f} N  {side}")
                if t_fell is not None:
                    lines.append("FELL")
                pil = Image.fromarray(img)
                ImageDraw.Draw(pil).multiline_text((16, 12), "\n".join(lines), font=font, fill=(255, 255, 255),
                                                   stroke_width=2, stroke_fill=(0, 0, 0), spacing=6)
                img = np.asarray(pil)
            writer.append_data(img)
            n_frame += 1
            if n_frame % int(10 * fps_real) == 0:
                print(f"  t {d.time:6.1f} s  프레임 {n_frame}", flush=True)
    writer.close()
    renderer.close()
    print(f"  저장 완료: {out}  ({n_frame} 프레임, {n_frame / fps_real:.1f} s)", flush=True)
    return t_fell is None


def main():
    ap = argparse.ArgumentParser(description="Convex MPC (MIT Humanoid, A42 설정) — 속도와 gait 만 고른다.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 2)[2])
    ap.add_argument("--vx", type=float, default=0.0, help="전진 속도 [m/s] (뒤로는 음수, 120 s 완주 범위 −1.1 ~ 1.5)")
    ap.add_argument("--vy", type=float, default=0.0, help="옆 속도 [m/s] (왼쪽 +, 120 s 완주 범위 ±0.9 — 전진 1.3 과 함께면 0.3)")
    ap.add_argument("--wz", type=float, default=0.0, help="회전 속도 [rad/s] (반시계 +, 제자리 120 s 완주 3.0)")
    ap.add_argument("--gait", default="walk", choices=["walk", "stand"], help="walk = 걷기 (기본) / stand = 제자리 서기")
    ap.add_argument("--seconds", type=float, default=120.0, help="걷기 (서기) 시간 [s] (기본 120 = 완주 판정 기준)")
    ap.add_argument("--headless", action="store_true", help="뷰어 없이 돌리고 PASS / FAIL 과 지표를 출력")
    ap.add_argument("--clean", action="store_true", help="뷰어의 참조 오버레이 (스윙 궤적 · 착지 목표 · CoM 참조) 끄기")
    ap.add_argument("--yawlead", type=float, default=None, choices=[0.0, 0.5, 1.0],
                    help="회전 선행 (기본 = yaml 1.0). 0 이면 제자리 2.5 rad/s 가 넘어진다 (알려진 한계)")
    ap.add_argument("--push", default=None, metavar="Fx:Fy:t0:dur",
                    help="스크립트 외란 [N, N, s, s] — 진행 방향 기준, 걷기 시작 (서기면 0) 뒤 t0 에 몸통 CoM 에. "
                         "음수로 시작하면 --push=-30:0:5:0.2 처럼 = 로 붙인다. "
                         "--record 에서는 쉼표로 여러 개: --push 30:0:5:0.2,-30:0:8:0.2")
    ap.add_argument("--var", default=None, help="고급: yaml 위에 덮어쓸 변형 키 'k=v,k=v' (engine.parse_variant 문법)")
    ap.add_argument("--record", default=None, metavar="FILE.mp4",
                    help="뷰어 대신 MP4 로 저장 (시뮬 시간 기준 프레임 → 재생 속도 = 실제). --seconds 로 길이를 정한다")
    ap.add_argument("--fps", type=float, default=50.0, help="녹화 프레임률 (기본 50 = 10 틱마다)")
    ap.add_argument("--size", default="1280x720", help="녹화 해상도 가로x세로 (기본 1280x720, 예: 1920x1080)")
    ap.add_argument("--nohud", action="store_true", help="녹화 화면의 글자 (시간 · 명령 · 실측 속도) 끄기")
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
    print(f"  Convex MPC (A42, yawlead {lead:g})  gait {a.gait}  cmd vx {a.vx:+.2f} vy {a.vy:+.2f} wz {a.wz:+.2f}"
          f"  {a.seconds:g} s")
    if "pushes" in kw and not a.record:
        sys.exit("  외란 여러 개 (쉼표) 는 --record 에서만 된다 — 뷰어 · 헤드리스는 하나만")
    if a.headless:
        r = engine.headless(a.vx, a.vy, a.wz, a.seconds, variant="A42", **kw)
        sys.exit(0 if r["pass"] else 1)
    if a.record:
        w, h = (int(x) for x in a.size.lower().split("x"))
        ok = record(a.vx, a.vy, a.wz, a.seconds, a.record, fps=a.fps, size=(w, h), clean=a.clean, hud=not a.nohud, **kw)
        sys.exit(0 if ok else 1)
    if a.clean:
        engine._draw_refs = lambda v, ctl, n_seg=16: setattr(v.user_scn, "ngeom", 0)
    engine.view(a.vx, a.vy, a.wz, a.seconds, **kw)


if __name__ == "__main__":
    main()
