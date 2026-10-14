"""발끝만 구간 (뒤꿈치 힘 0) 마다 발 heading · roll · pitch 변화와 발끝 꼭짓점 (발 좌표 (0.105, ±0.02 / 0, −0.04)) 의 수평 이동 (Q&A 10/6 Q65).
사용: toe_disp.py vx variant [noslip_iterations]   (환경변수 STIFF=시간상수:d 로 모든 geom 접촉 강성 변경 — 진단 전용)
"""
import sys, importlib.util
import numpy as np
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/diag")
import mujoco
spec = importlib.util.spec_from_file_location("walk", ROOT + "/10_walk.py")
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
from configs import resolve
vx, var = float(sys.argv[1]), sys.argv[2]
ns = int(sys.argv[3]) if len(sys.argv) > 3 else 0
kw = W.parse_variant(resolve(var)); prof = kw.pop("profile", "ramp")
cfg, model, ctl, walk_t = W._setup(vx, 0, 0, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
m, d = model.m, model.d
m.opt.noslip_iterations = ns
import os
if os.environ.get("STIFF"):                      # 접촉 강성: 모든 geom solref 시간상수 · solimp d
    tc, d0 = (float(x) for x in os.environ["STIFF"].split(":"))
    m.geom_solref[:, 0] = tc; m.geom_solimp[:, 0] = d0; m.geom_solimp[:, 1] = max(d0, 0.95) if d0 < 0.99 else 0.999
f6 = np.zeros(6)
P = {"L": np.array([0.105, 0.02, -0.04]), "C": np.array([0.105, 0.0, -0.04]), "R": np.array([0.105, -0.02, -0.04])}
seg = {0: None, 1: None}; out = []
for _ in range(int(round((walk_t + 8.0) / m.opt.timestep))):
    tl, ta = ctl.tick(); model.write_torque(tl, ta); mujoco.mj_step(m, d)
    if W._fell(d): break
    if d.time < walk_t + 3.0: continue
    for leg in (0, 1):
        fb = model.foot_body[leg]; R = d.xmat[fb].reshape(3, 3).copy(); p = d.xpos[fb].copy()
        fh = ft = 0.0
        for c in range(d.ncon):
            con = d.contact[c]
            if model._geom_foot[con.geom1] != leg and model._geom_foot[con.geom2] != leg: continue
            mujoco.mj_contactForce(m, d, c, f6)
            if (R.T @ (con.pos - p))[0] < 0.03: fh += f6[0]
            else: ft += f6[0]
        toe_only = ft > 5 and fh < 5
        snap = dict(pts={k: p + R @ v for k, v in P.items()}, head=np.arctan2(R[1, 0], R[0, 0]),
                    roll=np.arctan2(R[2, 1], R[2, 2]), pitch=np.arcsin(-R[2, 0]))
        if toe_only and seg[leg] is None:
            seg[leg] = [snap, snap]
        elif toe_only:
            seg[leg][1] = snap
        elif seg[leg] is not None:
            a, b = seg[leg]
            out.append([np.degrees(b["head"] - a["head"]), *(1e3 * np.linalg.norm(b["pts"][k][:2] - a["pts"][k][:2]) for k in "LCR"),
                        np.degrees(b["roll"] - a["roll"]), np.degrees(b["pitch"] - a["pitch"]), np.degrees(a["roll"])])
            seg[leg] = None
A = np.array(out)
med = np.median(np.abs(A), axis=0)
print(f"[vx {vx} {var.split(',')[-1]} noslip {m.opt.noslip_iterations} stiff {os.environ.get('STIFF','기본')}] 발끝만 구간 {len(A)} 개 중앙: heading 변화 {med[0]:.2f}°, 수평 이동 왼 모서리 {med[1]:.1f} · 가운데 {med[2]:.1f} · 오른 모서리 {med[3]:.1f} mm,"
      f" roll 변화 {med[4]:.1f}°, pitch 변화 {med[5]:.1f}°, 시작 roll {med[6]:.1f}°")
