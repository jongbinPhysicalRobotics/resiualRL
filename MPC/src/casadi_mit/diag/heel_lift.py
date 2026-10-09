"""디딤마다 뒤꿈치가 언제 뜨나 (Q&A 10/6 Q31). cop_phase.py 와 같은 창 (걷기 + 3 s 부터 seconds) · 같은 디딤 경과 (접촉표 위상).
디딤 (controller 디딤 · MPC Fz > 30 N) 마다: 50 ~ 60 % 칸에서 뒤꿈치 (발 중심 뒤 접촉점 수직력 > 5 N) 켜짐 비율,
그 칸 안 켜짐/꺼짐 전환 수, 마지막으로 뒤꿈치가 닿아 있던 위상.
사용: heel_lift.py vx vy wz seconds variant
"""
import sys, importlib.util, numpy as np
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(_ROOT))
import mujoco
spec = importlib.util.spec_from_file_location("walk", str(_ROOT / "10_walk.py")); W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
vx, vy, wz, sec, var = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
kw = W.parse_variant(var); prof = kw.pop("profile", "ramp")
cfg, model, ctl, walk_t = W._setup(vx, vy, wz, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
m, d = model.m, model.d; f6 = np.zeros(6)
st = {0: [], 1: []}; cur = {0: None, 1: None}
for _ in range(int(round((walk_t + 3.0 + sec) / m.opt.timestep))):
    tl, ta = ctl.tick(); s = ctl.state; u = np.asarray(ctl.u_hold)
    for leg in (0, 1):
        on = s.t > walk_t + 3.0 and ctl._active(leg, s.t) and u[2 + 3 * leg] > 30
        if not on:
            cur[leg] = None; continue
        if cur[leg] is None:
            cur[leg] = []; st[leg].append(cur[leg])
        Rf = np.asarray(s.foot_R_W[leg]); p0 = np.asarray(s.foot_pos_W[leg]); fh = 0.0
        for c in range(d.ncon):
            con = d.contact[c]
            if model._geom_foot[con.geom1] != leg and model._geom_foot[con.geom2] != leg:
                continue
            mujoco.mj_contactForce(m, d, c, f6)
            if Rf[:, 0].dot(con.pos - p0) <= 0:
                fh += abs(f6[0])
        ph = ctl.gait.phase(leg, s.t, ctl.clock) * ctl.clock.cycle / ctl.gait.stance
        cur[leg].append((ph, fh > 5))
    model.write_torque(tl, ta); mujoco.mj_step(m, d)
allb = []
print("다리 | 50~60 % 칸에서 뒤꿈치 켜짐 비율 | 그 칸 안 켜짐/꺼짐 전환 | 마지막으로 뒤꿈치가 떨어진 위상")
for leg in (0, 1):
    for seq in st[leg]:
        a = np.array(seq, dtype=float)
        if len(a) < 50:
            continue
        k = (a[:, 0] >= 0.5) & (a[:, 0] < 0.6)
        if not k.any():
            continue
        h = a[k, 1]; tog = int(np.sum(np.abs(np.diff(h)) > 0))
        last = a[np.flatnonzero(a[:, 1] > 0.5)[-1], 0] if (a[:, 1] > 0.5).any() else np.nan
        allb.append(h.mean())
        print(f"  {'LR'[leg]}  {h.mean():5.2f}   {tog:2d}   {last:.2f}")
print("평균", np.round(np.mean(allb), 2), "디딤 수", len(allb))
