"""고속 전도의 시작 — 스윙 초반 발이 땅에 다시 닿는가 (Q&A 10/6 Q61). 걷기 시작 + 2 s 부터:
  - 스윙 (일정상 공중) 마다: 이륙 뒤 발이 떨어진 시각, 다시 디딤으로 바뀐 시각 (조기 접촉 = 일정상 스윙인데 active),
    스윙 20 ~ 100 ms 의 발끝 · 뒤꿈치 최저 높이와 그때 스윙 목표 높이
  - 정상 스윙 (끝까지 감) 의 중앙값과, 스윙 50 % 전에 디딤으로 바뀐 스윙의 시간표 (8 ms 마다)
발끝 · 뒤꿈치 = 발 site ± l · 발 x 축 (z 는 site 높이 기준, 바닥 접촉 때 ≈ 0).
사용: swing_scuff.py vx seconds variant
"""
import sys, importlib.util
import numpy as np
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "diag"))
import mujoco
spec = importlib.util.spec_from_file_location("walk", str(_ROOT / "10_walk.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
from configs import resolve
from gait import cfg_get

TH = 5.0


def run(vx, seconds, var):
    kw = W.parse_variant(resolve(var)); prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(vx, 0.0, 0.0, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
    m, d = model.m, model.d
    l = float(cfg_get(cfg, "mpc.foot_half_length"))
    lim = model.ctrl_hi[model.leg_act.ravel()]
    f6 = np.zeros(6); rows = []; fell = -1.0
    for _ in range(int(round((walk_t + seconds) / m.opt.timestep))):
        tl, ta = ctl.tick(); model.write_torque(tl, ta); mujoco.mj_step(m, d)
        if W._fell(d):
            fell = d.time; break
        s = ctl.state
        if s.t < walk_t + 2.0:
            continue
        pd = ctl.swing_pdes(); tau = np.asarray(tl).reshape(-1)
        row = [s.t]
        for leg in (0, 1):
            Rf = np.asarray(s.foot_R_W[leg]); p0 = np.asarray(s.foot_pos_W[leg]); fh = ft = 0.0
            for c in range(d.ncon):
                con = d.contact[c]
                if model._geom_foot[con.geom1] != leg and model._geom_foot[con.geom2] != leg:
                    continue
                mujoco.mj_contactForce(m, d, c, f6)
                if Rf[:, 0].dot(con.pos - p0) > 0:
                    ft += abs(f6[0])
                else:
                    fh += abs(f6[0])
            row += [float(ctl.cm.scheduled[leg]), float(ctl._active(leg, s.t)), float(ctl.cm.estimated[leg]),
                    p0[2] + l * Rf[2, 0], p0[2] - l * Rf[2, 0], p0[2], pd[leg][2], np.degrees(np.arcsin(-Rf[2, 0])),
                    abs(tau[leg * 5 + 2]) / lim[leg * 5 + 2], abs(tau[leg * 5 + 3]) / lim[leg * 5 + 3], ft, fh,
                    (p0[0] - s.torso_pos_W[0]) * np.cos(s.yaw_unwrapped) + (p0[1] - s.torso_pos_W[1]) * np.sin(s.yaw_unwrapped)]
        rows.append(row)
    R = np.array(rows); t = R[:, 0]; dt = float(np.median(np.diff(t)))
    print(f"[vx {vx}] {'넘어짐 %.1f s (걷기 + %.1f)' % (fell, fell - walk_t) if fell > 0 else '버팀'}")
    NC = 13
    normal, bad = [], []
    for leg in (0, 1):
        b = 1 + NC * leg
        sch, act, est = R[:, b] > 0.5, R[:, b + 1] > 0.5, R[:, b + 2] > 0.5
        toe, heel, site, pz, pit, shp, skn, ft, fh, fwd = (R[:, b + k] for k in range(3, 13))
        for i in np.flatnonzero(~sch[1:] & sch[:-1]) + 1:            # 일정상 이륙
            j = i
            while j < len(t) and not sch[j]:
                j += 1
            if j >= len(t):
                continue
            Tsw = (j - i) * dt
            rel = np.flatnonzero(~est[i:j])
            if not len(rel):
                continue
            r0 = i + rel[0]
            ea = np.flatnonzero(act[r0:j])
            te = (r0 + ea[0] - i) * dt if len(ea) else None
            w = slice(i + int(0.02 / dt), i + int(0.10 / dt))
            ks = [i + int(round(q / dt)) for q in np.arange(0.0, 0.0961, 0.008)]
            info = dict(leg=leg, t=t[i], Tsw=Tsw, rel=(r0 - i) * dt, te=te, toe_min=toe[w].min(), heel_min=heel[w].min(),
                        pz_at=pz[w][np.argmin(toe[w])], sat_hp=(shp[i:i + int(0.08 / dt)] >= 1).mean(), fwd=fwd[i], i=i,
                        tl=np.array([[toe[k], site[k], pz[k], pit[k], shp[k]] for k in ks]))
            if te is not None and te < 0.5 * Tsw:
                bad.append(info)
                if len(bad) <= 2:
                    print(f"  ▶ 조기 디딤: {'LR'[leg]} t {t[i]:.2f} s, 발 떨어짐 {1e3*info['rel']:.0f} ms, 디딤으로 {1e3*te:.0f} ms (스윙 {1e3*Tsw:.0f} ms), 이륙 때 발 몸통 앞 {1e3*fwd[i]:+.0f} mm")
                    print("     ms   발끝z  뒤꿈z  site z 목표z  발pitch  hipP τ/한계 무릎   발끝N 뒤꿈N  est act")
                    for k in range(i, min(i + int((te + 0.02) / dt) + 1, j), max(int(0.008 / dt), 1)):
                        print(f"    {1e3*(t[k]-t[i]):4.0f} {1e3*toe[k]:6.1f} {1e3*heel[k]:6.1f} {1e3*site[k]:6.1f} {1e3*pz[k]:6.1f}  {pit[k]:+6.1f}   {shp[k]:5.2f}      {skn[k]:4.2f}  {ft[k]:6.0f} {fh[k]:5.0f}   {int(est[k])}   {int(act[k])}")
            else:                                                     # 끝까지 갔거나 후반 (50 % 뒤) 에 닿음
                normal.append(info)
    if normal:
        A = lambda k: np.array([x[k] for x in normal])
        print(f"  정상 스윙 {len(normal)} 개 중앙: 발 떨어짐 {1e3*np.median(A('rel')):.0f} ms, 20 ~ 100 ms 발끝 최저 {1e3*np.median(A('toe_min')):.1f} mm"
              f" (p10 {1e3*np.percentile(A('toe_min'),10):.1f}) · 그때 목표 {1e3*np.median(A('pz_at')):.1f} mm, 뒤꿈치 최저 {1e3*np.median(A('heel_min')):.1f} mm,"
              f" 처음 80 ms hip pitch 포화 {100*np.median(A('sat_hp')):.0f} %, 이륙 때 발 몸통 앞 {1e3*np.median(A('fwd')):+.0f} mm")
        T = np.nanmedian(np.array([x["tl"] for x in normal]), axis=0)
        print("     ms   발끝z  site z 목표z  발pitch  hipP τ/한계  (정상 스윙 중앙)")
        for q, r_ in zip(np.arange(0, 97, 8), T):
            print(f"    {q:4.0f} {1e3*r_[0]:6.1f} {1e3*r_[1]:6.1f} {1e3*r_[2]:6.1f}  {r_[3]:+6.1f}   {r_[4]:5.2f}")
    print(f"  스윙 50 % 전 디딤으로 바뀐 스윙: {len(bad)} 개" + (f" (첫 번째 t {bad[0]['t']:.2f} s)" if bad else ""))


if __name__ == "__main__":
    run(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
