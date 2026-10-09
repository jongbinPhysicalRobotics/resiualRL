"""발목 한계와 뒤꿈치 뜸 — 어느 쪽이 먼저인가 (Q&A 10/6 Q32). 걷기 시작 + 3 s 부터 seconds 동안, 디딤 (controller 디딤 · MPC Fz > 30 N) 마다:
  t_h1  디딤 20 % 이후 뒤꿈치 (발 중심 뒤 접촉점 수직력 합 > 5 N) 가 처음 뜬 틱
  t_lim 발목 각이 처음 −0.79 rad 이하 (한계 −0.80 의 0.01 안) 가 된 틱
그리고 t_h1 기준 앞뒤 시각의 평균: 발목 각 q_a, 발 pitch θ_f (+ = 발끝 내림 · 뒤꿈치 듦), 정강이 앞 기울기 θ_s (+ = 무릎이 발목보다 앞),
MPC 가 명령한 뒤꿈치 힘 (my + l·Fz)/(2l) · 실제 뒤꿈치 힘, MPC Fz · 실제 Fz.
발목 축 · 무릎 축이 같은 y 축이라 (mit_humanoid.xml) q_a ≈ θ_f − θ_s (평면 근사 — hip yaw · abad 로 조금 어긋남).
사용: ankle_heel.py vx vy wz seconds variant
"""
import sys, importlib.util
import numpy as np
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent          # casadi_mit
sys.path.insert(0, str(_ROOT))
import mujoco
spec = importlib.util.spec_from_file_location("walk", str(_ROOT / "10_walk.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)

ANKLE = ("a10_left_ankle", "a05_right_ankle")
SHANK = ("left_lower_leg", "right_lower_leg")
OFFS = (-60, -40, -20, -10, 0, 10, 20, 40, 60)          # t_h1 기준 [ms]


def run(vx, vy, wz, seconds, var):
    kw = W.parse_variant(var); prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(vx, vy, wz, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
    m, d = model.m, model.d
    dt = m.opt.timestep
    from gait import cfg_get
    l = float(cfg_get(cfg, "mpc.foot_half_length"))
    qadr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in ANKLE]
    sid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in SHANK]
    f6 = np.zeros(6); ez = np.array([0.0, 0.0, 1.0])
    st = {0: [], 1: []}; cur = {0: None, 1: None}
    for _ in range(int(round((walk_t + 3.0 + seconds) / dt))):
        tl, ta = ctl.tick(); s = ctl.state; u = np.asarray(ctl.u_hold)
        for leg in (0, 1):
            if not (s.t > walk_t + 3.0 and ctl._active(leg, s.t) and u[2 + 3 * leg] > 30):
                cur[leg] = None; continue
            if cur[leg] is None:
                cur[leg] = []; st[leg].append(cur[leg])
            Rf = np.asarray(s.foot_R_W[leg]); p0 = np.asarray(s.foot_pos_W[leg]); fh = 0.0; ft = 0.0
            for c in range(d.ncon):
                con = d.contact[c]
                if model._geom_foot[con.geom1] != leg and model._geom_foot[con.geom2] != leg:
                    continue
                mujoco.mj_contactForce(m, d, c, f6)
                if Rf[:, 0].dot(con.pos - p0) <= 0:
                    fh += abs(f6[0])
                else:
                    ft += abs(f6[0])
            xf = Rf[:, 0]; h = np.array([xf[0], xf[1], 0.0]); h /= np.linalg.norm(h)
            th_f = np.arctan2(-xf.dot(ez), xf.dot(h))
            zs = d.xmat[sid[leg]].reshape(3, 3)[:, 2]
            th_s = np.arctan2(zs.dot(h), zs.dot(ez))
            ph = ctl.gait.phase(leg, s.t, ctl.clock) * ctl.clock.cycle / ctl.gait.stance
            yaw = np.arctan2(Rf[1, 0], Rf[0, 0]); My = -np.sin(yaw) * u[6 + 3 * leg] + np.cos(yaw) * u[7 + 3 * leg]
            Fz = u[2 + 3 * leg]
            cur[leg].append((ph, float(fh > 5), d.qpos[qadr[leg]], th_f, th_s, (My + l * Fz) / (2 * l), fh, Fz, fh + ft))
        model.write_torque(tl, ta); mujoco.mj_step(m, d)
        if W._fell(d):
            print("fell", d.time); break
    ev, prof_ = [], []
    for leg in (0, 1):
        for seq in st[leg]:
            a = np.array(seq)
            if len(a) < 60:
                continue
            i0 = int(0.2 * len(a))
            off = np.flatnonzero(a[i0:, 1] < 0.5)
            lim = np.flatnonzero(a[:, 2] <= -0.79)
            if not len(off):
                continue
            i1 = i0 + off[0]
            il = lim[0] if len(lim) else -1
            ev.append([a[i1, 0], (i1 - il) * dt * 1e3 if il >= 0 else np.nan, a[i1, 2], a[i1, 3], a[i1, 4],
                       a[il, 0] if il >= 0 else np.nan])
            row = []
            for o in OFFS:
                j = i1 + int(round(o * 1e-3 / dt))
                row.append(a[j, 2:9] if 0 <= j < len(a) else np.full(7, np.nan))
            prof_.append(row)
    ev = np.array(ev); P = np.nanmean(np.array(prof_), axis=0)
    print(f"vx {vx} vy {vy} wz {wz}  디딤 {len(ev)}")
    print(f"  뒤꿈치 처음 뜸 위상 {np.nanmedian(ev[:, 0]):.2f} ({np.nanmin(ev[:, 0]):.2f} ~ {np.nanmax(ev[:, 0]):.2f})"
          f"  · 발목 한계 도달 위상 {np.nanmedian(ev[:, 5]):.2f} ({np.nanmin(ev[:, 5]):.2f} ~ {np.nanmax(ev[:, 5]):.2f})")
    print(f"  t_h1 − t_lim [ms] 중앙 {np.nanmedian(ev[:, 1]):+.0f} ({np.nanmin(ev[:, 1]):+.0f} ~ {np.nanmax(ev[:, 1]):+.0f}),"
          f" 한계 전 뜸 {np.sum(~(ev[:, 1] >= 0))}/{len(ev)}")
    print(f"  처음 뜰 때: 발목 {np.median(ev[:, 2]):+.3f} rad, 발 pitch {np.degrees(np.median(ev[:, 3])):+.1f}°, 정강이 {np.degrees(np.median(ev[:, 4])):+.1f}°")
    print("  t_h1 기준 [ms]   " + " ".join(f"{o:+5d}" for o in OFFS))
    for k, name, sc, fm in ((0, "발목 q_a [rad]  ", 1.0, "{:+5.2f}"), (1, "발 pitch θ_f [°]", 180 / np.pi, "{:+5.1f}"),
                            (2, "정강이 θ_s [°]  ", 180 / np.pi, "{:+5.1f}"), (3, "MPC 뒤꿈치 [N]  ", 1.0, "{:5.0f}"),
                            (4, "실제 뒤꿈치 [N] ", 1.0, "{:5.0f}"), (5, "MPC Fz [N]      ", 1.0, "{:5.0f}"), (6, "실제 Fz [N]     ", 1.0, "{:5.0f}")):
        print("  " + name + " " + " ".join(fm.format(P[i, k] * sc) for i in range(len(OFFS))))


if __name__ == "__main__":
    run(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
