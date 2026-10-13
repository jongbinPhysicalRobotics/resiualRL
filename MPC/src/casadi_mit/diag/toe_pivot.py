"""발끝 한 점 구간의 발 yaw 회전 (Q&A 10/6 Q61). 걷기 시작 + 3 s 부터 seconds 동안, 디딤 (실제 접촉) 마다:
  - 두 점 구간 · 발끝만 구간 · 뒤꿈치만 구간의 길이 [%] 와 그동안 발 yaw 변화 [°] (+ = 반시계)
  - 발끝만 구간에서 다리가 가한 비틀림 (최종 렌치, 지면 → 로봇, 발 site 둘레 Mz) 과 한 점이 받아 줄 수 있는 l·Fy 의 차 |Mz − l·Fy| [N·m]
    (최종 렌치 = stance_torque 입력: MPC 해 자른 값 + yaw hold (+ holdclip), oa_check 와 같은 걸이)
  - 디딤 전체 발 yaw 변화 · slip
사용: toe_pivot.py vx vy wz seconds variant
"""
import sys, importlib.util
import numpy as np
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "diag"))
import mujoco
spec = importlib.util.spec_from_file_location("walk", str(_ROOT / "10_walk.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
import leg_control as LC
import controller as CT
from configs import resolve
from gait import cfg_get

TH = 5.0


def run(vx, vy, wz, seconds, var):
    kw = W.parse_variant(resolve(var)); prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(vx, vy, wz, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
    m, d = model.m, model.d
    l = float(cfg_get(cfg, "mpc.foot_half_length"))
    cur = {}; applied = {0: None, 1: None}
    o_w, o_s = CT.walking_stance_torque, LC.stance_torque

    def ww(dyn, state, leg, u0, alpha, td, cfg_, yaw_hold_gains=None, line_proj=None):
        cur["leg"] = leg
        return o_w(dyn, state, leg, u0, alpha, td, cfg_, yaw_hold_gains=yaw_hold_gains, line_proj=line_proj)

    def ws(dyn, F, M):
        applied[cur.get("leg", 0)] = (-np.asarray(F, float), -np.asarray(M, float))   # 지면 → 로봇 (월드)
        return o_s(dyn, F, M)

    CT.walking_stance_torque, LC.stance_torque = ww, ws
    f6 = np.zeros(6); rows = []; fell = -1.0
    for _ in range(int(round((walk_t + 3.0 + seconds) / m.opt.timestep))):
        applied[0] = applied[1] = None
        tl, ta = ctl.tick(); model.write_torque(tl, ta); mujoco.mj_step(m, d)
        if W._fell(d):
            fell = d.time; break
        s = ctl.state
        if s.t < walk_t + 3.0:
            continue
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
            yaw = float(np.arctan2(Rf[1, 0], Rf[0, 0]))
            mz = fyl = np.nan
            if applied[leg] is not None:
                c_, s_ = np.cos(yaw), np.sin(yaw)
                Rt = np.array([[c_, s_, 0], [-s_, c_, 0], [0, 0, 1]])
                Fl, Ml = Rt @ applied[leg][0], Rt @ applied[leg][1]
                mz, fyl = Ml[2], l * Fl[1]
            row += [fh, ft, yaw, p0[0], p0[1], mz, fyl]
        rows.append(row)
    R = np.array(rows)
    out = {"flat": [], "toe": [], "heel": [], "mis": [], "tot": [], "slip": [], "frac": []}
    for leg in (0, 1):
        b = 1 + 7 * leg
        fh, ft = R[:, b] > TH, R[:, b + 1] > TH
        yaw = np.unwrap(R[:, b + 2]); xy = R[:, b + 3:b + 5]; mz, fyl = R[:, b + 5], R[:, b + 6]
        on = fh | ft
        i = 1
        while i < len(R):
            if on[i] and not on[i - 1]:
                j = i
                while j < len(R) and on[j]:
                    j += 1
                if j >= len(R):
                    break
                if j - i > 50:
                    seg = slice(i, j)
                    both, toe, heel = (fh & ft)[seg], (ft & ~fh)[seg], (fh & ~ft)[seg]
                    dy = np.diff(yaw[seg], prepend=yaw[i])
                    out["flat"].append(np.degrees(dy[both].sum())); out["toe"].append(np.degrees(dy[toe].sum()))
                    out["heel"].append(np.degrees(dy[heel].sum())); out["tot"].append(np.degrees(yaw[j - 1] - yaw[i]))
                    out["frac"].append([both.mean(), toe.mean(), heel.mean()])
                    out["slip"].append(1e3 * np.linalg.norm(xy[j - 1] - xy[i]))
                    k = np.flatnonzero(toe & np.isfinite(mz[seg]))
                    if len(k):
                        out["mis"].append(np.abs(mz[seg][k] - fyl[seg][k]).mean())
                i = j
            i += 1
    F = np.array(out["frac"])
    a = lambda k: np.abs(np.array(out[k]))
    print(f"[vx {vx} vy {vy} wz {wz}] {'넘어짐 %.1f s' % fell if fell > 0 else '버팀'}  디딤 {len(F)}  (두 점 {100*F[:,0].mean():.0f} % · 발끝만 {100*F[:,1].mean():.0f} % · 뒤꿈치만 {100*F[:,2].mean():.0f} %)")
    print(f"   발 yaw 변화 |·| 평균 (p90) [°]: 두 점 구간 {a('flat').mean():.2f} ({np.percentile(a('flat'),90):.2f}) · 발끝만 {a('toe').mean():.2f} ({np.percentile(a('toe'),90):.2f})"
          f" · 뒤꿈치만 {a('heel').mean():.2f} · 디딤 전체 {a('tot').mean():.2f} ({np.percentile(a('tot'),90):.2f})")
    if out["mis"]:
        print(f"   발끝만 구간 |Mz − l·Fy| 평균 {np.mean(out['mis']):.2f} N·m (p90 {np.percentile(out['mis'], 90):.2f})   slip {np.mean(out['slip']):.1f} mm")


if __name__ == "__main__":
    run(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
