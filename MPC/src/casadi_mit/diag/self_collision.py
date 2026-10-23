"""다리끼리 자기 충돌 세기 (중간보고서 10/23). 걷기 + 3 s 부터 seconds 동안, 매 시뮬 스텝 (2 ms) MuJoCo 접촉 목록에서
왼 다리 geom ↔ 오른 다리 geom 쌍 (허벅지 · 정강이 · 발 원기둥) 을 센다.
  접촉 시간 비율 = 그런 접촉이 하나라도 있는 스텝 / 전체 스텝
  충돌 횟수     = 접촉 없음 → 있음 으로 바뀐 횟수 (연속 스텝은 한 번)
  최대 겹침     = 접촉 dist 의 최솟값 (음수 = 파고든 깊이) [mm]
  쌍별 몫       = 어떤 부위끼리 닿았나 (스텝 수)
  정강이 최소 간격 = 두 정강이 원기둥 중심선 사이 최소 거리 (반지름 3.5 cm × 2 = 7 cm 아래면 겉면이 닿는 거리)
사용: self_collision.py variant [seconds]   (명령 목록은 아래 CMDS, multiprocessing.Pool 8)
"""
import sys, importlib.util
import numpy as np
from pathlib import Path
from multiprocessing import Pool
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "diag"))
import mujoco
from configs import resolve

CMDS = [(0.6, 0, 0), (1.0, 0, 0), (1.5, 0, 0), (-1.0, 0, 0), (0, 0.6, 0), (0, -0.6, 0),
        (0, 0, 1.0), (0, 0, 2.0), (0, 0, 2.5), (0, 0, 3.0), (0.6, 0, 2.5), (1.0, 0, 1.25), (1.0, 0.5, 0), (1.5, 0.2, 0)]


def _seg_dist(p1, q1, p2, q2):
    """두 선분 사이 최소 거리."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
    t = np.clip((b * s + f) / e, 0, 1)
    s = np.clip((b * t - c) / a, 0, 1)
    return float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * t)))


def _axis(d, g, half):
    c = d.geom_xpos[g]; z = d.geom_xmat[g].reshape(3, 3)[:, 2]
    return c - half * z, c + half * z


def run(args):
    vx, vy, wz, var, seconds = args
    spec = importlib.util.spec_from_file_location("walk", str(_ROOT / "10_walk.py"))
    W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
    kw = W.parse_variant(resolve(var)); prof = kw.pop("profile", "ramp")
    cfg, model, ctl, walk_t = W._setup(vx, vy, wz, kw.pop("solver", "osqp"), True, prof, False, False, 1.0, True, False, **kw)
    m, d = model.m, model.d
    name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    side = {}
    for g in range(m.ngeom):
        n = name(m.geom_bodyid[g])
        if any(k in n for k in ("upper_leg", "lower_leg", "foot")):
            side[g] = ("L" if n.startswith("left") else "R", n.split("_", 1)[1])
    shank = {s: next(g for g, v in side.items() if v == (s, "lower_leg")) for s in ("L", "R")}
    half = float(m.geom_size[shank["L"]][1])
    n_tot = n_hit = events = 0; prev = False; pen = 0.0; pairs = {}; dmin = np.inf; fell = None
    for _ in range(int(round((walk_t + 3.0 + seconds) / m.opt.timestep))):
        tl, ta = ctl.tick(); model.write_torque(tl, ta); mujoco.mj_step(m, d)
        if W._fell(d):
            fell = d.time - walk_t; break
        if d.time < walk_t + 3.0:
            continue
        n_tot += 1; hit = False
        for i in range(d.ncon):
            c = d.contact[i]; a, b = side.get(c.geom1), side.get(c.geom2)
            if a and b and a[0] != b[0]:
                hit = True; pen = min(pen, float(c.dist))
                key = "-".join(sorted((a[1], b[1]))); pairs[key] = pairs.get(key, 0) + 1
        n_hit += hit; events += hit and not prev; prev = hit
        dmin = min(dmin, _seg_dist(*_axis(d, shank["L"], half), *_axis(d, shank["R"], half)))
    return dict(cmd=(vx, vy, wz), frac=n_hit / max(n_tot, 1), events=events, rate=events / (n_tot * m.opt.timestep) if n_tot else 0,
                pen=-1e3 * pen, pairs=pairs, dmin=1e3 * dmin, fell=fell)


if __name__ == "__main__":
    var = sys.argv[1]; sec = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    with Pool(8) as pool:
        res = pool.map(run, [(vx, vy, wz, var, sec) for vx, vy, wz in CMDS])
    print(f"[{var}]  {sec:.0f} s 측정 (걷기 + 3 s 부터)")
    print("  명령 (vx, vy, wz)      접촉 시간 %  충돌 횟수 (회/s)  최대 겹침 mm  정강이 최소 간격 mm  부위 (스텝)")
    for r in res:
        vx, vy, wz = r["cmd"]
        tag = f"넘어짐 {r['fell']:.1f} s" if r["fell"] is not None else ""
        pr = ", ".join(f"{k} {v}" for k, v in sorted(r["pairs"].items(), key=lambda kv: -kv[1]))
        print(f"  ({vx:+.1f}, {vy:+.1f}, {wz:+.2f})   {100*r['frac']:6.2f}      {r['events']:4d} ({r['rate']:.2f})      {r['pen']:5.1f}         {r['dmin']:5.0f}          {pr} {tag}")
