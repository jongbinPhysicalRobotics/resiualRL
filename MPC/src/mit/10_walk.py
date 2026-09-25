"""MIT 휴머노이드 걷기 러너 — reference convex-MPC 포팅 (controller.MitController) 을 MuJoCo 에서 돌린다.

루프 (07 §2, spec 05 §1.2): tau = controller.tick()  (현재 MjData, mj_forward 없음) → write_torque (ctrlrange clamp)
→ mj_step.  시작: keyframe 'stand' + mj_forward 한 번 (D2). 컨트롤러는 t = 0 에 초기화 → StandingSettle 1.0 s → Walking.

명령 profile (D3):
    const (기본) : raw 명령이 첫 틱부터 있음 (reference headless 기본 의미 — filter 첫 호출 snap + walking 시작 틱 dt=0 snap)
    step         : raw 명령이 walking 시작 (settle 1.0 s) 부터
    ramp         : walking 시작부터 2 s 램프 (reference 램프 공식 x = sign·min(|v|, rate (t − t0)))
모두 reference 필터 (tau 0.92/0.8/0.70 s, clamp 0.7/0.5/2.0) 를 거친다.

실행 (ROOT 에서):
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe MPC/src/mit/10_walk.py --vx 0.6 --seconds 120
    ... --vy 0.3 | --wz 1.3 | --view | --solver quadprog | --noarmature | --profile step|ramp | --log
    ... --stand (FSM 을 Standing 에 고정: requested_mode standing) | --markers (U1) | --mu 0.8 (마찰 계수 배율)
--seconds = walking 시간 (시뮬 총 시간 = settle 1.0 s + seconds). 넘어짐 = torso z < 0.35 또는 |roll|,|pitch| > 0.8 rad.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import numpy as np

import config
import paths
from command import CommandSchedule
from controller import MitController
from mit_model import LEFT, RIGHT, MitModel, quat_to_roll_pitch

FALL_Z = 0.35
FALL_ANGLE = 0.8
X_LABELS = ["roll", "pitch", "yaw", "px", "py", "pz", "wx", "wy", "wz", "vx", "vy", "vz", "g"]
U_LABELS = ["FxL", "FyL", "FzL", "FxR", "FyR", "FzR", "MxL", "MyL", "MzL", "MxR", "MyR", "MzR"]


# ----------------------------------------------------------------------------------------------------------------
def make_schedule(profile: str, vx: float, vy: float, wz: float, walk_t: float) -> CommandSchedule:
    if profile == "const":
        return CommandSchedule.constant(vx, vy, wz)
    if profile == "step":
        return CommandSchedule.step_at(walk_t, vx, vy, wz)
    if profile == "ramp":
        prof = {}
        for k, v in (("x_dot", vx), ("y_dot", vy), ("psi_dot", wz)):
            prof[k] = {"final": v, "rate": abs(v) / 2.0 if v != 0 else 0.0, "start": walk_t}
        return CommandSchedule(prof)
    raise ValueError(f"unknown profile {profile!r}")


def parse_variant(s: str | None) -> dict:
    """'k=v,k=v' → kwargs. 키: solver, arm (0/1) / noarm, profile, markers (0/1), mu (배율), fsm2 (0/1),
    rhoint / maxit / eps (OSQP 설정 변형: adaptive_rho_interval, max_iter, eps_abs=eps_rel)."""
    kw = {}
    if not s or s == "base":
        return kw
    for item in s.split(","):
        item = item.strip()
        if not item or item == "base":
            continue
        k, _, v = item.partition("=")
        if k == "solver":
            kw["solver"] = v
        elif k == "arm":
            kw["armature"] = bool(int(v))
        elif k == "noarm":
            kw["armature"] = False
        elif k == "profile":
            kw["profile"] = v
        elif k == "markers":
            kw["markers"] = bool(int(v)) if v else True
        elif k == "mu":
            kw["friction_scale"] = float(v)
        elif k in ("rhoint", "maxit", "eps"):
            key = {"rhoint": "adaptive_rho_interval", "maxit": "max_iter", "eps": "eps_abs"}[k]
            kw.setdefault("osqp_settings", {})[key] = float(v) if k == "eps" else int(v)
            if k == "eps":
                kw["osqp_settings"]["eps_rel"] = float(v)
        elif k == "fsm2":
            kw["fsm_double_update"] = bool(int(v))
        elif k == "fix":                           # reference 특이점 고침 (controller.py fixes): fix=slide+yawref
            kw["fixes"] = tuple(x for x in v.split("+") if x)
        else:
            raise KeyError(f"unknown variant key {k!r}")
    return kw


# ----------------------------------------------------------------------------------------------------------------
class Recorder:
    """틱마다 한 행. 미리 할당한 배열."""

    def __init__(self, n: int):
        self.n = 0
        z = lambda *s: np.full((n,) + s, np.nan)   # noqa: E731
        self.t = z()
        self.base = z(3)
        self.rpy = z(3)
        self.x0 = z(13)
        self.xref0 = z(13)
        self.u = z(12)
        self.tau = z(10)
        self.tau_arm = z(8)
        self.est = z(2)
        self.sched = z(2)
        self.active = z(2)
        self.alpha = z(2)
        self.early = z(2)
        self.fn = z(2)
        self.foot = z(2, 3)
        self.desired = z(2, 3)
        self.pdes = z(2, 3)
        self.cmd = z(3)
        self.raw = z(3)
        self.walking = z()
        self.solved = np.zeros(n, dtype=bool)
        self.solve_ms = z()
        self.total_ms = z()
        self.cold = np.zeros(n, dtype=bool)
        self.fallback = np.zeros(n, dtype=bool)
        self.iters = z()
        self.stance0 = z(2)

    def add(self, ctl: MitController, tau_leg, tau_arm):
        i = self.n
        s = ctl.state
        self.t[i] = s.t
        self.base[i] = s.torso_pos_W
        self.rpy[i] = (s.roll, s.pitch, s.yaw_unwrapped)
        self.x0[i] = ctl.x0
        self.u[i] = ctl.u_hold
        self.tau[i] = np.asarray(tau_leg).reshape(-1)
        self.tau_arm[i] = np.asarray(tau_arm).reshape(-1)
        cm = ctl.cm
        self.est[i] = cm.estimated
        self.sched[i] = cm.scheduled if ctl.mode == "walking" else (1, 1)
        self.active[i] = [ctl._active(l, s.t) for l in (LEFT, RIGHT)]
        self.alpha[i] = [ctl._alpha(l) for l in (LEFT, RIGHT)]
        self.early[i] = cm.early
        self.fn[i] = s.foot_normal_force
        self.foot[i] = s.foot_pos_W
        self.desired[i] = ctl.desired
        self.pdes[i] = ctl.swing_pdes()
        f = ctl.filtered
        self.cmd[i] = (f.x_dot, f.y_dot, f.psi_dot)
        r = ctl.raw
        self.raw[i] = (r.x_dot, r.y_dot, r.psi_dot)
        self.walking[i] = 1.0 if ctl.mode == "walking" else 0.0
        info = ctl.tick_info.get("info")
        if info is not None:
            self.solved[i] = True
            self.solve_ms[i] = info.get("solve_ms", np.nan)
            self.total_ms[i] = info.get("total_ms", np.nan)
            self.cold[i] = bool(info.get("cold", False))
            self.fallback[i] = bool(info.get("fallback", False))
            self.iters[i] = info.get("iters", np.nan)
            self.stance0[i] = info.get("stance0", np.full(2, np.nan))
        if ctl.last_info is not None:
            self.xref0[i] = ctl.last_info.get("x_ref0", np.full(13, np.nan))
        self.n += 1

    def trim(self):
        n = self.n
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                out[k] = v[:n]
        return out


# ----------------------------------------------------------------------------------------------------------------
def _edges(b: np.ndarray):
    """(rising idx, falling idx) of a bool array."""
    b = b.astype(np.int8)
    d = np.diff(b)
    return np.nonzero(d == 1)[0] + 1, np.nonzero(d == -1)[0] + 1


def compute_metrics(R: dict, cmd=(0.0, 0.0, 0.0), walk_t=None, lim=None, window=100.0, dt=0.002) -> dict:
    t = R["t"]
    n = len(t)
    out = {}
    walking = R["walking"] > 0.5
    if walk_t is None:
        walk_t = float(t[np.argmax(walking)]) if walking.any() else float(t[0])
    t_end = float(t[-1]) if n else 0.0
    out["sim_t"] = t_end
    out["walk_t"] = walk_t
    out["survived"] = max(0.0, t_end - walk_t)
    w0 = max(t_end - window, walk_t + 5.0)
    win = (t >= w0) & walking
    if win.sum() < 50:
        win = (t >= walk_t + 1.0) & walking
    if win.sum() < 10:
        win = np.ones(n, dtype=bool)
    out["win_s"] = float(win.sum() * dt)
    yaw = R["rpy"][:, 2]
    v = R["x0"][:, 9:12]
    c, s = np.cos(yaw), np.sin(yaw)
    vh = np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], 1)
    out["vx"] = float(np.mean(vh[win, 0]))
    out["vy"] = float(np.mean(vh[win, 1]))
    iw = np.nonzero(win)[0]
    out["wz"] = float((yaw[iw[-1]] - yaw[iw[0]]) / max(t[iw[-1]] - t[iw[0]], 1e-9))
    trk = []
    for name, c_, a_ in (("x", cmd[0], out["vx"]), ("y", cmd[1], out["vy"]), ("wz", cmd[2], out["wz"])):
        if abs(c_) > 1e-9:
            out[f"track_{name}"] = 100.0 * a_ / c_
            trk.append(out[f"track_{name}"])
    out["track"] = float(np.mean(trk)) if trk else np.nan
    out["roll_sd"] = float(np.std(R["rpy"][win, 0]))
    out["pitch_sd"] = float(np.std(R["rpy"][win, 1]))
    out["pitch_mean"] = float(np.mean(R["rpy"][win, 1]))
    # yaw 진동: 한 cycle (0.5 s) 이동평균을 뺀 나머지의 95% 폭 (명령 yaw 램프/드리프트 제거)
    k = int(round(0.5 / dt))
    if n > k:
        ker = np.ones(k) / k
        ma = np.convolve(yaw, ker, mode="same")
        osc = yaw - ma
        valid = win.copy()
        valid[:k] = False
        valid[-k:] = False
        if valid.sum() > 10:
            out["yaw_osc95"] = float(np.percentile(osc[valid], 97.5) - np.percentile(osc[valid], 2.5))
        else:
            out["yaw_osc95"] = np.nan
    # 스윙/착지 (walking 구간)
    apex, td_err, early_sw, slips, n_sw = [], [], 0, [], 0
    for leg in (LEFT, RIGHT):
        act = (R["active"][:, leg] > 0.5) | ~walking
        sched = (R["sched"][:, leg] > 0.5) | ~walking
        est = R["est"][:, leg] > 0.5
        early = R["early"][:, leg] > 0.5
        rise_a, fall_a = _edges(act)            # 착지 (active on), 이륙 (active off)
        rise_s, _ = _edges(sched)
        fz = R["foot"][:, leg, 2]
        for f in fall_a:
            if not win[f]:
                continue
            nxt = rise_a[rise_a > f]
            if len(nxt) == 0:
                continue
            e = nxt[0]
            n_sw += 1
            apex.append(float(np.max(fz[f:e + 1])))
            if early[f:e + 1].any():
                early_sw += 1
            # 실제 접촉 (estimated 상승) − 명목 착지 (scheduled 상승)
            rs = rise_s[rise_s > f]
            if len(rs):
                ts_ = t[rs[0]]
                seg = est[f:min(n, rs[0] + int(0.15 / dt))]
                off = np.nonzero(~seg)[0]                      # 먼저 estimated 가 꺼지고 (이륙 확인)
                if len(off):
                    j = np.nonzero(seg[off[0]:])[0]            # 다시 켜지는 첫 틱 = 실제 착지
                    if len(j):
                        td_err.append(float(t[f + off[0] + j[0]] - ts_))
        # stance slip: 착지 + 10 ms 부터 이륙까지 발 xy 이동
        for r in rise_a:
            if not win[r]:
                continue
            nf = fall_a[fall_a > r]
            if len(nf) == 0:
                continue
            a0 = min(r + 5, nf[0] - 1)
            slips.append(float(np.linalg.norm(R["foot"][nf[0] - 1, leg, :2] - R["foot"][a0, leg, :2])))
    out["n_swings"] = n_sw
    out["apex_mm"] = 1e3 * float(np.mean(apex)) if apex else np.nan
    out["td_err_ms"] = 1e3 * float(np.mean(td_err)) if td_err else np.nan
    out["td_err_sd_ms"] = 1e3 * float(np.std(td_err)) if td_err else np.nan
    out["early_rate"] = early_sw / n_sw if n_sw else np.nan
    out["slip_mm"] = 1e3 * float(np.mean(slips)) if slips else np.nan
    out["slip_max_mm"] = 1e3 * float(np.max(slips)) if slips else np.nan
    if lim is not None:
        ratio = np.abs(R["tau"][win]) / lim[None, :]
        out["tau_ratio_max"] = float(ratio.max()) if ratio.size else np.nan
        out["tau_sat_frac"] = float((ratio >= 1.0).any(1).mean()) if ratio.size else np.nan
    solved = R["solved"]
    out["solves"] = int(solved.sum())
    out["cold_rate"] = float(R["cold"][solved].mean()) if solved.any() else np.nan
    out["fallbacks"] = int(R["fallback"].sum())
    out["solve_ms_med"] = float(np.nanmedian(R["total_ms"][solved])) if solved.any() else np.nan
    out["solve_ms_p99"] = float(np.nanpercentile(R["total_ms"][solved], 99)) if solved.any() else np.nan
    out["iters_med"] = float(np.nanmedian(R["iters"][solved])) if solved.any() else np.nan
    return out


# ----------------------------------------------------------------------------------------------------------------
def save_log(R: dict, tag: str, note: str = "") -> Path:
    paths.LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = paths.LOG_DIR / f"mit_{tag}_{stamp}"
    flat = {}
    for k, v in R.items():
        flat[k] = v
    np.savez_compressed(str(base) + ".npz", **flat, x_labels=np.array(X_LABELS), u_labels=np.array(U_LABELS),
                        note=np.array(note))
    # csv (10 틱마다 = 50 Hz)
    idx = np.arange(0, len(R["t"]), 10)
    cols = [("t", R["t"][idx, None])]
    cols += [(f"x_{l}", R["x0"][idx, i:i + 1]) for i, l in enumerate(X_LABELS)]
    cols += [(f"ref_{l}", R["xref0"][idx, i:i + 1]) for i, l in enumerate(X_LABELS)]
    cols += [(f"u_{l}", R["u"][idx, i:i + 1]) for i, l in enumerate(U_LABELS)]
    cols += [(f"tau{j}", R["tau"][idx, j:j + 1]) for j in range(10)]
    for nm in ("est", "active", "alpha"):
        cols += [(f"{nm}_L", R[nm][idx, 0:1]), (f"{nm}_R", R[nm][idx, 1:2])]
    for leg, sd in ((0, "L"), (1, "R")):
        cols += [(f"foot{sd}_{a}", R["foot"][idx, leg, i:i + 1]) for i, a in enumerate("xyz")]
        cols += [(f"des{sd}_{a}", R["desired"][idx, leg, i:i + 1]) for i, a in enumerate("xyz")]
    cols += [("solve_ms", R["total_ms"][idx, None])]
    header = ",".join(c[0] for c in cols)
    data = np.hstack([c[1] for c in cols])
    np.savetxt(str(base) + ".csv", data, delimiter=",", header=header, comments="", fmt="%.6g", encoding="utf-8-sig")
    return Path(str(base) + ".npz")


def _tag(vx, vy, wz, variant):
    f = lambda v: f"{v:.2f}".replace("-", "m").replace(".", "p")   # noqa: E731
    parts = [variant or "base"]
    if vx:
        parts.append(f"vx{f(vx)}")
    if vy:
        parts.append(f"vy{f(vy)}")
    if wz:
        parts.append(f"wz{f(wz)}")
    if not (vx or vy or wz):
        parts.append("inplace")
    return "walk_" + "_".join(parts)


def _setup(vx, vy, wz, solver, armature, profile, markers, stand, friction_scale, fsm_double_update, verbose,
           osqp_settings=None, fixes=()):
    cfg = config.load()
    model = MitModel(armature=armature, emulate_debug_markers=markers)
    model.reset(str(cfg.port.spawn_keyframe))
    walk_t = float(cfg.startup.post_init_standing_settle_time)
    sched = make_schedule(profile, vx, vy, wz, walk_t)
    ctl = MitController(model, cfg, sched, solver=solver, requested_mode="standing" if stand else None,
                        emulate_markers=markers, friction_scale=friction_scale,
                        fsm_double_update=fsm_double_update, osqp_settings=osqp_settings, verbose=verbose,
                        fixes=fixes)
    return cfg, model, ctl, walk_t


def _fell(d) -> bool:
    if not np.all(np.isfinite(d.qpos)):
        return True
    r, p = quat_to_roll_pitch(d.qpos[3:7])
    return d.qpos[2] < FALL_Z or abs(r) > FALL_ANGLE or abs(p) > FALL_ANGLE


def headless(vx=0.0, vy=0.0, wz=0.0, seconds=120.0, *, solver="osqp", armature=True, profile="const",
             markers=False, stand=False, friction_scale=1.0, fsm_double_update=True, osqp_settings=None, log=False,
             variant=None, verbose=True, print_every=10.0, keep_record=False, fixes=()):
    """walking seconds 동안 (시뮬 = settle + seconds). 반환 dict (metrics + fell 등)."""
    cfg, model, ctl, walk_t = _setup(vx, vy, wz, solver, armature, profile, markers, stand, friction_scale,
                                     fsm_double_update, verbose=False, osqp_settings=osqp_settings, fixes=fixes)
    m, d = model.m, model.d
    total = (0.0 if stand else walk_t) + seconds
    n = int(round(total / m.opt.timestep))
    rec = Recorder(n + 1)
    fell = False
    err = None
    tw = time.perf_counter()
    next_print = print_every
    for k in range(n):
        try:
            tau_leg, tau_arm = ctl.tick()
        except Exception as exc:                  # C++ 에선 runController 밖으로 전파 → 시뮬 중단
            err = f"{type(exc).__name__}: {exc}"
            fell = True
            break
        model.write_torque(tau_leg, tau_arm)
        rec.add(ctl, tau_leg, tau_arm)
        mujoco.mj_step(m, d)
        if _fell(d):
            fell = True
            break
        if verbose and d.time >= next_print:
            next_print += print_every
            s = ctl.state
            f = ctl.filtered
            print(f"  t {d.time:6.1f}  z {d.qpos[2]:.3f}  xy ({d.qpos[0]:+7.2f},{d.qpos[1]:+6.2f})  yaw {s.yaw_unwrapped:+7.2f}"
                  f"  r/p {s.roll:+.3f}/{s.pitch:+.3f}  cmd {f.x_dot:.2f},{f.y_dot:.2f},{f.psi_dot:.2f}  "
                  f"{ctl.fsm.state.value}  wall {time.perf_counter() - tw:5.0f}s", flush=True)
    wall = time.perf_counter() - tw
    R = rec.trim()
    lim = model.ctrl_hi[model.leg_act.ravel()]
    res = compute_metrics(R, (vx, vy, wz), None if stand else walk_t, lim, dt=m.opt.timestep)
    res["survived"] = max(0.0, float(d.time) - (0.0 if stand else walk_t))   # 마지막 mj_step 뒤 시각
    res.update(fell=fell, error=err, wall=wall, vx_cmd=vx, vy_cmd=vy, wz_cmd=wz, seconds=seconds,
               variant=variant or "base", solver=solver, armature=armature, profile=profile, markers=markers,
               transitions=ctl.transitions, t_end=float(d.time))
    res["pass"] = (not fell) and res["survived"] >= seconds - 1e-6
    if log:
        tag = _tag(vx, vy, wz, variant) + ("_stand" if stand else "")
        pth = save_log(R, tag, note=str({k: v for k, v in res.items() if not isinstance(v, (list, dict))}))
        res["log"] = str(pth)
    if keep_record:
        res["record"] = R
    if verbose:
        print_summary(res)
    return res


def print_summary(r: dict):
    print(f"[mit walk] cmd vx {r['vx_cmd']:+.2f} vy {r['vy_cmd']:+.2f} wz {r['wz_cmd']:+.2f}  variant {r['variant']}  "
          f"solver {r['solver']} armature {r['armature']} profile {r['profile']}  wall {r['wall']:.0f}s")
    status = "FELL" if r["fell"] else "ok"
    print(f"  survived {r['survived']:.2f} s of {r['seconds']:.0f} s walking ({status})"
          + (f"  error: {r['error']}" if r.get("error") else ""))
    print(f"  achieved (heading frame, last {r['win_s']:.0f} s): vx {r['vx']:+.3f}  vy {r['vy']:+.3f}  wz {r['wz']:+.3f}"
          f"   tracking {r.get('track', float('nan')):.1f} %")
    print(f"  roll sd {r['roll_sd']:.4f}  pitch sd {r['pitch_sd']:.4f} (mean {r['pitch_mean']:+.4f})  "
          f"yaw osc95 {r.get('yaw_osc95', float('nan')):.4f} rad")
    print(f"  swings {r['n_swings']}  apex {r['apex_mm']:.1f} mm  td err {r['td_err_ms']:+.1f}±{r['td_err_sd_ms']:.1f} ms  "
          f"early {r['early_rate']:.2f}  slip/step {r['slip_mm']:.1f} mm (max {r['slip_max_mm']:.1f})")
    print(f"  tau/limit max {r.get('tau_ratio_max', float('nan')):.2f}  sat frac {r.get('tau_sat_frac', float('nan')):.3f}")
    print(f"  MPC solves {r['solves']}  cold rate {r['cold_rate']:.3f}  fallbacks {r['fallbacks']}  "
          f"median {r['solve_ms_med']:.2f} ms (p99 {r['solve_ms_p99']:.1f})  iters {r['iters_med']:.0f}")
    if r.get("log"):
        print(f"  log: {r['log']}")
    verdict = "PASS" if r["pass"] else "FAIL"
    print(f"{verdict}  ({r['survived']:.1f} s)")


# ----------------------------------------------------------------------------------------------------------------
def view(vx=0.0, vy=0.0, wz=0.0, seconds=120.0, *, solver="osqp", armature=True, profile="const", markers=False,
         stand=False, friction_scale=1.0, fsm_double_update=True, follow=True, fixes=()):
    import mujoco.viewer
    cfg, model, ctl, walk_t = _setup(vx, vy, wz, solver, armature, profile, markers, stand, friction_scale,
                                     fsm_double_update, verbose=True, fixes=fixes)
    m, d = model.m, model.d
    total = (0.0 if stand else walk_t) + seconds
    sync_every = 17
    with mujoco.viewer.launch_passive(m, d) as v:
        if follow:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = model.base_body
            v.cam.distance = 2.5
            v.cam.elevation = -15
            v.cam.azimuth = 135
        t_wall0 = time.perf_counter()
        k = 0
        while v.is_running() and d.time < total:
            tau_leg, tau_arm = ctl.tick()
            model.write_torque(tau_leg, tau_arm)
            mujoco.mj_step(m, d)
            k += 1
            if _fell(d):
                print(f"fell at t = {d.time:.2f}")
                break
            if k % sync_every == 0:
                v.sync()
                lag = d.time - (time.perf_counter() - t_wall0)
                if lag > 0:
                    time.sleep(lag)
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--solver", default="osqp", choices=["osqp", "quadprog"])
    ap.add_argument("--noarmature", action="store_true")
    ap.add_argument("--profile", default="const", choices=["const", "step", "ramp"])
    ap.add_argument("--markers", action="store_true", help="U1 debug-marker mass emulation")
    ap.add_argument("--stand", action="store_true", help="FSM kept in Standing (requested mode standing)")
    ap.add_argument("--mu", type=float, default=1.0, help="friction coefficient scale (variant)")
    ap.add_argument("--fsm1", action="store_true", help="single FSM update per tick (variant)")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--fix", default="", help="reference 특이점 고침 (변형): slide, yawref, yawanchor 를 + 로 연결")
    a = ap.parse_args()
    kw = dict(solver=a.solver, armature=not a.noarmature, profile=a.profile, markers=a.markers, stand=a.stand,
              friction_scale=a.mu, fsm_double_update=not a.fsm1, fixes=tuple(x for x in a.fix.split("+") if x))
    if a.view:
        view(a.vx, a.vy, a.wz, a.seconds, **kw)
        return
    r = headless(a.vx, a.vy, a.wz, a.seconds, log=a.log, variant=a.variant, **kw)
    sys.exit(0 if r["pass"] else 1)


if __name__ == "__main__":
    main()
