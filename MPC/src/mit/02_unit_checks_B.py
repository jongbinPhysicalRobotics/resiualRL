"""Owner B 단위 점검: gait.py / planner.py / contact_manager.py (07_port_design.md §4 [B]).

    python 02_unit_checks_B.py        → PASS/FAIL 줄 + 숫자, 실패 있으면 exit 1

합성 입력만 쓴다 (다른 owner 모듈 불필요, config.py 가 있으면 그것으로 cfg 로드, 없으면 참조 yaml 을 직접 읽음).
기대값은 spec 03/05 의 식을 이 파일에서 따로 (독립적으로) 계산해 비교한다.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gait import LEFT, RIGHT, GaitScheduler, HorizonClock, OverrideStep, stance_table  # noqa: E402
from planner import (BodyTarget, SwingFootPlanner, SwingFootTrajectory, half_stance_preview_time,  # noqa: E402
                     lift_angle_near, swing_foot_yaw_psi_offset, swing_foot_yaw_target_world,
                     yaw_with_psi_offset)
from contact_manager import ContactManager  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  | {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# cfg
# ---------------------------------------------------------------------------
def _ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
    return d


def load_cfg():
    try:
        import config  # owner A
        return config.load(), "config.load() (MPC/src/mit/mit_controller.yaml)"
    except Exception as e:  # noqa: BLE001
        import yaml
        here = Path(__file__).resolve()
        root = next(p for p in here.parents if p.name == "MPC").parent
        y = root / "reference" / "config" / "mit_humanoid" / "my_controller.yaml"
        return _ns(yaml.safe_load(y.read_text(encoding="utf-8"))), f"reference yaml fallback ({e!r})"


cfg, cfg_src = load_cfg()
print(f"cfg: {cfg_src}")

DT = 0.002
CYCLE, SWING, STANCE = 0.5, 0.17, 0.33
OFF_Y = 0.075999602
COM_OFF = np.array([0.01531, 0.00286, 0.11657])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def accumulated_time(n_steps, t=0.0):
    """MuJoCo data.time 처럼 += 0.002 누적."""
    for _ in range(n_steps):
        t += DT
    return t


def make_state(t, foot_pos, yaw=0.0, torso=(0.0, 0.0, 0.68), v=(0.0, 0.0, 0.0), w=(0.0, 0.0, 0.0),
               force=(300.0, 300.0)):
    torso = np.asarray(torso, float)
    off_W = Rz(yaw) @ COM_OFF
    st = SimpleNamespace(
        t=t, yaw_unwrapped=yaw, torso_pos_W=torso,
        foot_pos_W=np.asarray(foot_pos, float).copy(),
        foot_normal_force=np.asarray(force, float).copy(),
        foot_contact=np.asarray(force, float) > 0.0,
        reduced=SimpleNamespace(com_offset_B=COM_OFF.copy()))
    x0 = np.zeros(13)
    x0[2] = yaw
    x0[3:6] = torso + off_W
    x0[9:12] = np.asarray(v, float) + np.cross(np.asarray(w, float), off_W)
    x0[6:9] = w
    x0[12] = -9.81
    return st, x0


def cmd(x=0.0, y=0.0, psi=0.0):
    return SimpleNamespace(x_dot=x, y_dot=y, psi_dot=psi, body_height_offset_m=0.0,
                           standing_roll_offset_rad=0.0, standing_pitch_offset_rad=0.0)


# ===========================================================================
print("\n=== 1. phase / contact table (one cycle) ===")
t_start = accumulated_time(1500)
check("accumulated data.time after 1500 steps == 2.999999999999891 (spec 03 §1.1)",
      t_start == 2.999999999999891, repr(t_start))
clock = HorizonClock(t_start, cfg)
gait = GaitScheduler(cfg, clock)
gait.set_mode("walking")

# (a) 틱 중간 시각 (경계 비트 영향 없음) 에서 spec 표와 비교
bad = 0
for i in range(250):
    tm = clock.t0 + (i + 0.5) * DT
    tc = (i + 0.5) * DT
    exp_L = not (0.08 <= tc < 0.25)
    exp_R = not (0.33 <= tc < 0.50)
    if gait.contact(LEFT, tm) != exp_L or gait.contact(RIGHT, tm) != exp_R:
        bad += 1
check("mid-tick contact == spec table (Left swing [0.08,0.25), Right swing [0.33,0.5))", bad == 0,
      f"{bad} mismatches of 250")

# (b) 실제 누적 tick 으로 한 cycle 표를 출력 (구간별)
t = t_start
rows = []
prev = None
for i in range(250):
    clock.sync(t)
    c = (gait.contact(LEFT, t), gait.contact(RIGHT, t))
    if c != prev:
        rows.append([t - clock.t0, c, gait.phase(LEFT, t), gait.phase(RIGHT, t)])
        prev = c
    t += DT
print("  cycle-time start  L R   p_L        p_R      (accumulated ticks from t0 = %r)" % clock.t0)
for r in rows:
    print(f"  {r[0]:.6f}          {'S' if r[1][0] else '-'} {'S' if r[1][1] else '-'}   {r[2]:.6f}  {r[3]:.6f}")
starts = [round(r[0], 6) for r in rows]
check("accumulated-tick segment starts ≈ [0, 0.08, 0.25, 0.33] (±1 tick)",
      len(starts) == 4 and all(abs(a - b) <= DT + 1e-9 for a, b in zip(starts, [0.0, 0.08, 0.25, 0.33])),
      str(starts))
clock.sync(t)   # 누적 t = 3.499999999999836 → t - t0 < 0.5 → 아직 안 넘어감 (경계 비트 민감)
t0_before = clock.t0
clock.sync(t + DT)
check("sync: t-t0 just below 0.5 keeps t0; next tick advances by exactly one cycle (t0 += 0.5)",
      t0_before == t_start and clock.t0 == t_start + 0.5, f"t={t!r} t0 {t0_before!r} -> {clock.t0!r}")
clock.reset(3.0)
check("remaining_swing_time(L, t0+0.10) = 0.5*(1-0.7) = 0.15",
      abs(gait.remaining_swing_time(LEFT, 3.10) - 0.15) < 1e-12, f"{gait.remaining_swing_time(LEFT, 3.10)!r}")
check("remaining_swing_time(R, t0) clamps to swing 0.17",
      gait.remaining_swing_time(RIGHT, 3.0) == 0.17, f"{gait.remaining_swing_time(RIGHT, 3.0)!r}")
check("remaining_swing_time(R, t0+0.40) = 0.5*(1-0.8) = 0.1",
      abs(gait.remaining_swing_time(RIGHT, 3.40) - 0.1) < 1e-12, f"{gait.remaining_swing_time(RIGHT, 3.40)!r}")
check("both_stance: DS at 0.05 and 0.30, not at 0.10 / 0.40",
      gait.both_stance(3.05) and gait.both_stance(3.30) and not gait.both_stance(3.10)
      and not gait.both_stance(3.40))
gs = GaitScheduler(cfg, clock)
gs.set_mode("standing")
check("standing: p = 0, c = True",
      gs.phase(LEFT, 3.1) == 0.0 and gs.contact(LEFT, 3.1) and gs.contact(RIGHT, 3.4))

# ===========================================================================
print("\n=== 2. horizon steps from the cycle origin ===")


def expected_steps(t0):
    """spec 02 §5.2: 식 그대로 따로 계산 (knife edge 포함)."""
    dt_mpc = 0.5 / float(25)
    L, R = [], []
    for k in range(25):
        tk = t0 + float(k) * dt_mpc
        pL = math.fmod((tk - t0) / 0.5 + 0.5, 1.0)
        pR = math.fmod((tk - t0) / 0.5 + 0.0, 1.0)
        L.append(0.0 <= pL < 0.33 / 0.5)
        R.append(0.0 <= pR < 0.33 / 0.5)
    return np.array([L, R]).T


nominal_L = np.array([k in range(0, 4) or k >= 13 for k in range(25)])
nominal_R = np.array([k <= 16 for k in range(25)])
clock = HorizonClock(3.0, cfg)
gait = GaitScheduler(cfg, clock)
gait.set_mode("walking")
ok_all = True
for tq in (3.0, 3.137, 3.30, 3.49, 3.6, 4.2):
    clock.sync(tq)
    st = stance_table(gait.horizon_steps(clock))
    same = np.array_equal(st, expected_steps(clock.t0))
    tk_ok = all(clock.tk(k) == clock.t0 + k * (0.5 / 25) for k in range(25))
    ok_all &= same and tk_ok
    print(f"  t={tq:.3f} t0={clock.t0!r:<20} L:{''.join('S' if s else '-' for s in st[:, 0])}"
          f"  R:{''.join('S' if s else '-' for s in st[:, 1])}")
check("horizon table (live) == independent expression, tk = t0 + k*0.02, table independent of t in the cycle",
      ok_all)
clock.reset(3.0)
st = stance_table(gait.horizon_steps(clock))
check("t0=3.0: L stance k∈{0..3,13..24}, R stance k∈{0..16} (k=4 Left swing)",
      np.array_equal(st[:, 0], nominal_L) and np.array_equal(st[:, 1], nominal_R))
pL4 = math.fmod(((3.0 + 4 * 0.02) - 3.0) / 0.5 + 0.5, 1.0)
check("knife edge t0=3.0: p_L(k=4)=0.6600000000000001 → swing", pL4 == 0.6600000000000001 and not st[4, 0],
      repr(pL4))
for t0v, exp in ((2.999999999999891, False), (3.999999999999891, True)):
    clock.reset(t0v)
    st = stance_table(gait.horizon_steps(clock))
    p = gait.phase(LEFT, clock.tk(4))
    check(f"knife edge t0={t0v!r}: Left k=4 {'stance' if exp else 'swing'}", bool(st[4, 0]) == exp, f"p_L={p!r}")
clock.reset(3.0)
ov = [OverrideStep(enabled=True, stance=np.array([False, True]), min_scale=np.array([0.3, 1.7]))]
steps = gait.horizon_steps(clock, ov)
check("override replaces step 0 only (stance F/T, min_scale clamped [0.3, 1.0]); step 1 nominal scale 1",
      (not steps[0].stance[0]) and steps[0].stance[1] and np.allclose(steps[0].min_scale, [0.3, 1.0])
      and steps[1].stance.all() and np.all(steps[1].min_scale == 1.0) and len(steps) == 25,
      f"step0={steps[0]}")
ov_dis = [OverrideStep(enabled=False, stance=np.array([False, False]), min_scale=np.zeros(2))]
check("disabled override ignored", stance_table(gait.horizon_steps(clock, ov_dis))[0].all())
gs.set_clock(clock)
check("standing horizon: all stance, scale 1",
      stance_table(gs.horizon_steps(clock)).all() and all(np.all(s.min_scale == 1) for s in gs.horizon_steps(clock)))

# ===========================================================================
print("\n=== 3. swing trajectory ===")
p0 = np.array([0.02, 0.08, 0.001])
p1 = np.array([0.15, 0.09, -0.005])
H, T = 0.06, 0.17
d = p1 - p0
zmid = p1[2] + H


def traj_at(tau, pf=p1):
    tr = SwingFootTrajectory()
    tr.reset(p0, pf, H, T)
    if tau > 0:
        tr.advance(tau)
    return tr


tr = traj_at(0.0)
a_exp0 = np.array([6 * d[0] / T**2, 6 * d[1] / T**2, 6 * (zmid - p0[2]) * (2 / T) ** 2])
check("reset tick: p = p_init, v = 0", np.array_equal(tr.position(), p0) and np.all(tr.velocity() == 0.0),
      f"p={tr.position()}, v={tr.velocity()}")
check("reset tick: a = [6dx/T², 6dy/T², 6dz(2/T)²] (not zero)", np.allclose(tr.acceleration(), a_exp0, rtol=1e-12),
      f"a={tr.acceleration()}")
tr = traj_at(T / 2)
check("apex (s=0.5): z = pFinal.z + h = 0.055, v.z = 0",
      abs(tr.position()[2] - 0.055) < 1e-15 and tr.velocity()[2] == 0.0,
      f"z={tr.position()[2]!r} vz={tr.velocity()[2]!r}")
check("apex xy = midpoint (smoothstep(0.5)=0.5)", np.allclose(tr.position()[:2], (p0[:2] + p1[:2]) / 2, atol=1e-15))
tr = traj_at(T)
check("touchdown (advance T): p = p_final, v = 0, inactive",
      np.allclose(tr.position(), p1, atol=1e-15) and np.all(np.abs(tr.velocity()) < 1e-15) and not tr.active,
      f"p={tr.position()} v={tr.velocity()} active={tr.active}")
# tick 으로 진행
tr = traj_at(0.0)
n = 0
zmax = -1.0
while tr.active and n < 200:
    tr.advance(DT)
    n += 1
    zmax = max(zmax, tr.position()[2])
check("ticked with dt=0.002: ends inactive at p_final with v=0 (85 or 86 ticks)",
      n in (85, 86) and np.allclose(tr.position(), p1, atol=1e-12) and np.all(np.abs(tr.velocity()) < 1e-9),
      f"ticks={n} remaining={tr.remaining_time()!r} max z={zmax:.6f}")
# finite-difference (s=0.5 근처 제외: z 가속도는 pInit.z≠pFinal.z 이면 불연속)
h = 1e-6
ev, ea, vmax, amax = 0.0, 0.0, 0.0, 0.0
for s in np.linspace(0.02, 0.98, 97):
    if abs(s - 0.5) < 0.02:
        continue
    tau = s * T
    pm, pp, pc = traj_at(tau - h), traj_at(tau + h), traj_at(tau)
    vfd = (pp.position() - pm.position()) / (2 * h)
    afd = (pp.velocity() - pm.velocity()) / (2 * h)
    ev = max(ev, np.max(np.abs(vfd - pc.velocity())))
    ea = max(ea, np.max(np.abs(afd - pc.acceleration())))
    vmax = max(vmax, np.max(np.abs(pc.velocity())))
    amax = max(amax, np.max(np.abs(pc.acceleration())))
check("finite-difference d/dt position == velocity", ev / vmax < 1e-6, f"max|err|={ev:.3e} (max|v|={vmax:.3f})")
check("finite-difference d/dt velocity == acceleration", ea / amax < 1e-6, f"max|err|={ea:.3e} (max|a|={amax:.2f})")
# set_final_position
p1b = p1 + np.array([0.05, -0.02, 0.01])
tr = traj_at(T / 2)
tr.set_final_position(p1b)
check("set_final_position re-targets (apex follows new pFinal.z + h, xy blend from pInit)",
      abs(tr.position()[2] - (p1b[2] + H)) < 1e-15 and np.allclose(tr.position()[:2], (p0[:2] + p1b[:2]) / 2),
      f"p={tr.position()}")
tr.deactivate()
check("deactivate: remaining 0, inactive, outputs at pFinal, v=0",
      tr.remaining_time() == 0.0 and not tr.active and np.allclose(tr.position(), p1b) and
      np.all(np.abs(tr.velocity()) < 1e-15))
p_before = tr.position().copy()
tr.advance(DT)
tr.set_final_position(p1)
check("inactive: advance no-op, set_final_position does not update outputs",
      np.array_equal(tr.position(), p_before))
try:
    SwingFootTrajectory().reset(p0, p1, H, 0.0)
    check("reset with T<=0 raises", False)
except ValueError:
    check("reset with T<=0 raises", True)
tr = SwingFootTrajectory()
tr.reset(p0, p1, H, 0.001)
print(f"  (info) 1 ms re-reset spike a = {tr.acceleration()}")

# ===========================================================================
print("\n=== 4. planner: preview time, yaw helpers, touchdown target ===")
for sp, exp in ((0.6, 0.87 * 0.33), (0.62, 0.78 * 0.33), (0.65, 0.78 * 0.33), (0.7, 0.76 * 0.33), (0.0, 0.87 * 0.33)):
    got = half_stance_preview_time(sp, cfg)
    check(f"preview time speed={sp}: {exp:.4f}", abs(got - exp) < 1e-12, repr(got))
yt = swing_foot_yaw_target_world(0.1, 1.3, 0.0, cfg)
check("swing yaw target = yaw + 1.0*psi_dot*Tp", abs(yt - (0.1 + 1.3 * 0.2871)) < 1e-12, repr(yt))
check("psi offset: L,+1.3 → +20°; R,+1.3 → 0; R,-0.1 → -10°; L,-0.1 → 0",
      abs(swing_foot_yaw_psi_offset(LEFT, 1.3) - math.radians(20)) < 1e-15
      and swing_foot_yaw_psi_offset(RIGHT, 1.3) == 0.0
      and abs(swing_foot_yaw_psi_offset(RIGHT, -0.1) + math.radians(10)) < 1e-15
      and swing_foot_yaw_psi_offset(LEFT, -0.1) == 0.0)
check("yaw_with_psi_offset(1.3, L, 3.1) = 3.1 + 20° (lift near base, no wrap)",
      abs(yaw_with_psi_offset(1.3, LEFT, 3.1) - (3.1 + math.radians(20))) < 1e-12)
check("lift_angle_near(-3.1, 3.1) = 3.1 + (2π - 6.2)",
      abs(lift_angle_near(-3.1, 3.1) - (3.1 + 2 * math.pi - 6.2)) < 1e-12)

# --- 4a. x_dot 0.6, psi_dot 0 (손 계산 경우) ---
clock = HorizonClock(3.0, cfg)
gait = GaitScheduler(cfg, clock)
gait.set_mode("walking")
pl = SwingFootPlanner(cfg)
yaw_b, yaw_m = 0.3, 0.25
feet = np.array([[1.02, 2.08, 0.0005], [1.03, 1.92, -0.0002]])
t = 3.0 + 0.334                                # Right swing 첫 tick (p_R = 0.668), Left stance
st, x0 = make_state(t, feet, yaw=yaw_m, torso=(1.0, 2.0, 0.68), v=(0.6, 0.0, 0.0))
pl.set_body_yaw_target(yaw_b)
out = pl.desired_foot_positions(st, x0, clock, gait, cmd(0.6), t)
Tp = (0.5 + 0.37) * 0.33
com = x0[3:6]
exp_R = com + Rz(yaw_b) @ np.array([0.6 * Tp, -OFF_Y, 0.0])
exp_R[2] = -0.005
print(f"  Tp={Tp!r}  COM={com}  target_R={out[RIGHT]}  expected={exp_R}")
check("x_dot 0.6: Right target = COM + Rz(yaw_body)·[0.6·Tp, -0.076, 0], z=-0.005",
      np.allclose(out[RIGHT], exp_R, atol=1e-12), f"err={np.max(np.abs(out[RIGHT] - exp_R)):.2e}")
check("after reset: stance Left seeded with measured foot pos", np.array_equal(out[LEFT], feet[LEFT]))
# 다음 tick: COM 이동해도 고정
t2 = t + DT
st2, x02 = make_state(t2, feet, yaw=yaw_m, torso=(1.05, 2.01, 0.68), v=(0.6, 0.0, 0.0))
out2 = pl.desired_foot_positions(st2, x02, clock, gait, cmd(0.7, 0.1, 0.5), t2)
check("target frozen during swing (COM/cmd change, no recompute)",
      np.array_equal(out2[RIGHT], out[RIGHT]) and np.array_equal(out2[LEFT], feet[LEFT]) and not any(pl.last_updated))
# 다음 cycle: Right stance → 같은 캐시, Left swing 시작 (cycle time 0.082) → 새 COM 으로 계산
t3 = 3.5 + 0.082
st3, x03 = make_state(t3, feet + 0.3, yaw=yaw_m, torso=(1.3, 2.0, 0.68))
out3 = pl.desired_foot_positions(st3, x03, clock, gait, cmd(0.6), t3)
exp_L = x03[3:6] + Rz(yaw_b) @ np.array([0.6 * Tp, OFF_Y, 0.0])
exp_L[2] = -0.005
check("next cycle: Right (stance) returns cached target, not measured foot", np.array_equal(out3[RIGHT], out[RIGHT]))
check("Left swing start: target = new COM + Rz(yaw_body)·[0.6·Tp, +0.076, 0]",
      np.allclose(out3[LEFT], exp_L, atol=1e-12) and clock.t0 == 3.5, f"t0={clock.t0!r} L={out3[LEFT]}")

# --- 4b. turning + lateral crossing guard ---
pl = SwingFootPlanner(cfg)
clock = HorizonClock(3.0, cfg)
gait = GaitScheduler(cfg, clock)
gait.set_mode("walking")
t = 3.0 + 0.082                                   # Left swing 첫 tick (p_L = 0.664)
st, x0 = make_state(t, feet, yaw=yaw_m, torso=(1.0, 2.0, 0.68))
pl.set_body_yaw_target(yaw_b)
c = cmd(0.0, -0.5, 1.0)
out = pl.desired_foot_positions(st, x0, clock, gait, c, t)
Tp = 0.87 * 0.33
p_L = math.fmod((t - 3.0) / 0.5 + 0.5, 1.0)
Trem = min(max(0.5 * (1.0 - p_L), 0.0), 0.17)
yaw_tr, yaw_td = yaw_b + 0.5 * 1.0 * Tp, yaw_b + 1.0 * Trem
step_W = Rz(yaw_tr) @ np.array([0.0, -0.5, 0.0]) * Tp
planned = Rz(yaw_b).T @ step_W + Rz(yaw_td - yaw_b) @ np.array([0.0, OFF_Y, 0.0])
raw_y = planned[1]
planned[1] = max(planned[1], 0.0)
exp_L = x0[3:6] + Rz(yaw_b) @ planned
exp_L[2] = -0.005
check("turn + y_dot=-0.5: Left target matches formula (yawTrans=yaw0+½ψ̇Tp, yawTd=yaw0+ψ̇Trem)",
      np.allclose(out[LEFT], exp_L, atol=1e-12), f"Trem={Trem:.6f} err={np.max(np.abs(out[LEFT] - exp_L)):.2e}")
check("lateral crossing guard active (planned_B.y clamped to 0)", raw_y < 0.0, f"raw planned y={raw_y:.4f}")

# ===========================================================================
print("\n=== 5. stop branch + latch + turn-dominant stop ===")
pl = SwingFootPlanner(cfg)
clock = HorizonClock(3.0, cfg)
gait = GaitScheduler(cfg, clock)
gait.set_mode("walking")
pl.set_body_yaw_target(yaw_b)
t = 3.0 + 0.340                                   # Right swing
vcom = (1.0, 0.2, 0.0)
st, x0 = make_state(t, feet, yaw=yaw_m, torso=(1.0, 2.0, 0.68), v=vcom)
o1 = pl.desired_foot_positions(st, x0, clock, gait, cmd(0.3), t)     # moving (swing first tick)
t += DT
st, x0 = make_state(t, feet, yaw=yaw_m, torso=(1.01, 2.0, 0.68), v=vcom)
o2 = pl.desired_foot_positions(st, x0, clock, gait, cmd(0.0), t)     # stop requested → just activated
vB = Rz(yaw_b).T @ x0[9:12]
offB = np.array([0.2 * vB[0], 0.2 * vB[1], 0.0])
nrm = math.hypot(offB[0], offB[1])
if nrm > 0.08:
    offB[:2] *= 0.08 / nrm
center = x0[3:6] + Rz(yaw_b) @ offB
p_R = math.fmod((t - 3.0) / 0.5, 1.0)
Trem = min(max(0.5 * (1 - p_R), 0.0), 0.17)
exp_R = center + Rz(yaw_b) @ (Rz(0.0 * Trem) @ np.array([0.0, -OFF_Y, 0.0]))
exp_R[2] = -0.005
check("stop just activated: swing target recomputed at capture-point center (gain 0.2, clipped to 0.08)",
      pl.stop_active and pl.stop_just_activated and np.allclose(o2[RIGHT], exp_R, atol=1e-12)
      and not np.allclose(o2[RIGHT], o1[RIGHT]),
      f"|0.2 v_B|={0.2 * math.hypot(vB[0], vB[1]):.3f} → clipped; target={o2[RIGHT]}")
t += DT
st, x0 = make_state(t, feet, yaw=yaw_m, torso=(1.03, 2.0, 0.68), v=vcom)
o3 = pl.desired_foot_positions(st, x0, clock, gait, cmd(0.0), t)
check("stop active but not just activated: target frozen", np.array_equal(o3[RIGHT], o2[RIGHT]) and pl.stop_active)
seq = []
for _ in range(7):
    t += DT
    st, x0 = make_state(t, feet, yaw=yaw_m, torso=(1.03, 2.0, 0.68), v=vcom)
    pl.desired_foot_positions(st, x0, clock, gait, cmd(0.3), t)
    seq.append(pl.stop_active)
check("latch: stays active 5 ticks after leaving deadband, then off",
      seq == [True] * 5 + [False, False], str(seq))
dead = []
for c_ in (cmd(0.019), cmd(0.021), cmd(0.0, 0.0, 0.02), cmd(0.0, 0.0, 0.021)):
    p_ = SwingFootPlanner(cfg)
    dead.append(p_._stop_recenter_requested(c_.x_dot, c_.y_dot, c_.psi_dot))
check("deadband 0.02 (<=): [0.019 T, 0.021 F, psi 0.02 T, psi 0.021 F]", dead == [True, False, True, False], str(dead))

# turn-dominant stop
pl = SwingFootPlanner(cfg)
pl.set_body_yaw_target(yaw_b)
t = 3.0 + 0.340
st, x0 = make_state(t, feet, yaw=yaw_m, torso=(1.0, 2.0, 0.68))
pl.desired_foot_positions(st, x0, clock, gait, cmd(0.0, 0.0, 1.0), t)          # turning in place
res = []
for k in range(3):
    t += DT
    tor = (1.0 + 0.01 * k, 2.0, 0.68)
    st, x0 = make_state(t, feet, yaw=yaw_m + 0.01 * k, torso=tor)
    o = pl.desired_foot_positions(st, x0, clock, gait, cmd(0.0), t)
    ce = x0[3:6].copy()
    ce[2] = -0.005
    e = ce + Rz(yaw_m + 0.01 * k) @ np.array([0.0, -OFF_Y, 0.0])
    e[2] = -0.005
    res.append(np.allclose(o[RIGHT], e, atol=1e-12) and pl.turn_stop_frame_valid and pl.last_updated[RIGHT])
check("turn-dominant stop: frame = measured COM/yaw, swing target recomputed every tick", all(res), str(res))
t += DT
st, x0 = make_state(t, feet, yaw=yaw_m, torso=(1.0, 2.0, 0.68))
for _ in range(6):
    pl.desired_foot_positions(st, x0, clock, gait, cmd(0.3), t)
    t += DT
check("turn-stop frame cleared once stop latch releases", not pl.turn_stop_frame_valid)

# ===========================================================================
print("\n=== 6. body target ===")
bt = BodyTarget()
feet_s = np.array([[0.02236, 0.08016, 0.0], [0.02236, -0.08016, 0.0]])
_, x0 = make_state(0.0, feet_s, torso=(0.0, 0.0, 0.679472))
bt.update(x0, 0.0, "standing", feet_s, cmd(0.6, 0.0, 1.3), cfg, com_offset_B=COM_OFF)
check("first call seeds nominal height = 0.679472 + com_offset.z; xy = mean foot; yaw 0 (dt=0)",
      abs(bt.nominal_height_W - (0.679472 + COM_OFF[2])) < 1e-15 and abs(bt.nominal_position_W[0] - 0.02236) < 1e-15
      and bt.nominal_position_W[1] == 0.0 and bt.euler_W[2] == 0.0, f"h={bt.nominal_height_W!r}")
for _ in range(10):
    bt.update(x0, DT, "walking", feet_s, cmd(0.6, 0.0, 1.3), cfg, com_offset_B=COM_OFF)
check("walking: xy = x0 xy, yaw = Σ psi_dot*dt (open loop), height unchanged",
      np.allclose(bt.nominal_position_W[:2], x0[3:5]) and abs(bt.euler_W[2] - 10 * 1.3 * DT) < 1e-15
      and bt.nominal_position_W[2] == bt.nominal_height_W, f"yaw={bt.euler_W[2]!r}")
try:
    BodyTarget().update(x0, 0.0, "standing", feet_s, cmd(), cfg)
    check("first call without com_offset_B raises (needs reduced.com_offset_B)", False)
except ValueError:
    check("first call without com_offset_B raises (needs reduced.com_offset_B)", True)

# ===========================================================================
print("\n=== 7. contact manager ===")
t = accumulated_time(1500)
clock = HorizonClock(t, cfg)
gait = GaitScheduler(cfg, clock)
gait.set_mode("walking")
cm = ContactManager(cfg)
feet_c = np.array([[0.02, 0.08, 0.0], [0.02, -0.08, 0.0]])
nominal = np.array([[0.2, 0.08, -0.005], [0.25, -0.08, -0.005]])
st, _ = make_state(t, feet_c, force=(300.0, 300.0))
cm.reset(st, gait, "walking", t)
cm.update(st, gait, "walking", nominal, t)
check("reset + same-tick update: dt 0, both active, alpha 1",
      cm.last_dt == 0.0 and cm.active_contact.all() and np.all(cm.ramp_alpha == 1.0))

log = []
early_tick = None
touch_ct = 0.40
foot_R = feet_c[RIGHT].copy()
swung = False
for i in range(1, 300):
    t += DT
    clock.sync(t)
    tc = t - clock.t0
    sched_R = gait.contact(RIGHT, t)
    if not swung and sched_R:
        fR = 300.0                                  # 명목 stance 동안 하중
    elif i * DT < touch_ct - 1e-9:
        fR = 0.0                                    # swing 시작부터 하중 0
        swung = True
    else:
        fR = 200.0                                  # 이른 착지 (명목 stance 0.5 보다 앞)
        if early_tick is None:
            early_tick = i
    if i * DT >= touch_ct - 1e-9:
        foot_R = foot_R + np.array([0.001, 0.0, 0.0])     # 착지 뒤 미끄러짐
    feet_now = feet_c.copy()
    feet_now[RIGHT] = foot_R
    fL = 0.0 if 0.08 <= i * DT < 0.25 else 300.0
    st, _ = make_state(t, feet_now, force=(fL, fR))
    cm.update(st, gait, "walking", nominal, t)
    managed = cm.managed_foot_positions(nominal)
    ov = cm.build_horizon_override()
    log.append(dict(i=i, t=t, tc=tc, sched=bool(cm.scheduled[RIGHT]), est=bool(cm.estimated[RIGHT]),
                    early=bool(cm.early[RIGHT]), act=bool(cm.active_contact[RIGHT]), a=float(cm.ramp_alpha[RIGHT]),
                    frozen=cm.frozen_touchdown_W[RIGHT].copy(), managed=managed[RIGHT].copy(),
                    ovs=bool(ov[0].stance[RIGHT]), ovm=float(ov[0].min_scale[RIGHT]), n_ov=len(ov),
                    foot=foot_R.copy(), rel=bool(cm.released_during_swing[RIGHT])))

sw0 = next(r for r in log if not r["sched"])
k0 = log.index(sw0)
est_seq = [r["est"] for r in log[k0:k0 + 5]]
check("hysteresis: force 0 from swing start → estimated stays on 3 ticks, off on 4th",
      est_seq == [True, True, True, False, False], f"swing starts tc={sw0['tc']:.4f}, est={est_seq}")
check("while estimated still on at swing start: not early (not released), active = scheduled = False",
      all((not r["early"]) and (not r["act"]) for r in log[k0:k0 + 4]))
ke = next(k for k, r in enumerate(log) if r["early"])
alph = [r["a"] for r in log[ke:ke + 8]]
print("  early contact from tc=%.4f: alpha = %s" % (log[ke]["tc"], [repr(a) for a in alph]))
exp_a = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0]
check("early contact: active, alpha 0 exactly on first tick, then 0.2,0.4,0.6,0.8,1.0 (±1e-9)",
      log[ke]["act"] and alph[0] == 0.0 and all(abs(a - b) < 1e-9 for a, b in zip(alph, exp_a))
      and alph[6] == 1.0, f"sched={log[ke]['sched']}")
frozen0 = log[ke]["foot"]
check("frozen touchdown = foot pos at first early tick; managed pos stays frozen while foot slides",
      all(np.array_equal(r["frozen"], frozen0) and np.array_equal(r["managed"], frozen0) for r in log[ke:ke + 10]
          if r["early"]) and not np.array_equal(log[ke + 5]["foot"], frozen0))
check("override: len 1, stance R = active, min_scale R = alpha",
      all(r["n_ov"] == 1 and r["ovs"] == r["act"] and (r["ovm"] == (r["a"] if r["act"] else 0.0)) for r in log))
clock_tmp = HorizonClock(clock.t0, cfg)
ov_now = cm.build_horizon_override()
steps_now = gait.horizon_steps(clock_tmp, ov_now)
check("override flows into horizon step 0 via GaitScheduler.horizon_steps",
      bool(steps_now[0].stance[RIGHT]) == bool(cm.active_contact[RIGHT])
      and steps_now[0].min_scale[RIGHT] == (cm.ramp_alpha[RIGHT] if cm.active_contact[RIGHT] else 0.0))
kw = next(k for k, r in enumerate(log) if k > ke and r["sched"])
check("scheduled stance reached: early off, active = scheduled, alpha stays 1 (no re-ramp), managed = nominal",
      (not log[kw]["early"]) and log[kw]["act"] and log[kw]["a"] == 1.0
      and np.array_equal(log[kw]["managed"], nominal[RIGHT]), f"tc={log[kw]['tc']:.4f}")
nonearly = [r for r in log if not r["early"]]
check("non-early ticks: managed = planner nominal", all(np.array_equal(r["managed"], nominal[RIGHT]) for r in nonearly))

# 중간 힘 (0.5..36 N) 은 상태 유지 + bounce
cm2 = ContactManager(cfg)
t = 3.0
clock = HorizonClock(t, cfg)
gait = GaitScheduler(cfg, clock)
gait.set_mode("walking")
st, _ = make_state(t, feet_c, force=(300.0, 300.0))
cm2.reset(st, gait, "walking", t)
seq = []
t = 3.0 + 0.36                                       # Right swing
for f in [0.0] * 5 + [10.0] * 3 + [50.0] + [10.0] * 3 + [0.0] * 5:
    st, _ = make_state(t, feet_c, force=(300.0, f))
    cm2.update(st, gait, "walking", nominal, t)
    seq.append((f, bool(cm2.estimated[RIGHT]), bool(cm2.early[RIGHT]), bool(cm2.active_contact[RIGHT])))
    t += DT
est = [s[1] for s in seq]
check("between thresholds holds; 50 N turns on in 1 tick; off after 4 ticks ≤ 0.5 N",
      est == [True] * 3 + [False] * 5 + [True] * 4 + [True] * 3 + [False] * 2, str(est))
check("early-contact bounce: unload for 4 ticks → early off, active off (back to swing)",
      seq[8][2] and seq[8][3] and (not seq[-1][2]) and (not seq[-1][3]))

# standing short-circuit
cm3 = ContactManager(cfg)
gs = GaitScheduler(cfg, clock)
gs.set_mode("standing")
st, _ = make_state(3.1, feet_c, force=(0.0, 0.0))
cm3.reset(st, gs, "standing", 3.1)
cm3.update(st, gs, "standing", nominal, 3.102)
check("standing: active both, alpha 1, managed = nominal, no early",
      cm3.active_contact.all() and np.all(cm3.ramp_alpha == 1.0) and not cm3.early.any()
      and np.array_equal(cm3.managed_foot_positions(nominal), nominal))

# late contact: MIT 는 꺼짐, 켜면 코드 경로 동작
cm4 = ContactManager(cfg)
check("MIT cfg: late contact handling disabled", cm4.enable_late is False and cm4.enable_early is True)
t = 3.0 + 0.40
clock = HorizonClock(3.0, cfg)
gait = GaitScheduler(cfg, clock)
gait.set_mode("walking")
st, _ = make_state(t, feet_c, force=(300.0, 0.0))
cm4.reset(st, gait, "walking", t)
for _ in range(60):                                 # Right swing → scheduled touchdown at wrap, no force
    t += DT
    clock.sync(t)
    cm4.update(st, gait, "walking", nominal, t)
check("late disabled: scheduled stance with zero force → active, not searching",
      cm4.active_contact[RIGHT] and not cm4.search_mode[RIGHT] and not cm4.late[RIGHT])
cm5 = ContactManager(cfg)
cm5.enable_late = True
t = 3.0 + 0.40
clock = HorizonClock(3.0, cfg)
gait = GaitScheduler(cfg, clock)
gait.set_mode("walking")
cm5.reset(st, gait, "walking", t)
states = []
for _ in range(60):
    t += DT
    clock.sync(t)
    cm5.update(st, gait, "walking", nominal, t)
    states.append((bool(cm5.late[RIGHT]), bool(cm5.search_mode[RIGHT]), bool(cm5.active_contact[RIGHT]),
                   float(cm5.commanded_target_W[RIGHT][2])))
lat = [s for s in states if s[0]]
check("late enabled (path check): search mode, inactive, target descends at 0.4 m/s from nominal",
      len(lat) > 5 and all(s[1] and not s[2] for s in lat) and lat[0][3] == nominal[RIGHT][2]
      and abs((lat[0][3] - lat[5][3]) - 5 * 0.4 * DT) < 1e-12,
      f"{len(lat)} late ticks, z0={lat[0][3] if lat else None}")
check("late path: managed pos = search target", np.array_equal(cm5.managed_foot_positions(nominal)[RIGHT],
                                                               cm5.commanded_target_W[RIGHT]))

# ===========================================================================
print()
if FAILS:
    print(f"FAILED {len(FAILS)} check(s): {FAILS}")
    sys.exit(1)
print("ALL PASS (owner B)")
