"""착지~stance 구간의 명령 wrench vs 실측 접촉력 분석.

스케줄상 착지(gait 전환) 시각을 t=0 으로 놓고 여러 스텝을 정렬해 평균낸다.
"MPC 가 아직 공중인 발에 하중을 명령하는가" 를 눈으로 보기 위한 것 (Q&A 9/16 Q3, 9/21).

사용:
  .venv/Scripts/python.exe MPC/src/baseline/12_contact_analysis.py --vx 0.5 --seconds 25
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_model, mpc_srb
import importlib
walk = importlib.import_module("09_walk")


def foot_contact_wrench(m, d, bid, origin):
    """발 bid 의 실측 접촉 wrench (world, origin 기준)."""
    F = np.zeros(3); M = np.zeros(3); f6 = np.zeros(6)
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
        if bid not in (b1, b2):
            continue
        mujoco.mj_contactForce(m, d, c, f6)
        fw = con.frame.reshape(3, 3).T @ f6[:3]
        if b1 == bid:
            fw = -fw
        F += fw; M += np.cross(con.pos - origin, fw)
    return F, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--settle", type=float, default=8.0)
    ap.add_argument("--pre", type=float, default=0.06, help="착지 전 창 [s]")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    m, d = g1_model.load_torque(); g1_model.set_crouch(m, d)
    ctl = walk.WalkController(m, d, vx_cmd=a.vx, kp_up=300.0, swing_id=True,
                              soft_land=True, lam_swing=True, wn_swing=30.0,
                              zeta_swing=0.7, stance_frac=0.57, q_py=300, gate_sy=False)
    fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link")
          for s in ("left", "right")]
    dt = m.opt.timestep
    T_st = ctl.gait.T_stance
    Mg = ctl.params.mass * 9.81

    rec = []          # (t, i, cmdFz, actFz, cmd_my, act_my, foot_z, stance?)
    prev = [True, True]
    td_times = [[], []]
    for k in range(int(a.seconds / dt)):
        t = k * dt
        mujoco.mj_forward(m, d)
        if k % walk.DECIM == 0:
            ctl.update_mpc(d, t)
        feet = mpc_srb.get_foot_positions(m, d)
        for i in range(2):
            st = ctl.gait.in_stance(t, i)
            if st and not prev[i] and t > a.settle:
                td_times[i].append(t)
            prev[i] = st
            Fa, Ma = foot_contact_wrench(m, d, fb[i], feet[i])
            rec.append((t, i, ctl.wr[i, 2], Fa[2], ctl.wr[i, 4], Ma[1],
                        feet[i, 2] - ctl.z_ground, float(st)))
        d.ctrl[:] = ctl.torque(d, t); mujoco.mj_step(m, d)
        if d.qpos[2] < 0.5:
            print(f"  ⚠ {t:.1f}s 전도"); break

    R = np.array(rec)
    print(f"명령 wrench vs 실측 접촉력 — 스케줄 착지 시각 정렬 (vx={a.vx}, "
          f"stance {T_st*1000:.0f} ms, 체중 {Mg:.0f} N)")
    print(f"이벤트: 왼 {len(td_times[0])}, 오른 {len(td_times[1])}\n")

    # 위상 격자: −pre ~ +T_st
    grid = np.arange(-a.pre, T_st + 1e-9, 0.004)
    prof = {k: [] for k in ("cmdFz", "actFz", "z", "cmd_my", "act_my")}
    lat = []
    for i in range(2):
        Ri = R[R[:, 1] == i]
        ti = Ri[:, 0]
        for td in td_times[i]:
            idx = np.searchsorted(ti, td + grid)
            idx = np.clip(idx, 0, len(ti) - 1)
            prof["cmdFz"].append(Ri[idx, 2]); prof["actFz"].append(Ri[idx, 3])
            prof["z"].append(Ri[idx, 6])
            prof["cmd_my"].append(Ri[idx, 4]); prof["act_my"].append(Ri[idx, 5])
            # 실제 접촉까지 지연
            after = Ri[(ti >= td) & (ti < td + 0.15)]
            on = after[after[:, 3] > 10.0]
            lat.append((on[0, 0] - td) * 1000 if len(on) else np.nan)
    P = {k: np.array(v) for k, v in prof.items()}
    lat = np.array(lat)

    print(f"{'스케줄 대비':>10s} {'명령 Fz':>9s} {'실측 Fz':>9s} {'차이':>8s} "
          f"{'발높이':>8s} {'접촉률':>7s}")
    for j, g in enumerate(grid):
        if abs(g * 1000) % 20 > 4 and g > -a.pre + 1e-9:
            continue
        c, ac, z = P["cmdFz"][:, j], P["actFz"][:, j], P["z"][:, j]
        mark = "  ← 스케줄 착지" if abs(g) < 2e-3 else ""
        print(f"{g*1000:+9.0f}ms {c.mean():8.1f}N {ac.mean():8.1f}N "
              f"{c.mean()-ac.mean():+7.1f}N {z.mean()*100:7.2f}cm "
              f"{100*np.mean(ac > 10):6.0f}%{mark}")

    print(f"\n실제 접촉까지 지연: 평균 {np.nanmean(lat):.1f} ms, "
          f"중앙값 {np.nanmedian(lat):.1f}, 최대 {np.nanmax(lat):.1f} ms "
          f"(미접촉 {int(np.isnan(lat).sum())}/{len(lat)})")

    # stance 전체에서 명령-실측 괴리
    st_mask = grid >= 0
    c_all = P["cmdFz"][:, st_mask]; a_all = P["actFz"][:, st_mask]
    print(f"stance 전체: 명령 평균 {c_all.mean():.1f} N, 실측 평균 {a_all.mean():.1f} N "
          f"(실측/명령 {100*a_all.mean()/c_all.mean():.0f} %)")
    early = grid[st_mask] < 0.05
    print(f"  초기 50 ms: 명령 {c_all[:, early].mean():.1f} N, "
          f"실측 {a_all[:, early].mean():.1f} N "
          f"({100*a_all[:, early].mean()/max(c_all[:, early].mean(),1e-9):.0f} %)")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        g = grid * 1000
        ax[0].plot(g, P["cmdFz"].mean(0), label="명령 Fz (MPC u)")
        ax[0].plot(g, P["actFz"].mean(0), label="실측 접촉 Fz")
        ax[0].fill_between(g, np.percentile(P["actFz"], 10, 0),
                           np.percentile(P["actFz"], 90, 0), alpha=0.2)
        ax[0].axvline(0, color="k", ls="--", alpha=0.5)
        ax[0].axhline(ctl.params.fz_min, color="r", ls=":", label=f"Fz_min={ctl.params.fz_min:.0f} N")
        ax[0].set_ylabel("Fz [N]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        ax[1].plot(g, P["z"].mean(0) * 100)
        ax[1].axvline(0, color="k", ls="--", alpha=0.5); ax[1].axhline(0, color="g", ls=":")
        ax[1].set_ylabel("발 높이 [cm]"); ax[1].grid(alpha=0.3)
        ax[2].plot(g, P["cmd_my"].mean(0), label="명령 my")
        ax[2].plot(g, P["act_my"].mean(0), label="실측 my")
        ax[2].axvline(0, color="k", ls="--", alpha=0.5)
        ax[2].set_ylabel("my [N·m]"); ax[2].set_xlabel("스케줄 착지 대비 [ms]")
        ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
        fig.suptitle(f"착지~stance: 명령 vs 실측 (vx={a.vx}, {len(lat)} 스텝 평균)")
        fig.tight_layout()
        from paths import MPC_ROOT
        out = MPC_ROOT / "logs" / f"contact_vx{a.vx:g}.png"
        fig.savefig(out, dpi=110); print(f"\n저장: {out.name}")


if __name__ == "__main__":
    main()
