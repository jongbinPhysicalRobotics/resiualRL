"""MPC 로그(npz) 플롯 — 상태 추종 / wrench / 솔버 통계를 PNG 로 저장.

사용:
  .venv/Scripts/python.exe MPC/src/plot_log.py                  # 가장 최근 로그
  .venv/Scripts/python.exe MPC/src/plot_log.py MPC/logs/xxx.npz     # 지정 로그
PNG 는 npz 와 같은 이름으로 저장된다.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]  # 한글
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent


def main():
    if len(sys.argv) > 1:
        f = Path(sys.argv[1])
        if not f.is_absolute():
            f = ROOT / f
    else:
        cands = sorted(glob.glob(str(ROOT / "logs" / "*.npz")),
                       key=lambda f: Path(f).stat().st_mtime)   # 최근 수정순
        if not cands:
            raise SystemExit("logs/ 에 npz 가 없음")
        f = Path(cands[-1])

    z = np.load(f, allow_pickle=True)
    t = z["t"]
    x = z["x"]
    xr = z["x_ref"]
    u = z["u"]
    note = str(z["note"]) if "note" in z else ""

    fig, axes = plt.subplots(6, 2, figsize=(13, 17), sharex=True)
    fig.suptitle(f"{f.name}\n{note}", fontsize=10)

    ax = axes[0, 0]
    ax.plot(t, np.degrees(x[:, 0]), label="roll")
    ax.plot(t, np.degrees(x[:, 1]), label="pitch")
    ax.plot(t, np.degrees(x[:, 2]) - np.degrees(xr[:, 2]), label="yaw-ref", alpha=0.6)
    ax.set_ylabel("attitude [deg]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for i, lab in ((3, "px"), (4, "py"), (5, "pz")):
        ax.plot(t, x[:, i] - xr[:, i], label=f"{lab}−ref")
    ax.set_ylabel("CoM err [m]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for i, lab in ((6, "wx"), (7, "wy"), (8, "wz")):
        ax.plot(t, x[:, i], label=lab)
    ax.set_ylabel("omega [rad/s]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for i, lab in ((9, "vx"), (10, "vy"), (11, "vz")):
        ax.plot(t, x[:, i], label=lab)
    ax.set_ylabel("v_CoM [m/s]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2, 0]
    ax.plot(t, u[:, 2], label="L_Fz")
    ax.plot(t, u[:, 8], label="R_Fz")
    if "fz_contact" in z:
        ax.plot(t, z["fz_contact"], "k--", alpha=0.5, label="fz 실측(합)")
    ax.set_ylabel("Fz [N]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2, 1]
    ax.plot(t, u[:, 0], label="L_Fx")
    ax.plot(t, u[:, 1], label="L_Fy", alpha=0.7)
    ax.plot(t, u[:, 6], label="R_Fx", alpha=0.7)
    ax.set_ylabel("F 수평 [N]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[3, 0]
    ax.plot(t, u[:, 3], label="L_mx")
    ax.plot(t, u[:, 4], label="L_my")
    ax.plot(t, u[:, 10], label="R_my", alpha=0.7)
    ax.set_ylabel("ankle moment [N·m]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[3, 1]
    if "solve_ms" in z:
        sm = z["solve_ms"]; med = float(np.median(sm)); thr = max(3.0 * med, 20.0)
        ax.plot(t, sm, ".", ms=2, label="solve")
        ax.axhline(med, color="g", ls="--", alpha=0.6, label=f"median {med:.1f} ms")
        sp = sm > thr
        if sp.any():
            ax.plot(t[sp], sm[sp], "rx", ms=5, label=f"spike >{thr:.0f} ms ({int(sp.sum())})")
        ax.set_ylabel("QP solve [ms]"); ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── 다리 토크 (한계: hip_p/yaw 88, hip_roll/knee 139, ankle 50) ──
    LIM = np.array([88, 139, 88, 139, 50, 50] * 2, float)
    ax = axes[4, 0]
    if "tau" in z:
        tau = z["tau"]
        for j, lab in ((0, "L_hip_p"), (3, "L_knee"), (4, "L_ank_p"),
                       (6, "R_hip_p"), (9, "R_knee"), (10, "R_ank_p")):
            ax.plot(t, tau[:, j], lw=0.8, label=lab)
        for y_, c in ((139, "k"), (-139, "k"), (50, "r"), (-50, "r")):
            ax.axhline(y_, color=c, ls=":", alpha=0.35)
        ax.set_ylabel("leg τ [N·m]  (점선=한계)")
        ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3)

    ax = axes[4, 1]
    if "tau" in z:
        util = np.abs(tau[:, :12]) / LIM
        ax.plot(t, util.max(axis=1), lw=0.8, label="max |τ|/limit (다리 12)")
        ax.axhline(1.0, color="r", ls="--", alpha=0.6)
        ax.set_ylabel("torque util"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── 스윙 추종: 참조 vs 실제 (발 site 기준, stance 중 NaN) ──
    have_sw = "foot_ref" in z and "foot_pos" in z
    ax = axes[5, 0]
    if have_sw:
        fr = z["foot_ref"].reshape(-1, 2, 3); fp = z["foot_pos"].reshape(-1, 2, 3)
        for i, (lab, c) in enumerate((("L", "C2"), ("R", "C0"))):
            ax.plot(t, fr[:, i, 2], color=c, ls="--", lw=0.9, label=f"{lab} ref z")
            ax.plot(t, fp[:, i, 2], color=c, lw=0.9, alpha=0.8, label=f"{lab} act z")
        ax.set_ylabel("foot z [m]"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.set_xlabel("t [s]")

    ax = axes[5, 1]
    if have_sw:
        err = np.linalg.norm(fr - fp, axis=2) * 100.0          # (n,2) cm, stance=NaN
        for i, (lab, c) in enumerate((("L", "C2"), ("R", "C0"))):
            ax.plot(t, err[:, i], color=c, lw=0.8, label=f"{lab} |ref−act|")
        if np.isfinite(err).any():
            ax.set_title(f"스윙 추종오차  평균 {np.nanmean(err):.2f} cm / 최대 {np.nanmax(err):.1f} cm",
                         fontsize=9)
        ax.set_ylabel("swing err [cm]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlabel("t [s]")

    out = f.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("저장:", out)


if __name__ == "__main__":
    main()
