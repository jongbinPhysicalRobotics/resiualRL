"""SRB 가정이 얼마나 맞나 — G1 · split (reference 방식: 상체 SRB + 골반 ω) 판 (Q&A 9/25 Q9). 측정 식은 MPC/src/mit/14_srb_validity.py 와 같다 (Probe 복사본).

걷는 중 '전신 평균 각속도' ω_전신 = I_c⁻¹ L_c 가 골반 각속도를 맞히는지, 각운동량 분해, 골반 자세 σ.
MIT 이식과 같은 MPC 정식화를 G1 에 올린 것 (24_walk_split.SplitWalk, 기본 = reference 스윙 게인). 걷기 루프는 affine 판과 같다 (mj_forward → update_mpc → torque → mj_step). 측정은 별도 MjData 복사본에서.
G1 기록 (9/24 Q17, baseline 0.5 m/s, 고정 관성): 상관 roll 0.13 · pitch 0.29 · yaw 0.17, RMS 비 0.52 · 0.09 · 2.15 — 이 스크립트 검증용.

실행 (ROOT 에서, 24_walk_split 과 같은 플래그):
    F="--uppd 300 --swingid --softland --lamswing --wn 30 --zeta 0.7 --sf 0.57 --qpy 300 --swingyaw --sidew 13 --tdscale 1.25 --copm 0.9"
    GAIT_ON=1 .venv/bin/python MPC/src/split/27_srb_validity.py --vx 0.5 --seconds 20 $F [--refq]
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import numpy as np

import g1_model
import walk_cli

walk = importlib.import_module("09_walk")
spl = importlib.import_module("24_walk_split")

AX = ("roll", "pitch", "yaw")


def rz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def leg_roots(m) -> list[int]:
    """hip pitch 몸체에서 위로 'hip' 관절이 붙은 마지막 몸체 = 다리 시작 (13_leg_inertia 와 같은 규칙)."""
    out = []
    for j in range(m.njnt):
        if "hip_pitch" not in m.joint(j).name:
            continue
        b = m.jnt_bodyid[j]
        while True:
            p = m.body_parentid[b]
            js = [k for k in range(m.njnt) if m.jnt_bodyid[k] == p]
            if p == 0 or not js or not any("hip" in m.joint(k).name for k in js):
                break
            b = p
        out.append(b)
    return out


def composite_inertia(m, d, C) -> np.ndarray:
    I = np.zeros((3, 3))
    for b in range(1, m.nbody):
        Rb = d.ximat[b].reshape(3, 3)
        r = d.xipos[b] - C
        I += Rb @ np.diag(m.body_inertia[b]) @ Rb.T + m.body_mass[b] * ((r @ r) * np.eye(3) - np.outer(r, r))
    return I


class Probe:
    """qpos/qvel 을 받아 복사본에서 전신·다리 각운동량과 몸통 각속도를 잰다."""

    def __init__(self, m, base_body: int, qpos0: np.ndarray):
        self.m, self.d = m, mujoco.MjData(m)
        self.base = base_body
        self.roots = leg_roots(m)
        self.d.qpos[:] = qpos0
        mujoco.mj_forward(m, self.d)
        yaw0 = self._yaw()
        R0 = rz(yaw0)
        self.I0_h = R0.T @ composite_inertia(m, self.d, self.d.subtree_com[0]) @ R0   # 시작 자세, heading 좌표
        self.rows = []

    def _yaw(self) -> float:
        R = self.d.xmat[self.base].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def sample(self, qpos, qvel):
        m, d = self.m, self.d
        d.qpos[:], d.qvel[:] = qpos, qvel
        mujoco.mj_forward(m, d)
        mujoco.mj_subtreeVel(m, d)
        C, vC = d.subtree_com[0].copy(), d.subtree_linvel[0].copy()
        L = d.subtree_angmom[0].copy()
        Lleg = np.zeros(3)
        for b in self.roots:
            mb = float(m.body_subtreemass[b])
            Lleg += d.subtree_angmom[b] + mb * np.cross(d.subtree_com[b] - C, d.subtree_linvel[b] - vC)
        Rb = d.xmat[self.base].reshape(3, 3)
        w_base = d.cvel[self.base][:3].copy()                    # 월드 좌표 각속도 (cvel 회전 성분)
        yaw = self._yaw()
        H = rz(yaw).T                                            # 월드 → heading
        Lh, Llh, wbh = H @ L, H @ Lleg, H @ w_base
        w_fix = np.linalg.solve(self.I0_h, Lh)
        w_inst = np.linalg.solve(H @ composite_inertia(m, d, C) @ H.T, Lh)
        roll = float(np.arctan2(Rb[2, 1], Rb[2, 2]))
        pitch = float(np.arcsin(-np.clip(Rb[2, 0], -1.0, 1.0)))
        self.rows.append(np.concatenate([wbh, w_fix, w_inst, Lh, Llh, [roll, pitch]]))

    def report(self, label: str) -> dict:
        A = np.array(self.rows)
        wb, wf, wi, L, Ll = A[:, 0:3], A[:, 3:6], A[:, 6:9], A[:, 9:12], A[:, 12:15]
        att = A[:, 15:17]

        def corr(a, b):
            a, b = a - a.mean(), b - b.mean()
            den = np.sqrt((a * a).sum() * (b * b).sum())
            return float((a * b).sum() / den) if den > 0 else float("nan")

        def rms(a):
            a = a - a.mean()
            return float(np.sqrt((a * a).mean()))

        out = dict(n=len(A))
        print(f"\n{label}  (표본 {len(A)})")
        print(f"    {'':26s} " + " ".join(f"{a:>13s}" for a in AX))
        for name, w in (("ω_전신 (관성 고정)", wf), ("ω_전신 (관성 그 순간)", wi)):
            cs = [corr(w[:, a], wb[:, a]) for a in range(3)]
            rs = [rms(w[:, a]) / max(rms(wb[:, a]), 1e-12) for a in range(3)]
            out[name] = (cs, rs)
            print(f"    {name + ' 상관 / RMS 비':26s} " + " ".join(f"{c:6.2f} / {r:4.2f}" for c, r in zip(cs, rs)))
        Lr = L - Ll
        print(f"    {'RMS L_전신 · L_다리 · L_나머지':26s} "
              + " ".join(f"{rms(L[:, a]):.3f}·{rms(Ll[:, a]):.3f}·{rms(Lr[:, a]):.3f}" for a in range(3)))
        print(f"    {'몸통 ω RMS [rad/s]':26s} " + " ".join(f"{rms(wb[:, a]):13.3f}" for a in range(3)))
        sd = np.degrees(att.std(axis=0))
        print(f"    몸통 자세 σ: roll {sd[0]:.2f}°  pitch {sd[1]:.2f}°")
        out["L"] = [(rms(L[:, a]), rms(Ll[:, a]), rms(Lr[:, a])) for a in range(3)]
        out["att_sd"] = sd.tolist()
        return out


def run(A: dict, refq: bool, t0: float) -> dict:
    m, d = g1_model.load_torque()
    g1_model.set_crouch(m, d)
    kw = dict(A["common"])
    kw["wz_cmd"] = kw.pop("wz", 0.0)             # headless() 인자명 → 생성자 인자명
    ctl = spl.SplitWalk(m, d, vx_cmd=A["vx"], wn3=spl.REF_WN3, kd3=spl.REF_KD3, ref_q=refq, **kw)
    probe = Probe(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"), d.qpos.copy())
    dt, seconds = m.opt.timestep, A["seconds"]
    every = max(1, int(round(0.01 / dt)))
    z0 = d.qpos[2]
    t = 0.0
    for k in range(int(seconds / dt)):
        t = k * dt
        mujoco.mj_forward(m, d)
        if k % walk.DECIM == 0:
            ctl.update_mpc(d, t)
        d.ctrl[:] = ctl.torque(d, t)
        mujoco.mj_step(m, d)
        if d.qpos[2] < z0 - 0.25:
            print(f"  넘어짐 t = {t:.2f} s — 그때까지 표본으로")
            break
        if k % every == 0 and t >= t0:
            probe.sample(d.qpos, d.qvel)
    return probe.report(f"G1  vx {A['vx']}  [split{' + refq' if refq else ''}]  창 {t0:.0f} ~ {t:.1f} s")


if __name__ == "__main__":
    A = walk_cli.parse()
    t0 = float(sys.argv[sys.argv.index("--t0") + 1]) if "--t0" in sys.argv else 8.0
    run(A, "--refq" in sys.argv, t0)
