"""SRB 가정이 얼마나 맞나 — 걷는 중 '전신 평균 각속도' 가 몸통 각속도를 맞히는지 (Q&A 9/25 Q9).

Di Carlo SRB 는 로봇 전체를 한 강체로 보고 Θ̇ = Rᵀω 를 쓴다. 전신이 정말 한 강체처럼 돌면
    ω_전신 = I_c⁻¹ L_c   (L_c: CoM 기준 전신 각운동량, I_c: CoM 기준 합성 관성)
이 몸통 (MIT: torso, G1: pelvis) 각속도와 같아야 한다. 다리를 휘두르면 몸통이 반대로 돌아 둘이 어긋난다.
G1 기록 (9/24 Q17, baseline 0.5 m/s, 고정 관성): 상관 roll 0.13 · pitch 0.29 · yaw 0.17, RMS 비 0.52 · 0.09 · 2.15.
같은 식을 G1 에 쓰는 짝: MPC/src/affine/27_srb_validity.py.

재는 것 (heading 좌표 = 몸통 yaw 로 돌린 월드, 샘플 100 Hz, 창 8 ~ 20 s):
    (1) ω_전신 (관성 고정: 시작 자세의 I_c 를 yaw 로 돌림 / 관성 그 순간) vs ω_몸통 — 상관, RMS 비 (1 이면 크기까지 맞음)
    (2) 각운동량 분해 RMS: L_전신, L_다리 (다리 부분나무, CoM 기준 상대속도), L_나머지 = L_전신 − L_다리
    (3) 몸통 자세 흔들림 σ (roll, pitch) [deg]
측정은 별도 MjData 복사본에서 한다 — 컨트롤러는 1 틱 묵은 기구학을 쓰므로 원본에 mj_forward 를 부르면 제어가 바뀐다.

실행 (ROOT 에서):
    .venv/bin/python MPC/src/mit/14_srb_validity.py --vx 0.5 0.6 [--var "profile=ramp,fix=yawref"]
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import numpy as np

walk = importlib.import_module("10_walk")

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


def run(vx: float, var: str, seconds: float, t0: float, t1: float) -> dict:
    kw = walk.parse_variant(var)
    cfg, model, ctl, walk_t = walk._setup(vx, 0.0, 0.0, kw.get("solver", "osqp"), kw.get("armature", True),
                                          kw.get("profile", "const"), kw.get("markers", False), False,
                                          kw.get("friction_scale", 1.0), kw.get("fsm_double_update", True), False,
                                          osqp_settings=kw.get("osqp_settings"), fixes=kw.get("fixes", ()),
                                          x_dot_max=kw.get("x_dot_max"))
    m, d = model.m, model.d
    probe = Probe(m, model.base_body, d.qpos.copy())
    n = int(round(seconds / m.opt.timestep))
    every = max(1, int(round(0.01 / m.opt.timestep)))
    for k in range(n):
        tau_leg, tau_arm = ctl.tick()
        model.write_torque(tau_leg, tau_arm)
        mujoco.mj_step(m, d)
        if walk._fell(d):
            print(f"  넘어짐 t = {d.time:.2f} s — 그때까지 표본으로")
            break
        if k % every == 0 and t0 <= d.time <= t1:
            probe.sample(d.qpos, d.qvel)
    return probe.report(f"MIT  vx {vx}  [{var}]  창 {t0:.0f} ~ {min(t1, d.time):.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, nargs="+", default=[0.5, 0.6])
    ap.add_argument("--var", default="profile=ramp,fix=yawref")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--t0", type=float, default=8.0)
    a = ap.parse_args()
    for vx in a.vx:
        run(vx, a.var, a.seconds, a.t0, a.seconds)


if __name__ == "__main__":
    main()
