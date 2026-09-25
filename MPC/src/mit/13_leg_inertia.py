"""다리 스윙 관성 — MIT vs G1 (Q&A 9/24 Q29 를 지금 모델로 다시, 9/25 Q9).

hip pitch 관절 아래 (휘둘리는 쪽) 의 관성을 hip pitch 축 기준으로, 나머지 (몸통 쪽) 의 pitch 관성을 자기 CoM 기준으로 잰다.
    스윙 관성  I_sw = Σ_b [ nᵀ R_b I_b R_bᵀ n + m_b |r_b⊥|² ]    (r_b⊥ = 링크 CoM 에서 hip pitch 축까지의 수직 거리)
    몸통 쪽    I_bd = 두 스윙 부분나무를 뺀 나머지의 합성 관성 (자기 CoM 기준) 의 yy 성분
    비율       I_sw (한쪽) / I_bd  — 다리를 휘두를 때 몸통이 반대로 도는 정도의 지표 (Q29)

실행 (ROOT 에서):
    .venv/bin/python MPC/src/mit/13_leg_inertia.py [--pbrs <mit_humanoid_fixed_arms.urdf>]
--pbrs: Q29 가 쓴 공개 URDF (se-hwan/pbrs-humanoid dev, resources/robots/mit_humanoid/). 주면 방법 검증용으로 같이 잰다.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import numpy as np

import paths

G1_SCENE = paths.MPC_ROOT.parent / "mujoco_menagerie" / "unitree_g1" / "scene.xml"


def load_pbrs(urdf: Path) -> mujoco.MjModel:
    """URDF 의 visual·collision 을 지우고 (메시 없이) 질량·관성만으로 컴파일. 발 관성은 balanceinertia 로 최소 보정 (Q29 와 같음)."""
    root = ET.parse(urdf).getroot()
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for e in link.findall(tag):
                link.remove(e)
    mj = ET.SubElement(root, "mujoco")
    ET.SubElement(mj, "compiler", balanceinertia="true", discardvisual="true", fusestatic="false")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def subtree(m, root: int) -> list[int]:
    out = []
    for b in range(1, m.nbody):
        a = b
        while a > 0 and a != root:
            a = m.body_parentid[a]
        if a == root:
            out.append(b)
    return out


def hinge(m, name_part: str) -> list[int]:
    return [j for j in range(m.njnt) if name_part in m.joint(j).name and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]


def leg_root(m, pitch_body: int) -> int:
    """hip pitch 몸체에서 위로 올라가며 'hip' 관절이 붙은 마지막 몸체 = 다리 시작."""
    b = pitch_body
    while True:
        p = m.body_parentid[b]
        js = [j for j in range(m.njnt) if m.jnt_bodyid[j] == p]
        if p == 0 or not js or not any("hip" in m.joint(j).name for j in js):
            return b
        b = p


def measure(m, qpos=None) -> dict:
    d = mujoco.MjData(m)
    if qpos is not None:
        d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    total = float(m.body_subtreemass[0])
    pitch_j = hinge(m, "hip_pitch")
    assert len(pitch_j) == 2, [m.joint(j).name for j in pitch_j]
    swing_sets, legs, I_sw, rows = [], [], [], []
    for j in pitch_j:
        pb = m.jnt_bodyid[j]
        bs = subtree(m, pb)
        swing_sets.append(bs)
        legs.append(subtree(m, leg_root(m, pb)))
        n, p = d.xaxis[j] / np.linalg.norm(d.xaxis[j]), d.xanchor[j]
        tot = 0.0
        for b in bs:
            Rb = d.ximat[b].reshape(3, 3)
            Ib = Rb @ np.diag(m.body_inertia[b]) @ Rb.T
            r = d.xipos[b] - p
            rperp = r - (r @ n) * n
            c = float(n @ Ib @ n + m.body_mass[b] * (rperp @ rperp))
            tot += c
            rows.append((m.joint(j).name, m.body(b).name, float(m.body_mass[b]), float(np.linalg.norm(rperp)), c))
        I_sw.append(tot)
    swing_all = set(swing_sets[0]) | set(swing_sets[1])
    body = [b for b in range(1, m.nbody) if b not in swing_all]
    mb = sum(m.body_mass[b] for b in body)
    c = sum(m.body_mass[b] * d.xipos[b] for b in body) / mb
    I = np.zeros((3, 3))
    for b in body:
        Rb = d.ximat[b].reshape(3, 3)
        r = d.xipos[b] - c
        I += Rb @ np.diag(m.body_inertia[b]) @ Rb.T + m.body_mass[b] * ((r @ r) * np.eye(3) - np.outer(r, r))
    leg_mass = sum(m.body_mass[b] for L in legs for b in L)
    swing_mass = sum(m.body_mass[b] for b in swing_all)
    return dict(total=total, leg_frac=leg_mass / total, swing_frac=swing_mass / total, I_sw=I_sw,
                I_bd=float(I[1, 1]), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbrs", type=Path, default=None, help="pbrs-humanoid mit_humanoid_fixed_arms.urdf (Q29 검증용)")
    a = ap.parse_args()

    cases = []
    mit = mujoco.MjModel.from_xml_path(str(paths.MIT_SCENE_XML))
    cases.append(("MIT 지금 모델 (관절 0)", mit, None))
    cases.append(("MIT 지금 모델 (stand 키프레임)", mit, mit.key("stand").qpos.copy()))
    if a.pbrs is not None:
        cases.append(("MIT pbrs URDF (관절 0) — Q29", load_pbrs(a.pbrs), None))
    g1 = mujoco.MjModel.from_xml_path(str(G1_SCENE))
    cases.append(("G1 (관절 0)", g1, None))
    q = g1.key("stand").qpos.copy() if g1.nkey else g1.qpos0.copy()
    for side in ("left", "right"):                              # baseline g1_model.set_crouch 와 같은 각도
        for jn, v in ((f"{side}_hip_pitch_joint", -0.30), (f"{side}_knee_joint", 0.60), (f"{side}_ankle_pitch_joint", -0.30)):
            q[g1.jnt_qposadr[g1.joint(jn).id]] = v
    cases.append(("G1 (웅크린 자세)", g1, q))

    print(f"{'모델':32s} {'총질량':>7s} {'다리':>6s} {'스윙쪽':>6s} {'스윙관성 L/R [kg·m²]':>22s} {'몸통 pitch':>10s} {'비율':>6s}")
    res = {}
    for name, m, qp in cases:
        r = measure(m, qp)
        res[name] = r
        sw = np.mean(r["I_sw"])
        print(f"{name:32s} {r['total']:6.2f}kg {100 * r['leg_frac']:5.1f}% {100 * r['swing_frac']:5.1f}%"
              f" {r['I_sw'][0]:10.3f} / {r['I_sw'][1]:.3f} {r['I_bd']:10.3f} {sw / r['I_bd']:6.2f}")

    for name in [n for n in res if n.startswith("MIT")][:1] + [n for n in res if "pbrs" in n]:
        r = res[name]
        print(f"\n{name} — 링크별 (hip pitch 축 기준, 한쪽 다리)")
        pj = r["rows"][0][0]
        tot = sum(c for j, *_x, c in r["rows"] if j == pj)
        for j, b, mass, dist, c in r["rows"]:
            if j == pj:
                print(f"    {b:24s} {mass:6.3f} kg   축에서 {100 * dist:5.1f} cm   {100 * c / tot:5.1f} %")


if __name__ == "__main__":
    main()
