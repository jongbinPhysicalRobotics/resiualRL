"""Step 1 — 부호 규약 검증. QP 코드 짜기 전에 반드시 통과해야 한다.

검증 내용:
  (0) 좌표축: x=앞, z=위 가 맞는지 모델에서 확인
  (1) lstsq 서 있기 해(접촉점 8개 힘)로 발 하나의 my 를 두 방식으로 계산:
        A. 외적 직접:  my = Σ_j [ (r_j − r_site) × f_j ]_y
        B. CoP 공식 :  my = −Fz · x_CoP,rel
      → 두 값이 같은 부호·같은 크기면 규약 통과
  (2) my 가 제약 범위 [−l_t·Fz, +l_h·Fz] 안인지
  (3) 기준점 주의: 명세의 예상값 −3.87 은 world 원점 기준(−163.5×0.0237).
      제약의 l_t/l_h 는 '발목(site) 기준'이므로 발목 기준 my ≈ −2.1 이 통과값.

사용: .venv/Scripts/python.exe MPC/src/05_sign_check.py
"""
from __future__ import annotations

import numpy as np
import mujoco
from importlib import import_module

import g1_model

m3 = import_module("03_grf_to_tau")

np.set_printoptions(precision=4, suppress=True)

m, d = g1_model.load_torque()
g1_model.set_crouch(m, d)
W = mujoco.mj_getTotalmass(m) * 9.81

print("=" * 78)
print("(0) 좌표축 확인")
print("=" * 78)
print("  gravity =", m.opt.gravity, " -> z가 위 (중력이 -z)")
for side in ("left", "right"):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
    xs = [m.geom_pos[g][0] for g in range(m.ngeom)
          if m.geom_bodyid[g] == bid and m.geom_group[g] == 3]
    print(f"  {side} 발가락 local x = {max(xs):+.3f}, 뒤꿈치 local x = {min(xs):+.3f}"
          f"  -> +x 가 앞 (발가락 방향)")

# 발 형상 상수 (모델에서 직접 추출 — 하드코딩 금지 원칙)
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
locals_ = np.array([m.geom_pos[g] for g in range(m.ngeom)
                    if m.geom_bodyid[g] == bid and m.geom_group[g] == 3])
l_t = locals_[:, 0].max()          # 발가락 (앞)
l_h = -locals_[:, 0].min()         # 뒤꿈치 (뒤)
w = np.abs(locals_[:, 1]).min()    # 좌우 반폭 (보수적: 최솟값)
print(f"\n  발 상수 (모델 추출): l_t={l_t:.3f}, l_h={l_h:.3f}, w={w:.3f}")
print(f"  ⚠ 명세는 w=0.03 이지만 뒤꿈치 y=±0.025 라 보수적으로 w={w:.3f} 사용")

print()
print("=" * 78)
print("(1) lstsq 해에서 발별 wrench 계산 — 두 방식 비교")
print("=" * 78)
pts = m3.contact_point_jacobians(m, d)
A = np.hstack([p[2][:, 0:6].T for p in pts])
f_ls, *_ = np.linalg.lstsq(A, d.qfrc_bias[0:6], rcond=None)

ok = True
for side, site in (("left", "left_foot"), ("right", "right_foot")):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
    r_site = d.site_xpos[sid]
    idx = [i for i, p in enumerate(pts) if p[3] == side]
    F = np.zeros(3)
    mom = np.zeros(3)          # 방식 A: 외적 직접
    for i in idx:
        fj = f_ls[3 * i:3 * i + 3]
        rj = pts[i][1] - r_site
        F += fj
        mom += np.cross(rj, fj)
    Fz = F[2]
    x_cop_rel = sum((pts[i][1][0] - r_site[0]) * f_ls[3 * i + 2] for i in idx) / Fz
    y_cop_rel = sum((pts[i][1][1] - r_site[1]) * f_ls[3 * i + 2] for i in idx) / Fz
    my_cop = -Fz * x_cop_rel   # 방식 B: CoP 공식
    mx_cop = +Fz * y_cop_rel

    lo, hi = -l_t * Fz, l_h * Fz
    in_range = lo <= mom[1] <= hi
    match = np.isclose(mom[1], my_cop, atol=1e-6) and np.isclose(mom[0], mx_cop, atol=1e-6)
    ok &= in_range and match

    print(f"\n  [{side}] Fz = {Fz:.2f} N,  site = {r_site}")
    print(f"    방식 A (외적):  mx = {mom[0]:+.4f},  my = {mom[1]:+.4f} N·m")
    print(f"    방식 B (CoP):   mx = {mx_cop:+.4f},  my = {my_cop:+.4f} N·m"
          f"   (x_CoP,rel = {x_cop_rel:+.4f} m)")
    print(f"    일치 여부: {'OK' if match else 'MISMATCH!'}")
    print(f"    my 범위 [{lo:.2f}, {hi:.2f}] 안: {'OK' if in_range else 'VIOLATION!'}")

print()
print("=" * 78)
print("(2) 기준점 정리 — 명세 예상값 −3.87 과의 차이")
print("=" * 78)
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
x_site = d.site_xpos[sid][0]
x_cop_world = sum(pts[i][1][0] * f_ls[3 * i + 2] for i in range(len(pts))) \
    / sum(f_ls[2::3])
print(f"  world CoP x       = {x_cop_world:+.4f}  (README의 0.0237)")
print(f"  발목 site x       = {x_site:+.4f}")
print(f"  발목 기준 CoP x   = {x_cop_world - x_site:+.4f}")
print(f"  −Fz·x_CoP(world 기준) = {-W/2 * x_cop_world:+.3f}  <- 명세의 −3.87 (world 원점 기준 모멘트)")
print(f"  −Fz·x_CoP(발목 기준)  = {-W/2 * (x_cop_world - x_site):+.3f}  <- 제약과 같은 기준 = 통과값")
print()
print("  결론: 제약(l_t/l_h)은 발목 기준이므로 my ≈ −2.1 N·m 가 올바른 기대값.")
print("        부호(음수 = CoP가 발목 앞 = 앞으로 기울음)는 두 기준 모두 동일.")
print()
print("STEP 1 " + ("통과 ✓" if ok else "실패 — 부호 규약 재검토 필요"))
