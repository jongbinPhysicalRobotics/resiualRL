"""준정적(quasi-static) 서 있기 컨트롤러.

매 스텝:
  1. 현재 q 에서 접촉점 자코비안 J_c 계산
  2. 베이스 6행 wrench 균형  g[0:6] = sum J_c[:,0:6]^T f_c  를 lstsq 로 풀어 f_c 배분
  3. tau = g[6:] - (sum J_c^T f_c)[6:]  를 액추에이터에 인가

이것은 MPC 가 아니다. MPC 의 '층1'(재료 만들기: 자코비안, 중력, wrench 균형)만
있고 '층2'(솔버)는 lstsq 한 줄로 대체한 상태 = F->tau 파이프라인만 검증하는 판.

사용:
  .venv/Scripts/python.exe MPC/src/04_stand.py            # 30초 + 외란 테스트 (헤드리스)
  .venv/Scripts/python.exe MPC/src/04_stand.py --view     # 뷰어로 보기
  .venv/Scripts/python.exe MPC/src/04_stand.py --view --pd # 자세 PD 를 얹어서 보기
"""
from __future__ import annotations

import sys

import numpy as np
import mujoco

import g1_model
from importlib import import_module

_m3 = import_module("03_grf_to_tau")
contact_point_jacobians = _m3.contact_point_jacobians
actuated_dofs = _m3.actuated_dofs


class QuasiStaticStand:
    """접촉력 배분 -> 중력보상 토크. 선택적으로 관절 PD 를 더한다."""

    def __init__(self, m, kp=0.0, kd=0.0):
        self.m = m
        self.adof = actuated_dofs(m)
        self.kp = kp
        self.kd = kd
        self.q_ref = None          # set_reference() 로 채운다
        self.last_f = None

    def set_reference(self, d):
        self.q_ref = d.qpos[7:].copy()

    def __call__(self, m, d):
        mujoco.mj_forward(m, d)
        pts = contact_point_jacobians(m, d)
        tau_c = np.zeros(m.nv)
        if pts:
            A = np.hstack([p[2][:, 0:6].T for p in pts])
            f, *_ = np.linalg.lstsq(A, d.qfrc_bias[0:6], rcond=None)
            self.last_f = f
            for i, p in enumerate(pts):
                tau_c += p[2].T @ f[3 * i:3 * i + 3]
        tau = d.qfrc_bias[self.adof] - tau_c[self.adof]
        if self.kp and self.q_ref is not None:
            tau += self.kp * (self.q_ref - d.qpos[7:]) - self.kd * d.qvel[self.adof]
        return tau


def headless(seconds=30.0, push_at=(10.0, 20.0), push_N=40.0, push_dur=0.1, kp=0.0, kd=0.0):
    m, d = g1_model.load_torque()
    g1_model.set_crouch(m, d)
    ctl = QuasiStaticStand(m, kp=kp, kd=kd)
    ctl.set_reference(d)

    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    z0 = d.qpos[2]
    dt = m.opt.timestep
    n = int(seconds / dt)

    print("  t[s]   pelvis_z    CoM_x    CoM_y   |tau|max   ncon   event")
    next_report = 0.0
    for k in range(n):
        t = k * dt
        d.ctrl[:] = ctl(m, d)

        d.xfrc_applied[:] = 0.0
        event = ""
        for pt in push_at:
            if pt <= t < pt + push_dur:
                d.xfrc_applied[pelvis, 0] = push_N
                event = "<- push %.0fN +x" % push_N
        mujoco.mj_step(m, d)

        if t >= next_report - 1e-9:
            print("  %5.1f   %8.4f  %7.4f  %7.4f   %8.2f    %2d   %s"
                  % (t, d.qpos[2], d.subtree_com[0][0], d.subtree_com[0][1],
                     np.abs(d.ctrl).max(), d.ncon, event))
            next_report += 2.0
        if d.qpos[2] < z0 - 0.20:
            print("  쓰러짐 @ %.2fs" % t)
            return False
    print()
    print("  %.0f초 생존. pelvis z: %.4f -> %.4f (drift %.4f m)"
          % (seconds, z0, d.qpos[2], d.qpos[2] - z0))
    return True


def view(kp=0.0, kd=0.0):
    import mujoco.viewer

    m, d = g1_model.load_torque()
    g1_model.set_crouch(m, d)
    ctl = QuasiStaticStand(m, kp=kp, kd=kd)
    ctl.set_reference(d)

    print("뷰어: tau = g - J^T f 로 서 있습니다. 창을 닫으면 종료.")
    print("  (뷰어에서 Ctrl+마우스 우클릭 드래그로 밀어볼 수 있습니다)")
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            d.ctrl[:] = ctl(m, d)
            mujoco.mj_step(m, d)
            v.sync()


if __name__ == "__main__":
    use_pd = "--pd" in sys.argv
    kp, kd = (30.0, 2.0) if use_pd else (0.0, 0.0)
    if "--view" in sys.argv:
        view(kp, kd)
    else:
        print("=" * 78)
        print("A) 순수 준정적 피드포워드 (PD 없음)")
        print("=" * 78)
        headless(kp=0.0, kd=0.0)
        print()
        print("=" * 78)
        print("B) 준정적 + 관절 PD (kp=30, kd=2)")
        print("=" * 78)
        headless(kp=30.0, kd=2.0)
