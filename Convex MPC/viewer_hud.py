"""뷰어 화면 글자 (sim/실제 시간, 배속, 명령·실측 속도 성분) — casadi_mit (Q&A 9/26 Q8 후속).

G1 viewer_hud.ViewerHUD (Q&A 9/22 Q5) 를 MIT 재작업용으로 다시 씀:
명령이 vx·vy·wz 로 따로 오므로 **성분별로 명령 | 실측(주기 평균) | 순간 | 추종 %** 를 나란히 보여 준다.
실측 vx·vy 는 전신 CoM 속도의 heading(몸통 yaw) 좌표 성분, wz 는 몸통 각속도 z (world).
MuJoCo 내장 글꼴이 한글을 못 그려서 표시는 영어.
"""
import collections
import time as _time

import mujoco
import numpy as np


class MitHUD:
    def __init__(self, m, base_body: int, every: int = 25, avg_window: float = 0.5):
        self.m = m
        self.base = int(base_body)
        self.every = max(1, int(every))
        n = max(1, int(round(avg_window / (self.every * m.opt.timestep))))
        self.buf = [collections.deque(maxlen=n) for _ in range(3)]      # vx, vy, wz
        self.hist = collections.deque()
        self.wall0 = _time.perf_counter()

    def update(self, viewer, d, t, k, cmd=(0.0, 0.0, 0.0), mode="", extra=()):
        if k % self.every:
            return
        mujoco.mj_subtreeVel(self.m, d)
        R = d.xmat[self.base].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        vel = d.subtree_linvel[0]                                        # 전신 CoM 속도 (world)
        now = (vel[0] * c + vel[1] * s,                                  # heading 앞
               -vel[0] * s + vel[1] * c,                                 # heading 옆
               float(d.cvel[self.base][2]))                              # 몸통 ω_z (world)
        for b, v in zip(self.buf, now):
            b.append(float(v))
        avg = [float(np.mean(b)) for b in self.buf]
        wall = _time.perf_counter() - self.wall0
        self.hist.append((t, wall))
        while len(self.hist) > 2 and wall - self.hist[0][1] > 1.0:
            self.hist.popleft()
        (t_a, w_a), (t_b, w_b) = self.hist[0], self.hist[-1]
        rtf = (t_b - t_a) / (w_b - w_a) if w_b - w_a > 1e-6 else 0.0
        labels = ["sim time", "real time", "speed ratio"]
        values = [f"{t:7.2f} s", f"{wall:7.2f} s", f"{rtf:5.2f} x"]
        if mode:
            labels.append("mode")
            values.append(str(mode))
        unit = ("m/s", "m/s", "rad/s")
        for i, name in enumerate(("vx", "vy", "wz")):
            labels.append(f"{name} cmd | avg | now")
            line = f"{cmd[i]:+5.2f} | {avg[i]:+5.2f} | {now[i]:+5.2f} {unit[i]}"
            if abs(cmd[i]) > 1e-6:
                line += f"  ({100.0 * avg[i] / cmd[i]:4.0f} %)"
            values.append(line)
        for lab, val in extra:                                           # 추가 줄 (외란 상태 등, Q&A 9/29 Q11)
            labels.append(str(lab))
            values.append(str(val))
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          "\n".join(labels), "\n".join(values)))
