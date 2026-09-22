"""뷰어 화면 글자 (sim/실제 시간, 배속, 속도) — MPC·배포 RL 뷰어 공용 (Q&A 9/22 Q5)."""
import numpy as np
import mujoco


class ViewerHUD:
    """뷰어 왼쪽 위 글자: sim 시간 / 실제 시간 / 배속 / 명령·현재 속도 (Q&A 9/22 Q5).

    속도 = 전신 CoM 속도의 진행 방향(골반 yaw) 성분. 순간값과 0.8 s(한 걸음 주기) 평균.
    배속 = 최근 1 s 동안 sim 시간 / 실제 시간 (1.0 = 실시간).
    MuJoCo 내장 글꼴이 한글을 못 그려서 표시 글자는 영어.
    """

    def __init__(self, m, vx_cmd, every=25, avg_window=0.8):
        import collections, time
        self._time = time
        self.m = m
        self.vx_cmd = vx_cmd
        self.every = every
        self.pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.vbuf = collections.deque(maxlen=max(1, int(round(avg_window / (every * m.opt.timestep)))))
        self.hist = collections.deque()
        self.wall0 = time.perf_counter()

    def update(self, viewer, d, t, k, extra=None):
        if k % self.every:
            return
        mujoco.mj_subtreeVel(self.m, d)
        R = d.xmat[self.pel].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        vel = d.subtree_linvel[0]
        v_now = vel[0] * np.cos(yaw) + vel[1] * np.sin(yaw)
        self.vbuf.append(v_now)
        v_avg = float(np.mean(self.vbuf))
        wall = self._time.perf_counter() - self.wall0
        self.hist.append((t, wall))
        while len(self.hist) > 2 and wall - self.hist[0][1] > 1.0:
            self.hist.popleft()
        (t_a, w_a), (t_b, w_b) = self.hist[0], self.hist[-1]
        rtf = (t_b - t_a) / (w_b - w_a) if w_b - w_a > 1e-6 else 0.0
        labels = ["sim time", "real time", "speed ratio", "cmd vx", "speed (0.8 s avg)", "speed now"]
        values = [f"{t:7.2f} s", f"{wall:7.2f} s", f"{rtf:5.2f} x",
                  f"{self.vx_cmd:5.2f} m/s", f"{v_avg:5.2f} m/s", f"{v_now:5.2f} m/s"]
        if self.vx_cmd > 1e-6:
            labels.append("tracking"); values.append(f"{100 * v_avg / self.vx_cmd:4.0f} %")
        if extra:
            for a_, b_ in extra:
                labels.append(a_); values.append(b_)
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          "\n".join(labels), "\n".join(values)))
