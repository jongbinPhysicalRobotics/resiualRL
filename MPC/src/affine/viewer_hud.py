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


def cpu_bench():
    """고정 작업 시간 [ms] — CPU 가 느려졌는지 보는 지표 (20_timing_log 와 같은 작업)."""
    import time
    t0 = time.perf_counter()
    A = np.arange(200 * 108, dtype=float).reshape(200, 108) % 7.0
    for _ in range(20):
        (A.T * 1.1) @ A
    s_ = 0
    for i in range(20000):
        s_ += i
    return 1000 * (time.perf_counter() - t0)


def boost_process():
    """Windows: 프로세스 우선순위 '높음', 프로세스·현재 스레드의 전원 스로틀링(EcoQoS) 끔,
    현재 스레드 우선순위 '가장 높음'. 하이브리드 CPU 에서 Windows 가 계산 스레드를 효율 코어·
    낮은 클럭으로 돌리는 것을 막으려는 것 (Q&A 9/22 Q7). 성공한 항목 수를 돌려준다."""
    import sys
    if sys.platform != "win32":
        return 0
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetCurrentThread.restype = wintypes.HANDLE
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetThreadInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]

    class PT(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                    ("StateMask", wintypes.ULONG)]
    hp, ht = k32.GetCurrentProcess(), k32.GetCurrentThread()
    st = PT(1, 0x1, 0x0)                         # 실행 속도 스로틀링: 제어함 · 끔
    ok = [bool(k32.SetPriorityClass(hp, 0x00000080)),                                  # HIGH_PRIORITY_CLASS
          bool(k32.SetProcessInformation(hp, 4, ctypes.byref(st), ctypes.sizeof(st))),  # ProcessPowerThrottling
          bool(k32.SetThreadInformation(ht, 3, ctypes.byref(st), ctypes.sizeof(st))),   # ThreadPowerThrottling
          bool(k32.SetThreadPriority(ht, 2))]                                           # THREAD_PRIORITY_HIGHEST
    return sum(ok)
