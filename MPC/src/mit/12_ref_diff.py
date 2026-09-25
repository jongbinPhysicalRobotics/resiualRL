"""12_ref_diff.py — reference (C++) 실행 결과 ↔ Python 이식 (이 폴더) 차분 검증 (Q&A 9/25 Q4 계획 2~5 단계).

reference/ 는 외부 clone (우리 저장소 .gitignore) 이라 이 파일 하나로 재현한다 (프로젝트 루트에서):

    .venv/bin/python MPC/src/mit/12_ref_diff.py prep         # 모델 복사본 (이름만 reference 규칙) + 수치 동일 검사 + 패치 적용
    (cd reference && VCPKG_ROOT=$HOME/.local/vcpkg cmake --build --preset dev -j6)   # prep 이 패치를 새로 넣었으면
    .venv/bin/python MPC/src/mit/12_ref_diff.py survival --osqp-ref   # 생존·추종 표: reference vs Python (이식 설정 / OSQP ref-eff)
    .venv/bin/python MPC/src/mit/12_ref_diff.py ensemble --k 6        # 명령 × (1 + k·1e-9) 로 생존 시간 분포 (혼돈 분산)
    .venv/bin/python MPC/src/mit/12_ref_diff.py lockstep --case vx06 --seconds 1.6 --snap 0,0.3,0.7,1.0,1.05,1.1,1.15 --osqp ref
    (케이스: stand, inplace, vx06, vy03, wz13 = 명령 t = 0 부터 / vx06r, vy03r, wz13r = 1.0 s 부터 2 s 램프.
     --osqp ref = Python OSQP 를 reference 빌드 실효값으로 (adaptive_rho_interval 50, check_dualgap 1), port = 이식 기본 100 / 0)
    .venv/bin/python MPC/src/mit/12_ref_diff.py snapdiff reference/logs/verify_mit/lock_vx06   # 스냅샷만 격리 비교

무엇을 하나
  prep     : MPC/models/mit_humanoid → reference/models/mit_humanoid 복사, body/joint/actuator 이름만
             reference sim/src/models/MitHumanoidSpec.cpp 규칙으로 (base→torso, *_foot→*_foot_link,
             *_lower_arm→*_forearm_link, aNN_<side>_<j> → <side>_<j>_joint / 구동기 <side>_<j>). 순서·수치 그대로 —
             두 모델을 컴파일해 이름 외 배열 전부 비교. reference 소스에 아래 PATCH (env 로만 켜짐) 가 없으면 git apply.
  survival : reference / Python 을 같은 시나리오로 돌려 생존 시간·추종 (Python 10_walk 과 같은 정의).
  lockstep : 같은 초기 상태 (keyframe stand) · 같은 명령으로 두 쪽을 돌려
             (1) 스냅샷 시각마다 Python 모듈에 스냅샷 입력을 넣은 격리 비교 (I*), (2) 같은 틱의 Python 내부값과 비교 (L*),
             (3) 틱마다 궤적 비교 (reference headless telemetry csv, 소수 6 자리).
  snapdiff : 이미 있는 스냅샷 폴더에 대해 격리 비교 (I*) 만.

reference 쪽 설정 (D2 와 맞춤): keyframe 'stand' 에서 시작 (CONVEXMPC_INIT_KEYFRAME), leg_initialization_time 0.001
(yaml 사본) → 첫 틱 (t = 0) 부터 runController = Python 이식의 D2 와 같은 시각표 (StandingSettle 1.0 s → Walking).
명령: const = t = 0 부터 (Python profile const = reference 헤드리스 기본의 의미), ramp = t = 1.0 s 부터 |v|/2 per s.
출력: reference/logs/verify_mit/<run>/ (telemetry.csv, stdout.txt, controller.yaml, logs/debug/*_mpc/*.json, py_*.npz).

넘어짐 판정 (두 쪽 같은 식, telemetry 에서 가능한 것): |roll| > 0.8 또는 |pitch| > 0.8 rad 또는 축소 몸체 COM z < 0.46 m
(≈ Python 10_walk 의 torso z < 0.35 — COM 은 torso 원점보다 약 0.11 m 위). survival 표의 Python 칸은 10_walk.headless 그대로.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import numpy as np

import config
import paths
from command import CommandSchedule
from controller import MitController
from gait import GaitScheduler, HorizonClock, OverrideStep
from leg_control import arm_pd, standing_torque, stance_torque, swing_torque, walking_stance_torque
from mit_model import LEFT, RIGHT, MitModel, Rz, quat_to_roll_pitch
from mpc import ConvexMPC, RefOut, build_reference, continuous_matrices, discretize_zoh, lift, lift_angle_near

REF = paths.REFERENCE_DIR
BIN = REF / "build" / "apps" / "main"
REF_YAML = REF / "config" / "mit_humanoid" / "my_controller.yaml"
OUT = REF / "logs" / "verify_mit"
WALK_T = 1.0
DT = 0.002
FALL_ANGLE = 0.8
FALL_COM_Z = 0.46
MAX_PROCS = 6
# reference 빌드의 OSQP 1.0.0 이 실제로 쓰는 값 (OsqpEigen 0.11 은 5 개만 설정, 나머지 osqp_set_default_settings):
# adaptive_rho = 1 (ITERATIONS), adaptive_rho_interval = 50, check_dualgap = 1. Python 이식 기본은 100 / 0.
OSQP_REF_EFFECTIVE = {"adaptive_rho_interval": 50, "check_dualgap": 1}

CASES = {
    "stand": dict(profile="none", stand=True),
    "inplace": dict(profile="none"),
    "vx06": dict(vx=0.6, profile="const"),
    "vy03": dict(vy=0.3, profile="const"),
    "wz13": dict(wz=1.3, profile="const"),
    "vx06r": dict(vx=0.6, profile="ramp"),
    "vy03r": dict(vy=0.3, profile="ramp"),
    "wz13r": dict(wz=1.3, profile="ramp"),
}


def _case(name):
    c = dict(vx=0.0, vy=0.0, wz=0.0, profile="none", stand=False)
    c.update(CASES[name])
    return c


def _load_walk():
    spec = importlib.util.spec_from_file_location("walk10", str(paths.SRC / "10_walk.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =================================================================================================================
# prep: 모델 복사본 + 패치
# =================================================================================================================
JOINT_RENAME = {
    "a01_right_hip_yaw": "right_hip_yaw", "a02_right_hip_abad": "right_hip_abad", "a03_right_hip_pitch": "right_hip_pitch",
    "a04_right_knee": "right_knee", "a05_right_ankle": "right_ankle",
    "a06_left_hip_yaw": "left_hip_yaw", "a07_left_hip_abad": "left_hip_abad", "a08_left_hip_pitch": "left_hip_pitch",
    "a09_left_knee": "left_knee", "a10_left_ankle": "left_ankle",
    "a11_right_shoulder_pitch": "right_shoulder_pitch", "a12_right_shoulder_abad": "right_shoulder_abad",
    "a13_right_shoulder_yaw": "right_shoulder_yaw", "a14_right_elbow": "right_elbow",
    "a15_left_shoulder_pitch": "left_shoulder_pitch", "a16_left_shoulder_abad": "left_shoulder_abad",
    "a17_left_shoulder_yaw": "left_shoulder_yaw", "a18_left_elbow": "left_elbow",
}
BODY_RENAME = {"base": "torso", "left_foot": "left_foot_link", "right_foot": "right_foot_link",
               "left_lower_arm": "left_forearm_link", "right_lower_arm": "right_forearm_link"}

PATCH = r'''diff --git a/My_Controller/src/StandingMpcDebugLogger.cpp b/My_Controller/src/StandingMpcDebugLogger.cpp
index 43ebf02..a9ea9a7 100644
--- a/My_Controller/src/StandingMpcDebugLogger.cpp
+++ b/My_Controller/src/StandingMpcDebugLogger.cpp
@@ -3,6 +3,7 @@
 #include <cerrno>
 #include <chrono>
 #include <cmath>
+#include <cstdlib>
 #include <ctime>
 #include <fstream>
 #include <iomanip>
@@ -77,7 +78,8 @@ TimestampStrings makeTimestampStrings() {
     localtime_r(&time, &localTime);
 
     std::ostringstream filename;
-    filename << std::put_time(&localTime, "%Y%m%d_%H%M%S");
+    filename << std::put_time(&localTime, "%Y%m%d_%H%M%S")
+             << '_' << std::setw(6) << std::setfill('0') << micros;  // [residualRL verify patch 4] unique per microsecond
 
     std::ostringstream readable;
     readable << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S")
@@ -109,7 +111,11 @@ std::string debugLogDirectoryNameForMode(const LocomotionMode mode) {
 }
 
 std::string makeLogPath(const TimestampStrings& timestamp, const LocomotionMode locomotionMode) {
-    const std::string root(PROJECT_ROOT_DIR);
+    // [residualRL verify patch 4] optional log root override (default: PROJECT_ROOT_DIR, unchanged).
+    const char* rootOverride = std::getenv("CONVEXMPC_DEBUG_LOG_ROOT");
+    const std::string root = (rootOverride != nullptr && rootOverride[0] != '\0')
+                                 ? std::string(rootOverride)
+                                 : std::string(PROJECT_ROOT_DIR);
     const std::string logsDir = root + "/logs";
     const std::string debugDir = logsDir + "/debug";
     const std::string mpcDebugDir = debugDir + "/" + debugLogDirectoryNameForMode(locomotionMode);
diff --git a/robot/src/RobotConfig.cpp b/robot/src/RobotConfig.cpp
index dcc07fc..49912de 100644
--- a/robot/src/RobotConfig.cpp
+++ b/robot/src/RobotConfig.cpp
@@ -1,5 +1,6 @@
 #include "RobotConfig.h"
 
+#include <cstdlib>
 #include <map>
 #include <stdexcept>
 #include <string>
@@ -83,6 +84,11 @@ RobotType activeRobotType() {
 }
 
 std::string robotConfigPath(const RobotType robotType) {
+    // [residualRL verify patch 3] optional controller-yaml override (default: shipped per-robot yaml).
+    if (const char* overridePath = std::getenv("CONVEXMPC_CONFIG_PATH");
+        overridePath != nullptr && overridePath[0] != '\0') {
+        return resolveProjectPath(overridePath);
+    }
     return resolveProjectPath(robotConfigRelativePath(robotType));
 }
 
diff --git a/sim/src/SimulationRunner.cpp b/sim/src/SimulationRunner.cpp
index 1f2b46e..543b9f1 100644
--- a/sim/src/SimulationRunner.cpp
+++ b/sim/src/SimulationRunner.cpp
@@ -265,6 +265,17 @@ void SimulationRunner::init() {
 		throw std::runtime_error("mj_makeData failed");
 	}
 
+	// [residualRL verify patch 1] optional start from a named keyframe (default: model qpos0, unchanged).
+	if (const char* keyEnv = std::getenv("CONVEXMPC_INIT_KEYFRAME");
+	    keyEnv != nullptr && std::string(keyEnv).size() > 0) {
+		const int keyId = mj_name2id(model, mjOBJ_KEY, keyEnv);
+		if (keyId < 0) {
+			throw std::runtime_error(std::string("CONVEXMPC_INIT_KEYFRAME not found: ") + keyEnv);
+		}
+		mj_resetDataKeyframe(model, data, keyId);
+		std::cout << "[verify] reset to keyframe '" << keyEnv << "'\n";
+	}
+
 	configureSimulationModel(model);
 	mj_forward(model, data);
 	_keyboardInputEnableTime =
@@ -523,6 +534,35 @@ void SimulationRunner::applyHeadlessUserCommandSchedule(const double simTime) {
 		_userCommand.x_dot = rampedXDot;
 	}
 
+	// [residualRL verify patch 2] y_dot / psi_dot schedule with the same start time and ramp formula
+	// as x_dot (unset env = unchanged behavior).
+	const auto applyAxisSchedule = [&](const char* finalName, const char* rateName, double& target) {
+		const char* axisFinalEnv = std::getenv(finalName);
+		if (axisFinalEnv == nullptr || std::string(axisFinalEnv).empty()) {
+			return;
+		}
+		const double axisFinal = std::atof(axisFinalEnv);
+		if (!std::isfinite(axisFinal)) {
+			return;
+		}
+		double axisRate = 0.0;
+		if (const char* axisRateEnv = std::getenv(rateName);
+		    axisRateEnv != nullptr && std::string(axisRateEnv).size() > 0) {
+			const double parsed = std::atof(axisRateEnv);
+			if (std::isfinite(parsed) && parsed > 0.0) {
+				axisRate = parsed;
+			}
+		}
+		if (simTime >= startTime) {
+			target = axisRate > 0.0
+			             ? std::copysign(std::min(std::abs(axisFinal), axisRate * (simTime - startTime)),
+			                             axisFinal)
+			             : axisFinal;
+		}
+	};
+	applyAxisSchedule("CONVEXMPC_HEADLESS_Y_DOT_FINAL", "CONVEXMPC_HEADLESS_Y_DOT_RATE", _userCommand.y_dot);
+	applyAxisSchedule("CONVEXMPC_HEADLESS_PSI_DOT_FINAL", "CONVEXMPC_HEADLESS_PSI_DOT_RATE", _userCommand.psi_dot);
+
 	if (!_headlessScheduleAnnounced) {
 		std::cout << "[HeadlessCommandSchedule] auto walking enabled, x_dot final="
 		          << finalXDot << " m/s, rate=" << rate
'''


def prep():
    src, dst = paths.MIT_DIR, REF / "models" / "mit_humanoid"
    x = (src / "mit_humanoid.xml").read_text()
    for old, new in BODY_RENAME.items():
        x, k = re.subn(rf'<body name="{old}"', f'<body name="{new}"', x)
        assert k == 1, old
    for old, new in JOINT_RENAME.items():
        x, k = re.subn(rf'<joint name="{old}"', f'<joint name="{new}_joint"', x)
        assert k == 1, old
        x, k = re.subn(rf'<motor name="{old}" joint="{old}"', f'<motor name="{new}" joint="{new}_joint"', x)
        assert k == 1, old
    x = x.replace('<mujoco model="mit_humanoid">',
                  '<mujoco model="mit_humanoid">\n  <!-- 2026-09-25 verification copy of residualRL MPC/models/mit_humanoid/'
                  'mit_humanoid.xml (MPC/src/mit/12_ref_diff.py prep): only body/joint/actuator NAMES changed to reference '
                  'sim/src/models/MitHumanoidSpec.cpp. Order, numbers, sites, keyframes identical. -->', 1)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "mit_humanoid.xml").write_text(x)
    shutil.copy2(src / "scene.xml", dst / "scene.xml")
    if (dst / "meshes_v3").exists():
        shutil.rmtree(dst / "meshes_v3")
    shutil.copytree(src / "meshes_v3", dst / "meshes_v3")
    a = mujoco.MjModel.from_xml_path(str(src / "scene.xml"))
    b = mujoco.MjModel.from_xml_path(str(dst / "scene.xml"))
    n, bad = 0, []
    for name in dir(a):
        if name.startswith("_") or name.startswith("name") or name in ("names", "paths", "text_data", "signature"):
            continue
        va = getattr(a, name, None)
        if isinstance(va, np.ndarray):
            n += 1
            vb = getattr(b, name)
            if va.shape != vb.shape or not np.array_equal(va, vb, equal_nan=True):
                bad.append(name)
    print(f"[prep] model copy -> {dst}")
    print(f"[prep] compiled-model arrays compared {n}, differing {len(bad)} {bad if bad else ''}")
    marker = "[residualRL verify patch 1]"
    if marker in (REF / "sim" / "src" / "SimulationRunner.cpp").read_text():
        print("[prep] reference verify patches already present")
    else:
        pf = OUT / "verify.patch"
        OUT.mkdir(parents=True, exist_ok=True)
        pf.write_text(PATCH)
        subprocess.run(["git", "apply", str(pf)], cwd=str(REF), check=True)
        print("[prep] applied verify patch — rebuild: (cd reference && VCPKG_ROOT=$HOME/.local/vcpkg "
              "cmake --build --preset dev -j6)")


# =================================================================================================================
# reference 실행
# =================================================================================================================
def write_yaml(path: Path, triggers=(), mode="walking", leg_init=0.001):
    s = REF_YAML.read_text()
    trig = ", ".join(repr(float(t)) for t in triggers)
    for a, b in (("  leg_initialization_time: 2.0", f"  leg_initialization_time: {leg_init}"),
                 ("  standing_mpc_debug_trigger_times: []", f"  standing_mpc_debug_trigger_times: [{trig}]"),
                 ("requested_locomotion_mode: walking", f"requested_locomotion_mode: {mode}")):
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    path.write_text(s)


def ref_env(case, run_dir: Path, total: float, yaml_path: Path, tel_dt: float, keyframe="stand"):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CONVEXMPC_")}
    env.update(CONVEXMPC_CONFIG_PATH=str(yaml_path), CONVEXMPC_DEBUG_LOG_ROOT=str(run_dir),
               CONVEXMPC_HEADLESS_STOP_TIME=repr(float(total)),
               CONVEXMPC_HEADLESS_TELEMETRY=str(run_dir / "telemetry.csv"),
               CONVEXMPC_HEADLESS_TELEMETRY_DT=repr(float(tel_dt)))
    if keyframe:
        env["CONVEXMPC_INIT_KEYFRAME"] = keyframe
    prof = case["profile"]
    if prof in ("const", "ramp"):
        env["CONVEXMPC_HEADLESS_AUTO_WALK"] = "1"
        for ax, v in (("X", case["vx"]), ("Y", case["vy"]), ("PSI", case["wz"])):
            env[f"CONVEXMPC_HEADLESS_{ax}_DOT_FINAL"] = repr(float(v))
            if prof == "ramp" and v != 0.0:
                env[f"CONVEXMPC_HEADLESS_{ax}_DOT_RATE"] = repr(abs(float(v)) / 2.0)
        env["CONVEXMPC_HEADLESS_X_DOT_START_TIME"] = "0" if prof == "const" else repr(WALK_T)
    return env


def read_telemetry(path: Path) -> dict:
    with open(path) as f:
        header = f.readline().strip().split(",")
        rows = []
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != len(header):
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    A = np.array(rows, dtype=float).reshape(-1, len(header))
    return {h: A[:, i] for i, h in enumerate(header)}


def _fall_mask(com_z, roll, pitch):
    return (np.abs(roll) > FALL_ANGLE) | (np.abs(pitch) > FALL_ANGLE) | (com_z < FALL_COM_Z)


def run_ref(case, total, run_dir: Path, triggers=(), tel_dt=0.01, kill_on_fall=True, fall_margin=0.5):
    run_dir.mkdir(parents=True, exist_ok=True)
    for old in run_dir.glob("logs/debug/*/*.json"):
        old.unlink()
    yaml_path = run_dir / "controller.yaml"
    write_yaml(yaml_path, triggers, mode="standing" if case["stand"] else "walking")
    env = ref_env(case, run_dir, total, yaml_path, tel_dt)
    tel = run_dir / "telemetry.csv"
    if tel.exists():
        tel.unlink()
    t_wall = time.perf_counter()
    with open(run_dir / "stdout.txt", "w") as log:
        p = subprocess.Popen([str(BIN), "m", "n"], cwd=str(REF), env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT)
        cols, pos, t_fall, t_last = None, 0, None, 0.0
        while p.poll() is None:
            time.sleep(0.5)
            if not tel.exists():
                continue
            with open(tel) as f:
                f.seek(pos)
                chunk = f.read()
            lines = chunk.split("\n")
            pos += len(chunk) - len(lines[-1])                      # 마지막 (불완전할 수 있는) 줄은 다음에
            for line in lines[:-1]:
                parts = line.split(",")
                if cols is None:
                    cols = {h: i for i, h in enumerate(parts)}
                    continue
                try:
                    tt = float(parts[cols["time"]])
                    cz, r, pi = (float(parts[cols[k]]) for k in ("com_z", "roll", "pitch"))
                except (ValueError, IndexError, KeyError):
                    continue
                t_last = tt
                if t_fall is None and _fall_mask(np.array([cz]), np.array([r]), np.array([pi]))[0]:
                    t_fall = tt
            if kill_on_fall and t_fall is not None and t_last > t_fall + fall_margin:
                p.terminate()
                p.wait()
                break
    wall = time.perf_counter() - t_wall
    T = read_telemetry(tel)
    fm = _fall_mask(T["com_z"], T["roll"], T["pitch"])
    t_fall = float(T["time"][np.argmax(fm)]) if fm.any() else None
    out = (run_dir / "stdout.txt").read_text(errors="replace")
    n_fail = out.count("maybeUpdateMpc failed")
    return dict(T=T, t_fall=t_fall, wall=wall, n_mpc_fail=n_fail, t_end=float(T["time"][-1]) if len(T["time"]) else 0.0)


def ref_metrics(T, case, t_fall, window=100.0):
    """10_walk.compute_metrics 와 같은 정의 (창: 마지막 100 s, walking 시작 + 5 s 이후). 속도 = 몸통 COM 속도 (telemetry)."""
    t = T["time"]
    walk_t = 0.0 if case["stand"] else WALK_T
    t_end = float(t[-1])
    if t_fall is not None:
        t_end = t_fall
    res = dict(survived=max(0.0, t_end - walk_t))
    w0 = max(t_end - window, walk_t + 5.0)
    win = (t >= w0) & (t <= t_end)
    if win.sum() < 10:
        return res
    yaw = T["yaw"]
    c, s = np.cos(yaw), np.sin(yaw)
    vh_x = c * T["vx"] + s * T["vy"]
    vh_y = -s * T["vx"] + c * T["vy"]
    iw = np.nonzero(win)[0]
    res["vx"] = float(np.mean(vh_x[win]))
    res["vy"] = float(np.mean(vh_y[win]))
    res["wz"] = float((yaw[iw[-1]] - yaw[iw[0]]) / max(t[iw[-1]] - t[iw[0]], 1e-9))
    trk = []
    for key, cmd in (("vx", case["vx"]), ("vy", case["vy"]), ("wz", case["wz"])):
        if abs(cmd) > 1e-9:
            trk.append(100.0 * res[key] / cmd)
    res["track"] = float(np.mean(trk)) if trk else float("nan")
    res["roll_sd"] = float(np.std(T["roll"][win]))
    res["pitch_sd"] = float(np.std(T["pitch"][win]))
    return res


# =================================================================================================================
# Python 실행 (lockstep: 틱 기록 + 스냅샷 틱의 내부값 포착)
# =================================================================================================================
def py_schedule(case, walk):
    if case["profile"] == "none":
        return CommandSchedule.constant(0.0, 0.0, 0.0)
    return walk.make_schedule(case["profile"], case["vx"], case["vy"], case["wz"], WALK_T)


def run_py(case, total, capture_iters=(), osqp_settings=None, fall_margin=0.5):
    walk = _load_walk()
    cfg = config.load()
    model = MitModel()
    model.reset(str(cfg.port.spawn_keyframe))
    m, d = model.m, model.d
    ctl = MitController(model, cfg, py_schedule(case, walk), requested_mode="standing" if case["stand"] else None,
                        osqp_settings=osqp_settings)
    capture_iters = set(int(i) for i in capture_iters)
    caps, cur, flag = {}, {}, {"on": False}
    orig_solve, orig_form = ConvexMPC.solve, ConvexMPC.formulate

    def form_hook(self_, ref, reduced):
        res = orig_form(self_, ref, reduced)
        if flag["on"]:
            cur["A_qp"], cur["B_qp"] = res[0].copy(), res[1].copy()
        return res

    def solve_hook(self_, x0, ref, steps, foot_yaw, reduced, mode, active=None):
        it = ctl.iteration
        flag["on"] = it in capture_iters
        if flag["on"]:
            cur.clear()
            cur.update(iteration=it, x0=np.array(x0, float), X_ref=ref.X_ref.copy(), psi=ref.psi.copy(),
                       r=ref.r.copy(), tk=ref.tk.copy(), stance=np.array([s.stance for s in steps]),
                       min_scale=np.array([s.min_scale for s in steps]), foot_yaw=np.array(foot_yaw, float),
                       mass=float(reduced.mass), com_offset_B=reduced.com_offset_B.copy(),
                       inertia_B=reduced.inertia_B.copy(), mode=str(mode))
        u0, info = orig_solve(self_, x0, ref, steps, foot_yaw, reduced, mode, active)
        if flag["on"]:
            cur.update(u0=np.array(u0, float), U=np.array(info["U"], float), cold=bool(info.get("cold")),
                       iters=int(info.get("iters", 0)), status=str(info.get("status")),
                       fallback=bool(info.get("fallback")))
        flag["on"] = False
        return u0, info

    ConvexMPC.solve, ConvexMPC.formulate = solve_hook, form_hook
    n = int(round(total / m.opt.timestep))
    keys = ("t", "com", "rpy", "vel", "angvel", "foot", "fn", "tau_norm", "des_f", "walking", "u", "tau")
    R = {k: [] for k in keys}
    b = model.base_body
    t_fall = None
    n_fallback = 0
    try:
        for _ in range(n):
            qpos0, qvel0 = d.qpos.copy(), d.qvel.copy()
            xpos_b, xquat_b = d.xpos[b].copy(), d.xquat[b].copy()
            tau_leg, tau_arm = ctl.tick()
            tau_full = model.write_torque(tau_leg, tau_arm)
            s, x0 = ctl.state, ctl.x0
            info = ctl.tick_info.get("info")
            if info is not None and info.get("fallback"):
                n_fallback += 1
            walking = ctl.mode == "walking"
            des = np.zeros((2, 3))
            if walking:
                for leg in (LEFT, RIGHT):
                    if ctl._active(leg, s.t):
                        seg = ctl.u_hold[0:3] if leg == LEFT else ctl.u_hold[3:6]
                        des[leg] = ctl._alpha(leg) * seg
            R["t"].append(s.t)
            R["com"].append(x0[3:6].copy())
            R["rpy"].append((s.roll, s.pitch, s.yaw_unwrapped))
            R["vel"].append(s.torso_vel_W.copy())
            R["angvel"].append(s.torso_angvel_W.copy())
            R["foot"].append(s.foot_pos_W.copy())
            R["fn"].append(s.foot_normal_force.copy())
            R["tau_norm"].append(float(np.linalg.norm(tau_full)))
            R["des_f"].append(des)
            R["walking"].append(walking)
            R["u"].append(ctl.u_hold.copy())
            R["tau"].append(tau_full.copy())
            if cur and cur.get("iteration") == ctl.iteration - 1 and "u0" in cur:
                cm = ctl.cm
                cur.update(t=s.t, qpos=qpos0, qvel=qvel0, xpos=xpos_b, xquat=xquat_b,
                           torso_pos=s.torso_pos_W.copy(), torso_quat=s.torso_quat_W.copy(),
                           torso_vel=s.torso_vel_W.copy(), torso_angvel=s.torso_angvel_W.copy(),
                           yaw_u=s.yaw_unwrapped, foot_pos=s.foot_pos_W.copy(), foot_vel=s.foot_vel_W.copy(),
                           foot_R=s.foot_R_W.copy(), fn=s.foot_normal_force.copy(), tau_full=tau_full.copy(),
                           desired=ctl.desired.copy(), t0=ctl.clock.t0, mode_now=ctl.mode,
                           filtered=np.array([ctl.filtered.x_dot, ctl.filtered.y_dot, ctl.filtered.psi_dot]),
                           active=np.array(cm.active_contact, bool).copy(), alpha=np.array(cm.ramp_alpha, float).copy(),
                           est=np.array(cm.estimated, bool).copy(),
                           td_yaw=np.array([ctl.legs[l].touchdown_yaw for l in (LEFT, RIGHT)]),
                           body_euler=ctl.body.euler_W.copy(), nominal_height=ctl.body.nominal_height_W)
                caps[cur["iteration"]] = dict(cur)
                cur.clear()
            mujoco.mj_step(m, d)
            if t_fall is None and _fall_mask(np.array([x0[5]]), np.array([s.roll]), np.array([s.pitch]))[0]:
                t_fall = s.t
            if t_fall is not None and s.t > t_fall + fall_margin:
                break
    finally:
        ConvexMPC.solve, ConvexMPC.formulate = orig_solve, orig_form
    out = {k: np.array(v) for k, v in R.items()}
    return dict(R=out, caps=caps, t_fall=t_fall, n_fallback=n_fallback, model=model, cfg=cfg)


def run_py_survival(case, seconds, osqp_settings=None):
    walk = _load_walk()
    prof = "const" if case["profile"] == "none" else case["profile"]
    r = walk.headless(case["vx"], case["vy"], case["wz"], seconds, profile=prof, stand=case["stand"],
                      verbose=False, print_every=1e9, osqp_settings=osqp_settings)
    return {k: r.get(k) for k in ("survived", "fell", "track", "vx", "vy", "wz", "roll_sd", "pitch_sd",
                                   "fallbacks", "error", "wall", "solves", "cold_rate")}


# =================================================================================================================
# 스냅샷
# =================================================================================================================
def jarr(o):
    if isinstance(o, dict) and "data" in o:
        return np.array(o["data"], dtype=float)
    return np.array(o, dtype=float)


def load_snapshots(run_dir: Path):
    files = sorted(run_dir.glob("logs/debug/*_mpc/*.json"))
    snaps = []
    for f in files:
        with open(f) as fh:
            snaps.append((f, json.load(fh)))
    snaps.sort(key=lambda fs: fs[1]["metadata"]["controller_iteration"])
    return snaps


def snap_fields(S):
    md = S["metadata"]
    legs = S["feet"]["legs"]
    cm = S["contact_manager_state"]["legs"]
    rt = S["reference_trajectory"]
    fo = S["formulation"]
    so = S["solution"]
    N = int(md["horizon_steps"])
    r = np.zeros((N, 2, 3))
    r[:, 0] = jarr(rt["r_left"]).T
    r[:, 1] = jarr(rt["r_right"]).T
    return SimpleNamespace(
        it=int(md["controller_iteration"]), t=float(md["controller_time"]), t0=float(md["horizon_clock_t0"]),
        mode=md["locomotion_mode"], state=md["locomotion_state"],
        cmd=S["user_command"], x0=jarr(S["initial_state"]["x0"]), yaw_u=float(S["initial_state"]["yaw_W_unwrapped"]),
        torso_pos=jarr(S["initial_state"]["torso_pos_W"]), torso_vel=jarr(S["initial_state"]["torso_lin_vel_W"]),
        torso_angvel=jarr(S["initial_state"]["torso_ang_vel_W"]),
        qpos=jarr(S["robot_state"]["full_qpos"]), qvel=jarr(S["robot_state"]["full_qvel"]),
        tau=jarr(S["robot_state"]["full_tau_command"]),
        mass=float(S["model"]["mass"]), com_loc=jarr(S["model"]["body_com_location_yaw_frame"]),
        inertia=jarr(S["model"]["body_inertia_yaw_frame"]),
        desired=np.array([jarr(S["feet"]["desired_foot_pos_W"]["left"]), jarr(S["feet"]["desired_foot_pos_W"]["right"])]),
        legs=legs, cm=cm,
        foot_pos=np.array([jarr(l["foot_pos_W"]) for l in legs]), foot_vel=np.array([jarr(l["foot_vel_W"]) for l in legs]),
        foot_R=np.array([jarr(l["R_WF"]) for l in legs]), q_leg=np.array([jarr(l["q"]) for l in legs]),
        qd_leg=np.array([jarr(l["qd"]) for l in legs]),
        td_yaw=np.array([float(l["touchdown_yaw_W"]) if l["touchdown_yaw_W"] is not None else np.nan for l in legs]),
        active=np.array([bool(c["active_contact"]) for c in cm]), alpha=np.array([float(c["contact_ramp_alpha"]) for c in cm]),
        est=np.array([bool(c["estimated_contact"]) for c in cm]),
        tk=jarr(rt["tk"]), psi=jarr(rt["psi"]), r=r, X_ref=jarr(rt["X_ref_by_step"]),
        A_c=jarr(fo["A_c"]), B_c=jarr(fo["B_c"]), I_w=jarr(fo["inertia_world"]), A_qp=jarr(fo["A_qp"]), B_qp=jarr(fo["B_qp"]),
        U=jarr(so["wrench_horizon"]), u0=jarr(so["first_wrench"]),
        standing_J=S["standing_wrench_to_torque"],
    )


def _mx(a):
    a = np.asarray(a, dtype=float)
    return float(np.nanmax(np.abs(a))) if a.size else 0.0


class Table:
    def __init__(self):
        self.rows = []

    def add(self, key, err, scale=None, tol=None, note=""):
        self.rows.append((key, float(err), None if scale is None else float(scale), tol, note))

    def print(self, title):
        print(f"\n  --- {title}")
        for key, err, scale, tol, note in self.rows:
            rel = f"  rel {err / scale:.1e}" if scale not in (None, 0.0) else ""
            ok = "" if tol is None else ("  OK" if err <= tol else "  **DIFF**")
            print(f"    {key:<46s} max|Δ| {err:10.3e}{rel}{ok}  {note}")


def isolated_checks(sf, model, cfg):
    """I*: 스냅샷 입력을 Python 모듈에 넣어 스냅샷 출력과 비교 (Python 궤적과 무관)."""
    tb = Table()
    walking = sf.mode == "walking"
    # I1 x0 재계산 (MC:656-675)
    q = sf.qpos[3:7]
    roll, pitch = quat_to_roll_pitch(q)
    off = Rz(sf.yaw_u) @ sf.com_loc
    x0 = np.concatenate([[roll, pitch, sf.yaw_u], sf.torso_pos + off, sf.torso_angvel,
                         sf.torso_vel + np.cross(sf.torso_angvel, off), [-9.81]])
    tb.add("I1 x0 (state fields -> x0 formula)", _mx(x0 - sf.x0), _mx(sf.x0), 1e-12)
    # I2 축소 몸체: full_qpos (torso = stale xpos/xquat, 관절 = fresh) 로 kinematics → 근사 (팔 q 가 1 틱 새것)
    d = model.d
    d.qpos[:] = sf.qpos
    d.qvel[:] = 0.0
    mujoco.mj_kinematics(model.m, d)
    red = model.reduced_body()
    tb.add("I2 reduced mass", abs(red.mass - sf.mass), sf.mass, 1e-12)
    tb.add("I2 reduced com_offset (approx: arm q 1 tick newer)", _mx(red.com_offset_B - sf.com_loc), _mx(sf.com_loc))
    tb.add("I2 reduced inertia (approx: arm q 1 tick newer)", _mx(red.inertia_B - sf.inertia), _mx(sf.inertia))
    # I3 참조 궤적 (seed 의 roll/pitch/z 는 X_ref[0] 에서 — 그 3 값은 순환, 나머지는 독립)
    cmd = SimpleNamespace(**sf.cmd)
    seed = sf.x0.copy()
    if walking:
        seed[0], seed[1], seed[5] = sf.X_ref[0, 0], sf.X_ref[0, 1], sf.X_ref[0, 5] - cmd.body_height_offset_m
    else:
        seed[0:3] = sf.X_ref[0, 0:3]
        seed[3:6] = sf.X_ref[0, 3:6]
        seed[5] = sf.X_ref[0, 5] - cmd.body_height_offset_m
    clock = HorizonClock(sf.t0, cfg)
    ref = build_reference(cmd, seed, sf.desired, clock, cfg)
    tb.add("I3 X_ref (25x13)", _mx(ref.X_ref - sf.X_ref), _mx(sf.X_ref), 1e-12)
    tb.add("I3 psi_k", _mx(ref.psi - sf.psi), None, 1e-12)
    tb.add("I3 r_left/right (desired - p_ref)", _mx(ref.r - sf.r), _mx(sf.r), 1e-12)
    tb.add("I3 tk", _mx(ref.tk - sf.tk), None, 1e-12)
    # I4 formulation (스냅샷 psi, r, 관성, 질량 → A_c, B_c, A_qp, B_qp)
    A_c, B_c, I_k = continuous_matrices(sf.psi, sf.r, sf.inertia, sf.mass)
    A_d, B_d = discretize_zoh(A_c, B_c, 0.02)
    A_qp, B_qp = lift(A_d, B_d)
    tb.add("I4 inertia_world per step", _mx(I_k - sf.I_w), _mx(sf.I_w), 1e-12)
    tb.add("I4 A_c", _mx(A_c - sf.A_c), _mx(sf.A_c), 1e-12)
    tb.add("I4 B_c", _mx(B_c - sf.B_c), _mx(sf.B_c), 1e-9)
    tb.add("I4 A_qp (325x13)", _mx(A_qp - sf.A_qp), _mx(sf.A_qp), 1e-9)
    tb.add("I4 B_qp (325x300)", _mx(B_qp - sf.B_qp), _mx(sf.B_qp), 1e-9)
    # I5 제약: Python 접촉표 (t0, override = CM 상태) + 발 yaw → 스냅샷 해 U 의 제약 위반
    gait = GaitScheduler(cfg, clock)
    gait.set_mode(sf.mode)
    active = sf.active if walking else np.array([True, True])
    override = ([OverrideStep(enabled=True, stance=active.copy(),
                              min_scale=np.array([sf.alpha[l] if active[l] else 0.0 for l in (LEFT, RIGHT)]))]
                if walking else None)
    steps = gait.horizon_steps(clock, override)
    foot_yaw = np.zeros(2)
    for leg in (LEFT, RIGHT):
        xa = sf.foot_R[leg][:, 0]
        foot_yaw[leg] = lift_angle_near(float(np.arctan2(xa[1], xa[0])), sf.td_yaw[leg]) if active[leg] else sf.td_yaw[leg]
    mpc = ConvexMPC(cfg, verbose_errors=False)
    blk, l, u, sig, stance = mpc.build_constraints(steps, foot_yaw)
    N = mpc.N
    Cu = np.einsum("krc,kc->kr", blk, sf.U)
    ub = u[:24 * N].reshape(N, 24)
    viol = np.maximum(Cu[:, :24] - ub, 0.0)
    eq = np.abs(Cu[:, 24:])
    tb.add("I5 ineq violation of ref U (friction/CoP/Fz) [N]", _mx(viol), None, None,
           "OSQP 해 자체의 부정확 (eps 1e-3·scale, 200 회) — 비교 기준 아님")
    tb.add("I5 swing-foot wrench of ref U (eq rows)", _mx(eq), None, None)
    # I6 QP: 스냅샷 A_qp/B_qp 로 Python 비용 → 정확 해 (quadprog) vs 스냅샷 해 (OSQP, warm)
    X_flat = sf.X_ref.reshape(-1)
    H, qv = mpc.build_cost(sf.A_qp, sf.B_qp, sf.x0, X_flat, sf.mode)
    qp = ConvexMPC(cfg, solver="quadprog", verbose_errors=False)
    info = {}
    U_ex = qp._solve_quadprog(H, qv, blk, u, sig, info)
    J = lambda U: float(0.5 * U @ H @ U + qv @ U)   # noqa: E731
    U_ref = sf.U.reshape(-1)
    Jex, Jref = J(U_ex), J(U_ref)
    tb.add("I6 u0 ref(OSQP) - exact QP: forces [N]", _mx(sf.u0[:6] - U_ex[:6]), _mx(U_ex[:6]), None)
    tb.add("I6 u0 ref(OSQP) - exact QP: moments [Nm]", _mx(sf.u0[6:12] - U_ex[6:12]), _mx(U_ex[6:12]), None)
    tb.add("I6 objective gap (J_ref - J*)/|J*|", (Jref - Jex) / abs(Jex), None, None)
    ref_obj = RefOut(X_ref=sf.X_ref, psi=sf.psi, r=sf.r, tk=sf.tk)
    red_obj = SimpleNamespace(mass=sf.mass, inertia_B=sf.inertia)
    for tag, st in (("port", None), ("ref-eff", OSQP_REF_EFFECTIVE)):
        cold = ConvexMPC(cfg, osqp_settings=st, verbose_errors=False)
        u_c, inf_c = cold.solve(sf.x0, ref_obj, steps, foot_yaw, red_obj, sf.mode, active=active)
        Uc = inf_c["U"].reshape(-1)
        tb.add(f"I6 u0 py-OSQP cold ({tag}) - exact: forces", _mx(u_c[:6] - U_ex[:6]), None, None,
               f"iters {inf_c.get('iters')} {inf_c.get('status')} gap {(J(Uc) - Jex) / abs(Jex):.1e}")
        tb.add(f"I6 u0 py-OSQP cold ({tag}) - ref u0 (meaningful if ref solve was cold)", _mx(u_c - sf.u0), None, None)
    # 같은 OSQP 설정 (ref-eff), 스냅샷 A_qp/B_qp 를 그대로 써서 cold 풀이 → OSQP 1.1.3 (py) vs 1.0.0 (C++) 만 남김
    m2 = ConvexMPC(cfg, osqp_settings=OSQP_REF_EFFECTIVE, verbose_errors=False)
    H2, q2 = m2.build_cost(sf.A_qp, sf.B_qp, sf.x0, X_flat, sf.mode)
    inf2 = {}
    try:
        U2 = m2._solve_osqp(H2[m2._P_rows, m2._P_cols], q2, blk.reshape(-1)[m2._A_gather], l, u, sig, inf2)
        tb.add("I6 u0 py-OSQP cold (ref-eff, snapshot A_qp/B_qp) - ref u0", _mx(U2[:12] - sf.u0), None, None,
               f"iters {inf2.get('iters')} {inf2.get('status')}")
    except Exception as exc:                       # MaxIter 두 번 → reference 라면 fallback (MC:1016-1052)
        tb.add("I6 u0 py-OSQP cold (ref-eff, snapshot A_qp/B_qp) - ref u0", float("nan"), None, None, f"failed: {exc}")
    # I7 토크
    tau_ref = sf.tau
    d.qpos[:] = sf.qpos
    d.qvel[:] = 0.0
    d.qvel[model.leg_dof.ravel()] = sf.qd_leg.ravel()
    d.time = -1.0 - sf.t
    mujoco.mj_kinematics(model.m, d)
    model._dyn_key = None
    st = SimpleNamespace(foot_pos_W=sf.foot_pos, foot_vel_W=sf.foot_vel, foot_R_W=sf.foot_R, qd_leg=sf.qd_leg,
                         q_arm=sf.qpos[model.arm_qadr], qd_arm=sf.qvel[model.arm_dof])
    q_arm_des = model.m.qpos0[model.arm_qadr] + np.asarray(cfg.initial_pose.arm_joint_offsets, float)[None, :]
    tau_arm = arm_pd(st, q_arm_des, cfg)
    tb.add("I7 arm PD torque (8)", _mx(tau_arm.ravel() - tau_ref[model.arm_act.ravel()]), _mx(tau_ref[model.arm_act]), 1e-9)
    if walking:
        for leg, nm in ((LEFT, "L"), (RIGHT, "R")):
            L = sf.legs[leg]
            assert list(L["q_indices"]) == list(model.leg_qadr[leg]), "leg index mismatch"
            act = np.array(L["actuator_indices"], int)
            dyn = model.leg_dynamics(leg)
            Jv_s, Jw_s = jarr(L["Jv_W"]), jarr(L["Jw_W"])
            tb.add(f"I7 {nm} Jv (py scratch vs ref aux)", _mx(dyn.Jv - Jv_s), _mx(Jv_s), 1e-9)
            tb.add(f"I7 {nm} Jw (py scratch vs ref aux)", _mx(dyn.Jw - Jw_s), _mx(Jw_s), 1e-9)
            mode_s = L["command_mode"]
            if mode_s == "stance_wrench":
                F, M = jarr(L["force_feedforward_W"]), jarr(L["moment_feedforward_W"])
                t_formula = stance_torque(SimpleNamespace(Jv=Jv_s, Jw=Jw_s), F, M)
                tb.add(f"I7 {nm} stance tau = JvT F + JwT M (ref J, ref FF)", _mx(t_formula - tau_ref[act]), _mx(tau_ref[act]), 1e-9)
                t_py = walking_stance_torque(dyn, st, leg, sf.u0, sf.alpha[leg], sf.td_yaw[leg], cfg)
                tb.add(f"I7 {nm} stance tau (py full: -a*u0 + yaw hold)", _mx(t_py - tau_ref[act]), _mx(tau_ref[act]), 1e-7,
                       f"alpha {sf.alpha[leg]:.2f}")
            elif mode_s == "swing_foot":
                t_py = swing_torque(dyn, st, leg, jarr(L["p_des_W"]), jarr(L["v_des_W"]), jarr(L["a_des_W"]),
                                    sf.td_yaw[leg], cfg)
                tb.add(f"I7 {nm} swing tau (py OSC+attitude, py M/bias)", _mx(t_py - tau_ref[act]), _mx(tau_ref[act]), 1e-7)
            else:
                tb.add(f"I7 {nm} mode {mode_s} (not compared)", 0.0)
    else:
        Jv, Jw = model.standing_jacobians()
        SJ = sf.standing_J
        if "standing_Jv_W" in SJ:
            tb.add("I7 standing Jv 6x10 (py vs ref aux)", _mx(Jv - jarr(SJ["standing_Jv_W"])), 1.0, 1e-9)
            tb.add("I7 standing Jw 6x10 (py vs ref aux)", _mx(Jw - jarr(SJ["standing_Jw_W"])), 1.0, 1e-9)
        t_py = standing_torque(Jv, Jw, sf.u0)
        tau_leg_ref = np.array([tau_ref[np.array(sf.legs[l]["actuator_indices"], int)] for l in (LEFT, RIGHT)])
        tb.add("I7 standing tau (py J, ref u0)", _mx(t_py - tau_leg_ref), _mx(tau_leg_ref), 1e-7)
    return tb, dict(U_ex=U_ex, steps=steps, foot_yaw=foot_yaw)


def lockstep_checks(sf, cap, model):
    """L*: 같은 틱의 Python 내부값 (같은 초기 상태·명령으로 돌린 Python) 과 스냅샷."""
    tb = Table()
    b = model.base_body
    tb.add("L0 time", abs(cap["t"] - sf.t), None, 1e-12)
    tb.add("L1 torso pos: ref full_qpos[0:3] vs py d.xpos (stale)", _mx(sf.qpos[0:3] - cap["xpos"]), None, None)
    tb.add("L1 torso pos: ref full_qpos[0:3] vs py d.qpos (fresh)", _mx(sf.qpos[0:3] - cap["qpos"][0:3]), None, None,
           "stale vs fresh (1 tick)")
    tb.add("L1 torso quat: ref vs py d.xquat (stale)", _mx(sf.qpos[3:7] - cap["xquat"]), None, None)
    tb.add("L1 joints q: ref vs py d.qpos (fresh)", _mx(sf.qpos[7:] - cap["qpos"][7:]), None, None)
    leg_dof = model.leg_dof.ravel()
    tb.add("L1 leg qd: ref vs py d.qvel", _mx(sf.qvel[leg_dof] - cap["qvel"][leg_dof]), None, None)
    tb.add("L1 torso lin vel (mj_objectVelocity BODY)", _mx(sf.torso_vel - cap["torso_vel"]), None, None)
    tb.add("L1 torso ang vel", _mx(sf.torso_angvel - cap["torso_angvel"]), None, None)
    tb.add("L1 foot pos (site, stale)", _mx(sf.foot_pos - cap["foot_pos"]), None, None)
    tb.add("L2 x0", _mx(sf.x0 - cap["x0"]), _mx(sf.x0), None)
    tb.add("L3 reduced mass", abs(sf.mass - cap["mass"]), sf.mass, None)
    tb.add("L3 reduced com_offset_B", _mx(sf.com_loc - cap["com_offset_B"]), None, None)
    tb.add("L3 reduced inertia_B", _mx(sf.inertia - cap["inertia_B"]), _mx(sf.inertia), None)
    tb.add("L4 horizon t0", abs(sf.t0 - cap["t0"]), None, None)
    tb.add("L4 filtered cmd", _mx(np.array([sf.cmd["x_dot"], sf.cmd["y_dot"], sf.cmd["psi_dot"]]) - cap["filtered"]), None, None)
    tb.add("L4 desired feet", _mx(sf.desired - cap["desired"]), None, None)
    tb.add("L4 X_ref", _mx(sf.X_ref - cap["X_ref"]), _mx(sf.X_ref), None)
    tb.add("L4 r", _mx(sf.r - cap["r"]), None, None)
    tb.add("L4 CM active/alpha", _mx(np.r_[sf.active.astype(float) - cap["active"].astype(float), sf.alpha - cap["alpha"]]),
           None, None)
    tb.add("L4 touchdown yaw", _mx(sf.td_yaw - cap["td_yaw"]), None, None)
    tb.add("L5 A_qp", _mx(sf.A_qp - cap["A_qp"]), _mx(sf.A_qp), None)
    tb.add("L5 B_qp", _mx(sf.B_qp - cap["B_qp"]), _mx(sf.B_qp), None)
    tb.add("L6 u0 forces [N] (warm/cold as it happened)", _mx(sf.u0[:6] - cap["u0"][:6]), _mx(sf.u0[:6]), None,
           f"py cold={cap['cold']} iters={cap['iters']} {cap['status']}")
    tb.add("L6 u0 moments [Nm]", _mx(sf.u0[6:] - cap["u0"][6:]), _mx(sf.u0[6:]), None)
    tb.add("L6 U horizon (300)", _mx(sf.U - cap["U"]), _mx(sf.U), None)
    tb.add("L7 tau all actuators (18, unclamped)", _mx(sf.tau - cap["tau_full"]), _mx(sf.tau), None)
    return tb


# =================================================================================================================
# 궤적 비교
# =================================================================================================================
TRAJ_GROUPS = [
    ("com [m]", ("com_x", "com_y", "com_z"), "com", None),
    ("roll/pitch/yaw [rad]", ("roll", "pitch", "yaw"), "rpy", None),
    ("torso vel [m/s]", ("vx", "vy", "vz"), "vel", None),
    ("torso angvel [rad/s]", ("wx", "wy", "wz"), "angvel", None),
    ("L foot pos [m]", ("left_foot_x", "left_foot_y", "left_foot_z"), "foot", 0),
    ("R foot pos [m]", ("right_foot_x", "right_foot_y", "right_foot_z"), "foot", 1),
    ("foot normal force [N]", ("left_fn", "right_fn"), "fn", None),
    ("desired GRF L [N]", ("left_desired_fx", "left_desired_fy", "left_desired_fz"), "des_f", 0),
    ("desired GRF R [N]", ("right_desired_fx", "right_desired_fy", "right_desired_fz"), "des_f", 1),
    ("torque norm [Nm]", ("torque_norm",), "tau_norm", None),
]


def compare_traj(T, R, windows=((0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0))):
    it_ref = np.round(T["time"] / DT).astype(int)
    it_py = np.round(R["t"] / DT).astype(int)
    common, ir, ip = np.intersect1d(it_ref, it_py, return_indices=True)
    t = common * DT
    print(f"\n  --- trajectory (reference telemetry csv, 6 decimals) vs Python, common ticks {len(common)} "
          f"({t[0]:.3f} … {t[-1]:.3f} s)")
    hdr = "    " + f"{'quantity':<24s}" + "".join(f"  [{a:.1f},{b:.1f})" for a, b in windows)
    print(hdr)
    diffs = {}
    for name, cols, key, sub in TRAJ_GROUPS:
        ref = np.stack([T[c][ir] for c in cols], 1)
        py = R[key][ip]
        if sub is not None:
            py = py[:, sub]
        py = py.reshape(len(ip), -1)
        dd = np.abs(ref - py).max(1)
        diffs[name] = dd
        cells = []
        for a, b_ in windows:
            w = (t >= a - 1e-9) & (t < b_ - 1e-9)
            cells.append(f"  {dd[w].max():9.2e}" if w.any() else f"  {'-':>9s}")
        print(f"    {name:<24s}" + "".join(cells))
    dcom = diffs["com [m]"]
    for thr in (1e-5, 1e-4, 1e-3, 1e-2, 5e-2):
        k = np.nonzero(dcom > thr)[0]
        print(f"    com |Δ| first > {thr:.0e} m at t = {t[k[0]]:.3f} s" if len(k) else f"    com |Δ| never > {thr:.0e} m")
    return t, diffs


# =================================================================================================================
# 서브커맨드
# =================================================================================================================
def _job(args):
    side, name, osqp = args[:3]
    k = args[3] if len(args) > 3 else 0                     # ensemble: 명령 크기 × (1 + k·1e-9) (혼돈 분산 측정)
    case = _case(name)
    if k:
        for ax in ("vx", "vy", "wz"):
            case[ax] = case[ax] * (1.0 + k * ENS_EPS)
    if side == "ref":
        total = 10.0 if case["stand"] else WALK_T + 120.0
        r = run_ref(case, total, OUT / (f"surv_{name}" + (f"_k{k}" if k else "")), tel_dt=0.01)
        met = ref_metrics(r["T"], case, r["t_fall"])
        met.update(fell=r["t_fall"] is not None, wall=r["wall"], mpc_fail=r["n_mpc_fail"], t_end=r["t_end"])
        return side, name, osqp, met
    seconds = 10.0 if case["stand"] else 120.0
    return side, name, osqp, run_py_survival(case, seconds, OSQP_REF_EFFECTIVE if osqp == "ref" else None)


def cmd_survival(a):
    from multiprocessing import Pool
    names = a.cases.split(",") if a.cases else list(CASES)
    jobs = []
    if not a.py_only:
        jobs += [("ref", n, "-") for n in names]
    if not a.ref_only:
        jobs += [("py", n, "port") for n in names]
        if a.osqp_ref:
            jobs += [("py", n, "ref") for n in names]
    res = {}
    with Pool(min(max(1, min(a.procs, MAX_PROCS)), len(jobs))) as pool:
        for side, name, osqp, met in pool.imap_unordered(_job, jobs):
            res[(side, name, osqp)] = met
            print(f"[done] {side:3s} {name:7s} osqp={osqp:4s} survived {met.get('survived', float('nan')):7.2f} s "
                  f"fell={met.get('fell')} track={met.get('track', float('nan')):.1f}%", flush=True)
    print("\n| case | reference (C++) | Python port (faithful) |" + (" Python + OSQP ref-eff |" if a.osqp_ref else ""))
    print("|---|---|---|" + ("---|" if a.osqp_ref else ""))

    def cell(m):
        if m is None:
            return "-"
        s = f"{'✗' if m.get('fell') else '✓'} {m['survived']:.1f} s"
        if not m.get("fell") and m.get("track") is not None and np.isfinite(m.get("track", np.nan)):
            s += f", {m['track']:.0f} %"
        if "mpc_fail" in m:
            s += f" (MPC fail {m['mpc_fail']})"
        elif m.get("fallbacks") is not None:
            s += f" (fallback {m['fallbacks']})"
        return s
    for n in names:
        row = f"| {n} | {cell(res.get(('ref', n, '-')))} | {cell(res.get(('py', n, 'port')))} |"
        if a.osqp_ref:
            row += f" {cell(res.get(('py', n, 'ref')))} |"
        print(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "survival.json", "w") as f:
        json.dump({f"{k[0]}|{k[1]}|{k[2]}": {kk: (vv if not isinstance(vv, (np.floating,)) else float(vv))
                                              for kk, vv in v.items()} for k, v in res.items()}, f, indent=1, default=str)


ENS_EPS = 1e-9


def cmd_ensemble(a):
    """같은 케이스를 명령 크기 × (1 + k·1e-9), k = 1..K 로 돌려 생존 시간 분포 (reference / Python port / Python ref-eff)."""
    from multiprocessing import Pool
    names = a.cases.split(",")
    jobs = []
    for n in names:
        for k in range(1, a.k + 1):
            jobs.append(("ref", n, "-", k))
            jobs.append(("py", n, "port", k))
            jobs.append(("py", n, "ref", k))
    res = {}
    with Pool(max(1, min(a.procs, MAX_PROCS))) as pool:
        for side, name, osqp, met, k in pool.imap_unordered(_ens_job, jobs):
            res[(side, name, osqp, k)] = met
            print(f"[done] {side:3s} {name:7s} osqp={osqp:4s} k={k} survived {met['survived']:7.2f} s fell={met.get('fell')}",
                  flush=True)
    print("\n| case | reference (C++) | Python port | Python + OSQP ref-eff |  (survived s, k = 1..K; ✓ = 120 s)")
    print("|---|---|---|---|")
    for n in names:
        cells = []
        for side, osqp in (("ref", "-"), ("py", "port"), ("py", "ref")):
            vals = [res[(side, n, osqp, k)] for k in range(1, a.k + 1)]
            cells.append(" ".join("✓" if not v.get("fell") else f"{v['survived']:.1f}" for v in vals)
                         + f"  (median {np.median([v['survived'] for v in vals]):.1f})")
        print(f"| {n} | " + " | ".join(cells) + " |")
    with open(OUT / "ensemble.json", "w") as f:
        json.dump({f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in res.items()}, f, indent=1, default=str)


def _ens_job(args):
    side, name, osqp, met = _job(args)
    return side, name, osqp, met, args[3]


def cmd_lockstep(a):
    case = _case(a.case)
    run_dir = OUT / f"lock_{a.case}{a.tag}"
    triggers = [float(x) for x in a.snap.split(",")] if a.snap else []
    print(f"[lockstep] case {a.case} {case}  seconds {a.seconds}  snapshots at {triggers}  -> {run_dir}")
    r = run_ref(case, a.seconds, run_dir, triggers=triggers, tel_dt=DT, kill_on_fall=False)
    snaps = load_snapshots(run_dir)
    iters = [S["metadata"]["controller_iteration"] for _, S in snaps]
    print(f"[lockstep] reference: fall at {r['t_fall']}  MPC fail {r['n_mpc_fail']}  snapshots {len(snaps)} (iterations {iters})")
    osqp = OSQP_REF_EFFECTIVE if a.osqp == "ref" else None
    p = run_py(case, a.seconds, capture_iters=iters, osqp_settings=osqp, fall_margin=1e9)
    print(f"[lockstep] python ({'OSQP ref-eff' if osqp else 'port settings'}): fall at {p['t_fall']}  "
          f"fallbacks {p['n_fallback']}")
    np.savez_compressed(run_dir / f"py_{a.osqp}.npz", **p["R"])
    model, cfg = p["model"], p["cfg"]
    for f, S in snaps:
        sf = snap_fields(S)
        print(f"\n=== snapshot {f.name}: t {sf.t:.3f} it {sf.it} mode {sf.mode} ({sf.state}) "
              f"active {sf.active.astype(int)} alpha {sf.alpha}")
        cap = p["caps"].get(sf.it)
        if cap is not None:
            lockstep_checks(sf, cap, model).print("L: lockstep (same tick, python run)")
        tb, _ = isolated_checks(sf, MitModel(), cfg)
        tb.print("I: isolated (snapshot inputs -> python modules)")
    compare_traj(r["T"], p["R"])


def cmd_snapdiff(a):
    cfg = config.load()
    for f, S in load_snapshots(Path(a.dir)):
        sf = snap_fields(S)
        print(f"\n=== snapshot {f.name}: t {sf.t:.3f} it {sf.it} mode {sf.mode}")
        tb, _ = isolated_checks(sf, MitModel(), cfg)
        tb.print("I: isolated")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    s = sub.add_parser("survival")
    s.add_argument("--cases", default="")
    s.add_argument("--ref-only", action="store_true")
    s.add_argument("--py-only", action="store_true")
    s.add_argument("--osqp-ref", action="store_true", help="Python 을 reference OSQP 실효 설정으로도 (interval 50, dualgap 1)")
    s.add_argument("--procs", type=int, default=MAX_PROCS, help="병렬 프로세스 수 (≤ 6)")
    s = sub.add_parser("lockstep")
    s.add_argument("--case", default="vx06", choices=list(CASES))
    s.add_argument("--seconds", type=float, default=3.0)
    s.add_argument("--snap", default="0,0.5,1.0,1.1")
    s.add_argument("--osqp", default="port", choices=["port", "ref"])
    s.add_argument("--tag", default="")
    s = sub.add_parser("ensemble")
    s.add_argument("--cases", default="vx06,vy03,wz13,vx06r,vy03r,wz13r")
    s.add_argument("--k", type=int, default=6)
    s.add_argument("--procs", type=int, default=MAX_PROCS)
    s = sub.add_parser("snapdiff")
    s.add_argument("dir")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep()
    elif a.cmd == "survival":
        cmd_survival(a)
    elif a.cmd == "ensemble":
        cmd_ensemble(a)
    elif a.cmd == "lockstep":
        cmd_lockstep(a)
    else:
        cmd_snapdiff(a)


if __name__ == "__main__":
    main()
