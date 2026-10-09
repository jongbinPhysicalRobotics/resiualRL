"""MIT 휴머노이드 걷기 러너 — CasADi 형식 재작업 폴더 (casadi_mit, Q&A 9/26 Q2·Q3).

mit 폴더와의 차이: MPC 가 mpc_casadi.CasadiCondensedMPC (동역학 기호식 + 자동미분, mit 이식과 1e-11 일치),
OSQP 기본값 = reference 실효 설정 (rhoint 50 · dualgap 1), 기본 명령 프로파일 = ramp + yawref (안정 구성),
GAIT_ON (gait.py 파일 값 1, 환경변수 우선): 1 = 걷기, 0 = 제자리 서기 (FSM standing).
발 모델: 기본 ref (mx 자유) — OA-MPC 선 접촉은 --foot line (아직 검증 전, Q&A 9/26 Q2 §4).


루프 (07 §2, spec 05 §1.2): tau = controller.tick()  (현재 MjData, mj_forward 없음) → write_torque (ctrlrange clamp)
→ mj_step.  시작: keyframe 'stand' + mj_forward 한 번 (D2). 컨트롤러는 t = 0 에 초기화 → StandingSettle 1.0 s → Walking.

명령 profile (D3):
    const (기본) : raw 명령이 첫 틱부터 있음 (reference headless 기본 의미 — filter 첫 호출 snap + walking 시작 틱 dt=0 snap)
    step         : raw 명령이 walking 시작 (settle 1.0 s) 부터
    ramp         : walking 시작부터 2 s 램프 (reference 램프 공식 x = sign·min(|v|, rate (t − t0)))
모두 reference 필터 (tau 0.92/0.8/0.70 s, clamp 0.7/0.5/2.0) 를 거친다.

실행 (ROOT 에서):
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe MPC/src/mit/10_walk.py --vx 0.6 --seconds 120
    ... --vy 0.3 | --wz 1.3 | --view | --solver quadprog | --noarmature | --profile step|ramp | --log
    ... --stand (FSM 을 Standing 에 고정: requested_mode standing) | --markers (U1) | --mu 0.8 (마찰 계수 배율)
--seconds = walking 시간 (시뮬 총 시간 = settle 1.0 s + seconds). 넘어짐 = torso z < 0.35 또는 |roll|,|pitch| > 0.8 rad.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import numpy as np

import config
import paths
from command import CommandSchedule
from controller import MitController
from mit_model import LEFT, RIGHT, MitModel, quat_to_roll_pitch

FALL_Z = 0.35
FALL_ANGLE = 0.8
X_LABELS = ["roll", "pitch", "yaw", "px", "py", "pz", "wx", "wy", "wz", "vx", "vy", "vz", "g"]
U_LABELS = ["FxL", "FyL", "FzL", "FxR", "FyR", "FzR", "MxL", "MyL", "MzL", "MxR", "MyR", "MzR"]


# ----------------------------------------------------------------------------------------------------------------
def make_schedule(profile: str, vx: float, vy: float, wz: float, walk_t: float) -> CommandSchedule:
    if profile == "const":
        return CommandSchedule.constant(vx, vy, wz)
    if profile == "step":
        return CommandSchedule.step_at(walk_t, vx, vy, wz)
    if profile.startswith("ramp"):                  # ramp = 2 s 에 목표 도달, rampT = T 초 (Q&A 9/30 Q3)
        t_ramp = float(profile[4:]) if len(profile) > 4 else 2.0
        prof = {}
        for k, v in (("x_dot", vx), ("y_dot", vy), ("psi_dot", wz)):
            prof[k] = {"final": v, "rate": abs(v) / t_ramp if v != 0 else 0.0, "start": walk_t}
        return CommandSchedule(prof)
    raise ValueError(f"unknown profile {profile!r}")


def parse_variant(s: str | None) -> dict:
    """'k=v,k=v' → kwargs. 키: solver, arm (0/1) / noarm, profile, markers (0/1), mu (배율), fsm2 (0/1),
    rhoint / maxit / eps (OSQP 설정 변형: adaptive_rho_interval, max_iter, eps_abs=eps_rel)."""
    kw = {}
    if not s or s == "base":
        return kw
    for item in s.split(","):
        item = item.strip()
        if not item or item == "base":
            continue
        k, _, v = item.partition("=")
        if k == "solver":
            kw["solver"] = v
        elif k == "arm":
            kw["armature"] = bool(int(v))
        elif k == "noarm":
            kw["armature"] = False
        elif k == "profile":
            kw["profile"] = v
        elif k == "markers":
            kw["markers"] = bool(int(v)) if v else True
        elif k == "mu":
            kw["friction_scale"] = float(v)
        elif k in ("rhoint", "maxit", "eps"):
            key = {"rhoint": "adaptive_rho_interval", "maxit": "max_iter", "eps": "eps_abs"}[k]
            kw.setdefault("osqp_settings", {})[key] = float(v) if k == "eps" else int(v)
            if k == "eps":
                kw["osqp_settings"]["eps_rel"] = float(v)
        elif k == "fsm2":
            kw["fsm_double_update"] = bool(int(v))
        elif k == "fix":                           # reference 특이점 고침 (controller.py fixes): fix=slide+yawref
            kw["fixes"] = tuple(x for x in v.split("+") if x)
        elif k == "admm":                          # 논문식 고정 반복 ADMM N 회 (ρ 고정, 종료 판정 · 재시도 없음 — Q&A 9/28 Q5)
            kw.setdefault("osqp_settings", {})["fixed_iter"] = int(v)
        elif k == "init":                          # 고정 반복의 초기 추정: nominal (논문) / warm / zero
            kw.setdefault("osqp_settings", {})["init"] = str(v)
        elif k == "admms":                         # 서기 모드 고정 ADMM 반복 수 (기본 = admm 과 같음) — Q&A 10/1 Q22
            kw.setdefault("osqp_settings", {})["fixed_iter_stand"] = int(v)
        elif k == "ruiz":                          # OSQP Ruiz 스케일링 반복 수 (기본 10) — Q&A 9/30 Q16
            kw.setdefault("osqp_settings", {})["scaling"] = int(v)
        elif k == "rho":                           # OSQP ρ (기본 0.1)
            kw.setdefault("osqp_settings", {})["rho"] = float(v)
        elif k == "dualgap":                       # OSQP check_dualgap (이식 0, reference 실효 1 — Q&A 9/25 Q11)
            kw.setdefault("osqp_settings", {})["check_dualgap"] = int(v)
        elif k == "xmax":                          # 명령 필터 전진 상한 (reference 0.7) — 1 m/s 시험 (Q&A 9/25 Q9)
            kw["x_dot_max"] = float(v)
        elif k == "yawtraj":                       # 스윙 발 yaw 를 최소 저크 궤적으로 (스윙 시간 비율, 기본 끔 — Q&A 9/29 Q1)
            kw.setdefault("swing_over", {})["yaw_trajectory_fraction"] = float(v)
        elif k == "yawff":                         # yawtraj 의 속도 목표 (기본 1) — 0 이면 위치만 보간 (Q&A 9/29 Q1)
            kw.setdefault("swing_over", {})["yaw_trajectory_ff"] = int(v)          # 0 / 1 / 2
        elif k in ("yawkp", "yawkd"):              # 스윙 발 yaw 자세 이득 (기본 yaml 305 / 18)
            kw.setdefault("swing_over", {})["yaw_kp" if k == "yawkp" else "yaw_kd"] = float(v)
        elif k == "khalfy":                        # 옆 방향 착지 미리보기 k_half 따로 (기본 = 전진과 같음 — Q&A 9/28 Q12)
            kw["k_half_y"] = float(v)
        elif k == "khalf":                         # 착지 미리보기 k_half 를 속도와 무관하게 고정 (기본: 0.37 / 0.28 / 0.26 전환)
            kw["k_half"] = float(v)
        elif k == "foot":                          # 발 접촉 모델: ref | line (OA-MPC 선 접촉)
            kw["foot_model"] = str(v)
        elif k == "standhold":                     # 서기 발 yaw hold (기본 1 — Q&A 9/26 Q6)
            kw["stand_yaw_hold"] = bool(int(v))
        elif k == "armsw":                         # 팔 반대 흔들기 k [rad/m] (기본 0 — Q&A 9/28)
            kw["arm_swing"] = float(v)
        elif k in ("tdz", "fref", "tswing", "tcycle", "hor", "kdz", "raii", "psioff", "raiy", "cent", "srbm", "walkbias", "vmeas", "vdir", "rais", "raisx", "wbcom", "swh", "tors", "nbs", "minw", "replan", "rvwb", "rpfz", "swprof", "swlim", "wmax", "ymax", "shw", "lproj", "wdu", "cmar", "fpd", "sjd", "sjdm", "uint", "uintm", "shabs", "hmpc", "msc", "ksx", "ksy", "ksw", "cmarw", "lprojs", "attank", "swq", "frefm", "zoff", "rsc", "fzmax", "mdt", "wsh", "wdum", "pitchtraj", "toeoff", "toeeps", "toepin", "latsway"):   # 설정 덮어쓰기 (Q&A 9/29 Q5 ~ Q8)
            path = {"tdz": "swing.touchdown_z", "fref": "mpc.force_ref_weight", "tswing": "timing.swing",
                    "tcycle": "timing.cycle", "hor": "timing.horizon", "kdz": "swing.kd_zeta",
                    "raii": "swing.raibert_int_gain", "psioff": "swing.psi_offset_scale",
                    "raiy": "swing.raibert_gain_y", "cent": "swing.centrifugal_gain",
                    "srbm": "mpc.srb_mass_scale", "walkbias": "swing.walk_stance_bias", "vmeas": "swing.step_vmeas_blend",
                    "vdir": "swing.step_vdir_blend",
                    "rais": "swing.raibert_stride_gain_y", "raisx": "swing.raibert_stride_gain_x",
                    "wbcom": "mpc.whole_body_com", "swh": "swing.height",
                    "tors": "mpc.torsional_friction_scale", "nbs": "mpc.iterations_between_solve",
                    "minw": "swing.min_step_width", "replan": "swing.replan",
                    "rvwb": "swing.raibert_vel_wholebody", "rpfz": "swing.replan_freeze", "swprof": "swing.xy_profile_ramp", "swlim": "swing.torque_limit_mode",
                    "wmax": "user_command_filter.psi_dot_max", "ymax": "user_command_filter.y_dot_max",
                    "shw": "swing.stance_yaw_fade_rate",
                    "lproj": "mpc.line_wrench_projection",   # lproj=1 투영 · 2 자르기: 디딤 발 최종 wrench 를 선 접촉 집합으로 (foot=line 만, Q&A 9/30 Q11)
                    "wdu": "mpc.input_rate_weight",          # wdu=w: MPC 입력 변화 비용 (힘 w, 모멘트 w/l², Q&A 9/30 Q12)
                    "cmar": "mpc.cop_margin_frac",           # cmar=ε: line 발 두 점이 각각 Fz 의 ε 이상 (CoP 행만 (1−2ε)l, Q&A 9/30 Q12)
                    "fpd": "swing.stance_foot_pitch_damping",   # fpd=kd: 딛는 발 pitch 각속도 감쇠 모멘트 (MPC 밖, Q&A 9/30 Q12)
                    "sjd": "swing.stance_joint_damping",        # sjd=kd: 딛는 발 고정 관절 감쇠 (논문 식 (21) K_d 항, Q&A 9/30 Q13)
                    "sjdm": "swing.stance_joint_damping_mode",  # sjdm=2: 감쇠를 수평 미끄럼 · yaw 에만
                    "uint": "mpc.input_interpolation",          # uint=s: 풀이 사이 u0 → u1 선형 보간, 기울기 배율 s (Q&A 9/30 Q13 · Q15)
                    "uintm": "mpc.input_interpolation_mode",    # uintm=1: 디딤이 바뀌는 다리는 보간 안 함 (Q&A 9/30 Q15)
                    "shabs": "swing.stance_yaw_rate_absolute",  # shabs=1: 딛는 발 yaw hold 감쇠를 절대 yaw 각속도로 (Q&A 9/30 Q13)
                    "hmpc": "mpc.hold_in_mpc_weight",           # hmpc=w: yaw hold 를 MPC Mz 참조 비용으로 (yawhold=0 과 함께, Q&A 9/30 Q14)
                    "msc": "mpc.moment_variable_scale",         # msc=1: 모멘트 변수를 발 반길이로 나눠 풀기 (해 같음, Q&A 9/30 Q16)
                    "ksx": "swing.k_sched_fast_x",              # ksx=k: 곧게 빨리 걸을 때 전진 k_half 목표 (Q&A 10/1 Q20)
                    "ksy": "swing.k_sched_lat_y",               # ksy=k: 옆으로 걸을 때 옆 k_half 목표
                    "ksw": "swing.k_sched_turn_rate",           # ksw=w: |ψ̇| ≥ w 면 기본 k (기본 0.5 rad/s)
                    "cmarw": "mpc.cop_margin_frac_walk",        # cmarw=ε: 걷기 전용 두 점 최소 하중 (서기는 cmar, Q&A 10/1 Q21)
                    "lprojs": "mpc.line_wrench_projection_stand",   # lprojs=1: 서기에서도 MPC 해 자르기 (Q&A 10/1 Q22)
                    "frefm": "mpc.force_ref_total_mass",        # frefm=1: fref 참조를 전신 무게 / 발 수로 (Q&A 10/6 Q5)
                    "zoff": "mpc.height_ref_offset",            # zoff=m: MPC 높이 참조를 처음 자세 + m (서기 · 걷기, Q&A 10/6 Q6)
                    "rsc": "mpc.walking_input_scale",           # rsc=s: 걷기 입력 가중치 12 개 전부 × s (wf · wmom 뒤에 곱함, Q&A 10/6 Q6)
                    "fzmax": "mpc.normal_force_max",            # fzmax=N: MPC 발 수직력 상한 (기본 yaml 500, 9/26 Q5 — Q&A 10/6 Q7)
                    "mdt": "timing.mpc_dt",                     # mdt=s: MPC 한 칸 길이 (기본 0.02, 칸 수 = 지평 / mdt — Q&A 10/6 Q9)
                    "wsh": "mpc.warm_shift_steps",              # wsh=s: warm start 를 s 칸 당김 (기본 1, 0.5 = 재풀이 10 ms 에 맞춤 — Q&A 10/6 Q9)
                    "wdum": "mpc.input_rate_mode",              # wdum=1: wdu 를 첫 칸 (직전 적용값에 묶기) 만, 2: 지평 안 이웃 칸 차이만 (Q&A 10/6 Q17)
                    "pitchtraj": "swing.pitch_trajectory_fraction",   # pitchtraj=f: 스윙 발 pitch 를 이륙 기울기 → 수평으로 스윙의 f 동안 (Q&A 10/6 Q28)
                    "toeoff": "mpc.toe_off_fraction",           # toeoff=f: 디딤 경과 f 이후 칸은 뒤꿈치 최소 하중을 풀어 CoP 를 발끝까지 (Q&A 10/6 Q30)
                    "toeeps": "mpc.toe_off_margin",             # toeeps=ε: 그 칸의 발끝 쪽 CoP 한계 (1 − 2ε)·l (기본 0 = 발끝 끝)
                    "toepin": "mpc.toe_off_pin",                # toepin=c: 그 칸의 CoP 를 발끝 쪽 [c·l, (1 − 2ε)·l] 에 묶음 (기본 없음 = 풀기만)
                    "latsway": "mpc.lateral_sway_scale",        # latsway=s: MPC 옆 속도 참조에 LIPM 주기 궤도 흔들림 × s (기본 0 = 없음, Q&A 10/6 Q37)
                    "attank": "swing.attitude_ankle_only",    # attank=1: 스윙 발 pitch 자세 모멘트를 발목에만 (Q&A 10/6 Q3)
                    "swq": "swing.blend_order"}[k]             # swq=5: 스윙 궤적 5 차 최소 저크 (기본 3 차 smoothstep)
            kw.setdefault("raw_over", {})[path] = float(v)
        elif k == "push":                          # 외란 (Q&A 9/29 Q11, 교수님 지시): Fx:Fy:t0:dur — 진행 방향 기준 [N],
            f_ = [float(x) for x in str(v).split(":")]   # t0 = 측정 창 시작 (걷기 시작 / 서기면 0) 뒤 [s], dur [s]. 몸통 CoM 에 가함
            kw["push"] = tuple(f_ + [10.0, 0.2][len(f_) - 2:]) if len(f_) < 4 else tuple(f_)
        elif k in ("shkp", "shkd"):                # 딛는 발 yaw hold 이득 (기본 yaml 20 / 4) — Q&A 9/30 Q10
            kw.setdefault("swing_over", {})["stance_yaw_kp" if k == "shkp" else "stance_yaw_kd"] = float(v)
        elif k == "fwid":                          # 명목 발 옆 간격 배율 (기본 1 = yaml ±0.076 m) — Q&A 9/30 Q14
            kw.setdefault("swing_over", {})["nominal_foot_offsets_B"] = [[0.0, 0.075999602 * float(v), 0.0],
                                                                        [0.0, -0.075999602 * float(v), 0.0]]
        elif k == "swwn":                          # 스윙 고유진동수 배율 (기본 1 = [151, 151, 110] rad/s) — Q&A 9/30 Q5
            kw.setdefault("swing_over", {})["natural_frequency"] = [151.0 * float(v), 151.0 * float(v), 110.0 * float(v)]
        elif k == "symcom":                        # 진단 전용 (Q&A 9/30 Q7): 몸통 무게중심 y 를 0 으로 (모델 좌우 대칭화)
            kw["sym_com"] = bool(int(v))
        elif k == "syminert":                      # 변형 (Q&A 10/6 Q40): 몸통 관성의 좌우 비대칭 곱 관성 ixy · iyz 만 0 (질량 · 대각 · ixz 그대로)
            kw["sym_inertia"] = bool(int(v))
        elif k == "hipfix":                        # 변형 (Q&A 10/6 Q41): 왼 hip_yaw 관성을 오른쪽의 거울상으로 (URDF ixz 부호 오류, 질량 · CoM 그대로)
            # ※ 10/9 (Q42) 부터 모델 파일 자체를 고쳐서 symcom · syminert · hipfix 는 효과가 없다 (이미 대칭). 옛 기록 재현용으로 남김.
            kw["hip_fix"] = bool(int(v))
        elif k == "wf":                            # 걷기 입력 가중치의 수평력 (Fx, Fy ×2, 기본 0.001) — 두 발 내력 억제 (Q&A 9/29 Q12)
            kw["w_force_xy"] = float(v)
        elif k == "pushrep":                       # 외란 반복 주기 [s] (0 = 한 번) — push 와 함께
            kw["push_repeat"] = float(v)
        elif k == "stand":                         # 서기 시험 (FSM 서기 고정, 시간 0 부터)
            kw["stand"] = bool(int(v))
        elif k == "xpath":                         # MPC 앞뒤 위치 참조를 경로선 쪽으로 (뒤처진 만큼 × a, ±5 cm, 기본 0 — Q&A 9/29 Q4)
            kw["xpath"] = float(v)
        elif k == "ypath":                         # MPC 옆 위치 참조를 경로선 쪽으로 (0 ~ 1, 기본 0 — Q&A 9/27 Q2)
            kw["ypath"] = float(v)
        elif k == "ypos":                          # 착지 옆 위치 되먹임 k [-] (기본 0 — Q&A 9/27 Q2)
            kw["ypos_gain"] = float(v)
        elif k == "rai":                           # 착지점 측정 속도 되먹임 K [s] (기본 0 = reference — Q&A 9/26 Q8)
            kw["raibert_gain"] = float(v)
        elif k == "sideswap":                      # 걸음 위상 좌우 맞바꿈 (Q&A 9/26 Q7)
            kw["side_swap"] = bool(int(v))
        elif k == "standbias":                     # 서기 다리 자중 보상 (기본 1 — Q&A 9/26 Q6)
            kw["stand_bias"] = bool(int(v))
        elif k == "pmin":                          # 발 두 점 (뒤꿈치 · 앞꿈치) 각각의 최소 수직력 [N] (기본 0 — Q&A 9/28 Q12)
            kw["point_fmin"] = float(v)
        elif k == "cop":                           # MPC CoP 앞뒤 한계 foot_half_length [m] (기본 yaml 0.065 — Q&A 9/28)
            kw["cop_half"] = float(v)
        elif k == "wmom":                          # MPC 걷기 입력 가중치 중 발 모멘트 6 개 (기본 yaml 1e-3 — Q&A 9/28)
            kw["w_moment"] = float(v)
        elif k in ("qpitch", "qyaw", "qroll", "qvx", "qvy", "qpx", "qpy", "qz", "qvz"):   # MPC 걷기 상태 가중치 (기본 yaml roll 50000 / pitch 9000 / yaw 500 / vx 100 / vy 70)
            kw.setdefault("q_state", {})[{"qroll": 0, "qpitch": 1, "qyaw": 2, "qvx": 9, "qvy": 10,   # qpx · qpy · qz · qvz: Q&A 10/6 Q6
                                          "qpx": 3, "qpy": 4, "qz": 5, "qvz": 11}[k]] = float(v)
        elif k == "cone":                          # 시뮬 마찰 원뿔: pyramidal (MuJoCo 기본, 마름모) / elliptic (원) — Q&A 9/28 Q6
            kw["sim_cone"] = str(v)
        elif k == "yawhold":                       # 걷기 디딤 발 yaw hold (MPC 밖 Mz, 기본 yaml 1) — Q&A 9/28 Q6
            kw["yaw_hold"] = bool(int(v))
            if int(v) == 2:                        # 2 = 두 점 접촉일 때만 (Q&A 9/29 Q5)
                kw.setdefault("swing_over", {})["stance_yaw_hold_mode"] = 2
        elif k == "yawlead":                       # 착지 yaw 선행 배율 swing_foot_yaw_lead_scale (기본 1.0) — Q&A 9/28 Q6
            kw["yaw_lead"] = float(v)
        elif k == "mpcmu":                         # MPC 제약의 µ (기본 yaml 1.0). 시뮬 마찰 (mu=) 과 별개. µ_t = 0.0657·µ 도 따라감
            kw["mpc_mu"] = float(v)
        else:
            raise KeyError(f"unknown variant key {k!r}")
    return kw


# ----------------------------------------------------------------------------------------------------------------
class Recorder:
    """틱마다 한 행. 미리 할당한 배열."""

    def __init__(self, n: int):
        self.n = 0
        z = lambda *s: np.full((n,) + s, np.nan)   # noqa: E731
        self.t = z()
        self.base = z(3)
        self.rpy = z(3)
        self.x0 = z(13)
        self.xref0 = z(13)
        self.u = z(12)
        self.tau = z(10)
        self.tau_arm = z(8)
        self.est = z(2)
        self.sched = z(2)
        self.active = z(2)
        self.alpha = z(2)
        self.early = z(2)
        self.fn = z(2)
        self.foot = z(2, 3)
        self.desired = z(2, 3)
        self.pdes = z(2, 3)
        self.cmd = z(3)
        self.raw = z(3)
        self.walking = z()
        self.solved = np.zeros(n, dtype=bool)
        self.solve_ms = z()
        self.total_ms = z()
        self.cold = np.zeros(n, dtype=bool)
        self.fallback = np.zeros(n, dtype=bool)
        self.iters = z()
        self.stance0 = z(2)

    def add(self, ctl: MitController, tau_leg, tau_arm):
        i = self.n
        s = ctl.state
        self.t[i] = s.t
        self.base[i] = s.torso_pos_W
        self.rpy[i] = (s.roll, s.pitch, s.yaw_unwrapped)
        self.x0[i] = ctl.x0
        self.u[i] = ctl.u_hold
        self.tau[i] = np.asarray(tau_leg).reshape(-1)
        self.tau_arm[i] = np.asarray(tau_arm).reshape(-1)
        cm = ctl.cm
        self.est[i] = cm.estimated
        self.sched[i] = cm.scheduled if ctl.mode == "walking" else (1, 1)
        self.active[i] = [ctl._active(l, s.t) for l in (LEFT, RIGHT)]
        self.alpha[i] = [ctl._alpha(l) for l in (LEFT, RIGHT)]
        self.early[i] = cm.early
        self.fn[i] = s.foot_normal_force
        self.foot[i] = s.foot_pos_W
        self.desired[i] = ctl.desired
        self.pdes[i] = ctl.swing_pdes()
        f = ctl.filtered
        self.cmd[i] = (f.x_dot, f.y_dot, f.psi_dot)
        r = ctl.raw
        self.raw[i] = (r.x_dot, r.y_dot, r.psi_dot)
        self.walking[i] = 1.0 if ctl.mode == "walking" else 0.0
        info = ctl.tick_info.get("info")
        if info is not None:
            self.solved[i] = True
            self.solve_ms[i] = info.get("solve_ms", np.nan)
            self.total_ms[i] = info.get("total_ms", np.nan)
            self.cold[i] = bool(info.get("cold", False))
            self.fallback[i] = bool(info.get("fallback", False))
            self.iters[i] = info.get("iters", np.nan)
            self.stance0[i] = info.get("stance0", np.full(2, np.nan))
        if ctl.last_info is not None:
            self.xref0[i] = ctl.last_info.get("x_ref0", np.full(13, np.nan))
        self.n += 1

    def trim(self):
        n = self.n
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                out[k] = v[:n]
        return out


# ----------------------------------------------------------------------------------------------------------------
def _edges(b: np.ndarray):
    """(rising idx, falling idx) of a bool array."""
    b = b.astype(np.int8)
    d = np.diff(b)
    return np.nonzero(d == 1)[0] + 1, np.nonzero(d == -1)[0] + 1


def compute_metrics(R: dict, cmd=(0.0, 0.0, 0.0), walk_t=None, lim=None, window=100.0, dt=0.002) -> dict:
    t = R["t"]
    n = len(t)
    out = {}
    walking = R["walking"] > 0.5
    if walk_t is None:
        walk_t = float(t[np.argmax(walking)]) if walking.any() else float(t[0])
    t_end = float(t[-1]) if n else 0.0
    out["sim_t"] = t_end
    out["walk_t"] = walk_t
    out["survived"] = max(0.0, t_end - walk_t)
    w0 = max(t_end - window, walk_t + 5.0)
    win = (t >= w0) & walking
    if win.sum() < 50:
        win = (t >= walk_t + 1.0) & walking
    if win.sum() < 10:
        win = np.ones(n, dtype=bool)
    out["win_s"] = float(win.sum() * dt)
    yaw = R["rpy"][:, 2]
    v = R["x0"][:, 9:12]
    c, s = np.cos(yaw), np.sin(yaw)
    vh = np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], 1)
    out["vx"] = float(np.mean(vh[win, 0]))
    out["vy"] = float(np.mean(vh[win, 1]))
    iw = np.nonzero(win)[0]
    out["wz"] = float((yaw[iw[-1]] - yaw[iw[0]]) / max(t[iw[-1]] - t[iw[0]], 1e-9))
    trk = []
    for name, c_, a_ in (("x", cmd[0], out["vx"]), ("y", cmd[1], out["vy"]), ("wz", cmd[2], out["wz"])):
        if abs(c_) > 1e-9:
            out[f"track_{name}"] = 100.0 * a_ / c_
            trk.append(out[f"track_{name}"])
    out["track"] = float(np.mean(trk)) if trk else np.nan
    out["roll_sd"] = float(np.std(R["rpy"][win, 0]))
    out["pitch_sd"] = float(np.std(R["rpy"][win, 1]))
    out["pitch_mean"] = float(np.mean(R["rpy"][win, 1]))
    # yaw 진동: 한 cycle (0.5 s) 이동평균을 뺀 나머지의 95% 폭 (명령 yaw 램프/드리프트 제거)
    k = int(round(0.5 / dt))
    if n > k:
        ker = np.ones(k) / k
        ma = np.convolve(yaw, ker, mode="same")
        osc = yaw - ma
        valid = win.copy()
        valid[:k] = False
        valid[-k:] = False
        if valid.sum() > 10:
            out["yaw_osc95"] = float(np.percentile(osc[valid], 97.5) - np.percentile(osc[valid], 2.5))
        else:
            out["yaw_osc95"] = np.nan
    # 스윙/착지 (walking 구간)
    apex, td_err, early_sw, slips, n_sw = [], [], 0, [], 0
    for leg in (LEFT, RIGHT):
        act = (R["active"][:, leg] > 0.5) | ~walking
        sched = (R["sched"][:, leg] > 0.5) | ~walking
        est = R["est"][:, leg] > 0.5
        early = R["early"][:, leg] > 0.5
        rise_a, fall_a = _edges(act)            # 착지 (active on), 이륙 (active off)
        rise_s, _ = _edges(sched)
        fz = R["foot"][:, leg, 2]
        for f in fall_a:
            if not win[f]:
                continue
            nxt = rise_a[rise_a > f]
            if len(nxt) == 0:
                continue
            e = nxt[0]
            n_sw += 1
            apex.append(float(np.max(fz[f:e + 1])))
            if early[f:e + 1].any():
                early_sw += 1
            # 실제 접촉 (estimated 상승) − 명목 착지 (scheduled 상승)
            rs = rise_s[rise_s > f]
            if len(rs):
                ts_ = t[rs[0]]
                seg = est[f:min(n, rs[0] + int(0.15 / dt))]
                off = np.nonzero(~seg)[0]                      # 먼저 estimated 가 꺼지고 (이륙 확인)
                if len(off):
                    j = np.nonzero(seg[off[0]:])[0]            # 다시 켜지는 첫 틱 = 실제 착지
                    if len(j):
                        td_err.append(float(t[f + off[0] + j[0]] - ts_))
        # stance slip: 착지 + 10 ms 부터 이륙까지 발 xy 이동
        for r in rise_a:
            if not win[r]:
                continue
            nf = fall_a[fall_a > r]
            if len(nf) == 0:
                continue
            a0 = min(r + 5, nf[0] - 1)
            slips.append(float(np.linalg.norm(R["foot"][nf[0] - 1, leg, :2] - R["foot"][a0, leg, :2])))
    out["n_swings"] = n_sw
    out["apex_mm"] = 1e3 * float(np.mean(apex)) if apex else np.nan
    out["td_err_ms"] = 1e3 * float(np.mean(td_err)) if td_err else np.nan
    out["td_err_sd_ms"] = 1e3 * float(np.std(td_err)) if td_err else np.nan
    out["early_rate"] = early_sw / n_sw if n_sw else np.nan
    out["slip_mm"] = 1e3 * float(np.mean(slips)) if slips else np.nan
    out["slip_max_mm"] = 1e3 * float(np.max(slips)) if slips else np.nan
    # 마찰 사용률 (MPC 명령 u, 디딤 발, Fz > 20 N): 원뿔 |F_xy|/Fz 와 사각뿔 max(|Fx|,|Fy|)/Fz (제약이 보는 양)
    fr_c, fr_p = [], []
    for leg, (ix, iy, iz) in enumerate(((0, 1, 2), (3, 4, 5))):
        m_ = win & (R["active"][:, leg] > 0.5) & (R["u"][:, iz] > 20.0)
        if m_.any():
            fx, fy, fz = R["u"][m_, ix], R["u"][m_, iy], R["u"][m_, iz]
            fr_c.append(np.hypot(fx, fy) / fz)
            fr_p.append(np.maximum(np.abs(fx), np.abs(fy)) / fz)
    if fr_c:
        fr_c, fr_p = np.concatenate(fr_c), np.concatenate(fr_p)
        out["fric_cone_p95"] = float(np.percentile(fr_c, 95))
        out["fric_cone_max"] = float(fr_c.max())
        out["fric_pyr_p95"] = float(np.percentile(fr_p, 95))
    else:
        out["fric_cone_p95"] = out["fric_cone_max"] = out["fric_pyr_p95"] = np.nan
    # 좌우 하중비 (명령 Fz 중앙값, 디딤·Fz > 20 N) — R/L. 1 이면 대칭 (Q&A 9/26 Q7)
    fzs = []
    for leg, iz in ((0, 2), (1, 5)):
        m_ = win & (R["active"][:, leg] > 0.5) & (R["u"][:, iz] > 20.0)
        fzs.append(float(np.median(R["u"][m_, iz])) if m_.any() else np.nan)
    out["load_RL"] = fzs[1] / fzs[0] if fzs[0] and np.isfinite(fzs[0]) else np.nan
    if lim is not None:
        ratio = np.abs(R["tau"][win]) / lim[None, :]
        out["tau_ratio_max"] = float(ratio.max()) if ratio.size else np.nan
        out["tau_sat_frac"] = float((ratio >= 1.0).any(1).mean()) if ratio.size else np.nan
    solved = R["solved"]
    out["solves"] = int(solved.sum())
    out["cold_rate"] = float(R["cold"][solved].mean()) if solved.any() else np.nan
    out["fallbacks"] = int(R["fallback"].sum())
    out["solve_ms_med"] = float(np.nanmedian(R["total_ms"][solved])) if solved.any() else np.nan
    out["solve_ms_p99"] = float(np.nanpercentile(R["total_ms"][solved], 99)) if solved.any() else np.nan
    out["iters_med"] = float(np.nanmedian(R["iters"][solved])) if solved.any() else np.nan
    return out


# ----------------------------------------------------------------------------------------------------------------
def save_log(R: dict, tag: str, note: str = "") -> Path:
    paths.LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = paths.LOG_DIR / f"mit_{tag}_{stamp}"
    flat = {}
    for k, v in R.items():
        flat[k] = v
    np.savez_compressed(str(base) + ".npz", **flat, x_labels=np.array(X_LABELS), u_labels=np.array(U_LABELS),
                        note=np.array(note))
    # csv (10 틱마다 = 50 Hz)
    idx = np.arange(0, len(R["t"]), 10)
    cols = [("t", R["t"][idx, None])]
    cols += [(f"x_{l}", R["x0"][idx, i:i + 1]) for i, l in enumerate(X_LABELS)]
    cols += [(f"ref_{l}", R["xref0"][idx, i:i + 1]) for i, l in enumerate(X_LABELS)]
    cols += [(f"u_{l}", R["u"][idx, i:i + 1]) for i, l in enumerate(U_LABELS)]
    cols += [(f"tau{j}", R["tau"][idx, j:j + 1]) for j in range(10)]
    for nm in ("est", "active", "alpha"):
        cols += [(f"{nm}_L", R[nm][idx, 0:1]), (f"{nm}_R", R[nm][idx, 1:2])]
    for leg, sd in ((0, "L"), (1, "R")):
        cols += [(f"foot{sd}_{a}", R["foot"][idx, leg, i:i + 1]) for i, a in enumerate("xyz")]
        cols += [(f"des{sd}_{a}", R["desired"][idx, leg, i:i + 1]) for i, a in enumerate("xyz")]
    cols += [("solve_ms", R["total_ms"][idx, None])]
    header = ",".join(c[0] for c in cols)
    data = np.hstack([c[1] for c in cols])
    np.savetxt(str(base) + ".csv", data, delimiter=",", header=header, comments="", fmt="%.6g", encoding="utf-8-sig")
    return Path(str(base) + ".npz")


def _tag(vx, vy, wz, variant):
    f = lambda v: f"{v:.2f}".replace("-", "m").replace(".", "p")   # noqa: E731
    parts = [variant or "base"]
    if vx:
        parts.append(f"vx{f(vx)}")
    if vy:
        parts.append(f"vy{f(vy)}")
    if wz:
        parts.append(f"wz{f(wz)}")
    if not (vx or vy or wz):
        parts.append("inplace")
    return "walk_" + "_".join(parts)


def _setup(vx, vy, wz, solver, armature, profile, markers, stand, friction_scale, fsm_double_update, verbose,
           osqp_settings=None, fixes=(), x_dot_max=None, k_half=None, foot_model="ref", mpc_mu=None,
           stand_yaw_hold=True, stand_bias=True, side_swap=False, raibert_gain=0.0, ypos_gain=0.0, ypath=0.0, arm_swing=0.0, cop_half=None, w_moment=None, q_state=None, sim_cone=None, yaw_hold=None,
           yaw_lead=None, point_fmin=None, k_half_y=None, swing_over=None, xpath=0.0, raw_over=None, w_force_xy=None,
           sym_com=False, sym_inertia=False, hip_fix=False):
    cfg = config.load()
    if (x_dot_max is not None or k_half is not None or mpc_mu is not None or cop_half is not None or w_moment is not None
            or q_state or yaw_hold is not None or yaw_lead is not None or point_fmin is not None or k_half_y is not None or swing_over or raw_over
            or w_force_xy is not None):
        raw = cfg.to_dict()
        if cop_half is not None:                  # MPC CoP 앞뒤 한계 (기본 yaml 0.065, 실제 선 ±0.075) — Q&A 9/28
            raw["mpc"]["foot_half_length"] = float(cop_half)
        if w_force_xy is not None:                # 걷기 입력 가중치의 수평력 (Fx, Fy ×2) — Q&A 9/29 Q12
            for i_ in (0, 1, 3, 4):
                raw["mpc"]["walking"]["input_weight_diag"][i_] = float(w_force_xy)
        if w_moment is not None:                  # 걷기 입력 가중치의 발 모멘트 (Mx, My, Mz ×2) — Q&A 9/28
            raw["mpc"]["walking"]["input_weight_diag"][6:12] = [float(w_moment)] * 6
        for path_, v_ in (raw_over or {}).items():  # 일반 설정 덮어쓰기 (tdz · fref · tswing · tcycle · hor · kdz) — Q&A 9/29 Q5
            sec, key = path_.split(".")
            raw[sec][key] = v_
        if raw_over and "mpc.walking_input_scale" in raw_over:   # 변형 rsc (Q&A 10/6 Q6)
            s_ = float(raw_over["mpc.walking_input_scale"])
            raw["mpc"]["walking"]["input_weight_diag"] = [float(w) * s_ for w in raw["mpc"]["walking"]["input_weight_diag"]]
        if raw_over and ("timing.swing" in raw_over or "timing.cycle" in raw_over):   # 입각 = 주기 − 스윙
            raw["timing"]["stance"] = float(raw["timing"]["cycle"]) - float(raw["timing"]["swing"])
        if raw_over and ("timing.horizon" in raw_over or "timing.mpc_dt" in raw_over):   # 스텝 간격 20 ms 유지 (mdt 를 주면 그 값)
            raw["timing"]["horizon_steps"] = int(round(float(raw["timing"]["horizon"]) / float(raw_over.get("timing.mpc_dt", 0.02))))
        for k_, v_ in (swing_over or {}).items():  # 스윙 설정 덮어쓰기 (yawtraj · yawkp · yawkd) — Q&A 9/29 Q1
            raw["swing"][k_] = v_
        if k_half_y is not None:                  # 옆 방향 k_half — Q&A 9/28 Q12
            raw["swing"]["lateral_half_stance_offset"] = float(k_half_y)
        if point_fmin is not None:                # 발 두 점 최소 수직력 — Q&A 9/28 Q12
            raw["mpc"]["point_force_min"] = float(point_fmin)
        if yaw_hold is not None:                  # 걷기 디딤 발 yaw hold (MPC 밖 Mz) — Q&A 9/28 Q6
            raw["swing"]["enable_stance_foot_yaw_hold"] = bool(yaw_hold)
        if yaw_lead is not None:                  # 착지 yaw 선행 배율 — Q&A 9/28 Q6
            raw["swing"]["swing_foot_yaw_lead_scale"] = float(yaw_lead)
        for i, w in (q_state or {}).items():      # 걷기 상태 가중치 (roll 0 / pitch 1 / yaw 2) — Q&A 9/28
            raw["mpc"]["walking"]["state_weight_diag"][i] = float(w)
        if mpc_mu is not None:                    # MPC 마찰 제약 µ 변형 (기본 = yaml 1.0, Q&A 9/26 Q4)
            raw["mpc"]["friction_coefficient"] = float(mpc_mu)
        if x_dot_max is not None:                 # 명령 필터 전진 상한 변형 (기본 = yaml 0.7)
            raw["user_command_filter"]["x_dot_max"] = float(x_dot_max)
        if k_half is not None:                    # swing.*half_stance_offset 세 값을 하나로 (기본 = 0.37 / 0.28 / 0.26)
            for key in ("body_velocity_half_stance_offset", "mid_speed_body_velocity_half_stance_offset",
                        "high_speed_body_velocity_half_stance_offset"):
                raw["swing"][key] = float(k_half)
        cfg = config.MitConfig(raw)
    model = MitModel(armature=armature, emulate_debug_markers=markers)
    if sym_com:                                   # 진단 전용: base 몸체 무게중심 y (+4.8 mm) 를 0 으로
        model.m.body_ipos[model.base_body, 1] = 0.0
    if sym_inertia:                               # 변형 syminert (Q&A 10/6 Q40): URDF 몸통 관성의 ixy · iyz 만 0 으로
        b_ = model.base_body                      # (좌우 대칭 몸통이면 0 이어야 하는 성분). 질량 · ixx · iyy · izz · ixz 는 그대로.
        R_ = np.zeros(9); mujoco.mju_quat2Mat(R_, model.m.body_iquat[b_]); R_ = R_.reshape(3, 3)
        I_ = R_ @ np.diag(model.m.body_inertia[b_]) @ R_.T             # 몸체 좌표, CoM 기준 전체 텐서 (= URDF 값)
        I_[0, 1] = I_[1, 0] = 0.0
        I_[1, 2] = I_[2, 1] = 0.0
        w_, V_ = np.linalg.eigh(I_)
        if np.linalg.det(V_) < 0.0:
            V_[:, 2] *= -1.0
        q_ = np.zeros(4); mujoco.mju_mat2Quat(q_, V_.reshape(-1))
        model.m.body_inertia[b_] = w_
        model.m.body_iquat[b_] = q_
        model._rb_inertia = model.m.body_inertia[model.reduced_bodies].copy()   # 축소 몸체 주관성 캐시도 갱신
        mujoco.mj_setConst(model.m, model.d)
    if hip_fix:                                   # 변형 hipfix (Q&A 10/6 Q41): URDF 왼 hip_yaw 의 ixz 가 거울상과 부호가 반대
        bR = mujoco.mj_name2id(model.m, mujoco.mjtObj.mjOBJ_BODY, "right_hip_yaw")   # (오른 −0.000558, 왼 +0.000558).
        bL = mujoco.mj_name2id(model.m, mujoco.mjtObj.mjOBJ_BODY, "left_hip_yaw")    # 왼 관성 = S·I_오른·S (S = diag(1,−1,1)).
        R_ = np.zeros(9); mujoco.mju_quat2Mat(R_, model.m.body_iquat[bR]); R_ = R_.reshape(3, 3)
        Sm = np.diag([1.0, -1.0, 1.0])
        I_ = Sm @ (R_ @ np.diag(model.m.body_inertia[bR]) @ R_.T) @ Sm
        w_, V_ = np.linalg.eigh(I_)
        if np.linalg.det(V_) < 0.0:
            V_[:, 2] *= -1.0
        q_ = np.zeros(4); mujoco.mju_mat2Quat(q_, V_.reshape(-1))
        model.m.body_inertia[bL] = w_
        model.m.body_iquat[bL] = q_                # 질량 · CoM 위치 (이미 거울상) 는 그대로
        mujoco.mj_setConst(model.m, model.d)
    if sim_cone is not None:                      # 시뮬 마찰 원뿔 (MuJoCo 기본 pyramidal) — Q&A 9/28 Q6
        model.m.opt.cone = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
                            "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}[sim_cone]
    model.reset(str(cfg.port.spawn_keyframe))
    walk_t = float(cfg.startup.post_init_standing_settle_time)
    sched = make_schedule(profile, vx, vy, wz, walk_t)
    ctl = MitController(model, cfg, sched, solver=solver, requested_mode="standing" if stand else None,
                        emulate_markers=markers, friction_scale=friction_scale,
                        fsm_double_update=fsm_double_update, osqp_settings=osqp_settings, verbose=verbose,
                        fixes=fixes, foot_model=foot_model, stand_yaw_hold=stand_yaw_hold, stand_bias=stand_bias,
                        side_swap=side_swap, raibert_gain=raibert_gain, ypos_gain=ypos_gain, ypath=ypath,
                        arm_swing=arm_swing, xpath=xpath)
    return cfg, model, ctl, walk_t


def _fell(d) -> bool:
    if not np.all(np.isfinite(d.qpos)):
        return True
    r, p = quat_to_roll_pitch(d.qpos[3:7])
    return d.qpos[2] < FALL_Z or abs(r) > FALL_ANGLE or abs(p) > FALL_ANGLE


def _push_active(push, push_repeat, t_rel) -> bool:
    """외란 (Q&A 9/29 Q11): t_rel = 측정 창 시작 뒤 시각. push = (Fx, Fy, t0, dur), push_repeat = 반복 주기 (0 = 한 번)."""
    tr = t_rel - push[2]
    if tr < 0.0:
        return False
    if push_repeat and push_repeat > 0.0:
        tr = tr % push_repeat
    return tr < push[3]


def _push_world(Fx, Fy, yaw):
    c_, s_ = np.cos(yaw), np.sin(yaw)
    return np.array([c_ * Fx - s_ * Fy, s_ * Fx + c_ * Fy, 0.0])


def headless(vx=0.0, vy=0.0, wz=0.0, seconds=120.0, *, solver="osqp", armature=True, profile="const",
             markers=False, stand=False, friction_scale=1.0, fsm_double_update=True, osqp_settings=None, log=False,
             variant=None, verbose=True, print_every=10.0, keep_record=False, fixes=(), x_dot_max=None, k_half=None,
             foot_model="ref", mpc_mu=None, stand_yaw_hold=True, stand_bias=True, side_swap=False, raibert_gain=0.0,
             ypos_gain=0.0, ypath=0.0, arm_swing=0.0, cop_half=None, w_moment=None, q_state=None, sim_cone=None,
             yaw_hold=None, yaw_lead=None, point_fmin=None, k_half_y=None, swing_over=None, xpath=0.0,
             raw_over=None, push=None, push_repeat=0.0, w_force_xy=None, sym_com=False, sym_inertia=False, hip_fix=False):
    """walking seconds 동안 (시뮬 = settle + seconds). 반환 dict (metrics + fell 등).

    push = (Fx, Fy, t0, dur): 진행 방향 (몸통 yaw) 기준 수평력 [N] 을 측정 창 시작 + t0 부터 dur 동안 몸통 CoM 에 가한다."""
    cfg, model, ctl, walk_t = _setup(vx, vy, wz, solver, armature, profile, markers, stand, friction_scale,
                                     fsm_double_update, verbose=False, osqp_settings=osqp_settings, fixes=fixes,
                                     x_dot_max=x_dot_max, k_half=k_half, foot_model=foot_model, mpc_mu=mpc_mu,
                                     stand_yaw_hold=stand_yaw_hold, stand_bias=stand_bias, side_swap=side_swap,
                                     raibert_gain=raibert_gain, ypos_gain=ypos_gain, ypath=ypath,
                                     arm_swing=arm_swing, cop_half=cop_half, w_moment=w_moment,
                                     q_state=q_state, sim_cone=sim_cone, yaw_hold=yaw_hold, yaw_lead=yaw_lead,
                                     point_fmin=point_fmin, k_half_y=k_half_y, swing_over=swing_over,
                                     xpath=xpath, raw_over=raw_over, w_force_xy=w_force_xy, sym_com=sym_com,
                                     sym_inertia=sym_inertia, hip_fix=hip_fix)
    m, d = model.m, model.d
    total = (0.0 if stand else walk_t) + seconds
    n = int(round(total / m.opt.timestep))
    rec = Recorder(n + 1)
    fell = False
    err = None
    tw = time.perf_counter()
    next_print = print_every
    for k in range(n):
        try:
            tau_leg, tau_arm = ctl.tick()
        except Exception as exc:                  # C++ 에선 runController 밖으로 전파 → 시뮬 중단
            err = f"{type(exc).__name__}: {exc}"
            fell = True
            break
        model.write_torque(tau_leg, tau_arm)
        rec.add(ctl, tau_leg, tau_arm)
        if push is not None:                      # 외란: 몸통 CoM 에 수평력 (xfrc_applied 는 월드 좌표, CoM 에 작용)
            if _push_active(push, push_repeat, d.time - (0.0 if stand else walk_t)):
                d.xfrc_applied[model.base_body, 0:3] = _push_world(push[0], push[1], ctl.state.yaw_unwrapped)
            else:
                d.xfrc_applied[model.base_body, :] = 0.0
        mujoco.mj_step(m, d)
        if _fell(d):
            fell = True
            break
        if verbose and d.time >= next_print:
            next_print += print_every
            s = ctl.state
            f = ctl.filtered
            print(f"  t {d.time:6.1f}  z {d.qpos[2]:.3f}  xy ({d.qpos[0]:+7.2f},{d.qpos[1]:+6.2f})  yaw {s.yaw_unwrapped:+7.2f}"
                  f"  r/p {s.roll:+.3f}/{s.pitch:+.3f}  cmd {f.x_dot:.2f},{f.y_dot:.2f},{f.psi_dot:.2f}  "
                  f"{ctl.fsm.state.value}  wall {time.perf_counter() - tw:5.0f}s", flush=True)
    wall = time.perf_counter() - tw
    R = rec.trim()
    lim = model.ctrl_hi[model.leg_act.ravel()]
    res = compute_metrics(R, (vx, vy, wz), None if stand else walk_t, lim, dt=m.opt.timestep)
    res["survived"] = max(0.0, float(d.time) - (0.0 if stand else walk_t))   # 마지막 mj_step 뒤 시각
    res.update(fell=fell, error=err, wall=wall, vx_cmd=vx, vy_cmd=vy, wz_cmd=wz, seconds=seconds,
               variant=variant or "base", solver=solver, armature=armature, profile=profile, markers=markers,
               transitions=ctl.transitions, t_end=float(d.time), push=push, push_repeat=push_repeat)
    res["pass"] = (not fell) and res["survived"] >= seconds - 1e-6
    if log:
        tag = _tag(vx, vy, wz, variant) + ("_stand" if stand else "")
        pth = save_log(R, tag, note=str({k: v for k, v in res.items() if not isinstance(v, (list, dict))}))
        res["log"] = str(pth)
    if keep_record:
        res["record"] = R
    if verbose:
        print_summary(res)
    return res


def print_summary(r: dict):
    print(f"[mit walk] cmd vx {r['vx_cmd']:+.2f} vy {r['vy_cmd']:+.2f} wz {r['wz_cmd']:+.2f}  variant {r['variant']}  "
          f"solver {r['solver']} armature {r['armature']} profile {r['profile']}  wall {r['wall']:.0f}s")
    status = "FELL" if r["fell"] else "ok"
    print(f"  survived {r['survived']:.2f} s of {r['seconds']:.0f} s walking ({status})"
          + (f"  error: {r['error']}" if r.get("error") else ""))
    print(f"  achieved (heading frame, last {r['win_s']:.0f} s): vx {r['vx']:+.3f}  vy {r['vy']:+.3f}  wz {r['wz']:+.3f}"
          f"   tracking {r.get('track', float('nan')):.1f} %")
    print(f"  roll sd {r['roll_sd']:.4f}  pitch sd {r['pitch_sd']:.4f} (mean {r['pitch_mean']:+.4f})  "
          f"yaw osc95 {r.get('yaw_osc95', float('nan')):.4f} rad")
    print(f"  swings {r['n_swings']}  apex {r['apex_mm']:.1f} mm  td err {r['td_err_ms']:+.1f}±{r['td_err_sd_ms']:.1f} ms  "
          f"early {r['early_rate']:.2f}  slip/step {r['slip_mm']:.1f} mm (max {r['slip_max_mm']:.1f})")
    print(f"  friction use |F_xy|/Fz p95 {r.get('fric_cone_p95', float('nan')):.2f} (max {r.get('fric_cone_max', float('nan')):.2f})"
          f"  pyramid max(|Fx|,|Fy|)/Fz p95 {r.get('fric_pyr_p95', float('nan')):.2f}")
    print(f"  tau/limit max {r.get('tau_ratio_max', float('nan')):.2f}  sat frac {r.get('tau_sat_frac', float('nan')):.3f}")
    print(f"  MPC solves {r['solves']}  cold rate {r['cold_rate']:.3f}  fallbacks {r['fallbacks']}  "
          f"median {r['solve_ms_med']:.2f} ms (p99 {r['solve_ms_p99']:.1f})  iters {r['iters_med']:.0f}")
    if r.get("log"):
        print(f"  log: {r['log']}")
    verdict = "PASS" if r["pass"] else "FAIL"
    print(f"{verdict}  ({r['survived']:.1f} s)")


# ----------------------------------------------------------------------------------------------------------------
def _draw_refs(v, ctl, n_seg=16):
    """뷰어 오버레이 (G1 09_walk._draw_swing 관례): 스윙 참조궤적 (왼발 초록, 오른발 파랑) + 현재 목표 (작은 구)
    + 착지 목표 (빨간 구, planner/contact manager 의 managed desired) + MPC 몸 CoM 참조 지평 (하늘색 선)."""
    import planner as _pl
    scn = v.user_scn
    scn.ngeom = 0

    def sphere(p, r, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0.0, 0.0]),
                            np.asarray(p, float), np.eye(3).flatten(), np.asarray(rgba, np.float32))
        scn.ngeom += 1

    def line(a, b, rgba, w=3.0):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, w, np.asarray(a, float), np.asarray(b, float))
        scn.ngeom += 1

    ref = getattr(ctl, "last_ref", None)
    if ref is not None:                                     # MPC 몸 CoM 참조 (지평 0.5 s)
        P = ref.X_ref[:, 3:6]
        for a, b in zip(P[:-1], P[1:]):
            line(a, b, (0.2, 0.85, 1.0, 0.6), 2.0)
    for leg, col in ((0, (0.1, 0.9, 0.1, 0.9)), (1, (0.25, 0.45, 1.0, 0.9))):
        rt = ctl.legs[leg]
        tr = rt.traj
        if tr.active and tr.T > 0.0:                        # 스윙 전체 궤적 (같은 blend 식으로 샘플)
            z_mid = float(tr.p_final[2]) + float(tr.height)
            pts = []
            for s in np.linspace(0.0, 1.0, n_seg):
                bxy = _pl._blend(s)
                xy = (1.0 - bxy) * tr.p_init[:2] + bxy * tr.p_final[:2]
                if s <= 0.5:
                    bz = _pl._blend(2.0 * s)
                    z = (1.0 - bz) * float(tr.p_init[2]) + bz * z_mid
                else:
                    bz = _pl._blend(2.0 * s - 1.0)
                    z = (1.0 - bz) * z_mid + bz * float(tr.p_final[2])
                pts.append(np.array([xy[0], xy[1], z]))
            for a, b in zip(pts[:-1], pts[1:]):
                line(a, b, col, 3.0)
            sphere(tr.position(), 0.012, col)               # 지금 이 틱의 스윙 목표
        sphere(ctl.desired[leg], 0.016, (1.0, 0.15, 0.15, 0.9))   # 착지 목표 (빨강)


def view(vx=0.0, vy=0.0, wz=0.0, seconds=120.0, *, solver="osqp", armature=True, profile="const", markers=False,
         stand=False, friction_scale=1.0, fsm_double_update=True, follow=True, fixes=(), x_dot_max=None,
         foot_model="ref", raibert_gain=0.0, ypos_gain=0.0, ypath=0.0, arm_swing=0.0, push=None, push_repeat=0.0,
         push_mag=30.0, push_dur=0.2, **setup_kw):
    """setup_kw: _setup 의 나머지 변형 인자 (osqp_settings, yaw_lead, cop_half … — parse_variant 결과 그대로).

    외란 (Q&A 9/29 Q11):
      스크립트  --var push=Fx:Fy:t0:dur[,pushrep=T] — 헤드리스 시험과 같은 정의 (몸 yaw 기준, 몸통 CoM)
      키보드    ↑ 앞 · ↓ 뒤 · ← 왼 · → 오른 으로 push_mag [N] 을 push_dur [s] 동안. PageUp / PageDown = ±10 N
      미는 동안 몸통에 빨간 화살표 (1 cm/N), HUD 에 크기 · 상태 표시.
    """
    import threading
    import mujoco.viewer
    from viewer_hud import MitHUD
    KEY = {265: (1.0, 0.0, "fwd"), 264: (-1.0, 0.0, "back"), 263: (0.0, 1.0, "left"), 262: (0.0, -1.0, "right")}
    ps = dict(mag=float(push_mag), req=None, until=-1.0, F=(0.0, 0.0), label="", n=0)

    def on_key(code):                                           # 뷰어 스레드 — 요청만 남기고 적용은 시뮬 루프에서
        if code in KEY:
            ux, uy, name = KEY[code]
            ps["req"] = (ux * ps["mag"], uy * ps["mag"], f"{name} {ps['mag']:.0f} N")
        elif code == 266:
            ps["mag"] = min(ps["mag"] + 10.0, 200.0)
        elif code == 267:
            ps["mag"] = max(ps["mag"] - 10.0, 0.0)

    cfg, model, ctl, walk_t = _setup(vx, vy, wz, solver, armature, profile, markers, stand, friction_scale,
                                     fsm_double_update, verbose=True, fixes=fixes, x_dot_max=x_dot_max,
                                     foot_model=foot_model, raibert_gain=raibert_gain, ypos_gain=ypos_gain, ypath=ypath,
                                     arm_swing=arm_swing, **setup_kw)
    m, d = model.m, model.d
    total = (0.0 if stand else walk_t) + seconds
    sync_every = 17
    before = set(threading.enumerate())
    print("  외란: ↑ 앞 · ↓ 뒤 · ← 왼 · → 오른 (0.2 s), PageUp/PageDown ±10 N" + (f", 스크립트 {push}" if push else ""))
    with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as v:
        ui_threads = [t for t in threading.enumerate() if t not in before]
        hud = MitHUD(m, model.base_body)
        if follow:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = model.base_body
            v.cam.distance = 2.5
            v.cam.elevation = -15
            v.cam.azimuth = 135
        t_wall0 = time.perf_counter()
        k = 0
        was_pushing = False
        while v.is_running() and d.time < total:
            tau_leg, tau_arm = ctl.tick()
            model.write_torque(tau_leg, tau_arm)
            if ps["req"] is not None:                           # 키보드 밀기 시작
                fx_, fy_, lab = ps["req"]
                ps.update(req=None, until=d.time + float(push_dur), F=(fx_, fy_), label=lab, n=ps["n"] + 1)
                print(f"  [push #{ps['n']}] t {d.time:6.2f}  {lab}  {push_dur:.2f} s")
            F_B = None
            if d.time < ps["until"]:
                F_B = ps["F"]
            elif push is not None and _push_active(push, push_repeat, d.time - (0.0 if stand else walk_t)):
                F_B = (push[0], push[1])
                ps["label"] = f"script {push[0]:+.0f},{push[1]:+.0f} N"
            if F_B is not None:                                 # 몸통 CoM 에 수평력 (몸 yaw 기준 → 월드)
                d.xfrc_applied[model.base_body, 0:3] = _push_world(F_B[0], F_B[1], ctl.state.yaw_unwrapped)
                was_pushing = True
            elif was_pushing:                                   # 끝날 때 한 번만 0 (마우스 섭동은 건드리지 않음)
                d.xfrc_applied[model.base_body, :] = 0.0
                was_pushing = False
            mujoco.mj_step(m, d)
            k += 1
            if _fell(d):
                print(f"fell at t = {d.time:.2f}")
                break
            f = ctl.filtered
            hud.update(v, d, d.time, k, cmd=(f.x_dot, f.y_dot, f.psi_dot), mode=ctl.fsm.state.value,
                       extra=[("push (arrows, PgUp/Dn)",
                               f"{ps['mag']:.0f} N" + (f"  ACTIVE: {ps['label']}" if F_B is not None else ""))])
            if k % sync_every == 0:
                with v.lock():
                    _draw_refs(v, ctl)                      # 참조 오버레이 (Q&A 9/26 Q7 후속 분석용)
                    if F_B is not None:                     # 외란 화살표 (1 cm/N, 빨강)
                        scn = v.user_scn
                        if scn.ngeom < scn.maxgeom:
                            p0 = d.xipos[model.base_body].copy()
                            Fw = _push_world(F_B[0], F_B[1], ctl.state.yaw_unwrapped)
                            g = scn.geoms[scn.ngeom]
                            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                                                np.eye(3).flatten(), np.array([1.0, 0.1, 0.1, 1.0], np.float32))
                            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.02, p0 - 0.01 * Fw, p0)
                            scn.ngeom += 1
                v.sync()
                lag = d.time - (time.perf_counter() - t_wall0)
                if lag > 0:
                    time.sleep(lag)
        time.sleep(1.0)
    # close() 는 종료 요청만 한다. 뷰어 스레드가 끝나기 전에 인터프리터가 내려가면
    # glfw.terminate 와 render_loop 가 겹쳐 segfault (Ubuntu, MuJoCo 3.12) → 스레드 종료를 기다린다.
    for t in ui_threads:
        t.join(timeout=5.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--solver", default="osqp", choices=["osqp", "quadprog"])
    ap.add_argument("--noarmature", action="store_true")
    ap.add_argument("--profile", default="ramp", choices=["const", "step", "ramp"])
    ap.add_argument("--markers", action="store_true", help="U1 debug-marker mass emulation")
    ap.add_argument("--stand", action="store_true", help="FSM kept in Standing (requested mode standing)")
    ap.add_argument("--mu", type=float, default=1.0, help="friction coefficient scale (variant)")
    ap.add_argument("--fsm1", action="store_true", help="single FSM update per tick (variant)")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--fix", default="yawref", help="reference 특이점 고침 (기본 yawref — 빈 문자열로 끔): slide+yawref+yawanchor")
    ap.add_argument("--foot", default="ref", choices=["ref", "line", "mx0"],
                    help="발 접촉 모델: ref (reference) / mx0 (+ mx=0 등식) / line (OA-MPC 선 접촉)")
    ap.add_argument("--rai", type=float, default=0.0, help="착지점 측정 속도 되먹임 K [s] (Q&A 9/26 Q8, 기본 0 = reference)")
    ap.add_argument("--ypos", type=float, default=0.0, help="착지점 옆 위치 되먹임 k (Q&A 9/27 Q2, MATLAB 0.3·y_com)")
    ap.add_argument("--ypath", type=float, default=0.0, help="MPC 옆 위치 참조를 경로선 쪽으로 0~1 (Q&A 9/27 Q2)")
    ap.add_argument("--armsw", type=float, default=0.0, help="팔 반대 흔들기 k [rad/m] (Q&A 9/28, 기본 0 = 고정 팔)")
    ap.add_argument("--var", default=None, help="11_gait_quality 와 같은 변형 키 'k=v,k=v' (위 플래그보다 우선). 예: --var yawlead=0,rho=1e-5 · --var rho=1e-5,init=warm,admm=25")
    ap.add_argument("--xmax", type=float, default=None, help="명령 필터 전진 상한 [m/s] (기본 yaml 0.7)")
    a = ap.parse_args()
    import gait as _gait
    if _gait.GAIT_ON == 0 and not a.stand:                    # GAIT_ON=0 → 제자리 서기 (환경변수 우선)
        print("  [GAIT_ON=0] 걷기 스케줄 끔 — FSM standing (제자리 서기)")
        a.stand = True
    kw = dict(solver=a.solver, armature=not a.noarmature, profile=a.profile, markers=a.markers, stand=a.stand,
              friction_scale=a.mu, fsm_double_update=not a.fsm1, fixes=tuple(x for x in a.fix.split("+") if x),
              x_dot_max=a.xmax, foot_model=a.foot, raibert_gain=a.rai, ypos_gain=a.ypos, ypath=a.ypath,
              arm_swing=a.armsw)
    if a.var:                                                   # 변형 키 (11_gait_quality 와 같은 문법) — 위 플래그 덮어씀
        kw.update(parse_variant(a.var))
    if a.view:
        view(a.vx, a.vy, a.wz, a.seconds, **kw)
        return
    r = headless(a.vx, a.vy, a.wz, a.seconds, log=a.log, variant=a.variant, **kw)
    sys.exit(0 if r["pass"] else 1)


if __name__ == "__main__":
    main()
