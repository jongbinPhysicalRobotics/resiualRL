"""09_walk.py 의 커맨드라인 플래그를 그대로 읽어 headless()/view() 인자로 바꾼다.

23_walk_affine.py · 24_walk_split.py 가 같이 쓴다 — 권장 실행 구성을 스크립트 이름만 바꿔 붙여 넣을 수 있게.
플래그 의미는 09_walk.py __main__ 과 CODE_MAP §5 참고.
"""
from __future__ import annotations

import sys


def _get(argv, key, default, cast=float):
    return cast(argv[argv.index(key) + 1]) if key in argv else default


def parse(argv=None) -> dict:
    argv = sys.argv if argv is None else argv
    if "--nogait" in argv:                       # 보행 스케줄 끔 = 가만히 서 있기 (gait.GAIT_ON, Q&A 9/24 Q8)
        import gait
        gait.GAIT_ON = 0
        print("  [서 있기] 보행 스케줄 끔 — 양발 항상 stance")
    sw_ = None
    if "--sidew" in argv:
        a = argv[argv.index("--sidew") + 1]
        sw_ = "auto" if a == "auto" else float(a) / 100.0
    spf = 0
    if "--quintic" in argv:
        spf = 2 if argv[argv.index("--quintic") + 1] == "all" else 1
    ng = "--nogate" in argv
    common = dict(
        kp_up=_get(argv, "--uppd", 60.0),
        swing_id="--swingid" in argv,
        wz=_get(argv, "--wz", 0.0),
        yaw_hold="--noyawhold" not in argv,
        soft_land="--softland" in argv,
        lam_swing="--lamswing" in argv,
        wn_swing=_get(argv, "--wn", 100.0),
        zeta_swing=_get(argv, "--zeta", 0.5),
        cap_y=_get(argv, "--capy", None),
        cycle=_get(argv, "--cycle", 0.8),
        stance_frac=_get(argv, "--sf", 0.75),
        lip_exact="--lip" in argv,
        q_py=_get(argv, "--qpy", None),
        td_mode=_get(argv, "--tdmode", "cont", str),
        gate_ff=not (ng or "--footframe" in argv),
        gate_sy=not (ng or "--swingyaw" in argv),
        swing_h=_get(argv, "--swingh", 0.05),
        jdot="--jdot" in argv,
        side_w=sw_,
        td_scale=_get(argv, "--tdscale", 1.0),
        td_dx=_get(argv, "--tddx", 0.0) / 100.0,
        cop_margin=_get(argv, "--copm", 1.0),
        du_f=_get(argv, "--duf", 0.0),
        du_m=_get(argv, "--dum", 0.0),
        wz_pelvis=_get(argv, "--wzpel", 0.0),
        wx_pelvis=_get(argv, "--wxpel", 0.0),
        wy_pelvis=_get(argv, "--wypel", 0.0),
        td_sink=_get(argv, "--tdsink", 0.0) / 1000.0,
        swing_prof=spf,
        lo_ramp=_get(argv, "--loramp", 0.0) / 1000.0,
        early_td="--earlytd" in argv,
        liftoff_fix="--oldliftoff" not in argv,
    )
    view_kw = dict(
        follow="--nofollow" not in argv,
        realtime="--fast" not in argv,
        max_sim=_get(argv, "--vsec", None),
        timelog=_get(argv, "--timelog", None, str),
        sync_every=_get(argv, "--syncevery", 17, int),
        draw_at_sync="--drawmpc" not in argv,
        boost="--noboost" not in argv,
        lite="--lite" in argv,
    )
    return dict(
        vx=_get(argv, "--vx", 0.0),
        seconds=_get(argv, "--seconds", 12.0),
        legmass=_get(argv, "--legmass", 1.0),
        decim=_get(argv, "--decim", None, int),
        view="--view" in argv,
        common=common,
        view_kw=view_kw,
    )
