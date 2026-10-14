"""채택 · 후보 설정 문자열 (10_walk.parse_variant 문법). Q&A 10/6 Q49 — 스크래치 도구가 사라져 저장소로 옮김.

A30: 10/1 발표 설정 (`Convex MPC/` 폴더와 같은 값)
A33: A30 + wsh=0.5 (warm start 반 칸, Q18) + kdz=0.8 (스윙 감쇠, Q27) + pitchtraj=0.7 (스윙 발 pitch 궤적, Q28)
A34: A33 + qpy=1e5 (옆 위치 가중치 1/10, Q39 · Q44 · Q45) — 10/9 채택
A35: A34 + qpitch=3000 (pitch 가중치 1/3, Q51) — 10/13 채택 (사용자)
A36: A35 + yawlead=1 (착지 yaw 선행 1 을 기본 · 주 결과로, Q52 · Q53) — 10/13 사용자 결정. 시험은 선행 0 / 0.5 / 1 모두
A38: A37 + admms=25,rhos=1e-5 (서기도 ADMM 25 회 — 서기 전용 고정 ρ 1e-5 로 정확해와 같은 외란 결과, Q55) — 10/13 (걷기는 A37 과 같음, ρ 통일은 걷기가 불안정해져 분리 유지 — Q57)
A39: A38 + kdz=0.5 (스윙 PD 감쇠비 0.8 → 0.5 — 고속 스윙 hip pitch 포화 감소, Q61 · 부작용 Q63) — 10/14 채택 (사용자)
A40: A39 + toetraj=1 (스윙 높이 궤적을 발끝에 — 고속 외란 개선, Q61 · Q62) — 10/14 보류 (사용자: 발목 한계로 뒤꿈치가 들리는 뿌리부터, Q64)
A41: A39 + armsw=1.0 (팔 반대 흔들기 0.3 → 1.0 rad/m — 위몸 yaw 흔들림 · 발끝 구간 발 회전 감소, Q73 ~ Q75) — 10/14 후보
A37: A36 + holdclip=1 (디딤 발 yaw hold 뒤 비틀림을 선 접촉 한계로 — 적용 렌치까지 OA-MPC 제약, Q50 · Q53 (나)) — 10/13 사용자 결정 (Q54 기준표)
모델은 10/9 대칭판 (Q42) 기준.
"""
A30 = ("profile=ramp,fix=half+yawref,ypath=0.3,init=warm,yawtraj=1.0,yawff=2,rai=0.2,vdir=0.7,nbs=5,rho=5e-5,hor=0.44,fref=1e-3,"
       "xmax=1.0,rvwb=1,replan=1,rpfz=0.08,kdz=1.2,wmax=2.5,armsw=0.3,shkp=60,shkd=12,shw=2.0,qyaw=750,foot=line,cop=0.075,"
       "cmar=0.075,cmarw=0.1,shabs=1,admm=25,admms=50,msc=1,lproj=3,lprojs=1,minw=0.1,swwn=0.46,tcycle=0.55,tswing=0.22,"
       "khalf=0.41,psioff=0,ksx=0.35,ksy=0.47,ymax=1.0,yawlead=0.5")
A33 = A30 + ",wsh=0.5,kdz=0.8,pitchtraj=0.7"
A34 = A33 + ",qpy=100000"
A35 = A34 + ",qpitch=3000"
A36 = A35 + ",yawlead=1"
A37 = A36 + ",holdclip=1"
A38 = A37 + ",admms=25,rhos=1e-5"
A39 = A38 + ",kdz=0.5"
A40 = A39 + ",toetraj=1"
A41 = A39 + ",armsw=1.0"
CONFIGS = {"A30": A30, "A33": A33, "A34": A34, "A35": A35, "A36": A36, "A37": A37, "A38": A38, "A39": A39, "A40": A40, "A41": A41}


def resolve(spec: str) -> str:
    """'A34' · 'A34,kdz=1.0' · 전체 문자열 모두 받는다. 앞부분이 설정 이름이면 그 문자열로 바꾼다."""
    head, _, rest = spec.partition(",")
    if head in CONFIGS:
        return CONFIGS[head] + ("," + rest if rest else "")
    return spec
