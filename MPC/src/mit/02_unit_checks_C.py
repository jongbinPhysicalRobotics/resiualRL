"""Owner C 단위 점검 — mpc.py, leg_control.py (07_port_design.md §4 [C]).

실행:  python 02_unit_checks_C.py        (PASS/FAIL 줄 + 숫자, 실패가 있으면 exit 1)

다른 owner 모듈 (mit_model, gait, planner) 없이 돈다: 입력은 합성 (numpy / SimpleNamespace) 이고,
실제 모델이 필요한 두 항목 (spec 04 §14.6 Λ 표본, 무릎 토크 부호) 은 mujoco 로 모델을 직접 읽는다.
config 는 이 폴더의 config.py (owner A) 가 있으면 그것을, 없으면 reference yaml 을 직접 읽는다.
"""
from __future__ import annotations

import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):   # BLAS 1 스레드 (속도 측정 안정)
    os.environ.setdefault(_v, "1")
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mpc as M                     # noqa: E402
import leg_control as LC            # noqa: E402

np.set_printoptions(precision=6, suppress=True, linewidth=150)
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# config / model 로딩
# ---------------------------------------------------------------------------
def load_cfg():
    try:
        import config
        return config.load()
    except Exception as exc:                      # owner A 파일이 아직 없을 때
        print(f"(config.py unavailable: {exc}; using reference yaml)")
        import yaml

        def ns(v):
            if isinstance(v, dict):
                return SimpleNamespace(**{k: ns(x) for k, x in v.items()})
            if isinstance(v, list):
                try:
                    return np.asarray(v, dtype=float)
                except (TypeError, ValueError):
                    return v
            return v
        p = HERE.parents[2] / "reference" / "config" / "mit_humanoid" / "my_controller.yaml"
        return ns(yaml.safe_load(p.read_text(encoding="utf-8")))


def model_dir() -> Path:
    return next(p for p in HERE.parents if p.name == "MPC") / "models" / "mit_humanoid"


def load_model():
    import mujoco
    root = model_dir()
    assets = {f.name: f.read_bytes() for f in root.rglob("*") if f.is_file()}
    m = mujoco.MjModel.from_xml_string((root / "scene.xml").read_text(encoding="utf-8"), assets)
    return m, mujoco.MjData(m)


CFG = load_cfg()
N, DT = M._horizon(CFG)
MASS = 14.227468
# spec 01 §4 reduced body at the yaml / stand arm pose (yaw frame)
I_SPEC01 = np.array([[0.519367, 0.001639, 0.005789],
                     [0.001639, 0.168406, -0.001019],
                     [0.005789, -0.001019, 0.402526]])


def reduced_body_from_model(key="stand"):
    """spec 02 §2.2 (setupRobotParams.cpp:397-491) 를 모델에서 직접 — 전체 정밀도 I_B, com offset."""
    import mujoco
    m, d = load_model()
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key))
    mujoco.mj_forward(m, d)
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    roots = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_hip_yaw", "right_hip_yaw")]

    def in_leg(b):
        while b > 0:
            if b in roots:
                return True
            b = m.body_parentid[b]
        return False
    bs = [b for b in range(1, m.nbody) if not in_leg(b) and m.body_mass[b] > 0]
    mass = m.body_mass[bs].sum()
    c = (m.body_mass[bs, None] * d.xipos[bs]).sum(0) / mass
    I = np.zeros((3, 3))
    for b in bs:
        R = d.ximat[b].reshape(3, 3)
        o = d.xipos[b] - c
        I += R @ np.diag(m.body_inertia[b]) @ R.T + m.body_mass[b] * (o @ o * np.eye(3) - np.outer(o, o))
    R_WT = d.xmat[base].reshape(3, 3)
    psi = np.arctan2(R_WT[1, 0], R_WT[0, 0])
    Rb = M.Rz(psi)
    sites = np.array([d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)]
                      for s in ("left_foot_contact_site", "right_foot_contact_site")])
    return float(mass), Rb.T @ (c - d.xpos[base]), Rb.T @ I @ Rb, d.xpos[base].copy(), sites


class Clock:
    def __init__(self, t0):
        self.t0 = t0

    def tk(self, k):
        return self.t0 + float(k) * DT


def step_list(stance_tbl, min_scale=None):
    ms = np.ones((len(stance_tbl), 2)) if min_scale is None else min_scale
    return [SimpleNamespace(stance=np.asarray(s, dtype=bool), min_scale=np.asarray(m_, dtype=float))
            for s, m_ in zip(stance_tbl, ms)]


def nominal_table(t0, t=None):
    """GaitScheduler c(side, t0 + k dt) (spec 02 §5.1-5.2) + step 0 = c(side, t)."""
    cycle, stance = float(CFG.timing.cycle), float(CFG.timing.stance)
    frac = stance / cycle
    tbl = np.zeros((N, 2), dtype=bool)
    for k in range(N):
        tk = t0 + k * DT
        for leg, phi in ((0, 0.5), (1, 0.0)):
            p = np.fmod((tk - t0) / cycle + phi, 1.0)
            tbl[k, leg] = 0.0 <= p < frac
    if t is not None:
        for leg, phi in ((0, 0.5), (1, 0.0)):
            p = np.fmod((t - t0) / cycle + phi, 1.0)
            tbl[0, leg] = 0.0 <= p < frac
    return tbl


# ===========================================================================
# 1. spec 02 §13 worked example
# ===========================================================================
def section_worked_example():
    print("\n=== 1. spec 02 §13 worked example (psi 0.3, p_ref [0,0,0.62]) ===")
    mass, com_off, I_B, _, _ = reduced_body_from_model("stand")
    check("reduced body at 'stand' = spec 01 §4", abs(mass - MASS) < 1e-6 and np.abs(I_B - I_SPEC01).max() < 1e-6,
          f"mass {mass:.6f} max|dI| {np.abs(I_B - I_SPEC01).max():.2e} com_off {com_off}")
    psi = np.full(N, 0.3)
    foot_L, foot_R = np.array([0.02, 0.076, 0.0]), np.array([-0.05, -0.076, 0.0])
    p_ref = np.array([0.0, 0.0, 0.62])
    r = np.tile(np.stack([foot_L - p_ref, foot_R - p_ref])[None], (N, 1, 1))
    A_c, B_c, I_k = M.continuous_matrices(psi, r, I_B, mass)
    A_d, B_d = M.discretize_zoh(A_c, B_c, DT)

    # 독립 구현 (worked_example.py 의 식 그대로)
    def skew(v):
        return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    Rk = M.Rz(0.3)
    Ikinv = Rk @ np.linalg.inv(I_B) @ Rk.T
    Ac = np.zeros((13, 13)); Ac[0:3, 6:9] = Rk.T; Ac[3:6, 9:12] = np.eye(3); Ac[9:12, 12] = [0, 0, 1]
    Bc = np.zeros((13, 12))
    Bc[6:9, 0:3] = Ikinv @ skew(r[0, 0]); Bc[6:9, 3:6] = Ikinv @ skew(r[0, 1])
    Bc[6:9, 6:9] = Ikinv; Bc[6:9, 9:12] = Ikinv; Bc[9:12, 0:3] = np.eye(3) / mass; Bc[9:12, 3:6] = np.eye(3) / mass
    A2 = Ac @ Ac
    Ad = np.eye(13) + DT * Ac + 0.5 * DT * DT * A2
    Bd = DT * Bc + 0.5 * DT * DT * (Ac @ Bc) + (DT ** 3 / 6.0) * (A2 @ Bc)
    e_ind = max(np.abs(A_c[0] - Ac).max(), np.abs(B_c[0] - Bc).max(), np.abs(A_d[0] - Ad).max(), np.abs(B_d[0] - Bd).max())
    check("A_c,B_c,A_d,B_d == independent numpy (worked_example.py formulas)", e_ind < 1e-9, f"max|d| {e_ind:.2e}")

    printed = {
        "I_k^-1": (B_c[0, 6:9, 6:9], [[2.28689, -1.148424, -0.03103], [-1.148424, 5.577205, 0.006429], [-0.03103, 0.006429, 2.484747]]),
        "B_c[6:9,0:3]": (B_c[0, 6:9, 0:3], [[0.714381, 1.417251, 0.196772], [-3.458356, -0.711894, -0.198824], [-0.192827, 0.030456, -0.002487]]),
        "A_d[0:3,6:9]": (A_d[0, 0:3, 6:9], [[0.019107, 0.00591, 0], [-0.00591, 0.019107, 0], [0, 0, 0.02]]),
        "B_d[6:9,0:3]": (B_d[0, 6:9, 0:3], [[0.014288, 0.028345, 0.003935], [-0.069167, -0.014238, -0.003976], [-0.003857, 0.000609, -0.00005]]),
        "B_d[0:3,0:3]": (B_d[0, 0:3, 0:3], [[-0.000068, 0.000229, 0.000026], [-0.000703, -0.00022, -0.00005], [-0.000039, 0.000006, 0]]),
        "B_d[9:12,0:3]": (B_d[0, 9:12, 0:3], 0.001406 * np.eye(3)),
        "B_d[3:6,0:3]": (B_d[0, 3:6, 0:3], 0.000014 * np.eye(3)),
        "A_d[3:6,12]": (A_d[0, 3:6, 12], [0, 0, 0.0002]),
        "A_d[9:12,12]": (A_d[0, 9:12, 12], [0, 0, 0.02]),
    }
    worst = max(np.abs(np.asarray(v) - np.asarray(p)).max() for v, p in printed.values())
    check("worked-example numbers vs spec 02 §13 (printed to 6 decimals)", worst < 1e-6, f"max|d| {worst:.2e}")
    cubic = np.linalg.norm(A_c[0] @ A_c[0] @ B_c[0])
    check("||A_c^2 B_c|| == 0 (cubic ZOH term vanishes)", cubic == 0.0, f"{cubic}")

    tmpl = M.foot_constraint_template(CFG)
    CW = M.yaw_rotated_block(tmpl, 0.3)
    c, s = np.cos(0.3), np.sin(0.3)
    CW_printed = np.array([[c, s, -1, 0, 0, 0], [-c, -s, -1, 0, 0, 0], [-s, c, -1, 0, 0, 0], [s, -c, -1, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0], [0, 0, -1, 0, 0, 0], [0, 0, -0.01, c, s, 0], [0, 0, -0.01, -c, -s, 0],
                           [0, 0, -0.065, -s, c, 0], [0, 0, -0.065, s, -c, 0], [0, 0, -0.0657, 0, 0, 1], [0, 0, -0.0657, 0, 0, -1]])
    blk = np.kron(np.eye(2), M.Rz(0.3).T)
    e_cw = max(np.abs(CW - CW_printed).max(), np.abs(CW - tmpl @ blk).max())
    check("C_W(yaw 0.3) == spec 02 §9.3 table == C_F·blkdiag(Rz^T,Rz^T)", e_cw < 1e-12, f"max|d| {e_cw:.2e}")

    # 들어올림: C++ 루프 그대로 (MPCFormulation.cpp:92-106) vs 배치 구현, 그리고 직접 시뮬레이션
    rng = np.random.default_rng(0)
    psi_r = 0.3 + 0.02 * np.arange(N)
    r_r = rng.normal(0, 0.1, (N, 2, 3)) + np.array([0, 0, -0.8])
    A_c, B_c, _ = M.continuous_matrices(psi_r, r_r, I_B, mass)
    A_d, B_d = M.discretize_zoh(A_c, B_c, DT)
    A_qp, B_qp = M.lift(A_d, B_d)
    Aq = np.zeros_like(A_qp); Bq = np.zeros_like(B_qp)
    pre = np.eye(13)
    for k in range(N):
        pre = A_d[k] @ pre
        Aq[13 * k:13 * k + 13] = pre
    for j in range(N):
        L = B_d[j].copy()
        for k in range(j, N):
            Bq[13 * k:13 * k + 13, 12 * j:12 * j + 12] = L
            if k + 1 < N:
                L = A_d[k + 1] @ L
    e_lift = max(np.abs(A_qp - Aq).max(), np.abs(B_qp - Bq).max())
    check("lift (batched diagonals) == C++ loop order, bitwise", e_lift == 0.0, f"max|d| {e_lift:.2e}")
    x0 = np.concatenate([rng.normal(0, 0.05, 12), [-9.81]])
    U = rng.normal(0, 50, 12 * N)
    x = x0.copy(); err = 0.0
    X = A_qp @ x0 + B_qp @ U
    for k in range(N):
        x = A_d[k] @ x + B_d[k] @ U[12 * k:12 * k + 12]
        err = max(err, np.abs(X[13 * k:13 * k + 13] - x).max())
    check("X = A_qp x0 + B_qp U reproduces x_{k+1} = A_d x_k + B_d u_k", err < 1e-9, f"max|d| {err:.2e}")

    # cost: 0.5 U'HU + q'U + const == Σ (x_{k+1}-xref)' Q_k (.) + u' R u
    ctl = M.ConvexMPC(CFG)
    X_ref = rng.normal(0, 0.1, (N, 13)); X_ref[:, 2] = psi_r; X_ref[:, 12] = -9.81
    H, q = ctl.build_cost(A_qp, B_qp, x0, X_ref.reshape(-1), "walking")
    Qd, Rd = ctl.Q_diag["walking"], ctl.R_diag["walking"]

    def J(Uv):
        Xv = A_qp @ x0 + B_qp @ Uv
        tot = 0.0
        for k in range(N):
            T = np.eye(13); RbW = M.Rz(X_ref[k, 2]).T
            T[3:6, 3:6] = T[6:9, 6:9] = T[9:12, 9:12] = RbW
            e = Xv[13 * k:13 * k + 13] - X_ref[k]
            tot += e @ (T.T @ np.diag(Qd) @ T) @ e + Uv[12 * k:12 * k + 12] @ np.diag(Rd) @ Uv[12 * k:12 * k + 12]
        return tot
    U1, U2 = rng.normal(0, 30, 12 * N), rng.normal(0, 30, 12 * N)
    lhs = J(U1) - J(U2)
    rhs = (0.5 * U1 @ H @ U1 + q @ U1) - (0.5 * U2 @ H @ U2 + q @ U2)
    rel = abs(lhs - rhs) / abs(lhs)
    check("QP objective == Σ e'Q_k e + u'Ru (yaw-rotated Q_k, no terminal)", rel < 1e-9,
          f"ΔJ {lhs:.6e} vs {rhs:.6e} rel {rel:.1e}; H sym {np.abs(H - H.T).max():.1e}")

    # 고정 패턴
    check("sparsity: P nnz = 300·301/2, A nnz = 76·N, shapes", ctl.P_pattern.nnz == 45150 and ctl.A_pattern.nnz == 76 * N
          and ctl.A_pattern.shape == (900, 300), f"P {ctl.P_pattern.nnz} A {ctl.A_pattern.nnz} {ctl.A_pattern.shape}")
    tbl = nominal_table(0.0)
    blkA, l, u, sig, _ = ctl.build_constraints(step_list(tbl), [0.3, -0.2])
    Ax = blkA.reshape(-1)[ctl._A_gather]
    import scipy.sparse as sp
    A_sparse = sp.csc_matrix((Ax, ctl.A_pattern.indices, ctl.A_pattern.indptr), shape=ctl.A_pattern.shape).toarray()
    A_dense = np.zeros((900, 300))
    for k in range(N):
        A_dense[24 * k:24 * k + 24, 12 * k:12 * k + 12] = blkA[k, :24]
        A_dense[600 + 12 * k:600 + 12 * k + 12, 12 * k:12 * k + 12] = blkA[k, 24:]
    # dense 블록 자체를 C++ directConstraintValue 식으로 다시 만든 것과 비교
    ref_dense = np.zeros((900, 300))
    CWL, CWR = M.yaw_rotated_block(tmpl, 0.3), M.yaw_rotated_block(tmpl, -0.2)
    for k in range(N):
        if tbl[k, 0]:
            ref_dense[24 * k:24 * k + 12, 12 * k + M.LEFT_MAP] = CWL
        else:
            for i in (0, 1, 2, 6, 7, 8):
                ref_dense[600 + 12 * k + i, 12 * k + i] = 1
        if tbl[k, 1]:
            ref_dense[24 * k + 12:24 * k + 24, 12 * k + M.RIGHT_MAP] = CWR
        else:
            for i in (3, 4, 5, 9, 10, 11):
                ref_dense[600 + 12 * k + i, 12 * k + i] = 1
    e_pat = max(np.abs(A_sparse - A_dense).max(), np.abs(A_dense - ref_dense).max())
    check("A values in fixed pattern == directConstraintValue (no entry lost)", e_pat == 0.0, f"max|d| {e_pat}")
    ub = u[:600].reshape(N, 24)
    ok_b = (np.all(ub[tbl[:, 0], 4] == 1500) and np.all(ub[tbl[:, 0], 5] == -5) and np.all(ub[~tbl[:, 0], :12] == 0)
            and np.all(ub[tbl[:, 1], 16] == 1500) and np.all(ub[tbl[:, 1], 17] == -5) and np.all(np.isneginf(l[:600]))
            and np.all(l[600:] == 0) and np.all(u[600:] == 0))
    check("C_bound: Fz ≤ 1500, −Fz ≤ −minScale·5 on stance, 0 on swing; eq l=u=0", bool(ok_b),
          f"L stance k: {np.flatnonzero(tbl[:, 0]).tolist()}")


# ===========================================================================
# 2. reference trajectory (spec 03 §4.3 / §8 compact python)
# ===========================================================================
def section_reference():
    print("\n=== 2. build_reference vs spec 03 §8 compact reference ===")

    def Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def spec_build(seed, cmd, t0, foot_L, foot_R):     # spec 03 §8 build_reference, verbatim
        uB = np.array([cmd['x_dot'], cmd['y_dot'], 0.0]); h = cmd['h']; psid = cmd['psi_dot']
        p = seed[3:6].copy(); p[2] += h; euler = seed[0:3].copy(); psi = seed[2]
        X = np.zeros((N, 13)); rL = np.zeros((N, 3)); rR = np.zeros((N, 3)); PSI = np.zeros(N); TK = np.zeros(N)
        for k in range(N):
            tk = t0 + k * DT
            if k > 0:
                psi += psid * DT
            v = Rz(psi) @ uB
            if k > 0:
                p = p + Rz(psi) @ np.array([uB[0], uB[1], 0.0]) * DT; p[2] = seed[5] + h
            TK[k] = tk; PSI[k] = psi; rL[k] = foot_L - p; rR[k] = foot_R - p
            X[k] = [euler[0], euler[1], psi, p[0], p[1], p[2], 0, 0, psid, v[0], v[1], v[2], seed[12]]
        return X, rL, rR, PSI, TK
    seed = np.array([0.01, -0.02, 7.1, 0.3, -0.2, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -9.81])
    cmd = dict(x_dot=0.6, y_dot=-0.3, psi_dot=1.3, h=0.02)
    fc = SimpleNamespace(x_dot=0.6, y_dot=-0.3, psi_dot=1.3, body_height_offset_m=0.02)
    feet = np.array([[0.4, -0.1, -0.005], [0.35, -0.3, -0.005]])
    ref = M.build_reference(fc, seed, feet, Clock(16.002), CFG)
    X, rL, rR, PSI, TK = spec_build(seed, cmd, 16.002, feet[0], feet[1])
    e = max(np.abs(ref.X_ref - X).max(), np.abs(ref.r[:, 0] - rL).max(), np.abs(ref.r[:, 1] - rR).max(),
            np.abs(ref.psi - PSI).max(), np.abs(ref.tk - TK).max())
    check("build_reference == spec 03 §8 (turn+lateral+height offset)", e < 1e-12, f"max|d| {e:.1e}")
    ok = (ref.X_ref[0, 2] == seed[2] and abs(ref.X_ref[24, 2] - (seed[2] + 24 * 1.3 * DT)) < 1e-12
          and np.all(ref.X_ref[:, 5] == seed[5] + 0.02) and np.all(ref.X_ref[:, 8] == 1.3)
          and np.all(ref.X_ref[0, 3:5] == seed[3:5]) and np.all(ref.X_ref[:, 6:8] == 0))
    check("ref quirks: k=0 yaw = seed, z = seed z + h every step, wz = psi_dot, xy(k=0) = seed", bool(ok),
          f"psi[24]-psi[0] {ref.psi[24] - ref.psi[0]:.6f}")

    st = SimpleNamespace(foot_R_W=np.stack([M.Rz(0.2 + 2 * np.pi), M.Rz(-3.0)]))
    y_act = M.mpc_foot_yaw(st, 0, True, 6.4)        # measured 0.2 (wrapped) lifted near 6.4 → 0.2+2π
    y_in = M.mpc_foot_yaw(st, 1, False, 1.234)
    Rdeg = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])   # foot x axis vertical → exception → touchdown yaw
    y_deg = M.mpc_foot_yaw(SimpleNamespace(foot_R_W=np.stack([Rdeg, Rdeg])), 0, True, 0.7)
    check("mpc_foot_yaw: lift near touchdown / inactive → touchdown / degenerate → touchdown",
          abs(y_act - (0.2 + 2 * np.pi)) < 1e-12 and y_in == 1.234 and y_deg == 0.7, f"{y_act:.6f} {y_in} {y_deg}")


# ===========================================================================
# 3. standing QP at 'stand' (synthetic inputs)
# ===========================================================================
def standing_problem(offset=True):
    """keyframe 'stand' 과 같은 합성 입력. 발 site (0.022359, ±0.080162, 0) — 전신 CoM 바로 아래,
    reduced COM (0.015306, 0.002857, 0.796046) = base 0.679472 + com_off. standing: foot z = −0.005 (MC:1396-1400),
    seed = [euler 0, nominalPosition = 발 평균 xy, nominalHeight = 0.796046]."""
    feet = np.array([[0.022359, 0.080162, -0.005], [0.022359, -0.080162, -0.005]])
    com = np.array([0.015306, 0.002857, 0.796046])
    x0 = np.zeros(13); x0[12] = -9.81
    x0[3:6] = com if offset else np.array([0.022359, 0.0, 0.796046])
    seed = x0.copy(); seed[0:3] = 0.0; seed[3:5] = feet[:, :2].mean(0); seed[5] = 0.796046
    ref = M.build_reference(SimpleNamespace(x_dot=0, y_dot=0, psi_dot=0, body_height_offset_m=0), seed, feet, Clock(0.0), CFG)
    reduced = SimpleNamespace(mass=MASS, inertia_B=I_SPEC01, com_offset_B=np.array([0.015306, 0.002857, 0.116574]))
    steps = step_list(np.ones((N, 2), dtype=bool))
    return x0, ref, steps, reduced


def constraint_violation(ctl, U, steps, foot_yaw):
    blk, l, u, _, _ = ctl.build_constraints(steps, foot_yaw)
    Ax = blk.reshape(-1)[ctl._A_gather]
    import scipy.sparse as sp
    A = sp.csc_matrix((Ax, ctl.A_pattern.indices, ctl.A_pattern.indptr), shape=ctl.A_pattern.shape)
    z = A @ U
    return float(max(np.max(l - z), np.max(z - u), 0.0))


def section_standing():
    print("\n=== 3. standing QP at 'stand' (synthetic: mass 14.227468, spec 01 inertia, feet (0.022359, ±0.080162)) ===")
    mg = MASS * 9.81
    # case: (이름, x0 com − reference, sum 허용, split 허용)
    cases = [("reduced COM on reference (over feet midpoint)", (0.0, 0.0), 0.01, 0.01),
             ("x only: COM 7.05 mm behind feet (stand x)", (0.015306 - 0.022359, 0.0), 0.10, 0.01),
             ("keyframe 'stand' as is: x −7.05 mm, y +2.857 mm", (0.015306 - 0.022359, 0.002857), 0.10, None)]
    for name, (dx, dy), tol_sum, tol_split in cases:
        x0, ref, steps, red = standing_problem(False)
        x0[3] += dx
        x0[4] += dy
        res = {}
        for solver in ("osqp", "quadprog"):
            ctl = M.ConvexMPC(CFG, solver=solver)
            u0, info = ctl.solve(x0, ref, steps, [0.0, 0.0], red, "standing")
            viol = constraint_violation(ctl, info["U"].reshape(-1), steps, [0.0, 0.0])
            res[solver] = (u0, info, viol)
        for solver, (u0, info, viol) in res.items():
            fzL, fzR = u0[2], u0[5]
            s = fzL + fzR
            split = (fzL - fzR) / s
            ok = (not info["fallback"]) and abs(s - mg) / mg < tol_sum and viol < 0.05
            if tol_split is not None:
                ok = ok and abs(split) < tol_split
            extra = ""
            if tol_split is None:
                # roll 모멘트 균형: 발의 Fy (−y 로 COM 을 되돌리는 힘) 가 만드는 −r_z·Fy 를 Fz 차이 r_y·Fz 가 상쇄
                r = ref.r[0]
                tau_x = sum(r[i, 1] * u0[3 * i + 2] - r[i, 2] * u0[3 * i + 1] for i in (0, 1)) + u0[6] + u0[9]
                extra = (f" | split explained: Fy {u0[[1, 4]].round(2)} × h {-r[0, 2]:.3f} → roll τ from Fy "
                         f"{sum(-r[i, 2] * u0[3 * i + 1] for i in (0, 1)):+.2f} N·m balanced by Fz diff "
                         f"{sum(r[i, 1] * u0[3 * i + 2] for i in (0, 1)):+.2f}; net τx about COM {tau_x:+.2f} N·m "
                         f"(Q_py 90000 on a 2.9 mm y error)")
            check(f"standing QP [{solver}] {name}", ok,
                  f"FzL {fzL:.3f} FzR {fzR:.3f} sum {s:.3f} (m g {mg:.3f}, {100 * (s / mg - 1):+.2f}%) "
                  f"split {100 * split:+.2f}%  viol {viol:.1e} N  {info['status']} it {info['iters']}{extra}")
        d = np.abs(res["osqp"][0] - res["quadprog"][0]).max()
        print(f"      osqp vs quadprog u0 max|d| {d:.3f} N; u0(quadprog) {np.round(res['quadprog'][0], 3).tolist()}")


# ===========================================================================
# 4. OSQP vs quadprog, 3 random stance patterns (walking weights)
# ===========================================================================
def random_walking_problem(rng, tbl, t0=0.0):
    ms = np.ones((N, 2))
    ms[0] = rng.uniform(0, 1, 2)                          # step-0 ramp min scale
    steps = step_list(tbl, ms)
    x0 = np.zeros(13); x0[12] = -9.81
    x0[0:3] = rng.normal(0, [0.03, 0.03, 0.1]); x0[3:6] = [0.0, 0.0, 0.79] + rng.normal(0, 0.01, 3)
    x0[6:9] = rng.normal(0, 0.2, 3); x0[9:12] = [0.5, 0.0, 0.0] + rng.normal(0, 0.1, 3)
    yaw = x0[2]
    seed = x0.copy(); seed[0:2] = 0.0; seed[5] = 0.796046
    feet = np.array([[0.1, 0.08, -0.005], [-0.05, -0.08, -0.005]])
    feet[:, :2] = (M.Rz(yaw)[:2, :2] @ feet[:, :2].T).T + x0[3:5]
    fc = SimpleNamespace(x_dot=0.6, y_dot=rng.uniform(-0.3, 0.3), psi_dot=rng.uniform(-1.3, 1.3), body_height_offset_m=0.0)
    ref = M.build_reference(fc, seed, feet, Clock(t0), CFG)
    red = SimpleNamespace(mass=MASS, inertia_B=I_SPEC01)
    foot_yaw = [yaw + rng.normal(0, 0.1), yaw + rng.normal(0, 0.1)]
    return x0, ref, steps, red, foot_yaw


def section_osqp_vs_quadprog():
    print("\n=== 4. OSQP (reference settings) vs quadprog (exact) — 3 random stance patterns ===")
    rng = np.random.default_rng(7)
    for trial in range(3):
        t = rng.uniform(0, 0.5)
        tbl = nominal_table(0.0, t)
        if trial == 2:                                     # 완전 무작위 패턴
            tbl = rng.uniform(size=(N, 2)) < 0.7
            tbl[0] = [True, True]
        x0, ref, steps, red, fy = random_walking_problem(rng, tbl)
        c_os = M.ConvexMPC(CFG, "osqp")
        c_qp = M.ConvexMPC(CFG, "quadprog")
        c_tight = M.ConvexMPC(CFG, "osqp", osqp_settings=dict(eps_abs=1e-9, eps_rel=1e-9, max_iter=200000,
                                                               check_dualgap=1))
        u_os, i_os = c_os.solve(x0, ref, steps, fy, red, "walking")
        u_qp, i_qp = c_qp.solve(x0, ref, steps, fy, red, "walking")
        u_ti, i_ti = c_tight.solve(x0, ref, steps, fy, red, "walking")
        U_os, U_qp, U_ti = (i["U"].reshape(-1) for i in (i_os, i_qp, i_ti))
        scale = np.abs(U_qp).max()
        d_os = np.abs(U_os - U_qp).max() / scale
        d0 = np.abs(u_os - u_qp).max() / np.abs(u_qp).max()
        d_ti = np.abs(U_ti - U_qp).max() / scale
        obj_gap = abs(i_os["obj"] - i_qp["obj"]) / abs(i_qp["obj"])
        v_os = constraint_violation(c_os, U_os, steps, fy)
        sig = "".join("LRB-"[(0 if a and not b else 1 if b and not a else 2 if a and b else 3)] for a, b in tbl)
        # (1) 같은 QP 인가: OSQP 를 1e-9 까지 풀면 quadprog 와 일치해야 한다
        check(f"same QP: tight OSQP (eps 1e-9) == quadprog, pattern {trial} [{sig}]", d_ti < 1e-4 and not i_ti["fallback"],
              f"|dU|/max|U| {d_ti:.1e} (it {i_ti['iters']})")
        # (2) reference 설정 OSQP (eps 1e-3, max_iter 200): OSQP 자신의 종료 기준 만족 + 목적함수 1e-3 이내
        ok = (not i_os["fallback"]) and i_os["status_val"] in (1, 2) and obj_gap < 1e-3 and v_os < 0.5
        check(f"reference-settings OSQP within OSQP tolerance, pattern {trial}", ok,
              f"{i_os['status']} it {i_os['iters']} prim_res {i_os['prim_res']:.1e} dual_res {i_os['dual_res']:.1e}: "
              f"obj gap {obj_gap:.1e}, viol {v_os:.1e} N | info: |dU|/max|U| {d_os:.1e}, |du0|/max|u0| {d0:.1e}, "
              f"Fz0 osqp {u_os[[2, 5]].round(2)} vs exact {u_qp[[2, 5]].round(2)}")


# ===========================================================================
# 5. warm/cold bookkeeping, shifted warm start, fallback; timing
# ===========================================================================
def section_solver_logic():
    print("\n=== 5. OSQP cold/warm logic, shifted warm start, retry + fallback ===")
    rng = np.random.default_rng(3)
    tbl = nominal_table(0.0, 0.01)
    x0, ref, steps, red, fy = random_walking_problem(rng, tbl)
    ctl = M.ConvexMPC(CFG, "osqp")
    _, i1 = ctl.solve(x0, ref, steps, fy, red, "walking")
    warm_expected = np.concatenate([i1["U"].reshape(-1)[12:], i1["U"].reshape(-1)[-12:]])
    ok_shift = np.array_equal(ctl._warm, warm_expected)
    _, i2 = ctl.solve(x0 + 1e-3, ref, steps, fy, red, "walking")
    tbl2 = tbl.copy(); tbl2[0] = [False, True]
    _, i3 = ctl.solve(x0, ref, step_list(tbl2), fy, red, "walking")
    _, i4 = ctl.solve(x0, ref, step_list(tbl2), fy, red, "walking")
    ok = i1["cold"] and not i1["sig_changed"] and not i2["cold"] and i3["cold"] and i3["sig_changed"] and not i4["cold"]
    check("cold on first solve & on signature change, warm otherwise", bool(ok),
          f"cold {[i['cold'] for i in (i1, i2, i3, i4)]} iters {[i['iters'] for i in (i1, i2, i3, i4)]}")
    check("shifted warm start = [U[12:], U[-12:]] after success", bool(ok_shift))

    # 실패 경로: max_iter 1 → MaxIterReached → fresh cold retry → 다시 실패 → fallback
    bad = M.ConvexMPC(CFG, "osqp", osqp_settings=dict(max_iter=1), verbose_errors=False)
    u_fb, i_fb = bad.solve(x0, ref, step_list(tbl2), fy, red, "walking")
    exp = np.zeros(12); exp[5] = MASS * 9.81            # step 0 = (L swing, R stance) → R 가 전부
    u_fb2, _ = bad.solve(x0, ref, steps, fy, red, "walking", active=np.array([True, True]))
    exp2 = np.zeros(12); exp2[2] = exp2[5] = MASS * 9.81 / 2
    check("failure → retry → fallback Fz = m|g|/nStance on active legs, warm start untouched",
          i_fb["fallback"] and i_fb["retry"] is True and np.allclose(u_fb, exp) and np.allclose(u_fb2, exp2)
          and not bad._has_prev,
          f"u_fb Fz {u_fb[[2, 5]].round(3)} / {u_fb2[[2, 5]].round(3)}  err '{i_fb['error']}'")

    # 성능: 걷는 것 같은 순서 (14 ms 간격, cycle 0.5 s, 약 8 cold / s)
    print("\n=== 6. timing (walking-like sequence, 14 ms solve spacing) ===")
    ctl = M.ConvexMPC(CFG, "osqp", verbose_errors=False)
    times, builds, solves, colds, iters, fails = [], [], [], 0, [], 0
    t0 = 0.0
    x = np.zeros(13); x[12] = -9.81; x[5] = 0.79
    for i in range(300):
        t = i * 0.014
        while t - t0 >= 0.5:
            t0 += 0.5
        tbl = nominal_table(t0, t)
        x[3] += 0.6 * 0.014; x[9] = 0.6 + 0.05 * np.sin(20 * t); x[1] = 0.02 * np.sin(12 * t); x[6:9] = 0.1 * np.sin(15 * t)
        x[2] += 0.3 * 0.014
        seed = x.copy(); seed[0:2] = 0; seed[5] = 0.796046
        feet = np.array([[x[3] + 0.1, x[4] + 0.08, -0.005], [x[3] - 0.05, x[4] - 0.08, -0.005]])
        ref = M.build_reference(SimpleNamespace(x_dot=0.6, y_dot=0, psi_dot=0.3, body_height_offset_m=0), seed, feet,
                                Clock(t0), CFG)
        u0, info = ctl.solve(x, ref, step_list(tbl), [x[2], x[2]], SimpleNamespace(mass=MASS, inertia_B=I_SPEC01), "walking")
        times.append(info["total_ms"]); builds.append(info["build_ms"]); solves.append(info["solve_ms"])
        colds += info["cold"]; iters.append(info["iters"]); fails += info["fallback"]
    times, builds, solves = map(np.array, (times, builds, solves))
    print(f"      300 solves: total median {np.median(times):.2f} ms (p90 {np.percentile(times, 90):.2f}, max {times.max():.2f}); "
          f"build median {np.median(builds):.2f} ms; osqp median {np.median(solves):.2f} ms; "
          f"cold {colds} ({colds / (300 * 0.014):.1f}/s); iters median {np.median(iters):.0f}; fallbacks {fails}")
    check("median MPC solve (build+OSQP) time reported", np.median(times) < 50.0, f"{np.median(times):.2f} ms")
    return float(np.median(times))


# ===========================================================================
# 7. leg control
# ===========================================================================
def planar_leg():
    """합성 5-DOF 다리 (sagittal): hip_yaw z, hip_abad x, hip_pitch y, knee y, ankle y. 무릎 굽힘 (+).
    길이는 MIT 대략값. 반환: Jv, Jw (3,5), 관절 위치, 발 위치."""
    q = np.array([0.0, 0.0, -0.4677, 0.997, -0.5292])
    hip = np.array([0.0, 0.08, 0.0])
    axes_local = [np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 1.0, 0]), np.array([0, 1.0, 0])]
    L1, L2, L3 = 0.30, 0.30, 0.04

    def Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    R = np.eye(3)
    pos = [hip.copy(), hip.copy(), hip.copy()]
    R = Ry(q[2])
    knee = hip + R @ np.array([0, 0, -L1])
    R = R @ Ry(q[3])
    ankle = knee + R @ np.array([0, 0, -L2])
    R = R @ Ry(q[4])
    foot = ankle + R @ np.array([0.03, 0, -L3])
    joints = pos + [knee, ankle]
    Jv = np.zeros((3, 5)); Jw = np.zeros((3, 5))
    for j, (pj, a) in enumerate(zip(joints, axes_local)):
        Jw[:, j] = a
        Jv[:, j] = np.cross(a, foot - pj)
    return q, Jv, Jw, knee, foot, R


def section_leg_control():
    print("\n=== 7. leg control ===")
    cfg = CFG
    # (a) swing Kp formula with a synthetic M/J that has Λ = spec 04 §14.6 sample
    lam_sample = np.array([[0.440, 0.024, 0.274], [0.024, 0.525, 0.154], [0.274, 0.154, 1.968]])
    rng = np.random.default_rng(1)
    Mm = np.diag([0.0302, 0.1106, 0.1803, 0.0335, 0.00099]) + 0.001 * np.eye(5)
    Lm = np.linalg.cholesky(Mm)
    Linv_lam = np.linalg.cholesky(np.linalg.inv(lam_sample))          # Jv M^-1 Jv' = inv(Λ)
    Qo, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    Jv = Linv_lam @ Qo[:3] @ Lm.T                                      # Jv M^-1 Jv' = Lλ Qo3 Qo3' Lλ' = inv(Λ)
    dyn = SimpleNamespace(Jv=Jv, Jw=np.zeros((3, 5)), M=Mm, JvDot_qd=np.zeros(3), bias=np.zeros(5))
    lam = LC.apparent_inertia(Jv, Mm)
    kp = np.diag(LC.swing_kp(dyn, cfg))
    kp_spec = np.array([10025, 11975, 23810])
    check("swing Λ / Kp = ω_n²·diag(Λ) for synthetic M/J with Λ = spec 04 §14.6",
          np.abs(lam - lam_sample).max() < 1e-6 and np.all(np.abs(kp / kp_spec - 1) < 0.005),
          f"diag Λ {np.diag(lam).round(4)} Kp {kp.round(0)} vs spec {kp_spec}")

    # (b) real model. spec 04 §14.6 의 표본은 이름과 달리 q = 0 (완전히 편 다리 → z 특이, Λ_zz ≈ 1e9) 이 아니라
    #     probe2.py 의 마지막 자세 = 다리 [0,0,−0.65,0.80,−0.15], 팔 [0,0,0,−1.65], 발 BODY 원점, armature 0 이다.
    wn2 = np.array([151.0, 151.0, 110.0]) ** 2
    try:
        import mujoco
        jl = ["a06_left_hip_yaw", "a07_left_hip_abad", "a08_left_hip_pitch", "a09_left_knee", "a10_left_ankle"]
        jr_ = ["a01_right_hip_yaw", "a02_right_hip_abad", "a03_right_hip_pitch", "a04_right_knee", "a05_right_ankle"]
        ja = ["a15_left_shoulder_pitch", "a16_left_shoulder_abad", "a17_left_shoulder_yaw", "a18_left_elbow",
              "a11_right_shoulder_pitch", "a12_right_shoulder_abad", "a13_right_shoulder_yaw", "a14_right_elbow"]

        def lam_at(key_or_legq, armature: bool, point: str):
            m, d = load_model()
            if not armature:
                m.dof_armature[:] = 0.0
            if isinstance(key_or_legq, str):
                mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key_or_legq))
            else:
                mujoco.mj_resetData(m, d)
                for names, vals in ((jl, key_or_legq), (jr_, key_or_legq), (ja, [0, 0, 0, -1.65] * 2)):
                    for n_, v in zip(names, vals):
                        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n_)]] = v
            mujoco.mj_forward(m, d)
            dofs = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n_)] for n_ in jl]
            Mf = np.zeros((m.nv, m.nv))
            mujoco.mj_fullM(m, d, Mf)
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            if point == "body":
                mujoco.mj_jacBody(m, d, jp, jr, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_foot"))
            else:
                mujoco.mj_jacSite(m, d, jp, jr, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot_contact_site"))
            return LC.apparent_inertia(jp[:, dofs], Mf[np.ix_(dofs, dofs)])
        lb = lam_at([0, 0, -0.65, 0.80, -0.15], False, "body")
        ok = np.abs(lb - lam_sample).max() < 2e-3
        check("real model: Λ at spec 04 §14.6 sample pose (legs [0,0,−.65,.8,−.15], foot origin, armature 0)", bool(ok),
              f"Λ diag {np.diag(lb).round(4)} max|dΛ| {np.abs(lb - lam_sample).max():.1e}; Kp {(wn2 * np.diag(lb)).round(0)}")
        for arm in (False, True):
            L = lam_at("stand", arm, "site")
            print(f"      info: 'stand' keyframe, site, armature={arm}: Λ diag {np.diag(L).round(4)} → Kp {(wn2 * np.diag(L)).round(0)} N/m")
    except Exception as exc:
        check("real model Λ sample", False, f"{type(exc).__name__}: {exc}")

    # (c) stance torque sign: 굽힌 다리 + 지면이 발을 위로 미는 힘 (GRF on robot +z) → 다리 명령 −F
    q, Jv, Jw, knee, footp, _ = planar_leg()
    F_grf = np.array([0.0, 0.0, 70.0])
    tau = LC.stance_torque(SimpleNamespace(Jv=Jv, Jw=Jw), -F_grf, np.zeros(3))
    r_x = footp[0] - knee[0]
    check("stance sign (synthetic bent leg): GRF +z → knee τ < 0 (extension) = r_x·F",
          tau[3] < 0 and abs(tau[3] - r_x * F_grf[2]) < 1e-12,
          f"knee q {q[3]:+.3f} (+ = flexion, foot {-r_x:.3f} m behind knee) → τ_knee {tau[3]:+.3f} N·m = (x_foot−x_knee)·Fz "
          f"= {r_x:+.4f}·{F_grf[2]:.0f}; τ = {tau.round(3)}")
    try:
        import mujoco
        m, d = load_model()
        mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand"))
        mujoco.mj_forward(m, d)
        jn = ["a06_left_hip_yaw", "a07_left_hip_abad", "a08_left_hip_pitch", "a09_left_knee", "a10_left_ankle"]
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jn]
        dofs = [m.jnt_dofadr[j] for j in jid]
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot_contact_site")
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, site)
        dynm = SimpleNamespace(Jv=jp[:, dofs], Jw=jr[:, dofs])
        u = np.zeros(12); u[2] = u[5] = MASS * 9.81 / 2
        tau_m = LC.stance_torque(dynm, -u[0:3], -u[6:9])
        kq = d.qpos[m.jnt_qposadr[jid[3]]]
        check("stance sign (model 'stand', left leg, Fz = m g/2): knee τ < 0 opposes flexion",
              tau_m[3] < 0 and kq > 0, f"knee q {kq:+.3f} τ {tau_m.round(3)}")
        # standing_torque (6×10 결합) == 다리별 stance_torque(−u)
        jnR = ["a01_right_hip_yaw", "a02_right_hip_abad", "a03_right_hip_pitch", "a04_right_knee", "a05_right_ankle"]
        dofsR = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jnR]
        siteR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_foot_contact_site")
        jpR = np.zeros((3, m.nv)); jrR = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jpR, jrR, siteR)
        Jv6 = np.zeros((6, 10)); Jw6 = np.zeros((6, 10))
        Jv6[0:3, 0:5] = jp[:, dofs]; Jw6[0:3, 0:5] = jr[:, dofs]
        Jv6[3:6, 5:10] = jpR[:, dofsR]; Jw6[3:6, 5:10] = jrR[:, dofsR]
        u = rng.normal(0, 20, 12)
        ts = LC.standing_torque(Jv6, Jw6, u)
        tL = LC.stance_torque(SimpleNamespace(Jv=jp[:, dofs], Jw=jr[:, dofs]), -u[0:3], -u[6:9])
        tR = LC.stance_torque(SimpleNamespace(Jv=jpR[:, dofsR], Jw=jrR[:, dofsR]), -u[3:6], -u[9:12])
        e = max(np.abs(ts[0] - tL).max(), np.abs(ts[1] - tR).max())
        check("standing_torque (6×10, −u) == per-leg stance_torque(−u)", e < 1e-12, f"max|d| {e:.1e}")
    except Exception as exc:
        check("stance sign on model", False, f"{exc}")

    # (d) walking stance: ramp α, yaw hold only z and only when α > 0
    q, Jv, Jw, knee, footp, Rf = planar_leg()
    yaw_foot = 0.1
    Rfoot = M.Rz(yaw_foot)
    st = SimpleNamespace(qd_leg=np.zeros((2, 5)), foot_R_W=np.stack([Rfoot, Rfoot]))
    dynl = SimpleNamespace(Jv=Jv, Jw=Jw)
    mz = LC.stance_yaw_hold_moment_z(st, 0, dynl, 0.3, cfg)
    u0 = rng.normal(0, 30, 12)
    t_a0 = LC.walking_stance_torque(dynl, st, 0, u0, 0.0, 0.3, cfg)
    t_a = LC.walking_stance_torque(dynl, st, 0, u0, 0.6, 0.3, cfg)
    t_exp = Jv.T @ (-0.6 * u0[0:3]) + Jw.T @ (-0.6 * u0[6:9] + np.array([0, 0, 0.6 * mz]))
    check("stance yaw hold: m_z = 20·(ψ_td − ψ_foot) − 4·rate, added α·m_z on M.z only; α=0 → zero torque",
          abs(mz - 20 * 0.2) < 1e-12 and np.abs(t_a - t_exp).max() < 1e-12 and np.abs(t_a0).max() == 0.0,
          f"m_z {mz:.4f} (yaw err 0.2 rad)")

    # (e) swing attitude signs: pitched foot (toe down) and yawed foot
    def Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    dyn_id = SimpleNamespace(Jv=np.eye(3, 5), Jw=np.eye(3, 5))
    st_p = SimpleNamespace(qd_leg=np.zeros((2, 5)), foot_R_W=np.stack([Ry(0.1), Ry(0.1)]))    # toe down 0.1 rad
    tp = LC.swing_attitude_torque(st_p, 0, dyn_id, 0.0, cfg)
    st_y = SimpleNamespace(qd_leg=np.zeros((2, 5)), foot_R_W=np.stack([M.Rz(0.2), M.Rz(0.2)]))
    ty = LC.swing_attitude_torque(st_y, 0, dyn_id, 0.5, cfg)
    check("swing attitude: pitch +0.1 (toe down) → −300·0.1 about foot y; yaw 0.2→0.5 → +305·0.3 about z; roll off",
          abs(tp[1] + 30.0) < 1e-9 and abs(tp[0]) < 1e-12 and abs(ty[2] - 91.5) < 1e-9 and abs(ty[0]) + abs(ty[1]) < 1e-12,
          f"τ_pitch {tp[:3].round(4)} τ_yaw {ty[:3].round(4)}")

    # (f) swing torque composition
    Mm5 = np.diag([0.05, 0.12, 0.2, 0.08, 0.05]) + 0.01
    dyn_s = SimpleNamespace(Jv=Jv, Jw=Jw, M=Mm5, JvDot_qd=np.array([0.1, -0.2, 0.3]), bias=np.arange(5.0))
    st_s = SimpleNamespace(qd_leg=np.array([np.linspace(-1, 1, 5)] * 2), foot_R_W=np.stack([np.eye(3)] * 2),
                           foot_pos_W=np.array([footp, footp]), foot_vel_W=np.array([[0.1, 0, 0.2]] * 2))
    p_des, v_des, a_des = footp + [0.01, 0.0, 0.02], np.array([0.3, 0, 0.1]), np.array([1.0, 0, -2.0])
    ts = LC.swing_torque(dyn_s, st_s, 0, p_des, v_des, a_des, 0.0, cfg)
    lam = LC.apparent_inertia(Jv, Mm5)
    Kp = np.diag(np.array([151.0, 151, 110]) ** 2 * np.diag(lam))
    te = Jv.T @ (Kp @ (p_des - footp) + 25 * (v_des - [0.1, 0, 0.2])) + Jv.T @ lam @ (a_des - dyn_s.JvDot_qd) + dyn_s.bias \
        + LC.swing_attitude_torque(st_s, 0, dyn_s, 0.0, cfg)
    check("swing torque = Jvᵀ(Kp e + Kd ė) + Jvᵀ Λ(a − J̇q̇) + bias + attitude", np.abs(ts - te).max() < 1e-9,
          f"τ {ts.round(3)}")

    # (g) arm PD
    st_a = SimpleNamespace(q_arm=np.array([[0.1, 0, 0, -1.6], [0, -0.1, 0, -1.7]]), qd_arm=np.ones((2, 4)))
    qdes = np.array([[0, 0, 0, -1.65]] * 2)
    ta = LC.arm_pd(st_a, qdes, cfg)
    te = 100 * (qdes - st_a.q_arm) - 5 * st_a.qd_arm
    check("arm PD kp 100 kd 5 toward [0,0,0,−1.65]", np.abs(ta - te).max() < 1e-12, f"{ta.round(3).tolist()}")


def main():
    t = time.perf_counter()
    section_worked_example()
    section_reference()
    section_standing()
    section_osqp_vs_quadprog()
    med = section_solver_logic()
    section_leg_control()
    print(f"\n{len(FAILS)} FAIL(s) {FAILS if FAILS else ''}; median MPC solve {med:.2f} ms; ran {time.perf_counter() - t:.1f} s")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
