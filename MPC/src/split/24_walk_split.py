"""분리 — MPC 와 스윙 제어를 reference 처럼 갈라 놓는다 (Q&A 9/23 Q9·Q10 의 '반작용을 다루지 않는' 설계).

09_walk 대비 무엇이 다른가 (새 파일 — 기존 코드 불변, 11_walk_srb_upper.UpperSRBController 를 상속):
  MPC 모델 : I = 상체(다리 제외) 합성 관성,  Θ·ω = 골반 (같은 강체).  병진(p, v, 질량)은 전신 그대로 —
             뉴턴 식은 전신 CoM 에서 정확하다 (11_walk v1 실패 기록).  reference 는 병진도 torso 프록시지만
             그쪽은 스탠스 bias 를 안 넣는 다른 토크 사상과 짝이라 그대로 못 옮긴다.
  스윙 토크: reference OperationalSpaceDynamics::computeSwingLegJointTorque 그대로 —
             τ_다리 = J_다리ᵀ[kp⊙e + kd⊙ė] + J_다리ᵀ Λ_다리 (a_ref − J̇_다리 q̇_다리)   (+ 다리 중력·코리올리는 qfrc_bias)
             kp = ω_n²⊙diag(Λ_다리) (축별 ω_n, 기본 G1 config [100,100,100]),  kd 고정 (기본 [20,20,23]),
             힘 클램프 없음.  J·J̇·M 전부 다리 6 dof 블록만 — 골반 줄은 만들지도 않는다 (Q10).
  발 자세  : 축별 PD (reference G1 config: roll 200/6, pitch 250/18, yaw 100/10).
  MPC 가중 : 기본은 우리 것.  --refq 면 reference G1 walking 가중치 + R 1e-7.  --refhorizon 이면 N=25, dt=0.02.
  뺀 것    : early_td / td_sink / lo_ramp 경로 (권장 구성에 없음), ContactManager (MPC 쪽 처리까지 같이
             해야 해서 오늘 범위 밖 — 9/22 Q13).

사용 (09_walk 와 같은 플래그 + 아래):
  .venv/Scripts/python.exe MPC/src/split/24_walk_split.py --vx 0.5 --seconds 40 [권장 플래그]
  .venv/Scripts/python.exe MPC/src/split/24_walk_split.py --view --vx 0.5 [권장 플래그]
  --wn3 100 100 100    축별 ω_n [rad/s]        --kd3 20 20 23    축별 kd 고정 [N·s/m]
  --fmax 400           스윙 힘 클램프 (기본 없음)  --ourgains        게인만 우리 것 (ω_n=--wn, kd=2ζω_nΛ, 클램프 400)
  --refq  --refhorizon --nojdot(J̇q̇ 항 끔, 비교용)
"""
from __future__ import annotations

import functools
import importlib
import sys

import numpy as np
import mujoco

import mpc_srb
from mpc_srb import rz
import walk_cli

walk = importlib.import_module("09_walk")
up = importlib.import_module("11_walk_srb_upper")

# reference/config/unitree_robots/g1/my_controller.yaml (2026-09-23 확인)
REF_WN3 = (100.0, 100.0, 100.0)
REF_KD3 = (20.0, 20.0, 23.0)
REF_FOOT_PD = dict(roll=(200.0, 6.0), pitch=(250.0, 18.0), yaw=(100.0, 10.0))
OUR_FOOT_PD = dict(roll=(60.0, 5.0), pitch=(60.0, 5.0), yaw=(60.0, 5.0))       # gait.SwingController kp_ori/kd_ori
REF_Q = [900, 900, 900, 3000, 42000, 50000, 10, 10, 10, 10, 10, 90, 0]
REF_R = 1e-7
REF_HORIZON, REF_DT = 25, 0.02


class SplitWalk(up.UpperSRBController):
    """회전만 상체 SRB (v2) + reference 스윙 OSC.  torque() 를 다시 쓴다 (스윙 가지가 다른 식이라 복사)."""
    I_SRC = "upper"
    W_SRC = "pelvis"

    def __init__(self, m, d, wn3=REF_WN3, kd3=REF_KD3, f_max=None, our_gains=False,
                 ref_q=False, use_jdot=True, foot_pd=None, **kw):
        if ref_q:
            kw = dict(kw)
            kw["q_over"] = {i: w for i, w in enumerate(REF_Q)}
            kw["q_py"] = None
        super().__init__(m, d, **kw)
        if ref_q:
            self.mpc.R = np.diag(np.full(mpc_srb.NU, REF_R))
            self.mpc.R_bar = np.kron(np.eye(self.mpc.N), self.mpc.R)
            self.mpc.r_bar = np.tile(np.diag(self.mpc.R), self.mpc.N)
        self.wn3 = np.asarray(wn3, float)
        self.kd3 = np.asarray(kd3, float)
        self.f_max = f_max
        self.our_gains = our_gains
        self.use_jdot = use_jdot
        self.foot_pd = foot_pd or REF_FOOT_PD
        self.qdd_max = 300.0                            # 피드포워드 특이자세 보호 (09_walk 와 같은 값)
        gains = (f"ω_n={self.wn_swing} ζ={self.zeta_swing} 클램프 400" if our_gains
                 else f"ω_n={self.wn3} kd={self.kd3} 클램프 {f_max}")
        print(f"  [split] 스윙 = Jᵀ[kp e + kd ė] + JᵀΛ_다리(a − J̇q̇) 다리 블록만, {gains}, "
              f"J̇q̇={'켬' if use_jdot else '끔'}, refQ={ref_q}, N={self.mpc.N} dt={self.mpc.dt}")

    # -------------------------------------------------- 500 Hz: 토크 (부모 복사 + 스윙 가지 교체)
    def torque(self, d, t):
        m = self.m
        tau_c = np.zeros(m.nv)
        for i, s_name in enumerate(mpc_srb.FOOT_SITES):
            st = self.gait.in_stance(t, i)
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s_name)
            if (self.liftoff_fix and not st and self._st_prev[i]
                    and not (0.0 <= t - self._sw_t0[i] < 0.02)):
                self.sw[i].start(d.site_xpos[sid].copy())        # 스윙 시작 틱에 시작점 (9/22 Q13)
                self._sw_t0[i] = t
            jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jacp, jacr, sid)
            if st:
                tau_c -= jacp.T @ self.wr[i, :3] + jacr.T @ self.wr[i, 3:]
                if self.yaw_hold and abs(self.wz_cmd) > 1e-12:
                    Rf = d.site_xmat[sid].reshape(3, 3)
                    psi_f = float(np.arctan2(Rf[1, 0], Rf[0, 0]))
                    if not self._st_prev[i]:
                        self.yaw_td[i] = psi_f
                    e = (self.yaw_td[i] - psi_f + np.pi) % (2 * np.pi) - np.pi
                    wz_f = float(jacr[2] @ d.qvel)
                    tau_c += jacr.T @ np.array([0.0, 0.0, walk.KP_YAWHOLD * e - walk.KD_YAWHOLD * wz_f])
            else:
                s = self.gait.swing_phase(t, i)
                dofs = self.leg_dofs[i]
                # --- 다리 블록만 잘라낸 재료 (reference LegSwingDynamicsProvider 와 같은 절차) ---
                Jl = jacp[:, dofs]
                mujoco.mj_fullM(m, d, self._Mfull)
                Ml = self._Mfull[np.ix_(dofs, dofs)]
                lam_inv = Jl @ np.linalg.solve(Ml, Jl.T)
                lam_inv[np.diag_indices(3)] += 1e-9
                Lam = np.linalg.inv(lam_inv)                     # Λ_다리 (골반 고정 가정)
                p_des, v_des = self.sw[i].target(s, self.p_land[i], self.gait.T_swing)
                a_des = self.sw[i].target_acc(s, self.p_land[i], self.gait.T_swing)
                p_foot = d.site_xpos[sid]
                v_foot = jacp @ d.qvel
                if self.our_gains:
                    kp = self.wn_swing ** 2 * np.diag(Lam)
                    kd = 2.0 * self.zeta_swing * self.wn_swing * np.diag(Lam)
                else:
                    kp = self.wn3 ** 2 * np.diag(Lam)            # 식 (3): K_p,i = ω_i² Λ_ii
                    kd = self.kd3                                 # 논문·reference: kd 는 고정 대각
                F = kp * (p_des - p_foot) + kd * (v_des - v_foot)
                fm = 400.0 if self.our_gains else self.f_max
                if fm is not None:
                    nF = np.linalg.norm(F)
                    if nF > fm:
                        F *= fm / nF
                a_res = a_des
                if self.use_jdot:
                    jdp = np.zeros((3, m.nv)); jdr = np.zeros((3, m.nv))
                    mujoco.mj_jacDot(m, d, jdp, jdr, d.site_xpos[sid], int(m.site_bodyid[sid]))
                    a_res = a_des - jdp[:, dofs] @ d.qvel[dofs]  # 식 (2): a − J̇q̇ (다리 열만)
                # 피드포워드 JᵀΛ_다리 a_res.  J 가 정사각(6×6, 위치+자세)이면 = M_다리 J⁻¹ a_res 와 같다
                # (9/23 Q3 검산 1e-14).  Λ 로 그대로 곱하면 무릎이 펴지는 특이자세에서 Λ 가 폭주해
                # 토크 한계 초과가 88 % 였다 (첫 A/B) — 부모(09_walk)처럼 ‖q̈‖ ≤ 300 으로 보호한 뒤 M_다리 로.
                J6 = np.vstack([jacp, jacr])[:, dofs]
                qdd = np.linalg.lstsq(J6, np.concatenate([a_res, np.zeros(3)]), rcond=None)[0]
                nq = np.linalg.norm(qdd)
                if nq > self.qdd_max:
                    qdd *= self.qdd_max / nq
                tau_c[dofs] += Jl.T @ F + Ml @ qdd                # 골반 줄은 만들지 않는다
                # --- 발 자세: 축별 PD (reference G1 config) ---
                R_f = d.site_xmat[sid].reshape(3, 3)
                axis = np.cross(R_f[:, 2], np.array([0.0, 0.0, 1.0]))
                w_foot = jacr @ d.qvel
                pr, pp, py = self.foot_pd["roll"], self.foot_pd["pitch"], self.foot_pd["yaw"]
                mom = np.array([pr[0] * axis[0] - pr[1] * w_foot[0],
                                pp[0] * axis[1] - pp[1] * w_foot[1],
                                -py[1] * w_foot[2]])
                if (not self.gate_sy) or abs(self.wz_cmd) > 1e-12:
                    yd = self.yaw_ref(t + self.gait.time_to_touchdown(t, i))
                    bias = min(np.radians(100.0) * abs(self.wz_cmd), np.radians(20.0))
                    if self.wz_cmd > 0 and i == 0:
                        yd += bias
                    elif self.wz_cmd < 0 and i == 1:
                        yd -= bias
                    yaw_f = np.arctan2(R_f[1, 0], R_f[0, 0])
                    mom[2] += py[0] * ((yd - yaw_f + np.pi) % (2 * np.pi) - np.pi)
                tau_c[dofs] += jacr[:, dofs].T @ mom
            self._st_prev[i] = st
        tau = d.qfrc_bias[self.adof] + tau_c[self.adof]
        for a in self.upper_act:
            jid = self.m.actuator_trnid[a, 0]
            qa = self.m.jnt_qposadr[jid]
            dofa = self.m.jnt_dofadr[jid]
            tau[a] += self.kp_up * (self.q_ref[qa - 7] - d.qpos[qa]) - self.kd_up * d.qvel[dofa]
        return tau


# ===========================================================================
if __name__ == "__main__":
    A = walk_cli.parse()
    argv = sys.argv

    def _v3(key, default):
        if key in argv:
            j = argv.index(key)
            return tuple(float(x) for x in argv[j + 1:j + 4])
        return default

    wn3 = _v3("--wn3", REF_WN3)
    kd3 = _v3("--kd3", REF_KD3)
    fmax = float(argv[argv.index("--fmax") + 1]) if "--fmax" in argv else None
    ours = "--ourgains" in argv
    refq = "--refq" in argv
    nojd = "--nojdot" in argv
    fpd = OUR_FOOT_PD if "--ourfootpd" in argv else None    # 발 자세 PD 를 우리 값(60/5)으로
    if "--refhorizon" in argv:
        walk.HORIZON, walk.DT_MPC = REF_HORIZON, REF_DT
        print(f"  [split] 지평 N={walk.HORIZON} dt={walk.DT_MPC} ({walk.HORIZON * walk.DT_MPC:.2f} s, reference)")
    if A["decim"]:
        walk.DECIM = A["decim"]
        print(f"  [실험] MPC 재풀이 {500 / walk.DECIM:.0f} Hz (DECIM={walk.DECIM})")
    ctor = functools.partial(SplitWalk, wn3=wn3, kd3=kd3, f_max=fmax, our_gains=ours,
                             ref_q=refq, use_jdot=not nojd, foot_pd=fpd)
    if A["view"]:
        walk.view(A["vx"], ctor=ctor, **A["common"], **A["view_kw"])
    else:
        tag = "split" + ("_ours" if ours else "_ref") + ("_refq" if refq else "")
        ok = walk.headless(vx=A["vx"], seconds=A["seconds"], legmass=A["legmass"], ctor=ctor,
                           variant=tag, **A["common"])
        print()
        print("분리 (상체 SRB + reference 스윙) " + ("통과 ✓" if ok else "실패"))
