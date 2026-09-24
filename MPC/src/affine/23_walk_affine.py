"""아핀항 MPC — 다리 각운동량을 '알려진 시변 항' 으로 SRB 동역학에 넣는다 (Q&A 9/22 Q9, 9/23 Q8).

09_walk 대비 무엇이 바뀌나 (새 파일 — 기존 코드는 손대지 않고 WalkController / WrenchMPC 를 상속):
  1) Θ̇ 행 :  Θ̇ = Rᵀ[ (I + (1−α)⊙(I_ub⁻¹ I_tot − I)) ω  −  (1−α)⊙ I_ub⁻¹ L_다리(t) ]
              α = 축별 골반 비율 (--wxpel/--wypel/--wzpel).  α=0 (전신 ω) 이면 9/22 Q9 식 그대로,
              α=1 (골반 ω) 이면 Θ̇ = Rᵀω 가 이미 맞으므로 보정 0.
  2) ω̇ 행 :  α>0 인 축은 I_w ω̇_골반 = τ − L̇_다리 → c 의 6:9 줄에 −I_w⁻¹(α⊙L̇_다리)   (--noaffw 로 끔)
  3) L_다리,k (k=0..N−1) 는 지평 각 스텝의 **계획된** 발 위치·속도로 질점 근사 (leg_momentum.LegPointModel).
     k=0 을 실측값에 앵커해 예측의 상수 편차를 지운다 (--noanchor 로 끔).
  4) condensed QP :  X = A_qp x0 + d + B_qp U  의 d 만 추가 → g 한 줄 변경.  H·제약·변수 수·풀이 시간 불변.

먼저 할 것 — 예측 정확도 (--check, 9/22 Q9 §6 의 1·2 단계):
  기본 컨트롤러(--affmode off)로 걷게 하면서 매 MPC 틱에
    (a) 질점 모델 vs 실측 L_다리            (모델 근사가 맞나)
    (b) I_ub⁻¹(L_전신 − L_다리) vs 실측 골반 ω   (9/16 의 0.95 재현 + 어느 L 정의가 맞나)
    (c) 지평 k 스텝 앞 예측 vs 그 시각 실측     (preview 로 쓸 만한가)
  예측이 틀리면 아핀항은 외란을 **더할** 뿐이다 — 이 표를 보고 켠다.

사용 (09_walk 와 같은 플래그 + 아래):
  .venv/Scripts/python.exe MPC/src/affine/23_walk_affine.py --check --vx 0.5 --seconds 20 [권장 플래그]
  .venv/Scripts/python.exe MPC/src/affine/23_walk_affine.py --vx 0.5 --seconds 40 [권장 플래그]
  .venv/Scripts/python.exe MPC/src/affine/23_walk_affine.py --view --vx 0.5 [권장 플래그]
  --affmode full|c|a|off   full(기본): A 블록 + c,  c: 오프셋만,  a: A 블록만,  off: 09_walk 와 동일 (대조군)
  --noanchor  --anchortau 0.1 (앵커 감쇠 시정수 s)  --afffade 0.4 (이 시각부터 아핀항 페이드, 기본 없음)  --noaffw
  --affscale 0.65 (예측 L 배율)  --nodemean (지평 평균 제거 끄기 — 기본은 켬)
  --orbit              상체 궤도항 m_상체(c_상체−C)×(v_상체−v_C) 까지 넣기 (Q&A 9/24 Q20·Q21). 기본 꺼짐.
                       정확한 식: ω_골반 = I_상체⁻¹ (L − L_다리 − 궤도항). 없으면 roll 을 1.8 배 과대 예측한다
                       120 s 결과 (Q21): 섞기 유지 (wxp 0.5) 에선 pitch σ −13~19 % 더 줄지만 약간 뒤로 기움 (−1~−3°).
                       roll 섞기를 빼면 (wxp 0) roll σ 가 2~2.4 배 — 계획 예측의 roll 이 부정확해서 (상관 0.06~0.22)
  --refshape           ω 참조를 "골반이 멈춰 있는 값" ω_ref,k = I_전신⁻¹ L_다리,k 로 (Q&A 9/24 Q30). 기본 꺼짐.
                       전신 ω 축 (α < 1) 에만 적용. 참조만 바꾸므로 볼록성 유지.
                       이유: 모델만 고치면 목표가 여전히 '전신 평균 ω = 0' 이라 MPC 가 골반을 다리 반대로 돌린다
                       (세 축 모두 전신 ω 로 하면 골반 yaw 16 ~ 20°, affine 을 켜도 그대로)

1 차 결과 (9/23 Q11, 40 s): 예측 pitch 상관 지평 전체 0.87~0.98.  c 항이 몸을 +7~13° 앞으로 기울였는데
(예측 L 의 DC = 모델 오차), 지평 평균 제거로 사라짐 — pitch σ 3.35→2.63 (0.5), 4.04→3.10 (0.7),
발가락 하중 0 구간 24→19 / 43→27 ms, 골반 yaw 4.6→3.3 / 6.1→4.5°, 추종은 그대로 (103 / 88 %).
"""
from __future__ import annotations

import functools
import importlib
import sys
import time

import numpy as np
import mujoco

import g1_model
import mpc_srb
from mpc_srb import NX, NU, NU_PER_FOOT, N_FEET, rz, continuous_AB, discretize
from mpc_qp import WrenchMPC, foot_constraints
from gait import SwingController, raibert_target
import leg_momentum as lm
import walk_cli

walk = importlib.import_module("09_walk")


# ===========================================================================
class AffineMPC(WrenchMPC):
    """WrenchMPC + 아핀항.  solve_gait 만 다시 쓴다 (부모 것을 복사하고 Ac 블록·c_k·d 를 넣었다)."""

    def __init__(self, params, I_ub, alpha, mode="full", aff_w=True, ref_shape=False, **kw):
        super().__init__(params, **kw)
        self.I_ub = np.asarray(I_ub, float)             # 상체 관성 (body/yaw frame, 자기 CoM 기준)
        self.alpha = np.asarray(alpha, float)           # 축별 골반 비율 (yaw frame)
        self.mode = mode
        self.aff_w = aff_w
        one_m_a = 1.0 - self.alpha
        # Θ̇ = M_th (Rᵀω) + …,   M_th = I + diag(1−α)(I_ub⁻¹ I_tot − I)
        self.M_th = np.eye(3) + np.diag(one_m_a) @ (np.linalg.solve(self.I_ub, params.I_body) - np.eye(3))
        self.I_ub_inv = np.linalg.inv(self.I_ub)
        self.L_provider = None       # (x0, X_ref, foot_traj, contact) -> (N,3) L_다리 (world, 전신 CoM 기준)
        self.last_L = None
        self.last_d = None
        self.ref_shape = bool(ref_shape)                # ω 참조 = 골반 정지 값 (Q30)

    def affine_c(self, psi, L_traj):
        """이산 아핀항 c_d (N,13).  c_d,k = c dt + ½ A c dt²  (A²c = 0 이라 정확한 ZOH)."""
        N = self.N
        Rz_ = rz(psi)
        Ac = np.zeros((NX, NX))
        Ac[0:3, 6:9] = (self.M_th if self.mode in ("full", "a") else np.eye(3)) @ Rz_.T
        Ac[3:6, 9:12] = np.eye(3)
        Ac[11, 12] = -1.0
        c_d = np.zeros((N, NX))
        if self.mode not in ("full", "c"):
            return c_d
        I_w_inv = np.linalg.inv(Rz_ @ self.p.I_body @ Rz_.T)
        Ldot = np.gradient(L_traj, self.dt, axis=0) if N > 1 else np.zeros_like(L_traj)
        for k in range(N):
            cc = np.zeros(NX)
            cc[0:3] = -(1.0 - self.alpha) * (self.I_ub_inv @ (Rz_.T @ L_traj[k]))
            if self.aff_w and self.alpha.any():
                cc[6:9] = -I_w_inv @ (Rz_ @ (self.alpha * (Rz_.T @ Ldot[k])))
            c_d[k] = cc * self.dt + 0.5 * (Ac @ cc) * self.dt * self.dt
        return c_d

    def solve_gait(self, x0, X_ref, psi, foot_pos_traj, contact, u_ref_traj,
                   psi_feet=None, fz_scale=None):
        L_traj = None
        if self.L_provider is not None:
            L_traj = np.asarray(self.L_provider(x0, X_ref, foot_pos_traj, contact), float)
            self.last_L = L_traj
        if self.mode == "off" or L_traj is None:
            return super().solve_gait(x0, X_ref, psi, foot_pos_traj, contact, u_ref_traj,
                                      psi_feet=psi_feet, fz_scale=fz_scale)
        t0 = time.perf_counter()
        N = self.N
        Rz_ = rz(psi)
        Ac, _ = continuous_AB(self.p, psi, foot_pos_traj[0] - x0[3:6])
        if self.mode in ("full", "a"):
            Ac[0:3, 6:9] = self.M_th @ Rz_.T          # ★ Θ–ω 블록 교체
        Ad, _ = discretize(Ac, np.zeros((NX, NU)), self.dt)
        c_d = self.affine_c(psi, L_traj)              # ★ 스텝별 아핀항
        if self.ref_shape:
            # ★ ω 참조를 "골반이 멈춰 있는 값" 으로: ω_골반 = I_상체⁻¹(I_전신 ω − L_다리) = 0 ⇒ ω = I_전신⁻¹ L_다리.
            #   전신 ω 축 (1−α) 만큼만 — 골반 ω 축은 이미 골반 정지가 목표다. 몸 yaw 좌표에서 축별로.
            X_ref = np.array(X_ref, float, copy=True)
            one_m_a = 1.0 - self.alpha
            for k in range(N):
                w_add_b = one_m_a * np.linalg.solve(self.p.I_body, Rz_.T @ L_traj[k])
                X_ref[k, 6:9] = X_ref[k, 6:9] + Rz_ @ w_add_b

        Bd = []
        for k in range(N):
            com_k = x0[3:6] if k == 0 else X_ref[k, 3:6]
            _, Bc = continuous_AB(self.p, psi, foot_pos_traj[k] - com_k)
            for i in range(N_FEET):
                if not contact[k, i]:
                    Bc[:, NU_PER_FOOT * i:NU_PER_FOOT * (i + 1)] = 0.0
            Bd.append(discretize(Ac, Bc, self.dt)[1])

        A_qp = np.zeros((NX * N, NX))
        B_qp = np.zeros((NX * N, NU * N))
        d = np.zeros(NX * N)
        A_pow = np.eye(NX)
        acc = np.zeros(NX)
        for k in range(N):
            A_pow = Ad @ A_pow
            A_qp[NX * k:NX * (k + 1)] = A_pow
            acc = Ad @ acc + c_d[k]                   # ★ d_k = Σ_{j≤k} Ad^{k−j} c_d,j
            d[NX * k:NX * (k + 1)] = acc
        for j in range(N):
            blk = Bd[j]
            for k in range(j, N):
                B_qp[NX * k:NX * (k + 1), NU * j:NU * (j + 1)] = blk
                blk = Ad @ blk
        self.last_d = d

        nU = NU * N
        free_mask = np.repeat(np.asarray(contact, bool).reshape(-1), NU_PER_FOOT)
        free = np.flatnonzero(free_mask)
        eq_idx = np.flatnonzero(~free_mask)
        B_r = B_qp[:, free]
        q, r = self.q_bar, self.r_bar[free]
        H = 2.0 * ((B_r.T * q) @ B_r)
        H[np.diag_indices_from(H)] += 2.0 * r
        g = 2.0 * B_r.T @ (q * (A_qp @ x0 + d - X_ref.reshape(-1))) \
            - 2.0 * r * u_ref_traj.reshape(-1)[free]          # ★ + d 가 유일한 비용 변화
        if self.du_w is not None and np.any(self.du_w > 0):
            D = np.eye(nU) - np.eye(nU, k=-NU)
            e = np.zeros(nU)
            if self.u_prev is not None:
                e[:NU] = self.u_prev
            else:
                D[:NU, :] = 0.0
            D_r = D[:, free]
            DW = D_r.T * np.tile(self.du_w, N)
            H = H + 2.0 * DW @ D_r
            g = g - 2.0 * DW @ e
        H = 0.5 * (H + H.T)

        if psi_feet is None:
            psi_feet = (psi, psi)
        Cf_i, df_i = zip(*(foot_constraints(self.p, psi=pf) for pf in psi_feet))
        blocks = [(k, i) for k in range(N) for i in range(N_FEET) if contact[k, i]]
        nr = Cf_i[0].shape[0]
        C_in = np.zeros((nr * len(blocks), nU))
        for b_, (k, i) in enumerate(blocks):
            base = NU * k + NU_PER_FOOT * i
            C_in[nr * b_:nr * (b_ + 1), base:base + NU_PER_FOOT] = Cf_i[i]
        d_list = []
        for (k, i) in blocks:
            dd = df_i[i]
            if fz_scale is not None and fz_scale[k, i] < 1.0:
                dd = dd.copy()
                dd[4] = self.p.fz_max * max(float(fz_scale[k, i]), 0.0)
                dd[5] = 0.0
            d_list.append(dd)
        d_in = np.concatenate(d_list)

        u_r = self._solve_qp_ineq(H, g, C_in[:, free], d_in)
        U = np.zeros(nU)
        U[free] = u_r
        u0 = U[:NU]
        self.u_prev = u0.copy()
        info = {
            "solve_ms": (time.perf_counter() - t0) * 1e3,
            "violation": float((C_in @ U - d_in).max()),
            "eq_violation": float(np.abs(U[eq_idx]).max()) if len(eq_idx) else 0.0,
            "solver": self.solver_name,
        }
        self.last = {"U": U, "Bd": Bd, "contact": contact}
        return u0, U, info


# ===========================================================================
class AffineWalk(walk.WalkController):
    """WalkController + AffineMPC + 지평 L_다리 예측.  update_mpc 는 부모 것 그대로 — 예측기는 MPC 가 부른다."""

    def __init__(self, m, d, aff_mode="full", aff_anchor=True, aff_w=True,
                 aff_anchor_tau=0.1, aff_fade=None, aff_scale=1.0, aff_demean=True, aff_orbit=False,
                 aff_refshape=False, **kw):
        super().__init__(m, d, **kw)
        self.aff_scale = float(aff_scale)               # 예측 L 배율 (--check: 계획 예측 RMS 가 실측의 1.4~1.6 배)
        self.aff_demean = bool(aff_demean)              # 지평 평균 제거
        self.aff_anchor_tau = float(aff_anchor_tau)     # 앵커 감쇠 시정수 [s]
        self.aff_fade = None if aff_fade is None else float(aff_fade)   # 이 시각부터 0.2 s 에 걸쳐 아핀항 페이드 [s]
        I_ub, m_ub, _ = lm.upper_body_inertia(m, d)
        self.m_ub = float(m_ub)
        self.aff_orbit = bool(aff_orbit)                # 상체 궤도항 (Q20)
        self.I_ub = I_ub
        yaw0 = mpc_srb.quat_to_euler_zyx(d.qpos[3:7])[2]
        self.mpc = AffineMPC(self.params, I_ub, self.w_pel, mode=aff_mode, aff_w=aff_w,
                             horizon=walk.HORIZON, dt=walk.DT_MPC,
                             q_diag=np.diag(self.mpc.Q), psi0=yaw0, du_w=self.mpc.du_w)
        self.mpc.L_provider = self.predict_leg_L
        self.mpc.ref_shape = bool(aff_refshape)
        self.legpm = [lm.LegPointModel(m, d, i) for i in range(2)]
        self.aff_anchor = aff_anchor
        self._t_now, self._d_now = 0.0, d
        self._sc = SwingController()                 # 미래 스윙 궤적 평가용 스크래치
        self.last_anchor = np.zeros(3)
        It, Iu = self.params.I_body, I_ub
        print(f"  [affine] 모드={aff_mode} 앵커={aff_anchor}(τ {self.aff_anchor_tau}) 페이드={self.aff_fade}"
              f" 배율={self.aff_scale} 평균제거={self.aff_demean} 궤도항={self.aff_orbit} ω̇항={aff_w} α={self.w_pel}"
              f"  I_ub/I_tot 대각 = {np.diag(Iu) / np.diag(It)}  (상체 {m_ub:.1f} kg)")

    def update_mpc(self, d, t):
        self._t_now, self._d_now = t, d
        return super().update_mpc(d, t)

    # ---- 지평 L_다리 예측 ---------------------------------------------------
    def _future_land(self, i, x0, t):
        """발 i 의 다음 스윙 착지 목표 (Raibert).  그 스윙은 t_lo 뒤에 시작하므로 CoM 을 명령 속도로
        t_lo 만큼 먼저 보내 놓고 잰다 — 지금 CoM 기준으로 재면 v·t_lo 만큼 뒤에 놓여 스윙 이동량이
        절반으로 줄어든다 (--check 에서 k ≥ 6 의 pitch 상관이 무너지던 원인)."""
        gait = self.gait
        vc = self.v_cmd(t)
        if gait.in_stance(t, i):
            t_lo = (gait.stance_frac - gait.phase(t, i)) * gait.T
        else:                                   # 지금 스윙 중이면 '다음' 스윙 = 착지 + 스탠스 뒤
            t_lo = gait.time_to_touchdown(t, i) + gait.T_stance
        p_com = x0[3:6] + vc * t_lo
        yaw_pl = self.yaw_ref(t + t_lo + 0.5 * gait.T_swing) if abs(self.wz_cmd) > 1e-12 else x0[2]
        combo = abs(self.wz_cmd) > 1e-12 and abs(self.vx_cmd) > 1e-9
        off = np.array(self.side_offset_at(t)[i], float)
        off[0] += self.td_dx
        return raibert_target(p_com, x0[9:12], yaw_pl, off, vc, gait.T_stance,
                              z_com=x0[5], z_ground=self.z_ground,
                              anchor_xy=None if combo else [p_com[0], self.com0[1]],
                              cap_y_max=self.cap_y, lip_exact=self.lip_exact,
                              ff_scale=self.td_scale)

    def _plan_L_at(self, tk, Ck, vCk, psi_k, x0, t, feet, foot_k, o_hip):
        """시각 tk 의 계획된 다리 각운동량 (두 다리 합, world, 전신 CoM 기준)."""
        gait, sc = self.gait, self._sc
        Rk = rz(psi_k)
        L, Sr, Sv = np.zeros(3), np.zeros(3), np.zeros(3)
        for i in range(2):
            p_hip = Ck + Rk @ o_hip[i]
            v_hip = vCk
            if gait.in_stance(tk, i):
                p_foot, v_foot = foot_k[i], np.zeros(3)
            else:
                s = gait.swing_phase(tk, i)
                same_swing = (not gait.in_stance(t, i)) and (tk - t) < gait.time_to_touchdown(t, i)
                if same_swing:
                    p0, p1 = self.sw[i].p_liftoff, self.p_land[i]
                else:                                   # 미래 스윙: 지금 자리(또는 다음 착지점)에서 출발
                    p0 = feet[i] if gait.in_stance(t, i) else self.p_land[i]
                    p1 = self._future_land(i, x0, t)
                sc.p_liftoff = p0
                p_foot, v_foot = sc.target(s, p1, gait.T_swing)
            Li, Sri, Svi = self.legpm[i].terms(Ck, vCk, p_hip, v_hip, p_foot, v_foot)
            L += Li; Sr += Sri; Sv += Svi
        if self.aff_orbit:
            # 상체 궤도항 (S_r × S_v)/m_상체 를 다리 몫에 합쳐 넣는다 → c 가 −I_상체⁻¹(L_다리 + 궤도항) 이 된다 (Q20)
            L = L + np.cross(Sr, Sv) / self.m_ub
        return L

    def predict_leg_L(self, x0, X_ref, foot_traj, contact):
        m, d, t = self.m, self._d_now, self._t_now
        N, DT = self.mpc.N, self.mpc.dt
        sc = self._sc
        sc.soft_land, sc.h_swing = self.sw[0].soft_land, self.sw[0].h_swing
        sc.quintic_xy, sc.quintic_z, sc.v_td = self.sw[0].quintic_xy, self.sw[0].quintic_z, self.sw[0].v_td
        Rz0 = rz(x0[2])
        C0, vC0 = x0[3:6], x0[9:12]
        o_hip = [Rz0.T @ (pm.hip_now(d) - C0) for pm in self.legpm]
        feet = mpc_srb.get_foot_positions(m, d)
        L = np.zeros((N, 3))
        tks = t + (np.arange(N) + 0.5) * DT
        for k in range(N):
            ref = X_ref[k]
            L[k] = self._plan_L_at(tks[k], ref[3:6], ref[9:12], ref[2], x0, t, feet, foot_traj[k], o_hip)
        if self.aff_anchor:
            # k=0 편차를 지우되 지평을 따라 τ 로 감쇠 — 상수로 더하면 먼 스텝을 오염시킨다 (--check (c))
            L_now_plan = self._plan_L_at(t, C0, vC0, x0[2], x0, t, feet, feet, o_hip)
            if self.aff_orbit:
                Lm, Srm, Svm = lm.measured_leg_terms(m, d)
                L_meas = Lm + np.cross(Srm, Svm) / self.m_ub
            else:
                L_meas = lm.measured_leg_L(m, d)
            self.last_anchor = L_meas - L_now_plan
            L += self.last_anchor[None, :] * np.exp(-(tks - t) / self.aff_anchor_tau)[:, None]
        if self.aff_fade is not None:
            # 먼 지평은 예측이 무너지므로 (--check (c): 0.3 s 너머) 아핀항을 서서히 0 으로
            L *= np.clip(1.0 - (tks - t - self.aff_fade) / 0.2, 0.0, 1.0)[:, None]
        if self.aff_demean:
            # 지평(0.8 s = 한 걸음 주기) 평균을 뺀다 — 정상 보행에서 골반 평균 pitch 율은 0 이므로
            # 예측 L 의 DC 는 모델 오차다. 첫 A/B 에서 c 항이 몸을 앞으로 7~21° 기울인 원인 후보.
            L = L - L.mean(0, keepdims=True)
        return L * self.aff_scale


# ===========================================================================
def check(vx, seconds, common, settle=8.0, orbit=False):
    """예측 정확도 — 기본 MPC(--affmode off)로 걷게 하면서 매 틱 예측·실측을 모아 상관을 찍는다."""
    m, d = g1_model.load_torque()
    g1_model.set_crouch(m, d)
    kw = dict(common)
    kw["wz_cmd"] = kw.pop("wz", 0.0)             # headless() 인자명 → 생성자 인자명
    ctl = AffineWalk(m, d, vx_cmd=vx, aff_mode="off", aff_anchor=False, aff_demean=False, aff_orbit=orbit, **kw)
    N, DT = ctl.mpc.N, ctl.mpc.dt
    dt = m.opt.timestep
    rows = dict(t=[], L_rel=[], L_abs=[], L_pm=[], L_pred=[], w_pel=[], w_L=[], yaw=[], L_orb=[])
    z0 = d.qpos[2]
    for k in range(int(seconds / dt)):
        t = k * dt
        mujoco.mj_forward(m, d)
        if k % walk.DECIM == 0:
            x0 = ctl.update_mpc(d, t)[0]
            if t > settle:
                xw = mpc_srb.get_state(m, d, I_body=ctl.params.I_body)    # 순수 전신 ω
                rows["t"].append(t)
                rows["L_rel"].append(lm.measured_leg_L(m, d, relative=True))
                _Lm, _Sr, _Sv = lm.measured_leg_terms(m, d)
                rows["L_orb"].append(_Lm + np.cross(_Sr, _Sv) / ctl.m_ub)
                rows["L_abs"].append(lm.measured_leg_L(m, d, relative=False))
                rows["L_pm"].append(lm.predict_leg_L_now(ctl.legpm, m, d))
                rows["L_pred"].append(ctl.mpc.last_L.copy())
                rows["w_pel"].append(lm.pelvis_omega_world(m, d))
                rows["w_L"].append(xw[6:9].copy())
                rows["yaw"].append(x0[2])
        d.ctrl[:] = ctl.torque(d, t)
        mujoco.mj_step(m, d)
        if d.qpos[2] < z0 - 0.25:
            print(f"  쓰러짐 @ {t:.2f}s — 그때까지의 표본으로 계산")
            break
    R = {k_: np.array(v) for k_, v in rows.items()}
    n = len(R["t"])
    if n < 50:
        print("표본 부족"); return
    ax = "roll pitch yaw".split()

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / den) if den > 0 else float("nan")

    def rms(a):
        return float(np.sqrt((a * a).mean()))

    print(f"\n예측 정확도 — vx {vx}, {settle:.0f}~{R['t'][-1]:.1f} s, MPC 틱 {n} 개\n")
    print("(a) 질점 모델 (지금 실제 엉덩이·발 속도로 평가) vs 실측 L_다리 (상대속도 정의)")
    print(f"    {'축':6s} {'상관':>6s} {'RMS 예측/실측':>14s}   실측 RMS [kg·m²/s]")
    for a in range(3):
        print(f"    {ax[a]:6s} {corr(R['L_pm'][:, a], R['L_rel'][:, a]):6.2f}"
              f" {rms(R['L_pm'][:, a]) / max(rms(R['L_rel'][:, a]), 1e-9):14.2f}   {rms(R['L_rel'][:, a]):.3f}")

    print("\n(b) 골반 ω 예측:  ω̂_pel = I_ub⁻¹ (I_tot ω_L − L)  vs 실측 골반 ω  — L 정의별 상관")
    I_ub, I_t = ctl.I_ub, ctl.params.I_body
    w_hat = {}
    for name, key in (("실측 L(상대속도)", "L_rel"), ("실측 L(절대속도)", "L_abs"),
                      ("실측 L + 상체 궤도항", "L_orb"),
                      ("질점 모델 (지금)", "L_pm"), ("계획 예측 k=0" + (" (+궤도항)" if orbit else ""), None)):
        out = np.zeros((n, 3))
        for j in range(n):
            Rz_ = rz(R["yaw"][j])
            Lj = R["L_pred"][j][0] if key is None else R[key][j]
            wb = Rz_.T @ R["w_L"][j]
            out[j] = Rz_ @ np.linalg.solve(I_ub, I_t @ wb - Rz_.T @ Lj)
        w_hat[name] = out
    print(f"    {'L 정의':18s} {'roll':>6s} {'pitch':>6s} {'yaw':>6s}     (지금 식 Θ̇=Rᵀω 즉 ω_L 자체: "
          + " ".join(f"{corr(R['w_L'][:, a], R['w_pel'][:, a]):5.2f}" for a in range(3)) + ")")
    for name, out in w_hat.items():
        print(f"    {name:22s} " + " ".join(f"{corr(out[:, a], R['w_pel'][:, a]):6.2f}" for a in range(3))
              + "    RMS 예측/실측 " + " ".join(f"{rms(out[:, a]) / max(rms(R['w_pel'][:, a]), 1e-9):.2f}" for a in range(3)))

    print("\n(c) 지평 k 스텝 앞 예측 vs 그 시각 실측 L_다리 (상대속도) — 상관 (원시 / 앵커 후)")
    tt = R["t"]
    anchor = R["L_rel"] - R["L_pred"][:, 0, :]          # k=0 편차 (앵커가 지우는 몫)
    print(f"    {'k':>3s} {'앞선 시간':>8s} | " + " ".join(f"{a:>12s}" for a in ax))
    for k in (0, 1, 3, 6, 9, 12, 15):
        if k >= N:
            continue
        ta = tt + (k + 0.5) * DT
        ok = ta <= tt[-1]
        meas = np.stack([np.interp(ta[ok], tt, R["L_rel"][:, a]) for a in range(3)], 1)
        raw = R["L_pred"][ok, k, :]
        anc = raw + anchor[ok]
        print(f"    {k:3d} {(k + 0.5) * DT:7.3f}s | "
              + " ".join(f"{corr(raw[:, a], meas[:, a]):5.2f}/{corr(anc[:, a], meas[:, a]):5.2f}" for a in range(3)))
    print("\n    읽는 법: (a)(b) 가 낮으면 질점 근사가 문제, (c) 가 k 에 따라 급락하면 계획 궤적이 실제와 다른 것.")


# ===========================================================================
if __name__ == "__main__":
    A = walk_cli.parse()
    mode = sys.argv[sys.argv.index("--affmode") + 1] if "--affmode" in sys.argv else "full"
    anchor = "--noanchor" not in sys.argv
    affw = "--noaffw" not in sys.argv
    atau = float(sys.argv[sys.argv.index("--anchortau") + 1]) if "--anchortau" in sys.argv else 0.1
    fade = float(sys.argv[sys.argv.index("--afffade") + 1]) if "--afffade" in sys.argv else None
    ascl = float(sys.argv[sys.argv.index("--affscale") + 1]) if "--affscale" in sys.argv else 1.0
    admn = "--nodemean" not in sys.argv          # 지평 평균 제거 (기본 켬 — 3 차 A/B 에서 앞기울기 제거)
    orb = "--orbit" in sys.argv                   # 상체 궤도항 (Q20)
    rsh = "--refshape" in sys.argv                # ω 참조 = 골반 정지 값 (Q30)
    if A["decim"]:
        walk.DECIM = A["decim"]
        print(f"  [실험] MPC 재풀이 {500 / walk.DECIM:.0f} Hz (DECIM={walk.DECIM})")
    if "--check" in sys.argv:
        check(A["vx"], A["seconds"], A["common"], orbit=orb)
        sys.exit(0)
    ctor = functools.partial(AffineWalk, aff_mode=mode, aff_anchor=anchor, aff_w=affw,
                             aff_anchor_tau=atau, aff_fade=fade, aff_scale=ascl, aff_demean=admn, aff_orbit=orb,
                             aff_refshape=rsh)
    if A["view"]:
        walk.view(A["vx"], ctor=ctor, **A["common"], **A["view_kw"])
    else:
        ok = walk.headless(vx=A["vx"], seconds=A["seconds"], legmass=A["legmass"], ctor=ctor,
                           variant=f"affine_{mode}", **A["common"])
        print()
        print(f"아핀항 MPC ({mode}) " + ("통과 ✓" if ok else "실패"))
