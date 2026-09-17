"""MPC 디버깅용 로거 — x, u, 참조, 토크, 솔버 통계를 기록해 npz + csv 로 저장.

사용:
    log = MPCLog("logs/standing_mpc")
    log.add(t=t, x=x, x_ref=x_ref, u=u, tau=tau, solve_ms=1.2, violation=0.0,
            fz_contact=327.1, ncon=8)
    ...
    log.save()      # -> logs/standing_mpc_YYYYmmdd_HHMMSS.npz / .csv

npz 키: t, x, x_ref, u, tau, solve_ms, violation, fz_contact, ncon,
        x_labels, u_labels  (플롯은 src/plot_log.py)
csv: t + x(13) + x_ref(13) + u(12) + solve_ms + violation  — 엑셀에서 바로 열림
"""
from __future__ import annotations

import csv
import datetime
from pathlib import Path

import numpy as np

from mpc_srb import X_LABELS, U_LABELS


class MPCLog:
    def __init__(self, prefix: str, note: str = ""):
        # prefix 는 프로젝트 루트 기준 상대경로 가능 ("logs/standing_qp" 등)
        root = Path(__file__).resolve().parent.parent
        self.path = (root / prefix) if not Path(prefix).is_absolute() else Path(prefix)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.note = note
        self.rows: dict[str, list] = {}

    def add(self, **kw):
        """키워드로 스칼라/벡터 아무거나 기록. 매 호출 같은 키 집합이어야 함."""
        for k, v in kw.items():
            self.rows.setdefault(k, []).append(np.asarray(v, dtype=float))

    def save(self) -> Path:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.path.with_name(f"{self.path.name}_{stamp}")

        arrays = {k: np.stack(v) for k, v in self.rows.items()}
        arrays["x_labels"] = np.array(X_LABELS)
        arrays["u_labels"] = np.array(U_LABELS)
        arrays["note"] = np.array(self.note)
        np.savez_compressed(base.with_suffix(".npz"), **arrays)

        # csv (주요 열만)
        with open(base.with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as fp:
            wcsv = csv.writer(fp)
            hdr = ["t"]
            hdr += [f"x_{s}" for s in X_LABELS]
            hdr += [f"ref_{s}" for s in X_LABELS]
            hdr += [f"u_{s}" for s in U_LABELS]
            hdr += ["solve_ms", "violation"]
            wcsv.writerow(hdr)
            n = len(self.rows["t"])
            for i in range(n):
                row = [float(self.rows["t"][i])]
                row += list(self.rows["x"][i])
                row += list(self.rows["x_ref"][i])
                row += list(self.rows["u"][i])
                row += [float(self.rows["solve_ms"][i]),
                        float(self.rows["violation"][i])]
                wcsv.writerow([f"{v:.6g}" for v in row])

        print(f"  로그 저장: {base.with_suffix('.npz').relative_to(self.path.parent.parent)}"
              f"  (+ .csv, {n} rows)")
        return base.with_suffix(".npz")
