"""Canonical figure for the quadrotor V1-vs-V3 finding: V3 (truncated BPTT) converges to
V1 (open-loop-plan) as the BPTT window h grows — i.e. the V3 failure at small h is
truncation/myopia (credit-assignment beyond the window), NOT active-set non-smoothness
(the MPC gradient is exact a.e.; FD cos=0.9998). Larger h *helps*, which refutes the
active-set hypothesis (more chained solves would have hurt).

Reads the canonical CSVs in results/ (no re-run needed):
  train_quadrotor_bptt_hard_seed0_h{8,16,24,32}.csv  (V3, lr=3e-3, seed0, varying h)
  train_quadrotor_plan_hard_seed0.csv                (V1 reference, lr=1e-2)

Usage:
  /home/jianghan/Workspace/diffmpc2/.venv-cudss/bin/python \\
    experiments/rl/drone_rl/plot_h_sweep.py
"""
from __future__ import annotations
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_RD = os.path.join(_HERE, "results")
DT = 0.05            # quadrotor_env.DT
TASK_STEPS = 34      # ~closed-loop steps to reach the goal (eval_steps=50 horizon)
HS = [8, 16, 24, 32]
COLORS = {8: "#d62728", 16: "#ff7f0e", 24: "#2ca02c", 32: "#1f77b4"}


def _eval_series(path):
    u, e = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            ec = r.get("eval_cost", "")
            if ec not in ("", "None"):
                u.append(int(r["update"])); e.append(float(ec))
    return np.array(u), np.array(e)


def main():
    # V1 reference (its converged eval is robust to lr)
    _, v1 = _eval_series(os.path.join(_RD, "train_quadrotor_plan_hard_seed0.csv"))
    v1_final = float(v1[-1])

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    finals = {}
    for h in HS:
        u, e = _eval_series(os.path.join(_RD, f"train_quadrotor_bptt_hard_seed0_h{h}.csv"))
        finals[h] = float(e[-1])
        ax[0].plot(u, e, "o-", color=COLORS[h], label=f"V3 h={h}  (h·dt={h*DT:.2f}s)")
    ax[0].axhline(v1_final, ls="--", color="gray", lw=1)
    ax[0].text(2, v1_final + 1.0, f"V1 plan_hard ≈ {v1_final:.1f}", color="gray", fontsize=8)
    ax[0].set_xlabel("update"); ax[0].set_ylabel("closed-loop eval cost (from START)")
    ax[0].set_title("V3 (truncated BPTT) eval vs window h   (lr=3e-3, seed 0)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    hd = np.array(HS) * DT
    ff = np.array([finals[h] for h in HS])
    ax[1].plot(hd, ff, "o-", color="black")
    for h, f in zip(HS, ff):
        ax[1].annotate(f"h={h}\n{f:.0f}", (h * DT, f), fontsize=8, ha="center", va="bottom")
    ax[1].axhline(v1_final, ls="--", color="gray")
    ax[1].text(0.45, v1_final + 1.0, f"V1 ≈ {v1_final:.1f}", color="gray", fontsize=8)
    ax[1].axvline(TASK_STEPS * DT, ls=":", color="purple", lw=1)
    ax[1].text(TASK_STEPS * DT - 0.03, ff.max() * 0.8, f"task ~{TASK_STEPS*DT:.1f}s",
               color="purple", fontsize=8, rotation=90, va="top")
    ax[1].set_xlabel("BPTT window time  h·dt  [s]"); ax[1].set_ylabel("final eval cost")
    ax[1].set_title("Final eval vs window length\nV3 → V1 as h grows (truncation, not active-set)")
    ax[1].grid(alpha=0.3)

    out = os.path.join(_RD, "quadrotor_v3_h_sweep.png")
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print("V1 final eval:", round(v1_final, 2))
    print("V3 final eval by h:", {h: round(finals[h], 2) for h in HS})
    print("saved", out)


if __name__ == "__main__":
    main()
