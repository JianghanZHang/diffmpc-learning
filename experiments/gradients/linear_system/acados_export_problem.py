"""Export the SAME random linear MPC as sweep_admm_tolerance.py so the acados QP-level check
(run in the turbompc-acados Docker) uses identical data. Saves A,B,b,Q,R,x0 + meta.

    python experiments/gradients/linear_system/acados_export_problem.py
"""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DL = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_DL, "external", "diffmpc2", "benchmarking", "linear-system"))
from utils import generate_problem_data, N_STATE, N_CTRL  # noqa: E402

HORIZON, UMAX, N_SAMPLES, SEED = 20, 1.0, 16, 0
Q, R, A, B, b, x0 = generate_problem_data(N_SAMPLES, SEED, n_state=N_STATE, n_ctrl=N_CTRL)
# turbompc discrete dynamics: x_{t+1} = x + dt*((A-I)x + B u + b), dt=1  =>  x_{t+1} = A x + B u + b
out = os.path.join(_HERE, "results", "data", "acados_linear_problem.npz")
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(out,
         A=np.asarray(A, float), B=np.asarray(B, float), b=np.asarray(b, float),
         Q_diag=np.diag(np.asarray(Q, float)), R_diag=np.diag(np.asarray(R, float)),
         x0=np.asarray(x0, float),
         nx=N_STATE, nu=N_CTRL, horizon=HORIZON, umax=UMAX, n_samples=N_SAMPLES, seed=SEED)
print(f"saved {out}: nx={N_STATE} nu={N_CTRL} H={HORIZON} umax={UMAX} n={N_SAMPLES} "
      f"| Q_diag[:3]={np.round(np.diag(Q)[:3],3)} R_diag={np.round(np.diag(R),3)} "
      f"| x0[0,:3]={np.round(x0[0,:3],3)}")
