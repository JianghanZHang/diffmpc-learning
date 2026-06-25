"""Loose-bound discriminator: does the slack-vs-hard gradient error vanish when the box is loose enough
that NO control is active? If yes -> the error is the Moreau active-constraint relaxation (real). If it
persists with no active constraints -> implementation bug. Compares the two ANALYTIC gradients directly
(no FD needed): cos(slack AD, hard AD) at increasing umax, sim_steps=50, gamma=1e4.

    python research/gradient-quality-diffnmpc/experiments/linear_system/loose_bound_check.py
"""
import os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from closed_loop_gradient_accuracy import (  # noqa: E402
    _cos, ad, cost_fn_A, WK, QK, RK, NX, NU, build_turbompc_linear_problem, generate_problem_data)

UMAXES = [1.0, 2.0, 5.0, 20.0]


def problem_umax(seed, batch, horizon, umax):
    dyn, pp_t = build_turbompc_linear_problem(horizon=horizon, umax=umax, n_state=NX, n_ctrl=NU)
    Q, R, A, B, b, x0 = generate_problem_data(batch, seed, n_state=NX, n_ctrl=NU)
    A_sd = jnp.asarray(A - np.eye(NX)); B_m = jnp.asarray(B); b_v = jnp.asarray(b)
    pp = dict(pp_t); pp["dynamics_state_dot_params"] = {"A": A_sd, "B": B_m, "b": b_v}
    pp[QK] = jnp.asarray(np.diag(Q)); pp[RK] = jnp.asarray(np.diag(R))
    return dyn, pp, {k: pp[k] for k in WK}, jnp.asarray(x0)


def main():
    n = 64
    print("loose-bound test: cos(slack AD, hard AD) as the box loosens (sim_steps=50, gamma=1e4):")
    print("  -> 1.0 as umax grows (fewer active controls) => the error is the active-constraint relaxation, NOT a bug")
    for umax in UMAXES:
        t = time.time()
        dyn, pp, w, x0b = problem_umax(0, n, 160, umax)
        g_hard = ad(cost_fn_A(dyn, pp, w, x0b, 1e-9, False, 50), w, x0b)
        g_slk = ad(cost_fn_A(dyn, pp, w, x0b, 1e-9, True, 50), w, x0b)
        cos = np.array([_cos(g_slk[i], g_hard[i]) for i in range(n)])
        print(f"  umax={umax:5.1f}: cos(slack,hard) med={np.median(cos):.4f} min={np.min(cos):.4f} "
              f"#<0.99={int((cos < 0.99).sum())}/{n}  ({time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
