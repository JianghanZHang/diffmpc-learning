"""Linear-system ADMM-tolerance gradient sweep (RQ2 / H2.2).

Testbed: diffmpc's random *linear* MPC (`build_turbompc_linear_problem`). Because the
dynamics are linear the SQP outer loop is exactly **1 iteration** (the NLP is a single QP),
so the inner-QP **ADMM convergence tolerance is the only solve-accuracy knob** — no SQP/NLP
tolerance to confound. We sweep it and ask how the backward gradient degrades, for two
backwards, vs the true (tight-solve) gradient:

  A  = TurboMPC analytic backward, use_slack=True (indicator + quadratic slack penalty gamma)
  B  = log-barrier smoothed backward (central_path_nlp_grad, fixed kappa, matched gamma)
  GT = convergence-checked FD of the TIGHTLY-solved slack forward (CLAUDE.md protocol)

One fixed random linear system; the **samples are initial states x0** (the diffmpc-as-policy
gradient varies with the operating point / active set). Open-loop single QP, box control
bounds (umax). Differentiable parameter = cost weights (Q diag, R diag); loss is a fixed
quadratic on the solution (distinct from the swept Q/R -> no envelope degeneracy).

Run (from repo root, GPU + cuDSS + x64):
    export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    python research/gradient-quality-diffnmpc/experiments/linear_system/sweep_admm_tolerance.py --smoke
    python research/gradient-quality-diffnmpc/experiments/linear_system/sweep_admm_tolerance.py --save_results
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DL_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))   # diffmpc-learning/
for _p in (os.path.join(_DL_ROOT, "src"),
           os.path.join(_DL_ROOT, "diffmpc2"),
           os.path.join(_DL_ROOT, "diffmpc2", "benchmarking", "linear-system")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_problem_setup import build_turbompc_linear_problem  # noqa: E402
from utils import generate_problem_data, N_STATE, N_CTRL  # noqa: E402
from turbompc.problems.optimal_control_problem import (  # noqa: E402
    OptimalControlProblem, OptimalControlProblemSlack,
)
from turbompc.solvers.turbompc_solver import (  # noqa: E402
    TurboMPCSolver, ForwardBackend, BackwardBackend,
)
from turbompc.utils.load_params import load_solver_params  # noqa: E402
from diffmpc_learning.solvers.backward import central_path_nlp_grad  # noqa: E402

NX, NU = N_STATE, N_CTRL
QK = "weights_penalization_reference_state_trajectory"
RK = "weights_penalization_control_squared"
WK = [QK, RK]
GAMMA = 1.0e4        # slack penalty (near-hard box), matched between A and B
KAPPA = 1.0e-6       # log-barrier parameter for B (fixed)
TIGHT_EPS = 1.0e-10  # "tight" ADMM tolerance defining the true-gradient reference


def loss(states, controls):       # fixed quadratic; weights != swept Q/R -> non-degenerate
    return 0.5 * jnp.sum(states ** 2) + 0.5 * jnp.sum(controls ** 2)


def loss_grad(states, controls):
    return jax.grad(loss, argnums=(0, 1))(states, controls)


def _flat(d):
    return np.concatenate([np.asarray(d[k]).reshape(-1) for k in WK])


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _solver_params(admm_eps):
    """1-SQP-iteration linear solve; the ADMM eps is the only solve-accuracy knob."""
    sp = dict(load_solver_params("turbompc.yaml"))
    sp["num_sqp_iteration_max"] = 1          # linear -> 1 SQP iteration is exact
    sp["tol_convergence"] = 1e-12            # moot at 1 iter
    sp["linesearch"] = False
    sp["admm"]["max_iter"] = 50000
    sp["admm"]["check_termination_every"] = 1
    sp["admm"]["eps_abs"] = admm_eps
    sp["admm"]["eps_rel"] = admm_eps
    return sp


def _slack_solver(dyn, pp_slack, admm_eps):
    return TurboMPCSolver(
        program=OptimalControlProblemSlack(dynamics=dyn, params=pp_slack),
        params=_solver_params(admm_eps),
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
        backward_backend=BackwardBackend.DIRECT_JAX_DENSE)


def _fd_converged_batch(loss_vec, w, x0_batch, eps_seq, plateau_rtol=1e-2, atol=1e-7):
    """Per-x0 convergence-checked central-difference gradient (CLAUDE.md).

    loss_vec(w_dict, x0_batch) -> (N,) losses (vmapped over x0). Returns
    (grad (N, wdim), converged (N,) bool): per scalar weight, the largest eps agreeing with
    the next-smaller (rel OR abs); a sample is flagged if any entry never plateaus.
    """
    N = x0_batch.shape[0]
    per_eps = []                                   # list of (N, wdim)
    for eps in eps_seq:
        cols = []
        for k in WK:
            base = np.asarray(w[k], dtype=float)
            for i in range(base.size):
                ap = base.copy(); ap[i] += eps
                am = base.copy(); am[i] -= eps
                wp = {**w, k: jnp.asarray(ap)}; wm = {**w, k: jnp.asarray(am)}
                cols.append((np.asarray(loss_vec(wp, x0_batch))
                             - np.asarray(loss_vec(wm, x0_batch))) / (2 * eps))
        per_eps.append(np.stack(cols, axis=1))     # (N, wdim)
    grad = np.array(per_eps[-1]); converged = np.zeros(N, bool)
    for n in range(N):
        ok_all = True
        for j in range(per_eps[0].shape[1]):
            seq = [pe[n, j] for pe in per_eps]
            chosen, ok = seq[-1], False
            for a, b in zip(seq[:-1], seq[1:]):
                if abs(a - b) <= plateau_rtol * abs(b) + atol:
                    chosen, ok = a, True
                    break
            grad[n, j] = chosen
            ok_all = ok_all and ok
        converged[n] = ok_all
    return grad, converged


def run(n_samples, tolerances, horizon, umax, eps_seq, seed, verbose=True):
    # ---- one fixed random linear system; samples = initial states ----
    dyn, pp_t = build_turbompc_linear_problem(horizon=horizon, umax=umax, n_state=NX, n_ctrl=NU)
    Q, R, A, B, b, x0 = generate_problem_data(n_samples, seed, n_state=NX, n_ctrl=NU)
    pp_t = dict(pp_t)
    pp_t["dynamics_state_dot_params"] = {
        "A": jnp.asarray(A - np.eye(NX)), "B": jnp.asarray(B), "b": jnp.asarray(b)}
    pp_t[QK] = jnp.asarray(np.diag(Q)); pp_t[RK] = jnp.asarray(np.diag(R))
    pp_slack_t = {**pp_t, "use_slack_variables": True, "slack_penalization_weight": GAMMA}
    x0_batch = jnp.asarray(x0)                        # (N, nx)
    w = {k: pp_t[k] for k in WK}
    wdim = sum(np.asarray(w[k]).size for k in WK)

    # B helper solver (eps irrelevant: B uses its own central-path solver, cp_tol swept)
    solver_plain = TurboMPCSolver(
        program=OptimalControlProblem(dynamics=dyn, params=pp_t), params=_solver_params(TIGHT_EPS),
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
        backward_backend=BackwardBackend.DIRECT_JAX_DENSE)

    def make_loss_vec(solver, pp_slack):
        ig = solver.initial_guess(pp_slack)
        def lv(ww, x0b):
            def one(x0i):
                sol = solver.solve(ig, pp_slack, {**ww, "initial_state": x0i})
                return loss(sol.states, sol.controls)
            return jax.vmap(one)(x0b)
        return jax.jit(lv)

    # ---- GT: convergence-checked FD of the TIGHT slack forward, per x0 ----
    t = time.time()
    gt_solver = _slack_solver(dyn, pp_slack_t, TIGHT_EPS)
    gt_loss_vec = make_loss_vec(gt_solver, pp_slack_t)
    gGT, conv = _fd_converged_batch(gt_loss_vec, w, x0_batch, eps_seq)
    maxu = np.asarray(jax.jit(jax.vmap(lambda x0i: jnp.max(jnp.abs(
        gt_solver.solve(gt_solver.initial_guess(pp_slack_t), pp_slack_t,
                        {**w, "initial_state": x0i}).controls))))(x0_batch))
    keep = conv
    print(f"[GT] {time.time()-t:.0f}s  FD plateau {int(keep.sum())}/{n_samples}  "
          f"max|u| in [{maxu.min():.3f},{maxu.max():.3f}] (umax={umax})")

    # ---- sweep ADMM tolerance: A (slack eps) and B (log-barrier cp_tol) ----
    rows = []
    for eps in tolerances:
        t = time.time()
        # A: analytic slack backward at this ADMM eps, vmapped over x0
        sa = _slack_solver(dyn, pp_slack_t, eps)
        iga = sa.initial_guess(pp_slack_t)
        def loss_a(ww, x0i):
            sol = sa.solve(iga, pp_slack_t, {**ww, "initial_state": x0i})
            return loss(sol.states, sol.controls)
        gA_fn = jax.jit(jax.vmap(jax.grad(loss_a, argnums=0), in_axes=(None, 0)))
        gA_d = gA_fn(w, x0_batch)
        gA = np.stack([_flat({k: gA_d[k][n] for k in WK}) for n in range(n_samples)])
        admm_iters_A = float(np.median(np.asarray(jax.jit(jax.vmap(lambda x0i: jnp.sum(
            sa.solve(iga, pp_slack_t, {**w, "initial_state": x0i}).admm_iters)))(x0_batch))))

        # B: log-barrier backward at matched cp_tol, looped over x0
        cfg = dict(slack_weight=GAMMA, target_kappa=KAPPA, conv_slack_weight=GAMMA,
                   kappa_anneal=False, linesearch=False, max_sqp_iter=1, tol=1e-9,
                   cp_tol=eps, cp_max_iter=50000, jit_inner=True)
        gB = np.zeros_like(gA); cp_iters_B = []
        for n in range(n_samples):
            pp_i = {**pp_t, "initial_state": x0_batch[n]}
            _, gBd, info = central_path_nlp_grad(solver_plain, pp_i, w, loss_grad, **cfg)
            gB[n] = _flat(gBd); cp_iters_B.append(int(info["iters"]))

        cosA = np.array([_cos(gA[n], gGT[n]) for n in range(n_samples)])
        cosB = np.array([_cos(gB[n], gGT[n]) for n in range(n_samples)])
        relA = np.array([_rel(gA[n], gGT[n]) for n in range(n_samples)])
        relB = np.array([_rel(gB[n], gGT[n]) for n in range(n_samples)])
        row = dict(eps=eps,
                   cosA=float(np.median(cosA[keep])), relA=float(np.median(relA[keep])),
                   relA_max=float(np.max(relA[keep])), itersA=admm_iters_A,
                   cosB=float(np.median(cosB[keep])), relB=float(np.median(relB[keep])),
                   relB_max=float(np.max(relB[keep])), itersB=float(np.median(cp_iters_B)))
        rows.append(row)
        if verbose:
            print(f"  eps={eps:.0e}  A: cos={row['cosA']:.6f} rel={row['relA']:.2e} "
                  f"iters={row['itersA']:.0f}   B: cos={row['cosB']:.6f} rel={row['relB']:.2e} "
                  f"iters={row['itersB']:.0f}   ({time.time()-t:.0f}s)")
    return rows, int(keep.sum()), n_samples


def write_results(rows, n_keep, n_samples, args, outdir):
    os.makedirs(outdir, exist_ok=True)
    cols = ["eps", "cosA", "relA", "relA_max", "itersA", "cosB", "relB", "relB_max", "itersB"]
    with open(os.path.join(outdir, "summary.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.6e}" if c != "eps" else f"{r['eps']:.0e}" for c in cols) + "\n")
    md = [
        "# Linear-system ADMM-tolerance gradient sweep",
        "",
        f"Fixed random linear MPC (nx={NX}, nu={NU}, horizon={args.horizon}, umax={args.umax}); "
        f"**1 SQP iteration**; samples = {n_samples} initial states ({n_keep} with FD plateau).",
        f"gamma={GAMMA:.0e} (matched A/B), kappa={KAPPA:.0e} (B), GT = convergence-checked FD of the "
        f"tight (eps={TIGHT_EPS:.0e}) slack forward. A = TurboMPC indicator+slack backward; "
        "B = log-barrier smoothed backward. Metrics are medians over non-flagged samples.",
        "",
        "| ADMM eps | A cos | A rel_l2 | A rel_max | A iters | B cos | B rel_l2 | B rel_max | B iters |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['eps']:.0e} | {r['cosA']:.6f} | {r['relA']:.2e} | {r['relA_max']:.2e} | "
                  f"{r['itersA']:.0f} | {r['cosB']:.6f} | {r['relB']:.2e} | {r['relB_max']:.2e} | "
                  f"{r['itersB']:.0f} |")
    md += ["",
           "A (indicator+slack) is the consistent backward of the slack forward: at tight ADMM eps it "
           "matches GT; as eps loosens the forward QP is under-solved and the gradient degrades. B "
           "(log-barrier) carries an O(kappa) bias even at tight eps. Compare the degradation of A vs B "
           "against the achieved ADMM iteration count.", ""]
    with open(os.path.join(outdir, "linear_system_sweep.md"), "w") as f:
        f.write("\n".join(md))
    print(f"\nwrote {outdir}/summary.csv + linear_system_sweep.md")


def main():
    p = argparse.ArgumentParser(description="Linear-system ADMM-tolerance gradient sweep (A vs B vs FD)")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--umax", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tolerances", type=float, nargs="+",
                   default=[1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10])
    p.add_argument("--eps_seq", type=float, nargs="+", default=[1e-3, 3e-4, 1e-4])
    p.add_argument("--smoke", action="store_true", help="2 tolerances, 4 samples")
    p.add_argument("--save_results", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.tolerances = [1e-2, 1e-10]; args.n_samples = 4

    print(f"Linear-system ADMM-tol sweep: nx={NX} nu={NU} H={args.horizon} umax={args.umax} "
          f"n_samples={args.n_samples} tols={args.tolerances}")
    rows, n_keep, n_samples = run(args.n_samples, args.tolerances, args.horizon, args.umax,
                                  args.eps_seq, args.seed)
    if args.save_results:
        write_results(rows, n_keep, n_samples, args, os.path.join(_HERE, "results"))


if __name__ == "__main__":
    main()
