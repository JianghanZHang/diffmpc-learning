"""B (log-barrier) closed-loop machinery: a custom_vjp MPC solve differentiable w.r.t. BOTH
cost weights and the initial state (the through-rollout path), plus a 50-step rollout and the
per-sample sweep. Linear MPC, 1 SQP iter (the solve is a single QP).

Step 1 of this file is a single-solve gradient verification (B's d(loss)/d(weights,state) vs FD)
— run directly to check correctness before the rollout:
    python research/gradient-quality-diffnmpc/experiments/linear_system/cl_b.py --verify
"""
from __future__ import annotations
import os, sys, argparse

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DL_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
for _p in (os.path.join(_DL_ROOT, "src"), os.path.join(_DL_ROOT, "diffmpc2"),
           os.path.join(_DL_ROOT, "diffmpc2", "benchmarking", "linear-system")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_problem_setup import build_turbompc_linear_problem
from utils import generate_problem_data, N_STATE, N_CTRL
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.turbompc_solver import TurboMPCSolver, ForwardBackend, BackwardBackend
from turbompc.utils.load_params import load_solver_params
from turbompc.solvers.qp_data import QPData, QPCostBlocks, QPEqualityBlocks, QPInequalityBlocks
from turbompc.solvers.qp_utils import ZShape, pack_x
from turbompc.solvers.backward.backward_kkt_jax import solve_backward_kkt
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver

from diffmpc_learning.solvers.central_path_admm import to_one_sided, solve_qp_central_path
from diffmpc_learning.solvers.backward import (
    relaxed_complementarity_weight, augment_D_with_relaxed_ineq,
)

NX, NU = N_STATE, N_CTRL
QK = "weights_penalization_reference_state_trajectory"
RK = "weights_penalization_control_squared"
WK = [QK, RK]
GAMMA, KAPPA, UMAX, HORIZON, SIM_STEPS = 1e4, 1e-6, 1.0, 20, 50
CP = dict(rho_bar=0.1, max_iter=4000, tol=1e-9)


def _solver_params(eps):
    sp = dict(load_solver_params("turbompc.yaml"))
    sp["num_sqp_iteration_max"] = 1; sp["tol_convergence"] = eps; sp["linesearch"] = False
    sp["admm"]["max_iter"] = 4000; sp["admm"]["check_termination_every"] = 1
    sp["admm"]["eps_abs"] = eps; sp["admm"]["eps_rel"] = eps
    return sp


def _solve_reduced_kkt_full(D_aug, E, eq_blocks, x_bar, nx, nu):
    """Solve [[P+G1WG1, Cᵀ],[C,0]][λ_x; λ_f] = [x_bar; 0]; return (λ_x (Np1,n), λ_f multipliers)."""
    Np1, n = D_aug.shape[0], D_aug.shape[1]
    N = Np1 - 1
    eq0 = QPEqualityBlocks(A0=eq_blocks.A0, A_minus=eq_blocks.A_minus, A_plus=eq_blocks.A_plus,
                           c0=jnp.zeros_like(eq_blocks.c0), c=jnp.zeros_like(eq_blocks.c))
    empty = QPInequalityBlocks(G=jnp.zeros((Np1, 0, n), x_bar.dtype),
                               l=jnp.zeros((Np1, 0), x_bar.dtype), u=jnp.zeros((Np1, 0), x_bar.dtype))
    bwd = QPData(cost=QPCostBlocks(D=D_aug, E=E, q=-x_bar), eq=eq0, ineq=empty)
    (lam_s, lam_c), mult = solve_backward_kkt(bwd, ZShape(horizon=N, num_states=nx, num_controls=nu))
    return pack_x(lam_s, lam_c), mult


def make_b_mpc_solve(solver, pp_base, schur, ig, nx, nu, kappa=KAPPA):
    """custom_vjp: b_solve(weights, state) -> (states, controls), diff w.r.t. weights AND state.

    Linear dynamics ⇒ no λᵀ∇²f term; the cost Hessian D is the exact Lagrangian Hessian. The
    state enters the QP via the initial-constraint RHS c0, so dL/dstate is the initial-constraint
    adjoint (sign verified vs FD).
    """
    program = solver.program
    n0 = None  # set from the first build

    def _qp1(weights, state):
        pp_i = {**pp_base, "initial_state": state, QK: weights[QK], RK: weights[RK]}
        qp = solver._build_qp_data(ig.states, ig.controls, pp_i)
        return to_one_sided(qp, GAMMA)

    @jax.custom_vjp
    def b_solve(weights, state):
        qp1 = _qp1(weights, state)
        x, _, _ = solve_qp_central_path(qp1, schur, target_kappa=kappa, slack_weight=GAMMA, **CP)
        return x[:, :nx], x[:, nx:]

    def b_fwd(weights, state):
        qp1 = _qp1(weights, state)
        x, duals, _ = solve_qp_central_path(qp1, schur, target_kappa=kappa, slack_weight=GAMMA, **CP)
        return (x[:, :nx], x[:, nx:]), (weights, state, qp1, x, duals)

    def b_bwd(res, g):
        weights, state, qp1, x, duals = res
        dL_ds, dL_dc = g
        y_g = duals[2]
        W = relaxed_complementarity_weight(qp1, x, y_g, GAMMA)
        D_aug = augment_D_with_relaxed_ineq(qp1.cost.D, qp1.ineq.G, W)   # linear ⇒ no dyn-Hessian
        x_bar = pack_x(dL_ds, dL_dc)
        lam_x, lam_f = _solve_reduced_kkt_full(D_aug, qp1.cost.E, qp1.eq, x_bar, nx, nu)
        lam_states, lam_controls = lam_x[:, :nx], lam_x[:, nx:]

        def contracted(w):
            pp_w = {**pp_base, "initial_state": state, QK: w[QK], RK: w[RK]}
            fx, fu = jax.grad(lambda s, c: program.cost(s, c, pp_w), argnums=(0, 1))(x[:, :nx], x[:, nx:])
            return jnp.sum(fx * lam_states) + jnp.sum(fu * lam_controls)
        dL_dw = jax.tree_util.tree_map(lambda z: -z, jax.grad(contracted)(weights))
        # dL/dstate = c0-adjoint (first n0=nx multipliers); sign verified vs FD
        dL_dstate = lam_f[:nx]
        return (dL_dw, dL_dstate)

    b_solve.defvjp(b_fwd, b_bwd)
    return b_solve


def _setup(seed):
    dyn, pp_t = build_turbompc_linear_problem(horizon=HORIZON, umax=UMAX, n_state=NX, n_ctrl=NU)
    Q, R, A, B, b, x0 = generate_problem_data(8, seed, n_state=NX, n_ctrl=NU)
    pp = dict(pp_t)
    pp["dynamics_state_dot_params"] = {"A": jnp.asarray(A - np.eye(NX)), "B": jnp.asarray(B), "b": jnp.asarray(b)}
    pp[QK] = jnp.asarray(np.diag(Q)); pp[RK] = jnp.asarray(np.diag(R))
    solver = TurboMPCSolver(program=OptimalControlProblem(dynamics=dyn, params=pp), params=_solver_params(1e-9),
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI, backward_backend=BackwardBackend.DIRECT_CUDSS_FFI)
    ig = solver.initial_guess({**pp, "initial_state": jnp.asarray(x0[0])})
    N = solver.program.horizon
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, NX, NU, pcg_params={"max_iter": 400, "tol_epsilon": 1e-12})
    return dyn, pp, solver, ig, schur, jnp.asarray(x0), np.asarray(A - np.eye(NX)), np.asarray(B), np.asarray(b)


def verify():
    """Single-solve: B's d<a,sol>/d(weights,state) vs FD."""
    dyn, pp, solver, ig, schur, x0, A, B, b = _setup(0)
    b_solve = make_b_mpc_solve(solver, pp, schur, ig, NX, NU)
    w = {k: pp[k] for k in WK}
    state = x0[0]
    rng = np.random.default_rng(0)
    Np1 = solver.program.horizon + 1
    a_s = jnp.asarray(rng.standard_normal((Np1, NX))); a_c = jnp.asarray(rng.standard_normal((Np1, NU)))
    def loss(w_, st_):
        s, c = b_solve(w_, st_); return jnp.sum(a_s * s) + jnp.sum(a_c * c)
    (dW, dstate) = jax.grad(loss, argnums=(0, 1))(w, state)
    # FD
    def L(w_, st_): return float(loss(w_, st_))
    eps = 1e-5
    # state FD
    fd_state = np.zeros(NX)
    for i in range(NX):
        sp = np.array(state); sp[i] += eps; sm = np.array(state); sm[i] -= eps
        fd_state[i] = (L(w, jnp.asarray(sp)) - L(w, jnp.asarray(sm))) / (2 * eps)
    # weight FD (a couple entries)
    fd_R = np.zeros(NU)
    for i in range(NU):
        Rp = np.array(w[RK]); Rp[i] += eps; Rm = np.array(w[RK]); Rm[i] -= eps
        fd_R[i] = (L({**w, RK: jnp.asarray(Rp)}, state) - L({**w, RK: jnp.asarray(Rm)}, state)) / (2 * eps)
    print("dL/dstate AD:", np.asarray(dstate))
    print("dL/dstate FD:", fd_state)
    print("  cos=", float(np.asarray(dstate)@fd_state/(np.linalg.norm(np.asarray(dstate))*np.linalg.norm(fd_state)+1e-30)),
          " rel=", float(np.linalg.norm(np.asarray(dstate)-fd_state)/(np.linalg.norm(fd_state)+1e-30)))
    print("dL/dR AD:", np.asarray(dW[RK]), " FD:", fd_R)


def _flat(d): return np.concatenate([np.asarray(d[k]).reshape(-1) for k in WK])
def _cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
def _rel(a, b): return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def run_B(n_samples, seed, kappa=KAPPA):
    import time
    dyn, pp_t = build_turbompc_linear_problem(horizon=HORIZON, umax=UMAX, n_state=NX, n_ctrl=NU)
    Q, R, A, B, b, x0 = generate_problem_data(n_samples, seed, n_state=NX, n_ctrl=NU)
    A_sd = jnp.asarray(A - np.eye(NX)); B_m = jnp.asarray(B); b_v = jnp.asarray(b)
    pp = dict(pp_t)
    pp["dynamics_state_dot_params"] = {"A": A_sd, "B": B_m, "b": b_v}
    pp[QK] = jnp.asarray(np.diag(Q)); pp[RK] = jnp.asarray(np.diag(R))
    x0_batch = jnp.asarray(x0); w = {k: pp[k] for k in WK}
    solver = TurboMPCSolver(program=OptimalControlProblem(dynamics=dyn, params=pp), params=_solver_params(1e-9),
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI, backward_backend=BackwardBackend.DIRECT_CUDSS_FFI)
    ig = solver.initial_guess({**pp, "initial_state": x0_batch[0]})
    N = solver.program.horizon
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, NX, NU, pcg_params={"max_iter": 400, "tol_epsilon": 1e-12})
    b_solve = make_b_mpc_solve(solver, pp, schur, ig, NX, NU, kappa=kappa)

    def rollout_cost(weights, state):
        def step(carry, _):
            st, cost = carry
            s, c = b_solve(weights, st)              # MPC solve from current state
            u0 = c[0]
            new_st = st + 1.0 * (A_sd @ st + B_m @ u0 + b_v)   # Euler step, dt=1 (= linear dynamics)
            return (new_st, cost + jnp.sum(new_st ** 2) + jnp.sum(u0 ** 2)), None
        (_, total), _ = jax.lax.scan(step, (state, jnp.asarray(0.0)), None, length=SIM_STEPS)
        return total

    cost_vec = jax.jit(jax.vmap(rollout_cost, in_axes=(None, 0)))
    gB_fn = jax.jit(jax.vmap(jax.grad(rollout_cost, argnums=0), in_axes=(None, 0)))

    t = time.time(); gB_d = gB_fn(w, x0_batch)
    gB = np.stack([_flat({k: gB_d[k][i] for k in WK}) for i in range(n_samples)])
    print(f"[B AD] {time.time()-t:.0f}s")

    t = time.time(); eps_seq = (3e-5, 1e-5, 3e-6, 1e-6); per_eps = []
    for eps in eps_seq:
        cols = []
        for k in WK:
            base = np.asarray(w[k], float)
            for i in range(base.size):
                ap = base.copy(); ap[i] += eps; am = base.copy(); am[i] -= eps
                wp = {**w, k: jnp.asarray(ap)}; wm = {**w, k: jnp.asarray(am)}
                cols.append((np.asarray(cost_vec(wp, x0_batch)) - np.asarray(cost_vec(wm, x0_batch))) / (2 * eps))
        per_eps.append(np.stack(cols, axis=1))
    gFD = np.array(per_eps[-1]); flagged = np.zeros(n_samples, bool)
    for n in range(n_samples):
        ok_all = True
        for j in range(per_eps[0].shape[1]):
            seq = [pe[n, j] for pe in per_eps]; chosen, ok = seq[-1], False
            for a_, b_ in zip(seq[:-1], seq[1:]):
                if abs(a_ - b_) <= 1e-2 * abs(b_) + 1e-7: chosen, ok = a_, True; break
            gFD[n, j] = chosen; ok_all = ok_all and ok
        flagged[n] = not ok_all
    print(f"[B FD] {time.time()-t:.0f}s  flagged {int(flagged.sum())}/{n_samples}")

    cosB = np.array([_cos(gB[n], gFD[n]) for n in range(n_samples)])
    relB = np.array([_rel(gB[n], gFD[n]) for n in range(n_samples)])
    keep = ~flagged
    print(f"\n=== B (log-barrier) vs convergence-checked FD, {int(keep.sum())}/{n_samples} non-flagged ===")
    print(f"  cos:    median={np.median(cosB[keep]):.5f}  min={np.min(cosB[keep]):.5f}  "
          f"#<0.99={int((cosB[keep]<0.99).sum())}  #<0={int((cosB[keep]<0).sum())}")
    print(f"  rel_l2: median={np.median(relB[keep]):.2e}  max={np.max(relB[keep]):.2e}")
    order = np.argsort(cosB)
    print("  worst-B samples (idx, cos, rel, flagged):")
    for i in order[:10]:
        print(f"    sample {i:3d}: cos={cosB[i]:+.5f} rel={relB[i]:.2e} flagged={bool(flagged[i])}")
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    np.savez(os.path.join(_HERE, "results", "closed_loop_B.npz"), cosB=cosB, relB=relB, flagged=flagged, gB=gB, gFD=gFD)
    print("saved results/closed_loop_B.npz")
    return cosB, relB, flagged


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.verify:
        verify()
    if a.run:
        print(f"Closed-loop B (sim_steps={SIM_STEPS}) gradient accuracy vs FD: n_samples={a.n_samples}")
        run_B(a.n_samples, a.seed)
