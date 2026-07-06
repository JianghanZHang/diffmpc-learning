"""Three-way comparison of the K=1 cartpole gradient d(C)/d(Q,R):
   turbompc-AD  vs  acados-exact (results/acados_grad.npz)  vs  convergence-verified FD.

Step A (always): compute turbompc AD + converged-FD on the SHARED 128 samples
   (results/data/x0_cartpole_128.npy), at K=1, tol=1e-9; save results/data/turbompc_grad.npz.
Step B (if results/data/acados_grad.npz exists): build the 3-way cos/relmag table + verdict ->
   results/acados_comparison.md.

Run LOCALLY (resolves turbompc from the consolidated diffmpc2/ checkout via
benchmark_cartpole_coupling's path shim).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")           # markdown result docs live at results/ root
DATA = os.path.join(RES, "data")              # npz/npy/csv live in results/data/
os.makedirs(DATA, exist_ok=True)
sys.path.insert(0, HERE)


def cosine(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def relmag(a, b):  # |a|/|b|
    return float(np.linalg.norm(a) / (np.linalg.norm(b) + 1e-30))


def compute_turbompc():
    from jax import config as jc; jc.update("jax_enable_x64", True)
    import jax, jax.numpy as jnp
    from jax import jit, vmap, grad, checkpoint
    import benchmark_cartpole_coupling as B
    from turbompc.solvers.turbompc_solver import (
        TurboMPCSolver, parse_forward_backend, parse_backward_backend)
    from turbompc.problems.optimal_control_problem import OptimalControlProblem

    x0 = np.load(os.path.join(DATA, "x0_cartpole_128.npy"))
    dyn, pp = B.build_cartpole_problem(25, 1e7, 0.04)
    w = {k: pp[k] for k in B.WEIGHT_KEYS}
    reward = B.make_reward(jnp.zeros(4), jnp.zeros(1))
    fwd, bwd = parse_forward_backend("admm_fused_cudss"), parse_backward_backend("direct_cudss_ffi")
    s = TurboMPCSolver(program=OptimalControlProblem(dynamics=dyn, params=dict(pp)),
                       params=B.solver_params(1e-9, 1e-9, 300, 1000),
                       forward_backend=fwd, backward_backend=bwd, use_full_hessian=True)
    init = s.solve(s.initial_guess(pp), problem_params=pp, weights=w)
    xj = jnp.asarray(x0)
    # export turbompc's converged solution trajectory (to warm-start acados at the SAME point)
    sol_fn = jit(vmap(lambda x: s.solve(init, {**pp, "initial_state": x}, w)))
    sols = sol_fn(xj); jax.block_until_ready(sols.controls)
    st, ct = np.asarray(sols.states), np.asarray(sols.controls)   # (128,26,4),(128,26,1)
    np.savez(os.path.join(DATA, "turbompc_solution.npz"), states=st, controls=ct)
    print(f"[turbompc] saved turbompc_solution.npz states{st.shape} controls{ct.shape} "
          f"u0[:3]={ct[:3,0,0]}")
    rs, rb = B.make_rollouts(s, dyn, pp, init, 1, reward)
    ad = jit(vmap(grad(checkpoint(lambda ww, x: rs(x, ww)[0]), argnums=0), in_axes=(None, 0)))(w, xj)
    jax.block_until_ready(ad)
    gAD = np.concatenate([np.asarray(ad[k]).reshape(x0.shape[0], -1) for k in B.WEIGHT_KEYS], axis=1)
    gGT, flag = B.fd_per_state_converged(rb, xj, w, B.WEIGHT_KEYS, [1e-5, 1e-6, 3e-7, 1e-7], 1e-2)
    np.savez(os.path.join(DATA, "turbompc_grad.npz"), gAD=gAD, gGT=gGT, flagged=flag)
    print(f"[turbompc] saved turbompc_grad.npz  gAD{gAD.shape} gGT{gGT.shape} flagged={int(flag.sum())}")
    return gAD, gGT, flag


def three_way(gAD, gGT, flag):
    z = np.load(os.path.join(DATA, "acados_grad.npz"))
    gAC = np.concatenate([z["grad_Q"], z["grad_R"].reshape(-1, 1)], axis=1)  # (128,5) [Q,R]
    n = gGT.shape[0]
    # comparison set: acados forward reached turbompc's SAME KKT point AND the FD is trustworthy
    matched = z["matched"].astype(bool) if "matched" in z else np.ones(n, bool)
    use = np.where(matched & (~flag.astype(bool)))[0]

    def stats(A, Bv, label):
        cs = np.array([cosine(A[i], Bv[i]) for i in use])
        rm = np.array([relmag(A[i], Bv[i]) for i in use])
        return f"{label:24s} cos med={np.median(cs):+.4f} min={cs.min():+.4f}  relmag med={np.median(rm):.4f}"

    lines = ["# acados vs turbompc vs converged-FD — K=1 cartpole d(C)/d(Q,R)\n",
             f"Comparison set = {use.size}/{n} samples where acados's INDEPENDENT GN forward reached "
             f"turbompc's same KKT point (|u0 diff|<1e-4) AND the converged-FD is non-flagged. "
             f"(Different-basin / flagged samples excluded — not the same point.) "
             f"cos = gradient direction; relmag = ||a||/||b|| (magnitude ratio).\n",
             "```",
             stats(gAC, gGT, "acados-exact vs FD"),
             stats(gAD, gGT, "turbompc-AD  vs FD"),
             stats(gAC, gAD, "acados-exact vs turbompc-AD"),
             "```",
             "\nVERDICT: if acados-exact ~ FD (relmag~1) while turbompc-AD ~ 0.85*FD, the ~15% gap is a "
             "turbompc backward error. If acados-exact ~ turbompc-AD (both ~0.85*FD), it is NOT a "
             "turbompc-specific bug (two independent exact-Hessian adjoints agree; the FD is the outlier)."]
    out = "\n".join(lines) + "\n"
    print(out)
    open(os.path.join(RES, "acados_comparison.md"), "w").write(out)
    print(f"wrote {RES}/acados_comparison.md")


if __name__ == "__main__":
    g = compute_turbompc()
    if os.path.exists(os.path.join(DATA, "acados_grad.npz")):
        three_way(*g)
    else:
        print("(acados_grad.npz not present yet — run the acados --gradient step in Docker, "
              "then re-run this for the 3-way table.)")
