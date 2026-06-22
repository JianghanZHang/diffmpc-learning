"""Per-sample gradient accuracy vs closed-loop coupling K and NLP-KKT tolerance (cartpole/quadrotor).

Ground truth = a CONVERGENCE-VERIFIED per-sample FD gradient: the FD step is shrunk until the
per-sample gradient plateaus (CLAUDE.md protocol); samples that never plateau are FLAGGED (they
sit on a discontinuity of the nonconvex policy map, where dF/dw is ill-defined) and excluded
from cos scoring. The NLP-tol sweep verifies the forward REACHES each set tol (no early exit)
without overshooting (achieved KKT ~ set tol, reported per (K,tol)). `--system` selects cartpole
(run) or quadrotor (set up, not run this session); `--umax 1e7` => no active inequality constraint.

  python benchmark_cartpole_coupling.py --system cartpole --umax 1e7 --calibrate     # smoke test
  python benchmark_cartpole_coupling.py --system cartpole --umax 1e7 --save_results   # full sweep
  python benchmark_cartpole_coupling.py --system quadrotor --calibrate                # quad set-up check

--- original design notes (cartpole swing-up) ---
Gradient accuracy vs CLOSED-LOOP COUPLING length on a nonlinear (cartpole) system.

Question: when a diffmpc-as-policy is trained by BPTT over a K-step closed-loop episode,
the gradient threads through K chained MPC solves coupled by the nonlinear closed-loop
dynamics. How does gradient accuracy (AD vs FD cosine) degrade as the coupling length K grows?

Design (see discussion):
  - System: cartpole (nx=4, nu=1), RK4, swing UP to upright from the pole hanging DOWN
    (theta_0 ~ pi) -- a maximal, highly-nonlinear perturbation with a poor (upright) warm-start.
  - Independent variables: coupling steps K in {1, 5, 10, 20} (closed-loop rollout length),
    crossed with the SET NLP-KKT tolerance --nlp_tol (e.g. 1e-1 1e-3 1e-5 1e-9).
  - Poor warm-start every MPC cycle: each solve cold-starts from the FIXED init_solution
    (the upright/all-zeros trajectory), never the previous step (warm_start=False). From the
    pole-down start this is a deliberately bad guess, so SQP must do real work to reach tol.
  - SQP iteration budget is a generous CEILING (--sqp_iter, non-binding): SQP terminates at
    the SET tol_convergence, so a loose tol genuinely leaves the forward under-converged.
  - Fixed: large ADMM budget (inner QP held tight), fd_eps=1e-5.
  - Differentiate w.r.t. the MPC cost weights Q (4) + R (1)  [diffmpc-as-policy gradient].
  - Constraints: --umax sets control bounds; run LOOSE first (bounds rarely active, isolates
    pure coupling) then TIGHT (couples with active-set).
  - Metric: cos / rel_l2 of the SWEPT-tolerance AD (backward) gradient against a FIXED ground
    truth = the FD gradient at gt_tol (always converged, default 1e-9). The GT is NOT the
    same-tolerance FD -- that only checks backward self-consistency and both drift together;
    comparing against the converged GT measures the actual error from under-converging.

Run from benchmarking/linear-system (cuDSS 0.7.1 on LD_LIBRARY_PATH):
  python benchmark_cartpole_coupling.py --umax 20 --calibrate     # verify convergence+stability
  python benchmark_cartpole_coupling.py --umax 20 --save_results   # loose sweep
  python benchmark_cartpole_coupling.py --umax 2  --save_results   # tight sweep
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

import numpy as np
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_threefry_partitionable", True)

import jax
import jax.numpy as jnp
from jax import checkpoint, jit, vmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Resolve `turbompc` from the consolidated diffmpc2/ checkout (a sibling of `research/` under
# diffmpc-learning/), NOT the pip-editable /home/jianghan/Workspace/diffmpc2 which ships a
# different `diffmpc` package. This file lives in research/.../experiments/cartpole/; walk up
# 4 dirs to diffmpc-learning/, then into diffmpc2/.
_DL_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))   # diffmpc-learning/
_SOLVER_ROOT = os.path.join(_DL_ROOT, "diffmpc2")
sys.path.insert(0, _SOLVER_ROOT)
from turbompc.dynamics.cartpole_dynamics import (  # noqa: E402
    CartpoleDynamics,
    default_parameters as CP_PARAMS,
    default_state_dot_parameters as CP_DYN,
)
from turbompc.dynamics.quadrotor_dynamics import (  # noqa: E402
    QuadrotorDynamics,
    quadrotor_parameters as QR_PARAMS,
    quadrotor_state_dot_parameters as QR_DYN,
)
from turbompc.dynamics.integrators import DiscretizationScheme  # noqa: E402
from turbompc.problems.optimal_control_problem import OptimalControlProblem  # noqa: E402
from turbompc.solvers.turbompc_solver import (  # noqa: E402
    TurboMPCSolver, parse_forward_backend, parse_backward_backend)
from turbompc.utils.load_params import load_solver_params  # noqa: E402
from turbompc.utils.timing import ProblemConfig, build_rollout_fn  # noqa: E402
# Eagerly register the fused-cuDSS ADMM FFI handler IF its compiled .so is present (the forward
# backend admm_fused_cudss otherwise imports it lazily inside the traced while_loop body,
# admm.py:750, which can fire too late -> "No FFI handler registered for admm_cudss_cuda_f64").
# The cuDSS FFI .so is an OPTIONAL build (`make` in diffmpc2/); when absent the benchmark runs on
# the pure-JAX backends (GPU-capable, no FFI build needed) which are the defaults below. NOTE: the
# release-cleanup .cu sources are the cuDSS-0.8 API; a working 0.7.1 build is required for the cuDSS
# backends on sm_120 (CARTPOLE_COUPLING_HANDOFF.md §2 — the 0.7.1 reverts are no longer vendored).
try:
    import turbompc.solvers.admm.admm_cudss_ffi_backend  # noqa: E402,F401
    _CUDSS_FFI_AVAILABLE = True
except Exception as _cudss_exc:  # .so not built / cuDSS version mismatch
    _CUDSS_FFI_AVAILABLE = False
    print(f"[benchmark] cuDSS FFI unavailable ({type(_cudss_exc).__name__}: {_cudss_exc});\n"
          f"            using pure-JAX backends. Build it in diffmpc2/ and pass --fwd/--bwd to use cuDSS.")

NX, NU = 4, 1
WEIGHT_KEYS = [
    "weights_penalization_reference_state_trajectory",  # Q diag (4)
    "weights_penalization_control_squared",             # R diag (1)
]
# Stabilizing nominal cost weights (penalize angle most). These are the θ we differentiate.
Q_NOMINAL = jnp.array([1.0, 1.0, 10.0, 1.0])
R_NOMINAL = jnp.array([0.1])


def make_reward(ref_state, ref_control):
    """Closed-loop evaluation cost (the rollout objective we differentiate): the rollout
    accumulates +(||state - ref_state||^2 + ||control - ref_control||^2). Cartpole reference is
    the origin (upright); quadrotor reference is hover (identity quaternion + hover thrust)."""
    ref_state = jnp.asarray(ref_state)
    ref_control = jnp.asarray(ref_control)
    def reward(state, control):
        return -(jnp.sum((state - ref_state) ** 2) + jnp.sum((control - ref_control) ** 2))
    return reward


def build_cartpole_problem(horizon: int, umax: float, dt: float) -> tuple:
    dynamics = CartpoleDynamics(CP_PARAMS)
    params: Dict[str, Any] = {
        "horizon": horizon,
        "discretization_resolution": dt,
        "discretization_scheme": int(DiscretizationScheme.RUNGEKUTTA4),
        "initial_state": jnp.zeros((NX,)),                     # overridden per seed
        "initial_guess_final_state": jnp.zeros((NX,)),
        "reference_state_trajectory": jnp.zeros((horizon + 1, NX)),   # upright at origin
        "reference_control_trajectory": jnp.zeros((horizon + 1, NU)),
        "penalize_control_reference": False,
        "rescale_optimization_variables": False,
        "constrain_initial_control": False,
        "initial_control": jnp.zeros((NU,)),
        "state_rescaling_min": -jnp.ones((NX,)),
        "state_rescaling_max": jnp.ones((NX,)),
        "control_rescaling_min": -jnp.ones((NU,)),
        "control_rescaling_max": jnp.ones((NU,)),
        "weights_penalization_reference_state_trajectory": Q_NOMINAL,
        "weights_penalization_final_state": jnp.zeros((NX,)),
        "weights_penalization_control_squared": R_NOMINAL,
        "weights_penalization_control_rate": jnp.zeros((NU,)),
        "state_min_bounds": -jnp.ones((NX,)) * 1.0e7,
        "state_max_bounds": jnp.ones((NX,)) * 1.0e7,
        "control_min_bounds": -jnp.ones((NU,)) * umax,
        "control_max_bounds": jnp.ones((NU,)) * umax,
        "dynamics_state_dot_params": {k: jnp.asarray(float(v)) for k, v in CP_DYN.items()},
    }
    return dynamics, params


def generate_cartpole_x0(batch_size: int, seed: int, scale: float = 1.0) -> np.ndarray:
    """Initial states with the POLE HANGING DOWN (theta ~ pi): the swing-up regime.
    State = [x, x_dot, theta, theta_dot]; the MPC reference is upright (theta=0), so starting
    at the bottom (theta=pi) is the maximal, highly-nonlinear perturbation. The fixed
    upright warm-start (init_solution solved at the origin) is far from the swing-up solution,
    so the SQP needs many iterations and a loose NLP-KKT tolerance genuinely under-converges.
    `scale` sets the random spread about the down position (1.0 => ~8.6 deg pole jitter)."""
    rng = np.random.default_rng(seed)
    base = np.array([0.0, 0.0, np.pi, 0.0])           # pole hanging straight DOWN
    std = np.array([0.2, 0.1, 0.15, 0.1]) * scale
    return base + rng.normal(0.0, 1.0, size=(batch_size, NX)) * std


# ---------------------------------------------------------------------------------------------
# Quadrotor (13 states, 4 controls; hover reference). SET UP for a parallel experiment; not run
# this session. State = [pos(3), vel(3), quat(4), omega(3)]; control = [thrust, tau_x, tau_y, tau_z].
# ---------------------------------------------------------------------------------------------
QR_NX, QR_NU = 13, 4
QR_Q = jnp.array([10., 10., 10.,  1., 1., 1.,  10., 10., 10., 10.,  1., 1., 1.])  # pos,vel,quat,omega
QR_R = jnp.array([0.1, 0.1, 0.1, 0.1])
QR_HOVER_THRUST = float(QR_DYN["mass"]) * 9.81                                    # m*g
QR_REF_STATE = jnp.array([0., 0., 0.,  0., 0., 0.,  1., 0., 0., 0.,  0., 0., 0.])  # hover, identity quat
QR_REF_CONTROL = jnp.array([QR_HOVER_THRUST, 0., 0., 0.])


def build_quadrotor_problem(horizon: int, umax: float, dt: float) -> tuple:
    dynamics = QuadrotorDynamics(QR_PARAMS)
    params: Dict[str, Any] = {
        "horizon": horizon,
        "discretization_resolution": dt,
        "discretization_scheme": int(DiscretizationScheme.RUNGEKUTTA4),
        "initial_state": QR_REF_STATE,                              # hover (valid quaternion!); overridden per seed
        "initial_guess_final_state": QR_REF_STATE,
        "reference_state_trajectory": jnp.tile(QR_REF_STATE, (horizon + 1, 1)),
        "reference_control_trajectory": jnp.tile(QR_REF_CONTROL, (horizon + 1, 1)),
        "penalize_control_reference": True,                          # track hover thrust
        "rescale_optimization_variables": False,
        "constrain_initial_control": False,
        "initial_control": QR_REF_CONTROL,
        "state_rescaling_min": -jnp.ones((QR_NX,)),
        "state_rescaling_max": jnp.ones((QR_NX,)),
        "control_rescaling_min": -jnp.ones((QR_NU,)),
        "control_rescaling_max": jnp.ones((QR_NU,)),
        "weights_penalization_reference_state_trajectory": QR_Q,
        "weights_penalization_final_state": jnp.zeros((QR_NX,)),
        "weights_penalization_control_squared": QR_R,
        "weights_penalization_control_rate": jnp.zeros((QR_NU,)),
        "state_min_bounds": -jnp.ones((QR_NX,)) * 1.0e7,
        "state_max_bounds": jnp.ones((QR_NX,)) * 1.0e7,
        "control_min_bounds": -jnp.ones((QR_NU,)) * umax,
        "control_max_bounds": jnp.ones((QR_NU,)) * umax,
        "dynamics_state_dot_params": {k: jnp.asarray(v) for k, v in QR_DYN.items()},
    }
    return dynamics, params


def generate_quadrotor_x0(batch_size: int, seed: int, scale: float = 1.0) -> np.ndarray:
    """Perturbations around hover (identity quaternion). The quaternion (indices 6:10) is
    renormalized so each sample is a valid unit quaternion."""
    rng = np.random.default_rng(seed)
    x = np.tile(np.asarray(QR_REF_STATE, dtype=float), (batch_size, 1))           # hover
    std = np.array([0.5, 0.5, 0.5,  0.2, 0.2, 0.2,
                    0.0, 0.1, 0.1, 0.1,  0.1, 0.1, 0.1]) * scale                  # q_0 not perturbed
    x = x + rng.normal(0.0, 1.0, size=(batch_size, QR_NX)) * std
    q = x[:, 6:10]
    x[:, 6:10] = q / np.linalg.norm(q, axis=1, keepdims=True)                     # renormalize quat
    return x


# system registry: name -> (problem builder, x0 generator, dims, eval-cost reference)
SYSTEMS = {
    "cartpole": dict(build=build_cartpole_problem, gen=generate_cartpole_x0,
                     nx=NX, nu=NU, ref_state=jnp.zeros((NX,)), ref_control=jnp.zeros((NU,))),
    "quadrotor": dict(build=build_quadrotor_problem, gen=generate_quadrotor_x0,
                      nx=QR_NX, nu=QR_NU, ref_state=QR_REF_STATE, ref_control=QR_REF_CONTROL),
}


def solver_params(nlp_tol: float, admm_eps: float, sqp_iter: int, admm_max_iter: int) -> Dict[str, Any]:
    sp = load_solver_params("turbompc.yaml")
    sp["num_sqp_iteration_max"] = sqp_iter
    sp["tol_convergence"] = nlp_tol             # NLP KKT (first-order) residual tolerance (swept)
    sp["warm_start_backward"] = False
    sp["linesearch"] = True            # nonlinear: keep line search for robust SQP
    sp["admm"]["max_iter"] = admm_max_iter
    # Check convergence every 25 iters (not every iter): still converges to eps (stops within
    # 25 iters of it), but avoids a host<->device sync per ADMM iter -> ~3x faster on cuDSS.
    sp["admm"]["check_termination_every"] = 25
    sp["admm"]["eps_abs"] = admm_eps            # inner-QP ADMM kept TIGHT to isolate NLP-KKT effect
    sp["admm"]["eps_rel"] = admm_eps
    return sp


def cosine(a, b, eps=1e-12):
    a, b = a.ravel(), b.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb + eps))


def rel_l2(a, b, eps=1e-12):
    a, b = a.ravel(), b.ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + eps))


def fd_per_state(rollout_batch, x0, weights, keys, eps):
    """PER-SAMPLE central-difference gradient that KEEPS the batch dim (unlike the shipped
    gradient_finite_diff, which does .sum() over the batch). Returns {key: (B, size)} -- each
    sample's own FD gradient. Cost = 2*Sum(weight sizes) batch rollout evals (same as the summed
    version, just un-reduced)."""
    out = {}
    for k in keys:
        base = np.asarray(weights[k])
        flat = base.ravel()
        cols = []
        for j in range(flat.size):
            fp = flat.copy(); fp[j] += eps
            fm = flat.copy(); fm[j] -= eps
            cp = np.asarray(rollout_batch(x0, {**weights, k: jnp.asarray(fp.reshape(base.shape))})[0])
            cm = np.asarray(rollout_batch(x0, {**weights, k: jnp.asarray(fm.reshape(base.shape))})[0])
            cols.append((cp - cm) / (2.0 * eps))
        out[k] = np.stack(cols, axis=1)  # (B, size)
    return out


def make_rollouts(solver, dynamics, problem_params, init_solution, K, reward_fn):
    """Return (rollout_single, rollout_batch). rollout_single(x, w) -> (cost, admm_iters) for one
    initial state (its own K-step closed-loop episode); rollout_batch is its vmap over the batch.
    The single fn is what we vmap(grad(.)) over to get PER-SAMPLE gradients."""
    cfg = ProblemConfig(
        dynamics=dynamics, problem_class=OptimalControlProblem,
        problem_params=problem_params, solver_params=None, weight_keys=WEIGHT_KEYS,
        reward_fn=reward_fn, update_per_seed=lambda s, b, p: ({}, None),
    )
    rollout_single = build_rollout_fn(
        config=cfg, solver=solver, problem_params=problem_params,
        init_solution=init_solution, warm_start=False, num_sim_steps=K)
    rollout_batch = jit(vmap(rollout_single, in_axes=(0, None)))
    return rollout_single, rollout_batch


def fd_per_state_converged(rollout_batch, x0, weights, keys, eps_seq, plateau_tol=1e-2):
    """CONVERGED per-sample FD ground truth (CLAUDE.md protocol). Computes the per-sample
    central-difference gradient at each eps in the DECREASING `eps_seq`; for each sample, takes the
    value at the LARGEST eps that has stabilized -- the first eps whose gradient agrees with the
    next-smaller one (rel_l2 < plateau_tol) = the converged plateau. Samples that never plateau are
    FLAGGED: the gradient is ill-defined there (sitting on a discontinuity / no usable eps above the
    solver noise floor). Returns (g (B, wdim), flagged (B,) bool)."""
    grads = []  # over eps: (B, wdim)
    for eps in eps_seq:
        g = fd_per_state(rollout_batch, x0, weights, keys, eps)
        grads.append(np.concatenate([g[k] for k in keys], axis=1))
    B = grads[0].shape[0]
    out = np.array(grads[-1])          # fallback: smallest-eps value
    flagged = np.ones(B, dtype=bool)
    for i in range(B):
        for t in range(len(eps_seq) - 1):
            a, b = grads[t][i], grads[t + 1][i]
            if np.linalg.norm(a - b) <= plateau_tol * (np.linalg.norm(b) + 1e-30):
                out[i] = a; flagged[i] = False; break
    return out, flagged


def run(args):
    dev = jax.devices()[0]
    print(f"JAX device: {dev} ({dev.platform})")
    fwd = parse_forward_backend(args.fwd)
    bwd = parse_backward_backend(args.bwd)
    # Auto-fall-back to pure-JAX if a cuDSS FFI backend was requested but its .so is unavailable.
    if not _CUDSS_FFI_AVAILABLE:
        if "CUDSS" in fwd.name:
            print(f"[benchmark] forward backend {fwd.name} needs cuDSS FFI (unavailable); "
                  "falling back to admm_jax_loop_pcg.")
            fwd = parse_forward_backend("admm_jax_loop_pcg")
        if "CUDSS" in bwd.name:
            print(f"[benchmark] backward backend {bwd.name} needs cuDSS FFI (unavailable); "
                  "falling back to direct_jax_dense.")
            bwd = parse_backward_backend("direct_jax_dense")
    sysspec = SYSTEMS[args.system]
    build, gen = sysspec["build"], sysspec["gen"]
    reward = make_reward(sysspec["ref_state"], sysspec["ref_control"])
    Ks = [args.calibrate_K] if args.calibrate else args.coupling
    n_seeds = 2 if args.calibrate else args.n_seeds
    nlp_tols = [args.nlp_tol[0]] if args.calibrate else args.nlp_tol
    eps_seq = args.gt_eps_seq

    print(f"system={args.system}  N={args.horizon} dt={args.dt}  umax={args.umax:.0e}  "
          f"NLP_KKT_tol(sweep)={nlp_tols}  coupling={Ks}  n_seeds={n_seeds}")
    print(f"  GT = CONVERGED per-sample FD at gt_tol={args.gt_tol:.0e} "
          f"(eps_seq={eps_seq}, plateau_tol={args.plateau_tol}); swept AD vs fixed GT")
    print(f"  fwd={args.fwd} bwd={args.bwd}  admm_eps={args.admm_eps:.0e} "
          f"sqp_ceiling={args.sqp_iter} admm={args.admm_max_iter}\n")

    x0_seeds = [jnp.asarray(gen(args.batch, s, args.x0_scale)) for s in range(n_seeds)]
    x0_warm = x0_seeds[0]

    results: Dict[tuple, Dict[str, np.ndarray]] = {}
    for K in Ks:
        dynamics, pp = build(args.horizon, args.umax, args.dt)
        weights = {k: pp[k] for k in WEIGHT_KEYS}

        # ---- GROUND TRUTH: CONVERGED per-sample FD at gt_tol (eps-convergence; CLAUDE.md) ----
        # For each sample, the FD step is shrunk until the gradient plateaus; samples that never
        # plateau are FLAGGED (sitting on a discontinuity). Cached once per K. gt[seed]->(B,wdim).
        gt_solver = TurboMPCSolver(
            program=OptimalControlProblem(dynamics=dynamics, params=dict(pp)),
            params=solver_params(args.gt_tol, args.admm_eps, args.sqp_iter, args.admm_max_iter),
            forward_backend=fwd, backward_backend=bwd, use_full_hessian=args.use_full_hessian)
        gt_init = gt_solver.solve(gt_solver.initial_guess(pp), problem_params=pp, weights=weights)
        _, gt_batch = make_rollouts(gt_solver, dynamics, pp, gt_init, K, reward)
        jax.block_until_ready(gt_batch(x0_warm, weights))
        gt, gt_flag = [], []
        for x0 in x0_seeds:
            g, fl = fd_per_state_converged(gt_batch, x0, weights, WEIGHT_KEYS, eps_seq, args.plateau_tol)
            gt.append(g); gt_flag.append(fl)
        flagged_all = np.concatenate(gt_flag)
        print(f"K={K:>3}  GT converged-FD: {int(flagged_all.sum())}/{flagged_all.size} samples "
              f"FLAGGED (no eps plateau -> at a discontinuity)")

        # ---- SWEEP: PER-SAMPLE AD at each SET tol vs the converged-FD GT ----
        for nlp_tol in nlp_tols:
            solver = TurboMPCSolver(
                program=OptimalControlProblem(dynamics=dynamics, params=dict(pp)),
                params=solver_params(nlp_tol, args.admm_eps, args.sqp_iter, args.admm_max_iter),
                forward_backend=fwd, backward_backend=bwd, use_full_hessian=args.use_full_hessian)
            init_solution = solver.solve(solver.initial_guess(pp), problem_params=pp, weights=weights)
            rollout_single, rollout_batch = make_rollouts(solver, dynamics, pp, init_solution, K, reward)
            per_state_grad = jit(vmap(jax.grad(
                checkpoint(lambda w, x: rollout_single(x, w)[0]), argnums=0), in_axes=(None, 0)))
            kkt_fn = jit(vmap(lambda x: solver.solve(
                init_solution, {**pp, "initial_state": x}, weights).convergence_error))
            jax.block_until_ready(rollout_batch(x0_warm, weights))

            cos_all, rel_all, gnorm, kkt = [], [], [], []
            for si, x0 in enumerate(x0_seeds):
                ad = per_state_grad(weights, x0); jax.block_until_ready(ad)
                B = x0.shape[0]
                gAd = np.concatenate([np.asarray(ad[k]).reshape(B, -1) for k in WEIGHT_KEYS], axis=1)
                gGt = gt[si]
                for i in range(B):
                    cos_all.append(cosine(gAd[i], gGt[i]))
                    rel_all.append(rel_l2(gAd[i], gGt[i]))
                gnorm.append(np.linalg.norm(gGt, axis=1))
                kkt.append(np.asarray(kkt_fn(x0)))

            cos = np.array(cos_all); rel = np.array(rel_all)
            gn = np.concatenate(gnorm); kk = np.concatenate(kkt); fl = flagged_all
            results[(nlp_tol, K)] = {"cos_all": cos, "rel_all": rel, "gnorm": gn,
                                     "kkt": kk, "flagged": fl}
            good = ~fl
            cg = cos[good]
            med = float(np.median(cg)) if cg.size else float("nan")
            cmin = float(cg.min()) if cg.size else float("nan")
            nlow = int(np.sum(cg < 0.99))
            reached = float(np.mean(kk <= nlp_tol))           # forward hit the tolerance (no early exit)
            overshoot = float(np.mean(kk < nlp_tol * 0.1))    # exceeded it by >1 order (over-converged)
            print(f"K={K:>3}  tol(set)={nlp_tol:.0e}  cos(non-flagged) median={med:.4f} "
                  f"min={cmin:+.3f} (#<0.99: {nlow}/{cg.size}; flagged {int(fl.sum())})  |  "
                  f"achieved KKT med={np.median(kk):.1e} reached(<=tol)={reached:.2f} "
                  f"overshoot(<tol/10)={overshoot:.2f}")

    if args.save_results and not args.calibrate:
        bound = "nobound" if args.umax >= 1e6 else f"umax{args.umax:g}"
        # write to research/.../experiments/<system>/results/<run-tag>/  (the system's results folder)
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.system, "results",
                              f"{bound}_N{args.horizon}_dt{args.dt}_seeds{n_seeds}_convFDtolsweep")
        outdir = os.path.abspath(outdir)
        os.makedirs(outdir, exist_ok=True)
        for (nt, K), agg in results.items():
            np.savez(os.path.join(outdir, f"tol_{nt:.0e}_K_{K}.npz".replace("e-0", "en")), **agg)
        with open(os.path.join(outdir, "summary.csv"), "w") as f:
            f.write("nlp_tol,K,cos_median,cos_min,n_below_0.99,n_good,n_flagged,n_samples,"
                    "gnorm_median,achieved_kkt_median,frac_reached,frac_overshoot,rel_l2_median\n")
            for (nt, K) in sorted(results):
                a = results[(nt, K)]; cos = a["cos_all"]; fl = a["flagged"]; cg = cos[~fl]; kk = a["kkt"]
                f.write(f"{nt:.0e},{K},{np.median(cg):.6f},{cg.min():.6f},{int(np.sum(cg<0.99))},"
                        f"{cg.size},{int(fl.sum())},{cos.size},{np.median(a['gnorm']):.4e},"
                        f"{np.median(kk):.3e},{np.mean(kk<=nt):.3f},{np.mean(kk<nt*0.1):.3f},"
                        f"{np.median(a['rel_all']):.4e}\n")
        print(f"\nSaved -> {outdir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", type=str, default="cartpole", choices=list(SYSTEMS),
                   help="dynamical system; quadrotor is SET UP but only cartpole is run this session")
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--dt", type=float, default=0.04)
    p.add_argument("--umax", type=float, default=1.0e7,
                   help="control bound magnitude; 1e7 => NO active inequality constraint (smooth NLP)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--x0_scale", type=float, default=1.0,
                   help="multiplier on initial-perturbation std (larger => SQP needs more iters)")
    p.add_argument("--coupling", type=int, nargs="*", default=[1, 5, 20])
    p.add_argument("--n_seeds", type=int, default=2,
                   help="batches of --batch states; EACH state is one sample, so 2 seeds x 64 = "
                        "128 per-sample points per (K,tol). Few seeds needed (samples come from batch)")
    p.add_argument("--nlp_tol", type=float, nargs="*", default=[1e-1, 1e-3, 1e-5, 1e-9],
                   help="SET NLP-KKT tolerance(s) to sweep; the forward must REACH each (no early "
                        "exit) without overshooting much (achieved KKT ~ set tol)")
    p.add_argument("--gt_tol", type=float, default=1e-9,
                   help="forward tolerance for the FD GROUND TRUTH (always converged)")
    p.add_argument("--gt_eps_seq", type=float, nargs="*", default=[1e-5, 1e-6, 3e-7, 1e-7],
                   help="DECREASING FD-step sequence for the converged GT (eps-convergence protocol)")
    p.add_argument("--plateau_tol", type=float, default=1e-2,
                   help="rel-l2 tolerance for declaring the per-sample FD converged (plateau)")
    p.add_argument("--admm_eps", type=float, default=1e-9,
                   help="inner-QP ADMM eps_abs/eps_rel (kept tight to isolate the NLP-KKT effect)")
    p.add_argument("--sqp_iter", type=int, default=300,
                   help="SQP iteration CEILING (non-binding): high enough that SQP REACHES the set "
                        "tol_convergence (no early exit), not capped")
    p.add_argument("--admm_max_iter", type=int, default=1000)
    p.add_argument("--fwd", type=str, default="admm_fused_cudss",
                   help="forward backend (default cuDSS FFI; needs build/ffi/*.so from a cuDSS-0.7.1 "
                        "build of diffmpc2/. Auto-falls-back to admm_jax_loop_pcg if the .so is absent.)")
    p.add_argument("--bwd", type=str, default="direct_cudss_ffi",
                   help="backward backend (default cuDSS FFI; auto-falls-back to direct_jax_dense if "
                        "the cuDSS .so is absent.)")
    p.add_argument("--use_full_hessian", action="store_true", default=True)
    p.add_argument("--save_results", action="store_true")
    p.add_argument("--calibrate", action="store_true", help="single K, 2 seeds, verbose")
    p.add_argument("--calibrate_K", type=int, default=1)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
