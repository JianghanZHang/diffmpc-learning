"""Training harness for Diff-WMPC drone task.

V1 = plan_hard : open-loop-plan gradient accumulation (Algorithm 1)
V3 = bptt_hard : SHAC truncated BPTT over h-step window

Usage (from repo root, cuDSS env):
    export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    PYTHONPATH=external/diffmpc2 \\
        /home/jianghan/Workspace/diffmpc2/.venv-cudss/bin/python \\
        experiments/rl/drone_rl/train.py
"""
from __future__ import annotations

import os
import sys
import csv
import time

# ---- sys.path bootstrap: same-dir modules + src + turbompc ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "../../../"))
_SRC = os.path.join(_REPO_ROOT, "src")
_TURBOMPC = os.path.join(_REPO_ROOT, "external", "diffmpc2")
for _p in (_HERE, _SRC, _TURBOMPC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from env import drone_env  # default env (the 6-state linear drone)
from mpc_layer import make_hard_layer
from policy import init_policy, make_theta_to_weights
from optimizer import adam_init
from gradient_modes import (
    open_loop_plan_update,
    shac_window_update,
    make_initial_guess,
    prime_guess,
    shift_guess,
)

# Results directory alongside this file
_RESULTS_DIR = os.path.join(_HERE, "results", "data")   # CSVs live under results/data
os.makedirs(_RESULTS_DIR, exist_ok=True)
# Trained NN policies live in a SEPARATE folder so a results-folder cleanup never deletes them.
_POLICY_DIR = os.path.join(_HERE, "trained_policies")
os.makedirs(_POLICY_DIR, exist_ok=True)

# Cache for JIT'd closed_loop_eval scan functions
# Keyed by (id(solver), id(t2w), id(dyn), id(pp_base), n_steps)
_EVAL_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# Closed-loop evaluation (no-grad realized rollout, JIT'd via jax.lax.scan)
# --------------------------------------------------------------------------- #

def closed_loop_eval(
    solver,
    t2w,
    dyn,
    policy,
    pp_base,
    x0,
    n_steps: int = 25,
    *,
    env=drone_env,
) -> dict:
    """No-gradient realized rollout from x0 for n_steps steps (JIT'd via jax.lax.scan).

    At each step: compute weights from policy, solve OCP (warm-start with
    receding-horizon shift), apply first control, simulate forward.

    The scan is JIT-compiled and cached per (solver, t2w, dyn, pp_base, n_steps)
    key — subsequent calls with the same stable objects reuse the compiled XLA kernel
    and finish in seconds rather than calling the Python loop.

    Warm-start: an initial guess is built and primed to the full solve-output pytree
    structure (admm_state populated), then receding-horizon-shifted at each step using
    shift_guess — cutting iteration count from ~3-7 SQP to ~1-2.

    Returns
    -------
    dict with keys:
        cost              : cumulative task_loss over the rollout
        min_margin        : minimum obs_margin (<=0 means always outside obstacle)
        n_violations      : number of steps with obs_margin > 0 (inside obstacle)
        final_dist_to_goal: ||x_final[:2] - GOAL[:2]||
    """
    simulate_step = env.simulate_step
    task_loss = env.task_loss
    obs_margin = env.obs_margin
    goal_dist = env.goal_dist

    x = jax.lax.stop_gradient(jnp.array(x0, dtype=jnp.float64))

    # Warm-start: build initial guess and prime to full pytree structure
    guess = make_initial_guess(solver, pp_base, x)
    guess = prime_guess(solver, t2w, policy, pp_base, x, guess)

    # Build / retrieve jitted scan function
    cache_key = (id(solver), id(t2w), id(dyn), id(pp_base), int(n_steps), id(env))
    if cache_key not in _EVAL_CACHE:
        def _eval_fn(phi, x_init, guess_init):
            def step_fn(carry, _):
                xk, g = carry
                w = t2w(phi, xk)
                sol = solver.solve(g, {**pp_base, "initial_state": xk}, w)
                g_next = jax.lax.stop_gradient(shift_guess(sol))
                u0 = jax.lax.stop_gradient(sol.controls[0])
                x_next = jax.lax.stop_gradient(simulate_step(dyn, xk, u0))
                step_cost = task_loss(x_next[None], u0[None])
                margin = obs_margin(x_next)
                violation = (margin > 0.0).astype(jnp.float64)
                return (x_next, g_next), (step_cost, margin, violation)

            (x_final, _), (costs, margins, violations) = jax.lax.scan(
                step_fn, (x_init, guess_init), xs=None, length=n_steps
            )
            # closest_margin = MAX over the rollout (closest approach; ~0 ⇒ grazing the
            # boundary, the active-set regime). min(margin) is the FURTHEST point and is a
            # misleading "did it graze" indicator. n_grazing = steps within 0.1 of the bound.
            return (x_final, jnp.sum(costs), jnp.max(margins),
                    jnp.sum((margins > -0.1).astype(jnp.float64)), jnp.sum(violations))

        _EVAL_CACHE[cache_key] = jax.jit(_eval_fn)

    eval_fn = _EVAL_CACHE[cache_key]
    x_final, total_cost, closest_margin, n_grazing, n_viol = eval_fn(policy, x, guess)
    jax.block_until_ready((x_final, total_cost, closest_margin, n_grazing, n_viol))

    final_dist = float(goal_dist(x_final))

    return {
        "cost": float(total_cost),
        "closest_margin": float(closest_margin),   # MAX margin = closest approach (~0 ⇒ grazing)
        "n_grazing": int(n_grazing),               # steps within 0.1 of the boundary
        "n_violations": int(n_viol),
        "final_dist_to_goal": final_dist,
    }


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #

def train(
    variant: str,
    seed: int,
    n_updates: int,
    *,
    env=drone_env,
    n_batch: int = 10,
    h: int = 8,
    lr: float = 3e-3,
    n_total: int = 40,
    eval_every: int = 10,
    eval_steps: int = 25,
    reset_dist: float = 0.1,
    barrier_glob: str = "filter",
    barrier_forward: str = "fused",
    barrier_mode: str = "elastic",
    barrier_bwd: str = "default",
    time_budget: float = 1200.0,
) -> dict:
    """Train the Diff-WMPC policy for the obstacle-avoidance task.

    Parameters
    ----------
    variant   : "plan_hard" (V1 open-loop-plan) or "bptt_hard" (V3 SHAC BPTT)
    seed      : JAX PRNG seed
    n_updates : number of policy-gradient update steps
    env       : the environment MODULE (``drone_env`` or ``quadrotor_env``); supplies
                build_problem_params / simulate_step / task_loss / obs_margin / goal_dist /
                sample_x0 / START / GOAL / QK / RK / NX / NU.
    n_batch   : steps per V1 update (open-loop-plan batch size)
    h         : window length for V3 SHAC update
    lr        : Adam learning rate
    n_total   : episode length (steps before forced reset)
    eval_every: how often (in updates) to run closed_loop_eval
    eval_steps: closed-loop eval horizon (steps)
    reset_dist: goal-distance threshold that triggers an episode reset

    Returns
    -------
    Summary dict with timing, first/last train loss, first/last eval cost,
    and final obstacle clearance.
    """
    assert variant in ("plan_hard", "bptt_hard", "plan_barrier", "bptt_barrier"), (
        f"Unknown variant '{variant}'. Must be one of plan_hard (V1), bptt_hard (V3), "
        f"plan_barrier (V2), bptt_barrier (V4)."
    )
    is_barrier = variant.endswith("_barrier")

    # ---- env API ----
    build_problem_params = env.build_problem_params
    simulate_step = env.simulate_step
    task_loss = env.task_loss
    obs_margin = env.obs_margin
    goal_dist = env.goal_dist
    sample_x0 = env.sample_x0
    QK, RK = env.QK, env.RK
    obs_dim, out_dim = env.NX, env.NX + env.NU

    # ------------------------------------------------------------------ #
    # Build env / solver / policy (STABLE objects for JIT cache)
    # ------------------------------------------------------------------ #
    print(f"Building environment and solver for env={env.__name__!r} variant={variant!r} ...")
    dyn, pp = build_problem_params()
    layer = make_hard_layer(dyn, pp)
    solver = layer.solver
    blayer = None
    if is_barrier:
        import barrier_modes
        # V2/V4: canonical diffmpc2 LogBarrier layer (elastic kappa=1e-4/gamma=1e2,
        # outer-slack FTB+filter globalization), warm-started internally.
        _bwd_over = {}
        if barrier_bwd == "case1":
            _bwd_over = {"bwd_w_from_dual": True, "bwd_pure_smooth_gamma": None}
        elif barrier_bwd == "smooth":
            _bwd_over = {"bwd_w_from_dual": False, "bwd_pure_smooth_gamma": 1e8}
        blayer = barrier_modes.make_barrier_ws_layer(
            solver, cfg={**_bwd_over,
                         "globalization": barrier_glob, "forward": barrier_forward,
                         # pure barrier: no elastic slack — the Moreau-envelope-style
                         # relaxation biases the fixed point by xi=y/gamma (exploitable);
                         # pure keeps iterates strictly feasible, bias O(kappa).
                         "use_slack": barrier_mode == "elastic"})

    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)
    policy = init_policy(init_key, obs_dim=obs_dim, out_dim=out_dim)
    t2w = make_theta_to_weights(pp[QK], pp[RK])
    opt_state = adam_init(policy)

    # ------------------------------------------------------------------ #
    # Episode state
    # ------------------------------------------------------------------ #
    steps_per_update = n_batch if variant.startswith("plan_") else h

    rng, noise_key = jax.random.split(rng)
    x = jax.lax.stop_gradient(sample_x0(noise_key, 0.02))
    # Prime ONCE to the full solve-output structure and cache it as the warm-start reused
    # on every episode reset.  Resets always return to ~START, so this primed guess keeps
    # post-reset solves WARM (~few SQP iters) instead of a ~10 s cold solve.  Gradient-safe:
    # the guess is stop_gradient'd by solver.solve (warm-start affects only iteration count).
    guess = make_initial_guess(solver, pp, x)
    guess = jax.lax.stop_gradient(prime_guess(solver, t2w, policy, pp, x, guess))
    guess_start = guess
    barrier_guess_start = None
    if is_barrier:
        # Prime the barrier layer's warm-start cell with one concrete solve at START
        # and cache the shifted solution for episode resets (mirrors guess_start).
        import barrier_modes  # noqa: F811
        blayer.solve({**pp, "initial_state": x}, t2w(policy, x))
        barrier_guess_start = blayer.snapshot()
    ep_step = 0

    # ------------------------------------------------------------------ #
    # CSV / bookkeeping
    # ------------------------------------------------------------------ #
    # basename only: after the env/ package reorg __name__ is "env.quadrotor_env"
    env_tag = env.__name__.split(".")[-1].replace("_env", "")
    # pure-barrier runs get their own files; elastic keeps the historical names
    file_variant = variant + ("_pure" if (is_barrier and barrier_mode == "pure") else "")
    csv_path = os.path.join(_RESULTS_DIR, f"train_{env_tag}_{file_variant}_seed{seed}.csv")
    csv_fields = [
        "update", "train_loss_mean", "grad_norm",
        "min_obs_margin", "max_obs_margin", "plan_margin_max", "wall_clock_s", "ep_step",
        "eval_cost", "eval_closest_margin", "eval_n_grazing", "eval_n_violations",
        # barrier arms: per-update forward-solve health (silent-non-convergence probe)
        "fwd_conv_max", "fwd_iters_max", "fwd_n_nonconv",
    ]
    csv_rows: list[dict] = []

    first_train_loss = None
    last_train_loss = None
    first_eval_cost = None
    last_eval_cost = None
    update_times: list[float] = []

    total_t0 = time.time()
    print(f"=== train(variant={variant!r}, seed={seed}, n_updates={n_updates}) ===\n")

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    for update_idx in range(n_updates):
        t0 = time.time()

        # ---- gradient update ----
        if variant == "plan_hard":
            policy, opt_state, x, guess, logs = open_loop_plan_update(
                solver, t2w, dyn, policy, opt_state, x, pp, guess,
                n_batch=n_batch, lr=lr,
                simulate_step=simulate_step, task_loss=task_loss, obs_margin=obs_margin,
            )
            jax.block_until_ready(logs["loss"])
            train_loss = float(jnp.mean(logs["loss"]))
            grad_norm = float(logs["accum_grad_norm"])
            plan_margin = float(jnp.max(logs["plan_obs_margin_max"]))
            # Realized obstacle margin over this update's steps:
            #   min = furthest point; max = CLOSEST approach (~0 ⇒ grazing the boundary).
            min_obs_margin = float(jnp.min(logs["obs_margin"]))
            max_obs_margin = float(jnp.max(logs["obs_margin"]))
        elif variant == "plan_barrier":  # V2
            import barrier_modes
            policy, opt_state, x, logs = barrier_modes.open_loop_plan_update_barrier(
                blayer, t2w, dyn, policy, opt_state, x, pp,
                n_batch=n_batch, lr=lr,
                simulate_step=simulate_step, task_loss=task_loss, obs_margin=obs_margin,
            )
            train_loss = float(jnp.mean(logs["loss"]))
            grad_norm = float(logs["accum_grad_norm"])
            plan_margin = float(jnp.max(logs["plan_obs_margin_max"]))
            min_obs_margin = float(jnp.min(logs["obs_margin"]))
            max_obs_margin = float(jnp.max(logs["obs_margin"]))
        elif variant == "bptt_barrier":  # V4
            import barrier_modes
            policy, opt_state, x, logs = barrier_modes.shac_window_update_barrier(
                blayer, t2w, dyn, policy, opt_state, x, pp,
                h=h, lr=lr,
                simulate_step=simulate_step, task_loss=task_loss, obs_margin=obs_margin,
            )
            train_loss = float(logs["loss"])
            grad_norm = float(logs["grad_norm"])
            plan_margin = float(jnp.max(logs["plan_obs_margin_max"]))
            min_obs_margin = float(jnp.min(logs["obs_margin"]))
            max_obs_margin = float(jnp.max(logs["obs_margin"]))
        else:  # bptt_hard
            policy, opt_state, x, guess, logs = shac_window_update(
                solver, t2w, dyn, policy, opt_state, x, pp, guess,
                h=h, lr=lr,
                simulate_step=simulate_step, task_loss=task_loss, obs_margin=obs_margin,
            )
            jax.block_until_ready(logs["loss"])
            train_loss = float(logs["loss"])
            grad_norm = float(logs["grad_norm"])
            plan_margin = float(jnp.max(logs["plan_obs_margin_max"]))
            # Realized obstacle margin over this update's steps:
            #   min = furthest point; max = CLOSEST approach (~0 ⇒ grazing the boundary).
            min_obs_margin = float(jnp.min(logs["obs_margin"]))
            max_obs_margin = float(jnp.max(logs["obs_margin"]))

        # ---- forward-solve health (barrier arms): SILENT non-convergence check.
        # The jitted fused loop exits at conv < sqp_tol OR the iteration cap — the
        # second branch returns an unconverged point with no error. Correlating
        # these per-update maxima with grad spikes tests the non-convergence
        # hypothesis for V4p's ~1e2..2e4 spikes (2026-07-07).
        fwd_conv_max, fwd_iters_max, fwd_n_nonconv = "", "", ""
        if is_barrier:
            b_iters, b_convs = blayer.pop_stats()
            if b_convs:
                fwd_conv_max = max(b_convs)
                fwd_iters_max = max(b_iters)
                fwd_n_nonconv = sum(1 for c in b_convs if c > blayer.cfg["sqp_tol"])
                print(f"    fwd iters/solve: {b_iters}  conv_max={fwd_conv_max:.2e}")
        if is_barrier and grad_norm > 100.0:
            spike_path = os.path.join(
                _POLICY_DIR, f"spike_{env_tag}_{file_variant}_upd{update_idx+1}.npz")
            _extra = {"x": np.asarray(x), "x_before": np.asarray(x_before)}
            if b_guess_before is not None:
                _extra["guess_states"] = np.asarray(b_guess_before[0])
                _extra["guess_controls"] = np.asarray(b_guess_before[1])
            np.savez(spike_path, **_extra,
                     **{k: np.asarray(v) for k, v in policy.items()})
            print(f"  [SPIKE] upd {update_idx+1}: grad={grad_norm:.3e} "
                  f"fwd_conv_max={fwd_conv_max} fwd_iters_max={fwd_iters_max} "
                  f"n_nonconv={fwd_n_nonconv} -> {os.path.basename(spike_path)}")

        wall_t = time.time() - t0
        if update_idx > 0:  # exclude first (JIT compile) update from mean
            update_times.append(wall_t)

        if first_train_loss is None:
            first_train_loss = train_loss
        last_train_loss = train_loss

        # ---- episode reset ----
        ep_step += steps_per_update
        dist_to_goal = float(goal_dist(x))
        if dist_to_goal < reset_dist or ep_step >= n_total:
            rng, noise_key = jax.random.split(rng)
            x = jax.lax.stop_gradient(sample_x0(noise_key, 0.02))
            guess = guess_start  # reuse the START-primed warm-start (avoid the cold re-prime)
            if is_barrier:
                blayer.reset(barrier_guess_start)
            ep_step = 0

        # ---- periodic evaluation ----
        eval_cost = eval_closest_margin = eval_n_grazing = eval_n_violations = None
        if (update_idx + 1) % eval_every == 0:
            if is_barrier:
                import barrier_modes
                ev = barrier_modes.closed_loop_eval_barrier(
                    blayer, t2w, dyn, policy, pp, env.START, n_steps=eval_steps, env=env)
            else:
                ev = closed_loop_eval(solver, t2w, dyn, policy, pp, env.START,
                                      n_steps=eval_steps, env=env)
            eval_cost = ev["cost"]
            eval_closest_margin = ev["closest_margin"]
            eval_n_grazing = ev["n_grazing"]
            eval_n_violations = ev["n_violations"]
            if first_eval_cost is None:
                first_eval_cost = eval_cost
            last_eval_cost = eval_cost
            print(
                f"  [{update_idx+1:3d}/{n_updates}] "
                f"loss={train_loss:.4f}  grad={grad_norm:.3e}  "
                f"eval_cost={eval_cost:.4f}  closest={eval_closest_margin:+.4f}  "
                f"graze={eval_n_grazing}  viol={eval_n_violations}  t={wall_t:.2f}s"
            )
        else:
            print(
                f"  [{update_idx+1:3d}/{n_updates}] "
                f"loss={train_loss:.4f}  grad={grad_norm:.3e}  "
                f"closest_obs={max_obs_margin:+.4f}  plan_margin={plan_margin:.4f}  t={wall_t:.2f}s"
            )

        csv_rows.append({
            "update": update_idx + 1,
            "train_loss_mean": train_loss,
            "grad_norm": grad_norm,
            "min_obs_margin": min_obs_margin,
            "max_obs_margin": max_obs_margin,
            "plan_margin_max": plan_margin,
            "wall_clock_s": wall_t,
            "ep_step": ep_step,
            "eval_cost": eval_cost,
            "eval_closest_margin": eval_closest_margin,
            "eval_n_grazing": eval_n_grazing,
            "eval_n_violations": eval_n_violations,
            "fwd_conv_max": fwd_conv_max,
            "fwd_iters_max": fwd_iters_max,
            "fwd_n_nonconv": fwd_n_nonconv,
        })
        # Incremental flush: a crash/kill must not lose completed updates.
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(csv_rows)

        # ---- bail-out guard: stop early if budget exceeded ----
        elapsed = time.time() - total_t0
        if elapsed > time_budget and update_idx < n_updates - 1:
            print(
                f"\n  [TIMEOUT] {elapsed:.0f}s elapsed after {update_idx+1} updates; "
                f"stopping early (budget {time_budget:.0f}s)."
            )
            n_updates = update_idx + 1  # update for summary
            break

    # ------------------------------------------------------------------ #
    # Persist results
    # ------------------------------------------------------------------ #
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV saved: {csv_path}")

    npz_path = os.path.join(_POLICY_DIR, f"train_{env_tag}_{file_variant}_seed{seed}_theta.npz")
    np.savez(npz_path, **{k: np.array(v) for k, v in policy.items()})
    print(f"Policy saved: {npz_path}")

    # ------------------------------------------------------------------ #
    # Final closed-loop eval (always from START)
    # ------------------------------------------------------------------ #
    print("\nRunning final eval from START ...")
    final_ev = closed_loop_eval(solver, t2w, dyn, policy, pp, env.START, n_steps=eval_steps, env=env)
    last_eval_cost = final_ev["cost"]
    final_clearance = final_ev["closest_margin"]

    total_wall = time.time() - total_t0
    mean_upd = float(np.mean(update_times)) if update_times else float("nan")

    return {
        "total_wall_clock_s": total_wall,
        "mean_update_time_s": mean_upd,
        "n_updates": n_updates,
        "first_train_loss": first_train_loss,
        "last_train_loss": last_train_loss,
        "first_eval_cost": first_eval_cost,
        "last_eval_cost": last_eval_cost,
        "final_closest_margin": final_clearance,        # MAX margin = closest approach (~0 ⇒ grazing)
        "final_n_grazing": final_ev["n_grazing"],
        "final_n_violations": final_ev["n_violations"],
    }


# --------------------------------------------------------------------------- #
# Entry point — first hard-box training run
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Diff-WMPC hard-box training (drone or quadrotor).")
    ap.add_argument("--env", choices=["drone", "quadrotor"], default="drone")
    ap.add_argument("--variant",
                    choices=["plan_hard", "bptt_hard", "plan_barrier", "bptt_barrier"],
                    default="plan_hard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_updates", type=int, default=100)
    ap.add_argument("--n_batch", type=int, default=10)
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--barrier_glob", default="filter", choices=["filter", "merit", "none"])
    ap.add_argument("--barrier_forward", default="fused", choices=["fused", "eager"])
    ap.add_argument("--barrier_mode", default="elastic", choices=["elastic", "pure"])
    ap.add_argument("--barrier_bwd", default="default", choices=["default", "smooth", "case1"])
    ap.add_argument("--time_budget", type=float, default=1200.0)
    ap.add_argument("--n_total", type=int, default=40)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=25)
    args = ap.parse_args()

    if args.env == "quadrotor":
        from env import quadrotor_env as env_mod
    else:
        env_mod = drone_env

    result = train(
        args.variant, seed=args.seed, n_updates=args.n_updates, env=env_mod,
        n_batch=args.n_batch, h=args.h, lr=args.lr,
        n_total=args.n_total, eval_every=args.eval_every, eval_steps=args.eval_steps,
        barrier_glob=args.barrier_glob, barrier_forward=args.barrier_forward,
        barrier_mode=args.barrier_mode, barrier_bwd=args.barrier_bwd,
        time_budget=args.time_budget,
    )

    print("\n" + "=" * 64)
    print(f"HARD-BOX TRAINING RESULTS  (env={args.env}, {args.variant}, seed={args.seed})")
    print("=" * 64)
    print(f"  Total wall-clock:        {result['total_wall_clock_s']:.1f} s")
    print(f"  Mean per-update time:    {result['mean_update_time_s']:.2f} s  (excl. compile)")
    print(f"  N updates run:           {result['n_updates']}")
    print(f"  Train loss first→last:   {result['first_train_loss']:.4f} → {result['last_train_loss']:.4f}")
    first_eval = result['first_eval_cost']
    last_eval = result['last_eval_cost']
    if first_eval is not None:
        print(f"  Eval cost first→last:    {first_eval:.4f} → {last_eval:.4f}")
    print(f"  Final closest margin:    {result['final_closest_margin']:+.4f}  (~0 ⇒ grazing the boundary)")
    print(f"  Final n_grazing / n_viol:{result['final_n_grazing']} / {result['final_n_violations']}")
    learning = (result["last_train_loss"] or 0.0) < (result["first_train_loss"] or float("inf"))
    print(f"  Loss decreased?          {'YES — learning' if learning else 'NO — not learning'}")
