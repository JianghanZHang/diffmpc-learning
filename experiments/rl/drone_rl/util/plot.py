"""Shared plotting library for drone_rl (+ the V1-vs-V3 training figures CLI).

ALL drone_rl plotting scripts import their style, palette, data locations, and
common drawing helpers from this module: INK/MUTED/GRID, ARM_COLORS, apply_style,
save_fig, end_label, draw_obstacle, draw_start_goal, DATA_DIR (results/data),
PLOT_DIR (results/plot), and the CSV loaders (_load_variant_seeds & friends).

Generates three figures:
  (a) eval_cost_vs_updates.png  — eval cost vs #updates, mean±std over seeds
  (b) grad_norm_vs_updates.png  — grad norm vs #updates, with markers where
                                   min_obs_margin > -0.1 (near active constraint)
  (c) closed_loop_traj.png      — x-y trajectories of the FINAL trained policy
                                   for each variant/seed, with obstacle circle

Usage (from repo root):
    PYTHONPATH=external/diffmpc2 \\
        /home/jianghan/Workspace/diffmpc2/.venv-cudss/bin/python \\
        experiments/rl/drone_rl/util/plot.py
"""
from __future__ import annotations

import os
import sys
import glob

# ---- sys.path bootstrap (this file lives in drone_rl/util/) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                       # drone_rl/
_REPO_ROOT = os.path.normpath(os.path.join(_PKG, "../../../"))
_SRC = os.path.join(_REPO_ROOT, "src")
_TURBOMPC = os.path.join(_REPO_ROOT, "external", "diffmpc2")
for _p in (_PKG, _SRC, _TURBOMPC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import csv

# Canonical result locations: CSV/JSON under results/data, figures under results/plot.
DATA_DIR = os.path.join(_PKG, "results", "data")
PLOT_DIR = os.path.join(_PKG, "results", "plot")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
_RESULTS_DIR = DATA_DIR   # CSV loading below reads from the data dir

# ---------------------------------------------------------------------------
# Shared figure style + helpers — ALL drone_rl plotting goes through these.
# ---------------------------------------------------------------------------
INK, MUTED, GRID = "#1a1a2e", "#5a5a6e", "#e3e3ea"
# Categorical palette, fixed slot order (dataviz reference palette, light mode).
ARM_COLORS = {
    "plan_hard": "#2a78d6",     # V1
    "plan_barrier": "#1baf7a",  # V2
    "bptt_hard": "#eda100",     # V3
    "bptt_barrier": "#008300",  # V4
}


def apply_style(ax):
    """House style: white ground, recessive grid, muted spines/ticks."""
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    return ax


def save_fig(fig, name, *, dpi=150):
    """Save a figure into results/plot/ and return the path."""
    out = os.path.join(PLOT_DIR, name)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {out}")
    return out


def end_label(ax, x, y, text, *, dy=0):
    """Direct label at the end of a line series."""
    ax.annotate(text, (x[-1], y[-1]), xytext=(4, dy), textcoords="offset points",
                fontsize=8, color=INK, fontweight="bold", va="center")


def draw_obstacle(ax, center, radius, *, label="obstacle"):
    """Filled obstacle disk with edge + centered label."""
    ax.add_patch(plt.Circle(center, radius, facecolor="#f3d9d9",
                            edgecolor="#b96a6a", linewidth=1.4, zorder=1))
    if label:
        ax.annotate(label, center, ha="center", va="center", fontsize=9,
                    color="#8a4a4a")


def draw_start_goal(ax, start_xy, goal_xy):
    """START square + GOAL star markers with labels."""
    ax.plot(*start_xy, marker="s", color=INK, ms=7, zorder=6)
    ax.annotate("START", start_xy, xytext=(8, -12), textcoords="offset points",
                fontsize=9, color=INK, fontweight="bold")
    ax.plot(*goal_xy, marker="*", color=INK, ms=13, zorder=6)
    ax.annotate("GOAL", goal_xy, xytext=(8, 4), textcoords="offset points",
                fontsize=9, color=INK, fontweight="bold")

# Env-dependent config (set by configure_env). Defaults to the drone for back-compat,
# but the OLD drone CSVs are named train_{variant}_seed*.csv (no env tag); new runs
# (drone or quadrotor) are train_{env_tag}_{variant}_seed*.csv.
ENV_NAME = "drone"
ENV_TAG = ""              # filename infix: "" → legacy drone files; "quadrotor"/"drone" → new files
OBS_C = np.array([-0.95, 0.025])
OBS_R = 0.3
START_XY = np.array([-1.9, 0.05])
GOAL_XY = np.array([0.0, 0.0])


def configure_env(env_name: str):
    """Set the geometry + filename tag from the chosen env module's constants."""
    global ENV_NAME, ENV_TAG, OBS_C, OBS_R, START_XY, GOAL_XY
    ENV_NAME = env_name
    if env_name == "quadrotor":
        from env import quadrotor_env as e
        ENV_TAG = "quadrotor"
    else:
        from env import drone_env as e
        ENV_TAG = "drone"
    OBS_C = np.asarray(e.OBS_C)
    OBS_R = float(e.OBS_R)
    START_XY = np.asarray(e.START[:2])
    GOAL_XY = np.asarray(e.GOAL[:2])


VARIANTS = ["plan_hard", "bptt_hard"]
VARIANT_LABELS = {"plan_hard": "V1 plan_hard", "bptt_hard": "V3 bptt_hard"}
VARIANT_COLORS = {"plan_hard": "#1f77b4", "bptt_hard": "#ff7f0e"}  # blue, orange

NEAR_ACTIVE_THRESH = -0.1  # min_obs_margin above this → near constraint boundary


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _parse_float(v):
    if v is None or v == "" or v == "None":
        return float("nan")
    return float(v)


def _fname_stem(variant: str) -> str:
    """Filename stem for a variant: ``train_{env_tag}_{variant}`` (or legacy ``train_{variant}``)."""
    return f"train_{ENV_TAG}_{variant}" if ENV_TAG else f"train_{variant}"


def _load_variant_seeds(variant: str) -> dict[int, list[dict]]:
    """Load all seed CSVs for a variant. Returns {seed: [rows]}."""
    stem = _fname_stem(variant)
    pattern = os.path.join(_RESULTS_DIR, f"{stem}_seed*.csv")
    paths = sorted(glob.glob(pattern))
    result = {}
    for p in paths:
        # Extract seed from filename: {stem}_seed0.csv
        base = os.path.basename(p)
        # seed is the last part before .csv
        seed_str = base.replace(f"{stem}_seed", "").replace(".csv", "")
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        result[seed] = _load_csv(p)
    return result


def _get_eval_series(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract (updates, eval_costs) from CSV rows (only rows with eval_cost set)."""
    updates, costs = [], []
    for r in rows:
        ec = _parse_float(r.get("eval_cost", ""))
        if not np.isnan(ec):
            updates.append(int(r["update"]))
            costs.append(ec)
    return np.array(updates), np.array(costs)


def _get_full_series(rows: list[dict], col: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract (updates, values) for a column present at every row."""
    updates, vals = [], []
    for r in rows:
        v = _parse_float(r.get(col, ""))
        updates.append(int(r["update"]))
        vals.append(v)
    return np.array(updates), np.array(vals)


# ---------------------------------------------------------------------------
# Figure (a): eval_cost vs #updates
# ---------------------------------------------------------------------------

def plot_eval_cost(ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    for variant in VARIANTS:
        seed_data = _load_variant_seeds(variant)
        if not seed_data:
            print(f"  [plot_eval_cost] No data for {variant}, skipping.")
            continue
        color = VARIANT_COLORS[variant]
        label = VARIANT_LABELS[variant]

        # Align eval series to common update grid
        all_updates_sets = []
        all_costs_by_seed = []
        for seed, rows in sorted(seed_data.items()):
            updates, costs = _get_eval_series(rows)
            if len(updates) == 0:
                continue
            all_updates_sets.append(set(updates.tolist()))
            all_costs_by_seed.append((updates, costs))

        if not all_costs_by_seed:
            continue

        # Common update indices (intersection across seeds)
        common_updates = sorted(
            set.intersection(*all_updates_sets) if len(all_updates_sets) > 1
            else all_updates_sets[0]
        )
        common_updates = np.array(common_updates)

        cost_matrix = []
        for updates, costs in all_costs_by_seed:
            # Map to common updates
            update_to_cost = dict(zip(updates.tolist(), costs.tolist()))
            row_costs = [update_to_cost.get(u, float("nan")) for u in common_updates]
            cost_matrix.append(row_costs)
        cost_matrix = np.array(cost_matrix, dtype=float)  # (n_seeds, n_updates)

        mean_costs = np.nanmean(cost_matrix, axis=0)
        std_costs = np.nanstd(cost_matrix, axis=0)

        ax.plot(common_updates, mean_costs, color=color, label=label, linewidth=2)
        ax.fill_between(
            common_updates,
            mean_costs - std_costs,
            mean_costs + std_costs,
            color=color, alpha=0.2,
        )

    ax.set_xlabel("Update #")
    ax.set_ylabel("Closed-loop eval cost (25 steps from START)")
    ax.set_title("Eval Cost vs Updates: V1 plan_hard vs V3 bptt_hard")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# ---------------------------------------------------------------------------
# Figure (b): grad norm vs #updates with near-active markers
# ---------------------------------------------------------------------------

def plot_grad_norm(ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    for variant in VARIANTS:
        seed_data = _load_variant_seeds(variant)
        if not seed_data:
            print(f"  [plot_grad_norm] No data for {variant}, skipping.")
            continue
        color = VARIANT_COLORS[variant]
        label = VARIANT_LABELS[variant]

        # Collect grad norm per seed, mean over seeds per update
        all_grad_by_seed = []
        all_margin_by_seed = []
        all_updates = None
        for seed, rows in sorted(seed_data.items()):
            updates, grads = _get_full_series(rows, "grad_norm")
            # CLOSEST approach (max margin); near 0 ⇒ grazing the active set. Fall back to
            # the legacy min_obs_margin column for older (pre-fix) CSVs.
            _, margins = _get_full_series(rows, "max_obs_margin")
            if np.all(np.isnan(margins)):
                _, margins = _get_full_series(rows, "min_obs_margin")
            if all_updates is None:
                all_updates = updates
            all_grad_by_seed.append(grads)
            all_margin_by_seed.append(margins)

        if not all_grad_by_seed or all_updates is None:
            continue

        min_len = min(len(g) for g in all_grad_by_seed)
        updates = all_updates[:min_len]
        grad_matrix = np.array([g[:min_len] for g in all_grad_by_seed], dtype=float)
        margin_matrix = np.array([m[:min_len] for m in all_margin_by_seed], dtype=float)

        mean_grad = np.nanmean(grad_matrix, axis=0)
        # Mark where ANY seed has min_obs_margin > NEAR_ACTIVE_THRESH
        near_active = np.any(margin_matrix > NEAR_ACTIVE_THRESH, axis=0)

        ax.semilogy(updates, mean_grad, color=color, label=label, linewidth=1.5)
        if np.any(near_active):
            ax.scatter(
                updates[near_active], mean_grad[near_active],
                color=color, marker="x", s=60, zorder=5,
                label=f"{label} (near-active, margin>-0.1)",
            )

    ax.set_xlabel("Update #")
    ax.set_ylabel("Grad norm (log scale)")
    ax.set_title("Grad Norm vs Updates\n(x = near-active-set, min_obs_margin > -0.1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    return fig


# ---------------------------------------------------------------------------
# Figure (c): closed-loop x-y trajectories from final policy
# ---------------------------------------------------------------------------

def _run_closed_loop_traj(variant: str, seed: int) -> np.ndarray | None:
    """Run closed-loop eval from START with the saved policy, return (n_steps+1, 2) xy."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    if ENV_NAME == "quadrotor":
        from env import quadrotor_env as e
    else:
        from env import drone_env as e
    build_problem_params, simulate_step, START, QK, RK = (
        e.build_problem_params, e.simulate_step, e.START, e.QK, e.RK)
    from mpc_layer import make_hard_layer
    from policy import make_theta_to_weights
    from gradient_modes import make_initial_guess, prime_guess, shift_guess

    npz_path = os.path.join(_PKG, "trained_policies", f"{_fname_stem(variant)}_seed{seed}_theta.npz")
    if not os.path.exists(npz_path):
        return None

    data = np.load(npz_path)
    policy = {k: jnp.array(data[k]) for k in data.files}

    dyn, pp = build_problem_params()
    layer = make_hard_layer(dyn, pp)
    solver = layer.solver
    t2w = make_theta_to_weights(pp[QK], pp[RK])

    n_steps = 25
    x = jax.lax.stop_gradient(jnp.array(START, dtype=jnp.float64))
    guess = make_initial_guess(solver, pp, x)
    guess = prime_guess(solver, t2w, policy, pp, x, guess)

    traj_xy = [np.array(x[:2])]
    for _ in range(n_steps):
        w = t2w(policy, x)
        sol = solver.solve(guess, {**pp, "initial_state": x}, w)
        guess = jax.lax.stop_gradient(shift_guess(sol))
        u0 = jax.lax.stop_gradient(sol.controls[0])
        x = jax.lax.stop_gradient(simulate_step(dyn, x, u0))
        traj_xy.append(np.array(x[:2]))

    return np.array(traj_xy)  # (n_steps+1, 2)


def plot_trajectories(ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    # Draw obstacle
    circle = plt.Circle(OBS_C, OBS_R, color="gray", alpha=0.4, label="Obstacle")
    ax.add_patch(circle)
    circle_edge = plt.Circle(OBS_C, OBS_R, color="gray", fill=False, linewidth=1.5)
    ax.add_patch(circle_edge)

    # Start / Goal markers
    ax.scatter(*START_XY, color="green", s=100, zorder=6, marker="o", label="START")
    ax.scatter(*GOAL_XY, color="red", s=100, zorder=6, marker="*", label="GOAL")

    any_traj = False
    for variant in VARIANTS:
        seed_data = _load_variant_seeds(variant)
        color = VARIANT_COLORS[variant]
        label = VARIANT_LABELS[variant]
        first = True
        for seed in sorted(seed_data.keys()):
            traj = _run_closed_loop_traj(variant, seed)
            if traj is None:
                print(f"  [plot_traj] No saved policy for {variant} seed {seed}, skipping.")
                continue
            any_traj = True
            ax.plot(
                traj[:, 0], traj[:, 1],
                color=color,
                linewidth=1.5 if not first else 2.0,
                alpha=0.7 if not first else 1.0,
                label=label if first else None,
            )
            first = False

    if not any_traj:
        ax.text(0.5, 0.5, "No saved policies found", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Closed-Loop Trajectories (final policy, 25 steps from START)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Plot Diff-WMPC V1-vs-V3 training results.")
    ap.add_argument("--env", choices=["drone", "quadrotor"], default="drone")
    args = ap.parse_args()
    configure_env(args.env)
    pre = f"{ENV_TAG}_" if ENV_TAG else ""
    print(f"Generating plots for env={ENV_NAME} (tag={ENV_TAG!r}) ...")

    fig_a = plot_eval_cost()
    path_a = os.path.join(PLOT_DIR, f"{pre}eval_cost_vs_updates.png")
    fig_a.tight_layout()
    fig_a.savefig(path_a, dpi=150)
    plt.close(fig_a)
    print(f"  Saved: {path_a}")

    fig_b = plot_grad_norm()
    path_b = os.path.join(PLOT_DIR, f"{pre}grad_norm_vs_updates.png")
    fig_b.tight_layout()
    fig_b.savefig(path_b, dpi=150)
    plt.close(fig_b)
    print(f"  Saved: {path_b}")

    fig_c = plot_trajectories()
    path_c = os.path.join(PLOT_DIR, f"{pre}closed_loop_traj.png")
    fig_c.tight_layout()
    fig_c.savefig(path_c, dpi=150)
    plt.close(fig_c)
    print(f"  Saved: {path_c}")

    print("\nDone. Figures written to:", PLOT_DIR)


if __name__ == "__main__":
    main()
