"""Uniform MPC layer interface for Diff-WMPC training.

Provides two interchangeable backends with the uniform signature::

    layer(problem_params, weights) -> (states, controls)

Both are differentiable w.r.t. ``weights`` AND ``problem_params["initial_state"]``:

* **HARD** (``make_hard_layer``): TurboMPC ADMM_FUSED_CUDSS forward + DIRECT_CUDSS_FFI
  backward.  Uses TurboMPC's ``get_differentiable_solve_function`` custom_vjp, verified
  correct in Part A (``test_inequality_hessian.py``, ``test_turbompc_x0_sensitivity.py``).

* **BARRIER** (``make_barrier_layer``): central-path SQP (``sqp_central_path``) forward +
  relaxed-NLP-KKT adjoint backward via ``make_central_path_diff``, verified in Part A
  (``test_central_path_x0_sensitivity.py`` gates 2-4).

Neither layer is ``jax.jit``-able as a unit (both run eager Python SQP loops); use
``jax.grad`` directly.

Export
------
``make_hard_layer(dynamics, pp_template, sp=None) -> layer``
``make_barrier_layer(dynamics, pp_template, *, cfg=None) -> layer``
``DEFAULT_BARRIER_CFG`` — the configuration used by ``make_barrier_layer`` by default.

Each returned ``layer`` also carries:
* ``layer.solver``   — the underlying ``TurboMPCSolver`` (for convergence checks).
For ``make_barrier_layer`` additionally:
* ``layer.cfg``      — the merged configuration dict.
* ``layer.solve_fn`` — the ``jax.custom_vjp`` function (for direct API comparison).
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------- #
# Path bootstrap: ensure src/ (diffmpc_learning) and external/turbompc are on
# sys.path regardless of whether tests/conftest.py is active (e.g. when this
# module is imported from a test run with PYTHONPATH=external/turbompc only).
# ---------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
# 3 levels up from rl/drone_rl/: rl -> experiments -> repo root
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "../../../"))
_SRC = os.path.join(_REPO_ROOT, "src")
_TURBOMPC = os.path.join(_REPO_ROOT, "external", "turbompc")
for _p in (_SRC, _TURBOMPC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
jax.config.update("jax_enable_x64", True)

from turbompc.problems.obstacle_avoidance import OptimalControlProblemObstacle
from turbompc.solvers.turbompc_solver import (
    TurboMPCSolver,
    ForwardBackend,
    BackwardBackend,
)
from turbompc.utils.load_params import load_solver_params
from diffmpc_learning.solvers.backward import make_central_path_diff

# ---------------------------------------------------------------------------- #
# Default barrier configuration
# Matches the verified _CFG in tests/python/solvers/test_central_path_x0_sensitivity.py
# ---------------------------------------------------------------------------- #
DEFAULT_BARRIER_CFG: dict = dict(
    slack_weight=1e4,
    target_kappa=1e-5,
    conv_slack_weight=1e4,
    kappa_anneal=True,
    kappa_anneal_start=1e-2,
    kappa_anneal_factor=0.3,
    linesearch=False,
    max_sqp_iter=60,
    tol=1e-6,
    jit_inner=True,
)


# ---------------------------------------------------------------------------- #
# Internal solver factories
# ---------------------------------------------------------------------------- #

def _build_hard_solver(dynamics, pp_template, sp=None):
    """TurboMPCSolver with ADMM_FUSED_CUDSS / DIRECT_CUDSS_FFI."""
    if sp is None:
        sp = dict(load_solver_params("turbompc.yaml"))
    else:
        sp = dict(sp)
    sp["num_sqp_iteration_max"] = 50          # cap; SQP early-exits at the NLP-KKT tol
    sp["tol_convergence"] = 1e-3              # NLP-KKT tolerance (user-directed)
    # QP-KKT tolerance (inner ADMM): yaml ships 1e-9 which NEVER early-terminates (runs full
    # max_iter every SQP iter) -> the main avoidable cost. 1e-6 lets the inner ADMM converge & stop.
    sp["admm"] = dict(sp["admm"])
    sp["admm"]["eps_abs"] = 1e-6
    sp["admm"]["eps_rel"] = 1e-6
    ocp = OptimalControlProblemObstacle(dynamics=dynamics, params=pp_template)
    return TurboMPCSolver(
        program=ocp,
        params=sp,
        forward_backend=ForwardBackend.ADMM_FUSED_CUDSS,
        backward_backend=BackwardBackend.DIRECT_CUDSS_FFI,
        use_full_hessian=True,
    )


def _build_barrier_solver(dynamics, pp_template, sp=None):
    """TurboMPCSolver with ADMM_JAX_LOOP_CUDSS_FFI / DIRECT_CUDSS_FFI.

    The forward uses the JAX-loop ADMM backend (required by sqp_central_path's inner solve
    via CUDSS_FFI Schur; FUSED cannot be used with the SQP loop driven externally).
    """
    if sp is None:
        sp = dict(load_solver_params("turbompc.yaml"))
    else:
        sp = dict(sp)
    sp["num_sqp_iteration_max"] = 60
    ocp = OptimalControlProblemObstacle(dynamics=dynamics, params=pp_template)
    return TurboMPCSolver(
        program=ocp,
        params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
        backward_backend=BackwardBackend.DIRECT_CUDSS_FFI,
        use_full_hessian=True,
    )


# ---------------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------------- #

def make_hard_layer(dynamics, pp_template, sp=None):
    """Build the HARD (TurboMPC ADMM_FUSED_CUDSS) MPC layer.

    The solver is built once and captured by the closure.  The ``initial_guess`` is
    recomputed from ``pp`` on every call (stop_gradient'd by TurboMPC's custom_vjp, so
    it does NOT block gradients through ``pp["initial_state"]`` or ``weights``).

    Args:
        dynamics:    DroneDynamics (or compatible) object.
        pp_template: problem params dict — used to build the OCP (structural template).
                     The actual ``pp`` is passed per call (can differ in ``initial_state``).
        sp:          solver params dict.  If ``None``, loads turbompc.yaml defaults with
                     ``num_sqp_iteration_max`` bumped to 50.

    Returns:
        layer: ``layer(pp, weights) -> (states, controls)`` differentiable w.r.t.
               ``weights`` and ``pp["initial_state"]``.
               Extra attribute: ``layer.solver`` (TurboMPCSolver).
    """
    solver = _build_hard_solver(dynamics, pp_template, sp)

    def layer(pp, weights):
        ig = solver.initial_guess(pp)
        sol = solver.solve(ig, pp, weights)
        return sol.states, sol.controls

    layer.solver = solver
    return layer


def make_barrier_layer(dynamics, pp_template, *, cfg=None):
    """Build the LOG-BARRIER (central-path SQP) MPC layer.

    The barrier solve is EAGER (not ``jit``/``scan``-able as a unit): the forward is a
    Python-loop SQP (``sqp_central_path``) and the backward is the relaxed-NLP-KKT adjoint
    (``_relaxed_nlp_backward``), both captured inside ``make_central_path_diff``'s
    ``jax.custom_vjp``.  Use ``jax.grad``, not ``jax.jit(solve)``.

    Args:
        dynamics:    DroneDynamics (or compatible) object.
        pp_template: problem params dict (structural template for OCP construction).
        cfg:         barrier configuration dict.  Defaults to ``DEFAULT_BARRIER_CFG``.

    Returns:
        layer: ``layer(pp, weights) -> (states, controls)`` differentiable w.r.t.
               ``weights`` and ``pp["initial_state"]``.
               Extra attributes:
               * ``layer.solver``   — TurboMPCSolver (for convergence checks).
               * ``layer.cfg``      — merged configuration dict.
               * ``layer.solve_fn`` — the ``jax.custom_vjp`` solve function.
    """
    if cfg is None:
        cfg = DEFAULT_BARRIER_CFG
    cfg = dict(cfg)

    solver = _build_barrier_solver(dynamics, pp_template)

    solve = make_central_path_diff(
        solver,
        slack_weight=cfg["slack_weight"],
        target_kappa=cfg["target_kappa"],
        include_ineq_hessian=True,
        linesearch=cfg["linesearch"],
        max_sqp_iter=cfg["max_sqp_iter"],
        tol=cfg["tol"],
        conv_slack_weight=cfg["conv_slack_weight"],
        kappa_anneal=cfg["kappa_anneal"],
        kappa_anneal_start=cfg["kappa_anneal_start"],
        kappa_anneal_factor=cfg["kappa_anneal_factor"],
        jit_inner=cfg["jit_inner"],
    )

    def layer(pp, weights):
        return solve(pp, weights)

    layer.solver = solver
    layer.cfg = cfg
    layer.solve_fn = solve
    return layer
