"""pytest configuration: add src/, external/diffmpc2/, benchmarking, and experiment dirs to sys.path."""
import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)

# 1. src/: makes `diffmpc_learning` importable
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# 2. external/diffmpc2/: makes `turbompc` importable (CANONICAL solver, branch LogBarrier-ADMM-QP —
#    see CLAUDE.md; superset of external/turbompc with the sign-corrected solver + inequality
#    Hessian + logbarrier QP backend. NOT the vendored diffmpc2/ at repo root, which has the
#    hard-box sign bug and lacks get_inequality_lagrangian_hessian)
_TURBOMPC = os.path.join(_REPO_ROOT, "external", "diffmpc2")
if _TURBOMPC not in sys.path:
    sys.path.insert(0, _TURBOMPC)

# 3. linear-system benchmarking: makes `build_turbompc_linear_problem` importable (obstacle fixture)
_BENCH = os.path.join(_TURBOMPC, "benchmarking", "linear-system")
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

# 4. cartpole experiment dir: makes `benchmark_cartpole_coupling` importable
_CARTPOLE = os.path.join(_REPO_ROOT, "experiments", "gradients", "cartpole")
if _CARTPOLE not in sys.path:
    sys.path.insert(0, _CARTPOLE)
