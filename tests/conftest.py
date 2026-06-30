"""pytest configuration: add src/, external/turbompc/, benchmarking, and experiment dirs to sys.path."""
import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)

# 1. src/: makes `diffmpc_learning` importable
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# 2. external/turbompc/: makes `turbompc` importable (CANONICAL solver, not diffmpc2 — see
#    CLAUDE.md; diffmpc2 has the hard-box sign bug and lacks get_inequality_lagrangian_hessian)
_TURBOMPC = os.path.join(_REPO_ROOT, "external", "turbompc")
if _TURBOMPC not in sys.path:
    sys.path.insert(0, _TURBOMPC)

# 3. linear-system benchmarking: makes `build_turbompc_linear_problem` importable (obstacle fixture)
_BENCH = os.path.join(_TURBOMPC, "benchmarking", "linear-system")
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

# 4. cartpole experiment dir: makes `benchmark_cartpole_coupling` importable
_CARTPOLE = os.path.join(
    _REPO_ROOT,
    "research", "gradient-quality-diffnmpc", "experiments", "cartpole",
)
if _CARTPOLE not in sys.path:
    sys.path.insert(0, _CARTPOLE)
