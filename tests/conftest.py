"""pytest configuration: add src/, diffmpc2/, and cartpole experiment dir to sys.path."""
import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)

# 1. src/: makes `diffmpc_learning` importable
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# 2. diffmpc2/: makes `turbompc` importable (vendored sibling, not a pip package)
_DIFFMPC2 = os.path.join(_REPO_ROOT, "diffmpc2")
if _DIFFMPC2 not in sys.path:
    sys.path.insert(0, _DIFFMPC2)

# 3. cartpole experiment dir: makes `benchmark_cartpole_coupling` importable
_CARTPOLE = os.path.join(
    _REPO_ROOT,
    "research", "gradient-quality-diffnmpc", "experiments", "cartpole",
)
if _CARTPOLE not in sys.path:
    sys.path.insert(0, _CARTPOLE)
