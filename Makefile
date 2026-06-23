# diffmpc_learning Makefile
# Usage:
#   make cuda    # Build CUDA FFI kernels (scaffold; no-op until .cu files added)
#   make test    # Run full test suite with cuDSS on PATH

SHELL := /bin/bash
CMAKE_SRC := src/diffmpc_learning/solvers/csrc
BUILD_DIR := build/ffi

.PHONY: cuda test

cuda:
	cmake -S $(CMAKE_SRC) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) -j

test:
	LD_LIBRARY_PATH="$(shell cat /tmp/cudss071_ldpath.txt):$$LD_LIBRARY_PATH" python -m pytest tests -v
