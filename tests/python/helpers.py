"""Test helpers — copied verbatim from diffmpc2/tests/helpers/problem_fixtures.py.

Only `cost_blocks_from_qr` is needed by the central-path test suite; the full
fixture file is reproduced here so the suite does NOT import diffmpc2's `tests`
package (that name would clash with this repo's `tests/`).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def cost_blocks_from_qr(
    Qmat: jnp.ndarray,
    Rmat: jnp.ndarray,
    Rd: jnp.ndarray,
    qvec: jnp.ndarray,
    rvec: jnp.ndarray,
):
    N = Qmat.shape[0] - 1
    nx = Qmat.shape[1]
    nu = Rmat.shape[1]
    n = nx + nu
    D = jnp.zeros((N + 1, n, n), dtype=Qmat.dtype)
    E = jnp.zeros((N, n, n), dtype=Qmat.dtype)
    for t in range(N + 1):
        D = D.at[t, :nx, :nx].set(Qmat[t])
        D = D.at[t, nx:, nx:].set(Rmat[t])
        if t > 0:
            D = D.at[t, nx:, nx:].add(Rd[t - 1])
        if t < N:
            D = D.at[t, nx:, nx:].add(Rd[t])
    for t in range(N):
        E = E.at[t, nx:, nx:].set(-Rd[t])
    q = jnp.concatenate([qvec, rvec], axis=-1)
    return D, E, q
