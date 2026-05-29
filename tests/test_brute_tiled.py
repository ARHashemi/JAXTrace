"""
Correctness test for the tiled / matmul-trick brute kernel against a
reference naive kernel. Run on a host with JAX configured (workstation
or LUMI), not on the controller.

Usage:
  python tests/test_brute_tiled.py
"""

from __future__ import annotations

import sys
import numpy as np

# Make the repo importable when run directly
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from jaxtrace.density import kernels
from jaxtrace.density.estimator import _make_brute_kernel


def _make_reference_kernel(kernel_name: str, d: int):
    """The pre-tiling reference implementation, frozen here for testing."""
    def per_chunk(Q, P, h, w):
        diff = Q[:, None, :] - P[None, :, :]
        r = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
        K = kernels.evaluate_kernel(kernel_name, r, h[None, :], d)
        return jnp.sum(K * w[None, :], axis=1)
    return jax.jit(per_chunk)


# Per-kernel relative-error tolerance. Compact-support kernels with a
# higher-order vanishing at the boundary (Wendland C2/C4, splines,
# Gaussian) tolerate the matmul-trick's ~1e-7 per-pair r²-perturbation
# almost invisibly because their kernel value near the cutoff is already
# many orders of magnitude smaller than the perturbation.
#
# Epanechnikov is the worst case: K = σ·max(0, 1 - q²) vanishes only
# linearly at q=1, so a tiny r²-perturbation moves a near-cutoff pair
# from value v to v ± Δ with v ~ Δ → relative error per *pair* can
# be O(1). The aggregated rel-err is then bounded by the fraction of
# pairs near the cutoff (small at large N, larger at small N). We give
# Epanechnikov a looser tolerance to reflect its known higher
# sensitivity to GEMM-based distance computation; the tolerance still
# falls well below visualisation-relevant resolution.
_KERNEL_TOL = {
    "gaussian":       1e-4,
    "wendland_c2":    1e-4,
    "wendland_c4":    1e-4,
    "cubic_spline":   1e-4,
    "quintic_spline": 1e-4,
    "epanechnikov":   2e-3,
}


def _run_one(kernel_name: str, N: int, M: int, particle_tile: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    # Particle and query positions in a unit cube
    P_np = rng.uniform(0.0, 1.0, size=(N, 3)).astype(np.float32)
    Q_np = rng.uniform(0.0, 1.0, size=(M, 3)).astype(np.float32)

    # Bandwidth: pick something where the kernel actually evaluates on a
    # decent fraction of pairs but not all of them, so both fast-path and
    # masked branches get tested.
    h_val = 0.15
    h_np = np.full((N,), h_val, dtype=np.float32)
    w_np = np.ones((N,), dtype=np.float32)

    # Pad particles up to particle_tile multiple (mirrors estimator.pad_particles)
    pad = (-N) % particle_tile
    if pad:
        P_np = np.concatenate([P_np, np.zeros((pad, 3), dtype=np.float32)], axis=0)
        h_np = np.concatenate([h_np, np.ones((pad,), dtype=np.float32)], axis=0)
        w_np = np.concatenate([w_np, np.zeros((pad,), dtype=np.float32)], axis=0)

    P = jnp.asarray(P_np); Q = jnp.asarray(Q_np); h = jnp.asarray(h_np); w = jnp.asarray(w_np)

    fn_ref = _make_reference_kernel(kernel_name, d=3)
    fn_new = _make_brute_kernel(kernel_name, d=3, particle_tile=particle_tile)

    rho_ref = np.asarray(fn_ref(Q, P, h, w))
    rho_new = np.asarray(fn_new(Q, P, h, w))

    abs_diff = np.abs(rho_ref - rho_new)
    rel_diff = abs_diff / np.maximum(np.abs(rho_ref), 1e-12)
    print(f"  {kernel_name:14s}  N={N:6d}  M={M:5d}  tile={particle_tile:5d}  "
          f"max(rho_ref)={float(rho_ref.max()):.4e}  "
          f"max|ref-new|={float(abs_diff.max()):.4e}  "
          f"max rel-err={float(rel_diff.max()):.4e}")

    tol = _KERNEL_TOL[kernel_name]
    assert float(rel_diff.max()) < tol, (
        f"{kernel_name}: relative error {rel_diff.max():.4e} exceeds {tol:.0e} tolerance"
    )


def main():
    print("Testing tiled brute kernel vs naive reference...")
    for kernel in ("wendland_c2", "wendland_c4", "cubic_spline",
                   "gaussian", "epanechnikov", "quintic_spline"):
        for N, M, T in [
            (   100,  50,  256),    # tiny: forces tile padding
            (  4096, 256,  4096),   # exactly 1 tile
            ( 12288, 256,  4096),   # exactly 3 tiles
            ( 50000, 512,  4096),   # many tiles, realistic chunk size
        ]:
            _run_one(kernel, N=N, M=M, particle_tile=T)
    print("\n[OK] all kernel/scale combinations within tolerance")


if __name__ == "__main__":
    main()
