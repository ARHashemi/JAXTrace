"""
Correctness test for the particle-hash octree backend (Backend P)
against the brute backend.

Both backends compute the same sum

    rho(q) = Σ_i K(‖q − x_i‖, h_i) · w_i

over all particles. The octree backend just restricts the inner loop
to particles within the kernel's compact support — outside, K=0 by
construction. So the two answers should be bit-identical up to
float32 round-off; we tolerate 1e-5 relative.

Run on a host with JAX configured (workstation or LUMI), not on the
controller.

Usage:
  python tests/test_octree_vs_brute.py
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import jax
import jax.numpy as jnp

from jaxtrace.density.estimator import (
    DensityEstimator,
    EstimatorConfig,
    pad_particles,
    _bucket_round,
)


# Per-kernel tolerance. Compact, smooth kernels are within float32 eps.
# Epanechnikov vanishes only linearly at the cutoff — the octree path is
# *exact* there (no GEMM, just direct distance + kernel eval), so it's no
# worse than brute. 1e-5 is fine for all.
_KERNEL_TOL = {
    "gaussian":       1e-5,
    "wendland_c2":    1e-5,
    "wendland_c4":    1e-5,
    "cubic_spline":   1e-5,
    "quintic_spline": 1e-5,
    "epanechnikov":   1e-5,
}


def _make_estimator(kernel_name: str, engine: str, particle_bucket: int = 4096) -> DensityEstimator:
    cfg = EstimatorConfig(
        kernel=kernel_name,
        d=3,
        normalization="unnormalized",   # so we compare raw kernel sums
        engine=engine,
        particle_bucket=particle_bucket,
    )
    return DensityEstimator(cfg=cfg, query_points=None)


def _run_one(kernel_name: str, N: int, M_side: int, h_val: float, seed: int = 0):
    """Compare brute vs octree on N particles, M_side^3 grid queries.

    Particles are scattered in a unit cube; queries are a regular grid
    covering the same cube. ``h_val`` is the bandwidth (single value,
    in physical units).
    """
    rng = np.random.default_rng(seed)
    P_np = rng.uniform(0.0, 1.0, size=(N, 3)).astype(np.float32)

    xs = np.linspace(0.0, 1.0, M_side, dtype=np.float32)
    Qx, Qy, Qz = np.meshgrid(xs, xs, xs, indexing="ij")
    Q_np = np.stack([Qx.ravel(), Qy.ravel(), Qz.ravel()], axis=1)
    M = Q_np.shape[0]

    # Per-axis bandwidth (N, 3). For an isotropic comparison test we use the
    # same h_val on every axis; the new kernel pipeline still works with
    # anisotropic h, that just isn't what this test exercises.
    h_np = np.full((N, 3), h_val, dtype=np.float32)
    w_np = np.ones((N,), dtype=np.float32)

    # Pad particles to the bucket size — the estimator does this internally
    # but we need it to keep the brute and octree results comparable.
    P = jnp.asarray(P_np); Q = jnp.asarray(Q_np); h = jnp.asarray(h_np); w = jnp.asarray(w_np)

    est_brute  = _make_estimator(kernel_name, engine="brute")
    est_octree = _make_estimator(kernel_name, engine="octree")

    rho_brute  = np.asarray(est_brute.evaluate(P, h, w, query_points=Q))
    rho_octree = np.asarray(est_octree.evaluate(P, h, w, query_points=Q))

    abs_diff = np.abs(rho_brute - rho_octree)
    rel_diff = abs_diff / np.maximum(np.abs(rho_brute), 1e-12)

    print(f"  {kernel_name:14s}  N={N:6d}  M={M:6d}  h={h_val:.3f}  "
          f"max(rho)={float(rho_brute.max()):.4e}  "
          f"nz_voxels(brute)={int((rho_brute>0).sum())}/{M}  "
          f"max|brute-octree|={float(abs_diff.max()):.4e}  "
          f"max rel-err={float(rel_diff.max()):.4e}")

    tol = _KERNEL_TOL[kernel_name]
    assert float(rel_diff.max()) < tol, (
        f"{kernel_name}: relative error {rel_diff.max():.4e} exceeds {tol:.0e}"
    )


def main():
    print("Testing particle-hash octree vs brute backend...")
    print()
    for kernel in ("wendland_c2", "wendland_c4", "cubic_spline",
                   "gaussian", "epanechnikov", "quintic_spline"):
        # Mixed (N, grid-side, h) — at small h the octree skips most pairs;
        # at large h almost every particle is in support of every query
        # and we exercise the fallback-to-brute path.
        for N, M_side, h_val in [
            ( 1024, 16, 0.10),    # large h relative to bbox; near-full neighbourhood
            ( 4096, 16, 0.05),    # moderate h
            (10000, 16, 0.03),    # small h; octree should really save work
            (10000, 32, 0.10),    # bigger grid, moderate h
        ]:
            _run_one(kernel, N=N, M_side=M_side, h_val=h_val)
    print()
    print("[OK] all kernel/scale combinations within tolerance")


if __name__ == "__main__":
    main()
