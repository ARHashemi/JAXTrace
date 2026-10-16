"""
Point-in-tet using precomputed 3×3 inverse transformation matrices.

This module provides a faster point-in-tet test by precomputing the inverse
transformation matrix for each tetrahedral element. The barycentric coordinates
are then computed via a single matrix-vector multiply (15 FLOPs) instead of
solving the full 3×3 system (145 FLOPs).

Expected speedup: 3-4× over current methods (6.6× computational, reduced by memory bandwidth)
Memory cost: 60 bytes per element (12 floats for M_inv + 3 floats for p0)
"""

import numpy as np
import jax
import jax.numpy as jnp
import jaxtrace.config as config


def precompute_inverse_matrices(
    connectivity: np.ndarray,
    node_positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute 3×3 inverse transformation matrices for all tetrahedral elements.

    For each tetrahedron with vertices p0, p1, p2, p3, we precompute:
        M = [p1-p0, p2-p0, p3-p0]  (3×3 matrix of edge vectors from p0)
        M_inv = inverse(M)

    Then, barycentric coordinates for query point pos are:
        local = pos - p0
        (λ1, λ2, λ3) = M_inv @ local
        λ0 = 1 - λ1 - λ2 - λ3

    Point is inside if all λi >= -tolerance.

    Parameters
    ----------
    connectivity : ndarray, shape (n_elements, 4), int32
        Tetrahedral element connectivity (4 node indices per element)

    node_positions : ndarray, shape (n_nodes, 3), float32
        Node coordinates (x, y, z)

    Returns
    -------
    M_inv_array : ndarray, shape (n_elements, 3, 3), float32
        Inverse transformation matrix for each element

    p0_array : ndarray, shape (n_elements, 3), float32
        First vertex (p0) for each element

    Notes
    -----
    - Degenerate elements (det(M) ≈ 0) are assigned zero inverse matrix
      and will always return "not inside" during point-in-tet tests
    - This precomputation is done once on CPU during mesh upload
    - GPU memory cost: 60 bytes per element = 210 MB for 3.5M elements
    """
    n_elements = connectivity.shape[0]

    # ⚠️ VECTORISED 2026-10-16. This was a Python loop over every element, which
    # cost 140 s at D2 scale (10.8M tets) -- the whole of stage 2. It does the
    # same det + inv that jaxtrace.damage.fields.tet_shape_gradients does with
    # numpy batching in 5.9 s, so the loop was the only reason it was slow.
    # Measured 15.7x faster and BIT-IDENTICAL, degenerate handling included.
    p = node_positions[connectivity]                       # (n_elem, 4, 3)
    M = np.stack([p[:, 1] - p[:, 0],
                  p[:, 2] - p[:, 0],
                  p[:, 3] - p[:, 0]], axis=-1)             # edge vectors as COLUMNS
    det = np.linalg.det(M)

    # Degenerate elements get a ZERO inverse, so point-in-tet always reports
    # "outside" for them -- same contract as the loop. Substituting the identity
    # before inverting keeps np.linalg.inv from raising on a singular batch.
    good = np.abs(det) >= 1e-15
    M_safe = np.where(good[:, None, None], M, np.eye(3)[None])
    M_inv_array = np.where(good[:, None, None], np.linalg.inv(M_safe), 0.0
                           ).astype(config.FLOAT_DTYPE_NP)
    p0_array = p[:, 0].astype(config.FLOAT_DTYPE_NP)

    n_degenerate = int((~good).sum())
    _absdet = np.abs(det[good])
    det_min = float(_absdet.min()) if _absdet.size else float('inf')
    det_max = float(_absdet.max()) if _absdet.size else float('-inf')

    # Report statistics
    print(f"\nPrecomputed inverse matrices:")
    print(f"  Elements: {n_elements:,}")
    print(f"  Degenerate: {n_degenerate} ({100*n_degenerate/n_elements:.4f}%)")
    if n_degenerate < n_elements:
        print(f"  Det range: [{det_min:.2e}, {det_max:.2e}]")
    print(f"  Memory: {M_inv_array.nbytes + p0_array.nbytes:,} bytes ({(M_inv_array.nbytes + p0_array.nbytes)/1024**2:.1f} MB)")

    return M_inv_array, p0_array


@jax.jit
def point_in_tet_inverse(
    pos: jax.Array,
    elem_id: jax.Array,
    M_inv_array: jax.Array,
    p0_array: jax.Array,
    tolerance: float = config.POINT_IN_TET_TOLERANCE
) -> jax.Array:
    """
    Point-in-tet test using precomputed inverse transformation matrix.

    Computes barycentric coordinates via matrix-vector multiply:
        local = pos - p0
        bary = M_inv @ local

    Then tests if all barycentric coordinates are non-negative (with tolerance).

    Parameters
    ----------
    pos : jax.Array, shape (3,)
        Query point coordinates

    elem_id : jax.Array, scalar int32
        Element index to test

    M_inv_array : jax.Array, shape (n_elements, 3, 3), float32
        Precomputed inverse matrices (on GPU)

    p0_array : jax.Array, shape (n_elements, 3), float32
        First vertex for each element (on GPU)

    tolerance : float, default=1e-6
        Numerical tolerance for containment test
        (allows points slightly outside due to floating-point errors)
        Note: 1e-6 is appropriate for float32 precision at mesh scale ~1.0

    Returns
    -------
    inside : jax.Array, scalar bool
        True if point is inside tetrahedron, False otherwise

    Performance
    -----------
    FLOPs breakdown:
    - Local vector: 3 subtractions = 3 FLOPs
    - Matrix-vector multiply: 3×3 multiply-adds = 9 muls + 6 adds = 15 FLOPs
    - Fourth coordinate: 3 adds + 1 sub = 4 FLOPs
    - Comparisons: 4 comparisons (not counted as FLOPs)
    Total: 22 FLOPs (vs 145 for baseline, 48 for Skala)

    Memory access:
    - 2 coalesced reads: M_inv[elem_id] (3×3=9 floats), p0[elem_id] (3 floats)
    - Total: 12 floats = 48 bytes
    vs current method:
    - 4 random reads: node_positions[connectivity[elem_id][i]] for i=0,1,2,3
    - Total: 12 floats = 48 bytes (but NON-coalesced)

    Expected speedup:
    - Computational: 145 / 22 = 6.6×
    - Memory-bound (coalesced vs random): ~2×
    - Combined: 3-4× realistic
    """
    # Load precomputed data (coalesced memory access on GPU)
    M_inv = M_inv_array[elem_id]  # (3, 3) - single coalesced read
    p0 = p0_array[elem_id]        # (3,) - single coalesced read

    # Transform to local coordinates
    local = pos - p0  # (3,) - 3 subtractions

    # Compute barycentric coordinates via matrix-vector multiply
    # bary = M_inv @ local
    # = [M_inv[0,:] @ local, M_inv[1,:] @ local, M_inv[2,:] @ local]
    bary = M_inv @ local  # (3,) - 9 muls + 6 adds = 15 FLOPs

    # Fourth barycentric coordinate
    b0 = 1.0 - jnp.sum(bary)  # 3 adds + 1 sub = 4 FLOPs

    # Containment test: all barycentric coordinates >= -tolerance
    # (Negative tolerance allows points slightly outside due to floating-point)
    inside = (bary[0] >= -tolerance) & \
             (bary[1] >= -tolerance) & \
             (bary[2] >= -tolerance) & \
             (b0 >= -tolerance)

    return inside


# Vectorized version for batch testing (useful for validation)
@jax.jit
def point_in_tet_inverse_batch(
    positions: jax.Array,
    elem_ids: jax.Array,
    M_inv_array: jax.Array,
    p0_array: jax.Array,
    tolerance: float = config.POINT_IN_TET_TOLERANCE
) -> jax.Array:
    """
    Vectorized point-in-tet test for batch of positions and elements.

    Parameters
    ----------
    positions : jax.Array, shape (n_queries, 3)
        Query point coordinates

    elem_ids : jax.Array, shape (n_queries,), int32
        Element indices to test (one per query)

    M_inv_array : jax.Array, shape (n_elements, 3, 3), float32
        Precomputed inverse matrices

    p0_array : jax.Array, shape (n_elements, 3), float32
        First vertex for each element

    tolerance : float, default=1e-6
        Numerical tolerance for containment test

    Returns
    -------
    inside : jax.Array, shape (n_queries,), bool
        True if point is inside corresponding tetrahedron
    """
    return jax.vmap(
        lambda pos, eid: point_in_tet_inverse(pos, eid, M_inv_array, p0_array, tolerance)
    )(positions, elem_ids)


# Integration with existing point-in-tet interface
def create_inverse_point_in_tet_fn(M_inv_array_gpu, p0_array_gpu, tolerance=config.POINT_IN_TET_TOLERANCE):
    """
    Create a point-in-tet function with precomputed data baked in.

    This returns a function compatible with existing point-in-tet interface:
        inside = fn(pos, elem_id, connectivity, node_positions)

    But internally uses the faster inverse matrix method.

    Parameters
    ----------
    M_inv_array_gpu : jax.Array, shape (n_elements, 3, 3)
        Precomputed inverse matrices (on GPU)

    p0_array_gpu : jax.Array, shape (n_elements, 3)
        First vertex for each element (on GPU)

    tolerance : float, default=1e-6
        Numerical tolerance for containment test

    Returns
    -------
    point_in_tet_fn : callable
        Function with signature: fn(pos, elem_id, connectivity, node_positions) -> bool
        (connectivity and node_positions are ignored, kept for compatibility)
    """
    @jax.jit
    def point_in_tet_fn(pos, elem_id, connectivity, node_positions):
        """Point-in-tet using precomputed inverse matrices."""
        return point_in_tet_inverse(pos, elem_id, M_inv_array_gpu, p0_array_gpu, tolerance)

    return point_in_tet_fn
