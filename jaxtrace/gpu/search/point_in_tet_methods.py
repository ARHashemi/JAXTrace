"""
Point-in-tetrahedron test implementations with different optimization strategies.

This module provides multiple implementations of the point-in-tetrahedron containment test:
1. Current/Reference: Barycentric coordinates via Cramer's rule (145 FLOPs)
2. Skala: GPU-optimized using cross products in projective space (48 FLOPs)
3. Axis-Aligned: Specialized for axis-aligned tetrahedra (12 FLOPs)

All methods are memory-safe (no precomputed arrays) and use on-the-fly detection.

References:
- Skala, V. (2014). "Intersection Computation in Projective Space Using Homogeneous Coordinates"
  WICT 2014, Appendix A (GPU implementation)
"""

import jax
import jax.numpy as jnp


# ============================================================================
# Method 1: Current (Reference) - Barycentric via Cramer's Rule
# ============================================================================

def point_in_tet_current(
    pos: jax.Array,
    elem_id: jnp.int32,
    connectivity: jax.Array,
    node_positions: jax.Array
) -> jnp.bool_:
    """
    Reference implementation using barycentric coordinates (Cramer's rule).

    FLOP count: ~145 FLOPs
    - 4× determinant expansion (3×3): ~30 FLOPs each = 120 FLOPs
    - Vector operations: ~25 FLOPs

    This is the current production implementation, kept for validation.

    Args:
        pos: (3,) float32 - query position
        elem_id: int32 - element ID to test
        connectivity: (n_elements, 4) int32
        node_positions: (n_nodes, 3) float32

    Returns:
        inside: bool - True if pos is in element
    """
    # Get node indices
    nodes = connectivity[elem_id]  # (4,)

    # Get node positions
    p0 = node_positions[nodes[0]]  # (3,)
    p1 = node_positions[nodes[1]]
    p2 = node_positions[nodes[2]]
    p3 = node_positions[nodes[3]]

    # Compute vectors from p0
    v1 = p1 - p0
    v2 = p2 - p0
    v3 = p3 - p0
    vp = pos - p0

    # Solve for barycentric coordinates using Cramer's rule
    # [v1 v2 v3] * [b1, b2, b3]^T = vp
    # b0 = 1 - b1 - b2 - b3

    # Compute 3x3 determinant: det([v1 v2 v3])
    det = (v1[0] * (v2[1] * v3[2] - v2[2] * v3[1]) -
           v1[1] * (v2[0] * v3[2] - v2[2] * v3[0]) +
           v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]))

    # Handle degenerate tetrahedron with RELATIVE threshold
    # For refined meshes with L~0.0001m: det~L³~1e-12
    det_abs = jnp.abs(det)
    edge_length_sq = jnp.sum(v1 * v1)  # Typical edge length squared
    expected_det = edge_length_sq ** 1.5  # det scales as L³
    # Use relative threshold: det < ε * L³ where ε = 1e-12
    is_degenerate = det_abs < 1e-12 * jnp.maximum(expected_det, 1e-15)

    # Compute barycentric coordinates
    det_inv = jnp.where(is_degenerate, 1.0, 1.0 / det)

    # b1 = det([vp v2 v3]) / det
    b1 = ((vp[0] * (v2[1] * v3[2] - v2[2] * v3[1]) -
           vp[1] * (v2[0] * v3[2] - v2[2] * v3[0]) +
           vp[2] * (v2[0] * v3[1] - v2[1] * v3[0])) * det_inv)

    # b2 = det([v1 vp v3]) / det
    b2 = ((v1[0] * (vp[1] * v3[2] - vp[2] * v3[1]) -
           v1[1] * (vp[0] * v3[2] - vp[2] * v3[0]) +
           v1[2] * (vp[0] * v3[1] - vp[1] * v3[0])) * det_inv)

    # b3 = det([v1 v2 vp]) / det
    b3 = ((v1[0] * (v2[1] * vp[2] - v2[2] * vp[1]) -
           v1[1] * (v2[0] * vp[2] - v2[2] * vp[0]) +
           v1[2] * (v2[0] * vp[1] - v2[1] * vp[0])) * det_inv)

    # b0 = 1 - b1 - b2 - b3
    b0 = 1.0 - b1 - b2 - b3

    # Check if all barycentric coordinates are non-negative
    # Use small tolerance for numerical stability at boundaries
    tol = -1e-6
    inside = (b0 >= tol) & (b1 >= tol) & (b2 >= tol) & (b3 >= tol) & (~is_degenerate)

    return inside


# ============================================================================
# Method 2: Skala - GPU-Optimized Using Cross Products
# ============================================================================

def point_in_tet_skala(
    pos: jax.Array,
    elem_id: jnp.int32,
    connectivity: jax.Array,
    node_positions: jax.Array
) -> jnp.bool_:
    """
    GPU-optimized implementation using cross products (Skala 2014).

    FLOP count: ~48 FLOPs
    - 4× cross products (3D): ~6 FLOPs each = 24 FLOPs
    - 4× dot products (3D): ~3 FLOPs each = 12 FLOPs
    - Normalization + comparison: ~12 FLOPs

    Key insight: Use triple scalar product V = a·(b×c) to compute signed volumes.
    This leverages GPU's native cross product instruction.

    Speedup: ~3× over current method (145 / 48 ≈ 3.0×)

    From Skala (2014), Equation 34-35 and Appendix A:
    - Barycentric λ_i = V_i / V_0
    - V_0 = x_1·(x_2 × x_3) where x_i are edge vectors
    - V_i computed by substituting query point into corresponding position

    Args:
        pos: (3,) float32 - query position
        elem_id: int32 - element ID to test
        connectivity: (n_elements, 4) int32
        node_positions: (n_nodes, 3) float32

    Returns:
        inside: bool - True if pos is in element
    """
    # Get node indices
    nodes = connectivity[elem_id]  # (4,)

    # Get node positions
    p0 = node_positions[nodes[0]]  # (3,)
    p1 = node_positions[nodes[1]]
    p2 = node_positions[nodes[2]]
    p3 = node_positions[nodes[3]]

    # Compute edge vectors from p0 (reference vertex)
    v1 = p1 - p0  # Edge to vertex 1
    v2 = p2 - p0  # Edge to vertex 2
    v3 = p3 - p0  # Edge to vertex 3
    vp = pos - p0  # Vector to query point

    # Compute signed volume of reference tetrahedron using triple scalar product
    # V_0 = v1 · (v2 × v3)
    # This is 6× the signed volume of the tetrahedron
    cross_23 = jnp.cross(v2, v3)  # 6 FLOPs
    V0 = jnp.dot(v1, cross_23)    # 3 FLOPs

    # Handle degenerate tetrahedron (same relative threshold as current method)
    V0_abs = jnp.abs(V0)
    edge_length_sq = jnp.sum(v1 * v1)
    expected_vol = edge_length_sq ** 1.5  # Volume scales as L³
    is_degenerate = V0_abs < 1e-12 * jnp.maximum(expected_vol, 1e-15)
    V0_safe = jnp.where(is_degenerate, 1.0, V0)

    # Compute barycentric coordinates by substituting query point
    # λ_1 = V_1 / V_0 where V_1 = vp · (v2 × v3)
    V1 = jnp.dot(vp, cross_23)  # 3 FLOPs (reuse cross_23)
    lambda1 = V1 / V0_safe

    # λ_2 = V_2 / V_0 where V_2 = v1 · (vp × v3)
    cross_p3 = jnp.cross(vp, v3)  # 6 FLOPs
    V2 = jnp.dot(v1, cross_p3)    # 3 FLOPs
    lambda2 = V2 / V0_safe

    # λ_3 = V_3 / V_0 where V_3 = v1 · (v2 × vp)
    cross_2p = jnp.cross(v2, vp)  # 6 FLOPs
    V3 = jnp.dot(v1, cross_2p)    # 3 FLOPs
    lambda3 = V3 / V0_safe

    # λ_0 = 1 - λ_1 - λ_2 - λ_3
    lambda0 = 1.0 - lambda1 - lambda2 - lambda3

    # Check if all barycentric coordinates are non-negative
    tol = -1e-6
    inside = (lambda0 >= tol) & (lambda1 >= tol) & (lambda2 >= tol) & (lambda3 >= tol) & (~is_degenerate)

    return inside


# ============================================================================
# Method 3: Axis-Aligned - Specialized for Rectilinear Meshes
# ============================================================================

def point_in_tet_axis_aligned(
    pos: jax.Array,
    elem_id: jnp.int32,
    connectivity: jax.Array,
    node_positions: jax.Array
) -> jnp.bool_:
    """
    Specialized implementation for axis-aligned tetrahedra (ThreadedA mesh).

    FLOP count: ~32 FLOPs (detection) + 12 FLOPs (fast path) = ~44 FLOPs average

    For axis-aligned tetrahedra where all edges are parallel to X, Y, or Z:
    1. Detect if tet is axis-aligned (check orthogonality: ~20 FLOPs)
    2. If axis-aligned: Direct barycentric computation (~12 FLOPs)
    3. If not: Fall back to Skala method (~48 FLOPs)

    Expected hit rate: ~100% for ThreadedA mesh (all tets are axis-aligned)

    Speedup: ~12× over current method for axis-aligned tets (145 / 12 ≈ 12×)
    Overall speedup: ~3× average (includes detection overhead + fallback)

    Memory: NO precomputed arrays - uses on-the-fly detection via jax.lax.cond

    Args:
        pos: (3,) float32 - query position
        elem_id: int32 - element ID to test
        connectivity: (n_elements, 4) int32
        node_positions: (n_nodes, 3) float32

    Returns:
        inside: bool - True if pos is in element
    """
    # Get node indices
    nodes = connectivity[elem_id]  # (4,)

    # Get node positions
    p0 = node_positions[nodes[0]]  # (3,)
    p1 = node_positions[nodes[1]]
    p2 = node_positions[nodes[2]]
    p3 = node_positions[nodes[3]]

    # Compute edge vectors from p0
    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0

    # Check if p0 is a right-angled vertex (all edges orthogonal)
    # For axis-aligned tets: e1⊥e2, e1⊥e3, e2⊥e3
    dot12 = jnp.dot(e1, e2)
    dot13 = jnp.dot(e1, e3)
    dot23 = jnp.dot(e2, e3)

    # Orthogonality threshold (more lenient for numerical stability)
    ortho_tol = 1e-8
    is_axis_aligned = (jnp.abs(dot12) < ortho_tol) & \
                      (jnp.abs(dot13) < ortho_tol) & \
                      (jnp.abs(dot23) < ortho_tol)

    # Fast path: Axis-aligned tetrahedron
    def axis_aligned_fast():
        """
        For axis-aligned tet with right-angled vertex at p0:
        - Each edge is parallel to exactly one axis
        - Barycentric coords computed by direct division
        - Example: if e1 = [L1, 0, 0], then λ_1 = (pos - p0)[0] / L1
        """
        local_pos = pos - p0

        # Find dominant axis for each edge (argmax of absolute value)
        idx1 = jnp.argmax(jnp.abs(e1))  # Edge 1's primary axis
        idx2 = jnp.argmax(jnp.abs(e2))  # Edge 2's primary axis
        idx3 = jnp.argmax(jnp.abs(e3))  # Edge 3's primary axis

        # Extract barycentric coordinates directly
        # b_i = (local_pos · axis_i) / (edge_i · axis_i)
        # Since edge is aligned: edge_i · axis_i = edge_length
        b1 = local_pos[idx1] / e1[idx1]
        b2 = local_pos[idx2] / e2[idx2]
        b3 = local_pos[idx3] / e3[idx3]
        b0 = 1.0 - b1 - b2 - b3

        # Check bounds
        tol = -1e-6
        return (b0 >= tol) & (b1 >= tol) & (b2 >= tol) & (b3 >= tol)

    # Slow path: Fall back to Skala method
    def general_fallback():
        """Fall back to Skala's cross-product method for non-axis-aligned tets."""
        return point_in_tet_skala(pos, elem_id, connectivity, node_positions)

    # Use conditional dispatch (no precomputation!)
    # JAX will compile both branches but only execute one
    inside = jax.lax.cond(
        is_axis_aligned,
        axis_aligned_fast,
        general_fallback
    )

    return inside


# ============================================================================
# Module-level storage for corrected AA metadata (set once at mesh load)
# ============================================================================

# Store individual arrays (not dataclass) to avoid JIT compilation issues
_aa_base_vertices_gpu = None
_aa_inv_edge_lengths_gpu = None
_aa_axis_indices_gpu = None
_aa_is_axis_aligned_gpu = None
_element_vertices_gpu = None
_M_inv_gpu = None  # Inverse matrices for "inverse" method
_p0_gpu = None     # First vertices for "inverse" method


def set_corrected_metadata(aa_metadata_gpu, element_vertices_gpu):
    """
    Set module-level corrected AA metadata and element vertices.

    Must be called once after mesh load before using corrected methods.

    Args:
        aa_metadata_gpu: AxisAlignedMetadata on GPU (dataclass with arrays)
        element_vertices_gpu: (n_elements, 4, 3) float32 on GPU
    """
    global _aa_base_vertices_gpu, _aa_inv_edge_lengths_gpu, _aa_axis_indices_gpu
    global _aa_is_axis_aligned_gpu, _element_vertices_gpu

    # Extract individual arrays from dataclass to avoid JIT issues
    _aa_base_vertices_gpu = aa_metadata_gpu.base_vertices
    _aa_inv_edge_lengths_gpu = aa_metadata_gpu.inv_edge_lengths
    _aa_axis_indices_gpu = aa_metadata_gpu.axis_indices
    _aa_is_axis_aligned_gpu = aa_metadata_gpu.is_axis_aligned
    _element_vertices_gpu = element_vertices_gpu


def set_inverse_matrices_gpu(M_inv_gpu, p0_gpu):
    """
    Set module-level precomputed inverse matrices for "inverse" method.

    Must be called once after mesh load before using "inverse" method.

    Args:
        M_inv_gpu: (n_elements, 3, 3) float32 on GPU - inverse transformation matrices
        p0_gpu: (n_elements, 3) float32 on GPU - first vertex of each element
    """
    global _M_inv_gpu, _p0_gpu
    _M_inv_gpu = M_inv_gpu
    _p0_gpu = p0_gpu


# ============================================================================
# Dispatcher with Configuration Switch
# ============================================================================

def point_in_tet_gpu(
    pos: jax.Array,
    elem_id: jnp.int32,
    connectivity: jax.Array,
    node_positions: jax.Array,
    method: str = "skala"
) -> jnp.bool_:
    """
    Dispatcher for point-in-tetrahedron test with configurable method.

    Available methods:

    OLD (original, some broken):
    - "current": Reference implementation (barycentric/Cramer's rule, 145 FLOPs)
    - "skala": GPU-optimized cross products (48 FLOPs, 3× speedup)
    - "axis_aligned": Specialized for axis-aligned meshes (BROKEN - only checks p0)

    NEW (corrected):
    - "pure_aa": Pure axis-aligned (11 FLOPs, 13× speedup, requires 100% AA mesh)
    - "skala_memory_opt": Skala with coalesced memory (48 FLOPs, ~2× speedup)
    - "branchless_hybrid": Hybrid AA/Skala (11-48 FLOPs, ~3-4× speedup for mixed meshes)

    Performance comparison (measured on FLA mesh):
    ┌──────────────────────┬───────────┬──────────────┬─────────────────────┐
    │ Method               │ FLOPs     │ Speedup      │ Notes               │
    ├──────────────────────┼───────────┼──────────────┼─────────────────────┤
    │ current              │ 145       │ 1.0× (base)  │ Baseline            │
    │ skala (OLD)          │ 48        │ 0.9× ❌      │ Memory-bound        │
    │ axis_aligned (OLD)   │ 332       │ 0.45× ❌     │ BROKEN (lax.cond)   │
    │ pure_aa (NEW)        │ 11        │ ?× TBD       │ 100% AA only        │
    │ skala_memory_opt     │ 48        │ ?× TBD       │ Coalesced memory    │
    │ branchless_hybrid    │ 11-48     │ ?× TBD       │ Mixed meshes        │
    └──────────────────────┴───────────┴──────────────┴─────────────────────┘

    NOT JIT-decorated to avoid overhead when used within already-JIT-compiled functions.

    Args:
        pos: (3,) float32 - query position
        elem_id: int32 - element ID to test
        connectivity: (n_elements, 4) int32
        node_positions: (n_nodes, 3) float32
        method: str - method selection

    Returns:
        inside: bool - True if pos is in element
    """
    # OLD methods
    if method == "current":
        return point_in_tet_current(pos, elem_id, connectivity, node_positions)
    elif method == "skala":
        return point_in_tet_skala(pos, elem_id, connectivity, node_positions)
    elif method == "axis_aligned":
        return point_in_tet_axis_aligned(pos, elem_id, connectivity, node_positions)

    # NEW corrected methods - require precomputed metadata
    elif method == "pure_aa":
        if _aa_base_vertices_gpu is None:
            raise RuntimeError("pure_aa method requires set_corrected_metadata() to be called first")
        from jaxtrace.gpu.search.aa_detection import point_in_tet_pure_aa_arrays
        return point_in_tet_pure_aa_arrays(
            pos, elem_id,
            _aa_base_vertices_gpu,
            _aa_inv_edge_lengths_gpu,
            _aa_axis_indices_gpu
        )

    elif method == "skala_memory_opt":
        if _element_vertices_gpu is None:
            raise RuntimeError("skala_memory_opt method requires set_corrected_metadata() to be called first")
        from jaxtrace.gpu.search.aa_detection import point_in_tet_skala_memory_opt
        return point_in_tet_skala_memory_opt(pos, elem_id, _element_vertices_gpu)

    elif method == "branchless_hybrid":
        if _aa_base_vertices_gpu is None or _element_vertices_gpu is None:
            raise RuntimeError("branchless_hybrid method requires set_corrected_metadata() to be called first")
        from jaxtrace.gpu.search.aa_detection import point_in_tet_branchless_hybrid_arrays
        return point_in_tet_branchless_hybrid_arrays(
            pos, elem_id,
            _element_vertices_gpu,
            _aa_base_vertices_gpu,
            _aa_inv_edge_lengths_gpu,
            _aa_axis_indices_gpu,
            _aa_is_axis_aligned_gpu
        )

    elif method == "inverse":
        if _M_inv_gpu is None or _p0_gpu is None:
            raise RuntimeError("inverse method requires set_inverse_matrices_gpu() to be called first")
        from jaxtrace.gpu.search.point_in_tet_inverse import point_in_tet_inverse
        return point_in_tet_inverse(pos, elem_id, _M_inv_gpu, _p0_gpu)

    else:
        raise ValueError(f"Unknown point-in-tet method: {method}. "
                        f"Valid options: 'current', 'skala', 'axis_aligned', "
                        f"'pure_aa', 'skala_memory_opt', 'branchless_hybrid', 'inverse'")
