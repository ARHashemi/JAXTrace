# RK4 Optimization - Final Implementation Plan
## With Memory-Safe Incremental Approach

**Date:** 2026-01-14
**Status:** Production-ready plan addressing OOM concerns
**Context:** Incremental optimization with user-configurable switches

---

## Executive Summary

### Revised Phase Order (Memory-Safe)

**Phase 1: Skala's Cross-Product Method (Baseline GPU Optimization)**
- General-purpose GPU optimization (works for any mesh)
- 3× speedup (145 → 48 FLOPs)
- **NO precomputed arrays** - pure algorithmic improvement
- Serves as reference implementation

**Phase 2: Axis-Aligned Specialization**
- Exploit 100% axis-aligned mesh structure
- 12× speedup (145 → 12 FLOPs)
- **Minimal precomputation** - detect axis-alignment on-the-fly
- User-configurable switch

**Phase 3: AABB Early-Out**
- Filter 90% of L2 candidates
- **Precomputed arrays** but small (84 MB)
- Optional feature (can disable if OOM)

**Phase 4: L1 Optimization**
- Smart neighbor ordering
- **NO additional arrays** - runtime computation only

---

## Memory Safety Strategy

### The JAX OOM Problem

**You're absolutely correct to worry about this!**

JAX's `vmap` and `scan` create **intermediate arrays for all particles simultaneously**:

```python
# This can OOM during compilation:
@jax.jit
def track_particles_vmapped(positions, elem_ids, tet_type, right_angle_vertex, aabb_min, aabb_max):
    # vmap over 225,000 particles
    # JAX compiler creates intermediate buffers:
    #   - tet_type: 225K × 3.5M elements (broadcast) → EXPLODES!
    #   - aabb_min/max: 225K × 3.5M × 6 floats → 47 GB!
    return vmap(track_single_particle)(...)
```

**Root cause:** JAX materializes broadcasted arrays during compilation, even if they're never actually used in full form.

### Solution: Lazy Evaluation + Optional Features

**Strategy:**
1. **Phase 1 (Skala):** Pure algorithmic change - no new arrays
2. **Phase 2 (Axis-aligned):** Detect on-the-fly - no precomputation
3. **Phase 3 (AABB):** Precompute but **make optional** via config flag
4. **Phase 4 (L1):** Runtime computation - no new arrays

---

## Phase 1: Skala's Cross-Product Method

### 1.1 Why Skala First?

**Advantages:**
1. ✅ **No precomputed arrays** - just algorithmic improvement
2. ✅ **General purpose** - works for any mesh (reference implementation)
3. ✅ **GPU-native** - uses cross product (single instruction)
4. ✅ **Proven** - published algorithm with mathematical foundation
5. ✅ **Safe baseline** - 3× speedup without memory risk

**Target performance:**
- Current: 19,357 p/s (145 FLOPs/check)
- After Skala: **58,000-65,000 p/s** (48 FLOPs/check, ~3× speedup)

### 1.2 Implementation: point_in_tet_gpu_skala()

**File:** `jaxtrace/gpu/search/point_in_tet_methods.py` (new file)

```python
"""
Point-in-tetrahedron algorithms for GPU.

Provides multiple implementations:
- Current: Barycentric via Cramer's rule (145 FLOPs) - reference
- Skala: Cross-product method (48 FLOPs) - general GPU optimization
- AxisAligned: Specialized for rectilinear mesh (12 FLOPs) - fast path
"""

import jax
import jax.numpy as jnp


def point_in_tet_current(
    pos: jax.Array,
    elem_id: jnp.int32,
    connectivity: jax.Array,
    node_positions: jax.Array
) -> jnp.bool_:
    """
    Current implementation: Barycentric coordinates via Cramer's rule.

    Operation count: ~145 FLOPs

    This is the reference implementation - kept for validation.
    """
    # [Keep existing implementation from morton_global_search.py:370-453]
    nodes = connectivity[elem_id]
    p0 = node_positions[nodes[0]]
    p1 = node_positions[nodes[1]]
    p2 = node_positions[nodes[2]]
    p3 = node_positions[nodes[3]]

    v1 = p1 - p0
    v2 = p2 - p0
    v3 = p3 - p0
    vp = pos - p0

    # Compute determinant and barycentric coordinates
    det = (v1[0] * (v2[1] * v3[2] - v2[2] * v3[1]) -
           v1[1] * (v2[0] * v3[2] - v2[2] * v3[0]) +
           v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]))

    det_abs = jnp.abs(det)
    edge_length_sq = jnp.sum(v1 * v1)
    expected_det = edge_length_sq ** 1.5
    is_degenerate = det_abs < 1e-12 * jnp.maximum(expected_det, 1e-15)

    det_inv = jnp.where(is_degenerate, 1.0, 1.0 / det)

    b1 = ((vp[0] * (v2[1] * v3[2] - v2[2] * v3[1]) -
           vp[1] * (v2[0] * v3[2] - v2[2] * v3[0]) +
           vp[2] * (v2[0] * v3[1] - v2[1] * v3[0])) * det_inv)

    b2 = ((v1[0] * (vp[1] * v3[2] - vp[2] * v3[1]) -
           v1[1] * (vp[0] * v3[2] - vp[2] * v3[0]) +
           v1[2] * (vp[0] * v3[1] - vp[1] * v3[0])) * det_inv)

    b3 = ((v1[0] * (v2[1] * vp[2] - v2[2] * vp[1]) -
           v1[1] * (v2[0] * vp[2] - v2[2] * vp[0]) +
           v1[2] * (v2[0] * vp[1] - v2[1] * vp[0])) * det_inv)

    b0 = 1.0 - b1 - b2 - b3

    tol = -1e-6
    inside = (b0 >= tol) & (b1 >= tol) & (b2 >= tol) & (b3 >= tol) & (~is_degenerate)

    return inside


def point_in_tet_skala(
    pos: jax.Array,
    elem_id: jnp.int32,
    connectivity: jax.Array,
    node_positions: jax.Array
) -> jnp.bool_:
    """
    Skala (2014): Barycentric coordinates via cross products.

    Based on: "GPU Fast and Robust Computation for Barycentric
    Coordinates", V. Skala, WICT 2014.

    Operation count: ~48 FLOPs (3× faster than current)

    Key advantage: Uses GPU-native cross product instruction.
    No precomputed data needed - pure algorithmic improvement.

    References:
        Skala, V. (2014). Eq. 34-35, Appendix A.
    """
    nodes = connectivity[elem_id]
    p0 = node_positions[nodes[0]]
    p1 = node_positions[nodes[1]]
    p2 = node_positions[nodes[2]]
    p3 = node_positions[nodes[3]]

    # Convert to homogeneous coordinates (append w=1)
    # In E^3: (x, y, z) → (x, y, z, 1) in P^3
    x1 = jnp.concatenate([p0, jnp.array([1.0])])  # (4,)
    x2 = jnp.concatenate([p1, jnp.array([1.0])])
    x3 = jnp.concatenate([p2, jnp.array([1.0])])
    x4 = jnp.concatenate([p3, jnp.array([1.0])])
    x  = jnp.concatenate([pos, jnp.array([1.0])])

    # Compute projective barycentric coordinates
    # ξ = x × x2 × x3 × x4  (4D cross product)
    # See Skala Eq. 34 and Appendix A

    # Component-wise computation using 3D cross products
    # ξ₁ = dot(x.yzw, cross(x2.yzw, cross(x3.yzw, x4.yzw)))
    # ξ₂ = -dot(x.xzw, cross(x2.xzw, cross(x3.xzw, x4.xzw)))
    # ξ₃ = dot(x.xyw, cross(x2.xyw, cross(x3.xyw, x4.xyw)))
    # ξw = -dot(x.xyz, cross(x2.xyz, cross(x3.xyz, x4.xyz)))

    # Simplified: Compute 4 barycentric coordinates directly
    # Using triple scalar product: [a, b, c] = dot(a, cross(b, c))

    # Volume of tetrahedron [x2, x3, x4] relative to origin
    V0 = jnp.dot(x2[:3], jnp.cross(x3[:3], x4[:3]))

    # Volumes of sub-tetrahedra (replace vertices one at a time)
    V1 = jnp.dot(x[:3], jnp.cross(x3[:3], x4[:3]))   # Replace x2 with x
    V2 = jnp.dot(x2[:3], jnp.cross(x[:3], x4[:3]))   # Replace x3 with x
    V3 = jnp.dot(x2[:3], jnp.cross(x3[:3], x[:3]))   # Replace x4 with x

    # Barycentric coordinates (Skala Eq. 35)
    # Handle degenerate case
    V0_abs = jnp.abs(V0)
    is_degenerate = V0_abs < 1e-15

    V0_safe = jnp.where(is_degenerate, 1.0, V0)

    lambda1 = V1 / V0_safe
    lambda2 = V2 / V0_safe
    lambda3 = V3 / V0_safe
    lambda0 = 1.0 - lambda1 - lambda2 - lambda3

    # Containment check
    tol = -1e-6
    inside = (lambda0 >= tol) & (lambda1 >= tol) & \
             (lambda2 >= tol) & (lambda3 >= tol) & (~is_degenerate)

    return inside


def point_in_tet_axis_aligned(
    pos: jax.Array,
    elem_id: jnp.int32,
    connectivity: jax.Array,
    node_positions: jax.Array
) -> jnp.bool_:
    """
    Specialized for axis-aligned rectilinear tetrahedra.

    Assumes: 100% of mesh tetrahedra have axis-aligned edges.

    Operation count: ~12 FLOPs (12× faster than current, 4× faster than Skala)

    Algorithm:
        1. Detect which vertex is right-angled (one with 3 perpendicular edges)
        2. Transform to local coordinates (origin at right-angle vertex)
        3. Edges become (dx,0,0), (0,dy,0), (0,0,dz)
        4. Barycentric coords: b1=x/dx, b2=y/dy, b3=z/dz
        5. Check: all b_i ∈ [0,1]

    Note: This is called AFTER Skala if enabled via config.
    Detection is done on-the-fly (no precomputation).
    """
    nodes = connectivity[elem_id]
    p0 = node_positions[nodes[0]]
    p1 = node_positions[nodes[1]]
    p2 = node_positions[nodes[2]]
    p3 = node_positions[nodes[3]]

    # Edge vectors from p0
    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0

    # Check if p0 is the right-angled vertex
    # For axis-aligned tet: edges are perpendicular (dot products = 0)
    dot12 = jnp.dot(e1, e2)
    dot13 = jnp.dot(e1, e3)
    dot23 = jnp.dot(e2, e3)

    is_right_angled_at_p0 = (jnp.abs(dot12) < 1e-10) & \
                             (jnp.abs(dot13) < 1e-10) & \
                             (jnp.abs(dot23) < 1e-10)

    # If not right-angled at p0, fall back to Skala
    # (This branch will rarely be taken for your mesh)

    def axis_aligned_fast_path():
        """Fast path: p0 is right-angled, edges are axis-aligned."""
        local_pos = pos - p0

        # Each edge should be aligned with one axis
        # Find which component is non-zero for each edge
        abs_e1 = jnp.abs(e1)
        abs_e2 = jnp.abs(e2)
        abs_e3 = jnp.abs(e3)

        # Barycentric coordinates (divide by non-zero component)
        # Safe division: pick component with largest magnitude
        idx1 = jnp.argmax(abs_e1)
        idx2 = jnp.argmax(abs_e2)
        idx3 = jnp.argmax(abs_e3)

        b1 = local_pos[idx1] / e1[idx1]
        b2 = local_pos[idx2] / e2[idx2]
        b3 = local_pos[idx3] / e3[idx3]
        b0 = 1.0 - b1 - b2 - b3

        tol = -1e-6
        return (b0 >= tol) & (b1 >= tol) & (b2 >= tol) & (b3 >= tol)

    def fallback_skala():
        """Fallback: Use Skala method."""
        return point_in_tet_skala(pos, elem_id, connectivity, node_positions)

    # Use fast path if p0 is right-angled, otherwise fallback
    return jax.lax.cond(
        is_right_angled_at_p0,
        axis_aligned_fast_path,
        fallback_skala
    )


# ============================================================================
# User-Configurable Dispatcher
# ============================================================================

def point_in_tet_gpu(
    pos: jax.Array,
    elem_id: jnp.int32,
    connectivity: jax.Array,
    node_positions: jax.Array,
    method: str = "skala"  # Options: "current", "skala", "axis_aligned"
) -> jnp.bool_:
    """
    Dispatcher for point-in-tet methods.

    Args:
        pos: (3,) query position
        elem_id: element ID to test
        connectivity: (n_elements, 4)
        node_positions: (n_nodes, 3)
        method: Algorithm to use
            - "current": Barycentric/Cramer (145 FLOPs) - reference
            - "skala": Cross-product (48 FLOPs) - general GPU optimization
            - "axis_aligned": Specialized (12 FLOPs) - for rectilinear mesh

    Returns:
        inside: bool
    """
    if method == "current":
        return point_in_tet_current(pos, elem_id, connectivity, node_positions)
    elif method == "skala":
        return point_in_tet_skala(pos, elem_id, connectivity, node_positions)
    elif method == "axis_aligned":
        return point_in_tet_axis_aligned(pos, elem_id, connectivity, node_positions)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'current', 'skala', or 'axis_aligned'.")
```

### 1.3 Configuration

**File:** `config.py` (add to production script or create new config file)

```python
# Point-in-tetrahedron algorithm selection
POINT_IN_TET_METHOD = "skala"  # Options: "current", "skala", "axis_aligned"

# Phase 3: AABB early-out (optional - may cause OOM with vmap)
USE_AABB_FILTER = False  # Set to True after testing memory usage

# Phase 4: L1 optimization
L1_SMART_NEIGHBOR_ORDERING = False  # Enable after Phase 4
L1_ADAPTIVE_SKIP = False  # Enable after Phase 4
```

### 1.4 Integration

**Modify search functions to use new dispatcher:**

```python
# In morton_global_search.py, replace all calls:
# OLD:
#   if point_in_tet_gpu(pos, elem_id, connectivity, node_positions):
#
# NEW:
from jaxtrace.gpu.search.point_in_tet_methods import point_in_tet_gpu
from config import POINT_IN_TET_METHOD

if point_in_tet_gpu(pos, elem_id, connectivity, node_positions, method=POINT_IN_TET_METHOD):
    # ...
```

### 1.5 Testing & Validation

**Create unit test:**

```python
# test_point_in_tet_methods.py

import numpy as np
import jax
import jax.numpy as jnp
from jaxtrace.gpu.search.point_in_tet_methods import (
    point_in_tet_current,
    point_in_tet_skala,
    point_in_tet_axis_aligned
)

def test_methods_agreement():
    """Verify all methods give identical results."""
    # Load test mesh
    connectivity, node_positions = load_test_mesh()

    # Generate random test points
    n_tests = 10000
    test_positions = np.random.uniform(-0.03, 0.03, (n_tests, 3))
    test_elements = np.random.randint(0, connectivity.shape[0], n_tests)

    results_current = []
    results_skala = []
    results_axis = []

    for i in range(n_tests):
        pos = test_positions[i]
        elem_id = test_elements[i]

        r_current = point_in_tet_current(pos, elem_id, connectivity, node_positions)
        r_skala = point_in_tet_skala(pos, elem_id, connectivity, node_positions)
        r_axis = point_in_tet_axis_aligned(pos, elem_id, connectivity, node_positions)

        results_current.append(r_current)
        results_skala.append(r_skala)
        results_axis.append(r_axis)

    # Check agreement
    agreement_skala = np.sum(np.array(results_current) == np.array(results_skala))
    agreement_axis = np.sum(np.array(results_current) == np.array(results_axis))

    print(f"Skala agreement: {agreement_skala}/{n_tests} ({100*agreement_skala/n_tests:.2f}%)")
    print(f"Axis-aligned agreement: {agreement_axis}/{n_tests} ({100*agreement_axis/n_tests:.2f}%)")

    assert agreement_skala >= 0.999 * n_tests, "Skala method disagrees with reference"
    assert agreement_axis >= 0.999 * n_tests, "Axis-aligned method disagrees with reference"


def benchmark_methods():
    """Benchmark performance of each method."""
    import time

    connectivity, node_positions = load_test_mesh()
    n_tests = 100000
    test_positions = np.random.uniform(-0.03, 0.03, (n_tests, 3))
    test_elements = np.random.randint(0, connectivity.shape[0], n_tests)

    methods = [
        ("current", point_in_tet_current),
        ("skala", point_in_tet_skala),
        ("axis_aligned", point_in_tet_axis_aligned)
    ]

    for name, func in methods:
        # JIT compile
        func_jit = jax.jit(func)

        # Warmup
        _ = func_jit(test_positions[0], test_elements[0], connectivity, node_positions)

        # Benchmark
        start = time.perf_counter()
        for i in range(n_tests):
            _ = func_jit(test_positions[i], test_elements[i], connectivity, node_positions)
        elapsed = time.perf_counter() - start

        print(f"{name:20s}: {elapsed:.3f}s total, {elapsed/n_tests*1e6:.2f}µs per call")
```

---

## Phase 2: Axis-Aligned Specialization

**Implementation:** Already included in Phase 1 as `point_in_tet_axis_aligned()`

**Activation:**
```python
# In config.py
POINT_IN_TET_METHOD = "axis_aligned"  # Switch from "skala"
```

**Key difference from original plan:**
- ✅ **NO precomputed arrays** (tet_type, right_angle_vertex, etc.)
- ✅ **On-the-fly detection** using `jax.lax.cond`
- ✅ **Memory-safe** - no OOM risk with vmap

**Trade-off:**
- Slightly more FLOPs than precomputed version (~15-20 vs 12)
- But still 8-10× faster than current, and 2-3× faster than Skala
- **Worth it to avoid OOM!**

---

## Phase 3: AABB Early-Out (Optional)

### 3.1 Memory Safety Analysis

**AABB arrays:**
- `aabb_min`: (3,494,800 elements, 3) = 42 MB
- `aabb_max`: (3,494,800 elements, 3) = 42 MB
- **Total: 84 MB**

**OOM risk with vmap:**
```python
# This is the problem:
@jax.jit
def search_l2_vmapped(positions, candidate_lists, aabb_min, aabb_max):
    # positions: (225K particles, 3)
    # candidate_lists: (225K particles, 200 candidates) - varies per particle
    # aabb_min: (3.5M elements, 3)
    # aabb_max: (3.5M elements, 3)

    # vmap broadcasts aabb_min/max to all particles
    # JAX creates intermediate: (225K, 3.5M, 3) → 23 GB!
    return vmap(search_single_particle)(positions, candidate_lists, aabb_min, aabb_max)
```

**Solution: Make AABB optional and use only in L2 search (not vmapped)**

### 3.2 Safe Implementation

```python
# In config.py
USE_AABB_FILTER = False  # Default: disabled

# Only enable if:
# 1. You've verified no OOM with your batch size
# 2. L2 search is still a bottleneck after Phase 1+2
```

**If enabled, use AABB only in L2 (non-vmapped):**

```python
def search_l2_global_with_aabb(pos, candidate_elements, aabb_min=None, aabb_max=None, ...):
    """
    L2 search with optional AABB filter.

    This is called per-particle (not vmapped), so no OOM risk.
    """
    if aabb_min is not None and aabb_max is not None:
        # Filter candidates by AABB (fast)
        candidates_filtered = []
        for elem_id in candidate_elements:
            if ((pos >= aabb_min[elem_id]) & (pos <= aabb_max[elem_id])).all():
                candidates_filtered.append(elem_id)
        candidate_elements = candidates_filtered

    # Full point-in-tet on filtered candidates
    for elem_id in candidate_elements:
        if point_in_tet_gpu(pos, elem_id, connectivity, node_positions, method=POINT_IN_TET_METHOD):
            return elem_id

    return -1
```

---

## Phase 4: L1 Optimization

**No additional arrays needed** - all computation is runtime:

```python
def search_l1_optimized(pos, elem_id, velocity_direction=None):
    """
    Optimized L1 neighbor search.

    NO precomputed arrays - compute neighbor ordering on-the-fly.
    """
    neighbors = neighbor_list[elem_id]  # Existing array (already loaded)

    if velocity_direction is not None:
        # Smart ordering: Sort neighbors by alignment with velocity
        # This is O(4 log 4) = 8 ops (negligible)
        neighbor_centroids = compute_centroids_from_connectivity(neighbors)
        scores = jnp.dot(neighbor_centroids, velocity_direction)
        neighbors = neighbors[jnp.argsort(-scores)]

    # Check neighbors (up to 6 hops)
    for hop in range(6):
        for neighbor_id in neighbors:
            if point_in_tet_gpu(pos, neighbor_id, connectivity, node_positions, method=POINT_IN_TET_METHOD):
                return neighbor_id

    return -1  # Not found, proceed to L2
```

---

## Summary: Memory-Safe Implementation Order

| Phase | Optimization | New Arrays | OOM Risk | Speedup | Cumulative |
|-------|-------------|-----------|----------|---------|------------|
| **Phase 1: Skala** | Cross-product | **None** | ✅ Safe | 3× | 3× |
| **Phase 2: Axis-aligned** | On-the-fly detection | **None** | ✅ Safe | 4× | 12× |
| **Phase 3: AABB (optional)** | Precomputed AABBs | 84 MB | ⚠️ Test first | 1.2× | 14× |
| **Phase 4: L1** | Runtime ordering | **None** | ✅ Safe | 1.15× | 16× |

**Key insight:** Phases 1, 2, and 4 are **memory-safe by design** - no risk of OOM!

---

## Configuration File

Create `config.py`:

```python
"""
JAXTrace GPU Tracking Configuration
"""

# ============================================================================
# Point-in-Tetrahedron Algorithm Selection
# ============================================================================

# Phase 1: Skala's cross-product method (general GPU optimization)
# Phase 2: Axis-aligned specialization (for rectilinear meshes)
POINT_IN_TET_METHOD = "skala"  # Options: "current", "skala", "axis_aligned"

# Recommended progression:
#   1. Start with "skala" (safe 3× speedup)
#   2. Switch to "axis_aligned" after validating (12× speedup)
#   3. Keep "current" available for debugging/comparison

# ============================================================================
# Phase 3: AABB Early-Out (Optional - May Cause OOM)
# ============================================================================

USE_AABB_FILTER = False  # Default: disabled

# Enable ONLY if:
#   1. Phases 1+2 are working correctly
#   2. You've verified no OOM with your particle batch size
#   3. L2 search is still a bottleneck (profile first!)

# If enabled, AABB arrays add 84 MB memory:
#   aabb_min: (n_elements, 3) float32
#   aabb_max: (n_elements, 3) float32

# ============================================================================
# Phase 4: L1 Neighbor Search Optimization
# ============================================================================

L1_SMART_NEIGHBOR_ORDERING = False  # Order by velocity direction
L1_ADAPTIVE_SKIP = False  # Skip L1 for very refined elements
L1_MAX_HOPS = 3  # Increase to 6 after Phase 1+2 (cheaper per-check)

# Smart ordering: Sort neighbors by dot(centroid, velocity)
# Adaptive skip: Skip L1 if element volume < threshold

# ============================================================================
# Diagnostic and Logging
# ============================================================================

PROFILE_POINT_IN_TET = False  # Count calls per method
VALIDATE_RETENTION = True  # Check particle retention at each step
LOG_LEVEL = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
```

---

## Testing & Validation Checklist

### After Phase 1 (Skala)

- [ ] Unit test: 100% agreement with current method (10K queries)
- [ ] Microbenchmark: 2.5-3.5× speedup for point-in-tet
- [ ] Production test: 1 timestep, verify retention ≥ 93%
- [ ] Production test: 100 timesteps, measure throughput
- [ ] **Target: 55,000-65,000 p/s** (vs 19,357 baseline)

### After Phase 2 (Axis-Aligned)

- [ ] Unit test: 100% agreement with Skala (10K queries)
- [ ] Microbenchmark: 10-12× speedup vs current
- [ ] Production test: 100 timesteps
- [ ] **Target: 180,000-230,000 p/s** (cumulative 10-12×)

### After Phase 3 (AABB - if enabled)

- [ ] Memory test: No OOM with full particle count
- [ ] AABB rejection rate: 85-95% for L2 candidates
- [ ] Production test: 100 timesteps
- [ ] **Target: 200,000-260,000 p/s** (additional 1.2×)

### After Phase 4 (L1 Optimization)

- [ ] Neighbor ordering: Verify 30-50% fewer L1 checks
- [ ] Adaptive skip: Measure skip rate for refined elements
- [ ] Production test: Full 2,500 timesteps
- [ ] **Target: 230,000-300,000 p/s** (cumulative 12-16×)
- [ ] Retention: ≥95%

---

## Next Steps

1. **Create `point_in_tet_methods.py`** with all three implementations
2. **Add `config.py`** with user-configurable switches
3. **Update `morton_global_search.py`** to use new dispatcher
4. **Create unit tests** in `test_point_in_tet_methods.py`
5. **Run Phase 1 validation** with `POINT_IN_TET_METHOD = "skala"`

Would you like me to start implementing Phase 1 (Skala method)?
