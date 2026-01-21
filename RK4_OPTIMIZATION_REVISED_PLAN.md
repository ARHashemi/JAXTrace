# RK4 Optimization - Revised Implementation Plan
## Specialized for Axis-Aligned Rectilinear Tetrahedral Mesh

**Date:** 2026-01-14
**Status:** Optimized plan based on mesh analysis
**Context:** All tetrahedral edges are axis-aligned → **Massive optimization potential**

---

## Executive Summary

### Your Mesh: Perfect Structure for Optimization

Based on [THREADEDA_MESH_ANALYSIS.md](docs/THREADEDA_MESH_ANALYSIS.md):
- **3.49M tetrahedral elements** (all with **axis-aligned edges**)
- **898K nodes** (after deduplication: ~571K nodes)
- **Rectilinear structure:** 4-6 tetrahedra per cube decomposition
- **Adaptive refinement:** 8.12e-14 to 2.13e-08 volume range (262,146× span)

### Critical Insight: 100% Axis-Aligned Tetrahedra

**All edges parallel to X, Y, or Z axes** → Point-in-tet becomes **trivial axis-aligned box checks**

**Current performance:**
- With L1: 19,357 p/s, 11.6s/step, 93.57% retention
- Without L1: 25,946 p/s, 8.7s/step, 70.51% retention

**Target performance (with optimizations):**
- **Phase 1+2 (axis-aligned point-in-tet + AABB):** 60,000-80,000 p/s (~3× speedup)
- **Phase 3 (L1 optimization):** 100,000+ p/s (~5× overall speedup)

---

## Why Previous Plan Needs Revision

### Original Plan (from RK4_PERFORMANCE_ANALYSIS_AND_OPTIMIZATION.md)

**Phase priorities were:**
1. Skala's cross-product method (3× FLOP reduction: 145 → 48 FLOPs)
2. AABB early-out (90% rejection for L2)
3. Exploit right-angled structure (research needed)

### Why This is WRONG for Your Mesh

**Your mesh has 100% axis-aligned tetrahedra** → You should exploit this FIRST, not third!

**Comparison:**

| Method | FLOPs | Speedup | Works for your mesh? |
|--------|-------|---------|---------------------|
| **Current (barycentric)** | 145 | 1× baseline | ✅ Yes (general) |
| **Skala (cross-product)** | 48 | 3× | ✅ Yes (general) |
| **Axis-aligned (specialized)** | **~12** | **12×** | ✅ **YES! (100% of tets)** |
| **AABB early-out** | 6 | N/A (pre-filter) | ✅ Yes (L2 only) |

**Correct priority:**
1. **Axis-aligned point-in-tet** (12× speedup for 100% of checks)
2. **AABB early-out** (filters 90% of L2 candidates before full test)
3. **Skip Skala's method** (unnecessary - axis-aligned is better)

---

## Revised Implementation Plan

### Phase 1: Axis-Aligned Point-in-Tetrahedron (HIGHEST PRIORITY)

**Rationale:** ALL your tetrahedra have axis-aligned edges → **12× speedup guaranteed**

#### 1.1 Understanding Axis-Aligned Tetrahedra

**Standard cube-to-tet decomposition (5 tets per cube):**

```
Cube with corners at (x0,y0,z0) to (x1,y1,z1):
  v0 = (x0, y0, z0)   v4 = (x0, y0, z1)
  v1 = (x1, y0, z0)   v5 = (x1, y0, z1)
  v2 = (x1, y1, z0)   v6 = (x1, y1, z1)
  v3 = (x0, y1, z0)   v7 = (x0, y1, z1)

5 tetrahedra (Kuhn decomposition):
  Tet 1: (v0, v1, v2, v5)
  Tet 2: (v0, v2, v3, v7)
  Tet 3: (v0, v5, v7, v4)
  Tet 4: (v2, v5, v6, v7)
  Tet 5: (v0, v2, v5, v7)  ← Central tet
```

**Alternative 6-tet decomposition (more common):**
Each tet has 3 right angles at one vertex, edges aligned with axes.

**Key property:** Every edge vector is one of:
- `(±dx, 0, 0)` - parallel to X
- `(0, ±dy, 0)` - parallel to Y
- `(0, 0, ±dz)` - parallel to Z

#### 1.2 Specialized Point-in-Tet Algorithm

**Method 1: Direct Axis-Aligned Check (Simplest)**

For tetrahedron with **one right-angled vertex** at `v0`:

```python
def point_in_tet_axis_aligned_simple(pos, elem_id, connectivity, node_positions):
    """
    Ultra-fast point-in-tet for axis-aligned tetrahedra.

    Assumes: One vertex (v0) has 3 right angles, edges aligned with axes.

    Algorithm:
      1. Transform to local coordinates (origin at v0)
      2. Edges become: e1=(dx,0,0), e2=(0,dy,0), e3=(0,0,dz)
      3. Barycentric coords: b1=x/dx, b2=y/dy, b3=z/dz
      4. Check: all b_i ∈ [0,1] and b0+b1+b2+b3 ≤ 1

    Operation count: ~12 FLOPs (vs 145 in current implementation)
    """
    # Get element nodes
    nodes = connectivity[elem_id]  # (4,)

    # Get node positions
    p0 = node_positions[nodes[0]]  # Right-angled vertex (assumed at index 0)
    p1 = node_positions[nodes[1]]
    p2 = node_positions[nodes[2]]
    p3 = node_positions[nodes[3]]

    # Transform to local coordinates (3 subtractions)
    local_pos = pos - p0  # (3,)

    # Compute edge vectors (3 × 3 = 9 subtractions)
    e1 = p1 - p0  # Should be (dx, 0, 0) for axis-aligned tet
    e2 = p2 - p0  # Should be (0, dy, 0)
    e3 = p3 - p0  # Should be (0, 0, dz)

    # Identify which axis each edge aligns with (3 comparisons)
    # Find non-zero component for each edge
    # For e1: if |e1[0]| > |e1[1]| and |e1[0]| > |e1[2]|, it's X-aligned

    # Barycentric coordinates (3 divisions)
    # Assuming e1 is X-aligned, e2 is Y-aligned, e3 is Z-aligned:
    b1 = local_pos[0] / e1[0]  # X component / edge1 length
    b2 = local_pos[1] / e2[1]  # Y component / edge2 length
    b3 = local_pos[2] / e3[2]  # Z component / edge3 length
    b0 = 1.0 - b1 - b2 - b3    # (3 subtractions)

    # Check containment (5 comparisons + 5 AND operations)
    tol = -1e-6
    inside = (b0 >= tol) & (b1 >= tol) & (b2 >= tol) & (b3 >= tol) & (b0 <= 1.0 + tol)

    return inside
```

**Operation count:**
- Coordinate transform: 3 subtractions
- Edge vectors: 9 subtractions
- Barycentric coords: 3 divisions + 3 subtractions
- Containment check: 5 comparisons + 5 AND
- **Total: ~23 ops** (including edge detection)

**Simplified (if edges pre-classified):**
- If we precompute which vertex is right-angled and edge orientations: **12 ops**

**Method 2: AABB Check (Even Simpler for Rectilinear Tets)**

For **right-angled tetrahedra**, the containment test is equivalent to:
1. Check if point is inside AABB of 4 vertices
2. Check if point is on correct side of diagonal plane

```python
def point_in_tet_axis_aligned_aabb(pos, elem_id, connectivity, node_positions):
    """
    Even faster: Treat axis-aligned tet as AABB + half-space test.

    For right-angled tet with v0 at right angle:
      - AABB: [v0, v0+e1+e2+e3]
      - Diagonal plane: b0+b1+b2+b3 = 1

    Operation count: ~10 FLOPs
    """
    nodes = connectivity[elem_id]
    p0, p1, p2, p3 = node_positions[nodes[0]], node_positions[nodes[1]], \
                     node_positions[nodes[2]], node_positions[nodes[3]]

    # AABB bounds (6 min/max operations)
    x_min, x_max = min(p0[0], p1[0], p2[0], p3[0]), max(p0[0], p1[0], p2[0], p3[0])
    y_min, y_max = min(p0[1], p1[1], p2[1], p3[1]), max(p0[1], p1[1], p2[1], p3[1])
    z_min, z_max = min(p0[2], p1[2], p2[2], p3[2]), max(p0[2], p1[2], p2[2], p3[2])

    # AABB check (6 comparisons)
    in_aabb = (pos[0] >= x_min) & (pos[0] <= x_max) & \
              (pos[1] >= y_min) & (pos[1] <= y_max) & \
              (pos[2] >= z_min) & (pos[2] <= z_max)

    if not in_aabb:
        return False

    # Diagonal plane check (for right-angled tet)
    # Transform to local coords and check b0+b1+b2+b3 ≤ 1
    local_pos = pos - p0
    e1, e2, e3 = p1 - p0, p2 - p0, p3 - p0

    # Barycentric sum (simplified)
    b_sum = local_pos[0]/e1[0] + local_pos[1]/e2[1] + local_pos[2]/e3[2]

    return (in_aabb) & (b_sum <= 1.0 + 1e-6) & (b_sum >= -1e-6)
```

**Operation count: ~16 ops** (6 min/max + 6 comparisons + 3 divisions + 1 comparison)

#### 1.3 Precomputation: Classify Tetrahedra

**Idea:** Precompute which tetrahedra are axis-aligned and cache metadata

```python
def precompute_tet_classification(connectivity, node_positions):
    """
    Classify each tetrahedron as:
      - Type 0: General (not axis-aligned) - use current method
      - Type 1: Axis-aligned with right angle at vertex 0
      - Type 2: Axis-aligned with right angle at vertex 1
      - ... etc

    Returns:
      tet_type: (n_elements,) int8 - type classification
      right_angle_vertex: (n_elements,) int8 - which vertex has right angle
      edge_axes: (n_elements, 3, 3) int8 - which axis each edge aligns with
    """
    n_elements = connectivity.shape[0]
    tet_type = np.zeros(n_elements, dtype=np.int8)
    right_angle_vertex = np.zeros(n_elements, dtype=np.int8)

    for elem_id in range(n_elements):
        nodes = connectivity[elem_id]
        p0, p1, p2, p3 = node_positions[nodes]

        # Check each vertex for right angle
        for v_idx in range(4):
            # Get edges from this vertex
            if v_idx == 0:
                e1, e2, e3 = p1 - p0, p2 - p0, p3 - p0
            elif v_idx == 1:
                e1, e2, e3 = p0 - p1, p2 - p1, p3 - p1
            # ... etc

            # Check if edges are mutually perpendicular and axis-aligned
            dot_12 = np.dot(e1, e2)
            dot_13 = np.dot(e1, e3)
            dot_23 = np.dot(e2, e3)

            if abs(dot_12) < 1e-10 and abs(dot_13) < 1e-10 and abs(dot_23) < 1e-10:
                # Found right-angled vertex
                # Check if edges are axis-aligned
                if is_axis_aligned(e1) and is_axis_aligned(e2) and is_axis_aligned(e3):
                    tet_type[elem_id] = 1  # Axis-aligned
                    right_angle_vertex[elem_id] = v_idx
                    break

    return tet_type, right_angle_vertex

def is_axis_aligned(edge):
    """Check if edge is parallel to X, Y, or Z axis."""
    abs_edge = np.abs(edge)
    # Edge is axis-aligned if exactly 2 components are ~zero
    num_zeros = np.sum(abs_edge < 1e-10)
    return num_zeros == 2
```

**Precomputation cost:**
- One-time during mesh loading
- ~10-20 seconds for 3.5M elements
- Storage: 3.5M × 1 byte = **3.5 MB** (tet_type)

**Expected result:** Based on your mesh analysis, **~95-100% of tets will be Type 1 (axis-aligned)**

#### 1.4 Implementation

**File:** [jaxtrace/gpu/search/morton_global_search.py](jaxtrace/gpu/search/morton_global_search.py)

**Step 1: Add precomputation to mesh loading**

```python
# In production script after deduplication
from jaxtrace.gpu.search.tet_classification import precompute_tet_classification

tet_type, right_angle_vertex = precompute_tet_classification(
    connectivity, node_positions
)

# Upload to GPU
tet_type_gpu = jax.device_put(tet_type, device)
right_angle_vertex_gpu = jax.device_put(right_angle_vertex, device)
```

**Step 2: Replace point_in_tet_gpu() with hybrid version**

```python
def point_in_tet_gpu_hybrid(
    pos: jax.Array,
    elem_id: jnp.int32,
    connectivity: jax.Array,
    node_positions: jax.Array,
    tet_type: jax.Array,
    right_angle_vertex: jax.Array
) -> jnp.bool_:
    """
    Hybrid point-in-tet: Fast path for axis-aligned, fallback for general.

    Args:
        pos: (3,) query position
        elem_id: element ID to test
        connectivity: (n_elements, 4)
        node_positions: (n_nodes, 3)
        tet_type: (n_elements,) - 0=general, 1=axis-aligned
        right_angle_vertex: (n_elements,) - which vertex is right-angled

    Returns:
        inside: bool
    """
    # Check tet type
    is_axis_aligned = tet_type[elem_id] == 1

    # Fast path: axis-aligned tet
    result_fast = point_in_tet_axis_aligned(
        pos, elem_id, connectivity, node_positions, right_angle_vertex
    )

    # Slow path: general tet (current method)
    result_slow = point_in_tet_gpu_current(
        pos, elem_id, connectivity, node_positions
    )

    # Return correct result based on tet type
    return jnp.where(is_axis_aligned, result_fast, result_slow)


def point_in_tet_axis_aligned(
    pos: jax.Array,
    elem_id: jnp.int32,
    connectivity: jax.Array,
    node_positions: jax.Array,
    right_angle_vertex: jax.Array
) -> jnp.bool_:
    """
    Fast point-in-tet for axis-aligned tetrahedra.
    ~12 FLOPs (vs 145 in general case).
    """
    nodes = connectivity[elem_id]

    # Reorder vertices so v0 is the right-angled vertex
    right_vertex_idx = right_angle_vertex[elem_id]

    # Use lax.switch to handle different vertex orderings
    def case_0():  # v0 is right-angled (already correct order)
        return nodes[0], nodes[1], nodes[2], nodes[3]
    def case_1():  # v1 is right-angled
        return nodes[1], nodes[0], nodes[2], nodes[3]
    def case_2():  # v2 is right-angled
        return nodes[2], nodes[0], nodes[1], nodes[3]
    def case_3():  # v3 is right-angled
        return nodes[3], nodes[0], nodes[1], nodes[2]

    v0_id, v1_id, v2_id, v3_id = jax.lax.switch(
        right_vertex_idx,
        [case_0, case_1, case_2, case_3]
    )

    p0 = node_positions[v0_id]
    p1 = node_positions[v1_id]
    p2 = node_positions[v2_id]
    p3 = node_positions[v3_id]

    # Transform to local coordinates
    local_pos = pos - p0  # (3,)

    # Edge vectors (guaranteed axis-aligned)
    e1 = p1 - p0  # (dx, 0, 0) or (0, dy, 0) or (0, 0, dz)
    e2 = p2 - p0
    e3 = p3 - p0

    # Find which component is non-zero for each edge
    # e1[0] != 0 → X-aligned, e1[1] != 0 → Y-aligned, e1[2] != 0 → Z-aligned

    # Safe division (avoid divide-by-zero)
    def safe_divide(numerator, edge):
        # Find non-zero component
        abs_edge = jnp.abs(edge)
        max_comp = jnp.argmax(abs_edge)
        denominator = edge[max_comp]
        result = numerator[max_comp] / denominator
        return result

    b1 = safe_divide(local_pos, e1)
    b2 = safe_divide(local_pos, e2)
    b3 = safe_divide(local_pos, e3)
    b0 = 1.0 - b1 - b2 - b3

    # Containment check
    tol = -1e-6
    inside = (b0 >= tol) & (b1 >= tol) & (b2 >= tol) & (b3 >= tol)

    return inside
```

**Expected speedup:**
- If 100% of tets are axis-aligned: **12× faster point-in-tet** (145 → 12 FLOPs)
- Overall RK4 speedup (if point-in-tet is 60% of runtime): **~4× overall**

---

### Phase 2: AABB Early-Out for L2 Search (HIGH PRIORITY)

**Rationale:** L2 global search checks 50-200 elements → AABB filters 90% before full test

#### 2.1 Precompute Element AABBs

```python
def compute_element_aabbs(connectivity, node_positions):
    """
    Precompute tight AABB for each element.

    For axis-aligned tets, AABB is exact (tet fills AABB perfectly).

    Returns:
        aabb_min: (n_elements, 3) float32
        aabb_max: (n_elements, 3) float32
    """
    n_elements = connectivity.shape[0]
    aabb_min = np.zeros((n_elements, 3), dtype=np.float32)
    aabb_max = np.zeros((n_elements, 3), dtype=np.float32)

    for elem_id in range(n_elements):
        nodes = connectivity[elem_id]
        positions = node_positions[nodes]  # (4, 3)

        aabb_min[elem_id] = positions.min(axis=0)
        aabb_max[elem_id] = positions.max(axis=0)

    return aabb_min, aabb_max
```

**Storage:** 3.5M elements × 6 floats × 4 bytes = **84 MB**

#### 2.2 Modified L2 Search with AABB Filter

```python
def search_l2_with_aabb_filter(
    pos: jax.Array,
    candidate_elements: jax.Array,  # (n_candidates,) from Morton search
    aabb_min: jax.Array,
    aabb_max: jax.Array,
    connectivity: jax.Array,
    node_positions: jax.Array,
    tet_type: jax.Array,
    right_angle_vertex: jax.Array
) -> jnp.int32:
    """
    L2 search with AABB pre-filter.

    For each candidate:
      1. AABB test (6 comparisons) - rejects 90% of candidates
      2. Full point-in-tet (12 ops for axis-aligned, 145 for general)

    Expected speedup: 5-10× for L2 search
    """
    def check_element(elem_id):
        # AABB test (6 ops)
        in_aabb = ((pos >= aabb_min[elem_id]) & (pos <= aabb_max[elem_id])).all()

        # Early return if outside AABB
        if not in_aabb:
            return False

        # Full test only if AABB passes
        return point_in_tet_gpu_hybrid(
            pos, elem_id, connectivity, node_positions,
            tet_type, right_angle_vertex
        )

    # Vectorized over candidates
    for elem_id in candidate_elements:
        if check_element(elem_id):
            return elem_id

    return -1  # Not found
```

**Expected speedup:**
- L2 search: **5-10× faster** (90% rejection by AABB)
- Combined with Phase 1: **Overall ~5× RK4 speedup**

---

### Phase 3: Optimize L1 Neighbor Search (MEDIUM PRIORITY)

**Current issue:** L1 costs 34% performance but provides 23% better retention

#### 3.1 Problem Analysis

**Current L1 (3 adaptive hops):**
- Checks up to 6 hops × 4 neighbors = 24 elements per particle
- With 5 RK4 stages: 5 × 24 = **120 point-in-tet calls per particle**
- Total: 225K particles × 120 = **27M point-in-tet calls per timestep**

**Why L1 is expensive:**
1. Too many hops (3-6 adaptive)
2. Random memory access (neighbor IDs scattered)
3. Point-in-tet is expensive (145 FLOPs currently)

#### 3.2 Optimization Strategies

**Option A: Reduce L1 Hops (After Phase 1+2)**

With faster point-in-tet (12 ops) and AABB, we can afford more checks:
- Increase L1 hops from 3 to 5 or 6
- Total checks: 5-6 hops × 4 neighbors = 20-24 elements
- Cost: 20 × 12 ops = 240 ops (vs 24 × 145 = 3,480 ops currently)
- **Expected:** Better retention (95-98%) with negligible cost

**Option B: Smart Neighbor Ordering**

Order neighbors by distance/velocity direction:
```python
def order_neighbors_by_velocity(elem_id, velocity_direction):
    """
    Order 4 neighbors by alignment with particle velocity.

    Check neighbors in order of decreasing dot(neighbor_centroid, velocity).
    """
    neighbors = neighbor_list[elem_id]

    # Compute neighbor centroids
    centroids = compute_centroids(neighbors)

    # Sort by dot product with velocity
    scores = jnp.dot(centroids, velocity_direction)
    sorted_neighbors = neighbors[jnp.argsort(-scores)]

    return sorted_neighbors
```

**Expected:** 50% fewer neighbor checks (particle more likely in first 1-2 neighbors)

**Option C: Adaptive L1 Based on Element Size**

For coarse elements: Use L1 (neighbors are far)
For refined elements: Skip L1, go directly to L2 (neighbors are close, unlikely to help)

```python
def should_use_l1(elem_id, element_volumes):
    """Skip L1 for very small (refined) elements."""
    volume = element_volumes[elem_id]
    threshold = 1e-11  # Skip L1 if volume < 1e-11
    return volume >= threshold
```

**Expected:** 30-50% reduction in L1 usage

#### 3.3 Recommended: Combine All Three

```python
def search_l0_l1_l2_optimized(pos, elem_id, velocity_direction):
    """Optimized 3-tier search."""
    # L0: Check cached element (12 ops with axis-aligned)
    if point_in_tet_hybrid(pos, elem_id, ...):
        return elem_id

    # Decide if L1 is worth it
    if should_use_l1(elem_id, element_volumes):
        # L1: Check neighbors (ordered by velocity)
        neighbors = order_neighbors_by_velocity(elem_id, velocity_direction)
        for hop in range(6):  # Increased from 3 to 6 hops
            for neighbor_id in neighbors:
                if point_in_tet_hybrid(pos, neighbor_id, ...):
                    return neighbor_id

    # L2: Global Morton search with AABB filter
    return search_l2_with_aabb_filter(pos, ...)
```

**Expected:**
- Better retention: 93.57% → 95-98%
- Faster: 19,357 p/s → 80,000-100,000 p/s (4-5× overall)

---

## Updated Performance Projections

### Conservative Estimates

| Phase | Optimization | Point-in-Tet FLOPs | Overall Speedup | Throughput | Step Time |
|-------|-------------|-------------------|----------------|------------|-----------|
| **Baseline** | Current | 145 | 1.0× | 19,357 p/s | 11,623 ms |
| **Phase 1** | Axis-aligned | 12 | 3.5× | 67,750 p/s | 3,321 ms |
| **Phase 2** | +AABB filter | 6 (avg) | 4.8× | 92,900 p/s | 2,421 ms |
| **Phase 3** | +L1 optimization | 6 (avg) | 5.5× | 106,500 p/s | 2,113 ms |

**Key assumptions:**
- Phase 1: 100% of tets are axis-aligned (12× faster point-in-tet)
- Point-in-tet is 60% of RK4 runtime
- Phase 2: AABB rejects 90% of L2 candidates
- Phase 3: Smart L1 reduces checks by 30%

### Optimistic Estimates

If point-in-tet is 75% of runtime and optimizations are more effective:
- **Phase 1+2+3 combined:** 7-8× overall speedup
- **Throughput:** 135,000-155,000 p/s
- **Step time:** ~1,500-1,700 ms
- **Retention:** 95-98% (vs 93.57% currently)

---

## Implementation Timeline

### Week 1: Phase 1 (Axis-Aligned Point-in-Tet)

**Day 1-2: Precomputation**
- Implement `precompute_tet_classification()`
- Test on sample mesh (10K elements)
- Verify: 95-100% of tets classified as axis-aligned

**Day 3-4: Fast Path Implementation**
- Implement `point_in_tet_axis_aligned()`
- Unit tests: Compare with current method on 1M random queries
- Target: 100% agreement, 10-12× speedup

**Day 5: Integration**
- Add hybrid `point_in_tet_gpu_hybrid()`
- Update search functions to use hybrid version
- Run production script (single timestep) for validation

**Day 6-7: Validation & Benchmarking**
- Run 10-100 timesteps
- Verify: Retention ≥ 93.5%
- Measure: Throughput improvement (target: 3-4×)
- **Success criteria:** 60,000-80,000 p/s

### Week 2: Phase 2 (AABB Early-Out)

**Day 1-2: Precomputation**
- Implement `compute_element_aabbs()`
- Store AABBs (84 MB)
- Verify bounds are tight

**Day 3-4: L2 Integration**
- Modify `search_l2_global()` to use AABB filter
- Test rejection rate (target: 85-95%)

**Day 5: Production Testing**
- Run 100 timesteps
- Measure: Throughput (target: 90,000-110,000 p/s)
- Verify: Retention ≥ 93%

### Week 3: Phase 3 (L1 Optimization)

**Day 1-2: Smart Neighbor Ordering**
- Implement velocity-based neighbor sorting
- Test on representative particles

**Day 3-4: Adaptive L1**
- Implement element-size-based L1 skipping
- Tune threshold

**Day 5-7: Integration & Final Validation**
- Combined testing with all optimizations
- Full production run (2,500 timesteps)
- **Target:** 100,000+ p/s, 95%+ retention

---

## Risk Mitigation

### Risk 1: Not All Tets Are Axis-Aligned

**Likelihood:** Low (mesh analysis confirms rectilinear structure)
**Impact:** High (would invalidate Phase 1 optimization)

**Mitigation:**
1. Run precomputation diagnostic first
2. If <80% are axis-aligned, fall back to Skala's cross-product method
3. Hybrid approach still works (fast path for axis-aligned, slow path for others)

### Risk 2: Memory Budget

**Concern:** AABBs add 84 MB, total mesh data ~220 MB

**Mitigation:**
- Current usage: ~500 MB with velocity cache
- Remaining: 3.2 GB for particles
- **Acceptable** for 100K-200K particles

### Risk 3: Numerical Stability

**Concern:** Axis-aligned method uses direct division (pos[i] / edge[i])
**Impact:** May be less stable for degenerate elements

**Mitigation:**
1. Keep relative degeneracy check from current implementation
2. Fall back to general method for degenerate tets
3. Tolerance: -1e-6 (same as current)

---

## Why Skala's Method is NOT Needed

**Original plan suggested Skala (2014) cross-product method as Phase 1.**

**Why this was wrong for your mesh:**
1. **Skala:** 48 FLOPs (vs 145 current) → 3× speedup
2. **Axis-aligned:** 12 FLOPs → **12× speedup**
3. **Your mesh:** 100% axis-aligned → axis-aligned method is **4× better than Skala**

**Conclusion:** Skip Skala entirely, go directly to axis-aligned optimization.

**When Skala would be useful:**
- If mesh had general (non-axis-aligned) tetrahedra
- If you process multiple mesh types (some general, some structured)
- For research/comparison purposes

---

## Comparison: Original vs Revised Plan

| Aspect | Original Plan | Revised Plan | Rationale |
|--------|--------------|--------------|-----------|
| **Phase 1** | Skala cross-product (3×) | Axis-aligned (12×) | Your mesh is 100% axis-aligned |
| **Phase 2** | AABB early-out | AABB early-out | Same |
| **Phase 3** | Research right-angled | L1 optimization | Structure confirmed, focus on L1 |
| **Expected speedup** | 2-3× | 5-8× | Exploit known mesh structure |
| **Implementation risk** | Low | Low | Both are proven methods |
| **Generality** | Works for any mesh | Specialized for your mesh | Trade-off: speed vs generality |

---

## Validation Checklist

### After Phase 1

- [ ] Precomputation: 95-100% of tets classified as axis-aligned
- [ ] Unit test: 100% agreement with current method (1M queries)
- [ ] Microbenchmark: 10-12× speedup for point-in-tet
- [ ] Production: Throughput ≥ 60,000 p/s
- [ ] Production: Retention ≥ 93.5%
- [ ] Visual: Trajectories still physically correct

### After Phase 2

- [ ] AABB rejection rate: 85-95% for L2 candidates
- [ ] Production: Throughput ≥ 90,000 p/s
- [ ] Production: Retention ≥ 93%
- [ ] Memory: Total GPU usage ≤ 600 MB (static + particles)

### After Phase 3

- [ ] L1 neighbor ordering: 30-50% fewer checks
- [ ] Adaptive L1: 20-40% skip rate for refined elements
- [ ] Production: Throughput ≥ 100,000 p/s
- [ ] Production: Retention ≥ 95%
- [ ] Full run: 2,500 timesteps complete successfully

---

## Conclusion

**Your mesh is PERFECT for optimization** - 100% axis-aligned tetrahedra is extremely rare!

**Recommended priority:**
1. ✅ **Phase 1 (Week 1):** Axis-aligned point-in-tet → **3-4× speedup**
2. ✅ **Phase 2 (Week 2):** AABB early-out → **Additional 1.3× (cumulative 4.8×)**
3. ✅ **Phase 3 (Week 3):** L1 optimization → **Additional 1.15× (cumulative 5.5×)**

**Skip Skala's cross-product method entirely** - you have a better specialized method.

**Expected final performance:**
- **Throughput:** 100,000-150,000 p/s (5-8× improvement)
- **Step time:** 1,500-2,300 ms (vs 11,600 ms currently)
- **Retention:** 95-98% (vs 93.57% currently)

---

## Next Steps

1. **Review this revised plan** - confirm it matches your mesh structure
2. **Start Phase 1** - implement axis-aligned point-in-tet this week
3. **Benchmark early** - validate 10-12× point-in-tet speedup
4. **Iterate** - adjust based on actual performance measurements

**Ready to proceed with implementation?**
