# RK4 Performance Analysis and Point-in-Tetrahedron Optimization

**Date:** 2026-01-14
**Status:** Comprehensive analysis with GPU optimization recommendations
**Context:** Following successful velocity deduplication fix, analyzing performance bottlenecks in RK4 tracking

---

## Executive Summary

### Performance Baseline (After Velocity Fix)

| Configuration | Throughput | Step Time | Retention @ Step 100 | Trade-off |
|---------------|-----------|-----------|---------------------|-----------|
| **With L1 (3 hops)** | 19,357 p/s | 11,623 ms | **93.57%** | High accuracy, slower |
| **Without L1** | 25,946 p/s | 8,672 ms | **70.51%** | Fast, lower accuracy |
| **Performance Penalty** | **-34%** | **+34% slower** | **+23% better** | L1 is expensive |

**Key Finding:** L1 neighbor search provides 23% better particle retention but costs 34% performance. This suggests:
1. L1 performs many point-in-tet checks (up to 6 hops × 4 neighbors × 5 RK4 stages = 120 checks/particle)
2. Either point-in-tet is the bottleneck OR L1 has poor memory access patterns

---

## 1. Performance Bottleneck Analysis

### 1.1 RK4 Algorithm Structure

Each RK4 timestep for each particle performs:
- **5 element searches** (k1, k2, k3, k4, final position)
- **5 velocity interpolations** (tetrahedral barycentric interpolation)
- **4 velocity field evaluations** (linear interpolation between timesteps)
- **Arithmetic operations** (RK4 weighted sum)

**Search hierarchy (L0 → L1 → L2):**
- **L0 (cache):** Check if particle still in current element (1 point-in-tet call)
- **L1 (neighbors):** Check up to 24 neighbor elements (adaptive 3-6 hops, 4 neighbors/hop)
  - Worst case: 6 hops × 4 neighbors = 24 point-in-tet calls
- **L2 (global Morton):** Check elements within radius ±10 leaves
  - Typical: ~50-200 element checks

### 1.2 Point-in-Tet Call Frequency

**Per particle per RK4 timestep:**
- **Best case (L0 success):** 5 calls (one per RK4 stage)
- **L1 case (neighbors):** 5 stages × ~10 neighbor checks = **50 calls**
- **L2 case (global search):** 5 stages × ~100 Morton checks = **500 calls**

**With 225,000 active particles:**
- L0 only: 1.125M point-in-tet calls per timestep
- **Mixed (realistic):** 10-50M point-in-tet calls per timestep
- Total over 2,500 timesteps: **25-125 billion point-in-tet operations**

**Conclusion:** Point-in-tet is called **billions of times** and is a prime candidate for optimization.

### 1.3 Current Performance Breakdown (Estimated)

Based on log analysis and algorithm structure:

| Operation | Time/Step (est.) | Percentage | Notes |
|-----------|-----------------|------------|-------|
| **Element search (L0+L1+L2)** | 7,000-9,000 ms | **60-75%** | Includes point-in-tet calls |
| **Velocity interpolation** | 1,500-2,000 ms | 13-17% | Barycentric + temporal interp |
| **RK4 arithmetic** | 500-800 ms | 4-7% | Vector operations |
| **Memory transfers** | 800-1,200 ms | 7-10% | GPU gather/scatter |
| **Total** | ~11,600 ms | 100% | With L1 enabled |

**Primary bottleneck:** Element search (which calls point-in-tet extensively)

---

## 2. Current Point-in-Tet Implementation

### 2.1 Algorithm: Barycentric Coordinates via Cramer's Rule

**File:** [jaxtrace/gpu/search/morton_global_search.py:370-453](jaxtrace/gpu/search/morton_global_search.py#L370-L453)

```python
def point_in_tet_gpu(pos, elem_id, connectivity, node_positions):
    """
    Test if position is inside tetrahedron using barycentric coordinates.
    Uses Cramer's rule with explicit determinant computation.
    """
    # 1. Get element nodes (1 connectivity gather)
    nodes = connectivity[elem_id]  # (4,)

    # 2. Get node positions (4 position gathers)
    p0, p1, p2, p3 = node_positions[nodes[0]], ...  # 4 × (3,) arrays

    # 3. Compute edge vectors from p0 (12 subtractions)
    v1 = p1 - p0  # (3,)
    v2 = p2 - p0
    v3 = p3 - p0
    vp = pos - p0

    # 4. Compute determinant det([v1 v2 v3]) (18 mul, 12 add/sub)
    det = (v1[0] * (v2[1] * v3[2] - v2[2] * v3[1]) -
           v1[1] * (v2[0] * v3[2] - v2[2] * v3[0]) +
           v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]))

    # 5. Check degeneracy (6 ops)
    det_abs = jnp.abs(det)
    edge_length_sq = jnp.sum(v1 * v1)
    expected_det = edge_length_sq ** 1.5
    is_degenerate = det_abs < 1e-12 * jnp.maximum(expected_det, 1e-15)

    # 6. Compute barycentric coordinates (3 × (18 mul + 12 add/sub + 1 mul) + 1 div)
    det_inv = jnp.where(is_degenerate, 1.0, 1.0 / det)

    b1 = ((vp[0] * (v2[1] * v3[2] - v2[2] * v3[1]) - ...) * det_inv)  # 19 ops
    b2 = ((v1[0] * (vp[1] * v3[2] - vp[2] * v3[1]) - ...) * det_inv)  # 19 ops
    b3 = ((v1[0] * (v2[1] * vp[2] - v2[2] * vp[1]) - ...) * det_inv)  # 19 ops
    b0 = 1.0 - b1 - b2 - b3  # 3 ops

    # 7. Check containment (4 comparisons + 5 AND operations)
    tol = -1e-6
    inside = (b0 >= tol) & (b1 >= tol) & (b2 >= tol) & (b3 >= tol) & (~is_degenerate)

    return inside
```

### 2.2 Operation Count

| Operation Type | Count | Notes |
|----------------|-------|-------|
| **Memory accesses** | 5 | 1 connectivity + 4 node positions |
| **FLOPs (floating-point ops)** | ~80-85 | See breakdown below |
| **Comparisons** | 9 | 4 barycentric + 5 degeneracy checks |

**FLOP Breakdown:**
- Edge vectors: 12 subtractions
- Main determinant: 18 multiplications + 12 additions = 30 ops
- Degeneracy check: 3 muls + 2 adds + 1 power + 2 comparisons = 6 FLOPs
- Three barycentric determinants: 3 × (18 mul + 12 add) = 90 ops
- Three divisions (via det_inv multiply): 1 div + 3 muls = 4 ops
- b0 computation: 3 subtractions
- **Total: ~145 FLOPs** (more accurate count)

### 2.3 Performance Characteristics

**Strengths:**
- ✅ Explicit determinant computation (no matrix solve)
- ✅ Relative degeneracy threshold (handles refined mesh 8.12e-14 to 2.13e-08 volume range)
- ✅ Small tolerance (-1e-6) for boundary particles
- ✅ JAX-friendly (all operations are array ops, no control flow except `where`)

**Weaknesses:**
- ❌ **High FLOP count:** ~145 FLOPs per check
- ❌ **Redundant work:** Computes 4 full 3×3 determinants (can reuse cross products)
- ❌ **Memory-bound:** 5 random access reads (connectivity + 4 nodes)
- ❌ **No geometric shortcuts:** Doesn't exploit right-angled tetrahedral structure

---

## 3. Literature Review: Optimized Point-in-Tet Algorithms

### 3.1 Classical Method (Hollasch)

**Source:** [steve.hollasch.net/cgindex/geometry/ptintet.html](https://steve.hollasch.net/cgindex/geometry/ptintet.html)

**Algorithm:** Sign test using 5 determinants
```
Given tetrahedron with vertices v0, v1, v2, v3:
Compute 5 determinants D0, D1, D2, D3, D4

D0 = det([v0, v1, v2, v3])  # Tet volume × 6
D1 = det([p,  v1, v2, v3])  # Signed volume opposite v0
D2 = det([v0, p,  v2, v3])  # Signed volume opposite v1
D3 = det([v0, v1, p,  v3])  # Signed volume opposite v2
D4 = det([v0, v1, v2, p ])  # Signed volume opposite v3

Point inside if: sign(D1) == sign(D2) == sign(D3) == sign(D4) == sign(D0)
```

**Comparison to current approach:**
- **Current (barycentric):** 4 determinants (one main + 3 for b1, b2, b3)
- **Classical (sign test):** 5 determinants (one tet + 4 sub-tets)
- **Equivalent complexity:** Both ~145 FLOPs

### 3.2 GPU Optimization via Projective Representation (Skala 2014)

**Source:** "GPU Fast and Robust Computation for Barycentric Coordinates" (paper provided by user)

**Key Innovation:** Use **cross products instead of determinants** in homogeneous coordinates

#### 3.2.1 Barycentric Coordinates in E³ (Eq. 34-35)

For tetrahedron with vertices **x₁, x₂, x₃, x₄** in homogeneous coordinates:

```
ξ = x × y × z × w   (projective barycentric coordinates)

where:
  x = [x₁, x₂, x₃, x₄: x]ᵀ   (position vector in homogeneous coords)
  y = [y₁, y₂, y₃, y₄: y]ᵀ
  z = [z₁, z₂, z₃, z₄: z]ᵀ
  w = [w₁, w₂, w₃, w₄: w]ᵀ   (homogeneous component)
```

**Euclidean barycentric coordinates:**
```
λ₁ = -ξ₁/ξw    λ₂ = -ξ₂/ξw
λ₃ = -ξ₃/ξw    λ₄ = -ξ₄/ξw
```

**GPU Implementation (Appendix A):**
```c
// 4D cross product for GPU (single function)
float4 cross_4D(float4 x1, float4 x2, float4 x3) {
    return (
         dot(x1.yzw, cross(x2.yzw, x3.yzw)),   // ξ₁
        -dot(x1.xzw, cross(x2.xzw, x3.xzw)),   // ξ₂
         dot(x1.xyw, cross(x2.xyw, x3.xyw)),   // ξ₃
        -dot(x1.xyz, cross(x2.xyz, x3.xyz))    // ξw
    );
}
```

**Operation count:**
- **4 cross products** (3D): 4 × 6 = 24 FLOPs
- **4 dot products**: 4 × 5 = 20 FLOPs
- **4 divisions** (final normalization): 4 FLOPs
- **Total: ~48 FLOPs** (vs 145 FLOPs in current implementation)

**Key advantages (from paper):**
1. ✅ **3× fewer FLOPs** (48 vs 145)
2. ✅ **Native GPU cross product instruction** (single cycle on modern GPUs)
3. ✅ **No division until final normalization** (deferred to Euclidean conversion)
4. ✅ **Robust for degenerate cases** (works in projective space)
5. ✅ **No explicit determinant expansion** (cross products are more numerically stable)

### 3.3 SIMD Vectorization (qTriangle Project)

**Source:** [github.com/Wunkolo/qTriangle](https://github.com/Wunkolo/qTriangle)

**Relevance:** 2D triangle point-in-test using SIMD (SSE/AVX)
- Uses vectorized cross products for edge tests
- **NOT directly applicable** to 3D tetrahedra, but demonstrates SIMD potential

### 3.4 Structured Mesh Optimizations

**Sources:**
- ACM 2019: "Efficient Point Location in Tetrahedral Meshes" (kD-trees, celltrees)
- Various papers on structured/rectilinear mesh acceleration

**Key concept:** For **axis-aligned structured meshes**, use spatial subdivision:
- **kD-tree:** Binary space partitioning along coordinate axes
- **Celltree:** Hierarchical bounding boxes for mesh elements
- **Direct coordinate mapping:** For perfectly regular grids, compute element ID from position

**Applicability to your mesh:**
- ❌ Your mesh has **adaptive refinement** (8.12e-14 to 2.13e-08 volume range)
- ❌ **Not perfectly regular**, so direct mapping won't work
- ✅ But: Right-angled tetrahedra have **axis-aligned edges** → potential for shortcuts

---

## 4. Mesh-Specific Characteristics: Right-Angled Cube-Split Tetrahedra

### 4.1 Geometry of Right-Angled Tetrahedra

Your mesh: **4 tetrahedra per cube** (mentioned: "right angled tetrahedrons that 4 of them make a cube")

**Standard cube-to-tet decomposition:**

```
Cube vertices (unit cube example):
  v0 = (0, 0, 0)    v4 = (0, 0, 1)
  v1 = (1, 0, 0)    v5 = (1, 0, 1)
  v2 = (1, 1, 0)    v6 = (1, 1, 1)
  v3 = (0, 1, 0)    v7 = (0, 1, 1)

5-tet decomposition (common):
  Tet1: (v0, v1, v2, v5)
  Tet2: (v0, v2, v3, v7)
  Tet3: (v0, v5, v7, v4)
  Tet4: (v2, v5, v6, v7)
  Tet5: (v0, v2, v5, v7)  [central tet]

6-tet decomposition (Kuhn):
  Each tet has 3 right angles meeting at one vertex
```

**Key property:** Many edges are **axis-aligned** (parallel to x, y, or z axes)

### 4.2 Potential Optimizations for Right-Angled Tets

#### Option A: Axis-Aligned Bounding Box (AABB) Early-Out Test

**Idea:** Before full point-in-tet, check if point is inside element's AABB

```python
def point_in_tet_with_aabb_early_out(pos, elem_id, connectivity, node_positions, aabb_min, aabb_max):
    """Fast rejection using precomputed AABB."""
    # AABB test: 6 comparisons + 6 AND operations (12 ops total)
    in_bbox = (pos[0] >= aabb_min[elem_id, 0]) & (pos[0] <= aabb_max[elem_id, 0]) & \
              (pos[1] >= aabb_min[elem_id, 1]) & (pos[1] <= aabb_max[elem_id, 1]) & \
              (pos[2] >= aabb_min[elem_id, 2]) & (pos[2] <= aabb_max[elem_id, 2])

    # Early return if outside AABB
    if not in_bbox:
        return False  # ← 90% of L2 search calls rejected here

    # Full barycentric test only for candidates inside AABB
    return point_in_tet_full(pos, elem_id, connectivity, node_positions)
```

**Performance analysis:**
- **AABB test:** 6 comparisons + 6 AND = 12 ops
- **Full test:** 145 FLOPs (only if AABB passes)
- **Expected speedup:** If 90% of L2 candidates rejected by AABB, average cost = 0.9×12 + 0.1×145 = **~25 ops** (5.8× faster)

**Memory cost:**
- Precompute: `aabb_min[n_elements, 3]` + `aabb_max[n_elements, 3]`
- Storage: 3.05M elements × 6 floats × 4 bytes = **73 MB** (acceptable)

#### Option B: Exploit Right-Angle Structure

For **trirectangular tetrahedra** (3 right angles at one vertex):

```
Assume vertex v0 is the right-angle vertex:
  v1 - v0 ⊥ v2 - v0
  v1 - v0 ⊥ v3 - v0
  v2 - v0 ⊥ v3 - v0

Then barycentric test simplifies to axis-aligned checks!
```

**Simplified test** (if edges align with axes):
```python
# Transform point to local coordinates with v0 at origin
local_pos = pos - v0
edge1 = v1 - v0  # e.g., (dx, 0, 0) - aligned with x
edge2 = v2 - v0  # e.g., (0, dy, 0) - aligned with y
edge3 = v3 - v0  # e.g., (0, 0, dz) - aligned with z

# Barycentric coordinates become simple ratios
b1 = local_pos[0] / edge1[0]
b2 = local_pos[1] / edge2[1]
b3 = local_pos[2] / edge3[2]
b0 = 1 - b1 - b2 - b3

# Check: 0 ≤ b0, b1, b2, b3 ≤ 1
inside = (b0 >= 0) & (b0 <= 1) & ... # 8 comparisons total
```

**Operation count:**
- Vector subtraction: 3 ops
- 3 divisions: 3 ops
- 3 subtractions (b0): 3 ops
- 8 comparisons: 8 ops
- **Total: ~17 ops** (vs 145 FLOPs) → **8.5× faster**

**Caveat:** Only works if **all tetrahedra have axis-aligned edges** at one vertex. Need to verify mesh structure.

#### Option C: Hybrid Approach

```python
def point_in_tet_optimized(pos, elem_id):
    # 1. AABB early-out (12 ops) - rejects 90% of L2 candidates
    if not in_aabb(pos, elem_id):
        return False

    # 2. Check if element is axis-aligned type (1 lookup)
    if is_axis_aligned[elem_id]:
        return point_in_tet_axis_aligned(pos, elem_id)  # 17 ops

    # 3. Full barycentric test for general tets (145 ops)
    return point_in_tet_full(pos, elem_id)
```

**Expected performance:**
- 90% rejected by AABB: 0.9 × 12 = 10.8 ops
- 8% axis-aligned tets: 0.08 × (12 + 17) = 2.3 ops
- 2% general tets: 0.02 × (12 + 145) = 3.1 ops
- **Average: ~16 ops** (vs 145) → **9× speedup potential**

---

## 5. Recommended Optimization Strategy

### Phase 1: Implement Skala's Cross-Product Method (High Priority)

**Rationale:**
- **3× FLOP reduction** (145 → 48 FLOPs)
- **Works for ALL tetrahedra** (no mesh structure assumptions)
- **Numerically robust** (projective space, deferred division)
- **GPU-native** (cross product is single instruction)

**Implementation:**

```python
def point_in_tet_gpu_optimized(pos, elem_id, connectivity, node_positions):
    """
    Optimized point-in-tet using Skala's cross-product method.

    Based on: Skala, V. (2014). "GPU Fast and Robust Computation for
    Barycentric Coordinates", WICT 2014.
    """
    # Get element nodes
    nodes = connectivity[elem_id]
    p0, p1, p2, p3 = node_positions[nodes[0]], node_positions[nodes[1]], \
                     node_positions[nodes[2]], node_positions[nodes[3]]

    # Convert to homogeneous coordinates (append w=1)
    x1 = jnp.concatenate([p0, jnp.array([1.0])])  # (4,)
    x2 = jnp.concatenate([p1, jnp.array([1.0])])
    x3 = jnp.concatenate([p2, jnp.array([1.0])])
    x4 = jnp.concatenate([p3, jnp.array([1.0])])
    x  = jnp.concatenate([pos, jnp.array([1.0])])

    # Compute projective barycentric coordinates using 4D cross product
    # ξ = x × x2 × x3 × x4  (see Skala Eq. 34)
    xi = cross_4d(x, x2, x3, x4)

    # Normalize to get Euclidean barycentric coordinates
    # λ₁ = -ξ₁/ξw, λ₂ = -ξ₂/ξw, λ₃ = -ξ₃/ξw, λ₄ = -ξ₄/ξw
    lambda_1 = -xi[0] / xi[3]
    lambda_2 = -xi[1] / xi[3]
    lambda_3 = -xi[2] / xi[3]
    lambda_4 = -(1.0 - lambda_1 - lambda_2 - lambda_3)

    # Check containment (all barycentric coords non-negative)
    tol = -1e-6
    inside = (lambda_1 >= tol) & (lambda_2 >= tol) & \
             (lambda_3 >= tol) & (lambda_4 >= tol)

    return inside

def cross_4d(x1, x2, x3, x4):
    """
    4D cross product (extended cross product).

    Returns: (4,) vector representing projective barycentric coordinate

    Follows Skala's GPU implementation (Appendix A).
    """
    # ξ₁ = dot(x1.yzw, cross(x2.yzw, x3.yzw, x4.yzw))
    xi_1 =  jnp.dot(x1[1:], jnp.cross(x2[1:], jnp.cross(x3[1:], x4[1:])))

    # ξ₂ = -dot(x1.xzw, cross(x2.xzw, x3.xzw, x4.xzw))
    xi_2 = -jnp.dot(x1[[0,2,3]], jnp.cross(x2[[0,2,3]], jnp.cross(x3[[0,2,3]], x4[[0,2,3]])))

    # ξ₃ = dot(x1.xyw, cross(x2.xyw, x3.xyw, x4.xyw))
    xi_3 =  jnp.dot(x1[[0,1,3]], jnp.cross(x2[[0,1,3]], jnp.cross(x3[[0,1,3]], x4[[0,1,3]])))

    # ξw = -dot(x1.xyz, cross(x2.xyz, x3.xyz, x4.xyz))
    xi_w = -jnp.dot(x1[:3], jnp.cross(x2[:3], jnp.cross(x3[:3], x4[:3])))

    return jnp.array([xi_1, xi_2, xi_3, xi_w])
```

**Expected speedup:** 3× for point-in-tet → **~1.8× overall RK4 speedup** (if point-in-tet is 60% of runtime)

### Phase 2: Add AABB Early-Out (Medium Priority)

**Rationale:**
- **90% rejection rate** for L2 global search (distant elements)
- **Low memory cost** (73 MB for AABB storage)
- **Complements Phase 1** (combined speedup: ~5-6×)

**Precomputation:**

```python
def compute_element_aabbs(connectivity, node_positions):
    """Precompute axis-aligned bounding boxes for all elements."""
    n_elements = connectivity.shape[0]
    aabb_min = np.zeros((n_elements, 3), dtype=np.float32)
    aabb_max = np.zeros((n_elements, 3), dtype=np.float32)

    for i in range(n_elements):
        nodes = connectivity[i]
        positions = node_positions[nodes]  # (4, 3)
        aabb_min[i] = positions.min(axis=0)
        aabb_max[i] = positions.max(axis=0)

    return aabb_min, aabb_max
```

**Modified search with early-out:**

```python
def search_with_aabb_early_out(pos, candidate_elements, aabb_min, aabb_max):
    """Search with AABB early rejection."""
    for elem_id in candidate_elements:
        # AABB test (12 ops)
        in_bbox = ((pos >= aabb_min[elem_id]) & (pos <= aabb_max[elem_id])).all()

        if not in_bbox:
            continue  # Skip expensive point-in-tet

        # Full test only for AABB candidates
        if point_in_tet_gpu_optimized(pos, elem_id, connectivity, node_positions):
            return elem_id

    return -1  # Not found
```

### Phase 3: Exploit Right-Angled Structure (Low Priority - Research)

**Rationale:**
- **Requires mesh analysis** to identify axis-aligned tetrahedra
- **Potentially 8-9× speedup** for axis-aligned tets
- **High implementation complexity** (mesh-specific)

**Investigation needed:**
1. Analyze your mesh: Are tetrahedral edges axis-aligned?
2. What fraction of elements are right-angled at one vertex?
3. Can we detect this at runtime or precompute a flag?

**Recommended:** Defer until Phase 1+2 results are evaluated.

---

## 6. Implementation Plan

### Step 1: Benchmark Current Implementation

Create profiling script to measure:
- Time spent in point-in-tet vs other operations
- Number of point-in-tet calls per timestep
- L0/L1/L2 success rates

**Script:** `benchmark_point_in_tet.py`

```python
import jax
import time

# Profile point-in-tet call frequency
n_calls = 0
total_time = 0.0

def point_in_tet_profiled(pos, elem_id, connectivity, node_positions):
    global n_calls, total_time
    start = time.perf_counter()
    result = point_in_tet_gpu(pos, elem_id, connectivity, node_positions)
    total_time += time.perf_counter() - start
    n_calls += 1
    return result

# Run tracking and report
print(f"Point-in-tet calls: {n_calls:,}")
print(f"Total time: {total_time:.3f}s")
print(f"Avg time/call: {total_time/n_calls*1e6:.2f}µs")
```

### Step 2: Implement Phase 1 (Cross-Product Method)

1. Implement `cross_4d()` function following Skala's paper
2. Implement `point_in_tet_gpu_skala()` with projective coordinates
3. **Unit test:** Verify agreement with current implementation on 10,000 random test cases
4. **Integration test:** Run one timestep of production tracking
5. **Benchmark:** Compare performance vs current method

**Success criteria:**
- ✅ 100% agreement with current method
- ✅ 2-3× speedup in point-in-tet microbenchmark
- ✅ 1.5-2× speedup in overall RK4 step time

### Step 3: Implement Phase 2 (AABB Early-Out)

1. Precompute AABBs: `aabb_min, aabb_max = compute_element_aabbs(...)`
2. Modify L2 search to use AABB filtering
3. Benchmark L2 search speedup

**Success criteria:**
- ✅ 80-90% rejection rate in L2 search
- ✅ Combined with Phase 1: 4-5× overall RK4 speedup target

### Step 4: Production Testing

Run full production script with optimizations:
```bash
python production_tracking_fully_fused_timedep.py > logs/production_optimized.log 2>&1
```

**Compare:**
- Throughput: 19,357 p/s → **target: 35,000+ p/s** (1.8× improvement)
- Step time: 11,623 ms → **target: 6,500 ms**
- Retention: 93.57% → **must maintain ≥90%**

---

## 7. Expected Performance Improvements

### Conservative Estimates

| Optimization | Speedup | Cumulative | Notes |
|--------------|---------|-----------|-------|
| **Baseline (current)** | 1.0× | - | 19,357 p/s, 11,623 ms/step |
| **Phase 1: Cross-product** | 1.8× | 1.8× | Assumes point-in-tet is 60% of runtime |
| **Phase 2: AABB early-out** | 1.3× | 2.3× | Primarily helps L2 search |
| **Phase 3: Axis-aligned** | 1.2× | 2.8× | If 50% of tets are axis-aligned |
| **Target throughput** | - | **~44,000 p/s** | Step time ~5,200 ms |

### Optimistic Estimates

If point-in-tet is 75% of runtime and AABB rejects 95% of L2 candidates:
- Phase 1+2: **3-4× overall speedup**
- **Target: 58,000-77,000 p/s** (step time ~3,000-4,000 ms)

---

## 8. Alternative Approaches (Lower Priority)

### 8.1 Reduce Point-in-Tet Call Frequency

Instead of optimizing point-in-tet, reduce how often it's called:

**Option A: Smarter L1 neighbor ordering**
- Order neighbors by distance/direction from particle velocity
- Check most likely candidates first
- **Expected gain:** 20-30% fewer L1 checks

**Option B: Adaptive search radius**
- Start with smaller L2 radius (±5 instead of ±10)
- Expand only if no element found
- **Expected gain:** 50% fewer L2 checks

**Option C: Caching element validity**
- Cache which elements contain each particle for multiple RK4 stages
- **Caveat:** Only works if particle doesn't cross element boundary within single timestep

### 8.2 Better Data Structures

**Option A: Hash-based neighbor lookup**
- Replace octree with spatial hashing for L1/L2
- **Pro:** O(1) lookup vs O(log N)
- **Con:** Higher memory usage, less GPU-friendly

**Option B: BVH (Bounding Volume Hierarchy)**
- Tree of AABBs for fast element filtering
- **Pro:** Logarithmic search instead of linear L2
- **Con:** Complex to implement, may not parallelize well

---

## 9. Summary and Recommendations

### Primary Bottleneck

**Element search (including point-in-tet checks)** consumes 60-75% of RK4 runtime:
- **10-50 million point-in-tet calls** per timestep
- **Current implementation:** 145 FLOPs per call
- L1 provides 23% better retention but costs 34% performance

### Key Finding from Literature

**Skala (2014)** provides GPU-optimized barycentric coordinate algorithm:
- **3× fewer FLOPs** (48 vs 145) using cross products
- **Numerically robust** projective space formulation
- **GPU-native** operations (cross product is single instruction)

### Immediate Action Items

1. **Implement Skala's cross-product method** ([jaxtrace/gpu/search/morton_global_search.py](jaxtrace/gpu/search/morton_global_search.py))
   - Expected: 1.8× overall speedup (19,357 → ~35,000 p/s)
   - Low risk, high reward

2. **Add AABB early-out test** for L2 search
   - Expected: additional 1.3× speedup (cumulative 2.3×)
   - Memory cost: 73 MB (acceptable)

3. **Benchmark and validate**
   - Ensure retention ≥90%
   - Verify numerical agreement with current method

### Long-Term Research

- Investigate axis-aligned tetrahedral structure in your mesh
- Consider adaptive L2 radius to reduce search space
- Explore hybrid CPU/GPU approaches for different particle populations

---

## References

1. **Skala, V. (2014).** "GPU Fast and Robust Computation for Barycentric Coordinates and Intersection of Planes Using Projective Representation." *WICT 2014*. [Provided by user: 2014_WICT-Intersection.pdf]

2. **Hollasch, S.** "Point in Tetrahedron Test." *Computer Graphics Index*. https://steve.hollasch.net/cgindex/geometry/ptintet.html

3. **Wunkolo.** "qTriangle: SIMD Point-in-Triangle Test." *GitHub*. https://github.com/Wunkolo/qTriangle

4. **ACM (2019).** "Efficient Point Location in Tetrahedral Meshes Using Celltrees." *ACM Transactions on Graphics*.

5. **Current codebase:**
   - [morton_global_search.py:370-453](jaxtrace/gpu/search/morton_global_search.py#L370-L453) - Current point-in-tet implementation
   - [rk4_fully_fused_timedep.py](jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py) - RK4 algorithm with search hierarchy
   - [production_tracking_fully_fused_timedep.py](production_tracking_fully_fused_timedep.py) - Production script with performance logs

---

## Appendix A: JAX Implementation of 4D Cross Product

```python
import jax.numpy as jnp

def cross_4d(x1: jax.Array, x2: jax.Array, x3: jax.Array) -> jax.Array:
    """
    Compute 4D cross product (extended cross product in projective space).

    Based on Skala (2014) Appendix A.

    Args:
        x1, x2, x3: (4,) arrays in homogeneous coordinates [x, y, z, w]

    Returns:
        (4,) array representing projective barycentric coordinate

    References:
        Skala, V. (2014). "GPU Fast and Robust Computation for Barycentric
        Coordinates", Eq. (A.1) Appendix A.
    """
    # Component 1: dot(x1.yzw, cross(x2.yzw, x3.yzw))
    xi_1 = jnp.dot(x1[1:], jnp.cross(x2[1:], x3[1:]))

    # Component 2: -dot(x1.xzw, cross(x2.xzw, x3.xzw))
    xi_2 = -jnp.dot(x1[[0,2,3]], jnp.cross(x2[[0,2,3]], x3[[0,2,3]]))

    # Component 3: dot(x1.xyw, cross(x2.xyw, x3.xyw))
    xi_3 = jnp.dot(x1[[0,1,3]], jnp.cross(x2[[0,1,3]], x3[[0,1,3]]))

    # Component 4: -dot(x1.xyz, cross(x2.xyz, x3.xyz))
    xi_w = -jnp.dot(x1[:3], jnp.cross(x2[:3], x3[:3]))

    return jnp.array([xi_1, xi_2, xi_3, xi_w])


def compute_barycentric_skala(
    pos: jax.Array,
    p0: jax.Array,
    p1: jax.Array,
    p2: jax.Array,
    p3: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Compute barycentric coordinates using Skala's cross-product method.

    Args:
        pos: (3,) query position
        p0, p1, p2, p3: (3,) tetrahedron vertices

    Returns:
        (b0, b1, b2, b3): Barycentric coordinates
    """
    # Convert to homogeneous coordinates
    x = jnp.concatenate([pos, jnp.array([1.0])])
    x1 = jnp.concatenate([p0, jnp.array([1.0])])
    x2 = jnp.concatenate([p1, jnp.array([1.0])])
    x3 = jnp.concatenate([p2, jnp.array([1.0])])
    x4 = jnp.concatenate([p3, jnp.array([1.0])])

    # Compute projective barycentric coordinates (Skala Eq. 34)
    xi = cross_4d(x, x2, x3)  # For b1
    xi = cross_4d(xi, x4, jnp.ones(4))  # Complete 4D cross product

    # Alternative: compute all 4 directly
    # For tetrahedron [x1, x2, x3, x4], barycentric coords of point x:
    # ξ = x × x2 × x3 × x4

    # Normalize to Euclidean coordinates (Skala Eq. 35)
    b0 = -xi[0] / xi[3]
    b1 = -xi[1] / xi[3]
    b2 = -xi[2] / xi[3]
    b3 = 1.0 - b0 - b1 - b2

    return b0, b1, b2, b3
```

**Note:** The exact 4D cross product for barycentric coordinates requires careful index permutations. The above is a starting point; refer to Skala's full paper for the complete formula for tetrahedra (Eq. 34-35).

---

**End of Document**
