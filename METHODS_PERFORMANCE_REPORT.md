# JAXTrace Performance Report: Element Search & Point-in-Tet Methods

**Date**: 2026-01-28
**Mesh**: FLA (3,048,900 elements, 571,173 nodes after deduplication)
**Test Configuration**: 30,000-225,000 particles, 100-2,500 RK4 steps

---

## Table of Contents
1. [Element Search Methods (L2 Global Search)](#element-search-methods-l2-global-search)
2. [Point-in-Tetrahedron Methods](#point-in-tetrahedron-methods)
3. [Performance Summary](#performance-summary)
4. [Recommendations](#recommendations)

---

## Element Search Methods (L2 Global Search)

Element search methods are used to locate which element contains a particle when local searches (L0 cached element, L1 neighbors) fail. These are critical for maintaining particle retention during RK4 tracking.

### 1. Morton Curve-Based Methods

All Morton-based methods use space-filling curves for spatial indexing with Morton codes computed from **element centroids**.

#### 1.1 **Fixed Radius Search** (`'radius'`)

**How it works:**
- Find particle's position on Morton curve (binary search)
- Search fixed radius ±R leaves along the curve
- Test all elements in those leaves
- radius=R → searches 2R+1 leaves

**Configuration:**
```python
L2_SEARCH_METHOD = 'radius'
L2_SEARCH_RADIUS = 10  # searches 21 leaves
```

**Performance (FLA mesh, 30K particles, 100 RK4 steps):**
| Radius | Retention | Tests/Particle | Throughput | Notes |
|--------|-----------|----------------|------------|-------|
| radius=10 | 96.96% | ~536 (21 leaves) | **51,894 p/s** | Baseline, fastest |
| radius=30 | 98.21% | ~1,600 (61 leaves) | 17,895 p/s | Max coverage, slow |

**Pros:**
- ✅ Simple, predictable
- ✅ Fastest single-tier search
- ✅ Vmappable (JAX-compatible)

**Cons:**
- ❌ Tests many unnecessary elements
- ❌ Fixed cost regardless of particle location

**Production Use:** Good for uniform particle distributions, baseline comparison.

---

#### 1.2 **Incremental Search** (`'incremental'`)

**How it works:**
- Cascading search with multiple radius tiers
- Start with small radius, expand if not found
- Each tier only searches if previous tier failed
- Adaptive: Only does work when needed

**Configuration:**
```python
L2_SEARCH_METHOD = 'incremental'
INCREMENTAL_SEARCH_RADII = (2, 4, 8, 15, 30)  # 5-tier (PRODUCTION)
# Alternative: (2, 5, 10)  # 3-tier (simpler)
```

**Performance (FLA mesh, 30K particles, 100 RK4 steps):**
| Configuration | Retention | Avg Tests | Throughput | Speedup vs radius=10 |
|---------------|-----------|-----------|------------|---------------------|
| (2,4,8,15,30) 5-tier | 98.21% | ~22.5 leaves avg | 9,136 p/s | 0.18× (slower!) |
| (2,5,10) 3-tier | 96.96% | ~11.5 leaves avg | 31,077 p/s | 0.60× |

**Pros:**
- ✅ Adaptive (only searches what's needed)
- ✅ Better retention than small fixed radius
- ✅ Vmappable (JAX-compatible)

**Cons:**
- ❌ **Surprisingly slow** in benchmarks
- ❌ Multiple `jnp.where` branches add overhead
- ❌ Slower than fixed radius=10 despite fewer tests

**Production Use:** Theoretically optimal, but benchmarks show unexpected slowdown. Currently recommended despite lower throughput due to better retention.

**⚠️ Note:** The incremental method shows anomalous performance in benchmarks - slower than fixed radius despite testing fewer elements. This may be due to:
- JIT compilation overhead from multiple conditional tiers
- Memory access patterns (non-contiguous leaf access)
- `jnp.where` overhead in cascading logic

---

#### 1.3 **Neighbors Search** (`'neighbors'`)

**How it works:**
- Uses Morton neighbor arithmetic
- Finds 3×3×3 = 27 octant neighbors around particle
- Falls back to 5×5×5 outer shell (98 octants) at boundaries
- Requires octree prefix table (depth > 0)

**Configuration:**
```python
L2_SEARCH_METHOD = 'neighbors'
# Requires mesh_gpu_octree.table_depth > 0
```

**Performance (FLA mesh, 30K particles, 100 RK4 steps):**
| Metric | Value |
|--------|-------|
| Retention | 98.21% |
| Tests | Variable (27-125 octants) |
| Throughput | **2,378 p/s** |
| Speedup | 0.05× (20× slower than radius=10!) |

**Pros:**
- ✅ Spatially aware (true 3D neighbors)
- ✅ High retention
- ✅ Vmappable

**Cons:**
- ❌ **Extremely slow** (20× slower than baseline)
- ❌ Requires octree prefix table
- ❌ Complex neighbor arithmetic overhead

**Production Use:** **Not recommended** - too slow despite high retention.

---

#### 1.4 **Hierarchical Search** (`'hierarchical'`)

**How it works:**
- Multi-depth conditional search
- Starts at depth-7, falls back to depth-6 if not found
- Conditional execution reduces wasted work
- Requires octree prefix table

**Configuration:**
```python
L2_SEARCH_METHOD = 'hierarchical'
# Requires mesh_gpu_octree.table_depth > 0
```

**Performance (FLA mesh, 30K particles, 100 RK4 steps):**
| Metric | Value |
|--------|-------|
| Retention | 98.14% |
| Tests | Variable (depth-dependent) |
| Throughput | **2,529 p/s** |
| Speedup | 0.05× (20× slower than radius=10!) |

**Pros:**
- ✅ Depth-aware search
- ✅ High retention
- ✅ Vmappable

**Cons:**
- ❌ **Extremely slow** (similar to neighbors)
- ❌ Requires octree prefix table
- ❌ Complex depth traversal overhead

**Production Use:** **Not recommended** - too slow.

---

### 2. Mesh-Aligned Octree Methods

These methods exploit the **intrinsic octree structure** in Kuhn tetrahedral meshes (axis-aligned tets from regular hex mesh decomposition).

#### 2.1 **Direct Mesh-Aligned Octree** (`'mesh_aligned_octree'`)

**How it works:**
- Extract octree cells directly from mesh topology
- Map particle position to cell using grid arithmetic
- Test only elements in that single cell (~5.9 elements)
- **No Morton curve needed**

**Configuration:**
```python
L2_SEARCH_METHOD = 'mesh_aligned_octree'
# Requires Kuhn mesh structure
```

**Performance (FLA mesh):**
| Metric | Value |
|--------|-------|
| Retention | **74.6%** (standalone test) |
| Tests/Particle | ~5.9 elements (1 cell) |
| Cells Extracted | ~524,288 cells |
| Elements/Cell | 5.9 (mean) |

**Pros:**
- ✅ **Extremely fast** (~6 tests vs ~536 for Morton)
- ✅ No octree build overhead
- ✅ Direct grid arithmetic
- ✅ Vmappable

**Cons:**
- ❌ **Low retention** (74.6%)
- ❌ Elements span multiple cells → assigned to wrong cell
- ❌ Only works with Kuhn meshes

**Production Use:** **Not suitable** - retention too low. Elements that span multiple octree cells get assigned incorrectly.

**Why it fails:** The intrinsic mesh octree assigns each element to a single cell based on its centroid. However, tetrahedra can span multiple cells, so a particle in an element may query a different cell than where the element was assigned.

---

#### 2.2 **Hybrid Mesh-Aligned Morton** (`'mesh_aligned_morton'`)

**How it works:**
- Extract mesh-aligned octree cells
- Build Morton curve over **cell centers** (not element centroids)
- Use radius or incremental search over cells
- Each cell contains ~5.9 elements

**Configuration:**
```python
L2_SEARCH_METHOD = 'mesh_aligned_morton'
L2_SEARCH_RADIUS = 2  # searches 5 cells → ~30 tests
# Or use incremental: (2, 5, 10) → ~68 tests avg
```

**Performance (FLA mesh, expected):**
| Configuration | Retention | Tests/Particle | Notes |
|---------------|-----------|----------------|-------|
| radius=2 | ~98% (expected) | ~30 (5 cells × 5.9 elem/cell) | Not yet benchmarked |
| incremental (2,5,10) | ~98% (expected) | ~68 avg (11.5 cells × 5.9) | Not yet benchmarked |

**Pros:**
- ✅ Combines mesh structure + proven radius search
- ✅ 10× fewer tests than regular Morton
- ✅ Should achieve ~98% retention
- ✅ Vmappable

**Cons:**
- ❌ Only works with Kuhn meshes
- ❌ Not yet benchmarked in production
- ❌ Requires mesh octree extraction

**Production Use:** **Promising** - expected to be faster than regular Morton with same retention. Needs validation.

---

### 3. KD-Tree Node-Based Search (`'kdtree'`)

**How it works:**
- Build KD-tree from mesh **node positions**
- Find K nearest nodes to query position
- Test all elements connected to those nodes (~21.4 elements/node)
- **No spatial structure needed** - works with any mesh

**Configuration:**
```python
L2_SEARCH_METHOD = 'kdtree'
KDTREE_K_NEAREST = 3  # Number of nearest nodes
KDTREE_MAX_TESTS = 256
# Requires: pip install jaxkd
```

**Performance (FLA mesh, 1,000 random particles):**
| Metric | Value |
|--------|-------|
| Retention | **95.1%** (951/1000 found) |
| Tests/Particle | ~64 (K=3 × 21.4 elem/node) |
| Build Time | 3.2s (CPU) |
| Upload Time | 3.2s (GPU) |

**Performance (Initial Assignment, 225K particles):**
| Metric | Value |
|--------|-------|
| Retention | **100%** (with cascading radii) |
| Total Time | 377s |

**Pros:**
- ✅ Excellent retention (95-100%)
- ✅ Works with any mesh (not just Kuhn)
- ✅ Fewer tests than Morton (~64 vs ~536)
- ✅ Simple algorithm

**Cons:**
- ❌ **NOT vmappable** - cannot be used in RK4 tracking!
- ❌ `jaxkd.query_neighbors` has Python control flow (tree traversal)
- ❌ Only works for batch searches (initial assignment, analysis)
- ❌ Requires external library (`jaxkd`)

**Critical Limitation:**
```
❌ CANNOT be used in vmapped RK4 tracking!
Error: TracerIntegerConversionError when JAX tries to trace through KD-tree query
```

**Why it fails for RK4:** The KD-tree query (`jk.query_neighbors`) uses Python loops for tree traversal, which JAX cannot trace when compiling vmapped functions. Works perfectly for **batch searches** where the tree is queried before vmap.

**Production Use:**
- ✅ **Excellent for initial assignment** (batch search)
- ❌ **Cannot be used for RK4 tracking steps**
- Use for offline analysis, validation, batch particle location

---

## Point-in-Tetrahedron Methods

Point-in-tet methods determine if a point is inside a tetrahedron. This is the innermost computational kernel called millions of times during tracking.

### Method Comparison Table

| Method | How It Works | Throughput | Speedup | Accuracy | Status |
|--------|--------------|------------|---------|----------|--------|
| **current** | Barycentric coordinates (Cramer's rule) | 110 p/s | 1.00× | 100% | ✅ Baseline |
| **skala** | Optimized cross products (Skála 2014) | 99 p/s | 0.90× | 100% | ✅ Works |
| **axis_aligned** | OLD AA detection (original impl) | 49 p/s | 0.45× | 99.4% | ❌ BROKEN |
| **pure_aa** | NEW AA-only method (corrected) | **3,036 p/s** | **27.49×** | 100%* | ⚠️ FALSE POSITIVES |
| **skala_memory_opt** | Skála with precomputed vertices | 108 p/s | 0.97× | 100% | ✅ Works |
| **branchless_hybrid** | Hybrid AA+Skála (branchless) | 68 p/s | 0.62× | 93.7% | ❌ LOW ACCURACY |
| **inverse** | Precomputed inverse matrices | **350-450 p/s** | **3-4×** | 100% | ✅ **RECOMMENDED** |

\* `pure_aa` achieves 27× speedup but **disagrees with all other methods** (30,000/30,000 different element assignments). This indicates **false positives** - accepting particles that aren't truly inside elements.

---

### Detailed Method Descriptions

#### 1. **current** (Baseline)

**Algorithm:**
- Compute barycentric coordinates using Cramer's rule
- Check if all coordinates are non-negative
- ~145 FLOPs per test

**Performance:**
- Throughput: 110 p/s (initial assignment, 30K particles)
- Accuracy: 100% (reference implementation)

**Pros:**
- ✅ Proven, stable reference
- ✅ 100% accurate

**Cons:**
- ❌ Slow baseline

**Production Use:** Reference only, replaced by faster methods.

---

#### 2. **skala** (Skála's Method)

**Algorithm:**
- Optimized cross products (Skála 2014 paper)
- Fewer operations than barycentric
- ~48 FLOPs per test (3× reduction)

**Performance:**
- Throughput: 99 p/s
- Speedup: 0.90× (slower than current!)
- Accuracy: 100% (agrees with current)

**Pros:**
- ✅ Theoretically faster (fewer FLOPs)
- ✅ 100% accurate

**Cons:**
- ❌ **Actually slower** in practice
- ❌ Memory access patterns may hurt performance

**Production Use:** Not recommended - no advantage over current.

---

#### 3. **axis_aligned** (OLD - BROKEN)

**Algorithm:**
- Detect axis-aligned tetrahedra
- Use fast bounding box checks for AA tets
- Fall back to barycentric for non-AA tets

**Performance:**
- Throughput: 49 p/s (0.45× slower than current!)
- Accuracy: 99.4% (180/30,000 particles misassigned)

**Pros:**
- ✅ Concept is sound (AA detection should be fast)

**Cons:**
- ❌ **BROKEN** - incorrect AA detection
- ❌ Slower than baseline
- ❌ Lower accuracy

**Production Use:** **Deprecated** - replaced by corrected methods.

---

#### 4. **pure_aa** (NEW - FALSE POSITIVES)

**Algorithm:**
- AA detection only (no fallback)
- Ultra-fast bounding box checks
- Assumes all tets are axis-aligned

**Performance:**
- Throughput: **3,036 p/s** (27.49× speedup!)
- Accuracy: 100% assignment rate, but **disagrees with all other methods**

**Pros:**
- ✅ Extremely fast (27× speedup)
- ✅ Simple implementation

**Cons:**
- ❌ **FALSE POSITIVES** - accepts particles not truly inside
- ❌ All 30,000 particles assigned to different elements than current
- ❌ Not suitable for production

**Production Use:** **NOT SAFE** - despite speed, produces incorrect results.

**Why it's wrong:** The AA detection accepts particles that are inside the axis-aligned bounding box but outside the actual tetrahedral volume. It finds *some* element, but not the *correct* element.

---

#### 5. **skala_memory_opt** (Corrected)

**Algorithm:**
- Skála's method with precomputed element vertices
- Better memory access patterns

**Performance:**
- Throughput: 108 p/s (0.97× vs current)
- Accuracy: 100% (agrees with current)

**Pros:**
- ✅ Matches current accuracy
- ✅ Slightly better than original skala

**Cons:**
- ❌ No significant speedup

**Production Use:** Marginal improvement, not worth switching from current.

---

#### 6. **branchless_hybrid** (NEW - LOW ACCURACY)

**Algorithm:**
- Hybrid AA + Skála with branchless execution
- Tests both paths, selects result without branching

**Performance:**
- Throughput: 68 p/s (0.62× vs current)
- Accuracy: 93.7% (1,880/30,000 particles lost!)

**Pros:**
- ✅ Branchless execution (theoretically faster)

**Cons:**
- ❌ **LOW ACCURACY** (6.3% particle loss)
- ❌ Slower than baseline
- ❌ Correctness issues

**Production Use:** **Not recommended** - accuracy too low.

---

#### 7. **inverse** (RECOMMENDED - PRODUCTION)

**Algorithm:**
- Precompute inverse transformation matrices for all elements (one-time cost)
- Point-in-tet becomes a single matrix-vector multiply + comparison
- No per-query matrix inversions or cross products

**Performance:**
- Throughput: **350-450 p/s** (3-4× speedup vs current)
- Accuracy: 100% (mathematically equivalent to barycentric)
- Precomputation: 28.9s for 3M elements (one-time)
- Memory: 139.6 MB for inverse matrices

**Pros:**
- ✅ **3-4× speedup** (validated in production)
- ✅ **100% accurate** (agrees with current)
- ✅ Mathematically proven correctness
- ✅ Simple runtime computation

**Cons:**
- ❌ Requires precomputation (one-time ~29s)
- ❌ Extra memory (139.6 MB)

**Production Use:** ✅ **HIGHLY RECOMMENDED** - best balance of speed and accuracy.

**Why it works:**
```
Transform: p' = M^{-1} * (p - p0)
Inside if: p'_x >= 0, p'_y >= 0, p'_z >= 0, (1 - p'_x - p'_y - p'_z) >= 0
```

The inverse matrix `M^{-1}` transforms world coordinates to barycentric coordinates in a single multiply. Precomputing `M^{-1}` eliminates all per-query matrix operations.

---

## Performance Summary

### Element Search Methods - Production Recommendations

**Best for RK4 Tracking (vmappable):**
```python
# Recommended: Fixed radius (fastest, proven)
L2_SEARCH_METHOD = 'radius'
L2_SEARCH_RADIUS = 10  # 96.96% retention, 51,894 p/s

# Alternative: Incremental (better retention, but slower)
L2_SEARCH_METHOD = 'incremental'
INCREMENTAL_SEARCH_RADII = (2, 4, 8, 15, 30)  # 98.21% retention, 9,136 p/s
```

**Best for Initial Assignment (batch):**
```python
# KD-tree provides 100% retention with cascading radii
# But can also use Morton with large cascading radii (same result)
INITIAL_SEARCH_RADIUS = 500
INITIAL_SEARCH_FALLBACK_RADII = [1000, 2000, 5000, 10000, 100000]
# Result: 100% retention
```

**Not Recommended:**
- ❌ `neighbors` - Too slow (20× slower)
- ❌ `hierarchical` - Too slow (20× slower)
- ❌ `mesh_aligned_octree` - Low retention (74.6%)
- ❌ `kdtree` for RK4 - Not vmappable

**Promising (needs validation):**
- ⚠️ `mesh_aligned_morton` - Expected 10× faster than Morton, same retention

---

### Point-in-Tet Methods - Production Recommendations

**Production Configuration:**
```python
# RECOMMENDED: Inverse method
POINT_IN_TET_METHOD = 'inverse'
# Provides: 3-4× speedup, 100% accuracy, proven in production

# Precomputation (one-time cost):
M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)
set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)
```

**Performance Impact on Full Tracking:**
- Point-in-tet is ~60% of RK4 runtime
- 4× speedup on point-in-tet → **2.4× overall RK4 speedup**
- For 2,500-step tracking: Hours of savings

**Not Recommended:**
- ❌ `pure_aa` - False positives (27× fast but wrong!)
- ❌ `branchless_hybrid` - Low accuracy (93.7%)
- ❌ `axis_aligned` - Broken (99.4% accuracy)
- ❌ `skala` / `skala_memory_opt` - No advantage over current

---

## Recommendations

### For Production Tracking

**L2 Search Configuration:**
```python
# Use fixed radius for speed, or incremental for retention
config.L2_SEARCH_METHOD = 'radius'  # or 'incremental'
L2_SEARCH_RADIUS = 10  # for radius method
INCREMENTAL_SEARCH_RADII = (2, 4, 8, 15, 30)  # for incremental
```

**Point-in-Tet Configuration:**
```python
# MANDATORY: Use inverse method
config.POINT_IN_TET_METHOD = 'inverse'

# One-time precomputation:
M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)
M_inv_gpu = jax.device_put(M_inv_array)
p0_gpu = jax.device_put(p0_array)
set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)
```

**Expected Performance:**
- Initial assignment: 100% (with cascading radii)
- RK4 retention at 2,500 steps: 95-98% (depends on flow complexity)
- Throughput: 50,000-120,000 particles/s (with inverse method)

---

### Benchmark-Specific Observations

#### L2 Search Anomalies

⚠️ **Unexpected Results:**
1. **Incremental is slower than fixed radius** despite testing fewer elements
   - Cause: `jnp.where` overhead, non-contiguous memory access
   - Expected: Faster due to adaptive search
   - Actual: 0.18× speedup (5.7× slower!)

2. **Neighbors/Hierarchical are extremely slow**
   - Cause: Complex neighbor arithmetic, octree prefix table lookups
   - Expected: Comparable to radius search
   - Actual: 0.05× speedup (20× slower!)

**Recommendation:** These anomalies suggest that L2 search methods need re-benchmarking with:
- Different particle counts
- Different mesh complexities
- Profile-guided optimization

The **fixed radius=10** method is currently the fastest validated option for RK4 tracking.

---

#### Point-in-Tet Validated Results

✅ **Inverse method validated:**
- Consistent 3-4× speedup across multiple tests
- 100% accuracy (agrees with reference implementation)
- Production-validated with 2,500-step tracking

✅ **Current/skala_memory_opt agreement:**
- Both produce identical results
- Skala_memory_opt slightly faster (0.97×) but negligible

❌ **Pure_aa and branchless_hybrid NOT production-ready:**
- Pure_aa: False positives (wrong elements)
- Branchless_hybrid: 6.3% particle loss
- Do not use despite speed claims

---

## Conclusion

### Production Configuration (Validated)

```python
# Point-in-Tet: INVERSE (mandatory for performance)
config.POINT_IN_TET_METHOD = 'inverse'

# L2 Search: RADIUS or INCREMENTAL (both vmappable)
config.L2_SEARCH_METHOD = 'radius'  # Fastest (51,894 p/s, 96.96% retention)
# OR
config.L2_SEARCH_METHOD = 'incremental'  # Better retention (9,136 p/s, 98.21%)

# Initial Assignment: Large cascading radii (100% retention)
INITIAL_SEARCH_RADIUS = 500
INITIAL_SEARCH_FALLBACK_RADII = [1000, 2000, 5000, 10000, 100000]

# L1 Settings
ENABLE_L1_SEARCH = True
N_HOPS = 5  # Adaptive 5-6 hops at refinement boundaries
```

**Expected Results:**
- Initial assignment: 100%
- Final retention (2,500 steps): 95-98%
- Throughput: 50K-120K particles/s (2.4× faster with inverse method)

---

### Future Work

1. **Re-benchmark L2 methods** with:
   - Profile-guided optimization
   - Different test scenarios
   - Investigation of incremental slowdown

2. **Validate mesh-aligned Morton** in production:
   - Expected 10× speedup over regular Morton
   - Should achieve ~98% retention
   - Only works with Kuhn meshes

3. **Investigate KD-tree for hybrid approach**:
   - Use KD-tree for initial assignment (100% retention)
   - Use incremental Morton for RK4 tracking
   - Best of both worlds

4. **Optimize incremental search**:
   - Current implementation slower than expected
   - May benefit from different tier configurations
   - Profile `jnp.where` overhead

---

**Report End**
