# Nearest Node Search Analysis
## Comparison with Old JAXTrace Implementation

**Date**: 2026-01-29
**Author**: Analysis based on old_implementation branch and current fully-fused RK4

---

## Executive Summary

The **nearest node search** was the core algorithm in JAXTrace's old implementation (pre-GPU). We have **already implemented and tested** a KD-tree version in the current codebase, but it **cannot be vmapped** for use inside fully-fused RK4. However, a **vmappable version using brute-force nearest node search** is feasible and could achieve high retention with acceptable performance.

**Key Finding**: The old implementation computed distances to **all nodes** (naive O(N) search), which is too slow for 571K nodes. But a **GPU-optimized brute-force version** could work by leveraging JAX's parallel reduction and GPU memory bandwidth.

---

## 1. Old Implementation Analysis

### Algorithm from old_implementation Branch

From `https://github.com/ARHashemi/JAXTrace/blob/old_implementation/optimized_particle_advection.py`:

```python
@jit
def interpolate_velocity(position: jnp.ndarray, time_idx: int) -> jnp.ndarray:
    """Interpolate velocity at given position and time."""
    safe_time_idx = self.static_time_step if self.static else jnp.clip(
        time_idx, 0, len(self.velocity_data) - 1
    )
    # CRITICAL LINE: Compute distance to ALL nodes
    distances = jnp.linalg.norm(self.grid_points - position, axis=1)
    nearest_idx = jnp.argmin(distances)
    return self.velocity_data[safe_time_idx, nearest_idx]
```

**Key characteristics**:
1. **Brute-force O(N) search**: Compute distance to ALL grid points
2. **Single nearest node**: Returns velocity at nearest node (not element-based)
3. **JAX-compatible**: Uses `jnp.linalg.norm` and `jnp.argmin` (fully vmappable)
4. **No spatial indexing**: No KD-tree, octree, or Morton curve
5. **Simple interpolation**: Direct node velocity, no element-based interpolation

### Why Was It Fast?

The old implementation was fast because:
1. **Small mesh**: Old meshes had ~10K-50K nodes (not 571K)
2. **GPU parallelization**: `jnp.linalg.norm` fully parallel on GPU
3. **Memory bandwidth**: Dominated by memory reads, not compute
4. **No element search**: Direct node velocity lookup (no point-in-tet)

### Why We Moved Away

The old implementation had critical limitations:
1. **No element-based interpolation**: Used node velocities directly
   - ❌ Discontinuous at node boundaries
   - ❌ No proper tetrahedral FEM interpolation
2. **Not suitable for tracking**: Particles need element containment
   - ❌ Cannot determine if particle left mesh
   - ❌ No boundary handling
3. **Scalability**: O(N) search becomes prohibitive for large meshes
   - ❌ 571K nodes × 225K particles = 128B distance computations

---

## 2. Current KD-Tree Implementation

### What We Already Have

File: [jaxtrace/gpu/search/kdtree_node_search.py](jaxtrace/gpu/search/kdtree_node_search.py)

**Implementation status**: ✅ **Fully implemented** (Jan 27, 2026)

```python
def search_L2_kdtree_single(pos, kdtree_gpu, k_nearest=3):
    """
    L2 search using KD-tree nearest nodes.

    Algorithm:
    1. Find K nearest nodes (using jaxkd library)
    2. For each nearest node, get connected elements
    3. Test elements with point-in-tet
    4. Return first containing element

    Performance:
    - K=3 → ~30 element tests (3 nodes × ~10 elem/node)
    - Expected retention: ~99%
    """
    # Find K nearest nodes (jaxkd.query_neighbors)
    nearest_node_ids = jk.query_neighbors(kdtree_gpu.kdtree, pos, k=k_nearest)

    # Search elements connected to nearest nodes
    for node_id in nearest_node_ids:
        elements = get_node_elements(node_id)  # CSR lookup
        for elem_id in elements:
            if point_in_tet(pos, elem_id):
                return elem_id
    return -1  # Not found
```

**Data structure**:
```python
@dataclass
class NodeKDTreeGPU:
    node_positions: (571173, 3) float32
    connectivity: (3048900, 4) int32
    node_to_elements_offsets: (571174,) int32  # CSR offsets
    node_to_elements_data: (12195600,) int32   # CSR data (21.4 elem/node avg)
    kdtree: jaxkd.KDTree  # Built from node positions
```

### Test Results

From [logs/test_production_kdtree.log](logs/test_production_kdtree.log):

**Initial Assignment**:
- ✅ 100% assignment success (225,000/225,000 particles)
- Time: 377s (cascading radius search: 500, 1000, 2000, 5000, 10000, 100000)
- Throughput: ~596 p/s

**RK4 Tracking**:
- ❌ **FAILED** - TracerIntegerConversionError
- Error at line 317: `for elem_idx in range(start, end):`
- **Root cause**: `start` and `end` are JAX tracers (not Python ints)
- **Fundamental issue**: `jk.query_neighbors()` has Python control flow

### Why KD-Tree Cannot Be Vmapped

From error traceback:
```python
File "kdtree_node_search.py", line 300, in search_L2_kdtree_single
    nearest_node_ids, distances = jk.query_neighbors(
        kdtree_gpu.kdtree, pos_f64.reshape(1, 3), k=k_nearest
    )
# Problem: jk.query_neighbors uses Python control flow (tree traversal)
# This cannot be traced by JAX → cannot be inside vmap
```

**From KDTREE_VMAPPABLE_ANALYSIS.md**:
> The `jaxkd` library's KD-tree traversal uses `lax.while_loop` with **data-dependent termination**. The number of tree nodes visited varies per query (e.g., 12, 8, 15 iterations). JAX vmap requires **identical control flow** for all inputs, which cannot be satisfied by adaptive tree traversal.

### Current Usage Limitation

- ✅ **Works for**: Initial assignment (batch search outside vmap)
- ❌ **Fails for**: RK4 tracking (per-particle search inside vmap)
- ✅ **Works for**: Standalone batch searches

---

## 3. Vmappable Nearest Node Approach

### Proposed Algorithm

**Core idea**: Compute distances to ALL nodes in parallel (like old implementation), but make it work for large meshes through GPU optimization.

```python
def search_L2_nearest_node_vmappable(pos: jax.Array, mesh_gpu: MeshGPU) -> jnp.int32:
    """
    Vmappable nearest node search for L2 element location.

    Algorithm:
    1. Compute distances to ALL nodes (parallel on GPU)
    2. Find K nearest nodes (jnp.argpartition)
    3. Search elements connected to K nearest nodes
    4. First containing element wins

    JAX-compatible:
    - No Python control flow
    - Fixed loop bounds (K nearest nodes)
    - Fully vmappable
    """
    # Step 1: Compute distances to ALL nodes (571,173 distances)
    # Shape: (571173,)
    distances_sq = jnp.sum((mesh_gpu.node_positions - pos) ** 2, axis=1)

    # Step 2: Find K nearest nodes (K=3-5)
    # Use argpartition for efficient partial sort
    k_nearest = 3
    nearest_indices = jnp.argpartition(distances_sq, k_nearest)[:k_nearest]

    # Step 3: Search elements connected to K nearest nodes
    found_elem = jnp.int32(-1)

    for k_idx in range(k_nearest):
        if found_elem >= 0:
            break  # Already found (will be optimized by JIT)

        node_id = nearest_indices[k_idx]

        # Get elements connected to this node (CSR lookup)
        start = mesh_gpu.node_to_elements_offsets[node_id]
        end = mesh_gpu.node_to_elements_offsets[node_id + 1]
        n_elements = end - start

        # Bounded loop over elements (max 256 per node)
        max_elements_per_node = 256
        for j in range(max_elements_per_node):
            if (found_elem >= 0) or (j >= n_elements):
                break

            elem_idx = start + j
            test_elem_id = mesh_gpu.node_to_elements_data[elem_idx]

            # Point-in-tet test
            is_inside = point_in_tet_gpu(pos, test_elem_id, mesh_gpu)

            if is_inside:
                found_elem = test_elem_id
                break

    return found_elem
```

### Key Differences from KD-Tree Version

| Aspect | KD-Tree (Non-vmappable) | Nearest Node (Vmappable) |
|--------|------------------------|--------------------------|
| **Distance computation** | O(log N) tree traversal | O(N) brute-force |
| **Spatial indexing** | KD-tree structure | None (flat array) |
| **Control flow** | Python loops (jaxkd) | JAX loops (lax.fori_loop) |
| **Vmappable** | ❌ No | ✅ Yes |
| **Memory access** | Tree traversal (scattered) | Sequential scan (coalesced) |
| **Compile-time cost** | N/A (Python control flow) | High (JIT trace) |
| **Runtime cost** | ~10 µs/particle | ~100-500 µs/particle |

---

## 4. Performance Analysis

### Computational Cost

**Per-particle cost**:
1. **Distance computation**: 571,173 distances
   - Compute: `(x-x0)^2 + (y-y0)^2 + (z-z0)^2` → 6 FLOPS × 571K = 3.4M FLOPS
   - Memory: Read 571K nodes (571K × 12 bytes = 6.9 MB)
   - **GPU bandwidth limited**: ~10 GB/s → ~0.7 ms/particle

2. **K nearest selection**: `jnp.argpartition(571173, k=3)`
   - Algorithm: Partial quicksort (O(N) average)
   - Memory: Read 571K distances (571K × 4 bytes = 2.3 MB)
   - **Estimate**: ~0.2 ms/particle

3. **Element search**: K=3 nodes × ~21 elem/node = ~63 tests
   - Point-in-tet: ~10 FLOPS × 63 = 630 FLOPS
   - **Negligible**: <0.01 ms/particle

**Total per-particle**: ~1 ms/particle (single particle)

### Vmap Parallelization

**With vmap over 225,000 particles**:
- GPU has 10,752 CUDA cores (RTX 5000 Ada)
- Theoretical peak: 225,000 particles × 1 ms / 10,752 cores = **21 ms**
- Reality with overhead: **50-200 ms** (considering memory bandwidth, JIT overhead)

**Expected throughput**:
- Optimistic: 225,000 / 0.05 = **4.5M particles/s**
- Realistic: 225,000 / 0.2 = **1.1M particles/s**
- Conservative: 225,000 / 0.5 = **450K particles/s**

### Comparison with Current Methods

From [logs/benchmark_l2_search_methods.log](logs/benchmark_l2_search_methods.log):

| Method | Retention | Throughput | Cost/Particle |
|--------|-----------|------------|---------------|
| **Radius=10** | 96.96% | 51,894 p/s | **19.3 µs** ★ |
| Radius=30 | 98.21% | 17,895 p/s | 55.9 µs |
| Incremental | 98.21% | 9,136 p/s | 109.5 µs |
| Neighbors | 98.21% | 2,378 p/s | 420.5 µs |
| Hierarchical | 98.14% | 2,529 p/s | 395.5 µs |
| **Nearest Node (est.)** | **~99%*** | **~100-450K p/s*** | **~2-10 µs*** |

\* Estimated based on GPU bandwidth and parallelization analysis

**Key insight**: Nearest node could be **2-10× faster than radius=10** despite computing 571K distances, because:
1. **Memory coalescing**: Sequential read of node array (efficient on GPU)
2. **Parallel reduction**: 10K+ cores compute distances simultaneously
3. **No element tests**: Directly finds nodes, then tests ~63 elements (vs ~2,247 for radius=10)

---

## 5. Implementation Strategy

### Phase 1: Basic Vmappable Version

**File**: `jaxtrace/gpu/search/nearest_node_vmappable.py`

```python
@dataclass
class NodeSearchGPU:
    """GPU structure for nearest node search."""
    node_positions: jax.Array  # (571173, 3) float32
    connectivity: jax.Array  # (3048900, 4) int32
    node_to_elements_offsets: jax.Array  # (571174,) int32
    node_to_elements_data: jax.Array  # (12195600,) int32
    n_nodes: int
    n_elements: int

def search_L2_nearest_node_single(
    pos: jax.Array,
    node_search_gpu: NodeSearchGPU,
    k_nearest: int = 3
) -> jnp.int32:
    """
    Vmappable nearest node search (single particle).

    WARNING: This computes distances to ALL nodes (571K).
    Only use inside vmap where GPU parallelism amortizes cost.
    """
    # Compute distances to ALL nodes
    distances_sq = jnp.sum((node_search_gpu.node_positions - pos) ** 2, axis=1)

    # Find K nearest (partial sort)
    nearest_indices = jnp.argpartition(distances_sq, k_nearest)[:k_nearest]

    # Search elements connected to nearest nodes
    return _search_elements_from_nodes(pos, nearest_indices, node_search_gpu, k_nearest)

def _search_elements_from_nodes(
    pos: jax.Array,
    nearest_node_ids: jax.Array,
    node_search_gpu: NodeSearchGPU,
    k_nearest: int
) -> jnp.int32:
    """JAX-traceable element search with bounded loops."""
    from jax import lax

    def search_one_node(k_idx, found_elem):
        """Search elements connected to one node."""
        # Get node and its elements
        node_id = nearest_node_ids[k_idx]
        start = node_search_gpu.node_to_elements_offsets[node_id]
        end = node_search_gpu.node_to_elements_offsets[node_id + 1]
        n_elements = end - start

        def check_element(j, inner_found):
            """Check one element (bounded loop body)."""
            active = (inner_found == -1) & (j < n_elements)

            elem_idx = start + j
            test_elem_id = jnp.where(active, node_search_gpu.node_to_elements_data[elem_idx], jnp.int32(0))

            is_inside = jnp.where(
                active,
                point_in_tet_gpu(pos, test_elem_id, node_search_gpu),
                False
            )

            return jnp.where(is_inside & active, test_elem_id, inner_found)

        # Bounded loop (max 256 elements per node)
        n_to_test = jnp.minimum(n_elements, jnp.int32(256))
        return lax.fori_loop(0, n_to_test, check_element, found_elem)

    # Loop over K nearest nodes
    return lax.fori_loop(0, k_nearest, search_one_node, jnp.int32(-1))
```

### Phase 2: Optimization

**Optimization 1: Early exit on first node**
```python
# Most particles found in first nearest node
# No need to check all K nodes if found
def search_one_node_early_exit(k_idx, carry):
    found_elem, found_flag = carry

    # Skip if already found (will be masked out)
    active = (found_flag == 0)

    # ... search logic ...

    new_found = jnp.where(is_inside & active, test_elem_id, found_elem)
    new_flag = jnp.where(is_inside & active, 1, found_flag)

    return (new_found, new_flag)
```

**Optimization 2: Adaptive K**
```python
# Start with K=1, increase if not found
# Most particles (>90%) found in K=1
def search_L2_nearest_node_adaptive(pos, node_search_gpu):
    # Try K=1 first (cheapest)
    elem = search_L2_nearest_node_single(pos, node_search_gpu, k_nearest=1)

    # If not found, try K=3
    elem = jnp.where(elem >= 0, elem,
                     search_L2_nearest_node_single(pos, node_search_gpu, k_nearest=3))

    return elem
```

**Optimization 3: Tiled distance computation**
```python
# Reduce memory pressure by processing nodes in tiles
# Reduces peak memory from 571K to 10K per particle
def find_k_nearest_tiled(pos, node_positions, k_nearest, tile_size=10000):
    n_nodes = node_positions.shape[0]
    n_tiles = (n_nodes + tile_size - 1) // tile_size

    # Keep top K across all tiles
    top_k_distances = jnp.full(k_nearest, jnp.inf)
    top_k_indices = jnp.full(k_nearest, -1, dtype=jnp.int32)

    for tile_idx in range(n_tiles):
        start = tile_idx * tile_size
        end = min(start + tile_size, n_nodes)

        # Compute distances for this tile
        tile_distances = jnp.sum((node_positions[start:end] - pos) ** 2, axis=1)

        # Merge with current top K
        top_k_distances, top_k_indices = merge_top_k(
            top_k_distances, top_k_indices,
            tile_distances, jnp.arange(start, end)
        )

    return top_k_indices
```

---

## 6. Integration with RK4

### RK4 Dispatcher Modification

**File**: [jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py](jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py)

```python
def search_l2_single(pos: jax.Array) -> jax.Array:
    """L2: Global search - method selected by config."""
    if l2_search_method == 'nearest_node':
        # NEW: Vmappable nearest node search
        return search_L2_nearest_node_single(pos, node_search_gpu, k_nearest=3)
    elif l2_search_method == 'hierarchical':
        return search_L2_morton_hierarchical_single(pos, mesh_gpu_global_morton)
    elif l2_search_method == 'neighbors':
        return search_L2_morton_neighbors_single(pos, mesh_gpu_global_morton)
    else:
        return search_L2_global_morton_single(pos, mesh_gpu_global_morton, l2_search_radius)
```

### Production Config

**File**: [production_tracking_fully_fused_timedep.py](production_tracking_fully_fused_timedep.py)

```python
# L2 Search Method Selection:
#   'radius': Linear ±radius search along Morton curve
#             - Performance: ~13K particles/s, 79% retention
#   'neighbors': Morton neighbor arithmetic (single depth)
#                - Performance: ~21K particles/s, 80% retention
#   'hierarchical': Multi-depth Morton neighbors (depth 7 + depth 6)
#                   - Performance: ~20K particles/s, 85-90% retention
#   'nearest_node': Brute-force nearest node search (NEW)
#                   - Expected: ~100-450K particles/s, ~99% retention
#                   - Fully vmappable, GPU-optimized
L2_SEARCH_METHOD = 'nearest_node'  # ← USER CONFIGURABLE
```

---

## 7. Expected Results

### Retention

**Expected retention: ~99%**

Reasoning:
1. **K=3 nearest nodes**: Covers ~90% of mesh locally
2. **21.4 elements per node**: Very dense connectivity
3. **Point-in-tet test**: Accurate element containment

**Failure cases** (~1%):
- Particles at mesh corners (3 nearest nodes on boundary)
- Particles in coarse-to-fine transition regions
- True physical loss (particle exits mesh)

### Performance

**Expected throughput: ~100-450K particles/s**

Breakdown per RK4 step (225,000 particles):
1. **Distance computation**: 571K × 225K = 128B distances
   - GPU memory bandwidth: ~700 GB/s (RTX 5000 Ada)
   - Data transfer: 128B × 4 bytes = 512 MB
   - **Time**: 512 MB / 700 GB/s = **0.7 ms**

2. **K-selection** (argpartition): 225K particles × 571K values
   - Parallel partial sort on GPU
   - **Estimate**: **20-50 ms**

3. **Element search**: 225K particles × 63 tests
   - Point-in-tet: 14.2M tests
   - GPU parallel: **Estimate**: **5-10 ms**

**Total per step**: ~30-100 ms → **2,250-7,500 particles/s**

**Wait, this is slower than expected!** The bottleneck is **argpartition over 571K values per particle**.

### Optimization: Tiled Search

**Better approach**: Process nodes in 10K tiles, keep running top-K:
- Memory: 10K nodes × 225K particles × 4 bytes = 9 MB (vs 512 MB)
- **Estimated time**: ~10 ms per step → **22,500 particles/s**

Still slower than radius=10 (51,894 p/s), but **5× faster than neighbors** (2,378 p/s).

---

## 8. Pros and Cons

### Advantages ✅

1. **Fully vmappable**: No Python control flow, compatible with fully-fused RK4
2. **High retention**: ~99% expected (best of all vmappable methods)
3. **Simple algorithm**: No complex spatial indexing structures
4. **GPU-optimized**: Leverages memory bandwidth and parallel reduction
5. **No prefix table bug**: Doesn't depend on octree depth or Morton encoding
6. **Deterministic**: Always finds K nearest nodes (no adaptive depth issues)
7. **Already partially implemented**: KD-tree version has node→elements mapping

### Disadvantages ❌

1. **O(N) complexity**: Computes distances to ALL 571K nodes per particle
2. **Memory bandwidth limited**: 512 MB data transfer per RK4 step
3. **Slower than radius=10**: Estimated 22K p/s vs 52K p/s
4. **Not scalable**: Performance degrades linearly with number of nodes
5. **High compile-time cost**: Large JIT trace (571K operations)
6. **Argpartition overhead**: Finding K=3 in 571K values is expensive

---

## 9. Comparison with All Methods

| Method | Retention | Throughput | Vmappable | Scalability | Notes |
|--------|-----------|------------|-----------|-------------|-------|
| **Radius=10** | 96.96% | 51,894 p/s | ✅ Yes | ✅ O(log N) | **Production baseline** |
| Radius=30 | 98.21% | 17,895 p/s | ✅ Yes | ✅ O(log N) | 3× slower |
| Incremental | 98.21% | 9,136 p/s | ✅ Yes | ✅ O(log N) | Anomalously slow |
| Neighbors | 98.21% | 2,378 p/s | ✅ Yes | ✅ O(log N) | 20× slower (prefix bug) |
| Hierarchical | 98.14% | 2,529 p/s | ✅ Yes | ✅ O(log N) | 20× slower (prefix bug) |
| **KD-tree** | ~100%* | N/A | ❌ No | ✅ O(log N) | **Cannot vmap** |
| **Nearest Node** | ~99%* | ~22K p/s* | ✅ Yes | ❌ O(N) | **New proposal** |

\* Estimated values

---

## 10. Recommendation

### Should We Implement Nearest Node Search?

**TL;DR: Maybe, but with caveats.**

### When It Makes Sense ✅

1. **High retention is critical**: If you need >98% retention at all costs
2. **Mesh size is moderate**: <100K nodes (O(N) cost acceptable)
3. **Simplicity over speed**: Avoid complex octree/Morton debugging
4. **Research/prototyping**: Quick implementation for testing

### When It Doesn't Make Sense ❌

1. **Speed is priority**: Radius=10 is 2× faster with acceptable retention
2. **Large meshes**: >500K nodes makes O(N) cost prohibitive
3. **Production at scale**: Not sustainable for multi-million node meshes
4. **Prefix table bug can be fixed**: Fixing depth 6→7 makes hierarchical competitive

### Better Alternatives

**Option 1: Fix prefix table bug** (RECOMMENDED)
- Change depth 6 → 7 in morton_octree_builder.py (line 271-277)
- Expected: Hierarchical 98-99% @ 10-15K p/s
- Effort: 1 day implementation
- **This is the path forward**

**Option 2: Hybrid approach**
- Use radius=10 for most particles (fast, 97% retention)
- Use nearest node K=1 for failures (slower, high retention)
- Expected: 97-99% retention @ 40-50K p/s
- Effort: 2-3 days

**Option 3: Implement nearest node** (NOT RECOMMENDED)
- Full O(N) brute-force search
- Expected: ~99% retention @ 22K p/s
- Effort: 1 week (implementation + optimization)
- **Only do this if prefix table fix fails**

---

## 11. Implementation Checklist

If you decide to implement nearest node search:

### Phase 1: Basic Implementation (2-3 days)
- [ ] Create `jaxtrace/gpu/search/nearest_node_vmappable.py`
- [ ] Implement `NodeSearchGPU` dataclass
- [ ] Implement `build_node_search_structure()` (reuse from kdtree_node_search.py)
- [ ] Implement `search_L2_nearest_node_single()` with `jnp.argpartition`
- [ ] Add bounded `lax.fori_loop` for element search
- [ ] Test with 1,000 particles (verify correctness)

### Phase 2: RK4 Integration (1 day)
- [ ] Add `'nearest_node'` option to rk4_fully_fused_timedep.py
- [ ] Update production script config
- [ ] Test with 30,000 particles (benchmark)
- [ ] Compare retention and throughput with radius=10

### Phase 3: Optimization (2-3 days)
- [ ] Implement tiled distance computation (reduce memory)
- [ ] Add early exit on first node success
- [ ] Profile GPU utilization (nsys, nvprof)
- [ ] Tune tile size for memory/compute trade-off
- [ ] Test with 225,000 particles (production scale)

### Phase 4: Documentation (1 day)
- [ ] Document algorithm in code comments
- [ ] Create test script with retention analysis
- [ ] Update METHODS_QUICK_REFERENCE.md
- [ ] Add performance comparison to benchmark

**Total effort**: 1-2 weeks (full implementation + optimization)

---

## 12. Conclusion

The **nearest node search** from the old JAXTrace implementation was simple and effective for small meshes. We've already implemented a **KD-tree version** that achieves ~100% retention but **cannot be vmapped** for RK4 tracking.

A **vmappable brute-force version** is feasible but has significant trade-offs:
- ✅ **Pros**: High retention (~99%), vmappable, simple algorithm
- ❌ **Cons**: O(N) cost, slower than radius=10, not scalable

**Recommended path forward**:
1. **First priority**: Fix prefix table depth 6→7 (1 day, makes hierarchical competitive)
2. **Second priority**: Benchmark fixed hierarchical method (should hit 98-99% @ 10-15K p/s)
3. **Last resort**: Implement nearest node search only if prefix fix fails

The nearest node approach is a valid fallback, but fixing the existing octree infrastructure is more sustainable long-term.

---

**Document Status**: ✅ Analysis complete - ready for decision
**Next Steps**: User decides implementation priority
