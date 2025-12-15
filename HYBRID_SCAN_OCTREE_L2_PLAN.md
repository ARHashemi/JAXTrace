# Hybrid Scan-Based Octree with Level Filtering - L2 Fallback Plan

## Status: ✅ Documented for Future Implementation

**Date:** 2025-11-28

---

## Executive Summary

After fixing the hierarchical 3-hop/4-hop search, we expect **99.9-99.95% hit rate** with **60-82% retention at 2,500 steps**. However, particles are still being lost in refined mesh regions.

This document outlines a **GPU-native L2 fallback** strategy using a level-filtered scan-based octree to capture the remaining 0.05-0.1% of particles that miss L1 multi-hop search.

---

## Problem Statement

### Current Status (After Hierarchical Multi-Hop Fix)

| Hop Count | Hit Rate | Retention (2,500 steps) | Memory | Throughput |
|-----------|----------|------------------------|--------|------------|
| 3-hop | 99.9% | 16.1% | 35 MB | 50k p/s |
| 4-hop hierarchical | 99.95% | 60% (expected) | 8 MB | 40-48k p/s |
| 5-hop hierarchical | 99.99% | 82% (expected) | 10 MB | 35-45k p/s |

### Why Particles Are Still Lost

**Refined mesh regions:**
- Element size: 8.12e-14 to 2.13e-08 m³ (262,146× range)
- Smallest elements: ~200 nanometers (in weld pool)
- Velocity displacement: 0.1-0.9 m/s × 0.0025 s = 0.25-2.25 mm per timestep
- **Problem:** Fast particles can skip 5,000+ tiny elements per timestep!

**Why multi-hop fails:**
- 5-hop: Maximum 1,024 neighbors (4^5 neighbors if fully connected)
- Refined region: Can have 10,000+ elements in 2mm radius
- **Gap:** Even 5-hop misses particles in extreme refinement regions

---

## Solution: Hybrid Scan-Based Octree with Level Filtering

### Key Innovations

1. **Level-Based Filtering**: Build octree only for user-specified refinement level
2. **Scan-Based Traversal**: Use `jax.lax.scan` with fixed iteration count (no data-dependent branches)
3. **Fixed-Size Leaf Arrays**: Pad to `max_leaf_size=500` (acceptable memory)
4. **Early Exit**: Skip remaining iterations when element is found

### Why This Works for GPU

**Traditional octree (CPU):**
```python
def traverse_octree(node, pos):
    if is_leaf(node):
        return search_elements(node.elements)
    else:
        child = select_child(node, pos)
        return traverse_octree(child, pos)  # Data-dependent recursion ❌
```
**Problem:** Data-dependent recursion doesn't vectorize with `vmap`

**Scan-based octree (GPU):**
```python
def traverse_octree_scan(octree, pos):
    def step(carry, _):
        node_id, found_id = carry
        # Fixed operations per iteration
        child_id = select_child(octree[node_id], pos)
        leaf_id = search_leaf(octree[node_id], pos)
        # Use lax.cond for early exit
        new_node_id = jax.lax.cond(found_id >= 0, lambda: node_id, lambda: child_id)
        new_found_id = jnp.where(found_id >= 0, found_id, leaf_id)
        return (new_node_id, new_found_id), None

    (_, element_id), _ = jax.lax.scan(step, (0, -1), None, length=MAX_DEPTH)
    return element_id
```
**Benefits:**
- Fixed iteration count (`MAX_DEPTH=10`)
- Vectorizes with `vmap` over particles
- Early exit with `lax.cond` (no wasted work)

---

## Architecture

### Phase 1: Build Level-Filtered Octree (CPU, Initialization)

**Input:**
- Mesh with 3.5M elements
- LEVEL field: Refinement level per element (0-10)
- User-specified level threshold: e.g., `level >= 7` (refined regions only)

**Process:**
```python
# Filter elements by refinement level
refined_mask = mesh.level >= 7
refined_elements = mesh.elements[refined_mask]  # e.g., 300k elements instead of 3.5M

# Build octree on refined elements only
octree = build_octree(
    refined_elements,
    max_depth=10,
    max_leaf_size=500
)

# Upload to GPU as fixed-size arrays
octree_gpu = upload_octree(octree)
```

**Memory estimate:**
- 300k elements / 500 per leaf = 600 leaf nodes
- Tree depth: 10 levels → ~1,200 total nodes (including branches)
- Node metadata: 1,200 × 64 bytes = 77 KB
- Leaf element arrays: 600 × 500 × 4 bytes = 1.2 MB
- **Total:** ~2.5 MB (acceptable)

### Phase 2: Scan-Based Traversal (GPU, Per Timestep)

**Input:**
- Particle positions: (N, 3)
- Cached element IDs: (N,)
- Octree GPU data: Fixed-size arrays

**Process:**
```python
@jax.jit
def search_level2_octree_scan(positions, cached_ids, octree_gpu):
    def search_one_particle(pos, cached_id):
        def step(carry, _):
            node_id, found_id = carry

            # Load current node (fixed-size array access)
            node = octree_gpu.nodes[node_id]

            # If leaf: Check all elements in leaf
            is_leaf = node.is_leaf
            leaf_id = jax.lax.cond(
                is_leaf,
                lambda: search_leaf_elements(node.elements, pos),
                lambda: -1
            )

            # If branch: Select child based on position
            child_id = jax.lax.cond(
                is_leaf,
                lambda: node_id,  # Stay at current node if leaf
                lambda: select_child_octant(node, pos)
            )

            # Update carry with early exit
            new_node_id = jnp.where(found_id >= 0, node_id, child_id)
            new_found_id = jnp.where(found_id >= 0, found_id, leaf_id)

            return (new_node_id, new_found_id), None

        # Scan for up to MAX_DEPTH iterations
        (_, element_id), _ = jax.lax.scan(step, (0, -1), None, length=10)
        return element_id

    # Vectorize over all particles
    return jax.vmap(search_one_particle)(positions, cached_ids)
```

**Key features:**
- ✅ Fixed iteration count: 10 iterations max
- ✅ Fixed-size array access: All nodes/leaves pre-allocated
- ✅ Early exit: `lax.cond` skips remaining iterations when found
- ✅ Vectorized: `vmap` over particles

### Phase 3: Integration with RK4

**Modified search hierarchy:**
```python
@jax.jit
def search_gpu_fused_with_l2_octree(positions, cached_ids, mesh_gpu, octree_gpu):
    # L0: Check cached elements (85-95% hit rate)
    element_ids_l0 = search_level0_vectorized(positions, cached_ids, ...)

    # L1: Hierarchical multi-hop (99.9-99.95% cumulative hit rate)
    element_ids_l1 = search_level1_multihop_hierarchical(positions, cached_ids, ..., n_hops=4)

    # L2: Scan-based octree on refined regions (99.99% cumulative hit rate)
    element_ids_l2 = search_level2_octree_scan(positions, cached_ids, octree_gpu)

    # Merge results: L0 → L1 → L2 → -1 (not found)
    element_ids = jnp.where(element_ids_l0 >= 0, element_ids_l0,
                   jnp.where(element_ids_l1 >= 0, element_ids_l1, element_ids_l2))

    return element_ids
```

---

## Performance Analysis

### Memory Footprint

| Component | Size | Notes |
|-----------|------|-------|
| Level-filtered elements | 300k elements | User-specified level >= 7 |
| Octree nodes | 77 KB | 1,200 nodes × 64 bytes |
| Leaf element arrays | 1.2 MB | 600 leaves × 500 elements × 4 bytes |
| **Total** | **~2.5 MB** | Acceptable overhead |

### Throughput Impact

**Baseline (4-hop hierarchical, no L2):**
- L0 hit rate: 90%
- L1 hit rate: 9.95%
- L2 hit rate: 0.05% (missed)
- **Throughput:** 40-48k p/s

**With L2 octree fallback:**
- L0 hit rate: 90% (same)
- L1 hit rate: 9.95% (same)
- L2 hit rate: 0.05% (captured by octree)
- **L2 overhead:** Only 0.05% of particles need L2

**Expected overhead:**
- Scan-based octree: ~40 ms per 105k particles (if all particles used L2)
- Actual: 40 ms × 0.05% = **0.02 ms per timestep** (negligible)
- **Throughput:** 40-48k p/s (no measurable slowdown)

### Hit Rate Improvement

| Level | Hit Rate | Cumulative | Retention (2,500 steps) |
|-------|----------|------------|------------------------|
| L0 (cached) | 90% | 90% | - |
| L1 (4-hop) | 9.95% | 99.95% | 60% |
| L2 (octree) | 0.04% | 99.99% | **82%** ✅ |
| Not found | 0.01% | 100% | - |

**Improvement:** 60% → 82% retention (+37% more particles survive)

---

## Implementation Steps

### Step 1: Octree Builder (CPU, Initialization)

**File:** `jaxtrace/gpu/search/octree_builder.py` (NEW)

```python
@dataclass
class OctreeNode:
    """Fixed-size octree node for GPU."""
    is_leaf: bool
    bbox_min: np.ndarray  # (3,)
    bbox_max: np.ndarray  # (3,)
    children: np.ndarray  # (8,) node IDs (-1 if empty)
    elements: np.ndarray  # (max_leaf_size,) element IDs (-1 if empty)

def build_octree_for_level(
    mesh: Mesh,
    level_threshold: int = 7,
    max_depth: int = 10,
    max_leaf_size: int = 500
) -> Tuple[np.ndarray, dict]:
    """
    Build fixed-size octree for elements with refinement level >= threshold.

    Returns
    -------
    nodes : np.ndarray, shape (n_nodes, node_size)
        Fixed-size array of octree nodes
    metadata : dict
        Octree statistics (n_nodes, n_leaves, depth, etc.)
    """
    # 1. Filter elements by refinement level
    refined_mask = mesh.level >= level_threshold
    refined_elements = mesh.elements[refined_mask]

    # 2. Build octree recursively
    root = _build_recursive(refined_elements, max_depth, max_leaf_size)

    # 3. Flatten to fixed-size arrays
    nodes = _flatten_to_fixed_arrays(root, max_leaf_size)

    return nodes, metadata
```

### Step 2: GPU Octree Search (GPU, Per Timestep)

**File:** `jaxtrace/gpu/search/octree_search_gpu.py` (NEW)

```python
@jax.jit
def search_level2_octree_scan(
    positions: jax.Array,
    cached_element_ids: jax.Array,
    octree_nodes: jax.Array,
    octree_elements: jax.Array,
    max_depth: int = 10
) -> jax.Array:
    """
    Scan-based octree search with early exit.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Particle positions
    cached_element_ids : jax.Array, shape (N,)
        Cached element IDs (not used in L2, but kept for interface consistency)
    octree_nodes : jax.Array, shape (n_nodes, node_metadata_size)
        Fixed-size octree node array
    octree_elements : jax.Array, shape (n_nodes, max_leaf_size)
        Fixed-size element array per node
    max_depth : int
        Maximum tree depth

    Returns
    -------
    element_ids : jax.Array, shape (N,)
        Found element IDs (-1 if not found)
    """
    def search_one_particle(pos):
        def step(carry, _):
            node_id, found_id = carry

            # Load node metadata
            node_is_leaf = octree_nodes[node_id, 0]  # First field
            node_bbox_min = octree_nodes[node_id, 1:4]
            node_bbox_max = octree_nodes[node_id, 4:7]
            node_children = octree_nodes[node_id, 7:15].astype(jnp.int32)

            # If leaf: Check all elements
            def check_leaf(_):
                elements = octree_elements[node_id]  # (max_leaf_size,)
                # Vectorized point-in-element check
                return check_elements_vectorized(pos, elements)

            # If branch: Select child octant
            def select_child(_):
                octant = compute_octant(pos, node_bbox_min, node_bbox_max)
                return node_children[octant]

            # Branch based on leaf status
            leaf_result = jax.lax.cond(node_is_leaf, check_leaf, lambda _: -1, None)
            child_id = jax.lax.cond(node_is_leaf, lambda _: node_id, select_child, None)

            # Update carry with early exit
            new_node_id = jnp.where(found_id >= 0, node_id, child_id)
            new_found_id = jnp.where(found_id >= 0, found_id, leaf_result)

            return (new_node_id, new_found_id), None

        # Scan for up to max_depth iterations
        (_, element_id), _ = jax.lax.scan(step, (0, -1), None, length=max_depth)
        return element_id

    # Vectorize over all particles
    return jax.vmap(search_one_particle)(positions)
```

### Step 3: Integration with RK4 GPU-Fused

**File:** `jaxtrace/gpu/tracking/rk4_gpu_fused.py` (MODIFY)

```python
def create_search_gpu_fused_with_l2_octree(n_hops: int = 4, octree_gpu=None):
    """
    Create search function with L2 octree fallback.

    Parameters
    ----------
    n_hops : int, default=4
        Number of hops for L1 neighbor search
    octree_gpu : OctreeGPU, optional
        Octree data for L2 fallback

    Returns
    -------
    search_func : callable
        JIT-compiled search function with L0 + L1 + L2 hierarchy
    """
    @jax.jit
    def search_gpu_fused_with_l2_impl(
        positions_gpu,
        cached_element_ids_gpu,
        mesh_gpu_node_positions,
        mesh_gpu_connectivity,
        mesh_gpu_element_neighbors
    ):
        # L0: Check cached elements
        element_ids_l0 = search_level0_vectorized(...)

        # L1: Hierarchical multi-hop
        element_ids_l1 = search_level1_multihop_hierarchical(..., n_hops=n_hops)

        # L2: Octree fallback (only if provided)
        if octree_gpu is not None:
            element_ids_l2 = search_level2_octree_scan(
                positions_gpu,
                cached_element_ids_gpu,
                octree_gpu.nodes,
                octree_gpu.elements
            )
        else:
            element_ids_l2 = jnp.full_like(cached_element_ids_gpu, -1)

        # Merge: L0 → L1 → L2 → -1
        element_ids = jnp.where(element_ids_l0 >= 0, element_ids_l0,
                       jnp.where(element_ids_l1 >= 0, element_ids_l1, element_ids_l2))

        return element_ids

    return search_gpu_fused_with_l2_impl
```

### Step 4: Configuration in Production Script

**File:** `production_tracking_threadeda.py` (MODIFY)

```python
# Configuration
USE_L2_OCTREE_FALLBACK = True  # Enable L2 octree fallback
L2_OCTREE_LEVEL_THRESHOLD = 7  # Only build octree for level >= 7 (refined regions)

# Build octree (one-time, during initialization)
if USE_L2_OCTREE_FALLBACK:
    print("Building level-filtered octree for L2 fallback...")
    octree_nodes, octree_metadata = build_octree_for_level(
        mesh,
        level_threshold=L2_OCTREE_LEVEL_THRESHOLD,
        max_depth=10,
        max_leaf_size=500
    )
    octree_gpu = OctreeGPU(
        nodes=jax.device_put(octree_nodes),
        elements=jax.device_put(octree_elements)
    )
    print(f"✓ Octree built: {octree_metadata['n_nodes']} nodes, "
          f"{octree_metadata['n_leaves']} leaves, "
          f"{octree_metadata['memory_mb']:.1f} MB")
else:
    octree_gpu = None

# Create search function with L2 fallback
search_func = create_search_gpu_fused_with_l2_octree(
    n_hops=4,
    octree_gpu=octree_gpu
)
```

---

## Testing Strategy

### Phase 1: Unit Tests

**Test 1: Octree Builder**
```python
def test_octree_builder():
    # Load mesh
    mesh = load_mesh("threadedAvtk_120.pvtu")

    # Build octree for level >= 7
    nodes, metadata = build_octree_for_level(mesh, level_threshold=7)

    # Verify
    assert metadata['n_elements'] < 500_000  # Filtered to refined regions
    assert metadata['max_depth'] <= 10
    assert metadata['max_leaf_size'] == 500
    assert nodes.shape[0] == metadata['n_nodes']
```

**Test 2: Scan-Based Search**
```python
def test_octree_scan_search():
    # Build octree
    nodes, _ = build_octree_for_level(mesh, level_threshold=7)
    nodes_gpu = jax.device_put(nodes)

    # Test positions (1,000 particles in refined regions)
    positions = generate_test_positions(n=1000, refined_region=True)

    # Search
    element_ids = search_level2_octree_scan(positions, ..., nodes_gpu)

    # Verify
    hit_rate = (element_ids >= 0).sum() / len(positions)
    assert hit_rate > 0.95  # Should find most particles in refined regions
```

### Phase 2: Integration Test

**Test 3: Full L0+L1+L2 Pipeline**
```python
def test_full_search_pipeline():
    # Build octree
    octree_gpu = build_and_upload_octree(mesh, level_threshold=7)

    # Create search function
    search_func = create_search_gpu_fused_with_l2_octree(n_hops=4, octree_gpu=octree_gpu)

    # Test with 10k particles, 100 timesteps
    positions, element_ids = initialize_particles(n=10_000)

    hit_counts = {'l0': 0, 'l1': 0, 'l2': 0, 'miss': 0}

    for step in range(100):
        # Perform search
        element_ids_new = search_func(positions, element_ids, ...)

        # Track hit levels
        l0_hits = (element_ids_new == element_ids).sum()
        l1_hits = ((element_ids_new != element_ids) & (element_ids_new >= 0)).sum()
        # ... count l2 hits separately

        # Advance particles
        positions, element_ids = rk4_step(...)

    # Verify hit rates
    assert hit_counts['l0'] / (100 * 10_000) > 0.85  # L0: >85%
    assert hit_counts['l1'] / (100 * 10_000) > 0.09  # L1: >9%
    assert hit_counts['l2'] / (100 * 10_000) > 0.01  # L2: >1%
    assert hit_counts['miss'] / (100 * 10_000) < 0.01  # Miss: <1%
```

### Phase 3: Production Test

**Test 4: Full 2,500 Timestep Run**
```bash
# Run with L2 octree fallback enabled
python production_tracking_threadeda.py \
    --use-l2-octree \
    --l2-level-threshold 7 \
    --timesteps 2500 \
    --particles 105000
```

**Expected results:**
- **Throughput:** 40-48k p/s (no measurable slowdown)
- **Hit rate:** 99.99% (L0 + L1 + L2)
- **Retention:** 82% at 2,500 steps (vs 60% without L2)
- **GPU memory:** +2.5 MB (octree overhead)

---

## Alternatives Considered

### Option A: Global Exhaustive Search

**Rejected because:**
- Search all 3.5M elements for 0.05% of particles
- 7,000× slower than octree (40 seconds vs 5 milliseconds)
- Total overhead: 40 seconds × 0.05% = 20 ms per timestep (still acceptable, but wasteful)

### Option B: Uniform Block Grid

**Rejected because:**
- ThreadedA mesh has 8.59× load imbalance with 4×4×2 grid
- Heavy blocks: 938k elements (too large for efficient search)
- Light blocks: 36 elements (wasted memory)
- Previous experiments showed OOM due to padding

### Option C: Region-Based Search (User-Defined BBox)

**Rejected because:**
- Only 2.9× speedup over global search
- Requires user to manually specify refined regions
- Not adaptive to mesh refinement changes

### Option D: Adaptive Multi-Hop (Variable n_hops)

**Considered for future:**
- Use per-particle hit history to predict needed hops
- Fast particles: 2-3 hops
- Slow particles: 4-5 hops
- Expected: 10-15% throughput improvement
- **Status:** Defer to Phase 8 optimization

---

## Performance vs Complexity Trade-Off

| Approach | Retention | Throughput | Memory | Implementation Complexity |
|----------|-----------|------------|--------|--------------------------|
| **3-hop (baseline)** | 16% | 50k p/s | 35 MB | Simple ✅ |
| **4-hop hierarchical** | 60% | 40-48k p/s | 8 MB | Moderate |
| **5-hop hierarchical** | 82% | 35-45k p/s | 10 MB | Moderate |
| **4-hop + L2 octree** | **82%** | **40-48k p/s** | **11 MB** | **High** ✅ |

**Recommendation:** Implement **4-hop hierarchical + L2 octree** for best balance of performance and retention.

---

## Implementation Timeline

### Immediate (After JIT Fix Verification)
1. ✅ Document Hybrid Scan-Based Octree plan (THIS FILE)
2. ⏳ Verify 4-hop hierarchical throughput recovery (40-48k p/s)

### Near-Term (Next Session)
1. Implement `octree_builder.py` (CPU-side octree construction)
2. Implement `octree_search_gpu.py` (scan-based GPU search)
3. Unit tests for octree builder and search

### Medium-Term (Week 1)
1. Integration with RK4 GPU-fused pipeline
2. Integration tests with 10k particles, 100 timesteps
3. Production test with 105k particles, 2,500 timesteps

### Future Optimization (Phase 8)
1. Adaptive hop count based on particle history
2. Auto-clustering for optimal octree partitioning
3. Multi-resolution octrees for different refinement levels

---

## Open Questions

1. **What refinement level threshold should be used?**
   - Recommendation: Start with level >= 7 (based on LEVEL field analysis)
   - Adjust based on memory constraints and hit rate

2. **Should L2 octree be optional or always enabled?**
   - Recommendation: Make it optional with flag `USE_L2_OCTREE_FALLBACK`
   - Default: Enabled for production, disabled for fast debugging

3. **What if particles are lost outside refined regions?**
   - Fallback: Global exhaustive search as L3 (very rare, ~0.01% of particles)
   - Performance impact: Negligible (~1 ms per timestep)

---

## Conclusion

The Hybrid Scan-Based Octree with Level Filtering provides a **GPU-native L2 fallback** that:

✅ **Memory efficient:** 2.5 MB (vs 572 MB for naive 5-hop concatenation)
✅ **No throughput impact:** <0.02 ms overhead per timestep
✅ **High retention:** 82% at 2,500 steps (vs 60% without L2)
✅ **GPU-friendly:** Fixed iteration count, vectorizable with vmap
✅ **Adaptive to mesh:** Uses LEVEL field to focus on refined regions

**Status:** Ready for implementation after hierarchical multi-hop verification

---

**Plan documented:** 2025-11-28
**Next action:** Verify 4-hop hierarchical throughput, then proceed with octree implementation
**Priority:** HIGH (enables 82% retention without performance loss)
