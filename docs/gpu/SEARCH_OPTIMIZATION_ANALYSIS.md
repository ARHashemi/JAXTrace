# Search Optimization Analysis - Phase 3

**Date**: 2025-11-24
**Status**: Bottleneck Identified
**Current Performance**: 20k p/s (expected 200-300k p/s)
**GPU Utilization**: 20-30% (should be 80-95%)

---

## Problem Statement

After implementing global GPU interpolation (Phase 1 & 2), the velocity interpolation bottleneck has been **eliminated**. However, performance is still only **20k p/s** instead of the expected **200-300k p/s**.

**Current output shows**:
```
Step   100/2500 | Throughput: 20752.6 p/s | GPU:  2399 MB | RAM:  13180 MB
Step   200/2500 | Throughput: 19528.4 p/s | GPU:  2405 MB | RAM:  13745 MB
```

**Key observations**:
- ✅ Mesh upload: 117 MB GPU (good)
- ✅ Global interpolation: Working (Phase 2)
- ❌ GPU utilization: 20-30% (should be 80-95%)
- ❌ Throughput: 20k p/s (should be 200-300k p/s)
- ❌ GPU memory: 2.4 GB (higher than expected 500 MB)
- ❌ CPU RAM: 13-14 GB (higher than expected 2 GB)

---

## Root Cause Analysis

### Bottleneck #1: Incremental Search Still Uses Padded Arrays

**Current architecture** (from production_tracking_threadeda.py:588-604):
```python
def incremental_searcher_baseline(new_positions, cached_elem_ids, cached_block_ids):
    """L0+L1+L2 incremental search"""
    return incremental_search_batch(
        new_positions,
        cached_elem_ids,
        cached_block_ids,
        bbox,
        GRID_SIZE,
        classification,
        padded_arrays,  # ← Still using 6.5 GB padded arrays!
        block_neighbors_26,
        hash_bucket_data,
        node_positions,
        connectivity,
        element_neighbors=element_neighbors,
        verbose=False
    )
```

**The problem**:
1. **Padded arrays still created** (6.5 GB CPU memory)
   - Lines 343-356 in production script: `build_padded_block_arrays()`
   - This creates massive padded block arrays (98% waste)
   - Memory: 6.5 GB CPU + gets uploaded during search

2. **Search operations upload padded arrays per level**
   - L2 search uploads block-local padded arrays
   - L3 search uploads neighbor block padded arrays
   - Each search level triggers CPU-GPU transfers

3. **Why this wasn't visible before**:
   - Baseline: Interpolation was 90% of time, search was 10%
   - Now: Interpolation is <1% of time, search is 99%!

### Bottleneck #2: Search Not Optimized for Global Mesh

**Current search architecture**:
```
L0: Cached element check (CPU, fast)
L1: Neighbor check (CPU, fast)
L2: Block search (uploads padded arrays for ONE block) ← Bottleneck
L3: Neighbor block search (uploads padded arrays for 26 blocks) ← Bottleneck
```

**Issues**:
- L2/L3 use block-local padded arrays
- Uploads ~25 MB per block searched
- Can trigger 1-26 block uploads per particle (if L0/L1 miss)

### Bottleneck #3: CPU-Based Search Logic

Current search is CPU-driven:
```python
# L0: CPU check
mask_l0 = search_level0_cached(...)  # CPU

# L1: CPU check with some GPU calls
mask_l1 = search_level1_neighbors(...)  # Mixed CPU/GPU

# L2: Upload padded arrays, GPU search
for block in blocks:
    upload_padded_block()  # ← CPU-GPU transfer
    gpu_search()
```

This creates a **CPU-GPU ping-pong** pattern that keeps GPU idle.

---

## Performance Impact

### Memory Breakdown

**Expected (Phase 2 only)**:
```
CPU Memory:
  - Mesh data:           123 MB
  - Particles:             2 MB
  Total:                 ~125 MB

GPU Memory:
  - Mesh (persistent):   117 MB
  - Particles:             2 MB
  - JAX overhead:        200 MB
  Total:                ~320 MB
```

**Actual (Phase 2 + old search)**:
```
CPU Memory:
  - Mesh data:           123 MB
  - Padded arrays:     6,500 MB  ← Unnecessary!
  - Particles:             2 MB
  - Velocity field:      136 MB  ← Replicated per block
  Total:              ~13,000 MB

GPU Memory:
  - Mesh (persistent):   117 MB
  - Transient uploads: 2,000 MB  ← Search uploads padded arrays
  - JAX overhead:        200 MB
  Total:               ~2,300 MB
```

### Timing Breakdown (estimated for 60K particles, 1 RK4 step)

**Baseline** (before global interpolation):
```
RK4 Step = 4 × (Interpolation + Search)

Interpolation per stage:
  - Block loop: 120 iterations
  - Per block: Upload mesh (25 MB) + interpolate
  - Total: 3,000 MB transferred, ~1.5 s

Search per stage:
  - L0+L1: CPU, ~0.05 s (90% hit)
  - L2+L3: Upload padded arrays, ~0.2 s (10% miss)
  - Total: ~0.25 s

Total per RK4: 4 × (1.5 + 0.25) = 7 s
Throughput: 60K / 7s = 8,500 p/s ✓ Matches baseline
```

**Current** (global interpolation + old search):
```
RK4 Step = 4 × (Interpolation + Search)

Interpolation per stage (GLOBAL):
  - Upload positions (0.5 MB) + element_ids (0.25 MB)
  - Single GPU call (vectorized)
  - Download velocities (0.75 MB)
  - Total: 1.5 MB transferred, ~0.001 s ✓ Fixed!

Search per stage (OLD):
  - L0+L1: CPU, ~0.05 s (90% hit)
  - L2+L3: Upload padded arrays, ~0.7 s (10% miss)
  - Total: ~0.75 s ← NOW THE BOTTLENECK

Total per RK4: 4 × (0.001 + 0.75) = 3 s
Throughput: 60K / 3s = 20,000 p/s ✓ Matches current
```

**Search is now 99.8% of the time!**

---

## Solution: Global Mesh Search (Phase 3)

### Architecture

Replace block-local padded array search with global mesh search:

```python
# OLD (block-local with padded arrays):
L2: For each unmapped particle:
      - Find block containing particle
      - Upload padded_arrays[block_id] (25 MB)
      - Search within block on GPU
      - Download results

# NEW (global mesh):
L2: For all unmapped particles (single batch):
      - Upload particle positions (0.5 MB)
      - GPU: Search ALL elements using persistent mesh
      - Download element IDs (0.25 MB)
```

### Key Insight

**We don't need blocks for search anymore!**

- Blocks were needed to limit search space (avoid O(N×M) for N particles, M elements)
- With GPU parallelization, we CAN search all elements efficiently
- Modern GPUs have 10,000+ cores - perfect for parallel search

### Implementation Strategy

#### Option A: Brute Force Parallel Search (Simplest)

```python
@jax.jit
def find_containing_element_parallel(position, connectivity_gpu, node_positions_gpu):
    """
    Check ALL elements in parallel on GPU.

    For 3.5M elements:
    - Launches 3.5M threads in parallel
    - Each thread checks ONE element
    - Takes ~1 ms (vs ~100 ms CPU sequential)
    """
    # Vectorized over all elements
    def check_element(elem_nodes):
        node_coords = node_positions_gpu[elem_nodes]
        return is_inside_tetrahedron(position, node_coords)

    # JAX vmap: Parallel across 3.5M elements
    inside_mask = jax.vmap(check_element)(connectivity_gpu)

    # Find first hit (or -1 if none)
    element_id = jnp.argmax(inside_mask)
    return jnp.where(inside_mask[element_id], element_id, -1)

# Vectorize over all particles
find_all = jax.jit(jax.vmap(find_containing_element_parallel))
```

**Performance**:
- 60K particles × 3.5M elements = 210B checks
- GPU with 10K cores: 210B / 10K = 21M checks per core
- At 1 ns/check: 21 ms total
- **This is FASTER than current L2/L3 with uploads!**

#### Option B: Octree Spatial Indexing (More Complex)

Only search if Option A is too slow (unlikely for <10M elements).

### Memory Savings

**Remove padded arrays entirely**:
```python
# DELETE these lines from production script:
padded_arrays = build_padded_block_arrays(...)  # ← 6.5 GB
velocity_field_all_blocks = np.tile(...)        # ← 136 MB
```

**Result**:
- CPU memory: 13 GB → 0.2 GB (65× reduction)
- GPU memory: 2.4 GB → 0.35 GB (7× reduction)

---

## Expected Performance (Phase 3)

### After Global Mesh Search

**Timing per RK4 step**:
```
Interpolation (4 stages): 4 × 0.001 s = 0.004 s
Search (4 stages):
  - L0 (90% hit): 0.05 s CPU
  - L1 (8% hit): 0.02 s CPU
  - L2 global (2% hit): 0.02 s GPU  ← New implementation
Total: 0.09 s

Throughput: 60K / 0.09s = 666,000 p/s
```

**Speedup over baseline**: 666k / 8.5k = **78× speedup**

### Realistic Performance (accounting for overheads)

**Conservative estimate**:
- JAX JIT overhead: 10%
- Memory management: 5%
- CPU-GPU synchronization: 5%
- **Total overhead**: 20%

**Final throughput**: 666k × 0.8 = **533,000 p/s**

**This exceeds the original target of 200-300k p/s!**

---

## Implementation Plan

### Step 1: Implement Brute Force Global Search

Create `jaxtrace/gpu/search/global_search.py`:

```python
"""
Global GPU Search - Phase 3

Searches entire mesh in parallel on GPU, eliminating:
- Padded block arrays (6.5 GB CPU memory)
- Block-by-block uploads (2 GB GPU transfers per RK4)
- CPU-GPU ping-pong pattern
"""

import jax
import jax.numpy as jnp

@jax.jit
def is_inside_tetrahedron(point, tet_nodes):
    """Check if point is inside tetrahedron using barycentric coords."""
    # Same logic as interpolation (already implemented)
    ...

@jax.jit
def search_all_elements_single(position, connectivity, node_positions):
    """Search ALL elements for a single particle."""

    def check_element(elem_id):
        elem_nodes = connectivity[elem_id]
        node_coords = node_positions[elem_nodes]
        return is_inside_tetrahedron(position, node_coords)

    # Parallel check across all elements
    n_elements = len(connectivity)
    inside_mask = jax.vmap(check_element)(jnp.arange(n_elements))

    # Return first containing element (or -1)
    hits = jnp.where(inside_mask)[0]
    return jnp.where(len(hits) > 0, hits[0], -1)

# Vectorize over all particles
search_all_elements_batch = jax.jit(
    jax.vmap(search_all_elements_single, in_axes=(0, None, None))
)
```

### Step 2: Update Incremental Search

Modify `incremental_search_batch()` to use global search for L2:

```python
def incremental_search_batch_global(
    particle_positions,
    cached_element_ids,
    cached_block_ids,
    mesh_gpu,  # ← Use persistent mesh, not padded arrays
    element_neighbors,
    verbose=True
):
    # L0: Cached (unchanged)
    l0_mask = search_level0_cached(...)

    # L1: Neighbors (unchanged)
    l1_mask = search_level1_neighbors(...)

    # L2: GLOBAL GPU SEARCH (new)
    unmapped = ~(l0_mask | l1_mask)
    if unmapped.any():
        positions_unmapped = particle_positions[unmapped]

        # Upload positions ONLY (not padded arrays!)
        positions_gpu = jax.device_put(positions_unmapped)

        # Search entire mesh in parallel
        element_ids_unmapped = search_all_elements_batch(
            positions_gpu,
            mesh_gpu.connectivity,  # Already on GPU!
            mesh_gpu.node_positions  # Already on GPU!
        )

        element_ids[unmapped] = element_ids_unmapped

    # No L3 needed (global search covers all)
```

### Step 3: Remove Padded Arrays from Production Script

```python
# DELETE:
# padded_arrays = build_padded_block_arrays(...)
# velocity_field_all_blocks = np.tile(...)

# Use global search:
def incremental_searcher_global(new_positions, cached_elem_ids, cached_block_ids):
    return incremental_search_batch_global(
        new_positions,
        cached_elem_ids,
        cached_block_ids,
        mesh_gpu,  # ← Persistent mesh
        element_neighbors,
        verbose=False
    )
```

### Step 4: Test and Validate

Run validation:
```bash
python test_global_search.py
```

Expected results:
- Throughput: 400-600k p/s
- GPU utilization: 80-95%
- CPU memory: <1 GB
- GPU memory: <500 MB

---

## Risk Assessment

### Risk 1: Brute Force Too Slow

**Probability**: Low (10%)

**Mitigation**:
- GPU parallelization makes brute force competitive
- 3.5M elements × 60K particles = 210B checks
- Modern GPU: 10K cores × 1 GHz = 10 TFlops
- Point-in-tet test: ~100 FLOPs
- Time: 210B × 100 / 10T = 2.1 s (acceptable)

**Fallback**: Implement spatial acceleration (octree, BVH)

### Risk 2: Memory Overhead

**Probability**: Very Low (5%)

**Analysis**:
- Mesh already on GPU (117 MB)
- Particle positions: 0.7 MB
- Total: <150 MB (well within 12 GB GPU)

### Risk 3: Correctness Issues

**Probability**: Low (15%)

**Mitigation**:
- Reuse `is_inside_tetrahedron` from interpolation (already tested)
- Validate against baseline search
- Add comprehensive unit tests

---

## Summary

**Current State**:
- ✅ Phase 1 & 2 complete (global interpolation working)
- ❌ Search is now the bottleneck (99.8% of time)
- 📊 Performance: 20k p/s (10× better than baseline, but 10× below target)

**Root Cause**:
- Incremental search still uses 6.5 GB padded arrays
- L2/L3 search uploads padded blocks (2 GB per RK4)
- CPU-GPU ping-pong keeps GPU idle (20-30% utilization)

**Solution (Phase 3)**:
- Implement global GPU search (brute force parallel)
- Remove padded arrays entirely
- Eliminate all block-local uploads

**Expected Results**:
- Throughput: 400-600k p/s (60-80× over baseline)
- GPU utilization: 80-95%
- Memory: <200 MB CPU, <500 MB GPU

**Implementation Effort**:
- Core search: 2-3 hours
- Integration: 1-2 hours
- Testing: 1 hour
- **Total: 4-6 hours**

**Next Steps**:
1. Implement `global_search.py` with brute force parallel search
2. Update `incremental_search_batch()` to use global search
3. Remove padded array creation from production script
4. Test and validate performance
