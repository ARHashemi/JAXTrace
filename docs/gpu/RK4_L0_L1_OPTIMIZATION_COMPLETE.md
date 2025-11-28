# RK4 L0+L1 Incremental Search Optimization - Implementation Complete

**Date**: 2025-11-18
**Status**: ✅ **Implementation Complete** - Ready for Testing
**Expected Speedup**: 10-50× for RK4 intermediate searches

---

## Executive Summary

Successfully implemented **L0 (cached element) + L1 (face-neighbor) incremental search** optimization for RK4 time integration. This addresses the critical performance bottleneck where RK4 Full mode was running at only 3-4 particles/second despite 99% GPU utilization.

### Problem Identified

From `logs/time_marching_rk4_FULL_TEST.log`:
- **RK4 Full**: 3-4 p/s (1000 particles)
- **Search time**: 96-98% of total timestep time
- **Root cause**: 4 full searches per timestep (k2, k3, k4, final) using `initial_search_batch()` which skips L0+L1 entirely

### Solution Implemented

Created incremental search that exploits **spatial coherence** in RK4:
- Small displacements (dt/2 or dt) → particles likely stay in same element or move to adjacent neighbor
- **L0 check** (<1 μs/particle): 60-80% hit rate expected
- **L1 check** (~5 μs/particle): 15-25% hit rate expected
- **L2+L3 fallback** (~10ms/particle): Only 5-10% of particles

**Expected Performance**:
- Current: 4 p/s → **Target: 40-200 p/s** (10-50× speedup)

---

## Implementation Details

### 1. Created Incremental Search Module

**File**: [`jaxtrace/gpu/search/incremental_search.py`](../../jaxtrace/gpu/search/incremental_search.py) (335 lines)

**Core Function**: `incremental_search_batch()`

```python
def incremental_search_batch(
    particle_positions: np.ndarray,
    cached_element_ids: np.ndarray,
    cached_block_ids: np.ndarray,
    domain_bounds: np.ndarray,
    grid_size: Tuple[int, int, int],
    block_classification: BlockClassification,
    padded_arrays: PaddedArrays,
    block_neighbors_26: np.ndarray,
    hash_bucket_data: Dict[int, HashBucketArrays],
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    element_neighbors: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, IncrementalSearchStats]:
```

**Search Hierarchy**:
1. **L0**: Vectorized check if particle still in cached element (JAX vmap)
2. **L1**: Check 4 face-adjacent neighbors (if `element_neighbors` available)
3. **L2+L3**: Fall back to full search for remaining particles (reuses `initial_search_batch()`)

**Statistics Tracking**: `IncrementalSearchStats` tracks hit rates per level

### 2. Added Optimized RK4 Function

**File**: [`jaxtrace/gpu/tracking/time_integration.py:317-480`](../../jaxtrace/gpu/tracking/time_integration.py#L317-L480)

**New Function**: `rk4_step_with_incremental_search()`

**Key Features**:
- Uses incremental search for **all 4 RK4 stages** (k2, k3, k4, final)
- Caches element IDs between stages for L0 checks
- Aggregates L0/L1/L2/L3 statistics across all stages
- Compatible with existing `ParticleTimeMarcher` interface

**Performance Estimate**:
```
Stage 2 (k2): 70% L0 + 20% L1 + 10% L2+L3
  = 0.7×1μs + 0.2×5μs + 0.1×10ms ≈ 1ms avg

Full search (old): 4 × 10ms = 40ms
Incremental (new): 4 × 1ms = 4ms
Speedup: ~10×
```

### 3. Element Neighbor Connectivity Support

**File**: [`jaxtrace/gpu/forest/element_adjacency.py:274-353`](../../jaxtrace/gpu/forest/element_adjacency.py#L274-L353)

**New Function**: `build_element_neighbors_array()`

Converts element neighbors from dictionary format to padded array format:
- **Input**: Connectivity array `(N_elements, 4)`
- **Output**: `element_neighbors` array `(N_elements, 4)`, int32, -1 padding
- **Purpose**: Enables L1 face-neighbor search in incremental search

**Usage**:
```python
from jaxtrace.gpu.forest import build_element_neighbors_array

# Build element neighbors from connectivity
element_neighbors = build_element_neighbors_array(connectivity, verbose=True)

# Pass to PaddedArrays during creation
padded_arrays = build_padded_block_arrays(
    element_to_block,
    assignment_stats,
    node_positions=node_positions,
    connectivity=connectivity,
    element_neighbors=element_neighbors,  # <-- NOW AVAILABLE
    verbose=True
)
```

### 4. Module Exports Updated

✅ **Search module**: Added `incremental_search_batch` and `IncrementalSearchStats` to [`jaxtrace/gpu/search/__init__.py`](../../jaxtrace/gpu/search/__init__.py)

✅ **Tracking module**: Added `rk4_step_with_incremental_search` to [`jaxtrace/gpu/tracking/__init__.py`](../../jaxtrace/gpu/tracking/__init__.py)

✅ **Forest module**: Added `build_element_neighbors_array` to [`jaxtrace/gpu/forest/__init__.py`](../../jaxtrace/gpu/forest/__init__.py)

---

## Files Modified/Created

### Created
1. **`jaxtrace/gpu/search/incremental_search.py`** (335 lines)
   - `incremental_search_batch()` - Main L0+L1 optimized search
   - `IncrementalSearchStats` - Statistics dataclass with hit rate tracking

### Modified
2. **`jaxtrace/gpu/tracking/time_integration.py`**
   - Added `rk4_step_with_incremental_search()` function (lines 317-480)

3. **`jaxtrace/gpu/forest/element_adjacency.py`**
   - Added `build_element_neighbors_array()` helper (lines 274-353)

4. **`jaxtrace/gpu/search/__init__.py`**
   - Added exports for incremental search module

5. **`jaxtrace/gpu/tracking/__init__.py`**
   - Added export for `rk4_step_with_incremental_search`

6. **`jaxtrace/gpu/forest/__init__.py`**
   - Added export for `build_element_neighbors_array`

---

## Usage Example

```python
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search import incremental_search_batch
from jaxtrace.gpu.tracking import rk4_step_with_incremental_search

# Step 1: Build element neighbors during preprocessing
element_neighbors = build_element_neighbors_array(connectivity, verbose=True)

# Step 2: Create PaddedArrays with element_neighbors
padded_arrays = build_padded_block_arrays(
    element_to_block,
    assignment_stats,
    node_positions=node_positions,
    connectivity=connectivity,
    element_neighbors=element_neighbors,  # Required for L1 search
    verbose=True
)

# Step 3: Create incremental searcher closure
def incremental_searcher(new_positions, cached_elem_ids, cached_block_ids):
    return incremental_search_batch(
        new_positions,
        cached_elem_ids,
        cached_block_ids,
        domain_bounds,
        grid_size,
        block_classification,
        padded_arrays,
        block_neighbors_26,
        hash_bucket_data,
        node_positions,
        connectivity,
        element_neighbors=element_neighbors,  # Enable L1
        verbose=False
    )

# Step 4: Use optimized RK4 in time marching
new_particle_data, rk4_stats = rk4_step_with_incremental_search(
    particle_data,
    velocity_interpolator,
    incremental_searcher,  # <-- L0+L1 optimized
    dt=0.001,
    current_time=0.0
)

# Step 5: Check hit rates
print(f"L0 hits: {rk4_stats['l0_total_hits']}/{rk4_stats['n_particles']*4}")
print(f"L1 hits: {rk4_stats['l1_total_hits']}/{rk4_stats['n_particles']*4}")
print(f"L2 hits: {rk4_stats['l2_total_hits']}/{rk4_stats['n_particles']*4}")
```

---

## Next Steps

### Immediate (Testing)

1. **Test L0+L1 Hit Rates**
   - Modify existing test to build `element_neighbors`
   - Use `rk4_step_with_incremental_search()` instead of `rk4_step_with_search()`
   - Verify hit rates match expectations (60-80% L0, 15-25% L1)

2. **Measure Performance Speedup**
   - Compare throughput (p/s) vs old RK4 Full
   - Target: 10-50× speedup
   - Expected: 40-200 p/s for 1000 particles (vs 4 p/s baseline)

### Integration (Next Phase)

**Per user's request, the following optimizations should be addressed AFTER L0+L1 testing**:

3. **Fix CPU-GPU Transfer Overhead**
   - Current issue: Block-by-block interpolation with 4-8 CPU↔GPU transfers per call
   - **User guidance**: "All transfers should take place after each batch of particles, not within subprocesses"
   - Refactor velocity interpolation to keep data on GPU
   - Move transfers to batch level instead of subprocess level

4. **Implement Async Data Prefetching**
   - **User guidance**: "Consider Async to prepare required data for next time step before, to have good GPU performance, instead of wait for loading"
   - Prepare next timestep's data while GPU computes current timestep
   - Overlap CPU-GPU transfers with computation

5. **Hierarchy Verification**
   - **User guidance**: Loop hierarchy should be: `time marching → particle batches → blocks`
   - Ensure all data transfers respect this hierarchy

---

## Performance Expectations

### Current Baseline (from logs)
```
Forward Euler:  14 p/s (1000 particles)
RK4 Simplified: 12 p/s (1000 particles)
RK4 Full:       4 p/s  (1000 particles)  ← BOTTLENECK
```

### With L0+L1 Optimization (Expected)
```
RK4 Full (L0+L1): 40-200 p/s (1000 particles)
  - Conservative (10× speedup): 40 p/s
  - Optimistic (50× speedup): 200 p/s
  - Realistic (20× speedup): 80 p/s
```

### Hit Rate Targets
```
L0 (cached):         60-80% (< 1 μs/particle)
L1 (neighbors):      15-25% (~ 5 μs/particle)
L2+L3 (full search): 5-10%  (~10 ms/particle)
```

---

## Technical Notes

### Why L0+L1 Works for RK4

**RK4 Displacement Analysis**:
- dt = 0.001s, velocity ~1 mm/s
- Displacement per stage: 0.0005-0.001 mm
- Element size: ~1-10 mm
- **Most particles stay in same element** across RK4 stages

**Spatial Coherence**:
- k2 position = x + dt/2 × k1  (small displacement)
- k3 position = x + dt/2 × k2  (also small)
- k4 position = x + dt × k3    (still small)

### Element Neighbors Format

**Global format** (required by L1 search):
```python
element_neighbors: np.ndarray  # (N_elements, 4), int32, -1 padding
# Example:
# Element 42 has 3 neighbors: [15, 108, 201]
# Row 42: [15, 108, 201, -1]

# Element 7 is fully interior with 4 neighbors
# Row 7: [3, 9, 12, 18]
```

**Block-local format** (stored in PaddedArrays):
```python
padded_arrays.element_neighbors: np.ndarray  # (n_blocks, max_elem, 4), int32, -1 padding
# This is built automatically by build_padded_block_arrays() when
# element_neighbors is provided
```

### JAX Compatibility

All search functions are GPU-native:
- L0 uses `jax.vmap` for vectorized element checks
- L1 uses JAX JIT compiled neighbor search
- L2+L3 reuses existing GPU-accelerated search

---

## Testing Checklist

- [x] ✅ Create `incremental_search.py` module
- [x] ✅ Implement `incremental_search_batch()` with L0+L1
- [x] ✅ Create `rk4_step_with_incremental_search()` function
- [x] ✅ Add `build_element_neighbors_array()` helper
- [x] ✅ Update all module exports
- [ ] ⏳ Test element neighbor building on ThreadedA mesh
- [ ] ⏳ Test incremental search L0/L1 hit rates
- [ ] ⏳ Benchmark RK4 with incremental search vs full search
- [ ] ⏳ Verify correctness (particle trajectories match)
- [ ] ⏳ Measure actual speedup factor

---

## References

**Related Files**:
- [`jaxtrace/gpu/search/level0_cached.py`](../../jaxtrace/gpu/search/level0_cached.py) - L0 cached element search
- [`jaxtrace/gpu/search/level1_neighbors.py`](../../jaxtrace/gpu/search/level1_neighbors.py) - L1 neighbor search
- [`jaxtrace/gpu/search/initial_assignment.py`](../../jaxtrace/gpu/search/initial_assignment.py) - L2+L3 full search (reused)

**Documentation**:
- [`docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md`](BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md) - Overall architecture
- [`logs/time_marching_rk4_FULL_TEST.log`](../../logs/time_marching_rk4_FULL_TEST.log) - Baseline performance data

---

## Summary

The L0+L1 incremental search optimization is **fully implemented** and ready for testing. This addresses the user's primary request to "utilize L0+L1 in RK4" and should provide a **10-50× speedup** for RK4 intermediate searches.

The implementation is modular, well-documented, and integrates cleanly with existing code. Next steps are to test hit rates and performance, then move on to the CPU-GPU transfer optimizations and async data prefetching as requested by the user.
