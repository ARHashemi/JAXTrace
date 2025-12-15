# HOT Morton Global L2 - Ready to Implement

**Date**: 2025-12-12
**Status**: ✅ PLAN FINALIZED - Ready for Phase-by-Phase Implementation

---

## Executive Summary

The original HOT Morton implementation was **fundamentally wrong** - it used a block-based approach with 256 cube-aligned blocks, which doesn't match your HOT design philosophy.

**Correct Design**: Single global Morton-sorted element list divided into fixed-capacity leaf segments (no blocks).

The revised plan has been finalized with proper understanding of:
1. **Global Morton structure** (no blocks)
2. **JAX vmap/jit constraints** (single jit, single vmap)
3. **Current RK4 architecture** (factory → wrapper → jitted function)

---

## Key Corrections Made

### ❌ What Was Wrong

**Original Implementation**:
```python
# WRONG - Used 256 blocks
grid_size = (8, 8, 4)  # 256 blocks
MAX_ELEMENTS_PER_BLOCK = 50000

def build_cube_aligned_blocks(...):
    # Per-block Morton sorting
    # Per-block octree leaves
    # Block-based search
```

**Problems**:
- Unnecessary complexity with 256 blocks
- Block size limits causing errors (474k elements > 50k limit)
- Doesn't match HOT philosophy
- Files: `hot_morton_builder.py`, `hot_morton_search.py` entirely wrong

### ✅ What Is Correct

**Revised Implementation**:
```python
# CORRECT - Single global structure
morton_codes = compute_morton_for_centroids(...)
sorted_indices = np.argsort(morton_codes)
elem_ids_sorted = elements[sorted_indices]

# Divide into fixed-size segments (leaves)
leaf_start[i] = i * leaf_capacity
leaf_length[i] = min(leaf_capacity, n_elements - i * leaf_capacity)

# GPU search: position → Morton code → leaf ID → bounded search
```

**Advantages**:
- Simple, elegant design
- No block management overhead
- ~500 lines vs ~2000 lines
- True HOT philosophy
- JAX-friendly (fixed shapes, bounded loops)

---

## Architecture Overview

### Phase 1: CPU Preprocessing
```
node_positions, connectivity
    ↓
Compute element centroids
    ↓
Compute Morton codes (bit interleaving)
    ↓
Global sort by Morton code
    ↓
Divide into fixed-capacity leaves (128-256 elements)
    ↓
elem_ids_sorted[N], leaf_start[L], leaf_length[L]
```

### Phase 2: GPU Search
```
Particle position (x, y, z)
    ↓
Compute Morton code
    ↓
Map to leaf ID (linear approximation)
    ↓
Get leaf offset: start = leaf_start[leaf_id]
    ↓
Search elem_ids_sorted[start:start+capacity] with bounded loop
    ↓
Element ID or -1
```

### Phase 3: RK4 Integration
```
Factory: create_rk4_step_gpu_fused_global_morton(mesh_morton_gpu)
    ↓
Wrapper: rk4_step_global_morton_impl(particle_data, velocity_field, dt, mesh_gpu)
    ↓
Jitted RK4: rk4_fused_global_morton(...)
    ↓
    For each stage (k1, k2, k3, k4, final):
        ↓
        search_l0_l1_l2_global_morton(...)
            ↓
            L0: search_level0_vectorized(...) [KEEP - already vectorized]
            ↓
            L1: search_level1_multihop_vectorized(...) [KEEP - already vectorized]
            ↓
            L2: jax.vmap(search_l2_single)(...) [NEW - single vmap for L2]
        ↓
        interpolate_velocity_batch_gpu(...)
```

---

## Critical JAX Constraints

### ⚠️ Nested JIT/VMAP Rules

**CORRECT Pattern**:
```python
# Factory function (NOT jitted)
def create_rk4_step_gpu_fused_global_morton(...):

    # Wrapper (NOT jitted)
    def rk4_step_global_morton_impl(...):

        # Inner RK4 (JITTED ONCE)
        @jax.jit
        def rk4_fused_global_morton(...):
            # Calls search functions (NOT jitted)
            element_ids = search_l0_l1_l2_global_morton(...)
            ...

        # Call jitted function
        result = rk4_fused_global_morton(...)
```

**Key Rules**:
1. ✅ **ONE JIT** at `rk4_fused_global_morton` level
2. ✅ **ONE VMAP** for L2 inside `search_l0_l1_l2_global_morton`
3. ❌ **NO JIT** on search functions (called from within jitted RK4)
4. ❌ **NO nested vmap** (L0/L1 already vectorized, L2 uses single vmap)

---

## Implementation Roadmap

### Phase 1: CPU Preprocessing (4-6 hours)

**File**: `jaxtrace/gpu/search/morton_global_builder.py` (NEW)

**Functions**:
- `interleave_bits_3d()` - 3D Morton encoding
- `compute_morton_codes_for_elements()` - Batch Morton computation
- `build_global_morton_sorted_list()` - Global sort
- `build_fixed_capacity_leaves()` - Leaf segmentation (uniform chunks)
- `build_global_morton_structure()` - End-to-end pipeline

**Output**: `GlobalMortonStructure` dataclass
- `elem_ids_sorted` (N,) int32
- `leaf_start` (L,) int32
- `leaf_length` (L,) int32
- Morton range (min, max)
- Mesh bounds

**Test**: Load ThreadedA mesh (3.5M elements), build structure, verify:
- All elements in exactly one leaf
- No leaf exceeds capacity
- Sorted order preserved

---

### Phase 2: GPU Search (6-8 hours)

**File**: `jaxtrace/gpu/search/morton_global_search.py` (NEW)

**Functions**:
- `interleave_bits_3d_jax()` - JAX Morton encoding
- `morton_encode_position_jax()` - Position → Morton code
- `position_to_leaf_id_linear()` - Morton code → leaf ID (linear approximation)
- `search_in_leaf_global()` - Bounded loop (`lax.fori_loop`) over leaf
- `search_L2_global_morton_single()` - Complete L2 for single particle (NO @jax.jit)
- `upload_global_morton_to_gpu()` - Upload structure to GPU

**Output**: `MeshGPUGlobalMorton` dataclass on GPU

**Test**: Unit test each function
- Morton encoding matches CPU version
- Leaf mapping finds correct leaf
- Bounded search finds element (compare vs brute force)

---

### Phase 3: RK4 Integration (4-6 hours)

**File**: `jaxtrace/gpu/tracking/rk4_gpu_fused.py` (MODIFY)

**Add Functions**:
- `search_l0_l1_l2_global_morton()` (NO @jax.jit)
  - Call existing `search_level0_vectorized`
  - Call existing `search_level1_multihop_vectorized`
  - Define `search_l2_single`, vmap it (ONE vmap)
- `create_rk4_step_gpu_fused_global_morton()` (factory)
  - Wrapper function (not jitted)
  - Inner jitted `rk4_fused_global_morton` (JITTED ONCE)
  - 5 search calls (k1, k2, k3, k4, final)

**Test**: Single timestep with 1K particles
- Verify no nested jit/vmap
- Verify no JAX OOM
- Verify element IDs valid

---

### Phase 4: Testing & Validation (6-8 hours)

**Validation Test**: `test_global_morton_validation.py`
- 1,000 particles (10×10×10 grid)
- 1 timestep
- ThreadedA mesh (3.5M elements)
- **Target**: >95% initial assignment, no OOM

**Production Test**: `production_tracking_global_morton.py`
- 105,000 particles (50×70×30 grid)
- 2,500 timesteps
- ThreadedA mesh
- **Target**: >95% retention, 40-50k p/s throughput

**Comparison**: vs Hierarchical 5-hop (91% retention baseline)

---

## Files to Create/Modify

### ✅ CREATE (NEW)
1. `jaxtrace/gpu/search/morton_global_builder.py` (~300 lines)
2. `jaxtrace/gpu/search/morton_global_search.py` (~250 lines)
3. `test_global_morton_validation.py` (~200 lines)
4. `production_tracking_global_morton.py` (~400 lines)

### ✏️ MODIFY (ADD TO)
5. `jaxtrace/gpu/tracking/rk4_gpu_fused.py` (add ~200 lines)

### 🗑️ DELETE (OPTIONAL - for cleanup)
6. `jaxtrace/gpu/search/hot_morton_builder.py` (wrong implementation)
7. `jaxtrace/gpu/search/hot_morton_search.py` (wrong implementation)
8. `test_hot_morton_validation.py` (uses wrong implementation)
9. `production_tracking_3hop_l2_hot_morton.py` (uses wrong implementation)

---

## What to Keep from Current Code

### ✅ KEEP (No Changes Required)

1. **Velocity Interpolation** (`jaxtrace/gpu/tracking/interpolation.py`)
   - `interpolate_velocity_batch_gpu()`
   - Barycentric coordinates
   - All interpolation kernels

2. **L0 Cached Search** (`jaxtrace/gpu/search/level0_cached.py`)
   - `search_level0_vectorized()`
   - `point_in_tet_jax()`

3. **L1 Multi-hop Neighbors** (`jaxtrace/gpu/search/incremental_search_vectorized.py`)
   - `search_level1_multihop_vectorized()`
   - `search_level1_multihop_hierarchical()`
   - All neighbor traversal logic

4. **RK4 Structure** (general pattern)
   - 4-stage RK4 computation
   - Stage-by-stage search + interpolation
   - Upload/compute/download pattern

5. **Mesh Loading** (`jaxtrace/gpu/mesh_loader.py`)
   - PVTU loading
   - Connectivity/node handling
   - Velocity field extraction

6. **Supporting Utilities**
   - `ParticleData` dataclass
   - Element neighbor building
   - Seeding functions

---

## Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| **Memory Overhead** | <200 MB | Global Morton arrays |
| **Initial Assignment** | >95% | L2 global Morton |
| **Final Retention (2.5K steps)** | >95% | L0+L1+L2 combined |
| **Throughput** | 40-50k p/s | Similar to Phase 3a |
| **L0 Hit Rate** | 85-95% | Same as baseline |
| **L1 Hit Rate** | 99-99.5% | 3-hop neighbors |
| **L2 Hit Rate** | >99.9% | Global Morton |
| **OOM Risk** | None | Fixed-size arrays, bounded loops |

---

## Implementation Order

### Day 1: Phase 1 (CPU Preprocessing)
1. ✅ Create `morton_global_builder.py`
2. ✅ Implement Morton encoding utilities
3. ✅ Implement global sort and leaf segmentation
4. ✅ Test on ThreadedA mesh
5. ✅ Verify all elements covered, no overflow

### Day 2: Phase 2 (GPU Search)
1. ✅ Create `morton_global_search.py`
2. ✅ Implement JAX Morton encoding
3. ✅ Implement position → leaf mapping
4. ✅ Implement bounded leaf search
5. ✅ Unit test each function
6. ✅ Upload structure to GPU

### Day 3: Phase 3 (RK4 Integration)
1. ✅ Add search function to `rk4_gpu_fused.py`
2. ✅ Add factory function
3. ✅ Verify no nested jit/vmap
4. ✅ Test with 1K particles, 1 step
5. ✅ Verify no OOM

### Day 4: Phase 4 (Testing)
1. ✅ Create validation test
2. ✅ Create production test
3. ✅ Run validation (1K × 1 step)
4. ✅ Run production (105K × 2.5K steps)
5. ✅ Compare vs hierarchical 5-hop
6. ✅ Document results

---

## Success Criteria

### Must-Have (Blocking)
- [ ] No OOM errors during preprocessing
- [ ] No OOM errors during GPU execution
- [ ] Initial assignment >95% success rate
- [ ] Single timestep completes without errors
- [ ] No nested jit/vmap warnings

### Should-Have (Target)
- [ ] Final retention >95% at 2,500 steps
- [ ] Throughput 40-50k p/s sustained
- [ ] Memory overhead <200 MB
- [ ] All JAX arrays have static shapes

### Nice-to-Have (Stretch)
- [ ] Final retention >98% at 2,500 steps
- [ ] Throughput >50k p/s
- [ ] Memory overhead <100 MB
- [ ] L2 hit rate >99.99%

---

## Next Action

**START**: Implement Phase 1 - CPU Preprocessing

**First Step**: Create `jaxtrace/gpu/search/morton_global_builder.py` with:
1. `interleave_bits_3d()` function
2. `compute_morton_codes_for_elements()` function
3. Test on small mesh (1K elements)

**User Confirmation**: Review [HOT_MORTON_REVISED_PLAN.md](HOT_MORTON_REVISED_PLAN.md) for complete details.

---

**Status**: ✅ READY TO IMPLEMENT

User should confirm plan correctness before proceeding with Phase 1 implementation.
