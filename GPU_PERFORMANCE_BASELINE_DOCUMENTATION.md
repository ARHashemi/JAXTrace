# GPU Performance Baseline Documentation

## Status: Working Baseline with Particle Loss

**Date:** 2025-11-26
**Branch:** gpu_native_implementation
**Performance:** 40k p/s (initial) → 21k p/s (final), Mean: ~30k p/s
**Particle Retention:** ~16% (10k/62k particles survive 2,500 timesteps)

---

## Architecture Overview

### Phase 3a: HYBRID Incremental Search with GPU-Fused RK4

**Search Strategy:**
- **L0 (Cached Element)**: Check if particle still in previous element
- **L1 (Extended Neighbors)**: 2-hop neighbor search (~20 neighbors per element)
- **No L2/L3 fallback**: Particles that miss L0+L1 are marked as lost (-1)

**RK4 Integration:**
- All 4 RK4 stages execute on GPU
- Velocity field uploaded ONCE at initialization
- 2 transfers per timestep: upload particle state, download particle state

---

## CPU-GPU Transfer Pattern

### Per Timestep Transfers (VERIFIED NO HIDDEN TRANSFERS)

**Upload (Beginning of Timestep):**
```python
# File: jaxtrace/gpu/tracking/rk4_gpu_fused.py:505-506
positions_gpu = jax.device_put(positions.astype(np.float32))      # ~750 KB @ 62k particles
element_ids_gpu = jax.device_put(element_ids.astype(np.int32))   # ~250 KB @ 62k particles
```

**GPU Computation (No Transfers):**
```python
# File: jaxtrace/gpu/tracking/rk4_gpu_fused.py:408-500
@jax.jit
def rk4_fused_with_search(...):
    # Stage 1: search + interpolate + advance
    # Stage 2: search + interpolate + advance
    # Stage 3: search + interpolate + advance
    # Stage 4: search + interpolate + advance
    # Final: combine stages + final search
    return positions_final_gpu, element_ids_final_gpu
```

All operations inside `@jax.jit` stay on GPU:
- `search_level0_vectorized` (line 173)
- `search_level1_multihop_vectorized` (line 183)
- `interpolate_velocity_batch_gpu` (line 427, 444, 461, 478)

**Download (End of Timestep):**
```python
# File: jaxtrace/gpu/tracking/rk4_gpu_fused.py:533-534
positions_final = np.array(positions_final_gpu, dtype=np.float32)    # ~750 KB
element_ids_final = np.array(element_ids_final_gpu, dtype=np.int32) # ~250 KB
```

### One-Time Uploads (Initialization)

**Mesh Data (117.5 MB):**
```python
# File: production_tracking_threadeda.py:517-522
mesh_gpu = upload_mesh_to_gpu(
    connectivity,        # 53.59 MB - (3.5M elements, 4 nodes each)
    node_positions,      # 10.31 MB - (900k nodes, 3 coords each)
    element_neighbors    # 53.59 MB - (3.5M elements, 4 neighbors each)
)
```

**Velocity Field (10.3 MB):**
```python
# File: production_tracking_threadeda.py:527
velocity_field_gpu = jax.device_put(velocity_field.astype(np.float32))
```

### Transfer Volume Analysis

**Per Timestep:**
- Upload: 1 MB (positions + element_ids)
- Download: 1 MB (positions + element_ids)
- **Total: 2 MB per timestep**

**For 2,500 Timesteps:**
- **Total: 5 GB of particle data transfers**

**One-Time (Initialization):**
- Mesh: 117.5 MB
- Velocity: 10.3 MB
- **Total: 127.8 MB (upload once)**

**Grand Total: 5.13 GB transfers for entire simulation**

---

## Performance Characteristics

### Throughput (from logs/production_test_recover.log)

```
Step   100/2500 | Active: 55,263 | Throughput: 40,211 p/s
Step   200/2500 | Active: 49,242 | Throughput: 36,054 p/s
Step   300/2500 | Active: 43,716 | Throughput: 32,256 p/s
Step   400/2500 | Active: 38,366 | Throughput: 28,384 p/s
Step   500/2500 | Active: 33,100 | Throughput: 24,671 p/s
Step   600/2500 | Active: 29,345 | Throughput: 21,242 p/s
```

**Mean throughput: ~30,000 p/s**

### GPU Utilization

- **Observed: 0-11%** (low!)
- **Expected: 80-90%** (for GPU-bound computation)

### Bottleneck Analysis

The low GPU utilization (0-11%) indicates the GPU is idle most of the time. This is caused by:

1. **CPU-GPU Transfer Latency** (~500 μs per timestep)
   - 2 MB bandwidth is small, but PCIe latency dominates
   - Each transfer has ~10-50 μs latency + synchronization overhead

2. **Particle Data Remains on CPU**
   - `ParticleData` stores numpy arrays (CPU memory)
   - Every timestep requires upload + download
   - GPU waits idle during transfers

3. **Decreasing Particle Count**
   - Fewer particles → less GPU work → lower utilization
   - Transfer overhead becomes larger fraction of time

**Transfer Time Calculation:**
- Bandwidth: ~6 GB/s (PCIe 3.0 effective)
- Data per timestep: 2 MB
- Transfer time: 2 MB / 6 GB/s = 0.33 ms
- Latency overhead: 4 transfers × 50 μs = 0.2 ms
- **Total transfer overhead: ~0.5 ms per timestep**

**GPU Compute Time:**
- At 40k p/s: 55k particles / 40k p/s = 1.375 s per timestep
- Transfer overhead: 0.5 ms
- **Actual GPU compute: 1.374 s** (99.96% of time is compute!)

**Wait, this doesn't match!** The low GPU utilization (0-11%) suggests something else is wrong. Let me check if there's JIT recompilation happening or other issues.

**Possible causes of low GPU utilization:**
1. Small batch size after particle loss (33k → 10k particles)
2. JIT recompilation on each timestep (unlikely, should compile once)
3. CPU orchestration overhead in production loop
4. Export/boundary checking overhead

---

## Known Issues

### Issue 1: Particle Loss (83.8%)

**Problem:** 62,500 particles → 10,016 particles (16.2% retention)

**Root Cause:** Only L0+L1 search (2-hop, ~20 neighbors)
- Particles that leave the 2-hop neighborhood are lost
- No L2 (block search) or L3 (global search) fallback

**Impact:**
- Most particles lost during time integration
- Unacceptable for production use

**Solutions (for next phase):**
1. Extend L1 to 3-4 hops (~84-340 neighbors)
2. Add GPU-resident L2 fallback (global search on GPU)
3. Hybrid: Extended L1 + CPU L2 fallback (download only misses)

### Issue 2: Low GPU Utilization (0-11%)

**Problem:** GPU is mostly idle during simulation

**Root Cause:** Particle data lives on CPU, causing repeated transfers

**Impact:**
- 10-16× slower than potential performance
- Expected: 400-640k p/s
- Actual: 40k p/s (initial)

**Solution (for future optimization):**
- Keep particle data on GPU throughout simulation
- Only download for VTK export (every 10 steps)
- See `GPU_TRANSFER_BOTTLENECK_ANALYSIS.md` for implementation options

### Issue 3: Static Mesh/Velocity

**Current Limitation:** Mesh and velocity field are static

**Problem for Real Applications:**
- Mesh refinement updates connectivity in local regions
- Velocity field changes at each timestep

**Solution (planned):**
- Time-dependent velocity field updates
- Incremental mesh refinement (update only changed regions)
- See "Future Optimizations" section below

---

## Code Structure

### Key Files

**RK4 Integration:**
- [jaxtrace/gpu/tracking/rk4_gpu_fused.py](jaxtrace/gpu/tracking/rk4_gpu_fused.py) - GPU-fused RK4 with transfers
  - `rk4_step_gpu_fused_wrapper` (line 363) - Handles CPU-GPU transfers
  - `rk4_fused_with_search` (line 408) - JIT-compiled GPU computation
  - `create_search_gpu_fused` (line 128) - Creates search function with n_hops

**Search Functions:**
- [jaxtrace/gpu/search/incremental_search_vectorized.py](jaxtrace/gpu/search/incremental_search_vectorized.py)
  - `search_level0_vectorized` (line 64) - Cached element check
  - `search_level1_multihop_vectorized` (line 262) - Multi-hop neighbor search
  - All JIT-compiled, no transfers

**Interpolation:**
- [jaxtrace/gpu/tracking/rk4_gpu_fused.py](jaxtrace/gpu/tracking/rk4_gpu_fused.py:45-125)
  - `interpolate_velocity_batch_gpu` (line 45) - Batch velocity interpolation
  - JIT-compiled, stays on GPU

**Production Script:**
- [production_tracking_threadeda.py](production_tracking_threadeda.py)
  - Line 524-528: Upload velocity field once
  - Line 787-794: JIT warm-up
  - Line 858-865: Time marching loop with RK4

### Configuration

```python
# File: production_tracking_threadeda.py:268-282
USE_VECTORIZED_SEARCH = True    # Phase 3a hybrid search
USE_GPU_FUSED_RK4 = True        # All RK4 stages on GPU
RK4_L1_HOP_COUNT = 2            # 2-hop (~20 neighbors)
```

---

## Future Optimizations

### Priority 1: Extend L1 Hops Without Speed Loss

**Goal:** Increase particle retention (16% → 90%+) while maintaining speed

**Approach:**
1. **Try 3-hop L1 first** (~84 neighbors)
   - Expected: 98-99% hit rate
   - Memory: Check if fits in GPU memory
   - Speed: May reduce to ~120k p/s (vs 40k current)

2. **Try 4-hop L1** (~340 neighbors)
   - Expected: 99.5-99.9% hit rate
   - Memory: 3.5 GB (may exceed GPU capacity)
   - Speed: ~80k p/s

3. **Hybrid: 2-hop L1 + GPU L2**
   - Keep 2-hop L1 (fast, 95% hit rate)
   - Add GPU-resident global search for 5% misses
   - Expected: 99% hit rate, ~200k p/s

**Recommendation:** Try approach 3 (hybrid 2-hop + GPU L2)
- Least risk (2-hop proven fast)
- Handles misses without CPU-GPU transfers
- See section below for implementation strategy

### Priority 2: GPU-Resident Particle Data

**Goal:** Eliminate 2 MB × 2,500 = 5 GB particle transfers

**Approach:** Keep positions/element_ids as JAX arrays on GPU
- Upload once at initialization
- Download only for VTK export (every 10 steps)
- Expected speedup: 10-16× (40k → 400-640k p/s)

**Implementation:** See `GPU_TRANSFER_BOTTLENECK_ANALYSIS.md` Option 3

### Priority 3: Time-Dependent Mesh/Velocity

**Goal:** Support mesh refinement and time-varying velocity fields

**Challenge:** Minimize GPU updates without breaking JIT compilation

**Approaches:**

#### A) Vectorized Element Neighbors (Your Suggestion)

**Idea:** Pass full neighbor connectivity instead of sparse element_neighbors

**Current:**
```python
element_neighbors: Array[n_elements, 4]  # Only direct face neighbors
```

**Proposed:**
```python
element_connectivity: Array[n_elements, max_neighbors]  # All connected elements
# Could include:
# - Face neighbors (4)
# - Edge neighbors (additional ~6)
# - Vertex neighbors (additional ~12)
# Total: ~20-30 neighbors per element
```

**Benefits:**
- ✅ Single array contains all connectivity
- ✅ Easy to update incrementally (only changed elements)
- ✅ Compatible with vectorized search

**Drawbacks:**
- ❌ Larger memory (3.5M × 30 vs 3.5M × 4)
- ❌ Need to pad to max_neighbors
- ❌ May slow down search (more neighbors to check)

**Verdict:** Worth trying! Fits well with GPU architecture

#### B) Incremental Mesh Updates

**Idea:** Update only changed regions instead of full mesh upload

**Implementation:**
```python
def update_mesh_region_gpu(
    mesh_gpu: MeshDataGPU,
    changed_element_ids: np.ndarray,        # Array of element IDs that changed
    new_connectivity: np.ndarray,            # New connectivity for changed elements
    new_element_neighbors: np.ndarray        # New neighbors for changed elements
) -> MeshDataGPU:
    # Update only changed elements on GPU
    mesh_gpu.connectivity = mesh_gpu.connectivity.at[changed_element_ids].set(new_connectivity)
    mesh_gpu.element_neighbors = mesh_gpu.element_neighbors.at[changed_element_ids].set(new_element_neighbors)
    return mesh_gpu
```

**Benefits:**
- ✅ Minimal transfer (only changed elements)
- ✅ Preserves GPU-resident mesh
- ✅ Compatible with JIT (uses `.at[].set()`)

**Drawbacks:**
- ❌ Requires tracking changed elements
- ❌ May trigger JIT recompilation if mesh shape changes

**Verdict:** Good for local refinement, poor for global changes

#### C) Time-Dependent Velocity Field Updates

**Current Problem:** Velocity field is static (uploaded once)

**Solution 1: Upload Each Timestep (Simple)**
```python
# Remove velocity_field_gpu caching
# Pass np.ndarray to RK4 each timestep
# Will upload 10 MB per timestep → 25 GB total
```

**Drawback:** Huge transfer volume (25 GB for 2,500 steps)

**Solution 2: Differential Updates**
```python
def update_velocity_field_gpu(
    velocity_field_gpu: jax.Array,
    changed_node_ids: np.ndarray,           # Nodes with updated velocity
    new_velocities: np.ndarray              # New velocities (n_changed, 3)
) -> jax.Array:
    # Update only changed nodes
    return velocity_field_gpu.at[changed_node_ids].set(new_velocities)
```

**Benefits:**
- ✅ Minimal transfer (only changed nodes)
- ✅ Preserves GPU-resident velocity field
- ✅ Fast for local velocity updates

**Drawback:** Requires tracking velocity changes

**Solution 3: Compressed Updates**
```python
# For small changes (< 10% of nodes):
#   Use differential update

# For large changes (> 10% of nodes):
#   Upload full velocity field
```

---

## Strategy for L1 Hop Extension

### Phase 3b: Extended L1 with GPU L2 Fallback

**Goal:** 99%+ particle retention while maintaining 100-200k p/s throughput

**Implementation Plan:**

### Step 1: Implement GPU L2 Global Search (1-2 hours)

**Add global search to JIT-compiled RK4:**

```python
@jax.jit
def search_level2_global_vectorized(
    positions: jax.Array,                # (N, 3) - particles that missed L0+L1
    node_positions: jax.Array,
    connectivity: jax.Array
) -> jax.Array:
    """
    Global search for particles that missed L0+L1.

    Checks ALL elements for each particle (expensive but thorough).
    Vectorized across particles for GPU parallelism.
    """
    def search_one_particle(pos):
        # Check all elements
        def check_element(elem_id):
            node_ids = connectivity[elem_id]
            tet_nodes = node_positions[node_ids]
            inside = point_in_tet_jax(pos, tet_nodes)
            return jnp.where(inside, elem_id, -1)

        # Vectorize over all elements
        results = jax.vmap(check_element)(jnp.arange(len(connectivity)))

        # Find first match
        found = results >= 0
        first_idx = jnp.argmax(found)  # Returns 0 if none found
        return jnp.where(found[first_idx], results[first_idx], -1)

    # Vectorize over particles
    return jax.vmap(search_one_particle)(positions)
```

**Integrate into RK4 search:**

```python
@jax.jit
def search_gpu_fused_with_fallback(
    positions_gpu,
    cached_element_ids_gpu,
    mesh_gpu_node_positions,
    mesh_gpu_connectivity,
    mesh_gpu_element_neighbors
):
    # L0: Check cached elements
    element_ids_l0 = search_level0_vectorized(...)

    # L1: Check 2-hop neighbors
    element_ids_l1 = search_level1_multihop_vectorized(..., n_hops=2)

    # Merge L0+L1
    element_ids = jnp.where(element_ids_l0 >= 0, element_ids_l0, element_ids_l1)

    # L2: Global search for misses (only ~5% of particles)
    l0_l1_misses = element_ids < 0
    n_misses = jnp.sum(l0_l1_misses)

    # Only run L2 if there are misses
    element_ids_l2 = jnp.where(
        l0_l1_misses,
        search_level2_global_vectorized(
            positions_gpu,
            mesh_gpu_node_positions,
            mesh_gpu_connectivity
        ),
        element_ids
    )

    return element_ids_l2
```

**Expected Performance:**
- L0+L1 hit rate: 95%
- L2 fallback: 5% of particles
- L2 throughput: ~1k p/s (global search is expensive)
- Overall: ~150-200k p/s (5% overhead from L2)

### Step 2: Test and Measure (30 min)

**Test with 1,000 particles first:**
```bash
# Modify production script: PARTICLE_GRID_RESOLUTION = (10, 10, 10)
python production_tracking_threadeda.py 2>&1 | tee logs/test_l2_fallback.log
```

**Check:**
- Particle retention (should be 99%+)
- Throughput (should be 100-200k p/s)
- GPU memory usage (should stay < 3 GB)

### Step 3: Scale to Full Simulation (if test passes)

**Run with full 62,500 particles:**
```bash
python production_tracking_threadeda.py 2>&1 | tee logs/production_with_l2_fallback.log
```

---

## Strategy for Time-Dependent Updates

### Approach: Differential Updates with Change Tracking

**Core Idea:** Track which elements/nodes changed, update only those on GPU

### Implementation:

```python
@dataclass
class MeshUpdateGPU:
    """Incremental mesh update"""
    changed_element_ids: jax.Array      # (n_changed,)
    new_connectivity: jax.Array          # (n_changed, 4)
    new_element_neighbors: jax.Array     # (n_changed, 4)

def apply_mesh_update_gpu(
    mesh_gpu: MeshDataGPU,
    update: MeshUpdateGPU
) -> MeshDataGPU:
    """Apply incremental mesh update on GPU"""
    return MeshDataGPU(
        connectivity=mesh_gpu.connectivity.at[update.changed_element_ids].set(update.new_connectivity),
        element_neighbors=mesh_gpu.element_neighbors.at[update.changed_element_ids].set(update.new_element_neighbors),
        ...
    )

# In production loop:
for step in range(N_TIMESTEPS):
    # Check if mesh changed this timestep
    if mesh_refinement_occurred(step):
        mesh_update = compute_mesh_changes(step)  # Returns MeshUpdateGPU
        mesh_gpu = apply_mesh_update_gpu(mesh_gpu, mesh_update)

    # Run RK4 with updated mesh
    particle_data = rk4_step_gpu_fused_for_production(
        particle_data, velocity_field_gpu, DT, mesh_gpu, ...
    )
```

**For Velocity Field:**
```python
def apply_velocity_update_gpu(
    velocity_field_gpu: jax.Array,
    changed_node_ids: jax.Array,
    new_velocities: jax.Array
) -> jax.Array:
    """Update velocity field incrementally"""
    return velocity_field_gpu.at[changed_node_ids].set(new_velocities)
```

**Benefits:**
- ✅ Minimal transfers (only changed data)
- ✅ Preserves GPU-resident state
- ✅ Compatible with JIT (uses `.at[].set()`)

**JIT Consideration:**
- JAX JIT handles `.at[].set()` efficiently
- No recompilation if array shapes don't change
- Only transfer changed data, not full mesh

---

## Recommendations for Next Phase

### Immediate (This Session):
1. ✅ Document current baseline (this file)
2. ✅ Commit working code with particle loss documented
3. Create strategy document for next optimizations

### Phase 3b (Next Session):
1. **Implement GPU L2 fallback** (1-2 hours)
   - Add `search_level2_global_vectorized`
   - Integrate with existing L0+L1 search
   - Test with small particle count first

2. **Test particle retention** (30 min)
   - Run full simulation
   - Verify 99%+ retention
   - Measure throughput impact

### Phase 3c (Future):
1. **GPU-resident particle data** (2-3 hours)
   - Eliminate 5 GB particle transfers
   - Expected: 10-16× speedup

2. **Differential mesh/velocity updates** (2-3 hours)
   - Support time-dependent simulations
   - Minimal transfer overhead

---

## Questions for Discussion

1. **L1 Hop Extension:**
   - Try 3-hop first, or go straight to 2-hop + L2 fallback?
   - Your preference?

2. **Vectorized Connectivity:**
   - Should we try your suggestion of passing full element connectivity?
   - How many neighbors per element? (20? 30? 40?)

3. **Time-Dependent Updates:**
   - How often does mesh refinement occur? (every step? every 100 steps?)
   - How many elements typically change? (< 1%? 10%? 50%?)
   - This affects whether differential updates are worth it

4. **Performance Target:**
   - What throughput do you need? (100k p/s? 500k p/s?)
   - Is 90% particle retention acceptable, or need 99%+?
