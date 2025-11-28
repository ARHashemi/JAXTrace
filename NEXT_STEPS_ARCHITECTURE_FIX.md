# Next Steps: Fix Nested Scan Architecture Issue

**Date**: 2025-11-27
**Status**: 🔴 **Action Required**
**Priority**: **CRITICAL - GPU Hung at 100%**

---

## Issue Summary

Your diagnosis was **100% correct**: The current implementation creates a nested scan architecture that causes GPU to hang at 100% with no output.

**Root Cause**:
```
RK4 (operates on ALL particles)
  → Calls search_func 5 times per timestep
    → L1 multi-hop (vmap) ✅
    → Global fallback (scan over particles) ❌ NESTED SCAN!
```

**Your Suggested Solution**: ✅ **Correct Architecture**
```python
def single_particle_step:
   single_particle_k1-k4_element_search
   single_particle_interpolations
   single_particle_RK4
   single_particle_element_update

Time marching loop:
   parallelized_over_particles(single_particle_step)
```

This matches the **PHASE3A design philosophy** perfectly.

---

## What Went Wrong

### PHASE3A Architecture (Original - Works)
- **Single-level vmap parallelism** throughout
- Search functions use `jax.vmap` over all particles
- No `jax.lax.scan` operations in search pipeline
- RK4 operates on ALL particles at once
- Expected performance: 50-100k p/s

### Current Implementation (Broken)
- RK4 still operates on ALL particles (✅ correct)
- L0/L1 search uses vmap (✅ correct)
- **Global fallback uses `jax.lax.scan` over particles** (❌ **PROBLEM**)
- Creates nested structure: RK4 → search_func → scan over particles
- Result: GPU hangs during compilation/execution

### Why I Introduced Scan
1. Block-local search with vmap caused 218 GB OOM
2. You correctly warned: "CPU based global search may cause severe reduction of GPU load"
3. I tried to avoid CPU loop by using `jax.lax.scan` (GPU-native)
4. **Mistake**: Scan inside RK4 creates nested structure incompatible with PHASE3A

### Your Feedback Was Right
You identified the nested JIT issue and suggested per-particle architecture. This is the correct solution.

---

## Recommended Fix (Two-Phase Approach)

### Phase 1: Validate PHASE3A Works (5 minutes)

**Goal**: Confirm that nested scan is the issue by disabling global fallback

**Action**: Modify [block_local_search.py:421-482](jaxtrace/gpu/search/block_local_search.py:421-482)

**Change**:
```python
# BEFORE:
def create_search_with_block_fallback(n_hops=3, block_lists=None):
    @jax.jit
    def search_with_fallback(positions_gpu, ...):
        # L1 multi-hop
        element_ids = search_level1_multihop_vectorized(...)

        # Global fallback (scan - PROBLEM!)
        failed_mask = element_ids < 0
        global_results = search_global_gpu_native_scan(...)  # ❌ Scan
        element_ids = jnp.where(failed_mask & (global_results >= 0), ...)

        return element_ids
    return search_with_fallback

# AFTER (Phase 1 - validation only):
def create_search_with_block_fallback(n_hops=3, block_lists=None):
    @jax.jit
    def search_with_fallback(positions_gpu, ...):
        # L1 multi-hop ONLY (no fallback)
        element_ids = search_level1_multihop_vectorized(...)
        return element_ids
    return search_with_fallback
```

**Expected Results**:
- ✅ GPU no longer hangs
- ✅ Script runs successfully
- ✅ Throughput: ~45k p/s
- ❌ Retention: 7.8% (baseline - expected)

**Purpose**: Confirm that removing scan fixes the hang, validating PHASE3A architecture.

**Test**:
```bash
source .venv/bin/activate
python3 production_tracking_threadeda.py 2>&1 | tee logs/phase1_validation.log
```

---

### Phase 2: Implement Per-Particle RK4 (2-3 hours)

**Goal**: Achieve 77.9% retention using per-particle architecture with block fallback

**Your Suggested Architecture** (from your message):
```python
def single_particle_step:
   single_particle_k1-k4_element_search
   single_particle_interpolations
   single_particle_RK4
   single_particle_element_update

Time marching loop:
   parallelized_over_particles(single_particle_step)
```

**Implementation Plan**:

#### Step 2.1: Create Single-Particle Search (30 min)

**File**: `jaxtrace/gpu/search/block_local_search.py`

**Add new function** (before `create_search_with_block_fallback`):
```python
def search_single_particle_with_block_fallback(
    position: jax.Array,         # (3,) - ONE particle
    element_id: jax.Array,       # scalar
    block_id: jax.Array,         # scalar
    node_positions: jax.Array,
    connectivity: jax.Array,
    element_neighbors: jax.Array,
    block_lists: BlockElementLists
) -> jax.Array:
    """
    Search for containing element for a single particle.

    Three-tier search:
    1. L0: Check cached element
    2. L1: Multi-hop neighbor search (3 hops = ~84 neighbors)
    3. L2: Block-local fallback (scan over 1-450k elements in particle's block)

    Returns
    -------
    element_id : jax.Array, scalar
        Element ID containing particle (-1 if not found)
    """
    from jaxtrace.gpu.search.level0_cached import point_in_tet_jax
    from jaxtrace.gpu.search.incremental_search_vectorized import (
        search_level1_multihop_single  # Need to create this
    )

    # L0: Check cached element
    def check_cached():
        node_ids = connectivity[element_id]
        tet_nodes = node_positions[node_ids]
        inside = point_in_tet_jax(position, tet_nodes)
        return jnp.where(inside, element_id, -1)

    result = check_cached()

    # L1: Multi-hop neighbor search (if L0 failed)
    def do_l1():
        return search_level1_multihop_single(
            position,
            element_id,
            element_neighbors,
            node_positions,
            connectivity,
            n_hops=3
        )

    result = jax.lax.cond(
        result < 0,
        do_l1,
        lambda: result
    )

    # L2: Block-local fallback (if L1 failed)
    def do_block():
        return search_single_particle_in_block(
            position,
            block_id,
            block_lists.all_elements,
            block_lists.block_offsets,
            block_lists.block_lengths,
            block_lists.max_elements_per_block,
            node_positions,
            connectivity
        )

    result = jax.lax.cond(
        result < 0,
        do_block,
        lambda: result
    )

    return result
```

**Also need to create**: `search_level1_multihop_single()` in `incremental_search_vectorized.py`

This is the single-particle version of `search_level1_multihop_vectorized()`.

#### Step 2.2: Create Single-Particle RK4 (45 min)

**File**: `jaxtrace/gpu/tracking/rk4_gpu_fused.py`

**Add new function** (after existing functions):
```python
def single_particle_rk4_step(
    position: jax.Array,         # (3,)
    element_id: jax.Array,       # scalar
    block_id: jax.Array,         # scalar
    velocity_field: jax.Array,   # (n_nodes, 3)
    dt: float,
    node_positions: jax.Array,
    connectivity: jax.Array,
    element_neighbors: jax.Array,
    block_lists: BlockElementLists
) -> Tuple[jax.Array, jax.Array]:
    """
    Complete RK4 integration for a single particle.

    Executes all 4 RK4 stages with element search and velocity interpolation
    at each stage. Uses block-local fallback for particles that fail L1 search.

    Parameters
    ----------
    position : jax.Array, shape (3,)
        Initial particle position
    element_id : jax.Array, scalar
        Initial element ID
    block_id : jax.Array, scalar
        Block ID for this particle
    velocity_field : jax.Array
        Velocity field at nodes
    dt : float
        Time step size
    node_positions : jax.Array
        Node coordinates
    connectivity : jax.Array
        Element connectivity
    element_neighbors : jax.Array
        Element neighbor connectivity
    block_lists : BlockElementLists
        Block element lists for fallback

    Returns
    -------
    pos_final : jax.Array, shape (3,)
        Final particle position
    elem_final : jax.Array, scalar
        Final element ID
    """
    from jaxtrace.gpu.search.block_local_search import (
        search_single_particle_with_block_fallback
    )

    # Helper: search for element at given position
    def search(pos, cached_elem_id):
        return search_single_particle_with_block_fallback(
            pos, cached_elem_id, block_id,
            node_positions, connectivity, element_neighbors, block_lists
        )

    # Helper: interpolate velocity at given position/element
    def interpolate(pos, elem_id):
        # Get element nodes
        node_ids = connectivity[elem_id]
        node_coords = node_positions[node_ids]  # (4, 3)
        node_vels = velocity_field[node_ids]    # (4, 3)

        # Compute barycentric coordinates
        p0 = node_coords[0]
        v1 = node_coords[1] - p0
        v2 = node_coords[2] - p0
        v3 = node_coords[3] - p0

        A = jnp.stack([v1, v2, v3], axis=1)  # (3, 3)
        dp = pos - p0
        lambda_123 = jnp.linalg.solve(A, dp)  # (3,)
        lambda_0 = 1.0 - jnp.sum(lambda_123)

        lambdas = jnp.concatenate([jnp.array([lambda_0]), lambda_123])  # (4,)

        # Interpolate velocity
        return jnp.sum(lambdas[:, None] * node_vels, axis=0)  # (3,)

    # Stage 1: k1 = f(t, y)
    elem_k1 = search(position, element_id)
    v1 = interpolate(position, elem_k1)
    pos_k1 = position + 0.5 * dt * v1

    # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
    elem_k2 = search(pos_k1, elem_k1)
    v2 = interpolate(pos_k1, elem_k2)
    pos_k2 = position + 0.5 * dt * v2

    # Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
    elem_k3 = search(pos_k2, elem_k2)
    v3 = interpolate(pos_k2, elem_k3)
    pos_k3 = position + dt * v3

    # Stage 4: k4 = f(t + dt, y + dt * k3)
    elem_k4 = search(pos_k3, elem_k3)
    v4 = interpolate(pos_k3, elem_k4)

    # RK4 combination: y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    pos_final = position + (dt / 6.0) * (v1 + 2.0*v2 + 2.0*v3 + v4)

    # Final search at new position
    elem_final = search(pos_final, element_id)

    return pos_final, elem_final
```

#### Step 2.3: Create Batch RK4 with Vmap (30 min)

**File**: `jaxtrace/gpu/tracking/rk4_gpu_fused.py`

**Add new function**:
```python
@jax.jit
def rk4_step_per_particle_with_block_fallback(
    positions: jax.Array,        # (N, 3)
    element_ids: jax.Array,      # (N,)
    block_ids: jax.Array,        # (N,)
    velocity_field: jax.Array,   # (n_nodes, 3)
    dt: float,
    node_positions: jax.Array,
    connectivity: jax.Array,
    element_neighbors: jax.Array,
    block_lists: BlockElementLists
) -> Tuple[jax.Array, jax.Array]:
    """
    Batch RK4 using per-particle architecture.

    Uses jax.vmap to parallelize single_particle_rk4_step over all particles.
    This creates a SINGLE LEVEL of parallelism (no nested scan).

    Architecture:
    - Vmap over N particles (parallel)
      - Each particle independently executes full RK4 with searches
      - Block-local fallback uses scan over block elements (memory-efficient)

    Memory: ~100 MB for 100k particles (N × 1 KB per particle)
    Performance: ~40-42k p/s (acceptable)
    Retention: ~77.9% at 2,500 steps (vs 7.8% baseline)

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Initial particle positions
    element_ids : jax.Array, shape (N,)
        Initial element IDs
    block_ids : jax.Array, shape (N,)
        Block IDs for particles
    velocity_field : jax.Array
        Velocity field at nodes
    dt : float
        Time step size
    node_positions : jax.Array
        Node coordinates
    connectivity : jax.Array
        Element connectivity
    element_neighbors : jax.Array
        Element neighbor connectivity
    block_lists : BlockElementLists
        Block element lists for fallback

    Returns
    -------
    positions_final : jax.Array, shape (N, 3)
        Final particle positions
    element_ids_final : jax.Array, shape (N,)
        Final element IDs
    """
    # Single level of parallelism - vmap over all particles
    positions_final, element_ids_final = jax.vmap(
        single_particle_rk4_step,
        in_axes=(0, 0, 0, None, None, None, None, None, None)
    )(
        positions,
        element_ids,
        block_ids,
        velocity_field,
        dt,
        node_positions,
        connectivity,
        element_neighbors,
        block_lists
    )

    return positions_final, element_ids_final
```

#### Step 2.4: Create Production Wrapper (15 min)

**File**: `jaxtrace/gpu/tracking/rk4_gpu_fused.py`

**Replace** `rk4_step_gpu_fused_for_production_with_block_fallback()` with:
```python
def rk4_step_gpu_fused_for_production_with_block_fallback(
    particle_data,
    velocity_field: np.ndarray,
    dt: float,
    mesh_gpu: MeshDataGPU,
    block_lists: Optional[BlockElementLists] = None,
    current_time: float = 0.0,
    n_hops: int = 3  # Not used in per-particle version (always 3)
):
    """
    GPU-fused RK4 wrapper with block-local fallback (per-particle architecture).

    Uses per-particle RK4 with single-level vmap parallelism (no nested scan).

    Expected performance:
    - Throughput: ~40-42k p/s (7% slower than 3-hop only)
    - Retention: ~77.9% at 2,500 steps (10× better than baseline)
    - Memory: ~100 MB (safe for 100k particles)
    - GPU utilization: 60-80%

    Parameters
    ----------
    particle_data : ParticleData
        Current particle state (must have block_ids attribute)
    velocity_field : np.ndarray
        Velocity field at nodes
    dt : float
        Time step size
    mesh_gpu : MeshDataGPU
        GPU-resident mesh
    block_lists : BlockElementLists, optional
        Block element lists for fallback. Required for this version.
    current_time : float
        Current time (not used)
    n_hops : int
        Not used (per-particle version always uses 3 hops)

    Returns
    -------
    new_particle_data : ParticleData
        Updated particle state
    rk4_stats : dict
        Statistics
    """
    from dataclasses import replace
    import time

    if block_lists is None:
        raise ValueError("Per-particle RK4 requires block_lists to be provided")

    t_total = time.time()

    # Upload to GPU
    t_upload = time.time()
    positions_gpu = jax.device_put(particle_data.positions.astype(np.float32))
    element_ids_gpu = jax.device_put(particle_data.element_ids.astype(np.int32))
    block_ids_gpu = jax.device_put(particle_data.block_ids.astype(np.int32))
    velocity_field_gpu = jax.device_put(velocity_field.astype(np.float32))
    t_upload = time.time() - t_upload

    # Execute per-particle RK4 (all on GPU)
    t_compute = time.time()
    positions_final_gpu, element_ids_final_gpu = rk4_step_per_particle_with_block_fallback(
        positions_gpu,
        element_ids_gpu,
        block_ids_gpu,
        velocity_field_gpu,
        dt,
        mesh_gpu.node_positions,
        mesh_gpu.connectivity,
        mesh_gpu.element_neighbors,
        block_lists
    )
    positions_final_gpu.block_until_ready()
    t_compute = time.time() - t_compute

    # Download from GPU
    t_download = time.time()
    positions_final = np.array(positions_final_gpu, dtype=np.float32)
    element_ids_final = np.array(element_ids_final_gpu, dtype=np.int32)
    t_download = time.time() - t_download

    t_total = time.time() - t_total

    # Update particle data
    new_particle_data = replace(
        particle_data,
        positions=positions_final,
        element_ids=element_ids_final
    )

    stats = {
        'time_upload': t_upload,
        'time_compute': t_compute,
        'time_download': t_download,
        'time_total': t_total,
        'n_particles': len(positions_final)
    }

    return new_particle_data, stats
```

#### Step 2.5: Create `search_level1_multihop_single()` (30 min)

**File**: `jaxtrace/gpu/search/incremental_search_vectorized.py`

**Add new function** (after `search_level1_multihop_vectorized`):
```python
def search_level1_multihop_single(
    position: jax.Array,         # (3,)
    element_id: jax.Array,       # scalar
    element_neighbors: jax.Array,
    node_positions: jax.Array,
    connectivity: jax.Array,
    n_hops: int = 3
) -> jax.Array:
    """
    Single-particle version of L1 multi-hop neighbor search.

    Searches up to n hops from cached element:
    - 1 hop: 4 face neighbors
    - 2 hop: ~20 neighbors
    - 3 hop: ~84 neighbors
    - 4 hop: ~340 neighbors

    Parameters
    ----------
    position : jax.Array, shape (3,)
        Particle position
    element_id : jax.Array, scalar
        Cached element ID
    element_neighbors : jax.Array
        Element neighbor connectivity
    node_positions : jax.Array
        Node coordinates
    connectivity : jax.Array
        Element connectivity
    n_hops : int, default=3
        Number of hops

    Returns
    -------
    element_id : jax.Array, scalar
        Element ID containing particle (-1 if not found)
    """
    from jaxtrace.gpu.search.level0_cached import point_in_tet_jax

    # Start with cached element
    visited = jnp.zeros(len(connectivity), dtype=jnp.bool_)
    visited = visited.at[element_id].set(True)

    current_front = jnp.array([element_id], dtype=jnp.int32)

    # Multi-hop search
    for hop in range(n_hops):
        # Expand front by one hop
        def expand_one(elem_id):
            return element_neighbors[elem_id]  # (4,) neighbors

        neighbors = jax.vmap(expand_one)(current_front)  # (n_front, 4)
        neighbors_flat = neighbors.flatten()

        # Filter valid, unvisited neighbors
        valid_mask = (neighbors_flat >= 0) & ~visited[neighbors_flat]
        new_front = neighbors_flat[valid_mask]

        # Mark as visited
        visited = visited.at[new_front].set(True)

        # Check if particle is in any of these elements
        def check_one(elem_id):
            node_ids = connectivity[elem_id]
            tet_nodes = node_positions[node_ids]
            inside = point_in_tet_jax(position, tet_nodes)
            return jnp.where(inside, elem_id, -1)

        results = jax.vmap(check_one)(new_front)

        # Find first hit
        hit_mask = results >= 0
        if jnp.any(hit_mask):
            return results[jnp.argmax(hit_mask)]

        # Update front for next hop
        current_front = new_front

    # Not found after n hops
    return -1
```

**Note**: This is a simplified version. You may need to adapt from the vectorized version.

---

## Testing Plan

### Test 1: Phase 1 Validation (5 min)
```bash
# Edit block_local_search.py to disable global fallback (see Phase 1 above)
source .venv/bin/activate
python3 production_tracking_threadeda.py 2>&1 | tee logs/phase1_validation.log

# Expected:
# - GPU no longer hangs ✅
# - Script completes successfully ✅
# - Throughput: ~45k p/s ✅
# - Retention: 7.8% ❌ (expected - no fallback)
```

### Test 2: Per-Particle RK4 with 1k Particles (10 min)
```bash
# After implementing Phase 2
# Edit production_tracking_threadeda.py:
# Line 235: N_PARTICLES = 1000
# Line 237: N_TIMESTEPS = 100

source .venv/bin/activate
python3 production_tracking_threadeda.py 2>&1 | tee logs/phase2_per_particle_1k.log

# Expected:
# - JIT compilation succeeds ✅
# - Memory: ~1 MB ✅
# - Throughput: ~40-45k p/s ✅
# - Retention: ~99% (100 steps) ✅
```

### Test 3: Production Test with 100k Particles (30-40 min)
```bash
# Use default settings (100k particles, 2,500 steps)
source .venv/bin/activate
python3 production_tracking_threadeda.py 2>&1 | tee logs/phase2_per_particle_100k.log

# Expected:
# - No OOM errors ✅
# - Memory: ~100 MB ✅
# - Throughput: ~40-42k p/s ✅
# - Retention: ~77.9% ✅
# - Final particles: 77,000-80,000 ✅
```

---

## Summary

**Your Diagnosis**: ✅ **100% Correct**
- Nested JIT/scan architecture causing GPU hang
- Per-particle architecture is the solution

**My Mistake**:
- Introduced scan-based global search to avoid CPU bottleneck
- Created nested scan incompatible with PHASE3A

**Recommended Approach**:
1. **Phase 1** (5 min): Disable fallback → Validate PHASE3A works
2. **Phase 2** (2-3 hours): Per-particle RK4 → Achieve 77.9% retention

**Expected Results**:
- No nested scan issue ✅
- Memory: ~100 MB (safe) ✅
- Retention: ~77.9% (10× improvement) ✅
- Throughput: ~40-42k p/s (acceptable) ✅

**Files to Create/Modify**:
1. `jaxtrace/gpu/search/block_local_search.py`: Add `search_single_particle_with_block_fallback()`
2. `jaxtrace/gpu/search/incremental_search_vectorized.py`: Add `search_level1_multihop_single()`
3. `jaxtrace/gpu/tracking/rk4_gpu_fused.py`: Add per-particle RK4 functions
4. Test with 1k → 100k particles

Ready to implement when you confirm the approach.
