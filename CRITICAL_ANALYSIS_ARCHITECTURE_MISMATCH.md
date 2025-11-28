# Critical Analysis: Architecture Mismatch and Nested Scan Issue

**Date**: 2025-11-27
**Status**: 🔴 **CRITICAL BLOCKER IDENTIFIED**
**Issue**: Nested scan architecture causing GPU hang

---

## Executive Summary

**The Problem**: Current implementation creates a **nested scan architecture** that causes GPU to hang at 100% utilization with no progress. This violates the PHASE3A design principles.

**Root Cause**: Scan-based global search was introduced inside the RK4 function, creating:
```
RK4 (operates on ALL particles)
  → 5× search_func() calls per timestep
    → L1 multi-hop (vmap over particles) ✅ OK
    → Global fallback (scan over particles) ❌ NESTED SCAN!
      → For each particle: vmap over 3.5M elements
```

**User's Diagnosis**: ✅ **CORRECT** - User identified this as nested JIT issue and suggested per-particle architecture.

**PHASE3A Architecture**: Uses **single-level vmap parallelism**, not nested scans.

---

## Architecture Comparison

### PHASE3A Design (Original - High Performance)

**File**: [docs/gpu/PHASE3A_COMPLETE_WITH_FUSED_RK4.md](docs/gpu/PHASE3A_COMPLETE_WITH_FUSED_RK4.md:226-470)

**Architecture**:
```python
@jax.jit
def rk4_step_gpu_fused(
    positions_initial_gpu,      # (N, 3) - ALL particles
    element_ids_initial_gpu,    # (N,) - ALL particles
    ...
):
    """Complete RK4 step entirely on GPU - ALL 4 STAGES ON ALL PARTICLES."""

    # Stage 1: k1 at x_n
    v1_gpu = interpolate_velocity_batch_gpu(
        positions_initial_gpu,  # (N, 3) - operates on ALL particles
        element_ids_initial_gpu,
        ...
    )

    # Stage 2: k2 at x_n + dt/2 * k1
    pos2_gpu, elem_ids_2_gpu, v2_gpu = rk4_stage_gpu(
        positions_initial_gpu,  # (N, 3) - operates on ALL particles
        element_ids_initial_gpu,
        v1_gpu,
        dt, 0.5, ...
    )

    # ... Stages 3, 4 ...

    # RK4 combination: x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    positions_final_gpu = positions_initial_gpu + (dt / 6.0) * (
        v1_gpu + 2.0*v2_gpu + 2.0*v3_gpu + v4_gpu
    )

    # Final search at new positions
    element_ids_final_gpu = search_gpu_fused(
        positions_final_gpu,  # (N, 3) - operates on ALL particles
        element_ids_initial_gpu,
        ...
    )

    return positions_final_gpu, element_ids_final_gpu
```

**Key Features**:
- ✅ Single `@jax.jit` function for entire RK4
- ✅ Operates on ALL particles at once (no per-particle loops)
- ✅ Uses `vmap` for parallelism (not `scan`)
- ✅ Search function (`search_gpu_fused`) uses **vmap-based vectorization**, not scan

**Search Architecture (PHASE3A)**:
```python
@jax.jit
def search_gpu_fused(
    positions_gpu,              # (N, 3) - ALL particles
    cached_element_ids_gpu,     # (N,)
    ...
) -> jax.Array:
    """Fused GPU search: L0 + L1, all on GPU."""

    # L0: Check cached elements (vmap over ALL particles)
    element_ids_gpu = search_level0_vectorized(
        positions_gpu,           # (N, 3) - vmap processes all at once
        cached_element_ids_gpu,
        ...
    )

    # L1: Check neighbors for L0 misses (vmap over failed particles)
    l0_miss_mask_gpu = element_ids_gpu < 0
    element_ids_l1_gpu = search_level1_extended_vectorized(
        positions_gpu[l0_miss_mask_gpu],  # Only failed particles
        cached_element_ids_gpu[l0_miss_mask_gpu],
        ...
    )

    # Update results (on GPU)
    element_ids_gpu = jnp.where(l1_full_gpu >= 0, l1_full_gpu, element_ids_gpu)

    return element_ids_gpu
```

**Parallelism Model**: ✅ **VMAP-BASED**
- `search_level0_vectorized`: Uses `jax.vmap(point_in_tet_jax)` over all particles
- `search_level1_extended_vectorized`: Uses `jax.vmap` for multi-hop neighbor checks
- **NO SCAN OPERATIONS** - Pure vmap parallelism

---

### Current Implementation (With Global Fallback - BROKEN)

**File**: [jaxtrace/gpu/tracking/rk4_gpu_fused.py](jaxtrace/gpu/tracking/rk4_gpu_fused.py:704-773)

**Architecture**:
```python
@jax.jit
def rk4_fused_with_search_and_fallback(
    positions_gpu,      # (N, 3)
    element_ids_gpu,    # (N,)
    block_ids_gpu,      # (N,)
    ...
):
    """GPU-fused RK4 with block fallback."""

    # Stage 1: k1 = f(t, y)
    element_ids_k1 = search_func(  # ❌ PROBLEM: search_func contains scan!
        positions_gpu,              # (N, 3) - ALL particles
        element_ids_gpu,
        block_ids_gpu,
        ...
    )
    velocities_k1 = interpolate_velocity_batch_gpu(...)
    positions_k1 = positions_gpu + 0.5 * dt * velocities_k1

    # Stages 2, 3, 4 - same pattern (4 more calls to search_func)
    # ...

    # Final search (5th call to search_func)
    element_ids_final_gpu = search_func(positions_final_gpu, ...)

    return positions_final_gpu, element_ids_final_gpu
```

**Where search_func comes from**:
```python
# In rk4_step_gpu_fused_with_block_fallback():
search_func = create_search_with_block_fallback(n_hops, block_lists)
```

**What search_func actually does**:
```python
# From block_local_search.py:421-482
@jax.jit
def search_with_fallback(
    positions_gpu,         # (N, 3)
    cached_element_ids_gpu,
    block_ids_gpu,
    ...
):
    # Tier 1: L1 multi-hop (vmap - OK)
    element_ids = search_level1_multihop_vectorized(
        positions_gpu,     # (N, 3) - vmap over all particles ✅
        cached_element_ids_gpu,
        ...
    )

    # Tier 2: Global fallback (scan - PROBLEM!)
    failed_mask = element_ids < 0

    global_results = search_global_gpu_native_scan(
        positions_gpu,     # (N, 3) - ALL particles (not just failed!)
        failed_mask,       # (N,) bool
        ...
    )  # ❌ Uses jax.lax.scan internally!

    element_ids = jnp.where(
        failed_mask & (global_results >= 0),
        global_results,
        element_ids
    )

    return element_ids
```

**The Problematic Global Search**:
```python
# From block_local_search.py:303-385
@jax.jit
def search_global_gpu_native_scan(
    positions,       # (N, 3)
    search_mask,     # (N,) bool
    ...
):
    """GPU-native global search using scan over particles."""

    def search_one_particle(carry, position_and_mask):
        position, should_search = position_and_mask

        def do_search(_):
            # Vmap over 3.5M elements for THIS ONE particle
            inside_mask = jax.vmap(lambda e: check_element(position, e))(
                jnp.arange(n_elements)  # 3.5M elements
            )
            first_hit = jnp.argmax(inside_mask)
            return jnp.where(inside_mask[first_hit], first_hit, -1)

        def skip_search(_):
            return -1

        elem_id = jax.lax.cond(should_search, do_search, skip_search, None)
        return carry, elem_id

    # ❌ NESTED SCAN - This is the problem!
    _, element_ids = jax.lax.scan(
        search_one_particle,
        None,
        (positions, search_mask)  # Scan over ALL N particles
    )

    return element_ids
```

**The Nested Structure**:
```
RK4 (JIT-compiled, operates on ALL particles):
  ↓ Calls search_func 5 times per timestep

  search_with_fallback (JIT-compiled):
    ↓ L1 multi-hop (vmap over particles) ✅
    ↓ Global fallback:

      search_global_gpu_native_scan (JIT-compiled):
        ↓ jax.lax.scan over ALL N particles ❌
          ↓ For each particle:
            ↓ jax.vmap over 3.5M elements
```

**Why This Hangs**:
- RK4 already operates on all particles as a batch
- Global search then scans over those same particles again
- JAX's XLA compiler tries to optimize this nested structure
- Creates deeply nested kernel graph that GPU struggles to execute
- Result: 100% GPU load with no actual progress

---

## Why the Transition from Vmap to Scan Happened

### Timeline of Changes

**Original Block-Local Search (Lines 179-300)**:
```python
def create_block_local_search_func(block_lists):
    @jax.jit
    def search_batch_in_blocks(
        positions,      # (N, 3)
        block_ids,      # (N,)
        ...
    ):
        def search_one(pos, block_id):
            # Search within this particle's block
            return search_single_particle_in_block_closure(...)

        # ❌ Vmap over 100k particles, each scanning 450k elements
        return jax.vmap(search_one)(positions, block_ids)

    return search_batch_in_blocks
```

**Problem**: 218 GB OOM (vmap over 100k particles × scan over 450k elements each)

**My Fix Attempt**: Replaced with global search using `jax.lax.scan`

**Why I Chose Scan**:
1. User explicitly said: "CPU based global search may cause severe reduction of GPU load"
2. I wanted to avoid Python for-loop (CPU-based)
3. I thought `jax.lax.scan` would be GPU-native alternative
4. Scan is memory-efficient for single particle (3.5 MB vs 218 GB)

**What I Missed**:
- PHASE3A uses vmap-based parallelism throughout
- Scan over particles is sequential (not parallel)
- Scan inside RK4 creates nested structure
- **User was right to suggest per-particle architecture instead**

---

## User's Proposed Solution (CORRECT)

**User's Message**:
> "Do you think the correct GPU implementation should be like:
> ```
> def single_particle_step:
>    single_particle_k1-k4_element_search
>    single_particle_interpolations
>    single_particle_RK4
>    single_particle_elemet_update
> Time marching loop:
>    paralelized_over_particles(single_particle_step)
> ```"

**Analysis**: ✅ **THIS IS CORRECT**

**Why This Would Work**:
```python
@jax.jit
def single_particle_rk4_step(
    position,         # (3,) - ONE particle
    element_id,       # scalar
    block_id,         # scalar
    velocity_field,
    ...
):
    """Complete RK4 for a single particle."""

    # Stage 1: k1
    elem_id_k1 = search_single_particle(position, element_id, block_id, ...)
    v1 = interpolate_single(position, elem_id_k1, ...)
    pos_k1 = position + 0.5 * dt * v1

    # Stages 2, 3, 4
    elem_id_k2 = search_single_particle(pos_k1, elem_id_k1, block_id, ...)
    v2 = interpolate_single(pos_k1, elem_id_k2, ...)
    # ... etc ...

    # RK4 combination
    pos_final = position + (dt/6) * (v1 + 2*v2 + 2*v3 + v4)
    elem_id_final = search_single_particle(pos_final, element_id, block_id, ...)

    return pos_final, elem_id_final

# Parallelize over all particles using vmap
@jax.jit
def batch_rk4_step(
    positions,        # (N, 3)
    element_ids,      # (N,)
    block_ids,        # (N,)
    ...
):
    """RK4 for ALL particles using vmap."""

    # Single level of parallelism - vmap over particles
    return jax.vmap(single_particle_rk4_step)(
        positions, element_ids, block_ids, ...
    )
```

**Key Advantages**:
- ✅ Single level of parallelism (vmap, not scan)
- ✅ Each particle independently executes full RK4 with searches
- ✅ No nested scans
- ✅ Matches PHASE3A philosophy
- ✅ Allows per-particle fallback strategies (block-local or global)

---

## How to Implement Per-Particle Fallback

**Option 1: Per-Particle Global Search (Memory-Intensive)**
```python
def search_single_particle_with_fallback(
    position,         # (3,)
    element_id,       # scalar
    block_id,         # scalar
    ...
):
    # L0: Check cached element
    result = search_level0_single(position, element_id, ...)

    # L1: Multi-hop neighbor search
    result = jax.lax.cond(
        result < 0,
        lambda: search_level1_multihop_single(position, element_id, ...),
        lambda: result
    )

    # L2: Global fallback (vmap over 3.5M elements for THIS particle)
    result = jax.lax.cond(
        result < 0,
        lambda: search_global_single_particle(position, ...),  # Vmap over 3.5M
        lambda: result
    )

    return result
```

**Memory**: 3.5 MB per particle × N particles (materialized sequentially via vmap)
**Performance**: Global search only triggered for failed particles (0.1%)
**Expected**: Similar to current scan-based approach but no nested scan issue

**Option 2: Per-Particle Block Search (Memory-Efficient)**
```python
def search_single_particle_with_fallback(
    position,         # (3,)
    element_id,       # scalar
    block_id,         # scalar
    block_lists,      # Block element lists
    ...
):
    # L0: Check cached element
    result = search_level0_single(position, element_id, ...)

    # L1: Multi-hop neighbor search
    result = jax.lax.cond(
        result < 0,
        lambda: search_level1_multihop_single(position, element_id, ...),
        lambda: result
    )

    # L2: Block-local fallback (scan over 1-450k elements in THIS particle's block)
    result = jax.lax.cond(
        result < 0,
        lambda: search_block_local_single_particle(position, block_id, block_lists, ...),
        lambda: result
    )

    return result
```

**Memory**: 1 KB per particle (scan over block elements)
**Performance**: Block search 2-50 ms per failed particle (vs 50-100 ms global)
**Expected**: Better performance than global, no OOM

**Why This Would Work**:
- Block search for single particle: 1 KB (already proven in 1k test)
- Vmap over 100k particles: 100k × 1 KB = 100 MB (sequential materialization)
- **NO nested vmap/scan** - Single vmap at top level only

---

## Root Cause Analysis Summary

| Aspect | PHASE3A (Original) | Current Implementation | Issue |
|--------|-------------------|----------------------|-------|
| **RK4 Architecture** | Single JIT function, operates on ALL particles | Same | ✅ OK |
| **Search Parallelism** | Vmap-based (L0, L1) | L0/L1: Vmap ✅<br>Global fallback: Scan ❌ | ❌ MISMATCH |
| **Nesting Level** | Single vmap at RK4 level | RK4 calls search_func with scan inside | ❌ NESTED |
| **Fallback Strategy** | Block-based L2 (if unmapped) | Global scan (all particles) | ❌ SCAN-BASED |
| **GPU Utilization** | 60-80% (Phase 3a target) | 100% stuck (compilation hang) | ❌ BROKEN |
| **Memory Model** | Vmap materializes in parallel | Scan materializes sequentially | ⚠️ INCOMPATIBLE |

**The Fundamental Issue**:
- PHASE3A designed for **vmap-based parallelism** at a single level
- Current implementation introduces **scan-based sequencing** inside vmap context
- JAX's XLA compiler cannot efficiently compile nested vmap/scan structures
- Result: GPU hangs during compilation/execution

---

## Solution Options

### Option A: Disable Global Fallback (Simplest - Accept Losses)
```python
# In create_search_with_block_fallback():
@jax.jit
def search_with_fallback(positions_gpu, ...):
    # Tier 1: L1 multi-hop ONLY (no fallback)
    element_ids = search_level1_multihop_vectorized(positions_gpu, ...)
    return element_ids

return search_with_fallback
```

**Pros**:
- ✅ No nested scan issue
- ✅ Matches PHASE3A architecture
- ✅ Immediate fix (5 minutes)

**Cons**:
- ❌ Accepts 0.1% particle loss per timestep
- ❌ Final retention: 7.8% (vs 77.9% target)
- ❌ Doesn't address user's original goal

**When to Use**: Quick validation that PHASE3A architecture works

---

### Option B: Restructure to Per-Particle RK4 with Global Fallback (User's Suggestion)
```python
@jax.jit
def single_particle_rk4_with_global_fallback(
    position, element_id, block_id, ...
):
    """Complete RK4 for one particle with global fallback."""
    # ... RK4 stages with global search fallback per stage ...
    return pos_final, elem_id_final

@jax.jit
def batch_rk4_with_fallback(positions, element_ids, block_ids, ...):
    """Vmap over all particles."""
    return jax.vmap(single_particle_rk4_with_global_fallback)(
        positions, element_ids, block_ids, ...
    )
```

**Pros**:
- ✅ Single-level vmap (no nested scan)
- ✅ Matches user's suggested architecture
- ✅ Global fallback works per-particle (3.5 MB each)
- ✅ Target retention: 77.9%

**Cons**:
- ⚠️ Vmap over 100k particles × global search (3.5 MB each)
- ⚠️ May still cause memory issues (100k × 3.5 MB = 350 GB if materialized at once)
- ⚠️ Requires full restructure of RK4 function

**Memory Risk**: JAX might try to materialize all 100k global searches at once

---

### Option C: Restructure to Per-Particle RK4 with Block Fallback (Best Balance)
```python
@jax.jit
def single_particle_rk4_with_block_fallback(
    position, element_id, block_id, block_lists, ...
):
    """Complete RK4 for one particle with block-local fallback."""

    def search_with_fallback_single(pos, elem_id, block_id):
        # L0 cached
        result = search_level0_single(pos, elem_id, ...)

        # L1 multi-hop
        result = jax.lax.cond(
            result < 0,
            lambda: search_level1_multihop_single(pos, elem_id, ...),
            lambda: result
        )

        # L2 block-local (scan over 1-450k elements in THIS particle's block)
        result = jax.lax.cond(
            result < 0,
            lambda: search_block_single(pos, block_id, block_lists, ...),
            lambda: result
        )

        return result

    # K1
    elem_k1 = search_with_fallback_single(position, element_id, block_id)
    v1 = interpolate_single(position, elem_k1, ...)
    pos_k1 = position + 0.5 * dt * v1

    # K2, K3, K4 - same pattern
    # ...

    # Final
    pos_final = position + (dt/6) * (v1 + 2*v2 + 2*v3 + v4)
    elem_final = search_with_fallback_single(pos_final, element_id, block_id)

    return pos_final, elem_final

@jax.jit
def batch_rk4_with_block_fallback(positions, element_ids, block_ids, ...):
    """Vmap over all particles."""
    return jax.vmap(single_particle_rk4_with_block_fallback)(
        positions, element_ids, block_ids, ...
    )
```

**Pros**:
- ✅ Single-level vmap (no nested scan)
- ✅ Memory-efficient: 1 KB per particle × 100k = 100 MB
- ✅ Target retention: 77.9%
- ✅ Matches user's architecture suggestion
- ✅ Block search already proven to work for 1k particles

**Cons**:
- ⚠️ Requires full restructure of RK4 function
- ⚠️ Each particle scans its block (1-450k elements) - acceptable overhead

**Memory**: 100 MB (safe, proven)
**Performance**: Similar to current scan-based approach but no nested issue

---

### Option D: End-of-Timestep Global Fallback (Compromise)
```python
# In time marching loop:
for step in range(N_TIMESTEPS):
    # RK4 with L1 multi-hop only (no fallback during RK4)
    particle_data, stats = rk4_step_gpu_fused_for_production(
        particle_data, velocity_field, dt, mesh_gpu, n_hops=3
    )

    # After RK4: Apply global fallback for failed particles
    failed_mask = particle_data.element_ids < 0
    n_failed = failed_mask.sum()

    if n_failed > 0:
        # CPU-based global search for failed particles only
        failed_positions = particle_data.positions[failed_mask]
        elem_ids_fallback, _, _ = initial_search_batch(
            failed_positions, ...  # Block-based or global search
        )
        particle_data.element_ids[failed_mask] = elem_ids_fallback
```

**Pros**:
- ✅ No nested scan (RK4 uses vmap-only)
- ✅ Minimal code changes
- ✅ Global/block fallback only once per timestep (not 5×)
- ✅ CPU-based fallback acceptable (only 100 particles)

**Cons**:
- ⚠️ Particles may be deactivated during RK4 if L1 fails
- ⚠️ Fallback applied after timestep (may miss some particles)
- ⚠️ Retention improvement might be less than 77.9%

**Performance**: Negligible impact (100 particles × 50 ms = 5 seconds total)

---

## Recommended Approach

### Phase 1: Immediate Fix (Test PHASE3A Architecture)
**Goal**: Validate that PHASE3A architecture works without fallback

**Action**: Disable global fallback (Option A)
```python
# In block_local_search.py, line 421:
def create_search_with_block_fallback(n_hops=3, block_lists=None):
    @jax.jit
    def search_with_fallback(positions_gpu, ...):
        # L1 multi-hop ONLY (no fallback)
        element_ids = search_level1_multihop_vectorized(positions_gpu, ...)
        return element_ids

    return search_with_fallback
```

**Expected Results**:
- ✅ GPU no longer hangs
- ✅ Throughput: ~45k p/s
- ❌ Retention: 7.8% (baseline)

**Time**: 5 minutes
**Purpose**: Confirm nested scan was the issue

---

### Phase 2: Implement Per-Particle RK4 with Block Fallback (Option C)
**Goal**: Achieve 77.9% retention without nested scan

**Steps**:

1. **Create single-particle search with block fallback**:
```python
# In block_local_search.py:
@jax.jit
def search_single_particle_with_block_fallback(
    position,         # (3,)
    element_id,       # scalar
    block_id,         # scalar
    node_positions,
    connectivity,
    element_neighbors,
    block_lists
):
    """Search for one particle with L0 → L1 → Block fallback."""
    # L0: Cached element
    result = point_in_tet_jax(position, get_tet_nodes(element_id, ...))
    result = jnp.where(result, element_id, -1)

    # L1: Multi-hop neighbors
    def do_l1():
        return search_level1_multihop_single(position, element_id, ...)
    result = jax.lax.cond(result < 0, do_l1, lambda: result)

    # L2: Block-local fallback
    def do_block():
        return search_block_single_particle(position, block_id, block_lists, ...)
    result = jax.lax.cond(result < 0, do_block, lambda: result)

    return result
```

2. **Create single-particle RK4**:
```python
# In rk4_gpu_fused.py:
def single_particle_rk4_step(
    position, element_id, block_id,
    velocity_field, dt,
    node_positions, connectivity, element_neighbors, block_lists
):
    """Complete RK4 for one particle."""

    # Define search function for this particle
    def search(pos, elem_id):
        return search_single_particle_with_block_fallback(
            pos, elem_id, block_id,
            node_positions, connectivity, element_neighbors, block_lists
        )

    # K1
    elem_k1 = search(position, element_id)
    v1 = interpolate_single(position, elem_k1, velocity_field, ...)
    pos_k1 = position + 0.5 * dt * v1

    # K2
    elem_k2 = search(pos_k1, elem_k1)
    v2 = interpolate_single(pos_k1, elem_k2, velocity_field, ...)
    pos_k2 = position + 0.5 * dt * v2

    # K3
    elem_k3 = search(pos_k2, elem_k2)
    v3 = interpolate_single(pos_k2, elem_k3, velocity_field, ...)
    pos_k3 = position + dt * v3

    # K4
    elem_k4 = search(pos_k3, elem_k3)
    v4 = interpolate_single(pos_k3, elem_k4, velocity_field, ...)

    # RK4 combination
    pos_final = position + (dt/6) * (v1 + 2*v2 + 2*v3 + v4)
    elem_final = search(pos_final, element_id)

    return pos_final, elem_final
```

3. **Vmap over all particles**:
```python
@jax.jit
def batch_rk4_per_particle(
    positions,        # (N, 3)
    element_ids,      # (N,)
    block_ids,        # (N,)
    velocity_field,
    dt,
    node_positions,
    connectivity,
    element_neighbors,
    block_lists
):
    """RK4 for all particles using vmap."""

    # Single level of parallelism
    return jax.vmap(single_particle_rk4_step)(
        positions, element_ids, block_ids,
        velocity_field, dt,
        node_positions, connectivity, element_neighbors, block_lists
    )
```

**Expected Results**:
- ✅ No nested scan (single vmap level)
- ✅ Memory: ~100 MB (safe)
- ✅ Retention: ~77.9%
- ✅ Throughput: ~40-42k p/s (acceptable)

**Time**: 2-3 hours implementation + testing

---

## Next Steps

1. **Read current implementation** of `search_level1_multihop_vectorized` to understand how to create single-particle version

2. **Implement Phase 1** (disable fallback) to confirm issue is nested scan

3. **Implement Phase 2** (per-particle RK4) for final solution

4. **Test with 1k particles** to validate memory and performance

5. **Production test with 100k particles** to measure retention improvement

---

## Conclusion

**User's Diagnosis**: ✅ **100% CORRECT**
- Identified nested JIT issue causing GPU hang
- Suggested per-particle architecture as solution
- This matches PHASE3A design principles

**My Mistake**:
- Implemented scan-based global search to avoid CPU bottleneck
- Created nested scan architecture incompatible with PHASE3A
- Should have restructured to per-particle RK4 instead

**Correct Solution**: **Option C - Per-Particle RK4 with Block Fallback**
- Single-level vmap parallelism
- Memory-efficient (100 MB)
- Target retention: 77.9%
- Matches user's suggested architecture

**Implementation Priority**:
1. Phase 1: Disable fallback (5 min) - Validate issue
2. Phase 2: Per-particle RK4 (2-3 hours) - Final solution
