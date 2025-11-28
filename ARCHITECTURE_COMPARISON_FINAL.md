# Architecture Comparison: Per-Particle vs Batch-Separated RK4

**Date**: 2025-11-27
**Status**: 🔴 **CRITICAL DECISION POINT**

---

## Executive Summary

After analyzing three approaches:
1. **Current (Nested Scan)** - ❌ GPU hangs, nested scan issue
2. **Per-Particle RK4 with Embedded Search** - ⚠️ Clean but risky
3. **Batch RK4 with Separated Search** - ✅ **RECOMMENDED**

**Key Insight from Analysis**:
> "Trying to encapsulate both RK4 and multilevel search inside one `single_particle_step` is likely to reintroduce exactly the problems (OOM, poor GPU utilization, 20k p/s cap) we previously diagnosed."

---

## Three Architectures Compared

### Architecture 1: Current Implementation (Nested Scan) ❌

**Structure**:
```python
@jax.jit
def rk4_fused_with_search_and_fallback(positions_gpu, ...):  # ALL particles
    # Stage 1
    element_ids_k1 = search_func(positions_gpu, ...)  # Calls search on ALL particles
        # Inside search_func:
        #   L1: vmap over particles ✅
        #   Global: scan over particles ❌ NESTED!
    velocities_k1 = interpolate(...)

    # Stages 2, 3, 4 - same pattern (4 more nested scan calls)
    # Final search - 5th nested scan call
```

**Issues**:
- ❌ Nested scan: RK4 operates on all particles → search scans over particles again
- ❌ GPU hangs at 100% with no output
- ❌ Deeply nested JIT structure incompatible with XLA

**Verdict**: **BROKEN - Must fix**

---

### Architecture 2: Per-Particle RK4 with Embedded Search ⚠️

**Structure** (my originally proposed fix):
```python
def single_particle_rk4_step(position, element_id, block_id, ...):
    """Complete RK4 for ONE particle."""

    # K1
    elem_k1 = search_single_particle_with_fallback(position, element_id, block_id, ...)
        # Inside search: L0 → L1 → Block-local (scan over block elements)
    v1 = interpolate_single(position, elem_k1, ...)
    pos_k1 = position + 0.5 * dt * v1

    # K2
    elem_k2 = search_single_particle_with_fallback(pos_k1, elem_k1, block_id, ...)
    v2 = interpolate_single(pos_k1, elem_k2, ...)
    pos_k2 = position + 0.5 * dt * v2

    # K3, K4 - same pattern (2 more per-particle searches)
    # Final search (5th per-particle search)

    return pos_final, elem_final

@jax.jit
def batch_rk4(positions, element_ids, block_ids, ...):
    # Single vmap over all particles
    return jax.vmap(single_particle_rk4_step)(positions, element_ids, block_ids, ...)
```

**Pros**:
- ✅ Clean, textbook RK4 structure
- ✅ Single-level vmap (no nested scan at RK4 level)
- ✅ JAX-idiomatic (pure functional)
- ✅ Fixes the nested scan issue

**Cons (Critical Analysis)**:
- ⚠️ **Search called 5× per particle** (k1, k2, k3, k4, final)
- ⚠️ **No early-exit benefit**: JAX can't early-exit per thread in vmap; all branches run, then masked
- ⚠️ **No cross-particle blockwise batching**: Each particle searches independently
- ⚠️ **Memory risk**: If `search_single_particle_with_fallback` has any internal vmap, creates `(N_particles × block_elems)` intermediates
- ⚠️ **Loses algorithmic advantage**: Sequential early-exit (L0 → L1 → L2) outperforms vectorized in PHASE3A analysis
- ⚠️ **Block search per particle**: 450k elements × 5 RK4 stages = 2.25M element checks per particle (even if most hit L0/L1)

**Expected Performance**:
- Memory: ~100 MB (if carefully implemented)
- Throughput: **20-30k p/s** (likely slower than expected due to 5× search overhead)
- Risk: **Reintroduces the 20k p/s cap we diagnosed earlier**

**Verdict**: **Conceptually clean but likely poor performance**

---

### Architecture 3: Batch RK4 with Separated Search ✅

**Structure** (recommended from analysis):
```python
@jax.jit
def rk4_step_with_separated_search(
    positions_gpu,      # (N, 3)
    element_ids_gpu,    # (N,)
    block_ids_gpu,      # (N,)
    velocity_field_gpu,
    dt,
    mesh_gpu,
    block_lists
):
    """
    RK4 with search and integration separated.

    Uses established blockwise batched search kernels (proven in PHASE3A).
    """

    # Stage 1: k1 = f(t, y)
    # Search for all particles at once using blockwise batched kernel
    element_ids_k1 = search_multi_level_batched(
        positions_gpu,      # (N, 3) - ALL particles
        element_ids_gpu,    # (N,) - cached
        block_ids_gpu,      # (N,)
        mesh_gpu,
        block_lists
    )
    # Interpolate velocities for all particles (vmap, cheap)
    velocities_k1 = jax.vmap(interpolate_single)(
        positions_gpu, element_ids_k1, velocity_field_gpu, mesh_gpu
    )
    # RK4 position update (vectorized across all particles)
    positions_k1 = positions_gpu + 0.5 * dt * velocities_k1

    # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
    element_ids_k2 = search_multi_level_batched(
        positions_k1,       # New positions for all particles
        element_ids_k1,     # Use k1 results as cache
        block_ids_gpu,
        mesh_gpu,
        block_lists
    )
    velocities_k2 = jax.vmap(interpolate_single)(positions_k1, element_ids_k2, ...)
    positions_k2 = positions_gpu + 0.5 * dt * velocities_k2

    # Stages 3, 4 - same pattern
    # ...

    # RK4 combination
    positions_final = positions_gpu + (dt/6) * (
        velocities_k1 + 2*velocities_k2 + 2*velocities_k3 + 2*velocities_k4
    )

    # Final search
    element_ids_final = search_multi_level_batched(
        positions_final, element_ids_gpu, block_ids_gpu, mesh_gpu, block_lists
    )

    return positions_final, element_ids_final
```

**What is `search_multi_level_batched`?**
```python
def search_multi_level_batched(
    positions_gpu,      # (N, 3)
    cached_element_ids_gpu,
    block_ids_gpu,
    mesh_gpu,
    block_lists
):
    """
    Batched multi-level search with blockwise optimization.

    Uses PROVEN PHASE3A kernels with early-exit and blockwise batching.
    """

    # L0: Cached element check (vmap over ALL particles)
    element_ids = search_level0_vectorized(positions_gpu, cached_element_ids_gpu, ...)

    # L1: Multi-hop neighbor search (vmap over L0 misses)
    l0_miss_mask = element_ids < 0
    element_ids_l1 = search_level1_multihop_vectorized(
        positions_gpu[l0_miss_mask],
        cached_element_ids_gpu[l0_miss_mask],
        ...
    )
    element_ids = element_ids.at[l0_miss_mask].set(element_ids_l1)

    # L2: Blockwise batched search (for L1 misses)
    l1_miss_mask = element_ids < 0
    n_l1_miss = jnp.sum(l1_miss_mask)

    def do_block_search():
        # Group particles by block, search within blocks
        # Uses PROVEN blockwise batching from initial_search_batch
        return search_blockwise_batched(
            positions_gpu[l1_miss_mask],
            block_ids_gpu[l1_miss_mask],
            block_lists,
            ...
        )

    element_ids_l2 = jax.lax.cond(
        n_l1_miss > 0,
        do_block_search,
        lambda: jnp.full(n_l1_miss, -1, dtype=jnp.int32)
    )
    element_ids = element_ids.at[l1_miss_mask].set(element_ids_l2)

    return element_ids
```

**Key Difference: `search_blockwise_batched` (NOT per-particle)**
```python
def search_blockwise_batched(
    positions,      # (n_failed, 3)
    block_ids,      # (n_failed,)
    block_lists,
    ...
):
    """
    Search within blocks, but BATCHED across particles in each block.

    For each unique block:
    - Extract particles in that block
    - Search those particles within block elements (vmap-based, not scan)
    - Memory: (n_particles_in_block, n_elems_in_block) at worst
    """
    unique_blocks = jnp.unique(block_ids)

    # For each block, batch-search particles in that block
    results = []
    for block_id in unique_blocks:
        mask = block_ids == block_id
        block_positions = positions[mask]

        # Get block elements
        start_idx = block_lists.block_offsets[block_id]
        block_len = block_lists.block_lengths[block_id]
        block_elements = jax.lax.dynamic_slice(
            block_lists.all_elements,
            (start_idx,),
            (block_lists.max_elements_per_block,)
        )[:block_len]

        # Batch search: (n_particles_in_block, n_elems_in_block)
        # Uses vmap over particles, vmap over elements
        # This is SAME as initial_search_batch structure
        def check_particle_in_block(pos):
            def check_element(elem_id):
                return point_in_tet_jax(pos, get_tet_nodes(elem_id, ...))

            inside_mask = jax.vmap(check_element)(block_elements)
            first_hit = jnp.argmax(inside_mask)
            return jnp.where(inside_mask[first_hit], block_elements[first_hit], -1)

        block_results = jax.vmap(check_particle_in_block)(block_positions)
        results.append(block_results)

    # Merge results
    return jnp.concatenate(results)
```

**Pros**:
- ✅ **Uses PROVEN PHASE3A kernels**: `search_level0_vectorized`, `search_level1_multihop_vectorized`
- ✅ **Early-exit benefit preserved**: L0 → L1 → L2 sequential filtering
- ✅ **Blockwise batching**: Memory bounded to `(n_particles_in_block, n_elems_in_block)`
- ✅ **No nested scan**: All search at same level, no scan operations
- ✅ **Separation of concerns**: Search optimized independently from integration
- ✅ **Reuses existing infrastructure**: Minimal new code
- ✅ **Instrumentable**: Can profile search vs integration separately

**Cons**:
- ⚠️ Slightly more complex control flow (5 search calls at top level)
- ⚠️ More state tracking (element_ids for each RK4 stage)

**Expected Performance**:
- Memory: ~100 MB for 100k particles (same as per-particle)
- Throughput: **40-50k p/s** (benefits from batching and early-exit)
- Retention: ~77.9% (same fallback coverage)

**Verdict**: ✅ **RECOMMENDED - Best of both worlds**

---

## Side-by-Side Comparison

| Aspect | Per-Particle RK4 | Batch RK4 + Separated Search |
|--------|------------------|------------------------------|
| **Architecture** | `vmap(single_particle_rk4)` | RK4 stages + batched search calls |
| **Search calls** | 5× per particle (embedded) | 5× batched (separated) |
| **Early-exit** | ❌ No (JAX can't per-thread) | ✅ Yes (sequential L0→L1→L2) |
| **Blockwise batching** | ❌ No (per-particle) | ✅ Yes (cross-particle) |
| **Memory pattern** | `(N, block_elems)` risk | `(n_in_block, elems_in_block)` |
| **Code reuse** | ❌ New single-particle search | ✅ Reuses PHASE3A kernels |
| **Complexity** | Simple (monolithic) | Moderate (separated) |
| **Expected throughput** | 20-30k p/s | 40-50k p/s |
| **Risk** | Reintroduces 20k p/s cap | Low (proven kernels) |

---

## Memory Analysis: Per-Particle vs Blockwise Batched

### Per-Particle Approach

**For each particle** (inside vmap):
```python
# Inside single_particle_rk4_step (vmapped over N particles):
elem = search_single_particle_with_fallback(pos, block_id, ...)
    # Inside search (per particle):
    # Block-local scan: scan over 1-450k elements
```

**Memory when vmapped**:
- JAX vmap materializes intermediate arrays for ALL particles
- If `search_single_particle_with_fallback` touches block elements:
  - Shape: `(N_particles, max_block_size)` = `(100k, 450k)` = **45B elements**
  - Size: 45B × 4 bytes = **180 GB** ❌

**Why this happens**:
- Even though single-particle scan is memory-efficient (1 KB)
- Vmap over 100k particles materializes intermediate arrays
- Each particle's scan creates temporaries
- XLA tries to optimize → allocates large intermediate tensors

---

### Blockwise Batched Approach

**For each unique block**:
```python
# At top level (not inside vmap):
for each block:
    particles_in_block = positions[block_mask]  # 10-5000 particles
    block_elements = ...  # 1k-450k elements

    # Batch search: vmap over particles in THIS block
    results = jax.vmap(check_in_block)(particles_in_block)
        # Inside: vmap over block elements
```

**Memory for each block**:
- Light blocks: `(10, 2k)` = 20k elements = 80 KB
- Medium blocks: `(100, 50k)` = 5M elements = 20 MB
- Heavy blocks: `(1000, 450k)` = 450M elements = 1.8 GB

**Total memory** (peak for heaviest block):
- **1.8 GB** (acceptable) ✅

**Why this works**:
- Process one block at a time (loop over unique blocks)
- Each block processed independently
- No cross-block intermediate arrays
- Memory bounded by single block size

---

## Critical Performance Issue: 5× Search Overhead

### Per-Particle Approach

**Search calls per particle**:
- K1: 1 search (L0 → L1 → maybe Block)
- K2: 1 search
- K3: 1 search
- K4: 1 search
- Final: 1 search
- **Total: 5 searches per particle per timestep**

**If no early-exit**:
- Each search potentially checks entire block (1-450k elements)
- Even with L0/L1 cache, k2/k3/k4 positions may miss cache
- 100k particles × 5 searches × 450k elements (worst case)
- **Loses the early-exit algorithmic advantage**

**JAX vmap limitation**:
- Can't early-exit per thread (all branches run, then masked)
- So even if L0 hits for 99% of particles:
  - All 100k particles execute L1 code
  - All 100k particles execute Block code
  - Results masked at the end
- **No actual early-exit benefit**

---

### Batch RK4 with Separated Search

**Search calls per timestep**:
- K1: 1 batched search (ALL particles)
  - L0: 100k particles, 99% hit → 1k to L1
  - L1: 1k particles, 99% hit → 10 to Block
  - Block: 10 particles only
- K2: 1 batched search (ALL particles)
  - Similar filtering
- K3, K4, Final: Same pattern
- **Total: 5 batched searches, but with early-exit filtering**

**With early-exit**:
- L0: 100k particles checked (cheap: 1 tet per particle)
- L1: 1k particles checked (~84 neighbors each)
- Block: 10 particles × 450k elements = 4.5M checks
- **Orders of magnitude less work than 5× per-particle**

**Estimated work**:
- Per-particle (no early-exit): 100k × 5 × 450k = **225B element checks**
- Batch separated (with early-exit): 100k × 5 × 1 + 1k × 5 × 84 + 10 × 5 × 450k = **~23M element checks**
- **Ratio: 10,000× less work** ✅

---

## Recommendation: Architecture 3 (Batch RK4 + Separated Search)

### Why This is Best

1. **Fixes nested scan issue** (like per-particle approach)
2. **Preserves early-exit benefit** (unlike per-particle)
3. **Enables blockwise batching** (unlike per-particle)
4. **Reuses proven PHASE3A kernels** (minimal new code)
5. **Memory-safe** (bounded by block size)
6. **Expected 40-50k p/s** (vs 20-30k for per-particle)

### Implementation Plan (Updated)

**Phase 1: Validate PHASE3A Works** (5 minutes)
- Disable global fallback
- Confirm GPU no longer hangs
- Expected: 45k p/s, 7.8% retention

**Phase 2: Implement Batch RK4 + Separated Search** (1-2 hours)

**Step 2.1: Create blockwise batched search** (30 min)
```python
# File: jaxtrace/gpu/search/block_local_search.py

def search_blockwise_batched(
    positions,      # (n_failed, 3)
    block_ids,      # (n_failed,)
    block_lists,
    mesh_gpu
):
    """
    Batch search within blocks for failed particles.

    Groups particles by block, searches within each block using vmap.
    Memory: (n_particles_in_block, n_elems_in_block) per block.
    """
    # Implementation above
    ...

def search_multi_level_batched(
    positions_gpu,
    cached_element_ids_gpu,
    block_ids_gpu,
    mesh_gpu,
    block_lists
):
    """
    Multi-level batched search: L0 → L1 → Blockwise L2.

    Uses proven PHASE3A kernels with early-exit.
    """
    # L0 vectorized
    element_ids = search_level0_vectorized(...)

    # L1 multi-hop vectorized
    l0_miss_mask = element_ids < 0
    element_ids_l1 = search_level1_multihop_vectorized(
        positions_gpu[l0_miss_mask], ...
    )
    element_ids = element_ids.at[l0_miss_mask].set(element_ids_l1)

    # L2 blockwise batched
    l1_miss_mask = element_ids < 0
    n_l1_miss = jnp.sum(l1_miss_mask)

    element_ids_l2 = jax.lax.cond(
        n_l1_miss > 0,
        lambda: search_blockwise_batched(
            positions_gpu[l1_miss_mask],
            block_ids_gpu[l1_miss_mask],
            block_lists,
            mesh_gpu
        ),
        lambda: jnp.full(n_l1_miss, -1, dtype=jnp.int32)
    )
    element_ids = element_ids.at[l1_miss_mask].set(element_ids_l2)

    return element_ids
```

**Step 2.2: Create batch RK4 with separated search** (30 min)
```python
# File: jaxtrace/gpu/tracking/rk4_gpu_fused.py

@jax.jit
def rk4_step_batch_separated_search(
    positions_gpu,
    element_ids_gpu,
    block_ids_gpu,
    velocity_field_gpu,
    dt,
    mesh_gpu,
    block_lists
):
    """
    RK4 with separated, batched search calls.

    Uses proven PHASE3A search kernels at each stage.
    """
    # Stage 1: k1
    element_ids_k1 = search_multi_level_batched(
        positions_gpu, element_ids_gpu, block_ids_gpu, mesh_gpu, block_lists
    )
    velocities_k1 = jax.vmap(interpolate_single)(
        positions_gpu, element_ids_k1, velocity_field_gpu, mesh_gpu
    )
    positions_k1 = positions_gpu + 0.5 * dt * velocities_k1

    # Stages 2, 3, 4 - same pattern
    # ...

    # RK4 combination
    positions_final = positions_gpu + (dt/6) * (
        velocities_k1 + 2*velocities_k2 + 2*velocities_k3 + velocities_k4
    )

    # Final search
    element_ids_final = search_multi_level_batched(
        positions_final, element_ids_gpu, block_ids_gpu, mesh_gpu, block_lists
    )

    return positions_final, element_ids_final
```

**Step 2.3: Create production wrapper** (15 min)
- Wrapper for CPU-GPU transfers
- Same interface as before

**Step 2.4: Test with 1k → 100k particles** (10-40 min)

---

## Summary

**Decision**: **Architecture 3 (Batch RK4 + Separated Search)** ✅

**Rationale**:
1. Fixes nested scan issue (like per-particle)
2. Preserves algorithmic advantages (early-exit, blockwise batching)
3. Reuses proven PHASE3A infrastructure
4. Expected 40-50k p/s (2× better than per-particle)
5. Memory-safe (1.8 GB peak vs 180 GB risk)

**Updated Implementation**: See NEXT_STEPS_ARCHITECTURE_FIX_V2.md

**Key Insight**:
> "The more robust, scalable design is: RK4 + interpolation vectorized across particles, Search blockwise batched with separate kernel calls per stage."

This respects all OOM findings, aligns with PHASE3A architecture, and uses JAX strengths: large batched kernels, clear separation, static blockwise memory.
