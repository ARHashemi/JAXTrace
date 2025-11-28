# Particle Loss Solution - Options Analysis

## Current Situation

**Test Results:** [logs/test_PHASE3A_production_threadeda.log](logs/test_PHASE3A_production_threadeda.log)

### Observed Particle Loss
- **Initial particles:** 103,671 (98.7% of 105k after initial assignment)
- **Final particles:** 16,685 (16.1% retention after 2,500 timesteps)
- **Loss pattern:** Steady decline from 95k → 16k particles
  - Step 100: 95,103 (91.7%)
  - Step 500: 64,292 (62.0%)
  - Step 1000: 49,730 (48.0%)
  - Step 1500: 33,060 (31.9%)
  - Step 2500: 16,685 (16.1%)

### Root Cause
**3-hop L1 search misses in refinement regions:**
- L1 3-hop: ~99.0-99.5% hit rate (estimated from loss pattern)
- **Per-timestep miss rate:** ~0.5% (particle leaves 84-neighbor region)
- **Cumulative effect:** (0.995)^2500 = 0.000008 → exponential decay
- **Refinement regions:** High-gradient flows push particles faster than neighbor search can track

### Critical Understanding: 5 Search Points per Timestep

In RK4, search is needed **5 times per timestep** (not just at the end):
1. **k1:** Search at current position
2. **k2:** Search at position + 0.5*dt*k1
3. **k3:** Search at position + 0.5*dt*k2
4. **k4:** Search at position + dt*k3
5. **Final:** Search at position + dt/6*(k1 + 2*k2 + 2*k3 + k4)

**Implication:** Any particle that fails search at k1-k4 will get invalid velocity and propagate error to final position.

---

## Options for Particle Loss Mitigation

### Option 1: Global Search ONLY at Final Position (OUTSIDE RK4)

**Concept:** Keep PHASE3A (L0+L1) inside RK4, apply L2 global fallback AFTER RK4 completes.

**Implementation:**
```python
# Inside production loop
positions, element_ids = rk4_step_gpu_fused_for_production(...)  # L0+L1 only

# OUTSIDE RK4: Apply global fallback for failures
failed_mask = element_ids < 0
if jnp.any(failed_mask):
    global_results = search_global_gpu_native_scan(
        positions, failed_mask, node_positions_gpu, connectivity_gpu
    )
    element_ids = jnp.where(failed_mask & (global_results >= 0),
                           global_results, element_ids)
```

**Architecture:**
- **RK4:** Pure PHASE3A (single @jax.jit, L0+L1 vmap)
- **L2 fallback:** Separate @jax.jit (scan over failed particles)
- **No nested JIT:** Two separate JIT kernels called sequentially

**Pros:**
✅ No nested JIT/scan (avoids GPU hang)
✅ Preserves PHASE3A performance inside RK4
✅ Recovers particles that failed at FINAL position
✅ Uses existing `search_global_gpu_native_scan` implementation
✅ Minimal code changes (add 5 lines to production script)

**Cons:**
❌ **Does NOT fix k1-k4 failures** - particles with mid-stage failures still lost
❌ Only recovers ~50% of lost particles (those that failed at final position)
❌ Adds 1 extra transfer per timestep (download failed mask)
❌ L2 scan overhead: ~5-10ms per timestep (if ~0.5% fail = ~500 particles)

**Performance Impact:**
- Throughput: 40-50k p/s → 35-45k p/s (10-15% slower)
- Retention: 16% → ~35-40% (2× improvement, but not enough)

**Verdict:** ⚠️ **Partial solution** - helps but doesn't solve k1-k4 failures.

---

### Option 2: Global Search at ALL 5 RK4 Positions (INSIDE RK4)

**Concept:** Apply L2 global fallback at each of the 5 search points inside RK4.

**Implementation:**
```python
@jax.jit
def rk4_fused_with_global_fallback(...):
    # k1 search
    element_ids_k1 = search_L0_L1(...)  # L0+L1 vmap
    failed_mask_k1 = element_ids_k1 < 0

    # L2 fallback for k1 failures
    global_k1 = search_global_gpu_native_scan(
        positions_gpu, failed_mask_k1, node_positions_gpu, connectivity_gpu
    )
    element_ids_k1 = jnp.where(failed_mask_k1, global_k1, element_ids_k1)

    velocities_k1 = interpolate(...)
    # ... repeat for k2, k3, k4, final
```

**Architecture:**
- **Single @jax.jit:** All RK4 stages + L2 fallbacks
- **5× L2 scan calls per timestep** (one per search point)
- **Nested scan:** RK4 vmap → L2 scan (THIS IS THE PROBLEM)

**Pros:**
✅ Fixes k1-k4 failures (comprehensive solution)
✅ Expected retention: 75-85% (vs 16% baseline)
✅ Uses existing `search_global_gpu_native_scan`

**Cons:**
❌ **CRITICAL: Nested scan architecture** - RK4 operates on all particles (implicit outer loop), L2 uses scan
❌ **GPU hang risk** - This is exactly what caused 100% GPU hang before
❌ 5× L2 overhead per timestep (vs 1× in Option 1)
❌ Compilation complexity increases significantly

**Performance Impact:**
- **Risk:** GPU hang at 100% (nested scan issue)
- **If it works:** Throughput 40-50k → 20-30k p/s (2× slower)
- Retention: 16% → 75-85% (5× improvement)

**Verdict:** ❌ **High risk** - nested scan problem from previous implementation.

---

### Option 3: Vectorized Global Search (New Implementation)

**Concept:** Implement L2 global search using pure vmap (no scan) to avoid nesting issues.

**Implementation:**
```python
@jax.jit
def search_global_vectorized(
    positions: jax.Array,      # (N, 3)
    search_mask: jax.Array,    # (N,) bool
    node_positions: jax.Array,
    connectivity: jax.Array
) -> jax.Array:
    """
    Vectorized global search: vmap over particles AND elements.

    Shape explosion: (N_particles, N_elements) boolean array
    Memory: 105k × 3.5M × 1 byte = 367 GB (WILL OOM!)
    """
    # ... this won't work due to memory
```

**Architecture:**
- Pure vmap (no scan)
- **Memory explosion:** (N_particles × N_elements) boolean array

**Verdict:** ❌ **Not viable** - 367 GB memory requirement for 100k particles.

---

### Option 4: Hybrid Approach - Global Search Only at Final Position (Optimized)

**Concept:** Option 1 but optimized to reduce transfer overhead.

**Implementation:**
```python
@jax.jit
def rk4_with_deferred_fallback(...):
    """RK4 with L0+L1, returns both positions and failure mask."""
    # ... PHASE3A RK4 ...
    return positions_final, element_ids_final, (element_ids_final < 0)

# Production loop
positions, element_ids, failed_mask = rk4_with_deferred_fallback(...)

# Apply L2 OUTSIDE, but using GPU-resident mask (no transfer)
element_ids = apply_global_fallback_gpu(positions, element_ids, failed_mask, mesh_gpu)
```

**Architecture:**
- RK4: PHASE3A + return failure mask
- L2: Separate JIT kernel using GPU-resident mask
- **Key:** Minimize CPU-GPU transfers by keeping mask on GPU

**Pros:**
✅ No nested scan (avoids GPU hang)
✅ Reduces transfer overhead (mask stays on GPU)
✅ Still recovers final-position failures
✅ Clean separation of concerns

**Cons:**
❌ Still doesn't fix k1-k4 failures (same as Option 1)
❌ Limited improvement (35-40% retention)

**Performance Impact:**
- Throughput: 40-50k → 37-47k p/s (5-8% slower)
- Retention: 16% → 35-40%

**Verdict:** ⚠️ **Better than Option 1**, but still partial solution.

---

### Option 5: Conditional Global Search (INSIDE RK4, No Nested Scan)

**Concept:** Use `jax.lax.cond` to conditionally trigger global search only when failures exceed threshold.

**Implementation:**
```python
@jax.jit
def rk4_fused_with_conditional_global(...):
    # k1 search
    element_ids_k1 = search_L0_L1(...)
    n_failed_k1 = jnp.sum(element_ids_k1 < 0)

    # Conditional global search (only if failures > 0)
    def apply_global_k1(_):
        failed_mask = element_ids_k1 < 0
        global_results = search_global_gpu_native_scan(...)
        return jnp.where(failed_mask, global_results, element_ids_k1)

    def skip_global_k1(_):
        return element_ids_k1

    element_ids_k1 = jax.lax.cond(
        n_failed_k1 > 0,
        apply_global_k1,
        skip_global_k1,
        None
    )
    # ... repeat for k2, k3, k4, final
```

**Architecture:**
- Single @jax.jit with conditional execution
- **Still has nested scan:** lax.cond → scan (nested within RK4 vmap)
- Conditions don't eliminate nesting, just add branching

**Verdict:** ❌ **Same nested scan problem** as Option 2.

---

### Option 6: Multi-Pass Strategy (OUTSIDE RK4)

**Concept:** Do multiple passes - first PHASE3A RK4 for all, then global search for failures, then re-do RK4 for recovered particles.

**Implementation:**
```python
# Pass 1: PHASE3A RK4 for all particles
positions1, element_ids1 = rk4_step_gpu_fused(...)

# Pass 2: Global search for failures
failed_mask = element_ids1 < 0
if jnp.any(failed_mask):
    element_ids_corrected = search_global_gpu_native_scan(
        positions_old, failed_mask, ...  # Search at OLD positions
    )

    # Pass 3: Re-do RK4 ONLY for recovered particles with correct starting elements
    positions_final, element_ids_final = rk4_step_gpu_fused(
        positions_old[recovered_mask],
        element_ids_corrected[recovered_mask],
        ...
    )
    # Merge results
```

**Pros:**
✅ No nested scan
✅ Recovers k1-k4 failures (by recomputing with correct starting elements)

**Cons:**
❌ **2-3× RK4 calls per timestep** for failed particles
❌ Complex bookkeeping (masking, merging)
❌ Throughput: 40-50k → 15-25k p/s (2-3× slower)

**Verdict:** ⚠️ **Comprehensive but expensive**.

---

### Option 7: Increase L1 Hops to 4 or 5

**Concept:** Increase neighbor search depth instead of adding L2.

**Configuration:**
```python
RK4_L1_HOP_COUNT = 4  # ~340 neighbors
# or
RK4_L1_HOP_COUNT = 5  # ~1,360 neighbors
```

**Pros:**
✅ No architectural changes
✅ No nested scan
✅ Pure vmap (PHASE3A)
✅ One-line change

**Cons:**
❌ 4-hop: ~340 neighbors → 4× slower L1 search
❌ 5-hop: ~1,360 neighbors → 16× slower L1 search
❌ May still miss in high-gradient regions
❌ Throughput: 40-50k → 10-20k p/s (4-hop) or 5-10k p/s (5-hop)

**Expected Retention:**
- 4-hop: 99.7% hit rate → ~47% retention (0.997^2500)
- 5-hop: 99.9% hit rate → ~82% retention (0.999^2500)

**Verdict:** ⚠️ **5-hop might work** but very slow. Worth testing.

---

## Summary Table

| Option | Retention | Throughput | Nested Scan Risk | Implementation |
|--------|-----------|------------|------------------|----------------|
| **1. L2 at Final (outside)** | ~35-40% | 35-45k p/s | ✅ None | Trivial (5 lines) |
| **2. L2 at All 5 (inside)** | 75-85% | 20-30k p/s | ❌ HIGH | Moderate |
| **3. Vectorized L2** | N/A | N/A | N/A | ❌ OOM (367 GB) |
| **4. Optimized L2 Final** | 35-40% | 37-47k p/s | ✅ None | Easy (15 lines) |
| **5. Conditional L2** | 75-85% | 20-30k p/s | ❌ HIGH | Moderate |
| **6. Multi-Pass** | 75-85% | 15-25k p/s | ✅ None | Complex |
| **7a. 4-hop L1** | ~47% | 10-20k p/s | ✅ None | Trivial (1 line) |
| **7b. 5-hop L1** | ~82% | 5-10k p/s | ✅ None | Trivial (1 line) |

---

## My Recommendations

### Recommendation A: Two-Stage Testing Approach

**Stage 1 - Quick Test (5 minutes):**
1. Try **Option 7b (5-hop L1)** first
   - Change `RK4_L1_HOP_COUNT = 5`
   - Run 100 timesteps and check retention
   - Expected: 82% retention, 5-10k p/s

**Stage 2 - If 5-hop too slow:**
2. Implement **Option 4 (Optimized L2 at Final)**
   - Keep 3-hop L1 inside RK4
   - Add GPU-native global fallback outside RK4
   - Expected: 35-40% retention, 37-47k p/s
   - No nested scan risk

### Recommendation B: Conservative Approach (Safest)

Implement **Option 4** directly:
- Proven architecture (no nested scan)
- Uses existing `search_global_gpu_native_scan`
- 2× improvement over baseline (16% → 35-40%)
- Maintains good throughput (37-47k p/s)
- Easy to implement and test

### Recommendation C: Aggressive Approach (Best Retention)

Try **Option 2** cautiously with monitoring:
- Best retention (75-85%)
- **Risk:** May cause GPU hang (nested scan)
- **Mitigation:** Test with 10 timesteps first, monitor GPU
- **Fallback:** If hangs, revert to Option 4

---

## Available Implementations

### Already Implemented (Can Use Immediately)

1. **`search_global_gpu_native_scan`** ([block_local_search.py:304](jaxtrace/gpu/search/block_local_search.py#L304))
   - GPU-native scan over particles
   - Works for Options 1, 2, 4, 5, 6
   - ⚠️ Causes nested scan if called inside RK4

2. **`create_search_gpu_fused(n_hops)`** ([rk4_gpu_fused.py:132](jaxtrace/gpu/tracking/rk4_gpu_fused.py#L132))
   - Current PHASE3A search (L0+L1)
   - Pure vmap, no scan
   - Can change `n_hops` for Option 7

### Need New Implementation

- **None** - all options can use existing code!

---

## Questions for You

1. **What retention target do you need?**
   - 35-40%: Option 4 (safe, fast)
   - 50-60%: Option 7a (4-hop, moderate speed)
   - 80%+: Option 7b (5-hop, slow) or Option 2 (risky)

2. **What throughput is acceptable?**
   - 35-47k p/s: Option 4
   - 20-30k p/s: Option 2 (if no hang)
   - 10-20k p/s: Option 7a
   - 5-10k p/s: Option 7b

3. **Risk tolerance for nested scan?**
   - Low risk: Options 1, 4, 7
   - High risk (for better retention): Options 2, 5

4. **Do you want to test 5-hop L1 first?**
   - It's literally a 1-line change
   - Can tell us if high hop count solves the problem
   - If 5-hop gives 80% retention, we don't need L2 at all

---

**My top suggestion:** Start with **Option 7b (5-hop L1)** to see if simple neighbor extension solves it. If too slow, implement **Option 4 (L2 at final, outside RK4)** as safe middle ground.

---

# GPU Memory Optimization Research Findings

## User's Constraints

You mentioned two critical GPU limitations:
1. **4-hop/5-hop L1 causes GPU OOM** - likely due to flattened neighbor arrays
2. **Block-wise search has memory issues** - especially heavy blocks with huge padded arrays

Let me address both with research-backed solutions.

---

## Issue 1: Multi-Hop Neighbor Search Memory Explosion

### Current Implementation Analysis

Your current `search_level1_multihop_vectorized` ([incremental_search_vectorized.py:235](jaxtrace/gpu/search/incremental_search_vectorized.py#L235)) uses **static concatenation**:

```python
# 3-hop: concatenate all levels
all_neighbors = jnp.concatenate([
    hop1,           # (4,)
    hop2_flat,      # (16,)
    hop3_flat       # (64,)
])  # Total: (84,) per particle

# 5-hop would be:
all_neighbors = jnp.concatenate([
    hop1,           # (4,)
    hop2_flat,      # (16,)
    hop3_flat,      # (64,)
    hop4_flat,      # (256,)
    hop5_flat       # (1024,)
])  # Total: (1,364,) per particle

# Then vmap over all particles:
# Memory: 105k particles × 1,364 neighbors × (checks + intermediate arrays)
# This creates MASSIVE materialization during JIT compilation
```

**Why this OOMs:**
- JAX materializes the full `(N_particles, N_neighbors)` array during compilation
- For 5-hop: 105k × 1,364 = 143M neighbor checks
- Each check involves 4-node tetrahedra → 572M node lookups
- Intermediate arrays during vmap compilation explode memory

### Solution 8: Chunked Multi-Hop Search (Memory-Efficient)

**Research Finding:** JAX 0.4.31+ supports `lax.map` with `batch_size` parameter for memory-limited batching [(Stack Overflow)](https://stackoverflow.com/questions/77527847/jax-vmap-limit-memory).

**Implementation:**
```python
@jax.jit
def search_level1_multihop_chunked(
    positions: jax.Array,
    cached_element_ids: jax.Array,
    element_neighbors: jax.Array,
    node_positions: jax.Array,
    connectivity: jax.Array,
    n_hops: int = 5,
    chunk_size: int = 1000  # Process 1k particles at a time
) -> jax.Array:
    """
    Memory-efficient multi-hop search using chunked processing.

    Uses lax.map with batch_size to avoid materializing all particles at once.
    """

    def check_one_particle_multihop(pos_and_cached):
        pos, cached_id = pos_and_cached
        # ... same multi-hop expansion logic ...
        # Build all_neighbors up to n_hops
        # Check all neighbors
        return found_element_id

    # Use lax.map with batch_size instead of vmap
    # This processes in chunks of `chunk_size` particles
    result = jax.lax.map(
        check_one_particle_multihop,
        (positions, cached_element_ids),
        batch_size=chunk_size  # ✅ Memory-limited batching
    )

    return result
```

**How it works:**
- `lax.map` with `batch_size=1000` performs a scan with `N // 1000` steps
- Each step vmaps over 1,000 particles (not all 105k)
- Memory: 1k × 1,364 = 1.36M checks (vs 143M for full vmap)
- **105× memory reduction!**

**Performance:**
- Slightly slower than pure vmap (scan overhead: ~5-10%)
- Still fully GPU-accelerated
- Enables 5-hop without OOM

**Source:** [JAX GitHub Discussion #11319](https://github.com/jax-ml/jax/issues/11319) - chunked vmap feature request

---

### Solution 9: Hierarchical Neighbor Search (Early-Exit per Hop)

**Research Finding:** "Scanning over reduction is significantly faster and more memory efficient than reducing over stacked output" [(JAX Issue #4968)](https://github.com/jax-ml/jax/issues/4968).

**Concept:** Instead of concatenating all hops and checking at once, check **incrementally** with early exit:

```python
@jax.jit
def search_level1_multihop_hierarchical(
    positions: jax.Array,
    cached_element_ids: jax.Array,
    element_neighbors: jax.Array,
    node_positions: jax.Array,
    connectivity: jax.Array,
    n_hops: int = 5
) -> jax.Array:
    """
    Hierarchical multi-hop search with early exit per hop level.

    Checks hop-by-hop, exits early if found. Avoids materializing
    all neighbors at once.
    """

    def check_one_particle_hierarchical(pos, cached_id):
        """Check particle with early exit per hop."""
        is_valid = (cached_id >= 0) & (cached_id < len(element_neighbors))
        safe_id = jnp.where(is_valid, cached_id, 0)

        # Hop 1: Check 4 neighbors
        hop1_neighbors = element_neighbors[safe_id]  # (4,)
        result = search_in_neighbors(pos, hop1_neighbors, connectivity, node_positions)

        # Early exit if found
        def continue_hop2(_):
            # Hop 2: Check 16 neighbors
            hop2_frontier = jax.vmap(lambda n: element_neighbors[jnp.where(n >= 0, n, 0)])(hop1_neighbors)
            hop2_flat = hop2_frontier.reshape(-1)  # (16,)
            result2 = search_in_neighbors(pos, hop2_flat, connectivity, node_positions)

            # Nested early exit for hop 3-5...
            def continue_hop3(_):
                # ... continue expanding
                return -1  # Placeholder

            return jax.lax.cond(result2 >= 0, lambda _: result2, continue_hop3, None)

        return jax.lax.cond(result >= 0, lambda _: result, continue_hop2, None)

    return jax.vmap(check_one_particle_hierarchical)(positions, cached_element_ids)
```

**Memory benefit:**
- Each hop level processes **independently** (not concatenated)
- Early exit reduces computation for most particles (found in hop 1-2)
- No massive `all_neighbors` array materialization

**Performance:**
- **Best case (found in hop 1-2):** 80-90% of particles, 4× faster than full 5-hop
- **Worst case (needs hop 5):** Same as full 5-hop
- **Average:** 40-60% faster than naive 5-hop

---

## Issue 2: Block-Wise Search Memory Explosion

### Current Implementation Problem

Your block-wise search uses **padded arrays** ([block_local_search.py](jaxtrace/gpu/search/block_local_search.py)):
- Heavy blocks: up to 450,004 elements
- Padded to fixed size: `(256 blocks, 450,004 elements)` = 6.6 GB
- **98% waste** for light blocks (most have < 10k elements)

### Solution 10: Ragged Array Emulation with Concatenated Storage

**Research Finding:** JAX recommends "concatenated storage format" for ragged data [(JAX Discussion #5184)](https://github.com/jax-ml/jax/discussions/5184).

**Implementation:**
```python
@dataclass
class CompactBlockLists:
    """
    Memory-efficient block element lists using concatenated storage.

    Instead of padded (256, 450004), uses:
    - data: (total_elements,) - all elements concatenated
    - start_indices: (256,) - where each block starts
    - lengths: (256,) - how many elements per block

    Memory: 3.5M × 4 bytes = 14 MB (vs 6.6 GB padded)
    """
    data: jax.Array              # (total_elements,) - concatenated element IDs
    start_indices: jax.Array     # (n_blocks,) - start index for each block
    lengths: jax.Array           # (n_blocks,) - number of elements per block
    max_block_size: int          # Maximum elements in any block

def build_compact_block_lists(block_assignments, n_blocks, n_elements):
    """Build concatenated storage format for block element lists."""
    # Count elements per block
    lengths = jnp.array([
        jnp.sum(block_assignments == b) for b in range(n_blocks)
    ])

    # Compute start indices (cumulative sum)
    start_indices = jnp.concatenate([jnp.array([0]), jnp.cumsum(lengths[:-1])])

    # Concatenate all element IDs
    data = []
    for b in range(n_blocks):
        block_elements = jnp.where(block_assignments == b)[0]
        data.append(block_elements)
    data = jnp.concatenate(data)

    return CompactBlockLists(
        data=data,
        start_indices=start_indices,
        lengths=lengths,
        max_block_size=int(jnp.max(lengths))
    )

@jax.jit
def search_in_block_compact(
    position: jax.Array,
    block_id: int,
    compact_lists: CompactBlockLists,
    node_positions: jax.Array,
    connectivity: jax.Array
) -> jax.Array:
    """
    Search in block using compact representation.

    Uses scan to iterate over variable-length block elements.
    """
    start = compact_lists.start_indices[block_id]
    length = compact_lists.lengths[block_id]

    # Extract this block's elements (variable length)
    # Pad to max_block_size for fixed-shape JIT
    block_elements = jax.lax.dynamic_slice(
        compact_lists.data,
        (start,),
        (compact_lists.max_block_size,)
    )

    # Mask for valid elements (first `length` are valid)
    valid_mask = jnp.arange(compact_lists.max_block_size) < length

    # Scan over elements with masking
    def check_element(carry, elem_and_valid):
        elem_id, is_valid = elem_and_valid

        def do_check(_):
            node_ids = connectivity[elem_id]
            tet_nodes = node_positions[node_ids]
            return jnp.where(point_in_tet_jax(position, tet_nodes), elem_id, -1)

        def skip_check(_):
            return -1

        result = jax.lax.cond(is_valid, do_check, skip_check, None)
        # Early exit: update carry if found
        new_carry = jnp.where(result >= 0, result, carry)
        return new_carry, result

    found_id, _ = jax.lax.scan(
        check_element,
        -1,  # Initial carry
        (block_elements, valid_mask)
    )

    return found_id
```

**Memory savings:**
- **Before:** 256 × 450,004 × 4 bytes = 6.6 GB (padded)
- **After:** 3.5M × 4 bytes + 256 × 8 bytes = 14 MB (concatenated)
- **Reduction: 470× smaller!**

**Performance:**
- Scan overhead: ~10-20% slower than padded vmap (if it fit in memory)
- But padded version OOMs, so this is the only viable option
- Light blocks: very fast (scan over 2-1000 elements)
- Heavy blocks: slower (scan over 450k elements), but necessary

**Source:** [JAX Ragged Array Discussion #5184](https://github.com/jax-ml/jax/discussions/5184)

---

### Solution 11: Spatial Hashing for Block Search (BVH/Octree Hybrid)

**Research Finding:** OLBVH (Octree Linear BVH) reduces memory by 75% and achieves 8-13× speedup for tetrahedral mesh search [(Springer Article)](https://link.springer.com/article/10.1007/s00371-020-01886-6).

**Concept:** Instead of storing all elements per block, use **spatial hash + BVH** for O(log n) search:

```python
@dataclass
class SpatialHashBVH:
    """
    Hybrid spatial hash + BVH for efficient block-local search.

    - Spatial hash: O(1) lookup for coarse block
    - BVH within block: O(log n) search in heavy blocks
    - Morton codes: GPU-friendly linear layout
    """
    # Spatial hash (uniform grid)
    grid_resolution: Tuple[int, int, int]  # e.g., (8, 8, 4)
    cell_size: jax.Array                    # (3,) - cell dimensions
    grid_min: jax.Array                     # (3,) - grid origin

    # BVH for heavy blocks (> 10k elements)
    bvh_nodes: jax.Array           # (n_nodes, 6) - AABB boxes (min_xyz, max_xyz)
    bvh_left_child: jax.Array      # (n_nodes,) - left child index
    bvh_right_child: jax.Array     # (n_nodes,) - right child index
    bvh_element_id: jax.Array      # (n_nodes,) - element ID (leaf nodes only)

    # Block → BVH mapping
    block_to_bvh_root: jax.Array   # (n_blocks,) - root node for each block's BVH

def build_spatial_hash_bvh(block_grid, element_bboxes, block_assignments):
    """
    Build hybrid spatial hash + BVH structure.

    Light blocks: direct element list
    Heavy blocks: BVH tree for O(log n) search
    """
    # ... implementation using Morton codes ...
    pass

@jax.jit
def search_with_bvh(
    position: jax.Array,
    block_id: int,
    spatial_bvh: SpatialHashBVH,
    connectivity: jax.Array,
    node_positions: jax.Array
) -> jax.Array:
    """
    BVH-accelerated search within block.

    Uses recursive tree traversal (scan-based for JIT).
    """
    bvh_root = spatial_bvh.block_to_bvh_root[block_id]

    # BVH traversal using scan (stack-based, GPU-friendly)
    def traverse_bvh(carry, _):
        stack, found_id = carry

        # Pop from stack
        node_id = stack[0]
        stack = stack[1:]

        # Check if leaf node
        is_leaf = spatial_bvh.bvh_left_child[node_id] < 0

        def check_leaf(_):
            elem_id = spatial_bvh.bvh_element_id[node_id]
            # Point-in-tet test
            # ...
            return elem_id if inside else -1

        def traverse_children(_):
            # Check AABB intersection, push children to stack
            # ...
            return -1

        result = jax.lax.cond(is_leaf, check_leaf, traverse_children, None)

        return (stack, jnp.where(result >= 0, result, found_id)), result

    # Scan over max_depth iterations
    initial_stack = jnp.array([bvh_root] + [-1] * 63)  # Stack size = tree depth
    (_, found_id), _ = jax.lax.scan(
        traverse_bvh,
        (initial_stack, -1),
        None,
        length=64  # Max BVH depth
    )

    return found_id
```

**Performance (from OLBVH paper):**
- **Light blocks (< 10k elements):** Direct search (current method)
- **Heavy blocks (450k elements):** BVH reduces from 450k checks → ~20-30 (log₂ 450k ≈ 19)
- **Speedup:** 8-13× for heavy block search
- **Memory:** 75% reduction vs padded arrays

**Implementation complexity:** High (requires BVH builder), but worthwhile for heavy blocks.

**Source:**
- [OLBVH Paper (Springer)](https://link.springer.com/article/10.1007/s00371-020-01886-6)
- [GPU BVH Thesis](https://www.ks.uiuc.edu/Research/vmd/projects/ece498/raytracing/GPU_BVHthesis.pdf)

---

### Solution 12: GPU Ray Tracing Cores for Particle Tracking

**Research Finding:** Novel GPU-accelerated particle tracking using **hardware ray tracing cores** (RTX GPUs) with BVH achieves significant speedup on tetrahedral meshes [(ScienceDirect 2021)](https://www.sciencedirect.com/science/article/abs/pii/S0010465521003337).

**Key insight:** Modern RTX GPUs have dedicated ray-triangle intersection hardware that can be repurposed for point-in-tetrahedron tests.

**Concept:**
- Decompose tetrahedra into 4 triangular faces
- Use RTX cores for ultra-fast ray-triangle intersection
- Point-in-tet = inside all 4 faces

**Implementation:** Requires CUDA/OptiX backend (not pure JAX), but could be wrapped as JAX custom primitive.

**Performance (from paper):**
- Tested on 819,726 tetrahedral mesh
- Speedup: 10-50× vs CPU neighbor search
- Works with neighbor search method (same as your approach)

**Limitation:** Requires RTX GPU (2000 series+), needs custom CUDA kernel.

**Source:** [GPU Ray Tracing for Particle Tracking](https://www.researchgate.net/publication/356157371_An_GPU-accelerated_particle_tracking_method_for_Eulerian-Lagrangian_simulations_using_hardware_ray_tracing_cores)

---

## Recommended Implementation Strategy

Based on your constraints and research findings:

### Phase 1: Enable 5-Hop L1 (Memory-Efficient)

**Implement Solution 8 (Chunked Multi-Hop):**
1. Replace `jax.vmap` with `jax.lax.map(batch_size=1000)` in `search_level1_multihop_vectorized`
2. Test with `RK4_L1_HOP_COUNT = 5`
3. Expected: 82% retention, 8-12k p/s (chunking overhead)

**Effort:** Low (10-20 lines of code)
**Risk:** None (proven JAX feature)
**Memory:** 105× reduction (should fit on your GPU)

### Phase 2: Optimize Block Search (If Needed)

**Implement Solution 10 (Compact Block Lists):**
1. Replace padded arrays with concatenated storage
2. Use scan with masking for variable-length blocks
3. Expected: 470× memory reduction (6.6 GB → 14 MB)

**Effort:** Moderate (50-100 lines)
**Risk:** Low (standard JAX pattern)
**Performance:** 10-20% slower than padded (if it fit), but enables block search

### Phase 3: Advanced Optimization (Future)

**Implement Solution 11 (Spatial Hash + BVH) for heavy blocks:**
1. Build BVH for blocks > 10k elements
2. Keep direct search for light blocks
3. Expected: 8-13× speedup on heavy blocks

**Effort:** High (200-300 lines + BVH builder)
**Risk:** Moderate (complex tree traversal)
**Payoff:** Enables efficient block-local search at all scales

---

## Sources

### JAX Memory Optimization
- [JAX vmap memory limiting](https://stackoverflow.com/questions/77527847/jax-vmap-limit-memory)
- [Chunked vmap feature request](https://github.com/jax-ml/jax/issues/11319)
- [Scan memory optimization](https://github.com/jax-ml/jax/issues/4968)
- [Ragged array workarounds](https://github.com/jax-ml/jax/discussions/5184)
- [JAX GPU performance tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html)

### Spatial Data Structures
- [OLBVH for volumetric meshes](https://link.springer.com/article/10.1007/s00371-020-01886-6)
- [BVH vs Octree comparison](https://computergraphics.stackexchange.com/questions/10098/is-bvh-faster-than-the-octree-kd-tree-for-raytracing-the-objects-on-a-gpu)
- [GPU particle tracking with ray tracing](https://www.sciencedirect.com/science/article/abs/pii/S0010465521003337)
- [Spatial hashing for GPUs](https://github.com/kodai100/Unity_GPUNearestNeighbor)

### JAX-MD (Neighbor Lists)
- [JAX-MD repository](https://github.com/jax-md/jax-md) - Dense, Sparse, OrderedSparse neighbor formats
- [Locally perfect spatial hashing](https://haug.codes/blog/locally-perfect-hashing/)

---

## Updated Recommendations

Given your GPU memory constraints, I now recommend:

**Primary recommendation:** **Solution 8 (Chunked 5-hop L1)**
- Enables 5-hop without OOM
- 82% retention expected
- Throughput: 8-12k p/s (acceptable for your use case)
- **Effort: 1 hour implementation**

**Secondary (if 5-hop still too slow):** **Solution 9 (Hierarchical early-exit)**
- Combines 5-hop capability with early exit optimization
- 40-60% faster than naive 5-hop on average
- **Effort: 2-3 hours implementation**

**Block search (if you need L2):** **Solution 10 (Compact representation)**
- Solves padding memory waste
- Enables block-local L2 fallback
- **Effort: 3-4 hours implementation**

Would you like me to implement Solution 8 (chunked 5-hop) first to test if it resolves the particle loss issue?
