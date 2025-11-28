# Implementation Comparison: V1 vs V2 vs Hybrid Vectorization

**Date**: 2025-11-16
**Context**: Performance bottleneck analysis for initial assignment and multi-level search
**Goal**: Determine optimal vectorization strategy for 1,000-5,000 p/s throughput on 4GB GPU

---

## Executive Summary

| Approach | Throughput | Memory | Status | Recommendation |
|----------|-----------|--------|--------|----------------|
| **V1 (Python Loop)** | 7-8 p/s initial<br>258-367 p/s multi-level | ~0 MB | ✅ Works | ❌ Too slow (150-700× below target) |
| **V2 (Full JAX vmap)** | Expected 5K-13K p/s | 9.8 GB | ❌ OOM crash | ❌ Exceeds 4GB GPU limit |
| **Hybrid (Block-grouped vmap)** | Expected 1K-5K p/s | ~400-800 MB | ⚙️ Testing | ✅ **RECOMMENDED** |

---

## Three Approaches Explained

### V1: Python Loop (Current Implementation)

**Location**:
- `jaxtrace/gpu/search/multi_level_search.py` (lines 188-299)
- `jaxtrace/gpu/search/initial_assignment.py` (lines 319-517)

**Architecture**:
```python
# CPU-side Python loop over particles
for i in range(n_particles):
    pos = positions_jax[i]
    elem_id = search_level0_cached(pos, ...)  # JIT-compiled kernel
    # ... more searches
```

**Performance**:
- **Initial assignment**: 7-8 p/s (50,000 particles = 2+ hours)
- **Multi-level search**: 258-367 p/s (benefits from 87% L0 cache hits)
- **Bottleneck**: Python loop overhead + JIT recompilation per iteration

**Why So Slow**:
1. Each iteration calls a JIT-compiled function from Python
2. Data transfer overhead (CPU ↔ GPU) for each particle
3. No parallelization across particles
4. Expected: 150-700× slower than vectorized version

**Memory Usage**: Minimal (~0 MB), only single particle processed at a time

**Verdict**: ❌ **Too slow** - Unacceptable for production use


---

### V2: Full JAX vmap Vectorization

**Location**:
- `OLD/search_v1_v2/multi_level_search_v2.py`
- `OLD/search_v1_v2/initial_assignment_v2.py`

**Architecture**:
```python
# Vectorize over ALL particles in single GPU kernel
search_all = jax.vmap(
    lambda pos, cached_elem, cached_block: search_single_particle_masked(
        pos, cached_elem, cached_block,
        node_positions,      # (895K, 3) - shared
        connectivity,        # (3.5M, 4) - shared
        padded_elements_all, # (256, 444K) - HUGE!
        ...
    )
)
element_ids = search_all(positions, cached_elems, cached_blocks)
```

**Key Innovation**: **Masked Execution Pattern**
- Execute ALL search levels (L0, L1, L2, L3) unconditionally
- Select first valid result using `jnp.where` masks
- Avoids `lax.cond` which causes memory explosion in vmap

**Expected Performance**:
- Target: 5,000-13,000 p/s (25-75× speedup over V1)
- Based on eliminating Python loop overhead

**Critical Failure: OOM Crash**
- Memory allocation: **9.8 GB** on 4GB GPU
- Root cause: Full padded arrays `(256, 444,040)` replicated for EACH particle in vmap
- Example: 50,000 particles × 433 MB = 21+ GB

**Why V2 Failed**:
```python
# Each particle in vmap gets its own copy of padded arrays:
jax.vmap(lambda pos: search(pos, padded_elements_all))
# Memory: n_particles × array_size
#       = 50,000 × 433 MB = 21 GB (!!!)
```

**Verdict**: ❌ **OOM crash** - Fundamentally incompatible with 4GB GPU limit


---

### Hybrid: Block-Grouped Vectorization (Recommended)

**Location**:
- `jaxtrace/gpu/search/initial_assignment.py` (lines 319-517, NEW implementation)

**Architecture**:
```python
# STEP 1: Vectorized block finding (ALL particles)
find_blocks_vmap = jax.vmap(
    lambda pos: find_containing_block_jax(pos, domain_bounds, grid_size)
)
particle_block_ids = find_blocks_vmap(positions)  # O(n_particles) GPU kernel

# STEP 2: Group particles by block (CPU)
particles_per_block = {}
for i, block_id in enumerate(particle_block_ids):
    if block_id >= 0:
        particles_per_block.setdefault(block_id, []).append(i)

# STEP 3: Vectorized search WITHIN each block
for block_id, particle_indices in particles_per_block.items():
    particle_batch = positions[particle_indices]  # Subset of particles

    # Vectorize hash bucket search for this block
    if is_heavy[block_id]:
        search_hash_vmap = jax.vmap(
            lambda pos: search_level2b_hash_bucket(
                pos, block_id,
                bucket_elements[block_id],    # SINGLE block's data
                bucket_counts[block_id],
                ...
            )
        )
        found_elem_ids = search_hash_vmap(particle_batch)
    else:
        # Vectorized light block search
        search_light_vmap = jax.vmap(
            lambda pos: search_level2a_light_block(
                pos, block_id,
                padded_elements[block_id],    # SINGLE block's data
                padded_counts[block_id],
                ...
            )
        )
        found_elem_ids = search_light_vmap(particle_batch)
```

**Key Advantages**:

1. **Memory Efficient**
   - Vmap over particles in SAME block only
   - Each vmap uses single block's data (~1.7 MB for heavy block)
   - Total memory: n_particles_in_block × 1.7 MB << 4 GB

2. **Preserves Parallelism**
   - Step 1: Fully vectorized block finding (GPU)
   - Step 3: Fully vectorized search within blocks (GPU)
   - Only Step 2 (grouping) is sequential CPU code

3. **Scales with Block Granularity**
   - ThreadedA: 256 blocks, avg 13,615 elem/block
   - Worst case: Heavy block 227 with 444K elements
   - Memory per vmap: ~1.7 MB (manageable)

**Expected Performance**:
- **Initial assignment**: 1,000-5,000 p/s (130-625× faster than V1)
- **Multi-level search**: Similar or better (with cache hits)
- Based on: Vectorized GPU kernels + minimal CPU overhead

**Memory Estimate**:
```
Block finding:     50,000 particles × 12 bytes = 0.6 MB
Particle grouping: 50,000 × 8 bytes (indices) = 0.4 MB
Per-block vmap:    ~200 particles × 1.7 MB = 340 MB (peak)
Total:             ~400-800 MB (well within 4GB)
```

**Limitations**:
- Requires Python loop over blocks (but only 256 iterations, not 50K)
- Less efficient than V2's full vmap IF we had unlimited memory
- Still has some CPU-GPU synchronization overhead between blocks

**Verdict**: ✅ **RECOMMENDED** - Best balance of performance and memory


---

## Detailed Performance Analysis

### Bottleneck Comparison

| Metric | V1 (Loop) | V2 (Full vmap) | Hybrid (Block vmap) |
|--------|-----------|----------------|---------------------|
| Python loop iterations | 50,000 | 0 | 256 |
| JIT recompilations | 50,000 | 1 | 256 |
| GPU kernel launches | 50,000+ | 1 | ~300 |
| Data transfers (CPU↔GPU) | 50,000 | 1 | ~300 |
| Memory allocation | 0.01 MB | 9.8 GB | 400-800 MB |

### Throughput Projections

**Initial Assignment (50,000 particles)**:

| Approach | Time | Throughput | Speedup |
|----------|------|-----------|---------|
| V1 (actual) | 2+ hours | 7-8 p/s | 1× baseline |
| V2 (expected) | N/A (OOM) | N/A | N/A |
| Hybrid (expected) | 10-50 s | 1,000-5,000 p/s | **130-625×** |

**Multi-Level Search (50,000 particles, 87% L0 cache)**:

| Approach | Time | Throughput | Speedup |
|----------|------|-----------|---------|
| V1 (actual) | 14.6 s | 3,428 p/s | 1× baseline |
| V2 (expected) | N/A (OOM) | N/A | N/A |
| Hybrid (expected) | 3-7 s | 7,000-17,000 p/s | **2-5×** |

*Note: Multi-level benefits less because 87% of searches already hit L0 cache (single element test)*


---

## Memory Breakdown: Why V2 Failed

### V2 Full Vmap Memory Explosion

```python
# V2 approach: vmap over ALL particles
search_all = jax.vmap(lambda pos: search(..., padded_elements_all))

# JAX replicates padded_elements_all for EACH particle:
# Memory = n_particles × array_size
```

**ThreadedA mesh memory**:
- Padded arrays: `(256, 444,040)` int32 = **433 MB**
- Node positions: `(895,972, 3)` float32 = 10.2 MB
- Connectivity: `(3,485,406, 4)` int32 = 53.1 MB
- **Total per particle**: ~500 MB

**For 50,000 particles**:
- Expected allocation: 50,000 × 500 MB = **25,000 GB** (!!)
- Actual OOM at: ~9.8 GB (JAX attempted optimization, still failed)

### Hybrid Block-Grouped Memory Efficiency

```python
# Hybrid approach: vmap over particles IN SAME BLOCK
for block_id in range(n_blocks):
    search_block = jax.vmap(
        lambda pos: search(..., padded_elements[block_id])  # Single row
    )
```

**Memory per block vmap**:
- Single block data: `(1, 444,040)` int32 = 1.7 MB (worst case)
- Node positions: 10.2 MB (shared, not replicated)
- Connectivity: 53.1 MB (shared, not replicated)
- **Total per particle in vmap**: ~1.7 MB

**For 50,000 particles** (distributed across 256 blocks):
- Avg particles/block: 195
- Peak vmap memory: 195 × 1.7 MB = **331 MB** (per block)
- Total with overhead: ~400-800 MB ✅


---

## Implementation Details: Hybrid Approach

### Algorithm Flow

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Vectorized Block Finding (GPU)                     │
│                                                              │
│ Input:  particle_positions (n_particles, 3)                │
│ Kernel: find_containing_block_jax (JAX JIT)                │
│ Method: jax.vmap over ALL particles                         │
│ Output: particle_block_ids (n_particles,)                  │
│ Time:   O(n_particles), ~0.1-0.5 s for 50K particles       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Group Particles by Block (CPU)                     │
│                                                              │
│ Method: Python dict comprehension                           │
│ Output: particles_per_block = {                             │
│           block_id: [particle_idx, ...],                    │
│           ...                                                │
│         }                                                    │
│ Time:   O(n_particles), ~0.01 s for 50K particles          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Vectorized L2 Search per Block (GPU)               │
│                                                              │
│ For each block_id:                                          │
│   Fetch: particle_batch = positions[particle_indices]      │
│   Heavy blocks:                                              │
│     search_hash_vmap = jax.vmap(search_level2b_hash_bucket)│
│   Light blocks:                                              │
│     search_light_vmap = jax.vmap(search_level2a_light_block)│
│   Execute: found_elem_ids = search_vmap(particle_batch)    │
│ Time:   O(n_blocks × avg_particles_per_block)              │
│         ~5-30 s for 50K particles                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: L3 Fallback for Unfound Particles (CPU loop)       │
│                                                              │
│ For each unfound particle:                                  │
│   Search neighbor blocks sequentially                       │
│ Expected: <1% of particles need this                        │
│ Time:   ~0.1-1 s                                            │
└─────────────────────────────────────────────────────────────┘
```

### Code Structure

**New file**: `jaxtrace/gpu/search/initial_assignment.py` (MODIFIED)

**Key functions**:
```python
def initial_search_batch_vectorized(
    particle_positions: np.ndarray,  # (n_particles, 3)
    domain_bounds: np.ndarray,
    grid_size: Tuple[int, int, int],
    classification: BlockClassification,
    padded_arrays: PaddedArrays,
    hash_bucket_data: Dict[int, HashBucketArrays],
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, InitialSearchStats]:
    """
    Vectorized initial particle assignment using block-grouped vmap.

    Returns:
        element_ids: (n_particles,) int32, -1 if not found
        block_ids: (n_particles,) int32, -1 if not found
        stats: Performance statistics
    """
```


---

## Hash Bucket Architecture Integration

### Why Hash Buckets Are Critical

**From `docs/FINAL_EXECUTABLE_PLAN.md` (lines 23-43)**:

> **Problem**: Some blocks may have 200K-1M elements even after 4×4×2 partitioning due to AMR clustering.
>
> **Solution**: Add **intra-block hash/bucket subdivision** in Phase 4.
>
> **Threshold**: If any block has >10,000 elements, subdivide with Morton/spatial hash.
>
> **Impact**:
> - Without hash: O(200K) search per particle in heavy blocks
> - With hash: O(4K) search per particle (50 buckets × 4K elem/bucket)
> - **50× speedup for heavy blocks**

### ThreadedA Heavy Blocks

**From test results**:
- Total blocks: 256
- Heavy blocks (>10K elements): 16
- Heaviest block: Block 227 with **444,040 elements**

**Search cost without hash buckets**:
- Linear scan: 444,040 element tests per particle
- At 50,000 particles: 22.2 billion element tests (!)

**Search cost WITH hash buckets** (8×8×8 = 512 buckets):
- Morton code lookup: O(1)
- Bucket scan: ~900 elements
- Neighbor bucket fallback: ~5,400 elements
- **Reduction**: 444,040 → ~900 (**493× improvement**)

### Hybrid Implementation Uses Hash Buckets

```python
# Heavy block search (vectorized over particles in block)
if is_heavy[block_id]:
    hash_arrays = hash_bucket_data[block_id]

    search_hash_vmap = jax.vmap(
        lambda pos: search_level2b_hash_bucket(
            pos,
            block_id,
            hash_arrays.bucket_elements,      # (512, ~900) elements
            hash_arrays.bucket_elem_counts,
            hash_arrays.bucket_neighbors_6,
            hash_arrays.n_buckets,
            hash_arrays.morton_bits,
            hash_arrays.block_bounds,
            node_positions,
            connectivity
        )
    )

    found_elem_ids = search_hash_vmap(particle_batch)
```

**Memory per particle in vmap**:
- Bucket arrays: `(512, 900)` int32 = 1.8 MB
- Much smaller than full block: `(1, 444K)` = 1.7 MB
- Enables efficient vectorization


---

## Recommendations

### Short-Term (Immediate)

1. ✅ **Deploy Hybrid vectorized initial assignment**
   - Replace Python loop in `initial_search_batch()` with block-grouped vmap
   - Expected: 130-625× speedup (7-8 p/s → 1,000-5,000 p/s)
   - Test file: `test_vectorized_initial_SIMPLE.py` (running now)

2. ✅ **Validate performance on ThreadedA 1K particles**
   - Target: >1,000 p/s throughput
   - Memory: <800 MB GPU allocation
   - Success criteria: Completes in <10 seconds

3. ⚙️ **Consider vectorizing multi-level search**
   - Same block-grouped approach
   - Lower priority (already 3,428 p/s due to L0 cache hits)
   - Expected improvement: 2-5× (cache-limited)

### Medium-Term (Next Week)

4. **Optimize block grouping step**
   - Current: Python dict/list comprehension (CPU)
   - Potential: JAX scatter/gather operations (GPU)
   - Expected gain: 10-50% (grouping is only ~0.01 s)

5. **Batch multiple blocks together**
   - Current: Loop over 256 blocks sequentially
   - Potential: Batch 4-8 blocks together in nested vmap
   - Trade-off: Memory vs parallelism
   - Expected gain: 20-40%

6. **Profile GPU kernel performance**
   - Use `jax.profiler` to identify bottlenecks
   - Optimize hash bucket search kernel
   - Tune bucket size (currently 200 target)

### Long-Term (Phase 2+)

7. **Full batched block-wise architecture** (Phase 2)
   - Implement `block_search.py` GPU kernels
   - Batch process entire blocks on GPU
   - See `jaxtrace/gpu/search/__init__.py` (lines 56-64)

8. **Multi-GPU support** (Phase 3+)
   - Distribute blocks across GPUs
   - Requires ghost region synchronization
   - Target: 100K+ particles across multiple GPUs


---

## Testing Strategy

### Unit Tests

✅ **Completed**:
- `test_phase1_batched_threadeda.py` - V1 baseline (3,428 p/s)
- Hash bucket building (16 heavy blocks, 1.29 s)
- Multi-level search validation (87% L0, 7% L1 hits)

⚙️ **In Progress**:
- `test_vectorized_initial_SIMPLE.py` - Hybrid vectorized initial assignment

📋 **Needed**:
- Benchmark hybrid vs V1 on 1K, 10K, 50K, 100K particles
- Memory profiling (GPU allocation tracking)
- Throughput scaling analysis

### Integration Tests

📋 **Needed**:
- End-to-end particle tracking (initial + multi-level + perturbation)
- Time-stepping with vectorized search
- Accuracy validation vs CPU tracker (<1% error)


---

## Conclusion

The **Hybrid Block-Grouped Vectorization** approach is the optimal solution given:

1. ✅ **Performance**: Expected 1,000-5,000 p/s (meets target)
2. ✅ **Memory**: 400-800 MB (well within 4GB GPU limit)
3. ✅ **Compatibility**: Works with existing hash bucket architecture
4. ✅ **Scalability**: Can optimize further with batching/GPU profiling

V1 (Python loop) is too slow, V2 (full vmap) exceeds memory limits. Hybrid is the **Goldilocks solution**.

---

**Next Steps**:
1. Wait for `test_vectorized_initial_SIMPLE.py` to complete
2. Analyze performance results
3. Deploy to multi-level search if successful
4. Proceed to Phase 2 batched block-wise architecture
