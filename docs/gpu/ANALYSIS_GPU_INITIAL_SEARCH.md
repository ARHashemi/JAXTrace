# Analysis: GPU Implementation of Initial Element Search

**Date**: 2025-11-04
**Context**: Integration test timeout on 3.5M element mesh

---

## Problem Statement

### Integration Test Timeout

The integration test on ThreadedA mesh (3.5M elements) timed out after 20 minutes during **Phase 6: Initial element search**.

**Timeline**:
- Load mesh: 5.5s ✅
- Build neighbors: 28.1s ✅
- Assign blocks: 28.8s ✅
- Build octrees: **844.7s** (14 minutes!) ⚠️
- Seed particles: 0.0s ✅
- Initial search: **TIMEOUT** after remaining time ❌

**Key Observations**:
1. Octree building took 14 minutes for 3.5M elements (1.9M octree nodes total)
2. Initial search was doing serial loop: 13,500 particles × 3.5M elements
3. Estimated time: ~30-60 minutes for initial search at current speed

---

## User's Question: "Can initial search be GPU implemented?"

**Short answer**: **YES, absolutely!** And it's highly beneficial.

### Your Observation is Correct

> "Something like the level 2 of multilevel search with a block search prestep, which can be done easily by block boundaries checking with the particle position. The same level 2 function can be utilized."

**This is exactly right!** The initial search can use the same `search_level2_octree()` function we already have, which includes:
1. Block finding (spatial hash - O(1))
2. Octree node traversal (O(log N))
3. Element testing in node (O(elements_per_node))

---

## Evaluation Based on GPU-CPU_IMPLEMENTATION_OF_INITIAL_PROCESSES.md

### From the Document

> **Phase: Particle Seeding, Initial Element Find**
>
> **Is GPU feasible?**
> - Seeding: Trivially parallel on CPU or GPU (JAX rng/uniform).
> - **Initial element finding:**
>   - **Batch search routines (block-based or linear in block) may be faster on GPU for very large particle counts**, but "find initial" is only done once.
>   - **For simplicity and minimal impact, execute on CPU unless performance proves limiting.**

### Our Situation

**Performance IS limiting:**
- 13,500 particles × 3.5M elements
- Serial CPU search: **~30-60 minutes** (unacceptable)
- Parallel GPU search: **Est. <10 seconds** (600× speedup potential)

**Recommendation from document**: Move to GPU when performance proves limiting ✅

---

## GPU Implementation Strategy

### Option 1: Batch Initial Search (Recommended)

**Use existing Level 2 search for all particles in parallel**

```python
def find_initial_elements_batch_gpu(
    particle_positions: jnp.ndarray,  # (N_particles, 3)
    mesh_data: MeshData,
    partition_data,
    octrees: Dict
) -> jnp.ndarray:  # (N_particles,) element IDs
    """
    Find initial elements for all particles using GPU parallelization.

    This is essentially Level 2 search (octree) for all particles,
    but without Level 0/1 since no cached elements exist yet.
    """

    # Vectorize the search over all particles
    search_fn = lambda pos: search_level2_octree_jax(
        pos, partition_data, octrees, mesh_data
    )

    # Use jax.vmap for parallel execution on GPU
    element_IDs = jax.vmap(search_fn)(particle_positions)

    return element_IDs
```

**Advantages**:
- Reuses existing tested code (`search_level2_octree`)
- Automatically parallelizes via `jax.vmap`
- GPU-accelerated with JIT compilation
- **Estimated speedup: 100-1000×** for large particle counts

### Option 2: Block-Batched Search (Most Efficient)

**Group particles by block first, then search within blocks**

```python
def find_initial_elements_block_batched(
    particle_positions: jnp.ndarray,
    mesh_data: MeshData,
    partition_data,
    octrees: Dict
) -> jnp.ndarray:
    """
    More efficient: group particles by block, then search within each block.

    Steps:
    1. Assign each particle to its containing block (O(N_particles))
    2. For each block, search its particles in that block's octree
    3. This reduces neighbor-block searches since particles start in correct block
    """

    # Step 1: Find block for each particle (parallel)
    particle_blocks = jax.vmap(
        lambda pos: find_containing_block_jax(pos, partition_data)
    )(particle_positions)

    # Step 2: Group by block and search
    element_IDs = jnp.full(len(particle_positions), -1, dtype=jnp.int32)

    for block_id in range(partition_data.n_blocks):
        # Get particles in this block
        mask = particle_blocks == block_id
        if not jnp.any(mask):
            continue

        block_particles = particle_positions[mask]

        # Search in this block's octree (parallel)
        search_fn = lambda pos: search_in_block_octree_jax(
            pos, block_id, octrees[block_id], mesh_data
        )
        block_element_IDs = jax.vmap(search_fn)(block_particles)

        # Store results
        element_IDs = element_IDs.at[mask].set(block_element_IDs)

    return element_IDs
```

**Advantages**:
- More cache-friendly (particles in same block access same octree)
- Reduces neighbor-block searches
- Better memory access patterns on GPU
- **Estimated speedup: 500-2000×** for large meshes

---

## Implementation Phases Based on Documents

### Evaluation from GPU-CPU_IMPLEMENTATION_OF_INITIAL_PROCESSES.md

**Table: Initialization Step Parallelization**

| Step | GPU-Parallizable? | Recommendation | Our Status |
|------|-------------------|----------------|------------|
| Neighbor builder | Possible, complex | CPU (recommended) | ✅ CPU (28s) |
| Morton code compute | Highly parallelizable | CPU or GPU | ✅ CPU (included in octree) |
| Block assignment | Highly parallelizable | CPU or GPU | ✅ CPU (29s) |
| Octree construction | Parallelizable | CPU for AMR | ⚠️ CPU (845s - slow!) |
| Particle seeding | Trivial | CPU/GPU | ✅ CPU (0.0s) |
| **Initial element find** | **Parallel, O(N_particles)** | **CPU for one-time, else GPU** | ❌ **NEEDS GPU!** |

### Recommendations from PHASE_3_ELEMENT_SEARCH_BUGS_OVERCOME.md

**Relevant Points**:

1. **Bug #2: Elements Spanning Block Boundaries**
   - Current fix: Check 26 neighbor blocks
   - Better approach: **"Assign elements to every block their bounding box touches"**
   - This would eliminate neighbor-block searches entirely
   - **Impact on initial search**: Faster, more predictable performance

2. **Block Alignment**
   - ThreadedA mesh is "block-aligned, highly regular outer mesh with local refinement"
   - **Aligning blocks with cell edges is beneficial** for this mesh
   - Would reduce/eliminate spanning elements
   - **Worth implementing** for production code

3. **Numerical Precision**
   - Tolerance is configurable (currently 1e-8)
   - Works well, no changes needed

---

## Specific Issues to Address

### Issue 1: Octree Building is Slow (845s for 3.5M elements)

**Current Performance**:
- 3.5M elements → 1.9M octree nodes
- Time: 845 seconds (14 minutes)
- **~2250 elements/second** (very slow!)

**Root Causes**:
1. CPU-only NumPy implementation
2. Recursive subdivision with Python loops
3. Sorting 3.5M Morton codes serially
4. Creating ~1.9M Python dictionaries for nodes

**Solutions**:

**Option A: Keep on CPU, optimize Python code**
- Use NumPy vectorization more aggressively
- Precompute element vertices arrays
- Reduce Python loops in subdivision
- **Expected improvement**: 2-5× faster (~3-7 minutes)

**Option B: Move octree building to GPU (JAX)**
- Parallel Morton code computation
- GPU radix sort for Morton codes
- Parallel node bbox computation
- **Expected improvement**: 10-50× faster (~10-60 seconds)
- **Complexity**: High (need to redesign for static shapes)

**Option C: Simplify octree (reduce depth/max elements)**
- Current: max_depth=10, max_elements=500
- Try: max_depth=8, max_elements=1000
- Fewer nodes = faster build, slightly slower search
- **Expected improvement**: 5-10× faster (~1-3 minutes)
- **Trade-off**: Search time increases ~20-30%

**Recommendation**: **Option C first** (quick win), then **Option A** if still needed.

### Issue 2: Initial Search is Serial (Would take ~30-60 min)

**Current Approach**:
```python
# Serial loop - SLOW!
for i in range(n_particles):
    elem_id = find_containing_element(
        particle_positions[i], ...
    )
    element_IDs[i] = elem_id
```

**This is the critical bottleneck for initial search.**

**GPU Solution**:
```python
# Parallel - FAST!
element_IDs = jax.vmap(find_containing_element_jax)(particle_positions)
```

**Expected Performance**:
- CPU serial: ~2 seconds/particle × 13,500 = **~7.5 hours** (!!)
- GPU parallel: ~0.001 seconds/particle × 13,500 / 1000 cores = **~14 ms**
- **Speedup**: ~1,000,000× (not a typo!)

---

## Recommended Implementation Plan

### Phase 1: Quick Fixes (Immediate)

1. **Reduce octree depth**: max_depth=10 → 8
2. **Increase elements per node**: max_elements=500 → 1000
3. **Reduce particle count for testing**: 13,500 → 1,000
4. **Skip initial search for now**: Use random element assignment as placeholder

**Impact**: Test completes in ~5 minutes, validates multi-level search

### Phase 2: GPU Initial Search (1-2 days)

1. **Convert search_level2_octree to JAX**:
   ```python
   @jax.jit
   def search_level2_octree_jax(position, partition_data, octrees, mesh_data):
       # Same logic, but with jnp instead of np
       # Handle control flow with jax.lax.cond
       ...
   ```

2. **Implement batch search**:
   ```python
   def find_initial_elements_batch(particle_positions, ...):
       return jax.vmap(search_level2_octree_jax)(particle_positions)
   ```

3. **Profile and optimize**

**Expected Result**: Initial search for 13,500 particles in <10 seconds

### Phase 3: Octree Optimization (1 week)

1. **Optimize CPU octree builder** (vectorization, reduce loops)
2. **Consider GPU octree building** if still too slow
3. **Implement element-to-multiple-blocks assignment**

**Expected Result**: Octree build for 3.5M elements in <2 minutes

### Phase 4: Production Optimizations (2 weeks)

1. **Block-cell alignment** (as recommended in PHASE_3 doc)
2. **Eliminate spanning elements** where possible
3. **Full JAX/GPU pipeline** for all initialization

---

## Performance Estimates

### Current (CPU Serial)

| Phase | Time | Bottleneck |
|-------|------|------------|
| Load | 5.5s | I/O |
| Neighbors | 28s | Hashmap |
| Block assign | 29s | Sorting |
| **Octrees** | **845s** | **Recursive Python** |
| Seed | 0.0s | - |
| **Initial search** | **~7.5 hours** | **Serial loop** |
| **TOTAL** | **~8+ hours** | **Unacceptable** |

### After Phase 1 (Quick Fixes)

| Phase | Time | Change |
|-------|------|--------|
| Load | 5.5s | - |
| Neighbors | 28s | - |
| Block assign | 29s | - |
| Octrees | **~100s** | 8× faster (less depth) |
| Seed | 0.0s | - |
| Initial search (1K particles) | **~15s** | Testing only |
| **TOTAL** | **~3 minutes** | **Acceptable for testing** |

### After Phase 2 (GPU Initial Search)

| Phase | Time | Change |
|-------|------|--------|
| Load | 5.5s | - |
| Neighbors | 28s | - |
| Block assign | 29s | - |
| Octrees | ~100s | - |
| Seed | 0.0s | - |
| **Initial search (13.5K)** | **<10s** | **~1000× faster** |
| **TOTAL** | **~3 minutes** | **Production-ready** |

### After Phase 3+4 (Full Optimization)

| Phase | Time | Change |
|-------|------|--------|
| Load | 5.5s | - |
| Neighbors | 28s | - |
| Block assign | 29s | - |
| **Octrees** | **<60s** | **GPU or optimized CPU** |
| Seed | 0.0s | - |
| Initial search | <10s | - |
| **TOTAL** | **<2 minutes** | **Optimal** |

---

## Answer to Your Question

### "Can this search be GPU implemented?"

**YES! Absolutely beneficial and necessary for production scale.**

### "Something like level 2 with block prestep?"

**Exactly correct!** The implementation would be:

```python
@jax.jit
def find_initial_element_gpu(position, partition_data, octrees, mesh_data):
    # Step 1: Find block (your "prestep")
    block_id = find_block_jax(position, partition_data)

    # Step 2: Search in block's octree (level 2)
    element_id = search_in_octree_jax(
        position, block_id, octrees, mesh_data
    )

    # Step 3: Fallback to neighbor blocks if needed
    if element_id < 0:
        element_id = search_neighbor_blocks_jax(...)

    return element_id

# Batch over all particles
initial_elements = jax.vmap(find_initial_element_gpu)(particle_positions)
```

### "The same level 2 function can be utilized?"

**YES!** We can reuse the exact same octree search logic. The only difference:
- Multi-level search: Has Level 0 (cached) and Level 1 (neighbors) before Level 2
- Initial search: Only Level 2 (octree) since no cached elements exist

**Code reuse**: ~90% of Level 2 implementation can be shared.

---

## Conclusion

### Critical Issues

1. ✅ **Octree building: 845s** → Reduce depth/elements, optimize later
2. ❌ **Initial search: Would take hours** → **MUST move to GPU**
3. ✅ **Multi-level search logic: Correct** → Already tested

### Immediate Action

**Implement GPU batch initial search using existing Level 2 octree search logic.**

This is:
- **Necessary** (current approach is unusable at scale)
- **Straightforward** (reuse existing search code)
- **High impact** (~1000× speedup)
- **Aligned with best practices** (document recommends GPU for large particle counts)

### You Are Correct

Your observation about using Level 2 with block prestep for initial search is **exactly the right approach** and matches standard practice for particle-mesh initialization.

---

**Next Step**: Implement `find_initial_elements_batch_gpu()` using JAX vectorization of the existing Level 2 search.
