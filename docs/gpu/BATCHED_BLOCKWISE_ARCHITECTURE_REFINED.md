# Batched Block-Wise Architecture - REFINED IMPLEMENTATION PLAN

**Date**: 2025-11-12
**Status**: ✅ **DESIGN REFINED** - Ready for step-by-step implementation
**Branch**: `gpu_native_implementation`
**Based on**: Critical review feedback addressing all identified concerns

---

## Document Context and References

### Purpose of This Document
This is the **executable implementation plan** for JAX GPU-native particle tracking with batched block-wise architecture. It incorporates all critical review feedback and provides step-by-step implementation guidance.

### Related Documents

**Original Plans and Goals**:
- [docs/FINAL_EXECUTABLE_PLAN.md](../FINAL_EXECUTABLE_PLAN.md) - Original 9-phase roadmap (Phases 0-4 complete)
- [docs/gpu/GPU_IMPLEMENTATION_PLAN_V5_CORRECTED_COMPREHENSIVE.md](GPU_IMPLEMENTATION_PLAN_V5_CORRECTED_COMPREHENSIVE.md) - V5 dictionary-based implementation

**What We've Completed**:
- **Phase 0-3**: Forest structure, padded arrays, particles, multi-level search (V1)
- **Phase 4**: JAX vectorization attempts (V2 - OOM on ThreadedA)
- [docs/gpu/PHASE4_VECTORIZATION_RESULTS.md](PHASE4_VECTORIZATION_RESULTS.md) - V2 results: 1.2× speedup on small mesh, OOM on ThreadedA
- [docs/gpu/STRATEGY3_CRITICAL_EVALUATION.md](STRATEGY3_CRITICAL_EVALUATION.md) - Why Strategy 3 doesn't solve OOM

**What We Want to Do Now**:
- Implement **batched block-wise architecture** (this document)
- Combine particle batching (Level 1) + block-wise processing (Level 2)
- Handle millions of particles without OOM on 4 GB GPU
- Target: 500 p/s baseline (Phase 1) → 4,000 p/s production (Phase 4)

**Current Code State**:
- Working V1 implementation: [jaxtrace/gpu/multi_level_search.py](../../jaxtrace/gpu/multi_level_search.py)
- V1 performance: 188 p/s on ThreadedA (1,000 particles)
- V2 (failed): [jaxtrace/gpu/search/multi_level_search_v2.py](../../jaxtrace/gpu/search/multi_level_search_v2.py)

**Critical Review**:
- [docs/gpu/Review_BATCHED_BLOCKWISE_ARCHITECTURE.md](Review_BATCHED_BLOCKWISE_ARCHITECTURE.md) - Identified 5 critical concerns
- [docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE.md](BATCHED_BLOCKWISE_ARCHITECTURE.md) - Original architecture proposal

**Key Mesh Characteristics (ThreadedA)**:
- 3,485,406 elements, 32 blocks
- Extreme imbalance: 4 heavy blocks contain 91% of elements
- Heaviest block: 948,960 elements (444 MB padded array)
- Current memory: 660 MB static data on GPU
- Detals of mesh analysis [docs/mesh_analysis_threadedA.md](mesh_analysis_threadedA.md)

---

## Executive Summary

This document presents the **refined implementation plan** for the batched block-wise architecture, addressing all critical concerns from the review:

1. ✅ **CPU-orchestrated block loop overhead** → Mitigation strategies added
2. ✅ **Heavy block padding costs** → Mandatory hash buckets + early warnings
3. ✅ **Block imbalance bottleneck** → Adaptive subdivision plan
4. ✅ **JAX control flow constraints** → Strict enforcement guidelines
5. ✅ **VRAM monitoring** → Runtime checks and fallbacks

**Key Changes from Original**:
- More conservative performance estimates (1,000-2,000 p/s baseline)
- Mandatory hash bucket search for ALL blocks >10K elements
- Runtime VRAM monitoring with automatic batch size reduction
- Strict JAX control flow enforcement (no Python loops in kernels)
- Pathological mesh detection and user warnings

---

## Addressing Review Concerns

### Concern 1: CPU-Orchestrated Block Loop Overhead ⚠️

**Review Finding**: "CPU-side per-block orchestration may reduce GPU occupancy, especially for many light blocks. Launch overhead can limit actual throughput."

**Mitigation Strategy**:

1. **Light Block Batching** (Phase 1 priority):
   - Combine multiple light blocks (<1K elements) into single kernel launch
   - Group up to 8 light blocks with total <8K elements
   - Reduces kernel launches from 32 → ~10 for ThreadedA

2. **Performance Monitoring**:
   - Track kernel launch overhead vs compute time
   - If launch overhead >10% of total time, escalate to multi-block kernels

3. **Implementation**:
```python
def blockwise_element_search_optimized(batch, mesh_data, kernels):
    """Block-wise search with light block batching."""
    particle_groups = group_particles_by_block(batch.block_ids, mesh_data.n_blocks)

    # Separate heavy vs light blocks
    heavy_blocks = []  # >10K elements
    light_blocks = []  # <1K elements
    medium_blocks = [] # 1K-10K elements

    for block_id, local_indices in particle_groups.items():
        n_elem = mesh_data.padded_counts[block_id]
        if n_elem > 10_000:
            heavy_blocks.append((block_id, local_indices))
        elif n_elem < 1_000:
            light_blocks.append((block_id, local_indices))
        else:
            medium_blocks.append((block_id, local_indices))

    # Process heavy blocks individually (must use hash)
    for block_id, local_indices in heavy_blocks:
        results = kernels.search_heavy_block_hashed(
            batch.positions[local_indices],
            batch.cached_element_ids[local_indices],
            block_id,
            mesh_data
        )
        batch.element_ids[local_indices] = results

    # Batch light blocks (up to 8 per kernel launch)
    for i in range(0, len(light_blocks), 8):
        light_batch = light_blocks[i:i+8]
        results = kernels.search_light_blocks_batched(light_batch, mesh_data)
        # ... update results

    # Process medium blocks individually
    for block_id, local_indices in medium_blocks:
        results = kernels.search_medium_block(
            batch.positions[local_indices],
            block_id,
            mesh_data
        )
        batch.element_ids[local_indices] = results

    return batch
```

**Expected Impact**: Reduces kernel launches by 50-70% for typical meshes.

**Literature References**:
- [JAX GPU Performance Tips](https://jax.readthedocs.io/en/latest/gpu_performance_tips.html)
- [jFoF: GPU cluster finding in JAX](https://arxiv.org/html/2510.26851v1) - Real-world example of spatial hashing
- [Optimizing GPU code with JAX and Pallas](https://www.youtube.com/watch?v=pRqRYcjufjA)

---

### Concern 2: Heavy Block Padded Array Cost ⚠️

**Review Finding**: "ThreadedA-style extremely heavy blocks (444k–900k elements) mean per-block arrays are still large (100–200MB). Hash bucket trick MUST ALWAYS be used."

**Mandatory Requirements**:

1. **Hash Buckets Required for Heavy Blocks**:
   - ANY block with >10K elements MUST use Morton hash buckets
   - Hash bucket size: 8192 buckets × 100 elements = max 800K elements per block
   - If block exceeds 800K elements → automatic subdivision (Phase 2)

2. **Runtime Validation**:
```python
def validate_mesh_for_gpu(mesh_data, gpu_memory_gb=4.0):
    """Validate mesh can be processed safely on GPU."""
    warnings = []
    errors = []

    # Check individual block sizes
    for block_id in range(mesh_data.n_blocks):
        n_elem = mesh_data.padded_counts[block_id]

        if n_elem > 800_000:
            errors.append(
                f"Block {block_id} has {n_elem:,} elements (>800K limit). "
                f"CRITICAL: Mesh must be subdivided before GPU processing."
            )
        elif n_elem > 100_000:
            warnings.append(
                f"Block {block_id} has {n_elem:,} elements (heavy). "
                f"Hash bucket search will be used (mandatory)."
            )

    # Check total padded array size
    max_elem_per_block = mesh_data.padded_elements.shape[1]
    n_blocks = mesh_data.n_blocks
    padded_size_mb = (n_blocks * max_elem_per_block * 4) / 1e6

    if padded_size_mb > (gpu_memory_gb * 1024 * 0.4):
        errors.append(
            f"Padded array size ({padded_size_mb:.0f} MB) exceeds 40% of GPU memory. "
            f"Consider increasing grid resolution or using sparse storage."
        )

    # Return validation results
    return {
        'valid': len(errors) == 0,
        'warnings': warnings,
        'errors': errors,
        'heavy_blocks': [i for i in range(n_blocks)
                        if mesh_data.padded_counts[i] > 10_000]
    }
```

3. **User-Facing Warnings**:
```python
# At startup
validation = validate_mesh_for_gpu(mesh_data, gpu_memory_gb=4.0)

if not validation['valid']:
    print("\n❌ MESH VALIDATION FAILED:")
    for error in validation['errors']:
        print(f"  - {error}")
    print("\nCannot proceed with GPU processing. Please address errors above.")
    sys.exit(1)

if validation['warnings']:
    print("\n⚠️  MESH VALIDATION WARNINGS:")
    for warning in validation['warnings']:
        print(f"  - {warning}")
    print()

if validation['heavy_blocks']:
    print(f"ℹ️  Found {len(validation['heavy_blocks'])} heavy blocks:")
    for bid in validation['heavy_blocks']:
        n_elem = mesh_data.padded_counts[bid]
        print(f"   Block {bid}: {n_elem:,} elements (hash search mandatory)")
```

---

### Concern 3: Block Imbalance Bottleneck ⚠️

**Review Finding**: "Four huge blocks can bottleneck overall throughput—subdivision, or adaptive refinement or chunking, must continue to be explored."

**Adaptive Block Subdivision** (Phase 2 feature):

1. **Detect Pathological Imbalance**:
```python
def detect_block_imbalance(mesh_data):
    """Detect if mesh has pathological block imbalance."""
    counts = mesh_data.padded_counts

    # Compute imbalance metrics
    max_count = counts.max()
    mean_count = counts[counts > 0].mean()
    imbalance_ratio = max_count / mean_count

    # Check if top 4 blocks dominate
    top4_counts = np.sort(counts)[-4:].sum()
    total_counts = counts.sum()
    top4_fraction = top4_counts / total_counts

    return {
        'imbalance_ratio': imbalance_ratio,
        'top4_fraction': top4_fraction,
        'pathological': (imbalance_ratio > 100 and top4_fraction > 0.8)
    }
```

2. **Subdivision Strategy** (for pathological cases):
   - Subdivide heavy blocks (>100K elements) into sub-blocks
   - Use octree refinement within heavy blocks
   - Target: max 50K elements per sub-block
   - Trade-off: More kernel launches, but better load balance

3. **Implementation Timeline**:
   - Phase 1: Detect and warn users
   - Phase 2: Implement automatic subdivision
   - Phase 3: Adaptive refinement based on particle distribution

---

## Advanced Mitigation Strategies from Literature

This section incorporates state-of-the-art techniques from recent JAX GPU research and best practices.

### Strategy 1: CSR/Sparse Storage for Hash Buckets (Phase 3+)

**Problem**: Padded hash buckets waste memory when occupancy is low (<10% full).

**Solution**: Compressed Sparse Row (CSR) format for variable-length bucket lists.

```python
@dataclass
class MortonHashCSR:
    """CSR format for Morton hash buckets (memory-efficient)."""
    bucket_ptrs: jnp.ndarray  # (n_buckets+1,) - start index for each bucket
    element_ids: jnp.ndarray  # (total_elements,) - concatenated element lists
    n_buckets: int

    @staticmethod
    def from_padded(padded_buckets: jnp.ndarray):
        """Convert padded (n_buckets, max_per_bucket) to CSR."""
        # Count valid elements per bucket
        valid_mask = padded_buckets >= 0
        counts = valid_mask.sum(axis=1)

        # Build CSR pointers
        bucket_ptrs = jnp.concatenate([
            jnp.array([0]),
            jnp.cumsum(counts)
        ])

        # Concatenate all valid elements
        element_ids = padded_buckets[valid_mask]

        return MortonHashCSR(bucket_ptrs, element_ids, len(padded_buckets))

@jax.jit
def search_csr_bucket(position, bucket_idx, morton_hash_csr, node_positions, connectivity):
    """Search elements in CSR bucket (memory-efficient)."""
    start_idx = morton_hash_csr.bucket_ptrs[bucket_idx]
    end_idx = morton_hash_csr.bucket_ptrs[bucket_idx + 1]
    n_candidates = end_idx - start_idx

    # Use lax.fori_loop for variable-length search
    def check_candidate(i, state):
        elem_id, found = state
        candidate = morton_hash_csr.element_ids[start_idx + i]
        contains = point_in_tetrahedron(
            position,
            node_positions[connectivity[candidate]]
        )
        return jnp.where(found, (elem_id, found), (candidate, contains))

    init_state = (-1, False)
    result = jax.lax.fori_loop(0, n_candidates, check_candidate, init_state)
    return result[0]
```

**When to Use**:
- If hash bucket occupancy < 10% on average
- After Phase 2 profiling shows significant wasted memory
- Only for heavy blocks where memory is critical

**Expected Savings**: 50-90% reduction in hash bucket memory for sparse meshes.

**Reference**: Similar to CSR graph storage in [jFoF cluster finding](https://arxiv.org/html/2510.26851v1)

---

### Strategy 2: GPU-Side Block Orchestration with lax.map (Phase 2)

**Problem**: CPU loop over blocks reduces GPU occupancy.

**Solution**: Use `jax.lax.map` to dispatch all blocks in parallel on GPU.

```python
@jax.jit
def search_all_blocks_gpu_side(
    particle_groups: Dict[int, jnp.ndarray],  # Must be static dict
    all_positions: jnp.ndarray,
    all_cached_elements: jnp.ndarray,
    mesh_data: MeshData
) -> jnp.ndarray:
    """
    GPU-side orchestration of per-block search.

    NOTE: This requires static block structure - all blocks must be
    present with padding for inactive blocks.
    """
    n_blocks = mesh_data.n_blocks
    max_particles_per_block = 50_000  # Conservative limit

    # Pad particle groups to static shape
    padded_groups = jnp.full((n_blocks, max_particles_per_block), -1, dtype=jnp.int32)
    group_counts = jnp.zeros(n_blocks, dtype=jnp.int32)

    for block_id, indices in particle_groups.items():
        n = len(indices)
        padded_groups = padded_groups.at[block_id, :n].set(indices)
        group_counts = group_counts.at[block_id].set(n)

    # Vectorize over all blocks using lax.map (GPU-parallel)
    def search_single_block(block_data):
        block_id, particle_indices, n_particles = block_data

        # Extract particle data
        local_positions = all_positions[particle_indices]
        local_cached = all_cached_elements[particle_indices]

        # Search (vmap over particles in block)
        results = jax.vmap(lambda pos, cached: search_in_block(
            pos, cached, block_id, mesh_data
        ))(local_positions, local_cached)

        # Mask invalid particles
        valid_mask = particle_indices >= 0
        return jnp.where(valid_mask, results, -1)

    # Map over all blocks in parallel on GPU
    block_data = (jnp.arange(n_blocks), padded_groups, group_counts)
    all_results = jax.lax.map(search_single_block, block_data)

    return all_results
```

**Trade-offs**:
- ✅ Better GPU utilization (no CPU loop)
- ✅ Lower kernel launch overhead
- ❌ Requires static padding (wastes some memory)
- ❌ More complex code

**When to Use**:
- After Phase 2 profiling shows kernel launch overhead >15%
- When GPU utilization is low (<60%)
- For meshes with many light blocks

**Reference**: [JAX GPU Performance Tips - Map vs VMap](https://jax.readthedocs.io/en/latest/gpu_performance_tips.html)

---

### Strategy 3: Adaptive Bucket Sizing (Phase 3)

**Problem**: Fixed bucket size (8192 buckets, 100 elem/bucket) may be suboptimal.

**Solution**: Dynamically adjust bucket count based on block density.

```python
def compute_adaptive_hash_buckets(
    block_elements: np.ndarray,
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    target_bucket_occupancy: float = 0.3
) -> Tuple[int, int]:
    """
    Compute optimal hash bucket parameters for a block.

    Target: 30% average occupancy (balance memory vs collision rate)
    """
    n_elem = len(block_elements)

    # Compute bounding box and typical element size
    all_nodes = connectivity[block_elements].flatten()
    bbox = node_positions[all_nodes].max(axis=0) - node_positions[all_nodes].min(axis=0)
    volume = np.prod(bbox)

    # Estimate spatial density
    elem_density = n_elem / volume if volume > 0 else n_elem

    # Adjust bucket count based on density
    if n_elem < 1_000:
        # Light block: single bucket (no hash needed)
        return 1, n_elem
    elif n_elem < 10_000:
        # Medium block: moderate buckets
        n_buckets = 512
        bucket_size = int(n_elem / (n_buckets * target_bucket_occupancy))
        return n_buckets, max(20, bucket_size)
    else:
        # Heavy block: many buckets
        n_buckets = 8192
        bucket_size = int(n_elem / (n_buckets * target_bucket_occupancy))
        return n_buckets, max(50, min(200, bucket_size))
```

**Expected Impact**: 20-40% memory reduction for mixed light/heavy meshes.

---

### Strategy 4: Chunked Particle Processing for Heavy Blocks (Phase 3)

**Problem**: Heavy block with many particles causes OOM even with hash buckets.

**Solution**: Further subdivide particle list for heavy blocks.

```python
def search_heavy_block_chunked(
    positions: jnp.ndarray,  # (n_particles, 3)
    cached_elements: jnp.ndarray,
    block_id: int,
    mesh_data: MeshData,
    chunk_size: int = 10_000  # Particles per chunk
) -> jnp.ndarray:
    """
    Search heavy block by processing particles in chunks.

    Prevents OOM when: n_particles × bucket_size × 4 bytes > GPU memory
    """
    n_particles = len(positions)
    results = np.full(n_particles, -1, dtype=np.int32)

    # Process in chunks
    for chunk_start in range(0, n_particles, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_particles)

        # Transfer chunk to GPU
        chunk_pos = jax.device_put(positions[chunk_start:chunk_end])
        chunk_cached = jax.device_put(cached_elements[chunk_start:chunk_end])

        # Search (JIT-compiled kernel)
        chunk_results = search_block_hash_kernel(
            chunk_pos, chunk_cached, block_id, mesh_data
        )

        # Transfer back
        results[chunk_start:chunk_end] = np.array(chunk_results)

    return results
```

**When to Use**:
- Block has >100K elements AND >50K particles
- Peak memory calculation exceeds 1 GB for single kernel call
- As automatic fallback when OOM detected

---

### Strategy 5: Block Splitting at Mesh Preprocessing (Phase 2)

**Problem**: Few blocks contain 90%+ of elements (extreme imbalance).

**Solution**: Subdivide heavy blocks during forest construction.

```python
def subdivide_heavy_blocks(
    element_to_block: np.ndarray,
    connectivity: np.ndarray,
    node_positions: np.ndarray,
    max_elements_per_block: int = 50_000
) -> Tuple[np.ndarray, int]:
    """
    Recursively subdivide blocks exceeding max_elements_per_block.

    Returns:
        Updated element_to_block mapping
        New total number of blocks
    """
    # Count elements per block
    block_counts = np.bincount(element_to_block)
    n_blocks = len(block_counts)

    # Find blocks needing subdivision
    heavy_blocks = np.where(block_counts > max_elements_per_block)[0]

    if len(heavy_blocks) == 0:
        return element_to_block, n_blocks

    print(f"Subdividing {len(heavy_blocks)} heavy blocks...")

    next_block_id = n_blocks
    for block_id in heavy_blocks:
        # Get elements in this block
        elem_mask = element_to_block == block_id
        block_elements = np.where(elem_mask)[0]

        # Compute bounding box
        elem_nodes = connectivity[block_elements].flatten()
        bbox = np.column_stack([
            node_positions[elem_nodes].min(axis=0),
            node_positions[elem_nodes].max(axis=0)
        ]).flatten()

        # Split along longest axis
        axis = np.argmax(bbox[3:6] - bbox[0:3])
        mid = (bbox[axis] + bbox[3+axis]) / 2

        # Classify elements based on centroid
        for elem_id in block_elements:
            centroid = node_positions[connectivity[elem_id]].mean(axis=0)
            if centroid[axis] > mid:
                element_to_block[elem_id] = next_block_id

        next_block_id += 1

    print(f"Subdivision complete: {n_blocks} → {next_block_id} blocks")

    # Recursively subdivide if still too heavy
    return subdivide_heavy_blocks(
        element_to_block, connectivity, node_positions, max_elements_per_block
    )
```

**Impact on ThreadedA**:
- Current: 32 blocks, max 948K elements/block
- After subdivision: ~120 blocks, max 50K elements/block
- Pros: Better load balance, reduced per-block memory
- Cons: More kernel launches (mitigated by light block batching)

**When to Apply**: During mesh loading if imbalance ratio >50

---

## Critical Evaluation: Pros and Cons of Each Strategy

### Strategy 1: CSR/Sparse Storage for Hash Buckets

**Pros**:
- ✅ **Massive memory savings** (50-90%) for sparse meshes with low bucket occupancy
- ✅ **Eliminates padding waste** - only stores actual elements
- ✅ **Proven technique** - CSR is standard for sparse graph storage
- ✅ **Compatible with JAX** - `lax.fori_loop` handles variable-length iteration
- ✅ **No accuracy loss** - exact same search results as padded version

**Cons**:
- ❌ **High implementation complexity** - requires careful index management
- ❌ **Harder to debug** - indirect indexing makes errors difficult to trace
- ❌ **May be slower** - `lax.fori_loop` over variable lengths less optimized than vmap over static shapes
- ❌ **Compilation overhead** - JAX may struggle to optimize variable-length loops
- ❌ **Not needed for most meshes** - only benefits very sparse cases

**Verdict**: **Implement in Phase 3+ only if profiling shows bucket occupancy <10%**. For ThreadedA (densely packed blocks), padded format is likely more efficient.

**Risk Level**: 🟡 Medium - complexity vs benefit trade-off

---

### Strategy 2: GPU-Side Block Orchestration with lax.map

**Pros**:
- ✅ **Eliminates CPU loop overhead** - all orchestration on GPU
- ✅ **Better GPU occupancy** - all blocks processed in parallel
- ✅ **Lower launch overhead** - single kernel launch instead of N launches
- ✅ **Scalable** - benefits increase with more blocks
- ✅ **JAX-native** - uses `lax.map` as designed

**Cons**:
- ❌ **Requires static padding** - all blocks must be present (wastes memory for empty blocks)
- ❌ **Memory pressure** - need to hold `(n_blocks, max_particles_per_block)` padded array
- ❌ **Limited by max particles** - if one block has 100K particles, all blocks need 100K padding
- ❌ **Compilation time** - large padded arrays increase JIT compile time
- ❌ **Imbalance still bottlenecks** - heavy blocks still dominate, just faster launch

**Verdict**: **Implement in Phase 2 IF profiling shows**:
- Kernel launch overhead >15% of total time
- GPU utilization <60%
- Most blocks have similar particle counts (no extreme imbalance)

**For ThreadedA**: Likely NOT beneficial due to extreme imbalance (4 heavy blocks dominate). CPU loop is acceptable.

**Risk Level**: 🟡 Medium - may not provide expected speedup

---

### Strategy 3: Adaptive Bucket Sizing

**Pros**:
- ✅ **Memory-efficient** - adjusts bucket count based on actual block size
- ✅ **Low complexity** - simple calculation at mesh load time
- ✅ **No runtime overhead** - computed once during preprocessing
- ✅ **Flexible** - different blocks get different bucket structures
- ✅ **Reduces collision rate** - target occupancy balances memory vs performance

**Cons**:
- ❌ **Variable bucket sizes** - complicates kernel code (need to handle 1, 512, or 8192 buckets)
- ❌ **JIT recompilation** - different bucket sizes may trigger recompiles
- ❌ **Modest gains** - 20-40% memory savings may not be worth added complexity
- ❌ **Testing burden** - need to test all bucket size code paths
- ❌ **May not help heavy blocks** - they still need many buckets regardless

**Verdict**: **Implement in Phase 3** as optimization after baseline works. Start with fixed bucket sizes for simplicity.

**Risk Level**: 🟢 Low - safe optimization, but low priority

---

### Strategy 4: Chunked Particle Processing for Heavy Blocks

**Pros**:
- ✅ **OOM prevention** - guarantees memory safety for any particle count
- ✅ **Simple fallback** - easy to implement as automatic recovery
- ✅ **Low complexity** - just wrap existing kernel in a loop
- ✅ **Deterministic** - easy to test and reason about
- ✅ **No accuracy loss** - exact same results as non-chunked

**Cons**:
- ❌ **Multiple kernel launches** - overhead scales with chunks (e.g., 10 chunks = 10× launch overhead)
- ❌ **Transfer overhead** - repeated CPU↔GPU transfers for each chunk
- ❌ **Slower** - serial processing of chunks loses parallelism
- ❌ **Only needed rarely** - most use cases don't hit this limit
- ❌ **Better alternatives exist** - block splitting (Strategy 5) solves root cause

**Verdict**: **Implement in Phase 2 as automatic fallback**, but prefer block splitting (Strategy 5) to avoid needing it.

**Use case**: Emergency fallback when OOM detected, not primary strategy.

**Risk Level**: 🟢 Low - simple safety net

---

### Strategy 5: Block Splitting at Mesh Preprocessing

**Pros**:
- ✅ **Solves root cause** - eliminates heavy block problem entirely
- ✅ **Better load balance** - more blocks = more parallelism opportunities
- ✅ **One-time cost** - preprocessing overhead only paid at mesh load
- ✅ **Reduces per-block memory** - smaller blocks = smaller arrays
- ✅ **Enables GPU-side orchestration** - more blocks → better for Strategy 2
- ✅ **Compatible with all other strategies** - improves everything downstream

**Cons**:
- ❌ **More kernel launches** - 32 blocks → 120 blocks = 3.75× more launches (but can batch light blocks)
- ❌ **Preprocessing time** - recursive subdivision adds mesh load time
- ❌ **May break assumptions** - existing code may assume specific block count
- ❌ **Neighbor relationships** - block boundaries complicate neighbor search
- ❌ **Cache locality** - particles crossing new boundaries may have worse cache behavior

**Verdict**: **HIGHEST PRIORITY - Implement in Phase 1** for ThreadedA. This is the most impactful strategy for extreme imbalance.

**Expected improvement**:
- ThreadedA: 32 blocks (max 948K elem) → ~120 blocks (max 50K elem)
- Memory per block: 444 MB → 23 MB (19× reduction)
- Launch overhead: Mitigated by light block batching (Strategy 1)

**Risk Level**: 🟢 Low - proven technique, high reward

---

## Summary: Strategy Evaluation Matrix

| Strategy | Complexity | Memory Impact | Performance Impact | Best For | Priority |
|----------|------------|---------------|-------------------|----------|----------|
| **Block Splitting** | Medium | 🟢 **-400 MB** | 🟢 **+50%** | ThreadedA-like imbalance | **#1** ✅ |
| **Chunked Heavy Blocks** | Low | 🟢 OOM prevention | 🟡 Slower but safe | Emergency fallback | **#2** ✅ |
| **GPU-Side Orchestration** | Medium | 🟡 +100 MB | 🟢 **+100%** | Many light blocks, low imbalance | #3 (conditional) |
| **Adaptive Buckets** | Low | 🟢 -50 MB | 🟢 +10% | Mixed light/heavy meshes | #4 |
| **CSR Buckets** | High | 🟢 **-200 MB** | 🟡 Uncertain | Very sparse meshes | #5 (if needed) |

---

## Recommended Implementation Order

### Phase 1: Foundation (Week 1)
1. ✅ **Block Splitting** (Strategy 5)
   - Implement `subdivide_heavy_blocks()` in forest builder
   - Test on ThreadedA: expect 32 → ~120 blocks
   - Validate memory reduction
   - **Rationale**: Solves root cause, enables all other optimizations

2. ✅ **Basic Batched Block-Wise** (Original Plan)
   - CPU loop over blocks (accept initial overhead)
   - Hash buckets for heavy blocks (mandatory)
   - Test correctness first, optimize later

### Phase 2: Safety & Optimization (Week 2)
3. ✅ **Chunked Heavy Blocks** (Strategy 4)
   - Automatic fallback for OOM
   - Simple wrapper around existing kernels
   - **Rationale**: Safety net while we test scaling

4. 🔄 **GPU-Side Orchestration** (Strategy 2) - **CONDITIONAL**
   - Profile kernel launch overhead
   - IF overhead >15%, implement `lax.map` version
   - IF overhead <15%, skip (not worth complexity)
   - **Rationale**: Data-driven decision

### Phase 3: Refinement (Week 3)
5. 🔄 **Adaptive Buckets** (Strategy 3) - **IF TIME PERMITS**
   - Profile bucket occupancy
   - Implement if clear benefit (>20% memory savings)
   - **Rationale**: Low-hanging fruit optimization

6. ❌ **CSR Buckets** (Strategy 1) - **DEFER TO FUTURE**
   - Only if still hitting OOM after all above
   - High complexity, uncertain benefit
   - **Rationale**: Last resort, likely not needed

---

## Decision Framework: When to Use Each Strategy

```
Start → Is block imbalance extreme (ratio >50)?
  ├─ YES → Block Splitting (#1) ✅ ALWAYS
  └─ NO  → Skip splitting

     → Run baseline batched block-wise
       Measure: launch overhead, GPU utilization, memory

     → Is kernel launch overhead >15%?
       ├─ YES → GPU-Side Orchestration (#3) ✅
       └─ NO  → Keep CPU loop

     → Do you still hit OOM?
       ├─ YES → Chunked Heavy Blocks (#2) ✅ AUTOMATIC FALLBACK
       └─ NO  → All good!

     → Is bucket occupancy <20%?
       ├─ YES → Adaptive Buckets (#4) or CSR (#5) 🔄 IF TIME
       └─ NO  → Fixed buckets are fine

Final State: Production-ready, memory-safe, performant
```

---

## Summary of Advanced Strategies

| Strategy | Phase | Complexity | Impact | When to Use |
|----------|-------|------------|--------|-------------|
| **Block splitting** | 1 | Medium | 🟢 **-400 MB, +50%** | Imbalance ratio >50 (ThreadedA) |
| **Chunked heavy blocks** | 2 | Low | 🟢 **OOM prevention** | Automatic fallback |
| GPU-side orchestration | 2 | Medium | 🟢 **+100%** | Launch overhead >15% |
| Adaptive buckets | 3 | Low | 🟢 **-50 MB** | Mixed meshes |
| CSR buckets | 3 | High | 🟢 **-200 MB** | Sparse occupancy <10% |

**Implementation Priority**:
1. **Phase 1**: ✅ Block splitting (preprocessing, solves root cause)
2. **Phase 2**: ✅ Chunked heavy blocks (safety fallback)
3. **Phase 2**: 🔄 GPU-side orchestration (IF profiling shows need)
4. **Phase 3**: 🔄 Adaptive buckets (optimization, if time)
5. **Phase 3**: ❌ CSR buckets (defer, only if still OOM)

---

### Concern 4: JAX Control Flow Constraints ⚠️⚠️⚠️

**Review Finding**: "All critical GPU-level routines should avoid Python for/if/continue, nested jit, or conditionals over device arrays."

**STRICT ENFORCEMENT RULES**:

#### ✅ ALLOWED in GPU Kernels:

1. **JAX control flow primitives**:
   - `jax.lax.cond()` - conditional execution
   - `jax.lax.fori_loop()` - static loops
   - `jax.lax.while_loop()` - dynamic loops (with care)
   - `jax.lax.scan()` - sequential operations
   - `jax.lax.switch()` - multi-way branching

2. **JAX array operations**:
   - `jnp.where()` - masked selection
   - `jax.vmap()` - vectorization
   - All `jnp.*` array functions

3. **Static Python values**:
   - Loop over static constants (e.g., `for i in range(4)` for 4 neighbors)
   - Indexing with compile-time constants

#### ❌ FORBIDDEN in GPU Kernels:

1. **Python control flow**:
   - `if elem_id >= 0: ...` ❌ (use `jnp.where` instead)
   - `for i in range(len(array)): ...` ❌ (use `jax.vmap` or `lax.fori_loop`)
   - `while condition: ...` ❌ (use `lax.while_loop`)
   - `break`, `continue` ❌ (redesign algorithm)

2. **Device array conditionals**:
   - `if jnp_array[0] > 0: ...` ❌ (causes tracer error)
   - Dynamic branching based on array values ❌

3. **Nested JIT without care**:
   - `@jax.jit` inside `@jax.jit` ❌ (usually unnecessary)
   - Calling JIT function from JIT function ✅ (but avoid recompilation)

#### Code Review Checklist:

```python
# ❌ BAD - Python if on device array
@jax.jit
def search_bad(position, elements):
    for elem_id in elements:
        if elem_id >= 0:  # ❌ Python if
            if point_in_tet(position, elem_id):  # ❌ Nested Python if
                return elem_id
    return -1

# ✅ GOOD - JAX primitives
@jax.jit
def search_good(position, elements):
    def check_element(elem_id):
        valid = elem_id >= 0
        contained = jnp.where(
            valid,
            point_in_tet(position, elem_id),
            False
        )
        return jnp.where(contained, elem_id, -1)

    # Vectorize over all elements
    results = jax.vmap(check_element)(elements)

    # Select first valid result
    valid_mask = results >= 0
    first_valid_idx = jnp.argmax(valid_mask)
    return jnp.where(valid_mask.any(), results[first_valid_idx], -1)
```

**Enforcement**:
- Every kernel must pass JAX tracing test
- Code review for any Python control flow
- Automated linting to detect forbidden patterns

---

### Concern 5: VRAM Monitoring and Fallback 🟢

**Review Finding**: "All VRAM accounting should be verified at real scale at startup and after block splitting/hash rebalancing."

**Runtime VRAM Monitoring**:

```python
def monitor_gpu_memory():
    """Get current GPU memory usage."""
    import jax
    # This uses CUDA/ROCm APIs via JAX backend
    devices = jax.devices()
    if not devices:
        return {'available_mb': 0, 'used_mb': 0}

    # JAX provides memory stats
    mem_info = devices[0].memory_stats() if hasattr(devices[0], 'memory_stats') else {}

    # Fallback to nvidia-smi if JAX doesn't provide stats
    if not mem_info:
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            used, total = map(int, result.stdout.strip().split(','))
            return {'available_mb': total - used, 'used_mb': used, 'total_mb': total}
        except:
            return {'available_mb': 4000, 'used_mb': 0, 'total_mb': 4000}  # Fallback

    return mem_info

def adaptive_batch_size(initial_batch_size, mesh_data):
    """Dynamically adjust batch size based on actual VRAM usage."""
    print(f"\nTesting batch size: {initial_batch_size:,} particles...")

    # Try a small test batch
    test_size = min(1000, initial_batch_size)

    mem_before = monitor_gpu_memory()

    try:
        # Run test batch
        test_particles = create_test_particles(test_size)
        _ = process_batch_gpu(test_particles, mesh_data, kernels)

        mem_after = monitor_gpu_memory()
        mem_used_mb = mem_after['used_mb'] - mem_before['used_mb']

        # Extrapolate to full batch
        mb_per_particle = mem_used_mb / test_size
        safe_batch_size = int((mem_after['available_mb'] * 0.7) / mb_per_particle)

        print(f"  Memory test: {mem_used_mb:.1f} MB for {test_size} particles")
        print(f"  Estimated: {mb_per_particle:.3f} MB/particle")
        print(f"  Safe batch size: {safe_batch_size:,} particles")

        return min(safe_batch_size, initial_batch_size)

    except Exception as e:
        if "out of memory" in str(e).lower():
            print(f"  ❌ OOM with {test_size} particles - reducing batch size")
            return initial_batch_size // 2
        else:
            raise

def time_march_with_fallback(particles, mesh_data, config):
    """Time march with automatic OOM recovery."""
    batch_size = config.batch_size

    while batch_size >= 1000:
        try:
            # Try with current batch size
            return time_march_batched_blockwise(
                particles, mesh_data, config.with_batch_size(batch_size)
            )
        except Exception as e:
            if "out of memory" in str(e).lower():
                print(f"\n⚠️  OOM detected with batch_size={batch_size:,}")
                print(f"   Reducing batch size to {batch_size//2:,} and retrying...")
                batch_size = batch_size // 2

                # Clear GPU memory
                jax.clear_caches()
                import gc
                gc.collect()
            else:
                raise

    raise RuntimeError("Cannot find safe batch size - mesh too large for GPU")
```

---

## Refined Performance Estimates

### Conservative Baseline (More Realistic)

**Per-batch processing time** (200K particles on ThreadedA):

| Operation | Time (Conservative) | Notes |
|-----------|---------------------|-------|
| Transfer to GPU | 10 ms | Without pinned memory |
| Velocity interpolation | 20 ms | JIT kernel, first iteration |
| RK4 integration | 80 ms | 4 substeps × 20 ms |
| Block assignment | 10 ms | Morton code lookup |
| Group by block (CPU) | 5 ms | Dictionary grouping |
| Block-wise search (32 blocks) | 120 ms | With kernel launch overhead |
| Transfer to RAM | 10 ms | Regular transfer |
| **Total** | **255 ms** | Per 200K particles |

**Throughput**: 200,000 / 0.255 = **784 particles/s per batch**

**For 1M particles**: 1,000,000 / 784 = 1,275 seconds ≈ **21 minutes per time step**

### Optimized Target (After Phase 2-3)

| Operation | Time (Optimized) | Improvements |
|-----------|------------------|--------------|
| Transfer to GPU | 3 ms | Pinned memory + async |
| Global GPU ops | 25 ms | Kernel fusion |
| Block-wise search | 40 ms | Light block batching + hash optimization |
| Transfer to RAM | 3 ms | Pinned + async |
| **Effective** | **50 ms** | With overlap |

**Throughput**: 200,000 / 0.050 = **4,000 particles/s**

**For 1M particles**: 1,000,000 / 4,000 = 250 seconds ≈ **4 minutes per time step**

**Speedup Roadmap**:
- Phase 1 baseline: 800 p/s (1× baseline)
- Phase 2 optimized: 2,000 p/s (2.5×)
- Phase 3 production: 4,000 p/s (5×)

---

## Implementation Phases (Revised)

### Phase 1: Core Implementation + Validation (Week 1)

**Priority: Correctness over performance**

**Tasks**:
1. ✅ Implement basic batched block-wise search (CPU loop, single-block kernels)
2. ✅ Add mesh validation (heavy block detection, VRAM checks)
3. ✅ Enforce JAX control flow rules (code review + linting)
4. ✅ Runtime VRAM monitoring
5. ✅ Hash bucket search for heavy blocks (mandatory)

**Success Criteria**:
- Process 200K particles on ThreadedA without OOM
- All heavy blocks (>10K elem) use hash buckets
- No Python control flow in GPU kernels
- Throughput > 500 p/s (baseline)

**Files to Create**:
```
jaxtrace/gpu/batching/
├── __init__.py
├── batch_config.py          # Configuration with auto-tune
├── batch_processor.py       # Main batching logic
├── memory_utils.py          # VRAM monitoring + validation
├── block_grouping.py        # Particle grouping by block
└── validation.py            # Mesh validation + warnings
```

**Files to Modify**:
```
jaxtrace/gpu/
├── multi_level_search.py    # Add block-wise search function
├── __init__.py              # Export batching API
└── search/
    └── block_search.py      # NEW: Per-block search kernels
```

**Tests**:
```
tests/gpu/
├── test_batch_processor.py  # Unit tests
├── test_memory_utils.py     # VRAM monitoring tests
├── test_validation.py       # Mesh validation tests
└── test_block_search.py     # Block-wise kernel tests

test_threadeda_batched_phase1.py  # Integration test
```

---

### Phase 2: Performance Optimization (Week 2)

**Priority: Achieve 2,000 p/s throughput**

**Tasks**:
1. Light block batching (combine <1K elem blocks)
2. Kernel launch profiling and optimization
3. Pinned memory allocators (via CuPy/DLPack if needed)
4. Async transfer with overlap
5. Memory usage profiling

**Success Criteria**:
- Throughput > 2,000 p/s on ThreadedA
- Kernel launch overhead < 15% of total time
- Peak GPU memory < 2 GB for 200K batch

**New Features**:
- Multi-block kernel for light blocks
- Adaptive batch size based on actual VRAM usage
- Transfer/compute overlap

---

### Phase 3: Production Hardening (Week 3)

**Priority: Robustness and scalability**

**Tasks**:
1. Automatic OOM recovery (reduce batch size)
2. Pathological mesh detection + warnings
3. Block subdivision for extreme imbalance
4. Comprehensive logging and progress bars
5. Performance profiling tools

**Success Criteria**:
- Graceful handling of OOM (auto-reduce batch size)
- Clear warnings for pathological meshes
- Throughput > 3,000 p/s on ThreadedA
- Handles meshes up to 10M elements

---

### Phase 4: Advanced Features (Week 4)

**Priority: Complete feature set**

**Tasks**:
1. Velocity interpolation GPU kernel
2. RK4 time integration GPU kernel
3. Complete time-marching pipeline
4. Multi-timestep integration
5. User documentation

**Success Criteria**:
- Full time-marching pipeline working
- Throughput > 4,000 p/s on ThreadedA
- Complete user guide with examples
- Production-ready release

---

## Critical Implementation Guidelines

### 1. JAX Kernel Development Checklist

Every GPU kernel MUST pass this checklist:

- [ ] No Python `if`/`for`/`while` over device arrays
- [ ] Uses `jax.vmap()` or `lax.fori_loop()` for iteration
- [ ] Uses `jnp.where()` for conditional selection
- [ ] No nested `@jax.jit` decorators
- [ ] No dynamic array shapes (use padding)
- [ ] Tested with `jax.make_jaxpr()` to verify XLA compilation
- [ ] Memory usage profiled with test batch

### 2. Memory Safety Checklist

Before running on production mesh:

- [ ] Run `validate_mesh_for_gpu()` - must pass
- [ ] Heavy blocks (>10K) use hash buckets
- [ ] Batch size auto-tuned or manually validated
- [ ] Test batch on GPU to measure actual VRAM usage
- [ ] Monitor VRAM during first full batch
- [ ] OOM fallback tested

### 3. Performance Profiling Checklist

For each optimization:

- [ ] Profile kernel launch overhead (should be <15%)
- [ ] Profile compute time vs transfer time
- [ ] Check GPU utilization (target >60%)
- [ ] Measure per-level search hit rates
- [ ] Compare before/after throughput

---

## Step-by-Step Execution Plan

### Step 1: Setup and Validation (Day 1)

1. Create directory structure
2. Implement `validation.py` with mesh checks
3. Test on ThreadedA mesh - expect warnings for heavy blocks
4. Verify all heavy blocks flagged correctly

### Step 2: Memory Utilities (Day 1-2)

1. Implement `memory_utils.py` with VRAM monitoring
2. Test GPU memory detection on your system
3. Implement batch size calculation
4. Test with different batch sizes on small mesh

### Step 3: Block Grouping (Day 2)

1. Implement `block_grouping.py` for particle grouping
2. Test grouping logic with synthetic data
3. Verify efficient dictionary implementation
4. Profile grouping time (should be <5ms for 200K particles)

### Step 4: Block Search Kernels (Day 3-4)

1. Implement single-block search kernel (following V1 logic)
2. Add hash bucket search for heavy blocks
3. Enforce JAX control flow rules
4. Test on single block with known results
5. Verify no Python control flow in compiled code

### Step 5: Batch Processor (Day 4-5)

1. Implement main batching loop
2. Integrate block grouping + block search
3. Test on small mesh (6K elements)
4. Test on ThreadedA with 1K particles
5. Verify memory usage stays under budget

### Step 6: Integration and Testing (Day 5-6)

1. Create integration test for ThreadedA
2. Test with 10K, 50K, 100K, 200K particles
3. Profile performance and memory
4. Identify bottlenecks
5. Document baseline performance

### Step 7: Optimization Loop (Day 7+)

1. Implement highest-impact optimization
2. Test and profile
3. Compare to baseline
4. Iterate

---

## Risk Mitigation

### High-Risk Items

1. **Heavy block OOM** 🔴
   - Mitigation: Mandatory hash buckets + validation
   - Fallback: Auto-subdivide blocks >800K elements

2. **Python control flow in kernels** 🔴
   - Mitigation: Strict code review + automated checks
   - Fallback: Rewrite kernel following JAX primitives

3. **Kernel launch overhead** 🟡
   - Mitigation: Light block batching
   - Fallback: Accept lower performance, focus on correctness first

4. **Unpredictable VRAM usage** 🟡
   - Mitigation: Runtime monitoring + adaptive batch size
   - Fallback: Conservative batch size (100K)

### Low-Risk Items

1. **Block grouping performance** 🟢
   - CPU dictionary is fast enough (<5ms)

2. **Transfer overhead** 🟢
   - Even without pinned memory, transfer is <10% of time

3. **Auto-tuning accuracy** 🟢
   - Conservative safety factors prevent OOM

---

## Success Metrics

### Phase 1 (Week 1)

- ✅ Processes 200K particles without OOM
- ✅ Correctness: 100% match with V1 results
- ✅ Throughput: >500 p/s (baseline)
- ✅ Memory: <2 GB peak usage

### Phase 2 (Week 2)

- ✅ Throughput: >2,000 p/s (4× Phase 1)
- ✅ Launch overhead: <15% total time
- ✅ GPU utilization: >60%

### Phase 3 (Week 3)

- ✅ Throughput: >3,000 p/s
- ✅ OOM recovery: Auto-reduces batch size
- ✅ Pathological mesh: Detects and warns

### Phase 4 (Week 4)

- ✅ Complete time-marching pipeline
- ✅ Throughput: >4,000 p/s
- ✅ Production-ready documentation

---

## Conclusion

This refined plan addresses all critical concerns from the review:

✅ **CPU loop overhead** → Light block batching
✅ **Heavy block cost** → Mandatory hash buckets + validation
✅ **Block imbalance** → Detection + adaptive subdivision (Phase 2)
✅ **JAX control flow** → Strict enforcement + checklist
✅ **VRAM monitoring** → Runtime checks + adaptive fallback

**Key Philosophy Changes**:
1. **Correctness first, performance second** - Phase 1 focuses on getting it right
2. **Conservative estimates** - Start with 500 p/s baseline, optimize to 4,000 p/s
3. **Mandatory safety checks** - Validate mesh before processing
4. **Graceful degradation** - Auto-reduce batch size on OOM
5. **User transparency** - Clear warnings for pathological cases

**Ready to proceed with Step 1: Setup and Validation.**

Shall we begin implementation?
