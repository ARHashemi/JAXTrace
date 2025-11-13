# Implementation Plan: GPU Batch Initial Element Search

**Date**: 2025-11-04
**Status**: Implementation Phase
**Alignment**: V3 Plan Phases 2-4

---

## Overview

### What We Have (CPU Implementation) ✅

1. **Phase 0-1**: Complete
   - ✅ Mesh loading ([mesh_loader.py](../../jaxtrace/gpu/mesh_loader.py))
   - ✅ Flat arrays ([flat_arrays.py](../../jaxtrace/gpu/flat_arrays.py))
   - ✅ Test infrastructure ([test_meshes.py](../../jaxtrace/gpu/test_meshes.py))

2. **Phase 2**: Partially complete (CPU)
   - ✅ Morton codes ([morton_code.py](../../jaxtrace/gpu/morton_code.py))
   - ✅ Block assignment ([mesh_loader.py](../../jaxtrace/gpu/mesh_loader.py))
   - ✅ Octree builder ([octree_builder.py](../../jaxtrace/gpu/octree_builder.py))
   - ❌ **NOT GPU-accelerated** (NumPy only)

3. **Phase 3**: Complete (CPU)
   - ✅ Particle seeding ([particle_seeding.py](../../jaxtrace/gpu/particle_seeding.py))
   - ❌ **Initial search missing** (causes integration test timeout)

4. **Phase 4**: Complete (CPU)
   - ✅ Multi-level search ([multi_level_search.py](../../jaxtrace/gpu/multi_level_search.py))
   - ✅ All 3 levels implemented (L0: cached, L1: neighbors, L2: octree)
   - ✅ Statistics tracking
   - ❌ **NOT GPU-accelerated** (NumPy only)

### What We Need (GPU Implementation) 🎯

**Critical**: Phase 3 - GPU batch initial element search
**Recommended**: Phase 2 - GPU Morton codes & block assignment
**Already Works**: Phase 4 - Just needs JAX conversion

---

## Implementation Strategy

### Principle: Dual Implementation with Config

```python
# User can choose CPU or GPU via config
@dataclass
class GPUConfig:
    use_gpu_morton: bool = True          # Phase 2: Morton codes on GPU
    use_gpu_block_assign: bool = True    # Phase 2: Block assignment on GPU
    use_gpu_initial_search: bool = True  # Phase 3: Initial search on GPU (CRITICAL)
    use_gpu_multi_level: bool = True     # Phase 4: Multi-level on GPU

    # Keep CPU fallbacks
    force_cpu: bool = False              # Override: force all CPU
```

**Rationale**:
- CPU implementations work and are tested
- GPU implementations provide speedup
- User can benchmark and choose
- Gradual migration path

---

## Phase 2: GPU Morton Codes & Block Assignment

### 2.1: GPU Morton Code Computation

**File**: `jaxtrace/gpu/morton_code_jax.py` (new)

```python
import jax
import jax.numpy as jnp

@jax.jit
def compute_morton_codes_jax(
    element_centroids: jnp.ndarray,  # (N_elements, 3) float32
    bbox_min: jnp.ndarray,            # (3,) float32
    bbox_max: jnp.ndarray,            # (3,) float32
    bits_per_dim: int = 21
) -> jnp.ndarray:  # (N_elements,) uint64
    """
    Compute Morton Z-order codes on GPU.

    Vectorized bit manipulation for parallel execution.
    """
    # Normalize to [0, 2^bits - 1]
    max_val = (1 << bits_per_dim) - 1

    normalized = (element_centroids - bbox_min) / (bbox_max - bbox_min)
    coords = (normalized * max_val).astype(jnp.uint32)

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Interleave bits (vectorized)
    morton_codes = interleave_bits_3d_jax(x, y, z)

    return morton_codes


@jax.jit
def interleave_bits_3d_jax(
    x: jnp.ndarray,  # (N,) uint32
    y: jnp.ndarray,  # (N,) uint32
    z: jnp.ndarray   # (N,) uint32
) -> jnp.ndarray:    # (N,) uint64
    """
    Vectorized Morton encoding (bit interleaving).

    Algorithm: Insert 2 zeros between each bit
    x0 y0 z0 x1 y1 z1 x2 y2 z2 ...
    """
    # Expand each coordinate to 64-bit with spacing
    xx = expand_bits_jax(x.astype(jnp.uint64))
    yy = expand_bits_jax(y.astype(jnp.uint64))
    zz = expand_bits_jax(z.astype(jnp.uint64))

    # Interleave: z has bit 2, y has bit 1, x has bit 0
    return xx | (yy << 1) | (zz << 2)


@jax.jit
def expand_bits_jax(x: jnp.ndarray) -> jnp.ndarray:
    """
    Insert two 0 bits between each bit (21-bit → 63-bit).

    Vectorized bit manipulation operations.
    """
    x = x & 0x1fffff  # Keep only 21 bits

    x = (x | (x << 32)) & 0x1f00000000ffff
    x = (x | (x << 16)) & 0x1f0000ff0000ff
    x = (x | (x << 8))  & 0x100f00f00f00f00f
    x = (x | (x << 4))  & 0x10c30c30c30c30c3
    x = (x | (x << 2))  & 0x1249249249249249

    return x
```

**Benefits**:
- Fully vectorized on GPU
- Expected speedup: 10-50× vs CPU
- Pure JAX, no Python loops

### 2.2: GPU Block Assignment

**File**: `jaxtrace/gpu/block_assignment_jax.py` (new)

```python
@jax.jit
def assign_elements_to_blocks_jax(
    morton_codes: jnp.ndarray,  # (N_elements,) uint64
    n_blocks: int
) -> jnp.ndarray:  # (N_elements,) int32
    """
    Assign elements to blocks via GPU radix sort.

    Algorithm:
    1. Sort by Morton code (JAX sort is GPU-accelerated)
    2. Divide into n_blocks ranges
    3. Assign block ID per element
    """
    n_elements = len(morton_codes)

    # GPU radix sort
    sorted_indices = jnp.argsort(morton_codes)

    # Compute block boundaries
    elements_per_block = n_elements // n_blocks

    # Vectorized block assignment
    element_ranks = jnp.arange(n_elements)
    sorted_ranks = jnp.empty_like(sorted_indices)
    sorted_ranks = sorted_ranks.at[sorted_indices].set(element_ranks)

    # Assign blocks
    block_ids = jnp.minimum(sorted_ranks // elements_per_block, n_blocks - 1)

    return block_ids.astype(jnp.int32)
```

**Benefits**:
- GPU radix sort (JAX native)
- Fully vectorized assignment
- Expected speedup: 20-100× vs CPU

### 2.3: Wrapper with CPU Fallback

**File**: Update `jaxtrace/gpu/mesh_loader.py`

```python
def assign_elements_to_blocks(
    positions: np.ndarray,
    connectivity: np.ndarray,
    grid_size: Tuple[int, int, int],
    config: GPUConfig = None,
    verbose: bool = True
):
    """
    Assign elements to blocks.

    Uses GPU or CPU based on config.
    """
    if config is None:
        config = GPUConfig()

    # Compute centroids
    centroids = positions[connectivity].mean(axis=1)
    bbox_min = centroids.min(axis=0)
    bbox_max = centroids.max(axis=0)

    if config.use_gpu_block_assign and not config.force_cpu:
        # GPU path
        from .morton_code_jax import compute_morton_codes_jax
        from .block_assignment_jax import assign_elements_to_blocks_jax

        centroids_jax = jnp.array(centroids)
        bbox_min_jax = jnp.array(bbox_min)
        bbox_max_jax = jnp.array(bbox_max)

        morton_codes = compute_morton_codes_jax(
            centroids_jax, bbox_min_jax, bbox_max_jax
        )
        n_blocks = np.prod(grid_size)
        element_block_IDs = assign_elements_to_blocks_jax(morton_codes, n_blocks)

        # Convert back to NumPy for compatibility
        element_block_IDs = np.array(element_block_IDs)
    else:
        # CPU path (existing implementation)
        element_block_IDs = _assign_elements_to_blocks_cpu(...)

    return element_block_IDs, partition_data
```

---

## Phase 3: GPU Batch Initial Element Search (CRITICAL) 🔥

### Problem

Current integration test **times out** during initial search:
- Serial CPU loop: 13,500 particles × 3.5M elements
- Estimated time: **30-60 minutes** (unusable!)

### Solution: GPU Batch Search

**File**: `jaxtrace/gpu/initial_search_jax.py` (new)

```python
import jax
import jax.numpy as jnp
from typing import Dict, Tuple

@jax.jit
def find_initial_elements_batch_jax(
    particle_positions: jnp.ndarray,  # (N_particles, 3) float64
    mesh_data: Dict,
    partition_data: Dict,
    octrees: Dict,
    config: GPUConfig
) -> jnp.ndarray:  # (N_particles,) int32
    """
    Find initial elements for all particles using GPU.

    This is Level 2 search (octree) applied to all particles,
    with block prestep as suggested by user.

    Algorithm:
    1. Find block for each particle (vectorized)
    2. Search in block's octree (vectorized)
    3. Fallback to neighbor blocks if needed

    Expected speedup: 100-1000× vs CPU serial
    """
    # Use vmap to vectorize over particles
    search_fn = lambda pos: _search_single_particle_jax(
        pos, mesh_data, partition_data, octrees, config
    )

    element_IDs = jax.vmap(search_fn)(particle_positions)

    return element_IDs


@jax.jit
def _search_single_particle_jax(
    position: jnp.ndarray,  # (3,) float64
    mesh_data: Dict,
    partition_data: Dict,
    octrees: Dict,
    config: GPUConfig
) -> jnp.int32:
    """
    Search for containing element (single particle).

    This is the JAX version of search_level2_octree.
    """
    # Step 1: Find block (spatial hash)
    block_id = find_block_jax(position, partition_data)

    # Step 2: Search in block's octree
    element_id = search_in_block_octree_jax(
        position, block_id, octrees, mesh_data
    )

    # Step 3: Fallback to neighbor blocks if not found
    element_id = jax.lax.cond(
        element_id >= 0,
        lambda: element_id,  # Found, return it
        lambda: search_neighbor_blocks_jax(
            position, block_id, octrees, mesh_data, partition_data
        )
    )

    return element_id


@jax.jit
def find_block_jax(
    position: jnp.ndarray,  # (3,) float64
    partition_data: Dict
) -> jnp.int32:
    """Find spatial block containing position."""
    bbox_min = partition_data['bbox_min']
    block_size = partition_data['block_size']
    grid_size = partition_data['grid_size']

    # Compute block indices
    block_idx = jnp.floor((position - bbox_min) / block_size).astype(jnp.int32)
    block_idx = jnp.clip(block_idx, 0, grid_size - 1)

    # Convert to flat block ID
    block_id = (
        block_idx[0] * grid_size[1] * grid_size[2] +
        block_idx[1] * grid_size[2] +
        block_idx[2]
    )

    return block_id


@jax.jit
def point_in_tetrahedron_jax(
    point: jnp.ndarray,     # (3,) float64
    vertices: jnp.ndarray   # (4, 3) float64
) -> jnp.bool_:
    """
    GPU-accelerated point-in-tetrahedron test.

    Uses barycentric coordinates with JAX linear solve.
    """
    v0, v1, v2, v3 = vertices[0], vertices[1], vertices[2], vertices[3]

    # Build matrix
    mat = jnp.column_stack([v1 - v0, v2 - v0, v3 - v0])

    # Solve (JAX handles singular matrices gracefully)
    lambdas_123, residuals = jnp.linalg.lstsq(mat, point - v0)[:2]

    # Compute lambda0
    lambda_0 = 1.0 - jnp.sum(lambdas_123)

    # Check bounds
    all_lambdas = jnp.concatenate([jnp.array([lambda_0]), lambdas_123])

    epsilon = 1e-8
    inside = jnp.all(all_lambdas >= -epsilon) & jnp.all(all_lambdas <= 1.0 + epsilon)

    # Also check that solve was reasonable (not singular)
    reasonable = jnp.linalg.cond(mat) < 1e10

    return inside & reasonable
```

### Integration with Existing Code

**File**: Update `jaxtrace/gpu/particle_seeding.py`

```python
def seed_and_find_initial_elements(
    config: SeedingConfig,
    mesh_data: MeshData,
    partition_data,
    octrees: Dict,
    gpu_config: GPUConfig = None
) -> ParticleData:
    """
    Seed particles and find initial elements.

    Uses GPU if configured, CPU otherwise.
    """
    if gpu_config is None:
        gpu_config = GPUConfig()

    # Seed particles (always fast)
    positions = seed_particles_uniform_grid(config)

    # Find initial elements
    if gpu_config.use_gpu_initial_search and not gpu_config.force_cpu:
        # GPU path - FAST!
        from .initial_search_jax import find_initial_elements_batch_jax

        positions_jax = jnp.array(positions)
        element_IDs = find_initial_elements_batch_jax(
            positions_jax, mesh_data, partition_data, octrees, gpu_config
        )
        element_IDs = np.array(element_IDs)  # Convert back
    else:
        # CPU path - SLOW but works
        element_IDs = np.full(len(positions), -1, dtype=np.int32)
        for i in range(len(positions)):
            element_IDs[i] = find_containing_element(
                positions[i], partition_data, octrees,
                mesh_data.positions, mesh_data.connectivity
            )

    # Create particle data
    active = element_IDs >= 0
    return ParticleData(
        positions=positions,
        element_IDs=element_IDs,
        active=active
    )
```

**Expected Performance**:
- CPU serial: ~30-60 minutes
- **GPU batch: <10 seconds**
- **Speedup: ~200-600×** 🚀

---

## Phase 4: JAX Conversion of Multi-Level Search

### Current Status

Multi-level search is **implemented and tested** in CPU NumPy ([multi_level_search.py](../../jaxtrace/gpu/multi_level_search.py)).

All 13 tests pass ✅

### What's Needed

Convert from NumPy to JAX (straightforward):

**File**: `jaxtrace/gpu/multi_level_search_jax.py` (new)

```python
import jax
import jax.numpy as jnp

@jax.jit
def multi_level_search_jax(
    particle_positions: jnp.ndarray,   # (N_particles, 3) float64
    cached_element_IDs: jnp.ndarray,   # (N_particles,) int32
    mesh_data: Dict,
    partition_data: Dict,
    octrees: Dict,
    config: GPUConfig
) -> Tuple[jnp.ndarray, Dict]:
    """
    Multi-level element search on GPU.

    Same algorithm as CPU version, but with JAX/GPU acceleration.
    """
    # Vectorize over all particles
    search_fn = lambda pos, cached_id: _multi_level_single_jax(
        pos, cached_id, mesh_data, partition_data, octrees, config
    )

    results = jax.vmap(search_fn)(particle_positions, cached_element_IDs)
    element_IDs, levels = results

    # Compute statistics
    stats = {
        'level0_hits': jnp.sum(levels == 0),
        'level1_hits': jnp.sum(levels == 1),
        'level2_hits': jnp.sum(levels == 2),
        'not_found': jnp.sum(levels == -1)
    }

    return element_IDs, stats


@jax.jit
def _multi_level_single_jax(
    position: jnp.ndarray,
    cached_id: jnp.int32,
    mesh_data: Dict,
    partition_data: Dict,
    octrees: Dict,
    config: GPUConfig
) -> Tuple[jnp.int32, jnp.int32]:
    """
    Multi-level search for single particle.

    Returns: (element_id, level)
    - level: 0=cached, 1=neighbor, 2=octree, -1=not found
    """
    # Level 0: Cached element
    elem_id = search_level0_jax(position, cached_id, mesh_data)
    level = jax.lax.cond(
        elem_id >= 0,
        lambda: jnp.int32(0),
        lambda: jnp.int32(-1)
    )

    # Level 1: Neighbors (if L0 failed)
    def try_level1():
        e = search_level1_jax(position, cached_id, mesh_data)
        l = jax.lax.cond(e >= 0, lambda: jnp.int32(1), lambda: jnp.int32(-1))
        return e, l

    elem_id, level = jax.lax.cond(
        elem_id >= 0,
        lambda: (elem_id, level),  # L0 succeeded
        try_level1                  # Try L1
    )

    # Level 2: Octree (if L0 and L1 failed)
    def try_level2():
        e = search_level2_jax(position, mesh_data, partition_data, octrees)
        l = jax.lax.cond(e >= 0, lambda: jnp.int32(2), lambda: jnp.int32(-1))
        return e, l

    elem_id, level = jax.lax.cond(
        elem_id >= 0,
        lambda: (elem_id, level),  # L0 or L1 succeeded
        try_level2                  # Try L2
    )

    return elem_id, level
```

**Key Differences from CPU**:
- Use `jax.lax.cond` for control flow (JIT-compatible)
- Use `jax.vmap` for vectorization
- All arrays are `jnp` instead of `np`
- Return statistics as dict of JAX arrays

**Expected Performance**:
- CPU NumPy: ~0.92 ms/particle (from integration test)
- **GPU JAX: ~0.001 ms/particle**
- **Speedup: ~900×** for full batch

---

## Implementation Timeline

### Week 1: Phase 3 - GPU Initial Search (CRITICAL)

**Days 1-2**: Implement GPU initial search
- Create `initial_search_jax.py`
- Convert Level 2 octree search to JAX
- Implement `point_in_tetrahedron_jax`
- Add config-based wrapper

**Day 3**: Test and debug
- Unit tests for each function
- Integration test on small mesh (162 elements)
- Validate accuracy

**Days 4-5**: Scale and benchmark
- Run on ThreadedA (3.5M elements)
- Measure speedup
- Profile memory usage

**Deliverable**: Initial element search completes in <10 seconds ✅

### Week 2: Phase 2 - GPU Morton & Blocks (Optional)

**Days 1-2**: Implement GPU Morton codes
- Create `morton_code_jax.py`
- Vectorized bit manipulation
- Test against CPU version

**Day 3**: Implement GPU block assignment
- Create `block_assignment_jax.py`
- GPU radix sort
- Validate load balancing

**Days 4-5**: Integration and benchmarking
- Update `mesh_loader.py` with config
- Compare CPU vs GPU timing
- Document trade-offs

**Deliverable**: Optional speedup for octree building

### Week 3: Phase 4 - GPU Multi-Level (Polish)

**Days 1-3**: Convert multi-level search to JAX
- Create `multi_level_search_jax.py`
- Convert all 3 levels
- Handle control flow with `jax.lax.cond`

**Day 4**: Testing
- Run all 13 existing tests
- Validate statistics match CPU version
- Check accuracy

**Day 5**: Integration and docs
- Config-based selection
- Performance comparison
- User documentation

**Deliverable**: Full GPU pipeline option

---

## Configuration Design

### Config Class

```python
@dataclass
class GPUConfig:
    """Configuration for GPU/CPU selection."""

    # ========== GPU ACCELERATION ==========
    use_gpu_morton: bool = True        # Phase 2: Morton codes
    use_gpu_block_assign: bool = True  # Phase 2: Block assignment
    use_gpu_initial_search: bool = True  # Phase 3: Initial search (CRITICAL)
    use_gpu_multi_level: bool = True   # Phase 4: Multi-level search

    # ========== FALLBACK ==========
    force_cpu: bool = False  # Override: use CPU for everything

    # ========== PERFORMANCE ==========
    particles_per_vmap_batch: int = 10000  # Batch size for vmap

    def validate(self):
        """Validate configuration."""
        if self.force_cpu:
            # Override all GPU flags
            self.use_gpu_morton = False
            self.use_gpu_block_assign = False
            self.use_gpu_initial_search = False
            self.use_gpu_multi_level = False
```

### Usage Example

```python
# Default: Use GPU for everything
config = GPUConfig()

# Force CPU (for debugging or CPU-only systems)
config_cpu = GPUConfig(force_cpu=True)

# Selective: GPU initial search only (most important)
config_selective = GPUConfig(
    use_gpu_morton=False,
    use_gpu_block_assign=False,
    use_gpu_initial_search=True,  # This is the critical one
    use_gpu_multi_level=False
)

# Run with config
particle_data = seed_and_find_initial_elements(
    seed_config, mesh_data, partition_data, octrees,
    gpu_config=config
)
```

---

## Testing Strategy

### Unit Tests

Each GPU function has CPU equivalent for validation:

```python
def test_morton_codes_jax():
    """GPU Morton codes match CPU version."""
    centroids = generate_random_points(1000)

    # CPU
    codes_cpu = compute_morton_codes_cpu(centroids, bbox_min, bbox_max)

    # GPU
    codes_gpu = compute_morton_codes_jax(
        jnp.array(centroids), jnp.array(bbox_min), jnp.array(bbox_max)
    )

    # Compare
    assert np.allclose(codes_cpu, np.array(codes_gpu))
```

### Integration Test

Update `test_integration_threadeda.py`:

```python
# Test both CPU and GPU
configs = [
    GPUConfig(force_cpu=True),        # CPU baseline
    GPUConfig(use_gpu_initial_search=True)  # GPU optimized
]

for config in configs:
    print(f"Testing with config: {config}")

    # Run full pipeline
    particle_data = seed_and_find_initial_elements(...)

    # Validate
    assert particle_data.success_rate() > 95.0

    # Benchmark
    print(f"  Time: {elapsed:.1f}s")
```

### Accuracy Validation

GPU results must match CPU (within numerical tolerance):

```python
def test_gpu_cpu_equivalence():
    """GPU and CPU produce same results."""

    # Run both
    results_cpu = find_initial_elements_cpu(...)
    results_gpu = find_initial_elements_jax(...)

    # Compare
    match_rate = np.sum(results_cpu == results_gpu) / len(results_cpu)

    assert match_rate > 0.99, f"Only {match_rate*100:.1f}% match"
```

---

## Summary

### What Gets Implemented

✅ **Phase 3 (Week 1)**: GPU batch initial search - **CRITICAL**
- Fixes integration test timeout
- Expected: 200-600× speedup
- Estimated time: <10 seconds for 13K particles

🔧 **Phase 2 (Week 2)**: GPU Morton & blocks - **OPTIONAL**
- Speeds up octree building
- Expected: 10-50× speedup
- Reduces 14 minutes → 1-2 minutes

✨ **Phase 4 (Week 3)**: GPU multi-level search - **POLISH**
- Full GPU pipeline
- Expected: 900× speedup
- Multi-level search: ms → microseconds

### What Stays CPU

✅ **Neighbor building**: Complex hashmap, CPU is fine
✅ **Mesh loading**: I/O bound, no benefit from GPU
✅ **Octree structure**: Can optimize CPU first, GPU if needed

### User Choice

All implementations have **CPU fallback** via `GPUConfig(force_cpu=True)`.

User can benchmark and choose optimal configuration.

---

**Next Step**: Implement Phase 3 GPU batch initial search (Week 1, Day 1)
