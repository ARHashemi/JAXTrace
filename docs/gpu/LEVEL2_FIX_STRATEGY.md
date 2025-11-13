# Level 2 Search Fix Strategy

## Problem Statement

The GPU Level 2 search is the performance bottleneck:

1. **Original implementation (lax.scan)**: Processes 1000 elements with 90% dummies → 40× slower than CPU
2. **New implementation (full vmap)**: Processes 3.5M elements → Out of memory

## Root Cause

The CPU creates a **compact list** of block elements using `np.where`, then iterates with early termination:

```python
# CPU: Efficient
block_element_ids = np.where(element_to_block == block_id)[0]  # Compact list
for elem_id in block_element_ids:  # Only ~100 iterations
    if point_in_element(...):
        return elem_id  # Early exit
```

The GPU cannot easily replicate this because:
- `jnp.where` doesn't create compact lists in JIT context
- `jax.lax.scan` doesn't support early termination
- `jax.vmap` processes all elements (no early exit)

## Solution: Pre-Computed Block Element Arrays

### Strategy

Pre-compute a padded array of elements per block **once on CPU**, then transfer to GPU:

```python
block_elements: np.ndarray  # [n_blocks, max_elements_per_block]
block_counts: np.ndarray    # [n_blocks]
```

**Example for ThreadedA (32 blocks):**
- Block 0: 100 elements → pad to [e0, e1, ..., e99, -1, -1, ..., -1] (200 total)
- Block 1: 50 elements → pad to [e0, e1, ..., e49, -1, -1, ..., -1] (200 total)
- ...
- Shape: [32, 200] = 6,400 entries (vs 3,494,800 full mesh!)

### Implementation

#### Step 1: Build Block Element Lists (CPU, preprocessing)

```python
def build_block_element_lists(
    element_to_block: np.ndarray,
    n_blocks: int,
    max_elements_per_block: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build compact element lists for each block.

    Args:
        element_to_block: Element-to-block mapping [N_elements]
        n_blocks: Number of blocks
        max_elements_per_block: Maximum elements to store per block

    Returns:
        block_elements: [n_blocks, max_elements_per_block], padded with -1
        block_counts: [n_blocks], actual count of elements in each block

    Note:
        This is computed ONCE on CPU during initialization.
        Uses np.where to create compact lists (fast on CPU).
    """
    block_elements = np.full(
        (n_blocks, max_elements_per_block),
        -1,
        dtype=np.int32
    )
    block_counts = np.zeros(n_blocks, dtype=np.int32)

    for block_id in range(n_blocks):
        # CPU: np.where creates compact list efficiently
        elements = np.where(element_to_block == block_id)[0]
        count = len(elements)

        if count > max_elements_per_block:
            # Truncate if block is too large (rare, only for imbalanced grids)
            print(f"⚠️  Block {block_id} has {count} elements, "
                  f"truncating to {max_elements_per_block}")
            elements = elements[:max_elements_per_block]
            count = max_elements_per_block

        # Store elements (rest remain -1)
        block_elements[block_id, :count] = elements
        block_counts[block_id] = count

    return block_elements, block_counts
```

#### Step 2: Update GPUParticleTracker to Store These

```python
class GPUParticleTracker:
    def __init__(self, positions, connectivity, element_neighbors,
                 element_to_block, domain_bounds, grid_size):
        """Initialize tracker with pre-computed block element lists."""

        # ... existing initialization ...

        # NEW: Build block element lists on CPU
        n_blocks = np.prod(grid_size)

        # Determine max_elements_per_block from data
        block_sizes = []
        for block_id in range(n_blocks):
            count = np.sum(element_to_block == block_id)
            block_sizes.append(count)

        # Use 95th percentile + 20% buffer
        max_size = int(np.percentile(block_sizes, 95) * 1.2)
        print(f"  Max elements per block: {max_size}")

        self.block_elements, self.block_counts = build_block_element_lists(
            element_to_block, n_blocks, max_size
        )

        # Transfer to GPU
        self.block_elements_gpu = jax.device_put(self.block_elements)
        self.block_counts_gpu = jax.device_put(self.block_counts)

        print(f"  Block element lists: {self.block_elements.shape}")
        print(f"  Memory: {self.block_elements.nbytes / 1024**2:.1f} MB")
```

#### Step 3: Update Level 2 Search Kernel

```python
@jax.jit
def search_block_elements_jax(
    point: jnp.ndarray,
    block_id: int,
    block_elements: jnp.ndarray,      # [n_blocks, max_per_block]
    block_counts: jnp.ndarray,        # [n_blocks]
    positions: jnp.ndarray,
    connectivity: jnp.ndarray,
) -> Tuple[bool, int]:
    """
    Level 2: Block-local search using pre-computed element lists.

    Args:
        point: 3D point [3]
        block_id: Block containing point
        block_elements: Pre-computed element lists [n_blocks, max_per_block]
        block_counts: Actual element counts [n_blocks]
        positions: Node positions [N_nodes, 3]
        connectivity: Element connectivity [N_elements, 4]

    Returns:
        (found, element_id)

    Algorithm:
        1. Get pre-computed element list for this block
        2. vmap over elements (only max_per_block, not full mesh!)
        3. Return first match

    Note:
        This avoids both the lax.scan bottleneck (too many dummies)
        and the vmap bottleneck (too many elements).

        For ThreadedA with max_per_block=200:
        - Check 200 elements (not 1000 or 3.5M!)
        - ~100 real, ~100 padding
        - 5× faster than original
    """
    # Validate block
    is_valid_block = block_id >= 0

    # Get element list for this block [max_per_block]
    elements = block_elements[block_id]
    count = block_counts[block_id]

    # Check each element (vectorized)
    def check_element(elem_id):
        """Check if point is in element."""
        is_valid = elem_id >= 0

        # Safe indexing
        safe_id = jnp.where(is_valid, elem_id, 0)
        element_node_ids = connectivity[safe_id]
        vertices = positions[element_node_ids]

        # Check containment
        is_inside = jnp.where(
            is_valid,
            point_in_tetrahedron_safe(point, vertices),
            False
        )

        return is_inside, jnp.where(is_inside, elem_id, -1)

    # vmap over elements in this block (NOT full mesh!)
    found_array, result_array = jax.vmap(check_element)(elements)

    # Find first match
    found_any = jnp.any(found_array)
    first_match_idx = jnp.argmax(found_array)
    result_id = result_array[first_match_idx]

    # Return
    final_found = found_any & is_valid_block
    final_result = jnp.where(final_found, result_id, -1)

    return final_found, final_result
```

#### Step 4: Update find_containing_element_gpu Signature

```python
@jax.jit
def find_containing_element_gpu(
    point: jnp.ndarray,
    cached_element_id: int,
    block_id: int,
    element_neighbors: jnp.ndarray,
    block_elements: jnp.ndarray,      # NEW
    block_counts: jnp.ndarray,        # NEW
    positions: jnp.ndarray,
    connectivity: jnp.ndarray,
) -> int:
    """Find containing element using three-tier search."""

    # Level 0: Cached element
    found, elem_id = search_cached_element_jax(
        point, cached_element_id, positions, connectivity
    )
    if found:
        return elem_id

    # Level 1: Neighbors
    found, elem_id = search_neighbor_elements_jax(
        point, cached_element_id, element_neighbors, positions, connectivity
    )
    if found:
        return elem_id

    # Level 2: Block search (NEW signature)
    found, elem_id = search_block_elements_jax(
        point, block_id, block_elements, block_counts,  # NEW
        positions, connectivity
    )
    if found:
        return elem_id

    return -1


# Vectorized version
find_containing_elements_batch = jax.jit(jax.vmap(
    find_containing_element_gpu,
    in_axes=(0, 0, 0, None, None, None, None, None)  # Added 2 Nones
))
```

#### Step 5: Update GPUParticleTracker.update_particle_elements

```python
def update_particle_elements(self, particles, batch_size=None):
    """Update particle elements using GPU."""

    # ... existing code ...

    # Call with new arguments
    new_elem_gpu = find_containing_elements_batch(
        pos_gpu,
        cached_ids_gpu,
        block_ids_gpu,
        self.element_neighbors_gpu,
        self.block_elements_gpu,      # NEW
        self.block_counts_gpu,        # NEW
        self.positions_gpu,
        self.connectivity_gpu
    )

    # ... rest of code ...
```

## Expected Performance

### Memory Usage

**Before (lax.scan):**
- Sparse arrays: ~1000 elements per search
- For 10K particles: 10K × 1000 = 10M entries
- Memory: 10M × 4 bytes = 40 MB intermediate arrays

**Before (full vmap):**
- Full mesh: 3.5M elements per search
- For 10K particles: 10K × 3.5M = 35B entries
- Memory: 35B × 4 bytes = 140 GB (out of memory!)

**After (pre-computed lists):**
- Fixed array: 32 blocks × 200 elements = 6,400 entries
- Stored once on GPU: 6,400 × 4 bytes = 25 KB
- Per batch: 10K particles × 200 = 2M entries
- Memory: 2M × 4 bytes = 8 MB intermediate arrays

**Memory reduction: 40 MB → 8 MB (5× smaller)**

### Computation

**CPU (10K particles):**
- Level 0: 8,500 hits → 8,500 checks
- Level 1: 1,000 hits → ~4,000 checks
- Level 2: 500 particles × ~100 elements = 50,000 checks
- **Total: ~62,500 checks**
- Time: 0.7s (single-threaded)

**GPU Before (10K particles):**
- Level 0: 8,500 hits → 8,500 checks (parallel)
- Level 1: 1,000 hits → ~4,000 checks (parallel)
- Level 2: 500 particles × 1000 elements = **500,000 checks**
- **Total: ~512,500 checks** (8× more than CPU!)
- Time: 31s (overhead from checking dummies)

**GPU After (10K particles):**
- Level 0: 8,500 hits → 8,500 checks (parallel)
- Level 1: 1,000 hits → ~4,000 checks (parallel)
- Level 2: 500 particles × 200 elements = **100,000 checks**
- **Total: ~112,500 checks** (1.8× more than CPU)
- Time: Expected 0.1-0.2s (parallel speedup)

**Expected speedup vs CPU: 3-7×**

### Why GPU Will Be Faster

1. **Parallel Level 0/1**: 8× speedup from parallelism
2. **Parallel Level 2**: 4× speedup from parallelism
3. **Batch processing**: Amortizes overhead

The 1.8× more checks is acceptable because:
- GPU does them in parallel (thousands of cores)
- No early termination penalty (vmap completes in one pass)
- Transfer overhead is minimal (~10ms for 10K particles)

## Implementation Plan

1. ✅ Document algorithms side-by-side
2. ⬜ Implement `build_block_element_lists` function
3. ⬜ Update `GPUParticleTracker.__init__` to build lists
4. ⬜ Update `search_block_elements_jax` to use lists
5. ⬜ Update `find_containing_element_gpu` signature
6. ⬜ Update `find_containing_elements_batch` vmap axes
7. ⬜ Update tracker.update_particle_elements calls
8. ⬜ Run tests to verify correctness
9. ⬜ Benchmark performance (expect 3-7× speedup vs CPU)
10. ⬜ Update documentation and notebook

## Risk Mitigation

**Risk**: Some blocks may have more elements than max_per_block

**Mitigation**:
- Analyze actual distribution before setting max
- Use 95th percentile + buffer (not max)
- Print warning if truncation occurs
- For ThreadedA: max is 938K, but 95th percentile is ~200
- Large blocks are rare (load imbalance, will be fixed in Phase 8)

**Risk**: Padding wastes computation

**Mitigation**:
- Acceptable: checking 200 vs 100 is only 2× overhead
- Still 5× better than checking 1000
- GPU parallelism makes this negligible
- Future: Phase 9 hash octree will eliminate this entirely
