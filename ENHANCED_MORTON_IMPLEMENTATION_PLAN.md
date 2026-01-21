# Enhanced Morton Neighbor Search - Implementation Plan

**Date**: 2025-12-31
**Goal**: Implement 5×5×5 boundary fallback to improve retention from 82% → 87-92%
**Estimated Time**: 8 hours

---

## Problem Analysis

### Current Implementation
- **Function**: `search_L2_morton_neighbors_single` ([morton_global_search.py:607](jaxtrace/gpu/search/morton_global_search.py#L607))
- **Search pattern**: 3×3×3 = 27 octants
- **Success rate**: 67.7% (from diagnostic)
- **Failure mode**: Particles near refined/coarse boundaries where Morton neighbors ≠ spatial neighbors

### Root Cause
Morton Z-order curve has **discontinuities** at octree boundaries:
- Particles at refinement boundaries
- Morton 3×3×3 neighbors may not include containing element
- Need larger search radius to cross discontinuity gaps

---

## Solution Design

### Enhanced Search Strategy

**Two-tier search**:
1. **Tier 1 (Fast path)**: 3×3×3 search (27 octants) - 67% success
2. **Tier 2 (Fallback)**: 5×5×5 outer shell (98 octants) - only if Tier 1 fails

**Total octants**: 27 + 98 = 125 (5×5×5)

**Performance**:
- 67% particles: 27 octants (fast, unchanged)
- 33% particles: 125 octants (4.6× slower)
- **Average**: 0.67 × 27 + 0.33 × 125 = 59 octants/particle
- **Overhead**: 2.2× vs current (acceptable for +5-10% retention)

---

## Implementation Approach

### Option A: New Function (RECOMMENDED)

Create `search_L2_morton_neighbors_extended` that wraps existing function:

```python
def search_L2_morton_neighbors_extended(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMorton
) -> jnp.int32:
    """
    Enhanced Morton neighbor search with 5×5×5 boundary fallback.

    Tier 1: 3×3×3 search (27 octants) - fast path
    Tier 2: 5×5×5 outer shell (98 octants) - boundary fallback
    """
    # Tier 1: Standard 3×3×3 search
    elem_id = search_L2_morton_neighbors_single(pos, mesh_gpu)

    if elem_id >= 0:
        return elem_id  # Success - fast path

    # Tier 2: Search outer shell of 5×5×5
    # Generate 5×5×5 neighbors minus inner 3×3×3
    elem_id = search_5x5x5_outer_shell(pos, mesh_gpu)

    return elem_id
```

**Advantages**:
- Minimal changes to existing code
- Backward compatible (can keep both functions)
- Easy to test and rollback

### Option B: Modify Existing Function

Replace `search_L2_morton_neighbors_single` with enhanced version.

**Advantages**:
- Single code path
- Cleaner API

**Disadvantages**:
- More risky (breaks existing function)
- Harder to A/B test

**Verdict**: Use Option A for initial implementation, then merge if successful

---

## Detailed Implementation Steps

### Step 1: Create Extended Neighbor Generator (2 hours)

**File**: `jaxtrace/gpu/search/morton_neighbors.py`

**Add function**:
```python
def get_98_extended_neighbor_prefixes_jax(
    center_prefix: jnp.uint64,
    depth: int,
    max_coord: jnp.int32
) -> jax.Array:
    """
    Generate Morton prefixes for outer shell of 5×5×5 neighborhood.

    Returns 98 octants: all (dx, dy, dz) where max(|dx|, |dy|, |dz|) == 2
    Excludes inner 3×3×3 (where max(|dx|, |dy|, |dz|) <= 1)

    Index mapping: idx ∈ [0, 98)
    - dx ∈ [-2, -1, 0, 1, 2]
    - dy ∈ [-2, -1, 0, 1, 2]
    - dz ∈ [-2, -1, 0, 1, 2]
    - Exclude: |dx| <= 1 AND |dy| <= 1 AND |dz| <= 1

    Total: 5³ - 3³ = 125 - 27 = 98
    """
    cx, cy, cz = decode_morton_prefix_jax(center_prefix, depth)

    neighbors = jnp.zeros(98, dtype=jnp.uint64)

    def compute_extended_neighbor(linear_idx, neighbors_arr):
        # Map linear_idx to (dx, dy, dz) skipping inner 3×3×3

        # Generate 5×5×5 index (0-124)
        full_idx = linear_idx + jnp.where(linear_idx >= 13, 27, 0)  # Skip inner 27
        # Actually this logic is wrong, need to think more carefully

        # Better approach: enumerate all 125, skip inner 27
        # OR: generate only outer shell directly

        # Direct enumeration of outer shell:
        # All (dx, dy, dz) where max(|dx|, |dy|, |dz|) == 2

        # Simpler: use 5×5×5 grid, filter out inner 3×3×3
        dz = (full_idx % 5) - 2  # -2, -1, 0, 1, 2
        dy = ((full_idx // 5) % 5) - 2
        dx = ((full_idx // 25) % 5) - 2

        # Check if in outer shell: max(|dx|, |dy|, |dz|) == 2
        max_offset = jnp.maximum(jnp.maximum(jnp.abs(dx), jnp.abs(dy)), jnp.abs(dz))
        is_outer = max_offset == 2

        # Compute neighbor coordinates
        nx = cx + dx
        ny = cy + dy
        nz = cz + dz

        # Clamp to valid range
        nx = jnp.clip(nx, 0, max_coord)
        ny = jnp.clip(ny, 0, max_coord)
        nz = jnp.clip(nz, 0, max_coord)

        # Encode back to Morton prefix
        neighbor_prefix = encode_morton_prefix_jax(nx, ny, nz, depth)

        # Insert if outer shell (else keep zero)
        neighbors_arr = jnp.where(
            is_outer,
            neighbors_arr.at[linear_idx].set(neighbor_prefix),
            neighbors_arr
        )

        return neighbors_arr

    # WAIT - this approach won't work because we're iterating 98 times
    # but mapping to 125 indices. Let me fix this.

    # CORRECTED APPROACH:
    # Iterate 125 times (full 5×5×5), skip inner 27, write to output array

    return neighbors
```

**ISSUE**: JAX array updates in loops are tricky. Let me use a simpler approach:

```python
def get_98_extended_neighbor_prefixes_jax(
    center_prefix: jnp.uint64,
    depth: int,
    max_coord: jnp.int32
) -> jax.Array:
    """Generate 98 outer-shell neighbors of 5×5×5 grid."""

    cx, cy, cz = decode_morton_prefix_jax(center_prefix, depth)

    # Generate all 125 neighbors of 5×5×5 grid
    neighbors_125 = jnp.zeros(125, dtype=jnp.uint64)

    def compute_neighbor(idx, neighbors_arr):
        # Map idx ∈ [0, 125) to (dx, dy, dz) ∈ [-2, 2]³
        dz = (idx % 5) - 2
        dy = ((idx // 5) % 5) - 2
        dx = ((idx // 25) % 5) - 2

        nx = jnp.clip(cx + dx, 0, max_coord)
        ny = jnp.clip(cy + dy, 0, max_coord)
        nz = jnp.clip(cz + dz, 0, max_coord)

        neighbor_prefix = encode_morton_prefix_jax(nx, ny, nz, depth)

        neighbors_arr = neighbors_arr.at[idx].set(neighbor_prefix)

        return neighbors_arr

    neighbors_125 = lax.fori_loop(0, 125, compute_neighbor, neighbors_125)

    # Filter out inner 3×3×3 (indices 26-78 in specific pattern)
    # Actually, let's compute which indices to KEEP (outer shell)

    # Create mask for outer shell
    def is_outer_shell(idx):
        dz = (idx % 5) - 2
        dy = ((idx // 5) % 5) - 2
        dx = ((idx // 25) % 5) - 2

        max_offset = jnp.maximum(jnp.maximum(jnp.abs(dx), jnp.abs(dy)), jnp.abs(dz))
        return max_offset == 2

    # Build mask (vectorized)
    indices = jnp.arange(125)
    mask = jax.vmap(is_outer_shell)(indices)

    # Extract outer shell elements
    neighbors_98 = neighbors_125[mask]

    return neighbors_98
```

**PROBLEM**: `neighbors_125[mask]` may not give exactly 98 elements due to edge cases.

**SIMPLEST APPROACH**: Just iterate 125 times, mark inner 27 as invalid (-1), let search skip them.

Actually, let me reconsider the whole approach...

---

## REVISED IMPLEMENTATION (Simpler)

### Key Insight
Instead of generating 98 separate neighbors, **just modify the loop bound** from 27 to 125.

### Simplest Implementation

**File**: `jaxtrace/gpu/search/morton_global_search.py`

**Add new function**:
```python
def search_L2_morton_neighbors_extended(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMorton,
    search_5x5x5: jnp.bool_ = False
) -> jnp.int32:
    """
    Morton neighbor search with optional 5×5×5 fallback.

    If search_5x5x5=False: searches 3×3×3 = 27 octants (standard)
    If search_5x5x5=True: searches 5×5×5 = 125 octants (extended)
    """
    from jaxtrace.gpu.search.morton_neighbors import (
        decode_morton_prefix_jax,
        encode_morton_prefix_jax,
    )

    # Compute Morton code
    morton_query = morton_encode_position_jax(
        pos, mesh_gpu.bbox_min, mesh_gpu.bbox_max, mesh_gpu.max_depth
    )

    table_depth_int = int(mesh_gpu.table_depth)
    center_prefix = morton_query

    # Decode center octant
    cx, cy, cz = decode_morton_prefix_jax(center_prefix, table_depth_int)
    max_coord = jnp.int32((2 ** table_depth_int) - 1)

    # Choose grid size based on search_5x5x5
    grid_size = jnp.where(search_5x5x5, 5, 3)
    radius = jnp.where(search_5x5x5, 2, 1)
    n_neighbors = grid_size ** 3

    # Search all neighbors
    def search_neighbor(i, state):
        elem_id, found = state
        active = ~found

        # Map i to (dx, dy, dz) in [-radius, radius]³
        offset = grid_size
        dz = (i % offset) - radius
        dy = ((i // offset) % offset) - radius
        dx = ((i // (offset * offset)) % offset) - radius

        # Compute neighbor coordinates
        nx = jnp.clip(cx + dx, 0, max_coord)
        ny = jnp.clip(cy + dy, 0, max_coord)
        nz = jnp.clip(cz + dz, 0, max_coord)

        # Encode neighbor prefix
        neighbor_prefix = encode_morton_prefix_jax(nx, ny, nz, table_depth_int)

        # Look up leaves and search (same logic as original)
        # ... [rest of search_in_leaf logic] ...

        return (elem_id, found)

    init_state = (jnp.int32(-1), False)
    final_elem_id, final_found = lax.fori_loop(
        0,
        n_neighbors,  # 27 or 125 based on search_5x5x5
        search_neighbor,
        init_state
    )

    return final_elem_id
```

**PROBLEM**: `n_neighbors` is data-dependent, but `lax.fori_loop` requires **constant** upper bound!

---

## FINAL APPROACH (Actually Correct for JAX)

### Solution: Two-Stage Search with Fixed Loop Bounds

```python
def search_L2_morton_neighbors_enhanced(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMorton
) -> jnp.int32:
    """
    Enhanced Morton search: 3×3×3 then 5×5×5 outer shell if needed.
    """
    # Stage 1: 3×3×3 search (EXISTING FUNCTION)
    elem_id = search_L2_morton_neighbors_single(pos, mesh_gpu)

    # If found, return immediately
    found_3x3x3 = elem_id >= 0

    # Stage 2: Search 5×5×5 outer shell (98 octants) if not found
    # Use jnp.where to make this data-independent for JAX
    elem_id_extended = search_5x5x5_outer_shell(pos, mesh_gpu, elem_id, found_3x3x3)

    # Return best result
    return jnp.where(found_3x3x3, elem_id, elem_id_extended)


def search_5x5x5_outer_shell(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMorton,
    current_elem: jnp.int32,
    already_found: jnp.bool_
) -> jnp.int32:
    """
    Search outer shell of 5×5×5 neighborhood (98 octants).

    Searches all octants where max(|dx|, |dy|, |dz|) == 2.
    Skips inner 3×3×3 (already searched).
    """
    from jaxtrace.gpu.search.morton_neighbors import (
        decode_morton_prefix_jax,
        encode_morton_prefix_jax,
    )

    # Compute Morton code and decode
    morton_query = morton_encode_position_jax(
        pos, mesh_gpu.bbox_min, mesh_gpu.bbox_max, mesh_gpu.max_depth
    )

    table_depth_int = int(mesh_gpu.table_depth)
    cx, cy, cz = decode_morton_prefix_jax(morton_query, table_depth_int)
    max_coord = jnp.int32((2 ** table_depth_int) - 1)

    # Search all 125 octants, skip inner 27
    def search_neighbor(i, state):
        elem_id, found = state

        # Skip if already found
        active = (~found) & (~already_found)

        # Map i ∈ [0, 125) to (dx, dy, dz) ∈ [-2, 2]³
        dz = (i % 5) - 2
        dy = ((i // 5) % 5) - 2
        dx = ((i // 25) % 5) - 2

        # Skip inner 3×3×3: |dx| <= 1 AND |dy| <= 1 AND |dz| <= 1
        max_offset = jnp.maximum(jnp.maximum(jnp.abs(dx), jnp.abs(dy)), jnp.abs(dz))
        is_outer = max_offset == 2

        active = active & is_outer

        # Compute neighbor coordinates
        nx = jnp.clip(cx + dx, 0, max_coord)
        ny = jnp.clip(cy + dy, 0, max_coord)
        nz = jnp.clip(cz + dz, 0, max_coord)

        # Encode neighbor prefix
        neighbor_prefix = encode_morton_prefix_jax(nx, ny, nz, table_depth_int)

        # Look up leaves for this prefix
        shift_amount = 63 - (table_depth_int * 3)
        prefix_idx = lax.shift_right_logical(neighbor_prefix, jnp.uint64(shift_amount))
        prefix_idx = prefix_idx.astype(jnp.int32)
        prefix_idx = jnp.clip(prefix_idx, 0, mesh_gpu.prefix_start.shape[0] - 1)

        first_leaf = mesh_gpu.prefix_start[prefix_idx]
        num_leaves_in_prefix = mesh_gpu.prefix_length[prefix_idx]

        has_leaves = num_leaves_in_prefix > 0
        valid_leaf = first_leaf >= 0

        # Search up to 3 leaves in this prefix (same as existing code)
        def search_single_leaf(leaf_offset, current_elem, current_found):
            leaf_id = first_leaf + leaf_offset
            valid = (leaf_offset < num_leaves_in_prefix) & (leaf_id >= 0) & (leaf_id < mesh_gpu.n_leaves) & (~current_found)
            result = jnp.where(valid, search_in_leaf_global(pos, leaf_id, mesh_gpu), jnp.int32(-1))
            improved = result >= 0
            new_elem = jnp.where(improved, result, current_elem)
            new_found = current_found | improved
            return new_elem, new_found

        # Unroll 3-leaf search
        elem_0, found_0 = search_single_leaf(0, jnp.int32(-1), False)
        elem_1_search, found_1_search = search_single_leaf(1, elem_0, found_0)
        elem_1 = jnp.where(found_0, elem_0, elem_1_search)
        found_1 = found_0 | found_1_search
        elem_2_search, found_2_search = search_single_leaf(2, elem_1, found_1)
        elem_2 = jnp.where(found_1, elem_1, elem_2_search)

        elem_neighbor = jnp.where(
            active & has_leaves & valid_leaf,
            elem_2,
            jnp.int32(-1)
        )

        # Update if found
        improve = (elem_neighbor >= 0) & active
        elem_id = jnp.where(improve, elem_neighbor, elem_id)
        found = found | improve

        return (elem_id, found)

    # Search all 125 octants (98 will be active due to is_outer filter)
    init_state = (current_elem, already_found)
    final_elem_id, final_found = lax.fori_loop(
        0,
        125,  # Fixed bound - JAX compatible
        search_neighbor,
        init_state
    )

    return final_elem_id
```

**This approach**:
- ✅ Fixed loop bound (125) - JAX compatible
- ✅ Uses masking to skip inner 3×3×3 - no control flow
- ✅ Reuses existing search_in_leaf logic
- ✅ Data-independent execution (all 125 iterations run, but masked)

---

## Implementation Timeline

### Phase 1: Add Helper Function (2 hours)
- Add `search_5x5x5_outer_shell` to `morton_global_search.py`
- Copy existing logic from `search_L2_morton_neighbors_single`
- Modify loop to iterate 125 times with outer-shell masking

### Phase 2: Add Enhanced Wrapper (1 hour)
- Add `search_L2_morton_neighbors_enhanced` to `morton_global_search.py`
- Calls 3×3×3 first, then 5×5×5 if needed
- Export from module

### Phase 3: Integration with RK4 (1 hour)
- Modify `rk4_fully_fused_timedep.py`
- Replace `search_L2_morton_neighbors_single` with `search_L2_morton_neighbors_enhanced`
- No other changes needed (API compatible)

### Phase 4: Testing (2 hours)
- Run production script
- Verify retention improvement
- Check throughput degradation
- A/B test: old vs enhanced search

### Phase 5: Optimization (2 hours if needed)
- Profile hot spots
- Consider reducing 5×5×5 to 4×4×4 if throughput too slow
- Consider adaptive radius based on element size

---

## Testing Plan

### Test 1: Unit Test (30 min)
Create test for outer-shell generation:
```python
def test_outer_shell_count():
    # 5×5×5 = 125
    # 3×3×3 = 27
    # Outer shell = 98
    assert count_outer_shell_octants() == 98

def test_no_duplicates():
    # Ensure 3×3×3 and outer shell don't overlap
    neighbors_27 = get_3x3x3_neighbors(...)
    neighbors_98 = get_outer_shell(...)
    assert len(set(neighbors_27) & set(neighbors_98)) == 0
```

### Test 2: Production Test (1 hour)
```bash
python production_tracking_fully_fused_timedep.py > logs/enhanced_morton_test.log 2>&1
```

**Success metrics**:
- Retention @ step 100: >87% (currently 82.45%)
- Throughput: >4,000 p/s (acceptable if retention improves)

### Test 3: A/B Comparison (30 min)
Run with both old and new search, compare:
- Retention curves
- Throughput
- L2 success rate

---

## Risk Mitigation

### Risk 1: Performance Degradation
**Mitigation**: Keep both functions, make enhanced search opt-in initially

### Risk 2: JAX Compilation Issues
**Mitigation**: Use only safe operations (jnp.where, fixed loops, no lax.cond)

### Risk 3: Still Insufficient Retention
**Mitigation**: If 5×5×5 insufficient, can extend to 7×7×7 (343 octants)

---

## Next Steps

1. Implement `search_5x5x5_outer_shell` function
2. Implement `search_L2_morton_neighbors_enhanced` wrapper
3. Add unit tests
4. Integrate with RK4
5. Run production test
6. Analyze results and iterate

---

**Status**: Design complete. Ready to implement.
