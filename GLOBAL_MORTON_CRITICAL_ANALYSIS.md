# Global Morton L2 Search - Critical Analysis and Next Steps

**Date**: 2025-12-13
**Status**: ⚠️ PHASE 1 INCOMPLETE - Need True Octree Leaves

---

## Executive Summary

The current global Morton implementation achieves **only 12.7% success rate** on centroid-based tests, revealing fundamental issues with the leaf segmentation approach. The problem is NOT with:
- Morton encoding/decoding (matches HOT spec exactly)
- Search radius (tested 2 and 4, no improvement)
- Point-in-tet tests (100% correctness when found)

The problem IS with:
- **Fixed-capacity leaf segmentation** instead of true octree leaves
- **Linear Morton approximation** instead of prefix-based mapping

---

## Test Results Analysis

### Accuracy Test Results (Radius=2 vs Radius=4)

| Metric | Radius=2 | Radius=4 | Change |
|--------|----------|----------|--------|
| **Centroid Success** | 12.88% | 12.75% | -0.13% |
| **Perturbed Success** | 16.36% | 16.54% | +0.18% |
| **Centroid Correctness** | 100% | 100% | - |
| **Perturbed Correctness** | 100% | 100% | - |
| **Centroid Throughput** | 5,707 p/s | 28,443 p/s | 5× faster |
| **Perturbed Throughput** | 15,379 p/s | 41,574 p/s | 2.7× faster |

### Critical Observations

1. **Radius Has NO Effect on Success Rate**
   - Increasing from radius=2 to radius=4 changed success by <0.2%
   - This proves the issue is NOT about searching nearby leaves
   - The problem is that the CORRECT leaf is far away in the Morton curve

2. **100% Correctness When Found**
   - Every particle found is in the correct element (point-in-tet verified)
   - This proves Morton encoding/decoding is correct
   - This proves the search logic itself works

3. **Perturbed Faster Than Centroid**
   - Perturbed positions are 2.7-5× faster
   - Perturbed has higher success rate (16.5% vs 12.7%)
   - This is STRANGE and suggests caching/JIT effects

4. **Low Overall Success Rate**
   - Only ~13% of element centroids found in their own Morton leaves
   - This matches the 81% initial assignment in production (many particles outside mesh + 13% L2 success)

---

## Root Cause: Fixed-Capacity Leaves vs True Octree

### Current Implementation (WRONG)

```python
# Phase 1: Fixed-capacity segmentation (CURRENT)
leaf_start[i] = i * C                          # Leaf i starts at i*256
leaf_length[i] = min(C, n_elements - i*C)      # Up to 256 elements per leaf

# Leaf boundaries are ARBITRARY - just cut every 256 elements in Morton order
# Leaf 0: elements [0, 255] in Morton order
# Leaf 1: elements [256, 511] in Morton order
# ...
```

**Problem**: An element's centroid might be in octant A, but its Morton leaf contains elements from octants A, B, C because we cut arbitrarily every 256 elements.

**Example**:
```
Element 1000: centroid = (0.5, 0.5, 0.5)
Morton code = 0x1A2B3C (binary search finds it in leaf 3)

But when we query position (0.5, 0.5, 0.5):
- Compute Morton code: 0x1A2B3C
- Map to leaf: (0x1A2B3C - morton_min) / (morton_max - morton_min) * n_leaves
- Linear approximation puts it in leaf 7 (WRONG!)

Why? Because Morton distribution is NON-UNIFORM:
- Mesh is denser in some regions (small elements → high Morton density)
- Mesh is sparser in other regions (large elements → low Morton density)
- Linear approximation: t = (m - m_min) / (m_max - m_min) assumes uniform distribution
- Reality: 27% mean deviation from uniform (from validation tests)
```

### Correct HOT Design (NEEDED)

```python
# True octree-aligned leaves
# Each leaf = contiguous Morton prefix (octant)

# Example: Level-1 octree (8 leaves)
Leaf 0: Morton codes [0b000xxxxxx, 0b000yyyyyy] - octant (0,0,0)
Leaf 1: Morton codes [0b001xxxxxx, 0b001yyyyyy] - octant (0,0,1)
Leaf 2: Morton codes [0b010xxxxxx, 0b010yyyyyy] - octant (0,1,0)
...
Leaf 7: Morton codes [0b111xxxxxx, 0b111yyyyyy] - octant (1,1,1)

# Position → Leaf mapping:
def position_to_leaf_octree(pos):
    m = morton_encode(pos)
    # Extract top B bits (e.g., B=12 for 4096 leaves)
    prefix = m >> (63 - B)
    # Lookup in prefix table
    leaf_id = prefix_to_leaf[prefix]
    return leaf_id
```

**Advantages**:
1. **Geometric alignment**: Each leaf corresponds to a spatial octant
2. **Exact mapping**: Prefix table gives O(1) position→leaf
3. **Spatial locality**: Elements in same leaf are spatially close
4. **No approximation**: No linear interpolation needed

---

## Morton Encoding Verification

Comparing current implementation against HOT specs:

### HOT Specification (from HOT_Morton_leafwise_plan_GPT-think.md)

```python
# Lines 40-53 from spec
def morton_encode_point(p, bbox_min, bbox_max, L):
    scale = (2**L - 1) / (bbox_max - bbox_min)
    u = np.floor((p - bbox_min) * scale).astype(np.uint64)  # (3,)
    return interleave_bits_3d(u[0], u[1], u[2])             # uint64

# Lines 47-53 from spec
m_e = sum_{i=0}^{L-1} (x_i 2^{3i} + y_i 2^{3i+1} + z_i 2^{3i+2})
```

### Current Implementation (jaxtrace/gpu/search/morton_global_search.py)

```python
# Lines 119-136 (JAX version)
def morton_encode_position_jax(
    pos: jax.Array,
    bbox_min: jax.Array,
    bbox_max: jax.Array,
    max_depth: int
) -> jnp.uint64:
    normalized = (pos - bbox_min) / (bbox_max - bbox_min)
    normalized = jnp.clip(normalized, 0.0, 1.0)
    grid_max = (2 ** max_depth) - 1
    u = jnp.floor(normalized * grid_max).astype(jnp.uint32)
    return interleave_bits_3d_jax(u[0], u[1], u[2])

# Lines 48-68 (bit interleaving)
def interleave_bits_3d_jax(x: jnp.uint32, y: jnp.uint32, z: jnp.uint32) -> jnp.uint64:
    morton = jnp.uint64(0)
    for i in range(21):
        morton |= ((x >> i) & 1) << (3*i + 0)
        morton |= ((y >> i) & 1) << (3*i + 1)
        morton |= ((z >> i) & 1) << (3*i + 2)
    return morton
```

✅ **VERIFIED**: Encoding matches HOT spec exactly (same formula, same bit order)

---

## Search Radius Verification

### Current Implementation (jaxtrace/gpu/search/morton_global_search.py)

```python
# Lines 328-407: search_L2_global_morton_single
def search_L2_global_morton_single(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMorton,
    search_radius: jnp.int32 = jnp.int32(1)
) -> jnp.int32:
    # Map position to leaf (binary search for accurate mapping)
    center_leaf_id = position_to_leaf_id(pos, mesh_gpu)

    # Search center leaf first
    elem_id = search_in_leaf_global(pos, center_leaf_id, mesh_gpu)
    found = elem_id >= 0

    # Search neighboring leaves if not found
    def search_neighbor(i, state):
        elem_id, found = state
        active = ~found

        # Compute neighbor offset: maps i ∈ [0, 2*radius] to offsets [-radius, -1, +1, +radius]
        offset = jnp.where(i < search_radius,
                          i - search_radius,      # [-radius, ..., -1]
                          i - search_radius + 1)  # [+1, ..., +radius]

        neighbor_leaf_id = center_leaf_id + offset
        neighbor_leaf_id = jnp.clip(neighbor_leaf_id, 0, mesh_gpu.n_leaves - 1)

        elem_neighbor = jnp.where(
            active,
            search_in_leaf_global(pos, neighbor_leaf_id, mesh_gpu),
            jnp.int32(-1)
        )

        improve = (elem_neighbor >= 0) & active
        elem_id = jnp.where(improve, elem_neighbor, elem_id)
        found = found | improve

        return (elem_id, found)

    # Search 2*search_radius neighbors
    final_elem_id, final_found = lax.fori_loop(
        0,
        2 * search_radius,
        search_neighbor,
        (elem_id, found)
    )

    return final_elem_id
```

✅ **VERIFIED**: Radius logic is correct
- Searches center leaf + radius neighbors on each side
- Radius=2: searches leaves [center-2, center-1, center, center+1, center+2] = 5 leaves total
- Radius=4: searches leaves [center-4, ..., center, ..., center+4] = 9 leaves total

❌ **BUT**: Radius doesn't help because:
- Linear mapping `position_to_leaf_id()` is WRONG for non-uniform distributions
- The correct leaf might be 100+ leaves away in Morton order
- Searching ±4 leaves only covers a tiny fraction of the space

---

## Why Perturbed Is Faster

### Hypothesis: JIT Compilation Effects

```python
# Test 1: Centroid positions (first call)
- JIT compiles search_L2_global_morton_single
- Compilation overhead: ~1.1 seconds (difference in total time)
- Result: 5,707 p/s (slow due to compilation)

# Test 2: Perturbed positions (second call)
- JIT already compiled from Test 1
- No compilation overhead
- Result: 41,574 p/s (7× faster, same pattern as compiled vs uncompiled)
```

### Hypothesis: Position Distribution

```python
# Centroid positions
- Exactly at element centers
- Might trigger worst-case search patterns
- Lower cache hit rate?

# Perturbed positions
- Slightly off-center
- Random perturbations might distribute better across leaves
- Better cache patterns?
```

This effect is NOT concerning - it's a JIT artifact. Both show the same fundamental problem: low success rate.

---

## Next Steps: Implement True Octree Leaves

### Phase 1: Adaptive Octree Construction (CPU)

**Goal**: Replace fixed-capacity leaves with geometric octree-aligned leaves

**Algorithm**: Top-down octree subdivision with capacity constraint

```python
def build_adaptive_octree_leaves(
    morton_sorted: np.ndarray,      # (n_elements,) uint64 - sorted Morton codes
    elem_ids_sorted: np.ndarray,    # (n_elements,) int32 - sorted element IDs
    leaf_capacity: int = 256,       # Max elements per leaf
    max_depth: int = 21             # Max octree depth
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Build adaptive octree with capacity-constrained leaves.

    Each leaf:
    - Is aligned with spatial octant (Morton prefix)
    - Contains ≤ leaf_capacity elements
    - Has contiguous range in morton_sorted array

    Returns:
        leaf_start: (n_leaves,) int32 - start index in elem_ids_sorted
        leaf_length: (n_leaves,) int32 - element count
        octree_info: dict with prefix_to_leaf mapping
    """

    # Recursive subdivision
    def subdivide_node(
        start_idx: int,
        end_idx: int,
        morton_prefix: int,
        prefix_bits: int,
        depth: int
    ) -> List[Leaf]:

        n_elements = end_idx - start_idx

        # Base case 1: Leaf is small enough
        if n_elements <= leaf_capacity:
            return [Leaf(start_idx, n_elements, morton_prefix, prefix_bits)]

        # Base case 2: Maximum depth reached
        if depth >= max_depth:
            # Split into multiple fixed-capacity leaves
            leaves = []
            for i in range(0, n_elements, leaf_capacity):
                leaf_start = start_idx + i
                leaf_length = min(leaf_capacity, n_elements - i)
                leaves.append(Leaf(leaf_start, leaf_length, morton_prefix, prefix_bits))
            return leaves

        # Recursive case: Subdivide into 8 octants
        # Extract next 3 bits of Morton code (one octree level = 3 bits)
        next_bit_pos = 63 - (prefix_bits + 3)

        # Find split points for 8 octants
        octant_ranges = []
        for octant in range(8):
            # Compute Morton prefix for this octant
            octant_prefix = (morton_prefix << 3) | octant
            octant_morton_min = octant_prefix << next_bit_pos
            octant_morton_max = ((octant_prefix + 1) << next_bit_pos) - 1

            # Binary search for range
            range_start = np.searchsorted(morton_sorted[start_idx:end_idx],
                                         octant_morton_min, side='left') + start_idx
            range_end = np.searchsorted(morton_sorted[start_idx:end_idx],
                                       octant_morton_max, side='right') + start_idx

            if range_end > range_start:
                octant_ranges.append((range_start, range_end, octant_prefix))

        # Recursively subdivide non-empty octants
        leaves = []
        for range_start, range_end, octant_prefix in octant_ranges:
            leaves.extend(subdivide_node(
                range_start, range_end,
                octant_prefix,
                prefix_bits + 3,
                depth + 1
            ))

        return leaves

    # Start subdivision from root
    leaves = subdivide_node(
        start_idx=0,
        end_idx=len(morton_sorted),
        morton_prefix=0,
        prefix_bits=0,
        depth=0
    )

    # Convert to arrays
    n_leaves = len(leaves)
    leaf_start = np.array([leaf.start for leaf in leaves], dtype=np.int32)
    leaf_length = np.array([leaf.length for leaf in leaves], dtype=np.int32)

    # Build prefix→leaf mapping for O(1) lookup
    max_prefix_bits = max(leaf.prefix_bits for leaf in leaves)
    prefix_to_leaf = build_prefix_table(leaves, max_prefix_bits)

    octree_info = {
        'n_leaves': n_leaves,
        'max_depth': max(leaf.depth for leaf in leaves),
        'min_elements': min(leaf.length for leaf in leaves),
        'max_elements': max(leaf.length for leaf in leaves),
        'avg_elements': np.mean(leaf_length),
        'prefix_to_leaf': prefix_to_leaf,
        'max_prefix_bits': max_prefix_bits
    }

    return leaf_start, leaf_length, octree_info
```

### Phase 2: Prefix Table for O(1) Lookup (CPU)

```python
def build_prefix_table(
    leaves: List[Leaf],
    max_prefix_bits: int
) -> np.ndarray:
    """
    Build lookup table: Morton prefix → leaf ID.

    For efficient GPU lookup, we use a flat array indexed by
    the top B bits of the Morton code.

    Example: If max_prefix_bits=12, table has 2^12=4096 entries.

    Returns:
        prefix_to_leaf: (2^max_prefix_bits,) int32
    """
    table_size = 2 ** max_prefix_bits
    prefix_to_leaf = np.full(table_size, -1, dtype=np.int32)

    for leaf_id, leaf in enumerate(leaves):
        # Each leaf covers a range of prefixes
        # If leaf has prefix 0b101 with 3 bits, it covers all prefixes 0b101xxxxxxxxx

        prefix = leaf.morton_prefix
        prefix_bits = leaf.prefix_bits

        # Shift to align with table indexing
        prefix_aligned = prefix << (max_prefix_bits - prefix_bits)

        # Fill all table entries for this leaf
        prefix_range = 2 ** (max_prefix_bits - prefix_bits)
        for i in range(prefix_range):
            table_idx = prefix_aligned + i
            prefix_to_leaf[table_idx] = leaf_id

    return prefix_to_leaf
```

### Phase 3: GPU Position→Leaf Mapping (JAX)

```python
def position_to_leaf_id_octree(
    pos: jax.Array,
    mesh_gpu_morton: MeshGPUGlobalMorton
) -> jnp.int32:
    """
    Map position to leaf ID using octree prefix table.

    O(1) lookup - no approximation.

    Args:
        pos: (3,) float32
        mesh_gpu_morton: GPU structure with prefix_to_leaf table

    Returns:
        leaf_id: int32
    """
    # Compute Morton code
    m = morton_encode_position_jax(
        pos,
        mesh_gpu_morton.bbox_min,
        mesh_gpu_morton.bbox_max,
        mesh_gpu_morton.max_depth
    )

    # Extract top B bits as prefix
    prefix = m >> (63 - mesh_gpu_morton.max_prefix_bits)

    # Lookup in table (O(1))
    leaf_id = mesh_gpu_morton.prefix_to_leaf[prefix]

    # Handle invalid (outside mesh)
    valid = leaf_id >= 0
    return jnp.where(valid, leaf_id, jnp.int32(0))
```

---

## Implementation Plan

### Priority 1: Fix Leaf Segmentation

1. **Create `morton_octree_builder.py`**
   - Implement adaptive octree subdivision
   - Build prefix table for O(1) lookup
   - Target: >95% success rate on centroid test

2. **Update `morton_global_search.py`**
   - Replace `position_to_leaf_id_linear()` with `position_to_leaf_id_octree()`
   - Upload prefix table to GPU
   - Add prefix_to_leaf array to MeshGPUGlobalMorton

3. **Validation**
   - Run centroid accuracy test
   - Expect >95% success (particles at centroids should find their elements)
   - Verify radius=1 is sufficient (octree leaves are spatially coherent)

### Priority 2: Binary Search Fallback

If octree leaves alone don't reach >95%, add binary search:

```python
def position_to_leaf_id_binary_search(
    pos: jax.Array,
    mesh_gpu_morton: MeshGPUGlobalMorton
) -> jnp.int32:
    """
    Binary search in Morton-sorted array to find exact leaf.

    More accurate than linear approximation, works for any distribution.
    """
    m = morton_encode_position_jax(pos, ...)

    # Binary search for insertion point
    idx = binary_search_morton(m, mesh_gpu_morton.morton_sorted)

    # Map index to leaf ID
    leaf_id = idx // mesh_gpu_morton.leaf_capacity

    return jnp.clip(leaf_id, 0, mesh_gpu_morton.n_leaves - 1)
```

Note: We already have binary search implemented (`morton_binary_search_leaf` in morton_global_search.py:142-198), but it's used with fixed-capacity leaves. With octree leaves, we need to:
1. Store leaf boundaries instead of fixed capacity
2. Use binary search to find which leaf contains the Morton code

### Priority 3: Multi-Leaf Search

If single-leaf search fails, search nearby leaves in Morton order:

```python
def search_L2_global_morton_multi_leaf(
    pos: jax.Array,
    mesh_gpu_morton: MeshGPUGlobalMorton,
    max_leaves: int = 3
) -> jnp.int32:
    """
    Search multiple leaves in Morton-sorted order.

    Args:
        max_leaves: Maximum number of leaves to check
    """
    center_leaf = position_to_leaf_id_octree(pos, mesh_gpu_morton)

    # Search center leaf + neighbors in Morton order
    # (neighbors in Morton space = spatially nearby)
    for offset in range(-max_leaves, max_leaves + 1):
        leaf_id = jnp.clip(center_leaf + offset, 0, mesh_gpu_morton.n_leaves - 1)
        elem_id = search_in_leaf_global(pos, leaf_id, mesh_gpu_morton)
        if elem_id >= 0:
            return elem_id

    return jnp.int32(-1)
```

---

## Expected Outcomes

### After Octree Leaves Implementation

| Metric | Current (Fixed-Capacity) | Expected (Octree) | Improvement |
|--------|-------------------------|-------------------|-------------|
| **Centroid Success** | 12.7% | >95% | 7.5× |
| **Perturbed Success** | 16.5% | >80% | 5× |
| **Initial Assignment** | 81% | >95% | +14% |
| **Retention (2.5K steps)** | ~80% | >95% | +15% |
| **Search Radius Needed** | 4+ | 1-2 | Halved |
| **Leaf Coverage** | Arbitrary | Geometric | Better locality |

### Why Octree Will Work

1. **Spatial Coherence**: Each leaf = spatial octant, so position→leaf mapping is geometric, not approximated
2. **O(1) Lookup**: Prefix table gives exact leaf in constant time
3. **No Distribution Assumptions**: Works regardless of element density distribution
4. **Smaller Search Radius**: Spatial neighbors in same/nearby leaves
5. **HOT-Compliant**: Matches HOT paper's octree design

---

## Summary

### Current Status

✅ **Working Correctly**:
- Morton encoding/decoding (matches HOT spec)
- Point-in-tet tests (100% correctness)
- Search radius logic (verified in code)
- RK4 integration (no crashes, proper execution)

❌ **Broken**:
- Leaf segmentation (fixed-capacity instead of octree)
- Position→leaf mapping (linear approximation fails)

### Root Cause

The implementation is at **Phase 1** of the HOT plan but uses a simplified "fixed-capacity" approach instead of true octree leaves. This was intentional as a first step, but the test results show it's insufficient.

### Next Action

Implement Phase 5 from HOT_MORTON_REVISED_PLAN.md:
1. Adaptive octree subdivision (CPU)
2. Prefix table for O(1) lookup (CPU)
3. Octree-based position→leaf mapping (GPU)

This will complete the full HOT Morton implementation and achieve >95% success rates.
