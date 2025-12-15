# Phase 1: CSR-Style Hash Buckets Implementation Plan

## Goal
Replace padded hash bucket arrays with CSR (Compressed Sparse Row) style ranges to solve OOM.

## Current Implementation Analysis

### Current Structure (hash_bucket.py:320-346)
```python
# PADDED ARRAYS (PROBLEM):
bucket_elements = np.full((n_buckets, max_elem_per_bucket), -1, dtype=np.int32)
# Shape: (n_buckets, max_elem_per_bucket)
# Memory: n_buckets × max_elem_per_bucket × 4 bytes

# For heavy block with 949K elements:
#   n_buckets = 4,748
#   max_elem_per_bucket = 250
#   Memory = 4,748 × 250 × 4 = 4.7 MB per heavy block
#   Total (64 heavy blocks) = 301 MB
```

### Current Search (level2b_heavy.py:136-142)
```python
# Access bucket by indexing into padded array:
elem_id_primary = search_bucket_elements(
    position,
    hash_bucket_elements[bucket_id],  # (max_elem_per_bucket,) with -1 padding
    hash_bucket_counts[bucket_id],
    node_positions,
    connectivity
)
```

## Target CSR Structure

### New Data Structure
```python
# CSR-STYLE (SOLUTION):
# 1. Single flat array with ALL elements (Morton-sorted):
sorted_elements = np.array([elem_ids sorted by Morton code], dtype=np.int32)
# Shape: (total_n_elements,)
# Memory: total_n_elements × 4 bytes

# 2. CSR range array (start, end) per bucket:
bucket_ranges = np.zeros((n_buckets, 2), dtype=np.int32)
# Shape: (n_buckets, 2)
# Memory: n_buckets × 2 × 4 bytes

# For heavy block with 949K elements:
#   sorted_elements: 949K × 4 = 3.8 MB
#   bucket_ranges: 4,748 × 2 × 4 = 38 KB
#   Total per heavy block: 3.8 MB (was 4.7 MB)
#   Total (64 heavy blocks): 243 MB (was 301 MB)
#   Savings: 58 MB (19%)
```

### New HashBucketArrays Dataclass
```python
@dataclass
class HashBucketArraysCSR:
    """CSR-style hash bucket arrays (Phase 1)."""
    block_id: int
    n_buckets: int
    sorted_elements: np.ndarray  # (n_elements,) int32, Morton-sorted
    bucket_ranges: np.ndarray    # (n_buckets, 2) int32, [start, end)
    morton_bits: int
    block_bounds: np.ndarray     # (6,) float32
    bucket_neighbors_6: np.ndarray  # (n_buckets, 6) int32
```

### New Search Function
```python
@jax.jit
def search_bucket_elements_csr(
    position: jax.Array,
    sorted_elements: jax.Array,
    bucket_start: int,
    bucket_end: int,
    node_positions: jax.Array,
    connectivity: jax.Array
) -> int:
    """Search elements in CSR range [start, end)."""
    # Extract bucket elements (no padding!)
    bucket_elements = sorted_elements[bucket_start:bucket_end]

    # Search (same logic as before, but no -1 padding to handle)
    def check_element(elem_id):
        node_ids = connectivity[elem_id]
        tet_nodes = node_positions[node_ids]
        return point_in_tet_jax(position, tet_nodes)

    inside_flags = jax.vmap(check_element)(bucket_elements)

    found_indices = jnp.where(inside_flags, jnp.arange(len(bucket_elements)), len(bucket_elements))
    first_match_idx = jnp.min(found_indices)

    return jnp.where(first_match_idx < len(bucket_elements), bucket_elements[first_match_idx], -1)
```

## Implementation Steps

### Step 1: Create CSR Builder Function
**File**: `jaxtrace/gpu/search/hash_bucket_csr.py` (NEW)
**Task**: Implement `build_hash_bucket_arrays_csr()` that:
1. Computes Morton codes (reuse existing)
2. Sorts elements by Morton code
3. Builds CSR ranges per bucket

**Validation**: Test with synthetic data (100K elements)

### Step 2: Create CSR Search Function
**File**: `jaxtrace/gpu/search/level2b_heavy_csr.py` (NEW)
**Task**: Implement `search_level2b_hash_bucket_csr()` that:
1. Uses CSR ranges instead of padded arrays
2. Maintains same search logic (primary + 6 neighbors)
3. Compatible with existing L2b interface

**Validation**: Test that it finds same elements as padded version

### Step 3: Update Initial Assignment
**File**: `jaxtrace/gpu/search/initial_assignment.py` (MODIFY)
**Task**: Replace L2b padded with L2b CSR
1. Import CSR builder/search
2. Switch construction to CSR
3. Switch search calls to CSR

**Validation**: Run test_octree_vs_blockwise_initialization.py

### Step 4: Verify OOM Fix
**Task**: Re-run OOM test case
**Expected**: No OOM with same test configuration

## Compatibility Checks

### JAX JIT Compatibility
- ✓ CSR ranges are static-shape arrays: `(n_buckets, 2)`
- ✓ Slicing `sorted_elements[start:end]` is JIT-compatible
- ⚠️ **Dynamic slice sizes**: JAX requires special handling

**Solution**: Use `jax.lax.dynamic_slice` with `size` parameter:
```python
# NOT JIT-compatible:
bucket_elems = sorted_elements[start:end]  # Variable size!

# JIT-compatible:
max_bucket_size = 300  # Fixed upper bound
bucket_elems = jax.lax.dynamic_slice(
    sorted_elements,
    (start,),
    (min(end - start, max_bucket_size),)
)
# Pad if needed
```

### Nested vmap/scan Compatibility
- ✓ No new nesting introduced
- ✓ Same vmap pattern as current L2b
- ✓ No scan required

### Memory Savings
- Current: 301 MB (64 heavy blocks × 4.7 MB)
- CSR: 243 MB (19% reduction)
- **Further savings when combined with Phase 2** (per-block octrees reduce duplication)

## Testing Strategy

### Unit Tests
1. **Test Morton sorting preserves spatial locality**
   - Elements with similar positions have similar Morton codes
   - Verify CSR ranges are contiguous

2. **Test CSR search finds correct elements**
   - Compare against padded version
   - Test boundary cases (empty buckets, full buckets)

3. **Test memory usage**
   - Measure before/after with heavy block
   - Verify 19% reduction

### Integration Tests
1. **Test with test_octree_vs_blockwise_initialization.py**
   - Should not OOM with (8,8,4) grid
   - Should find same elements as padded version

2. **Test with production_tracking_scenario2.py**
   - Verify no regression in accuracy or performance
   - Measure memory usage

## Risks and Mitigations

### Risk 1: Dynamic Slice Size in JIT
**Problem**: `sorted_elements[start:end]` has variable size
**Mitigation**: Use bounded slice + masking
```python
max_bucket_size = 300  # 95th percentile × 1.5
actual_size = end - start
bucket_elems_padded = jax.lax.dynamic_slice(sorted_elements, (start,), (max_bucket_size,))
valid_mask = jnp.arange(max_bucket_size) < actual_size
```

### Risk 2: Neighbor Bucket CSR Access
**Problem**: Accessing 6 neighbor buckets requires 6 CSR lookups
**Mitigation**: Vectorize with vmap (same as current)

### Risk 3: Breaking Existing Code
**Problem**: Other code may depend on padded arrays
**Mitigation**:
- Keep padded version as `hash_bucket.py`
- Add CSR version as `hash_bucket_csr.py`
- Switch initial_assignment.py to CSR first
- Deprecate padded version in Phase 2

## Success Criteria

✓ **No OOM**: test_octree_vs_blockwise_initialization.py runs without OOM
✓ **Memory reduction**: 19% less memory for hash buckets
✓ **No accuracy loss**: Same search results as padded version
✓ **No performance loss**: Same or better throughput
✓ **JAX compatible**: All functions JIT-compile successfully

## Files to Create/Modify

### NEW Files
1. `jaxtrace/gpu/search/hash_bucket_csr.py` - CSR builder
2. `jaxtrace/gpu/search/level2b_heavy_csr.py` - CSR search
3. `test_hash_bucket_csr.py` - Unit tests

### MODIFIED Files
1. `jaxtrace/gpu/search/initial_assignment.py` - Switch to CSR

### UNCHANGED Files
1. `hash_bucket.py` - Keep for reference/fallback
2. `level2b_heavy.py` - Keep for reference/fallback

## Next Steps After Phase 1

Once Phase 1 is validated:
- **Phase 2**: Add per-block flat octree (depth 6-8)
  - Builds on CSR structure
  - Further reduces memory (90% total reduction)
  - 10× L2b speedup

- **Phase 3**: Vectorize initial_search_batch
  - Replace Python loops with vmap
  - 100-500× initial assignment speedup
