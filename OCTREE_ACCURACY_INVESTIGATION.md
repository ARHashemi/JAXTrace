# Octree Accuracy Investigation - Unexpected Results

## Test Results Summary

**Expected**: 12.7% → >95% improvement  
**Actual**: 12.7% → 10.8% regression

### Comparison Table

| Metric | Fixed-Capacity (r=4) | Adaptive Octree (r=1) | Expected | Status |
|--------|----------------------|------------------------|----------|--------|
| Centroid Success | 12.75% | 10.80% | >95% | ❌ WORSE |
| Perturbed Success | 16.54% | 10.50% | >80% | ❌ WORSE |
| Correctness | 100.0% | 100.0% | 100% | ✅ OK |
| Search Radius | 4 | 1 | 0-1 | ✅ OK |
| Throughput | 28K/s | 45K/s | Similar | ✅ OK |

### Key Observations

1. **Worse Performance**: Octree is WORSE than fixed-capacity
2. **100% Correctness**: When element is found, it's always correct (point-in-tet passes)
3. **Low Success Rate**: Only finding 10.8% of centroids in their own octree leaf
4. **Structure Validation Passed**: Octree builder test showed correct structure

## Possible Causes

### 1. Prefix Table Mapping Bug ⚠️ LIKELY

**Symptom**: Only 10.8% of centroids found (should be >95%)

**Hypothesis**: Prefix extraction or table lookup is wrong

**Evidence**:
- Octree builder validation passed (structure is correct)
- Leaf coherence is good (2.55 ratio)
- 100% correctness when found (point-in-tet works)
- But wrong leaf is being looked up

**Check**:
```python
# In position_to_leaf_id_octree():
table_depth_int = int(mesh_gpu.table_depth)  # table_depth=6
prefix_bits_int = table_depth_int * 3        # 18 bits
shift_amount = 63 - prefix_bits_int          # 63 - 18 = 45

prefix = lax.shift_right_logical(m, jnp.uint64(shift_amount))
```

**Potential Issues**:
- Shift amount calculation (63 - 18 = 45)
- Morton code bit order (MSB vs LSB)
- Prefix table indexing

### 2. Morton Code Bit Order ⚠️ POSSIBLE

**Hypothesis**: Prefix bits might be at LSB instead of MSB

**Evidence**:
- If Morton codes are stored with prefix at LSB, shifting right by 45 would extract wrong bits
- Need to verify: are prefixes at MSB or LSB of 63-bit Morton code?

**Check HOT spec**: Verify bit ordering in morton_encode_position_jax

### 3. Prefix Table Construction Bug ⚠️ POSSIBLE

**Hypothesis**: build_prefix_table() might be creating wrong mappings

**Evidence**:
- 99.9% coverage (looks correct)
- But maybe mapping wrong prefixes to wrong leaves

**Check**:
```python
# In build_prefix_table():
for leaf_id, leaf in enumerate(leaves):
    leaf_depth = leaf.prefix_bits // 3
    
    if leaf_depth >= table_depth:
        # Extract table_depth-bit prefix from leaf prefix
        shift = leaf.prefix_bits - (table_depth * 3)
        prefix = leaf.morton_prefix >> shift
        prefix_table[prefix] = leaf_id
```

**Potential Issues**:
- Shift calculation might be wrong
- Morton prefix format mismatch

### 4. Table Depth Too Shallow ⚠️ UNLIKELY

**Hypothesis**: table_depth=6 (18 bits) might not be enough precision

**Evidence**:
- Against: 99.9% coverage suggests table is dense enough
- Against: Fixed-capacity uses same Morton codes and gets 12.7%
- Unlikely to be the issue

## Debug Strategy

### Step 1: Add Diagnostic Logging

Add debug output to position_to_leaf_id_octree():
```python
def position_to_leaf_id_octree(pos, mesh_gpu):
    m = morton_encode_position_jax(...)
    
    # DEBUG: Print first few lookups
    if DEBUG:
        print(f"Position: {pos}")
        print(f"Morton code: 0x{m:016X}")
        print(f"Table depth: {table_depth_int}")
        print(f"Shift amount: {shift_amount}")
        print(f"Prefix: {prefix} (0x{prefix:X})")
        print(f"Leaf ID: {leaf_id}")
    
    return leaf_id
```

### Step 2: Verify Prefix Extraction

Create unit test:
```python
# Test that centroid of element E maps to leaf containing element E
for elem_id in sample_elements:
    centroid = compute_centroid(elem_id)
    morton = morton_encode_position(centroid)
    prefix = extract_prefix(morton, table_depth)
    leaf_id = prefix_table[prefix]
    
    # Check: elem_id should be in this leaf
    assert elem_id in elem_ids_sorted[leaf_start[leaf_id]:leaf_start[leaf_id]+leaf_length[leaf_id]]
```

### Step 3: Compare CPU vs GPU Lookup

Verify that Python int() conversion doesn't break the lookup:
```python
# CPU version
table_depth_int = int(mesh_gpu.table_depth)

# vs JAX version
table_depth_jax = jnp.int32(mesh_gpu.table_depth)
```

### Step 4: Verify Against Fixed-Capacity Baseline

Run same centroids through both:
- Fixed-capacity (12.7% success)
- Adaptive octree (10.8% success)

Check if they're finding DIFFERENT elements or SAME elements.

## Root Cause Hypothesis

**Most Likely**: Prefix extraction is wrong due to bit ordering assumption.

**Reasoning**:
1. Structure validation passed → octree build is correct
2. Coherence is good → leaves are spatially aligned
3. Correctness is 100% → point-in-tet works
4. Success rate is WORSE → wrong leaf being looked up
5. Only explanation: prefix→leaf_id mapping is broken

**Fix Direction**:
- Verify Morton code bit layout (MSB vs LSB for prefixes)
- Check prefix extraction formula
- Add assertions to validate prefix→leaf correctness during build

## Next Steps

1. Add diagnostic logging to position_to_leaf_id_octree()
2. Run test with single element and trace through lookup
3. Verify prefix extraction matches build_prefix_table() expectations
4. Fix bit ordering / shift calculation
5. Re-run accuracy test

