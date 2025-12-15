# Octree Bit Order Investigation

## Problem Summary

After implementing Phase 5 (Adaptive Octree Subdivision), accuracy tests show 10.8% success instead of >95% expected. User's insight: "This inconsistency may arise because the morton is hashing elements before octree, instead of hashing octree."

## Initial Diagnosis

**Root Cause Hypothesis**: Bit order mismatch between Morton code encoding and octree prefix extraction.

### Morton Code Structure (Verified Correct)

```
Bit  0: x0 (coarsest X bit)
Bit  1: y0 (coarsest Y bit)
Bit  2: z0 (coarsest Z bit)
Bit  3: x1
Bit  4: y1
Bit  5: z2
...
Bit 60: x20 (finest X bit)
Bit 61: y20 (finest Y bit)
Bit 62: z20 (finest Z bit)
```

**Key property**: LSB (bits 0-2) = coarse octant, MSB (bits 60-62) = fine detail

### Original Implementation (MSB Extraction)

```python
# In compute_octant_ranges():
shift = 63 - (prefix_bits + 3)  # e.g., 63 - 3 = 60 for depth 1
morton_mid = morton_sorted[mid] >> shift  # Extracts bits [62:60]

# In build_prefix_table():
shift = leaf.prefix_bits - (table_depth * 3)
prefix = leaf.morton_prefix >> shift  # Extracts from MSB

# In position_to_leaf_id_octree():
shift_amount = 63 - prefix_bits_int  # e.g., 63 - 18 = 45 for depth 6
prefix = m >> shift_amount  # Extracts bits [62:45]
```

**Result**: 10.8% accuracy (worse than 12.7% baseline)

## First Fix Attempt: LSB Extraction

Changed all three functions to extract from LSB using masks:

```python
# compute_octant_ranges():
prefix_mask = (1 << (prefix_bits + 3)) - 1
morton_mid = morton_sorted[mid] & prefix_mask  # Extract from LSB

# build_prefix_table():
prefix_mask = (1 << (table_depth * 3)) - 1
prefix = leaf.morton_prefix & prefix_mask

# position_to_leaf_id_octree():
prefix_mask = jnp.uint64((1 << prefix_bits_int) - 1)
prefix = m & prefix_mask
```

**Result**: 0.0% accuracy, 9.3M leaves (should be ~32K)
**Conclusion**: LSB extraction broke octree subdivision completely

## Why LSB Fix Failed

The octree subdivision builds prefixes using:

```python
octant_prefix = (morton_prefix << 3) | octant  # Line 77
```

This grows prefixes from the LEFT (MSB side), not from the right (LSB side). Example:

```
Depth 0: prefix = 0b0 (root)
Depth 1: prefix = 0b0 << 3 | octant = 0b000_octant (3 bits)
Depth 2: prefix = 0b000_octant << 3 | octant2 = 0b000_octant_octant2 (6 bits)
```

The prefix grows **leftward** (like: `0b101010...`), which aligns with MSB extraction, NOT LSB extraction.

## The Real Problem

There's a fundamental mismatch between:
1. **Morton encoding**: Coarse bits at LSB (bits 0-2)
2. **Octree prefix building**: Prefixes grow from MSB (left-shift)

The octree subdivision is trying to align with "octant structure" but isn't properly aligned with Morton code bit positions.

## Current Status

**Reverted changes** - back to MSB extraction (10.8% accuracy)

**Running diagnostic** - `diagnose_morton_octree_bug.py` to trace a single element through:
1. Build phase: element → Morton code → leaf assignment
2. Query phase: centroid → Morton code → prefix table lookup
3. Compare: Does the lookup find the correct leaf?

This will reveal whether:
- MSB extraction is actually correct (octree build uses MSB convention)
- LSB extraction is correct (should align with Morton coarse bits)
- BOTH are wrong (deeper issue in octree/prefix design)

## Next Steps

1. ✅ Run diagnostic to understand which extraction method should work
2. If MSB is correct: Bug is elsewhere (perhaps in how prefixes map to leaves)
3. If LSB is correct: Need to rebuild octree with LSB-aligned prefixes
4. If both wrong: Fundamental redesign needed

## Key Files

- `jaxtrace/gpu/search/morton_octree_builder.py` - Octree build with prefix-based subdivision
- `jaxtrace/gpu/search/morton_global_search.py` - GPU search with prefix table lookup
- `test_octree_accuracy.py` - Centroid-based accuracy test
- `diagnose_morton_octree_bug.py` - Single-element trace diagnostic

## Test Results

| Implementation | Centroid Success | Leaves | Status |
|----------------|------------------|--------|--------|
| Fixed-capacity (baseline) | 12.7% | N/A | ✅ |
| Octree (MSB extraction) | 10.8% | 32,168 | ❌ Worse than baseline |
| Octree (LSB extraction) | 0.0% | 9.3M | ❌ Completely broken |

---

*Investigation ongoing - diagnostic running...*
