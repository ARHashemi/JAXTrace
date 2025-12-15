# Morton Encoding Bug - ROOT CAUSE FOUND

## Executive Summary

**Diagnostic Result**: Both MSB and LSB extraction methods fail because the Morton codes themselves are WRONG.

The octree builder's vectorized Morton encoding produces **incorrect codes with garbage in the high 32 bits**.

## The Smoking Gun

```
Test Element ID: 1003299
Centroid: (-0.001777, 0.000499, -0.004492)

Morton code (BUILD):  0x62498A64BCA466A2  ← High 32 bits have junk!
Morton code (QUERY):  0x00000000BCA466A2  ← Only low 32 bits

Expected leaf: 16115
MSB lookup result: 0
LSB lookup result: 6

Both lookups FAIL because the Morton codes don't match!
```

## Root Cause

The vectorized bit interleaving in `morton_octree_builder.py` lines 417-420:

```python
# Interleave bits (vectorized)
for i in range(21):
    morton_codes |= ((u[:, 0] >> i) & 1).astype(np.uint64) << (3*i + 0)
    morton_codes |= ((u[:, 1] >> i) & 1).astype(np.uint64) << (3*i + 1)
    morton_codes |= ((u[:, 2] >> i) & 1).astype(np.uint64) << (3*i + 2)
```

**Problem**: When `i >= 11`, the shift amount `3*i` exceeds 32 bits:
- `i=11`: shift by 33, 34, 35
- `i=20`: shift by 60, 61, 62

The issue is that `u` is `uint32`, and `(u[:, 0] >> i)` is computed as uint32 before being cast to uint64. For large `i`, this creates undefined behavior or incorrect results.

Additionally, the expression `((u[:, 0] >> i) & 1)` extracts a single bit, which is fine, but when multiple operations compound, numerical precision issues may occur.

## The Fix

Replace the vectorized bit interleaving with a proper implementation that:

1. Casts `u` to `uint64` BEFORE any bit operations
2. Ensures all intermediate values are uint64
3. Or uses a tested Morton encoding library/function

### Option 1: Cast to uint64 first

```python
# Cast to uint64 BEFORE bit operations
u = u.astype(np.uint64)

# Interleave bits
for i in range(21):
    morton_codes |= ((u[:, 0] >> i) & 1) << (3*i + 0)
    morton_codes |= ((u[:, 1] >> i) & 1) << (3*i + 1)
    morton_codes |= ((u[:, 2] >> i) & 1) << (3*i + 2)
```

### Option 2: Use proven Morton encoding

Check if there's already a working Morton encoder elsewhere in the codebase (e.g., in `morton_global_search.py` for GPU) and use the same logic.

## Why This Explains Everything

1. **10.8% accuracy**: Some elements accidentally got correct low 32 bits, prefix table lookups sometimes work by chance
2. **Octree structure validates**: The octree subdivision ITSELF uses the broken Morton codes consistently, so the tree structure is self-consistent (32K leaves, good coherence)
3. **Query lookups fail**: Queries compute DIFFERENT Morton codes (correct ones), so prefix table lookups find wrong leaves

## Impact

This bug affects:
- ✅ **NOT** the octree subdivision logic (uses broken codes consistently)
- ✅ **NOT** the prefix table construction (uses broken codes consistently)
- ❌ **YES** the query-time Morton encoding (computes correct codes that don't match)
- ❌ **YES** all accuracy tests (lookups fail due to mismatch)

## Next Steps

1. Fix the vectorized Morton encoding in `morton_octree_builder.py` line 414
2. Ensure GPU Morton encoding in `morton_global_search.py` uses THE SAME logic
3. Re-run octree builder test to verify same leaf count (~32K)
4. Re-run accuracy test - should jump from 10.8% → >95%

## Files to Fix

1. `jaxtrace/gpu/search/morton_octree_builder.py`:415 - Cast `u` to uint64 before bit ops
2. Verify `jaxtrace/gpu/search/morton_global_search.py` uses matching encoding

---

**Status**: Ready to fix - single line change
**Expected result**: 10.8% → >95% accuracy
**Performance impact**: None (only fixes correctness)
