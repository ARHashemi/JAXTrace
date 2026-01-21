# Morton Neighbor Critical Bug Fix

**Date**: 2025-12-25
**Status**: ✅ CRITICAL BUG FIXED

---

## The Bug That Caused 67% Particle Loss

### Root Cause

**File**: `jaxtrace/gpu/search/morton_global_search.py` line 661 (BEFORE fix)

```python
# WRONG: Shifted Morton code to create small integer
center_prefix = lax.shift_right_logical(morton_query, jnp.uint64(shift_amount))
# Result: center_prefix = 0x00000000001DA77 (small integer)

# Passed to neighbor generation
neighbor_prefixes = get_26_neighbor_prefixes_jax(
    center_prefix,  # BUG: This is a small integer, not left-aligned!
    table_depth_int,
    max_coord
)
```

### What Went Wrong

1. **Morton codes are left-aligned** in uint64:
   ```
   Position (61, 35, 27) at depth 7:
   Morton code = 0x0769DC0000000000 (left-aligned)
   ```

2. **Line 661 shifted it right**, extracting just the index value:
   ```python
   center_prefix = morton_query >> 42
   # Result: 0x1DA77 (small integer, right-aligned)
   ```

3. **`decode_morton_prefix_jax` expects LEFT-aligned** codes:
   ```python
   def decode_morton_prefix_jax(prefix, depth):
       # Extracts bits from positions 60-62, 57-59, 54-56, ...
       # Expects prefix like: 0x0769DC0000000000
       # Got instead:          0x00000000001DA77
       # Decoded to: (0, 0, 0) ❌
   ```

4. **All particles searched around (0,0,0)**:
   ```
   Correct neighbors: (60,34,26), (60,34,27), (61,35,27), ...
   Wrong neighbors:   (0,0,0), (0,0,1), (0,1,0), ...  ← Mesh corner!
   ```

5. **Result**: 67% of particles searching wrong region → LOST

---

## The Fix

**File**: `jaxtrace/gpu/search/morton_global_search.py` line 660

**BEFORE** (broken):
```python
# Extract prefix at table depth
table_depth_int = int(mesh_gpu.table_depth)
prefix_bits = table_depth_int * 3
shift_amount = 63 - prefix_bits
center_prefix = lax.shift_right_logical(morton_query, jnp.uint64(shift_amount))
```

**AFTER** (fixed):
```python
# Keep Morton code left-aligned for neighbor generation
# Note: decode_morton_prefix_jax expects left-aligned uint64!
table_depth_int = int(mesh_gpu.table_depth)
center_prefix = morton_query  # Keep full 64-bit, left-aligned
```

---

## Diagnostic Proof

**Test script**: `diagnose_morton_neighbor_bug.py`

**Results**:
```
Correct input (left-aligned 0x0769DC0000000000):
  Decoded center: (61, 35, 27) ✅
  Neighbor [0]: (60, 34, 26) ✅
  Neighbor [13]: (61, 35, 27) ✅ (center)

Wrong input (shifted 0x1DA77):
  Decoded center: (0, 0, 0) ❌
  Neighbor [0]: (0, 0, 0) ❌
  Neighbor [13]: (0, 0, 0) ❌ (ALL neighbors at origin!)
```

---

## Expected Results After Fix

| Metric | Before (Broken) | After (Fixed) |
|--------|-----------------|---------------|
| **Particle search location** | (0,0,0) mesh corner | Actual particle position |
| **Spatial neighbors** | Wrong region | Correct 26 neighbors |
| **Retention @ step 100** | 67.57% | **~85-90%** (expected) |
| **Throughput** | 22,890 p/s | Similar or better |

---

## Why This Took So Long to Find

1. **The code LOOKED correct** - Morton encode/decode functions worked fine in isolation
2. **JAX uint64 arithmetic** - Silent overflow/underflow, no error messages
3. **Subtle type mismatch** - Small integer vs left-aligned, both valid uint64
4. **Nested function calls** - Hard to trace intermediate values through JIT compilation

---

## Lessons Learned

### For Morton Code Handling

1. **Always document bit layout**:
   ```python
   # Morton codes stored LEFT-ALIGNED in uint64:
   # Bits 60-62: level 0 octant
   # Bits 57-59: level 1 octant
   # ...
   # Bits 0-2: unused (always zero)
   ```

2. **Use type hints for alignment**:
   ```python
   def decode_morton(prefix_left_aligned: jnp.uint64, depth: int):
       """Decode Morton prefix (must be LEFT-ALIGNED!)"""
   ```

3. **Test with diagnostic values**:
   ```python
   # Known position → known Morton code → check decode
   assert decode(encode(61, 35, 27)) == (61, 35, 27)
   ```

### For JAX Debugging

1. **Extract intermediate values** with jax.debug.print
2. **Test scalar versions** before vmapping
3. **Compare with NumPy** reference implementation
4. **Use explicit type annotations** to catch mismatches

---

## Next Steps

### Immediate (Testing)

1. ✅ Bug fixed in `morton_global_search.py`
2. ⏳ Run production test with `L2_SEARCH_METHOD='neighbors'`
3. ⏳ Verify retention improves to 85-90%
4. ⏳ Confirm particles search correct spatial regions

### Follow-up (Optimization)

If retention is still <90% after this fix, consider:

1. **Multi-leaf search per prefix**: Some depth-7 prefixes may have 2-3 leaves
2. **Hierarchical fallback**: Search depth-6 neighbors if depth-7 fails
3. **Hybrid approach**: Morton neighbors + small radius fallback (±2 leaves)

But expect THIS fix to solve the 67% loss issue - it was a catastrophic bug!

---

## Summary

**One-line explanation**: All particles were searching neighbors around (0,0,0) instead of their actual position due to bit-alignment mismatch.

**One-line fix**: Keep Morton codes left-aligned when passing to neighbor generation.

**Expected impact**: Retention improves from 67% to 85-90%, fixing the critical particle loss.
