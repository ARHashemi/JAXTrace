# Phase 1 Fix: Replace Innermost Loop with lax.fori_loop

**Date**: 2026-01-12
**Status**: ✅ IMPLEMENTED - Ready for Testing

---

## What Was Changed

### Single Function Modified
**File**: `jaxtrace/gpu/search/morton_global_search.py`
**Function**: `search_in_leaf_global` (lines 455-503)

**Before** (8-iteration unrolled Python loop):
```python
found_elem = jnp.int32(-1)
for j in range(8):  # ← Unrolled by JAX JIT
    active = (found_elem == -1) & (j < length)
    # ... point-in-tet check ...
    found_elem = jnp.where(inside & active, elem_id, found_elem)
return found_elem
```

**After** (bounded lax.fori_loop):
```python
def check_element(j, found_elem):
    active = (found_elem == -1) & (j < length)
    # ... point-in-tet check ...
    return jnp.where(inside & active, elem_id, found_elem)

found_elem = lax.fori_loop(0, 8, check_element, jnp.int32(-1))
return found_elem
```

---

## Expected Impact

### RAM Reduction During Compilation

| L2 Method | Unrolled Iters Before | Unrolled Iters After | RAM Before | RAM After | Reduction |
|-----------|----------------------|---------------------|------------|-----------|-----------|
| **Radius** | 40 | 5 | 90 GB | 11 GB | 8× |
| **Neighbors** | 648 | 81 | 2.2 TB | 275 GB | 8× |
| **Hierarchical** | 3,456 | 432 | 11.7 TB | 1.46 TB | 8× |
| **Enhanced** | 3,000 | 375 | 10.1 TB | 1.26 TB | 8× |

**Key insight**: This fix reduces XLA graph size by **8× across all methods** because the innermost 8-element loop is now bounded instead of unrolled.

---

## Testing Instructions

### Test 1: Verify 'radius' method (baseline, should still work)
```bash
# Edit production_tracking_fully_fused_timedep.py line 127
L2_SEARCH_METHOD = 'radius'

# Run test
python production_tracking_fully_fused_timedep.py > logs/phase1_test_radius.log 2>&1
```

**Expected result**:
- ✅ Compilation succeeds (11 GB RAM, well within limits)
- ✅ Performance similar to before (bounded loop overhead is minimal)
- ✅ Retention unchanged

---

### Test 2: Test 'neighbors' method (critical test)
```bash
# Edit production_tracking_fully_fused_timedep.py line 127
L2_SEARCH_METHOD = 'neighbors'

# Run test
python production_tracking_fully_fused_timedep.py > logs/phase1_test_neighbors.log 2>&1
```

**Expected result**:
- ✅ Should work! (275 GB RAM, feasible on most systems)
- ⚠️ If still crashes → Need Phase 2 fix (middle loop)

---

### Test 3: Test 'hierarchical' method (worst case)
```bash
# Edit production_tracking_fully_fused_timedep.py line 127
L2_SEARCH_METHOD = 'hierarchical'

# Run test
python production_tracking_fully_fused_timedep.py > logs/phase1_test_hierarchical.log 2>&1
```

**Expected result**:
- ⚠️ May still crash (1.46 TB RAM, exceeds typical 512 GB systems)
- ✅ Should work on systems with ≥2 TB RAM
- 🔴 If crashes → Definitely need Phase 2 fix

---

## Monitoring During Tests

### Watch Compilation RAM Usage
```bash
# In another terminal during test:
watch -n 1 'ps aux | grep python | grep production'
```

Look for RSS (resident set size) column during "Compiling..." phase.

### Check for OOM Errors
```bash
# Check system logs
sudo dmesg | tail -50 | grep -i "out of memory"
```

### Expected Timeline
- **First run** (cold start): 1-5 minutes compilation + execution
  - This is when RAM spike happens
  - If it crashes, happens here
- **Subsequent runs**: Much faster (cached compilation)

---

## Success Criteria

### Phase 1 Success (No Phase 2 needed):
- ✅ 'radius' works (baseline)
- ✅ 'neighbors' works (most common use case)
- ✅ 'hierarchical' works (nice to have)

### Phase 1 Partial Success (Phase 2 needed):
- ✅ 'radius' works
- ✅ 'neighbors' works
- 🔴 'hierarchical' crashes → Implement Phase 2

### Phase 1 Failure (Phase 2 required):
- ✅ 'radius' works
- 🔴 'neighbors' crashes → Implement Phase 2 immediately

---

## Rollback Instructions

If Phase 1 causes issues, revert with:

```bash
# Restore original unrolled loop version
git diff jaxtrace/gpu/search/morton_global_search.py
git checkout jaxtrace/gpu/search/morton_global_search.py
```

Or manually replace `lax.fori_loop` with the original unrolled loop.

---

## Next Steps Based on Results

### If Test 2 ('neighbors') Succeeds:
**Outcome**: Phase 1 is sufficient for most use cases!
- 'radius' works (baseline)
- 'neighbors' works (most common)
- Can use these methods in production

**Optional**: Implement Phase 2 if you need 'hierarchical' method.

### If Test 2 ('neighbors') Fails:
**Required**: Implement Phase 2 (middle loop fix)
- Replace 3-leaf and 8-leaf loops with `lax.fori_loop`
- Expected impact: 24-64× total reduction
- 'neighbors': 2.2 TB → 92 GB
- 'hierarchical': 11.7 TB → 183 GB

### If Test 3 ('hierarchical') Fails:
**Expected**: This is likely even with Phase 1 (1.46 TB is high)
**Action**: Proceed to Phase 2 fix

---

## Performance Notes

**Bounded loop overhead**: ~5-10% slower execution compared to unrolled loops
- Unrolled: Direct sequential operations (fast)
- Bounded: Loop control overhead (slightly slower)
- **Trade-off is worth it** to enable compilation!

**JAX optimization**: JAX's XLA compiler optimizes `lax.fori_loop` well
- Converts to efficient while/for loops in GPU code
- Minimal runtime overhead
- Big win during compilation (8× less RAM)

---

## Current Status

✅ **Phase 1 implemented** in [morton_global_search.py:455-503](jaxtrace/gpu/search/morton_global_search.py#L455-L503)

**Ready for testing!** Run tests in order:
1. Test 'radius' (verify no regression)
2. Test 'neighbors' (critical test)
3. Test 'hierarchical' (optional, likely needs Phase 2)

Report results and we'll proceed to Phase 2 if needed.
