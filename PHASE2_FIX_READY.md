# Phase 2 Fix: Replace Middle Loops with lax.fori_loop

**Status**: 📝 DOCUMENTED - Ready to Implement if Phase 1 Insufficient

---

## When to Implement Phase 2

**Implement Phase 2 if:**
- ✅ Phase 1 Test 1 (radius) succeeds
- 🔴 Phase 1 Test 2 (neighbors) fails with OOM
- 🔴 Phase 1 Test 3 (hierarchical) fails with OOM (expected)

**Skip Phase 2 if:**
- ✅ Phase 1 Test 2 (neighbors) succeeds AND you don't need hierarchical

---

## What Phase 2 Fixes

Phase 2 targets the **middle loops** (leaf iterations within octants) in:

1. **L2 Neighbors**: 3-leaf loop per octant
2. **L2 Hierarchical**: 8-leaf loop per octant (depth-7 and depth-6)
3. **L2 Enhanced**: 3-leaf loop per octant (both tiers)

These are currently unrolled Python loops that need to become bounded `lax.fori_loop`.

---

## Expected Impact (Combined Phase 1 + Phase 2)

| Method | Before P1 | After P1 | After P2 | Total Reduction |
|--------|-----------|----------|----------|-----------------|
| **Neighbors** | 2.2 TB | 275 GB | **92 GB** | 24× |
| **Hierarchical** | 11.7 TB | 1.46 TB | **183 GB** | 64× |
| **Enhanced** | 10.1 TB | 1.26 TB | **421 GB** | 24× |

**All methods should work on 512 GB systems after Phase 2!**

---

## Changes Required

### Fix 1: L2 Neighbors - 3-leaf loop
**File**: `jaxtrace/gpu/search/morton_global_search.py`
**Function**: `search_L2_morton_neighbors_single` (lines ~679-690)

**Current** (unrolled):
```python
for i in range(27):  # Octants (keep unrolled)
    # ... prefix lookup ...

    octant_elem = jnp.int32(-1)
    octant_found = jnp.bool_(False)

    # UNROLLED: 3 iterations
    for leaf_offset in range(3):
        leaf_id = first_leaf + leaf_offset
        valid = (leaf_offset < num_leaves_in_prefix) & ...
        result = search_in_leaf_global(pos, leaf_id, mesh_gpu)
        improved = result >= 0
        octant_elem = jnp.where(improved, result, octant_elem)
        octant_found = octant_found | improved
```

**Phase 2 fix**:
```python
for i in range(27):  # Octants (keep unrolled)
    # ... prefix lookup ...

    def search_leaves_in_octant(leaf_offset, state):
        """Search one leaf in octant (bounded loop body)."""
        octant_elem, octant_found = state
        leaf_id = first_leaf + leaf_offset
        valid = (leaf_offset < num_leaves_in_prefix) & (leaf_id >= 0) & (leaf_id < mesh_gpu.n_leaves) & jnp.logical_not(octant_found)

        result = jnp.where(
            valid,
            search_in_leaf_global(pos, leaf_id, mesh_gpu),
            jnp.int32(-1)
        )
        improved = result >= 0

        return (
            jnp.where(improved, result, octant_elem),
            octant_found | improved
        )

    # BOUNDED LOOP: No unrolling (3 iterations)
    octant_elem, octant_found = lax.fori_loop(
        0, 3,
        search_leaves_in_octant,
        (jnp.int32(-1), jnp.bool_(False))
    )
```

---

### Fix 2: L2 Enhanced (5×5×5) - 3-leaf loop
**File**: `jaxtrace/gpu/search/morton_global_search.py`
**Function**: `search_5x5x5_outer_shell` (lines ~794-800)

**Same pattern as Fix 1** - replace the inner 3-leaf loop with `lax.fori_loop`.

---

### Fix 3: L2 Hierarchical Depth-7 - 8-leaf loop
**File**: `jaxtrace/gpu/search/morton_global_search.py`
**Function**: `search_L2_morton_hierarchical_single` (lines ~920-933)

**Current** (unrolled):
```python
for i in range(27):  # Octants depth-7 (keep unrolled)
    # ... prefix lookup ...

    octant_elem = jnp.int32(-1)
    octant_found = jnp.bool_(False)

    # UNROLLED: 8 iterations
    for leaf_offset in range(8):
        leaf_id = first_leaf + leaf_offset
        valid = (leaf_offset < num_leaves) & ...
        result = search_in_leaf_global(pos, leaf_id, mesh_gpu)
        improved = result >= 0
        octant_elem = jnp.where(improved, result, octant_elem)
        octant_found = octant_found | improved
```

**Phase 2 fix**:
```python
for i in range(27):  # Octants depth-7 (keep unrolled)
    # ... prefix lookup ...

    def search_leaves_depth7(leaf_offset, state):
        """Search one leaf at depth-7 (bounded loop body)."""
        octant_elem, octant_found = state
        leaf_id = first_leaf + leaf_offset
        valid = (leaf_offset < num_leaves) & (leaf_id >= 0) & (leaf_id < mesh_gpu.n_leaves) & jnp.logical_not(octant_found)

        result = jnp.where(
            valid,
            search_in_leaf_global(pos, leaf_id, mesh_gpu),
            jnp.int32(-1)
        )
        improved = result >= 0

        return (
            jnp.where(improved, result, octant_elem),
            octant_found | improved
        )

    # BOUNDED LOOP: No unrolling (8 iterations)
    octant_elem, octant_found = lax.fori_loop(
        0, 8,
        search_leaves_depth7,
        (jnp.int32(-1), jnp.bool_(False))
    )
```

---

### Fix 4: L2 Hierarchical Depth-6 - 8-leaf loop
**File**: `jaxtrace/gpu/search/morton_global_search.py`
**Function**: `search_L2_morton_hierarchical_single` (lines ~963-976)

**Same pattern as Fix 3** - replace the inner 8-leaf loop with `lax.fori_loop`, but for depth-6 section.

---

## Summary of Phase 2 Changes

**Total functions to modify**: 4 inner loops
1. Neighbors: 3-leaf loop → `lax.fori_loop(0, 3, ...)`
2. Enhanced: 3-leaf loop → `lax.fori_loop(0, 3, ...)`
3. Hierarchical depth-7: 8-leaf loop → `lax.fori_loop(0, 8, ...)`
4. Hierarchical depth-6: 8-leaf loop → `lax.fori_loop(0, 8, ...)`

**Keep unrolled**: Octant loops (27, 98, 125 iterations) - these are outer loops and OK to stay unrolled.

---

## Testing After Phase 2

### Test 1: Verify 'neighbors'
```bash
L2_SEARCH_METHOD = 'neighbors'
python production_tracking_fully_fused_timedep.py > logs/phase2_test_neighbors.log 2>&1
```

**Expected**: ✅ Works! (92 GB RAM)

### Test 2: Verify 'hierarchical'
```bash
L2_SEARCH_METHOD = 'hierarchical'
python production_tracking_fully_fused_timedep.py > logs/phase2_test_hierarchical.log 2>&1
```

**Expected**: ✅ Works! (183 GB RAM)

### Test 3: Verify 'radius' (regression test)
```bash
L2_SEARCH_METHOD = 'radius'
python production_tracking_fully_fused_timedep.py > logs/phase2_test_radius.log 2>&1
```

**Expected**: ✅ Still works! (11 GB RAM)

---

## Performance Trade-offs

**Phase 1 overhead**: ~5% (1 level of bounded loops)
**Phase 2 overhead**: ~10% (2 levels of bounded loops)

**Total slowdown**: ~15% execution time vs fully unrolled
**Total RAM saving**: 24-64× during compilation

**Verdict**: Worth the trade-off! Code that crashes is 0% fast. 😄

---

## Implementation Time

**Phase 2 implementation**: ~30-45 minutes
- 4 similar modifications
- Copy-paste pattern from Phase 1
- Adjust loop bounds (3 vs 8)
- Update function names for clarity

**Phase 2 testing**: ~10-30 minutes
- 3 test runs (neighbors, hierarchical, radius)
- Each takes 2-10 minutes (depending on compilation cache)

**Total**: ~1-2 hours for Phase 2 complete

---

## Next Steps

1. **First**: Run Phase 1 tests (see PHASE1_FIX_SUMMARY.md)
2. **If neighbors fails**: Report back, I'll implement Phase 2 immediately
3. **If hierarchical fails but neighbors works**: Optional Phase 2 (only if you need hierarchical)
4. **If all Phase 1 tests pass**: Celebrate! Phase 2 not needed! 🎉

---

## Ready to Implement

When you're ready for Phase 2, just say:
- "Implement Phase 2" → I'll make all 4 changes
- "Implement Phase 2 for neighbors only" → I'll make just Fix 1
- "Implement Phase 2 for hierarchical only" → I'll make Fixes 3 & 4

I'm standing by with the code ready to go! 🚀
