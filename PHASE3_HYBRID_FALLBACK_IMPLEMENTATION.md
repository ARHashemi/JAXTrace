# Phase 3: Hybrid Fallback Search Implementation

**Date**: 2026-02-15
**Goal**: Handle particle loss in coarse element blocks and poorly-connected mesh regions
**Approach**: 3×3×3 local search with radius-based fallback

---

## Problem Analysis

### Observed Behavior:
- **Phase 1+2 Results**: 18.84% retention @ 2500 steps
- **Baseline radius=10**: 40.97% retention @ 2500 steps
- **All mesh-aligned methods**: Identical 18.84% retention

### Key Observations:
1. ✅ **100% searchability** in 3×3×3 diagnostic (static test)
2. ❌ **Particle loss during tracking** (dynamic test)
3. 🔍 **User observation**: "Huge particle loss when entering coarse element blocks"

### Root Cause Hypothesis:
The 3×3×3 search works perfectly for **static point location** but fails during **dynamic tracking** when:
- Particles enter **coarse mesh regions** with poor connectivity
- **Large time steps** cause particles to jump across multiple cells
- **Mesh partitioning boundaries** between VTK parts have connectivity issues

---

## Solution: Hybrid Fallback Search

### Strategy:
1. **Primary**: 3×3×3 local search (fast, handles 99%+ of cases)
2. **Fallback**: Radius-based Morton search (comprehensive, handles edge cases)

### Implementation:

**File**: `jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py`

**Lines 297-320**:
```python
if mesh_aligned_octree_use_multi_local:
    # MULTI-CELL LOCAL (3×3×3): Search 27-cell neighborhood
    # ~494 tests/particle (27 cells × 18.31 elem/cell)
    elem_id, _ = search_mesh_aligned_octree_multi_local(
        pos,
        mesh_aligned_octree,
        max_tests=jnp.int32(600)
    )

    # PHASE 3 FALLBACK: If 3×3×3 fails and radius search is available, use it
    # This helps with coarse element blocks and poorly-connected mesh regions
    if l2_search_radius is not None:
        # Only use fallback if 3×3×3 failed to find element
        elem_id = jnp.where(
            elem_id < 0,  # If not found
            search_L2_global_morton_single(pos, mesh_gpu_global_morton, l2_search_radius),
            elem_id  # Otherwise keep 3×3×3 result
        )
```

**File**: `benchmark_l2_search_methods_with-export.py`

**Lines 554-555**: Added `l2_search_radius` parameter
```python
l2_search_method='radius',  # Used for Phase 3 fallback
l2_search_radius=l2_radius,  # Phase 3: fallback radius (None disables fallback)
```

**Lines 1077-1083**: Added Phase 3 test configuration
```python
{
    'name': 'Mesh-Aligned Multi-Cell + 3×3×3 + Fallback (Phase 3)',
    'l2_method': 'mesh_aligned_octree_multi_local',
    'l2_radius': 15,  # Fallback to radius search if 3×3×3 fails
    'description': 'Multi-cell + 3×3×3 local with radius=15 fallback for coarse regions',
    'expected_leaves': '~494-600 tests/particle (3×3×3 + fallback when needed)'
}
```

---

## How It Works

### Normal Case (99%+ of particles):
```
Particle position → 3×3×3 search → Element found → ✅ Use result
```

### Edge Case (coarse blocks, bad connectivity):
```
Particle position → 3×3×3 search → NOT found (-1)
                  → Fallback to radius=15 search
                  → Wider Morton search → Element found → ✅ Use result
```

### Performance Impact:
- **Best case**: 3×3×3 finds element → ~494 tests (same as Phase 2)
- **Fallback case**: 3×3×3 fails → additional radius search → ~494 + radius tests
- **Expected**: <1% fallback rate → minimal overhead

---

## Expected Results

### Retention Targets:

| Configuration | Expected Retention @ 2500 steps |
|---------------|--------------------------------|
| Phase 2 (3×3×3 only) | 18.84% (current) |
| **Phase 3 (3×3×3 + fallback)** | **30-40%+** |
| Baseline (radius=10) | 40.97% (reference) |

### Why This Should Help:

1. **Coarse block transitions**: Radius search spans multiple refinement levels
2. **Poor connectivity**: Radius search doesn't rely on octree structure
3. **Large jumps**: Radius search covers wider spatial region
4. **Mesh boundaries**: Radius search crosses VTK part boundaries better

---

## Testing

### Command:
```bash
python benchmark_l2_search_methods_with-export.py 2>&1 | tee logs/benchmark_phase3.log
```

### What to Look For:

1. **Retention improvement**: Should see >30% retention @ 2500 steps
2. **Performance**: Should be close to Phase 2 (minimal fallback overhead)
3. **Comparison**: How close to baseline radius=10 (40.97%)?

### Success Criteria:

- ✅ **Minimal**: Retention >25% (better than 18.84%)
- ✅ **Good**: Retention >35% (approaching baseline)
- ✅ **Excellent**: Retention ~40% (matching baseline)

---

## Next Steps If Phase 3 Succeeds:

If Phase 3 achieves >35% retention, the remaining gap to ideal is likely:

1. **Mesh connectivity issues** - VTK part merging/deduplication
2. **Velocity field topology** - particles leaving domain
3. **Element quality** - degenerate elements

Investigate with:
- Analyze unfound particles locations
- Check VTK part boundaries
- Review deduplication logic

---

## Next Steps If Phase 3 Fails:

If Phase 3 retention is still <25%, the problem is **not the search algorithm**. Investigate:

1. **Run connectivity diagnostic** - check VTK merging
2. **Analyze particle trajectories** - where do they go?
3. **Check element neighbors** - are there gaps in connectivity?
4. **Review deduplication** - are duplicate nodes causing issues?

---

## Summary

**Phase 3** adds a **safety net** for the 3×3×3 search:
- **Fast path**: 3×3×3 local search (covers 99%+)
- **Safety net**: Radius fallback (handles edge cases)
- **Target**: Bridge the gap from 18.84% to ~40% retention

**Key Insight**: The identical 18.84% across all mesh-aligned methods suggests a common bottleneck, likely related to mesh connectivity or coarse region transitions rather than search algorithm limitations.
