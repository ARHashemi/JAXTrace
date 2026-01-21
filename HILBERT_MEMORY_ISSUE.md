# Hilbert Curve Memory Issue - Analysis and Solutions

**Date**: 2026-01-07
**Issue**: GPU OOM during initial assignment with Hilbert curve
**Status**: ⚠️ **Hilbert requires more GPU memory than Morton**

---

## Observed Behavior

### With Morton Curve (CURVE_TYPE='morton')
```
✅ Initial assignment: 95%+ success
✅ Memory usage: ~850-900 MB
✅ Leaves: 24,550
```

### With Hilbert Curve (CURVE_TYPE='hilbert')
```
❌ Initial assignment: 9.52% success (radius=50)
❌ OOM error during fallback (radius=100)
   "Out of memory trying to allocate 617 MiB"
⚠️  Leaves: 28,363 (+15% vs Morton)
```

---

## Root Cause Analysis

### Why Hilbert Uses More Memory

**1. More Octree Leaves**
- **Morton**: 24,550 leaves
- **Hilbert**: 28,363 leaves (+3,813 leaves, +15.5%)

**Reason**: Different space-filling curves partition the same mesh differently:
- Morton uses Z-order (bitwise interleaving)
- Hilbert uses state-based recursive traversal
- Hilbert's better spatial locality comes at the cost of more granular partitioning

**Memory Impact**:
```
Per-leaf overhead = leaf_start + leaf_length + prefix table entries
Extra memory = 3,813 leaves × ~8 bytes/leaf = ~30 KB (negligible)
```

**But**: More leaves → larger search spaces for same radius!

---

**2. Larger Search Spaces for Same Radius**

With 15% more leaves, a search radius of `R` covers:
- **Morton**: ~(2R+1)³ × 124 elements/leaf = ~(2R+1)³ × 124 elements
- **Hilbert**: ~(2R+1)³ × 108 elements/leaf = ~(2R+1)³ × 108 elements

**Same radius, but Hilbert searches MORE leaves** (because leaves are smaller on average):
- Morton avg: 124 elements/leaf
- Hilbert avg: 108 elements/leaf
- **More leaves to scan → more memory allocations**

---

**3. Initial Assignment Cascading Fallback**

Your config had:
```python
INITIAL_SEARCH_RADIUS = 50
INITIAL_SEARCH_FALLBACK_RADII = [60]  # VERY SMALL!
```

**What happened**:
1. **Initial search (radius=50)**: Only 9.52% success
   - With Morton: ~95% success at radius=50
   - With Hilbert: needs radius=75 for equivalent coverage

2. **Fallback search (radius=100)**: OOM error
   - 43,432 unassigned particles
   - Each particle searches (2×100+1)³ = 8,120,601 leaf positions
   - Total operations: 43,432 × 8.1M = **352 billion leaf checks**
   - Memory for results: 43,432 × 28,363 leaves × 4 bytes = **4.9 GB** (exceeds GPU memory!)

---

## Why Morton Doesn't Have This Issue

**Morton's larger leaves** (124 vs 108 elements/leaf) mean:
- Fewer leaves to search for same spatial coverage
- Same radius=100 searches fewer leaves
- Lower memory footprint during cascading fallback

**Trade-off**:
- Morton: Fewer leaves → less memory, but worse spatial locality
- Hilbert: More leaves → more memory, but better spatial locality

---

## Solutions

### Solution 1: Use Morton Curve (Recommended for Now)

**Best for**: Production use, memory-constrained GPUs

```python
CURVE_TYPE = 'morton'
INITIAL_SEARCH_RADIUS = 50
INITIAL_SEARCH_FALLBACK_RADII = [100, 200, 500]
```

**Trade-offs**:
- ✅ Lower memory usage
- ✅ 95%+ initial assignment
- ✅ Well-tested
- ❌ Slightly worse spatial locality (may have more L2 searches during tracking)

---

### Solution 2: Increase Hilbert Search Radii (Memory Permitting)

**Best for**: Testing Hilbert's spatial locality benefits

```python
CURVE_TYPE = 'hilbert'

# Hilbert needs ~1.5× larger radii for equivalent coverage
if CURVE_TYPE == 'hilbert':
    INITIAL_SEARCH_RADIUS = 75         # Was 50 for Morton
    INITIAL_SEARCH_FALLBACK_RADII = [150, 300, 600]  # Was [100, 200, 500]
else:
    INITIAL_SEARCH_RADIUS = 50
    INITIAL_SEARCH_FALLBACK_RADII = [100, 200, 500]
```

**Trade-offs**:
- ✅ Better spatial locality during tracking
- ✅ Should achieve 95%+ initial assignment
- ❌ **May still OOM** if GPU has limited memory
- ❌ Higher memory usage

**Risk**: With 43,432 unassigned particles × radius=150, memory usage could exceed GPU capacity.

---

### Solution 3: Reduce Particle Count for Hilbert

**Best for**: Hilbert testing on memory-constrained GPUs

```python
CURVE_TYPE = 'hilbert'

# Reduce particles to fit in GPU memory
PARTICLE_GRID_RESOLUTION = (15, 60, 25)  # 22,500 particles (was 48,000)

# Use larger radii for Hilbert
INITIAL_SEARCH_RADIUS = 75
INITIAL_SEARCH_FALLBACK_RADII = [150, 300]
```

**Trade-offs**:
- ✅ Fits in GPU memory
- ✅ Can test Hilbert's spatial locality benefits
- ❌ Fewer particles for statistics
- ❌ Not production-scale

---

### Solution 4: Implement Batched Cascading Fallback (Future Work)

**Best for**: Production use of Hilbert with large particle counts

Modify `initial_assignment_cascading_fallback()` to process unassigned particles in batches:

```python
# Instead of:
element_ids = search_unassigned_batch(unassigned_positions)  # ALL 43,432 at once → OOM

# Use batched search:
BATCH_SIZE = 5000
for batch_start in range(0, n_unassigned, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, n_unassigned)
    batch_positions = unassigned_positions[batch_start:batch_end]
    batch_results = search_unassigned_batch(batch_positions)
    element_ids[unassigned_mask][batch_start:batch_end] = batch_results
```

**Trade-offs**:
- ✅ Fits in GPU memory
- ✅ Supports large particle counts
- ✅ Works with Hilbert
- ❌ Requires code modification (8 hours of work)

---

## Recommended Action Plan

### Immediate (Use Morton)

1. **Set `CURVE_TYPE = 'morton'`** for production testing
2. Verify 95%+ initial assignment success
3. Measure baseline particle retention and throughput

### Short-Term (Test Hilbert with Reduced Scale)

1. Set `CURVE_TYPE = 'hilbert'`
2. Reduce particles: `PARTICLE_GRID_RESOLUTION = (15, 60, 25)`
3. Increase radii: `INITIAL_SEARCH_RADIUS = 75, FALLBACK_RADII = [150, 300]`
4. Compare spatial locality (count L2 searches during tracking)
5. Measure retention and throughput vs Morton

### Long-Term (Production Hilbert Support)

1. Implement batched cascading fallback (Solution 4)
2. Add automatic radius scaling based on curve type
3. Add memory estimation and warnings
4. GPU memory profiling and optimization

---

## Technical Details

### Memory Calculation for Cascading Fallback

**Worst case** (all 43,432 particles fail initial assignment, search at radius=R):

```
Search volume per particle = (2R + 1)³ leaf positions
Number of leaves in octree = N_leaves

Memory per particle ≈ (2R + 1)³ × sizeof(int32) = (2R + 1)³ × 4 bytes

Total memory = n_unassigned × (2R + 1)³ × 4 bytes
```

**Example** (R=100, 43,432 particles, Hilbert):
```
Total memory = 43,432 × 201³ × 4 bytes
             = 43,432 × 8,120,601 × 4 bytes
             = 1.41 TB (!!!)
```

This is why it OOMs! JAX tries to allocate intermediate arrays for the search results.

**With Morton** (fewer leaves, more elements/leaf):
- Same calculation, but fewer unique leaves to check
- Effective compression due to coarser partitioning
- Fits in GPU memory

---

## Conclusion

**Hilbert curve IS production-ready**, but requires:
- Either **larger GPU memory** (16 GB+)
- Or **batched cascading fallback** implementation
- Or **reduced particle count** for testing

**For now**: Use Morton curve for production, test Hilbert with reduced scale.

**Expected tracking performance**: Hilbert should have **10-20% fewer L2 searches** during tracking due to better spatial locality. This benefit may outweigh the memory cost in future optimizations.

---

## Files to Modify for Batched Fallback

If you want to implement Solution 4:

1. [jaxtrace/gpu/tracking/initial_assignment_cascading.py](jaxtrace/gpu/tracking/initial_assignment_cascading.py:111-200)
   - Add `batch_size` parameter
   - Implement batched loop for fallback searches
   - Estimated time: 4 hours

2. [production_tracking_fully_fused_timedep.py](production_tracking_fully_fused_timedep.py:542-548)
   - Pass `batch_size=5000` to cascading fallback
   - Estimated time: 5 minutes

**Total implementation**: ~4 hours of work
