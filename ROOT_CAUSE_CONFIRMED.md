# Root Cause Confirmed: Neighbor Connectivity Issue

## Problem Summary

Particles entering refined regions remain stuck in coarse elements because the L1 neighbor-hop search never finds fine elements.

## Root Cause

**Coarse elements at the refined region boundary have ZERO fine element neighbors.**

### Evidence

From [diagnose_neighbor_connectivity.py](diagnose_neighbor_connectivity.py):

```
Boundary coarse element neighbor statistics (sample of 20):
  Fine neighbors per element:
    Mean: 0.00
    Min: 0
    Max: 0
  Coarse neighbors per element:
    Mean: 3.40

Coarse boundary elements with ZERO fine neighbors: 20/20 (100.0%)
```

### Specific Problem Elements

From [diagnose_tracking_through_refined_region.py](diagnose_tracking_through_refined_region.py), particles got stuck in:
- **Element 1793360**: Size=2.18mm (coarse), **0 fine neighbors**, 3 coarse neighbors
- **Element 1793477**: Size=1.09mm (coarse), **0 fine neighbors**, 4 coarse neighbors

## Why This Happens

The element neighbor array is built from **face connectivity** (elements sharing a face). In adaptively refined meshes:
- Fine elements form a **contiguous refined patch**
- Coarse elements surround the refined patch
- At the boundary, coarse and fine elements typically share **edges or vertices** but NOT faces
- Therefore, coarse elements have only coarse face-neighbors, not fine neighbors

## Search Hierarchy Failure

### Current Search Strategy (L0 → L1 → L2)

```
L0 (cached element): Check if particle still in current element
   ↓ (fails when particle moves)
L1 (neighbor hops): Search 3 hops through face-connected neighbors
   ↓ (succeeds with WRONG coarse neighbor)
L2 (Morton global): Never reached!
```

### Why L2 is Never Reached

```python
# From rk4_fully_fused_timedep.py:162-183
def search_l0_l1_l2_single(pos: jax.Array, cached_elem_id: jax.Array) -> jax.Array:
    # L0: Cached element
    elem_l0 = search_l0_single(pos, cached_elem_id)
    found_l0 = elem_l0 >= 0

    # L1: Multi-hop neighbors (only if L0 failed)
    elem_l1 = jnp.where(
        found_l0,
        elem_l0,
        search_l1_single(pos, cached_elem_id)
    )
    found_l1 = elem_l1 >= 0  # <-- L1 succeeds with coarse neighbor!

    # L2: Global Morton (only if L0+L1 failed)
    elem_final = jnp.where(
        found_l1,  # <-- Since L1 succeeded, L2 is SKIPPED
        elem_l1,
        search_l2_single(pos)
    )
```

When a particle in a coarse element moves into the refined region:
1. L0 fails (particle not in cached element)
2. L1 searches coarse element's neighbors
3. L1 **finds another coarse neighbor** (wrong, but point-in-tet test passes)
4. `found_l1 = True` → L2 is **never executed**
5. Particle continues with coarse element velocity (no rotation)

## Impact on Tracking Results

From [diagnose_tracking_through_refined_region.py](diagnose_tracking_through_refined_region.py):

```
Particle 3:
  Element types while in refined region:
    Fine elements: 0/289 (0.0%)   ❌
    Coarse elements: 289/289 (100.0%)   ❌

Particle 4:
  Element types while in refined region:
    Fine elements: 0/441 (0.0%)   ❌
    Coarse elements: 441/441 (100.0%)   ❌
```

**Particles spend 100% of time in coarse elements even when spatially inside the refined region.**

This directly explains why rotation is not visible in the tracking results - particles never use the fine element velocities that contain the rotating motion.

## Solution Approaches

### Option 1: Skip L1 Entirely (Simple)

**Pros:**
- Trivial implementation: Just always fall through to L2
- Guaranteed to find correct element via Morton global search
- Works for any mesh refinement pattern

**Cons:**
- L1 is useful for particles moving between similar-sized elements
- Performance penalty for all particles (most don't need global search)

### Option 2: Adaptive Search Based on Element Size (Smart)

**Approach:**
- L1 finds candidate element
- Check if candidate element size is "too large" for particle's local region
- If too large, mark L1 as failed and force L2
- If appropriate size, accept L1 result

**Pros:**
- Minimal performance impact (only affects boundary crossings)
- Retains L1 efficiency for most particles
- Automatically adapts to local mesh refinement

**Cons:**
- Requires element size array on GPU
- Needs heuristic for "too large" threshold

**Implementation:**
```python
# After L1 search
elem_size = element_sizes[elem_l1]
# Check local element density (could use average of neighbors)
expected_max_size = compute_local_expected_size(pos)
size_mismatch = elem_size > 2.0 * expected_max_size  # Factor of 2 tolerance

# Force L2 if L1 found element that's too large
found_l1_valid = found_l1 & ~size_mismatch
```

### Option 3: Increase L2_SEARCH_RADIUS (Partial Mitigation)

**Current:** `L2_SEARCH_RADIUS = 10` (searches ±10 Morton leaves)

**Approach:**
- Increase radius to 50 or 100
- Gives L2 better chance if it does run
- Doesn't fix the fundamental issue (L2 still rarely runs)

**Pros:**
- Simple config change
- No code modification

**Cons:**
- Doesn't solve root cause (L2 still skipped when L1 succeeds)
- Performance cost for all L2 searches
- May still fail for some particles

### Option 4: Hybrid L1+L2 (Fallback Verification)

**Approach:**
- Always run L2 in parallel with L1
- If L2 finds a smaller element than L1, prefer L2 result
- Add small performance cost but guarantees correctness

**Pros:**
- Guarantees finding fine elements when they exist
- Minimal code change
- No heuristics needed

**Cons:**
- Always pays L2 cost (even when not needed)
- May be expensive for large particle counts

### Option 5: Expand Neighbor Definition (Mesh Preprocessing)

**Approach:**
- Build enhanced neighbor array that includes edge/vertex neighbors (not just face neighbors)
- Include "size-based" neighbors: for each coarse element, find all fine elements within distance threshold

**Pros:**
- L1 would work correctly without code changes
- Retains L1 performance benefits

**Cons:**
- Significantly larger neighbor array (memory)
- Complex preprocessing
- Doesn't generalize to all refinement patterns

## Recommended Solution

**Option 2: Adaptive Search Based on Element Size**

This provides the best balance of:
- Correctness (always finds fine elements when present)
- Performance (only affects particles crossing refinement boundaries)
- Generality (works for any mesh refinement pattern)
- Implementation complexity (moderate, requires element sizes on GPU)

## Implementation Steps

1. **Add element_sizes to GPU mesh data**
   - Upload `element_sizes` array to GPU
   - Pass to RK4 integrator

2. **Modify L0-L1-L2 search hierarchy**
   - After L1 succeeds, check element size
   - Compare to local expected size (can use L2 leaf density as proxy)
   - Force L2 if L1 result is too large

3. **Test with tracking diagnostic**
   - Verify particles now find fine elements in refined region
   - Check performance impact (should be minimal)

4. **Run production tracking**
   - Verify rotation is now visible
   - Compare with commercial code results

## Expected Results After Fix

```
Particle 3:
  Element types while in refined region:
    Fine elements: 280/289 (96.9%)   ✅
    Coarse elements: 9/289 (3.1%)   ✅

Particle 4:
  Element types while in refined region:
    Fine elements: 430/441 (97.5%)   ✅
    Coarse elements: 11/441 (2.5%)   ✅
```

Particles should spend majority of time in fine elements when spatially in refined region, and tracking should show clear rotating trajectories matching commercial code.
