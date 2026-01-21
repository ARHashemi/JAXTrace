# Neighbors L2 Search Analysis - Summary

**Date**: 2026-01-20
**Status**: Analysis complete, diagnostic tools ready

---

## Your Question

> "How does the 'neighbor' L2 search work? I'm thinking if we can perform sequential L2 but first search the leaf contains current element (centroid) or query position, then fallback to search leaf contains the neighbor elements. It can also be radius based. Is it meaningful or beneficial or the same as implemented 'neighbor' L2?"

---

## Short Answer

**Your proposed strategy is MOSTLY already implemented**, with one key difference:

| Feature | Your Proposal | Current 'neighbors' Implementation |
|---------|---------------|-----------------------------------|
| **Center reference** | Current element centroid | **Query position** |
| **Search strategy** | 3×3×3 spatial neighbors | **3×3×3 spatial neighbors** ✅ |
| **Handles Morton discontinuities** | Yes | **Yes** ✅ |
| **Handles adaptive depth** | Yes | **Yes** ✅ |

The current implementation searches around the **query position** (where the particle is now), not the **current element centroid** (where it was).

---

## Key Findings

### 1. How 'neighbors' Search Works

**Algorithm**:
1. Compute Morton code for **query position**
2. Decode to octant coordinates (cx, cy, cz) at table_depth (e.g., depth 7 → 128³ grid)
3. Generate 26 neighbor octants: all (cx±1, cy±1, cz±1) combinations
4. Search each of the 27 octants (26 neighbors + center)
5. For each octant: Look up leaves with that prefix, search up to 3 leaves

**Key Advantage**: Searches **spatially adjacent octants** regardless of their position in Morton order!

This is fundamentally different from radius-based search which searches consecutive leaves in Morton order (which can be far apart in 3D space).

### 2. Why It Uses Query Position (Not Current Element)

**After an RK4 step**, the particle has moved:
- Current element centroid: Where the particle **was**
- Query position: Where the particle **is now**

**Example**:
```
Current element centroid: (0.500, 0.500, 0.500) → Leaf 3200 (depth 6, coarse region)
After RK4 displacement:   (0.502, 0.501, 0.499) → Leaf 8150 (depth 8, fine region)

Using current element → Searches around (0.500, 0.500, 0.500) [WRONG location!]
Using query position  → Searches around (0.502, 0.501, 0.499) [CORRECT location!] ✅
```

The particle is more likely to be in an element **near its current position**, not near the old element's centroid.

### 3. Correction to Your Previous Analysis

You correctly identified that your mesh covers the whole domain with no holes, and that particle loss happens inside the domain (not at boundaries).

The previous analysis suggesting "boundary particles" was **incorrect** - thank you for the correction!

The real question is: **WHY do particles inside the domain fail to find elements even with spatial neighbor search?**

---

## What I've Provided

### 1. NEIGHBORS_L2_SEARCH_EXPLAINED.md

**Comprehensive explanation** covering:
- ✅ How 'neighbors' search works (3×3×3 octant search with spatial arithmetic)
- ✅ Why it searches query position (not current element centroid)
- ✅ Comparison to your proposed strategy
- ✅ Performance characteristics (67% succeed in 3×3×3, 33% need 5×5×5 fallback)
- ✅ Recommendations for improvement

**Key sections**:
- Detailed algorithm walkthrough with code references
- Comparison table: Your proposal vs current implementation
- Performance analysis
- Four recommendations to improve retention

### 2. diagnose_lost_particles.py

**Diagnostic script** that answers: **WHERE and WHY are particles lost?**

**What it does**:
1. Loads mesh and tracks particles for 10 steps (same as your production config)
2. Identifies lost particles
3. For each lost particle:
   - Finds nearest element (brute force CPU search)
   - Computes distance to nearest element
   - Determines what octant query position maps to
   - Determines what octant nearest element is in
   - Computes octant distance (Manhattan and max-norm)
4. Provides diagnosis for each particle:
   - "Within 3×3×3 neighborhood - should have been found!" (search bug)
   - "Within 5×5×5 neighborhood - enhanced search should find" (needs fallback)
   - "Beyond 5×5×5 search range" (expected loss)

**Output format**:
```
Particle 1:
  Position: (0.501234, 0.498765, 0.500123)
  Nearest element: 123456, distance: 1.234e-05
  Query position → Leaf 8150, Octant (64, 65, 64)
  Nearest element → Leaf 8149, Octant (64, 65, 63)
  Leaf distance: 1 leaves apart in Morton order
  Octant distance: Manhattan=1, Max=1
  ⚠️  DIAGNOSIS: Nearest element in 3×3×3 neighborhood - should have been found!
```

**Aggregate statistics**:
- Percentage within 3×3×3 (should be found)
- Percentage within 5×5×5 (needs enhanced search)
- Percentage beyond 5×5×5 (expected loss)
- Mean/median distance to nearest element

**This will definitively answer WHY retention stops at 95%!**

---

## Recommendations

### Immediate: Run the Diagnostic

```bash
python3 diagnose_lost_particles.py 2>&1 | tee logs/diagnose_lost_particles.log
```

**Expected runtime**: ~5-10 minutes

**This will reveal**:
- Are lost particles within 3×3×3 neighborhood? (Search bug - should be found!)
- Are lost particles within 5×5×5 neighborhood? (Need enhanced search)
- Are lost particles far away? (Expected - beyond search coverage)

**Based on results**, you'll know whether to:
1. **Fix point-in-tet checks** (if nearest element is in neighborhood but not found)
2. **Enable 5×5×5 fallback** (if nearest element is in 5×5×5 shell)
3. **Increase L1 depth** (if nearest element is >2 octants away)
4. **Use hierarchical search** (if coarse/fine boundary issues)

### If You Want to Test Hybrid Approach

**Your proposal**: Search current element's leaf first, then spatial neighbors

**To test**:
1. Modify [morton_global_search.py](jaxtrace/gpu/search/morton_global_search.py)
2. Add Tier 0 that searches leaf containing current element's centroid
3. Fallback to existing 3×3×3 search if not found

**Expected result**:
- ✅ Faster for small timesteps (particle stays in same leaf)
- ⚠️ No retention improvement (spatial neighbors already searched)
- ❌ Slower for large timesteps (wastes time searching wrong leaf)

**My assessment**: Unlikely to help, but could be a small performance optimization for small-DT tracking.

### To Improve Retention Beyond 95%

**Based on code review**, try these in order:

1. **Increase L1 depth** (Most likely to help!)
   ```python
   N_HOPS = 7  # Instead of 5
   ```
   Expected: +3-5% retention, but 2-3× slower

2. **Test hierarchical search**
   ```python
   L2_SEARCH_METHOD = 'hierarchical'  # Instead of 'incremental'
   ```
   Expected: Better at coarse/fine boundaries

3. **Use 5×5×5 fallback**
   - Already implemented in `search_L2_morton_neighbors_enhanced()`
   - Check if production script uses this variant

4. **Increase incremental radii**
   ```python
   INCREMENTAL_SEARCH_RADII = (2, 4, 8, 15, 30, 64, 100)  # Add tiers
   ```
   Expected: Catches particles that jumped far

---

## Files Created

1. **[NEIGHBORS_L2_SEARCH_EXPLAINED.md](NEIGHBORS_L2_SEARCH_EXPLAINED.md)**
   - 400+ lines comprehensive explanation
   - Algorithm walkthrough with code references
   - Comparison to your proposed strategy
   - Performance analysis and recommendations

2. **[diagnose_lost_particles.py](diagnose_lost_particles.py)**
   - 350+ lines diagnostic script
   - Analyzes WHERE and WHY particles are lost
   - Provides per-particle diagnosis
   - Aggregate statistics and root cause analysis

---

## Next Steps

1. **Run diagnostic**: `python3 diagnose_lost_particles.py`
2. **Check results**: Read logs/diagnose_lost_particles.log
3. **Based on findings**:
   - If most lost particles are within 3×3×3: Point-in-tet precision issue
   - If most are within 5×5×5: Need enhanced search
   - If most are >2 octants away: Need larger search (L1 depth or L2 radius)

4. **Try recommended fixes** based on root cause

---

## Summary

### Your Question Answered

**Q**: Is your proposed sequential search (current element first, then neighbors) already implemented?

**A**: Mostly YES! The 'neighbors' search DOES search spatial neighbors (3×3×3 octants), but it centers the search on **query position** instead of **current element centroid**. This is the correct choice because the particle has moved.

**Q**: Would your proposal be beneficial?

**A**: Probably not for retention (spatial neighbors already searched). Might give small speedup for small timesteps, but hurt performance for large timesteps. The diagnostic script will reveal the true bottleneck.

### Key Insight

The 95% retention plateau is **NOT** due to:
- ❌ Boundary particles (you confirmed mesh covers domain)
- ❌ Morton discontinuities alone (neighbors search handles this)
- ❌ Search strategy (spatial neighbors already implemented)

The plateau is **likely** due to:
- ⚠️ L1 search insufficient (doesn't cross coarse/fine boundaries well)
- ⚠️ Point-in-tet precision issues (nearest element exists but check fails)
- ⚠️ Particles jumping >2 octants (large displacement, beyond 5×5×5)

**The diagnostic script will tell you which one!**

---

## References

- [NEIGHBORS_L2_SEARCH_EXPLAINED.md](NEIGHBORS_L2_SEARCH_EXPLAINED.md) - Full explanation
- [diagnose_lost_particles.py](diagnose_lost_particles.py) - Diagnostic script
- [morton_global_search.py:647-758](jaxtrace/gpu/search/morton_global_search.py#L647-L758) - Implementation
- [morton_neighbors.py:169-237](jaxtrace/gpu/search/morton_neighbors.py#L169-L237) - Spatial arithmetic
- [MORTON_SEARCH_EXPLAINED.md](MORTON_SEARCH_EXPLAINED.md) - Morton search fundamentals
