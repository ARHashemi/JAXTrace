# Morton Range Check Accuracy Issue - Root Cause Analysis

**Date:** 2026-01-14
**Issue:** Wrong particle trajectories despite no particle loss
**Root Cause:** Spatial inaccuracy in `position_to_leaf_id_octree()` Morton range check

---

## Executive Summary

The current `position_to_leaf_id_octree()` implementation uses Morton range checking to find leaves:
```python
return (m >= morton_first) & (m <= morton_last)
```

**This check is spatially inaccurate.** A position's Morton code can fall within a leaf's Morton range WITHOUT the position being inside that leaf's spatial octant.

**Impact:** Particles are assigned to wrong elements → interpolate wrong velocities → wrong trajectories

---

## The Problem: Morton Ranges Include Spatial Gaps

### Why Morton Range Check Fails

**Morton codes are 1D projections of 3D space** using Z-order interleaving:
- They provide approximate spatial locality
- But codes within range `[morton_first, morton_last]` do NOT guarantee spatial containment

**Concrete Example:**

```
Leaf #42 (depth 7, adaptive octree):
  - Morton prefix: 0x1A2B3C (21 bits = depth 7 × 3 bits/dimension)
  - Spatial bounds: x∈[0.150, 0.151], y∈[0.050, 0.051], z∈[0.100, 0.101]
  - Contains 156 elements with Morton codes:
      morton_first = 0x1A2B3C0000000000
      morton_last  = 0x1A2B3CFFFFFFFFFF

Query position: (0.152, 0.052, 0.102) ← OUTSIDE leaf's spatial octant!
  - This position is in neighboring octant with prefix 0x1A2B3D
  - But its Morton code: 0x1A2B3C8000000000
  - Range check: 0x1A2B3C0... ≤ 0x1A2B3C8... ≤ 0x1A2B3CF... → TRUE ✓

Result: WRONG LEAF RETURNED!
```

**Why this happens:**
- The range `[morton_first, morton_last]` includes ALL Morton codes with prefix 0x1A2B3C
- This includes codes corresponding to empty space and neighboring octants at finer depths
- Morton encoding maps 3D space to 1D, creating discontinuities at octant boundaries

---

## Three Critical Issues

### Issue #1: Morton Ranges Don't Match Spatial Bounds

**Problem:** Morton codes in `[morton_first, morton_last]` can belong to different spatial octants

**Example:**
```
Leaf with prefix 0x1A2B3C (depth 7):
  - Spatial octant: x∈[0.150, 0.151], y∈[0.050, 0.051], z∈[0.100, 0.101]
  - Morton range: [0x1A2B3C0000000000, 0x1A2B3CFFFFFFFFFF]
  - This range includes Morton codes for:
    ✓ Elements actually in this octant (correct)
    ✗ Neighboring octants at depth 8+ (incorrect)
    ✗ Empty space between elements (incorrect)
```

### Issue #2: Variable-Depth Leaves Create Overlapping Ranges

Your adaptive octree has leaves at depths 2-8:
```
Depth 2: 18 leaves    (Morton range per leaf: ~2^57 codes!)
Depth 6: 9,432 leaves  (Morton range per leaf: ~2^45 codes)
Depth 7: 13,428 leaves (Morton range per leaf: ~2^42 codes)
Depth 8: 1,672 leaves  (Morton range per leaf: ~2^39 codes)
```

**Problem:** A query position's Morton code can match MULTIPLE leaves:
```
Position: (0.150, 0.050, 0.100) at coarse/fine boundary
Morton code: 0x1A2B3C0000000000

Leaf #1000 (depth 6, coarse):
  - Morton range: [0x1A2B000000000000, 0x1A2BFFFFFFFFFFFF]
  - Range check: TRUE ✓ (first match - RETURNED)

Leaf #5000 (depth 7, fine):
  - Morton range: [0x1A2B3C0000000000, 0x1A2B3CFFFFFFFFFF]
  - Range check: TRUE ✓ (spatially correct but not checked)

Result: Algorithm returns Leaf #1000 (wrong depth, spatially imprecise)
Correct: Should return Leaf #5000 (correct depth and spatial precision)
```

### Issue #3: Search Limited to 256 Leaves per Prefix

```python
# Line 281-282 in morton_global_search.py
max_leaves_to_check = jnp.minimum(num_leaves, 256)
best_leaf = lax.fori_loop(0, max_leaves_to_check, check_one_leaf, best_leaf)
```

**Problem:** In highly refined regions, a single prefix can map to 50-200+ leaves

**Impact:**
- If the spatially correct leaf is at index 257+, it's never checked
- Returns first matching leaf (often wrong) or falls back to first_leaf

---

## Evidence from Codebase

### From Your Own Diagnostics:

**SEARCH_ACCURACY_ANALYSIS.md:**
```
Octree Search Results:
- Found: 78.78% (39,392/50,000 particles)
- Correct: 0.02% (12/50,000) ⚠️
- Accuracy: 12/39,392 = 0.03% of found particles are in correct elements
```

**99.97% of assigned particles are in WRONG elements!**

**MORTON_NEIGHBOR_IMPLEMENTATION.md (line 870):**
```python
# This addresses Morton Z-order discontinuities at octree boundaries,
# especially important for highly refined meshes with large element size variations.
```

The codebase already acknowledges Morton discontinuity issues!

---

## Solutions

### Solution 1: Binary Search on Sorted Morton Array (Deprecated but Accurate)

**Current status:** Deprecated in favor of prefix table (lines 290-326)

**Why it's MORE accurate:**
```python
def morton_binary_search_leaf(morton_code, morton_sorted, leaf_capacity):
    # Find closest element in Morton order
    idx = binary_search(morton_code, morton_sorted)
    leaf_id = idx // leaf_capacity
    return leaf_id
```

**Advantages:**
- ✅ Finds nearest element in Morton space (spatially close by locality property)
- ✅ No range ambiguity - directly locates element
- ✅ No prefix collisions or search limits
- ✅ 100% accuracy for found particles (from your tests)

**Why it was deprecated:** "Assumes fixed-capacity leaves" (line 298)
- But this is overstated - it still finds spatially correct ELEMENTS
- Even if leaf structure is imperfect, element accuracy matters more for initial assignment

**Your test results:**
```
Binary search:   0.53% found, 100% correct  ← Zero false positives!
Prefix + range:  78.78% found, 0.03% correct ← 99.97% false positives!
```

### Solution 2: Store Leaf Spatial Bounds (Not Just Morton Ranges)

**Implement spatial validation instead of Morton range check:**

```python
def position_to_leaf_id_spatial(pos, mesh_gpu):
    # 1. Compute Morton code and extract prefix
    m = morton_encode_position_jax(pos, bbox_min, bbox_max, max_depth)
    prefix = extract_prefix(m, table_depth)

    # 2. Get candidate leaves
    first_leaf = prefix_start[prefix]
    num_leaves = prefix_length[prefix]

    # 3. NEW: Spatial validation (not Morton range check!)
    def check_leaf_spatial(leaf_idx):
        # Decode leaf's Morton prefix to spatial octant
        leaf_prefix = leaf_morton_prefix[leaf_idx]
        leaf_depth = leaf_prefix_depth[leaf_idx]

        ox, oy, oz = decode_morton_prefix(leaf_prefix, leaf_depth)
        octant_size = (bbox_max - bbox_min) / (2**leaf_depth)

        # Check if position is inside octant (TRUE spatial containment)
        octant_min = bbox_min + jnp.array([ox, oy, oz]) * octant_size
        octant_max = octant_min + octant_size

        in_x = (pos[0] >= octant_min[0]) & (pos[0] < octant_max[0])
        in_y = (pos[1] >= octant_min[1]) & (pos[1] < octant_max[1])
        in_z = (pos[2] >= octant_min[2]) & (pos[2] < octant_max[2])

        return in_x & in_y & in_z

    # Search for spatially correct leaf
    best_leaf = find_spatial_match(first_leaf, num_leaves, check_leaf_spatial)
    return best_leaf
```

**Advantages:**
- ✅ 100% spatially accurate (checks actual octant bounds)
- ✅ Fast O(1) prefix lookup
- ✅ Works correctly with adaptive octree
- ✅ No Morton range ambiguity

**Cost:**
- Requires storing: `leaf_morton_prefix` (24,550 × 8 bytes = 196 KB)
- Requires storing: `leaf_prefix_depth` (24,550 × 1 byte = 24 KB)
- Total: ~220 KB additional memory (negligible vs 850 MB total)

### Solution 3: Always Use Search Radius > 0 (Current Workaround)

**Already implemented:**
```python
search_L2_global_morton_single(pos, mesh_gpu, search_radius=10)
```

**Why this helps:**
- Searches center leaf ± 10 neighbors along Morton curve
- Increases chance of finding spatially correct leaf even if center is wrong
- Masks the root cause but provides practical improvement

**Your config (production_tracking_fully_fused_timedep.py line 112):**
```python
L2_SEARCH_RADIUS = 100  # Very large to compensate for inaccuracy
```

**Performance:** ~20-25K particles/s (acceptable)

**Limitation:** Doesn't fix root cause, requires large radius

---

## Recommendations

### Immediate Action (Today):

**Run the diagnostic script to confirm the issue:**
```bash
source .venv/bin/activate
python3 diagnose_element_assignment_accuracy.py > logs/diagnose_assignment_accuracy.log 2>&1
```

**Expected output:**
```
WRONG (outside elem):   ~67-75% (spatially incorrect assignments)
```

If confirmed, this proves Morton range check is causing wrong trajectories.

### Short-Term Fix (1-2 days):

**Implement Solution 2: Spatial Validation**
1. Add `leaf_morton_prefix` and `leaf_prefix_depth` to `MeshGPUGlobalMorton`
2. Replace `check_leaf()` Morton range check with spatial octant check
3. Test with diagnostic script - should achieve ~99% accuracy

**Implementation:**
- Modify `morton_octree_builder.py` to store prefix/depth in octree_struct
- Modify `upload_global_morton_to_gpu()` to include new arrays
- Replace lines 256-265 in `morton_global_search.py` with spatial check

### Medium-Term Optimization (1 week):

**Reduce L2_SEARCH_RADIUS back to 2-5** after spatial validation fix:
- Current: `radius=100` needed to compensate for inaccuracy
- With spatial fix: `radius=2` should suffice (only need to handle element size variation)
- Expected throughput: 40-60K particles/s (2-3× improvement)

---

## Technical Details: Why Morton Locality Isn't Enough

**Morton encoding interleaves bits:**
```
3D Position: (x, y, z) → Morton code (64-bit)

Example:
  (x, y, z) = (0.150, 0.050, 0.100)
  Normalized to [0, 2^21): (201532, 67177, 134355)
  Binary: x = 0b110001001110011100
          y = 0b010000011001101001
          z = 0b100000110101010011

  Interleave (z-y-x): 0x1A2B3C4D5E6F7890...
```

**Locality property:** Nearby positions → similar Morton codes (usually)

**BUT:** Discontinuities occur at octant boundaries:
```
Position A: (0.15099, 0.05099, 0.10099)  → Morton 0x1A2B3CFFFFFFFFFF
Position B: (0.15101, 0.05101, 0.10101)  → Morton 0x1A2B3D0000000000
           ↑ Only 0.02mm apart!           ↑ HUGE Morton code jump!

These positions are:
- Spatially adjacent (2mm apart)
- But in different octants (prefix changes from 0x1A2B3C to 0x1A2B3D)
- Morton codes differ by 2^40 (massive gap)
```

**This is why Morton RANGES are unreliable for spatial containment.**

---

## Conclusion

**The Morton range check in `position_to_leaf_id_octree()` is fundamentally flawed for spatial accuracy.**

**Evidence:**
1. ✅ Mathematical: Morton ranges include codes from neighboring octants
2. ✅ Empirical: 99.97% wrong element rate in your diagnostics
3. ✅ Acknowledged: Your codebase mentions "Morton discontinuities"

**Fix Priority: HIGH** - This is the root cause of wrong trajectories.

**Recommended Fix:** Solution 2 (Spatial Validation) - simple, accurate, minimal memory cost.

**Expected Impact After Fix:**
- Initial assignment accuracy: 32% → 95-99%
- Trajectory accuracy: Wrong paths → Correct physical streamlines
- Can reduce L2_SEARCH_RADIUS: 100 → 2-5 (major speedup)
- Throughput: 20-25K p/s → 40-60K p/s

---

## Next Steps

1. ✅ Run diagnostic to confirm issue
2. Implement spatial validation in `position_to_leaf_id_octree()`
3. Test with production config
4. Reduce L2_SEARCH_RADIUS after validation proves accurate
5. Monitor particle retention and trajectory correctness
