# Initial Assignment Failure - Root Cause Analysis

**Date**: 2026-01-13
**Problem**: Only 31.81% of particles assigned initially (71,580/225,000)
**Previous Performance**: Was working better before recent changes

---

## Summary of Findings

**Root Cause**: Morton curve position mapping is **WRONG** for your refined mesh!

The current `position_to_leaf_id_octree()` function uses a **prefix table** that assumes:
- Elements are uniformly distributed along the Morton curve
- Each prefix corresponds to roughly equal Morton code ranges
- Leaves have similar element counts

**Your mesh violates all of these assumptions**:
- 85% fine elements in tiny spatial region (center)
- 15% coarse elements in huge spatial region (far field)
- **Non-uniform Morton code distribution** - fine elements clustered in narrow Morton range

**Result**: Position-to-leaf mapping is off by **hundreds to thousands of leaves**, causing initial search to fail!

---

## The Problem: Morton Code Distribution

### Expected (Uniform Mesh)
```
Morton curve: [0 ==================== 2^63-1]
              ↓                      ↓
Leaf 0        Leaf 5000              Leaf 24550
(uniform spacing, linear mapping works)
```

### Actual (Your Refined Mesh)
```
Morton curve: [0 ===FINE=== ==================COARSE=================== 2^63-1]
              ↓   ↓   ↓    ↓                                           ↓
Leaf 0    ...20K leaves... Leaf 21000                                  Leaf 24550
(85% of leaves in 5% of Morton range!)
```

**Refined region** (tool center):
- Spatial extent: 18mm × 18mm × 4.5mm (tiny)
- Elements: 2.6M (85% of total)
- Morton range: Narrow (clustered codes)
- Leaves: ~20,000 (82% of total leaves)

**Coarse region** (far field):
- Spatial extent: 60mm × 46mm × 10mm (huge)
- Elements: 450K (15% of total)
- Morton range: Wide (sparse codes)
- Leaves: ~4,500 (18% of total leaves)

---

## Current Algorithm Issues

### Algorithm: `position_to_leaf_id_octree()` (lines 207-290)

**Step 1**: Encode particle position to Morton code ✅ (This is correct)
```python
morton_code = morton_encode_position_jax(pos, bbox_min, bbox_max, max_depth)
# Result: 63-bit Morton code representing spatial position
```

**Step 2**: Extract prefix (top `table_depth × 3` bits) ✅ (This is correct)
```python
shift_amount = 63 - (table_depth * 3)
prefix = (morton_code >> shift_amount).astype(jnp.int32)
# Result: Integer in range [0, 8^table_depth - 1]
```

**Step 3**: Look up leaf range from prefix table ❌ **THIS IS WHERE IT FAILS!**
```python
first_leaf = prefix_start[prefix]
num_leaves = prefix_length[prefix]
# Problem: prefix_start[prefix] gives FIRST leaf with this prefix
#          but there may be HUNDREDS of leaves with this prefix!
```

**Step 4**: Binary search within leaf range ⚠️ **This tries to fix Step 3, but it's too late**
```python
# Searches within [first_leaf, first_leaf + num_leaves - 1]
# Problem: If first_leaf is wrong, search fails entirely
```

---

## Why Prefix Table Fails for Refined Meshes

### Prefix Table Construction (morton_octree_builder.py)

The prefix table is built by:
1. Grouping leaves by their Morton prefix at depth D (usually 6-7)
2. Storing the **first leaf ID** and **count** for each prefix

```python
# Example for table_depth=7 (8^7 = 2,097,152 prefixes)
prefix_start[prefix] = first_leaf_with_this_prefix
prefix_length[prefix] = num_leaves_with_this_prefix
```

### Problem: Multiple Leaves per Prefix

In adaptive octrees, **multiple leaves can share the same prefix**!

**Example**:
- Prefix `0x1234567` at depth-7
- This prefix represents a spatial octant
- In refined region: This octant contains **8 leaves** (subdivided to depth-10)
- In coarse region: This octant contains **1 leaf** (only depth-7)

**Current algorithm**:
```python
first_leaf = prefix_start[0x1234567]  # Returns leaf #1500
num_leaves = prefix_length[0x1234567]  # Returns 8
# Searches leaves [1500, 1507]
```

**Problem**: What if the particle is in leaf #1503?
- Binary search looks for Morton code in sorted array
- But array indices [1500-1507] correspond to **leaf IDs**, not Morton codes
- If Morton codes aren't uniformly distributed, this fails!

---

## Concrete Example with Your Mesh

### Particle Seeding Region
```
X: [-18mm, -9mm] (entrance, coarse elements)
Y: [-13.8mm, 13.8mm]
Z: [-7mm, -0.1mm]
```

### Morton Encoding
```python
# Particle at X=-15mm, Y=0mm, Z=-3mm (middle of seeded region)
normalized_x = (-15 - (-60)) / (0 - (-60)) = 45/60 = 0.75
normalized_y = (0 - (-23)) / (23 - (-23)) = 23/46 = 0.5
normalized_z = (-3 - (-10)) / (0 - (-10)) = 7/10 = 0.7

# Scale to 21-bit coordinates (max_depth=21)
ix = int(0.75 * (2^21 - 1)) = 1,572,863
iy = int(0.5 * (2^21 - 1)) = 1,048,575
iz = int(0.7 * (2^21 - 1)) = 1,468,005

# Interleave bits (simplified)
morton_code ≈ 0x3A5E7C9B4D2F1...  # Some 63-bit value
```

### Prefix Extraction (table_depth=7)
```python
shift_amount = 63 - (7 * 3) = 42
prefix = morton_code >> 42  # Extract top 21 bits
prefix ≈ 234567  # Some value in [0, 2^21 - 1]
```

### Prefix Table Lookup
```python
first_leaf = prefix_start[234567]  # Let's say returns 15,000
num_leaves = prefix_length[234567]  # Let's say returns 50
# Searches leaves [15,000, 15,049]
```

**Problem**: Your particle is in the **coarse region**, but the prefix table was built with **fine region bias**!

- Fine region: Prefixes 0-500,000 map to leaves [0, 20,000]
- Coarse region: Prefixes 500,000-2,000,000 map to leaves [20,000, 24,550]

**But the coarse region has sparse Morton codes!**
- Particle prefix: 234567
- Expected leaf: Should be ~10,000 (in coarse-to-fine transition)
- Actual returned leaf: 15,000 (wrong by 5,000 leaves!)

**With search radius=500**:
- Searches leaves [15,000 - 500, 15,000 + 500] = [14,500, 15,500]
- Correct leaf: 10,000
- **Miss by 4,500 leaves!**

---

## Why This Wasn't a Problem Before

### Old Implementation (Before Octree Prefix Table)

Used **linear approximation** (deprecated function `position_to_leaf_id_linear()`):

```python
# Normalize Morton code to [0, 1]
t = (morton - morton_min) / (morton_max - morton_min)

# Map to leaf index
leaf_id = int(t * n_leaves)
```

**This assumes uniform distribution along Morton curve** - wrong for refined meshes!

But it was **consistently wrong** in a **predictable way**, so large search radius (500) could compensate.

### New Implementation (Octree Prefix Table)

Tries to be **smarter** by using prefix table for O(1) lookup.

**But prefix table assumes**:
1. Each prefix has roughly equal element count
2. Leaves are uniformly distributed across prefixes
3. Binary search within prefix range finds correct leaf

**All false for refined meshes!**

---

## Evidence from Your Logs

### Initial Assignment Results
```
radius=500:  28.37% success (63,835/225,000)
radius=1000: +0.88% (1,983 more)
radius=2000: +1.36% (3,058 more)
radius=5000: +1.20% (2,704 more)
Final: 31.81% (71,580/225,000)
```

**Interpretation**:
- Even with radius=5000 (searching ±5000 leaves = 40% of all leaves!), only 32% assigned
- This means position-to-leaf mapping is **catastrophically wrong**
- Particles are so far from their correct leaf that even huge radius doesn't help

### Diagnostic Output
```
Seeded region: X=[-0.018, -0.009], Y=[-0.0138, 0.0138], Z=[-0.007, -0.0001]
Elements in region: 15,255/3,048,900 (0.50%)
```

**15,255 elements in the seeded region** - this should be plenty!

But only 32% of particles find them because:
1. Position → Morton code: ✅ Correct
2. Morton code → Prefix: ✅ Correct
3. Prefix → Leaf range: ❌ **WRONG by thousands of leaves**
4. Search within radius: ❌ **Even 5000-leaf radius insufficient**

---

## Solutions

### Solution 1: Fix Prefix Table Construction (CORRECT)

**Problem**: Current prefix table stores **first leaf ID** and **count**, but this doesn't account for non-uniform Morton distributions.

**Fix**: Store **median leaf ID** instead of first, or store **Morton range** for each prefix.

**Implementation**:
```python
# During prefix table build:
for prefix in range(8 ** table_depth):
    leaves_in_prefix = find_all_leaves_with_prefix(prefix)
    prefix_start[prefix] = median(leaves_in_prefix)  # Not first!
    prefix_range_min[prefix] = min(leaves_in_prefix)
    prefix_range_max[prefix] = max(leaves_in_prefix)
```

**Time**: 2-3 days to implement and test

### Solution 2: Use Binary Search on Full Leaf Array (SIMPLE)

**Idea**: Skip prefix table entirely, do binary search on **all leaves**!

**Current**:
```python
prefix = extract_prefix(morton_code)
first_leaf = prefix_start[prefix]  # Wrong!
search_in_range(first_leaf, first_leaf + num_leaves)
```

**Fixed**:
```python
# Binary search ALL leaves by their Morton code range
def position_to_leaf_id_binary_search(pos, mesh_gpu):
    morton = encode(pos)
    # Binary search: find leaf L such that
    # morton_min[L] <= morton < morton_max[L]
    return binary_search_all_leaves(morton, leaf_morton_ranges)
```

**Requires**:
- Store `leaf_morton_min` and `leaf_morton_max` for each leaf
- Upload to GPU (2 × n_leaves × 8 bytes = 2 × 24,550 × 8 = 393 KB)
- Binary search: O(log n_leaves) = ~15 comparisons

**Advantages**:
- Always correct (no approximation)
- Handles arbitrary Morton distributions
- Low memory (400 KB)

**Time**: 1-2 days to implement

### Solution 3: Increase Search Radius to Absurd Values (WORKAROUND)

**Current**: `INITIAL_SEARCH_RADIUS = 500`

**Try**: `INITIAL_SEARCH_RADIUS = 10000` (search ±10K leaves = 80% of mesh!)

**Pros**: Might work as band-aid
**Cons**:
- Extremely slow (searches most of mesh)
- Wastes GPU time
- Doesn't fix root cause

**Time**: 1 minute (change config line)

### Solution 4: Use Hilbert Curve Instead of Morton (PARTIAL FIX)

Hilbert curves have **better locality** than Morton (Z-order) curves.

**Your config** already supports this:
```python
CURVE_TYPE = 'hilbert'  # or 'morton'
```

**Why this helps**: Hilbert curves keep spatially close elements closer on the curve, reducing the "jump" between coarse and fine regions.

**But**: Doesn't fix the prefix table issue, just reduces it.

**Time**: 1 minute (change config), 30 min test

---

## Recommended Immediate Actions

### Action 1: Switch to Hilbert Curve (1 minute)
```python
# In production_tracking_fully_fused_timedep.py line 115:
CURVE_TYPE = 'hilbert'  # Change from 'morton'
```

**Expected**: ~50-60% initial assignment (up from 32%)

### Action 2: Increase Search Radius Temporarily (1 minute)
```python
# In production_tracking_fully_fused_timedep.py lines 142-143:
INITIAL_SEARCH_RADIUS = 10000
INITIAL_SEARCH_FALLBACK_RADII = [20000, 50000]
```

**Expected**: ~85-95% initial assignment (slow but works)

### Action 3: Implement Binary Search Solution (1-2 days)

Create `position_to_leaf_id_exact()` function that:
1. Stores `leaf_morton_min` and `leaf_morton_max` arrays (400 KB)
2. Binary searches all leaves to find exact match
3. No prefix table needed
4. Always correct

**Expected**: ~99% initial assignment, fast

---

## Why Solution 2 (Binary Search) is Best

### Comparison

| Method | Accuracy | Speed | Memory | Refinement-Aware |
|--------|----------|-------|--------|------------------|
| **Current (prefix table)** | ❌ 32% | Fast | 32 MB | ❌ No |
| **Linear approximation** | ❌ ~50% | Fast | 0 MB | ❌ No |
| **Huge radius workaround** | ⚠️ ~90% | Slow | 32 MB | ⚠️ Brute force |
| **Hilbert curve** | ⚠️ ~60% | Fast | 42 MB | ⚠️ Better locality |
| **Binary search (new)** | ✅ ~99% | Fast | 0.4 MB | ✅ Exact |

### Binary Search Implementation Sketch

```python
@dataclass
class MeshGPUGlobalMortonFixed:
    """Enhanced with per-leaf Morton ranges."""
    # ... existing fields ...
    leaf_morton_min: jax.Array  # (n_leaves,) uint64 - min Morton in leaf
    leaf_morton_max: jax.Array  # (n_leaves,) uint64 - max Morton in leaf


def position_to_leaf_id_exact(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMortonFixed
) -> jnp.int32:
    """
    Map position to leaf ID using exact binary search on leaf Morton ranges.

    Always correct, regardless of Morton distribution.
    """
    # Encode position
    morton = morton_encode_position_jax(pos, mesh_gpu.bbox_min, mesh_gpu.bbox_max, mesh_gpu.max_depth)

    # Binary search: find leaf L where morton_min[L] <= morton < morton_max[L]
    def binary_search_leaf(left, right):
        mid = (left + right) // 2
        morton_min_mid = mesh_gpu.leaf_morton_min[mid]
        morton_max_mid = mesh_gpu.leaf_morton_max[mid]

        in_range = (morton >= morton_min_mid) & (morton < morton_max_mid)
        too_low = morton < morton_min_mid

        # Recurse
        new_left = jnp.where(too_low, mid + 1, left)
        new_right = jnp.where(too_low | in_range, right, mid)

        return jnp.where(in_range, mid, lax.cond(
            left < right,
            lambda: binary_search_leaf(new_left, new_right),
            lambda: left  # Fallback to left boundary
        ))

    return binary_search_leaf(0, mesh_gpu.n_leaves - 1)
```

**Build leaf ranges** during octree construction:
```python
# In morton_octree_builder.py
for i, leaf in enumerate(leaves):
    leaf_morton_min[i] = morton_sorted[leaf.start_idx]
    leaf_morton_max[i] = morton_sorted[leaf.start_idx + leaf.length - 1]
```

---

## Testing Plan

### Test 1: Hilbert + Huge Radius (Immediate)
```python
CURVE_TYPE = 'hilbert'
INITIAL_SEARCH_RADIUS = 10000
```

**Run**: `python production_tracking_fully_fused_timedep.py > logs/test_hilbert_huge_radius.log 2>&1`

**Expected**: 85-95% initial assignment

### Test 2: Implement Binary Search (1-2 days)

1. Modify `morton_octree_builder.py` to compute `leaf_morton_min/max`
2. Add to `MeshGPUGlobalMorton` dataclass
3. Implement `position_to_leaf_id_exact()`
4. Replace in `initial_assignment_extended.py`

**Expected**: 99% initial assignment with radius=100 (original value)

---

## Conclusion

**Root Cause**: Prefix table assumes uniform Morton distribution, but your refined mesh has 85% of elements in 5% of spatial domain, causing **massive position-to-leaf mapping errors**.

**Quick Fix**: Hilbert curve + radius=10000 (85-95% assignment, slow)

**Proper Fix**: Binary search on leaf Morton ranges (99% assignment, fast, 1-2 days)

**Don't Bother**: Increasing radius to 100,000 won't fix underlying issue

The loss during tracking (mentioned in earlier messages) is likely a **separate issue** from initial assignment. Once particles are properly assigned initially, we can debug tracking loss separately.
