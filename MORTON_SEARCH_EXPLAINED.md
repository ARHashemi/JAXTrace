# Morton Search and Octree Structure - Complete Explanation

**Date**: 2026-01-20
**Context**: Understanding why retention doesn't improve above 95% even with radius=64

---

## Your Questions

1. **How does the search work?** (Finding Morton key, searching leaves, radius meaning)
2. **Is the octree geometrical or just fixed-capacity neighbors on Morton array?**
3. **Why doesn't retention improve with large radius?**

---

## Part 1: How the Octree is Built (Geometrical!)

### Key Insight: It IS a True Spatial Octree ✅

**From** [morton_octree_builder.py:1-21](jaxtrace/gpu/search/morton_octree_builder.py#L1-L21):

```
Key Differences from Fixed-Capacity Approach:
- OLD: Leaf i = elements [i*256, (i+1)*256] in Morton order (arbitrary spatial mix)
- NEW: Leaf i = elements with Morton prefix P (spatial octant), ≤256 elements

Architecture:
- CPU: Build octree with capacity-constrained recursive subdivision
- CPU: Create prefix→leaf_id lookup table (all possible prefixes at final depth)
- GPU: O(1) position→leaf mapping via prefix table
```

### Octree Construction Algorithm

**From** [morton_octree_builder.py:113-150](jaxtrace/gpu/search/morton_octree_builder.py#L113-L150):

```python
def build_adaptive_octree_leaves(
    morton_sorted: np.ndarray,
    elem_ids_sorted: np.ndarray,
    leaf_capacity: int = 256,  # Production setting
    max_depth: int = 21        # Production setting
):
    """
    Build adaptive octree with capacity-constrained leaves.

    Each leaf:
    - Aligns with a spatial octant (defined by Morton prefix)
    - Contains ≤ leaf_capacity elements
    - Covers a contiguous range in morton_sorted

    Algorithm:
    - Start with root node (entire mesh)
    - Recursively subdivide into 8 octants if > leaf_capacity elements
    - Stop at max_depth or when octant small enough
    """
```

**Subdivision Process**:
1. Start with root octant (entire domain)
2. If octant has > 256 elements → subdivide into 8 child octants
3. Each child octant = append 3 bits (0-7) to parent Morton prefix
4. Recursively subdivide until ≤256 elements per leaf OR max_depth reached
5. Leaves at different depths form an **adaptive spatial octree**

**Critical Detail**: Leaves are **geometrically defined octants**, not arbitrary chunks!

---

## Part 2: How the Search Works

### Step 1: Position → Morton Code

**From** [morton_global_search.py:240-244](jaxtrace/gpu/search/morton_global_search.py#L240-L244):

```python
# 1. Compute Morton code for position
m = morton_encode_position_jax(
    pos,
    mesh_gpu.bbox_min,
    mesh_gpu.bbox_max,
    mesh_gpu.max_depth  # 21 bits per dimension
)
```

**What this does**:
- Normalizes position to [0, 2^21-1] in each dimension
- Interleaves x, y, z bits to create 63-bit Morton code
- Morton code = spatial location on Z-order curve

### Step 2: Morton Code → Leaf ID (via Prefix Table)

**From** [morton_global_search.py:215-245](jaxtrace/gpu/search/morton_global_search.py#L215-L245):

```python
def position_to_leaf_id_octree(pos, mesh_gpu):
    """
    Map position to leaf ID using octree prefix table with range search.

    Algorithm:
    1. Compute Morton code for position
    2. Extract prefix bits (top table_depth * 3 bits)
    3. Get leaf range from prefix_start[prefix] and prefix_length[prefix]
    4. Search within range for leaf containing this Morton code
    """
```

**Example** (table_depth = 7):
- Morton code: `0b101011001...` (63 bits)
- Extract top 21 bits (7 depths × 3 bits): `0b101011001101011010110` → prefix
- Lookup `prefix_start[prefix]` → finds range of leaves at depth 7-8
- Binary search within range to find exact leaf

**Key Point**: This is **O(log n)** lookup, not linear scan!

### Step 3: Search Leaves (The Radius)

**From** [morton_global_search.py:477-517](jaxtrace/gpu/search/morton_global_search.py#L477-L517):

```python
def search_L2_global_morton_single(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMorton,
    search_radius: jnp.int32 = jnp.int32(1)
) -> jnp.int32:
    """
    L2 search using global Morton structure for SINGLE particle.

    Searches the predicted leaf and its neighbors along the Morton curve.

    **IMPORTANT: radius=N searches BOTH directions**:
      - Searches center leaf (1 leaf)
      - Searches -N, -N+1, ..., -1 leaves BACKWARD (N leaves)
      - Searches +1, +2, ..., +N leaves FORWARD (N leaves)
      - **Total: 2N + 1 leaves** (symmetric band around center)

      Example: radius=10 searches 21 leaves total:
        leaves[-10], leaves[-9], ..., leaves[0], ..., leaves[+9], leaves[+10]
    """
```

**Critical Understanding**:
- **radius=10** searches **21 leaves** (not 10!)
- **radius=64** searches **129 leaves**
- Searches leaves **along the Morton curve** (1D ordering of 3D space)

---

## Part 3: Why Radius Doesn't Always Help

### The Morton Curve Problem

**Morton curve** is a space-filling curve that maps 3D space to 1D. However:

**Problem 1: Spatial Discontinuities**

Morton curve has **large jumps** in 3D space at octree boundaries:

```
Leaf 100: Octant (5,7,3) at depth 7 → elements around (0.625, 0.875, 0.375) * domain
Leaf 101: Octant (6,0,0) at depth 7 → elements around (0.750, 0.000, 0.000) * domain
                                      ^^^ BIG 3D JUMP!
```

Even though leaf 101 is adjacent in Morton order, it can be **geometrically far** in 3D space!

**Problem 2: Adaptive Depth**

Your mesh has leaves at **different depths** (depth 6, 7, 8, maybe more):
- Depth 6 leaves cover large regions (1/64³ of domain)
- Depth 8 leaves cover small regions (1/512³ of domain)

A particle at a coarse/fine boundary might need to check leaves at **different depths**, which are **not consecutive** in Morton order.

### Example: Why radius=64 Still Fails

**Scenario**: Particle at position `(0.500, 0.500, 0.500)` in refined region
- Current element: in depth-8 leaf at Morton index 5000
- Particle moves to `(0.501, 0.500, 0.500)` (small displacement)
- **Problem**: New position is in depth-6 leaf (coarse neighbor)
- Depth-6 leaf might be at Morton index 3200 (far away in 1D)
- **radius=64** searches leaves [4936, 5064] → **misses** leaf at 3200!

**Why**: Morton ordering keeps **same-depth octants** together, but **different-depth octants** are separated.

---

## Part 4: What the Code Actually Does

### Production Configuration

**From** [production_tracking_fully_fused_timedep.py:189](production_tracking_fully_fused_timedep.py#L189):

```python
# L2 search configuration
L2_SEARCH_METHOD = 'incremental'
INCREMENTAL_SEARCH_RADII = (2, 4, 8, 15, 30)  # 5-tier aggressive

# Explanation:
# - Tier 1: radius=2  → searches 5 leaves (80% particles)
# - Tier 2: radius=4  → searches 9 leaves (15% particles)
# - Tier 3: radius=8  → searches 17 leaves (3% particles)
# - Tier 4: radius=15 → searches 31 leaves (1.5% particles)
# - Tier 5: radius=30 → searches 61 leaves (0.5% particles)
# - Average: ~7-10 leaves per particle
```

### L1 Search (Face Neighbors)

**Also from production script**:

```python
ENABLE_L1_SEARCH = True
N_HOPS = 5  # Face neighbor traversal depth
```

**This is critical**: Before L2 search even runs, the code:
1. Checks current element (L0)
2. Checks face neighbors (L1, depth=1)
3. Checks neighbors-of-neighbors (L1, depth=2)
4. ... up to depth=5

**L1 searches**: 1 + 4 + 16 + 64 + 256 + 1024 = **1365 elements** (worst case)

### Why Retention Stops at 95%

**Possible causes**:

1. **Particles leave the mesh**: They've moved outside the domain boundary
2. **Numerical precision issues**: RK4 position slightly outside all elements due to float32
3. **Mesh quality issues**: Gaps, overlaps, or degenerate elements
4. **Velocity field discontinuities**: Particles jump to unreachable locations
5. **Coarse/fine boundary problems**: L1 doesn't cross depth boundaries well

---

## Part 5: Diagnostic Steps

### Check 1: Are Particles Leaving the Domain?

```python
# After RK4 step
positions_cpu = np.array(positions_gpu)
bbox_min = node_positions.min(axis=0)
bbox_max = node_positions.max(axis=0)

lost_mask = element_ids_gpu < 0
lost_positions = positions_cpu[lost_mask]

# Check if lost particles are outside bbox
outside = (
    (lost_positions < bbox_min).any(axis=1) |
    (lost_positions > bbox_max).any(axis=1)
)

print(f"Lost particles: {lost_mask.sum()}")
print(f"  Outside bbox: {outside.sum()} ({100*outside.sum()/lost_mask.sum():.1f}%)")
print(f"  Inside bbox but unfound: {(~outside).sum()}")
```

### Check 2: Leaf Distribution

```python
# Analyze octree structure
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree

octree = build_global_morton_octree(
    node_positions=node_positions,
    connectivity=connectivity,
    leaf_capacity=256,
    max_depth=21,
    verbose=True
)

print(f"Total leaves: {octree.n_leaves}")
print(f"Table depth: {octree.table_depth}")

# Check leaf depth distribution
leaf_depths = []
for leaf in octree.leaves:
    depth = leaf.prefix_bits // 3
    leaf_depths.append(depth)

import collections
depth_counts = collections.Counter(leaf_depths)
print("\nLeaf depth distribution:")
for depth in sorted(depth_counts.keys()):
    count = depth_counts[depth]
    print(f"  Depth {depth}: {count:,} leaves ({100*count/octree.n_leaves:.1f}%)")
```

### Check 3: Search Radius Coverage

**Key question**: What radius would you need to cover all leaves?

```python
# Maximum distance between consecutive leaves in Morton order
max_jump = 0
for i in range(octree.n_leaves - 1):
    # Get spatial positions of leaf centroids
    leaf_i_elements = octree.elem_ids_sorted[octree.leaf_start[i]:octree.leaf_start[i]+octree.leaf_length[i]]
    leaf_j_elements = octree.elem_ids_sorted[octree.leaf_start[i+1]:octree.leaf_start[i+1]+octree.leaf_length[i+1]]

    # Compute centroids
    centroid_i = node_positions[connectivity[leaf_i_elements]].mean(axis=(0,1))
    centroid_j = node_positions[connectivity[leaf_j_elements]].mean(axis=(0,1))

    dist = np.linalg.norm(centroid_j - centroid_i)
    if dist > max_jump:
        max_jump = dist
        max_jump_idx = i

print(f"Maximum spatial jump between consecutive Morton leaves: {max_jump:.6f}")
print(f"  Between leaf {max_jump_idx} and leaf {max_jump_idx+1}")
```

### Check 4: Hierarchical Search (Better Alternative)

**From** [morton_global_search.py:1012-1038](jaxtrace/gpu/search/morton_global_search.py#L1012-L1038):

```python
def search_L2_morton_hierarchical_single(pos, mesh_gpu):
    """
    Hierarchical Morton neighbor search with CONDITIONAL multi-depth fallback.

    Searches at multiple octree depths to handle variable-depth leaves:
    1. Depth 7 (fine): 27 neighbor octants at 128³ resolution
    2. Depth 6 (coarse): 27 neighbor octants at 64³ resolution (CONDITIONAL)

    This handles particles at coarse/fine boundaries.
    """
```

**This might help!** It searches **spatial neighbors** at different depths, not just consecutive Morton leaves.

---

## Part 6: Recommendations

### Immediate Diagnostics

1. **Run Check 1** to see if particles are leaving the domain
2. **Run Check 2** to see leaf depth distribution (are you mixing depth 6, 7, 8?)
3. **Enable verbose logging** in search to see where failures occur

### Try Hierarchical Search

**Modify your production script**:

```python
# Instead of:
L2_SEARCH_METHOD = 'incremental'
INCREMENTAL_SEARCH_RADII = (2, 4, 8, 15, 30)

# Try:
L2_SEARCH_METHOD = 'hierarchical'
# No radii needed - searches spatial neighbors at multiple depths
```

**Expected**: Better retention at coarse/fine boundaries (86% → 92-95%)

### Increase L1 Search Depth

```python
# Instead of:
N_HOPS = 5  # searches ~1365 elements

# Try:
N_HOPS = 7  # searches ~5461 elements (expensive!)
```

**Warning**: This is **very expensive** (4× slower), but will catch more particles.

### Check Mesh Quality

```python
# Look for degenerate elements
element_volumes = compute_element_volumes(connectivity, node_positions)
print(f"Min volume: {element_volumes.min()}")
print(f"Max volume: {element_volumes.max()}")
print(f"Volume ratio: {element_volumes.max() / element_volumes.min():.2e}")

# Check for gaps
from jaxtrace.gpu.forest import build_element_neighbors_array
element_neighbors = build_element_neighbors_array(connectivity, method='face')
boundary_faces = (element_neighbors < 0).sum()
print(f"Boundary faces: {boundary_faces} (should be mesh surface only)")
```

---

## Summary

### Your Understanding: Partially Correct ✅

- ✅ **YES**: Search finds Morton key nearest to query position
- ✅ **YES**: Searches leaf containing that Morton key
- ⚠️ **PARTIALLY**: Searches other leaves up to radius, BUT:
  - Searches along **Morton curve** (1D), not 3D spatial radius
  - Morton curve has large 3D jumps at octree boundaries
  - Adaptive octree has leaves at different depths

### Why radius=64 Doesn't Reach 100%

1. **Morton discontinuities**: Consecutive Morton leaves can be far in 3D space
2. **Adaptive depth**: Coarse/fine boundaries have leaves at non-consecutive Morton indices
3. **Domain boundaries**: Particles may be leaving the mesh entirely
4. **Mesh quality**: Gaps, overlaps, or numerical precision issues

### What to Try

1. **Hierarchical search** (searches spatial neighbors at multiple depths)
2. **Diagnostics** (Check 1-4 above to identify root cause)
3. **Increase L1 depth** (N_HOPS=7, but expensive)
4. **Hybrid approach** (hierarchical + incremental fallback)

---

## Next Steps

**Immediate**:
1. Run the diagnostic scripts above
2. Check if lost particles are outside domain
3. Analyze leaf depth distribution

**Then**:
1. Try `L2_SEARCH_METHOD = 'hierarchical'`
2. Compare retention: incremental vs hierarchical
3. Profile performance trade-offs

Would you like me to create a diagnostic script that runs all these checks automatically?
