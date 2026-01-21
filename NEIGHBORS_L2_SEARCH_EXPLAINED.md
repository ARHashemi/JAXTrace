# Neighbors L2 Search - How It Works and Your Proposed Improvement

**Date**: 2026-01-20
**Context**: Understanding 'neighbors' L2 search vs proposed sequential search strategy

---

## Your Question

> "How does the 'neighbor' L2 search work? I'm thinking if we can perform sequential L2 but first search the leaf contains current element (centroid) or query position, then fallback to search leaf contains the neighbor elements. It can also be radius based. Is it meaningful or beneficial or the same as implemented 'neighbor' L2?"

---

## How the Implemented 'neighbors' L2 Search Works

### High-Level Algorithm

**From** [morton_global_search.py:647-758](jaxtrace/gpu/search/morton_global_search.py#L647-L758):

The 'neighbors' search uses a **3×3×3 octant search** strategy:

1. **Compute Morton code** for query position
2. **Decode to octant coordinates** (cx, cy, cz) at table_depth (typically depth 7)
3. **Find 26 neighbor octants** in 3×3×3 grid: all combinations of (cx±1, cy±1, cz±1)
4. **Encode each neighbor** back to Morton prefix
5. **Search each of the 27 octants** (26 neighbors + center)
6. **For each octant**: Look up leaves with that prefix, search up to 3 leaves per octant

### Key Differences from Radius-Based Search

| Feature | Radius-Based | Neighbors-Based |
|---------|--------------|-----------------|
| **Search strategy** | Linear along Morton curve | Spatial 3×3×3 grid |
| **Octants searched** | 2N+1 leaves (consecutive) | 27 octants (spatial neighbors) |
| **Handles adaptive depth** | ❌ No - misses different-depth neighbors | ✅ Yes - searches all spatial neighbors |
| **Handles Morton discontinuities** | ❌ No - consecutive leaves may be far in 3D | ✅ Yes - explicitly computes spatial neighbors |
| **Performance** | Fast (linear scan) | Slower (27× prefix lookups) |

### Detailed Implementation

**Step 1: Position → Morton Code → Octant Coordinates**

```python
# From morton_global_search.py:670-681
morton_query = morton_encode_position_jax(pos, bbox_min, bbox_max, max_depth)
# morton_query = 63-bit Morton code (left-aligned uint64)

# Decode to octant coordinates at table_depth (e.g., depth 7)
# This gives octant position in 128×128×128 grid
cx, cy, cz = decode_morton_prefix_jax(morton_query, table_depth)
# Example: morton_query → (cx=63, cy=84, cz=101) in 128³ grid
```

**Step 2: Generate 26 Neighbor Octants**

**From** [morton_neighbors.py:169-237](jaxtrace/gpu/search/morton_neighbors.py#L169-L237):

```python
# Generate 27 neighbor prefixes (3×3×3 cube)
neighbor_prefixes = get_26_neighbor_prefixes_jax(
    center_prefix=morton_query,
    depth=table_depth,
    max_coord=(2**table_depth - 1)
)

# Process:
# For each offset (dx, dy, dz) in {-1, 0, +1}³:
#   1. Compute neighbor coordinates: (cx+dx, cy+dy, cz+dz)
#   2. Clamp to valid range [0, 2^depth - 1]
#   3. Encode back to Morton prefix
# Returns array of 27 Morton prefixes
```

**Step 3: Search Each Octant**

**From** [morton_global_search.py:696-758](jaxtrace/gpu/search/morton_global_search.py#L696-L758):

```python
# For each of the 27 neighbor octants:
for octant in range(27):
    neighbor_prefix = neighbor_prefixes[octant]

    # Look up leaves with this prefix in prefix table
    prefix_idx = neighbor_prefix >> shift_amount
    first_leaf = prefix_start[prefix_idx]
    num_leaves = prefix_length[prefix_idx]

    # Search up to 3 leaves in this octant
    for leaf_offset in range(3):
        leaf_id = first_leaf + leaf_offset
        if leaf_offset < num_leaves:
            elem_id = search_in_leaf_global(pos, leaf_id, mesh_gpu)
            if elem_id >= 0:
                return elem_id  # Found!
```

### Example: What Gets Searched

**Scenario**: Particle at position `(0.500, 0.500, 0.500)` at table_depth=7

1. **Position → Morton code** → octant (64, 64, 64) in 128³ grid
2. **Generate 27 neighbors**:
   - (63,63,63), (63,63,64), (63,63,65)
   - (63,64,63), (63,64,64), (63,64,65)
   - ... 21 more octants ...
   - (65,65,65)

3. **Search all 27 octants** regardless of where they appear in Morton order
4. **Each octant may have 1-3 leaves** at different depths (adaptive octree)

**Key insight**: The 27 octants are **spatially guaranteed neighbors**, NOT consecutive in Morton order!

---

## Your Proposed Sequential Search Strategy

### What You're Suggesting

> "Perform sequential L2 but first search the leaf contains current element (centroid) or query position, then fallback to search leaf contains the neighbor elements. It can also be radius based."

**Interpretation**:
1. **Tier 1**: Search the leaf containing **current element's centroid**
2. **Tier 2**: Search leaves containing **face neighbor elements' centroids**
3. (Optional) **Tier 3**: Search radius-based if still not found

### Critical Question: Which "Neighbors"?

Your proposal has an important ambiguity:

**Option A**: **L1 Face Neighbors** (Mesh connectivity neighbors)
- These are the 1-4 elements sharing a face with current element
- Already handled by **L1 search** with `N_HOPS=5`!

**Option B**: **Spatial Octant Neighbors** (What you likely mean)
- Search leaf containing current element first
- Then search leaves of 26 spatially adjacent octants
- **This is EXACTLY what 'neighbors' search does!**

---

## Comparison: Your Proposal vs Implemented 'neighbors' Search

### Is Your Proposal Already Implemented?

**SHORT ANSWER**: **YES**, but with one difference!

The implemented 'neighbors' search DOES search spatially adjacent octants. However:

| Feature | Your Proposal | Implemented 'neighbors' |
|---------|---------------|-------------------------|
| **Tier 1** | Search leaf containing **current element** | Search leaf containing **query position** |
| **Tier 2** | Search leaves of neighbor octants | Search leaves of 26 neighbor octants |
| **Center reference** | Current element centroid | Query position |

### Key Difference: Current Element vs Query Position

**Your proposal**: "first search the leaf contains current element (centroid)"
- Use current element's centroid to find center leaf
- Then search neighbor octants around current element

**Implemented**: "search the leaf contains query position"
- Use query position to find center leaf
- Then search neighbor octants around query position

**Why this matters**:

After an RK4 step, the particle has moved from its current element. The query position may be:
- In a **different leaf** than current element's centroid
- Close to a **leaf boundary**
- In a **different depth octant** (coarse/fine transition)

**Which is better?**

**Query position is better** because:
1. **Spatial locality**: The particle IS at query position, not at element centroid
2. **Handles large displacement**: RK4 can move particle far from current element
3. **Correct for velocity discontinuities**: Element centroid may be in completely wrong region

**Example**:
```
Current element centroid: (0.500, 0.500, 0.500) → Leaf 3200 (depth 6, coarse)
After RK4 step:           (0.502, 0.501, 0.499) → Leaf 8150 (depth 8, fine)

Using current element → searches octants around (0.500, 0.500, 0.500)
Using query position  → searches octants around (0.502, 0.501, 0.499) ✅
```

The particle is more likely to be in an element near the query position!

---

## Performance Analysis

### Implemented 'neighbors' Search

**From production logs** (based on your 3.3M element mesh):

```
Tier 1 (3×3×3 search): 67% particles succeed
Tier 2 (5×5×5 fallback): 33% particles need extended search
Total overhead: ~2× vs radius-based search
```

**Why some particles need Tier 2 (5×5×5 fallback)**:

The enhanced version [morton_global_search.py:887-919](jaxtrace/gpu/search/morton_global_search.py#L887-L919) adds a second tier:

1. **Tier 1**: 3×3×3 search (27 octants) - fast
2. **Tier 2**: 5×5×5 outer shell (98 octants) - boundary fallback

This handles **Morton Z-order discontinuities at octree boundaries**.

### Your Proposed Sequential Search (With Current Element)

**Potential performance**:

✅ **Advantage**: If particle hasn't moved far, searching current element's leaf first might be faster
❌ **Disadvantage**: After large displacement, current element's leaf may be useless

**Expected behavior**:
- **Small timesteps** (DT=1e-6): Query position ≈ current element centroid → Similar performance
- **Large timesteps** (DT=1e-3): Query position ≠ current element centroid → Worse performance

### Hybrid Approach?

**Possible optimization**:
```python
# Tier 1: Search leaf containing current element (fast check)
elem_id = search_in_leaf_of_element(pos, current_elem, mesh_gpu)
if elem_id >= 0:
    return elem_id

# Tier 2: Search 3×3×3 around query position (current implementation)
elem_id = search_L2_morton_neighbors_single(pos, mesh_gpu)
if elem_id >= 0:
    return elem_id

# Tier 3: Search 5×5×5 around query position (fallback)
elem_id = search_5x5x5_outer_shell(pos, mesh_gpu, current_elem, False)
return elem_id
```

**When this helps**:
- Current element's leaf is very large (coarse region)
- Particle moved slightly within same large octant
- Saves 27× prefix lookups

**When this hurts**:
- Current element's leaf is small (fine region)
- Particle crossed octant boundary
- Wastes time searching wrong leaf first

---

## Why Retention Still Stops at ~95%?

Given that 'neighbors' search DOES search spatial neighbors (not just Morton consecutive leaves), why does retention still plateau?

### Possible Causes

1. **L1 Search Limitation** (Most likely!)
   - L1 uses face neighbor traversal with `N_HOPS=5`
   - Face neighbors DON'T cross coarse/fine boundaries well
   - After 5 hops, may still be in wrong depth region

2. **3×3×3 Octant Coverage Insufficient**
   - At table_depth=7, each octant spans 1/128³ of domain
   - Large RK4 displacement might move particle >1 octant away
   - 5×5×5 fallback helps, but may still miss distant particles

3. **Velocity Field Discontinuities**
   - Mesh has refined and coarse regions with different velocity scales
   - RK4 might integrate velocity from wrong region
   - Particle ends up in unexpected location

4. **Numerical Precision (float32)**
   - Barycentric checks fail at element boundaries
   - Particle reported as "not in element" due to roundoff

5. **Mesh Quality Issues**
   - Extremely stretched elements (262K× volume variation in your mesh!)
   - Point-in-tet checks numerically unstable for degenerate elements
   - Some elements effectively "invisible" to search

### Why Radius-Based Search Also Fails

From your diagnostic log:
- radius=5 (11 leaves): **86.67%**
- radius=64 (129 leaves): **87.89%**
- radius=100 (201 leaves): **89.00%**

Even radius=100 (201 leaves!) only reaches 89%, not 95%.

**Root cause**: The 107 large Morton jumps (>10% domain diagonal) mean:
- Some spatially close leaves are **>100 leaves apart** in Morton order
- Radius-based search misses them
- 'neighbors' search should find them (it's spatial, not linear)

---

## Recommendations

### 1. Test Your Hybrid Approach

**Modify** [morton_global_search.py](jaxtrace/gpu/search/morton_global_search.py) to add Tier 0:

```python
def search_L2_morton_hybrid_single(
    pos: jax.Array,
    current_elem: jnp.int32,
    mesh_gpu: MeshGPUGlobalMorton
) -> jnp.int32:
    """
    Hybrid search: current element leaf first, then spatial neighbors.
    """
    # Tier 0: Search leaf containing current element (if valid)
    has_current = current_elem >= 0
    if has_current:
        # Get current element's centroid
        elem_centroid = mesh_gpu.node_positions[mesh_gpu.connectivity[current_elem]].mean(axis=0)

        # Find leaf containing centroid
        current_leaf_id = position_to_leaf_id_octree(elem_centroid, mesh_gpu)

        # Search that leaf
        elem_id = search_in_leaf_global(pos, current_leaf_id, mesh_gpu)
        if elem_id >= 0:
            return elem_id

    # Tier 1: Standard 3×3×3 search around query position
    elem_id = search_L2_morton_neighbors_single(pos, mesh_gpu)
    if elem_id >= 0:
        return elem_id

    # Tier 2: 5×5×5 fallback
    elem_id = search_5x5x5_outer_shell(pos, mesh_gpu, current_elem, False)
    return elem_id
```

**Expected result**:
- Faster for small timesteps (particles stay in same leaf)
- Similar retention (spatial neighbors still searched if needed)

### 2. Increase L1 Search Depth

The real bottleneck might be L1, not L2!

**Current**: `N_HOPS=5` → searches ~1,365 elements
**Try**: `N_HOPS=7` → searches ~5,461 elements (4× more!)

**From** [production_tracking_fully_fused_timedep.py:180-181](production_tracking_fully_fused_timedep.py#L180-L181):

```python
# Change:
N_HOPS = 5

# To:
N_HOPS = 7  # Warning: 4× slower L1, but may improve retention
```

**Expected**: +3-5% retention, but ~2-3× overall slowdown.

### 3. Diagnose Lost Particles

The key question: **WHERE are the 5% lost particles?**

Create a diagnostic that:
1. Captures lost particle positions
2. Finds nearest element to each lost position (brute force CPU search)
3. Analyzes WHY search failed to find that element

**Script**:
```python
# After tracking
lost_mask = element_ids_gpu < 0
lost_positions = np.array(positions_gpu[lost_mask])

# For each lost particle (CPU)
for pos in lost_positions[:10]:  # Sample 10
    # Brute force: Find nearest element
    elem_centroids = node_positions[connectivity].mean(axis=1)
    distances = np.linalg.norm(elem_centroids - pos, axis=1)
    nearest_elem = distances.argmin()
    dist = distances[nearest_elem]

    print(f"Lost particle at {pos}")
    print(f"  Nearest element: {nearest_elem}, distance: {dist:.6f}")

    # Find leaf of nearest element
    centroid = elem_centroids[nearest_elem]
    morton_code = morton_encode_position(centroid, bbox_min, bbox_max, 21)
    leaf_id = find_leaf_for_morton_code(morton_code, octree_struct)

    print(f"  Nearest element in leaf: {leaf_id}")

    # Check what query position maps to
    query_morton = morton_encode_position(pos, bbox_min, bbox_max, 21)
    query_leaf = find_leaf_for_morton_code(query_morton, octree_struct)

    print(f"  Query position maps to leaf: {query_leaf}")
    print(f"  Leaf distance: {abs(query_leaf - leaf_id)}")
```

**This will reveal**:
- Are lost particles >3 octants away from nearest element?
- Are they at coarse/fine boundaries?
- Do they map to empty leaves?

### 4. Test 'hierarchical' Search Method

The code mentions a hierarchical search method [morton_global_search.py:1012-1038](jaxtrace/gpu/search/morton_global_search.py#L1012-L1038):

```python
L2_SEARCH_METHOD = 'hierarchical'
```

**This searches at MULTIPLE depths** (depth 7 AND depth 6), which might handle coarse/fine boundaries better than fixed-depth 3×3×3 search.

---

## Summary

### Your Question: Is Your Proposal Already Implemented?

**YES**, mostly! The 'neighbors' search DOES:
- ✅ Search leaves of spatially adjacent octants
- ✅ Handle Morton discontinuities (uses spatial arithmetic, not linear scan)
- ✅ Handle adaptive depth (searches all octants regardless of depth)

**Difference**: It searches around **query position**, not **current element centroid**.

### Is Your Proposal Beneficial?

**Maybe!** Adding Tier 0 (search current element's leaf first) could:
- ✅ Speed up small-timestep tracking
- ❌ Hurt large-timestep tracking
- ⚠️ Probably not improve retention (spatial neighbors already searched)

### What to Try Next

1. **Increase L1 depth** (`N_HOPS=7`) - Most likely to improve retention
2. **Diagnose lost particles** - Find out WHERE and WHY they're lost
3. **Test 'hierarchical' search** - Multi-depth spatial search
4. **Hybrid Tier 0** - Add current element leaf as fast path

### Root Cause of 95% Plateau

**Not yet determined!** Possibilities:
- L1 search insufficient (doesn't cross coarse/fine boundaries well)
- 3×3×3 octant coverage too small for large displacements
- Mesh quality issues (degenerate elements, numerical precision)
- Velocity field discontinuities (particles jump to unexpected locations)

**Next step**: Run the lost particle diagnostic to identify root cause!

---

## References

- [morton_global_search.py:647-758](jaxtrace/gpu/search/morton_global_search.py#L647-L758) - 3×3×3 neighbors search
- [morton_global_search.py:887-919](jaxtrace/gpu/search/morton_global_search.py#L887-L919) - 5×5×5 enhanced neighbors
- [morton_neighbors.py:169-237](jaxtrace/gpu/search/morton_neighbors.py#L169-L237) - Spatial neighbor arithmetic
- [MORTON_SEARCH_EXPLAINED.md](MORTON_SEARCH_EXPLAINED.md) - Morton search fundamentals
- [SEARCH_RETENTION_ANALYSIS.md](SEARCH_RETENTION_ANALYSIS.md) - Diagnostic results
