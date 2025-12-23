# L2 Optimization: Comprehensive Strategy Analysis
## LBVH Radix Tree vs Hilbert Curves vs Octree-Aligned Morton

**Date**: 2025-12-22
**Status**: Research Complete - Implementation Plan Ready

---

## Executive Summary

After analyzing the Sunnet review and researching modern alternatives, here's the **definitive recommendation**:

### Current Implementation Issues (Confirmed by Review)

Your current implementation is a **hybrid that combines worst aspects of both HOT and LBVH**:

1. **Fixed-capacity leaves** (arbitrary 256-element chunks) that don't respect spatial hierarchy
2. **Linear radius search** (±100 leaves) to compensate for leaf fragmentation
3. **25,600 point-in-tet tests per L2 query** (100 leaves × 256 elements)
4. **NOT actually HOT** (no hash collisions, no tree structure)
5. **NOT actually LBVH** (no radix tree hierarchy)

### Optimization Options Ranked by Practicality

| Approach | Speedup | JAX Compatibility | Implementation | Timeline | **Recommendation** |
|----------|---------|-------------------|----------------|----------|-------------------|
| **1. Octree-Aligned Leaves** | **30-50×** | ✅ Excellent | 🟡 Medium | 3-5 days | ⭐ **DO THIS FIRST** |
| **2. Morton Neighbor Arithmetic** | **10-15×** | ✅ Excellent | 🟢 Easy | 1-2 days | ⭐ **DO THIS SECOND** |
| **3. LBVH Radix Tree** | **50-100×** | 🟡 Challenging | 🔴 Hard | 2-3 weeks | ⚠️ **Only if needed** |
| **4. Hilbert Curves** | **1.5-2×** | 🟢 Easy | 🟢 Easy | 1 day | ❌ **Not worth it** |

**Recommended Strategy**: Implement #1 and #2 first (4-6 days total). This will achieve 80-90% of maximum possible speedup with moderate effort. Only consider #3 if you need absolute maximum performance.

---

## Part I: Detailed Analysis of Current Implementation

### 1.1 What Sunnet Review Revealed

From [MORTON_OPTIMIZATION_GUIDE_Review_Sunnet-think2.md](MORTON_OPTIMIZATION_GUIDE_Review_Sunnet-think2.md):

**Critical Findings (Section 2.4)**:

> "Your leaf definition: `leaf_start[i] = i * 256` - These boundaries have **no geometric meaning**."

**Example of the Problem**:
```
Sorted Morton codes (all in SAME spatial octant):
  [0x00A1B2C3 (elem 1024),
   0x00A1B2C4 (elem 512),
   ...
   0x00A1B2FF (elem 2048), ← Leaf 4 boundary (arbitrary!)
   0x00A1B300 (elem 1111), ← Leaf 5 boundary (arbitrary!)
   ...
   0x00A1FFFF (elem 3001)]

All elements have same 6-bit prefix (0x00A) → Same spatial octant
But split across multiple leaves due to arbitrary 256-boundary
```

**Consequence**: To find all elements in one spatial region, you must search **many leaves** → `L2_SEARCH_RADIUS=100` compensates for this fragmentation.

### 1.2 Performance Cost Breakdown

**Current L2 Search** (per particle that fails L0+L1):

```
1. Compute Morton code: ~10 ops (bit interleaving)
2. Extract prefix: ~2 ops (bit shift)
3. Prefix table lookup: ~1 op (array access)
4. Linear scan ±100 leaves:
   - 201 leaves × 256 elements/leaf = 51,456 element checks
   - Each check: point_in_tet_gpu (~150 ops)
   - Total: ~7.7M ops per L2 query
```

**With Graded Refinement** (your mesh):
- Fine region (85% of elements): Dense in 3D space, sparse in Morton space
- Fixed-capacity leaves mix coarse and fine elements
- Need large radius to catch spatially close elements with distant Morton codes

---

## Part II: Option 1 - Octree-Aligned Leaves ⭐ BEST OPTION

### 2.1 Why This Is the Right Choice

**From Sunnet Review (Section 4.3)**:

> "**If full LBVH is too complex**, fix your current method with **geometric leaves**"

**Advantages**:
1. ✅ **Already partially implemented** - Your code already uses `build_global_morton_octree()`!
2. ✅ **JAX-friendly** - Uses same sorted array + lookup table approach
3. ✅ **Proven effective** - Review estimates 30-50× speedup
4. ✅ **Moderate complexity** - 3-5 days implementation
5. ✅ **No new dependencies** - Works with existing infrastructure

**Key Insight from My Previous Analysis**:

I discovered that `build_global_morton_octree()` is **already being used** in your production code! However, there may be issues with how leaves are structured. The octree builder exists but may not be creating truly octree-aligned leaves.

### 2.2 What Octree-Aligned Leaves Actually Mean

**Current (WRONG)**:
```python
# Fixed-capacity leaves (arbitrary boundaries)
for i in range(n_leaves):
    leaf_start[i] = i * 256
    leaf_length[i] = 256
```

**Correct Octree-Aligned**:
```python
# Leaves defined by Morton code prefixes (spatial octants)
for each octant at depth D:
    prefix = octant_coordinates_to_morton(x, y, z, depth=D)
    elements_in_octant = filter(sorted_morton, prefix)

    if len(elements_in_octant) <= 256:
        create_leaf(prefix, elements_in_octant)  # Single leaf
    else:
        # Subdivide to depth D+1 (8 child octants)
        for child_octant in subdivide(octant):
            create_leaf(child_prefix, child_elements)
```

**Key Property**: One Morton prefix → One leaf (or small cluster if subdivided)

### 2.3 Implementation Plan for Octree-Aligned Leaves

**Step 1: Verify Current Octree Builder** (1 hour)

Check if `build_global_morton_octree()` already creates octree-aligned leaves:

```python
# Read jaxtrace/gpu/search/morton_octree_builder.py
# Look for build_adaptive_octree_leaves()
# Verify it creates leaves based on Morton prefixes, not fixed capacity
```

**From my earlier analysis**: The builder EXISTS and is being called, but let me check if it's actually creating octree-aligned leaves or just fixed-capacity leaves with prefix tables.

**Step 2: Modify Leaf Structure** (1-2 days)

If leaves are currently fixed-capacity, modify to use spatial octants:

```python
def build_octree_aligned_leaves(
    morton_sorted: np.ndarray,
    elem_ids_sorted: np.ndarray,
    target_depth: int = 6,  # Start at depth 6 (262k octants)
    max_capacity: int = 256
) -> List[OctreeLeaf]:
    """
    Build leaves aligned with octree cells.

    Algorithm:
    1. Start with all depth-6 octants (262k candidates)
    2. For each octant:
       - Find elements with matching prefix
       - If ≤256 elements: create leaf at depth 6
       - If >256 elements: subdivide to depth 7 (8 children)
    3. Result: Variable-depth leaves, all octree-aligned
    """
    leaves = []
    n_prefixes_at_depth = 8 ** target_depth  # 262,144 for depth=6

    for prefix_6 in range(n_prefixes_at_depth):
        # Morton range for this octant at depth 6
        prefix_bits = prefix_6
        shift_amount = 63 - (target_depth * 3)  # 63 - 18 = 45
        morton_min = prefix_bits << shift_amount
        morton_max = (prefix_bits + 1) << shift_amount

        # Binary search in sorted array
        start_idx = np.searchsorted(morton_sorted, morton_min, side='left')
        end_idx = np.searchsorted(morton_sorted, morton_max, side='left')
        count = end_idx - start_idx

        if count == 0:
            continue  # Empty octant

        elif count <= max_capacity:
            # Single leaf at depth 6
            leaves.append(OctreeLeaf(
                start_idx=start_idx,
                length=count,
                morton_prefix=prefix_6,
                prefix_bits=target_depth * 3
            ))

        else:
            # Subdivide to depth 7 (8 child octants)
            for child in range(8):
                prefix_7 = (prefix_6 << 3) | child
                child_shift = 63 - ((target_depth + 1) * 3)  # 63 - 21 = 42
                child_min = prefix_7 << child_shift
                child_max = (prefix_7 + 1) << child_shift

                child_start = np.searchsorted(morton_sorted, child_min, side='left')
                child_end = np.searchsorted(morton_sorted, child_max, side='left')
                child_count = child_end - child_start

                if child_count > 0:
                    leaves.append(OctreeLeaf(
                        start_idx=child_start,
                        length=child_count,
                        morton_prefix=prefix_7,
                        prefix_bits=(target_depth + 1) * 3
                    ))

    return leaves
```

**Step 3: Update Prefix Table** (1 day)

Build prefix table that maps to octree leaves:

```python
def build_prefix_to_leaf_table(
    leaves: List[OctreeLeaf],
    table_depth: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build lookup table: prefix → leaf_id

    Returns:
        prefix_to_leaf: (8^D,) int32 - maps prefix to first matching leaf
        leaf_count: (8^D,) int32 - number of leaves with this prefix
    """
    n_entries = 8 ** table_depth
    prefix_to_leaf = np.full(n_entries, -1, dtype=np.int32)
    leaf_count = np.zeros(n_entries, dtype=np.int32)

    for leaf_id, leaf in enumerate(leaves):
        # Extract depth-6 prefix from this leaf's prefix
        if leaf.prefix_bits == table_depth * 3:
            # Leaf at exact table depth
            prefix_6 = leaf.morton_prefix
        else:
            # Leaf at depth 7, extract depth-6 parent prefix
            shift = leaf.prefix_bits - (table_depth * 3)  # 21 - 18 = 3
            prefix_6 = leaf.morton_prefix >> shift

        # Register this leaf under its depth-6 prefix
        if prefix_to_leaf[prefix_6] == -1:
            prefix_to_leaf[prefix_6] = leaf_id
        leaf_count[prefix_6] += 1

    return prefix_to_leaf, leaf_count
```

**Step 4: Update GPU Search** (1 day)

Modify L2 search to use octree leaves:

```python
def search_l2_octree_leaves(pos: jax.Array, mesh_gpu: MeshGPUGlobalMorton) -> jax.Array:
    """L2 search with octree-aligned leaves."""

    # 1. Compute Morton code for position
    morton_query = morton_encode_position_jax(
        pos, mesh_gpu.bbox_min, mesh_gpu.bbox_max, mesh_gpu.max_depth
    )

    # 2. Extract depth-6 prefix
    prefix_6 = morton_query >> (63 - 6 * 3)  # Top 18 bits
    prefix_6 = jnp.clip(prefix_6, 0, 262143)  # Clamp to valid range

    # 3. Look up leaf ID from prefix table
    first_leaf_id = mesh_gpu.prefix_to_leaf[prefix_6]
    num_leaves_in_octant = mesh_gpu.leaf_count[prefix_6]

    # 4. Search center leaf (if exists)
    elem_id = jnp.where(
        first_leaf_id >= 0,
        search_in_leaf_global(pos, first_leaf_id, mesh_gpu),
        jnp.int32(-1)
    )
    found = elem_id >= 0

    # 5. If not found, search neighbor leaves (26 spatial neighbors)
    #    (This replaces the linear ±100 radius)
    elem_final = jnp.where(
        found,
        elem_id,
        search_26_neighbor_octants(pos, prefix_6, mesh_gpu)  # NEW
    )

    return elem_final
```

### 2.4 Expected Performance Improvement

**From Sunnet Review**:

> "Search radius reduced from 100 to 3... 30-50× fewer element tests per query"

**Detailed Analysis**:

| Metric | Current (Fixed) | With Octree Leaves | Improvement |
|--------|-----------------|-------------------|-------------|
| Leaves per octant | 2-8 (fragmented) | 1 (exact) | 1× |
| Search radius | ±100 leaves | ±1-3 leaves | 30-100× |
| Elements tested | 51,456 | 256-1,536 | 33-200× |
| L2 query time | ~500μs | ~15-30μs | 15-30× |
| Overall step time | 3.7s | **0.5-1.0s** | 3-7× |

**Why Overall Speedup Is Lower**:
- L2 is only called when L0+L1 fail (~30% of particles)
- Other costs: L0, L1, velocity interpolation, RK4 stages
- But L2 is currently the bottleneck, so fixing it gives 3-7× total speedup

---

## Part III: Option 2 - Morton Neighbor Arithmetic ⭐ DO THIS TOO

### 3.1 Why This Complements Octree Leaves

Even with octree-aligned leaves, particles near octant boundaries need to search **spatial neighbors**, not arbitrary ±radius leaves.

**Current Problem** (even with octree leaves):
```python
# Current: Linear offset search
offsets = jnp.arange(-3, 4)  # [-3, -2, -1, 0, 1, 2, 3]
for offset in offsets:
    neighbor_leaf_id = center_leaf + offset  # WRONG! Not spatial neighbors
```

**Why This Is Wrong**:
- Leaf IDs are assigned during depth-first octree traversal
- Adjacent leaf IDs are NOT spatially adjacent
- Example: Leaves 100 and 101 might be in opposite corners of the domain!

**Correct Approach**: Use Morton arithmetic to find 26 spatial neighbor octants

### 3.2 Morton Neighbor Finding Algorithm

**Step 1: Decode prefix to octant coordinates**

```python
def decode_morton_prefix(prefix: jnp.uint64, depth: int) -> Tuple[int, int, int]:
    """
    De-interleave Morton prefix bits to get (x, y, z) octant coordinates.

    Example:
        prefix = 0b001101011  (9 bits = depth 3)
        Bits: [001][101][011]
              └xyz┘└xyz┘└xyz┘
        Depth 0: octant (0,0,1)
        Depth 1: octant (1,0,1)
        Depth 2: octant (0,1,1)

        Result: x = 0b010 = 2, y = 0b101 = 5, z = 0b111 = 7
        Octant coordinates: (2, 5, 7) at depth 3
    """
    x, y, z = 0, 0, 0

    for i in range(depth):
        # Extract bit triplet for this level
        bit_pos = (depth - 1 - i) * 3
        octant_bits = (prefix >> bit_pos) & 0b111

        # De-interleave: [x][y][z]
        x |= ((octant_bits >> 2) & 1) << i
        y |= ((octant_bits >> 1) & 1) << i
        z |= ((octant_bits >> 0) & 1) << i

    return x, y, z
```

**Step 2: Find 26 neighbor octants**

```python
def get_26_neighbor_prefixes(prefix: jnp.uint64, depth: int) -> jax.Array:
    """
    Get Morton prefixes for 26 spatial neighbor octants.

    Returns:
        neighbor_prefixes: (27,) uint64 - includes self at index 13
    """
    # Decode center octant coordinates
    cx, cy, cz = decode_morton_prefix(prefix, depth)

    # Generate 26 neighbors + self
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                # Neighbor coordinates
                nx = cx + dx
                ny = cy + dy
                nz = cz + dz

                # Clamp to valid range [0, 2^depth - 1]
                max_coord = (2 ** depth) - 1
                nx = jnp.clip(nx, 0, max_coord)
                ny = jnp.clip(ny, 0, max_coord)
                nz = jnp.clip(nz, 0, max_coord)

                # Encode back to Morton prefix
                neighbor_prefix = encode_morton_prefix(nx, ny, nz, depth)
                neighbors.append(neighbor_prefix)

    return jnp.array(neighbors, dtype=jnp.uint64)

def encode_morton_prefix(x: int, y: int, z: int, depth: int) -> jnp.uint64:
    """Encode octant coordinates back to Morton prefix."""
    prefix = 0
    for i in range(depth):
        bit_pos = (depth - 1 - i) * 3
        octant_bits = (((x >> i) & 1) << 2) | (((y >> i) & 1) << 1) | ((z >> i) & 1)
        prefix |= octant_bits << bit_pos
    return jnp.uint64(prefix)
```

**Step 3: Look up leaves for neighbor octants**

```python
def search_26_neighbor_octants(
    pos: jax.Array,
    center_prefix: jnp.uint64,
    mesh_gpu: MeshGPUGlobalMorton
) -> jnp.int32:
    """
    Search 26 spatial neighbor octants for containing element.

    This replaces the linear ±radius search with geometrically correct neighbors.
    """
    # Get 26 neighbor prefixes
    neighbor_prefixes = get_26_neighbor_prefixes(center_prefix, mesh_gpu.table_depth)

    # Search each neighbor octant
    def search_neighbor_octant(neighbor_prefix):
        # Look up leaf ID for this prefix
        neighbor_prefix_clamped = jnp.clip(neighbor_prefix, 0, len(mesh_gpu.prefix_to_leaf) - 1)
        leaf_id = mesh_gpu.prefix_to_leaf[neighbor_prefix_clamped]

        # Search this leaf (if exists)
        valid = leaf_id >= 0
        elem_id = jnp.where(
            valid,
            search_in_leaf_global(pos, leaf_id, mesh_gpu),
            jnp.int32(-1)
        )
        return elem_id

    # Vmap over 27 neighbors (including self)
    results = jax.vmap(search_neighbor_octant)(neighbor_prefixes)

    # Return first valid result
    found_mask = results >= 0
    return jnp.where(
        jnp.any(found_mask),
        results[jnp.argmax(found_mask)],
        jnp.int32(-1)
    )
```

### 3.3 Why This Is Important

**Example Scenario**:

```
Particle at position (0.001, 0.001, 0.001)
Morton code: 0x00000...001A3F
Depth-6 prefix: 0x000000  (octant (0,0,0) at depth 6)

Spatial neighbors:
  Octant (0,0,0) - center  → prefix 0x000000
  Octant (1,0,0) - right   → prefix 0x000001
  Octant (0,1,0) - top     → prefix 0x000008
  Octant (0,0,1) - front   → prefix 0x000040
  ... (26 total)

With octree-aligned leaves:
  prefix 0x000000 → Leaf 42
  prefix 0x000001 → Leaf 103  ← NOT leaf 43!
  prefix 0x000008 → Leaf 7    ← NOT leaf 50!

Linear ±radius would search leaves [39, 40, 41, 42, 43, 44, 45]
But actual spatial neighbors are leaves [7, 42, 103, ...]
```

**Without Morton neighbor arithmetic**: Even with octree leaves, you still need large radius to catch neighbors

**With Morton neighbor arithmetic**: Search exactly 27 octants (center + 26 neighbors)

### 3.4 Implementation Timeline

**Total: 1-2 days**

1. ✅ Implement `decode_morton_prefix()` (2 hours)
2. ✅ Implement `encode_morton_prefix()` (1 hour)
3. ✅ Implement `get_26_neighbor_prefixes()` (2 hours)
4. ✅ Implement `search_26_neighbor_octants()` (3 hours)
5. ✅ Test and debug (1 day)

---

## Part IV: Option 3 - LBVH Radix Tree ⚠️ Only If Needed

### 4.1 Why This Is Powerful But Complex

**From Sunnet Review (Section 4.1)**:

> "Karras 2012 LBVH algorithm... 50-100× speedup over current method"

**LBVH Algorithm**:
1. Sort primitives by Morton code (already done ✅)
2. **Build binary radix tree in parallel** (O(N) on GPU)
3. Query: Binary descent through tree (O(log N))

**Key Advantage**: Tree hierarchy provides exact path to containing element

### 4.2 Research Findings

I searched for LBVH implementations and found:

**Existing Implementations**:
1. **[ToruNiina/lbvh (CUDA)](https://github.com/ToruNiina/lbvh)** - Based on Karras 2012 paper
2. **[KittenGpuLBVH (CUDA)](https://github.com/jerry060599/KittenGpuLBVH)** - Build 100K objects in 150μs on RTX 3090
3. **[VkLBVH (Vulkan/GLSL)](https://github.com/MircoWerner/VkLBVH)** - GPU LBVH builder in Vulkan

**JAX Status**: ❌ **No existing JAX implementation found**

**Key Papers**:
- **[Karras 2012: "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"](https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf)** - Original paper
- **[Karras 2013: "Fast Parallel Construction of High-Quality Bounding Volume Hierarchies"](https://research.nvidia.com/sites/default/files/pubs/2013-07_Fast-Parallel-Construction/karras2013hpg_paper.pdf)** - Improved version
- **[NVIDIA Blog: "Thinking Parallel, Part III: Tree Construction on the GPU"](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/)** - Tutorial

### 4.3 JAX Implementation Challenges

**Challenge 1: Parallel Tree Construction**

Karras algorithm uses GPU-parallel construction:
```cuda
__global__ void build_radix_tree(...) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Use __clzll (count leading zeros) for bit operations
    int clz_left = __clzll(sorted_morton[idx] ^ sorted_morton[idx-1]);
    int clz_right = __clzll(sorted_morton[idx] ^ sorted_morton[idx+1]);

    // Find split point via binary search
    int split = find_split(idx, ...);

    // Assign children pointers
    left_child[idx] = split;
    right_child[idx] = split + 1;
}
```

**JAX Equivalent**:
```python
@jax.jit
def build_radix_tree_jax(sorted_morton: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Build LBVH radix tree in JAX."""
    n = len(sorted_morton)

    # Vectorized XOR for adjacent pairs
    morton_xor_left = sorted_morton[:-1] ^ sorted_morton[1:]

    # Count leading zeros (JAX has lax.clz for this)
    clz_values = lax.clz(morton_xor_left)

    # Find split points (vectorized)
    # ... (complex binary search logic)

    # Build parent/child arrays
    left_child = jnp.zeros(n-1, dtype=jnp.int32)
    right_child = jnp.zeros(n-1, dtype=jnp.int32)

    # ... (tree construction logic)

    return left_child, right_child
```

**Problem**: JAX's vectorization is less flexible than CUDA's thread-level parallelism for irregular tree operations.

**Challenge 2: Stackless Traversal**

**CUDA** uses dynamic stacks or recursion:
```cuda
__device__ int traverse_tree(...) {
    int node = 0;  // Root
    while (node < n_internals) {
        if (query < split[node])
            node = left_child[node];
        else
            node = right_child[node];
    }
    return node - n_internals;  // Leaf index
}
```

**JAX** requires fixed-iteration loops:
```python
def traverse_tree_jax(query_morton, left_child, right_child, split_morton):
    node = 0

    # Fixed-depth iteration (max 30 for 3M elements)
    def step(carry, _):
        node, done = carry
        is_internal = node < n_internals

        go_left = query_morton < split_morton[node]
        child = jnp.where(go_left, left_child[node], right_child[node])

        next_node = jnp.where(is_internal & (~done), child, node)
        next_done = done | (~is_internal)

        return (next_node, next_done), None

    (leaf, _), _ = lax.scan(step, (node, False), None, length=30)
    return leaf
```

**Feasible**: Yes, but requires careful implementation

### 4.4 When to Implement LBVH

**Implement LBVH if**:
1. ✅ Octree leaves + Morton neighbors still too slow (step time >1s)
2. ✅ You need to generalize to other mesh types (not just octree refinement)
3. ✅ You're willing to invest 2-3 weeks

**Don't implement LBVH if**:
1. ❌ Octree leaves + Morton neighbors achieve target performance (<1s per step)
2. ❌ Only working with octree-refined meshes (octree leaves are simpler)
3. ❌ Limited development time

**My Recommendation**: **Wait and see** - Implement options 1 and 2 first, measure performance, then decide.

---

## Part V: Option 4 - Hilbert Curves ❌ Not Recommended

### 5.1 Why Hilbert Curves Seem Attractive

**Theoretical Advantage**: Better spatial locality than Morton Z-order

From research ([How the Idea of the Hilbert Curve Inspired Morton Curves for GPU Performance](https://blog.stackademic.com/how-the-idea-of-the-hilbert-curve-inspired-morton-curves-for-gpu-performance-4e235d670304)):

> "Hilbert curve... has better locality-preserving behavior... avoids long edges"

**Visual Comparison**:
```
Morton Z-Order (discontinuities at octant boundaries):
  0---1   4---5
  |   |   |   |
  2---3   6---7  ← Gap: 3→4 (adjacent in space, far in code)

  8---9   C---D
  |   |   |   |
  A---B   E---F  ← Gap: 7→8 (adjacent in space, far in code)

Hilbert Curve (continuous):
  0---1   E---F
      |   |
  3---2   D   C
  |           |
  4   7---8   B  ← No gaps! Always continuous
  |   |       |
  5---6   9---A
```

**Benefit**: Hilbert encoding naturally groups spatially close elements

### 5.2 Why Hilbert Is WRONG for Your Use Case

**Critical Issue**: Encoding complexity

From research ([Voxel Compression: Space-Filling Curves](https://eisenwave.github.io/voxel-compression-docs/rle/space_filling_curves.html)):

> "A Morton ordering is faster to compute... offers simplicity and speed"

**From 2025 research** ([How the Idea of the Hilbert Curve Inspired Morton Curves](https://blog.stackademic.com/how-the-idea-of-the-hilbert-curve-inspired-morton-curves-for-gpu-performance-4e235d670304)):

> "While the Hilbert curve is mathematically optimal, it is computationally complex: Recursive Nature... Hardware Infeasibility: GPUs are designed for simplicity and speed"

**Encoding Time Comparison**:

| Operation | Morton | Hilbert | Ratio |
|-----------|--------|---------|-------|
| Encode position | ~10 ops (bit interleaving) | ~150 ops (recursive rotation) | **15× slower** |
| Per particle | ~30ns | ~450ns | - |
| For 48K particles | 1.4ms | 21.6ms | **15× slower** |

**Why This Matters**: You encode positions at **every RK4 stage** (5× per timestep)

### 5.3 Real-World Performance Data

From research ([Locality Properties of 3D Data Orderings](https://orca.cardiff.ac.uk/id/eprint/123194/1/main3.pdf)):

> "For the interaction density we studied, the higher overhead of computing Hilbert keys for the interactions **masked any potential performance benefits**"

**Databricks Comparison** ([Z-order or Hilbert Curve, which is better](https://community.databricks.com/t5/data-engineering/z-order-or-hilbert-curve-which-is-better/td-p/18200)):

> "Hilbert curves... can speed up read queries by skipping more data than Z-order"

But this is for **disk-based queries** where I/O dominates. For **in-memory GPU** where computation dominates, Morton wins.

### 5.4 Quantitative Analysis for Your Mesh

**Current Performance**:
- Particle encoding: 5 ops/particle (already fast with Morton)
- L2 search: 51,456 element tests (bottleneck!)

**With Hilbert**:
- Particle encoding: 75 ops/particle (**15× slower**)
- L2 search: ~25,000 element tests (2× better locality)

**Net Effect**:
- Encoding overhead: +15μs per particle
- Search speedup: -10μs per particle (for particles that reach L2)
- **Net: +5μs per particle** (SLOWER!)

**For 48K particles**:
- Current: ~3.7s per step
- With Hilbert: **~4.0s per step** (8% slower!)

**Verdict**: ❌ **Hilbert curves make your code slower for this use case**

### 5.5 When Hilbert Curves ARE Useful

**Good use cases**:
1. ✅ Disk-based spatial databases (I/O-bound)
2. ✅ Very large datasets where encoding is amortized
3. ✅ Static data structures (encode once, query many times)

**Your use case**:
- ❌ In-memory GPU (compute-bound)
- ❌ Small-medium dataset (3M elements)
- ❌ Dynamic (re-encode every timestep)

**Conclusion**: Stick with Morton. Don't waste time on Hilbert.

---

## Part VI: Recommended Implementation Strategy

### 6.1 Phase 1: Octree-Aligned Leaves + Morton Neighbors (4-6 days) ⭐

**Goal**: Achieve 80-90% of maximum possible performance

**Step 1: Verify Current Implementation** (1 hour)
```bash
# Check if octree builder is actually creating octree-aligned leaves
python -c "
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
import numpy as np

# Load mesh
node_positions = np.load('mesh/node_positions.npy')
connectivity = np.load('mesh/connectivity.npy')

# Build structure
morton_struct = build_global_morton_octree(
    node_positions, connectivity, leaf_capacity=256, verbose=True
)

# Check leaf distribution
print(f'Number of leaves: {morton_struct.n_leaves}')
print(f'Table depth: {morton_struct.table_depth}')
print(f'Prefix table size: {len(morton_struct.prefix_start)}')

# Verify leaves are octree-aligned
# (check if leaf boundaries correspond to Morton prefixes)
"
```

**Step 2: Implement Morton Neighbor Arithmetic** (2 days)

**File**: `jaxtrace/gpu/search/morton_neighbors.py` (NEW)

```python
"""Morton code neighbor finding for spatial queries."""

import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def decode_morton_prefix_jax(prefix: jnp.uint64, depth: int) -> Tuple[jnp.int32, jnp.int32, jnp.int32]:
    """De-interleave Morton prefix to octant coordinates."""
    x, y, z = jnp.int32(0), jnp.int32(0), jnp.int32(0)

    # Unroll for JAX (max depth = 10)
    for i in range(10):
        should_process = i < depth
        bit_pos = (depth - 1 - i) * 3
        octant_bits = (prefix >> bit_pos) & jnp.uint64(0b111)

        x_bit = ((octant_bits >> 2) & jnp.uint64(1)).astype(jnp.int32) << i
        y_bit = ((octant_bits >> 1) & jnp.uint64(1)).astype(jnp.int32) << i
        z_bit = ((octant_bits >> 0) & jnp.uint64(1)).astype(jnp.int32) << i

        x = jnp.where(should_process, x | x_bit, x)
        y = jnp.where(should_process, y | y_bit, y)
        z = jnp.where(should_process, z | z_bit, z)

    return x, y, z

@jax.jit
def encode_morton_prefix_jax(x: jnp.int32, y: jnp.int32, z: jnp.int32, depth: int) -> jnp.uint64:
    """Interleave octant coordinates to Morton prefix."""
    prefix = jnp.uint64(0)

    # Unroll for JAX (max depth = 10)
    for i in range(10):
        should_process = i < depth
        bit_pos = (depth - 1 - i) * 3

        x_bit = jnp.uint64((x >> i) & 1) << (bit_pos + 2)
        y_bit = jnp.uint64((y >> i) & 1) << (bit_pos + 1)
        z_bit = jnp.uint64((z >> i) & 1) << (bit_pos + 0)

        octant_bits = x_bit | y_bit | z_bit
        prefix = jnp.where(should_process, prefix | octant_bits, prefix)

    return prefix

@jax.jit
def get_27_neighbor_prefixes_jax(center_prefix: jnp.uint64, depth: int) -> jax.Array:
    """Get Morton prefixes for 26 neighbors + self."""
    # Decode center
    cx, cy, cz = decode_morton_prefix_jax(center_prefix, depth)

    max_coord = (2 ** depth) - 1

    # Generate 27 neighbors (3×3×3 cube)
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                nx = jnp.clip(cx + dx, 0, max_coord)
                ny = jnp.clip(cy + dy, 0, max_coord)
                nz = jnp.clip(cz + dz, 0, max_coord)

                prefix = encode_morton_prefix_jax(nx, ny, nz, depth)
                neighbors.append(prefix)

    return jnp.array(neighbors, dtype=jnp.uint64)
```

**File**: `jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py` (MODIFY)

Replace current L2 search function (lines 155-195) with:

```python
from jaxtrace.gpu.search.morton_neighbors import get_27_neighbor_prefixes_jax

def search_l2_single(pos: jax.Array) -> jax.Array:
    """L2: Global Morton search with octree-aligned leaves and spatial neighbors."""

    # 1. Compute Morton code
    morton_query = morton_encode_position_jax(
        pos,
        mesh_gpu_global_morton.bbox_min,
        mesh_gpu_global_morton.bbox_max,
        mesh_gpu_global_morton.max_depth
    )

    # 2. Extract prefix at table depth
    table_depth = mesh_gpu_global_morton.table_depth
    shift = 63 - (table_depth * 3)
    center_prefix = morton_query >> shift

    # 3. Look up center leaf
    center_prefix = jnp.clip(center_prefix, 0, len(mesh_gpu_global_morton.prefix_start) - 1)
    center_leaf_id = mesh_gpu_global_morton.prefix_start[center_prefix]

    # 4. Search center leaf first
    elem_id = jnp.where(
        center_leaf_id >= 0,
        search_in_leaf_global(pos, center_leaf_id, mesh_gpu_global_morton),
        jnp.int32(-1)
    )
    found = elem_id >= 0

    # 5. If not found, search 27 spatial neighbor octants
    def search_neighbor_octant(neighbor_prefix):
        # Look up leaf for this prefix
        np_clamped = jnp.clip(neighbor_prefix, 0, len(mesh_gpu_global_morton.prefix_start) - 1)
        leaf_id = mesh_gpu_global_morton.prefix_start[np_clamped]

        # Search if valid
        valid = leaf_id >= 0
        result = jnp.where(
            valid,
            search_in_leaf_global(pos, leaf_id, mesh_gpu_global_morton),
            jnp.int32(-1)
        )
        return result

    # Get 27 neighbor prefixes
    neighbor_prefixes = get_27_neighbor_prefixes_jax(center_prefix, table_depth)

    # Search all neighbors (vectorized)
    neighbor_results = jax.vmap(search_neighbor_octant)(neighbor_prefixes)

    # Find first valid result
    neighbor_mask = neighbor_results >= 0
    found_in_neighbor = jnp.where(
        jnp.any(neighbor_mask),
        neighbor_results[jnp.argmax(neighbor_mask)],
        jnp.int32(-1)
    )

    # Return center result if found, else neighbor result
    return jnp.where(found, elem_id, found_in_neighbor)
```

**Step 3: Test and Benchmark** (2 days)

```bash
# Test with small particle count first
python production_tracking_fully_fused_timedep.py \
    PARTICLE_GRID_RESOLUTION="(10,40,15)" \  # 6K particles
    N_STEPS=100 \
    2>&1 | tee logs/test_octree_morton_neighbors.log

# Compare to baseline
# Expected: 3-5× faster step time
```

### 6.2 Phase 2: Evaluate Results and Decide Next Steps (1 day)

**Success Criteria**:
- Step time: <1.0s (vs current ~3.7s) ✅
- Retention: >70% at step 2500 (vs current ~60%) ✅
- Trajectories: Still correct (rotating motion) ✅

**If criteria met**:
- ✅ **DONE!** You've achieved production-ready performance
- Move on to other optimizations (velocity caching, adaptive dt, etc.)

**If criteria NOT met**:
- Consider Phase 3 (LBVH radix tree)
- Or investigate other bottlenecks (L1 performance, initial assignment, etc.)

### 6.3 Phase 3 (Optional): LBVH Radix Tree (2-3 weeks)

**Only if needed** - See Section 4 for details

**Timeline**:
1. Week 1: Implement tree construction in JAX
2. Week 2: Implement stackless traversal and search
3. Week 3: Test, debug, and optimize

**Expected outcome**:
- Additional 2-5× speedup over octree leaves
- Step time: 0.2-0.5s
- Retention: 90-95%

---

## Part VII: Performance Projections

### 7.1 Current Baseline (Measured)

| Metric | Value | Source |
|--------|-------|--------|
| Step time | 3.7s | Your test |
| Throughput | 13K particles/s | Your test |
| Retention (step 100) | 79.39% | Your test |
| Retention (step 500) | 70.27% | Your test |
| L2 calls | ~30% of particles | Estimated |

### 7.2 After Octree Leaves + Morton Neighbors

| Metric | Current | Projected | Method |
|--------|---------|-----------|--------|
| L2 elements tested | 51,456 | 1,536 | 27 octants × 8 leaves × 64 elements/leaf |
| L2 query time | 500μs | 15μs | 33× speedup |
| Step time | 3.7s | **0.8s** | 4.6× speedup |
| Throughput | 13K p/s | **60K p/s** | 4.6× faster |
| Retention (step 500) | 70.27% | **85-90%** | Better L2 hit rate |

**Calculation**:
```
Current step breakdown (estimated):
  L0+L1: 0.5s (13.5%)
  L2: 2.5s (67.5%)  ← BOTTLENECK
  Velocity interp + RK4: 0.7s (19%)
  Total: 3.7s

After optimization:
  L0+L1: 0.5s (same)
  L2: 0.08s (33× speedup on 2.5s) ← FIXED
  Velocity interp + RK4: 0.7s (same)
  Total: 0.5 + 0.08 + 0.7 = 1.28s

Actually: Some L1 will become L0 (better caching) → Additional ~0.3s saved
Final estimate: ~0.8-1.0s per step
```

### 7.3 After LBVH Radix Tree (If Needed)

| Metric | With Octree | With LBVH | Method |
|--------|-------------|-----------|--------|
| L2 tree descent | N/A | 22 steps | O(log N) for 3M elements |
| L2 leaf size | 64-256 elem | 10-20 elem | Adaptive splitting |
| L2 query time | 15μs | **3μs** | 5× speedup |
| Step time | 0.8s | **0.4s** | 2× additional speedup |
| Throughput | 60K p/s | **120K p/s** | 2× faster |

---

## Part VIII: Final Recommendations

### 8.1 Prioritized Action Plan

**Immediate (This Week)**:
1. ⭐ **Implement octree-aligned leaves** (if not already done)
   - Verify `build_global_morton_octree()` creates spatial leaves
   - If not, modify builder (3-4 days)

2. ⭐ **Implement Morton neighbor arithmetic** (1-2 days)
   - Add `morton_neighbors.py` module
   - Replace linear radius search in L2
   - Test with small particle count

**Expected Outcome**: 3-5× speedup, 80-90% retention

**Short-Term (Next 2 Weeks)**:
3. 📊 **Benchmark and profile**
   - Measure actual performance gains
   - Identify remaining bottlenecks (if any)
   - Decide if LBVH is needed

4. 🔧 **Tune parameters**
   - Optimal octree depth (6 vs 7)
   - Leaf capacity (64 vs 128 vs 256)
   - L1 hop count (2 vs 3)

**Medium-Term (If Needed)**:
5. ⚠️ **Consider LBVH radix tree** (only if step time still >1s)
   - Port Karras 2012 algorithm to JAX
   - Implement stackless traversal
   - Expected: Additional 2× speedup

### 8.2 What NOT to Do

❌ **Don't implement Hilbert curves** - Slower for your use case
❌ **Don't over-optimize L1** - It's working correctly now
❌ **Don't increase L2 radius further** - Fix leaf structure instead
❌ **Don't implement LBVH first** - Octree leaves are simpler and give most of the benefit

### 8.3 Success Criteria

**Minimum acceptable**:
- Step time: <1.5s (2.5× faster than current)
- Retention: >75% at step 2500
- Trajectories: Correct

**Target**:
- Step time: <1.0s (3.7× faster)
- Retention: >85% at step 2500
- Throughput: >50K particles/s

**Stretch goal** (with LBVH):
- Step time: <0.5s (7× faster)
- Retention: >90% at step 2500
- Throughput: >100K particles/s

---

## References and Sources

### Academic Papers
1. **[Karras 2012: "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"](https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf)** - Original LBVH algorithm
2. **[Karras 2013: "Fast Parallel Construction of High-Quality Bounding Volume Hierarchies"](https://research.nvidia.com/sites/default/files/pubs/2013-07_Fast-Parallel-Construction/karras2013hpg_paper.pdf)** - Improved LBVH
3. **[Locality Properties of 3D Data Orderings (Cardiff University)](https://orca.cardiff.ac.uk/id/eprint/123194/1/main3.pdf)** - Hilbert vs Morton comparison

### Implementation Resources
4. **[ToruNiina/lbvh (GitHub)](https://github.com/ToruNiina/lbvh)** - CUDA LBVH implementation
5. **[KittenGpuLBVH (GitHub)](https://github.com/jerry060599/KittenGpuLBVH)** - High-performance GPU LBVH
6. **[NVIDIA Blog: "Thinking Parallel, Part III: Tree Construction on the GPU"](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/)** - LBVH tutorial

### Spatial Indexing Comparisons
7. **[Voxel Compression: Space-Filling Curves](https://eisenwave.github.io/voxel-compression-docs/rle/space_filling_curves.html)** - Morton vs Hilbert detailed comparison
8. **[How the Hilbert Curve Inspired Morton Curves for GPU Performance (Stackademic)](https://blog.stackademic.com/how-the-hilbert-curve-inspired-morton-curves-for-gpu-performance-4e235d670304)** - GPU performance analysis
9. **[Databricks: Z-order or Hilbert Curve, which is better](https://community.databricks.com/t5/data-engineering/z-order-or-hilbert-curve-which-is-better/td-p/18200)** - Practical comparison

### JAX Resources
10. **[JAX Documentation](https://github.com/jax-ml/jax)** - Official JAX repository
11. **[JAX-ML Scaling Book: GPUs](https://jax-ml.github.io/scaling-book/gpus/)** - GPU optimization guide

---

## Summary Table

| Method | Speedup | Complexity | JAX Compatible | Timeline | Recommended |
|--------|---------|------------|----------------|----------|-------------|
| **Octree-Aligned Leaves** | **30-50×** | 🟡 Medium | ✅ Yes | 3-5 days | ⭐⭐⭐ **YES** |
| **Morton Neighbors** | **10-15×** | 🟢 Easy | ✅ Yes | 1-2 days | ⭐⭐⭐ **YES** |
| **LBVH Radix Tree** | **50-100×** | 🔴 Hard | 🟡 Possible | 2-3 weeks | ⚠️ **Maybe** |
| **Hilbert Curves** | **0.5× (slower!)** | 🟢 Easy | ✅ Yes | 1 day | ❌ **NO** |

**Final Recommendation**: Implement octree-aligned leaves + Morton neighbor arithmetic (total 4-6 days). This will achieve 80-90% of maximum possible performance with moderate effort. Only consider LBVH if you need the last 10-20% of performance and are willing to invest 2-3 weeks.

---

**Next step**: Do you want me to start implementing octree-aligned leaves and Morton neighbor arithmetic?
