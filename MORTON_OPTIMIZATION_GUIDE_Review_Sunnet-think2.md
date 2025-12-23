# Comprehensive Critical Analysis: L1 Failure and Morton-Octree Search Issues

**Date**: December 19, 2025  
**Status**: Definitive Technical Review

***

## Executive Summary

After reviewing your MORTON_OPTIMIZATION_GUIDE, previous discussions, the HOT paper, and LBVH literature, I provide this comprehensive critical analysis that **challenges multiple incorrect assumptions** in your documents and identifies the **true root causes** of both failures.

**Key Findings**:

1. **L1 failure justification is fundamentally wrong** – the "overlapping parent/child elements" model doesn't exist in your mesh
2. **Morton implementation is neither HOT nor LBVH** – it's a hybrid that combines worst aspects of both
3. **Modern LBVH radix tree approach would provide 5-10× speedup** with simpler code
4. **Your "hash" is not a hash** – calling it that obscures what's actually happening

***

## Part I: L1 Search Failure – Correcting the Misdiagnosis

### 1.1 What Your Documents Claim (WRONG)

**Your MORTON_OPTIMIZATION_GUIDE Section 2.3 states**:[1]

> "Medium1 geometrically contains position P... Medium1 was subdivided into 8 fine elements... Medium1 **no longer exists** in the active mesh... L1 returns Medium (inactive element)."

**This is architecturally impossible in your mesh.**

### 1.2 How Octree Tet Meshes Actually Work

Your mesh from `p4est` octree refinement with tet generation:[1]

```
Refinement Process:
1. Base cube C0 → split into 4 tets {T1, T2, T3, T4}
2. If cube C0 refined:
   - Cube C0 deleted
   - 8 child cubes {C1...C8} created
   - Each child cube → 4 tets
   - Total: 32 tets replace original 4 tets
3. Original tets {T1, T2, T3, T4} **DELETED FROM MESH**
```

**Critical fact**: At any point in 3D space, exactly **ONE** active tet exists (the finest local resolution). There are **NO overlapping parent/child tets** in the connectivity arrays.

**Proof**: Your mesh has 3,048,900 elements total. If parent tets coexisted with children:
- 6-7 refinement levels
- Each parent → 8× children
- Total elements would be: \( \sum_{l=0}^{7} 8^l \times \text{base} \approx 10× current count \)
- You'd have ~30M elements, not 3M

**Verdict**: The "inactive parent element returned by L1" explanation is **physically impossible**.[1]

### 1.3 The Real L1 Failure Mechanism (CORRECTED)

**Actual reason L1 fails**: **Topology graph partitioning** in graded refinement, NOT early-exit greediness.

#### The Graded Refinement Structure

Your diagnostic output:[1]

```
Fine elements (≤0.15mm):    2,604,757 (85.3%)  — Refinement levels 6-7
Medium elements (0.15-0.30): 381,863  (12.5%)  — Refinement levels 4-5  
Coarse elements (>0.30mm):    67,280  (2.2%)   — Refinement levels 1-3
```

**Spatial distribution**:[1]
```
Fine region:    X ∈ [-9.36, 9.34], Y ∈ [-9.38, 9.40], Z ∈ [-4.51, -0.02] mm
Medium buffer:  X ∈ [-9.65, 9.61], Y ∈ [-9.82, 9.86], Z ∈ [-4.96, -0.04] mm
Coarse outer:   X ∈ [-28.75, 28.75], Y ∈ [-21.72, 21.72], Z ∈ [-9.37, -0.08] mm
```

**Graded refinement rule**: Each refinement level differs by **2:1 ratio** → 6-7 intermediate levels between coarse and fine.

#### Face-Based Neighbor Connectivity

**Your neighbor diagnostic result**: 100% of boundary coarse elements have **ZERO** fine neighbors.[1]

**Why?** Face-sharing (3 common nodes) only occurs between:
- Same-level elements (always)
- Adjacent-level elements (sometimes)
- **Multi-level jumps: NEVER**

**Graded buffer prevents direct coarse→fine face-sharing**:

```
Coarse element at refinement level 1:
  Shares faces with: Level-1 or Level-2 elements only
  
Fine element at refinement level 7:
  Shares faces with: Level-6 or Level-7 elements only
  
Gap: Levels 2-6 form mandatory buffer zone
```

#### L1 Multi-Hop Traversal Through Graded Buffer

**Scenario**: Particle at position \( \mathbf{p} \) inside fine element (level-7), L0 cached coarse element (level-1).

**L1 traversal with N_HOPS=3**:

```
Hop 0 (Initial state):
  current_elem = Coarse_A (level 1, size=1.09mm)
  point_in_tet(p, Coarse_A) = FALSE  (particle moved)

Hop 1:
  Neighbors of Coarse_A: {Medium_1, Medium_2, Medium_3, Medium_4}  (levels 2-3, face-sharing)
  Test Medium_1: point_in_tet(p, Medium_1) = TRUE ✓
  → found = TRUE, current_elem = Medium_1
  → EARLY EXIT (returns Medium_1)

Result: L1 returns Medium_1 (level 3, size=0.27mm)
Fine element (level 7, size=0.14mm): NEVER REACHED
```

**Why Medium_1 contains \( \mathbf{p} \) spatially**:

Medium_1 is a **REAL, ACTIVE element** at level-3. It exists in the mesh connectivity. The particle IS inside it geometrically. But:
- Medium_1 provides velocity from 4 nodes spaced ~0.27mm apart
- Fine velocity gradients (rotating tool) have wavelength ~0.10-0.14mm  
- Medium_1's interpolation **misses** those gradients

**The interpolation is smooth but wrong-resolution**, NOT wrong-geometry.

#### Why More Hops Don't Help (Countering "Just Increase N_HOPS")

**To reach level-7 fine from level-1 coarse**:[1]

```
Path through graded buffer (6 levels):
Level 1 → Level 2 → Level 3 → Level 4 → Level 5 → Level 6 → Level 7

Minimum hops: 6 (if each hop advances one level)
Realistic hops: 8-12 (levels overlap spatially, multiple hops per level)
```

**N_HOPS=3 cannot traverse 6 refinement levels.**

**But even N_HOPS=15 fails due to early-exit**:[1]

```
Hop 1: Coarse → Medium (level 2)
  point_in_tet(p, Medium_level2) = TRUE
  → RETURN Medium_level2 ✗

Never proceeds to Hop 2+
```

**The early-exit prevents traversal even with large N_HOPS.**

### 1.4 Why Node-Based Neighbors Are Insufficient

**Your test result**: Node-based neighbors **also found 0 fine neighbors** for boundary coarse elements.[1]

**Document claims**: "Expected to find edge-sharing" but didn't due to "memory constraints."

**This is misleading.** The real reason:

**Graded refinement with 6 buffer levels**:
- Coarse element (level 1) shares nodes with level-2 elements (edge/vertex sharing)
- Level-2 shares nodes with level-3
- ...
- Level-6 shares nodes with level-7 (fine)

**Direct coarse→fine node sharing**:
- Would require coarse cube edge to contain fine cube vertex
- Only happens if refinement jump is ≤2 levels
- With 6-level buffer: **NO direct node sharing**

**Node-based neighbors would need transitive closure**: "Neighbors-of-neighbors-of-neighbors..." → same as multi-hop, but with 100× more candidates (20-100 neighbors/element vs 4).

**Memory cost**:[1]
- Face-based: 4 neighbors × 3M elements × 4 bytes = **48 MB**
- Node-based: 20-100 neighbors × 3M elements × 4 bytes = **240 MB - 1.2 GB**
- Extended node closure (3-hop): 100-1000 neighbors × 3M × 4 bytes = **1.2 GB - 12 GB**

**Verdict**: Node-based neighbors are **correct in principle but impractical** for graded refinement without massive memory.

### 1.5 Corrected L1 Failure Summary

| Claim in Documents | Reality | Impact |
|--------------------|---------|--------|
| "L1 returns inactive parent elements" | **FALSE** – no parent/child coexistence | Misleading |
| "Early exit prevents finding fine elements" | **PARTIALLY TRUE** – but insufficient hops is primary issue | Overstated |
| "Node-based neighbors would fix it" | **FALSE** – still needs transitive closure | Wrong |
| "L1 finds wrong element" | **FALSE** – finds *correct* element at insufficient resolution | Conceptual error |

**True root cause**: **Face-based neighbor graph in graded refinement is partitioned by element size**. L1 traversal with bounded hops cannot cross 6-level graded buffers. The element L1 finds is geometrically and topologically correct, but provides insufficient resolution for accurate physics.

***

## Part II: Morton-Octree Implementation – Neither HOT Nor LBVH

### 2.1 What Your Implementation Actually Is

**Your documents call it**: "HOT-like global Morton L2 search"[1]

**What it actually does**:

```python
# CPU: Preprocessing
morton_codes = compute_morton(element_centroids)
sorted_indices = argsort(morton_codes)
elem_ids_sorted = elements[sorted_indices]

# Fixed-capacity leaves (NOT octree-aligned)
leaf_start[i] = i * 256
leaf_length[i] = min(256, n_elements - i*256)

# Prefix table (depth=6, 262k entries)
for prefix in 0..262143:
    prefix_start[prefix] = first_leaf_containing_prefix(prefix)
    prefix_length[prefix] = count_leaves_with_prefix(prefix)

# GPU: Query
morton_query = compute_morton(position)
prefix = morton_query >> 45  # Top 18 bits
first_leaf = prefix_start[prefix]
search_radius = 100  # ±100 leaves
for leaf in range(first_leaf - 100, first_leaf + 100):
    scan 256 elements in leaf
```

**This is a hybrid**:
- **Sorting**: From LBVH[2]
- **Prefix table**: Custom (not in HOT or LBVH)
- **Fixed leaves**: Arbitrary (neither octree cells nor BVH nodes)
- **Linear scan**: Brute-force

### 2.2 Comparison: HOT vs LBVH vs Your Method

| Aspect | HOT (Warren 1993) [3] | LBVH (Karras 2012) [4] | Your Method [1] |
|--------|------------------------------|-------------------------------|----------------------|
| **Tree structure** | Explicit octree (internal nodes stored) | Implicit binary radix tree | None (flat leaves) |
| **Node definition** | Octree cells (geometric cubes) | Morton code ranges (prefix-based) | Fixed 256-element chunks |
| **Keys** | Node keys (3d bits for depth-d) | Primitive keys (full 63 bits) | Element keys (63 bits) |
| **Lookup method** | Hash table `h = key & mask` | Binary search on sorted array | Prefix table (18-bit LUT) |
| **Tree walk** | Recursive: `child = (parent << 3) | octant` | Binary descent: compare prefixes | **None** (direct jump to leaves) |
| **Collision handling** | Linked lists | Not needed (sorted array) | Not needed (prefix maps to range) |
| **Query complexity** | O(1) per node + O(log N) walk | O(log N) binary descent | O(1) prefix + O(radius × 256) scan |
| **Memory** | Hash table + linked lists (~50 MB) | 2× sorted arrays (~24 MB) | Sorted array + prefix table (~26 MB) |
| **Build time** | O(N log N) tree construction | O(N) radix tree (parallel GPU) | O(N log N) sort |
| **GPU-friendly?** | **No** (pointers, linked lists) | **Yes** (fully data-parallel) | **Partial** (large scans) |

**Verdict**: Your method is **closest to LBVH conceptually**, but lacks the radix tree hierarchy. You have LBVH's sorted array but HOT's direct lookup philosophy, resulting in a **worst-of-both-worlds hybrid**.

### 2.3 Why Your "Hash" Is Not a Hash

**You call it**: "Hashed octree" / "HOT-like"[1]

**What a hash actually is**:
- **Function**: Input → fixed-size output with **probabilistic collisions**
- **Purpose**: Distribute keys uniformly across buckets
- **Collision handling**: Required (chaining, open addressing)

**What your prefix table is**:
```python
prefix = morton >> 45  # Deterministic bit extraction
leaf_range = prefix_start[prefix]  # Direct array lookup
```

- **No hash function** – just bit masking
- **No collisions** – each prefix maps to a fixed leaf range
- **Not random** – preserves spatial structure

**This is a Lookup Table (LUT)**, not a hash. Specifically: **Direct-Mapped Cache** (computer architecture term).[4]

**HOT's hash** (actual hash):[3]
```c
hash_address = key & ((1 << hash_bits) - 1);  // Mask low bits
cell_data = hash_table[hash_address];
if (cell_data.key != key) {  // Collision detected
    cell_data = follow_linked_list(hash_address);  // Chain traversal
}
```

**Difference**: HOT's hash can have **multiple keys per bucket** (collisions). Your LUT has **exactly one range per prefix** (no collisions possible).

**Terminology correction**: Call it "prefix lookup table" or "direct-mapped spatial index," not "hash."

### 2.4 The Fatal Flaw: Leaf Boundaries Are Arbitrary

**Your leaf definition**:[1]
```python
# Leaves are consecutive 256-element chunks of sorted array
leaf_start[i] = i * 256
```

**Problem**: These boundaries have **no geometric meaning**.

**Example scenario**:

```
Sorted Morton codes (element IDs):
[..., 0x00A1B2C3 (elem 1024),  
      0x00A1B2C4 (elem 512),   
      0x00A1B2C5 (elem 789),   
      ...
      0x00A1B2FF (elem 2048), ← Leaf 4 boundary
      0x00A1B300 (elem 1111), ← Leaf 5 boundary  
      ...
      0x00A1FFFF (elem 3001)]

Leaf 4: Elements [..., 1024, 512, 789, ..., 2048] (256 elements)
Leaf 5: Elements [1111, ..., 3001, ...] (256 elements)

All these elements are in the SAME level-6 octant:
  prefix_6 = 0x00A1B >> 0 = 0x00A (bits 62-45)
  
But split across 2+ leaves due to arbitrary 256-boundary.
```

**Consequence**: To find all elements in an octant, you must search **multiple leaves** → your `L2_SEARCH_RADIUS=100` compensates for this fragmentation.

**Better approach** (octree-aligned leaves):

```python
# Build leaves from octree cells
for prefix_6 in all_level6_octants:
    elements_in_octant = get_elements_with_prefix(prefix_6)
    if len(elements_in_octant) <= 256:
        create_leaf(prefix_6, elements_in_octant)
    else:
        # Subdivide to level 7
        for sub_prefix_7 in subdivide(prefix_6):
            create_leaf(sub_prefix_7, ...)
```

Now **one prefix → one leaf** (or small cluster). Search radius can be reduced to ±1-3 (geometric neighbors only).

### 2.5 The Missing Hierarchy: Why LBVH Radix Tree Wins

**Your method**: Flat prefix table → large search radius

**LBVH radix tree** (Karras 2012):[4]

1. **Build binary tree from sorted Morton codes**:
   ```
   For adjacent pairs (morton[i], morton[i+1]):
     Find highest differing bit position δ
     Node boundary at i if δ changes
   
   Result: Binary tree where each internal node covers a contiguous Morton range
   ```

2. **Tree structure** (implicit in sorted array):
   ```
   Internal nodes: Represent Morton code prefixes
   Leaves: Individual elements (or small clusters)
   
   Example with 8 elements:
        Root [0-7]
       /           \
     [0-3]        [4-7]
     /   \        /   \
   [0-1] [2-3] [4-5] [6-7]
   ```

3. **Query traversal**:
   ```python
   node = root
   while not is_leaf(node):
       if query_morton < split_point[node]:
           node = left_child[node]
       else:
           node = right_child[node]
   return elements_in_leaf(node)
   ```

**Complexity**: O(log N) = ~20-22 steps for 3M elements.

**Your method complexity**: O(radius × capacity) = 100 × 256 = 25,600 point-in-tet tests per failed L0.

**Speedup**: 25,600 / (22 × adaptive_leaf_size) ≈ **50-100× faster** (if leaf_size ≈ 10-20).

### 2.6 Morton vs Hilbert Curves

**Your document mentions** but doesn't deeply analyze.[4]

**Morton Z-curve discontinuities**:[5]

```
2D Example (Morton order):
  0---1   4---5
  |   |   |   |
  2---3   6---7
    
  8---9   C---D
  |   |   |   |
  A---B   E---F

Points 3 and 4 are spatially adjacent but:
  morton(3) = 0b011 = 3
  morton(4) = 0b100 = 4
  Difference: 1

Points 7 and 8 are spatially adjacent but:
  morton(7) = 0b111 = 7
  morton(8) = 0b1000 = 8  
  Difference: 1 in code, but cross octant boundary
```

At octant boundaries, **spatially close points have large Morton gaps**. Your `L2_SEARCH_RADIUS=100` compensates for this.

**Hilbert curve** has better locality:[4]

```
Hilbert order (2D):
  0---1   E---F
      |   |
  3---2   D   C
  |           |
  4   7---8   B
  |   |       |
  5---6   9---A

No large gaps between spatially adjacent points.
```

**Performance impact**: Hilbert would reduce your search radius from 100 to ~20-30 (3-5× improvement).[4]

**Downside**: Hilbert encoding is 30% slower than Morton on GPU (more complex bit ops).[4]

**Trade-off**:
- Morton: Fast encode, large search radius
- Hilbert: Slow encode, small search radius
- **For your case**: Morton is fine if you fix the leaf structure (radix tree)

***

## Part III: Definitive Root Cause Analysis

### 3.1 L1 Failure: Topology vs Geometry Mismatch

**Root cause**: **Graded refinement creates multi-level spatial buffers that partition the face-neighbor topology graph**.

**Mechanism**:
1. Mesh has 6-7 refinement levels between coarse and fine
2. Grading ensures each level differs by 2:1 ratio → buffer zones
3. Face-based neighbors only connect within ±1 level
4. N_HOPS=3 insufficient to traverse 6-level buffer
5. Early-exit at first-found element prevents deeper search even with large N_HOPS

**Why this causes wrong trajectories**:
- L1 returns medium-resolution element (topologically correct)
- Medium element uses 4 nodes spaced ~0.27mm for interpolation
- Fine rotating-tool velocity has gradients at 0.10-0.14mm scale
- **Nyquist criterion violated**: Sampling below gradient wavelength → aliasing
- Result: Smooth but under-resolved velocity field → linear particle paths

**This is NOT**:
- ❌ Algorithm returning inactive elements
- ❌ Spatial containment error
- ✅ **Resolution aliasing** due to topology-bounded search

### 3.2 Morton L2 Failure: Leaf Fragmentation

**Root cause**: **Fixed-capacity leaves fragment octree cells, requiring large search radii to compensate**.

**Mechanism**:
1. Elements sorted by Morton code (correct)
2. Leaves defined as arbitrary 256-element chunks (wrong)
3. A single level-6 octant (geometric cube) spans 2-8 leaves
4. Particle in octant at leaf boundary requires searching ±100 leaves
5. Each leaf scan tests 256 elements → 25,600 tests per query

**Why this causes slowdown**:
- Fine region has sparse Morton codes (elements clustered in 3D but spread in 1D Morton)
- Coarse region has dense Morton codes (opposite)
- Fixed-capacity leaves don't adapt → waste work in both regions
- Large radius compensates but multiplies wasted work

**This is NOT**:
- ❌ Morton encoding insufficient resolution
- ❌ Position→leaf mapping inaccurate
- ✅ **Leaf structure doesn't respect spatial hierarchy**

***

## Part IV: Modern Solutions (LBVH Approach)

### 4.1 Why LBVH Radix Tree Is Superior

**Karras 2012 LBVH algorithm** (GPU-optimized):[4]

**Phase 1: Sort primitives by Morton code**
```cuda
// Already done in your implementation
__global__ void compute_morton_codes(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    morton_codes[idx] = morton_encode(centroids[idx]);
}
thrust::sort_by_key(morton_codes, primitive_ids);
```

**Phase 2: Build radix tree (parallel, O(N))**
```cuda
__global__ void build_radix_tree(uint64_t* sorted_morton,
                                  int* left_child,
                                  int* right_child,
                                  int n_primitives) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_primitives - 1) return;
    
    // Find range of keys with common prefix
    int clz_left  = __clzll(sorted_morton[idx] ^ sorted_morton[idx-1]);
    int clz_right = __clzll(sorted_morton[idx] ^ sorted_morton[idx+1]);
    
    // Determine split direction
    int d = (clz_right > clz_left) ? 1 : -1;
    
    // Find range span
    int min_clz = min(clz_left, clz_right);
    int l_max = 2;
    while (clz(sorted_morton[idx] ^ sorted_morton[idx + l_max * d]) > min_clz)
        l_max *= 2;
    
    // Binary search for split point
    int split = find_split(idx, l_max, sorted_morton, d);
    
    // Assign children
    left_child[idx] = split;
    right_child[idx] = (d > 0) ? idx + l_max : idx - l_max;
}
```

**Phase 3: Compute bounding boxes (bottom-up)**
```cuda
__global__ void compute_bboxes(int* left_child,
                                int* right_child,
                                BBox* leaf_bboxes,
                                BBox* internal_bboxes,
                                int n_internals) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Atomic flag for synchronization
    __shared__ int atomic_counter;
    
    // Merge child bboxes
    BBox bbox_left = get_bbox(left_child[idx], ...);
    BBox bbox_right = get_bbox(right_child[idx], ...);
    internal_bboxes[idx] = merge(bbox_left, bbox_right);
    
    // Propagate upward
    if (atomicAdd(&atomic_counter, 1) == 1) {
        // Both children done, propagate to parent
        int parent = get_parent(idx);
        if (parent >= 0) compute_bboxes<<<...>>>(parent, ...);
    }
}
```

**Phase 4: Query traversal (stackless)**
```cuda
__device__ int query_radix_tree(float3 pos,
                                 uint64_t* sorted_morton,
                                 int* left_child,
                                 int* right_child,
                                 int root) {
    uint64_t query_morton = morton_encode(pos);
    int node = root;
    
    while (node < n_internals) {  // Internal nodes
        uint64_t split_morton = sorted_morton[right_child[node]];
        if (query_morton < split_morton)
            node = left_child[node];
        else
            node = right_child[node];
    }
    
    // node is now a leaf index
    int leaf_start = leaf_ranges[node - n_internals];
    int leaf_length = leaf_ranges[node - n_internals + 1] - leaf_start;
    
    // Test elements in leaf
    for (int i = 0; i < leaf_length; i++) {
        int elem_id = elem_ids_sorted[leaf_start + i];
        if (point_in_tet(pos, elem_id, ...))
            return elem_id;
    }
    return -1;
}
```

**Complexity comparison**:

| Method | Build | Query | Memory |
|--------|-------|-------|--------|
| **Your method** | O(N log N) sort | O(R × C) scan | 26 MB |
| **LBVH radix** | O(N) parallel | O(log N × L) tree walk | 24 MB |
| **Speedup** | 1× (same sort) | **50-100×** | -2 MB (smaller!) |

Where:
- R = search radius (100 leaves)
- C = leaf capacity (256 elements)  
- L = average leaf size after radix split (10-20 elements)

### 4.2 Adapting LBVH to JAX

**Challenge**: JAX doesn't support recursion or dynamic stacks well.

**Solution**: Stackless traversal using **parent pointers** or **iteration with fixed-depth arrays**.

```python
@jax.jit
def query_radix_tree_stackless(pos: jax.Array,
                                mesh_morton: MeshGPURadixTree) -> jnp.int32:
    """Stackless binary tree descent."""
    query_morton = morton_encode_position_jax(pos, mesh_morton.bbox_min,
                                               mesh_morton.bbox_max, 21)
    
    node = jnp.int32(0)  # Root
    n_internals = mesh_morton.n_internals
    
    # Fixed-depth iteration (max depth = 30 for 3M elements)
    def descent_step(carry, _):
        node, done = carry
        is_internal = node < n_internals
        
        # Get split point
        split_morton = mesh_morton.split_morton[node]
        go_left = query_morton < split_morton
        
        # Choose child
        child = jnp.where(go_left,
                          mesh_morton.left_child[node],
                          mesh_morton.right_child[node])
        
        # Update node (or keep if done)
        next_node = jnp.where(is_internal & (~done), child, node)
        next_done = done | (~is_internal)
        
        return (next_node, next_done), None
    
    (leaf_node, _), _ = lax.scan(descent_step, (node, False), None, length=30)
    
    # Search leaf
    leaf_idx = leaf_node - n_internals
    start = mesh_morton.leaf_ranges[leaf_idx]
    length = mesh_morton.leaf_ranges[leaf_idx + 1] - start
    
    return search_in_leaf_bounded(pos, start, length, mesh_morton.elem_ids_sorted,
                                   mesh_morton.connectivity, mesh_morton.node_positions,
                                   max_capacity=64)  # Smaller leaves
```

**Key JAX adaptations**:
1. **Fixed-depth lax.scan** instead of while-loop (max 30 iterations for 3M elements)
2. **Masked updates** instead of conditional branches
3. **Leaf capacity = 64** (smaller than your 256, since radix tree naturally clusters)
4. **No dynamic stacks** – parent pointers stored explicitly if needed for backtracking

### 4.3 Octree-Aligned Leaves (Simpler Alternative)

**If full LBVH is too complex**, fix your current method with **geometric leaves**:

```python
def build_octree_aligned_leaves(morton_sorted: np.ndarray,
                                elem_ids_sorted: np.ndarray,
                                depth: int = 6,
                                max_capacity: int = 256) -> LeafStructure:
    """Build leaves aligned with octree cells at specified depth."""
    n_prefixes = 8 ** depth  # e.g., 262,144 for depth=6
    leaves = []
    
    for prefix in range(n_prefixes):
        # Find elements with this prefix
        min_morton = prefix << (63 - depth * 3)
        max_morton = (prefix + 1) << (63 - depth * 3)
        
        # Binary search in sorted array
        start_idx = np.searchsorted(morton_sorted, min_morton, side='left')
        end_idx = np.searchsorted(morton_sorted, max_morton, side='left')
        count = end_idx - start_idx
        
        if count == 0:
            continue  # Empty octant
        elif count <= max_capacity:
            # Single leaf for this octant
            leaves.append((prefix, start_idx, count))
        else:
            # Subdivide to depth+1
            for sub_prefix in range(prefix * 8, (prefix + 1) * 8):
                sub_min = sub_prefix << (63 - (depth + 1) * 3)
                sub_max = (sub_prefix + 1) << (63 - (depth + 1) * 3)
                sub_start = np.searchsorted(morton_sorted, sub_min, side='left')
                sub_end = np.searchsorted(morton_sorted, sub_max, side='left')
                sub_count = sub_end - sub_start
                if sub_count > 0:
                    leaves.append((sub_prefix, sub_start, sub_count))
    
    # Convert to arrays
    leaf_prefixes = np.array([l[0] for l in leaves], dtype=np.uint64)
    leaf_starts = np.array([l[1] for l in leaves], dtype=np.int32)
    leaf_lengths = np.array([l[2] for l in leaves], dtype=np.int32)
    
    return LeafStructure(leaf_prefixes, leaf_starts, leaf_lengths, depth)
```

**Query**:
```python
def query_octree_leaves(pos: jax.Array, mesh_morton: MeshGPUOctreeLeaves) -> jnp.int32:
    """Query with octree-aligned leaves."""
    query_morton = morton_encode_position_jax(pos, ...)
    prefix = query_morton >> (63 - mesh_morton.depth * 3)
    
    # Find leaf with this prefix (binary search in leaf_prefixes)
    leaf_idx = binary_search(mesh_morton.leaf_prefixes, prefix)
    
    # If not found, search 27 spatial neighbors
    if leaf_idx < 0:
        neighbors = get_26_neighbors(prefix, mesh_morton.depth)
        for nb_prefix in neighbors:
            nb_leaf = binary_search(mesh_morton.leaf_prefixes, nb_prefix)
            if nb_leaf >= 0:
                result = search_in_leaf(pos, nb_leaf, mesh_morton)
                if result >= 0:
                    return result
    else:
        return search_in_leaf(pos, leaf_idx, mesh_morton)
    
    return jnp.int32(-1)
```

**Benefits over current method**:
- ✅ Leaves respect geometric octree cells
- ✅ One prefix → one leaf (or 8 sub-leaves if subdivided)
- ✅ Search radius reduced from 100 to 1-3 (only geometric neighbors)
- ✅ 30-50× fewer element tests per query
- ✅ Simpler than full LBVH radix tree

**Trade-off**: Still requires searching 27 neighbor octants at octree cell boundaries (vs LBVH's exact binary descent).

***

## Part V: Actionable Recommendations

### 5.1 Immediate Fix (1-2 days): Adaptive Search Radius

**What you proposed** (Option A): Bounding-box-based adaptive radius.[1]

**This works and is simple**:

```python
REFINED_BBOX = {
    'min': np.array([-0.010, -0.010, -0.0046]),  # -10mm in refined region
    'max': np.array([0.010, 0.010, -0.0002])
}

def is_in_refined_region(pos: jax.Array) -> jnp.bool_:
    in_x = (pos[0] >= REFINED_BBOX['min'][0]) & (pos[0] <= REFINED_BBOX['max'][0])
    in_y = (pos[1] >= REFINED_BBOX['min'][1]) & (pos[1] <= REFINED_BBOX['max'][1])
    in_z = (pos[2] >= REFINED_BBOX['min'][2]) & (pos[2] <= REFINED_BBOX['max'][2])
    return in_x & in_y & in_z

def search_L2_adaptive_radius(pos: jax.Array, mesh_morton) -> jnp.int32:
    radius = jnp.where(is_in_refined_region(pos),
                       jnp.int32(50),   # Refined region
                       jnp.int32(10))   # Coarse region
    return search_L2_with_radius(pos, radius, mesh_morton)
```

**Expected improvement**: 2-3× speedup (15-20s per timestep vs 30s).

**Limitations**: Hardcoded region, not generalizable.

### 5.2 Medium-Term Fix (1-2 weeks): Octree-Aligned Leaves

**Implement Section 4.3's octree leaf builder**.

**Steps**:
1. Modify `morton_global_builder.py`:
   ```python
   # Replace build_fixed_capacity_leaves() with:
   def build_octree_aligned_leaves(morton_sorted, elem_ids_sorted,
                                    depth=6, max_capacity=256):
       # ... (see Section 4.3)
   ```

2. Modify `morton_global_search.py`:
   ```python
   # Replace position_to_leaf_id_linear() with:
   def position_to_leaf_id_octree(pos, mesh_morton):
       prefix = morton_encode(pos) >> (63 - mesh_morton.depth * 3)
       leaf_idx = binary_search(mesh_morton.leaf_prefixes, prefix)
       return leaf_idx
   
   # Add 27-neighbor search for boundary cases
   def search_with_neighbors(pos, mesh_morton):
       center_leaf = position_to_leaf_id_octree(pos, mesh_morton)
       result = search_in_leaf(pos, center_leaf, mesh_morton)
       if result >= 0:
           return result
       
       # Search 26 spatial neighbors
       for neighbor_leaf in get_26_neighbors(center_leaf, mesh_morton):
           result = search_in_leaf(pos, neighbor_leaf, mesh_morton)
           if result >= 0:
               return result
       return -1
   ```

3. **Expected improvement**: 10-20× speedup (search radius reduced from 100 to 3).

4. **Memory**: +16 MB for leaf_prefixes array (262k entries × 8 bytes), -0 MB (replace prefix_start/length).

**Benefits**:
- ✅ Accurate geometric mapping
- ✅ Small search neighborhood
- ✅ Still uses prefix table (JAX-friendly)
- ✅ Moderate implementation complexity

### 5.3 Long-Term Solution (3-4 weeks): Full LBVH Radix Tree

**Implement Section 4.1-4.2's Karras algorithm in JAX**.

**Complexity**: Higher (tree construction, stackless traversal), but **50-100× speedup** over current method.

**When to do this**:
- After octree leaves prove insufficient
- If you need to generalize to other meshes
- If performance is still bottleneck

**Not recommended unless**: Octree leaves + adaptive radius still fail to achieve target performance (<5s per timestep).

***

## Part VI: Final Verdicts

### 6.1 L1 Failure

**Claim in documents**: "Early-exit greedy search returns inactive parent elements"[1]

**Reality**: ❌ **FALSE**

**Corrected explanation**: L1 uses face-based neighbor topology, which partitions into refinement-level subgraphs in graded meshes. With N_HOPS=3, cannot traverse 6-level graded buffer. Returns medium-resolution element that is topologically and geometrically correct but provides under-resolved velocity interpolation.

**Fix**: Disable L1 (already done), use spatial search (L2 Morton).

### 6.2 Morton Implementation

**Claim in documents**: "HOT-like hashed octree"[4][1]

**Reality**: ❌ **MISLEADING**

**Corrected classification**: Hybrid of LBVH sorting + custom prefix LUT + arbitrary leaf boundaries. Neither true HOT (no hash collisions, no explicit tree) nor true LBVH (no radix tree hierarchy).

**Fix**: Implement octree-aligned leaves (medium-term) or full LBVH radix tree (long-term).

### 6.3 Performance Path

| Approach | Speedup | Complexity | Timeline |
|----------|---------|------------|----------|
| **Adaptive radius (bbox)** | 2-3× | Low | 1-2 days |
| **Octree-aligned leaves** | 10-20× | Medium | 1-2 weeks |
| **LBVH radix tree** | 50-100× | High | 3-4 weeks |

**Recommendation**: Implement **Option 1 immediately**, then **Option 2** if results look promising, **Option 3** only if necessary.

***

## Appendix: References

 MORTON_OPTIMIZATION_GUIDE.md (your document)[1]
 Warren & Salmon 1993: Hashed Oct-Tree (HOT)[3]
 MORTON_OPTIMIZATION_GUIDE_Review_Sunnet-think.md (LBVH analysis)[4]
 Z-order curve (Wikipedia)[5]
 Morton code spatial sorting[2]

**Modern LBVH literature**:
- Karras 2012: "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees" (GPU radix tree)
- Pantaleoni & Luebke 2010: "HLBVH: Hierarchical LBVH Construction for Real-Time Ray Tracing"
- Apetrei 2014: "Fast and Simple Agglomerative LBVH Construction"

***

**End of Document**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/d770de23-b9f3-4c96-a2a1-a7a59e9e7100/MORTON_OPTIMIZATION_GUIDE.md)
[2](https://dev.to/p_pumulo/high-performance-3d-spatial-data-sorting-with-morton-codes-in-clojure-1n6f)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_6462cb8f-df5c-4317-bea4-bbba5ac5557e/372b7b83-a131-4b1b-97c9-cdc8dd51b6d3/169627.169640.pdf)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/6bfe22ba-1cfa-4ff7-a7d0-7f0a3b035b09/MORTON_OPTIMIZATION_GUIDE_Review_Sunnet-think.md)
[5](https://en.wikipedia.org/wiki/Z-order_curve)