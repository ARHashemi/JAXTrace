# Comprehensive Roadmap: Achieving 100K p/s + Zero Retention + Time-Dependent Mesh Support

**Date**: 2025-12-31
**Goal**: Zero retention (100% particle tracking accuracy), 100K p/s throughput, full time-dependent mesh support
**Framework**: JAX with fully-fused RK4 architecture

---

## Executive Summary: Integrated Findings

After exhaustive analysis combining:
1. Literature review of modern GPU search methods (LBVH, RT Cores, kNN hashing)
2. Critical review from ADVANCED_SPATIAL_SEARCH_CRITICAL_REVIEW_SUNNET.md
3. Detailed paper analysis (Jakob & Guthe 2021, Morrical et al. 2020, Karras 2013)
4. Current implementation diagnostics (7K p/s, 83.74% initial assignment)

**Key Conclusions**:

| Finding | Source | Implication |
|---------|--------|-------------|
| **100K p/s is unrealistic on single GPU** | [2] Lines 173-200 | Theoretical JAX ceiling: 50-70K p/s |
| **Morton octree IS state-of-the-art for JAX** | [1][2] | No algorithm replacement needed |
| **Current 7K p/s is due to bugs, not algorithm** | [1] Lines 204-278 | Fix bugs → 15-20K p/s immediately |
| **Node-based search helps refinement boundaries** | [1] Lines 126-168, [2] Lines 456-485 | +5-10% retention potential |
| **Multi-GPU required for 100K p/s** | [2] Lines 488-512 | 4-8 GPUs needed |
| **Zero retention physically impossible** | [2] Lines 598-600 | 95-98% realistic max |

**Recommendation**: Phased approach with realistic milestones

---

## Phase 1: Critical Bug Fixes (13 hours) 🔴 **URGENT - IMMEDIATE ACTION**

### Expected Results After Phase 1
- **Throughput**: 15-20K p/s (from 7K) → +186-286% improvement
- **Retention**: 90-95% @ step 100 (from 83.74%) → +7-13% improvement
- **Dev time**: 13 hours total
- **Risk**: Low (minor code changes, no architectural changes)

---

### 1.1 Fix Initial Assignment Failure (1 hour)

**Problem**: 16% particles (7,806/48,000) outside mesh bounds before tracking starts
**Source**: [1] Lines 208-228, [2] Lines 424-429

**Root Cause Analysis**:
```python
# Current (WRONG)
positions = sample_uniform_box(seeding_volume)
# seeding_volume may extend beyond mesh_bbox → particles spawn outside mesh
```

**Diagnostic**:
```python
# Add to production_tracking_fully_fused_timedep.py after particle seeding
unassigned_mask = element_ids_gpu == -1
unassigned_pos = positions_gpu[unassigned_mask]

outside_x = (unassigned_pos[:, 0] < bbox_min[0]) | (unassigned_pos[:, 0] > bbox_max[0])
outside_y = (unassigned_pos[:, 1] < bbox_min[1]) | (unassigned_pos[:, 1] > bbox_max[1])
outside_z = (unassigned_pos[:, 2] < bbox_min[2]) | (unassigned_pos[:, 2] > bbox_max[2])
outside_bbox = outside_x | outside_y | outside_z

print(f"Particles outside bbox: {jnp.sum(outside_bbox)}/{jnp.sum(unassigned_mask)}")
```

**Fix Implementation**:

**File**: [production_tracking_fully_fused_timedep.py](production_tracking_fully_fused_timedep.py)

**Location**: After line where particles are seeded (before initial assignment)

```python
# BEFORE INITIAL ASSIGNMENT: Clip particles to mesh bounding box
print("\n=== Clipping particle positions to mesh bounds ===")
original_positions = positions.copy()

# Add 1% safety margin to avoid boundary numerical issues
margin = 0.01
bbox_min_safe = mesh_bbox_min + margin * (mesh_bbox_max - mesh_bbox_min)
bbox_max_safe = mesh_bbox_max - margin * (mesh_bbox_max - mesh_bbox_min)

positions_clipped = jnp.clip(positions, bbox_min_safe, bbox_max_safe)
positions = positions_clipped

# Diagnostic: How many particles were clipped?
moved = jnp.sum(jnp.any(positions != original_positions, axis=1))
print(f"Particles clipped to mesh bounds: {moved}/{len(positions)}")
print(f"Mesh bounds: [{mesh_bbox_min[0]:.4f}, {mesh_bbox_max[0]:.4f}] × "
      f"[{mesh_bbox_min[1]:.4f}, {mesh_bbox_max[1]:.4f}] × "
      f"[{mesh_bbox_min[2]:.4f}, {mesh_bbox_max[2]:.4f}]")
```

**Expected Gain**: Initial assignment 95-98% (vs 83.74%)

**Test**:
```bash
python production_tracking_fully_fused_timedep.py 2>&1 | tee logs/phase1_initial_assignment_fix.log
```

**Success Criteria**:
- Initial assignment >95%
- No crashes
- Retention @ step 100 improves by 5-10%

---

### 1.2 Multi-Leaf Search Optimization (4 hours)

**Problem**: 67% throughput loss (21K → 7K p/s) from searching 3 leaves when most prefixes have only 1
**Source**: [1] Lines 230-254, [2] Lines 431-441

**Root Cause**:
```python
# Current: ALWAYS searches 3 leaves (unrolled)
elem_0, found_0 = search_single_leaf(0, -1, False)
elem_1, found_1 = search_single_leaf(1, elem_0, found_0)  # Wasted if found_0 = True
elem_2, found_2 = search_single_leaf(2, elem_1, found_1)  # Wasted if found_1 = True

# JAX compiles all 3 branches, even if early exit via jnp.where
```

**Statistics** (from previous logs):
- 1 leaf: ~90% of prefixes (fast path)
- 2 leaves: ~8% of prefixes
- 3+ leaves: ~2% of prefixes

**Fix Implementation**:

**File**: [jaxtrace/gpu/search/morton_global_search.py](jaxtrace/gpu/search/morton_global_search.py)

**Location**: Lines 697-727 (current multi-leaf unrolled search)

**REPLACE**:
```python
# OLD: Unrolled 3-leaf search (always executes all branches)
def search_single_leaf(leaf_offset, current_elem, current_found):
    # ... existing logic ...
    pass

elem_0, found_0 = search_single_leaf(0, -1, False)
elem_1_search, found_1_search = search_single_leaf(1, elem_0, found_0)
elem_1 = jnp.where(found_0, elem_0, elem_1_search)
found_1 = found_0 | found_1_search
# ... etc for leaf 2
```

**WITH**:
```python
from jax import lax

# NEW: Use lax.switch to branch on num_leaves (early exit)
def search_1_leaf(pos, first_leaf, mesh_gpu):
    """Fast path: Single leaf (90% of prefixes)."""
    return search_in_leaf_global(pos, first_leaf, mesh_gpu)

def search_2_leaves(pos, first_leaf, mesh_gpu):
    """2 leaves: Sequential with early exit."""
    # Leaf 0
    elem_0 = search_in_leaf_global(pos, first_leaf, mesh_gpu)
    # Leaf 1 (only if leaf 0 failed)
    elem_1 = lax.cond(
        elem_0 >= 0,
        lambda: elem_0,  # Found in leaf 0, return it
        lambda: search_in_leaf_global(pos, first_leaf + 1, mesh_gpu)  # Try leaf 1
    )
    return elem_1

def search_3_leaves(pos, first_leaf, mesh_gpu):
    """3+ leaves: Sequential with early exit."""
    # Leaf 0
    elem_0 = search_in_leaf_global(pos, first_leaf, mesh_gpu)
    # Leaf 1 (only if leaf 0 failed)
    elem_1 = lax.cond(
        elem_0 >= 0,
        lambda: elem_0,
        lambda: search_in_leaf_global(pos, first_leaf + 1, mesh_gpu)
    )
    # Leaf 2 (only if leaf 1 failed)
    elem_2 = lax.cond(
        elem_1 >= 0,
        lambda: elem_1,
        lambda: search_in_leaf_global(pos, first_leaf + 2, mesh_gpu)
    )
    return elem_2

# Dispatch based on num_leaves
# lax.switch executes ONLY the selected branch (not all like jnp.where)
elem_neighbor = lax.switch(
    jnp.clip(num_leaves - 1, 0, 2),  # 0 = 1 leaf, 1 = 2 leaves, 2 = 3+ leaves
    [
        lambda: search_1_leaf(pos, first_leaf, mesh_gpu),  # Case 0: 1 leaf
        lambda: search_2_leaves(pos, first_leaf, mesh_gpu),  # Case 1: 2 leaves
        lambda: search_3_leaves(pos, first_leaf, mesh_gpu)   # Case 2: 3+ leaves
    ]
)
```

**Why This Works**:
- `lax.switch` compiles all branches but **executes only one** at runtime (unlike `jnp.where`)
- 90% of particles hit fast path (1 leaf) → no wasted searches
- Inside each branch, `lax.cond` provides early exit (not inside vmap, safe to use)

**Expected Gain**: 15-20K p/s throughput (vs 7K)

**Test**:
```bash
python production_tracking_fully_fused_timedep.py 2>&1 | tee logs/phase1_multileaf_optimization.log
```

**Success Criteria**:
- Throughput >15K p/s
- No OOM errors (lax.switch doesn't have vmap issue like lax.cond)
- Retention stays same or improves slightly

---

### 1.3 Adaptive L1 Hop Count for Refinement Boundaries (8 hours)

**Problem**: L1 multi-hop search fails when particle crosses from refined to coarse region
**Source**: [1] Lines 257-278, [2] Lines 442-447

**Root Cause**:
```
Particle in refined element (size=0.01) moves to coarse element (size=0.1)
L1 searches: refined neighbors (all size=0.01, within 3 hops)
Never reaches coarse element (outside 3-hop neighborhood)
→ L1 fails → L2 must find it (expensive)
```

**Current L1 Implementation**:
```python
# Fixed 3 hops for all particles
N_HOPS = 3
```

**Fix Implementation**:

**File**: [jaxtrace/gpu/search/morton_global_search.py](jaxtrace/gpu/search/morton_global_search.py)

**Location**: In `search_l1_single()` function (or wherever L1 multi-hop is implemented)

**ADD**:
```python
def search_l1_adaptive_single(pos, start_elem_id, mesh_gpu):
    """
    Adaptive L1 multi-hop search with refinement-aware hop count.

    Detects element size mismatches and extends search radius accordingly.
    """
    # Get element volumes (precomputed in mesh_gpu)
    start_volume = mesh_gpu.element_volumes[start_elem_id]

    # Get neighbors of starting element
    neighbor_ids = mesh_gpu.element_neighbors[start_elem_id]  # Shape: (N_NEIGHBORS,)
    valid_neighbors = neighbor_ids >= 0
    neighbor_volumes = jnp.where(
        valid_neighbors,
        mesh_gpu.element_volumes[neighbor_ids],
        start_volume  # Pad invalid with start volume to avoid div-by-zero
    )

    # Compute size ratio: small/large
    # If start is refined and neighbors are coarse → ratio << 1
    avg_neighbor_volume = jnp.mean(jnp.where(valid_neighbors, neighbor_volumes, start_volume))
    size_ratio = start_volume / (avg_neighbor_volume + 1e-10)

    # Adaptive hop count
    # If current element much smaller than neighbors → crossing to coarse region
    # Extend search radius to 6 hops (2× normal)
    n_hops_adaptive = jnp.where(
        size_ratio < 0.1,  # 10× size difference threshold
        6,  # Extended search for refinement boundary
        3   # Normal search for same-level elements
    )

    # Multi-hop search with adaptive count
    # Unroll max 6 iterations (supports both 3-hop and 6-hop cases)
    def hop_step(hop, state):
        current_elem, found = state

        # Check if we should search this hop
        should_search = (hop < n_hops_adaptive) & (~found) & (current_elem >= 0)

        # Get neighbors of current element
        neighbors = mesh_gpu.element_neighbors[current_elem]

        # Search all neighbors at this hop
        def check_neighbor(i, neighbor_state):
            neighbor_elem, neighbor_found = neighbor_state
            neighbor_id = neighbors[i]
            valid = (neighbor_id >= 0) & (~neighbor_found)

            # Point-in-tet test
            result = jnp.where(
                valid,
                point_in_tet_test(pos, neighbor_id, mesh_gpu),
                jnp.int32(-1)
            )
            improved = result >= 0
            return (jnp.where(improved, result, neighbor_elem), neighbor_found | improved)

        # Search all neighbors (unroll neighbor loop)
        hop_result, hop_found = lax.fori_loop(
            0,
            mesh_gpu.max_neighbors,  # Typically 4 for tets
            check_neighbor,
            (current_elem, found)
        )

        # Update state only if we should search this hop
        return (
            jnp.where(should_search, hop_result, current_elem),
            jnp.where(should_search, hop_found, found)
        )

    # Multi-hop loop (max 6 hops)
    final_elem, final_found = lax.fori_loop(
        0,
        6,  # Max hops
        hop_step,
        (start_elem_id, False)
    )

    return jnp.where(final_found, final_elem, jnp.int32(-1))

# REPLACE existing search_l1_single with search_l1_adaptive_single
```

**Why This Works**:
- Detects refinement boundary crossings via volume ratio
- Automatically extends search radius when needed
- 90% of particles stay at 3 hops (no performance impact)
- 10% crossing boundaries get 6 hops (2× cost but finds particle)

**Expected Gain**: +3-5% retention from boundary crossings

**Test**:
```bash
python production_tracking_fully_fused_timedep.py 2>&1 | tee logs/phase1_adaptive_l1.log
```

**Success Criteria**:
- Retention @ step 100 improves by 3-5%
- Throughput stays >15K p/s (adaptive logic minimal overhead)
- No crashes or OOM

---

### Phase 1 Milestones & Testing

**After implementing all 3 fixes**, run comprehensive test:

```bash
# Full production run with all Phase 1 fixes
python production_tracking_fully_fused_timedep.py 2>&1 | tee logs/phase1_complete.log
```

**Expected Results**:

| Metric | Before Phase 1 | After Phase 1 | Improvement |
|--------|----------------|---------------|-------------|
| **Initial assignment** | 83.74% | **95-98%** | +11-14% |
| **Retention @ step 100** | 83.66% | **90-95%** | +7-13% |
| **Retention @ step 400** | 78.60% | **87-92%** | +8-13% |
| **Throughput** | 7,000 p/s | **15,000-20,000 p/s** | +114-186% |
| **Dev time** | - | 13 hours | - |

**If Phase 1 achieves >90% retention @ step 100**: Proceed to Phase 2
**If Phase 1 achieves <90% retention**: Debug remaining issues before Phase 2

---

## Phase 2: Node-Based Boundary Search (40 hours) 🔵 **IF >90% RETENTION NEEDED**

**Only implement if Phase 1 doesn't achieve 90%+ retention in refined regions**

### Expected Results After Phase 2
- **Retention**: 93-98% @ step 100 (from 90-95%) → +3-8% improvement
- **Throughput**: 12-18K p/s (slight slowdown from node kNN overhead)
- **Dev time**: 40 hours
- **Risk**: Medium (new data structure, JAX complexity)

---

### 2.1 Rationale: Why Node-Based Helps

**Problem**: Element-based search misses particles at refinement boundaries
**Source**: [1] Lines 126-168, [2] Lines 456-485

**Element-Based (Current)**:
```
Particle moves from refined elem (0.01) to coarse elem (0.1)
Element octree: Spatial cells sized for largest elements
Cell contains 1-10 coarse elements + 100-1000 refined elements
Must search all elements in cell → expensive + misses
```

**Node-Based (Proposed)**:
```
Find 3 nearest mesh nodes to particle position
Nodes on boundary shared by BOTH refined + coarse elements
Check incident elements of found nodes (20-50 elements total)
Direct containment → finds boundary crossing efficiently
```

**Key Insight from PUMI-Tally paper** [1]:
> "Unlike other methods, we do not require expensive particle-in-element localization procedures"

But they don't handle refinement boundaries! Node-based captures both sides.

---

### 2.2 Implementation Strategy

**Step 1: Precompute Node Octree (5 hours)**

**File**: New file `jaxtrace/gpu/forest/node_octree_builder.py`

```python
import jax.numpy as jnp
from jaxtrace.gpu.search.morton_global_search import compute_morton_code_3d

def build_node_octree(node_positions, table_depth=7):
    """
    Build Morton octree over mesh nodes (not elements).

    Args:
        node_positions: (N_nodes, 3) array of node coordinates
        table_depth: Octree depth (7 for ~300K nodes)

    Returns:
        node_octree: Dictionary with prefix table, node lists per leaf
    """
    n_nodes = len(node_positions)

    # Compute bounding box
    bbox_min = jnp.min(node_positions, axis=0)
    bbox_max = jnp.max(node_positions, axis=0)

    # Compute Morton codes for all nodes
    morton_codes = jnp.zeros(n_nodes, dtype=jnp.uint64)
    for i in range(n_nodes):
        morton_codes = morton_codes.at[i].set(
            compute_morton_code_3d(node_positions[i], bbox_min, bbox_max)
        )

    # Sort nodes by Morton code
    sort_indices = jnp.argsort(morton_codes)
    sorted_morton = morton_codes[sort_indices]
    sorted_node_ids = jnp.arange(n_nodes)[sort_indices]

    # Build prefix table (same as element octree)
    shift = 63 - (table_depth * 3)
    prefixes = sorted_morton >> shift

    # Find start/length for each prefix
    max_prefix = 1 << (table_depth * 3)
    prefix_start = jnp.full(max_prefix, -1, dtype=jnp.int32)
    prefix_length = jnp.zeros(max_prefix, dtype=jnp.int32)

    unique_prefixes, prefix_counts = jnp.unique(prefixes, return_counts=True)

    # Populate prefix table (loop over unique prefixes)
    cumsum = jnp.cumsum(jnp.concatenate([jnp.array([0]), prefix_counts[:-1]]))
    prefix_start = prefix_start.at[unique_prefixes].set(cumsum)
    prefix_length = prefix_length.at[unique_prefixes].set(prefix_counts)

    return {
        'prefix_start': prefix_start,
        'prefix_length': prefix_length,
        'sorted_node_ids': sorted_node_ids,
        'node_positions': node_positions,
        'bbox_min': bbox_min,
        'bbox_max': bbox_max,
        'table_depth': table_depth
    }
```

**Step 2: Build Node-to-Element Connectivity (8 hours)**

**Challenge**: Jagged arrays (variable valence) incompatible with JAX vmap

**Solution**: Pad to max valence + use masks

```python
def build_node_to_elements_padded(connectivity, n_nodes, max_valence=50):
    """
    Build node → incident elements mapping (padded to fixed shape).

    Args:
        connectivity: (N_elements, 4) array of element→node connectivity
        n_nodes: Total number of nodes
        max_valence: Maximum elements per node (pad to this)

    Returns:
        node_to_elements: (N_nodes, max_valence) padded array (-1 for invalid)
        node_valence: (N_nodes,) actual number of incident elements per node
    """
    n_elements = len(connectivity)

    # Count valence for each node
    node_valence = jnp.zeros(n_nodes, dtype=jnp.int32)
    for elem_id in range(n_elements):
        for local_node in range(4):  # 4 nodes per tet
            node_id = connectivity[elem_id, local_node]
            node_valence = node_valence.at[node_id].add(1)

    # Check max valence
    actual_max_valence = jnp.max(node_valence)
    print(f"Node valence: min={jnp.min(node_valence)}, max={actual_max_valence}, avg={jnp.mean(node_valence):.1f}")

    if actual_max_valence > max_valence:
        print(f"WARNING: max_valence={max_valence} < actual={actual_max_valence}, will truncate!")

    # Allocate padded array
    node_to_elements = jnp.full((n_nodes, max_valence), -1, dtype=jnp.int32)

    # Fill incident elements (CPU loop, done once during preprocessing)
    node_counts = jnp.zeros(n_nodes, dtype=jnp.int32)  # Track current fill count

    for elem_id in range(n_elements):
        for local_node in range(4):
            node_id = connectivity[elem_id, local_node]
            count = node_counts[node_id]

            if count < max_valence:
                node_to_elements = node_to_elements.at[node_id, count].set(elem_id)
                node_counts = node_counts.at[node_id].add(1)

    return node_to_elements, node_valence
```

**Step 3: Node kNN Search (12 hours)**

**File**: Add to `jaxtrace/gpu/search/morton_global_search.py`

```python
def search_node_knn_single(pos, k, node_octree):
    """
    Find k nearest nodes to query position using node octree.

    Args:
        pos: (3,) query position
        k: Number of nearest neighbors
        node_octree: Precomputed node octree dict

    Returns:
        nearest_node_ids: (k,) indices of k nearest nodes (-1 if not found)
    """
    # Compute Morton code for query position
    morton_query = compute_morton_code_3d(
        pos,
        node_octree['bbox_min'],
        node_octree['bbox_max']
    )

    # Extract octree prefix (depth-7)
    shift = 63 - (node_octree['table_depth'] * 3)
    prefix = morton_query >> shift
    prefix_int = jnp.int32(prefix)

    # Look up nodes in this prefix
    first_node_idx = node_octree['prefix_start'][prefix_int]
    num_nodes = node_octree['prefix_length'][prefix_int]

    has_nodes = (first_node_idx >= 0) & (num_nodes > 0)

    if not has_nodes:
        return jnp.full(k, -1, dtype=jnp.int32)

    # Get candidate nodes (up to 64 nodes in this leaf)
    max_candidates = 64
    candidate_indices = jnp.arange(max_candidates)
    candidate_node_ids = jnp.where(
        candidate_indices < num_nodes,
        node_octree['sorted_node_ids'][first_node_idx + candidate_indices],
        -1
    )

    # Compute distances to all candidates
    candidate_positions = node_octree['node_positions'][candidate_node_ids]
    distances = jnp.linalg.norm(candidate_positions - pos, axis=1)

    # Mark invalid candidates with infinite distance
    distances = jnp.where(candidate_node_ids >= 0, distances, jnp.inf)

    # Find k nearest (sort distances, take top k)
    sorted_indices = jnp.argsort(distances)
    nearest_k_indices = sorted_indices[:k]

    nearest_node_ids = candidate_node_ids[nearest_k_indices]

    return nearest_node_ids
```

**Step 4: Hybrid Search (Element vs Node-Based) (15 hours)**

**File**: Modify `jaxtrace/gpu/search/morton_global_search.py`

```python
def search_l2_hybrid_single(pos, cached_elem_id, mesh_gpu, node_octree, node_to_elements):
    """
    Hybrid L2 search: Element-based (bulk) vs Node-based (refinement boundary).

    Strategy:
    1. Detect if particle is near refinement boundary (element size mismatch)
    2. If yes → Use node-based kNN search (captures both refined + coarse)
    3. If no → Use element-based Morton octree (faster for bulk)
    """
    # Detect refinement boundary
    cached_volume = mesh_gpu.element_volumes[cached_elem_id]
    neighbor_ids = mesh_gpu.element_neighbors[cached_elem_id]
    valid_neighbors = neighbor_ids >= 0
    neighbor_volumes = jnp.where(
        valid_neighbors,
        mesh_gpu.element_volumes[neighbor_ids],
        cached_volume
    )
    avg_neighbor_volume = jnp.mean(jnp.where(valid_neighbors, neighbor_volumes, cached_volume))

    # Size ratio: If >5× difference → near boundary
    size_ratio_min = cached_volume / (jnp.max(neighbor_volumes) + 1e-10)
    size_ratio_max = jnp.max(neighbor_volumes) / (cached_volume + 1e-10)
    at_boundary = (size_ratio_min < 0.2) | (size_ratio_max > 5.0)

    # Branch: Node-based vs Element-based
    def node_based_search():
        """Node-based kNN for refinement boundaries."""
        # Find 3 nearest nodes
        nearest_nodes = search_node_knn_single(pos, k=3, node_octree=node_octree)

        # Collect all incident elements (from 3 nodes)
        # Each node has up to max_valence=50 elements
        candidate_elements = []
        for i in range(3):
            node_id = nearest_nodes[i]
            if node_id >= 0:
                incident = node_to_elements[node_id]  # (max_valence,) padded array
                # Append valid elements (not -1)
                valid_incident = incident[incident >= 0]
                candidate_elements.append(valid_incident)

        # Flatten and remove duplicates
        all_candidates = jnp.concatenate(candidate_elements) if candidate_elements else jnp.array([], dtype=jnp.int32)
        unique_candidates = jnp.unique(all_candidates)

        # Search all candidate elements
        def check_candidate(i, state):
            elem_id, found = state
            candidate_elem = unique_candidates[i]
            valid = (i < len(unique_candidates)) & (~found) & (candidate_elem >= 0)

            result = jnp.where(
                valid,
                point_in_tet_test(pos, candidate_elem, mesh_gpu),
                jnp.int32(-1)
            )
            improved = result >= 0
            return (jnp.where(improved, result, elem_id), found | improved)

        final_elem, _ = lax.fori_loop(
            0,
            150,  # Max 3 nodes × 50 elements = 150 (with duplicates)
            check_candidate,
            (jnp.int32(-1), False)
        )
        return final_elem

    def element_based_search():
        """Element-based Morton octree for bulk regions."""
        return search_L2_morton_neighbors_single(pos, mesh_gpu)

    # Dispatch based on boundary detection
    return lax.cond(
        at_boundary,
        node_based_search,
        element_based_search
    )
```

**Why This Works**:
- **90% of particles** in bulk regions → fast element-based search
- **10% at boundaries** → node-based captures both refined + coarse elements
- Boundary nodes shared by both element types → guaranteed to find particle

**Expected Gain**: +5-10% retention in refined regions

---

### 2.3 Integration with Fully-Fused RK4

**File**: Modify `production_tracking_fully_fused_timedep.py`

**Changes**:
1. Precompute node octree and node→element connectivity (one-time setup)
2. Pass `node_octree` and `node_to_elements` to search functions
3. No changes to RK4 structure (just modified L2 search)

```python
# In setup section (before RK4 loop)
print("\n=== Building node octree for refinement boundaries ===")
from jaxtrace.gpu.forest.node_octree_builder import build_node_octree, build_node_to_elements_padded

# Build node octree (300K nodes → depth-7 octree)
node_positions = mesh_vertices  # (N_nodes, 3)
node_octree = build_node_octree(node_positions, table_depth=7)
print(f"Node octree: {len(node_octree['prefix_start'])} prefixes")

# Build node→element connectivity
node_to_elements, node_valence = build_node_to_elements_padded(
    connectivity=mesh_connectivity,  # (N_elements, 4)
    n_nodes=len(node_positions),
    max_valence=50  # Conservative upper bound
)
print(f"Node connectivity: max_valence={jnp.max(node_valence)}, avg={jnp.mean(node_valence):.1f}")

# Upload to GPU
node_octree_gpu = jax.device_put(node_octree)
node_to_elements_gpu = jax.device_put(node_to_elements)

# Modify search to use hybrid L2
# No changes to RK4 loop structure!
```

**Memory Cost**:
- Node octree: ~5 MB (similar to element octree)
- Node→element connectivity: 300K × 50 × 4 bytes = **60 MB**
- Total overhead: **65 MB** (acceptable)

---

### Phase 2 Milestones & Testing

**After implementing node-based search**, run test:

```bash
python production_tracking_fully_fused_timedep.py 2>&1 | tee logs/phase2_node_based.log
```

**Expected Results**:

| Metric | After Phase 1 | After Phase 2 | Improvement |
|--------|---------------|---------------|-------------|
| **Retention @ step 100** | 90-95% | **93-98%** | +3-8% |
| **Retention in refined region** | 85-90% | **95-98%** | +5-13% |
| **Throughput** | 15-20K p/s | **12-18K p/s** | -10-15% (acceptable) |
| **Dev time** | 13 hours | 53 hours | +40 hours |

**If Phase 2 achieves >95% retention**: Goal nearly achieved, proceed to Phase 3 for scaling
**If Phase 2 achieves <95% retention**: Diagnose remaining failure modes (degeneracy, boundary exit)

---

## Phase 3: Multi-GPU Scaling for 100K p/s (80 hours) 🟣 **ARCHITECTURAL CHANGE**

**Only implement if >100K p/s throughput is hard requirement**

### Expected Results After Phase 3
- **Throughput**: 60-120K p/s (with 4-8 GPUs) → 5-10× improvement
- **Retention**: Same as Phase 2 (93-98%)
- **Dev time**: 80 hours
- **Risk**: High (distributed computing, data sharding, communication overhead)
- **Cost**: 4-8 GPUs required

---

### 3.1 Rationale: Why Single GPU Cannot Reach 100K p/s

**Theoretical Analysis** [2] Lines 173-200:

```
RTX 4090 GPU:
- 82.6 TFLOP/s (float32 peak)
- 1 TB/s memory bandwidth

RK4 step per particle:
- 5 searches (k1, k2, k3, k4, final)
- ~94 point-in-tet tests per search (worst case L2)
- 100 FLOPs per point-in-tet
- Total: 5 × 94 × 100 = 47,000 FLOPs/particle

Compute limit:
  82,600 GFLOP/s ÷ 47,000 FLOPs/particle = 1.76M particles/s

Memory bandwidth limit (L0+L1+L2 cache):
  1,000 GB/s ÷ (48K × 94 tests × 64 bytes) = 3.5 steps/s
  = 168K particles/s (memory bottleneck)

JAX efficiency vs CUDA: 30-50% [2] Line 194
  168K × 0.4 = 67K particles/s theoretical JAX max

Realistic after optimization: 50-70K p/s
Your target: 100K p/s → Need 2× parallelization → 2 GPUs minimum
```

**Conclusion**: **Single GPU cannot reach 100K p/s** with JAX. Need 2-8 GPUs.

---

### 3.2 Multi-GPU Strategy: Data Parallelism with JAX pmap

**Approach**: Shard particles across GPUs, track independently, communicate at boundaries

**File**: New file `production_tracking_fully_fused_timedep_multigpu.py`

```python
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

# Step 1: Detect available GPUs
devices = jax.devices("gpu")
n_gpus = len(devices)
print(f"Available GPUs: {n_gpus}")

if n_gpus < 2:
    print("WARNING: Multi-GPU requires 2+ GPUs. Falling back to single GPU.")
    # Use single-GPU version
else:
    print(f"Using {n_gpus} GPUs for distributed particle tracking")

# Step 2: Shard particles across GPUs
n_particles_total = 48000
n_particles_per_gpu = n_particles_total // n_gpus

# Replicate mesh on all GPUs (read-only, no communication needed)
mesh_gpu_replicated = jax.device_put_replicated(mesh_gpu, devices)

# Shard particles (each GPU gets subset)
sharding = PositionalSharding(devices).reshape(n_gpus, 1)

positions_sharded = jax.device_put(
    positions.reshape(n_gpus, n_particles_per_gpu, 3),
    sharding
)
element_ids_sharded = jax.device_put(
    element_ids.reshape(n_gpus, n_particles_per_gpu),
    sharding
)

# Step 3: Define parallel RK4 step (each GPU processes its shard independently)
@jax.pmap  # Parallel map across GPUs
def rk4_step_parallel_gpu(positions_local, element_ids_local, mesh_gpu):
    """
    RK4 step for particles on THIS GPU only.
    No inter-GPU communication during step (particles stay local).
    """
    # Same as single-GPU version, but operates on local shard
    # vmap over n_particles_per_gpu (not n_particles_total)
    return rk4_step_single_gpu_vmap(positions_local, element_ids_local, mesh_gpu)

# Step 4: Run parallel tracking loop
for step in range(n_steps):
    # Each GPU processes its particle shard in parallel
    positions_sharded, element_ids_sharded = rk4_step_parallel_gpu(
        positions_sharded,
        element_ids_sharded,
        mesh_gpu_replicated  # Read-only, no communication
    )

    # Optional: Synchronize active particle counts across GPUs
    active_counts_per_gpu = jnp.sum(element_ids_sharded >= 0, axis=1)
    total_active = jnp.sum(active_counts_per_gpu)

    if step % 10 == 0:
        print(f"Step {step}: Active particles = {total_active} "
              f"(per GPU: {active_counts_per_gpu})")

# Step 5: Gather results from all GPUs
final_positions = jax.device_get(positions_sharded).reshape(n_particles_total, 3)
final_element_ids = jax.device_get(element_ids_sharded).reshape(n_particles_total)

print(f"Final retention: {jnp.sum(final_element_ids >= 0) / n_particles_total * 100:.2f}%")
```

**Why This Works**:
- **Mesh replicated** on all GPUs (read-only, no updates)
- **Particles sharded** across GPUs (independent tracking)
- **No communication** during RK4 steps (particles don't cross GPU boundaries)
- **Linear scaling**: 4 GPUs → 4× throughput

**Expected Scaling**:

| GPUs | Particles/GPU | Throughput/GPU | Total Throughput | Communication Overhead |
|------|---------------|----------------|------------------|------------------------|
| 1 | 48,000 | 15-20K p/s | 15-20K p/s | 0% |
| 2 | 24,000 | 15-20K p/s | **30-40K p/s** | <5% |
| 4 | 12,000 | 15-20K p/s | **60-80K p/s** | <10% |
| 8 | 6,000 | 15-20K p/s | **120-160K p/s** | <15% |

**To reach 100K p/s**: Need **4 GPUs minimum** (60-80K with overhead → 100K achievable)

---

### 3.3 Handling Time-Dependent Mesh in Multi-GPU

**Challenge**: 50 velocity timesteps × 357 MB = 17.85 GB per GPU

**Solution**: Replicate mesh on all GPUs (affordable with modern GPU VRAM)

```python
# Each GPU gets full mesh (all 50 timesteps)
# RTX 4090: 24 GB VRAM
# Mesh: 17.85 GB
# Overhead: 2 GB (octrees, connectivity)
# Particles: 0.5 GB (48K × 12KB sharded across GPUs)
# Total per GPU: ~20 GB (fits in 24 GB)

mesh_all_timesteps_gpu = jax.device_put_replicated(
    mesh_all_timesteps,  # All 50 timesteps
    devices  # Replicate on all GPUs
)

# No rebuild needed during simulation (same as single-GPU)
# Zero communication overhead for mesh access
```

**Memory requirement per GPU**: ~20 GB (fits in RTX 4090 24 GB VRAM)

---

### 3.4 Alternative: Time-Domain Parallelism (Advanced)

**If particle-domain parallelism insufficient**, consider **time-domain parallelism**:

**Approach**: Each GPU processes different timestep slice

```python
# GPU 0: Steps 0-249
# GPU 1: Steps 250-499
# GPU 2: Steps 500-749
# GPU 3: Steps 750-999

# Pipeline: GPU 0 finishes → pass particles to GPU 1 → etc.
# Communication only at timestep boundaries (4 times total)
```

**Pros**:
- ✅ Fits in VRAM (each GPU needs only 10-15 timesteps)
- ✅ No per-step communication

**Cons**:
- ❌ Pipeline latency (must wait for slowest GPU)
- ❌ Load imbalance (later timesteps have fewer active particles)

**Recommendation**: Use **particle-domain parallelism** (simpler, better load balance)

---

### Phase 3 Milestones & Testing

**After implementing multi-GPU**, run scaling test:

```bash
# Test with different GPU counts
CUDA_VISIBLE_DEVICES=0,1 python production_tracking_fully_fused_timedep_multigpu.py  # 2 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python production_tracking_fully_fused_timedep_multigpu.py  # 4 GPUs
```

**Expected Results**:

| GPUs | Throughput | Retention | Speedup vs 1 GPU |
|------|------------|-----------|------------------|
| 1 | 15-20K p/s | 93-98% | 1.0× (baseline) |
| 2 | 30-38K p/s | 93-98% | 1.9-2.0× |
| 4 | **60-76K p/s** | 93-98% | 3.8-4.0× |
| 8 | **120-152K p/s** | 93-98% | 7.6-8.0× |

**If 4 GPUs achieve >60K p/s**: Likely to reach 100K with 6-8 GPUs
**If scaling <3.5× on 4 GPUs**: Communication overhead or load imbalance issue

---

## Phase 4: CUDA Rewrite (300 hours) ❌ **NOT RECOMMENDED**

**Only if abandoning JAX is acceptable**

### Expected Results After Phase 4
- **Throughput**: 50-150K p/s (single GPU) → 3-5× improvement over JAX
- **Retention**: Same as Phase 2 (93-98%)
- **Dev time**: 300+ hours
- **Risk**: Very High (abandon JAX, lose differentiability, massive rewrite)

---

### 4.1 Why CUDA Could Be Faster

**JAX limitations** [2] Lines 173-200:
- JIT overhead: ~30% performance loss vs hand-written CUDA
- Memory layout: JAX compiler decides, may not be optimal
- Kernel fusion: Limited compared to custom CUDA kernels
- Register allocation: No control in JAX

**CUDA advantages**:
- Direct register control (Jakob & Guthe paper: 19× speedup from register heaps)
- Custom memory layouts (coalesced access, shared memory)
- Atomic operations for dynamic indexing
- Warp-level primitives for efficient reductions

**From LBVH paper** [9]:
> "Register-based heap: 2.82 IPC vs 0.15 IPC for global memory heap (19× speedup)"

**But**: This is for **kNN in point clouds**, NOT **point-in-tet in meshes**

---

### 4.2 CUDA Rewrite Scope

**What needs rewriting**:

1. **Morton octree construction** (500 lines CUDA)
   - Radix sort on GPU (use CUB library)
   - Prefix table construction (atomic increments)

2. **Point-in-tet kernels** (200 lines CUDA)
   - Barycentric coordinate computation
   - Determinant calculations (optimized for registers)

3. **Multi-level search (L0+L1+L2)** (800 lines CUDA)
   - Shared memory for element caching
   - Warp-level neighbor search

4. **RK4 integration** (400 lines CUDA)
   - Fused kernel (all 5 searches in single launch)

5. **Time-dependent mesh handling** (300 lines CUDA)
   - Cyclic indexing for velocity fields

**Total**: ~2,200 lines of CUDA C++ (vs 500 lines JAX)

**Dev time estimate**: 300+ hours (debugging, optimization, validation)

---

### 4.3 What You LOSE by Abandoning JAX

| Feature | JAX | CUDA | Impact |
|---------|-----|------|--------|
| **Automatic differentiation** | ✅ `jax.grad` | ❌ Manual | Lose ability to optimize seeding |
| **Easy prototyping** | ✅ Python | ❌ C++ | 10× slower dev cycle |
| **Portability** | ✅ CPU/GPU/TPU | ❌ NVIDIA only | Lock-in to NVIDIA |
| **Debugging** | ✅ Python tools | ❌ cuda-gdb | Harder debugging |
| **Compilation** | ✅ Auto JIT | ❌ Manual nvcc | Build system complexity |

**Critical question** [2] Lines 598-605:
> "Q3: Is JAX a non-negotiable constraint?"

If **yes** → DO NOT do CUDA rewrite
If **no** → CUDA rewrite could achieve 50-150K p/s single GPU

---

### 4.4 Recommendation: DO NOT Rewrite in CUDA

**Reasons**:

1. **Multi-GPU JAX achieves 100K p/s** (Phase 3 with 4-8 GPUs)
2. **300 hours >> 80 hours** (CUDA vs Multi-GPU dev time)
3. **Lose differentiability** (critical for future optimization)
4. **Hardware lock-in** (NVIDIA-only, no AMD/Intel)
5. **LBVH paper speedup doesn't apply** (point clouds ≠ tetrahedral meshes)

**From your critical review** [2] Lines 586-634:
> "Your Morton octree is a Ferrari. You're driving it at 20 mph because of implementation bugs. Don't buy a new car - learn to drive the one you have."

**Verdict**: Fix bugs (Phase 1), add node-based search (Phase 2), scale with GPUs (Phase 3). **DO NOT rewrite in CUDA.**

---

## Summary: Recommended Implementation Order

### Phased Roadmap

| Phase | Goal | Dev Time | Expected Results | Decision Point |
|-------|------|----------|------------------|----------------|
| **Phase 1** | Fix critical bugs | 13 hours | 15-20K p/s, 90-95% retention | If <90% retention → Phase 2 |
| **Phase 2** | Node-based boundary search | 40 hours | 12-18K p/s, 93-98% retention | If <95% retention → diagnose |
| **Phase 3** | Multi-GPU scaling | 80 hours | 60-160K p/s (4-8 GPUs), 93-98% retention | If need >100K p/s |
| **Phase 4** | CUDA rewrite | 300 hours | 50-150K p/s, 93-98% retention | ❌ **NOT RECOMMENDED** |

---

### Critical Decision Points

**After Phase 1 (13 hours)**:

✅ **If retention >90% @ step 100**:
- SUCCESS! Goal nearly achieved
- Consider Phase 3 (multi-GPU) if throughput <100K required
- Skip Phase 2 (node-based) to save 40 hours

❌ **If retention <90% @ step 100**:
- Proceed to Phase 2 (node-based boundary search)
- Diagnose remaining failure modes (degeneracy, boundary exit)

**After Phase 2 (53 hours total)**:

✅ **If retention >95% @ step 100**:
- EXCELLENT! Near-zero particle loss achieved
- Proceed to Phase 3 if throughput >100K required

❌ **If retention <95% @ step 100**:
- Diagnose remaining issues:
  - Degenerate elements (DEGENERACY_THRESHOLD too strict?)
  - Physical boundary exit (particles truly leaving mesh)
  - Velocity interpolation errors (time_idx cycling bug?)

**After Phase 3 (133 hours total)**:

✅ **If 4 GPUs achieve >100K p/s**:
- GOAL ACHIEVED! 🎉
- Production-ready system

❌ **If 4 GPUs achieve <80K p/s**:
- Diagnose communication overhead
- Try 6-8 GPUs
- Last resort: Consider CUDA rewrite (Phase 4)

---

## Realistic Final Expectations

### What IS Achievable

| Metric | Current | Phase 1 | Phase 1+2 | Phase 1+2+3 (4 GPUs) |
|--------|---------|---------|-----------|----------------------|
| **Throughput** | 7K p/s | 15-20K p/s | 12-18K p/s | **60-80K p/s** |
| **Retention** | 83.74% | 90-95% | 93-98% | 93-98% |
| **Active @ step 100** | 40,158 | 43,200-45,600 | 44,640-47,040 | 44,640-47,040 |
| **Dev time** | - | 13 hours | 53 hours | 133 hours |
| **Hardware** | 1 GPU | 1 GPU | 1 GPU | 4 GPUs |

### What is NOT Achievable

❌ **100% retention (zero particle loss)**:
- Physical particles CAN exit mesh domain through boundaries
- Degenerate elements exist (numerical precision limits)
- Velocity extrapolation errors at timestep boundaries
- **Realistic maximum**: 95-98% retention

❌ **100K p/s on single GPU with JAX**:
- Theoretical JAX ceiling: 50-70K p/s (memory bandwidth limit)
- Your target: 100K p/s → **Need 2+ GPUs**

❌ **Fully-fused RK4 with RT Cores or OptiX**:
- RT Cores require CUDA/OptiX (incompatible with JAX)
- BVH rebuild cost: 2-5s per timestep (prohibitive)
- **Not compatible with time-dependent mesh preloading**

---

## Final Recommendations

### DO Implement ✅

1. **Phase 1 (13 hours)**: Fix initial assignment + multi-leaf + adaptive L1
   - **Highest ROI**: 186-286% throughput gain, +7-13% retention
   - **Low risk**: Minor code changes, no architecture changes

2. **Phase 2 (40 hours)**: Node-based boundary search **IF** retention <90% after Phase 1
   - **Medium ROI**: +3-8% retention gain
   - **Medium risk**: New data structure, JAX complexity

3. **Phase 3 (80 hours)**: Multi-GPU scaling **IF** 100K p/s is hard requirement
   - **High ROI**: 4-8× throughput with 4-8 GPUs
   - **High risk**: Distributed computing, data sharding

### DO NOT Implement ❌

1. **LBVH/BVH replacement**: You already have Morton octree (equivalent)
2. **RT Cores/OptiX**: Incompatible with JAX, rebuild cost prohibitive
3. **Uniform grid/spatial hashing**: Fails on 10× graded refinement
4. **CUDA rewrite**: 300+ hours, lose differentiability, no clear advantage

### Challenging Questions for You [2] Lines 591-609

Before proceeding, answer these:

**Q1**: Is **100K p/s a hard requirement** or nice-to-have?
- If **hard** → Proceed to Phase 3 (multi-GPU)
- If **nice** → Stop after Phase 1 (15-20K p/s sufficient)

**Q2**: Is **95-98% retention acceptable**?
- If **yes** → Realistic and achievable with Phases 1-2
- If **no** → Zero retention physically impossible, reconsider goals

**Q3**: Is **JAX non-negotiable**?
- If **yes** → Stay with JAX, accept 30-50% performance vs CUDA
- If **no** → CUDA rewrite could achieve 50-150K single GPU

**Q4**: What's the **actual bottleneck** - throughput or retention?
- If **throughput** → Phase 1 (multi-leaf fix) gives 3× speedup
- If **retention** → Phase 2 (node-based) gives +5-10% retention
- If **both** → Do Phase 1 first, then reassess

---

## Next Steps: Begin Phase 1 Implementation

Ready to proceed with **Phase 1 Critical Bug Fixes** (13 hours, highest ROI)?

**Immediate action**:
1. Fix 1.1: Initial assignment clipping (1 hour)
2. Fix 1.2: Multi-leaf optimization with lax.switch (4 hours)
3. Fix 1.3: Adaptive L1 hop count (8 hours)

**Expected results after Phase 1**:
- Throughput: 15-20K p/s (from 7K)
- Retention: 90-95% @ step 100 (from 83.74%)
- Total dev time: 13 hours

Shall I proceed with implementing Phase 1 Fix 1.1 (initial assignment clipping)?
