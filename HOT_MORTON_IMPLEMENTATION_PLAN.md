# HOT-like Morton Element Search: Implementation Plan

**Date**: 2025-12-11
**Status**: Ready for Implementation
**Objective**: JAX-compatible GPU-native element search using Morton codes and hierarchical octree

---

## Executive Summary

This plan synthesizes the three HOT-like design documents into a concrete, implementable architecture that addresses the **critical JAX memory limitation** discovered during Phase 2 L2 Morton testing.

**Key Insight from Phase 2 Failure**: Dynamic indexing into global mesh arrays (`connectivity[elem_id]`, `node_positions[node_ids]`) inside vmap causes JAX to materialize 4.88 TiB, leading to OOM.

**Solution**: Use **CSR (Compressed Sparse Row) with contiguous Morton segments**, NOT global indexing. All element data for a leaf is stored in **pre-fetched, padded local arrays** to avoid dynamic global mesh access.

---

## Critical Design Differences from Phase 2 Morton

### Phase 2 Morton (FAILED - OOM)
```python
# Inside vmapped search:
elem_id = block_element_ids[block_id, i]  # Get global element ID
node_ids = connectivity[elem_id]  # ❌ DYNAMIC GLOBAL ACCESS
tet_nodes = node_positions[node_ids]  # ❌ MATERIALIZATION EXPLOSION
```

### HOT Morton (THIS PLAN - OOM-SAFE)
```python
# Preprocessing: Build per-leaf local connectivity
# For each leaf ℓ, pre-compute and store:
leaf_connectivity[ℓ, 0:capacity, 0:4] = local node IDs (padded)
leaf_node_coords[ℓ, 0:max_nodes, 0:3] = unique node positions (padded)

# Inside vmapped search:
local_conn = leaf_connectivity[leaf_id]  # Static shape access
local_nodes = leaf_node_coords[leaf_id]  # Static shape access
# Test point-in-tet using LOCAL arrays only (no global indexing)
```

**Memory trade-off**: More preprocessing and storage, but **zero dynamic global mesh access** during search.

---

## Architecture Overview

### 1. Hierarchical Search Levels

```
L0: Cached Element (85-90% hit)
 ↓ miss
L1: Face Neighbors (8-12% hit, cumulative 95%+)
 ↓ miss
L2: Morton Octree within Block (~4% hit, cumulative 99%+)
 ↓ miss
L3: Neighbor Blocks (<1% hit, cumulative 99.9%+)
```

### 2. Data Structure Hierarchy

```
Domain
  └─> Blocks (cube-aligned, no shared elements)
       └─> Block A
            └─> Octree (adaptive depth)
                 ├─> Leaf 1: Morton segment [s₁, s₁+len₁)
                 │    ├─> Local connectivity (len₁, 4)
                 │    ├─> Local node coords (unique_nodes, 3)
                 │    └─> Elem → local node mapping
                 ├─> Leaf 2: Morton segment [s₂, s₂+len₂)
                 └─> ...
```

---

## Phase 1: Preprocessing (CPU)

### Step 1.1: Mesh Loading
```python
from jaxtrace.io.pvtu import load_mesh_from_pvtu

connectivity, node_positions = load_mesh_from_pvtu(mesh_file)
element_neighbors = build_element_neighbors_array(connectivity)
```

### Step 1.2: Cube-Aligned Blocks

**Goal**: Partition elements into blocks that align with mesh's cubic grid structure.

```python
def build_cube_aligned_blocks(connectivity, node_positions,
                              block_size=(8, 8, 4), max_elems=50000):
    """
    Assign each element to a block based on its parent cube.

    Returns:
        element_to_block: (n_elems,) int32
        block_metadata: list of dicts with bbox, elements, etc.
    """
    n_elems = len(connectivity)

    # Map elements to cubes (from mesh generator metadata or centroid)
    element_centroids = compute_element_centroids(connectivity, node_positions)
    element_to_cube = map_centroids_to_cubes(element_centroids)

    # Group cubes into coarse blocks
    cube_to_block = {}
    for cube_id in np.unique(element_to_cube):
        bi, bj, bk = cube_id // block_size
        block_key = (bi, bj, bk)
        if block_key not in cube_to_block:
            cube_to_block[block_key] = len(cube_to_block)

    element_to_block = np.array([
        cube_to_block[element_to_cube[e] // block_size]
        for e in range(n_elems)
    ], dtype=np.int32)

    # Subdivide heavy blocks
    element_to_block, n_blocks = subdivide_heavy_blocks(
        element_to_block, element_centroids, max_elems_per_block=max_elems
    )

    # Build block metadata
    block_metadata = []
    for b in range(n_blocks):
        mask = (element_to_block == b)
        elems = np.where(mask)[0]
        bbox = compute_bbox(node_positions[connectivity[elems].flatten()])
        block_metadata.append({
            'block_id': b,
            'elements': elems,
            'bbox': bbox,
            'n_elems': len(elems)
        })

    return element_to_block, block_metadata
```

### Step 1.3: Per-Block Morton Sorting

```python
def compute_morton_code_3d(point, bbox, L=21):
    """
    Compute 3D Morton code (Z-order curve).

    Args:
        point: (x, y, z)
        bbox: [xmin, ymin, zmin, xmax, ymax, zmax]
        L: bits per dimension (21 → 63-bit Morton code)

    Returns:
        uint64 Morton code
    """
    x, y, z = point
    xmin, ymin, zmin, xmax, ymax, zmax = bbox

    # Normalize to [0, 2^L - 1]
    ux = int((x - xmin) / (xmax - xmin) * (2**L - 1))
    uy = int((y - ymin) / (ymax - ymin) * (2**L - 1))
    uz = int((z - zmin) / (zmax - zmin) * (2**L - 1))

    # Clamp
    ux = np.clip(ux, 0, 2**L - 1)
    uy = np.clip(uy, 0, 2**L - 1)
    uz = np.clip(uz, 0, 2**L - 1)

    # Interleave bits
    morton = 0
    for i in range(L):
        morton |= ((ux >> i) & 1) << (3*i + 0)
        morton |= ((uy >> i) & 1) << (3*i + 1)
        morton |= ((uz >> i) & 1) << (3*i + 2)

    return morton


def sort_block_by_morton(block_metadata, connectivity, node_positions, L=21):
    """
    Sort elements in a block by Morton code.

    Returns:
        elem_ids_sorted: (n_elems_block,) sorted element IDs
        morton_keys: (n_elems_block,) sorted Morton codes
    """
    elems = block_metadata['elements']
    bbox = block_metadata['bbox']

    # Compute centroids and Morton codes
    centroids = np.array([
        node_positions[connectivity[e]].mean(axis=0) for e in elems
    ])

    morton_codes = np.array([
        compute_morton_code_3d(c, bbox, L) for c in centroids
    ], dtype=np.uint64)

    # Sort
    sort_idx = np.argsort(morton_codes)
    elem_ids_sorted = elems[sort_idx]
    morton_keys = morton_codes[sort_idx]

    return elem_ids_sorted, morton_keys
```

### Step 1.4: Octree Leaf Construction

```python
def build_octree_leaves_from_morton(morton_keys, elem_ids_sorted, bbox,
                                    max_leaf_capacity=256, max_depth=8, L=21):
    """
    Build octree leaves by recursively subdividing Morton-sorted array.

    Each leaf represents a contiguous segment of the Morton-sorted array.

    Returns:
        leaves: list of dicts with:
            - start: int (index into elem_ids_sorted)
            - length: int (number of elements)
            - depth: int (octree depth)
            - prefix: int (Morton prefix for this leaf)
            - bbox: (6,) leaf bounding box
    """
    leaves = []

    def subdivide(start, end, depth, prefix):
        """Recursive subdivision."""
        n = end - start

        # Leaf condition
        if n <= max_leaf_capacity or depth >= max_depth:
            leaf_bbox = compute_bbox_from_elements(
                elem_ids_sorted[start:end], connectivity, node_positions
            )
            leaves.append({
                'start': start,
                'length': n,
                'depth': depth,
                'prefix': prefix,
                'bbox': leaf_bbox
            })
            return

        # Split into 8 octants
        # Find boundaries where top (depth+1)*3 bits change
        child_ranges = find_octant_boundaries(
            morton_keys[start:end], depth, L
        )

        for c in range(8):
            c_start, c_end = child_ranges[c]
            if c_end > c_start:
                child_prefix = (prefix << 3) + c
                subdivide(start + c_start, start + c_end,
                         depth + 1, child_prefix)

    # Start from root
    subdivide(0, len(elem_ids_sorted), depth=0, prefix=0)

    return leaves


def find_octant_boundaries(morton_segment, depth, L):
    """
    Find boundaries where elements split into 8 child octants.

    Args:
        morton_segment: sorted Morton codes in current node
        depth: current depth
        L: max depth

    Returns:
        child_ranges: list of 8 (start, end) pairs
    """
    shift = 3 * (L - depth - 1)  # Shift to get next 3 bits
    octants = (morton_segment >> shift) & 0x7  # Extract octant bits

    child_ranges = []
    for c in range(8):
        mask = (octants == c)
        indices = np.where(mask)[0]
        if len(indices) > 0:
            child_ranges.append((indices[0], indices[-1] + 1))
        else:
            child_ranges.append((0, 0))  # Empty octant

    return child_ranges
```

### Step 1.5: **CRITICAL** - Build Local Leaf Connectivity

**This is the key difference from Phase 2 that avoids OOM!**

```python
def build_leaf_local_connectivity(leaf, elem_ids_sorted, connectivity,
                                  node_positions, max_capacity=256):
    """
    Build LOCAL connectivity for a leaf to avoid global mesh indexing.

    For each leaf, create:
    1. Local connectivity: maps local elem idx → local node indices
    2. Local node coords: unique node positions for this leaf
    3. Global-to-local node mapping

    This allows point-in-tet to work ENTIRELY with local arrays,
    avoiding dynamic global mesh access that causes JAX OOM.

    Args:
        leaf: dict with 'start', 'length'
        elem_ids_sorted: global sorted element IDs
        connectivity: global (n_elems, 4) connectivity
        node_positions: global (n_nodes, 3) positions
        max_capacity: padding size

    Returns:
        leaf_local_connectivity: (max_capacity, 4) int32, padded with -1
        leaf_node_coords: (max_local_nodes, 3) float32
        n_local_nodes: int
    """
    start = leaf['start']
    length = leaf['length']

    # Get global element IDs in this leaf
    global_elem_ids = elem_ids_sorted[start:start+length]

    # Get global connectivity for these elements
    global_conn = connectivity[global_elem_ids]  # (length, 4)

    # Find unique nodes
    unique_global_nodes = np.unique(global_conn.flatten())
    n_local_nodes = len(unique_global_nodes)

    # Build global → local node mapping
    global_to_local = {g: l for l, g in enumerate(unique_global_nodes)}

    # Build local connectivity
    local_connectivity = np.full((max_capacity, 4), -1, dtype=np.int32)
    for i in range(length):
        global_nodes = global_conn[i]
        local_nodes = [global_to_local[gn] for gn in global_nodes]
        local_connectivity[i] = local_nodes

    # Extract local node coordinates
    max_local_nodes = min(n_local_nodes, max_capacity * 4)  # Upper bound
    leaf_node_coords = np.zeros((max_local_nodes, 3), dtype=np.float32)
    leaf_node_coords[:n_local_nodes] = node_positions[unique_global_nodes]

    return local_connectivity, leaf_node_coords, n_local_nodes
```

### Step 1.6: Pack All Leaves into Arrays

```python
def pack_block_octree_to_arrays(leaves, elem_ids_sorted, connectivity,
                                node_positions, max_leaves=500, max_capacity=256):
    """
    Pack all leaves into static-shape arrays for GPU upload.

    Returns:
        leaf_ranges: (max_leaves, 2) int32 - (start, length) in elem_ids_sorted
        leaf_local_connectivity: (max_leaves, max_capacity, 4) int32
        leaf_node_coords: (max_leaves, max_local_nodes, 3) float32
        leaf_n_local_nodes: (max_leaves,) int32
        leaf_bboxes: (max_leaves, 6) float32
        n_leaves: int (actual count)
    """
    n_leaves = len(leaves)
    max_local_nodes = max_capacity * 4  # Worst case: all elements have unique nodes

    # Initialize arrays
    leaf_ranges = np.zeros((max_leaves, 2), dtype=np.int32)
    leaf_local_connectivity = np.full((max_leaves, max_capacity, 4), -1, dtype=np.int32)
    leaf_node_coords = np.zeros((max_leaves, max_local_nodes, 3), dtype=np.float32)
    leaf_n_local_nodes = np.zeros(max_leaves, dtype=np.int32)
    leaf_bboxes = np.zeros((max_leaves, 6), dtype=np.float32)

    for i, leaf in enumerate(leaves):
        # Store range
        leaf_ranges[i] = [leaf['start'], leaf['length']]

        # Build local connectivity
        local_conn, local_coords, n_local = build_leaf_local_connectivity(
            leaf, elem_ids_sorted, connectivity, node_positions, max_capacity
        )

        leaf_local_connectivity[i] = local_conn
        leaf_node_coords[i, :n_local] = local_coords
        leaf_n_local_nodes[i] = n_local
        leaf_bboxes[i] = leaf['bbox']

    return (leaf_ranges, leaf_local_connectivity, leaf_node_coords,
            leaf_n_local_nodes, leaf_bboxes, n_leaves)
```

---

## Phase 2: GPU Upload

```python
@dataclass
class MeshGPUHOT:
    """GPU-resident mesh with HOT Morton structures."""

    # Global mesh (for L0/L1 only, NOT accessed in L2/L3)
    connectivity: jax.Array  # (n_elems, 4) - used ONLY in L0/L1
    node_positions: jax.Array  # (n_nodes, 3) - used ONLY in L0/L1
    element_neighbors: jax.Array  # (n_elems, 4)
    element_to_block: jax.Array  # (n_elems,)

    # Block metadata
    n_blocks: int
    block_bboxes: jax.Array  # (n_blocks, 6)
    block_n_leaves: jax.Array  # (n_blocks,)
    block_neighbors_26: jax.Array  # (n_blocks, 26) for L3

    # Per-block octree/Morton (CSR-style)
    elem_ids_sorted: jax.Array  # (n_blocks, max_elems_per_block) - global IDs
    morton_keys: jax.Array  # (n_blocks, max_elems_per_block)

    # Per-leaf LOCAL structures (KEY: avoids global indexing)
    leaf_ranges: jax.Array  # (n_blocks, max_leaves, 2) - (start, length)
    leaf_local_connectivity: jax.Array  # (n_blocks, max_leaves, max_capacity, 4)
    leaf_node_coords: jax.Array  # (n_blocks, max_leaves, max_local_nodes, 3)
    leaf_n_local_nodes: jax.Array  # (n_blocks, max_leaves)
    leaf_bboxes: jax.Array  # (n_blocks, max_leaves, 6)

    # Constants
    max_elems_per_block: int
    max_leaves_per_block: int
    max_leaf_capacity: int
    max_local_nodes_per_leaf: int
    morton_depth: int  # L (e.g., 21)


def upload_mesh_hot_to_gpu(connectivity, node_positions, element_neighbors,
                            element_to_block, block_metadata_list):
    """Upload all HOT structures to GPU."""

    # Global mesh (L0/L1 only)
    connectivity_gpu = jax.device_put(connectivity.astype(np.int32))
    node_positions_gpu = jax.device_put(node_positions.astype(np.float32))
    element_neighbors_gpu = jax.device_put(element_neighbors.astype(np.int32))
    element_to_block_gpu = jax.device_put(element_to_block.astype(np.int32))

    # Determine padding sizes
    n_blocks = len(block_metadata_list)
    max_elems = max(b['n_elems'] for b in block_metadata_list)
    max_leaves = max(len(b['leaves']) for b in block_metadata_list)
    max_capacity = 256  # Fixed
    max_local_nodes = max_capacity * 4

    # Initialize padded arrays
    block_bboxes = np.zeros((n_blocks, 6), dtype=np.float32)
    block_n_leaves = np.zeros(n_blocks, dtype=np.int32)

    elem_ids_sorted_all = np.full((n_blocks, max_elems), -1, dtype=np.int32)
    morton_keys_all = np.zeros((n_blocks, max_elems), dtype=np.uint64)

    leaf_ranges_all = np.zeros((n_blocks, max_leaves, 2), dtype=np.int32)
    leaf_local_conn_all = np.full((n_blocks, max_leaves, max_capacity, 4), -1, dtype=np.int32)
    leaf_node_coords_all = np.zeros((n_blocks, max_leaves, max_local_nodes, 3), dtype=np.float32)
    leaf_n_local_nodes_all = np.zeros((n_blocks, max_leaves), dtype=np.int32)
    leaf_bboxes_all = np.zeros((n_blocks, max_leaves, 6), dtype=np.float32)

    # Pack each block
    for b, meta in enumerate(block_metadata_list):
        n_e = meta['n_elems']
        n_l = len(meta['leaves'])

        block_bboxes[b] = meta['bbox']
        block_n_leaves[b] = n_l

        elem_ids_sorted_all[b, :n_e] = meta['elem_ids_sorted']
        morton_keys_all[b, :n_e] = meta['morton_keys']

        leaf_ranges_all[b, :n_l] = meta['leaf_ranges']
        leaf_local_conn_all[b, :n_l] = meta['leaf_local_connectivity']
        leaf_node_coords_all[b, :n_l] = meta['leaf_node_coords']
        leaf_n_local_nodes_all[b, :n_l] = meta['leaf_n_local_nodes']
        leaf_bboxes_all[b, :n_l] = meta['leaf_bboxes']

    # Upload to GPU
    mesh_gpu = MeshGPUHOT(
        connectivity=connectivity_gpu,
        node_positions=node_positions_gpu,
        element_neighbors=element_neighbors_gpu,
        element_to_block=element_to_block_gpu,
        n_blocks=n_blocks,
        block_bboxes=jax.device_put(block_bboxes),
        block_n_leaves=jax.device_put(block_n_leaves),
        block_neighbors_26=jax.device_put(block_neighbors_26),
        elem_ids_sorted=jax.device_put(elem_ids_sorted_all),
        morton_keys=jax.device_put(morton_keys_all),
        leaf_ranges=jax.device_put(leaf_ranges_all),
        leaf_local_connectivity=jax.device_put(leaf_local_conn_all),
        leaf_node_coords=jax.device_put(leaf_node_coords_all),
        leaf_n_local_nodes=jax.device_put(leaf_n_local_nodes_all),
        leaf_bboxes=jax.device_put(leaf_bboxes_all),
        max_elems_per_block=max_elems,
        max_leaves_per_block=max_leaves,
        max_leaf_capacity=max_capacity,
        max_local_nodes_per_leaf=max_local_nodes,
        morton_depth=21
    )

    return mesh_gpu
```

---

## Phase 3: GPU Query (JAX-Compatible)

### L2 Search with LOCAL Connectivity

```python
@jax.jit
def search_leaf_segment_local(pos, block_id, leaf_id, mesh_gpu):
    """
    Search elements in a leaf using LOCAL connectivity.

    KEY: No global mesh indexing - everything is local!
    """
    valid_leaf = (leaf_id >= 0) & (leaf_id < mesh_gpu.block_n_leaves[block_id])

    def test_segment():
        # Get LOCAL structures for this leaf
        local_connectivity = mesh_gpu.leaf_local_connectivity[block_id, leaf_id]  # (capacity, 4)
        local_node_coords = mesh_gpu.leaf_node_coords[block_id, leaf_id]  # (max_local_nodes, 3)
        n_local_nodes = mesh_gpu.leaf_n_local_nodes[block_id, leaf_id]

        start, length = mesh_gpu.leaf_ranges[block_id, leaf_id]

        def test_one_elem(j, found_elem):
            active = (found_elem == -1) & (j < length)

            # Get local connectivity (NO global mesh access!)
            local_node_ids = jnp.where(active, local_connectivity[j], 0)

            # Get node coords using LOCAL indexing
            tet_nodes = local_node_coords[local_node_ids]  # (4, 3)

            # Point-in-tet using local data
            inside = jnp.where(active, point_in_tet_jax(pos, tet_nodes), False)

            # Map back to GLOBAL element ID
            global_idx = start + j
            global_elem_id = mesh_gpu.elem_ids_sorted[block_id, global_idx]

            return jnp.where(inside & active, global_elem_id, found_elem)

        # Bounded loop (static shape)
        return lax.fori_loop(0, mesh_gpu.max_leaf_capacity, test_one_elem, jnp.int32(-1))

    return lax.cond(valid_leaf, test_segment, lambda: jnp.int32(-1))
```

**Why this works**:
1. `local_connectivity` and `local_node_coords` are **pre-fetched fixed-size arrays**
2. Indexing `local_connectivity[j]` and `local_node_coords[local_node_ids]` uses **static shapes only**
3. **NO** access to global `connectivity` or `node_positions` inside the loop
4. JAX sees only bounded loops over fixed-size local arrays → **no materialization explosion**

---

## Memory Analysis

### Per-Block Leaf Storage (worst case: 50k elements, 200 leaves, capacity=256)

**Global mesh** (shared, L0/L1 only):
- connectivity: 3.5M × 4 × 4B = 56 MB
- node_positions: 900k × 3 × 4B = 10.8 MB
- element_neighbors: 3.5M × 4 × 4B = 56 MB

**Per-block Morton** (32 blocks):
- elem_ids_sorted: 32 × 50k × 4B = 6.4 MB
- morton_keys: 32 × 50k × 8B = 12.8 MB

**Per-leaf local connectivity** (32 blocks × 200 leaves):
- leaf_local_connectivity: 32 × 200 × 256 × 4 × 4B = 66 MB
- leaf_node_coords: 32 × 200 × 1024 × 3 × 4B = 787 MB (⚠️ LARGEST)
- leaf_ranges: 32 × 200 × 2 × 4B = 0.05 MB
- leaf_n_local_nodes: 32 × 200 × 4B = 0.03 MB

**Total**: ~996 MB (acceptable for 4 GB GPU)

**Optimization**: Most leaves have FAR fewer than 1024 unique nodes. Typical: ~100-200 unique nodes per leaf → ~77-154 MB instead of 787 MB.

---

## Implementation Phases

### Phase 1: CPU Preprocessing (1-2 days)
- [ ] Implement cube-aligned block construction
- [ ] Implement Morton encoding and sorting
- [ ] Implement octree leaf building
- [ ] **Implement local connectivity extraction** (critical!)
- [ ] Test on small mesh (1k elements)

### Phase 2: GPU Upload (0.5 days)
- [ ] Implement `MeshGPUHOT` dataclass
- [ ] Implement upload function
- [ ] Verify memory usage

### Phase 3: L2 Morton Search (1 day)
- [ ] Implement Morton encoding on GPU (JAX)
- [ ] Implement leaf lookup (prefix table or linear scan)
- [ ] Implement `search_leaf_segment_local` with local connectivity
- [ ] Test single-particle search

### Phase 4: Multi-Level Search (0.5 days)
- [ ] Integrate L0 (cached)
- [ ] Integrate L1 (neighbors) - reuse existing
- [ ] Integrate L2 (HOT Morton)
- [ ] Integrate L3 (neighbor blocks)

### Phase 5: Fused RK4 Integration (0.5 days)
- [ ] Create `search_single_particle_hot` wrapper
- [ ] Vmap for batched search
- [ ] Integrate into existing fused RK4

### Phase 6: Testing & Validation (1 day)
- [ ] Unit tests: Morton encoding, point-in-tet, leaf search
- [ ] Integration test: L0+L1+L2+L3 hierarchy
- [ ] Production test: 105k particles, 2,500 steps
- [ ] Verify retention >95%, throughput >100k p/s

**Total**: ~5-6 days implementation

---

## Success Criteria

1. ✅ **No JAX OOM errors** with 3.5M element mesh
2. ✅ **>95% retention** at 2,500 timesteps
3. ✅ **>100k particles/second** throughput
4. ✅ **<1 GB GPU memory** for HOT structures
5. ✅ **L0+L1 hit rate >95%**, L2 hit rate >4%

---

## Advantages Over Phase 2 Morton

| Aspect | Phase 2 Morton (Failed) | HOT Morton (This Plan) |
|--------|------------------------|------------------------|
| Global mesh access | ✗ Dynamic `connectivity[elem_id]` | ✅ Pre-fetched local arrays |
| Memory during search | ✗ 4.88 TiB materialization | ✅ <1 GB fixed-size locals |
| JAX compatibility | ✗ Nested vmap explosion | ✅ Bounded loops, static shapes |
| Preprocessing | ✅ Simple (Morton sort only) | ✗ Complex (local connectivity) |
| Memory overhead | ✅ 8 MB | ⚠️ ~100-800 MB (depends on leaf size) |

**Trade-off**: More complex preprocessing and higher memory, but **guaranteed OOM-safe**.

---

## Next Steps

1. ✅ Review and approve this plan
2. Begin Phase 1: CPU preprocessing implementation
3. Test on small mesh to validate approach
4. Proceed to GPU implementation once validated

This plan directly addresses the JAX memory limitation while maintaining the theoretical benefits of HOT-style Morton search.
