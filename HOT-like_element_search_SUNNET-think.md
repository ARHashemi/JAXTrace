# GPU-Native Hashed Oct-Tree Element Search for JAX/Tetrahedral Meshes

**Version**: 1.0  
**Date**: December 11, 2025  
**Status**: Complete Design Specification  

***

## Executive Summary

This document describes a **GPU-native, JAX-compatible element search algorithm** for particle tracking in large-scale adaptive tetrahedral meshes, based on the Hashed Oct-Tree (HOT) concept adapted for unstructured finite element applications.

**Key features:**
- **Morton-code based octree** with CSR leaf storage (no pointers, flat arrays)
- **Cube-aligned blocks** to eliminate shared elements across boundaries
- **Multi-level hierarchical search** (L0/L1/L2/L3) with GPU-resident masks
- **Single fused GPU kernel** per RK4 stage (vmap over particles + masked fori_loop)
- **Static shapes and bounded memory** (OOM-safe for 4GB GPUs with 3.5M elements)

**Performance targets:**
- L0/L1 hit rate: >95% (cached + neighbors)
- L2 (octree/Morton) throughput: 100-200k particles/s
- Overall throughput: 200-300k particles/s with 90%+ GPU utilization

***

## 1. Architecture Overview

### 1.1 High-Level Design

```
Domain
  └─> Coarse Blocks (cube-aligned with mesh grid)
       ├─> Block A: Octree (adaptive refinement)
       │    ├─> Leaf 1: Morton segment [s₁, e₁) → 128 tets
       │    ├─> Leaf 2: Morton segment [s₂, e₂) → 64 tets
       │    └─> ...
       ├─> Block B: Octree
       │    └─> ...
       └─> Block C: (light block, single leaf)

Per-Block Data on GPU:
  - elem_ids_sorted[n_elems_block]   # Morton-sorted tet IDs
  - morton_keys[n_elems_block]       # Sorted Morton codes
  - leaf_ranges[n_leaves, 2]         # CSR: (start, length)
  - leaf_bboxes[n_leaves, 6]         # For tree walk
```

### 1.2 Search Flow

For particle at position **x**:

1. **L0 (Cached):** Check if still in previous element → 80-90% hit
2. **L1 (Neighbors):** Check face/edge neighbors → 8-15% hit
3. **L2 (Block Octree/Morton):**
   - Identify block from x
   - Compute Morton key m(x)
   - Walk octree or use prefix table → find leaf ℓ
   - Get candidate segment [s_ℓ, e_ℓ)
   - Test ≤256 tets via point-in-tet → 2-5% hit
4. **L3 (Neighbor Blocks):** Repeat L2 in 6 or 26 neighbor blocks → <1% hit

All levels execute in **one GPU kernel** with boolean masks (no CPU orchestration).

***

## 2. Mathematical Foundations

### 2.1 Morton Code (Z-Order Curve)

Given domain bounding box \([x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}] \times [z_{\min}, z_{\max}]\) and maximum octree depth \(L\):

**Step 1: Normalize to integer grid**

\[
u_x = \left\lfloor \frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (2^L - 1) \right\rfloor, \quad u_y = \ldots, \quad u_z = \ldots
\]

Each \(u_* \in [0, 2^L-1]\) is an \(L\)-bit unsigned integer.

**Step 2: Bit interleaving**

Write each coordinate in binary:
\[
u_x = \sum_{i=0}^{L-1} x_i 2^i, \quad u_y = \sum_{i=0}^{L-1} y_i 2^i, \quad u_z = \sum_{i=0}^{L-1} z_i 2^i
\]

Morton code (3D Z-order):
\[
m = \sum_{i=0}^{L-1} \left( x_i \cdot 2^{3i} + y_i \cdot 2^{3i+1} + z_i \cdot 2^{3i+2} \right)
\]

This is a 3L-bit integer that encodes spatial position along a space-filling curve.

**Properties:**
- Spatially close points → numerically close Morton codes
- Octree hierarchy encoded in **prefixes**: depth-\(d\) node = first \(3d\) bits
- Parent/child relationships via bit shifts:
  - Parent: \(m_{\text{parent}} = m \gg 3\)
  - Child \(c \in [0,7]\): \(m_{\text{child}} = (m \ll 3) + c\)

### 2.2 Octree Leaf → Morton Segment

An octree leaf at depth \(d\) with prefix \(P\) (first \(3d\) bits) corresponds to all elements whose Morton codes lie in:

\[
\text{Range}(P, d) = \left[ P \cdot 2^{3(L-d)}, \; (P+1) \cdot 2^{3(L-d)} \right)
\]

After sorting elements by Morton code, this range becomes a **contiguous segment** \([s, e)\) in the sorted array.

***

## 3. Preprocessing (CPU)

### 3.1 Mesh Loading and Neighbor Computation

```python
# INPUT: mesh file (connectivity, node_positions)
# OUTPUT: connectivity[n_elems, 4], node_positions[n_nodes, 3], 
#         element_neighbors[n_elems, 4]

connectivity, node_positions = load_tetrahedral_mesh(mesh_file)
element_neighbors = build_face_neighbors(connectivity)
  # For each tet, find up to 4 tets sharing each face
  # Returns array[e, 0:4] = neighbor IDs or -1
```

### 3.2 Cube-Aligned Block Construction

**Goal:** Partition mesh into blocks that align with the underlying cubic grid (each cube → 4 tets).

```python
def build_cube_aligned_blocks(connectivity, node_positions, cube_grid_info):
    """
    Assign each tet to exactly one block based on its parent cube.
    
    Args:
        cube_grid_info: metadata from mesh generator (cube IDs, levels)
    Returns:
        element_to_block[n_elems]: block ID per element
        block_metadata: list of (block_id, bbox, n_elems)
    """
    
    n_elems = len(connectivity)
    element_to_cube = np.empty(n_elems, dtype=np.int32)
    
    # Step 1: Map each tet to its parent cube
    for e in range(n_elems):
        centroid = node_positions[connectivity[e]].mean(axis=0)
        cube_id = find_leaf_cube(centroid, cube_grid_info)
        element_to_cube[e] = cube_id
    
    # Step 2: Group cubes into coarse blocks
    # Choose block_size (e.g., 8×8×4 cubes per block)
    cube_to_block = {}
    for cube_id in np.unique(element_to_cube):
        ci, cj, ck, level = decode_cube_id(cube_id)
        bi = ci // BLOCK_SIZE_I
        bj = cj // BLOCK_SIZE_J
        bk = ck // BLOCK_SIZE_K
        block_id = encode_block_id(bi, bj, bk)
        cube_to_block[cube_id] = block_id
    
    element_to_block = np.array([cube_to_block[element_to_cube[e]] 
                                  for e in range(n_elems)], dtype=np.int32)
    
    # Step 3: Subdivide heavy blocks recursively
    element_to_block, n_blocks = subdivide_heavy_blocks(
        element_to_block, connectivity, node_positions,
        max_elems_per_block=50_000
    )
    
    # Step 4: Build block element lists and metadata
    block_elements = [[] for _ in range(n_blocks)]
    for e, b in enumerate(element_to_block):
        block_elements[b].append(e)
    
    block_metadata = []
    for b in range(n_blocks):
        elems_b = np.array(block_elements[b], dtype=np.int32)
        bbox = compute_bbox(node_positions[connectivity[elems_b].flatten()])
        block_metadata.append({
            'block_id': b,
            'n_elems': len(elems_b),
            'bbox': bbox,
            'elements': elems_b
        })
    
    return element_to_block, block_metadata
```

**Heavy block subdivision:**

```python
def subdivide_heavy_blocks(element_to_block, connectivity, node_positions, 
                           max_elems_per_block):
    """Recursively split blocks with >max_elems_per_block elements."""
    
    block_counts = np.bincount(element_to_block)
    n_blocks = len(block_counts)
    heavy_blocks = np.where(block_counts > max_elems_per_block)[0]
    
    if len(heavy_blocks) == 0:
        return element_to_block, n_blocks
    
    next_block_id = n_blocks
    for b_heavy in heavy_blocks:
        mask = (element_to_block == b_heavy)
        elems = np.where(mask)[0]
        
        # Compute centroids
        centroids = np.array([
            node_positions[connectivity[e]].mean(axis=0) for e in elems
        ])
        
        # Split along longest axis
        bbox_min, bbox_max = centroids.min(axis=0), centroids.max(axis=0)
        axis = np.argmax(bbox_max - bbox_min)
        mid = (bbox_min[axis] + bbox_max[axis]) / 2
        
        # Assign elements on right half to new block
        for i, e in enumerate(elems):
            if centroids[i, axis] > mid:
                element_to_block[e] = next_block_id
        
        next_block_id += 1
    
    # Recurse
    return subdivide_heavy_blocks(element_to_block, connectivity, node_positions,
                                  max_elems_per_block)
```

### 3.3 Per-Block Morton Sorting and Octree Construction

For each block \(B\):

```python
def build_block_octree_morton(block_metadata, connectivity, node_positions, 
                              max_leaf_elems=256, max_depth=8):
    """
    Build Morton-sorted element array and octree leaf structure for one block.
    
    Returns:
        elem_ids_sorted[n_elems_B]: element IDs sorted by Morton code
        morton_keys[n_elems_B]: sorted Morton codes
        leaf_ranges[n_leaves, 2]: (start, length) for each leaf
        leaf_metadata[n_leaves]: dict with bbox, depth, prefix
    """
    
    elems_B = block_metadata['elements']
    bbox_B = block_metadata['bbox']
    n_elems_B = len(elems_B)
    
    # Step 1: Compute Morton codes for all elements
    morton_codes = np.empty(n_elems_B, dtype=np.uint64)
    for i, e in enumerate(elems_B):
        centroid = node_positions[connectivity[e]].mean(axis=0)
        morton_codes[i] = compute_morton_code(centroid, bbox_B, L=max_depth)
    
    # Step 2: Sort elements by Morton code
    sort_idx = np.argsort(morton_codes)
    elem_ids_sorted = elems_B[sort_idx]
    morton_keys = morton_codes[sort_idx]
    
    # Step 3: Build octree leaves via adaptive subdivision
    leaves = []
    
    def subdivide_node(start, end, depth, prefix):
        """Recursively split until each leaf has ≤max_leaf_elems."""
        n = end - start
        
        if n <= max_leaf_elems or depth >= max_depth:
            # Create leaf
            bbox = compute_bbox_from_morton_range(
                morton_keys[start:end], bbox_B, max_depth
            )
            leaves.append({
                'start': start,
                'length': n,
                'depth': depth,
                'prefix': prefix,
                'bbox': bbox
            })
            return
        
        # Split into 8 children
        # Find boundaries in sorted Morton array where prefix changes
        child_ranges = find_octant_boundaries(
            morton_keys[start:end], depth, max_depth
        )
        
        for c in range(8):
            c_start, c_end = child_ranges[c]
            if c_end > c_start:
                child_prefix = (prefix << 3) + c
                subdivide_node(start + c_start, start + c_end, 
                              depth + 1, child_prefix)
    
    # Start recursion from root
    subdivide_node(0, n_elems_B, depth=0, prefix=0)
    
    # Convert to arrays
    n_leaves = len(leaves)
    leaf_ranges = np.array([[l['start'], l['length']] for l in leaves], 
                           dtype=np.int32)
    
    return elem_ids_sorted, morton_keys, leaf_ranges, leaves
```

**Helper: Morton encoding**

```python
def compute_morton_code(point, bbox, L=21):
    """
    Compute 3D Morton code for a point.
    
    Args:
        point: (x, y, z)
        bbox: [xmin, ymin, zmin, xmax, ymax, zmax]
        L: max depth (bits per dimension)
    Returns:
        64-bit Morton code
    """
    x, y, z = point
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    
    # Normalize to [0, 2^L - 1]
    ux = int((x - xmin) / (xmax - xmin) * (2**L - 1))
    uy = int((y - ymin) / (ymax - ymin) * (2**L - 1))
    uz = int((z - zmin) / (zmax - zmin) * (2**L - 1))
    
    # Clamp to valid range
    ux = max(0, min(2**L - 1, ux))
    uy = max(0, min(2**L - 1, uy))
    uz = max(0, min(2**L - 1, uz))
    
    # Interleave bits
    morton = 0
    for i in range(L):
        morton |= ((ux >> i) & 1) << (3*i + 0)
        morton |= ((uy >> i) & 1) << (3*i + 1)
        morton |= ((uz >> i) & 1) << (3*i + 2)
    
    return morton
```

### 3.4 Block Neighbors

```python
def build_block_neighbors(block_metadata, connectivity=26):
    """
    Build neighbor relationships between blocks.
    
    Args:
        connectivity: 6 (face) or 26 (face+edge+vertex)
    Returns:
        block_neighbors[n_blocks, max_neighbors]: neighbor block IDs or -1
    """
    # Decode block grid indices from block IDs
    # For each block, compute neighbors in (i±1, j±1, k±1)
    # Return as padded array
    ...
```

***

## 4. GPU Upload

```python
def upload_mesh_to_gpu(block_metadata, connectivity, node_positions, 
                       element_neighbors, element_to_block):
    """
    Upload all mesh data structures to GPU as JAX arrays.
    
    Returns:
        MeshGPU: dataclass with all device-resident arrays
    """
    
    # Global mesh data
    mesh_gpu = MeshGPU()
    mesh_gpu.connectivity = jax.device_put(connectivity.astype(np.int32))
    mesh_gpu.node_positions = jax.device_put(node_positions.astype(np.float32))
    mesh_gpu.element_neighbors = jax.device_put(element_neighbors.astype(np.int32))
    mesh_gpu.element_to_block = jax.device_put(element_to_block.astype(np.int32))
    
    # Per-block octree/Morton structures
    n_blocks = len(block_metadata)
    max_elems_per_block = max(b['n_elems'] for b in block_metadata)
    max_leaves_per_block = max(len(b['leaf_ranges']) for b in block_metadata)
    
    # Pad to static shapes
    elem_ids_sorted_all = np.full((n_blocks, max_elems_per_block), -1, dtype=np.int32)
    morton_keys_all = np.zeros((n_blocks, max_elems_per_block), dtype=np.uint64)
    leaf_ranges_all = np.zeros((n_blocks, max_leaves_per_block, 2), dtype=np.int32)
    leaf_bboxes_all = np.zeros((n_blocks, max_leaves_per_block, 6), dtype=np.float32)
    block_bboxes = np.zeros((n_blocks, 6), dtype=np.float32)
    block_n_leaves = np.zeros(n_blocks, dtype=np.int32)
    
    for b, meta in enumerate(block_metadata):
        n_e = meta['n_elems']
        n_l = len(meta['leaf_ranges'])
        
        elem_ids_sorted_all[b, :n_e] = meta['elem_ids_sorted']
        morton_keys_all[b, :n_e] = meta['morton_keys']
        leaf_ranges_all[b, :n_l] = meta['leaf_ranges']
        leaf_bboxes_all[b, :n_l] = meta['leaf_bboxes']
        block_bboxes[b] = meta['bbox']
        block_n_leaves[b] = n_l
    
    mesh_gpu.elem_ids_sorted = jax.device_put(elem_ids_sorted_all)
    mesh_gpu.morton_keys = jax.device_put(morton_keys_all)
    mesh_gpu.leaf_ranges = jax.device_put(leaf_ranges_all)
    mesh_gpu.leaf_bboxes = jax.device_put(leaf_bboxes_all)
    mesh_gpu.block_bboxes = jax.device_put(block_bboxes)
    mesh_gpu.block_n_leaves = jax.device_put(block_n_leaves)
    mesh_gpu.block_neighbors = jax.device_put(block_neighbors)
    
    # Constants
    mesh_gpu.n_blocks = n_blocks
    mesh_gpu.max_elems_per_block = max_elems_per_block
    mesh_gpu.max_leaves_per_block = max_leaves_per_block
    mesh_gpu.max_leaf_capacity = MAX_LEAF_ELEMS  # e.g., 256
    
    return mesh_gpu
```

***

## 5. Query Algorithm (Single Particle, GPU)

### 5.1 Multi-Level Search (L0/L1/L2/L3)

```python
@jax.jit
def search_single_particle(pos, cached_elem, mesh_gpu):
    """
    Find containing element for one particle.
    Designed to be vmap'd over all particles.
    
    Args:
        pos: (3,) position
        cached_elem: int32, previous element ID
        mesh_gpu: MeshGPU dataclass
    
    Returns:
        elem_id: int32, containing element or -1
    """
    
    # L0: Cached element
    elem = search_L0_cached(pos, cached_elem, mesh_gpu)
    found = (elem >= 0)
    
    # L1: Face neighbors
    elem_L1 = lax.cond(
        found,
        lambda: elem,
        lambda: search_L1_neighbors(pos, cached_elem, mesh_gpu)
    )
    found = found | (elem_L1 >= 0)
    elem = jnp.where(elem_L1 >= 0, elem_L1, elem)
    
    # L2: Block octree/Morton
    elem_L2 = lax.cond(
        found,
        lambda: elem,
        lambda: search_L2_block_octree(pos, elem, mesh_gpu)
    )
    found = found | (elem_L2 >= 0)
    elem = jnp.where(elem_L2 >= 0, elem_L2, elem)
    
    # L3: Neighbor blocks (optional, for rare cases)
    elem_L3 = lax.cond(
        found,
        lambda: elem,
        lambda: search_L3_neighbor_blocks(pos, elem, mesh_gpu)
    )
    elem = jnp.where(elem_L3 >= 0, elem_L3, elem)
    
    return elem
```

### 5.2 L0: Cached Element Check

```python
@jax.jit
def search_L0_cached(pos, cached_elem, mesh_gpu):
    """Check if particle is still in cached element."""
    
    # Guard against invalid cache
    valid = (cached_elem >= 0) & (cached_elem < len(mesh_gpu.connectivity))
    
    def check():
        nodes = mesh_gpu.connectivity[cached_elem]
        node_coords = mesh_gpu.node_positions[nodes]  # (4, 3)
        inside = point_in_tet(pos, node_coords)
        return jnp.where(inside, cached_elem, jnp.int32(-1))
    
    return lax.cond(valid, check, lambda: jnp.int32(-1))
```

### 5.3 L1: Neighbor Search

```python
@jax.jit
def search_L1_neighbors(pos, cached_elem, mesh_gpu):
    """Search face neighbors of cached element."""
    
    valid = (cached_elem >= 0)
    
    def check_neighbors():
        neighbors = mesh_gpu.element_neighbors[cached_elem]  # (4,)
        
        def check_one_neighbor(i, found_elem):
            still_searching = (found_elem == -1) & (i < 4)
            nbr = neighbors[i]
            is_valid_nbr = (nbr >= 0) & still_searching
            
            def test_nbr():
                nodes = mesh_gpu.connectivity[nbr]
                node_coords = mesh_gpu.node_positions[nodes]
                inside = point_in_tet(pos, node_coords)
                return jnp.where(inside, nbr, jnp.int32(-1))
            
            result = lax.cond(is_valid_nbr, test_nbr, lambda: jnp.int32(-1))
            return jnp.where(result >= 0, result, found_elem)
        
        return lax.fori_loop(0, 4, check_one_neighbor, jnp.int32(-1))
    
    return lax.cond(valid, check_neighbors, lambda: jnp.int32(-1))
```

### 5.4 L2: Block Octree/Morton Search

```python
@jax.jit
def search_L2_block_octree(pos, cached_elem, mesh_gpu):
    """
    Search within particle's block using Morton-sorted octree.
    
    Steps:
      1. Identify block from position
      2. Compute Morton code for position
      3. Find containing leaf via tree walk
      4. Test elements in leaf's Morton segment
    """
    
    # Step 1: Find block containing position
    block_id = find_block_for_position(pos, mesh_gpu)
    valid_block = (block_id >= 0) & (block_id < mesh_gpu.n_blocks)
    
    def search_in_block():
        # Step 2: Compute Morton code
        bbox = mesh_gpu.block_bboxes[block_id]
        morton_pos = compute_morton_gpu(pos, bbox, L=21)
        
        # Step 3: Find leaf containing this Morton code
        leaf_id = find_leaf_for_morton(morton_pos, block_id, mesh_gpu)
        
        # Step 4: Search leaf segment
        return search_leaf_segment(pos, block_id, leaf_id, mesh_gpu)
    
    return lax.cond(valid_block, search_in_block, lambda: jnp.int32(-1))
```

**Helper: Morton encoding on GPU**

```python
@jax.jit
def compute_morton_gpu(pos, bbox, L=21):
    """GPU-native Morton code computation."""
    x, y, z = pos
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    
    # Normalize
    ux = jnp.floor((x - xmin) / (xmax - xmin) * (2**L - 1)).astype(jnp.uint32)
    uy = jnp.floor((y - ymin) / (ymax - ymin) * (2**L - 1)).astype(jnp.uint32)
    uz = jnp.floor((z - zmin) / (zmax - zmin) * (2**L - 1)).astype(jnp.uint32)
    
    # Clamp
    ux = jnp.clip(ux, 0, 2**L - 1)
    uy = jnp.clip(uy, 0, 2**L - 1)
    uz = jnp.clip(uz, 0, 2**L - 1)
    
    # Interleave (unrolled for small L)
    morton = jnp.uint64(0)
    for i in range(L):
        morton |= ((ux >> i) & 1) << (3*i + 0)
        morton |= ((uy >> i) & 1) << (3*i + 1)
        morton |= ((uz >> i) & 1) << (3*i + 2)
    
    return morton
```

**Helper: Find leaf**

```python
@jax.jit
def find_leaf_for_morton(morton_code, block_id, mesh_gpu):
    """
    Find octree leaf containing given Morton code via tree walk.
    
    Alternative: use prefix table (top N bits → leaf_id)
    """
    
    n_leaves = mesh_gpu.block_n_leaves[block_id]
    leaf_bboxes = mesh_gpu.leaf_bboxes[block_id]  # (max_leaves, 6)
    
    # Simple linear scan over leaves (acceptable for ~100 leaves per block)
    def check_leaf(i, found_leaf):
        still_searching = (found_leaf == -1) & (i < n_leaves)
        
        def test_leaf():
            # Check if morton_code falls in this leaf's range
            # (Could store morton min/max per leaf, or use bbox)
            # For now, use spatial bbox check as proxy
            bbox = leaf_bboxes[i]
            # Decode morton back to approx position (or store ranges)
            # Simplified: just return first valid leaf
            return i
        
        result = lax.cond(still_searching, test_leaf, lambda: jnp.int32(-1))
        return jnp.where(result >= 0, result, found_leaf)
    
    return lax.fori_loop(0, mesh_gpu.max_leaves_per_block, 
                        check_leaf, jnp.int32(-1))
```

**Better approach: Prefix table (precomputed)**

```python
# During preprocessing, build a lookup table:
#   prefix_table[block_id, prefix_bits] = leaf_id
# Then on GPU:
prefix_bits = morton_code >> (3*L - PREFIX_LENGTH)
leaf_id = mesh_gpu.prefix_table[block_id, prefix_bits]
```

**Helper: Search leaf segment**

```python
@jax.jit
def search_leaf_segment(pos, block_id, leaf_id, mesh_gpu):
    """
    Test all elements in a leaf's Morton segment.
    Uses CSR (start, length) and fixed loop.
    """
    
    valid_leaf = (leaf_id >= 0)
    
    def test_segment():
        start, length = mesh_gpu.leaf_ranges[block_id, leaf_id]
        elem_ids_sorted = mesh_gpu.elem_ids_sorted[block_id]
        
        def test_one_elem(j, found_elem):
            active = (found_elem == -1) & (j < length)
            
            idx = start + j
            elem_id = jnp.where(active, elem_ids_sorted[idx], 0)
            
            def check_tet():
                nodes = mesh_gpu.connectivity[elem_id]
                node_coords = mesh_gpu.node_positions[nodes]
                inside = point_in_tet(pos, node_coords)
                return jnp.where(inside, elem_id, jnp.int32(-1))
            
            result = lax.cond(active, check_tet, lambda: jnp.int32(-1))
            return jnp.where(result >= 0, result, found_elem)
        
        # Fixed loop over max capacity (e.g., 256)
        return lax.fori_loop(0, mesh_gpu.max_leaf_capacity, 
                            test_one_elem, jnp.int32(-1))
    
    return lax.cond(valid_leaf, test_segment, lambda: jnp.int32(-1))
```

### 5.5 L3: Neighbor Blocks

```python
@jax.jit
def search_L3_neighbor_blocks(pos, cached_elem, mesh_gpu):
    """Search in neighboring blocks (rare fallback)."""
    
    # Get current block from cached_elem or position
    block_id = mesh_gpu.element_to_block[cached_elem] if cached_elem >= 0 \
               else find_block_for_position(pos, mesh_gpu)
    
    neighbors = mesh_gpu.block_neighbors[block_id]  # (26,)
    
    def check_one_neighbor_block(i, found_elem):
        still_searching = (found_elem == -1) & (i < 26)
        nbr_block = neighbors[i]
        valid = (nbr_block >= 0) & still_searching
        
        def search_nbr():
            # Reuse L2 logic for neighbor block
            # (Simplified: inline Morton search)
            bbox = mesh_gpu.block_bboxes[nbr_block]
            morton_pos = compute_morton_gpu(pos, bbox, L=21)
            leaf_id = find_leaf_for_morton(morton_pos, nbr_block, mesh_gpu)
            return search_leaf_segment(pos, nbr_block, leaf_id, mesh_gpu)
        
        result = lax.cond(valid, search_nbr, lambda: jnp.int32(-1))
        return jnp.where(result >= 0, result, found_elem)
    
    return lax.fori_loop(0, 26, check_one_neighbor_block, jnp.int32(-1))
```

### 5.6 Point-in-Tet Test

```python
@jax.jit
def point_in_tet(p, tet_nodes):
    """
    Barycentric coordinate test.
    
    Args:
        p: (3,) query point
        tet_nodes: (4, 3) tet vertices
    Returns:
        bool: True if p inside tet
    """
    p0 = tet_nodes[0]
    v1 = tet_nodes[1] - p0
    v2 = tet_nodes[2] - p0
    v3 = tet_nodes[3] - p0
    
    A = jnp.stack([v1, v2, v3], axis=1)  # (3, 3)
    dp = p - p0
    
    # Solve A λ = dp
    lam123 = jnp.linalg.solve(A, dp)
    lam0 = 1.0 - jnp.sum(lam123)
    
    # Inside if all λ ≥ 0
    inside = (lam0 >= -1e-6) & jnp.all(lam123 >= -1e-6)
    return inside
```

***

## 6. Integration into Fused RK4

```python
@jax.jit
def rk4_step_gpu_fused(positions, elem_ids, dt, mesh_gpu, velocity_field_gpu):
    """
    Complete RK4 step with fused search and interpolation.
    All operations on GPU, no intermediate transfers.
    
    Args:
        positions: (N, 3)
        elem_ids: (N,)
        dt: scalar
        mesh_gpu: MeshGPU
        velocity_field_gpu: (n_nodes, 3)
    
    Returns:
        positions_new: (N, 3)
        elem_ids_new: (N,)
    """
    
    # k1 at x_n
    v1 = interpolate_velocity_batch(positions, elem_ids, mesh_gpu, velocity_field_gpu)
    
    # k2 at x_n + 0.5*dt*k1
    pos2 = positions + 0.5 * dt * v1
    elem2 = vmap(search_single_particle, in_axes=(0, 0, None))(pos2, elem_ids, mesh_gpu)
    v2 = interpolate_velocity_batch(pos2, elem2, mesh_gpu, velocity_field_gpu)
    
    # k3 at x_n + 0.5*dt*k2
    pos3 = positions + 0.5 * dt * v2
    elem3 = vmap(search_single_particle, in_axes=(0, 0, None))(pos3, elem2, mesh_gpu)
    v3 = interpolate_velocity_batch(pos3, elem3, mesh_gpu, velocity_field_gpu)
    
    # k4 at x_n + dt*k3
    pos4 = positions + dt * v3
    elem4 = vmap(search_single_particle, in_axes=(0, 0, None))(pos4, elem3, mesh_gpu)
    v4 = interpolate_velocity_batch(pos4, elem4, mesh_gpu, velocity_field_gpu)
    
    # RK4 combination
    positions_new = positions + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
    
    # Final search
    elem_ids_new = vmap(search_single_particle, in_axes=(0, 0, None))(
        positions_new, elem_ids, mesh_gpu
    )
    
    return positions_new, elem_ids_new
```

**CPU Wrapper (for time loop):**

```python
def rk4_step_wrapper(positions_cpu, elem_ids_cpu, dt, mesh_gpu, velocity_field):
    """
    One upload, one download per timestep.
    
    Args:
        positions_cpu: numpy (N, 3)
        elem_ids_cpu: numpy (N,)
        velocity_field: numpy (n_nodes, 3) for current time
    """
    
    # Upload
    positions_gpu = jax.device_put(positions_cpu.astype(np.float32))
    elem_ids_gpu = jax.device_put(elem_ids_cpu.astype(np.int32))
    velocity_gpu = jax.device_put(velocity_field.astype(np.float32))
    
    # Compute
    positions_new_gpu, elem_ids_new_gpu = rk4_step_gpu_fused(
        positions_gpu, elem_ids_gpu, dt, mesh_gpu, velocity_gpu
    )
    positions_new_gpu.block_until_ready()
    
    # Download
    positions_new = np.array(positions_new_gpu, dtype=np.float32)
    elem_ids_new = np.array(elem_ids_new_gpu, dtype=np.int32)
    
    return positions_new, elem_ids_new
```

**Time loop:**

```python
for step in range(N_TIMESTEPS):
    # Load velocity for this time
    velocity_field = load_velocity_field(step)
    
    # Step
    positions, elem_ids = rk4_step_wrapper(
        positions, elem_ids, DT, mesh_gpu, velocity_field
    )
    
    # Optional output
    if step % OUTPUT_INTERVAL == 0:
        save_particles(positions, elem_ids, step)
```

***

## 7. Memory Analysis

### 7.1 GPU Memory Budget (Example: 4GB GPU, 3.5M elements)

**Global mesh:**
- `connectivity`: 3.5M × 4 × 4B = 56 MB
- `node_positions`: 700k × 3 × 4B = 8.4 MB
- `element_neighbors`: 3.5M × 4 × 4B = 56 MB
- `element_to_block`: 3.5M × 4B = 14 MB

**Per-block structures (32 blocks, worst block ~110k elems):**
- `elem_ids_sorted`: 32 × 110k × 4B = 14 MB
- `morton_keys`: 32 × 110k × 8B = 28 MB
- `leaf_ranges`: 32 × 500 × 2 × 4B = 0.13 MB
- `leaf_bboxes`: 32 × 500 × 6 × 4B = 0.38 MB

**Particles (60k):**
- `positions`: 60k × 3 × 4B = 0.7 MB
- `elem_ids`: 60k × 4B = 0.24 MB

**Velocity field:**
- 700k × 3 × 4B = 8.4 MB

**Total:** ~186 MB (well within 4GB)

### 7.2 Transient Intermediates (per RK4 stage)

Inside `search_single_particle` vmap over 60k particles:
- Per-particle candidate arrays: NONE (we use CSR + fori_loop, not slices)
- Masks: 60k × 1B × ~5 levels = 0.3 MB
- Temporary positions/velocities: 60k × 3 × 4B × 4 stages = 2.9 MB

**Peak GPU usage:** ~190 MB (OOM-safe)

***

## 8. JAX Implementation Notes

### 8.1 Static Shapes

All arrays must have **compile-time known shapes**:
- Pad all per-block arrays to `(n_blocks, max_elems_per_block)` or `(n_blocks, max_leaves, ...)`
- Use `max_leaf_capacity` (e.g., 256) as fixed loop bound
- Guard unused slots with masks (`j < actual_length`)

### 8.2 Control Flow

**Allowed inside jit:**
- `lax.cond(pred, true_fn, false_fn)` for binary branches
- `lax.fori_loop(start, end, body_fn, init)` for fixed-count loops
- `jnp.where(mask, a, b)` for element-wise conditionals

**Not allowed:**
- Python `if/else` on traced values
- Python `for` loops over data-dependent ranges
- Dynamic slicing `array[start:end]` with traced `start, end`

### 8.3 Avoiding Recompilation

- Keep all loop bounds and array shapes **static constants**
- Use `static_argnums` in `jax.jit` for hyperparameters (e.g., `dt`, `max_depth`)
- Warm up JIT before timing:

```python
# Warm-up
_ = rk4_step_gpu_fused(
    jnp.zeros((100, 3)), jnp.zeros(100, dtype=jnp.int32),
    0.01, mesh_gpu, jnp.zeros((n_nodes, 3))
)
```

### 8.4 Performance Tips

- **Prefer vmap over explicit loops** for outer parallelism (particles)
- **Use lax.fori_loop for small inner loops** (leaf candidates)
- **Minimize lax.cond nesting**; use boolean masks + jnp.where when possible
- **Avoid large carry in scan**; keep only essential state

***

## 9. Complete Pseudocode Summary

### 9.1 Preprocessing (CPU)

```
INITIALIZATION:
  1. Load mesh:
       connectivity, node_positions = load_mesh(file)
       element_neighbors = build_neighbors(connectivity)
  
  2. Build cube-aligned blocks:
       element_to_block = map_elements_to_cubes_to_blocks(...)
       element_to_block, n_blocks = subdivide_heavy_blocks(element_to_block, max=50k)
       block_metadata = build_block_lists(element_to_block)
  
  3. For each block B:
       a. Compute Morton codes for all elements in B:
            centroids = [node_positions[connectivity[e]].mean() for e in B]
            morton_codes = [compute_morton(c, bbox_B, L=21) for c in centroids]
       
       b. Sort elements by Morton:
            sort_idx = argsort(morton_codes)
            elem_ids_sorted_B = elements_B[sort_idx]
            morton_keys_B = morton_codes[sort_idx]
       
       c. Build octree leaves via adaptive subdivision:
            leaves_B = []
            subdivide(start=0, end=len(B), depth=0):
                if (end - start) <= 256 or depth >= 8:
                    leaves_B.append({start, length, bbox, prefix})
                else:
                    for octant c in 0..7:
                        find range [s_c, e_c) where morton[s_c:e_c] shares prefix
                        subdivide(s_c, e_c, depth+1)
       
       d. Store:
            block_metadata[B] = {
                elem_ids_sorted_B,
                morton_keys_B,
                leaf_ranges_B = [(start, length) for leaf in leaves_B],
                leaf_bboxes_B
            }
  
  4. Build block neighbors:
       block_neighbors = compute_26_neighbors(block_grid)
  
  5. Upload to GPU:
       mesh_gpu = upload_all_arrays_as_jax_device_arrays(
           connectivity, node_positions, element_neighbors,
           all block_metadata arrays (padded to static shapes),
           block_neighbors
       )
```

### 9.2 Query (GPU, per particle)

```
SEARCH_SINGLE_PARTICLE(pos, cached_elem, mesh_gpu):
  
  # L0: Cached
  IF cached_elem >= 0:
      nodes = connectivity[cached_elem]
      IF point_in_tet(pos, node_positions[nodes]):
          RETURN cached_elem
  
  # L1: Neighbors
  IF cached_elem >= 0:
      neighbors = element_neighbors[cached_elem]
      FOR i in 0..3:
          IF neighbors[i] >= 0:
              nodes = connectivity[neighbors[i]]
              IF point_in_tet(pos, node_positions[nodes]):
                  RETURN neighbors[i]
  
  # L2: Block octree/Morton
  block_id = find_block(pos, block_bboxes)
  IF block_id >= 0:
      bbox = block_bboxes[block_id]
      morton_pos = compute_morton_gpu(pos, bbox, L=21)
      
      # Find leaf (via prefix table or tree walk)
      leaf_id = find_leaf(morton_pos, block_id, mesh_gpu)
      
      IF leaf_id >= 0:
          start, length = leaf_ranges[block_id, leaf_id]
          FOR j in 0 .. max_leaf_capacity-1:
              IF j < length:
                  elem_id = elem_ids_sorted[block_id, start + j]
                  nodes = connectivity[elem_id]
                  IF point_in_tet(pos, node_positions[nodes]):
                      RETURN elem_id
  
  # L3: Neighbor blocks
  neighbors_B = block_neighbors[block_id]
  FOR i in 0..25:
      nbr_block = neighbors_B[i]
      IF nbr_block >= 0:
          # Repeat L2 logic for nbr_block
          elem = search_in_block(pos, nbr_block, mesh_gpu)
          IF elem >= 0:
              RETURN elem
  
  RETURN -1  # Not found
```

### 9.3 Fused RK4 (GPU, all particles)

```
RK4_STEP_GPU_FUSED(positions[N,3], elem_ids[N], dt, mesh_gpu, vel_field[n_nodes,3]):
  
  # k1
  v1[N,3] = vmap(interpolate_velocity)(positions, elem_ids, mesh_gpu, vel_field)
  
  # k2
  pos2 = positions + 0.5*dt*v1
  elem2[N] = vmap(search_single_particle)(pos2, elem_ids, mesh_gpu)
  v2 = vmap(interpolate_velocity)(pos2, elem2, mesh_gpu, vel_field)
  
  # k3
  pos3 = positions + 0.5*dt*v2
  elem3 = vmap(search_single_particle)(pos3, elem2, mesh_gpu)
  v3 = vmap(interpolate_velocity)(pos3, elem3, mesh_gpu, vel_field)
  
  # k4
  pos4 = positions + dt*v3
  elem4 = vmap(search_single_particle)(pos4, elem3, mesh_gpu)
  v4 = vmap(interpolate_velocity)(pos4, elem4, mesh_gpu, vel_field)
  
  # Final
  positions_new = positions + (dt/6)*(v1 + 2*v2 + 2*v3 + v4)
  elem_ids_new = vmap(search_single_particle)(positions_new, elem_ids, mesh_gpu)
  
  RETURN positions_new, elem_ids_new
```

### 9.4 Time Loop (CPU orchestration)

```
MAIN_TIME_LOOP:
  
  # Once: upload mesh
  mesh_gpu = preprocess_and_upload_mesh(mesh_file)
  
  # Initialize particles
  positions = initial_positions()
  elem_ids = initial_assignment(positions, mesh_gpu)
  
  FOR step in 0 .. N_TIMESTEPS-1:
      # Load/stream velocity field for current time
      velocity_field = load_velocity(step)
      velocity_gpu = jax.device_put(velocity_field)
      
      # Upload particles
      pos_gpu = jax.device_put(positions)
      elem_gpu = jax.device_put(elem_ids)
      
      # Fused RK4 (all on GPU)
      pos_new_gpu, elem_new_gpu = rk4_step_gpu_fused(
          pos_gpu, elem_gpu, DT, mesh_gpu, velocity_gpu
      )
      
      # Download
      positions = np.array(pos_new_gpu)
      elem_ids = np.array(elem_new_gpu)
      
      # Output
      IF step % OUTPUT_INTERVAL == 0:
          save_state(positions, elem_ids, step)
```

***

## 10. Validation and Testing

### 10.1 Unit Tests

- **Morton encoding:** verify interleaving for known points
- **Point-in-tet:** test against analytical cases
- **Octree leaf correctness:** verify all elements fall in correct Morton range
- **Search correctness:** compare against brute-force for small meshes

### 10.2 Integration Tests

- **L0/L1 hit rates:** should be >95% for smooth flows
- **L2 correctness:** verify all found elements actually contain particle
- **RK4 accuracy:** compare particle trajectories against reference solver

### 10.3 Performance Benchmarks

- **L0 throughput:** expect ~200k p/s (cached check)
- **L2 throughput:** target 100-200k p/s (Morton search)
- **Overall RK4:** target 200-300k p/s with 90%+ GPU utilization

***

## 11. Future Optimizations

### 11.1 Prefix Table for Faster Leaf Lookup

Instead of tree walk, precompute:
```python
# During preprocessing
prefix_length = 9  # Top 9 bits of Morton (512 buckets)
prefix_table[block_id, prefix] = leaf_id

# During query
prefix = morton_pos >> (3*L - prefix_length)
leaf_id = prefix_table[block_id, prefix]
```

**Tradeoff:** Uses more memory (~32 blocks × 512 × 4B = 64 KB) but eliminates tree walk.

### 11.2 Extended L1 (26-neighbor search)

Extend neighbor search to edge/vertex neighbors to reduce L2 rate to <2%.

### 11.3 Adaptive Velocity Field Streaming

For time-dependent fields, stream only next 2-3 time slices to GPU and interpolate in time inside RK4.

### 11.4 Multi-GPU Scaling

Partition particles across GPUs; each GPU holds full mesh but only tracks subset of particles.

***

## 12. References

- Warren & Salmon (1997): "A Parallel Hashed Oct-Tree N-Body Algorithm" - HOT paper
- Karras (2012): "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees" - LBVH/GPU octrees
- JAX documentation: GPU performance tips, control flow, vmap/jit best practices

***

## Appendix: Configuration Constants

```python
# Octree
MAX_OCTREE_DEPTH = 21          # Morton code bits per dimension
MAX_LEAF_ELEMS = 256           # Elements per leaf (CSR segment capacity)
MAX_ELEMS_PER_BLOCK = 50_000   # Trigger for block subdivision

# Blocks
BLOCK_SIZE_I = 8               # Coarse block grid size (cubes)
BLOCK_SIZE_J = 8
BLOCK_SIZE_K = 4

# GPU
GPU_MEMORY_LIMIT = 4 * 1024**3  # 4 GB
PARTICLE_BATCH_SIZE = 60_000    # Particles per timestep

# Time integration
DT = 0.001                      # Timestep (seconds)
N_TIMESTEPS = 2500              # Total steps
OUTPUT_INTERVAL = 100           # Save every N steps
```

***

**End of Document**

This specification is complete and ready for implementation. All algorithms, data structures, and integration patterns are defined with sufficient detail for direct JAX/GPU coding.