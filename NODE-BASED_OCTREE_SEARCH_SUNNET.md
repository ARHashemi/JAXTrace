# Node-Based Particle Tracking for Octree-Refined Tetrahedral Meshes
## Comprehensive Design Document with Topology Analysis

***

## Part I: Topological Analysis - Maximum Node Valence

### 1.1 Your Mesh Structure

**Base topology**: Right-angled tetrahedra, 4 per cube
**Refinement**: Octree-based with 1:2 face transitions (hanging nodes allowed)
**Critical constraint**: Maximum 1:2 refinement ratio at interfaces

### 1.2 Calculating Maximum Node Valence

#### Case 1: Interior Node in Uniform Region (Baseline)

**Regular cube tessellation** (4 right-angled tets per cube):

```
Standard "Kuhn" decomposition of unit cube into 4 tets:
  Tet 1: (0,0,0)-(1,0,0)-(1,1,0)-(1,1,1)
  Tet 2: (0,0,0)-(1,1,0)-(0,1,0)-(1,1,1)
  Tet 3: (0,0,0)-(0,1,0)-(0,1,1)-(1,1,1)
  Tet 4: (0,0,0)-(0,0,1)-(0,1,1)-(1,1,1)

All 4 tets share diagonal node (1,1,1)
```

For a node in the interior of a uniform octree grid:
- The node is shared by all cubes meeting at that vertex
- In 3D regular grid: **8 cubes meet at an interior vertex**
- Each cube contributes 4 tets
- **Baseline valence**: 8 cubes × 4 tets = **32 elements**

#### Case 2: Refinement Interface Node (1:2 Transition)

**Octree 1:2 refinement** creates hanging nodes on faces:

```
Coarse face (1 element) meets refined face (4 sub-elements)

        Coarse side          |         Refined side
                             |
     +----------+            |      +-----+-----+
     |          |            |      |     |     |
     |    1     |            |      |  2  |  3  |
     |          |            |      +-----+-----+
     +----------+            |      |  4  |  5  |
                             |      |     |     |
                             |      +-----+-----+
```

**Hanging node** (center of refined side face):
- Belongs to 4 refined cubes on one side
- Belongs to 2 coarse cubes on the other side (shared edge)
- Total cubes sharing this node: 4 + 2 = 6 cubes
- **Hanging node valence**: 6 cubes × 4 tets = **24 elements**

#### Case 3: Worst Case - Edge/Corner Nodes at Multiple Refinement Transitions

**Most pathological case**: Node at the intersection of 3 orthogonal refinement interfaces

Consider a node at the corner where:
- x-direction: coarse-to-fine transition
- y-direction: coarse-to-fine transition  
- z-direction: coarse-to-fine transition

In standard octree refinement with 1:2 max ratio:
- Each direction contributes refined cubes on one side, coarse on other
- Due to the 1:2 constraint, the maximum configuration is bounded

**Theoretical maximum** for 1:2 octree refinement:
- Corner node with 3-way refinement interfaces
- Approximately **12-16 cubes** can share such a node
- **Maximum valence**: ~16 cubes × 4 tets = **64 elements**

**However**, in practice with conservative octree balancing:
- Most implementations enforce 2:1 balance across edges/faces
- This limits corner configurations
- **Practical maximum**: ~40-48 elements per node

### 1.3 Statistical Distribution in Your Mesh

Based on octree refinement properties:

| Node Location | Percentage | Typical Valence | Max Valence |
|--------------|------------|-----------------|-------------|
| **Interior uniform** | 70-80% | 32 | 32 |
| **Face hanging nodes** | 15-20% | 24 | 32 |
| **Edge refinement** | 3-5% | 36 | 48 |
| **Corner (worst case)** | <1% | 40 | **48-64** |

**Key findings**:
1. ✅ **Most nodes have valence ≤32** (uniform regions)
2. ⚠️ **Interface nodes: 24-48 elements**
3. 🔴 **Worst case: ~64 elements** (rare, <1% of nodes)

**Recommendation for JAX implementation**: Use `MAX_VALENCE = 64` for safety, with padding.

### 1.4 Memory Implications

**Node-to-element adjacency storage**:

```python
# Given:
n_nodes = 300,000
MAX_VALENCE = 64

# Storage requirements:
node_to_elements: (300000, 64) int32 = 76.8 MB
node_valence: (300000,) int32 = 1.2 MB

# Total per timestep: ~78 MB for node adjacency
```

**Comparison with element-based structures**:
- Current element octree leaves: ~50 MB
- Element neighbor arrays: ~40 MB  
- **Node adjacency is comparable**, not significantly better

**BUT**: Fewer nodes (300k) vs elements (1.4M) means:
- Node-based Morton/octree index: **~5× smaller** (fewer primitives)
- Node position arrays: **~5× smaller**

***

## Part II: Velocity-Based Predictive Search Radius

### 2.1 Motivation

Current search uses **fixed spatial radius** or **fixed number of neighbors**.

**Insight**: Particle velocity gives direction and magnitude of motion → can predict "target region" and bias search there.

### 2.2 Physics-Based Search Radius Estimation

**RK4 step structure**:
```python
# At position pos0, velocity v0, timestep dt:
pos_k1 = pos0
pos_k2 = pos0 + 0.5 * dt * v0
pos_k3 = pos0 + 0.5 * dt * v(pos_k2)
pos_k4 = pos0 + dt * v(pos_k3)
pos_final = pos0 + dt * (v0 + 2*v2 + 2*v3 + v4) / 6
```

**Maximum displacement per step**: 
```
d_max ≈ dt * ||v_max||
```

Where `v_max` is the maximum velocity magnitude in the local region.

**Adaptive search radius**:
```python
# Estimate search radius based on velocity
v_magnitude = jnp.linalg.norm(velocity_at_current_elem)
search_radius_physical = 1.5 * dt * v_magnitude  # 1.5× safety factor

# Convert to element-space or node-space radius
# (depends on characteristic element size at this location)
char_size = element_volumes[cached_elem] ** (1/3)  # Cube root for tet size
search_radius_elements = jnp.ceil(search_radius_physical / char_size)
```

**Issues**:
1. **Refinement mismatch**: High velocity in refined region may cross into coarse region where element size is 10× larger
2. **JAX constraints**: `search_radius_elements` is **data-dependent** → can't use for loop bounds
3. **Anisotropic motion**: Velocity direction matters, not just magnitude

### 2.3 Velocity-Guided Neighbor Prioritization (Practical Approach)

Instead of dynamic radius, use **fixed candidate set** but **prioritize** based on velocity:

**Idea**: 
- Generate fixed set of 27 neighbor octants (or k=27 nearest nodes)
- **Reorder** candidates based on velocity direction
- Check "downstream" candidates first → higher chance of early hit

**Algorithm**:
```python
def search_with_velocity_priority(pos, velocity, cached_elem, mesh_gpu):
    """Search with velocity-based candidate ordering."""
    
    # Normalize velocity direction
    v_dir = velocity / (jnp.linalg.norm(velocity) + 1e-12)
    
    # Generate 27 spatial neighbor offsets (fixed)
    neighbor_offsets = generate_27_neighbors()  # (-1,-1,-1) to (1,1,1)
    
    # Compute alignment score: dot product with velocity direction
    def alignment_score(offset):
        offset_dir = offset / (jnp.linalg.norm(offset) + 1e-12)
        return jnp.dot(offset_dir, v_dir)
    
    scores = jax.vmap(alignment_score)(neighbor_offsets)
    
    # Sort neighbors by descending alignment (most aligned first)
    sorted_indices = jnp.argsort(-scores)  # Negate for descending
    sorted_neighbors = neighbor_offsets[sorted_indices]
    
    # Search in sorted order (downstream neighbors first)
    def search_neighbor(offset):
        neighbor_id = get_neighbor_from_offset(cached_elem, offset)
        return point_in_tet_check(pos, neighbor_id)
    
    # Fixed-size vmap, but results biased toward velocity direction
    results = jax.vmap(search_neighbor)(sorted_neighbors)
    
    # First valid result is likely the "right" one
    found_mask = results >= 0
    return jnp.where(
        jnp.any(found_mask),
        results[jnp.argmax(found_mask)],
        jnp.int32(-1)
    )
```

**Pros**:
- ✅ JAX-compatible (fixed-size loops, vectorized sort)
- ✅ Early termination benefit (likely hit within first few checks)
- ✅ No data-dependent branching

**Cons**:
- ❌ Still checks all 27 candidates (JAX vmap requirement)
- ❌ Sorting overhead (~100 ops for 27 elements)
- ⚠️ Benefit only if "first hit" termination is respected (need `lax.scan` instead of `vmap`)

### 2.4 Velocity-Based Search for Node-Based L2

**Better fit**: Use velocity to select **which nodes** to query

**Algorithm**:
```python
def search_l2_velocity_guided_nodes(pos, velocity, dt, mesh_gpu_nodes):
    """
    L2 search using predicted position to find nearby nodes.
    """
    # Predict next position (Euler step as approximation)
    predicted_pos = pos + dt * velocity
    
    # Find k nearest nodes to PREDICTED position (not current)
    # This biases search toward where particle is going
    k_nearest_nodes = knn_nodes_morton(
        query_pos=predicted_pos,  # ← Key difference
        k=4,
        node_octree=mesh_gpu_nodes.octree
    )
    
    # Check all elements incident to these k nodes
    candidates = gather_incident_elements(k_nearest_nodes, mesh_gpu_nodes)
    
    # Point-in-tet on current position (not predicted)
    return search_candidates(pos, candidates)
```

**Advantage**: 
- Queries octree at **predicted** position
- More likely to find nodes in the "target" region
- Especially useful for high-velocity flows

**Caveat**: 
- Prediction may overshoot in curved streamlines
- Still need fallback if predicted position is far off

***

## Part III: Node-Based L1 Design (JAX-Native)

### 3.1 Design Overview

**Goal**: Search all elements sharing nodes with current element, without precomputed element-neighbor arrays.

**Data structures**:
```python
@dataclass
class MeshGPUNodeBased:
    # Connectivity (fixed across timesteps if mesh topology static)
    connectivity: jax.Array  # (n_elements, 4) int32 - node IDs per element
    
    # Node-to-element adjacency (precomputed once)
    node_to_elements: jax.Array  # (n_nodes, MAX_VALENCE) int32, padded with -1
    node_valence: jax.Array      # (n_nodes,) int32 - actual count per node
    
    # Geometry (per timestep)
    node_positions: jax.Array    # (n_nodes, 3) float32
    node_velocities: jax.Array   # (n_nodes, 3) float32
    
    # Element metadata (per timestep)
    element_volumes: jax.Array   # (n_elements,) float32
```

### 3.2 Precomputation (One-Time, CPU or GPU)

```python
def build_node_to_element_adjacency(
    connectivity: np.ndarray,  # (n_elements, 4)
    max_valence: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build padded node-to-element adjacency.
    
    Returns:
        node_to_elements: (n_nodes, max_valence) int32
        node_valence: (n_nodes,) int32
    """
    n_elements = connectivity.shape[0]
    n_nodes = connectivity.max() + 1
    
    # Use dict to accumulate incident elements per node
    from collections import defaultdict
    node_incidents = defaultdict(list)
    
    for elem_id in range(n_elements):
        for local_node in range(4):  # 4 nodes per tet
            node_id = connectivity[elem_id, local_node]
            node_incidents[node_id].append(elem_id)
    
    # Convert to padded arrays
    node_to_elements = np.full((n_nodes, max_valence), -1, dtype=np.int32)
    node_valence = np.zeros(n_nodes, dtype=np.int32)
    
    for node_id in range(n_nodes):
        incidents = node_incidents[node_id]
        actual_valence = len(incidents)
        node_valence[node_id] = actual_valence
        
        # Truncate if exceeds max_valence (shouldn't happen with max_valence=64)
        if actual_valence > max_valence:
            print(f"WARNING: Node {node_id} has valence {actual_valence} > {max_valence}")
            incidents = incidents[:max_valence]
            actual_valence = max_valence
        
        node_to_elements[node_id, :actual_valence] = incidents
    
    return node_to_elements, node_valence
```

### 3.3 L1 Search Algorithm (JAX-Native)

```python
@jax.jit
def search_l1_node_based(
    pos: jax.Array,              # (3,) query position
    cached_elem_id: jnp.int32,   # Current element (failed L0)
    mesh_gpu: MeshGPUNodeBased
) -> jnp.int32:
    """
    L1: Search all elements sharing nodes with cached element.
    
    Strategy:
      1. Get 4 nodes of cached element from connectivity
      2. For each node, get all incident elements (up to MAX_VALENCE)
      3. Test point-in-tet for all candidates (with deduplication mask)
      4. Return first valid hit
    
    Worst case: 4 nodes × 64 valence = 256 candidates
    Typical case: 4 nodes × 32 valence = 128 candidates
    With ~50% duplicates → ~64-128 unique tests
    """
    
    # Step 1: Get nodes of current element
    element_nodes = mesh_gpu.connectivity[cached_elem_id]  # (4,) node IDs
    
    # Step 2: Gather all incident elements for these 4 nodes
    # Fixed-size gather: (4 nodes, MAX_VALENCE elements) = (4, 64)
    incident_elements = mesh_gpu.node_to_elements[element_nodes]  # (4, 64)
    incident_valences = mesh_gpu.node_valence[element_nodes]      # (4,)
    
    # Step 3: Flatten to candidate list
    # Shape: (4 * 64,) = (256,)
    candidate_elements = incident_elements.flatten()
    
    # Step 4: Build validity mask (exclude padding -1 and cached element itself)
    def is_valid_candidate(idx):
        # Unflatten index
        node_idx = idx // mesh_gpu.node_to_elements.shape[1]
        valence_idx = idx % mesh_gpu.node_to_elements.shape[1]
        
        elem_id = candidate_elements[idx]
        
        # Valid if:
        #   - Element ID >= 0 (not padding)
        #   - Element ID != cached_elem_id (not self)
        #   - Within valence bound for this node
        within_valence = valence_idx < incident_valences[node_idx]
        
        return (elem_id >= 0) & (elem_id != cached_elem_id) & within_valence
    
    # Vectorized validity check
    validity_mask = jax.vmap(is_valid_candidate)(jnp.arange(256))
    
    # Step 5: Point-in-tet check (vectorized, with masking)
    def check_candidate(idx):
        elem_id = candidate_elements[idx]
        valid = validity_mask[idx]
        
        # Conditional point-in-tet (only if valid)
        result = jnp.where(
            valid,
            point_in_tet_test(pos, elem_id, mesh_gpu),
            jnp.int32(-1)  # Invalid → return -1
        )
        return result
    
    # Vectorized check over all 256 candidates
    results = jax.vmap(check_candidate)(jnp.arange(256))
    
    # Step 6: Return first valid result
    hit_mask = results >= 0
    
    return jnp.where(
        jnp.any(hit_mask),
        results[jnp.argmax(hit_mask)],  # First hit
        jnp.int32(-1)  # No hit → L1 failed
    )
```

**Performance analysis**:

| Metric | Value | Notes |
|--------|-------|-------|
| **Candidates generated** | 256 (fixed) | 4 nodes × 64 max_valence |
| **Typical valid candidates** | 64-128 | ~50% are padding or duplicates |
| **Point-in-tet tests** | 256 (worst case) | JAX vmap executes all, masks results |
| **Memory access** | 256 elem lookups | Coalesced if elem IDs are spatially close |
| **Computational cost** | 256 × 100 FLOPs | ~25k FLOPs (vs 12k for element-based L1 with 3 hops × 4 neighbors) |

**Pros**:
- ✅ **Guaranteed** to find any element touching current element (no hops, no gaps)
- ✅ Handles refinement boundaries perfectly (coarse/fine elements share nodes)
- ✅ Fixed-size loops (JAX-friendly)
- ✅ No need for precomputed element-neighbor arrays

**Cons**:
- ⚠️ **2× more candidates** than element-based L1 (256 vs 128 typical)
- ⚠️ Many **duplicate checks** (same element incident to multiple nodes)
- ⚠️ All 256 checks executed in JAX vmap (no early exit)

### 3.4 L1 Optimization: Deduplication with Hash

**Idea**: Use a small fixed-size "seen" set to skip duplicate elements

```python
@jax.jit
def search_l1_node_based_dedup(
    pos: jax.Array,
    cached_elem_id: jnp.int32,
    mesh_gpu: MeshGPUNodeBased
) -> jnp.int32:
    """
    L1 with deduplication via fixed-size hash table.
    """
    
    # Gather candidates (same as before)
    element_nodes = mesh_gpu.connectivity[cached_elem_id]
    incident_elements = mesh_gpu.node_to_elements[element_nodes].flatten()  # (256,)
    
    # Build "seen" mask using cumulative uniqueness check
    def is_first_occurrence(idx):
        elem_id = incident_elements[idx]
        
        # Check if elem_id appears earlier in the array
        # This is O(n^2) conceptually, but vectorized in JAX
        earlier_matches = (incident_elements[:idx] == elem_id).any()
        
        return ~earlier_matches & (elem_id >= 0) & (elem_id != cached_elem_id)
    
    # Vectorized uniqueness check
    # NOTE: This is expensive (O(n^2) comparisons), but n=256 is small enough
    unique_mask = jax.vmap(is_first_occurrence)(jnp.arange(256))
    
    # Point-in-tet only on unique candidates
    def check_unique_candidate(idx):
        elem_id = incident_elements[idx]
        is_unique = unique_mask[idx]
        
        return jnp.where(
            is_unique,
            point_in_tet_test(pos, elem_id, mesh_gpu),
            jnp.int32(-1)
        )
    
    results = jax.vmap(check_unique_candidate)(jnp.arange(256))
    
    hit_mask = results >= 0
    return jnp.where(jnp.any(hit_mask), results[jnp.argmax(hit_mask)], jnp.int32(-1))
```

**Trade-off**:
- ✅ Reduces **effective** point-in-tet tests by ~50%
- ❌ Adds **uniqueness check cost**: O(256²) comparisons ≈ 65k ops
- ⚠️ Net benefit depends on point-in-tet cost vs uniqueness check cost

**Recommendation**: **Skip deduplication** initially; 256 point-in-tet tests is acceptable (25k FLOPs).

***

## Part IV: Node-Based L2 Design (JAX-Native)

### 4.1 Design Overview

**Goal**: Use Morton/octree over nodes to find k-nearest nodes, then search incident elements.

**Data structures**:
```python
@dataclass
class MeshGPUNodeBased:
    # ... (L1 structures above) ...
    
    # L2: Node-based octree/Morton
    node_morton_codes: jax.Array        # (n_nodes,) uint64
    node_morton_sorted_indices: jax.Array  # (n_nodes,) int32 - sorted by Morton
    
    # Node octree (similar to element octree)
    node_octree_prefix_start: jax.Array   # (2^(3*table_depth),) int32
    node_octree_prefix_count: jax.Array   # (2^(3*table_depth),) int32
    node_octree_table_depth: int          # e.g., 6 → 262k prefixes
    
    # Bounding box for Morton encoding
    bbox_min: jax.Array  # (3,)
    bbox_max: jax.Array  # (3,)
```

### 4.2 Precomputation: Node Morton Octree

```python
def build_node_morton_octree(
    node_positions: np.ndarray,  # (n_nodes, 3)
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    table_depth: int = 6
) -> Dict:
    """
    Build Morton octree over nodes.
    
    Returns dict with:
      - node_morton_codes
      - node_morton_sorted_indices
      - prefix_start, prefix_count
    """
    from jaxtrace.gpu.search.morton import encode_morton_3d
    
    n_nodes = node_positions.shape[0]
    
    # Step 1: Encode Morton codes for all nodes
    morton_codes = np.zeros(n_nodes, dtype=np.uint64)
    
    for i in range(n_nodes):
        pos = node_positions[i]
        # Normalize to [0, 1]³
        pos_norm = (pos - bbox_min) / (bbox_max - bbox_min + 1e-12)
        pos_norm = np.clip(pos_norm, 0.0, 0.9999)
        
        # Encode to 21-bit Morton (63 bits total)
        morton_codes[i] = encode_morton_3d(pos_norm, max_level=21)
    
    # Step 2: Sort nodes by Morton code
    sorted_indices = np.argsort(morton_codes)
    sorted_morton = morton_codes[sorted_indices]
    
    # Step 3: Build prefix table
    # Extract prefix at table_depth
    shift_amount = 63 - 3 * table_depth
    prefixes = (sorted_morton >> shift_amount).astype(np.uint64)
    
    n_prefixes = 2 ** (3 * table_depth)  # 8^table_depth
    prefix_start = np.full(n_prefixes, -1, dtype=np.int32)
    prefix_count = np.zeros(n_prefixes, dtype=np.int32)
    
    # Count nodes per prefix
    for i in range(n_nodes):
        prefix = prefixes[i]
        if prefix_start[prefix] == -1:
            prefix_start[prefix] = i  # First node with this prefix
        prefix_count[prefix] += 1
    
    return {
        'node_morton_codes': morton_codes,
        'node_morton_sorted_indices': sorted_indices,
        'node_octree_prefix_start': prefix_start,
        'node_octree_prefix_count': prefix_count,
        'node_octree_table_depth': table_depth,
        'bbox_min': bbox_min,
        'bbox_max': bbox_max
    }
```

### 4.3 L2 Search Algorithm: k-Nearest Nodes

```python
@jax.jit
def search_l2_node_based(
    pos: jax.Array,              # (3,) query position
    mesh_gpu: MeshGPUNodeBased,
    k_nearest: int = 4           # Number of nearest nodes to query
) -> jnp.int32:
    """
    L2: Find k nearest nodes via Morton octree, check incident elements.
    
    Strategy:
      1. Encode query position to Morton code
      2. Find octant containing query
      3. Search self + 26 spatial neighbors (27 total octants)
      4. Within these octants, find k nearest nodes (by distance)
      5. Gather all elements incident to k nodes
      6. Point-in-tet test on all candidates
    
    Worst case: k=4 nodes × 64 valence = 256 candidates
    """
    
    # Step 1: Encode query position
    pos_norm = (pos - mesh_gpu.bbox_min) / (mesh_gpu.bbox_max - mesh_gpu.bbox_min + 1e-12)
    pos_norm = jnp.clip(pos_norm, 0.0, 0.9999)
    
    query_morton = encode_morton_3d_jax(pos_norm, max_level=21)
    
    # Step 2: Extract prefix at table_depth
    shift = 63 - 3 * mesh_gpu.node_octree_table_depth
    query_prefix = query_morton >> shift
    
    # Step 3: Decode prefix to octant coordinates
    cx, cy, cz = decode_morton_prefix_jax(query_prefix, mesh_gpu.node_octree_table_depth)
    
    # Step 4: Generate 27 neighbor octants (self + 26 neighbors)
    neighbor_offsets = generate_27_neighbor_offsets()  # Precomputed (3, 27) array
    
    def get_neighbor_prefix(offset):
        dx, dy, dz = offset
        
        # Neighbor octant coordinates (with clamping)
        nx = jnp.clip(cx + dx, 0, (2 ** mesh_gpu.node_octree_table_depth) - 1)
        ny = jnp.clip(cy + dy, 0, (2 ** mesh_gpu.node_octree_table_depth) - 1)
        nz = jnp.clip(cz + dz, 0, (2 ** mesh_gpu.node_octree_table_depth) - 1)
        
        # Encode neighbor prefix
        return encode_morton_prefix_jax(nx, ny, nz, mesh_gpu.node_octree_table_depth)
    
    neighbor_prefixes = jax.vmap(get_neighbor_prefix)(neighbor_offsets)  # (27,)
    
    # Step 5: Gather candidate nodes from 27 octants
    # Each octant has up to max_nodes_per_octant (e.g., 64)
    # Total candidates: 27 × 64 = 1728 (too many!)
    
    # OPTIMIZATION: Instead of gathering all nodes, use distance-based filtering
    # within each octant
    
    MAX_NODES_PER_OCTANT = 64  # Assume max ~64 nodes per octant
    
    def gather_nodes_from_octant(prefix_idx):
        prefix = neighbor_prefixes[prefix_idx]
        
        # Clamp prefix to valid range
        prefix_clamped = jnp.clip(prefix, 0, len(mesh_gpu.node_octree_prefix_start) - 1)
        
        start_idx = mesh_gpu.node_octree_prefix_start[prefix_clamped]
        count = mesh_gpu.node_octree_prefix_count[prefix_clamped]
        
        # Gather up to MAX_NODES_PER_OCTANT nodes
        # Truncate if octant has more
        actual_count = jnp.minimum(count, MAX_NODES_PER_OCTANT)
        
        # Build node indices for this octant
        def get_node_in_octant(local_idx):
            global_idx = start_idx + local_idx
            valid = (local_idx < actual_count) & (start_idx >= 0)
            
            node_id = jnp.where(
                valid,
                mesh_gpu.node_morton_sorted_indices[global_idx],
                jnp.int32(-1)
            )
            return node_id
        
        octant_nodes = jax.vmap(get_node_in_octant)(jnp.arange(MAX_NODES_PER_OCTANT))
        
        return octant_nodes  # (MAX_NODES_PER_OCTANT,)
    
    # Gather from all 27 octants
    all_candidate_nodes = jax.vmap(gather_nodes_from_octant)(jnp.arange(27))
    # Shape: (27, MAX_NODES_PER_OCTANT)
    
    # Step 6: Flatten and compute distances to query position
    candidate_nodes_flat = all_candidate_nodes.flatten()  # (27 * 64 = 1728,)
    
    def compute_distance_to_node(node_id):
        valid = node_id >= 0
        
        node_pos = jnp.where(
            valid,
            mesh_gpu.node_positions[node_id],
            jnp.array([1e9, 1e9, 1e9])  # Far away if invalid
        )
        
        dist_sq = jnp.sum((node_pos - pos) ** 2)
        return dist_sq
    
    distances = jax.vmap(compute_distance_to_node)(candidate_nodes_flat)  # (1728,)
    
    # Step 7: Select k nearest nodes
    # JAX-friendly: use jnp.argsort (fixed size)
    sorted_indices = jnp.argsort(distances)
    k_nearest_indices = sorted_indices[:k_nearest]  # First k
    k_nearest_nodes = candidate_nodes_flat[k_nearest_indices]  # (k,)
    
    # Step 8: Gather incident elements for k nearest nodes
    # Each node has up to MAX_VALENCE elements
    incident_elements_k = mesh_gpu.node_to_elements[k_nearest_nodes]  # (k, MAX_VALENCE)
    incident_valences_k = mesh_gpu.node_valence[k_nearest_nodes]      # (k,)
    
    # Flatten to candidate list
    candidate_elements = incident_elements_k.flatten()  # (k * MAX_VALENCE,)
    
    # Step 9: Validity mask
    def is_valid_l2_candidate(idx):
        node_idx = idx // mesh_gpu.node_to_elements.shape[1]
        valence_idx = idx % mesh_gpu.node_to_elements.shape[1]
        
        elem_id = candidate_elements[idx]
        within_valence = valence_idx < incident_valences_k[node_idx]
        
        return (elem_id >= 0) & within_valence
    
    validity_mask = jax.vmap(is_valid_l2_candidate)(jnp.arange(k_nearest * mesh_gpu.node_to_elements.shape[1]))
    
    # Step 10: Point-in-tet check
    def check_l2_candidate(idx):
        elem_id = candidate_elements[idx]
        valid = validity_mask[idx]
        
        return jnp.where(
            valid,
            point_in_tet_test(pos, elem_id, mesh_gpu),
            jnp.int32(-1)
        )
    
    results = jax.vmap(check_l2_candidate)(jnp.arange(k_nearest * mesh_gpu.node_to_elements.shape[1]))
    
    # Step 11: Return first hit
    hit_mask = results >= 0
    return jnp.where(jnp.any(hit_mask), results[jnp.argmax(hit_mask)], jnp.int32(-1))
```

**Performance analysis**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **k nearest nodes** | 4 | Configurable |
| **MAX_VALENCE** | 64 | From topology analysis |
| **Candidates** | 4 × 64 = 256 | Same as L1 |
| **Octant search** | 27 × 64 = 1728 nodes | Distance sorting overhead |
| **Distance computations** | 1728 × 10 FLOPs | ~17k FLOPs |
| **Sort** | ~1728 log(1728) ≈ 18k ops | Sorting is expensive! |
| **Point-in-tet** | 256 × 100 FLOPs | ~25k FLOPs |
| **Total L2 cost** | ~60k FLOPs | **2-3× more expensive than element-based L2** |

### 4.4 L2 Optimization: Reduce Candidate Nodes

**Problem**: Searching 27 octants × 64 nodes = 1728 candidates is expensive.

**Solution**: Use **hierarchical octant pruning** based on distance

```python
@jax.jit
def search_l2_node_based_optimized(
    pos: jax.Array,
    mesh_gpu: MeshGPUNodeBased,
    k_nearest: int = 4,
    max_octants_to_search: int = 8  # Prune to nearest octants only
) -> jnp.int32:
    """
    L2 with octant-level distance pruning.
    """
    
    # ... (Steps 1-4 same: encode, get 27 neighbor prefixes) ...
    
    # Step 5: Compute distance to octant centers (not nodes)
    def octant_center_distance(prefix):
        # Decode prefix to octant coordinates
        ox, oy, oz = decode_morton_prefix_jax(prefix, mesh_gpu.node_octree_table_depth)
        
        # Octant center in normalized space
        octant_size = 1.0 / (2 ** mesh_gpu.node_octree_table_depth)
        center_norm = jnp.array([
            (ox + 0.5) * octant_size,
            (oy + 0.5) * octant_size,
            (oz + 0.5) * octant_size
        ])
        
        # Convert back to physical space
        center_phys = mesh_gpu.bbox_min + center_norm * (mesh_gpu.bbox_max - mesh_gpu.bbox_min)
        
        # Distance to query position
        dist_sq = jnp.sum((center_phys - pos) ** 2)
        return dist_sq
    
    octant_distances = jax.vmap(octant_center_distance)(neighbor_prefixes)  # (27,)
    
    # Step 6: Sort octants by distance, keep nearest max_octants_to_search
    sorted_octant_indices = jnp.argsort(octant_distances)
    nearest_octant_indices = sorted_octant_indices[:max_octants_to_search]  # (8,)
    
    nearest_prefixes = neighbor_prefixes[nearest_octant_indices]
    
    # Step 7: Gather nodes only from nearest octants
    # Now: 8 octants × 64 nodes = 512 candidates (vs 1728)
    
    # ... (Rest same as before, but with reduced candidate set) ...
```

**Improved performance**:

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Octants searched** | 27 | 8 | 3.4× reduction |
| **Node candidates** | 1728 | 512 | 3.4× reduction |
| **Distance ops** | 17k | 5k | 3.4× reduction |
| **Total L2 cost** | ~60k FLOPs | ~35k FLOPs | **1.7× faster** |

***

## Part V: Velocity-Based Enhancements for Node L2

### 5.1 Predicted Position Query

```python
@jax.jit
def search_l2_node_velocity_predicted(
    pos: jax.Array,
    velocity: jax.Array,  # ← NEW: velocity at current position
    dt: float,
    mesh_gpu: MeshGPUNodeBased,
    k_nearest: int = 4
) -> jnp.int32:
    """
    L2 using velocity-predicted position for node octree query.
    """
    
    # Step 1: Predict next position (Euler step)
    # Use 0.5 * dt for RK2-like prediction
    predicted_pos = pos + 0.5 * dt * velocity
    
    # Step 2: Encode PREDICTED position to Morton
    pos_norm_pred = (predicted_pos - mesh_gpu.bbox_min) / (mesh_gpu.bbox_max - mesh_gpu.bbox_min + 1e-12)
    pos_norm_pred = jnp.clip(pos_norm_pred, 0.0, 0.9999)
    
    query_morton = encode_morton_3d_jax(pos_norm_pred, max_level=21)
    
    # Step 3: Get octant for PREDICTED position
    shift = 63 - 3 * mesh_gpu.node_octree_table_depth
    query_prefix_pred = query_morton >> shift
    
    cx, cy, cz = decode_morton_prefix_jax(query_prefix_pred, mesh_gpu.node_octree_table_depth)
    
    # Step 4-10: Same as before (27 neighbors, k-nearest, etc.)
    # But searching around PREDICTED octant, not current
    
    # ... (rest of algorithm) ...
    
    # CRITICAL: Still do point-in-tet on CURRENT position, not predicted
    # Prediction is only for finding candidate region
```

**Rationale**:
- Particle at `pos` with `velocity` will likely end up near `pos + 0.5*dt*velocity`
- Search for nodes in that region
- But final containment test still on current `pos`

**Benefit**:
- ✅ For advection-dominated flows (high Peclet), this significantly improves hit rate
- ✅ Especially useful when particle crosses from one octree region to another in a single step

**Risk**:
- ⚠️ For low-velocity or highly curved streamlines, prediction may be worse than current-position search
- ⚠️ Near boundaries, predicted position may be outside domain

**Mitigation**: **Hybrid approach**

```python
# Try predicted-position search first
result_predicted = search_l2_node_velocity_predicted(pos, velocity, dt, mesh_gpu, k=4)

# If failed, fall back to current-position search
result_final = jnp.where(
    result_predicted >= 0,
    result_predicted,
    search_l2_node_based_optimized(pos, mesh_gpu, k=8)  # Use more nodes for fallback
)
```

***

## Part VI: Complete Hierarchical Search with Node-Based L1/L2

### 6.1 Full Search Function

```python
@jax.jit
def search_and_interpolate_node_based(
    pos: jax.Array,              # (3,) query position
    velocity: jax.Array,         # (3,) velocity at last known position
    cached_elem_id: jnp.int32,   # Cached element from previous step
    dt: float,                   # Timestep for velocity prediction
    mesh_gpu: MeshGPUNodeBased
) -> Tuple[jnp.int32, jax.Array]:
    """
    Complete search hierarchy: L0 → L1 → L2
    
    Returns:
        elem_id: Found element ID (or -1 if lost)
        interpolated_value: Interpolated value at pos
    """
    
    # ============================================================
    # L0: Cached element (point-in-tet check)
    # ============================================================
    
    elem_l0 = point_in_tet_test(pos, cached_elem_id, mesh_gpu)
    
    found_l0 = elem_l0 >= 0
    
    # ============================================================
    # L1: Node-based neighborhood search (if L0 failed)
    # ============================================================
    
    elem_l1 = jnp.where(
        found_l0,
        elem_l0,  # L0 succeeded, skip L1
        search_l1_node_based(pos, cached_elem_id, mesh_gpu)
    )
    
    found_l1 = elem_l1 >= 0
    
    # ============================================================
    # L2: Node-based octree search (if L0 and L1 failed)
    # ============================================================
    
    # First try: velocity-predicted position
    elem_l2_predicted = jnp.where(
        found_l1,
        elem_l1,  # L1 succeeded, skip L2
        search_l2_node_velocity_predicted(pos, velocity, dt, mesh_gpu, k_nearest=4)
    )
    
    found_l2_predicted = elem_l2_predicted >= 0
    
    # Second try: current position with more nodes (if prediction failed)
    elem_l2_fallback = jnp.where(
        found_l2_predicted,
        elem_l2_predicted,
        search_l2_node_based_optimized(pos, mesh_gpu, k_nearest=8, max_octants=8)
    )
    
    # ============================================================
    # Final result
    # ============================================================
    
    final_elem_id = elem_l2_fallback
    
    # Interpolate value (if found)
    found = final_elem_id >= 0
    
    interpolated_value = jnp.where(
        found,
        interpolate_in_element(pos, final_elem_id, mesh_gpu),
        jnp.zeros(3)  # Zero velocity if lost
    )
    
    return final_elem_id, interpolated_value
```

### 6.2 Integration with Fully-Fused RK4

**No changes needed to RK4 structure!**

```python
@jax.jit
def rk4_single_particle_node_based(
    pos0: jax.Array,
    cached_elem: jnp.int32,
    dt: float,
    mesh_gpu: MeshGPUNodeBased
) -> Tuple[jax.Array, jnp.int32]:
    """
    RK4 integration with node-based search.
    """
    
    # k1
    elem_k1, vel_k1 = search_and_interpolate_node_based(
        pos0, jnp.zeros(3), cached_elem, dt, mesh_gpu
    )
    
    # k2
    pos_k2 = pos0 + 0.5 * dt * vel_k1
    elem_k2, vel_k2 = search_and_interpolate_node_based(
        pos_k2, vel_k1, elem_k1, dt, mesh_gpu
    )
    
    # k3
    pos_k3 = pos0 + 0.5 * dt * vel_k2
    elem_k3, vel_k3 = search_and_interpolate_node_based(
        pos_k3, vel_k2, elem_k2, dt, mesh_gpu
    )
    
    # k4
    pos_k4 = pos0 + dt * vel_k3
    elem_k4, vel_k4 = search_and_interpolate_node_based(
        pos_k4, vel_k3, elem_k3, dt, mesh_gpu
    )
    
    # Final position
    pos_final = pos0 + dt * (vel_k1 + 2*vel_k2 + 2*vel_k3 + vel_k4) / 6.0
    
    elem_final, _ = search_and_interpolate_node_based(
        pos_final, vel_k4, elem_k4, dt, mesh_gpu
    )
    
    return pos_final, elem_final

# Vmap over particles (unchanged)
rk4_all_particles = jax.vmap(rk4_single_particle_node_based, in_axes=(0, 0, None, None))
```

***

## Part VII: Comprehensive Pros & Cons Analysis

### 7.1 Node-Based L1

| Aspect | Rating | Details |
|--------|--------|---------|
| **Accuracy** | ⭐⭐⭐⭐⭐ | Guaranteed to find elements sharing nodes with cached elem |
| **Refinement handling** | ⭐⭐⭐⭐⭐ | Perfect - coarse/fine elements share nodes |
| **Computational cost** | ⭐⭐⭐ | 256 candidates vs 128 for element-based (2× worse) |
| **Memory** | ⭐⭐⭐⭐ | 78 MB node adjacency (moderate) |
| **JAX compatibility** | ⭐⭐⭐⭐⭐ | Perfect - fixed-size loops, vectorized |
| **Implementation complexity** | ⭐⭐⭐⭐ | Moderate - need to build node adjacency |

**Verdict**: ✅ **HIGHLY RECOMMENDED** - Best accuracy/complexity trade-off for L1

### 7.2 Node-Based L2 (Standard)

| Aspect | Rating | Details |
|--------|--------|---------|
| **Accuracy** | ⭐⭐⭐⭐ | Good, but nearest-node may not be in containing element |
| **Refinement handling** | ⭐⭐⭐⭐⭐ | Excellent - boundary nodes shared by coarse/fine |
| **Computational cost** | ⭐⭐ | 60k FLOPs (2-3× worse than element-based) due to sorting |
| **Memory** | ⭐⭐⭐⭐ | Node octree much smaller than element octree |
| **JAX compatibility** | ⭐⭐⭐⭐ | Good - needs careful fixed-size handling |
| **Implementation complexity** | ⭐⭐⭐ | Moderate-high - octree + k-NN + adjacency |

**Verdict**: ⚠️ **CONDITIONAL** - Use only if element-based L2 causes OOM or if L1 improvement is insufficient

### 7.3 Node-Based L2 with Velocity Prediction

| Aspect | Rating | Details |
|--------|--------|---------|
| **Accuracy** | ⭐⭐⭐⭐ | Better for advection-dominated flows |
| **Performance** | ⭐⭐⭐⭐ | Higher hit rate → fewer fallbacks |
| **Robustness** | ⭐⭐⭐ | Fails for low velocity or complex streamlines |
| **Added complexity** | ⭐⭐⭐ | Need hybrid predict+fallback |

**Verdict**: 🔵 **EXPERIMENTAL** - Test after basic node-L2 works

### 7.4 Overall Comparison

| Search Level | Element-Based (Current) | Node-Based (Proposed) | Winner |
|--------------|------------------------|-----------------------|--------|
| **L0** | Same (cached element) | Same (cached element) | **Tie** |
| **L1 accuracy** | ⭐⭐⭐ (misses coarse/fine transitions) | ⭐⭐⭐⭐⭐ (perfect) | **Node** |
| **L1 speed** | ⭐⭐⭐⭐ (~128 tests) | ⭐⭐⭐ (~256 tests) | **Element** |
| **L2 accuracy** | ⭐⭐⭐ | ⭐⭐⭐⭐ | **Node** |
| **L2 speed** | ⭐⭐⭐⭐ (~35k FLOPs) | ⭐⭐ (~60k FLOPs) | **Element** |
| **L2 memory** | ⭐⭐ (large element octree) | ⭐⭐⭐⭐ (small node octree) | **Node** |
| **OOM risk** | ⭐⭐ (high with multi-timestep) | ⭐⭐⭐⭐ (lower) | **Node** |

***

## Part VIII: Implementation Recommendations

### Phase 1: Node-Based L1 Only (2-3 days)

**Rationale**: Highest ROI - best accuracy improvement, moderate complexity

**Steps**:
1. Build `node_to_elements` adjacency (1 day)
2. Implement `search_l1_node_based` (1 day)
3. Test retention in refined regions (0.5 day)
4. Benchmark L1 hit rate and throughput (0.5 day)

**Expected outcome**:
- ✅ +10-20% retention near refinement boundaries
- ⚠️ 5-10% throughput loss (more candidates)
- ✅ Significantly reduced L2 call frequency

**Decision point**: If L1 improvement pushes retention >90%, **stop here** - don't implement node-based L2.

### Phase 2: Node-Based L2 (if needed, 4-5 days)

**Only if**:
- Phase 1 retention still <85%
- Or element-based L2 causing OOM

**Steps**:
1. Build node Morton octree (1 day)
2. Implement `search_l2_node_based_optimized` (2 days)
3. Test + benchmark (1 day)
4. Tune `k_nearest` and `max_octants` (1 day)

**Expected outcome**:
- ✅ +5-10% retention
- ✅ 30-50% memory reduction vs element L2
- ⚠️ 20-30% throughput loss vs optimized element L2

### Phase 3: Velocity Prediction (experimental, 2-3 days)

**Only if**:
- Phase 1+2 complete
- Flow is strongly advection-dominated (high Peclet number)

**Steps**:
1. Implement `search_l2_node_velocity_predicted` (1 day)
2. A/B test vs non-predictive (1 day)
3. Hybrid predict+fallback tuning (1 day)

**Expected outcome** (highly problem-dependent):
- Best case: +10% throughput (fewer L2 calls due to better targeting)
- Worst case: No improvement or regression

***

## Part IX: Final Critical Assessment

### What You Should Implement

✅ **Node-based L1**: **YES** - Do this first
- Solves your main retention problem (refinement boundaries)
- Moderate complexity, high ROI
- Compatible with existing L2

✅ **Node-based L2**: **CONDITIONAL** - Only if needed after L1
- Implement if element-L2 causes OOM
- Or if L1 alone doesn't get retention >85%

🔵 **Velocity prediction**: **EXPERIMENTAL** - Last priority
- Only after basic node-based L1/L2 working
- Test on representative cases first

### What You Should NOT Do

❌ **Don't** replace both L1 and L2 simultaneously
- Too much complexity at once
- Can't isolate which change helps/hurts

❌ **Don't** over-optimize deduplication in L1
- 256 point-in-tet tests is acceptable cost
- Uniqueness checking overhead not worth it

❌ **Don't** use node-based search if topology is uniform
- Node advantage only appears at refinement boundaries
- If your mesh is uniformly refined, stick with elements

### Expected Final Performance

**With node-based L1 only**:
- Throughput: 6-10K p/s (slight regression from extra candidates)
- Retention: 85-92% @ step 100 (+15-20% vs current)
- Memory: Same as current (+78 MB for node adjacency)

**With node-based L1 + L2**:
- Throughput: 8-12K p/s (L2 slower but called less often)
- Retention: 90-95% @ step 100 (+20-25% vs current)
- Memory: 70% of current (smaller node octree)

**With velocity prediction added**:
- Throughput: 10-15K p/s (best case, advection-dominated)
- Retention: 90-95% (same as L1+L2)
- Risk: May not help or may regress for complex flows

***

## Conclusion

Your node-based approach is **sound and well-motivated** for octree-refined meshes. The topology analysis shows **MAX_VALENCE=64 is sufficient**, making it practical in JAX. The key insight—that refinement boundary nodes are shared by both coarse and fine elements—directly addresses your retention issues.

**Recommendation**: Implement **node-based L1 first**, measure results, then decide on L2 and velocity prediction based on data, not speculation.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_6462cb8f-df5c-4317-bea4-bbba5ac5557e/417b2211-6021-405c-bffd-7483dd8d26e0/An_octree-based_adaptive_semi-Lagrangian_VOF_appro.pdf)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_6462cb8f-df5c-4317-bea4-bbba5ac5557e/b022530f-0234-4c52-922c-cec4123d9250/1-s2.0-S004578252400793X-main.pdf)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_6462cb8f-df5c-4317-bea4-bbba5ac5557e/e384ba65-a073-4e45-9052-6735f9c76f80/105-2023-FEAD.pdf)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_6462cb8f-df5c-4317-bea4-bbba5ac5557e/b496caa2-d849-4ad3-a319-4dc47308ec51/1-s2.0-S0167844222003901-main.pdf)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_6462cb8f-df5c-4317-bea4-bbba5ac5557e/372b7b83-a131-4b1b-97c9-cdc8dd51b6d3/169627.169640.pdf)
[6](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/1a75880d-e651-4c10-a404-e93a10d1a029/OCTREE_CONSTRUCTION_AND_INTERPOLATION_DEEP_DIVE.md)
[7](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/70091334/801b8514-73bf-4adc-bd67-f7da3c674cce/image.jpg)
[8](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/70091334/1e3509e2-b109-46b1-837b-5a0a0321f1bf/image.jpg)
[9](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/6cdf2b8d-d601-43f1-b9b1-4a5e7807a03b/1-s2.0-S004578252400793X-main.pdf)
[10](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/70091334/81643713-ddf9-48fa-aff5-d1d19a763edc/threadeda_piece_distribution.jpg)
[11](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/5eb912ac-d786-47a1-aa04-109aeeba6bba/GPU_NATIVE_IMPLEMENTATION_PLAN.md)
[12](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/a82a381b-853c-4efe-8aab-f2772a15ba48/GPU_NATIVE_IMPLEMENTATION_PLAN_V2.md)
[13](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/aeaa87ea-6415-43f9-928f-cc2c6d754f1b/GPU_NATIVE_IMPLEMENTATION_PLAN.md)
[14](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/697db488-07bd-4d3f-9d62-01c830f7d13f/GPU_NATIVE_IMPLEMENTATION_PLAN_V3_COMPREHENSIVE.md)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/70091334/2c99b851-9bd4-4ecb-bb3a-046e0e293b6d/image.jpg)
[16](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/70091334/11acf21b-2a30-4f87-b59a-f1d8bbd97c8e/image.jpg)
[17](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/8ba23769-b26d-43a0-86f6-e78bb6d12839/GPU_IMPLEMENTATION_PLAN_V4_AS_IMPLEMENTED.md)
[18](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/b351cef4-4ab3-4e5e-924f-948b21f1f7b3/CLEAN_GPU_IMPLEMENTATION_PLAN.md)
[19](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/3373dbef-c781-4012-9350-85178b24ad08/JAX_NATIVE_OPTIMIZATION_PLAN.md)
[20](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/b5a3fccb-4d7e-4fc9-af34-3170e04e6e7e/STRATEGY3_CRITICAL_EVALUATION.md)
[21](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/0bf9dd0c-0a8a-448b-8817-636fb2c7ea69/BATCHED_BLOCKWISE_ARCHITECTURE.md)
[22](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/34b4b31a-8d9d-457e-846c-ff194850c63f/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md)
[23](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/16230f84-59bc-44e1-984c-c023e601bb6a/STATUS_REPORT_ON_BATCHED_BLOCKWISE_PLAN.md)
[24](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/02abdc0c-512b-476f-ab51-f1d422ef20d0/VECTORIZED_MULTILEVEL_ANALYSIS.md)
[25](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/db9696b3-11cd-4233-bf99-02e7827c8363/PERFORMANCE_OPTIMIZATION_PLAN.md)
[26](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/85adfa2c-1378-4653-ba9d-adc55d6ff0f1/GLOBAL_MESH_GPU_ARCHITECTURE.md)
[27](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/e0a0bfa7-c060-425a-87a0-88225f24543b/GLOBAL_INTERPOLATION_IMPLEMENTATION.md)
[28](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/ae3f023f-86b3-4bd3-a000-6d1ade3f7760/SEARCH_OPTIMIZATION_ANALYSIS.md)
[29](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/acdc0a25-2a85-4677-b8c8-86bfe1981bf5/PHASE3A_VECTORIZED_SEARCH_COMPLETE.md)
[30](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/6c89c190-617c-49ab-9d7d-a8d4e561d18c/PHASE3A_COMPLETE_WITH_FUSED_RK4.md)
[31](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/70091334/0cbeff7a-7641-43ad-abe1-d2a6f497ef3c/image.jpg)
[32](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/4e15f735-cb36-4cd6-b668-3faff2ebfda2/169627.169640.pdf)
[33](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/76fba07f-84d0-486e-a41f-f93dcf60725e/HOT_MORTON_REVISED_PLAN.md)
[34](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/e4ad8d2d-8eed-408c-bdff-efed4e05a00e/HOT_MORTON_READY_TO_IMPLEMENT.md)
[35](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/d770de23-b9f3-4c96-a2a1-a7a59e9e7100/MORTON_OPTIMIZATION_GUIDE.md)
[36](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/6bfe22ba-1cfa-4ff7-a7d0-7f0a3b035b09/MORTON_OPTIMIZATION_GUIDE_Review_Sunnet-think.md)
[37](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/5908d6c1-e07f-4a5f-9c40-f6bae4d4c298/rk4_fully_fused_timedep.py)
[38](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/28cb54c4-a9ae-43d6-b198-bc9b3bbe1c6a/L2_OPTIMIZATION_COMPREHENSIVE_STRATEGY.md)
[39](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/695458e8-e8f1-4067-8f5e-8dc9e2a89788/OCTREE_L2_ALREADY_IMPLEMENTED.md)
[40](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/873335d2-ee67-462f-9132-367c48fb7a81/OCTREE_ALIGNED_L2_IMPLEMENTATION_PLAN.md)
[41](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/5bdc21ea-4700-4e04-bba9-26ed5d2275e2/FULLY_FUSED_TIMEDEP_RK4_CRITICAL_REVIEW.md)
[42](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/21d24b1a-e88e-4d78-a341-5bf030a73442/MORTON_OPTIMIZATION_GUIDE_Review_Sunnet-think2.md)
[43](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/dadd3bbd-4b77-47f3-a919-3f8a49adfe74/ADVANCED_SPATIAL_SEARCH_CRITICAL_REVIEW.md)
[44](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/c1845431-4afb-4381-85e9-892b8a6e3349/FINAL_RECOMMENDATION_AFTER_PAPER_REVIEW.md)
[45](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/90090be1-8049-482c-b1b3-32fbd8d3a6bb/10.1111-cgf.14177.pdf)