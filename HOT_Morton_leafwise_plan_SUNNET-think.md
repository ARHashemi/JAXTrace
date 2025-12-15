Below is a **revised, simplified, phase‑by‑phase implementation plan** for integrating a **global Morton‑linearized octree L2 search** into your existing fused RK4 + L0/L1 codebase. Early phases use a single global Morton‑sorted element list with leaf offsets; blocks are deferred to later optimization phases.

***

# Morton‑Based Octree L2 Search: Phased Implementation Plan

**Goal:** Replace the current L2 fallback with a JAX‑native, HOT‑like search that uses a global Morton‑sorted element array divided into octree leaves of fixed capacity (C = 128–256 elements per leaf).[1][2]

**What stays:** Your Phase‑3a fused RK4, L0 cached search, L1 multi‑hop neighbor search, and GPU‑resident interpolation.[3]

**What changes:** L2 search is rewritten from scratch to use Morton keys, leaf offsets, and bounded `lax.fori_loop` over small candidate sets.

***

## Phase 0: Current baseline (starting point)

**Status:** Already implemented.

**Components:**

- Fused RK4 with GPU‑resident search and interpolation.[3]
- L0: cached element check (vectorized).
- L1: face neighbors + extended multi‑hop neighbors (vectorized, ~95% hit rate).[3]
- L2 fallback: currently block‑based with padded arrays or CPU‑side filtering (slow, causes OOM for heavy blocks).[4]

**Performance:** ~50–100k p/s, but bottlenecked by L2 for 5–10% of particles.[3]

***

## Phase 1: Build global Morton structure (CPU preprocessing)

**Objective:** Create a **single global Morton‑sorted element list** and divide it into octree leaves with capacity ≤C.

### 1.1 Compute Morton codes for all elements

For each element \(e\):

- Compute centroid \(c_e = \text{mean}(\text{node\_positions}[\text{connectivity}[e]])\).
- Normalize to bounding box and encode Morton key:

\[
m_e = \text{morton\_encode}(c_e, \text{bbox\_min}, \text{bbox\_max}, L)
\]

where \(L\) is maximum tree depth (e.g., 18–21 bits per dimension for typical meshes).

**Pseudocode:**

```python
import numpy as np

def morton_encode_3d(x, y, z, bbox_min, bbox_max, L):
    """
    Compute Morton (Z-order) code for point (x,y,z).
    L: max depth (bits per dimension).
    Returns uint64.
    """
    scale = (2**L - 1) / (bbox_max - bbox_min)
    ux = int(np.floor((x - bbox_min[0]) * scale[0]))
    uy = int(np.floor((y - bbox_min[1]) * scale[1]))
    uz = int(np.floor((z - bbox_min[2]) * scale[2]))
    
    # Interleave bits
    morton = 0
    for i in range(L):
        morton |= ((ux >> i) & 1) << (3*i + 0)
        morton |= ((uy >> i) & 1) << (3*i + 1)
        morton |= ((uz >> i) & 1) << (3*i + 2)
    return morton

# For all elements
n_elems = connectivity.shape[0]
morton_codes = np.empty(n_elems, dtype=np.uint64)
centroids = np.empty((n_elems, 3), dtype=np.float32)

for e in range(n_elems):
    nodes = connectivity[e]
    centroids[e] = node_positions[nodes].mean(axis=0)
    morton_codes[e] = morton_encode_3d(
        centroids[e,0], centroids[e,1], centroids[e,2],
        bbox_min, bbox_max, L
    )
```

### 1.2 Sort elements by Morton code

```python
sorted_order = np.argsort(morton_codes)
morton_sorted = morton_codes[sorted_order]
elem_ids_sorted = np.arange(n_elems, dtype=np.int32)[sorted_order]
```

This is the **global Morton‑sorted element list** that will live on GPU.

### 1.3 Build octree leaves with fixed capacity

Walk the sorted Morton array and split ranges until each leaf contains ≤C elements (e.g., C=256).

**Conceptual algorithm:**

- Start with full range `[0, n_elems)` as root.
- For each node:
  - If `length ≤ C`: mark as leaf, record `(start, length)`.
  - Else: split into 8 octants based on the next 3 bits of Morton codes, recurse.

This produces a list of leaves:

```python
leaves = []  # each entry: (start_idx, length, depth, morton_prefix)
```

**Simplified pseudocode:**

```python
def build_leaves_recursive(start, end, depth, prefix, max_depth=L):
    length = end - start
    if length <= C or depth >= max_depth:
        # Leaf
        leaves.append({
            'start': start,
            'length': length,
            'depth': depth,
            'prefix': prefix
        })
        return
    
    # Find split points for 8 octants
    # Group elements by next 3 bits at current depth
    shift = 3 * (max_depth - depth - 1)
    octant_ranges = [[] for _ in range(8)]
    
    for i in range(start, end):
        octant = (morton_sorted[i] >> shift) & 0x7
        octant_ranges[octant].append(i)
    
    # Recurse into non-empty octants
    for octant in range(8):
        if len(octant_ranges[octant]) > 0:
            child_prefix = (prefix << 3) | octant
            child_start = octant_ranges[octant][0]
            child_end = octant_ranges[octant][-1] + 1
            build_leaves_recursive(
                child_start, child_end, 
                depth + 1, child_prefix, max_depth
            )

leaves = []
build_leaves_recursive(0, n_elems, depth=0, prefix=0, max_depth=L)
```

After this, convert to flat arrays:

```python
n_leaves = len(leaves)
leaf_start  = np.array([leaf['start'] for leaf in leaves], dtype=np.int32)
leaf_length = np.array([leaf['length'] for leaf in leaves], dtype=np.int32)
leaf_depth  = np.array([leaf['depth'] for leaf in leaves], dtype=np.int32)
leaf_prefix = np.array([leaf['prefix'] for leaf in leaves], dtype=np.uint64)
```

### 1.4 Build prefix→leaf lookup table

For fast query‑time mapping from Morton code to leaf ID, create a **prefix table**:

- Choose a fixed prefix length \(B\) (e.g., 9–12 bits, covering depth ~3–4).
- For each possible prefix \(p \in [0, 2^B)\):
  - Find which leaf owns elements with that prefix.
  - Store `prefix_to_leaf[p] = leaf_id` (or -1 if empty).

```python
B = 9  # prefix bits (adjustable)
prefix_to_leaf = np.full(2**B, -1, dtype=np.int32)

for leaf_id, leaf in enumerate(leaves):
    # Leaf covers Morton range [m_start, m_end)
    m_start = morton_sorted[leaf['start']]
    m_end   = morton_sorted[leaf['start'] + leaf['length'] - 1] if leaf['length'] > 0 else m_start
    
    # Map prefixes
    p_start = m_start >> (3*L - B)
    p_end   = m_end >> (3*L - B)
    for p in range(p_start, p_end + 1):
        if p < 2**B:
            prefix_to_leaf[p] = leaf_id
```

### 1.5 Upload to GPU

```python
import jax

mesh_morton_gpu = {
    'elem_ids_sorted':  jax.device_put(elem_ids_sorted),   # (N_elems,) int32
    'leaf_start':       jax.device_put(leaf_start),        # (N_leaves,) int32
    'leaf_length':      jax.device_put(leaf_length),       # (N_leaves,) int32
    'prefix_to_leaf':   jax.device_put(prefix_to_leaf),    # (2^B,) int32
    'bbox_min':         jax.device_put(bbox_min),          # (3,) float32
    'bbox_max':         jax.device_put(bbox_max),          # (3,) float32
    'L':                L,                                  # scalar int
    'B':                B,                                  # scalar int
    'C':                C,                                  # max leaf capacity
    # Also keep existing mesh arrays:
    'connectivity':     mesh_gpu.connectivity,
    'node_positions':   mesh_gpu.node_positions,
    'element_neighbors': mesh_gpu.element_neighbors,
}
```

**Deliverables for Phase 1:**

- `elem_ids_sorted`, `leaf_start`, `leaf_length`, `prefix_to_leaf` arrays on GPU.
- CPU helper script `build_morton_structure.py`.
- Unit test: verify sorted order, leaf coverage (every element in exactly one leaf).

**Time estimate:** 4–6 hours.

***

## Phase 2: Implement L2 search for single position (GPU kernel)

**Objective:** Write a JAX function that, given a position, finds the containing element by querying the Morton leaf structure.

### 2.1 Morton encode on GPU

```python
import jax.numpy as jnp

def morton_encode_jax(pos, bbox_min, bbox_max, L):
    """
    Compute Morton code for position pos (3,) on GPU.
    Returns uint64 (or uint32 if L*3 ≤ 32).
    """
    scale = (2**L - 1) / (bbox_max - bbox_min)
    u = jnp.floor((pos - bbox_min) * scale).astype(jnp.uint32)  # (3,)
    
    # Interleave bits (explicit loop unrolled or via shifts)
    morton = jnp.uint64(0)
    for i in range(L):
        morton |= ((u[0] >> i) & 1) << (3*i + 0)
        morton |= ((u[1] >> i) & 1) << (3*i + 1)
        morton |= ((u[2] >> i) & 1) << (3*i + 2)
    return morton
```

(Note: JAX does not natively support uint64 bit operations in all contexts; use uint32 if \(3L \le 32\) or split into two uint32s.)

### 2.2 Map position to leaf ID

```python
def position_to_leaf_id(pos, mesh_morton):
    """
    Map position to leaf ID via prefix table.
    """
    m = morton_encode_jax(pos, mesh_morton['bbox_min'], mesh_morton['bbox_max'], mesh_morton['L'])
    B = mesh_morton['B']
    prefix = (m >> (3*mesh_morton['L'] - B)).astype(jnp.int32)
    leaf_id = mesh_morton['prefix_to_leaf'][prefix]
    return leaf_id
```

### 2.3 Search within a leaf (bounded loop)

```python
from jax import lax

def point_in_tet_jax(pos, elem_id, connectivity, node_positions):
    """
    Check if pos is inside tet elem_id.
    Returns boolean.
    (Same as existing interpolation barycentric logic.)
    """
    nodes = connectivity[elem_id]
    X = node_positions[nodes]  # (4,3)
    p0 = X[0]
    v1 = X[1] - p0
    v2 = X[2] - p0
    v3 = X[3] - p0
    A = jnp.stack([v1, v2, v3], axis=1)  # (3,3)
    dp = pos - p0
    lambdas_123 = jnp.linalg.solve(A, dp)
    lam0 = 1.0 - jnp.sum(lambdas_123)
    lambdas = jnp.concatenate([jnp.array([lam0]), lambdas_123])
    return jnp.all(lambdas >= -1e-6)

def search_in_leaf_L2(pos, leaf_id, mesh_morton):
    """
    Search for pos in the specified leaf.
    Returns element ID or -1 if not found.
    """
    start = mesh_morton['leaf_start'][leaf_id]
    length = mesh_morton['leaf_length'][leaf_id]
    C = mesh_morton['C']
    
    def body(j, found_elem):
        active = (found_elem == -1) & (j < length)
        idx = start + j
        elem_id = jnp.where(active, mesh_morton['elem_ids_sorted'][idx], 0)
        
        inside = jnp.where(
            active,
            point_in_tet_jax(pos, elem_id, mesh_morton['connectivity'], mesh_morton['node_positions']),
            False
        )
        
        return jnp.where(inside & active, elem_id, found_elem)
    
    init = jnp.int32(-1)
    found_elem = lax.fori_loop(0, C, body, init)
    return found_elem
```

### 2.4 Full L2 search (single particle)

```python
def search_L2_morton_single(pos, mesh_morton):
    """
    L2 search: map pos → leaf → search in leaf.
    Returns element ID or -1.
    """
    leaf_id = position_to_leaf_id(pos, mesh_morton)
    # If leaf_id == -1 (unmapped prefix), return -1
    elem_id = jnp.where(
        leaf_id >= 0,
        search_in_leaf_L2(pos, leaf_id, mesh_morton),
        jnp.int32(-1)
    )
    return elem_id
```

**Deliverables for Phase 2:**

- File: `jaxtrace/gpu/search/morton_l2_search.py`
- Functions: `morton_encode_jax`, `position_to_leaf_id`, `search_in_leaf_L2`, `search_L2_morton_single`
- Unit test: single position queries; validate correctness against brute‑force search.

**Time estimate:** 6–8 hours.

***

## Phase 3: Integrate L2 into multi‑level search and batch (vmap)

**Objective:** Replace old L2 fallback with new Morton L2; keep L0/L1 unchanged; add batching via vmap.

### 3.1 Multi‑level search for single particle (L0 + L1 + Morton‑L2)

```python
def multilevel_search_single_with_morton_L2(pos, cached_elem_id, mesh_gpu, mesh_morton):
    """
    L0: cached element
    L1: face + multi-hop neighbors
    L2: Morton-leaf search
    Returns: element_id
    """
    elem_id = cached_elem_id
    
    # L0: cached
    elem_id = search_L0_single(pos, elem_id, mesh_gpu.connectivity, mesh_gpu.node_positions)
    found = elem_id >= 0
    
    # L1: neighbors (reuse existing Phase-3a vectorized L1)
    elem_id_L1 = search_L1_extended_single(
        pos, elem_id, 
        mesh_gpu.element_neighbors,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions
    )
    improve_L1 = (elem_id_L1 >= 0) & (~found)
    elem_id = jnp.where(improve_L1, elem_id_L1, elem_id)
    found = found | improve_L1
    
    # L2: Morton leaf
    def do_L2():
        return search_L2_morton_single(pos, mesh_morton)
    
    def skip_L2():
        return elem_id
    
    elem_id = lax.cond(~found, do_L2, skip_L2)
    
    return elem_id
```

### 3.2 Batched search (vmap over particles)

```python
@jax.jit
def multilevel_search_batch_with_morton_L2(positions, cached_elem_ids, mesh_gpu, mesh_morton):
    """
    Batched L0+L1+L2 search over all particles.
    positions: (N,3)
    cached_elem_ids: (N,)
    Returns: elem_ids_new (N,)
    """
    search_fn = lambda pos, cached: multilevel_search_single_with_morton_L2(
        pos, cached, mesh_gpu, mesh_morton
    )
    elem_ids_new = jax.vmap(search_fn)(positions, cached_elem_ids)
    return elem_ids_new
```

### 3.3 Integration into fused RK4

Replace the search call in `rk4_step_gpu_fused`:

```python
# OLD (Phase 3a):
# elem_ids_new = search_gpu_fused(positions, cached_elem_ids, mesh_gpu)

# NEW (with Morton L2):
elem_ids_new = multilevel_search_batch_with_morton_L2(
    positions, cached_elem_ids, mesh_gpu, mesh_morton
)
```

Full RK4:

```python
@jax.jit
def rk4_step_gpu_fused_with_morton_L2(positions_initial,
                                       elem_ids_initial,
                                       dt,
                                       mesh_gpu,
                                       mesh_morton,
                                       velocity_field_gpu):
    # v1
    v1 = interpolate_velocity_batch_gpu(
        positions_initial, elem_ids_initial,
        mesh_gpu.connectivity, mesh_gpu.node_positions,
        velocity_field_gpu
    )
    
    # k2
    pos2 = positions_initial + 0.5 * dt * v1
    elem2 = multilevel_search_batch_with_morton_L2(
        pos2, elem_ids_initial, mesh_gpu, mesh_morton
    )
    v2 = interpolate_velocity_batch_gpu(pos2, elem2, mesh_gpu.connectivity,
                                        mesh_gpu.node_positions, velocity_field_gpu)
    
    # k3
    pos3 = positions_initial + 0.5 * dt * v2
    elem3 = multilevel_search_batch_with_morton_L2(pos3, elem2, mesh_gpu, mesh_morton)
    v3 = interpolate_velocity_batch_gpu(pos3, elem3, mesh_gpu.connectivity,
                                        mesh_gpu.node_positions, velocity_field_gpu)
    
    # k4
    pos4 = positions_initial + dt * v3
    elem4 = multilevel_search_batch_with_morton_L2(pos4, elem3, mesh_gpu, mesh_morton)
    v4 = interpolate_velocity_batch_gpu(pos4, elem4, mesh_gpu.connectivity,
                                        mesh_gpu.node_positions, velocity_field_gpu)
    
    # RK4 combination
    positions_final = positions_initial + (dt/6.0) * (v1 + 2*v2 + 2*v3 + v4)
    elem_final = multilevel_search_batch_with_morton_L2(
        positions_final, elem_ids_initial, mesh_gpu, mesh_morton
    )
    
    return positions_final, elem_final
```

**Deliverables for Phase 3:**

- Updated `rk4_step_gpu_fused` with Morton L2.
- Batch test script: run 10k particles, 10 timesteps; verify correctness and measure throughput.

**Time estimate:** 4–6 hours.

***

## Phase 4: Test, validate, and tune

**Objective:** Ensure correctness, measure performance, tune parameters (C, B, L).

### 4.1 Correctness validation

- Run on small test mesh (~10k elements): compare element IDs from Morton L2 vs brute‑force for 1000 random positions.
- Run on full ThreadedA mesh: check that <1% of particles remain unmapped after L0+L1+L2.

### 4.2 Performance benchmarking

Measure throughput (particles/second) for:

- L0+L1 only (current Phase‑3a baseline).
- L0+L1+Morton‑L2.

Expected results:

- L2 hit rate: ~1–5% (most particles found via L0/L1).
- Per‑particle L2 cost: ~10–50 μs (depends on C and leaf depth).
- Overall throughput: target 100–200k p/s (2–4× over current 50k p/s).

### 4.3 Parameter tuning

- **Leaf capacity C:** test 128, 192, 256; smaller C = deeper tree but smaller candidate sets.
- **Prefix bits B:** test 9, 12, 15; larger B = larger table but fewer lookup misses.
- **Max depth L:** 18–21 (balance precision vs bit‑interleaving cost).

### 4.4 Edge cases

- Particles outside domain: should return `-1` gracefully.
- Particles near leaf boundaries: may need neighbor‑leaf fallback (deferred to Phase 5).

**Deliverables for Phase 4:**

- Test suite: `test_morton_l2_correctness.py`, `test_morton_l2_performance.py`.
- Performance report: throughput, GPU utilization, memory usage.
- Tuned parameters documented.

**Time estimate:** 6–8 hours.

***

## Phase 5 (Optional): Add neighbor‑leaf fallback (L3)

**Objective:** Handle rare cases where particle is near leaf boundary and not found in primary leaf.

For a particle not found in its primary leaf:

- Identify neighboring leaves (spatially adjacent octants).
- Test those leaves in a small fixed loop (e.g., 6 or 26 neighbors).

This is analogous to the block‑neighbor L3 in the earlier design, but at leaf granularity.

**Implementation sketch:**

```python
def search_L3_neighbor_leaves(pos, primary_leaf_id, mesh_morton):
    """
    Search in spatial neighbors of primary_leaf_id.
    """
    # Precompute leaf neighbor table (CPU preprocessing)
    # leaf_neighbors[leaf_id, k] = neighbor_leaf_id or -1
    
    def body(k, found_elem):
        active = (found_elem == -1)
        neighbor_leaf = mesh_morton['leaf_neighbors'][primary_leaf_id, k]
        valid = (neighbor_leaf >= 0) & active
        
        elem = jnp.where(
            valid,
            search_in_leaf_L2(pos, neighbor_leaf, mesh_morton),
            jnp.int32(-1)
        )
        improve = (elem >= 0) & valid
        return jnp.where(improve, elem, found_elem)
    
    init = jnp.int32(-1)
    found = lax.fori_loop(0, MAX_LEAF_NEIGHBORS, body, init)
    return found
```

Add to `multilevel_search_single_with_morton_L2`:

```python
# After L2
if still not found:
    elem_id = search_L3_neighbor_leaves(pos, primary_leaf_id, mesh_morton)
```

**Time estimate:** 4–6 hours (optional, only if Phase 4 shows >1% unmapped after L2).

***

## Phase 6 (Future): Add coarse blocks for extreme scalability

**Objective:** For meshes >10M elements or extremely refined regions, partition into coarse blocks (cube‑aligned) and build per‑block Morton structures.

This is the full design from the previous document, deferred until the global Morton L2 is proven and profiled.

**When to implement:**

- If Phase 4 shows memory or performance issues for very large meshes.
- If per‑block parallel search would improve load balancing.

**Effort:** 10–15 hours (requires block partitioning, per‑block prefix tables, block‑neighbor search).

***

## Summary roadmap

| Phase | Task | Deliverables | Time | Dependencies |
|-------|------|--------------|------|--------------|
| 0 | Baseline (current) | Phase‑3a fused RK4 + L0/L1 | Done | — |
| 1 | Build Morton structure (CPU) | `elem_ids_sorted`, `leaf_start/length`, `prefix_to_leaf` | 4–6h | — |
| 2 | Implement L2 (GPU, single pos) | `search_L2_morton_single` | 6–8h | Phase 1 |
| 3 | Integrate into fused RK4 (batch) | Updated `rk4_step_gpu_fused` | 4–6h | Phase 2 |
| 4 | Test, validate, tune | Test suite, performance report | 6–8h | Phase 3 |
| 5 | (Optional) Neighbor‑leaf L3 | `search_L3_neighbor_leaves` | 4–6h | Phase 4 |
| 6 | (Future) Coarse blocks | Per‑block Morton structures | 10–15h | Phase 5 |

**Total core implementation (Phases 1–4):** ~20–28 hours.

**Expected outcome after Phase 4:**

- L2 search fully GPU‑resident, bounded candidate sets, no OOM.
- Throughput: 100–200k p/s (2–4× current).
- <1% unmapped particles (vs 5–10% currently falling back to slow L2).
- Memory: <500 MB GPU (vs several GB in old padded block arrays).

This phased plan lets you validate each piece incrementally, keep your existing working code, and defer complex optimizations (blocks, neighbor‑leaf fallback) until they are proven necessary by profiling real workloads.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/4e15f735-cb36-4cd6-b668-3faff2ebfda2/169627.169640.pdf)
[2](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/6c89c190-617c-49ab-9d7d-a8d4e561d18c/PHASE3A_COMPLETE_WITH_FUSED_RK4.md)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/ae3f023f-86b3-4bd3-a000-6d1ade3f7760/SEARCH_OPTIMIZATION_ANALYSIS.md)