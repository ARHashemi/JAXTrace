The architecture is a **hashed‑octree‑like (HOT‑like) element search** built around Morton (Z‑order) keys, CSR ranges per leaf, and a bounded inner loop; it is designed to run entirely inside a fused JAX RK4 step, under a vmap over particles, with static shapes and no OOM‑prone padded arrays. The core idea is: precompute Morton codes for all elements, build an octree whose leaves each own at most \(C\) elements (e.g. 64–256), store for each leaf a CSR segment into the Morton‑sorted element list, and at query time map each particle position to the leaf via its Morton key, then test only the elements in that leaf (and optionally neighbor leaves) using a small lax.fori_loop inside the vmap.[1][2][3][4]

***

## 1. Global goals and constraints

- Mesh: large unstructured tetrahedral mesh, but geometrically derived from cubic grid cells (each cube → 4 right‑angled tets).
- Time stepping: fused GPU‑resident RK4, vmap over particles, no CPU orchestration per substep.[1]
- Search: L0/L1 incremental (cached + neighbor tets) plus L2/L3 fallback that is:
  - GPU‑friendly and fully batched.
  - Bounded in per‑particle candidate count (≤C).
  - Free of giant padded arrays and CPU–GPU ping‑pong that previously caused OOM.[3][5]

The HOT‑like element search is the L2/L3 mechanism; L0/L1 remain as in Phase‑3a and run first.[1]

***

## 2. Data structures overview

### 2.1 Coarse blocks aligned with cubes

- The mesh is organized into **cubic cells** (cubes), each split into 4 tets.
- Coarse **blocks** are defined as unions of whole cubes (no cube is split between blocks), so every tet belongs to exactly one cube and exactly one block; this avoids shared elements between blocks and simplifies block search.[3]
- For each element \(e\):
  - Determine its owning cube ID (e.g. \((i,j,k,\text{level})\)).
  - Map cube to block ID \(\text{block}(e)\) via integer division in cube index space.

Resulting arrays (CPU, later uploaded):

- `elem_to_block[e] : int`
- `block_elements[b] : list[int]` (CSR‑encoded later)

### 2.2 Morton codes and HOT‑style keys

Given domain bbox and maximum octree depth \(L\):

For a point \((x,y,z)\):

\[
u_x = \left\lfloor \frac{x - x_{\min}}{x_{\max}-x_{\min}} (2^L-1)\right\rfloor,\quad
u_y = \left\lfloor \frac{y - y_{\min}}{y_{\max}-y_{\min}} (2^L-1)\right\rfloor,\quad
u_z = \left\lfloor \frac{z - z_{\min}}{z_{\max}-z_{\min}} (2^L-1)\right\rfloor.
\]

Write \(u_x = \sum_{i=0}^{L-1} x_i 2^i\) etc., \(x_i\in\{0,1\}\). Morton key:

\[
m(x) = \sum_{i=0}^{L-1} \bigl( x_i 2^{3i} + y_i 2^{3i+1} + z_i 2^{3i+2} \bigr).
\]

This is the same bit‑interleaving used in Z‑order curves and in HOT.[2][4]

For each element \(e\), compute centroid \(c_e\) and its Morton key \(m_e = m(c_e)\).

### 2.3 Per‑block Morton linearization and leaves

Within each block \(b\):

- Gather its elements: `E_b = block_elements[b]`.
- Compute centroids and Morton codes: `m_e` for each \(e ∈ E_b\).
- Sort by Morton: obtain arrays
  - `keys_sorted_b[k] = sorted Morton codes`,
  - `elem_ids_sorted_b[k] = element IDs`.
- Build a local octree for block \(b\) by recursive spatial subdivision until each leaf contains at most \(C\) elements (capacity). Each leaf \(\ell\) has:
  - Depth \(d_\ell\),
  - Morton prefix \(P_\ell\) of length \(3d_\ell\),
  - Index range \([s_\ell, e_\ell)\) in `elem_ids_sorted_b`, defined by all keys in the interval

    \[
    [P_\ell 2^{3(L-d_\ell)},\; (P_\ell+1)2^{3(L-d_\ell)}).
    \]

Implementation: either direct tree construction and then computing `[s,e)` from keys, or use a linear‑octree / LBVH‑like construction.[6][7]

For GPU/JAX, we convert per‑block arrays to **flat CSR‑like global arrays**:

- Concatenate all `elem_ids_sorted_b` into single array `elem_ids_sorted` (global).
- Maintain `block_elem_offsets[b]` pointing to start index of block \(b\) in this global array.
- For each leaf \(\ell\) in block \(b\), store:

  - `leaf_block_id[ℓ] : int`
  - `leaf_start[ℓ] : int` (global index into `elem_ids_sorted`)
  - `leaf_length[ℓ] : int` with \(0<\text{length}_\ell\le C\)

Optionally also:

- `leaf_prefix[ℓ] : int` and `leaf_depth[ℓ] : int`
- A `prefix_to_leaf` table or a compact per‑block structure to map Morton key prefixes to leaf IDs.

All of these are static‑shape arrays suitable for device_put and use in jit.[3]

***

## 3. CPU‑side preprocessing pseudocode

```python
def build_blocks_and_leaves(connectivity, node_positions, cube_grid, L, C):
    # Step 1: map elements to cubes and blocks
    n_elems = connectivity.shape[0]
    elem_to_cube = np.empty(n_elems, dtype=np.int64)
    for e in range(n_elems):
        centroid = node_positions[connectivity[e]].mean(axis=0)
        elem_to_cube[e] = find_leaf_cube_for_point(centroid, cube_grid)

    cube_to_block = {}   # maps cube_id -> block_id
    next_block_id = 0
    for cube_id in np.unique(elem_to_cube):
        ci, cj, ck, level = decode_cube_id(cube_id)
        bi = ci // BLOCK_SIZE_I
        bj = cj // BLOCK_SIZE_J
        bk = ck // BLOCK_SIZE_K
        block_key = (bi, bj, bk)
        if block_key not in cube_to_block:
            cube_to_block[block_key] = next_block_id
            next_block_id += 1

    n_blocks = next_block_id
    elem_to_block = np.empty(n_elems, dtype=np.int32)
    for e in range(n_elems):
        cube_id = elem_to_cube[e]
        ci, cj, ck, level = decode_cube_id(cube_id)
        bi = ci // BLOCK_SIZE_I
        bj = cj // BLOCK_SIZE_J
        bk = ck // BLOCK_SIZE_K
        elem_to_block[e] = cube_to_block[(bi, bj, bk)]

    # Step 2: per-block element lists
    block_elements = [[] for _ in range(n_blocks)]
    for e, b in enumerate(elem_to_block):
        block_elements[b].append(e)

    # Optional: subdivide heavy blocks further until len(block_elements[b]) <= MAX_ELEM_PER_BLOCK
    elem_to_block, block_elements = subdivide_heavy_blocks_if_needed(
        elem_to_block, block_elements, connectivity, node_positions,
        max_elements_per_block=MAX_ELEM_PER_BLOCK
    )

    # Step 3: per-block Morton sort & leaf construction
    global_elem_ids_sorted = []
    leaf_block_id = []
    leaf_start    = []
    leaf_length   = []
    block_elem_offsets = np.zeros(n_blocks+1, dtype=np.int64)

    current_offset = 0
    for b in range(n_blocks):
        elems_b = np.array(block_elements[b], dtype=np.int32)
        block_elem_offsets[b] = current_offset

        if len(elems_b) == 0:
            continue

        centroids_b = compute_centroids(connectivity[elems_b], node_positions)
        morton_b    = morton_encode_batch(centroids_b, bbox_min, bbox_max, L)

        order = np.argsort(morton_b)
        elems_sorted_b  = elems_b[order]
        morton_sorted_b = morton_b[order]

        # Build octree leaves with capacity C
        leaves_b = build_octree_leaves_from_sorted_keys(
            morton_sorted_b, C, L
        )
        # each leaf: (prefix, depth, start_local, length_local)

        for prefix, depth, s_local, length_local in leaves_b:
            leaf_block_id.append(b)
            leaf_start.append(current_offset + s_local)
            leaf_length.append(length_local)

        global_elem_ids_sorted.append(elems_sorted_b)
        current_offset += len(elems_sorted_b)

    block_elem_offsets[n_blocks] = current_offset

    elem_ids_sorted = np.concatenate(global_elem_ids_sorted, axis=0)
    leaf_block_id = np.array(leaf_block_id, dtype=np.int32)
    leaf_start    = np.array(leaf_start,    dtype=np.int64)
    leaf_length   = np.array(leaf_length,   dtype=np.int32)

    return elem_to_block, block_elem_offsets, elem_ids_sorted, \
           leaf_block_id, leaf_start, leaf_length
```

`build_octree_leaves_from_sorted_keys` is a CPU routine that walks the sorted key array and splits ranges until each leaf has ≤C elements, based on key prefixes; its internal details can follow standard linear octree or LBVH algorithms.[7][6]

Upload all outputs to GPU once.

***

## 4. GPU‑side query for a single position (L2/L3 engine)

Inside a fused RK4 step, after L0/L1 have been tried, we need a batched L2/L3 search. At the lowest level, we design a **leaf search for a single particle** in a given leaf ID.

### 4.1 Single‑leaf candidate loop

Assume:

- `elem_ids_sorted : (N_elems,) int32` (global).
- `leaf_start : (N_leaves,) int64`
- `leaf_length : (N_leaves,) int32`
- Fixed capacity `C` (max elements per leaf).

For a particle position `pos`, and leaf ID `ℓ`:

```python
import jax
import jax.numpy as jnp
from jax import lax

MAX_LEAF_ELEMS = 256   # example

def point_in_tet(pos, elem_id, connectivity, node_positions):
    nodes = connectivity[elem_id]          # (4,)
    X     = node_positions[nodes]         # (4,3)
    # Compute barycentric coordinates (shared with interpolation)
    p0 = X[0]
    v1 = X[1] - p0
    v2 = X[2] - p0
    v3 = X[3] - p0
    A  = jnp.stack([v1, v2, v3], axis=1)   # (3,3)
    dp = pos - p0
    lambdas_123 = jnp.linalg.solve(A, dp)
    lam0 = 1.0 - jnp.sum(lambdas_123)
    lambdas = jnp.concatenate([jnp.array([lam0]), lambdas_123])
    return jnp.all(lambdas >= -1e-6)  # inside or on faces
```

```python
def search_in_leaf_single(pos,
                          leaf_id,
                          elem_ids_sorted,
                          leaf_start,
                          leaf_length,
                          connectivity,
                          node_positions):
    start  = leaf_start[leaf_id]
    length = leaf_length[leaf_id]

    def body(j, found_elem):
        active = (found_elem == -1) & (j < length)
        # guard index
        idx    = start + j
        elem_id = jnp.where(active, elem_ids_sorted[idx], 0)

        inside = jnp.where(
            active,
            point_in_tet(pos, elem_id, connectivity, node_positions),
            False,
        )

        return jnp.where(inside & active, elem_id, found_elem)

    init = jnp.int32(-1)
    found_elem = lax.fori_loop(0, MAX_LEAF_ELEMS, body, init)
    return found_elem
```

This loop has **static upper bound** `MAX_LEAF_ELEMS`, so JAX/XLA can compile it once; masking by `j < length` ensures we do not read beyond the leaf’s actual size.[8][9]

### 4.2 Mapping a position to a leaf ID

Given a particle position `pos` and block ID `b` (from `elem_to_block` or from cached element), we need to find a *local leaf* index for this block.

Two main options:

1. **Explicit octree walk** (key‑based):

   - Compute Morton key `m = morton_encode(pos)`.
   - Starting from the block root, at each depth `d`:
     - Use the `(3d..3d+2)` bits of `m` to index child octant `c` in `[0,7]`.
     - Look up child node ID in a small per‑block table.
   - Stop at a leaf; get its *global* leaf ID.

   This needs per‑block arrays like `block_children[b, node, 8]` and `block_is_leaf[b, node]`. These can be made static and GPU‑friendly, but are more complex to store.

2. **Prefix/bucket lookup** (simpler and closer to your CSR idea):

   - Define a fixed prefix length `B` bits (top bits of Morton code within block).
   - For each block `b`, precompute an array `block_prefix_to_leaf[b, 2^B]`:
     - Either direct leaf ID, or `-1` for empty/merged prefixes.
   - At query:
     - `prefix = (m >> (3L - B)) & ((1 << B) - 1)`
     - `leaf_id = block_prefix_to_leaf[b, prefix]`
     - If `leaf_id==-1` (e.g. more refinement than prefix allows), fall back to a small search over a few candidate leaves.

Given the domain and mesh size, you can choose modest `B` (e.g. 9–12) to limit table size while preserving locality.

For JAX, the second approach is usually easier: `block_prefix_to_leaf` is a static 2‑D array, indexing is constant‑time and jit‑friendly.

Pseudocode sketch:

```python
def morton_encode_point(pos, bbox_min, bbox_max, L):
    # pos: (3,)
    scale = (2**L - 1) / (bbox_max - bbox_min)
    u = jnp.floor((pos - bbox_min) * scale).astype(jnp.uint32)   # (3,)
    return interleave_bits_3d(u[0], u[1], u[2])   # uint64

def leaf_id_for_position(pos, block_id, mesh_meta):
    m = morton_encode_point(pos, mesh_meta.bbox_min, mesh_meta.bbox_max, mesh_meta.L)
    prefix = (m >> (3*mesh_meta.L - mesh_meta.B)) & ((1 << mesh_meta.B) - 1)
    leaf_id = mesh_meta.block_prefix_to_leaf[block_id, prefix]
    return leaf_id
```

`interleave_bits_3d` is the usual Morton bit‑interleaver, written with shifts/masks in JAX.

***

## 5. Multi‑level search for one particle (L0/L1 + HOT‑L2/L3)

For each particle, the full search is:

1. **L0: cached element** – same as Phase‑3a.[1]
2. **L1: neighbor/multi‑hop elements** – same as Phase‑3a, using `element_neighbors` and maybe 2‑hop extended neighbors.[3][1]
3. **L2: HOT‑leaf within current block**:
   - If still `elem_id == -1`, use block ID from cached element or re‑compute from position.
   - Map `pos`→`leaf_id` via Morton prefix.
   - Run `search_in_leaf_single` on that leaf.
4. **L3: neighbor blocks**:
   - If still not found, loop over neighbor blocks (e.g. 6 or 26) and for each:
     - Map `pos`→`leaf_id_neighbor` and run `search_in_leaf_single`.
   - Usually only needed for particles near block faces.

Pseudocode for **one particle**:

```python
def multilevel_search_single(pos,
                             cached_elem_id,
                             cached_block_id,
                             mesh_gpu):
    elem_id = cached_elem_id
    block_id = cached_block_id

    # L0: cached element
    elem_id = search_L0_single(pos, elem_id, mesh_gpu.connectivity, mesh_gpu.node_positions)
    found = elem_id >= 0

    # L1: neighbors (vectorizable)
    elem_id_L1 = search_L1_single(pos, elem_id, mesh_gpu.element_neighbors,
                                  mesh_gpu.connectivity, mesh_gpu.node_positions)
    improve_L1 = (elem_id_L1 >= 0) & (~found)
    elem_id = jnp.where(improve_L1, elem_id_L1, elem_id)
    found   = found | improve_L1

    # Update block from elem if found
    block_id = jnp.where(found, mesh_gpu.elem_to_block[elem_id], block_id)

    # L2: same block, HOT-leaf
    def do_L2(elem_id, found, block_id):
        leaf_id = leaf_id_for_position(pos, block_id, mesh_gpu.meta)
        elem_id_L2 = search_in_leaf_single(
            pos, leaf_id,
            mesh_gpu.elem_ids_sorted,
            mesh_gpu.leaf_start,
            mesh_gpu.leaf_length,
            mesh_gpu.connectivity,
            mesh_gpu.node_positions
        )
        improve_L2 = (elem_id_L2 >= 0) & (~found)
        elem_id = jnp.where(improve_L2, elem_id_L2, elem_id)
        found   = found | improve_L2
        # update block if found
        block_id_new = jnp.where(improve_L2, mesh_gpu.elem_to_block[elem_id], block_id)
        return elem_id, found, block_id_new

    elem_id, found, block_id = do_L2(elem_id, found, block_id)

    # L3: neighbor blocks (small fixed loop)
    def l3_body(k, carry):
        elem_id, found, block_id = carry
        active = ~found
        nb = mesh_gpu.block_neighbors_26[block_id, k]  # neighbor block id
        # if nb < 0, skip
        valid_nb = (nb >= 0) & active

        leaf_id_nb = jnp.where(
            valid_nb,
            leaf_id_for_position(pos, nb, mesh_gpu.meta),
            0
        )

        elem_id_L3 = search_in_leaf_single(
            pos, leaf_id_nb,
            mesh_gpu.elem_ids_sorted,
            mesh_gpu.leaf_start,
            mesh_gpu.leaf_length,
            mesh_gpu.connectivity,
            mesh_gpu.node_positions
        )

        improve_L3 = (elem_id_L3 >= 0) & valid_nb & (~found)
        elem_id = jnp.where(improve_L3, elem_id_L3, elem_id)
        found   = found | improve_L3
        block_id = jnp.where(improve_L3, nb, block_id)
        return elem_id, found, block_id

    init = (elem_id, found, block_id)
    elem_id, found, block_id = lax.fori_loop(0, N_NEIGHBORS_26, l3_body, init)

    return elem_id, block_id
```

All loops have **static bounds** (`MAX_LEAF_ELEMS`, `N_NEIGHBORS_26`), so this compiles once and runs efficiently.[10][8]

***

## 6. Batched search and integration into fused RK4 (vmap)

### 6.1 Batched multi‑level search

Define a batched version over particles:

```python
@jax.jit
def multilevel_search_batch(positions,
                            cached_elem_ids,
                            cached_block_ids,
                            mesh_gpu):
    # positions: (N,3)
    # cached_elem_ids, cached_block_ids: (N,)
    search_single = lambda p, e_id, b_id: multilevel_search_single(
        p, e_id, b_id, mesh_gpu
    )
    elem_ids_new, block_ids_new = jax.vmap(search_single)(
        positions, cached_elem_ids, cached_block_ids
    )
    return elem_ids_new, block_ids_new
```

This runs **one kernel** that performs L0/L1/L2/L3 for all particles at once, using `vmap` over particles and `lax.fori_loop` over bounded candidate sets and neighbors.[8][1]

### 6.2 Using multilevel search inside fused RK4

Your existing fused RK4 already does:

- Compute v1 at initial positions.
- For each stage (k2, k3, k4):
  - Compute new positions on GPU,
  - Call a search kernel (currently L0/L1 only),
  - Interpolate velocities at new positions.[1]

You now swap in `multilevel_search_batch`:

```python
@jax.jit
def rk4_step_gpu_fused(positions_initial,
                        elem_ids_initial,
                        block_ids_initial,
                        dt,
                        mesh_gpu,
                        velocity_field_gpu):
    # Stage 1
    v1 = interpolate_velocity_batch_gpu(
        positions_initial,
        elem_ids_initial,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        velocity_field_gpu
    )

    # Stage 2
    pos2 = positions_initial + 0.5 * dt * v1
    elem2, block2 = multilevel_search_batch(
        pos2, elem_ids_initial, block_ids_initial, mesh_gpu
    )
    v2 = interpolate_velocity_batch_gpu(pos2, elem2,
                                        mesh_gpu.connectivity,
                                        mesh_gpu.node_positions,
                                        velocity_field_gpu)

    # Stage 3
    pos3 = positions_initial + 0.5 * dt * v2
    elem3, block3 = multilevel_search_batch(pos3, elem2, block2, mesh_gpu)
    v3 = interpolate_velocity_batch_gpu(pos3, elem3,
                                        mesh_gpu.connectivity,
                                        mesh_gpu.node_positions,
                                        velocity_field_gpu)

    # Stage 4
    pos4 = positions_initial + dt * v3
    elem4, block4 = multilevel_search_batch(pos4, elem3, block3, mesh_gpu)
    v4 = interpolate_velocity_batch_gpu(pos4, elem4,
                                        mesh_gpu.connectivity,
                                        mesh_gpu.node_positions,
                                        velocity_field_gpu)

    positions_final = positions_initial + (dt/6.0) * (v1 + 2*v2 + 2*v3 + v4)

    elem_final, block_final = multilevel_search_batch(
        positions_final, elem_ids_initial, block_ids_initial, mesh_gpu
    )

    return positions_final, elem_final, block_final
```

CPU wrapper stays as in Phase‑3a: upload positions, element IDs, block IDs, and velocity field once per time step; call `rk4_step_gpu_fused`; download final positions/IDs/blocks once.[1]

***

## 7. Performance and OOM considerations

- **Bounded per‑particle work**:
  - Each particle tests at most `MAX_LEAF_ELEMS` tets in its leaf plus `N_NEIGHBORS_26 * MAX_LEAF_ELEMS` candidates in neighboring blocks; all these constants are small and tunable.
- **Static shapes**:
  - All arrays are flat; loops are bounded; no dynamic slicing returns variable‑length arrays, avoiding recompilation and OOM.[9][3]
- **No padded global block arrays**:
  - CSR with per‑leaf `(start,length)` eliminates the old 6.5 GB padded structures that caused your previous OOM and transfer bottlenecks.[5][3]
- **Full GPU residency**:
  - Mesh, Morton structures, and search live on GPU; only positions/element IDs/velocity field snapshots are transferred per step, exactly as in your current fused RK4 implementation.[1]

This architecture is therefore:

- HOT‑like (hashed octree via Morton keys and CSR ranges), closely related to Warren’s HOT / 2HOT algorithms but tailored for unstructured tet element search.[4][6]
- Fully compatible with JAX/XLA (static shapes, bit‑ops, vmap + fori_loop).
- A direct upgrade of your existing Phase‑3a vectorized L0/L1 + fused RK4 design, resolving the L2 bottleneck while preserving GPU performance and OOM safety.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/6c89c190-617c-49ab-9d7d-a8d4e561d18c/PHASE3A_COMPLETE_WITH_FUSED_RK4.md)
[2](https://en.wikipedia.org/wiki/Z-order_curve)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/34b4b31a-8d9d-457e-846c-ff194850c63f/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/4e15f735-cb36-4cd6-b668-3faff2ebfda2/169627.169640.pdf)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/ae3f023f-86b3-4bd3-a000-6d1ade3f7760/SEARCH_OPTIMIZATION_ANALYSIS.md)
[6](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/)
[7](https://arxiv.org/pdf/2307.06345.pdf)
[8](https://docs.jax.dev/en/latest/gpu_performance_tips.html)
[9](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)
[10](https://github.com/google/jax/discussions/12441)