Below is a self‑contained design document for your HOT‑like element search, tailored to a **single‑position query** that runs inside a **GPU‑fused RK4 step**, with **vmap over particles**. It integrates the JAX constraints, Morton/octree ideas, and your existing Phase‑3a fused RK4 architecture.[1][2]

***

## Overview

The goal is a GPU‑friendly element search for a tetrahedral mesh that:

- Uses **Morton (Z‑order) keys** and an **octree‑like hierarchy** (HOT style) for locality.  
- Stores all data in **flat arrays with static shapes** suitable for JAX/XLA.  
- For each particle position, returns a containing element using at most **C candidates per leaf** (e.g. C=256), avoiding OOM.  
- Runs inside a **single `jit`‑compiled, vmap’d RK4 kernel**, without CPU‑side control flow or dynamic shapes.[2][1]

The pipeline has two phases:

1. **Initialization (CPU → GPU, once per mesh snapshot)**  
   Build Morton codes, linear octree leaves, and CSR ranges; upload to GPU.

2. **Query (GPU, per RK4 substage)**  
   For all particles in parallel: L0/L1 neighbor search + HOT‑style leaf lookup + bounded point‑in‑tet search.

***

## Initialization Phase

### 1. Mesh and bounding box

Given:

- `connectivity[e, 4]` – node indices of each tet.  
- `node_pos[n, 3]` – node coordinates.

Compute:

```text
bbox_min = node_pos.min(axis=0)
bbox_max = node_pos.max(axis=0)
```

These are used to normalize coordinates for Morton encoding and to define the global octree domain.

### 2. Morton encoding

Choose maximum octree depth \(L\) (e.g. 10–16, depending on smallest cube size). For each element:

1. Centroid:

\[
c_e = \frac{1}{4} \sum_{k=0}^{3} x_{n_k},
\]

where \(x_{n_k}\) are node coordinates of tet \(e\).

2. Normalized integer coordinates:

\[
u_x = \left\lfloor \frac{c_{e,x} - x_{\min}}{x_{\max}-x_{\min}} (2^L-1)\right\rfloor,
\]
(similarly \(u_y, u_z\)), each in \([0, 2^L-1]\).

3. Morton key (3D Z‑order):

\[
m_e = \sum_{i=0}^{L-1} \bigl( x_i 2^{3i+0} + y_i 2^{3i+1} + z_i 2^{3i+2}\bigr),
\]

where \(x_i\) is bit \(i\) of \(u_x\) etc.[3][4]

CPU pseudocode:

```python
def morton_encode_int3(ux, uy, uz, L):
    # Interleave bits of ux, uy, uz (standard bit-spread trick)
    # Returns uint64 morton code
    ...

def build_element_morton_codes(connectivity, node_pos, bbox_min, bbox_max, L):
    n_elem = connectivity.shape[0]
    morton = np.empty(n_elem, dtype=np.uint64)

    scale = (2**L - 1) / (bbox_max - bbox_min)

    for e in range(n_elem):
        nodes = connectivity[e]          # (4,)
        c = node_pos[nodes].mean(axis=0)
        u = np.clip(((c - bbox_min) * scale).astype(np.int64), 0, 2**L - 1)
        morton[e] = morton_encode_int3(u[0], u[1], u[2], L)

    return morton
```

### 3. Sort elements by Morton code

```python
morton_e = build_element_morton_codes(...)
sort_idx = np.argsort(morton_e)
keys_sorted = morton_e[sort_idx]         # uint64, shape (n_elem,)
elem_ids_sorted = sort_idx.astype(np.int32)
```

This produces a 1‑D “Z‑order curve” of elements with good spatial locality.[2]

### 4. Build linear octree leaves with bounded capacity

Goal: partition the Morton‑sorted array into **leaves of max capacity C** (e.g. 128–256), each representing a small cube.

A data‑structure‑friendly approach:

- Traverse the global implicit octree top‑down.  
- At each node defined by prefix \(P\) (bit length \(3d\)), determine the subrange of `keys_sorted` sharing that prefix.  
- If `count <= C` or `d == L`, mark as leaf and record its range `[start, end)`.  
- Otherwise, create its 8 children and recurse.

CPU pseudocode (conceptual):

```python
Leaf = collections.namedtuple("Leaf", "prefix depth start end")

def build_leaves_from_sorted_keys(keys_sorted, L, C):
    leaves = []

    # Node stack holds (prefix, depth, start, end) in the sorted array
    stack = [(0, 0, 0, len(keys_sorted))]  # root: empty prefix

    while stack:
        prefix, depth, start, end = stack.pop()
        count = end - start
        if count <= C or depth == L:
            leaves.append(Leaf(prefix, depth, start, end))
            continue

        # Split into 8 octants by next 3 bits
        # For each child c in 0..7, find [s_c, e_c) by scanning keys' bits
        for child_oct in range(8):
            child_prefix = (prefix << 3) | child_oct
            s_c, e_c = find_range_for_child(keys_sorted, start, end,
                                            depth, child_oct, L)
            if s_c < e_c:
                stack.append((child_prefix, depth+1, s_c, e_c))

    return sorted(leaves, key=lambda leaf: leaf.start)
```

Then pack `leaves` into dense arrays:

```python
n_leaves = len(leaves)
leaf_prefix   = np.array([ℓ.prefix for ℓ in leaves], dtype=np.uint64)
leaf_depth    = np.array([ℓ.depth for ℓ in leaves], dtype=np.int32)
leaf_start    = np.array([ℓ.start for ℓ in leaves], dtype=np.int32)
leaf_length   = np.array([ℓ.end - ℓ.start for ℓ in leaves], dtype=np.int32)
max_leaf_elems = C     # design constant
```

This is a **pointerless linear octree**: the tree hierarchy is encoded in `leaf_prefix` and `leaf_depth`; the data lives in `keys_sorted` / `elem_ids_sorted` segments.[5][6]

### 5. Fast mapping from Morton key to leaf index

To avoid a binary search over leaves at query time, precompute a **prefix→leaf lookup** table for some fixed number of high bits \(B\) (e.g. 12–18):

- Let `P = B` high bits of Morton code.  
- Build `prefix_leaf[P]` such that it maps to either:
  - A single leaf index \(ℓ\), or  
  - A small range of candidate leaves \([ℓ_0, ℓ_1)\) (still cheap to scan).

Simplest for your case: because leaves are small and relatively regular, you can choose \(B\) so that each prefix maps to at most a few leaves and store `[leaf_start_idx, leaf_end_idx)` (indices into the `leaf_*` arrays).

CPU sketch:

```python
B = 16  # high bits used
n_prefix = 1 << B
prefix_leaf_start = np.full(n_prefix, -1, np.int32)
prefix_leaf_end   = np.full(n_prefix, -1, np.int32)

for leaf_id, leaf in enumerate(leaves):
    # Compute high-B prefix interval of all Morton codes in this leaf
    kmin = keys_sorted[leaf.start]
    kmax = keys_sorted[leaf.start + leaf.length - 1]
    p0 = kmin >> (3*L - B)
    p1 = kmax >> (3*L - B)

    for p in range(p0, p1+1):
        if prefix_leaf_start[p] < 0:
            prefix_leaf_start[p] = leaf_id
        prefix_leaf_end[p] = leaf_id + 1
```

This yields:

- `prefix_leaf_start[p]`, `prefix_leaf_end[p]` arrays of length \(2^B\).  
- At query time, high bits of `m(x)` give you `p`, and candidate leaves are in `[prefix_leaf_start[p], prefix_leaf_end[p])`.

### 6. Upload to GPU

Bundle everything into a mesh GPU struct and upload once, as in your existing architecture.[1][2]

```python
mesh_gpu = MeshGPU(
    connectivity       = device_put(connectivity),
    node_pos           = device_put(node_pos),
    element_neighbors  = device_put(element_neighbors),   # for L0/L1
    bbox_min           = device_put(bbox_min),
    bbox_max           = device_put(bbox_max),
    L                  = L,
    keys_sorted        = device_put(keys_sorted),
    elem_ids_sorted    = device_put(elem_ids_sorted),
    leaf_prefix        = device_put(leaf_prefix),
    leaf_depth         = device_put(leaf_depth),
    leaf_start         = device_put(leaf_start),
    leaf_length        = device_put(leaf_length),
    max_leaf_elems     = C,
    prefix_leaf_start  = device_put(prefix_leaf_start),
    prefix_leaf_end    = device_put(prefix_leaf_end),
)
```

***

## Query Phase: HOT‑like single‑position search

### 1. Morton encoding on GPU

For a particle at position `pos` (3‑vector), use same normalization and bit interleaving as in preprocessing, but implemented in JAX (`jnp`):

```python
def morton_encode_pos(pos, bbox_min, bbox_max, L):
    scale = (2**L - 1.0) / (bbox_max - bbox_min)
    u = jnp.floor((pos - bbox_min) * scale).astype(jnp.uint32)
    ux = jnp.clip(u[0], 0, 2**L - 1)
    uy = jnp.clip(u[1], 0, 2**L - 1)
    uz = jnp.clip(u[2], 0, 2**L - 1)
    return morton_encode_int3_jax(ux, uy, uz, L)  # bit ops only
```

This produces `m = uint64`.

### 2. Map key to candidate leaves

Use the precomputed prefix table:

```python
def morton_to_leaf_range(m, mesh):
    B = mesh.prefix_bits  # e.g. 16
    shift = 3 * mesh.L - B
    p = (m >> shift).astype(jnp.int32)
    leaf_start_idx = mesh.prefix_leaf_start[p]
    leaf_end_idx   = mesh.prefix_leaf_end[p]
    return leaf_start_idx, leaf_end_idx
```

Typically `leaf_end_idx - leaf_start_idx` is 1 or a small integer.

### 3. Scan leaf segments with bounded loop

For each candidate leaf \(ℓ\) in `[leaf_start_idx, leaf_end_idx)`, do:

- Load `(start, length) = (leaf_start[ℓ], leaf_length[ℓ])`.  
- Run a `lax.fori_loop` from 0 to `max_leaf_elems` with guard `j < length`.  
- For each `j`, index `elem_ids_sorted[start + j]` and run `point_in_tet`.

JAX‑safe single‑position search:

```python
def search_hot_single(pos, cached_elem_id, mesh):
    # L0: cached element
    elem = check_cached_element(pos, cached_elem_id, mesh)
    found = elem >= 0

    # L1: neighbors (vectorized neighbor list)
    elem_l1 = search_neighbors(pos, elem, mesh)
    use_l1  = (~found) & (elem_l1 >= 0)
    elem    = jnp.where(use_l1, elem_l1, elem)
    found   = found | use_l1

    # L2/L3: HOT-like search if still not found
    def do_hot_search(_):
        m = morton_encode_pos(pos, mesh.bbox_min, mesh.bbox_max, mesh.L)
        leaf_i0, leaf_i1 = morton_to_leaf_range(m, mesh)

        def leaf_body(i, state):
            found_elem = state
            # Early exit if already found
            def scan_leaf():
                ℓ = leaf_i0 + i
                start = mesh.leaf_start[ℓ]
                length = mesh.leaf_length[ℓ]
                C = mesh.max_leaf_elems

                def body(j, acc):
                    still_search = (acc == -1) & (j < length)
                    idx = start + j
                    elem_id = mesh.elem_ids_sorted[idx]
                    inside = jnp.where(
                        still_search,
                        point_in_tet(pos, elem_id, mesh),
                        False,
                    )
                    return jnp.where(inside & still_search, elem_id, acc)

                return lax.fori_loop(0, C, body, found_elem)

            return lax.cond(found_elem == -1, scan_leaf, lambda: found_elem)

        # loop over candidate leaves
        n_leaf_cand = leaf_i1 - leaf_i0
        return lax.fori_loop(0, n_leaf_cand, leaf_body, jnp.int32(-1))

    elem_hot = lax.cond(found, lambda _: elem, do_hot_search, None)
    # Prefer HOT result if we were not found before
    elem_final = jnp.where(found, elem, elem_hot)
    return elem_final
```

Note:

- All loops have static upper bounds (`max_leaf_elems`, small `n_leaf_cand ≤ few`).  
- No dynamic slicing; only scalar indexing into `elem_ids_sorted[start + j]`.  
- Control flow uses `lax.cond` and `jnp.where`, which JAX supports under `jit`.[7][8]

***

## Integration with fused RK4 and vmap

### 1. Batched search for all particles

Define a batched version using `vmap` over the particle axis:

```python
batched_search_hot = jax.jit(
    jax.vmap(
        search_hot_single,
        in_axes=(0, 0, None),     # pos[i], cached_elem_id[i], mesh
        out_axes=0,
    )
)
```

Within your fused RK4 kernel, you call this batched search after each tentative position update.

### 2. Fused RK4 structure (sketch)

Your Phase‑3a RK4 is already fully GPU‑resident with vectorized L0/L1. Replace the previous incremental search call with `batched_search_hot`:[1]

```python
@jax.jit
def rk4_step_gpu_fused(positions, elem_ids, dt, mesh, vel_field):
    # k1
    v1 = interpolate_velocity_batch_gpu(positions, elem_ids, mesh, vel_field)
    pos2 = positions + 0.5 * dt * v1
    elem2 = batched_search_hot(pos2, elem_ids, mesh)

    # k2
    v2 = interpolate_velocity_batch_gpu(pos2, elem2, mesh, vel_field)
    pos3 = positions + 0.5 * dt * v2
    elem3 = batched_search_hot(pos3, elem2, mesh)

    # k3
    v3 = interpolate_velocity_batch_gpu(pos3, elem3, mesh, vel_field)
    pos4 = positions + dt * v3
    elem4 = batched_search_hot(pos4, elem3, mesh)

    # k4
    v4 = interpolate_velocity_batch_gpu(pos4, elem4, mesh, vel_field)

    positions_final = positions + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
    elem_final = batched_search_hot(positions_final, elem_ids, mesh)

    return positions_final, elem_final
```

This yields:

- **One JIT‑compiled kernel per RK4 step**, containing:
  - vmap’d interpolation,  
  - vmap’d L0/L1/HOT search,  
  - all Morton/CSR logic.  
- Mesh + Morton data stay resident on GPU.  
- Per step, you only upload the velocity slice(s) and download final positions/IDs, as in your existing design.[1]

***

## Summary of key design properties

- **HOT‑like structure**:
  - Morton keys encode both spatial position and hierarchy, as in hashed oct‑trees.[3]
  - Octree is **linearized**: leaves are contiguous Morton segments with max capacity C.  
  - Instead of dynamic hash tables, you use CSR (start,length) and prefix→leaf lookup tables, which are static arrays ideal for JAX.[2]

- **GPU/JAX compatibility**:
  - All arrays have static shapes; candidate loops use fixed upper bounds and masking (`j < length`).  
  - Control flow uses `lax.fori_loop` and `lax.cond` under a single `jit`; there are no Python loops over device data.[9][1]
  - The search runs inside `vmap` over particles, fully fused with RK4.

- **Performance / OOM safety**:
  - Per‑particle L2/L3 complexity is capped at O(C) tests, independent of total elements.  
  - L0/L1 neighbors (which already give you >90% hits) run first and very cheaply, so HOT search is only needed for a small fraction of particles.[1]
  - No padded `(N_local × N_block_elems)` arrays are ever materialized; only `(N_particles × C)`‑scale work in the worst case, well within your 4 GB GPU budget.[2]

This document should serve as a clear roadmap: from preprocessing, to data structures, to JAX‑compliant kernels, to integration with your fused RK4 time marcher.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/6c89c190-617c-49ab-9d7d-a8d4e561d18c/PHASE3A_COMPLETE_WITH_FUSED_RK4.md)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/34b4b31a-8d9d-457e-846c-ff194850c63f/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/4e15f735-cb36-4cd6-b668-3faff2ebfda2/169627.169640.pdf)
[4](https://en.wikipedia.org/wiki/Z-order_curve)
[5](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/)
[6](https://d-nb.info/1217140409/34)
[7](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)
[8](https://github.com/google/jax/discussions/12441)
[9](https://docs.jax.dev/en/latest/gpu_performance_tips.html)
[10](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/0bf9dd0c-0a8a-448b-8817-636fb2c7ea69/BATCHED_BLOCKWISE_ARCHITECTURE.md)