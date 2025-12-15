The revised roadmap is to introduce the HOT‑like L2 search in **phases**, starting from the simplest global Morton‑sorted structure (no blocks) and progressively adding sophistication only if needed. The fused RK4, interpolation, and L0/L1 (cached + multi‑hop neighbors) remain as in your current Phase‑3a implementation; only L2 is redesigned.[1]

***

## Phase 0 – Keep existing fused RK4 + L0/L1

Goal: preserve what works.

- Use your current:
  - GPU‑resident interpolation (`interpolate_velocity_batch_gpu`).[1]
  - Fused RK4 (`rk4_step_gpu_fused`).[1]
  - L0 (cached element) + extended L1 (multi‑hop neighbors) search on GPU.[2][1]
- Only change what happens after L0/L1 fail (L2).

This isolates the new HOT‑like search and reduces debugging complexity.

***

## Phase 1 – Global Morton‑sorted L2 without blocks

### 1.1 Preprocessing (CPU)

No blocks in this phase. You treat the mesh as one big block.

1. **Compute Morton codes for element centroids**

- For each tet \(e\):

  \[
  c_e = \frac{1}{4} \sum_{i=1}^4 x_{e,i}
  \]

  where \(x_{e,i}\) are node coordinates.

- Map \(c_e\) to integer grid \((u_x,u_y,u_z)\) with depth \(L\) and compute Morton key \(m_e\) by bit‑interleaving.[3][4]

2. **Sort elements by Morton code**

```python
m_codes = morton_encode_batch(centroids, bbox_min, bbox_max, L)  # (n_elems,)
order   = np.argsort(m_codes)
keys_sorted     = m_codes[order]        # uint64
elem_ids_sorted = np.arange(n_elems, dtype=np.int32)[order]
```

3. **Partition sorted list into fixed‑size leaves**

- Choose leaf capacity \(C\) (e.g. 128 or 256 elements).
- Define number of leaves:

  \[
  N_\text{leaves} = \left\lceil \frac{N_\text{elems}}{C} \right\rceil.
  \]

- For leaf \(\ell\):

  - `start_ℓ = ℓ * C`
  - `length_ℓ = min(C, N_elems - start_ℓ)`

This is a **very simple “linear octree”**: not strictly geometric yet, but gives you fixed segments of the Morton curve with bounded size.

Data to upload to GPU:

```python
elem_ids_sorted_gpu = device_put(elem_ids_sorted)     # (N_elems,)
leaf_start_gpu       = device_put(start_array)        # (N_leaves,)
leaf_length_gpu      = device_put(length_array)       # (N_leaves,)
# Also bbox_min, bbox_max, L for morton encoding
```

### 1.2 Mapping a position to a leaf index

With this first phase you do **not** use a full geometric octree. You start with a crude but easy mapping:

- Compute Morton code \(m(x)\).
- Map to leaf index by proportionality along the Morton axis:

  \[
  \ell = \left\lfloor \frac{m(x)}{m_\text{max}+1} \cdot N_\text{leaves} \right\rfloor,
  \]

  where \(m_\text{max}\) is the maximum Morton code over elements.

In code (GPU/JAX):

```python
def morton_encode_point(pos, bbox_min, bbox_max, L):
    scale = (2**L - 1) / (bbox_max - bbox_min)
    u = jnp.floor((pos - bbox_min) * scale).astype(jnp.uint32)  # (3,)
    return interleave_bits_3d(u[0], u[1], u[2])  # uint64

def leaf_index_from_morton_global(m, m_min, m_max, n_leaves):
    # Normalize m to [0,1], then scale
    t = (m - m_min) / (m_max - m_min + 1)
    idx = jnp.floor(t * n_leaves).astype(jnp.int32)
    return jnp.clip(idx, 0, n_leaves - 1)
```

This is **not perfectly geometric**, but because the elements are in Morton order, nearby spatial cells tend to land in nearby leaves, and each leaf holds at most \(C\) candidates. You can refine the mapping in later phases.

### 1.3 L2 search kernel (single particle, single leaf)

Same pattern as before, but now `leaf_id` is global and there is no block:

```python
MAX_LEAF_ELEMS = C

def search_in_leaf_single_global(pos,
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
        idx    = start + j
        elem_id = jnp.where(active, elem_ids_sorted[idx], 0)
        inside = jnp.where(
            active,
            point_in_tet(pos, elem_id, connectivity, node_positions),
            False,
        )
        return jnp.where(inside & active, elem_id, found_elem)

    init = jnp.int32(-1)
    return lax.fori_loop(0, MAX_LEAF_ELEMS, body, init)
```

### 1.4 Integrate into multilevel search (replacing old L2)

For each particle:

- Run existing L0/L1 (unchanged).
- If still not found:

```python
def L2_global_morton_single(pos, elem_id, found,
                            elem_ids_sorted, leaf_start, leaf_length,
                            conn, nodes, bbox_min, bbox_max, L, m_min, m_max, n_leaves):
    m = morton_encode_point(pos, bbox_min, bbox_max, L)
    leaf_id = leaf_index_from_morton_global(m, m_min, m_max, n_leaves)
    elem_L2 = search_in_leaf_single_global(
        pos, leaf_id,
        elem_ids_sorted, leaf_start, leaf_length,
        conn, nodes
    )
    improve = (elem_L2 >= 0) & (~found)
    elem_id = jnp.where(improve, elem_L2, elem_id)
    found   = found | improve
    return elem_id, found
```

Vectorize over particles with `vmap` and plug this into your existing fused RK4 where L2 used to call the block‑based fallback.[1]

This already gives you:

- Bounded candidate count (\(≤C\)) per particle for L2.
- Static shapes and JAX‑friendly loops.
- No blocks and no complex per‑block data, which keeps the first implementation simple.

***

## Phase 2 – Better leaves: true octree alignment (still global)

Once Phase 1 works, you can make leaves better aligned with real octree geometry, still without blocks.

### 2.1 Replace naive equal‑size segments by geometric leaves

- Build a global octree (on CPU) based on mesh refinement or cube structure.  
- For each **leaf \(\ell\)** of that octree:
  - Compute its Morton prefix \(P_\ell\) (length \(3d_\ell\) bits).  
  - Compute the Morton key interval for that leaf.  
  - In the sorted `keys_sorted`, find the minimal and maximal indices satisfying that interval, and store them as `(start_ℓ, length_ℓ)`.  

This changes only how `leaf_start` and `leaf_length` are built; the GPU code is identical, except `leaf_index_from_morton_global` becomes:

```python
def leaf_index_from_morton_prefix(m, leaf_prefixes, leaf_masks):
    # leaf_prefixes: (N_leaves,) Morton prefix values
    # leaf_masks:    (N_leaves,) masks to isolate prefix bits
    # Simplest is a small search over leaves, but you can build
    # a prefix->leaf lookup table to avoid O(N_leaves) scans.
    ...
```

In a first geometric version you can tolerate a small `fori_loop` over all leaves for the rare L2 subset; later you introduce a prefix lookup table if needed.

The important point is: each leaf still has ≤C elements, and you still use `(start,length)` CSR segments; only the mapping `m(x) → leaf_id` changes.

***

## Phase 3 – Refined mapping: prefix buckets / simple hash

When geometric leaves are in place, you refine the mapping from Morton key to leaf:

- Choose a prefix length \(B\) bits (e.g. 10–12) for a simple, static lookup.
- Precompute on CPU:

  ```python
  # For each possible prefix p in [0, 2^B):
  #   find which leaf(s) cover that prefix, pick the most specific leaf ID
  block_prefix_to_leaf[p] = leaf_id or -1
  ```

- Upload `prefix_to_leaf` to GPU.
- At query:

  ```python
  m = morton_encode_point(pos, bbox_min, bbox_max, L)
  prefix = (m >> (3*L - B)) & ((1<<B) - 1)
  leaf_id = prefix_to_leaf[prefix]
  ```

If `leaf_id == -1` (rare for adaptive meshes if B is large enough), you can either:

- Do a small fallback scan over neighboring prefixes, or
- Fall back to a slower L2 path (e.g. neighbor L0/L1 or global brute force for that 0.1%).

This is where your design becomes clearly HOT‑like: **Morton key + simple hash/prefix table → leaf segment**, no pointer chasing.[4]

All GPU code for L2 still uses the same `search_in_leaf_single_global`.

***

## Phase 4 – Optional: introduce blocks for memory/performance

Only if profiling shows you need more locality or VRAM savings, you introduce **per‑block variants** of the Phase‑2/3 structure:

- Partition elements into blocks aligned with cube grid as described earlier.[2]
- For each block, build its own Morton‑sorted `elem_ids_sorted_b`, leaf ranges, and prefix→leaf table.
- Concatenate block arrays into global arrays with CSR offsets; add `elem_to_block` so you can find the block from cached element.
- L2 then becomes:
  - Use block from cached element (or from position),
  - Use block‑local prefix table to map `m(x)` → `leaf_id`,
  - Search in that leaf.

This is a direct extension of the Phase‑1/2/3 global scheme and can be postponed until the simpler global scheme is validated.

***

## How this plugs into fused RK4

Throughout all phases:

- The **incremental search** continues to do:
  - L0 cached element.
  - L1 neighbor / multi‑hop on GPU (unchanged).[2][1]
  - New **L2 Morton‑leaf search** as described above.
- The fused RK4 (`rk4_step_gpu_fused`) calls this incremental search batch at each substage for all particles; this part of your code remains structurally the same.[1]

In pseudocode at Phase 1:

```python
def multilevel_search_single_phase1(pos, cached_elem_id, mesh_gpu):
    elem_id = cached_elem_id
    found   = elem_id >= 0

    # L0: cached
    elem_id_L0 = search_L0_single(...)
    improve_L0 = (elem_id_L0 >= 0)
    elem_id = jnp.where(improve_L0, elem_id_L0, elem_id)
    found   = found | improve_L0

    # L1: neighbors
    elem_id_L1 = search_L1_single(...)
    improve_L1 = (elem_id_L1 >= 0) & (~found)
    elem_id    = jnp.where(improve_L1, elem_id_L1, elem_id)
    found      = found | improve_L1

    # L2: HOT-like global Morton leaf
    elem_id, found = L2_global_morton_single(
        pos, elem_id, found,
        mesh_gpu.elem_ids_sorted,
        mesh_gpu.leaf_start,
        mesh_gpu.leaf_length,
        mesh_gpu.connectivity,
        mesh_gpu.node_positions,
        mesh_gpu.bbox_min,
        mesh_gpu.bbox_max,
        mesh_gpu.L,
        mesh_gpu.m_min,
        mesh_gpu.m_max,
        mesh_gpu.n_leaves
    )

    return elem_id
```

Vectorize with `vmap` and call from RK4 as in your current fused implementation; you only changed what happens inside L2.

***

## Summary of the phased plan

- **Phase 0:** Keep fused RK4, interpolation, and vectorized L0/L1 exactly as now; isolate changes to L2.[1]
- **Phase 1 (no blocks):**  
  - Global Morton codes for element centroids.  
  - Sort once; partition sorted list into fixed‑size segments (leaves) of capacity \(C\).  
  - Upload `elem_ids_sorted`, `leaf_start`, `leaf_length`.  
  - L2: `m(x)` → approximate leaf index by linear mapping → search bounded segment.  
- **Phase 2 (geometric leaves):**  
  - Replace equal‑size segments by true octree leaves; each leaf still ≤C elements.  
  - Precompute `[start_ℓ,length_ℓ]` per leaf from Morton ranges.  
  - Keep same L2 pattern; only improve leaf construction.  
- **Phase 3 (prefix/hash mapping):**  
  - Precompute global `prefix_to_leaf` table; map `m(x)` → `leaf_id` via high‑bit prefix.  
  - L2 becomes: Morton encode → prefix → leaf → `search_in_leaf_single_global`.  
- **Phase 4 (optional):**  
  - Add cube‑aligned blocks and per‑block Morton/octree leaves for better locality and memory control.

This roadmap gives you a clear incremental path: you can first implement and validate the **simplest HOT‑like global L2** (Phase 1) inside your existing fused RK4, then progressively move to full octree‑aligned leaves and, only if needed, blockwise refinement.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/6c89c190-617c-49ab-9d7d-a8d4e561d18c/PHASE3A_COMPLETE_WITH_FUSED_RK4.md)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/34b4b31a-8d9d-457e-846c-ff194850c63f/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md)
[3](https://en.wikipedia.org/wiki/Z-order_curve)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/4e15f735-cb36-4cd6-b668-3faff2ebfda2/169627.169640.pdf)