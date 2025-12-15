Your revised plan is valid and fits well with the existing fused RK4 + L0/L1 infrastructure: start with a **single global HOT‑like structure** (no blocks), then later refine to blockwise if needed. The key is a Morton‑sorted global element list, cut into fixed‑capacity “leaves” (segments), and a JAX‑friendly L2 kernel that only scans one such segment per query, inside vmap and fused RK4.[1][2]

Below is a phase‑by‑phase roadmap, self‑contained, with core pseudocode.

***

## Phase 0 – Preconditions and goals

- Mesh: unstructured tetrahedral mesh (connectivity, node_positions).
- Existing GPU pieces to keep:
  - Fused RK4 step with vmap over all particles.[1]
  - L0: cached element point‑in‑tet.
  - L1: neighbor / multi‑hop neighbor search (face + extended neighbors).[1]
  - Velocity interpolation on GPU.
- New target:
  - Replace old L2 with a **global Morton + octree leaf L2**:
    - A single Morton‑sorted list of all elements.
    - Fixed‑capacity leaves (e.g. C = 128–256 elements/leaf).
    - On GPU, only two arrays are needed for L2:
      - `elem_ids_sorted` (global Morton order of elements).
      - `leaf_start` (offset of each leaf in that array; optionally `leaf_length`).
- Constraints:
  - L2 search must run **inside fused RK4**, under vmap, so:
    - No Python loops over particles.
    - No dynamic‑length slices; use fixed upper bound C and masks.
    - No padded `(N_particles × N_elems)` arrays to avoid OOM.[2]

***

## Phase 1 – Global Morton + leaf segmentation (CPU, once per mesh)

### 1.1 Compute Morton codes for elements

Choose global bbox \([x_{\min},x_{\max}]\times[y_{\min},y_{\max}]\times[z_{\min},z_{\max}]\) and max depth \(L\).

For each element \(e\):

1. Compute centroid \(c_e \in \mathbb{R}^3\).
2. Map to integer grid:

   \[
   u_x = \left\lfloor \frac{c_{e,x} - x_{\min}}{x_{\max}-x_{\min}} (2^L-1)\right\rfloor
   \]

   similarly \(u_y,u_z\), each in \([0,2^L-1]\).

3. Interleave bits to Morton code \(m_e\):

   \[
   m_e = \sum_{i=0}^{L-1} \bigl(x_i 2^{3i} + y_i 2^{3i+1} + z_i 2^{3i+2}\bigr)
   \]

   where \(x_i\) is bit \(i\) of \(u_x\), etc.[3][4]

Pseudocode:

```python
def morton_encode_point(p, bbox_min, bbox_max, L):
    scale = (2**L - 1) / (bbox_max - bbox_min)
    u = np.floor((p - bbox_min) * scale).astype(np.uint64)  # (3,)
    return interleave_bits_3d(u[0], u[1], u[2])             # uint64

morton = np.empty(n_elems, dtype=np.uint64)
for e in range(n_elems):
    nodes = connectivity[e]
    centroid = node_positions[nodes].mean(axis=0)
    morton[e] = morton_encode_point(centroid, bbox_min, bbox_max, L)
```

### 1.2 Global Morton sort and fixed‑capacity leaves

Let C be fixed (e.g. 128 or 256).

1. Sort element IDs by Morton:

```python
order = np.argsort(morton)
elem_ids_sorted = np.arange(n_elems, dtype=np.int32)[order]
morton_sorted   = morton[order]
```

2. Partition `elem_ids_sorted` into leaves of capacity C:

Simplest **phase‑1** version (uniform leaves):

- Leaf ℓ has:

  - `leaf_start[ℓ] = ℓ * C`
  - `leaf_length[ℓ] = min(C, n_elems - leaf_start[ℓ])`

- Number of leaves:

  \[
  N_\text{leaves} = \left\lceil\frac{n_\text{elems}}{C}\right\rceil.
  \]

This ignores geometric octree constraints (leaf boundaries may not align perfectly with spatial octants), but is sufficient as a first HOT‑like structure: you still only test ≤C elements per query.

Later phases can refine this to true octree leaves (prefix‑based segmentation), but uniform C‑sized chunks already give you *global Morton buckets* with bounded candidate counts.

3. Upload to GPU:

```python
mesh_gpu.elem_ids_sorted = jax.device_put(elem_ids_sorted.astype(np.int32))
mesh_gpu.leaf_start      = jax.device_put(leaf_start.astype(np.int32))
mesh_gpu.leaf_length     = jax.device_put(leaf_length.astype(np.int32))
mesh_gpu.n_leaves        = np.int32(N_leaves)
mesh_gpu.bbox_min        = jax.device_put(bbox_min.astype(np.float32))
mesh_gpu.bbox_max        = jax.device_put(bbox_max.astype(np.float32))
mesh_gpu.L               = L
mesh_gpu.C               = C
```

At this phase there is **no block structure**; the HOT structure is global over all elements.

***

## Phase 2 – Single‑position L2 search using global leaves (GPU)

### 2.1 Map position → Morton code → leaf index

For a position `pos: (3,)`:

1. Compute Morton code \(m(x)\) with the same transform as above (now in JAX).
2. Map it to a **leaf index**:

Phase‑1 simplest mapping:

- Use linear chunks: given sorted Morton array, approximate leaf index by:

  \[
  \ell = \left\lfloor \frac{\operatorname{rank}(m(x))}{C} \right\rfloor
  \]

- Since you do not know rank(m(x)) exactly, just take a **uniform mapping in Morton space**:

  - Normalize m(x) to  over key range:[5]

    \[
    t = \frac{m(x) - m_{\text{min}}}{m_{\text{max}} - m_{\text{min}}}
    \]

  - `leaf_id = clamp(int(t * N_leaves), 0, N_leaves-1)`.

This is not exact, but because both elements and queries are “uniformly” distributed along Morton order, it tends to send you to the correct bucket or a nearby one. You then just test up to C candidates in that bucket; if not found and you worry, you can look in `leaf_id±1` as a tiny extension.

Pseudocode (JAX):

```python
def morton_encode_point_jax(pos, bbox_min, bbox_max, L):
    scale = (2**L - 1.0) / (bbox_max - bbox_min)
    u = jnp.floor((pos - bbox_min) * scale).astype(jnp.uint32)
    return interleave_bits_3d_jax(u[0], u[1], u[2])  # uint64
```

```python
def leaf_id_for_position_global(pos, mesh_gpu):
    m = morton_encode_point_jax(pos, mesh_gpu.bbox_min, mesh_gpu.bbox_max, mesh_gpu.L)
    # Assume we precomputed morton range on CPU:
    m_min = mesh_gpu.morton_min
    m_max = mesh_gpu.morton_max
    t = (m - m_min) / (m_max - m_min + 1)
    approx = jnp.floor(t * mesh_gpu.n_leaves).astype(jnp.int32)
    return jnp.clip(approx, 0, mesh_gpu.n_leaves - 1)
```

Later phase: replace this heuristic mapping with a true octree/prefix mapping.

### 2.2 Fixed‑capacity candidate loop (no dynamic slices)

Given `leaf_id`, we compute:

- `start = leaf_start[leaf_id]`
- `length = leaf_length[leaf_id]` (≤ C)

We run a `lax.fori_loop` from `0` to `C`, masking by `j < length` and `found_elem == -1`.

```python
MAX_LEAF_ELEMS = C  # e.g. 128

def search_L2_global_single(pos,
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
            False
        )
        return jnp.where(inside & active, elem_id, found_elem)

    init = jnp.int32(-1)
    found_elem = lax.fori_loop(0, MAX_LEAF_ELEMS, body, init)
    return found_elem
```

This is fully jit/vmap‑safe: all shapes and loop bounds are static; only scalar masks depend on data.[6][7]

***

## Phase 3 – Integrate new global L2 into existing multi‑level search

You keep your current L0 and multi‑hop L1 (vectorized neighbors) exactly as in Phase‑3a.  L2 is replaced by a call to the new global Morton‑leaf kernel.[1]

### 3.1 Single‑particle multilevel search (L0+L1+L2 global)

```python
def multilevel_search_single_global(pos,
                                    cached_elem_id,
                                    mesh_gpu):
    elem_id = cached_elem_id

    # L0: cached element
    elem_id_L0 = search_L0_single(pos, elem_id,
                                  mesh_gpu.connectivity,
                                  mesh_gpu.node_positions)
    found = elem_id_L0 >= 0
    elem_id = jnp.where(found, elem_id_L0, elem_id)

    # L1: neighbor / multi-hop
    elem_id_L1 = search_L1_single(pos, elem_id,
                                  mesh_gpu.element_neighbors,
                                  mesh_gpu.connectivity,
                                  mesh_gpu.node_positions)
    improve_L1 = (elem_id_L1 >= 0) & (~found)
    elem_id = jnp.where(improve_L1, elem_id_L1, elem_id)
    found   = found | improve_L1

    # L2: global HOT leaf (if still unfound)
    def do_L2(elem_id, found):
        leaf_id = leaf_id_for_position_global(pos, mesh_gpu)
        elem_id_L2 = search_L2_global_single(
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
        return elem_id, found

    elem_id, found = do_L2(elem_id, found)

    return elem_id
```

### 3.2 Batched version for fused RK4

```python
@jax.jit
def multilevel_search_batch_global(positions,
                                   cached_elem_ids,
                                   mesh_gpu):
    search_one = lambda p, e: multilevel_search_single_global(p, e, mesh_gpu)
    elem_ids_new = jax.vmap(search_one)(positions, cached_elem_ids)
    return elem_ids_new
```

Now you have a JAX‑native, fully GPU, global L0+L1+L2 search with bounded work per particle.

***

## Phase 4 – Use new L2 inside fused RK4 (no change to RK4 structure)

Your existing fused RK4 already has the right skeleton: positions and element IDs live on GPU, and each stage calls search + interpolation; only start/end states are transferred.[1]

You just swap in `multilevel_search_batch_global` wherever L2 was previously used.

Sketch:

```python
@jax.jit
def rk4_step_gpu_fused_globalL2(positions_initial,
                                elem_ids_initial,
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
    elem2 = multilevel_search_batch_global(pos2, elem_ids_initial, mesh_gpu)
    v2 = interpolate_velocity_batch_gpu(pos2, elem2,
                                        mesh_gpu.connectivity,
                                        mesh_gpu.node_positions,
                                        velocity_field_gpu)

    # Stage 3
    pos3 = positions_initial + 0.5 * dt * v2
    elem3 = multilevel_search_batch_global(pos3, elem2, mesh_gpu)
    v3 = interpolate_velocity_batch_gpu(pos3, elem3,
                                        mesh_gpu.connectivity,
                                        mesh_gpu.node_positions,
                                        velocity_field_gpu)

    # Stage 4
    pos4 = positions_initial + dt * v3
    elem4 = multilevel_search_batch_global(pos4, elem3, mesh_gpu)
    v4 = interpolate_velocity_batch_gpu(pos4, elem4,
                                        mesh_gpu.connectivity,
                                        mesh_gpu.node_positions,
                                        velocity_field_gpu)

    positions_final = positions_initial + (dt/6.0) * (v1 + 2*v2 + 2*v3 + v4)

    elem_final = multilevel_search_batch_global(positions_final,
                                                elem_ids_initial,
                                                mesh_gpu)
    return positions_final, elem_final
```

The CPU wrapper and velocity‑field streaming remain as in your current Phase‑3a implementation.[1]

***

## Phase 5 – Later refinements (optional roadmap)

Once Phase 1–4 are working and profiled, refine L2:

- Replace “uniform Morton buckets” by true **octree leaves**:
  - Build an adaptive octree so each leaf holds ≤C elements and aligns with spatial octants.
  - For each leaf, store `(start,length)` into `elem_ids_sorted`.
  - Replace `leaf_id_for_position_global` with a proper prefix‑based tree walk (using bits of Morton code) or a small prefix→leaf table.

- Introduce **blocks** only if needed:
  - If global HOT search still too expensive in refined regions, split the Morton array into block ranges (e.g. by cube grid indices), and keep separate `[start,length]` ranges per block; extend multilevel search to use cached block IDs, as in your previous batched‑blockwise plans.[8]

These refinements do not change the **JAX pattern**: vmap over particles, fixed‑bound `lax.fori_loop` over ≤C candidates, L0+L1+L2 fused inside RK4, and a single upload/download per step.

***

This phased roadmap lets you:  
1) implement a simple, global HOT‑like L2 without block complexity,  
2) integrate it cleanly into the existing fused RK4 and L0/L1, and  
3) later evolve it into a full octree / blockwise HOT if profiling shows the need.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/6c89c190-617c-49ab-9d7d-a8d4e561d18c/PHASE3A_COMPLETE_WITH_FUSED_RK4.md)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/ae3f023f-86b3-4bd3-a000-6d1ade3f7760/SEARCH_OPTIMIZATION_ANALYSIS.md)
[3](https://en.wikipedia.org/wiki/Z-order_curve)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/4e15f735-cb36-4cd6-b668-3faff2ebfda2/169627.169640.pdf)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/70091334/0cbeff7a-7641-43ad-abe1-d2a6f497ef3c/image.jpg)
[6](https://docs.jax.dev/en/latest/gpu_performance_tips.html)
[7](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)
[8](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/34b4b31a-8d9d-457e-846c-ff194850c63f/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md)