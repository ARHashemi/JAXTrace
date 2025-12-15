The best design is: coarse global blocks for spatial indexing; recursively split only heavy blocks into per‑block octrees until each leaf holds a bounded number of tets; keep elements in Morton order and expose per‑leaf/per‑bucket candidate ranges (CSR, not padded); and run a single fused JAX kernel per RK4 stage that performs L0/L1/multi‑hop, then masked L2/L3 queries against the local octree/buckets, with one upload and one download per time step to avoid PCIe ping‑pong. This preserves your proven gains from vectorized L0/L1 and fused RK4, and fixes the L2 bottleneck/OOM risk without re‑introducing padded arrays or CPU orchestration.[1][2][3]

## Heavy blocks: balancing and splitting
- Identify heavy blocks (e.g., >50k tets) and split them by longest‑axis bisection until the per‑block element count falls under a limit; this mitigates extreme skew like “4 blocks contain 91% of elements” and prevents single‑kernel OOM pressure.[3]
- Keep “light block batching” for launch efficiency (merge multiple small blocks per launch), and enable chunked particle processing only as an automatic fallback when peak memory for a heavy block is exceeded.[3]
- Remove padded arrays entirely; they were responsible for GB‑scale transfers and >99% time in search in earlier builds.[2]

## Morton/hash inside each block (compact, GPU‑friendly)
- For each block, compute Morton codes of tet centroids, sort element IDs by Morton, and build either:
  - a flat octree whose leaves store [start,end) ranges into the Morton‑sorted element array, or
  - a fixed Morton “hash” with CSR ranges per bucket (no per‑bucket padding).[3]
- With bounded leaf/bucket occupancy (e.g., 64–128 tets), L2/L3 work per particle stays O(depth + leaf_size) and avoids the O(N_particles × N_block_elems) intermediates that caused OOM.[3]
- This keeps all arrays flat with static shapes and lets inner loops be small `lax.fori_loop`s, which JAX/XLA fuses well.[3]

## End‑to‑end design and pseudocode
- Notes: comments describe CPU vs GPU residence; all mesh arrays are uploaded once per mesh snapshot; velocity fields are streamed per step; one fused kernel per RK4 step to minimize transfers. The structure matches your Phase‑3a fused RK4, extended to include bounded L2/L3.[1]

Initialization (CPU)
- Build neighbors and coarse blocks; split heavy blocks recursively.
- Per block, compute Morton codes, sort element IDs, build either octree leaves or CSR bucket ranges.
- Upload to GPU: connectivity, node_positions, element_neighbors, element_to_block, block metadata, Morton‑sorted element arrays, and per‑block leaf/bucket structures.[3]

Pseudocode (CPU preprocessing)
- Build blocks and split heavy ones:
  - element_to_block = assign_by_centroid(connectivity, node_positions, grid_size)
  - while any(block_count > max_elems): split_by_longest_axis(block)  # updates element_to_block[3]
- Build per‑block Morton/CSR or octree:
  - centroids_B = centroids(elems_B)
  - morton_B    = morton_encode(centroids_B, bbox_B)
  - sorted_ids_B = argsort(morton_B); elems_sorted_B = elems_B[sorted_ids_B]
  - either:
    - octree_B = build_flat_octree_over_bbox(elems_sorted_B, morton_B, max_leaf_elems)
    - or csr_B = build_csr_buckets_from_morton(elems_sorted_B, morton_B, n_buckets, max_bucket_elems)[3]

Initial assignment (GPU)
- Assign particles to coarse blocks by position; per block, query its octree/buckets to find containing tet; for rare misses, try neighbor blocks.

Fused RK4 step (GPU)
- One JIT per step; no CPU filtering; masks handle the hierarchy.[1]

Pseudocode (GPU kernels; schematic)
- multilevel_search_batch(positions, cached_elem, mesh):
  - L0: elem = search_L0_point_in_tet(positions, cached_elem)
  - L1: elem1 = search_L1_neighbors_masked(positions, elem, element_neighbors); elem = where(hit1 & ~found, elem1, elem)
  - L2: elem2 = search_L2_block_octree_masked(positions, elem, block_structs, morton_sorted_ids, leaf_ranges_or_csr); elem = where(hit2 & ~found, elem2, elem)
  - L3: elem3 = search_L3_neighbor_blocks_masked(positions, elem, neighbor_block_structs); elem = where(hit3 & ~found, elem3, elem)
  - return elem
- interpolate_velocity_batch_gpu(positions, elem, connectivity, node_positions, vel_field)  # as in Phase‑3a[1]
- rk4_step_gpu_fused(positions, elem, dt, mesh, vel_fields):
  - v1 = interpolate(positions, elem, …)
  - pos2, elem2 = advect_and_search(positions, elem, v1, 0.5*dt)
  - v2 = interpolate(pos2, elem2, …)
  - pos3, elem3 = advect_and_search(positions, elem2, v2, 0.5*dt)
  - v3 = interpolate(pos3, elem3, …)
  - pos4, elem4 = advect_and_search(positions, elem3, v3, 1.0*dt)
  - positions_final = positions + dt/6*(v1 + 2*v2 + 2*v3 + v4)
  - elem_final = multilevel_search_batch(positions_final, elem, mesh)
  - return positions_final, elem_final[1]
- advect_and_search(positions, elem, v, alpha_dt):
  - pos_new = positions + alpha_dt * v
  - elem_new = multilevel_search_batch(pos_new, elem, mesh)
  - return pos_new, elem_new

Inner masked L2/L3 (GPU; bounded per‑particle loop)
- search_L2_block_octree_masked(positions, elem, block_structs, morton_sorted_ids, leaf_ranges):
  - block_id = element_to_block_masked(elem, positions)  # cached or recomputed
  - leaf = descend_flat_octree(block_id, positions)  # fixed small depth loop
  - [s,e) = leaf_ranges[leaf]; candidates = morton_sorted_ids[s:e]
  - for j in range(max_leaf_elems): if j < (e-s): test point‑in‑tet(candidates[j])
  - return found_elem_or_minus_one
- The CSR‑bucket variant replaces leaf_ranges with bucket_ranges and an index computation from Morton(positions).

Time marching (CPU wrapper)
- For each step:
  - Upload only current velocity time slice(s); mesh stays resident.[1]
  - Call rk4_step_gpu_fused once; download final positions and element_ids once; collect stats.[1]

## Transfers, performance, and OOM guards
- Keep all intermediates on GPU; your Phase‑3a changes cut per‑stage transfers from ~55 MB to ~1 MB and raised GPU utilization from ~2% to ~60–80%, which this design preserves since L2/L3 now operate on bounded candidates entirely on device.[1]
- Remove padded block arrays completely; they were the main cause of 6.5 GB host allocation and multi‑GB transient GPU uploads during L2/L3.[2]
- Safeguards:
  - Adaptive block splitting thresholds; mandatory hash/CSR for any block >10k elems; runtime VRAM checks and chunked particle processing fallback for pathological cases.[3]
  - Batch multiple light blocks per launch to amortize overhead.[3]

This integrated design retains your validated strengths (vectorized L0/L1, fused RK4, minimal transfers) and fixes the deep‑search bottleneck by bounding L2/L3 work with per‑block octrees or Morton‑CSR buckets, thereby maximizing throughput while preventing OOM.[2][1][3]

[1](PHASE3A_COMPLETE_WITH_FUSED_RK4.md)
[2](SEARCH_OPTIMIZATION_ANALYSIS.md)
[3](BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md)


# One of the problems that causes error, is how to build coarse blocks. The current mesh, as I explained previously, is a tetrahedral, which means I have cubic blocks divided to 4 right-angled tetrahedrons. In the refined regions, in octree hierarchical way, the cubic blocks divided to 4 consecutively, until required level of refinement, then each refined block divided to tetrahedrons. With geometrical divisions, we faced an issue when the blocks' boundaries cross elements, cause shared elements between blocks, leads to huge block information arrays and confusion of search algorithm. Is it beneficial/possible to have the coarse divisions exactly aligned with mesh grids (cubes contains 4 tetrahedrons)? How to do it in detail?

Yes, aligning coarse blocks exactly with the existing cubic “grid” of 4-tet cubes is both beneficial and possible, and it directly fixes the main problem you describe: *blocks whose boundaries cut through elements*.

Let’s walk through what that means, how to do it, and why it helps for GPU/JAX.

***

## 1. Why your current geometric coarse blocks cause trouble

Right now you’re doing something like:

- Define a coarse Cartesian block grid over the domain (e.g. Nx×Ny×Nz).
- For each element, assign it to all blocks whose bounding boxes overlap the element.

This leads to:

- **Shared elements between many blocks** when blocks cut through tets.
- Large, redundant `block_elements` arrays.
- Confusing logic: a particle in an element could in principle “belong” to multiple blocks, which complicates L2/L3 search and block-based hashing.

For your mesh, that’s unnecessary, because:

> Your mesh is built from **cubes that are always split into exactly 4 right-angled tets** at each refinement level.

So there is a natural “grid cell” structure underneath: the tet mesh is just a subdivision of underlying cubes.

***

## 2. Key idea: Coarse blocks aligned to cube grid

Instead of arbitrary geometric blocks, **define blocks as unions of entire cubes**, never cutting through cubes. Each cube contains 4 tets; these tets all belong to the same block.

This guarantees:

- No tet is ever shared between blocks.
- Block boundaries coincide with cube faces, which are also mesh faces.
- Element-to-block mapping becomes **one-to-one** and trivial.

### Benefits

- `element_to_block[e]` is unique for each element.
- `block_elements[b]` is a simple list; no duplication.
- L2/L3 search becomes “search only the block that contains the cube containing the tet.”
- You can still have multiple blocks along the domain (for load balancing) but you never cut through tets.

***

## 3. How to build cube-aligned coarse blocks in detail

Assume your mesh was generated from an octree-like cubic hierarchy:

- Level 0: root cube(s).
- Refinement: each cube subdivided into 8 children (standard octree) or 4 along specific axes (your “4 cubes” language).
- Each leaf cube is then split into 4 right-angled tets.

### Step 1: Recover/define cube indices for each element

If your mesh generator can give you a cube ID (or (i,j,k,level)) per tet, use that. If not, you can reconstruct it:

- For each tet:
  - Take its centroid.
  - Based on global cube grid spacing and refinement pattern, find which *cube* it belongs to.
  - That cube is **the unique owner** of that tet.

Represent cube IDs as integer triples (i,j,k,level) or compressed into a Morton code.

### Step 2: Define coarse blocks as ranges of cube indices

Now define blocks not in continuous space, but in **cube index space**:

- For example, choose coarse block sizes in cube-grid coordinates:
  - `block_size_i`, `block_size_j`, `block_size_k`.
- For a leaf cube at index `(ci, cj, ck, level)`:
  - Compute “coarse block index”:
    - `bi = ci // block_size_i`
    - `bj = cj // block_size_j`
    - `bk = ck // block_size_k`
  - `block_id = encode(bi, bj, bk)`.

All 4 tets belonging to that cube inherit this `block_id`. No tet crosses blocks, by construction.

> Even in refined regions, each tet belongs to exactly one leaf cube and thus to exactly one block.

### Step 3: Build block→element lists

Once you have `element_to_block[e]` for all elements:

```python
# CPU
block_elements = [[] for _ in range(n_blocks)]
for e in range(n_elements):
    b = element_to_block[e]
    block_elements[b].append(e)

# Optionally pad or convert to CSR ranges before uploading to GPU.
```

***

## 4. Handling refinement hierarchy (octree) within each block

Within each block:

- You can build a **per-block octree** or Morton buckets based on the cubes belonging to that block.
- Because each block is a union of whole cubes:
  - The per-block octree just reuses the same cubic subdivision you already have from the AMR/octree, but restricted to that block’s subdomain.
  - Per-block Morton codes can be computed for cube centers or element centers within that block.

No cube is split across blocks, so:

- A per-block octree never needs to reference elements outside that block.
- L2/L3 search becomes clean: once you know the block, you never worry about crossing block boundaries at L2 (only at L3, for very rare cross-block moves).

***

## 5. Efficiency & GPU/JAX compatibility

### Memory

- **No duplicate elements across blocks** → smaller `block_elements` / hash structures.
- Per-block Morton arrays / octree leaves scale **linearly** with elements; no inflated duplication.

### Search complexity

- L0/L1 still very cheap (cached + neighbor tets).
- L2:
  - For each particle:
    - block_id known from cached element or position→cube→block.
    - Use per-block octree/Morton leaf to get a **small candidate range** (e.g. 32–128 tets).
  - Complexity per particle: O(depth + leaf_size), independent of total elems per block.
- L3:
  - If not found, search a few neighbor blocks (26 or fewer) with the same per-block routine.

### JAX/GPU

- All block structures are **flat arrays** with static shapes (possibly padded or CSR).
- For each kernel:
  - `vmap` over particles, `lax.fori_loop` over small candidate ranges.
- No need for massive padded `(N_blocks, max_elems_per_block)` structures; you can use CSR (block_offsets + elements) and keep per-leaf ranges for candidates.

***

## 6. Example CPU preprocessing pseudocode

```python
def build_cube_aligned_blocks(connectivity, node_positions, cube_grid):
    # cube_grid encodes how the domain is discretized into cubes (i,j,k,level)

    n_elems = connectivity.shape[0]
    elem_to_cube = np.empty(n_elems, dtype=int)

    for e in range(n_elems):
        tet_nodes = connectivity[e]            # (4,)
        coords    = node_positions[tet_nodes]  # (4,3)
        centroid  = coords.mean(axis=0)

        cube_id   = find_leaf_cube_for_point(centroid, cube_grid)
        elem_to_cube[e] = cube_id

    # Map cubes to coarse blocks (bi, bj, bk)
    cube_to_block = {}
    for cube_id in unique(elem_to_cube):
        ci, cj, ck, level = decode_cube_id(cube_id)
        bi = ci // block_size_i
        bj = cj // block_size_j
        bk = ck // block_size_k
        block_id = encode_block_id(bi, bj, bk)
        cube_to_block[cube_id] = block_id

    elem_to_block = np.empty(n_elems, dtype=int)
    for e in range(n_elems):
        elem_to_block[e] = cube_to_block[elem_to_cube[e]]

    # Build block_elements
    n_blocks = 1 + max(elem_to_block)
    block_elements = [[] for _ in range(n_blocks)]
    for e, b in enumerate(elem_to_block):
        block_elements[b].append(e)

    # Convert block_elements to CSR / padded arrays as needed
    block_offsets = np.zeros(n_blocks+1, dtype=int)
    for b in range(n_blocks):
        block_offsets[b+1] = block_offsets[b] + len(block_elements[b])

    block_elems_flat = np.empty(block_offsets[-1], dtype=int)
    for b in range(n_blocks):
        start = block_offsets[b]
        end   = block_offsets[b+1]
        block_elems_flat[start:end] = block_elements[b]

    return elem_to_cube, elem_to_block, block_offsets, block_elems_flat
```

***

## 7. Integrating Morton/hash

Now, for each block `b`:

```python
start = block_offsets[b]
end   = block_offsets[b+1]
elems_b = block_elems_flat[start:end]

centroids_b = compute_centroids(connectivity[elems_b], node_positions)
morton_b    = compute_morton(centroids_b, bbox_of_block(b))

sorted_idx  = np.argsort(morton_b)
elems_sorted_b  = elems_b[sorted_idx]
morton_sorted_b = morton_b[sorted_idx]

# Build CSR buckets or octree leaves over the Morton axis
bucket_ranges_b = build_bucket_ranges(morton_sorted_b, max_bucket_elems)
# Store elems_sorted_b and bucket_ranges_b in global arrays for GPU.
```

Because the blocks are cube-aligned, you *know* the bounding box and the refinement pattern, so mapping positions to cubes (and thus to blocks) is reliable and cheap.

***

## 8. Summary

- Your intuition is correct: **yes**, it is beneficial to have coarse blocks aligned exactly with the underlying cube grid that generates the 4-tet cells.
- This solves the “shared element between blocks” problem and greatly simplifies/economizes block data structures.
- The **best design** for your code is:

  1. Define coarse blocks as unions of whole cubes, never cutting cubes.
  2. Map each tet to exactly one cube → exactly one block.
  3. Within each block, build an octree/Morton-based structure (CSR buckets) over its tets.
  4. Use this for L2/L3 search in your fused JAX GPU kernel.

This yields:

- Clean element/block ownership,
- Small candidate lists at L2/L3,
- Flat, static arrays compatible with JAX/XLA, and
- Avoids the OOM and confusion that came from geometric block boundaries slicing through elements.

[1](BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md)
[2](STRATEGY3_CRITICAL_EVALUATION.md)