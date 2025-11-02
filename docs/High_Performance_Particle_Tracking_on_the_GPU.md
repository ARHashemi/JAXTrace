<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# High-Performance Particle Tracking on the GPU:

## A Comprehensive Forest-of-Octrees Implementation Guide


***

## I. Overview and Goals

- **Objective:** Achieve efficient, scalable, and robust particle tracking using forest-of-octrees mesh refinement on GPU architectures, supporting 500K+ particles and advanced RK4 integration in adaptive domains.
- **Core principles:**
    - Leverage spatial batching for maximal memory locality and throughput.
    - Avoid hierarchical pointer traversals in kernels; flatten all data.
    - Maintain scalably synchronizable state for AMR (adaptive mesh refinement), particles, and time marching.
    - Explicitly address memory- and compilation-related bottlenecks (e.g., JAX/XLA, CUDA).

***

## II. Critical Pre-initialization

### Workflow Overview Diagram

```
+----------------+
| Initialization |
+-------+--------+
        |
        v
+-------+--------+         +-------------------+
| Mesh Import or |-------> | AMR + Octree      |
| Generation     |         | Construction      |
+----------------+         +-------------------+
        |                            |
        v                            v
+---------------+         +-------------------------+
| Initial       |<------  | Forest-of-octrees Block |
| Particle Seeding |      | Partitioning            |
+----------------+        +------------------------+
        |
        v
+-------------------+
| Setup Batching by |
| Octree Block      |
+-------------------+
```


### Main Steps

**1. Mesh Construction and AMR:**

- Start from either a pre-generated mesh or build mesh with AMR/refinement rules discussed (cutting at the interface, “layers \& levels” criterion, 2:1 balance, etc.).

**2. Forest-of-Octrees Partitioning:**

- Decompose global octree at a coarse level: each block becomes root of a sub-octree; each will be maintained as an independently updatable, flat array.

**3. Precompute Mesh and Block Data Structures:**

- For each block:
    - Arrays of cell centroids, half-sizes, element neighbors, neighbor block IDs, flat lists of element indices.
    - Ghost/halo regions for boundaries (with room for inter-block exchange).
- Build lookup tables: block bounding boxes, Morton code <-> block index, block <-> GPU assignment.

**4. Initial Particle Assignment:**

- Seed particles into the domain; assign to starting elements and blocks using spatial search or precomputed mapping.
- Cache the current element ID, block ID, and (optionally) Morton code in each particle’s data structure.

**5. GPU Data Upload:**

- All block data, mesh, and particles transferred to GPU as contiguous, flat arrays.

***

## III. Time Marching and Parallelism

### Flowchart (Summary)

```
for t in range(N_steps):
    for block in blocks (in parallel!):
        for particle in block (in parallel!):
            - local search, RK4, update
    re-batch particles by block
    ghost exchange + mesh AMR (if needed)
    [output as needed]
```

(In JAX: outer scan for time, vmap per block, vmap per particle!)

***

### Key Implementation Strategy

- **Outer "loop":** time marching; in JAX use `lax.scan`, in CUDA one host loop.
- **Blocks:** use vmap over blocks (JAX) or multiple thread blocks (CUDA).
- **Particles:** use vmap per particle (JAX) or assign one thread per particle (CUDA).
- **No explicit for over all particles or blocks inside GPU kernel.**

***

## IV. Element Search and Efficient Particle Update

### Pseudocode (“Kernel Skeleton” for block)

```pseudocode
def update_particles_block(particles, mesh, field, block_id):
    for p in particles (parallel!):
        pos, elem_id = p.position, p.element_id
        # 1. Fast check: still inside previous element?
        if point_in_element(pos, mesh[elem_id]):
            pass
        # 2. Try neighbors (usually <8)
        else:
            found = False
            for nbr_id in mesh[elem_id].neighbors:
                if point_in_element(pos, mesh[nbr_id]):
                    elem_id = nbr_id
                    found = True
                    break
            # 3. If not found, search all block's elements (rare, O(log n))
            if not found:
                elem_id = block_element_search(pos, block)
        # 4. Interpolate field, integrate (RK4, etc.)
        vel = interpolate_velocity(pos, mesh, field)
        pos_new = rk4(pos, vel)
        # 5. Assign updated position, check if new block needed
        block_id_new = locate_block_for_position(pos_new)
        p.position, p.element_id, p.block_id = pos_new, elem_id, block_id_new
```


### Comment

- The **element cache + neighbor check** serves >85% of particles per step for AMR, nearly O(1) time.
- **Block search** is needed only for those on boundaries.

***

## V. Ghost/Halo Data Management

### Diagram

```
+-----------+     <---->       +-----------+
| Block 1   |------------------| Block 2   |
| Mesh data |   exchange data  | Mesh data |
| Ghost     |                  | Ghost     |
+-----------+                  +-----------+
```

**Strategies:**

- Update ghost/halo arrays at edges after each step only for blocks with boundary-crossers.
- Use flat, contiguous arrays for ghosts for coalesced reads.

***

## VI. Particle Batching, Spatial Rebatching

### Steps

1. **At end of each step**, update block ID for each particle.
2. **Parallel sort or scatter** particles array so that particles are batched by block for next step.
    - Use radix sort, parallel partition, or bucket scatter (O(N)).
3. **Maintain an index array or segmented structure**: for each block, know particle range.

***

## VII. JAX/CUDA Data and Memory Design

**Flat, static arrays for:**

- particle positions (N_particles, 3)
- velocities, element/block IDs (N_particles,)
- block mesh arrays (N_blocks), each with cell centroids, offsets
- neighbor lookup tables (fixed size per cell)
- ghost cell arrays (for field/mesh at boundaries)
- (optional) Morton codes for fast block/element search

**Never:**

- Use dynamic slices or variable-length arrays as state or carry in JAX scan!
- Accumulate growing lists during time marching.
- Pass large arrays as closures/hidden state in JAX `lax.scan`.

***

## VIII. Memory Efficiency: Avoiding JAX/XLA Memory Explosion

- **Scan carry should only include the flat particle array (and the minimal step accumulator if needed).**
- **Block/mesh data should be static—passed as constants, not closed over.**
- **Only output the final (or reduced/output) state, not all intermediates, unless needed.**
- **Use `lax.scan(time_step_fn, particles_init, xs=None)` for O(1) memory.**

***

## IX. Complete Top-Down Flowchart

```
   +-----------+
   |   Start   |
   +-----------+
        |
        v
+------------------+
|  Initialization  |
+------------------+
        |
        v
+--------------------------+
|  Partition AMR mesh into |
|  forest-of-octrees       |
+--------------------------+
        |
        v
+------------------------------+
|  Assign/init particles       |
|  (pos, v, elemID, blockID)   |
+------------------------------+
        |
        v
+------------------------------------------+
| for t in timesteps:                      |
|   For each block in parallel:            |
|       For particles in block (parallel): |
|           -> search, RK4, update         |
|   Sync ghosts and handle movers          |
|   Rebatch particles by new block         |
+------------------------------------------+
        |
        v
+------------+
|   Output   |
+------------+
        |
        v
+-----------+
|   Finish  |
+-----------+
```


***

## X. GPU-Efficient Pseudocode Snippets for Key Steps

### 1. Element Search / Neighbor O(1)

```pseudocode
def find_element_for_particle(particle, mesh):
    elem = particle.elem_id
    if mesh[elem].contains(particle.pos):
        return elem
    for nbr in mesh[elem].neighbors:
        if mesh[nbr].contains(particle.pos):
            return nbr
    # Fallback: search all elements in block/octree using hash/octree
    return block_element_search(particle.pos, mesh)
```


### 2. Particle Rebatch After Step

```pseudocode
# Pseudocode for GPU-parallel rebatch by block
def rebatch_particles_by_block(particles, N_blocks):
    # particles: array of structs with block_id
    particles_sorted = parallel_sort_by(particles, key=lambda p: p.block_id)
    # Compute start/end indices for each block (segment boundaries)
    block_indices = compute_block_ranges(particles_sorted, N_blocks)
    return particles_sorted, block_indices
```


### 3. Time Marching in JAX with scan

```python
def step_fn(particles, _):
    # group by block
    block_batches = rebatch_by_block_id(particles)
    # vmap over block for block-local batch update
    updated_batches = jax.vmap(update_block)(block_batches)
    # flatten
    particles_flat = flatten(updated_batches)
    return particles_flat, None

final_particles, _ = jax.lax.scan(step_fn, particles_init, xs=None, length=N_steps)
```


***

## XI. Critical Best Practices Checklist

- [x] Partition domain into forest-of-octrees, each mapped to GPU block/batch.
- [x] Store all particle \& mesh data as flat, static arrays (max coalescence).
- [x] Assign each particle an element and block ID, update at each step.
- [x] Outer loop = scan (JAX) or host for-loop; **never Python for over particles inside GPU/jitted function**.
- [x] Inner updates = fully batched (threads/vmap).
- [x] After each step, rebatch particles by block ID (GPU-parallel sort/scatter).
- [x] Use cached element + neighbor traversal for O(1) search in nearly all cases.
- [x] Update ghost/halo data only when needed, using coalesced accesses.
- [x] Carry in scan = only particles (and optionally step index), not all mesh data.
- [x] Avoid memory explosion in JAX/XLA with static shapes and minimal scan state.


***

## XII. Data Residency and Memory Management: GPU, RAM, and LAX Best Practices

### Core Principle

- **All arrays needed in tight, per-time-step kernels must reside in GPU memory** for maximal performance.
- **Static/constant data** should be buffered and "closed over" in compiled functions for efficient reuse.
- **Only the minimal required state should be carried (mutated or dynamic) between steps/time-marches.**

***

### A. Arrays That MUST Reside on the GPU (Device Arrays)

For full GPU performance, the following arrays should exist entirely in GPU memory and be updated in-place (if needed):

#### 1. **Particle Arrays** (*mutated every time step*)
- `positions` (N_particles, D): Particle positions (float32/64)
- `velocities` (N_particles, D): (if using, e.g. for RK4 or velocity caching)
- `element_ids` (N_particles,): Index of mesh element currently containing each particle
- `block_ids` (N_particles,): ID of forest-of-octrees block containing each particle
- (Optional) `morton_codes`, `active_flags`, or user metadata

> **These are updated and reshuffled on GPU at every step.**  
> **These arrays form the main carry/state for `lax.scan` or per-step kernel.**

#### 2. **Mesh Block Data** (*generally constant/static during marching, not mutated per step*)
- `block_offsets` (N_blocks+1,): Start/end cell index for each octree block
- `cell_centers` (N_cells, D): Centroid coordinates for all leaf cells
- `cell_half_sizes` (N_cells, D)
- `cell_element_ids` (N_cells, max_elements_per_cell): List of elements/leaves in each cell (flattened for JAX)
- `cell_neighbor_ids` (N_cells, n_neighbors): Neighbor indices for walk/search
- `block_meta` (N_blocks,): Metadata (bounds, ghost regions, etc)
- `field_data` (N_cells, ...): Local velocity/field data at mesh nodes, for interpolation

> **All of these are uploaded ONCE to GPU before marching.**  
> **Pass as “static” arguments to JIT-compiled JAX functions.**  
> **Do NOT copy these from RAM every time step!**

#### 3. **Ghost/Halo Buffers**
- `ghost_cells` (N_ghost, ...): Flat array for ghost cell/field data surrounding each octree block.
> **Updated only when block boundaries/ghosts change or after particle moves across blocks.**  
> **Remains on GPU except for very large distributed/multi-GPU deployments.**

***

### B. Arrays That Live on CPU/RAM

**Should NOT be used inside per-step kernels due to transfer cost:**
- Mesh generation, load, or mesh refinement history.
- Per-step output logs (positions, fields) not needed during integration.
- Checkpointing, visualization, or debug buffers.
- Any code, utility functions, or lookup tables not directly needed in integration/search/interpolation.

***

### C. Buffering, Compilation, Sharing in JAX/CUDA

**In JAX (and similarly for other “fusion” JIT compilers):**

#### 1. **Pass static data as closed/static/function arguments to JIT functions**
- All block/mesh data (`block_offsets`, `mesh`, `cell_centers`, etc.) passed as arguments *or* as partials/closures, but must be *static* (not modified, not returned by scan).
- These are JIT-buffered and reused for duration of compiled function—no per-step copy!

#### 2. **Per-time-step/carry arrays**
- Only pass `particles` (and other mutating state) as the *carry* or output of `scan`/per-step kernel.
- **Never return (or mutate) the entire mesh in the carry/state.**

#### 3. **Ghost data**
- Only update ghost arrays in blocks that experience particle boundary crossings.
- Use flat, preallocated buffers for each block.

#### 4. **Memory efficiency**
- Store arrays in minimal precision needed (float32 versus float64).
- Use padding or fixed-size buffers where dynamic arrays would cause memory explosion or non-compilation.

***

### D. Data Sharing: Best Practice

- **All blocks/octrees share “read-only” mesh, field, ghost information—this remains on GPU and is passed as shared/static data to kernels/functions.**
- **Each thread (particle) has exclusive write to its own output (position, velocity, IDs), reducing risk of race conditions.**
- **Write access to mesh/field arrays by kernels is *strongly discouraged* unless AMR is performed at intervals, and then via synchronized update.**

***

### E. Efficient Data Layout Table

| Array              | Live on | GPU-resident | Updated per step? | Shared (read-only) | In scan/carry? | Notes                        |
|--------------------|---------|--------------|------------------|--------------------|---------------|-------------------------------|
| `positions`        | GPU     | Yes          | Yes              | No                 | Yes           | Reshuffled, updated per step  |
| `element_ids`      | GPU     | Yes          | Yes              | No                 | Yes           | Cached, updated per step      |
| `block_ids`        | GPU     | Yes          | Yes              | No                 | Yes           | Updated if cross block bndry  |
| `velocities`       | GPU     | Yes          | Yes/optional     | No                 | Yes           | For RK4/profiling (optional)  |
| `block_offsets`    | GPU     | Yes          | No               | Yes                | No            | Static/shared                 |
| `cell_centers`     | GPU     | Yes          | No               | Yes                | No            | Static/shared                 |
| `cell_element_ids` | GPU     | Yes          | No               | Yes                | No            | Static/shared (flattened arr) |
| `field_data`       | GPU     | Yes          | No or infrequent | Yes                | No            | Static/shared                 |
| `ghost_cells`      | GPU     | Yes          | Only for bndry   | Yes                | No            | Fast copy if needed           |

***

### F. Key Points for JAX/`lax.scan` Compilability

- Only the `particles` array **should be in the scan carry/output**—all other arrays stay static and live as resident constants on GPU during compile/execution.
- Never “append” to arrays within a scan or allow shape/length of arrays to change within compiled code.
- Keep all arrays preallocated and flat—JAX/XLA optimize best for static shapes and allow full fusion/tiling for GPU kernels.
- If mesh/field is updated during marching (e.g., moving boundaries, AMR), perform this outside scan, re-upload, and restart scan (or treat mesh as new “epoch” between major remeshing)

***

### G. Data Movement in Practice

- GPU-to-GPU (across multi-GPU nodes): Only exchange minimal boundary/ghost/particle data—never whole arrays.
- CPU-to-GPU: Only at init, mesh changes, or for outputs/checkpoints—never every step.
- Between kernels/calls: All arrays stay in device (GPU) memory and referenced by pointer—**do not copy device→host nor between thread blocks except for explicit communication.**

***

## How This Looks at a Time Step

| At kernel start (on GPU):             | At kernel end (on GPU):        |
|---------------------------------------|--------------------------------|
| positions, element_ids, block_ids     | Updated positions, IDs         |
| cell_centers, cell_elements, field... | (unchanged–read-only)          |
| ghost_cells (updated as needed)       | ghost_cells (may be refreshed) |

- **All per-particle arrays are updated in-place.**
- **All block/mesh data is just accessed as needed, never copied, never part of “carry” in scan.**

***

By following these data policies, you will have both high memory efficiency and maximal GPU throughput, immunizing your code against both memory explosion and transfer bottlenecks—even as you scale particle count or mesh complexity.


***

## XIII. References and Further Reading

- p4est: Parallel Adaptive Mesh Refinement on Forests of Octrees ([link])[^1]
- AGAL code: GPU-native block AMR ([link])[^2]
- AMReX: Block-structured AMR on GPUs ([link])[^3]
- JAX best practices ([docs])[^4]
- CUDA particle tracking benchmarks ([NVIDIA])[^5]
- High-dimensional partitioning for AMR ([p4est])[^6]

***

## XIV. Closing Notes

The entire architecture builds on the principles of spatial/data locality, maximum parallelism, and memory throughput.
*Every time step*: all particles in a block are processed together, leveraging ghost data, local mesh, and efficient neighbor search. Rebatching after each step ensures continued spatial locality and full occupancy. Static data layouts and minimal scan carry are absolute keys to prevent memory issues.

This document provides all architectural, strategic, and practical details needed for a robust, scalable implementation of GPU-native particle tracking in forest-of-octrees adaptive mesh.
<div align="center">⁂</div>

[^1]: https://www.p4est.org

[^2]: https://arxiv.org/html/2502.16310v1

[^3]: https://www.sciencedirect.com/science/article/abs/pii/S002199912400072X

[^4]: https://docs.jax.dev/en/latest/faq.html

[^5]: https://developer.download.nvidia.com/compute/DevZone/C/html_x64/5_Simulations/particles/doc/particles.pdf

[^6]: https://epubs.siam.org/doi/10.1137/100791634

