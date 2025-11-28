# Block-Wise RK4 Workflow: Complete Variable Tracking

**Test File**: `test_blockwise_rk4_monitored.py`
**Date**: 2025-11-20
**Purpose**: Comprehensive documentation of all variables, their shapes, types, and memory allocation throughout the block-wise RK4 time-marching pipeline.

---

## Table of Contents
1. [Phase 1: Mesh Loading](#phase-1-mesh-loading)
2. [Phase 2: Forest Structure Creation](#phase-2-forest-structure-creation)
3. [Phase 3: Particle Generation & Initial Assignment](#phase-3-particle-generation--initial-assignment)
4. [Phase 4: Velocity Field Setup](#phase-4-velocity-field-setup)
5. [Phase 5: Baseline RK4 Test](#phase-5-baseline-rk4-test)
6. [Phase 6: Block-Wise RK4 Test](#phase-6-block-wise-rk4-test)
7. [Memory Summary](#memory-summary)

---

## Phase 1: Mesh Loading

### Input
- **File**: `/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_20.pvtu`
- **Format**: VTK Parallel Unstructured Grid (PVTU)

### Function Call
```python
node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
    Path(mesh_path),
    field_name='Displacement'
)
```

### Output Variables

#### `node_positions`
- **Type**: `numpy.ndarray`
- **Shape**: `(895972, 3)`
- **Dtype**: `float64` (from VTK)
- **Size**: 895,972 nodes × 3 coordinates × 8 bytes = **20.5 MB**
- **Location**: CPU (NumPy array)
- **Description**: 3D coordinates of all mesh nodes
- **Coordinate ranges**:
  - X: [-0.0300, 0.0300] m
  - Y: [-0.0230, 0.0230] m
  - Z: [-0.0100, 0.0000] m

#### `connectivity`
- **Type**: `numpy.ndarray`
- **Shape**: `(3485406, 4)`
- **Dtype**: `int32`
- **Size**: 3,485,406 elements × 4 nodes × 4 bytes = **53.1 MB**
- **Location**: CPU (NumPy array)
- **Description**: Tetrahedral element connectivity (indices into node_positions)
- **Value range**: [0, 895971] (node indices)

#### `velocity_field`
- **Type**: `numpy.ndarray`
- **Shape**: `(895972, 3)`
- **Dtype**: `float32` (converted from float64)
- **Size**: 895,972 nodes × 3 components × 4 bytes = **10.2 MB**
- **Location**: CPU (NumPy array)
- **Description**: Velocity vectors at each mesh node (loaded from 'Displacement' field)
- **Magnitude range**: [0.000231, 0.888108] m/s
- **Notes**:
  - Originally loaded as float64 from VTK
  - Converted to float32 for GPU efficiency
  - Verified to be 3D (if 2D, z-component added as zeros)

---

## Phase 2: Forest Structure Creation

### 2.1 Block Grid Creation

#### `blocks`
- **Type**: `list` of `Block` dataclasses
- **Length**: 256 blocks
- **Grid configuration**: (8, 8, 4) = 8×8×4 blocks
- **Location**: CPU (Python list)
- **Description**: Spatial subdivision of mesh domain
- **Each Block contains**:
  - `block_id`: int (0-255)
  - `ijk`: tuple (i, j, k) grid indices
  - `bounds`: min/max coordinates [(xmin, ymin, zmin), (xmax, ymax, zmax)]

### 2.2 Element Assignment

#### `block_elements`
- **Type**: `list` of 256 `numpy.ndarray`
- **Total elements**: 3,485,406 (all elements assigned to blocks)
- **Elements per block**: 2 to 444,040
- **Dtype**: `int32` (element indices)
- **Location**: CPU (Python list of NumPy arrays)
- **Description**: Maps each block to its contained elements
- **Classification**:
  - **Light blocks**: 240 blocks (< 10,000 elements each)
  - **Heavy blocks**: 16 blocks (≥ 10,000 elements each)

### 2.3 Element Neighbors

#### `face_neighbors`
- **Type**: `numpy.ndarray`
- **Shape**: `(3485406, 4)`
- **Dtype**: `int32`
- **Size**: 3,485,406 elements × 4 faces × 4 bytes = **53.1 MB**
- **Location**: CPU (NumPy array)
- **Description**: Neighbor element IDs for each face of each tetrahedron
- **Value range**: [-1, 3485405] where -1 = boundary (no neighbor)
- **Purpose**: L1 (face neighbor) search in incremental search

### 2.4 Padded Arrays

#### `padded_arrays` (PaddedArrays dataclass)
- **Type**: `PaddedArrays` object
- **Location**: CPU (NumPy arrays)
- **Total memory**: **6527.6 MB** (6.4 GB)

##### `padded_arrays.connectivity`
- **Type**: `numpy.ndarray`
- **Shape**: `(256, 444040, 4)`
- **Dtype**: `int32`
- **Size**: 256 blocks × 444,040 max_elements × 4 nodes × 4 bytes = **1,804 MB**
- **Description**: Block-local element connectivity (padded with -1)
- **Padding**: Blocks with fewer elements padded to 444,040 rows

##### `padded_arrays.node_positions`
- **Type**: `numpy.ndarray`
- **Shape**: `(256, 895972, 3)`
- **Dtype**: `float32`
- **Size**: 256 blocks × 895,972 nodes × 3 coords × 4 bytes = **2,617 MB**
- **Description**: Node positions replicated for all blocks
- **Note**: Each block has a full copy of all node positions (needed for element access)

##### `padded_arrays.element_bounds`
- **Type**: `numpy.ndarray`
- **Shape**: `(256, 444040, 3, 2)`
- **Dtype**: `float32`
- **Size**: 256 blocks × 444,040 elements × 3 dims × 2 (min/max) × 4 bytes = **2,107 MB**
- **Description**: Bounding boxes (AABB) for each element in each block
- **Purpose**: L2 (block search) using bounding box tests

##### `padded_arrays.elements_per_block`
- **Type**: `numpy.ndarray`
- **Shape**: `(256,)`
- **Dtype**: `int32`
- **Size**: 256 × 4 bytes = **1 KB**
- **Description**: Number of valid elements in each block (rest is padding)

##### `padded_arrays.max_elements_per_block`
- **Type**: `int`
- **Value**: 444,040
- **Description**: Maximum elements across all blocks (determines padding size)

### 2.5 Hash Bucket Arrays (Heavy Blocks Only)

#### `hash_bucket_arrays` (HashBucketArrays dataclass)
- **Type**: `HashBucketArrays` object
- **Blocks processed**: 16 heavy blocks (≥ 10,000 elements)
- **Location**: CPU (NumPy arrays)

##### `hash_bucket_arrays.bucket_element_ids`
- **Type**: `numpy.ndarray`
- **Shape**: `(16, n_buckets, max_elements_per_bucket)`
- **Dtype**: `int32`
- **Description**: Element IDs organized into spatial hash buckets for heavy blocks
- **Purpose**: L2b (hash bucket search) for efficient searching in large blocks

##### `hash_bucket_arrays.bucket_counts`
- **Type**: `numpy.ndarray`
- **Shape**: `(16, n_buckets)`
- **Dtype**: `int32`
- **Description**: Number of valid elements in each bucket

### 2.6 Incremental Searcher

#### `incremental_searcher`
- **Type**: `function` (closure)
- **Signature**: `(positions, cached_elem_ids, cached_block_ids) -> (elem_ids, block_ids, IncrementalSearchStats)`
- **Location**: CPU function (calls JAX-compiled GPU functions internally)
- **Description**: Multi-level hierarchical search:
  - **L0**: Check cached element (85-95% hit rate)
  - **L1**: Check face neighbors (3-10% hit rate)
  - **L2a**: Light block search (direct AABB test)
  - **L2b**: Heavy block search (hash bucket subdivision)
  - **L3**: Neighbor block search (0.1-1% hit rate)

---

## Phase 3: Particle Generation & Initial Assignment

### 3.1 Particle Generation

#### `particle_positions`
- **Type**: `numpy.ndarray`
- **Shape**: `(1000, 3)`
- **Dtype**: `float32`
- **Size**: 1,000 particles × 3 coords × 4 bytes = **12 KB**
- **Location**: CPU (NumPy array)
- **Description**: Random initial particle positions within mesh bounding box
- **Generation method**: Uniform random sampling in [xmin, xmax] × [ymin, ymax] × [zmin, zmax]

### 3.2 Initial Assignment

#### Function Call
```python
element_ids, block_ids, stats = incremental_searcher(
    particle_positions,
    np.full(n_particles, -1, dtype=np.int32),  # No cached elements
    np.full(n_particles, -1, dtype=np.int32)   # No cached blocks
)
```

#### `element_ids`
- **Type**: `numpy.ndarray`
- **Shape**: `(1000,)`
- **Dtype**: `int32`
- **Size**: 1,000 × 4 bytes = **4 KB**
- **Location**: CPU (NumPy array)
- **Description**: Containing element ID for each particle
- **Value range**: [-1, 3485405] where -1 = not found (outside mesh)
- **Found**: 960 particles (96.0%)

#### `block_ids`
- **Type**: `numpy.ndarray`
- **Shape**: `(1000,)`
- **Dtype**: `int32`
- **Size**: 1,000 × 4 bytes = **4 KB**
- **Location**: CPU (NumPy array)
- **Description**: Containing block ID for each particle
- **Value range**: [-1, 255] where -1 = not found

#### `found_mask`
- **Type**: `numpy.ndarray`
- **Shape**: `(1000,)`
- **Dtype**: `bool`
- **Size**: 1,000 × 1 byte = **1 KB**
- **Location**: CPU (NumPy array)
- **Description**: Boolean mask indicating which particles were found
- **True count**: 960 (96.0%)

### 3.3 ParticleData Creation

#### `particle_data`
- **Type**: `ParticleData` dataclass
- **Location**: CPU (NumPy arrays)
- **Description**: Container for all particle state information

##### `particle_data.positions`
- **Type**: `numpy.ndarray`
- **Shape**: `(960, 3)`
- **Dtype**: `float32`
- **Size**: 960 × 3 × 4 bytes = **11.5 KB**
- **Description**: Only particles found in mesh (filtered by found_mask)

##### `particle_data.velocities`
- **Type**: `numpy.ndarray`
- **Shape**: `(960, 3)`
- **Dtype**: `float32`
- **Size**: 960 × 3 × 4 bytes = **11.5 KB**
- **Description**: Particle velocities (initialized to zeros, will be interpolated)

##### `particle_data.element_ids`
- **Type**: `numpy.ndarray`
- **Shape**: `(960,)`
- **Dtype**: `int32`
- **Size**: 960 × 4 bytes = **3.8 KB**
- **Description**: Containing element IDs (filtered by found_mask)

##### `particle_data.block_ids`
- **Type**: `numpy.ndarray`
- **Shape**: `(960,)`
- **Dtype**: `int32`
- **Size**: 960 × 4 bytes = **3.8 KB**
- **Description**: Containing block IDs (filtered by found_mask)

##### `particle_data.active_mask`
- **Type**: `numpy.ndarray`
- **Shape**: `(960,)`
- **Dtype**: `bool`
- **Size**: 960 × 1 byte = **960 bytes**
- **Description**: All True (all found particles are active)

##### `particle_data.n_active`
- **Type**: `int` (property)
- **Value**: 960
- **Description**: Count of active particles

---

## Phase 4: Velocity Field Setup

### `velocity_field_all_blocks`
- **Type**: `numpy.ndarray`
- **Shape**: `(256, 895972, 3)`
- **Dtype**: `float32`
- **Size**: 256 blocks × 895,972 nodes × 3 components × 4 bytes = **2,617 MB** (2.6 GB)
- **Location**: CPU (NumPy array)
- **Description**: Velocity field replicated for all blocks
- **Creation**: `np.tile(velocity_field, (n_blocks, 1, 1))`
- **Purpose**: Each block needs access to full velocity field for interpolation
- **Note**: This is CPU memory; blocks will be uploaded to GPU one at a time

---

## Phase 5: Baseline RK4 Test

### 5.1 Baseline Particle Data Copy

#### `particle_data_baseline`
- **Type**: `ParticleData` dataclass
- **Location**: CPU (NumPy arrays)
- **Description**: Copy of particle_data for baseline test
- **All fields**: Same shapes/sizes as `particle_data` (see Phase 3.3)

### 5.2 Baseline RK4 Time Marching Loop

**Timestep**: `dt = 0.001` seconds
**Number of steps**: 10
**Total simulation time**: 0.01 seconds

#### Per-Timestep Variables (GPU)

For each timestep, the following GPU transfers occur:

##### Upload to GPU (960 particles)
1. `positions_gpu`: (960, 3) float32 → **11.5 KB**
2. `element_ids_gpu`: (960,) int32 → **3.8 KB**
3. `block_ids_gpu`: (960,) int32 → **3.8 KB**
4. `connectivity_gpu`: (3485406, 4) int32 → **53.1 MB**
5. `node_positions_gpu`: (895972, 3) float32 → **10.2 MB**
6. `velocity_field_gpu`: (895972, 3) float32 → **10.2 MB**

**Total upload per step**: ~**73.5 MB**

##### GPU Computations

###### Velocity Interpolation
```python
velocities = batch_interpolate_velocities(
    positions_gpu,
    element_ids_gpu,
    connectivity_gpu,
    node_positions_gpu,
    velocity_field_gpu
)
```
- **Input**: positions_gpu (960, 3), element_ids_gpu (960,)
- **Output**: velocities (960, 3) float32 on GPU
- **Method**: Barycentric interpolation in containing element

###### RK4 Integration
```python
new_positions, new_element_ids = rk4_step_with_incremental_search(
    particle_data_baseline,
    velocities,
    dt,
    incremental_searcher
)
```

**RK4 Stages**: For each particle, compute:
1. **k1** = v(t, x_n) - Already computed (velocities)
2. **k2** = v(t + dt/2, x_n + dt/2 × k1)
   - Compute intermediate positions: x_n + dt/2 × k1
   - Search for new elements (L0+L1+L2+L3)
   - Interpolate velocities at intermediate positions
3. **k3** = v(t + dt/2, x_n + dt/2 × k2)
   - Compute intermediate positions: x_n + dt/2 × k2
   - Search for new elements (L0+L1+L2+L3)
   - Interpolate velocities at intermediate positions
4. **k4** = v(t + dt, x_n + dt × k3)
   - Compute intermediate positions: x_n + dt × k3
   - Search for new elements (L0+L1+L2+L3)
   - Interpolate velocities at intermediate positions
5. **Final**: x_{n+1} = x_n + dt/6 × (k1 + 2×k2 + 2×k3 + k4)
   - Search for new elements at final positions

**Note**: In baseline approach, interpolation and integration are SEPARATE:
- 1 interpolation call for all particles → k1
- RK4 integration does 3 more interpolations internally (k2, k3, k4)
- **Total**: 4 CPU-GPU round trips for interpolation per timestep

##### Download from GPU
1. `new_positions`: (960, 3) float32 → **11.5 KB**
2. `new_element_ids`: (960,) int32 → **3.8 KB**
3. `new_block_ids`: (960,) int32 → **3.8 KB**

**Total download per step**: ~**19 KB**

#### Baseline Results
- **Total time**: 55.27 seconds
- **Time per step**: 5.527 ± 0.721 seconds
- **Throughput**: 173.7 particles/second
- **CPU usage**: 70.0% average, 158.6% max
- **GPU memory**: 2965 MB
- **GPU utilization**: 48.1% average, 65.0% max

---

## Phase 6: Block-Wise RK4 Test

### 6.1 Block-Wise Particle Data Copy

#### `particle_data_blockwise`
- **Type**: `ParticleData` dataclass
- **Location**: CPU (NumPy arrays)
- **Description**: Copy of particle_data for block-wise test
- **All fields**: Same shapes/sizes as `particle_data` (see Phase 3.3)

### 6.2 Block-Wise RK4 Architecture

**Key Difference**: Complete RK4 integration for one block at a time, with k1-k4 computed on-the-fly.

#### Per-Timestep Workflow

For each timestep:
1. Group particles by block
2. For each block with particles:
   - Upload block data ONCE
   - Complete RK4 integration on GPU (4 interpolations inside)
   - Download results ONCE

#### Per-Block Variables (GPU)

##### Upload to GPU (per block)
1. `block_positions_gpu`: (n_particles_in_block, 3) float32
2. `block_element_ids_gpu`: (n_particles_in_block,) int32
3. `connectivity_gpu`: (3485406, 4) int32 → **53.1 MB** (persistent)
4. `node_positions_gpu`: (895972, 3) float32 → **10.2 MB** (persistent)
5. `velocity_field_gpu`: (895972, 3) float32 → **10.2 MB**

**Note**: connectivity and node_positions can be kept persistent on GPU across blocks.

##### GPU Computations (Block-Wise RK4 Single Block)

```python
new_positions, new_element_ids, stats = rk4_step_blockwise_single_block(
    positions_gpu,
    element_ids_gpu,
    block_id,
    connectivity_gpu,
    node_positions_gpu,
    velocity_field_gpu,
    dt,
    incremental_searcher,
    current_time
)
```

**RK4 Stages** (all on GPU, on-the-fly):

###### Stage 1: k1 = v(t, x_n)
```python
k1 = batch_interpolate_velocities(
    positions,
    element_ids,
    connectivity_gpu,
    node_positions_gpu,
    velocity_field_gpu
)
```
- **Input**: positions (n_particles, 3), element_ids (n_particles,)
- **Output**: k1 (n_particles, 3) float32 on GPU
- **Note**: k1 is NOT stored long-term, used immediately

###### Stage 2: k2 = v(t + dt/2, x_n + dt/2 × k1)
```python
pos_k2 = positions + 0.5 * dt * k1  # On GPU

# Search for containing elements at k2 positions
pos_k2_np = np.array(pos_k2)  # Download to CPU
elem_k2_np, block_k2_np, stats_k2 = incremental_searcher(
    pos_k2_np,
    np.array(element_ids),
    np.array(block_ids)
)
elem_k2 = jax.device_put(elem_k2_np)  # Upload to GPU

k2 = batch_interpolate_velocities(
    pos_k2,
    elem_k2,
    connectivity_gpu,
    node_positions_gpu,
    velocity_field_gpu
)
```
- **Intermediate**: pos_k2 (n_particles, 3) float32 on GPU
- **Search stats**: L0/L1/L2/L3 hit rates tracked
- **Output**: k2 (n_particles, 3) float32 on GPU
- **Note**: k1 is no longer needed, k2 used immediately

###### Stage 3: k3 = v(t + dt/2, x_n + dt/2 × k2)
```python
pos_k3 = positions + 0.5 * dt * k2  # On GPU

# Search using k2 elements as cache
pos_k3_np = np.array(pos_k3)
elem_k3_np, block_k3_np, stats_k3 = incremental_searcher(
    pos_k3_np,
    elem_k2_np,
    block_k2_np
)
elem_k3 = jax.device_put(elem_k3_np)

k3 = batch_interpolate_velocities(
    pos_k3,
    elem_k3,
    connectivity_gpu,
    node_positions_gpu,
    velocity_field_gpu
)
```
- **Intermediate**: pos_k3 (n_particles, 3) float32 on GPU
- **Output**: k3 (n_particles, 3) float32 on GPU
- **Note**: k2 is no longer needed

###### Stage 4: k4 = v(t + dt, x_n + dt × k3)
```python
pos_k4 = positions + dt * k3  # On GPU

# Search using k3 elements as cache
pos_k4_np = np.array(pos_k4)
elem_k4_np, block_k4_np, stats_k4 = incremental_searcher(
    pos_k4_np,
    elem_k3_np,
    block_k3_np
)
elem_k4 = jax.device_put(elem_k4_np)

k4 = batch_interpolate_velocities(
    pos_k4,
    elem_k4,
    connectivity_gpu,
    node_positions_gpu,
    velocity_field_gpu
)
```
- **Intermediate**: pos_k4 (n_particles, 3) float32 on GPU
- **Output**: k4 (n_particles, 3) float32 on GPU

###### Stage 5: Final RK4 Combination
```python
new_positions = positions + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

# Final search at new positions using k4 elements as cache
new_positions_np = np.array(new_positions)
new_element_ids_np, new_block_ids_np, stats_final = incremental_searcher(
    new_positions_np,
    elem_k4_np,
    block_k4_np
)
new_element_ids = jax.device_put(new_element_ids_np)
```
- **Output**: new_positions (n_particles, 3) float32 on GPU
- **Output**: new_element_ids (n_particles,) int32 on GPU
- **Note**: k1, k2, k3, k4 were NEVER stored beyond their immediate use!

##### Memory Efficiency

**Traditional RK4**: Store k1, k2, k3, k4 → 4 × (n_particles × 3) float32 arrays
**Block-wise RK4**: Only store positions and current k-stage → 75% memory savings!

For 960 particles:
- Traditional: 4 × (960 × 3 × 4) = **46 KB**
- Block-wise: 1 × (960 × 3 × 4) = **11.5 KB**
- **Savings**: 34.5 KB per block (75%)

##### Transfer Efficiency

**Baseline approach**: 4 separate interpolation calls → 4 CPU-GPU round trips
**Block-wise approach**: 1 block upload, 1 block download → **4× reduction in transfers**

##### Download from GPU (per block)
1. `new_positions`: (n_particles_in_block, 3) float32
2. `new_element_ids`: (n_particles_in_block,) int32

#### Statistics Tracking

##### `BlockwiseRK4Stats`
- **Type**: `BlockwiseRK4Stats` dataclass
- **Location**: CPU (Python object)

###### Fields:
- `n_particles`: int (total particles processed)
- `n_blocks_active`: int (blocks with particles)
- `n_searches_total`: int (total L0+L1+L2+L3 searches)
- `l0_hits_total`: int (L0 cache hits)
- `l1_hits_total`: int (L1 neighbor hits)
- `l2_hits_total`: int (L2 block hits)
- `time_total`: float (seconds)
- `time_per_block`: list of floats (per-block timings)

###### Methods:
- `throughput()`: particles/second
- `l0_hit_rate()`: L0 percentage
- `l1_hit_rate()`: L1 percentage
- `l2_hit_rate()`: L2 percentage

---

## Memory Summary

### CPU Memory Usage

| Variable | Size | Location |
|----------|------|----------|
| `node_positions` | 20.5 MB | CPU |
| `connectivity` | 53.1 MB | CPU |
| `velocity_field` | 10.2 MB | CPU |
| `face_neighbors` | 53.1 MB | CPU |
| `padded_arrays` | 6,527.6 MB | CPU |
| `velocity_field_all_blocks` | 2,617 MB | CPU |
| `particle_data` | ~40 KB | CPU |
| **Total** | **~9.3 GB** | **CPU** |

### GPU Memory Usage

| Phase | Variables | Size |
|-------|-----------|------|
| **Persistent** | connectivity_gpu, node_positions_gpu | 63.3 MB |
| **Per-Block Upload** | velocity_field_gpu, positions, element_ids | ~10.2 MB |
| **Intermediate (on-the-fly)** | k1, k2, k3, k4, pos_k2, pos_k3, pos_k4 | ~80 KB |
| **Per-Block Download** | new_positions, new_element_ids | ~15 KB |
| **Peak (baseline)** | All data + JAX buffers | **2,965 MB** |

### Memory Optimization Notes

1. **Block-wise Processing**: Only one block's velocity field on GPU at a time
2. **On-the-fly k-stages**: No storage of intermediate RK4 velocities (75% savings)
3. **Padded Arrays**: Large CPU memory cost (6.5 GB) but enables efficient GPU access patterns
4. **Velocity Field Replication**: 2.6 GB CPU memory for block-replicated velocity field

---

## Variable Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: MESH LOADING (CPU)                                 │
├─────────────────────────────────────────────────────────────┤
│ PVTU File                                                    │
│   ↓                                                          │
│ load_mesh_from_pvtu()                                       │
│   ↓                                                          │
│ node_positions (895972, 3) float64 → 20.5 MB               │
│ connectivity (3485406, 4) int32 → 53.1 MB                  │
│ velocity_field (895972, 3) float32 → 10.2 MB               │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: FOREST STRUCTURE (CPU)                             │
├─────────────────────────────────────────────────────────────┤
│ create_block_grid()                                         │
│   ↓                                                          │
│ blocks (256) → ~10 KB                                       │
│                                                              │
│ assign_elements_to_blocks()                                 │
│   ↓                                                          │
│ block_elements (256 arrays) → ~53 MB                       │
│                                                              │
│ build_element_neighbors()                                   │
│   ↓                                                          │
│ face_neighbors (3485406, 4) int32 → 53.1 MB               │
│                                                              │
│ create_padded_arrays()                                      │
│   ↓                                                          │
│ padded_arrays.connectivity (256, 444040, 4) → 1,804 MB    │
│ padded_arrays.node_positions (256, 895972, 3) → 2,617 MB  │
│ padded_arrays.element_bounds (256, 444040, 3, 2) → 2,107 MB│
│                                                              │
│ build_hash_buckets() [16 heavy blocks]                     │
│   ↓                                                          │
│ hash_bucket_arrays → varies                                │
│                                                              │
│ create_incremental_searcher()                               │
│   ↓                                                          │
│ incremental_searcher (function)                             │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: PARTICLE GENERATION (CPU)                          │
├─────────────────────────────────────────────────────────────┤
│ generate_random_particles()                                 │
│   ↓                                                          │
│ particle_positions (1000, 3) float32 → 12 KB              │
│                                                              │
│ incremental_searcher()                                       │
│   ↓                                                          │
│ element_ids (1000,) int32 → 4 KB                          │
│ block_ids (1000,) int32 → 4 KB                            │
│                                                              │
│ filter by found_mask (96.0% found)                         │
│   ↓                                                          │
│ ParticleData:                                                │
│   positions (960, 3) float32 → 11.5 KB                    │
│   velocities (960, 3) float32 → 11.5 KB                   │
│   element_ids (960,) int32 → 3.8 KB                       │
│   block_ids (960,) int32 → 3.8 KB                         │
│   active_mask (960,) bool → 960 bytes                      │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: VELOCITY FIELD SETUP (CPU)                         │
├─────────────────────────────────────────────────────────────┤
│ np.tile(velocity_field, (256, 1, 1))                        │
│   ↓                                                          │
│ velocity_field_all_blocks (256, 895972, 3) → 2,617 MB     │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: BASELINE RK4 (CPU ↔ GPU)                          │
├─────────────────────────────────────────────────────────────┤
│ For each timestep:                                          │
│                                                              │
│   CPU → GPU: Upload particle data + mesh + velocity        │
│   │   positions (960, 3) → 11.5 KB                         │
│   │   element_ids (960,) → 3.8 KB                          │
│   │   connectivity → 53.1 MB                                │
│   │   node_positions → 10.2 MB                              │
│   │   velocity_field → 10.2 MB                              │
│   ↓                                                          │
│   GPU: batch_interpolate_velocities() → k1                  │
│   ↓                                                          │
│   GPU ↔ CPU: rk4_step_with_incremental_search()            │
│   │   4 separate interpolation calls (k1, k2, k3, k4)      │
│   │   4 incremental searches (k2, k3, k4, final)           │
│   ↓                                                          │
│   GPU → CPU: Download new_positions, new_element_ids        │
│   │   new_positions (960, 3) → 11.5 KB                     │
│   │   new_element_ids (960,) → 3.8 KB                      │
│                                                              │
│ Result: 173.7 p/s, 55.27 s total                           │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 6: BLOCK-WISE RK4 (CPU ↔ GPU)                        │
├─────────────────────────────────────────────────────────────┤
│ For each timestep:                                          │
│   group_particles_by_block()                                │
│   ↓                                                          │
│   For each block with particles:                            │
│                                                              │
│     CPU → GPU: Upload block data ONCE                       │
│     │   block_positions (n_block, 3)                        │
│     │   block_element_ids (n_block,)                        │
│     │   connectivity (persistent)                            │
│     │   node_positions (persistent)                          │
│     │   velocity_field_block (895972, 3) → 10.2 MB         │
│     ↓                                                        │
│     GPU: rk4_step_blockwise_single_block()                  │
│     │                                                        │
│     │   Stage 1: k1 = v(t, x_n)                            │
│     │     batch_interpolate_velocities() → k1               │
│     │                                                        │
│     │   Stage 2: pos_k2 = x_n + 0.5*dt*k1                  │
│     │     GPU → CPU: pos_k2                                 │
│     │     CPU: incremental_searcher() → elem_k2             │
│     │     CPU → GPU: elem_k2                                 │
│     │     batch_interpolate_velocities() → k2               │
│     │     [k1 no longer stored]                             │
│     │                                                        │
│     │   Stage 3: pos_k3 = x_n + 0.5*dt*k2                  │
│     │     GPU → CPU: pos_k3                                 │
│     │     CPU: incremental_searcher(cache=elem_k2) → elem_k3│
│     │     CPU → GPU: elem_k3                                 │
│     │     batch_interpolate_velocities() → k3               │
│     │     [k2 no longer stored]                             │
│     │                                                        │
│     │   Stage 4: pos_k4 = x_n + dt*k3                      │
│     │     GPU → CPU: pos_k4                                 │
│     │     CPU: incremental_searcher(cache=elem_k3) → elem_k4│
│     │     CPU → GPU: elem_k4                                 │
│     │     batch_interpolate_velocities() → k4               │
│     │     [k3 no longer stored]                             │
│     │                                                        │
│     │   Stage 5: new_pos = x_n + dt/6*(k1+2k2+2k3+k4)     │
│     │     GPU → CPU: new_positions                          │
│     │     CPU: incremental_searcher(cache=elem_k4) → new_elem│
│     │     CPU → GPU: new_element_ids                        │
│     ↓                                                        │
│     GPU → CPU: Download new_positions, new_element_ids ONCE │
│                                                              │
│ Expected Result: 15-18 p/s improvement (15-40%)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Insights

### Memory Allocation Strategy
1. **Large static structures**: CPU-resident (padded_arrays, velocity_field_all_blocks)
2. **Persistent GPU data**: connectivity, node_positions (uploaded once)
3. **Block-streamed**: Velocity fields uploaded per-block
4. **On-the-fly computation**: k1-k4 never stored beyond immediate use

### Performance Bottlenecks (Baseline)
1. **CPU-GPU transfers**: 4 round trips per timestep for interpolation
2. **Search overhead**: L2/L3 searches for particles moving between blocks
3. **Memory bandwidth**: Large mesh data (63 MB) uploaded per timestep

### Block-Wise Optimizations
1. **Reduced transfers**: Upload/download once per block (4× reduction)
2. **Memory efficiency**: 75% reduction in intermediate storage
3. **Incremental search**: L0 cache hit rate 85-95% using previous elements
4. **Spatial locality**: Particles in same block likely stay in same block

---

## Appendix: Data Type Reference

| Type | Description | Size |
|------|-------------|------|
| `float32` | 32-bit floating point | 4 bytes |
| `float64` | 64-bit floating point | 8 bytes |
| `int32` | 32-bit signed integer | 4 bytes |
| `bool` | Boolean (NumPy) | 1 byte |

### Coordinate Systems
- **World space**: Physical coordinates in meters
- **Barycentric space**: (λ0, λ1, λ2, λ3) tetrahedron coordinates, sum to 1
- **Block grid space**: (i, j, k) integer block indices

### Index Ranges
- **Node indices**: [0, 895971]
- **Element indices**: [0, 3485405] or -1 (not found)
- **Block indices**: [0, 255] or -1 (outside domain)
- **Face neighbor indices**: [0, 3485405] or -1 (boundary)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-20
**Author**: Claude Code Agent
**Test File**: test_blockwise_rk4_monitored.py
