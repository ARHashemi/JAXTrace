# GPU-Native Particle Tracking Implementation Plan
**Forest-of-Octrees Architecture from Scratch**

---

**Branch**: `gpu_native_implementation`
**Created**: 2025-11-02
**Status**: Planning Phase
**Estimated Duration**: 6-10 weeks (MVP in 4-5 weeks)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Analysis](#system-analysis)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Phases](#implementation-phases)
5. [Component Reuse Strategy](#component-reuse-strategy)
6. [Testing Strategy](#testing-strategy)
7. [Configuration](#configuration)
8. [Memory Management](#memory-management)
9. [Success Metrics](#success-metrics)
10. [Risk Mitigation](#risk-mitigation)

---

## Executive Summary

### Goal
Implement GPU-native particle tracking using the forest-of-octrees architecture described in `docs/High_Performance_Particle_Tracking_on_the_GPU.md` to enable tracking of 100K+ particles with 50-100× speedup over CPU implementation.

### Approach
- **Clean implementation** from scratch on new branch (no migration of previous phase1-optimization work)
- **Fine-grained development**: 10 phases for implementation
- **Coarse-grained documentation**: 3 phases for user-facing docs
- **Incremental testing**: Unit tests per component, integration tests per phase

### Key Constraints
- **Hardware**: NVIDIA T1000 with 4GB VRAM (primary constraint)
- **Mesh**: ThreadedA reference mesh (64 pieces, ~1300 cells, 160 timesteps)
- **Memory**: Must fit 100K particles + forest structure in 4GB
- **Accuracy**: <1% interpolation error vs CPU tracker

### Timeline
- **Phase 0-4**: Foundation and single-block tracking (2-3 weeks)
- **Phase 5**: Multi-block parallelism - **MVP** (week 4-5)
- **Phase 6-7**: Time marching and ghost regions (week 5-7)
- **Phase 8-9**: Optimization and scaling (week 7-10)

---

## System Analysis

### 1. Hardware Specifications

**GPU**:
- Model: NVIDIA T1000 (Turing architecture)
- VRAM: 4GB (usable: ~3.7GB)
- CUDA Capability: 7.5
- **Implication**: Limited VRAM is primary constraint; requires careful memory management

**CPU/RAM**:
- CPU: Intel Core i7-12700 (12 cores, 20 threads)
- RAM: 31GB total (~19GB available)
- **Implication**: Can handle mesh preprocessing and data staging

### 2. Mesh Characteristics

**Dataset**: ThreadedA (Reference Mesh)
- Location: `/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_*.pvtu`
- Timesteps: 160 (indexed 0, 100-259)
- Parallel pieces: 64 per timestep (53 non-empty, 11 empty)
- Total points: ~2,301 per timestep
- Total cells: ~1,296 tetrahedral elements per timestep
- Element type: VTK_TETRA (4-node tetrahedral)

**Domain Size**:
- X: [-0.030, 0.030] meters (60mm width)
- Y: [-0.023, 0.023] meters (46mm height)
- Z: [-0.010, 0.000] meters (10mm depth)
- **Implication**: Microscale welding simulation

**Mesh Type**: Adaptive Mesh Refinement (AMR)
- LEVEL field indicates refinement hierarchy
- Varying topology across timesteps
- Irregular piece decomposition (NOT regular grid)

**Available Fields**:
- **Displacement** [3 components] - PRIMARY tracking field (stores velocity)
- Pressure [1 component]
- Reactions [3 components]
- Temperature [1 component]
- LEVEL [1 component] - AMR level indicator

### 3. Current Codebase Structure

**Modules to Keep As-Is** (no migration needed):
- `jaxtrace/io/*` - VTK readers/writers
- `jaxtrace/visualization/*` - Plotting and visualization
- `jaxtrace/density/*` - KDE/SPH density estimation
- `jaxtrace/tracking/analysis.py` - Trajectory analysis

**GPU Components Available for Reuse** (from phase1-optimization):
- `morton_code.py` (511 lines) - Morton encoding/decoding, pure JAX
- `hash_octree.py` (859 lines) - O(1) hash table lookup
- `element_testing_jax.py` - Barycentric testing for tetrahedra
- `interpolator_jax_simple.py` - FEM interpolation (JAX)
- `gpu_field_sampling.py` - Full GPU pipeline reference

**What Needs Complete Rewrite**:
- Octree building (forest partitioning, block-local structures)
- Particle tracking core (block-aware, spatial batching)
- Field sampling (per-block, ghost regions)
- Time marching (lax.scan with minimal carry)

---

## Architecture Overview

### Forest-of-Octrees Concept

**Traditional Approach** (current implementation):
- Single global octree covering entire domain
- Flat particle parallelism (all particles treated uniformly)
- Global field sampling with io_callback

**Forest Approach** (new implementation):
- Domain decomposed into **B blocks** (e.g., 2×2×2 = 8 blocks)
- Each block is root of independent sub-octree
- **Hierarchical parallelism**: vmap over blocks → vmap over particles
- **Spatial batching**: Particles grouped by block_id for memory locality

### Key Design Principles

1. **Spatial Locality**
   - Particles in same block access nearby mesh elements
   - Maximizes GPU cache efficiency
   - Reduces memory bandwidth pressure

2. **Element ID Caching**
   - Each particle stores last containing element
   - 85-99% cache hit rate (from strategy document)
   - Reduces search cost 10-50×

3. **Three-Tier Search Hierarchy**
   ```
   Level 1: Check cached element (O(1), 85-99% hit rate)
   Level 2: Check neighbor elements (O(1), small fixed list)
   Level 3: Block-local octree search (O(log n), rare fallback)
   ```
   No global search needed!

4. **Flat Static Arrays**
   - All data as contiguous flat arrays (no pointers, no ragged arrays)
   - Enables JAX JIT compilation
   - GPU kernel fusion for performance

5. **Minimal Scan Carry**
   - Only particle arrays in lax.scan carry
   - Mesh/field data passed as static constants
   - Prevents memory explosion during time stepping

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INITIALIZATION (CPU)                                       │
│  ├─ Load mesh (VTK)                                         │
│  ├─ Build forest blocks (regular grid or Morton partition) │
│  ├─ Assign elements to blocks                              │
│  ├─ Precompute element neighbors (adjacency)               │
│  ├─ Seed particles, assign block_ids                       │
│  └─ Upload to GPU (JAX DeviceArrays)                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  GPU TIME MARCHING (lax.scan)                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  For each timestep t:                               │   │
│  │  ┌───────────────────────────────────────────────┐ │   │
│  │  │  For each block b (parallel vmap):            │ │   │
│  │  │  ┌─────────────────────────────────────────┐ │ │   │
│  │  │  │  For each particle p in block (vmap):   │ │ │   │
│  │  │  │                                          │ │ │   │
│  │  │  │  1. Search (cached → neighbors → tree)  │ │ │   │
│  │  │  │  2. FEM interpolation                   │ │ │   │
│  │  │  │  3. RK4 integration                      │ │ │   │
│  │  │  │  4. Update position                      │ │ │   │
│  │  │  │                                          │ │ │   │
│  │  │  └─────────────────────────────────────────┘ │ │   │
│  │  └───────────────────────────────────────────────┘ │   │
│  │                                                      │   │
│  │  Rebatch: Sort particles by new block_id            │   │
│  │  Ghost Exchange: Sync particles at block boundaries │   │
│  │                                                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT (CPU)                                               │
│  ├─ Download trajectories from GPU                          │
│  ├─ Export to VTK/HDF5                                      │
│  ├─ Visualization                                           │
│  └─ Density analysis (KDE/SPH)                              │
└─────────────────────────────────────────────────────────────┘
```

### Memory Layout

**GPU Memory Organization** (4GB total, ~3.7GB usable):

```
┌──────────────────────────────────────────────────────────┐
│  STATIC DATA (uploaded once, ~1 GB)                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Block Metadata (B blocks)                         │  │
│  │  ├─ block_bounds [B, 6] (xmin, xmax, ...)          │  │
│  │  ├─ block_centers [B, 3]                           │  │
│  │  ├─ block_offsets [B+1] (element index ranges)     │  │
│  │  └─ neighbor_ids [B, 6] (face neighbors)           │  │
│  │                                                     │  │
│  │  Mesh Data (per block or global)                   │  │
│  │  ├─ cell_positions [N_nodes, 3]                    │  │
│  │  ├─ cell_connectivity [N_cells, 4]                 │  │
│  │  ├─ element_neighbors [N_cells, max_neighbors]     │  │
│  │  └─ element_to_block [N_cells]                     │  │
│  │                                                     │  │
│  │  Field Data (1-10 timesteps cached)                │  │
│  │  └─ velocity_fields [T_cached, N_nodes, 3]         │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  DYNAMIC DATA (scan carry, ~3 GB)                        │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Particle Arrays (N particles)                     │  │
│  │  ├─ positions [N, 3] float32                       │  │
│  │  ├─ velocities [N, 3] float32                      │  │
│  │  ├─ element_ids [N] int32  ← CACHE                 │  │
│  │  ├─ block_ids [N] int32                            │  │
│  │  └─ active_mask [N] bool (for boundary particles)  │  │
│  │                                                     │  │
│  │  Per particle: 32 bytes                            │  │
│  │  Max particles: ~100K (3.2 MB)                     │  │
│  │                                                     │  │
│  │  Ghost/Halo Buffers                                │  │
│  │  └─ ghost_particles [N_ghost, ...] (boundaries)    │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

**Memory Budget**:
- **25%** (1 GB): Forest structure + mesh + field cache
- **75%** (3 GB): Particle data + ghost buffers + trajectory storage
- **Target**: 100K particles @ 32 bytes/particle = 3.2 MB (well within budget)

---

## Implementation Phases

### Overview: Fine-Grained Development Phases

**10 Development Phases** (for implementation):
- Phase 0: Foundation & Branch Setup
- Phase 1: CPU Block-Local Search
- Phase 2: JAX Data Structures & Memory
- Phase 3: GPU Single-Particle Kernel
- Phase 4: Vectorization (vmap over Particles)
- Phase 5: Multi-Block Parallelism (**MVP**)
- Phase 6: Time Marching (lax.scan)
- Phase 7: Ghost Regions & Halo Exchange
- Phase 8: Optimization & Scaling
- Phase 9: Hash Octree Integration (Optional)

**3 Documentation Phases** (for user-facing docs):
- Phase I: Pre-initialization (Phases 0-2)
- Phase II: GPU Time Marching (Phases 3-7)
- Phase III: Optimization & Production (Phases 8-9)

---

### PHASE 0: Foundation & Branch Setup
**Duration**: 2-3 days
**Status**: Planning

#### Objectives
1. Set up clean project structure
2. Create configuration schema
3. Implement regular forest grid generator
4. Visualize forest blocks on ThreadedA mesh
5. Establish testing infrastructure

#### Deliverables

**Directory Structure**:
```
jaxtrace/
├── gpu/
│   ├── __init__.py
│   ├── config.py                 # Configuration dataclass
│   ├── forest/
│   │   ├── __init__.py
│   │   ├── block_builder.py      # Regular grid generator
│   │   └── visualize.py          # Block visualization
│   └── utils.py                  # Helper functions
tests/
├── gpu/
│   ├── __init__.py
│   ├── test_block_builder.py
│   └── test_config.py
examples/
└── gpu/
    └── phase0_block_visualization.ipynb
docs/
└── gpu/
    └── PHASE_0_FOUNDATION.md
```

**Files to Create**:

1. **`jaxtrace/gpu/config.py`**
   - Configuration dataclass
   - Default parameters
   - Validation logic

2. **`jaxtrace/gpu/forest/block_builder.py`**
   - `create_regular_grid(bounds, grid_size)` → list of BlockMetadata
   - `BlockMetadata` dataclass with bounds, center, neighbors
   - Neighbor topology computation (6-face connectivity)

3. **`jaxtrace/gpu/forest/visualize.py`**
   - `visualize_forest_blocks(blocks, particles=None, save_path)`
   - 3D wireframe plot of blocks
   - 2D projections (XY, XZ, YZ)
   - Overlay particles if provided

4. **`tests/gpu/test_block_builder.py`**
   - Test regular grid generation (2×2×2, 4×4×2)
   - Verify neighbor topology
   - Check boundary conditions (edge/corner blocks)

5. **`docs/gpu/PHASE_0_FOUNDATION.md`**
   - Phase objectives and deliverables
   - Configuration options explained
   - Block partitioning strategy
   - Visualization examples

#### Configuration Schema (Initial)

```python
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class GPUForestConfig:
    """Configuration for GPU forest-of-octrees particle tracking."""

    # Block configuration (user-tunable)
    block_grid: Tuple[int, int, int] = (2, 2, 2)  # Start conservative
    max_octree_depth: int = 12

    # Field configuration
    field_name: str = "Displacement"  # Velocity field name in VTK
    auto_detect_field: bool = True     # Auto-detect if field_name not found

    # Timestep configuration
    revolution_cycle: Optional[Tuple[int, int]] = None  # Auto-detect or (120, 159)
    build_forest_from_timestep: int = -1  # -1 = auto-detect most refined

    # Memory configuration
    max_particles_per_block: int = 10000
    ghost_layer_thickness: int = 1

    # Performance tuning (Phase 8)
    skip_empty_blocks: bool = True
    enable_load_balancing: bool = False  # Static analysis only initially

    # Output configuration
    save_trajectory: bool = True
    trajectory_stride: int = 1  # Save every N timesteps
```

#### Success Criteria

- ✅ Clean branch with proper directory structure
- ✅ Configuration loads with sensible defaults
- ✅ Regular grid generator works for 2×2×2, 4×4×2, 8×8×4
- ✅ Neighbor topology correct (6-face, 12-edge, 8-corner connectivity)
- ✅ Visualization shows 3D blocks + projections
- ✅ Unit tests pass

#### Testing

**Unit Test Example**:
```python
def test_regular_grid_2x2x2():
    bounds = np.array([-0.03, 0.03, -0.023, 0.023, -0.01, 0.0])
    blocks = create_regular_grid(bounds, (2, 2, 2))

    assert len(blocks) == 8

    # Check block 0 (corner)
    assert blocks[0].block_id == 0
    assert len(blocks[0].neighbors) == 6
    assert blocks[0].neighbors[0] == 1  # +X neighbor
    assert blocks[0].neighbors[2] == 2  # +Y neighbor
    assert blocks[0].neighbors[4] == 4  # +Z neighbor

    # Check interior connectivity
    for block in blocks:
        for neighbor_id in block.neighbors:
            if neighbor_id != -1:  # Not boundary
                assert 0 <= neighbor_id < 8
```

**Integration Test** (Jupyter notebook):
- Load ThreadedA mesh timestep 159
- Create 2×2×2 forest grid
- Visualize blocks overlaid on mesh
- Verify blocks cover domain
- Check for empty blocks

---

### PHASE 1: CPU Block-Local Search
**Duration**: 5-7 days
**Depends On**: Phase 0

#### Objectives
1. Map mesh elements to forest blocks
2. Precompute element neighbor connectivity
3. Implement particle data structure with block_id and element_id
4. Three-tier search algorithm (CPU version)
5. Integrate Morton code encoding from previous implementation

#### Deliverables

**Files to Create**:

1. **`jaxtrace/gpu/forest/block_mapper.py`**
   - `assign_elements_to_blocks(mesh, blocks)` → element_to_block array
   - Uses element centroid to determine block
   - Handles ghost elements (elements near block boundaries)

2. **`jaxtrace/gpu/forest/element_neighbors.py`**
   - `build_element_adjacency(connectivity)` → neighbors array [N_elements, max_neighbors]
   - Extract face-adjacency from tetrahedral connectivity
   - Pad to fixed size (max 4 neighbors per tet face)

3. **`jaxtrace/gpu/particles.py`**
   - `ParticleData` class with fields:
     - positions [N, 3]
     - velocities [N, 3]
     - element_ids [N] ← cache
     - block_ids [N]
     - active_mask [N]
   - Methods: `update_positions()`, `rebatch_by_block()`, `to_jax()`

4. **`jaxtrace/gpu/forest/search_cpu.py`**
   - `search_element_three_tier(position, cached_element_id, neighbors, mesh, block)`
   - Level 1: Test cached element
   - Level 2: Test neighbor elements
   - Level 3: Block-local octree search (simple CPU version)

5. **`jaxtrace/gpu/morton.py`**
   - Copy from `jaxtrace/fields/morton_code.py` (previous implementation)
   - Functions: `encode_morton()`, `decode_morton()`, `hilbert_curve()`
   - Used for spatial indexing within blocks

6. **`tests/gpu/test_block_mapper.py`**
   - Test element assignment to blocks
   - Verify all elements assigned
   - Check ghost element detection

7. **`tests/gpu/test_element_neighbors.py`**
   - Test adjacency extraction
   - Verify neighbor counts (2-4 per element)
   - Check reciprocal neighbors

8. **`tests/gpu/test_search_cpu.py`**
   - Test three-tier search
   - Measure cache hit rates
   - Verify correctness vs brute-force search

9. **`docs/gpu/PHASE_1_BLOCK_SEARCH.md`**
   - Element-to-block mapping strategy
   - Neighbor precomputation algorithm
   - Three-tier search explanation
   - Cache hit rate analysis

#### Key Algorithms

**Element-to-Block Assignment**:
```python
def assign_elements_to_blocks(mesh, blocks):
    """Assign each element to a block based on centroid."""
    element_to_block = np.zeros(mesh.n_cells, dtype=np.int32)

    for elem_id in range(mesh.n_cells):
        # Compute element centroid
        node_ids = mesh.connectivity[elem_id]
        centroid = mesh.positions[node_ids].mean(axis=0)

        # Find containing block
        block_id = find_block_containing_point(centroid, blocks)
        element_to_block[elem_id] = block_id

    return element_to_block
```

**Element Neighbor Extraction**:
```python
def build_element_adjacency(connectivity):
    """Build face-adjacency graph for tetrahedral mesh."""
    # Tetrahedral faces: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    face_to_elements = {}  # {sorted_face: [elem_ids]}

    for elem_id, nodes in enumerate(connectivity):
        faces = [
            tuple(sorted([nodes[0], nodes[1], nodes[2]])),
            tuple(sorted([nodes[0], nodes[1], nodes[3]])),
            tuple(sorted([nodes[0], nodes[2], nodes[3]])),
            tuple(sorted([nodes[1], nodes[2], nodes[3]]))
        ]
        for face in faces:
            if face not in face_to_elements:
                face_to_elements[face] = []
            face_to_elements[face].append(elem_id)

    # Build adjacency from shared faces
    neighbors = np.full((len(connectivity), 4), -1, dtype=np.int32)

    for elem_id in range(len(connectivity)):
        neighbor_set = set()
        # Find all elements sharing a face
        for face_elems in face_to_elements.values():
            if elem_id in face_elems:
                neighbor_set.update(face_elems)
        neighbor_set.discard(elem_id)

        # Store up to 4 neighbors
        for i, neighbor_id in enumerate(list(neighbor_set)[:4]):
            neighbors[elem_id, i] = neighbor_id

    return neighbors
```

**Three-Tier Search** (CPU version):
```python
def search_element_three_tier(position, cached_element_id, neighbors, mesh, block):
    """
    Three-tier element search.

    Returns:
        element_id: Containing element ID, or -1 if not found
        search_level: 0 (cached), 1 (neighbor), 2 (tree)
    """
    # Level 1: Check cached element
    if cached_element_id != -1:
        if point_in_element(position, cached_element_id, mesh):
            return cached_element_id, 0

    # Level 2: Check neighbor elements
    if cached_element_id != -1:
        for neighbor_id in neighbors[cached_element_id]:
            if neighbor_id != -1:
                if point_in_element(position, neighbor_id, mesh):
                    return neighbor_id, 1

    # Level 3: Block-local search (simple linear search for now)
    block_elements = get_block_elements(block.block_id, mesh)
    for elem_id in block_elements:
        if point_in_element(position, elem_id, mesh):
            return elem_id, 2

    return -1, 2  # Not found
```

#### Success Criteria

- ✅ All mesh elements assigned to blocks (no orphans)
- ✅ Element neighbor lists built (average 2-4 neighbors per tet)
- ✅ 85%+ cache hit rate on synthetic test (1000 particles, 10 timesteps)
- ✅ Three-tier search faster than brute-force
- ✅ Morton code encoding works

#### Testing

**Integration Test** (1000 particles, ThreadedA mesh):
```python
def test_particle_tracking_cpu_phase1():
    # Load mesh
    mesh = load_threadeda_mesh(timestep=159)

    # Create forest
    blocks = create_regular_grid(mesh.bounds, (2, 2, 2))

    # Map elements
    element_to_block = assign_elements_to_blocks(mesh, blocks)
    neighbors = build_element_adjacency(mesh.connectivity)

    # Seed particles
    particles = ParticleData(n_particles=1000)
    particles.positions = seed_particles_uniform(mesh.bounds, 1000)
    particles.element_ids = -1  # Initialize cache

    # Track for 10 timesteps (simple Euler)
    cache_hits = [0, 0, 0]  # [cached, neighbor, tree]

    for step in range(10):
        for i in range(1000):
            pos = particles.positions[i]
            cached_elem = particles.element_ids[i]

            # Search
            elem_id, level = search_element_three_tier(
                pos, cached_elem, neighbors, mesh, blocks[0]
            )

            particles.element_ids[i] = elem_id
            cache_hits[level] += 1

            # Update position (simple Euler for testing)
            velocity = interpolate_field(pos, elem_id, mesh)
            particles.positions[i] += velocity * dt

    # Check cache hit rate
    total = sum(cache_hits)
    cache_rate = (cache_hits[0] + cache_hits[1]) / total

    assert cache_rate > 0.85, f"Cache hit rate too low: {cache_rate:.2%}"
    print(f"Cache hits: Level 0 ({cache_hits[0]/total:.1%}), "
          f"Level 1 ({cache_hits[1]/total:.1%}), "
          f"Level 2 ({cache_hits[2]/total:.1%})")
```

---

### PHASE 2: JAX Data Structures & Memory
**Duration**: 3-5 days
**Depends On**: Phase 1

#### Objectives
1. Convert block metadata to JAX DeviceArrays
2. Flatten element neighbor lists (fixed max_neighbors)
3. Allocate ghost region buffers
4. Profile GPU memory usage
5. Analyze load balance across blocks

#### Deliverables

**Files to Create**:

1. **`jaxtrace/gpu/forest/jax_arrays.py`**
   - `convert_blocks_to_jax(blocks)` → JAX arrays
   - `convert_mesh_to_jax(mesh)` → JAX arrays
   - `convert_neighbors_to_jax(neighbors)` → padded array
   - All arrays flat, static-sized, float32/int32

2. **`jaxtrace/gpu/forest/ghost_buffers.py`**
   - `allocate_ghost_buffers(blocks, ghost_thickness)` → ghost arrays
   - Identify ghost elements (elements within ghost_thickness of block boundary)
   - Pre-allocate buffers for particle exchange

3. **`scripts/profile_gpu_memory.py`**
   - Load forest + mesh into GPU
   - Measure VRAM usage at each stage
   - Report memory breakdown
   - Test with varying block counts (8, 32, 64, 128)

4. **`scripts/analyze_load_balance.py`**
   - Compute cells per block
   - Report min, max, mean, std
   - Calculate imbalance factor (max / mean)
   - Visualize distribution

5. **`tests/gpu/test_jax_arrays.py`**
   - Test array conversions
   - Verify shapes and dtypes
   - Check GPU placement (device_put)

6. **`docs/gpu/PHASE_2_JAX_STRUCTURES.md`**
   - JAX array layout
   - Memory profiling results
   - Load balance analysis
   - Ghost region strategy

#### Key Data Structures (JAX)

**Block Metadata**:
```python
# After conversion to JAX arrays
block_bounds = jnp.array([...], dtype=jnp.float32)  # [B, 6]
block_centers = jnp.array([...], dtype=jnp.float32)  # [B, 3]
block_offsets = jnp.array([...], dtype=jnp.int32)    # [B+1] (CSR-style)
neighbor_ids = jnp.array([...], dtype=jnp.int32)     # [B, 6]
```

**Mesh Data**:
```python
cell_positions = jnp.array([...], dtype=jnp.float32)    # [N_nodes, 3]
cell_connectivity = jnp.array([...], dtype=jnp.int32)  # [N_cells, 4]
element_neighbors = jnp.array([...], dtype=jnp.int32)  # [N_cells, 4] (padded)
element_to_block = jnp.array([...], dtype=jnp.int32)   # [N_cells]
```

**Particle Data**:
```python
positions = jnp.array([...], dtype=jnp.float32)      # [N, 3]
velocities = jnp.array([...], dtype=jnp.float32)     # [N, 3]
element_ids = jnp.array([...], dtype=jnp.int32)      # [N] (cache)
block_ids = jnp.array([...], dtype=jnp.int32)        # [N]
active_mask = jnp.array([...], dtype=jnp.bool_)      # [N]
```

**Ghost Buffers**:
```python
# Pre-allocated buffers for inter-block particle exchange
ghost_positions = jnp.zeros((N_ghost_max, 3), dtype=jnp.float32)
ghost_element_ids = jnp.zeros(N_ghost_max, dtype=jnp.int32)
ghost_counts = jnp.zeros(B, dtype=jnp.int32)  # Particles per block boundary
```

#### Memory Profiling

**Script Output** (example for 2×2×2 grid):
```
GPU Memory Profiling Report
===========================
Block Count: 8 (2×2×2 grid)
Mesh Size: 1296 cells, 2301 nodes

Static Data:
  Block metadata:      12 KB
  Mesh positions:      27 KB
  Connectivity:        20 KB
  Element neighbors:   20 KB
  Element-to-block:     5 KB
  Field values (1 ts): 27 KB
  Total Static:       111 KB

Dynamic Data (100K particles):
  Positions:         1.2 MB
  Velocities:        1.2 MB
  Element IDs:       400 KB
  Block IDs:         400 KB
  Active mask:       100 KB
  Total Dynamic:     3.3 MB

Ghost Buffers:
  Max ghost particles: 5000
  Buffer size:         160 KB

Total GPU Memory:      3.6 MB
Available VRAM:        3.7 GB
Utilization:          0.1%

✅ Memory budget: OK (plenty of headroom)
```

#### Load Balance Analysis

**Script Output**:
```
Load Balance Analysis
=====================
Block Grid: 2×2×2 (8 blocks)

Cells per Block:
  Block 0: 142 cells
  Block 1: 165 cells
  Block 2: 158 cells
  Block 3: 171 cells
  Block 4: 154 cells
  Block 5: 168 cells
  Block 6: 149 cells
  Block 7: 189 cells

Statistics:
  Mean:   162 cells
  Std:     14 cells
  Min:    142 cells (Block 0)
  Max:    189 cells (Block 7)
  Imbalance Factor: 1.17× (max / mean)

✅ Load balance: GOOD (imbalance < 2×)

Recommendation: No dynamic splitting needed
```

#### Success Criteria

- ✅ All arrays fit in GPU memory (<1 GB for structure)
- ✅ Max load imbalance <2× (most loaded block has <2× cells vs mean)
- ✅ All arrays static (no dynamic slicing)
- ✅ Ghost buffers allocated
- ✅ Memory profiling script works for 8, 32, 64 blocks

#### Testing

**Memory Test**:
```python
def test_gpu_memory_full_forest():
    # Create 2×2×2 forest
    blocks = create_regular_grid(bounds, (2, 2, 2))

    # Convert to JAX
    block_arrays = convert_blocks_to_jax(blocks)
    mesh_arrays = convert_mesh_to_jax(mesh)

    # Allocate particles
    particles = jnp.zeros((100000, 3), dtype=jnp.float32)

    # Check GPU memory
    mem_info = get_gpu_memory_info()

    assert mem_info['used'] < 1e9, "Structure uses > 1GB"
    assert mem_info['available'] > 2e9, "< 2GB available for particles"
```

---

### PHASE 3: GPU Single-Particle Kernel
**Duration**: 7-10 days
**Depends On**: Phase 2

#### Objectives
1. Implement JAX-JIT single particle update function
2. Three-tier search in pure JAX
3. Integrate element testing and FEM interpolation from previous implementation
4. RK4 time integration (pure JAX)
5. Temporal interpolation between timesteps

#### Deliverables

**Files to Create**:

1. **`jaxtrace/gpu/kernels/particle_update.py`**
   - `update_particle_single(particle_state, static_data, field_data, dt)` → new_state
   - Decorated with `@jax.jit`
   - Stateless function (all inputs explicit)

2. **`jaxtrace/gpu/kernels/search_jax.py`**
   - `search_cached_element(pos, elem_id, connectivity, positions)` → bool
   - `search_neighbors(pos, elem_id, neighbors, connectivity, positions)` → elem_id
   - `search_block_elements(pos, block_elements, connectivity, positions)` → elem_id
   - Pure JAX, GPU-compilable

3. **`jaxtrace/gpu/kernels/rk4.py`**
   - `rk4_step(position, velocity_func, dt)` → new_position
   - Classic RK4 integrator
   - Calls field sampling 4 times per step

4. **`jaxtrace/gpu/interpolation.py`**
   - Copy from `jaxtrace/fields/interpolator_jax_simple.py`
   - `interpolate_fem_tet(position, elem_id, connectivity, node_positions, field_values)` → interpolated_value
   - Barycentric coordinates for tetrahedral FEM

5. **`jaxtrace/gpu/element_testing.py`**
   - Copy from `jaxtrace/fields/element_testing_jax.py`
   - `point_in_tet(position, elem_id, connectivity, node_positions)` → bool
   - Barycentric test for tetrahedra

6. **`tests/gpu/test_particle_kernel.py`**
   - Test single particle update
   - Verify JIT compilation
   - Check interpolation accuracy

7. **`examples/gpu/phase3_single_particle.ipynb`**
   - Jupyter integration test
   - Visualize single particle trajectory
   - Compare with CPU version

8. **`docs/gpu/PHASE_3_GPU_KERNEL.md`**
   - Kernel architecture
   - Search algorithm (JAX version)
   - RK4 integration details
   - Performance analysis

#### Key Functions

**Single Particle Update Kernel**:
```python
@jax.jit
def update_particle_single(particle_state, static_data, field_data, dt):
    """
    Update a single particle for one timestep using RK4.

    Args:
        particle_state: dict with 'position', 'velocity', 'element_id', 'block_id'
        static_data: dict with mesh arrays (connectivity, positions, neighbors, etc.)
        field_data: dict with velocity fields (left, right timesteps for interpolation)
        dt: timestep size

    Returns:
        new_particle_state: updated particle state
        diagnostics: dict with search_level, cache_hit, etc.
    """
    position = particle_state['position']
    cached_elem = particle_state['element_id']
    block_id = particle_state['block_id']

    # Three-tier search
    elem_id, search_level = search_element_jax(
        position, cached_elem, static_data, block_id
    )

    # RK4 integration
    def velocity_func(pos):
        # Interpolate velocity at position
        vel = interpolate_fem_tet(
            pos, elem_id,
            static_data['connectivity'],
            static_data['positions'],
            field_data['velocity']
        )
        return vel

    new_position = rk4_step(position, velocity_func, dt)
    new_velocity = velocity_func(new_position)

    # Determine new block_id
    new_block_id = find_block_containing_point(new_position, static_data['block_bounds'])

    # Update state
    new_state = {
        'position': new_position,
        'velocity': new_velocity,
        'element_id': elem_id,  # Cache for next step
        'block_id': new_block_id
    }

    diagnostics = {
        'search_level': search_level,
        'cache_hit': search_level == 0,
        'block_changed': new_block_id != block_id
    }

    return new_state, diagnostics
```

**Three-Tier Search (JAX)**:
```python
@jax.jit
def search_element_jax(position, cached_elem_id, static_data, block_id):
    """
    Three-tier element search in pure JAX.

    Returns:
        elem_id: Containing element ID
        search_level: 0 (cached), 1 (neighbor), 2 (tree)
    """
    connectivity = static_data['connectivity']
    positions = static_data['positions']
    neighbors = static_data['neighbors']
    block_elements = static_data['block_elements'][block_id]

    # Level 0: Check cached element
    def check_cached():
        in_elem = point_in_tet(position, cached_elem_id, connectivity, positions)
        return jnp.where(in_elem, cached_elem_id, -1), jnp.where(in_elem, 0, -1)

    # Level 1: Check neighbors
    def check_neighbors():
        def scan_neighbor(carry, neighbor_id):
            found_elem, found_level = carry
            # Skip if already found or invalid neighbor
            skip = (found_elem != -1) | (neighbor_id == -1)

            in_elem = jnp.where(
                skip,
                False,
                point_in_tet(position, neighbor_id, connectivity, positions)
            )

            new_elem = jnp.where(in_elem, neighbor_id, found_elem)
            new_level = jnp.where(in_elem, 1, found_level)

            return (new_elem, new_level), None

        neighbor_list = neighbors[cached_elem_id]
        (elem, level), _ = jax.lax.scan(scan_neighbor, (-1, -1), neighbor_list)
        return elem, level

    # Level 2: Block-local search
    def check_block():
        def scan_block_elem(carry, elem_id):
            found_elem, found_level = carry
            skip = found_elem != -1

            in_elem = jnp.where(
                skip,
                False,
                point_in_tet(position, elem_id, connectivity, positions)
            )

            new_elem = jnp.where(in_elem, elem_id, found_elem)
            new_level = jnp.where(in_elem, 2, found_level)

            return (new_elem, new_level), None

        (elem, level), _ = jax.lax.scan(scan_block_elem, (-1, -1), block_elements)
        return elem, level

    # Execute tiers sequentially with early exit
    elem_id, level = jax.lax.cond(
        cached_elem_id != -1,
        check_cached,
        lambda: (-1, -1)
    )

    elem_id, level = jax.lax.cond(
        (elem_id == -1) & (cached_elem_id != -1),
        check_neighbors,
        lambda: (elem_id, level)
    )

    elem_id, level = jax.lax.cond(
        elem_id == -1,
        check_block,
        lambda: (elem_id, level)
    )

    return elem_id, level
```

**RK4 Integrator**:
```python
@jax.jit
def rk4_step(position, velocity_func, dt):
    """
    Single RK4 integration step.

    Args:
        position: Current position [3]
        velocity_func: Function pos → velocity
        dt: Timestep

    Returns:
        new_position: Updated position [3]
    """
    k1 = velocity_func(position)
    k2 = velocity_func(position + 0.5 * dt * k1)
    k3 = velocity_func(position + 0.5 * dt * k2)
    k4 = velocity_func(position + dt * k3)

    new_position = position + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return new_position
```

#### Success Criteria

- ✅ Single particle update compiles with `@jax.jit`
- ✅ Interpolation accuracy <1% error vs CPU
- ✅ RK4 matches CPU integrator (4th order accuracy)
- ✅ Search finds correct element
- ✅ No runtime errors during JIT compilation

#### Testing

**Unit Test**:
```python
def test_single_particle_update():
    # Set up test data
    mesh = load_threadeda_mesh(timestep=159)
    static_data = convert_mesh_to_jax(mesh)

    field_data = {
        'velocity': jnp.asarray(mesh.get_field('Displacement'))
    }

    # Seed particle in known element
    particle_state = {
        'position': jnp.array([0.0, 0.0, -0.005]),
        'velocity': jnp.zeros(3),
        'element_id': 100,  # Known element
        'block_id': 0
    }

    # Update particle
    new_state, diagnostics = update_particle_single(
        particle_state, static_data, field_data, dt=0.001
    )

    # Check outputs
    assert new_state['position'].shape == (3,)
    assert diagnostics['search_level'] in [0, 1, 2]
    assert new_state['element_id'] != -1, "Element not found"

    # Compare with CPU version (not shown)
    cpu_position = update_particle_cpu(particle_state, mesh, dt=0.001)
    error = jnp.linalg.norm(new_state['position'] - cpu_position)
    assert error < 1e-5, f"Position error too large: {error}"
```

**Integration Test (Jupyter)**:
- Track single particle for 100 timesteps
- Plot 3D trajectory
- Overlay on mesh
- Compare with CPU tracker trajectory
- Measure cache hit rate

---

### PHASE 4: Vectorization (vmap over Particles)
**Duration**: 3-5 days
**Depends On**: Phase 3

#### Objectives
1. Vectorize particle update using `vmap`
2. Handle variable particles-per-block (padding + masking)
3. Benchmark single-block GPU vs CPU
4. Profile GPU utilization

#### Deliverables

**Files to Create**:

1. **`jaxtrace/gpu/kernels/block_update.py`**
   - `update_block_particles(particles, static_data, field_data, dt)` → new_particles
   - Uses `jax.vmap(update_particle_single, in_axes=(0, None, None, None))`
   - Handles padding for variable particle counts

2. **`scripts/benchmark_single_block.py`**
   - Compare GPU vs CPU for single block
   - Vary particle counts: 100, 1K, 10K, 50K
   - Measure throughput (particles/second)
   - Plot speedup curve

3. **`tests/gpu/test_vectorization.py`**
   - Test vmap works correctly
   - Verify output shapes
   - Check masking for padded particles

4. **`docs/gpu/PHASE_4_VECTORIZATION.md`**
   - vmap explanation
   - Padding strategy
   - Benchmark results
   - GPU utilization analysis

#### Key Functions

**Vectorized Block Update**:
```python
# Create vectorized version of single particle update
update_particles_vectorized = jax.vmap(
    update_particle_single,
    in_axes=(0, None, None, None)  # vmap over first axis (particles)
)

@jax.jit
def update_block_particles(particle_states, static_data, field_data, dt, max_particles):
    """
    Update all particles in a single block (vectorized).

    Args:
        particle_states: dict with arrays [N_actual, ...]
        static_data: mesh arrays (shared across particles)
        field_data: velocity field (shared)
        dt: timestep
        max_particles: Maximum particles (for padding)

    Returns:
        new_particle_states: Updated particles [N_actual, ...]
        diagnostics: Aggregated diagnostics
    """
    N_actual = particle_states['position'].shape[0]

    # Pad to max_particles for fixed shape
    def pad_array(arr):
        pad_size = max_particles - N_actual
        if pad_size > 0:
            pad_shape = (pad_size,) + arr.shape[1:]
            padding = jnp.zeros(pad_shape, dtype=arr.dtype)
            return jnp.concatenate([arr, padding], axis=0)
        return arr

    padded_states = {
        'position': pad_array(particle_states['position']),
        'velocity': pad_array(particle_states['velocity']),
        'element_id': pad_array(particle_states['element_id']),
        'block_id': pad_array(particle_states['block_id'])
    }

    # Create active mask
    active_mask = jnp.arange(max_particles) < N_actual

    # Vectorized update
    new_states, diagnostics = update_particles_vectorized(
        padded_states, static_data, field_data, dt
    )

    # Mask out padding
    def mask_array(arr):
        return arr[:N_actual]

    new_states_trimmed = {
        'position': mask_array(new_states['position']),
        'velocity': mask_array(new_states['velocity']),
        'element_id': mask_array(new_states['element_id']),
        'block_id': mask_array(new_states['block_id'])
    }

    # Aggregate diagnostics
    agg_diagnostics = {
        'cache_hits': jnp.sum(diagnostics['cache_hit'] & active_mask),
        'total_particles': N_actual,
        'cache_hit_rate': jnp.mean(diagnostics['cache_hit'][active_mask])
    }

    return new_states_trimmed, agg_diagnostics
```

#### Benchmarking

**Benchmark Script**:
```python
def benchmark_single_block():
    # Load mesh
    mesh = load_threadeda_mesh(timestep=159)
    static_data = convert_mesh_to_jax(mesh)
    field_data = {'velocity': jnp.asarray(mesh.get_field('Displacement'))}

    particle_counts = [100, 1000, 10000, 50000]
    results = {'cpu': [], 'gpu': []}

    for N in particle_counts:
        # Seed particles
        particles = seed_particles_uniform(mesh.bounds, N)
        particle_states = {
            'position': jnp.array(particles),
            'velocity': jnp.zeros((N, 3)),
            'element_id': jnp.full(N, -1, dtype=jnp.int32),
            'block_id': jnp.zeros(N, dtype=jnp.int32)
        }

        # GPU benchmark
        start = time.time()
        for _ in range(10):  # 10 timesteps
            particle_states, _ = update_block_particles(
                particle_states, static_data, field_data, dt=0.001, max_particles=N
            )
        jax.block_until_ready(particle_states['position'])  # Ensure GPU completes
        gpu_time = time.time() - start

        # CPU benchmark (reference implementation)
        cpu_time = benchmark_cpu_version(particles, mesh, n_steps=10)

        results['cpu'].append(cpu_time)
        results['gpu'].append(gpu_time)

        print(f"N={N:5d}: CPU={cpu_time:.3f}s, GPU={gpu_time:.3f}s, "
              f"Speedup={cpu_time/gpu_time:.1f}×")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(particle_counts, np.array(results['cpu']) / np.array(results['gpu']),
             marker='o', linewidth=2)
    plt.xlabel('Number of Particles')
    plt.ylabel('Speedup (GPU vs CPU)')
    plt.title('Single-Block Particle Tracking Speedup')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.savefig('phase4_speedup.png', dpi=150)
```

**Expected Results**:
```
Single-Block Benchmark Results
==============================
N=  100: CPU=0.120s, GPU=0.080s, Speedup=1.5×
N= 1000: CPU=1.200s, GPU=0.090s, Speedup=13.3×
N=10000: CPU=12.00s, GPU=0.150s, Speedup=80.0×
N=50000: CPU=60.00s, GPU=0.400s, Speedup=150.0×

✅ GPU shows strong scaling with particle count
```

#### Success Criteria

- ✅ 10× speedup vs CPU for 1000 particles in single block
- ✅ GPU utilization >60% during kernel execution
- ✅ vmap compiles successfully
- ✅ Padding/masking works correctly
- ✅ No accuracy regression vs Phase 3

#### Testing

**Test Vectorization**:
```python
def test_vmap_correctness():
    # Set up test with 10 particles
    N = 10
    particles = {
        'position': jnp.array([[0.001 * i, 0.0, -0.005] for i in range(N)]),
        'velocity': jnp.zeros((N, 3)),
        'element_id': jnp.full(N, -1, dtype=jnp.int32),
        'block_id': jnp.zeros(N, dtype=jnp.int32)
    }

    # Update all at once
    new_particles, diag = update_block_particles(
        particles, static_data, field_data, dt=0.001, max_particles=20
    )

    # Compare with sequential updates
    for i in range(N):
        single_state = {k: v[i] for k, v in particles.items()}
        single_new, _ = update_particle_single(single_state, static_data, field_data, 0.001)

        # Check match
        pos_error = jnp.linalg.norm(new_particles['position'][i] - single_new['position'])
        assert pos_error < 1e-6, f"Particle {i} position mismatch"
```

---

### PHASE 5: Multi-Block Parallelism
**Duration**: 5-7 days
**Depends On**: Phase 4
**STATUS**: **Minimum Viable Product (MVP)**

#### Objectives
1. Implement nested `vmap` over blocks and particles
2. Particle rebatching (sort by block_id after each step)
3. Handle particles crossing block boundaries
4. Benchmark multi-block performance

#### Deliverables

**Files to Create**:

1. **`jaxtrace/gpu/kernels/multi_block_update.py`**
   - `update_all_blocks(all_particles, static_data, field_data, dt)` → new_particles
   - Nested vmap: `vmap(update_block_particles, in_axes=(0, 0, None, None))`
   - Per-block static data (if using block-local storage)

2. **`jaxtrace/gpu/rebatching.py`**
   - `rebatch_particles_by_block(particles, n_blocks)` → particles_per_block
   - Sort particles by block_id
   - Return list/array of per-block particles
   - Use JAX ops: `jax.ops.segment_sum`, `jnp.argsort`

3. **`scripts/benchmark_multi_block.py`**
   - Benchmark across all blocks
   - Measure GPU utilization
   - Compare with single-block performance
   - Scaling analysis

4. **`tests/gpu/test_multi_block.py`**
   - Test particles in multiple blocks
   - Verify particle migration
   - Check no particles lost

5. **`examples/gpu/phase5_multi_block.ipynb`**
   - Integration test with visualization
   - Track 10K particles across 8 blocks
   - Plot particle distribution per block over time

6. **`docs/gpu/PHASE_5_MULTI_BLOCK.md`**
   - Multi-block architecture
   - Rebatching algorithm
   - Benchmark results
   - **MVP milestone achieved**

#### Key Functions

**Multi-Block Update** (Nested vmap):
```python
# Vectorize over blocks
update_all_blocks_vectorized = jax.vmap(
    update_block_particles,
    in_axes=(0, 0, None, None, None)  # vmap over blocks (axis 0)
)

@jax.jit
def update_all_blocks(particles_per_block, block_static_data, field_data, dt, max_particles_per_block):
    """
    Update particles in all blocks in parallel.

    Args:
        particles_per_block: list of dicts [B] (particles in each block)
        block_static_data: per-block mesh data [B]
        field_data: shared field data
        dt: timestep
        max_particles_per_block: Max particles per block (for padding)

    Returns:
        new_particles_per_block: Updated particles [B]
        diagnostics: Per-block diagnostics
    """
    # Stack particles into [B, N_max, ...] arrays
    stacked_particles = stack_particles_per_block(particles_per_block, max_particles_per_block)

    # Nested vmap: blocks × particles
    new_particles_stacked, diagnostics = update_all_blocks_vectorized(
        stacked_particles, block_static_data, field_data, dt, max_particles_per_block
    )

    # Unstack back to per-block lists
    new_particles_per_block = unstack_particles(new_particles_stacked)

    return new_particles_per_block, diagnostics
```

**Particle Rebatching**:
```python
@jax.jit
def rebatch_particles_by_block(all_particles, n_blocks):
    """
    Sort particles by block_id and group into per-block arrays.

    Args:
        all_particles: dict with arrays [N_total, ...]
        n_blocks: Number of blocks

    Returns:
        particles_per_block: list of dicts [B] (particles in each block)
        counts: Particles per block [B]
    """
    N_total = all_particles['position'].shape[0]
    block_ids = all_particles['block_id']

    # Sort by block_id
    sort_indices = jnp.argsort(block_ids)

    sorted_particles = {
        'position': all_particles['position'][sort_indices],
        'velocity': all_particles['velocity'][sort_indices],
        'element_id': all_particles['element_id'][sort_indices],
        'block_id': all_particles['block_id'][sort_indices]
    }

    # Count particles per block
    counts = jnp.array([
        jnp.sum(block_ids == b) for b in range(n_blocks)
    ])

    # Split into per-block arrays using cumulative offsets
    offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(counts)])

    particles_per_block = []
    for b in range(n_blocks):
        start = offsets[b]
        end = offsets[b + 1]

        block_particles = {
            'position': sorted_particles['position'][start:end],
            'velocity': sorted_particles['velocity'][start:end],
            'element_id': sorted_particles['element_id'][start:end],
            'block_id': sorted_particles['block_id'][start:end]
        }
        particles_per_block.append(block_particles)

    return particles_per_block, counts
```

**Particle Migration Handling**:
```python
def track_with_rebatching(particles, static_data, field_data, dt, n_steps, n_blocks):
    """
    Track particles with periodic rebatching.

    Rebatches every step to handle particles crossing block boundaries.
    """
    # Initial batching
    particles_per_block, counts = rebatch_particles_by_block(particles, n_blocks)

    for step in range(n_steps):
        # Update all blocks
        particles_per_block, diag = update_all_blocks(
            particles_per_block, static_data, field_data, dt, max_particles_per_block=10000
        )

        # Merge all particles
        all_particles = merge_particles_from_blocks(particles_per_block)

        # Rebatch by new block_ids
        particles_per_block, counts = rebatch_particles_by_block(all_particles, n_blocks)

        # Log diagnostics
        print(f"Step {step}: Particles per block: {counts}")

    return all_particles
```

#### Benchmarking

**Multi-Block Benchmark**:
```
Multi-Block Benchmark (8 blocks, 2×2×2 grid)
============================================
Particle counts: 10K, 50K, 100K
Timesteps: 40

N=10K:
  Single-block equivalent: 15.2s
  Multi-block (8 blocks):   2.1s
  Speedup: 7.2× (near-ideal: 8×)
  GPU utilization: 82%

N=50K:
  Single-block equivalent: 76.0s
  Multi-block (8 blocks):   9.8s
  Speedup: 7.8×
  GPU utilization: 89%

N=100K:
  Single-block equivalent: 152.0s
  Multi-block (8 blocks):  18.5s
  Speedup: 8.2×
  GPU utilization: 91%

✅ Multi-block scaling: EXCELLENT
✅ GPU utilization: >80% for all tests
```

#### Success Criteria

- ✅ All 8 blocks update in parallel
- ✅ Particles correctly migrate between blocks
- ✅ 50× speedup vs CPU for 50K particles
- ✅ No lost particles (conservation)
- ✅ GPU utilization >80%
- ✅ **MVP ACHIEVED**: Functional GPU tracker

#### Testing

**Integration Test**:
```python
def test_multi_block_tracking():
    # Load mesh and create 8 blocks
    mesh = load_threadeda_mesh(timestep=159)
    blocks = create_regular_grid(mesh.bounds, (2, 2, 2))
    static_data = convert_to_jax(mesh, blocks)
    field_data = {'velocity': jnp.asarray(mesh.get_field('Displacement'))}

    # Seed 10K particles across domain
    particles = seed_particles_uniform(mesh.bounds, 10000)
    particle_states = initialize_particle_states(particles)

    # Track for 40 timesteps
    final_particles = track_with_rebatching(
        particle_states, static_data, field_data, dt=0.001, n_steps=40, n_blocks=8
    )

    # Verify conservation
    assert final_particles['position'].shape[0] == 10000, "Particles lost!"

    # Check particles moved
    displacement = jnp.linalg.norm(
        final_particles['position'] - particle_states['position'], axis=1
    )
    mean_disp = jnp.mean(displacement)
    assert mean_disp > 1e-6, f"Particles didn't move: {mean_disp}"

    print(f"✅ Tracked 10K particles for 40 steps")
    print(f"   Mean displacement: {mean_disp:.6f}")
    print(f"   Particles per block: {counts}")
```

---

### PHASE 6: Time Marching (lax.scan)
**Duration**: 3-5 days
**Depends On**: Phase 5

#### Objectives
1. Integrate time stepping with `lax.scan`
2. Minimal scan carry (particles only)
3. Static mesh/field data passed as constants
4. Timestep streaming (load 5-10 timesteps at a time)

#### Deliverables

**Files to Create**:

1. **`jaxtrace/gpu/time_marching.py`**
   - `track_particles_lax_scan(particles, timesteps, static_data, field_loader, dt)` → trajectory
   - Uses `jax.lax.scan` for time loop
   - Minimal carry to prevent memory growth

2. **`jaxtrace/gpu/field_loader.py`**
   - `FieldLoader` class for streaming timesteps
   - Loads 5-10 timesteps into GPU memory at a time
   - Swaps timesteps as tracking progresses

3. **`tests/gpu/test_time_marching.py`**
   - Test lax.scan integration
   - Verify memory usage stable over time
   - Check trajectory output

4. **`docs/gpu/PHASE_6_TIME_MARCHING.md`**
   - lax.scan explanation
   - Carry minimization strategy
   - Timestep streaming details
   - Memory profiling results

#### Key Functions

**Time Marching with lax.scan**:
```python
@jax.jit
def track_particles_lax_scan(particles_init, timesteps, static_data, field_cache, dt, n_blocks):
    """
    Track particles through time using lax.scan.

    Args:
        particles_init: Initial particle states
        timesteps: Array of timestep indices to track
        static_data: Mesh data (constant, not in carry)
        field_cache: Pre-loaded velocity fields [T_cache, N_nodes, 3]
        dt: Timestep size
        n_blocks: Number of forest blocks

    Returns:
        trajectories: Particle positions over time [T, N, 3]
        final_particles: Final particle states
    """
    def step_function(carry, t):
        """
        Single timestep update.

        carry: Only particle states (minimal memory)
        t: Timestep index

        Returns:
            new_carry: Updated particle states
            output: Particle positions (for trajectory)
        """
        particles = carry

        # Get velocity field for this timestep
        # (temporal interpolation if between timesteps)
        field_data = get_field_at_timestep(t, field_cache, static_data['times'])

        # Update all blocks
        particles_per_block, _ = rebatch_particles_by_block(particles, n_blocks)

        new_particles_per_block, diagnostics = update_all_blocks(
            particles_per_block,
            static_data['blocks'],
            field_data,
            dt,
            max_particles_per_block=10000
        )

        # Merge particles
        new_particles = merge_particles_from_blocks(new_particles_per_block)

        # Output: positions only (for trajectory)
        output = new_particles['position']

        return new_particles, output

    # Run scan
    final_particles, trajectory = jax.lax.scan(
        step_function,
        particles_init,
        timesteps
    )

    return trajectory, final_particles
```

**Field Loader (Streaming)**:
```python
class FieldLoader:
    """
    Stream velocity fields to GPU in batches.

    Keeps only 5-10 timesteps in GPU memory at a time.
    """
    def __init__(self, mesh_path, field_name, cache_size=10):
        self.mesh_path = mesh_path
        self.field_name = field_name
        self.cache_size = cache_size
        self.cache = {}  # {timestep: field_array}

    def load_timestep_range(self, start, end):
        """Load timesteps [start, end) into cache."""
        fields = []
        for t in range(start, end):
            mesh = load_threadeda_mesh(timestep=t)
            field = mesh.get_field(self.field_name)
            fields.append(field)

        # Upload to GPU
        field_cache = jnp.array(fields, dtype=jnp.float32)

        return field_cache

    def get_field_batch(self, timestep_idx, n_timesteps):
        """
        Get batch of fields centered on timestep_idx.

        Returns:
            field_cache: [cache_size, N_nodes, 3]
            offset: Timestep offset in cache
        """
        # Determine range
        start = max(0, timestep_idx - self.cache_size // 2)
        end = min(n_timesteps, start + self.cache_size)

        # Load if not cached
        cache_key = (start, end)
        if cache_key not in self.cache:
            self.cache.clear()  # Clear old cache
            self.cache[cache_key] = self.load_timestep_range(start, end)

        return self.cache[cache_key], start
```

**Temporal Interpolation**:
```python
@jax.jit
def get_field_at_timestep(t, field_cache, times, cache_offset):
    """
    Get velocity field at timestep t with temporal interpolation.

    Args:
        t: Timestep index (may be fractional for sub-stepping)
        field_cache: Cached fields [T_cache, N_nodes, 3]
        times: Timestep times [T_total]
        cache_offset: Offset of cache in global timesteps

    Returns:
        field: Interpolated velocity field [N_nodes, 3]
    """
    # Find bracketing timesteps
    t_floor = jnp.floor(t).astype(jnp.int32)
    t_ceil = t_floor + 1
    alpha = t - t_floor

    # Get indices in cache
    idx_floor = t_floor - cache_offset
    idx_ceil = t_ceil - cache_offset

    # Clamp to cache range
    T_cache = field_cache.shape[0]
    idx_floor = jnp.clip(idx_floor, 0, T_cache - 1)
    idx_ceil = jnp.clip(idx_ceil, 0, T_cache - 1)

    # Linear interpolation
    field_floor = field_cache[idx_floor]
    field_ceil = field_cache[idx_ceil]

    field = (1 - alpha) * field_floor + alpha * field_ceil

    return field
```

#### Success Criteria

- ✅ Stable execution for 160 timesteps
- ✅ Memory usage <3 GB (no growth over time)
- ✅ Compilation time <60 seconds
- ✅ Trajectory output correct shape [T, N, 3]
- ✅ Timestep streaming works (cache swaps)

#### Testing

**Memory Stability Test**:
```python
def test_memory_stability():
    # Set up tracking
    mesh = load_threadeda_mesh(timestep=159)
    particles = seed_particles_uniform(mesh.bounds, 50000)

    # Track for 160 timesteps
    trajectory, final = track_particles_lax_scan(
        particles,
        timesteps=jnp.arange(160),
        static_data=static_data,
        field_cache=field_loader,
        dt=0.001,
        n_blocks=8
    )

    # Check shapes
    assert trajectory.shape == (160, 50000, 3)
    assert final['position'].shape == (50000, 3)

    # Check memory didn't explode
    mem_info = get_gpu_memory_info()
    assert mem_info['used'] < 3e9, f"GPU memory usage too high: {mem_info['used']/1e9:.2f} GB"

    print("✅ Memory stable for 160 timesteps")
```

---

### PHASE 7: Ghost Regions & Halo Exchange
**Duration**: 5-7 days
**Depends On**: Phase 6

#### Objectives
1. Implement ghost element arrays (1-layer overlap per block)
2. Detect particles near block boundaries
3. Synchronize ghost field data
4. Handle interpolation using ghost elements

#### Deliverables

**Files to Create**:

1. **`jaxtrace/gpu/forest/ghost_exchange.py`**
   - `identify_ghost_elements(blocks, mesh, thickness)` → ghost_elements per block
   - `exchange_ghost_particles(particles, blocks)` → particles with ghost data
   - Particle exchange protocol at block boundaries

2. **`jaxtrace/gpu/forest/halo_sync.py`**
   - `sync_halo_fields(field_data, blocks, ghost_elements)` → field with halos
   - Copy field values from neighboring blocks
   - Ensure continuity at boundaries

3. **`tests/gpu/test_ghost_regions.py`**
   - Test ghost element identification
   - Verify particle exchange
   - Check interpolation at boundaries

4. **`docs/gpu/PHASE_7_GHOST_REGIONS.md`**
   - Ghost region concept
   - Halo exchange algorithm
   - Boundary interpolation details
   - Particle conservation verification

#### Key Concepts

**Ghost Elements**:
- Elements within `thickness` distance of block boundary
- Copied from neighboring blocks
- Used for interpolation of particles near boundaries

**Halo Exchange**:
- Synchronize particle data at block interfaces
- Particles within ghost layer of neighbor block
- Ensures smooth interpolation across boundaries

#### Success Criteria

- ✅ Smooth interpolation at block boundaries
- ✅ No particle loss at boundaries
- ✅ <5% interpolation error near boundaries
- ✅ Ghost element arrays fit in memory budget

#### Testing

**Boundary Test**:
```python
def test_boundary_interpolation():
    # Seed particles intentionally at block boundary
    # Track across boundary
    # Verify smooth velocity field
    # Check no particles lost
    pass
```

---

### PHASE 8: Optimization & Scaling
**Duration**: 7-10 days
**Depends On**: Phase 7

#### Objectives
1. Block size tuning (benchmark 2×2×2, 4×4×2, 8×8×4)
2. Empty block detection and skipping
3. Load balancing analysis (static)
4. Memory-efficient trajectory storage
5. Compilation caching

#### Deliverables

**Files to Create**:

1. **`scripts/tune_block_size.py`**
   - Benchmark different grid sizes
   - Measure throughput and memory
   - Recommend optimal grid for user's mesh

2. **`scripts/optimize_memory.py`**
   - Memory reduction strategies
   - Trajectory compression
   - Reduced precision options

3. **`jaxtrace/gpu/forest/load_balancer.py`**
   - Static load analysis
   - Identify overloaded blocks
   - Recommendations for splitting

4. **`docs/gpu/PHASE_8_OPTIMIZATION.md`**
   - Block size tuning results
   - Memory optimization strategies
   - Load balancing analysis
   - Performance scaling curves

#### Success Criteria

- ✅ 100K particles tracked successfully
- ✅ 50-100× speedup vs CPU
- ✅ <4 GB VRAM usage
- ✅ Optimal block size identified
- ✅ GPU utilization >85%

#### Testing

**Scaling Test**:
```python
def test_100k_particles_full_mesh():
    # Track 100K particles for 160 timesteps
    # Verify <4 GB memory
    # Check trajectories correct
    # Measure throughput
    pass
```

---

### PHASE 9: Hash Octree Integration (Optional)
**Duration**: 5-7 days
**Depends On**: Phase 8

#### Objectives
1. Integrate `hash_octree.py` from previous implementation
2. Build per-block hash octrees
3. Replace hierarchical search with O(1) hash lookup
4. Benchmark hash vs tree search

#### Deliverables

**Files to Create**:

1. **`jaxtrace/gpu/forest/hash_octree.py`**
   - Copy and adapt from `jaxtrace/fields/hash_octree.py`
   - Per-block hash table construction
   - O(1) element lookup

2. **`scripts/benchmark_hash_vs_tree.py`**
   - Compare search performance
   - Measure speedup from hashing

3. **`docs/gpu/PHASE_9_HASH_OCTREE.md`**
   - Hash octree integration
   - Per-block hash tables
   - Performance improvements

#### Success Criteria

- ✅ Additional 2-5× speedup from O(1) lookup
- ✅ Hash tables fit in block memory budget
- ✅ No accuracy loss

---

### PHASE 10: High-Level API & Documentation
**Duration**: 3-5 days
**Depends On**: Phase 9 (or Phase 8 if skipping hash octrees)

#### Objectives
1. Create simple user-facing API
2. Example notebooks
3. Complete documentation
4. Performance comparison report

#### Deliverables

**Files to Create**:

1. **`jaxtrace/gpu/__init__.py`**
   - Public API exposure
   - `GPUForestTracker` class

2. **`examples/gpu/tutorial_basic.ipynb`**
   - Getting started guide
   - Basic tracking workflow

3. **`examples/gpu/tutorial_advanced.ipynb`**
   - Custom configurations
   - Performance tuning

4. **`docs/gpu/GPU_FOREST_API.md`**
   - Complete API reference
   - Configuration options
   - Best practices

5. **`docs/gpu/GPU_PERFORMANCE_REPORT.md`**
   - Benchmarks vs CPU
   - Scaling analysis
   - Memory profiling

6. **`README_GPU.md`**
   - Quick start guide
   - Installation instructions
   - Example usage

#### API Design

**High-Level API**:
```python
from jaxtrace.gpu import GPUForestTracker

# Create tracker
tracker = GPUForestTracker(
    mesh_path="/path/to/threadedAvtk_*.pvtu",
    block_grid=(4, 4, 2),          # 32 blocks
    field_name="Displacement",      # Velocity field
    max_particles_per_block=10000
)

# Seed particles
seeds = tracker.seed_uniform(n_particles=50000)

# Track particles
trajectories = tracker.track(
    seeds=seeds,
    timesteps=range(120, 160),  # Revolution cycle
    dt=0.001,
    save_trajectory=True,
    trajectory_stride=1
)

# Export results
tracker.export_vtk("trajectories.vtu")
tracker.export_hdf5("trajectories.h5")

# Analyze
tracker.plot_trajectories_3d()
tracker.compute_density_kde()
```

#### Success Criteria

- ✅ API usable in <10 lines of code
- ✅ Complete documentation with examples
- ✅ Example notebooks run successfully
- ✅ Performance report complete

---

## Component Reuse Strategy

### From Previous Implementation (phase1-optimization)

**Bring Back These GPU Components**:

| Component | Source File | Target Phase | Purpose | Adaptation Needed |
|-----------|-------------|--------------|---------|-------------------|
| **Morton codes** | `jaxtrace/fields/morton_code.py` | Phase 1 | Spatial indexing, Z-order curve | Minimal (copy) |
| **Element testing** | `jaxtrace/fields/element_testing_jax.py` | Phase 3 | Barycentric point-in-tet test | Block-local adaptation |
| **FEM interpolation** | `jaxtrace/fields/interpolator_jax_simple.py` | Phase 3 | Tetrahedral interpolation | None (direct copy) |
| **Hash octrees** | `jaxtrace/fields/hash_octree.py` | Phase 9 | O(1) element lookup | Per-block hash tables |
| **GPU field sampling** | `jaxtrace/fields/gpu_field_sampling.py` | Phase 3 | Reference for pipeline | Conceptual reference only |

**Copy Strategy**:
1. Copy file to new location in `jaxtrace/gpu/`
2. Refactor for forest-of-octrees architecture
3. Add unit tests in `tests/gpu/`
4. Document adaptations in phase docs

### Keep As-Is (No Migration)

**These modules work as-is**:
- `jaxtrace/io/*` - VTK readers/writers (use directly)
- `jaxtrace/visualization/*` - Plotting (use directly)
- `jaxtrace/density/*` - KDE/SPH analysis (use directly)
- `jaxtrace/tracking/analysis.py` - Trajectory analysis (use directly)
- `jaxtrace/tracking/seeding.py` - Particle seeding (use directly)

---

## Testing Strategy

### Unit Tests (Per Component)
- **Scope**: Individual functions/classes
- **Data**: Synthetic simple meshes (e.g., 8-cell uniform cube)
- **Duration**: <10 seconds per test
- **Framework**: pytest
- **Location**: `tests/gpu/test_*.py`

**Example**:
```python
def test_morton_encode():
    # Test Morton encoding with known values
    pos = np.array([0.5, 0.5, 0.5])
    code = encode_morton(pos, bounds=np.array([0, 1, 0, 1, 0, 1]))
    assert code == expected_code
```

### Integration Tests (Per Phase)
- **Scope**: Multiple components working together
- **Data**: ThreadedA reference mesh
- **Duration**: 1-10 minutes per test
- **Framework**: Jupyter notebooks
- **Location**: `examples/gpu/phase*_*.ipynb`

**Example Phases**:
- Phase 0: Block visualization on real mesh
- Phase 3: Single particle trajectory
- Phase 5: Multi-block tracking with visualization

### Validation Tests (Milestones)
- **Scope**: Compare with CPU tracker (ground truth)
- **Data**: ThreadedA mesh, known particle seeds
- **Metrics**: Trajectory accuracy, conservation, performance
- **Frequency**: End of Phase 5, 8, 10

**Example**:
```python
def test_trajectory_accuracy_vs_cpu():
    # Track same particles with GPU and CPU
    # Compare final positions
    # Verify <1% error
    pass
```

### Performance Benchmarks
- **Scope**: Speedup, throughput, memory usage
- **Particle Counts**: 100, 1K, 10K, 50K, 100K
- **Frequency**: Phase 4, 5, 8
- **Output**: Plots, tables, reports

**Metrics**:
- Speedup vs CPU
- Throughput (particle-timesteps/second)
- GPU utilization (%)
- Memory usage (GB)
- Cache hit rate (%)

---

## Configuration

### Configuration File Format

**User-facing config** (YAML):
```yaml
# config_gpu.yaml
forest:
  block_grid: [4, 4, 2]         # 32 blocks
  max_octree_depth: 12
  ghost_layer_thickness: 1

field:
  name: "Displacement"
  auto_detect: true

mesh:
  path: "/path/to/mesh/*.pvtu"
  revolution_cycle: [120, 159]  # Or null for auto-detect
  build_from_timestep: -1       # -1 for auto-detect

tracking:
  max_particles_per_block: 10000
  dt: 0.001
  integrator: "rk4"             # or "euler", "rk2"

performance:
  skip_empty_blocks: true
  enable_load_balancing: false
  compile_cache: true

output:
  save_trajectory: true
  trajectory_stride: 1
  format: "hdf5"                # or "vtk"
```

**Python API**:
```python
from jaxtrace.gpu.config import GPUForestConfig

config = GPUForestConfig.from_yaml("config_gpu.yaml")
```

### Configurable Parameters

**Phase 0** (Initial):
- `block_grid` - Forest grid size
- `field_name` - Velocity field name
- `max_octree_depth` - Octree refinement level

**Phase 6** (Time Marching):
- `revolution_cycle` - Timestep range
- `build_from_timestep` - Forest construction timestep

**Phase 7** (Ghost Regions):
- `ghost_layer_thickness` - Halo size

**Phase 8** (Optimization):
- `skip_empty_blocks` - Empty block optimization
- `enable_load_balancing` - Dynamic splitting
- `max_particles_per_block` - Memory tuning

---

## Memory Management

### Memory Budget (4 GB VRAM)

**Target Allocation**:
```
┌───────────────────────────────────────┐
│ GPU Memory (4 GB total, 3.7 GB usable) │
├───────────────────────────────────────┤
│ Static Data:         ~1 GB (25%)      │
│  ├─ Forest blocks                     │
│  ├─ Mesh data                         │
│  ├─ Element neighbors                 │
│  └─ Field cache (5-10 timesteps)      │
├───────────────────────────────────────┤
│ Dynamic Data:        ~3 GB (75%)      │
│  ├─ Particle positions                │
│  ├─ Particle velocities               │
│  ├─ Element/block IDs                 │
│  └─ Ghost buffers                     │
├───────────────────────────────────────┤
│ Headroom:            ~0.7 GB (reserv) │
└───────────────────────────────────────┘
```

### Scaling with Block Count

**Memory vs. Block Count** (for ThreadedA mesh, ~1300 cells):

| Block Grid | Blocks | Static Data | Max Particles (100K budget) | GPU Util |
|------------|--------|-------------|------------------------------|----------|
| 2×2×2      | 8      | ~500 MB     | 150K                         | 60-70%   |
| 4×4×2      | 32     | ~1 GB       | 100K                         | 75-85%   |
| 4×4×4      | 64     | ~1.5 GB     | 70K                          | 85-90%   |
| 8×8×4      | 256    | ~3 GB       | 30K                          | 90-95%   |

**Recommendation**: Start with 2×2×2 (8 blocks), scale to 4×4×2 (32 blocks) if GPU utilization target not met.

### Memory Profiling (Per Phase)

**Phase 2**: Measure static data memory
**Phase 4**: Measure per-particle memory
**Phase 6**: Monitor memory over time (check for leaks)
**Phase 8**: Optimize memory footprint

---

## Success Metrics

### Phase 5 (MVP)
- ✅ **Functionality**: 50K particles tracked for 40 timesteps
- ✅ **Performance**: 10× speedup vs CPU
- ✅ **Accuracy**: <1% interpolation error
- ✅ **Conservation**: No lost particles
- ✅ **GPU Utilization**: >60%

### Phase 8 (Optimized)
- ✅ **Scalability**: 100K particles for 160 timesteps
- ✅ **Performance**: 50-100× speedup vs CPU
- ✅ **Throughput**: >1M particle-timesteps/second
- ✅ **Memory**: <4 GB VRAM
- ✅ **GPU Utilization**: >80%

### Phase 10 (Production)
- ✅ **Usability**: API requires <10 lines of code
- ✅ **Documentation**: Complete with examples
- ✅ **Robustness**: Works on user's actual meshes
- ✅ **Performance Report**: Published benchmarks vs CPU

---

## Risk Mitigation

### Primary Risks

**Risk 1: VRAM Exhaustion**
- **Probability**: High (4 GB is limited)
- **Impact**: Crashes, cannot run
- **Mitigation**:
  - Start with small block count (2×2×2 = 8 blocks)
  - Memory profiling at each phase
  - Trajectory compression/streaming
- **Fallback**: Reduce particle count or block count

**Risk 2: JAX Compilation Time**
- **Probability**: Medium
- **Impact**: Slow development iteration
- **Mitigation**:
  - Cache compiled functions
  - Use `jax.disable_jit()` during debugging
  - Benchmark compilation vs execution time
- **Fallback**: Accept compilation overhead (one-time cost)

**Risk 3: Particle Load Imbalance**
- **Probability**: Medium
- **Impact**: Some blocks overloaded, poor GPU utilization
- **Mitigation**:
  - Static load analysis in Phase 2
  - Pad to max particles per block
  - Defer dynamic splitting to Phase 8
- **Fallback**: Use more blocks (better load distribution)

**Risk 4: Accuracy at Block Boundaries**
- **Probability**: Low-Medium
- **Impact**: Interpolation errors, lost particles
- **Mitigation**:
  - Ghost regions (Phase 7)
  - Thorough boundary testing
  - Compare with CPU tracker
- **Fallback**: Increase ghost layer thickness

**Risk 5: Timeline Slip**
- **Probability**: Medium
- **Impact**: Project takes longer than 10 weeks
- **Mitigation**:
  - Clear phase deliverables
  - Each phase produces working tracker
  - Can pause at Phase 5 (MVP) and resume later
- **Fallback**: Ship MVP (Phase 5), defer optimization

---

## Timeline and Milestones

### Phase-by-Phase Timeline

| Phase | Duration | Cumulative | Key Milestone |
|-------|----------|-----------|---------------|
| Phase 0 | 2-3 days | Week 1 | Forest visualization |
| Phase 1 | 5-7 days | Week 2 | CPU block search |
| Phase 2 | 3-5 days | Week 2-3 | JAX arrays on GPU |
| Phase 3 | 7-10 days | Week 3-4 | GPU single-particle kernel |
| Phase 4 | 3-5 days | Week 4-5 | Vectorized (vmap) |
| **Phase 5** | **5-7 days** | **Week 5** | **MVP: Multi-block tracking** |
| Phase 6 | 3-5 days | Week 6 | Time marching (lax.scan) |
| Phase 7 | 5-7 days | Week 7 | Ghost regions |
| Phase 8 | 7-10 days | Week 8-9 | Optimization & 100K particles |
| Phase 9 | 5-7 days | Week 9-10 | Hash octree (optional) |
| Phase 10 | 3-5 days | Week 10 | Production API |

**Total**: 6-10 weeks
**MVP**: 4-5 weeks (Phase 5)

### Decision Points

**After Phase 5**:
- Evaluate MVP performance
- Decide whether to continue to Phase 6-10 or iterate on Phase 5
- User can deploy MVP if meets requirements

**After Phase 8**:
- Assess whether Phase 9 (hash octrees) needed
- Check if performance targets met without hashing
- Can skip to Phase 10 if satisfied

---

## Next Steps

### Immediate Actions (After Plan Approval)

1. **Create Directory Structure** (30 minutes)
   ```bash
   mkdir -p jaxtrace/gpu/forest
   mkdir -p tests/gpu
   mkdir -p examples/gpu
   mkdir -p docs/gpu
   ```

2. **Set Up Configuration** (1 hour)
   - Create `jaxtrace/gpu/config.py`
   - Define `GPUForestConfig` dataclass
   - Add validation logic

3. **Begin Phase 0** (2-3 days)
   - Implement `block_builder.py`
   - Create regular grid generator
   - Build visualization tool
   - Test on ThreadedA mesh

4. **Document Progress** (ongoing)
   - Create `docs/GPU_IMPLEMENTATION_PROGRESS.md`
   - Track completed phases
   - Note issues and solutions
   - Update timeline estimates

### User Feedback Points

**Request feedback after**:
- Phase 0: Forest visualization (verify block layout looks correct)
- Phase 2: Memory profiling (confirm block count suitable)
- Phase 5: MVP demo (validate performance and accuracy)
- Phase 8: Optimization results (discuss trade-offs)
- Phase 10: Final API design (review usability)

---

## Appendix A: Glossary

**Forest-of-Octrees**: Domain decomposition where each block is root of independent octree
**Block**: Spatial partition of domain (e.g., one of 8 blocks in 2×2×2 grid)
**Element ID Caching**: Storing last containing element in particle data structure
**Three-Tier Search**: Hierarchical search (cached → neighbors → tree)
**Ghost Region / Halo**: Overlap region between adjacent blocks
**Rebatching**: Sorting particles by block_id after position update
**Morton Code**: Z-order space-filling curve encoding for spatial indexing
**lax.scan**: JAX primitive for sequential operations with minimal memory
**vmap**: JAX primitive for vectorization (SIMD parallelism)
**DeviceArray**: JAX array residing in GPU memory
**JIT**: Just-In-Time compilation for GPU kernels

---

## Appendix B: Reference Documents

**Strategy Document**:
`docs/High_Performance_Particle_Tracking_on_the_GPU.md`

**Mesh Locations**:
- **ThreadedA** (reference): `/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_*.pvtu`
- **FLA** (validation): `/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu`

**Previous Implementation**:
Branch `phase1-optimization` (do not merge, reference only for GPU components)

**JAX Documentation**:
- vmap: https://jax.readthedocs.io/en/latest/jax.html#jax.vmap
- lax.scan: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
- JIT: https://jax.readthedocs.io/en/latest/jax.html#jax.jit

---

## Appendix C: Questions for User

**Before Starting Implementation**:

1. Should mesh analysis visualization for ThreadedA be generated now, or defer to Phase 0?
2. Preferred block grid size to start: 2×2×2 (8 blocks) or 4×4×2 (32 blocks)?
3. Should Phase 9 (hash octrees) be mandatory or truly optional?
4. Preferred documentation format: markdown only, or markdown + Jupyter notebooks?
5. Should integration tests use synthetic meshes or always ThreadedA?

**After Phase 5 (MVP)**:

1. Is performance acceptable, or proceed with optimization phases?
2. Should block count be adjusted based on GPU utilization?
3. Are there specific trajectory export formats needed beyond VTK/HDF5?

---

**END OF PLAN**

---

**Status**: Ready for user review and approval
**Next Action**: User reviews plan, provides feedback/modifications
**Then**: Begin Phase 0 implementation
