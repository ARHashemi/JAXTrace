# CLEAN GPU IMPLEMENTATION PLAN
**GPU-Native Particle Tracking with Forest-of-Octrees Architecture**

**Date**: 2025-11-06
**Branch**: `gpu_native_implementation` (clean restart)
**Reference Mesh**: ThreadedA (3.5M elements, 900K nodes)
**Target**: 1M particles with <500 MB GPU memory
**Hardware**: NVIDIA T1000 (4GB VRAM)

---

## TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)
2. [Phase 0: Mesh Analysis & Validation](#phase-0-mesh-analysis--validation)
3. [Phase 1: Forest Structure & Block Partitioning](#phase-1-forest-structure--block-partitioning)
4. [Phase 2: Element Neighbors & Padded Block Arrays](#phase-2-element-neighbors--padded-block-arrays)
5. [Phase 3: Particle Seeding & Initial Assignment](#phase-3-particle-seeding--initial-assignment)
6. [Phase 4: GPU Multi-Level Element Search](#phase-4-gpu-multi-level-element-search)
7. [Future Phases (5-9)](#phases-5-9-future-phases-summary)
8. [Appendices](#appendix)

---

## EXECUTIVE SUMMARY

This plan implements GPU-native particle tracking **from scratch**, incorporating all lessons learned from previous attempts (V4/V5). The implementation follows a strict **phase-by-phase** approach where each phase is reviewed, approved, implemented, tested, and committed before proceeding.

### Core Principles (Non-Negotiable)

1.  **Padded 2D Arrays** - `(n_blocks, max_elem)` NOT global flattening
2.  **Multi-Level Search** - L0 (cache) ’ L1 (neighbors) ’ L2 (block) ’ L3 (neighbor blocks)
3.  **Minimal Scan Carry** - Only particle state (~33 bytes/particle)
4.  **Static Arrays** - No dynamic slicing in JIT
5.  **Block-Local Search** - Preserves spatial partitioning

### Key Lessons from Previous Attempts

**V4 Failures** L:
- **Global flattening of blocks** ’ 45 GB memory explosion
- **No multi-level hierarchy** ’ missed 90% cache hits
- **Nested vmap over particles × elements** ’ memory explosion
- **Dictionary access in JIT** ’ compilation errors

**V5 Solutions** :
- **Padded 2D arrays** ’ 372 MB memory (120× improvement)
- **Full 4-level search** ’ 85-95% L0 hit rate (10-20× speedup)
- **26-neighbor block topology** ’ handles spanning elements
- **Convert dicts to arrays BEFORE JIT** ’ compilation works

### Memory Budget

**Target: <500 MB for 1M particles on ThreadedA (3.5M elements)**

```
Static Data (uploaded once to GPU):
  Node positions:           10.8 MB
  Element connectivity:     56.0 MB
  Element neighbors:        56.0 MB
  Element-to-block:         14.0 MB
  Block elements (padded): 192.0 MB   Critical: V5 solution
  Block neighbors (26):      0.003 MB
  Velocity field:           10.8 MB
  --------------------------------
  Total Static:            339.6 MB 

Particle Data (scan carry):
  Positions (3×float64):    24.0 MB
  Element IDs (int32):       4.0 MB
  Block IDs (int32):         4.0 MB
  Active mask (bool):        1.0 MB
  --------------------------------
  Total Particles:          33.0 MB 

TOTAL:                     372.6 MB  (vs 45 GB in V4)
```

### Implementation Strategy

**No Current Code Reuse**: All code will be written fresh, applying V5 lessons from the start.

**Phase-by-Phase Approval**: Each phase must be:
1. **Reviewed** - Plan presented for approval
2. **Implemented** - Code written with tests
3. **Tested** - All tests passing
4. **Documented** - Phase completion doc written
5. **Committed** - One commit per phase

**Documentation**: Incremental, per-phase, no redundancy. Each phase doc references previous phases by link.

---

## PHASE 0: MESH ANALYSIS & VALIDATION

**Duration**: 1 day
**Dependencies**: None
**Goal**: Verify ThreadedA mesh characteristics and establish baseline

### Objectives

1. Verify documented mesh statistics are correct
2. Validate spatial domain and element counts
3. Confirm optimal block decomposition (4×4×2 = 32 blocks)
4. Establish baseline memory/performance metrics

### Background: ThreadedA Mesh (From Previous Analysis)

**Expected Characteristics**:
- **Location**: `/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_*.pvtu`
- **Total Elements**: 3,494,800 - 3,512,279 tetrahedral cells
- **Total Nodes**: 898,502 - 900,658 points
- **Timesteps**: 160 (indexed 0, 100-259)
- **Parallel Pieces**: 64 per timestep
- **Domain**: [-0.030, 0.030] × [-0.023, 0.023] × [-0.010, 0.000] meters
- **Type**: Adaptive Mesh Refinement (AMR) for welding simulation
- **Element Volume Range**: 8.124e-14 to 2.130e-08 (262,146× ratio)
- **Neighbor Statistics**: 0-4 neighbors/element (78.9% interior with 4 neighbors)

### Tasks

#### Task 0.1: Verify Existing Analysis

**Check if complete** (no reimplementation needed):
- Review `docs/THREADEDA_MESH_ANALYSIS.md` or similar
- Check if contains: element counts, domain bounds, refinement statistics, connectivity
- Validate data quality: within 5% of expected values

**If analysis exists and is valid**: Skip to Task 0.3

#### Task 0.2: Re-run Mesh Analysis (Only If Needed)

**File to Create**: `tools/analyze_threadeda_mesh.py` (~400 lines)

**Key Functions**:
- `load_pvtu_file()` - Load parallel VTK file
- `extract_mesh_data()` - Get nodes, elements, fields
- `compute_mesh_statistics()` - Element stats, domain bounds, connectivity
- `analyze_block_decompositions()` - Load balance for different grids
- `generate_markdown_report()` - Create documentation

**Run**: `python tools/analyze_threadeda_mesh.py`

**Output**:
- `docs/phase0/mesh_statistics.json`
- `docs/phase0/THREADEDA_MESH_VALIDATED.md`

#### Task 0.3: Validate Block Decomposition

**Analyze different grid sizes**:

| Grid Size | Blocks | Imbalance | Elem/Block (mean) | Recommended? |
|-----------|--------|-----------|-------------------|--------------|
| 2×2×2     | 8      | ~1.1×     | ~437K             | Too coarse   |
| **4×4×2** | **32** | **~8.6×** | **~110K**         |  **YES**   |
| 8×8×4     | 256    | ~32×      | ~14K              | Too fragmented |
| 16×16×8   | 2048   | ~43×      | ~1.7K             | Too fragmented |

**Selection Criteria**:
-  Block count: 32-64 (good GPU occupancy)
-  Imbalance: <10× (acceptable for AMR)
-  Elements/block: 50K-200K (manageable)
-  95th percentile: ~859K elements (with padding fits GPU)

**Decision**: **4×4×2 = 32 blocks** 

#### Task 0.4: Create Recommended Configuration

**File**: `config/threadeda_recommended.yaml`

**Content**:
```yaml
mesh:
  name: "ThreadedA"
  path: "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule"
  pattern: "threadedAvtk_*.pvtu"
  n_elements: 3500000
  n_nodes: 900000
  domain_bounds:
    x: [-0.030, 0.030]
    y: [-0.023, 0.023]
    z: [-0.010, 0.000]

forest:
  grid_size: [4, 4, 2]  # 32 blocks
  n_blocks: 32
  max_elements_per_block: 1300000  # 95th %ile × 1.5 padding
  expected_mean_elements: 110000
  expected_imbalance: 8.6

storage:
  field_storage: "nodes"  # NOT "elements"
  octree_storage: "padded"  # Required for JAX
  block_storage: "padded"  # Required for JAX

particles:
  target_count: 1000000

memory_budget:
  static_data_mb: 340
  particle_data_mb: 33
  total_target_mb: 500
```

### Deliverables

1.  `docs/phase0/THREADEDA_MESH_VALIDATED.md` (~10 pages)
   - Mesh characteristics table
   - Block decomposition comparison
   - Memory budget breakdown
   - Recommended configuration rationale

2.  `docs/phase0/mesh_statistics.json`
3.  `config/threadeda_recommended.yaml`
4.  Optional: `tools/analyze_threadeda_mesh.py` (if needed)

### Success Criteria

-  Element count: 3.4M - 3.6M (within 5%)
-  Node count: 850K - 950K
-  Domain bounds match expected
-  Optimal grid: 4×4×2 justified
-  Load imbalance: <10× documented
-  Memory budget: <500 MB

### Phase 0 Completion

**Commit Message**:
```
Phase 0: Mesh analysis and validation complete

- Verified ThreadedA mesh: 3.5M elements, 900K nodes
- Selected 4×4×2 grid (32 blocks, 8.6× imbalance)
- Documented memory budget: <500 MB for 1M particles
- Created recommended configuration
```

---

## PHASE 1: FOREST STRUCTURE & BLOCK PARTITIONING

**Duration**: 2-3 days
**Dependencies**: Phase 0 complete
**Goal**: Create forest grid, assign elements to blocks, build neighbor topology

### Objectives

1. Create regular forest grid (4×4×2 = 32 blocks)
2. Assign all 3.5M elements to blocks
3. Build 26-neighbor block topology
4. Validate: All elements assigned, no gaps

### Why This Phase is Critical

- **Forest structure** is foundation for block-local search
- **Element-to-block mapping** enables O(1) block lookup
- **26-neighbor topology** handles elements spanning boundaries
- **Without this**: Cannot implement block-local search (Phase 4)

### Tasks

#### Task 1.1: Regular Forest Grid Generator

**File**: `jaxtrace/gpu/forest/block_grid.py` (NEW, ~300 lines)

**Data Structures**:
```python
@dataclass
class Block:
    block_id: int
    bounds: np.ndarray  # [xmin, xmax, ymin, ymax, zmin, zmax]
    center: np.ndarray  # [x, y, z]
    grid_index: Tuple[int, int, int]
    neighbors_6: np.ndarray  # [+x, -x, +y, -y, +z, -z], -1 = boundary
    neighbors_26: np.ndarray  # All 26 neighbors, -1 padded
```

**Key Functions**:
- `create_regular_grid()` - Divide domain into uniform blocks
- `compute_6_neighbors()` - Face neighbors (grid arithmetic)
- `compute_26_neighbors()` - All adjacent (6 faces + 12 edges + 8 corners)
- `position_to_block_id()` - Fast O(1) mapping

**Performance**: <1s for 32 blocks

#### Task 1.2: Element-to-Block Assignment

**File**: `jaxtrace/gpu/forest/element_mapper.py` (NEW, ~250 lines)

**Key Functions**:
- `compute_element_centroids()` - Mean of 4 tet vertices
- `assign_elements_to_blocks()` - Map centroid ’ block ID
- `validate_element_assignment()` - Check all assigned, no duplicates

**Algorithm**:
```
For each element:
    1. Compute centroid = (n0 + n1 + n2 + n3) / 4
    2. Find block: block_id = position_to_block_id(centroid)
    3. Store: element_to_block[elem_id] = block_id
Build inverse: block_to_elements[block_id] = [elem_ids...]
```

**Performance**: <5s for 3.5M elements (>700K elem/s)

#### Task 1.3: Load Balance Analysis

**File**: `jaxtrace/gpu/forest/load_balance.py` (NEW, ~150 lines)

**Key Functions**:
- `compute_load_balance_stats()` - Min, max, mean, percentiles
- `print_load_balance_report()` - Pretty output

**Metrics**:
- Imbalance factor: max / mean
- Empty blocks count
- 95th percentile element count

### Tests

**File**: `tests/gpu/test_block_grid.py` (~300 lines, 12 tests)
- `test_grid_2x2x2()` - 8 blocks, correct bounds
- `test_grid_4x4x2()` - 32 blocks (ThreadedA config)
- `test_6_neighbors_corner()` - Corner has 3 neighbors
- `test_6_neighbors_interior()` - Interior has 6 neighbors
- `test_26_neighbors_corner()` - Corner has 7 neighbors
- `test_26_neighbors_interior()` - Interior has 26 neighbors
- `test_neighbor_symmetry()` - A neighbor of B Ô B neighbor of A
- `test_position_to_block_id()` - O(1) mapping correct

**File**: `tests/gpu/test_element_mapper.py` (~300 lines, 10 tests)
- `test_centroid_single_tet()` - Correct centroid
- `test_assign_simple_mesh()` - 8 elements ’ 8 blocks
- `test_assign_performance()` - 3.5M elements in <5s
- `test_boundary_elements()` - Consistent assignment
- `test_validation()` - All checks pass

**Run**: `pytest tests/gpu/test_block_grid.py tests/gpu/test_element_mapper.py -v`

### Deliverables

1.  **3 new modules**: `block_grid.py`, `element_mapper.py`, `load_balance.py`
2.  **2 test files**: 22+ tests
3.  **Documentation**: `docs/phase1/FOREST_STRUCTURE_COMPLETE.md` (~15 pages)
4.  **Figures**: Block grid 3D, element distribution, load balance histogram

### Success Criteria

-  All 3.5M elements assigned
-  Load imbalance: 8-10× (acceptable, documented)
-  26-neighbor topology correct
-  Memory: <50 MB for structure
-  Performance: <5s for assignment
-  All tests passing (22+)

### Phase 1 Completion

**Commit Message**:
```
Phase 1: Forest structure and block partitioning complete

- Created 4×4×2 regular grid (32 blocks)
- Assigned 3.5M elements in 3.2s (1.1M elem/s)
- Built 26-neighbor topology
- Load imbalance: 8.6× (acceptable for AMR)
- Memory: 42 MB
- Tests: 22 passing
```

---

## PHASE 2: ELEMENT NEIGHBORS & PADDED BLOCK ARRAYS

**Duration**: 2-3 days
**Dependencies**: Phase 1 complete
**Goal**: Extract element neighbors, create **padded block arrays** (V5 solution)

### Objectives

1. Extract element face-adjacency (tetrahedral mesh)
2. Convert block-to-elements dict to **padded 2D arrays** (CRITICAL for JAX)
3. Validate: All adjacencies correct, padding works

### Why This Phase is Critical

- **Element neighbors** enable Level 1 search (8% hit rate)
- **Padded 2D arrays** solve V4's memory explosion (45 GB ’ 372 MB)
- **This is the V5 breakthrough** that makes block-local search work in JAX

### The V5 Solution Explained

**V4 Problem**:
```python
# This fails in JAX JIT
@jax.jit
def search(pos, block_id, octrees: Dict):
    octree = octrees[block_id]  # L Can't trace dictionary access
```

**V4 Wrong Fix**:
```python
# Flatten all blocks into single array - DESTROYS spatial partitioning
all_element_ids = []
for block_id, octree in octrees.items():
    all_element_ids.extend(octree.elements)  # Merges ALL blocks

# Result: O(N_particles × N_elements) = 45 GB memory L
```

**V5 Correct Solution**:
```python
# Convert dict to padded 2D array BEFORE JIT
block_elements = np.full((n_blocks, max_elem), -1, dtype=np.int32)
for bid, octree in octrees.items():
    block_elements[bid, :len(octree.elements)] = octree.elements

# Now works in JIT!
@jax.jit
def search(pos, block_id, block_elements):
    elems = block_elements[block_id]  #  Static indexing
    # Result: O(N_particles × max_elem_per_block) = 2 GB 
```

**Impact**: 22× memory reduction (45 GB ’ 2 GB)

### Tasks

#### Task 2.1: Element Neighbor Extraction

**File**: `jaxtrace/gpu/mesh/element_neighbors.py` (NEW, ~400 lines)

**Algorithm**:
```
Tetrahedral mesh: Each element has 4 triangular faces
Face neighbors: Elements sharing a face are neighbors

Steps:
1. Extract all faces from all elements (N_elements × 4 faces)
2. Canonicalize each face (sort 3 node indices)
3. Build face ’ elements dictionary
4. For faces with 2 elements ’ mark as neighbors
5. For faces with 1 element ’ boundary (no neighbor)

Result: (N_elements, 4) array with neighbor IDs, -1 for boundaries
```

**Key Functions**:
- `extract_tetrahedral_faces()` - Get 4 faces per tet
- `build_face_to_elements_map()` - Face ’ elements dict
- `compute_element_neighbors()` - Main algorithm
- `validate_neighbor_symmetry()` - If A neighbor of B, then B neighbor of A
- `compute_neighbor_statistics()` - Mean, boundary count

**Performance**: <60s for 3.5M elements (>58K elem/s)

#### Task 2.2: Padded Block Element Arrays (V5 Critical Innovation)

**File**: `jaxtrace/gpu/arrays/padded_blocks.py` (NEW, ~450 lines)

**Data Structure**:
```python
@dataclass
class PaddedBlockArrays:
    """
    Padded block element arrays for JAX JIT.

    This solves V4's memory explosion by preserving spatial
    partitioning while enabling static indexing in JAX JIT.
    """
    block_elements: np.ndarray  # (n_blocks, max_elem) int32, -1 padded
    block_elem_counts: np.ndarray  # (n_blocks,) int32
    block_neighbors_26: np.ndarray  # (n_blocks, 26) int32
    max_elem_per_block: int
    n_blocks: int
    total_elements: int
    padding_waste: float  # Fraction wasted (0-1)
```

**Key Functions**:
- `build_padded_block_arrays()` - Convert dict ’ padded 2D array
- `validate_padded_arrays()` - Check correctness (6 checks)
- `print_memory_comparison()` - Show V4 vs V5 improvement

**Algorithm**:
```
1. Compute max_elem_per_block = 95th percentile × 1.5
2. Allocate (n_blocks, max_elem) array filled with -1
3. For each block:
     Copy element IDs to row
     Leave rest as -1 padding
4. Extract 26-neighbor array from blocks
```

**Memory Calculation** (ThreadedA, 1M particles):
```
V4: 1M × 3.5M × 8 bytes = 28 TB L
V5: 1M × 1.3M × 8 bytes = 10.4 GB 
Improvement: 2,692× less memory
```

### Tests

**File**: `tests/gpu/test_element_neighbors.py` (~400 lines, 12 tests)
- `test_simple_tet_mesh()` - 2 tets sharing face
- `test_boundary_element()` - Boundary has <4 neighbors
- `test_interior_element()` - Interior has 4 neighbors
- `test_threadeda_neighbors()` - 3.5M elements, mean 2-3
- `test_neighbor_symmetry()` - Mutual neighbors

**File**: `tests/gpu/test_padded_blocks.py` (~400 lines, 10 tests)
- `test_simple_padding()` - 3 blocks, different sizes
- `test_threadeda_size()` - 3.5M elements, <300 MB
- `test_validation()` - All checks pass
- `test_v4_vs_v5_memory()` - Show dramatic improvement

**Run**: `pytest tests/gpu/test_element_neighbors.py tests/gpu/test_padded_blocks.py -v`

### Deliverables

1.  **3 new modules**: `element_neighbors.py`, `padded_blocks.py`, `validation.py`
2.  **2 test files**: 22+ tests
3.  **Documentation**: `docs/phase2/PADDED_ARRAYS_COMPLETE.md` (~20 pages)
   - V4 problem ’ V5 solution explanation
   - Memory comparison table
   - Validation results
4.  **Figures**: Memory comparison chart, padding visualization

### Success Criteria

-  Element neighbors: Mean 2-4/element
-  Neighbor symmetry: 100% validated
-  Padded arrays: All 3.5M elements preserved
-  Memory: <200 MB (vs 45 GB V4)
-  Padding waste: <50%
-  Performance: <60s for neighbors
-  All tests passing (22+)

### Phase 2 Completion

**Commit Message**:
```
Phase 2: Element neighbors and padded block arrays complete

- Extracted face-adjacent neighbors in 48s
- Mean neighbors: 3.52/element
- Built padded arrays: (32, 1.3M) with 42% padding
- Memory: 192 MB (vs 45 GB in V4)
- Improvement: 234× less memory 
- Validation: 100% correctness
- Tests: 22 passing

This is the V5 breakthrough for block-local search in JAX.
```

---

## PHASE 3: PARTICLE SEEDING & INITIAL ASSIGNMENT

**Duration**: 1-2 days
**Dependencies**: Phase 2 complete
**Goal**: Seed particles, assign initial containing element (CPU baseline)

### Objectives

1. Seed particles in domain (uniform, gaussian, custom)
2. Find initial containing element using CPU search (ground truth)
3. Create particle state structure (minimal for scan carry)
4. Validate: >98% particles found

### Tasks

#### Task 3.1: Particle Seeding

**File**: `jaxtrace/gpu/particles/seeding.py` (NEW, ~300 lines)

**Key Functions**:
- `seed_particles_uniform()` - Random uniform in domain
- `seed_particles_gaussian()` - Normal distribution
- `seed_particles_from_file()` - Load from CSV/NPY

**Performance**: <1s for 1M particles

#### Task 3.2: CPU Initial Element Search

**File**: `jaxtrace/gpu/search/cpu_search.py` (NEW, ~400 lines)

**Algorithm** (Ground truth for validation):
```
For each particle:
    1. Find block using grid arithmetic (O(1))
    2. Get elements in block from padded arrays
    3. Test each element with barycentric coordinates
    4. If not found, search 26 neighbor blocks
    5. If still not found, global search (rare)

Returns: element_id or -1
```

**Key Functions**:
- `find_containing_element_cpu()` - Single particle
- `find_containing_elements_batch_cpu()` - Batch with progress bar

**Performance**: <30s for 10K particles on 3.5M elements

#### Task 3.3: Particle Data Structure

**File**: `jaxtrace/gpu/particles/state.py` (NEW, ~200 lines)

**Data Structure**:
```python
@dataclass
class ParticleState:
    """Minimal particle state for scan carry."""
    positions: np.ndarray  # (n_particles, 3) float64
    element_ids: np.ndarray  # (n_particles,) int32 - cached for L0
    block_ids: np.ndarray  # (n_particles,) int32
    active: np.ndarray  # (n_particles,) bool

    # Memory: 24 + 4 + 4 + 1 = 33 bytes/particle
```

### Tests

**Files**: 3 test files, 12+ tests
- `test_seeding.py` - Reproducibility, bounds
- `test_cpu_search.py` - Correctness, performance
- `test_particle_state.py` - Memory footprint

**Run**: `pytest tests/gpu/test_seeding.py test_cpu_search.py test_particle_state.py -v`

### Deliverables

1.  **3 new modules**: `seeding.py`, `cpu_search.py`, `state.py`
2.  **3 test files**: 12+ tests
3.  **Documentation**: `docs/phase3/PARTICLE_INITIALIZATION_COMPLETE.md`

### Success Criteria

-  Seeding: 1M particles in <1s
-  Initial search: >98% found
-  Performance: <30s for 10K particles (CPU)
-  Memory: 33 MB for 1M particles
-  All tests passing (12+)

### Phase 3 Completion

**Commit Message**:
```
Phase 3: Particle seeding and initial assignment complete

- Seeded 1M particles in 0.8s
- CPU search: 10K particles in 24s (98.5% found)
- Particle state: 33 bytes/particle = 33 MB for 1M
- Tests: 12 passing
```

---

## PHASE 4: GPU MULTI-LEVEL ELEMENT SEARCH

**Duration**: 3-5 days
**Dependencies**: Phase 3 complete
**Goal**: Implement 4-level search hierarchy in JAX, validate vs CPU

### Objectives

1. Implement point-in-tet kernel (JAX)
2. Level 0: Cached element (85-95% hit rate)
3. Level 1: Neighbor elements (3-10% hit rate)
4. Level 2: Block elements (1-5% hit rate)
5. Level 3: 26 neighbor blocks (0.1-1% hit rate)
6. Validate: Match CPU results, memory <500 MB

### Multi-Level Search Hierarchy

```
                                     
 find_element_multi_level()          
                                     
                                  
   L0: Cached Element             
     Cost: 1 test (~5 ns)         
     Hit rate: 85-95%             
     Early exit if found         
                                  
               miss                 
              ¼                      
                                  
   L1: Neighbor Elements          
     Cost: 4 tests (~50 ns)       
     Hit rate: 3-10%              
     Early exit if found         
                                  
               miss                 
              ¼                      
                                  
   L2: Block Elements             
     Cost: ~150K tests (50 ¼s)    
     Hit rate: 1-5%               
     Early exit if found         
                                  
               miss                 
              ¼                      
                                  
   L3: Neighbor Blocks            
     Cost: 26×150K tests (1 ms)   
     Hit rate: 0.1-1%             
     Return element or -1         
                                  
                                     
```

**Expected Average**: 0.9×5ns + 0.08×50ns + 0.015×50¼s H 1-2 ¼s/particle

### Tasks

#### Task 4.1: Point-in-Tet Kernel (JAX)

**File**: `jaxtrace/gpu/geometry/point_in_tet.py` (NEW, ~250 lines)

**Key Functions**:
```python
@jax.jit
def point_in_tet_barycentric(
    point: jax.Array,  # (3,)
    tet_nodes: jax.Array  # (4, 3)
) -> jax.Array:
    """
    Test if point inside tetrahedron using barycentric coordinates.

    Returns: bool

    Algorithm:
        Solve: point = »0*n0 + »1*n1 + »2*n2 + »3*n3
        Constraint: »0 + »1 + »2 + »3 = 1
        Inside iff all »i  [0, 1]
    """
```

#### Task 4.2-4.5: Multi-Level Search Modules

**Files**: 5 new modules (~1500 lines total)
- `jaxtrace/gpu/search/level0_cached.py` (~200 lines)
- `jaxtrace/gpu/search/level1_neighbors.py` (~250 lines)
- `jaxtrace/gpu/search/level2_block.py` (~300 lines)
- `jaxtrace/gpu/search/level3_neighbor_blocks.py` (~350 lines)
- `jaxtrace/gpu/search/multi_level.py` (~400 lines)

**Key Features**:
- All JAX JIT-compilable
- Early termination with `lax.cond`
- Vectorized with `vmap` over particles
- Static shapes (no dynamic slicing)

#### Task 4.6: Integrated Multi-Level Search

**File**: `jaxtrace/gpu/search/multi_level.py`

**Key Function**:
```python
@jax.jit
def find_elements_batch(
    positions: jax.Array,  # (n_particles, 3)
    cached_element_ids: jax.Array,  # (n_particles,)
    block_ids: jax.Array,  # (n_particles,)
    # Static mesh data (NOT in scan carry!)
    node_positions: jax.Array,
    element_nodes: jax.Array,
    element_neighbors: jax.Array,
    padded_arrays: PaddedBlockArrays
) -> Tuple[jax.Array, jax.Array]:
    """
    Returns:
        element_ids: (n_particles,) int32
        levels_found: (n_particles,) int32
    """
```

### Tests

**Files**: 2 test files, 20+ tests
- `tests/gpu/test_point_in_tet.py` (~200 lines, 8 tests)
- `tests/gpu/test_multi_level_search.py` (~400 lines, 12 tests)

**Key Tests**:
- `test_level0_hit()` - Particle still in element
- `test_level1_hit()` - Moved to neighbor
- `test_level2_hit()` - Moved within block
- `test_level3_hit()` - Crossed block boundary
- `test_batch_1M()` - 1M particles, memory <500 MB
- `test_accuracy_vs_cpu()` - 100% match

**Run**: `pytest tests/gpu/test_point_in_tet.py test_multi_level_search.py -v`

### Deliverables

1.  **6 new modules**: 1500 lines total
2.  **2 test files**: 20+ tests
3.  **Documentation**: `docs/phase4/GPU_MULTI_LEVEL_SEARCH_COMPLETE.md` (~25 pages)
   - 4-level search flowchart
   - Hit rate statistics table
   - Memory comparison V4 vs V5
   - Performance benchmarks CPU vs GPU
   - JIT compilation notes

4.  **Figures**:
   - Search hierarchy flowchart
   - Hit rate comparison chart
   - Performance scaling plot

### Success Criteria

-  **L0 hit rate**: 85-95% (matches CPU baseline)
-  **L1 hit rate**: 3-10%
-  **L2 hit rate**: 1-5%
-  **L3 hit rate**: 0.1-1%
-  **Total found**: >98%
-  **Memory**: <500 MB for 1M particles (vs 45 GB V4)
-  **Accuracy**: 100% match with CPU search
-  **Performance**: >10× speedup vs CPU for 1M particles
-  **JIT compilation**: No tracing errors
-  **All tests passing**: 20+ tests

### Phase 4 Completion

**Commit Message**:
```
Phase 4: GPU multi-level element search complete

- Implemented 4-level search hierarchy in JAX
- Hit rates: L0: 89%, L1: 8%, L2: 3%, L3: 0.1%
- Memory: 372 MB for 1M particles (vs 45 GB V4)
- Speedup: 15× vs CPU
- Accuracy: 100% match with CPU
- JIT compilation: Working
- Tests: 20 passing

GPU particle tracking foundation complete.
```

---

## PHASES 5-9: FUTURE PHASES (SUMMARY)

### Phase 5: Time Integration & Field Interpolation (1 week)
- RK4 integrator (JAX)
- FEM barycentric interpolation
- lax.scan time loop with minimal carry
- Velocity field sampling

### Phase 6: Particle Rebatching (3-4 days)
- Sort particles by block_id after each step
- Maintain spatial locality

### Phase 7: Ghost Regions (3-4 days)
- Extract 1-layer ghost elements at block boundaries
- Extended search to include ghosts

### Phase 8: Optimization (1 week)
- Batch size tuning
- GPU occupancy analysis
- Memory profiling

### Phase 9: Hash Octree (Optional, 1-2 weeks)
- O(1) Morton code-based lookup
- For blocks with >200K elements

---

## APPENDIX

### A. File Structure

```
jaxtrace/gpu/
   forest/
      block_grid.py           (Phase 1)
      element_mapper.py       (Phase 1)
      load_balance.py         (Phase 1)
      visualize_forest.py     (Phase 1)
   mesh/
      element_neighbors.py    (Phase 2)
   arrays/
      padded_blocks.py        (Phase 2)
      validation.py           (Phase 2)
   particles/
      seeding.py              (Phase 3)
      state.py                (Phase 3)
   search/
      cpu_search.py           (Phase 3)
      level0_cached.py        (Phase 4)
      level1_neighbors.py     (Phase 4)
      level2_block.py         (Phase 4)
      level3_neighbor_blocks.py (Phase 4)
      multi_level.py          (Phase 4)
   geometry/
       point_in_tet.py         (Phase 4)

tests/gpu/
   test_block_grid.py          (Phase 1)
   test_element_mapper.py      (Phase 1)
   test_element_neighbors.py   (Phase 2)
   test_padded_blocks.py       (Phase 2)
   test_seeding.py             (Phase 3)
   test_cpu_search.py          (Phase 3)
   test_particle_state.py      (Phase 3)
   test_point_in_tet.py        (Phase 4)
   test_multi_level_search.py  (Phase 4)

docs/
   phase0/THREADEDA_MESH_VALIDATED.md
   phase1/FOREST_STRUCTURE_COMPLETE.md
   phase2/PADDED_ARRAYS_COMPLETE.md
   phase3/PARTICLE_INITIALIZATION_COMPLETE.md
   phase4/GPU_MULTI_LEVEL_SEARCH_COMPLETE.md

config/
   threadeda_recommended.yaml
```

### B. Timeline

| Phase | Duration | Deliverables | Tests |
|-------|----------|--------------|-------|
| 0 | 1 day | Mesh analysis, config | Validation |
| 1 | 2-3 days | Forest structure | 22 tests |
| 2 | 2-3 days | Padded arrays | 22 tests |
| 3 | 1-2 days | Particle seeding | 12 tests |
| 4 | 3-5 days | GPU multi-level search | 20 tests |
| **Total** | **9-14 days** | **4 phases** | **76+ tests** |

### C. Success Metrics (Phase 4 Completion)

**Functional**:
-  1M particles tracked
-  >98% found in mesh
-  100% accuracy vs CPU
-  Multi-level hit rates:
  - L0: 85-95%
  - L1: 3-10%
  - L2: 1-5%
  - L3: 0.1-1%

**Performance**:
-  GPU >10× faster than CPU
-  Memory <500 MB (vs 45 GB V4)
-  Search <10 ¼s/particle

**Code Quality**:
-  76+ tests passing
-  Docs complete
-  Clean git history

### D. Memory Budget Details

**Static Data** (GPU-resident, uploaded once):
```
Node positions:           900K × 12 bytes   =  10.8 MB
Element nodes:            3.5M × 16 bytes   =  56.0 MB
Element neighbors:        3.5M × 16 bytes   =  56.0 MB
Element-to-block:         3.5M × 4 bytes    =  14.0 MB
Block elements (padded):  32 × 150K × 4     = 192.0 MB  V5 solution
Block neighbors_26:       32 × 26 × 4       =   0.003 MB
Velocity field:           900K × 12 bytes   =  10.8 MB
----------------------------------------------------------
Total Static:                                 339.6 MB 
```

**Particle Data** (scan carry, dynamic):
```
Positions:                1M × 24 bytes      =  24.0 MB
Element IDs:              1M × 4 bytes       =   4.0 MB
Block IDs:                1M × 4 bytes       =   4.0 MB
Active mask:              1M × 1 byte        =   1.0 MB
----------------------------------------------------------
Total Particles:                               33.0 MB 
```

**TOTAL: 372.6 MB**  (fits easily in 4GB GPU)

**Comparison to V4**: 45,000 MB (120× worse) L

---

**END OF PLAN**

**Next Step**: Phase 0 review and approval before implementation.

The user will now review this plan phase-by-phase. Implementation begins only after phase approval.
