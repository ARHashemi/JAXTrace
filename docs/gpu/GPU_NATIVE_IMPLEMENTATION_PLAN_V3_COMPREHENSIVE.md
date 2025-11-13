# GPU-Native Particle Tracking: Comprehensive Implementation Plan V3

**Status**: Phase-by-phase incremental development plan
**Date**: 2025-11-03
**Architecture**: Flat arrays, JAX-native, minimal scan carry
**Target**: 1M particles, 3.5M elements, 10-100× CPU speedup

---

## Executive Summary

This plan implements GPU-native particle tracking using **flat static arrays** with **incremental, testable phases**. Key principles:

1. ✅ **Flat arrays only** - all data structures are fixed-size JAX arrays
2. ✅ **Minimal scan carry** - only particle positions/element_IDs/active mask
3. ✅ **Static mesh/field data** - never in scan carry, passed as constants
4. ✅ **Configurable storage** - padded vs flat, nodes vs elements
5. ✅ **Incremental development** - each phase is independently testable
6. ✅ **Memory safety** - guaranteed no memory explosion
7. ✅ **JAX-optimal** - designed for XLA fusion and GPU coalescing

---

## Configuration System

### User-Configurable Options

```python
@dataclass
class GPUConfig:
    """
    Configuration for GPU particle tracking.

    All options have sensible defaults optimized for JAX GPU execution.
    Users can override based on mesh characteristics and memory constraints.
    """

    # ========================================================================
    # STORAGE OPTIONS
    # ========================================================================

    field_storage: str = "nodes"
    """Field data storage: 'nodes' or 'elements'
    - 'nodes': (N_nodes, 3) - memory efficient, requires gather [RECOMMENDED]
    - 'elements': (N_elements, 4, 3) - faster access, duplicated data
    """

    octree_storage: str = "padded"
    """Octree element storage: 'padded' or 'flat'
    - 'padded': (N_nodes, max_elem) - static shape, JAX-optimal [RECOMMENDED]
    - 'flat': flat array + start/count - compact if high variance
    """

    block_storage: str = "padded"
    """Block element storage: 'padded' or 'flat'
    - 'padded': (N_blocks, max_elem) - static shape [RECOMMENDED]
    - 'flat': flat array + start/count - compact storage
    """

    # ========================================================================
    # CAPACITY LIMITS (for padded storage)
    # ========================================================================

    max_neighbors: int = 4
    """Maximum neighbors per element (typical: 4 for tetrahedral mesh)"""

    max_octree_neighbors: int = 26
    """Maximum neighbor octree nodes (3D: 26, 2D: 8)"""

    max_octree_children: int = 8
    """Maximum children per octree node (3D: 8, 2D: 4)"""

    max_elements_per_octree_node: int = 1000
    """Maximum elements per octree leaf node (tune based on mesh)"""

    max_elements_per_block: int = 10000
    """Maximum elements per spatial block (tune based on mesh)"""

    max_particles_per_block: int = 100000
    """Maximum particles per block for batching"""

    # ========================================================================
    # OPTIONAL STORAGE (NOT RECOMMENDED)
    # ========================================================================

    store_particle_velocities: bool = False
    """Store particle velocities in scan carry [NOT RECOMMENDED]
    - If False: velocities interpolated per step (saves 24 bytes/particle)
    - If True: velocities stored in carry (use only if physics requires it)
    """

    store_particle_block_ids: bool = False
    """Store particle block IDs in scan carry [NOT RECOMMENDED]
    - If False: block IDs derived from element_block_IDs[elem_ID] (saves 4 bytes/particle)
    - If True: block IDs stored in carry (redundant, not recommended)
    """

    # ========================================================================
    # PRECISION
    # ========================================================================

    position_dtype: str = "float64"
    """Particle position precision (float64 for accuracy)"""

    velocity_dtype: str = "float32"
    """Field velocity precision (float32 sufficient for most cases)"""

    mesh_dtype: str = "float32"
    """Mesh coordinate precision (float32 sufficient)"""

    # ========================================================================
    # PERFORMANCE TUNING
    # ========================================================================

    n_blocks: int = 32
    """Number of spatial blocks (tune based on mesh size, typically 32-256)"""

    morton_code_bits: int = 21
    """Bits per dimension for Morton code (21 = 2^21 cells per dim = 2M^3 total)"""

    enable_jit: bool = True
    """JIT compile all kernels (should always be True for performance)"""

    use_hash_octree: bool = True
    """Use hash table for O(1) element search (Phase 9 feature)"""

    particles_per_vmap_batch: int = 10000
    """Particles per vmap batch (tune to avoid GPU OOM)"""

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def validate(self):
        """Validate configuration and warn about suboptimal choices."""
        assert self.field_storage in ["nodes", "elements"], \
            f"Invalid field_storage: {self.field_storage}"
        assert self.octree_storage in ["padded", "flat"], \
            f"Invalid octree_storage: {self.octree_storage}"
        assert self.block_storage in ["padded", "flat"], \
            f"Invalid block_storage: {self.block_storage}"

        if self.store_particle_velocities:
            warnings.warn(
                "Storing particle velocities increases memory by 24 bytes/particle. "
                "Consider deriving velocities per step unless physics requires storage."
            )

        if self.store_particle_block_ids:
            warnings.warn(
                "Storing particle block IDs is redundant (can be derived from element_block_IDs). "
                "This adds 4 bytes/particle to scan carry unnecessarily."
            )

        if self.octree_storage == "flat" or self.block_storage == "flat":
            warnings.warn(
                "Flat storage requires lax.dynamic_slice which is slightly slower than padded indexing. "
                "Use only if padding waste is significant (>50%)."
            )

        if self.field_storage == "elements":
            warnings.warn(
                "Element-based field storage duplicates data at shared nodes. "
                "Memory usage: N_elements × 4 vs N_nodes (typically 6× more). "
                "Use only if profiling shows gather is a bottleneck."
            )

    def memory_estimate(self, n_particles: int, n_nodes: int, n_elements: int) -> Dict[str, float]:
        """
        Estimate GPU memory usage in MB.

        Args:
            n_particles: Number of particles
            n_nodes: Number of mesh nodes
            n_elements: Number of mesh elements

        Returns:
            Dict with memory breakdown in MB
        """
        memory = {}

        # Scan carry (dynamic)
        carry_bytes_per_particle = (
            3 * 8 +  # positions (float64)
            1 * 4 +  # element_IDs (int32)
            1 * 1    # active (bool)
        )
        if self.store_particle_velocities:
            carry_bytes_per_particle += 3 * 8  # velocities (float64)
        if self.store_particle_block_ids:
            carry_bytes_per_particle += 1 * 4  # block_IDs (int32)

        memory['scan_carry'] = n_particles * carry_bytes_per_particle / 1024**2

        # Static mesh data
        memory['node_positions'] = n_nodes * 3 * 4 / 1024**2  # float32
        memory['element_nodes'] = n_elements * 4 * 4 / 1024**2  # int32
        memory['element_neighbors'] = n_elements * self.max_neighbors * 4 / 1024**2
        memory['element_block_IDs'] = n_elements * 4 / 1024**2

        # Static field data
        if self.field_storage == "nodes":
            memory['velocities'] = n_nodes * 3 * 4 / 1024**2
        else:  # elements
            memory['velocities'] = n_elements * 4 * 3 * 4 / 1024**2

        # Octree data (estimate)
        n_octree_nodes = self.n_blocks * 8  # Rough estimate
        if self.octree_storage == "padded":
            memory['octree_elements'] = (
                n_octree_nodes * self.max_elements_per_octree_node * 4 / 1024**2
            )
        else:  # flat
            memory['octree_elements'] = n_elements * 4 / 1024**2  # At most all elements

        memory['octree_metadata'] = n_octree_nodes * (3 * 4 + 3 * 4 + 8 * 4) / 1024**2

        # Block data
        if self.block_storage == "padded":
            memory['block_elements'] = (
                self.n_blocks * self.max_elements_per_block * 4 / 1024**2
            )
        else:  # flat
            memory['block_elements'] = n_elements * 4 / 1024**2

        # Total
        memory['total'] = sum(memory.values())

        return memory

    def print_config(self):
        """Print configuration summary."""
        print("=" * 80)
        print("GPU Configuration")
        print("=" * 80)
        print(f"Storage Options:")
        print(f"  Field storage:  {self.field_storage}")
        print(f"  Octree storage: {self.octree_storage}")
        print(f"  Block storage:  {self.block_storage}")
        print(f"\nCapacity Limits:")
        print(f"  Max neighbors:              {self.max_neighbors}")
        print(f"  Max elements per octree:    {self.max_elements_per_octree_node:,}")
        print(f"  Max elements per block:     {self.max_elements_per_block:,}")
        print(f"  Max particles per block:    {self.max_particles_per_block:,}")
        print(f"\nOptional Storage:")
        print(f"  Store particle velocities:  {self.store_particle_velocities}")
        print(f"  Store particle block IDs:   {self.store_particle_block_ids}")
        print(f"\nPerformance:")
        print(f"  Number of blocks:           {self.n_blocks}")
        print(f"  JIT compilation:            {self.enable_jit}")
        print(f"  Hash octree:                {self.use_hash_octree}")
        print("=" * 80)


# Default configuration
DEFAULT_CONFIG = GPUConfig()
```

---

## Phase 0: Mesh Analysis and Infrastructure Bootstrapping

**Duration**: 1 week
**Dependencies**: None
**Goal**: Understand mesh characteristics and set up testing infrastructure

### Objectives

1. Analyze ThreadedA mesh statistics
2. Set up testing infrastructure
3. Create synthetic test meshes
4. Establish CI/CD pipeline

### Tasks

#### 0.1: Mesh Analysis Tools

```python
# File: jaxtrace/gpu/mesh_analysis.py

def analyze_mesh_statistics(mesh_path: Path) -> Dict:
    """
    Analyze mesh for GPU partitioning decisions.

    Returns statistics that inform configuration:
    - Element count distribution per potential block
    - Node sharing patterns (affects field storage choice)
    - Neighbor connectivity statistics
    - Spatial bounding box
    """
    stats = {
        'n_nodes': ...,
        'n_elements': ...,
        'element_size_histogram': ...,
        'node_degree_distribution': ...,  # How many elements share each node
        'spatial_extent': ...,
        'recommended_n_blocks': ...,  # Based on element count
        'recommended_max_elements_per_block': ...,
    }
    return stats


def visualize_block_partitioning(
    mesh: Mesh,
    n_blocks: int,
    morton_bits: int = 21
) -> plt.Figure:
    """
    Visualize proposed block partitioning.

    Shows:
    - 3D mesh with block boundaries
    - Element count per block (histogram)
    - Load imbalance factor
    """
    pass


def recommend_config(mesh_stats: Dict) -> GPUConfig:
    """
    Recommend configuration based on mesh analysis.

    Logic:
    - If node_degree > 6: use node-based field storage
    - If element_variance < 2×: use padded storage
    - If element_variance > 10×: use flat storage
    - n_blocks: mesh_elements // 100K (typical)
    """
    pass
```

#### 0.2: Synthetic Test Mesh Generator

```python
# File: jaxtrace/gpu/test_meshes.py

def create_unit_cube_mesh(
    nx: int = 4,
    ny: int = 4,
    nz: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create simple structured tetrahedral mesh for testing.

    Returns:
        node_positions: (N_nodes, 3)
        element_nodes: (N_elements, 4)
    """
    pass


def create_test_velocity_field(
    node_positions: np.ndarray,
    field_type: str = "uniform"
) -> np.ndarray:
    """
    Create test velocity field.

    field_type options:
    - "uniform": constant velocity [1, 0, 0]
    - "rotation": solid body rotation
    - "vortex": 3D vortex field
    - "shear": linear shear flow
    """
    pass
```

#### 0.3: Testing Infrastructure

```python
# File: tests/gpu/conftest.py

import pytest


@pytest.fixture
def small_test_mesh():
    """2×2×2 cube mesh (48 elements) for unit tests."""
    return create_unit_cube_mesh(2, 2, 2)


@pytest.fixture
def medium_test_mesh():
    """10×10×10 cube mesh (6K elements) for integration tests."""
    return create_unit_cube_mesh(10, 10, 10)


@pytest.fixture
def default_gpu_config():
    """Default GPU configuration for tests."""
    return GPUConfig(
        n_blocks=8,
        max_elements_per_block=1000,
        enable_jit=True
    )


@pytest.fixture
def threadedA_mesh():
    """ThreadedA mesh (3.5M elements) for performance tests."""
    # Load from disk
    pass
```

### Deliverables

- [ ] `jaxtrace/gpu/mesh_analysis.py` - Analysis tools
- [ ] `jaxtrace/gpu/test_meshes.py` - Synthetic mesh generator
- [ ] `tests/gpu/conftest.py` - Testing fixtures
- [ ] `docs/mesh_analysis_threadedA.md` - ThreadedA statistics report
- [ ] Visualization notebooks showing block partitioning
- [ ] CI pipeline configuration (GitHub Actions)

### Success Criteria

✅ Can load and analyze ThreadedA mesh
✅ Recommended configuration generated from mesh stats
✅ Synthetic meshes generated for all test sizes
✅ All tests pass on CPU (no GPU required yet)
✅ CI pipeline runs successfully

---

## Phase 1: Load Mesh and Flat Data Structures

**Duration**: 1 week
**Dependencies**: Phase 0
**Goal**: Create JAX-compatible flat arrays from mesh data

### Objectives

1. Load mesh into flat arrays
2. Build element-node connectivity
3. Build element-neighbor connectivity
4. Load field data (velocities)
5. All arrays must be JAX-compatible

### Data Structures (Minimal)

```python
# ============================================================================
# MESH DATA - ALL STATIC, NEVER IN SCAN CARRY
# ============================================================================

# Nodes
node_positions: jnp.ndarray  # (N_nodes, 3) float32

# Elements
element_nodes: jnp.ndarray        # (N_elements, 4) int32
element_neighbors: jnp.ndarray    # (N_elements, max_neighbors) int32, padded with -1
element_block_IDs: jnp.ndarray    # (N_elements,) int32 - assigned in Phase 2

# Field data (configurable)
if config.field_storage == "nodes":
    velocities: jnp.ndarray       # (N_nodes, 3) float32
else:  # elements
    element_velocities: jnp.ndarray  # (N_elements, 4, 3) float32
```

### Tasks

#### 1.1: Mesh Loader

```python
# File: jaxtrace/gpu/mesh_loader.py

def load_mesh_to_flat_arrays(
    mesh_path: Path,
    config: GPUConfig
) -> Dict[str, jnp.ndarray]:
    """
    Load mesh and convert to flat JAX arrays.

    Returns dict with keys:
    - 'node_positions': (N_nodes, 3) float32
    - 'element_nodes': (N_elements, 4) int32
    - All arrays are JAX DeviceArrays, ready for GPU
    """
    # Load with existing VTK reader
    mesh_data = load_vtk_mesh(mesh_path)

    # Convert to JAX arrays with correct dtypes
    node_positions = jnp.array(mesh_data['points'], dtype=jnp.float32)
    element_nodes = jnp.array(mesh_data['connectivity'], dtype=jnp.int32)

    return {
        'node_positions': node_positions,
        'element_nodes': element_nodes,
    }
```

#### 1.2: Neighbor Builder

```python
# File: jaxtrace/gpu/neighbor_builder.py

def build_element_neighbors(
    element_nodes: np.ndarray,
    max_neighbors: int = 4
) -> np.ndarray:
    """
    Build element neighbor array (padded).

    Algorithm:
    1. Build face-to-element hash map
    2. For each element, find elements sharing faces
    3. Pad to max_neighbors with -1

    Returns:
        element_neighbors: (N_elements, max_neighbors) int32
    """
    n_elements = element_nodes.shape[0]
    neighbors = np.full((n_elements, max_neighbors), -1, dtype=np.int32)

    # Build face connectivity
    face_to_elements = defaultdict(list)
    for elem_id in range(n_elements):
        faces = extract_faces(element_nodes[elem_id])
        for face in faces:
            face_key = tuple(sorted(face))
            face_to_elements[face_key].append(elem_id)

    # Assign neighbors
    for elem_id in range(n_elements):
        faces = extract_faces(element_nodes[elem_id])
        neigh_list = []
        for face in faces:
            face_key = tuple(sorted(face))
            for other_elem in face_to_elements[face_key]:
                if other_elem != elem_id and other_elem not in neigh_list:
                    neigh_list.append(other_elem)

        # Pad to max_neighbors
        for i, neigh in enumerate(neigh_list[:max_neighbors]):
            neighbors[elem_id, i] = neigh

    return neighbors
```

#### 1.3: Field Loader

```python
# File: jaxtrace/gpu/field_loader.py

def load_velocity_field(
    mesh_path: Path,
    config: GPUConfig
) -> jnp.ndarray:
    """
    Load velocity field in configured format.

    Returns:
        If config.field_storage == "nodes":
            velocities: (N_nodes, 3) float32
        If config.field_storage == "elements":
            element_velocities: (N_elements, 4, 3) float32
    """
    # Load node-based velocities from VTK
    node_velocities = load_vtk_point_data(mesh_path, 'velocity')

    if config.field_storage == "nodes":
        return jnp.array(node_velocities, dtype=jnp.float32)

    else:  # Convert to element-based
        element_velocities = np.zeros((n_elements, 4, 3), dtype=np.float32)
        for elem_id in range(n_elements):
            node_ids = element_nodes[elem_id]
            element_velocities[elem_id] = node_velocities[node_ids]

        return jnp.array(element_velocities, dtype=jnp.float32)
```

### Deliverables

- [ ] `jaxtrace/gpu/mesh_loader.py` - Mesh loading
- [ ] `jaxtrace/gpu/neighbor_builder.py` - Neighbor connectivity
- [ ] `jaxtrace/gpu/field_loader.py` - Field data loading
- [ ] `tests/gpu/test_mesh_loader.py` - Unit tests
- [ ] Documentation for data formats

### Success Criteria

✅ ThreadedA mesh loads into flat arrays
✅ All arrays are JAX DeviceArrays
✅ Element neighbors correctly identified (validate with known meshes)
✅ Field data in both node and element formats
✅ Memory usage matches estimates
✅ All tests pass

---

## Phase 2: Block/Octree Partitioning & Morton Codes

**Duration**: 1.5 weeks
**Dependencies**: Phase 1
**Goal**: Partition mesh into spatial blocks using Morton codes

### Objectives

1. Compute Morton codes for element centroids
2. Assign elements to blocks
3. Build block element arrays (padded or flat)
4. Build octree structure (optional, for Phase 9)

### Data Structures

```python
# ============================================================================
# BLOCK PARTITIONING
# ============================================================================

element_block_IDs: jnp.ndarray    # (N_elements,) int32

# OPTION 1: Padded (RECOMMENDED)
if config.block_storage == "padded":
    block_elements: jnp.ndarray   # (N_blocks, max_elements_per_block) int32

# OPTION 2: Flat
if config.block_storage == "flat":
    block_elements_flat: jnp.ndarray      # (total_refs,) int32
    block_element_start: jnp.ndarray      # (N_blocks,) int32
    block_element_count: jnp.ndarray      # (N_blocks,) int32

# ============================================================================
# OCTREE STRUCTURE (for Phase 9)
# ============================================================================

octree_node_centers: jnp.ndarray      # (N_octree_nodes, 3) float32
octree_node_halfsize: jnp.ndarray     # (N_octree_nodes, 3) float32
octree_node_children: jnp.ndarray     # (N_octree_nodes, 8) int32, -1 if leaf
octree_node_block_IDs: jnp.ndarray    # (N_octree_nodes,) int32
octree_node_neighbors: jnp.ndarray    # (N_octree_nodes, 26) int32

# Octree elements (configurable)
if config.octree_storage == "padded":
    octree_node_elements: jnp.ndarray  # (N_nodes, max_elem_per_node) int32
else:  # flat
    octree_elements: jnp.ndarray       # (flat) int32
    octree_element_start: jnp.ndarray  # (N_nodes,) int32
    octree_element_count: jnp.ndarray  # (N_nodes,) int32
```

### Tasks

#### 2.1: Morton Code Generator

```python
# File: jaxtrace/gpu/morton.py

def compute_morton_codes(
    element_centroids: np.ndarray,
    domain_bounds: np.ndarray,
    bits_per_dim: int = 21
) -> np.ndarray:
    """
    Compute Morton Z-order codes for element centroids.

    Args:
        element_centroids: (N_elements, 3) float32
        domain_bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
        bits_per_dim: Bits per dimension (21 = 2M cells per dim)

    Returns:
        morton_codes: (N_elements,) uint64

    Algorithm:
        1. Normalize coordinates to [0, 2^bits - 1]
        2. Interleave bits: x0 y0 z0 x1 y1 z1 ...
        3. Result: 3 × bits_per_dim = 63 bits total
    """
    # Normalize to [0, 2^bits - 1]
    xmin, xmax, ymin, ymax, zmin, zmax = domain_bounds
    max_val = (1 << bits_per_dim) - 1

    x = ((element_centroids[:, 0] - xmin) / (xmax - xmin) * max_val).astype(np.uint64)
    y = ((element_centroids[:, 1] - ymin) / (ymax - ymin) * max_val).astype(np.uint64)
    z = ((element_centroids[:, 2] - zmin) / (zmax - zmin) * max_val).astype(np.uint64)

    # Interleave bits
    morton_codes = interleave_bits_3d(x, y, z)

    return morton_codes


@jax.jit
def interleave_bits_3d(x, y, z):
    """Interleave bits of 3 integers (Morton encoding)."""
    # Bit manipulation for Z-order curve
    # (Implementation details omitted for brevity)
    pass
```

#### 2.2: Block Assignment

```python
# File: jaxtrace/gpu/block_assignment.py

def assign_elements_to_blocks(
    morton_codes: np.ndarray,
    n_blocks: int
) -> Tuple[np.ndarray, Dict]:
    """
    Assign elements to blocks based on Morton codes.

    Algorithm:
    1. Sort elements by Morton code
    2. Divide into n_blocks contiguous ranges
    3. Assign block ID to each element

    Returns:
        element_block_IDs: (N_elements,) int32
        block_metadata: Dict with per-block statistics
    """
    n_elements = len(morton_codes)
    sorted_indices = np.argsort(morton_codes)

    # Assign blocks
    elements_per_block = n_elements // n_blocks
    element_block_IDs = np.zeros(n_elements, dtype=np.int32)

    for block_id in range(n_blocks):
        start = block_id * elements_per_block
        end = start + elements_per_block if block_id < n_blocks - 1 else n_elements
        block_elem_indices = sorted_indices[start:end]
        element_block_IDs[block_elem_indices] = block_id

    # Compute statistics
    block_counts = np.bincount(element_block_IDs, minlength=n_blocks)
    metadata = {
        'block_element_counts': block_counts,
        'min_elements': block_counts.min(),
        'max_elements': block_counts.max(),
        'mean_elements': block_counts.mean(),
        'load_imbalance': block_counts.max() / block_counts.mean(),
    }

    return element_block_IDs, metadata
```

#### 2.3: Build Block Element Arrays

```python
# File: jaxtrace/gpu/block_builder.py

def build_block_elements_padded(
    element_block_IDs: np.ndarray,
    n_blocks: int,
    max_elements_per_block: int
) -> np.ndarray:
    """
    Build padded block element array.

    Returns:
        block_elements: (N_blocks, max_elements_per_block) int32
        Padded with -1 for blocks with fewer elements
    """
    block_elements = np.full(
        (n_blocks, max_elements_per_block),
        -1,
        dtype=np.int32
    )

    for block_id in range(n_blocks):
        elem_ids = np.where(element_block_IDs == block_id)[0]
        count = min(len(elem_ids), max_elements_per_block)
        block_elements[block_id, :count] = elem_ids[:count]

        if len(elem_ids) > max_elements_per_block:
            warnings.warn(
                f"Block {block_id} has {len(elem_ids)} elements, "
                f"truncated to {max_elements_per_block}"
            )

    return block_elements


def build_block_elements_flat(
    element_block_IDs: np.ndarray,
    n_blocks: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build flat block element array with start/count.

    Returns:
        block_elements_flat: (total_refs,) int32
        block_element_start: (N_blocks,) int32
        block_element_count: (N_blocks,) int32
    """
    block_elem_lists = [[] for _ in range(n_blocks)]
    for elem_id, block_id in enumerate(element_block_IDs):
        block_elem_lists[block_id].append(elem_id)

    # Flatten
    block_elements_flat = []
    block_element_start = np.zeros(n_blocks, dtype=np.int32)
    block_element_count = np.zeros(n_blocks, dtype=np.int32)

    offset = 0
    for block_id in range(n_blocks):
        block_element_start[block_id] = offset
        block_element_count[block_id] = len(block_elem_lists[block_id])
        block_elements_flat.extend(block_elem_lists[block_id])
        offset += len(block_elem_lists[block_id])

    return (
        np.array(block_elements_flat, dtype=np.int32),
        block_element_start,
        block_element_count
    )
```

### Deliverables

- [ ] `jaxtrace/gpu/morton.py` - Morton code generator
- [ ] `jaxtrace/gpu/block_assignment.py` - Block assignment
- [ ] `jaxtrace/gpu/block_builder.py` - Block array builder
- [ ] `tests/gpu/test_morton.py` - Morton code tests
- [ ] `tests/gpu/test_block_assignment.py` - Block assignment tests
- [ ] Visualization of block partitioning for ThreadedA
- [ ] Performance report: block load imbalance

### Success Criteria

✅ Morton codes correctly computed and sorted
✅ Elements assigned to blocks with spatial coherence
✅ Block element arrays built in both padded and flat formats
✅ Load imbalance factor < 2.0× (ideal), acceptable up to 5×
✅ Visualization shows spatial clustering
✅ All tests pass

---

## Phase 3: Particle Data, Seeding, & Static Assignment

**Duration**: 1 week
**Dependencies**: Phase 1, 2
**Goal**: Initialize particles in flat arrays

### Objectives

1. Define minimal particle data structure
2. Implement particle seeding strategies
3. Find initial element for each particle
4. All arrays JAX-compatible

### Data Structures (Scan Carry - MINIMAL)

```python
# ============================================================================
# PARTICLE STATE - ONLY THIS IN SCAN CARRY
# ============================================================================

particle_positions: jnp.ndarray       # (N_particles, 3) float64
particle_element_IDs: jnp.ndarray     # (N_particles,) int32
particle_active: jnp.ndarray          # (N_particles,) bool

# OPTIONAL (NOT RECOMMENDED)
if config.store_particle_velocities:
    particle_velocities: jnp.ndarray  # (N_particles, 3) float64

if config.store_particle_block_ids:
    particle_block_IDs: jnp.ndarray   # (N_particles,) int32
```

### Tasks

#### 3.1: Particle Seeding

```python
# File: jaxtrace/gpu/particle_seeder.py

def seed_particles_uniform(
    domain_bounds: np.ndarray,
    n_particles: int,
    seed: int = 42
) -> np.ndarray:
    """
    Seed particles uniformly in domain.

    Args:
        domain_bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
        n_particles: Number of particles
        seed: Random seed

    Returns:
        particle_positions: (N_particles, 3) float64
    """
    rng = np.random.default_rng(seed)

    xmin, xmax, ymin, ymax, zmin, zmax = domain_bounds

    positions = np.zeros((n_particles, 3), dtype=np.float64)
    positions[:, 0] = rng.uniform(xmin, xmax, n_particles)
    positions[:, 1] = rng.uniform(ymin, ymax, n_particles)
    positions[:, 2] = rng.uniform(zmin, zmax, n_particles)

    return positions


def seed_particles_from_field(
    mesh_data: Dict,
    field_data: Dict,
    n_particles: int,
    method: str = "streamline"
) -> np.ndarray:
    """
    Seed particles based on field characteristics.

    Methods:
    - "streamline": Seed on inlet boundary, perpendicular to flow
    - "vortex": Seed in high vorticity regions
    - "weighted": Seed with probability ~ |velocity|
    """
    pass
```

#### 3.2: Initial Element Finding

```python
# File: jaxtrace/gpu/element_finder.py

def find_initial_elements(
    particle_positions: np.ndarray,
    mesh_data: Dict,
    config: GPUConfig
) -> np.ndarray:
    """
    Find initial element for each particle.

    Algorithm (CPU-based for initialization):
    1. Compute block ID from position (via Morton code)
    2. Get elements in that block
    3. Linear search through block elements
    4. Use point-in-tetrahedron test

    Returns:
        particle_element_IDs: (N_particles,) int32
        -1 if particle outside domain
    """
    n_particles = len(particle_positions)
    element_IDs = np.full(n_particles, -1, dtype=np.int32)

    for i, pos in enumerate(particle_positions):
        # Get block ID
        block_id = position_to_block_id(pos, domain_bounds, config.n_blocks)

        if block_id < 0:
            continue  # Outside domain

        # Get elements in block
        if config.block_storage == "padded":
            block_elem_ids = block_elements[block_id]
            block_elem_ids = block_elem_ids[block_elem_ids >= 0]  # Filter padding
        else:  # flat
            start = block_element_start[block_id]
            count = block_element_count[block_id]
            block_elem_ids = block_elements_flat[start:start+count]

        # Search elements
        for elem_id in block_elem_ids:
            if point_in_tetrahedron_cpu(pos, elem_id, mesh_data):
                element_IDs[i] = elem_id
                break

    return element_IDs
```

### Deliverables

- [ ] `jaxtrace/gpu/particle_seeder.py` - Seeding strategies
- [ ] `jaxtrace/gpu/element_finder.py` - Initial element finding
- [ ] `tests/gpu/test_particle_seeder.py` - Seeding tests
- [ ] `tests/gpu/test_element_finder.py` - Element finding tests
- [ ] Documentation for seeding strategies

### Success Criteria

✅ Particles seeded in various patterns
✅ Initial elements correctly found (>95% success rate for interior particles)
✅ Particle arrays are JAX DeviceArrays
✅ Memory usage: 29 bytes/particle (minimal config)
✅ All tests pass

---

## Phase 4: Local Element Search & Neighbor Caching

**Duration**: 2 weeks
**Dependencies**: Phase 3
**Goal**: Implement multi-level GPU element search

### Objectives

1. Level 0: Cached element search (85-95% hit rate)
2. Level 1: Neighbor element search (3-10% hit rate)
3. Level 2: Block element search (1-5% hit rate)
4. All fully vectorized with vmap
5. No dynamic allocation

### Search Algorithm

```python
@jax.jit
def multi_level_search(
    particle_pos: jnp.ndarray,           # (N_particles, 3)
    particle_elem_ID: jnp.ndarray,       # (N_particles,)
    mesh_data: Dict,
    config: GPUConfig
) -> jnp.ndarray:
    """
    Multi-level element search (GPU-optimized).

    Search hierarchy:
    1. Level 0: Check cached element (particle_elem_ID)
    2. Level 1: Check neighbor elements (element_neighbors[particle_elem_ID])
    3. Level 2: Check all elements in block (via element_block_IDs)

    Returns:
        new_element_IDs: (N_particles,) int32
    """
    # Level 0: Cached element
    found_L0, elem_L0 = search_level0_batch(
        particle_pos,
        particle_elem_ID,
        mesh_data['element_nodes'],
        mesh_data['node_positions']
    )

    # Level 1: Neighbors (only for particles not found in L0)
    found_L1, elem_L1 = search_level1_batch(
        particle_pos,
        particle_elem_ID,
        found_L0,
        mesh_data['element_neighbors'],
        mesh_data['element_nodes'],
        mesh_data['node_positions']
    )

    # Level 2: Block search (only for particles not found in L0 or L1)
    found_L2, elem_L2 = search_level2_batch(
        particle_pos,
        particle_elem_ID,
        found_L0,
        found_L1,
        mesh_data['element_block_IDs'],
        mesh_data['block_elements'],  # Or block_elements_flat + start/count
        mesh_data['element_nodes'],
        mesh_data['node_positions'],
        config
    )

    # Combine results
    new_elem_IDs = jnp.where(
        found_L0, elem_L0,
        jnp.where(found_L1, elem_L1,
        jnp.where(found_L2, elem_L2, -1))
    )

    return new_elem_IDs
```

### Tasks

#### 4.1: Point-in-Tetrahedron (GPU)

```python
# File: jaxtrace/gpu/geometry.py

@jax.jit
def point_in_tetrahedron_batch(
    points: jnp.ndarray,      # (N, 3)
    vertices: jnp.ndarray     # (N, 4, 3)
) -> jnp.ndarray:
    """
    Vectorized point-in-tetrahedron test.

    Algorithm:
    1. Compute barycentric coordinates via linear solve
    2. Check all coordinates >= -epsilon
    3. Check sum <= 1 + epsilon

    Returns:
        inside: (N,) bool
    """
    # Extract vertices
    v0, v1, v2, v3 = vertices[:, 0], vertices[:, 1], vertices[:, 2], vertices[:, 3]

    # Build matrix A = [v1-v0, v2-v0, v3-v0]
    A = jnp.stack([v1 - v0, v2 - v0, v3 - v0], axis=-1)  # (N, 3, 3)
    b = points - v0  # (N, 3)

    # Check condition number
    cond = jnp.linalg.cond(A)
    well_conditioned = cond < 1e6

    # Solve
    lambdas = jnp.linalg.solve(A, b)  # (N, 3)
    lambda0 = 1.0 - jnp.sum(lambdas, axis=1)

    # Check bounds
    epsilon = 1e-6
    valid = (
        (lambda0 >= -epsilon) &
        jnp.all(lambdas >= -epsilon, axis=1) &
        (jnp.sum(lambdas, axis=1) <= 1.0 + epsilon)
    )

    return valid & well_conditioned
```

#### 4.2: Level 0 Search (Cached)

```python
# File: jaxtrace/gpu/search_level0.py

@jax.jit
def search_level0_batch(
    particle_pos: jnp.ndarray,
    particle_elem_ID: jnp.ndarray,
    element_nodes: jnp.ndarray,
    node_positions: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Level 0: Check if particles still in cached elements.

    Returns:
        found: (N_particles,) bool
        element_IDs: (N_particles,) int32
    """
    valid = particle_elem_ID >= 0
    safe_IDs = jnp.where(valid, particle_elem_ID, 0)

    # Get vertices: (N_particles, 4, 3)
    elem_node_IDs = element_nodes[safe_IDs]
    vertices = node_positions[elem_node_IDs]

    # Check containment
    inside = point_in_tetrahedron_batch(particle_pos, vertices)

    found = inside & valid
    result = jnp.where(found, particle_elem_ID, -1)

    return found, result
```

#### 4.3: Level 1 Search (Neighbors)

```python
# File: jaxtrace/gpu/search_level1.py

@jax.jit
def search_level1_batch(
    particle_pos: jnp.ndarray,
    particle_elem_ID: jnp.ndarray,
    found_L0: jnp.ndarray,
    element_neighbors: jnp.ndarray,
    element_nodes: jnp.ndarray,
    node_positions: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Level 1: Check neighbor elements.

    Uses vmap to check all neighbors in parallel for each particle.
    """
    needs_search = ~found_L0 & (particle_elem_ID >= 0)
    safe_IDs = jnp.where(particle_elem_ID >= 0, particle_elem_ID, 0)

    # Get neighbors: (N_particles, max_neighbors)
    neighbors = element_neighbors[safe_IDs]

    def check_particle_neighbors(pos, neighs, search):
        """Check all neighbors for one particle."""
        def check_neighbor(neigh_ID):
            valid = neigh_ID >= 0
            safe_ID = jnp.where(valid, neigh_ID, 0)
            elem_node_IDs = element_nodes[safe_ID]
            verts = node_positions[elem_node_IDs]
            inside = point_in_tetrahedron_batch(
                pos.reshape(1, 3),
                verts.reshape(1, 4, 3)
            )[0]
            return valid & inside, jnp.where(valid & inside, neigh_ID, -1)

        # vmap over neighbors (fixed size: max_neighbors)
        found_arr, id_arr = jax.vmap(check_neighbor)(neighs)

        found_any = jnp.any(found_arr)
        first_idx = jnp.argmax(found_arr)
        result = jnp.where(found_any & search, id_arr[first_idx], -1)

        return found_any & search, result

    # vmap over all particles
    found, result = jax.vmap(check_particle_neighbors)(
        particle_pos, neighbors, needs_search
    )

    return found, result
```

#### 4.4: Level 2 Search (Block)

```python
# File: jaxtrace/gpu/search_level2.py

@jax.jit
def search_level2_batch_padded(
    particle_pos: jnp.ndarray,
    particle_elem_ID: jnp.ndarray,
    found_L0: jnp.ndarray,
    found_L1: jnp.ndarray,
    element_block_IDs: jnp.ndarray,
    block_elements: jnp.ndarray,  # (N_blocks, max_elem_per_block)
    element_nodes: jnp.ndarray,
    node_positions: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Level 2: Search block elements (padded storage).

    Key insight: Derive block ID from element, not store separately.
    """
    needs_search = ~found_L0 & ~found_L1 & (particle_elem_ID >= 0)

    # Derive block ID from cached element (NOT stored in particle state!)
    safe_elem = jnp.where(particle_elem_ID >= 0, particle_elem_ID, 0)
    particle_block_IDs = element_block_IDs[safe_elem]

    def check_particle_block(pos, block_id, search):
        """Check all elements in this particle's block."""
        if not search:
            return False, -1

        # Get elements in block
        block_elem_IDs = block_elements[block_id]  # (max_elem_per_block,)

        def check_element(elem_id):
            valid = elem_id >= 0
            safe_id = jnp.where(valid, elem_id, 0)
            elem_node_IDs = element_nodes[safe_id]
            verts = node_positions[elem_node_IDs]
            inside = point_in_tetrahedron_batch(
                pos.reshape(1, 3),
                verts.reshape(1, 4, 3)
            )[0]
            return valid & inside, jnp.where(valid & inside, elem_id, -1)

        # vmap over block elements (fixed size: max_elem_per_block)
        found_arr, id_arr = jax.vmap(check_element)(block_elem_IDs)

        found_any = jnp.any(found_arr)
        first_idx = jnp.argmax(found_arr)
        result = jnp.where(found_any, id_arr[first_idx], -1)

        return found_any, result

    # vmap over all particles
    found, result = jax.vmap(check_particle_block)(
        particle_pos, particle_block_IDs, needs_search
    )

    return found, result


@jax.jit
def search_level2_batch_flat(
    particle_pos: jnp.ndarray,
    particle_elem_ID: jnp.ndarray,
    found_L0: jnp.ndarray,
    found_L1: jnp.ndarray,
    element_block_IDs: jnp.ndarray,
    block_elements_flat: jnp.ndarray,
    block_element_start: jnp.ndarray,
    block_element_count: jnp.ndarray,
    element_nodes: jnp.ndarray,
    node_positions: jnp.ndarray,
    max_check: int = 1000
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Level 2: Search block elements (flat storage).

    Uses lax.dynamic_slice to access variable-length block element lists.
    """
    needs_search = ~found_L0 & ~found_L1 & (particle_elem_ID >= 0)

    # Derive block ID
    safe_elem = jnp.where(particle_elem_ID >= 0, particle_elem_ID, 0)
    particle_block_IDs = element_block_IDs[safe_elem]

    def check_particle_block(pos, block_id, search):
        if not search:
            return False, -1

        # Get block element range
        start = block_element_start[block_id]
        count = block_element_count[block_id]

        # Dynamic slice (JAX-compatible)
        elem_slice = jax.lax.dynamic_slice(
            block_elements_flat,
            (start,),
            (max_check,)
        )

        def check_element(idx, elem_id):
            valid = (idx < count) & (elem_id >= 0)
            safe_id = jnp.where(valid, elem_id, 0)
            elem_node_IDs = element_nodes[safe_id]
            verts = node_positions[elem_node_IDs]
            inside = point_in_tetrahedron_batch(
                pos.reshape(1, 3),
                verts.reshape(1, 4, 3)
            )[0]
            return valid & inside, jnp.where(valid & inside, elem_id, -1)

        indices = jnp.arange(max_check)
        found_arr, id_arr = jax.vmap(check_element)(indices, elem_slice)

        found_any = jnp.any(found_arr)
        first_idx = jnp.argmax(found_arr)
        result = jnp.where(found_any, id_arr[first_idx], -1)

        return found_any, result

    # vmap over all particles
    found, result = jax.vmap(check_particle_block)(
        particle_pos, particle_block_IDs, needs_search
    )

    return found, result
```

### Deliverables

- [ ] `jaxtrace/gpu/geometry.py` - Point-in-tet kernel
- [ ] `jaxtrace/gpu/search_level0.py` - Level 0 search
- [ ] `jaxtrace/gpu/search_level1.py` - Level 1 search
- [ ] `jaxtrace/gpu/search_level2.py` - Level 2 search (both padded and flat)
- [ ] `jaxtrace/gpu/search.py` - Combined multi-level search
- [ ] `tests/gpu/test_geometry.py` - Geometry tests
- [ ] `tests/gpu/test_search.py` - Search tests
- [ ] Performance benchmarks for each level

### Success Criteria

✅ Point-in-tet kernel is JIT-compiled and vectorized
✅ Level 0 achieves 85-95% hit rate
✅ Level 1 achieves 3-10% hit rate (of L0 misses)
✅ Level 2 finds remaining particles
✅ Block IDs derived on-the-fly (not stored)
✅ Both padded and flat storage work correctly
✅ Memory usage is constant per particle
✅ All searches are fully vectorized (no Python loops)
✅ All tests pass

---

## Phase 5: Field Interpolation on GPU

**Duration**: 1 week
**Dependencies**: Phase 4
**Goal**: GPU-accelerated field interpolation

### Objectives

1. Barycentric interpolation for tetrahedral elements
2. Support both node-based and element-based field storage
3. Fully vectorized with vmap
4. Handle boundary cases (particles on faces/edges)

### Tasks

#### 5.1: Barycentric Interpolation

```python
# File: jaxtrace/gpu/interpolation.py

@jax.jit
def interpolate_field_batch(
    particle_pos: jnp.ndarray,        # (N_particles, 3)
    particle_elem_ID: jnp.ndarray,    # (N_particles,)
    element_nodes: jnp.ndarray,       # (N_elements, 4)
    node_positions: jnp.ndarray,      # (N_nodes, 3)
    field_values: jnp.ndarray,        # (N_nodes, 3) or (N_elements, 4, 3)
    config: GPUConfig
) -> jnp.ndarray:
    """
    Interpolate field at particle positions.

    Args:
        particle_pos: Particle positions
        particle_elem_ID: Element containing each particle
        element_nodes: Element connectivity
        node_positions: Node coordinates
        field_values: Field data (configured format)
        config: GPU configuration

    Returns:
        interpolated_values: (N_particles, 3) field at particle positions
    """
    valid = particle_elem_ID >= 0
    safe_IDs = jnp.where(valid, particle_elem_ID, 0)

    # Get element node IDs
    elem_node_IDs = element_nodes[safe_IDs]  # (N_particles, 4)

    # Get node positions for each particle's element
    elem_node_pos = node_positions[elem_node_IDs]  # (N_particles, 4, 3)

    # Get field values
    if config.field_storage == "nodes":
        # Gather from nodes
        elem_field_vals = field_values[elem_node_IDs]  # (N_particles, 4, 3)
    else:  # elements
        # Direct access
        elem_field_vals = field_values[safe_IDs]  # (N_particles, 4, 3)

    # Compute barycentric coordinates
    bary_coords = compute_barycentric_coords_batch(
        particle_pos,
        elem_node_pos
    )  # (N_particles, 4)

    # Interpolate: sum(λᵢ × field[vᵢ])
    interpolated = jnp.einsum('ij,ijk->ik', bary_coords, elem_field_vals)

    # Mask invalid particles
    interpolated = jnp.where(valid[:, None], interpolated, 0.0)

    return interpolated


@jax.jit
def compute_barycentric_coords_batch(
    points: jnp.ndarray,      # (N, 3)
    vertices: jnp.ndarray     # (N, 4, 3)
) -> jnp.ndarray:
    """
    Compute barycentric coordinates.

    Returns:
        bary_coords: (N, 4) - λ₀, λ₁, λ₂, λ₃ for each point
    """
    v0, v1, v2, v3 = vertices[:, 0], vertices[:, 1], vertices[:, 2], vertices[:, 3]

    # Build matrix A = [v1-v0, v2-v0, v3-v0]
    A = jnp.stack([v1 - v0, v2 - v0, v3 - v0], axis=-1)  # (N, 3, 3)
    b = points - v0  # (N, 3)

    # Solve for λ₁, λ₂, λ₃
    lambdas = jnp.linalg.solve(A, b)  # (N, 3)

    # Compute λ₀
    lambda0 = 1.0 - jnp.sum(lambdas, axis=1, keepdims=True)  # (N, 1)

    # Concatenate
    bary_coords = jnp.concatenate([lambda0, lambdas], axis=1)  # (N, 4)

    return bary_coords
```

### Deliverables

- [ ] `jaxtrace/gpu/interpolation.py` - Interpolation kernels
- [ ] `tests/gpu/test_interpolation.py` - Interpolation tests
- [ ] Validation against analytical solutions
- [ ] Performance benchmarks

### Success Criteria

✅ Interpolation is fully vectorized
✅ Supports both node and element storage
✅ Accuracy matches CPU implementation (< 1e-6 error)
✅ Handles boundary cases (particles on faces)
✅ JIT-compiled
✅ All tests pass

---

## Phase 6: Time Marching Loop and RK4 Integration

**Duration**: 2 weeks
**Dependencies**: Phase 5
**Goal**: Implement time integration with minimal scan carry

### Objectives

1. RK4 time integration kernel
2. Time-marching loop with lax.scan
3. Minimal scan carry (positions, element_IDs, active)
4. Velocities interpolated per step (not stored)

### Time Integration

```python
# File: jaxtrace/gpu/integrator.py

@jax.jit
def rk4_step_batch(
    particle_pos: jnp.ndarray,
    particle_elem_ID: jnp.ndarray,
    dt: float,
    mesh_data: Dict,
    field_data: Dict,
    config: GPUConfig
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    RK4 time integration for all particles.

    Algorithm:
    1. k1 = f(t, y) - velocity at current position
    2. k2 = f(t + dt/2, y + dt/2 * k1) - velocity at midpoint
    3. k3 = f(t + dt/2, y + dt/2 * k2) - velocity at midpoint (alternate)
    4. k4 = f(t + dt, y + dt * k3) - velocity at endpoint
    5. y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Note: Velocities are interpolated at each stage, NOT stored.

    Returns:
        new_positions: (N_particles, 3)
        new_element_IDs: (N_particles,) - updated after each sub-step
    """
    # k1: Velocity at current position
    k1 = interpolate_field_batch(
        particle_pos,
        particle_elem_ID,
        mesh_data['element_nodes'],
        mesh_data['node_positions'],
        field_data['velocities'],
        config
    )

    # Midpoint 1: y + dt/2 * k1
    pos_mid1 = particle_pos + 0.5 * dt * k1
    elem_mid1 = multi_level_search(pos_mid1, particle_elem_ID, mesh_data, config)

    # k2: Velocity at midpoint 1
    k2 = interpolate_field_batch(
        pos_mid1,
        elem_mid1,
        mesh_data['element_nodes'],
        mesh_data['node_positions'],
        field_data['velocities'],
        config
    )

    # Midpoint 2: y + dt/2 * k2
    pos_mid2 = particle_pos + 0.5 * dt * k2
    elem_mid2 = multi_level_search(pos_mid2, elem_mid1, mesh_data, config)

    # k3: Velocity at midpoint 2
    k3 = interpolate_field_batch(
        pos_mid2,
        elem_mid2,
        mesh_data['element_nodes'],
        mesh_data['node_positions'],
        field_data['velocities'],
        config
    )

    # Endpoint: y + dt * k3
    pos_end = particle_pos + dt * k3
    elem_end = multi_level_search(pos_end, elem_mid2, mesh_data, config)

    # k4: Velocity at endpoint
    k4 = interpolate_field_batch(
        pos_end,
        elem_end,
        mesh_data['element_nodes'],
        mesh_data['node_positions'],
        field_data['velocities'],
        config
    )

    # Final position: y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    new_pos = particle_pos + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Final element search
    new_elem = multi_level_search(new_pos, elem_end, mesh_data, config)

    # Deactivate particles that left domain
    active = new_elem >= 0

    return new_pos, new_elem, active
```

### Time Loop with lax.scan

```python
# File: jaxtrace/gpu/time_loop.py

@jax.jit
def time_step_fn(particle_state, static_data):
    """
    Single time step (called by lax.scan).

    Args:
        particle_state: Dict with ONLY:
            - 'positions': (N_particles, 3) float64
            - 'element_IDs': (N_particles,) int32
            - 'active': (N_particles,) bool
        static_data: Tuple of (mesh_data, field_data, config, dt)

    Returns:
        new_particle_state: Updated particle state
        None: No history accumulated
    """
    mesh_data, field_data, config, dt = static_data

    # Unpack particle state (MINIMAL - only what's in carry)
    positions = particle_state['positions']
    element_IDs = particle_state['element_IDs']
    active = particle_state['active']

    # RK4 integration
    new_pos, new_elem, new_active = rk4_step_batch(
        positions,
        element_IDs,
        dt,
        mesh_data,
        field_data,
        config
    )

    # Return new state (MINIMAL)
    new_state = {
        'positions': new_pos,
        'element_IDs': new_elem,
        'active': new_active
    }

    return new_state, None  # No history in carry!


def run_simulation(
    initial_state: Dict,
    mesh_data: Dict,
    field_data: Dict,
    config: GPUConfig,
    dt: float,
    n_steps: int
) -> Dict:
    """
    Run full simulation with lax.scan.

    Args:
        initial_state: Dict with positions, element_IDs, active
        mesh_data: Static mesh arrays (NOT in carry)
        field_data: Static field arrays (NOT in carry)
        config: GPU configuration
        dt: Time step size
        n_steps: Number of steps

    Returns:
        final_state: Final particle state
    """
    static_data = (mesh_data, field_data, config, dt)

    # Run time loop
    final_state, _ = jax.lax.scan(
        time_step_fn,
        initial_state,
        static_data,
        length=n_steps
    )

    return final_state
```

### Deliverables

- [ ] `jaxtrace/gpu/integrator.py` - RK4 kernel
- [ ] `jaxtrace/gpu/time_loop.py` - Time loop with scan
- [ ] `tests/gpu/test_integrator.py` - Integration tests
- [ ] `tests/gpu/test_time_loop.py` - Time loop tests
- [ ] Validation against known trajectories
- [ ] Memory profiling of scan carry

### Success Criteria

✅ RK4 kernel is JIT-compiled
✅ Scan carry contains ONLY positions/element_IDs/active
✅ Velocities interpolated at each RK4 stage (not stored)
✅ Memory usage is constant: 29 bytes/particle
✅ Convergence tests pass (error ~ O(dt⁴))
✅ All tests pass

---

## Phase 7: Particle Block and Spatial Re-batching

**Duration**: 1 week
**Dependencies**: Phase 6
**Goal**: Spatial rebatching for cache locality

### Objectives

1. Batch particles by block after each step
2. Process blocks in parallel (vmap)
3. Unbatch particles after processing
4. Handle load imbalance (padding)

### Block Batching

```python
# File: jaxtrace/gpu/block_batching.py

@jax.jit
def batch_particles_by_block(
    particle_pos: jnp.ndarray,
    particle_elem_ID: jnp.ndarray,
    particle_active: jnp.ndarray,
    element_block_IDs: jnp.ndarray,
    n_blocks: int,
    max_particles_per_block: int
) -> Tuple[Dict, jnp.ndarray]:
    """
    Batch particles by their current block.

    Algorithm:
    1. Derive block ID from element ID (not stored!)
    2. Count particles per block
    3. Scatter particles into block-indexed arrays
    4. Pad to max_particles_per_block

    Returns:
        particles_by_block: Dict with keys:
            - 'positions': (N_blocks, max_per_block, 3)
            - 'element_IDs': (N_blocks, max_per_block)
            - 'active': (N_blocks, max_per_block)
        block_particle_counts: (N_blocks,) actual counts
    """
    # Derive block IDs (NOT stored in particle state!)
    valid_elem = particle_elem_ID >= 0
    safe_elem = jnp.where(valid_elem, particle_elem_ID, 0)
    particle_block_IDs = element_block_IDs[safe_elem]
    particle_block_IDs = jnp.where(valid_elem, particle_block_IDs, -1)

    # Count particles per block
    block_counts = jnp.zeros(n_blocks, dtype=jnp.int32)
    for block_id in range(n_blocks):
        block_counts = block_counts.at[block_id].set(
            jnp.sum((particle_block_IDs == block_id) & particle_active)
        )

    # Allocate block arrays
    n_particles = len(particle_pos)
    block_pos = jnp.zeros((n_blocks, max_particles_per_block, 3), dtype=particle_pos.dtype)
    block_elem = jnp.full((n_blocks, max_particles_per_block), -1, dtype=jnp.int32)
    block_active = jnp.zeros((n_blocks, max_particles_per_block), dtype=bool)

    # Scatter particles into blocks (using JAX scatter operations)
    for block_id in range(n_blocks):
        mask = (particle_block_IDs == block_id) & particle_active
        block_particle_indices = jnp.where(mask, size=max_particles_per_block, fill_value=-1)[0]

        valid_mask = block_particle_indices >= 0
        safe_indices = jnp.where(valid_mask, block_particle_indices, 0)

        block_pos = block_pos.at[block_id].set(
            jnp.where(valid_mask[:, None], particle_pos[safe_indices], 0.0)
        )
        block_elem = block_elem.at[block_id].set(
            jnp.where(valid_mask, particle_elem_ID[safe_indices], -1)
        )
        block_active = block_active.at[block_id].set(valid_mask)

    particles_by_block = {
        'positions': block_pos,
        'element_IDs': block_elem,
        'active': block_active
    }

    return particles_by_block, block_counts


@jax.jit
def unbatch_particles(
    particles_by_block: Dict,
    block_counts: jnp.ndarray,
    n_particles: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Unbatch particles from block structure to flat arrays.

    Returns:
        particle_pos: (N_particles, 3)
        particle_elem_ID: (N_particles,)
        particle_active: (N_particles,)
    """
    n_blocks, max_per_block, _ = particles_by_block['positions'].shape

    # Flatten
    flat_pos = particles_by_block['positions'].reshape(-1, 3)
    flat_elem = particles_by_block['element_IDs'].reshape(-1)
    flat_active = particles_by_block['active'].reshape(-1)

    # Take first n_particles (rest is padding)
    particle_pos = flat_pos[:n_particles]
    particle_elem = flat_elem[:n_particles]
    particle_active = flat_active[:n_particles]

    return particle_pos, particle_elem, particle_active
```

### Deliverables

- [ ] `jaxtrace/gpu/block_batching.py` - Batching utilities
- [ ] `tests/gpu/test_block_batching.py` - Batching tests
- [ ] Performance analysis of batching overhead

### Success Criteria

✅ Particles correctly batched by block
✅ Block IDs derived (not stored)
✅ Padding handled correctly
✅ Unbatching recovers original arrays
✅ All tests pass

---

## Phase 8: Ghost/Halo Region Support

**Duration**: 1 week
**Dependencies**: Phase 7
**Goal**: Handle block boundaries robustly

### Objectives

1. Identify block boundaries
2. Handle particles crossing blocks
3. Support ghost/halo cells for field interpolation

(Details omitted for brevity - this phase is less critical for initial implementation)

---

## Phase 9: Hash Octree Integration and Optimization

**Duration**: 2 weeks
**Dependencies**: Phase 4
**Goal**: O(1) element search with hash table

### Objectives

1. Build per-block hash table (Morton key → element ID)
2. Integrate hash search as Level 2 alternative
3. Benchmark hash vs linear search

(Details omitted for brevity - this is a performance optimization)

---

## Phase 10: Full Pipeline Integration and Performance Benchmarking

**Duration**: 1 week
**Dependencies**: All previous phases
**Goal**: End-to-end validation and performance analysis

### Objectives

1. Run full ThreadedA simulation (1M particles, 1000 steps)
2. Benchmark performance (particles/s, speedup vs CPU)
3. Memory profiling (verify no leaks)
4. Export results (VTK, HDF5)
5. Documentation and examples

### Final Benchmarks

**Expected Performance (1M particles, 3.5M elements):**

| Metric | Target | Acceptable |
|--------|--------|------------|
| GPU memory | <500 MB | <1 GB |
| Time per step | 0.1-0.2s | <0.5s |
| Speedup vs CPU | 10-100× | >5× |
| Hit rates | L0: 85-95%, L1: 3-10%, L2: 1-5% | L0: >80% |
| Accuracy | <1e-6 vs CPU | <1e-4 |

### Deliverables

- [ ] Full simulation script
- [ ] Performance report
- [ ] Memory analysis
- [ ] User documentation
- [ ] Example notebooks
- [ ] CI/CD integration

### Success Criteria

✅ 1M particles × 1000 steps completes successfully
✅ GPU memory < 1 GB
✅ Speedup > 5× vs CPU
✅ No memory leaks
✅ Results match CPU (< 1e-4 error)
✅ All tests pass

---

## Project Timeline

| Phase | Duration | Dependencies | Deliverables |
|-------|----------|--------------|--------------|
| 0 | 1 week | None | Mesh analysis, test infra |
| 1 | 1 week | 0 | Flat arrays, mesh loading |
| 2 | 1.5 weeks | 1 | Morton codes, block assignment |
| 3 | 1 week | 1, 2 | Particle seeding |
| 4 | 2 weeks | 3 | Multi-level search |
| 5 | 1 week | 4 | Field interpolation |
| 6 | 2 weeks | 5 | RK4, time loop |
| 7 | 1 week | 6 | Block batching |
| 8 | 1 week | 7 | Ghost/halo (optional) |
| 9 | 2 weeks | 4 | Hash octree (optional) |
| 10 | 1 week | All | Integration, benchmarks |

**Total**: 12-14 weeks (3-3.5 months)

---

## Memory Safety Guarantees

### Scan Carry (Minimal)

```python
# ONLY these arrays in scan carry:
positions: 1M × 3 × 8 = 24 MB
element_IDs: 1M × 4 = 4 MB
active: 1M × 1 = 1 MB
# TOTAL: 29 MB
```

### Static Data (NOT in Carry)

```python
# All mesh/field data passed as constants:
node_positions: 900K × 3 × 4 = 10.8 MB
element_nodes: 3.5M × 4 × 4 = 56 MB
element_neighbors: 3.5M × 4 × 4 = 56 MB
element_block_IDs: 3.5M × 4 = 14 MB
velocities: 900K × 3 × 4 = 10.8 MB
block_elements: 32 × 10K × 4 = 1.3 MB
octree (estimate): 10 MB
# TOTAL: ~160 MB
```

### Grand Total

**Total GPU memory: 29 MB (carry) + 160 MB (static) = ~190 MB**

✅ Fits comfortably in 8 GB GPU

---

## Conclusion

This comprehensive V3 plan provides:

1. ✅ **Incremental development** - each phase is independently testable
2. ✅ **Minimal scan carry** - 29 MB for 1M particles
3. ✅ **Static mesh/field data** - never in scan carry
4. ✅ **Configurable storage** - users can tune for their mesh
5. ✅ **JAX-optimal design** - flat arrays, vmap, JIT everywhere
6. ✅ **Memory safety** - guaranteed no explosion
7. ✅ **Clear milestones** - 10 phases with success criteria
8. ✅ **Comprehensive testing** - unit, integration, performance
9. ✅ **Production-ready** - handles 1M particles, 3.5M elements
10. ✅ **Well-documented** - every function, every choice explained

**Expected outcome**: 10-100× speedup over CPU with <200 MB GPU memory.

**Next step**: Begin Phase 0 - mesh analysis and infrastructure setup.
