# JAXTrace Code Structure Cheat Sheet

**Last Updated**: 2025-11-05
**Purpose**: Quick reference for navigating JAXTrace codebase

---

## 📁 Directory Structure

```
JAXTrace/
├── jaxtrace/
│   ├── io/                    # I/O modules
│   ├── mesh/                  # Mesh data structures
│   ├── fields/                # Field interpolation
│   ├── gpu/                   # GPU acceleration (V5)
│   ├── integrators/           # Time integration
│   ├── tracers/               # Particle tracers
│   └── utils/                 # Utilities
├── docs/                      # Documentation
├── tests/                     # Test suite
├── utils/                     # Testing utilities
└── logs/                      # Test logs
```

---

## 🔧 Key Modules

### I/O System (`jaxtrace/io/`)

**Main Entry**: `jaxtrace.io`
- `open_dataset()` - Universal dataset opener (auto-detect format)
- `VTKUnstructuredTimeSeriesReader` - Unstructured VTK time series
- `VTKStructuredSeries` - Structured VTK grids
- `open_vtk_time_series()` - Direct VTK time series loader

**VTK Reader**: `jaxtrace/io/vtk_reader.py`
```python
from jaxtrace.io import VTKUnstructuredTimeSeriesReader

# Read unstructured VTK time series
reader = VTKUnstructuredTimeSeriesReader(directory="path/to/vtk")
timesteps = reader.get_timesteps()
nodes = reader.read_nodes(timesteps[0])
connectivity = reader.read_connectivity(timesteps[0])
field_data = reader.read_point_data(timesteps[0], "velocities")
```

**VTK Writer**: `jaxtrace/io/vtk_writer.py`
```python
from jaxtrace.io import VTKTrajectoryWriter

writer = VTKTrajectoryWriter()
writer.write_trajectory(trajectory, "output.vtp")
writer.write_time_series(trajectory, "output_dir/")
```

**Supported Formats**:
- `.pvtu` - Parallel VTK Unstructured Grid (ThreadedA mesh format)
- `.vtu` - VTK Unstructured Grid
- `.vts` - VTK Structured Grid
- `.vtr` - VTK Rectilinear Grid
- `.h5` - HDF5 (if h5py available)

---

### Mesh System (`jaxtrace/mesh/`)

**UnstructuredMesh**: `jaxtrace/mesh/unstructured.py`
```python
from jaxtrace.mesh.unstructured import UnstructuredMesh

mesh = UnstructuredMesh(nodes, connectivity)
# nodes: (N_nodes, 3) float32
# connectivity: (N_elements, nodes_per_elem) int32
```

---

### GPU Acceleration (`jaxtrace/gpu/`)

#### V5 Block-Local Search (NEW)

**Block Element Arrays**: `jaxtrace/gpu/forest/block_elements.py`
```python
from jaxtrace.gpu.forest.block_elements import build_padded_block_arrays

block_arrays = build_padded_block_arrays(
    octrees, element_to_block, blocks, verbose=True
)
# Returns: BlockElementArrays with padded 2D arrays
```

**Multi-Level Search**: `jaxtrace/gpu/block_local_search_jax.py`
```python
from jaxtrace.gpu.block_local_search_jax import find_elements_batch_multi_level_jax

element_IDs = find_elements_batch_multi_level_jax(
    particle_positions_jax,
    cached_elem_ids_jax,
    particle_block_ids_jax,
    mesh_data_jax,
    block_data_jax
)
# Full 4-level hierarchy: cached → neighbors → block → neighbor blocks
```

**Initial Search (V5 Integration)**: `jaxtrace/gpu/initial_search_jax.py`
```python
from jaxtrace.gpu.initial_search_jax import find_initial_elements_batch, GPUConfig

config = GPUConfig(
    use_block_local_search=True,  # Enable V5
    use_gpu_multi_level=True       # Enable 4-level hierarchy
)

element_IDs, stats = find_initial_elements_batch(
    particle_positions,
    mesh_data,
    partition_data,
    octrees,
    blocks=blocks,                  # Required for V5
    element_to_block=element_to_block,  # Required for V5
    element_neighbors=element_neighbors, # For multi-level
    config=config
)
```

#### Block Infrastructure

**Block Builder**: `jaxtrace/gpu/forest/block_builder.py`
```python
from jaxtrace.gpu.forest.block_builder import (
    create_regular_forest_grid,
    position_to_block_id
)

blocks = create_regular_forest_grid(domain_bounds, grid_size=(4, 4, 2))
block_id = position_to_block_id(position, domain_bounds, grid_size)
```

**Block Mapper**: `jaxtrace/gpu/forest/block_mapper.py`
```python
from jaxtrace.gpu.forest.block_mapper import assign_elements_to_blocks

element_to_block = assign_elements_to_blocks(
    positions, connectivity, blocks, domain_bounds, grid_size
)
```

**Element Neighbors**: `jaxtrace/gpu/forest/element_neighbors.py`
```python
from jaxtrace.gpu.forest.element_neighbors import build_element_adjacency

element_neighbors = build_element_adjacency(
    connectivity, max_neighbors=32
)
# Returns: (N_elements, max_neighbors) with face-adjacency
```

---

### Fields (`jaxtrace/fields/`)

**Octree Builders**:
- `coarse_octree_builder.py` - Build coarse octrees per block
- `fine_octree_builder.py` - Build fine octrees
- `direct_octree_interpolator_jax.py` - JAX interpolation

**FEM Interpolators**:
- `fem_interpolator.py` - Basic FEM interpolation
- `octree_fem_interpolator.py` - Octree-accelerated FEM
- `direct_octree_fem_interpolator.py` - Direct octree FEM

---

## 🧪 Testing Utilities

### Resource Monitoring (`utils/resource_monitor.py`)

```python
from utils.resource_monitor import ResourceMonitor

monitor = ResourceMonitor()

with monitor.stage("Load Mesh"):
    mesh = load_mesh(...)

with monitor.stage("GPU Search"):
    element_IDs = search(...)

monitor.print_summary()
monitor.save_log("results.json")
```

**Tracks per stage**:
- GPU memory (allocated, reserved, peak)
- GPU utilization (via pynvml)
- CPU memory (RSS, VMS)
- CPU utilization
- Timing and deltas

---

## 📊 Common Workflows

### 1. Load ThreadedA Mesh

```python
from jaxtrace.io import VTKUnstructuredTimeSeriesReader
from jaxtrace.mesh.unstructured import UnstructuredMesh

# ThreadedA mesh location
mesh_dir = "../Edgar/ThreadedA/post/0eule"

# Read VTK files
reader = VTKUnstructuredTimeSeriesReader(mesh_dir)
timesteps = reader.get_timesteps()
t0 = timesteps[0]

# Load mesh data
nodes = reader.read_nodes(t0)
connectivity = reader.read_connectivity(t0)
velocities = reader.read_point_data(t0, "velocities")

# Create mesh
mesh = UnstructuredMesh(nodes, connectivity)
```

**File Pattern**: `threadedAvtk_*.pvtu`

### 2. Build V5 Block Infrastructure

```python
from jaxtrace.gpu.forest.block_builder import create_regular_forest_grid
from jaxtrace.gpu.forest.block_mapper import assign_elements_to_blocks
from jaxtrace.gpu.forest.element_neighbors import build_element_adjacency
from jaxtrace.fields.coarse_octree_builder import build_octree_for_block
from jaxtrace.gpu.forest.block_elements import build_padded_block_arrays

# 1. Create blocks
grid_size = (4, 4, 2)  # 32 blocks
bbox = np.array([xmin, xmax, ymin, ymax, zmin, zmax])
blocks = create_regular_forest_grid(bbox, grid_size)

# 2. Assign elements to blocks
element_to_block = assign_elements_to_blocks(
    nodes, connectivity, blocks, bbox, grid_size
)

# 3. Build octrees per block
octrees = {}
for block in blocks:
    elem_ids = np.where(element_to_block == block.block_id)[0]
    if len(elem_ids) > 0:
        octrees[block.block_id] = build_octree_for_block(
            block.block_id, elem_ids, nodes, connectivity,
            block.bounds, max_depth=3
        )

# 4. Build element neighbors
element_neighbors = build_element_adjacency(connectivity, max_neighbors=32)

# 5. Build V5 padded arrays
block_arrays = build_padded_block_arrays(
    octrees, element_to_block, blocks, verbose=True
)
```

### 3. Run V5 GPU Search

```python
from jaxtrace.gpu.initial_search_jax import find_initial_elements_batch, GPUConfig

# Configure V5
config = GPUConfig(
    use_block_local_search=True,
    use_gpu_multi_level=True,
    validate_block_arrays=True
)

# Prepare data
mesh_data = {
    'positions': nodes,
    'connectivity': connectivity
}

partition_data = {
    'bbox_global': bbox,
    'grid_size': grid_size
}

# Run search
element_IDs, stats = find_initial_elements_batch(
    particle_positions,
    mesh_data,
    partition_data,
    octrees,
    blocks=blocks,
    element_to_block=element_to_block,
    element_neighbors=element_neighbors,
    config=config,
    verbose=True
)

print(f"Found: {stats['n_found']}/{stats['n_particles']}")
print(f"Used V5: {stats['used_v5']}")
```

---

## 🎯 Key Data Structures

### Mesh Data
```python
mesh_data = {
    'positions': np.ndarray,     # (N_nodes, 3) float32
    'connectivity': np.ndarray,  # (N_elements, 4) int32 for tets
    'bbox': np.ndarray           # [xmin, xmax, ymin, ymax, zmin, zmax]
}
```

### Block Metadata
```python
@dataclass
class BlockMetadata:
    block_id: int
    bounds: np.ndarray          # [xmin, xmax, ymin, ymax, zmin, zmax]
    center: np.ndarray          # [x, y, z]
    grid_index: Tuple[int, int, int]
    neighbors: np.ndarray       # [6] face neighbors, -1 for boundary
```

### Block Element Arrays (V5)
```python
@dataclass
class BlockElementArrays:
    block_elements: np.ndarray       # (n_blocks, max_elem), -1 padded
    block_elem_counts: np.ndarray    # (n_blocks,) actual counts
    block_neighbors_26: np.ndarray   # (n_blocks, 26) neighbor IDs
    max_elem_per_block: int
    n_blocks: int
    total_elements: int
```

### Octree Data
```python
@dataclass
class OctreeData:
    block_id: int
    sorted_element_IDs: np.ndarray
    bounds: np.ndarray
    # ... other octree fields
```

---

## 🚀 Performance Tips

1. **Always use V5 for GPU search** (block-local, not global)
2. **Build element neighbors once** (cache for reuse)
3. **Validate block arrays** in development, disable in production
4. **Use resource monitoring** to track memory and performance
5. **Batch particles** in groups of 1K-10K for optimal GPU usage

---

## 📍 Mesh File Locations

### ThreadedA
- **Path**: `../Edgar/ThreadedA/post/0eule/`
- **Pattern**: `threadedAvtk_*.pvtu`
- **Size**: 3.5M elements, 900K nodes
- **Type**: Unstructured tetrahedral mesh

### FLA
- **Path**: `../Edgar/FLA/post/0eule/`
- **Pattern**: `fla_vtk_*.pvtu`
- **Size**: 2.1M elements, 600K nodes
- **Type**: Unstructured tetrahedral mesh

---

## 🐛 Common Issues

### 1. ModuleNotFoundError for VTK
```python
# Use correct import
from jaxtrace.io import VTKUnstructuredTimeSeriesReader  # ✅
from jaxtrace.io.vtk_io import VTKSeries  # ❌ Wrong
```

### 2. JAX Dictionary Indexing Error
```python
# V4 WRONG:
octrees[block_id]  # ❌ Can't index dict with traced value

# V5 CORRECT:
block_elements[block_id]  # ✅ Array indexing works
```

### 3. Memory Explosion
```python
# V4 global search: 45 GB ❌
# V5 block-local search: <200 MB ✅

# Always use:
config = GPUConfig(use_block_local_search=True)
```

---

## 📚 Key Documents

- **V5 Implementation**: `docs/gpu/V5_IMPLEMENTATION_COMPLETE.md`
- **V5 Plan**: `docs/gpu/GPU_IMPLEMENTATION_PLAN_V5_CORRECTED_COMPREHENSIVE.md`
- **Summary**: `V5_IMPLEMENTATION_SUMMARY.md`
- **This Cheatsheet**: `CODE_STRUCTURE_CHEATSHEET.md`

---

**Quick Start**: See section "Common Workflows" above for copy-paste ready code.
