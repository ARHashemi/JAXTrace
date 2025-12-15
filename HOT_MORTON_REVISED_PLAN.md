# HOT Morton L2 Search - REVISED Implementation Plan

**Date**: 2025-12-12
**Status**: ✅ PLAN REVISED - Ready for Implementation

---

## Executive Summary

Implementing a **global HOT-like Morton L2 search** to replace the incorrect block-based approach. This design uses a single Morton-sorted element list divided into fixed-capacity segments (leaves), with simple offset-based lookup on GPU.

**Key Correction**: The original implementation wrongly used 256 cube-aligned blocks with per-block Morton sorting. The correct design uses **NO blocks** - just one global Morton curve with leaf segments.

---

## What Was Wrong (Original Implementation)

**Incorrect Architecture**:
```python
# ❌ WRONG - Used 256 blocks
def build_cube_aligned_blocks(grid_size=(8, 8, 4)):  # 256 blocks
    # Per-block Morton sorting
    # Per-block octree leaves
    # Block-based data structures
```

**Problems**:
1. Unnecessary block complexity
2. Per-block processing overhead
3. Block size limits (max 50k elements) causing errors
4. Doesn't match HOT design philosophy

---

## Correct HOT Morton Design

**Simple Global Architecture**:
```python
# ✅ CORRECT - Single global Morton list
morton_codes = compute_morton_for_centroids(connectivity, node_positions)
sorted_indices = np.argsort(morton_codes)
elem_ids_sorted = np.arange(n_elements)[sorted_indices]

# Divide into fixed-size segments (leaves)
# Leaf 0: elements [0, C)
# Leaf 1: elements [C, 2C)
# ...
# Leaf N: elements [N*C, min(N*C+C, n_elements))

leaf_start[i] = i * C
leaf_length[i] = min(C, n_elements - i * C)
```

**GPU Search**:
1. Compute Morton code for particle position
2. Map Morton code → leaf ID (via hash/prefix table)
3. Get leaf offset: `start = leaf_start[leaf_id]`
4. Search `elem_ids_sorted[start:start+C]` using bounded loop
5. No dynamic slicing - use `lax.fori_loop` with fixed bound C

---

## Phase 1: Global Morton Structure (CPU Preprocessing)

**File**: `jaxtrace/gpu/search/morton_global_builder.py` (NEW)

### 1.1 Compute Morton Codes

```python
def interleave_bits_3d(x: np.uint32, y: np.uint32, z: np.uint32) -> np.uint64:
    """
    Interleave 3D coordinates into Morton (Z-order) code.

    Morton code: interleave bits of (x,y,z) coordinates
    Example: x=101b, y=110b, z=011b → morton=zyxzyxzyx=011110011b
    """
    morton = np.uint64(0)
    for i in range(21):  # Up to 21 bits per dimension (63 total)
        morton |= ((x >> i) & 1) << (3*i + 0)
        morton |= ((y >> i) & 1) << (3*i + 1)
        morton |= ((z >> i) & 1) << (3*i + 2)
    return morton

def compute_morton_codes_for_elements(
    node_positions: np.ndarray,  # (n_nodes, 3)
    connectivity: np.ndarray,     # (n_elements, 4)
    bbox_min: np.ndarray,         # (3,)
    bbox_max: np.ndarray,         # (3,)
    max_depth: int = 21           # Bits per dimension
) -> np.ndarray:
    """
    Compute Morton code for each element centroid.

    Returns:
        morton_codes: (n_elements,) uint64
    """
    n_elements = connectivity.shape[0]
    morton_codes = np.empty(n_elements, dtype=np.uint64)

    # Scaling factor for coordinate → integer grid
    scale = (2**max_depth - 1) / (bbox_max - bbox_min)

    for e in range(n_elements):
        # Compute centroid
        nodes = connectivity[e]
        centroid = node_positions[nodes].mean(axis=0)

        # Normalize to [0, 2^max_depth - 1]
        normalized = (centroid - bbox_min) * scale
        ux = np.uint32(np.floor(normalized[0]))
        uy = np.uint32(np.floor(normalized[1]))
        uz = np.uint32(np.floor(normalized[2]))

        # Interleave bits
        morton_codes[e] = interleave_bits_3d(ux, uy, uz)

    return morton_codes
```

### 1.2 Global Sort

```python
def build_global_morton_sorted_list(
    morton_codes: np.ndarray  # (n_elements,) uint64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort elements by Morton code.

    Returns:
        elem_ids_sorted: (n_elements,) int32 - element IDs in Morton order
        morton_sorted: (n_elements,) uint64 - sorted Morton codes
    """
    sorted_indices = np.argsort(morton_codes)
    elem_ids_sorted = np.arange(len(morton_codes), dtype=np.int32)[sorted_indices]
    morton_sorted = morton_codes[sorted_indices]

    return elem_ids_sorted, morton_sorted
```

### 1.3 Leaf Segmentation (Simple Phase 1 Version)

```python
def build_fixed_capacity_leaves(
    elem_ids_sorted: np.ndarray,  # (n_elements,) int32
    leaf_capacity: int = 256       # Fixed max elements per leaf
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divide sorted element list into fixed-capacity segments.

    Phase 1: Simple equal-size segments
    - Leaf 0: [0, C)
    - Leaf 1: [C, 2C)
    - ...

    Later phases can use true octree-aligned leaves.

    Returns:
        leaf_start: (n_leaves,) int32 - start index of each leaf
        leaf_length: (n_leaves,) int32 - element count per leaf
    """
    n_elements = len(elem_ids_sorted)
    n_leaves = (n_elements + leaf_capacity - 1) // leaf_capacity

    leaf_start = np.arange(n_leaves, dtype=np.int32) * leaf_capacity
    leaf_length = np.minimum(
        np.full(n_leaves, leaf_capacity, dtype=np.int32),
        n_elements - leaf_start
    )

    return leaf_start, leaf_length
```

### 1.4 Position → Leaf Mapping (Linear Approximation)

```python
def build_morton_leaf_mapping(
    morton_sorted: np.ndarray,  # (n_elements,) uint64
    n_leaves: int
) -> Tuple[np.uint64, np.uint64]:
    """
    Compute Morton range for linear leaf mapping.

    Phase 1: Linear approximation
    leaf_id ≈ (morton - morton_min) / (morton_max - morton_min) * n_leaves

    Returns:
        morton_min: uint64
        morton_max: uint64
    """
    return morton_sorted[0], morton_sorted[-1]
```

### 1.5 Complete Preprocessing Pipeline

```python
@dataclass
class GlobalMortonStructure:
    """Global HOT Morton structure - NO blocks"""

    # Sorted element list
    elem_ids_sorted: np.ndarray      # (n_elements,) int32
    morton_sorted: np.ndarray        # (n_elements,) uint64 - for debugging

    # Leaf segments
    leaf_start: np.ndarray           # (n_leaves,) int32
    leaf_length: np.ndarray          # (n_leaves,) int32
    n_leaves: int

    # Morton mapping
    morton_min: np.uint64
    morton_max: np.uint64

    # Mesh bounds
    bbox_min: np.ndarray             # (3,) float32
    bbox_max: np.ndarray             # (3,) float32
    max_depth: int                   # Morton depth (bits per dim)
    leaf_capacity: int               # Max elements per leaf

def build_global_morton_structure(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    leaf_capacity: int = 256,
    max_depth: int = 21,
    verbose: bool = True
) -> GlobalMortonStructure:
    """
    Build complete global HOT Morton structure.

    Steps:
    1. Compute Morton codes for element centroids
    2. Sort elements by Morton code
    3. Divide into fixed-capacity leaves
    4. Compute Morton range for mapping

    Returns:
        GlobalMortonStructure with all arrays ready for GPU upload
    """
    if verbose:
        logger = logging.getLogger(__name__)
        logger.info("Building Global HOT Morton Structure")
        logger.info(f"  Elements: {connectivity.shape[0]:,}")
        logger.info(f"  Leaf capacity: {leaf_capacity}")
        logger.info(f"  Max depth: {max_depth}")

    # Step 1: Compute bounding box
    bbox_min = node_positions.min(axis=0).astype(np.float32)
    bbox_max = node_positions.max(axis=0).astype(np.float32)

    # Step 2: Compute Morton codes
    morton_codes = compute_morton_codes_for_elements(
        node_positions, connectivity, bbox_min, bbox_max, max_depth
    )

    # Step 3: Global sort
    elem_ids_sorted, morton_sorted = build_global_morton_sorted_list(morton_codes)

    # Step 4: Leaf segmentation
    leaf_start, leaf_length = build_fixed_capacity_leaves(
        elem_ids_sorted, leaf_capacity
    )
    n_leaves = len(leaf_start)

    # Step 5: Morton mapping
    morton_min, morton_max = build_morton_leaf_mapping(morton_sorted, n_leaves)

    if verbose:
        logger.info(f"  Number of leaves: {n_leaves:,}")
        logger.info(f"  Morton range: [{morton_min}, {morton_max}]")
        avg_per_leaf = len(elem_ids_sorted) / n_leaves
        logger.info(f"  Avg elements per leaf: {avg_per_leaf:.1f}")

    return GlobalMortonStructure(
        elem_ids_sorted=elem_ids_sorted,
        morton_sorted=morton_sorted,
        leaf_start=leaf_start,
        leaf_length=leaf_length,
        n_leaves=n_leaves,
        morton_min=morton_min,
        morton_max=morton_max,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        max_depth=max_depth,
        leaf_capacity=leaf_capacity
    )
```

---

## Phase 2: GPU Search Kernel

**File**: `jaxtrace/gpu/search/morton_global_search.py` (NEW)

### 2.1 Morton Encoding (JAX)

```python
import jax
import jax.numpy as jnp
from jax import lax

def interleave_bits_3d_jax(x: jnp.uint32, y: jnp.uint32, z: jnp.uint32) -> jnp.uint64:
    """JAX version of bit interleaving for Morton code."""
    morton = jnp.uint64(0)
    for i in range(21):
        morton |= ((x >> i) & 1) << (3*i + 0)
        morton |= ((y >> i) & 1) << (3*i + 1)
        morton |= ((z >> i) & 1) << (3*i + 2)
    return morton

def morton_encode_position_jax(
    pos: jnp.ndarray,      # (3,) float32
    bbox_min: jnp.ndarray, # (3,) float32
    bbox_max: jnp.ndarray, # (3,) float32
    max_depth: int
) -> jnp.uint64:
    """
    Compute Morton code for position on GPU.

    Args:
        pos: 3D position
        bbox_min, bbox_max: domain bounds
        max_depth: bits per dimension

    Returns:
        Morton code (uint64)
    """
    scale = (2**max_depth - 1) / (bbox_max - bbox_min)
    u = jnp.floor((pos - bbox_min) * scale).astype(jnp.uint32)
    return interleave_bits_3d_jax(u[0], u[1], u[2])
```

### 2.2 Position → Leaf Mapping

```python
def position_to_leaf_id_linear(
    pos: jnp.ndarray,
    mesh_morton_gpu
) -> jnp.int32:
    """
    Map position to leaf ID using linear Morton approximation.

    Phase 1: Simple linear mapping along Morton curve
    leaf_id ≈ (morton - morton_min) / (morton_max - morton_min) * n_leaves

    Later phases can use prefix table for exact geometric mapping.
    """
    # Compute Morton code
    m = morton_encode_position_jax(
        pos,
        mesh_morton_gpu.bbox_min,
        mesh_morton_gpu.bbox_max,
        mesh_morton_gpu.max_depth
    )

    # Linear approximation
    m_min = mesh_morton_gpu.morton_min
    m_max = mesh_morton_gpu.morton_max
    t = (m - m_min).astype(jnp.float32) / (m_max - m_min + 1).astype(jnp.float32)

    # Map to leaf index
    leaf_id_approx = jnp.floor(t * mesh_morton_gpu.n_leaves).astype(jnp.int32)

    # Clamp to valid range
    return jnp.clip(leaf_id_approx, 0, mesh_morton_gpu.n_leaves - 1)
```

### 2.3 Search Within Leaf (Bounded Loop)

```python
def search_in_leaf_global(
    pos: jnp.ndarray,           # (3,) float32
    leaf_id: jnp.int32,
    mesh_morton_gpu,
    mesh_gpu                    # For connectivity and node_positions
) -> jnp.int32:
    """
    Search for position within a single Morton leaf.

    Uses bounded lax.fori_loop over leaf_capacity elements.
    Early exit when element found.

    Returns:
        element_id (int32) or -1 if not found
    """
    start = mesh_morton_gpu.leaf_start[leaf_id]
    length = mesh_morton_gpu.leaf_length[leaf_id]
    max_capacity = mesh_morton_gpu.leaf_capacity

    def body(j, found_elem):
        # Early exit if already found
        active = (found_elem == -1) & (j < length)

        # Get global element ID from sorted list
        idx = start + j
        elem_id = jnp.where(
            active,
            mesh_morton_gpu.elem_ids_sorted[idx],
            jnp.int32(0)  # Safe dummy value
        )

        # Point-in-tet test (reuse existing function)
        inside = jnp.where(
            active,
            point_in_tet_jax(pos, elem_id, mesh_gpu.connectivity, mesh_gpu.node_positions),
            False
        )

        # Update found element
        return jnp.where(inside & active, elem_id, found_elem)

    # Bounded loop - static shape for JAX
    init_elem = jnp.int32(-1)
    found_elem = lax.fori_loop(0, max_capacity, body, init_elem)

    return found_elem
```

### 2.4 Complete L2 Search (Single Particle)

```python
def search_L2_global_morton_single(
    pos: jnp.ndarray,
    mesh_morton_gpu,
    mesh_gpu
) -> jnp.int32:
    """
    L2 search: Global Morton leaf lookup.

    Steps:
    1. Compute Morton code for position
    2. Map to leaf ID
    3. Search bounded segment

    Returns:
        element_id or -1
    """
    # Map position to leaf
    leaf_id = position_to_leaf_id_linear(pos, mesh_morton_gpu)

    # Handle invalid leaf (outside domain)
    valid_leaf = (leaf_id >= 0) & (leaf_id < mesh_morton_gpu.n_leaves)

    # Search within leaf
    elem_id = jnp.where(
        valid_leaf,
        search_in_leaf_global(pos, leaf_id, mesh_morton_gpu, mesh_gpu),
        jnp.int32(-1)
    )

    return elem_id
```

### 2.5 Upload to GPU

```python
@dataclass
class MeshGPUGlobalMorton:
    """GPU-resident global Morton structure - NO blocks"""

    # Sorted element list
    elem_ids_sorted: jax.Array     # (n_elements,) int32

    # Leaf segments
    leaf_start: jax.Array          # (n_leaves,) int32
    leaf_length: jax.Array         # (n_leaves,) int32
    n_leaves: int
    leaf_capacity: int

    # Morton mapping
    morton_min: jnp.uint64
    morton_max: jnp.uint64

    # Domain bounds
    bbox_min: jax.Array            # (3,) float32
    bbox_max: jax.Array            # (3,) float32
    max_depth: int

def upload_global_morton_to_gpu(
    morton_struct: GlobalMortonStructure
) -> MeshGPUGlobalMorton:
    """Upload global Morton structure to GPU."""
    return MeshGPUGlobalMorton(
        elem_ids_sorted=jax.device_put(morton_struct.elem_ids_sorted),
        leaf_start=jax.device_put(morton_struct.leaf_start),
        leaf_length=jax.device_put(morton_struct.leaf_length),
        n_leaves=morton_struct.n_leaves,
        leaf_capacity=morton_struct.leaf_capacity,
        morton_min=jnp.uint64(morton_struct.morton_min),
        morton_max=jnp.uint64(morton_struct.morton_max),
        bbox_min=jax.device_put(morton_struct.bbox_min),
        bbox_max=jax.device_put(morton_struct.bbox_max),
        max_depth=morton_struct.max_depth
    )
```

---

## Phase 3: Integration into Fused RK4

**File**: `jaxtrace/gpu/tracking/rk4_gpu_fused.py` (MODIFY)

### 3.1 Multi-Level Search Function (Keep Current Architecture)

**IMPORTANT**: The current fused RK4 uses vectorized L0 and L1, then vmap for L2. We keep this pattern.

```python
@jax.jit
def search_l0_l1_l2_global_morton(
    positions_gpu: jax.Array,             # (N, 3)
    cached_element_ids_gpu: jax.Array,    # (N,)
    mesh_gpu_node_positions: jax.Array,
    mesh_gpu_connectivity: jax.Array,
    mesh_gpu_element_neighbors: jax.Array,
    mesh_morton_gpu,
    n_hops: int = 3
) -> jax.Array:
    """
    L0 + L1 + L2 Global Morton search.

    Architecture matches current fused RK4:
    - L0: Vectorized cached search
    - L1: Vectorized multi-hop neighbor search
    - L2: Single-particle function, vmapped

    Args:
        positions_gpu: (N, 3) float32
        cached_element_ids_gpu: (N,) int32
        mesh_gpu_*: standard mesh arrays
        mesh_morton_gpu: global Morton structure

    Returns:
        element_ids: (N,) int32
    """
    # ========== L0: Cached Element (KEEP - Vectorized) ==========
    element_ids_l0 = search_level0_vectorized(
        positions_gpu,
        cached_element_ids_gpu,
        mesh_gpu_node_positions,
        mesh_gpu_connectivity
    )

    # ========== L1: Multi-hop Neighbors (KEEP - Vectorized) ==========
    element_ids_l1 = search_level1_multihop_vectorized(
        positions_gpu,
        element_ids_l0,
        mesh_gpu_node_positions,
        mesh_gpu_connectivity,
        mesh_gpu_element_neighbors,
        n_hops=n_hops
    )

    # ========== L2: Global Morton Leaf (NEW - Single particle, vmapped) ==========
    def search_l2_single(pos, elem_id_l1):
        """
        L2 search for single particle.
        Only runs if L1 failed (elem_id_l1 < 0).
        """
        need_l2 = elem_id_l1 < 0

        # Compute element ID using global Morton
        elem_id_l2 = jnp.where(
            need_l2,
            search_L2_global_morton_single(pos, mesh_morton_gpu, mesh_gpu_connectivity, mesh_gpu_node_positions),
            elem_id_l1  # Keep L1 result if found
        )

        return elem_id_l2

    # Vmap L2 search over all particles
    element_ids_final = jax.vmap(search_l2_single)(positions_gpu, element_ids_l1)

    return element_ids_final
```

**Key Points**:
- L0 and L1 are already vectorized (current implementation)
- L2 is single-particle, then vmapped (matches current pattern)
- NO nested vmap/jit inside RK4
- Search function is called directly, not wrapped in another JIT

### 3.2 RK4 Integration (Factory Function)

```python
def create_rk4_step_gpu_fused_global_morton(
    mesh_gpu_morton,
    n_hops: int = 3
):
    """
    Create fused RK4 step function with global Morton L2 search.

    This matches the current architecture pattern:
    - Factory function creates the RK4 wrapper
    - Inner function handles data upload/download
    - Jitted RK4 function performs all GPU computation

    KEEP from Phase 3a:
    - Factory function pattern
    - Upload/download wrapper
    - GPU-resident interpolation
    - L0 + L1 searches

    REPLACE:
    - L2 search now uses global Morton (NO blocks)

    Parameters
    ----------
    mesh_gpu_morton : MeshGPUGlobalMorton
        GPU-resident global Morton structure
    n_hops : int, default=3
        Number of hops for L1 neighbor search

    Returns
    -------
    rk4_step_func : callable
        Function with signature (particle_data, velocity_field, dt, mesh_gpu, current_time)
    """

    def rk4_step_global_morton_impl(
        particle_data,
        velocity_field,
        dt: float,
        mesh_gpu: MeshDataGPU,
        current_time: float = 0.0
    ):
        """
        Production wrapper for GPU-fused RK4 with global Morton L2.

        Parameters
        ----------
        particle_data : ParticleData
            Particle data with positions, element_ids
        velocity_field : np.ndarray or jax.Array
            Velocity field at nodes (n_nodes, 3)
        dt : float
            Time step size
        mesh_gpu : MeshDataGPU
            GPU-resident mesh data (standard mesh arrays)
        current_time : float
            Current simulation time (unused, for API compatibility)

        Returns
        -------
        particle_data_updated : ParticleData
            Updated particle data
        stats : dict
            Timing statistics
        """
        # Extract particle data
        positions = particle_data.positions
        element_ids = particle_data.element_ids

        # Timing: Upload
        t_upload = time.time()

        # Upload positions and element IDs to GPU (if not already)
        if isinstance(positions, np.ndarray):
            positions_gpu = jax.device_put(positions.astype(np.float32))
        else:
            positions_gpu = positions

        if isinstance(element_ids, np.ndarray):
            element_ids_gpu = jax.device_put(element_ids.astype(np.int32))
        else:
            element_ids_gpu = element_ids

        # Upload velocity field
        if isinstance(velocity_field, np.ndarray):
            velocity_field_gpu = jax.device_put(velocity_field.astype(np.float32))
        else:
            velocity_field_gpu = velocity_field

        t_upload = time.time() - t_upload

        # Create inner jitted RK4 function
        @jax.jit
        def rk4_fused_global_morton(
            positions_gpu,
            element_ids_gpu,
            dt,
            connectivity_gpu,
            node_positions_gpu,
            element_neighbors_gpu,
            velocity_field_gpu
        ):
            """
            GPU-fused RK4 with L0+L1+L2 global Morton search.

            NO nested jit/vmap - all search is done via search_l0_l1_l2_global_morton.
            """

            # Stage 1: k1 = f(t, y)
            element_ids_k1 = search_l0_l1_l2_global_morton(
                positions_gpu,
                element_ids_gpu,
                node_positions_gpu,
                connectivity_gpu,
                element_neighbors_gpu,
                mesh_gpu_morton,
                n_hops=n_hops
            )
            velocities_k1 = interpolate_velocity_batch_gpu(
                positions_gpu,
                element_ids_k1,
                connectivity_gpu,
                node_positions_gpu,
                velocity_field_gpu
            )

            # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
            positions_k2 = positions_gpu + 0.5 * dt * velocities_k1
            element_ids_k2 = search_l0_l1_l2_global_morton(
                positions_k2,
                element_ids_k1,
                node_positions_gpu,
                connectivity_gpu,
                element_neighbors_gpu,
                mesh_gpu_morton,
                n_hops=n_hops
            )
            velocities_k2 = interpolate_velocity_batch_gpu(
                positions_k2,
                element_ids_k2,
                connectivity_gpu,
                node_positions_gpu,
                velocity_field_gpu
            )

            # Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
            positions_k3 = positions_gpu + 0.5 * dt * velocities_k2
            element_ids_k3 = search_l0_l1_l2_global_morton(
                positions_k3,
                element_ids_k2,
                node_positions_gpu,
                connectivity_gpu,
                element_neighbors_gpu,
                mesh_gpu_morton,
                n_hops=n_hops
            )
            velocities_k3 = interpolate_velocity_batch_gpu(
                positions_k3,
                element_ids_k3,
                connectivity_gpu,
                node_positions_gpu,
                velocity_field_gpu
            )

            # Stage 4: k4 = f(t + dt, y + dt * k3)
            positions_k4 = positions_gpu + dt * velocities_k3
            element_ids_k4 = search_l0_l1_l2_global_morton(
                positions_k4,
                element_ids_k3,
                node_positions_gpu,
                connectivity_gpu,
                element_neighbors_gpu,
                mesh_gpu_morton,
                n_hops=n_hops
            )
            velocities_k4 = interpolate_velocity_batch_gpu(
                positions_k4,
                element_ids_k4,
                connectivity_gpu,
                node_positions_gpu,
                velocity_field_gpu
            )

            # RK4 combination: y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            positions_final = positions_gpu + (dt / 6.0) * (
                velocities_k1 + 2*velocities_k2 + 2*velocities_k3 + velocities_k4
            )

            # Final search at new positions
            element_ids_final = search_l0_l1_l2_global_morton(
                positions_final,
                element_ids_gpu,  # Use initial as cached (could use k4)
                node_positions_gpu,
                connectivity_gpu,
                element_neighbors_gpu,
                mesh_gpu_morton,
                n_hops=n_hops
            )

            return positions_final, element_ids_final

        # Timing: GPU computation
        t_compute = time.time()
        positions_final_gpu, element_ids_final_gpu = rk4_fused_global_morton(
            positions_gpu,
            element_ids_gpu,
            dt,
            mesh_gpu.connectivity,
            mesh_gpu.node_positions,
            mesh_gpu.element_neighbors,
            velocity_field_gpu
        )
        # Force completion
        positions_final_gpu.block_until_ready()
        t_compute = time.time() - t_compute

        # Timing: Download
        t_download = time.time()
        positions_final = np.array(positions_final_gpu, dtype=np.float32)
        element_ids_final = np.array(element_ids_final_gpu, dtype=np.int32)
        t_download = time.time() - t_download

        t_total = time.time() - (t_upload + t_compute + t_download)

        # Update particle data
        from dataclasses import replace
        particle_data_updated = replace(
            particle_data,
            positions=positions_final,
            element_ids=element_ids_final
        )

        stats = {
            'time_upload': t_upload,
            'time_compute': t_compute,
            'time_download': t_download,
            'time_total': t_total,
            'n_particles': len(positions)
        }

        return particle_data_updated, stats

    # Return the wrapper function
    return rk4_step_global_morton_impl
```

**Key Architectural Points**:
1. **NO nested jit** - `rk4_fused_global_morton` is jitted once, search functions are NOT jitted again
2. **Single vmap in L2** - Only `search_l2_single` is vmapped inside `search_l0_l1_l2_global_morton`
3. **Matches current pattern** - Factory → wrapper → jitted computation
4. **NO block IDs** - Global Morton doesn't need block tracking

---

## Phase 4: Testing and Validation

### 4.1 Unit Tests

**File**: `test_global_morton_correctness.py` (NEW)

```python
"""Test global Morton structure correctness."""

def test_morton_encoding():
    """Test Morton code computation."""
    pass

def test_leaf_coverage():
    """Verify every element in exactly one leaf."""
    pass

def test_position_to_leaf():
    """Test position → leaf mapping."""
    pass

def test_search_in_leaf():
    """Test bounded search within leaf."""
    pass

def test_L2_vs_brute_force():
    """Compare L2 results against brute-force search."""
    pass
```

### 4.2 Quick Validation Test

**File**: `test_global_morton_validation.py` (NEW - replaces old test)

Same structure as old validation test, but:
- Remove all block-based configuration
- Use global Morton structure
- Test with 1,000 particles, 1 timestep
- Verify >95% initial assignment
- Verify no OOM

### 4.3 Production Test

**File**: `production_tracking_global_morton.py` (NEW - replaces old production script)

Same structure as old production script, but:
- Remove block configuration
- Use global Morton preprocessing
- 105,000 particles, 2,500 timesteps
- Target >95% retention

---

## What to Keep from Current Code

### ✅ KEEP (No Changes)

1. **Velocity Interpolation** (`jaxtrace/gpu/tracking/interpolation.py`):
   - `interpolate_velocity_batch_gpu()`
   - Barycentric coordinate computation
   - All interpolation kernels

2. **L0 Cached Search** (`jaxtrace/gpu/search/`):
   - `search_L0_single()`
   - `search_level0_vectorized()`

3. **L1 Multi-hop Neighbors** (`jaxtrace/gpu/search/`):
   - `search_L1_multihop_single()`
   - `search_level1_multihop_vectorized()`
   - All neighbor traversal logic

4. **RK4 Structure** (`jaxtrace/gpu/tracking/rk4_gpu_fused.py`):
   - 4-stage RK4 pattern
   - Stage-by-stage search + interpolation
   - Final position computation
   - Particle data updates

5. **Mesh Loading** (`jaxtrace/gpu/mesh_loader.py`):
   - PVTU loading
   - Connectivity/node handling
   - Velocity field extraction

6. **Particle Utilities** (`jaxtrace/gpu/particles.py`):
   - `ParticleData` dataclass
   - Seeding functions

7. **Element Neighbors** (`jaxtrace/gpu/forest.py`):
   - `build_element_neighbors_array()`

### ❌ REMOVE/REPLACE

1. **Block-based Files** (DELETE):
   - `jaxtrace/gpu/search/hot_morton_builder.py` - entire file wrong
   - `jaxtrace/gpu/search/hot_morton_search.py` - entire file wrong

2. **Block-based Configuration**:
   - Remove `GRID_SIZE = (8, 8, 4)`
   - Remove `MAX_ELEMENTS_PER_BLOCK`
   - Remove `compute_block_ids_batch()`
   - Remove all block-related code

3. **Old Test Scripts**:
   - Replace `test_hot_morton_validation.py`
   - Replace `production_tracking_3hop_l2_hot_morton.py`

---

## Critical JAX Constraints

### ⚠️ AVOID Nested JIT/VMAP

**Current Architecture (CORRECT)**:
```python
# Factory function (NOT jitted)
def create_rk4_step_gpu_fused_global_morton(...):

    # Wrapper function (NOT jitted)
    def rk4_step_global_morton_impl(...):

        # Inner RK4 function (JITTED ONCE)
        @jax.jit
        def rk4_fused_global_morton(...):
            # Calls search_l0_l1_l2_global_morton
            # Which has ONE vmap for L2
            ...

        # Call jitted function
        result = rk4_fused_global_morton(...)
```

**WRONG Patterns to Avoid**:
```python
# ❌ WRONG: Nested jit
@jax.jit
def outer():
    @jax.jit  # BAD - nested jit
    def inner():
        ...

# ❌ WRONG: Jitting already-jitted function
@jax.jit
def search_fn(...):
    ...

@jax.jit
def rk4(...):
    search_fn(...)  # BAD if search_fn is already jitted

# ❌ WRONG: Multiple vmaps in search path
element_ids = jax.vmap(jax.vmap(search_single))  # BAD - nested vmap
```

**CORRECT Pattern**:
```python
# ✅ GOOD: Single jit at top level
def search_l0_l1_l2_global_morton(...):  # NOT jitted
    # L0, L1: vectorized operations (no vmap needed)
    ...

    # L2: single vmap
    def search_l2_single(pos, elem_id_l1):
        ...

    result = jax.vmap(search_l2_single)(positions, element_ids)  # ONE vmap only
    return result

@jax.jit  # JITTED ONCE at top level
def rk4_fused_global_morton(...):
    # Calls search_l0_l1_l2_global_morton (not jitted)
    elem_ids = search_l0_l1_l2_global_morton(...)
    ...
```

### Key Rules:
1. **ONE JIT per RK4 step** - at the `rk4_fused_global_morton` level
2. **ONE VMAP for L2** - inside `search_l0_l1_l2_global_morton`
3. **NO JIT on search functions** - they're called from within jitted RK4
4. **L0/L1 already vectorized** - no vmap needed, use existing vectorized functions

---

## Implementation Checklist

### Phase 1: CPU Preprocessing
- [ ] Create `jaxtrace/gpu/search/morton_global_builder.py`
- [ ] Implement `interleave_bits_3d()`
- [ ] Implement `compute_morton_codes_for_elements()`
- [ ] Implement `build_global_morton_sorted_list()`
- [ ] Implement `build_fixed_capacity_leaves()`
- [ ] Implement `build_global_morton_structure()`
- [ ] Test on ThreadedA mesh (3.5M elements)

### Phase 2: GPU Search
- [ ] Create `jaxtrace/gpu/search/morton_global_search.py`
- [ ] Implement `interleave_bits_3d_jax()`
- [ ] Implement `morton_encode_position_jax()`
- [ ] Implement `position_to_leaf_id_linear()`
- [ ] Implement `search_in_leaf_global()` with bounded `lax.fori_loop`
- [ ] Implement `search_L2_global_morton_single()` (single particle, NO jit decorator)
- [ ] Implement `upload_global_morton_to_gpu()`
- [ ] Unit test each function independently

### Phase 3: RK4 Integration
- [ ] Modify `jaxtrace/gpu/tracking/rk4_gpu_fused.py`
- [ ] Add `search_l0_l1_l2_global_morton()` (NO jit decorator)
  - Use existing `search_level0_vectorized`
  - Use existing `search_level1_multihop_vectorized`
  - Define `search_l2_single` and vmap it (ONE vmap only)
- [ ] Add `create_rk4_step_gpu_fused_global_morton()` (factory function)
  - Inner wrapper function (not jitted)
  - Inner jitted `rk4_fused_global_morton` (JITTED ONCE)
- [ ] Remove old block-based L2 code (if desired)

### Phase 4: Testing
- [ ] Create `test_global_morton_validation.py` (1K particles, 1 step)
  - Remove all block configuration
  - Use global Morton preprocessing
  - Test initial assignment >95%
  - Verify no OOM
- [ ] Create `production_tracking_global_morton.py` (105K particles, 2.5K steps)
  - Remove all block configuration
  - Use global Morton preprocessing
  - Target >95% retention
  - Target 40-50k p/s throughput
- [ ] Run validation test
- [ ] Run production test
- [ ] Compare performance vs hierarchical 5-hop

---

## Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| **Initial Assignment** | >95% | L2 global Morton |
| **Retention (2,500 steps)** | >95% | L0+L1+L2 combined |
| **Throughput** | 40-50k p/s | Similar to Phase 3a |
| **Memory Overhead** | <200 MB | Global Morton arrays |
| **L0 Hit Rate** | 85-95% | Same as baseline |
| **L1 Hit Rate** | 99-99.5% | 3-hop neighbors |
| **L2 Hit Rate** | >99.9% | Global Morton |

---

## Future Enhancements (Optional)

### Phase 5: Geometric Octree Leaves
Replace linear leaf segmentation with true octree-aligned leaves:
- Split based on Morton prefix (3 bits per octree level)
- Each leaf = contiguous Morton range aligned with octant
- Better spatial locality

### Phase 6: Prefix Table Mapping
Replace linear Morton mapping with prefix lookup:
- Build `prefix_to_leaf[2^B]` table (e.g., B=12 → 4096 entries)
- Exact geometric mapping instead of approximation
- O(1) leaf lookup

### Phase 7: Blocks (If Needed)
Only add blocks if profiling shows necessity:
- Cube-aligned block partitioning
- Per-block Morton structures
- Better memory locality for very large meshes (>10M elements)

---

## Summary

**Key Changes from Original (Wrong) Implementation**:

| Aspect | Original (WRONG) | Revised (CORRECT) |
|--------|------------------|-------------------|
| **Structure** | 256 cube blocks | Single global list |
| **Morton Sorting** | Per-block | Global (entire mesh) |
| **Leaves** | Per-block octrees | Global leaf segments |
| **Search** | Block ID → leaf → search | Morton code → leaf → search |
| **Complexity** | High (blocks + leaves) | Low (just leaves) |
| **Memory** | 100-800 MB | <200 MB expected |
| **Code** | ~2000 lines | ~500 lines expected |

**Advantages of Global Design**:
1. **Simpler**: No block management overhead
2. **Smaller**: Less code, easier to debug
3. **JAX-friendly**: Fixed-size arrays, bounded loops
4. **Scalable**: Can add blocks later if needed
5. **HOT-like**: True hash-based octree philosophy

---

**Status**: ✅ PLAN COMPLETE - Ready to implement Phase 1

**Next Action**: Implement `morton_global_builder.py` and test on ThreadedA mesh
