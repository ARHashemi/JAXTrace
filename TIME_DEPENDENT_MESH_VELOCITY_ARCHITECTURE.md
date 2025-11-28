# Time-Dependent Mesh and Velocity Field Architecture

**Date:** 2025-11-27
**Status:** Design Document
**Branch:** gpu_native_implementation

---

## Executive Summary

This document provides a comprehensive architecture for supporting time-dependent mesh refinement and velocity field updates in GPU particle tracking simulations, with emphasis on:

1. **Minimal GPU memory transfers** (differential updates only)
2. **Preservation of JIT compilation** (no recompilation overhead)
3. **Efficient handling of local changes** (no cascade updates)
4. **Robust error handling** (detect and recover from topology changes)

**Key Design Principle:** Use JAX's `.at[].set()` operator for in-place GPU updates, avoiding full mesh reupload.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Use Cases](#use-cases)
3. [Architecture Overview](#architecture-overview)
4. [Time-Dependent Mesh Refinement](#time-dependent-mesh-refinement)
5. [Time-Dependent Velocity Field](#time-dependent-velocity-field)
6. [Combined Updates](#combined-updates)
7. [Implementation Details](#implementation-details)
8. [Performance Analysis](#performance-analysis)
9. [Error Handling and Validation](#error-handling-and-validation)
10. [Testing Strategy](#testing-strategy)
11. [API Design](#api-design)
12. [Migration Path](#migration-path)

---

## Problem Statement

### Current Limitation: Static Mesh and Velocity

**Current Architecture:**
- Mesh uploaded ONCE at initialization (117.5 MB)
- Velocity field uploaded ONCE at initialization (10.3 MB)
- Both remain static throughout simulation

**Real-World Requirements:**

**1. Adaptive Mesh Refinement (AMR)**
- Welding pool solidification: Refine mesh in high-gradient regions
- Moving heat source: Refine mesh following torch position
- Error-driven refinement: Refine where numerical error is high
- **Frequency:** Every 10-100 timesteps
- **Scope:** Local (1-5% of elements)

**2. Time-Varying Velocity Fields**
- Fluid flow evolution: Velocity changes at each timestep
- Thermal convection: Temperature-dependent flow patterns
- Multi-physics coupling: Velocity from separate solver
- **Frequency:** Every timestep or every N timesteps
- **Scope:** Global or local regions

### Challenge: Minimize GPU Transfer Overhead

**Naive Approach (UNACCEPTABLE):**
```python
# Every timestep:
mesh_gpu = upload_entire_mesh(mesh_cpu)  # 117.5 MB upload!
velocity_field_gpu = upload_velocity(velocity_cpu)  # 10.3 MB upload!
# Total: 127.8 MB × 2,500 steps = 320 GB transfers!
```

**Smart Approach (TARGET):**
```python
# Every timestep (only if changes occurred):
mesh_gpu = update_changed_elements(mesh_gpu, changed_ids, new_data)  # 0.56 MB
velocity_field_gpu = update_changed_nodes(velocity_field_gpu, changed_ids, new_velocities)  # Variable
# Total: << 1 GB transfers for entire simulation
```

---

## Use Cases

### Use Case 1: Local Mesh Refinement During Welding

**Scenario:**
- Welding torch moves along seam
- Refine mesh in molten pool region (high temperature gradient)
- Coarsen mesh in solidified region (low gradients)

**Characteristics:**
- **Frequency:** Every 50 timesteps (every 0.125 s)
- **Changed elements:** 1-2% (35k-70k elements)
- **Affected region:** Localized around torch position
- **Topology changes:** Yes (element insertion/deletion)

**Requirements:**
- Update connectivity for refined/coarsened elements
- Update element neighbors (face adjacency)
- Particles in refined region: reassign to new elements
- Particles in coarsened region: keep in merged element

### Use Case 2: Global Velocity Field Updates

**Scenario:**
- Velocity field computed by separate CFD solver
- Updated every 10 timesteps
- Entire domain updated (global change)

**Characteristics:**
- **Frequency:** Every 10 timesteps
- **Changed nodes:** 100% (900k nodes)
- **Transfer size:** 10.3 MB per update
- **Total transfer:** 10.3 MB × 250 updates = 2.58 GB

**Requirements:**
- Upload entire velocity field (no differential update possible)
- Minimize transfer time (use GPU-resident array)
- No mesh topology changes

### Use Case 3: Local Velocity Field Updates (Subregion)

**Scenario:**
- Fluid flow simulation with localized changes
- Only update nodes in active flow region
- 90% of domain is static

**Characteristics:**
- **Frequency:** Every timestep
- **Changed nodes:** 10% (90k nodes)
- **Transfer size:** 1.03 MB per timestep
- **Total transfer:** 1.03 MB × 2,500 = 2.58 GB

**Requirements:**
- Identify changed nodes (mask or index array)
- Update only changed nodes (differential)
- Preserve GPU-resident array

### Use Case 4: Mesh Refinement + Velocity Update (Coupled)

**Scenario:**
- Mesh refined in high-gradient region
- Velocity field interpolated onto new nodes
- Both mesh and velocity updated simultaneously

**Characteristics:**
- **Frequency:** Every 50 timesteps
- **Mesh changes:** 1-2% (35k-70k elements)
- **Velocity changes:** 1-2% nodes (9k-18k nodes)
- **Transfer size:** 0.56 MB (mesh) + 0.1 MB (velocity) = 0.66 MB

**Requirements:**
- Coordinate mesh and velocity updates
- Interpolate velocity onto new nodes (CPU or GPU)
- Atomic update (both succeed or both fail)

---

## Architecture Overview

### Core Principle: Differential Updates with JAX `.at[].set()`

**JAX Indexing Operators:**

```python
# Create new array with updated values (JAX is immutable)
new_array = old_array.at[indices].set(new_values)

# Equivalent to (but more efficient than):
new_array = old_array.copy()
new_array[indices] = new_values
```

**Key Properties:**
1. **Immutable:** Returns new array (doesn't modify original)
2. **Efficient:** Only copies changed elements, not entire array
3. **JIT-Compatible:** Compiles to efficient GPU kernel
4. **No Recompilation:** Shape-invariant (as long as array size doesn't change)

### Design Pattern: Update Functions Return New GPU State

```python
@dataclass
class MeshDataGPU:
    """GPU-resident mesh (immutable)"""
    connectivity: jax.Array          # (n_elements, 4)
    node_positions: jax.Array        # (n_nodes, 3)
    element_neighbors: jax.Array     # (n_elements, 4)
    n_elements: int
    n_nodes: int

def update_mesh_elements(
    mesh_gpu: MeshDataGPU,
    changed_element_ids: np.ndarray,      # (n_changed,) - CPU array
    new_connectivity: np.ndarray,          # (n_changed, 4) - CPU array
    new_element_neighbors: np.ndarray      # (n_changed, 4) - CPU array
) -> MeshDataGPU:
    """
    Update changed elements in GPU mesh.

    Returns NEW MeshDataGPU with updated arrays.
    Old MeshDataGPU is unchanged (immutable).
    """
    # Upload only changed data (not entire mesh!)
    changed_ids_gpu = jax.device_put(changed_element_ids)
    new_conn_gpu = jax.device_put(new_connectivity)
    new_neighb_gpu = jax.device_put(new_element_neighbors)

    # Update arrays on GPU (efficient, no full copy)
    updated_connectivity = mesh_gpu.connectivity.at[changed_ids_gpu].set(new_conn_gpu)
    updated_neighbors = mesh_gpu.element_neighbors.at[changed_ids_gpu].set(new_neighb_gpu)

    # Return new mesh (old mesh unchanged)
    return MeshDataGPU(
        connectivity=updated_connectivity,
        node_positions=mesh_gpu.node_positions,  # Unchanged
        element_neighbors=updated_neighbors,
        n_elements=mesh_gpu.n_elements,
        n_nodes=mesh_gpu.n_nodes
    )
```

**Transfer Volume:**
- Changed element IDs: `n_changed × 4 bytes`
- New connectivity: `n_changed × 4 nodes × 4 bytes = n_changed × 16 bytes`
- New neighbors: `n_changed × 4 neighbors × 4 bytes = n_changed × 16 bytes`
- **Total: `n_changed × 36 bytes`**

**For 35k changed elements:**
- Transfer: 35k × 36 = 1.26 MB (vs 117.5 MB full upload!)
- **Speedup: 93× less data transferred**

---

## Time-Dependent Mesh Refinement

### Mesh Refinement Operations

**Three Types of Refinement:**

**1. Element Subdivision**
- Split one element into multiple smaller elements
- Increases element count (n_elements changes!)
- Requires adding new nodes and connectivity

**2. Element Coarsening**
- Merge multiple elements into one larger element
- Decreases element count (n_elements changes!)
- May remove nodes

**3. Node Repositioning**
- Move existing nodes (no topology change)
- Element count unchanged
- Only node positions updated

### Challenge: Handling Element Count Changes

**Problem:** JAX JIT compilation assumes FIXED array shapes!

```python
# Initial mesh
connectivity: Array[3,512,384, 4]  # n_elements = 3,512,384

# After refinement (added 1,000 elements)
connectivity: Array[3,513,384, 4]  # n_elements = 3,513,384  ← SHAPE CHANGED!
```

**Impact:** JIT recompilation required (expensive!)

**Solution 1: Pre-Allocate Buffer (RECOMMENDED)**

```python
# Allocate mesh with extra capacity
MAX_ELEMENTS = 4_000_000  # Allow 15% growth
connectivity_gpu = jnp.zeros((MAX_ELEMENTS, 4), dtype=jnp.int32)
connectivity_gpu = connectivity_gpu.at[:n_elements_initial].set(initial_connectivity)

# Track active elements
active_element_mask = jnp.zeros(MAX_ELEMENTS, dtype=jnp.bool_)
active_element_mask = active_element_mask.at[:n_elements_initial].set(True)

# After refinement: add new elements at end
connectivity_gpu = connectivity_gpu.at[n_elements_current:n_elements_new].set(new_elements)
active_element_mask = active_element_mask.at[n_elements_current:n_elements_new].set(True)
```

**Advantages:**
- ✅ No shape changes (array size fixed at MAX_ELEMENTS)
- ✅ No JIT recompilation
- ✅ Fast updates (only modify active region)

**Disadvantages:**
- ❌ Memory overhead (pre-allocate 15% extra)
- ❌ Must track active element mask
- ❌ Limited growth (cannot exceed MAX_ELEMENTS)

**Solution 2: Accept JIT Recompilation on Refinement**

```python
# Allow shape to change
connectivity_gpu = jnp.concatenate([connectivity_gpu, new_elements_gpu])

# This triggers JIT recompilation (expensive!)
# But only happens every 50-100 steps
```

**Advantages:**
- ✅ No memory overhead
- ✅ No growth limit
- ✅ Exact mesh size

**Disadvantages:**
- ❌ JIT recompilation on every refinement (5-10 seconds!)
- ❌ Particles cannot be tracked during recompilation

**Recommendation:** Use Solution 1 (pre-allocate buffer) for most cases.

### Mesh Refinement Update Pattern

```python
def apply_mesh_refinement(
    mesh_gpu: MeshDataGPU,
    refinement_data: MeshRefinementData
) -> MeshDataGPU:
    """
    Apply mesh refinement to GPU mesh.

    Parameters
    ----------
    mesh_gpu : MeshDataGPU
        Current GPU mesh
    refinement_data : MeshRefinementData
        Refinement operation data
        - changed_element_ids: (n_changed,) - Elements to update
        - new_connectivity: (n_changed, 4) - New element-node connectivity
        - new_element_neighbors: (n_changed, 4) - New face neighbors
        - new_node_positions: (n_new_nodes, 3) - Positions of added nodes
        - node_id_offset: Offset for new node IDs

    Returns
    -------
    mesh_gpu_updated : MeshDataGPU
        Updated GPU mesh
    """
    # Upload changed data (small transfer)
    changed_ids_gpu = jax.device_put(refinement_data.changed_element_ids)
    new_conn_gpu = jax.device_put(refinement_data.new_connectivity)
    new_neighb_gpu = jax.device_put(refinement_data.new_element_neighbors)
    new_node_pos_gpu = jax.device_put(refinement_data.new_node_positions)

    # Update connectivity
    updated_connectivity = mesh_gpu.connectivity.at[changed_ids_gpu].set(new_conn_gpu)

    # Update element neighbors
    updated_neighbors = mesh_gpu.element_neighbors.at[changed_ids_gpu].set(new_neighb_gpu)

    # Add new nodes (if any)
    if len(refinement_data.new_node_positions) > 0:
        node_offset = refinement_data.node_id_offset
        n_new_nodes = len(refinement_data.new_node_positions)
        updated_node_positions = mesh_gpu.node_positions.at[node_offset:node_offset+n_new_nodes].set(new_node_pos_gpu)
    else:
        updated_node_positions = mesh_gpu.node_positions

    # Return updated mesh
    return MeshDataGPU(
        connectivity=updated_connectivity,
        node_positions=updated_node_positions,
        element_neighbors=updated_neighbors,
        n_elements=mesh_gpu.n_elements + refinement_data.n_elements_added,
        n_nodes=mesh_gpu.n_nodes + len(refinement_data.new_node_positions)
    )
```

### Computing Element Neighbors After Refinement

**Challenge:** When mesh topology changes, element neighbors must be recomputed.

**Naive Approach (WRONG):**
```python
# Recompute ALL element neighbors (expensive!)
new_neighbors = compute_element_neighbors_global(connectivity)  # O(n_elements²)
```

**Smart Approach (CORRECT):**
```python
# Only recompute neighbors for changed elements
def compute_element_neighbors_local(
    connectivity: np.ndarray,              # (n_elements, 4)
    changed_element_ids: np.ndarray,       # (n_changed,)
    search_radius: int = 2                 # Search within N-hop radius
) -> np.ndarray:
    """
    Compute face neighbors for changed elements only.

    Face neighbors = elements sharing a face (3 nodes in common).

    Strategy:
    1. For each changed element, get 4 faces
    2. For each face, find other elements with same 3 nodes
    3. That element is the face neighbor

    Complexity: O(n_changed × avg_degree) where avg_degree ≈ 20-50
    """
    neighbors = np.full((len(changed_element_ids), 4), -1, dtype=np.int32)

    # Build node-to-element map (only for changed region + neighborhood)
    region_elements = get_neighborhood(connectivity, changed_element_ids, search_radius)
    node_to_elem_map = build_node_to_element_map(connectivity[region_elements])

    for i, elem_id in enumerate(changed_element_ids):
        elem_nodes = connectivity[elem_id]  # (4,) node IDs

        # Generate 4 faces (each face = 3 nodes)
        faces = [
            (elem_nodes[0], elem_nodes[1], elem_nodes[2]),  # Face 0 (opposite node 3)
            (elem_nodes[0], elem_nodes[1], elem_nodes[3]),  # Face 1 (opposite node 2)
            (elem_nodes[0], elem_nodes[2], elem_nodes[3]),  # Face 2 (opposite node 1)
            (elem_nodes[1], elem_nodes[2], elem_nodes[3])   # Face 3 (opposite node 0)
        ]

        for face_idx, face_nodes in enumerate(faces):
            # Find element sharing this face
            # (element with same 3 nodes)
            neighbor_id = find_element_with_nodes(
                node_to_elem_map,
                face_nodes,
                exclude=elem_id
            )
            neighbors[i, face_idx] = neighbor_id

    return neighbors
```

**Complexity:**
- Node-to-element map: O(n_region × 4) where n_region ≈ n_changed × 50
- Face matching: O(n_changed × 4 × avg_degree) ≈ O(n_changed × 100)
- **Total: O(n_changed × n_neighbors) ≈ O(35k × 100) = 3.5M operations (fast!)**

**Transfer Volume:**
- Input: changed_element_ids (35k × 4 bytes = 140 KB)
- Output: neighbors (35k × 4 × 4 bytes = 560 KB)
- **Total: 700 KB**

---

## Time-Dependent Velocity Field

### Velocity Field Update Patterns

**Pattern 1: Global Update (All Nodes)**

```python
def update_velocity_field_global(
    velocity_field_gpu: jax.Array,         # (n_nodes, 3) - current
    new_velocity_field: np.ndarray          # (n_nodes, 3) - new values
) -> jax.Array:
    """
    Replace entire velocity field.

    Use when: Most nodes changed (>50%)
    Transfer: 10.3 MB for 900k nodes
    """
    return jax.device_put(new_velocity_field.astype(np.float32))
```

**Pattern 2: Differential Update (Changed Nodes Only)**

```python
def update_velocity_field_differential(
    velocity_field_gpu: jax.Array,         # (n_nodes, 3) - current
    changed_node_ids: np.ndarray,           # (n_changed,) - node indices
    new_velocities: np.ndarray              # (n_changed, 3) - new values
) -> jax.Array:
    """
    Update only changed nodes.

    Use when: Few nodes changed (<50%)
    Transfer: n_changed × 3 × 4 bytes
    Example: 10% nodes → 1.03 MB (vs 10.3 MB global)
    """
    changed_ids_gpu = jax.device_put(changed_node_ids)
    new_vels_gpu = jax.device_put(new_velocities.astype(np.float32))

    return velocity_field_gpu.at[changed_ids_gpu].set(new_vels_gpu)
```

**Pattern 3: Regional Update (Bounding Box)**

```python
def update_velocity_field_region(
    velocity_field_gpu: jax.Array,         # (n_nodes, 3)
    node_positions: np.ndarray,             # (n_nodes, 3)
    bbox: Tuple[float, float, float, float, float, float],  # (xmin, xmax, ymin, ymax, zmin, zmax)
    compute_velocity_func: Callable         # Function to compute velocity in region
) -> jax.Array:
    """
    Update velocity field in spatial region.

    Use when: Local changes (moving heat source, jet injection)
    """
    # Find nodes in bounding box (CPU)
    in_region = (
        (node_positions[:, 0] >= bbox[0]) & (node_positions[:, 0] <= bbox[1]) &
        (node_positions[:, 1] >= bbox[2]) & (node_positions[:, 1] <= bbox[3]) &
        (node_positions[:, 2] >= bbox[4]) & (node_positions[:, 2] <= bbox[5])
    )
    node_ids_in_region = np.where(in_region)[0]

    # Compute new velocities (CPU or GPU)
    new_velocities = compute_velocity_func(node_positions[node_ids_in_region])

    # Upload and update
    return update_velocity_field_differential(
        velocity_field_gpu,
        node_ids_in_region,
        new_velocities
    )
```

**Pattern 4: Incremental Update (Time Integration)**

```python
def update_velocity_field_incremental(
    velocity_field_gpu: jax.Array,         # (n_nodes, 3) - current
    velocity_delta_gpu: jax.Array           # (n_nodes, 3) - change (already on GPU!)
) -> jax.Array:
    """
    Update velocity by adding delta.

    Use when: Velocity computed by GPU-resident solver
    No CPU-GPU transfer! (delta already on GPU)
    """
    return velocity_field_gpu + velocity_delta_gpu
```

### Decision Tree: Which Update Pattern to Use?

```
Is velocity computed on GPU?
├─ YES → Use Pattern 4 (Incremental, no transfer)
└─ NO
    └─ How many nodes changed?
        ├─ >50% → Use Pattern 1 (Global, 10.3 MB)
        ├─ <50%, scattered → Use Pattern 2 (Differential)
        └─ <50%, localized → Use Pattern 3 (Regional)
```

### Velocity Field Interpolation for Refined Mesh

**Challenge:** When mesh is refined, new nodes are added. What velocity do they have?

**Option 1: Interpolate from Nearby Nodes (CPU)**

```python
def interpolate_velocity_at_new_nodes(
    new_node_positions: np.ndarray,     # (n_new, 3) - positions
    node_positions: np.ndarray,          # (n_nodes, 3) - existing
    velocity_field: np.ndarray           # (n_nodes, 3) - existing
) -> np.ndarray:
    """
    Interpolate velocity at new node positions.

    Uses inverse distance weighting (IDW) or RBF interpolation.
    """
    from scipy.interpolate import LinearNDInterpolator

    # Create interpolator (expensive, but only once per refinement)
    interpolator = LinearNDInterpolator(node_positions, velocity_field)

    # Interpolate at new positions
    new_velocities = interpolator(new_node_positions)  # (n_new, 3)

    return new_velocities
```

**Option 2: Interpolate from Parent Element (Mesh-Aware)**

```python
def interpolate_velocity_from_parent_element(
    new_node_positions: np.ndarray,     # (n_new, 3)
    parent_element_ids: np.ndarray,      # (n_new,) - which element was subdivided
    connectivity: np.ndarray,            # (n_elements, 4)
    node_positions: np.ndarray,          # (n_nodes, 3)
    velocity_field: np.ndarray           # (n_nodes, 3)
) -> np.ndarray:
    """
    Interpolate velocity using parent element's shape functions.

    More accurate than IDW (respects element boundaries).
    """
    new_velocities = np.zeros((len(new_node_positions), 3))

    for i, (pos, parent_id) in enumerate(zip(new_node_positions, parent_element_ids)):
        # Get parent element nodes
        parent_nodes = connectivity[parent_id]
        parent_node_pos = node_positions[parent_nodes]  # (4, 3)
        parent_node_vel = velocity_field[parent_nodes]  # (4, 3)

        # Compute barycentric coordinates
        bary_coords = compute_barycentric_coords(pos, parent_node_pos)

        # Interpolate velocity
        new_velocities[i] = bary_coords @ parent_node_vel

    return new_velocities
```

**Recommendation:** Use Option 2 (mesh-aware) for mesh refinement.

---

## Combined Updates

### Update Coordination: Mesh + Velocity

**Challenge:** Mesh and velocity must be updated atomically (both succeed or both fail).

```python
@dataclass
class MeshVelocityUpdate:
    """Combined mesh and velocity field update."""
    # Mesh updates
    changed_element_ids: np.ndarray        # (n_changed_elems,)
    new_connectivity: np.ndarray            # (n_changed_elems, 4)
    new_element_neighbors: np.ndarray      # (n_changed_elems, 4)
    new_node_positions: np.ndarray         # (n_new_nodes, 3)
    node_id_offset: int

    # Velocity updates
    changed_velocity_node_ids: np.ndarray  # (n_changed_vels,)
    new_velocities: np.ndarray              # (n_changed_vels, 3)

    # Metadata
    timestamp: float
    refinement_reason: str  # "gradient", "error", "manual"

def apply_combined_update(
    mesh_gpu: MeshDataGPU,
    velocity_field_gpu: jax.Array,
    update: MeshVelocityUpdate
) -> Tuple[MeshDataGPU, jax.Array]:
    """
    Apply mesh and velocity updates atomically.

    Returns
    -------
    mesh_gpu_new : MeshDataGPU
        Updated mesh
    velocity_field_gpu_new : jax.Array
        Updated velocity field
    """
    # Validate update
    validate_mesh_velocity_update(mesh_gpu, velocity_field_gpu, update)

    # Apply mesh update
    mesh_gpu_new = apply_mesh_refinement(mesh_gpu, update)

    # Apply velocity update
    velocity_field_gpu_new = update_velocity_field_differential(
        velocity_field_gpu,
        update.changed_velocity_node_ids,
        update.new_velocities
    )

    return mesh_gpu_new, velocity_field_gpu_new
```

### Update Frequency Strategies

**Strategy 1: Synchronous (Mesh and Velocity Together)**
```python
# Every 50 timesteps: both mesh and velocity updated
if step % 50 == 0:
    refinement_data = compute_mesh_refinement(...)
    velocity_update = compute_velocity_for_refined_mesh(refinement_data, ...)
    update = MeshVelocityUpdate(refinement_data, velocity_update)
    mesh_gpu, velocity_field_gpu = apply_combined_update(mesh_gpu, velocity_field_gpu, update)
```

**Strategy 2: Asynchronous (Independent Updates)**
```python
# Mesh refinement: every 50 timesteps
if step % 50 == 0:
    mesh_gpu = apply_mesh_refinement(mesh_gpu, refinement_data)

# Velocity update: every 10 timesteps
if step % 10 == 0:
    velocity_field_gpu = update_velocity_field_global(velocity_field_gpu, new_velocity)
```

**Strategy 3: Event-Driven (On-Demand)**
```python
# Mesh refinement: when error exceeds threshold
if estimate_error() > ERROR_THRESHOLD:
    mesh_gpu = apply_mesh_refinement(mesh_gpu, refinement_data)

# Velocity update: when flow pattern changes significantly
if velocity_change_magnitude() > VELOCITY_THRESHOLD:
    velocity_field_gpu = update_velocity_field_differential(...)
```

---

## Implementation Details

### Data Structures

```python
@dataclass
class MeshDataGPU:
    """GPU-resident mesh data (immutable)."""
    connectivity: jax.Array          # (MAX_ELEMENTS, 4) int32
    node_positions: jax.Array        # (MAX_NODES, 3) float32
    element_neighbors: jax.Array     # (MAX_ELEMENTS, 4) int32
    n_elements: int                  # Active element count
    n_nodes: int                     # Active node count
    max_elements: int                # Buffer capacity
    max_nodes: int                   # Buffer capacity
    memory_mb: float

@dataclass
class MeshRefinementData:
    """Mesh refinement operation data (CPU)."""
    changed_element_ids: np.ndarray        # (n_changed,) int32
    new_connectivity: np.ndarray            # (n_changed, 4) int32
    new_element_neighbors: np.ndarray      # (n_changed, 4) int32
    new_node_positions: np.ndarray         # (n_new_nodes, 3) float32
    node_id_offset: int                    # Where to insert new nodes
    n_elements_added: int                  # Net change in element count

    def validate(self):
        """Validate refinement data."""
        assert self.changed_element_ids.ndim == 1
        assert self.new_connectivity.shape == (len(self.changed_element_ids), 4)
        assert self.new_element_neighbors.shape == (len(self.changed_element_ids), 4)
        assert self.new_node_positions.ndim == 2 and self.new_node_positions.shape[1] == 3

@dataclass
class VelocityUpdateData:
    """Velocity field update data (CPU)."""
    changed_node_ids: np.ndarray          # (n_changed,) int32
    new_velocities: np.ndarray             # (n_changed, 3) float32
    update_type: str                       # "differential", "global", "regional"

    def validate(self):
        """Validate velocity update data."""
        assert self.changed_node_ids.ndim == 1
        assert self.new_velocities.shape == (len(self.changed_node_ids), 3)
```

### Update Functions

```python
def update_mesh_connectivity_gpu(
    mesh_gpu: MeshDataGPU,
    changed_ids: np.ndarray,
    new_connectivity: np.ndarray
) -> MeshDataGPU:
    """Update element connectivity on GPU."""
    changed_ids_gpu = jax.device_put(changed_ids)
    new_conn_gpu = jax.device_put(new_connectivity.astype(np.int32))

    updated_connectivity = mesh_gpu.connectivity.at[changed_ids_gpu].set(new_conn_gpu)

    return replace(mesh_gpu, connectivity=updated_connectivity)

def update_mesh_neighbors_gpu(
    mesh_gpu: MeshDataGPU,
    changed_ids: np.ndarray,
    new_neighbors: np.ndarray
) -> MeshDataGPU:
    """Update element neighbors on GPU."""
    changed_ids_gpu = jax.device_put(changed_ids)
    new_neighb_gpu = jax.device_put(new_neighbors.astype(np.int32))

    updated_neighbors = mesh_gpu.element_neighbors.at[changed_ids_gpu].set(new_neighb_gpu)

    return replace(mesh_gpu, element_neighbors=updated_neighbors)

def add_nodes_gpu(
    mesh_gpu: MeshDataGPU,
    new_node_positions: np.ndarray,
    node_offset: int
) -> MeshDataGPU:
    """Add new nodes to GPU mesh."""
    new_pos_gpu = jax.device_put(new_node_positions.astype(np.float32))
    n_new = len(new_node_positions)

    updated_positions = mesh_gpu.node_positions.at[node_offset:node_offset+n_new].set(new_pos_gpu)

    return replace(
        mesh_gpu,
        node_positions=updated_positions,
        n_nodes=mesh_gpu.n_nodes + n_new
    )
```

### Production Integration

```python
# File: production_tracking_threadeda.py

# Configuration
ENABLE_MESH_REFINEMENT = False  # Enable/disable mesh updates
REFINEMENT_FREQUENCY = 50       # Refine every N timesteps
ENABLE_VELOCITY_UPDATES = False # Enable/disable velocity updates
VELOCITY_UPDATE_FREQUENCY = 10  # Update every N timesteps

# Initialization
mesh_gpu = upload_mesh_to_gpu_with_buffer(
    connectivity, node_positions, element_neighbors,
    max_elements=int(n_elements * 1.15),  # 15% buffer
    max_nodes=int(n_nodes * 1.15)
)

velocity_field_gpu = jax.device_put(velocity_field.astype(np.float32))

# Time marching loop
for step in range(N_TIMESTEPS):
    # RK4 step
    particle_data, rk4_stats = rk4_step_gpu_fused_for_production(
        particle_data,
        velocity_field_gpu,  # GPU-resident velocity
        DT,
        mesh_gpu,
        current_time=step * DT,
        n_hops=RK4_L1_HOP_COUNT
    )

    # Mesh refinement (if enabled)
    if ENABLE_MESH_REFINEMENT and step % REFINEMENT_FREQUENCY == 0:
        refinement_data = compute_mesh_refinement(
            mesh_cpu,  # Keep CPU copy for refinement computation
            particle_data,
            refinement_criteria
        )
        mesh_gpu = apply_mesh_refinement(mesh_gpu, refinement_data)
        mesh_cpu = apply_mesh_refinement_cpu(mesh_cpu, refinement_data)

    # Velocity field update (if enabled)
    if ENABLE_VELOCITY_UPDATES and step % VELOCITY_UPDATE_FREQUENCY == 0:
        velocity_update = compute_velocity_update(
            mesh_cpu,
            particle_data,
            step * DT
        )
        velocity_field_gpu = apply_velocity_update(velocity_field_gpu, velocity_update)

    # Export (existing code)
    if step % EXPORT_FREQUENCY == 0:
        ...
```

---

## Performance Analysis

### Transfer Volume Analysis

**Scenario: Welding simulation with mesh refinement every 50 steps**

**Mesh Refinement:**
- Frequency: Every 50 timesteps (50 updates over 2,500 steps)
- Changed elements per update: 35k (1% of 3.5M elements)
- Transfer per update: 35k × 36 bytes = 1.26 MB
- **Total mesh transfer: 50 × 1.26 MB = 63 MB**

**Velocity Field Update:**
- Frequency: Every 10 timesteps (250 updates)
- Changed nodes: 10% (90k nodes)
- Transfer per update: 90k × 3 × 4 bytes = 1.08 MB
- **Total velocity transfer: 250 × 1.08 MB = 270 MB**

**Particle Data Transfer (current bottleneck):**
- Frequency: Every timestep (2,500 transfers)
- Transfer per timestep: 2 MB (positions + element_ids)
- **Total particle transfer: 2,500 × 2 MB = 5,000 MB**

**Total Transfer:**
- Mesh: 63 MB (1.2%)
- Velocity: 270 MB (5.2%)
- Particle: 5,000 MB (93.6%)
- **Total: 5,333 MB for entire simulation**

**Key Insight:** Even with frequent mesh and velocity updates, particle data transfers dominate (93.6%)!

**After Phase 3c (GPU-resident particles):**
- Mesh: 63 MB (19%)
- Velocity: 270 MB (81%)
- Particle: ~50 MB (only VTK exports, 0%)
- **Total: 383 MB (14× reduction!)**

### Computational Overhead

**Mesh Refinement Overhead:**
- Compute neighbors: O(n_changed × n_neighbors) ≈ 35k × 100 = 3.5M ops
- Upload to GPU: 1.26 MB @ 6 GB/s = **0.21 ms**
- Total per update: ~10-50 ms (negligible compared to 1-2 s per timestep)

**Velocity Update Overhead:**
- Upload to GPU: 1.08 MB @ 6 GB/s = **0.18 ms**
- Total per update: ~1 ms (negligible)

**JIT Recompilation Overhead:**
- With pre-allocated buffer: **0 ms** (no recompilation!)
- Without buffer (shape change): **5-10 seconds per refinement** (UNACCEPTABLE!)

### Memory Overhead

**Pre-Allocated Buffer:**
- Original mesh: 117.5 MB
- 15% buffer: 117.5 × 0.15 = 17.6 MB
- **Total: 135.1 MB (15% overhead)**

**Verdict:** 15% memory overhead is acceptable for avoiding JIT recompilation.

---

## Error Handling and Validation

### Validation Checks

```python
def validate_mesh_refinement(
    mesh_gpu: MeshDataGPU,
    refinement_data: MeshRefinementData
) -> None:
    """Validate mesh refinement data."""
    # Check element IDs are valid
    if np.any(refinement_data.changed_element_ids < 0):
        raise ValueError("Negative element IDs")
    if np.any(refinement_data.changed_element_ids >= mesh_gpu.n_elements):
        raise ValueError(f"Element IDs exceed mesh size ({mesh_gpu.n_elements})")

    # Check connectivity references valid nodes
    max_node_id = mesh_gpu.n_nodes + len(refinement_data.new_node_positions) - 1
    if np.any(refinement_data.new_connectivity < 0):
        raise ValueError("Negative node IDs in connectivity")
    if np.any(refinement_data.new_connectivity > max_node_id):
        raise ValueError(f"Node IDs exceed mesh size ({max_node_id})")

    # Check buffer capacity
    new_n_elements = mesh_gpu.n_elements + refinement_data.n_elements_added
    if new_n_elements > mesh_gpu.max_elements:
        raise ValueError(f"Element count ({new_n_elements}) exceeds buffer capacity ({mesh_gpu.max_elements})")

    new_n_nodes = mesh_gpu.n_nodes + len(refinement_data.new_node_positions)
    if new_n_nodes > mesh_gpu.max_nodes:
        raise ValueError(f"Node count ({new_n_nodes}) exceeds buffer capacity ({mesh_gpu.max_nodes})")

def validate_velocity_update(
    velocity_field_gpu: jax.Array,
    update_data: VelocityUpdateData
) -> None:
    """Validate velocity field update data."""
    n_nodes = velocity_field_gpu.shape[0]

    # Check node IDs are valid
    if np.any(update_data.changed_node_ids < 0):
        raise ValueError("Negative node IDs")
    if np.any(update_data.changed_node_ids >= n_nodes):
        raise ValueError(f"Node IDs exceed velocity field size ({n_nodes})")

    # Check velocities are finite
    if not np.all(np.isfinite(update_data.new_velocities)):
        raise ValueError("Non-finite velocities (NaN or Inf)")
```

### Particle Reassignment After Refinement

**Challenge:** Particles may be in elements that were subdivided/coarsened.

```python
def reassign_particles_after_refinement(
    particle_data: ParticleData,
    refinement_data: MeshRefinementData,
    mesh_gpu: MeshDataGPU
) -> ParticleData:
    """
    Reassign particles to new elements after mesh refinement.

    Strategy:
    1. Identify particles in changed elements
    2. Search for new containing element (within refined region)
    3. Update particle element_ids
    """
    # Identify particles in changed elements
    particles_in_changed = np.isin(particle_data.element_ids, refinement_data.changed_element_ids)

    if not np.any(particles_in_changed):
        return particle_data  # No particles affected

    # Get affected particles
    affected_positions = particle_data.positions[particles_in_changed]
    affected_ids = particle_data.element_ids[particles_in_changed]

    # Search for new containing elements
    # (Use search_level1_multihop_vectorized with high hop count)
    new_element_ids = search_particles_in_refined_region(
        affected_positions,
        affected_ids,
        mesh_gpu,
        search_region=refinement_data.changed_element_ids
    )

    # Update particle element IDs
    updated_element_ids = particle_data.element_ids.copy()
    updated_element_ids[particles_in_changed] = new_element_ids

    return replace(particle_data, element_ids=updated_element_ids)
```

---

## Testing Strategy

### Unit Tests

**Test 1: Differential Mesh Update**
```python
def test_differential_mesh_update():
    # Setup: Create simple mesh
    mesh_gpu = create_test_mesh_gpu(n_elements=1000, n_nodes=500)

    # Update 10 elements
    changed_ids = np.array([10, 20, 30, 40, 50])
    new_connectivity = np.random.randint(0, 500, size=(5, 4))
    new_neighbors = np.random.randint(-1, 1000, size=(5, 4))

    refinement = MeshRefinementData(
        changed_element_ids=changed_ids,
        new_connectivity=new_connectivity,
        new_element_neighbors=new_neighbors,
        new_node_positions=np.array([]),
        node_id_offset=500,
        n_elements_added=0
    )

    # Apply update
    mesh_gpu_new = apply_mesh_refinement(mesh_gpu, refinement)

    # Verify
    assert mesh_gpu_new.n_elements == mesh_gpu.n_elements
    assert mesh_gpu_new.n_nodes == mesh_gpu.n_nodes

    # Download and check
    connectivity_cpu = np.array(mesh_gpu_new.connectivity)
    for i, elem_id in enumerate(changed_ids):
        assert np.array_equal(connectivity_cpu[elem_id], new_connectivity[i])
```

**Test 2: Velocity Field Differential Update**
```python
def test_velocity_field_differential():
    # Setup
    velocity_gpu = jax.device_put(np.zeros((1000, 3), dtype=np.float32))

    # Update 100 nodes
    changed_ids = np.arange(100, 200)
    new_vels = np.random.randn(100, 3).astype(np.float32)

    # Apply
    velocity_gpu_new = update_velocity_field_differential(
        velocity_gpu, changed_ids, new_vels
    )

    # Verify
    velocity_cpu = np.array(velocity_gpu_new)
    assert np.allclose(velocity_cpu[100:200], new_vels)
    assert np.allclose(velocity_cpu[:100], 0)  # Unchanged
    assert np.allclose(velocity_cpu[200:], 0)  # Unchanged
```

### Integration Tests

**Test 3: Mesh Refinement + Particle Reassignment**
```python
def test_mesh_refinement_with_particles():
    # Setup: mesh + particles
    mesh_gpu = upload_test_mesh()
    particle_data = create_test_particles(n=1000)

    # Refine mesh (split element 100)
    refinement = create_refinement_split_element(elem_id=100)
    mesh_gpu_new = apply_mesh_refinement(mesh_gpu, refinement)

    # Reassign particles
    particle_data_new = reassign_particles_after_refinement(
        particle_data, refinement, mesh_gpu_new
    )

    # Verify: particles previously in element 100 now in one of the split elements
    particles_in_100 = particle_data.element_ids == 100
    new_elements = particle_data_new.element_ids[particles_in_100]
    assert np.all((new_elements == 100) | (new_elements == refinement.new_element_id))
```

**Test 4: Combined Mesh + Velocity Update**
```python
def test_combined_mesh_velocity_update():
    # Setup
    mesh_gpu = upload_test_mesh()
    velocity_gpu = jax.device_put(np.random.randn(mesh_gpu.n_nodes, 3))

    # Create combined update
    update = MeshVelocityUpdate(...)

    # Apply
    mesh_gpu_new, velocity_gpu_new = apply_combined_update(
        mesh_gpu, velocity_gpu, update
    )

    # Verify both updated correctly
    ...
```

### Performance Tests

**Test 5: Measure Transfer Overhead**
```python
def test_mesh_update_transfer_time():
    mesh_gpu = upload_large_mesh()

    # Measure differential update time
    changed_ids = np.arange(35000)
    new_data = create_random_refinement(35000)

    t_start = time.time()
    mesh_gpu_new = apply_mesh_refinement(mesh_gpu, new_data)
    jax.block_until_ready(mesh_gpu_new.connectivity)  # Force completion
    t_elapsed = time.time() - t_start

    print(f"Transfer time: {t_elapsed*1000:.2f} ms")
    assert t_elapsed < 0.01  # Should be < 10 ms
```

---

## API Design

### High-Level API

```python
class TimeHere's the complete document (continuing from the API Design section):

```python
class TimeDependentSimulation:
    """
    High-level API for time-dependent mesh and velocity simulations.

    Handles:
    - Mesh refinement/coarsening
    - Velocity field updates
    - Particle reassignment
    - GPU memory management
    """

    def __init__(
        self,
        mesh_path: str,
        velocity_field: np.ndarray,
        particle_data: ParticleData,
        config: SimulationConfig
    ):
        """
        Initialize time-dependent simulation.

        Parameters
        ----------
        mesh_path : str
            Path to initial mesh file
        velocity_field : np.ndarray
            Initial velocity field (n_nodes, 3)
        particle_data : ParticleData
            Initial particle state
        config : SimulationConfig
            Simulation configuration
        """
        # Load mesh
        self.mesh_cpu = load_mesh(mesh_path)

        # Upload to GPU with buffer
        self.mesh_gpu = upload_mesh_to_gpu_with_buffer(
            self.mesh_cpu.connectivity,
            self.mesh_cpu.node_positions,
            self.mesh_cpu.element_neighbors,
            max_elements=int(self.mesh_cpu.n_elements * config.mesh_buffer_factor),
            max_nodes=int(self.mesh_cpu.n_nodes * config.mesh_buffer_factor)
        )

        # Upload velocity field
        self.velocity_field_gpu = jax.device_put(velocity_field.astype(np.float32))

        # Store particle data
        self.particle_data = particle_data

        # Configuration
        self.config = config

        # Statistics
        self.stats = {
            'n_mesh_updates': 0,
            'n_velocity_updates': 0,
            'total_mesh_transfer_mb': 0.0,
            'total_velocity_transfer_mb': 0.0
        }

    def step(self, dt: float, current_time: float) -> ParticleData:
        """
        Execute one timestep with potential mesh/velocity updates.

        Parameters
        ----------
        dt : float
            Timestep size
        current_time : float
            Current simulation time

        Returns
        -------
        particle_data : ParticleData
            Updated particle state
        """
        # Check if mesh refinement needed
        if self._should_refine_mesh(current_time):
            self._apply_mesh_refinement()

        # Check if velocity update needed
        if self._should_update_velocity(current_time):
            self._apply_velocity_update(current_time)

        # RK4 step
        self.particle_data, rk4_stats = rk4_step_gpu_fused_for_production(
            self.particle_data,
            self.velocity_field_gpu,
            dt,
            self.mesh_gpu,
            current_time=current_time,
            n_hops=self.config.rk4_l1_hop_count
        )

        return self.particle_data

    def _should_refine_mesh(self, current_time: float) -> bool:
        """Check if mesh refinement should be performed."""
        if not self.config.enable_mesh_refinement:
            return False

        # Frequency-based
        if self.config.refinement_frequency > 0:
            step = int(current_time / self.config.dt)
            return step % self.config.refinement_frequency == 0

        # Error-based (TODO: implement error estimation)
        # error = estimate_error(self.particle_data, self.mesh_cpu)
        # return error > self.config.error_threshold

        return False

    def _should_update_velocity(self, current_time: float) -> bool:
        """Check if velocity field should be updated."""
        if not self.config.enable_velocity_updates:
            return False

        step = int(current_time / self.config.dt)
        return step % self.config.velocity_update_frequency == 0

    def _apply_mesh_refinement(self):
        """Apply mesh refinement."""
        # Compute refinement (CPU)
        refinement_data = self.config.refinement_strategy(
            self.mesh_cpu,
            self.particle_data,
            self.config.refinement_criteria
        )

        # Validate
        validate_mesh_refinement(self.mesh_gpu, refinement_data)

        # Apply to GPU mesh
        self.mesh_gpu = apply_mesh_refinement(self.mesh_gpu, refinement_data)

        # Apply to CPU mesh
        self.mesh_cpu = apply_mesh_refinement_cpu(self.mesh_cpu, refinement_data)

        # Reassign particles
        self.particle_data = reassign_particles_after_refinement(
            self.particle_data,
            refinement_data,
            self.mesh_gpu
        )

        # Update statistics
        self.stats['n_mesh_updates'] += 1
        transfer_mb = compute_transfer_size_mb(refinement_data)
        self.stats['total_mesh_transfer_mb'] += transfer_mb

        print(f"✓ Mesh refined: {len(refinement_data.changed_element_ids)} elements updated ({transfer_mb:.2f} MB)")

    def _apply_velocity_update(self, current_time: float):
        """Apply velocity field update."""
        # Compute new velocity (CPU or GPU)
        velocity_update = self.config.velocity_update_strategy(
            self.mesh_cpu,
            self.particle_data,
            current_time
        )

        # Validate
        validate_velocity_update(self.velocity_field_gpu, velocity_update)

        # Apply to GPU
        if velocity_update.update_type == "global":
            self.velocity_field_gpu = update_velocity_field_global(
                self.velocity_field_gpu,
                velocity_update.new_velocities
            )
            transfer_mb = velocity_update.new_velocities.nbytes / 1e6
        else:
            self.velocity_field_gpu = update_velocity_field_differential(
                self.velocity_field_gpu,
                velocity_update.changed_node_ids,
                velocity_update.new_velocities
            )
            transfer_mb = (velocity_update.changed_node_ids.nbytes + velocity_update.new_velocities.nbytes) / 1e6

        # Update statistics
        self.stats['n_velocity_updates'] += 1
        self.stats['total_velocity_transfer_mb'] += transfer_mb

        print(f"✓ Velocity updated: {len(velocity_update.changed_node_ids)} nodes ({transfer_mb:.2f} MB)")

    def get_statistics(self) -> dict:
        """Get simulation statistics."""
        return self.stats.copy()
```

### Configuration

```python
@dataclass
class SimulationConfig:
    """Configuration for time-dependent simulation."""
    # Mesh refinement
    enable_mesh_refinement: bool = False
    refinement_frequency: int = 50  # Refine every N steps (0 = error-driven)
    mesh_buffer_factor: float = 1.15  # Pre-allocate 15% extra capacity
    refinement_criteria: str = "gradient"  # "gradient", "error", "manual"
    refinement_strategy: Callable = None  # Function to compute refinement

    # Velocity updates
    enable_velocity_updates: bool = False
    velocity_update_frequency: int = 10  # Update every N steps
    velocity_update_strategy: Callable = None  # Function to compute velocity

    # RK4 parameters
    rk4_l1_hop_count: int = 3
    dt: float = 0.0025

    # Error thresholds (for adaptive refinement)
    error_threshold: float = 1e-3
    coarsen_threshold: float = 1e-5
```

---

## Migration Path

### Phase 1: Enable Velocity Field Updates (Easiest)

**Effort:** 2-3 hours

**Changes Required:**
1. Add `update_velocity_field_differential()` function
2. Add configuration flags to production script
3. Test with synthetic velocity field

**Example:**
```python
# production_tracking_threadeda.py

ENABLE_VELOCITY_UPDATES = True
VELOCITY_UPDATE_FREQUENCY = 10

# Time marching loop
for step in range(N_TIMESTEPS):
    # RK4 step
    particle_data, rk4_stats = rk4_step_gpu_fused_for_production(...)

    # Velocity update
    if ENABLE_VELOCITY_UPDATES and step % VELOCITY_UPDATE_FREQUENCY == 0:
        # Example: Add sinusoidal perturbation
        changed_node_ids = np.arange(0, mesh_gpu.n_nodes, 10)  # Every 10th node
        perturbation = 0.01 * np.sin(2 * np.pi * step / 100) * np.ones((len(changed_node_ids), 3))

        velocity_field_gpu = update_velocity_field_differential(
            velocity_field_gpu,
            changed_node_ids,
            perturbation
        )
```

### Phase 2: Enable Node Repositioning (Medium)

**Effort:** 4-6 hours

**Changes Required:**
1. Implement `update_node_positions_gpu()` function
2. Add node movement strategy (e.g., Laplacian smoothing)
3. No topology changes (element count unchanged)

**Use Case:** Mesh smoothing, ALE (Arbitrary Lagrangian-Eulerian) methods

### Phase 3: Enable Element Subdivision (Advanced)

**Effort:** 1-2 weeks

**Changes Required:**
1. Implement element subdivision algorithms (CPU)
2. Implement `compute_element_neighbors_local()` function
3. Implement particle reassignment after refinement
4. Add mesh buffer management
5. Extensive testing

**Use Case:** Adaptive mesh refinement (AMR)

---

## Summary and Recommendations

### Key Takeaways

1. **Differential Updates are Essential**
   - Full mesh upload: 117.5 MB per update (UNACCEPTABLE)
   - Differential update: 0.56-1.26 MB per update (93-99% reduction)

2. **JAX `.at[].set()` Enables Efficient GPU Updates**
   - No full array copy required
   - JIT-compatible (no recompilation)
   - Fast GPU kernel

3. **Pre-Allocated Buffers Avoid JIT Recompilation**
   - 15% memory overhead is acceptable
   - Avoids 5-10 second recompilation penalty
   - Enables shape-invariant updates

4. **Element Neighbor Computation is Local**
   - Only recompute for changed elements
   - O(n_changed × n_neighbors) complexity
   - Fast enough for real-time refinement

5. **Particle Data Transfers Still Dominate**
   - Even with mesh/velocity updates, particles are 93% of transfers
   - Phase 3c (GPU-resident particles) is still the top priority

### Recommendations

**Immediate (Phase 3b):**
1. ✅ Implement velocity field differential updates (easiest)
2. ✅ Test with synthetic velocity perturbations
3. ✅ Measure transfer overhead (should be < 1 ms)

**Short-Term (Phase 3c):**
1. ⚠️ Implement GPU-resident particle data (highest priority!)
2. ⚠️ Eliminates 5 GB particle transfers (93% of total)
3. ⚠️ Expected: 10-16× speedup

**Long-Term (Phase 3d):**
1. 🔄 Implement mesh refinement (element subdivision)
2. 🔄 Implement particle reassignment after refinement
3. 🔄 Test with realistic AMR scenarios

### Decision Matrix: When to Use Each Update Pattern

| Update Type | Frequency | Transfer Size | Use Case |
|-------------|-----------|---------------|----------|
| **Velocity Global** | Low (every 10-100 steps) | 10.3 MB | CFD coupling, global flow changes |
| **Velocity Differential** | Medium (every 1-10 steps) | 0.1-2 MB | Local flow changes, moving sources |
| **Velocity Incremental** | High (every step) | 0 MB | GPU-resident velocity solver |
| **Mesh Connectivity** | Low (every 50-100 steps) | 0.56 MB | Topology changes, subdivision |
| **Node Repositioning** | Medium (every 10-50 steps) | 0.1-1 MB | Mesh smoothing, ALE methods |
| **Combined Mesh+Velocity** | Low (every 50-100 steps) | 0.66 MB | AMR with velocity interpolation |

### Performance Targets (After Full Implementation)

**Current Baseline (2-hop, no time-dependency):**
- Throughput: 40k p/s
- Particle retention: 16%
- Total transfers: 5 GB

**With 3-Hop (Phase 3a):**
- Throughput: 15-20k p/s
- Particle retention: 90%+
- Total transfers: 5 GB

**With GPU-Resident Particles (Phase 3c):**
- Throughput: 150-320k p/s (10-16× improvement)
- Particle retention: 90%+
- Total transfers: 50 MB

**With Time-Dependent Updates (Phase 3d):**
- Throughput: 150-320k p/s (same as Phase 3c)
- Particle retention: 90%+
- Total transfers: 50 MB + 300 MB (mesh/velocity) = **350 MB (14× less than current!)**

**Final Target:**
- Throughput: 150-320k p/s
- Particle retention: 90%+
- Mesh refinement: Supported (every 50 steps)
- Velocity updates: Supported (every 10 steps)
- Total transfers: 350 MB (vs 5 GB current)

---

## Conclusion

Time-dependent mesh and velocity field updates are feasible on GPU with minimal performance impact, provided differential update strategies are used. The key insights are:

1. **Use JAX `.at[].set()` for efficient in-place updates**
2. **Pre-allocate buffers to avoid JIT recompilation**
3. **Only transfer changed data (not entire mesh/velocity)**
4. **Particle data transfers are still the bottleneck (Phase 3c priority!)**

This architecture enables realistic multi-physics simulations (AMR, fluid-structure interaction, thermal convection) while maintaining high GPU throughput.

**Next Steps:**
1. Implement velocity field differential updates (Phase 3b, 2-3 hours)
2. Test with synthetic perturbations
3. Proceed to Phase 3c (GPU-resident particles) for maximum speedup
4. Implement mesh refinement later (Phase 3d, when needed for production)
