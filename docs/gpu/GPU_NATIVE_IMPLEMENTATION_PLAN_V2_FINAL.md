# GPU-Native Particle Tracking Implementation Plan V2 - FINAL

**Status**: Reviewed and optimized for JAX GPU compatibility
**Date**: 2025-11-03
**Based on**: User review of V2 plan

---

## Executive Summary

This plan implements a **flat array, GPU-native architecture** incorporating all JAX best practices and user feedback. Key changes from initial V2:

1. ✅ **Removed `particle_velocities`** from scan carry (interpolated per step)
2. ✅ **Removed `particle_block_IDs`** from scan carry (derived from `element_block_IDs[particle_element_IDs]`)
3. ✅ **Field data stored at nodes** `(N_nodes, 3)` with gathering via `element_nodes`
4. ✅ **Configurable octree/block storage** (padded 2D vs flat+start/count)
5. ✅ **Minimal scan carry** (positions, element_IDs, active mask only)

---

## Configuration Options

### Config Class for User Control

```python
@dataclass
class GPUConfig:
    """Configuration for GPU particle tracking."""

    # Field data storage
    field_storage: str = "nodes"  # "nodes" or "elements"
    # - "nodes": (N_nodes, 3) - memory efficient, requires gather
    # - "elements": (N_elements, 4, 3) - faster access, duplicated data

    # Octree element storage
    octree_storage: str = "padded"  # "padded" or "flat"
    # - "padded": (N_octree_nodes, max_elements_per_node) - static shape, JAX-optimal
    # - "flat": flat array + start/count - compact if variance is high

    # Block element storage
    block_storage: str = "padded"  # "padded" or "flat"
    # - "padded": (N_blocks, max_elements_per_block) - static shape
    # - "flat": flat array + start/count - compact storage

    # Maximum sizes (for padded storage)
    max_neighbors: int = 4           # Maximum neighbors per element
    max_octree_neighbors: int = 26   # Maximum neighbor octree nodes (3D: 26)
    max_elements_per_octree_node: int = 1000  # For padded octree storage
    max_elements_per_block: int = 10000       # For padded block storage

    # Memory optimization
    store_particle_velocities: bool = False  # Store velocities in carry (NOT recommended)
    store_particle_block_ids: bool = False   # Store block IDs in carry (NOT recommended)

    # Precision
    position_dtype: str = "float64"  # Particle positions
    field_dtype: str = "float32"     # Field data (velocity, temperature)
    mesh_dtype: str = "float32"      # Mesh coordinates

    # Performance tuning
    particles_per_block_batch: int = 10000  # For vmap batching
    jit_compile: bool = True                # JIT compile all kernels

    def validate(self):
        """Validate configuration."""
        assert self.field_storage in ["nodes", "elements"]
        assert self.octree_storage in ["padded", "flat"]
        assert self.block_storage in ["padded", "flat"]
        if self.store_particle_velocities:
            warnings.warn("Storing particle velocities increases memory usage - not recommended")
        if self.store_particle_block_ids:
            warnings.warn("Storing particle block IDs is redundant - can be derived")
```

---

## Core Data Structures (FINAL)

### Minimal Scan Carry (Dynamic)

```python
# ============================================================================
# PARTICLE STATE - ONLY THIS GOES IN SCAN CARRY
# ============================================================================

particle_positions = jnp.array([...], dtype=float64)      # (N_particles, 3)
particle_element_IDs = jnp.array([...], dtype=int32)      # (N_particles,)
particle_active = jnp.array([...], dtype=bool)            # (N_particles,)

# Optional (NOT recommended - use config to enable)
particle_velocities = jnp.array([...], dtype=float64)     # (N_particles, 3) [if config.store_particle_velocities]
particle_block_IDs = jnp.array([...], dtype=int32)        # (N_particles,) [if config.store_particle_block_ids]

# Total minimal memory: N_particles × (3×8 + 1×4 + 1×1) = N_particles × 29 bytes
# For 1M particles: 29 MB
```

### Static Mesh Data (NOT in Carry)

```python
# ============================================================================
# MESH DATA - STATIC, PASSED AS CONSTANTS
# ============================================================================

# Nodes
node_positions = jnp.array([...], dtype=float32)          # (N_nodes, 3)

# Elements
element_nodes = jnp.array([...], dtype=int32)             # (N_elements, 4)
element_neighbors = jnp.array([...], dtype=int32)         # (N_elements, max_neighbors) - padded with -1
element_block_IDs = jnp.array([...], dtype=int32)         # (N_elements,)

# Field data (OPTION 1: node-based - RECOMMENDED)
if config.field_storage == "nodes":
    velocities = jnp.array([...], dtype=float32)          # (N_nodes, 3)
    # Access per element: velocities[element_nodes[elem_id, :], :]

# Field data (OPTION 2: element-based - faster but duplicated)
if config.field_storage == "elements":
    element_velocities = jnp.array([...], dtype=float32)  # (N_elements, 4, 3)
    # Direct access: element_velocities[elem_id]
```

### Static Octree Data (NOT in Carry)

```python
# ============================================================================
# OCTREE DATA - STATIC
# ============================================================================

octree_node_centers = jnp.array([...], dtype=float32)     # (N_octree_nodes, 3)
octree_node_halfsize = jnp.array([...], dtype=float32)    # (N_octree_nodes, 3)
octree_node_children = jnp.array([...], dtype=int32)      # (N_octree_nodes, 8) - child IDs, -1 if leaf
octree_node_block_IDs = jnp.array([...], dtype=int32)     # (N_octree_nodes,)
octree_node_neighbors = jnp.array([...], dtype=int32)     # (N_octree_nodes, max_octree_neighbors)

# OPTION 1: Padded 2D array (RECOMMENDED for JAX)
if config.octree_storage == "padded":
    octree_node_elements = jnp.array([...], dtype=int32)  # (N_octree_nodes, max_elements_per_node)
    # Access: octree_node_elements[node_id, :]
    # Mask invalid with: elem_id >= 0

# OPTION 2: Flat array with start/count (for high variance)
if config.octree_storage == "flat":
    octree_elements = jnp.array([...], dtype=int32)       # (total_element_refs,)
    octree_element_start = jnp.array([...], dtype=int32)  # (N_octree_nodes,)
    octree_element_count = jnp.array([...], dtype=int32)  # (N_octree_nodes,)
    # Access: octree_elements[start:start+count] via lax.dynamic_slice
```

### Static Block Data (NOT in Carry)

```python
# ============================================================================
# BLOCK DATA - STATIC
# ============================================================================

# OPTION 1: Padded 2D array (RECOMMENDED)
if config.block_storage == "padded":
    block_elements = jnp.array([...], dtype=int32)        # (N_blocks, max_elements_per_block)
    # Access: block_elements[block_id, :]

# OPTION 2: Flat array with start/count
if config.block_storage == "flat":
    block_elements_flat = jnp.array([...], dtype=int32)   # (total_block_elements,)
    block_element_start = jnp.array([...], dtype=int32)   # (N_blocks,)
    block_element_count = jnp.array([...], dtype=int32)   # (N_blocks,)
```

---

## JAX Compatibility Analysis

### 1. Field Data Storage

**OPTION 1: Node-based (RECOMMENDED)**
```python
velocities = jnp.array([...], dtype=float32)  # (N_nodes, 3)

# Access in element:
elem_velocities = velocities[element_nodes[elem_id, :], :]  # (4, 3)
```

**JAX Compatibility**: ✅ EXCELLENT
- Static indexing via `element_nodes`
- Memory efficient (no duplication)
- Vectorizes well: `velocities[element_nodes[elem_ids, :], :]` for batch

**OPTION 2: Element-based (OPTIONAL)**
```python
element_velocities = jnp.array([...], dtype=float32)  # (N_elements, 4, 3)

# Direct access:
elem_velocities = element_velocities[elem_id]  # (4, 3)
```

**JAX Compatibility**: ✅ EXCELLENT
- Faster access (no gather)
- More memory (duplicated node data)
- Good for cache locality

**Recommendation**: Use **node-based** unless profiling shows gather overhead is significant.

---

### 2. Octree Element Storage

**OPTION 1: Padded 2D (RECOMMENDED)**
```python
octree_node_elements = jnp.array([...], dtype=int32)  # (N_nodes, max_elements)

# Access:
elements_in_node = octree_node_elements[node_id, :]
valid_mask = elements_in_node >= 0
```

**JAX Compatibility**: ✅ OPTIMAL
- Static shape (JIT-friendly)
- Fast indexing
- Easy masking for invalid elements
- Predictable memory usage

**OPTION 2: Flat array + start/count**
```python
octree_elements = jnp.array([...], dtype=int32)
octree_element_start = jnp.array([...], dtype=int32)
octree_element_count = jnp.array([...], dtype=int32)

# Access:
start = octree_element_start[node_id]
count = octree_element_count[node_id]
elements_in_node = jax.lax.dynamic_slice(octree_elements, (start,), (max_check,))
valid_mask = jnp.arange(max_check) < count
```

**JAX Compatibility**: ✅ GOOD
- More compact (no wasted padding)
- Requires `lax.dynamic_slice` (slightly slower than direct indexing)
- Good if variance is very high

**Recommendation**: Use **padded 2D** unless memory is critical and variance is extreme (>10× difference in sizes).

---

### 3. Block Element Storage

**Same analysis as octree above.**

**Recommendation**: Use **padded 2D** for static shapes and optimal JAX performance.

---

### 4. Particle Velocities

**STORING (NOT RECOMMENDED)**
```python
particle_velocities = jnp.array([...], dtype=float64)  # (N_particles, 3)
# Must update in scan carry
```

**JAX Compatibility**: ✅ COMPATIBLE but wasteful
- Increases scan carry by 24 bytes/particle
- For 1M particles: +24 MB
- Velocity is always interpolated from field, so storing is redundant

**NOT STORING (RECOMMENDED)**
```python
# Derive per step:
particle_velocities = interpolate_velocity_batch(
    particle_positions,
    particle_element_IDs,
    element_nodes,
    velocities  # or element_velocities
)
```

**JAX Compatibility**: ✅ OPTIMAL
- No extra carry memory
- Velocity computed on-the-fly from static field
- Same computational cost (must interpolate for RK4 sub-steps anyway)

**Recommendation**: **Do NOT store** unless there's a physics reason (e.g., particle has inertia, drag, or velocity-dependent source terms).

---

### 5. Particle Block IDs

**STORING (NOT RECOMMENDED)**
```python
particle_block_IDs = jnp.array([...], dtype=int32)  # (N_particles,)
```

**JAX Compatibility**: ✅ COMPATIBLE but redundant
- Increases scan carry by 4 bytes/particle
- For 1M particles: +4 MB
- Can be derived instantly from element

**NOT STORING (RECOMMENDED)**
```python
# Derive on-the-fly:
particle_block_IDs = element_block_IDs[particle_element_IDs]
```

**JAX Compatibility**: ✅ OPTIMAL
- Zero extra memory
- Single array indexing (extremely fast on GPU)
- Always consistent (no risk of stale block IDs)

**Recommendation**: **Do NOT store** - derive as needed.

---

## Updated Multi-Level Search (With Config)

```python
@jax.jit
def multi_level_search(
    particle_pos: jnp.ndarray,           # (N_particles, 3)
    particle_elem_ID: jnp.ndarray,       # (N_particles,)
    config: GPUConfig,
    mesh_data: Dict,
    octree_data: Dict
) -> jnp.ndarray:
    """
    Multi-level element search with configurable storage.

    Args:
        particle_pos: Particle positions
        particle_elem_ID: Cached element IDs
        config: GPU configuration
        mesh_data: Static mesh arrays
        octree_data: Static octree arrays

    Returns:
        new_element_IDs: Updated element IDs (N_particles,)
    """
    # Unpack mesh data
    element_nodes = mesh_data['element_nodes']
    element_neighbors = mesh_data['element_neighbors']
    element_block_IDs = mesh_data['element_block_IDs']
    node_positions = mesh_data['node_positions']

    # Derive block IDs on-the-fly (not stored)
    valid_elem = particle_elem_ID >= 0
    safe_elem = jnp.where(valid_elem, particle_elem_ID, 0)
    particle_block_IDs = element_block_IDs[safe_elem]  # Derived, not stored!

    # Level 0: Cached element
    found_L0, elem_L0 = search_cached_elements_batch(
        particle_pos, particle_elem_ID, element_nodes, node_positions
    )

    # Level 1: Neighbor elements
    found_L1, elem_L1 = search_neighbor_elements_batch(
        particle_pos, particle_elem_ID, found_L0,
        element_neighbors, element_nodes, node_positions
    )

    # Level 2: Octree node elements (config-dependent storage)
    if config.octree_storage == "padded":
        found_L2, elem_L2 = search_octree_node_padded_batch(
            particle_pos, particle_block_IDs, found_L0, found_L1,
            octree_data['octree_node_elements'],
            octree_data['octree_node_block_IDs'],
            element_nodes, node_positions
        )
    else:  # flat
        found_L2, elem_L2 = search_octree_node_flat_batch(
            particle_pos, particle_block_IDs, found_L0, found_L1,
            octree_data['octree_elements'],
            octree_data['octree_element_start'],
            octree_data['octree_element_count'],
            octree_data['octree_node_block_IDs'],
            element_nodes, node_positions
        )

    # Level 3: Neighbor octree nodes (similar config-dependent logic)
    # ... similar to Level 2 ...

    # Combine results
    new_elem_IDs = jnp.where(
        found_L0, elem_L0,
        jnp.where(found_L1, elem_L1,
        jnp.where(found_L2, elem_L2,
            -1  # Not found
        ))
    )

    return new_elem_IDs
```

---

## Updated Time Loop (Minimal Carry)

```python
@jax.jit
def time_step_fn(particle_state, static_data):
    """
    Single time step with minimal scan carry.

    Args:
        particle_state: Dict with ONLY:
            - positions: (N_particles, 3)
            - element_IDs: (N_particles,)
            - active: (N_particles,)
        static_data: Dict with ALL mesh/field/octree data (NOT in carry)

    Returns:
        new_particle_state: Updated particle arrays
        None: No history accumulated
    """
    # Unpack particle state (ONLY these are in carry)
    positions = particle_state['positions']
    element_IDs = particle_state['element_IDs']
    active = particle_state['active']

    # Unpack static data (mesh, field, octree - NOT in carry)
    config = static_data['config']
    mesh_data = static_data['mesh']
    field_data = static_data['field']
    octree_data = static_data['octree']

    # Derive block IDs on-the-fly (NOT stored in carry)
    valid_elem = element_IDs >= 0
    safe_elem = jnp.where(valid_elem, element_IDs, 0)
    block_IDs = mesh_data['element_block_IDs'][safe_elem]

    # 1. Rebatch particles by block (for spatial locality)
    particles_by_block, counts = batch_particles_by_block(
        positions, element_IDs, block_IDs, config.N_blocks
    )

    # 2. Process blocks in parallel (vmap over blocks)
    updated_blocks = jax.vmap(block_update_fn, in_axes=(0, 0, None, None))(
        particles_by_block,
        counts,
        mesh_data,
        field_data
    )

    # 3. Unbatch particles
    new_positions, new_element_IDs, new_active = unbatch_particles(updated_blocks)

    # Return NEW state (minimal - only what changed)
    new_particle_state = {
        'positions': new_positions,
        'element_IDs': new_element_IDs,
        'active': new_active
    }

    return new_particle_state, None  # No history in carry!


@jax.jit
def block_update_fn(particles, count, mesh_data, field_data):
    """
    Update all particles in one block.

    Note: Velocities are interpolated per particle, NOT stored.
    """
    # Vectorize over particles (all in parallel)
    updated = jax.vmap(particle_update_fn, in_axes=(0, 0, None, None))(
        particles['positions'],
        particles['element_IDs'],
        mesh_data,
        field_data
    )

    # Mask out padding
    mask = jnp.arange(particles['positions'].shape[0]) < count
    updated = jax.tree_map(lambda x, m=mask: jnp.where(m[:, None], x, 0), updated)

    return updated


@jax.jit
def particle_update_fn(pos, elem_ID, mesh_data, field_data):
    """
    Update single particle (called via vmap).

    Note: Velocity is computed here, NOT retrieved from particle state.
    """
    # 1. Element search
    new_elem_ID = multi_level_search(
        pos.reshape(1, 3),
        jnp.array([elem_ID]),
        config,
        mesh_data,
        octree_data
    )[0]

    # 2. Interpolate velocity (NOT stored in particle state)
    if new_elem_ID >= 0:
        if config.field_storage == "nodes":
            # Gather from nodes
            elem_node_IDs = mesh_data['element_nodes'][new_elem_ID]
            elem_velocities = field_data['velocities'][elem_node_IDs]
        else:  # elements
            # Direct access
            elem_velocities = field_data['element_velocities'][new_elem_ID]

        interp_vel = interpolate_velocity(pos, elem_velocities, mesh_data, new_elem_ID)
    else:
        interp_vel = jnp.zeros(3)

    # 3. RK4 time integration
    new_pos = rk4_step(pos, interp_vel, dt, mesh_data, field_data, new_elem_ID)

    return new_pos, new_elem_ID


# Main simulation loop
initial_state = {
    'positions': particle_positions,
    'element_IDs': particle_element_IDs,
    'active': particle_active
}

static_data = {
    'config': config,
    'mesh': mesh_data,
    'field': field_data,
    'octree': octree_data
}

# Run simulation
final_state, trajectory = jax.lax.scan(
    time_step_fn,
    initial_state,
    static_data,
    length=N_time_steps
)
# If trajectory needed, use scan's xs output
```

---

## Memory Comparison

### Minimal Carry (RECOMMENDED)

```python
# Scan carry:
positions: 1M × 3 × 8 = 24 MB
element_IDs: 1M × 4 = 4 MB
active: 1M × 1 = 1 MB
# Total: 29 MB
```

### With Optional Fields (NOT RECOMMENDED)

```python
# If config.store_particle_velocities = True:
+ velocities: 1M × 3 × 8 = 24 MB
# Total: 53 MB (+83%)

# If config.store_particle_block_ids = True:
+ block_IDs: 1M × 4 = 4 MB
# Total: 33 MB (+14%)

# With both:
# Total: 57 MB (+97%)
```

**Conclusion**: Recommended config saves 50% memory in scan carry with no performance cost!

---

## Summary of Recommendations

| Feature | Recommended | Alternative | Config Flag |
|---------|------------|-------------|-------------|
| **Field storage** | Node-based `(N_nodes, 3)` | Element-based `(N_elements, 4, 3)` | `field_storage` |
| **Octree storage** | Padded 2D `(N, max_elem)` | Flat + start/count | `octree_storage` |
| **Block storage** | Padded 2D `(N, max_elem)` | Flat + start/count | `block_storage` |
| **Particle velocities** | NOT stored (derive) | Store in carry | `store_particle_velocities` (default=False) |
| **Particle block IDs** | NOT stored (derive) | Store in carry | `store_particle_block_ids` (default=False) |

---

## Implementation Checklist

- [ ] Create `GPUConfig` dataclass with validation
- [ ] Implement node-based field storage with gather
- [ ] Implement element-based field storage (optional)
- [ ] Implement padded octree/block storage
- [ ] Implement flat octree/block storage (optional)
- [ ] Remove `particle_velocities` from scan carry
- [ ] Remove `particle_block_IDs` from scan carry
- [ ] Derive block IDs on-the-fly: `element_block_IDs[particle_element_IDs]`
- [ ] Interpolate velocities per step (not store)
- [ ] Test both storage options with small meshes
- [ ] Benchmark memory usage for each config
- [ ] Profile performance for each config
- [ ] Document trade-offs in user guide

---

## Expected Performance (Minimal Carry)

**For 1M particles, 3.5M elements, 1000 time steps:**

| Metric | Value |
|--------|-------|
| **Scan carry** | 29 MB |
| **Static data** | ~300 MB (mesh + field + octree) |
| **Total GPU memory** | ~330 MB |
| **Time per step** | 0.1-0.2s |
| **Total time** | 100-200s |
| **Speedup vs CPU** | 10-100× |

---

## Conclusion

This FINAL plan incorporates all user feedback and represents the **optimal JAX GPU design** for particle tracking:

1. ✅ Minimal scan carry (29 bytes/particle)
2. ✅ No redundant storage (velocities/block IDs derived)
3. ✅ Static mesh/field data (not in carry)
4. ✅ Configurable storage (padded vs flat)
5. ✅ Node-based fields (memory efficient)
6. ✅ Flat arrays with masking
7. ✅ Full JAX compatibility
8. ✅ Memory safe (no explosion)

**Next step**: Implement with `GPUConfig` to allow users to choose trade-offs based on their specific mesh characteristics and memory constraints.
