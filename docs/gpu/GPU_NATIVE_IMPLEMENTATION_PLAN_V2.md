# GPU-Native Particle Tracking Implementation Plan V2

**Status**: New design based on flat arrays and JAX best practices
**Date**: 2025-11-03
**Replaces**: Previous forest-of-octrees hierarchical design

---

## Design Philosophy

This plan implements a **flat array, GPU-native architecture** that is fully compatible with JAX's execution model. All data structures are designed for:

1. **Static shapes**: All arrays have fixed dimensions known at compile time
2. **Vectorization**: All operations use `vmap` for parallelization
3. **No dynamic allocation**: Everything is pre-allocated and padded
4. **Memory safety**: Minimal scan carry, no history accumulation
5. **XLA optimization**: Enables full fusion and coalesced memory access

---

## Core Data Structures

### All Arrays Are Flat and Fixed-Size

```python
# ============================================================================
# MESH DATA (Static, passed as constants to JIT kernels)
# ============================================================================

# Elements
element_IDs = jnp.array([...], dtype=int32)         # (N_elements,)
element_nodes = jnp.array([...], dtype=int32)       # (N_elements, 4) - tetrahedral node IDs
element_block_IDs = jnp.array([...], dtype=int32)   # (N_elements,) - which block each element is in
element_neighbors = jnp.array([...], dtype=int32)   # (N_elements, max_neighbors) - padded with -1

# Nodes
node_IDs = jnp.array([...], dtype=int32)            # (N_nodes,)
node_positions = jnp.array([...], dtype=float32)    # (N_nodes, 3)

# Field data (velocity, temperature, etc.)
element_velocities = jnp.array([...], dtype=float32)  # (N_elements, 4, 3) - velocity at each node
element_temperatures = jnp.array([...], dtype=float32)  # (N_elements, 4) - if needed

# ============================================================================
# OCTREE DATA (Static, for spatial search)
# ============================================================================

octree_node_IDs = jnp.array([...], dtype=int32)           # (N_octree_nodes,)
octree_node_centers = jnp.array([...], dtype=float32)     # (N_octree_nodes, 3)
octree_node_halfsize = jnp.array([...], dtype=float32)    # (N_octree_nodes, 3)
octree_node_children = jnp.array([...], dtype=int32)      # (N_octree_nodes, 8) - child IDs, -1 if leaf
octree_node_block_IDs = jnp.array([...], dtype=int32)     # (N_octree_nodes,) - block ID for each node
octree_node_neighbors = jnp.array([...], dtype=int32)     # (N_octree_nodes, max_neighbors) - neighbor node IDs
octree_node_element_start = jnp.array([...], dtype=int32) # (N_octree_nodes,) - start index in element list
octree_node_element_count = jnp.array([...], dtype=int32) # (N_octree_nodes,) - number of elements in this node

# Flat list of elements per octree node (enables fast indexing)
octree_elements = jnp.array([...], dtype=int32)           # (total_element_refs,) - flattened element IDs

# ============================================================================
# PARTICLE DATA (Dynamic, carried in scan)
# ============================================================================

particle_IDs = jnp.array([...], dtype=int32)              # (N_particles,)
particle_positions = jnp.array([...], dtype=float64)      # (N_particles, 3)
particle_velocities = jnp.array([...], dtype=float64)     # (N_particles, 3)
particle_element_IDs = jnp.array([...], dtype=int32)      # (N_particles,) - cached element, -1 if unknown
particle_block_IDs = jnp.array([...], dtype=int32)        # (N_particles,) - current block
particle_active = jnp.array([...], dtype=bool)            # (N_particles,) - active mask

# ============================================================================
# BLOCK BATCHING DATA (For spatial locality)
# ============================================================================

block_IDs = jnp.array([...], dtype=int32)                 # (N_blocks,)
block_element_starts = jnp.array([...], dtype=int32)      # (N_blocks,) - start index in element list
block_element_counts = jnp.array([...], dtype=int32)      # (N_blocks,) - number of elements in block
```

### Key Design Principles

1. **Padding with -1**: All neighbor/child arrays are padded to fixed size with -1 for "no neighbor"
2. **Flat indexing**: Instead of "get all elements in block X", we use "element Y is in which block?"
3. **Static shapes**: All array dimensions are known at compile time
4. **No Python objects**: Only JAX arrays, no dataclasses or Python lists
5. **Masking over filtering**: Use boolean masks instead of dynamic array creation

---

## Multi-Level Search Algorithm

### Flat Search Logic (JAX-Compatible)

```python
@jax.jit
def multi_level_search(
    particle_pos: jnp.ndarray,           # (N_particles, 3)
    particle_elem_ID: jnp.ndarray,       # (N_particles,)
    element_nodes: jnp.ndarray,          # (N_elements, 4)
    element_neighbors: jnp.ndarray,      # (N_elements, max_neighbors)
    element_block_IDs: jnp.ndarray,      # (N_elements,)
    node_positions: jnp.ndarray,         # (N_nodes, 3)
    octree_node_block_IDs: jnp.ndarray,  # (N_octree_nodes,)
    octree_neighbors: jnp.ndarray,       # (N_octree_nodes, max_neighbors)
    octree_elements: jnp.ndarray,        # (total_element_refs,)
    octree_element_start: jnp.ndarray,   # (N_octree_nodes,)
    octree_element_count: jnp.ndarray    # (N_octree_nodes,)
) -> jnp.ndarray:                        # Returns: (N_particles,) new element IDs
    """
    Multi-level element search using flat arrays.

    Search hierarchy:
    1. Cached element (particle_elem_ID)
    2. Neighbor elements (element_neighbors[particle_elem_ID])
    3. Current octree node elements (octree_node with block_ID == element_block_IDs[particle_elem_ID])
    4. Neighbor octree node elements (octree_neighbors of current node)
    5. Parent octree node (rare fallback)

    All levels are fully vectorized with vmap.
    """

    # Level 0: Check cached element
    # Use particle_elem_ID as index into elements
    found_L0, elem_L0 = search_cached_elements_batch(
        particle_pos, particle_elem_ID, element_nodes, node_positions
    )

    # Level 1: Check neighbor elements
    # Use element_neighbors[particle_elem_ID] to get neighbor list
    found_L1, elem_L1 = search_neighbor_elements_batch(
        particle_pos, particle_elem_ID, found_L0,
        element_neighbors, element_nodes, node_positions
    )

    # Level 2: Search current octree node
    # Get octree_node_ID from element_block_IDs[particle_elem_ID]
    # Get elements in node using octree_elements[start:start+count]
    found_L2, elem_L2 = search_octree_node_batch(
        particle_pos, particle_elem_ID, found_L0, found_L1,
        element_block_IDs, octree_node_block_IDs,
        octree_elements, octree_element_start, octree_element_count,
        element_nodes, node_positions
    )

    # Level 3: Search neighbor octree nodes
    # Use octree_neighbors[octree_node_ID] to get neighbor nodes
    found_L3, elem_L3 = search_neighbor_octree_nodes_batch(
        particle_pos, particle_elem_ID, found_L0, found_L1, found_L2,
        element_block_IDs, octree_node_block_IDs, octree_neighbors,
        octree_elements, octree_element_start, octree_element_count,
        element_nodes, node_positions
    )

    # Combine results (use first level that found element)
    new_elem_IDs = jnp.where(
        found_L0, elem_L0,
        jnp.where(found_L1, elem_L1,
        jnp.where(found_L2, elem_L2,
        jnp.where(found_L3, elem_L3,
            -1  # Not found
        )))
    )

    return new_elem_IDs
```

### Key Implementation Details

**Level 0 (Cached Element)**:
```python
@jax.jit
def search_cached_elements_batch(particle_pos, particle_elem_ID, element_nodes, node_positions):
    """Check if particles are still in cached elements."""
    # Vectorize over all particles at once
    valid = particle_elem_ID >= 0
    safe_IDs = jnp.where(valid, particle_elem_ID, 0)

    # Get vertices: [N_particles, 4, 3]
    elem_node_IDs = element_nodes[safe_IDs]
    vertices = node_positions[elem_node_IDs]

    # Check containment (vectorized point-in-tet)
    inside = point_in_tetrahedron_batch(particle_pos, vertices)

    found = inside & valid
    result = jnp.where(found, particle_elem_ID, -1)

    return found, result
```

**Level 1 (Neighbor Elements)**:
```python
@jax.jit
def search_neighbor_elements_batch(particle_pos, particle_elem_ID, found_L0,
                                    element_neighbors, element_nodes, node_positions):
    """Check neighbor elements for particles not found in Level 0."""
    needs_search = ~found_L0 & (particle_elem_ID >= 0)
    safe_IDs = jnp.where(particle_elem_ID >= 0, particle_elem_ID, 0)

    # Get neighbors: [N_particles, max_neighbors]
    neighbors = element_neighbors[safe_IDs]

    def check_particle_neighbors(pos, neighs, search):
        """Check all neighbors for one particle."""
        def check_neighbor(neigh_ID):
            valid = neigh_ID >= 0
            safe_ID = jnp.where(valid, neigh_ID, 0)
            verts = node_positions[element_nodes[safe_ID]]
            inside = point_in_tet(pos, verts)
            return valid & inside, jnp.where(valid & inside, neigh_ID, -1)

        # vmap over neighbors (fixed size, e.g., max_neighbors=4)
        found_arr, id_arr = jax.vmap(check_neighbor)(neighs)

        found_any = jnp.any(found_arr)
        first_idx = jnp.argmax(found_arr)
        result = jnp.where(found_any & search, id_arr[first_idx], -1)

        return found_any & search, result

    # vmap over all particles
    found, result = jax.vmap(check_particle_neighbors)(particle_pos, neighbors, needs_search)

    return found, result
```

**Level 2 (Octree Node Elements)**:
```python
@jax.jit
def search_octree_node_batch(particle_pos, particle_elem_ID, found_L0, found_L1,
                              element_block_IDs, octree_node_block_IDs,
                              octree_elements, octree_element_start, octree_element_count,
                              element_nodes, node_positions):
    """Search elements in current octree node."""
    needs_search = ~found_L0 & ~found_L1 & (particle_elem_ID >= 0)

    # Get block ID for each particle from its cached element
    safe_elem_IDs = jnp.where(particle_elem_ID >= 0, particle_elem_ID, 0)
    particle_block_IDs = element_block_IDs[safe_elem_IDs]

    # Find octree node for this block
    # For each particle: which octree node has octree_node_block_IDs == particle_block_IDs[i]?
    # (This assumes one octree node per block - adjust if hierarchy is deeper)

    def check_particle_octree_node(pos, block_ID, search):
        """Check all elements in this particle's octree node."""
        if not search:
            return False, -1

        # Find octree node with this block_ID
        # In practice, you'd have a lookup: block_ID -> octree_node_ID
        # For now, assume 1:1 mapping or use jnp.where
        octree_node_ID = block_ID  # Simplified - adjust for your octree structure

        # Get element range for this node
        start = octree_element_start[octree_node_ID]
        count = octree_element_count[octree_node_ID]

        # Get elements in this node
        # Use fixed-size slice (pad if necessary)
        max_check = 1000  # Fixed maximum
        elem_slice = jax.lax.dynamic_slice(octree_elements, (start,), (max_check,))

        def check_element(elem_idx, elem_ID):
            """Check if element contains point."""
            valid = (elem_idx < count) & (elem_ID >= 0)
            safe_ID = jnp.where(valid, elem_ID, 0)
            verts = node_positions[element_nodes[safe_ID]]
            inside = point_in_tet(pos, verts)
            return valid & inside, jnp.where(valid & inside, elem_ID, -1)

        # vmap over elements in node
        elem_indices = jnp.arange(max_check)
        found_arr, id_arr = jax.vmap(check_element)(elem_indices, elem_slice)

        found_any = jnp.any(found_arr)
        first_idx = jnp.argmax(found_arr)
        result = jnp.where(found_any, id_arr[first_idx], -1)

        return found_any, result

    # vmap over all particles
    found, result = jax.vmap(check_particle_octree_node)(
        particle_pos, particle_block_IDs, needs_search
    )

    return found, result
```

---

## Loop Nesting Structure

### Optimal JAX/GPU Loop Hierarchy

```python
# ============================================================================
# OUTER LOOP: Time Marching (lax.scan)
# ============================================================================

@jax.jit
def time_step_fn(particle_state, t):
    """
    Single time step for all particles.

    Args:
        particle_state: Dict with particle arrays (positions, velocities, element_IDs, etc.)
        t: Current time (not used if dt is constant)

    Returns:
        new_particle_state: Updated particle arrays
        None: No history accumulated (use scan's xs if needed)
    """
    # All mesh/field data is captured from closure (static, not in carry)

    # 1. Rebatch particles by block for spatial locality
    particles_by_block, block_particle_counts = batch_particles_by_block(
        particle_state['positions'],
        particle_state['element_IDs'],
        element_block_IDs,
        N_blocks
    )

    # 2. Process all blocks in parallel (vmap over blocks)
    updated_blocks = jax.vmap(block_update_fn)(
        particles_by_block,
        block_particle_counts,
        mesh_data_by_block,  # Static
        field_data_by_block  # Static
    )

    # 3. Flatten particles back to global array
    new_particle_state = unbatch_particles(updated_blocks)

    return new_particle_state, None  # No history in carry


# Run full simulation
initial_state = {
    'positions': particle_positions,
    'velocities': particle_velocities,
    'element_IDs': particle_element_IDs,
    'block_IDs': particle_block_IDs,
    'active': particle_active
}

# Static mesh/field data (NOT in scan carry)
mesh_data = {
    'element_nodes': element_nodes,
    'element_neighbors': element_neighbors,
    'element_block_IDs': element_block_IDs,
    'node_positions': node_positions,
    'octree_*': octree_arrays,
}

field_data = {
    'element_velocities': element_velocities,
}

# Time integration with lax.scan
final_state, trajectory = jax.lax.scan(
    time_step_fn,
    initial_state,
    jnp.arange(N_time_steps),
    length=N_time_steps
)
# trajectory is optional - only store if needed (uses memory)

# ============================================================================
# MIDDLE LAYER: Block-Level Processing (vmap over blocks)
# ============================================================================

@jax.jit
def block_update_fn(particles_in_block, n_particles, block_mesh, block_field):
    """
    Update all particles in one block.

    Args:
        particles_in_block: Dict with particle arrays for this block [max_particles_per_block, ...]
        n_particles: Actual number of particles (rest are padding)
        block_mesh: Mesh data for this block (subset of global mesh)
        block_field: Field data for this block

    Returns:
        updated_particles: Updated particle arrays
    """
    # Vectorize over particles in block
    updated = jax.vmap(particle_update_fn)(
        particles_in_block['positions'],
        particles_in_block['velocities'],
        particles_in_block['element_IDs'],
        block_mesh,
        block_field
    )

    # Mask out padding (particles beyond n_particles)
    mask = jnp.arange(particles_in_block['positions'].shape[0]) < n_particles
    updated = jax.tree_map(lambda x, m=mask: jnp.where(m[:, None], x, 0), updated)

    return updated


# ============================================================================
# INNER LAYER: Per-Particle Processing (vmap over particles)
# ============================================================================

@jax.jit
def particle_update_fn(pos, vel, elem_ID, block_mesh, block_field):
    """
    Update single particle (called via vmap for all particles in block).

    Args:
        pos: Particle position [3]
        vel: Particle velocity [3]
        elem_ID: Cached element ID (scalar)
        block_mesh: Mesh data
        block_field: Field data

    Returns:
        new_pos, new_vel, new_elem_ID
    """
    # 1. Element search (multi-level)
    new_elem_ID = multi_level_search(
        pos.reshape(1, 3),  # Make it a batch of 1
        jnp.array([elem_ID]),
        block_mesh['element_nodes'],
        block_mesh['element_neighbors'],
        block_mesh['element_block_IDs'],
        block_mesh['node_positions'],
        # ... octree data ...
    )[0]  # Extract scalar result

    # 2. Interpolate velocity at current position
    if new_elem_ID >= 0:
        interp_vel = interpolate_velocity(
            pos, new_elem_ID, block_mesh, block_field
        )
    else:
        interp_vel = jnp.zeros(3)  # Particle left domain

    # 3. RK4 time integration
    new_pos, new_vel = rk4_step(
        pos, vel, interp_vel, dt,
        block_mesh, block_field, new_elem_ID
    )

    return new_pos, new_vel, new_elem_ID


# ============================================================================
# FINEST LAYER: Neighbor/Element Search (lax.fori_loop or small vmap)
# ============================================================================

# Already shown in multi_level_search above
# Uses vmap over fixed-size arrays (e.g., 4 neighbors, 8 octree children)
```

---

## Memory Safety Guarantees

### What Goes in Scan Carry (Dynamic)

**ONLY** these arrays are in the scan carry:
```python
particle_state = {
    'positions': jnp.ndarray,      # (N_particles, 3)
    'velocities': jnp.ndarray,     # (N_particles, 3)
    'element_IDs': jnp.ndarray,    # (N_particles,)
    'block_IDs': jnp.ndarray,      # (N_particles,)
    'active': jnp.ndarray,         # (N_particles,)
}
```

**Total memory**: N_particles × (3 + 3 + 1 + 1 + 1) × 4 bytes = N_particles × 36 bytes

For 1M particles: 36 MB (acceptable!)

### What Is Static (NOT in Carry)

**ALL** mesh and field data:
```python
# Captured from closure or passed as static args
mesh_data = {
    'element_nodes': ...,          # (N_elements, 4)
    'element_neighbors': ...,      # (N_elements, max_neighbors)
    'element_block_IDs': ...,      # (N_elements,)
    'node_positions': ...,         # (N_nodes, 3)
    'octree_*': ...,               # All octree arrays
}

field_data = {
    'element_velocities': ...,     # (N_elements, 4, 3)
}
```

These are **constant** throughout the simulation (unless mesh adapts, then passed as new static args).

### Pitfalls Avoided

❌ **DO NOT**:
- Put mesh/field data in scan carry
- Accumulate full trajectory in scan carry (use `xs` output if needed)
- Create Python lists inside jitted functions
- Use dynamic array shapes
- Nest scans (one is enough)

✅ **DO**:
- Keep scan carry minimal (only particles)
- Pass mesh/field as static arguments
- Use fixed-size arrays with padding
- Use masking instead of filtering
- Pre-allocate all arrays

---

## Implementation Phases

### Phase 1: Core Data Structures (Week 1)

**Goal**: Implement flat array data structures

**Tasks**:
1. Convert mesh to flat arrays
   - `element_nodes`, `element_neighbors`, `element_block_IDs`
   - `node_positions`
   - Pad neighbors to max_neighbors with -1

2. Build octree flat arrays
   - `octree_node_centers`, `octree_node_halfsize`
   - `octree_node_children` (padded to 8 with -1)
   - `octree_node_block_IDs`
   - `octree_elements` (flattened element list)
   - `octree_element_start`, `octree_element_count`

3. Initialize particle arrays
   - `particle_positions`, `particle_velocities`
   - `particle_element_IDs`, `particle_block_IDs`
   - `particle_active`

**Deliverables**:
- `jaxtrace/gpu/data_structures.py` - Array builders
- Unit tests for data structure creation
- Verify all arrays are JAX-compatible

### Phase 2: Multi-Level Search (Week 2)

**Goal**: Implement element search using flat arrays

**Tasks**:
1. Point-in-tetrahedron (vectorized)
   - `point_in_tetrahedron_batch(positions, vertices)`

2. Level 0: Cached element search
   - `search_cached_elements_batch(...)`

3. Level 1: Neighbor element search
   - `search_neighbor_elements_batch(...)`

4. Level 2: Octree node search
   - `search_octree_node_batch(...)`

5. Level 3: Neighbor octree nodes
   - `search_neighbor_octree_nodes_batch(...)`

6. Combine all levels
   - `multi_level_search(...)`

**Deliverables**:
- `jaxtrace/gpu/search_v2.py` - Search kernels
- Unit tests with small test meshes
- Benchmark against CPU baseline
- Memory profiling

### Phase 3: Velocity Interpolation (Week 3)

**Goal**: Implement field interpolation on GPU

**Tasks**:
1. Barycentric interpolation
   - `interpolate_velocity_batch(positions, element_IDs, field_data)`

2. Handle boundary cases
   - Particles on element faces
   - Particles outside domain

3. Vectorize over all particles

**Deliverables**:
- `jaxtrace/gpu/interpolation_v2.py`
- Unit tests for interpolation accuracy
- Compare with FEM analytical solutions

### Phase 4: RK4 Integration (Week 4)

**Goal**: Implement time integration on GPU

**Tasks**:
1. RK4 kernel (vectorized)
   - `rk4_step_batch(positions, velocities, field_data, dt)`

2. Sub-step element search
   - Search at each RK4 stage (k1, k2, k3, k4)

3. Adaptive time stepping (optional)

**Deliverables**:
- `jaxtrace/gpu/integrator_v2.py`
- Unit tests with known trajectories
- Convergence tests

### Phase 5: Block Batching (Week 5)

**Goal**: Implement spatial batching for locality

**Tasks**:
1. Batch particles by block
   - `batch_particles_by_block(particle_data, element_block_IDs)`

2. Block-level processing
   - `block_update_fn(particles, block_mesh, block_field)`

3. Unbatch particles
   - `unbatch_particles(particles_by_block)`

4. Handle load imbalance
   - Pad blocks to same size
   - Mask out padding

**Deliverables**:
- `jaxtrace/gpu/batching_v2.py`
- Tests with varying block sizes
- Memory profiling

### Phase 6: Time Loop Integration (Week 6)

**Goal**: Implement full time-marching loop

**Tasks**:
1. Single time step
   - `time_step_fn(particle_state, static_data)`

2. lax.scan wrapper
   - `run_simulation(initial_state, mesh_data, field_data, N_steps)`

3. Trajectory output (optional)
   - Use scan's `xs` for history

4. Memory optimization
   - Verify scan carry size
   - Profile GPU memory usage

**Deliverables**:
- `jaxtrace/gpu/simulator_v2.py`
- End-to-end integration tests
- Performance benchmarks
- Memory usage reports

### Phase 7: Optimization & Testing (Week 7)

**Goal**: Optimize and validate

**Tasks**:
1. Profile and optimize
   - JAX profiler
   - Memory profiler
   - Identify bottlenecks

2. Comprehensive testing
   - Validation cases (known solutions)
   - Large-scale tests (1M particles)
   - Edge cases (boundaries, degenerate elements)

3. Documentation
   - API reference
   - Usage examples
   - Performance guide

**Deliverables**:
- Optimized kernels
- Full test suite
- Documentation
- Benchmarks report

---

## Expected Performance

### Memory Usage

**For 1M particles, 3.5M elements mesh**:

| Component | Size | Memory |
|-----------|------|--------|
| Particle state (carry) | 1M × 36 bytes | 36 MB |
| Mesh data (static) | 3.5M × 16 bytes | 56 MB |
| Field data (static) | 3.5M × 48 bytes | 168 MB |
| Octree (static) | ~100K nodes × 80 bytes | 8 MB |
| **Total** | | **~270 MB** |

✅ Fits comfortably in 8 GB GPU memory

### Computational Performance

**Expected throughput** (based on flat array design):

- Level 0 hit (85%): ~10M particles/s (simple indexing + point-in-tet)
- Level 1 hit (10%): ~1M particles/s (4 neighbors × point-in-tet)
- Level 2 hit (5%): ~100K particles/s (100 elements × point-in-tet)

**Overall**: ~5-10M particle updates/s on modern GPU (A100/V100)

**Time per step** (1M particles): ~0.1-0.2s

**Speedup vs CPU**: 10-100× (depends on mesh size and hit rates)

---

## Comparison: Old vs New Design

| Aspect | Old Design (Forest-of-Octrees) | New Design (Flat Arrays) |
|--------|-------------------------------|------------------------|
| **Data Structure** | Hierarchical BlockMetadata objects | Flat JAX arrays |
| **Block Element Access** | `np.where(element_to_block == block_id)[0]` | `element_block_IDs[elem_id]` |
| **Element Search** | Dynamic list creation | Fixed-size vmap |
| **Memory Scaling** | O(n_particles × max_per_block) | O(n_particles × max_neighbors) |
| **JAX Compatibility** | ❌ Dynamic shapes | ✅ Static shapes |
| **GPU Memory** | 15 GB (OOM at 10K particles) | 270 MB (1M particles) |
| **Performance** | 50,000× slower than CPU | Expected 10-100× faster than CPU |
| **Load Imbalance** | Crashes on imbalanced meshes | Handles via padding/masking |

---

## Success Criteria

### Phase Completion Criteria

- ✅ All arrays are flat with fixed dimensions
- ✅ No dynamic allocation in JIT kernels
- ✅ Scan carry contains only particle arrays (<100 MB)
- ✅ All mesh/field data is static
- ✅ Memory usage scales linearly with N_particles
- ✅ Performance is 10-100× faster than CPU
- ✅ All tests pass (unit, integration, validation)
- ✅ Memory profiling shows no leaks
- ✅ Can run 1M particles for 1000 steps without OOM

### Non-Goals

- ❌ Mesh adaptation (future work)
- ❌ Multi-GPU (future work)
- ❌ Adaptive time stepping (optional)
- ❌ Field evolution (static fields only)

---

## Conclusion

This plan implements a **GPU-native, JAX-compatible** particle tracking system using:

1. **Flat arrays** instead of hierarchical structures
2. **Static shapes** instead of dynamic allocation
3. **Masking** instead of filtering
4. **Minimal scan carry** to avoid memory explosion
5. **vmap parallelization** at every level

This design is fundamentally different from the previous forest-of-octrees approach and addresses all the memory and performance issues identified in Phase 2 analysis.

**Expected outcome**: 10-100× speedup over CPU with <1 GB GPU memory for 1M particles.
