"""
Single-particle search functions for L0, L1 (multi-hop), and L2 (octree).

Architecture:
- Single-particle functions operate on scalar inputs
- Python `if` statements for early exit (not jax.lax.cond or jnp.where)
- Outer vmap for parallelization over all particles
- Inner vmap still used for sub-processes (checking neighbors, etc.)

Usage pattern:
    def single_particle_rk4_step(position, element_id, ...):
        # Search with early exit
        elem_id = search_single_particle_with_fallback(position, element_id, ...)
        # Interpolate velocity
        velocity = interpolate_single_particle(position, elem_id, ...)
        # Update position
        ...
        return pos_new, elem_id_new

    # Batch processing with outer vmap
    batch_rk4_step = jax.vmap(single_particle_rk4_step)
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def point_in_tet_single_particle(
    point: jax.Array,
    tet_nodes: jax.Array,
    tolerance: float = 1e-10
) -> jax.Array:
    """
    Test if point is inside tetrahedron using barycentric coordinates.

    Parameters
    ----------
    point : jax.Array, shape (3,)
        Point position
    tet_nodes : jax.Array, shape (4, 3)
        Tetrahedron node positions
    tolerance : float
        Numerical tolerance for boundary cases

    Returns
    -------
    inside : jax.Array, scalar bool
        True if point is inside tetrahedron
    """
    v0, v1, v2, v3 = tet_nodes[0], tet_nodes[1], tet_nodes[2], tet_nodes[3]

    # Build matrix for barycentric coordinates
    mat = jnp.column_stack([v1 - v0, v2 - v0, v3 - v0])

    # Solve for barycentric coordinates
    det = jnp.linalg.det(mat)

    # Handle degenerate case
    is_degenerate = jnp.abs(det) < tolerance

    # Compute barycentric coordinates
    rhs = point - v0
    lambdas_123 = jnp.linalg.solve(mat, rhs)
    lambda_0 = 1.0 - jnp.sum(lambdas_123)

    all_lambdas = jnp.concatenate([jnp.array([lambda_0]), lambdas_123])

    # Check if all in [0, 1] with tolerance
    inside = jnp.all(all_lambdas >= -tolerance) & jnp.all(all_lambdas <= 1.0 + tolerance)

    # Return false for degenerate tets
    return jnp.where(is_degenerate, False, inside)


def search_level0_single(
    position: jax.Array,
    cached_element_id: jax.Array,
    node_positions: jax.Array,
    connectivity: jax.Array
) -> jax.Array:
    """
    L0 search for single particle: Check if still in cached element.

    Returns scalar element ID (not using jnp.where for control flow).

    Parameters
    ----------
    position : jax.Array, shape (3,)
        Particle position
    cached_element_id : jax.Array, scalar int32
        Last known element ID
    node_positions : jax.Array, shape (n_nodes, 3)
        Node coordinates
    connectivity : jax.Array, shape (n_elements, 4)
        Element-to-node connectivity

    Returns
    -------
    element_id : jax.Array, scalar int32
        cached_element_id if still inside, else -1
    """
    # Check if cached element is valid
    is_valid = (cached_element_id >= 0) & (cached_element_id < len(connectivity))

    # Get tet nodes (safe indexing for invalid cases)
    safe_idx = jnp.where(is_valid, cached_element_id, 0)
    node_ids = connectivity[safe_idx]
    tet_nodes = node_positions[node_ids]

    # Test if still inside
    inside = point_in_tet_single_particle(position, tet_nodes)

    # Return cached_element_id only if valid AND inside
    return jnp.where(is_valid & inside, cached_element_id, jnp.int32(-1))


def search_level1_multihop_single(
    position: jax.Array,
    cached_element_id: jax.Array,
    element_neighbors: jax.Array,
    node_positions: jax.Array,
    connectivity: jax.Array
) -> jax.Array:
    """
    L1 multi-hop search for single particle.

    Uses inner vmap for checking neighbors but returns scalar result.

    Parameters
    ----------
    position : jax.Array, shape (3,)
        Particle position
    cached_element_id : jax.Array, scalar int32
        Cached element ID from previous timestep
    element_neighbors : jax.Array, shape (n_elements, 4)
        Face neighbor connectivity (4 neighbors per element)
    node_positions : jax.Array, shape (n_nodes, 3)
        Node coordinates
    connectivity : jax.Array, shape (n_elements, 4)
        Element-to-node connectivity

    Returns
    -------
    element_id : jax.Array, scalar int32
        Found element ID (-1 if not found)

    Note
    ----
    Fixed at 5 hops for maximum accuracy (1,024 neighbors checked).
    Hop sizes: 1-hop=4, 2-hop=16, 3-hop=64, 4-hop=256, 5-hop=1,024
    """
    # Check if cached element is valid
    is_valid_cached = (cached_element_id >= 0) & (cached_element_id < len(element_neighbors))
    safe_cached_id = jnp.where(is_valid_cached, cached_element_id, 0)

    # Helper: Check a list of neighbors and return first match
    def check_neighbors_vectorized(neighbors_to_check):
        """Check list of neighbors and return first match (-1 if none)."""
        def check_neighbor(neighbor_id):
            valid = neighbor_id >= 0
            safe_id = jnp.where(valid, neighbor_id, 0)
            node_ids = connectivity[safe_id]
            tet_nodes = node_positions[node_ids]
            inside = point_in_tet_single_particle(position, tet_nodes)
            return jnp.where(valid & inside, safe_id, jnp.int32(-1))

        # Vectorize over neighbors
        found_ids = jax.vmap(check_neighbor)(neighbors_to_check)

        # Find first match
        n_neighbors = len(neighbors_to_check)
        found_indices = jnp.where(found_ids >= 0, jnp.arange(n_neighbors), n_neighbors)
        first_idx = jnp.min(found_indices)
        return jnp.where(first_idx < n_neighbors, found_ids[first_idx], jnp.int32(-1))

    # Helper: Expand frontier by one hop
    def expand_one_hop(neighbor_id):
        valid = neighbor_id >= 0
        safe_id = jnp.where(valid, neighbor_id, 0)
        return element_neighbors[safe_id]  # (4,)

    # Hop 1: Check 4 face neighbors
    hop1_neighbors = element_neighbors[safe_cached_id]  # (4,)
    result = check_neighbors_vectorized(hop1_neighbors)

    # Expand hop 1 → hop 2 (4 → 16)
    hop2_list = []
    for i in range(4):
        hop2_list.append(expand_one_hop(hop1_neighbors[i]))
    hop2_flat = jnp.concatenate(hop2_list)  # (16,)
    result2 = check_neighbors_vectorized(hop2_flat)
    result = jnp.where(result >= 0, result, result2)

    # Expand hop 2 → hop 3 (16 → 64)
    hop3_list = []
    for i in range(16):
        hop3_list.append(expand_one_hop(hop2_flat[i]))
    hop3_flat = jnp.concatenate(hop3_list)  # (64,)
    result3 = check_neighbors_vectorized(hop3_flat)
    result = jnp.where(result >= 0, result, result3)

    # Expand hop 3 → hop 4 (64 → 256)
    hop4_list = []
    for i in range(64):
        hop4_list.append(expand_one_hop(hop3_flat[i]))
    hop4_flat = jnp.concatenate(hop4_list)  # (256,)
    result4 = check_neighbors_vectorized(hop4_flat)
    result = jnp.where(result >= 0, result, result4)

    # Expand hop 4 → hop 5 (256 → 1,024)
    hop5_list = []
    for i in range(256):
        hop5_list.append(expand_one_hop(hop4_flat[i]))
    hop5_flat = jnp.concatenate(hop5_list)  # (1024,)
    result5 = check_neighbors_vectorized(hop5_flat)
    result = jnp.where(result >= 0, result, result5)

    # Only return result if cached_id was valid
    return jnp.where(is_valid_cached, result, jnp.int32(-1))


def search_level2_octree_single(
    position: jax.Array,
    octree_node_metadata: jax.Array,
    octree_node_elements: jax.Array,
    node_positions: jax.Array,
    connectivity: jax.Array
) -> jax.Array:
    """
    L2 octree search for single particle using lax.scan.

    Traverses octree from root to find containing element.

    Parameters
    ----------
    position : jax.Array, shape (3,)
        Particle position
    octree_node_metadata : jax.Array, shape (n_nodes, 15)
        Octree node metadata:
        - [0]: is_leaf (0.0 or 1.0)
        - [1:4]: bbox_min (x, y, z)
        - [4:7]: bbox_max (x, y, z)
        - [7:15]: children node IDs (8 values, -1 if empty)
    octree_node_elements : jax.Array, shape (n_nodes, max_leaf_size)
        Element IDs per node (-1 padding for leaves, unused for branches)
    node_positions : jax.Array, shape (n_nodes_mesh, 3)
        Node coordinates (for point-in-tet checks)
    connectivity : jax.Array, shape (n_elements, 4)
        Element-to-node connectivity

    Returns
    -------
    element_id : jax.Array, scalar int32
        Found element ID (-1 if not found)

    Note
    ----
    Fixed at 10 iterations (max octree depth).
    """
    def compute_octant(pos, bbox_min, bbox_max):
        """Compute octant index (0-7) for position within bounding box."""
        bbox_mid = (bbox_min + bbox_max) / 2.0

        # Binary encoding: [x >= mid_x] | [y >= mid_y] << 1 | [z >= mid_z] << 2
        octant = (
            (pos[0] >= bbox_mid[0]).astype(jnp.int32) +
            ((pos[1] >= bbox_mid[1]).astype(jnp.int32) << 1) +
            ((pos[2] >= bbox_mid[2]).astype(jnp.int32) << 2)
        )
        return octant

    def check_leaf_elements(pos, leaf_elements):
        """Check if position is inside any element in leaf node."""
        def check_one_element(elem_id):
            valid = elem_id >= 0
            safe_id = jnp.where(valid, elem_id, 0)

            # Get tet nodes
            node_ids = connectivity[safe_id]
            tet_nodes = node_positions[node_ids]

            # Check if inside
            inside = point_in_tet_single_particle(pos, tet_nodes)

            return jnp.where(valid & inside, safe_id, jnp.int32(-1))

        # Vectorize over all elements in leaf
        found_ids = jax.vmap(check_one_element)(leaf_elements)

        # Return first match
        n_elements = len(leaf_elements)
        found_indices = jnp.where(found_ids >= 0, jnp.arange(n_elements), n_elements)
        first_idx = jnp.min(found_indices)

        return jnp.where(first_idx < n_elements, found_ids[first_idx], jnp.int32(-1))

    def step(carry, _):
        """
        Single octree traversal step.

        Carries:
        - node_id: Current node index
        - found_id: Found element ID (-1 if not found yet)
        """
        node_id, found_id = carry

        # Load node metadata
        node_meta = octree_node_metadata[node_id]
        is_leaf = node_meta[0] > 0.5  # 1.0 for leaf, 0.0 for branch
        bbox_min = node_meta[1:4]
        bbox_max = node_meta[4:7]
        children = node_meta[7:15].astype(jnp.int32)

        # For leaf: check elements
        leaf_result = check_leaf_elements(position, octree_node_elements[node_id])

        # For branch: select child
        octant = compute_octant(position, bbox_min, bbox_max)
        child_id = children[octant]
        # If child is empty (-1), stay at current node
        next_child_id = jnp.where(child_id >= 0, child_id, node_id)

        # Select result based on leaf status
        new_found_id = jnp.where(is_leaf, leaf_result, jnp.int32(-1))
        new_node_id = jnp.where(is_leaf, node_id, next_child_id)

        # Early exit: if already found, keep current state
        final_node_id = jnp.where(found_id >= 0, node_id, new_node_id)
        final_found_id = jnp.where(found_id >= 0, found_id, new_found_id)

        return (final_node_id, final_found_id), None

    # Scan for up to 10 iterations (max octree depth)
    (_, element_id), _ = jax.lax.scan(
        step,
        (jnp.int32(0), jnp.int32(-1)),  # Initial carry: (root_node_id=0, found_id=-1)
        None,
        length=10
    )

    return element_id


def search_single_particle_with_fallback(
    position: jax.Array,
    element_id: jax.Array,
    node_positions: jax.Array,
    connectivity: jax.Array,
    element_neighbors: jax.Array,
    octree_node_metadata: jax.Array,
    octree_node_elements: jax.Array
) -> jax.Array:
    """
    Hierarchical search with fallback logic (L0 → L1 → L2).

    NOTE: Python `if` statements cannot be used with JAX traced values.
    When this function is called inside jax.jit or jax.vmap, all values are
    traced and Python `if result_ID < 0` causes TracerBoolConversionError.

    Instead, we use jnp.where to merge results:
    - All three search levels execute (no true early exit possible in JAX)
    - Results are merged: use L0 if found, else L1, else L2

    This is the only way to implement fallback logic in JAX JIT-compiled code.

    Parameters
    ----------
    position : jax.Array, shape (3,)
        Particle position
    element_id : jax.Array, scalar int32
        Cached element ID from previous timestep
    node_positions : jax.Array, shape (n_nodes, 3)
        Node coordinates
    connectivity : jax.Array, shape (n_elements, 4)
        Element-to-node connectivity
    element_neighbors : jax.Array, shape (n_elements, 4)
        Face neighbor connectivity
    octree_node_metadata : jax.Array, shape (n_nodes, 15)
        Octree metadata
    octree_node_elements : jax.Array, shape (n_nodes, max_leaf_size)
        Octree element arrays

    Returns
    -------
    element_id : jax.Array, scalar int32
        Found element ID (-1 if not found at any level)
    """
    # L0: Check cached element
    element_id_l0 = search_level0_single(position, element_id, node_positions, connectivity)

    # L1: Multi-hop neighbor search (fixed at 5 hops)
    element_id_l1 = search_level1_multihop_single(
        position, element_id, element_neighbors, node_positions, connectivity
    )

    # Merge L0 and L1: use L0 if found, else L1
    element_id_l0_l1 = jnp.where(element_id_l0 >= 0, element_id_l0, element_id_l1)

    # L2: Global octree fallback (fixed at 10 iterations)
    element_id_l2 = search_level2_octree_single(
        position, octree_node_metadata, octree_node_elements,
        node_positions, connectivity
    )

    # Merge L0+L1 and L2: use L0+L1 if found, else L2
    element_id_final = jnp.where(element_id_l0_l1 >= 0, element_id_l0_l1, element_id_l2)

    return element_id_final


def interpolate_single_particle(
    position: jax.Array,
    element_id: jax.Array,
    node_positions: jax.Array,
    connectivity: jax.Array,
    velocity_field: jax.Array
) -> jax.Array:
    """
    Interpolate velocity at particle position using barycentric coordinates.

    Parameters
    ----------
    position : jax.Array, shape (3,)
        Particle position
    element_id : jax.Array, scalar int32
        Element ID containing particle
    node_positions : jax.Array, shape (n_nodes, 3)
        Node coordinates
    connectivity : jax.Array, shape (n_elements, 4)
        Element-to-node connectivity
    velocity_field : jax.Array, shape (n_nodes, 3)
        Velocity at each node

    Returns
    -------
    velocity : jax.Array, shape (3,)
        Interpolated velocity at particle position (zero if element_id < 0)
    """
    # Check if element is valid
    is_valid = (element_id >= 0) & (element_id < len(connectivity))
    safe_id = jnp.where(is_valid, element_id, 0)

    # Get tet nodes and velocities
    node_ids = connectivity[safe_id]
    tet_nodes = node_positions[node_ids]  # (4, 3)
    tet_velocities = velocity_field[node_ids]  # (4, 3)

    # Compute barycentric coordinates
    v0, v1, v2, v3 = tet_nodes[0], tet_nodes[1], tet_nodes[2], tet_nodes[3]
    mat = jnp.column_stack([v1 - v0, v2 - v0, v3 - v0])

    # Solve for barycentric coordinates
    rhs = position - v0
    lambdas_123 = jnp.linalg.solve(mat, rhs)
    lambda_0 = 1.0 - jnp.sum(lambdas_123)

    # Combine all lambdas
    all_lambdas = jnp.concatenate([jnp.array([lambda_0]), lambdas_123])  # (4,)

    # Interpolate velocity: v = sum(lambda_i * v_i)
    velocity = jnp.sum(all_lambdas[:, jnp.newaxis] * tet_velocities, axis=0)  # (3,)

    # Return zero velocity if element is invalid
    return jnp.where(is_valid, velocity, jnp.zeros(3))


def single_particle_rk4_step(
    position: jax.Array,
    element_id: jax.Array,
    dt: float,
    node_positions: jax.Array,
    connectivity: jax.Array,
    element_neighbors: jax.Array,
    octree_node_metadata: jax.Array,
    octree_node_elements: jax.Array,
    velocity_field: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    Single-particle RK4 integration step.

    This is the complete RK4 integration for ONE particle, including:
    - Search for containing element at each stage
    - Velocity interpolation at each stage
    - RK4 position update

    Parameters
    ----------
    position : jax.Array, shape (3,)
        Particle position
    element_id : jax.Array, scalar int32
        Cached element ID from previous timestep
    dt : float
        Time step size
    node_positions : jax.Array, shape (n_nodes, 3)
        Node coordinates
    connectivity : jax.Array, shape (n_elements, 4)
        Element-to-node connectivity
    element_neighbors : jax.Array, shape (n_elements, 4)
        Face neighbor connectivity
    octree_node_metadata : jax.Array, shape (n_nodes, 15)
        Octree metadata
    octree_node_elements : jax.Array, shape (n_nodes, max_leaf_size)
        Octree element arrays
    velocity_field : jax.Array, shape (n_nodes, 3)
        Velocity at each node

    Returns
    -------
    position_new : jax.Array, shape (3,)
        Updated particle position
    element_id_new : jax.Array, scalar int32
        Updated element ID
    """
    # Stage 1: k1
    elem_id_k1 = search_single_particle_with_fallback(
        position, element_id, node_positions, connectivity, element_neighbors,
        octree_node_metadata, octree_node_elements
    )
    v1 = interpolate_single_particle(
        position, elem_id_k1, node_positions, connectivity, velocity_field
    )
    pos_k1 = position + 0.5 * dt * v1

    # Stage 2: k2
    elem_id_k2 = search_single_particle_with_fallback(
        pos_k1, elem_id_k1, node_positions, connectivity, element_neighbors,
        octree_node_metadata, octree_node_elements
    )
    v2 = interpolate_single_particle(
        pos_k1, elem_id_k2, node_positions, connectivity, velocity_field
    )
    pos_k2 = position + 0.5 * dt * v2

    # Stage 3: k3
    elem_id_k3 = search_single_particle_with_fallback(
        pos_k2, elem_id_k2, node_positions, connectivity, element_neighbors,
        octree_node_metadata, octree_node_elements
    )
    v3 = interpolate_single_particle(
        pos_k2, elem_id_k3, node_positions, connectivity, velocity_field
    )
    pos_k3 = position + dt * v3

    # Stage 4: k4
    elem_id_k4 = search_single_particle_with_fallback(
        pos_k3, elem_id_k3, node_positions, connectivity, element_neighbors,
        octree_node_metadata, octree_node_elements
    )
    v4 = interpolate_single_particle(
        pos_k3, elem_id_k4, node_positions, connectivity, velocity_field
    )

    # RK4 combination
    position_new = position + (dt / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4)

    # Final search at new position
    element_id_new = search_single_particle_with_fallback(
        position_new, elem_id_k4, node_positions, connectivity, element_neighbors,
        octree_node_metadata, octree_node_elements
    )

    return position_new, element_id_new


# Batch RK4 wrapper with outer vmap
def batch_rk4_step(
    positions: jax.Array,
    element_ids: jax.Array,
    dt: float,
    node_positions: jax.Array,
    connectivity: jax.Array,
    element_neighbors: jax.Array,
    octree_node_metadata: jax.Array,
    octree_node_elements: jax.Array,
    velocity_field: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    Batch RK4 integration using vmap over single-particle RK4.

    This is the outer parallelization layer that processes all particles
    in parallel on the GPU using vmap.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Particle positions
    element_ids : jax.Array, shape (N,)
        Cached element IDs
    dt : float
        Time step size
    [other parameters same as single_particle_rk4_step]

    Returns
    -------
    positions_new : jax.Array, shape (N, 3)
        Updated particle positions
    element_ids_new : jax.Array, shape (N,)
        Updated element IDs
    """
    # Create vmapped version of single-particle RK4
    # All mesh data (node_positions, connectivity, etc.) are broadcast
    # Only positions and element_ids are vectorized
    vmapped_rk4 = jax.vmap(
        lambda pos, elem_id: single_particle_rk4_step(
            pos, elem_id, dt, node_positions, connectivity, element_neighbors,
            octree_node_metadata, octree_node_elements, velocity_field
        )
    )

    return vmapped_rk4(positions, element_ids)
