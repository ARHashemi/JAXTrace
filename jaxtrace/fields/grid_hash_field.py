"""
GPU-Accelerated Uniform Grid Hash Field for Temporal Batching

Fast spatial indexing using uniform grid hash table.
Designed for AMR data with variable mesh topology across timesteps.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class GridHashMesh:
    """Uniform grid hash mesh structure."""

    # Mesh geometry (variable per timestep)
    # Note: Kept as NumPy arrays to save GPU memory
    points: np.ndarray          # (N, 3) node positions
    connectivity: np.ndarray    # (M, 4) tetrahedral connectivity
    field_values: np.ndarray    # (N, 3) velocity at nodes

    # Grid hash structure
    grid_min: np.ndarray        # (3,) domain min
    grid_max: np.ndarray        # (3,) domain max
    cell_size: np.ndarray       # (3,) grid cell size
    grid_dims: np.ndarray       # (3,) number of cells per dimension

    # Hash table (elements -> grid cells)
    cell_elements: np.ndarray   # (n_cells, max_elem_per_cell) element indices
    cell_counts: np.ndarray     # (n_cells,) number of elements per cell
    max_elem_per_cell: int


def build_grid_hash_mesh(points: np.ndarray,
                        connectivity: np.ndarray,
                        field_values: np.ndarray,
                        grid_resolution: int = 32) -> GridHashMesh:
    """
    Build uniform grid hash spatial index.

    Much faster than octree (~100× faster build time).
    Good accuracy for uniform and AMR meshes.

    Parameters
    ----------
    points : np.ndarray
        Node positions (N, 3)
    connectivity : np.ndarray
        Tetrahedral connectivity (M, 4)
    field_values : np.ndarray
        Velocity at nodes (N, 3)
    grid_resolution : int
        Number of grid cells per dimension (default: 32)
        Higher = more memory but faster queries

    Returns
    -------
    GridHashMesh
        Grid hash structure
    """

    points = np.asarray(points, dtype=np.float32)
    connectivity = np.asarray(connectivity, dtype=np.int32)
    field_values = np.asarray(field_values, dtype=np.float32)

    n_elements = connectivity.shape[0]

    # Compute domain bounding box
    grid_min = points.min(axis=0)
    grid_max = points.max(axis=0)

    # Add small padding to avoid boundary issues
    domain_size = grid_max - grid_min
    padding = 0.01 * domain_size
    grid_min = grid_min - padding
    grid_max = grid_max + padding

    # Compute grid parameters
    grid_dims = np.array([grid_resolution, grid_resolution, grid_resolution], dtype=np.int32)
    cell_size = (grid_max - grid_min) / grid_dims

    # Compute element bounding boxes
    element_mins = np.zeros((n_elements, 3), dtype=np.float32)
    element_maxs = np.zeros((n_elements, 3), dtype=np.float32)

    for elem_idx in range(n_elements):
        node_indices = connectivity[elem_idx]
        elem_points = points[node_indices]
        element_mins[elem_idx] = elem_points.min(axis=0)
        element_maxs[elem_idx] = elem_points.max(axis=0)

    # Assign elements to grid cells
    n_cells = grid_resolution ** 3

    # First pass: count elements per cell
    cell_counts_temp = np.zeros(n_cells, dtype=np.int32)

    for elem_idx in range(n_elements):
        elem_min = element_mins[elem_idx]
        elem_max = element_maxs[elem_idx]

        # Find grid cell range this element overlaps
        cell_min = np.floor((elem_min - grid_min) / cell_size).astype(np.int32)
        cell_max = np.floor((elem_max - grid_min) / cell_size).astype(np.int32)

        # Clamp to grid bounds
        cell_min = np.maximum(cell_min, 0)
        cell_max = np.minimum(cell_max, grid_dims - 1)

        # Add to all overlapping cells
        for iz in range(cell_min[2], cell_max[2] + 1):
            for iy in range(cell_min[1], cell_max[1] + 1):
                for ix in range(cell_min[0], cell_max[0] + 1):
                    cell_idx = iz * grid_dims[1] * grid_dims[0] + iy * grid_dims[0] + ix
                    cell_counts_temp[cell_idx] += 1

    max_elem_per_cell = max(int(cell_counts_temp.max()), 1)

    # Second pass: fill element lists
    cell_elements = np.full((n_cells, max_elem_per_cell), -1, dtype=np.int32)
    cell_counts = np.zeros(n_cells, dtype=np.int32)

    for elem_idx in range(n_elements):
        elem_min = element_mins[elem_idx]
        elem_max = element_maxs[elem_idx]

        cell_min = np.floor((elem_min - grid_min) / cell_size).astype(np.int32)
        cell_max = np.floor((elem_max - grid_min) / cell_size).astype(np.int32)

        cell_min = np.maximum(cell_min, 0)
        cell_max = np.minimum(cell_max, grid_dims - 1)

        for iz in range(cell_min[2], cell_max[2] + 1):
            for iy in range(cell_min[1], cell_max[1] + 1):
                for ix in range(cell_min[0], cell_max[0] + 1):
                    cell_idx = iz * grid_dims[1] * grid_dims[0] + iy * grid_dims[0] + ix
                    count = cell_counts[cell_idx]
                    if count < max_elem_per_cell:
                        cell_elements[cell_idx, count] = elem_idx
                        cell_counts[cell_idx] += 1

    # Keep as NumPy arrays to save GPU memory
    # Will be converted to JAX only during interpolation as needed
    return GridHashMesh(
        points=points,  # Keep as numpy
        connectivity=connectivity,  # Keep as numpy
        field_values=field_values,  # Keep as numpy
        grid_min=grid_min,  # Keep as numpy
        grid_max=grid_max,  # Keep as numpy
        cell_size=cell_size,  # Keep as numpy
        grid_dims=grid_dims,  # Keep as numpy
        cell_elements=cell_elements,  # Keep as numpy
        cell_counts=cell_counts,  # Keep as numpy
        max_elem_per_cell=max_elem_per_cell
    )


@jax.jit
def point_in_tetrahedron(point: jnp.ndarray,
                        tet_nodes: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute barycentric coordinates (same as octree version)."""

    v0, v1, v2, v3 = tet_nodes[0], tet_nodes[1], tet_nodes[2], tet_nodes[3]

    v0p = point - v0
    v01 = v1 - v0
    v02 = v2 - v0
    v03 = v3 - v0

    mat = jnp.stack([v01, v02, v03], axis=1)
    det = jnp.linalg.det(mat)

    # Avoid division by zero
    det = jnp.where(jnp.abs(det) < 1e-10, 1.0, det)

    mat1 = jnp.stack([v0p, v02, v03], axis=1)
    mat2 = jnp.stack([v01, v0p, v03], axis=1)
    mat3 = jnp.stack([v01, v02, v0p], axis=1)

    b1 = jnp.linalg.det(mat1) / det
    b2 = jnp.linalg.det(mat2) / det
    b3 = jnp.linalg.det(mat3) / det
    b0 = 1.0 - b1 - b2 - b3

    bary_coords = jnp.array([b0, b1, b2, b3])

    tol = -1e-4
    is_inside = jnp.all(bary_coords >= tol) & jnp.all(bary_coords <= 1.0 + tol)

    return is_inside, bary_coords


@jax.jit
def interpolate_grid_hash(query_point: jnp.ndarray,
                         points: jnp.ndarray,
                         connectivity: jnp.ndarray,
                         field_values: jnp.ndarray,
                         grid_min: jnp.ndarray,
                         grid_max: jnp.ndarray,
                         cell_size: jnp.ndarray,
                         grid_dims: jnp.ndarray,
                         cell_elements: jnp.ndarray,
                         cell_counts: jnp.ndarray) -> jnp.ndarray:
    """
    Interpolate velocity at query point using grid hash.

    Fast spatial query using uniform grid.
    """

    # Find grid cell containing query point
    cell_idx_float = (query_point - grid_min) / cell_size
    cell_idx = jnp.floor(cell_idx_float).astype(jnp.int32)

    # Clamp to grid bounds
    cell_idx = jnp.maximum(cell_idx, 0)
    cell_idx = jnp.minimum(cell_idx, grid_dims - 1)

    # Convert 3D index to 1D
    cell_flat = cell_idx[2] * grid_dims[1] * grid_dims[0] + cell_idx[1] * grid_dims[0] + cell_idx[0]

    # Get candidate elements in this cell
    candidates = cell_elements[cell_flat]
    n_candidates = cell_counts[cell_flat]

    # Search for containing element
    def check_element(carry, elem_idx):
        found_prev, value_prev, score_prev = carry

        is_valid = elem_idx >= 0

        def check_current():
            node_indices = connectivity[elem_idx]
            tet_nodes = points[node_indices]

            is_inside, bary_coords = point_in_tetrahedron(query_point, tet_nodes)

            # Interpolate
            node_values = field_values[node_indices]
            interpolated = jnp.dot(bary_coords, node_values)

            # Score based on barycentric quality
            bary_min = jnp.min(bary_coords)
            bary_max = jnp.max(bary_coords)
            score = -jnp.maximum(jnp.abs(bary_min), jnp.abs(bary_max - 1.0))

            return is_inside, interpolated, score

        def skip():
            return jnp.array(False), jnp.zeros(3, dtype=jnp.float32), jnp.array(-9999.0)

        found_curr, value_curr, score_curr = jax.lax.cond(is_valid, check_current, skip)

        # Update best
        is_better = found_curr & (score_curr > score_prev)
        new_found = found_prev | found_curr
        new_value = jnp.where(is_better, value_curr, value_prev)
        new_score = jnp.where(is_better, score_curr, score_prev)

        return (new_found, new_value, new_score), None

    init = (jnp.array(False), jnp.zeros(3, dtype=jnp.float32), jnp.array(-9999.0))
    (found, interpolated, _), _ = jax.lax.scan(check_element, init, candidates)

    # Fallback: inverse distance weighting from nearest cell's first element
    def fallback():
        first_elem = candidates[0]
        node_indices = connectivity[first_elem]
        node_positions = points[node_indices]
        node_values = field_values[node_indices]

        dists = jnp.sqrt(jnp.sum((node_positions - query_point)**2, axis=1) + 1e-10)
        weights = 1.0 / dists
        weights = weights / jnp.sum(weights)

        return jnp.dot(weights, node_values)

    result = jnp.where(found, interpolated, fallback())

    return result


def create_grid_hash_interpolator(mesh: GridHashMesh, streaming: bool = True, batch_size: int = 1000):
    """
    Create grid hash interpolator.

    Parameters
    ----------
    mesh : GridHashMesh
        Grid hash mesh structure
    streaming : bool
        If True, use CPU-based streaming (low memory, slower)
        If False, use batched GPU mode (balanced memory, faster)
    batch_size : int
        Number of particles to process per GPU batch (default: 1000)
        Only used when streaming=False
    """

    if streaming:
        # CPU STREAMING MODE: Pure CPU interpolation (low memory, slower)
        return _create_streaming_interpolator(mesh)
    else:
        # BATCHED GPU MODE: Keep mesh on CPU, transfer batches to GPU (balanced)
        return _create_batched_gpu_interpolator(mesh, batch_size)


def _create_streaming_interpolator(mesh: GridHashMesh):
    """
    Create streaming interpolator: Keep mesh on CPU, do interpolation in batches.

    This significantly reduces GPU memory usage by:
    1. Keeping mesh data on CPU (NumPy arrays)
    2. Finding candidate elements on CPU (grid hash lookup)
    3. Transferring only relevant data to GPU for interpolation
    """

    # Keep mesh on CPU (already NumPy arrays)
    points = mesh.points
    connectivity = mesh.connectivity
    field_values = mesh.field_values
    grid_min = mesh.grid_min
    grid_max = mesh.grid_max
    cell_size = mesh.cell_size
    grid_dims = mesh.grid_dims
    cell_elements = mesh.cell_elements
    cell_counts = mesh.cell_counts

    # Convert grid metadata to JAX (small arrays, safe for GPU)
    grid_min_jax = jnp.array(grid_min)
    grid_max_jax = jnp.array(grid_max)
    cell_size_jax = jnp.array(cell_size)
    grid_dims_jax = jnp.array(grid_dims)

    def grid_hash_interpolate_streaming(query_points: jnp.ndarray) -> jnp.ndarray:
        """
        Interpolate at query points using streaming approach.

        Process in batches to avoid loading entire mesh on GPU.
        """

        # Convert query points to NumPy for CPU processing
        query_np = np.array(query_points)
        n_queries = query_np.shape[0]
        results = np.zeros((n_queries, 3), dtype=np.float32)

        # Process each query point (could be batched for efficiency)
        for i in range(n_queries):
            qp = query_np[i]

            # Find grid cell (CPU)
            cell_idx_float = (qp - grid_min) / cell_size
            cell_idx = np.floor(cell_idx_float).astype(np.int32)
            cell_idx = np.maximum(cell_idx, 0)
            cell_idx = np.minimum(cell_idx, grid_dims - 1)

            # Convert 3D to 1D index
            cell_flat = int(cell_idx[2] * grid_dims[1] * grid_dims[0] +
                          cell_idx[1] * grid_dims[0] + cell_idx[0])

            # Get candidate elements (CPU)
            candidates = cell_elements[cell_flat]
            n_candidates = cell_counts[cell_flat]

            # Interpolate using candidates (CPU-based for now, can optimize later)
            found = False
            best_value = np.zeros(3, dtype=np.float32)
            best_score = -9999.0

            for elem_idx in candidates[:n_candidates]:
                if elem_idx < 0:
                    break

                # Get element nodes
                node_indices = connectivity[elem_idx]
                tet_nodes = points[node_indices]

                # Check if point is inside (barycentric coords)
                is_inside, bary_coords = _point_in_tet_cpu(qp, tet_nodes)

                if is_inside:
                    # Interpolate
                    node_values = field_values[node_indices]
                    interpolated = np.dot(bary_coords, node_values)

                    # Score based on barycentric quality
                    bary_min = np.min(bary_coords)
                    bary_max = np.max(bary_coords)
                    score = -max(abs(bary_min), abs(bary_max - 1.0))

                    if score > best_score:
                        found = True
                        best_value = interpolated
                        best_score = score

            # Fallback: IDW from first element's nodes
            if not found and n_candidates > 0:
                first_elem = candidates[0]
                if first_elem >= 0:
                    node_indices = connectivity[first_elem]
                    node_positions = points[node_indices]
                    node_values = field_values[node_indices]

                    dists = np.sqrt(np.sum((node_positions - qp)**2, axis=1) + 1e-10)
                    weights = 1.0 / dists
                    weights = weights / np.sum(weights)

                    best_value = np.dot(weights, node_values)

            results[i] = best_value

        # Convert results back to JAX array
        return jnp.array(results)

    return grid_hash_interpolate_streaming


def _create_batched_gpu_interpolator(mesh: GridHashMesh, batch_size: int = 1000):
    """
    Create batched GPU interpolator: Pre-load mesh to GPU, process particles in batches.

    This provides a balance between:
    - CPU streaming (slow but low memory)
    - Full GPU non-batched (fast but may OOM on 18k particles at once)

    Strategy:
    1. Pre-convert mesh data to GPU ONCE (accept memory cost for mesh)
    2. Process particles in small batches (limit memory per batch)
    3. Use JIT-compiled GPU interpolation
    4. Accumulate results on CPU
    """

    # PRE-CONVERT MESH TO GPU ONCE
    # This is the memory-heavy part, but we only do it once per timestep
    points_jax = jnp.array(mesh.points)
    connectivity_jax = jnp.array(mesh.connectivity)
    field_values_jax = jnp.array(mesh.field_values)
    grid_min_jax = jnp.array(mesh.grid_min)
    grid_max_jax = jnp.array(mesh.grid_max)
    cell_size_jax = jnp.array(mesh.cell_size)
    grid_dims_jax = jnp.array(mesh.grid_dims)
    cell_elements_jax = jnp.array(mesh.cell_elements)
    cell_counts_jax = jnp.array(mesh.cell_counts)

    # Create JIT-compiled batch interpolation function
    @jax.jit
    def interpolate_batch_gpu(query_points_jax):
        """
        Interpolate a batch of points on GPU.
        Mesh is already on GPU (pre-converted above).
        """

        interpolate_single = lambda qp: interpolate_grid_hash(
            qp,
            points_jax,
            connectivity_jax,
            field_values_jax,
            grid_min_jax,
            grid_max_jax,
            cell_size_jax,
            grid_dims_jax,
            cell_elements_jax,
            cell_counts_jax
        )

        return jax.vmap(interpolate_single)(query_points_jax)

    def grid_hash_interpolate_batched(query_points: jnp.ndarray) -> jnp.ndarray:
        """Interpolate at query points using batched GPU processing."""

        # Convert to NumPy for batching logic
        query_np = np.array(query_points)
        n_queries = len(query_np)
        results = np.zeros((n_queries, 3), dtype=np.float32)

        # Process in batches (only particles, mesh already on GPU)
        n_batches = int(np.ceil(n_queries / batch_size))

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_queries)

            batch_queries = query_np[start_idx:end_idx]

            # Transfer only this batch of particles to GPU
            query_points_jax = jnp.array(batch_queries)

            # Interpolate on GPU (mesh already there)
            batch_results_jax = interpolate_batch_gpu(query_points_jax)

            # Transfer results back to CPU
            results[start_idx:end_idx] = np.array(batch_results_jax)

        # Convert final results to JAX array
        return jnp.array(results)

    return grid_hash_interpolate_batched


def _create_full_gpu_interpolator(mesh: GridHashMesh):
    """
    Legacy full GPU interpolator (high memory usage).

    Kept for backward compatibility and comparison.
    """

    # Convert mesh data to JAX arrays once (for this interpolator instance)
    # This happens on-demand and keeps arrays on GPU during use
    points_jax = jnp.array(mesh.points)
    connectivity_jax = jnp.array(mesh.connectivity)
    field_values_jax = jnp.array(mesh.field_values)
    grid_min_jax = jnp.array(mesh.grid_min)
    grid_max_jax = jnp.array(mesh.grid_max)
    cell_size_jax = jnp.array(mesh.cell_size)
    grid_dims_jax = jnp.array(mesh.grid_dims)
    cell_elements_jax = jnp.array(mesh.cell_elements)
    cell_counts_jax = jnp.array(mesh.cell_counts)

    @jax.jit
    def grid_hash_interpolate(query_points: jnp.ndarray) -> jnp.ndarray:
        """Interpolate at query points."""

        interpolate_single = lambda qp: interpolate_grid_hash(
            qp,
            points_jax,
            connectivity_jax,
            field_values_jax,
            grid_min_jax,
            grid_max_jax,
            cell_size_jax,
            grid_dims_jax,
            cell_elements_jax,
            cell_counts_jax
        )

        return jax.vmap(interpolate_single)(query_points)

    return grid_hash_interpolate


def _point_in_tet_cpu(point: np.ndarray, tet_nodes: np.ndarray) -> tuple:
    """
    Check if point is inside tetrahedron using barycentric coordinates (CPU version).

    Returns
    -------
    is_inside : bool
        True if point is inside tetrahedron
    bary_coords : np.ndarray
        Barycentric coordinates (4,)
    """

    v0, v1, v2, v3 = tet_nodes[0], tet_nodes[1], tet_nodes[2], tet_nodes[3]

    v0p = point - v0
    v01 = v1 - v0
    v02 = v2 - v0
    v03 = v3 - v0

    # Solve for barycentric coordinates using Cramer's rule
    # [v01 v02 v03] @ [λ1 λ2 λ3]^T = v0p

    mat = np.column_stack([v01, v02, v03])
    det = np.linalg.det(mat)

    if abs(det) < 1e-10:
        # Degenerate tetrahedron
        return False, np.zeros(4, dtype=np.float32)

    mat_inv = np.linalg.inv(mat)
    lambdas = mat_inv @ v0p

    lambda1, lambda2, lambda3 = lambdas
    lambda0 = 1.0 - lambda1 - lambda2 - lambda3

    # Check if inside (all barycentric coords >= 0, with small tolerance)
    tolerance = -1e-6
    is_inside = (lambda0 >= tolerance and lambda1 >= tolerance and
                lambda2 >= tolerance and lambda3 >= tolerance)

    bary_coords = np.array([lambda0, lambda1, lambda2, lambda3], dtype=np.float32)

    return is_inside, bary_coords
