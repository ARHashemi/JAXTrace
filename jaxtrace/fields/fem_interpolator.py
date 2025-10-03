"""
Finite Element Method (FEM) Interpolator for JAXTrace

High-performance JAX-based tetrahedral mesh interpolation using:
- Spatial hash grid for O(1) element lookup
- Barycentric coordinates for accurate interpolation
- Fully JIT-compiled for GPU acceleration
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class TetrahedralMesh:
    """Tetrahedral mesh data structure optimized for JAX."""

    # Mesh geometry
    points: jnp.ndarray          # (N, 3) node coordinates
    connectivity: jnp.ndarray     # (M, 4) tetrahedral connectivity

    # Spatial hash grid for fast lookup
    grid_cell_size: jnp.ndarray  # (3,) cell size per dimension
    grid_min: jnp.ndarray        # (3,) grid minimum
    grid_size: Tuple[int, int, int]  # Grid dimensions
    cell_to_elements: jnp.ndarray  # (grid_cells, max_elems_per_cell) element indices
    cell_elem_counts: jnp.ndarray  # (grid_cells,) number of elements per cell

    # Pre-computed for fast interpolation
    element_bounds: jnp.ndarray   # (M, 2, 3) min/max bounds per element


def build_tetrahedral_mesh(points: np.ndarray,
                           connectivity: np.ndarray,
                           grid_resolution: int = 32) -> TetrahedralMesh:
    """
    Build optimized tetrahedral mesh structure for fast interpolation.

    Parameters
    ----------
    points : np.ndarray
        Node coordinates (N, 3)
    connectivity : np.ndarray
        Tetrahedral connectivity (M, 4) - indices into points array
    grid_resolution : int
        Spatial hash grid resolution (cells per dimension)

    Returns
    -------
    TetrahedralMesh
        Optimized mesh structure for JAX interpolation
    """

    points = np.asarray(points, dtype=np.float32)
    connectivity = np.asarray(connectivity, dtype=np.int32)

    n_elements = connectivity.shape[0]

    # Compute bounding box
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = bbox_max - bbox_min

    # Spatial hash grid setup - use adaptive resolution for anisotropic domains
    # Target approximately grid_resolution cells per dimension, adjusted for aspect ratio
    total_cells_target = grid_resolution ** 3

    # Compute aspect ratios
    aspect_ratios = bbox_size / bbox_size.max()

    # Adjust grid size per dimension to maintain total cell count
    grid_size_float = aspect_ratios * (total_cells_target ** (1/3))
    grid_size = tuple(np.maximum(1, np.ceil(grid_size_float).astype(int)))

    # Compute cell size per dimension
    cell_size = bbox_size / np.array(grid_size)
    n_grid_cells = int(np.prod(grid_size))

    print(f"🔨 Building FEM interpolator:")
    print(f"   Mesh: {points.shape[0]} points, {n_elements} tetrahedra")
    print(f"   Grid: {grid_size} cells, size={cell_size}")

    # Compute element bounds and assign to grid cells
    print(f"   Building spatial hash grid...")
    element_bounds = np.zeros((n_elements, 2, 3), dtype=np.float32)
    element_to_cells = [[] for _ in range(n_grid_cells)]

    for elem_idx in range(n_elements):
        # Get element nodes
        node_indices = connectivity[elem_idx]
        elem_points = points[node_indices]

        # Compute bounds
        elem_min = elem_points.min(axis=0)
        elem_max = elem_points.max(axis=0)
        element_bounds[elem_idx, 0] = elem_min
        element_bounds[elem_idx, 1] = elem_max

        # Find overlapping grid cells (using per-dimension cell sizes)
        cell_min = np.floor((elem_min - bbox_min) / cell_size).astype(int)
        cell_max = np.floor((elem_max - bbox_min) / cell_size).astype(int)

        # Clamp to grid bounds
        grid_size_arr = np.array(grid_size)
        cell_min = np.clip(cell_min, 0, grid_size_arr - 1)
        cell_max = np.clip(cell_max, 0, grid_size_arr - 1)

        # Add element to all overlapping cells
        for i in range(cell_min[0], cell_max[0] + 1):
            for j in range(cell_min[1], cell_max[1] + 1):
                for k in range(cell_min[2], cell_max[2] + 1):
                    cell_idx = i * grid_size[1] * grid_size[2] + j * grid_size[2] + k
                    element_to_cells[cell_idx].append(elem_idx)

        if (elem_idx + 1) % 100000 == 0:
            print(f"      Processed {elem_idx + 1}/{n_elements} elements...")

    # Convert to padded array
    print(f"   Converting to padded array...")
    max_elems_per_cell = max(len(cells) for cells in element_to_cells)
    print(f"   Max elements per cell: {max_elems_per_cell}")

    cell_to_elements = np.full((n_grid_cells, max_elems_per_cell), -1, dtype=np.int32)
    cell_elem_counts = np.zeros(n_grid_cells, dtype=np.int32)

    for cell_idx, elem_list in enumerate(element_to_cells):
        n_elems = len(elem_list)
        cell_elem_counts[cell_idx] = n_elems
        if n_elems > 0:
            cell_to_elements[cell_idx, :n_elems] = elem_list

    # Convert to JAX arrays
    return TetrahedralMesh(
        points=jnp.array(points),
        connectivity=jnp.array(connectivity),
        grid_cell_size=jnp.array(cell_size, dtype=jnp.float32),
        grid_min=jnp.array(bbox_min, dtype=jnp.float32),
        grid_size=grid_size,
        cell_to_elements=jnp.array(cell_to_elements),
        cell_elem_counts=jnp.array(cell_elem_counts),
        element_bounds=jnp.array(element_bounds)
    )


@jax.jit
def point_in_tetrahedron(point: jnp.ndarray,
                         tet_nodes: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Check if point is inside tetrahedron and compute barycentric coordinates.

    Uses barycentric coordinate method - fastest and most numerically stable.

    Parameters
    ----------
    point : jnp.ndarray
        Query point (3,)
    tet_nodes : jnp.ndarray
        Tetrahedron node coordinates (4, 3)

    Returns
    -------
    is_inside : jnp.ndarray
        Boolean scalar, True if point is inside
    bary_coords : jnp.ndarray
        Barycentric coordinates (4,)
    """

    # Extract vertices
    v0, v1, v2, v3 = tet_nodes[0], tet_nodes[1], tet_nodes[2], tet_nodes[3]

    # Compute vectors
    v0p = point - v0
    v01 = v1 - v0
    v02 = v2 - v0
    v03 = v3 - v0

    # Compute 3x3 matrix determinant (tetrahedron volume × 6)
    # Using scalar triple product
    mat = jnp.stack([v01, v02, v03], axis=1)  # (3, 3)
    det = jnp.linalg.det(mat)

    # Avoid division by zero for degenerate tets
    det = jnp.where(jnp.abs(det) < 1e-10, 1.0, det)

    # Solve for barycentric coordinates using Cramer's rule
    # b1 = det([v0p, v02, v03]) / det
    # b2 = det([v01, v0p, v03]) / det
    # b3 = det([v01, v02, v0p]) / det
    # b0 = 1 - b1 - b2 - b3

    mat1 = jnp.stack([v0p, v02, v03], axis=1)
    mat2 = jnp.stack([v01, v0p, v03], axis=1)
    mat3 = jnp.stack([v01, v02, v0p], axis=1)

    b1 = jnp.linalg.det(mat1) / det
    b2 = jnp.linalg.det(mat2) / det
    b3 = jnp.linalg.det(mat3) / det
    b0 = 1.0 - b1 - b2 - b3

    bary_coords = jnp.array([b0, b1, b2, b3])

    # Point is inside if all barycentric coordinates are in [0, 1]
    # Use small tolerance for numerical stability
    tol = -1e-6
    is_inside = jnp.all(bary_coords >= tol) & jnp.all(bary_coords <= 1.0 + tol)

    return is_inside, bary_coords


@jax.jit
def interpolate_in_mesh(query_point: jnp.ndarray,
                       mesh_points: jnp.ndarray,
                       connectivity: jnp.ndarray,
                       field_values: jnp.ndarray,
                       grid_min: jnp.ndarray,
                       grid_cell_size: jnp.ndarray,
                       grid_size: Tuple[int, int, int],
                       cell_to_elements: jnp.ndarray,
                       cell_elem_counts: jnp.ndarray,
                       element_bounds: jnp.ndarray) -> jnp.ndarray:
    """
    Interpolate field value at query point using FEM interpolation.

    Fully JIT-compiled for GPU execution.

    Parameters
    ----------
    query_point : jnp.ndarray
        Query position (3,)
    mesh_points : jnp.ndarray
        All mesh node coordinates (N, 3)
    connectivity : jnp.ndarray
        Tetrahedral connectivity (M, 4)
    field_values : jnp.ndarray
        Field values at mesh nodes (N, 3) for velocity field
    grid_min : jnp.ndarray
        Grid minimum (3,)
    grid_cell_size : jnp.ndarray
        Grid cell size per dimension (3,)
    grid_size : Tuple[int, int, int]
        Grid dimensions
    cell_to_elements : jnp.ndarray
        Grid cell to element mapping
    cell_elem_counts : jnp.ndarray
        Number of elements per grid cell
    element_bounds : jnp.ndarray
        Element bounding boxes

    Returns
    -------
    jnp.ndarray
        Interpolated field value (3,) for velocity
    """

    # Find grid cell containing query point
    cell_coords = jnp.floor((query_point - grid_min) / grid_cell_size).astype(jnp.int32)
    cell_coords = jnp.clip(cell_coords, 0, jnp.array(grid_size) - 1)

    # Convert 3D cell coords to 1D index
    cell_idx = (cell_coords[0] * grid_size[1] * grid_size[2] +
                cell_coords[1] * grid_size[2] +
                cell_coords[2])

    # Get candidate elements in this cell
    n_candidates = cell_elem_counts[cell_idx]
    candidate_elements = cell_to_elements[cell_idx]

    # Search for containing element using vectorized operations
    def check_element(elem_idx):
        """Check if point is in element and return interpolated value."""
        # Skip invalid elements (-1 padding)
        is_valid = elem_idx >= 0

        # Get element bounds for quick rejection
        elem_min = element_bounds[elem_idx, 0]
        elem_max = element_bounds[elem_idx, 1]
        in_bounds = jnp.all(query_point >= elem_min) & jnp.all(query_point <= elem_max)

        # Get element nodes
        node_indices = connectivity[elem_idx]
        tet_nodes = mesh_points[node_indices]

        # Check if inside and get barycentric coords
        is_inside, bary_coords = point_in_tetrahedron(query_point, tet_nodes)

        # Interpolate field value using barycentric coordinates
        node_values = field_values[node_indices]  # (4, 3)
        interpolated = jnp.dot(bary_coords, node_values)  # (3,)

        # Return value if inside, otherwise zeros
        found = is_valid & in_bounds & is_inside
        return found, interpolated

    # Check all candidate elements (using scan for efficiency)
    def scan_fn(carry, elem_idx):
        found_prev, value_prev = carry
        found_curr, value_curr = check_element(elem_idx)

        # Keep first found value
        found = found_prev | found_curr
        value = jnp.where(found_prev, value_prev, value_curr)

        return (found, value), None

    # Initialize with not found
    init_carry = (jnp.array(False), jnp.zeros(3, dtype=jnp.float32))

    # Scan through ALL candidates (padded array)
    # Use mask to skip invalid elements (-1 padding already handled in check_element)
    (found, interpolated_value), _ = jax.lax.scan(
        scan_fn,
        init_carry,
        candidate_elements  # Scan all elements, check_element handles -1 padding
    )

    # If not found, fall back to nearest neighbor
    # (This handles edge cases and points outside mesh)
    def nearest_neighbor_fallback():
        distances = jnp.sum((mesh_points - query_point)**2, axis=1)
        nearest_idx = jnp.argmin(distances)
        return field_values[nearest_idx]

    result = jnp.where(found, interpolated_value, nearest_neighbor_fallback())

    return result


def create_fem_interpolator(mesh: TetrahedralMesh):
    """
    Create a JIT-compiled FEM interpolator function.

    Parameters
    ----------
    mesh : TetrahedralMesh
        Pre-built tetrahedral mesh structure

    Returns
    -------
    callable
        JIT-compiled interpolator function(query_points, field_values) -> interpolated_values
    """

    @jax.jit
    def fem_interpolate(query_points: jnp.ndarray,
                       field_values: jnp.ndarray) -> jnp.ndarray:
        """
        Interpolate field values at multiple query points.

        Parameters
        ----------
        query_points : jnp.ndarray
            Query positions (M, 3)
        field_values : jnp.ndarray
            Field values at mesh nodes (N, 3)

        Returns
        -------
        jnp.ndarray
            Interpolated values (M, 3)
        """

        # Vectorize over query points
        interpolate_single = lambda qp: interpolate_in_mesh(
            qp,
            mesh.points,
            mesh.connectivity,
            field_values,
            mesh.grid_min,
            mesh.grid_cell_size,
            mesh.grid_size,
            mesh.cell_to_elements,
            mesh.cell_elem_counts,
            mesh.element_bounds
        )

        return jax.vmap(interpolate_single)(query_points)

    return fem_interpolate


# Example usage
if __name__ == "__main__":
    # Test with simple mesh
    print("Testing FEM interpolator...")

    # Create simple tet mesh (two tets forming a cube)
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],  # Tet 1
        [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]   # Tet 2
    ], dtype=np.float32)

    connectivity = np.array([
        [0, 1, 2, 3],  # Tet 1
        [4, 5, 6, 7]   # Tet 2
    ], dtype=np.int32)

    # Build mesh
    mesh = build_tetrahedral_mesh(points, connectivity, grid_resolution=4)

    # Create interpolator
    interpolator = create_fem_interpolator(mesh)

    # Test interpolation
    field_values = jnp.array(points, dtype=jnp.float32)  # Use positions as field for testing
    query_points = jnp.array([[0.5, 0.5, 0.5]], dtype=jnp.float32)

    result = interpolator(query_points, field_values)
    print(f"Query point: {query_points[0]}")
    print(f"Interpolated value: {result[0]}")
    print(f"✅ FEM interpolator test complete!")
