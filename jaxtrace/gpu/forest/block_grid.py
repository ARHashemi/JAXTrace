"""
Forest Block Grid Generator

Implements regular grid partitioning of spatial domain into forest blocks.
Each block is a root of an independent sub-octree for block-local search.

Part of Phase 1: Forest Structure & Block Partitioning
"""

from dataclasses import dataclass
import numpy as np
from typing import Tuple, List


@dataclass
class Block:
    """
    Represents a single block in the forest grid.

    Attributes:
        block_id: Unique identifier (0 to n_blocks-1)
        bounds: [xmin, xmax, ymin, ymax, zmin, zmax] spatial extent
        center: [x, y, z] block center
        grid_index: (i, j, k) position in regular grid
        neighbors_6: [+x, -x, +y, -y, +z, -z] face neighbors, -1 = boundary
        neighbors_26: All 26 neighbors (6 faces + 12 edges + 8 corners), -1 padded
    """
    block_id: int
    bounds: np.ndarray  # (6,) float32
    center: np.ndarray  # (3,) float32
    grid_index: Tuple[int, int, int]
    neighbors_6: np.ndarray  # (6,) int32
    neighbors_26: np.ndarray  # (26,) int32

    @property
    def volume(self) -> float:
        """Compute block volume."""
        dx = self.bounds[1] - self.bounds[0]
        dy = self.bounds[3] - self.bounds[2]
        dz = self.bounds[5] - self.bounds[4]
        return float(dx * dy * dz)

    @property
    def size(self) -> np.ndarray:
        """Block size in each dimension [dx, dy, dz]."""
        return np.array([
            self.bounds[1] - self.bounds[0],
            self.bounds[3] - self.bounds[2],
            self.bounds[5] - self.bounds[4]
        ], dtype=np.float32)

    def contains_point(self, point: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if point is inside block (with tolerance for boundaries).

        Args:
            point: (3,) array [x, y, z]
            tolerance: Boundary tolerance

        Returns:
            True if point inside block (inclusive boundaries)
        """
        return (
            self.bounds[0] - tolerance <= point[0] <= self.bounds[1] + tolerance and
            self.bounds[2] - tolerance <= point[1] <= self.bounds[3] + tolerance and
            self.bounds[4] - tolerance <= point[2] <= self.bounds[5] + tolerance
        )


def create_regular_grid(
    domain_bounds: np.ndarray,  # [xmin, xmax, ymin, ymax, zmin, zmax]
    grid_size: Tuple[int, int, int]  # (nx, ny, nz)
) -> List[Block]:
    """
    Create uniform grid of blocks dividing the domain.

    Algorithm:
        1. Divide domain uniformly: dx = (xmax - xmin) / nx
        2. For each grid cell (i, j, k):
             - Compute bounds
             - Compute center
             - Assign block_id = i + j*nx + k*nx*ny (Z-order)
        3. Compute 6-face neighbors (simple grid arithmetic)
        4. Compute 26-neighbors (all adjacent cells)

    Args:
        domain_bounds: Global domain [xmin, xmax, ymin, ymax, zmin, zmax]
        grid_size: Number of blocks in each dimension (nx, ny, nz)

    Returns:
        List of Block objects, ordered by block_id

    Performance: O(n_blocks), trivial for 32 blocks

    Example:
        >>> bounds = np.array([-0.03, 0.03, -0.023, 0.023, -0.01, 0.0])
        >>> blocks = create_regular_grid(bounds, (4, 4, 2))
        >>> len(blocks)
        32
        >>> blocks[0].grid_index
        (0, 0, 0)
    """
    nx, ny, nz = grid_size
    n_blocks = nx * ny * nz

    # Compute block sizes
    dx = (domain_bounds[1] - domain_bounds[0]) / nx
    dy = (domain_bounds[3] - domain_bounds[2]) / ny
    dz = (domain_bounds[5] - domain_bounds[4]) / nz

    blocks = []

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                block_id = i + j * nx + k * nx * ny

                # Compute bounds
                bounds = np.array([
                    domain_bounds[0] + i * dx,
                    domain_bounds[0] + (i + 1) * dx,
                    domain_bounds[2] + j * dy,
                    domain_bounds[2] + (j + 1) * dy,
                    domain_bounds[4] + k * dz,
                    domain_bounds[4] + (k + 1) * dz
                ], dtype=np.float32)

                # Compute center
                center = np.array([
                    (bounds[0] + bounds[1]) / 2,
                    (bounds[2] + bounds[3]) / 2,
                    (bounds[4] + bounds[5]) / 2
                ], dtype=np.float32)

                # Compute neighbors
                neighbors_6 = compute_6_neighbors(i, j, k, nx, ny, nz)
                neighbors_26 = compute_26_neighbors(i, j, k, nx, ny, nz)

                blocks.append(Block(
                    block_id=block_id,
                    bounds=bounds,
                    center=center,
                    grid_index=(i, j, k),
                    neighbors_6=neighbors_6,
                    neighbors_26=neighbors_26
                ))

    return blocks


def compute_6_neighbors(
    i: int, j: int, k: int,
    nx: int, ny: int, nz: int
) -> np.ndarray:
    """
    Compute 6 face neighbors for block (i, j, k).

    Returns:
        [+x, -x, +y, -y, +z, -z] neighbor block IDs, -1 for boundaries

    Example:
        >>> compute_6_neighbors(0, 0, 0, 2, 2, 2)  # Corner block
        array([ 1, -1,  2, -1,  4, -1], dtype=int32)
    """
    neighbors = np.full(6, -1, dtype=np.int32)

    # +x direction (i+1)
    if i + 1 < nx:
        neighbors[0] = (i + 1) + j * nx + k * nx * ny
    # -x direction (i-1)
    if i - 1 >= 0:
        neighbors[1] = (i - 1) + j * nx + k * nx * ny
    # +y direction (j+1)
    if j + 1 < ny:
        neighbors[2] = i + (j + 1) * nx + k * nx * ny
    # -y direction (j-1)
    if j - 1 >= 0:
        neighbors[3] = i + (j - 1) * nx + k * nx * ny
    # +z direction (k+1)
    if k + 1 < nz:
        neighbors[4] = i + j * nx + (k + 1) * nx * ny
    # -z direction (k-1)
    if k - 1 >= 0:
        neighbors[5] = i + j * nx + (k - 1) * nx * ny

    return neighbors


def compute_26_neighbors(
    i: int, j: int, k: int,
    nx: int, ny: int, nz: int
) -> np.ndarray:
    """
    Compute all 26 neighbors (6 faces + 12 edges + 8 corners).

    Returns:
        (26,) int32 array, -1 padded for boundaries

    Ordering:
        Iterate over all 27 cells (including self), exclude self
        Order: dk=-1,0,+1; dj=-1,0,+1; di=-1,0,+1 (excluding di=dj=dk=0)

    Example:
        >>> compute_26_neighbors(1, 1, 1, 3, 3, 3)  # Interior block
        array([...], dtype=int32)  # All 26 neighbors valid

        >>> compute_26_neighbors(0, 0, 0, 2, 2, 2)  # Corner block
        array([...], dtype=int32)  # Only 7 neighbors valid, rest -1
    """
    neighbors = []

    # Iterate over all 27 cells (including self)
    for dk in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for di in [-1, 0, 1]:
                # Skip self
                if di == 0 and dj == 0 and dk == 0:
                    continue

                ni, nj, nk = i + di, j + dj, k + dk

                # Check bounds
                if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                    neighbor_id = ni + nj * nx + nk * nx * ny
                    neighbors.append(neighbor_id)
                else:
                    neighbors.append(-1)

    return np.array(neighbors, dtype=np.int32)


def position_to_block_id(
    position: np.ndarray,  # (3,) [x, y, z]
    domain_bounds: np.ndarray,  # [xmin, xmax, ymin, ymax, zmin, zmax]
    grid_size: Tuple[int, int, int]  # (nx, ny, nz)
) -> int:
    """
    Fast O(1) mapping from position to block ID.

    Algorithm:
        1. Compute grid index: (i, j, k) = floor((pos - min) / block_size)
        2. Convert to block_id = i + j*nx + k*nx*ny
        3. Clamp to grid bounds (handle boundaries)

    Args:
        position: Point coordinates [x, y, z]
        domain_bounds: Global domain bounds
        grid_size: Grid dimensions (nx, ny, nz)

    Returns:
        block_id (int) or -1 if outside domain

    Performance: O(1) - constant time

    Example:
        >>> bounds = np.array([0, 2, 0, 2, 0, 2])
        >>> pos = np.array([0.5, 0.5, 0.5])
        >>> position_to_block_id(pos, bounds, (2, 2, 2))
        0
        >>> pos = np.array([1.5, 1.5, 1.5])
        >>> position_to_block_id(pos, bounds, (2, 2, 2))
        7
    """
    nx, ny, nz = grid_size

    # Check if inside domain (with small tolerance)
    tol = 1e-10
    if not (domain_bounds[0] - tol <= position[0] <= domain_bounds[1] + tol and
            domain_bounds[2] - tol <= position[1] <= domain_bounds[3] + tol and
            domain_bounds[4] - tol <= position[2] <= domain_bounds[5] + tol):
        return -1

    # Compute block sizes
    dx = (domain_bounds[1] - domain_bounds[0]) / nx
    dy = (domain_bounds[3] - domain_bounds[2]) / ny
    dz = (domain_bounds[5] - domain_bounds[4]) / nz

    # Compute grid indices
    i = int((position[0] - domain_bounds[0]) / dx)
    j = int((position[1] - domain_bounds[2]) / dy)
    k = int((position[2] - domain_bounds[4]) / dz)

    # Clamp to grid (handle boundary cases where point is exactly on max bound)
    i = max(0, min(i, nx - 1))
    j = max(0, min(j, ny - 1))
    k = max(0, min(k, nz - 1))

    return i + j * nx + k * nx * ny


def find_block_containing_point(
    point: np.ndarray,  # (3,)
    blocks: List[Block]
) -> int:
    """
    Find block containing point using linear search.

    Use position_to_block_id() for O(1) lookup instead if domain bounds
    and grid size are known.

    Args:
        point: Coordinates [x, y, z]
        blocks: List of all blocks

    Returns:
        block_id or -1 if not found

    Performance: O(n_blocks) - use only for validation/testing
    """
    for block in blocks:
        if block.contains_point(point):
            return block.block_id
    return -1


def infer_grid_size(blocks: List[Block]) -> Tuple[int, int, int]:
    """
    Infer grid size from list of blocks.

    Assumes blocks are from a regular grid.

    Args:
        blocks: List of blocks

    Returns:
        (nx, ny, nz) grid dimensions

    Example:
        >>> blocks = create_regular_grid(np.array([0,1,0,1,0,1]), (2,2,2))
        >>> infer_grid_size(blocks)
        (2, 2, 2)
    """
    # Find maximum grid indices
    max_i = max(b.grid_index[0] for b in blocks)
    max_j = max(b.grid_index[1] for b in blocks)
    max_k = max(b.grid_index[2] for b in blocks)

    return (max_i + 1, max_j + 1, max_k + 1)
