"""
Forest Block Builder.

Creates regular forest grid decomposition for spatial partitioning.
Implements block metadata structures and neighbor topology computation.
"""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np


@dataclass
class BlockMetadata:
    """
    Metadata for a single forest block.

    Each block represents a spatial partition of the domain and serves as
    the root of an independent octree for element search.

    Attributes:
        block_id: Unique block identifier (0 to n_blocks-1)
        bounds: Bounding box [xmin, xmax, ymin, ymax, zmin, zmax]
        center: Block center point [x, y, z]
        grid_index: 3D grid index (i, j, k) in forest grid
        neighbors: Neighbor block IDs for 6 faces [+x, -x, +y, -y, +z, -z]
                  -1 indicates domain boundary
    """
    block_id: int
    bounds: np.ndarray  # [6] float
    center: np.ndarray  # [3] float
    grid_index: Tuple[int, int, int]
    neighbors: np.ndarray  # [6] int, -1 for boundary

    def __post_init__(self):
        """Validate block metadata."""
        assert self.bounds.shape == (6,), f"bounds must be [6], got {self.bounds.shape}"
        assert self.center.shape == (3,), f"center must be [3], got {self.center.shape}"
        assert self.neighbors.shape == (6,), f"neighbors must be [6], got {self.neighbors.shape}"

        # Validate bounds ordering
        assert self.bounds[0] < self.bounds[1], "xmin must be < xmax"
        assert self.bounds[2] < self.bounds[3], "ymin must be < ymax"
        assert self.bounds[4] < self.bounds[5], "zmin must be < zmax"

    @property
    def volume(self) -> float:
        """Block volume."""
        dx = self.bounds[1] - self.bounds[0]
        dy = self.bounds[3] - self.bounds[2]
        dz = self.bounds[5] - self.bounds[4]
        return dx * dy * dz

    @property
    def size(self) -> np.ndarray:
        """Block size [dx, dy, dz]."""
        return np.array([
            self.bounds[1] - self.bounds[0],
            self.bounds[3] - self.bounds[2],
            self.bounds[5] - self.bounds[4]
        ])

    def contains_point(self, point: np.ndarray, tolerance: float = 0.0) -> bool:
        """
        Check if point is inside block (with optional tolerance).

        Args:
            point: 3D point [x, y, z]
            tolerance: Tolerance for boundary points (default 0)

        Returns:
            True if point is inside block
        """
        return (
            self.bounds[0] - tolerance <= point[0] <= self.bounds[1] + tolerance and
            self.bounds[2] - tolerance <= point[1] <= self.bounds[3] + tolerance and
            self.bounds[4] - tolerance <= point[2] <= self.bounds[5] + tolerance
        )


def create_regular_forest_grid(
    domain_bounds: np.ndarray,
    grid_size: Tuple[int, int, int]
) -> List[BlockMetadata]:
    """
    Create regular forest grid partitioning.

    Decomposes domain into a regular grid of blocks. Each block has simple
    6-face connectivity (no diagonal neighbors). This provides predictable
    load balancing and simple neighbor topology for GPU kernels.

    Args:
        domain_bounds: Global domain bounds [xmin, xmax, ymin, ymax, zmin, zmax]
        grid_size: Grid dimensions (nx, ny, nz)

    Returns:
        List of BlockMetadata, ordered by block_id = i + j*nx + k*nx*ny

    Example:
        >>> bounds = np.array([-0.03, 0.03, -0.023, 0.023, -0.01, 0.0])
        >>> blocks = create_regular_forest_grid(bounds, (4, 4, 2))
        >>> len(blocks)
        32
        >>> blocks[0].grid_index
        (0, 0, 0)
        >>> blocks[0].neighbors[0]  # +x neighbor
        1
    """
    nx, ny, nz = grid_size
    n_blocks = nx * ny * nz

    # Domain size
    dx_total = domain_bounds[1] - domain_bounds[0]
    dy_total = domain_bounds[3] - domain_bounds[2]
    dz_total = domain_bounds[5] - domain_bounds[4]

    # Block size
    dx_block = dx_total / nx
    dy_block = dy_total / ny
    dz_block = dz_total / nz

    blocks = []

    for block_id in range(n_blocks):
        # Compute 3D grid index from block_id
        i = block_id % nx
        j = (block_id // nx) % ny
        k = block_id // (nx * ny)

        # Block bounds
        xmin = domain_bounds[0] + i * dx_block
        xmax = xmin + dx_block
        ymin = domain_bounds[2] + j * dy_block
        ymax = ymin + dy_block
        zmin = domain_bounds[4] + k * dz_block
        zmax = zmin + dz_block

        bounds = np.array([xmin, xmax, ymin, ymax, zmin, zmax], dtype=np.float32)

        # Block center
        center = np.array([
            (xmin + xmax) / 2,
            (ymin + ymax) / 2,
            (zmin + zmax) / 2
        ], dtype=np.float32)

        # Compute neighbors (6-face connectivity)
        neighbors = compute_block_neighbors(i, j, k, nx, ny, nz)

        block = BlockMetadata(
            block_id=block_id,
            bounds=bounds,
            center=center,
            grid_index=(i, j, k),
            neighbors=neighbors
        )

        blocks.append(block)

    return blocks


def compute_block_neighbors(
    i: int, j: int, k: int,
    nx: int, ny: int, nz: int
) -> np.ndarray:
    """
    Compute 6-face neighbor block IDs.

    Neighbor order: [+x, -x, +y, -y, +z, -z]
    Returns -1 for domain boundaries.

    Args:
        i, j, k: Block grid indices
        nx, ny, nz: Grid dimensions

    Returns:
        neighbors: [6] array of neighbor block IDs (-1 for boundary)
    """
    neighbors = np.full(6, -1, dtype=np.int32)

    # +x neighbor (i+1, j, k)
    if i + 1 < nx:
        neighbors[0] = (i + 1) + j * nx + k * nx * ny

    # -x neighbor (i-1, j, k)
    if i - 1 >= 0:
        neighbors[1] = (i - 1) + j * nx + k * nx * ny

    # +y neighbor (i, j+1, k)
    if j + 1 < ny:
        neighbors[2] = i + (j + 1) * nx + k * nx * ny

    # -y neighbor (i, j-1, k)
    if j - 1 >= 0:
        neighbors[3] = i + (j - 1) * nx + k * nx * ny

    # +z neighbor (i, j, k+1)
    if k + 1 < nz:
        neighbors[4] = i + j * nx + (k + 1) * nx * ny

    # -z neighbor (i, j, k-1)
    if k - 1 >= 0:
        neighbors[5] = i + j * nx + (k - 1) * nx * ny

    return neighbors


def find_block_containing_point(
    point: np.ndarray,
    blocks: List[BlockMetadata],
    tolerance: float = 1e-6
) -> int:
    """
    Find which block contains a point.

    Uses simple linear search (fast for small block counts like 32).
    For large block counts (>1000), consider spatial hashing.

    Args:
        point: 3D point [x, y, z]
        blocks: List of block metadata
        tolerance: Tolerance for boundary points

    Returns:
        block_id: ID of containing block, or -1 if outside all blocks

    Example:
        >>> blocks = create_regular_forest_grid(bounds, (4, 4, 2))
        >>> point = np.array([0.0, 0.0, -0.005])
        >>> block_id = find_block_containing_point(point, blocks)
        >>> block_id >= 0
        True
    """
    for block in blocks:
        if block.contains_point(point, tolerance):
            return block.block_id

    return -1


def position_to_block_id(
    position: np.ndarray,
    domain_bounds: np.ndarray,
    grid_size: Tuple[int, int, int]
) -> int:
    """
    Fast position → block_id mapping without block metadata.

    Directly computes block_id from position for regular grids.
    Much faster than linear search for large block counts.

    Args:
        position: 3D point [x, y, z]
        domain_bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
        grid_size: (nx, ny, nz)

    Returns:
        block_id: Block ID (0 to n_blocks-1), or -1 if outside domain
    """
    nx, ny, nz = grid_size

    # Check if inside domain
    if not (domain_bounds[0] <= position[0] <= domain_bounds[1] and
            domain_bounds[2] <= position[1] <= domain_bounds[3] and
            domain_bounds[4] <= position[2] <= domain_bounds[5]):
        return -1

    # Compute grid indices
    dx = (domain_bounds[1] - domain_bounds[0]) / nx
    dy = (domain_bounds[3] - domain_bounds[2]) / ny
    dz = (domain_bounds[5] - domain_bounds[4]) / nz

    i = int((position[0] - domain_bounds[0]) / dx)
    j = int((position[1] - domain_bounds[2]) / dy)
    k = int((position[2] - domain_bounds[4]) / dz)

    # Clamp to grid
    i = np.clip(i, 0, nx - 1)
    j = np.clip(j, 0, ny - 1)
    k = np.clip(k, 0, nz - 1)

    # Compute block_id
    block_id = i + j * nx + k * nx * ny

    return int(block_id)
