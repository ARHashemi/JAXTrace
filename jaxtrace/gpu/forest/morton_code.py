"""
Morton Code (Z-Order Curve) Implementation.

Converts 3D spatial coordinates to 1D Morton codes for spatial indexing.
Morton codes preserve spatial locality - nearby points have nearby codes.

Used for hash octree construction and efficient spatial queries.
"""

import numpy as np
import jax.numpy as jnp
from typing import Union, Optional


def expand_bits(v: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
    """
    Expand 10-bit integer by inserting two zeros between each bit.

    This is the core operation for Morton code computation. For 3D coordinates,
    we interleave bits from x, y, z to create a single Morton code.

    Example:
        Input:  0b0000000001 (1)
        Output: 0b0000000000000000000001 (1)

        Input:  0b0000000010 (2)
        Output: 0b0000000000000000001000 (8)

    Args:
        v: 10-bit integer(s)

    Returns:
        30-bit integer(s) with expanded bits
    """
    # Check if using JAX or NumPy
    lib = jnp if isinstance(v, jnp.ndarray) else np

    v = v & 0x3FF  # Keep only 10 bits
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


def compact_bits(v: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
    """
    Compact 30-bit Morton code by extracting every 3rd bit.

    Inverse of expand_bits. Used for decoding Morton codes back to coordinates.

    Args:
        v: 30-bit Morton code component

    Returns:
        10-bit coordinate
    """
    lib = jnp if isinstance(v, jnp.ndarray) else np

    v = v & 0x09249249
    v = (v | (v >> 2)) & 0x030C30C3
    v = (v | (v >> 4)) & 0x0300F00F
    v = (v | (v >> 8)) & 0x030000FF
    v = (v | (v >> 16)) & 0x000003FF
    return v


def encode_morton_3d(
    x: Union[np.ndarray, jnp.ndarray],
    y: Union[np.ndarray, jnp.ndarray],
    z: Union[np.ndarray, jnp.ndarray]
) -> Union[np.ndarray, jnp.ndarray]:
    """
    Encode 3D coordinates to Morton code.

    Interleaves bits from x, y, z coordinates to create a single 30-bit Morton code
    that preserves spatial locality.

    Args:
        x: X coordinates (10-bit integers, range [0, 1023])
        y: Y coordinates (10-bit integers, range [0, 1023])
        z: Z coordinates (10-bit integers, range [0, 1023])

    Returns:
        Morton codes (30-bit integers)

    Example:
        >>> x, y, z = np.array([0, 1, 2]), np.array([0, 0, 0]), np.array([0, 0, 0])
        >>> morton = encode_morton_3d(x, y, z)
        >>> morton
        array([0, 1, 2], dtype=uint32)
    """
    lib = jnp if isinstance(x, jnp.ndarray) else np

    # Expand each coordinate
    xx = expand_bits(x)
    yy = expand_bits(y)
    zz = expand_bits(z)

    # Interleave: xxx...x yyy...y zzz...z → xyzxyzxyz...xyz
    morton = xx | (yy << 1) | (zz << 2)

    return morton.astype(lib.uint32 if lib == jnp else np.uint32)


def decode_morton_3d(
    morton: Union[np.ndarray, jnp.ndarray]
) -> tuple:
    """
    Decode Morton code back to 3D coordinates.

    Args:
        morton: Morton codes (30-bit integers)

    Returns:
        (x, y, z): Decoded coordinates (10-bit integers)

    Example:
        >>> morton = np.array([0, 1, 2, 7], dtype=np.uint32)
        >>> x, y, z = decode_morton_3d(morton)
        >>> x
        array([0, 1, 0, 1], dtype=uint32)
    """
    lib = jnp if isinstance(morton, jnp.ndarray) else np

    x = compact_bits(morton)
    y = compact_bits(morton >> 1)
    z = compact_bits(morton >> 2)

    return x.astype(lib.uint32 if lib == jnp else np.uint32), \
           y.astype(lib.uint32 if lib == jnp else np.uint32), \
           z.astype(lib.uint32 if lib == jnp else np.uint32)


def normalize_positions(
    positions: Union[np.ndarray, jnp.ndarray],
    bounds: Union[np.ndarray, jnp.ndarray],
    max_value: int = 1023
) -> Union[np.ndarray, jnp.ndarray]:
    """
    Normalize positions to integer coordinates [0, max_value].

    Maps continuous positions in bounding box to discrete grid coordinates.

    Args:
        positions: Positions [N, 3] in physical coordinates
        bounds: Bounding box [xmin, xmax, ymin, ymax, zmin, zmax]
        max_value: Maximum coordinate value (default 1023 for 10-bit)

    Returns:
        Normalized integer coordinates [N, 3] in range [0, max_value]

    Example:
        >>> positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        >>> bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        >>> normalized = normalize_positions(positions, bounds)
        >>> normalized
        array([[  0,   0,   0],
               [511, 511, 511]], dtype=uint32)
    """
    lib = jnp if isinstance(positions, jnp.ndarray) else np

    # Extract bounds
    xmin, xmax = bounds[0], bounds[1]
    ymin, ymax = bounds[2], bounds[3]
    zmin, zmax = bounds[4], bounds[5]

    # Normalize to [0, 1]
    x_norm = (positions[:, 0] - xmin) / (xmax - xmin)
    y_norm = (positions[:, 1] - ymin) / (ymax - ymin)
    z_norm = (positions[:, 2] - zmin) / (zmax - zmin)

    # Clamp to [0, 1]
    x_norm = lib.clip(x_norm, 0.0, 1.0)
    y_norm = lib.clip(y_norm, 0.0, 1.0)
    z_norm = lib.clip(z_norm, 0.0, 1.0)

    # Scale to [0, max_value] and convert to integer
    x_int = (x_norm * max_value).astype(lib.uint32 if lib == jnp else np.uint32)
    y_int = (y_norm * max_value).astype(lib.uint32 if lib == jnp else np.uint32)
    z_int = (z_norm * max_value).astype(lib.uint32 if lib == jnp else np.uint32)

    # Stack into [N, 3]
    normalized = lib.stack([x_int, y_int, z_int], axis=1)

    return normalized


def positions_to_morton(
    positions: Union[np.ndarray, jnp.ndarray],
    bounds: Union[np.ndarray, jnp.ndarray]
) -> Union[np.ndarray, jnp.ndarray]:
    """
    Convert 3D positions to Morton codes.

    Convenience function that combines normalization and encoding.

    Args:
        positions: Positions [N, 3] in physical coordinates
        bounds: Bounding box [xmin, xmax, ymin, ymax, zmin, zmax]

    Returns:
        Morton codes [N] (30-bit integers)

    Example:
        >>> positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        >>> bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        >>> morton = positions_to_morton(positions, bounds)
        >>> morton.shape
        (2,)
    """
    # Normalize to integer coordinates
    normalized = normalize_positions(positions, bounds)

    # Extract x, y, z
    x = normalized[:, 0]
    y = normalized[:, 1]
    z = normalized[:, 2]

    # Encode to Morton
    morton = encode_morton_3d(x, y, z)

    return morton


def sort_by_morton(
    positions: np.ndarray,
    bounds: np.ndarray,
    data: Optional[np.ndarray] = None
) -> tuple:
    """
    Sort positions by Morton code.

    Reorders positions (and optionally associated data) in Morton order for
    spatial locality.

    Args:
        positions: Positions [N, 3]
        bounds: Bounding box [xmin, xmax, ymin, ymax, zmin, zmax]
        data: Optional associated data [N, ...]

    Returns:
        (sorted_positions, sorted_data, morton_codes, sort_indices)

    Example:
        >>> positions = np.random.uniform(0, 1, (1000, 3))
        >>> bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        >>> sorted_pos, _, morton, _ = sort_by_morton(positions, bounds)
        >>> # sorted_pos is now in Morton order (spatially coherent)
    """
    # Compute Morton codes
    morton = positions_to_morton(positions, bounds)

    # Sort by Morton code
    sort_indices = np.argsort(morton)
    sorted_positions = positions[sort_indices]
    sorted_morton = morton[sort_indices]

    sorted_data = None
    if data is not None:
        sorted_data = data[sort_indices]

    return sorted_positions, sorted_data, sorted_morton, sort_indices


def compute_block_morton_range(
    block_bounds: np.ndarray,
    domain_bounds: np.ndarray
) -> tuple:
    """
    Compute Morton code range for a block.

    Useful for determining which Morton codes fall within a block's spatial extent.

    Args:
        block_bounds: Block bounding box [xmin, xmax, ymin, ymax, zmin, zmax]
        domain_bounds: Global domain bounds [xmin, xmax, ymin, ymax, zmin, zmax]

    Returns:
        (morton_min, morton_max): Morton code range for block

    Example:
        >>> block_bounds = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])
        >>> domain_bounds = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        >>> min_code, max_code = compute_block_morton_range(block_bounds, domain_bounds)
    """
    # Compute Morton codes for block corners
    corners = np.array([
        [block_bounds[0], block_bounds[2], block_bounds[4]],  # min corner
        [block_bounds[1], block_bounds[3], block_bounds[5]]   # max corner
    ])

    morton_codes = positions_to_morton(corners, domain_bounds)
    morton_min = np.min(morton_codes)
    morton_max = np.max(morton_codes)

    return int(morton_min), int(morton_max)
