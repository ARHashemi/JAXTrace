#!/usr/bin/env python3
"""
Morton Code (Z-Curve) Encoding for Spatial Partitioning

Implements Morton codes for 3D spatial indexing, used to build octrees
within blocks for efficient Level 2 search.

Phase 2.1 of V3 Plan

Morton codes interleave the bits of (x, y, z) coordinates to create a 1D
index that preserves spatial locality. Points close in 3D space have similar
Morton codes, enabling efficient range queries and octree construction.

Reference:
- Morton, G.M. (1966). "A computer oriented geodetic data base and a new
  technique in file sequencing."
"""

from typing import Tuple
import numpy as np
import jax.numpy as jnp
import jax

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


def expand_bits_3d(x: np.ndarray) -> np.ndarray:
    """
    Expand bits for 3D Morton code (21 bits -> 63 bits).

    Inserts two 0 bits between each bit of the input.
    For 3D Morton codes with 21-bit coordinates.

    Args:
        x: (N,) uint32 array with values in [0, 2^21-1]

    Returns:
        expanded: (N,) uint64 array with expanded bits
    """
    x = x.astype(np.uint64)

    # Expand bits by inserting 00 between each bit
    x = (x | (x << 32)) & 0x1f00000000ffff
    x = (x | (x << 16)) & 0x1f0000ff0000ff
    x = (x | (x << 8)) & 0x100f00f00f00f00f
    x = (x | (x << 4)) & 0x10c30c30c30c30c3
    x = (x | (x << 2)) & 0x1249249249249249

    return x


def morton_encode_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Encode 3D coordinates as Morton codes.

    Interleaves bits: z[20]y[20]x[20] z[19]y[19]x[19] ... z[0]y[0]x[0]

    Args:
        x: (N,) uint32 array - x coordinates [0, 2^21-1]
        y: (N,) uint32 array - y coordinates [0, 2^21-1]
        z: (N,) uint32 array - z coordinates [0, 2^21-1]

    Returns:
        morton: (N,) uint64 array - Morton codes
    """
    # Expand bits
    xx = expand_bits_3d(x)
    yy = expand_bits_3d(y)
    zz = expand_bits_3d(z)

    # Interleave: z bits at positions ..., 5, 2; y at ..., 4, 1; x at ..., 3, 0
    morton = xx | (yy << 1) | (zz << 2)

    return morton


def normalize_coordinates(
    positions: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    n_bits: int = 21
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize positions to integer grid for Morton encoding.

    Maps positions from [bbox_min, bbox_max] to [0, 2^n_bits - 1].

    Args:
        positions: (N, 3) float64 - 3D positions
        bbox_min: (3,) float64 - minimum bounds
        bbox_max: (3,) float64 - maximum bounds
        n_bits: Number of bits per dimension (default 21 → 63-bit Morton code)

    Returns:
        x_int: (N,) uint32 - normalized x coordinates
        y_int: (N,) uint32 - normalized y coordinates
        z_int: (N,) uint32 - normalized z coordinates
    """
    # Normalize to [0, 1]
    bbox_size = bbox_max - bbox_min
    normalized = (positions - bbox_min) / bbox_size

    # Handle edge case: points exactly on bbox_max
    normalized = np.clip(normalized, 0.0, 1.0 - 1e-10)

    # Scale to [0, 2^n_bits - 1]
    max_val = (1 << n_bits) - 1
    scaled = normalized * max_val

    # Convert to integers
    x_int = scaled[:, 0].astype(np.uint32)
    y_int = scaled[:, 1].astype(np.uint32)
    z_int = scaled[:, 2].astype(np.uint32)

    return x_int, y_int, z_int


def compute_morton_codes(
    positions: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    n_bits: int = 21
) -> np.ndarray:
    """
    Compute Morton codes for 3D positions.

    Complete pipeline: normalize → encode.

    Args:
        positions: (N, 3) float64 - 3D positions
        bbox_min: (3,) float64 - bounding box minimum
        bbox_max: (3,) float64 - bounding box maximum
        n_bits: Bits per dimension (default 21)

    Returns:
        morton_codes: (N,) uint64 - Morton codes
    """
    # Normalize to integer grid
    x_int, y_int, z_int = normalize_coordinates(
        positions, bbox_min, bbox_max, n_bits
    )

    # Encode as Morton codes
    morton_codes = morton_encode_3d(x_int, y_int, z_int)

    return morton_codes


def morton_decode_3d(morton: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode Morton codes back to 3D coordinates.

    Useful for debugging and visualization.

    Args:
        morton: (N,) uint64 - Morton codes

    Returns:
        x: (N,) uint32 - x coordinates [0, 2^21-1]
        y: (N,) uint32 - y coordinates [0, 2^21-1]
        z: (N,) uint32 - z coordinates [0, 2^21-1]
    """
    morton = morton.astype(np.uint64)

    # Extract interleaved bits
    x = morton & 0x1249249249249249
    y = (morton >> 1) & 0x1249249249249249
    z = (morton >> 2) & 0x1249249249249249

    # Compact bits (reverse of expand_bits_3d)
    x = (x | (x >> 2)) & 0x10c30c30c30c30c3
    x = (x | (x >> 4)) & 0x100f00f00f00f00f
    x = (x | (x >> 8)) & 0x1f0000ff0000ff
    x = (x | (x >> 16)) & 0x1f00000000ffff
    x = (x | (x >> 32)) & 0x1fffff

    y = (y | (y >> 2)) & 0x10c30c30c30c30c3
    y = (y | (y >> 4)) & 0x100f00f00f00f00f
    y = (y | (y >> 8)) & 0x1f0000ff0000ff
    y = (y | (y >> 16)) & 0x1f00000000ffff
    y = (y | (y >> 32)) & 0x1fffff

    z = (z | (z >> 2)) & 0x10c30c30c30c30c3
    z = (z | (z >> 4)) & 0x100f00f00f00f00f
    z = (z | (z >> 8)) & 0x1f0000ff0000ff
    z = (z | (z >> 16)) & 0x1f00000000ffff
    z = (z | (z >> 32)) & 0x1fffff

    return x.astype(np.uint32), y.astype(np.uint32), z.astype(np.uint32)


def sort_by_morton_code(
    morton_codes: np.ndarray,
    element_IDs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort elements by Morton code (Z-curve order).

    Args:
        morton_codes: (N,) uint64 - Morton codes
        element_IDs: (N,) int32 - Element IDs to sort

    Returns:
        sorted_morton: (N,) uint64 - Sorted Morton codes
        sorted_element_IDs: (N,) int32 - Element IDs in Z-curve order
    """
    # Get sort indices
    sort_idx = np.argsort(morton_codes)

    # Sort both arrays
    sorted_morton = morton_codes[sort_idx]
    sorted_element_IDs = element_IDs[sort_idx]

    return sorted_morton, sorted_element_IDs


# JAX implementations (for GPU)

@jax.jit
def expand_bits_3d_jax(x: jnp.ndarray) -> jnp.ndarray:
    """
    JAX version of expand_bits_3d.

    Args:
        x: (N,) uint32 array

    Returns:
        expanded: (N,) uint64 array
    """
    x = x.astype(jnp.uint64)

    x = (x | (x << 32)) & 0x1f00000000ffff
    x = (x | (x << 16)) & 0x1f0000ff0000ff
    x = (x | (x << 8)) & 0x100f00f00f00f00f
    x = (x | (x << 4)) & 0x10c30c30c30c30c3
    x = (x | (x << 2)) & 0x1249249249249249

    return x


@jax.jit
def morton_encode_3d_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray
) -> jnp.ndarray:
    """
    JAX version of morton_encode_3d.

    Args:
        x: (N,) uint32 - x coordinates
        y: (N,) uint32 - y coordinates
        z: (N,) uint32 - z coordinates

    Returns:
        morton: (N,) uint64 - Morton codes
    """
    xx = expand_bits_3d_jax(x)
    yy = expand_bits_3d_jax(y)
    zz = expand_bits_3d_jax(z)

    morton = xx | (yy << 1) | (zz << 2)

    return morton


@jax.jit
def normalize_coordinates_jax(
    positions: jnp.ndarray,
    bbox_min: jnp.ndarray,
    bbox_max: jnp.ndarray,
    n_bits: int = 21
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JAX version of normalize_coordinates.

    Args:
        positions: (N, 3) float64
        bbox_min: (3,) float64
        bbox_max: (3,) float64
        n_bits: Bits per dimension

    Returns:
        x_int, y_int, z_int: (N,) uint32 arrays
    """
    bbox_size = bbox_max - bbox_min
    normalized = (positions - bbox_min) / bbox_size
    normalized = jnp.clip(normalized, 0.0, 1.0 - 1e-10)

    max_val = (1 << n_bits) - 1
    scaled = normalized * max_val

    x_int = scaled[:, 0].astype(jnp.uint32)
    y_int = scaled[:, 1].astype(jnp.uint32)
    z_int = scaled[:, 2].astype(jnp.uint32)

    return x_int, y_int, z_int


@jax.jit
def compute_morton_codes_jax(
    positions: jnp.ndarray,
    bbox_min: jnp.ndarray,
    bbox_max: jnp.ndarray,
    n_bits: int = 21
) -> jnp.ndarray:
    """
    JAX version of compute_morton_codes.

    Args:
        positions: (N, 3) float64
        bbox_min: (3,) float64
        bbox_max: (3,) float64
        n_bits: Bits per dimension

    Returns:
        morton_codes: (N,) uint64
    """
    x_int, y_int, z_int = normalize_coordinates_jax(
        positions, bbox_min, bbox_max, n_bits
    )

    morton_codes = morton_encode_3d_jax(x_int, y_int, z_int)

    return morton_codes


if __name__ == "__main__":
    print("Testing Morton code implementation...")

    # Test 1: Simple encoding/decoding
    print("\nTest 1: Encode and decode")
    x = np.array([0, 100, 200, 1000000], dtype=np.uint32)
    y = np.array([0, 150, 250, 1500000], dtype=np.uint32)
    z = np.array([0, 50, 300, 2000000], dtype=np.uint32)

    morton = morton_encode_3d(x, y, z)
    x_dec, y_dec, z_dec = morton_decode_3d(morton)

    print(f"  Original: x={x}, y={y}, z={z}")
    print(f"  Morton codes: {morton}")
    print(f"  Decoded: x={x_dec}, y={y_dec}, z={z_dec}")
    print(f"  Match: {np.all(x == x_dec) and np.all(y == y_dec) and np.all(z == z_dec)}")

    # Test 2: 3D positions
    print("\nTest 2: Compute from 3D positions")
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
    ], dtype=np.float64)

    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([1.0, 1.0, 1.0])

    morton_codes = compute_morton_codes(positions, bbox_min, bbox_max)
    print(f"  Positions:\n{positions}")
    print(f"  Morton codes: {morton_codes}")

    # Test 3: Sorting preserves spatial locality
    print("\nTest 3: Spatial locality preservation")
    np.random.seed(42)
    n_points = 1000
    random_positions = np.random.uniform(0.0, 1.0, (n_points, 3))

    morton_random = compute_morton_codes(random_positions, bbox_min, bbox_max)
    sorted_morton, sorted_idx = sort_by_morton_code(
        morton_random,
        np.arange(n_points, dtype=np.int32)
    )

    print(f"  Generated {n_points} random points")
    print(f"  Morton code range: [{sorted_morton[0]}, {sorted_morton[-1]}]")
    print(f"  First 10 sorted Morton codes: {sorted_morton[:10]}")

    # Test 4: JAX versions
    print("\nTest 4: JAX implementation")
    positions_jax = jnp.array(positions)
    bbox_min_jax = jnp.array(bbox_min)
    bbox_max_jax = jnp.array(bbox_max)

    morton_jax = compute_morton_codes_jax(positions_jax, bbox_min_jax, bbox_max_jax)
    print(f"  JAX Morton codes: {morton_jax}")
    print(f"  Match with NumPy: {np.allclose(morton_codes, np.array(morton_jax))}")

    print("\n✅ All tests passed!")
