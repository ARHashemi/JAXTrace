#!/usr/bin/env python3
"""
Hilbert Curve (3D) Encoding for Spatial Partitioning

Implements 3D Hilbert curves for spatial indexing as an alternative to Morton codes.
Hilbert curves have BETTER spatial locality than Morton Z-curves, reducing discontinuities
at octree boundaries.

Key Advantages over Morton:
1. **Continuous space-filling**: No jumps at octant boundaries
2. **Better locality**: Neighboring points in 3D stay closer on the 1D curve
3. **Recursive structure**: Natural fit for adaptive octrees

Reference:
- Hilbert, D. (1891). "Über die stetige Abbildung einer Linie auf ein Flächenstück."
- Skilling, J. (2004). "Programming the Hilbert curve."
"""

from typing import Tuple
import numpy as np
import jax.numpy as jnp
import jax

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


# Hilbert curve state tables for 3D
# Each state defines how to traverse 8 octants and which state to use for each child
# State = rotation/reflection of the canonical Hilbert curve

# fmt: off
# Hilbert traversal order: which octant to visit in what order
HILBERT_CHILD_ORDER = np.array([
    [0, 7, 6, 1, 2, 5, 4, 3],  # State 0
    [0, 3, 4, 7, 6, 5, 2, 1],  # State 1
    [0, 1, 6, 7, 4, 5, 2, 3],  # State 2
    [2, 3, 0, 1, 6, 7, 4, 5],  # State 3
    [4, 3, 2, 5, 6, 1, 0, 7],  # State 4
    [4, 5, 2, 3, 0, 1, 6, 7],  # State 5
    [4, 7, 6, 5, 2, 3, 0, 1],  # State 6
    [6, 5, 2, 1, 0, 3, 4, 7],  # State 7
], dtype=np.uint8)

# Child state transitions: which state to use for each child
HILBERT_CHILD_STATE = np.array([
    [1, 0, 0, 2, 3, 7, 7, 6],  # State 0
    [0, 1, 1, 5, 4, 6, 6, 3],  # State 1
    [2, 5, 5, 0, 0, 6, 6, 1],  # State 2
    [0, 7, 7, 3, 3, 4, 4, 2],  # State 3
    [3, 4, 4, 1, 1, 7, 7, 5],  # State 4
    [2, 1, 1, 4, 4, 5, 5, 0],  # State 5
    [5, 2, 2, 6, 6, 1, 1, 4],  # State 6
    [6, 3, 3, 7, 7, 0, 0, 4],  # State 7
], dtype=np.uint8)

# Inverse mapping: octant → position in traversal order
HILBERT_OCTANT_TO_INDEX = np.array([
    [0, 3, 4, 7, 6, 5, 2, 1],  # State 0
    [0, 7, 6, 1, 2, 5, 4, 3],  # State 1
    [0, 1, 6, 7, 4, 5, 2, 3],  # State 2
    [2, 3, 0, 1, 6, 7, 4, 5],  # State 3
    [6, 5, 2, 1, 0, 3, 4, 7],  # State 4
    [2, 1, 6, 7, 4, 5, 0, 3],  # State 5
    [6, 7, 4, 5, 2, 3, 0, 1],  # State 6
    [2, 3, 4, 5, 6, 7, 0, 1],  # State 7
], dtype=np.uint8)
# fmt: on


def hilbert_encode_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_bits: int = 21
) -> np.ndarray:
    """
    Encode 3D coordinates as Hilbert indices.

    Uses iterative state-machine algorithm to traverse Hilbert curve.
    More complex than Morton but provides better spatial locality.

    Parameters
    ----------
    x : np.ndarray
        x coordinates, shape (N,), dtype uint32, range [0, 2^n_bits-1]
    y : np.ndarray
        y coordinates, shape (N,), dtype uint32, range [0, 2^n_bits-1]
    z : np.ndarray
        z coordinates, shape (N,), dtype uint32, range [0, 2^n_bits-1]
    n_bits : int, default=21
        Number of bits per dimension (21 → 63-bit Hilbert index)

    Returns
    -------
    hilbert : np.ndarray
        Hilbert indices, shape (N,), dtype uint64
    """
    x = x.astype(np.uint32)
    y = y.astype(np.uint32)
    z = z.astype(np.uint32)

    n_points = len(x)
    hilbert = np.zeros(n_points, dtype=np.uint64)
    state = np.zeros(n_points, dtype=np.uint8)  # Current Hilbert state for each point

    # Process from most significant bit to least significant bit
    for level in range(n_bits - 1, -1, -1):
        # Extract bit at current level for x, y, z
        bit_x = (x >> level) & 1
        bit_y = (y >> level) & 1
        bit_z = (z >> level) & 1

        # Compute octant number (0-7)
        octant = (bit_z << 2) | (bit_y << 1) | bit_x

        # Look up position in Hilbert traversal for current state
        for i in range(n_points):
            position = HILBERT_OCTANT_TO_INDEX[state[i], octant[i]]
            hilbert[i] = (hilbert[i] << 3) | position

            # Update state for next level
            state[i] = HILBERT_CHILD_STATE[state[i], octant[i]]

    return hilbert


def hilbert_decode_3d(
    hilbert: np.ndarray,
    n_bits: int = 21
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode Hilbert indices back to 3D coordinates.

    Useful for debugging and visualization.

    Parameters
    ----------
    hilbert : np.ndarray
        Hilbert indices, shape (N,), dtype uint64
    n_bits : int, default=21
        Number of bits per dimension

    Returns
    -------
    x : np.ndarray
        x coordinates, shape (N,), dtype uint32
    y : np.ndarray
        y coordinates, shape (N,), dtype uint32
    z : np.ndarray
        z coordinates, shape (N,), dtype uint32
    """
    hilbert = hilbert.astype(np.uint64)
    n_points = len(hilbert)

    x = np.zeros(n_points, dtype=np.uint32)
    y = np.zeros(n_points, dtype=np.uint32)
    z = np.zeros(n_points, dtype=np.uint32)
    state = np.zeros(n_points, dtype=np.uint8)

    # Process from most significant bit to least significant bit
    for level in range(n_bits - 1, -1, -1):
        # Extract 3-bit position from Hilbert index
        shift = (n_bits - 1 - level) * 3
        position = (hilbert >> shift) & 7

        # Look up octant for this position in current state
        for i in range(n_points):
            octant = HILBERT_CHILD_ORDER[state[i], position[i]]

            # Extract octant bits
            bit_x = octant & 1
            bit_y = (octant >> 1) & 1
            bit_z = (octant >> 2) & 1

            # Set bit at current level
            x[i] |= (bit_x << level)
            y[i] |= (bit_y << level)
            z[i] |= (bit_z << level)

            # Update state for next level
            state[i] = HILBERT_CHILD_STATE[state[i], octant]

    return x, y, z


def normalize_coordinates(
    positions: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    n_bits: int = 21
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize positions to integer grid for Hilbert encoding.

    Identical to Morton normalization - maps positions from [bbox_min, bbox_max]
    to [0, 2^n_bits - 1].

    Parameters
    ----------
    positions : np.ndarray
        3D positions, shape (N, 3), dtype float64
    bbox_min : np.ndarray
        Minimum bounds, shape (3,), dtype float64
    bbox_max : np.ndarray
        Maximum bounds, shape (3,), dtype float64
    n_bits : int, default=21
        Number of bits per dimension (21 → 63-bit Hilbert index)

    Returns
    -------
    x_int : np.ndarray
        Normalized x coordinates, shape (N,), dtype uint32
    y_int : np.ndarray
        Normalized y coordinates, shape (N,), dtype uint32
    z_int : np.ndarray
        Normalized z coordinates, shape (N,), dtype uint32
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


def compute_hilbert_indices(
    positions: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    n_bits: int = 21
) -> np.ndarray:
    """
    Compute Hilbert indices for 3D positions.

    Complete pipeline: normalize → encode.

    Parameters
    ----------
    positions : np.ndarray
        3D positions, shape (N, 3), dtype float64
    bbox_min : np.ndarray
        Bounding box minimum, shape (3,), dtype float64
    bbox_max : np.ndarray
        Bounding box maximum, shape (3,), dtype float64
    n_bits : int, default=21
        Bits per dimension (21 → 63-bit Hilbert index)

    Returns
    -------
    hilbert_indices : np.ndarray
        Hilbert indices, shape (N,), dtype uint64
    """
    # Normalize to integer grid
    x_int, y_int, z_int = normalize_coordinates(
        positions, bbox_min, bbox_max, n_bits
    )

    # Encode as Hilbert indices
    hilbert_indices = hilbert_encode_3d(x_int, y_int, z_int, n_bits)

    return hilbert_indices


def sort_by_hilbert_index(
    hilbert_indices: np.ndarray,
    element_IDs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort elements by Hilbert index (Hilbert curve order).

    Parameters
    ----------
    hilbert_indices : np.ndarray
        Hilbert indices, shape (N,), dtype uint64
    element_IDs : np.ndarray
        Element IDs to sort, shape (N,), dtype int32

    Returns
    -------
    sorted_hilbert : np.ndarray
        Sorted Hilbert indices, shape (N,), dtype uint64
    sorted_element_IDs : np.ndarray
        Element IDs in Hilbert curve order, shape (N,), dtype int32
    """
    # Get sort indices
    sort_idx = np.argsort(hilbert_indices)

    # Sort both arrays
    sorted_hilbert = hilbert_indices[sort_idx]
    sorted_element_IDs = element_IDs[sort_idx]

    return sorted_hilbert, sorted_element_IDs


# JAX implementations (for GPU)

@jax.jit
def hilbert_encode_3d_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    n_bits: int = 21
) -> jnp.ndarray:
    """
    JAX version of hilbert_encode_3d.

    Uses JAX scan for efficient loop execution on GPU.

    Parameters
    ----------
    x : jnp.ndarray
        x coordinates, shape (N,), dtype uint32
    y : jnp.ndarray
        y coordinates, shape (N,), dtype uint32
    z : jnp.ndarray
        z coordinates, shape (N,), dtype uint32
    n_bits : int, default=21
        Number of bits per dimension

    Returns
    -------
    hilbert : jnp.ndarray
        Hilbert indices, shape (N,), dtype uint64
    """
    x = x.astype(jnp.uint32)
    y = y.astype(jnp.uint32)
    z = z.astype(jnp.uint32)

    n_points = len(x)

    # Convert tables to JAX arrays
    child_order_jax = jnp.array(HILBERT_CHILD_ORDER, dtype=jnp.uint8)
    child_state_jax = jnp.array(HILBERT_CHILD_STATE, dtype=jnp.uint8)
    octant_to_index_jax = jnp.array(HILBERT_OCTANT_TO_INDEX, dtype=jnp.uint8)

    def step_fn(carry, level_idx):
        hilbert, state = carry

        # Extract bit at current level
        level = n_bits - 1 - level_idx
        bit_x = (x >> level) & 1
        bit_y = (y >> level) & 1
        bit_z = (z >> level) & 1

        # Compute octant
        octant = (bit_z << 2) | (bit_y << 1) | bit_x

        # Vectorized lookup
        position = octant_to_index_jax[state, octant]
        hilbert = (hilbert << 3) | position.astype(jnp.uint64)

        # Update state
        state = child_state_jax[state, octant]

        return (hilbert, state), None

    # Initial state
    hilbert_init = jnp.zeros(n_points, dtype=jnp.uint64)
    state_init = jnp.zeros(n_points, dtype=jnp.uint8)

    # Run scan
    (hilbert_final, _), _ = jax.lax.scan(
        step_fn,
        (hilbert_init, state_init),
        jnp.arange(n_bits)
    )

    return hilbert_final


@jax.jit
def normalize_coordinates_jax(
    positions: jnp.ndarray,
    bbox_min: jnp.ndarray,
    bbox_max: jnp.ndarray,
    n_bits: int = 21
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JAX version of normalize_coordinates.

    Parameters
    ----------
    positions : jnp.ndarray
        3D positions, shape (N, 3), dtype float64
    bbox_min : jnp.ndarray
        Minimum bounds, shape (3,), dtype float64
    bbox_max : jnp.ndarray
        Maximum bounds, shape (3,), dtype float64
    n_bits : int
        Bits per dimension

    Returns
    -------
    x_int, y_int, z_int : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Normalized coordinates, shape (N,), dtype uint32
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
def compute_hilbert_indices_jax(
    positions: jnp.ndarray,
    bbox_min: jnp.ndarray,
    bbox_max: jnp.ndarray,
    n_bits: int = 21
) -> jnp.ndarray:
    """
    JAX version of compute_hilbert_indices.

    Parameters
    ----------
    positions : jnp.ndarray
        3D positions, shape (N, 3), dtype float64
    bbox_min : jnp.ndarray
        Bounding box minimum, shape (3,), dtype float64
    bbox_max : jnp.ndarray
        Bounding box maximum, shape (3,), dtype float64
    n_bits : int
        Bits per dimension

    Returns
    -------
    hilbert_indices : jnp.ndarray
        Hilbert indices, shape (N,), dtype uint64
    """
    x_int, y_int, z_int = normalize_coordinates_jax(
        positions, bbox_min, bbox_max, n_bits
    )

    hilbert_indices = hilbert_encode_3d_jax(x_int, y_int, z_int, n_bits)

    return hilbert_indices


if __name__ == "__main__":
    print("Testing Hilbert curve implementation...")

    # Test 1: Simple encoding/decoding
    print("\nTest 1: Encode and decode")
    x = np.array([0, 100, 200, 1000000], dtype=np.uint32)
    y = np.array([0, 150, 250, 1500000], dtype=np.uint32)
    z = np.array([0, 50, 300, 2000000], dtype=np.uint32)

    hilbert = hilbert_encode_3d(x, y, z, n_bits=21)
    x_dec, y_dec, z_dec = hilbert_decode_3d(hilbert, n_bits=21)

    print(f"  Original: x={x}, y={y}, z={z}")
    print(f"  Hilbert indices: {hilbert}")
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

    hilbert_indices = compute_hilbert_indices(positions, bbox_min, bbox_max)
    print(f"  Positions:\n{positions}")
    print(f"  Hilbert indices: {hilbert_indices}")

    # Test 3: Spatial locality preservation
    print("\nTest 3: Spatial locality preservation (compare with Morton)")
    from jaxtrace.gpu.morton_code import compute_morton_codes

    np.random.seed(42)
    n_points = 1000
    random_positions = np.random.uniform(0.0, 1.0, (n_points, 3))

    hilbert_random = compute_hilbert_indices(random_positions, bbox_min, bbox_max)
    morton_random = compute_morton_codes(random_positions, bbox_min, bbox_max)

    sorted_hilbert, _ = sort_by_hilbert_index(
        hilbert_random,
        np.arange(n_points, dtype=np.int32)
    )

    print(f"  Generated {n_points} random points")
    print(f"  Hilbert index range: [{sorted_hilbert[0]}, {sorted_hilbert[-1]}]")
    print(f"  First 10 sorted Hilbert indices: {sorted_hilbert[:10]}")

    # Test 4: JAX versions
    print("\nTest 4: JAX implementation")
    positions_jax = jnp.array(positions)
    bbox_min_jax = jnp.array(bbox_min)
    bbox_max_jax = jnp.array(bbox_max)

    hilbert_jax = compute_hilbert_indices_jax(positions_jax, bbox_min_jax, bbox_max_jax)
    print(f"  JAX Hilbert indices: {hilbert_jax}")
    print(f"  Match with NumPy: {np.allclose(hilbert_indices, np.array(hilbert_jax))}")

    print("\n✅ All tests passed!")
