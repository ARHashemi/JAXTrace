#!/usr/bin/env python3
"""
Morton Code (Z-order) Encoding for Octree Optimization.

Phase 2 optimization: Replace explicit node positions (24 bytes) with
implicit Morton codes (8 bytes) for 3× memory reduction.

Morton codes provide:
- Compact spatial encoding (8 bytes vs 24 bytes)
- Implicit parent/child relationships via bit operations
- Better spatial coherence (sequential codes = nearby in 3D)
- Fast binary search O(log n)

References:
- Morton, G.M. (1966). "A computer Oriented Geodetic Data Base"
- Karras, T. (2012). "Thinking Parallel, Part III: Tree Construction on the GPU"
"""

import numpy as np
import numba
import jax
import jax.numpy as jnp
from typing import Tuple


@numba.njit
def encode_morton_3d(x: float, y: float, z: float, level: int,
                      domain_min: np.ndarray, domain_max: np.ndarray) -> np.uint64:
    """
    Encode 3D position and tree level into 64-bit Morton code.

    Morton encoding interleaves the bits of x, y, z coordinates to create
    a single integer that preserves spatial locality (Z-order curve).

    Bit layout (64 bits total):
    - Bits 0-7:   Level (8 bits, max depth 255)
    - Bits 8-63:  Morton code (56 bits = 18 bits per dimension, depth ~18)

    Args:
        x, y, z: 3D position in world coordinates
        level: Tree depth level (0 = root, max 255)
        domain_min: Minimum bounds of domain (3,)
        domain_max: Maximum bounds of domain (3,)

    Returns:
        morton_code: 64-bit unsigned integer encoding position and level

    Example:
        >>> domain_min = np.array([-1.0, -1.0, -1.0])
        >>> domain_max = np.array([1.0, 1.0, 1.0])
        >>> code = encode_morton_3d(0.0, 0.0, 0.0, 5, domain_min, domain_max)
        >>> # Decodes back to center of domain at level 5
    """
    # Normalize coordinates to [0, 2^18) range (18 bits per dimension)
    # This gives us 18 levels of depth (2^18 = 262,144 subdivisions)
    MAX_BITS = 18
    MAX_VAL = np.uint64((1 << MAX_BITS) - 1)

    # Convert to normalized coordinates [0, 1]
    nx = (x - domain_min[0]) / (domain_max[0] - domain_min[0])
    ny = (y - domain_min[1]) / (domain_max[1] - domain_min[1])
    nz = (z - domain_min[2]) / (domain_max[2] - domain_min[2])

    # Scale to integer range and convert to uint64
    ix = np.uint64(nx * float(MAX_VAL))
    iy = np.uint64(ny * float(MAX_VAL))
    iz = np.uint64(nz * float(MAX_VAL))

    # Clamp to valid range
    if ix > MAX_VAL:
        ix = MAX_VAL
    if iy > MAX_VAL:
        iy = MAX_VAL
    if iz > MAX_VAL:
        iz = MAX_VAL

    # Interleave bits: z0y0x0 z1y1x1 z2y2x2 ... (56 bits total)
    # This creates a Z-order curve that preserves spatial locality
    morton = np.uint64(0)
    for i in range(MAX_BITS):
        bit_x = np.uint64((ix >> i) & np.uint64(1))
        bit_y = np.uint64((iy >> i) & np.uint64(1))
        bit_z = np.uint64((iz >> i) & np.uint64(1))
        shift_x = np.uint64(3 * i)
        shift_y = np.uint64(3 * i + 1)
        shift_z = np.uint64(3 * i + 2)
        morton |= (bit_x << shift_x) | (bit_y << shift_y) | (bit_z << shift_z)

    # Pack: morton code (56 bits) + level (8 bits)
    # Level in lower 8 bits for faster access
    return (morton << 8) | np.uint64(level)


@numba.njit
def decode_morton_3d(morton_code: np.uint64, domain_min: np.ndarray,
                      domain_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Decode Morton code back to 3D bounding box and level.

    Extracts the tree level and spatial position from a Morton code,
    reconstructing the node's axis-aligned bounding box.

    Args:
        morton_code: 64-bit Morton code from encode_morton_3d()
        domain_min: Minimum bounds of domain (3,)
        domain_max: Maximum bounds of domain (3,)

    Returns:
        node_min: Minimum corner of node bounding box (3,)
        node_max: Maximum corner of node bounding box (3,)
        level: Tree depth level

    Example:
        >>> code = encode_morton_3d(0.5, 0.5, 0.5, 3, domain_min, domain_max)
        >>> node_min, node_max, level = decode_morton_3d(code, domain_min, domain_max)
        >>> # Returns bounds of octree node at level 3 containing (0.5, 0.5, 0.5)
    """
    # Extract level from lower 8 bits
    level = int(morton_code & np.uint64(0xFF))

    # Extract Morton code from upper 56 bits
    morton = np.uint64(morton_code >> np.uint64(8))

    # De-interleave bits to recover x, y, z indices
    MAX_BITS = 18
    ix = np.uint64(0)
    iy = np.uint64(0)
    iz = np.uint64(0)

    for i in range(MAX_BITS):
        shift_x = np.uint64(3 * i)
        shift_y = np.uint64(3 * i + 1)
        shift_z = np.uint64(3 * i + 2)
        i_shift = np.uint64(i)
        ix |= ((morton >> shift_x) & np.uint64(1)) << i_shift
        iy |= ((morton >> shift_y) & np.uint64(1)) << i_shift
        iz |= ((morton >> shift_z) & np.uint64(1)) << i_shift

    # Convert indices to world coordinates
    domain_size = domain_max - domain_min
    scale = domain_size / (1 << MAX_BITS)

    # Calculate node size at this level
    # At level L, each dimension is divided into 2^L parts
    node_size = domain_size / (1 << level)

    # Compute minimum corner
    # Scale down from MAX_BITS resolution to level resolution
    level_scale = 1 << (MAX_BITS - level)
    node_min = domain_min + np.array([
        (ix // level_scale) * node_size[0],
        (iy // level_scale) * node_size[1],
        (iz // level_scale) * node_size[2]
    ], dtype=np.float32)

    node_max = node_min + node_size

    return node_min, node_max, level


@numba.njit
def get_morton_parent(morton_code: np.uint64) -> np.uint64:
    """
    Get parent node's Morton code via bit shifting.

    The parent of a node at level L is at level L-1 with the same
    spatial position but coarser resolution.

    Args:
        morton_code: Child node's Morton code

    Returns:
        parent_code: Parent node's Morton code (level - 1)

    Example:
        >>> child = encode_morton_3d(0.25, 0.25, 0.25, 5, domain_min, domain_max)
        >>> parent = get_morton_parent(child)
        >>> # Parent is at level 4
    """
    level = np.uint64(morton_code & np.uint64(0xFF))

    if level == 0:
        # Root node has no parent
        return morton_code

    # Extract Morton code
    morton = np.uint64(morton_code >> np.uint64(8))

    # Right-shift by 3 bits to go up one level (remove one x,y,z triple)
    parent_morton = np.uint64(morton >> np.uint64(3))

    # Pack with decremented level
    return (parent_morton << np.uint64(8)) | (level - np.uint64(1))


@numba.njit
def get_morton_children(morton_code: np.uint64) -> np.ndarray:
    """
    Get all 8 children's Morton codes via bit operations.

    An octree node has 8 children, corresponding to the 8 octants.
    Each child adds one more bit to x, y, z coordinates.

    Args:
        morton_code: Parent node's Morton code

    Returns:
        children: Array of 8 Morton codes for children (8,)
                  Order: [000, 001, 010, 011, 100, 101, 110, 111]
                         where bits are (z, y, x)

    Example:
        >>> parent = encode_morton_3d(0.0, 0.0, 0.0, 3, domain_min, domain_max)
        >>> children = get_morton_children(parent)
        >>> # Returns 8 child Morton codes at level 4
    """
    level = np.uint64(morton_code & np.uint64(0xFF))
    morton = np.uint64(morton_code >> np.uint64(8))

    # Left-shift by 3 to make room for new x,y,z bits
    base_morton = np.uint64(morton << np.uint64(3))
    child_level = np.uint64(level + np.uint64(1))

    # Generate all 8 children by adding 0-7 (all 3-bit combinations)
    children = np.empty(8, dtype=np.uint64)
    for i in range(8):
        child_morton = np.uint64(base_morton | np.uint64(i))
        children[i] = (child_morton << np.uint64(8)) | child_level

    return children


@numba.njit
def morton_distance(code1: np.uint64, code2: np.uint64) -> int:
    """
    Compute spatial distance between two Morton codes.

    Returns the number of most-significant bits that differ.
    Smaller distance = spatially closer in the octree.

    Args:
        code1, code2: Morton codes to compare

    Returns:
        distance: Number of differing high-order bits (spatial distance metric)

    Example:
        >>> code1 = encode_morton_3d(0.0, 0.0, 0.0, 5, domain_min, domain_max)
        >>> code2 = encode_morton_3d(0.1, 0.0, 0.0, 5, domain_min, domain_max)
        >>> dist = morton_distance(code1, code2)
        >>> # Nearby points have small distance
    """
    # XOR gives bits that differ
    xor = (code1 >> 8) ^ (code2 >> 8)

    # Count leading zeros (position of highest differing bit)
    # This gives a measure of spatial distance
    if xor == 0:
        return 0

    # Count bits set (spatial distance)
    distance = 0
    while xor > 0:
        distance += 1
        xor >>= 1

    return distance


# ============================================================================
# Phase 3B: Pure NumPy/JAX Implementations (Zero Numba)
# ============================================================================

def encode_morton_3d_numpy(x: float, y: float, z: float, level: int,
                           domain_min: np.ndarray, domain_max: np.ndarray) -> np.uint64:
    """
    Phase 3B: Pure NumPy Morton encoding (NO Numba, NO JAX).

    Used during hash octree BUILDING phase (CPU, one-time cost).
    Identical algorithm to Numba version but using pure NumPy/Python.

    Args:
        x, y, z: 3D position in world coordinates
        level: Tree depth level (0 = root, max 255)
        domain_min: Minimum bounds of domain (3,)
        domain_max: Maximum bounds of domain (3,)

    Returns:
        morton_code: 64-bit unsigned integer encoding position and level
    """
    MAX_BITS = 18
    MAX_VAL = np.uint64((1 << MAX_BITS) - 1)

    # Normalize to [0, 1]
    nx = (x - domain_min[0]) / (domain_max[0] - domain_min[0])
    ny = (y - domain_min[1]) / (domain_max[1] - domain_min[1])
    nz = (z - domain_min[2]) / (domain_max[2] - domain_min[2])

    # Scale to integer range
    ix = np.uint64(nx * float(MAX_VAL))
    iy = np.uint64(ny * float(MAX_VAL))
    iz = np.uint64(nz * float(MAX_VAL))

    # Clamp to valid range
    ix = min(ix, MAX_VAL)
    iy = min(iy, MAX_VAL)
    iz = min(iz, MAX_VAL)

    # Interleave bits (pure Python loop - OK for CPU building phase)
    morton = np.uint64(0)
    for i in range(MAX_BITS):
        bit_x = np.uint64((ix >> i) & np.uint64(1))
        bit_y = np.uint64((iy >> i) & np.uint64(1))
        bit_z = np.uint64((iz >> i) & np.uint64(1))
        morton |= (bit_x << (3*i)) | (bit_y << (3*i + 1)) | (bit_z << (3*i + 2))

    # Pack: morton code (56 bits) + level (8 bits)
    return (morton << 8) | np.uint64(level)


@jax.jit
def encode_morton_3d_jax(point: jnp.ndarray, level: int,
                         domain_min: jnp.ndarray, domain_max: jnp.ndarray) -> jnp.uint64:
    """
    Phase 3B: Pure JAX Morton encoding (GPU-compilable).

    Used during hash octree LOOKUP phase (GPU, hot path).
    This version is JIT-compilable and runs on GPU.

    Args:
        point: 3D position [3] in world coordinates
        level: Tree depth level (0 = root, max 255)
        domain_min: Minimum bounds of domain [3]
        domain_max: Maximum bounds of domain [3]

    Returns:
        morton_code: 64-bit unsigned integer encoding position and level
    """
    MAX_BITS = 18
    MAX_VAL = jnp.uint32((1 << MAX_BITS) - 1)

    # Normalize to [0, 1]
    normalized = (point - domain_min) / (domain_max - domain_min)

    # Scale to integer range
    scaled = (normalized * jnp.float32(MAX_VAL)).astype(jnp.uint32)
    ix, iy, iz = scaled[0], scaled[1], scaled[2]

    # Clamp to valid range (JAX-safe)
    ix = jnp.clip(ix, 0, MAX_VAL)
    iy = jnp.clip(iy, 0, MAX_VAL)
    iz = jnp.clip(iz, 0, MAX_VAL)

    # Interleave bits (loop gets unrolled at JIT time - safe!)
    morton = jnp.uint64(0)
    for i in range(MAX_BITS):
        # Extract bits
        bit_x = jnp.uint64((ix >> i) & jnp.uint32(1))
        bit_y = jnp.uint64((iy >> i) & jnp.uint32(1))
        bit_z = jnp.uint64((iz >> i) & jnp.uint32(1))
        # Interleave: z-y-x order
        morton = morton | (bit_x << (3*i)) | (bit_y << (3*i + 1)) | (bit_z << (3*i + 2))

    # Pack: morton code (56 bits) + level (8 bits)
    return (morton << 8) | jnp.uint64(level)


# Vectorized JAX version for batch encoding
encode_morton_3d_batch_jax = jax.vmap(
    encode_morton_3d_jax,
    in_axes=(0, None, None, None)  # vmap over points (axis 0)
)


# ============================================================================
# Direct Grid Coordinate Encoding
# ============================================================================

def morton_encode_3d(i: int, j: int, k: int, level: int) -> np.uint64:
    """
    Encode integer grid coordinates directly to Morton code.

    This is used for building octrees where we track integer grid positions.
    Ensures unique Morton codes for each node at each depth level.

    Args:
        i, j, k: Integer grid coordinates at this depth (0 to 2^level - 1)
        level: Octree depth level (0-18)

    Returns:
        Morton code: 64-bit uint with interleaved bits + level

    Format: [54 bits spatial (18*3)] [8 bits level] [2 bits unused]

    Example:
        At level 2 (4x4x4 grid):
        morton_encode_3d(1, 2, 3, 2) → unique code for cell (1,2,3)
    """
    # Validate inputs
    max_coord = (1 << level) - 1
    assert 0 <= i <= max_coord, f"i={i} out of range [0, {max_coord}] for level {level}"
    assert 0 <= j <= max_coord, f"j={j} out of range [0, {max_coord}] for level {level}"
    assert 0 <= k <= max_coord, f"k={k} out of range [0, {max_coord}] for level {level}"
    assert 0 <= level <= 18, f"level={level} out of range [0, 18]"

    # Convert to uint64
    ix = np.uint64(i)
    iy = np.uint64(j)
    iz = np.uint64(k)

    # Interleave bits: Z-order curve
    morton = np.uint64(0)
    for bit in range(18):  # 18 bits max per coordinate
        morton |= ((ix >> bit) & np.uint64(1)) << (3 * bit)
        morton |= ((iy >> bit) & np.uint64(1)) << (3 * bit + 1)
        morton |= ((iz >> bit) & np.uint64(1)) << (3 * bit + 2)

    # Add level in lower 8 bits
    return (morton << 8) | np.uint64(level)


# ============================================================================
# Utility Functions
# ============================================================================

def compute_morton_codes_batch(positions: np.ndarray, levels: np.ndarray,
                                 domain_min: np.ndarray, domain_max: np.ndarray) -> np.ndarray:
    """
    Encode multiple positions to Morton codes (vectorized).

    Args:
        positions: Array of 3D positions (N, 3)
        levels: Array of levels for each position (N,)
        domain_min: Minimum bounds of domain (3,)
        domain_max: Maximum bounds of domain (3,)

    Returns:
        morton_codes: Array of Morton codes (N,)
    """
    n = positions.shape[0]
    morton_codes = np.empty(n, dtype=np.uint64)

    for i in range(n):
        morton_codes[i] = encode_morton_3d(
            positions[i, 0], positions[i, 1], positions[i, 2],
            levels[i], domain_min, domain_max
        )

    return morton_codes


def decode_morton_codes_batch(morton_codes: np.ndarray, domain_min: np.ndarray,
                               domain_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode multiple Morton codes to bounding boxes (vectorized).

    Args:
        morton_codes: Array of Morton codes (N,)
        domain_min: Minimum bounds of domain (3,)
        domain_max: Maximum bounds of domain (3,)

    Returns:
        mins: Array of minimum corners (N, 3)
        maxs: Array of maximum corners (N, 3)
        levels: Array of levels (N,)
    """
    n = len(morton_codes)
    mins = np.empty((n, 3), dtype=np.float32)
    maxs = np.empty((n, 3), dtype=np.float32)
    levels = np.empty(n, dtype=np.int32)

    for i in range(n):
        node_min, node_max, level = decode_morton_3d(
            morton_codes[i], domain_min, domain_max
        )
        mins[i] = node_min
        maxs[i] = node_max
        levels[i] = level

    return mins, maxs, levels


def get_memory_savings(n_nodes: int) -> dict:
    """
    Calculate memory savings from using Morton codes.

    Args:
        n_nodes: Number of octree nodes

    Returns:
        savings: Dictionary with memory statistics
    """
    # Old storage: center (12B) + half_size (12B) = 24B per node
    old_bytes = n_nodes * 24

    # New storage: morton_code (8B) per node
    new_bytes = n_nodes * 8

    savings_bytes = old_bytes - new_bytes
    savings_factor = old_bytes / new_bytes if new_bytes > 0 else 0

    return {
        'n_nodes': n_nodes,
        'old_bytes': old_bytes,
        'new_bytes': new_bytes,
        'savings_bytes': savings_bytes,
        'savings_mb': savings_bytes / (1024 ** 2),
        'savings_factor': savings_factor,
        'old_mb': old_bytes / (1024 ** 2),
        'new_mb': new_bytes / (1024 ** 2),
    }
