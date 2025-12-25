#!/usr/bin/env python3
"""
Morton Neighbor Arithmetic - Spatial Octant Navigation

Implements Morton neighbor finding using bit arithmetic to identify the 26
spatially adjacent octants around a query point. This replaces linear ±radius
search with geometrically correct spatial neighbor search.

Key Concepts:
- Morton codes are Z-order space-filling curves created by interleaving (x,y,z) bits
- A Morton prefix at depth D identifies a specific octant in a 2^D × 2^D × 2^D grid
- Spatial neighbors are found by: decode → find neighbors in grid → re-encode

Architecture:
- All functions designed for single-particle processing (vmapped externally)
- No @jax.jit decorators (called from within JIT-compiled functions)
- Bounded loops compatible with JAX tracing
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple


# ============================================================================
# Morton Prefix Decoding
# ============================================================================

def decode_morton_prefix_jax(prefix: jnp.uint64, depth: int) -> Tuple[jnp.int32, jnp.int32, jnp.int32]:
    """
    De-interleave Morton prefix bits to extract (x, y, z) octant coordinates.

    Morton encoding interleaves bits as [z][y][x][z][y][x]... from MSB to LSB.
    This function reverses that process to recover octant coordinates.

    Example:
        prefix = 0b001101011 (9 bits = depth 3, stored in top 9 bits of uint64)
        Actual value: 0b001101011000...000 (left-aligned in 64-bit)

        Extract bit triplets (from MSB):
        Level 0: [001] → octant (0,0,1) at depth 0
        Level 1: [101] → octant (1,0,1) at depth 1
        Level 2: [011] → octant (0,1,1) at depth 2

        De-interleave to coordinates:
        x = 010₂ = 2
        y = 101₂ = 5
        z = 111₂ = 7

        Result: octant (2, 5, 7) in 2³ × 2³ × 2³ grid

    Args:
        prefix: uint64 - Morton prefix with top (depth * 3) bits meaningful
        depth: int - octree depth (number of subdivision levels)

    Returns:
        (x, y, z): Octant coordinates in range [0, 2^depth - 1]

    Note:
        NOT JIT-decorated - designed to be called within JIT-compiled functions.
    """
    # Initialize coordinates
    x = jnp.int32(0)
    y = jnp.int32(0)
    z = jnp.int32(0)

    # Extract bits level by level (unrolled for JAX tracing)
    # Maximum depth is 21 (63 bits / 3), but we'll support up to depth 10 for practicality
    # For depth > 10, would need more unrolling

    # Use lax.fori_loop for bounded iteration (JAX-compatible)
    def extract_level(i, coords):
        x, y, z = coords

        # Only process if i < depth
        active = i < depth

        # Bit position for this level (from MSB)
        # Morton codes are stored with MSB first in uint64
        bit_pos = (63 - 3) - i * 3  # Start at bit 60 for level 0

        # Extract 3-bit octant code for this level
        octant_bits = lax.shift_right_logical(prefix, jnp.uint64(bit_pos))
        octant_bits = octant_bits & jnp.uint64(0b111)
        octant_bits = octant_bits.astype(jnp.int32)

        # De-interleave: Morton uses [z][y][x] bit order
        x_bit = (octant_bits >> 2) & 1  # Bit 2 is x
        y_bit = (octant_bits >> 1) & 1  # Bit 1 is y
        z_bit = (octant_bits >> 0) & 1  # Bit 0 is z

        # Accumulate coordinates (build from MSB to LSB)
        # Shift left to make room for next bit, then OR in new bit
        x_new = (x << 1) | x_bit
        y_new = (y << 1) | y_bit
        z_new = (z << 1) | z_bit

        # Only update if active
        x = jnp.where(active, x_new, x)
        y = jnp.where(active, y_new, y)
        z = jnp.where(active, z_new, z)

        return (x, y, z)

    # Extract all levels (max 21, but typically depth ≤ 10)
    coords = lax.fori_loop(0, min(depth, 21), extract_level, (x, y, z))

    return coords


def encode_morton_prefix_jax(
    x: jnp.int32,
    y: jnp.int32,
    z: jnp.int32,
    depth: int
) -> jnp.uint64:
    """
    Encode octant coordinates (x, y, z) to Morton prefix.

    Interleaves coordinate bits to create a Morton code prefix that identifies
    a specific octant at the given depth.

    Args:
        x, y, z: Octant coordinates in range [0, 2^depth - 1]
        depth: Number of octree levels

    Returns:
        prefix: uint64 - Morton prefix with top (depth * 3) bits set

    Note:
        NOT JIT-decorated - designed to be called within JIT-compiled functions.
    """
    prefix = jnp.uint64(0)

    # Interleave bits level by level
    def interleave_level(i, prefix_acc):
        # Only process if i < depth
        active = i < depth

        # Extract bit i from each coordinate (from MSB to LSB)
        bit_idx = depth - 1 - i
        x_bit = (x >> bit_idx) & 1
        y_bit = (y >> bit_idx) & 1
        z_bit = (z >> bit_idx) & 1

        # Combine into 3-bit octant code: [z][y][x]
        octant_bits = jnp.uint64((z_bit << 2) | (y_bit << 1) | (x_bit << 0))

        # Position in output Morton code (from MSB)
        bit_pos = (63 - 3) - i * 3

        # Insert into prefix
        octant_shifted = lax.shift_left(octant_bits, jnp.uint64(bit_pos))
        prefix_new = prefix_acc | octant_shifted

        # Only update if active
        return jnp.where(active, prefix_new, prefix_acc)

    prefix = lax.fori_loop(0, min(depth, 21), interleave_level, prefix)

    return prefix


# ============================================================================
# Spatial Neighbor Finding
# ============================================================================

def get_26_neighbor_prefixes_jax(
    center_prefix: jnp.uint64,
    depth: int,
    max_coord: jnp.int32
) -> jax.Array:
    """
    Generate Morton prefixes for 26 spatial neighbor octants (+ center = 27).

    Given a Morton prefix identifying an octant, finds all 26 spatially adjacent
    octants in the 3×3×3 neighborhood (excluding center).

    Process:
    1. Decode center Morton prefix → (cx, cy, cz) coordinates
    2. Generate 27 neighbor coordinates: (cx±1, cy±1, cz±1)
    3. Clamp coordinates to valid range
    4. Encode each neighbor back to Morton prefix

    Args:
        center_prefix: uint64 - Morton prefix of center octant
        depth: int - Octree depth
        max_coord: int32 - Maximum valid coordinate (2^depth - 1)

    Returns:
        neighbor_prefixes: (27,) uint64 - Morton prefixes of all neighbors + center
                                          Center is at index 13 (middle of 3×3×3 cube)

    Note:
        Returns 27 prefixes (not 26) for simplicity. Center is included.
        NOT JIT-decorated - designed to be called within JIT-compiled functions.
    """
    # Decode center octant coordinates
    cx, cy, cz = decode_morton_prefix_jax(center_prefix, depth)

    # Pre-allocate neighbor array (27 neighbors including center)
    neighbors = jnp.zeros(27, dtype=jnp.uint64)

    # Generate 3×3×3 = 27 neighbors
    # Index mapping: idx = (dx+1)*9 + (dy+1)*3 + (dz+1)
    # Center (dx=0, dy=0, dz=0) is at idx = 1*9 + 1*3 + 1 = 13

    def compute_neighbor(idx, neighbors_arr):
        """Compute one neighbor and insert into array."""
        # Decode index to (dx, dy, dz) offsets
        dz = (idx % 3) - 1  # -1, 0, or 1
        dy = ((idx // 3) % 3) - 1
        dx = ((idx // 9) % 3) - 1

        # Compute neighbor coordinates
        nx = cx + dx
        ny = cy + dy
        nz = cz + dz

        # Clamp to valid range [0, max_coord]
        nx = jnp.clip(nx, 0, max_coord)
        ny = jnp.clip(ny, 0, max_coord)
        nz = jnp.clip(nz, 0, max_coord)

        # Encode back to Morton prefix
        neighbor_prefix = encode_morton_prefix_jax(nx, ny, nz, depth)

        # Insert into array
        neighbors_arr = neighbors_arr.at[idx].set(neighbor_prefix)

        return neighbors_arr

    # Generate all 27 neighbors using bounded loop
    neighbors = lax.fori_loop(0, 27, compute_neighbor, neighbors)

    return neighbors


# ============================================================================
# Validation & Testing Utilities
# ============================================================================

def validate_morton_roundtrip(prefix: jnp.uint64, depth: int) -> bool:
    """
    Test that encode(decode(prefix)) == prefix.

    Useful for debugging and validation.
    """
    # Decode
    x, y, z = decode_morton_prefix_jax(prefix, depth)

    # Re-encode
    prefix_reconstructed = encode_morton_prefix_jax(x, y, z, depth)

    # Compare (only top depth*3 bits matter)
    bits_to_check = depth * 3
    shift = 63 - bits_to_check

    prefix_masked = lax.shift_right_logical(prefix, jnp.uint64(shift))
    reconstructed_masked = lax.shift_right_logical(prefix_reconstructed, jnp.uint64(shift))

    return prefix_masked == reconstructed_masked


# ============================================================================
# Integration with Existing Morton Structure
# ============================================================================

def prefix_to_neighbor_leaf_ids(
    center_prefix: jnp.uint64,
    depth: int,
    prefix_start: jax.Array,
    prefix_length: jax.Array,
    n_leaves: int
) -> jax.Array:
    """
    Convert 26 spatial neighbor prefixes to leaf IDs.

    This bridges Morton neighbor arithmetic with the existing octree leaf structure.

    Args:
        center_prefix: uint64 - Morton prefix of query octant
        depth: int - Table depth for prefix lookup
        prefix_start: Array - prefix→first_leaf_id mapping
        prefix_length: Array - prefix→num_leaves mapping
        n_leaves: int - Total number of leaves (for clamping)

    Returns:
        neighbor_leaf_ids: (27,) int32 - Leaf IDs for each neighbor (-1 if none)

    Note:
        Some prefixes may have no leaves (empty octants) → returns -1
        NOT JIT-decorated - designed to be called within JIT-compiled functions.
    """
    # Get all 26 neighbor prefixes + center
    max_coord = (2 ** depth) - 1
    neighbor_prefixes = get_26_neighbor_prefixes_jax(
        center_prefix,
        depth,
        jnp.int32(max_coord)
    )

    # Look up leaf ID for each neighbor prefix
    def prefix_to_leaf(neighbor_prefix):
        # Extract prefix index (shift to get top depth*3 bits)
        shift_amount = 63 - (depth * 3)
        prefix_idx = lax.shift_right_logical(neighbor_prefix, jnp.uint64(shift_amount))
        prefix_idx = prefix_idx.astype(jnp.int32)

        # Clamp to valid prefix table range
        prefix_idx = jnp.clip(prefix_idx, 0, prefix_start.shape[0] - 1)

        # Look up first leaf for this prefix
        first_leaf = prefix_start[prefix_idx]
        num_leaves_in_prefix = prefix_length[prefix_idx]

        # Return first leaf if exists, else -1
        # Note: For now, we only return the first leaf per prefix
        # Future: Could search all leaves in prefix range
        return jnp.where(num_leaves_in_prefix > 0, first_leaf, jnp.int32(-1))

    # Map over all 27 neighbors
    neighbor_leaf_ids = jax.vmap(prefix_to_leaf)(neighbor_prefixes)

    return neighbor_leaf_ids
