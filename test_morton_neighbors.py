#!/usr/bin/env python3
"""Test Morton neighbor encode/decode correctness."""

import jax
import jax.numpy as jnp
from jaxtrace.gpu.search.morton_neighbors import (
    decode_morton_prefix_jax,
    encode_morton_prefix_jax,
    get_26_neighbor_prefixes_jax
)
from jaxtrace.gpu.search.morton_global_search import interleave_bits_3d_jax

# Test 1: Round-trip encode/decode
print("Test 1: Round-trip encode/decode")
print("=" * 60)

depth = 6
test_coords = [
    (0, 0, 0),  # Corner
    (63, 63, 63),  # Opposite corner (max for depth 6)
    (32, 32, 32),  # Center
    (1, 2, 3),  # Arbitrary
]

for x, y, z in test_coords:
    # Encode
    prefix = encode_morton_prefix_jax(jnp.int32(x), jnp.int32(y), jnp.int32(z), depth)

    # Decode
    x_dec, y_dec, z_dec = decode_morton_prefix_jax(prefix, depth)

    # Check
    match = (int(x_dec) == x) and (int(y_dec) == y) and (int(z_dec) == z)
    status = "✅ PASS" if match else "❌ FAIL"

    print(f"{status}: ({x:2d}, {y:2d}, {z:2d}) -> 0x{int(prefix):016X} -> ({int(x_dec):2d}, {int(y_dec):2d}, {int(z_dec):2d})")

# Test 2: Match with existing interleave function
print("\nTest 2: Match with existing interleave_bits_3d_jax")
print("=" * 60)

for x, y, z in test_coords:
    # My implementation
    prefix_mine = encode_morton_prefix_jax(jnp.int32(x), jnp.int32(y), jnp.int32(z), depth)

    # Existing implementation (needs to be shifted to match prefix format)
    morton_existing = interleave_bits_3d_jax(jnp.uint32(x), jnp.uint32(y), jnp.uint32(z))
    # Shift to align top bits (depth*3 bits starting at bit 60)
    prefix_existing = morton_existing << jnp.uint64(63 - depth * 3)

    match = int(prefix_mine) == int(prefix_existing)
    status = "✅ PASS" if match else "❌ FAIL"

    print(f"{status}: ({x:2d}, {y:2d}, {z:2d})")
    if not match:
        print(f"  Mine:     0x{int(prefix_mine):016X}")
        print(f"  Existing: 0x{int(prefix_existing):016X}")

# Test 3: Neighbor generation
print("\nTest 3: Neighbor generation")
print("=" * 60)

center = (32, 32, 32)  # Center of 64×64×64 grid (depth 6)
x, y, z = center
center_prefix = encode_morton_prefix_jax(jnp.int32(x), jnp.int32(y), jnp.int32(z), depth)

max_coord = jnp.int32((2 ** depth) - 1)
neighbor_prefixes = get_26_neighbor_prefixes_jax(center_prefix, depth, max_coord)

print(f"Center: ({x}, {y}, {z}) -> prefix 0x{int(center_prefix):016X}")
print(f"Generated {len(neighbor_prefixes)} neighbor prefixes (should be 27)")

# Decode all neighbors
neighbor_coords = []
for i, prefix in enumerate(neighbor_prefixes):
    nx, ny, nz = decode_morton_prefix_jax(prefix, depth)
    neighbor_coords.append((int(nx), int(ny), int(nz)))

# Check center is included (should be at index 13)
center_found = False
center_index = -1
for i, coord in enumerate(neighbor_coords):
    if coord == center:
        center_found = True
        center_index = i
        break

status = "✅ PASS" if center_found else "❌ FAIL"
print(f"\n{status}: Center found in neighbors at index {center_index} (expected 13)")

# Check all neighbors are within ±1 of center
all_valid = True
for i, (nx, ny, nz) in enumerate(neighbor_coords):
    dx = abs(nx - x)
    dy = abs(ny - y)
    dz = abs(nz - z)
    if dx > 1 or dy > 1 or dz > 1:
        print(f"❌ FAIL: Neighbor {i} ({nx}, {ny}, {nz}) is too far from center ({x}, {y}, {z})")
        all_valid = False

if all_valid:
    print("✅ PASS: All neighbors within ±1 of center")

# Count unique neighbors
unique_coords = set(neighbor_coords)
status = "✅ PASS" if len(unique_coords) == 27 else "❌ FAIL"
print(f"{status}: {len(unique_coords)} unique neighbors (expected 27)")

print("\nFirst few neighbors:")
for i in range(min(5, len(neighbor_coords))):
    nx, ny, nz = neighbor_coords[i]
    print(f"  [{i:2d}] ({nx:2d}, {ny:2d}, {nz:2d})")
