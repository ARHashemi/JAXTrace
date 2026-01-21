#!/usr/bin/env python3
"""
Diagnose the Morton neighbor search bug by tracing actual values.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jaxtrace.gpu.search.morton_neighbors import (
    decode_morton_prefix_jax,
    encode_morton_prefix_jax,
    get_26_neighbor_prefixes_jax
)

def test_morton_encoding():
    """Test Morton encoding/decoding to understand value ranges."""

    print("=" * 80)
    print("Morton Encoding/Decoding Value Trace")
    print("=" * 80)

    # Test coordinates at depth 7 (128^3 grid)
    x, y, z = 61, 35, 27
    depth = 7

    print(f"\n1. Input coordinates: ({x}, {y}, {z}) at depth {depth}")
    print(f"   Grid size: {2**depth}^3 = {(2**depth)**3:,} octants")

    # Encode to Morton prefix
    morton_prefix = encode_morton_prefix_jax(
        jnp.int32(x),
        jnp.int32(y),
        jnp.int32(z),
        depth
    )

    print(f"\n2. Encoded Morton prefix (uint64):")
    print(f"   Hex: 0x{int(morton_prefix):016X}")
    print(f"   Dec: {int(morton_prefix):,}")
    print(f"   Binary (top 24 bits): {bin(int(morton_prefix) >> 40)}")

    # Check if left-aligned
    shift_for_index = 63 - (depth * 3)
    prefix_as_index = int(morton_prefix) >> shift_for_index
    print(f"\n3. Extract as table index (shift right by {shift_for_index}):")
    print(f"   Index: {prefix_as_index:,}")
    print(f"   Hex: 0x{prefix_as_index:X}")

    # Decode back
    x_dec, y_dec, z_dec = decode_morton_prefix_jax(morton_prefix, depth)
    print(f"\n4. Decoded coordinates: ({int(x_dec)}, {int(y_dec)}, {int(z_dec)})")
    print(f"   Match: {int(x_dec)==x and int(y_dec)==y and int(z_dec)==z}")

    # Now test with SHIFTED prefix (the bug scenario)
    morton_shifted = morton_prefix >> jnp.uint64(shift_for_index)
    print(f"\n5. WRONG: Pass shifted prefix to decode:")
    print(f"   Shifted prefix: 0x{int(morton_shifted):X}")
    x_wrong, y_wrong, z_wrong = decode_morton_prefix_jax(morton_shifted, depth)
    print(f"   Decoded (WRONG): ({int(x_wrong)}, {int(y_wrong)}, {int(z_wrong)})")
    print(f"   Match: {int(x_wrong)==x and int(y_wrong)==y and int(z_wrong)==z}")

    return morton_prefix, prefix_as_index


def test_neighbor_generation():
    """Test 26-neighbor generation."""

    print("\n" + "=" * 80)
    print("26-Neighbor Generation Test")
    print("=" * 80)

    # Center coordinates
    cx, cy, cz = 61, 35, 27
    depth = 7
    max_coord = (2 ** depth) - 1

    # Encode center
    center_prefix_correct = encode_morton_prefix_jax(
        jnp.int32(cx), jnp.int32(cy), jnp.int32(cz), depth
    )

    print(f"\n1. Center coordinates: ({cx}, {cy}, {cz})")
    print(f"   Center prefix (left-aligned): 0x{int(center_prefix_correct):016X}")

    # Generate neighbors with CORRECT input (left-aligned)
    neighbors_correct = get_26_neighbor_prefixes_jax(
        center_prefix_correct,
        depth,
        jnp.int32(max_coord)
    )

    print(f"\n2. Generated {len(neighbors_correct)} neighbor prefixes (INCLUDING center)")
    print(f"   First 5 prefixes:")
    for i in range(5):
        n_prefix = neighbors_correct[i]
        # Decode to check
        nx, ny, nz = decode_morton_prefix_jax(n_prefix, depth)
        print(f"     [{i}] 0x{int(n_prefix):016X} → ({int(nx)}, {int(ny)}, {int(nz)})")

    # Find center in neighbors (should be at index 13)
    print(f"\n3. Center should be at index 13:")
    center_idx_13 = neighbors_correct[13]
    cx_13, cy_13, cz_13 = decode_morton_prefix_jax(center_idx_13, depth)
    print(f"   Index 13: ({int(cx_13)}, {int(cy_13)}, {int(cz_13)})")
    print(f"   Match: {int(cx_13)==cx and int(cy_13)==cy and int(cz_13)==cz}")

    # NOW TEST WITH WRONG INPUT (shifted prefix - the bug)
    shift_amount = 63 - (depth * 3)
    center_prefix_wrong = center_prefix_correct >> jnp.uint64(shift_amount)

    print(f"\n4. WRONG: Pass shifted prefix to get_26_neighbor_prefixes_jax:")
    print(f"   Shifted center prefix: 0x{int(center_prefix_wrong):X} (small integer!)")

    neighbors_wrong = get_26_neighbor_prefixes_jax(
        center_prefix_wrong,
        depth,
        jnp.int32(max_coord)
    )

    print(f"\n5. Neighbors generated from WRONG input:")
    print(f"   First 5 prefixes:")
    for i in range(5):
        n_prefix = neighbors_wrong[i]
        nx, ny, nz = decode_morton_prefix_jax(n_prefix, depth)
        print(f"     [{i}] 0x{int(n_prefix):016X} → ({int(nx)}, {int(ny)}, {int(nz)})")

    print(f"\n6. Check if neighbors match:")
    print(f"   Correct [0]: 0x{int(neighbors_correct[0]):X}")
    print(f"   Wrong   [0]: 0x{int(neighbors_wrong[0]):X}")
    print(f"   Match: {int(neighbors_correct[0]) == int(neighbors_wrong[0])}")

    return neighbors_correct, neighbors_wrong


def test_prefix_lookup():
    """Test prefix table lookup logic."""

    print("\n" + "=" * 80)
    print("Prefix Table Lookup Test")
    print("=" * 80)

    depth = 7
    table_size = 8 ** depth  # 2,097,152 entries

    print(f"\n1. Table configuration:")
    print(f"   Depth: {depth}")
    print(f"   Size: {table_size:,} entries")
    print(f"   Memory: {table_size * 8 // (1024**2)} MB (for int32 arrays)")

    # Simulate a neighbor prefix (left-aligned)
    neighbor_prefix_full = encode_morton_prefix_jax(
        jnp.int32(62), jnp.int32(35), jnp.int32(27), depth
    )

    print(f"\n2. Neighbor prefix (left-aligned):")
    print(f"   Hex: 0x{int(neighbor_prefix_full):016X}")

    # Extract index (CORRECT way)
    shift_amount = 63 - (depth * 3)
    prefix_idx_correct = int(neighbor_prefix_full) >> shift_amount

    print(f"\n3. Extract index (shift right by {shift_amount}):")
    print(f"   Index: {prefix_idx_correct:,}")
    print(f"   Valid: {0 <= prefix_idx_correct < table_size}")

    # WRONG way (double shift - if input was already shifted)
    neighbor_prefix_already_shifted = prefix_idx_correct  # Simulate bug
    prefix_idx_wrong = neighbor_prefix_already_shifted >> shift_amount

    print(f"\n4. WRONG: Double shift (if input was already small integer):")
    print(f"   Input (already shifted): 0x{neighbor_prefix_already_shifted:X}")
    print(f"   Shift again by {shift_amount}: {prefix_idx_wrong}")
    print(f"   Valid: {0 <= prefix_idx_wrong < table_size}")
    print(f"   → This would look up wrong prefix in table!")

    return prefix_idx_correct, prefix_idx_wrong


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MORTON NEIGHBOR BUG DIAGNOSTIC")
    print("=" * 80)

    # Run tests
    morton_prefix, table_idx = test_morton_encoding()
    neighbors_correct, neighbors_wrong = test_neighbor_generation()
    idx_correct, idx_wrong = test_prefix_lookup()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - ROOT CAUSE ANALYSIS")
    print("=" * 80)

    print("\nThe bug is in morton_global_search.py line 661:")
    print("```python")
    print("# WRONG: Shifts Morton code right, creating small integer")
    print("center_prefix = lax.shift_right_logical(morton_query, jnp.uint64(shift_amount))")
    print("```")

    print("\nThis creates center_prefix as small integer (e.g., 0x7A3B5C)")
    print("But decode_morton_prefix_jax EXPECTS left-aligned uint64!")

    print("\nConsequence:")
    print("  1. decode_morton_prefix_jax receives 0x00000000007A3B5C")
    print("  2. Tries to decode from TOP bits (60-62, 57-59, ...)")
    print("  3. Finds all zeros → decodes to (0, 0, 0)!")
    print("  4. Generates 26 neighbors around (0, 0, 0) instead of actual position")
    print("  5. All particles search wrong region → 67% loss!")

    print("\n" + "=" * 80)
    print("FIX:")
    print("=" * 80)
    print("\nChange line 661 to:")
    print("```python")
    print("# Keep Morton code left-aligned")
    print("center_prefix = morton_query  # Don't shift!")
    print("```")

    print("\nThis passes left-aligned code to get_26_neighbor_prefixes_jax")
    print("which then correctly decodes and generates actual spatial neighbors.")

    print("\n" + "=" * 80)
