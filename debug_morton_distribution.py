#!/usr/bin/env python3
"""
Debug script to analyze Morton code distribution and hash collisions.
This helps understand why hash table insertion is failing even with scrambling.
"""

import numpy as np
from jaxtrace.fields.hash_octree import (
    hash_morton_scrambled,
    next_prime,
    EMPTY_SLOT,
    MAX_PROBES
)
from jaxtrace.fields.fine_octree_builder import build_fine_octree
from jaxtrace.io import read_ugrid_vtk
from pathlib import Path

print("=" * 80)
print("MORTON CODE DISTRIBUTION ANALYSIS")
print("=" * 80)

# Load the actual mesh
mesh_file = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_8.pvtu")
print(f"\nLoading mesh: {mesh_file}")
mesh_data = read_ugrid_vtk(str(mesh_file))
print(f"Mesh: {len(mesh_data.points)} points, {len(mesh_data.cells)} cells")

# Build octree to get leaf nodes
print("\nBuilding fine octree...")
bbox_min = np.min(mesh_data.points, axis=0).astype(np.float32)
bbox_max = np.max(mesh_data.points, axis=0).astype(np.float32)

fine_octree = build_fine_octree(
    mesh_data.points,
    mesh_data.cells,
    bbox_min,
    bbox_max,
    max_depth=12,
    max_elements_per_node=30
)

# Extract leaf morton codes
is_leaf = fine_octree.node_is_leaf
leaf_indices = np.where(is_leaf)[0]
morton_codes_np = np.asarray(fine_octree.node_morton_codes, dtype=np.uint64)
leaf_morton_codes = morton_codes_np[leaf_indices]

n_leaves = len(leaf_morton_codes)
print(f"\nTotal leaves: {n_leaves:,}")

# Test different load factors
load_factors = [0.3, 0.4, 0.5, 0.6, 0.7, 0.77]

print("\n" + "=" * 80)
print("HASH DISTRIBUTION ANALYSIS")
print("=" * 80)

for load_factor in load_factors:
    print(f"\n{'='*80}")
    print(f"LOAD FACTOR: {load_factor}")
    print(f"{'='*80}")

    # Compute table size
    table_size = next_prime(int(np.ceil(n_leaves / load_factor)))
    print(f"Table size: {table_size:,} (prime)")
    print(f"Actual load: {n_leaves / table_size:.3f}")

    # Initialize hash table
    hash_table = np.full(table_size, EMPTY_SLOT, dtype=np.uint64)
    probe_counts = []

    # Try inserting all leaves
    failed_at = -1
    collision_histogram = {}

    for i, morton_code in enumerate(leaf_morton_codes):
        initial_slot = hash_morton_scrambled(morton_code, table_size)

        # Count probes needed
        probes_needed = 0
        inserted = False

        for probe in range(MAX_PROBES):
            current_slot = (initial_slot + probe) % table_size

            if hash_table[current_slot] == EMPTY_SLOT:
                hash_table[current_slot] = morton_code
                probes_needed = probe + 1
                inserted = True
                break

        if inserted:
            probe_counts.append(probes_needed)
            if probes_needed not in collision_histogram:
                collision_histogram[probes_needed] = 0
            collision_histogram[probes_needed] += 1
        else:
            failed_at = i
            print(f"\n❌ INSERTION FAILED at leaf {i}/{n_leaves}")
            print(f"   Morton code: {morton_code}")
            print(f"   Initial slot: {initial_slot}")

            # Show what's in the collision cluster
            print(f"\n   Collision cluster:")
            for probe in range(min(MAX_PROBES + 10, table_size)):
                slot = (initial_slot + probe) % table_size
                if hash_table[slot] == EMPTY_SLOT:
                    print(f"      Probe {probe}: EMPTY (first empty slot found!)")
                    break
                else:
                    # Compute where this morton code SHOULD have hashed
                    occupying_code = hash_table[slot]
                    original_slot = hash_morton_scrambled(occupying_code, table_size)
                    displacement = (slot - original_slot) % table_size
                    print(f"      Probe {probe}: slot {slot} occupied by morton {occupying_code} (displaced by {displacement})")

            break

    if failed_at == -1:
        print(f"\n✅ ALL {n_leaves:,} LEAVES INSERTED SUCCESSFULLY")

        # Statistics
        probe_counts = np.array(probe_counts)
        print(f"\n📊 Probe Statistics:")
        print(f"   Mean probes: {np.mean(probe_counts):.2f}")
        print(f"   Median probes: {np.median(probe_counts):.1f}")
        print(f"   Max probes: {np.max(probe_counts)}")
        print(f"   Std dev: {np.std(probe_counts):.2f}")

        print(f"\n📊 Collision Histogram:")
        for probes in sorted(collision_histogram.keys())[:10]:
            count = collision_histogram[probes]
            pct = 100 * count / n_leaves
            bar = '█' * int(pct)
            print(f"   {probes:2d} probe(s): {count:6,} ({pct:5.2f}%) {bar}")

        if len(collision_histogram) > 10:
            print(f"   ... ({len(collision_histogram) - 10} more entries)")
            max_probes = max(collision_histogram.keys())
            print(f"   Max: {max_probes} probes ({collision_histogram[max_probes]} leaves)")
    else:
        print(f"\n❌ FAILED after {failed_at:,} insertions ({100*failed_at/n_leaves:.1f}%)")

        if probe_counts:
            probe_counts = np.array(probe_counts)
            print(f"\n📊 Probe Statistics (before failure):")
            print(f"   Mean probes: {np.mean(probe_counts):.2f}")
            print(f"   Max probes: {np.max(probe_counts)}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
