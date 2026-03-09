#!/usr/bin/env python3
"""
Verify if current search approach is fundamentally correct.

Test: For particles that ARE inside mesh elements, does the current
multi-level search find them?
"""

import numpy as np
from pathlib import Path

# Import JAX
import jax
import jax.numpy as jnp

from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.search.mesh_aligned_point_location import search_mesh_aligned_octree_batch
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

print(f"{'='*80}")
print("Search Correctness Verification")
print(f"{'='*80}\n")

# Load mesh
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'

print("Loading mesh...")
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    base_path=MESH_BASE_PATH,
    file_pattern=MESH_FILE_PATTERN,
    timestep_range=VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=False
)

node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
    node_positions,
    connectivity,
    velocity_sequence=velocity_sequence,
    verbose=False
)

print(f"  Loaded {connectivity.shape[0]:,} elements, {node_positions.shape[0]:,} nodes\n")

# Extract octree cells
print("Extracting octree cells...")
cells = extract_octree_cells_single(
    node_positions,
    connectivity,
    tolerance=1e-6,
    verbose=False
)
print(f"  Unique cells: {cells.n_cells:,}\n")

# Upload to GPU
print("Uploading to GPU...")
octree_gpu = upload_mesh_aligned_octree_to_gpu(
    node_positions=node_positions,
    connectivity=connectivity,
    octree_cells=cells,
    verbose=False
)
print("  Done\n")

# Generate test particles INSIDE elements (ground truth)
print("Generating test particles at element centroids...")
n_test_elements = 1000
np.random.seed(42)
test_elem_ids = np.random.choice(connectivity.shape[0], size=n_test_elements, replace=False)

test_positions = []
ground_truth_elements = []

for elem_id in test_elem_ids:
    node_ids = connectivity[elem_id]
    vertices = node_positions[node_ids]
    centroid = vertices.mean(axis=0)

    test_positions.append(centroid)
    ground_truth_elements.append(elem_id)

test_positions = np.array(test_positions, dtype=np.float32)
ground_truth_elements = np.array(ground_truth_elements, dtype=np.int32)

print(f"  Generated {len(test_positions):,} test positions at element centroids\n")

# Upload to GPU and search
test_positions_gpu = jnp.array(test_positions)

print(f"Searching for {len(test_positions):,} particles...")
found_elements, n_tests = search_mesh_aligned_octree_batch(
    test_positions_gpu,
    octree_gpu,
    max_tests=100
)
jax.block_until_ready((found_elements, n_tests))

# Convert back to CPU
found_elements_cpu = np.array(found_elements)
n_tests_cpu = np.array(n_tests)

# Analyze results
n_found = np.sum(found_elements_cpu >= 0)
n_correct = np.sum(found_elements_cpu == ground_truth_elements)
n_wrong_element = np.sum((found_elements_cpu >= 0) & (found_elements_cpu != ground_truth_elements))
n_not_found = np.sum(found_elements_cpu == -1)

print(f"  Found: {n_found:,} / {len(test_positions):,} ({100.0 * n_found / len(test_positions):.1f}%)\n")

print(f"{'='*80}")
print("RESULTS")
print(f"{'='*80}\n")

print(f"Ground truth: Particles placed at element centroids (MUST be inside)")
print(f"  Total particles: {len(test_positions):,}")
print(f"  Found correct element: {n_correct:,} ({100.0 * n_correct / len(test_positions):.1f}%)")
print(f"  Found wrong element: {n_wrong_element:,} ({100.0 * n_wrong_element / len(test_positions):.1f}%)")
print(f"  Not found at all: {n_not_found:,} ({100.0 * n_not_found / len(test_positions):.1f}%)")
print()

if n_correct == len(test_positions):
    print("✅ PERFECT: All particles found in correct elements!")
    print("   Current search approach is CORRECT.")
elif n_found == len(test_positions):
    print("⚠️  PARTIAL: All particles found, but some in wrong elements.")
    print("   This suggests neighboring cells are being checked (which is good!)")
    print("   Or point-in-tet has precision issues.")
else:
    print(f"❌ FAILED: {n_not_found:,} particles not found ({100.0 * n_not_found / len(test_positions):.1f}%)")
    print("   Current search approach is INCORRECT.")
    print()
    print("   Possible reasons:")
    print("   1. Particles in centroids should be in the parent cell we extract")
    print("   2. If not found, it means we're not checking the right cell")
    print("   3. This could be due to:")
    print("      - Grid index computation mismatch")
    print("      - Morton encoding mismatch")
    print("      - Level lookup mismatch")

print()

# Detailed analysis of not-found particles
if n_not_found > 0:
    print(f"{'='*80}")
    print(f"Analysis of NOT FOUND Particles (first 5)")
    print(f"{'='*80}\n")

    not_found_mask = found_elements_cpu == -1
    not_found_indices = np.where(not_found_mask)[0][:5]

    for idx in not_found_indices:
        elem_id = ground_truth_elements[idx]
        pos = test_positions[idx]

        print(f"Particle {idx} (should be in element {elem_id}):")
        print(f"  Position (centroid): {pos}")

        # Get element's cell info
        elem_cell_idx = cells.element_to_cells[elem_id]
        cell_level = cells.cell_levels[elem_cell_idx]
        cell_morton = cells.cell_morton_codes[elem_cell_idx]
        cell_grid_idx = cells.cell_grid_indices[elem_cell_idx]
        cell_size = cells.cell_sizes[elem_cell_idx]

        print(f"  Element's cell:")
        print(f"    Level: {cell_level}")
        print(f"    Grid: {cell_grid_idx}")
        print(f"    Morton: {cell_morton}")
        print(f"    Cell size: {cell_size}")

        # Compute what query would compute
        level_cell_size = np.array(octree_gpu.level_cell_sizes)[cell_level]

        i_query = int(np.floor(pos[0] / level_cell_size[0]))
        j_query = int(np.floor(pos[1] / level_cell_size[1]))
        k_query = int(np.floor(pos[2] / level_cell_size[2]))

        print(f"  Query would compute:")
        print(f"    Grid: ({i_query}, {j_query}, {k_query})")
        print(f"    Using cell size: {level_cell_size}")

        if (i_query, j_query, k_query) == tuple(cell_grid_idx):
            print(f"    ✅ Grid indices MATCH - query should find this cell")
            print(f"    → Problem might be in binary search or point-in-tet")
        else:
            print(f"    ❌ Grid indices MISMATCH - query will not find this cell")
            print(f"    → Problem is in grid computation")

            # Check if centroid is actually inside the cube
            cube_corner = cell_grid_idx * cell_size
            cube_max = cube_corner + cell_size

            inside_cube = np.all(pos >= cube_corner) and np.all(pos < cube_max)
            print(f"    → Is centroid inside parent cube? {inside_cube}")
            if not inside_cube:
                print(f"       Cube bounds: [{cube_corner}, {cube_max})")
                print(f"       Position: {pos}")

        print()

print(f"{'='*80}\n")
