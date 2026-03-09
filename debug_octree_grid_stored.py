#!/usr/bin/env python3
"""
Debug: Check what grid indices are actually STORED in the octree
vs what we'd compute from the same element's centroid.
"""

import numpy as np
from pathlib import Path

from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import (
    extract_octree_cells_single,
    find_axis_aligned_edges_single,
    find_parent_cube,
)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

print(f"{'='*80}")
print("Octree Grid Index Storage Debug")
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

print(f"  Loaded {connectivity.shape[0]:,} elements\n")

# Extract octree cells
print("Extracting octree cells...")
cells = extract_octree_cells_single(
    node_positions,
    connectivity,
    tolerance=1e-6,
    verbose=False
)
print(f"  Extracted {cells.n_cells:,} cells\n")

# Test the problematic element from verify_search_correctness.log
# Particle 0 (Element 2232962)
elem_id = 2232962

node_ids = connectivity[elem_id]
vertices = node_positions[node_ids]
centroid = vertices.mean(axis=0)

# Get stored cell info
cell_idx = cells.element_to_cells[elem_id]
stored_grid = cells.cell_grid_indices[cell_idx]
stored_level = cells.cell_levels[cell_idx]
stored_cell_size = cells.cell_sizes[cell_idx]

print(f"Element {elem_id}:")
print(f"  Centroid: {centroid}")
print()
print(f"  Stored in octree:")
print(f"    Cell index: {cell_idx}")
print(f"    Grid indices: {stored_grid}")
print(f"    Level: {stored_level}")
print(f"    Cell size: {stored_cell_size}")
print()

# Recompute using current code
cell_size, level = find_axis_aligned_edges_single(vertices, tolerance=1e-6)
cube_corner, cube_center, i, j, k = find_parent_cube(vertices, cell_size, tolerance=1e-6)

print(f"  Recomputed NOW with current code:")
print(f"    Cell size: {cell_size}")
print(f"    Level: {level}")
print(f"    Grid indices: [{i}, {j}, {k}]")
print(f"    Cube corner: {cube_corner}")
print()

# Check if centroid is inside the stored cube
stored_cube_corner = stored_grid * stored_cell_size
stored_cube_max = stored_cube_corner + stored_cell_size

inside_stored = np.all(centroid >= stored_cube_corner) and np.all(centroid < stored_cube_max)

print(f"  Is centroid inside STORED cube?")
print(f"    Stored cube: [{stored_cube_corner}, {stored_cube_max})")
print(f"    Centroid: {centroid}")
print(f"    Inside? {inside_stored}")
print()

# Check if centroid is inside the recomputed cube
recomputed_cube_corner = np.array([i, j, k]) * cell_size
recomputed_cube_max = recomputed_cube_corner + cell_size

inside_recomputed = np.all(centroid >= recomputed_cube_corner) and np.all(centroid < recomputed_cube_max)

print(f"  Is centroid inside RECOMPUTED cube?")
print(f"    Recomputed cube: [{recomputed_cube_corner}, {recomputed_cube_max})")
print(f"    Centroid: {centroid}")
print(f"    Inside? {inside_recomputed}")
print()

# Compare grid indices
if (i, j, k) == tuple(stored_grid):
    print(f"  ✅ Grid indices MATCH between stored and recomputed")
else:
    print(f"  ❌ Grid indices MISMATCH!")
    print(f"     Stored: {stored_grid}")
    print(f"     Recomputed: [{i}, {j}, {k}]")
    print(f"     Diff: [{i - stored_grid[0]}, {j - stored_grid[1]}, {k - stored_grid[2]}]")
print()

print(f"{'='*80}")
print("CONCLUSION")
print(f"{'='*80}\n")

if not inside_stored and inside_recomputed:
    print("✅ Current code is CORRECT!")
    print("   The octree was built with OLD/BROKEN code.")
    print("   Need to rebuild octree with current code.")
elif inside_stored and not inside_recomputed:
    print("❌ Current code is BROKEN!")
    print("   The octree was built with correct code, but current code is wrong.")
elif not inside_stored and not inside_recomputed:
    print("⚠️  BOTH are broken!")
    print("   Neither stored nor recomputed grids contain the centroid.")
    print("   Fundamental algorithm issue.")
else:
    print("🤔 Both are correct?")
    print("   This shouldn't happen if verify_search_correctness.py failed.")

print()
