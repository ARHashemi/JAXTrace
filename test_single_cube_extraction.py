#!/usr/bin/env python3
"""
Test corrected single-cube cell extraction.

Expected results:
- ~1 cell per element (not 8)
- ~5-6 elements per cell (not 37)
- ~500k-600k unique cells (parent cubes)
"""

import numpy as np
from pathlib import Path
import time

from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single

# Load mesh
print("Loading mesh...")
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'

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

print(f"  Loaded {connectivity.shape[0]:,} elements, {node_positions.shape[0]:,} nodes")
print(f"  Removed {n_duplicates_removed:,} duplicate nodes")

# Extract cells using corrected approach
t0 = time.time()

cells = extract_octree_cells_single(
    node_positions,
    connectivity,
    tolerance=1e-6,
    verbose=True
)

t_extract = time.time() - t0

print(f"\n{'='*80}")
print("VALIDATION")
print(f"{'='*80}")

print(f"\nExtraction time: {t_extract:.2f}s")

print(f"\nResults:")
print(f"  Unique cells: {cells.n_cells:,}")
print(f"  Cells per element: {cells.cells_per_element_mean:.2f}")
print(f"  Elements per cell: {cells.elements_per_cell_mean:.2f}")

print(f"\nExpected vs Actual:")
print(f"  Cells per element:")
print(f"    Expected: ~1.0")
print(f"    Actual:   {cells.cells_per_element_mean:.2f}")
if abs(cells.cells_per_element_mean - 1.0) < 0.1:
    print(f"    ✅ CORRECT!")
else:
    print(f"    ❌ WRONG!")

print(f"\n  Elements per cell:")
print(f"    Expected: ~5-6 (Kuhn subdivision)")
print(f"    Actual:   {cells.elements_per_cell_mean:.2f}")
if 4.0 <= cells.elements_per_cell_mean <= 8.0:
    print(f"    ✅ REASONABLE!")
else:
    print(f"    ❌ UNEXPECTED!")

print(f"\n  Number of unique cells:")
print(f"    Expected: ~500k-600k parent cubes")
print(f"    Actual:   {cells.n_cells:,}")
if 400000 <= cells.n_cells <= 700000:
    print(f"    ✅ REASONABLE!")
else:
    print(f"    ⚠  Different than expected")

# Compare to old wrong approach
print(f"\n{'='*80}")
print("COMPARISON TO OLD (WRONG) APPROACH")
print(f"{'='*80}")
print(f"\nOld approach (bbox overlap, 8 cells):")
print(f"  Cells: ~652k")
print(f"  Cells per element: 8.0")
print(f"  Elements per cell: 37.4")
print(f"  Searchability: 2.4% ❌")

print(f"\nNew approach (single parent cube):")
print(f"  Cells: {cells.n_cells:,}")
print(f"  Cells per element: {cells.cells_per_element_mean:.2f}")
print(f"  Elements per cell: {cells.elements_per_cell_mean:.2f}")
print(f"  Searchability: TBD (needs GPU test)")

print(f"\n{'='*80}\n")
