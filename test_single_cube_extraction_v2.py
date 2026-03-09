#!/usr/bin/env python3
"""
Test corrected single-cube cell extraction with level-aware Morton encoding.

Expected results (v3):
- ~1 cell per element (not changed)
- ~5-6 elements per cell (NOW FIXED!)
- More unique cells (separated by refinement level)
"""

import numpy as np
from pathlib import Path
import time
import sys

# Import directly to avoid JAX dependency
sys.path.insert(0, '/home/arhashemi/Workspace/welding/JAXTrace')
import importlib.util
spec = importlib.util.spec_from_file_location(
    "mesh_aligned_octree_single_cell",
    "/home/arhashemi/Workspace/welding/JAXTrace/jaxtrace/gpu/search/mesh_aligned_octree_single_cell.py"
)
octree_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(octree_module)
extract_octree_cells_single = octree_module.extract_octree_cells_single

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

# Load mesh
print("Loading mesh...")
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

# Extract cells using corrected approach v3
t0 = time.time()

cells = extract_octree_cells_single(
    node_positions,
    connectivity,
    tolerance=1e-6,
    verbose=True
)

t_extract = time.time() - t0

print(f"\n{'='*80}")
print("VALIDATION (Version 3)")
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
print(f"    Expected: More than v2 (248k) due to level separation")
print(f"    Actual:   {cells.n_cells:,}")
if cells.n_cells > 300000:
    print(f"    ✅ Increased as expected!")
else:
    print(f"    ⚠  Still similar to v2")

# Compare to previous versions
print(f"\n{'='*80}")
print("COMPARISON TO PREVIOUS VERSIONS")
print(f"{'='*80}")
print(f"\nv1 (bbox overlap, 8 cells):")
print(f"  Cells: ~652k")
print(f"  Cells per element: 8.0")
print(f"  Elements per cell: 37.4")
print(f"  Searchability: 2.4% ❌")

print(f"\nv2 (single parent cube, morton only):")
print(f"  Cells: 248,321")
print(f"  Cells per element: 1.00")
print(f"  Elements per cell: 12.27 ❌")
print(f"  Problem: Level collisions")

print(f"\nv3 (single parent cube, morton + level):")
print(f"  Cells: {cells.n_cells:,}")
print(f"  Cells per element: {cells.cells_per_element_mean:.2f}")
print(f"  Elements per cell: {cells.elements_per_cell_mean:.2f}")
print(f"  Searchability: TBD (needs GPU test)")

print(f"\n{'='*80}\n")
