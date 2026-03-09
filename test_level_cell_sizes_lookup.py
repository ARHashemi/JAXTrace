#!/usr/bin/env python3
"""
Test if level_cell_sizes lookup works correctly.
"""

import numpy as np
from pathlib import Path

# Import JAX
import jax
import jax.numpy as jnp

from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

print(f"{'='*80}")
print("Level Cell Sizes Lookup Test")
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

# Check level_cell_sizes
print(f"{'='*80}")
print("Level Cell Sizes in GPU Structure")
print(f"{'='*80}\n")

level_cell_sizes = np.array(octree_gpu.level_cell_sizes)
print(f"level_cell_sizes shape: {level_cell_sizes.shape}")
print()

for level in range(level_cell_sizes.shape[0]):
    cell_size = level_cell_sizes[level]
    if np.any(cell_size != 0):
        print(f"Level {level:2d}: {cell_size}")

print(f"\n{'='*80}")
print("Verification: Compare with Phase 2 data")
print(f"{'='*80}\n")

unique_levels = np.unique(cells.cell_levels)
for level in sorted(unique_levels):
    level_mask = cells.cell_levels == level
    first_cell_size = cells.cell_sizes[level_mask][0]
    gpu_cell_size = level_cell_sizes[level]

    print(f"Level {level:2d}:")
    print(f"  Phase 2 (first cell): {first_cell_size}")
    print(f"  GPU structure:        {gpu_cell_size}")
    if np.allclose(first_cell_size, gpu_cell_size):
        print(f"  ✅ MATCH")
    else:
        print(f"  ❌ MISMATCH!")
    print()

print(f"{'='*80}\n")
