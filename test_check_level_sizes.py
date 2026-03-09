#!/usr/bin/env python3
"""
Check if level_cell_sizes is populated correctly
"""

import numpy as np
from pathlib import Path

from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single
from jaxtrace.gpu.search.mesh_aligned_octree_with_neighbor_table import (
    add_neighbor_table_to_octree,
    upload_octree_with_neighbors_to_gpu
)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

# Load mesh
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
node_positions, connectivity, _ = load_velocity_sequence_from_pvtu(
    base_path=MESH_BASE_PATH,
    file_pattern="featurelessAvtk_{timestep}.pvtu",
    timestep_range=(158, 159),
    field_name='Displacement',
    verbose=False
)
node_positions, connectivity, _, _ = deduplicate_nodes(
    node_positions, connectivity, velocity_sequence=None, verbose=False
)

# Extract octree
octree_cells = extract_octree_cells_single(node_positions, connectivity, verbose=False)

# Build neighbor table
octree_with_neighbors = add_neighbor_table_to_octree(octree_cells, verbose=False)

# Upload
octree_gpu = upload_octree_with_neighbors_to_gpu(connectivity, node_positions, octree_with_neighbors, verbose=False)

# Check level_cell_sizes
print("level_cell_sizes shape:", octree_gpu.level_cell_sizes.shape)
print("level_cell_sizes:")
for level in range(octree_gpu.level_cell_sizes.shape[0]):
    size = octree_gpu.level_cell_sizes[level]
    print(f"  Level {level}: {size}")

# Check which levels actually exist
unique_levels = np.unique(octree_cells.cell_levels)
print(f"\nActual levels in octree: {unique_levels}")
