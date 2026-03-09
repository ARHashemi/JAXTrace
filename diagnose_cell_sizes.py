#!/usr/bin/env python3
"""
Diagnose cell size variation across refinement levels.

This will help us understand if we can use a single representative cell size
per level, or if we need more sophisticated handling.
"""

import numpy as np
from pathlib import Path
import time
import sys

# Import JAX first
print("Importing JAX...")
import jax
import jax.numpy as jnp

print("Loading modules...")

from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

print(f"{'='*80}")
print("Cell Size Analysis Across Refinement Levels")
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
t0 = time.time()
cells = extract_octree_cells_single(
    node_positions,
    connectivity,
    tolerance=1e-6,
    verbose=False
)
t_extract = time.time() - t0

print(f"  Extraction time: {t_extract:.2f}s")
print(f"  Unique cells: {cells.n_cells:,}\n")

# Analyze cell sizes per level
print(f"{'='*80}")
print("Cell Size Analysis Per Level")
print(f"{'='*80}\n")

unique_levels = np.unique(cells.cell_levels)
print(f"Refinement levels: {sorted(unique_levels)}\n")

for level in sorted(unique_levels):
    level_mask = cells.cell_levels == level
    level_cell_sizes = cells.cell_sizes[level_mask]

    n_cells_at_level = np.sum(level_mask)

    # Statistics for each dimension
    mean_size = np.mean(level_cell_sizes, axis=0)
    std_size = np.std(level_cell_sizes, axis=0)
    min_size = np.min(level_cell_sizes, axis=0)
    max_size = np.max(level_cell_sizes, axis=0)

    print(f"Level {level:2d} ({n_cells_at_level:6,} cells):")
    print(f"  X: mean={mean_size[0]:.10f}, std={std_size[0]:.10f}, [{min_size[0]:.10f}, {max_size[0]:.10f}]")
    print(f"  Y: mean={mean_size[1]:.10f}, std={std_size[1]:.10f}, [{min_size[1]:.10f}, {max_size[1]:.10f}]")
    print(f"  Z: mean={mean_size[2]:.10f}, std={std_size[2]:.10f}, [{min_size[2]:.10f}, {max_size[2]:.10f}]")

    # Check if anisotropic
    ratio_xy = mean_size[0] / mean_size[1] if mean_size[1] > 0 else 0
    ratio_xz = mean_size[0] / mean_size[2] if mean_size[2] > 0 else 0
    ratio_yz = mean_size[1] / mean_size[2] if mean_size[2] > 0 else 0

    print(f"  Anisotropy: X/Y={ratio_xy:.4f}, X/Z={ratio_xz:.4f}, Y/Z={ratio_yz:.4f}")

    # Check variability within level
    cv_x = std_size[0] / mean_size[0] if mean_size[0] > 0 else 0
    cv_y = std_size[1] / mean_size[1] if mean_size[1] > 0 else 0
    cv_z = std_size[2] / mean_size[2] if mean_size[2] > 0 else 0

    print(f"  Variability (CV): X={cv_x:.6f}, Y={cv_y:.6f}, Z={cv_z:.6f}")

    # Show expected theoretical size at this level (assuming base=1.0)
    theoretical_size = 1.0 / (2.0 ** level)
    print(f"  Theoretical size (1.0 / 2^{level}): {theoretical_size:.10f}")
    print()

# Key insight check
print(f"{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}\n")

print("1. Is cell size constant within each level?")
for level in sorted(unique_levels):
    level_mask = cells.cell_levels == level
    level_cell_sizes = cells.cell_sizes[level_mask]

    # Check if all cells at this level have identical sizes
    unique_sizes_at_level = np.unique(level_cell_sizes, axis=0)

    if len(unique_sizes_at_level) == 1:
        print(f"   Level {level:2d}: YES - All cells have identical size {unique_sizes_at_level[0]}")
    else:
        print(f"   Level {level:2d}: NO - {len(unique_sizes_at_level)} unique cell sizes")
        if len(unique_sizes_at_level) <= 5:
            for i, size in enumerate(unique_sizes_at_level):
                count = np.sum(np.all(level_cell_sizes == size, axis=1))
                print(f"      Size {i+1}: {size} ({count} cells)")

print("\n2. What base size should we use?")
# Try to infer base size from level 8 (coarsest)
coarsest_level = min(unique_levels)
level_mask = cells.cell_levels == coarsest_level
mean_size_coarsest = np.mean(cells.cell_sizes[level_mask], axis=0)

inferred_base_x = mean_size_coarsest[0] * (2 ** coarsest_level)
inferred_base_y = mean_size_coarsest[1] * (2 ** coarsest_level)
inferred_base_z = mean_size_coarsest[2] * (2 ** coarsest_level)

print(f"   From level {coarsest_level} cells:")
print(f"     Base X: {inferred_base_x:.10f}")
print(f"     Base Y: {inferred_base_y:.10f}")
print(f"     Base Z: {inferred_base_z:.10f}")

print("\n3. Verification: Do inferred base sizes predict cell sizes at other levels?")
for level in sorted(unique_levels):
    level_mask = cells.cell_levels == level
    mean_size = np.mean(cells.cell_sizes[level_mask], axis=0)

    predicted_x = inferred_base_x / (2 ** level)
    predicted_y = inferred_base_y / (2 ** level)
    predicted_z = inferred_base_z / (2 ** level)

    error_x = abs(mean_size[0] - predicted_x) / mean_size[0] * 100
    error_y = abs(mean_size[1] - predicted_y) / mean_size[1] * 100
    error_z = abs(mean_size[2] - predicted_z) / mean_size[2] * 100

    print(f"   Level {level:2d}: Error X={error_x:.4f}%, Y={error_y:.4f}%, Z={error_z:.4f}%")

print(f"\n{'='*80}")
print("RECOMMENDATION")
print(f"{'='*80}\n")

# Check if we can use a simple formula
max_cv = 0
for level in sorted(unique_levels):
    level_mask = cells.cell_levels == level
    level_cell_sizes = cells.cell_sizes[level_mask]
    mean_size = np.mean(level_cell_sizes, axis=0)
    std_size = np.std(level_cell_sizes, axis=0)
    cv = np.max(std_size / mean_size)
    max_cv = max(max_cv, cv)

if max_cv < 0.01:
    print("✅ Cell sizes are highly uniform within each level (CV < 1%)")
    print("   Recommendation: Use mean cell size per level for lookups")
    print(f"\n   Suggested base sizes:")
    print(f"     BASE_X = {inferred_base_x:.10f}")
    print(f"     BASE_Y = {inferred_base_y:.10f}")
    print(f"     BASE_Z = {inferred_base_z:.10f}")
    print(f"\n   For query at level L, use:")
    print(f"     cell_size_x = BASE_X / (2^L)")
    print(f"     cell_size_y = BASE_Y / (2^L)")
    print(f"     cell_size_z = BASE_Z / (2^L)")
else:
    print(f"⚠  Cell sizes vary within levels (max CV = {max_cv:.4f})")
    print("   Recommendation: Store per-level cell sizes in GPU structure")

print(f"\n{'='*80}\n")
