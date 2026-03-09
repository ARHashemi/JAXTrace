#!/usr/bin/env python3
"""
Check mesh bbox and test with a found particle.
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
print("Bbox and Found Particle Analysis")
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

# Check actual mesh bbox
bbox_min = np.min(node_positions, axis=0)
bbox_max = np.max(node_positions, axis=0)

print(f"Mesh bounding box:")
print(f"  X: [{bbox_min[0]:.10f}, {bbox_max[0]:.10f}] (range: {bbox_max[0] - bbox_min[0]:.10f})")
print(f"  Y: [{bbox_min[1]:.10f}, {bbox_max[1]:.10f}] (range: {bbox_max[1] - bbox_min[1]:.10f})")
print(f"  Z: [{bbox_min[2]:.10f}, {bbox_max[2]:.10f}] (range: {bbox_max[2] - bbox_min[2]:.10f})")
print()

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
    octree_cells=cells
)
print("  Done\n")

# Generate test particles
np.random.seed(42)
n_particles = 10000
particle_positions_cpu = np.random.uniform(
    low=bbox_min,
    high=bbox_max,
    size=(n_particles, 3)
).astype(np.float32)

particle_positions_gpu = jnp.array(particle_positions_cpu)

# Search
print(f"Searching for {n_particles:,} particles...")
found_elements, n_tests = search_mesh_aligned_octree_batch(
    particle_positions_gpu,
    octree_gpu,
    max_tests=100
)
jax.block_until_ready((found_elements, n_tests))

# Convert back to CPU for analysis
found_elements_cpu = np.array(found_elements)
n_tests_cpu = np.array(n_tests)

n_found = np.sum(found_elements_cpu >= 0)
print(f"  Found: {n_found:,} / {n_particles:,} ({100.0 * n_found / n_particles:.1f}%)\n")

# Find a FOUND particle
found_mask = found_elements_cpu >= 0
found_indices = np.where(found_mask)[0]

if len(found_indices) > 0:
    # Pick first found particle
    idx = found_indices[0]
    test_pos = particle_positions_cpu[idx]
    elem_id = found_elements_cpu[idx]
    n_test = n_tests_cpu[idx]

    print(f"{'='*80}")
    print(f"Analysis of FOUND Particle #{idx}")
    print(f"{'='*80}\n")

    print(f"Position: {test_pos}")
    print(f"Found in element: {elem_id}")
    print(f"Number of tests: {n_test}\n")

    # Get element info
    elem_cell_idx = cells.element_to_cells[elem_id]
    cell_morton = cells.cell_morton_codes[elem_cell_idx]
    cell_level = cells.cell_levels[elem_cell_idx]
    cell_size = cells.cell_sizes[elem_cell_idx]
    cell_grid_idx = cells.cell_grid_indices[elem_cell_idx]

    print(f"Element's cell:")
    print(f"  Level: {cell_level}")
    print(f"  Grid indices: {cell_grid_idx}")
    print(f"  Cell size: {cell_size}")
    print(f"  Morton code: {cell_morton}\n")

    # Compute what the query computed
    base_x = 1.28
    base_y = 1.3084106445312499
    base_z = 1.28

    offset = (1 << 19)
    max_coord = (1 << 20)

    cell_size_x_query = base_x / (2.0 ** cell_level)
    cell_size_y_query = base_y / (2.0 ** cell_level)
    cell_size_z_query = base_z / (2.0 ** cell_level)

    i_query = int(np.floor(test_pos[0] / cell_size_x_query))
    j_query = int(np.floor(test_pos[1] / cell_size_y_query))
    k_query = int(np.floor(test_pos[2] / cell_size_z_query))

    print(f"Query computation at level {cell_level}:")
    print(f"  Cell sizes: X={cell_size_x_query:.10f}, Y={cell_size_y_query:.10f}, Z={cell_size_z_query:.10f}")
    print(f"  Grid indices: ({i_query}, {j_query}, {k_query})")
    print(f"  Actual element grid: ({cell_grid_idx[0]}, {cell_grid_idx[1]}, {cell_grid_idx[2]})")

    if (i_query, j_query, k_query) == tuple(cell_grid_idx):
        print(f"  ✅ MATCH! Query correctly found this element.\n")
    else:
        print(f"  ❌ MISMATCH! This should not have been found.\n")

    # Check actual cell sizes vs query
    print(f"Cell size comparison:")
    print(f"  Actual:  X={cell_size[0]:.10f}, Y={cell_size[1]:.10f}, Z={cell_size[2]:.10f}")
    print(f"  Query:   X={cell_size_x_query:.10f}, Y={cell_size_y_query:.10f}, Z={cell_size_z_query:.10f}")
    print(f"  Error:   X={abs(cell_size[0] - cell_size_x_query):.10e}, Y={abs(cell_size[1] - cell_size_y_query):.10e}, Z={abs(cell_size[2] - cell_size_z_query):.10e}")

# Now test some unfound particles
unfound_mask = found_elements_cpu == -1
unfound_indices = np.where(unfound_mask)[0]

if len(unfound_indices) > 0:
    print(f"\n{'='*80}")
    print(f"Analysis of UNFOUND Particles (first 3)")
    print(f"{'='*80}\n")

    for i, idx in enumerate(unfound_indices[:3]):
        test_pos = particle_positions_cpu[idx]
        print(f"Unfound particle #{idx}:")
        print(f"  Position: {test_pos}")

        # Check if in bbox
        in_bbox = np.all(test_pos >= bbox_min) and np.all(test_pos <= bbox_max)
        print(f"  In bbox: {in_bbox}")

        if not in_bbox:
            print(f"  ⚠  Particle is OUTSIDE mesh bbox!")
        print()

print(f"{'='*80}\n")
