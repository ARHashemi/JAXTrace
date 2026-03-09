#!/usr/bin/env python3
"""
Quick test of mesh-aligned Morton hybrid approach.

Verifies:
1. Cell extraction works
2. Morton structure builds correctly
3. GPU upload succeeds
4. Search returns valid element IDs
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single
from jaxtrace.gpu.search.mesh_aligned_morton_builder import (
    build_mesh_aligned_morton_structure,
    validate_mesh_aligned_morton_structure,
)
from jaxtrace.gpu.search.mesh_aligned_morton_search import (
    upload_mesh_aligned_morton_to_gpu,
    search_L2_mesh_aligned_morton_batch,
    search_L2_mesh_aligned_grid_neighbors_batch,
)
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
from jaxtrace.gpu.search.point_in_tet_methods import set_inverse_matrices_gpu
import jaxtrace.config as config

# Configuration
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
TIMESTEP_RANGE = (158, 159)  # Just one timestep for quick test
VELOCITY_FIELD_NAME = 'Displacement'
N_TEST_PARTICLES = 1000

# Search method: 'morton' or 'grid'
SEARCH_METHOD = 'grid'  # Use grid-based neighbor search (TRUE spatial neighbors)
SEARCH_RADIUS = 1  # For morton: 1D radius, For grid: 3D radius (1 = 3×3×3 = 27 cells)

config.POINT_IN_TET_METHOD = 'inverse'

print("=" * 80)
print(f"Mesh-Aligned Morton Hybrid Approach - Quick Test ({SEARCH_METHOD.upper()} search)")
print("=" * 80)

# Load mesh
print("\n[1/6] Loading mesh...")
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    base_path=MESH_BASE_PATH,
    file_pattern=MESH_FILE_PATTERN,
    timestep_range=TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=False
)

print(f"  Nodes: {node_positions.shape[0]:,}")
print(f"  Elements: {connectivity.shape[0]:,}")

# Deduplicate
print("\n[2/6] Deduplicating nodes...")
node_positions, connectivity, n_duplicates, _ = deduplicate_nodes(
    node_positions, connectivity, verbose=False
)
print(f"  After dedup: {node_positions.shape[0]:,} nodes, {connectivity.shape[0]:,} elements")

# Extract mesh-aligned octree cells
print("\n[3/6] Extracting mesh-aligned octree cells...")
mesh_octree_cells = extract_octree_cells_single(
    node_positions, connectivity, tolerance=1e-6, verbose=True
)

# Build mesh-aligned Morton structure
print("\n[4/6] Building mesh-aligned Morton structure...")
mesh_aligned_morton_struct = build_mesh_aligned_morton_structure(
    node_positions, connectivity, mesh_octree_cells=mesh_octree_cells, verbose=True
)

# Validate
print("\n[5/6] Validating structure...")
is_valid = validate_mesh_aligned_morton_structure(
    mesh_aligned_morton_struct, connectivity, verbose=True
)

if not is_valid:
    print("\n❌ VALIDATION FAILED!")
    sys.exit(1)

# Precompute inverse matrices for point-in-tet
print("\n[6/7] Precomputing inverse matrices...")
M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)
M_inv_gpu = jax.device_put(M_inv_array)
p0_gpu = jax.device_put(p0_array)
set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)
print(f"  ✅ Precomputed {M_inv_array.shape[0]:,} inverse matrices")

# Upload to GPU
print("\n[7/7] Testing GPU search...")
mesh_aligned_morton_gpu = upload_mesh_aligned_morton_to_gpu(
    node_positions, connectivity, mesh_aligned_morton_struct, verbose=True
)

# Generate test particles (random positions inside mesh bounding box)
print(f"\n  Generating {N_TEST_PARTICLES:,} random test particles...")
print(f"  Using {SEARCH_METHOD.upper()} search with radius={SEARCH_RADIUS}")

if SEARCH_METHOD == 'grid':
    grid_width = 2 * SEARCH_RADIUS + 1
    expected_cells = grid_width ** 3
    print(f"    Grid size: {grid_width}×{grid_width}×{grid_width} = {expected_cells} cells")
else:
    expected_cells = 2 * SEARCH_RADIUS + 1
    print(f"    Morton cells: {expected_cells} cells (along 1D curve)")

bbox_min = node_positions.min(axis=0)
bbox_max = node_positions.max(axis=0)
bbox_size = bbox_max - bbox_min

# Sample uniformly in bbox
rng = np.random.default_rng(42)
positions = bbox_min + rng.random((N_TEST_PARTICLES, 3)) * bbox_size
positions = positions.astype(np.float32)
positions_gpu = jax.device_put(positions)

# Search
print(f"  Searching...")

if SEARCH_METHOD == 'grid':
    elem_ids = search_L2_mesh_aligned_grid_neighbors_batch(
        positions_gpu,
        mesh_aligned_morton_gpu,
        grid_radius=jnp.int32(SEARCH_RADIUS),
        max_tests_per_cell=jnp.int32(256)
    )
else:
    elem_ids = search_L2_mesh_aligned_morton_batch(
        positions_gpu,
        mesh_aligned_morton_gpu,
        search_radius=jnp.int32(SEARCH_RADIUS),
        max_tests_per_cell=jnp.int32(256)
    )

elem_ids = np.array(elem_ids)
n_found = np.sum(elem_ids >= 0)
success_rate = n_found / N_TEST_PARTICLES * 100

print(f"\n{'='*80}")
print(f"RESULTS - {SEARCH_METHOD.upper()} SEARCH")
print(f"{'='*80}")
print(f"  Test particles: {N_TEST_PARTICLES:,}")
print(f"  Found: {n_found:,} ({success_rate:.1f}%)")
print(f"  Not found: {N_TEST_PARTICLES - n_found:,}")
print(f"\n  Cells: {mesh_aligned_morton_struct.n_cells:,}")
print(f"  Elements per cell: {mesh_aligned_morton_struct.elements_per_cell_mean:.1f} (mean)")
print(f"  Search radius: {SEARCH_RADIUS}")
print(f"  Cells searched: {expected_cells} per particle")
print(f"  Expected tests: ~{expected_cells * mesh_aligned_morton_struct.elements_per_cell_mean:.0f} per particle")

# Expected success rate (rough estimate based on mesh coverage)
# Random particles in bbox: ~30-50% inside mesh
# Of those inside: expect ~98% found with proper neighbor search
print(f"\n  Expected retention for particles inside mesh: ~98%")
print(f"  Actual success rate includes void regions: {success_rate:.1f}%")

if SEARCH_METHOD == 'grid':
    print(f"\n  Note: Grid search finds TRUE spatial neighbors (3D cube)")
else:
    print(f"\n  Note: Morton search follows 1D curve (may miss spatial neighbors)")

if n_found > 0:
    print(f"\n✅ TEST PASSED: Search returns valid element IDs")
else:
    print(f"\n❌ TEST FAILED: No particles found")
    sys.exit(1)

print(f"\n{'='*80}")
print(f"✅ All tests passed!")
print(f"{'='*80}\n")
