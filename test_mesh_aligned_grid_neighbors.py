#!/usr/bin/env python3
"""
Test grid-based neighbor search vs Morton radius search.

Compares:
1. Morton radius search (searches along 1D Morton curve - WRONG for sparse cells)
2. Grid neighbor search (searches 3D grid neighbors - CORRECT for spatial neighbors)
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
TIMESTEP_RANGE = (158, 159)
N_TEST_PARTICLES = 1000

config.POINT_IN_TET_METHOD = 'inverse'

print("=" * 80)
print("Grid-Based Neighbor Search Test")
print("=" * 80)

# Load mesh
print("\n[1/5] Loading mesh...")
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    base_path=MESH_BASE_PATH,
    file_pattern=MESH_FILE_PATTERN,
    timestep_range=TIMESTEP_RANGE,
    field_name='Displacement',
    verbose=False
)
print(f"  Nodes: {node_positions.shape[0]:,}, Elements: {connectivity.shape[0]:,}")

# Deduplicate
print("\n[2/5] Deduplicating...")
node_positions, connectivity, n_duplicates, _ = deduplicate_nodes(
    node_positions, connectivity, verbose=False
)
print(f"  After dedup: {node_positions.shape[0]:,} nodes, {connectivity.shape[0]:,} elements")

# Extract cells and build structure
print("\n[3/5] Building structure...")
mesh_octree_cells = extract_octree_cells_single(
    node_positions, connectivity, tolerance=1e-6, verbose=False
)
mesh_aligned_morton_struct = build_mesh_aligned_morton_structure(
    node_positions, connectivity, mesh_octree_cells=mesh_octree_cells, verbose=False
)
print(f"  Cells: {mesh_aligned_morton_struct.n_cells:,}")
print(f"  Elements per cell: {mesh_aligned_morton_struct.elements_per_cell_mean:.1f} (mean)")

# Precompute inverse matrices
print("\n[4/5] Precomputing inverse matrices...")
M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)
M_inv_gpu = jax.device_put(M_inv_array)
p0_gpu = jax.device_put(p0_array)
set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)

# Upload to GPU
print("\n[5/5] Uploading to GPU...")
mesh_aligned_morton_gpu = upload_mesh_aligned_morton_to_gpu(
    node_positions, connectivity, mesh_aligned_morton_struct, verbose=False
)

# Generate test particles
print(f"\nGenerating {N_TEST_PARTICLES:,} random test particles...")
bbox_min = node_positions.min(axis=0)
bbox_max = node_positions.max(axis=0)
bbox_size = bbox_max - bbox_min

rng = np.random.default_rng(42)
positions = bbox_min + rng.random((N_TEST_PARTICLES, 3)) * bbox_size
positions = positions.astype(np.float32)
positions_gpu = jax.device_put(positions)

print("\n" + "=" * 80)
print("COMPARISON: Morton Radius vs Grid Neighbors")
print("=" * 80)

# Test configurations
test_configs = [
    {
        'name': 'Morton radius=2',
        'method': 'morton',
        'param': 2,
        'expected_cells': 5,
    },
    {
        'name': 'Morton radius=10',
        'method': 'morton',
        'param': 10,
        'expected_cells': 21,
    },
    {
        'name': 'Morton radius=50',
        'method': 'morton',
        'param': 50,
        'expected_cells': 101,
    },
    {
        'name': 'Grid neighbors 3×3×3 (radius=1)',
        'method': 'grid',
        'param': 1,
        'expected_cells': 27,
    },
    {
        'name': 'Grid neighbors 5×5×5 (radius=2)',
        'method': 'grid',
        'param': 2,
        'expected_cells': 125,
    },
]

results = []

for cfg in test_configs:
    print(f"\n[{cfg['name']}]")
    print(f"  Expected cells searched: {cfg['expected_cells']}")
    print(f"  Expected tests: ~{cfg['expected_cells'] * mesh_aligned_morton_struct.elements_per_cell_mean:.0f}")

    if cfg['method'] == 'morton':
        elem_ids = search_L2_mesh_aligned_morton_batch(
            positions_gpu,
            mesh_aligned_morton_gpu,
            search_radius=jnp.int32(cfg['param']),
            max_tests_per_cell=jnp.int32(256)
        )
    else:  # grid
        elem_ids = search_L2_mesh_aligned_grid_neighbors_batch(
            positions_gpu,
            mesh_aligned_morton_gpu,
            grid_radius=jnp.int32(cfg['param']),
            max_tests_per_cell=jnp.int32(256)
        )

    elem_ids = np.array(elem_ids)
    n_found = np.sum(elem_ids >= 0)
    success_rate = n_found / N_TEST_PARTICLES * 100

    print(f"  Found: {n_found:,} / {N_TEST_PARTICLES:,} ({success_rate:.1f}%)")

    results.append({
        'name': cfg['name'],
        'method': cfg['method'],
        'param': cfg['param'],
        'cells': cfg['expected_cells'],
        'found': n_found,
        'rate': success_rate,
    })

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"{'Method':<40} {'Cells':<10} {'Tests~':<10} {'Found':<10} {'Rate':<10}")
print("-" * 80)

for r in results:
    tests = int(r['cells'] * mesh_aligned_morton_struct.elements_per_cell_mean)
    print(f"{r['name']:<40} {r['cells']:<10} {tests:<10} {r['found']:<10} {r['rate']:<10.1f}%")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("\nKey Insight:")
print("  - Morton radius searches along 1D Morton curve (wrong for sparse cells)")
print("  - Grid neighbors searches 3D cube (correct spatial neighbors)")
print("\nExpected behavior:")
print("  - Random particles in bbox: ~30-50% fall inside mesh (rest in void)")
print("  - For particles INSIDE mesh, grid search should find ~98%")
print(f"  - Expected overall: ~35% (0.4 × 0.98 ≈ 39%, accounting for void)")

best_grid = max([r for r in results if r['method'] == 'grid'], key=lambda x: x['rate'])
best_morton = max([r for r in results if r['method'] == 'morton'], key=lambda x: x['rate'])

print(f"\nBest grid result: {best_grid['name']} = {best_grid['rate']:.1f}%")
print(f"Best Morton result: {best_morton['name']} = {best_morton['rate']:.1f}%")

if best_grid['rate'] > best_morton['rate']:
    improvement = best_grid['rate'] - best_morton['rate']
    print(f"\n✅ Grid search BETTER by {improvement:.1f} percentage points!")
else:
    print(f"\n⚠️  Grid search not better (needs investigation)")

print("\n" + "=" * 80)
