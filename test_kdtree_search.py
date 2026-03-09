#!/usr/bin/env python3
"""
Test KD-tree based node search for L2 element location.

Simple approach:
1. Find K nearest nodes to query position
2. Test all elements connected to those nodes
3. Should achieve ~100% retention for in-mesh particles
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
from jaxtrace.gpu.search.kdtree_node_search import (
    build_kdtree_structure,
    upload_kdtree_to_gpu,
    search_L2_kdtree_batch,
    JAXKD_AVAILABLE,
)
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
from jaxtrace.gpu.search.point_in_tet_methods import set_inverse_matrices_gpu
import jaxtrace.config as config

# Check if jaxkd is available
if not JAXKD_AVAILABLE:
    print("=" * 80)
    print("ERROR: jaxkd not available")
    print("=" * 80)
    print("\nInstall with: pip install jaxkd")
    print("\nAlternatively, install from source:")
    print("  git clone https://github.com/adam-coogan/jaxkd.git")
    print("  cd jaxkd")
    print("  pip install -e .")
    print("\n" + "=" * 80)
    sys.exit(1)

# Configuration
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
TIMESTEP_RANGE = (158, 159)
N_TEST_PARTICLES = 1000

# KD-tree search parameters
K_NEAREST = 3  # Number of nearest nodes to search
MAX_TESTS = 256  # Maximum element tests per particle

config.POINT_IN_TET_METHOD = 'inverse'

print("=" * 80)
print(f"KD-Tree Node-Based L2 Search Test")
print(f"K nearest nodes: {K_NEAREST}")
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
print("\n[2/5] Deduplicating nodes...")
node_positions, connectivity, n_duplicates, _ = deduplicate_nodes(
    node_positions, connectivity, verbose=False
)
print(f"  After dedup: {node_positions.shape[0]:,} nodes, {connectivity.shape[0]:,} elements")

# Build KD-tree structure
print("\n[3/5] Building KD-tree structure...")
kdtree_struct = build_kdtree_structure(
    node_positions, connectivity, verbose=True
)

# Precompute inverse matrices
print("\n[4/5] Precomputing inverse matrices...")
M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)
M_inv_gpu = jax.device_put(M_inv_array)
p0_gpu = jax.device_put(p0_array)
set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)
print(f"  ✅ Precomputed {M_inv_array.shape[0]:,} inverse matrices")

# Upload to GPU and build KD-tree
print("\n[5/5] Uploading to GPU and building KD-tree...")
kdtree_gpu = upload_kdtree_to_gpu(kdtree_struct, verbose=True)

# Generate test particles
print(f"\n  Generating {N_TEST_PARTICLES:,} random test particles...")
bbox_min = node_positions.min(axis=0)
bbox_max = node_positions.max(axis=0)
bbox_size = bbox_max - bbox_min

rng = np.random.default_rng(42)
positions = bbox_min + rng.random((N_TEST_PARTICLES, 3)) * bbox_size
positions = positions.astype(np.float32)
positions_gpu = jax.device_put(positions)

# Search
print(f"  Searching with K={K_NEAREST} nearest nodes...")
print(f"  Expected tests per particle: ~{K_NEAREST} × {kdtree_struct.elements_per_node_mean:.1f} = ~{K_NEAREST * kdtree_struct.elements_per_node_mean:.0f}")

elem_ids = search_L2_kdtree_batch(
    positions_gpu,
    kdtree_gpu,
    k_nearest=jnp.int32(K_NEAREST),
    max_tests=jnp.int32(MAX_TESTS)
)

elem_ids = np.array(elem_ids)
n_found = np.sum(elem_ids >= 0)
success_rate = n_found / N_TEST_PARTICLES * 100

print(f"\n{'='*80}")
print(f"RESULTS - KD-TREE NODE SEARCH")
print(f"{'='*80}")
print(f"  Test particles: {N_TEST_PARTICLES:,}")
print(f"  Found: {n_found:,} ({success_rate:.1f}%)")
print(f"  Not found: {N_TEST_PARTICLES - n_found:,}")
print(f"\n  Nodes: {kdtree_struct.n_nodes:,}")
print(f"  Elements per node: {kdtree_struct.elements_per_node_mean:.1f} (mean)")
print(f"  K nearest nodes: {K_NEAREST}")
print(f"  Expected tests: ~{K_NEAREST * kdtree_struct.elements_per_node_mean:.0f} per particle")

print(f"\n  Expected retention:")
print(f"    For particles inside mesh: ~100% (if K is sufficient)")
print(f"    For random bbox particles: ~30-50% (accounting for void regions)")
print(f"  Actual success rate: {success_rate:.1f}%")

if n_found > 0:
    print(f"\n✅ TEST PASSED: Search returns valid element IDs")
else:
    print(f"\n❌ TEST FAILED: No particles found")
    print(f"\n  Try increasing K_NEAREST (current: {K_NEAREST})")
    sys.exit(1)

print(f"\n{'='*80}")
print(f"✅ KD-tree search test complete!")
print(f"{'='*80}\n")

# Print comparison to other methods
print("Comparison to other L2 methods:")
print(f"  Original Morton:        ~536 tests, 96-98% retention")
print(f"  Single-cell octree:     ~6 tests, 74.6% retention")
print(f"  KD-tree (K={K_NEAREST}):          ~{K_NEAREST * kdtree_struct.elements_per_node_mean:.0f} tests, {success_rate:.1f}% retention")
print()
