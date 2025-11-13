"""
Quick test to verify L3 neighbor search integration works.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/arhashemi/Workspace/welding/JAXTrace')

from jaxtrace.gpu.test_meshes import generate_test_mesh, TestMeshConfig
from jaxtrace.gpu.forest import assign_elements_to_blocks, build_padded_block_arrays
from jaxtrace.gpu.search import (
    classify_blocks,
    multi_level_search_batch,
)

print("=" * 80)
print("L3 NEIGHBOR SEARCH INTEGRATION TEST")
print("=" * 80)

# Generate small test mesh
print("\n[1] Generating test mesh...")
config = TestMeshConfig(
    mesh_type="cube",
    n_elements=500,
    domain_bounds=np.array([0, 10, 0, 10, 0, 10], dtype=np.float32)
)
mesh_data = generate_test_mesh(config)
node_positions = mesh_data["node_positions"]
connectivity = mesh_data["connectivity"]
element_neighbors = mesh_data["element_neighbors"]

print(f"  Mesh: {len(connectivity):,} elements, {len(node_positions):,} nodes")

# Build forest structure
print("\n[2] Building forest structure...")
bbox = config.domain_bounds
grid_size = (4, 4, 2)

element_to_block, stats = assign_elements_to_blocks(
    node_positions, connectivity, bbox, grid_size
)
padded = build_padded_block_arrays(element_to_block, stats)

# Classify blocks
classification = classify_blocks(padded, threshold=10000)
print(f"  Blocks: {len(padded.block_sizes)} total, {len(classification.light_blocks)} light, {len(classification.heavy_blocks)} heavy")

# Create test particles (some will need L3 search)
print("\n[3] Creating test particles...")
n_particles = 100
particle_positions = np.random.uniform(0, 10, (n_particles, 3)).astype(np.float32)

# Use random cached elements (some will be wrong, forcing fallback to L2/L3)
cached_element_ids = np.random.randint(0, len(connectivity), n_particles, dtype=np.int32)
cached_block_ids = np.random.randint(0, len(padded.block_sizes), n_particles, dtype=np.int32)

print(f"  Particles: {n_particles}")

# Build block neighbors (simplified - use stats from phase 1)
from jaxtrace.gpu.forest import create_regular_grid
blocks = create_regular_grid(bbox, grid_size)
block_neighbors_26 = np.array([b.neighbors_26 for b in blocks], dtype=np.int32)

# Run multi-level search
print("\n[4] Running multi-level search with L3...")
element_ids, block_ids, search_stats = multi_level_search_batch(
    particle_positions,
    cached_element_ids,
    cached_block_ids,
    classification,
    padded.block_elements,
    padded.block_sizes,
    block_neighbors_26,
    None,  # No hash buckets for this simple test
    node_positions,
    connectivity,
    element_neighbors,
    verbose=False
)

print("\n[RESULTS]")
print(f"  Particles: {search_stats.n_particles:,}")
n_found = search_stats.l0_hits + search_stats.l1_hits + search_stats.l2_hits + search_stats.l3_hits
print(f"  Found: {n_found:,} ({100*n_found/search_stats.n_particles:.1f}%)")
print(f"  L0 (cached) hits: {search_stats.l0_hits:,} ({100*search_stats.l0_hits/search_stats.n_particles:.1f}%)")
print(f"  L1 (neighbors) hits: {search_stats.l1_hits:,} ({100*search_stats.l1_hits/search_stats.n_particles:.1f}%)")
print(f"  L2 (block) hits: {search_stats.l2_hits:,} ({100*search_stats.l2_hits/search_stats.n_particles:.1f}%)")
print(f"  L3 (neighbor blocks) hits: {search_stats.l3_hits:,} ({100*search_stats.l3_hits/search_stats.n_particles:.1f}%)")
print(f"  Not found: {search_stats.not_found:,} ({100*search_stats.not_found/search_stats.n_particles:.1f}%)")

# Check if L3 is working (should have > 0% hits now)
if search_stats.l3_hits > 0:
    print("\n✅ L3 INTEGRATION SUCCESSFUL - Neighbor block search is working!")
else:
    print("\n⚠️  L3 INTEGRATION MAY HAVE ISSUES - No neighbor block hits detected")
    print("   (This might be OK if all particles found in L0-L2)")

print("\n" + "=" * 80)
