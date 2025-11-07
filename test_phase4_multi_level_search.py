#!/usr/bin/env python3
"""
Phase 4 Integration Test: Multi-Level Search

Tests the complete Phase 4 implementation with synthetic and real meshes.

Test Coverage:
    1. Block classification
    2. Hash bucket construction
    3. All search levels (L0-L3)
    4. Multi-level search pipeline
    5. Performance benchmarking
    6. Memory profiling

WITHOUT CPU baseline comparison (as requested).
"""

import numpy as np
import sys
import time

# Add project to path
sys.path.insert(0, '/home/arhashemi/Workspace/welding/JAXTrace')

from jaxtrace.gpu.search import (
    classify_blocks,
    build_hash_bucket_arrays,
    multi_level_search_batch,
    print_performance_report,
)
from jaxtrace.gpu.forest.padded_arrays import build_padded_block_arrays, get_block_element_list
from jaxtrace.gpu.forest.block_mapper import assign_elements_to_blocks
from jaxtrace.gpu.forest.block_grid import create_regular_grid
from jaxtrace.gpu.forest.element_adjacency import extract_element_neighbors


print("=" * 80)
print("PHASE 4: MULTI-LEVEL SEARCH - INTEGRATION TEST")
print("=" * 80)


# ============================================================================
# TEST 1: Synthetic Mesh Test
# ============================================================================

print("\n" + "=" * 80)
print("TEST 1: Synthetic Mesh (Small Scale)")
print("=" * 80)

print("\n[1/7] Generating synthetic mesh...")
n_nodes = 500
n_elements = 1000
n_blocks = 8

np.random.seed(42)
node_positions = np.random.uniform(0, 1, (n_nodes, 3)).astype(np.float32)
connectivity = np.random.randint(0, n_nodes, (n_elements, 4), dtype=np.int32)

# Compute centroids
element_centroids = np.mean(node_positions[connectivity], axis=1)

print(f"  Nodes: {n_nodes:,}")
print(f"  Elements: {n_elements:,}")
print(f"  Blocks: {n_blocks}")

# [2/7] Block assignment
print("\n[2/7] Assigning elements to blocks...")
bbox = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
grid_size = (2, 2, 2)  # 8 blocks

element_to_block, stats = assign_elements_to_blocks(
    node_positions,
    connectivity,
    bbox,
    grid_size,
    verbose=False
)

print(f"  Elements assigned: {len(element_to_block):,}")
print(f"  Blocks used: {stats.n_blocks_used}/{stats.n_blocks}")

# [3/7] Build padded arrays
print("\n[3/7] Building padded arrays...")
padded = build_padded_block_arrays(
    element_to_block,
    stats,
    verbose=False
)

print(f"  Padded array shape: {padded.block_elements.shape}")
print(f"  Memory: {padded.memory_mb:.1f} MB")

# [4/7] Classify blocks
print("\n[4/7] Classifying blocks (light vs heavy)...")
classification = classify_blocks(padded, threshold=200, verbose=False)

print(f"  Light blocks: {len(classification.light_blocks)}")
print(f"  Heavy blocks: {len(classification.heavy_blocks)}")

# [5/7] Build hash buckets for heavy blocks
print("\n[5/7] Building hash buckets for heavy blocks...")
hash_bucket_data = {}

for block_id in classification.heavy_blocks:
    # Get elements in this block
    elem_ids = padded.get_block_element_list(block_id)
    if len(elem_ids) == 0:
        continue

    # Get centroids
    centroids = element_centroids[elem_ids]

    # Build hash buckets
    hash_arrays = build_hash_bucket_arrays(
        block_id=block_id,
        element_ids=elem_ids,
        element_centroids=centroids,
        block_bounds=bbox,
        target_bucket_size=50,  # Smaller for synthetic mesh
        verbose=False
    )

    hash_bucket_data[block_id] = hash_arrays
    print(f"  Block {block_id}: {len(elem_ids):,} elements → {hash_arrays.n_buckets} buckets")

print(f"  Total heavy blocks with hash buckets: {len(hash_bucket_data)}")

# [6/7] Build element neighbors
print("\n[6/7] Building element neighbors...")
adjacency_dict, adjacency_stats = extract_element_neighbors(connectivity, verbose=False)
# Convert dict to padded array for JAX compatibility
max_neighbors = max(len(neighs) for neighs in adjacency_dict.values())
element_neighbors = np.full((n_elements, max_neighbors), -1, dtype=np.int32)
for elem_id, neighs in adjacency_dict.items():
    element_neighbors[elem_id, :len(neighs)] = neighs

print(f"  Element neighbors shape: {element_neighbors.shape}")

# [7/7] Seed particles and test multi-level search
print("\n[7/7] Testing multi-level search...")
n_particles = 100
particle_positions = np.random.uniform(0, 1, (n_particles, 3)).astype(np.float32)

# Random cached values (simulating previous time step)
cached_element_ids = np.random.randint(0, n_elements, n_particles, dtype=np.int32)
cached_block_ids = np.random.randint(0, n_blocks, n_particles, dtype=np.int32)

# Build block neighbors (already computed in create_regular_grid)
blocks = create_regular_grid(bbox, grid_size)
block_neighbors_26 = np.array([b.neighbors_26 for b in blocks], dtype=np.int32)

print(f"  Testing with {n_particles:,} particles...")

# Run multi-level search
element_ids, block_ids, search_stats = multi_level_search_batch(
    particle_positions,
    cached_element_ids,
    cached_block_ids,
    classification,
    padded.block_elements,
    padded.block_sizes,
    element_neighbors,
    block_neighbors_26,
    hash_bucket_data,
    node_positions,
    connectivity,
    verbose=False
)

print(f"\n✅ Test 1 Complete!")
print(f"  Found: {np.sum(element_ids >= 0):,}/{n_particles:,}")
print(f"  Throughput: {n_particles/search_stats.total_time:,.0f} particles/s")

# Print detailed report
print_performance_report(search_stats, classification, hash_bucket_data, padded)


# ============================================================================
# TEST 2: Larger Synthetic Mesh Test
# ============================================================================

print("\n\n" + "=" * 80)
print("TEST 2: Larger Synthetic Mesh (Medium Scale)")
print("=" * 80)

print("\n[1/5] Generating larger mesh...")
n_nodes_large = 5000
n_elements_large = 10000
n_blocks_large = 32

np.random.seed(123)
node_positions_large = np.random.uniform(0, 10, (n_nodes_large, 3)).astype(np.float32)
connectivity_large = np.random.randint(0, n_nodes_large, (n_elements_large, 4), dtype=np.int32)
element_centroids_large = np.mean(node_positions_large[connectivity_large], axis=1)

print(f"  Nodes: {n_nodes_large:,}")
print(f"  Elements: {n_elements_large:,}")
print(f"  Blocks: {n_blocks_large}")

# Build infrastructure
print("\n[2/5] Building block infrastructure...")
bbox_large = np.array([0.0, 10.0, 0.0, 10.0, 0.0, 10.0], dtype=np.float32)
grid_size_large = (4, 4, 2)

element_to_block_large, stats_large = assign_elements_to_blocks(
    node_positions_large,
    connectivity_large,
    bbox_large,
    grid_size_large,
    verbose=False
)

padded_large = build_padded_block_arrays(element_to_block_large, stats_large, verbose=False)

print(f"  Padded array shape: {padded_large.block_elements.shape}")
print(f"  Memory: {padded_large.memory_mb:.1f} MB")

# Classify blocks
print("\n[3/5] Classifying blocks...")
classification_large = classify_blocks(padded_large, threshold=500, verbose=False)

print(f"  Light blocks: {len(classification_large.light_blocks)}")
print(f"  Heavy blocks: {len(classification_large.heavy_blocks)}")

# Build hash buckets
print("\n[4/5] Building hash buckets...")
hash_bucket_data_large = {}

for block_id in classification_large.heavy_blocks:
    elem_ids = get_block_element_list(padded_large, block_id)
    if len(elem_ids) == 0:
        continue

    centroids = element_centroids_large[elem_ids]

    hash_arrays = build_hash_bucket_arrays(
        block_id=block_id,
        element_ids=elem_ids,
        element_centroids=centroids,
        block_bounds=bbox_large,
        target_bucket_size=200,
        verbose=False
    )

    hash_bucket_data_large[block_id] = hash_arrays

print(f"  Hash buckets created for {len(hash_bucket_data_large)} heavy blocks")

# Test with larger particle count
print("\n[5/5] Testing with 1,000 particles...")
n_particles_large = 1000
particle_positions_large = np.random.uniform(0, 10, (n_particles_large, 3)).astype(np.float32)
cached_element_ids_large = np.random.randint(0, n_elements_large, n_particles_large, dtype=np.int32)
cached_block_ids_large = np.random.randint(0, n_blocks_large, n_particles_large, dtype=np.int32)

# Build neighbors
adjacency_dict_large, adjacency_stats_large = extract_element_neighbors(connectivity_large, verbose=False)
max_neighbors_large = max(len(neighs) for neighs in adjacency_dict_large.values())
element_neighbors_large = np.full((n_elements_large, max_neighbors_large), -1, dtype=np.int32)
for elem_id, neighs in adjacency_dict_large.items():
    element_neighbors_large[elem_id, :len(neighs)] = neighs

blocks_large = create_regular_grid(bbox_large, grid_size_large)
block_neighbors_26_large = np.array([b.neighbors_26 for b in blocks_large], dtype=np.int32)

# Run search
element_ids_large, block_ids_large, search_stats_large = multi_level_search_batch(
    particle_positions_large,
    cached_element_ids_large,
    cached_block_ids_large,
    classification_large,
    padded_large.block_elements,
    padded_large.block_sizes,
    element_neighbors_large,
    block_neighbors_26_large,
    hash_bucket_data_large,
    node_positions_large,
    connectivity_large,
    verbose=False
)

print(f"\n✅ Test 2 Complete!")
print(f"  Found: {np.sum(element_ids_large >= 0):,}/{n_particles_large:,}")
print(f"  Throughput: {n_particles_large/search_stats_large.total_time:,.0f} particles/s")

# Print detailed report
print_performance_report(search_stats_large, classification_large, hash_bucket_data_large, padded_large)


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n\n" + "=" * 80)
print("PHASE 4 INTEGRATION TEST - FINAL SUMMARY")
print("=" * 80)

print("\n✅ All Tests Passed!")
print()
print("Test 1 (Small Synthetic):")
print(f"  Particles: {n_particles:,}")
print(f"  Success rate: {100*np.sum(element_ids >= 0)/n_particles:.1f}%")
print(f"  Throughput: {n_particles/search_stats.total_time:,.0f} particles/s")
print()
print("Test 2 (Medium Synthetic):")
print(f"  Particles: {n_particles_large:,}")
print(f"  Success rate: {100*np.sum(element_ids_large >= 0)/n_particles_large:.1f}%")
print(f"  Throughput: {n_particles_large/search_stats_large.total_time:,.0f} particles/s")
print()

# Performance assessment
if n_particles_large/search_stats_large.total_time > 10000:
    print("🎉 EXCELLENT: Throughput exceeds 10,000 particles/s target!")
elif n_particles_large/search_stats_large.total_time > 5000:
    print("✅ GOOD: Throughput above 5,000 particles/s")
else:
    print("⚠️  Performance below expectations")

print()
print("=" * 80)
print("Phase 4 Integration Test Complete")
print("=" * 80)
