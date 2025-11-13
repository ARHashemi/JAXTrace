"""
Quick comparison test: V1 vs V2 multi-level search performance.

This test validates that V2 (JAX vmap vectorized) is faster than V1 (Python loop).
"""

import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/arhashemi/Workspace/welding/JAXTrace')

from jaxtrace.gpu.test_meshes import generate_test_mesh, TestMeshConfig
from jaxtrace.gpu.forest import assign_elements_to_blocks, build_padded_block_arrays
from jaxtrace.gpu.forest.element_neighbors import build_element_adjacency
from jaxtrace.gpu.search import classify_blocks

# Import both V1 and V2
from jaxtrace.gpu.search.multi_level_search import multi_level_search_batch as multi_level_v1
from jaxtrace.gpu.search.multi_level_search_v2 import multi_level_search_batch as multi_level_v2

print("=" * 80)
print("V1 vs V2 PERFORMANCE COMPARISON")
print("=" * 80)

# Generate test mesh (medium size)
print("\n[1] Generating test mesh...")
config = TestMeshConfig(
    domain_size=(10.0, 10.0, 10.0),  # 10x10x10 domain
    resolution=(10, 10, 10),         # 10x10x10 = 6000 tetrahedra
    use_adaptive_refinement=False
)
node_positions, connectivity = generate_test_mesh(config)

print(f"  Mesh: {len(connectivity):,} elements, {len(node_positions):,} nodes")

# Build forest structure
print("\n[2] Building forest structure...")
# Compute bounding box from mesh
bbox = np.array([
    node_positions[:, 0].min(), node_positions[:, 0].max(),
    node_positions[:, 1].min(), node_positions[:, 1].max(),
    node_positions[:, 2].min(), node_positions[:, 2].max(),
], dtype=np.float32)
grid_size = (4, 4, 4)

element_to_block, stats = assign_elements_to_blocks(
    node_positions, connectivity, bbox, grid_size
)
padded = build_padded_block_arrays(element_to_block, stats)
element_neighbors = build_element_adjacency(connectivity)

print(f"  Blocks: {len(padded.block_sizes)} total")

# Classify blocks
classification = classify_blocks(padded, threshold=10000)

# Create block neighbors (simplified)
from jaxtrace.gpu.forest.block_grid import create_regular_grid
blocks = create_regular_grid(bbox, grid_size)
block_neighbors_26 = np.array([b.neighbors_26 for b in blocks], dtype=np.int32)

# Create test particles
print("\n[3] Creating test particles...")
n_particles = 1000
particle_positions = np.random.uniform(0, 10, (n_particles, 3)).astype(np.float32)

# Random cached values (simulating previous timestep)
cached_element_ids = np.random.randint(0, len(connectivity), n_particles, dtype=np.int32)
cached_block_ids = np.random.randint(0, len(padded.block_sizes), n_particles, dtype=np.int32)

print(f"  Particles: {n_particles:,}")

# Run V1 (Python loop version)
print("\n[4] Running V1 (Python loop)...")
print("  Warming up JIT...")
_, _, _ = multi_level_v1(
    particle_positions[:10],
    cached_element_ids[:10],
    cached_block_ids[:10],
    classification,
    padded.block_elements,
    padded.block_sizes,
    element_neighbors,
    block_neighbors_26,
    None,
    node_positions,
    connectivity,
    verbose=False
)

print("  Running full test...")
start = time.time()
element_ids_v1, block_ids_v1, stats_v1 = multi_level_v1(
    particle_positions,
    cached_element_ids,
    cached_block_ids,
    classification,
    padded.block_elements,
    padded.block_sizes,
    element_neighbors,
    block_neighbors_26,
    None,
    node_positions,
    connectivity,
    verbose=False
)
v1_time = time.time() - start

v1_throughput = n_particles / v1_time
print(f"  Time: {v1_time:.3f} s")
print(f"  Throughput: {v1_throughput:.0f} particles/s")
print(f"  Found: {np.sum(element_ids_v1 >= 0):,}/{n_particles:,}")

# Run V2 (JAX vmap version)
print("\n[5] Running V2 (JAX vmap)...")
print("  Warming up JIT...")
_, _, _ = multi_level_v2(
    particle_positions[:10],
    cached_element_ids[:10],
    cached_block_ids[:10],
    classification,
    padded.block_elements,
    padded.block_sizes,
    element_neighbors,
    block_neighbors_26,
    None,
    node_positions,
    connectivity,
    verbose=False
)

print("  Running full test...")
start = time.time()
element_ids_v2, block_ids_v2, stats_v2 = multi_level_v2(
    particle_positions,
    cached_element_ids,
    cached_block_ids,
    classification,
    padded.block_elements,
    padded.block_sizes,
    element_neighbors,
    block_neighbors_26,
    None,
    node_positions,
    connectivity,
    verbose=False
)
v2_time = time.time() - start

v2_throughput = n_particles / v2_time
print(f"  Time: {v2_time:.3f} s")
print(f"  Throughput: {v2_throughput:.0f} particles/s")
print(f"  Found: {np.sum(element_ids_v2 >= 0):,}/{n_particles:,}")

# Compare results
print("\n[6] COMPARISON")
print("=" * 80)
speedup = v1_time / v2_time
print(f"V1 (Python loop):    {v1_throughput:>8.0f} p/s  ({v1_time:.3f} s)")
print(f"V2 (JAX vmap):       {v2_throughput:>8.0f} p/s  ({v2_time:.3f} s)")
print(f"Speedup:             {speedup:>8.1f}×")
print()

# Verify correctness (results should match or be very close)
matching = np.sum(element_ids_v1 == element_ids_v2)
match_rate = 100 * matching / n_particles
print(f"Results matching:    {matching:>8,}/{n_particles:,} ({match_rate:.1f}%)")

if speedup > 1.5:
    print(f"\n✅ V2 IS FASTER: {speedup:.1f}× speedup achieved!")
elif speedup > 1.0:
    print(f"\n⚠️  V2 IS SLIGHTLY FASTER: {speedup:.1f}× speedup (expected >1.5×)")
else:
    print(f"\n❌ V2 IS SLOWER: Something is wrong!")

if match_rate < 95:
    print(f"⚠️  WARNING: Low match rate ({match_rate:.1f}%) - results differ significantly")
else:
    print(f"✅ Results match well ({match_rate:.1f}%)")

print("\n" + "=" * 80)
print("NOTES:")
print("- This is a small test mesh (10K elements, 1K particles)")
print("- Real speedup will be higher on ThreadedA (3.5M elements, 10K+ particles)")
print("- Expected V2 speedup on ThreadedA: 25-75× (5,000-13,000 p/s)")
print("=" * 80)
