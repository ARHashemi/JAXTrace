"""
Test Initial Assignment with 10,000 Particles

Verifies that the fixed initial_assignment.py works without OOM errors.
"""

import numpy as np
from pathlib import Path
import time

# GPU imports
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.forest import (
    create_regular_grid,
    assign_elements_to_blocks,
    build_element_neighbors_array,
    build_padded_block_arrays,
)
from jaxtrace.gpu.search import (
    classify_blocks,
    build_hash_bucket_arrays,
    initial_search_batch,
)

print("=" * 80)
print("10K PARTICLE INITIAL ASSIGNMENT TEST")
print("=" * 80)
print()

# ================================================================================
# MESH LOADING
# ================================================================================
print("=" * 80)
print("MESH LOADING")
print("=" * 80)
print()

mesh_path = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_20.pvtu"
print(f"Loading: {mesh_path}")

t0 = time.perf_counter()
node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
    Path(mesh_path),
    field_name='Displacement'
)
t_load = time.perf_counter() - t0

print(f"✓ Mesh loaded ({t_load:.2f} s):")
print(f"  Nodes: {len(node_positions):,}")
print(f"  Elements: {len(connectivity):,}")
print()

# ================================================================================
# FOREST STRUCTURE
# ================================================================================
print("=" * 80)
print("FOREST STRUCTURE")
print("=" * 80)
print()

# Create block grid
grid_size = (8, 8, 4)
blocks = create_block_grid(node_positions, connectivity, grid_size)
print(f"✓ Block grid created: {len(blocks)} blocks")

# Assign elements to blocks
t0 = time.perf_counter()
block_elements = assign_elements_to_blocks(
    connectivity,
    node_positions,
    blocks
)
t_assign = time.perf_counter() - t0
print(f"✓ Element assignment ({t_assign:.2f} s)")

# Build element neighbors
t0 = time.perf_counter()
face_neighbors = build_element_neighbors(connectivity)
t_neighbors = time.perf_counter() - t0
print(f"✓ Element neighbors built ({t_neighbors:.2f} s)")

# Create padded arrays
t0 = time.perf_counter()
padded_arrays = create_padded_arrays(
    blocks,
    block_elements,
    connectivity,
    node_positions
)
t_padded = time.perf_counter() - t0
print(f"✓ Padded arrays ({t_padded:.2f} s):")
print(f"  Shape: ({len(blocks)}, {padded_arrays.max_elements_per_block})")
print(f"  Memory: {padded_arrays.memory_mb:.1f} MB")
print()

# Classify blocks
block_classification = classify_blocks(block_elements, threshold=10000)
print(f"✓ Block classification:")
print(f"  Light blocks: {len(block_classification.light_blocks)}")
print(f"  Heavy blocks: {len(block_classification.heavy_blocks)}")
print()

# Build hash buckets for heavy blocks
if block_classification.heavy_blocks:
    print(f"Building hash buckets for {len(block_classification.heavy_blocks)} heavy blocks...")
    hash_bucket_data = {}
    for block_id in block_classification.heavy_blocks:
        bucket_arrays = build_hash_bucket_arrays(
            block_id,
            block_elements[block_id],
            connectivity,
            node_positions,
            bucket_size=8
        )
        hash_bucket_data[block_id] = bucket_arrays
    print(f"✓ Hash buckets built")
else:
    hash_bucket_data = {}
print()

# Build block neighbor connectivity
n_blocks = len(blocks)
block_neighbors_26 = []
for i in range(n_blocks):
    neighbors = []
    block = blocks[i]
    for j in range(n_blocks):
        if i == j:
            continue
        other = blocks[j]
        # Check if blocks are adjacent (26-connectivity)
        if (abs(block.ijk[0] - other.ijk[0]) <= 1 and
            abs(block.ijk[1] - other.ijk[1]) <= 1 and
            abs(block.ijk[2] - other.ijk[2]) <= 1):
            neighbors.append(j)
    # Pad to 26
    while len(neighbors) < 26:
        neighbors.append(-1)
    block_neighbors_26.append(neighbors[:26])
block_neighbors_26 = np.array(block_neighbors_26, dtype=np.int32)

# ================================================================================
# PARTICLE GENERATION
# ================================================================================
print("=" * 80)
print("PARTICLE GENERATION")
print("=" * 80)
print()

n_particles = 10000
print(f"Generating {n_particles:,} test particles...")

# Get domain bounds
xmin, xmax = blocks[0].bounds[0][0], blocks[0].bounds[1][0]
ymin, ymax = blocks[0].bounds[0][1], blocks[0].bounds[1][1]
zmin, zmax = blocks[0].bounds[0][2], blocks[0].bounds[1][2]

for block in blocks:
    xmin = min(xmin, block.bounds[0][0])
    xmax = max(xmax, block.bounds[1][0])
    ymin = min(ymin, block.bounds[0][1])
    ymax = max(ymax, block.bounds[1][1])
    zmin = min(zmin, block.bounds[0][2])
    zmax = max(zmax, block.bounds[1][2])

domain_bounds = np.array([xmin, xmax, ymin, ymax, zmin, zmax], dtype=np.float32)

# Generate random particles
np.random.seed(42)
particle_positions = np.random.uniform(
    low=[xmin, ymin, zmin],
    high=[xmax, ymax, zmax],
    size=(n_particles, 3)
).astype(np.float32)

print(f"✓ Generated {n_particles:,} particles")
print()

# ================================================================================
# INITIAL ASSIGNMENT (THE CRITICAL TEST)
# ================================================================================
print("=" * 80)
print("INITIAL ASSIGNMENT")
print("=" * 80)
print()

print(f"Finding containing elements for {n_particles:,} particles...")
print("(This should NOT cause OOM with the fix)")
print()

t0 = time.perf_counter()
element_ids, block_ids, search_stats = initial_search_batch(
    particle_positions,
    domain_bounds,
    grid_size,
    block_classification,
    padded_arrays,
    block_neighbors_26,
    hash_bucket_data,
    node_positions,
    connectivity,
    verbose=True
)
t_search = time.perf_counter() - t0

n_found = np.sum(element_ids >= 0)

print()
print(f"✓ Initial assignment ({t_search:.2f} s):")
print(f"  Found: {n_found}/{n_particles} ({100*n_found/n_particles:.1f}%)")
print(f"  Throughput: {n_found/t_search:.1f} p/s")
print()

# ================================================================================
# RESULTS
# ================================================================================
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

print(f"✅ SUCCESS: Initial assignment completed without OOM!")
print(f"   Particles: {n_particles:,}")
print(f"   Found: {n_found:,} ({100*n_found/n_particles:.1f}%)")
print(f"   Throughput: {n_found/t_search:.1f} p/s")
print(f"   Time: {t_search:.2f} s")
print()

print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
