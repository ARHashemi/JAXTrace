#!/usr/bin/env python3
"""
Minimal test of GPU search function to isolate issues.
"""

import sys
import numpy as np
sys.path.insert(0, '/home/arhashemi/Workspace/welding/JAXTrace')

print("=" * 80)
print("MINIMAL GPU SEARCH TEST")
print("=" * 80)

print("\n1. Importing modules...")
from jaxtrace.gpu.test_meshes import generate_test_mesh, TINY_MESH
from jaxtrace.gpu.mesh_loader import assign_elements_to_blocks
from jaxtrace.gpu.octree_builder import build_octrees_per_block
from jaxtrace.gpu.initial_search_jax import find_initial_elements_batch, GPUConfig

print("2. Generating tiny mesh...")
positions, connectivity = generate_test_mesh(TINY_MESH)
print(f"   Mesh: {len(connectivity)} elements")

print("3. Assigning to blocks...")
element_block_IDs, partition_data = assign_elements_to_blocks(
    positions, connectivity, (2, 2, 1), verbose=False
)

print("4. Building octrees...")
octrees = build_octrees_per_block(
    positions, connectivity, element_block_IDs, partition_data,
    max_elements_per_node=50,
    verbose=False
)

print("5. Creating test particles (just 3)...")
# Create 3 particles at element centroids
test_particles = []
for i in [10, 50, 100]:
    vertices = positions[connectivity[i]]
    centroid = vertices.mean(axis=0)
    test_particles.append(centroid)

particle_positions = np.array(test_particles)
print(f"   Created {len(particle_positions)} test particles")

print("6. Preparing mesh data...")
mesh_data = {
    'positions': positions,
    'connectivity': connectivity
}

print("7. Testing GPU search...")
gpu_config = GPUConfig(use_gpu_initial_search=True, force_cpu=False)

try:
    element_IDs, stats = find_initial_elements_batch(
        particle_positions,
        mesh_data,
        partition_data,
        octrees,
        config=gpu_config,
        verbose=True
    )
    print(f"\n   Found elements: {element_IDs}")
    print(f"   Used GPU: {stats['used_gpu']}")
    print("\n✅ SUCCESS!")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
