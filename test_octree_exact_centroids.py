#!/usr/bin/env python3
"""
Test octree search with particles placed at EXACT element centroids (no perturbation).

This tests whether the centroid-based octree assignment causes issues even
when particles are at exact centroid locations.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import time

# Load mesh
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu

# Octree
from jaxtrace.gpu.search.octree_builder import build_octree_for_level, flatten_octree_to_arrays
from jaxtrace.gpu.search.octree_search_gpu import search_level2_octree_scan

# VTK
import vtk
from vtk.util import numpy_support

print("=" * 80)
print("OCTREE SEARCH TEST - EXACT CENTROIDS (NO PERTURBATION)")
print("=" * 80)
print()

# Load mesh
mesh_path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_120.pvtu")
print(f"Loading mesh: {mesh_path.name}")
node_positions, connectivity, _ = load_mesh_from_pvtu(mesh_path, field_name='Displacement')
print(f"✓ Loaded: {len(node_positions):,} nodes, {len(connectivity):,} elements")
print()

# Load LEVEL field
print("Loading LEVEL field...")
reader = vtk.vtkXMLPUnstructuredGridReader()
reader.SetFileName(str(mesh_path))
reader.Update()
vtk_mesh = reader.GetOutput()
cell_data = vtk_mesh.GetCellData()
point_data = vtk_mesh.GetPointData()

level_field = None
if cell_data.HasArray('LEVEL'):
    level_field = numpy_support.vtk_to_numpy(cell_data.GetArray('LEVEL')).astype(np.float32)
elif point_data.HasArray('LEVEL'):
    node_level = numpy_support.vtk_to_numpy(point_data.GetArray('LEVEL')).astype(np.float32)
    level_field = np.array([
        node_level[connectivity[i]].max()
        for i in range(len(connectivity))
    ], dtype=np.float32)
print("✓ Loaded LEVEL field")
print()

# Compute element centroids
print("Computing element centroids...")
element_centroids = np.array([
    node_positions[connectivity[i]].mean(axis=0)
    for i in range(len(connectivity))
], dtype=np.float32)
print(f"✓ Computed {len(element_centroids):,} centroids")
print()

# Build octree
print("Building octree...")
element_ids = np.arange(len(connectivity), dtype=np.int32)

t_start = time.time()
nodes, metadata = build_octree_for_level(
    element_centroids,
    element_ids,
    level_field=level_field,
    level_threshold=1.1,
    max_depth=15,
    max_leaf_size=50,
    use_levelset=True
)
t_build = time.time() - t_start

print(f"✓ Built octree ({t_build:.2f} s)")
print(f"  Filtered elements: {metadata['n_elements']:,}/{len(connectivity):,}")
print(f"  Total nodes: {metadata['n_nodes']:,}")
print(f"  Max depth: {metadata['max_depth']}")
print()

# Flatten and upload to GPU
node_metadata_np, node_elements_np = flatten_octree_to_arrays(nodes, max_leaf_size=50)
octree_metadata_gpu = jax.device_put(node_metadata_np)
octree_elements_gpu = jax.device_put(node_elements_np)
node_positions_gpu = jax.device_put(node_positions)
connectivity_gpu = jax.device_put(connectivity)
print("✓ Uploaded octree to GPU")
print()

# Generate test particles at EXACT centroids (no perturbation)
print("=" * 80)
print("TEST: Particles at EXACT element centroids")
print("=" * 80)
print()

np.random.seed(42)
n_test = 50000

# Sample random elements
test_element_ids = np.random.choice(len(connectivity), size=n_test, replace=False)

# Place particles at EXACT centroids (zero perturbation)
test_particles = element_centroids[test_element_ids]

print(f"Generated {n_test:,} particles at EXACT centroids")
print(f"  Perturbation: ZERO (exact centroids)")
print()

# Upload to GPU
positions_gpu = jax.device_put(test_particles)
cached_ids = jnp.full(n_test, -1, dtype=jnp.int32)  # All need search

# Warm up JIT
print("Warming up JIT...")
_ = search_level2_octree_scan(
    positions_gpu[:10],
    cached_ids[:10],
    octree_metadata_gpu,
    octree_elements_gpu,
    node_positions_gpu,
    connectivity_gpu,
    max_depth=15
)
print("✓ JIT warm-up complete")
print()

# Run search
print("Running octree search...")
t_start = time.time()
found_elements = search_level2_octree_scan(
    positions_gpu,
    cached_ids,
    octree_metadata_gpu,
    octree_elements_gpu,
    node_positions_gpu,
    connectivity_gpu,
    max_depth=15
)
jax.block_until_ready(found_elements)
t_search = time.time() - t_start

found_elements_cpu = np.array(found_elements)

print(f"✓ Search complete ({t_search:.4f} s)")
print(f"  Throughput: {n_test/t_search:,.1f} p/s")
print()

# Analyze results
n_found = (found_elements_cpu >= 0).sum()
n_correct = (found_elements_cpu == test_element_ids).sum()

print("=" * 80)
print("RESULTS")
print("=" * 80)
print()
print(f"Particles tested: {n_test:,}")
print(f"Particles found: {n_found:,}/{n_test:,} ({100*n_found/n_test:.2f}%)")
print(f"Correct assignments: {n_correct:,}/{n_test:,} ({100*n_correct/n_test:.2f}%)")
print()

if n_found > 0:
    accuracy = n_correct / n_found * 100
    print(f"Accuracy (of found): {n_correct}/{n_found} ({accuracy:.2f}%)")
    print()

if n_correct < n_test * 0.95:
    print("✗ FAIL: Particles at exact centroids should be found in correct elements!")
    print()
    print("This confirms the hypothesis:")
    print("  Elements are assigned to octree leaves based on their CENTROIDS,")
    print("  but particles (even at exact centroids) navigate to different leaves!")
    print()
    print("ROOT CAUSE:")
    print("  An element may EXTEND into multiple octants, but is only stored in")
    print("  ONE octant (based on centroid). Particles inside the element but in")
    print("  different octants won't find it.")
    print()
else:
    print("✓ PASS: Particles at exact centroids found correctly")
    print()

# Show first few mismatches
if n_found > n_correct:
    print("=" * 80)
    print("FIRST 10 MISMATCHES")
    print("=" * 80)
    print()

    n_shown = 0
    for i in range(n_test):
        if found_elements_cpu[i] >= 0 and found_elements_cpu[i] != test_element_ids[i]:
            print(f"Particle {i}:")
            print(f"  Position (exact centroid): {test_particles[i]}")
            print(f"  True element ID: {test_element_ids[i]}")
            print(f"  Found element ID: {found_elements_cpu[i]}")
            print()
            n_shown += 1
            if n_shown >= 10:
                break

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

if n_correct < n_test * 0.95:
    print("The centroid-based octree assignment is FUNDAMENTALLY FLAWED for")
    print("tetrahedral meshes where elements can span multiple octants.")
    print()
    print("SOLUTION:")
    print("  Assign elements to ALL octree leaves they intersect (bounding-box based),")
    print("  not just the leaf containing their centroid.")
    print()
else:
    print("Centroid-based assignment works for exact centroids.")
    print("The issue must be with perturbation or particle placement.")
    print()
