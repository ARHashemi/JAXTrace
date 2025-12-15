#!/usr/bin/env python3
"""
Validate octree centroid assignment hypothesis.

This test checks if the 99.97% inaccuracy is due to elements being assigned
to octree leaves based on centroids, causing particles inside those elements
to search in the wrong octree leaves.

Test procedure:
1. Load mesh and build octree
2. Generate particles at element centroids with small perturbations
3. For each particle:
   a. Navigate octree to find which leaf it reaches
   b. Check if the true element is in that leaf's element list
   c. If not, we've confirmed the centroid assignment bug
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Tuple

# Load mesh
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu

# Octree construction
from jaxtrace.gpu.search.octree_builder import build_octree_for_level, flatten_octree_to_arrays

# Octree utilities
import vtk
from vtk.util import numpy_support

print("=" * 80)
print("OCTREE CENTROID ASSIGNMENT HYPOTHESIS TEST")
print("=" * 80)
print()

# ============================================================================
# Load Mesh
# ============================================================================
mesh_path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_120.pvtu")
print(f"Loading mesh: {mesh_path.name}")
node_positions, connectivity, velocity_field = load_mesh_from_pvtu(mesh_path, field_name='Displacement')
print(f"✓ Loaded: {len(node_positions):,} nodes, {len(connectivity):,} elements")
print()

# ============================================================================
# Build Octree
# ============================================================================
print("Building octree...")

# Load LEVEL field
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

# Compute element centroids
element_centroids = np.array([
    node_positions[connectivity[i]].mean(axis=0)
    for i in range(len(connectivity))
], dtype=np.float32)

element_ids = np.arange(len(connectivity), dtype=np.int32)

# Build octree
nodes, metadata = build_octree_for_level(
    element_centroids,
    element_ids,
    level_field=level_field,
    level_threshold=1.1,
    max_depth=15,
    max_leaf_size=50,
    use_levelset=True
)

print(f"✓ Built octree")
print(f"  Nodes: {metadata['n_nodes']:,}")
print(f"  Leaves: {metadata['n_leaves']:,}")
print(f"  Max depth: {metadata['max_depth']}")
print()

# Flatten for GPU
node_metadata_np, node_elements_np = flatten_octree_to_arrays(nodes, max_leaf_size=50)

# ============================================================================
# Generate Test Particles
# ============================================================================
print("Generating test particles...")
np.random.seed(42)

n_test = 1000  # Test 1000 particles

# Sample random elements
test_element_ids = np.random.choice(len(connectivity), size=n_test, replace=False)

# Compute minimum element size for perturbation
min_edge_length = float('inf')
for i in range(min(1000, len(connectivity))):
    nodes_elem = node_positions[connectivity[i]]
    edges = [
        np.linalg.norm(nodes_elem[1] - nodes_elem[0]),
        np.linalg.norm(nodes_elem[2] - nodes_elem[0]),
        np.linalg.norm(nodes_elem[3] - nodes_elem[0]),
        np.linalg.norm(nodes_elem[2] - nodes_elem[1]),
        np.linalg.norm(nodes_elem[3] - nodes_elem[1]),
        np.linalg.norm(nodes_elem[3] - nodes_elem[2])
    ]
    min_edge_length = min(min_edge_length, min(edges))

perturbation_scale = 0.01 * min_edge_length

# Generate particles at centroids with small perturbation
test_particles = []
for elem_id in test_element_ids:
    centroid = element_centroids[elem_id]
    perturbation = np.random.uniform(-perturbation_scale, perturbation_scale, size=3)
    particle_pos = centroid + perturbation
    test_particles.append(particle_pos)

test_particles = np.array(test_particles, dtype=np.float32)
print(f"✓ Generated {n_test} test particles")
print(f"  Perturbation scale: {perturbation_scale:.6e}")
print()

# ============================================================================
# Test: Manual Octree Navigation
# ============================================================================
print("=" * 80)
print("TEST: Check if true elements are in reached octree leaves")
print("=" * 80)
print()

def compute_octant_np(pos, bbox_min, bbox_max):
    """Compute octant index (0-7) for position."""
    bbox_mid = (bbox_min + bbox_max) / 2.0
    octant = (
        int(pos[0] >= bbox_mid[0]) +
        (int(pos[1] >= bbox_mid[1]) << 1) +
        (int(pos[2] >= bbox_mid[2]) << 2)
    )
    return octant

def navigate_octree(pos, nodes):
    """Navigate octree to find leaf containing position."""
    node_id = 0  # Start at root
    path = [0]

    for depth in range(20):  # Max depth limit
        node = nodes[node_id]

        if node.is_leaf:
            return node_id, path

        # Compute octant
        octant = compute_octant_np(pos, node.bbox_min, node.bbox_max)
        child_id = node.children[octant]

        if child_id < 0:
            # No child in this octant, stay at current node
            return node_id, path

        # Move to child
        node_id = child_id
        path.append(node_id)

    # Should not reach here
    return node_id, path

# Test each particle
n_true_elem_in_leaf = 0
n_true_elem_not_in_leaf = 0

mismatch_details = []

for i, (particle_pos, true_elem_id) in enumerate(zip(test_particles, test_element_ids)):
    # Navigate to leaf
    leaf_id, path = navigate_octree(particle_pos, nodes)

    # Get elements in leaf
    leaf_elements = nodes[leaf_id].elements
    leaf_elements = leaf_elements[leaf_elements >= 0]  # Remove padding

    # Check if true element is in leaf
    if true_elem_id in leaf_elements:
        n_true_elem_in_leaf += 1
    else:
        n_true_elem_not_in_leaf += 1

        # Save mismatch details
        if len(mismatch_details) < 10:  # Save first 10 mismatches
            true_elem_centroid = element_centroids[true_elem_id]
            true_leaf_id, true_path = navigate_octree(true_elem_centroid, nodes)

            mismatch_details.append({
                'particle_id': i,
                'particle_pos': particle_pos,
                'true_elem_id': true_elem_id,
                'true_elem_centroid': true_elem_centroid,
                'particle_leaf_id': leaf_id,
                'true_centroid_leaf_id': true_leaf_id,
                'particle_path': path,
                'true_centroid_path': true_path,
                'n_elements_in_particle_leaf': len(leaf_elements)
            })

    # Progress
    if (i + 1) % 100 == 0:
        print(f"  Tested {i+1}/{n_test} particles...")

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()
print(f"True element in reached leaf:     {n_true_elem_in_leaf}/{n_test} ({100*n_true_elem_in_leaf/n_test:.2f}%)")
print(f"True element NOT in reached leaf: {n_true_elem_not_in_leaf}/{n_test} ({100*n_true_elem_not_in_leaf/n_test:.2f}%)")
print()

if n_true_elem_not_in_leaf > 0:
    print("✗ HYPOTHESIS CONFIRMED!")
    print("  Elements are assigned to wrong octree leaves due to centroid-based assignment.")
    print()
    print("  This explains the 99.97% inaccuracy:")
    print("  - Particle navigates to leaf A based on its position")
    print("  - True element is in leaf B (based on element's centroid)")
    print("  - Particle searches in leaf A, finds wrong element (or nothing)")
    print()

    print("=" * 80)
    print("MISMATCH EXAMPLES")
    print("=" * 80)
    print()

    for j, detail in enumerate(mismatch_details):
        print(f"Mismatch {j+1}:")
        print(f"  Particle position:        {detail['particle_pos']}")
        print(f"  True element ID:          {detail['true_elem_id']}")
        print(f"  True element centroid:    {detail['true_elem_centroid']}")
        print(f"  Particle navigated to leaf:  {detail['particle_leaf_id']} (path: {' -> '.join(map(str, detail['particle_path']))})")
        print(f"  True centroid navigates to:  {detail['true_centroid_leaf_id']} (path: {' -> '.join(map(str, detail['true_centroid_path']))})")
        print(f"  Elements in particle's leaf: {detail['n_elements_in_particle_leaf']}")
        print(f"  Distance particle to centroid: {np.linalg.norm(detail['particle_pos'] - detail['true_elem_centroid']):.6e}")
        print()

else:
    print("✓ HYPOTHESIS REJECTED")
    print("  All particles navigated to leaves containing their true elements.")
    print("  The inaccuracy must come from another source.")
    print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

if n_true_elem_not_in_leaf / n_test > 0.10:  # If >10% mismatches
    print("The centroid-based element assignment is causing particles to search")
    print("in the wrong octree leaves, leading to wrong or missing element assignments.")
    print()
    print("RECOMMENDED FIX:")
    print("1. Implement multi-octant search (check neighboring leaves)")
    print("2. Or: Assign elements to ALL overlapping leaves (bounding-box based)")
    print()
else:
    print("The centroid assignment is working reasonably well (<10% mismatches).")
    print("The 99.97% inaccuracy must come from another bug.")
    print()
