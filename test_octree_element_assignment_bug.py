#!/usr/bin/env python3
"""
Debug octree element assignment bug.

The centroid hypothesis test showed that particles navigate to the SAME leaf
as their element centroids (99.1% match), but the true element is NOT in
that leaf's element list!

This means elements are being assigned to the WRONG leaves during construction.

This test finds where each element was actually placed.
"""

import numpy as np
from pathlib import Path

# Load mesh
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu

# Octree construction
from jaxtrace.gpu.search.octree_builder import build_octree_for_level

# Octree utilities
import vtk
from vtk.util import numpy_support

print("=" * 80)
print("OCTREE ELEMENT ASSIGNMENT BUG TEST")
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
print("Computing element centroids...")
element_centroids = np.array([
    node_positions[connectivity[i]].mean(axis=0)
    for i in range(len(connectivity))
], dtype=np.float32)
print(f"✓ Computed {len(element_centroids):,} centroids")
print()

element_ids = np.arange(len(connectivity), dtype=np.int32)

# Build octree
print("Building octree...")
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
print(f"  Nodes: {metadata['n_elements']:,}")
print(f"  Leaves: {metadata['n_leaves']:,}")
print(f"  Max depth: {metadata['max_depth']}")
print()

# ============================================================================
# Build Element -> Leaf Mapping
# ============================================================================
print("Building element -> leaf mapping...")

element_to_leaf = {}  # element_id -> leaf_id

for leaf_id, node in enumerate(nodes):
    if not node.is_leaf:
        continue

    # Get elements in this leaf
    leaf_elements = node.elements[node.elements >= 0]

    # Map each element to this leaf
    for elem_id in leaf_elements:
        if elem_id in element_to_leaf:
            print(f"⚠️  WARNING: Element {elem_id} assigned to multiple leaves: {element_to_leaf[elem_id]} and {leaf_id}")
        element_to_leaf[elem_id] = leaf_id

print(f"✓ Mapped {len(element_to_leaf):,} elements to leaves")
print()

# ============================================================================
# Test: Check Mismatches
# ============================================================================
print("=" * 80)
print("TEST: Compare where centroids navigate vs where elements are stored")
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

# Test first 100 elements that were assigned to leaves
test_element_ids = list(element_to_leaf.keys())[:1000]

n_match = 0
n_mismatch = 0

mismatch_examples = []

for elem_id in test_element_ids:
    # Where was element assigned during construction?
    assigned_leaf_id = element_to_leaf[elem_id]

    # Where does centroid navigate to?
    centroid = element_centroids[elem_id]
    navigated_leaf_id, path = navigate_octree(centroid, nodes)

    if assigned_leaf_id == navigated_leaf_id:
        n_match += 1
    else:
        n_mismatch += 1

        if len(mismatch_examples) < 10:
            assigned_leaf = nodes[assigned_leaf_id]
            navigated_leaf = nodes[navigated_leaf_id]

            mismatch_examples.append({
                'elem_id': elem_id,
                'centroid': centroid,
                'assigned_leaf_id': assigned_leaf_id,
                'navigated_leaf_id': navigated_leaf_id,
                'assigned_leaf_bbox': (assigned_leaf.bbox_min, assigned_leaf.bbox_max),
                'navigated_leaf_bbox': (navigated_leaf.bbox_min, navigated_leaf.bbox_max),
                'assigned_leaf_depth': assigned_leaf.depth,
                'navigated_leaf_depth': navigated_leaf.depth,
            })

print(f"Tested {len(test_element_ids)} elements:")
print(f"  Assigned leaf == Navigated leaf: {n_match}/{len(test_element_ids)} ({100*n_match/len(test_element_ids):.2f}%)")
print(f"  Assigned leaf != Navigated leaf: {n_mismatch}/{len(test_element_ids)} ({100*n_mismatch/len(test_element_ids):.2f}%)")
print()

if n_mismatch > 0:
    print("✗ BUG CONFIRMED!")
    print("  Elements are being assigned to different leaves than where their centroids navigate!")
    print()

    print("=" * 80)
    print("MISMATCH EXAMPLES")
    print("=" * 80)
    print()

    for i, detail in enumerate(mismatch_examples):
        print(f"Mismatch {i+1}:")
        print(f"  Element ID: {detail['elem_id']}")
        print(f"  Centroid: {detail['centroid']}")
        print(f"  Assigned to leaf: {detail['assigned_leaf_id']} (depth {detail['assigned_leaf_depth']})")
        print(f"    Bbox: min={detail['assigned_leaf_bbox'][0]}, max={detail['assigned_leaf_bbox'][1]}")
        print(f"  Centroid navigates to leaf: {detail['navigated_leaf_id']} (depth {detail['navigated_leaf_depth']})")
        print(f"    Bbox: min={detail['navigated_leaf_bbox'][0]}, max={detail['navigated_leaf_bbox'][1]}")

        # Check if centroid is inside assigned leaf bbox
        c = detail['centroid']
        amin, amax = detail['assigned_leaf_bbox']
        inside_assigned = np.all((c >= amin) & (c <= amax))
        print(f"  Centroid inside assigned leaf bbox: {inside_assigned}")

        # Check if centroid is inside navigated leaf bbox
        nmin, nmax = detail['navigated_leaf_bbox']
        inside_navigated = np.all((c >= nmin) & (c <= nmax))
        print(f"  Centroid inside navigated leaf bbox: {inside_navigated}")
        print()

else:
    print("✓ NO BUG: All elements assigned to correct leaves")
    print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

if n_mismatch > 0:
    print("The octree construction has a bug where elements are assigned to leaves")
    print("that don't match where their centroids navigate during search.")
    print()
    print("This is the ROOT CAUSE of the 99.97% inaccuracy!")
    print()
else:
    print("Element assignment matches navigation. Bug must be elsewhere.")
    print()
