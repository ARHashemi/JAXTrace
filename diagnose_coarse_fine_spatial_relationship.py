#!/usr/bin/env python3
"""
Diagnose Spatial Relationship Between Coarse and Fine Elements
===============================================================

Investigate why coarse elements have NO fine neighbors even with node-based
neighbor detection. Check if they actually share nodes or if there's a
spatial gap between refined and coarse regions.
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu

print("="*80)
print("COARSE-FINE SPATIAL RELATIONSHIP ANALYSIS")
print("="*80)

# Load mesh
print("\n[1/3] Loading mesh...")
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 120)
VELOCITY_FIELD_NAME = 'Displacement'

node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    MESH_BASE_PATH,
    MESH_FILE_PATTERN,
    VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=False
)

print(f"  Nodes: {len(node_positions):,}")
print(f"  Elements: {len(connectivity):,}")

# Classify elements
print("\n[2/3] Classifying elements by size...")
element_sizes = np.zeros(len(connectivity))
element_centroids = np.zeros((len(connectivity), 3))

for i, elem_nodes_idx in enumerate(connectivity):
    elem_nodes = node_positions[elem_nodes_idx]
    element_centroids[i] = elem_nodes.mean(axis=0)
    max_edge = max([np.linalg.norm(elem_nodes[j] - elem_nodes[k])
                    for j in range(4) for k in range(j+1, 4)])
    element_sizes[i] = max_edge

# Use actual size thresholds rather than percentiles
fine_threshold = 0.00015  # 0.15mm
medium_max = 0.0003  # 0.3mm

fine_mask = element_sizes <= fine_threshold
medium_mask = (element_sizes > fine_threshold) & (element_sizes <= medium_max)
coarse_mask = element_sizes > medium_max

print(f"  Fine elements (≤{fine_threshold*1000:.2f}mm): {fine_mask.sum():,}")
print(f"  Medium elements ({fine_threshold*1000:.2f}-{medium_max*1000:.2f}mm): {medium_mask.sum():,}")
print(f"  Coarse elements (>{medium_max*1000:.2f}mm): {coarse_mask.sum():,}")

# Find element near problem location (-10mm, 0, -2mm)
print("\n[3/3] Analyzing elements near tracking test location...")
test_location = np.array([-0.010, 0.0, -0.002])  # -10mm, 0, -2mm

# Find closest fine and coarse elements
fine_indices = np.where(fine_mask)[0]
coarse_indices = np.where(coarse_mask)[0]

fine_distances = np.linalg.norm(element_centroids[fine_indices] - test_location, axis=1)
coarse_distances = np.linalg.norm(element_centroids[coarse_indices] - test_location, axis=1)

closest_fine_idx = fine_indices[np.argmin(fine_distances)]
closest_coarse_idx = coarse_indices[np.argmin(coarse_distances)]

print(f"\nAt test location ({test_location[0]*1000:.2f}, {test_location[1]*1000:.2f}, {test_location[2]*1000:.2f}) mm:")
print(f"\nClosest FINE element:")
print(f"  Element ID: {closest_fine_idx}")
print(f"  Size: {element_sizes[closest_fine_idx]*1000:.4f} mm")
print(f"  Centroid: ({element_centroids[closest_fine_idx][0]*1000:.2f}, {element_centroids[closest_fine_idx][1]*1000:.2f}, {element_centroids[closest_fine_idx][2]*1000:.2f}) mm")
print(f"  Distance: {fine_distances[np.argmin(fine_distances)]*1000:.4f} mm")
print(f"  Nodes: {connectivity[closest_fine_idx]}")

print(f"\nClosest COARSE element:")
print(f"  Element ID: {closest_coarse_idx}")
print(f"  Size: {element_sizes[closest_coarse_idx]*1000:.4f} mm")
print(f"  Centroid: ({element_centroids[closest_coarse_idx][0]*1000:.2f}, {element_centroids[closest_coarse_idx][1]*1000:.2f}, {element_centroids[closest_coarse_idx][2]*1000:.2f}) mm")
print(f"  Distance: {coarse_distances[np.argmin(coarse_distances)]*1000:.4f} mm")
print(f"  Nodes: {connectivity[closest_coarse_idx]}")

# Check for shared nodes
fine_nodes = set(connectivity[closest_fine_idx])
coarse_nodes = set(connectivity[closest_coarse_idx])
shared_nodes = fine_nodes.intersection(coarse_nodes)

print(f"\nShared nodes between closest fine and coarse elements:")
if len(shared_nodes) > 0:
    print(f"  ✅ {len(shared_nodes)} shared nodes: {sorted(shared_nodes)}")
else:
    print(f"  ❌ NO shared nodes")

# Find ALL fine elements within 2mm of closest coarse element
print(f"\nFine elements within 2mm of closest coarse element:")
coarse_centroid = element_centroids[closest_coarse_idx]
nearby_fine_mask = np.linalg.norm(element_centroids[fine_indices] - coarse_centroid, axis=1) < 0.002
nearby_fine_indices = fine_indices[nearby_fine_mask]

print(f"  Found {len(nearby_fine_indices)} fine elements nearby")

if len(nearby_fine_indices) > 0:
    # Check if any share nodes with coarse element
    any_shared = False
    for fine_idx in nearby_fine_indices[:10]:  # Check first 10
        fine_nodes_set = set(connectivity[fine_idx])
        if len(fine_nodes_set.intersection(coarse_nodes)) > 0:
            any_shared = True
            shared = fine_nodes_set.intersection(coarse_nodes)
            print(f"    Element {fine_idx}: {len(shared)} shared nodes")

    if not any_shared:
        print(f"  ❌ None of the nearby fine elements share nodes with coarse element!")
        print(f"  This explains why node-based neighbors didn't find them.")

# Spatial distribution analysis
print(f"\n{'='*80}")
print("SPATIAL DISTRIBUTION ANALYSIS")
print(f"{'='*80}\n")

# Find bounding boxes
fine_bbox_min = element_centroids[fine_indices].min(axis=0)
fine_bbox_max = element_centroids[fine_indices].max(axis=0)

medium_indices = np.where(medium_mask)[0]
if len(medium_indices) > 0:
    medium_bbox_min = element_centroids[medium_indices].min(axis=0)
    medium_bbox_max = element_centroids[medium_indices].max(axis=0)
else:
    medium_bbox_min = medium_bbox_max = np.array([0, 0, 0])

coarse_bbox_min = element_centroids[coarse_indices].min(axis=0)
coarse_bbox_max = element_centroids[coarse_indices].max(axis=0)

print(f"Fine region bounding box:")
print(f"  X: [{fine_bbox_min[0]*1000:.2f}, {fine_bbox_max[0]*1000:.2f}] mm")
print(f"  Y: [{fine_bbox_min[1]*1000:.2f}, {fine_bbox_max[1]*1000:.2f}] mm")
print(f"  Z: [{fine_bbox_min[2]*1000:.2f}, {fine_bbox_max[2]*1000:.2f}] mm")

if len(medium_indices) > 0:
    print(f"\nMedium region bounding box:")
    print(f"  X: [{medium_bbox_min[0]*1000:.2f}, {medium_bbox_max[0]*1000:.2f}] mm")
    print(f"  Y: [{medium_bbox_min[1]*1000:.2f}, {medium_bbox_max[1]*1000:.2f}] mm")
    print(f"  Z: [{medium_bbox_min[2]*1000:.2f}, {medium_bbox_max[2]*1000:.2f}] mm")

print(f"\nCoarse region bounding box:")
print(f"  X: [{coarse_bbox_min[0]*1000:.2f}, {coarse_bbox_max[0]*1000:.2f}] mm")
print(f"  Y: [{coarse_bbox_min[1]*1000:.2f}, {coarse_bbox_max[1]*1000:.2f}] mm")
print(f"  Z: [{coarse_bbox_min[2]*1000:.2f}, {coarse_bbox_max[2]*1000:.2f}] mm")

# Check overlap
print(f"\nRegion overlap:")
fine_coarse_overlap_x = (fine_bbox_min[0] < coarse_bbox_max[0]) and (fine_bbox_max[0] > coarse_bbox_min[0])
fine_coarse_overlap_y = (fine_bbox_min[1] < coarse_bbox_max[1]) and (fine_bbox_max[1] > coarse_bbox_min[1])
fine_coarse_overlap_z = (fine_bbox_min[2] < coarse_bbox_max[2]) and (fine_bbox_max[2] > coarse_bbox_min[2])
fine_coarse_overlap = fine_coarse_overlap_x and fine_coarse_overlap_y and fine_coarse_overlap_z

if fine_coarse_overlap:
    print(f"  ✅ Fine and coarse regions OVERLAP spatially")
else:
    print(f"  ❌ Fine and coarse regions DO NOT overlap!")

# Check if there's a medium buffer zone
if len(medium_indices) > 0:
    medium_fine_overlap = (
        (medium_bbox_min[0] < fine_bbox_max[0]) and (medium_bbox_max[0] > fine_bbox_min[0]) and
        (medium_bbox_min[1] < fine_bbox_max[1]) and (medium_bbox_max[1] > fine_bbox_min[1]) and
        (medium_bbox_min[2] < fine_bbox_max[2]) and (medium_bbox_max[2] > fine_bbox_min[2])
    )
    medium_coarse_overlap = (
        (medium_bbox_min[0] < coarse_bbox_max[0]) and (medium_bbox_max[0] > coarse_bbox_min[0]) and
        (medium_bbox_min[1] < coarse_bbox_max[1]) and (medium_bbox_max[1] > coarse_bbox_min[1]) and
        (medium_bbox_min[2] < coarse_bbox_max[2]) and (medium_bbox_max[2] > coarse_bbox_min[2])
    )

    if medium_fine_overlap and medium_coarse_overlap:
        print(f"  ℹ️  Medium-sized elements form a BUFFER zone between fine and coarse")
        print(f"     Fine → Medium → Coarse (NOT direct fine-coarse contact)")

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}\n")

if not fine_coarse_overlap:
    print("Fine and coarse regions don't overlap - they're in different spatial locations!")
    print("This mesh may not have direct coarse-fine boundaries.")
elif len(medium_indices) > 0 and medium_fine_overlap and medium_coarse_overlap:
    print("The mesh has a GRADED refinement structure:")
    print("  Fine → Medium → Coarse")
    print("\nThis means:")
    print("  - Fine elements share nodes/edges with MEDIUM elements")
    print("  - Medium elements share nodes/edges with COARSE elements")
    print("  - Fine and Coarse elements DON'T share nodes directly")
    print("\nImplication:")
    print("  L1 neighbor search from COARSE element will find MEDIUM neighbors,")
    print("  but needs 2+ hops to reach FINE elements!")
    print(f"\nCurrent N_HOPS = 3 may be insufficient.")
    print(f"Try increasing N_HOPS to 5-10 to traverse coarse→medium→fine.")
else:
    print("Unexpected mesh structure - investigate further")

print("="*80)
