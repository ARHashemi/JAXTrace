#!/usr/bin/env python3
"""
Diagnose Neighbor Connectivity at Refined/Coarse Boundary
==========================================================

Check if fine elements are neighbors of coarse elements at the
boundary of the refined region.

If fine elements are NOT in neighbor lists of coarse boundary elements,
then L1 search will never find them!
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.forest import build_element_neighbors_array

print("="*80)
print("NEIGHBOR CONNECTIVITY ANALYSIS")
print("="*80)

# Load mesh
print("\n[1/3] Loading mesh...")
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 120)  # Only load one timestep
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
print("\n[2/3] Classifying elements...")
element_sizes = np.zeros(len(connectivity))
element_centroids = np.zeros((len(connectivity), 3))

for i, elem_nodes_idx in enumerate(connectivity):
    elem_nodes = node_positions[elem_nodes_idx]
    element_centroids[i] = elem_nodes.mean(axis=0)
    max_edge = max([np.linalg.norm(elem_nodes[j] - elem_nodes[k])
                    for j in range(4) for k in range(j+1, 4)])
    element_sizes[i] = max_edge

fine_threshold = np.percentile(element_sizes, 10)
coarse_threshold = np.percentile(element_sizes, 90)

fine_mask = element_sizes < fine_threshold
medium_mask = (element_sizes >= fine_threshold) & (element_sizes < coarse_threshold)
coarse_mask = element_sizes >= coarse_threshold

print(f"  Fine elements (<{fine_threshold*1000:.4f}mm): {fine_mask.sum():,}")
print(f"  Medium elements: {medium_mask.sum():,}")
print(f"  Coarse elements (>{coarse_threshold*1000:.4f}mm): {coarse_mask.sum():,}")

# Build neighbor array
print("\n[3/3] Building element neighbors...")
t0 = time.time()
element_neighbors = build_element_neighbors_array(connectivity)
t_build = time.time() - t0
print(f"  Built in {t_build:.2f}s")

# Find refined region boundary
fine_centroids = element_centroids[fine_mask]
refined_bbox_min = fine_centroids.min(axis=0)
refined_bbox_max = fine_centroids.max(axis=0)

# Expand bbox slightly to find boundary elements
margin = 0.001  # 1mm margin
boundary_bbox_min = refined_bbox_min - margin
boundary_bbox_max = refined_bbox_max + margin

# Find coarse elements near refined region boundary
boundary_coarse_mask = coarse_mask.copy()
for i in range(len(connectivity)):
    if coarse_mask[i]:
        centroid = element_centroids[i]
        near_boundary = (np.all(centroid >= boundary_bbox_min) and
                        np.all(centroid <= boundary_bbox_max))
        boundary_coarse_mask[i] = near_boundary

boundary_coarse_indices = np.where(boundary_coarse_mask)[0]

print(f"\n  Coarse elements near refined boundary: {len(boundary_coarse_indices):,}")

# Sample some boundary coarse elements
print(f"\nAnalyzing neighbor connectivity...")

if len(boundary_coarse_indices) > 0:
    # Sample up to 20 boundary elements
    sample_indices = boundary_coarse_indices[:min(20, len(boundary_coarse_indices))]

    fine_neighbor_counts = []
    medium_neighbor_counts = []
    coarse_neighbor_counts = []
    total_neighbor_counts = []

    for elem_id in sample_indices:
        neighbors = element_neighbors[elem_id]
        valid_neighbors = neighbors[neighbors >= 0]

        n_fine = sum(1 for n in valid_neighbors if fine_mask[n])
        n_medium = sum(1 for n in valid_neighbors if medium_mask[n])
        n_coarse = sum(1 for n in valid_neighbors if coarse_mask[n])

        fine_neighbor_counts.append(n_fine)
        medium_neighbor_counts.append(n_medium)
        coarse_neighbor_counts.append(n_coarse)
        total_neighbor_counts.append(len(valid_neighbors))

    print(f"\nBoundary coarse element neighbor statistics (sample of {len(sample_indices)}):")
    print(f"  Fine neighbors per element:")
    print(f"    Mean: {np.mean(fine_neighbor_counts):.2f}")
    print(f"    Min: {np.min(fine_neighbor_counts)}")
    print(f"    Max: {np.max(fine_neighbor_counts)}")
    print(f"  Medium neighbors per element:")
    print(f"    Mean: {np.mean(medium_neighbor_counts):.2f}")
    print(f"  Coarse neighbors per element:")
    print(f"    Mean: {np.mean(coarse_neighbor_counts):.2f}")
    print(f"  Total neighbors per element:")
    print(f"    Mean: {np.mean(total_neighbor_counts):.2f}")

    # Count elements with ZERO fine neighbors
    n_zero_fine = sum(1 for n in fine_neighbor_counts if n == 0)
    pct_zero_fine = 100 * n_zero_fine / len(sample_indices)

    print(f"\n  Coarse boundary elements with ZERO fine neighbors: {n_zero_fine}/{len(sample_indices)} ({pct_zero_fine:.1f}%)")

    if pct_zero_fine > 50:
        print(f"\n  ❌ PROBLEM FOUND!")
        print(f"     Most coarse boundary elements have NO fine neighbors!")
        print(f"     L1 search (neighbor hops) will NEVER find fine elements!")
        print(f"     Particles will remain stuck in coarse elements even when")
        print(f"     they move into refined region.")
    else:
        print(f"\n  ✅ Good: Most boundary elements have fine neighbors")

    # Show detailed example
    print(f"\nDetailed example (first boundary element):")
    elem_id = sample_indices[0]
    neighbors = element_neighbors[elem_id]
    valid_neighbors = neighbors[neighbors >= 0]

    print(f"  Element {elem_id}:")
    print(f"    Size: {element_sizes[elem_id]*1000:.4f} mm (coarse)")
    print(f"    Centroid: ({element_centroids[elem_id][0]*1000:.2f}, {element_centroids[elem_id][1]*1000:.2f}, {element_centroids[elem_id][2]*1000:.2f}) mm")
    print(f"    Neighbors ({len(valid_neighbors)}):")
    for n in valid_neighbors:
        if fine_mask[n]:
            neighbor_type = "FINE  "
        elif medium_mask[n]:
            neighbor_type = "medium"
        else:
            neighbor_type = "coarse"
        print(f"      {n:7d}: {neighbor_type}, size={element_sizes[n]*1000:.4f}mm")

# Check specific elements from tracking diagnostic
print(f"\n{'='*80}")
print("CHECKING SPECIFIC ELEMENTS FROM TRACKING DIAGNOSTIC")
print(f"{'='*80}\n")

# From tracking diagnostic, particles were stuck in elements 1793360 and 1793477
problem_elements = [1793360, 1793477]

for elem_id in problem_elements:
    if elem_id < len(connectivity):
        neighbors = element_neighbors[elem_id]
        valid_neighbors = neighbors[neighbors >= 0]

        n_fine = sum(1 for n in valid_neighbors if fine_mask[n])
        n_medium = sum(1 for n in valid_neighbors if medium_mask[n])
        n_coarse = sum(1 for n in valid_neighbors if coarse_mask[n])

        print(f"Element {elem_id} (from tracking diagnostic):")
        print(f"  Size: {element_sizes[elem_id]*1000:.4f} mm ({'FINE' if fine_mask[elem_id] else 'COARSE'})")
        print(f"  Centroid: ({element_centroids[elem_id][0]*1000:.2f}, {element_centroids[elem_id][1]*1000:.2f}, {element_centroids[elem_id][2]*1000:.2f}) mm")
        print(f"  Neighbors: {len(valid_neighbors)} total")
        print(f"    Fine: {n_fine}")
        print(f"    Medium: {n_medium}")
        print(f"    Coarse: {n_coarse}")

        if n_fine == 0:
            print(f"  ❌ This element has ZERO fine neighbors!")
            print(f"     L1 search will never find fine elements from here.")
        print()

print("="*80)
