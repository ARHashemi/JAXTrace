#!/usr/bin/env python3
"""
Test Node-Based Neighbor Construction
======================================

Test the new node-based neighbor algorithm and verify:
1. It finds fine element neighbors of coarse boundary elements
2. Memory usage is acceptable for GPU
3. Comparison with face-based neighbors
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
print("NODE-BASED NEIGHBOR CONSTRUCTION TEST")
print("="*80)

# Load mesh
print("\n[1/5] Loading mesh...")
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 120)  # Only load one timestep
VELOCITY_FIELD_NAME = 'Displacement'

t0 = time.time()
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    MESH_BASE_PATH,
    MESH_FILE_PATTERN,
    VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=False
)
t_load = time.time() - t0

print(f"  Loaded in {t_load:.2f}s")
print(f"  Nodes: {len(node_positions):,}")
print(f"  Elements: {len(connectivity):,}")

# Classify elements
print("\n[2/5] Classifying elements...")
element_sizes = np.zeros(len(connectivity))
element_centroids = np.zeros((len(connectivity), 3))

for i, elem_nodes_idx in enumerate(connectivity):
    elem_nodes = node_positions[elem_nodes_idx]
    element_centroids[i] = elem_nodes.mean(axis=0)
    max_edge = max([np.linalg.norm(elem_nodes[j] - elem_nodes[k])
                    for j in range(4) for k in range(j+1, 4)])
    element_sizes[i] = max_edge

fine_threshold = np.percentile(element_sizes, 10)
fine_mask = element_sizes < fine_threshold
coarse_mask = element_sizes > np.percentile(element_sizes, 90)

print(f"  Fine elements (<{fine_threshold*1000:.4f}mm): {fine_mask.sum():,}")
print(f"  Coarse elements: {coarse_mask.sum():,}")

# Build face-based neighbors (baseline)
print("\n[3/5] Building FACE-BASED neighbors...")
t0 = time.time()
face_neighbors = build_element_neighbors_array(connectivity, verbose=True, method='face')
t_face = time.time() - t0

face_memory_mb = (face_neighbors.nbytes) / (1024**2)
print(f"  Built in {t_face:.2f}s")
print(f"  Memory: {face_memory_mb:.1f} MB")
print(f"  Shape: {face_neighbors.shape}")

# Build node-based neighbors (NEW)
print("\n[4/5] Building NODE-BASED neighbors...")
t0 = time.time()
node_neighbors = build_element_neighbors_array(connectivity, verbose=True, method='node')
t_node = time.time() - t0

node_memory_mb = (node_neighbors.nbytes) / (1024**2)
print(f"  Built in {t_node:.2f}s")
print(f"  Memory: {node_memory_mb:.1f} MB")
print(f"  Shape: {node_neighbors.shape}")

# Compare memory
print(f"\n  Memory comparison:")
print(f"    Face-based: {face_memory_mb:.1f} MB")
print(f"    Node-based: {node_memory_mb:.1f} MB")
print(f"    Ratio: {node_memory_mb/face_memory_mb:.1f}x")

# Verify node-based finds fine neighbors for coarse boundary elements
print("\n[5/5] Verifying fine neighbor detection...")

# Find refined region boundary
fine_centroids = element_centroids[fine_mask]
refined_bbox_min = fine_centroids.min(axis=0)
refined_bbox_max = fine_centroids.max(axis=0)

# Expand bbox slightly
margin = 0.001  # 1mm
boundary_bbox_min = refined_bbox_min - margin
boundary_bbox_max = refined_bbox_max + margin

# Find coarse elements near boundary
boundary_coarse_indices = []
for i in range(len(connectivity)):
    if coarse_mask[i]:
        centroid = element_centroids[i]
        near_boundary = (np.all(centroid >= boundary_bbox_min) and
                        np.all(centroid <= boundary_bbox_max))
        if near_boundary:
            boundary_coarse_indices.append(i)

boundary_coarse_indices = np.array(boundary_coarse_indices)
print(f"  Coarse elements near refined boundary: {len(boundary_coarse_indices):,}")

if len(boundary_coarse_indices) > 0:
    # Sample 20 boundary elements
    sample_indices = boundary_coarse_indices[:min(20, len(boundary_coarse_indices))]

    # Count fine neighbors for each method
    face_fine_counts = []
    node_fine_counts = []

    for elem_id in sample_indices:
        # Face-based neighbors
        face_neighs = face_neighbors[elem_id]
        face_valid = face_neighs[face_neighs >= 0]
        face_fine = sum(1 for n in face_valid if fine_mask[n])
        face_fine_counts.append(face_fine)

        # Node-based neighbors
        node_neighs = node_neighbors[elem_id]
        node_valid = node_neighs[node_neighs >= 0]
        node_fine = sum(1 for n in node_valid if fine_mask[n])
        node_fine_counts.append(node_fine)

    print(f"\n  Boundary coarse element statistics (sample of {len(sample_indices)}):")
    print(f"\n  FACE-BASED neighbors:")
    print(f"    Fine neighbors: mean={np.mean(face_fine_counts):.2f}, min={np.min(face_fine_counts)}, max={np.max(face_fine_counts)}")
    print(f"    Elements with 0 fine neighbors: {sum(1 for c in face_fine_counts if c == 0)}/{len(sample_indices)}")

    print(f"\n  NODE-BASED neighbors:")
    print(f"    Fine neighbors: mean={np.mean(node_fine_counts):.2f}, min={np.min(node_fine_counts)}, max={np.max(node_fine_counts)}")
    print(f"    Elements with 0 fine neighbors: {sum(1 for c in node_fine_counts if c == 0)}/{len(sample_indices)}")

    # Success criterion
    node_with_fine = sum(1 for c in node_fine_counts if c > 0)
    face_with_fine = sum(1 for c in face_fine_counts if c > 0)

    print(f"\n  Results:")
    print(f"    Face-based: {face_with_fine}/{len(sample_indices)} boundary elements have fine neighbors")
    print(f"    Node-based: {node_with_fine}/{len(sample_indices)} boundary elements have fine neighbors")

    if node_with_fine > face_with_fine:
        print(f"\n  ✅ SUCCESS: Node-based finds {node_with_fine - face_with_fine} more boundary elements with fine neighbors!")
    else:
        print(f"\n  ⚠️  WARNING: Node-based didn't find more fine neighbors than face-based")

# Test specific problem elements from tracking diagnostic
print(f"\n{'='*80}")
print("TESTING SPECIFIC PROBLEM ELEMENTS")
print(f"{'='*80}\n")

problem_elements = [1793360, 1793477]

for elem_id in problem_elements:
    if elem_id < len(connectivity):
        # Face-based
        face_neighs = face_neighbors[elem_id]
        face_valid = face_neighs[face_neighs >= 0]
        face_fine = sum(1 for n in face_valid if fine_mask[n])

        # Node-based
        node_neighs = node_neighbors[elem_id]
        node_valid = node_neighs[node_neighs >= 0]
        node_fine = sum(1 for n in node_valid if fine_mask[n])

        print(f"Element {elem_id}:")
        print(f"  Size: {element_sizes[elem_id]*1000:.4f} mm")
        print(f"  Centroid: ({element_centroids[elem_id][0]*1000:.2f}, {element_centroids[elem_id][1]*1000:.2f}, {element_centroids[elem_id][2]*1000:.2f}) mm")
        print(f"\n  Face-based neighbors:")
        print(f"    Total: {len(face_valid)}")
        print(f"    Fine: {face_fine}")
        print(f"    Status: {'❌ ZERO fine neighbors' if face_fine == 0 else '✅'}")
        print(f"\n  Node-based neighbors:")
        print(f"    Total: {len(node_valid)}")
        print(f"    Fine: {node_fine}")
        print(f"    Status: {'❌ ZERO fine neighbors' if node_fine == 0 else f'✅ {node_fine} fine neighbors found!'}")

        if node_fine > 0:
            print(f"\n  Sample of fine neighbors from node-based:")
            fine_neighbor_ids = [n for n in node_valid if fine_mask[n]]
            for i, n in enumerate(fine_neighbor_ids[:5]):
                print(f"    {n}: size={element_sizes[n]*1000:.4f}mm")

        print()

print("="*80)
print("SUMMARY")
print("="*80)

print(f"\nMemory requirements:")
print(f"  Face-based: {face_memory_mb:.1f} MB")
print(f"  Node-based: {node_memory_mb:.1f} MB")
print(f"  Additional: {node_memory_mb - face_memory_mb:.1f} MB")

if node_memory_mb < 2000:  # 2 GB threshold
    print(f"\n✅ Node-based neighbor array should fit in GPU memory (<2GB)")
else:
    print(f"\n⚠️  Node-based neighbor array may be too large for GPU memory (>{node_memory_mb/1024:.1f}GB)")

print(f"\nRecommendation:")
if node_fine_counts and np.mean(node_fine_counts) > 0:
    print(f"  Use method='node' for production tracking to capture rotation in refined region")
    print(f"  Expected improvement: L1 search will find fine elements from coarse boundary")
else:
    print(f"  Node-based didn't improve fine neighbor detection - investigate further")

print("="*80)
