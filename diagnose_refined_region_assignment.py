#!/usr/bin/env python3
"""
Diagnose Element Assignment in Refined Regions
===============================================

Check if particles in refined (small element) regions are being correctly
assigned to their actual containing elements, or if search is failing and
assigning them to wrong (larger) elements.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.tracking.initial_assignment_extended import initial_assignment_extended_batch

print("="*80)
print("REFINED REGION ASSIGNMENT DIAGNOSIS")
print("="*80)

# Load mesh
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 120)
VELOCITY_FIELD_NAME = 'Displacement'

print("\n[1/4] Loading mesh...")
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    MESH_BASE_PATH,
    MESH_FILE_PATTERN,
    VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=False
)

velocity_field = velocity_sequence[0]

# Compute element sizes
print("\n[2/4] Analyzing mesh refinement...")
element_sizes = np.zeros(len(connectivity))
element_centroids = np.zeros((len(connectivity), 3))

for i, elem_nodes_idx in enumerate(connectivity):
    elem_nodes = node_positions[elem_nodes_idx]
    centroid = elem_nodes.mean(axis=0)
    element_centroids[i] = centroid

    # Compute characteristic element size (max edge length)
    max_edge = 0.0
    for j in range(4):
        for k in range(j+1, 4):
            edge_len = np.linalg.norm(elem_nodes[j] - elem_nodes[k])
            max_edge = max(max_edge, edge_len)
    element_sizes[i] = max_edge

print(f"\nElement size distribution:")
print(f"  Min: {element_sizes.min():.6e} m = {element_sizes.min()*1000:.6f} mm")
print(f"  Mean: {element_sizes.mean():.6e} m = {element_sizes.mean()*1000:.6f} mm")
print(f"  Max: {element_sizes.max():.6e} m = {element_sizes.max()*1000:.6f} mm")
print(f"  Std: {element_sizes.std():.6e} m")

# Identify refined regions (smallest 10% of elements)
size_threshold_fine = np.percentile(element_sizes, 10)
size_threshold_coarse = np.percentile(element_sizes, 90)

fine_elem_mask = element_sizes < size_threshold_fine
coarse_elem_mask = element_sizes > size_threshold_coarse

n_fine = fine_elem_mask.sum()
n_coarse = coarse_elem_mask.sum()

print(f"\nRefinement analysis:")
print(f"  Fine elements (< {size_threshold_fine*1000:.6f} mm): {n_fine:,} ({100*n_fine/len(connectivity):.1f}%)")
print(f"  Coarse elements (> {size_threshold_coarse*1000:.6f} mm): {n_coarse:,} ({100*n_coarse/len(connectivity):.1f}%)")

# Compute velocity statistics for fine vs coarse elements
fine_elem_indices = np.where(fine_elem_mask)[0]
coarse_elem_indices = np.where(coarse_elem_mask)[0]

def compute_element_velocity_stats(elem_indices, velocity_field, connectivity, label):
    """Compute mean velocity magnitude in given elements."""
    vel_mags = []
    for elem_idx in elem_indices:
        elem_nodes_idx = connectivity[elem_idx]
        elem_vels = velocity_field[elem_nodes_idx]
        elem_vel_mag = np.linalg.norm(elem_vels, axis=1).mean()
        vel_mags.append(elem_vel_mag)
    vel_mags = np.array(vel_mags)

    print(f"\n{label} element velocity statistics:")
    print(f"  Mean |vel|: {vel_mags.mean():.6e} m/s")
    print(f"  Max |vel|: {vel_mags.max():.6e} m/s")
    print(f"  Min |vel|: {vel_mags.min():.6e} m/s")
    print(f"  Std |vel|: {vel_mags.std():.6e} m/s")

    return vel_mags

# Sample subset for performance
n_sample = min(1000, len(fine_elem_indices))
fine_sample = np.random.choice(fine_elem_indices, n_sample, replace=False)
coarse_sample = np.random.choice(coarse_elem_indices, n_sample, replace=False)

fine_vels = compute_element_velocity_stats(fine_sample, velocity_field, connectivity, "Fine (refined)")
coarse_vels = compute_element_velocity_stats(coarse_sample, velocity_field, connectivity, "Coarse")

# Set up search structures
print("\n[3/4] Setting up search structures...")
mesh_gpu_connectivity = jax.device_put(connectivity)
mesh_gpu_node_positions = jax.device_put(node_positions)

element_neighbors = build_element_neighbors_array(connectivity)
mesh_gpu_element_neighbors = jax.device_put(element_neighbors)

morton_cpu = build_global_morton_octree(node_positions, connectivity, leaf_capacity=256)
mesh_gpu_global_morton = upload_global_morton_to_gpu(morton_cpu, connectivity, node_positions)

# Test particle assignment accuracy
print("\n[4/4] Testing particle assignment accuracy...")

def test_assignment_accuracy(test_elem_indices, label):
    """
    For each test element:
    1. Place particle at element centroid
    2. Search for containing element
    3. Check if found element matches ground truth
    """
    n_test = len(test_elem_indices)
    test_positions = element_centroids[test_elem_indices].astype(np.float32)

    # Upload to GPU
    test_positions_gpu = jax.device_put(test_positions)

    # Search for elements
    found_elem_ids = initial_assignment_extended_batch(
        test_positions_gpu,
        mesh_gpu_global_morton,
        max_radius=50
    )

    found_elem_ids_cpu = np.array(found_elem_ids)

    # Check accuracy
    correct = (found_elem_ids_cpu == test_elem_indices)
    accuracy = correct.sum() / n_test * 100

    print(f"\n{label}:")
    print(f"  Tested {n_test} particles at element centroids")
    print(f"  Correct assignments: {correct.sum()}/{n_test} ({accuracy:.2f}%)")
    print(f"  Incorrect assignments: {(~correct).sum()}/{n_test} ({100-accuracy:.2f}%)")

    # Analyze incorrect assignments
    if (~correct).any():
        incorrect_indices = np.where(~correct)[0]
        print(f"\n  Analyzing incorrect assignments:")

        # Sample up to 10 incorrect cases
        n_show = min(10, len(incorrect_indices))
        for i in range(n_show):
            idx = incorrect_indices[i]
            true_elem = test_elem_indices[idx]
            found_elem = found_elem_ids_cpu[idx]

            true_size = element_sizes[true_elem]
            found_size = element_sizes[found_elem] if found_elem >= 0 else -1

            true_vel = np.linalg.norm(velocity_field[connectivity[true_elem]], axis=1).mean()
            found_vel = np.linalg.norm(velocity_field[connectivity[found_elem]], axis=1).mean() if found_elem >= 0 else 0.0

            print(f"    Case {i+1}: True elem {true_elem} (size={true_size*1000:.4f}mm, vel={true_vel:.6e}) "
                  f"-> Found elem {found_elem} (size={found_size*1000:.4f}mm, vel={found_vel:.6e})")

    return accuracy, found_elem_ids_cpu

# Test fine elements (most critical)
fine_accuracy, fine_found = test_assignment_accuracy(
    fine_sample[:100],  # Test 100 fine elements
    "FINE ELEMENTS (refined regions)"
)

# Test coarse elements
coarse_accuracy, coarse_found = test_assignment_accuracy(
    coarse_sample[:100],  # Test 100 coarse elements
    "COARSE ELEMENTS"
)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if fine_accuracy < 95.0:
    print(f"\n❌ CRITICAL: Fine element assignment accuracy is LOW ({fine_accuracy:.1f}%)")
    print(f"   Particles in refined regions are being assigned to WRONG elements!")
    print(f"   This explains why fine element velocities are not being used.")
    print(f"\n   Possible causes:")
    print(f"   1. Morton octree search failing in refined regions")
    print(f"   2. L2 search radius too small for small elements")
    print(f"   3. Point-in-tet test failing due to numerical precision")
    print(f"   4. Element neighbor search not covering refined regions")
elif fine_accuracy < 99.0:
    print(f"\n⚠️  Fine element assignment accuracy is MODERATE ({fine_accuracy:.1f}%)")
    print(f"   Some particles in refined regions may be assigned to wrong elements.")
else:
    print(f"\n✅ Fine element assignment accuracy is GOOD ({fine_accuracy:.1f}%)")
    print(f"   Search is working correctly in refined regions.")

if abs(fine_accuracy - coarse_accuracy) > 5.0:
    print(f"\n⚠️  Accuracy differs significantly between fine and coarse regions:")
    print(f"   Fine: {fine_accuracy:.1f}%, Coarse: {coarse_accuracy:.1f}%")
    print(f"   This indicates mesh-resolution-dependent search issues.")

print("="*80)
