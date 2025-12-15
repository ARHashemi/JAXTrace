#!/usr/bin/env python3
"""
Debug test for point-in-tet accuracy issues.

This test checks if point-in-tet works correctly by:
1. Taking element centroids (exact, no perturbation)
2. Checking if centroid is inside its own element
3. Should be 100% true
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path

# Load mesh
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.search.octree_search_gpu import point_in_tet_jax

print("=" * 80)
print("POINT-IN-TET DEBUG TEST")
print("=" * 80)
print()

# Load mesh
mesh_path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_120.pvtu")
print(f"Loading mesh: {mesh_path.name}")
node_positions, connectivity, velocity_field = load_mesh_from_pvtu(mesh_path, field_name='Displacement')
print(f"✓ Loaded: {len(node_positions):,} nodes, {len(connectivity):,} elements")
print()

# Compute element centroids
print("Computing element centroids...")
element_centroids = np.array([
    node_positions[connectivity[i]].mean(axis=0)
    for i in range(min(1000, len(connectivity)))  # Test first 1000 elements
], dtype=np.float32)
print(f"✓ Computed {len(element_centroids):,} centroids")
print()

# Test 1: Centroid should be inside its own element
print("=" * 80)
print("TEST 1: Centroid Inside Own Element (No Perturbation)")
print("=" * 80)
print()

n_correct = 0
n_tested = len(element_centroids)

for i in range(n_tested):
    elem_id = i
    centroid = element_centroids[i]

    # Get element nodes
    node_ids = connectivity[elem_id]
    tet_nodes = node_positions[node_ids]

    # Check shape
    if i == 0:
        print(f"First element check:")
        print(f"  Centroid shape: {centroid.shape}")
        print(f"  node_ids: {node_ids}")
        print(f"  node_ids shape: {node_ids.shape}")
        print(f"  tet_nodes shape: {tet_nodes.shape}")
        print()

    # Convert to JAX
    centroid_jax = jnp.array(centroid)
    tet_nodes_jax = jnp.array(tet_nodes)

    # Check if inside
    inside = point_in_tet_jax(centroid_jax, tet_nodes_jax)

    if inside:
        n_correct += 1

    # Print first few results
    if i < 5:
        print(f"Element {elem_id}: inside = {inside}")

print()
print(f"RESULTS:")
print(f"  Tested: {n_tested} centroids")
print(f"  Inside own element: {n_correct}/{n_tested} ({100*n_correct/n_tested:.1f}%)")
print()

if n_correct < n_tested * 0.99:
    print("⚠️  FAIL: Centroids should be 100% inside their own elements!")
    print("     Point-in-tet implementation is BROKEN.")
else:
    print("✓ PASS: Point-in-tet works correctly for centroids")

print()

# Test 2: Test with small perturbation (1% of element size)
print("=" * 80)
print("TEST 2: Centroid + Small Perturbation (1% of min edge)")
print("=" * 80)
print()

# Compute minimum edge length
min_edge_length = float('inf')
for i in range(min(100, len(connectivity))):
    nodes = node_positions[connectivity[i]]
    edges = [
        np.linalg.norm(nodes[1] - nodes[0]),
        np.linalg.norm(nodes[2] - nodes[0]),
        np.linalg.norm(nodes[3] - nodes[0]),
        np.linalg.norm(nodes[2] - nodes[1]),
        np.linalg.norm(nodes[3] - nodes[1]),
        np.linalg.norm(nodes[3] - nodes[2])
    ]
    min_edge_length = min(min_edge_length, min(edges))

perturbation_scale = 0.01 * min_edge_length
print(f"Minimum edge length: {min_edge_length:.6e}")
print(f"Perturbation scale: {perturbation_scale:.6e} (1%)")
print()

n_correct_perturbed = 0
np.random.seed(42)

for i in range(min(100, n_tested)):  # Test first 100
    elem_id = i
    centroid = element_centroids[i]

    # Add small perturbation
    perturbation = np.random.uniform(-perturbation_scale, perturbation_scale, size=3)
    perturbed_pos = centroid + perturbation

    # Get element nodes
    node_ids = connectivity[elem_id]
    tet_nodes = node_positions[node_ids]

    # Convert to JAX
    perturbed_pos_jax = jnp.array(perturbed_pos)
    tet_nodes_jax = jnp.array(tet_nodes)

    # Check if inside
    inside = point_in_tet_jax(perturbed_pos_jax, tet_nodes_jax)

    if inside:
        n_correct_perturbed += 1

    # Print first few
    if i < 5:
        print(f"Element {elem_id}: inside = {inside} (perturbation = {np.linalg.norm(perturbation):.6e})")

print()
print(f"RESULTS:")
print(f"  Tested: 100 perturbed centroids")
print(f"  Still inside own element: {n_correct_perturbed}/100 ({n_correct_perturbed}%)")
print()

if n_correct_perturbed < 80:
    print("⚠️  WARNING: Many particles left their element with 1% perturbation")
    print("     This suggests elements are very small or perturbation is too large")
elif n_correct_perturbed < 100:
    print("⚠️  EXPECTED: Some particles may have moved to neighboring elements")
    print(f"     {100 - n_correct_perturbed}% moved out (normal for boundary cases)")
else:
    print("✓ EXCELLENT: All perturbed particles stayed inside")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"1. Exact centroids: {n_correct}/{n_tested} inside ({100*n_correct/n_tested:.1f}%)")
print(f"2. Perturbed (1%): {n_correct_perturbed}/100 inside ({n_correct_perturbed}%)")
print()

if n_correct == n_tested:
    print("✓ Point-in-tet algorithm is CORRECT")
    print("  Issue must be elsewhere (octree traversal, element ID mapping, etc.)")
else:
    print("✗ Point-in-tet algorithm is BROKEN")
    print("  This explains the 99.97% inaccuracy in octree search!")
print()
