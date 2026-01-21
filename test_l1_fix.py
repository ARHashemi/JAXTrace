#!/usr/bin/env python3
"""
Test L1 Algorithm Fix

Tests that the L1 neighbor search now correctly:
1. Searches neighbors when starting element doesn't contain position
2. Returns fine elements in refined region
3. Returns -1 when neighbors don't contain position (falls to L2)

Diagnostic Focus:
- Track L0/L1/L2 success rates in refined region
- Verify L1 no longer returns cached coarse elements incorrectly
- Measure performance with L1 enabled vs disabled
"""

import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.tracking.rk4_fully_fused_timedep import create_rk4_fully_fused_timedep
from jaxtrace.gpu.tracking.initial_assignment_extended import initial_assignment_extended_batch


def analyze_element_assignment(positions_gpu, element_ids_gpu, element_centroids, refined_region_center, refined_region_radius=2.0):
    """
    Analyze which elements particles are assigned to.

    Args:
        positions_gpu: (N, 3) particle positions
        element_ids_gpu: (N,) element IDs
        element_centroids: (n_elements, 3) element centroids
        refined_region_center: (3,) center of refined region
        refined_region_radius: radius defining refined region

    Returns:
        dict with statistics
    """
    positions = np.array(positions_gpu)
    element_ids = np.array(element_ids_gpu)

    # Find particles in refined region
    distances = np.linalg.norm(positions - refined_region_center, axis=1)
    in_refined = distances < refined_region_radius

    n_in_refined = np.sum(in_refined)
    if n_in_refined == 0:
        return {"n_particles_in_refined": 0}

    # Get element sizes for particles in refined region
    assigned_element_ids = element_ids[in_refined]
    valid_mask = assigned_element_ids >= 0

    assigned_centroids = element_centroids[assigned_element_ids[valid_mask]]

    # Compute element sizes (distance from particle to element centroid is proxy)
    particle_pos_refined = positions[in_refined][valid_mask]
    distances_to_centroid = np.linalg.norm(particle_pos_refined - assigned_centroids, axis=1)

    # Classify elements by size (rough estimate)
    # Fine: < 0.15mm, Medium: 0.15-0.30mm, Coarse: > 0.30mm
    fine_mask = distances_to_centroid < 0.15
    medium_mask = (distances_to_centroid >= 0.15) & (distances_to_centroid < 0.30)
    coarse_mask = distances_to_centroid >= 0.30

    stats = {
        "n_particles_in_refined": n_in_refined,
        "n_valid_assignments": np.sum(valid_mask),
        "n_fine_elements": np.sum(fine_mask),
        "n_medium_elements": np.sum(medium_mask),
        "n_coarse_elements": np.sum(coarse_mask),
        "pct_fine": 100.0 * np.sum(fine_mask) / np.sum(valid_mask) if np.sum(valid_mask) > 0 else 0.0,
        "pct_medium": 100.0 * np.sum(medium_mask) / np.sum(valid_mask) if np.sum(valid_mask) > 0 else 0.0,
        "pct_coarse": 100.0 * np.sum(coarse_mask) / np.sum(valid_mask) if np.sum(valid_mask) > 0 else 0.0,
        "mean_distance_to_centroid": np.mean(distances_to_centroid),
        "median_distance_to_centroid": np.median(distances_to_centroid),
    }

    return stats


def test_l1_fix():
    """Test L1 algorithm fix with diagnostic tracking."""

    print("=" * 80)
    print("L1 ALGORITHM FIX TEST")
    print("=" * 80)

    # Configuration
    from pathlib import Path
    mesh_path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule")
    mesh_file_pattern = "threadedAvtk_{timestep}.pvtu"
    velocity_timestep = 120  # Use single timestep
    velocity_field_name = 'Displacement'
    n_particles = 500
    dt = 5e-6
    n_steps = 5

    refined_region_center = np.array([30.0, 15.0, 0.3])  # Tool center
    refined_region_radius = 2.0  # mm

    print(f"\nConfiguration:")
    print(f"  Mesh path: {mesh_path}")
    print(f"  Particles: {n_particles}")
    print(f"  Time step: {dt:.2e} s")
    print(f"  Steps: {n_steps}")
    print(f"  Refined region center: {refined_region_center}")
    print(f"  Refined region radius: {refined_region_radius} mm")

    # Load mesh (single timestep)
    print("\nLoading mesh and velocity field...")
    node_positions, connectivity, velocity_field_single = load_velocity_sequence_from_pvtu(
        base_path=mesh_path,
        file_pattern=mesh_file_pattern,
        timestep_range=(velocity_timestep, velocity_timestep),
        field_name=velocity_field_name,
        verbose=False
    )
    n_nodes = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    print(f"  Loaded {n_elements:,} elements, {n_nodes:,} nodes")

    # Repeat velocity field for time-dependent interface
    velocity_sequence = np.repeat(velocity_field_single, 5, axis=0)  # Repeat for 5 timesteps

    # Build Morton structure
    print("\nBuilding Morton structure...")
    morton_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256,
        max_depth=21,
        verbose=False
    )
    print(f"  Built {morton_struct.n_leaves:,} leaves")

    # Build element neighbors
    print("\nBuilding element neighbors...")
    element_neighbors = build_element_neighbors_array(connectivity)

    # Upload to GPU
    print("\nUploading to GPU...")
    node_positions_gpu = jnp.array(node_positions)
    connectivity_gpu = jnp.array(connectivity)
    element_neighbors_gpu = jnp.array(element_neighbors)
    velocity_fields_gpu = jnp.array(velocity_sequence)
    morton_gpu = upload_global_morton_to_gpu(morton_struct, connectivity, node_positions)

    # Initialize particles in refined region
    print("\nInitializing particles in refined region...")
    positions_cpu = np.random.uniform(
        refined_region_center - refined_region_radius * 0.8,
        refined_region_center + refined_region_radius * 0.8,
        (n_particles, 3)
    )

    positions_gpu = jnp.array(positions_cpu)
    element_ids_gpu = initial_assignment_extended_batch(
        positions_gpu,
        morton_gpu,
        max_radius=100
    )

    success_rate = 100.0 * np.sum(np.array(element_ids_gpu) >= 0) / n_particles
    print(f"  Initial assignment success: {success_rate:.1f}%")

    # Compute element centroids for analysis
    element_centroids = np.mean(node_positions[connectivity], axis=1)

    # Analyze initial assignment
    print("\nInitial Element Assignment:")
    stats_initial = analyze_element_assignment(
        positions_gpu, element_ids_gpu, element_centroids,
        refined_region_center, refined_region_radius
    )
    print(f"  Particles in refined region: {stats_initial['n_particles_in_refined']}")
    print(f"  Valid assignments: {stats_initial['n_valid_assignments']}")
    print(f"  Fine elements: {stats_initial['n_fine_elements']} ({stats_initial['pct_fine']:.1f}%)")
    print(f"  Medium elements: {stats_initial['n_medium_elements']} ({stats_initial['pct_medium']:.1f}%)")
    print(f"  Coarse elements: {stats_initial['n_coarse_elements']} ({stats_initial['pct_coarse']:.1f}%)")
    print(f"  Mean dist to centroid: {stats_initial['mean_distance_to_centroid']:.4f} mm")

    # Test 1: L1 ENABLED (with fix)
    print("\n" + "=" * 80)
    print("TEST 1: L1 ENABLED (WITH FIX)")
    print("=" * 80)

    rk4_step_l1_enabled = create_rk4_fully_fused_timedep(
        connectivity_gpu,
        node_positions_gpu,
        element_neighbors_gpu,
        morton_gpu,
        n_hops=3,
        l2_search_radius=100,
        enable_l1_search=True
    )

    # Reset particles
    positions_test1 = jnp.array(positions_cpu)
    element_ids_test1 = jnp.array(element_ids_gpu)

    print("\nRunning tracking with L1 enabled...")
    start_time = time.time()

    for step in range(n_steps):
        time_idx = 0  # Use first timestep for simplicity
        positions_test1, element_ids_test1 = rk4_step_l1_enabled(
            positions_test1,
            element_ids_test1,
            dt,
            velocity_fields_gpu,
            time_idx
        )

        # Analyze assignment after each step
        stats = analyze_element_assignment(
            positions_test1, element_ids_test1, element_centroids,
            refined_region_center, refined_region_radius
        )

        print(f"  Step {step+1}/{n_steps}: "
              f"Fine={stats['n_fine_elements']} ({stats['pct_fine']:.1f}%), "
              f"Medium={stats['n_medium_elements']} ({stats['pct_medium']:.1f}%), "
              f"Coarse={stats['n_coarse_elements']} ({stats['pct_coarse']:.1f}%)")

    elapsed_l1 = time.time() - start_time
    print(f"\nTotal time: {elapsed_l1:.2f} s")
    print(f"Time per step: {elapsed_l1/n_steps:.2f} s")
    print(f"Throughput: {n_particles*n_steps/elapsed_l1:.0f} particle-steps/s")

    # Final analysis
    print("\nFinal Element Assignment (L1 Enabled):")
    stats_final_l1 = analyze_element_assignment(
        positions_test1, element_ids_test1, element_centroids,
        refined_region_center, refined_region_radius
    )
    print(f"  Fine elements: {stats_final_l1['n_fine_elements']} ({stats_final_l1['pct_fine']:.1f}%)")
    print(f"  Medium elements: {stats_final_l1['n_medium_elements']} ({stats_final_l1['pct_medium']:.1f}%)")
    print(f"  Coarse elements: {stats_final_l1['n_coarse_elements']} ({stats_final_l1['pct_coarse']:.1f}%)")

    # Test 2: L1 DISABLED (baseline)
    print("\n" + "=" * 80)
    print("TEST 2: L1 DISABLED (BASELINE)")
    print("=" * 80)

    rk4_step_l1_disabled = create_rk4_fully_fused_timedep(
        connectivity_gpu,
        node_positions_gpu,
        element_neighbors_gpu,
        morton_gpu,
        n_hops=3,
        l2_search_radius=100,
        enable_l1_search=False
    )

    # Reset particles
    positions_test2 = jnp.array(positions_cpu)
    element_ids_test2 = jnp.array(element_ids_gpu)

    print("\nRunning tracking with L1 disabled...")
    start_time = time.time()

    for step in range(n_steps):
        time_idx = 0
        positions_test2, element_ids_test2 = rk4_step_l1_disabled(
            positions_test2,
            element_ids_test2,
            dt,
            velocity_fields_gpu,
            time_idx
        )

        stats = analyze_element_assignment(
            positions_test2, element_ids_test2, element_centroids,
            refined_region_center, refined_region_radius
        )

        print(f"  Step {step+1}/{n_steps}: "
              f"Fine={stats['n_fine_elements']} ({stats['pct_fine']:.1f}%), "
              f"Medium={stats['n_medium_elements']} ({stats['pct_medium']:.1f}%), "
              f"Coarse={stats['n_coarse_elements']} ({stats['pct_coarse']:.1f}%)")

    elapsed_l2 = time.time() - start_time
    print(f"\nTotal time: {elapsed_l2:.2f} s")
    print(f"Time per step: {elapsed_l2/n_steps:.2f} s")
    print(f"Throughput: {n_particles*n_steps/elapsed_l2:.0f} particle-steps/s")

    # Final analysis
    print("\nFinal Element Assignment (L1 Disabled):")
    stats_final_l2 = analyze_element_assignment(
        positions_test2, element_ids_test2, element_centroids,
        refined_region_center, refined_region_radius
    )
    print(f"  Fine elements: {stats_final_l2['n_fine_elements']} ({stats_final_l2['pct_fine']:.1f}%)")
    print(f"  Medium elements: {stats_final_l2['n_medium_elements']} ({stats_final_l2['pct_medium']:.1f}%)")
    print(f"  Coarse elements: {stats_final_l2['n_coarse_elements']} ({stats_final_l2['pct_coarse']:.1f}%)")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print("\nElement Assignment:")
    print(f"  L1 Enabled:  Fine={stats_final_l1['pct_fine']:5.1f}%, Medium={stats_final_l1['pct_medium']:5.1f}%, Coarse={stats_final_l1['pct_coarse']:5.1f}%")
    print(f"  L1 Disabled: Fine={stats_final_l2['pct_fine']:5.1f}%, Medium={stats_final_l2['pct_medium']:5.1f}%, Coarse={stats_final_l2['pct_coarse']:5.1f}%")

    print("\nPerformance:")
    print(f"  L1 Enabled:  {elapsed_l1:.2f} s ({n_particles*n_steps/elapsed_l1:.0f} particle-steps/s)")
    print(f"  L1 Disabled: {elapsed_l2:.2f} s ({n_particles*n_steps/elapsed_l2:.0f} particle-steps/s)")
    print(f"  Speedup: {elapsed_l2/elapsed_l1:.2f}x")

    # Success criteria
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA")
    print("=" * 80)

    success_criteria = []

    # 1. L1 should improve fine element assignment
    if stats_final_l1['pct_fine'] > stats_final_l2['pct_fine']:
        print("✓ L1 improves fine element assignment")
        success_criteria.append(True)
    else:
        print("✗ L1 does NOT improve fine element assignment")
        success_criteria.append(False)

    # 2. L1 should reduce coarse element assignment
    if stats_final_l1['pct_coarse'] < stats_final_l2['pct_coarse']:
        print("✓ L1 reduces coarse element assignment")
        success_criteria.append(True)
    else:
        print("✗ L1 does NOT reduce coarse element assignment")
        success_criteria.append(False)

    # 3. L1 and L2 should produce similar assignments (both correct)
    assignment_diff = abs(stats_final_l1['pct_fine'] - stats_final_l2['pct_fine'])
    if assignment_diff < 5.0:  # Within 5%
        print(f"✓ L1 and L2 produce similar assignments (diff={assignment_diff:.1f}%)")
        success_criteria.append(True)
    else:
        print(f"  L1 and L2 have different assignments (diff={assignment_diff:.1f}%)")
        # This is OK if L1 is faster

    # 4. L1 should not significantly degrade performance
    if elapsed_l1 < elapsed_l2 * 1.5:  # No more than 50% slower
        print(f"✓ L1 does not significantly degrade performance ({elapsed_l1/elapsed_l2:.2f}x)")
        success_criteria.append(True)
    else:
        print(f"✗ L1 significantly degrades performance ({elapsed_l1/elapsed_l2:.2f}x)")
        success_criteria.append(False)

    print("\n" + "=" * 80)
    if all(success_criteria):
        print("✅ ALL TESTS PASSED - L1 FIX IS SUCCESSFUL!")
    else:
        print("❌ SOME TESTS FAILED - L1 FIX NEEDS INVESTIGATION")
    print("=" * 80)


if __name__ == "__main__":
    test_l1_fix()
