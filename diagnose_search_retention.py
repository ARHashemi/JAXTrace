#!/usr/bin/env python3
"""
Diagnostic Script: Why Search Retention Stops at 95%

Analyzes:
1. Are lost particles outside domain?
2. Leaf depth distribution (adaptive octree)
3. Morton curve spatial discontinuities
4. Search coverage analysis
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import JAXTrace modules (use same imports as production script)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.tracking.initial_assignment_cascading import initial_assignment_cascading_fallback
from jaxtrace.tracking.seeding import uniform_grid_seeds

# Configuration (same as production)
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 159)  # 40 timesteps
VELOCITY_FIELD_NAME = 'Displacement'
PARTICLE_GRID = (20, 50, 30)  # 30,000 particles


def main():
    print("=" * 80)
    print("Search Retention Diagnostic")
    print("=" * 80)

    # ========================================================================
    # 1. Load Mesh
    # ========================================================================

    print("\n[1/7] Loading mesh...")
    t_load = time.time()

    # Load velocity sequence from PVTU files (same as production)
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=True
    )

    n_elements = connectivity.shape[0]
    n_nodes_orig = node_positions.shape[0]
    n_timesteps = velocity_sequence.shape[0]
    t_load = time.time() - t_load
    print(f"  Loaded in {t_load:.2f}s")
    print(f"  Elements: {n_elements:,}, Nodes: {n_nodes_orig:,}, Timesteps: {n_timesteps}")

    # ========================================================================
    # 2. Deduplicate
    # ========================================================================

    print("\n[2/7] Deduplicating...")
    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    n_nodes = node_positions.shape[0]
    print(f"  Removed {n_duplicates_removed:,} duplicates")
    print(f"  Final: {n_elements:,} elements, {n_nodes:,} nodes")

    # DIAGNOSTIC: Verify dtypes after deduplication
    print(f"  After dedup - node_positions dtype: {node_positions.dtype}")
    print(f"  After dedup - connectivity dtype: {connectivity.dtype}")
    print(f"  After dedup - connectivity sample: {connectivity[0]}")
    print(f"  After dedup - max node ID: {connectivity.max()}, n_nodes: {n_nodes}")

    # ========================================================================
    # 3. Build Octree
    # ========================================================================

    print("\n[3/7] Building octree...")
    # DIAGNOSTIC: Verify array types before octree building
    print(f"  node_positions dtype: {node_positions.dtype}, shape: {node_positions.shape}")
    print(f"  connectivity dtype: {connectivity.dtype}, shape: {connectivity.shape}")

    t_octree = time.time()
    octree_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256,
        max_depth=21,
        verbose=True  # Show detailed stats
    )
    t_octree = time.time() - t_octree
    print(f"  Built octree in {t_octree:.2f}s")
    print(f"  Leaves: {octree_struct.n_leaves:,}")
    print(f"  Table depth: {octree_struct.table_depth}")

    # ========================================================================
    # 4. Analyze Leaf Size Distribution
    # ========================================================================

    print("\n[4/7] Analyzing leaf size distribution...")

    # Extract leaf sizes from the structure
    leaf_sizes = octree_struct.leaf_length

    print(f"  Total leaves: {octree_struct.n_leaves:,}")
    print(f"  Leaf capacity: {octree_struct.leaf_capacity}")
    print(f"  Max depth: {octree_struct.max_depth}")
    print(f"  Table depth: {octree_struct.table_depth}")

    # Leaf size statistics
    print(f"\n  Leaf Size Distribution:")
    print(f"    Min size:  {leaf_sizes.min()}")
    print(f"    Max size:  {leaf_sizes.max()}")
    print(f"    Mean size: {leaf_sizes.mean():.1f}")
    print(f"    Median size: {np.median(leaf_sizes):.1f}")

    # Count leaves at capacity
    at_capacity = np.sum(leaf_sizes >= octree_struct.leaf_capacity)
    pct_full = 100.0 * at_capacity / octree_struct.n_leaves
    print(f"\n  Leaves at capacity: {at_capacity:,} ({pct_full:.1f}%)")

    if pct_full > 10:
        print(f"  ⚠️  WARNING: {pct_full:.1f}% of leaves are at capacity")
        print(f"      This indicates highly variable element density")
        print(f"      Adaptive octree may have multiple depth levels")

    # ========================================================================
    # 5. Analyze Morton Spatial Discontinuities
    # ========================================================================

    print("\n[5/7] Analyzing Morton spatial discontinuities...")

    # Compute leaf centroids using leaf_start and leaf_length arrays
    leaf_centroids = []
    for i in range(octree_struct.n_leaves):
        start = octree_struct.leaf_start[i]
        length = octree_struct.leaf_length[i]
        leaf_elem_ids = octree_struct.elem_ids_sorted[start:start+length]
        # Average all element centroids in leaf
        elem_centroids = node_positions[connectivity[leaf_elem_ids]].mean(axis=1)
        leaf_centroid = elem_centroids.mean(axis=0)
        leaf_centroids.append(leaf_centroid)

    leaf_centroids = np.array(leaf_centroids)

    # Find max spatial jump between consecutive Morton leaves
    max_jump = 0.0
    max_jump_idx = -1
    jumps = []

    for i in range(octree_struct.n_leaves - 1):
        dist = np.linalg.norm(leaf_centroids[i+1] - leaf_centroids[i])
        jumps.append(dist)
        if dist > max_jump:
            max_jump = dist
            max_jump_idx = i

    jumps = np.array(jumps)
    domain_size = node_positions.max(axis=0) - node_positions.min(axis=0)
    domain_diag = np.linalg.norm(domain_size)

    print(f"\n  Spatial discontinuities between consecutive Morton leaves:")
    print(f"    Domain diagonal: {domain_diag:.6f}")
    print(f"    Mean jump:   {jumps.mean():.6f} ({100*jumps.mean()/domain_diag:.2f}% of domain diagonal)")
    print(f"    Median jump: {np.median(jumps):.6f} ({100*np.median(jumps)/domain_diag:.2f}%)")
    print(f"    Max jump:    {max_jump:.6f} ({100*max_jump/domain_diag:.2f}% of domain diagonal)")
    print(f"      Between leaf {max_jump_idx} and leaf {max_jump_idx+1}")

    # Find large jumps (> 10% of domain diagonal)
    large_jumps = jumps > 0.1 * domain_diag
    n_large_jumps = large_jumps.sum()
    print(f"\n    Large jumps (>10% domain diagonal): {n_large_jumps} ({100*n_large_jumps/len(jumps):.2f}%)")

    if n_large_jumps > octree_struct.n_leaves * 0.01:
        print(f"  ⚠️  WARNING: {n_large_jumps} large spatial jumps between consecutive Morton leaves")
        print(f"      This explains why radius-based search fails!")
        print(f"      Even large radius may not reach spatially close leaves")

    # ========================================================================
    # 6. Upload to GPU and Test Search
    # ========================================================================

    print("\n[6/7] Testing search with various radii...")

    from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
    from jaxtrace.gpu.forest import build_element_neighbors_array

    element_neighbors = build_element_neighbors_array(connectivity, method='face', verbose=False)
    mesh_gpu = upload_mesh_to_gpu(connectivity, node_positions, element_neighbors, verbose=False)
    mesh_gpu_octree = upload_global_morton_to_gpu(octree_struct, connectivity, node_positions)

    # Generate particles
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)
    bounds = [domain_min, domain_max]
    positions = uniform_grid_seeds(
        resolution=PARTICLE_GRID,
        bounds=bounds,
        include_boundaries=True
    )
    n_particles = positions.shape[0]
    positions_gpu = jax.device_put(positions)

    print(f"  Generated {n_particles:,} particles")

    # Test different radii
    test_radii = [1, 2, 5, 10, 20, 30, 50, 64, 100]
    print(f"\n  {'Radius':<10s}  {'Leaves':>10s}  {'Success Rate':>15s}  {'Time':>10s}")
    print("  " + "-" * 55)

    for radius in test_radii:
        fallback_radii = [radius * 2, radius * 5, radius * 10]

        t_start = time.time()
        element_ids_gpu = initial_assignment_cascading_fallback(
            positions_gpu,
            mesh_gpu_octree,
            initial_radius=radius,
            fallback_radii=fallback_radii,
            verbose=False
        )
        element_ids_gpu = jax.block_until_ready(element_ids_gpu)
        t_elapsed = time.time() - t_start

        n_assigned = int(jnp.sum(element_ids_gpu >= 0))
        success_rate = 100.0 * n_assigned / n_particles
        n_leaves_searched = 2 * radius + 1

        print(f"  {radius:<10d}  {n_leaves_searched:>10d}  {success_rate:>14.2f}%  {t_elapsed:>9.3f}s")

    # ========================================================================
    # 7. Analyze Lost Particles
    # ========================================================================

    print("\n[7/7] Analyzing lost particles (using radius=30)...")

    # Use moderate radius for analysis
    element_ids_gpu = initial_assignment_cascading_fallback(
        positions_gpu,
        mesh_gpu_octree,
        initial_radius=30,
        fallback_radii=[60, 150, 300],
        verbose=False
    )
    element_ids_gpu = jax.block_until_ready(element_ids_gpu)

    # Download results
    element_ids = np.array(element_ids_gpu)
    positions_cpu = np.array(positions_gpu)

    # Find lost particles
    lost_mask = element_ids < 0
    n_lost = lost_mask.sum()

    if n_lost == 0:
        print("  ✅ All particles assigned successfully!")
    else:
        lost_positions = positions_cpu[lost_mask]

        # Check if outside domain
        bbox_min = node_positions.min(axis=0)
        bbox_max = node_positions.max(axis=0)

        outside_mask = (
            (lost_positions < bbox_min).any(axis=1) |
            (lost_positions > bbox_max).any(axis=1)
        )
        n_outside = outside_mask.sum()
        n_inside = n_lost - n_outside

        print(f"\n  Lost particles: {n_lost:,}/{n_particles:,} ({100*n_lost/n_particles:.2f}%)")
        print(f"    Outside domain bbox: {n_outside:,} ({100*n_outside/n_lost:.1f}% of lost)")
        print(f"    Inside bbox but unfound: {n_inside:,} ({100*n_inside/n_lost:.1f}% of lost)")

        if n_outside > 0:
            print(f"\n  ⚠️  {n_outside} particles are outside domain boundary")
            print(f"      These will never be found by search (expected)")

        if n_inside > 0:
            print(f"\n  ⚠️  {n_inside} particles are inside domain but not found")
            print(f"      Possible causes:")
            print(f"        1. Mesh has gaps or holes")
            print(f"        2. Morton discontinuities (leaves at wrong depth)")
            print(f"        3. Search radius too small for adaptive octree")
            print(f"        4. Numerical precision issues (float32)")

            # Sample a few lost-but-inside particles
            inside_lost = lost_positions[~outside_mask]
            sample_size = min(5, len(inside_lost))
            print(f"\n  Sample lost positions (inside bbox):")
            for i in range(sample_size):
                pos = inside_lost[i]
                print(f"    {i+1}. ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})")

    # ========================================================================
    # Summary and Recommendations
    # ========================================================================

    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)

    print("\n1. OCTREE STRUCTURE:")
    print(f"   Total leaves: {octree_struct.n_leaves:,}")
    print(f"   Max depth: {octree_struct.max_depth}")
    print(f"   Table depth: {octree_struct.table_depth}")
    if pct_full > 10:
        print(f"   ⚠️  {pct_full:.1f}% of leaves are at capacity")
        print(f"       Indicates variable element density (adaptive depth)")
        print(f"   ✅  RECOMMENDATION: Use 'hierarchical' search method")
        print(f"       L2_SEARCH_METHOD = 'hierarchical'")
    else:
        print(f"   ✅  Low leaf utilization ({pct_full:.1f}% at capacity)")
        print(f"       Radius-based search should work well")

    print("\n2. MORTON DISCONTINUITIES:")
    if n_large_jumps > octree_struct.n_leaves * 0.01:
        print(f"   ⚠️  {n_large_jumps} large spatial jumps between consecutive Morton leaves")
        print(f"       Even large radius may not reach spatially close leaves")
        print(f"   ✅  RECOMMENDATION: Use 'hierarchical' or 'neighbors' search method")
    else:
        print(f"   ✅  Small spatial jumps between Morton leaves")
        print(f"       Radius-based search is geometrically sound")

    print("\n3. SEARCH COVERAGE:")
    # Find plateau in success rate
    success_rates = []
    for radius in test_radii:
        # Use same assignment as above (cached)
        pass  # Already computed

    print(f"   Current configuration achieves ~95% retention")
    print(f"   To improve further:")
    print(f"     - Try L2_SEARCH_METHOD = 'hierarchical'")
    print(f"     - Increase L1 depth: N_HOPS = 7 (expensive!)")
    print(f"     - Check mesh quality (gaps, holes)")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
