#!/usr/bin/env python3
"""
Diagnostic Script: Analyze Lost Particles

Identifies WHERE and WHY particles are lost during tracking.

For each lost particle:
1. Find nearest element (brute force CPU search)
2. Determine distance to nearest element
3. Check what leaf the query position maps to
4. Check what leaf the nearest element is in
5. Analyze WHY search failed (octant distance, depth difference, etc.)
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import JAXTrace modules
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import (
    upload_global_morton_to_gpu,
    morton_encode_position_jax
)
from jaxtrace.gpu.tracking.rk4_fully_fused_timedep import create_rk4_fully_fused_timedep
from jaxtrace.tracking.seeding import uniform_grid_seeds

# Configuration (same as production)
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 159)
VELOCITY_FIELD_NAME = 'Displacement'
PARTICLE_GRID = (20, 50, 30)  # 30,000 particles

# Tracking parameters
N_STEPS = 10  # First 10 steps
DT = 1e-3
ENABLE_L1_SEARCH = True
N_HOPS = 5
L2_SEARCH_METHOD = 'incremental'
INCREMENTAL_SEARCH_RADII = (2, 4, 8, 15, 30)

# Diagnostic parameters
MAX_LOST_TO_ANALYZE = 20  # Analyze up to 20 lost particles


def morton_encode_position_cpu(pos, bbox_min, bbox_max, max_depth):
    """CPU version of Morton encoding for diagnostics."""
    # Normalize to [0, 2^max_depth - 1]
    normalized = (pos - bbox_min) / (bbox_max - bbox_min)
    max_val = (1 << max_depth) - 1
    coords = (normalized * max_val).astype(np.uint32)
    coords = np.clip(coords, 0, max_val)

    x, y, z = coords

    # Interleave bits
    morton = np.uint64(0)
    for i in range(max_depth):
        bit_pos = max_depth - 1 - i
        x_bit = (x >> bit_pos) & 1
        y_bit = (y >> bit_pos) & 1
        z_bit = (z >> bit_pos) & 1

        octant = (x_bit << 0) | (y_bit << 1) | (z_bit << 2)
        morton |= np.uint64(octant) << (60 - i * 3)

    return morton


def find_leaf_for_morton_code(morton_code, octree_struct):
    """Find leaf ID containing this Morton code."""
    table_depth = octree_struct.table_depth
    shift_amount = 63 - (table_depth * 3)

    # Extract prefix
    prefix_idx = int(morton_code >> shift_amount)
    prefix_idx = min(prefix_idx, octree_struct.prefix_start.shape[0] - 1)

    first_leaf = octree_struct.prefix_start[prefix_idx]
    num_leaves = octree_struct.prefix_length[prefix_idx]

    if num_leaves == 0 or first_leaf < 0:
        return -1

    # Search for leaf containing this Morton code
    for leaf_offset in range(num_leaves):
        leaf_id = first_leaf + leaf_offset
        if leaf_id >= octree_struct.n_leaves:
            break

        # Get leaf Morton range
        leaf_start = octree_struct.leaf_start[leaf_id]
        leaf_length = octree_struct.leaf_length[leaf_id]

        if leaf_length == 0:
            continue

        # Check if morton_code is in this leaf's range
        first_elem_idx = leaf_start
        last_elem_idx = leaf_start + leaf_length - 1

        first_morton = octree_struct.morton_sorted[first_elem_idx]
        last_morton = octree_struct.morton_sorted[last_elem_idx]

        if first_morton <= morton_code <= last_morton:
            return leaf_id

    # Not found, return first leaf in prefix
    return first_leaf if num_leaves > 0 else -1


def decode_morton_prefix_cpu(morton_code, depth):
    """Decode Morton code to (x, y, z) octant coordinates."""
    x, y, z = 0, 0, 0

    for i in range(depth):
        bit_pos = 60 - i * 3
        octant = (morton_code >> bit_pos) & 0b111

        x_bit = (octant >> 0) & 1
        y_bit = (octant >> 1) & 1
        z_bit = (octant >> 2) & 1

        x = (x << 1) | x_bit
        y = (y << 1) | y_bit
        z = (z << 1) | z_bit

    return x, y, z


def main():
    print("=" * 80)
    print("Lost Particles Diagnostic")
    print("=" * 80)

    # ========================================================================
    # 1. Load Mesh
    # ========================================================================

    print("\n[1/6] Loading mesh...")
    t_load = time.time()

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

    print("\n[2/6] Deduplicating...")
    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    n_nodes = node_positions.shape[0]
    print(f"  Removed {n_duplicates_removed:,} duplicates")
    print(f"  Final: {n_elements:,} elements, {n_nodes:,} nodes")

    # ========================================================================
    # 3. Build Octree
    # ========================================================================

    print("\n[3/6] Building octree...")
    t_octree = time.time()
    octree_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256,
        max_depth=21,
        verbose=False
    )
    t_octree = time.time() - t_octree
    print(f"  Built octree in {t_octree:.2f}s")
    print(f"  Leaves: {octree_struct.n_leaves:,}")
    print(f"  Table depth: {octree_struct.table_depth}")

    # ========================================================================
    # 4. Upload to GPU and Track
    # ========================================================================

    print("\n[4/6] Tracking particles...")

    # Upload to GPU
    element_neighbors = build_element_neighbors_array(connectivity, method='face', verbose=False)
    mesh_gpu = upload_mesh_to_gpu(connectivity, node_positions, element_neighbors, verbose=False)
    mesh_gpu_octree = upload_global_morton_to_gpu(octree_struct, connectivity, node_positions)
    velocity_sequence_gpu = jax.device_put(velocity_sequence.astype(np.float32))

    # Compute element volumes
    v0 = node_positions[connectivity[:, 0]]
    v1 = node_positions[connectivity[:, 1]]
    v2 = node_positions[connectivity[:, 2]]
    v3 = node_positions[connectivity[:, 3]]
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0
    cross_e2_e3 = np.cross(e2, e3)
    det = np.sum(e1 * cross_e2_e3, axis=1)
    element_volumes = np.abs(det) / 6.0
    element_volumes_gpu = jax.device_put(element_volumes.astype(np.float32))

    # Generate particles (with clipping like production)
    bbox_min = node_positions.min(axis=0)
    bbox_max = node_positions.max(axis=0)
    margin = bbox_max - bbox_min
    par_bounds = [bbox_min + 0.01 * margin, bbox_max - 0.01 * margin]

    positions = uniform_grid_seeds(
        resolution=PARTICLE_GRID,
        bounds=par_bounds,
        include_boundaries=True
    )
    n_particles = positions.shape[0]
    positions_gpu = jax.device_put(positions.astype(np.float32))
    print(f"  Generated {n_particles:,} particles")

    # Initial assignment
    from jaxtrace.gpu.tracking.initial_assignment_cascading import initial_assignment_cascading_fallback
    element_ids_gpu = initial_assignment_cascading_fallback(
        positions_gpu,
        mesh_gpu_octree,
        initial_radius=2,
        fallback_radii=[4, 8, 15, 30],
        verbose=False
    )

    n_initial = int(jnp.sum(element_ids_gpu >= 0))
    print(f"  Initial assignment: {n_initial}/{n_particles} ({100*n_initial/n_particles:.2f}%)")

    # Create RK4 stepper
    rk4_step = create_rk4_fully_fused_timedep(
        mesh_gpu_connectivity=mesh_gpu.connectivity,
        mesh_gpu_node_positions=mesh_gpu.node_positions,
        mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
        mesh_gpu_element_volumes=element_volumes_gpu,
        mesh_gpu_global_morton=mesh_gpu_octree,
        n_hops=N_HOPS,
        l2_search_radius=None,
        enable_l1_search=ENABLE_L1_SEARCH,
        l2_search_method=L2_SEARCH_METHOD,
        l2_incremental_radii=INCREMENTAL_SEARCH_RADII
    )

    # Track for N_STEPS
    print(f"  Tracking for {N_STEPS} steps...")
    for step in range(N_STEPS):
        positions_gpu, element_ids_gpu = rk4_step(
            positions_gpu,
            element_ids_gpu,
            DT,
            velocity_sequence_gpu,
            step
        )
        positions_gpu = jax.block_until_ready(positions_gpu)
        element_ids_gpu = jax.block_until_ready(element_ids_gpu)

        n_active = int(jnp.sum(element_ids_gpu >= 0))
        retention = 100.0 * n_active / n_particles
        print(f"    Step {step+1:3d}: {n_active:6d} active ({retention:5.2f}%)")

    # ========================================================================
    # 5. Analyze Lost Particles
    # ========================================================================

    print("\n[5/6] Analyzing lost particles...")

    # Download results
    positions_final = np.array(positions_gpu)
    element_ids_final = np.array(element_ids_gpu)

    # Find lost particles
    lost_mask = element_ids_final < 0
    n_lost = lost_mask.sum()
    print(f"  Total lost particles: {n_lost}/{n_particles} ({100*n_lost/n_particles:.2f}%)")

    if n_lost == 0:
        print("  ✅ No lost particles to analyze!")
        return

    lost_positions = positions_final[lost_mask]
    n_to_analyze = min(MAX_LOST_TO_ANALYZE, n_lost)
    print(f"  Analyzing first {n_to_analyze} lost particles...")

    # Precompute element centroids (CPU)
    elem_centroids = node_positions[connectivity].mean(axis=1)

    # Analyze each lost particle
    results = []
    for i in range(n_to_analyze):
        pos = lost_positions[i]

        # Find nearest element (brute force)
        distances = np.linalg.norm(elem_centroids - pos, axis=1)
        nearest_elem = distances.argmin()
        dist_to_nearest = distances[nearest_elem]

        # Morton analysis
        morton_query = morton_encode_position_cpu(pos, bbox_min, bbox_max, 21)
        morton_nearest = morton_encode_position_cpu(elem_centroids[nearest_elem], bbox_min, bbox_max, 21)

        # Find leaves
        query_leaf = find_leaf_for_morton_code(morton_query, octree_struct)
        nearest_leaf = find_leaf_for_morton_code(morton_nearest, octree_struct)

        # Decode to octant coordinates at table_depth
        table_depth = octree_struct.table_depth
        query_octant = decode_morton_prefix_cpu(morton_query, table_depth)
        nearest_octant = decode_morton_prefix_cpu(morton_nearest, table_depth)

        # Octant distance (Manhattan distance in octant grid)
        # Cast to int to avoid uint64 overflow when subtracting
        octant_dist_manhattan = sum(abs(int(a) - int(b)) for a, b in zip(query_octant, nearest_octant))
        octant_dist_max = max(abs(int(a) - int(b)) for a, b in zip(query_octant, nearest_octant))

        results.append({
            'pos': pos,
            'nearest_elem': nearest_elem,
            'dist_to_nearest': dist_to_nearest,
            'query_leaf': query_leaf,
            'nearest_leaf': nearest_leaf,
            'leaf_distance': abs(query_leaf - nearest_leaf) if query_leaf >= 0 and nearest_leaf >= 0 else -1,
            'query_octant': query_octant,
            'nearest_octant': nearest_octant,
            'octant_dist_manhattan': octant_dist_manhattan,
            'octant_dist_max': octant_dist_max
        })

    # ========================================================================
    # 6. Summary and Diagnosis
    # ========================================================================

    print("\n[6/6] Diagnostic Summary")
    print("=" * 80)

    # Print individual results
    print(f"\nIndividual Lost Particle Analysis (first {n_to_analyze}):")
    print("-" * 80)

    for i, res in enumerate(results):
        print(f"\nParticle {i+1}:")
        print(f"  Position: ({res['pos'][0]:.6f}, {res['pos'][1]:.6f}, {res['pos'][2]:.6f})")
        print(f"  Nearest element: {res['nearest_elem']}, distance: {res['dist_to_nearest']:.6e}")
        print(f"  Query position → Leaf {res['query_leaf']}, Octant {res['query_octant']}")
        print(f"  Nearest element → Leaf {res['nearest_leaf']}, Octant {res['nearest_octant']}")
        print(f"  Leaf distance: {res['leaf_distance']} leaves apart in Morton order")
        print(f"  Octant distance: Manhattan={res['octant_dist_manhattan']}, Max={res['octant_dist_max']}")

        # Diagnosis
        if res['octant_dist_max'] <= 1:
            print(f"  ⚠️  DIAGNOSIS: Nearest element in 3×3×3 neighborhood - should have been found!")
        elif res['octant_dist_max'] == 2:
            print(f"  ⚠️  DIAGNOSIS: Nearest element in 5×5×5 neighborhood - enhanced search should find")
        elif res['octant_dist_max'] > 2:
            print(f"  ⚠️  DIAGNOSIS: Nearest element >2 octants away - beyond 5×5×5 search range")

        if res['dist_to_nearest'] > 1e-3:
            print(f"  ⚠️  DIAGNOSIS: Large distance to nearest element - particle may have left domain")

    # Aggregate statistics
    print("\n" + "=" * 80)
    print("Aggregate Statistics:")
    print("-" * 80)

    distances = [r['dist_to_nearest'] for r in results]
    octant_dists = [r['octant_dist_max'] for r in results]
    leaf_dists = [r['leaf_distance'] for r in results if r['leaf_distance'] >= 0]

    print(f"\nDistance to nearest element:")
    print(f"  Mean:   {np.mean(distances):.6e}")
    print(f"  Median: {np.median(distances):.6e}")
    print(f"  Max:    {np.max(distances):.6e}")
    print(f"  Min:    {np.min(distances):.6e}")

    print(f"\nOctant distance (max-norm):")
    print(f"  Within 3×3×3 (≤1): {sum(1 for d in octant_dists if d <= 1)} ({100*sum(1 for d in octant_dists if d <= 1)/len(octant_dists):.1f}%)")
    print(f"  Within 5×5×5 (≤2): {sum(1 for d in octant_dists if d <= 2)} ({100*sum(1 for d in octant_dists if d <= 2)/len(octant_dists):.1f}%)")
    print(f"  Beyond 5×5×5 (>2): {sum(1 for d in octant_dists if d > 2)} ({100*sum(1 for d in octant_dists if d > 2)/len(octant_dists):.1f}%)")

    if leaf_dists:
        print(f"\nLeaf distance (Morton order):")
        print(f"  Mean:   {np.mean(leaf_dists):.1f} leaves")
        print(f"  Median: {np.median(leaf_dists):.1f} leaves")
        print(f"  Max:    {np.max(leaf_dists):.0f} leaves")

    # Root cause analysis
    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS:")
    print("-" * 80)

    within_3x3x3 = sum(1 for d in octant_dists if d <= 1)
    within_5x5x5 = sum(1 for d in octant_dists if d <= 2)
    beyond_5x5x5 = sum(1 for d in octant_dists if d > 2)

    if within_3x3x3 > 0:
        pct = 100 * within_3x3x3 / len(octant_dists)
        print(f"\n⚠️  CRITICAL: {within_3x3x3} ({pct:.1f}%) lost particles have nearest element in 3×3×3 neighborhood!")
        print(f"   These SHOULD have been found by neighbors search.")
        print(f"   Possible causes:")
        print(f"     - Point-in-tet check failed (numerical precision)")
        print(f"     - Element is degenerate/invalid")
        print(f"     - Query position exactly on element boundary")

    if within_5x5x5 - within_3x3x3 > 0:
        pct = 100 * (within_5x5x5 - within_3x3x3) / len(octant_dists)
        print(f"\n⚠️  {within_5x5x5 - within_3x3x3} ({pct:.1f}%) lost particles have nearest element in 5×5×5 shell")
        print(f"   These should be found by enhanced neighbors search.")
        print(f"   If lost, check if L2_SEARCH_METHOD='neighbors' is using 5×5×5 fallback.")

    if beyond_5x5x5 > 0:
        pct = 100 * beyond_5x5x5 / len(octant_dists)
        print(f"\n✅  {beyond_5x5x5} ({pct:.1f}%) lost particles have nearest element >2 octants away")
        print(f"   This is expected - beyond search coverage.")
        print(f"   To improve retention:")
        print(f"     - Increase L1 depth: N_HOPS=7 (expensive)")
        print(f"     - Use larger L2 radius (radius=64+)")
        print(f"     - Use hierarchical search at multiple depths")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
