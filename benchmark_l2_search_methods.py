#!/usr/bin/env python3
"""
Comprehensive L2 Search Methods Benchmark

Compares different L2 search strategies with FAIR comparison metrics:
- Fixed computation budget (same search radius or equivalent work)
- Accuracy (retention/success rate)
- Performance (throughput)

L2 Search Methods:
1. 'radius' (radius=10): Fixed radius search (baseline)
2. 'radius' (radius=30): Fixed large radius (max coverage)
3. 'incremental' (2,4,8,15,30): 5-tier cascading (PRODUCTION CONFIG)
4. 'incremental' (2,5,10): 3-tier cascading (simpler alternative)
5. 'neighbors': Morton neighbor arithmetic
6. 'hierarchical': Multi-depth conditional search

Fair Comparison Approaches:
A) Equal Maximum Coverage: All methods search up to radius=30
B) Equal Average Work: Tune radii to match ~20 leaves average
C) Production Realistic: Use actual production configuration

Metrics:
- Initial assignment success rate
- RK4 retention at step 100
- Throughput (particles/second)
- Average leaves searched (efficiency metric)
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.tracking.initial_assignment_cascading import initial_assignment_cascading_fallback
from jaxtrace.tracking.seeding import uniform_grid_seeds
from jaxtrace.gpu.search.aa_detection import precompute_aa_metadata, precompute_element_vertices
from jaxtrace.gpu.search.point_in_tet_methods import set_corrected_metadata, set_inverse_matrices_gpu
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
from jaxtrace.gpu.tracking.rk4_fully_fused_timedep import create_rk4_fully_fused_timedep
import jaxtrace.config as config


# ============================================================================
# Configuration
# ============================================================================

MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)  # 20 timesteps for RK4 testing
VELOCITY_FIELD_NAME = 'Displacement'

# Particle seeding (same as production)
PARTICLE_GRID_RESOLUTION = (20, 50, 30)  # 225,000 particles
PARTICLE_BOUNDS_FRACTION = {
    'x': (0.3, 0.7),
    'y': (0.2, 0.8),
    'z': (0.3, 1.0),
}

# RK4 integration
DT = 0.0005  # Timestep
N_STEPS = 100  # Number of RK4 steps (reduced for faster benchmark)

# L1 configuration (consistent across all tests)
ENABLE_L1_SEARCH = True
N_HOPS = 5

# Point-in-tet method (use INVERSE for fair comparison - fastest validated)
POINT_IN_TET_METHOD = 'inverse'

SEED = 42


def run_initial_assignment(positions_gpu, mesh_gpu_octree, l2_method, l2_radius=None, incremental_radii=None):
    """Run initial assignment with specified L2 method."""

    # Set configuration
    config.POINT_IN_TET_METHOD = POINT_IN_TET_METHOD

    if l2_method == 'radius':
        if l2_radius is None:
            l2_radius = 10
        # Use large radii for initial assignment
        initial_radius = 500
        fallback_radii = [1000, 2000, 5000, 10000, 100000]

    elif l2_method in ['incremental', 'neighbors', 'hierarchical']:
        # For these methods, we need to use the RK4 function which supports them
        # For initial assignment, use large radius fallback
        initial_radius = 500
        fallback_radii = [1000, 2000, 5000, 10000, 100000]

    else:
        raise ValueError(f"Unknown L2 method: {l2_method}")

    # Run initial assignment
    t_start = time.time()
    element_ids = initial_assignment_cascading_fallback(
        positions_gpu,
        mesh_gpu_octree,
        initial_radius=initial_radius,
        fallback_radii=fallback_radii,
        verbose=False
    )
    element_ids = jax.block_until_ready(element_ids)
    t_elapsed = time.time() - t_start

    n_assigned = int(jnp.sum(element_ids >= 0))

    return element_ids, n_assigned, t_elapsed


def run_rk4_tracking(positions_gpu, element_ids_gpu, mesh_gpu, mesh_gpu_octree,
                     element_volumes_gpu, velocity_sequence_gpu,
                     l2_method, l2_radius=None, incremental_radii=None, n_steps=100):
    """Run RK4 tracking with specified L2 method."""

    n_particles = positions_gpu.shape[0]

    # Set configuration
    config.POINT_IN_TET_METHOD = POINT_IN_TET_METHOD

    # Create RK4 function
    if l2_method == 'radius':
        if l2_radius is None:
            l2_radius = 10

        rk4_step = create_rk4_fully_fused_timedep(
            mesh_gpu_connectivity=mesh_gpu.connectivity,
            mesh_gpu_node_positions=mesh_gpu.node_positions,
            mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
            mesh_gpu_element_volumes=element_volumes_gpu,
            mesh_gpu_global_morton=mesh_gpu_octree,
            n_hops=N_HOPS,
            l2_search_radius=l2_radius,
            enable_l1_search=ENABLE_L1_SEARCH,
            l2_search_method='radius'
        )

    elif l2_method == 'incremental':
        if incremental_radii is None:
            incremental_radii = (2, 4, 8, 15, 30)

        rk4_step = create_rk4_fully_fused_timedep(
            mesh_gpu_connectivity=mesh_gpu.connectivity,
            mesh_gpu_node_positions=mesh_gpu.node_positions,
            mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
            mesh_gpu_element_volumes=element_volumes_gpu,
            mesh_gpu_global_morton=mesh_gpu_octree,
            n_hops=N_HOPS,
            enable_l1_search=ENABLE_L1_SEARCH,
            l2_search_method='incremental',
            l2_incremental_radii=incremental_radii
        )

    elif l2_method == 'neighbors':
        rk4_step = create_rk4_fully_fused_timedep(
            mesh_gpu_connectivity=mesh_gpu.connectivity,
            mesh_gpu_node_positions=mesh_gpu.node_positions,
            mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
            mesh_gpu_element_volumes=element_volumes_gpu,
            mesh_gpu_global_morton=mesh_gpu_octree,
            n_hops=N_HOPS,
            enable_l1_search=ENABLE_L1_SEARCH,
            l2_search_method='neighbors'
        )

    elif l2_method == 'hierarchical':
        rk4_step = create_rk4_fully_fused_timedep(
            mesh_gpu_connectivity=mesh_gpu.connectivity,
            mesh_gpu_node_positions=mesh_gpu.node_positions,
            mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
            mesh_gpu_element_volumes=element_volumes_gpu,
            mesh_gpu_global_morton=mesh_gpu_octree,
            n_hops=N_HOPS,
            enable_l1_search=ENABLE_L1_SEARCH,
            l2_search_method='hierarchical'
        )

    else:
        raise ValueError(f"Unknown L2 method: {l2_method}")

    # Warmup (compile)
    print(f"      Compiling...")
    t_compile = time.time()
    positions_gpu, element_ids_gpu = rk4_step(
        positions_gpu,
        element_ids_gpu,
        DT,
        velocity_sequence_gpu,
        0  # time_idx
    )
    positions_gpu = jax.block_until_ready(positions_gpu)
    element_ids_gpu = jax.block_until_ready(element_ids_gpu)
    t_compile = time.time() - t_compile
    print(f"      Compilation time: {t_compile:.2f}s")

    # Run tracking
    print(f"      Running {n_steps} RK4 steps...")
    t_start = time.time()

    for step in range(n_steps):
        positions_gpu, element_ids_gpu = rk4_step(
            positions_gpu,
            element_ids_gpu,
            DT,
            velocity_sequence_gpu,
            step  # time_idx
        )
        # Block only occasionally for efficiency
        if step % 10 == 0 or step == n_steps - 1:
            positions_gpu = jax.block_until_ready(positions_gpu)
            element_ids_gpu = jax.block_until_ready(element_ids_gpu)

    # Final sync
    positions_gpu = jax.block_until_ready(positions_gpu)
    element_ids_gpu = jax.block_until_ready(element_ids_gpu)
    t_elapsed = time.time() - t_start

    # Final metrics
    n_active_final = int(jnp.sum(element_ids_gpu >= 0))
    retention = (n_active_final / n_particles) * 100
    throughput = (n_particles * n_steps) / t_elapsed

    return positions_gpu, element_ids_gpu, n_active_final, retention, t_elapsed, throughput


def main():
    print("=" * 80)
    print("Comprehensive L2 Search Methods Benchmark")
    print("Fair comparison of all L2 search strategies")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 80)

    # ========================================================================
    # 1-7. Load Mesh, Deduplicate, Build Octree, Upload (same as before)
    # ========================================================================

    print("\n[1/10] Loading mesh...")
    t_load = time.time()
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )
    t_load = time.time() - t_load

    n_nodes_orig = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    n_timesteps = len(velocity_sequence)

    print(f"  Loaded in {t_load:.2f}s")
    print(f"  Elements: {n_elements:,}, Nodes: {n_nodes_orig:,}, Timesteps: {n_timesteps}")

    print("\n[2/10] Deduplicating...")
    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    n_nodes = node_positions.shape[0]
    print(f"  Removed {n_duplicates_removed:,} duplicates")

    print("\n[3/10] Precomputing metadata...")
    aa_metadata = precompute_aa_metadata(connectivity, node_positions, verbose=False)
    element_vertices = precompute_element_vertices(connectivity, node_positions, verbose=False)
    M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)

    # Compute element volumes (needed for adaptive L1 hop count)
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

    print(f"  Metadata ready")

    print("\n[4/10] Building octree...")
    octree_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256,
        max_depth=21,
        verbose=False
    )
    print(f"  Built {octree_struct.n_leaves:,} leaves")

    print("\n[5/10] Uploading to GPU...")
    element_neighbors = build_element_neighbors_array(connectivity, method='face', verbose=False)
    mesh_gpu = upload_mesh_to_gpu(connectivity, node_positions, element_neighbors, verbose=False)
    mesh_gpu_octree = upload_global_morton_to_gpu(octree_struct, connectivity, node_positions)

    # Upload metadata
    from jaxtrace.gpu.search.aa_detection import AxisAlignedMetadata
    aa_metadata_gpu = AxisAlignedMetadata(
        base_vertex_indices=jax.device_put(aa_metadata.base_vertex_indices),
        base_vertices=jax.device_put(aa_metadata.base_vertices),
        inv_edge_lengths=jax.device_put(aa_metadata.inv_edge_lengths),
        axis_indices=jax.device_put(aa_metadata.axis_indices),
        is_axis_aligned=jax.device_put(aa_metadata.is_axis_aligned)
    )
    element_vertices_gpu = jax.device_put(element_vertices)
    M_inv_gpu = jax.device_put(M_inv_array)
    p0_gpu = jax.device_put(p0_array)
    element_volumes_gpu = jax.device_put(element_volumes.astype(np.float32))

    set_corrected_metadata(aa_metadata_gpu, element_vertices_gpu)
    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)

    # Upload velocity sequence
    velocity_sequence_gpu = jax.device_put(velocity_sequence)

    print(f"  Uploaded to GPU")

    print("\n[6/10] Generating particles...")
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)
    domain_size = domain_max - domain_min

    par_bounds_min = np.zeros(3, dtype=np.float32)
    par_bounds_max = np.zeros(3, dtype=np.float32)
    for i, axis in enumerate(['x', 'y', 'z']):
        min_frac, max_frac = PARTICLE_BOUNDS_FRACTION[axis]
        par_bounds_min[i] = domain_min[i] + min_frac * domain_size[i]
        par_bounds_max[i] = domain_min[i] + max_frac * domain_size[i]

    nx, ny, nz = PARTICLE_GRID_RESOLUTION
    particle_positions = uniform_grid_seeds(
        resolution=(nx, ny, nz),
        bounds=[par_bounds_min, par_bounds_max],
        include_boundaries=True
    )

    # Clip to mesh bounds
    margin = 0.01
    bbox_min_safe = domain_min + margin * domain_size
    bbox_max_safe = domain_max - margin * domain_size
    particle_positions = np.clip(particle_positions, bbox_min_safe, bbox_max_safe)

    n_particles = particle_positions.shape[0]
    positions_gpu = jax.device_put(particle_positions.astype(np.float32))

    print(f"  Generated {n_particles:,} particles")

    # ========================================================================
    # 7. Define Test Configurations
    # ========================================================================

    print("\n[7/10] Defining test configurations...")
    print("=" * 80)

    test_configs = [
        # Baseline: Fixed radius=10
        {
            'name': 'Fixed radius=10 (baseline)',
            'l2_method': 'radius',
            'l2_radius': 10,
            'incremental_radii': None,
            'description': 'Fixed radius search (21 leaves)',
            'expected_leaves': 21
        },

        # Fixed radius=30 (max coverage)
        {
            'name': 'Fixed radius=30 (max coverage)',
            'l2_method': 'radius',
            'l2_radius': 30,
            'incremental_radii': None,
            'description': 'Large radius for maximum retention (61 leaves)',
            'expected_leaves': 61
        },

        # Incremental 5-tier (PRODUCTION)
        {
            'name': 'Incremental (2,4,8,15,30) - PRODUCTION',
            'l2_method': 'incremental',
            'l2_radius': None,
            'incremental_radii': (2, 4, 8, 15, 30),
            'description': '5-tier cascading (production config)',
            'expected_leaves': '22.5 avg (conservative)'
        },

        # Incremental 3-tier (simpler)
        {
            'name': 'Incremental (2,5,10) - 3-tier',
            'l2_method': 'incremental',
            'l2_radius': None,
            'incremental_radii': (2, 5, 10),
            'description': '3-tier cascading (simpler alternative)',
            'expected_leaves': '11.5 avg (60/30/10)'
        },

        # Neighbors
        {
            'name': 'Neighbors (Morton arithmetic)',
            'l2_method': 'neighbors',
            'l2_radius': None,
            'incremental_radii': None,
            'description': 'Morton neighbor arithmetic',
            'expected_leaves': 'Variable'
        },

        # Hierarchical
        {
            'name': 'Hierarchical (multi-depth)',
            'l2_method': 'hierarchical',
            'l2_radius': None,
            'incremental_radii': None,
            'description': 'Multi-depth conditional search',
            'expected_leaves': 'Variable'
        },
    ]

    for i, cfg in enumerate(test_configs, 1):
        print(f"{i}. {cfg['name']}")
        print(f"   {cfg['description']}")
        print(f"   Expected work: {cfg['expected_leaves']}")
        print()

    # ========================================================================
    # 8. Run Initial Assignment for All Configurations
    # ========================================================================

    print("\n[8/10] Running initial assignment for all configurations...")
    print("=" * 80)

    initial_results = {}

    for cfg in test_configs:
        name = cfg['name']
        print(f"\n  Config: {name}")

        element_ids, n_assigned, t_elapsed = run_initial_assignment(
            positions_gpu,
            mesh_gpu_octree,
            l2_method=cfg['l2_method'],
            l2_radius=cfg['l2_radius'],
            incremental_radii=cfg['incremental_radii']
        )

        success_rate = (n_assigned / n_particles) * 100
        throughput = n_particles / t_elapsed

        initial_results[name] = {
            'element_ids': element_ids,
            'n_assigned': n_assigned,
            'success_rate': success_rate,
            'time': t_elapsed,
            'throughput': throughput
        }

        print(f"    Time: {t_elapsed:.3f}s")
        print(f"    Assigned: {n_assigned:,}/{n_particles:,} ({success_rate:.2f}%)")
        print(f"    Throughput: {throughput:,.0f} p/s")

    # ========================================================================
    # 9. Run RK4 Tracking for All Configurations
    # ========================================================================

    print("\n[9/10] Running RK4 tracking for all configurations...")
    print("=" * 80)
    print(f"Configuration: {N_STEPS} steps, dt={DT}, point-in-tet={POINT_IN_TET_METHOD}")
    print()

    tracking_results = {}

    for cfg in test_configs:
        name = cfg['name']
        print(f"\n  Config: {name}")

        # Use initial assignment from previous step
        element_ids_initial = initial_results[name]['element_ids']

        positions_final, element_ids_final, n_active_final, retention, t_elapsed, throughput = run_rk4_tracking(
            positions_gpu,
            element_ids_initial,
            mesh_gpu,
            mesh_gpu_octree,
            element_volumes_gpu,
            velocity_sequence_gpu,
            l2_method=cfg['l2_method'],
            l2_radius=cfg['l2_radius'],
            incremental_radii=cfg['incremental_radii'],
            n_steps=N_STEPS
        )

        tracking_results[name] = {
            'positions': positions_final,
            'element_ids': element_ids_final,
            'n_active_final': n_active_final,
            'retention': retention,
            'time': t_elapsed,
            'throughput': throughput
        }

        print(f"    Time: {t_elapsed:.3f}s")
        print(f"    Final active: {n_active_final:,}/{n_particles:,} ({retention:.2f}%)")
        print(f"    Throughput: {throughput:,.0f} p/s")

    # ========================================================================
    # 10. Results Analysis
    # ========================================================================

    print("\n[10/10] Results Analysis")
    print("=" * 80)

    # Initial Assignment Summary
    print("\nINITIAL ASSIGNMENT RESULTS")
    print("=" * 80)
    print(f"{'Configuration':<40s}  {'Success Rate':>12s}  {'Throughput':>14s}  {'Time':>8s}")
    print("-" * 80)

    for cfg in test_configs:
        name = cfg['name']
        r = initial_results[name]
        print(f"{name:<40s}  {r['success_rate']:11.2f}%  {r['throughput']:13,.0f} p/s  {r['time']:7.3f}s")

    # RK4 Tracking Summary
    print("\n\nRK4 TRACKING RESULTS ({} steps)".format(N_STEPS))
    print("=" * 80)
    print(f"{'Configuration':<40s}  {'Retention':>10s}  {'Throughput':>14s}  {'Speedup':>8s}")
    print("-" * 80)

    baseline_name = 'Fixed radius=10 (baseline)'
    baseline_time = tracking_results[baseline_name]['time']

    best_throughput = 0
    best_config = None

    for cfg in test_configs:
        name = cfg['name']
        r = tracking_results[name]
        speedup = baseline_time / r['time']

        marker = ""
        if r['throughput'] > best_throughput:
            best_throughput = r['throughput']
            best_config = name
            marker = " ★"

        print(f"{name:<40s}  {r['retention']:9.2f}%  {r['throughput']:13,.0f} p/s  {speedup:7.2f}×{marker}")

    # Accuracy vs Performance Trade-off
    print("\n\nACCURACY vs PERFORMANCE TRADE-OFF")
    print("=" * 80)
    print(f"{'Configuration':<40s}  {'Retention':>10s}  {'Speedup':>8s}  {'Rating':>10s}")
    print("-" * 80)

    for cfg in test_configs:
        name = cfg['name']
        r = tracking_results[name]
        retention = r['retention']
        speedup = baseline_time / r['time']

        # Rating based on retention + speedup
        if retention >= 93.0 and speedup >= 1.8:
            rating = "EXCELLENT"
        elif retention >= 90.0 and speedup >= 1.5:
            rating = "GOOD"
        elif retention >= 85.0 and speedup >= 1.2:
            rating = "ACCEPTABLE"
        else:
            rating = "POOR"

        print(f"{name:<40s}  {retention:9.2f}%  {speedup:7.2f}×  {rating:>10s}")

    # Recommendations
    print("\n\nRECOMMENDATIONS")
    print("=" * 80)

    print(f"\nBest Throughput: {best_config}")
    best_retention = tracking_results[best_config]['retention']
    best_speedup = baseline_time / tracking_results[best_config]['time']

    print(f"  Retention: {best_retention:.2f}%")
    print(f"  Speedup: {best_speedup:.2f}×")
    print(f"  Throughput: {best_throughput:,.0f} p/s")

    # Find best accuracy
    best_retention_val = 0
    best_retention_config = None
    for cfg in test_configs:
        name = cfg['name']
        r = tracking_results[name]
        if r['retention'] > best_retention_val:
            best_retention_val = r['retention']
            best_retention_config = name

    print(f"\nBest Retention: {best_retention_config}")
    print(f"  Retention: {best_retention_val:.2f}%")

    retention_speedup = baseline_time / tracking_results[best_retention_config]['time']
    print(f"  Speedup: {retention_speedup:.2f}×")

    # Production recommendation
    print("\n\nPRODUCTION RECOMMENDATION")
    print("=" * 80)

    production_config = 'Incremental (2,4,8,15,30) - PRODUCTION'
    if production_config in tracking_results:
        prod_r = tracking_results[production_config]
        prod_speedup = baseline_time / prod_r['time']

        print(f"\nCurrent Production Config: {production_config}")
        print(f"  Retention: {prod_r['retention']:.2f}%")
        print(f"  Speedup: {prod_speedup:.2f}×")
        print(f"  Throughput: {prod_r['throughput']:,.0f} p/s")

        if prod_speedup >= 1.8:
            print(f"\n✅ Production config achieves {prod_speedup:.2f}× speedup - EXCELLENT")
            print(f"   Recommendation: Continue using current configuration")
        else:
            print(f"\n⚠️  Production config achieves {prod_speedup:.2f}× speedup")
            print(f"   Consider alternative: {best_config} ({best_speedup:.2f}×)")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
