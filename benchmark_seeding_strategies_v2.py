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
7. 'mesh_aligned_octree': Mesh-aligned octree search (NEW - Kuhn meshes only)

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
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.search.mesh_aligned_morton_builder import build_mesh_aligned_morton_structure
from jaxtrace.gpu.search.mesh_aligned_morton_search import upload_mesh_aligned_morton_to_gpu
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
PARTICLE_GRID_RESOLUTION = (60, 90, 60)  # 300,000 particles (increased from 225K)
PARTICLE_BOUNDS_FRACTION = {
    'x': (0.3, 0.7),
    'y': (0.2, 0.8),
    'z': (0.3, 1.0),
}

# RK4 integration
DT = 0.0025 # Timestep
N_STEPS = 100  # Number of RK4 steps (reduced for faster benchmark)

# L1 configuration (consistent across all tests)
ENABLE_L1_SEARCH = True
N_HOPS = 5

# Point-in-tet method (use INVERSE for fair comparison - fastest validated)
POINT_IN_TET_METHOD = 'inverse'

SEED = 42


def run_initial_assignment(positions_gpu, mesh_gpu_octree, l2_method, l2_radius=None, incremental_radii=None,
                           mesh_aligned_octree_neighbors_gpu=None, mesh_aligned_octree_multi_gpu=None):
    """Run initial assignment with specified L2 method."""

    # Set configuration
    config.POINT_IN_TET_METHOD = POINT_IN_TET_METHOD

    if l2_method == 'mesh_aligned_neighbors':
        # Use mesh-aligned neighbor search for initial assignment
        from jaxtrace.gpu.search.mesh_aligned_search_with_neighbors import search_batch_with_precomputed_neighbors

        octree_to_use = mesh_aligned_octree_neighbors_gpu
        max_tests = 20

        t_start = time.time()
        element_ids, n_tests = search_batch_with_precomputed_neighbors(
            positions_gpu,
            octree_to_use,
            levels_to_try=(14, 13, 12),
            max_tests_per_cell=max_tests
        )
        element_ids = jax.block_until_ready(element_ids)
        t_elapsed = time.time() - t_start

    elif l2_method == 'radius':
        if l2_radius is None:
            l2_radius = 10
        # Use large radii for initial assignment
        initial_radius = 500
        fallback_radii = [1000, 2000, 5000, 10000, 100000]

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

    elif l2_method in ['incremental', 'neighbors', 'hierarchical', 'mesh_aligned_octree', 'mesh_aligned_octree_multi', 'mesh_aligned_octree_multi_local', 'mesh_aligned_morton']:
        # For these methods, we need to use the RK4 function which supports them
        # For initial assignment, use large radius fallback
        initial_radius = 500
        fallback_radii = [1000, 2000, 5000, 10000, 100000]

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

    else:
        raise ValueError(f"Unknown L2 method: {l2_method}")

    n_assigned = int(jnp.sum(element_ids >= 0))

    return element_ids, n_assigned, t_elapsed


def run_rk4_tracking(positions_gpu, element_ids_gpu, mesh_gpu, mesh_gpu_octree,
                     element_volumes_gpu, velocity_sequence_gpu,
                     l2_method, l2_radius=None, incremental_radii=None, n_steps=100,
                     mesh_aligned_octree_gpu=None, mesh_aligned_morton_gpu=None,
                     mesh_aligned_octree_neighbors_gpu=None, mesh_aligned_octree_multi_gpu=None):
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

    elif l2_method == 'mesh_aligned_octree':
        # Temporarily set config for this test
        original_l2_method = config.L2_SEARCH_METHOD
        config.L2_SEARCH_METHOD = 'mesh_aligned_octree'

        rk4_step = create_rk4_fully_fused_timedep(
            mesh_gpu_connectivity=mesh_gpu.connectivity,
            mesh_gpu_node_positions=mesh_gpu.node_positions,
            mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
            mesh_gpu_element_volumes=element_volumes_gpu,
            mesh_gpu_global_morton=mesh_gpu_octree,
            n_hops=N_HOPS,
            enable_l1_search=ENABLE_L1_SEARCH,
            l2_search_method='radius',  # Fallback method (won't be used)
            mesh_aligned_octree=mesh_aligned_octree_gpu
        )

        # Restore config
        config.L2_SEARCH_METHOD = original_l2_method

    elif l2_method == 'mesh_aligned_morton':
        # Temporarily set config for this test
        original_l2_method = config.L2_SEARCH_METHOD
        config.L2_SEARCH_METHOD = 'mesh_aligned_morton'

        rk4_step = create_rk4_fully_fused_timedep(
            mesh_gpu_connectivity=mesh_gpu.connectivity,
            mesh_gpu_node_positions=mesh_gpu.node_positions,
            mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
            mesh_gpu_element_volumes=element_volumes_gpu,
            mesh_gpu_global_morton=mesh_gpu_octree,
            l2_search_radius=l2_radius if l2_radius is not None else 2,
            l2_incremental_radii=incremental_radii if incremental_radii is not None else (2, 5, 10),
            n_hops=N_HOPS,
            enable_l1_search=ENABLE_L1_SEARCH,
            l2_search_method='incremental' if incremental_radii is not None else 'radius',
            mesh_aligned_morton=mesh_aligned_morton_gpu
        )

        # Restore config
        config.L2_SEARCH_METHOD = original_l2_method

    elif l2_method == 'mesh_aligned_octree_multi':
        # Multi-cell vertex registration (Phase 2) - CENTER CELL ONLY
        original_l2_method = config.L2_SEARCH_METHOD
        config.L2_SEARCH_METHOD = 'mesh_aligned_octree'

        rk4_step = create_rk4_fully_fused_timedep(
            mesh_gpu_connectivity=mesh_gpu.connectivity,
            mesh_gpu_node_positions=mesh_gpu.node_positions,
            mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
            mesh_gpu_element_volumes=element_volumes_gpu,
            mesh_gpu_global_morton=mesh_gpu_octree,
            n_hops=N_HOPS,
            enable_l1_search=ENABLE_L1_SEARCH,
            l2_search_method='radius',  # Fallback method (won't be used)
            mesh_aligned_octree=mesh_aligned_octree_multi_gpu,  # Use multi-cell octree
            mesh_aligned_octree_use_multi_local=False  # Center cell only
        )

        # Restore config
        config.L2_SEARCH_METHOD = original_l2_method

    elif l2_method == 'mesh_aligned_octree_multi_local':
        # Multi-cell vertex registration (Phase 2) + 2×2×2 LOCAL SEARCH (Option A)
        original_l2_method = config.L2_SEARCH_METHOD
        config.L2_SEARCH_METHOD = 'mesh_aligned_octree'

        rk4_step = create_rk4_fully_fused_timedep(
            mesh_gpu_connectivity=mesh_gpu.connectivity,
            mesh_gpu_node_positions=mesh_gpu.node_positions,
            mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
            mesh_gpu_element_volumes=element_volumes_gpu,
            mesh_gpu_global_morton=mesh_gpu_octree,
            n_hops=N_HOPS,
            enable_l1_search=ENABLE_L1_SEARCH,
            l2_search_method='radius',  # Fallback method (won't be used)
            mesh_aligned_octree=mesh_aligned_octree_multi_gpu,  # Use multi-cell octree
            mesh_aligned_octree_use_multi_local=True  # NEW: 2×2×2 local search
        )

        # Restore config
        config.L2_SEARCH_METHOD = original_l2_method

    elif l2_method == 'mesh_aligned_neighbors':
        # Temporarily set config for this test
        original_l2_method = config.L2_SEARCH_METHOD
        config.L2_SEARCH_METHOD = 'mesh_aligned_neighbors'

        rk4_step = create_rk4_fully_fused_timedep(
            mesh_gpu_connectivity=mesh_gpu.connectivity,
            mesh_gpu_node_positions=mesh_gpu.node_positions,
            mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
            mesh_gpu_element_volumes=element_volumes_gpu,
            mesh_gpu_global_morton=mesh_gpu_octree,
            n_hops=N_HOPS,
            enable_l1_search=ENABLE_L1_SEARCH,
            l2_search_method='radius',  # Fallback (won't be used)
            mesh_aligned_octree_neighbors=mesh_aligned_octree_neighbors_gpu
        )

        # Restore config
        config.L2_SEARCH_METHOD = original_l2_method

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

    # Build mesh-aligned octree (single-cell registration)
    print(f"\n  Building mesh-aligned octree (single-cell)...")
    t_mesh_octree = time.time()
    mesh_octree_cells = extract_octree_cells_single(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    t_mesh_octree = time.time() - t_mesh_octree
    print(f"    Extracted {mesh_octree_cells.n_cells:,} cells in {t_mesh_octree:.2f}s")
    print(f"    Elements per cell: {mesh_octree_cells.elements_per_cell_mean:.2f}")
    print(f"    Cells per element: {mesh_octree_cells.cells_per_element_mean:.2f}")

    # Build mesh-aligned octree (multi-cell vertex registration)
    print(f"\n  Building mesh-aligned octree (multi-cell vertex registration)...")
    t_mesh_octree_multi = time.time()
    mesh_octree_cells_multi = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    t_mesh_octree_multi = time.time() - t_mesh_octree_multi
    print(f"    Extracted {mesh_octree_cells_multi.n_cells:,} cells in {t_mesh_octree_multi:.2f}s")
    print(f"    Elements per cell: {mesh_octree_cells_multi.elements_per_cell_mean:.2f}")
    print(f"    Cells per element: {mesh_octree_cells_multi.cells_per_element_mean:.2f}")

    # Build mesh-aligned octree with neighbor table (Option B)
    print(f"\n  Building mesh-aligned octree with neighbor table (Option B)...")
    from jaxtrace.gpu.search.mesh_aligned_octree_with_neighbor_table import (
        add_neighbor_table_to_octree,
        upload_octree_with_neighbors_to_gpu
    )
    t_neighbor_build = time.time()
    octree_with_neighbors = add_neighbor_table_to_octree(mesh_octree_cells, verbose=False)
    t_neighbor_build = time.time() - t_neighbor_build
    print(f"    Neighbor table built in {t_neighbor_build:.2f}s")
    # Compute mean neighbors from cell_neighbors array (count non-negative entries)
    mean_neighbors = (octree_with_neighbors.cell_neighbors >= 0).sum(axis=1).mean()
    print(f"    Mean neighbors per cell: {mean_neighbors:.1f}")

    print("\n[5/10] Uploading to GPU...")
    element_neighbors = build_element_neighbors_array(connectivity, method='face', verbose=False)
    mesh_gpu = upload_mesh_to_gpu(connectivity, node_positions, element_neighbors, verbose=False)
    mesh_gpu_octree = upload_global_morton_to_gpu(octree_struct, connectivity, node_positions)

    # Upload mesh-aligned structures
    print(f"  Uploading mesh-aligned octree (single-cell)...")
    mesh_aligned_octree_gpu = upload_mesh_aligned_octree_to_gpu(
        node_positions, connectivity, mesh_octree_cells, verbose=False
    )

    print(f"  Uploading mesh-aligned octree (multi-cell)...")
    mesh_aligned_octree_multi_gpu = upload_mesh_aligned_octree_to_gpu(
        node_positions, connectivity, mesh_octree_cells_multi, verbose=False
    )

    print(f"  Uploading mesh-aligned octree with neighbors (single-cell)...")
    mesh_aligned_octree_neighbors_gpu = upload_octree_with_neighbors_to_gpu(
        connectivity, node_positions, octree_with_neighbors, verbose=False
    )

    # For multi-cell octree, we don't need a neighbor table since each element
    # is already registered in ~4 cells (vertices at cube corners).
    # The neighbor table is designed for single-cell registration where elements
    # only appear in one cell. For multi-cell, searching neighbors would mean
    # searching ~4 cells × 26 neighbors = ~104 cells, which is excessive.
    #
    # Instead, we'll use the multi-cell octree directly without neighbors.
    # This will search ~4 cells per particle (much better than single-cell's 1 cell).
    print(f"\n  NOTE: Multi-cell octree doesn't use neighbor table")
    print(f"    Multi-cell registration already covers ~4 cells per element")
    print(f"    Neighbor table would search ~104 cells (excessive)")
    mesh_aligned_octree_neighbors_multi_gpu = None  # Not used

    print(f"  Building mesh-aligned Morton (hybrid)...")
    mesh_aligned_morton_struct = build_mesh_aligned_morton_structure(
        node_positions, connectivity, mesh_octree_cells=mesh_octree_cells, verbose=False
    )
    print(f"    Elements per cell: mean={mesh_aligned_morton_struct.elements_per_cell_mean:.1f}, "
          f"max={mesh_aligned_morton_struct.elements_per_cell_max}")
    mesh_aligned_morton_gpu = upload_mesh_aligned_morton_to_gpu(
        node_positions, connectivity, mesh_aligned_morton_struct, verbose=False
    )

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

    print("\n[6/10] Generating particles (perturbed element centroids)...")
    # Use perturbed element centroids for realistic seeding
    # This ensures particles start inside elements (not in voids)

    # Compute mesh bounds
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)

    # Filter elements in middle 40% of x-axis (0.3 to 0.7 of x range)
    n_elements = connectivity.shape[0]
    element_centroids = np.zeros((n_elements, 3), dtype=np.float32)
    for elem_idx in range(n_elements):
        elem_nodes = connectivity[elem_idx]
        elem_positions = node_positions[elem_nodes]
        element_centroids[elem_idx] = elem_positions.mean(axis=0)

    x_min_filter = domain_min[0] + 0.3 * (domain_max[0] - domain_min[0])
    x_max_filter = domain_min[0] + 0.7 * (domain_max[0] - domain_min[0])

    valid_elements_mask = (element_centroids[:, 0] >= x_min_filter) & (element_centroids[:, 0] <= x_max_filter)
    valid_element_ids = np.where(valid_elements_mask)[0]

    print(f"  Filtering elements: middle 40% of x-axis")
    print(f"    X range: [{x_min_filter:.6f}, {x_max_filter:.6f}]")
    print(f"    Valid elements: {len(valid_element_ids):,} / {n_elements:,} ({100*len(valid_element_ids)/n_elements:.1f}%)")

    # Calculate desired particle count from grid resolution
    np.random.seed(42)
    nx, ny, nz = PARTICLE_GRID_RESOLUTION
    n_particles = nx * ny * nz

    # Select random elements from valid set
    selected_elements = np.random.choice(valid_element_ids, n_particles, replace=True)

    # Compute element centroids
    particle_positions = np.zeros((n_particles, 3), dtype=np.float32)
    for i, elem_idx in enumerate(selected_elements):
        elem_nodes = connectivity[elem_idx]
        elem_positions = node_positions[elem_nodes]
        particle_positions[i] = elem_positions.mean(axis=0)

    # Add small perturbations (10% of smallest element size)
    # Calculate characteristic element sizes (sample for speed)
    sample_size = min(100000, len(valid_element_ids))
    element_sizes = np.zeros(sample_size, dtype=np.float32)
    for i in range(sample_size):
        elem_idx = valid_element_ids[i % len(valid_element_ids)]
        elem_nodes = connectivity[elem_idx]
        elem_positions = node_positions[elem_nodes]
        # Compute min edge length
        edges = []
        for j in range(4):
            for k in range(j+1, 4):
                edge_len = np.linalg.norm(elem_positions[j] - elem_positions[k])
                edges.append(edge_len)
        element_sizes[i] = min(edges)

    min_element_size = np.percentile(element_sizes[element_sizes > 0], 5)

    # Generate 5 seeding strategies
    seeding_strategies = []
    perturbation_scales = [0.0, 0.1, 1.0, 2.0, 3.0]  # Multiples of min_element_size

    for scale_factor in perturbation_scales:
        perturbation_scale = min_element_size * scale_factor
        positions = particle_positions.copy()

        if scale_factor > 0:
            perturbations = np.random.randn(n_particles, 3).astype(np.float32) * perturbation_scale
            positions += perturbations
            strategy_name = f"Perturbed {scale_factor:.1f}× min element"
        else:
            perturbations = np.zeros((n_particles, 3), dtype=np.float32)
            strategy_name = "Element centroids (no perturbation)"

        seeding_strategies.append({
            'name': strategy_name,
            'scale_factor': scale_factor,
            'positions': positions,
            'perturbation_scale': perturbation_scale,
            'mean_perturbation': np.linalg.norm(perturbations, axis=1).mean()
        })

    # Use default strategy (0.1× perturbation) for compatibility
    positions_gpu = jax.device_put(seeding_strategies[1]['positions'])
    ground_truth_element_ids = selected_elements.copy()
    ground_truth_element_ids_gpu = jax.device_put(ground_truth_element_ids)

    print(f"  Generated {len(seeding_strategies)} seeding strategies:")
    for strategy in seeding_strategies:
        print(f"    - {strategy['name']}: scale={strategy['perturbation_scale']:.6e}, mean={strategy['mean_perturbation']:.6e}")
    print(f"  Ground truth element IDs stored for all strategies")

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

        # # Fixed radius=30 (max coverage)
        # {
        #     'name': 'Fixed radius=30 (max coverage)',
        #     'l2_method': 'radius',
        #     'l2_radius': 30,
        #     'incremental_radii': None,
        #     'description': 'Large radius for maximum retention (61 leaves)',
        #     'expected_leaves': 61
        # },

        # # Incremental 5-tier (PRODUCTION)
        # {
        #     'name': 'Incremental (2,4,8,15,30) - PRODUCTION',
        #     'l2_method': 'incremental',
        #     'l2_radius': None,
        #     'incremental_radii': (2, 4, 8, 15, 30),
        #     'description': '5-tier cascading (production config)',
        #     'expected_leaves': '22.5 avg (conservative)'
        # },

        # # Incremental 3-tier (simpler)
        # {
        #     'name': 'Incremental (2,5,10) - 3-tier',
        #     'l2_method': 'incremental',
        #     'l2_radius': None,
        #     'incremental_radii': (2, 5, 10),
        #     'description': '3-tier cascading (simpler alternative)',
        #     'expected_leaves': '11.5 avg (60/30/10)'
        # },

        # # Neighbors
        # {
        #     'name': 'Neighbors (Morton arithmetic)',
        #     'l2_method': 'neighbors',
        #     'l2_radius': None,
        #     'incremental_radii': None,
        #     'description': 'Morton neighbor arithmetic',
        #     'expected_leaves': 'Variable'
        # },

        # # Hierarchical
        # {
        #     'name': 'Hierarchical (multi-depth)',
        #     'l2_method': 'hierarchical',
        #     'l2_radius': None,
        #     'incremental_radii': None,
        #     'description': 'Multi-depth conditional search',
        #     'expected_leaves': 'Variable'
        # },

        # Mesh-aligned octree (DIRECT)
        {
            'name': 'Mesh-Aligned Octree (direct)',
            'l2_method': 'mesh_aligned_octree',
            'l2_radius': None,
            'incremental_radii': None,
            'description': 'Direct cell lookup (center cell only)',
            'expected_leaves': '~5.9 elements/cell'
        },

        # Mesh-aligned Morton (HYBRID - NEW)
        {
            'name': 'Mesh-Aligned Morton r=2 (HYBRID - NEW)',
            'l2_method': 'mesh_aligned_morton',
            'l2_radius': 2,
            'incremental_radii': None,
            'description': 'Morton radius over cell centers (5 cells)',
            'expected_leaves': '~30 tests (5 cells × 5.9 elem/cell)'
        },

        # Mesh-aligned Morton incremental
        {
            'name': 'Mesh-Aligned Morton (2,5,10) (HYBRID - NEW)',
            'l2_method': 'mesh_aligned_morton',
            'l2_radius': None,
            'incremental_radii': (2, 5, 10),
            'description': 'Incremental radius over cell centers',
            'expected_leaves': '~68 tests avg (11.5 cells × 5.9 elem/cell)'
        },

        # Mesh-aligned neighbors (Option B - NEW)
        {
            'name': 'Mesh-Aligned Neighbors (Option B - NEW)',
            'l2_method': 'mesh_aligned_neighbors',
            'l2_radius': None,
            'incremental_radii': None,
            'description': 'Pre-computed neighbor table (27 cells @ 3 levels)',
            'expected_leaves': '~13.9 tests/particle, 99.95% for centroids'
        },

        # Mesh-aligned octree MULTI-CELL vertex registration (NEW - Phase 2)
        {
            'name': 'Mesh-Aligned Octree Multi-Cell (Phase 2 - NEW)',
            'l2_method': 'mesh_aligned_octree_multi',
            'l2_radius': None,
            'incremental_radii': None,
            'description': 'Multi-cell vertex registration (~4 cells per element)',
            'expected_leaves': '~94 tests/particle (~4 cells × ~23.6 elem/cell)'
        },

        # Mesh-aligned octree MULTI-CELL with 2×2×2 local search (Option A)
        {
            'name': 'Mesh-Aligned Multi-Cell + 2×2×2 Local (Option A - NEW)',
            'l2_method': 'mesh_aligned_octree_multi_local',
            'l2_radius': None,
            'incremental_radii': None,
            'description': 'Multi-cell + 2×2×2 local neighborhood search (8 cells)',
            'expected_leaves': '~146 tests/particle (8 cells × 18.31 elem/cell)'
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
            incremental_radii=cfg['incremental_radii'],
            mesh_aligned_octree_neighbors_gpu=mesh_aligned_octree_neighbors_gpu,
            mesh_aligned_octree_multi_gpu=mesh_aligned_octree_multi_gpu
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
    # 9. Test All Seeding Strategies (No RK4 Tracking)
    # ========================================================================

    print("\n[9/10] Testing all seeding strategies with initial assignment...")
    print("=" * 80)
    print(f"Testing {len(seeding_strategies)} seeding strategies × {len(test_configs)} methods")
    print()

    seeding_results = {}  # seeding_results[strategy_name][method_name] = {...}

    for strategy in seeding_strategies:
        strategy_name = strategy['name']
        positions_gpu_strategy = jax.device_put(strategy['positions'])

        print(f"\nSeeding Strategy: {strategy_name}")
        print(f"  Perturbation scale: {strategy['perturbation_scale']:.6e} ({strategy['scale_factor']:.1f}× min element)")
        print("-" * 80)

        seeding_results[strategy_name] = {}

        for cfg in test_configs:
            name = cfg['name']

            element_ids, n_assigned, t_elapsed = run_initial_assignment(
                positions_gpu_strategy,
                mesh_gpu_octree,
                l2_method=cfg['l2_method'],
                l2_radius=cfg['l2_radius'],
                incremental_radii=cfg['incremental_radii'],
                mesh_aligned_octree_neighbors_gpu=mesh_aligned_octree_neighbors_gpu,
                mesh_aligned_octree_multi_gpu=mesh_aligned_octree_multi_gpu
            )

            success_rate = (n_assigned / n_particles) * 100
            throughput = n_particles / t_elapsed

            seeding_results[strategy_name][name] = {
                'element_ids': element_ids,
                'n_assigned': n_assigned,
                'success_rate': success_rate,
                'time': t_elapsed,
                'throughput': throughput
            }

            print(f"  {name:<40s}: {success_rate:6.2f}% ({n_assigned:,}/{n_particles:,})")

    # ========================================================================
    # 10. Results Analysis
    # ========================================================================

    print("\n[10/10] Results Analysis")
    print("=" * 80)

    # Seeding Strategy × Method Matrix
    print("\nSEEDING STRATEGY × METHOD SEARCHABILITY MATRIX")
    print("=" * 80)
    print(f"{'Strategy \\ Method':<30s}", end="")
    for cfg in test_configs:
        print(f"  {cfg['name'][:20]:>20s}", end="")
    print()
    print("-" * (30 + 22 * len(test_configs)))

    for strategy in seeding_strategies:
        strategy_name = strategy['name']
        scale = strategy['scale_factor']
        print(f"{scale:.1f}× min element{'':<17s}", end="")
        for cfg in test_configs:
            name = cfg['name']
            success = seeding_results[strategy_name][name]['success_rate']
            print(f"  {success:19.2f}%", end="")
        print()

    # Best Method per Seeding Strategy
    print("\n\nBEST METHOD PER SEEDING STRATEGY")
    print("=" * 80)
    print(f"{'Strategy':<30s}  {'Best Method':<40s}  {'Success Rate':>12s}")
    print("-" * 80)

    for strategy in seeding_strategies:
        strategy_name = strategy['name']
        best_success = 0
        best_method = None

        for cfg in test_configs:
            name = cfg['name']
            success = seeding_results[strategy_name][name]['success_rate']
            if success > best_success:
                best_success = success
                best_method = name

        print(f"{strategy_name:<30s}  {best_method:<40s}  {best_success:11.2f}%")

    # Method Robustness Analysis
    print("\n\nMETHOD ROBUSTNESS ACROSS PERTURBATION LEVELS")
    print("=" * 80)
    print(f"{'Method':<40s}  {'Avg Success':>12s}  {'Min Success':>12s}  {'Max Success':>12s}  {'Std Dev':>8s}")
    print("-" * 80)

    for cfg in test_configs:
        name = cfg['name']
        success_rates = [seeding_results[strategy['name']][name]['success_rate']
                         for strategy in seeding_strategies]
        avg_success = np.mean(success_rates)
        min_success = np.min(success_rates)
        max_success = np.max(success_rates)
        std_success = np.std(success_rates)

        print(f"{name:<40s}  {avg_success:11.2f}%  {min_success:11.2f}%  {max_success:11.2f}%  {std_success:7.2f}%")

    # Key Findings
    print("\n\nKEY FINDINGS")
    print("=" * 80)

    # Find multi-cell vs single-cell comparison
    multi_cell_name = 'Mesh-Aligned Multi-Cell + 2×2×2 Local (Option A - NEW)'
    single_cell_name = 'Mesh-Aligned Single-Cell (direct)'

    if multi_cell_name in test_configs[0] or any(cfg['name'] == multi_cell_name for cfg in test_configs):
        print("\nMulti-Cell vs Single-Cell Comparison:")
        for strategy in seeding_strategies:
            strategy_name = strategy['name']
            scale = strategy['scale_factor']

            multi_success = None
            single_success = None

            for cfg in test_configs:
                if cfg['name'] == multi_cell_name:
                    multi_success = seeding_results[strategy_name][cfg['name']]['success_rate']
                elif cfg['name'] == single_cell_name:
                    single_success = seeding_results[strategy_name][cfg['name']]['success_rate']

            if multi_success is not None and single_success is not None:
                improvement = multi_success - single_success
                print(f"  {scale:.1f}× perturbation: Multi={multi_success:6.2f}%, Single={single_success:6.2f}%, Δ={improvement:+6.2f}%")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
