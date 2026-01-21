#!/usr/bin/env python3
"""
Compare pure_aa accuracy against trusted 'current' method.

Only tests 2 methods:
1. current (trusted baseline)
2. pure_aa (suspect - needs validation)
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
from jaxtrace.gpu.search.aa_detection import (
    precompute_aa_metadata,
    precompute_element_vertices,
    AxisAlignedMetadata
)
from jaxtrace.gpu.search.point_in_tet_methods import set_corrected_metadata
import jaxtrace.config as config

# Configuration
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 121)
VELOCITY_FIELD_NAME = 'Displacement'

PARTICLE_GRID_RESOLUTION = (20, 50, 30)
PARTICLE_BOUNDS_FRACTION = {
    'x': (0.3, 0.7),
    'y': (0.2, 0.8),
    'z': (0.3, 1.0),
}

INITIAL_SEARCH_RADIUS = 500
INITIAL_SEARCH_FALLBACK_RADII = [1000, 2000, 5000, 10000, 100000]

def main():
    print("=" * 80)
    print("AA Accuracy Test: pure_aa vs current (TRUSTED)")
    print("=" * 80)

    # Load mesh
    print("\n[1/8] Loading mesh...")
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )
    n_elements = connectivity.shape[0]
    print(f"  Elements: {n_elements:,}, Nodes: {node_positions.shape[0]:,}")

    # Deduplicate
    print("\n[2/8] Deduplicating nodes...")
    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    n_nodes = node_positions.shape[0]
    print(f"  Nodes: {n_nodes:,} (removed {n_duplicates_removed:,} duplicates)")

    # Precompute AA metadata
    print("\n[3/8] Precomputing AA metadata...")
    aa_metadata = precompute_aa_metadata(connectivity, node_positions, verbose=False)
    element_vertices = precompute_element_vertices(connectivity, node_positions, verbose=False)
    n_aa_elements = int(np.sum(aa_metadata.is_axis_aligned))
    aa_percentage = (n_aa_elements / n_elements) * 100
    print(f"  AA elements: {n_aa_elements:,}/{n_elements:,} ({aa_percentage:.2f}%)")

    # Build Morton octree
    print("\n[4/8] Building Morton octree...")
    octree_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256,
        max_depth=21,
        verbose=False
    )
    print(f"  Built {octree_struct.n_leaves:,} leaves")

    # Upload to GPU
    print("\n[5/8] Uploading to GPU...")
    element_neighbors = build_element_neighbors_array(connectivity, method='face', verbose=False)
    mesh_gpu = upload_mesh_to_gpu(
        connectivity=connectivity,
        node_positions=node_positions,
        element_neighbors=element_neighbors,
        verbose=False
    )
    mesh_gpu_octree = upload_global_morton_to_gpu(
        octree_struct,
        connectivity,
        node_positions
    )

    # Upload AA metadata
    aa_metadata_gpu = AxisAlignedMetadata(
        base_vertex_indices=jax.device_put(aa_metadata.base_vertex_indices),
        base_vertices=jax.device_put(aa_metadata.base_vertices),
        inv_edge_lengths=jax.device_put(aa_metadata.inv_edge_lengths),
        axis_indices=jax.device_put(aa_metadata.axis_indices),
        is_axis_aligned=jax.device_put(aa_metadata.is_axis_aligned)
    )
    element_vertices_gpu = jax.device_put(element_vertices)
    set_corrected_metadata(aa_metadata_gpu, element_vertices_gpu)
    print("  Uploaded")

    # Generate particles
    print("\n[6/8] Generating particles...")
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)
    domain_size = domain_max - domain_min

    par_bounds_min = np.zeros(3, dtype=np.float32)
    par_bounds_max = np.zeros(3, dtype=np.float32)
    for i, axis in enumerate(['x', 'y', 'z']):
        min_frac, max_frac = PARTICLE_BOUNDS_FRACTION[axis]
        par_bounds_min[i] = domain_min[i] + min_frac * domain_size[i]
        par_bounds_max[i] = domain_min[i] + max_frac * domain_size[i]
    par_bounds = [par_bounds_min, par_bounds_max]

    nx, ny, nz = PARTICLE_GRID_RESOLUTION
    particle_positions = uniform_grid_seeds(
        resolution=(nx, ny, nz),
        bounds=par_bounds,
        include_boundaries=True
    )

    # Clip to mesh bounds
    mesh_bbox_min = domain_min
    mesh_bbox_max = domain_max
    margin = 0.01
    bbox_min_safe = mesh_bbox_min + margin * (mesh_bbox_max - mesh_bbox_min)
    bbox_max_safe = mesh_bbox_max - margin * (mesh_bbox_max - mesh_bbox_min)
    particle_positions = np.clip(particle_positions, bbox_min_safe, bbox_max_safe)

    n_particles = particle_positions.shape[0]
    print(f"  Generated {n_particles:,} particles")

    positions_gpu = jax.device_put(particle_positions.astype(np.float32))

    # Test 2 methods
    print("\n[7/8] Testing methods...")
    print("=" * 80)

    results = {}

    for method_name in ['current', 'pure_aa']:
        print(f"\nMethod: {method_name}")
        config.POINT_IN_TET_METHOD = method_name

        print("  Warming up JIT...")
        _ = initial_assignment_cascading_fallback(
            positions_gpu[:100],
            mesh_gpu_octree,
            initial_radius=INITIAL_SEARCH_RADIUS,
            fallback_radii=[INITIAL_SEARCH_FALLBACK_RADII[0]],
            verbose=False
        )

        print("  Running full initial assignment...")
        t_start = time.time()
        element_ids_gpu = initial_assignment_cascading_fallback(
            positions_gpu,
            mesh_gpu_octree,
            initial_radius=INITIAL_SEARCH_RADIUS,
            fallback_radii=INITIAL_SEARCH_FALLBACK_RADII,
            verbose=True
        )
        element_ids_gpu = jax.block_until_ready(element_ids_gpu)
        t_elapsed = time.time() - t_start

        n_assigned = int(jnp.sum(element_ids_gpu >= 0))
        success_rate = (n_assigned / n_particles) * 100
        throughput = n_particles / t_elapsed

        results[method_name] = {
            'time': t_elapsed,
            'n_assigned': n_assigned,
            'success_rate': success_rate,
            'throughput': throughput,
            'element_ids': element_ids_gpu
        }

        print(f"  Time: {t_elapsed:.2f}s")
        print(f"  Assigned: {n_assigned:,}/{n_particles:,} ({success_rate:.2f}%)")
        print(f"  Throughput: {throughput:,.0f} particles/s")

    # Compare results
    print("\n[8/8] Accuracy Comparison")
    print("=" * 80)

    ids_current = results['current']['element_ids']
    ids_pure_aa = results['pure_aa']['element_ids']

    # Assignment agreement
    assigned_mask_current = ids_current >= 0
    assigned_mask_pure_aa = ids_pure_aa >= 0

    both_assigned = assigned_mask_current & assigned_mask_pure_aa
    only_current = assigned_mask_current & (~assigned_mask_pure_aa)
    only_pure_aa = (~assigned_mask_current) & assigned_mask_pure_aa

    n_both = int(jnp.sum(both_assigned))
    n_only_current = int(jnp.sum(only_current))
    n_only_pure_aa = int(jnp.sum(only_pure_aa))

    print(f"\nAssignment Status:")
    print(f"  Both assigned:        {n_both:,} particles")
    print(f"  Only current:         {n_only_current:,} particles")
    print(f"  Only pure_aa:         {n_only_pure_aa:,} particles  {'❌ FALSE POSITIVES!' if n_only_pure_aa > 0 else '✅'}")

    # For particles assigned by both, check if they agree on element ID
    if n_both > 0:
        same_element = ids_current[both_assigned] == ids_pure_aa[both_assigned]
        n_agree = int(jnp.sum(same_element))
        n_disagree = n_both - n_agree

        print(f"\nFor {n_both:,} particles assigned by both:")
        print(f"  Same element:         {n_agree:,} particles ({100*n_agree/n_both:.2f}%)")
        print(f"  Different element:    {n_disagree:,} particles ({100*n_disagree/n_both:.2f}%)  {'❌ WRONG!' if n_disagree > 0 else '✅'}")

    # Overall accuracy
    print(f"\n" + "=" * 80)
    print("VERDICT:")
    print("=" * 80)

    if n_only_pure_aa > 0:
        print(f"❌ FAILED: pure_aa assigned {n_only_pure_aa:,} particles that current didn't")
        print(f"   This indicates FALSE POSITIVES - pure_aa is accepting particles outside elements")
        print(f"   Root cause: Likely tolerance issue or algorithm bug")
    elif n_both > 0:
        disagree_pct = 100 * n_disagree / n_both
        if disagree_pct > 1.0:
            print(f"❌ FAILED: pure_aa disagrees on {disagree_pct:.1f}% of element assignments")
            print(f"   Root cause: Algorithm incorrectly computing barycentric coordinates")
        elif disagree_pct > 0.1:
            print(f"⚠️  WARNING: pure_aa disagrees on {disagree_pct:.2f}% of element assignments")
            print(f"   May be acceptable for production (boundary cases)")
        else:
            speedup = results['current']['time'] / results['pure_aa']['time']
            print(f"✅ PASSED: pure_aa matches current with <0.1% disagreement")
            print(f"   Speedup: {speedup:.2f}×")
            print(f"   Safe for production use")
    else:
        print(f"❌ FAILED: No particles were assigned by both methods")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
