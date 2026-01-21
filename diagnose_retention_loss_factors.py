#!/usr/bin/env python3
"""
Diagnose retention loss: Float precision vs Morton locality.

Quick test to identify which factor causes particle loss during tracking.
Uses same imports and structure as production_tracking_fully_fused_timedep.py.
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Same imports as production script
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import (
    upload_global_morton_to_gpu,
    position_to_leaf_id_octree,
    search_in_leaf_global,
    point_in_tet_gpu
)

# Configuration (same as production)
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 120)  # Just load first timestep for diagnostic
VELOCITY_FIELD_NAME = 'Displacement'

# Test parameters
N_TEST_PARTICLES = 1000  # Small test set for quick iteration
PARTICLE_SEED_BOUNDS = (
    np.array([-0.018, -0.014, -0.007], dtype=np.float32),
    np.array([-0.009, 0.014, 0.000], dtype=np.float32)
)


def test_precision_losses(mesh_gpu, mesh_gpu_morton, element_volumes_gpu, particle_positions):
    """
    Test how many particles are lost due to float precision issues.

    Method: For each particle that fails search, check if it's CLOSE to an element
    but rejected due to numerical tolerance.
    """

    print("\n=== Testing Float Precision Losses ===")

    # Run global Morton search for all particles
    @jax.jit
    def search_all(positions):
        def search_one(pos):
            # Position to leaf
            leaf_id = position_to_leaf_id_octree(pos, mesh_gpu_morton)
            # Search in leaf
            elem_id = search_in_leaf_global(pos, leaf_id, mesh_gpu_morton)
            return elem_id

        return jax.vmap(search_one)(positions)

    positions_gpu = jax.device_put(particle_positions)
    element_ids = search_all(positions_gpu)
    element_ids = np.array(element_ids)

    # Identify failed searches
    failed_mask = element_ids == -1
    n_failed = np.sum(failed_mask)

    print(f"  Search results: {N_TEST_PARTICLES - n_failed}/{N_TEST_PARTICLES} found ({100*(1-n_failed/N_TEST_PARTICLES):.2f}%)")

    if n_failed == 0:
        print("  No failures - cannot test precision losses")
        return 0

    # For each failed particle, check if it's CLOSE to an element
    failed_positions = particle_positions[failed_mask]
    connectivity_cpu = np.array(mesh_gpu.connectivity)
    node_positions_cpu = np.array(mesh_gpu.node_positions)
    element_volumes_cpu = np.array(element_volumes_gpu)

    # Test: For each failed position, find nearest element centroid
    print(f"\n  Analyzing {n_failed} failed particles...")
    precision_losses = 0

    # Sample 100 failed particles (for speed)
    sample_size = min(100, n_failed)
    sample_idx = np.random.choice(n_failed, size=sample_size, replace=False)
    sampled_failed_positions = failed_positions[sample_idx]

    # Compute all element centroids
    element_centroids = node_positions_cpu[connectivity_cpu].mean(axis=1)

    for i, pos in enumerate(sampled_failed_positions):
        # Find nearest element
        distances = np.linalg.norm(element_centroids - pos, axis=1)
        nearest_elem_id = np.argmin(distances)
        nearest_dist = distances[nearest_elem_id]

        # Get element characteristic length
        vol = element_volumes_cpu[nearest_elem_id]
        char_length = vol ** (1.0/3.0)

        # If distance < 0.5 * char_length, particle should be inside or very close
        if nearest_dist < 0.5 * char_length:
            # This is likely a precision loss!
            # Test with RELAXED tolerance
            nodes = connectivity_cpu[nearest_elem_id]
            p0, p1, p2, p3 = node_positions_cpu[nodes]

            # Compute barycentric coordinates (same as point_in_tet_gpu)
            v1, v2, v3 = p1 - p0, p2 - p0, p3 - p0
            vp = pos - p0

            det = np.dot(v1, np.cross(v2, v3))
            if abs(det) > 1e-15:
                b1 = np.dot(vp, np.cross(v2, v3)) / det
                b2 = np.dot(v1, np.cross(vp, v3)) / det
                b3 = np.dot(v1, np.cross(v2, vp)) / det
                b0 = 1.0 - b1 - b2 - b3

                # Test with standard tolerance (-1e-6)
                inside_standard = (b0 >= -1e-6) and (b1 >= -1e-6) and (b2 >= -1e-6) and (b3 >= -1e-6)

                # Test with adaptive tolerance (0.01% of char_length as fraction of normalized coords)
                # Adaptive: scale tolerance by element size
                tol_adaptive = -(char_length * 1e-4) / np.linalg.norm(p1 - p0)  # Relative to edge length

                inside_adaptive = (b0 >= tol_adaptive) and (b1 >= tol_adaptive) and (b2 >= tol_adaptive) and (b3 >= tol_adaptive)

                if not inside_standard and inside_adaptive:
                    precision_losses += 1
                    if precision_losses <= 5:  # Print first 5 examples
                        print(f"    Example {precision_losses}: dist={nearest_dist:.2e}, char_length={char_length:.2e}")
                        print(f"      Barycentrics: [{b0:.6f}, {b1:.6f}, {b2:.6f}, {b3:.6f}]")
                        print(f"      Standard tol (-1e-6): REJECT, Adaptive tol ({tol_adaptive:.2e}): ACCEPT")

    # Extrapolate to all failed particles
    precision_loss_rate = precision_losses / sample_size
    total_precision_losses = int(precision_loss_rate * n_failed)

    print(f"\n  Precision losses: {precision_losses}/{sample_size} sampled ({100*precision_loss_rate:.1f}%)")
    print(f"  Estimated total precision losses: {total_precision_losses}/{n_failed} failed particles")
    print(f"  Impact on retention: -{100*total_precision_losses/N_TEST_PARTICLES:.2f}%")

    return total_precision_losses


def main():
    print("="*80)
    print("Retention Loss Factor Diagnostic")
    print("="*80)

    # Load mesh and velocity (same as production script)
    print(f"\n[1/5] Loading velocity sequence...")
    print(f"  Base path: {MESH_BASE_PATH}")
    print(f"  Pattern: {MESH_FILE_PATTERN}")
    print(f"  Timesteps: {VELOCITY_TIMESTEP_RANGE[0]}-{VELOCITY_TIMESTEP_RANGE[1]}")

    t_load = time.time()
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=True
    )
    t_load = time.time() - t_load

    n_elements = len(connectivity)
    n_nodes = len(node_positions)

    print(f"\n  Loaded mesh:")
    print(f"    Elements: {n_elements:,}")
    print(f"    Nodes: {n_nodes:,}")
    print(f"    Load time: {t_load:.2f}s")

    # Build neighbors (same as production)
    print(f"\n[2/5] Building element neighbors...")
    t_neighbors = time.time()
    element_neighbors = build_element_neighbors_array(connectivity, method='face')
    t_neighbors = time.time() - t_neighbors
    print(f"    Build time: {t_neighbors:.2f}s")

    # Upload mesh to GPU (same as production)
    print(f"\n[3/5] Uploading mesh to GPU...")
    mesh_gpu = upload_mesh_to_gpu(
        connectivity=connectivity,
        node_positions=node_positions,
        element_neighbors=element_neighbors,
        verbose=False
    )

    # Compute element volumes (same as production)
    print("  Computing element volumes...")
    t_volumes = time.time()
    v0 = node_positions[connectivity[:, 0]]
    v1 = node_positions[connectivity[:, 1]]
    v2 = node_positions[connectivity[:, 2]]
    v3 = node_positions[connectivity[:, 3]]
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0
    cross_e2_e3 = np.cross(e2, e3)
    det = np.sum(e1 * cross_e2_e3, axis=1)
    element_volumes_cpu = np.abs(det) / 6.0
    element_volumes_gpu = jax.device_put(element_volumes_cpu.astype(np.float32))
    t_volumes = time.time() - t_volumes

    print(f"    Element volumes computed: {len(element_volumes_cpu):,}")
    print(f"    Volume range: [{element_volumes_cpu.min():.2e}, {element_volumes_cpu.max():.2e}]")
    print(f"    Median volume: {np.median(element_volumes_cpu):.2e}")
    print(f"    Size ratio: {element_volumes_cpu.max() / element_volumes_cpu.min():.1f}×")
    print(f"    Computation time: {t_volumes:.2f}s")

    # Build Morton octree (same as production)
    print(f"\n[4/5] Building Morton octree...")
    t_morton = time.time()
    morton_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256,
        max_depth=21,
        verbose=False
    )
    mesh_gpu_morton = upload_global_morton_to_gpu(
        morton_struct,
        connectivity,
        node_positions
    )
    t_morton = time.time() - t_morton

    print(f"    Build time: {t_morton:.2f}s")
    print(f"    Leaves: {morton_struct.n_leaves:,}")
    print(f"    Max depth: {morton_struct.max_depth}")

    # Seed test particles (same region as production)
    print(f"\n[5/5] Seeding {N_TEST_PARTICLES} test particles...")
    print(f"  Seed region: X=[{PARTICLE_SEED_BOUNDS[0][0]:.6f}, {PARTICLE_SEED_BOUNDS[1][0]:.6f}]")
    print(f"               Y=[{PARTICLE_SEED_BOUNDS[0][1]:.6f}, {PARTICLE_SEED_BOUNDS[1][1]:.6f}]")
    print(f"               Z=[{PARTICLE_SEED_BOUNDS[0][2]:.6f}, {PARTICLE_SEED_BOUNDS[1][2]:.6f}]")

    particle_positions = np.random.uniform(
        PARTICLE_SEED_BOUNDS[0],
        PARTICLE_SEED_BOUNDS[1],
        size=(N_TEST_PARTICLES, 3)
    ).astype(np.float32)

    # Run diagnostic
    print("\n" + "="*80)
    print("RUNNING DIAGNOSTICS")
    print("="*80)

    precision_losses = test_precision_losses(mesh_gpu, mesh_gpu_morton, element_volumes_gpu, particle_positions)

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print(f"Precision losses:  ~{precision_losses} particles (~{100*precision_losses/N_TEST_PARTICLES:.1f}%)")
    print()

    if precision_losses > N_TEST_PARTICLES * 0.05:
        print("⚠️  PRECISION IS THE DOMINANT FACTOR")
        print("   Recommendation: Implement adaptive tolerance (4 hours)")
        print("   Expected gain: +5-10% retention")
        print()
        print("   Next steps:")
        print("   1. Implement adaptive tolerance in point_in_tet_gpu")
        print("   2. Test with production script")
        print("   3. If still < 95% retention: Add coordinate normalization")
    elif precision_losses > N_TEST_PARTICLES * 0.02:
        print("⚠️  PRECISION IS A MINOR FACTOR")
        print("   Recommendation: Consider adaptive tolerance (4 hours) as secondary fix")
        print("   Expected gain: +2-5% retention")
    else:
        print("✅ PRECISION IS NOT THE DOMINANT FACTOR")
        print("   Recommendation: Investigate Morton locality or other factors")
        print()
        print("   Next steps:")
        print("   1. Implement enhanced Morton neighbor search (8 hours)")
        print("   2. Test with production script")
        print("   3. If still < 95%: Investigate time-dependent mesh issues")

    print()


if __name__ == '__main__':
    main()
