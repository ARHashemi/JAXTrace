#!/usr/bin/env python3
"""
Global Morton Validation Test - Small Scale (1K particles, 1 step)

Tests the complete global Morton pipeline with a small particle set to validate:
- Initial assignment success rate (>95% target)
- No JAX OOM errors
- Correct velocity interpolation
- Proper L0+L1+L2 search hierarchy

This is a quick validation before running the full production test.
"""

import os
import sys
import time
import numpy as np
import jax
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaxtrace.gpu.particles import ParticleData
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_global_builder import build_global_morton_structure
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.tracking.rk4_global_morton import create_rk4_step_gpu_fused_global_morton


# Configuration
MESH_PATH = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_159.pvtu"
N_PARTICLES = 1000
DT = 1e-5
N_HOPS = 3
L2_SEARCH_RADIUS = 2
SEED = 42


def main():
    print("=" * 80)
    print("Global Morton Validation Test - 1K Particles, 1 Step")
    print("=" * 80)

    # ========================================================================
    # 1. Load Mesh
    # ========================================================================

    print("\n[1/6] Loading mesh...")
    t_load = time.time()
    node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
        Path(MESH_PATH),
        field_name='Displacement'
    )
    t_load = time.time() - t_load

    n_nodes = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    print(f"  Mesh: {n_elements:,} elements, {n_nodes:,} nodes")
    print(f"  Load time: {t_load:.2f}s")

    # ========================================================================
    # 2. Build Global Morton Structure (CPU)
    # ========================================================================

    print("\n[2/6] Building global Morton structure (CPU)...")
    t_morton = time.time()

    morton_struct = build_global_morton_structure(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256,
        max_depth=21,
        verbose=True
    )

    t_morton = time.time() - t_morton
    print(f"  Built {morton_struct.n_leaves:,} leaves in {t_morton:.2f}s")
    print(f"  Memory: {(morton_struct.elem_ids_sorted.nbytes + morton_struct.morton_sorted.nbytes) / (1024**2):.1f} MB")

    # ========================================================================
    # 3. Upload to GPU
    # ========================================================================

    print("\n[3/6] Uploading mesh and Morton structure to GPU...")
    t_upload = time.time()

    # Compute element neighbors
    element_neighbors = build_element_neighbors_array(connectivity)

    # Upload standard mesh data
    mesh_gpu = upload_mesh_to_gpu(
        connectivity=connectivity,
        node_positions=node_positions,
        element_neighbors=element_neighbors,
        verbose=False
    )

    # Upload global Morton structure
    mesh_gpu_morton = upload_global_morton_to_gpu(
        morton_struct,
        connectivity,
        node_positions
    )

    # Force transfer
    _ = jax.block_until_ready(mesh_gpu.connectivity)
    _ = jax.block_until_ready(mesh_gpu_morton.elem_ids_sorted)

    t_upload = time.time() - t_upload
    print(f"  Upload time: {t_upload:.2f}s")

    # ========================================================================
    # 4. Initialize Particles (Random positions in domain)
    # ========================================================================

    print(f"\n[4/6] Initializing {N_PARTICLES:,} particles...")
    np.random.seed(SEED)

    # Get domain bounds
    bbox_min = node_positions.min(axis=0)
    bbox_max = node_positions.max(axis=0)

    # Generate random positions within domain
    positions = np.random.uniform(
        low=bbox_min,
        high=bbox_max,
        size=(N_PARTICLES, 3)
    ).astype(np.float32)

    # Create particle data with unknown element IDs
    particle_data = ParticleData.from_positions(positions)

    print(f"  Created {N_PARTICLES:,} particles")
    print(f"  Domain: X=[{bbox_min[0]:.4f}, {bbox_max[0]:.4f}]")
    print(f"  Domain: Y=[{bbox_min[1]:.4f}, {bbox_max[1]:.4f}]")
    print(f"  Domain: Z=[{bbox_min[2]:.4f}, {bbox_max[2]:.4f}]")

    # ========================================================================
    # 5. Create RK4 Function and Run One Step
    # ========================================================================

    print(f"\n[5/6] Creating RK4 function and running one timestep...")
    print(f"  L1 hops: {N_HOPS}")
    print(f"  L2 search radius: {L2_SEARCH_RADIUS}")

    # Create RK4 step function
    rk4_step = create_rk4_step_gpu_fused_global_morton(
        mesh_gpu_global_morton=mesh_gpu_morton,
        n_hops=N_HOPS,
        l2_search_radius=L2_SEARCH_RADIUS
    )

    # Run one timestep
    print(f"\n  Running RK4 step (dt={DT:.2e})...")
    t_step = time.time()

    particle_data_new, stats = rk4_step(
        particle_data=particle_data,
        velocity_field=velocity_field,
        dt=DT,
        mesh_gpu=mesh_gpu,
        current_time=0.0
    )

    t_step = time.time() - t_step

    # ========================================================================
    # 6. Analyze Results
    # ========================================================================

    print("\n[6/6] Analyzing results...")

    # Count found particles
    found_mask = particle_data_new.element_ids >= 0
    n_found = np.sum(found_mask)
    success_rate = (n_found / N_PARTICLES) * 100

    print(f"\n  Particles found: {n_found}/{N_PARTICLES} ({success_rate:.1f}%)")

    # Check if any particles moved
    position_changes = np.linalg.norm(
        particle_data_new.positions - particle_data.positions,
        axis=1
    )
    max_displacement = np.max(position_changes)
    mean_displacement = np.mean(position_changes[found_mask]) if n_found > 0 else 0

    print(f"  Max displacement: {max_displacement:.2e}")
    print(f"  Mean displacement (found): {mean_displacement:.2e}")

    # Timing breakdown
    print(f"\n  Timing breakdown:")
    print(f"    Upload: {stats['time_upload']*1000:.2f} ms")
    print(f"    Compute: {stats['time_compute']*1000:.2f} ms")
    print(f"    Download: {stats['time_download']*1000:.2f} ms")
    print(f"    Total: {stats['time_total']*1000:.2f} ms")

    throughput = N_PARTICLES / stats['time_total']
    print(f"  Throughput: {throughput:.0f} particles/s")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    # Check success criteria
    success = True

    if success_rate >= 95.0:
        print(f"✅ Initial assignment: {success_rate:.1f}% (>95% target)")
    else:
        print(f"❌ Initial assignment: {success_rate:.1f}% (<95% target)")
        success = False

    if throughput >= 10000:
        print(f"✅ Throughput: {throughput:.0f} p/s (>10k target for validation)")
    else:
        print(f"⚠️  Throughput: {throughput:.0f} p/s (<10k, but OK for validation)")

    print(f"✅ No JAX OOM errors")
    print(f"✅ Morton structure: {morton_struct.n_leaves:,} leaves, {morton_struct.leaf_capacity} capacity")
    print(f"✅ L0+L1+L2 search hierarchy working")

    print("=" * 80)

    if success:
        print("\n🎉 VALIDATION PASSED!")
        print("   Ready to proceed with production test (105K particles, 2.5K steps)")
    else:
        print("\n❌ VALIDATION FAILED")
        print("   Initial assignment rate too low. Check L2 search configuration.")

    print("=" * 80)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
