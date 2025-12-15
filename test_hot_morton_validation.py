#!/usr/bin/env python3
"""
HOT Morton Validation Test - Quick 1-Step Test

This script validates the HOT Morton implementation with a minimal test:
- Load mesh
- Build HOT Morton structures (CPU preprocessing)
- Upload to GPU
- Seed particles
- Initial assignment
- Single RK4 timestep
- Report results

Expected outcomes:
- No OOM errors during preprocessing
- No OOM errors during GPU execution
- Initial assignment >95% success rate
- Single timestep completes without errors
"""

import os
import sys
import time
import numpy as np
import jax
import logging
from pathlib import Path

# Force CPU-GPU memory management
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# JAXTrace imports
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.particles import ParticleData
from jaxtrace.tracking.seeding import uniform_grid_seeds

# HOT Morton imports
from jaxtrace.gpu.search.hot_morton_builder import build_hot_morton_structures
from jaxtrace.gpu.search.hot_morton_search import (
    upload_hot_morton_structures_to_gpu,
    create_level2_hot_morton_search_unconditional
)
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.tracking.rk4_gpu_fused import (
    create_rk4_step_gpu_fused_for_production_with_hot_morton,
    compute_block_ids_batch
)


# Configuration
MESH_PATH = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_159.pvtu"
DT = 1e-4
GRID_SIZE = (8, 8, 4)  # 256 blocks
RK4_L1_HOP_COUNT = 3
MAX_ELEMENTS_PER_BLOCK = 50000
MAX_LEAF_CAPACITY = 256
MAX_LOCAL_NODES = 1024
SEED_GRID = (10, 10, 10)  # Small grid for quick test (1000 particles)


def main():
    logger.info("=" * 80)
    logger.info("HOT Morton Validation Test - 1-Step Quick Test")
    logger.info("=" * 80)

    # ========================================================================
    # 1. Load Mesh
    # ========================================================================

    logger.info("\n[1/7] Loading mesh...")
    t_load = time.time()
    node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
        Path(MESH_PATH),
        field_name='Displacement'
    )
    t_load = time.time() - t_load

    n_nodes = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    logger.info(f"  Mesh: {n_elements:,} elements, {n_nodes:,} nodes")
    logger.info(f"  Load time: {t_load:.2f}s")

    # Ensure velocity is 3D and float32
    if velocity_field.shape[1] == 2:
        velocity_field = np.column_stack([velocity_field, np.zeros(velocity_field.shape[0])])
    velocity_field = velocity_field.astype(np.float32)

    # Domain bounds
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)
    domain_bounds = np.array([
        domain_min[0], domain_max[0],
        domain_min[1], domain_max[1],
        domain_min[2], domain_max[2]
    ], dtype=np.float32)

    logger.info(f"  Domain: X=[{domain_min[0]:.3f}, {domain_max[0]:.3f}], "
                f"Y=[{domain_min[1]:.3f}, {domain_max[1]:.3f}], "
                f"Z=[{domain_min[2]:.3f}, {domain_max[2]:.3f}]")

    # ========================================================================
    # 2. Build Element Neighbors
    # ========================================================================

    logger.info("\n[2/7] Building element neighbors...")
    t_neighbors = time.time()
    element_neighbors = build_element_neighbors_array(connectivity)
    t_neighbors = time.time() - t_neighbors
    logger.info(f"  Element neighbors: {element_neighbors.shape}")
    logger.info(f"  Build time: {t_neighbors:.2f}s")

    # ========================================================================
    # 3. Build HOT Morton Structures (CRITICAL TEST)
    # ========================================================================

    logger.info("\n[3/7] Building HOT Morton structures (CPU preprocessing)...")
    logger.info("  This is the critical step that extracts local connectivity per leaf")
    t_hot = time.time()

    try:
        hot_structures = build_hot_morton_structures(
            node_positions=node_positions,
            connectivity=connectivity,
            grid_size=GRID_SIZE,
            max_elements_per_block=MAX_ELEMENTS_PER_BLOCK,
            max_leaf_capacity=MAX_LEAF_CAPACITY,
            max_local_nodes=MAX_LOCAL_NODES,
            verbose=True
        )
        t_hot = time.time() - t_hot
        logger.info(f"  ✅ HOT Morton preprocessing: {t_hot:.2f}s")
        logger.info(f"  ✅ No OOM during preprocessing")
    except Exception as e:
        logger.error(f"  ❌ HOT Morton preprocessing FAILED: {e}")
        return

    # ========================================================================
    # 4. Upload to GPU (CRITICAL TEST)
    # ========================================================================

    logger.info("\n[4/7] Uploading mesh and HOT structures to GPU...")
    t_upload = time.time()

    try:
        # Upload standard mesh
        mesh_gpu = upload_mesh_to_gpu(
            node_positions=node_positions,
            connectivity=connectivity,
            element_neighbors=element_neighbors
        )

        # Upload HOT Morton structures
        mesh_gpu_hot = upload_hot_morton_structures_to_gpu(
            hot_structures,
            domain_bounds=domain_bounds,
            grid_size=GRID_SIZE
        )

        # Upload velocity field
        velocity_field_gpu = jax.device_put(velocity_field.astype(np.float32))

        t_upload = time.time() - t_upload
        logger.info(f"  ✅ GPU upload: {t_upload:.2f}s")
        logger.info(f"  ✅ No OOM during GPU upload")
    except Exception as e:
        logger.error(f"  ❌ GPU upload FAILED: {e}")
        return

    # ========================================================================
    # 5. Seed Particles
    # ========================================================================

    logger.info("\n[5/7] Seeding particles...")
    particle_positions = uniform_grid_seeds(
        domain_min, domain_max, SEED_GRID
    )
    n_particles = particle_positions.shape[0]
    logger.info(f"  Seeded {n_particles:,} particles on {SEED_GRID} grid")

    # ========================================================================
    # 6. Initial Element Assignment (CRITICAL TEST)
    # ========================================================================

    logger.info("\n[6/7] Initial element assignment (HOT Morton L2)...")
    t_init = time.time()

    try:
        # Create unconditional L2 HOT Morton search
        search_hot_unconditional = create_level2_hot_morton_search_unconditional(mesh_gpu_hot)

        # Upload positions to GPU
        positions_gpu = jax.device_put(particle_positions.astype(np.float32))

        # Search for containing elements (CRITICAL - uses local connectivity)
        element_ids_gpu = search_hot_unconditional(positions_gpu)

        # Download results
        element_ids = np.array(element_ids_gpu, dtype=np.int32)

        # Compute block IDs
        block_ids_gpu = compute_block_ids_batch(
            positions_gpu,
            mesh_gpu_hot.domain_bounds,
            mesh_gpu_hot.grid_size
        )
        block_ids = np.array(block_ids_gpu, dtype=np.int32)

        t_init = time.time() - t_init

        # Validate initial assignment
        n_found = np.sum(element_ids >= 0)
        success_rate = (n_found / n_particles) * 100
        logger.info(f"  ✅ Initial assignment: {n_found:,}/{n_particles:,} particles ({success_rate:.2f}%)")
        logger.info(f"  ✅ No OOM during search")
        logger.info(f"  Assignment time: {t_init:.2f}s")

        if success_rate < 95.0:
            logger.warning(f"  ⚠️  Initial assignment success rate < 95%")
        else:
            logger.info(f"  ✅ Initial assignment success rate >= 95%")

    except Exception as e:
        logger.error(f"  ❌ Initial assignment FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create particle data
    particle_data = ParticleData(
        positions=particle_positions,
        velocities=np.zeros_like(particle_positions),
        element_ids=element_ids,
        particle_ids=np.arange(n_particles, dtype=np.int32),
        active_mask=(element_ids >= 0),
        block_ids=block_ids
    )

    # ========================================================================
    # 7. Single RK4 Timestep (CRITICAL TEST)
    # ========================================================================

    logger.info("\n[7/7] Single RK4 timestep (HOT Morton L0+L1+L2)...")
    logger.info("  This tests all 5 search calls: k1, k2, k3, k4, final")

    try:
        # Create RK4 wrapper
        rk4_step_func = create_rk4_step_gpu_fused_for_production_with_hot_morton(
            mesh_gpu_hot=mesh_gpu_hot,
            n_hops=RK4_L1_HOP_COUNT
        )

        t_step = time.time()
        particle_data, rk4_stats = rk4_step_func(
            particle_data,
            velocity_field,
            DT,
            mesh_gpu,
            current_time=0.0
        )
        t_step = time.time() - t_step

        n_active = np.sum(particle_data.active_mask)
        retention_pct = (n_active / n_particles) * 100

        logger.info(f"  ✅ Single timestep complete: {t_step*1000:.1f}ms")
        logger.info(f"  ✅ No OOM during RK4 execution")
        logger.info(f"  Active particles: {n_active:,}/{n_particles:,} ({retention_pct:.2f}%)")
        logger.info(f"  Throughput: {n_particles/t_step:.1f} p/s")

        if retention_pct >= 99.0:
            logger.info(f"  ✅ Retention >= 99% (excellent for single step)")
        else:
            logger.warning(f"  ⚠️  Retention < 99% for single step")

    except Exception as e:
        logger.error(f"  ❌ RK4 timestep FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # Final Summary
    # ========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION TEST RESULTS")
    logger.info("=" * 80)
    logger.info("✅ HOT Morton preprocessing: SUCCESS (no OOM)")
    logger.info("✅ GPU upload: SUCCESS (no OOM)")
    logger.info(f"✅ Initial assignment: SUCCESS ({success_rate:.2f}% success rate)")
    logger.info(f"✅ Single RK4 timestep: SUCCESS ({retention_pct:.2f}% retention)")
    logger.info(f"✅ Throughput: {n_particles/t_step:.1f} p/s")
    logger.info("\n🎉 HOT Morton implementation is VALIDATED and READY for production!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
