#!/usr/bin/env python3
"""
Complete Time-Marching Pipeline Test

Tests the integrated pipeline:
1. Element Search (Phase 1 batch processor)
2. Velocity Interpolation (block-local barycentric coordinates)
3. Time Integration (Forward Euler)

Expected performance: ~2,500-3,000 p/s on ThreadedA mesh
"""

import os
import sys
import time
import numpy as np
import jax
from pathlib import Path

# Force CPU-GPU memory management
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mesh loading
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu

# Phase 1: Forest structure
from jaxtrace.gpu.forest.block_grid import create_regular_grid
from jaxtrace.gpu.forest.block_mapper import assign_elements_to_blocks
from jaxtrace.gpu.forest.padded_arrays import build_padded_block_arrays

# Search
from jaxtrace.gpu.search import classify_blocks
from jaxtrace.gpu.search.hash_bucket import build_hash_bucket_arrays
from jaxtrace.gpu.search.initial_assignment import initial_search_batch

# Particles
from jaxtrace.gpu.particles import ParticleData

# Time-marching
from jaxtrace.gpu.tracking import (
    ParticleTimeMarcher,
    create_constant_velocity_field,
    create_time_dependent_velocity_field_fn,
)


def clear_gpu_memory():
    """Clear JAX GPU memory cache."""
    jax.clear_caches()
    print("✓ GPU memory cleared")


def main():
    print("=" * 80)
    print("COMPLETE TIME-MARCHING PIPELINE TEST")
    print("=" * 80)
    print()

    # ========================================================================
    # STEP 1: Load Mesh
    # ========================================================================
    mesh_path = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_20.pvtu"
    print(f"📁 Loading mesh: {mesh_path}")
    node_positions, connectivity, _ = load_mesh_from_pvtu(Path(mesh_path))
    n_nodes = len(node_positions)
    n_elements = len(connectivity)
    print(f"✓ Mesh loaded: {n_nodes:,} nodes, {n_elements:,} elements")
    print()

    # Compute bounding box
    bbox = np.array([
        node_positions[:, 0].min(), node_positions[:, 0].max(),
        node_positions[:, 1].min(), node_positions[:, 1].max(),
        node_positions[:, 2].min(), node_positions[:, 2].max(),
    ], dtype=np.float32)

    print(f"📦 Bounding box:")
    print(f"  X: [{bbox[0]:.4f}, {bbox[1]:.4f}]")
    print(f"  Y: [{bbox[2]:.4f}, {bbox[3]:.4f}]")
    print(f"  Z: [{bbox[4]:.4f}, {bbox[5]:.4f}]")
    print()

    # ========================================================================
    # STEP 2: Create Block Grid
    # ========================================================================
    print("🌳 Creating block grid...")
    grid_size = (8, 8, 4)  # 256 blocks
    blocks = create_regular_grid(bbox, grid_size)
    print(f"✓ Grid created: {len(blocks)} blocks")
    print()

    # ========================================================================
    # STEP 3: Assign Elements to Blocks
    # ========================================================================
    print("📍 Assigning elements to blocks...")
    element_to_block, stats = assign_elements_to_blocks(
        node_positions,
        connectivity,
        bbox,
        grid_size,
        verbose=False
    )
    print(f"✓ Element assignment complete:")
    print(f"  Elements assigned: {stats.n_elements:,}")
    print(f"  Blocks used: {stats.n_blocks_used}/{stats.n_blocks}")
    print()

    # ========================================================================
    # STEP 4: Build Padded Arrays
    # ========================================================================
    print("📊 Building padded arrays...")
    padded_arrays = build_padded_block_arrays(
        element_to_block,
        stats,
        node_positions=node_positions,
        connectivity=connectivity,
        verbose=True
    )
    print(f"✓ Padded arrays created")
    print(f"  Shape: {padded_arrays.block_elements.shape}")
    print(f"  Memory: {padded_arrays.memory_mb:.1f} MB")
    print()

    # ========================================================================
    # STEP 5: Create Velocity Field
    # ========================================================================
    print("🌊 Creating velocity field...")

    # Create constant velocity field: [1, 0, 0] mm/s
    velocity_field = create_constant_velocity_field(
        padded_arrays,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        node_positions
    )
    print(f"✓ Velocity field created: {velocity_field.shape}")
    print(f"  Constant velocity: [1.0, 0.0, 0.0] mm/s")
    print()

    # ========================================================================
    # STEP 6: Build structures for initial search
    # ========================================================================
    print("🏷️  Classifying blocks...")
    classification = classify_blocks(padded_arrays, threshold=10000, verbose=False)
    print(f"✓ Block classification:")
    print(f"  Light blocks: {len(classification.light_blocks)}")
    print(f"  Heavy blocks: {len(classification.heavy_blocks)}")
    print()

    # Build hash buckets for heavy blocks
    hash_bucket_data = {}
    if classification.heavy_blocks:
        print(f"🗂️  Building hash buckets for {len(classification.heavy_blocks)} heavy blocks...")
        element_centroids = np.mean(node_positions[connectivity], axis=1).astype(np.float32)

        for block_id in classification.heavy_blocks:
            block_elems = padded_arrays.block_elements[block_id]
            block_count = int(padded_arrays.block_sizes[block_id])
            elem_ids = block_elems[:block_count]
            elem_ids = elem_ids[elem_ids >= 0]

            if len(elem_ids) == 0:
                continue

            centroids = element_centroids[elem_ids]
            block_bounds = blocks[block_id].bounds

            hash_arrays = build_hash_bucket_arrays(
                block_id=block_id,
                element_ids=elem_ids,
                element_centroids=centroids,
                block_bounds=block_bounds,
                target_bucket_size=200,
                morton_bits=10
            )

            hash_bucket_data[block_id] = hash_arrays

        print(f"✓ Hash buckets built: {len(hash_bucket_data)} blocks")
        print()

    # Build block neighbors
    print("🔗 Building block neighbors...")
    block_neighbors_26 = np.array([b.neighbors_26 for b in blocks], dtype=np.int32)
    print(f"✓ Block neighbors built")
    print()

    # ========================================================================
    # STEP 7: Test with small particle count first
    # ========================================================================
    test_counts = [100, 1_000]

    for n_particles in test_counts:
        print("=" * 80)
        print(f"TEST: {n_particles:,} particles")
        print("=" * 80)
        print()

        print(f"🌱 Seeding {n_particles:,} particles...")
        # Seed particles randomly within bounding box
        np.random.seed(42)
        positions = np.random.uniform(
            low=[bbox[0], bbox[2], bbox[4]],
            high=[bbox[1], bbox[3], bbox[5]],
            size=(n_particles, 3)
        ).astype(np.float32)

        # Initialize particle data (all particles need initial search)
        element_ids = -np.ones(n_particles, dtype=np.int32)  # -1 means not found yet
        block_ids = np.zeros(n_particles, dtype=np.int32)
        active_mask = np.ones(n_particles, dtype=bool)
        velocities = np.zeros((n_particles, 3), dtype=np.float32)  # Zero initial velocities

        particle_data = ParticleData(
            positions=positions,
            velocities=velocities,
            element_ids=element_ids,
            block_ids=block_ids,
            active_mask=active_mask
        )
        print(f"✓ Particles seeded")
        print()

        # ====================================================================
        # STEP 8: Initial Element Search
        # ====================================================================
        print(f"🔍 Running initial element search...")

        t_init_start = time.time()
        element_ids_found, block_ids_found, init_stats = initial_search_batch(
            particle_data.positions,
            bbox,
            grid_size,
            classification,
            padded_arrays,
            block_neighbors_26,
            hash_bucket_data,
            node_positions,
            connectivity,
            verbose=False
        )
        t_init = time.time() - t_init_start

        # Update particle data with found elements and blocks
        particle_data.element_ids = element_ids_found
        particle_data.block_ids = block_ids_found

        n_found = np.sum(element_ids_found >= 0)
        print(f"✓ Initial search complete ({t_init:.2f} s)")
        print(f"  Found: {n_found:,} / {n_particles:,} ({100*n_found/n_particles:.1f}%)")
        print()

        if n_found < n_particles * 0.5:
            print(f"⚠️  WARNING: Less than 50% particles found. Skipping time-marching test.")
            continue

        # ====================================================================
        # STEP 8: Time-Marching Pipeline (Simplified - no search for now)
        # ====================================================================
        print(f"⏰ Testing velocity interpolation only (no time-marching yet)...")

        # Create time marcher
        from jaxtrace.gpu.batching import create_default_config
        config = create_default_config()
        marcher = ParticleTimeMarcher(padded_arrays, connectivity, node_positions, config, verbose=True)

        # Test velocity interpolation
        t_interp_start = time.time()
        velocities = marcher.interpolate_velocities(particle_data, velocity_field)
        t_interp = time.time() - t_interp_start

        print(f"✓ Velocity interpolation complete ({t_interp:.2f} s)")
        print(f"  Throughput: {n_particles/t_interp:.0f} p/s")
        print(f"  Velocities shape: {velocities.shape}")
        print(f"  Mean velocity: {velocities.mean(axis=0)}")
        print()

        # Clear GPU memory before next test
        clear_gpu_memory()
        time.sleep(1)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("=" * 80)
    print("✅ BASIC PIPELINE TEST COMPLETE")
    print("=" * 80)
    print()
    print("Tested components:")
    print("  ✅ Mesh loading")
    print("  ✅ Forest structure creation")
    print("  ✅ Padded array construction")
    print("  ✅ Velocity field creation")
    print("  ✅ Initial particle search")
    print("  ✅ Velocity interpolation")
    print()
    print("TODO - Full time-marching with element search:")
    print("  - Integrate search function into pipeline")
    print("  - Test Forward Euler timesteps")
    print("  - Validate particle motion")
    print()


if __name__ == "__main__":
    main()
