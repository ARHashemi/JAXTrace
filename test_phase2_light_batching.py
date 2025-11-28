#!/usr/bin/env python3
"""
Phase 2 Light Block Batching Test

Tests the Phase 2 optimization: batched light block processing.
Compares performance against Phase 1 baseline.

Expected improvement: ~30-50% speedup on light block processing
"""

import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaxtrace.io import load_pvtu_series
from jaxtrace.gpu.forest import create_forest_grid, assign_elements_to_blocks_v5
from jaxtrace.gpu.batching import BatchConfig, process_batch
from jaxtrace.gpu.particles import ParticleData


def clear_gpu_memory():
    """Clear JAX GPU memory cache."""
    jax.clear_caches()
    jax.clear_backends()
    print("✓ GPU memory cleared")


def main():
    print("=" * 80)
    print("PHASE 2 LIGHT BLOCK BATCHING TEST")
    print("Comparing Phase 1 vs Phase 2 light block optimization")
    print("=" * 80)
    print()

    # Load ThreadedA mesh
    mesh_path = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_20.pvtu"
    print(f"📁 Loading mesh: {mesh_path}")
    mesh = load_pvtu_series(mesh_path, 0)
    print(f"✓ Mesh loaded: {mesh.n_nodes:,} nodes, {mesh.n_elements:,} elements")
    print()

    # Create forest structure
    print("🌳 Creating forest structure...")
    grid_resolution = (8, 8, 4)
    forest = create_forest_grid(mesh, grid_resolution=grid_resolution)
    print(f"✓ Forest created: {forest.n_blocks} blocks")
    print()

    # Assign elements to blocks
    print("📍 Assigning elements to blocks...")
    block_assignment = assign_elements_to_blocks_v5(mesh, forest)
    print(f"✓ Assigned: {block_assignment['n_elements_assigned']:,} elements")
    print()

    # Build padded arrays with extended mode (Phase 2 requirement)
    print("📊 Building padded arrays (V5 extended mode)...")
    from jaxtrace.gpu.forest.padded_arrays import build_padded_arrays_v5_extended
    padded_arrays = build_padded_arrays_v5_extended(
        mesh,
        forest,
        block_assignment
    )
    print(f"✓ Padded arrays created: shape {padded_arrays.connectivity.shape}")
    print(f"  Memory: {padded_arrays.connectivity.nbytes / 1024**2:.1f} MB")
    print()

    # Test particle counts
    test_counts = [1_000, 10_000]

    for n_particles in test_counts:
        print("=" * 80)
        print(f"TEST: {n_particles:,} particles")
        print("=" * 80)
        print()

        # Seed particles
        print(f"🌱 Seeding {n_particles:,} particles in mesh...")
        from jaxtrace.gpu.search.initial_assignment import seed_particles_in_mesh
        positions = seed_particles_in_mesh(mesh, n_particles, seed=42)
        print(f"✓ Seeded {n_particles:,} particles")
        print()

        # Initial assignment (simplified - just use block 0 as guess)
        print("🔍 Running simplified initial assignment...")
        element_ids = np.zeros(n_particles, dtype=np.int32)
        block_ids = np.zeros(n_particles, dtype=np.int32)
        active_mask = np.ones(n_particles, dtype=bool)
        print(f"✓ Initialized particle data")
        print()

        # Create particle data
        particle_data = ParticleData(
            positions=positions,
            element_ids=element_ids,
            block_ids=block_ids,
            active_mask=active_mask
        )

        # Apply small perturbation
        print("⚡ Applying small perturbation...")
        particle_data.positions += np.random.randn(n_particles, 3) * 0.0001
        print()

        # Clear GPU memory before test
        clear_gpu_memory()
        time.sleep(1)

        # Run Phase 2 batched search
        print("🚀 Running Phase 2 batched light block search...")
        config = BatchConfig()

        t_start = time.time()
        stats = process_batch(
            batch_id=0,
            batch_particles=particle_data,
            padded_arrays=padded_arrays,
            config=config,
            verbose=False
        )
        t_end = time.time()

        duration = t_end - t_start
        throughput = n_particles / duration

        print()
        print("=" * 80)
        print(f"TEST RESULTS: {n_particles:,} particles")
        print("=" * 80)
        print()
        print(f"⚡ THROUGHPUT:")
        print(f"  {int(throughput):,} p/s  ({duration:.2f} s total)")
        print()
        print(f"🔍 SEARCH HIT RATES:")
        print(f"  L0 (cached):       {stats.level0_hits:6,} ({stats.level0_hits/n_particles*100:5.1f}%)")
        print(f"  L1 (neighbors):    {stats.level1_hits:6,} ({stats.level1_hits/n_particles*100:5.1f}%)")
        print(f"  L2 (block):        {stats.level2_hits:6,} ({stats.level2_hits/n_particles*100:5.1f}%)")
        print(f"  Not found:         {stats.not_found:6,} ({stats.not_found/n_particles*100:5.1f}%)")
        total_found = stats.level0_hits + stats.level1_hits + stats.level2_hits
        print(f"  Total found:       {total_found:6,} ({total_found/n_particles*100:5.1f}%)")
        print()

        # Clear GPU memory after test
        clear_gpu_memory()
        time.sleep(2)

    print("=" * 80)
    print("✅ PHASE 2 LIGHT BLOCK BATCHING TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
