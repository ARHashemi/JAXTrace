#!/usr/bin/env python3
"""
Small Mesh Test: Batch Processor with ThreadedA Mesh

Tests Phase 2 batch processor integration with ThreadedA mesh but reduced particle count.

Test Configuration:
- Mesh: ThreadedA (3.5M elements)
- Particles: 1000 test particles (small for verification)
- Grid: 8x8x4 (256 blocks)
- Focus: Correctness and Phase 2 kernel integration

This test verifies:
1. Mesh loading from pvtu files
2. Phase 1 forest structure creation
3. Phase 2 padded arrays and neighbors
4. Phase 3 particle seeding
5. Batch processor with Phase 2 search kernels
6. Search statistics tracking (L0/L1/L2 hits)
"""

import sys
import time
from pathlib import Path
import numpy as np
import jax.numpy as jnp

print("\n" + "=" * 80)
print("THREADEDA TEST: Batch Processor with 1K Particles")
print("=" * 80)

# Mesh path - using ThreadedA mesh (timestep 20, mesh is refined during first 20 timesteps)
MESH_PATH = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_20.pvtu"

# Test 1: Imports
print("\n[1/6] Testing imports...")
try:
    from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
    from jaxtrace.gpu.forest import (
        create_regular_grid,
        assign_elements_to_blocks,
        build_padded_block_arrays,
    )
    from jaxtrace.gpu.forest.element_neighbors import build_element_adjacency
    from jaxtrace.gpu.particles import ParticleData
    from jaxtrace.gpu.batching import (
        BatchConfig,
        BatchStatistics,
        group_particles_by_block,
    )
    from jaxtrace.gpu.search import (
        search_particles_in_block,
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Load mesh
print("\n[2/6] Loading ThreadedA mesh...")
try:
    print(f"  Path: {MESH_PATH}")
    node_positions, connectivity, _ = load_mesh_from_pvtu(Path(MESH_PATH))

    print(f"✓ Mesh loaded successfully")
    print(f"  Nodes: {len(node_positions):,}")
    print(f"  Elements: {len(connectivity):,}")

    # Compute bounding box
    bbox = np.array([
        node_positions[:, 0].min(), node_positions[:, 0].max(),
        node_positions[:, 1].min(), node_positions[:, 1].max(),
        node_positions[:, 2].min(), node_positions[:, 2].max(),
    ], dtype=np.float32)

    print(f"\n  Bounding box:")
    print(f"    X: [{bbox[0]:.3f}, {bbox[1]:.3f}]")
    print(f"    Y: [{bbox[2]:.3f}, {bbox[3]:.3f}]")
    print(f"    Z: [{bbox[4]:.3f}, {bbox[5]:.3f}]")

except Exception as e:
    print(f"✗ Mesh loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create forest structure (Phase 1)
print("\n[3/6] Creating forest structure (Phase 1)...")
try:
    # Determine grid size based on mesh size
    n_elements = len(connectivity)

    if n_elements < 10000:
        grid_size = (4, 4, 2)  # 32 blocks for small mesh
    elif n_elements < 100000:
        grid_size = (6, 6, 3)  # 108 blocks for medium mesh
    else:
        grid_size = (8, 8, 4)  # 256 blocks for large mesh

    print(f"  Grid size: {grid_size} ({np.prod(grid_size)} blocks)")

    # Create block grid
    blocks = create_regular_grid(bbox, grid_size)
    print(f"✓ Created {len(blocks)} blocks")

    # Assign elements to blocks
    print(f"\n  Assigning elements to blocks...")
    element_to_block, stats = assign_elements_to_blocks(
        node_positions,
        connectivity,
        bbox,
        grid_size,
        verbose=False
    )

    print(f"✓ Element assignment complete")
    print(f"  Elements assigned: {stats.n_elements:,}")
    print(f"  Blocks used: {stats.n_blocks_used}/{stats.n_blocks}")
    print(f"  Elements per block: {stats.min_elements} - {stats.max_elements} (avg: {stats.mean_elements:.1f})")
    print(f"  Heavy blocks (>10K): {len(stats.heavy_blocks)}")

except Exception as e:
    print(f"✗ Forest structure creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create padded arrays (Phase 2)
print("\n[4/6] Creating padded arrays (Phase 2)...")
try:
    # Compute element neighbors
    print(f"  Computing element neighbors...")
    start_time = time.time()
    element_neighbors = build_element_adjacency(connectivity)
    duration = time.time() - start_time

    print(f"✓ Computed neighbors in {duration:.2f}s")

    # Create padded arrays
    print(f"\n  Creating padded arrays...")
    start_time = time.time()
    padded_arrays = build_padded_block_arrays(
        element_to_block=element_to_block,
        stats=stats,
        node_positions=node_positions,
        connectivity=connectivity,
        element_neighbors=element_neighbors,
        verbose=False
    )
    duration = time.time() - start_time

    print(f"✓ Created padded arrays in {duration:.2f}s")
    print(f"  Block sizes: {padded_arrays.block_sizes.min()} - {padded_arrays.block_sizes.max()}")
    print(f"  Max block size: {padded_arrays.max_block_size}")
    print(f"  Total padded elements: {padded_arrays.connectivity.shape[0] * padded_arrays.connectivity.shape[1]:,}")

except Exception as e:
    print(f"✗ Padded arrays creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Seed particles (Phase 3)
print("\n[5/6] Seeding particles (Phase 3)...")
try:
    n_particles = 1000  # 1K particles for first real mesh test

    print(f"  Seeding {n_particles} particles uniformly...")

    # Generate random positions within bounding box
    rng = np.random.RandomState(42)
    bbox_min = np.array([bbox[0], bbox[2], bbox[4]], dtype=np.float32)
    bbox_max = np.array([bbox[1], bbox[3], bbox[5]], dtype=np.float32)
    bbox_size = bbox_max - bbox_min

    random_01 = rng.uniform(0.0, 1.0, (n_particles, 3)).astype(np.float32)
    positions = bbox_min + random_01 * bbox_size

    # Create ParticleData from positions
    particle_data = ParticleData.from_positions(positions)

    print(f"✓ Seeded {particle_data.n_particles} particles")
    print(f"  Active particles: {particle_data.n_active}")

except Exception as e:
    print(f"✗ Particle seeding failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Run batch processor
print("\n[6/6] Running batch processor with Phase 2 kernels...")
try:
    # Initial particle assignment to blocks
    print(f"\n  Step 1: Assigning particles to blocks...")
    from jaxtrace.gpu.search import initial_search_batch

    # Run initial assignment
    stats_initial = initial_search_batch(
        particle_positions=particle_data.positions,
        particle_block_ids=particle_data.block_ids,
        bbox=bbox,
        grid_size=grid_size,
        blocks=blocks,
        padded_arrays=padded_arrays,
        batch_size=1000,
        verbose=True,
    )

    print(f"✓ Initial assignment complete")
    print(f"  Found: {stats_initial.n_found}/{stats_initial.n_particles}")
    print(f"  Not found: {stats_initial.n_not_found}")
    print(f"  Duration: {stats_initial.duration_sec:.3f}s")

    # Update particle data with assignment results
    particle_data.block_ids[:] = stats_initial.block_ids
    particle_data.element_ids[:] = stats_initial.element_ids
    particle_data.active_mask[:] = stats_initial.found_mask

    print(f"\n  Step 2: Grouping particles by block...")
    grouping = group_particles_by_block(particle_data)

    print(f"✓ Grouped particles")
    print(f"  Unique blocks: {len(grouping.block_ids)}")
    print(f"  Particles per block: {[len(indices) for indices in grouping.particle_indices[:5]][:5]}")

    # Process particles block-by-block
    print(f"\n  Step 3: Processing particles with Phase 2 search kernels...")
    start_time = time.time()

    # Create statistics object
    stats = BatchStatistics(
        batch_id=0,
        n_particles=n_particles,
        n_active_blocks=len(grouping.block_ids)
    )

    # Process each block
    for i, (block_id, particle_indices) in enumerate(zip(grouping.block_ids, grouping.particle_indices)):
        if len(particle_indices) == 0:
            continue

        # Get block data
        block_size = padded_arrays.block_sizes[block_id]
        block_positions = jnp.array(particle_data.positions[particle_indices], dtype=jnp.float32)
        block_element_ids = jnp.array(particle_data.element_ids[particle_indices], dtype=jnp.int32)
        block_ids_array = jnp.full(len(particle_indices), block_id, dtype=jnp.int32)
        block_active = jnp.array(particle_data.active_mask[particle_indices], dtype=jnp.bool_)

        block_connectivity = jnp.array(padded_arrays.connectivity[block_id, :block_size], dtype=jnp.int32)
        block_node_positions = jnp.array(padded_arrays.node_positions[block_id], dtype=jnp.float32)
        block_neighbors = jnp.array(padded_arrays.element_neighbors[block_id, :block_size], dtype=jnp.int32)

        # Call Phase 2 search kernel
        result = search_particles_in_block(
            particle_positions=block_positions,
            particle_element_ids=block_element_ids,
            particle_block_ids=block_ids_array,
            particle_active=block_active,
            block_id=block_id,
            block_connectivity=block_connectivity,
            block_node_positions=block_node_positions,
            block_element_neighbors=block_neighbors,
            block_size=block_size
        )

        # Update particle data
        particle_data.element_ids[particle_indices] = np.array(result.new_element_ids)

        # Accumulate statistics
        stats.level0_hits += int(result.level0_hits)
        stats.level1_hits += int(result.level1_hits)
        stats.level2_hits += int(result.level2_hits)
        stats.not_found += int(result.not_found)

    duration = time.time() - start_time
    stats.time_total = duration

    print(f"✓ Batch processing complete in {duration:.3f}s")
    print(f"\n  [BATCH STATISTICS]")
    print(f"    Particles: {stats.n_particles}")
    print(f"    Active blocks: {stats.n_active_blocks}")
    print(f"\n  [SEARCH STATISTICS]")
    print(f"    Level 0 hits (cached): {stats.level0_hits} ({100*stats.level0_hits/stats.n_particles:.1f}%)")
    print(f"    Level 1 hits (neighbors): {stats.level1_hits} ({100*stats.level1_hits/stats.n_particles:.1f}%)")
    print(f"    Level 2 hits (block search): {stats.level2_hits} ({100*stats.level2_hits/stats.n_particles:.1f}%)")
    print(f"    Not found: {stats.not_found} ({100*stats.not_found/stats.n_particles:.1f}%)")

    # Validate statistics
    total_processed = stats.level0_hits + stats.level1_hits + stats.level2_hits + stats.not_found
    assert total_processed == stats.n_particles, f"Stats mismatch: {total_processed} != {stats.n_particles}"

    print(f"\n  [PERFORMANCE]")
    throughput = stats.throughput_particles_per_sec()
    print(f"    Throughput: {throughput:.1f} particles/s")
    print(f"    Duration: {duration:.3f}s")

except Exception as e:
    print(f"✗ Batch processor test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("✅ THREADEDA 1K PARTICLE TEST PASSED")
print("=" * 80)
print("\nTest Summary:")
print(f"  Mesh: {len(connectivity):,} elements, {stats.n_blocks_used} blocks")
print(f"  Particles: {n_particles}")
print(f"  Search hit rates:")
print(f"    L0 (cached): {100*stats.level0_hits/stats.n_particles:.1f}%")
print(f"    L1 (neighbors): {100*stats.level1_hits/stats.n_particles:.1f}%")
print(f"    L2 (block search): {100*stats.level2_hits/stats.n_particles:.1f}%")
print(f"  Throughput: {throughput:.1f} particles/s")
print("\nPhase 2 Integration Status:")
print("  ✓ Mesh loading working")
print("  ✓ Forest structure creation working")
print("  ✓ Padded arrays creation working")
print("  ✓ Particle seeding working")
print("  ✓ Initial assignment working")
print("  ✓ Batch processor with Phase 2 kernels working")
print("  ✓ Search statistics tracking working")
print("\nNext steps:")
print("  - Test on ThreadedA mesh (3.5M elements)")
print("  - Scale to 1K, 10K, 50K, 100K, 200K particles")
print("  - Measure target throughput (>500 p/s)")
print("  - Enable hash bucket optimization for heavy blocks")
print("=" * 80 + "\n")
