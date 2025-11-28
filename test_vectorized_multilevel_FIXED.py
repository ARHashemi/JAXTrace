"""
Test vectorized multi-level search performance with proper GPU memory management.

This test validates the vectorized multi_level_search_batch_vectorized() implementation
achieves 5,000-15,000 p/s compared to the sequential version at 3,428 p/s.

KEY FIX: Clears JAX caches and GPU memory between tests to prevent OOM.
"""

import numpy as np
import time
from pathlib import Path
import jax
import gc

# Mesh loading
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu

# Forest structure
from jaxtrace.gpu.forest.block_grid import create_regular_grid
from jaxtrace.gpu.forest.block_mapper import assign_elements_to_blocks
from jaxtrace.gpu.forest.padded_arrays import build_padded_block_arrays
from jaxtrace.gpu.forest.element_neighbors import build_element_adjacency

# Search infrastructure
from jaxtrace.gpu.search import (
    classify_blocks,
    build_hash_bucket_arrays,
    initial_search_batch,
)
from jaxtrace.gpu.search.multi_level_search import (
    multi_level_search_batch,  # Sequential version
    multi_level_search_batch_vectorized,  # Vectorized version
)

# Utilities
from jaxtrace.gpu.batching import get_gpu_memory_info


def main():
    """Test vectorized multi-level search performance."""

    print("\n" + "=" * 80)
    print("VECTORIZED MULTI-LEVEL SEARCH TEST (FIXED)")
    print("=" * 80)

    # Load ThreadedA mesh
    mesh_path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_20.pvtu")
    print(f"\n📁 Loading mesh: {mesh_path}")
    node_positions, connectivity, _ = load_mesh_from_pvtu(mesh_path)
    print(f"✓ Mesh loaded:")
    print(f"  Nodes:    {len(node_positions):,}")
    print(f"  Elements: {len(connectivity):,}")

    # Compute bounding box
    bbox = np.array([
        node_positions[:, 0].min(), node_positions[:, 0].max(),
        node_positions[:, 1].min(), node_positions[:, 1].max(),
        node_positions[:, 2].min(), node_positions[:, 2].max(),
    ], dtype=np.float32)

    print(f"\n📦 Bounding box:")
    print(f"  X: [{bbox[0]:.4f}, {bbox[1]:.4f}]")
    print(f"  Y: [{bbox[2]:.4f}, {bbox[3]:.4f}]")
    print(f"  Z: [{bbox[4]:.4f}, {bbox[5]:.4f}]")

    # Create block grid
    grid_size = (8, 8, 4)
    print(f"\n🌳 Creating forest structure (grid: {grid_size})...")
    blocks = create_regular_grid(bbox, grid_size)
    print(f"✓ Total blocks: {len(blocks)}")

    # Assign elements to blocks
    print(f"\n📍 Assigning elements to blocks...")
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

    # Build padded arrays
    print(f"\n📊 Building padded arrays (V5 extended mode)...")
    padded_arrays = build_padded_block_arrays(
        element_to_block,
        stats,
        node_positions=node_positions,
        connectivity=connectivity,
        verbose=False
    )

    print(f"✓ Padded arrays created:")
    print(f"  Shape: {padded_arrays.block_elements.shape}")
    print(f"  Memory: {padded_arrays.memory_mb:.1f} MB")

    # Build element neighbors
    print(f"\n🔗 Building element adjacency (face neighbors)...")
    start = time.time()
    element_neighbors = build_element_adjacency(connectivity)
    duration = time.time() - start
    print(f"✓ Adjacency complete ({duration:.2f} s)")

    # Classify blocks
    print(f"\n🏷️  Classifying blocks (threshold: 10K elements)...")
    classification = classify_blocks(padded_arrays, threshold=10000, verbose=False)

    print(f"✓ Block classification:")
    print(f"  Light blocks: {len(classification.light_blocks)}")
    print(f"  Heavy blocks: {len(classification.heavy_blocks)}")

    # Build hash buckets
    hash_bucket_data = {}
    if classification.heavy_blocks:
        print(f"\n🗂️  Building hash buckets for {len(classification.heavy_blocks)} heavy blocks...")
        element_centroids = np.mean(node_positions[connectivity], axis=1).astype(np.float32)

        start = time.time()
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

        duration = time.time() - start
        print(f"✓ Hash buckets built: {len(hash_bucket_data)} in {duration:.2f} s")

    # Get block neighbors
    block_neighbors_26 = np.array([b.neighbors_26 for b in blocks], dtype=np.int32)

    # Run scaling tests comparing sequential vs vectorized
    print(f"\n{'='*80}")
    print("SCALING TESTS: Sequential vs Vectorized")
    print(f"{'='*80}")

    # Conservative particle counts for 4GB GPU
    particle_counts = [1000, 10000, 30000]

    results = []

    for n_particles in particle_counts:
        print(f"\n{'='*80}")
        print(f"TEST: {n_particles:,} particles")
        print(f"{'='*80}")

        # Seed particles
        print(f"\n🌱 Seeding {n_particles:,} particles...")
        rng = np.random.RandomState(42)
        particle_positions = np.zeros((n_particles, 3), dtype=np.float32)
        particle_positions[:, 0] = rng.uniform(bbox[0], bbox[1], n_particles)
        particle_positions[:, 1] = rng.uniform(bbox[2], bbox[3], n_particles)
        particle_positions[:, 2] = rng.uniform(bbox[4], bbox[5], n_particles)
        print(f"✓ Seeded {n_particles:,} particles")

        # Run initial assignment to get cached elements
        print(f"\n🔍 Running initial assignment...")
        cached_element_ids, cached_block_ids, init_stats = initial_search_batch(
            particle_positions,
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
        print(f"✓ Initial assignment complete:")
        print(f"  Found: {init_stats.n_found}/{n_particles} ({100*init_stats.n_found/n_particles:.1f}%)")

        # Apply small perturbation to simulate particle movement
        print(f"\n⚡ Applying small perturbation (0.1mm)...")
        perturbation = rng.uniform(-0.0001, 0.0001, particle_positions.shape).astype(np.float32)
        perturbed_positions = particle_positions + perturbation

        # Clear GPU before tests
        print(f"\n🧹 Clearing GPU memory before tests...")
        jax.clear_caches()
        gc.collect()

        mem_info = get_gpu_memory_info()
        baseline_mem = mem_info.used_mb
        print(f"   Baseline GPU memory: {baseline_mem:.1f} MB")

        # =====================================================================
        # TEST 1: Sequential multi-level search (baseline)
        # =====================================================================
        print(f"\n{'─'*80}")
        print(f"TEST 1: Sequential multi-level search (baseline)")
        print(f"{'─'*80}")

        start = time.time()
        seq_elem_ids, seq_block_ids, seq_stats = multi_level_search_batch(
            perturbed_positions,
            cached_element_ids,
            cached_block_ids,
            classification,
            padded_arrays.block_elements,
            padded_arrays.block_sizes,
            element_neighbors,
            block_neighbors_26,
            hash_bucket_data,
            node_positions,
            connectivity,
            verbose=False
        )
        seq_duration = time.time() - start

        seq_found = np.sum(seq_elem_ids >= 0)
        seq_throughput = n_particles / seq_duration

        print(f"\n📊 Sequential Results:")
        print(f"  Throughput: {seq_throughput:,.0f} p/s")
        print(f"  Duration:   {seq_duration:.2f} s")
        print(f"  Found:      {seq_found:,}/{n_particles} ({100*seq_found/n_particles:.1f}%)")
        print(f"\n  Hit Rates:")
        print(f"    L0: {seq_stats.l0_hits:6,} ({100*seq_stats.l0_hits/n_particles:5.1f}%)")
        print(f"    L1: {seq_stats.l1_hits:6,} ({100*seq_stats.l1_hits/n_particles:5.1f}%)")
        print(f"    L2: {seq_stats.l2_hits:6,} ({100*seq_stats.l2_hits/n_particles:5.1f}%)")
        print(f"    L3: {seq_stats.l3_hits:6,} ({100*seq_stats.l3_hits/n_particles:5.1f}%)")

        # Save sequential results for comparison before cleanup
        seq_elem_ids_copy = seq_elem_ids.copy()

        # Clear GPU memory after sequential test
        print(f"\n🧹 Clearing GPU memory...")
        del seq_elem_ids, seq_block_ids
        jax.clear_caches()
        gc.collect()

        mem_info = get_gpu_memory_info()
        print(f"   GPU memory after cleanup: {mem_info.used_mb:.1f} MB")

        # =====================================================================
        # TEST 2: Vectorized multi-level search
        # =====================================================================
        print(f"\n{'─'*80}")
        print(f"TEST 2: Vectorized multi-level search")
        print(f"{'─'*80}")

        start = time.time()
        vec_elem_ids, vec_block_ids, vec_stats = multi_level_search_batch_vectorized(
            perturbed_positions,
            cached_element_ids,
            cached_block_ids,
            classification,
            padded_arrays.block_elements,
            padded_arrays.block_sizes,
            element_neighbors,
            block_neighbors_26,
            hash_bucket_data,
            node_positions,
            connectivity,
            verbose=False
        )
        vec_duration = time.time() - start

        vec_found = np.sum(vec_elem_ids >= 0)
        vec_throughput = n_particles / vec_duration

        print(f"\n📊 Vectorized Results:")
        print(f"  Throughput: {vec_throughput:,.0f} p/s")
        print(f"  Duration:   {vec_duration:.2f} s")
        print(f"  Found:      {vec_found:,}/{n_particles} ({100*vec_found/n_particles:.1f}%)")
        print(f"\n  Hit Rates:")
        print(f"    L0: {vec_stats.l0_hits:6,} ({100*vec_stats.l0_hits/n_particles:5.1f}%)")
        print(f"    L1: {vec_stats.l1_hits:6,} ({100*vec_stats.l1_hits/n_particles:5.1f}%)")
        print(f"    L2: {vec_stats.l2_hits:6,} ({100*vec_stats.l2_hits/n_particles:5.1f}%)")
        print(f"    L3: {vec_stats.l3_hits:6,} ({100*vec_stats.l3_hits/n_particles:5.1f}%)")

        # =====================================================================
        # COMPARISON
        # =====================================================================
        speedup = vec_throughput / seq_throughput

        print(f"\n{'─'*80}")
        print(f"COMPARISON")
        print(f"{'─'*80}")
        print(f"  Speedup:        {speedup:.2f}×")
        print(f"  Sequential:     {seq_throughput:,.0f} p/s")
        print(f"  Vectorized:     {vec_throughput:,.0f} p/s")

        if speedup >= 1.5:
            print(f"  ✅ Significant speedup achieved!")
        elif speedup >= 1.1:
            print(f"  ✅ Moderate speedup achieved")
        else:
            print(f"  ⚠️  Minimal speedup")

        # Validate correctness
        print(f"\n  Correctness Check:")
        matches = np.sum(vec_elem_ids == seq_elem_ids_copy)
        match_rate = 100 * matches / n_particles
        print(f"    Element ID matches: {matches:,}/{n_particles} ({match_rate:.1f}%)")

        if match_rate >= 99.0:
            print(f"    ✅ Results match (within tolerance)")
        else:
            print(f"    ⚠️  Results differ significantly!")

        results.append({
            'n_particles': n_particles,
            'seq_throughput': seq_throughput,
            'vec_throughput': vec_throughput,
            'speedup': speedup,
            'match_rate': match_rate,
        })

        # Clear GPU memory between tests to prevent OOM
        print(f"\n🧹 Clearing GPU memory between tests...")
        del vec_elem_ids, vec_block_ids, perturbed_positions
        del cached_element_ids, cached_block_ids, particle_positions
        del seq_elem_ids_copy
        jax.clear_caches()
        gc.collect()

        mem_info = get_gpu_memory_info()
        print(f"   GPU memory after cleanup: {mem_info.used_mb:.1f} MB")

    # Summary table
    print(f"\n{'='*80}")
    print("SCALING TEST SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Particles':>12} | {'Sequential':>12} | {'Vectorized':>12} | {'Speedup':>8} | {'Match':>8}")
    print("─" * 75)

    for r in results:
        print(f"{r['n_particles']:>12,} | {r['seq_throughput']:>10.0f} p/s | {r['vec_throughput']:>10.0f} p/s | {r['speedup']:>6.2f}× | {r['match_rate']:>6.1f}%")

    # Final assessment
    print(f"\n{'='*80}")

    best_vec_throughput = max(r['vec_throughput'] for r in results)
    avg_speedup = np.mean([r['speedup'] for r in results])

    if best_vec_throughput >= 5000:
        print("✅ EXCELLENT: Vectorized version exceeds 5,000 p/s target!")
        print(f"   Best throughput: {best_vec_throughput:,.0f} p/s")
        print(f"   Average speedup: {avg_speedup:.2f}×")
    elif avg_speedup >= 1.5:
        print("✅ GOOD: Significant speedup achieved")
        print(f"   Best throughput: {best_vec_throughput:,.0f} p/s")
        print(f"   Average speedup: {avg_speedup:.2f}×")
    else:
        print("⚠️  MARGINAL: Speedup below expectations")
        print(f"   Best throughput: {best_vec_throughput:,.0f} p/s")
        print(f"   Average speedup: {avg_speedup:.2f}×")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
