"""
Test vectorized initial assignment performance.

This test validates the vectorized initial_search_batch() implementation
achieves 1,000-5,000 p/s instead of the 7-8 p/s from the Python loop version.
"""

import numpy as np
import time
from pathlib import Path

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

# Utilities
from jaxtrace.gpu.batching import get_gpu_memory_info

def test_vectorized_initial_assignment():
    """Test vectorized initial assignment on ThreadedA mesh."""

    print("\n" + "=" * 80)
    print("VECTORIZED INITIAL ASSIGNMENT TEST")
    print("=" * 80)

    # Load ThreadedA mesh
    mesh_path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_20.pvtu")
    print(f"\nLoading mesh: {mesh_path}")
    node_positions, connectivity, _ = load_mesh_from_pvtu(mesh_path)
    n_elements = len(connectivity)
    n_nodes = len(node_positions)
    print(f"✓ Mesh loaded: {n_elements:,} elements, {n_nodes:,} nodes")

    # Compute bounding box
    bbox = [
        float(node_positions[:, 0].min()), float(node_positions[:, 0].max()),
        float(node_positions[:, 1].min()), float(node_positions[:, 1].max()),
        float(node_positions[:, 2].min()), float(node_positions[:, 2].max()),
    ]
    print(f"✓ Bounding box:")
    print(f"  X: [{bbox[0]:.4f}, {bbox[1]:.4f}]")
    print(f"  Y: [{bbox[2]:.4f}, {bbox[3]:.4f}]")
    print(f"  Z: [{bbox[4]:.4f}, {bbox[5]:.4f}]")

    # Create forest structure
    grid_size = (8, 8, 4)
    print(f"\nCreating forest structure (grid: {grid_size})...")
    blocks, block_neighbors_26 = create_regular_grid(bbox, grid_size)
    n_blocks = len(blocks)
    print(f"✓ Grid created: {n_blocks} blocks")

    # Assign elements to blocks
    print(f"\nAssigning elements to blocks...")
    element_centers = np.mean(node_positions[connectivity], axis=1).astype(np.float32)
    element_to_block = assign_elements_to_blocks(element_centers, blocks)
    print(f"✓ Element assignment complete")

    # Build padded arrays
    print(f"\nBuilding padded arrays...")
    padded_arrays = build_padded_block_arrays(
        n_blocks,
        n_elements,
        element_to_block,
        verbose=True
    )
    print(f"✓ Padded arrays: shape {padded_arrays.block_elements.shape}")

    # Build element neighbors
    print(f"\nBuilding element adjacency...")
    start = time.time()
    element_neighbors = build_element_adjacency(connectivity, verbose=True)
    duration = time.time() - start
    print(f"✓ Adjacency complete ({duration:.2f} s)")

    # Classify blocks
    print(f"\nClassifying blocks (threshold: 10K elements)...")
    classification = classify_blocks(padded_arrays, threshold=10000, verbose=False)
    print(f"✓ Light blocks: {len(classification.light_blocks)}")
    print(f"✓ Heavy blocks: {len(classification.heavy_blocks)}")

    # Build hash buckets for heavy blocks
    hash_bucket_data = {}
    if classification.heavy_blocks:
        print(f"\nBuilding hash buckets for {len(classification.heavy_blocks)} heavy blocks...")

        start = time.time()
        for block_id in classification.heavy_blocks:
            # Get elements in this block
            block_elems = padded_arrays.block_elements[block_id, :padded_arrays.block_sizes[block_id]]
            block = blocks[block_id]

            hash_bucket_data[block_id] = build_hash_bucket_arrays(
                block_elems,
                element_centers,
                node_positions,
                connectivity,
                block['bounds'],
                n_buckets=(8, 8, 8),
                verbose=False
            )
        duration = time.time() - start
        print(f"✓ Hash buckets built: {len(hash_bucket_data)} in {duration:.2f} s")

    # Domain bounds
    domain_bounds = np.array([
        bbox[0], bbox[1],  # xmin, xmax
        bbox[2], bbox[3],  # ymin, ymax
        bbox[4], bbox[5],  # zmin, zmax
    ], dtype=np.float32)

    # Run scaling tests
    print("\n" + "=" * 80)
    print("SCALING TESTS")
    print("=" * 80)

    particle_counts = [1000, 10000, 50000]

    results = []

    for n_particles in particle_counts:
        print(f"\n{'=' * 80}")
        print(f"TEST: {n_particles:,} particles")
        print(f"{'=' * 80}")

        # Seed particles randomly in domain
        print(f"\nSeeding {n_particles:,} particles...")
        rng = np.random.RandomState(42)
        particle_positions = np.zeros((n_particles, 3), dtype=np.float32)
        particle_positions[:, 0] = rng.uniform(bbox[0], bbox[1], n_particles)
        particle_positions[:, 1] = rng.uniform(bbox[2], bbox[3], n_particles)
        particle_positions[:, 2] = rng.uniform(bbox[4], bbox[5], n_particles)
        print(f"✓ Seeded {n_particles:,} particles")

        # Get memory before
        mem_info = get_gpu_memory_info()
        vram_start_mb = mem_info.used_mb

        # Run initial assignment
        print(f"\nRunning vectorized initial assignment...")
        start = time.time()

        element_ids, block_ids, stats = initial_search_batch(
            particle_positions,
            domain_bounds,
            grid_size,
            classification,
            padded_arrays,
            block_neighbors_26,
            hash_bucket_data,
            node_positions,
            connectivity,
            verbose=True
        )

        duration = time.time() - start

        # Get memory after
        mem_info = get_gpu_memory_info()
        vram_end_mb = mem_info.used_mb
        vram_delta = vram_end_mb - vram_start_mb

        # Results
        n_found = np.sum(element_ids >= 0)
        throughput = n_particles / duration

        print(f"\n{'=' * 80}")
        print(f"RESULTS: {n_particles:,} particles")
        print(f"{'=' * 80}")
        print(f"\n⚡ THROUGHPUT:")
        print(f"      {throughput:.0f} p/s  ({duration:.2f} s total)")

        # Check if meets targets
        if throughput >= 1000:
            print(f"  ✅ MEETS target (>1,000 p/s)")
        else:
            print(f"  ⚠️  Below target (>1,000 p/s)")

        print(f"\n💾 MEMORY:")
        print(f"  Start:    {vram_start_mb:6.1f} MB")
        print(f"  Peak:     {vram_end_mb:6.1f} MB")
        print(f"  Delta:   +{vram_delta:6.1f} MB")

        print(f"\n🔍 SEARCH RESULTS:")
        print(f"  L2 (primary block):    {stats.l2_hits:8,} ({100*stats.l2_hits/n_particles:5.1f}%)")
        print(f"  L3 (neighbor blocks):  {stats.l3_hits:8,} ({100*stats.l3_hits/n_particles:5.1f}%)")
        print(f"  Total found:           {n_found:8,} ({100*n_found/n_particles:5.1f}%)")
        print(f"  Not found:             {stats.n_not_found:8,} ({100*stats.n_not_found/n_particles:5.1f}%)")

        results.append({
            'n_particles': n_particles,
            'throughput': throughput,
            'duration': duration,
            'vram_delta': vram_delta,
            'found_rate': 100 * n_found / n_particles
        })

    # Summary table
    print(f"\n{'=' * 80}")
    print("SCALING TEST SUMMARY")
    print(f"{'=' * 80}\n")

    print(f"{'Particles':>12} | {'Throughput':>12} | {'VRAM Δ':>10} | {'Found':>8} | {'Status':>8}")
    print("-" * 70)

    for r in results:
        status = "✅ PASS" if r['throughput'] >= 1000 else "⚠️  SLOW"
        print(f"{r['n_particles']:>12,} | {r['throughput']:>10.0f} p/s | {r['vram_delta']:>8.1f} MB | {r['found_rate']:>6.1f}% | {status}")

    # Performance check
    print(f"\n{'=' * 80}")

    # Check best throughput
    best_throughput = max(r['throughput'] for r in results)

    if best_throughput >= 5000:
        print("✅ EXCELLENT: Exceeds 5,000 p/s target!")
    elif best_throughput >= 1000:
        print("✅ PASS: Meets 1,000 p/s minimum target")
    else:
        print("❌ FAIL: Below 1,000 p/s target")
        print(f"   Expected: 1,000-5,000 p/s")
        print(f"   Achieved: {best_throughput:.0f} p/s")

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    test_vectorized_initial_assignment()
    print("\n✅ Vectorized initial assignment test complete!")
