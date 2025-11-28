"""
Simple test for vectorized initial assignment - copies setup from test_phase1_batched_threadeda.py
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


def main():
    """Test vectorized initial assignment."""

    print("\n" + "=" * 80)
    print("VECTORIZED INITIAL ASSIGNMENT TEST (SIMPLE)")
    print("=" * 80)

    # Load mesh (copy from test_phase1_batched_threadeda.py lines 162-166)
    mesh_path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_20.pvtu")
    print(f"\n📁 Loading mesh: {mesh_path}")
    node_positions, connectivity, _ = load_mesh_from_pvtu(mesh_path)
    print(f"✓ Mesh loaded:")
    print(f"  Nodes:    {len(node_positions):,}")
    print(f"  Elements: {len(connectivity):,}")

    # Compute bounding box (lines 169-178)
    bbox = np.array([
        node_positions[:, 0].min(), node_positions[:, 0].max(),
        node_positions[:, 1].min(), node_positions[:, 1].max(),
        node_positions[:, 2].min(), node_positions[:, 2].max(),
    ], dtype=np.float32)

    print(f"\n📦 Bounding box:")
    print(f"  X: [{bbox[0]:.4f}, {bbox[1]:.4f}]")
    print(f"  Y: [{bbox[2]:.4f}, {bbox[3]:.4f}]")
    print(f"  Z: [{bbox[4]:.4f}, {bbox[5]:.4f}]")

    # Create block grid (lines 181-183)
    grid_size = (8, 8, 4)
    print(f"\n🌳 Creating forest structure (grid: {grid_size})...")
    blocks = create_regular_grid(bbox, grid_size)
    print(f"✓ Total blocks: {len(blocks)}")

    # Assign elements to blocks (lines 186-200)
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
    print(f"  Elements per block: {stats.min_elements} - {stats.max_elements} (avg: {stats.mean_elements:.1f})")
    print(f"  Imbalance ratio: {stats.imbalance_ratio:.2f}×")
    print(f"  Heavy blocks (>10K): {len(stats.heavy_blocks)}")

    # Build padded arrays (lines 203-215)
    print(f"\n📊 Building padded arrays (V5 extended mode)...")
    padded_arrays = build_padded_block_arrays(
        element_to_block,
        stats,
        node_positions=node_positions,
        connectivity=connectivity,
        verbose=True
    )

    print(f"✓ Padded arrays created:")
    print(f"  Shape: {padded_arrays.block_elements.shape}")
    print(f"  Memory: {padded_arrays.memory_mb:.1f} MB")
    print(f"  Max elements per block: {padded_arrays.max_elements_per_block}")

    # Build element neighbors (lines 218-223)
    print(f"\n🔗 Building element adjacency (face neighbors)...")
    start = time.time()
    element_neighbors = build_element_adjacency(connectivity)
    duration = time.time() - start

    print(f"✓ Adjacency complete ({duration:.2f} s):")
    print(f"  Elements with neighbors: {np.sum(np.any(element_neighbors >= 0, axis=1))}/{len(connectivity)}")

    # Classify blocks (lines 248-257)
    print(f"\n🏷️  Classifying blocks (threshold: 10K elements)...")
    classification = classify_blocks(padded_arrays, threshold=10000, verbose=False)

    print(f"✓ Block classification:")
    print(f"  Light blocks (<{classification.threshold}):  {len(classification.light_blocks)}")
    print(f"  Heavy blocks (≥{classification.threshold}): {len(classification.heavy_blocks)}")

    # Build hash buckets (copy correct API from test_phase1_batched_threadeda.py)
    hash_bucket_data = {}
    if classification.heavy_blocks:
        print(f"\n🗂️  Building hash buckets for {len(classification.heavy_blocks)} heavy blocks...")
        element_centroids = np.mean(node_positions[connectivity], axis=1).astype(np.float32)

        start = time.time()
        for block_id in classification.heavy_blocks:
            # Get block elements (same as working test lines 269-272)
            block_elems = padded_arrays.block_elements[block_id]
            block_count = int(padded_arrays.block_sizes[block_id])
            elem_ids = block_elems[:block_count]
            elem_ids = elem_ids[elem_ids >= 0]

            if len(elem_ids) == 0:
                continue

            centroids = element_centroids[elem_ids]
            block_bounds = blocks[block_id].bounds

            # Correct API (lines 280-287)
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

    # Get block neighbors (extract from Block objects)
    block_neighbors_26 = np.array([b.neighbors_26 for b in blocks], dtype=np.int32)

    # Domain bounds
    domain_bounds = bbox  # Already in correct format

    # Test vectorized initial assignment
    print(f"\n{'='*80}")
    print("TEST: 1,000 particles")
    print(f"{'='*80}")

    n_particles = 1000

    # Seed particles
    print(f"\n🌱 Seeding {n_particles:,} particles...")
    rng = np.random.RandomState(42)
    particle_positions = np.zeros((n_particles, 3), dtype=np.float32)
    particle_positions[:, 0] = rng.uniform(bbox[0], bbox[1], n_particles)
    particle_positions[:, 1] = rng.uniform(bbox[2], bbox[3], n_particles)
    particle_positions[:, 2] = rng.uniform(bbox[4], bbox[5], n_particles)
    print(f"✓ Seeded {n_particles:,} particles")

    # Run initial assignment
    print(f"\n⏱️  Running vectorized initial assignment...")

    mem_info = get_gpu_memory_info()
    vram_start = mem_info.used_mb

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

    mem_info = get_gpu_memory_info()
    vram_end = mem_info.used_mb

    # Results
    n_found = np.sum(element_ids >= 0)
    throughput = n_particles / duration

    print(f"\n{'='*80}")
    print(f"RESULTS: {n_particles:,} particles")
    print(f"{'='*80}")

    print(f"\n⚡ THROUGHPUT:")
    print(f"      {throughput:.0f} p/s  ({duration:.2f} s total)")

    if throughput >= 1000:
        print(f"  ✅ MEETS target (>1,000 p/s)")
        print(f"  🎉 Speedup over old version: {throughput/7.5:.0f}× faster!")
    else:
        print(f"  ⚠️  Below target (>1,000 p/s)")

    print(f"\n💾 MEMORY:")
    print(f"  Start:    {vram_start:6.1f} MB")
    print(f"  Peak:     {vram_end:6.1f} MB")
    print(f"  Delta:   +{vram_end - vram_start:6.1f} MB")

    print(f"\n🔍 SEARCH RESULTS:")
    print(f"  L2 (primary block):    {stats.l2_hits:8,} ({100*stats.l2_hits/n_particles:5.1f}%)")
    print(f"  L3 (neighbor blocks):  {stats.l3_hits:8,} ({100*stats.l3_hits/n_particles:5.1f}%)")
    print(f"  Total found:           {n_found:8,} ({100*n_found/n_particles:5.1f}%)")
    print(f"  Not found:             {stats.n_not_found:8,} ({100*stats.n_not_found/n_particles:5.1f}%)")

    print(f"\n{'='*80}")
    if throughput >= 1000:
        print("✅ VECTORIZED INITIAL ASSIGNMENT TEST PASSED!")
    else:
        print("❌ VECTORIZED INITIAL ASSIGNMENT TEST FAILED")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
