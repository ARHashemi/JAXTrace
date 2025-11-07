"""
Monitoring & Profiling Module - Phase 4, Task 4.9

Provides performance metrics collection and reporting for multi-level search.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import json
import time


@dataclass
class PerformanceMetrics:
    """
    Detailed performance metrics for multi-level search.

    Attributes:
        per_level_times: Time spent in each level
        per_level_hit_rates: Success rate for each level
        heavy_block_stats: Statistics for heavy block searches
        light_block_stats: Statistics for light block searches
        memory_usage: Memory consumption breakdown
        throughput: Particles processed per second
    """
    per_level_times: Dict[str, float]
    per_level_hit_rates: Dict[str, float]
    heavy_block_stats: Dict[str, any]
    light_block_stats: Dict[str, any]
    memory_usage: Dict[str, float]
    throughput: float


def print_performance_report(
    search_stats,
    block_classification,
    hash_bucket_data=None,
    padded_arrays=None
):
    """
    Print comprehensive performance report.

    Parameters
    ----------
    search_stats : SearchStats
        Statistics from multi_level_search_batch
    block_classification : BlockClassification
        Block classification info
    hash_bucket_data : Dict[int, HashBucketArrays], optional
        Hash bucket data for heavy blocks
    padded_arrays : PaddedArrays, optional
        Phase 2 padded arrays
    """
    print("\n" + "=" * 80)
    print("PHASE 4: MULTI-LEVEL SEARCH PERFORMANCE REPORT")
    print("=" * 80)

    # Overall statistics
    print("\n📊 OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total particles:     {search_stats.n_particles:,}")
    total_found = search_stats.l0_hits + search_stats.l1_hits + search_stats.l2_hits + search_stats.l3_hits
    print(f"Found:               {total_found:,} ({100*total_found/search_stats.n_particles:.1f}%)")
    print(f"Not found:           {search_stats.not_found:,} ({100*search_stats.not_found/search_stats.n_particles:.1f}%)")
    print(f"Total time:          {search_stats.total_time:.2f} s")
    print(f"Throughput:          {search_stats.n_particles/search_stats.total_time:,.0f} particles/s")

    # Per-level performance
    print("\n🎯 PER-LEVEL PERFORMANCE")
    print("-" * 80)
    print(f"{'Level':<20} {'Hits':>10} {'Hit Rate':>12} {'Time (s)':>12} {'Time %':>10}")
    print("-" * 80)

    levels = [
        ("L0: Cached", search_stats.l0_hits, search_stats.l0_time),
        ("L1: Neighbors", search_stats.l1_hits, search_stats.l1_time),
        ("L2: Block", search_stats.l2_hits, search_stats.l2_time),
        ("L3: Neighbor Blocks", search_stats.l3_hits, search_stats.l3_time),
    ]

    for level_name, hits, level_time in levels:
        hit_rate = 100 * hits / search_stats.n_particles if search_stats.n_particles > 0 else 0
        time_pct = 100 * level_time / search_stats.total_time if search_stats.total_time > 0 else 0
        print(f"{level_name:<20} {hits:>10,} {hit_rate:>11.1f}% {level_time:>12.2f} {time_pct:>9.1f}%")

    # Block classification
    print("\n🗂️  BLOCK CLASSIFICATION")
    print("-" * 80)
    n_light = len(block_classification.light_blocks)
    n_heavy = len(block_classification.heavy_blocks)
    total_blocks = n_light + n_heavy
    print(f"Light blocks:        {n_light} ({100*n_light/total_blocks:.1f}%)")
    print(f"Heavy blocks:        {n_heavy} ({100*n_heavy/total_blocks:.1f}%)")
    print(f"Threshold:           {block_classification.threshold:,} elements")

    if n_heavy > 0:
        heavy_stats = block_classification.get_heavy_block_stats()
        print(f"\nHeavy block stats:")
        print(f"  Min elements:      {heavy_stats['min']:,}")
        print(f"  Max elements:      {heavy_stats['max']:,}")
        print(f"  Mean elements:     {heavy_stats['mean']:,.0f}")
        print(f"  Total elements:    {heavy_stats['total']:,}")

    # Hash bucket performance
    if hash_bucket_data:
        print("\n🔢 HASH BUCKET PERFORMANCE")
        print("-" * 80)
        total_buckets = 0
        total_memory = 0.0

        for block_id, hash_arrays in hash_bucket_data.items():
            total_buckets += hash_arrays.n_buckets
            total_memory += hash_arrays.estimate_memory()

        print(f"Heavy blocks with hash buckets: {len(hash_bucket_data)}")
        print(f"Total buckets:                  {total_buckets:,}")
        print(f"Total hash bucket memory:       {total_memory:.1f} MB")

        if len(hash_bucket_data) > 0:
            avg_buckets = total_buckets / len(hash_bucket_data)
            avg_memory = total_memory / len(hash_bucket_data)
            print(f"Avg buckets per heavy block:    {avg_buckets:.0f}")
            print(f"Avg memory per heavy block:     {avg_memory:.1f} MB")

    # Memory summary
    print("\n💾 MEMORY USAGE")
    print("-" * 80)

    memory_total = 0.0

    if padded_arrays:
        padded_mb = padded_arrays.memory_mb
        print(f"Padded arrays (Phase 2):  {padded_mb:.1f} MB")
        memory_total += padded_mb

    if hash_bucket_data:
        hash_mb = sum(h.estimate_memory() for h in hash_bucket_data.values())
        print(f"Hash buckets (Phase 4):   {hash_mb:.1f} MB")
        memory_total += hash_mb

    print(f"{'Total:':25} {memory_total:.1f} MB")

    if memory_total < 500:
        print(f"Budget remaining:         {500 - memory_total:.1f} MB (Target: 500 MB)")
        print("✅ Within memory budget")
    else:
        print(f"⚠️  Over budget by {memory_total - 500:.1f} MB")

    # Performance assessment
    print("\n✨ PERFORMANCE ASSESSMENT")
    print("-" * 80)

    throughput = search_stats.n_particles / search_stats.total_time if search_stats.total_time > 0 else 0

    if throughput > 10000:
        print(f"✅ Excellent: {throughput:,.0f} particles/s (target: >10,000)")
    elif throughput > 5000:
        print(f"⚠️  Good: {throughput:,.0f} particles/s (target: >10,000)")
    else:
        print(f"❌ Below target: {throughput:,.0f} particles/s (target: >10,000)")

    # L0 cache hit rate assessment
    l0_rate = 100 * search_stats.l0_hits / search_stats.n_particles if search_stats.n_particles > 0 else 0
    if l0_rate >= 85:
        print(f"✅ L0 hit rate: {l0_rate:.1f}% (target: 85-95%)")
    else:
        print(f"⚠️  L0 hit rate: {l0_rate:.1f}% (target: 85-95%)")

    # Success rate assessment
    success_rate = 100 * total_found / search_stats.n_particles if search_stats.n_particles > 0 else 0
    if success_rate >= 99:
        print(f"✅ Success rate: {success_rate:.1f}% (target: >99%)")
    else:
        print(f"⚠️  Success rate: {success_rate:.1f}% (target: >99%)")

    print("=" * 80)


def save_performance_log(
    search_stats,
    output_file: str,
    metadata: Dict = None
):
    """
    Save performance statistics to JSON file.

    Parameters
    ----------
    search_stats : SearchStats
        Search statistics
    output_file : str
        Output JSON file path
    metadata : Dict, optional
        Additional metadata to include
    """
    data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_particles': int(search_stats.n_particles),
        'l0_hits': int(search_stats.l0_hits),
        'l1_hits': int(search_stats.l1_hits),
        'l2_hits': int(search_stats.l2_hits),
        'l3_hits': int(search_stats.l3_hits),
        'not_found': int(search_stats.not_found),
        'l0_time': float(search_stats.l0_time),
        'l1_time': float(search_stats.l1_time),
        'l2_time': float(search_stats.l2_time),
        'l3_time': float(search_stats.l3_time),
        'total_time': float(search_stats.total_time),
        'throughput': float(search_stats.n_particles / search_stats.total_time) if search_stats.total_time > 0 else 0,
    }

    if metadata:
        data['metadata'] = metadata

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n💾 Performance log saved to: {output_file}")


if __name__ == "__main__":
    """Test monitoring module."""
    print("Testing Monitoring Module...")

    # Create mock SearchStats
    class MockSearchStats:
        def __init__(self):
            self.n_particles = 10000
            self.l0_hits = 8500
            self.l1_hits = 800
            self.l2_hits = 400
            self.l3_hits = 200
            self.not_found = 100
            self.l0_time = 0.1
            self.l1_time = 0.2
            self.l2_time = 0.5
            self.l3_time = 1.0
            self.total_time = 1.8

    # Create mock classification
    class MockClassification:
        def __init__(self):
            self.light_blocks = list(range(28))
            self.heavy_blocks = [21, 22, 25, 26]
            self.threshold = 10000

        def get_heavy_block_stats(self):
            return {
                'min': 828000,
                'max': 950000,
                'mean': 890000,
                'total': 3560000
            }

    stats = MockSearchStats()
    classification = MockClassification()

    print_performance_report(stats, classification)

    print("\n✅ Monitoring module test complete!")
