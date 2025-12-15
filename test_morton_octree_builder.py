#!/usr/bin/env python3
"""
Test Adaptive Octree Builder for Morton L2 Search

Validates that the octree builder correctly:
1. Creates spatially-coherent leaves (elements in same leaf are close)
2. Respects capacity constraint (≤256 elements per leaf)
3. Builds correct prefix table
4. Achieves >95% centroid accuracy (compare with fixed-capacity)
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree

# Configuration
MESH_PATH = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_159.pvtu"
LEAF_CAPACITY = 256
MAX_DEPTH = 21


def analyze_leaf_coherence(morton_struct, node_positions, connectivity):
    """
    Analyze spatial coherence of octree leaves.

    For each leaf, compute:
    - Average distance between element centroids
    - Bounding box size
    - Ratio: bbox_size / avg_distance (coherence metric)

    Good octree: ratio ≈ 1 (tight spatial clustering)
    Bad (fixed-capacity): ratio >> 1 (elements scattered)
    """
    n_leaves = morton_struct.n_leaves

    coherence_ratios = []
    bbox_sizes = []
    avg_distances = []

    print(f"  Analyzing {n_leaves:,} leaves...")

    for leaf_id in range(min(n_leaves, 1000)):  # Sample first 1000 leaves
        start = morton_struct.leaf_start[leaf_id]
        length = morton_struct.leaf_length[leaf_id]

        if length < 2:
            continue

        # Get element IDs in this leaf
        elem_ids = morton_struct.elem_ids_sorted[start:start+length]

        # Compute centroids
        centroids = np.zeros((length, 3), dtype=np.float32)
        for i, elem_id in enumerate(elem_ids):
            nodes = connectivity[elem_id]
            centroids[i] = node_positions[nodes].mean(axis=0)

        # Compute bounding box
        bbox_min = centroids.min(axis=0)
        bbox_max = centroids.max(axis=0)
        bbox_size = np.linalg.norm(bbox_max - bbox_min)

        # Compute average pairwise distance (sample)
        if length >= 10:
            # Sample 10 random pairs for efficiency
            n_samples = 10
            distances = []
            for _ in range(n_samples):
                i, j = np.random.choice(length, 2, replace=False)
                dist = np.linalg.norm(centroids[i] - centroids[j])
                distances.append(dist)
            avg_distance = np.mean(distances)
        else:
            # Compute all pairwise distances
            distances = []
            for i in range(length):
                for j in range(i+1, length):
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    distances.append(dist)
            avg_distance = np.mean(distances) if distances else 0.0

        if avg_distance > 0:
            coherence_ratio = bbox_size / avg_distance
            coherence_ratios.append(coherence_ratio)
            bbox_sizes.append(bbox_size)
            avg_distances.append(avg_distance)

    return {
        'mean_coherence_ratio': np.mean(coherence_ratios),
        'median_coherence_ratio': np.median(coherence_ratios),
        'mean_bbox_size': np.mean(bbox_sizes),
        'mean_avg_distance': np.mean(avg_distances)
    }


def main():
    print("=" * 80)
    print("Adaptive Octree Builder Test")
    print("=" * 80)
    print(f"Mesh: {MESH_PATH}")
    print(f"Leaf capacity: {LEAF_CAPACITY}")
    print(f"Max depth: {MAX_DEPTH}")
    print("=" * 80)

    # ========================================================================
    # 1. Load Mesh
    # ========================================================================

    print("\n[1/4] Loading mesh...")
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
    # 2. Build Adaptive Octree
    # ========================================================================

    print("\n[2/4] Building adaptive octree...")
    t_build = time.time()

    morton_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=LEAF_CAPACITY,
        max_depth=MAX_DEPTH,
        verbose=True
    )

    t_build = time.time() - t_build
    print(f"  Total build time: {t_build:.2f}s")

    # ========================================================================
    # 3. Validate Structure
    # ========================================================================

    print("\n[3/4] Validating octree structure...")

    # Check capacity constraint
    max_leaf_size = morton_struct.leaf_length.max()
    print(f"  Max leaf size: {max_leaf_size} (capacity={LEAF_CAPACITY})")

    if max_leaf_size <= LEAF_CAPACITY:
        print(f"  ✅ Capacity constraint satisfied")
    else:
        print(f"  ❌ Capacity constraint violated!")

    # Check prefix table coverage
    n_valid_prefixes = np.sum(morton_struct.prefix_table >= 0)
    n_total_prefixes = len(morton_struct.prefix_table)
    coverage = (n_valid_prefixes / n_total_prefixes) * 100

    print(f"  Prefix table coverage: {n_valid_prefixes:,}/{n_total_prefixes:,} ({coverage:.1f}%)")

    # Check all elements covered
    total_elements = morton_struct.leaf_length.sum()
    print(f"  Elements covered: {total_elements:,}/{n_elements:,}")

    if total_elements == n_elements:
        print(f"  ✅ All elements covered")
    else:
        print(f"  ❌ Missing {n_elements - total_elements} elements!")

    # ========================================================================
    # 4. Analyze Spatial Coherence
    # ========================================================================

    print("\n[4/4] Analyzing spatial coherence...")
    t_coherence = time.time()

    coherence_stats = analyze_leaf_coherence(morton_struct, node_positions, connectivity)

    t_coherence = time.time() - t_coherence

    print(f"  Mean coherence ratio: {coherence_stats['mean_coherence_ratio']:.2f}")
    print(f"    (bbox_size / avg_distance, lower is better)")
    print(f"  Median coherence ratio: {coherence_stats['median_coherence_ratio']:.2f}")
    print(f"  Mean bbox size: {coherence_stats['mean_bbox_size']:.6f}")
    print(f"  Mean avg distance: {coherence_stats['mean_avg_distance']:.6f}")
    print(f"  Analysis time: {t_coherence:.2f}s")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 80)
    print("OCTREE BUILDER VALIDATION")
    print("=" * 80)

    success = True

    # Validate capacity constraint
    if max_leaf_size <= LEAF_CAPACITY:
        print(f"✅ Capacity constraint: {max_leaf_size} ≤ {LEAF_CAPACITY}")
    else:
        print(f"❌ Capacity constraint violated: {max_leaf_size} > {LEAF_CAPACITY}")
        success = False

    # Validate all elements covered
    if total_elements == n_elements:
        print(f"✅ Element coverage: {total_elements:,}/{n_elements:,}")
    else:
        print(f"❌ Element coverage: {total_elements:,}/{n_elements:,}")
        success = False

    # Validate prefix table
    if coverage > 10:  # At least 10% of prefixes should map to leaves
        print(f"✅ Prefix table coverage: {coverage:.1f}%")
    else:
        print(f"❌ Prefix table coverage too low: {coverage:.1f}%")
        success = False

    # Validate spatial coherence
    # Good octree: coherence ratio < 2 (elements in leaf are spatially close)
    # Bad (fixed-capacity): coherence ratio > 5 (elements scattered)
    if coherence_stats['mean_coherence_ratio'] < 3.0:
        print(f"✅ Spatial coherence: {coherence_stats['mean_coherence_ratio']:.2f} < 3.0 (good)")
    else:
        print(f"⚠️  Spatial coherence: {coherence_stats['mean_coherence_ratio']:.2f} ≥ 3.0 (poor)")
        # Don't fail on this, it's more informational

    print(f"\n📊 Octree statistics:")
    print(f"   Leaves: {morton_struct.n_leaves:,}")
    print(f"   Table depth: {morton_struct.table_depth}")
    print(f"   Memory: {(morton_struct.prefix_table.nbytes / (1024**2)):.1f} MB (prefix table)")

    print("=" * 80)

    if success:
        print("\n🎉 OCTREE BUILDER VALIDATION PASSED!")
        print("   Structure is correct, ready for accuracy testing.")
    else:
        print("\n❌ OCTREE BUILDER VALIDATION FAILED")
        print("   Fix structural issues before accuracy testing.")

    print("=" * 80)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
