#!/usr/bin/env python3
"""
Diagnose presence and characteristics of degenerate elements in the mesh.

This script analyzes the mesh for elements with problematic geometric properties:
1. Zero or near-zero volume
2. Negative volume (inverted elements)
3. Poor aspect ratio
4. Collapsed edges

Focus areas:
- The 1,826 non-Kuhn elements (only 2 axis-aligned edges)
- The 344 elements using fallback registration (1 cell instead of 4)
- Elements in "coarse blocks" where particle loss occurs
"""

import numpy as np
import time
from pathlib import Path
from collections import defaultdict

# Import mesh loading (same pattern as benchmark)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

# Import octree extraction
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi

# Configuration (same as benchmark)
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'


def compute_tet_volume(vertices):
    """
    Compute signed volume of tetrahedron.

    Volume = (1/6) * |det([v1-v0, v2-v0, v3-v0])|

    Returns:
        volume: Signed volume (negative if inverted)
    """
    v0, v1, v2, v3 = vertices

    # Edge vectors from v0
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0

    # Determinant via triple product
    det = np.dot(e1, np.cross(e2, e3))

    volume = det / 6.0

    return volume


def compute_aspect_ratio(vertices):
    """
    Compute aspect ratio of tetrahedron.

    Aspect ratio = (longest edge) / (shortest altitude)

    Higher values indicate more elongated/flattened elements.
    """
    # Compute all 6 edge lengths
    edges = []
    for i in range(4):
        for j in range(i+1, 4):
            edge_len = np.linalg.norm(vertices[j] - vertices[i])
            edges.append(edge_len)

    longest_edge = max(edges)
    shortest_edge = min(edges)

    # Compute volume-based altitude approximation
    volume = abs(compute_tet_volume(vertices))

    if volume < 1e-15:
        return np.inf

    # Approximate shortest altitude from volume
    # Volume = (1/3) * base_area * height
    # Use longest edge as base approximation
    base_area_approx = longest_edge**2 * np.sqrt(3) / 4  # Equilateral triangle

    if base_area_approx < 1e-15:
        return np.inf

    altitude_approx = 3 * volume / base_area_approx

    if altitude_approx < 1e-15:
        return np.inf

    aspect_ratio = longest_edge / altitude_approx

    return aspect_ratio


def compute_edge_collapse_ratio(vertices):
    """
    Check for collapsed edges (nearly coincident vertices).

    Returns:
        min_edge_ratio: (min edge) / (max edge)

    Values close to 0 indicate collapsed edges.
    """
    edges = []
    for i in range(4):
        for j in range(i+1, 4):
            edge_len = np.linalg.norm(vertices[j] - vertices[i])
            edges.append(edge_len)

    min_edge = min(edges)
    max_edge = max(edges)

    if max_edge < 1e-15:
        return 0.0

    return min_edge / max_edge


def analyze_element_quality(connectivity, node_positions, elem_id):
    """Comprehensive quality analysis for a single element."""
    node_ids = connectivity[elem_id]
    vertices = node_positions[node_ids]

    # Geometric properties
    volume = compute_tet_volume(vertices)
    abs_volume = abs(volume)
    aspect_ratio = compute_aspect_ratio(vertices)
    edge_collapse_ratio = compute_edge_collapse_ratio(vertices)

    # Edge lengths
    edges = []
    for i in range(4):
        for j in range(i+1, 4):
            edge_len = np.linalg.norm(vertices[j] - vertices[i])
            edges.append(edge_len)

    min_edge = min(edges)
    max_edge = max(edges)
    mean_edge = np.mean(edges)

    # Centroid
    centroid = vertices.mean(axis=0)

    # Classification
    is_inverted = volume < 0
    is_near_zero_volume = abs_volume < 1e-10
    is_poor_aspect = aspect_ratio > 100  # Arbitrary threshold
    is_collapsed = edge_collapse_ratio < 0.01  # 1% threshold

    return {
        'elem_id': elem_id,
        'volume': volume,
        'abs_volume': abs_volume,
        'aspect_ratio': aspect_ratio,
        'edge_collapse_ratio': edge_collapse_ratio,
        'min_edge': min_edge,
        'max_edge': max_edge,
        'mean_edge': mean_edge,
        'centroid': centroid,
        'is_inverted': is_inverted,
        'is_near_zero_volume': is_near_zero_volume,
        'is_poor_aspect': is_poor_aspect,
        'is_collapsed': is_collapsed,
        'is_degenerate': is_inverted or is_near_zero_volume or is_poor_aspect or is_collapsed
    }


def main():
    print("="*80)
    print("Degenerate Element Diagnostic")
    print("="*80)
    print()

    # Load mesh (same pattern as benchmark)
    print("[1/5] Loading mesh...")
    t0 = time.time()
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )

    n_nodes_orig = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"    Elements: {n_elements:,}, Nodes: {n_nodes_orig:,}")

    print("  Deduplicating...")
    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    n_nodes = node_positions.shape[0]
    print(f"    Removed {n_duplicates_removed:,} duplicates ({n_nodes:,} nodes remaining)")

    # Extract octree to identify non-Kuhn elements
    print("\n[2/5] Extracting multi-cell octree to identify non-Kuhn elements...")
    t0 = time.time()
    octree_multi = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    print(f"  Extracted in {time.time()-t0:.1f}s")

    # Identify non-Kuhn elements from octree extraction
    # Re-run the extraction logic to track which elements are non-Kuhn
    from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import find_axis_aligned_edges_single

    print("\n[3/5] Identifying non-Kuhn elements...")
    t0 = time.time()

    non_kuhn_elements = []
    kuhn_elements = []

    for elem_id in range(n_elements):
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        cell_size, level = find_axis_aligned_edges_single(vertices, tolerance=1e-6)

        if np.any(cell_size == 0):
            non_kuhn_elements.append(elem_id)
        else:
            kuhn_elements.append(elem_id)

        if (elem_id + 1) % 500000 == 0:
            print(f"    Processed {elem_id + 1:,}/{n_elements:,}...")

    n_non_kuhn = len(non_kuhn_elements)
    n_kuhn = len(kuhn_elements)

    print(f"  Identified in {time.time()-t0:.1f}s")
    print(f"    Kuhn elements: {n_kuhn:,} ({100*n_kuhn/n_elements:.2f}%)")
    print(f"    Non-Kuhn elements: {n_non_kuhn:,} ({100*n_non_kuhn/n_elements:.2f}%)")

    # Analyze quality of ALL elements
    print("\n[4/5] Analyzing element quality (this may take a while)...")
    t0 = time.time()

    all_quality = []

    # Sample strategy: Analyze all non-Kuhn + random sample of Kuhn
    elements_to_analyze = set(non_kuhn_elements)

    # Add random sample of Kuhn elements
    np.random.seed(42)
    n_kuhn_sample = min(10000, n_kuhn)  # Sample 10k Kuhn elements
    kuhn_sample = np.random.choice(kuhn_elements, size=n_kuhn_sample, replace=False)
    elements_to_analyze.update(kuhn_sample)

    print(f"  Analyzing {len(elements_to_analyze):,} elements:")
    print(f"    - All {n_non_kuhn:,} non-Kuhn elements")
    print(f"    - {n_kuhn_sample:,} sampled Kuhn elements")

    for i, elem_id in enumerate(sorted(elements_to_analyze)):
        quality = analyze_element_quality(connectivity, node_positions, elem_id)
        quality['is_non_kuhn'] = elem_id in non_kuhn_elements
        all_quality.append(quality)

        if (i + 1) % 1000 == 0:
            print(f"    Analyzed {i + 1:,}/{len(elements_to_analyze):,}...")

    print(f"  Analyzed in {time.time()-t0:.1f}s")

    # Statistical analysis
    print("\n[5/5] Statistical Analysis")
    print("="*80)

    # Overall statistics
    volumes = np.array([q['abs_volume'] for q in all_quality])
    aspect_ratios = np.array([q['aspect_ratio'] for q in all_quality if np.isfinite(q['aspect_ratio'])])
    edge_collapse_ratios = np.array([q['edge_collapse_ratio'] for q in all_quality])

    n_inverted = sum(1 for q in all_quality if q['is_inverted'])
    n_near_zero_vol = sum(1 for q in all_quality if q['is_near_zero_volume'])
    n_poor_aspect = sum(1 for q in all_quality if q['is_poor_aspect'])
    n_collapsed = sum(1 for q in all_quality if q['is_collapsed'])
    n_degenerate = sum(1 for q in all_quality if q['is_degenerate'])

    print(f"\n  Volume Statistics (absolute):")
    print(f"    Min:    {volumes.min():.6e}")
    print(f"    Max:    {volumes.max():.6e}")
    print(f"    Mean:   {volumes.mean():.6e}")
    print(f"    Median: {np.median(volumes):.6e}")
    print(f"    Std:    {volumes.std():.6e}")

    print(f"\n  Aspect Ratio Statistics:")
    print(f"    Min:    {aspect_ratios.min():.2f}")
    print(f"    Max:    {aspect_ratios.max():.2f}")
    print(f"    Mean:   {aspect_ratios.mean():.2f}")
    print(f"    Median: {np.median(aspect_ratios):.2f}")
    print(f"    Std:    {aspect_ratios.std():.2f}")

    print(f"\n  Edge Collapse Ratio Statistics:")
    print(f"    Min:    {edge_collapse_ratios.min():.6f}")
    print(f"    Max:    {edge_collapse_ratios.max():.6f}")
    print(f"    Mean:   {edge_collapse_ratios.mean():.6f}")
    print(f"    Median: {np.median(edge_collapse_ratios):.6f}")

    print(f"\n  Degenerate Element Counts (of {len(all_quality):,} analyzed):")
    print(f"    Inverted (volume < 0):        {n_inverted:6,} ({100*n_inverted/len(all_quality):5.2f}%)")
    print(f"    Near-zero volume (< 1e-10):   {n_near_zero_vol:6,} ({100*n_near_zero_vol/len(all_quality):5.2f}%)")
    print(f"    Poor aspect ratio (> 100):    {n_poor_aspect:6,} ({100*n_poor_aspect/len(all_quality):5.2f}%)")
    print(f"    Collapsed edges (< 1%):       {n_collapsed:6,} ({100*n_collapsed/len(all_quality):5.2f}%)")
    print(f"    ANY degenerate property:      {n_degenerate:6,} ({100*n_degenerate/len(all_quality):5.2f}%)")

    # Breakdown by Kuhn vs Non-Kuhn
    print(f"\n  Degenerate Elements by Type:")

    kuhn_quality = [q for q in all_quality if not q['is_non_kuhn']]
    non_kuhn_quality = [q for q in all_quality if q['is_non_kuhn']]

    n_kuhn_analyzed = len(kuhn_quality)
    n_non_kuhn_analyzed = len(non_kuhn_quality)

    n_kuhn_degenerate = sum(1 for q in kuhn_quality if q['is_degenerate'])
    n_non_kuhn_degenerate = sum(1 for q in non_kuhn_quality if q['is_degenerate'])

    print(f"    Kuhn elements:     {n_kuhn_degenerate:6,}/{n_kuhn_analyzed:6,} ({100*n_kuhn_degenerate/n_kuhn_analyzed if n_kuhn_analyzed > 0 else 0:5.2f}%)")
    print(f"    Non-Kuhn elements: {n_non_kuhn_degenerate:6,}/{n_non_kuhn_analyzed:6,} ({100*n_non_kuhn_degenerate/n_non_kuhn_analyzed if n_non_kuhn_analyzed > 0 else 0:5.2f}%)")

    # Volume comparison
    if n_kuhn_analyzed > 0 and n_non_kuhn_analyzed > 0:
        kuhn_volumes = np.array([q['abs_volume'] for q in kuhn_quality])
        non_kuhn_volumes = np.array([q['abs_volume'] for q in non_kuhn_quality])

        print(f"\n  Volume Comparison:")
        print(f"    Kuhn mean:     {kuhn_volumes.mean():.6e}")
        print(f"    Non-Kuhn mean: {non_kuhn_volumes.mean():.6e}")
        print(f"    Ratio:         {non_kuhn_volumes.mean() / kuhn_volumes.mean():.3f}x")

    # Aspect ratio comparison
    if n_kuhn_analyzed > 0 and n_non_kuhn_analyzed > 0:
        kuhn_aspects = np.array([q['aspect_ratio'] for q in kuhn_quality if np.isfinite(q['aspect_ratio'])])
        non_kuhn_aspects = np.array([q['aspect_ratio'] for q in non_kuhn_quality if np.isfinite(q['aspect_ratio'])])

        if len(kuhn_aspects) > 0 and len(non_kuhn_aspects) > 0:
            print(f"\n  Aspect Ratio Comparison:")
            print(f"    Kuhn mean:     {kuhn_aspects.mean():.2f}")
            print(f"    Non-Kuhn mean: {non_kuhn_aspects.mean():.2f}")
            print(f"    Ratio:         {non_kuhn_aspects.mean() / kuhn_aspects.mean():.3f}x")

    # Worst offenders
    print(f"\n  Worst 10 Elements (by aspect ratio):")
    worst_by_aspect = sorted(all_quality, key=lambda q: q['aspect_ratio'] if np.isfinite(q['aspect_ratio']) else 0, reverse=True)[:10]

    for i, q in enumerate(worst_by_aspect, 1):
        elem_type = "Non-Kuhn" if q['is_non_kuhn'] else "Kuhn"
        print(f"    {i:2d}. Element {q['elem_id']:7,} ({elem_type:8s}): aspect={q['aspect_ratio']:8.1f}, vol={q['abs_volume']:.3e}, collapse={q['edge_collapse_ratio']:.4f}")

    print(f"\n  Worst 10 Elements (by volume):")
    worst_by_volume = sorted(all_quality, key=lambda q: q['abs_volume'])[:10]

    for i, q in enumerate(worst_by_volume, 1):
        elem_type = "Non-Kuhn" if q['is_non_kuhn'] else "Kuhn"
        print(f"    {i:2d}. Element {q['elem_id']:7,} ({elem_type:8s}): vol={q['abs_volume']:.3e}, aspect={q['aspect_ratio']:8.1f}, collapse={q['edge_collapse_ratio']:.4f}")

    # Spatial distribution of degenerate elements
    print(f"\n  Spatial Distribution of Degenerate Elements:")

    degenerate_centroids = np.array([q['centroid'] for q in all_quality if q['is_degenerate']])

    if len(degenerate_centroids) > 0:
        print(f"    X: min={degenerate_centroids[:,0].min():.6f}, max={degenerate_centroids[:,0].max():.6f}")
        print(f"    Y: min={degenerate_centroids[:,1].min():.6f}, max={degenerate_centroids[:,1].max():.6f}")
        print(f"    Z: min={degenerate_centroids[:,2].min():.6f}, max={degenerate_centroids[:,2].max():.6f}")
    else:
        print(f"    No degenerate elements found!")

    print("\n" + "="*80)
    print("Degenerate Element Diagnostic Complete")
    print("="*80)

    # Summary recommendations
    print("\nRecommendations:")

    if n_degenerate == 0:
        print("  ✅ No degenerate elements detected!")
        print("  → Particle loss is NOT due to degenerate geometry")
    else:
        print(f"  ⚠️  Found {n_degenerate:,} degenerate elements ({100*n_degenerate/len(all_quality):.2f}% of analyzed)")

        if n_inverted > 0:
            print(f"  ⚠️  {n_inverted:,} inverted elements (negative volume) - CRITICAL ISSUE")
            print("     → These will cause point-in-tet failures and incorrect velocity interpolation")

        if n_near_zero_vol > 0:
            print(f"  ⚠️  {n_near_zero_vol:,} near-zero volume elements - NUMERICAL INSTABILITY")
            print("     → These amplify floating-point errors in barycentric coordinate calculation")

        if n_poor_aspect > 0:
            print(f"  ⚠️  {n_poor_aspect:,} high aspect ratio elements - POTENTIAL ISSUE")
            print("     → These may cause numerical instability in interpolation")

        if n_collapsed > 0:
            print(f"  ⚠️  {n_collapsed:,} collapsed edge elements - GEOMETRIC DEGENERACY")
            print("     → Nearly coincident vertices can cause point-in-tet failures")

        if n_non_kuhn_degenerate > 0:
            print(f"\n  🔍 {n_non_kuhn_degenerate:,}/{n_non_kuhn_analyzed:,} non-Kuhn elements are degenerate")
            print("     → These may be the transition elements causing particle loss!")


if __name__ == "__main__":
    main()
