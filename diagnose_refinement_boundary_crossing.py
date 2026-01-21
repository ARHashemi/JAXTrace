#!/usr/bin/env python3
"""
Diagnose Refinement Boundary Crossing Failures

Critical finding from production test:
- Retention WORSE: 85.67% @ step 100 (vs 82.45% baseline)
- Performance CATASTROPHIC: 1,246 p/s (vs 6,500 p/s baseline, 5× slower!)
- Particle loss at refined/coarse boundaries (node 88456 region)

This diagnostic investigates:
1. Element neighbor construction at refinement boundaries
2. L1 face-based neighbor search completeness
3. Morton octree coverage of boundary elements
4. Spatial gaps in neighbor connectivity

Goal: Identify WHY particles crossing from refined→coarse elements fail search
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import (
    upload_global_morton_to_gpu,
    position_to_leaf_id_octree,
    search_in_leaf_global,
    point_in_tet_gpu
)

# Configuration (same as production)
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 120)
VELOCITY_FIELD_NAME = 'Displacement'

# Focus on problem element (from user observation: element 222615)
PROBLEM_ELEMENT_ID = 222698#222615


def analyze_element_neighbors_at_boundaries(connectivity, node_positions, element_neighbors, element_volumes):
    """
    Analyze neighbor construction quality at refinement boundaries.

    Check:
    1. Do small elements have large neighbors?
    2. Do large elements have small neighbors?
    3. Are face-based neighbors symmetric?
    4. Are there missing neighbors that should exist geometrically?
    """

    print("\n" + "="*80)
    print("ANALYSIS 1: Element Neighbor Construction at Refinement Boundaries")
    print("="*80)

    n_elements = len(connectivity)

    # Debug: Check element volume distribution
    print("\n[1.1] Checking element volume distribution...")
    print(f"  Volume range: [{element_volumes.min():.2e}, {element_volumes.max():.2e}]")
    print(f"  Volume median: {np.median(element_volumes):.2e}")
    print(f"  Unique volumes: {len(np.unique(element_volumes)):,}")

    # Find elements at refinement boundaries (10× size ratio with neighbors)
    print("\n[1.2] Identifying refinement boundary elements...")
    print(f"  Sampling 10,000 elements for speed...")

    boundary_elements = []
    boundary_stats = []

    # Sample elements for speed
    sample_size = min(10000, n_elements)
    sample_indices = np.random.choice(n_elements, size=sample_size, replace=False)

    for i, elem_id in enumerate(sample_indices):
        if i % 2000 == 0:
            print(f"    Checked {i}/{sample_size} elements, found {len(boundary_elements)} boundaries...")

        elem_vol = element_volumes[elem_id]
        neighbors = element_neighbors[elem_id]

        # Get valid neighbors (>= 0)
        valid_neighbors = neighbors[neighbors >= 0]

        if len(valid_neighbors) == 0:
            continue

        neighbor_vols = element_volumes[valid_neighbors]
        max_neighbor_vol = neighbor_vols.max()
        min_neighbor_vol = neighbor_vols.min()

        # Check for refinement boundary (10× size ratio)
        size_ratio_max = max_neighbor_vol / (elem_vol + 1e-20)
        size_ratio_min = elem_vol / (min_neighbor_vol + 1e-20)

        at_boundary = (size_ratio_max > 10.0) or (size_ratio_min > 10.0)

        if at_boundary:
            boundary_elements.append(elem_id)
            boundary_stats.append({
                'elem_id': elem_id,
                'volume': elem_vol,
                'n_neighbors': len(valid_neighbors),
                'ratio_max': size_ratio_max,
                'ratio_min': size_ratio_min,
                'has_large_neighbor': size_ratio_max > 10.0,
                'has_small_neighbor': size_ratio_min > 10.0
            })

    print(f"  Found {len(boundary_elements):,} boundary elements in {sample_size:,} sampled")
    print(f"  Estimated total: {int(len(boundary_elements) * n_elements / sample_size):,} ({100*len(boundary_elements)/sample_size:.2f}%)")

    # Analyze neighbor count distribution at boundaries
    print("\n[1.3] Neighbor count distribution at boundaries...")

    if len(boundary_stats) == 0:
        print("  ⚠️  No refinement boundary elements found!")
        print("  This suggests either:")
        print("    1. All elements are similar size (no refinement)")
        print("    2. Face-based neighbors don't capture size transitions")
        return [], []

    neighbor_counts = [s['n_neighbors'] for s in boundary_stats]
    print(f"  Mean neighbors: {np.mean(neighbor_counts):.2f}")
    print(f"  Min neighbors: {np.min(neighbor_counts)}")
    print(f"  Max neighbors: {np.max(neighbor_counts)}")

    # Count elements with < 4 neighbors (incomplete!)
    incomplete = sum(1 for c in neighbor_counts if c < 4)
    print(f"  Elements with <4 neighbors: {incomplete:,} ({100*incomplete/len(neighbor_counts):.2f}%)")

    # Check symmetry: if A is neighbor of B, is B neighbor of A?
    print("\n[1.4] Checking neighbor symmetry...")

    asymmetric_pairs = []
    for stats in boundary_stats[:1000]:  # Sample first 1000 boundary elements
        elem_id = stats['elem_id']
        neighbors = element_neighbors[elem_id]
        valid_neighbors = neighbors[neighbors >= 0]

        for neighbor_id in valid_neighbors:
            # Check if elem_id is in neighbor's neighbor list
            neighbor_neighbors = element_neighbors[neighbor_id]
            if elem_id not in neighbor_neighbors:
                asymmetric_pairs.append((elem_id, neighbor_id))

    print(f"  Asymmetric pairs found: {len(asymmetric_pairs)}")
    if len(asymmetric_pairs) > 0:
        print(f"  ❌ PROBLEM: Neighbor relation is not symmetric!")
        for i, (a, b) in enumerate(asymmetric_pairs[:5]):
            print(f"    Example {i+1}: elem {a} → elem {b}, but NOT elem {b} → elem {a}")
    else:
        print(f"  ✅ Neighbor relation is symmetric")

    return boundary_elements, boundary_stats


def analyze_l1_search_coverage(connectivity, node_positions, element_neighbors, element_volumes):
    """
    Test L1 search: Given a particle at boundary of a small element,
    can L1 search find the adjacent large element?
    """

    print("\n" + "="*80)
    print("ANALYSIS 2: L1 Face-Based Neighbor Search Coverage")
    print("="*80)

    # Find small→large transitions (refined→coarse)
    print("\n[2.1] Finding small→large element transitions...")
    print("  Sampling 10,000 elements for speed...")

    small_to_large = []

    # Sample elements for speed
    sample_size = min(10000, len(connectivity))
    sample_indices = np.random.choice(len(connectivity), size=sample_size, replace=False)

    for i, elem_id in enumerate(sample_indices):
        if i % 2000 == 0:
            print(f"    Checked {i}/{sample_size} elements, found {len(small_to_large)} transitions...")

        elem_vol = element_volumes[elem_id]
        neighbors = element_neighbors[elem_id]
        valid_neighbors = neighbors[neighbors >= 0]

        if len(valid_neighbors) == 0:
            continue

        neighbor_vols = element_volumes[valid_neighbors]

        # Find neighbors >10× larger
        large_neighbors = valid_neighbors[neighbor_vols > elem_vol * 10.0]

        if len(large_neighbors) > 0:
            small_to_large.append({
                'small_elem': elem_id,
                'small_vol': elem_vol,
                'large_neighbors': large_neighbors,
                'large_vols': element_volumes[large_neighbors]
            })

    print(f"  Found {len(small_to_large):,} small→large transitions in {sample_size:,} sampled")

    if len(small_to_large) == 0:
        print("  ⚠️  No small→large transitions found!")
        print("  This is VERY unexpected - refinement should create size transitions")
        return []

    # For each transition, test if L1 can cross it
    print("\n[2.2] Testing L1 3-hop search across transitions...")

    # Simulate L1 search: start from small element, do 3 hops, can we reach large neighbor?
    l1_failures = []

    for i, trans in enumerate(small_to_large[:100]):  # Sample 100 transitions
        small_elem = trans['small_elem']
        large_neighbors = trans['large_neighbors']

        # BFS search: 3 hops from small element
        visited = set([small_elem])
        frontier = [small_elem]

        for hop in range(3):
            next_frontier = []
            for elem in frontier:
                neighbors = element_neighbors[elem]
                valid_neighbors = neighbors[neighbors >= 0]
                for neighbor in valid_neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier

        # Check if any large neighbor is reachable
        reachable_large = [ln for ln in large_neighbors if ln in visited]

        if len(reachable_large) == 0:
            l1_failures.append({
                'small_elem': small_elem,
                'large_neighbors': large_neighbors,
                'visited': visited,
                'n_visited': len(visited)
            })

    print(f"  L1 failures: {len(l1_failures)}/100 tested transitions")
    if len(l1_failures) > 0:
        print(f"  ❌ PROBLEM: L1 cannot reach large neighbors in 3 hops!")
        for i, fail in enumerate(l1_failures[:3]):
            print(f"    Example {i+1}: elem {fail['small_elem']} → {fail['large_neighbors'][0]}")
            print(f"      Visited {fail['n_visited']} elements in 3 hops, but target not reached")
    else:
        print(f"  ✅ L1 can reach all large neighbors in 3 hops")

    return l1_failures


def analyze_morton_octree_coverage(connectivity, node_positions, element_volumes, mesh_gpu_morton):
    """
    Check Morton octree coverage at refinement boundaries.

    OPTIMIZED VERSION: Uses vectorized JAX operations to run in seconds instead of hours.

    Test:
    1. Are small elements and large neighbors in same or adjacent leaves?
    2. Is Morton distance between boundary elements large?
    3. Do octree boundaries align with mesh refinement boundaries?
    """

    print("\n" + "="*80)
    print("ANALYSIS 3: Morton Octree Coverage at Refinement Boundaries")
    print("="*80)

    # For each element, find its Morton leaf
    print("\n[3.1] Computing Morton leaf assignments (VECTORIZED)...")

    n_elements = len(connectivity)
    element_centroids = node_positions[connectivity].mean(axis=1)

    # OPTIMIZATION: Use JAX vmap to process all centroids in parallel
    print(f"  Processing {n_elements:,} elements in batched GPU operations...")

    # Convert to GPU arrays
    centroids_gpu = jax.device_put(element_centroids.astype(np.float32))

    # Vectorized leaf ID computation
    @jax.jit
    def batch_leaf_ids(centroids):
        return jax.vmap(lambda pos: position_to_leaf_id_octree(pos, mesh_gpu_morton))(centroids)

    # Process in batches to avoid OOM
    batch_size = 50000
    n_batches = (n_elements + batch_size - 1) // batch_size

    element_leaf_ids = np.zeros(n_elements, dtype=np.int32)

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_elements)

        batch_centroids = centroids_gpu[start_idx:end_idx]
        batch_leaf_ids = batch_leaf_ids(batch_centroids)
        element_leaf_ids[start_idx:end_idx] = np.array(batch_leaf_ids, dtype=np.int32)

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
            print(f"    Processed {end_idx:,}/{n_elements:,} elements...")

    print(f"  ✅ Computed leaf assignments for {n_elements:,} elements")
    print(f"  Unique leaves used: {len(np.unique(element_leaf_ids)):,}/{mesh_gpu_morton.n_leaves}")

    # Analyze leaf size distribution
    print("\n[3.2] Analyzing leaf size distribution...")

    leaf_elem_counts = np.bincount(element_leaf_ids, minlength=mesh_gpu_morton.n_leaves)

    print(f"  Mean elements per leaf: {leaf_elem_counts.mean():.1f}")
    print(f"  Min elements per leaf: {leaf_elem_counts.min()}")
    print(f"  Max elements per leaf: {leaf_elem_counts.max()}")
    print(f"  Leaves with >256 elements: {np.sum(leaf_elem_counts > 256)}")

    # Find leaves that span large size variation
    print("\n[3.3] Finding leaves with large element size variation...")

    high_variation_leaves = []

    for leaf_id in range(mesh_gpu_morton.n_leaves):
        elems_in_leaf = np.where(element_leaf_ids == leaf_id)[0]

        if len(elems_in_leaf) < 2:
            continue

        vols = element_volumes[elems_in_leaf]
        size_ratio = vols.max() / (vols.min() + 1e-20)

        if size_ratio > 1000:
            high_variation_leaves.append({
                'leaf_id': leaf_id,
                'n_elements': len(elems_in_leaf),
                'size_ratio': size_ratio,
                'vol_min': vols.min(),
                'vol_max': vols.max()
            })

    print(f"  Leaves with >1000× size variation: {len(high_variation_leaves)}")
    if len(high_variation_leaves) > 0:
        print(f"  ⚠️  WARNING: Some leaves contain both tiny and huge elements!")
        for i, leaf in enumerate(high_variation_leaves[:5]):
            print(f"    Leaf {leaf['leaf_id']}: {leaf['n_elements']} elements, {leaf['size_ratio']:.0f}× size ratio")

    return high_variation_leaves, element_leaf_ids


def analyze_single_node_region(node_id, connectivity, node_positions, element_neighbors, element_volumes):
    """
    Analyze region around a single node.

    Returns statistics about elements and neighbors sharing this node.
    """
    # Find elements containing this node
    elements_with_node = np.where(np.any(connectivity == node_id, axis=1))[0]

    if len(elements_with_node) == 0:
        return None

    results = {
        'node_id': node_id,
        'n_elements': len(elements_with_node),
        'element_ids': elements_with_node,
        'volumes': [],
        'neighbor_stats': []
    }

    for elem_id in elements_with_node:
        elem_vol = element_volumes[elem_id]
        char_length = elem_vol ** (1.0/3.0)

        neighbors = element_neighbors[elem_id]
        valid_neighbors = neighbors[neighbors >= 0]

        neighbor_vols = element_volumes[valid_neighbors] if len(valid_neighbors) > 0 else np.array([])
        size_ratios = neighbor_vols / (elem_vol + 1e-20) if len(neighbor_vols) > 0 else np.array([])

        results['volumes'].append(elem_vol)
        results['neighbor_stats'].append({
            'elem_id': elem_id,
            'volume': elem_vol,
            'char_length': char_length,
            'n_neighbors': len(valid_neighbors),
            'neighbor_ids': valid_neighbors,
            'neighbor_volumes': neighbor_vols,
            'size_ratios': size_ratios,
            'has_large_jump': np.any(size_ratios > 10.0) if len(size_ratios) > 0 else False
        })

    return results


def analyze_problem_region(connectivity, node_positions, element_neighbors, element_volumes):
    """
    Analyze specific problem element 222615 and ALL its neighbors.

    Enhanced analysis:
    1. Find element 222615
    2. Find ALL neighbors of element 222615
    3. Find ALL nodes shared by element 222615 and its neighbors
    4. Run detailed analysis (1, 2, 4) for EACH shared node
    """

    print("\n" + "="*80)
    print(f"ANALYSIS 4: Problem Element {PROBLEM_ELEMENT_ID} and All Neighbors")
    print("="*80)

    # Validate element ID
    if PROBLEM_ELEMENT_ID >= len(connectivity) or PROBLEM_ELEMENT_ID < 0:
        print(f"  ❌ Element {PROBLEM_ELEMENT_ID} out of range [0, {len(connectivity)-1}]")
        return

    print(f"\n[4.1] Analyzing element {PROBLEM_ELEMENT_ID}...")

    # Get element info
    elem_vol = element_volumes[PROBLEM_ELEMENT_ID]
    elem_char_length = elem_vol ** (1.0/3.0)
    elem_nodes = connectivity[PROBLEM_ELEMENT_ID]

    print(f"  Volume: {elem_vol:.2e}")
    print(f"  Char length: {elem_char_length:.2e}")
    print(f"  Nodes: {elem_nodes}")

    # Get neighbors
    neighbors = element_neighbors[PROBLEM_ELEMENT_ID]
    valid_neighbors = neighbors[neighbors >= 0]

    print(f"  Direct neighbors: {len(valid_neighbors)}")

    if len(valid_neighbors) == 0:
        print(f"  ⚠️  Element {PROBLEM_ELEMENT_ID} has NO neighbors!")
        return

    # Analyze neighbor sizes
    neighbor_vols = element_volumes[valid_neighbors]
    size_ratios = neighbor_vols / (elem_vol + 1e-20)

    print(f"  Neighbor size ratios: {size_ratios.min():.2f}× to {size_ratios.max():.2f}×")
    print(f"  Neighbor IDs: {valid_neighbors}")

    # Check for refinement boundary
    has_large_jump = np.any(size_ratios > 10.0) or np.any(size_ratios < 0.1)
    if has_large_jump:
        print(f"  ✅ AT REFINEMENT BOUNDARY (>10× size jump detected)")
        large_neighbors = valid_neighbors[size_ratios > 10.0]
        small_neighbors = valid_neighbors[size_ratios < 0.1]
        if len(large_neighbors) > 0:
            print(f"     Large neighbors (>10×): {large_neighbors}")
        if len(small_neighbors) > 0:
            print(f"     Small neighbors (<0.1×): {small_neighbors}")
    else:
        print(f"  ⚠️  NO REFINEMENT BOUNDARY (all neighbors similar size)")

    # Find all nodes shared by element and its neighbors
    print(f"\n[4.2] Finding all shared nodes...")

    all_nodes = set(elem_nodes)
    for neighbor_id in valid_neighbors:
        neighbor_nodes = connectivity[neighbor_id]
        all_nodes.update(neighbor_nodes)

    shared_nodes = sorted(list(all_nodes))
    print(f"  Total unique nodes: {len(shared_nodes)}")
    print(f"  Node IDs: {shared_nodes[:20]}{'...' if len(shared_nodes) > 20 else ''}")

    # Run detailed analysis for each shared node
    print(f"\n[4.3] Detailed analysis for each shared node...")
    print(f"  (Analyzing {len(shared_nodes)} nodes)")

    node_analysis_results = []

    for i, node_id in enumerate(shared_nodes):
        if i > 0:#and i % 10 == 0:
            print(f"    Analyzed {i}/{len(shared_nodes)} nodes...")

        node_result = analyze_single_node_region(
            node_id, connectivity, node_positions, element_neighbors, element_volumes
        )

        if node_result is not None:
            node_analysis_results.append(node_result)

    # Summary of node analysis
    print(f"\n[4.4] Summary of shared node analysis...")

    nodes_with_size_jumps = 0
    max_size_ratio_found = 0.0
    problem_nodes = []

    for result in node_analysis_results:
        for stat in result['neighbor_stats']:
            if len(stat['size_ratios']) > 0:
                max_ratio = stat['size_ratios'].max()
                max_size_ratio_found = max(max_size_ratio_found, max_ratio)

                if stat['has_large_jump']:
                    nodes_with_size_jumps += 1
                    problem_nodes.append({
                        'node_id': result['node_id'],
                        'elem_id': stat['elem_id'],
                        'max_ratio': max_ratio
                    })
                    break  # Count node only once

    print(f"  Nodes at refinement boundaries: {nodes_with_size_jumps}/{len(node_analysis_results)}")
    print(f"  Maximum size ratio found: {max_size_ratio_found:.2f}×")

    if len(problem_nodes) > 0:
        print(f"\n  🔍 PROBLEM NODES (with >10× size jumps):")
        for i, pn in enumerate(problem_nodes[:10]):
            print(f"     Node {pn['node_id']}: max ratio {pn['max_ratio']:.2f}× (element {pn['elem_id']})")
        if len(problem_nodes) > 10:
            print(f"     ... and {len(problem_nodes) - 10} more")
    else:
        print(f"\n  ⚠️  NO PROBLEM NODES FOUND")
        print(f"     This suggests:")
        print(f"       1. Element {PROBLEM_ELEMENT_ID} is NOT at refinement boundary")
        print(f"       2. Face-based neighbors miss the actual size transitions")
        print(f"       3. Particle loss may be due to missing neighbors, not size jumps")

    # Detailed output for first 5 problem nodes
    if len(problem_nodes) > 0:
        print(f"\n[4.5] Detailed breakdown for first 5 problem nodes...")

        for idx, pn in enumerate(problem_nodes[:5]):
            node_id = pn['node_id']
            result = next(r for r in node_analysis_results if r['node_id'] == node_id)

            print(f"\n  Node {node_id}:")
            print(f"    Elements sharing this node: {result['n_elements']}")

            for stat in result['neighbor_stats']:
                if stat['has_large_jump']:
                    print(f"    Element {stat['elem_id']}:")
                    print(f"      Volume: {stat['volume']:.2e}, Char length: {stat['char_length']:.2e}")
                    print(f"      Neighbors: {stat['n_neighbors']}")
                    print(f"      Size ratios: {stat['size_ratios'].min():.2f}× to {stat['size_ratios'].max():.2f}×")

                    # Show which neighbors are large
                    large_mask = stat['size_ratios'] > 10.0
                    if np.any(large_mask):
                        large_neighbor_ids = stat['neighbor_ids'][large_mask]
                        large_ratios = stat['size_ratios'][large_mask]
                        print(f"      Large neighbors: {list(zip(large_neighbor_ids, large_ratios))}")


def main():
    print("="*80)
    print("Refinement Boundary Crossing Diagnostic")
    print("="*80)

    # Load mesh
    print(f"\n[1/4] Loading mesh...")
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )

    n_elements = len(connectivity)
    n_nodes = len(node_positions)

    print(f"  Elements: {n_elements:,}")
    print(f"  Nodes: {n_nodes:,}")

    # Build neighbors
    print(f"\n[2/4] Building element neighbors...")
    t_start = time.time()
    element_neighbors = build_element_neighbors_array(connectivity, method='face')
    t_elapsed = time.time() - t_start
    print(f"  Build time: {t_elapsed:.2f}s")

    # Compute element volumes
    print(f"\n[3/4] Computing element volumes...")
    v0 = node_positions[connectivity[:, 0]]
    v1 = node_positions[connectivity[:, 1]]
    v2 = node_positions[connectivity[:, 2]]
    v3 = node_positions[connectivity[:, 3]]
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0
    cross_e2_e3 = np.cross(e2, e3)
    det = np.sum(e1 * cross_e2_e3, axis=1)
    element_volumes = np.abs(det) / 6.0

    print(f"  Volume range: [{element_volumes.min():.2e}, {element_volumes.max():.2e}]")
    print(f"  Size ratio: {element_volumes.max() / element_volumes.min():.0f}×")

    # Build Morton octree
    print(f"\n[4/4] Building Morton octree...")
    morton_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256,
        max_depth=21,
        verbose=False
    )
    mesh_gpu_morton = upload_global_morton_to_gpu(
        morton_struct,
        connectivity,
        node_positions
    )
    print(f"  Leaves: {morton_struct.n_leaves:,}")

    # Run analyses
    print("\n" + "="*80)
    print("RUNNING DIAGNOSTICS")
    print("="*80)

    boundary_elements, boundary_stats = analyze_element_neighbors_at_boundaries(
        connectivity, node_positions, element_neighbors, element_volumes
    )

    l1_failures = analyze_l1_search_coverage(
        connectivity, node_positions, element_neighbors, element_volumes
    )

    # SKIPPED: Analysis 3 (Morton) is for ALL elements, not specific to element 222615
    # Results from previous run: 4 high-variation leaves
    # Uncomment below if you need to run it again:
    # high_variation_leaves, element_leaf_ids = analyze_morton_octree_coverage(
    #     connectivity, node_positions, element_volumes, mesh_gpu_morton
    # )

    # ENHANCED: Now analyzes element 222615 and ALL its neighbors + shared nodes
    analyze_problem_region(
        connectivity, node_positions, element_neighbors, element_volumes
    )

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)

    print(f"\n1. Refinement boundary elements: {len(boundary_elements):,}")
    print(f"2. L1 3-hop search failures: {len(l1_failures)}/100 tested")
    # print(f"3. High-variation Morton leaves: {len(high_variation_leaves)}")

    if len(l1_failures) > 0:
        print(f"\n❌ CRITICAL: L1 search cannot cross refinement boundaries in 3 hops!")
        print(f"   Root cause: Face-based neighbors insufficient for 10× size jumps")
        print(f"   Solution: Increase L1 hop count OR use node-based neighbors")

    print()


if __name__ == '__main__':
    main()
