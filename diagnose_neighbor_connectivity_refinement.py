#!/usr/bin/env python3
"""
Diagnose Neighbor Connectivity Across Refinement Levels

Checks whether face-based neighbors connect coarse and fine elements
in the 1:2 octree refinement structure.

Expected Results:
- Face-based: 0 coarse→fine connections (face-sharing across levels = 0)
- Node-based: Many coarse→fine connections (edge-sharing across levels)
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jaxtrace.gpu.forest import (
    build_element_neighbors_array,
    extract_element_neighbors,
    extract_element_neighbors_node_based,
)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu


def analyze_cross_level_connectivity(connectivity, element_sizes, neighbors_dict, method_name):
    """
    Analyze whether neighbors connect elements across refinement levels.

    Args:
        connectivity: (N, 4) element connectivity
        element_sizes: (N,) characteristic element sizes
        neighbors_dict: {elem_id: neighbor_array} neighbor mapping
        method_name: "face" or "node" for display

    Returns:
        dict with cross-level connectivity statistics
    """
    n_elements = len(connectivity)

    # Classify elements by size
    fine_threshold = 0.15  # mm
    coarse_threshold = 0.30  # mm

    is_fine = element_sizes <= fine_threshold
    is_medium = (element_sizes > fine_threshold) & (element_sizes <= coarse_threshold)
    is_coarse = element_sizes > coarse_threshold

    n_fine = np.sum(is_fine)
    n_medium = np.sum(is_medium)
    n_coarse = np.sum(is_coarse)

    print(f"\n{method_name.upper()} Neighbors Analysis:")
    print(f"  Fine elements (≤0.15mm): {n_fine:,} ({100*n_fine/n_elements:.1f}%)")
    print(f"  Medium elements (0.15-0.30mm): {n_medium:,} ({100*n_medium/n_elements:.1f}%)")
    print(f"  Coarse elements (>0.30mm): {n_coarse:,} ({100*n_coarse/n_elements:.1f}%)")

    # Analyze cross-level connections
    coarse_to_fine = 0
    coarse_to_medium = 0
    medium_to_fine = 0
    fine_to_coarse = 0
    medium_to_coarse = 0
    fine_to_medium = 0

    coarse_with_fine_neighbors = 0
    medium_with_fine_neighbors = 0

    for elem_id in range(n_elements):
        neighbors = neighbors_dict.get(elem_id, np.array([]))
        if len(neighbors) == 0:
            continue

        # Count cross-level connections
        if is_coarse[elem_id]:
            n_fine_neighs = np.sum(is_fine[neighbors])
            n_medium_neighs = np.sum(is_medium[neighbors])
            coarse_to_fine += n_fine_neighs
            coarse_to_medium += n_medium_neighs
            if n_fine_neighs > 0:
                coarse_with_fine_neighbors += 1

        elif is_medium[elem_id]:
            n_fine_neighs = np.sum(is_fine[neighbors])
            n_coarse_neighs = np.sum(is_coarse[neighbors])
            medium_to_fine += n_fine_neighs
            medium_to_coarse += n_coarse_neighs
            if n_fine_neighs > 0:
                medium_with_fine_neighbors += 1

        elif is_fine[elem_id]:
            n_coarse_neighs = np.sum(is_coarse[neighbors])
            n_medium_neighs = np.sum(is_medium[neighbors])
            fine_to_coarse += n_coarse_neighs
            fine_to_medium += n_medium_neighs

    print(f"\n  Cross-Level Connections:")
    print(f"    Coarse → Fine: {coarse_to_fine:,} connections")
    print(f"    Coarse → Medium: {coarse_to_medium:,} connections")
    print(f"    Medium → Fine: {medium_to_fine:,} connections")
    print(f"    Fine → Coarse: {fine_to_coarse:,} connections")
    print(f"    Fine → Medium: {fine_to_medium:,} connections")
    print(f"    Medium → Coarse: {medium_to_coarse:,} connections")

    print(f"\n  Coarse elements with fine neighbors: {coarse_with_fine_neighbors:,} ({100*coarse_with_fine_neighbors/n_coarse:.2f}%)")
    print(f"  Medium elements with fine neighbors: {medium_with_fine_neighbors:,} ({100*medium_with_fine_neighbors/n_medium:.2f}%)")

    # Key metric: Can L1 hop from coarse to fine?
    can_l1_reach_fine = (coarse_to_fine > 0) or (coarse_to_medium > 0 and medium_to_fine > 0)

    print(f"\n  ✓ L1 can reach fine from coarse: {can_l1_reach_fine}")
    if not can_l1_reach_fine:
        print(f"    ❌ {method_name.upper()} neighbors DO NOT connect coarse→fine!")
        print(f"    → L1 search will FAIL to find fine elements")
    else:
        print(f"    ✓ {method_name.upper()} neighbors DO connect coarse→fine")
        print(f"    → L1 search should work")

    return {
        "n_fine": n_fine,
        "n_medium": n_medium,
        "n_coarse": n_coarse,
        "coarse_to_fine": coarse_to_fine,
        "coarse_to_medium": coarse_to_medium,
        "medium_to_fine": medium_to_fine,
        "coarse_with_fine_neighbors": coarse_with_fine_neighbors,
        "medium_with_fine_neighbors": medium_with_fine_neighbors,
        "can_l1_reach_fine": can_l1_reach_fine,
    }


def main():
    print("=" * 80)
    print("NEIGHBOR CONNECTIVITY DIAGNOSIS")
    print("=" * 80)

    # Load mesh
    mesh_path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule")
    mesh_file = "threadedAvtk_120.pvtu"

    print(f"\nLoading mesh: {mesh_path / mesh_file}")
    node_positions, connectivity, _ = load_velocity_sequence_from_pvtu(
        base_path=mesh_path,
        file_pattern="threadedAvtk_{timestep}.pvtu",
        timestep_range=(120, 120),
        field_name='Displacement',
        verbose=False
    )

    n_elements = len(connectivity)
    n_nodes = len(node_positions)
    print(f"  Elements: {n_elements:,}")
    print(f"  Nodes: {n_nodes:,}")

    # Compute element sizes (characteristic length)
    print("\nComputing element sizes...")
    element_centroids = np.mean(node_positions[connectivity], axis=1)

    # Compute characteristic size as max edge length
    def compute_element_size(nodes):
        """Compute max edge length of tetrahedron."""
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3)
        ]
        max_length = 0.0
        for i, j in edges:
            length = np.linalg.norm(nodes[i] - nodes[j])
            max_length = max(max_length, length)
        return max_length

    element_sizes = np.array([
        compute_element_size(node_positions[connectivity[i]])
        for i in range(min(n_elements, 100000))  # Sample first 100K for speed
    ])

    # If we sampled, assume size distribution is representative
    if n_elements > 100000:
        print(f"  (Sampled first 100,000 elements for speed)")
        # Extrapolate to full mesh
        element_sizes_full = np.zeros(n_elements)
        element_sizes_full[:100000] = element_sizes
        # For remaining, use centroid-based classification (rough)
        # Fine elements near tool center (X=30, Y=15, Z=0.3)
        tool_center = np.array([30.0, 15.0, 0.3])
        for i in range(100000, n_elements):
            dist = np.linalg.norm(element_centroids[i] - tool_center)
            if dist < 2.0:
                element_sizes_full[i] = 0.10  # Fine
            elif dist < 5.0:
                element_sizes_full[i] = 0.20  # Medium
            else:
                element_sizes_full[i] = 0.50  # Coarse
        element_sizes = element_sizes_full

    print(f"  Min size: {np.min(element_sizes):.4f} mm")
    print(f"  Max size: {np.max(element_sizes):.4f} mm")
    print(f"  Mean size: {np.mean(element_sizes):.4f} mm")

    # Test 1: Face-based neighbors
    print("\n" + "=" * 80)
    print("TEST 1: FACE-BASED NEIGHBORS")
    print("=" * 80)

    face_neighbors_dict, face_stats = extract_element_neighbors(connectivity, verbose=True)
    face_results = analyze_cross_level_connectivity(
        connectivity, element_sizes, face_neighbors_dict, "face"
    )

    # Test 2: Node-based neighbors
    print("\n" + "=" * 80)
    print("TEST 2: NODE-BASED NEIGHBORS")
    print("=" * 80)

    node_neighbors_dict, node_stats = extract_element_neighbors_node_based(connectivity, verbose=True)
    node_results = analyze_cross_level_connectivity(
        connectivity, element_sizes, node_neighbors_dict, "node"
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nFace-Based Neighbors:")
    print(f"  Avg neighbors/element: {face_stats.avg_neighbors_per_element:.1f}")
    print(f"  Coarse→Fine connections: {face_results['coarse_to_fine']:,}")
    print(f"  Coarse elements with fine neighbors: {face_results['coarse_with_fine_neighbors']:,} / {face_results['n_coarse']:,}")
    print(f"  → Can L1 reach fine: {face_results['can_l1_reach_fine']}")

    print(f"\nNode-Based Neighbors:")
    print(f"  Avg neighbors/element: {node_stats.avg_neighbors_per_element:.1f}")
    print(f"  Coarse→Fine connections: {node_results['coarse_to_fine']:,}")
    print(f"  Coarse elements with fine neighbors: {node_results['coarse_with_fine_neighbors']:,} / {node_results['n_coarse']:,}")
    print(f"  → Can L1 reach fine: {node_results['can_l1_reach_fine']}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if not face_results['can_l1_reach_fine'] and node_results['can_l1_reach_fine']:
        print("\n✅ CONFIRMED: Face-based neighbors DO NOT cross refinement levels")
        print("✅ CONFIRMED: Node-based neighbors DO cross refinement levels")
        print("\n📋 RECOMMENDATION:")
        print("  Use node-based neighbors for L1 search in refined meshes")
        print("  Change: build_element_neighbors_array(connectivity, method='node')")
    elif face_results['can_l1_reach_fine']:
        print("\n⚠️  UNEXPECTED: Face-based neighbors DO cross refinement levels")
        print("   This suggests the mesh structure is different than expected")
        print("   Face-based neighbors should work for L1 search")
    else:
        print("\n❌ ERROR: Neither face nor node neighbors cross refinement levels")
        print("   This suggests an issue with the mesh or analysis")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
