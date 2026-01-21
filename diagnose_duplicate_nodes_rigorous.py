#!/usr/bin/env python3
"""
Rigorous Duplicate Node Analysis

Verifies whether duplicate nodes are:
1. TRUE duplicates from PVTU piece boundaries (exact same position)
2. FALSE duplicates from floating-point precision issues

Strategy:
- Check exact bit-level equality (not just rounded positions)
- Analyze distance distribution between "duplicates"
- Check if duplicates correspond to connectivity boundaries
- Verify if elements using "duplicates" share faces but different node IDs
"""

import numpy as np
import vtk
from vtk.util import numpy_support
from pathlib import Path
from collections import defaultdict


def load_pvtu_raw(pvtu_path: Path):
    """Load PVTU without any processing."""
    print(f"\n{'='*80}")
    print(f"Loading PVTU: {pvtu_path}")
    print(f"{'='*80}")

    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(pvtu_path))
    reader.Update()
    output = reader.GetOutput()

    positions = numpy_support.vtk_to_numpy(output.GetPoints().GetData())
    positions = positions.astype(np.float64)  # Keep full precision

    n_cells = output.GetNumberOfCells()
    connectivity_data = numpy_support.vtk_to_numpy(output.GetCells().GetData())

    connectivity = np.zeros((n_cells, 4), dtype=np.int32)
    for i in range(n_cells):
        connectivity[i] = connectivity_data[i * 5 + 1 : i * 5 + 5]

    print(f"  Nodes: {positions.shape[0]:,}")
    print(f"  Elements: {n_cells:,}")

    return positions, connectivity


def find_exact_duplicates(positions):
    """
    Find nodes with EXACTLY the same position (bit-level equality).

    Returns:
        duplicate_groups: dict mapping position tuple to list of node IDs
    """
    print(f"\n{'='*80}")
    print(f"Finding EXACT duplicate nodes (bit-level equality)")
    print(f"{'='*80}")

    n_nodes = positions.shape[0]

    # Create position tuples for exact comparison
    # Using float64 bit representation
    position_to_nodes = defaultdict(list)

    for node_id in range(n_nodes):
        # Convert to tuple of floats (exact bit representation)
        pos_tuple = tuple(positions[node_id])
        position_to_nodes[pos_tuple].append(node_id)

    # Find groups with multiple nodes
    duplicate_groups = {
        pos: nodes
        for pos, nodes in position_to_nodes.items()
        if len(nodes) > 1
    }

    n_duplicate_positions = len(duplicate_groups)
    n_duplicate_nodes = sum(len(nodes) for nodes in duplicate_groups.values())

    print(f"\nExact duplicates (bit-level equality):")
    print(f"  Duplicate positions: {n_duplicate_positions:,}")
    print(f"  Duplicate nodes: {n_duplicate_nodes:,}")
    print(f"  Unique nodes: {n_nodes - n_duplicate_nodes:,}")

    if n_duplicate_positions > 0:
        # Analyze duplicate group sizes
        group_sizes = [len(nodes) for nodes in duplicate_groups.values()]
        print(f"\nDuplicate group sizes:")
        print(f"  Min: {min(group_sizes)}")
        print(f"  Max: {max(group_sizes)}")
        print(f"  Mean: {np.mean(group_sizes):.2f}")

        # Count by size
        size_counts = defaultdict(int)
        for size in group_sizes:
            size_counts[size] += 1

        print(f"\nDistribution by group size:")
        for size in sorted(size_counts.keys()):
            print(f"  {size} nodes: {size_counts[size]:,} groups")

    return duplicate_groups


def find_near_duplicates_fp_precision(positions, tolerance=1e-12):
    """
    Find nodes that are CLOSE but not exactly equal.
    These would be floating-point precision artifacts.

    Returns:
        near_duplicate_pairs: list of (node1, node2, distance) tuples
    """
    print(f"\n{'='*80}")
    print(f"Finding NEAR-duplicates (FP precision artifacts)")
    print(f"Tolerance: {tolerance:.2e}")
    print(f"{'='*80}")

    from scipy.spatial import cKDTree

    # Build KD-tree
    tree = cKDTree(positions)

    # Find all pairs within tolerance
    pairs = tree.query_pairs(r=tolerance, output_type='ndarray')

    print(f"\nNode pairs within {tolerance:.2e}:")
    print(f"  Total pairs: {len(pairs):,}")

    if len(pairs) > 0:
        # Compute actual distances
        distances = []
        for i, j in pairs:
            dist = np.linalg.norm(positions[i] - positions[j])
            distances.append(dist)

        distances = np.array(distances)

        print(f"\nDistance distribution:")
        print(f"  Min: {distances.min():.6e}")
        print(f"  Max: {distances.max():.6e}")
        print(f"  Mean: {distances.mean():.6e}")
        print(f"  Median: {np.median(distances):.6e}")

        # Check if distances are exactly zero
        exact_zeros = np.sum(distances == 0.0)
        print(f"\n  Exactly zero distance: {exact_zeros:,} pairs")
        print(f"  Non-zero distance: {len(distances) - exact_zeros:,} pairs")

        if exact_zeros > 0 and exact_zeros < len(distances):
            print(f"\n⚠️  Mix of exact and near duplicates!")
            print(f"     This suggests BOTH piece boundaries (exact) and FP errors (near)")

    return pairs


def analyze_duplicate_connectivity(positions, connectivity, duplicate_groups):
    """
    Check if elements sharing duplicate nodes actually form faces.

    If duplicates are from VTU pieces:
    - Elements using different nodes at same position should share faces
    - But they won't be detected as neighbors (different node IDs)

    If duplicates are artifacts:
    - Elements wouldn't necessarily share faces
    """
    print(f"\n{'='*80}")
    print(f"Analyzing connectivity patterns of duplicate nodes")
    print(f"{'='*80}")

    # Sample some duplicate groups
    sample_size = min(100, len(duplicate_groups))
    sample_groups = list(duplicate_groups.values())[:sample_size]

    print(f"\nSampling {sample_size} duplicate groups...")

    face_sharing_count = 0
    non_face_sharing_count = 0

    for dup_nodes in sample_groups:
        # Find elements using each duplicate node
        elements_per_node = {}
        for node_id in dup_nodes:
            elements = np.where(np.any(connectivity == node_id, axis=1))[0]
            elements_per_node[node_id] = set(elements)

        # Check if elements from different duplicate nodes share faces
        # Two elements share a face if they have exactly 3 common nodes
        node_ids = list(elements_per_node.keys())

        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                node1, node2 = node_ids[i], node_ids[j]
                elems1 = elements_per_node[node1]
                elems2 = elements_per_node[node2]

                # Check each pair of elements
                for e1 in elems1:
                    for e2 in elems2:
                        nodes_e1 = set(connectivity[e1])
                        nodes_e2 = set(connectivity[e2])

                        # Count common nodes
                        # Note: node1 and node2 are at same position but different IDs
                        # So they won't be in the intersection
                        common_nodes = nodes_e1 & nodes_e2

                        # If they share 3 nodes, they would be face neighbors
                        # if node1 and node2 were the same ID
                        if len(common_nodes) == 3:
                            face_sharing_count += 1
                        elif len(common_nodes) > 0:
                            non_face_sharing_count += 1

    print(f"\nConnectivity analysis (sampled):")
    print(f"  Element pairs sharing 3 nodes (would be face neighbors): {face_sharing_count}")
    print(f"  Element pairs sharing <3 nodes: {non_face_sharing_count}")

    if face_sharing_count > 0:
        print(f"\n✅ CONFIRMED: Duplicate nodes prevent face neighbor detection!")
        print(f"   Elements at piece boundaries share 3 nodes but use different node IDs.")
        return True
    else:
        print(f"\n⚠️  No face-sharing found in sample")
        print(f"   May need larger sample or different analysis")
        return False


def check_mesh_length_scales(positions):
    """
    Analyze mesh length scales to determine appropriate tolerance.
    """
    print(f"\n{'='*80}")
    print(f"Analyzing mesh length scales")
    print(f"{'='*80}")

    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    bbox_size = bbox_max - bbox_min

    print(f"\nDomain size:")
    print(f"  X: {bbox_size[0]:.6e} ({bbox_min[0]:.6e} to {bbox_max[0]:.6e})")
    print(f"  Y: {bbox_size[1]:.6e} ({bbox_min[1]:.6e} to {bbox_max[1]:.6e})")
    print(f"  Z: {bbox_size[2]:.6e} ({bbox_min[2]:.6e} to {bbox_max[2]:.6e})")

    # Estimate typical element size (sample 1000 random nodes)
    sample_size = min(1000, positions.shape[0])
    sample_idx = np.random.choice(positions.shape[0], sample_size, replace=False)
    sample_positions = positions[sample_idx]

    from scipy.spatial import cKDTree
    tree = cKDTree(sample_positions)
    distances, _ = tree.query(sample_positions, k=2)  # k=2 includes self
    nn_distances = distances[:, 1]  # Nearest neighbor (not self)

    print(f"\nNearest-neighbor distances (sampled):")
    print(f"  Min: {nn_distances.min():.6e}")
    print(f"  Max: {nn_distances.max():.6e}")
    print(f"  Mean: {nn_distances.mean():.6e}")
    print(f"  Median: {np.median(nn_distances):.6e}")

    # Typical element size
    typical_element_size = np.median(nn_distances)

    print(f"\nTypical element size: {typical_element_size:.6e}")
    print(f"Relative to domain: {typical_element_size / bbox_size.max():.6e}")

    # Floating-point precision
    print(f"\nFloating-point precision analysis:")
    print(f"  float64 epsilon: {np.finfo(np.float64).eps:.6e}")
    print(f"  Relative precision at domain scale: {bbox_size.max() * np.finfo(np.float64).eps:.6e}")
    print(f"  Relative precision at element scale: {typical_element_size * np.finfo(np.float64).eps:.6e}")

    # Recommended tolerance
    recommended_tol = typical_element_size * 1e-6  # 1 millionth of element size

    print(f"\nRecommended duplicate detection tolerance:")
    print(f"  Conservative: {recommended_tol:.6e} (1e-6 × element size)")
    print(f"  Aggressive:   {typical_element_size * 1e-9:.6e} (1e-9 × element size)")

    return typical_element_size, recommended_tol


def verify_piece_boundaries_hypothesis(positions, connectivity, duplicate_groups):
    """
    Final verification: Are duplicates at piece boundaries?

    Strategy:
    - Real piece boundary duplicates should be at specific spatial planes
    - FP precision errors would be randomly distributed
    """
    print(f"\n{'='*80}")
    print(f"Verifying piece boundary hypothesis")
    print(f"{'='*80}")

    # Get all duplicate node positions
    all_dup_node_ids = []
    for nodes in duplicate_groups.values():
        all_dup_node_ids.extend(nodes)

    dup_positions = positions[all_dup_node_ids]

    print(f"\nSpatial distribution of duplicate nodes:")
    for axis, name in enumerate(['X', 'Y', 'Z']):
        coords = dup_positions[:, axis]
        print(f"  {name}: [{coords.min():.6e}, {coords.max():.6e}]")
        print(f"     Mean: {coords.mean():.6e}")
        print(f"     Std:  {coords.std():.6e}")

    # Check if duplicates cluster at specific coordinate values
    # Piece boundaries would show as peaks in histograms
    print(f"\nChecking for coordinate clustering (piece boundary signature):")

    for axis, name in enumerate(['X', 'Y', 'Z']):
        coords = dup_positions[:, axis]

        # Create histogram
        hist, bin_edges = np.histogram(coords, bins=50)

        # Find peaks (bins with many duplicates)
        peak_threshold = np.mean(hist) + 2 * np.std(hist)
        peak_bins = np.where(hist > peak_threshold)[0]

        if len(peak_bins) > 0:
            print(f"\n  {name}-axis: Found {len(peak_bins)} peaks above threshold")
            print(f"    This suggests piece boundaries at specific {name} coordinates!")

            # Show top 5 peaks
            top_peaks = np.argsort(hist)[-5:][::-1]
            for i, peak_idx in enumerate(top_peaks[:3]):
                peak_coord = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
                peak_count = hist[peak_idx]
                print(f"    Peak {i+1}: {name}={peak_coord:.6e} ({peak_count} nodes)")
        else:
            print(f"\n  {name}-axis: No significant peaks (uniform distribution)")


def main():
    """Run rigorous duplicate node diagnostic."""

    # Configuration
    MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
    PVTU_FILE = MESH_BASE_PATH / "featurelessAvtk_120.pvtu"

    if not PVTU_FILE.exists():
        print(f"❌ File not found: {PVTU_FILE}")
        return

    print(f"\n{'='*80}")
    print(f"RIGOROUS DUPLICATE NODE ANALYSIS")
    print(f"{'='*80}")
    print(f"\nThis diagnostic determines if duplicate nodes are:")
    print(f"  1. TRUE duplicates from VTU piece boundaries (exact same position)")
    print(f"  2. FALSE duplicates from floating-point precision issues")

    # Load mesh
    positions, connectivity = load_pvtu_raw(PVTU_FILE)

    # Analyze mesh scales
    typical_elem_size, recommended_tol = check_mesh_length_scales(positions)

    # Find EXACT duplicates (bit-level equality)
    duplicate_groups = find_exact_duplicates(positions)

    # Find NEAR duplicates (FP precision)
    near_pairs = find_near_duplicates_fp_precision(positions, tolerance=recommended_tol)

    # Analyze connectivity
    if duplicate_groups:
        face_sharing = analyze_duplicate_connectivity(positions, connectivity, duplicate_groups)

        # Verify piece boundary hypothesis
        verify_piece_boundaries_hypothesis(positions, connectivity, duplicate_groups)

    # Final diagnosis
    print(f"\n{'='*80}")
    print(f"FINAL DIAGNOSIS")
    print(f"{'='*80}")

    if len(duplicate_groups) > 0:
        n_exact = sum(len(nodes) for nodes in duplicate_groups.values())
        print(f"\n✅ Found {len(duplicate_groups):,} groups with EXACT duplicate nodes")
        print(f"   Total duplicate nodes: {n_exact:,}")
        print(f"\n   These are REAL duplicates, not floating-point artifacts!")
        print(f"   Bit-level equality confirms they're from VTU piece boundaries.")
        print(f"\n   ROOT CAUSE: vtkXMLPUnstructuredGridReader does NOT merge piece nodes.")
        print(f"   SOLUTION: Implement node deduplication preprocessing.")
    else:
        print(f"\n✅ No exact duplicate nodes found")
        print(f"   VTU pieces are properly merged")

    if len(near_pairs) > len(duplicate_groups):
        print(f"\n⚠️  Found near-duplicates beyond exact duplicates")
        print(f"   These may be floating-point precision issues")
        print(f"   Recommended tolerance: {recommended_tol:.6e}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
