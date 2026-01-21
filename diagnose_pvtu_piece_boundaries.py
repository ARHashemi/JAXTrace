#!/usr/bin/env python3
"""
Diagnostic: PVTU Piece Boundary Connectivity

Investigates if particle loss occurs at PVTU piece boundaries.

This script analyzes:
1. How VTK loads PVTU files (does it merge pieces properly?)
2. Whether face-based neighbors exist across piece boundaries
3. Whether particle loss correlates with piece boundary locations
4. Whether Morton/Hilbert curve separates pieces spatially

Key Questions:
- Does vtkXMLPUnstructuredGridReader merge pieces into a single mesh?
- Are nodes shared properly across piece boundaries?
- Do elements across piece boundaries have proper face-based neighbors?
- Are particle loss locations clustered at piece boundaries?
"""

import numpy as np
import vtk
from vtk.util import numpy_support
from pathlib import Path
import jax
import jax.numpy as jnp
from collections import defaultdict

# Import JAXTrace modules
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu


def load_pvtu_with_piece_info(pvtu_path: Path):
    """
    Load PVTU file and extract piece information.

    Returns:
        positions: (n_nodes, 3) float32
        connectivity: (n_elements, 4) int32
        piece_info: dict with piece metadata
    """
    print(f"\n{'='*80}")
    print(f"Loading PVTU with piece information")
    print(f"{'='*80}")
    print(f"File: {pvtu_path}")

    # Use VTK reader
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(pvtu_path))
    reader.Update()
    output = reader.GetOutput()

    # Extract positions
    positions = numpy_support.vtk_to_numpy(output.GetPoints().GetData())
    positions = positions.astype(np.float32)

    # Extract connectivity
    n_cells = output.GetNumberOfCells()
    connectivity_data = numpy_support.vtk_to_numpy(output.GetCells().GetData())

    connectivity = np.zeros((n_cells, 4), dtype=np.int32)
    for i in range(n_cells):
        connectivity[i] = connectivity_data[i * 5 + 1 : i * 5 + 5]

    print(f"\nMerged mesh:")
    print(f"  Nodes: {positions.shape[0]:,}")
    print(f"  Elements: {n_cells:,}")

    # Check if piece information is available
    piece_info = {}

    # Try to get piece IDs from cell data
    cell_data = output.GetCellData()
    if cell_data.GetNumberOfArrays() > 0:
        print(f"\nCell data arrays:")
        for i in range(cell_data.GetNumberOfArrays()):
            array_name = cell_data.GetArrayName(i)
            array = numpy_support.vtk_to_numpy(cell_data.GetArray(i))
            print(f"  - {array_name}: shape={array.shape}, dtype={array.dtype}")

            # Store arrays that might indicate piece membership
            if 'piece' in array_name.lower() or 'block' in array_name.lower() or 'partition' in array_name.lower():
                piece_info[array_name] = array

    # If no piece info in cell data, try to infer from spatial clustering
    if not piece_info:
        print(f"\nNo explicit piece IDs found in cell data.")
        print(f"Note: VTK merges all pieces into a single mesh.")

    return positions, connectivity, piece_info


def find_piece_boundaries_by_spatial_gaps(positions, connectivity):
    """
    Try to identify piece boundaries by looking for spatial discontinuities.

    Strategy:
    - Compute element centroids
    - Look for large gaps in spatial distribution
    - Elements across gaps might be from different pieces
    """
    print(f"\n{'='*80}")
    print(f"Detecting potential piece boundaries via spatial gaps")
    print(f"{'='*80}")

    n_elements = connectivity.shape[0]

    # Compute element centroids
    print(f"Computing {n_elements:,} element centroids...")
    centroids = np.zeros((n_elements, 3), dtype=np.float32)
    for i in range(n_elements):
        centroids[i] = positions[connectivity[i]].mean(axis=0)

    # Compute bounding box
    bbox_min = centroids.min(axis=0)
    bbox_max = centroids.max(axis=0)
    bbox_size = bbox_max - bbox_min

    print(f"\nDomain bounding box:")
    print(f"  X: [{bbox_min[0]:.6f}, {bbox_max[0]:.6f}] (size: {bbox_size[0]:.6f})")
    print(f"  Y: [{bbox_min[1]:.6f}, {bbox_max[1]:.6f}] (size: {bbox_size[1]:.6f})")
    print(f"  Z: [{bbox_min[2]:.6f}, {bbox_max[2]:.6f}] (size: {bbox_size[2]:.6f})")

    # Partition domain into grid and count elements per cell
    # This helps identify if pieces are spatially separated
    grid_res = 20
    grid_size = bbox_size / grid_res

    print(f"\nSpatial distribution analysis ({grid_res}³ grid):")

    grid_counts = np.zeros((grid_res, grid_res, grid_res), dtype=np.int32)

    for i in range(n_elements):
        idx = ((centroids[i] - bbox_min) / grid_size).astype(np.int32)
        idx = np.clip(idx, 0, grid_res - 1)
        grid_counts[idx[0], idx[1], idx[2]] += 1

    n_occupied = np.sum(grid_counts > 0)
    n_empty = grid_res**3 - n_occupied

    print(f"  Occupied cells: {n_occupied:,} / {grid_res**3:,}")
    print(f"  Empty cells: {n_empty:,}")
    print(f"  Elements per occupied cell:")
    occupied_counts = grid_counts[grid_counts > 0]
    print(f"    Min: {occupied_counts.min():,}")
    print(f"    Max: {occupied_counts.max():,}")
    print(f"    Mean: {occupied_counts.mean():.1f}")
    print(f"    Std: {occupied_counts.std():.1f}")

    # If there are empty cells, pieces might be spatially separated
    if n_empty > 0:
        print(f"\n⚠️  Found {n_empty} empty grid cells - pieces may be spatially separated!")
    else:
        print(f"\n✅ All grid cells occupied - mesh is spatially continuous")

    return centroids


def analyze_neighbor_connectivity_across_domain(positions, connectivity, element_neighbors, centroids):
    """
    Analyze whether neighbor connectivity is uniform across the domain.

    If pieces are disconnected, we should see regions with fewer neighbors.
    """
    print(f"\n{'='*80}")
    print(f"Analyzing neighbor connectivity patterns")
    print(f"{'='*80}")

    n_elements = connectivity.shape[0]

    # Count neighbors per element
    n_neighbors = np.sum(element_neighbors >= 0, axis=1)

    print(f"\nNeighbor statistics:")
    print(f"  Elements with 0 neighbors: {np.sum(n_neighbors == 0):,}")
    print(f"  Elements with 1 neighbor:  {np.sum(n_neighbors == 1):,}")
    print(f"  Elements with 2 neighbors: {np.sum(n_neighbors == 2):,}")
    print(f"  Elements with 3 neighbors: {np.sum(n_neighbors == 3):,}")
    print(f"  Elements with 4 neighbors: {np.sum(n_neighbors == 4):,}")

    # Spatial distribution of under-connected elements
    under_connected = n_neighbors < 4
    n_under = np.sum(under_connected)

    print(f"\n{'='*80}")
    print(f"Under-connected elements (<4 neighbors): {n_under:,} / {n_elements:,} ({100*n_under/n_elements:.2f}%)")
    print(f"{'='*80}")

    if n_under > 0:
        under_centroids = centroids[under_connected]

        print(f"\nSpatial distribution of under-connected elements:")
        for axis, name in enumerate(['X', 'Y', 'Z']):
            coords = under_centroids[:, axis]
            print(f"  {name}: [{coords.min():.6f}, {coords.max():.6f}] (mean: {coords.mean():.6f})")

        # Are they clustered or distributed?
        from scipy.spatial import cKDTree

        # Build KD-tree of under-connected centroids
        if n_under > 1:
            tree = cKDTree(under_centroids)

            # Find nearest neighbor distances
            distances, _ = tree.query(under_centroids, k=2)  # k=2 includes self
            nn_distances = distances[:, 1]  # First neighbor (not self)

            print(f"\nNearest-neighbor distances between under-connected elements:")
            print(f"  Min: {nn_distances.min():.6e}")
            print(f"  Max: {nn_distances.max():.6e}")
            print(f"  Mean: {nn_distances.mean():.6e}")
            print(f"  Median: {np.median(nn_distances):.6e}")

            # If median is small, they're clustered (likely at piece boundaries)
            # If median is large, they're distributed (boundary elements)
            if np.median(nn_distances) < 0.001:
                print(f"\n⚠️  Under-connected elements are CLUSTERED (median NN distance < 0.001)")
                print(f"    This suggests piece boundaries or mesh defects!")
            else:
                print(f"\n✅ Under-connected elements are DISTRIBUTED")
                print(f"    This is normal for boundary elements")

    return n_neighbors


def check_node_sharing_across_suspected_boundaries(positions, connectivity, centroids):
    """
    Check if nodes are shared across spatial regions.

    If PVTU pieces are properly merged, nodes at piece boundaries should be shared.
    If not, we'll see duplicate nodes at the same position.
    """
    print(f"\n{'='*80}")
    print(f"Checking for duplicate nodes (piece boundary merging)")
    print(f"{'='*80}")

    n_nodes = positions.shape[0]

    # Build spatial hash of node positions
    # Round to µm precision
    positions_rounded = np.round(positions * 1e6).astype(np.int64)

    # Find duplicate positions
    unique_positions, inverse, counts = np.unique(
        positions_rounded,
        axis=0,
        return_inverse=True,
        return_counts=True
    )

    n_duplicates = np.sum(counts > 1)

    print(f"\nNode uniqueness check:")
    print(f"  Total nodes: {n_nodes:,}")
    print(f"  Unique positions: {len(unique_positions):,}")
    print(f"  Duplicate positions: {n_duplicates:,}")

    if n_duplicates > 0:
        print(f"\n⚠️  WARNING: Found {n_duplicates:,} positions with multiple nodes!")
        print(f"    This suggests PVTU pieces may not be properly merged!")
        print(f"    Elements across piece boundaries will NOT be neighbors!")

        # Find elements using duplicate nodes
        duplicate_positions_mask = counts[inverse] > 1
        elements_with_duplicates = []

        for elem_id in range(connectivity.shape[0]):
            nodes = connectivity[elem_id]
            if np.any(duplicate_positions_mask[nodes]):
                elements_with_duplicates.append(elem_id)

        print(f"\n  Elements using duplicate nodes: {len(elements_with_duplicates):,}")

        return True  # Merging issue detected
    else:
        print(f"\n✅ All nodes are unique - PVTU pieces properly merged")
        return False  # No merging issue


def analyze_morton_curve_piece_separation(positions, connectivity, centroids):
    """
    Check if Morton curve assigns elements from different pieces to distant leaves.

    Even if pieces are properly merged topologically, the Morton curve might
    separate them spatially if they're in different regions of space.
    """
    print(f"\n{'='*80}")
    print(f"Analyzing Morton curve spatial ordering")
    print(f"{'='*80}")

    # Build Morton octree
    print(f"\nBuilding Morton octree...")
    morton_struct = build_global_morton_octree(
        node_positions=positions,
        connectivity=connectivity,
        leaf_capacity=256,
        max_depth=21,
        verbose=False
    )

    print(f"  Leaves: {morton_struct.n_leaves:,}")
    print(f"  Elements per leaf:")
    print(f"    Min: {morton_struct.leaf_length.min()}")
    print(f"    Max: {morton_struct.leaf_length.max()}")
    print(f"    Mean: {morton_struct.leaf_length.mean():.1f}")

    # Check spatial coherence of leaves
    # For each leaf, compute spatial extent of its elements
    print(f"\nAnalyzing spatial coherence of Morton leaves...")

    leaf_spreads = []

    for leaf_id in range(min(100, morton_struct.n_leaves)):  # Sample first 100 leaves
        start = morton_struct.leaf_start[leaf_id]
        length = morton_struct.leaf_length[leaf_id]

        # Get elements in this leaf
        elem_ids = morton_struct.elem_ids_sorted[start:start+length]

        # Get their centroids
        leaf_centroids = centroids[elem_ids]

        # Compute spatial spread (max distance between any two centroids)
        if len(leaf_centroids) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(leaf_centroids)
            max_spread = distances.max()
            leaf_spreads.append(max_spread)

    if leaf_spreads:
        leaf_spreads = np.array(leaf_spreads)
        print(f"\nSpatial spread within leaves (first 100 leaves):")
        print(f"  Min: {leaf_spreads.min():.6e}")
        print(f"  Max: {leaf_spreads.max():.6e}")
        print(f"  Mean: {leaf_spreads.mean():.6e}")
        print(f"  Median: {np.median(leaf_spreads):.6e}")

        # If max spread is large, Morton curve may be grouping spatially distant elements
        if leaf_spreads.max() > 0.01:
            print(f"\n⚠️  WARNING: Some leaves contain spatially distant elements!")
            print(f"    Max spread: {leaf_spreads.max():.6e}")
            print(f"    This could indicate Morton curve is mixing pieces!")

    return morton_struct


def main():
    """Run full diagnostic on PVTU piece boundary connectivity."""

    # Configuration
    MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
    PVTU_FILE = MESH_BASE_PATH / "featurelessAvtk_120.pvtu"

    if not PVTU_FILE.exists():
        print(f"❌ File not found: {PVTU_FILE}")
        return

    print(f"\n{'='*80}")
    print(f"PVTU PIECE BOUNDARY CONNECTIVITY DIAGNOSTIC")
    print(f"{'='*80}")
    print(f"\nThis diagnostic investigates if particle loss occurs at PVTU piece boundaries.")
    print(f"\nKey questions:")
    print(f"  1. Does VTK merge PVTU pieces properly (shared nodes)?")
    print(f"  2. Do elements across piece boundaries have face-based neighbors?")
    print(f"  3. Does Morton curve separate pieces spatially?")

    # Step 1: Load PVTU with piece info
    positions, connectivity, piece_info = load_pvtu_with_piece_info(PVTU_FILE)

    # Step 2: Detect potential piece boundaries via spatial gaps
    centroids = find_piece_boundaries_by_spatial_gaps(positions, connectivity)

    # Step 3: Build face-based neighbors
    print(f"\n{'='*80}")
    print(f"Building face-based neighbor graph")
    print(f"{'='*80}")
    element_neighbors = build_element_neighbors_array(connectivity, method='face', verbose=True)

    # Step 4: Analyze neighbor connectivity patterns
    n_neighbors = analyze_neighbor_connectivity_across_domain(
        positions, connectivity, element_neighbors, centroids
    )

    # Step 5: Check for duplicate nodes (merging issues)
    has_duplicates = check_node_sharing_across_suspected_boundaries(
        positions, connectivity, centroids
    )

    # Step 6: Analyze Morton curve spatial ordering
    morton_struct = analyze_morton_curve_piece_separation(
        positions, connectivity, centroids
    )

    # Final diagnosis
    print(f"\n{'='*80}")
    print(f"DIAGNOSIS SUMMARY")
    print(f"{'='*80}")

    if has_duplicates:
        print(f"\n❌ CRITICAL ISSUE DETECTED:")
        print(f"   PVTU pieces have DUPLICATE NODES at boundaries!")
        print(f"   Elements across piece boundaries are NOT topologically connected!")
        print(f"   This WILL cause particle loss at piece boundaries!")
        print(f"\n   ROOT CAUSE: VTK is not merging piece nodes properly.")
        print(f"   SOLUTION: Regenerate PVTU files with proper node sharing,")
        print(f"             or post-process to merge duplicate nodes.")
    else:
        print(f"\n✅ PVTU pieces are properly merged (nodes shared)")
        print(f"   Elements across piece boundaries should be neighbors.")
        print(f"\n   If particle loss still occurs:")
        print(f"   - Check if L2 Morton search separates pieces spatially")
        print(f"   - Check if refined regions span piece boundaries")
        print(f"   - Verify neighbor graph with node-based method")

    print(f"\n{'='*80}")
    print(f"Diagnostic complete.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
