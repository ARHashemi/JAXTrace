#!/usr/bin/env python3
"""
Node Deduplication Preprocessing

Fixes PVTU piece boundary connectivity by merging duplicate nodes.

This module:
1. Detects nodes at exactly the same position (from VTU pieces)
2. Creates a unified node ID mapping
3. Remaps connectivity to use unified node IDs
4. Compacts node position array
5. Exports fixed mesh for tracking

Key Features:
- Exact bit-level duplicate detection (not tolerance-based)
- Preserves mesh topology and geometry
- Updates connectivity consistently
- Validates neighbor connectivity after fix
"""

import numpy as np
import vtk
from vtk.util import numpy_support
from pathlib import Path
from typing import Tuple, Dict
import time


def load_mesh_with_duplicates(pvtu_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mesh from PVTU file (with duplicate nodes).

    Returns:
        positions: (n_nodes, 3) float64
        connectivity: (n_elements, 4) int32
    """
    print(f"Loading mesh: {pvtu_path}")

    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(pvtu_path))
    reader.Update()
    output = reader.GetOutput()

    positions = numpy_support.vtk_to_numpy(output.GetPoints().GetData())
    positions = positions.astype(np.float64)

    n_cells = output.GetNumberOfCells()
    connectivity_data = numpy_support.vtk_to_numpy(output.GetCells().GetData())

    connectivity = np.zeros((n_cells, 4), dtype=np.int32)
    for i in range(n_cells):
        connectivity[i] = connectivity_data[i * 5 + 1 : i * 5 + 5]

    print(f"  Nodes: {positions.shape[0]:,}")
    print(f"  Elements: {n_cells:,}")

    return positions, connectivity


def build_node_deduplication_map(positions: np.ndarray, verbose: bool = True) -> np.ndarray:
    """
    Build mapping from duplicate node IDs to canonical node IDs.

    Uses exact bit-level equality (not tolerance).

    Parameters:
        positions: (n_nodes, 3) float64 - node coordinates

    Returns:
        node_map: (n_nodes,) int32 - node_map[old_id] = new_canonical_id
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Building node deduplication map")
        print(f"{'='*80}")

    n_nodes = positions.shape[0]

    # Create position tuples for exact comparison
    position_tuples = [tuple(pos) for pos in positions]

    # Build mapping: position -> canonical node ID
    position_to_canonical_id = {}
    node_map = np.zeros(n_nodes, dtype=np.int32)

    n_unique = 0
    n_duplicate = 0

    for node_id in range(n_nodes):
        if verbose and node_id % 100000 == 0 and node_id > 0:
            print(f"  Processed {node_id:,} / {n_nodes:,} nodes...")

        pos_tuple = position_tuples[node_id]

        if pos_tuple not in position_to_canonical_id:
            # First occurrence - assign as canonical
            canonical_id = n_unique
            position_to_canonical_id[pos_tuple] = canonical_id
            node_map[node_id] = canonical_id
            n_unique += 1
        else:
            # Duplicate - map to existing canonical ID
            canonical_id = position_to_canonical_id[pos_tuple]
            node_map[node_id] = canonical_id
            n_duplicate += 1

    if verbose:
        print(f"\nNode deduplication statistics:")
        print(f"  Original nodes: {n_nodes:,}")
        print(f"  Unique nodes:   {n_unique:,}")
        print(f"  Duplicate nodes: {n_duplicate:,} ({100*n_duplicate/n_nodes:.1f}%)")

    return node_map


def compact_node_positions(positions: np.ndarray, node_map: np.ndarray) -> np.ndarray:
    """
    Create compacted node position array with duplicates removed.

    Parameters:
        positions: (n_nodes_old, 3) float64
        node_map: (n_nodes_old,) int32 - maps old IDs to new canonical IDs

    Returns:
        compacted_positions: (n_nodes_new, 3) float64
    """
    n_unique = node_map.max() + 1

    print(f"\n{'='*80}")
    print(f"Compacting node position array")
    print(f"{'='*80}")
    print(f"  Original nodes: {positions.shape[0]:,}")
    print(f"  Compacted nodes: {n_unique:,}")

    compacted_positions = np.zeros((n_unique, 3), dtype=np.float64)

    # For each original node, assign its position to the canonical ID
    # (All duplicates will write the same position, so order doesn't matter)
    for old_id in range(positions.shape[0]):
        new_id = node_map[old_id]
        compacted_positions[new_id] = positions[old_id]

    return compacted_positions


def remap_connectivity(connectivity: np.ndarray, node_map: np.ndarray) -> np.ndarray:
    """
    Remap connectivity to use canonical node IDs.

    Parameters:
        connectivity: (n_elements, 4) int32 - old node IDs
        node_map: (n_nodes_old,) int32 - maps old IDs to canonical IDs

    Returns:
        remapped_connectivity: (n_elements, 4) int32 - canonical node IDs
    """
    print(f"\n{'='*80}")
    print(f"Remapping connectivity")
    print(f"{'='*80}")

    n_elements = connectivity.shape[0]
    remapped_connectivity = np.zeros_like(connectivity)

    for elem_id in range(n_elements):
        if elem_id % 500000 == 0 and elem_id > 0:
            print(f"  Processed {elem_id:,} / {n_elements:,} elements...")

        # Remap each node ID
        for local_node in range(4):
            old_node_id = connectivity[elem_id, local_node]
            new_node_id = node_map[old_node_id]
            remapped_connectivity[elem_id, local_node] = new_node_id

    print(f"  Remapped {n_elements:,} elements")

    return remapped_connectivity


def validate_merged_mesh(positions: np.ndarray, connectivity: np.ndarray):
    """
    Validate the merged mesh for correctness.

    Checks:
    1. No duplicate positions remain
    2. All connectivity node IDs are valid
    3. No degenerate elements (duplicate nodes in connectivity)
    """
    print(f"\n{'='*80}")
    print(f"Validating merged mesh")
    print(f"{'='*80}")

    n_nodes = positions.shape[0]
    n_elements = connectivity.shape[0]

    # Check 1: No duplicate positions
    position_tuples = [tuple(pos) for pos in positions]
    unique_positions = set(position_tuples)

    if len(unique_positions) == n_nodes:
        print(f"✅ No duplicate nodes in compacted mesh ({n_nodes:,} unique)")
    else:
        print(f"❌ WARNING: Found {n_nodes - len(unique_positions):,} duplicate positions!")

    # Check 2: Valid connectivity
    max_node_id = connectivity.max()
    min_node_id = connectivity.min()

    if min_node_id >= 0 and max_node_id < n_nodes:
        print(f"✅ All connectivity node IDs valid [0, {n_nodes-1}]")
    else:
        print(f"❌ Invalid connectivity: min={min_node_id}, max={max_node_id}, n_nodes={n_nodes}")

    # Check 3: No degenerate elements
    degenerate_count = 0
    for elem_id in range(n_elements):
        nodes = connectivity[elem_id]
        if len(set(nodes)) < 4:
            degenerate_count += 1

    if degenerate_count == 0:
        print(f"✅ No degenerate elements (all have 4 unique nodes)")
    else:
        print(f"❌ WARNING: Found {degenerate_count:,} degenerate elements!")
        print(f"   These elements have duplicate node IDs after merging.")

    # Check 4: Mesh topology
    print(f"\nMesh statistics:")
    print(f"  Nodes: {n_nodes:,}")
    print(f"  Elements: {n_elements:,}")

    # Compute element volumes to check for negative/zero volumes
    sample_size = min(10000, n_elements)
    sample_ids = np.random.choice(n_elements, sample_size, replace=False)

    volumes = []
    for elem_id in sample_ids:
        nodes = connectivity[elem_id]
        v0, v1, v2, v3 = positions[nodes]
        e1 = v1 - v0
        e2 = v2 - v0
        e3 = v3 - v0
        det = np.dot(e1, np.cross(e2, e3))
        volume = abs(det) / 6.0
        volumes.append(volume)

    volumes = np.array(volumes)

    print(f"  Element volumes (sampled):")
    print(f"    Min: {volumes.min():.6e}")
    print(f"    Max: {volumes.max():.6e}")
    print(f"    Mean: {volumes.mean():.6e}")

    if volumes.min() > 1e-15:
        print(f"✅ All sampled elements have positive volume")
    else:
        print(f"❌ WARNING: Some elements have near-zero volume!")


def save_merged_mesh_npz(
    positions: np.ndarray,
    connectivity: np.ndarray,
    output_path: Path
):
    """
    Save merged mesh to NPZ file for fast loading.

    Parameters:
        positions: (n_nodes, 3) float64
        connectivity: (n_elements, 4) int32
        output_path: Path to output .npz file
    """
    print(f"\n{'='*80}")
    print(f"Saving merged mesh")
    print(f"{'='*80}")
    print(f"  Output: {output_path}")

    np.savez_compressed(
        output_path,
        node_positions=positions.astype(np.float32),  # Convert to float32 for GPU
        connectivity=connectivity.astype(np.int32)
    )

    file_size_mb = output_path.stat().st_size / (1024**2)
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"✅ Mesh saved successfully")


def merge_duplicate_nodes_full_pipeline(
    pvtu_path: Path,
    output_path: Path,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete pipeline to merge duplicate nodes.

    Parameters:
        pvtu_path: Path to input PVTU file
        output_path: Path to output NPZ file
        verbose: Print progress

    Returns:
        compacted_positions: (n_nodes_new, 3) float64
        remapped_connectivity: (n_elements, 4) int32
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"NODE DEDUPLICATION PIPELINE")
        print(f"{'='*80}")
        print(f"Input:  {pvtu_path}")
        print(f"Output: {output_path}")

    t_start = time.time()

    # Step 1: Load mesh with duplicates
    positions, connectivity = load_mesh_with_duplicates(pvtu_path)

    # Step 2: Build deduplication map
    node_map = build_node_deduplication_map(positions, verbose=verbose)

    # Step 3: Compact node positions
    compacted_positions = compact_node_positions(positions, node_map)

    # Step 4: Remap connectivity
    remapped_connectivity = remap_connectivity(connectivity, node_map)

    # Step 5: Validate
    validate_merged_mesh(compacted_positions, remapped_connectivity)

    # Step 6: Save
    save_merged_mesh_npz(compacted_positions, remapped_connectivity, output_path)

    t_total = time.time() - t_start

    if verbose:
        print(f"\n{'='*80}")
        print(f"Pipeline complete: {t_total:.1f}s")
        print(f"{'='*80}")

    return compacted_positions, remapped_connectivity


def main():
    """Run node deduplication on FLA mesh."""

    # Configuration
    MESH_BASE_PATH = Path("/path/to/FLA/post/0eule")
    PVTU_FILE = MESH_BASE_PATH / "featurelessAvtk_120.pvtu"
    OUTPUT_FILE = MESH_BASE_PATH / "featurelessAvtk_120_merged.npz"

    if not PVTU_FILE.exists():
        print(f"❌ File not found: {PVTU_FILE}")
        return

    # Run pipeline
    compacted_positions, remapped_connectivity = merge_duplicate_nodes_full_pipeline(
        PVTU_FILE,
        OUTPUT_FILE,
        verbose=True
    )

    print(f"\n{'='*80}")
    print(f"✅ Node deduplication complete!")
    print(f"{'='*80}")
    print(f"\nTo use the merged mesh:")
    print(f"  data = np.load('{OUTPUT_FILE}')")
    print(f"  node_positions = data['node_positions']")
    print(f"  connectivity = data['connectivity']")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
