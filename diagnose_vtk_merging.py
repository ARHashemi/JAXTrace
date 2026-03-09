#!/usr/bin/env python3
"""
Diagnose VTK mesh part merging and deduplication impact on particle loss.

Key questions:
1. Are the 1,591 multi-level cross-cell faces at VTK part boundaries?
2. Are the 344 fallback elements concentrated at part boundaries?
3. Does deduplication create connectivity gaps or mismatches?
4. Are "coarse element blocks" from separate VTK parts?
5. Is there a spatial correlation between part boundaries and particle loss regions?

This investigates whether the mesh assembly process (merging multiple VTK parts
and deduplicating nodes) introduces defects that cause particle tracking failures.
"""

import numpy as np
import time
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET

# Import mesh loading
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu

# Import octree extraction
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import find_axis_aligned_edges_single

# Configuration
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
PVTU_FILE = MESH_BASE_PATH / MESH_FILE_PATTERN.format(timestep=158)


def parse_pvtu_structure(pvtu_path):
    """
    Parse PVTU file to get VTK part structure.

    Returns:
        list of (filename, node_offset, element_offset) for each part
    """
    tree = ET.parse(pvtu_path)
    root = tree.getroot()

    # Find all piece references
    pieces = []
    for piece in root.findall(".//Piece"):
        source = piece.get("Source")
        if source:
            pieces.append(source)

    print(f"  Found {len(pieces)} VTK parts in PVTU file")

    return pieces


def load_individual_vtk_parts(pvtu_path, verbose=True):
    """
    Load each VTK part separately to track element origins.

    Returns:
        parts: list of {
            'filename': str,
            'node_offset': int,
            'element_offset': int,
            'n_nodes': int,
            'n_elements': int
        }
    """
    if verbose:
        print(f"  Parsing PVTU structure: {pvtu_path}")

    # Get part filenames
    piece_files = parse_pvtu_structure(pvtu_path)

    # For this analysis, we'll use the merged mesh but track which elements
    # came from which part based on element ID ranges

    # Load merged mesh to get total counts
    from jaxtrace.gpu.mesh_loader_timedep import load_mesh_from_pvtu

    node_positions_merged, connectivity_merged, _ = load_mesh_from_pvtu(
        pvtu_path, field_name='Displacement', verbose=False
    )

    n_total_nodes = node_positions_merged.shape[0]
    n_total_elements = connectivity_merged.shape[0]

    if verbose:
        print(f"  Merged mesh: {n_total_elements:,} elements, {n_total_nodes:,} nodes")

    # Estimate part boundaries (assuming sequential merging)
    n_parts = len(piece_files)
    approx_elements_per_part = n_total_elements // n_parts

    parts = []
    for i, piece_file in enumerate(piece_files):
        elem_start = i * approx_elements_per_part
        elem_end = (i + 1) * approx_elements_per_part if i < n_parts - 1 else n_total_elements

        parts.append({
            'filename': piece_file,
            'part_id': i,
            'element_offset': elem_start,
            'n_elements': elem_end - elem_start
        })

    return parts, n_total_elements


def main():
    print("="*80)
    print("VTK Mesh Merging and Deduplication Diagnostic")
    print("="*80)
    print()

    # Parse VTK part structure
    print("[1/6] Analyzing VTK part structure...")
    t0 = time.time()

    try:
        parts, n_total_elements = load_individual_vtk_parts(PVTU_FILE, verbose=True)
        n_parts = len(parts)
    except Exception as e:
        print(f"  ⚠️  Could not parse VTK part structure: {e}")
        print(f"  Continuing with merged mesh analysis...")
        parts = []
        n_parts = 0

    print(f"  Analyzed in {time.time()-t0:.1f}s")

    # Load merged mesh
    print("\n[2/6] Loading merged mesh...")
    t0 = time.time()

    node_positions_orig, connectivity_orig, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=(158, 159),
        field_name='Displacement',
        verbose=False
    )

    n_nodes_orig = node_positions_orig.shape[0]
    n_elements = connectivity_orig.shape[0]

    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"    Elements: {n_elements:,}, Nodes: {n_nodes_orig:,}")

    # Deduplicate and track mapping
    print("\n[3/6] Deduplicating with node mapping tracking...")
    t0 = time.time()

    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions_orig, connectivity_orig,
        velocity_sequence=velocity_sequence,
        verbose=True
    )

    n_nodes = node_positions.shape[0]

    print(f"  Deduplicated in {time.time()-t0:.1f}s")
    print(f"    Removed {n_duplicates_removed:,} duplicates ({100.0*n_duplicates_removed/n_nodes_orig:.2f}%)")
    print(f"    Final nodes: {n_nodes:,}")

    # Analyze deduplication impact on connectivity
    print("\n[4/6] Analyzing deduplication impact...")

    # Check if any elements have duplicate nodes (degenerate after dedup)
    degenerate_after_dedup = []

    for elem_id in range(n_elements):
        node_ids = connectivity[elem_id]
        unique_nodes = len(set(node_ids))

        if unique_nodes < 4:
            degenerate_after_dedup.append(elem_id)

    n_degenerate = len(degenerate_after_dedup)

    print(f"  Elements with duplicate nodes after deduplication: {n_degenerate}")

    if n_degenerate > 0:
        print(f"  ⚠️  {n_degenerate} elements became degenerate due to deduplication!")
        print(f"     Sample element IDs: {degenerate_after_dedup[:10]}")
    else:
        print(f"  ✅ No degenerate elements created by deduplication")

    # Extract octree to identify problematic elements
    print("\n[5/6] Extracting octree to find problematic elements...")
    t0 = time.time()

    octree_multi = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )

    print(f"  Extracted in {time.time()-t0:.1f}s")

    # Identify fallback elements (single-cell registration)
    elem_to_cells_offsets = octree_multi.element_to_cells_offsets

    fallback_elements = []
    for elem_id in range(n_elements):
        start = elem_to_cells_offsets[elem_id]
        end = elem_to_cells_offsets[elem_id + 1]
        n_cells = end - start

        if n_cells == 1:
            fallback_elements.append(elem_id)

    n_fallback = len(fallback_elements)

    print(f"  Fallback elements (1 cell): {n_fallback}")

    # Identify non-Kuhn elements
    non_kuhn_elements = []

    for elem_id in range(n_elements):
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        cell_size, level = find_axis_aligned_edges_single(vertices, tolerance=1e-6)

        if np.any(cell_size == 0):
            non_kuhn_elements.append(elem_id)

        if (elem_id + 1) % 500000 == 0:
            print(f"    Processed {elem_id + 1:,}/{n_elements:,}...")

    n_non_kuhn = len(non_kuhn_elements)

    print(f"  Non-Kuhn elements: {n_non_kuhn}")

    # Spatial analysis
    print("\n[6/6] Spatial analysis of problematic elements...")

    # Compute centroids for fallback elements
    if n_fallback > 0:
        fallback_centroids = np.array([
            node_positions[connectivity[elem_id]].mean(axis=0)
            for elem_id in fallback_elements
        ])

        print(f"\n  Fallback element spatial distribution:")
        print(f"    X: min={fallback_centroids[:,0].min():.6f}, max={fallback_centroids[:,0].max():.6f}, range={fallback_centroids[:,0].max()-fallback_centroids[:,0].min():.6f}")
        print(f"    Y: min={fallback_centroids[:,1].min():.6f}, max={fallback_centroids[:,1].max():.6f}, range={fallback_centroids[:,1].max()-fallback_centroids[:,1].min():.6f}")
        print(f"    Z: min={fallback_centroids[:,2].min():.6f}, max={fallback_centroids[:,2].max():.6f}, range={fallback_centroids[:,2].max()-fallback_centroids[:,2].min():.6f}")

        # Check if spatially concentrated
        x_range = fallback_centroids[:,0].max() - fallback_centroids[:,0].min()
        y_range = fallback_centroids[:,1].max() - fallback_centroids[:,1].min()
        z_range = fallback_centroids[:,2].max() - fallback_centroids[:,2].min()

        total_volume = x_range * y_range * z_range

        # Mesh bounding box
        all_centroids = np.array([
            node_positions[connectivity[elem_id]].mean(axis=0)
            for elem_id in np.random.choice(n_elements, size=min(10000, n_elements), replace=False)
        ])

        mesh_x_range = all_centroids[:,0].max() - all_centroids[:,0].min()
        mesh_y_range = all_centroids[:,1].max() - all_centroids[:,1].min()
        mesh_z_range = all_centroids[:,2].max() - all_centroids[:,2].min()
        mesh_volume = mesh_x_range * mesh_y_range * mesh_z_range

        concentration_ratio = total_volume / mesh_volume if mesh_volume > 0 else 0

        print(f"\n  Spatial concentration:")
        print(f"    Fallback elements occupy {100.0*concentration_ratio:.2f}% of mesh volume")

        if concentration_ratio < 0.1:
            print(f"    ⚠️  Highly concentrated! Likely at specific boundaries/interfaces")

    # Analyze part boundaries (if available)
    if n_parts > 0:
        print(f"\n  VTK Part Analysis ({n_parts} parts):")

        for part in parts:
            elem_start = part['element_offset']
            elem_end = elem_start + part['n_elements']

            # Count problematic elements in this part
            fallback_in_part = sum(1 for e in fallback_elements if elem_start <= e < elem_end)
            non_kuhn_in_part = sum(1 for e in non_kuhn_elements if elem_start <= e < elem_end)

            print(f"    Part {part['part_id']:2d}: elements {elem_start:7,}-{elem_end:7,}")
            print(f"      Fallback: {fallback_in_part:4,} ({100.0*fallback_in_part/part['n_elements'] if part['n_elements'] > 0 else 0:.3f}%)")
            print(f"      Non-Kuhn: {non_kuhn_in_part:4,} ({100.0*non_kuhn_in_part/part['n_elements'] if part['n_elements'] > 0 else 0:.3f}%)")

    print("\n" + "="*80)
    print("VTK Merging Diagnostic Complete")
    print("="*80)

    # Summary
    print("\nKey Findings:")

    print(f"\n1. Deduplication:")
    print(f"   - Original nodes: {n_nodes_orig:,}")
    print(f"   - Duplicates removed: {n_duplicates_removed:,} ({100.0*n_duplicates_removed/n_nodes_orig:.2f}%)")
    print(f"   - Final nodes: {n_nodes:,}")
    print(f"   - Degenerate elements created: {n_degenerate}")

    print(f"\n2. Problematic Elements:")
    print(f"   - Fallback (1 cell): {n_fallback} ({100.0*n_fallback/n_elements:.3f}%)")
    print(f"   - Non-Kuhn: {n_non_kuhn} ({100.0*n_non_kuhn/n_elements:.3f}%)")

    if n_fallback > 0:
        print(f"\n3. Spatial Distribution:")
        print(f"   - Fallback elements concentrated: {concentration_ratio < 0.1}")

    print("\nRecommendations:")

    if n_degenerate > 0:
        print(f"  ⚠️  CRITICAL: {n_degenerate} elements have duplicate nodes after deduplication!")
        print("     → These elements cannot be tracked (zero volume)")
        print("     → Check deduplication tolerance (currently uses spatial hashing)")

    if n_duplicates_removed > 100000:
        print(f"  ⚠️  High duplicate rate ({100.0*n_duplicates_removed/n_nodes_orig:.1f}%)")
        print("     → Suggests multiple VTK parts with shared boundaries")
        print("     → Verify deduplication doesn't create connectivity mismatches")

    if n_fallback > 0 and concentration_ratio < 0.1:
        print(f"  ⚠️  Fallback elements highly concentrated in small region")
        print("     → Likely at VTK part boundaries or mesh defects")
        print("     → Investigate this spatial region for assembly issues")


if __name__ == "__main__":
    main()
