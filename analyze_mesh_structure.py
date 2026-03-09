#!/usr/bin/env python3
"""
Analyze mesh structure from timesteps 0-10 to understand octree subdivision.

Goal: Understand how the mesh is actually constructed:
- What are the "cells" (octree cubes)?
- How are cubes subdivided into tetrahedra?
- What is the relationship between tet vertices and cube boundaries?
"""

import numpy as np
from pathlib import Path
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_FIELD_NAME = 'Displacement'

def analyze_timestep(timestep):
    """Analyze one timestep to understand mesh structure."""
    print(f"\n{'='*80}")
    print(f"Analyzing Timestep {timestep}")
    print(f"{'='*80}")

    # Load mesh
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=(timestep, timestep+1),
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )

    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions,
        connectivity,
        velocity_sequence=velocity_sequence,
        verbose=False
    )

    print(f"\nMesh info:")
    print(f"  Nodes: {node_positions.shape[0]:,}")
    print(f"  Elements: {connectivity.shape[0]:,}")

    # Analyze mesh bounding box
    bbox_min = node_positions.min(axis=0)
    bbox_max = node_positions.max(axis=0)
    bbox_size = bbox_max - bbox_min

    print(f"\nBounding box:")
    print(f"  Min: ({bbox_min[0]:.6f}, {bbox_min[1]:.6f}, {bbox_min[2]:.6f})")
    print(f"  Max: ({bbox_max[0]:.6f}, {bbox_max[1]:.6f}, {bbox_max[2]:.6f})")
    print(f"  Size: ({bbox_size[0]:.6f}, {bbox_size[1]:.6f}, {bbox_size[2]:.6f})")

    # Find unique coordinate values (to understand grid structure)
    print(f"\nAnalyzing coordinate structure...")

    for dim, name in enumerate(['X', 'Y', 'Z']):
        coords = node_positions[:, dim]
        unique_coords = np.unique(coords)

        # Compute differences to find cell sizes
        if len(unique_coords) > 1:
            diffs = np.diff(unique_coords)
            # Filter out very small differences (duplicates/noise)
            diffs = diffs[diffs > 1e-8]

            if len(diffs) > 0:
                min_diff = diffs.min()
                max_diff = diffs.max()
                unique_diffs = np.unique(np.round(diffs, 10))

                print(f"\n  {name}-axis:")
                print(f"    Unique values: {len(unique_coords):,}")
                print(f"    Min spacing: {min_diff:.10f}")
                print(f"    Max spacing: {max_diff:.10f}")
                print(f"    Unique spacings: {len(unique_diffs)}")

                if len(unique_diffs) <= 10:
                    print(f"    All spacings:")
                    for d in unique_diffs[:10]:
                        count = np.sum(np.abs(diffs - d) < 1e-10)
                        # Check if power of 2
                        if d > 0:
                            level = -np.log2(d)
                            print(f"      {d:.10f} ({count:6,} occurrences) → 2^({level:.2f})")

    # Analyze element sizes
    print(f"\nAnalyzing element sizes...")

    tolerance = 1e-6
    axis_aligned_counts = {'X': 0, 'Y': 0, 'Z': 0}
    edge_lengths = []

    # Sample first 1000 elements
    sample_size = min(1000, connectivity.shape[0])

    for elem_id in range(sample_size):
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        # Check all 6 edges
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3)
        ]

        for i, j in edges:
            edge_vec = vertices[j] - vertices[i]
            edge_len = np.linalg.norm(edge_vec)

            is_x = (abs(edge_vec[1]) < tolerance) and (abs(edge_vec[2]) < tolerance)
            is_y = (abs(edge_vec[0]) < tolerance) and (abs(edge_vec[2]) < tolerance)
            is_z = (abs(edge_vec[0]) < tolerance) and (abs(edge_vec[1]) < tolerance)

            if is_x:
                axis_aligned_counts['X'] += 1
                edge_lengths.append(('X', edge_len))
            elif is_y:
                axis_aligned_counts['Y'] += 1
                edge_lengths.append(('Y', edge_len))
            elif is_z:
                axis_aligned_counts['Z'] += 1
                edge_lengths.append(('Z', edge_len))

    print(f"  Axis-aligned edges (sample of {sample_size} elements):")
    print(f"    X-aligned: {axis_aligned_counts['X']:,}")
    print(f"    Y-aligned: {axis_aligned_counts['Y']:,}")
    print(f"    Z-aligned: {axis_aligned_counts['Z']:,}")

    # Find unique edge lengths
    if edge_lengths:
        edge_lens = [l for _, l in edge_lengths]
        unique_lens = np.unique(np.round(edge_lens, 10))

        print(f"\n  Unique axis-aligned edge lengths:")
        for length in unique_lens[:20]:
            count = np.sum(np.abs(np.array(edge_lens) - length) < 1e-10)
            if length > 0:
                level = -np.log2(length)
                print(f"    {length:.10f} ({count:6,} edges) → level {level:.2f}")

    # Analyze element volume distribution
    print(f"\nAnalyzing element volumes...")
    volumes = []

    for elem_id in range(sample_size):
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        # Tetrahedron volume = |det([v1-v0, v2-v0, v3-v0])| / 6
        v0, v1, v2, v3 = vertices
        mat = np.array([v1 - v0, v2 - v0, v3 - v0]).T
        volume = abs(np.linalg.det(mat)) / 6.0
        volumes.append(volume)

    volumes = np.array(volumes)
    print(f"  Volume statistics (sample):")
    print(f"    Min: {volumes.min():.2e}")
    print(f"    Max: {volumes.max():.2e}")
    print(f"    Mean: {volumes.mean():.2e}")
    print(f"    Median: {np.median(volumes):.2e}")
    print(f"    Std: {volumes.std():.2e}")

    # Check if volumes follow pattern
    unique_vols = np.unique(np.round(volumes / volumes.min(), 2))
    if len(unique_vols) <= 20:
        print(f"  Volume ratios (relative to minimum):")
        for ratio in unique_vols:
            count = np.sum(np.abs(volumes / volumes.min() - ratio) < 0.1)
            print(f"    {ratio:.2f}× min volume: {count:,} elements")

# Analyze timesteps 0, 1, 2, 5, 10
for t in [0, 1, 2, 5, 10]:
    try:
        analyze_timestep(t)
    except Exception as e:
        print(f"\nFailed to analyze timestep {t}: {e}")

print(f"\n{'='*80}")
print("Analysis complete")
print(f"{'='*80}\n")
