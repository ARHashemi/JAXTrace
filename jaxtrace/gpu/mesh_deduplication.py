#!/usr/bin/env python3
"""
Mesh Node Deduplication

Fixes PVTU piece boundary connectivity by merging duplicate nodes.

This module provides preprocessing functions to:
1. Detect nodes at exactly the same position (from VTU pieces)
2. Create unified node ID mapping
3. Remap connectivity to use unified node IDs
4. Compact node position array

Critical for PVTU files where VTK does not merge boundary nodes!
"""

import numpy as np
from typing import Tuple


def deduplicate_nodes(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    velocity_sequence: np.ndarray = None,
    scalar_sequences: dict = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Merge duplicate nodes with exact same position.

    This fixes PVTU piece boundary connectivity where VTK's loader
    creates separate node IDs for nodes at the same position.

    Parameters
    ----------
    node_positions : np.ndarray
        (n_nodes, 3) float64 - node coordinates
    connectivity : np.ndarray
        (n_elements, 4) int32 - element connectivity (tetrahedral)
    velocity_sequence : np.ndarray, optional
        (n_timesteps, n_nodes, 3) float32 - velocity at nodes for each timestep
        If provided, will be remapped to deduplicated node IDs
    scalar_sequences : dict[str, np.ndarray], optional
        Optional per-node scalar field sequences, each shaped
        ``(n_timesteps, n_nodes)``. Remapped in place; the dict is mutated to
        hold the deduplicated arrays. Use this for fields like Temperature,
        Pressure, LEVEL that should follow the dedup of the velocity stack.
    verbose : bool, default=True
        Print progress and statistics

    Returns
    -------
    compacted_positions : np.ndarray
        (n_unique_nodes, 3) float64 - unique node positions
    remapped_connectivity : np.ndarray
        (n_elements, 4) int32 - connectivity using deduplicated node IDs
    n_duplicates : int
        Number of duplicate nodes removed
    remapped_velocity_sequence : np.ndarray or None
        (n_timesteps, n_unique_nodes, 3) float32 - remapped velocity sequence
        None if velocity_sequence was not provided

    Notes
    -----
    - Uses exact bit-level equality (not tolerance-based)
    - Preserves first occurrence as canonical node ID
    - No degenerate elements created (validated)
    - Typical reduction: 20-30% for PVTU meshes
    - Velocity remapping preserves timestep ordering and physical values
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Deduplicating nodes (fixing PVTU piece boundaries)")
        print(f"{'='*80}")

    n_nodes = node_positions.shape[0]
    n_elements = connectivity.shape[0]

    # Step 1: Build deduplication map
    if verbose:
        print(f"  Original nodes: {n_nodes:,}")
        print(f"  Detecting exact duplicates...")

    position_tuples = [tuple(pos) for pos in node_positions]
    position_to_canonical_id = {}
    node_map = np.zeros(n_nodes, dtype=np.int32)

    n_unique = 0
    for node_id in range(n_nodes):
        pos_tuple = position_tuples[node_id]
        if pos_tuple not in position_to_canonical_id:
            position_to_canonical_id[pos_tuple] = n_unique
            node_map[node_id] = n_unique
            n_unique += 1
        else:
            canonical_id = position_to_canonical_id[pos_tuple]
            node_map[node_id] = canonical_id

    n_duplicates = n_nodes - n_unique

    if verbose:
        print(f"  Unique nodes:   {n_unique:,}")
        print(f"  Duplicate nodes: {n_duplicates:,} ({100*n_duplicates/n_nodes:.1f}%)")

    if n_duplicates == 0:
        if verbose:
            print(f"\n✅ No duplicate nodes found - mesh already clean!")
            print(f"{'='*80}\n")
        return node_positions, connectivity, 0, velocity_sequence

    # Step 2: Compact node positions
    if verbose:
        print(f"\n  Compacting node array...")

    compacted_positions = np.zeros((n_unique, 3), dtype=node_positions.dtype)
    for old_id in range(n_nodes):
        new_id = node_map[old_id]
        compacted_positions[new_id] = node_positions[old_id]

    # Step 3: Remap connectivity
    if verbose:
        print(f"  Remapping connectivity...")

    remapped_connectivity = np.zeros_like(connectivity)
    for elem_id in range(n_elements):
        for local_node in range(4):
            old_node_id = connectivity[elem_id, local_node]
            new_node_id = node_map[old_node_id]
            remapped_connectivity[elem_id, local_node] = new_node_id

    # Step 3.5: Remap velocity sequence if provided
    remapped_velocity_sequence = None
    if velocity_sequence is not None:
        if verbose:
            print(f"  Remapping velocity sequence...")

        n_timesteps = velocity_sequence.shape[0]

        # Verify input shape
        if velocity_sequence.shape[1] != n_nodes:
            raise ValueError(
                f"Velocity sequence shape mismatch: "
                f"expected (n_timesteps, {n_nodes}, 3), "
                f"got {velocity_sequence.shape}"
            )

        # Create compacted velocity array
        remapped_velocity_sequence = np.zeros(
            (n_timesteps, n_unique, 3),
            dtype=velocity_sequence.dtype
        )

        # Remap velocities using node_map
        # For each timestep, copy velocity from old node ID to new node ID
        for old_id in range(n_nodes):
            new_id = node_map[old_id]
            # Copy velocity for all timesteps at once
            remapped_velocity_sequence[:, new_id, :] = velocity_sequence[:, old_id, :]

        if verbose:
            print(f"    Original velocity: {velocity_sequence.shape}")
            print(f"    Remapped velocity: {remapped_velocity_sequence.shape}")

    # Step 3.6: Remap scalar sequences if provided
    if scalar_sequences:
        if verbose:
            print(f"  Remapping {len(scalar_sequences)} scalar sequence(s)...")
        for name, seq in list(scalar_sequences.items()):
            if seq is None:
                continue
            if seq.ndim != 2 or seq.shape[1] != n_nodes:
                raise ValueError(
                    f"Scalar sequence '{name}' shape mismatch: "
                    f"expected (n_timesteps, {n_nodes}), got {seq.shape}"
                )
            n_t = seq.shape[0]
            remapped = np.zeros((n_t, n_unique), dtype=seq.dtype)
            for old_id in range(n_nodes):
                new_id = node_map[old_id]
                remapped[:, new_id] = seq[:, old_id]
            scalar_sequences[name] = remapped
            if verbose:
                print(f"    {name}: {seq.shape} -> {remapped.shape}")

    # Step 4: Validate
    if verbose:
        print(f"  Validating...")

    # Check for degenerate elements
    degenerate_count = 0
    for elem_id in range(n_elements):
        nodes = remapped_connectivity[elem_id]
        if len(set(nodes)) < 4:
            degenerate_count += 1

    if degenerate_count > 0:
        print(f"\n⚠️  WARNING: Found {degenerate_count:,} degenerate elements!")
        print(f"   Some elements have duplicate node IDs after merging.")
        print(f"   This may indicate mesh quality issues.")
    else:
        if verbose:
            print(f"  ✅ No degenerate elements")

    # Check connectivity bounds
    max_node_id = remapped_connectivity.max()
    min_node_id = remapped_connectivity.min()
    if min_node_id < 0 or max_node_id >= n_unique:
        raise ValueError(
            f"Invalid connectivity after remapping: "
            f"min={min_node_id}, max={max_node_id}, n_unique={n_unique}"
        )

    if verbose:
        print(f"\n✅ Node deduplication complete!")
        print(f"  Removed {n_duplicates:,} duplicate nodes")
        print(f"  Mesh size: {n_nodes:,} → {n_unique:,} nodes")
        if velocity_sequence is not None:
            print(f"  Velocity arrays remapped: {n_timesteps} timesteps")
        print(f"{'='*80}\n")

    return compacted_positions, remapped_connectivity, n_duplicates, remapped_velocity_sequence


def check_for_duplicates(node_positions: np.ndarray) -> int:
    """
    Quick check for duplicate nodes (without fixing).

    Parameters
    ----------
    node_positions : np.ndarray
        (n_nodes, 3) float64 - node coordinates

    Returns
    -------
    n_duplicates : int
        Number of duplicate nodes detected
    """
    position_tuples = [tuple(pos) for pos in node_positions]
    n_unique = len(set(position_tuples))
    n_duplicates = node_positions.shape[0] - n_unique
    return n_duplicates


if __name__ == "__main__":
    # Example usage
    print("Node Deduplication Module")
    print("=" * 80)
    print("\nThis module fixes PVTU piece boundary connectivity.")
    print("\nUsage:")
    print("  from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes")
    print("  positions, connectivity, n_dup = deduplicate_nodes(positions, connectivity)")
    print("\n" + "=" * 80)
