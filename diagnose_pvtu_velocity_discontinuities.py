"""
Diagnose Velocity Field Discontinuities at PVTU Piece Boundaries

This script tests if velocity values are inconsistent at PVTU piece boundaries
BEFORE deduplication, and verifies that deduplication doesn't accidentally merge
nearby nodes in refined regions.

Analysis approach:
1. Load mesh WITHOUT deduplication
2. Identify duplicate nodes at exact same position (PVTU boundaries)
3. Check if duplicate nodes have DIFFERENT velocity values
4. Apply deduplication
5. Verify no elements were lost or corrupted
6. Check minimum node spacing after deduplication (ensure refined regions safe)

Usage:
    python diagnose_pvtu_velocity_discontinuities.py > logs/diagnose_pvtu_velocity.log
"""

import numpy as np
from pathlib import Path
import time

# Import JAXTrace modules
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

# ============================================================================
# Configuration (match production script)
# ============================================================================

MESH_BASE_PATH = Path('/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule')
MESH_FILE_PATTERN = 'featurelessAvtk_{timestep}.pvtu'
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'

# ============================================================================
# Main Diagnostic
# ============================================================================

def main():
    print("=" * 80)
    print("Diagnosing PVTU Piece Boundary Issues")
    print("=" * 80)
    print("\nThis diagnostic checks:")
    print("  1. Velocity discontinuities at duplicate nodes (PVTU boundaries)")
    print("  2. Mesh integrity after deduplication")
    print("  3. Safety of deduplication in refined regions")

    # ========================================================================
    # 1. Load mesh WITHOUT deduplication
    # ========================================================================

    print("\n[1/5] Loading mesh WITHOUT deduplication...")

    node_positions_original, connectivity_original, velocity_sequence_original = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=True
    )

    n_nodes_original = len(node_positions_original)
    n_elements_original = len(connectivity_original)
    n_velocity_steps = len(velocity_sequence_original)

    print(f"\nOriginal mesh (BEFORE deduplication):")
    print(f"  Nodes: {n_nodes_original:,}")
    print(f"  Elements: {n_elements_original:,}")
    print(f"  Velocity timesteps: {n_velocity_steps}")

    # ========================================================================
    # 2. Identify duplicate nodes at PVTU boundaries
    # ========================================================================

    print("\n[2/5] Identifying duplicate nodes at PVTU piece boundaries...")
    print("  Building position-to-nodes map (exact bit-level equality)...")

    t0 = time.time()
    position_tuples = [tuple(pos) for pos in node_positions_original]
    position_to_nodes = {}

    for node_id, pos_tuple in enumerate(position_tuples):
        if (node_id + 1) % 100000 == 0:
            print(f"    Processed {node_id+1:,}/{n_nodes_original:,} nodes...")

        if pos_tuple not in position_to_nodes:
            position_to_nodes[pos_tuple] = []
        position_to_nodes[pos_tuple].append(node_id)

    # Find positions with duplicate nodes
    duplicate_positions = {}  # position -> list of node IDs
    for pos, node_list in position_to_nodes.items():
        if len(node_list) > 1:
            duplicate_positions[pos] = node_list

    n_unique_positions = len(position_to_nodes)
    n_duplicate_positions = len(duplicate_positions)
    n_duplicate_nodes = sum(len(nodes) for nodes in duplicate_positions.values())

    print(f"\n  Analysis complete in {time.time() - t0:.2f}s")
    print(f"\n  Total nodes: {n_nodes_original:,}")
    print(f"  Unique positions: {n_unique_positions:,}")
    print(f"  Positions with duplicates: {n_duplicate_positions:,}")
    print(f"  Duplicate nodes: {n_duplicate_nodes:,} ({100.0 * n_duplicate_nodes / n_nodes_original:.2f}%)")

    # ========================================================================
    # 3. Check velocity discontinuities at duplicate nodes
    # ========================================================================

    print("\n[3/5] Checking velocity values at duplicate nodes...")
    print("  Question: Do nodes at SAME position have DIFFERENT velocities?")

    # Use first velocity timestep
    velocity_field = velocity_sequence_original[0]

    print(f"\n  Velocity field shape: {velocity_field.shape}")
    print(f"  Velocity magnitude range: [{np.linalg.norm(velocity_field, axis=1).min():.6e}, {np.linalg.norm(velocity_field, axis=1).max():.6e}]")

    # Check velocity differences at duplicate positions
    print(f"\n  Checking {n_duplicate_positions:,} positions with duplicates...")

    max_velocity_diff = 0.0
    max_velocity_diff_position = None
    max_velocity_diff_nodes = None

    velocity_differences = []

    n_checked = 0
    for pos, node_list in duplicate_positions.items():
        if n_checked % 10000 == 0 and n_checked > 0:
            print(f"    Checked {n_checked:,}/{n_duplicate_positions:,} positions...")
        n_checked += 1

        # Get velocities at all duplicate nodes
        velocities = velocity_field[node_list]  # (n_duplicates, 3)

        # Compute pairwise differences
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                vel_diff = np.linalg.norm(velocities[i] - velocities[j])
                velocity_differences.append(vel_diff)

                if vel_diff > max_velocity_diff:
                    max_velocity_diff = vel_diff
                    max_velocity_diff_position = pos
                    max_velocity_diff_nodes = (node_list[i], node_list[j])

    velocity_differences = np.array(velocity_differences)

    print(f"\n  Velocity Discontinuity Statistics at Duplicate Nodes:")
    print(f"    Pairs checked: {len(velocity_differences):,}")
    print(f"    Mean difference: {velocity_differences.mean():.6e}")
    print(f"    Median difference: {np.median(velocity_differences):.6e}")
    print(f"    Std deviation: {velocity_differences.std():.6e}")
    print(f"    Min difference: {velocity_differences.min():.6e}")
    print(f"    Max difference: {velocity_differences.max():.6e}")
    print(f"    95th percentile: {np.percentile(velocity_differences, 95):.6e}")
    print(f"    99th percentile: {np.percentile(velocity_differences, 99):.6e}")

    # Count significant discontinuities
    threshold_1e6 = (velocity_differences > 1e-6).sum()
    threshold_1e8 = (velocity_differences > 1e-8).sum()
    threshold_1e10 = (velocity_differences > 1e-10).sum()

    print(f"\n  Number of pairs with velocity difference:")
    print(f"    > 1e-10: {threshold_1e10:,} ({100.0 * threshold_1e10 / len(velocity_differences):.2f}%)")
    print(f"    > 1e-8:  {threshold_1e8:,} ({100.0 * threshold_1e8 / len(velocity_differences):.2f}%)")
    print(f"    > 1e-6:  {threshold_1e6:,} ({100.0 * threshold_1e6 / len(velocity_differences):.2f}%)")

    if max_velocity_diff > 0:
        print(f"\n  Maximum velocity discontinuity:")
        print(f"    Magnitude: {max_velocity_diff:.6e}")
        print(f"    Position: {max_velocity_diff_position}")
        print(f"    Node IDs: {max_velocity_diff_nodes}")
        print(f"    Velocities:")
        print(f"      Node {max_velocity_diff_nodes[0]}: {velocity_field[max_velocity_diff_nodes[0]]}")
        print(f"      Node {max_velocity_diff_nodes[1]}: {velocity_field[max_velocity_diff_nodes[1]]}")

    # ========================================================================
    # 4. Apply deduplication and verify mesh integrity
    # ========================================================================

    print("\n[4/5] Applying deduplication and verifying mesh integrity...")

    print("\n  Running deduplicate_nodes()...")
    t0 = time.time()
    node_positions_dedup, connectivity_dedup, n_duplicates_removed, velocity_sequence_dedup = deduplicate_nodes(
        node_positions_original, connectivity_original,
        velocity_sequence=velocity_sequence_original,
        verbose=True
    )
    t_dedup = time.time() - t0

    n_nodes_dedup = len(node_positions_dedup)
    n_elements_dedup = len(connectivity_dedup)

    print(f"\n  Deduplication complete in {t_dedup:.2f}s")
    print(f"\n  After deduplication:")
    print(f"    Nodes: {n_nodes_dedup:,} (removed {n_duplicates_removed:,})")
    print(f"    Elements: {n_elements_dedup:,}")

    # Verify element count unchanged
    if n_elements_dedup != n_elements_original:
        print(f"\n  ❌ CRITICAL ERROR: Element count changed!")
        print(f"     Before: {n_elements_original:,}, After: {n_elements_dedup:,}")
        print(f"     Lost {n_elements_original - n_elements_dedup:,} elements!")
        return

    print(f"  ✅ Element count preserved: {n_elements_dedup:,}")

    # Verify connectivity valid
    max_node_id = np.max(connectivity_dedup)
    if max_node_id >= n_nodes_dedup:
        print(f"\n  ❌ CRITICAL ERROR: Invalid connectivity!")
        print(f"     Max node ID: {max_node_id}, but only {n_nodes_dedup} nodes")
        return

    print(f"  ✅ Connectivity valid (max node ID {max_node_id} < {n_nodes_dedup})")

    # Verify no degenerate elements
    print(f"\n  Checking for degenerate elements...")
    n_degenerate = 0
    for elem_id in range(n_elements_dedup):
        nodes = connectivity_dedup[elem_id]
        if len(set(nodes)) < 4:
            n_degenerate += 1

    if n_degenerate > 0:
        print(f"  ❌ CRITICAL ERROR: Found {n_degenerate:,} degenerate elements!")
        return

    print(f"  ✅ No degenerate elements")

    # Verify velocity sequence shape
    if velocity_sequence_dedup.shape != (n_velocity_steps, n_nodes_dedup, 3):
        print(f"\n  ❌ CRITICAL ERROR: Velocity sequence shape mismatch!")
        print(f"     Expected: ({n_velocity_steps}, {n_nodes_dedup}, 3)")
        print(f"     Got: {velocity_sequence_dedup.shape}")
        return

    print(f"  ✅ Velocity sequence correctly remapped: {velocity_sequence_dedup.shape}")

    # ========================================================================
    # 5. Check minimum node spacing (safety in refined regions)
    # ========================================================================

    print("\n[5/5] Checking minimum node spacing after deduplication...")
    print("  Question: Did we accidentally merge nearby nodes in refined regions?")

    # Sample 10,000 random nodes and find nearest neighbor
    n_sample = min(10000, n_nodes_dedup)
    sample_indices = np.random.choice(n_nodes_dedup, n_sample, replace=False)

    print(f"\n  Sampling {n_sample:,} nodes to find minimum spacing...")

    min_distances = []
    for i, node_id in enumerate(sample_indices):
        if (i + 1) % 1000 == 0:
            print(f"    Processed {i+1:,}/{n_sample:,} nodes...")

        pos = node_positions_dedup[node_id]

        # Find nearest neighbor (excluding self)
        distances = np.linalg.norm(node_positions_dedup - pos, axis=1)
        distances[node_id] = np.inf  # Exclude self
        min_dist = np.min(distances)
        min_distances.append(min_dist)

    min_distances = np.array(min_distances)

    print(f"\n  Minimum Node Spacing Statistics (after deduplication):")
    print(f"    Mean: {min_distances.mean():.6e}")
    print(f"    Median: {np.median(min_distances):.6e}")
    print(f"    Std: {min_distances.std():.6e}")
    print(f"    Min: {min_distances.min():.6e}")
    print(f"    Max: {min_distances.max():.6e}")
    print(f"    1st percentile: {np.percentile(min_distances, 1):.6e}")
    print(f"    5th percentile: {np.percentile(min_distances, 5):.6e}")

    # Compare to element size range
    print(f"\n  Element Size Analysis:")
    element_sizes = []
    for elem_id in range(min(10000, n_elements_dedup)):
        nodes = connectivity_dedup[elem_id]
        node_coords = node_positions_dedup[nodes]

        # Compute characteristic element size (max edge length)
        max_edge = 0.0
        for i in range(4):
            for j in range(i + 1, 4):
                edge_len = np.linalg.norm(node_coords[i] - node_coords[j])
                max_edge = max(max_edge, edge_len)

        element_sizes.append(max_edge)

    element_sizes = np.array(element_sizes)

    print(f"    Element edge length (sampled {len(element_sizes):,} elements):")
    print(f"      Min: {element_sizes.min():.6e}")
    print(f"      Max: {element_sizes.max():.6e}")
    print(f"      Ratio: {element_sizes.max() / element_sizes.min():.1f}×")

    # Safety check: minimum node spacing should be >= smallest element edge
    if min_distances.min() < element_sizes.min() * 0.5:
        print(f"\n  ⚠️  WARNING: Minimum node spacing ({min_distances.min():.6e}) is less than")
        print(f"      half the smallest element edge ({element_sizes.min():.6e})")
        print(f"      Deduplication may have merged nodes that shouldn't be merged!")
    else:
        print(f"\n  ✅ Minimum node spacing is reasonable")

    # ========================================================================
    # FINAL VERDICT
    # ========================================================================

    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    print(f"\n1. Velocity Discontinuities at PVTU Boundaries:")
    if velocity_differences.max() > 1e-6:
        print(f"   ❌ CONFIRMED: Significant velocity discontinuities detected!")
        print(f"      Max difference: {velocity_differences.max():.6e}")
        print(f"      {threshold_1e6:,} duplicate pairs have diff > 1e-6")
        print(f"\n   ROOT CAUSE: Different VTU pieces have different velocity values")
        print(f"                at shared boundary nodes.")
        print(f"\n   IMPACT: Particles crossing PVTU boundaries will experience")
        print(f"           discontinuous velocity field → incorrect trajectories")
        print(f"\n   SOLUTION: Average velocity values during deduplication")
    elif velocity_differences.max() > 1e-10:
        print(f"   ⚠️  MINOR: Small velocity discontinuities detected")
        print(f"      Max difference: {velocity_differences.max():.6e}")
        print(f"      Likely due to float32 rounding, not physical discontinuity")
        print(f"\n   IMPACT: Negligible - differences are at machine precision level")
    else:
        print(f"   ✅ NO ISSUES: Velocity values are consistent at PVTU boundaries")

    print(f"\n2. Mesh Integrity After Deduplication:")
    print(f"   ✅ Element count preserved: {n_elements_dedup:,}")
    print(f"   ✅ Connectivity valid")
    print(f"   ✅ No degenerate elements")
    print(f"   ✅ Velocity sequence correctly remapped")

    print(f"\n3. Deduplication Safety in Refined Regions:")
    if min_distances.min() < element_sizes.min() * 0.5:
        print(f"   ⚠️  CONCERN: Minimum node spacing may be too small")
    else:
        print(f"   ✅ Node spacing is reasonable in refined regions")

    print("\n" + "=" * 80)
    print("COMPLETED")
    print("=" * 80)

if __name__ == '__main__':
    main()
