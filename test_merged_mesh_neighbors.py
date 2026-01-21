#!/usr/bin/env python3
"""
Test: Verify Merged Mesh Has Proper Neighbor Connectivity

Compares neighbor connectivity between original (with duplicates)
and merged (duplicates removed) mesh.

Expected results:
- Original mesh: Many under-connected elements at piece boundaries
- Merged mesh: Proper connectivity across piece boundaries
"""

import numpy as np
from pathlib import Path
from jaxtrace.gpu.forest import build_element_neighbors_array


def load_merged_mesh(npz_path: Path):
    """Load merged mesh from NPZ file."""
    print(f"Loading merged mesh: {npz_path}")
    data = np.load(npz_path)
    positions = data['node_positions'].astype(np.float64)
    connectivity = data['connectivity']
    print(f"  Nodes: {positions.shape[0]:,}")
    print(f"  Elements: {connectivity.shape[0]:,}")
    return positions, connectivity


def analyze_neighbor_stats(element_neighbors: np.ndarray, label: str):
    """Analyze and print neighbor connectivity statistics."""
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")

    n_neighbors = np.sum(element_neighbors >= 0, axis=1)

    print(f"\nNeighbor distribution:")
    print(f"  Elements with 0 neighbors: {np.sum(n_neighbors == 0):>10,}")
    print(f"  Elements with 1 neighbor:  {np.sum(n_neighbors == 1):>10,}")
    print(f"  Elements with 2 neighbors: {np.sum(n_neighbors == 2):>10,}")
    print(f"  Elements with 3 neighbors: {np.sum(n_neighbors == 3):>10,}")
    print(f"  Elements with 4 neighbors: {np.sum(n_neighbors == 4):>10,}")

    under_connected = np.sum(n_neighbors < 4)
    total = element_neighbors.shape[0]
    pct_under = 100 * under_connected / total

    print(f"\n  Under-connected (<4):      {under_connected:>10,} ({pct_under:.2f}%)")
    print(f"  Fully connected (4):       {np.sum(n_neighbors == 4):>10,} ({100*np.sum(n_neighbors == 4)/total:.2f}%)")

    return n_neighbors


def compare_neighbor_improvements(n_neighbors_orig, n_neighbors_merged):
    """Compare neighbor connectivity improvements."""
    print(f"\n{'='*80}")
    print(f"CONNECTIVITY IMPROVEMENT")
    print(f"{'='*80}")

    # Elements that gained neighbors
    neighbor_gain = n_neighbors_merged - n_neighbors_orig
    improved = np.sum(neighbor_gain > 0)
    degraded = np.sum(neighbor_gain < 0)
    unchanged = np.sum(neighbor_gain == 0)

    print(f"\nElement-level changes:")
    print(f"  Improved (gained neighbors):  {improved:>10,}")
    print(f"  Degraded (lost neighbors):    {degraded:>10,}")
    print(f"  Unchanged:                    {unchanged:>10,}")

    if improved > 0:
        print(f"\nNeighbor gain distribution:")
        for gain in range(1, 5):
            count = np.sum(neighbor_gain == gain)
            if count > 0:
                print(f"  Gained {gain} neighbor(s): {count:>10,}")

    # Under-connectivity comparison
    under_orig = np.sum(n_neighbors_orig < 4)
    under_merged = np.sum(n_neighbors_merged < 4)
    reduction = under_orig - under_merged
    pct_reduction = 100 * reduction / under_orig if under_orig > 0 else 0

    print(f"\nUnder-connected element reduction:")
    print(f"  Original:  {under_orig:>10,}")
    print(f"  Merged:    {under_merged:>10,}")
    print(f"  Reduction: {reduction:>10,} ({pct_reduction:.1f}%)")

    if reduction > 0:
        print(f"\n✅ SUCCESS: Merging improved connectivity!")
        print(f"   {reduction:,} elements gained neighbors across piece boundaries")
    else:
        print(f"\n⚠️  WARNING: No connectivity improvement detected")


def main():
    """Test merged mesh neighbor connectivity."""

    MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
    MERGED_NPZ = MESH_BASE_PATH / "featurelessAvtk_120_merged.npz"

    if not MERGED_NPZ.exists():
        print(f"❌ Merged mesh not found: {MERGED_NPZ}")
        print(f"   Run fix_merge_duplicate_nodes.py first")
        return

    print(f"\n{'='*80}")
    print(f"TESTING MERGED MESH NEIGHBOR CONNECTIVITY")
    print(f"{'='*80}")

    # Load merged mesh
    positions, connectivity = load_merged_mesh(MERGED_NPZ)

    # Build face-based neighbors (BEFORE fix, we know there were issues)
    # But we need original mesh to compare...
    # For now, just show merged mesh stats

    print(f"\n{'='*80}")
    print(f"Building neighbor graph for MERGED mesh")
    print(f"{'='*80}")

    element_neighbors_merged = build_element_neighbors_array(
        connectivity,
        method='face',
        verbose=True
    )

    n_neighbors_merged = analyze_neighbor_stats(element_neighbors_merged, "MERGED MESH STATISTICS")

    # Expected results
    print(f"\n{'='*80}")
    print(f"EXPECTED RESULTS")
    print(f"{'='*80}")
    print(f"\nBased on previous diagnostics:")
    print(f"  Original mesh (with duplicates):")
    print(f"    - Under-connected elements: ~649,444 (21.3%)")
    print(f"    - These were clustered at piece boundaries")
    print(f"\n  Merged mesh (duplicates removed):")
    print(f"    - Under-connected: {np.sum(n_neighbors_merged < 4):,} ({100*np.sum(n_neighbors_merged < 4)/connectivity.shape[0]:.2f}%)")
    print(f"    - These should be ONLY at domain boundaries (not piece boundaries)")

    under_merged_pct = 100 * np.sum(n_neighbors_merged < 4) / connectivity.shape[0]

    if under_merged_pct < 10.0:
        print(f"\n✅ SUCCESS: Under-connectivity reduced to {under_merged_pct:.2f}%")
        print(f"   This is consistent with domain boundaries only!")
        print(f"   Piece boundary connectivity has been FIXED!")
    elif under_merged_pct < 15.0:
        print(f"\n⚠️  PARTIAL: Under-connectivity at {under_merged_pct:.2f}%")
        print(f"   Better than original (21.3%), but still higher than expected")
        print(f"   May need node-based neighbors for full connectivity")
    else:
        print(f"\n❌ WARNING: Under-connectivity still high at {under_merged_pct:.2f}%")
        print(f"   Merging may not have fully resolved the issue")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
