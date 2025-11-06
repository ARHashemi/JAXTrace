"""
Phase 2: Test element adjacency extraction on ThreadedA mesh.

Tests face-adjacency neighbor extraction on the 3.5M element mesh.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import time

from jaxtrace.gpu.forest.element_adjacency import (
    extract_element_neighbors,
    validate_neighbor_symmetry,
)


def load_threadeda_mesh():
    """Load ThreadedA mesh using VTK directly."""
    mesh_file = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_159.pvtu"

    print(f"Loading: {mesh_file}")
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(mesh_file)
    reader.Update()
    mesh = reader.GetOutput()

    # Extract connectivity
    cells = mesh.GetCells()
    connectivity_vtk = vtk_to_numpy(cells.GetConnectivityArray()).astype(np.int32)
    offsets = vtk_to_numpy(cells.GetOffsetsArray()).astype(np.int32)

    # Reshape connectivity for tetrahedral elements
    n_elements = len(offsets) - 1
    connectivity = np.zeros((n_elements, 4), dtype=np.int32)
    for i in range(n_elements):
        start = offsets[i]
        end = offsets[i + 1]
        connectivity[i] = connectivity_vtk[start:end]

    return connectivity


def main():
    print("=" * 80)
    print("Phase 2: ThreadedA Element Adjacency Extraction Test")
    print("=" * 80)

    # Load mesh
    print("\nLoading ThreadedA mesh...")
    t0 = time.time()
    connectivity = load_threadeda_mesh()
    t_load = time.time() - t0
    
    print(f"  Elements: {connectivity.shape[0]:,}")
    print(f"  Load time: {t_load:.2f} s")

    # Extract neighbors
    print("\nExtracting element face-adjacency neighbors...")
    print("(This will take a few minutes for 3.5M elements...)")
    t0 = time.time()
    neighbors, stats = extract_element_neighbors(connectivity, verbose=True)
    t_extract = time.time() - t0

    print(f"\nExtraction time: {t_extract:.1f} s ({connectivity.shape[0]/t_extract:,.0f} elements/s)")

    # Print detailed statistics
    print("\n" + "=" * 80)
    print("ADJACENCY STATISTICS")
    print("=" * 80)
    print(stats)

    # Analyze neighbor distribution
    print("\n" + "-" * 80)
    print("Neighbor Distribution Analysis:")
    print("-" * 80)
    
    neighbor_counts = [len(neighbors[i]) for i in range(connectivity.shape[0])]
    count_distribution = {}
    for count in neighbor_counts:
        count_distribution[count] = count_distribution.get(count, 0) + 1
    
    print("Elements by neighbor count:")
    for n_neighbors in sorted(count_distribution.keys()):
        count = count_distribution[n_neighbors]
        pct = 100 * count / connectivity.shape[0]
        print(f"  {n_neighbors} neighbors: {count:,} elements ({pct:.2f}%)")

    # Validate symmetry
    print("\n" + "-" * 80)
    print("Validation:")
    print("-" * 80)
    print("Checking neighbor symmetry (1000 samples)...")
    valid = validate_neighbor_symmetry(neighbors, n_samples=1000)

    if valid:
        print("✅ VALIDATION PASSED")
    else:
        print("❌ VALIDATION FAILED")
        return False

    # Memory estimation
    print("\n" + "-" * 80)
    print("Memory Estimation for Neighbor Storage:")
    print("-" * 80)
    
    # Variable-length storage (current)
    total_neighbors = sum(len(neighbors[i]) for i in range(connectivity.shape[0]))
    var_length_mb = (total_neighbors * 4) / (1024**2)  # int32
    print(f"  Variable-length storage: {var_length_mb:.1f} MB")
    print(f"  Total neighbor pairs: {total_neighbors:,}")
    print(f"  Avg neighbors/element: {total_neighbors / connectivity.shape[0]:.2f}")

    # Fixed padded storage would be much larger (not used for neighbors)
    max_neighbors = stats.max_neighbors_per_element
    padded_mb = (connectivity.shape[0] * max_neighbors * 4) / (1024**2)
    print(f"\n  Padded array (not used): {padded_mb:.1f} MB")
    print(f"  Padding waste: {padded_mb - var_length_mb:.1f} MB ({100*(padded_mb - var_length_mb)/padded_mb:.1f}%)")

    # Success
    print("\n" + "=" * 80)
    print("PHASE 2 TASK 1: SUCCESS")
    print("=" * 80)
    print(f"✅ Neighbor extraction complete ({connectivity.shape[0]:,} elements)")
    print(f"✅ Statistics computed")
    print(f"✅ Symmetry validated")
    print(f"✅ Memory: {var_length_mb:.1f} MB (variable-length storage)")
    print("\nReady for Task 2: Build padded block arrays")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
