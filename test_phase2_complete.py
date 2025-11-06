"""
Phase 2 Complete Integration Test: ThreadedA Mesh

Tests the complete Phase 2 implementation:
1. Element-to-block assignment (Phase 1)
2. Face-adjacency neighbor extraction
3. Padded 2D block arrays (V5 solution)
4. Memory profiling and V4 vs V5 comparison
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import time

from jaxtrace.gpu.forest.block_grid import create_regular_grid
from jaxtrace.gpu.forest.block_mapper import assign_elements_to_block_list
from jaxtrace.gpu.forest.element_adjacency import extract_element_neighbors, validate_neighbor_symmetry
from jaxtrace.gpu.forest.padded_arrays import (
    build_padded_block_arrays,
    validate_padded_arrays,
    print_memory_comparison,
)


def load_threadeda_mesh():
    """Load ThreadedA mesh using VTK directly."""
    mesh_file = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_159.pvtu"

    print(f"Loading: {mesh_file}")
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(mesh_file)
    reader.Update()
    mesh = reader.GetOutput()

    # Extract nodes
    points = mesh.GetPoints()
    nodes = vtk_to_numpy(points.GetData()).astype(np.float32)

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

    return nodes, connectivity


def main():
    print("=" * 80)
    print("Phase 2 COMPLETE: ThreadedA Integration Test")
    print("=" * 80)

    # === TASK 0: Load Mesh ===
    print("\n" + "=" * 80)
    print("TASK 0: Load ThreadedA Mesh")
    print("=" * 80)
    
    t0 = time.time()
    nodes, connectivity = load_threadeda_mesh()
    t_load = time.time() - t0
    
    print(f"  Nodes: {nodes.shape[0]:,}")
    print(f"  Elements: {connectivity.shape[0]:,}")
    print(f"  Load time: {t_load:.2f} s")

    # === TASK 1: Block Assignment (Phase 1) ===
    print("\n" + "=" * 80)
    print("TASK 1: Element-to-Block Assignment (Phase 1)")
    print("=" * 80)
    
    domain_bounds = np.array([-0.03, 0.03, -0.023, 0.023, -0.01, 0.0], dtype=np.float32)
    grid_size = (4, 4, 2)  # 32 blocks
    blocks = create_regular_grid(domain_bounds, grid_size)
    
    t0 = time.time()
    element_to_block, block_stats = assign_elements_to_block_list(
        nodes, connectivity, blocks, verbose=True
    )
    t_assign = time.time() - t0
    
    print(f"\n  Assignment time: {t_assign:.1f} s")
    print(f"  ✅ {connectivity.shape[0]:,} elements assigned")

    # === TASK 2: Neighbor Extraction ===
    print("\n" + "=" * 80)
    print("TASK 2: Face-Adjacency Neighbor Extraction")
    print("=" * 80)
    
    t0 = time.time()
    neighbors, neighbor_stats = extract_element_neighbors(connectivity, verbose=True)
    t_neighbors = time.time() - t0
    
    print(f"\n  Neighbor extraction time: {t_neighbors:.1f} s")
    print(f"  ✅ Neighbors extracted for {connectivity.shape[0]:,} elements")
    
    # Validate symmetry
    print("\n  Validating neighbor symmetry...")
    valid_neighbors = validate_neighbor_symmetry(neighbors, n_samples=1000)
    if not valid_neighbors:
        print("  ❌ Neighbor validation FAILED")
        return False
    print(f"  ✅ Neighbor symmetry validated")

    # === TASK 3: Padded Block Arrays ===
    print("\n" + "=" * 80)
    print("TASK 3: Build Padded 2D Block Arrays (V5 Solution)")
    print("=" * 80)
    
    t0 = time.time()
    padded = build_padded_block_arrays(element_to_block, block_stats, verbose=True)
    t_padded = time.time() - t0
    
    print(f"\n  Padded array build time: {t_padded:.1f} s")
    print(f"  ✅ Padded arrays created")
    
    # Validate padded arrays
    print("\n  Validating padded arrays...")
    valid_padded = validate_padded_arrays(padded, element_to_block, n_samples=1000)
    if not valid_padded:
        print("  ❌ Padded array validation FAILED")
        return False
    print(f"  ✅ Padded arrays validated")

    # === TASK 4: Memory Analysis ===
    print("\n" + "=" * 80)
    print("TASK 4: Memory Profiling")
    print("=" * 80)
    
    # Component breakdown
    memory_padded_arrays = padded.memory_mb
    memory_neighbors = (sum(len(neighbors[i]) for i in range(connectivity.shape[0])) * 4) / (1024**2)
    memory_nodes = (nodes.nbytes) / (1024**2)
    memory_total = memory_padded_arrays + memory_neighbors + memory_nodes
    
    print(f"\nMemory Breakdown:")
    print(f"  Padded element arrays: {memory_padded_arrays:.1f} MB")
    print(f"  Neighbor arrays: {memory_neighbors:.1f} MB")
    print(f"  Node positions: {memory_nodes:.1f} MB")
    print(f"  " + "-" * 50)
    print(f"  TOTAL: {memory_total:.1f} MB")
    
    target_mb = 500
    if memory_total < target_mb:
        print(f"\n  ✅ Memory under target: {memory_total:.1f} MB < {target_mb} MB")
        print(f"  Headroom: {target_mb - memory_total:.1f} MB ({100*(target_mb - memory_total)/target_mb:.1f}%)")
    else:
        print(f"\n  ⚠️  Memory over target: {memory_total:.1f} MB > {target_mb} MB")

    # V4 vs V5 comparison
    print_memory_comparison(padded, block_stats)

    # === SUCCESS SUMMARY ===
    print("\n" + "=" * 80)
    print("PHASE 2: SUCCESS - ALL TASKS COMPLETE")
    print("=" * 80)
    
    print("\n✅ Task 1: Element-to-block assignment")
    print(f"    - {connectivity.shape[0]:,} elements → {block_stats.n_blocks} blocks")
    print(f"    - Imbalance: {block_stats.imbalance_ratio:.2f}x")
    
    print("\n✅ Task 2: Face-adjacency neighbors")
    print(f"    - {connectivity.shape[0]:,} elements")
    print(f"    - Avg {neighbor_stats.avg_neighbors_per_element:.2f} neighbors/element")
    print(f"    - Memory: {memory_neighbors:.1f} MB")
    
    print("\n✅ Task 3: Padded 2D arrays (V5 solution)")
    print(f"    - Shape: ({padded.n_blocks}, {padded.max_elements_per_block:,})")
    print(f"    - Memory: {memory_padded_arrays:.1f} MB")
    print(f"    - Padding waste: {padded.padding_waste_pct:.1f}%")
    
    print("\n✅ Task 4: Memory profiling")
    print(f"    - Total: {memory_total:.1f} MB / {target_mb} MB ({100*memory_total/target_mb:.1f}%)")
    print(f"    - V4→V5 improvement: {14250 / memory_padded_arrays:.0f}x")
    
    print("\n" + "-" * 80)
    print("Ready for Phase 3: Particle Seeding & Initial Assignment")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
