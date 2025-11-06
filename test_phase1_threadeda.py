"""
Phase 1 ThreadedA Integration Test

Tests element-to-block assignment on the actual ThreadedA mesh.
"""

import numpy as np
from pathlib import Path
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from jaxtrace.gpu.forest.block_grid import create_regular_grid
from jaxtrace.gpu.forest.block_mapper import assign_elements_to_block_list, validate_assignment


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
    print("Phase 1: ThreadedA Element-to-Block Assignment Test")
    print("=" * 80)

    # Load ThreadedA mesh
    print("\nLoading ThreadedA mesh...")
    nodes, connectivity = load_threadeda_mesh()
    
    print(f"  Nodes: {nodes.shape[0]:,}")
    print(f"  Elements: {connectivity.shape[0]:,}")
    print(f"  Element type: {connectivity.shape[1]} nodes/element")
    
    # Create block grid (from Phase 0 analysis)
    domain_bounds = np.array([-0.03, 0.03, -0.023, 0.023, -0.01, 0.0], dtype=np.float32)
    grid_size = (4, 4, 2)  # 32 blocks
    
    print(f"\nCreating block grid: {grid_size[0]}×{grid_size[1]}×{grid_size[2]} = {grid_size[0]*grid_size[1]*grid_size[2]} blocks")
    blocks = create_regular_grid(domain_bounds, grid_size)
    print(f"  Block 0 bounds: {blocks[0].bounds}")
    print(f"  Block 0 volume: {blocks[0].volume:.6e}")
    
    # Assign elements to blocks
    print(f"\nAssigning {connectivity.shape[0]:,} elements to {len(blocks)} blocks...")
    element_to_block, stats = assign_elements_to_block_list(
        nodes, connectivity, blocks,
        heavy_threshold=10000,
        verbose=True
    )
    
    # Print detailed statistics
    print("\n" + "=" * 80)
    print("PHASE 1 RESULTS: Element-to-Block Assignment Statistics")
    print("=" * 80)
    print(stats)
    
    # Verify against Phase 0 predictions
    print("\n" + "-" * 80)
    print("Comparison with Phase 0 Predictions:")
    print("-" * 80)
    print(f"  Phase 0 predicted ~110K elements/block")
    print(f"  Actual mean: {stats.mean_elements:,.0f}")
    print(f"  Actual median: {stats.median_elements:,.0f}")
    print(f"  Phase 0 predicted 8.6x imbalance")
    print(f"  Actual imbalance: {stats.imbalance_ratio:.2f}x")
    
    # Check elements outside domain
    n_outside = np.sum(element_to_block == -1)
    pct_outside = 100 * n_outside / connectivity.shape[0]
    print(f"\n  Elements outside domain: {n_outside} ({pct_outside:.4f}%)")
    
    if pct_outside > 0.1:
        print(f"  ⚠️  WARNING: {pct_outside:.2f}% elements outside domain")
    else:
        print(f"  ✅ Good: <0.1% elements outside domain")
    
    # Heavy block analysis
    print("\n" + "-" * 80)
    print("Heavy Block Analysis (>10,000 elements):")
    print("-" * 80)
    
    if stats.heavy_blocks:
        print(f"  Found {len(stats.heavy_blocks)} heavy blocks:")
        for bid in sorted(stats.heavy_blocks):
            count = stats.elements_per_block[bid]
            print(f"    Block {bid:2d}: {count:,} elements")
        print(f"\n  ⚡ These blocks will use hash bucket search in Phase 4")
    else:
        print(f"  ✅ No heavy blocks detected")
    
    # Validate assignment
    print("\n" + "-" * 80)
    print("Validation:")
    print("-" * 80)
    print("  Checking element centroid containment (1000 samples)...")
    valid = validate_assignment(element_to_block, nodes, connectivity, blocks, n_samples=1000)
    
    if valid:
        print("  ✅ VALIDATION PASSED")
    else:
        print("  ❌ VALIDATION FAILED")
        return False
    
    # Memory estimation
    print("\n" + "-" * 80)
    print("Memory Estimation:")
    print("-" * 80)
    max_elem = stats.max_elements
    memory_mb = (32 * max_elem * 4) / (1024**2)  # int32 padded arrays
    print(f"  Max elements per block: {max_elem:,}")
    print(f"  Padded array size: (32, {max_elem:,})")
    print(f"  Memory for element IDs: {memory_mb:.1f} MB")
    print(f"  Target: <500 MB total")
    
    if memory_mb < 100:
        print(f"  ✅ Excellent: {memory_mb:.1f} MB << 500 MB target")
    elif memory_mb < 250:
        print(f"  ✅ Good: {memory_mb:.1f} MB < 500 MB target")
    else:
        print(f"  ⚠️  Warning: {memory_mb:.1f} MB approaching 500 MB target")
    
    # Success summary
    print("\n" + "=" * 80)
    print("PHASE 1: SUCCESS")
    print("=" * 80)
    print("✅ Block grid created (32 blocks)")
    print(f"✅ Elements assigned ({connectivity.shape[0]:,} elements)")
    print(f"✅ Statistics computed (imbalance: {stats.imbalance_ratio:.2f}x)")
    print(f"✅ Heavy blocks identified ({len(stats.heavy_blocks)} blocks)")
    print("✅ Assignment validated")
    print("\nReady to proceed to Phase 2: Element Neighbors & Padded Block Arrays")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
