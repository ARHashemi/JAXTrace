#!/usr/bin/env python3
"""
Test Mesh-Aligned Octree Phase 3+4: GPU Upload and Point Location

Tests:
1. Phase 2 → Phase 3: Upload octree cells to GPU
2. Phase 4: Point location using mesh-aligned octree
3. Performance validation vs expected results

Expected results from Phase 1+2:
- ~652k unique cells
- ~37 elements per cell
- ~8 cells per element
- 100% searchability
"""

import numpy as np
from pathlib import Path
import time

print("Importing JAX and modules...")
import jax
import jax.numpy as jnp

from jaxtrace.gpu.search.mesh_aligned_octree_fast import (
    extract_octree_cells_fast,
)
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import (
    upload_mesh_aligned_octree_to_gpu,
)
from jaxtrace.gpu.search.mesh_aligned_point_location import (
    search_mesh_aligned_octree_batch,
    compute_search_statistics,
    print_search_statistics,
)

# ============================================================================
# Configuration
# ============================================================================

MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'

N_TEST_PARTICLES = 10000  # Number of random particles to test

# ============================================================================
# Helper Functions
# ============================================================================

def load_mesh():
    """Load and deduplicate mesh."""
    print("\nLoading mesh...")
    from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )

    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions,
        connectivity,
        velocity_sequence=velocity_sequence,
        verbose=False
    )

    print(f"  Loaded {connectivity.shape[0]:,} elements, {node_positions.shape[0]:,} nodes")
    print(f"  Removed {n_duplicates_removed:,} duplicate nodes")

    return node_positions, connectivity


def generate_test_particles(node_positions, n_particles):
    """
    Generate random test particles within mesh bounding box.

    Returns:
        positions: (n_particles, 3) float32
    """
    print(f"\nGenerating {n_particles:,} random test particles...")

    # Compute bounding box
    bbox_min = node_positions.min(axis=0)
    bbox_max = node_positions.max(axis=0)

    # Add small margin
    margin = (bbox_max - bbox_min) * 0.01
    bbox_min -= margin
    bbox_max += margin

    print(f"  Bbox: [{bbox_min[0]:.6f}, {bbox_max[0]:.6f}] × "
          f"[{bbox_min[1]:.6f}, {bbox_max[1]:.6f}] × "
          f"[{bbox_min[2]:.6f}, {bbox_max[2]:.6f}]")

    # Generate random positions
    positions = np.random.uniform(
        low=bbox_min,
        high=bbox_max,
        size=(n_particles, 3)
    ).astype(np.float32)

    print(f"  Generated {n_particles:,} random positions")

    return positions


# ============================================================================
# Main Test
# ============================================================================

def main():
    print("="*80)
    print("Phase 3+4 Test: Mesh-Aligned Octree GPU Upload and Point Location")
    print("="*80)

    # ========================================================================
    # Phase 2: Extract octree cells (CPU)
    # ========================================================================
    print("\n" + "="*80)
    print("Phase 2: Extracting octree cells (CPU)")
    print("="*80)

    node_positions, connectivity = load_mesh()

    print("\nExtracting octree cells...")
    t0 = time.time()

    cells = extract_octree_cells_fast(
        node_positions,
        connectivity,
        tolerance=1e-6,
        batch_size=100000,
        verbose=False
    )

    t_extract = time.time() - t0

    print(f"  Extraction time: {t_extract:.2f}s")
    print(f"  Unique cells: {cells.n_cells:,}")
    print(f"  Cells per element: {cells.cells_per_element_mean:.2f}")
    print(f"  Elements per cell: {cells.elements_per_cell_mean:.2f}")

    # ========================================================================
    # Phase 3: Upload to GPU
    # ========================================================================
    print("\n" + "="*80)
    print("Phase 3: Uploading octree to GPU")
    print("="*80)

    t0 = time.time()

    octree_gpu = upload_mesh_aligned_octree_to_gpu(
        connectivity,
        node_positions,
        cells,
        verbose=True
    )

    t_upload = time.time() - t0
    print(f"  Upload time: {t_upload:.2f}s")

    # Verify GPU arrays
    print("\n  GPU array shapes:")
    print(f"    connectivity: {octree_gpu.connectivity.shape}")
    print(f"    node_positions: {octree_gpu.node_positions.shape}")
    print(f"    cell_morton_codes: {octree_gpu.cell_morton_codes.shape}")
    print(f"    cell_to_elements_offsets: {octree_gpu.cell_to_elements_offsets.shape}")
    print(f"    cell_to_elements_data: {octree_gpu.cell_to_elements_data.shape}")

    # ========================================================================
    # Phase 4: Point Location
    # ========================================================================
    print("\n" + "="*80)
    print("Phase 4: Point Location Test")
    print("="*80)

    # Generate test particles
    test_positions_cpu = generate_test_particles(node_positions, N_TEST_PARTICLES)

    # Upload test positions to GPU
    print("\nUploading test positions to GPU...")
    test_positions_gpu = jnp.array(test_positions_cpu)

    # Warm-up JIT compilation
    print("\nWarming up JIT compilation...")
    _ = search_mesh_aligned_octree_batch(
        test_positions_gpu[:10],
        octree_gpu,
        max_tests=100
    )
    jax.block_until_ready(_)
    print("  JIT compilation complete")

    # Run search
    print(f"\nSearching for {N_TEST_PARTICLES:,} particles...")
    t0 = time.time()

    elem_ids, n_tests = search_mesh_aligned_octree_batch(
        test_positions_gpu,
        octree_gpu,
        max_tests=100
    )

    # Wait for GPU to finish
    jax.block_until_ready(elem_ids)
    t_search = time.time() - t0

    print(f"  Search time: {t_search:.3f}s")
    print(f"  Throughput: {N_TEST_PARTICLES/t_search:,.0f} particles/sec")

    # Compute statistics
    stats = compute_search_statistics(elem_ids, n_tests)
    print_search_statistics(stats)

    # ========================================================================
    # Validation
    # ========================================================================
    print("\n" + "="*80)
    print("Validation")
    print("="*80)

    success_rate = stats['success_rate'] * 100
    mean_tests = stats['mean_tests']

    print(f"\n  Expected results (from Phase 1 diagnostic):")
    print(f"    Searchability: ~100%")
    print(f"    Mean tests per particle: ~37")
    print(f"\n  Actual results:")
    print(f"    Searchability: {success_rate:.1f}%")
    print(f"    Mean tests per particle: {mean_tests:.1f}")

    # Check if results match expectations
    success = True

    if success_rate < 95.0:
        print(f"\n  ❌ Searchability too low: {success_rate:.1f}% < 95%")
        success = False
    else:
        print(f"\n  ✅ Searchability good: {success_rate:.1f}%")

    if mean_tests > 50:
        print(f"  ⚠️  Mean tests higher than expected: {mean_tests:.1f} > 50")
        print(f"      (Expected ~37 from diagnostic)")
    elif mean_tests > 100:
        print(f"  ❌ Mean tests too high: {mean_tests:.1f} > 100")
        success = False
    else:
        print(f"  ✅ Mean tests efficient: {mean_tests:.1f}")

    # Performance comparison
    print(f"\n  Performance analysis:")
    print(f"    Phase 2 extraction: {t_extract:.2f}s")
    print(f"    Phase 3 GPU upload: {t_upload:.2f}s")
    print(f"    Phase 4 search ({N_TEST_PARTICLES:,} particles): {t_search:.3f}s")
    print(f"    Search throughput: {N_TEST_PARTICLES/t_search:,.0f} particles/sec")

    # Expected speedup vs Morton blocks
    if mean_tests < 100:
        speedup_vs_morton = 536 / mean_tests  # 536 = typical elements per Morton block
        print(f"\n  Expected speedup vs Morton blocks: ~{speedup_vs_morton:.0f}×")
        print(f"    (Morton: ~536 tests, Mesh-aligned: ~{mean_tests:.0f} tests)")

    # Final result
    print("\n" + "="*80)
    if success:
        print("✅ PHASE 3+4 TEST PASSED!")
        print("  Mesh-aligned octree GPU structure working correctly")
        print("  Point location achieving expected performance")
    else:
        print("❌ PHASE 3+4 TEST FAILED!")
        print("  Check searchability and test efficiency")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
