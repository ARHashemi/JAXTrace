#!/usr/bin/env python3
"""
Test Phase 3+4: Mesh-Aligned Octree GPU with v3 corrections.

Changes from previous version:
- Uses corrected Phase 2 with (morton, level) cell keys
- Updated Phase 4 to search 8 levels (14→13→12→11→10→9→8→7)
- Expected: ~100% searchability with ~11-12 tests per particle

Expected improvements:
- Searchability: 2.4% → ~100% (42× improvement)
- Tests per particle: ~87 → ~11-12 (8× reduction)
- Cells: 652k → 265k (2.5× reduction)
"""

import numpy as np
from pathlib import Path
import time
import jax
import jax.numpy as jnp

print("Importing JAX and modules...")

from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.search.mesh_aligned_point_location import (
    search_mesh_aligned_octree_batch,
    compute_search_statistics,
)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

print(f"{'='*80}")
print("Phase 3+4 Test: Mesh-Aligned Octree GPU v3")
print(f"{'='*80}\n")

# ============================================================================
# Phase 2: Extract octree cells (CPU) - v3 with (morton, level)
# ============================================================================

print(f"{'='*80}")
print("Phase 2: Extracting octree cells v3 (CPU)")
print(f"{'='*80}\n")

MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'

print("Loading mesh...")
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
print(f"  Removed {n_duplicates_removed:,} duplicate nodes\n")

print("Extracting octree cells...")
t0 = time.time()
cells = extract_octree_cells_single(
    node_positions,
    connectivity,
    tolerance=1e-6,
    verbose=False
)
t_extract = time.time() - t0

print(f"  Extraction time: {t_extract:.2f}s")
print(f"  Unique cells: {cells.n_cells:,}")
print(f"  Cells per element: {cells.cells_per_element_mean:.2f}")
print(f"  Elements per cell: {cells.elements_per_cell_mean:.2f}")

# ============================================================================
# Phase 3: Upload octree to GPU
# ============================================================================

print(f"\n{'='*80}")
print("Phase 3: Uploading octree to GPU")
print(f"{'='*80}")

print("Uploading mesh-aligned octree to GPU...")
t0 = time.time()
octree_gpu = upload_mesh_aligned_octree_to_gpu(
    node_positions=node_positions,
    connectivity=connectivity,
    octree_cells=cells
)
t_upload = time.time() - t0

# Estimate GPU memory
gpu_memory_mb = (
    octree_gpu.connectivity.nbytes +
    octree_gpu.node_positions.nbytes +
    octree_gpu.cell_morton_codes.nbytes +
    octree_gpu.cell_levels.nbytes +
    octree_gpu.cell_sizes.nbytes +
    octree_gpu.cell_grid_indices.nbytes +
    octree_gpu.cell_to_elements_offsets.nbytes +
    octree_gpu.cell_to_elements_data.nbytes
) / (1024 * 1024)

print(f"  GPU memory: {gpu_memory_mb:.1f} MB")
print(f"  Cells: {cells.n_cells:,}")
print(f"  Elements: {connectivity.shape[0]:,}")
print(f"  Mean elements/cell: {cells.elements_per_cell_mean:.1f}")
print(f"  CSR data entries: {octree_gpu.cell_to_elements_data.shape[0]:,}")
print(f"  Upload time: {t_upload:.2f}s")

print(f"\n  GPU array shapes:")
print(f"    connectivity: {octree_gpu.connectivity.shape}")
print(f"    node_positions: {octree_gpu.node_positions.shape}")
print(f"    cell_morton_codes: {octree_gpu.cell_morton_codes.shape}")
print(f"    cell_levels: {octree_gpu.cell_levels.shape}")
print(f"    cell_to_elements_offsets: {octree_gpu.cell_to_elements_offsets.shape}")
print(f"    cell_to_elements_data: {octree_gpu.cell_to_elements_data.shape}")

# ============================================================================
# Phase 4: Point Location Test
# ============================================================================

print(f"\n{'='*80}")
print("Phase 4: Point Location Test")
print(f"{'='*80}\n")

# Generate random test particles within mesh bbox
bbox_min = octree_gpu.bbox_min
bbox_max = octree_gpu.bbox_max

n_particles = 10000
print(f"Generating {n_particles:,} random test particles...")
print(f"  Bbox: [{bbox_min[0]:.6f}, {bbox_max[0]:.6f}] × [{bbox_min[1]:.6f}, {bbox_max[1]:.6f}] × [{bbox_min[2]:.6f}, {bbox_max[2]:.6f}]")

np.random.seed(42)
particle_positions_cpu = np.random.uniform(
    low=bbox_min,
    high=bbox_max,
    size=(n_particles, 3)
).astype(np.float32)

print(f"  Generated {n_particles:,} random positions\n")

print("Uploading test positions to GPU...")
particle_positions_gpu = jnp.array(particle_positions_cpu)

# Warm up JIT compilation
print("\nWarming up JIT compilation...")
_ = search_mesh_aligned_octree_batch(
    particle_positions_gpu[:10],
    octree_gpu,
    max_tests=100
)
print("  JIT compilation complete")

# Run search
print(f"\nSearching for {n_particles:,} particles...")
t0 = time.time()
found_elements, n_tests = search_mesh_aligned_octree_batch(
    particle_positions_gpu,
    octree_gpu,
    max_tests=100
)
jax.block_until_ready((found_elements, n_tests))
t_search = time.time() - t0

throughput = n_particles / t_search
print(f"  Search time: {t_search:.3f}s")
print(f"  Throughput: {throughput:,.0f} particles/sec")

# Compute statistics
stats = compute_search_statistics(found_elements, n_tests)

# Print statistics manually (function doesn't accept label parameter)
print("\nMesh-Aligned Octree v3 Search Statistics:")
print(f"{'='*60}")
print(f"  Particles searched: {n_particles:,}")
print(f"  Found: {stats['n_found']:,} ({stats['success_rate']*100:.2f}%)")
print(f"  Point-in-tet tests:")
print(f"    Mean: {stats['mean_tests']:.1f}")
print(f"    Median: {stats['median_tests']:.0f}")
print(f"    Max: {stats['max_tests']:,}")
print(f"{'='*60}")

# ============================================================================
# Validation
# ============================================================================

print(f"\n{'='*80}")
print("Validation")
print(f"{'='*80}\n")

searchability_pct = 100.0 * stats['n_found'] / n_particles
mean_tests = stats['mean_tests']

print(f"  Expected results:")
print(f"    Searchability: ~100%")
print(f"    Mean tests per particle: ~11-12")
print()

print(f"  Actual results:")
print(f"    Searchability: {searchability_pct:.1f}%")
print(f"    Mean tests per particle: {mean_tests:.1f}")
print()

# Validation checks
searchability_ok = searchability_pct >= 95.0
tests_ok = mean_tests <= 20.0

if searchability_pct >= 99.0:
    print(f"  ✅ Excellent searchability: {searchability_pct:.1f}% >= 99%")
elif searchability_ok:
    print(f"  ✅ Good searchability: {searchability_pct:.1f}% >= 95%")
else:
    print(f"  ❌ Searchability too low: {searchability_pct:.1f}% < 95%")

if mean_tests <= 15.0:
    print(f"  ✅ Excellent efficiency: {mean_tests:.1f} tests <= 15")
elif tests_ok:
    print(f"  ✅ Good efficiency: {mean_tests:.1f} tests <= 20")
else:
    print(f"  ⚠  Higher tests than expected: {mean_tests:.1f} > 20")

print(f"\n  Performance analysis:")
print(f"    Phase 2 extraction: {t_extract:.2f}s")
print(f"    Phase 3 GPU upload: {t_upload:.2f}s")
print(f"    Phase 4 search ({n_particles:,} particles): {t_search:.3f}s")
print(f"    Search throughput: {throughput:,.0f} particles/sec")

# Compare to previous versions
print(f"\n  Comparison to previous versions:")
print(f"    v1 (bbox overlap):")
print(f"      Cells: ~652k, Elements/cell: 37.4, Searchability: 2.4%")
print(f"    v3 (morton + level):")
print(f"      Cells: {cells.n_cells:,}, Elements/cell: {cells.elements_per_cell_mean:.1f}, Searchability: {searchability_pct:.1f}%")
print(f"    Improvement:")
print(f"      Tests: {37.4:.1f} → {mean_tests:.1f} ({37.4/mean_tests:.1f}× reduction)")
print(f"      Searchability: 2.4% → {searchability_pct:.1f}% ({searchability_pct/2.4:.1f}× improvement)")

# Final verdict
print(f"\n{'='*80}")
if searchability_ok and tests_ok:
    print("✅ PHASE 3+4 TEST PASSED!")
    print(f"   Searchability: {searchability_pct:.1f}% (target: ≥95%)")
    print(f"   Efficiency: {mean_tests:.1f} tests (target: ≤20)")
else:
    print("❌ PHASE 3+4 TEST FAILED!")
    if not searchability_ok:
        print(f"   Searchability too low: {searchability_pct:.1f}% < 95%")
    if not tests_ok:
        print(f"   Too many tests: {mean_tests:.1f} > 20")
print(f"{'='*80}\n")

# Debug info if searchability is low
if searchability_pct < 95.0:
    print(f"\n{'='*80}")
    print("DEBUG: Searchability Analysis")
    print(f"{'='*80}\n")

    # Check which levels have cells
    unique_levels = np.unique(cells.cell_levels)
    print(f"Refinement levels present in mesh: {sorted(unique_levels)}")

    level_counts = {}
    for level in unique_levels:
        count = np.sum(cells.cell_levels == level)
        level_counts[level] = count

    print(f"\nLevel distribution:")
    for level in sorted(level_counts.keys()):
        count = level_counts[level]
        pct = 100.0 * count / cells.n_cells
        print(f"  Level {level:2d}: {count:8,} cells ({pct:5.2f}%)")

    print(f"\nPhase 4 searches levels: 14, 13, 12, 11, 10, 9, 8, 7")
    print(f"Missing levels in search: {set(unique_levels) - set(range(7, 15))}")

    # Sample some unfound particles
    unfound_mask = found_elements == -1
    n_unfound = np.sum(unfound_mask)
    if n_unfound > 0:
        unfound_positions = particle_positions_cpu[unfound_mask]
        print(f"\nSample unfound particle positions (first 5):")
        for i in range(min(5, n_unfound)):
            pos = unfound_positions[i]
            print(f"  [{pos[0]:.8f}, {pos[1]:.8f}, {pos[2]:.8f}]")

    print(f"\n{'='*80}\n")
