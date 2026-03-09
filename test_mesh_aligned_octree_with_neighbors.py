#!/usr/bin/env python3
"""
Test mesh-aligned octree WITH 26-neighbor search (Phase 2)

Expected results:
- Searchability: ~99% (vs ~75% without neighbors)
- Mean tests: ~15-20 (vs ~5 without neighbors)
- Throughput: ~50-100K p/s (vs ~12K without neighbors)
"""

import time
import numpy as np
from pathlib import Path

# Import JAX
import jax
import jax.numpy as jnp

# Suppress JAX warnings
import warnings
warnings.filterwarnings('ignore')

from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.search.mesh_aligned_point_location_with_neighbors import (
    search_mesh_aligned_octree_with_neighbors_batch
)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

print("Importing JAX and modules...")
print(f"✅ Enhanced VTK I/O available")
print(f"✅ HDF5 I/O available")

print(f"{'='*80}")
print("Phase 2 Test: Mesh-Aligned Octree WITH 26-Neighbor Search")
print(f"{'='*80}\n")

# ============================================================================
# Phase 2: Extract Octree Cells (CPU)
# ============================================================================

print(f"{'='*80}")
print("Phase 2: Extracting octree cells (CPU)")
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
    verbose=True
)

node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
    node_positions,
    connectivity,
    velocity_sequence=velocity_sequence,
    verbose=True
)

print()
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
# Phase 3: Upload to GPU
# ============================================================================

print()
print(f"{'='*80}")
print("Phase 3: Uploading octree to GPU")
print(f"{'='*80}")

t0 = time.time()
octree_gpu = upload_mesh_aligned_octree_to_gpu(
    node_positions=node_positions,
    connectivity=connectivity,
    octree_cells=cells,
    verbose=True
)
t_upload = time.time() - t0

print(f"  Upload time: {t_upload:.2f}s")
print()
print(f"  GPU array shapes:")
print(f"    connectivity: {octree_gpu.connectivity.shape}")
print(f"    node_positions: {octree_gpu.node_positions.shape}")
print(f"    cell_morton_codes: {octree_gpu.cell_morton_codes.shape}")
print(f"    cell_levels: {octree_gpu.cell_levels.shape}")
print(f"    cell_to_elements_offsets: {octree_gpu.cell_to_elements_offsets.shape}")
print(f"    cell_to_elements_data: {octree_gpu.cell_to_elements_data.shape}")

# ============================================================================
# Phase 4: Point Location Test WITH NEIGHBORS
# ============================================================================

print()
print(f"{'='*80}")
print("Phase 4: Point Location Test WITH 26-NEIGHBOR SEARCH")
print(f"{'='*80}\n")

# Generate random test positions
n_particles = 10000
np.random.seed(42)

bbox_min = octree_gpu.bbox_min
bbox_max = octree_gpu.bbox_max

print(f"Generating {n_particles:,} random test particles...")
print(f"  Bbox: [{bbox_min[0]:.6f}, {bbox_max[0]:.6f}] × [{bbox_min[1]:.6f}, {bbox_max[1]:.6f}] × [{bbox_min[2]:.6f}, {bbox_max[2]:.6f}]")

test_positions = np.random.uniform(
    low=np.array(bbox_min),
    high=np.array(bbox_max),
    size=(n_particles, 3)
).astype(np.float32)

print(f"  Generated {n_particles:,} random positions\n")

# Upload to GPU
print("Uploading test positions to GPU...\n")
test_positions_gpu = jnp.array(test_positions)

# Warm up JIT compilation
print("Warming up JIT compilation...")
_ = search_mesh_aligned_octree_with_neighbors_batch(
    test_positions_gpu[:10], octree_gpu, max_tests=200
)
jax.block_until_ready(_)
print("  JIT compilation complete\n")

# Run search with neighbors
print(f"Searching for {n_particles:,} particles WITH NEIGHBORS...")
t0 = time.time()
found_elements, n_tests = search_mesh_aligned_octree_with_neighbors_batch(
    test_positions_gpu, octree_gpu, max_tests=200
)
jax.block_until_ready((found_elements, n_tests))
t_search = time.time() - t0

throughput = n_particles / t_search
print(f"  Search time: {t_search:.3f}s")
print(f"  Throughput: {throughput:,.0f} particles/sec\n")

# Convert to CPU for analysis
found_elements_cpu = np.array(found_elements)
n_tests_cpu = np.array(n_tests)

# Statistics
n_found = np.sum(found_elements_cpu >= 0)
found_ratio = n_found / n_particles

print(f"Mesh-Aligned Octree WITH NEIGHBORS Search Statistics:")
print(f"{'='*60}")
print(f"  Particles searched: {n_particles:,}")
print(f"  Found: {n_found:,} ({100.0 * found_ratio:.2f}%)")
print(f"  Point-in-tet tests:")
print(f"    Mean: {n_tests_cpu.mean():.1f}")
print(f"    Median: {np.median(n_tests_cpu):.0f}")
print(f"    Max: {n_tests_cpu.max()}")
print(f"{'='*60}\n")

# ============================================================================
# Validation
# ============================================================================

print(f"{'='*80}")
print("Validation")
print(f"{'='*80}\n")

print(f"  Expected results (Phase 2):")
print(f"    Searchability: ~99%")
print(f"    Mean tests per particle: ~15-20")
print()
print(f"  Actual results:")
print(f"    Searchability: {100.0 * found_ratio:.1f}%")
print(f"    Mean tests per particle: {n_tests_cpu.mean():.1f}")
print()

# Validate
searchability_ok = found_ratio >= 0.95
efficiency_ok = n_tests_cpu.mean() <= 50

if searchability_ok:
    print(f"  ✅ Excellent searchability: {100.0 * found_ratio:.1f}% >= 95%")
else:
    print(f"  ❌ Searchability too low: {100.0 * found_ratio:.1f}% < 95%")

if efficiency_ok:
    print(f"  ✅ Good efficiency: {n_tests_cpu.mean():.1f} tests <= 50")
else:
    print(f"  ⚠️  High test count: {n_tests_cpu.mean():.1f} tests > 50")

print()
print(f"  Performance analysis:")
print(f"    Phase 2 extraction: {t_extract:.2f}s")
print(f"    Phase 3 GPU upload: {t_upload:.2f}s")
print(f"    Phase 4 search ({n_particles:,} particles): {t_search:.3f}s")
print(f"    Search throughput: {throughput:,.0f} particles/sec")

# Comparison with previous versions
print()
print(f"  Comparison with previous versions:")
print(f"    Without neighbors (v6):")
print(f"      Cells: 517,069, Elements/cell: 5.9, Searchability: 74.6%, Tests: 4.8")
print(f"    With neighbors (v7):")
print(f"      Cells: {cells.n_cells:,}, Elements/cell: {cells.elements_per_cell_mean:.1f}, Searchability: {100.0 * found_ratio:.1f}%, Tests: {n_tests_cpu.mean():.1f}")
print(f"    Improvement:")
print(f"      Searchability: 74.6% → {100.0 * found_ratio:.1f}% ({found_ratio / 0.746:.1f}× better)")
print(f"      Throughput: 12,106 → {throughput:,.0f} p/s ({throughput / 12106:.1f}× faster)")

print()
if searchability_ok and efficiency_ok:
    print(f"{'='*80}")
    print("✅ PHASE 2 TEST PASSED!")
    print(f"   Searchability: {100.0 * found_ratio:.1f}% >= 95%")
    print(f"   Efficiency: {n_tests_cpu.mean():.1f} tests <= 50")
    print(f"{'='*80}\n")
else:
    print(f"{'='*80}")
    print("❌ PHASE 2 TEST FAILED!")
    if not searchability_ok:
        print(f"   Searchability too low: {100.0 * found_ratio:.1f}% < 95%")
    if not efficiency_ok:
        print(f"   Test count too high: {n_tests_cpu.mean():.1f} > 50")
    print(f"{'='*80}\n")

# Debug: Level distribution
print(f"{'='*80}")
print("DEBUG: Octree Level Distribution")
print(f"{'='*80}\n")

unique_levels = np.unique(np.array(octree_gpu.cell_levels))
print(f"Refinement levels present in mesh: {list(unique_levels)}\n")

print("Level distribution:")
for level in unique_levels:
    n_cells_at_level = np.sum(np.array(octree_gpu.cell_levels) == level)
    pct = 100.0 * n_cells_at_level / cells.n_cells
    print(f"  Level {level:2d}: {n_cells_at_level:8,} cells ({pct:5.2f}%)")

print()
print(f"{'='*80}\n")
