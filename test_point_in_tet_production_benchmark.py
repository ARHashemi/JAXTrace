#!/usr/bin/env python3
"""
Point-in-Tetrahedron PRODUCTION Benchmark

Tests point-in-tet methods in REALISTIC production scenario:
- Load real mesh with Morton octree
- Seed particles using production logic
- Run cascading initial assignment (radius L2 search)
- Benchmark 6 methods: current, skala, axis_aligned (old), pure_aa, skala_memory_opt, branchless_hybrid (corrected)
- Compare assignment success rate and timing

This tests the ACTUAL production use case where point-in-tet is called
millions of times within L2 Morton radius search during initial assignment.
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.tracking.initial_assignment_cascading import initial_assignment_cascading_fallback
from jaxtrace.tracking.seeding import uniform_grid_seeds
from jaxtrace.gpu.search.aa_detection import (
    precompute_aa_metadata,
    precompute_element_vertices,
)
from jaxtrace.gpu.search.point_in_tet_methods import set_corrected_metadata
import jaxtrace.config as config


# ============================================================================
# Configuration (same as production script)
# ============================================================================

MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 121)  # Load just ONE timestep for testing
VELOCITY_FIELD_NAME = 'Displacement'

# Particle seeding (same as production)
PARTICLE_GRID_RESOLUTION = (20, 50, 30)  # 225,000 particles
PARTICLE_BOUNDS_FRACTION = {
    'x': (0.3, 0.7),
    'y': (0.2, 0.8),
    'z': (0.3, 1.0),
}

# Initial assignment (same as production)
INITIAL_SEARCH_RADIUS = 500
INITIAL_SEARCH_FALLBACK_RADII = [1000, 2000, 5000, 10000, 100000]

SEED = 42


def main():
    print("=" * 80)
    print("Point-in-Tetrahedron PRODUCTION Benchmark")
    print("Testing within REAL initial assignment (cascading radius search)")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 80)

    # ========================================================================
    # 1. Load Mesh (same as production)
    # ========================================================================

    print("\n[1/7] Loading mesh from PVTU...")
    t_load = time.time()
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )
    t_load = time.time() - t_load

    n_nodes_orig = node_positions.shape[0]
    n_elements = connectivity.shape[0]

    print(f"  Mesh loaded in {t_load:.2f}s")
    print(f"  Elements: {n_elements:,}, Nodes: {n_nodes_orig:,}")

    # ========================================================================
    # 2. Deduplicate Nodes (same as production)
    # ========================================================================

    print(f"\n[2/7] Deduplicating nodes...")
    t_dedup = time.time()
    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    t_dedup = time.time() - t_dedup

    n_nodes = node_positions.shape[0]
    print(f"  Removed {n_duplicates_removed:,} duplicates in {t_dedup:.2f}s")
    print(f"  Nodes: {n_nodes:,}")

    # ========================================================================
    # 3. Precompute Corrected AA Metadata
    # ========================================================================

    print(f"\n[3/9] Precomputing corrected AA metadata...")
    t_aa_start = time.time()
    aa_metadata = precompute_aa_metadata(connectivity, node_positions, verbose=True)
    t_aa_elapsed = time.time() - t_aa_start
    print(f"  AA metadata precomputed in {t_aa_elapsed:.2f}s")

    n_aa_elements = int(np.sum(aa_metadata.is_axis_aligned))
    aa_percentage = (n_aa_elements / n_elements) * 100
    print(f"  Axis-aligned elements: {n_aa_elements:,}/{n_elements:,} ({aa_percentage:.2f}%)")

    if aa_percentage >= 99.9:
        print(f"  ✅ Mesh is 100% axis-aligned - pure_aa method will be optimal")
    elif aa_percentage >= 50.0:
        print(f"  ⚠️  Mesh is mixed ({aa_percentage:.1f}% AA) - branchless_hybrid recommended")
    else:
        print(f"  ⚠️  Mesh is mostly non-AA ({aa_percentage:.1f}% AA) - skala_memory_opt recommended")

    # ========================================================================
    # 4. Precompute Element Vertices (Memory Optimization)
    # ========================================================================

    print(f"\n[4/9] Precomputing element vertices (memory optimization)...")
    t_elem_start = time.time()
    element_vertices = precompute_element_vertices(connectivity, node_positions, verbose=True)
    t_elem_elapsed = time.time() - t_elem_start
    print(f"  Element vertices precomputed in {t_elem_elapsed:.2f}s")

    # ========================================================================
    # 5. Build Morton Octree (same as production)
    # ========================================================================

    print(f"\n[5/9] Building Morton octree...")
    t_octree = time.time()
    octree_struct = build_global_morton_octree(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=256,
        max_depth=21,
        verbose=False
    )
    t_octree = time.time() - t_octree
    print(f"  Built {octree_struct.n_leaves:,} leaves in {t_octree:.2f}s")

    # ========================================================================
    # 6. Upload to GPU (same as production)
    # ========================================================================

    print("\n[6/9] Uploading to GPU...")

    # Build element neighbors
    element_neighbors = build_element_neighbors_array(connectivity, method='face', verbose=False)

    # Upload mesh
    mesh_gpu = upload_mesh_to_gpu(
        connectivity=connectivity,
        node_positions=node_positions,
        element_neighbors=element_neighbors,
        verbose=False
    )

    # Upload Morton structure
    mesh_gpu_octree = upload_global_morton_to_gpu(
        octree_struct,
        connectivity,
        node_positions
    )

    print(f"  Uploaded mesh and Morton octree to GPU")

    # Upload AA metadata to GPU (manually upload each field)
    from jaxtrace.gpu.search.aa_detection import AxisAlignedMetadata
    aa_metadata_gpu = AxisAlignedMetadata(
        base_vertex_indices=jax.device_put(aa_metadata.base_vertex_indices),
        base_vertices=jax.device_put(aa_metadata.base_vertices),
        inv_edge_lengths=jax.device_put(aa_metadata.inv_edge_lengths),
        axis_indices=jax.device_put(aa_metadata.axis_indices),
        is_axis_aligned=jax.device_put(aa_metadata.is_axis_aligned)
    )
    element_vertices_gpu = jax.device_put(element_vertices)

    # Set corrected metadata for new methods
    set_corrected_metadata(aa_metadata_gpu, element_vertices_gpu)
    print(f"  Corrected AA metadata registered for new methods")

    # ========================================================================
    # 7. Generate Particles (same as production)
    # ========================================================================

    print(f"\n[7/9] Generating particles...")

    # Compute domain bounds
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)
    domain_size = domain_max - domain_min

    # Compute particle bounds from fractions
    par_bounds_min = np.zeros(3, dtype=np.float32)
    par_bounds_max = np.zeros(3, dtype=np.float32)
    for i, axis in enumerate(['x', 'y', 'z']):
        min_frac, max_frac = PARTICLE_BOUNDS_FRACTION[axis]
        par_bounds_min[i] = domain_min[i] + min_frac * domain_size[i]
        par_bounds_max[i] = domain_min[i] + max_frac * domain_size[i]
    par_bounds = [par_bounds_min, par_bounds_max]

    nx, ny, nz = PARTICLE_GRID_RESOLUTION

    # Generate uniform grid
    particle_positions = uniform_grid_seeds(
        resolution=(nx, ny, nz),
        bounds=par_bounds,
        include_boundaries=True
    )

    # Clip to mesh bounds
    mesh_bbox_min = domain_min
    mesh_bbox_max = domain_max
    margin = 0.01
    bbox_min_safe = mesh_bbox_min + margin * (mesh_bbox_max - mesh_bbox_min)
    bbox_max_safe = mesh_bbox_max - margin * (mesh_bbox_max - mesh_bbox_min)
    particle_positions = np.clip(particle_positions, bbox_min_safe, bbox_max_safe)

    n_particles = particle_positions.shape[0]
    print(f"  Generated {n_particles:,} particles")

    # Upload to GPU
    positions_gpu = jax.device_put(particle_positions.astype(np.float32))

    # ========================================================================
    # 8. Benchmark: Initial Assignment with Different Methods
    # ========================================================================

    print("\n[8/9] Benchmarking initial assignment with 6 methods...")
    print(f"  Configuration:")
    print(f"    Initial radius: {INITIAL_SEARCH_RADIUS}")
    print(f"    Fallback radii: {INITIAL_SEARCH_FALLBACK_RADII}")
    print(f"    Particles: {n_particles:,}")
    print("")
    print(f"  Methods to test:")
    print(f"    OLD (broken): current, skala, axis_aligned")
    print(f"    NEW (corrected): pure_aa, skala_memory_opt, branchless_hybrid")
    print("")

    results = {}

    for method_name in ['current', 'skala', 'axis_aligned', 'pure_aa', 'skala_memory_opt', 'branchless_hybrid']:
        print(f"  Testing method: {method_name}")

        # Set config
        config.POINT_IN_TET_METHOD = method_name

        # Warmup (compile)
        print(f"    Warming up JIT...")
        _ = initial_assignment_cascading_fallback(
            positions_gpu[:100],  # Small subset for warmup
            mesh_gpu_octree,
            initial_radius=INITIAL_SEARCH_RADIUS,
            fallback_radii=[INITIAL_SEARCH_FALLBACK_RADII[0]],  # Just first fallback
            verbose=False
        )

        # # Benchmark
        print(f"    Running full initial assignment...")
        t_start = time.time()
        element_ids_gpu = initial_assignment_cascading_fallback(
            positions_gpu,
            mesh_gpu_octree,
            initial_radius=INITIAL_SEARCH_RADIUS,
            fallback_radii=INITIAL_SEARCH_FALLBACK_RADII,
            verbose=True
        )
        element_ids_gpu = jax.block_until_ready(element_ids_gpu)
        t_elapsed = time.time() - t_start

        # Check assignment success
        n_assigned = int(jnp.sum(element_ids_gpu >= 0))
        success_rate = (n_assigned / n_particles) * 100
        throughput = n_particles / t_elapsed

        results[method_name] = {
            'time': t_elapsed,
            'n_assigned': n_assigned,
            'success_rate': success_rate,
            'throughput': throughput,
            'element_ids': element_ids_gpu
        }

        print(f"    Time: {t_elapsed:.2f}s")
        print(f"    Assigned: {n_assigned:,}/{n_particles:,} ({success_rate:.2f}%)")
        print(f"    Throughput: {throughput:,.0f} particles/s")
        print("")

    # ========================================================================
    # 9. Results Analysis
    # ========================================================================

    print("\n[9/9] Results Summary")
    print("=" * 80)
    print("INITIAL ASSIGNMENT BENCHMARK")
    print("=" * 80)
    print(f"Mesh: {n_elements:,} elements, {n_nodes:,} nodes")
    print(f"Particles: {n_particles:,}")
    print(f"Morton leaves: {octree_struct.n_leaves:,}")
    print("")
    print("Timing Results:")
    print("=" * 80)

    baseline_time = results['current']['time']
    baseline_throughput = results['current']['throughput']

    print("OLD Methods (original implementation):")
    print("-" * 80)
    for method_name in ['current', 'skala', 'axis_aligned']:
        r = results[method_name]
        speedup = baseline_time / r['time']
        print(f"  {method_name:20s}  {r['time']:8.2f}s  "
              f"{r['throughput']:12,.0f} p/s  "
              f"[{speedup:.2f}× speedup]  "
              f"({r['success_rate']:.2f}% assigned)")

    print("")
    print("NEW Methods (corrected implementation):")
    print("-" * 80)
    for method_name in ['pure_aa', 'skala_memory_opt', 'branchless_hybrid']:
        r = results[method_name]
        speedup = baseline_time / r['time']
        print(f"  {method_name:20s}  {r['time']:8.2f}s  "
              f"{r['throughput']:12,.0f} p/s  "
              f"[{speedup:.2f}× speedup]  "
              f"({r['success_rate']:.2f}% assigned)")

    print("")
    print("Assignment Agreement:")
    print("=" * 80)

    # Check if all methods assigned the same particles
    ids_current = results['current']['element_ids']

    # Compare all methods against baseline (current)
    def check_agreement(ids_a, ids_b, name_a, name_b):
        """Check if two methods agree on assignments."""
        # Agreement: both assigned or both unassigned
        agree_assignment = jnp.all((ids_a >= 0) == (ids_b >= 0))

        # For assigned particles, check if they got same element
        assigned_mask_a = ids_a >= 0
        assigned_mask_b = ids_b >= 0
        assigned_both = assigned_mask_a & assigned_mask_b

        if jnp.sum(assigned_both) > 0:
            same_element = jnp.all(ids_a[assigned_both] == ids_b[assigned_both])
        else:
            same_element = True

        passed = agree_assignment and same_element

        print(f"  {name_a:10s} ↔ {name_b:20s}  {'✅ PASS' if passed else '❌ FAIL'}")

        if not passed:
            n_diff_assignment = jnp.sum((ids_a >= 0) != (ids_b >= 0))
            n_diff_element = jnp.sum((ids_a != ids_b) & assigned_both)
            print(f"    Different assignment status: {n_diff_assignment}")
            print(f"    Different elements: {n_diff_element}")

        return passed

    all_passed = True

    # Compare OLD methods against baseline
    all_passed &= check_agreement(ids_current, results['skala']['element_ids'], 'current', 'skala')
    all_passed &= check_agreement(ids_current, results['axis_aligned']['element_ids'], 'current', 'axis_aligned')

    # Compare NEW methods against baseline
    all_passed &= check_agreement(ids_current, results['pure_aa']['element_ids'], 'current', 'pure_aa')
    all_passed &= check_agreement(ids_current, results['skala_memory_opt']['element_ids'], 'current', 'skala_memory_opt')
    all_passed &= check_agreement(ids_current, results['branchless_hybrid']['element_ids'], 'current', 'branchless_hybrid')

    print("")
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Find best corrected method
    corrected_methods = ['pure_aa', 'skala_memory_opt', 'branchless_hybrid']
    best_method = max(corrected_methods, key=lambda m: results[m]['throughput'])
    best_speedup = baseline_time / results[best_method]['time']
    best_throughput = results[best_method]['throughput']

    print(f"\nBest Corrected Method: {best_method}")
    print(f"  Speedup: {best_speedup:.2f}×")
    print(f"  Throughput: {best_throughput:,.0f} p/s")
    print(f"  Assignment rate: {results[best_method]['success_rate']:.2f}%")

    if best_speedup >= 3.0:
        print(f"\n✅ EXCELLENT: {best_method} achieves {best_speedup:.2f}× speedup (target: 3-4×)")
        print(f"   Recommend: Use POINT_IN_TET_METHOD='{best_method}' in production")
    elif best_speedup >= 2.0:
        print(f"\n✅ GOOD: {best_method} achieves {best_speedup:.2f}× speedup")
        print(f"   Recommend: Use POINT_IN_TET_METHOD='{best_method}' in production")
    elif best_speedup >= 1.5:
        print(f"\n⚠️  MODEST: {best_method} achieves {best_speedup:.2f}× speedup")
        print(f"   May be beneficial for long production runs")
    else:
        print(f"\n❌ POOR: Best method only achieves {best_speedup:.2f}× speedup")
        print(f"   Further optimization needed")

    # Compare corrected vs OLD axis_aligned
    speedup_axis_old = baseline_time / results['axis_aligned']['time']
    print(f"\nCorrected vs OLD axis_aligned:")
    print(f"  OLD axis_aligned: {speedup_axis_old:.2f}× speedup (BROKEN)")
    print(f"  NEW {best_method}: {best_speedup:.2f}× speedup")
    improvement = best_speedup / speedup_axis_old
    print(f"  Improvement: {improvement:.2f}× faster than OLD implementation")

    # Check agreement
    if all_passed:
        print(f"\n✅ All methods produce identical assignments - safe for production")
    else:
        print(f"\n❌ WARNING: Methods disagree on assignments - investigate before production use")

    print("\n" + "=" * 80)
    print("Production Extrapolation (2,500 timesteps):")
    print("=" * 80)

    # Estimate: Initial assignment once + 2,500 RK4 steps
    # Assume point-in-tet is ~60% of RK4 time
    # Each RK4 step does ~5-10 point-in-tet calls per particle (L0 miss + L1 miss + L2 search)

    print(f"Assumptions:")
    print(f"  - Initial assignment: 1× (one-time cost)")
    print(f"  - RK4 integration: 2,500 steps")
    print(f"  - Point-in-tet is 60% of RK4 time")
    print(f"  - Speedup applies to point-in-tet portion only")
    print("")

    print("OLD Methods:")
    for method_name in ['current', 'skala', 'axis_aligned']:
        speedup = baseline_time / results[method_name]['time']
        print(f"  {method_name:20s}  Initial: {results[method_name]['time']:.2f}s, "
              f"Expected RK4 speedup: {speedup:.2f}×")

    print("")
    print("NEW Methods (corrected):")
    for method_name in ['pure_aa', 'skala_memory_opt', 'branchless_hybrid']:
        speedup = baseline_time / results[method_name]['time']
        print(f"  {method_name:20s}  Initial: {results[method_name]['time']:.2f}s, "
              f"Expected RK4 speedup: {speedup:.2f}×")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
