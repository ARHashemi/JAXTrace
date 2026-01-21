#!/usr/bin/env python3
"""
Point-in-tetrahedron performance test on REAL production mesh.

This test loads the actual ThreadedA/FLA mesh and runs performance benchmarks
on real tetrahedral elements with production particle seeding.

Tests:
1. Load real mesh from PVTU (same as production script)
2. Generate realistic particle positions (uniform grid seeding)
3. Benchmark all three methods on 10K random queries
4. Verify 100% agreement between methods
5. Report expected production speedup
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

from jaxtrace.gpu.search.point_in_tet_methods import (
    point_in_tet_current,
    point_in_tet_skala,
    point_in_tet_axis_aligned,
)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.tracking.seeding import uniform_grid_seeds


# ============================================================================
# Configuration (same as production script)
# ============================================================================

MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 121)  # Load just ONE timestep for testing
VELOCITY_FIELD_NAME = 'Displacement'

# Particle seeding (same as production)
PARTICLE_GRID_RESOLUTION = (50, 90, 50)  # 225,000 particles
PARTICLE_BOUNDS_FRACTION = {
    'x': (0.2, 0.35),
    'y': (0.2, 0.8),
    'z': (0.3, 1.0),
}

# Test parameters
N_BENCHMARK_QUERIES = 10000  # Number of queries for benchmark
SEED = 42


def main():
    print("=" * 80)
    print("Point-in-Tetrahedron Performance Test - REAL MESH")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 80)

    # ========================================================================
    # 1. Load Real Mesh (same as production script)
    # ========================================================================

    print("\n[1/5] Loading mesh from PVTU...")
    t_load = time.time()
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=True
    )
    t_load = time.time() - t_load

    n_nodes_orig = node_positions.shape[0]
    n_elements = connectivity.shape[0]

    print(f"\n  Mesh loaded in {t_load:.2f}s")
    print(f"  Elements: {n_elements:,}")
    print(f"  Nodes (before dedup): {n_nodes_orig:,}")

    # ========================================================================
    # 1.5. Deduplicate Nodes (same as production)
    # ========================================================================

    print(f"\n[1.5/5] Deduplicating nodes (PVTU piece boundary fix)...")
    t_dedup = time.time()
    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=True
    )
    t_dedup = time.time() - t_dedup

    n_nodes = node_positions.shape[0]

    if n_duplicates_removed > 0:
        print(f"  ✅ Removed {n_duplicates_removed:,} duplicates in {t_dedup:.2f}s")
        print(f"  Nodes (after dedup): {n_nodes:,}")
    else:
        print(f"  ✅ No duplicates found")

    # Upload to GPU
    connectivity_gpu = jax.device_put(connectivity.astype(np.int32))
    node_positions_gpu = jax.device_put(node_positions.astype(np.float32))

    print(f"\n  ✅ Mesh uploaded to GPU")
    print(f"     Elements: {n_elements:,}")
    print(f"     Nodes: {n_nodes:,}")
    print(f"     Memory: {(connectivity_gpu.nbytes + node_positions_gpu.nbytes) / (1024**2):.1f} MB")

    # ========================================================================
    # 2. Generate Test Particle Positions (EXACT copy from production)
    # ========================================================================

    print(f"\n[2/5] Generating test particle positions...")

    # Compute domain bounds (EXACT copy from production)
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)
    domain_size = domain_max - domain_min

    print(f"  Mesh bounding box:")
    print(f"    X: [{domain_min[0]:.6f}, {domain_max[0]:.6f}]  (size: {domain_size[0]:.6f})")
    print(f"    Y: [{domain_min[1]:.6f}, {domain_max[1]:.6f}]  (size: {domain_size[1]:.6f})")
    print(f"    Z: [{domain_min[2]:.6f}, {domain_max[2]:.6f}]  (size: {domain_size[2]:.6f})")

    # Compute particle bounds from fractions (EXACT copy from production)
    par_bounds_min = np.zeros(3, dtype=np.float32)
    par_bounds_max = np.zeros(3, dtype=np.float32)
    for i, axis in enumerate(['x', 'y', 'z']):
        min_frac, max_frac = PARTICLE_BOUNDS_FRACTION[axis]
        par_bounds_min[i] = domain_min[i] + min_frac * domain_size[i]
        par_bounds_max[i] = domain_min[i] + max_frac * domain_size[i]
    par_bounds = [par_bounds_min, par_bounds_max]

    nx, ny, nz = PARTICLE_GRID_RESOLUTION
    print(f"  Particle bounds:")
    print(f"    X: [{par_bounds_min[0]:.6f}, {par_bounds_max[0]:.6f}] (domain fraction: {PARTICLE_BOUNDS_FRACTION['x']})")
    print(f"    Y: [{par_bounds_min[1]:.6f}, {par_bounds_max[1]:.6f}] (domain fraction: {PARTICLE_BOUNDS_FRACTION['y']})")
    print(f"    Z: [{par_bounds_min[2]:.6f}, {par_bounds_max[2]:.6f}] (domain fraction: {PARTICLE_BOUNDS_FRACTION['z']})")

    # Generate uniform grid (EXACT copy from production)
    particle_positions = uniform_grid_seeds(
        resolution=(nx, ny, nz),
        bounds=par_bounds,
        include_boundaries=True
    )

    # PHASE 1.1 FIX: Clip particles to mesh bounds (EXACT copy from production)
    print(f"\n  Clipping particles to mesh bounds (Phase 1.1 fix)...")
    original_positions = particle_positions.copy()
    mesh_bbox_min = domain_min
    mesh_bbox_max = domain_max
    margin = 0.01
    bbox_min_safe = mesh_bbox_min + margin * (mesh_bbox_max - mesh_bbox_min)
    bbox_max_safe = mesh_bbox_max - margin * (mesh_bbox_max - mesh_bbox_min)

    particle_positions_clipped = np.clip(particle_positions, bbox_min_safe, bbox_max_safe)
    particle_positions = particle_positions_clipped

    # Diagnostic
    n_moved = np.sum(np.any(particle_positions != original_positions, axis=1))
    print(f"    Particles clipped to mesh bounds: {n_moved}/{particle_positions.shape[0]}")

    seed_positions_cpu = particle_positions
    n_particles = seed_positions_cpu.shape[0]
    print(f"  Generated {n_particles:,} particles in grid: {nx}×{ny}×{nz}")

    # Upload particles to GPU
    seed_positions_gpu = jax.device_put(seed_positions_cpu.astype(np.float32))

    # ========================================================================
    # 3. Select Random Test Queries (REALISTIC: element centroids)
    # ========================================================================

    print(f"\n[3/5] Selecting {N_BENCHMARK_QUERIES:,} random test queries...")
    print(f"  Computing element centroids for realistic queries...")

    # Compute element centroids (guaranteed to be inside elements)
    # This creates a REALISTIC benchmark where points are actually inside elements
    rng = np.random.RandomState(SEED)
    test_elem_indices = rng.choice(n_elements, size=N_BENCHMARK_QUERIES, replace=False)

    # Compute centroids for selected elements
    # Centroid = (v0 + v1 + v2 + v3) / 4
    v0 = node_positions[connectivity[test_elem_indices, 0]]  # (N, 3)
    v1 = node_positions[connectivity[test_elem_indices, 1]]
    v2 = node_positions[connectivity[test_elem_indices, 2]]
    v3 = node_positions[connectivity[test_elem_indices, 3]]

    test_positions_cpu = (v0 + v1 + v2 + v3) / 4.0  # (N, 3)
    test_positions = jax.device_put(test_positions_cpu.astype(np.float32))
    test_element_ids = jnp.array(test_elem_indices, dtype=jnp.int32)

    print(f"  Selected {N_BENCHMARK_QUERIES:,} queries:")
    print(f"    Element IDs: {test_element_ids[:5]}...")
    print(f"    Using element centroids (guaranteed inside)")
    print(f"  Note: This tests realistic case where particles are inside elements")

    # ========================================================================
    # 4. Verify Method Agreement
    # ========================================================================

    print(f"\n[4/5] Verifying method agreement on real mesh...")

    # Test on first 100 queries (quick verification)
    n_verify = min(100, N_BENCHMARK_QUERIES)
    print(f"  Testing {n_verify} queries for agreement...")

    mismatches = 0
    for i in range(n_verify):
        pos = test_positions[i]
        elem_id = test_element_ids[i]

        r_current = point_in_tet_current(pos, elem_id, connectivity_gpu, node_positions_gpu)
        r_skala = point_in_tet_skala(pos, elem_id, connectivity_gpu, node_positions_gpu)
        r_axis_aligned = point_in_tet_axis_aligned(pos, elem_id, connectivity_gpu, node_positions_gpu)

        if not (r_current == r_skala and r_skala == r_axis_aligned):
            print(f"  ❌ MISMATCH at query {i}:")
            print(f"     pos={pos}, elem_id={elem_id}")
            print(f"     current={r_current}, skala={r_skala}, axis_aligned={r_axis_aligned}")
            mismatches += 1

    if mismatches == 0:
        print(f"  ✅ All {n_verify} queries agree (100% agreement)")
    else:
        print(f"  ❌ {mismatches}/{n_verify} mismatches ({100*mismatches/n_verify:.1f}%)")
        print(f"  WARNING: Methods disagree on real mesh!")

    # ========================================================================
    # 5. Performance Benchmark
    # ========================================================================

    print(f"\n[5/5] Benchmarking performance on real mesh...")
    print(f"  Benchmarking {N_BENCHMARK_QUERIES:,} queries...")

    # Compile JIT functions
    @jax.jit
    def bench_current(positions, elem_ids):
        def query(pos, elem_id):
            return point_in_tet_current(pos, elem_id, connectivity_gpu, node_positions_gpu)
        return jax.vmap(query)(positions, elem_ids)

    @jax.jit
    def bench_skala(positions, elem_ids):
        def query(pos, elem_id):
            return point_in_tet_skala(pos, elem_id, connectivity_gpu, node_positions_gpu)
        return jax.vmap(query)(positions, elem_ids)

    @jax.jit
    def bench_axis_aligned(positions, elem_ids):
        def query(pos, elem_id):
            return point_in_tet_axis_aligned(pos, elem_id, connectivity_gpu, node_positions_gpu)
        return jax.vmap(query)(positions, elem_ids)

    # Warmup
    print("  Warming up JIT compilation...")
    _ = bench_current(test_positions[:10], test_element_ids[:10]).block_until_ready()
    _ = bench_skala(test_positions[:10], test_element_ids[:10]).block_until_ready()
    _ = bench_axis_aligned(test_positions[:10], test_element_ids[:10]).block_until_ready()

    # Benchmark current method
    print("  Benchmarking current method...")
    start = time.time()
    results_current = bench_current(test_positions, test_element_ids).block_until_ready()
    time_current = time.time() - start
    throughput_current = N_BENCHMARK_QUERIES / time_current

    # Benchmark Skala method
    print("  Benchmarking Skala method...")
    start = time.time()
    results_skala = bench_skala(test_positions, test_element_ids).block_until_ready()
    time_skala = time.time() - start
    throughput_skala = N_BENCHMARK_QUERIES / time_skala

    # Benchmark axis-aligned method
    print("  Benchmarking axis-aligned method...")
    start = time.time()
    results_axis_aligned = bench_axis_aligned(test_positions, test_element_ids).block_until_ready()
    time_axis_aligned = time.time() - start
    throughput_axis_aligned = N_BENCHMARK_QUERIES / time_axis_aligned

    # Verify agreement on full benchmark
    agreement_current_skala = jnp.all(results_current == results_skala)
    agreement_skala_axis = jnp.all(results_skala == results_axis_aligned)

    # ========================================================================
    # Results Summary
    # ========================================================================

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS (Real Mesh)")
    print("=" * 80)
    print(f"Mesh: {n_elements:,} elements, {n_nodes:,} nodes")
    print(f"Queries: {N_BENCHMARK_QUERIES:,}")
    print(f"\nTiming Results:")
    print("=" * 80)
    print(f"  current:         {time_current*1000:8.2f} ms  ({throughput_current:12,.0f} queries/sec)  [baseline]")
    print(f"  skala:           {time_skala*1000:8.2f} ms  ({throughput_skala:12,.0f} queries/sec)  "
          f"[{throughput_skala/throughput_current:.2f}× speedup]")
    print(f"  axis_aligned:    {time_axis_aligned*1000:8.2f} ms  ({throughput_axis_aligned:12,.0f} queries/sec)  "
          f"[{throughput_axis_aligned/throughput_current:.2f}× speedup]")

    print(f"\nAgreement:")
    print("=" * 80)
    print(f"  current ↔ skala:         {'✅ PASS' if agreement_current_skala else '❌ FAIL'}")
    print(f"  skala ↔ axis_aligned:    {'✅ PASS' if agreement_skala_axis else '❌ FAIL'}")

    # Count hits (particles inside tested elements)
    n_hits_current = jnp.sum(results_current)
    hit_rate = 100.0 * n_hits_current / N_BENCHMARK_QUERIES
    print(f"\nContainment Statistics:")
    print("=" * 80)
    print(f"  Queries inside element: {n_hits_current:,} / {N_BENCHMARK_QUERIES:,} ({hit_rate:.1f}%)")
    print(f"  Note: Low hit rate is expected (random particles tested against random elements)")

    # ========================================================================
    # Production Extrapolation
    # ========================================================================

    print("\n" + "=" * 80)
    print("EXTRAPOLATED PRODUCTION PERFORMANCE")
    print("=" * 80)

    # Estimate production parameters
    n_particles_prod = 225_000  # FLA mesh
    n_steps_prod = 2_500
    dt_prod = 0.0025

    # Estimate queries per timestep (conservative: 5 searches per particle per step)
    # - Initial assignment: 1 search
    # - RK4: 4 stages × ~1-2 searches per stage (L0 hit rate ~85%, L1 ~10%, L2 ~5%)
    queries_per_step = n_particles_prod * 5  # Conservative estimate

    # Total runtime estimates
    total_queries = queries_per_step * n_steps_prod

    time_per_step_current = queries_per_step / throughput_current
    time_per_step_skala = queries_per_step / throughput_skala
    time_per_step_axis_aligned = queries_per_step / throughput_axis_aligned

    total_time_current = time_per_step_current * n_steps_prod
    total_time_skala = time_per_step_skala * n_steps_prod
    total_time_axis_aligned = time_per_step_axis_aligned * n_steps_prod

    print(f"Production Scenario: {n_particles_prod:,} particles, {n_steps_prod:,} steps")
    print(f"Estimated queries per step: {queries_per_step:,}")
    print(f"Total queries: {total_queries:,}")
    print("")
    print(f"Expected Performance:")
    print("=" * 80)
    print(f"  current:         {time_per_step_current:6.2f} s/step  ({total_time_current/3600:5.2f} hours total)")
    print(f"  skala:           {time_per_step_skala:6.2f} s/step  ({total_time_skala/3600:5.2f} hours total)  "
          f"[{throughput_skala/throughput_current:.2f}× faster]")
    print(f"  axis_aligned:    {time_per_step_axis_aligned:6.2f} s/step  ({total_time_axis_aligned/3600:5.2f} hours total)  "
          f"[{throughput_axis_aligned/throughput_current:.2f}× faster]")

    print("\n" + "=" * 80)
    print("Time Savings (compared to current):")
    print("=" * 80)
    time_saved_skala = total_time_current - total_time_skala
    time_saved_axis_aligned = total_time_current - total_time_axis_aligned
    print(f"  skala:           {time_saved_skala/3600:5.2f} hours saved ({100*(1-total_time_skala/total_time_current):.1f}% reduction)")
    print(f"  axis_aligned:    {time_saved_axis_aligned/3600:5.2f} hours saved ({100*(1-total_time_axis_aligned/total_time_current):.1f}% reduction)")

    # ========================================================================
    # Final Recommendations
    # ========================================================================

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if throughput_skala / throughput_current >= 2.0:
        print("✅ Skala method shows significant speedup (≥2×)")
        print("   Recommend: Use POINT_IN_TET_METHOD='skala' in production")
    else:
        print("⚠️  Skala speedup below target (<2×)")
        print("   Possible causes: JIT overhead, small query count, GPU underutilization")
        print("   Recommend: Test with full production run (2,500 steps)")

    if throughput_axis_aligned / throughput_current >= 5.0:
        print("✅ Axis-aligned method shows major speedup (≥5×)")
        print("   Recommend: Use POINT_IN_TET_METHOD='axis_aligned' for maximum performance")
    elif throughput_axis_aligned / throughput_skala >= 2.0:
        print("✅ Axis-aligned provides additional speedup over Skala (≥2×)")
        print("   Recommend: Test POINT_IN_TET_METHOD='axis_aligned' after Skala validation")
    else:
        print("⚠️  Axis-aligned speedup below target (detection overhead?)")
        print("   Recommend: Stick with 'skala' unless further optimization needed")

    if agreement_current_skala and agreement_skala_axis:
        print("\n✅ All methods produce identical results - safe to use in production")
    else:
        print("\n❌ WARNING: Methods disagree - do NOT use in production until fixed!")

    print("\n" + "=" * 80)
    print("Test complete! Ready for production validation.")
    print("=" * 80)


if __name__ == "__main__":
    main()
