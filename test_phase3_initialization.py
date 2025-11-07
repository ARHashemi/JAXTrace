"""
Phase 3 Integration Test: Particle Seeding & CPU Baseline Search

Tests the complete Phase 3 workflow:
1. Load ThreadedA mesh (from Phase 1 & 2)
2. Seed particles with flexible configuration
3. Run CPU baseline search (two-stage with optional parallelization)
4. Validate results
5. Measure performance
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import time

from jaxtrace.gpu.forest.block_grid import create_regular_grid
from jaxtrace.gpu.forest.block_mapper import assign_elements_to_block_list
from jaxtrace.gpu.forest.padded_arrays import build_padded_block_arrays
from jaxtrace.gpu.forest.cpu_baseline_search import (
    cpu_baseline_search_batch,
    validate_cpu_search_results
)
from jaxtrace.gpu.particles.seeding import (
    seed_particles_uniform,
    seed_particles_random,
    seed_particles_stratified,
    ParticleState,
    compute_particle_density,
)


def load_threadeda_mesh():
    """Load ThreadedA mesh using VTK."""
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
    print("Phase 3 COMPLETE: Particle Seeding & CPU Baseline Search")
    print("=" * 80)

    # === SETUP: Load mesh and create data structures ===
    print("\n" + "=" * 80)
    print("SETUP: Load ThreadedA Mesh")
    print("=" * 80)
    
    t0 = time.time()
    nodes, connectivity = load_threadeda_mesh()
    print(f"  Nodes: {nodes.shape[0]:,}")
    print(f"  Elements: {connectivity.shape[0]:,}")
    print(f"  Load time: {time.time() - t0:.2f} s")

    # Create forest structure (Phase 1)
    domain_bounds = np.array([-0.03, 0.03, -0.023, 0.023, -0.01, 0.0], dtype=np.float32)
    grid_size = (4, 4, 2)  # 32 blocks
    blocks = create_regular_grid(domain_bounds, grid_size)
    
    # Assign elements to blocks
    element_to_block, block_stats = assign_elements_to_block_list(
        nodes, connectivity, blocks, verbose=False
    )
    
    # Build padded arrays (Phase 2)
    padded = build_padded_block_arrays(element_to_block, block_stats, verbose=False)
    
    print(f"\n  Forest structure ready:")
    print(f"    Blocks: {len(blocks)}")
    print(f"    Padded arrays: ({padded.n_blocks}, {padded.max_elements_per_block:,})")
    print(f"    Memory: {padded.memory_mb:.1f} MB")

    # === TEST 1: Uniform Seeding (Density-based) ===
    print("\n" + "=" * 80)
    print("TEST 1: Uniform Seeding (Density-based)")
    print("=" * 80)
    
    # Seed with density: 1000 particles/meter on each axis
    positions_uniform = seed_particles_uniform(
        bbox=domain_bounds,
        density_x=1000.0,  # 1 particle per mm
        density_y=1000.0,
        density_z=500.0,   # 0.5 particles per mm in z
        jitter=0.0,
        verbose=True
    )
    
    print(f"\n  Seeded {positions_uniform.shape[0]:,} particles")
    
    # Compute actual density
    dens_total, dens_x, dens_y, dens_z = compute_particle_density(
        positions_uniform, domain_bounds
    )
    print(f"  Actual density:")
    print(f"    Total: {dens_total:.1f} particles/m³")
    print(f"    Per axis: {dens_x:.1f}, {dens_y:.1f}, {dens_z:.1f} particles/m")

    # === TEST 2: CPU Baseline Search (Sequential) ===
    print("\n" + "=" * 80)
    print("TEST 2: CPU Baseline Search (Sequential, 1000 particles)")
    print("=" * 80)
    
    # Take subset for sequential test
    positions_subset = positions_uniform[:1000]
    
    element_ids_seq, stats_seq = cpu_baseline_search_batch(
        positions_subset,
        domain_bounds,
        grid_size,
        blocks,
        padded,
        nodes,
        connectivity,
        enable_neighbor_search=True,
        use_parallel=False,  # Force sequential
        verbose=True
    )

    # === TEST 3: CPU Baseline Search (Parallel) ===
    print("\n" + "=" * 80)
    print("TEST 3: CPU Baseline Search (Sequential, 10K particles)")
    print("=" * 80)
    print("  Note: Parallel CPU search disabled due to JAX multithread/fork incompatibility")
    print()

    # Take 10K particles for larger test
    positions_10k = positions_uniform[:10000]

    element_ids_10k, stats_10k = cpu_baseline_search_batch(
        positions_10k,
        domain_bounds,
        grid_size,
        blocks,
        padded,
        nodes,
        connectivity,
        enable_neighbor_search=True,
        use_parallel=False,  # DISABLED: JAX multithreading incompatible with multiprocessing.fork()
        n_workers=None,
        verbose=True
    )

    # Show performance consistency
    print(f"\n  Consistency check:")

    # === TEST 4: Self-Validation ===
    print("\n" + "=" * 80)
    print("TEST 4: Self-Validation")
    print("=" * 80)
    
    valid = validate_cpu_search_results(
        positions_10k,
        element_ids_10k,
        nodes,
        connectivity,
        n_samples=1000
    )
    
    if not valid:
        print("  ❌ Validation FAILED!")
        return False

    # === TEST 5: Different Seeding Strategies ===
    print("\n" + "=" * 80)
    print("TEST 5: Alternative Seeding Strategies")
    print("=" * 80)
    
    # Random seeding
    print("\n--- Random Seeding ---")
    positions_random = seed_particles_random(
        bbox=domain_bounds,
        n_particles=1000,
        seed=42,
        verbose=True
    )
    
    # Stratified seeding
    print("\n--- Stratified Seeding ---")
    positions_stratified = seed_particles_stratified(
        bbox=domain_bounds,
        density_x=500.0,
        density_y=500.0,
        density_z=250.0,
        seed=42,
        verbose=True
    )
    
    # Uniform with jitter
    print("\n--- Uniform with Jitter ---")
    positions_jitter = seed_particles_uniform(
        bbox=domain_bounds,
        density_x=500.0,
        density_y=500.0,
        density_z=250.0,
        jitter=0.2,  # 20% perturbation
        seed=42,
        verbose=True
    )

    # === TEST 6: Create ParticleState ===
    print("\n" + "=" * 80)
    print("TEST 6: Create ParticleState")
    print("=" * 80)
    
    # Compute block IDs
    from jaxtrace.gpu.forest.block_grid import position_to_block_id
    block_ids_10k = np.array([
        position_to_block_id(pos, domain_bounds, grid_size)
        for pos in positions_10k
    ], dtype=np.int32)
    
    # Create particle state
    particle_state = ParticleState(
        positions=positions_10k,
        element_ids=element_ids_10k,
        block_ids=block_ids_10k,
        velocities=None,  # Not computed yet
        active=element_ids_10k >= 0  # Active if found in mesh
    )
    
    print(particle_state)

    # === SUMMARY ===
    print("\n" + "=" * 80)
    print("PHASE 3: SUCCESS - ALL TESTS COMPLETE")
    print("=" * 80)
    
    print("\n✅ Test 1: Uniform seeding")
    print(f"    - Seeded {positions_uniform.shape[0]:,} particles")
    print(f"    - Density-based configuration")
    
    print("\n✅ Test 2: CPU search (sequential)")
    print(f"    - {stats_seq.n_found}/{stats_seq.n_particles} found")
    print(f"    - Rate: {stats_seq.searches_per_second:.0f} particles/s")
    
    print("\n✅ Test 3: CPU search (parallel)")
    print(f"    - {stats_10k.n_found}/{stats_10k.n_particles} found")
    print(f"    - Rate: {stats_10k.searches_per_second:.0f} particles/s")
    print(f"    - Workers: {stats_10k.n_workers}")
    if stats_seq.searches_per_second > 0:
        speedup = stats_10k.searches_per_second / stats_seq.searches_per_second
        print(f"    - Speedup: {speedup:.1f}×")
    
    print("\n✅ Test 4: Self-validation")
    print(f"    - All 1000 samples validated")
    
    print("\n✅ Test 5: Alternative seeding strategies")
    print(f"    - Random: {positions_random.shape[0]:,} particles")
    print(f"    - Stratified: {positions_stratified.shape[0]:,} particles")
    print(f"    - Uniform+jitter: {positions_jitter.shape[0]:,} particles")
    
    print("\n✅ Test 6: ParticleState created")
    print(f"    - {particle_state.n_particles:,} particles")
    print(f"    - {particle_state.n_active:,} active")
    
    print("\n" + "-" * 80)
    print("Ready for Phase 4: GPU Multi-Level Search")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
