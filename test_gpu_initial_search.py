#!/usr/bin/env python3
"""
Quick test of GPU-accelerated initial element search.

Tests on a small mesh to verify correctness before running on ThreadedA.
"""

import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jaxtrace.gpu.test_meshes import generate_test_mesh, SMALL_BALANCED_MESH
from jaxtrace.gpu.mesh_loader import assign_elements_to_blocks
from jaxtrace.gpu.octree_builder import build_octrees_per_block
from jaxtrace.gpu.particle_seeding import seed_particles_uniform_grid, SeedingConfig
from jaxtrace.gpu.initial_search_jax import find_initial_elements_batch, GPUConfig

print("=" * 80)
print("TESTING GPU INITIAL ELEMENT SEARCH")
print("=" * 80)
print()

# Generate test mesh
print("Phase 1: Generating test mesh...")
positions, connectivity = generate_test_mesh(SMALL_BALANCED_MESH)
print(f"  Mesh: {len(connectivity):,} elements, {len(positions):,} nodes")
print()

# Assign to blocks
print("Phase 2: Assigning elements to blocks...")
element_block_IDs, partition_data = assign_elements_to_blocks(
    positions, connectivity, (2, 2, 1), verbose=False
)
print(f"  Grid: {partition_data.grid_size}")
print()

# Build octrees
print("Phase 3: Building octrees...")
octrees = build_octrees_per_block(
    positions, connectivity, element_block_IDs, partition_data,
    max_elements_per_node=50,
    verbose=False
)
print(f"  Built {len(octrees)} octrees")
print()

# Seed particles
print("Phase 4: Seeding particles...")
config = SeedingConfig(
    bbox_min=partition_data.bbox_min,
    bbox_max=partition_data.bbox_max,
    density_per_axis=(10, 10, 5),  # 500 particles
    seed=42
)
particle_positions = seed_particles_uniform_grid(config)
n_particles = len(particle_positions)
print(f"  Seeded {n_particles:,} particles")
print()

# Prepare mesh data
mesh_data = {
    'positions': positions,
    'connectivity': connectivity
}

# Test GPU search
print("Phase 5: Running GPU initial search...")
print("-" * 80)

gpu_config = GPUConfig(
    use_gpu_initial_search=True,
    force_cpu=False
)

t0 = time.time()
element_IDs_gpu, stats_gpu = find_initial_elements_batch(
    particle_positions,
    mesh_data,
    partition_data,
    octrees,
    config=gpu_config,
    verbose=True
)
t_gpu = time.time() - t0

print()
print(f"GPU search completed in {t_gpu:.3f}s")
print(f"  Implementation: {'GPU' if stats_gpu['used_gpu'] else 'CPU fallback'}")
print(f"  Found: {stats_gpu['n_found']}/{n_particles} ({100*stats_gpu['n_found']/n_particles:.1f}%)")
print()

# Test CPU search for comparison
print("Phase 6: Running CPU search for comparison...")
print("-" * 80)

cpu_config = GPUConfig(force_cpu=True)

t0 = time.time()
element_IDs_cpu, stats_cpu = find_initial_elements_batch(
    particle_positions,
    mesh_data,
    partition_data,
    octrees,
    config=cpu_config,
    verbose=True
)
t_cpu = time.time() - t0

print()
print(f"CPU search completed in {t_cpu:.3f}s")
print(f"  Found: {stats_cpu['n_found']}/{n_particles} ({100*stats_cpu['n_found']/n_particles:.1f}%)")
print()

# Compare results
print("Phase 7: Comparing GPU vs CPU results...")
print("-" * 80)

matches = np.sum(element_IDs_gpu == element_IDs_cpu)
agreement = 100 * matches / n_particles

print(f"Agreement: {matches}/{n_particles} ({agreement:.1f}%)")

if agreement >= 95.0:
    print("✅ PASS: GPU and CPU results match (≥95%)")
else:
    print(f"⚠️  WARNING: Agreement {agreement:.1f}% below 95%")
    print()
    print("Mismatches:")
    mismatch_indices = np.where(element_IDs_gpu != element_IDs_cpu)[0]
    for idx in mismatch_indices[:10]:  # Show first 10
        print(f"  Particle {idx}: GPU={element_IDs_gpu[idx]}, CPU={element_IDs_cpu[idx]}")

print()

# Performance summary
print("=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
print(f"Mesh: {len(connectivity):,} elements")
print(f"Particles: {n_particles:,}")
print()
print(f"GPU time: {t_gpu:.3f}s ({stats_gpu['time_per_particle_ms']:.3f} ms/particle)")
print(f"CPU time: {t_cpu:.3f}s ({stats_cpu['time_per_particle_ms']:.3f} ms/particle)")
print()

if stats_gpu['used_gpu'] and t_cpu > t_gpu:
    speedup = t_cpu / t_gpu
    print(f"Speedup: {speedup:.2f}×")
    print("✅ GPU acceleration working!")
elif not stats_gpu['used_gpu']:
    print("⚠️  GPU not used (fell back to CPU)")
else:
    print("⚠️  GPU not faster than CPU (may need larger mesh)")

print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
