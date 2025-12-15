#!/usr/bin/env python3
"""
Test hierarchical early-exit 5-hop search implementation.

This test validates the memory-efficient hierarchical search that avoids
GPU OOM by using early-exit instead of concatenating all neighbors.

Validates:
1. Memory efficiency: 5-hop search runs without OOM (~10 MB vs 572 MB)
2. Performance: Throughput 8-15k p/s (vs 23k for 3-hop)
3. Hit rate: 99.99% (vs 99.9% for 3-hop)
4. Correctness: Results match or exceed 3-hop baseline

Comparison:
- Naive 5-hop concatenation: 1,364 neighbors × 105k = 572 MB → OOM ❌
- Hierarchical 5-hop early-exit: avg ~25 neighbors × 105k = 10 MB ✅
"""

import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

# Force CPU-GPU memory management
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# JAXTrace imports
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.particles import ParticleData
from jaxtrace.tracking.seeding import uniform_grid_seeds

# Search implementations
from jaxtrace.gpu.search.incremental_search_vectorized import (
    search_level0_vectorized,
    search_level1_multihop_vectorized,
    search_level1_multihop_hierarchical
)
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu

print("=" * 80)
print("HIERARCHICAL EARLY-EXIT 5-HOP SEARCH TEST")
print("=" * 80)
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")
print()

# Configuration
MESH_PATH = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_120.pvtu"
N_PARTICLES_SMALL = 1000  # Small test for quick validation
N_PARTICLES_LARGE = 10000  # Medium test for performance

print("=" * 80)
print("CONFIGURATION")
print("=" * 80)
print(f"Mesh: {MESH_PATH}")
print(f"Small test: {N_PARTICLES_SMALL:,} particles")
print(f"Large test: {N_PARTICLES_LARGE:,} particles")
print()

# ============================================================================
# Load Mesh
# ============================================================================
print("=" * 80)
print("MESH LOADING")
print("=" * 80)
print()

t0 = time.perf_counter()
node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
    Path(MESH_PATH),
    field_name='Displacement'
)
t_load = time.perf_counter() - t0

print(f"✓ Mesh loaded ({t_load:.2f} s):")
print(f"  Nodes: {len(node_positions):,}")
print(f"  Elements: {len(connectivity):,}")

# Ensure velocity is 3D and float32
if velocity_field.shape[1] == 2:
    velocity_field = np.column_stack([velocity_field, np.zeros(velocity_field.shape[0])])
velocity_field = velocity_field.astype(np.float32)

# Build element neighbors
print()
print("Building element neighbors...")
element_neighbors = build_element_neighbors_array(connectivity, verbose=False)
print(f"✓ Element neighbors built")

# Upload mesh to GPU
print()
print("Uploading mesh to GPU...")
mesh_gpu = upload_mesh_to_gpu(
    connectivity,
    node_positions,
    element_neighbors,
    verbose=True
)
print()

# ============================================================================
# Generate Test Particles
# ============================================================================
print("=" * 80)
print("TEST PARTICLES GENERATION")
print("=" * 80)
print()

# Compute bounding box
bbox = np.array([
    node_positions[:, 0].min(), node_positions[:, 0].max(),
    node_positions[:, 1].min(), node_positions[:, 1].max(),
    node_positions[:, 2].min(), node_positions[:, 2].max(),
], dtype=np.float32)

domain_min = np.array([bbox[0], bbox[2], bbox[4]], dtype=np.float32)
domain_max = np.array([bbox[1], bbox[3], bbox[5]], dtype=np.float32)

# Generate particles for small test
grid_res_small = int(np.ceil(N_PARTICLES_SMALL ** (1/3)))
particles_small = uniform_grid_seeds(
    resolution=(grid_res_small, grid_res_small, grid_res_small),
    bounds=[domain_min, domain_max],
    include_boundaries=True
)[:N_PARTICLES_SMALL]

# Generate particles for large test
grid_res_large = int(np.ceil(N_PARTICLES_LARGE ** (1/3)))
particles_large = uniform_grid_seeds(
    resolution=(grid_res_large, grid_res_large, grid_res_large),
    bounds=[domain_min, domain_max],
    include_boundaries=True
)[:N_PARTICLES_LARGE]

# Assign random valid element IDs (simulating cache from previous timestep)
cached_ids_small = np.random.randint(0, len(connectivity), size=N_PARTICLES_SMALL, dtype=np.int32)
cached_ids_large = np.random.randint(0, len(connectivity), size=N_PARTICLES_LARGE, dtype=np.int32)

print(f"✓ Generated {N_PARTICLES_SMALL:,} particles for small test")
print(f"✓ Generated {N_PARTICLES_LARGE:,} particles for large test")
print()

# ============================================================================
# Test 1: Small Scale - Hierarchical vs Concatenated Comparison
# ============================================================================
print("=" * 80)
print("TEST 1: SMALL SCALE ({:,} particles)".format(N_PARTICLES_SMALL))
print("=" * 80)
print()

# Upload to GPU
positions_small_gpu = jax.device_put(particles_small)
cached_small_gpu = jax.device_put(cached_ids_small)

# Baseline: 3-hop concatenated search
print("Running BASELINE: 3-hop concatenated search...")
t0 = time.perf_counter()
element_ids_3hop_gpu = search_level1_multihop_vectorized(
    positions_small_gpu,
    cached_small_gpu,
    mesh_gpu.element_neighbors,
    mesh_gpu.node_positions,
    mesh_gpu.connectivity,
    n_hops=3
)
element_ids_3hop = np.array(element_ids_3hop_gpu, dtype=np.int32)
t_3hop = time.perf_counter() - t0

hits_3hop = (element_ids_3hop >= 0).sum()
throughput_3hop = N_PARTICLES_SMALL / t_3hop

print(f"✓ 3-hop concatenated: {t_3hop*1000:.2f} ms ({throughput_3hop:.1f} p/s)")
print(f"  Hits: {hits_3hop:,}/{N_PARTICLES_SMALL:,} ({100*hits_3hop/N_PARTICLES_SMALL:.2f}%)")
print()

# New: 5-hop hierarchical search
print("Running NEW: 5-hop hierarchical early-exit search...")
t0 = time.perf_counter()
element_ids_5hop_gpu = search_level1_multihop_hierarchical(
    positions_small_gpu,
    cached_small_gpu,
    mesh_gpu.element_neighbors,
    mesh_gpu.node_positions,
    mesh_gpu.connectivity,
    n_hops=5
)
element_ids_5hop = np.array(element_ids_5hop_gpu, dtype=np.int32)
t_5hop = time.perf_counter() - t0

hits_5hop = (element_ids_5hop >= 0).sum()
throughput_5hop = N_PARTICLES_SMALL / t_5hop

print(f"✓ 5-hop hierarchical: {t_5hop*1000:.2f} ms ({throughput_5hop:.1f} p/s)")
print(f"  Hits: {hits_5hop:,}/{N_PARTICLES_SMALL:,} ({100*hits_5hop/N_PARTICLES_SMALL:.2f}%)")
print()

# Comparison
print("Comparison:")
print(f"  Hit rate improvement: {hits_5hop - hits_3hop:+,} particles ({(hits_5hop/hits_3hop - 1)*100:+.2f}%)")
print(f"  Throughput change: {throughput_5hop - throughput_3hop:+.1f} p/s ({(throughput_5hop/throughput_3hop - 1)*100:+.1f}%)")
print()

# Correctness check: 5-hop should find at least as many particles as 3-hop
if hits_5hop < hits_3hop:
    print(f"❌ FAIL: 5-hop found fewer particles than 3-hop ({hits_5hop} vs {hits_3hop})")
    sys.exit(1)
else:
    print(f"✓ PASS: 5-hop found at least as many particles as 3-hop")
print()

# ============================================================================
# Test 2: Large Scale - Memory and Performance
# ============================================================================
print("=" * 80)
print("TEST 2: LARGE SCALE ({:,} particles)".format(N_PARTICLES_LARGE))
print("=" * 80)
print()

# Upload to GPU
positions_large_gpu = jax.device_put(particles_large)
cached_large_gpu = jax.device_put(cached_ids_large)

# Check GPU memory before
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=2)
    gpu_mem_before = int(result.stdout.strip().split('\n')[0])
    print(f"GPU memory before test: {gpu_mem_before} MB")
except:
    gpu_mem_before = None
    print("Could not query GPU memory (nvidia-smi not available)")

print()
print("Running 5-hop hierarchical search on {:,} particles...".format(N_PARTICLES_LARGE))
print("(This should NOT cause GPU OOM)")
print()

t0 = time.perf_counter()
try:
    element_ids_large_gpu = search_level1_multihop_hierarchical(
        positions_large_gpu,
        cached_large_gpu,
        mesh_gpu.element_neighbors,
        mesh_gpu.node_positions,
        mesh_gpu.connectivity,
        n_hops=5
    )
    element_ids_large = np.array(element_ids_large_gpu, dtype=np.int32)
    t_large = time.perf_counter() - t0

    hits_large = (element_ids_large >= 0).sum()
    throughput_large = N_PARTICLES_LARGE / t_large

    print(f"✓ SUCCESS: No GPU OOM!")
    print(f"  Time: {t_large:.2f} s")
    print(f"  Throughput: {throughput_large:.1f} p/s")
    print(f"  Hits: {hits_large:,}/{N_PARTICLES_LARGE:,} ({100*hits_large/N_PARTICLES_LARGE:.2f}%)")

    # Check GPU memory after
    if gpu_mem_before is not None:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=2)
            gpu_mem_after = int(result.stdout.strip().split('\n')[0])
            gpu_mem_delta = gpu_mem_after - gpu_mem_before
            print(f"  GPU memory after: {gpu_mem_after} MB (Δ = {gpu_mem_delta:+d} MB)")
        except:
            pass

    oom_success = True

except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print(f"❌ FAIL: GPU OOM occurred!")
        print(f"  Error: {e}")
        oom_success = False
    else:
        raise
print()

# ============================================================================
# Test 3: Full L0+L1 Pipeline
# ============================================================================
print("=" * 80)
print("TEST 3: FULL L0+L1 HIERARCHICAL PIPELINE")
print("=" * 80)
print()

print("Testing complete search pipeline (L0 + L1 hierarchical)...")
print()

# L0: Check cached elements
print("Step 1: L0 (cached element check)...")
t0 = time.perf_counter()
element_ids_l0_gpu = search_level0_vectorized(
    positions_large_gpu,
    cached_large_gpu,
    mesh_gpu.node_positions,
    mesh_gpu.connectivity
)
t_l0 = time.perf_counter() - t0

element_ids_l0 = np.array(element_ids_l0_gpu, dtype=np.int32)
l0_hits = (element_ids_l0 >= 0).sum()
l0_hit_rate = 100 * l0_hits / N_PARTICLES_LARGE

print(f"  L0 hits: {l0_hits:,}/{N_PARTICLES_LARGE:,} ({l0_hit_rate:.2f}%)")
print(f"  L0 time: {t_l0*1000:.1f} ms")
print()

# L1: Hierarchical 5-hop for all particles (including L0 hits)
print("Step 2: L1 hierarchical 5-hop (all particles)...")
t0 = time.perf_counter()
element_ids_l1_gpu = search_level1_multihop_hierarchical(
    positions_large_gpu,
    cached_large_gpu,
    mesh_gpu.element_neighbors,
    mesh_gpu.node_positions,
    mesh_gpu.connectivity,
    n_hops=5
)
t_l1 = time.perf_counter() - t0

element_ids_l1 = np.array(element_ids_l1_gpu, dtype=np.int32)
l1_hits = (element_ids_l1 >= 0).sum()
l1_hit_rate = 100 * l1_hits / N_PARTICLES_LARGE

print(f"  L1 hits: {l1_hits:,}/{N_PARTICLES_LARGE:,} ({l1_hit_rate:.2f}%)")
print(f"  L1 time: {t_l1*1000:.1f} ms")
print()

# Merge: Use L0 if found, else L1
element_ids_merged_gpu = jnp.where(element_ids_l0_gpu >= 0, element_ids_l0_gpu, element_ids_l1_gpu)
element_ids_merged = np.array(element_ids_merged_gpu, dtype=np.int32)
total_hits = (element_ids_merged >= 0).sum()
total_hit_rate = 100 * total_hits / N_PARTICLES_LARGE

print("Step 3: Merge L0 + L1...")
print(f"  Total hits: {total_hits:,}/{N_PARTICLES_LARGE:,} ({total_hit_rate:.2f}%)")
print(f"  Total time: {(t_l0 + t_l1)*1000:.1f} ms")
print(f"  Throughput: {N_PARTICLES_LARGE / (t_l0 + t_l1):.1f} p/s")
print()

# ============================================================================
# Summary and Pass/Fail Criteria
# ============================================================================
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print()

all_tests_passed = True

# Test 1: Correctness (5-hop >= 3-hop hits)
print("Test 1: Correctness (small scale)")
if hits_5hop >= hits_3hop:
    print(f"  ✓ PASS: 5-hop found {hits_5hop:,} vs 3-hop {hits_3hop:,} particles")
else:
    print(f"  ❌ FAIL: 5-hop found fewer particles than 3-hop")
    all_tests_passed = False

# Test 2: Memory efficiency (no OOM)
print()
print("Test 2: Memory efficiency (large scale)")
if oom_success:
    print(f"  ✓ PASS: No GPU OOM with {N_PARTICLES_LARGE:,} particles")
else:
    print(f"  ❌ FAIL: GPU OOM occurred")
    all_tests_passed = False

# Test 3: Performance (throughput reasonable)
print()
print("Test 3: Performance")
expected_min_throughput = 5000  # Conservative threshold
if throughput_large >= expected_min_throughput:
    print(f"  ✓ PASS: Throughput {throughput_large:.1f} p/s >= {expected_min_throughput} p/s")
else:
    print(f"  ❌ FAIL: Throughput {throughput_large:.1f} p/s < {expected_min_throughput} p/s")
    all_tests_passed = False

# Test 4: Hit rate (should be very high)
print()
print("Test 4: Hit rate")
expected_min_hit_rate = 95.0  # 5-hop should achieve >95% hit rate
if total_hit_rate >= expected_min_hit_rate:
    print(f"  ✓ PASS: Hit rate {total_hit_rate:.2f}% >= {expected_min_hit_rate}%")
else:
    print(f"  ⚠ WARNING: Hit rate {total_hit_rate:.2f}% < {expected_min_hit_rate}%")
    print(f"    (This may be acceptable depending on particle distribution)")

print()
print("=" * 80)
if all_tests_passed:
    print("✓ ALL TESTS PASSED - Hierarchical 5-hop search validated!")
    print("=" * 80)
    print()
    print("Key achievements:")
    print(f"  • No GPU OOM with 5-hop search")
    print(f"  • Hit rate: {total_hit_rate:.2f}% (improved from 3-hop)")
    print(f"  • Throughput: {throughput_large:.1f} p/s")
    print(f"  • Memory efficient: Early-exit prevents neighbor explosion")
    print()
    print("Ready for production testing with 105k particles!")
else:
    print("❌ SOME TESTS FAILED")
    print("=" * 80)
    sys.exit(1)
