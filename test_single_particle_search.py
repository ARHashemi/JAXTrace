"""
Test single-particle search implementations.

Comprehensive validation and performance benchmarking:
- Correctness: Element-by-element comparison with batch implementations
- Accuracy: Point-in-tet validation for random sample
- Performance: Detailed timing and throughput metrics
- Hit rates: L0, L1, L2 success statistics
- Memory usage: GPU memory tracking
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import subprocess
import psutil

# Import current batch-level implementations
from jaxtrace.gpu.search.incremental_search_vectorized import (
    search_level0_vectorized,
    search_level1_multihop_hierarchical
)
from jaxtrace.gpu.search.octree_search_gpu import search_level2_octree_scan
from jaxtrace.gpu.search.level0_cached import point_in_tet_jax

# Import new single-particle implementations
from jaxtrace.gpu.search.single_particle_search import (
    search_level0_single,
    search_level1_multihop_single,
    search_level2_octree_single,
    search_single_particle_with_fallback,
    point_in_tet_single_particle
)

print("="*80)
print("SINGLE-PARTICLE SEARCH TEST")
print("="*80)
print()

# Load mesh data from production script
print("Loading mesh data...")
sys.path.insert(0, '/home/arhashemi/Workspace/welding/JAXTrace')

from production_tracking_3hop_l2_octree import (
    mesh_gpu,
    octree_gpu,
    node_positions,
    connectivity
)

print(f"✓ Mesh loaded: {len(connectivity):,} elements, {len(node_positions):,} nodes")
print(f"✓ Octree loaded: {len(octree_gpu.node_metadata):,} nodes")
print()

# Generate test particles
N_TEST = 1000
print(f"Generating {N_TEST:,} test particles...")

# Use refined region bounds from production script
x_min, x_max = -0.03 * 0.3, -0.03 * 0.3 + 0.01  # Refined region
y_min, y_max = -0.023, 0.023
z_min, z_max = -0.01, 0.0

np.random.seed(42)
positions_np = np.column_stack([
    np.random.uniform(x_min, x_max, N_TEST),
    np.random.uniform(y_min, y_max, N_TEST),
    np.random.uniform(z_min, z_max, N_TEST)
])

positions = jnp.array(positions_np)

# Assign initial elements (use hash bucket search from production)
from jaxtrace.gpu.search.initial_assignment import search_level2b_hash_bucket
print("Initial assignment...")
element_ids_init = search_level2b_hash_bucket(
    positions,
    mesh_gpu.padded_arrays.block_ids,
    mesh_gpu.padded_arrays.block_element_lists,
    mesh_gpu.node_positions,
    mesh_gpu.connectivity
)
cached_ids = element_ids_init

n_found = jnp.sum(cached_ids >= 0).item()
print(f"✓ Initial assignment: {n_found}/{N_TEST} found ({100*n_found/N_TEST:.1f}%)")
print()

# Move particles slightly (simulate small timestep)
dt = 0.0025
velocities = jnp.ones((N_TEST, 3)) * 0.01  # Small velocity
positions_moved = positions + dt * velocities

print("="*80)
print("TEST 1: L0 Search - Correctness, Accuracy, Performance")
print("="*80)

# GPU memory before
gpu_mem_before = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True).stdout.strip())

# Batch-level L0
t0 = time.perf_counter()
element_ids_l0_batch = search_level0_vectorized(
    positions_moved,
    cached_ids,
    mesh_gpu.node_positions,
    mesh_gpu.connectivity
)
element_ids_l0_batch.block_until_ready()
t_l0_batch = time.perf_counter() - t0

# Single-particle L0 with vmap
@jax.jit
def l0_single_vmapped(positions, cached_ids, node_pos, conn):
    return jax.vmap(search_level0_single)(
        positions, cached_ids, node_pos, conn
    )

# Warm-up
_ = l0_single_vmapped(positions_moved[:10], cached_ids[:10], mesh_gpu.node_positions, mesh_gpu.connectivity)

t0 = time.perf_counter()
element_ids_l0_single = l0_single_vmapped(
    positions_moved,
    cached_ids,
    mesh_gpu.node_positions,
    mesh_gpu.connectivity
)
element_ids_l0_single.block_until_ready()
t_l0_single = time.perf_counter() - t0

# GPU memory after
gpu_mem_after = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True).stdout.strip())

# Correctness: Element-by-element comparison
l0_match = jnp.allclose(element_ids_l0_batch, element_ids_l0_single)
n_mismatch = jnp.sum(element_ids_l0_batch != element_ids_l0_single).item()
l0_hits_batch = jnp.sum(element_ids_l0_batch >= 0).item()
l0_hits_single = jnp.sum(element_ids_l0_single >= 0).item()

# Accuracy: Point-in-tet validation for found particles
n_validate_l0 = min(500, l0_hits_single)
found_indices = jnp.where(element_ids_l0_single >= 0)[0]
validate_indices = np.random.choice(found_indices, n_validate_l0, replace=False)

n_correct_l0 = 0
for i in validate_indices:
    pos = positions_moved[i]
    elem_id = element_ids_l0_single[i]
    node_ids = connectivity[elem_id]
    tet_nodes = node_positions[node_ids]
    is_inside = bool(np.array(point_in_tet_jax(jax.device_put(pos), jax.device_put(tet_nodes))))
    if is_inside:
        n_correct_l0 += 1

accuracy_l0 = 100 * n_correct_l0 / n_validate_l0 if n_validate_l0 > 0 else 0

# Performance metrics
throughput_batch = N_TEST / t_l0_batch if t_l0_batch > 0 else 0
throughput_single = N_TEST / t_l0_single if t_l0_single > 0 else 0
speedup = t_l0_batch / t_l0_single if t_l0_single > 0 else 0

print("Correctness:")
print(f"  Batch hits:    {l0_hits_batch}/{N_TEST} ({100*l0_hits_batch/N_TEST:.1f}%)")
print(f"  Single hits:   {l0_hits_single}/{N_TEST} ({100*l0_hits_single/N_TEST:.1f}%)")
print(f"  Results match: {l0_match}")
print(f"  Mismatches:    {n_mismatch}")
print()
print("Accuracy (point-in-tet validation):")
print(f"  Validated:     {n_validate_l0} particles")
print(f"  Correct:       {n_correct_l0}/{n_validate_l0} ({accuracy_l0:.1f}%)")
print()
print("Performance:")
print(f"  Batch time:    {t_l0_batch*1000:.2f} ms ({throughput_batch:,.0f} p/s)")
print(f"  Single time:   {t_l0_single*1000:.2f} ms ({throughput_single:,.0f} p/s)")
print(f"  Speedup:       {speedup:.2f}×")
print(f"  GPU memory:    {gpu_mem_before} MB → {gpu_mem_after} MB (Δ{gpu_mem_after - gpu_mem_before} MB)")
print()

# TEST 2: L1 Search
print("="*80)
print("TEST 2: L1 Multi-Hop Search - Correctness, Accuracy, Performance")
print("="*80)

N_HOPS = 5

# GPU memory before
gpu_mem_before = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True).stdout.strip())

# Batch-level L1
t0 = time.perf_counter()
element_ids_l1_batch = search_level1_multihop_hierarchical(
    positions_moved,
    cached_ids,
    mesh_gpu.element_neighbors,
    mesh_gpu.node_positions,
    mesh_gpu.connectivity,
    n_hops=N_HOPS
)
element_ids_l1_batch.block_until_ready()
t_l1_batch = time.perf_counter() - t0

# Single-particle L1 with vmap
@jax.jit
def l1_single_vmapped(positions, cached_ids, neighbors, node_pos, conn, n_hops):
    def search_one(pos, cached_id):
        return search_level1_multihop_single(
            pos, cached_id, neighbors, node_pos, conn, n_hops
        )
    return jax.vmap(search_one)(positions, cached_ids)

# Warm-up
_ = l1_single_vmapped(positions_moved[:10], cached_ids[:10], mesh_gpu.element_neighbors,
                      mesh_gpu.node_positions, mesh_gpu.connectivity, N_HOPS)

t0 = time.perf_counter()
element_ids_l1_single = l1_single_vmapped(
    positions_moved,
    cached_ids,
    mesh_gpu.element_neighbors,
    mesh_gpu.node_positions,
    mesh_gpu.connectivity,
    N_HOPS
)
element_ids_l1_single.block_until_ready()
t_l1_single = time.perf_counter() - t0

# GPU memory after
gpu_mem_after = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True).stdout.strip())

# Correctness: Element-by-element comparison
l1_match = jnp.allclose(element_ids_l1_batch, element_ids_l1_single)
n_mismatch = jnp.sum(element_ids_l1_batch != element_ids_l1_single).item()
l1_hits_batch = jnp.sum(element_ids_l1_batch >= 0).item()
l1_hits_single = jnp.sum(element_ids_l1_single >= 0).item()

# Accuracy: Point-in-tet validation for found particles (excluding L0 hits)
l1_only_mask = (element_ids_l1_single >= 0) & (element_ids_l0_single < 0)
l1_only_indices = jnp.where(l1_only_mask)[0]
n_l1_only = len(l1_only_indices)
n_validate_l1 = min(500, n_l1_only)

n_correct_l1 = 0
if n_validate_l1 > 0:
    validate_indices = np.random.choice(l1_only_indices, n_validate_l1, replace=False)
    for i in validate_indices:
        pos = positions_moved[i]
        elem_id = element_ids_l1_single[i]
        node_ids = connectivity[elem_id]
        tet_nodes = node_positions[node_ids]
        is_inside = bool(np.array(point_in_tet_jax(jax.device_put(pos), jax.device_put(tet_nodes))))
        if is_inside:
            n_correct_l1 += 1

accuracy_l1 = 100 * n_correct_l1 / n_validate_l1 if n_validate_l1 > 0 else 0

# Performance metrics
throughput_batch = N_TEST / t_l1_batch if t_l1_batch > 0 else 0
throughput_single = N_TEST / t_l1_single if t_l1_single > 0 else 0
speedup = t_l1_batch / t_l1_single if t_l1_single > 0 else 0

print(f"Multi-hop level: {N_HOPS} hops")
print()
print("Correctness:")
print(f"  Batch hits:    {l1_hits_batch}/{N_TEST} ({100*l1_hits_batch/N_TEST:.1f}%)")
print(f"  Single hits:   {l1_hits_single}/{N_TEST} ({100*l1_hits_single/N_TEST:.1f}%)")
print(f"  L1-only hits:  {n_l1_only} (excluding L0 hits)")
print(f"  Results match: {l1_match}")
print(f"  Mismatches:    {n_mismatch}")
print()
print("Accuracy (point-in-tet validation, L1-only particles):")
print(f"  Validated:     {n_validate_l1} particles")
print(f"  Correct:       {n_correct_l1}/{n_validate_l1} ({accuracy_l1:.1f}%)")
print()
print("Performance:")
print(f"  Batch time:    {t_l1_batch*1000:.2f} ms ({throughput_batch:,.0f} p/s)")
print(f"  Single time:   {t_l1_single*1000:.2f} ms ({throughput_single:,.0f} p/s)")
print(f"  Speedup:       {speedup:.2f}×")
print(f"  GPU memory:    {gpu_mem_before} MB → {gpu_mem_after} MB (Δ{gpu_mem_after - gpu_mem_before} MB)")
print()

# TEST 3: L2 Octree Search
print("="*80)
print("TEST 3: L2 Octree Search - Correctness, Accuracy, Performance")
print("="*80)

# Merge L0+L1 to get unfound particles
element_ids_l0_l1_batch = jnp.where(element_ids_l0_batch >= 0, element_ids_l0_batch, element_ids_l1_batch)
element_ids_l0_l1_single = jnp.where(element_ids_l0_single >= 0, element_ids_l0_single, element_ids_l1_single)
n_unfound = jnp.sum(element_ids_l0_l1_batch < 0).item()
print(f"Particles needing L2: {n_unfound}/{N_TEST} ({100*n_unfound/N_TEST:.1f}%)")
print()

# GPU memory before
gpu_mem_before = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True).stdout.strip())

# Batch-level L2
t0 = time.perf_counter()
element_ids_l2_batch = search_level2_octree_scan(
    positions_moved,
    element_ids_l0_l1_batch,
    octree_gpu.node_metadata,
    octree_gpu.node_elements,
    mesh_gpu.node_positions,
    mesh_gpu.connectivity,
    max_depth=10
)
element_ids_l2_batch.block_until_ready()
t_l2_batch = time.perf_counter() - t0

# Single-particle L2 with vmap
@jax.jit
def l2_single_vmapped(positions, octree_meta, octree_elems, node_pos, conn):
    def search_one(pos):
        return search_level2_octree_single(
            pos, octree_meta, octree_elems, node_pos, conn, max_depth=10
        )
    return jax.vmap(search_one)(positions)

# Warm-up
_ = l2_single_vmapped(positions_moved[:10], octree_gpu.node_metadata, octree_gpu.node_elements,
                      mesh_gpu.node_positions, mesh_gpu.connectivity)

t0 = time.perf_counter()
element_ids_l2_single_all = l2_single_vmapped(
    positions_moved,
    octree_gpu.node_metadata,
    octree_gpu.node_elements,
    mesh_gpu.node_positions,
    mesh_gpu.connectivity
)
element_ids_l2_single_all.block_until_ready()
t_l2_single = time.perf_counter() - t0

# Merge with L0+L1 results
element_ids_l2_single = jnp.where(
    element_ids_l0_l1_single >= 0,
    element_ids_l0_l1_single,
    element_ids_l2_single_all
)

# GPU memory after
gpu_mem_after = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True).stdout.strip())

# Correctness: Element-by-element comparison
l2_match = jnp.allclose(element_ids_l2_batch, element_ids_l2_single)
n_mismatch = jnp.sum(element_ids_l2_batch != element_ids_l2_single).item()
l2_hits_batch = jnp.sum(element_ids_l2_batch >= 0).item()
l2_hits_single = jnp.sum(element_ids_l2_single >= 0).item()

# Accuracy: Point-in-tet validation for L2-only particles
l2_only_mask = (element_ids_l2_single >= 0) & (element_ids_l0_l1_single < 0)
l2_only_indices = jnp.where(l2_only_mask)[0]
n_l2_only = len(l2_only_indices)
n_validate_l2 = min(500, n_l2_only)

n_correct_l2 = 0
if n_validate_l2 > 0:
    validate_indices = np.random.choice(l2_only_indices, n_validate_l2, replace=False)
    for i in validate_indices:
        pos = positions_moved[i]
        elem_id = element_ids_l2_single[i]
        node_ids = connectivity[elem_id]
        tet_nodes = node_positions[node_ids]
        is_inside = bool(np.array(point_in_tet_jax(jax.device_put(pos), jax.device_put(tet_nodes))))
        if is_inside:
            n_correct_l2 += 1

accuracy_l2 = 100 * n_correct_l2 / n_validate_l2 if n_validate_l2 > 0 else 0

# Performance metrics
throughput_batch = N_TEST / t_l2_batch if t_l2_batch > 0 else 0
throughput_single = N_TEST / t_l2_single if t_l2_single > 0 else 0
speedup = t_l2_batch / t_l2_single if t_l2_single > 0 else 0

print("Correctness:")
print(f"  Batch hits:    {l2_hits_batch}/{N_TEST} ({100*l2_hits_batch/N_TEST:.1f}%)")
print(f"  Single hits:   {l2_hits_single}/{N_TEST} ({100*l2_hits_single/N_TEST:.1f}%)")
print(f"  L2-only hits:  {n_l2_only} (excluding L0+L1 hits)")
print(f"  Results match: {l2_match}")
print(f"  Mismatches:    {n_mismatch}")
print()
print("Accuracy (point-in-tet validation, L2-only particles):")
print(f"  Validated:     {n_validate_l2} particles")
print(f"  Correct:       {n_correct_l2}/{n_validate_l2} ({accuracy_l2:.1f}%)")
print()
print("Performance:")
print(f"  Batch time:    {t_l2_batch*1000:.2f} ms ({throughput_batch:,.0f} p/s)")
print(f"  Single time:   {t_l2_single*1000:.2f} ms ({throughput_single:,.0f} p/s)")
print(f"  Speedup:       {speedup:.2f}×")
print(f"  GPU memory:    {gpu_mem_before} MB → {gpu_mem_after} MB (Δ{gpu_mem_after - gpu_mem_before} MB)")
print()

# TEST 4: Fused Search
print("="*80)
print("TEST 4: Fused L0+L1+L2 Search - Complete Pipeline")
print("="*80)

# GPU memory before
gpu_mem_before = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True).stdout.strip())

@jax.jit
def fused_single_vmapped(positions, cached_ids, node_pos, conn, neighbors, octree_meta, octree_elems):
    def search_one(pos, cached_id):
        return search_single_particle_with_fallback(
            pos, cached_id, node_pos, conn, neighbors,
            octree_meta, octree_elems, n_hops=5, max_octree_depth=10
        )
    return jax.vmap(search_one)(positions, cached_ids)

# Warm-up
_ = fused_single_vmapped(positions_moved[:10], cached_ids[:10], mesh_gpu.node_positions,
                         mesh_gpu.connectivity, mesh_gpu.element_neighbors,
                         octree_gpu.node_metadata, octree_gpu.node_elements)

t0 = time.perf_counter()
element_ids_fused = fused_single_vmapped(
    positions_moved,
    cached_ids,
    mesh_gpu.node_positions,
    mesh_gpu.connectivity,
    mesh_gpu.element_neighbors,
    octree_gpu.node_metadata,
    octree_gpu.node_elements
)
element_ids_fused.block_until_ready()
t_fused = time.perf_counter() - t0

# GPU memory after
gpu_mem_after = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True).stdout.strip())

# Correctness
fused_match = jnp.allclose(element_ids_l2_batch, element_ids_fused)
n_mismatch = jnp.sum(element_ids_l2_batch != element_ids_fused).item()
fused_hits = jnp.sum(element_ids_fused >= 0).item()

# Accuracy: Point-in-tet validation for all found particles
n_validate_fused = min(1000, fused_hits)
found_indices = jnp.where(element_ids_fused >= 0)[0]
validate_indices = np.random.choice(found_indices, n_validate_fused, replace=False)

n_correct_fused = 0
for i in validate_indices:
    pos = positions_moved[i]
    elem_id = element_ids_fused[i]
    node_ids = connectivity[elem_id]
    tet_nodes = node_positions[node_ids]
    is_inside = bool(np.array(point_in_tet_jax(jax.device_put(pos), jax.device_put(tet_nodes))))
    if is_inside:
        n_correct_fused += 1

accuracy_fused = 100 * n_correct_fused / n_validate_fused if n_validate_fused > 0 else 0

# Performance
throughput_fused = N_TEST / t_fused if t_fused > 0 else 0

print("Correctness:")
print(f"  Fused hits:         {fused_hits}/{N_TEST} ({100*fused_hits/N_TEST:.1f}%)")
print(f"  Matches batch:      {fused_match}")
print(f"  Mismatches:         {n_mismatch}")
print()
print("Accuracy (point-in-tet validation, all found particles):")
print(f"  Validated:          {n_validate_fused} particles")
print(f"  Correct:            {n_correct_fused}/{n_validate_fused} ({accuracy_fused:.1f}%)")
print()
print("Performance:")
print(f"  Fused time:         {t_fused*1000:.2f} ms ({throughput_fused:,.0f} p/s)")
print(f"  GPU memory:         {gpu_mem_before} MB → {gpu_mem_after} MB (Δ{gpu_mem_after - gpu_mem_before} MB)")
print()

# Summary
print("="*80)
print("COMPREHENSIVE SUMMARY")
print("="*80)
print()

print("=" * 80)
print("CORRECTNESS VALIDATION")
print("=" * 80)
correctness_table = f"""
Level        Batch Hits    Single Hits   Match     Mismatches
------------ ------------- ------------- --------- -----------
L0           {l0_hits_batch:5d}/{N_TEST:5d}    {l0_hits_single:5d}/{N_TEST:5d}    {str(l0_match):9s} {jnp.sum(element_ids_l0_batch != element_ids_l0_single).item():5d}
L1           {l1_hits_batch:5d}/{N_TEST:5d}    {l1_hits_single:5d}/{N_TEST:5d}    {str(l1_match):9s} {jnp.sum(element_ids_l1_batch != element_ids_l1_single).item():5d}
L2           {l2_hits_batch:5d}/{N_TEST:5d}    {l2_hits_single:5d}/{N_TEST:5d}    {str(l2_match):9s} {jnp.sum(element_ids_l2_batch != element_ids_l2_single).item():5d}
Fused        {l2_hits_batch:5d}/{N_TEST:5d}    {fused_hits:5d}/{N_TEST:5d}    {str(fused_match):9s} {n_mismatch:5d}
"""
print(correctness_table)

print("=" * 80)
print("ACCURACY VALIDATION (Point-in-Tet)")
print("=" * 80)
accuracy_table = f"""
Level        Validated     Correct       Accuracy
------------ ------------- ------------- ---------
L0           {n_validate_l0:5d}         {n_correct_l0:5d}         {accuracy_l0:6.1f}%
L1 (only)    {n_validate_l1:5d}         {n_correct_l1:5d}         {accuracy_l1:6.1f}%
L2 (only)    {n_validate_l2:5d}         {n_correct_l2:5d}         {accuracy_l2:6.1f}%
Fused (all)  {n_validate_fused:5d}         {n_correct_fused:5d}         {accuracy_fused:6.1f}%
"""
print(accuracy_table)

print("=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)
performance_table = f"""
Level        Batch Time    Single Time   Throughput (single)   Speedup
------------ ------------- ------------- --------------------- --------
L0           {t_l0_batch*1000:7.2f} ms    {t_l0_single*1000:7.2f} ms    {N_TEST/t_l0_single:10,.0f} p/s       {t_l0_batch/t_l0_single:6.2f}×
L1           {t_l1_batch*1000:7.2f} ms    {t_l1_single*1000:7.2f} ms    {N_TEST/t_l1_single:10,.0f} p/s       {t_l1_batch/t_l1_single:6.2f}×
L2           {t_l2_batch*1000:7.2f} ms    {t_l2_single*1000:7.2f} ms    {N_TEST/t_l2_single:10,.0f} p/s       {t_l2_batch/t_l2_single:6.2f}×
Fused        N/A           {t_fused*1000:7.2f} ms    {throughput_fused:10,.0f} p/s       N/A
"""
print(performance_table)

print("=" * 80)
print("HIT RATE BREAKDOWN")
print("=" * 80)
hit_rate_table = f"""
Level        Hits          Miss Rate     Cumulative Coverage
------------ ------------- ------------- --------------------
L0           {l0_hits_single:5d}         {100*(N_TEST-l0_hits_single)/N_TEST:6.1f}%        {100*l0_hits_single/N_TEST:6.1f}%
L1           {n_l1_only:5d}         {100*(N_TEST-l1_hits_single)/N_TEST:6.1f}%        {100*l1_hits_single/N_TEST:6.1f}%
L2           {n_l2_only:5d}         {100*(N_TEST-fused_hits)/N_TEST:6.1f}%        {100*fused_hits/N_TEST:6.1f}%
"""
print(hit_rate_table)

print("=" * 80)
print("FINAL VERDICT")
print("=" * 80)
all_passed = l0_match and l1_match and l2_match and fused_match
all_accurate = (accuracy_l0 > 99.0 and (accuracy_l1 > 99.0 or n_validate_l1 == 0) and
                (accuracy_l2 > 99.0 or n_validate_l2 == 0) and accuracy_fused > 99.0)

if all_passed and all_accurate:
    print("✓ ALL TESTS PASSED")
    print("  • Correctness: Single-particle implementations match batch versions")
    print(f"  • Accuracy: {accuracy_fused:.1f}% point-in-tet validation")
    print("  • Performance: Speedups documented above")
    print()
    print("READY FOR INTEGRATION INTO RK4")
else:
    print("✗ TESTS FAILED")
    if not all_passed:
        print("  Correctness issues:")
        if not l0_match:
            print("    • L0 results do not match batch implementation")
        if not l1_match:
            print("    • L1 results do not match batch implementation")
        if not l2_match:
            print("    • L2 results do not match batch implementation")
        if not fused_match:
            print("    • Fused results do not match batch implementation")
    if not all_accurate:
        print("  Accuracy issues:")
        if accuracy_l0 <= 99.0:
            print(f"    • L0 accuracy: {accuracy_l0:.1f}% (expected >99%)")
        if n_validate_l1 > 0 and accuracy_l1 <= 99.0:
            print(f"    • L1 accuracy: {accuracy_l1:.1f}% (expected >99%)")
        if n_validate_l2 > 0 and accuracy_l2 <= 99.0:
            print(f"    • L2 accuracy: {accuracy_l2:.1f}% (expected >99%)")
        if accuracy_fused <= 99.0:
            print(f"    • Fused accuracy: {accuracy_fused:.1f}% (expected >99%)")

print()
print("="*80)
