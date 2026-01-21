#!/usr/bin/env python3
"""
Validation and benchmark script for point-in-tet inverse matrix method.

Tests:
1. Correctness: 100% agreement with current baseline method
2. Performance: Measure speedup (expect 3-4×)
3. Edge cases: Points near faces, edges, vertices
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from jaxtrace.gpu.search.point_in_tet_inverse import (
    precompute_inverse_matrices,
    point_in_tet_inverse,
    point_in_tet_inverse_batch
)
from jaxtrace.gpu.search.point_in_tet_methods import (
    point_in_tet_current,
    point_in_tet_skala,
    set_inverse_matrices_gpu
)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

print("=" * 80)
print("POINT-IN-TET INVERSE MATRIX VALIDATION & BENCHMARK")
print("=" * 80)

# Load production mesh (same pattern as production_tracking_fully_fused_timedep.py)
print("\n1. Loading production mesh...")
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 121)  # Load only 2 timesteps for test (minimal)
VELOCITY_FIELD_NAME = 'Displacement'

try:
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )

    # Deduplicate nodes (PVTU piece boundary fix)
    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )

    n_elements = connectivity.shape[0]
    n_nodes = node_positions.shape[0]

    print(f"   ✓ Loaded mesh from PVTU sequence")
    print(f"   Elements: {n_elements:,}")
    print(f"   Nodes: {n_nodes:,}")
    if n_duplicates_removed > 0:
        print(f"   Duplicates removed: {n_duplicates_removed:,}")
except Exception as e:
    print(f"   ✗ Failed to load mesh: {e}")
    print(f"   Please ensure mesh files exist at {MESH_BASE_PATH}")
    exit(1)

# Precompute inverse matrices
print("\n2. Precomputing inverse matrices...")
try:
    M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)
    print(f"   ✓ Precomputed inverse matrices")
except Exception as e:
    print(f"   ✗ Failed to precompute: {e}")
    exit(1)

# Upload to GPU
print("\n3. Uploading to GPU...")
try:
    connectivity_gpu = jax.device_put(connectivity)
    node_positions_gpu = jax.device_put(node_positions)
    M_inv_gpu = jax.device_put(M_inv_array)
    p0_gpu = jax.device_put(p0_array)

    # Register with dispatcher
    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)

    print(f"   ✓ Uploaded to GPU")
    total_memory_mb = (
        connectivity.nbytes + node_positions.nbytes +
        M_inv_array.nbytes + p0_array.nbytes
    ) / (1024**2)
    print(f"   Total GPU memory: {total_memory_mb:.1f} MB")
except Exception as e:
    print(f"   ✗ Failed to upload: {e}")
    exit(1)

# Generate comprehensive test dataset
print("\n4. Generating test positions...")
n_test = 100000
rng = np.random.RandomState(42)

bbox_min = node_positions.min(axis=0)
bbox_max = node_positions.max(axis=0)
bbox_range = bbox_max - bbox_min

# Generate diverse test positions
test_positions_list = []
test_elem_ids_list = []

# 50% random interior points
n_interior = n_test // 2
for i in range(n_interior):
    test_positions_list.append(bbox_min + rng.random(3) * bbox_range)
    test_elem_ids_list.append(rng.randint(0, n_elements))

# 30% points near element centers (high probability of being inside)
n_centers = int(0.3 * n_test)
for i in range(n_centers):
    elem_id = rng.randint(0, n_elements)
    nodes = node_positions[connectivity[elem_id]]
    center = nodes.mean(axis=0)
    # Jitter slightly around center
    jitter = (rng.random(3) - 0.5) * 0.1 * bbox_range / 100
    test_positions_list.append(center + jitter)
    test_elem_ids_list.append(elem_id)

# 20% points near boundary (edge cases)
n_boundary = n_test - n_interior - n_centers
for i in range(n_boundary):
    # Random point on bbox faces
    face = rng.randint(0, 6)  # 6 faces
    pos = bbox_min + rng.random(3) * bbox_range
    if face < 2:  # x faces
        pos[0] = bbox_min[0] if face == 0 else bbox_max[0]
    elif face < 4:  # y faces
        pos[1] = bbox_min[1] if face == 2 else bbox_max[1]
    else:  # z faces
        pos[2] = bbox_min[2] if face == 4 else bbox_max[2]
    test_positions_list.append(pos)
    test_elem_ids_list.append(rng.randint(0, n_elements))

test_positions = np.array(test_positions_list, dtype=np.float32)
test_elem_ids = np.array(test_elem_ids_list, dtype=np.int32)

test_positions_gpu = jax.device_put(test_positions)
test_elem_ids_gpu = jax.device_put(test_elem_ids)

print(f"   ✓ Generated {n_test:,} test queries")
print(f"   Distribution:")
print(f"   - Interior random: {n_interior:,} ({100*n_interior/n_test:.0f}%)")
print(f"   - Near centers: {n_centers:,} ({100*n_centers/n_test:.0f}%)")
print(f"   - Near boundary: {n_boundary:,} ({100*n_boundary/n_test:.0f}%)")

# Correctness validation
print("\n5. Validating correctness (100% agreement required)...")
print("   Comparing: inverse vs current (baseline)")

# Create JIT-compiled comparison functions
@jax.jit
def test_current(pos, elem_id):
    return point_in_tet_current(pos, elem_id, connectivity_gpu, node_positions_gpu)

@jax.jit
def test_inverse(pos, elem_id):
    return point_in_tet_inverse(pos, elem_id, M_inv_gpu, p0_gpu)

# Vectorize
test_current_vmap = jax.jit(jax.vmap(test_current))
test_inverse_vmap = jax.jit(jax.vmap(test_inverse))

# Warmup
_ = test_current_vmap(test_positions_gpu[:100], test_elem_ids_gpu[:100])
_ = test_inverse_vmap(test_positions_gpu[:100], test_elem_ids_gpu[:100])
jax.block_until_ready(_)

print("   Running validation...")
results_current = test_current_vmap(test_positions_gpu, test_elem_ids_gpu)
results_inverse = test_inverse_vmap(test_positions_gpu, test_elem_ids_gpu)
jax.block_until_ready(results_current)
jax.block_until_ready(results_inverse)

# Compare results
results_current_np = np.array(results_current)
results_inverse_np = np.array(results_inverse)

agreements = (results_current_np == results_inverse_np)
agreement_count = np.sum(agreements)
agreement_rate = 100.0 * agreement_count / n_test

print(f"\n   Agreement: {agreement_count:,} / {n_test:,} ({agreement_rate:.6f}%)")

if agreement_rate < 100.0:
    disagreements = np.where(~agreements)[0]
    print(f"   ✗ VALIDATION FAILED!")
    print(f"   Disagreements: {len(disagreements):,}")
    print(f"\n   First 10 disagreements:")
    for i in disagreements[:10]:
        print(f"   Query {i}: pos={test_positions[i]}, elem={test_elem_ids[i]}")
        print(f"     Current: {results_current_np[i]}, Inverse: {results_inverse_np[i]}")
    exit(1)
else:
    print(f"   ✓ VALIDATION PASSED: 100% agreement!")

# Performance benchmark
print("\n6. Benchmarking performance...")
print(f"   Testing {n_test:,} queries × 5 iterations")

# Benchmark current method
print("\n   Current (baseline) method:")
times_current = []
for iteration in range(5):
    start = time.time()
    results = test_current_vmap(test_positions_gpu, test_elem_ids_gpu)
    jax.block_until_ready(results)
    elapsed = time.time() - start
    times_current.append(elapsed)
    throughput = n_test / elapsed
    print(f"   Iteration {iteration+1}: {elapsed:.3f}s ({throughput:,.0f} queries/s)")

avg_current = np.mean(times_current)
std_current = np.std(times_current)
throughput_current = n_test / avg_current

# Benchmark inverse method
print("\n   Inverse matrix method:")
times_inverse = []
for iteration in range(5):
    start = time.time()
    results = test_inverse_vmap(test_positions_gpu, test_elem_ids_gpu)
    jax.block_until_ready(results)
    elapsed = time.time() - start
    times_inverse.append(elapsed)
    throughput = n_test / elapsed
    print(f"   Iteration {iteration+1}: {elapsed:.3f}s ({throughput:,.0f} queries/s)")

avg_inverse = np.mean(times_inverse)
std_inverse = np.std(times_inverse)
throughput_inverse = n_test / avg_inverse

# Compute speedup
speedup = avg_current / avg_inverse

print(f"\n   Current:  {avg_current:.3f}s ± {std_current:.3f}s ({throughput_current:,.0f} queries/s)")
print(f"   Inverse:  {avg_inverse:.3f}s ± {std_inverse:.3f}s ({throughput_inverse:,.0f} queries/s)")
print(f"   Speedup:  {speedup:.2f}×")

# Compare to Skala
print("\n7. Comparing to skala_memory_opt...")
try:
    from jaxtrace.gpu.search.aa_detection import precompute_element_vertices, point_in_tet_skala_memory_opt
    from jaxtrace.gpu.search.point_in_tet_methods import set_corrected_metadata, AxisAlignedMetadata

    # Precompute element vertices
    element_vertices_cpu = precompute_element_vertices(connectivity, node_positions, verbose=False)
    element_vertices_gpu = jax.device_put(element_vertices_cpu)

    # Register (with dummy AA metadata)
    dummy_aa = AxisAlignedMetadata(
        base_vertex_indices=jax.device_put(np.zeros(1, dtype=np.int8)),
        base_vertices=jax.device_put(np.zeros((1, 3), dtype=np.float32)),
        inv_edge_lengths=jax.device_put(np.zeros((1, 3), dtype=np.float32)),
        axis_indices=jax.device_put(np.zeros((1, 3), dtype=np.int8)),
        is_axis_aligned=jax.device_put(np.zeros(1, dtype=bool))
    )
    set_corrected_metadata(dummy_aa, element_vertices_gpu)

    @jax.jit
    def test_skala_opt(pos, elem_id):
        return point_in_tet_skala_memory_opt(pos, elem_id, element_vertices_gpu)

    test_skala_opt_vmap = jax.jit(jax.vmap(test_skala_opt))

    # Warmup
    _ = test_skala_opt_vmap(test_positions_gpu[:100], test_elem_ids_gpu[:100])
    jax.block_until_ready(_)

    # Benchmark
    print("   Running skala_memory_opt benchmark...")
    times_skala = []
    for iteration in range(5):
        start = time.time()
        results = test_skala_opt_vmap(test_positions_gpu, test_elem_ids_gpu)
        jax.block_until_ready(results)
        elapsed = time.time() - start
        times_skala.append(elapsed)

    avg_skala = np.mean(times_skala)
    throughput_skala = n_test / avg_skala
    speedup_vs_skala = avg_skala / avg_inverse

    print(f"   Skala:    {avg_skala:.3f}s ({throughput_skala:,.0f} queries/s)")
    print(f"   Inverse vs Skala speedup: {speedup_vs_skala:.2f}×")

except Exception as e:
    print(f"   (Skipped: {e})")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Correctness: 100% agreement with baseline")
print(f"✓ Performance: {speedup:.2f}× speedup over current method")
print(f"  - Current:  {throughput_current:,.0f} queries/s")
print(f"  - Inverse:  {throughput_inverse:,.0f} queries/s")

if speedup >= 3.0:
    print(f"\n✓ EXCELLENT: Achieved {speedup:.2f}× speedup (target: 3-4×)")
elif speedup >= 2.0:
    print(f"\n✓ GOOD: Achieved {speedup:.2f}× speedup (slightly below 3-4× target)")
else:
    print(f"\n⚠ WARNING: Only {speedup:.2f}× speedup (expected 3-4×)")
    print("  This may indicate memory bandwidth saturation")

print("\nNext steps:")
print("1. Run production test with POINT_IN_TET_METHOD='inverse'")
print("2. Verify retention % is identical to current method")
print("3. Measure end-to-end speedup in RK4 tracking")
print("=" * 80)
