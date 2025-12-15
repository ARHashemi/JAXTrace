#!/usr/bin/env python3
"""
Quick test to verify JIT fix for hierarchical search.
Tests only the search functions, not the full pipeline.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time

print("=" * 80)
print("HIERARCHICAL SEARCH JIT FIX VERIFICATION")
print("=" * 80)
print()

# Check JAX backend
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")
print()

# Generate test data
print("Generating test data...")
N_PARTICLES = 10_000
N_ELEMENTS = 100_000
N_NODES = 30_000

positions = jnp.array(np.random.randn(N_PARTICLES, 3).astype(np.float32))
cached_ids = jnp.array(np.random.randint(0, N_ELEMENTS, N_PARTICLES).astype(np.int32))
element_neighbors = jnp.array(np.random.randint(-1, N_ELEMENTS, (N_ELEMENTS, 4)).astype(np.int32))
node_positions = jnp.array(np.random.randn(N_NODES, 3).astype(np.float32))
connectivity = jnp.array(np.random.randint(0, N_NODES, (N_ELEMENTS, 4)).astype(np.int32))

print(f"Test data:")
print(f"  Particles: {N_PARTICLES:,}")
print(f"  Elements: {N_ELEMENTS:,}")
print(f"  Nodes: {N_NODES:,}")
print()

# Import search functions
from jaxtrace.gpu.tracking.rk4_gpu_fused import create_search_gpu_fused_hierarchical

print("=" * 80)
print("TEST 1: JIT Compilation Time")
print("=" * 80)
print()

# Test 4-hop hierarchical
print("Creating 4-hop hierarchical search function...")
t_start = time.time()
search_func_4hop = create_search_gpu_fused_hierarchical(n_hops=4)
t_create = time.time() - t_start
print(f"  Factory creation: {t_create*1000:.1f} ms")

print()
print("First call (JIT compilation)...")
t_start = time.time()
result_4hop = search_func_4hop(
    positions,
    cached_ids,
    node_positions,
    connectivity,
    element_neighbors
)
result_4hop.block_until_ready()
t_jit = time.time() - t_start
print(f"  ✓ JIT compilation + execution: {t_jit:.2f} s")

# Expected: 2-5 seconds (includes compilation)
if t_jit < 10.0:
    print(f"  ✅ PASS: JIT time reasonable (<10s)")
    jit_test_passed = True
else:
    print(f"  ❌ FAIL: JIT time too long (>{10}s)")
    jit_test_passed = False

print()
print("=" * 80)
print("TEST 2: Execution Time (After JIT)")
print("=" * 80)
print()

# Second call should be fast (pre-compiled)
print("Second call (pre-compiled kernel)...")
t_start = time.time()
result_4hop = search_func_4hop(
    positions,
    cached_ids,
    node_positions,
    connectivity,
    element_neighbors
)
result_4hop.block_until_ready()
t_exec = time.time() - t_start
throughput = N_PARTICLES / t_exec
print(f"  ✓ Execution: {t_exec*1000:.1f} ms")
print(f"  ✓ Throughput: {throughput:,.0f} p/s")

# Expected: >5k p/s
if throughput > 5000:
    print(f"  ✅ PASS: Throughput reasonable (>5k p/s)")
    throughput_test_passed = True
else:
    print(f"  ❌ FAIL: Throughput too low (<5k p/s)")
    throughput_test_passed = False

print()
print("=" * 80)
print("TEST 3: Repeated Calls (No Re-Tracing)")
print("=" * 80)
print()

# Multiple calls should have consistent timing
print("Running 5 repeated calls...")
times = []
for i in range(5):
    t_start = time.time()
    result = search_func_4hop(
        positions,
        cached_ids,
        node_positions,
        connectivity,
        element_neighbors
    )
    result.block_until_ready()
    t_call = time.time() - t_start
    times.append(t_call)
    print(f"  Call {i+1}: {t_call*1000:.1f} ms ({N_PARTICLES/t_call:,.0f} p/s)")

mean_time = np.mean(times)
std_time = np.std(times)
print()
print(f"Mean: {mean_time*1000:.1f} ms ± {std_time*1000:.1f} ms")

# Check consistency (std should be <20% of mean)
if std_time / mean_time < 0.2:
    print(f"  ✅ PASS: Timing consistent (no re-tracing)")
    consistency_test_passed = True
else:
    print(f"  ❌ FAIL: Timing inconsistent (possible re-tracing)")
    consistency_test_passed = False

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

all_passed = jit_test_passed and throughput_test_passed and consistency_test_passed

if all_passed:
    print("✅ ALL TESTS PASSED")
    print()
    print("JIT fix verified:")
    print(f"  - JIT compilation: {t_jit:.2f} s (reasonable)")
    print(f"  - Execution time: {mean_time*1000:.1f} ms (fast)")
    print(f"  - Throughput: {N_PARTICLES/mean_time:,.0f} p/s (good)")
    print(f"  - Timing consistency: ±{std_time/mean_time*100:.1f}% (no re-tracing)")
else:
    print("❌ SOME TESTS FAILED")
    print()
    print("Failed tests:")
    if not jit_test_passed:
        print("  - JIT compilation time too long")
    if not throughput_test_passed:
        print("  - Throughput too low")
    if not consistency_test_passed:
        print("  - Timing inconsistent (possible re-tracing)")

print()
print("=" * 80)
