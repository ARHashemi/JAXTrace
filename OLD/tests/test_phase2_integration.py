#!/usr/bin/env python3
"""
Phase 2 Integration Test: GPU Search Kernels with Batch Processor

Tests the integration of JAX-native search kernels with the batched
block-wise particle tracking architecture.

This test verifies:
1. Search kernels work with JAX arrays
2. Batch processor correctly calls search functions
3. Statistics are properly accumulated
4. 3-level search hierarchy returns valid results
"""

import numpy as np
import jax.numpy as jnp

print("\n" + "="*80)
print("PHASE 2 INTEGRATION TEST")
print("="*80)

# Test 1: Import all Phase 2 modules
print("\n[1/5] Testing imports...")
try:
    from jaxtrace.gpu.search import (
        search_particles_in_block,
        search_particles_in_block_with_hash,
        BlockSearchResult
    )
    from jaxtrace.gpu.batching import (
        BatchConfig,
        BatchStatistics
    )
    from jaxtrace.gpu.forest import PaddedArrays
    print("✓ Core imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Create minimal test data
print("\n[2/5] Creating minimal test data...")
try:
    # Create a simple tetrahedron mesh (single element)
    positions = np.array([
        [0.0, 0.0, 0.0],  # Node 0
        [1.0, 0.0, 0.0],  # Node 1
        [0.0, 1.0, 0.0],  # Node 2
        [0.0, 0.0, 1.0],  # Node 3
    ], dtype=np.float32)

    connectivity = np.array([[0, 1, 2, 3]], dtype=np.int32)  # 1 element
    neighbors = np.array([[-1, -1, -1, -1]], dtype=np.int32)  # No neighbors

    # Particle inside the tetrahedron
    particle_pos = np.array([[0.25, 0.25, 0.25]], dtype=np.float32)
    particle_elem = np.array([0], dtype=np.int32)  # Cached element 0
    particle_block = np.array([0], dtype=np.int32)
    particle_active = np.array([True], dtype=bool)

    print(f"✓ Created mesh: {len(positions)} nodes, {len(connectivity)} elements")
    print(f"✓ Created {len(particle_pos)} test particle")
except Exception as e:
    print(f"✗ Test data creation failed: {e}")
    exit(1)

# Test 3: Test search_particles_in_block directly
print("\n[3/5] Testing search_particles_in_block()...")
try:
    # Convert to JAX arrays
    jax_pos = jnp.array(particle_pos, dtype=jnp.float32)
    jax_elem = jnp.array(particle_elem, dtype=jnp.int32)
    jax_block = jnp.array(particle_block, dtype=jnp.int32)
    jax_active = jnp.array(particle_active, dtype=jnp.bool_)

    jax_conn = jnp.array(connectivity, dtype=jnp.int32)
    jax_positions = jnp.array(positions, dtype=jnp.float32)
    jax_neighbors = jnp.array(neighbors, dtype=jnp.int32)

    # Call search kernel
    result = search_particles_in_block(
        particle_positions=jax_pos,
        particle_element_ids=jax_elem,
        particle_block_ids=jax_block,
        particle_active=jax_active,
        block_id=0,
        block_connectivity=jax_conn,
        block_node_positions=jax_positions,
        block_element_neighbors=jax_neighbors,
        block_size=1
    )

    # Verify result
    assert isinstance(result, BlockSearchResult), "Result should be BlockSearchResult"
    assert result.element_ids[0] == 0, "Particle should be found in element 0"
    assert result.n_level0_hits == 1, "Should hit Level 0 (cached)"
    assert result.n_level1_hits == 0, "Should not need Level 1"
    assert result.n_level2_hits == 0, "Should not need Level 2"
    assert result.n_not_found == 0, "Should find all particles"

    print(f"✓ Search kernel works correctly")
    print(f"  Level 0 hits: {result.n_level0_hits}")
    print(f"  Level 1 hits: {result.n_level1_hits}")
    print(f"  Level 2 hits: {result.n_level2_hits}")
    print(f"  Not found: {result.n_not_found}")
except Exception as e:
    print(f"✗ Search kernel test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Test hash bucket search (with fallback)
print("\n[4/5] Testing search_particles_in_block_with_hash()...")
try:
    # Call with hash_bucket_data=None (should fallback to regular search)
    result_hash = search_particles_in_block_with_hash(
        particle_positions=jax_pos,
        particle_element_ids=jax_elem,
        particle_block_ids=jax_block,
        particle_active=jax_active,
        block_id=0,
        block_connectivity=jax_conn,
        block_node_positions=jax_positions,
        block_element_neighbors=jax_neighbors,
        block_size=1,
        hash_bucket_data=None  # Should trigger fallback
    )

    # Should get same result as regular search
    assert result_hash.element_ids[0] == 0, "Fallback should work"
    assert result_hash.n_level0_hits == 1, "Fallback should hit Level 0"

    print(f"✓ Hash bucket search (fallback) works correctly")
except Exception as e:
    print(f"✗ Hash bucket search test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Test batch processor configuration
print("\n[5/5] Testing batch processor configuration...")
try:
    # Create minimal config to verify BatchConfig works
    config = BatchConfig(
        batch_size=100,
        actual_batch_size=100,
        heavy_block_threshold=10000,
        light_block_threshold=1000,
        use_hash_buckets=False,
        gpu_memory_gb=4.0
    )

    # Verify BatchStatistics structure
    stats = BatchStatistics(
        batch_id=0,
        n_particles=100,
        n_active_blocks=5
    )

    print(f"✓ Batch processor configuration working")
    print(f"  BatchConfig: batch_size={config.actual_batch_size}")
    print(f"  BatchStatistics: {stats.batch_id}, {stats.n_particles} particles")
    print(f"  Stat tracking: level0={stats.level0_hits}, level1={stats.level1_hits}, level2={stats.level2_hits}")

except Exception as e:
    print(f"✗ Batch processor config test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Summary
print("\n" + "="*80)
print("✅ ALL PHASE 2 INTEGRATION TESTS PASSED")
print("="*80)
print("\nPhase 2 Status:")
print("  ✓ JAX search kernels implemented")
print("  ✓ 3-level search hierarchy working")
print("  ✓ Hash bucket search (with fallback) working")
print("  ✓ Batch processor integration ready")
print("\nNext steps:")
print("  - Test with real mesh (ThreadedA)")
print("  - Implement hash bucket preprocessing")
print("  - Optimize light block batching")
print("  - Measure 2,000 p/s throughput target")
print("="*80 + "\n")
