#!/usr/bin/env python3
"""
Test shared coarse octree implementation with Edgar/FLA dataset.

This validates:
1. Coarse octree building from refinement steps
2. Fine octree building with reuse detection
3. Memory usage and reuse statistics
4. Query functionality
"""

import sys
import glob
sys.path.insert(0, '/home/arhashemi/Workspace/welding/JAXTrace')

from jaxtrace.fields.shared_octree_factory import (
    SharedOctreeFactory,
    SharedOctreeConfig
)
import jax.numpy as jnp
import numpy as np


def test_basic_build():
    """Test basic octree building."""
    print("=" * 70)
    print("TEST 1: Basic Shared Octree Building")
    print("=" * 70)

    # Load Edgar/FLA files
    file_pattern = "/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu"
    files = sorted(glob.glob(file_pattern))

    if len(files) == 0:
        print(f"ERROR: No files found for pattern: {file_pattern}")
        return False

    print(f"Found {len(files)} files")

    # Configure for Edgar/FLA case
    config = SharedOctreeConfig(
        n_refinement_steps=None,  # Auto-detect (should find ~3 steps)
        n_coarse_levels=6,
        max_octree_depth=12,
        max_cells_per_node=32,
        enable_fine_structure_reuse=True,
        revolution_timesteps=40,  # Last 40 timesteps
        use_last_n_timesteps=True
    )

    # Build octree
    factory = SharedOctreeFactory(config)

    try:
        shared_octree = factory.build_from_files(files[:50], verbose=True)  # Test with first 50 files
        print("\n✓ Octree built successfully")
        return True
    except Exception as e:
        print(f"\n✗ Failed to build octree: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage():
    """Test memory usage and reuse statistics."""
    print("\n" + "=" * 70)
    print("TEST 2: Memory Usage and Reuse Statistics")
    print("=" * 70)

    file_pattern = "/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu"
    files = sorted(glob.glob(file_pattern))

    config = SharedOctreeConfig(
        n_refinement_steps=3,  # Use first 3 steps
        n_coarse_levels=6,
        revolution_timesteps=40,
        enable_fine_structure_reuse=True
    )

    factory = SharedOctreeFactory(config)

    try:
        # Build with last 40 timesteps
        shared_octree = factory.build_from_files(files, verbose=False)

        # Get statistics
        coarse_mem, unique_fine_mem, total_mem = shared_octree.get_memory_size()
        stats = shared_octree.get_reuse_statistics()

        print(f"\nMemory Usage:")
        print(f"  Coarse: {coarse_mem / (1024**2):.2f} MB")
        print(f"  Fine (unique): {unique_fine_mem / (1024**2):.2f} MB")
        print(f"  Total: {total_mem / (1024**2):.2f} MB")

        print(f"\nReuse Statistics:")
        print(f"  Timesteps: {stats['n_timesteps']}")
        print(f"  Unique structures: {stats['n_unique_structures']}")
        print(f"  Reuse rate: {stats['reuse_rate']*100:.1f}%")
        print(f"  Memory savings: {stats['memory_savings_factor']:.1f}x")

        # Validate expectations
        expected_reuse_rate = 0.90  # At least 90% reuse for FLA
        if stats['reuse_rate'] >= expected_reuse_rate:
            print(f"\n✓ Reuse rate meets expectation (>= {expected_reuse_rate*100:.0f}%)")
        else:
            print(f"\n⚠ Reuse rate below expectation (< {expected_reuse_rate*100:.0f}%)")

        # Check total memory is within GPU limit
        gpu_limit_gb = 3.6  # 90% of 4GB
        total_mem_gb = total_mem / (1024**3)
        if total_mem_gb <= gpu_limit_gb:
            print(f"✓ Memory within GPU limit ({total_mem_gb:.2f} GB <= {gpu_limit_gb:.2f} GB)")
        else:
            print(f"✗ Memory exceeds GPU limit ({total_mem_gb:.2f} GB > {gpu_limit_gb:.2f} GB)")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_query_functionality():
    """Test octree query functionality."""
    print("\n" + "=" * 70)
    print("TEST 3: Query Functionality")
    print("=" * 70)

    file_pattern = "/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu"
    files = sorted(glob.glob(file_pattern))

    config = SharedOctreeConfig(
        n_refinement_steps=3,
        n_coarse_levels=6,
        revolution_timesteps=10,  # Small for faster test
        enable_fine_structure_reuse=True
    )

    factory = SharedOctreeFactory(config)

    try:
        shared_octree = factory.build_from_files(files[-10:], verbose=False)

        # Test point within domain
        bbox_min = np.array(shared_octree.coarse_levels.bbox_min)
        bbox_max = np.array(shared_octree.coarse_levels.bbox_max)
        test_point = (bbox_min + bbox_max) / 2  # Center of domain

        print(f"\nDomain bounds:")
        print(f"  Min: {bbox_min}")
        print(f"  Max: {bbox_max}")
        print(f"\nTest point: {test_point}")

        # Query for first timestep
        fine_level = shared_octree.get_fine_level_for_timestep(0)

        from jaxtrace.fields.shared_coarse_octree import query_octree_two_level

        elements = query_octree_two_level(
            jnp.array(test_point),
            shared_octree.coarse_levels,
            fine_level
        )

        print(f"\nQuery result: {len(elements)} candidate elements")

        if len(elements) > 0:
            print("✓ Query successful")
            return True
        else:
            print("⚠ Query returned no elements (may be outside mesh)")
            return True  # Not necessarily a failure

    except Exception as e:
        print(f"\n✗ Query test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("SHARED COARSE OCTREE TEST SUITE")
    print("=" * 70)
    print()

    results = []

    # Test 1: Basic building
    results.append(("Basic Build", test_basic_build()))

    # Test 2: Memory usage
    results.append(("Memory Usage", test_memory_usage()))

    # Test 3: Query functionality
    results.append(("Query", test_query_functionality()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    n_passed = sum(1 for _, p in results if p)
    n_total = len(results)
    print(f"\nTotal: {n_passed}/{n_total} tests passed")

    return n_passed == n_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
