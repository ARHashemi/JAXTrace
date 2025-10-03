#!/usr/bin/env python3
"""
Memory Optimization Test Script

Quick test to verify memory optimization features work correctly.
"""

import os
import sys
import traceback

# Set memory optimization environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

def test_memory_tracker():
    """Test memory tracking functionality."""
    print("🧪 Testing GPU Memory Tracker...")

    try:
        from jaxtrace.utils import (
            initialize_memory_tracking,
            get_memory_tracker,
            track_memory,
            track_variable_memory,
            track_operation_memory,
            configure_memory_optimization
        )

        # Initialize memory tracking
        tracker = initialize_memory_tracking(
            log_file="test_memory_tracking.json",
            enable_detailed_tracking=True,
            track_stack_traces=False
        )

        # Configure memory optimization
        configure_memory_optimization()

        # Set baseline
        tracker.set_baseline()

        # Test tracking operations
        with track_operation_memory("test_operation"):
            import numpy as np
            test_array = np.random.rand(1000, 3).astype(np.float32)
            track_variable_memory("test_array", test_array)
            track_memory("test_checkpoint")

        # Get memory summary
        summary = tracker.get_memory_summary()
        print(f"   ✅ Memory tracking works: {summary['total_snapshots']} snapshots recorded")

        return True

    except Exception as e:
        print(f"   ❌ Memory tracker test failed: {e}")
        traceback.print_exc()
        return False


def test_memory_optimized_loader():
    """Test memory-optimized data loader."""
    print("🧪 Testing Memory-Optimized Loader...")

    try:
        from jaxtrace.io.memory_optimized_loader import (
            MemoryOptimizedConfig,
            MemoryOptimizedLoader,
            estimate_memory_usage
        )

        # Test configuration
        config = MemoryOptimizedConfig(
            essential_fields_only=True,
            max_memory_per_timestep_mb=50.0,
            dtype="float32"
        )

        # Test loader initialization
        loader = MemoryOptimizedLoader(config)

        print(f"   ✅ Memory-optimized loader initialized successfully")
        print(f"   Essential fields: {loader.essential_fields}")
        print(f"   Configuration: {config}")

        return True

    except Exception as e:
        print(f"   ❌ Memory-optimized loader test failed: {e}")
        traceback.print_exc()
        return False


def test_jax_memory_configuration():
    """Test JAX memory configuration."""
    print("🧪 Testing JAX Memory Configuration...")

    try:
        # Check environment variables
        preallocate = os.getenv("XLA_PYTHON_CLIENT_PREALLOCATE")
        mem_fraction = os.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION")

        print(f"   XLA_PYTHON_CLIENT_PREALLOCATE: {preallocate}")
        print(f"   XLA_PYTHON_CLIENT_MEM_FRACTION: {mem_fraction}")

        if preallocate == "false":
            print("   ✅ JAX memory preallocation disabled")
        else:
            print("   ❌ JAX memory preallocation not disabled")
            return False

        # Try importing JAX to verify configuration
        try:
            import jax
            import jax.numpy as jnp

            # Test small array creation
            test_array = jnp.array([1, 2, 3], dtype=jnp.float32)
            print(f"   ✅ JAX import successful, test array: {test_array}")

        except ImportError:
            print("   ⚠️  JAX not available, but configuration is set correctly")

        return True

    except Exception as e:
        print(f"   ❌ JAX memory configuration test failed: {e}")
        return False


def test_synthetic_field_creation():
    """Test memory-efficient synthetic field creation."""
    print("🧪 Testing Synthetic Field Creation...")

    try:
        import numpy as np
        from jaxtrace.fields import TimeSeriesField
        from jaxtrace.utils import track_operation_memory, track_variable_memory

        with track_operation_memory("synthetic_field_test"):
            # Create small synthetic field
            x = np.linspace(-1, 1, 5, dtype=np.float32)
            y = np.linspace(-1, 1, 5, dtype=np.float32)
            z = np.array([0.0], dtype=np.float32)

            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            track_variable_memory("test_positions", positions)

            # Create simple time series
            times = np.array([0.0, 1.0], dtype=np.float32)
            velocities = np.random.rand(2, len(positions), 3).astype(np.float32)
            track_variable_memory("test_velocities", velocities)

            # Create field
            field = TimeSeriesField(
                data=velocities,
                times=times,
                positions=positions,
                interpolation="linear",
                extrapolation="constant"
            )
            track_variable_memory("test_field", field)

            # Test field functionality
            bounds_min, bounds_max = field.get_spatial_bounds()
            print(f"   ✅ Synthetic field created: bounds {bounds_min} to {bounds_max}")

        return True

    except Exception as e:
        print(f"   ❌ Synthetic field test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all memory optimization tests."""
    print("="*60)
    print("MEMORY OPTIMIZATION TEST SUITE")
    print("="*60)

    tests = [
        ("Memory Tracker", test_memory_tracker),
        ("Memory-Optimized Loader", test_memory_optimized_loader),
        ("JAX Memory Configuration", test_jax_memory_configuration),
        ("Synthetic Field Creation", test_synthetic_field_creation),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)

        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All memory optimization tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())