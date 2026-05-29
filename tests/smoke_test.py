#!/usr/bin/env python3
"""
JAXTrace Smoke Test

Quick import and basic functionality test to ensure the package is working.
This test should run fast and catch major import/API issues.
"""

import sys
import traceback
from pathlib import Path

# Add project root to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_core_imports():
    """Test that core JAXTrace modules import successfully."""
    print("Testing core imports...")

    try:
        import jaxtrace as jt
        print(f"✅ jaxtrace {jt.__version__}")

        # Test JAX availability
        print(f"✅ JAX available: {jt.JAX_AVAILABLE}")

        # Test core modules
        from jaxtrace.io import open_dataset
        from jaxtrace.fields import TimeSeriesField
        from jaxtrace.tracking import create_tracker, random_seeds
        from jaxtrace.density import DensityRunner, DensityRunnerConfig, DensityEstimator

        print("✅ Core modules imported successfully")
        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality with minimal data."""
    print("\nTesting basic functionality...")

    try:
        import numpy as np
        import jaxtrace as jt
        from jaxtrace.tracking import create_tracker

        # Create minimal synthetic field
        times = np.array([0.0, 1.0])
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        velocity_data = np.zeros((2, 4, 3), dtype=np.float32)  # (time, points, vector)
        velocity_data[:, :, 0] = 0.1  # uniform x-velocity

        field = jt.TimeSeriesField(
            data=velocity_data,
            times=times,
            positions=positions
        )
        print("✅ TimeSeriesField created")

        # Test particle seeding
        particles = jt.random_seeds(n=100, bounds=((0,0,0), (1,1,1)), rng_seed=42)
        assert particles.shape == (100, 3)
        print("✅ Particle seeding works")

        # Test tracker creation (don't run simulation to keep it fast)
        tracker = create_tracker(
            integrator_name="rk4",
            field=field,
            boundary_condition=jt.periodic_boundary(((0,0,0), (1,1,1))),
            max_memory_gb=1.0
        )
        print("✅ Tracker created successfully")

        # Test density estimators
        kde = jt.KDEEstimator(positions=particles, bandwidth_rule="scott")
        print("✅ KDE estimator created")

        sph = jt.SPHDensityEstimator(positions=particles, smoothing_length=0.1)
        print("✅ SPH estimator created")

        return True

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_system_info():
    """Test system information functions."""
    print("\nTesting system information...")

    try:
        import jaxtrace as jt

        # Test system requirements check
        jt.check_system_requirements()
        print("✅ System requirements check completed")

        return True

    except Exception as e:
        print(f"❌ System info test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("JAXTrace Smoke Test")
    print("=" * 50)

    tests = [
        ("Core Imports", test_core_imports),
        ("Basic Functionality", test_basic_functionality),
        ("System Info", test_system_info),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        print(f"\n🧪 Running: {name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {name}: PASSED")
            else:
                print(f"❌ {name}: FAILED")
        except Exception as e:
            print(f"❌ {name}: ERROR - {e}")

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All smoke tests PASSED!")
        return 0
    else:
        print("💥 Some smoke tests FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)