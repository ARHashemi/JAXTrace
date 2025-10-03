#!/usr/bin/env python3
"""Test VTK loading with the actual data files."""

import os

# Set memory optimization
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def test_vtk_loading():
    """Test VTK loading with different approaches."""

    try:
        from jaxtrace.io import open_dataset

        # Test with exact file pattern
        data_path = "/home/arhashemi/Workspace/welding/Cases/004_caseCoarse.gid/post/0eule/"

        print(f"Testing VTK loading from: {data_path}")

        # Check files exist
        import glob
        pvtu_files = glob.glob(os.path.join(data_path, "*.pvtu"))
        print(f"Found {len(pvtu_files)} PVTU files")

        if pvtu_files:
            print("First few files:")
            for f in sorted(pvtu_files)[:5]:
                print(f"  {os.path.basename(f)}")

        # Try loading with different patterns
        patterns_to_try = [
            os.path.join(data_path, "004_caseCoarse_*.pvtu"),
            os.path.join(data_path, "*.pvtu"),
            data_path,
        ]

        for pattern in patterns_to_try:
            print(f"\n🔍 Trying pattern: {pattern}")
            try:
                dataset = open_dataset(pattern, max_time_steps=5)
                print(f"   ✅ Success! Dataset: {type(dataset)}")

                # Check if it has time series functionality
                if hasattr(dataset, 'get_time_values'):
                    times = dataset.get_time_values()
                    if times is not None:
                        print(f"   Found {len(times)} time values: {times[:5]}")
                    else:
                        print("   No time values found")

                if hasattr(dataset, 'load_time_series'):
                    print("   Has load_time_series method")

                return dataset

            except Exception as e:
                print(f"   ❌ Failed: {e}")

        return None

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_memory_optimized_loading():
    """Test memory-optimized loading approach."""

    try:
        from jaxtrace.io.memory_optimized_loader import estimate_memory_usage, load_optimized_dataset

        data_path = "/home/arhashemi/Workspace/welding/Cases/004_caseCoarse.gid/post/0eule/"

        print(f"\n🧪 Testing memory-optimized loading")

        # Test patterns
        patterns_to_try = [
            os.path.join(data_path, "*.pvtu"),
            data_path,
        ]

        for pattern in patterns_to_try:
            print(f"\n🔍 Trying memory-optimized pattern: {pattern}")
            try:
                # First estimate memory usage
                estimate = estimate_memory_usage(pattern, num_timesteps=5)
                print(f"   Memory estimate: {estimate}")

                if "error" not in estimate:
                    # Try to load a small dataset
                    field = load_optimized_dataset(
                        data_pattern=pattern,
                        max_memory_per_timestep_mb=200.0,
                        max_time_steps=3,
                        dtype="float32"
                    )
                    print(f"   ✅ Memory-optimized loading success!")
                    print(f"   Field: {field}")
                    return field

            except Exception as e:
                print(f"   ❌ Failed: {e}")

        return None

    except Exception as e:
        print(f"❌ Memory-optimized test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("="*60)
    print("VTK LOADING TEST")
    print("="*60)

    # Test regular loading
    dataset = test_vtk_loading()

    # Test memory-optimized loading
    field = test_memory_optimized_loading()

    print(f"\n" + "="*60)
    print("RESULTS:")
    print(f"Regular loading: {'✅ SUCCESS' if dataset else '❌ FAILED'}")
    print(f"Memory-optimized loading: {'✅ SUCCESS' if field else '❌ FAILED'}")
    print("="*60)