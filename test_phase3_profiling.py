#!/usr/bin/env python3
"""
Phase 3 Hash Octree Profiling Test

This script runs comprehensive profiling tests comparing:
1. CPU octree search (Phase 1/2 baseline)
2. GPU hash octree search (Phase 3)

Profiling metrics:
- Wall-clock time
- GPU utilization
- CPU utilization
- RAM usage
- GPU memory usage
- Speedup factor

Usage:
    python test_phase3_profiling.py
"""

import os
import sys
import time
import psutil
import numpy as np
from pathlib import Path

# Enable JAX 64-bit mode for hash octree
import jax
jax.config.update("jax_enable_x64", True)

# GPU optimization
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"


def get_gpu_info():
    """Get GPU information using nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpu_info.append({
                        'name': parts[0],
                        'memory_total_mb': float(parts[1]),
                        'memory_used_mb': float(parts[2]),
                        'utilization_pct': float(parts[3])
                    })
            return gpu_info
        return None
    except Exception as e:
        print(f"Warning: Could not get GPU info: {e}")
        return None


def profile_memory_and_time(func, *args, **kwargs):
    """Profile a function's execution time and memory usage."""
    import gc

    # Force garbage collection before profiling
    gc.collect()

    # Get baseline memory
    process = psutil.Process()
    baseline_ram_mb = process.memory_info().rss / 1024 / 1024
    baseline_gpu = get_gpu_info()
    baseline_gpu_mb = baseline_gpu[0]['memory_used_mb'] if baseline_gpu else 0

    # Record start time
    start_time = time.time()

    # Execute function
    result = func(*args, **kwargs)

    # Record end time
    end_time = time.time()
    elapsed = end_time - start_time

    # Get peak memory after execution
    peak_ram_mb = process.memory_info().rss / 1024 / 1024
    peak_gpu = get_gpu_info()
    peak_gpu_mb = peak_gpu[0]['memory_used_mb'] if peak_gpu else 0

    # Get GPU utilization
    gpu_util = peak_gpu[0]['utilization_pct'] if peak_gpu else 0

    # Get CPU utilization (average over last second)
    cpu_util = psutil.cpu_percent(interval=0.1)

    return {
        'result': result,
        'time_sec': elapsed,
        'ram_delta_mb': peak_ram_mb - baseline_ram_mb,
        'ram_peak_mb': peak_ram_mb,
        'gpu_delta_mb': peak_gpu_mb - baseline_gpu_mb,
        'gpu_peak_mb': peak_gpu_mb,
        'gpu_util_pct': gpu_util,
        'cpu_util_pct': cpu_util
    }


def run_profiling_test(use_hash_octree=False, n_particles=1000, n_timesteps=500):
    """Run a single profiling test with specified configuration."""

    print("\n" + "="*80)
    print(f"PROFILING TEST: {'GPU Hash Octree' if use_hash_octree else 'CPU Octree Search'}")
    print("="*80)
    print(f"Particles: {n_particles}")
    print(f"Timesteps: {n_timesteps}")
    print("-"*80)

    # Import here to avoid early JAX initialization
    from example_workflow import main

    # Configuration for profiling
    config = {
        # Data
        'data_pattern': "/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu",
        'max_timesteps_to_load': 40,

        # Octree
        'n_coarse_levels': 6,
        'max_octree_depth': 12,
        'max_elements_per_leaf': 32,
        'revolution_timesteps': 40,

        # Phase 3: Enable/disable hash octree
        'use_hash_octree': use_hash_octree,

        # Particles (small region for faster testing)
        'particle_concentrations': {'x': 20, 'y': 20, 'z': 10},
        'particle_distribution': 'uniform',
        'particle_bounds': [
            np.array([-0.026, -0.023, -0.01]),
            np.array([-0.01, -0.005, 0.0])
        ],

        # Tracking (reduced for profiling)
        'n_timesteps': n_timesteps,
        'dt': 0.0025,
        'time_span': (120, 125),  # Short time window
        'batch_size': min(n_particles, 5000),
        'integrator': 'rk4',

        # Boundaries
        'flow_axis': 'x',
        'boundary_inlet': 'reflective',
        'boundary_outlet': 'reflective',

        # Visualization (disabled for profiling)
        'perform_density_analysis': False,

        # GPU
        'device': 'gpu',
        'memory_limit_gb': 3.0,
    }

    print("Starting profiling...")
    print("-"*80)

    # Profile the entire workflow
    def run_workflow():
        try:
            main(config=config)
            return True
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()
            return False

    profile_results = profile_memory_and_time(run_workflow)

    print("\n" + "-"*80)
    print("PROFILING RESULTS")
    print("-"*80)
    print(f"Execution time:   {profile_results['time_sec']:.2f} seconds")
    print(f"RAM usage:        {profile_results['ram_delta_mb']:+.1f} MB (peak: {profile_results['ram_peak_mb']:.1f} MB)")
    print(f"GPU memory:       {profile_results['gpu_delta_mb']:+.1f} MB (peak: {profile_results['gpu_peak_mb']:.1f} MB)")
    print(f"GPU utilization:  {profile_results['gpu_util_pct']:.1f}%")
    print(f"CPU utilization:  {profile_results['cpu_util_pct']:.1f}%")
    print(f"Success:          {profile_results['result']}")
    print("-"*80)

    return profile_results


def compare_implementations():
    """Compare CPU octree vs GPU hash octree performance."""

    print("\n" + "="*80)
    print("PHASE 3: GPU HASH OCTREE vs CPU OCTREE COMPARISON")
    print("="*80)

    # Check GPU availability
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"\nGPU: {gpu_info[0]['name']}")
        print(f"GPU Memory: {gpu_info[0]['memory_total_mb']:.0f} MB")
        print(f"Current GPU Usage: {gpu_info[0]['memory_used_mb']:.0f} MB ({gpu_info[0]['utilization_pct']:.1f}% util)")
    else:
        print("\nWarning: Could not detect GPU")

    # System info
    print(f"\nCPU: {psutil.cpu_count()} cores")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

    # Test configuration
    n_particles = 1000
    n_timesteps = 200

    print(f"\nTest parameters:")
    print(f"  Particles: {n_particles}")
    print(f"  Timesteps: {n_timesteps}")

    # Run tests
    print("\n" + "="*80)
    print("TEST 1/2: CPU Octree Search (Baseline)")
    print("="*80)

    try:
        cpu_results = run_profiling_test(
            use_hash_octree=False,
            n_particles=n_particles,
            n_timesteps=n_timesteps
        )
        cpu_success = cpu_results['result']
    except Exception as e:
        print(f"\n❌ CPU test failed: {e}")
        import traceback
        traceback.print_exc()
        cpu_success = False
        cpu_results = None

    print("\n" + "="*80)
    print("TEST 2/2: GPU Hash Octree (Phase 3)")
    print("="*80)

    try:
        gpu_results = run_profiling_test(
            use_hash_octree=True,
            n_particles=n_particles,
            n_timesteps=n_timesteps
        )
        gpu_success = gpu_results['result']
    except Exception as e:
        print(f"\n❌ GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        gpu_success = False
        gpu_results = None

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    if cpu_success and gpu_success and cpu_results and gpu_results:
        speedup = cpu_results['time_sec'] / gpu_results['time_sec']
        ram_diff = gpu_results['ram_delta_mb'] - cpu_results['ram_delta_mb']
        gpu_mem_diff = gpu_results['gpu_delta_mb'] - cpu_results['gpu_delta_mb']

        print("\n| Metric                 | CPU Octree  | GPU Hash    | Improvement |")
        print("|------------------------|-------------|-------------|-------------|")
        print(f"| Execution Time         | {cpu_results['time_sec']:7.2f} sec | {gpu_results['time_sec']:7.2f} sec | {speedup:6.2f}x     |")
        print(f"| RAM Usage              | {cpu_results['ram_delta_mb']:7.1f} MB  | {gpu_results['ram_delta_mb']:7.1f} MB  | {ram_diff:+7.1f} MB  |")
        print(f"| GPU Memory             | {cpu_results['gpu_delta_mb']:7.1f} MB  | {gpu_results['gpu_delta_mb']:7.1f} MB  | {gpu_mem_diff:+7.1f} MB  |")
        print(f"| GPU Utilization        | {cpu_results['gpu_util_pct']:7.1f} %   | {gpu_results['gpu_util_pct']:7.1f} %   | {gpu_results['gpu_util_pct'] - cpu_results['gpu_util_pct']:+7.1f} %   |")
        print(f"| CPU Utilization        | {cpu_results['cpu_util_pct']:7.1f} %   | {gpu_results['cpu_util_pct']:7.1f} %   | {gpu_results['cpu_util_pct'] - cpu_results['cpu_util_pct']:+7.1f} %   |")

        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)

        if speedup > 1.0:
            print(f"✅ GPU hash octree is {speedup:.2f}x FASTER than CPU octree")
        else:
            print(f"⚠️  GPU hash octree is {1/speedup:.2f}x SLOWER than CPU octree")

        if ram_diff < 0:
            print(f"✅ GPU hash octree uses {abs(ram_diff):.1f} MB LESS RAM")
        else:
            print(f"⚠️  GPU hash octree uses {ram_diff:.1f} MB MORE RAM")

        if gpu_results['gpu_util_pct'] > cpu_results['gpu_util_pct']:
            print(f"✅ GPU utilization improved by {gpu_results['gpu_util_pct'] - cpu_results['gpu_util_pct']:.1f}%")

        print("\n" + "="*80)

        # Success determination
        if speedup >= 1.5 and gpu_results['gpu_util_pct'] > 50:
            print("🎉 PHASE 3 HASH OCTREE PERFORMANCE: EXCELLENT")
            print(f"   - {speedup:.1f}x speedup achieved")
            print(f"   - {gpu_results['gpu_util_pct']:.0f}% GPU utilization")
            return True
        elif speedup >= 1.0:
            print("✅ PHASE 3 HASH OCTREE PERFORMANCE: GOOD")
            print(f"   - {speedup:.1f}x speedup achieved")
            print(f"   - Consider optimizations for higher GPU utilization")
            return True
        else:
            print("⚠️  PHASE 3 HASH OCTREE PERFORMANCE: NEEDS OPTIMIZATION")
            print(f"   - Only {speedup:.2f}x speedup (slower than CPU)")
            print(f"   - Possible bottlenecks: Morton encoding, element testing")
            return False

    elif cpu_success and not gpu_success:
        print("\n❌ GPU hash octree test failed, but CPU octree works")
        print("   Check Phase 3 implementation for errors")
        return False

    elif not cpu_success and gpu_success:
        print("\n✅ GPU hash octree works, but CPU octree baseline failed")
        print("   This is unexpected - check test configuration")
        return True

    else:
        print("\n❌ Both tests failed - check system configuration")
        return False


def main():
    """Main profiling entry point."""

    print("="*80)
    print("PHASE 3: GPU-NATIVE HASH OCTREE PROFILING TEST")
    print("="*80)
    print("\nThis script compares CPU octree search vs GPU hash octree lookup")
    print("with comprehensive profiling of time, memory, and GPU utilization.")
    print("\n" + "="*80)

    # Check dependencies
    try:
        import jax
        import numpy as np
        import psutil
        print("\n✅ Dependencies: JAX, NumPy, psutil")
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("Install with: pip install jax numpy psutil")
        return False

    # Check JAX device
    devices = jax.devices()
    print(f"✅ JAX devices: {devices}")

    # Run comparison
    success = compare_implementations()

    print("\n" + "="*80)
    if success:
        print("✅ PROFILING TEST COMPLETED SUCCESSFULLY")
    else:
        print("⚠️  PROFILING TEST COMPLETED WITH ISSUES")
    print("="*80)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
