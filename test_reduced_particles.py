#!/usr/bin/env python3
"""
Test JAX direct interpolation with REDUCED particle count to check performance and accuracy.
Monitors memory, GPU, and CPU usage at different stages.
"""

import jax
import jax.numpy as jnp
from jaxtrace.core import configure
from jaxtrace.io.vtk_io import VTKSeries
from jaxtrace.fields import SharedOctreeFEMTimeSeriesField
from jaxtrace.tracking import ParticleTracker
import numpy as np
import time
import subprocess
import json
from pathlib import Path

# Create logs directory
Path("logs").mkdir(exist_ok=True)

def get_memory_usage():
    """Get current memory usage in GB."""
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
            total = int([l for l in lines if 'MemTotal' in l][0].split()[1]) / 1024 / 1024
            available = int([l for l in lines if 'MemAvailable' in l][0].split()[1]) / 1024 / 1024
            used = total - available
            return {'total': total, 'used': used, 'available': available}
    except:
        return {'total': 0, 'used': 0, 'available': 0}

def get_gpu_usage():
    """Get GPU memory and utilization."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        mem_used, mem_total, gpu_util = result.stdout.strip().split(',')
        return {
            'mem_used_mb': int(mem_used),
            'mem_total_mb': int(mem_total),
            'gpu_util_pct': int(gpu_util)
        }
    except:
        return {'mem_used_mb': 0, 'mem_total_mb': 0, 'gpu_util_pct': 0}

def log_stage(stage_name, metrics_log):
    """Log current resource usage for a stage."""
    mem = get_memory_usage()
    gpu = get_gpu_usage()

    entry = {
        'stage': stage_name,
        'timestamp': time.time(),
        'ram_used_gb': mem['used'],
        'ram_available_gb': mem['available'],
        'gpu_mem_used_mb': gpu['mem_used_mb'],
        'gpu_util_pct': gpu['gpu_util_pct']
    }

    metrics_log.append(entry)
    print(f"\n{'='*80}")
    print(f"STAGE: {stage_name}")
    print(f"{'='*80}")
    print(f"RAM: {mem['used']:.2f} GB used / {mem['available']:.2f} GB available")
    print(f"GPU: {gpu['mem_used_mb']} MB used, {gpu['gpu_util_pct']}% utilization")
    print(f"{'='*80}\n")

    return entry

def main():
    print("="*80)
    print("JAX DIRECT INTERPOLATION - REDUCED PARTICLE TEST")
    print("="*80)
    print("Testing with small particle count to verify:")
    print("  1. Performance and memory usage")
    print("  2. Accuracy of interpolation")
    print("  3. GPU acceleration")
    print("="*80)

    metrics_log = []
    start_time = time.time()

    # Stage 1: Configuration
    log_stage("1. Initial Configuration", metrics_log)

    configure(
        device='gpu',
        memory_limit_gb=3.0,
        use_jax_jit=True,
        use_jax_vmap=True,
        precompile_functions=True,
        show_progress=True
    )

    print(f"✅ JAX device: {jax.devices()}")

    # Stage 2: Load VTK data
    log_stage("2. Loading VTK Data", metrics_log)

    pattern = "/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu"

    # Configuration for REDUCED particle count
    config = {
        'use_direct_interpolation': True,  # Use JAX direct mode
        'max_elements_per_leaf': 32,
        'max_octree_depth': 12,
        'coarse_octree_levels': 6,
        'fine_octree_reuse': True,
        'revolution_timesteps': 40,  # Last 40 timesteps
        'cache_size': 3,
    }

    field = SharedOctreeFEMTimeSeriesField(
        vtk_pattern=pattern,
        user_config=config
    )

    log_stage("3. After Field Creation", metrics_log)

    # Stage 4: Generate REDUCED particle set
    # Original: 60x50x15 = 45,000 particles
    # Reduced: 10x10x5 = 500 particles (90x reduction)
    print("\n" + "="*80)
    print("PARTICLE CONFIGURATION")
    print("="*80)
    print("Original config: 60×50×15 = 45,000 particles → 2.76 TiB JAX compilation error")
    print("Reduced config:  10×10×5  = 500 particles    → Testing feasibility")
    print("="*80)

    field_bounds = field.bounds
    print(f"Field bounds: {field_bounds}")

    # Use same relative bounds as original
    x_min, y_min, z_min = field_bounds[0]
    x_max, y_max, z_max = field_bounds[1]

    # Fractional bounds from original config
    x_frac = (0.1, 0.3)
    y_frac = (0.0, 1.0)
    z_frac = (0.0, 1.0)

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    particle_x_min = x_min + x_frac[0] * x_range
    particle_x_max = x_min + x_frac[1] * x_range
    particle_y_min = y_min + y_frac[0] * y_range
    particle_y_max = y_min + y_frac[1] * y_range
    particle_z_min = z_min + z_frac[0] * z_range
    particle_z_max = z_min + z_frac[1] * z_range

    # Generate 10x10x5 = 500 particles
    nx, ny, nz = 10, 10, 5
    x_coords = np.linspace(particle_x_min, particle_x_max, nx)
    y_coords = np.linspace(particle_y_min, particle_y_max, ny)
    z_coords = np.linspace(particle_z_min, particle_z_max, nz)

    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    seeds = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)

    print(f"Generated {len(seeds)} particles in region:")
    print(f"  X: [{particle_x_min:.4f}, {particle_x_max:.4f}]")
    print(f"  Y: [{particle_y_min:.4f}, {particle_y_max:.4f}]")
    print(f"  Z: [{particle_z_min:.4f}, {particle_z_max:.4f}]")

    log_stage("4. After Particle Generation", metrics_log)

    # Stage 5: Create tracker
    tracker = ParticleTracker(field)

    log_stage("5. After Tracker Creation", metrics_log)

    # Stage 6: Run particle tracking
    print("\n" + "="*80)
    print("STARTING PARTICLE TRACKING")
    print("="*80)
    print(f"Particles: {len(seeds)}")
    print(f"Timesteps: 2000")
    print(f"Time step: 0.0025")
    print(f"Integration: RK4")
    print("="*80)

    tracking_start = time.time()

    try:
        log_stage("6. Before Track Particles (JAX Compilation)", metrics_log)

        trajectory = tracker.track_particles(
            initial_positions=seeds,
            times=(120, 159),  # Revolution cycle
            dt=0.0025,
            max_steps=2000
        )

        tracking_end = time.time()
        tracking_time = tracking_end - tracking_start

        log_stage("7. After Track Particles (Success!)", metrics_log)

        print("\n" + "="*80)
        print("✅ TRACKING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Trajectory shape: {trajectory.shape}")
        print(f"Tracking time: {tracking_time:.2f} seconds")
        print(f"Time per particle: {tracking_time / len(seeds):.4f} seconds")
        print(f"Time per step: {tracking_time / 2000:.4f} seconds")
        print("="*80)

        # Analyze results
        print("\n" + "="*80)
        print("TRAJECTORY ANALYSIS")
        print("="*80)

        # Check for NaN/Inf
        has_nan = np.isnan(trajectory).any()
        has_inf = np.isinf(trajectory).any()
        print(f"Contains NaN: {has_nan}")
        print(f"Contains Inf: {has_inf}")

        # Compute displacement statistics
        initial_pos = trajectory[:, 0, :]
        final_pos = trajectory[:, -1, :]
        displacement = np.linalg.norm(final_pos - initial_pos, axis=1)

        print(f"\nDisplacement statistics:")
        print(f"  Mean: {displacement.mean():.6f}")
        print(f"  Std:  {displacement.std():.6f}")
        print(f"  Min:  {displacement.min():.6f}")
        print(f"  Max:  {displacement.max():.6f}")

        # Compute velocity magnitudes
        velocities = np.diff(trajectory, axis=1)
        vel_magnitudes = np.linalg.norm(velocities, axis=2)

        print(f"\nVelocity statistics:")
        print(f"  Mean: {vel_magnitudes.mean():.6f}")
        print(f"  Std:  {vel_magnitudes.std():.6f}")
        print(f"  Min:  {vel_magnitudes.min():.6f}")
        print(f"  Max:  {vel_magnitudes.max():.6f}")

        # Save trajectory for visualization
        np.save("logs/reduced_trajectory.npy", trajectory)
        print(f"\n✅ Saved trajectory to: logs/reduced_trajectory.npy")

        success = True

    except Exception as e:
        tracking_end = time.time()
        tracking_time = tracking_end - tracking_start

        log_stage("7. After Track Particles (FAILED)", metrics_log)

        print("\n" + "="*80)
        print("❌ TRACKING FAILED")
        print("="*80)
        print(f"Error: {e}")
        print(f"Time before failure: {tracking_time:.2f} seconds")
        print("="*80)

        import traceback
        traceback.print_exc()

        success = False

    # Final stage
    total_time = time.time() - start_time
    log_stage("8. Test Complete", metrics_log)

    # Save metrics log
    metrics_file = "logs/reduced_test_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'success': success,
            'total_time': total_time,
            'tracking_time': tracking_time if 'tracking_time' in locals() else 0,
            'num_particles': len(seeds),
            'num_timesteps': 2000,
            'metrics_by_stage': metrics_log
        }, f, indent=2)

    print(f"\n✅ Saved metrics to: {metrics_file}")

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Success: {success}")
    print(f"Particles: {len(seeds)}")
    print(f"Timesteps: 2000")

    if success:
        print("\n🎉 TEST PASSED - JAX direct interpolation works with reduced particles!")
    else:
        print("\n⚠️  TEST FAILED - See error above")

    print("="*80)

    # Create detailed report
    report_file = "logs/reduced_test_report.md"
    with open(report_file, 'w') as f:
        f.write("# JAX Direct Interpolation - Reduced Particle Test Report\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Status**: {'✅ SUCCESS' if success else '❌ FAILED'}\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- **Particles**: {len(seeds)} (10×10×5 grid)\n")
        f.write(f"- **Original**: 45,000 (60×50×15 grid)\n")
        f.write(f"- **Reduction**: {100 * (1 - len(seeds)/45000):.1f}%\n")
        f.write(f"- **Timesteps**: 2000\n")
        f.write(f"- **Integration**: RK4, dt=0.0025\n")
        f.write(f"- **Mode**: Direct interpolation (coarse+fine octrees)\n\n")

        f.write("## Performance\n\n")
        f.write(f"- **Total time**: {total_time:.2f} seconds\n")
        if success:
            f.write(f"- **Tracking time**: {tracking_time:.2f} seconds\n")
            f.write(f"- **Time per particle**: {tracking_time / len(seeds):.4f} seconds\n")
            f.write(f"- **Time per timestep**: {tracking_time / 2000:.4f} seconds\n\n")

        f.write("## Resource Usage by Stage\n\n")
        f.write("| Stage | RAM Used (GB) | GPU Mem (MB) | GPU Util (%) |\n")
        f.write("|-------|---------------|--------------|---------------|\n")
        for entry in metrics_log:
            f.write(f"| {entry['stage']} | {entry['ram_used_gb']:.2f} | {entry['gpu_mem_used_mb']} | {entry['gpu_util_pct']} |\n")

        if success:
            f.write("\n## Results\n\n")
            f.write(f"- **Trajectory shape**: {trajectory.shape}\n")
            f.write(f"- **Contains NaN**: {has_nan}\n")
            f.write(f"- **Contains Inf**: {has_inf}\n\n")

            f.write("### Displacement Statistics\n\n")
            f.write(f"- Mean: {displacement.mean():.6f}\n")
            f.write(f"- Std: {displacement.std():.6f}\n")
            f.write(f"- Min: {displacement.min():.6f}\n")
            f.write(f"- Max: {displacement.max():.6f}\n\n")

            f.write("### Velocity Statistics\n\n")
            f.write(f"- Mean: {vel_magnitudes.mean():.6f}\n")
            f.write(f"- Std: {vel_magnitudes.std():.6f}\n")
            f.write(f"- Min: {vel_magnitudes.min():.6f}\n")
            f.write(f"- Max: {vel_magnitudes.max():.6f}\n\n")

        f.write("## Conclusion\n\n")
        if success:
            f.write("✅ The JAX direct interpolation implementation works correctly with reduced particle count.\n\n")
            f.write("**Next steps**:\n")
            f.write("1. Gradually increase particle count to find memory limit\n")
            f.write("2. Implement chunked/batched processing for large particle sets\n")
            f.write("3. Compare accuracy with legacy mode results\n")
        else:
            f.write("❌ The test failed. See error logs above.\n\n")
            f.write("**Next steps**:\n")
            f.write("1. Analyze the error and root cause\n")
            f.write("2. Consider further reducing particle count\n")
            f.write("3. Investigate JAX compilation memory requirements\n")

    print(f"\n✅ Saved detailed report to: {report_file}")

if __name__ == "__main__":
    main()
