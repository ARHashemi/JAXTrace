#!/usr/bin/env python3
"""
JAXTrace GPU-Optimized Fast Workflow

This is the FASTEST possible configuration:
- No memory tracking (overhead removed)
- All data pre-loaded to GPU
- JIT compilation for everything
- Minimal CPU-GPU transfers
- Reflective boundaries (JIT-compatible)
- Maximum batch size
"""

import os
# GPU optimization - MUST be before JAX imports
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

import numpy as np
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import jaxtrace as jt
from jaxtrace.fields import TimeSeriesField
from jaxtrace.tracking import create_tracker, uniform_grid_seeds
from jaxtrace.tracking.boundary import reflective_boundary
from jaxtrace.io import load_optimized_dataset

print("="*80)
print(f"JAXTrace v{jt.__version__} - GPU-Optimized Fast Workflow")
print("="*80)


def main():
    """Ultra-fast GPU-optimized workflow."""

    # 1. Configure JAXTrace for GPU
    print("\n1. GPU CONFIGURATION")
    print("="*80)

    jt.configure(
        dtype="float32",
        device="gpu",  # GPU mode
        memory_limit_gb=3.0  # 75% of 4GB
    )

    print(f"✅ Device: {jax.devices()}")
    print(f"✅ Backend: {jax.default_backend()}")

    # 2. Load and GPU-optimize velocity field
    print("\n2. LOAD & GPU-OPTIMIZE FIELD")
    print("="*80)

    field = load_and_gpu_optimize_field()

    # 3. Fast GPU particle tracking
    print("\n3. GPU-ACCELERATED TRACKING")
    print("="*80)

    trajectory = fast_gpu_tracking(field)

    # 4. Export results
    print("\n4. EXPORT")
    print("="*80)

    export_results(trajectory)

    print("\n" + "="*80)
    print("✅ WORKFLOW COMPLETE!")
    print("="*80)


def load_and_gpu_optimize_field():
    """Load field and convert everything to GPU JAX arrays."""

    pattern = "/home/arhashemi/Workspace/welding/Cases/004_caseCoarse.gid/post/0eule/004_caseCoarse_*.pvtu"

    print(f"📁 Loading: {pattern}")

    try:
        # Load optimized (essential fields only, float32)
        field = load_optimized_dataset(
            data_pattern=pattern,
            max_memory_per_timestep_mb=100.0,
            max_time_steps=40,
            dtype="float32"
        )

        print(f"✅ Loaded {field.T} timesteps, {field.N} points")

    except Exception as e:
        print(f"⚠️  VTK loading failed: {e}")
        print(f"📝 Creating synthetic field...")
        field = create_fast_synthetic_field()

    # Convert ALL data to JAX arrays on GPU
    print(f"🔄 Converting to JAX arrays on GPU...")

    field.data = jnp.array(field.data)
    field.positions = jnp.array(field.positions)
    field.times = jnp.array(field.times)

    # Update internal device arrays (critical for JIT!)
    field._data_dev = jax.device_put(field.data)
    field._times_dev = jax.device_put(field.times)
    field._pos_dev = jax.device_put(field.positions)

    # Verify GPU placement
    data_mb = field.data.nbytes / 1024 / 1024
    print(f"✅ Field on GPU: {data_mb:.1f} MB")
    print(f"   Devices: {field.data.devices()}")

    return field


def create_fast_synthetic_field():
    """Create synthetic field directly as JAX arrays."""

    # Small grid for speed
    x = np.linspace(-2, 2, 20, dtype=np.float32)
    y = np.linspace(-2, 2, 20, dtype=np.float32)
    z = np.linspace(-0.5, 0.5, 3, dtype=np.float32)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    times = np.linspace(0, 5, 15, dtype=np.float32)

    # Generate velocity
    velocity_data = []
    for t in times:
        strength = 1.0 + 0.5 * np.sin(2 * np.pi * t / 5)
        center_x = 0.3 * np.cos(2 * np.pi * t / 5)
        center_y = 0.3 * np.sin(2 * np.pi * t / 5)

        dx = positions[:, 0] - center_x
        dy = positions[:, 1] - center_y
        r_squared = dx**2 + dy**2 + 1e-6

        vx = -strength * dy / r_squared
        vy = strength * dx / r_squared
        vz = np.zeros_like(vx)

        velocities = np.stack([vx, vy, vz], axis=1).astype(np.float32)
        velocity_data.append(velocities)

    velocity_data = np.array(velocity_data)

    # Convert to JAX arrays BEFORE creating field
    velocity_data = jnp.array(velocity_data)
    positions = jnp.array(positions)
    times = jnp.array(times)

    return TimeSeriesField(
        data=velocity_data,
        times=times,
        positions=positions,
        interpolation="linear",
        extrapolation="constant"
    )


def fast_gpu_tracking(field):
    """Ultra-fast GPU tracking with JIT compilation."""

    # Get bounds
    bounds_min, bounds_max = field.get_spatial_bounds()
    print(f"📏 Bounds: {bounds_min} to {bounds_max}")

    # Particle seeding
    print(f"🎯 Seeding particles...")
    seeds = uniform_grid_seeds(
        resolution=(30, 25, 10),  # 7500 particles
        bounds=[bounds_min, bounds_max],
        include_boundaries=True
    )
    print(f"✅ {len(seeds)} particles")

    # GPU-friendly boundary (JIT-compilable)
    boundary = reflective_boundary([bounds_min, bounds_max])

    # Tracking config
    n_timesteps = 1500
    batch_size = len(seeds)  # Single batch for maximum speed!
    dt = 0.005

    print(f"🎯 Tracking config:")
    print(f"   Timesteps: {n_timesteps}")
    print(f"   Batch size: {batch_size} (single batch)")
    print(f"   dt: {dt}")

    # Create tracker with explicit JIT options
    print(f"🚀 Creating GPU tracker...")
    tracker = create_tracker(
        integrator_name='rk4',
        field=field,
        boundary_condition=boundary,
        batch_size=batch_size,
        record_velocities=False,  # Faster without velocity recording
        # Explicit JIT options
        use_jax_jit=True,
        static_compilation=True,
        use_scan_jit=True,
        device_put_inputs=True,
        progress_callback=None  # No progress overhead
    )

    # Verify JIT compilation
    if tracker._compiled_step is not None:
        print(f"   ✅ JIT: Single step COMPILED")
    else:
        print(f"   ❌ JIT: Single step FAILED")

    if tracker._compiled_simulate is not None:
        print(f"   ✅ JIT: Full scan COMPILED")
    else:
        print(f"   ❌ JIT: Full scan FAILED")

    # Run tracking
    print(f"\n🏃 Running tracking...")
    print(f"   First run: Expect 10-30s JIT compilation")
    print(f"   This is a ONE-TIME cost")

    start = time.time()

    trajectory = tracker.track_particles(
        initial_positions=seeds,
        time_span=(0.0, 4.0),
        n_timesteps=n_timesteps,
        dt=dt
    )

    elapsed = time.time() - start

    print(f"\n✅ Tracking complete!")
    print(f"   Time: {elapsed:.2f} seconds")
    print(f"   Particles: {len(seeds)}")
    print(f"   Timesteps: {n_timesteps}")
    print(f"   Rate: {len(seeds) * n_timesteps / elapsed:.0f} particle-steps/sec")

    return trajectory


def export_results(trajectory):
    """Export trajectory to VTK."""

    from jaxtrace.io import export_trajectory_to_vtk

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    try:
        print(f"💾 Exporting to VTK...")
        export_trajectory_to_vtk(
            trajectory=trajectory,
            filename=str(output_dir / "trajectory_fast.vtp"),
            include_velocities=False,
            time_series=False
        )
        print(f"   ✅ Saved: output/trajectory_fast.vtp")
    except Exception as e:
        print(f"   ⚠️  Export failed: {e}")


if __name__ == "__main__":
    main()
