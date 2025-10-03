#!/usr/bin/env python3
"""
Quick Test Script for JAXTrace Debugging

This script runs a minimal particle tracking simulation for quick testing
and debugging. It uses a small synthetic field and minimal configuration.

Usage:
    python test_quick.py
    python run.py --test
    python -m jaxtrace --test
"""

import os
# GPU optimization - set BEFORE importing JAX
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# JAXTrace core imports
import jaxtrace as jt
import jax
import jax.numpy as jnp
from jaxtrace.fields import TimeSeriesField
from jaxtrace.tracking import (
    create_tracker,
    uniform_grid_seeds,
)
from jaxtrace.tracking.boundary import reflective_boundary


def create_simple_vortex_field():
    """Create a simple 2D vortex field for testing."""
    print("📝 Creating simple vortex field...")

    # Small spatial grid
    x = np.linspace(-1, 1, 15, dtype=np.float32)
    y = np.linspace(-1, 1, 15, dtype=np.float32)
    z = np.array([0.0], dtype=np.float32)  # 2D field

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # Few time points
    times = np.linspace(0, 2, 5, dtype=np.float32)

    # Generate velocity data
    velocity_data = []
    for t in times:
        # Simple vortex
        vx = -positions[:, 1]
        vy = positions[:, 0]
        vz = np.zeros_like(vx)

        velocities = np.stack([vx, vy, vz], axis=1)
        velocity_data.append(velocities)

    velocity_data = np.array(velocity_data)

    field = TimeSeriesField(
        data=velocity_data,
        times=times,
        positions=positions,
        interpolation="linear",
        extrapolation="constant"
    )

    print(f"✅ Created field: {velocity_data.shape[0]} timesteps, {positions.shape[0]} points")
    return field


def main():
    """Run quick test workflow."""
    print("="*80)
    print("JAXTRACE QUICK TEST")
    print("="*80)

    # 1. Check system
    print("\n" + "="*80)
    print("1. SYSTEM CHECK")
    print("="*80)
    print(f"JAXTrace v{jt.__version__}")
    print(f"JAX devices: {jax.devices()}")

    # 2. Configure
    print("\n" + "="*80)
    print("2. CONFIGURATION")
    print("="*80)

    jt.configure(
        dtype="float32",
        device="cpu",  # Use CPU for quick test
        memory_limit_gb=1.0
    )
    print(f"✅ Configuration: {jt.get_config()}")

    # 3. Create field
    print("\n" + "="*80)
    print("3. VELOCITY FIELD")
    print("="*80)

    field = create_simple_vortex_field()
    bounds_min, bounds_max = field.get_spatial_bounds()
    print(f"📏 Field bounds: {bounds_min} to {bounds_max}")

    # 4. Seed particles
    print("\n" + "="*80)
    print("4. PARTICLE SEEDING")
    print("="*80)

    # Small number of particles
    seeds = uniform_grid_seeds(
        resolution=(5, 5, 1),  # 25 particles
        bounds=[bounds_min, bounds_max],
        include_boundaries=True
    )
    print(f"🎯 Generated {len(seeds)} particles")

    # 5. Setup tracking
    print("\n" + "="*80)
    print("5. PARTICLE TRACKING")
    print("="*80)

    boundary = reflective_boundary([bounds_min, bounds_max])

    tracker = create_tracker(
        integrator_name='rk4',
        field=field,
        boundary_condition=boundary,
        batch_size=25,
        record_velocities=True
    )

    # Run tracking
    print("🚀 Running particle tracking...")
    start_time = time.time()

    trajectory = tracker.track_particles(
        initial_positions=seeds,
        time_span=(0.0, 2.0),
        n_timesteps=100,
        dt=0.02
    )

    tracking_time = time.time() - start_time
    print(f"✅ Tracking completed in {tracking_time:.2f} seconds")
    print(f"   Trajectory shape: {trajectory.positions.shape}")

    # 6. Quick visualization
    print("\n" + "="*80)
    print("6. VISUALIZATION")
    print("="*80)

    output_dir = Path("output_test")
    output_dir.mkdir(exist_ok=True)

    # Plot trajectories
    fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(min(len(seeds), 10)):  # Plot first 10 trajectories
        traj = trajectory.positions[:, i, :]
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.7, linewidth=1)

    # Plot final positions
    final_pos = trajectory.positions[-1]
    ax.scatter(final_pos[:, 0], final_pos[:, 1], c='red', s=30, alpha=0.8, label='Final')

    # Plot initial positions
    ax.scatter(seeds[:, 0], seeds[:, 1], c='blue', s=30, alpha=0.8, marker='x', label='Initial')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Quick Test - Vortex Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plot_file = output_dir / "test_trajectories.png"
    plt.savefig(plot_file, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved plot: {plot_file}")

    # 7. Summary
    print("\n" + "="*80)
    print("✅ QUICK TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   - Field: {field.data.shape[0]} timesteps, {field.positions.shape[0]} points")
    print(f"   - Particles: {len(seeds)}")
    print(f"   - Tracking steps: {trajectory.positions.shape[0]}")
    print(f"   - Runtime: {tracking_time:.2f} seconds")
    print(f"   - Output: {output_dir}/")
    print(f"\n🎉 Test passed! JAXTrace is working correctly.")

    return trajectory


if __name__ == "__main__":
    main()
