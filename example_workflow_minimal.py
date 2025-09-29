#!/usr/bin/env python3
"""
Minimal JAXTrace Example

This example demonstrates the core JAXTrace workflow in ~50 lines:
1. Create synthetic velocity field
2. Seed particles randomly in domain
3. Track particles with periodic boundaries
4. Analyze final positions with KDE
5. Save basic visualization

For real VTK data, replace the synthetic field with:
field = jt.open_dataset("path/to/data/*.vtk").load_time_series()
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import jaxtrace as jt
from jaxtrace.tracking import create_tracker


def create_synthetic_field():
    """Create a simple synthetic velocity field for testing."""
    # Create structured grid
    n = 10
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    z = np.linspace(0, 1, n)

    # Create meshgrid and flatten to coordinate list
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    positions = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Two time steps with uniform flow
    times = np.array([0.0, 1.0])

    # Create velocity data: uniform flow in x-direction
    n_points = n**3
    velocity_data = np.zeros((2, n_points, 3))  # (time, points, vector)
    velocity_data[:, :, 0] = 0.1  # u = 0.1 m/s in x-direction

    return jt.TimeSeriesField(
        data=velocity_data,
        times=times,
        positions=positions,
        interpolation="linear",
        extrapolation="constant"
    )


def main():
    print("JAXTrace Minimal Example")
    print("=" * 30)

    # Configuration
    n_particles = 1000
    dt = 0.01
    t_final = 5.0
    domain_bounds = ((0, 0, 0), (1, 1, 1))

    # 1. Create velocity field (synthetic for this example)
    print("Creating velocity field...")
    field = create_synthetic_field()

    # 2. Seed particles randomly
    print(f"Seeding {n_particles} particles...")
    initial_positions = jt.random_seeds(
        n=n_particles,
        bounds=domain_bounds,
        rng_seed=42
    )

    # 3. Create tracker with periodic boundaries
    print("Setting up particle tracker...")
    tracker = create_tracker(
        integrator_name="rk4",
        field=field,
        boundary_condition=jt.periodic_boundary(domain_bounds),
        max_memory_gb=2.0,
        use_jax_jit=True
    )

    # 4. Track particles
    n_steps = int(t_final / dt)
    print(f"Tracking particles for {t_final}s with dt={dt} ({n_steps} steps)...")
    trajectory = tracker.track_particles(
        initial_positions=initial_positions,
        time_span=(0.0, t_final),
        dt=dt,
        n_timesteps=n_steps
    )
    print(f"✅ Completed: {trajectory.T} timesteps, {trajectory.N} particles")

    # 5. Analyze with KDE
    print("Analyzing density with KDE...")
    final_positions = trajectory.positions[-1]  # Final positions
    kde = jt.KDEEstimator(
        positions=final_positions,
        bandwidth_rule="scott",
        normalize=True
    )

    # 6. Create visualization
    print("Creating visualization...")
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot particle trajectories (2D projection)
    for i in range(min(100, n_particles)):  # Show first 100 trajectories
        x_traj = trajectory.positions[:, i, 0]
        y_traj = trajectory.positions[:, i, 1]
        ax1.plot(x_traj, y_traj, alpha=0.3, linewidth=0.5)

    ax1.scatter(initial_positions[:, 0], initial_positions[:, 1],
               s=2, c='green', alpha=0.7, label='Start')
    ax1.scatter(final_positions[:, 0], final_positions[:, 1],
               s=2, c='red', alpha=0.7, label='End')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Particle Trajectories (XY view)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot KDE density (2D)
    try:
        X, Y, density = kde.evaluate_2d()
        im = ax2.contourf(X, Y, density, levels=20, cmap='viridis')
        plt.colorbar(im, ax=ax2, label='Density')
        ax2.scatter(final_positions[:, 0], final_positions[:, 1],
                   s=1, c='white', alpha=0.3)
    except Exception as e:
        # Fallback if KDE fails
        ax2.scatter(final_positions[:, 0], final_positions[:, 1], s=2, alpha=0.7)
        print(f"KDE visualization failed: {e}")

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Final Density (KDE)')
    ax2.set_aspect('equal')

    # Save plot
    output_dir = Path("output_minimal")
    output_dir.mkdir(exist_ok=True)

    plt.tight_layout()
    plot_path = output_dir / "minimal_example.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization: {plot_path}")

    # Print summary statistics
    print("\nSummary:")
    print(f"  Initial positions: ({initial_positions.min():.3f}, {initial_positions.max():.3f})")
    print(f"  Final positions: ({final_positions.min():.3f}, {final_positions.max():.3f})")
    print(f"  Mean displacement: {np.mean(np.linalg.norm(final_positions - initial_positions, axis=1)):.3f}")
    print(f"  Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()