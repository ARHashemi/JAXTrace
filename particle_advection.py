import vtk_velocity_reader as vtk_vr
import optimized_particle_advection as opa
import particle_visualizer as pviz
import numpy as np
import time

def main():
    """Main function demonstrating the particle advection workflow."""
    
    # Example usage
    print("=== Lagrangian Particle Advection with JAX ===")
    
    # 1. Load VTK data
    print("\n1. Loading VTK velocity data...")
    vtk_reader = vtk_vr.VTKVelocityReader("../Cases/001_caseCoarse.gid/post/0eule/*_caseCoarse_*.pvtu")  # Adjust pattern as needed
    
    try:
        velocity_data = vtk_reader.load_velocity_data()
        print(f"Loaded {len(vtk_reader.time_files)} time steps")
        print(f"Grid points: {velocity_data['points'].shape}")
        print(f"Velocity data shape: {velocity_data['velocity'].shape}")
    except Exception as e:
        print(f"Error loading VTK data: {e}")
        print("Creating synthetic data for demonstration...")
        
        # Create synthetic data for demonstration
        n_points = 10000
        n_times = 2500
        grid_points = np.random.uniform(-1, 1, (n_points, 3))
        velocity_data_synthetic = np.random.uniform(-0.1, 0.1, (n_times, n_points, 3))
        time_steps = np.linspace(0, 1, n_times)
        
        velocity_data = {
            'points': grid_points,
            'velocity': velocity_data_synthetic,
            'times': time_steps
        }
    
    # 2. Initialize particle advection
    print("\n2. Initializing JAX particle advection...")
    advection = opa.MemoryOptimizedJAXParticleAdvection(
        velocity_data['points'],
        velocity_data['velocity'],
        velocity_data['times'] [140],
        static_time_step=140  # Use a specific time step for static velocity
    )
    
    # 3. Create initial particle positions
    print("\n3. Creating initial particle grid...")
    box_bounds = ((-0.025, -0.010), (-0.0198, 0.0198), (-0.008, 0.0))  # Adjust as needed
    resolution = (10, 50, 20)  # Adjust density as needed
    
    initial_positions = advection.create_particle_grid(box_bounds, resolution)
    print(f"Created {initial_positions.shape[0]} particles")
    
    # 4. Advect particles
    print("\n4. Advecting particles...")
    dt = 0.003  # Time stepfreeIrani@2025


    n_steps = 2500 # Number of time steps to advect
      
    start_time = time.time()    
    final_positions = advection.advect_particles_minimal_memory(initial_positions, dt, n_steps)#, trajectories , save_trajectory=True, initial_transient=20
    end_time = time.time()
    
    print(f"Advection completed in {end_time - start_time:.2f} seconds")
    print(f"Trajectory shape: {final_positions.shape}")
    
    # 5. Visualize results
    print("\n5. Creating visualizations...")
    # pvis.quick_visualize_positions(final_pos, initial_positions, mode='analysis', plane='yz', save_path='particle_advection_analysis.png', slab_thickness=0.2, bandwidth=0.5)
    # JAX KDE with custom bandwidth
    # Create visualizer
    viz = pviz.ParticlePositionVisualizer(final_positions=final_positions, 
                                   initial_positions=initial_positions)
    viz.plot_combined_analysis(
        method='jax_kde',
        bandwidth=0.001,
        slab_thickness=0.2,
        bandwidth_method='scott',
        plane='yz', save_path='particle_advection_kde_analysis_yz.png'
    )

    # SPH with adaptive smoothing
    viz.plot_combined_analysis(
        method='sph',
        kernel_type='cubic_spline',
        adaptive=True,
        n_neighbors=5,
        smoothing_length=0.001,
        slab_thickness=0.2,
        plane='yz',
        save_path='particle_advection_sph_analysis_yz.png'
    )

    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    # Check JAX GPU availability
    # print(f"JAX devices: {jax.devices()}")
    # print(f"JAX backend: {jax.default_backend()}")
    
    main()
