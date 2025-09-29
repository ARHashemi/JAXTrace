def _generate_inlet_grid_particles_clean(n_particles: int, inlet_position: float,
                                       axis_idx: int, bounds: np.ndarray,
                                       concentrations: Dict[str, int]) -> np.ndarray:
    """
    Generate particles in a uniform grid directly on the inlet face with perfect spacing.

    Parameters
    ----------
    n_particles : int
        Number of particles to generate
    inlet_position : float
        Position along the flow axis where inlet is located
    axis_idx : int
        Index of the flow axis (0=x, 1=y, 2=z)
    bounds : np.ndarray
        Domain bounds [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    concentrations : dict
        User-defined concentrations {'x': int, 'y': int, 'z': int}

    Returns
    -------
    np.ndarray
        Generated particles at inlet, shape (n_particles, 3)
    """
    import numpy as np

    axis_names = ['x', 'y', 'z']

    # Get the two dimensions perpendicular to the flow axis for the inlet face
    other_axes = [i for i in range(3) if i != axis_idx]

    # Get the resolution for the two inlet face dimensions
    inlet_resolutions = []
    for i in other_axes:
        axis_name = axis_names[i]
        resolution = max(1, int(concentrations[axis_name]))
        inlet_resolutions.append(resolution)

    # Create coordinate arrays for the inlet face dimensions
    coord_arrays = []
    for i, res in enumerate(inlet_resolutions):
        axis_i = other_axes[i]
        coords = np.linspace(bounds[0, axis_i], bounds[1, axis_i], res, dtype=np.float32)
        coord_arrays.append(coords)

    # Create 2D meshgrid for the inlet face
    if len(coord_arrays) == 2:
        Grid1, Grid2 = np.meshgrid(coord_arrays[0], coord_arrays[1], indexing='ij')
        grid1_flat = Grid1.ravel()
        grid2_flat = Grid2.ravel()
    elif len(coord_arrays) == 1:
        # Handle 1D case (degenerate inlet)
        grid1_flat = coord_arrays[0]
        grid2_flat = np.array([0.0])  # Dummy dimension
    else:
        # Fallback
        grid1_flat = np.array([0.0])
        grid2_flat = np.array([0.0])

    # Total particles in the inlet face grid
    total_inlet_particles = len(grid1_flat) if len(coord_arrays) == 1 else len(grid1_flat)

    # Generate particles
    particles = np.zeros((n_particles, 3), dtype=np.float32)

    # Set flow axis coordinate to inlet position
    particles[:, axis_idx] = inlet_position

    # Improved distribution strategy: ensure even coverage without duplicates
    if n_particles <= total_inlet_particles:
        # Use systematic sampling for optimal distribution
        if n_particles == total_inlet_particles:
            # Perfect match - use all grid points
            indices = np.arange(total_inlet_particles)
        else:
            # Sample evenly across the grid
            step = total_inlet_particles / n_particles
            indices = np.round(np.arange(0, total_inlet_particles, step)[:n_particles]).astype(int)
            # Ensure indices are unique and within bounds
            indices = np.clip(indices, 0, total_inlet_particles - 1)
            indices = np.unique(indices)

            # If we lost some due to duplicates, add more spread out
            while len(indices) < n_particles:
                remaining = n_particles - len(indices)
                # Find unused indices
                all_indices = set(range(total_inlet_particles))
                used_indices = set(indices)
                unused_indices = list(all_indices - used_indices)

                if unused_indices:
                    # Add as many unused indices as needed
                    add_count = min(remaining, len(unused_indices))
                    # Space them out evenly
                    if add_count == len(unused_indices):
                        indices = np.concatenate([indices, unused_indices])
                    else:
                        step = len(unused_indices) / add_count
                        add_indices = [unused_indices[int(i * step)] for i in range(add_count)]
                        indices = np.concatenate([indices, add_indices])
                else:
                    break

                indices = np.unique(indices)
                # Safety check
                if len(indices) >= total_inlet_particles:
                    break
    else:
        # More particles than grid points - use round-robin
        base_count = n_particles // total_inlet_particles
        remainder = n_particles % total_inlet_particles

        indices = []
        for i in range(total_inlet_particles):
            indices.extend([i] * base_count)

        # Add remainder particles evenly distributed
        if remainder > 0:
            step = total_inlet_particles / remainder
            extra_indices = [int(i * step) for i in range(remainder)]
            indices.extend(extra_indices)

        indices = np.array(indices[:n_particles])

    # Assign coordinates
    for i in range(n_particles):
        grid_idx = indices[i]

        # Assign coordinates for the inlet face dimensions
        if len(other_axes) >= 1:
            particles[i, other_axes[0]] = grid1_flat[grid_idx]
        if len(other_axes) >= 2 and len(coord_arrays) == 2:
            particles[i, other_axes[1]] = grid2_flat[grid_idx]
        elif len(other_axes) >= 2:
            # For degenerate case, use middle of bounds for second dimension
            particles[i, other_axes[1]] = (bounds[0, other_axes[1]] + bounds[1, other_axes[1]]) / 2

    return particles

# Test the function
if __name__ == "__main__":
    # Test parameters
    concentrations = {'x': 3, 'y': 4, 'z': 3}
    bounds = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    axis_idx = 0  # x-axis flow
    inlet_position = 0.0

    for n_particles in [3, 5, 7, 9, 12, 15]:
        particles = _generate_inlet_grid_particles_clean(
            n_particles, inlet_position, axis_idx, bounds, concentrations
        )

        print(f"\n{n_particles} particles:")
        actual_points = [(p[1], p[2]) for p in particles]
        unique_points = list(set((round(y, 3), round(z, 3)) for y, z in actual_points))
        duplicates = len(actual_points) - len(unique_points)

        print(f"  Unique: {len(unique_points)}, Duplicates: {duplicates}")
        if len(particles) <= 8:
            for i, (y, z) in enumerate(actual_points):
                print(f"    {i}: ({y:.3f}, {z:.3f})")