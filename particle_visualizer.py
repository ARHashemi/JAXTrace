import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from typing import Optional, Union, Tuple, List, Callable
import warnings

# JAX imports for KDE
try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.stats import gaussian_kde
    from jax import jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not available. JAX KDE and SPH functionality will be disabled.")

class DensityEstimator:
    """Base class for density estimation methods."""
    
    def __init__(self, positions: np.ndarray, **kwargs):
        self.positions = positions
        self.kwargs = kwargs
    
    def evaluate(self, grid_points: np.ndarray) -> np.ndarray:
        """Evaluate density at grid points."""
        raise NotImplementedError

class JAXGaussianKDE(DensityEstimator):
    """JAX-based Gaussian KDE estimator."""
    
    def __init__(self, positions: np.ndarray, 
                 bandwidth: Optional[float] = None,
                 bandwidth_method: str = 'scott',
                 weights: Optional[np.ndarray] = None):
        """
        Initialize JAX Gaussian KDE.
        
        Args:
            positions: Particle positions (N, 2) for 2D density
            bandwidth: Fixed bandwidth value (overrides bandwidth_method)
            bandwidth_method: Method for bandwidth selection ('scott', 'silverman', 'custom')
            weights: Particle weights (N,), optional
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for JAXGaussianKDE")
        
        super().__init__(positions)
        self.positions_jax = jnp.array(positions)
        self.weights = jnp.array(weights) if weights is not None else None
        self.n_particles, self.n_dims = self.positions_jax.shape
        
        # Calculate bandwidth
        if bandwidth is not None:
            self.bandwidth = bandwidth
        else:
            self.bandwidth = self._calculate_bandwidth(bandwidth_method)
        
        # Compile evaluation function
        self._evaluate_jit = jit(self._evaluate_density)
    
    def _calculate_bandwidth(self, method: str) -> float:
        """Calculate bandwidth using specified method."""
        if method == 'scott':
            return self.n_particles ** (-1.0 / (self.n_dims + 4))
        elif method == 'silverman':
            return (self.n_particles * (self.n_dims + 2) / 4.0) ** (-1.0 / (self.n_dims + 4))
        else:
            raise ValueError("bandwidth_method must be 'scott' or 'silverman'")
    
    def _evaluate_density(self, eval_points: jnp.ndarray) -> jnp.ndarray:
        """Evaluate KDE at given points using JAX."""
        def single_point_density(point):
            # Calculate squared distances
            diff = self.positions_jax - point[None, :]
            squared_distances = jnp.sum(diff**2, axis=1)
            
            # Gaussian kernel
            kernel_values = jnp.exp(-0.5 * squared_distances / (self.bandwidth**2))
            
            # Apply weights if provided
            if self.weights is not None:
                kernel_values *= self.weights
            
            # Normalize
            normalization = (2 * jnp.pi * self.bandwidth**2)**(self.n_dims/2) * self.n_particles
            return jnp.sum(kernel_values) / normalization
        
        return vmap(single_point_density)(eval_points)
    
    def evaluate(self, grid_points: np.ndarray) -> np.ndarray:
        """Evaluate density at grid points."""
        grid_jax = jnp.array(grid_points)
        return np.array(self._evaluate_jit(grid_jax))

class SPHDensityEstimator(DensityEstimator):
    """Smoothed Particle Hydrodynamics density estimator."""
    
    def __init__(self, positions: np.ndarray,
                 smoothing_length: Optional[float] = None,
                 kernel_type: str = 'cubic_spline',
                 adaptive: bool = False,
                 n_neighbors: int = 32,
                 masses: Optional[np.ndarray] = None):
        """
        Initialize SPH density estimator.
        
        Args:
            positions: Particle positions (N, 2) for 2D density
            smoothing_length: Fixed smoothing length
            kernel_type: Type of kernel ('cubic_spline', 'gaussian', 'wendland')
            adaptive: Whether to use adaptive smoothing lengths
            n_neighbors: Number of neighbors for adaptive smoothing
            masses: Particle masses (N,), defaults to uniform
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for SPHDensityEstimator")
        
        super().__init__(positions)
        self.positions_jax = jnp.array(positions)
        self.n_particles, self.n_dims = self.positions_jax.shape
        self.kernel_type = kernel_type
        self.adaptive = adaptive
        self.n_neighbors = n_neighbors
        
        # Set masses
        if masses is not None:
            self.masses = jnp.array(masses)
        else:
            self.masses = jnp.ones(self.n_particles)
        
        # Calculate smoothing lengths
        if smoothing_length is not None:
            self.smoothing_lengths = jnp.full(self.n_particles, smoothing_length)
        else:
            self.smoothing_lengths = self._calculate_smoothing_lengths()
        
        # Compile evaluation function
        self._evaluate_jit = jit(self._evaluate_density)
    
    def _calculate_smoothing_lengths(self) -> jnp.ndarray:
        """Calculate smoothing lengths (adaptive or fixed)."""
        if self.adaptive:
            return self._adaptive_smoothing_lengths()
        else:
            # Fixed smoothing length based on particle spacing
            n_per_dim = self.n_particles**(1.0/self.n_dims)
            h = 1.0 / n_per_dim * 2.0  # Rough estimate
            return jnp.full(self.n_particles, h)
    
    def _adaptive_smoothing_lengths(self) -> jnp.ndarray:
        """Calculate adaptive smoothing lengths based on nearest neighbors."""
        def get_smoothing_length(i):
            pos_i = self.positions_jax[i]
            distances = jnp.linalg.norm(self.positions_jax - pos_i[None, :], axis=1)
            # Get distance to k-th nearest neighbor
            sorted_distances = jnp.sort(distances)
            return sorted_distances[self.n_neighbors] * 1.2  # Scale factor
        
        return vmap(get_smoothing_length)(jnp.arange(self.n_particles))
    
    def _kernel_function(self, r: jnp.ndarray, h: float) -> jnp.ndarray:
        """SPH kernel function."""
        q = r / h
        
        if self.kernel_type == 'cubic_spline':
            # Cubic spline kernel (2D)
            sigma = 10.0 / (7.0 * jnp.pi * h**2)
            
            result = jnp.where(
                q <= 1.0,
                sigma * (1.0 - 1.5 * q**2 + 0.75 * q**3),
                jnp.where(
                    q <= 2.0,
                    sigma * 0.25 * (2.0 - q)**3,
                    0.0
                )
            )
            return result
            
        elif self.kernel_type == 'gaussian':
            # Gaussian kernel
            sigma = 1.0 / (jnp.pi * h**2)
            return sigma * jnp.exp(-q**2)
            
        elif self.kernel_type == 'wendland':
            # Wendland C2 kernel (2D)
            sigma = 7.0 / (jnp.pi * h**2)
            return jnp.where(
                q <= 1.0,
                sigma * (1.0 - q)**4 * (1.0 + 4.0 * q),
                0.0
            )
        else:
            raise ValueError("Unknown kernel type")
    
    def _evaluate_density(self, eval_points: jnp.ndarray) -> jnp.ndarray:
        """Evaluate SPH density at given points."""
        def single_point_density(point):
            # Calculate distances to all particles
            distances = jnp.linalg.norm(self.positions_jax - point[None, :], axis=1)
            
            # Calculate kernel contributions
            def particle_contribution(i):
                h_i = self.smoothing_lengths[i]
                kernel_val = self._kernel_function(distances[i], h_i)
                return self.masses[i] * kernel_val
            
            contributions = vmap(particle_contribution)(jnp.arange(self.n_particles))
            return jnp.sum(contributions)
        
        return vmap(single_point_density)(eval_points)
    
    def evaluate(self, grid_points: np.ndarray) -> np.ndarray:
        """Evaluate density at grid points."""
        grid_jax = jnp.array(grid_points)
        return np.array(self._evaluate_jit(grid_jax))

class ParticlePositionVisualizer:
    """Visualization tools for particle positions and trajectories."""
    
    def __init__(self, 
                 final_positions: Optional[np.ndarray] = None,
                 initial_positions: Optional[np.ndarray] = None,
                 trajectories: Optional[np.ndarray] = None):
        """
        Initialize visualizer with flexible input options.
        
        Args:
            final_positions: Final particle positions (N, 3)
            initial_positions: Initial particle positions (N, 3) 
            trajectories: Full particle trajectories (N, n_steps, 3) - optional
        """
        self.final_positions = final_positions
        self.initial_positions = initial_positions
        self.trajectories = trajectories
        
        # Validate inputs
        if trajectories is not None:
            self.trajectories = trajectories
            if final_positions is None:
                self.final_positions = trajectories[:, -1, :]
            if initial_positions is None:
                self.initial_positions = trajectories[:, 0, :]
        elif final_positions is not None:
            self.final_positions = final_positions
        else:
            raise ValueError("Must provide either final_positions or trajectories")
    
    def calculate_kde_density(self,
                            positions: str = 'final',
                            plane: str = 'xy',
                            position: float = 0.0,
                            slab_thickness: float = 0.1,
                            grid_resolution: int = 100,
                            method: str = 'jax_kde',
                            bandwidth: Optional[float] = None,
                            bandwidth_method: str = 'scott',
                            kernel_type: str = 'cubic_spline',
                            smoothing_length: Optional[float] = None,
                            adaptive: bool = False,
                            n_neighbors: int = 32,
                            weights: Optional[np.ndarray] = None,
                            masses: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate KDE or SPH density with full user control.
        
        Args:
            positions: Which positions to use ('initial', 'final')
            plane: Cross-section plane ('xy', 'xz', 'yz')
            position: Position along the third axis for slicing
            slab_thickness: Thickness of the slice
            grid_resolution: Resolution of the evaluation grid
            method: Density estimation method ('jax_kde', 'sph', 'seaborn')
            
            # KDE parameters
            bandwidth: Bandwidth for KDE (overrides bandwidth_method)
            bandwidth_method: Method for bandwidth selection ('scott', 'silverman')
            weights: Particle weights for KDE
            
            # SPH parameters
            kernel_type: SPH kernel type ('cubic_spline', 'gaussian', 'wendland')
            smoothing_length: Fixed smoothing length for SPH
            adaptive: Whether to use adaptive smoothing lengths
            n_neighbors: Number of neighbors for adaptive smoothing
            masses: Particle masses for SPH
            
        Returns:
            X, Y, Z: Meshgrid coordinates and density values
        """
        # Define plane mapping
        plane_maps = {
            'xy': (0, 1, 2, 'X', 'Y', 'Z'),
            'xz': (0, 2, 1, 'X', 'Z', 'Y'),
            'yz': (1, 2, 0, 'Y', 'Z', 'X')
        }
        if plane not in plane_maps:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")
        axis1, axis2, axis3, label1, label2, label3 = plane_maps[plane]
        
        # Get positions
        if positions == 'final':
            pos_3d = self.final_positions
        elif positions == 'initial':
            if self.initial_positions is None:
                raise ValueError("Initial positions not available")
            pos_3d = self.initial_positions
        else:
            raise ValueError("positions must be 'initial' or 'final'")
        
        # Filter particles in slab
        mask = np.abs(pos_3d[:, axis3] - position) <= slab_thickness / 2
        pos_slab = pos_3d[mask]
        
        if len(pos_slab) == 0:
            raise ValueError("No particles found in the specified slab")
        
        # Extract 2D positions for density calculation
        pos_2d = pos_slab[:, [axis1, axis2]]
        
        # Create evaluation grid
        x_min, x_max = pos_2d[:, 0].min(), pos_2d[:, 0].max()
        y_min, y_max = pos_2d[:, 1].min(), pos_2d[:, 1].max()
        
        # Add some padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = 0.1
        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range
        
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        
        # Calculate density
        if method == 'jax_kde':
            if not JAX_AVAILABLE:
                raise ImportError("JAX is required for jax_kde method")
            
            # Filter weights if provided
            filtered_weights = weights[mask] if weights is not None else None
            
            kde = JAXGaussianKDE(pos_2d, 
                               bandwidth=bandwidth,
                               bandwidth_method=bandwidth_method,
                               weights=filtered_weights)
            Z_flat = kde.evaluate(grid_points)
            
        elif method == 'sph':
            if not JAX_AVAILABLE:
                raise ImportError("JAX is required for sph method")
            
            # Filter masses if provided
            filtered_masses = masses[mask] if masses is not None else None
            
            sph = SPHDensityEstimator(pos_2d,
                                    smoothing_length=smoothing_length,
                                    kernel_type=kernel_type,
                                    adaptive=adaptive,
                                    n_neighbors=n_neighbors,
                                    masses=filtered_masses)
            Z_flat = sph.evaluate(grid_points)
            
        elif method == 'seaborn':
            # Fallback to seaborn (for compatibility)
            import seaborn as sns
            from scipy.stats import gaussian_kde as scipy_kde
            
            kde = scipy_kde(pos_2d.T)
            Z_flat = kde(grid_points.T)
            
        else:
            raise ValueError("method must be 'jax_kde', 'sph', or 'seaborn'")
        
        Z = Z_flat.reshape(X.shape)
        return X, Y, Z
    
    def plot_3d_positions(self, 
                         show_initial: bool = True,
                         show_final: bool = True,
                         show_trajectories: bool = None,
                         n_show: int = 1000, 
                         save_path: Optional[str] = None):
        """
        Plot 3D particle positions.
        
        Args:
            show_initial: Whether to show initial positions
            show_final: Whether to show final positions
            show_trajectories: Whether to show trajectory lines (if available)
            n_show: Maximum number of particles to show
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Determine what to show
        if show_trajectories is None:
            show_trajectories = self.trajectories is not None
        
        # Sample particles to show
        n_particles = len(self.final_positions)
        if n_particles > n_show:
            indices = np.random.choice(n_particles, n_show, replace=False)
        else:
            indices = np.arange(n_particles)
        
        # Plot trajectories if available and requested
        if show_trajectories and self.trajectories is not None:
            for i in indices:
                traj = self.trajectories[i]
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                       alpha=0.4, linewidth=0.6, color='gray')
        
        # Plot initial positions
        if show_initial and self.initial_positions is not None:
            init_pos = self.initial_positions[indices]
            ax.scatter(init_pos[:, 0], init_pos[:, 1], init_pos[:, 2], 
                      c='green', s=30, label='Initial', alpha=0.8)
        
        # Plot final positions
        if show_final:
            final_pos = self.final_positions[indices]
            ax.scatter(final_pos[:, 0], final_pos[:, 1], final_pos[:, 2], 
                      c='red', s=30, label='Final', alpha=0.8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        
        title_parts = []
        if show_initial: title_parts.append("Initial")
        if show_final: title_parts.append("Final")
        if show_trajectories and self.trajectories is not None: 
            title_parts.append("Trajectories")
        
        ax.set_title(f'Particle {" & ".join(title_parts)}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_interactive_3d(self, 
                           show_initial: bool = True,
                           show_final: bool = True,
                           show_trajectories: bool = None,
                           n_show: int = 1000):
        """
        Create interactive 3D plot using Plotly.
        
        Args:
            show_initial: Whether to show initial positions
            show_final: Whether to show final positions  
            show_trajectories: Whether to show trajectory lines
            n_show: Maximum number of particles to show
        """
        # Sample particles
        n_particles = len(self.final_positions)
        if n_particles > n_show:
            indices = np.random.choice(n_particles, n_show, replace=False)
        else:
            indices = np.arange(n_particles)
        
        fig = go.Figure()
        
        # Determine what to show
        if show_trajectories is None:
            show_trajectories = self.trajectories is not None
        
        # Add trajectories
        if show_trajectories and self.trajectories is not None:
            for i, idx in enumerate(indices):
                traj = self.trajectories[idx]
                fig.add_trace(go.Scatter3d(
                    x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                    mode='lines',
                    name=f'Trajectory {idx}',
                    line=dict(width=2, color='gray'),
                    showlegend=False,
                    opacity=0.6
                ))
        
        # Add initial positions
        if show_initial and self.initial_positions is not None:
            init_pos = self.initial_positions[indices]
            fig.add_trace(go.Scatter3d(
                x=init_pos[:, 0], y=init_pos[:, 1], z=init_pos[:, 2],
                mode='markers',
                marker=dict(color='green', size=6),
                name='Initial Positions'
            ))
        
        # Add final positions
        if show_final:
            final_pos = self.final_positions[indices]
            fig.add_trace(go.Scatter3d(
                x=final_pos[:, 0], y=final_pos[:, 1], z=final_pos[:, 2],
                mode='markers',
                marker=dict(color='red', size=6),
                name='Final Positions'
            ))
        
        fig.update_layout(
            title='Interactive Particle Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        fig.show()
    
    def plot_displacement_analysis(self, save_path: Optional[str] = None):
        """
        Analyze and plot particle displacements.
        
        Args:
            save_path: Path to save figure
        """
        if self.initial_positions is None:
            raise ValueError("Initial positions required for displacement analysis")
        
        # Calculate displacements
        displacement = self.final_positions - self.initial_positions
        displacement_magnitude = np.linalg.norm(displacement, axis=1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Displacement magnitude histogram
        ax = axes[0, 0]
        ax.hist(displacement_magnitude, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Displacement Magnitude')
        ax.set_ylabel('Count')
        ax.set_title('Displacement Magnitude Distribution')
        ax.grid(True, alpha=0.3)
        
        # Displacement components
        ax = axes[0, 1]
        ax.hist(displacement[:, 0], bins=30, alpha=0.7, label='X', density=True)
        ax.hist(displacement[:, 1], bins=30, alpha=0.7, label='Y', density=True)
        ax.hist(displacement[:, 2], bins=30, alpha=0.7, label='Z', density=True)
        ax.set_xlabel('Displacement')
        ax.set_ylabel('Density')
        ax.set_title('Displacement Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3D displacement vectors (sample)
        ax = axes[1, 0]
        n_sample = min(1000, len(displacement))
        sample_idx = np.random.choice(len(displacement), n_sample, replace=False)
        
        ax.scatter(self.initial_positions[sample_idx, 0], 
                  self.initial_positions[sample_idx, 1], 
                  c='green', s=10, alpha=0.6, label='Initial')
        ax.scatter(self.final_positions[sample_idx, 0], 
                  self.final_positions[sample_idx, 1], 
                  c='red', s=10, alpha=0.6, label='Final')
        
        # Add displacement vectors
        for i in sample_idx[:100]:  # Show only first 100 vectors
            ax.arrow(self.initial_positions[i, 0], self.initial_positions[i, 1],
                    displacement[i, 0], displacement[i, 1],
                    head_width=0.01, head_length=0.01, fc='blue', ec='blue', alpha=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Displacement Vectors (XY Plane)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Displacement magnitude vs distance from origin
        ax = axes[1, 1]
        initial_distance = np.linalg.norm(self.initial_positions, axis=1)
        ax.scatter(initial_distance, displacement_magnitude, alpha=0.6, s=1)
        ax.set_xlabel('Initial Distance from Origin')
        ax.set_ylabel('Displacement Magnitude')
        ax.set_title('Displacement vs Initial Position')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print(f"Displacement Statistics:")
        print(f"  Mean displacement: {np.mean(displacement_magnitude):.4f}")
        print(f"  Std displacement: {np.std(displacement_magnitude):.4f}")
        print(f"  Max displacement: {np.max(displacement_magnitude):.4f}")
        print(f"  Min displacement: {np.min(displacement_magnitude):.4f}")
    
    def plot_cross_sections(self, 
                           plane: str = 'xy', 
                           position: float = 0.0,
                           slab_thickness: float = 0.1,
                           show_initial: bool = True,
                           show_final: bool = True,
                           save_path: Optional[str] = None):
        """
        Plot cross-sections of particle positions.
        
        Args:
            plane: Cross-section plane ('xy', 'xz', 'yz')
            position: Position along the third axis for slicing
            slab_thickness: Thickness of the slice
            show_initial: Whether to show initial positions
            show_final: Whether to show final positions
            save_path: Path to save figure
        """
        # Define plane mappings
        plane_maps = {
            'xy': (0, 1, 2, 'X', 'Y', 'Z'),
            'xz': (0, 2, 1, 'X', 'Z', 'Y'),
            'yz': (1, 2, 0, 'Y', 'Z', 'X')
        }
        
        if plane not in plane_maps:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")
        
        axis1, axis2, axis3, label1, label2, label3 = plane_maps[plane]
        
        # Filter particles in the slab
        def filter_slab(positions):
            if positions is None:
                return None
            mask = np.abs(positions[:, axis3] - position) <= slab_thickness / 2
            return positions[mask]
        
        initial_slab = filter_slab(self.initial_positions) if show_initial else None
        final_slab = filter_slab(self.final_positions) if show_final else None
        
        # Create plots
        n_plots = sum([show_initial, show_final])
        if n_plots == 0:
            raise ValueError("Must show at least initial or final positions")
        
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot initial positions
        if show_initial and initial_slab is not None:
            ax = axes[plot_idx]
            scatter = ax.scatter(initial_slab[:, axis1], initial_slab[:, axis2], 
                               c=initial_slab[:, axis3], alpha=0.6, s=5, 
                               cmap='viridis', label='Initial')
            ax.set_xlabel(label1)
            ax.set_ylabel(label2)
            ax.set_title(f'Initial Positions - {plane.upper()} Plane')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label=label3)
            plot_idx += 1
        
        # Plot final positions
        if show_final and final_slab is not None:
            ax = axes[plot_idx]
            scatter = ax.scatter(final_slab[:, axis1], final_slab[:, axis2], 
                               c=final_slab[:, axis3], alpha=0.6, s=5, 
                               cmap='plasma', label='Final')
            ax.set_xlabel(label1)
            ax.set_ylabel(label2)
            ax.set_title(f'Final Positions - {plane.upper()} Plane')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label=label3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_kde_density(self,
                        positions: str = 'final',
                        plane: str = 'xy',
                        position: float = 0.0,
                        slab_thickness: float = 0.1,
                        method: str = 'jax_kde',
                        grid_resolution: int = 100,
                        levels: int = 10,
                        figsize: Tuple[int, int] = (12, 5),
                        cmap: str = 'viridis',
                        save_path: Optional[str] = None,
                        **kwargs):
        """
        Plot KDE or SPH density of particle positions.
        
        Args:
            positions: Which positions to use ('initial', 'final')
            plane: Cross-section plane ('xy', 'xz', 'yz')
            position: Position along the third axis for slicing
            slab_thickness: Thickness of the slice
            method: Density estimation method ('jax_kde', 'sph', 'seaborn')
            grid_resolution: Resolution of the evaluation grid
            levels: Number of contour levels
            figsize: Figure size
            cmap: Colormap
            save_path: Path to save figure
            **kwargs: Additional parameters for density estimation methods
        """
        # Calculate density
        X, Y, Z = self.calculate_kde_density(
            positions=positions,
            plane=plane,
            position=position,
            slab_thickness=slab_thickness,
            grid_resolution=grid_resolution,
            method=method,
            **kwargs
        )
        
        # Get plane labels
        plane_maps = {
            'xy': ('X', 'Y', 'Z'),
            'xz': ('X', 'Z', 'Y'),
            'yz': ('Y', 'Z', 'X')
        }
        label1, label2, label3 = plane_maps[plane]
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Plot filled contours
        contourf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.8)
        plt.colorbar(contourf, ax=ax, label='Density')
        
        # Add contour lines
        contour = ax.contour(X, Y, Z, levels=levels//2, colors='black', alpha=0.3, linewidths=0.5)
        
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        ax.set_title(f'{method.upper()} Density - {plane.upper()} plane at {label3}={position:.3f}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_combined_analysis(self,
                              plane: str = 'xy',
                              position: float = 0.0,
                              slab_thickness: float = 0.1,
                              method: str = 'jax_kde',
                              grid_resolution: int = 100,
                              save_path: Optional[str] = None,
                              **kwargs):
        """
        Combined analysis plot with positions, displacements, and advanced density estimation.
        
        Args:
            plane: Cross-section plane ('xy', 'xz', 'yz')
            position: Position along the third axis for slicing
            slab_thickness: Thickness of the slice
            method: Density estimation method ('jax_kde', 'sph', 'seaborn')
            grid_resolution: Resolution of the evaluation grid
            save_path: Path to save figure
            **kwargs: Additional parameters for density estimation methods
        """
        # Define plane mapping
        plane_maps = {
            'xy': (0, 1, 2, 'X', 'Y', 'Z'),
            'xz': (0, 2, 1, 'X', 'Z', 'Y'),
            'yz': (1, 2, 0, 'Y', 'Z', 'X')
        }
        if plane not in plane_maps:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")
        axis1, axis2, axis3, label1, label2, label3 = plane_maps[plane]
        
        # Filter particles in slab
        mask = np.abs(self.final_positions[:, axis3] - position) <= slab_thickness / 2
        final_slab = self.final_positions[mask]
        
        if len(final_slab) == 0:
            raise ValueError("No particles found in the specified slab")
        
        # Calculate density using selected method
        X, Y, Z = self.calculate_kde_density(
            positions='final',
            plane=plane,
            position=position,
            slab_thickness=slab_thickness,
            grid_resolution=grid_resolution,
            method=method,
            **kwargs
        )
        
        # Setup figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Top-left: Final positions scatter
        ax = axes[0, 0]
        scatter = ax.scatter(final_slab[:, axis1], final_slab[:, axis2], 
                           c=final_slab[:, axis3], alpha=0.6, s=10, cmap='viridis')
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        ax.set_title('Final Positions')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label=label3)
        
        # Top-right: Density heatmap
        ax = axes[0, 1]
        contourf = ax.contourf(X, Y, Z, levels=20, cmap='plasma', alpha=0.8)
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        ax.set_title(f'{method.upper()} Density')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.colorbar(contourf, ax=ax, label='Density')
        
        # Bottom-left: Initial vs Final (if available)
        ax = axes[1, 0]
        if self.initial_positions is not None:
            initial_mask = np.abs(self.initial_positions[:, axis3] - position) <= slab_thickness / 2
            initial_slab = self.initial_positions[initial_mask]
            
            ax.scatter(initial_slab[:, axis1], initial_slab[:, axis2], 
                      c='green', alpha=0.6, s=5, label='Initial')
            ax.scatter(final_slab[:, axis1], final_slab[:, axis2], 
                      c='red', alpha=0.6, s=5, label='Final')
            ax.legend()
            ax.set_title('Initial vs Final Positions')
        else:
            ax.scatter(final_slab[:, axis1], final_slab[:, axis2], 
                      c='red', alpha=0.6, s=10)
            ax.set_title('Final Positions')
        
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Bottom-right: Density contours
        ax = axes[1, 1]
        contour = ax.contour(X, Y, Z, levels=8, alpha=0.7, linewidths=1.0)#, colors='black'
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.3f')
        ax.scatter(final_slab[:, axis1], final_slab[:, axis2], 
                  c='black', alpha=0.1, s=1)
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        fig.colorbar(contour, ax=ax, shrink=0.8)
        ax.set_title(f'{method.upper()} Density Contours')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.suptitle(f'Particle Analysis ({method.upper()}) - {plane.upper()} plane at {label3}={position:.3f}', 
                     fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Convenience function for quick visualization
def quick_visualize_positions(final_positions: np.ndarray,
                            initial_positions: Optional[np.ndarray] = None,
                            mode: str = '3d',
                            **kwargs):
    """
    Quick visualization function for particle positions.
    
    Args:
        final_positions: Final particle positions (N, 3)
        initial_positions: Initial particle positions (N, 3), optional
        mode: Visualization mode ('3d', 'cross_section', 'kde', 'analysis')
        **kwargs: Additional arguments for the visualization method
    """
    viz = ParticlePositionVisualizer(final_positions, initial_positions)
    
    if mode == '3d':
        viz.plot_3d_positions(**kwargs)
    elif mode == 'cross_section':
        viz.plot_cross_sections(**kwargs)
    elif mode == 'kde':
        viz.plot_kde_density(**kwargs)
    elif mode == 'analysis':
        viz.plot_combined_analysis(**kwargs)
    else:
        raise ValueError("mode must be '3d', 'cross_section', 'kde', or 'analysis'")

# Example usage with new density estimation methods
def example_usage():
    """Example of how to use the visualizer with different density estimation methods."""
    
    # Generate example data
    np.random.seed(42)
    n_particles = 10000
    
    # Create clustered data
    centers = [(0, 0, 0), (3, 3, 0), (-2, 4, 1)]
    final_pos = []
    initial_pos = []
    
    for center in centers:
        n_cluster = n_particles // len(centers)
        cluster_final = np.random.multivariate_normal(center, np.eye(3) * 0.5, n_cluster)
        cluster_initial = cluster_final + np.random.normal(0, 0.2, (n_cluster, 3))
        
        final_pos.append(cluster_final)
        initial_pos.append(cluster_initial)
    
    final_positions = np.vstack(final_pos)
    initial_positions = np.vstack(initial_pos)
    
    # Create visualizer
    viz = ParticlePositionVisualizer(final_positions=final_positions, 
                                   initial_positions=initial_positions)
    
    print("Example 1: JAX KDE Analysis")
    # JAX KDE with custom parameters
    viz.plot_combined_analysis(
        plane='xy', position=0.0, slab_thickness=0.5,
        method='jax_kde',
        bandwidth=0.3,
        bandwidth_method='scott'
    )
    
    if JAX_AVAILABLE:
        print("Example 2: SPH Density Analysis")
        # SPH density estimation
        viz.plot_combined_analysis(
            plane='xy', position=0.0, slab_thickness=0.5,
            method='sph',
            kernel_type='cubic_spline',
            adaptive=True,
            n_neighbors=20
        )
        
        print("Example 3: Custom KDE calculation")
        # Direct density calculation
        X, Y, Z = viz.calculate_kde_density(
            positions='final',
            plane='xy',
            position=0.0,
            slab_thickness=0.5,
            method='jax_kde',
            bandwidth=0.2,
            grid_resolution=150
        )
        
        # Custom plotting
        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, Z, levels=20, cmap='plasma')
        plt.colorbar(label='Density')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Custom JAX KDE Density Plot')
        plt.axis('equal')
        plt.show()
    
    print("Example 4: Seaborn fallback")
    # Fallback to seaborn
    viz.plot_kde_density(
        positions='final',
        plane='xy',
        position=0.0,
        slab_thickness=0.5,
        method='seaborn'
    )

if __name__ == "__main__":
    example_usage()