# def plot_combined_analysis(self,
#                               plane: str = 'xy',
#                               position: float = 0.0,
#                               slab_thickness: float = 0.1,
#                               method: str = 'jax_kde',
#                               grid_resolution: int = 100,
#                               figsize: Tuple[int, int] = (15, 12),
#                               normalize: bool = True,
#                               threshold_percent: float = 25.0,
#                               save_path: Optional[str] = None,
#                               **kwargs):
#         """
#         Combined analysis plot with positions, density, and threshold density visualization.
        
#         Args:
#             plane: Cross-section plane ('xy', 'xz', 'yz')
#             position: Position along the third axis for slicing
#             slab_thickness: Thickness of the slice
#             method: Density estimation method ('jax_kde', 'sph', 'seaborn')
#             grid_resolution: Resolution of the evaluation grid
#             figsize: Figure size
#             normalize: Whether to normalize density
#             threshold_percent: Threshold percentage for density cutoff (default 25%)
#             save_path: Path to save figure
#             **kwargs: Additional parameters for density estimation methods
#         """
#         # Define plane mapping
#         plane_maps = {
#             'xy': (0, 1, 2, 'X', 'Y', 'Z'),
#             'xz': (0, 2, 1, 'X', 'Z', 'Y'),
#             'yz': (1, 2, 0, 'Y', 'Z', 'X')
#         }
#         if plane not in plane_maps:
#             raise ValueError("plane must be 'xy', 'xz', or 'yz'")
#         axis1, axis2, axis3, label1, label2, label3 = plane_maps[plane]
        
#         # Filter particles in slab
#         mask = np.abs(self.final_positions[:, axis3] - position) <= slab_thickness / 2
#         final_slab = self.final_positions[mask]
        
#         if len(final_slab) == 0:
#             raise ValueError("No particles found in the specified slab")
        
#         # Calculate density
#         X, Y, Z = self.calculate_density(
#             positions='final',
#             plane=plane,
#             position=position,
#             slab_thickness=slab_thickness,
#             grid_resolution=grid_resolution,
#             method=method,
#             normalize=normalize,
#             **kwargs
#         )
        
#         # Apply threshold for bottom-left plot
#         Z_thresh = self.apply_density_threshold(Z, threshold_percent, 'percent_max')
        
#         # Setup figure
#         fig, axes = plt.subplots(2, 2, figsize=figsize)
        
#         # Top-left: Final positions scatter
#         ax = axes[0, 0]
#         scatter = ax.scatter(final_slab[:, axis1], final_slab[:, axis2], 
#                            c=final_slab[:, axis3], alpha=0.6, s=10, cmap='viridis')
#         ax.set_xlabel(label1)
#         ax.set_ylabel(label2)
#         ax.set_title('Final Particle Positions')
#         ax.grid(True, alpha=0.3)
#         ax.set_aspect('equal')
#         plt.colorbar(scatter, ax=ax, label=label3)
        
#         # Top-right: Full density heatmap
#         ax = axes[0, 1]
#         contourf = ax.contourf(X, Y, Z, levels=20, cmap='plasma', alpha=0.8)
#         ax.set_xlabel(label1)
#         ax.set_ylabel(label2)
        
#         density_title = f'{"Normalized " if normalize else ""}{method.upper()} Density'
#         ax.set_title(density_title)
#         ax.grid(True, alpha=0.3)
#         ax.set_aspect('equal')
        
#         density_label = "Normalized Density" if normalize else "Density"
#         plt.colorbar(contourf, ax=ax, label=density_label)
        
#         # Bottom-left: Density with threshold cutoff (NEW - replaces initial vs final)
#         ax = axes[1, 0]
#         im = ax.imshow(Z_thresh, extent=[X.min(), X.max(), Y.min(), Y.max()], 
#                       origin='lower', cmap='plasma', aspect='equal')
        
#         # Overlay some particles for reference
#         if len(final_slab) > 1000:
#             sample_idx = np.random.choice(len(final_slab), 1000, replace=False)
#             sample_particles = final_slab[sample_idx]
#         else:
#             sample_particles = final_slab
            
#         ax.scatter(sample_particles[:, axis1], sample_particles[:, axis2], 
#                   c='white', s=1, alpha=0.5, edgecolors='black', linewidths=0.1)
        
#         ax.set_xlabel(label1)
#         ax.set_ylabel(label2)
#         ax.set_title(f'Density >{threshold_percent}% Threshold')
#         ax.grid(True, alpha=0.3)
#         ax.set_aspect('equal')
        
#         thresh_label = f"Density (>{threshold_percent}% max)"
#         plt.colorbar(im, ax=ax, label=thresh_label)
        
#         # Bottom-right: Density contours with statistics
#         ax = axes[1, 1]
#         contour = ax.contour(X, Y, Z, levels=8, alpha=0.7, linewidths=1.0)
#         ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
        
#         # Add some sample particles as background
#         sample_size = min(500, len(final_slab))
#         sample_idx = np.random.choice(len(final_slab), sample_size, replace=False)
#         sample_particles = final_slab[sample_idx]
#         ax.scatter(sample_particles[:, axis1], sample_particles[:, axis2], 
#                   c='black', alpha=0.1, s=1)
        
#         ax.set_xlabel(label1)
#         ax.set_ylabel(label2)
#         plt.colorbar(contour, ax=ax, shrink=0.8, label=density_label)
#         ax.set_title(f'{method.upper()} Density Contours')
#         ax.grid(True, alpha=0.3)
#         ax.set_aspect('equal')
        
#         # Add density statistics as text
#         density_stats = self._calculate_density_statistics(Z, Z_thresh, threshold_percent)
#         stats_text = f"Max: {density_stats['max']:.2f}\n"
#         stats_text += f"Mean: {density_stats['mean']:.2f}\n"
#         stats_text += f"Std: {density_stats['std']:.2f}\n"
#         stats_text += f"Above {threshold_percent}%: {density_stats['above_threshold_percent']:.1f}%"
        
#         ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
#                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
#                verticalalignment='top', fontsize=9)
        
#         # Overall title
#         norm_title = "Normalized " if normalize else ""
#         plt.suptitle(f'{norm_title}Particle Density Analysis ({method.upper()}) - {plane.upper()} plane at {label3}={position:.3f}', 
#                      fontsize=14)
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.show()
    
#     def _calculate_density_statistics(self, 
#                                     density: np.ndarray, 
#                                     thresholded_density: np.ndarray,
#                                     threshold_percent: float) -> Dict:
#         """Calculate statistics for density field."""
#         # Remove NaN values for statistics
#         valid_density = density[~np.isnan(density)]
#         valid_thresh_density = thresholded_density[~np.isnan(thresholded_density)]
        
#         stats = {
#             'max': float(np.max(valid_density)) if len(valid_density) > 0 else 0.0,
#             'mean': float(np.mean(valid_density)) if len(valid_density) > 0 else 0.0,
#             'std': float(np.std(valid_density)) if len(valid_density) > 0 else 0.0,
#             'min': float(np.min(valid_density)) if len(valid_density) > 0 else 0.0,
#             'above_threshold_percent': 0.0
#         }
        
#         if len(valid_density) > 0 and len(valid_thresh_density) > 0:
#             stats['above_threshold_percent'] = (len(valid_thresh_density) / len(valid_density)) * 100
        
#         return stats

"""
Particle Visualizer Module for JAXTrace

Advanced visualization tools for particle positions, trajectories, and density analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
from typing import Optional, Union, Tuple, List, Callable, Dict
import warnings

# Optional imports
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive visualizations will be disabled.")

# JAX imports for advanced density estimation
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not available. Advanced density estimation will use fallback methods.")


class DensityEstimator:
    """Base class for density estimation methods."""
    
    def __init__(self, positions: np.ndarray, **kwargs):
        self.positions = positions
        self.kwargs = kwargs
    
    def evaluate(self, grid_points: np.ndarray) -> np.ndarray:
        """Evaluate density at grid points."""
        raise NotImplementedError


class JAXGaussianKDE(DensityEstimator):
    """JAX-based Gaussian KDE estimator for high performance."""
    
    def __init__(self, positions: np.ndarray, 
                 bandwidth: Optional[float] = None,
                 bandwidth_method: str = 'scott',
                 weights: Optional[np.ndarray] = None):
        """
        Initialize JAX Gaussian KDE.
        
        Args:
            positions: Particle positions (N, 2) for 2D density
            bandwidth: Fixed bandwidth value (overrides bandwidth_method)
            bandwidth_method: Method for bandwidth selection ('scott', 'silverman')
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
    """Smoothed Particle Hydrodynamics density estimator with memory optimization."""
    
    def __init__(self, positions: np.ndarray,
                 smoothing_length: Optional[float] = None,
                 kernel_type: str = 'cubic_spline',
                 adaptive: bool = False,
                 n_neighbors: int = 32,
                 masses: Optional[np.ndarray] = None,
                 max_particles_for_adaptive: int = 5000):
        """
        Initialize SPH density estimator.
        
        Args:
            positions: Particle positions (N, 2) for 2D density
            smoothing_length: Fixed smoothing length
            kernel_type: Type of kernel ('cubic_spline', 'gaussian', 'wendland')
            adaptive: Whether to use adaptive smoothing lengths
            n_neighbors: Number of neighbors for adaptive smoothing
            masses: Particle masses (N,), defaults to uniform
            max_particles_for_adaptive: Maximum particles for adaptive method (memory limit)
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for SPHDensityEstimator")
        
        super().__init__(positions)
        self.positions_jax = jnp.array(positions)
        self.n_particles, self.n_dims = self.positions_jax.shape
        self.kernel_type = kernel_type
        self.adaptive = adaptive
        self.n_neighbors = min(n_neighbors, self.n_particles - 1)  # Ensure valid neighbor count
        self.max_particles_for_adaptive = max_particles_for_adaptive
        
        # Set masses
        if masses is not None:
            self.masses = jnp.array(masses)
        else:
            self.masses = jnp.ones(self.n_particles)
        
        # Check if we should use adaptive smoothing based on memory constraints
        if self.adaptive and self.n_particles > self.max_particles_for_adaptive:
            print(f"Warning: Too many particles ({self.n_particles}) for adaptive SPH. Using fixed smoothing length.")
            self.adaptive = False
        
        # Calculate smoothing lengths
        if smoothing_length is not None:
            self.smoothing_lengths = jnp.full(self.n_particles, smoothing_length)
        else:
            self.smoothing_lengths = self._calculate_smoothing_lengths()
        
        # Compile evaluation function
        self._evaluate_jit = jit(self._evaluate_density)
    
    def _calculate_smoothing_lengths(self) -> jnp.ndarray:
        """Calculate smoothing lengths (adaptive or fixed) with memory optimization."""
        if self.adaptive and self.n_particles <= self.max_particles_for_adaptive:
            try:
                return self._adaptive_smoothing_lengths()
            except Exception as e:
                print(f"Warning: Adaptive smoothing failed ({e}). Falling back to fixed smoothing.")
                self.adaptive = False
                return self._fixed_smoothing_length()
        else:
            return self._fixed_smoothing_length()
    
    def _fixed_smoothing_length(self) -> jnp.ndarray:
        """Calculate fixed smoothing length based on particle spacing."""
        # Estimate average particle spacing
        n_per_dim = self.n_particles**(1.0/self.n_dims)
        
        # Get domain extents
        pos_min = jnp.min(self.positions_jax, axis=0)
        pos_max = jnp.max(self.positions_jax, axis=0)
        domain_size = jnp.mean(pos_max - pos_min)
        
        # Average spacing with some safety factor
        h = domain_size / n_per_dim * 2.0
        
        # Ensure reasonable bounds
        h = jnp.clip(h, 0.01, domain_size * 0.1)
        
        return jnp.full(self.n_particles, h)
    
    def _adaptive_smoothing_lengths(self) -> jnp.ndarray:
        """Calculate adaptive smoothing lengths with memory-efficient approach."""
        # Use chunked processing for memory efficiency
        chunk_size = min(500, self.n_particles)  # Process in smaller chunks
        smoothing_lengths = []
        
        for start_idx in range(0, self.n_particles, chunk_size):
            end_idx = min(start_idx + chunk_size, self.n_particles)
            chunk_indices = jnp.arange(start_idx, end_idx)
            
            # Process this chunk
            def get_smoothing_length_chunk(i):
                pos_i = self.positions_jax[i]
                distances = jnp.linalg.norm(self.positions_jax - pos_i[None, :], axis=1)
                # Use argpartition for memory efficiency instead of full sort
                neighbor_indices = jnp.argpartition(distances, self.n_neighbors)
                kth_distance = distances[neighbor_indices[self.n_neighbors]]
                return kth_distance * 1.2  # Scale factor
            
            chunk_lengths = vmap(get_smoothing_length_chunk)(chunk_indices)
            smoothing_lengths.append(chunk_lengths)
        
        return jnp.concatenate(smoothing_lengths)
    
    def _kernel_function(self, r: jnp.ndarray, h: float) -> jnp.ndarray:
        """SPH kernel function."""
        q = r / jnp.maximum(h, 1e-10)  # Avoid division by zero
        
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
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def _evaluate_density(self, eval_points: jnp.ndarray) -> jnp.ndarray:
        """Evaluate SPH density at given points with memory optimization."""
        def single_point_density(point):
            # Calculate distances to all particles
            distances = jnp.linalg.norm(self.positions_jax - point[None, :], axis=1)
            
            # For memory efficiency, only consider particles within reasonable distance
            # Use average smoothing length to filter
            avg_h = jnp.mean(self.smoothing_lengths)
            cutoff_distance = avg_h * 3.0  # 3x smoothing length cutoff
            
            # Filter particles within cutoff
            valid_mask = distances <= cutoff_distance
            valid_indices = jnp.where(valid_mask, jnp.arange(self.n_particles), -1)
            valid_indices = valid_indices[valid_indices >= 0]
            
            # If too few particles, use all
            if len(valid_indices) < 10:
                valid_indices = jnp.arange(self.n_particles)
            
            # Calculate kernel contributions for valid particles only
            def particle_contribution(i):
                # Safe indexing
                actual_i = jnp.where(i < len(valid_indices), valid_indices[i], 0)
                h_i = self.smoothing_lengths[actual_i]
                dist_i = distances[actual_i]
                mass_i = self.masses[actual_i]
                
                kernel_val = self._kernel_function(dist_i, h_i)
                return jnp.where(i < len(valid_indices), mass_i * kernel_val, 0.0)
            
            # Process contributions
            max_contributions = min(len(valid_indices), 1000)  # Limit for memory
            contributions = vmap(particle_contribution)(jnp.arange(max_contributions))
            return jnp.sum(contributions)
        
        return vmap(single_point_density)(eval_points)
    
    def evaluate(self, grid_points: np.ndarray) -> np.ndarray:
        """Evaluate density at grid points with chunked processing for memory efficiency."""
        grid_jax = jnp.array(grid_points)
        
        # Process in chunks to avoid memory issues
        chunk_size = min(1000, len(grid_points))
        results = []
        
        for start_idx in range(0, len(grid_points), chunk_size):
            end_idx = min(start_idx + chunk_size, len(grid_points))
            chunk = grid_jax[start_idx:end_idx]
            
            try:
                chunk_result = self._evaluate_jit(chunk)
                results.append(chunk_result)
            except Exception as e:
                print(f"Warning: SPH evaluation failed for chunk {start_idx}-{end_idx}. Using fallback.")
                # Fallback: simple distance-based density
                fallback_result = self._fallback_density_evaluation(chunk)
                results.append(fallback_result)
        
        return np.array(jnp.concatenate(results))
    
    def _fallback_density_evaluation(self, eval_points: jnp.ndarray) -> jnp.ndarray:
        """Fallback density evaluation using simple distance weighting."""
        def simple_density(point):
            distances = jnp.linalg.norm(self.positions_jax - point[None, :], axis=1)
            # Simple inverse distance weighting with cutoff
            avg_h = jnp.mean(self.smoothing_lengths)
            weights = jnp.where(distances < avg_h * 2, 1.0 / (distances + 1e-6), 0.0)
            return jnp.sum(weights) / self.n_particles
        
        return vmap(simple_density)(eval_points)


class ParticleVisualizer:
    """Advanced visualization tools for particle positions, trajectories, and density analysis."""
    
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
        
        print(f"ParticleVisualizer initialized with {len(self.final_positions)} particles")
    
    def calculate_density(self,
                         positions: str = 'final',
                         plane: str = 'xy',
                         position: float = 0.0,
                         slab_thickness: float = 0.1,
                         grid_resolution: int = 100,
                         method: str = 'jax_kde',
                         normalize: bool = True,
                         **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate particle density with various methods and normalization options.
        
        Args:
            positions: Which positions to use ('initial', 'final')
            plane: Cross-section plane ('xy', 'xz', 'yz')
            position: Position along the third axis for slicing
            slab_thickness: Thickness of the slice
            grid_resolution: Resolution of the evaluation grid
            method: Density estimation method ('jax_kde', 'sph', 'seaborn')
            normalize: Whether to normalize density to be dimensionless
            **kwargs: Additional parameters for density estimation methods
            
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
        
        # Add padding
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
            
            kde_kwargs = {k: v for k, v in kwargs.items() 
                         if k in ['bandwidth', 'bandwidth_method', 'weights']}
            kde = JAXGaussianKDE(pos_2d, **kde_kwargs)
            Z_flat = kde.evaluate(grid_points)
            
        elif method == 'sph':
            if not JAX_AVAILABLE:
                raise ImportError("JAX is required for sph method")
            
            # Check if we have too many particles for SPH
            if len(pos_2d) > 10000:
                print(f"Warning: Large particle count ({len(pos_2d)}) for SPH. Consider using 'jax_kde' instead.")
                print("Proceeding with memory-optimized SPH...")
            
            sph_kwargs = {k: v for k, v in kwargs.items() 
                         if k in ['smoothing_length', 'kernel_type', 'adaptive', 'n_neighbors', 'masses']}
            
            # Set conservative defaults for large datasets
            if 'adaptive' not in sph_kwargs and len(pos_2d) > 5000:
                sph_kwargs['adaptive'] = False
                print("Using fixed smoothing length for large dataset")
            
            if 'n_neighbors' not in sph_kwargs:
                sph_kwargs['n_neighbors'] = min(32, len(pos_2d) // 10)
            
            try:
                sph = SPHDensityEstimator(pos_2d, **sph_kwargs)
                Z_flat = sph.evaluate(grid_points)
            except Exception as e:
                print(f"SPH density estimation failed: {e}")
                print("Falling back to JAX KDE...")
                # Fallback to KDE
                kde = JAXGaussianKDE(pos_2d, bandwidth=0.1)
                Z_flat = kde.evaluate(grid_points)
            
        elif method == 'seaborn':
            # Fallback to scipy
            from scipy.stats import gaussian_kde as scipy_kde
            kde = scipy_kde(pos_2d.T)
            Z_flat = kde(grid_points.T)
            
        else:
            raise ValueError("method must be 'jax_kde', 'sph', or 'seaborn'")
        
        Z = Z_flat.reshape(X.shape)
        
        # Apply normalization if requested
        if normalize:
            Z = self._normalize_density(Z, pos_2d, X, Y, method)
        
        return X, Y, Z
    
    def _normalize_density(self, density: np.ndarray, 
                          particle_positions: np.ndarray,
                          X: np.ndarray, Y: np.ndarray,
                          method: str) -> np.ndarray:
        """
        Normalize density to be dimensionless.
        
        Args:
            density: Raw density values
            particle_positions: 2D particle positions used for calculation
            X, Y: Coordinate meshgrids
            method: Density estimation method used
            
        Returns:
            Normalized density array
        """
        n_particles = len(particle_positions)
        
        # Calculate grid cell area for normalization
        dx = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
        dy = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0
        cell_area = dx * dy
        
        # Calculate total domain area
        domain_area = (X.max() - X.min()) * (Y.max() - Y.min())
        
        if method in ['jax_kde', 'seaborn']:
            # For KDE: normalize so integral over domain equals number of particles
            # This makes density represent "particles per unit area"
            total_density = np.sum(density) * cell_area
            if total_density > 0:
                normalized_density = density * n_particles / total_density
            else:
                normalized_density = density
            
            # Convert to dimensionless by dividing by average density
            average_density = n_particles / domain_area
            dimensionless_density = normalized_density / average_density
            
        elif method == 'sph':
            # For SPH: already represents physical density, normalize by average
            average_density = n_particles / domain_area
            dimensionless_density = density / average_density
            
        else:
            # Fallback: simple normalization by max value
            max_density = np.max(density)
            dimensionless_density = density / max_density if max_density > 0 else density
        
        return dimensionless_density
    
    def apply_density_threshold(self, 
                               density: np.ndarray,
                               threshold_percent: float = 25.0,
                               threshold_type: str = 'percent_max') -> np.ndarray:
        """
        Apply threshold to density data, setting values below threshold to NaN.
        
        Args:
            density: Density data array
            threshold_percent: Threshold percentage (0-100)
            threshold_type: Type of threshold ('percent_max', 'percentile', 'absolute')
            
        Returns:
            Thresholded density array
        """
        density_thresh = density.copy()
        
        if threshold_type == 'percent_max':
            # Threshold as percentage of maximum density
            max_density = np.max(density[~np.isnan(density)])
            threshold_value = (threshold_percent / 100.0) * max_density
            
        elif threshold_type == 'percentile':
            # Threshold as percentile of density values
            threshold_value = np.percentile(density[~np.isnan(density)], threshold_percent)
            
        elif threshold_type == 'absolute':
            # Direct threshold value
            threshold_value = threshold_percent
            
        else:
            raise ValueError("threshold_type must be 'percent_max', 'percentile', or 'absolute'")
        
        # Apply threshold
        density_thresh[density_thresh < threshold_value] = np.nan
        
        return density_thresh
    
    def plot_3d_positions(self, 
                         show_initial: bool = True,
                         show_final: bool = True,
                         show_trajectories: bool = None,
                         n_show: int = 1000,
                         cam_view: Tuple[float, float, float] = (45, 30, 0),
                         figsize: Tuple[int, int] = (12, 10),
                         save_path: Optional[str] = None):
        """
        Plot 3D particle positions and trajectories.
        
        Args:
            show_initial: Whether to show initial positions
            show_final: Whether to show final positions
            show_trajectories: Whether to show trajectory lines (if available)
            n_show: Maximum number of particles to show
            cam_view: Camera view parameters (azim, elev, roll)
            figsize: Figure size
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=figsize)
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
                      c='green', s=5, label='Initial', alpha=0.7)
        
        # Plot final positions
        if show_final:
            final_pos = self.final_positions[indices]
            ax.scatter(final_pos[:, 0], final_pos[:, 1], final_pos[:, 2], 
                      c='red', s=5, label='Final', alpha=0.7)
        
        # Set camera view
        ax.view_init(azim=cam_view[0], elev=cam_view[1], roll=cam_view[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        
        title_parts = []
        if show_initial: title_parts.append("Initial")
        if show_final: title_parts.append("Final")
        if show_trajectories and self.trajectories is not None: 
            title_parts.append("Trajectories")
        
        ax.set_title(f'Particle {" & ".join(title_parts)} ({len(indices)} particles)')

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
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive plots")
        
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
            title=f'Interactive Particle Visualization ({len(indices)} particles)',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=900,
            height=700
        )
        
        fig.show()
    
    def plot_displacement_analysis(self, 
                                  figsize: Tuple[int, int] = (15, 12),
                                  save_path: Optional[str] = None):
        """
        Analyze and plot particle displacements.
        
        Args:
            figsize: Figure size
            save_path: Path to save figure
        """
        if self.initial_positions is None:
            raise ValueError("Initial positions required for displacement analysis")
        
        # Calculate displacements
        displacement = self.final_positions - self.initial_positions
        displacement_magnitude = np.linalg.norm(displacement, axis=1)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
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
        
        # 2D displacement vectors (sample)
        ax = axes[1, 0]
        n_sample = min(1000, len(displacement))
        sample_idx = np.random.choice(len(displacement), n_sample, replace=False)
        
        ax.scatter(self.initial_positions[sample_idx, 0], 
                  self.initial_positions[sample_idx, 1], 
                  c='green', s=10, alpha=0.6, label='Initial')
        ax.scatter(self.final_positions[sample_idx, 0], 
                  self.final_positions[sample_idx, 1], 
                  c='red', s=10, alpha=0.6, label='Final')
        
        # Add displacement vectors (subset)
        vector_sample = sample_idx[:min(100, len(sample_idx))]
        for i in vector_sample:
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
                           figsize: Tuple[int, int] = (12, 6),
                           save_path: Optional[str] = None):
        """
        Plot cross-sections of particle positions.
        
        Args:
            plane: Cross-section plane ('xy', 'xz', 'yz')
            position: Position along the third axis for slicing
            slab_thickness: Thickness of the slice
            show_initial: Whether to show initial positions
            show_final: Whether to show final positions
            figsize: Figure size
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
        
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
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
            ax.set_aspect('equal')
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
            ax.set_aspect('equal')
            plt.colorbar(scatter, ax=ax, label=label3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_density(self,
                    positions: str = 'final',
                    plane: str = 'xy',
                    position: float = 0.0,
                    slab_thickness: float = 0.1,
                    method: str = 'jax_kde',
                    grid_resolution: int = 100,
                    levels: int = 10,
                    figsize: Tuple[int, int] = (10, 8),
                    cmap: str = 'viridis',
                    normalize: bool = True,
                    threshold_percent: Optional[float] = None,
                    threshold_type: str = 'percent_max',
                    save_path: Optional[str] = None,
                    **kwargs):
        """
        Plot density of particle positions with normalization and threshold options.
        
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
            normalize: Whether to normalize density to be dimensionless
            threshold_percent: Threshold percentage (None for no threshold)
            threshold_type: Type of threshold ('percent_max', 'percentile', 'absolute')
            save_path: Path to save figure
            **kwargs: Additional parameters for density estimation methods
        """
        # Calculate density
        X, Y, Z = self.calculate_density(
            positions=positions,
            plane=plane,
            position=position,
            slab_thickness=slab_thickness,
            grid_resolution=grid_resolution,
            method=method,
            normalize=normalize,
            **kwargs
        )
        
        # Apply threshold if specified
        if threshold_percent is not None:
            Z = self.apply_density_threshold(Z, threshold_percent, threshold_type)
        
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
        
        # Colorbar label
        density_label = "Normalized Density" if normalize else "Density"
        if threshold_percent is not None:
            density_label += f" (>{threshold_percent}% threshold)"
        
        plt.colorbar(contourf, ax=ax, label=density_label)
        
        # Add contour lines
        contour = ax.contour(X, Y, Z, levels=levels//2, colors='black', alpha=0.3, linewidths=0.5)
        
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        
        # Create title
        method_title = method.upper().replace('_', ' ')
        norm_title = "Normalized " if normalize else ""
        title = f'{norm_title}{method_title} Density - {plane.upper()} plane at {label3}={position:.3f}'
        if threshold_percent is not None:
            title += f' (>{threshold_percent}% threshold)'
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_density_with_threshold(self,
                                   positions: str = 'final',
                                   plane: str = 'xy',
                                   position: float = 0.0,
                                   slab_thickness: float = 0.1,
                                   method: str = 'jax_kde',
                                   grid_resolution: int = 100,
                                   threshold_percent: float = 25.0,
                                   threshold_type: str = 'percent_max',
                                   figsize: Tuple[int, int] = (10, 8),
                                   cmap: str = 'viridis',
                                   normalize: bool = True,
                                   show_particles: bool = True,
                                   save_path: Optional[str] = None,
                                   **kwargs):
        """
        Plot density field with threshold cutoff, showing blank areas below threshold.
        
        Args:
            positions: Which positions to use ('initial', 'final')
            plane: Cross-section plane ('xy', 'xz', 'yz')
            position: Position along the third axis for slicing
            slab_thickness: Thickness of the slice
            method: Density estimation method
            grid_resolution: Resolution of the evaluation grid
            threshold_percent: Threshold percentage (default 25% of max density)
            threshold_type: Type of threshold ('percent_max', 'percentile', 'absolute')
            figsize: Figure size
            cmap: Colormap
            normalize: Whether to normalize density
            show_particles: Whether to overlay particle positions
            save_path: Path to save figure
            **kwargs: Additional parameters for density estimation
        """
        # Calculate density
        X, Y, Z = self.calculate_density(
            positions=positions,
            plane=plane,
            position=position,
            slab_thickness=slab_thickness,
            grid_resolution=grid_resolution,
            method=method,
            normalize=normalize,
            **kwargs
        )
        
        # Apply threshold
        Z_thresh = self.apply_density_threshold(Z, threshold_percent, threshold_type)
        
        # Get plane labels and axis mapping
        plane_maps = {
            'xy': (0, 1, 2, 'X', 'Y', 'Z'),
            'xz': (0, 2, 1, 'X', 'Z', 'Y'),
            'yz': (1, 2, 0, 'Y', 'Z', 'X')
        }
        axis1, axis2, axis3, label1, label2, label3 = plane_maps[plane]
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Plot thresholded density
        im = ax.imshow(Z_thresh, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                      origin='lower', cmap=cmap, aspect='equal')
        
        # Colorbar
        density_label = "Normalized Density" if normalize else "Density"
        density_label += f" (>{threshold_percent}% threshold)"
        cbar = plt.colorbar(im, ax=ax, label=density_label)
        
        # Show particle positions if requested
        if show_particles:
            # Get positions
            if positions == 'final':
                pos_3d = self.final_positions
            else:
                pos_3d = self.initial_positions
            
            # Filter particles in slab
            mask = np.abs(pos_3d[:, axis3] - position) <= slab_thickness / 2
            pos_slab = pos_3d[mask]
            
            if len(pos_slab) > 0:
                # Sample particles for performance
                if len(pos_slab) > 2000:
                    sample_idx = np.random.choice(len(pos_slab), 2000, replace=False)
                    pos_slab = pos_slab[sample_idx]
                
                ax.scatter(pos_slab[:, axis1], pos_slab[:, axis2], 
                         c='white', s=0.5, alpha=0.7, edgecolors='black', linewidths=0.1)
        
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        
        # Title
        method_title = method.upper().replace('_', ' ')
        norm_title = "Normalized " if normalize else ""
        title = f'{norm_title}{method_title} Density (>{threshold_percent}% Threshold)'
        title += f' - {plane.upper()} at {label3}={position:.3f}'
        ax.set_title(title)
        
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_combined_analysis(self,
                              plane: str = 'xy',
                              position: float = 0.0,
                              slab_thickness: float = 0.1,
                              method: str = 'jax_kde',
                              grid_resolution: int = 100,
                              figsize: Tuple[int, int] = (15, 12),
                              save_path: Optional[str] = None,
                              **kwargs):
        """
        Combined analysis plot with positions, displacements, and density estimation.
        
        Args:
            plane: Cross-section plane ('xy', 'xz', 'yz')
            position: Position along the third axis for slicing
            slab_thickness: Thickness of the slice
            method: Density estimation method ('jax_kde', 'sph', 'seaborn')
            grid_resolution: Resolution of the evaluation grid
            figsize: Figure size
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
        
        # Calculate density
        X, Y, Z = self.calculate_density(
            positions='final',
            plane=plane,
            position=position,
            slab_thickness=slab_thickness,
            grid_resolution=grid_resolution,
            method=method,
            **kwargs
        )
        
        # Setup figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
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
        contour = ax.contour(X, Y, Z, levels=8, alpha=0.7, linewidths=1.0)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.3f')
        ax.scatter(final_slab[:, axis1], final_slab[:, axis2], 
                  c='black', alpha=0.1, s=1)
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        plt.colorbar(contour, ax=ax, shrink=0.8)
        ax.set_title(f'{method.upper()} Density Contours')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.suptitle(f'Particle Analysis ({method.upper()}) - {plane.upper()} plane at {label3}={position:.3f}', 
                     fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_interactive_density(self,
                                positions: str = 'final',
                                plane: str = 'xy',
                                position: float = 0.0,
                                slab_thickness: float = 0.1,
                                method: str = 'jax_kde',
                                grid_resolution: int = 100,
                                colorscale: str = 'Plasma',
                                normalize: bool = True,
                                threshold_percent: Optional[float] = None,
                                threshold_type: str = 'percent_max',
                                show_particles: bool = True,
                                **kwargs):
        """
        Create interactive density plot using Plotly with normalization and threshold support.
        
        Args:
            positions: Which positions to use ('initial', 'final')
            plane: Cross-section plane ('xy', 'xz', 'yz')
            position: Position along the third axis for slicing
            slab_thickness: Thickness of the slice
            method: Density estimation method
            grid_resolution: Resolution of the evaluation grid
            colorscale: Plotly colorscale name
            normalize: Whether to normalize density
            threshold_percent: Threshold percentage for cutoff (None for no threshold)
            threshold_type: Type of threshold ('percent_max', 'percentile', 'absolute')
            show_particles: Whether to show particle overlay
            **kwargs: Additional parameters for density estimation
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive density plots")
        
        # Calculate density
        X, Y, Z = self.calculate_density(
            positions=positions,
            plane=plane,
            position=position,
            slab_thickness=slab_thickness,
            grid_resolution=grid_resolution,
            method=method,
            normalize=normalize,
            **kwargs
        )
        
        # Apply threshold if specified
        if threshold_percent is not None:
            Z_plot = self.apply_density_threshold(Z, threshold_percent, threshold_type)
        else:
            Z_plot = Z
        
        # Get plane labels
        plane_maps = {
            'xy': ('X', 'Y', 'Z'),
            'xz': ('X', 'Z', 'Y'),
            'yz': ('Y', 'Z', 'X')
        }
        label1, label2, label3 = plane_maps[plane]
        
        # Create figure
        fig = go.Figure()
        
        # Colorbar title
        density_label = "Normalized Density" if normalize else "Density"
        if threshold_percent is not None:
            density_label += f" (>{threshold_percent}% threshold)"
        
        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=Z_plot,
            x=X[0, :],
            y=Y[:, 0],
            colorscale=colorscale,
            name='Density',
            hoverongaps=False,
            colorbar=dict(title=density_label)
        ))
        
        # Add particles if requested
        if show_particles:
            plane_map = {'xy': (0, 1, 2), 'xz': (0, 2, 1), 'yz': (1, 2, 0)}
            axis1, axis2, axis3 = plane_map[plane]
            
            if positions == 'final':
                pos_3d = self.final_positions
            else:
                pos_3d = self.initial_positions
                
            mask = np.abs(pos_3d[:, axis3] - position) <= slab_thickness / 2
            pos_slab = pos_3d[mask]
            
            # Sample particles for performance
            if len(pos_slab) > 1000:
                sample_idx = np.random.choice(len(pos_slab), 1000, replace=False)
                pos_slab = pos_slab[sample_idx]
            
            fig.add_trace(go.Scatter(
                x=pos_slab[:, axis1],
                y=pos_slab[:, axis2],
                mode='markers',
                marker=dict(color='black', size=2, opacity=0.5),
                name='Particles'
            ))
        
        # Update layout
        method_title = method.upper().replace('_', ' ')
        norm_title = "Normalized " if normalize else ""
        title = f'{norm_title}{method_title} Density - {plane.upper()} plane at {label3}={position:.3f}'
        if threshold_percent is not None:
            title += f' (>{threshold_percent}% threshold)'
        
        fig.update_layout(
            title=title,
            xaxis_title=label1,
            yaxis_title=label2,
            width=800,
            height=600
        )
        
        fig.show()
    
    @staticmethod
    def create_density_colormap(base_color: str = 'viridis', 
                               n_colors: int = 256,
                               alpha_min: float = 0.0,
                               alpha_max: float = 1.0) -> LinearSegmentedColormap:
        """
        Create a custom colormap for density plots with transparency.
        
        Args:
            base_color: Base colormap name or single color
            n_colors: Number of colors in the colormap
            alpha_min: Minimum alpha (transparency) value
            alpha_max: Maximum alpha value
            
        Returns:
            Custom colormap with transparency gradient
        """
        if isinstance(base_color, str) and base_color in plt.colormaps():
            # Use existing colormap as base
            base_cmap = plt.get_cmap(base_color)
            colors = base_cmap(np.linspace(0, 1, n_colors))
        else:
            # Create colormap from single color
            if isinstance(base_color, str):
                # Convert color name to RGB
                base_rgb = mcolors.to_rgb(base_color)
            else:
                base_rgb = base_color
            
            # Create gradient from white to base color
            colors = np.zeros((n_colors, 4))
            for i in range(n_colors):
                t = i / (n_colors - 1)
                # Interpolate from white to base color
                colors[i, :3] = (1 - t) * np.array([1, 1, 1]) + t * np.array(base_rgb)
                colors[i, 3] = 1.0  # Full alpha initially
        
        # Apply alpha gradient
        alphas = np.linspace(alpha_min, alpha_max, n_colors)
        colors[:, 3] = alphas
        
        return ListedColormap(colors)
    
    @staticmethod
    def apply_density_threshold(Z: np.ndarray, 
                               threshold: Optional[float] = None,
                               threshold_percentile: Optional[float] = None) -> np.ndarray:
        """
        Apply threshold to density data, setting values below threshold to NaN.
        
        Args:
            Z: Density data array
            threshold: Absolute threshold value
            threshold_percentile: Percentile threshold (0-100)
            
        Returns:
            Thresholded density array
        """
        Z_thresh = Z.copy()
        
        if threshold is not None:
            Z_thresh[Z_thresh < threshold] = np.nan
        elif threshold_percentile is not None:
            threshold_val = np.percentile(Z_thresh[~np.isnan(Z_thresh)], threshold_percentile)
            Z_thresh[Z_thresh < threshold_val] = np.nan
        
        return Z_thresh
    
    def get_visualization_info(self) -> Dict:
        """
        Get comprehensive information about the visualizer state.
        
        Returns:
            Dictionary containing visualizer information
        """
        info = {
            'n_particles': len(self.final_positions),
            'has_initial_positions': self.initial_positions is not None,
            'has_trajectories': self.trajectories is not None,
            'final_position_bounds': {
                'x': (float(np.min(self.final_positions[:, 0])), float(np.max(self.final_positions[:, 0]))),
                'y': (float(np.min(self.final_positions[:, 1])), float(np.max(self.final_positions[:, 1]))),
                'z': (float(np.min(self.final_positions[:, 2])), float(np.max(self.final_positions[:, 2])))
            },
            'jax_available': JAX_AVAILABLE,
            'plotly_available': PLOTLY_AVAILABLE
        }
        
        if self.initial_positions is not None:
            info['initial_position_bounds'] = {
                'x': (float(np.min(self.initial_positions[:, 0])), float(np.max(self.initial_positions[:, 0]))),
                'y': (float(np.min(self.initial_positions[:, 1])), float(np.max(self.initial_positions[:, 1]))),
                'z': (float(np.min(self.initial_positions[:, 2])), float(np.max(self.initial_positions[:, 2])))
            }
        
        if self.trajectories is not None:
            info['trajectory_shape'] = self.trajectories.shape
            info['n_timesteps'] = self.trajectories.shape[1]
        
        return info
    
    def export_to_vtk(self,
                     output_directory: str = "vtk_output",
                     base_filename: str = "particle_results",
                     include_density: bool = True,
                     density_params: Optional[Dict] = None,
                     time_value: Optional[float] = None) -> Dict[str, str]:
        """
        Export particle tracking results to VTK format for ParaView.
        
        Args:
            output_directory: Directory to save VTK files
            base_filename: Base filename for all outputs
            include_density: Whether to calculate and export density field
            density_params: Parameters for density calculation
            time_value: Time value for the final state
            
        Returns:
            Dictionary mapping data type to saved file path
        """
        from .utils import VTKWriter
        
        # Create VTK writer
        writer = VTKWriter(output_directory)
        
        # Prepare density data if requested
        density_data = None
        if include_density:
            if density_params is None:
                density_params = {
                    'plane': 'xy',
                    'position': 0.0,
                    'slab_thickness': 0.5,
                    'method': 'jax_kde' if JAX_AVAILABLE else 'seaborn',
                    'grid_resolution': 100
                }
            
            try:
                X, Y, Z = self.calculate_density(
                    positions='final',
                    **density_params
                )
                
                density_data = {
                    'X': X,
                    'Y': Y, 
                    'density': Z,
                    'plane': density_params.get('plane', 'xy'),
                    'position': density_params.get('position', 0.0)
                }
                print("Calculated density field for export")
                
            except Exception as e:
                print(f"Warning: Could not calculate density field: {e}")
                include_density = False
        
        # Save all results
        saved_files = writer.save_combined_results(
            final_positions=self.final_positions,
            density_data=density_data,
            trajectories=self.trajectories,
            initial_positions=self.initial_positions,
            base_filename=base_filename,
            time_value=time_value
        )
        
        # Create ParaView state file for easy loading
        state_file = writer.create_paraview_state_file(saved_files)
        saved_files['paraview_state'] = state_file
        
        print(f"\nVTK Export Summary:")
        print(f"Output directory: {output_directory}")
        print(f"Files created:")
        for data_type, filepath in saved_files.items():
            print(f"  {data_type}: {os.path.basename(filepath)}")
        
        print(f"\nTo load in ParaView:")
        print(f"1. Open ParaView")
        print(f"2. File -> Load State -> {os.path.basename(state_file)}")
        print(f"3. Or manually load individual .vtp/.vti files")
        
    def compare_density_methods(self,
                               plane: str = 'xy',
                               position: float = 0.0,
                               slab_thickness: float = 0.1,
                               methods: List[str] = ['jax_kde', 'sph', 'seaborn'],
                               figsize: Tuple[int, int] = (18, 6),
                               normalize: bool = True,
                               threshold_percent: Optional[float] = None,
                               save_path: Optional[str] = None,
                               **kwargs):
        """
        Compare different density estimation methods side by side with normalization.
        
        Args:
            plane: Cross-section plane ('xy', 'xz', 'yz')
            position: Position along the third axis for slicing
            slab_thickness: Thickness of the slice
            methods: List of methods to compare
            figsize: Figure size
            normalize: Whether to normalize density
            threshold_percent: Threshold percentage (None for no threshold)
            save_path: Path to save figure
            **kwargs: Additional parameters for density estimation
        """
        n_methods = len(methods)
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)
        if n_methods == 1:
            axes = [axes]
        
        for i, method in enumerate(methods):
            try:
                X, Y, Z = self.calculate_density(
                    plane=plane,
                    position=position,
                    slab_thickness=slab_thickness,
                    method=method,
                    normalize=normalize,
                    **kwargs
                )
                
                # Apply threshold if specified
                if threshold_percent is not None:
                    Z = self.apply_density_threshold(Z, threshold_percent, 'percent_max')
                
                ax = axes[i]
                contourf = ax.contourf(X, Y, Z, levels=15, cmap='viridis', alpha=0.8)
                
                # Colorbar label
                density_label = "Normalized Density" if normalize else "Density"
                if threshold_percent is not None:
                    density_label += f" (>{threshold_percent}%)"
                
                plt.colorbar(contourf, ax=ax, label=density_label)
                
                method_title = method.upper().replace('_', ' ')
                norm_title = "Normalized " if normalize else ""
                title = f'{norm_title}{method_title}'
                if threshold_percent is not None:
                    title += f' (>{threshold_percent}%)'
                
                ax.set_title(title)
                ax.set_xlabel('X' if plane == 'xy' else ('X' if plane == 'xz' else 'Y'))
                ax.set_ylabel('Y' if plane == 'xy' else ('Z' if plane == 'xz' else 'Z'))
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error with {method}:\n{str(e)}', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{method.upper()} (Error)')
        
        suptitle = f'Density Method Comparison - {plane.upper()} plane at position={position:.3f}'
        if normalize:
            suptitle = "Normalized " + suptitle
        plt.suptitle(suptitle)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_density_threshold_comparison(self,
                                        plane: str = 'xy',
                                        position: float = 0.0,
                                        slab_thickness: float = 0.1,
                                        method: str = 'jax_kde',
                                        thresholds: List[float] = [10, 25, 50],
                                        figsize: Tuple[int, int] = (18, 6),
                                        normalize: bool = True,
                                        save_path: Optional[str] = None,
                                        **kwargs):
        """
        Compare different threshold levels for density visualization.
        
        Args:
            plane: Cross-section plane ('xy', 'xz', 'yz')
            position: Position along the third axis for slicing
            slab_thickness: Thickness of the slice
            method: Density estimation method
            thresholds: List of threshold percentages to compare
            figsize: Figure size
            normalize: Whether to normalize density
            save_path: Path to save figure
            **kwargs: Additional parameters for density estimation
        """
        # Calculate base density once
        X, Y, Z = self.calculate_density(
            plane=plane,
            position=position,
            slab_thickness=slab_thickness,
            method=method,
            normalize=normalize,
            **kwargs
        )
        
        n_plots = len(thresholds) + 1  # +1 for no threshold
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        
        # Plot without threshold
        ax = axes[0]
        im = ax.imshow(Z, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                      origin='lower', cmap='viridis', aspect='equal')
        
        density_label = "Normalized Density" if normalize else "Density"
        plt.colorbar(im, ax=ax, label=density_label)
        ax.set_title('No Threshold')
        ax.set_xlabel('X' if plane == 'xy' else ('X' if plane == 'xz' else 'Y'))
        ax.set_ylabel('Y' if plane == 'xy' else ('Z' if plane == 'xz' else 'Z'))
        ax.grid(True, alpha=0.3)
        
        # Plot with different thresholds
        for i, threshold in enumerate(thresholds):
            ax = axes[i + 1]
            Z_thresh = self.apply_density_threshold(Z, threshold, 'percent_max')
            
            im = ax.imshow(Z_thresh, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                          origin='lower', cmap='viridis', aspect='equal')
            
            thresh_label = f"Density (>{threshold}%)"
            plt.colorbar(im, ax=ax, label=thresh_label)
            ax.set_title(f'>{threshold}% Threshold')
            ax.set_xlabel('X' if plane == 'xy' else ('X' if plane == 'xz' else 'Y'))
            ax.set_ylabel('Y' if plane == 'xy' else ('Z' if plane == 'xz' else 'Z'))
            ax.grid(True, alpha=0.3)
        
        method_title = method.upper().replace('_', ' ')
        norm_title = "Normalized " if normalize else ""
        suptitle = f'{norm_title}{method_title} Density Threshold Comparison - {plane.upper()} plane'
        plt.suptitle(suptitle)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def __repr__(self) -> str:
        """String representation of the ParticleVisualizer."""
        parts = [f"ParticleVisualizer(n_particles={len(self.final_positions)}"]
        
        if self.initial_positions is not None:
            parts.append("has_initial=True")
        if self.trajectories is not None:
            parts.append(f"has_trajectories=True, n_steps={self.trajectories.shape[1]}")
        
        return ", ".join(parts) + ")"


# Convenience functions for quick visualization (kept for backward compatibility)
def quick_visualize_positions(final_positions: np.ndarray,
                             initial_positions: Optional[np.ndarray] = None,
                             trajectories: Optional[np.ndarray] = None,
                             mode: str = '3d',
                             **kwargs):
    """
    Quick visualization function for particle positions.
    
    Args:
        final_positions: Final particle positions (N, 3)
        initial_positions: Initial particle positions (N, 3), optional
        trajectories: Full trajectories (N, n_steps, 3), optional
        mode: Visualization mode ('3d', 'cross_section', 'density', 'analysis')
        **kwargs: Additional arguments for the visualization method
    """
    viz = ParticleVisualizer(final_positions, initial_positions, trajectories)
    
    if mode == '3d':
        viz.plot_3d_positions(**kwargs)
    elif mode == 'interactive_3d':
        viz.plot_interactive_3d(**kwargs)
    elif mode == 'cross_section':
        viz.plot_cross_sections(**kwargs)
    elif mode == 'density':
        viz.plot_density(**kwargs)
    elif mode == 'analysis':
        viz.plot_combined_analysis(**kwargs)
    elif mode == 'displacement':
        viz.plot_displacement_analysis(**kwargs)
    elif mode == 'density_threshold':
        viz.plot_density_with_threshold(**kwargs)
    elif mode == 'compare_methods':
        viz.compare_density_methods(**kwargs)
    elif mode == 'compare_thresholds':
        viz.plot_density_threshold_comparison(**kwargs)
    else:
        raise ValueError("mode must be one of: '3d', 'interactive_3d', 'cross_section', 'density', "
                        "'analysis', 'displacement', 'density_threshold', 'compare_methods', 'compare_thresholds'")


def compare_density_methods(final_positions: np.ndarray, **kwargs):
    """
    Convenience function for comparing density methods.
    
    Args:
        final_positions: Final particle positions (N, 3)
        **kwargs: Arguments passed to ParticleVisualizer.compare_density_methods()
    """
    viz = ParticleVisualizer(final_positions=final_positions)
    viz.compare_density_methods(**kwargs)


def plot_density_threshold_comparison(final_positions: np.ndarray, **kwargs):
    """
    Convenience function for comparing density thresholds.
    
    Args:
        final_positions: Final particle positions (N, 3)
        **kwargs: Arguments passed to ParticleVisualizer.plot_density_threshold_comparison()
    """
    viz = ParticleVisualizer(final_positions=final_positions)
    viz.plot_density_threshold_comparison(**kwargs)