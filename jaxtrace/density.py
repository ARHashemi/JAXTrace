"""
Density Estimation Module for JAXTrace

Advanced density estimation methods including KDE and SPH for 2D and 3D particle data.
"""

import numpy as np
from typing import Optional, Union, Tuple, Dict, List
import warnings

# JAX imports for advanced density estimation
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not available. Advanced density estimation will use fallback methods.")


class BaseDensityEstimator:
    """Base class for density estimation methods."""
    
    def __init__(self, 
                 positions: np.ndarray,
                 dimensions: str = '2d',
                 plane: str = 'xy',
                 position: float = 0.0,
                 slab_thickness: float = 0.1,
                 **kwargs):
        """
        Initialize base density estimator.
        
        Args:
            positions: Particle positions (N, 3) for 3D or (N, 2) for 2D
            dimensions: '2d' or '3d' density calculation
            plane: Cross-section plane for 2D ('xy', 'xz', 'yz')
            position: Position along the third axis for 2D slicing
            slab_thickness: Thickness of the slice for 2D
            **kwargs: Additional parameters
        """
        self.original_positions = np.array(positions)
        self.dimensions = dimensions.lower()
        self.plane = plane.lower()
        self.position = position
        self.slab_thickness = slab_thickness
        
        if self.dimensions not in ['2d', '3d']:
            raise ValueError("dimensions must be '2d' or '3d'")
        
        # Process positions based on dimensions
        if self.dimensions == '2d':
            self.positions = self._extract_2d_positions()
        else:
            self.positions = self.original_positions
            if self.positions.shape[1] != 3:
                raise ValueError("3D density calculation requires 3D positions (N, 3)")
        
        self.n_particles, self.n_dims = self.positions.shape
        
    def _extract_2d_positions(self) -> np.ndarray:
        """Extract 2D positions from 3D data based on plane and slab."""
        if self.original_positions.shape[1] != 3:
            raise ValueError("2D extraction requires 3D positions (N, 3)")
        
        # Define plane mapping
        plane_maps = {
            'xy': (0, 1, 2),
            'xz': (0, 2, 1),
            'yz': (1, 2, 0)
        }
        
        if self.plane not in plane_maps:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")
        
        axis1, axis2, axis3 = plane_maps[self.plane]
        
        # Filter particles in slab
        mask = np.abs(self.original_positions[:, axis3] - self.position) <= self.slab_thickness / 2
        pos_slab = self.original_positions[mask]
        
        if len(pos_slab) == 0:
            raise ValueError("No particles found in the specified slab")
        
        print(f"Extracted {len(pos_slab)} particles from {self.plane} plane at {self.position}")
        
        # Extract 2D positions
        return pos_slab[:, [axis1, axis2]]
    
    def evaluate(self, eval_points: np.ndarray) -> np.ndarray:
        """Evaluate density at given points."""
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def create_evaluation_grid(self, resolution: Union[int, Tuple[int, ...]], 
                              bounds: Optional[Tuple] = None,
                              padding: float = 0.1) -> Union[Tuple[np.ndarray, np.ndarray], 
                                                           Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create evaluation grid for density calculation.
        
        Args:
            resolution: Grid resolution (int for uniform, tuple for per-dimension)
            bounds: Custom bounds (for 2D: (xmin, xmax, ymin, ymax), for 3D: (xmin, xmax, ymin, ymax, zmin, zmax))
            padding: Padding factor for automatic bounds
            
        Returns:
            For 2D: (X, Y) meshgrids
            For 3D: (X, Y, Z) meshgrids
        """
        if self.dimensions == '2d':
            return self._create_2d_grid(resolution, bounds, padding)
        else:
            return self._create_3d_grid(resolution, bounds, padding)
    
    def _create_2d_grid(self, resolution: Union[int, Tuple[int, int]], 
                       bounds: Optional[Tuple], padding: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create 2D evaluation grid."""
        if isinstance(resolution, int):
            nx = ny = resolution
        else:
            nx, ny = resolution
        
        # Get bounds
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
        else:
            xmin, xmax = self.positions[:, 0].min(), self.positions[:, 0].max()
            ymin, ymax = self.positions[:, 1].min(), self.positions[:, 1].max()
            
            # Add padding
            x_range = xmax - xmin
            y_range = ymax - ymin
            xmin -= padding * x_range
            xmax += padding * x_range
            ymin -= padding * y_range
            ymax += padding * y_range
        
        x_grid = np.linspace(xmin, xmax, nx)
        y_grid = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        return X, Y
    
    def _create_3d_grid(self, resolution: Union[int, Tuple[int, int, int]], 
                       bounds: Optional[Tuple], padding: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create 3D evaluation grid."""
        if isinstance(resolution, int):
            nx = ny = nz = resolution
        else:
            nx, ny, nz = resolution
        
        # Get bounds
        if bounds is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = bounds
        else:
            xmin, xmax = self.positions[:, 0].min(), self.positions[:, 0].max()
            ymin, ymax = self.positions[:, 1].min(), self.positions[:, 1].max()
            zmin, zmax = self.positions[:, 2].min(), self.positions[:, 2].max()
            
            # Add padding
            x_range = xmax - xmin
            y_range = ymax - ymin
            z_range = zmax - zmin
            xmin -= padding * x_range
            xmax += padding * x_range
            ymin -= padding * y_range
            ymax += padding * y_range
            zmin -= padding * z_range
            zmax += padding * z_range
        
        x_grid = np.linspace(xmin, xmax, nx)
        y_grid = np.linspace(ymin, ymax, ny)
        z_grid = np.linspace(zmin, zmax, nz)
        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        
        return X, Y, Z


class JAXGaussianKDE(BaseDensityEstimator):
    """JAX-based Gaussian KDE estimator for high performance 2D and 3D density estimation."""
    
    def __init__(self, positions: np.ndarray,
                 dimensions: str = '2d',
                 plane: str = 'xy',
                 position: float = 0.0,
                 slab_thickness: float = 0.1,
                 bandwidth: Optional[float] = None,
                 bandwidth_method: str = 'scott',
                 weights: Optional[np.ndarray] = None,
                 **kwargs):
        """
        Initialize JAX Gaussian KDE.
        
        Args:
            positions: Particle positions (N, 3) for 3D or (N, 2) for 2D
            dimensions: '2d' or '3d' density calculation
            plane: Cross-section plane for 2D ('xy', 'xz', 'yz')
            position: Position along the third axis for 2D slicing
            slab_thickness: Thickness of the slice for 2D
            bandwidth: Fixed bandwidth value (overrides bandwidth_method)
            bandwidth_method: Method for bandwidth selection ('scott', 'silverman')
            weights: Particle weights (N,), optional
            **kwargs: Additional parameters
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for JAXGaussianKDE")
        
        super().__init__(positions, dimensions, plane, position, slab_thickness, **kwargs)
        
        self.positions_jax = jnp.array(self.positions)
        self.weights = jnp.array(weights) if weights is not None else None
        
        # Calculate bandwidth
        if bandwidth is not None:
            self.bandwidth = bandwidth
        else:
            self.bandwidth = self._calculate_bandwidth(bandwidth_method)
        
        print(f"JAX KDE initialized: {self.dimensions.upper()}, {self.n_particles} particles, bandwidth={self.bandwidth:.4f}")
        
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
    
    def evaluate(self, eval_points: np.ndarray) -> np.ndarray:
        """Evaluate density at evaluation points."""
        eval_jax = jnp.array(eval_points)
        
        # For 3D, process in chunks to manage memory
        if self.dimensions == '3d' and len(eval_points) > 10000:
            return self._evaluate_chunked(eval_jax)
        else:
            return np.array(self._evaluate_jit(eval_jax))
    
    def _evaluate_chunked(self, eval_points: jnp.ndarray, chunk_size: int = 5000) -> np.ndarray:
        """Evaluate density in chunks for memory efficiency."""
        results = []
        for start_idx in range(0, len(eval_points), chunk_size):
            end_idx = min(start_idx + chunk_size, len(eval_points))
            chunk = eval_points[start_idx:end_idx]
            chunk_result = self._evaluate_jit(chunk)
            results.append(chunk_result)
        return np.concatenate(results)


class SPHDensityEstimator(BaseDensityEstimator):
    """Smoothed Particle Hydrodynamics density estimator for 2D and 3D."""
    
    def __init__(self, positions: np.ndarray,
                 dimensions: str = '2d',
                 plane: str = 'xy',
                 position: float = 0.0,
                 slab_thickness: float = 0.1,
                 smoothing_length: Optional[float] = None,
                 kernel_type: str = 'cubic_spline',
                 adaptive: bool = False,
                 n_neighbors: int = 32,
                 masses: Optional[np.ndarray] = None,
                 max_particles_for_adaptive: int = 5000,
                 **kwargs):
        """
        Initialize SPH density estimator.
        
        Args:
            positions: Particle positions (N, 3) for 3D or (N, 2) for 2D
            dimensions: '2d' or '3d' density calculation
            plane: Cross-section plane for 2D ('xy', 'xz', 'yz')
            position: Position along the third axis for 2D slicing
            slab_thickness: Thickness of the slice for 2D
            smoothing_length: Fixed smoothing length
            kernel_type: Type of kernel ('cubic_spline', 'gaussian', 'wendland')
            adaptive: Whether to use adaptive smoothing lengths
            n_neighbors: Number of neighbors for adaptive smoothing
            masses: Particle masses (N,), defaults to uniform
            max_particles_for_adaptive: Maximum particles for adaptive method
            **kwargs: Additional parameters
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for SPHDensityEstimator")
        
        super().__init__(positions, dimensions, plane, position, slab_thickness, **kwargs)
        
        self.positions_jax = jnp.array(self.positions)
        self.kernel_type = kernel_type
        self.adaptive = adaptive
        self.n_neighbors = min(n_neighbors, self.n_particles - 1)
        self.max_particles_for_adaptive = max_particles_for_adaptive
        
        # Set masses
        if masses is not None:
            # Filter masses for 2D case
            if self.dimensions == '2d' and len(masses) != self.n_particles:
                # Assume masses correspond to original 3D positions
                mask = np.abs(self.original_positions[:, 2] - self.position) <= self.slab_thickness / 2
                self.masses = jnp.array(masses[mask])
            else:
                self.masses = jnp.array(masses)
        else:
            self.masses = jnp.ones(self.n_particles)
        
        # Check adaptive feasibility
        if self.adaptive and self.n_particles > self.max_particles_for_adaptive:
            print(f"Warning: Too many particles ({self.n_particles}) for adaptive SPH. Using fixed smoothing.")
            self.adaptive = False
        
        # Calculate smoothing lengths
        if smoothing_length is not None:
            self.smoothing_lengths = jnp.full(self.n_particles, smoothing_length)
        else:
            self.smoothing_lengths = self._calculate_smoothing_lengths()
        
        print(f"SPH initialized: {self.dimensions.upper()}, {self.n_particles} particles, "
              f"{'adaptive' if self.adaptive else 'fixed'} smoothing")
        
        # Compile evaluation function
        self._evaluate_jit = jit(self._evaluate_density)
    
    def _calculate_smoothing_lengths(self) -> jnp.ndarray:
        """Calculate smoothing lengths with memory optimization."""
        if self.adaptive and self.n_particles <= self.max_particles_for_adaptive:
            try:
                return self._adaptive_smoothing_lengths()
            except Exception as e:
                print(f"Warning: Adaptive smoothing failed ({e}). Using fixed smoothing.")
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
        
        # Average spacing with safety factor
        h = domain_size / n_per_dim * 2.0
        
        # Ensure reasonable bounds
        h = jnp.clip(h, 0.01, domain_size * 0.1)
        
        return jnp.full(self.n_particles, h)
    
    def _adaptive_smoothing_lengths(self) -> jnp.ndarray:
        """Calculate adaptive smoothing lengths with chunked processing."""
        chunk_size = min(500, self.n_particles)
        smoothing_lengths = []
        
        for start_idx in range(0, self.n_particles, chunk_size):
            end_idx = min(start_idx + chunk_size, self.n_particles)
            chunk_indices = jnp.arange(start_idx, end_idx)
            
            def get_smoothing_length_chunk(i):
                pos_i = self.positions_jax[i]
                distances = jnp.linalg.norm(self.positions_jax - pos_i[None, :], axis=1)
                neighbor_indices = jnp.argpartition(distances, self.n_neighbors)
                kth_distance = distances[neighbor_indices[self.n_neighbors]]
                return kth_distance * 1.2
            
            chunk_lengths = vmap(get_smoothing_length_chunk)(chunk_indices)
            smoothing_lengths.append(chunk_lengths)
        
        return jnp.concatenate(smoothing_lengths)
    
    def _kernel_function(self, r: jnp.ndarray, h: float) -> jnp.ndarray:
        """SPH kernel function with 2D/3D support."""
        q = r / jnp.maximum(h, 1e-10)
        
        if self.kernel_type == 'cubic_spline':
            if self.n_dims == 2:
                # 2D cubic spline kernel
                sigma = 10.0 / (7.0 * jnp.pi * h**2)
            else:
                # 3D cubic spline kernel
                sigma = 8.0 / (jnp.pi * h**3)
            
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
            if self.n_dims == 2:
                sigma = 1.0 / (jnp.pi * h**2)
            else:
                sigma = 1.0 / ((jnp.pi * h**2)**(3/2))
            return sigma * jnp.exp(-q**2)
            
        elif self.kernel_type == 'wendland':
            if self.n_dims == 2:
                sigma = 7.0 / (jnp.pi * h**2)
            else:
                sigma = 21.0 / (2.0 * jnp.pi * h**3)
            return jnp.where(
                q <= 1.0,
                sigma * (1.0 - q)**4 * (1.0 + 4.0 * q),
                0.0
            )
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def _evaluate_density(self, eval_points: jnp.ndarray) -> jnp.ndarray:
        """Evaluate SPH density with 2D/3D support."""
        def single_point_density(point):
            distances = jnp.linalg.norm(self.positions_jax - point[None, :], axis=1)
            
            # Use cutoff for efficiency
            avg_h = jnp.mean(self.smoothing_lengths)
            cutoff_distance = avg_h * 3.0
            
            valid_mask = distances <= cutoff_distance
            valid_indices = jnp.where(valid_mask, jnp.arange(self.n_particles), -1)
            valid_indices = valid_indices[valid_indices >= 0]
            
            if len(valid_indices) < 10:
                valid_indices = jnp.arange(self.n_particles)
            
            def particle_contribution(i):
                actual_i = jnp.where(i < len(valid_indices), valid_indices[i], 0)
                h_i = self.smoothing_lengths[actual_i]
                dist_i = distances[actual_i]
                mass_i = self.masses[actual_i]
                
                kernel_val = self._kernel_function(dist_i, h_i)
                return jnp.where(i < len(valid_indices), mass_i * kernel_val, 0.0)
            
            max_contributions = min(len(valid_indices), 1000)
            contributions = vmap(particle_contribution)(jnp.arange(max_contributions))
            return jnp.sum(contributions)
        
        return vmap(single_point_density)(eval_points)
    
    def evaluate(self, eval_points: np.ndarray) -> np.ndarray:
        """Evaluate density with chunked processing for memory efficiency."""
        eval_jax = jnp.array(eval_points)
        
        # Use chunked processing for large datasets
        chunk_size = 1000 if self.dimensions == '2d' else 500
        results = []
        
        for start_idx in range(0, len(eval_points), chunk_size):
            end_idx = min(start_idx + chunk_size, len(eval_points))
            chunk = eval_jax[start_idx:end_idx]
            
            try:
                chunk_result = self._evaluate_jit(chunk)
                results.append(chunk_result)
            except Exception as e:
                print(f"Warning: SPH evaluation failed for chunk {start_idx}-{end_idx}. Using fallback.")
                fallback_result = self._fallback_density_evaluation(chunk)
                results.append(fallback_result)
        
        return np.array(jnp.concatenate(results))
    
    def _fallback_density_evaluation(self, eval_points: jnp.ndarray) -> jnp.ndarray:
        """Fallback density evaluation using simple distance weighting."""
        def simple_density(point):
            distances = jnp.linalg.norm(self.positions_jax - point[None, :], axis=1)
            avg_h = jnp.mean(self.smoothing_lengths)
            weights = jnp.where(distances < avg_h * 2, 1.0 / (distances + 1e-6), 0.0)
            return jnp.sum(weights) / self.n_particles
        
        return vmap(simple_density)(eval_points)


class DensityCalculator:
    """High-level interface for density calculation with automatic method selection."""
    
    def __init__(self, positions: np.ndarray):
        """
        Initialize density calculator.
        
        Args:
            positions: Particle positions (N, 3) or (N, 2)
        """
        self.positions = np.array(positions)
        
    def calculate_density(self,
                         method: str = 'jax_kde',
                         dimensions: str = '2d',
                         plane: str = 'xy',
                         position: float = 0.0,
                         slab_thickness: float = 0.1,
                         resolution: Union[int, Tuple] = 100,
                         bounds: Optional[Tuple] = None,
                         normalize: bool = True,
                         **kwargs) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], 
                                          Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calculate density using specified method and parameters.
        
        Args:
            method: Density estimation method ('jax_kde', 'sph', 'scipy_kde')
            dimensions: '2d' or '3d' density calculation
            plane: Cross-section plane for 2D ('xy', 'xz', 'yz')
            position: Position along the third axis for 2D slicing
            slab_thickness: Thickness of the slice for 2D
            resolution: Grid resolution
            bounds: Custom bounds for evaluation grid
            normalize: Whether to normalize density
            **kwargs: Additional parameters for density estimation methods
            
        Returns:
            For 2D: (X, Y, density)
            For 3D: (X, Y, Z, density)
        """
        # Create density estimator
        if method == 'jax_kde':
            if not JAX_AVAILABLE:
                print("JAX not available, falling back to scipy KDE")
                method = 'scipy_kde'
            else:
                estimator = JAXGaussianKDE(
                    self.positions, dimensions, plane, position, slab_thickness, **kwargs
                )
        elif method == 'sph':
            if not JAX_AVAILABLE:
                raise ImportError("JAX is required for SPH method")
            estimator = SPHDensityEstimator(
                self.positions, dimensions, plane, position, slab_thickness, **kwargs
            )
        elif method == 'scipy_kde':
            estimator = self._create_scipy_kde_estimator(
                dimensions, plane, position, slab_thickness, **kwargs
            )
        else:
            raise ValueError("method must be 'jax_kde', 'sph', or 'scipy_kde'")
        
        # Create evaluation grid
        if dimensions == '2d':
            X, Y = estimator.create_evaluation_grid(resolution, bounds)
            grid_points = np.column_stack([X.ravel(), Y.ravel()])
            
            # Evaluate density
            density_flat = estimator.evaluate(grid_points)
            density = density_flat.reshape(X.shape)
            
            # Normalize if requested
            if normalize:
                density = self._normalize_density_2d(density, estimator.positions, X, Y, method)
            
            return X, Y, density
            
        else:  # 3D
            X, Y, Z = estimator.create_evaluation_grid(resolution, bounds)
            grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            
            # Evaluate density
            density_flat = estimator.evaluate(grid_points)
            density = density_flat.reshape(X.shape)
            
            # Normalize if requested
            if normalize:
                density = self._normalize_density_3d(density, estimator.positions, X, Y, Z, method)
            
            return X, Y, Z, density
    
    def _create_scipy_kde_estimator(self, dimensions, plane, position, slab_thickness, **kwargs):
        """Create scipy KDE estimator as fallback."""
        from scipy.stats import gaussian_kde
        
        class ScipyKDEEstimator(BaseDensityEstimator):
            def __init__(self, positions, dimensions, plane, position, slab_thickness, **kwargs):
                super().__init__(positions, dimensions, plane, position, slab_thickness, **kwargs)
                self.kde = gaussian_kde(self.positions.T)
                
            def evaluate(self, eval_points):
                return self.kde(eval_points.T)
                
            def create_evaluation_grid(self, resolution, bounds, padding=0.1):
                return super().create_evaluation_grid(resolution, bounds, padding)
        
        return ScipyKDEEstimator(self.positions, dimensions, plane, position, slab_thickness, **kwargs)
    
    def _normalize_density_2d(self, density, particle_positions, X, Y, method):
        """Normalize 2D density to be dimensionless."""
        n_particles = len(particle_positions)
        
        # Calculate grid cell area
        dx = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
        dy = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0
        cell_area = dx * dy
        
        # Calculate domain area
        domain_area = (X.max() - X.min()) * (Y.max() - Y.min())
        
        if method in ['jax_kde', 'scipy_kde']:
            # Normalize so integral equals particle count
            total_density = np.sum(density) * cell_area
            if total_density > 0:
                normalized_density = density * n_particles / total_density
            else:
                normalized_density = density
            
            # Convert to dimensionless
            average_density = n_particles / domain_area
            dimensionless_density = normalized_density / average_density
            
        elif method == 'sph':
            # SPH already represents physical density
            average_density = n_particles / domain_area
            dimensionless_density = density / average_density
            
        else:
            # Fallback normalization
            max_density = np.max(density)
            dimensionless_density = density / max_density if max_density > 0 else density
        
        return dimensionless_density
    
    def _normalize_density_3d(self, density, particle_positions, X, Y, Z, method):
        """Normalize 3D density to be dimensionless."""
        n_particles = len(particle_positions)
        
        # Calculate grid cell volume
        dx = X[1, 0, 0] - X[0, 0, 0] if X.shape[0] > 1 else 1.0
        dy = Y[0, 1, 0] - Y[0, 0, 0] if Y.shape[1] > 1 else 1.0
        dz = Z[0, 0, 1] - Z[0, 0, 0] if Z.shape[2] > 1 else 1.0
        cell_volume = dx * dy * dz
        
        # Calculate domain volume
        domain_volume = (X.max() - X.min()) * (Y.max() - Y.min()) * (Z.max() - Z.min())
        
        if method in ['jax_kde', 'scipy_kde']:
            # Normalize so integral equals particle count
            total_density = np.sum(density) * cell_volume
            if total_density > 0:
                normalized_density = density * n_particles / total_density
            else:
                normalized_density = density
            
            # Convert to dimensionless
            average_density = n_particles / domain_volume
            dimensionless_density = normalized_density / average_density
            
        elif method == 'sph':
            # SPH already represents physical density
            average_density = n_particles / domain_volume
            dimensionless_density = density / average_density
            
        else:
            # Fallback normalization
            max_density = np.max(density)
            dimensionless_density = density / max_density if max_density > 0 else density
        
        return dimensionless_density


def apply_density_threshold(density: np.ndarray,
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


def calculate_density_statistics(density: np.ndarray, 
                                thresholded_density: Optional[np.ndarray] = None,
                                threshold_percent: Optional[float] = None) -> Dict:
    """
    Calculate comprehensive statistics for density field.
    
    Args:
        density: Original density data
        thresholded_density: Density data with threshold applied
        threshold_percent: Threshold percentage used
        
    Returns:
        Dictionary with density statistics
    """
    # Remove NaN values for statistics
    valid_density = density[~np.isnan(density)]
    
    stats = {
        'max': float(np.max(valid_density)) if len(valid_density) > 0 else 0.0,
        'mean': float(np.mean(valid_density)) if len(valid_density) > 0 else 0.0,
        'std': float(np.std(valid_density)) if len(valid_density) > 0 else 0.0,
        'min': float(np.min(valid_density)) if len(valid_density) > 0 else 0.0,
        'median': float(np.median(valid_density)) if len(valid_density) > 0 else 0.0,
        'total_points': len(density.ravel()),
        'valid_points': len(valid_density),
        'nan_points': len(density.ravel()) - len(valid_density)
    }
    
    # Add percentiles
    if len(valid_density) > 0:
        stats['percentile_25'] = float(np.percentile(valid_density, 25))
        stats['percentile_75'] = float(np.percentile(valid_density, 75))
        stats['percentile_90'] = float(np.percentile(valid_density, 90))
        stats['percentile_95'] = float(np.percentile(valid_density, 95))
    
    # Add threshold statistics if provided
    if thresholded_density is not None and threshold_percent is not None:
        valid_thresh_density = thresholded_density[~np.isnan(thresholded_density)]
        stats['threshold_percent'] = threshold_percent
        stats['above_threshold_points'] = len(valid_thresh_density)
        
        if len(valid_density) > 0:
            stats['above_threshold_fraction'] = len(valid_thresh_density) / len(valid_density)
            stats['above_threshold_percentage'] = (len(valid_thresh_density) / len(valid_density)) * 100
        else:
            stats['above_threshold_fraction'] = 0.0
            stats['above_threshold_percentage'] = 0.0
    
    return stats


# Convenience functions for quick density calculation
def quick_density_2d(positions: np.ndarray,
                     plane: str = 'xy',
                     position: float = 0.0,
                     slab_thickness: float = 0.1,
                     method: str = 'jax_kde',
                     resolution: int = 100,
                     normalize: bool = True,
                     **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quick 2D density calculation.
    
    Args:
        positions: Particle positions (N, 3)
        plane: Cross-section plane ('xy', 'xz', 'yz')
        position: Position along the third axis
        slab_thickness: Thickness of the slice
        method: Density estimation method
        resolution: Grid resolution
        normalize: Whether to normalize density
        **kwargs: Additional parameters
        
    Returns:
        X, Y, density arrays
    """
    calculator = DensityCalculator(positions)
    return calculator.calculate_density(
        method=method,
        dimensions='2d',
        plane=plane,
        position=position,
        slab_thickness=slab_thickness,
        resolution=resolution,
        normalize=normalize,
        **kwargs
    )


def quick_density_3d(positions: np.ndarray,
                     method: str = 'jax_kde',
                     resolution: Union[int, Tuple[int, int, int]] = 50,
                     bounds: Optional[Tuple] = None,
                     normalize: bool = True,
                     **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Quick 3D density calculation.
    
    Args:
        positions: Particle positions (N, 3)
        method: Density estimation method
        resolution: Grid resolution (int or tuple)
        bounds: Custom bounds (xmin, xmax, ymin, ymax, zmin, zmax)
        normalize: Whether to normalize density
        **kwargs: Additional parameters
        
    Returns:
        X, Y, Z, density arrays
    """
    calculator = DensityCalculator(positions)
    return calculator.calculate_density(
        method=method,
        dimensions='3d',
        resolution=resolution,
        bounds=bounds,
        normalize=normalize,
        **kwargs
    )


# Export main classes and functions
__all__ = [
    'BaseDensityEstimator',
    'JAXGaussianKDE',
    'SPHDensityEstimator', 
    'DensityCalculator',
    'apply_density_threshold',
    'calculate_density_statistics',
    'quick_density_2d',
    'quick_density_3d'
]