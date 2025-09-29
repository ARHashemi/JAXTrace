# jaxtrace/tracking/seeding.py  
"""  
Seed position generators for particle tracking.  

Provides various seeding strategies including uniform random sampling,  
structured grids, and custom distributions with consistent float32  
data types and (N,3) shape enforcement.  
"""  

from __future__ import annotations  
from typing import Union, Tuple, List, Optional, Dict, Any, Callable  
import numpy as np  
import warnings  

# Import JAX utilities with fallback  
from ..utils.jax_utils import JAX_AVAILABLE  

if JAX_AVAILABLE:  
    try:  
        import jax  
        import jax.numpy as jnp  
    except Exception:  
        JAX_AVAILABLE = False  

if not JAX_AVAILABLE:  
    import numpy as jnp  # type: ignore  
    # Mock JAX random for NumPy fallback  
    class MockRandom:  
        def PRNGKey(self, seed):  
            np.random.seed(seed)  
            return None  
        def uniform(self, key, shape, dtype=None):  
            return np.random.uniform(0, 1, shape).astype(dtype or np.float32)  
        def normal(self, key, shape, dtype=None):  
            return np.random.normal(0, 1, shape).astype(dtype or np.float32)  
    class MockJax:  
        random = MockRandom()  
    jax = MockJax()  


def _ensure_float32(data: np.ndarray) -> np.ndarray:  
    """Convert data to float32 for consistency and JAX performance."""  
    return np.asarray(data, dtype=np.float32)  


def _ensure_seed_positions_shape(positions: np.ndarray) -> np.ndarray:  
    """  
    Ensure seed positions have shape (N, 3) with float32 dtype.  
    
    Parameters  
    ----------  
    positions : np.ndarray  
        Position data in various formats  
        
    Returns  
    -------  
    np.ndarray  
        Positions with shape (N, 3), dtype float32  
    """  
    pos = _ensure_float32(positions)  
    
    if pos.ndim == 1:  
        if len(pos) == 2:  
            # Single 2D position: (2,) -> (1, 3)  
            pos = np.array([[pos[0], pos[1], 0.0]], dtype=np.float32)  
        elif len(pos) == 3:  
            # Single 3D position: (3,) -> (1, 3)  
            pos = pos.reshape(1, 3)  
        else:  
            raise ValueError(f"1D position must have 2 or 3 elements, got {len(pos)}")  
    elif pos.ndim == 2:  
        N, D = pos.shape  
        if D == 2:  
            # 2D positions: (N, 2) -> (N, 3)  
            z_zeros = np.zeros((N, 1), dtype=np.float32)  
            pos = np.concatenate([pos, z_zeros], axis=1)  
        elif D == 3:  
            # Already correct format  
            pass  
        else:  
            raise ValueError(f"2D position array must have 2 or 3 columns, got {D}")  
    else:  
        raise ValueError(f"Position array must be 1D or 2D, got {pos.ndim}D")  
    
    return pos.astype(np.float32, copy=False)  


def _validate_bounds(bounds: np.ndarray) -> np.ndarray:  
    """  
    Validate and standardize domain bounds.  
    
    Parameters  
    ----------  
    bounds : array-like  
        Domain bounds in various formats  
        
    Returns  
    -------  
    np.ndarray  
        Standardized bounds, shape (2, 3), dtype float32  
    """  
    bounds = _ensure_float32(bounds)  
    
    if bounds.ndim == 1:  
        if len(bounds) == 4:  
            # 2D bounds: [x_min, x_max, y_min, y_max] -> [[x_min, y_min, 0], [x_max, y_max, 0]]  
            bounds = np.array([[bounds[0], bounds[2], 0.0],  
                              [bounds[1], bounds[3], 0.0]], dtype=np.float32)  
        elif len(bounds) == 6:  
            # 3D bounds: [x_min, x_max, y_min, y_max, z_min, z_max] -> [[x_min, y_min, z_min], [x_max, y_max, z_max]]  
            bounds = np.array([[bounds[0], bounds[2], bounds[4]],  
                              [bounds[1], bounds[3], bounds[5]]], dtype=np.float32)  
        else:  
            raise ValueError(f"1D bounds must have 4 or 6 elements, got {len(bounds)}")  
    elif bounds.ndim == 2:  
        if bounds.shape == (2, 2):  
            # 2D bounds: [[x_min, x_max], [y_min, y_max]] -> [[x_min, y_min, 0], [x_max, y_max, 0]]  
            bounds = np.array([[bounds[0, 0], bounds[1, 0], 0.0],  
                              [bounds[0, 1], bounds[1, 1], 0.0]], dtype=np.float32)  
        elif bounds.shape == (2, 3):  
            # Already correct format: [[x_min, y_min, z_min], [x_max, y_max, z_max]]  
            pass  
        elif bounds.shape == (3, 2):  
            # Transpose format: [[x_min, x_max], [y_min, y_max], [z_min, z_max]] -> [[x_min, y_min, z_min], [x_max, y_max, z_max]]  
            bounds = bounds.T  
        else:  
            raise ValueError(f"2D bounds must have shape (2,2), (2,3), or (3,2), got {bounds.shape}")  
    else:  
        raise ValueError(f"Bounds must be 1D or 2D array, got {bounds.ndim}D")  
    
    # Validate that min < max  
    if not np.all(bounds[0] < bounds[1]):  
        raise ValueError(f"Invalid bounds: min {bounds[0]} >= max {bounds[1]}")  
    
    return bounds  


# ---------- Basic seeding strategies ----------  

def random_seeds(n: int, bounds: Union[np.ndarray, List], rng_seed: int = 0,   
                dtype=np.float32) -> np.ndarray:  
    """  
    Uniformly sample n seed positions within bounds.  
    
    Parameters  
    ----------  
    n : int  
        Number of seed positions to generate  
    bounds : array-like  
        Domain bounds in various formats  
    rng_seed : int  
        Random number generator seed for reproducibility  
    dtype : dtype  
        Data type for generated positions  
        
    Returns  
    -------  
    np.ndarray  
        Random seed positions, shape (n, 3), dtype float32  
    """  
    if n <= 0:  
        return np.zeros((0, 3), dtype=np.float32)  
    
    # Validate and standardize bounds  
    bounds_std = _validate_bounds(np.asarray(bounds))  # (2, 3)  
    
    if JAX_AVAILABLE:  
        # Use JAX for random generation  
        key = jax.random.PRNGKey(rng_seed)  
        u = jax.random.uniform(key, shape=(n, 3), dtype=jnp.float32)  
        lo = jnp.asarray(bounds_std[0], dtype=jnp.float32)  
        hi = jnp.asarray(bounds_std[1], dtype=jnp.float32)  
        positions = lo + u * (hi - lo)  
        return np.asarray(positions, dtype=np.float32)  
    else:  
        # NumPy fallback  
        np.random.seed(rng_seed)  
        u = np.random.uniform(0, 1, size=(n, 3)).astype(np.float32)  
        lo = bounds_std[0]  
        hi = bounds_std[1]  
        positions = lo + u * (hi - lo)  
        return positions.astype(np.float32)  


def uniform_grid_seeds(resolution: Union[int, Tuple[int, int, int]],   
                      bounds: Union[np.ndarray, List],   
                      include_boundaries: bool = True) -> np.ndarray:  
    """  
    Generate seeds on uniform grid within bounds.  
    
    Parameters  
    ----------  
    resolution : int or tuple  
        Grid resolution. If int, uses same resolution for all dimensions.  
    bounds : array-like  
        Domain bounds  
    include_boundaries : bool  
        Whether to include points exactly on domain boundaries  
        
    Returns  
    -------  
    np.ndarray  
        Grid seed positions, shape (N, 3), dtype float32  
    """  
    # Validate bounds  
    bounds_std = _validate_bounds(np.asarray(bounds))  # (2, 3)  
    
    # Handle resolution specification  
    if isinstance(resolution, int):  
        nx = ny = nz = resolution  
    else:  
        if len(resolution) == 2:  
            nx, ny = resolution  
            nz = 1  
        elif len(resolution) == 3:  
            nx, ny, nz = resolution  
        else:  
            raise ValueError(f"Resolution must be int or tuple of 2-3 ints, got {resolution}")  
    
    # Create coordinate arrays  
    x_min, y_min, z_min = bounds_std[0]  
    x_max, y_max, z_max = bounds_std[1]  
    
    if include_boundaries:  
        x_coords = np.linspace(x_min, x_max, nx, dtype=np.float32)  
        y_coords = np.linspace(y_min, y_max, ny, dtype=np.float32)  
        z_coords = np.linspace(z_min, z_max, nz, dtype=np.float32) if nz > 1 else np.array([z_min], dtype=np.float32)  
    else:  
        # Interior points only  
        dx = (x_max - x_min) / (nx + 1)  
        dy = (y_max - y_min) / (ny + 1)  
        dz = (z_max - z_min) / (nz + 1) if nz > 1 else 0.0  
        
        x_coords = np.linspace(x_min + dx, x_max - dx, nx, dtype=np.float32)  
        y_coords = np.linspace(y_min + dy, y_max - dy, ny, dtype=np.float32)  
        z_coords = np.linspace(z_min + dz, z_max - dz, nz, dtype=np.float32) if nz > 1 else np.array([z_min], dtype=np.float32)  
    
    # Create meshgrid and flatten  
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')  
    
    positions = np.column_stack([  
        X.ravel(),  
        Y.ravel(),   
        Z.ravel()  
    ]).astype(np.float32)  
    
    return positions  


def line_seeds(start: Union[List, np.ndarray], end: Union[List, np.ndarray],   
               n: int) -> np.ndarray:  
    """  
    Generate seeds along a line between two points.  
    
    Parameters  
    ----------  
    start : array-like  
        Starting point, shape (2,) or (3,)  
    end : array-like  
        Ending point, shape (2,) or (3,)  
    n : int  
        Number of seed points  
        
    Returns  
    -------  
    np.ndarray  
        Line seed positions, shape (n, 3), dtype float32  
    """  
    if n <= 0:  
        return np.zeros((0, 3), dtype=np.float32)  
    
    start = _ensure_seed_positions_shape(np.asarray(start))[0]  # (3,)  
    end = _ensure_seed_positions_shape(np.asarray(end))[0]      # (3,)  
    
    # Generate line parameters  
    t = np.linspace(0, 1, n, dtype=np.float32)  
    
    # Interpolate positions  
    positions = start[None, :] + t[:, None] * (end - start)[None, :]  
    
    return positions.astype(np.float32)  


def circle_seeds(center: Union[List, np.ndarray], radius: float, n: int,  
                plane: str = "xy", start_angle: float = 0.0) -> np.ndarray:  
    """  
    Generate seeds on a circle.  
    
    Parameters  
    ----------  
    center : array-like  
        Circle center, shape (2,) or (3,)  
    radius : float  
        Circle radius  
    n : int  
        Number of seed points  
    plane : str  
        Circle plane: 'xy', 'xz', or 'yz'  
    start_angle : float  
        Starting angle in radians  
        
    Returns  
    -------  
    np.ndarray  
        Circle seed positions, shape (n, 3), dtype float32  
    """  
    if n <= 0:  
        return np.zeros((0, 3), dtype=np.float32)  
    
    center = _ensure_seed_positions_shape(np.asarray(center))[0]  # (3,)  
    
    # Generate angles  
    angles = np.linspace(start_angle, start_angle + 2*np.pi, n, endpoint=False, dtype=np.float32)  
    
    # Create circle points in specified plane  
    positions = np.zeros((n, 3), dtype=np.float32)  
    
    if plane == "xy":  
        positions[:, 0] = center[0] + radius * np.cos(angles)  
        positions[:, 1] = center[1] + radius * np.sin(angles)  
        positions[:, 2] = center[2]  
    elif plane == "xz":  
        positions[:, 0] = center[0] + radius * np.cos(angles)  
        positions[:, 1] = center[1]  
        positions[:, 2] = center[2] + radius * np.sin(angles)  
    elif plane == "yz":  
        positions[:, 0] = center[0]  
        positions[:, 1] = center[1] + radius * np.cos(angles)  
        positions[:, 2] = center[2] + radius * np.sin(angles)  
    else:  
        raise ValueError(f"Unknown plane: {plane}. Use 'xy', 'xz', or 'yz'")  
    
    return positions  


def sphere_seeds(center: Union[List, np.ndarray], radius: float, n: int,  
                distribution: str = "uniform") -> np.ndarray:  
    """  
    Generate seeds on or in a sphere.  
    
    Parameters  
    ----------  
    center : array-like  
        Sphere center, shape (2,) or (3,)  
    radius : float  
        Sphere radius  
    n : int  
        Number of seed points  
    distribution : str  
        Distribution type: 'uniform' (volume), 'surface' (surface only)  
        
    Returns  
    -------  
    np.ndarray  
        Sphere seed positions, shape (n, 3), dtype float32  
    """  
    if n <= 0:  
        return np.zeros((0, 3), dtype=np.float32)  
    
    center = _ensure_seed_positions_shape(np.asarray(center))[0]  # (3,)  
    
    # Generate random unit vectors  
    if JAX_AVAILABLE:  
        key = jax.random.PRNGKey(0)  
        # Normal distribution for unit vectors  
        directions = jax.random.normal(key, shape=(n, 3), dtype=jnp.float32)  
        norms = jnp.linalg.norm(directions, axis=1, keepdims=True)  
        unit_vectors = directions / norms  
        
        if distribution == "uniform":  
            # Uniform volume distribution  
            key2 = jax.random.split(key)[0]  
            r_cubed = jax.random.uniform(key2, shape=(n, 1), dtype=jnp.float32)  
            radii = radius * (r_cubed ** (1/3))  
        elif distribution == "surface":  
            # Surface only  
            radii = jnp.full((n, 1), radius, dtype=jnp.float32)  
        else:  
            raise ValueError(f"Unknown distribution: {distribution}")  
        
        positions = np.asarray(center + radii * unit_vectors, dtype=np.float32)  
    else:  
        # NumPy fallback  
        directions = np.random.normal(0, 1, size=(n, 3)).astype(np.float32)  
        norms = np.linalg.norm(directions, axis=1, keepdims=True)  
        unit_vectors = directions / norms  
        
        if distribution == "uniform":  
            r_cubed = np.random.uniform(0, 1, size=(n, 1)).astype(np.float32)  
            radii = radius * (r_cubed ** (1/3))  
        elif distribution == "surface":
            radii = np.full((n, 1), radius, dtype=np.float32)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        positions = center + radii * unit_vectors
    
    return positions.astype(np.float32)


def rectangular_seeds(bounds: Union[np.ndarray, List], n: int, 
                     distribution: str = "uniform") -> np.ndarray:
    """
    Generate seeds in a rectangular region.
    
    Parameters
    ----------
    bounds : array-like
        Rectangular bounds
    n : int
        Number of seed points
    distribution : str
        Distribution type: 'uniform', 'boundary', 'corners'
        
    Returns
    -------
    np.ndarray
        Rectangular seed positions, shape (n, 3), dtype float32
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    bounds_std = _validate_bounds(np.asarray(bounds))  # (2, 3)
    
    if distribution == "uniform":
        return random_seeds(n, bounds_std)
    
    elif distribution == "boundary":
        # Generate seeds on boundary faces
        positions = []
        n_per_face = n // 6  # 6 faces for 3D box
        remainder = n % 6
        
        x_min, y_min, z_min = bounds_std[0]
        x_max, y_max, z_max = bounds_std[1]
        
        # Face seeds
        faces = [
            # X faces
            ([x_min, x_min], [y_min, y_max], [z_min, z_max]),  # x=x_min
            ([x_max, x_max], [y_min, y_max], [z_min, z_max]),  # x=x_max
            # Y faces  
            ([x_min, x_max], [y_min, y_min], [z_min, z_max]),  # y=y_min
            ([x_min, x_max], [y_max, y_max], [z_min, z_max]),  # y=y_max
            # Z faces
            ([x_min, x_max], [y_min, y_max], [z_min, z_min]),  # z=z_min
            ([x_min, x_max], [y_min, y_max], [z_max, z_max]),  # z=z_max
        ]
        
        for i, (x_range, y_range, z_range) in enumerate(faces):
            n_face = n_per_face + (1 if i < remainder else 0)
            if n_face > 0:
                face_bounds = np.array([
                    [x_range[0], y_range[0], z_range[0]],
                    [x_range[1], y_range[1], z_range[1]]
                ], dtype=np.float32)
                face_seeds = random_seeds(n_face, face_bounds)
                positions.append(face_seeds)
        
        if positions:
            return np.concatenate(positions, axis=0)
        else:
            return np.zeros((0, 3), dtype=np.float32)
    
    elif distribution == "corners":
        # 8 corners of the box
        x_min, y_min, z_min = bounds_std[0]
        x_max, y_max, z_max = bounds_std[1]
        
        corners = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_min, y_max, z_min], [x_max, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_min, y_max, z_max], [x_max, y_max, z_max],
        ], dtype=np.float32)
        
        # Repeat corners to reach desired count
        if n <= 8:
            return corners[:n]
        else:
            # Add some interior points
            n_corners = 8
            n_interior = n - n_corners
            interior_seeds = random_seeds(n_interior, bounds_std)
            return np.concatenate([corners, interior_seeds], axis=0)
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# ---------- Advanced seeding strategies ----------

def gaussian_cluster_seeds(centers: Union[List, np.ndarray], std: float, 
                          n_per_cluster: int, rng_seed: int = 0) -> np.ndarray:
    """
    Generate seeds in Gaussian clusters around specified centers.
    
    Parameters
    ----------
    centers : array-like
        Cluster centers, shape (n_clusters, 2) or (n_clusters, 3)
    std : float
        Standard deviation for Gaussian distribution
    n_per_cluster : int
        Number of seeds per cluster
    rng_seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Clustered seed positions, shape (n_clusters * n_per_cluster, 3)
    """
    centers = _ensure_seed_positions_shape(np.asarray(centers))  # (n_clusters, 3)
    n_clusters = centers.shape[0]
    
    if n_per_cluster <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    all_positions = []
    
    for i, center in enumerate(centers):
        if JAX_AVAILABLE:
            key = jax.random.PRNGKey(rng_seed + i)
            offsets = jax.random.normal(key, shape=(n_per_cluster, 3), dtype=jnp.float32) * std
            cluster_positions = center + np.asarray(offsets)
        else:
            np.random.seed(rng_seed + i)
            offsets = np.random.normal(0, std, size=(n_per_cluster, 3)).astype(np.float32)
            cluster_positions = center + offsets
        
        all_positions.append(cluster_positions)
    
    return np.concatenate(all_positions, axis=0).astype(np.float32)


def stratified_seeds(bounds: Union[np.ndarray, List], n: int, 
                    n_strata: Union[int, Tuple[int, int, int]] = 4,
                    rng_seed: int = 0) -> np.ndarray:
    """
    Generate stratified random seeds for better spatial distribution.
    
    Parameters
    ----------
    bounds : array-like
        Domain bounds
    n : int
        Total number of seed points
    n_strata : int or tuple
        Number of strata per dimension
    rng_seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Stratified seed positions, shape (n, 3), dtype float32
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    bounds_std = _validate_bounds(np.asarray(bounds))  # (2, 3)
    
    # Handle strata specification
    if isinstance(n_strata, int):
        nx = ny = nz = n_strata
    else:
        if len(n_strata) == 2:
            nx, ny = n_strata
            nz = 1
        elif len(n_strata) == 3:
            nx, ny, nz = n_strata
        else:
            raise ValueError(f"n_strata must be int or tuple of 2-3 ints")
    
    total_strata = nx * ny * nz
    n_per_stratum = n // total_strata
    remainder = n % total_strata
    
    # Create stratum bounds
    x_min, y_min, z_min = bounds_std[0]
    x_max, y_max, z_max = bounds_std[1]
    
    x_edges = np.linspace(x_min, x_max, nx + 1, dtype=np.float32)
    y_edges = np.linspace(y_min, y_max, ny + 1, dtype=np.float32)
    z_edges = np.linspace(z_min, z_max, nz + 1, dtype=np.float32) if nz > 1 else np.array([z_min, z_max], dtype=np.float32)
    
    all_positions = []
    stratum_idx = 0
    
    np.random.seed(rng_seed)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Define stratum bounds
                stratum_bounds = np.array([
                    [x_edges[i], y_edges[j], z_edges[k]],
                    [x_edges[i+1], y_edges[j+1], z_edges[k+1]]
                ], dtype=np.float32)
                
                # Number of seeds in this stratum
                n_stratum = n_per_stratum + (1 if stratum_idx < remainder else 0)
                
                if n_stratum > 0:
                    stratum_seeds = random_seeds(n_stratum, stratum_bounds, 
                                               rng_seed=rng_seed + stratum_idx)
                    all_positions.append(stratum_seeds)
                
                stratum_idx += 1
    
    if all_positions:
        return np.concatenate(all_positions, axis=0)
    else:
        return np.zeros((0, 3), dtype=np.float32)


def custom_distribution_seeds(bounds: Union[np.ndarray, List], n: int,
                            pdf_func: Callable[[np.ndarray], np.ndarray],
                            max_iterations: int = 1000,
                            rng_seed: int = 0) -> np.ndarray:
    """
    Generate seeds using rejection sampling with custom probability density function.
    
    Parameters
    ----------
    bounds : array-like
        Domain bounds
    n : int
        Number of seed points to generate
    pdf_func : callable
        Probability density function: pdf_func(positions) -> densities
    max_iterations : int
        Maximum iterations for rejection sampling
    rng_seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Custom distributed seed positions, shape (n, 3), dtype float32
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    bounds_std = _validate_bounds(np.asarray(bounds))  # (2, 3)
    
    accepted_positions = []
    iterations = 0
    np.random.seed(rng_seed)
    
    # Estimate maximum PDF value for rejection sampling
    test_positions = random_seeds(1000, bounds_std, rng_seed=rng_seed)
    test_densities = pdf_func(test_positions)
    max_density = np.max(test_densities) * 1.1  # Add safety margin
    
    while len(accepted_positions) < n and iterations < max_iterations:
        # Generate candidate positions
        batch_size = min(n * 2, 1000)  # Generate extra to improve efficiency
        candidates = random_seeds(batch_size, bounds_std, 
                                rng_seed=rng_seed + iterations)
        
        # Evaluate PDF
        densities = pdf_func(candidates)
        
        # Rejection sampling
        u = np.random.uniform(0, max_density, size=len(candidates))
        accepted_mask = u <= densities
        
        if np.any(accepted_mask):
            accepted_positions.append(candidates[accepted_mask])
        
        iterations += 1
    
    if not accepted_positions:
        warnings.warn("Rejection sampling failed - returning uniform random seeds")
        return random_seeds(n, bounds_std, rng_seed=rng_seed)
    
    all_accepted = np.concatenate(accepted_positions, axis=0)
    
    if len(all_accepted) >= n:
        return all_accepted[:n].astype(np.float32)
    else:
        # Fill remaining with uniform random
        remaining = n - len(all_accepted)
        additional = random_seeds(remaining, bounds_std, rng_seed=rng_seed + 9999)
        return np.concatenate([all_accepted, additional], axis=0).astype(np.float32)


# ---------- Field-informed seeding ----------

def field_informed_seeds(velocity_field, bounds: Union[np.ndarray, List], 
                        n: int, metric: str = "divergence", 
                        sample_resolution: Tuple[int, int, int] = (50, 50, 50),
                        rng_seed: int = 0) -> np.ndarray:
    """
    Generate seeds based on velocity field properties.
    
    Parameters
    ----------
    velocity_field : BaseField
        Velocity field for informed seeding
    bounds : array-like
        Domain bounds for seeding
    n : int
        Number of seed points
    metric : str
        Field metric for seeding: 'magnitude', 'divergence', 'vorticity'
    sample_resolution : tuple
        Resolution for field sampling
    rng_seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Field-informed seed positions, shape (n, 3), dtype float32
    """
    from ..fields import BaseField
    
    if not isinstance(velocity_field, BaseField):
        raise TypeError("velocity_field must be a BaseField instance")
    
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    bounds_std = _validate_bounds(np.asarray(bounds))  # (2, 3)
    
    # Create sampling grid
    grid_positions = uniform_grid_seeds(sample_resolution, bounds_std)
    
    try:
        # Sample velocity field
        if hasattr(velocity_field, 'sample_at_positions'):
            velocities = velocity_field.sample_at_positions(grid_positions, t=0.0)
        else:
            # Fallback for different field interfaces
            velocities = velocity_field.sample(grid_positions)
        
        # Compute field metric
        if metric == "magnitude":
            weights = np.linalg.norm(velocities, axis=1)  # (N_grid,)
        elif metric == "divergence":
            # Simplified divergence approximation
            # For proper divergence, would need field derivatives
            weights = np.linalg.norm(velocities, axis=1)
            warnings.warn("Using velocity magnitude as divergence approximation")
        elif metric == "vorticity":
            # Simplified vorticity approximation  
            weights = np.linalg.norm(velocities, axis=1)
            warnings.warn("Using velocity magnitude as vorticity approximation")
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Normalize weights to create probability distribution
        weights = weights + 1e-10  # Avoid zero weights
        weights = weights / np.sum(weights)
        
        # Custom PDF based on field metric
        def field_pdf(positions):
            # Find nearest grid points and interpolate weights
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(grid_positions)
                distances, indices = tree.query(positions, k=1)
                return weights[indices]
            except ImportError:
                # Simple nearest neighbor fallback
                distances = np.linalg.norm(
                    positions[:, None, :] - grid_positions[None, :, :], 
                    axis=2
                )
                nearest_indices = np.argmin(distances, axis=1)
                return weights[nearest_indices]
        
        # Generate seeds using custom distribution
        return custom_distribution_seeds(
            bounds_std, n, field_pdf, 
            rng_seed=rng_seed
        )
        
    except Exception as e:
        warnings.warn(f"Field-informed seeding failed: {e}. Using uniform random seeds.")
        return random_seeds(n, bounds_std, rng_seed=rng_seed)


# ---------- Geometry-based seeding ----------

def boundary_seeds(geometry, n: int, distribution: str = "uniform") -> np.ndarray:
    """
    Generate seeds on geometric boundaries.
    
    Parameters
    ----------
    geometry : object
        Geometry object with boundary information
    n : int
        Number of seed points
    distribution : str
        Distribution along boundary: 'uniform', 'curvature'
        
    Returns
    -------
    np.ndarray
        Boundary seed positions, shape (n, 3), dtype float32
        
    Notes
    -----
    This is a placeholder implementation. Full implementation would
    require specific geometry classes.
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    # Placeholder implementation - would need actual geometry handling
    warnings.warn("boundary_seeds is a placeholder - implement with actual geometry classes")
    
    # Return some default boundary seeds (rectangular boundary)
    if hasattr(geometry, 'bounds'):
        bounds = geometry.bounds
        return rectangular_seeds(bounds, n, distribution="boundary")
    else:
        # Default rectangular boundary
        bounds = np.array([[-1, -1, -1], [1, 1, 1]], dtype=np.float32)
        return rectangular_seeds(bounds, n, distribution="boundary")


# ---------- Utility functions ----------

def validate_seeds(positions: np.ndarray, bounds: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Validate seed positions.
    
    Parameters
    ----------
    positions : np.ndarray
        Seed positions to validate
    bounds : np.ndarray, optional
        Expected bounds for validation
        
    Returns
    -------
    dict
        Validation results
    """
    results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'stats': {}
    }
    
    try:
        # Ensure proper format
        positions = _ensure_seed_positions_shape(positions)
        n_seeds, dims = positions.shape
        
        # Basic validation
        if dims != 3:
            results['errors'].append(f"Positions must have 3 dimensions, got {dims}")
            results['valid'] = False
        
        if n_seeds == 0:
            results['warnings'].append("No seed positions provided")
        
        # Check for NaN/infinite values
        if not np.all(np.isfinite(positions)):
            results['errors'].append("Positions contain non-finite values (NaN/inf)")
            results['valid'] = False
        
        # Check data type
        if positions.dtype != np.float32:
            results['warnings'].append(f"Positions dtype {positions.dtype} is not optimal float32")
        
        # Bounds validation
        if bounds is not None:
            bounds_std = _validate_bounds(bounds)
            pos_min = np.min(positions, axis=0)
            pos_max = np.max(positions, axis=0)
            
            if np.any(pos_min < bounds_std[0]) or np.any(pos_max > bounds_std[1]):
                results['warnings'].append("Some positions are outside specified bounds")
        
        # Statistics
        if n_seeds > 0:
            pos_min = np.min(positions, axis=0)
            pos_max = np.max(positions, axis=0)
            pos_mean = np.mean(positions, axis=0)
            pos_std = np.std(positions, axis=0)
            
            results['stats'] = {
                'num_seeds': n_seeds,
                'bounds_min': pos_min.tolist(),
                'bounds_max': pos_max.tolist(),
                'mean': pos_mean.tolist(),
                'std': pos_std.tolist(),
                'memory_mb': positions.nbytes / (1024**2)
            }
    
    except Exception as e:
        results['errors'].append(f"Validation failed: {str(e)}")
        results['valid'] = False
    
    return results


def combine_seed_strategies(strategies: List[Dict[str, Any]]) -> np.ndarray:
    """
    Combine multiple seeding strategies.
    
    Parameters
    ----------
    strategies : list of dict
        List of strategy dictionaries with 'type', 'n', and other parameters
        
    Returns
    -------
    np.ndarray
        Combined seed positions, shape (total_n, 3), dtype float32
        
    Examples
    --------
    >>> strategies = [
    ...     {'type': 'random', 'n': 100, 'bounds': [[-1, 1], [-1, 1]]},
    ...     {'type': 'circle', 'n': 50, 'center': [0, 0], 'radius': 0.5},
    ...     {'type': 'line', 'n': 25, 'start': [-1, -1], 'end': [1, 1]}
    ... ]
    >>> seeds = combine_seed_strategies(strategies)
    """
    all_positions = []
    
    for strategy in strategies:
        strategy_type = strategy.pop('type')
        n = strategy.pop('n', 0)
        
        if n <= 0:
            continue
            
        if strategy_type == 'random':
            positions = random_seeds(n, **strategy)
        elif strategy_type == 'grid':
            positions = uniform_grid_seeds(strategy.pop('resolution', n), **strategy)
        elif strategy_type == 'line':
            positions = line_seeds(n=n, **strategy)
        elif strategy_type == 'circle':
            positions = circle_seeds(n=n, **strategy)
        elif strategy_type == 'sphere':
            positions = sphere_seeds(n=n, **strategy)
        elif strategy_type == 'rectangular':
            positions = rectangular_seeds(n=n, **strategy)
        elif strategy_type == 'gaussian_cluster':
            positions = gaussian_cluster_seeds(n_per_cluster=n, **strategy)
        elif strategy_type == 'stratified':
            positions = stratified_seeds(n=n, **strategy)
        else:
            warnings.warn(f"Unknown strategy type: {strategy_type}")
            continue
        
        all_positions.append(positions)
    
    if not all_positions:
        return np.zeros((0, 3), dtype=np.float32)
    
    return np.concatenate(all_positions, axis=0).astype(np.float32)


def seed_density_analysis(positions: np.ndarray, bins: int = 50) -> Dict[str, Any]:
    """
    Analyze spatial density distribution of seed points.
    
    Parameters
    ----------
    positions : np.ndarray
        Seed positions, shape (n, 3)
    bins : int
        Number of bins for density histogram
        
    Returns
    -------
    dict
        Density analysis results
    """
    positions = _ensure_seed_positions_shape(positions)
    n_seeds = positions.shape[0]
    
    if n_seeds == 0:
        return {'empty': True}
    
    # Compute bounds
    pos_min = np.min(positions, axis=0)
    pos_max = np.max(positions, axis=0)
    domain_volume = np.prod(pos_max - pos_min)
    
    # Overall density
    mean_density = n_seeds / domain_volume if domain_volume > 0 else 0.0
    
    # Per-dimension densities
    densities_1d = {}
    for i, dim in enumerate(['x', 'y', 'z']):
        if pos_max[i] > pos_min[i]:
            hist, _ = np.histogram(positions[:, i], bins=bins, 
                                 range=(pos_min[i], pos_max[i]))
            densities_1d[dim] = {
                'histogram': hist.tolist(),
                'mean': float(np.mean(hist)),
                'std': float(np.std(hist)),
                'max': float(np.max(hist))
            }
    
    # Nearest neighbor distances
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(positions)
        distances, _ = tree.query(positions, k=2)  # k=2 to exclude self
        nn_distances = distances[:, 1]  # Second nearest (first is self)
        
        nn_stats = {
            'mean': float(np.mean(nn_distances)),
            'std': float(np.std(nn_distances)),
            'min': float(np.min(nn_distances)),
            'max': float(np.max(nn_distances))
        }
    except ImportError:
        nn_stats = {'unavailable': 'scipy required'}
    
    return {
        'num_seeds': n_seeds,
        'bounds': {
            'min': pos_min.tolist(),
            'max': pos_max.tolist(),
            'extent': (pos_max - pos_min).tolist()
        },
        'domain_volume': domain_volume,
        'mean_density': mean_density,
        'densities_1d': densities_1d,
        'nearest_neighbor': nn_stats
    }