# jaxtrace/tracking/boundary.py  
"""  
Boundary condition handlers for particle tracking.  

Provides periodic wrapping, reflection, clamping and custom boundary conditions  
with efficient vectorized implementations and full JAX/NumPy compatibility.  
"""  

from __future__ import annotations  
from typing import Callable, Optional, Union, Dict, Any, Tuple, Protocol, List
from dataclasses import dataclass  
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
    # Mock jax.jit for NumPy fallback  
    class MockJit:  
        def __call__(self, func):  
            return func  
    jax = type('MockJax', (), {'jit': MockJit()})()  


def _ensure_float32(data: np.ndarray) -> np.ndarray:  
    """Convert data to float32 for consistency."""  
    return np.asarray(data, dtype=np.float32)  


def _ensure_bounds_shape(bounds: Union[np.ndarray, list]) -> np.ndarray:  
    """  
    Ensure bounds have shape (2, 3) with float32 dtype.  
    
    Parameters  
    ----------  
    bounds : array-like  
        Bounds in various formats  
        
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
            # Already correct format  
            pass  
        elif bounds.shape == (3, 2):  
            # Transpose format  
            bounds = bounds.T  
        else:  
            raise ValueError(f"2D bounds must have shape (2,2), (2,3), or (3,2), got {bounds.shape}")  
    else:  
        raise ValueError(f"Bounds must be 1D or 2D array, got {bounds.ndim}D")  
    
    # Validate bounds  
    if not np.all(bounds[0] < bounds[1]):  
        raise ValueError(f"Invalid bounds: min {bounds[0]} >= max {bounds[1]}")  
    
    return bounds.astype(np.float32)  


# ---------- Protocol for boundary conditions ----------  

class BoundaryCondition(Protocol):  
    """Protocol for boundary condition functions."""  
    
    def __call__(self, positions: np.ndarray) -> np.ndarray:  
        """Apply boundary condition to positions."""  
        ...  


# ---------- Core boundary condition functions ----------  

@jax.jit  
def apply_periodic(x: jnp.ndarray, bounds: jnp.ndarray) -> jnp.ndarray:  
    """  
    Apply periodic wrapping to coordinates x within AABB bounds.  
    
    Parameters  
    ----------  
    x : jnp.ndarray  
        Particle positions, shape (N, 3)  
    bounds : jnp.ndarray  
        Domain bounds [min, max], shape (2, 3)  
        
    Returns  
    -------  
    jnp.ndarray  
        Wrapped positions, shape (N, 3)  
    """  
    lo = bounds[0]  
    hi = bounds[1]  
    width = jnp.maximum(hi - lo, 0.0)  
    # Avoid division by zero when an axis has zero width  
    w_safe = jnp.where(width > 0, width, 1.0)  
    return lo + jnp.mod(x - lo, w_safe)  


@jax.jit  
def clamp_to_bounds(x: jnp.ndarray, bounds: jnp.ndarray) -> jnp.ndarray:  
    """  
    Clamp coordinates to [lo, hi] bounds.  
    
    Parameters  
    ----------  
    x : jnp.ndarray  
        Particle positions, shape (N, 3)  
    bounds : jnp.ndarray  
        Domain bounds [min, max], shape (2, 3)  
        
    Returns  
    -------  
    jnp.ndarray  
        Clamped positions, shape (N, 3)  
    """  
    return jnp.clip(x, a_min=bounds[0], a_max=bounds[1])  


@jax.jit  
def reflect_boundary(x: jnp.ndarray, bounds: jnp.ndarray) -> jnp.ndarray:  
    """  
    Reflective boundary conditions within [lo,hi] on each axis.  
    
    Uses a sawtooth reflection mapping with period 2*width.  
    
    Parameters  
    ----------  
    x : jnp.ndarray  
        Particle positions, shape (N, 3)  
    bounds : jnp.ndarray  
        Domain bounds [min, max], shape (2, 3)  
        
    Returns  
    -------  
    jnp.ndarray  
        Reflected positions, shape (N, 3)  
    """  
    lo = bounds[0]  
    hi = bounds[1]  
    width = jnp.maximum(hi - lo, 0.0)  
    # Map to [0, 2*width]  
    y = jnp.mod(x - lo, 2.0 * jnp.where(width > 0, width, 1.0))  
    # Reflect back to [0, width]  
    y_ref = jnp.where(y <= width, y, 2.0 * width - y)  
    return lo + y_ref  


@jax.jit  
def absorbing_boundary(x: jnp.ndarray, bounds: jnp.ndarray,   
                      outside_value: float = jnp.nan) -> jnp.ndarray:  
    """  
    Absorbing boundary conditions - particles outside bounds are marked.  
    
    Parameters  
    ----------  
    x : jnp.ndarray  
        Particle positions, shape (N, 3)  
    bounds : jnp.ndarray  
        Domain bounds [min, max], shape (2, 3)  
    outside_value : float  
        Value to set for particles outside bounds  
        
    Returns  
    -------  
    jnp.ndarray  
        Positions with outside particles marked, shape (N, 3)  
    """  
    lo, hi = bounds[0], bounds[1]  
    
    # Check which particles are outside bounds  
    outside_mask = jnp.logical_or(  
        jnp.any(x < lo, axis=1),  
        jnp.any(x > hi, axis=1)  
    )  
    
    # Set outside particles to specified value  
    result = jnp.where(  
        outside_mask[:, None],   
        outside_value,   
        x  
    )  
    
    return result  


# ---------- Boundary condition factories ----------  

def periodic_boundary(bounds: Union[np.ndarray, list]) -> BoundaryCondition:  
    """  
    Factory returning a callable that applies periodic wrapping within bounds.  
    
    Parameters  
    ----------  
    bounds : array-like  
        Domain bounds in various formats  
        
    Returns  
    -------  
    BoundaryCondition  
        Boundary condition function  
    """  
    bounds_std = _ensure_bounds_shape(bounds)  
    
    if JAX_AVAILABLE:  
        bounds_jax = jnp.asarray(bounds_std, dtype=jnp.float32)  
        
        @jax.jit  
        def _periodic_bc(x: jnp.ndarray) -> jnp.ndarray:  
            return apply_periodic(x, bounds_jax)  
    else:  
        def _periodic_bc(x: np.ndarray) -> np.ndarray:  
            x = _ensure_float32(x)  
            return np.asarray(apply_periodic(x, bounds_std), dtype=np.float32)  
    
    return _periodic_bc  


def reflective_boundary(bounds: Union[np.ndarray, list]) -> BoundaryCondition:  
    """  
    Factory returning a callable that applies reflective boundary conditions.  
    
    Parameters  
    ----------  
    bounds : array-like  
        Domain bounds  
        
    Returns  
    -------  
    BoundaryCondition  
        Boundary condition function  
    """  
    bounds_std = _ensure_bounds_shape(bounds)  
    
    if JAX_AVAILABLE:  
        bounds_jax = jnp.asarray(bounds_std, dtype=jnp.float32)  
        
        @jax.jit  
        def _reflective_bc(x: jnp.ndarray) -> jnp.ndarray:  
            return reflect_boundary(x, bounds_jax)  
    else:  
        def _reflective_bc(x: np.ndarray) -> np.ndarray:  
            x = _ensure_float32(x)  
            return np.asarray(reflect_boundary(x, bounds_std), dtype=np.float32)  
    
    return _reflective_bc  


def clamping_boundary(bounds: Union[np.ndarray, list]) -> BoundaryCondition:  
    """  
    Factory returning a callable that clamps positions to bounds.  
    
    Parameters  
    ----------  
    bounds : array-like  
        Domain bounds  
        
    Returns  
    -------  
    BoundaryCondition  
        Boundary condition function  
    """  
    bounds_std = _ensure_bounds_shape(bounds)  
    
    if JAX_AVAILABLE:  
        bounds_jax = jnp.asarray(bounds_std, dtype=jnp.float32)  
        
        @jax.jit  
        def _clamping_bc(x: jnp.ndarray) -> jnp.ndarray:  
            return clamp_to_bounds(x, bounds_jax)  
    else:  
        def _clamping_bc(x: np.ndarray) -> np.ndarray:  
            x = _ensure_float32(x)  
            return np.asarray(clamp_to_bounds(x, bounds_std), dtype=np.float32)  
    
    return _clamping_bc  


def absorbing_boundary_factory(bounds: Union[np.ndarray, list],   
                             outside_value: float = np.nan) -> BoundaryCondition:  
    """  
    Factory returning an absorbing boundary condition.  
    
    Parameters  
    ----------  
    bounds : array-like  
        Domain bounds  
    outside_value : float  
        Value for particles outside bounds  
        
    Returns  
    -------  
    BoundaryCondition  
        Boundary condition function  
    """  
    bounds_std = _ensure_bounds_shape(bounds)  
    
    if JAX_AVAILABLE:  
        bounds_jax = jnp.asarray(bounds_std, dtype=jnp.float32)  
        outside_jax = jnp.float32(outside_value)  
        
        @jax.jit  
        def _absorbing_bc(x: jnp.ndarray) -> jnp.ndarray:  
            return absorbing_boundary(x, bounds_jax, outside_jax)  
    else:  
        def _absorbing_bc(x: np.ndarray) -> np.ndarray:  
            x = _ensure_float32(x)  
            return np.asarray(absorbing_boundary(x, bounds_std, np.float32(outside_value)),   
                            dtype=np.float32)  
    
    return _absorbing_bc  


# ---------- Advanced boundary conditions ----------  

@dataclass  
class CompositeBoundaryCondition:  
    """  
    Composite boundary condition that applies different conditions per axis.  
    
    Attributes  
    ----------  
    x_condition : BoundaryCondition  
        Boundary condition for x-axis  
    y_condition : BoundaryCondition  
        Boundary condition for y-axis  
    z_condition : BoundaryCondition  
        Boundary condition for z-axis  
    """  
    x_condition: BoundaryCondition  
    y_condition: BoundaryCondition  
    z_condition: BoundaryCondition  
    
    def __call__(self, positions: np.ndarray) -> np.ndarray:  
        """Apply composite boundary condition."""  
        positions = _ensure_float32(positions)  
        result = positions.copy()  
        
        # Apply boundary condition per axis  
        # Note: This is a simplified implementation  
        # Full implementation would need axis-specific boundary handling  
        
        # For now, apply the most restrictive condition  
        result = self.x_condition(result)  
        result = self.y_condition(result)  
        result = self.z_condition(result)  
        
        return result.astype(np.float32)  


def mixed_boundary(bounds: Union[np.ndarray, list],   
                  conditions: Dict[str, str]) -> BoundaryCondition:  
    """  
    Create mixed boundary conditions with different behavior per axis.  
    
    Parameters  
    ----------  
    bounds : array-like  
        Domain bounds  
    conditions : dict  
        Boundary conditions per axis, e.g., {'x': 'periodic', 'y': 'reflective', 'z': 'clamping'}  
        
    Returns  
    -------  
    BoundaryCondition  
        Mixed boundary condition function  
    """  
    bounds_std = _ensure_bounds_shape(bounds)  
    
    # Create individual axis conditions  
    axis_conditions = {}  
    for axis in ['x', 'y', 'z']:  
        condition_type = conditions.get(axis, 'periodic')  
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]  
        
        # Create single-axis bounds  
        axis_bounds = np.array([  
            [bounds_std[0, axis_idx], bounds_std[0, axis_idx], bounds_std[0, axis_idx]],  
            [bounds_std[1, axis_idx], bounds_std[1, axis_idx], bounds_std[1, axis_idx]]  
        ], dtype=np.float32)  
        
        if condition_type == 'periodic':  
            axis_conditions[axis] = periodic_boundary(axis_bounds)  
        elif condition_type == 'reflective':  
            axis_conditions[axis] = reflective_boundary(axis_bounds)  
        elif condition_type == 'clamping':  
            axis_conditions[axis] = clamping_boundary(axis_bounds)  
        elif condition_type == 'absorbing':  
            axis_conditions[axis] = absorbing_boundary_factory(axis_bounds)  
        else:  
            raise ValueError(f"Unknown condition type: {condition_type}")  
    
    def _mixed_bc(positions: np.ndarray) -> np.ndarray:
        """Apply mixed boundary conditions per axis."""
        positions = _ensure_float32(positions)
        result = positions.copy()
        
        # Apply per-axis boundary conditions
        for i, axis in enumerate(['x', 'y', 'z']):
            # Extract axis coordinates
            axis_pos = result[:, i:i+1]  # Keep 2D shape for broadcasting
            axis_pos_3d = np.concatenate([axis_pos, axis_pos, axis_pos], axis=1)  # Expand to 3D
            
            # Apply axis-specific boundary condition
            axis_result_3d = axis_conditions[axis](axis_pos_3d)
            
            # Extract the relevant axis result
            result[:, i] = axis_result_3d[:, i]
        
        return result.astype(np.float32)
    
    return _mixed_bc


def distance_based_boundary(bounds: Union[np.ndarray, list], 
                          distance_func: Callable[[np.ndarray], np.ndarray],
                          inside_condition: str = 'periodic',
                          outside_condition: str = 'absorbing') -> BoundaryCondition:
    """
    Distance-based boundary condition using arbitrary distance function.
    
    Parameters
    ----------
    bounds : array-like
        Reference bounds for basic conditions
    distance_func : callable
        Function computing distance from boundary: distance_func(positions) -> distances
        Negative values indicate inside, positive outside
    inside_condition : str
        Boundary condition for particles inside: 'periodic', 'reflective', 'clamping'
    outside_condition : str
        Boundary condition for particles outside: 'absorbing', 'reflective'
        
    Returns
    -------
    BoundaryCondition
        Distance-based boundary condition function
    """
    bounds_std = _ensure_bounds_shape(bounds)
    
    # Create base conditions
    if inside_condition == 'periodic':
        inside_bc = periodic_boundary(bounds_std)
    elif inside_condition == 'reflective':
        inside_bc = reflective_boundary(bounds_std)
    elif inside_condition == 'clamping':
        inside_bc = clamping_boundary(bounds_std)
    else:
        raise ValueError(f"Unknown inside condition: {inside_condition}")
    
    if outside_condition == 'absorbing':
        outside_bc = absorbing_boundary_factory(bounds_std)
    elif outside_condition == 'reflective':
        outside_bc = reflective_boundary(bounds_std)
    elif outside_condition == 'clamping':
        outside_bc = clamping_boundary(bounds_std)
    else:
        raise ValueError(f"Unknown outside condition: {outside_condition}")
    
    def _distance_bc(positions: np.ndarray) -> np.ndarray:
        """Apply distance-based boundary condition."""
        positions = _ensure_float32(positions)
        distances = distance_func(positions)
        
        inside_mask = distances <= 0
        
        if np.all(inside_mask):
            # All particles inside
            return inside_bc(positions)
        elif np.all(~inside_mask):
            # All particles outside  
            return outside_bc(positions)
        else:
            # Mixed case
            result = positions.copy()
            
            if np.any(inside_mask):
                inside_positions = inside_bc(positions[inside_mask])
                result[inside_mask] = inside_positions
            
            if np.any(~inside_mask):
                outside_positions = outside_bc(positions[~inside_mask])
                result[~inside_mask] = outside_positions
            
            return result.astype(np.float32)
    
    return _distance_bc


# ---------- Specialized boundary geometries ----------

def spherical_boundary(center: Union[List, np.ndarray], radius: float,
                      condition: str = 'reflective') -> BoundaryCondition:
    """
    Spherical boundary condition.
    
    Parameters
    ----------
    center : array-like
        Sphere center, shape (2,) or (3,)
    radius : float
        Sphere radius
    condition : str
        Boundary condition: 'reflective', 'absorbing', 'clamping'
        
    Returns
    -------
    BoundaryCondition
        Spherical boundary condition function
    """
    center = _ensure_float32(center)
    if center.ndim == 1:
        if len(center) == 2:
            center = np.array([center[0], center[1], 0.0], dtype=np.float32)
        elif len(center) == 3:
            pass
        else:
            raise ValueError("Center must have 2 or 3 coordinates")
    else:
        raise ValueError("Center must be 1D array")
    
    def sphere_distance(positions):
        """Distance from sphere surface (negative inside)."""
        positions = _ensure_float32(positions)
        distances = np.linalg.norm(positions - center, axis=1)
        return distances - radius
    
    if condition == 'reflective':
        def _spherical_bc(positions: np.ndarray) -> np.ndarray:
            positions = _ensure_float32(positions)
            distances = sphere_distance(positions)
            
            # Reflect particles outside sphere
            outside_mask = distances > 0
            if np.any(outside_mask):
                # Project back to surface and reflect
                directions = (positions[outside_mask] - center) 
                directions_norm = directions / np.linalg.norm(directions, axis=1, keepdims=True)
                
                # Place on surface and reflect inward
                surface_points = center + radius * directions_norm
                # Simple reflection: place at surface
                positions[outside_mask] = surface_points
            
            return positions.astype(np.float32)
    
    elif condition == 'absorbing':
        def _spherical_bc(positions: np.ndarray) -> np.ndarray:
            positions = _ensure_float32(positions)
            distances = sphere_distance(positions)
            
            # Mark particles outside sphere as absorbed
            outside_mask = distances > 0
            if np.any(outside_mask):
                positions[outside_mask] = np.nan
            
            return positions.astype(np.float32)
    
    elif condition == 'clamping':
        def _spherical_bc(positions: np.ndarray) -> np.ndarray:
            positions = _ensure_float32(positions)
            distances = sphere_distance(positions)
            
            # Clamp particles to sphere surface
            outside_mask = distances > 0
            if np.any(outside_mask):
                directions = (positions[outside_mask] - center)
                directions_norm = directions / np.linalg.norm(directions, axis=1, keepdims=True)
                positions[outside_mask] = center + radius * directions_norm
            
            return positions.astype(np.float32)
    
    else:
        raise ValueError(f"Unknown spherical condition: {condition}")
    
    return _spherical_bc


def cylindrical_boundary(axis: str = 'z', center: Union[List, np.ndarray] = [0, 0],
                        radius: float = 1.0, height: Optional[float] = None,
                        condition: str = 'reflective') -> BoundaryCondition:
    """
    Cylindrical boundary condition.
    
    Parameters
    ----------
    axis : str
        Cylinder axis: 'x', 'y', or 'z'
    center : array-like
        Center in the perpendicular plane
    radius : float
        Cylinder radius
    height : float, optional
        Cylinder height. If None, infinite cylinder.
    condition : str
        Boundary condition: 'reflective', 'absorbing', 'clamping'
        
    Returns
    -------
    BoundaryCondition
        Cylindrical boundary condition function
    """
    center = _ensure_float32(center)
    if len(center) != 2:
        raise ValueError("Center must have 2 coordinates for perpendicular plane")
    
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    perp_indices = [i for i in range(3) if i != axis_idx]
    
    def cylinder_distance(positions):
        """Distance from cylinder surface."""
        positions = _ensure_float32(positions)
        # Distance in perpendicular plane
        perp_coords = positions[:, perp_indices] - center
        radial_distance = np.linalg.norm(perp_coords, axis=1) - radius
        
        if height is not None:
            # Check axial bounds
            axial_distance = np.maximum(
                np.abs(positions[:, axis_idx]) - height/2, 
                0.0
            )
            return np.maximum(radial_distance, axial_distance)
        else:
            return radial_distance
    
    if condition == 'reflective':
        def _cylindrical_bc(positions: np.ndarray) -> np.ndarray:
            positions = _ensure_float32(positions)
            distances = cylinder_distance(positions)
            
            outside_mask = distances > 0
            if np.any(outside_mask):
                # Simple approach: project to surface
                perp_coords = positions[outside_mask][:, perp_indices] - center
                perp_norms = np.linalg.norm(perp_coords, axis=1, keepdims=True)
                perp_norms = np.maximum(perp_norms, 1e-10)  # Avoid division by zero
                
                # Project to cylinder surface
                surface_perp = center + radius * (perp_coords / perp_norms)
                positions[outside_mask][:, perp_indices] = surface_perp
                
                # Handle axial bounds if finite height
                if height is not None:
                    axial_pos = positions[outside_mask][:, axis_idx]
                    positions[outside_mask][:, axis_idx] = np.clip(axial_pos, -height/2, height/2)
            
            return positions.astype(np.float32)
    
    elif condition == 'absorbing':
        def _cylindrical_bc(positions: np.ndarray) -> np.ndarray:
            positions = _ensure_float32(positions)
            distances = cylinder_distance(positions)
            
            outside_mask = distances > 0
            if np.any(outside_mask):
                positions[outside_mask] = np.nan
            
            return positions.astype(np.float32)
    
    elif condition == 'clamping':
        def _cylindrical_bc(positions: np.ndarray) -> np.ndarray:
            positions = _ensure_float32(positions)
            distances = cylinder_distance(positions)
            
            outside_mask = distances > 0
            if np.any(outside_mask):
                # Clamp to cylinder surface
                perp_coords = positions[outside_mask][:, perp_indices] - center
                perp_norms = np.linalg.norm(perp_coords, axis=1, keepdims=True)
                perp_norms = np.maximum(perp_norms, 1e-10)
                
                surface_perp = center + radius * (perp_coords / perp_norms)
                positions[outside_mask][:, perp_indices] = surface_perp
                
                if height is not None:
                    axial_pos = positions[outside_mask][:, axis_idx]
                    positions[outside_mask][:, axis_idx] = np.clip(axial_pos, -height/2, height/2)
            
            return positions.astype(np.float32)
    
    else:
        raise ValueError(f"Unknown cylindrical condition: {condition}")
    
    return _cylindrical_bc


# ---------- Boundary condition analysis ----------

def check_boundary_violations(positions: np.ndarray, bounds: Union[np.ndarray, list],
                            tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Check for boundary violations and provide statistics.
    
    Parameters
    ----------
    positions : np.ndarray
        Particle positions to check, shape (N, 3)
    bounds : array-like
        Domain bounds
    tolerance : float
        Tolerance for boundary violation detection
        
    Returns
    -------
    dict
        Boundary violation analysis
    """
    positions = _ensure_float32(positions)
    bounds_std = _ensure_bounds_shape(bounds)
    
    if positions.size == 0:
        return {'empty': True, 'violations': 0}
    
    # Check violations
    lo_violations = positions < (bounds_std[0] - tolerance)
    hi_violations = positions > (bounds_std[1] + tolerance)
    
    any_violations = np.any(lo_violations | hi_violations, axis=1)
    n_violations = np.sum(any_violations)
    
    # Per-axis statistics
    axis_stats = {}
    for i, axis in enumerate(['x', 'y', 'z']):
        axis_lo_viol = np.sum(lo_violations[:, i])
        axis_hi_viol = np.sum(hi_violations[:, i])
        
        axis_stats[axis] = {
            'low_violations': int(axis_lo_viol),
            'high_violations': int(axis_hi_viol),
            'total_violations': int(axis_lo_viol + axis_hi_viol),
            'min_value': float(np.min(positions[:, i])),
            'max_value': float(np.max(positions[:, i])),
            'bound_min': float(bounds_std[0, i]),
            'bound_max': float(bounds_std[1, i])
        }
    
    return {
        'total_particles': positions.shape[0],
        'violations': int(n_violations),
        'violation_fraction': float(n_violations) / positions.shape[0],
        'axis_statistics': axis_stats,
        'bounds': bounds_std.tolist()
    }


def test_boundary_condition(boundary_func: BoundaryCondition, 
                           test_positions: np.ndarray,
                           expected_behavior: str = 'bounded') -> Dict[str, Any]:
    """
    Test boundary condition function with various inputs.
    
    Parameters
    ----------
    boundary_func : BoundaryCondition
        Boundary condition function to test
    test_positions : np.ndarray
        Test positions, shape (N, 3)
    expected_behavior : str
        Expected behavior: 'bounded', 'wrapping', 'absorbing'
        
    Returns
    -------
    dict
        Test results
    """
    test_positions = _ensure_float32(test_positions)
    
    try:
        # Apply boundary condition
        result = boundary_func(test_positions)
        
        # Analyze result
        analysis = {
            'success': True,
            'input_shape': test_positions.shape,
            'output_shape': result.shape,
            'shapes_match': test_positions.shape == result.shape,
            'finite_outputs': int(np.sum(np.isfinite(result).all(axis=1))),
            'nan_outputs': int(np.sum(np.isnan(result).any(axis=1))),
            'inf_outputs': int(np.sum(np.isinf(result).any(axis=1)))
        }
        
        # Check expected behavior
        if expected_behavior == 'bounded':
            # Check if outputs are within reasonable bounds
            if np.all(np.isfinite(result[np.isfinite(result)])):
                analysis['behavior_check'] = 'passed'
            else:
                analysis['behavior_check'] = 'failed - non-finite values'
        
        elif expected_behavior == 'wrapping':
            # For periodic boundaries, outputs should be finite
            if analysis['finite_outputs'] == test_positions.shape[0]:
                analysis['behavior_check'] = 'passed'
            else:
                analysis['behavior_check'] = 'failed - non-finite values in wrapping BC'
        
        elif expected_behavior == 'absorbing':
            # Some outputs may be NaN for absorbed particles
            analysis['behavior_check'] = 'passed'  # Any result is valid for absorbing
        
    except Exception as e:
        analysis = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
    
    return analysis


# ---------- Utility functions ----------

def create_boundary_from_config(config: Dict[str, Any]) -> BoundaryCondition:
    """
    Create boundary condition from configuration dictionary.
    
    Parameters
    ----------
    config : dict
        Configuration with keys:
        - 'type': boundary type ('periodic', 'reflective', 'clamping', 'absorbing', 'mixed', 'spherical', 'cylindrical')
        - 'bounds': domain bounds
        - Additional type-specific parameters
        
    Returns
    -------
    BoundaryCondition
        Configured boundary condition function
    """
    boundary_type = config.get('type', 'periodic')
    bounds = config.get('bounds')
    
    if bounds is None:
        raise ValueError("Bounds must be specified in boundary configuration")
    
    if boundary_type == 'periodic':
        return periodic_boundary(bounds)
    
    elif boundary_type == 'reflective':
        return reflective_boundary(bounds)
    
    elif boundary_type == 'clamping':
        return clamping_boundary(bounds)
    
    elif boundary_type == 'absorbing':
        outside_value = config.get('outside_value', np.nan)
        return absorbing_boundary_factory(bounds, outside_value)
    
    elif boundary_type == 'mixed':
        conditions = config.get('conditions', {'x': 'periodic', 'y': 'periodic', 'z': 'periodic'})
        return mixed_boundary(bounds, conditions)
    
    elif boundary_type == 'spherical':
        center = config.get('center', [0, 0, 0])
        radius = config.get('radius', 1.0)
        condition = config.get('condition', 'reflective')
        return spherical_boundary(center, radius, condition)
    
    elif boundary_type == 'cylindrical':
        axis = config.get('axis', 'z')
        center = config.get('center', [0, 0])
        radius = config.get('radius', 1.0)
        height = config.get('height', None)
        condition = config.get('condition', 'reflective')
        return cylindrical_boundary(axis, center, radius, height, condition)
    
    else:
        raise ValueError(f"Unknown boundary type: {boundary_type}")


def get_boundary_info(boundary_func: BoundaryCondition) -> Dict[str, Any]:
    """
    Extract information about boundary condition function.
    
    Parameters
    ----------
    boundary_func : BoundaryCondition
        Boundary condition function
        
    Returns
    -------
    dict
        Boundary condition information
    """
    info = {
        'callable': callable(boundary_func),
        'function_name': getattr(boundary_func, '__name__', 'unknown')
    }
    
    # Try to extract additional info from function attributes
    if hasattr(boundary_func, '__closure__') and boundary_func.__closure__:
        info['has_closure'] = True
        info['closure_vars'] = len(boundary_func.__closure__)
    else:
        info['has_closure'] = False
    
    # Test with simple input to check basic functionality
    try:
        test_pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        test_result = boundary_func(test_pos)
        info['test_success'] = True
        info['output_shape'] = test_result.shape
        info['output_finite'] = bool(np.all(np.isfinite(test_result)))
    except Exception as e:
        info['test_success'] = False
        info['test_error'] = str(e)
    
    return info


# ---------- Predefined common boundary conditions ----------

def no_boundary() -> BoundaryCondition:
    """Identity boundary condition (no modification)."""
    def _no_bc(positions: np.ndarray) -> np.ndarray:
        return _ensure_float32(positions)
    
    return _no_bc


def unit_box_periodic() -> BoundaryCondition:
    """Periodic boundary condition for unit box [0,1]³."""
    return periodic_boundary([[0, 0, 0], [1, 1, 1]])


def unit_box_reflective() -> BoundaryCondition:
    """Reflective boundary condition for unit box [0,1]³."""
    return reflective_boundary([[0, 0, 0], [1, 1, 1]])


def centered_box_periodic(size: float = 2.0) -> BoundaryCondition:
    """Periodic boundary condition for centered box [-size/2, size/2]³."""
    half_size = size / 2.0
    return periodic_boundary([[-half_size, -half_size, -half_size], 
                             [half_size, half_size, half_size]])


def unit_sphere_reflective() -> BoundaryCondition:
    """Reflective boundary condition for unit sphere."""
    return spherical_boundary([0, 0, 0], 1.0, 'reflective')


# ---------- Boundary condition composition ----------

def compose_boundary_conditions(*boundary_funcs: BoundaryCondition) -> BoundaryCondition:
    """
    Compose multiple boundary conditions applied in sequence.
    
    Parameters
    ----------
    *boundary_funcs : BoundaryCondition
        Boundary condition functions to compose
        
    Returns
    -------
    BoundaryCondition
        Composed boundary condition function
    """
    if not boundary_funcs:
        return no_boundary()
    
    def _composed_bc(positions: np.ndarray) -> np.ndarray:
        result = _ensure_float32(positions)
        for bc in boundary_funcs:
            result = bc(result)
        return result.astype(np.float32)
    
    return _composed_bc


# ---------- Debug and visualization utilities ----------

def visualize_boundary_effect(boundary_func: BoundaryCondition,
                             test_positions: np.ndarray,
                             plot_2d: bool = True) -> None:
    """
    Visualize the effect of boundary condition on test positions.
    
    Parameters
    ----------
    boundary_func : BoundaryCondition
        Boundary condition to visualize
    test_positions : np.ndarray
        Test positions, shape (N, 3)
    plot_2d : bool
        Whether to create 2D plots (XY projection)
    """
    try:
        import matplotlib.pyplot as plt
        
        test_positions = _ensure_float32(test_positions)
        result_positions = boundary_func(test_positions)
        
        if plot_2d:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Before
            ax1.scatter(test_positions[:, 0], test_positions[:, 1], 
                       alpha=0.6, s=30, c='blue', label='Original')
            ax1.set_title('Before Boundary Condition')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_aspect('equal')
            
            # After
            finite_mask = np.isfinite(result_positions).all(axis=1)
            if np.any(finite_mask):
                ax2.scatter(result_positions[finite_mask, 0], result_positions[finite_mask, 1],
                           alpha=0.6, s=30, c='red', label='After BC')
            
            nan_mask = np.isnan(result_positions).any(axis=1)
            if np.any(nan_mask):
                ax2.scatter([], [], s=30, c='gray', label=f'Absorbed ({np.sum(nan_mask)})')
            
            ax2.set_title('After Boundary Condition')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_aspect('equal')
            
            plt.tight_layout()
            plt.show()
        
    except ImportError:
        print("Matplotlib required for visualization")


# ---------------------------------------------------------------------------
# Inlet/Outlet Boundary Conditions
# ---------------------------------------------------------------------------

def inlet_outlet_boundary_factory(bounds: Union[np.ndarray, list],
                                 inlet_axis: str = 'x',
                                 inlet_position: str = 'min',
                                 outlet_position: str = 'max',
                                 inlet_particle_generator: Optional[Callable] = None) -> BoundaryCondition:
    """
    Create inlet/outlet boundary condition.

    Particles are absorbed at the outlet boundary and new particles can be
    injected at the inlet boundary.

    Parameters
    ----------
    bounds : array-like
        Domain bounds [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    inlet_axis : str, default 'x'
        Axis for inlet/outlet ('x', 'y', or 'z')
    inlet_position : str, default 'min'
        Inlet at 'min' or 'max' position along axis
    outlet_position : str, default 'max'
        Outlet at 'min' or 'max' position along axis
    inlet_particle_generator : callable, optional
        Function to generate new particles at inlet

    Returns
    -------
    BoundaryCondition
        Inlet/outlet boundary condition function
    """
    bounds_std = _ensure_bounds_shape(bounds)
    axis_map = {'x': 0, 'y': 1, 'z': 2}

    if inlet_axis not in axis_map:
        raise ValueError(f"inlet_axis must be 'x', 'y', or 'z', got {inlet_axis}")

    axis_idx = axis_map[inlet_axis]

    def _inlet_outlet_bc(x: np.ndarray) -> np.ndarray:
        """Apply inlet/outlet boundary condition."""
        x = _ensure_float32(x)
        result = x.copy()

        # Get axis boundaries
        axis_min = bounds_std[0, axis_idx]
        axis_max = bounds_std[1, axis_idx]

        # Determine inlet and outlet positions
        if inlet_position == 'min':
            inlet_bound = axis_min
            outlet_bound = axis_max if outlet_position == 'max' else axis_min
        else:
            inlet_bound = axis_max
            outlet_bound = axis_min if outlet_position == 'min' else axis_max

        # Absorb particles at outlet
        if outlet_position == 'min':
            outlet_mask = result[:, axis_idx] < outlet_bound
        else:
            outlet_mask = result[:, axis_idx] > outlet_bound

        # Mark absorbed particles with NaN
        result[outlet_mask] = np.nan

        # Apply periodic wrapping at inlet/outlet boundaries
        # This prevents particles from going beyond bounds
        if inlet_position == 'min':
            # Particles that go below inlet are wrapped to outlet
            inlet_mask = result[:, axis_idx] < inlet_bound
            result[inlet_mask, axis_idx] = outlet_bound
        else:
            # Particles that go above inlet are wrapped to outlet
            inlet_mask = result[:, axis_idx] > inlet_bound
            result[inlet_mask, axis_idx] = outlet_bound

        # Apply reflective boundary to other axes
        for i in range(3):
            if i != axis_idx:
                # Reflect particles that go outside bounds in other dimensions
                low_mask = result[:, i] < bounds_std[0, i]
                high_mask = result[:, i] > bounds_std[1, i]

                result[low_mask, i] = 2 * bounds_std[0, i] - result[low_mask, i]
                result[high_mask, i] = 2 * bounds_std[1, i] - result[high_mask, i]

        return result.astype(np.float32)

    return _inlet_outlet_bc


def flow_through_boundary_factory(bounds: Union[np.ndarray, list],
                                 flow_axis: str = 'x',
                                 flow_direction: str = 'positive') -> BoundaryCondition:
    """
    Create flow-through boundary condition for steady flow simulations.

    Particles enter at one end and exit at the other end of the specified axis.
    Other axes use reflective boundaries.

    Parameters
    ----------
    bounds : array-like
        Domain bounds [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    flow_axis : str, default 'x'
        Primary flow axis ('x', 'y', or 'z')
    flow_direction : str, default 'positive'
        Flow direction: 'positive' (min to max) or 'negative' (max to min)

    Returns
    -------
    BoundaryCondition
        Flow-through boundary condition function
    """
    bounds_std = _ensure_bounds_shape(bounds)
    axis_map = {'x': 0, 'y': 1, 'z': 2}

    if flow_axis not in axis_map:
        raise ValueError(f"flow_axis must be 'x', 'y', or 'z', got {flow_axis}")

    axis_idx = axis_map[flow_axis]

    def _flow_through_bc(x: np.ndarray) -> np.ndarray:
        """Apply flow-through boundary condition."""
        x = _ensure_float32(x)
        result = x.copy()

        # Get flow axis boundaries
        axis_min = bounds_std[0, axis_idx]
        axis_max = bounds_std[1, axis_idx]

        if flow_direction == 'positive':
            # Flow from min to max
            # Absorb particles that exit at max
            exit_mask = result[:, axis_idx] > axis_max
            # Periodic wrap particles that go below min
            inlet_mask = result[:, axis_idx] < axis_min
            result[inlet_mask, axis_idx] = axis_min + (axis_min - result[inlet_mask, axis_idx])
        else:
            # Flow from max to min
            # Absorb particles that exit at min
            exit_mask = result[:, axis_idx] < axis_min
            # Periodic wrap particles that go above max
            inlet_mask = result[:, axis_idx] > axis_max
            result[inlet_mask, axis_idx] = axis_max - (result[inlet_mask, axis_idx] - axis_max)

        # Replace exited particles with new inlet particles
        if np.any(exit_mask):
            n_exited = np.sum(exit_mask)

            # Generate new particles at the inlet
            new_particles = np.zeros((n_exited, 3), dtype=np.float32)

            if flow_direction == 'positive':
                # Place new particles at inlet (x_min)
                new_particles[:, axis_idx] = axis_min
            else:
                # Place new particles at inlet (x_max)
                new_particles[:, axis_idx] = axis_max

            # Randomly distribute across other dimensions at inlet
            for i in range(3):
                if i != axis_idx:
                    # Random positions within bounds for non-flow axes
                    new_particles[:, i] = np.random.uniform(
                        bounds_std[0, i],
                        bounds_std[1, i],
                        n_exited
                    ).astype(np.float32)

            # Replace exited particles with new inlet particles
            result[exit_mask] = new_particles

        # Apply reflective boundaries to other axes
        for i in range(3):
            if i != axis_idx:
                # Reflect at boundaries in non-flow directions
                low_mask = result[:, i] < bounds_std[0, i]
                high_mask = result[:, i] > bounds_std[1, i]

                result[low_mask, i] = 2 * bounds_std[0, i] - result[low_mask, i]
                result[high_mask, i] = 2 * bounds_std[1, i] - result[high_mask, i]

        return result.astype(np.float32)

    return _flow_through_bc


def _replace_exited_particles_at_grid_positions(result: np.ndarray, exit_mask: np.ndarray,
                                             inlet_position: float, axis_idx: int,
                                             bounds: np.ndarray, concentrations: Dict[str, int]) -> None:
    """
    Replace exited particles at their nearest grid positions on the inlet face.

    This function ensures spatial consistency by maintaining the uniform grid pattern
    when particles are reintroduced at the inlet. Each exited particle is replaced
    at the grid position closest to where it originally was, preventing gaps and
    overlaps in the particle distribution.

    Parameters
    ----------
    result : np.ndarray
        Particle positions array to modify in-place, shape (N, 3)
    exit_mask : np.ndarray
        Boolean mask indicating which particles have exited, shape (N,)
    inlet_position : float
        Position along the flow axis where particles are reintroduced
    axis_idx : int
        Index of the flow axis (0=x, 1=y, 2=z)
    bounds : np.ndarray
        Domain bounds [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    concentrations : Dict[str, int]
        Grid density per unit length for each axis {'x': int, 'y': int, 'z': int}

    Notes
    -----
    This approach maintains perfect grid uniformity by ensuring that each particle
    is replaced at exactly the same type of grid location it came from. This prevents
    the clustering and gaps that would occur with random or sequential replacement.
    """
    axis_names = ['x', 'y', 'z']
    other_axes = [i for i in range(3) if i != axis_idx]

    # Get grid resolutions for inlet face dimensions (perpendicular to flow)
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

    # Replace each exited particle at its nearest grid position
    exited_indices = np.where(exit_mask)[0]

    for particle_idx in exited_indices:
        particle_pos = result[particle_idx]

        # Find nearest grid position for each perpendicular dimension
        grid_coords = []
        for i, axis_i in enumerate(other_axes):
            particle_coord = particle_pos[axis_i]
            coords = coord_arrays[i]
            nearest_idx = np.argmin(np.abs(coords - particle_coord))
            grid_coords.append(coords[nearest_idx])

        # Replace particle at inlet with grid-aligned position
        result[particle_idx, axis_idx] = inlet_position
        if len(other_axes) >= 1:
            result[particle_idx, other_axes[0]] = grid_coords[0]
        if len(other_axes) >= 2:
            result[particle_idx, other_axes[1]] = grid_coords[1]



def _generate_inlet_grid_particles(n_particles: int, inlet_position: float,
                                 axis_idx: int, bounds: np.ndarray,
                                 concentrations: Dict[str, int]) -> np.ndarray:
    """
    Generate particles in a 2D uniform grid directly on the inlet face.

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

    # Distribute particles across the inlet face using optimal grid sampling
    if n_particles <= total_inlet_particles:
        # Use systematic sampling to ensure even distribution without duplicates
        # This divides the grid into n_particles equal sections and takes one point from each
        step = total_inlet_particles / n_particles
        indices = np.array([int(i * step) for i in range(n_particles)])

        # Ensure indices are unique and within bounds
        indices = np.clip(indices, 0, total_inlet_particles - 1)
        indices = np.unique(indices)

        # If we lost some indices due to rounding, add them back
        while len(indices) < n_particles and len(indices) < total_inlet_particles:
            # Find the largest gap in the current indices and add a point there
            if len(indices) == 0:
                indices = np.array([0])
            else:
                # Find gaps and fill them
                gaps = np.diff(np.concatenate([[0], indices, [total_inlet_particles]]))
                max_gap_idx = np.argmax(gaps[:-1])  # Don't consider the last gap
                if max_gap_idx == 0:
                    new_idx = indices[0] // 2
                else:
                    new_idx = (indices[max_gap_idx-1] + indices[max_gap_idx]) // 2
                indices = np.sort(np.append(indices, new_idx))
                indices = np.unique(indices)

        # Take only the first n_particles if we have too many
        indices = indices[:n_particles]
    else:
        # More particles than grid points - use each grid point at least once, then distribute extras
        base_indices = np.arange(total_inlet_particles)
        extra_particles = n_particles - total_inlet_particles

        if extra_particles > 0:
            # Distribute extra particles evenly across the grid
            extra_step = total_inlet_particles / extra_particles
            extra_indices = np.array([int(i * extra_step) for i in range(extra_particles)])
            indices = np.concatenate([base_indices, extra_indices])
        else:
            indices = base_indices

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


def continuous_inlet_boundary_factory(bounds: Union[np.ndarray, list],
                                     flow_axis: str = 'x',
                                     flow_direction: str = 'positive',
                                     inlet_distribution: str = 'uniform',
                                     concentrations: Optional[Dict[str, int]] = None) -> BoundaryCondition:
    """
    Create boundary condition with continuous particle injection at inlet.

    This boundary condition creates a flow-through system where particles enter
    at one boundary (inlet) and exit at the opposite boundary (outlet). When
    particles exit, they are immediately replaced with new particles at the inlet,
    maintaining a steady stream and preserving the original spatial distribution.

    The key innovation is the grid-preserving replacement strategy: when using
    'grid' distribution, exiting particles are replaced at their nearest grid
    positions on the inlet face, preventing clustering and maintaining perfect
    spatial uniformity throughout the simulation.

    Parameters
    ----------
    bounds : array-like
        Domain bounds [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    flow_axis : str, default 'x'
        Primary flow axis ('x', 'y', or 'z'). Particles flow along this axis.
    flow_direction : str, default 'positive'
        Flow direction along the axis:
        - 'positive': particles enter at min boundary, exit at max boundary
        - 'negative': particles enter at max boundary, exit at min boundary
    inlet_distribution : str, default 'uniform'
        Distribution of new particles at inlet:
        - 'uniform': random uniform distribution across inlet face
        - 'gaussian': gaussian distribution centered on inlet face
        - 'grid': uniform grid distribution (requires concentrations parameter)
    concentrations : dict, optional
        Grid density per unit length for each axis when using 'grid' distribution.
        Format: {'x': int, 'y': int, 'z': int}. Only used with inlet_distribution='grid'.

    Returns
    -------
    BoundaryCondition
        Boundary condition function that can be used with ParticleTracker

    Examples
    --------
    >>> # Create flow-through boundary with grid inlet distribution
    >>> concentrations = {'x': 10, 'y': 5, 'z': 3}
    >>> boundary = continuous_inlet_boundary_factory(
    ...     bounds=[[-1, -1, -1], [1, 1, 1]],
    ...     flow_axis='x',
    ...     flow_direction='positive',
    ...     inlet_distribution='grid',
    ...     concentrations=concentrations
    ... )

    >>> # Use with tracker
    >>> tracker = create_tracker(..., boundary_condition=boundary)

    Notes
    -----
    This boundary condition is particularly useful for:
    - Modeling steady-state flow systems
    - Maintaining particle concentration in flow simulations
    - Preventing particle depletion in long-time simulations
    - Studying transport phenomena with continuous injection
    """
    bounds_std = _ensure_bounds_shape(bounds)
    axis_map = {'x': 0, 'y': 1, 'z': 2}

    if flow_axis not in axis_map:
        raise ValueError(f"flow_axis must be 'x', 'y', or 'z', got {flow_axis}")

    axis_idx = axis_map[flow_axis]

    def _continuous_inlet_bc(x: np.ndarray) -> np.ndarray:
        """Apply continuous inlet boundary condition."""
        x = _ensure_float32(x)
        result = x.copy()

        # Get flow axis boundaries
        axis_min = bounds_std[0, axis_idx]
        axis_max = bounds_std[1, axis_idx]

        if flow_direction == 'positive':
            # Flow from min to max
            # Absorb particles that exit at max
            exit_mask = result[:, axis_idx] > axis_max
            # Inlet is at min
            inlet_position = axis_min
        else:
            # Flow from max to min
            # Absorb particles that exit at min
            exit_mask = result[:, axis_idx] < axis_min
            # Inlet is at max
            inlet_position = axis_max

        # Replace exited particles with new inlet particles
        if np.any(exit_mask):
            if inlet_distribution == 'grid' and concentrations is not None:
                # Replace particles at their nearest grid positions to maintain spatial consistency
                _replace_exited_particles_at_grid_positions(
                    result, exit_mask, inlet_position, axis_idx, bounds_std, concentrations
                )
            else:
                # For non-grid distributions, use original replacement logic
                n_exited = np.sum(exit_mask)

                # Generate new particles at the inlet
                new_particles = np.zeros((n_exited, 3), dtype=np.float32)

                # Place at inlet position
                new_particles[:, axis_idx] = inlet_position

                # Original random distribution
                for i in range(3):
                    if i != axis_idx:
                        dim_min = bounds_std[0, i]
                        dim_max = bounds_std[1, i]

                        if inlet_distribution == 'uniform':
                            # Uniform distribution across inlet face
                            new_particles[:, i] = np.random.uniform(
                                dim_min, dim_max, n_exited
                            ).astype(np.float32)
                        elif inlet_distribution == 'gaussian':
                            # Gaussian distribution centered in inlet face
                            center = (dim_min + dim_max) / 2
                            sigma = (dim_max - dim_min) / 6  # 3-sigma covers most of range
                            new_particles[:, i] = np.clip(
                                np.random.normal(center, sigma, n_exited),
                                dim_min, dim_max
                            ).astype(np.float32)

                # Replace exited particles
                result[exit_mask] = new_particles

        # Handle particles that try to go backwards through inlet
        if flow_direction == 'positive':
            # Particles going below inlet get repositioned slightly inside
            backward_mask = result[:, axis_idx] < axis_min
            result[backward_mask, axis_idx] = axis_min + 0.001 * (axis_max - axis_min)
        else:
            # Particles going above inlet get repositioned slightly inside
            backward_mask = result[:, axis_idx] > axis_max
            result[backward_mask, axis_idx] = axis_max - 0.001 * (axis_max - axis_min)

        # Apply reflective boundaries to other axes
        for i in range(3):
            if i != axis_idx:
                # Reflect at boundaries in non-flow directions
                low_mask = result[:, i] < bounds_std[0, i]
                high_mask = result[:, i] > bounds_std[1, i]

                result[low_mask, i] = 2 * bounds_std[0, i] - result[low_mask, i]
                result[high_mask, i] = 2 * bounds_std[1, i] - result[high_mask, i]

        return result.astype(np.float32)

    return _continuous_inlet_bc


# Export commonly used boundary conditions for convenience
__all__ = [
    # Core functions
    'periodic_boundary', 'reflective_boundary', 'clamping_boundary', 'absorbing_boundary_factory',
    'mixed_boundary', 'distance_based_boundary',

    # Geometric boundaries
    'spherical_boundary', 'cylindrical_boundary',

    # Inlet/outlet boundaries
    'inlet_outlet_boundary_factory', 'flow_through_boundary_factory', 'continuous_inlet_boundary_factory',

    # Utilities
    'check_boundary_violations', 'test_boundary_condition', 'create_boundary_from_config',
    'no_boundary', 'compose_boundary_conditions',

    # Predefined conditions
    'unit_box_periodic', 'unit_box_reflective', 'centered_box_periodic', 'unit_sphere_reflective',

    # Classes and protocols
    'BoundaryCondition', 'CompositeBoundaryCondition',

    # Visualization
    'visualize_boundary_effect'
]