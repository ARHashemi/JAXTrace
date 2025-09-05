# jaxtrace/fields/base.py
"""
Base protocols and utilities for spatial fields.

Defines the Field and TimeDependentField protocols, grid metadata,
and barycentric coordinate helpers for unstructured meshes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Tuple
import numpy as np

# Import JAX utilities for array handling
from ..utils.jax_utils import JAX_AVAILABLE

if JAX_AVAILABLE:
    try:
        import jax.numpy as jnp
    except Exception:
        JAX_AVAILABLE = False

if not JAX_AVAILABLE:
    import numpy as jnp  # type: ignore


@dataclass(frozen=True)
class GridMeta:
    """Structured grid metadata."""
    origin: jnp.ndarray   # (3,) grid origin coordinates  
    spacing: jnp.ndarray  # (3,) grid spacing dx,dy,dz
    shape: Tuple[int, int, int]  # (Nx, Ny, Nz) grid dimensions
    bounds: jnp.ndarray   # (2, 3) [[xmin,ymin,zmin], [xmax,ymax,zmax]]


class BaseField(Protocol):
    """
    Base protocol for spatial fields.
    
    This is the base protocol that all field implementations should follow.
    Provides consistent interface for spatial sampling of velocity fields.
    """
    
    def sample(self, positions: jnp.ndarray) -> jnp.ndarray:
        """
        Sample field at positions.

        Parameters
        ----------
        positions : jnp.ndarray
            Positions to sample, shape (N, 3)
            
        Returns
        -------
        jnp.ndarray
            Field values, shape (N, C) where C is number of components
        """
        ...
    
    def get_spatial_bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Return spatial bounds of the field domain.
        
        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            (bounds_min, bounds_max) each shape (3,) as [xmin,ymin,zmin], [xmax,ymax,zmax]
        """
        ...


class Field(Protocol):
    """
    Protocol for static spatial fields.
    
    Implementations should provide efficient spatial sampling
    of scalar or vector fields on static topology.
    """
    
    def sample(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Sample field at positions x.

        Parameters
        ----------
        x : jnp.ndarray
            Positions to sample, shape (N, 3)
            
        Returns
        -------
        jnp.ndarray
            Field values, shape (N, C) where C is number of components
        """
        ...


class TimeDependentField(Protocol):
    """
    Protocol for time-dependent fields with temporal interpolation.
    
    Implementations should handle temporal interpolation between
    available time slices and provide spatial and temporal bounds.
    """
    
    def sample_at_positions(self, positions: jnp.ndarray, t: float) -> jnp.ndarray:
        """
        Sample field at positions and time t.

        Parameters
        ----------
        positions : jnp.ndarray
            Positions to sample, shape (N, 3)
        t : float
            Time value for sampling
            
        Returns
        -------
        jnp.ndarray
            Field values, shape (N, 3) for velocity fields
        """
        ...
    
    def sample_at_time(self, t: float) -> jnp.ndarray:
        """
        Sample field at specific time across all grid points.

        Parameters
        ----------
        t : float
            Time value for sampling
            
        Returns
        -------
        jnp.ndarray
            Field values at all grid points, shape (N, 3)
        """
        ...

    def get_spatial_bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Return spatial bounds of the field domain.
        
        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            (bounds_min, bounds_max) each shape (3,) as [xmin,ymin,zmin], [xmax,ymax,zmax]
        """
        ...
    
    def get_time_bounds(self) -> Tuple[float, float]:
        """
        Return temporal bounds of the field data.
        
        Returns
        -------
        Tuple[float, float]
            (t_min, t_max) time range
        """
        ...


# ---------- Barycentric coordinate helpers for unstructured meshes ----------

def barycentric_coords_triangle(
    p: jnp.ndarray, 
    a: jnp.ndarray, 
    b: jnp.ndarray, 
    c: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute barycentric coordinates of point p in triangle (a,b,c).
    
    Works in 2D or 3D (for planar triangles). Uses numerically stable method.
    
    Parameters
    ----------
    p : jnp.ndarray
        Query point, shape (3,)
    a, b, c : jnp.ndarray
        Triangle vertices, each shape (3,)
        
    Returns
    -------
    jnp.ndarray
        Barycentric coordinates [λ1, λ2, λ3], shape (3,)
        where p = λ1*a + λ2*b + λ3*c and λ1 + λ2 + λ3 = 1
    """
    v0 = b - a
    v1 = c - a  
    v2 = p - a
    
    d00 = jnp.dot(v0, v0)
    d01 = jnp.dot(v0, v1)
    d11 = jnp.dot(v1, v1)
    d20 = jnp.dot(v2, v0)
    d21 = jnp.dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    inv_denom = jnp.where(jnp.abs(denom) > 1e-15, 1.0 / denom, 0.0)
    
    v = (d11 * d20 - d01 * d21) * inv_denom
    w = (d00 * d21 - d01 * d20) * inv_denom
    u = 1.0 - v - w
    
    return jnp.stack([u, v, w], axis=0)


def barycentric_coords_tetrahedron(
    p: jnp.ndarray,
    a: jnp.ndarray, 
    b: jnp.ndarray,
    c: jnp.ndarray, 
    d: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute barycentric coordinates of point p in tetrahedron (a,b,c,d).
    
    Parameters
    ----------
    p : jnp.ndarray
        Query point, shape (3,)
    a, b, c, d : jnp.ndarray  
        Tetrahedron vertices, each shape (3,)
        
    Returns
    -------
    jnp.ndarray
        Barycentric coordinates [λ1, λ2, λ3, λ4], shape (4,)
        where p = λ1*a + λ2*b + λ3*c + λ4*d and sum(λ) = 1
    """
    # Set up system: [b-a, c-a, d-a] * [λ2, λ3, λ4]^T = p-a
    M = jnp.stack([b - a, c - a, d - a], axis=1)  # shape (3, 3)
    rhs = p - a  # shape (3,)
    
    # Solve using pseudo-inverse for numerical stability
    try:
        coeffs = jnp.linalg.solve(M, rhs)  # [λ2, λ3, λ4]
    except:
        # Fallback to pseudo-inverse if singular
        coeffs = jnp.linalg.pinv(M) @ rhs
        
    l2, l3, l4 = coeffs
    l1 = 1.0 - (l2 + l3 + l4)
    
    return jnp.stack([l1, l2, l3, l4], axis=0)


# ---------- High-order shape functions for quadratic elements ----------

def tri6_shape_functions(barycentric: jnp.ndarray) -> jnp.ndarray:
    """
    Compute 6-node quadratic triangle shape functions from barycentric coordinates.
    
    Node ordering: [v1, v2, v3, e12, e23, e31] where e_ij is midpoint of edge i-j.
    
    Parameters
    ----------
    barycentric : jnp.ndarray
        Barycentric coordinates [λ1, λ2, λ3], shape (3,)
        
    Returns
    -------
    jnp.ndarray
        Shape function values [N1, ..., N6], shape (6,)
    """
    l1, l2, l3 = barycentric
    
    return jnp.stack([
        l1 * (2.0 * l1 - 1.0),      # Corner node 1
        l2 * (2.0 * l2 - 1.0),      # Corner node 2  
        l3 * (2.0 * l3 - 1.0),      # Corner node 3
        4.0 * l1 * l2,               # Edge node 1-2
        4.0 * l2 * l3,               # Edge node 2-3
        4.0 * l3 * l1                # Edge node 3-1
    ], axis=0)


def tet10_shape_functions(barycentric: jnp.ndarray) -> jnp.ndarray:
    """
    Compute 10-node quadratic tetrahedron shape functions from barycentric coordinates.
    
    Node ordering: [v1, v2, v3, v4, e12, e23, e31, e14, e24, e34]
    where e_ij is midpoint of edge i-j.
    
    Parameters
    ----------
    barycentric : jnp.ndarray
        Barycentric coordinates [λ1, λ2, λ3, λ4], shape (4,)
        
    Returns
    -------
    jnp.ndarray
        Shape function values [N1, ..., N10], shape (10,)
    """
    l1, l2, l3, l4 = barycentric
    
    return jnp.stack([
        l1 * (2.0 * l1 - 1.0),      # Corner node 1
        l2 * (2.0 * l2 - 1.0),      # Corner node 2
        l3 * (2.0 * l3 - 1.0),      # Corner node 3  
        l4 * (2.0 * l4 - 1.0),      # Corner node 4
        4.0 * l1 * l2,               # Edge node 1-2
        4.0 * l2 * l3,               # Edge node 2-3
        4.0 * l3 * l1,               # Edge node 3-1
        4.0 * l1 * l4,               # Edge node 1-4
        4.0 * l2 * l4,               # Edge node 2-4
        4.0 * l3 * l4                # Edge node 3-4
    ], axis=0)


# ---------- Additional utilities for field implementations ----------

def _ensure_float32(data: np.ndarray) -> np.ndarray:
    """Convert data to float32 for consistency and JAX performance."""
    return np.asarray(data, dtype=np.float32)


def _ensure_positions_shape(positions: np.ndarray) -> np.ndarray:
    """
    Ensure positions have shape (N, 3) regardless of 2D/3D analysis.
    
    Parameters
    ----------
    positions : np.ndarray
        Input positions, shape (N, 2) or (N, 3)
        
    Returns
    -------
    np.ndarray
        Positions with shape (N, 3), float32 dtype
    """
    pos = _ensure_float32(positions)
    
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)  # Single point case
    
    if pos.ndim != 2:
        raise ValueError(f"Positions must be 2D array, got shape {pos.shape}")
    
    if pos.shape[1] == 2:
        # Convert 2D to 3D by adding zero z-coordinate
        z_zeros = np.zeros((pos.shape[0], 1), dtype=np.float32)
        pos = np.concatenate([pos, z_zeros], axis=1)
    elif pos.shape[1] == 3:
        # Already 3D, ensure float32
        pos = pos.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Positions must have 2 or 3 columns, got {pos.shape[1]}")
    
    return pos