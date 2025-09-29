# jaxtrace/fields/structured.py  
"""  
Structured grid velocity field sampling with consistent data types.  

Provides efficient trilinear interpolation on regular grids with float32  
optimization and (N,3) shape enforcement. Includes a JAX-native sampler  
and a NumPy fallback, plus boundary handling.  

Boundary modes:  
- 'clamp'    : clamp to boundary cells  
- 'zero'     : return zero velocity for out-of-bounds  
- 'nan'      : return NaN velocity for out-of-bounds  
- 'periodic' : wrap indices periodically  
"""  

from __future__ import annotations  

from dataclasses import dataclass  
from typing import Tuple, Optional  
import numpy as np  

# Import JAX utilities with fallback  
from ..utils.jax_utils import JAX_AVAILABLE  

if JAX_AVAILABLE:  
    try:  
        import jax  
        import jax.numpy as jnp  
        from jax import jit  
    except Exception:  
        JAX_AVAILABLE = False  

if not JAX_AVAILABLE:  
    import numpy as jnp  # type: ignore  

    # Mock jit decorator  
    def jit(func):  
        return func  

from .base import BaseField, GridMeta, _ensure_float32, _ensure_positions_shape  


# ------------------------- Internal helpers -------------------------  

def _ensure_positions_shape_jax(positions: "jnp.ndarray") -> "jnp.ndarray":  
    """  
    Ensure positions have shape (N, 3) using jax.numpy only.  
    Do not call this in NumPy mode; use _ensure_positions_shape instead.  
    """  
    pos = jnp.asarray(positions, dtype=jnp.float32)  
    if pos.ndim == 1:  
        pos = pos.reshape((1, pos.shape[0]))  
    if pos.ndim != 2:  
        raise ValueError(f"Positions must be 2D array, got shape {pos.shape}")  
    if pos.shape[1] == 2:  
        z = jnp.zeros((pos.shape[0], 1), dtype=jnp.float32)  
        pos = jnp.concatenate([pos, z], axis=1)  
    elif pos.shape[1] != 3:  
        raise ValueError(f"Positions must have 2 or 3 columns, got {pos.shape[1]}")  
    return pos  


# ------------------------- Main class -------------------------  

@dataclass  
class StructuredGridSampler(BaseField):  
    """  
    Structured grid field with trilinear interpolation.  

    Attributes  
    ----------  
    velocity_data : np.ndarray  
        Velocity field on grid, shape (Nx, Ny, Nz, 3), float32  
    grid_x : np.ndarray  
        X coordinates, shape (Nx,), float32  
    grid_y : np.ndarray  
        Y coordinates, shape (Ny,), float32  
    grid_z : np.ndarray  
        Z coordinates, shape (Nz,), float32  
    bounds_handling : str  
        Boundary mode: 'clamp' | 'zero' | 'nan' | 'periodic'  
    grid_meta : GridMeta, optional  
        Precomputed grid metadata  
    """  
    velocity_data: np.ndarray  # (Nx, Ny, Nz, 3)  
    grid_x: np.ndarray         # (Nx,)  
    grid_y: np.ndarray         # (Ny,)  
    grid_z: np.ndarray         # (Nz,)  
    bounds_handling: str = "clamp"  # 'clamp' | 'zero' | 'nan' | 'periodic'  
    grid_meta: Optional[GridMeta] = None  

    # Device copies for JAX path  
    _vel_dev: Optional["jnp.ndarray"] = None  
    _gx_dev: Optional["jnp.ndarray"] = None  
    _gy_dev: Optional["jnp.ndarray"] = None  
    _gz_dev: Optional["jnp.ndarray"] = None  

    # Precompiled JAX sampler  
    _sampler_jax: Optional[callable] = None  

    def __post_init__(self):  
        # Convert all data to consistent float32  
        self.velocity_data = _ensure_float32(self.velocity_data)  
        self.grid_x = _ensure_float32(self.grid_x)  
        self.grid_y = _ensure_float32(self.grid_y)  
        self.grid_z = _ensure_float32(self.grid_z)  

        # Validate shapes  
        if self.velocity_data.ndim != 4:  
            raise ValueError(f"velocity_data must be 4D (Nx,Ny,Nz,3), got {self.velocity_data.shape}")  

        Nx, Ny, Nz, D = self.velocity_data.shape  
        if D != 3:  
            raise ValueError(f"velocity_data must have 3 components, got {D}")  

        if self.grid_x.shape != (Nx,):  
            raise ValueError(f"grid_x shape {self.grid_x.shape} doesn't match velocity_data Nx={Nx}")  
        if self.grid_y.shape != (Ny,):  
            raise ValueError(f"grid_y shape {self.grid_y.shape} doesn't match velocity_data Ny={Ny}")  
        if self.grid_z.shape != (Nz,):  
            raise ValueError(f"grid_z shape {self.grid_z.shape} doesn't match velocity_data Nz={Nz}")  

        # Store grid metadata  
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz  
        self.dx = float(self.grid_x[1] - self.grid_x[0]) if Nx > 1 else 1.0  
        self.dy = float(self.grid_y[1] - self.grid_y[0]) if Ny > 1 else 1.0  
        self.dz = float(self.grid_z[1] - self.grid_z[0]) if Nz > 1 else 1.0  

        # Bounds (float range in physical coordinates)  
        self.x_min, self.x_max = float(self.grid_x[0]), float(self.grid_x[-1])  
        self.y_min, self.y_max = float(self.grid_y[0]), float(self.grid_y[-1])  
        self.z_min, self.z_max = float(self.grid_z[0]), float(self.grid_z[-1])  

        # Create grid metadata if not provided  
        if self.grid_meta is None:  
            self.grid_meta = GridMeta(  
                origin=jnp.array([self.x_min, self.y_min, self.z_min], dtype=jnp.float32),  
                spacing=jnp.array([self.dx, self.dy, self.dz], dtype=jnp.float32),  
                shape=(Nx, Ny, Nz),  
                bounds=jnp.array([[self.x_min, self.y_min, self.z_min],  
                                  [self.x_max, self.y_max, self.z_max]], dtype=jnp.float32)  
            )  

        # Prepare JAX device arrays and compile kernel if JAX available  
        if JAX_AVAILABLE:  
            # Device-resident arrays to avoid host-device transfers per call  
            self._vel_dev = jax.device_put(jnp.asarray(self.velocity_data, dtype=jnp.float32))  
            self._gx_dev = jax.device_put(jnp.asarray(self.grid_x, dtype=jnp.float32))  
            self._gy_dev = jax.device_put(jnp.asarray(self.grid_y, dtype=jnp.float32))  
            self._gz_dev = jax.device_put(jnp.asarray(self.grid_z, dtype=jnp.float32))  

            # Pre-capture constants for JIT  
            self._dx_j = jnp.asarray(self.dx, dtype=jnp.float32)  
            self._dy_j = jnp.asarray(self.dy, dtype=jnp.float32)  
            self._dz_j = jnp.asarray(self.dz, dtype=jnp.float32)  
            self._x_min_j = jnp.asarray(self.x_min, dtype=jnp.float32)  
            self._y_min_j = jnp.asarray(self.y_min, dtype=jnp.float32)  
            self._z_min_j = jnp.asarray(self.z_min, dtype=jnp.float32)  
            self._Nx_j = jnp.asarray(self.Nx, dtype=jnp.int32)  
            self._Ny_j = jnp.asarray(self.Ny, dtype=jnp.int32)  
            self._Nz_j = jnp.asarray(self.Nz, dtype=jnp.int32)  

            mode = self.bounds_handling.lower()  
            if mode not in ("clamp", "zero", "nan", "periodic"):  
                raise ValueError(f"Unsupported bounds_handling: {self.bounds_handling}")  

            @jit  
            def _trilinear_jax(positions: jnp.ndarray) -> jnp.ndarray:  
                # Ensure shape (N,3)  
                pos = _ensure_positions_shape_jax(positions)  
                x = pos[:, 0]  
                y = pos[:, 1]  
                z = pos[:, 2]  

                # Compute continuous indices in grid space  
                fx = (x - self._x_min_j) / self._dx_j  
                fy = (y - self._y_min_j) / self._dy_j  
                fz = (z - self._z_min_j) / self._dz_j  

                # Bounds handling  
                if mode == "periodic":  
                    fx = fx % self._Nx_j  
                    fy = fy % self._Ny_j  
                    fz = fz % self._Nz_j  
                    valid_mask = jnp.ones_like(x, dtype=bool)  
                else:  
                    # Valid if within physical bounds (inclusive)  
                    valid_x = (x >= self._x_min_j) & (x <= self._x_min_j + (self._Nx_j - 1) * self._dx_j)  
                    valid_y = (y >= self._y_min_j) & (y <= self._y_min_j + (self._Ny_j - 1) * self._dy_j)  
                    valid_z = (z >= self._z_min_j) & (z <= self._z_min_j + (self._Nz_j - 1) * self._dz_j)  
                    valid_mask = valid_x & valid_y & valid_z  

                    if mode == "clamp":  
                        fx = jnp.clip(fx, 0.0, jnp.maximum(self._Nx_j - 1, 0))  
                        fy = jnp.clip(fy, 0.0, jnp.maximum(self._Ny_j - 1, 0))  
                        fz = jnp.clip(fz, 0.0, jnp.maximum(self._Nz_j - 1, 0))  
                    else:  
                        # For zero/nan, we can still clamp indices to avoid OOB on gather  
                        fx = jnp.clip(fx, 0.0, jnp.maximum(self._Nx_j - 1, 0))  
                        fy = jnp.clip(fy, 0.0, jnp.maximum(self._Ny_j - 1, 0))  
                        fz = jnp.clip(fz, 0.0, jnp.maximum(self._Nz_j - 1, 0))  

                # Integer neighbors  
                i0 = jnp.floor(fx).astype(jnp.int32)  
                j0 = jnp.floor(fy).astype(jnp.int32)  
                k0 = jnp.floor(fz).astype(jnp.int32)  

                if mode == "periodic":  
                    i1 = (i0 + 1) % self._Nx_j  
                    j1 = (j0 + 1) % self._Ny_j  
                    k1 = (k0 + 1) % self._Nz_j  
                else:  
                    i1 = jnp.clip(i0 + 1, 0, jnp.maximum(self._Nx_j - 1, 0))  
                    j1 = jnp.clip(j0 + 1, 0, jnp.maximum(self._Ny_j - 1, 0))  
                    k1 = jnp.clip(k0 + 1, 0, jnp.maximum(self._Nz_j - 1, 0))  

                # Fractional weights  
                wx = fx - i0.astype(jnp.float32)  
                wy = fy - j0.astype(jnp.float32)  
                wz = fz - k0.astype(jnp.float32)  

                # Gather corner velocities (elementwise indexing)  
                V = self._vel_dev  # (Nx, Ny, Nz, 3)  
                v000 = V[i0, j0, k0]  
                v100 = V[i1, j0, k0]  
                v010 = V[i0, j1, k0]  
                v110 = V[i1, j1, k0]  
                v001 = V[i0, j0, k1]  
                v101 = V[i1, j0, k1]  
                v011 = V[i0, j1, k1]  
                v111 = V[i1, j1, k1]  

                # Trilinear interpolation  
                c00 = v000 * (1.0 - wx)[:, None] + v100 * wx[:, None]  
                c10 = v010 * (1.0 - wx)[:, None] + v110 * wx[:, None]  
                c01 = v001 * (1.0 - wx)[:, None] + v101 * wx[:, None]  
                c11 = v011 * (1.0 - wx)[:, None] + v111 * wx[:, None]  

                c0 = c00 * (1.0 - wy)[:, None] + c10 * wy[:, None]  
                c1 = c01 * (1.0 - wy)[:, None] + c11 * wy[:, None]  

                vals = c0 * (1.0 - wz)[:, None] + c1 * wz[:, None]  # (N, 3)  

                if mode == "zero":  
                    vals = jnp.where(valid_mask[:, None], vals, jnp.zeros_like(vals))  
                elif mode == "nan":  
                    vals = jnp.where(valid_mask[:, None], vals, jnp.full_like(vals, jnp.nan))  
                # clamp and periodic already handled  

                return vals  

            self._sampler_jax = _trilinear_jax  
        else:  
            self._sampler_jax = None  # NumPy path  

    # ------------------------- Public API -------------------------  

    def sample(self, positions: np.ndarray) -> "jnp.ndarray":  
        """  
        Sample the velocity field at given positions.  

        Parameters  
        ----------  
        positions : array-like, shape (N, 2|3)  
            Query positions.  

        Returns  
        -------  
        array-like  
            Velocity vectors at positions, shape (N, 3).  
            - If JAX is available, returns a jax.numpy array (jnp.ndarray).  
            - Otherwise, returns a numpy array (np.ndarray).  
        """  
        mode = self.bounds_handling.lower()  
        if mode not in ("clamp", "zero", "nan", "periodic"):  
            raise ValueError(f"Unsupported bounds_handling: {self.bounds_handling}")  

        if JAX_AVAILABLE and self._sampler_jax is not None:  
            pos = _ensure_positions_shape_jax(positions)  
            return self._sampler_jax(pos)  
        else:  
            # NumPy fallback  
            pos = _ensure_positions_shape(positions)  
            x = pos[:, 0]  
            y = pos[:, 1]  
            z = pos[:, 2]  

            # Convert to continuous grid indices  
            fx = (x - self.x_min) / self.dx  
            fy = (y - self.y_min) / self.dy  
            fz = (z - self.z_min) / self.dz  

            if mode == "periodic":  
                fx = np.mod(fx, max(self.Nx, 1))  
                fy = np.mod(fy, max(self.Ny, 1))  
                fz = np.mod(fz, max(self.Nz, 1))  
                valid_mask = np.ones_like(x, dtype=bool)  
            else:  
                valid_x = (x >= self.x_min) & (x <= self.x_min + (self.Nx - 1) * self.dx)  
                valid_y = (y >= self.y_min) & (y <= self.y_min + (self.Ny - 1) * self.dy)  
                valid_z = (z >= self.z_min) & (z <= self.z_min + (self.Nz - 1) * self.dz)  
                valid_mask = valid_x & valid_y & valid_z  

                fx = np.clip(fx, 0.0, max(self.Nx - 1, 0))  
                fy = np.clip(fy, 0.0, max(self.Ny - 1, 0))  
                fz = np.clip(fz, 0.0, max(self.Nz - 1, 0))  

            i0 = np.floor(fx).astype(np.int32)  
            j0 = np.floor(fy).astype(np.int32)  
            k0 = np.floor(fz).astype(np.int32)  
            if mode == "periodic":  
                i1 = (i0 + 1) % max(self.Nx, 1)  
                j1 = (j0 + 1) % max(self.Ny, 1)  
                k1 = (k0 + 1) % max(self.Nz, 1)  
            else:  
                i1 = np.clip(i0 + 1, 0, max(self.Nx - 1, 0))
                j1 = np.clip(j0 + 1, 0, max(self.Ny - 1, 0))
                k1 = np.clip(k0 + 1, 0, max(self.Nz - 1, 0))

            wx = fx - i0.astype(np.float32)
            wy = fy - j0.astype(np.float32)
            wz = fz - k0.astype(np.float32)

            V = self.velocity_data  # (Nx, Ny, Nz, 3)
            v000 = V[i0, j0, k0]
            v100 = V[i1, j0, k0]
            v010 = V[i0, j1, k0]
            v110 = V[i1, j1, k0]
            v001 = V[i0, j0, k1]
            v101 = V[i1, j0, k1]
            v011 = V[i0, j1, k1]
            v111 = V[i1, j1, k1]

            c00 = v000 * (1.0 - wx)[:, None] + v100 * wx[:, None]
            c10 = v010 * (1.0 - wx)[:, None] + v110 * wx[:, None]
            c01 = v001 * (1.0 - wx)[:, None] + v101 * wx[:, None]
            c11 = v011 * (1.0 - wx)[:, None] + v111 * wx[:, None]

            c0 = c00 * (1.0 - wy)[:, None] + c10 * wy[:, None]
            c1 = c01 * (1.0 - wy)[:, None] + c11 * wy[:, None]

            vals = c0 * (1.0 - wz)[:, None] + c1 * wz[:, None]  # (N, 3)

            if mode == "zero":
                vals[~valid_mask] = 0.0
            elif mode == "nan":
                vals[~valid_mask] = np.nan

            return vals

    def get_spatial_bounds(self) -> Tuple["jnp.ndarray", "jnp.ndarray"]:
        """
        Return spatial bounds as (min, max), each shape (3,).
        """
        bounds_min = jnp.asarray([self.x_min, self.y_min, self.z_min], dtype=jnp.float32)
        bounds_max = jnp.asarray([self.x_max, self.y_max, self.z_max], dtype=jnp.float32)
        return bounds_min, bounds_max
    
# ------------------------- Factory functions -------------------------
def create_structured_field_from_arrays(
    velocity_data: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: Optional[np.ndarray] = None,
    **kwargs
) -> StructuredGridSampler:
    """
    Create structured field from coordinate arrays.
    
    Parameters
    ----------
    velocity_data : np.ndarray
        Velocity field, shape (Nx, Ny, [Nz,] 3)
    x_coords : np.ndarray
        X coordinates, shape (Nx,)
    y_coords : np.ndarray
        Y coordinates, shape (Ny,)
    z_coords : np.ndarray, optional
        Z coordinates, shape (Nz,). If None, creates 2D field with single Z layer.
    **kwargs
        Additional StructuredGridSampler arguments
        
    Returns
    -------
    StructuredGridSampler
        Configured structured field
    """
    velocity_data = _ensure_float32(velocity_data)
    x_coords = _ensure_float32(x_coords)
    y_coords = _ensure_float32(y_coords)
    
    if z_coords is None:
        # Create 2D field by adding singleton Z dimension
        if velocity_data.ndim == 3:  # (Nx, Ny, 3)
            velocity_data = velocity_data[:, :, None, :]  # (Nx, Ny, 1, 3)
        z_coords = np.array([0.0], dtype=np.float32)
    else:
        z_coords = _ensure_float32(z_coords)
    
    return StructuredGridSampler(
        velocity_data=velocity_data,
        grid_x=x_coords,
        grid_y=y_coords,
        grid_z=z_coords,
        **kwargs
    )


def create_structured_field_from_vtk_data(
    vtk_data: dict,
    velocity_field_name: str = "velocity",
    **kwargs
) -> StructuredGridSampler:
    """
    Create structured field from VTK structured grid data.
    
    Parameters
    ----------
    vtk_data : dict
        VTK data dictionary containing 'dimensions', 'origin', 'spacing', and field data
    velocity_field_name : str
        Name of velocity field in VTK data
    **kwargs
        Additional StructuredGridSampler arguments
        
    Returns
    -------
    StructuredGridSampler
        Configured structured field
    """
    if 'dimensions' not in vtk_data:
        raise ValueError("VTK data must contain 'dimensions'")
    
    if 'origin' not in vtk_data:
        raise ValueError("VTK data must contain 'origin'")
    
    if 'spacing' not in vtk_data:
        raise ValueError("VTK data must contain 'spacing'")
    
    if velocity_field_name not in vtk_data:
        raise ValueError(f"VTK data must contain velocity field '{velocity_field_name}'")
    
    # Extract grid parameters
    dims = vtk_data['dimensions']  # (Nx, Ny, Nz)
    origin = vtk_data['origin']    # (x0, y0, z0)
    spacing = vtk_data['spacing']  # (dx, dy, dz)
    
    Nx, Ny, Nz = dims
    
    # Create coordinate arrays
    x_coords = np.linspace(origin[0], origin[0] + (Nx-1)*spacing[0], Nx, dtype=np.float32)
    y_coords = np.linspace(origin[1], origin[1] + (Ny-1)*spacing[1], Ny, dtype=np.float32)
    z_coords = np.linspace(origin[2], origin[2] + (Nz-1)*spacing[2], Nz, dtype=np.float32)
    
    # Extract velocity data and reshape to grid format
    velocity_data = vtk_data[velocity_field_name]
    if velocity_data.ndim == 2:  # (N_points, 3)
        # Reshape to (Nx, Ny, Nz, 3)
        velocity_data = velocity_data.reshape(Nx, Ny, Nz, 3)
    elif velocity_data.ndim == 4:  # Already in correct format
        pass
    else:
        raise ValueError(f"Unexpected velocity data shape: {velocity_data.shape}")
    
    return StructuredGridSampler(
        velocity_data=velocity_data,
        grid_x=x_coords,
        grid_y=y_coords,
        grid_z=z_coords,
        **kwargs
    )


def create_uniform_grid(
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    resolution: Union[int, Tuple[int, int, int]],
    velocity_function: callable,
    **kwargs
) -> StructuredGridSampler:
    """
    Create uniform structured grid with analytical velocity function.
    
    Parameters
    ----------
    bounds : tuple
        Grid bounds ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    resolution : int or tuple
        Grid resolution. If int, uses same resolution for all dimensions.
    velocity_function : callable
        Function(x, y, z) -> (u, v, w) that returns velocity at coordinates
    **kwargs
        Additional StructuredGridSampler arguments
        
    Returns
    -------
    StructuredGridSampler
        Configured structured field
    """
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    
    if isinstance(resolution, int):
        Nx = Ny = Nz = resolution
    else:
        Nx, Ny, Nz = resolution
    
    # Create coordinate arrays
    x_coords = np.linspace(x_min, x_max, Nx, dtype=np.float32)
    y_coords = np.linspace(y_min, y_max, Ny, dtype=np.float32)
    z_coords = np.linspace(z_min, z_max, Nz, dtype=np.float32)
    
    # Create meshgrid and evaluate velocity function
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    
    # Evaluate velocity function at all grid points
    velocity_data = np.zeros((Nx, Ny, Nz, 3), dtype=np.float32)
    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                x, y, z = X[i, j, k], Y[i, j, k], Z[i, j, k]
                velocity_data[i, j, k, :] = velocity_function(x, y, z)
    
    return StructuredGridSampler(
        velocity_data=velocity_data,
        grid_x=x_coords,
        grid_y=y_coords,
        grid_z=z_coords,
        **kwargs
    )