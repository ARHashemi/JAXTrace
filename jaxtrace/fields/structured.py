# jaxtrace/fields/structured.py  
"""  
Structured grid velocity field sampling with consistent data types.  

Provides efficient interpolation on regular grids with float32  
optimization and (N,3) shape enforcement, enhanced VTK integration,  
and comprehensive boundary handling.  
"""  

from __future__ import annotations  
from dataclasses import dataclass  
from typing import Tuple, Optional, Union  
import numpy as np  

# Import JAX utilities with fallback  
from ..utils.jax_utils import JAX_AVAILABLE  

if JAX_AVAILABLE:  
    try:  
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


@dataclass  
class StructuredGridSampler(BaseField):  
    """  
    Structured grid field with trilinear interpolation.  
    
    Optimized for regular grids with consistent float32 data types  
    and high-performance JAX interpolation when available.  
    Enhanced with comprehensive boundary handling and VTK integration.  
    
    Attributes  
    ----------  
    velocity_data : np.ndarray  
        Velocity field on grid, shape (Nx, Ny, Nz, 3) - standardized to float32  
    grid_x : np.ndarray  
        X coordinates, shape (Nx,) - standardized to float32  
    grid_y : np.ndarray  
        Y coordinates, shape (Ny,) - standardized to float32  
    grid_z : np.ndarray  
        Z coordinates, shape (Nz,) - standardized to float32  
    bounds_handling : str  
        Boundary condition for out-of-bounds points: 'clamp' | 'zero' | 'nan' | 'periodic'  
    grid_meta : GridMeta, optional  
        Precomputed grid metadata for optimization  
    """  
    velocity_data: np.ndarray  # (Nx, Ny, Nz, 3)  
    grid_x: np.ndarray         # (Nx,)  
    grid_y: np.ndarray         # (Ny,)  
    grid_z: np.ndarray         # (Nz,)  
    bounds_handling: str = "clamp"  # 'clamp' | 'zero' | 'nan' | 'periodic'  
    grid_meta: Optional[GridMeta] = None  

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
        
        # Validate grid regularity  
        if Nx > 2:  
            dx_var = np.var(np.diff(self.grid_x))  
            if dx_var > 1e-6 * self.dx**2:  
                import warnings  
                warnings.warn("Grid X spacing is not uniform, interpolation may be inaccurate")  
        
        if Ny > 2:  
            dy_var = np.var(np.diff(self.grid_y))  
            if dy_var > 1e-6 * self.dy**2:  
                import warnings  
                warnings.warn("Grid Y spacing is not uniform, interpolation may be inaccurate")  
        
        if Nz > 2:  
            dz_var = np.var(np.diff(self.grid_z))  
            if dz_var > 1e-6 * self.dz**2:  
                import warnings  
                warnings.warn("Grid Z spacing is not uniform, interpolation may be inaccurate")  
        
        # Grid bounds  
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
        
        # Compile JAX interpolation if available  
        if JAX_AVAILABLE:  
            self._sample_jax = jit(self._trilinear_interpolate_jax)  
        else:  
            self._sample_jax = self._trilinear_interpolate_numpy  

    def sample(self, positions: np.ndarray) -> np.ndarray:  
        """  
        Sample velocity field at arbitrary positions using trilinear interpolation.  
        
        Parameters  
        ----------  
        positions : np.ndarray  
            Query positions, shape (N, 2) or (N, 3)  
            
        Returns  
        -------  
        np.ndarray  
            Interpolated velocities, shape (N, 3), dtype float32  
        """  
        # Ensure consistent position format  
        pos = _ensure_positions_shape(positions)  # (N, 3), float32  
        
        if pos.shape[0] == 0:  
            return np.zeros((0, 3), dtype=np.float32)  
        
        if JAX_AVAILABLE:  
            pos_jax = jnp.asarray(pos, dtype=jnp.float32)  
            vel_jax = self._sample_jax(pos_jax)  
            return np.asarray(vel_jax, dtype=np.float32)  
        else:  
            return self._trilinear_interpolate_numpy(pos)  

    def _trilinear_interpolate_jax(self, positions: jnp.ndarray) -> jnp.ndarray:  
        """JAX-optimized trilinear interpolation."""  
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]  
        
        # Convert to grid coordinates  
        fx = (x - self.x_min) / self.dx  
        fy = (y - self.y_min) / self.dy  
        fz = (z - self.z_min) / self.dz  
        
        # Handle bounds  
        if self.bounds_handling == "periodic":  
            fx = fx % self.Nx  
            fy = fy % self.Ny  
            fz = fz % self.Nz  
        elif self.bounds_handling == "clamp":  
            fx = jnp.clip(fx, 0, self.Nx - 1)  
            fy = jnp.clip(fy, 0, self.Ny - 1)  
            fz = jnp.clip(fz, 0, self.Nz - 1)  
        
        # Integer indices  
        ix = jnp.floor(fx).astype(jnp.int32)  
        iy = jnp.floor(fy).astype(jnp.int32)  
        iz = jnp.floor(fz).astype(jnp.int32)  
        
        # Handle periodic boundaries for indices  
        if self.bounds_handling == "periodic":  
            ix = ix % self.Nx  
            iy = iy % self.Ny  
            iz = iz % self.Nz  
            ix_p1 = (ix + 1) % self.Nx  
            iy_p1 = (iy + 1) % self.Ny  
            iz_p1 = (iz + 1) % self.Nz  
        else:  
            # Clamp indices to valid range  
            ix = jnp.clip(ix, 0, self.Nx - 2)  
            iy = jnp.clip(iy, 0, self.Ny - 2)  
            iz = jnp.clip(iz, 0, self.Nz - 2)  
            ix_p1 = ix + 1  
            iy_p1 = iy + 1  
            iz_p1 = iz + 1  
        
        # Fractional parts  
        tx = fx - jnp.floor(fx) if self.bounds_handling == "periodic" else fx - ix  
        ty = fy - jnp.floor(fy) if self.bounds_handling == "periodic" else fy - iy  
        tz = fz - jnp.floor(fz) if self.bounds_handling == "periodic" else fz - iz  
        
        # Get corner values  
        v000 = self.velocity_data[ix, iy, iz]          # (N, 3)  
        v100 = self.velocity_data[ix_p1, iy, iz]  
        v010 = self.velocity_data[ix, iy_p1, iz]  
        v110 = self.velocity_data[ix_p1, iy_p1, iz]  
        v001 = self.velocity_data[ix, iy, iz_p1]  
        v101 = self.velocity_data[ix_p1, iy, iz_p1]  
        v011 = self.velocity_data[ix, iy_p1, iz_p1]  
        v111 = self.velocity_data[ix_p1, iy_p1, iz_p1]  
        
        # Trilinear interpolation  
        tx = tx[:, None]  # (N, 1)  
        ty = ty[:, None]  # (N, 1)  
        tz = tz[:, None]  # (N, 1)  
        
        v_xy0 = v000 * (1 - tx) * (1 - ty) + v100 * tx * (1 - ty) + \
                v010 * (1 - tx) * ty + v110 * tx * ty  
        
        v_xy1 = v001 * (1 - tx) * (1 - ty) + v101 * tx * (1 - ty) + \
                v011 * (1 - tx) * ty + v111 * tx * ty  
        
        result = v_xy0 * (1 - tz) + v_xy1 * tz  
        
        # Handle out-of-bounds for non-periodic cases  
        if self.bounds_handling not in ("clamp", "periodic"):  
            original_fx = (positions[:, 0] - self.x_min) / self.dx  
            original_fy = (positions[:, 1] - self.y_min) / self.dy  
            original_fz = (positions[:, 2] - self.z_min) / self.dz  
            
            oob_mask = (  
                (original_fx < 0) | (original_fx >= self.Nx) |  
                (original_fy < 0) | (original_fy >= self.Ny) |  
                (original_fz < 0) | (original_fz >= self.Nz)  
            )[:, None]  
            
            if self.bounds_handling == "zero":  
                result = jnp.where(oob_mask, 0.0, result)  
            elif self.bounds_handling == "nan":  
                result = jnp.where(oob_mask, jnp.nan, result)  
        
        return result.astype(jnp.float32)  

    def _trilinear_interpolate_numpy(self, positions: np.ndarray) -> np.ndarray:  
        """NumPy fallback for trilinear interpolation."""  
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]  
        
        # Convert to grid coordinates  
        fx = (x - self.x_min) / self.dx  
        fy = (y - self.y_min) / self.dy  
        fz = (z - self.z_min) / self.dz  
        
        # Handle bounds  
        if self.bounds_handling == "periodic":  
            fx = fx % self.Nx  
            fy = fy % self.Ny  
            fz = fz % self.Nz  
        elif self.bounds_handling == "clamp":  
            fx = np.clip(fx, 0, self.Nx - 1)  
            fy = np.clip(fy, 0, self.Ny - 1)  
            fz = np.clip(fz, 0, self.Nz - 1)  
        
        # Integer indices  
        ix = np.floor(fx).astype(np.int32)  
        iy = np.floor(fy).astype(np.int32)  
        iz = np.floor(fz).astype(np.int32)  
        
        # Handle periodic boundaries for indices  
        if self.bounds_handling == "periodic":  
            ix = ix % self.Nx  
            iy = iy % self.Ny  
            iz = iz % self.Nz  
            ix_p1 = (ix + 1) % self.Nx  
            iy_p1 = (iy + 1) % self.Ny  
            iz_p1 = (iz + 1) % self.Nz  
        else:  
            # Clamp indices  
            ix = np.clip(ix, 0, self.Nx - 2)  
            iy = np.clip(iy, 0, self.Ny - 2)  
            iz = np.clip(iz, 0, self.Nz - 2)  
            ix_p1 = ix + 1  
            iy_p1 = iy + 1  
            iz_p1 = iz + 1  
        
        # Fractional parts  
        tx = fx - ix
        ty = fy - iy
        tz = fz - iz
        
        # Get corner values - ensure indices are within bounds
        ix = np.clip(ix, 0, self.Nx - 1)
        iy = np.clip(iy, 0, self.Ny - 1)
        iz = np.clip(iz, 0, self.Nz - 1)
        ix_p1 = np.clip(ix_p1, 0, self.Nx - 1)
        iy_p1 = np.clip(iy_p1, 0, self.Ny - 1)
        iz_p1 = np.clip(iz_p1, 0, self.Nz - 1)
        
        v000 = self.velocity_data[ix, iy, iz]          # (N, 3)
        v100 = self.velocity_data[ix_p1, iy, iz]
        v010 = self.velocity_data[ix, iy_p1, iz]
        v110 = self.velocity_data[ix_p1, iy_p1, iz]
        v001 = self.velocity_data[ix, iy, iz_p1]
        v101 = self.velocity_data[ix_p1, iy, iz_p1]
        v011 = self.velocity_data[ix, iy_p1, iz_p1]
        v111 = self.velocity_data[ix_p1, iy_p1, iz_p1]
        
        # Trilinear interpolation
        tx = tx[:, None]  # (N, 1)
        ty = ty[:, None]  # (N, 1)
        tz = tz[:, None]  # (N, 1)
        
        v_xy0 = v000 * (1 - tx) * (1 - ty) + v100 * tx * (1 - ty) + \
                v010 * (1 - tx) * ty + v110 * tx * ty
        
        v_xy1 = v001 * (1 - tx) * (1 - ty) + v101 * tx * (1 - ty) + \
                v011 * (1 - tx) * ty + v111 * tx * ty
        
        result = v_xy0 * (1 - tz) + v_xy1 * tz
        
        # Handle out-of-bounds for non-periodic cases
        if self.bounds_handling not in ("clamp", "periodic"):
            original_fx = (positions[:, 0] - self.x_min) / self.dx
            original_fy = (positions[:, 1] - self.y_min) / self.dy
            original_fz = (positions[:, 2] - self.z_min) / self.dz
            
            oob_mask = (
                (original_fx < 0) | (original_fx >= self.Nx) |
                (original_fy < 0) | (original_fy >= self.Ny) |
                (original_fz < 0) | (original_fz >= self.Nz)
            )[:, None]
            
            if self.bounds_handling == "zero":
                result = np.where(oob_mask, 0.0, result)
            elif self.bounds_handling == "nan":
                result = np.where(oob_mask, np.nan, result)
        
        return result.astype(np.float32)

    def get_spatial_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return spatial bounds of the structured grid."""
        bounds_min = np.array([self.x_min, self.y_min, self.z_min], dtype=np.float32)
        bounds_max = np.array([self.x_max, self.y_max, self.z_max], dtype=np.float32)
        return bounds_min, bounds_max

    def get_grid_metadata(self) -> GridMeta:
        """Return grid metadata."""
        return self.grid_meta

    def memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        velocity_size = self.velocity_data.size * 4  # float32
        grid_size = (self.grid_x.size + self.grid_y.size + self.grid_z.size) * 4
        return (velocity_size + grid_size) / 1024**2

    def validate_data(self) -> bool:
        """
        Validate grid data consistency.
        
        Returns
        -------
        bool
            True if valid
            
        Raises
        ------
        ValueError
            If data is inconsistent
        """
        # Check for NaN/infinite values
        if not np.all(np.isfinite(self.velocity_data)):
            raise ValueError("velocity_data contains non-finite values")
        
        if not np.all(np.isfinite(self.grid_x)):
            raise ValueError("grid_x contains non-finite values")
        
        if not np.all(np.isfinite(self.grid_y)):
            raise ValueError("grid_y contains non-finite values")
        
        if not np.all(np.isfinite(self.grid_z)):
            raise ValueError("grid_z contains non-finite values")
        
        # Check monotonicity
        if not np.all(np.diff(self.grid_x) > 0):
            raise ValueError("grid_x must be monotonically increasing")
        
        if not np.all(np.diff(self.grid_y) > 0):
            raise ValueError("grid_y must be monotonically increasing")
        
        if not np.all(np.diff(self.grid_z) > 0):
            raise ValueError("grid_z must be monotonically increasing")
        
        return True

    def extract_2d_slice(self, plane: str = "xy", index: Optional[int] = None) -> 'StructuredGridSampler':
        """
        Extract 2D slice from 3D grid.
        
        Parameters
        ----------
        plane : str
            Slice plane: 'xy', 'xz', or 'yz'
        index : int, optional
            Index for slice. If None, uses middle of domain.
            
        Returns
        -------
        StructuredGridSampler
            2D slice sampler
        """
        if plane == "xy":
            if index is None:
                index = self.Nz // 2
            if index < 0 or index >= self.Nz:
                raise ValueError(f"Z index {index} out of range [0, {self.Nz-1}]")
            
            velocity_2d = self.velocity_data[:, :, index, :]  # (Nx, Ny, 3)
            velocity_2d = velocity_2d[:, :, None, :]          # (Nx, Ny, 1, 3)
            grid_z_slice = np.array([self.grid_z[index]], dtype=np.float32)
            
            return StructuredGridSampler(
                velocity_data=velocity_2d,
                grid_x=self.grid_x,
                grid_y=self.grid_y,
                grid_z=grid_z_slice,
                bounds_handling=self.bounds_handling
            )
            
        elif plane == "xz":
            if index is None:
                index = self.Ny // 2
            if index < 0 or index >= self.Ny:
                raise ValueError(f"Y index {index} out of range [0, {self.Ny-1}]")
            
            velocity_2d = self.velocity_data[:, index, :, :]  # (Nx, Nz, 3)
            velocity_2d = velocity_2d[:, None, :, :]          # (Nx, 1, Nz, 3)
            grid_y_slice = np.array([self.grid_y[index]], dtype=np.float32)
            
            return StructuredGridSampler(
                velocity_data=velocity_2d,
                grid_x=self.grid_x,
                grid_y=grid_y_slice,
                grid_z=self.grid_z,
                bounds_handling=self.bounds_handling
            )
            
        elif plane == "yz":
            if index is None:
                index = self.Nx // 2
            if index < 0 or index >= self.Nx:
                raise ValueError(f"X index {index} out of range [0, {self.Nx-1}]")
            
            velocity_2d = self.velocity_data[index, :, :, :]  # (Ny, Nz, 3)
            velocity_2d = velocity_2d[None, :, :, :]          # (1, Ny, Nz, 3)
            grid_x_slice = np.array([self.grid_x[index]], dtype=np.float32)
            
            return StructuredGridSampler(
                velocity_data=velocity_2d,
                grid_x=grid_x_slice,
                grid_y=self.grid_y,
                grid_z=self.grid_z,
                bounds_handling=self.bounds_handling
            )
        else:
            raise ValueError(f"Unknown plane '{plane}'. Use 'xy', 'xz', or 'yz'.")

    def sample_at_grid_point(self, i: int, j: int, k: int) -> np.ndarray:
        """
        Sample velocity at specific grid point.
        
        Parameters
        ----------
        i, j, k : int
            Grid indices
            
        Returns
        -------
        np.ndarray
            Velocity at grid point, shape (3,)
        """
        if not (0 <= i < self.Nx and 0 <= j < self.Ny and 0 <= k < self.Nz):
            raise ValueError(f"Grid indices ({i}, {j}, {k}) out of bounds")
        
        return self.velocity_data[i, j, k].copy()

    def get_velocity_magnitude_range(self) -> Tuple[float, float]:
        """Get range of velocity magnitudes in the field."""
        speed = np.linalg.norm(self.velocity_data, axis=-1)  # (Nx, Ny, Nz)
        return float(np.min(speed)), float(np.max(speed))

    def get_divergence(self) -> np.ndarray:
        """
        Compute velocity divergence using finite differences.
        
        Returns
        -------
        np.ndarray
            Divergence field, shape (Nx, Ny, Nz)
        """
        # Central differences with boundary handling
        div = np.zeros((self.Nx, self.Ny, self.Nz), dtype=np.float32)
        
        # dU/dx
        div[1:-1, :, :] += (self.velocity_data[2:, :, :, 0] - self.velocity_data[:-2, :, :, 0]) / (2 * self.dx)
        div[0, :, :] += (self.velocity_data[1, :, :, 0] - self.velocity_data[0, :, :, 0]) / self.dx  # Forward diff
        div[-1, :, :] += (self.velocity_data[-1, :, :, 0] - self.velocity_data[-2, :, :, 0]) / self.dx  # Backward diff
        
        # dV/dy
        div[:, 1:-1, :] += (self.velocity_data[:, 2:, :, 1] - self.velocity_data[:, :-2, :, 1]) / (2 * self.dy)
        div[:, 0, :] += (self.velocity_data[:, 1, :, 1] - self.velocity_data[:, 0, :, 1]) / self.dy
        div[:, -1, :] += (self.velocity_data[:, -1, :, 1] - self.velocity_data[:, -2, :, 1]) / self.dy
        
        # dW/dz
        div[:, :, 1:-1] += (self.velocity_data[:, :, 2:, 2] - self.velocity_data[:, :, :-2, 2]) / (2 * self.dz)
        div[:, :, 0] += (self.velocity_data[:, :, 1, 2] - self.velocity_data[:, :, 0, 2]) / self.dz
        div[:, :, -1] += (self.velocity_data[:, :, -1, 2] - self.velocity_data[:, :, -2, 2]) / self.dz
        
        return div


# Factory functions

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


# Aliases for backwards compatibility
StructuredVelocityField = StructuredGridSampler
StructuredSampler = StructuredGridSampler