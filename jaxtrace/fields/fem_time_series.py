"""
Time Series Field with FEM Interpolation

Enhanced TimeSeriesField that uses finite element interpolation
instead of nearest-neighbor for improved accuracy and speed.
"""

import numpy as np
import jax.numpy as jnp
from typing import Optional, Tuple

from .time_series import TimeSeriesField
from .fem_interpolator import build_tetrahedral_mesh, create_fem_interpolator, TetrahedralMesh


class FEMTimeSeriesField(TimeSeriesField):
    """
    Time series field with FEM interpolation.

    Uses tetrahedral finite element interpolation for spatial queries,
    providing higher accuracy than nearest-neighbor while maintaining
    JIT compilation compatibility.
    """

    def __init__(self,
                 data: np.ndarray,
                 times: np.ndarray,
                 positions: np.ndarray,
                 connectivity: np.ndarray,
                 interpolation: str = "linear",
                 extrapolation: str = "constant",
                 _source_info: Optional[dict] = None,
                 fem_grid_resolution: int = 32):
        """
        Initialize FEM time series field.

        Parameters
        ----------
        data : np.ndarray
            Velocity data (T, N, 3)
        times : np.ndarray
            Time values (T,)
        positions : np.ndarray
            Node positions (N, 3)
        connectivity : np.ndarray
            Tetrahedral connectivity (M, 4)
        interpolation : str
            Time interpolation method
        extrapolation : str
            Time extrapolation method
        _source_info : dict, optional
            Source metadata
        fem_grid_resolution : int
            Spatial hash grid resolution for element lookup
        """

        # Initialize base class
        super().__init__(
            data=data,
            times=times,
            positions=positions,
            interpolation=interpolation,
            extrapolation=extrapolation,
            _source_info=_source_info
        )

        # Build FEM mesh structure
        print(f"🔨 Building FEM interpolation mesh...")
        self.fem_mesh = build_tetrahedral_mesh(
            positions,
            connectivity,
            grid_resolution=fem_grid_resolution
        )

        # Create JIT-compiled interpolator
        self.fem_interpolator = create_fem_interpolator(self.fem_mesh)

        print(f"✅ FEM interpolation ready!")

    def sample_at_positions(self, query_positions: np.ndarray, t: float) -> jnp.ndarray:
        """
        Sample field at positions using FEM interpolation.

        Overrides base class to use finite element interpolation
        instead of nearest-neighbor.

        Parameters
        ----------
        query_positions : np.ndarray
            Query positions (M, 3)
        t : float
            Query time

        Returns
        -------
        jnp.ndarray
            Interpolated velocities (M, 3)
        """

        # Ensure JAX array
        query_positions = jnp.asarray(query_positions, dtype=jnp.float32)

        # Get field values at all nodes for this time
        # Uses temporal interpolation from base class
        field_at_nodes = self._sample_all_nodes_at_time_jax(jnp.asarray(t, dtype=jnp.float32))

        # Apply FEM spatial interpolation
        interpolated_values = self.fem_interpolator(query_positions, field_at_nodes)

        return interpolated_values

    def __repr__(self) -> str:
        return (
            f"FEMTimeSeriesField("
            f"data={self.data.shape}, "
            f"times={self.times.shape}, "
            f"positions={self.positions.shape}, "
            f"elements={self.fem_mesh.connectivity.shape[0]}, "
            f"interpolation='{self.interpolation}', "
            f"extrapolation='{self.extrapolation}')"
        )
