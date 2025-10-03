"""
Time Series Field with Octree FEM Interpolation

For meshes with adaptive refinement (multiple resolution levels).
"""

import numpy as np
import jax.numpy as jnp
from typing import Optional

from .time_series import TimeSeriesField
from .octree_fem_interpolator import build_octree_mesh, create_octree_fem_interpolator, OctreeMesh


class OctreeFEMTimeSeriesField(TimeSeriesField):
    """
    Time series field with adaptive octree FEM interpolation.

    Optimized for meshes with adaptive refinement where element sizes
    vary significantly across the domain.
    """

    def __init__(self,
                 data: np.ndarray,
                 times: np.ndarray,
                 positions: np.ndarray,
                 connectivity: np.ndarray,
                 interpolation: str = "linear",
                 extrapolation: str = "constant",
                 _source_info: Optional[dict] = None,
                 max_elements_per_leaf: int = 32,
                 max_depth: int = 12):
        """
        Initialize octree FEM time series field.

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
        max_elements_per_leaf : int
            Maximum elements before octree subdivision (default: 32)
        max_depth : int
            Maximum octree depth (default: 12)
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

        # Build octree mesh structure
        print(f"🌲 Creating octree FEM interpolation field...")
        self.octree_mesh = build_octree_mesh(
            positions,
            connectivity,
            max_elements_per_leaf=max_elements_per_leaf,
            max_depth=max_depth
        )

        # Create JIT-compiled interpolator
        self.octree_interpolator = create_octree_fem_interpolator(self.octree_mesh)

        print(f"✅ Octree FEM interpolation ready!")

    def sample_at_positions(self, query_positions: np.ndarray, t: float) -> jnp.ndarray:
        """
        Sample field at positions using octree FEM interpolation.

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
        field_at_nodes = self._sample_all_nodes_at_time_jax(jnp.asarray(t, dtype=jnp.float32))

        # Apply octree FEM spatial interpolation
        interpolated_values = self.octree_interpolator(query_positions, field_at_nodes)

        return interpolated_values

    def __repr__(self) -> str:
        num_leaves = int(jnp.sum(self.octree_mesh.nodes_is_leaf))
        return (
            f"OctreeFEMTimeSeriesField("
            f"data={self.data.shape}, "
            f"times={self.times.shape}, "
            f"positions={self.positions.shape}, "
            f"elements={self.octree_mesh.connectivity.shape[0]}, "
            f"octree_nodes={self.octree_mesh.nodes_min.shape[0]}, "
            f"octree_leaves={num_leaves}, "
            f"interpolation='{self.interpolation}', "
            f"extrapolation='{self.extrapolation}')"
        )
