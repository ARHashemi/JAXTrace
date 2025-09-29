"""
Density estimators: KDE and SPH with JAX acceleration and CPU fallbacks.

Exports:
- KDEEstimator: Gaussian KDE with Scott/Silverman bandwidth rules
- SPHDensityEstimator: SPH density with cubic spline/Wendland kernels
- Utility kernels and neighbor search
"""

from .kde import KDEEstimator
from .sph import SPHDensityEstimator
from .kernels import (
    gaussian_kernel,
    cubic_spline_kernel_2d,
    cubic_spline_kernel_3d,
    wendland_c2_kernel_2d,
    wendland_c2_kernel_3d,
    scott_bandwidth,
    silverman_bandwidth,
)
from .neighbors import HashGridNeighbors

__all__ = [
    "KDEEstimator",
    "SPHDensityEstimator",
    "gaussian_kernel",
    "cubic_spline_kernel_2d",
    "cubic_spline_kernel_3d",
    "wendland_c2_kernel_2d", 
    "wendland_c2_kernel_3d",
    "scott_bandwidth",
    "silverman_bandwidth",
    "HashGridNeighbors",
]