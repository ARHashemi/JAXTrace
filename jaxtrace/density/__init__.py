"""
GPU/JAX density estimation for the JAXTrace particle cloud.

Public API
----------

  - :class:`DensityRunnerConfig`, :class:`DensityRunner` — the one-stop
    integration point used by ``run_tracking.py`` and the offline
    post-processor.

  - :class:`EstimatorConfig`, :class:`DensityEstimator` — low-level
    estimator with brute-force / Morton-hash backends.

  - :mod:`kernels`, :mod:`bandwidth`, :mod:`grid`, :mod:`inside_mesh`,
    :mod:`time_accumulator`, :mod:`writers` — building blocks if you
    want to assemble the pipeline manually.
"""

from .runner import DensityRunner, DensityRunnerConfig
from .estimator import DensityEstimator, EstimatorConfig
from .grid import (
    VoxelGrid,
    make_voxel_grid,
    bbox_union,
    positions_bbox,
    trajectory_bbox_union_from_vtkhdf,
    iterate_vtkhdf_steps,
    prefetch_vtkhdf_steps,
)
from .bandwidth import resolve_bandwidth, particle_bbox
from .inside_mesh import compute_inside_mesh_mask, inside_mask_to_3d
from .time_accumulator import TimeAccumulator
from .writers import (
    DensityWriterConfig,
    DensityWriterThread,
    write_time_average,
)
from .kernels import KERNEL_NAMES, kernel_support, evaluate_kernel

__all__ = [
    "DensityRunner",
    "DensityRunnerConfig",
    "DensityEstimator",
    "EstimatorConfig",
    "VoxelGrid",
    "make_voxel_grid",
    "bbox_union",
    "positions_bbox",
    "trajectory_bbox_union_from_vtkhdf",
    "iterate_vtkhdf_steps",
    "prefetch_vtkhdf_steps",
    "resolve_bandwidth",
    "particle_bbox",
    "compute_inside_mesh_mask",
    "inside_mask_to_3d",
    "TimeAccumulator",
    "DensityWriterConfig",
    "DensityWriterThread",
    "write_time_average",
    "KERNEL_NAMES",
    "kernel_support",
    "evaluate_kernel",
]
