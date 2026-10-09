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
from .bandwidth import resolve_bandwidth, particle_bbox, initial_particle_spacing
from .union import (
    UnionResult,
    dedup_batch,
    dedup_incremental,
    union_no_dedup,
    write_union_vtkhdf,
    write_union_npy,
    read_union_vtkhdf,
    read_union_attrs,
    roi_box_from_fraction,
)
from .comoving import (
    CoMovingParticleResult,
    CoMovingDensityResult,
    UniformReferenceResult,
    compute_comoving_displacement,
    compute_comoving_reference_density,
    build_uniform_reference_cloud,
)
from .streamlines import (
    StreamlineResult,
    select_streamline_particles,
    collect_streamlines_from_vtkhdf,
    write_streamlines_vtkhdf,
)
from .inside_mesh import compute_inside_mesh_mask, inside_mask_to_3d
from .time_accumulator import TimeAccumulator
from .writers import (
    DensityWriterConfig,
    DensityWriterThread,
    write_time_average,
    read_time_average_vtkhdf,
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
    "initial_particle_spacing",
    "UnionResult",
    "dedup_batch",
    "dedup_incremental",
    "union_no_dedup",
    "write_union_vtkhdf",
    "write_union_npy",
    "read_union_vtkhdf",
    "read_union_attrs",
    "roi_box_from_fraction",
    "CoMovingParticleResult",
    "CoMovingDensityResult",
    "UniformReferenceResult",
    "compute_comoving_displacement",
    "compute_comoving_reference_density",
    "build_uniform_reference_cloud",
    "StreamlineResult",
    "select_streamline_particles",
    "collect_streamlines_from_vtkhdf",
    "write_streamlines_vtkhdf",
    "compute_inside_mesh_mask",
    "inside_mask_to_3d",
    "TimeAccumulator",
    "DensityWriterConfig",
    "DensityWriterThread",
    "write_time_average",
    "read_time_average_vtkhdf",
    "KERNEL_NAMES",
    "kernel_support",
    "evaluate_kernel",
]
