"""
GPU-Accelerated Particle Tracking

Complete time-marching pipeline for particle tracking in FEM meshes.

Components:
- velocity_interpolation.py: Block-local velocity interpolation using barycentric coordinates
- batch_velocity_interpolation.py: Batch-level interpolation with GPU-resident data (DEPRECATED)
- gpu_block_filtering.py: GPU-native particle filtering by block (Priority 2)
- time_integration.py: Forward Euler and RK4 time integration
- blockwise_rk4.py: Block-wise RK4 with integrated interpolation (RECOMMENDED)
- time_marching.py: Single-timestep pipeline (interpolate → integrate → search)
- async_time_marcher.py: Multi-timestep async pipeline with CPU-GPU overlap
"""

from .velocity_interpolation import (
    interpolate_velocity_in_element,
    batch_interpolate_velocities,
    compute_barycentric_coordinates
)

from .batch_velocity_interpolation import (
    interpolate_velocities_batched,
    interpolate_velocities_batched_simple,
    interpolate_velocities_block_by_block,
    BatchInterpolationStats
)

from .gpu_block_filtering import (
    filter_particles_by_block_gpu,
    count_particles_per_block_gpu,
    get_non_empty_blocks_gpu,
    get_block_particle_count,
    precompile_block_filters,
    BlockParticleData
)

from .time_integration import (
    forward_euler_step,
    forward_euler_step_with_active_mask,
    compute_adaptive_timestep,
    rk4_step_with_search,
    rk4_step_with_incremental_search,
    rk4_step_simple
)

from .blockwise_rk4 import (
    rk4_step_blockwise,
    rk4_step_blockwise_single_block,
    BlockwiseRK4Stats
)

from .time_marching import (
    ParticleTimeMarcher,
    create_constant_velocity_field,
    create_time_dependent_velocity_field_fn,
    benchmark_time_marching_pipeline
)

__all__ = [
    # Velocity interpolation
    'interpolate_velocity_in_element',
    'batch_interpolate_velocities',
    'compute_barycentric_coordinates',

    # Batch-level velocity interpolation (DEPRECATED)
    'interpolate_velocities_batched',
    'interpolate_velocities_batched_simple',
    'interpolate_velocities_block_by_block',
    'BatchInterpolationStats',

    # GPU block filtering (Priority 2)
    'filter_particles_by_block_gpu',
    'count_particles_per_block_gpu',
    'get_non_empty_blocks_gpu',
    'get_block_particle_count',
    'precompile_block_filters',
    'BlockParticleData',

    # Time integration
    'forward_euler_step',
    'forward_euler_step_with_active_mask',
    'compute_adaptive_timestep',
    'rk4_step_with_search',
    'rk4_step_with_incremental_search',
    'rk4_step_simple',

    # Block-wise RK4 (RECOMMENDED)
    'rk4_step_blockwise',
    'rk4_step_blockwise_single_block',
    'BlockwiseRK4Stats',

    # Time-marching pipeline
    'ParticleTimeMarcher',
    'create_constant_velocity_field',
    'create_time_dependent_velocity_field_fn',
    'benchmark_time_marching_pipeline',
]
