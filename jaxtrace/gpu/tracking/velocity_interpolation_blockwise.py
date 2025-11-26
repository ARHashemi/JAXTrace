"""
Baseline Block-wise Velocity Interpolation (Preserved for Compatibility)

This module preserves the original block-by-block interpolation approach.
While this has known performance bottlenecks (repeated CPU-GPU transfers),
it is kept for:
- Backwards compatibility
- Fallback for OOM cases
- A/B performance testing
- Reference implementation

Performance characteristics:
- Throughput: ~5,000-7,000 particles/second
- Memory: 17 GB CPU RAM, 2.3 GB GPU
- CPU-GPU transfers: 4.9 GB per RK4 step
- GPU utilization: 40-50% (transfer-limited)

For production workloads, prefer the global mesh implementation:
  from jaxtrace.gpu.tracking.velocity_interpolation_global import create_global_interpolator
"""

import numpy as np
import jax
from typing import Callable

from jaxtrace.gpu.particles import ParticleData
from jaxtrace.gpu.batching.block_grouping import group_particles_by_block
from jaxtrace.gpu.tracking import batch_interpolate_velocities


def create_blockwise_interpolator(
    velocity_field_all_blocks: np.ndarray,
    padded_arrays,
    connectivity_gpu,
    node_positions_gpu
) -> Callable[[ParticleData, float], np.ndarray]:
    """
    Create block-by-block velocity interpolator (baseline implementation).

    WARNING: This approach has significant performance bottlenecks:
    - Block-by-block processing requires 120-200 CPU-GPU transfers per RK4 step
    - Total data transfer: ~4.9 GB per RK4 step
    - GPU utilization: 40-50% (transfer-limited, not compute-limited)

    Parameters
    ----------
    velocity_field_all_blocks : ndarray, shape (n_blocks, n_nodes, 3)
        Velocity field replicated for each block
    padded_arrays : PaddedBlockArrays
        Padded block data structure with block_sizes
    connectivity_gpu : DeviceArray
        GPU-resident element connectivity (persistent)
    node_positions_gpu : DeviceArray
        GPU-resident node positions (persistent)

    Returns
    -------
    interpolator : Callable[[ParticleData, float], ndarray]
        Function that interpolates velocities for particle data
        Returns: velocities array, shape (n_particles, 3)
    """

    def interpolator(pdata: ParticleData, t: float) -> np.ndarray:
        """
        Interpolate velocities using block-by-block approach.

        Process:
        1. Group particles by block ID
        2. For each block:
           - Extract particle positions/elements for that block
           - Upload to GPU (❌ CPU-GPU transfer)
           - Upload velocity field for that block (❌ CPU-GPU transfer)
           - Call GPU interpolation kernel
           - Download results (❌ GPU-CPU transfer)
        3. Assemble results

        Bottleneck: Steps 2a-2e happen 120-200 times per RK4 step!
        """
        n = len(pdata.positions)
        velocities = np.zeros((n, 3), dtype=np.float32)

        # Group particles by block
        grouping = group_particles_by_block(
            pdata.block_ids,
            padded_arrays.block_sizes
        )

        # Process each block independently (THE BOTTLENECK)
        for block_id, particle_indices in grouping.groups.items():
            if len(particle_indices) == 0:
                continue

            # Extract data for this block
            block_positions = pdata.positions[particle_indices]
            block_element_ids = pdata.element_ids[particle_indices]

            # ❌ CPU-GPU transfers (happens 120-200 times per RK4 step!)
            block_positions_gpu = jax.device_put(block_positions)
            block_element_ids_gpu = jax.device_put(block_element_ids)
            block_vfield_gpu = jax.device_put(velocity_field_all_blocks[block_id])

            # GPU interpolation (fast, but limited by transfer overhead)
            block_velocities = batch_interpolate_velocities(
                block_positions_gpu,
                block_element_ids_gpu,
                connectivity_gpu,
                node_positions_gpu,
                block_vfield_gpu
            )

            # ❌ GPU-CPU transfer
            velocities[particle_indices] = np.array(block_velocities)

        return velocities

    return interpolator
