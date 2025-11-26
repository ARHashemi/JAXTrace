"""
Batch-level velocity interpolation with GPU-resident data.

⚠️ DEPRECATION WARNING:
This module contains DEPRECATED functions that will cause GPU OOM for large batches.
The correct approach is block-wise processing as validated in Phase 1.

DEPRECATED Functions:
- interpolate_velocities_batched() → Will cause OOM (holds all particles on GPU)
- interpolate_velocities_batched_simple() → Will cause OOM (holds all particles on GPU)

RECOMMENDED Function:
- interpolate_velocities_block_by_block() → Correct block-wise approach

See docs/gpu/PHASE1_IMPLEMENTATION_STATUS.md for architectural analysis.

Architecture Comparison:
- ❌ FLAWED (batched): Upload ALL particles → Process all blocks → Download ALL results (causes OOM)
- ✅ CORRECT (block-wise): For each block: Upload block → Process → Download (constant memory)
"""

import time
import warnings
import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable, Optional
from dataclasses import dataclass

from ..particles import ParticleData
from ..forest import PaddedArrays
from .velocity_interpolation import batch_interpolate_velocities
from .gpu_block_filtering import filter_particles_by_block_gpu
from ..batching.block_grouping import group_particles_by_block


@dataclass
class BatchInterpolationStats:
    """Statistics for batch-level velocity interpolation."""
    n_particles: int
    n_active_blocks: int
    time_upload: float = 0.0  # CPU→GPU transfer
    time_gpu_processing: float = 0.0  # GPU computation
    time_download: float = 0.0  # GPU→CPU transfer
    time_total: float = 0.0

    def throughput(self) -> float:
        """Particles per second."""
        if self.time_total == 0:
            return 0.0
        return self.n_particles / self.time_total

    def transfer_fraction(self) -> float:
        """Fraction of time spent on transfers."""
        if self.time_total == 0:
            return 0.0
        return (self.time_upload + self.time_download) / self.time_total


def interpolate_velocities_batched(
    particle_data: ParticleData,
    velocity_field_all_blocks: np.ndarray,
    connectivity_gpu: jnp.ndarray,
    node_positions_gpu: jnp.ndarray,
    padded_arrays: PaddedArrays,
    return_stats: bool = False
) -> tuple[np.ndarray, Optional[BatchInterpolationStats]]:
    """
    ⚠️ DEPRECATED: This function will cause GPU OOM for large batches.

    Use interpolate_velocities_block_by_block() instead.

    Interpolate velocities for entire particle batch with GPU-resident data.

    This implements the user-specified loop hierarchy:
    1. Time marching (outer)
    2. Particle batches (THIS LEVEL - single CPU→GPU→CPU round-trip)
    3. Blocks (GPU-only filtering and processing)

    Parameters
    ----------
    particle_data : ParticleData
        Particle batch to process (any size, typically 1K-200K particles)
    velocity_field_all_blocks : np.ndarray
        Velocity fields for all blocks, shape (n_blocks, max_nodes, 3), float32
    connectivity_gpu : jnp.ndarray
        Element connectivity on GPU (persistent), shape (n_elements, 4), int32
    node_positions_gpu : jnp.ndarray
        Node positions on GPU (persistent), shape (n_nodes, 3), float32
    padded_arrays : PaddedArrays
        Padded block arrays (for block size info)
    return_stats : bool
        Return detailed timing statistics (default: False)

    Returns
    -------
    velocities : np.ndarray
        Interpolated velocities, shape (n_particles, 3), float32
    stats : BatchInterpolationStats or None
        Timing statistics (if return_stats=True)

    Notes
    -----
    **Data Transfer Pattern:**
    - ONCE at start: CPU→GPU (positions, element_ids, block_ids, velocity_fields)
    - GPU-only: Filter particles by block, interpolate velocities for each block
    - ONCE at end: GPU→CPU (velocities)

    **Comparison to Old Approach:**
    - OLD: N × (CPU→GPU + GPU→CPU) for N blocks
    - NEW: 1 × (CPU→GPU + GPU→CPU) for entire batch
    - Expected improvement: 5-10% throughput gain

    **Memory Usage:**
    - GPU: ~n_particles × 32 bytes (positions, element_ids, block_ids)
    - GPU: ~n_relevant_blocks × n_nodes × 12 bytes (velocity fields)
    - Total: Modest, well within GPU capacity for 200K particle batches

    Examples
    --------
    >>> # Initialize (upload mesh to GPU once)
    >>> connectivity_gpu = jax.device_put(connectivity)
    >>> node_positions_gpu = jax.device_put(node_positions)
    >>>
    >>> # Interpolate for batch (single upload/download)
    >>> velocities, stats = interpolate_velocities_batched(
    ...     particle_data=batch,
    ...     velocity_field_all_blocks=vfield,
    ...     connectivity_gpu=connectivity_gpu,
    ...     node_positions_gpu=node_positions_gpu,
    ...     padded_arrays=padded,
    ...     return_stats=True
    ... )
    >>> print(f"Transfer overhead: {100*stats.transfer_fraction():.1f}%")
    """
    warnings.warn(
        "interpolate_velocities_batched() is DEPRECATED and will cause GPU OOM for large batches. "
        "Use interpolate_velocities_block_by_block() instead. "
        "See docs/gpu/PHASE1_IMPLEMENTATION_STATUS.md",
        DeprecationWarning,
        stacklevel=2
    )

    t_total_start = time.time()

    n_particles = len(particle_data.positions)
    velocities_cpu = np.zeros((n_particles, 3), dtype=np.float32)

    if n_particles == 0:
        if return_stats:
            stats = BatchInterpolationStats(
                n_particles=0,
                n_active_blocks=0,
                time_total=0.0
            )
            return velocities_cpu, stats
        return velocities_cpu, None

    # ============================================================================
    # BATCH LEVEL: SINGLE CPU→GPU TRANSFER
    # ============================================================================
    t_upload_start = time.time()

    # Upload particle data to GPU (ONCE for entire batch)
    positions_gpu = jax.device_put(particle_data.positions)
    element_ids_gpu = jax.device_put(particle_data.element_ids)
    block_ids_gpu = jax.device_put(particle_data.block_ids)

    t_upload_end = time.time()
    time_upload = t_upload_end - t_upload_start

    # ============================================================================
    # GPU PROCESSING: FILTER AND INTERPOLATE PER BLOCK
    # ============================================================================
    t_gpu_start = time.time()

    # Group particles by block (CPU-side, for scheduling)
    # NOTE: This only computes indices, doesn't transfer data
    grouping = group_particles_by_block(
        particle_data.block_ids,
        padded_arrays.block_sizes
    )
    n_active_blocks = len(grouping.groups)

    # Allocate result array on GPU
    velocities_gpu = jnp.zeros((n_particles, 3), dtype=jnp.float32)

    # Process each active block
    for block_id, particle_indices_cpu in grouping.groups.items():
        if len(particle_indices_cpu) == 0:
            continue

        # Filter particles for this block (ON GPU)
        block_positions, block_element_ids, block_indices = filter_particles_by_block_gpu(
            positions_gpu,
            element_ids_gpu,
            block_ids_gpu,
            target_block_id=block_id
        )

        # Upload velocity field for this block
        block_velocity_field_gpu = jax.device_put(velocity_field_all_blocks[block_id])

        # Interpolate velocities on GPU
        block_velocities = batch_interpolate_velocities(
            block_positions,
            block_element_ids,
            connectivity_gpu,
            node_positions_gpu,
            block_velocity_field_gpu
        )

        # Scatter block results back into full velocities array (ON GPU)
        velocities_gpu = velocities_gpu.at[block_indices].set(block_velocities)

    t_gpu_end = time.time()
    time_gpu = t_gpu_end - t_gpu_start

    # ============================================================================
    # BATCH LEVEL: SINGLE GPU→CPU TRANSFER
    # ============================================================================
    t_download_start = time.time()

    # Download results (ONCE for entire batch)
    velocities_cpu = np.array(velocities_gpu)

    t_download_end = time.time()
    time_download = t_download_end - t_download_start

    # ============================================================================
    # STATISTICS
    # ============================================================================
    t_total_end = time.time()
    time_total = t_total_end - t_total_start

    if return_stats:
        stats = BatchInterpolationStats(
            n_particles=n_particles,
            n_active_blocks=n_active_blocks,
            time_upload=time_upload,
            time_gpu_processing=time_gpu,
            time_download=time_download,
            time_total=time_total
        )
        return velocities_cpu, stats

    return velocities_cpu, None


def interpolate_velocities_batched_simple(
    particle_data: ParticleData,
    velocity_field_all_blocks: np.ndarray,
    connectivity_gpu: jnp.ndarray,
    node_positions_gpu: jnp.ndarray,
    padded_arrays: PaddedArrays,
) -> np.ndarray:
    """
    ⚠️ DEPRECATED: This function will cause GPU OOM for large batches.

    Use interpolate_velocities_block_by_block() instead.

    Simplified batch-level velocity interpolation (no statistics).

    This is a convenience wrapper for interpolate_velocities_batched()
    that only returns velocities (no timing statistics).

    Parameters
    ----------
    particle_data : ParticleData
        Particle batch to process
    velocity_field_all_blocks : np.ndarray
        Velocity fields for all blocks
    connectivity_gpu : jnp.ndarray
        Element connectivity on GPU (persistent)
    node_positions_gpu : jnp.ndarray
        Node positions on GPU (persistent)
    padded_arrays : PaddedArrays
        Padded block arrays

    Returns
    -------
    velocities : np.ndarray
        Interpolated velocities, shape (n_particles, 3)

    Examples
    --------
    >>> velocities = interpolate_velocities_batched_simple(
    ...     particle_data, velocity_field, connectivity_gpu, node_positions_gpu, padded
    ... )
    """
    velocities, _ = interpolate_velocities_batched(
        particle_data,
        velocity_field_all_blocks,
        connectivity_gpu,
        node_positions_gpu,
        padded_arrays,
        return_stats=False
    )
    return velocities


# Backward compatibility: Old block-by-block approach
# (Kept for comparison / testing)
def interpolate_velocities_block_by_block(
    particle_data: ParticleData,
    velocity_field_all_blocks: np.ndarray,
    connectivity: np.ndarray,
    node_positions: np.ndarray,
    padded_arrays: PaddedArrays,
) -> np.ndarray:
    """
    OLD APPROACH: Block-by-block velocity interpolation (DEPRECATED).

    This is the OLD implementation with block-level CPU↔GPU transfers.
    Kept for comparison and backward compatibility.

    NEW USERS: Use interpolate_velocities_batched() instead.

    Performance:
    - OLD (this function): ~10 p/s
    - NEW (batched): ~11-12 p/s (5-10% faster)

    Why slower:
    - Transfers data CPU→GPU→CPU for EACH block
    - Excessive data movement overhead
    """
    # Pre-upload connectivity and node_positions to GPU
    connectivity_gpu = jax.device_put(connectivity)
    node_positions_gpu = jax.device_put(node_positions)

    n_particles = len(particle_data.positions)
    velocities = np.zeros((n_particles, 3), dtype=np.float32)

    # Group particles by block
    grouping = group_particles_by_block(
        particle_data.block_ids,
        padded_arrays.block_sizes
    )

    # Process each block separately (with transfers per block)
    for block_id, particle_indices in grouping.groups.items():
        if len(particle_indices) == 0:
            continue

        # Extract data for this block (CPU)
        block_positions = particle_data.positions[particle_indices]
        block_element_ids = particle_data.element_ids[particle_indices]

        # 🔴 TRANSFER: CPU→GPU (per block)
        block_positions_gpu = jax.device_put(block_positions)
        block_element_ids_gpu = jax.device_put(block_element_ids)
        block_velocity_field_gpu = jax.device_put(velocity_field_all_blocks[block_id])

        # ✅ GPU: Interpolate
        block_velocities = batch_interpolate_velocities(
            block_positions_gpu,
            block_element_ids_gpu,
            connectivity_gpu,
            node_positions_gpu,
            block_velocity_field_gpu
        )

        # 🔴 TRANSFER: GPU→CPU (per block)
        velocities[particle_indices] = np.array(block_velocities)

    return velocities
