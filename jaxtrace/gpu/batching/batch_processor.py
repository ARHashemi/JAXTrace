"""
Main batch processor for GPU particle tracking.

Part of Phase 1: Setup and Validation
Implements the core batching loop from:
docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md (lines 253-384, 1047-1159)

This module coordinates the two-level batched block-wise architecture:
Level 1: Particle batching (200K particles per batch)
Level 2: Block-wise processing within each batch

Key functions:
- process_batch(): Process one batch of particles block-by-block
- track_particles_batched(): Main entry point for batched tracking
"""

import time
import numpy as np
import jax.numpy as jnp
from typing import Optional, Dict, Tuple, Callable
from dataclasses import dataclass, field

from .batch_config import BatchConfig
from .block_grouping import group_particles_by_block, ParticleGrouping
from .memory_utils import monitor_batch_memory_usage
from ..particles import ParticleData
from ..forest import PaddedArrays
from ..search import (
    search_particles_in_block,
    search_particles_in_block_with_hash,
    batch_search_light_blocks,
    BlockSearchResult
)


@dataclass
class BatchStatistics:
    """
    Statistics for a single batch processing run.

    Tracks timing, memory, and search performance for one batch.
    """

    batch_id: int
    n_particles: int
    n_active_blocks: int

    # Timing (seconds)
    time_grouping: float = 0.0
    time_block_processing: float = 0.0
    time_total: float = 0.0

    # Memory (MB)
    vram_start_mb: float = 0.0
    vram_end_mb: float = 0.0
    vram_delta_mb: float = 0.0

    # Block-wise timing
    time_per_block: Dict[int, float] = field(default_factory=dict)
    particles_per_block: Dict[int, int] = field(default_factory=dict)

    # Search level statistics
    level0_hits: int = 0
    level1_hits: int = 0
    level2_hits: int = 0
    not_found: int = 0

    def throughput_particles_per_sec(self) -> float:
        """Calculate batch throughput in particles/second."""
        if self.time_total == 0:
            return 0.0
        return self.n_particles / self.time_total

    def __repr__(self) -> str:
        return (
            f"BatchStatistics(batch={self.batch_id}, "
            f"n_particles={self.n_particles}, "
            f"time={self.time_total:.3f}s, "
            f"throughput={self.throughput_particles_per_sec():.0f} p/s)"
        )


@dataclass
class ProcessorStatistics:
    """
    Overall statistics for complete particle tracking session.

    Aggregates statistics across all batches.
    """

    total_particles: int = 0
    total_timesteps: int = 0
    total_batches: int = 0

    # Timing (seconds)
    total_time: float = 0.0
    time_per_timestep: float = 0.0

    # Per-batch statistics
    batch_stats: list[BatchStatistics] = field(default_factory=list)

    # Search level hit rates (aggregated)
    total_level0_hits: int = 0
    total_level1_hits: int = 0
    total_level2_hits: int = 0
    total_not_found: int = 0

    def average_throughput(self) -> float:
        """Average throughput across all batches (particles/second)."""
        if self.total_time == 0:
            return 0.0
        return self.total_particles / self.total_time

    def hit_rate_level0(self) -> float:
        """Percentage found in cached element."""
        total = self.total_level0_hits + self.total_level1_hits + self.total_level2_hits + self.total_not_found
        if total == 0:
            return 0.0
        return 100.0 * self.total_level0_hits / total

    def hit_rate_level1(self) -> float:
        """Percentage found in neighbor elements."""
        total = self.total_level0_hits + self.total_level1_hits + self.total_level2_hits + self.total_not_found
        if total == 0:
            return 0.0
        return 100.0 * self.total_level1_hits / total

    def hit_rate_level2(self) -> float:
        """Percentage found via block search."""
        total = self.total_level0_hits + self.total_level1_hits + self.total_level2_hits + self.total_not_found
        if total == 0:
            return 0.0
        return 100.0 * self.total_level2_hits / total

    def print_summary(self):
        """Print comprehensive processing summary."""
        print("\n" + "="*80)
        print("BATCH PROCESSING SUMMARY")
        print("="*80)

        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  Total particles: {self.total_particles:,}")
        print(f"  Total timesteps: {self.total_timesteps}")
        print(f"  Total batches: {self.total_batches}")
        print(f"  Total time: {self.total_time:.2f}s")

        print(f"\n⚡ THROUGHPUT:")
        print(f"  Average: {self.average_throughput():.0f} particles/s")
        if self.batch_stats:
            throughputs = [b.throughput_particles_per_sec() for b in self.batch_stats]
            print(f"  Min: {min(throughputs):.0f} particles/s")
            print(f"  Max: {max(throughputs):.0f} particles/s")

        print(f"\n🎯 SEARCH HIT RATES:")
        print(f"  Level 0 (cached): {self.hit_rate_level0():.1f}%")
        print(f"  Level 1 (neighbors): {self.hit_rate_level1():.1f}%")
        print(f"  Level 2 (block): {self.hit_rate_level2():.1f}%")

        if self.total_batches > 5:
            print(f"\n📈 BATCH TIMING (last 5):")
            for stat in self.batch_stats[-5:]:
                print(f"  Batch {stat.batch_id}: {stat.time_total:.3f}s "
                      f"({stat.throughput_particles_per_sec():.0f} p/s)")

        print("="*80 + "\n")


def process_batch(
    batch_particles: ParticleData,
    padded_arrays: PaddedArrays,
    config: BatchConfig,
    search_kernel: Callable,
    batch_id: int = 0,
    verbose: bool = True
) -> Tuple[ParticleData, BatchStatistics]:
    """
    Process one batch of particles using block-wise approach.

    This implements the core Level 2 logic: within a single batch,
    group particles by block and process each block separately.

    Based on logic from:
    docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md (lines 253-384)

    Parameters
    ----------
    batch_particles : ParticleData
        Particle batch to process (up to config.batch_size particles)
    padded_arrays : PaddedArrays
        Padded block arrays on GPU
    config : BatchConfig
        Batch configuration
    search_kernel : Callable
        JAX-compiled block search kernel function
        Signature: search_kernel(block_id, particle_indices) -> updated_particles
    batch_id : int
        Batch ID for logging (default: 0)
    verbose : bool
        Print progress messages (default: True)

    Returns
    -------
    updated_particles : ParticleData
        Particles with updated element_ids and block_ids
    stats : BatchStatistics
        Batch processing statistics

    Notes
    -----
    Processing flow:
    1. Group particles by block (CPU, <5ms)
    2. For each block with particles:
       a. Heavy blocks (>10K elem): Use hash buckets
       b. Medium blocks (1K-10K): Direct search
       c. Light blocks (<1K): Batch together (Phase 2)
    3. Update particle states

    Memory usage:
    - Input batch: ~n_particles × 32 bytes
    - No additional GPU allocation (padded arrays already loaded)
    - Expected time: 5-20ms per 200K particles (Phase 1 baseline)

    Examples
    --------
    >>> # Process one batch
    >>> updated, stats = process_batch(
    ...     batch_particles=particles[0:200_000],
    ...     padded_arrays=padded,
    ...     config=config,
    ...     search_kernel=jitted_search_fn,
    ...     batch_id=0
    ... )
    >>> print(f"Batch throughput: {stats.throughput_particles_per_sec():.0f} p/s")
    """
    t_start = time.time()

    # Initialize statistics
    stats = BatchStatistics(
        batch_id=batch_id,
        n_particles=batch_particles.n_active,
        n_active_blocks=0
    )

    # Track VRAM at batch start
    mem_start = monitor_batch_memory_usage(f"Batch {batch_id} start")
    stats.vram_start_mb = mem_start['used_mb']

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing Batch {batch_id}: {batch_particles.n_active:,} particles")
        print(f"{'='*60}")

    # Step 1: Group particles by block (CPU operation)
    t_group_start = time.time()

    grouping = group_particles_by_block(
        particle_block_ids=batch_particles.block_ids,
        block_sizes=padded_arrays.block_sizes,
        heavy_threshold=config.heavy_block_threshold,
        light_threshold=config.light_block_threshold
    )

    stats.time_grouping = time.time() - t_group_start
    stats.n_active_blocks = grouping.n_blocks_active

    if verbose:
        print(f"  Grouping: {stats.time_grouping*1000:.1f}ms")
        print(f"  Active blocks: {grouping.n_blocks_active}")
        print(f"    Heavy: {len(grouping.heavy_blocks)}")
        print(f"    Medium: {len(grouping.medium_blocks)}")
        print(f"    Light: {len(grouping.light_blocks)}")

    # Step 2: Process blocks
    t_blocks_start = time.time()

    # Process heavy blocks first (most expensive)
    for block_id in grouping.heavy_blocks:
        t_block_start = time.time()
        particle_indices = grouping.groups[block_id]
        n_particles_in_block = len(particle_indices)

        if verbose and n_particles_in_block > 100:
            n_elem = padded_arrays.block_sizes[block_id]
            print(f"  Block {block_id} (heavy, {n_elem:,} elem): {n_particles_in_block:,} particles", end="")

        # Extract particles for this block
        block_positions = jnp.array(batch_particles.positions[particle_indices], dtype=jnp.float32)
        block_element_ids = jnp.array(batch_particles.element_ids[particle_indices], dtype=jnp.int32)
        block_ids_array = jnp.full(n_particles_in_block, block_id, dtype=jnp.int32)
        block_active = jnp.array(batch_particles.active_mask[particle_indices], dtype=jnp.bool_)

        # Get block data from padded arrays
        block_size = padded_arrays.block_sizes[block_id]
        block_connectivity = jnp.array(padded_arrays.connectivity[block_id, :block_size], dtype=jnp.int32)
        block_node_positions = jnp.array(padded_arrays.node_positions[block_id], dtype=jnp.float32)
        block_neighbors = jnp.array(padded_arrays.element_neighbors[block_id, :block_size], dtype=jnp.int32)

        # Call search kernel for this block
        if config.use_hash_buckets:
            # Heavy blocks use hash bucket search (Strategy 1)
            # TODO: Pass hash bucket data when preprocessing is implemented
            result = search_particles_in_block_with_hash(
                particle_positions=block_positions,
                particle_element_ids=block_element_ids,
                particle_block_ids=block_ids_array,
                particle_active=block_active,
                block_id=block_id,
                block_connectivity=block_connectivity,
                block_node_positions=block_node_positions,
                block_element_neighbors=block_neighbors,
                block_size=block_size,
                hash_bucket_data=None  # Falls back to standard search for now
            )
        else:
            # Standard 3-level search
            result = search_particles_in_block(
                particle_positions=block_positions,
                particle_element_ids=block_element_ids,
                particle_block_ids=block_ids_array,
                particle_active=block_active,
                block_id=block_id,
                block_connectivity=block_connectivity,
                block_node_positions=block_node_positions,
                block_element_neighbors=block_neighbors,
                block_size=block_size
            )

        # Update batch particles with search results
        batch_particles.element_ids[particle_indices] = np.array(result.new_element_ids)

        # Accumulate statistics
        stats.level0_hits += int(result.level0_hits)
        stats.level1_hits += int(result.level1_hits)
        stats.level2_hits += int(result.level2_hits)
        stats.not_found += int(result.not_found)

        t_block_end = time.time()
        stats.time_per_block[block_id] = t_block_end - t_block_start
        stats.particles_per_block[block_id] = n_particles_in_block

        if verbose and n_particles_in_block > 100:
            print(f" -> {(t_block_end - t_block_start)*1000:.1f}ms "
                  f"[L0:{result.level0_hits} L1:{result.level1_hits} L2:{result.level2_hits}]")

    # Process medium blocks
    for block_id in grouping.medium_blocks:
        t_block_start = time.time()
        particle_indices = grouping.groups[block_id]
        n_particles_in_block = len(particle_indices)

        # Extract particles for this block
        block_positions = jnp.array(batch_particles.positions[particle_indices], dtype=jnp.float32)
        block_element_ids = jnp.array(batch_particles.element_ids[particle_indices], dtype=jnp.int32)
        block_ids_array = jnp.full(n_particles_in_block, block_id, dtype=jnp.int32)
        block_active = jnp.array(batch_particles.active_mask[particle_indices], dtype=jnp.bool_)

        # Get block data from padded arrays
        block_size = padded_arrays.block_sizes[block_id]
        block_connectivity = jnp.array(padded_arrays.connectivity[block_id, :block_size], dtype=jnp.int32)
        block_node_positions = jnp.array(padded_arrays.node_positions[block_id], dtype=jnp.float32)
        block_neighbors = jnp.array(padded_arrays.element_neighbors[block_id, :block_size], dtype=jnp.int32)

        # Standard 3-level search (medium blocks don't need hash buckets)
        result = search_particles_in_block(
            particle_positions=block_positions,
            particle_element_ids=block_element_ids,
            particle_block_ids=block_ids_array,
            particle_active=block_active,
            block_id=block_id,
            block_connectivity=block_connectivity,
            block_node_positions=block_node_positions,
            block_element_neighbors=block_neighbors,
            block_size=block_size
        )

        # Update batch particles with search results
        batch_particles.element_ids[particle_indices] = np.array(result.new_element_ids)

        # Accumulate statistics
        stats.level0_hits += int(result.level0_hits)
        stats.level1_hits += int(result.level1_hits)
        stats.level2_hits += int(result.level2_hits)
        stats.not_found += int(result.not_found)

        t_block_end = time.time()
        stats.time_per_block[block_id] = t_block_end - t_block_start
        stats.particles_per_block[block_id] = n_particles_in_block

    # Process light blocks
    # Phase 2 optimization: batch them together if enabled (config.batch_light_blocks)
    # Otherwise fall back to individual processing (Phase 1 baseline)
    if len(grouping.light_blocks) > 0:
        if config.batch_light_blocks:
            # Phase 2: Batched light block processing (experimental - may be slower)
            t_light_batch_start = time.time()

            # Collect all particles in light blocks
            light_particle_indices = []
            for block_id in grouping.light_blocks:
                light_particle_indices.extend(grouping.groups[block_id])

            if len(light_particle_indices) > 0:
                # Convert to numpy array
                light_particle_indices = np.array(light_particle_indices)

                # Extract all light block particles
                light_positions = jnp.array(batch_particles.positions[light_particle_indices], dtype=jnp.float32)
                light_element_ids = jnp.array(batch_particles.element_ids[light_particle_indices], dtype=jnp.int32)
                light_block_ids = jnp.array(batch_particles.block_ids[light_particle_indices], dtype=jnp.int32)
                light_active = jnp.array(batch_particles.active_mask[light_particle_indices], dtype=jnp.bool_)

                if verbose:
                    print(f"  Processing {len(grouping.light_blocks)} light blocks (BATCHED MODE): {len(light_particle_indices)} particles")

                # Use batched light block search (Phase 2 optimization)
                result = batch_search_light_blocks(
                    particle_positions=light_positions,
                    particle_element_ids=light_element_ids,
                    particle_block_ids=light_block_ids,
                    particle_active=light_active,
                    light_block_ids=np.array(grouping.light_blocks),
                    padded_arrays=padded_arrays,
                    batch_size=16  # Process 16 blocks per iteration
                )

                # Update batch particles with search results
                batch_particles.element_ids[light_particle_indices] = np.array(result.element_ids)

                # Accumulate statistics
                stats.level0_hits += result.n_level0_hits
                stats.level1_hits += result.n_level1_hits
                stats.level2_hits += result.n_level2_hits
                stats.not_found += result.n_not_found

                t_light_batch_end = time.time()

                if verbose:
                    print(f"    Light blocks (batched): {(t_light_batch_end - t_light_batch_start)*1000:.1f}ms "
                          f"[L0:{result.n_level0_hits} L1:{result.n_level1_hits} L2:{result.n_level2_hits}]")
        else:
            # Phase 1: Individual light block processing (baseline - better performance)
            for block_id in grouping.light_blocks:
                particle_indices = grouping.groups[block_id]
                n_particles_in_block = len(particle_indices)

                if n_particles_in_block == 0:
                    continue

                t_block_start = time.time()

                # Extract particles for this block
                block_positions = jnp.array(batch_particles.positions[particle_indices], dtype=jnp.float32)
                block_element_ids = jnp.array(batch_particles.element_ids[particle_indices], dtype=jnp.int32)
                block_ids_array = jnp.full(n_particles_in_block, block_id, dtype=jnp.int32)
                block_active = jnp.array(batch_particles.active_mask[particle_indices], dtype=jnp.bool_)

                # Get block data from padded arrays
                block_size = padded_arrays.block_sizes[block_id]
                block_connectivity = padded_arrays.connectivity[block_id, :block_size]
                block_node_positions = padded_arrays.node_positions[block_id]
                block_neighbors = padded_arrays.element_neighbors[block_id, :block_size]

                # Search particles in this light block
                result = search_particles_in_block(
                    particle_positions=block_positions,
                    particle_element_ids=block_element_ids,
                    particle_block_ids=block_ids_array,
                    particle_active=block_active,
                    block_id=block_id,
                    block_connectivity=block_connectivity,
                    block_node_positions=block_node_positions,
                    block_element_neighbors=block_neighbors,
                    block_size=block_size
                )

                # Update batch particles
                batch_particles.element_ids[particle_indices] = np.array(result.element_ids)

                # Accumulate statistics
                stats.level0_hits += int(result.n_level0_hits)
                stats.level1_hits += int(result.n_level1_hits)
                stats.level2_hits += int(result.n_level2_hits)
                stats.not_found += int(result.n_not_found)

                t_block_end = time.time()
                stats.time_per_block[block_id] = t_block_end - t_block_start
                stats.particles_per_block[block_id] = n_particles_in_block

    stats.time_block_processing = time.time() - t_blocks_start

    # Track VRAM at batch end
    mem_end = monitor_batch_memory_usage(f"Batch {batch_id} end")
    stats.vram_end_mb = mem_end['used_mb']
    stats.vram_delta_mb = stats.vram_end_mb - stats.vram_start_mb

    stats.time_total = time.time() - t_start

    if verbose:
        print(f"\n  Batch complete: {stats.time_total*1000:.1f}ms total")
        print(f"  Throughput: {stats.throughput_particles_per_sec():.0f} particles/s")
        print(f"  VRAM: {stats.vram_start_mb:.0f} -> {stats.vram_end_mb:.0f} MB "
              f"({stats.vram_delta_mb:+.0f} MB)")
        print(f"{'='*60}")

    # Return updated particles (for now, return as-is since kernels are placeholders)
    return batch_particles, stats


def track_particles_batched(
    particles: ParticleData,
    padded_arrays: PaddedArrays,
    config: BatchConfig,
    search_kernel: Optional[Callable] = None,
    n_timesteps: int = 1,
    verbose: bool = True
) -> Tuple[ParticleData, ProcessorStatistics]:
    """
    Track particles using batched block-wise architecture.

    This is the main entry point for Phase 1 batched tracking.
    Implements the two-level architecture:
    - Level 1: Particle batching (to prevent OOM)
    - Level 2: Block-wise processing (within each batch)

    Based on architecture from:
    docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md (lines 1047-1159)

    Parameters
    ----------
    particles : ParticleData
        All particles to track (can be > GPU memory)
    padded_arrays : PaddedArrays
        Padded block arrays (already on GPU)
    config : BatchConfig
        Validated batch configuration
    search_kernel : Callable, optional
        JAX-compiled search kernel. If None, uses placeholder (Phase 1).
    n_timesteps : int
        Number of timesteps to advance (default: 1)
    verbose : bool
        Print progress messages (default: True)

    Returns
    -------
    final_particles : ParticleData
        Particles after n_timesteps with updated positions and cached IDs
    stats : ProcessorStatistics
        Complete processing statistics

    Notes
    -----
    Memory model:
    - Static mesh data: padded_arrays on GPU (e.g., 660 MB for ThreadedA)
    - Per-batch particle data: batch_size × 32 bytes (e.g., 12.8 MB for 200K)
    - Total peak: static + batch ≈ 673 MB << 4 GB (safe)

    Expected performance (Phase 1 baseline):
    - ThreadedA mesh (3.5M elements, 32 blocks):
      * 200K particles/batch
      * 500 particles/s baseline (5-20ms per batch)
      * Scales linearly with batch count

    Phase 2-4 optimizations will improve to 4,000 p/s target.

    Examples
    --------
    >>> # Setup
    >>> from jaxtrace.gpu.batching import create_default_config, validate_config
    >>> from jaxtrace.gpu.forest import build_padded_block_arrays
    >>>
    >>> # Configure
    >>> config = create_default_config(gpu_memory_gb=4.0)
    >>> config = validate_config(config, padded_arrays, auto_tune_batch_size=True)
    >>>
    >>> # Track particles
    >>> final_particles, stats = track_particles_batched(
    ...     particles=particles,
    ...     padded_arrays=padded_arrays,
    ...     config=config,
    ...     n_timesteps=10,
    ...     verbose=True
    ... )
    >>>
    >>> # Review results
    >>> stats.print_summary()
    >>> print(f"Average throughput: {stats.average_throughput():.0f} p/s")
    """
    t_total_start = time.time()

    # Initialize overall statistics
    stats = ProcessorStatistics(
        total_particles=particles.n_total,
        total_timesteps=n_timesteps
    )

    if verbose:
        print("\n" + "="*80)
        print("BATCHED BLOCK-WISE PARTICLE TRACKING")
        print("="*80)
        print(f"Total particles: {particles.n_total:,}")
        print(f"Active particles: {particles.n_active:,}")
        print(f"Timesteps: {n_timesteps}")
        print(f"Batch size: {config.actual_batch_size:,}")
        print(f"Estimated batches: {int(np.ceil(particles.n_active / config.actual_batch_size))}")
        print("="*80)

    # Placeholder for search kernel
    if search_kernel is None:
        # Phase 1: Use placeholder
        # Phase 2-4: This will be the JAX JIT-compiled multi-level search
        search_kernel = lambda block_id, indices: None

    # Main timestep loop
    for timestep in range(n_timesteps):
        if verbose:
            print(f"\n{'#'*80}")
            print(f"# TIMESTEP {timestep + 1}/{n_timesteps}")
            print(f"{'#'*80}")

        t_timestep_start = time.time()

        # Batch loop: Process particles in batches
        n_active = particles.n_active
        batch_size = config.actual_batch_size
        n_batches = int(np.ceil(n_active / batch_size))

        batch_id_offset = stats.total_batches

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, n_active)
            batch_id = batch_id_offset + batch_idx

            # Extract batch (CPU operation)
            # Get indices of active particles
            active_indices = np.where(particles.active_mask)[0]
            batch_indices = active_indices[batch_start:batch_end]

            # Create batch particle data
            batch_particles = ParticleData(
                positions=particles.positions[batch_indices].copy(),
                velocities=particles.velocities[batch_indices].copy(),
                element_ids=particles.element_ids[batch_indices].copy(),
                block_ids=particles.block_ids[batch_indices].copy(),
                active_mask=particles.active_mask[batch_indices].copy()
            )

            # Process this batch
            updated_batch, batch_stats = process_batch(
                batch_particles=batch_particles,
                padded_arrays=padded_arrays,
                config=config,
                search_kernel=search_kernel,
                batch_id=batch_id,
                verbose=verbose
            )

            # Update main particle array with batch results
            particles.positions[batch_indices] = updated_batch.positions
            particles.velocities[batch_indices] = updated_batch.velocities
            particles.element_ids[batch_indices] = updated_batch.element_ids
            particles.block_ids[batch_indices] = updated_batch.block_ids
            particles.active_mask[batch_indices] = updated_batch.active_mask

            # Accumulate statistics
            stats.batch_stats.append(batch_stats)
            stats.total_batches += 1

            # Accumulate search level statistics
            stats.total_level0_hits += batch_stats.level0_hits
            stats.total_level1_hits += batch_stats.level1_hits
            stats.total_level2_hits += batch_stats.level2_hits
            stats.total_not_found += batch_stats.not_found

        stats.time_per_timestep = time.time() - t_timestep_start

        if verbose:
            print(f"\nTimestep {timestep + 1} complete: {stats.time_per_timestep:.2f}s")
            print(f"Active particles: {particles.n_active:,}")

    stats.total_time = time.time() - t_total_start

    if verbose:
        stats.print_summary()

    return particles, stats


def print_batch_statistics(stats: ProcessorStatistics, detailed: bool = False):
    """
    Print detailed batch processing statistics.

    Parameters
    ----------
    stats : ProcessorStatistics
        Statistics to print
    detailed : bool
        Include per-batch breakdown (default: False)
    """
    stats.print_summary()

    if detailed and stats.batch_stats:
        print("\n" + "="*80)
        print("DETAILED BATCH BREAKDOWN")
        print("="*80)

        for batch_stat in stats.batch_stats:
            print(f"\nBatch {batch_stat.batch_id}:")
            print(f"  Particles: {batch_stat.n_particles:,}")
            print(f"  Active blocks: {batch_stat.n_active_blocks}")
            print(f"  Grouping: {batch_stat.time_grouping*1000:.1f}ms")
            print(f"  Block processing: {batch_stat.time_block_processing*1000:.1f}ms")
            print(f"  Total: {batch_stat.time_total*1000:.1f}ms")
            print(f"  Throughput: {batch_stat.throughput_particles_per_sec():.0f} p/s")
            print(f"  VRAM delta: {batch_stat.vram_delta_mb:+.1f} MB")

        print("="*80 + "\n")
