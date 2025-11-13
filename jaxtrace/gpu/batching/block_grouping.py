"""
Particle grouping by block for block-wise GPU processing.

Part of Phase 1: Setup and Validation
Implements CPU-side particle grouping logic from:
docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md (lines 86-132)

Key function:
- group_particles_by_block(): Efficiently group particles into per-block lists
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ParticleGrouping:
    """Result of grouping particles by block."""

    # Per-block particle indices
    groups: Dict[int, np.ndarray]  # {block_id: particle_indices}

    # Statistics
    n_blocks_active: int  # Number of blocks with particles
    particles_per_block: Dict[int, int]  # {block_id: count}

    # Block categories (for optimization)
    heavy_blocks: List[int]  # >10K elements
    medium_blocks: List[int]  # 1K-10K elements
    light_blocks: List[int]  # <1K elements

    def __repr__(self) -> str:
        return (
            f"ParticleGrouping(\n"
            f"  Active blocks: {self.n_blocks_active}\n"
            f"  Heavy: {len(self.heavy_blocks)}, "
            f"Medium: {len(self.medium_blocks)}, "
            f"Light: {len(self.light_blocks)}\n"
            f")"
        )


def group_particles_by_block(
    particle_block_ids: np.ndarray,
    block_sizes: np.ndarray,
    heavy_threshold: int = 10_000,
    light_threshold: int = 1_000
) -> ParticleGrouping:
    """
    Group particles by their assigned block IDs.

    This is the CPU-side grouping operation that happens once per batch.
    Based on logic from:
    docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md (lines 86-132)

    Parameters
    ----------
    particle_block_ids : np.ndarray
        Block ID for each particle, shape (n_particles,), int32
    block_sizes : np.ndarray
        Number of elements in each block, shape (n_blocks,), int32
    heavy_threshold : int
        Element count threshold for heavy blocks (default: 10K)
    light_threshold : int
        Element count threshold for light blocks (default: 1K)

    Returns
    -------
    grouping : ParticleGrouping
        Grouped particles with block categorization

    Notes
    -----
    This function performs efficient CPU-side grouping using numpy operations.
    Expected time: <5ms for 200K particles (from refined plan benchmarks)

    Block categories:
    - Heavy (>10K elem): Require hash buckets, processed individually
    - Medium (1K-10K): Processed individually
    - Light (<1K elem): Can be batched together (Phase 2 optimization)

    Examples
    --------
    >>> # After block assignment
    >>> grouping = group_particles_by_block(
    ...     particle_block_ids=particle_data.block_ids,
    ...     block_sizes=padded_arrays.block_sizes
    ... )
    >>> # Process heavy blocks first
    >>> for block_id in grouping.heavy_blocks:
    ...     indices = grouping.groups[block_id]
    ...     process_heavy_block(block_id, indices)
    """
    n_particles = len(particle_block_ids)

    # Get unique block IDs with particles
    unique_blocks = np.unique(particle_block_ids)

    # Group particles by block using efficient numpy operations
    groups = {}
    particles_per_block = {}

    for block_id in unique_blocks:
        # Find all particles in this block
        mask = particle_block_ids == block_id
        indices = np.where(mask)[0]

        groups[block_id] = indices
        particles_per_block[block_id] = len(indices)

    # Categorize blocks by element count
    heavy_blocks = []
    medium_blocks = []
    light_blocks = []

    for block_id in unique_blocks:
        n_elem = block_sizes[block_id]

        if n_elem >= heavy_threshold:
            heavy_blocks.append(int(block_id))
        elif n_elem >= light_threshold:
            medium_blocks.append(int(block_id))
        else:
            light_blocks.append(int(block_id))

    return ParticleGrouping(
        groups=groups,
        n_blocks_active=len(unique_blocks),
        particles_per_block=particles_per_block,
        heavy_blocks=heavy_blocks,
        medium_blocks=medium_blocks,
        light_blocks=light_blocks
    )


def print_grouping_stats(grouping: ParticleGrouping, stage: str = ""):
    """Print statistics about particle grouping."""
    if stage:
        print(f"\n{'='*60}")
        print(f"Particle Grouping Stats - {stage}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"Particle Grouping Stats")
        print(f"{'='*60}")

    print(f"  Active blocks: {grouping.n_blocks_active}")
    print(f"  Block categories:")
    print(f"    Heavy (>10K elem): {len(grouping.heavy_blocks)}")
    print(f"    Medium (1K-10K): {len(grouping.medium_blocks)}")
    print(f"    Light (<1K): {len(grouping.light_blocks)}")

    if grouping.particles_per_block:
        counts = list(grouping.particles_per_block.values())
        print(f"  Particles per block:")
        print(f"    Min: {min(counts):,}")
        print(f"    Max: {max(counts):,}")
        print(f"    Mean: {np.mean(counts):,.0f}")

    print(f"{'='*60}\n")


def batch_light_blocks(
    grouping: ParticleGrouping,
    max_blocks_per_batch: int = 8,
    max_particles_per_batch: int = 8_000
) -> List[List[int]]:
    """
    Batch light blocks together for efficient processing.

    This is a Phase 2 optimization from:
    docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md (lines 76-132)

    Parameters
    ----------
    grouping : ParticleGrouping
        Particle grouping from group_particles_by_block()
    max_blocks_per_batch : int
        Maximum number of light blocks to combine (default: 8)
    max_particles_per_batch : int
        Maximum particles in a combined batch (default: 8K)

    Returns
    -------
    batches : List[List[int]]
        List of light block batches, each batch is list of block IDs

    Notes
    -----
    Light block batching reduces kernel launch overhead by combining
    multiple small blocks into single GPU kernel calls.

    Expected speedup: 50-70% reduction in kernel launches for typical meshes
    (from refined plan estimates)

    Examples
    --------
    >>> batches = batch_light_blocks(grouping)
    >>> for light_batch in batches:
    ...     # Process all blocks in batch together
    ...     process_light_blocks_batched(light_batch, grouping)
    """
    if not grouping.light_blocks:
        return []

    batches = []
    current_batch = []
    current_particles = 0

    for block_id in grouping.light_blocks:
        n_particles = grouping.particles_per_block[block_id]

        # Check if adding this block would exceed limits
        if (len(current_batch) >= max_blocks_per_batch or
            current_particles + n_particles > max_particles_per_batch):
            # Start new batch
            if current_batch:
                batches.append(current_batch)
            current_batch = [block_id]
            current_particles = n_particles
        else:
            # Add to current batch
            current_batch.append(block_id)
            current_particles += n_particles

    # Add final batch
    if current_batch:
        batches.append(current_batch)

    return batches
