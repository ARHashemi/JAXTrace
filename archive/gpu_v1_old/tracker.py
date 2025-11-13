"""
GPU Particle Tracker.

High-level interface for GPU-accelerated particle tracking with forest-of-octrees.
Handles device transfer, batching, and statistics collection.
"""

import time
from typing import Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from .particles import ParticleData, partition_particles_by_block
from .search import SearchStatistics
from .kernels import (
    find_containing_elements_batch,
    positions_to_block_ids_batch,
    build_block_element_lists,
)


class GPUParticleTracker:
    """
    GPU-accelerated particle tracker with block-level batching.

    This class manages:
    - Device transfer (CPU ↔ GPU)
    - Block-level batching for parallelism
    - Element search statistics
    - Memory management

    Usage:
        # Create tracker
        tracker = GPUParticleTracker(
            positions, connectivity, neighbors, element_to_block,
            domain_bounds, grid_size
        )

        # Update particle elements (GPU-accelerated)
        particles_updated = tracker.update_particle_elements(particles)

        # Print statistics
        tracker.print_statistics()
    """

    def __init__(
        self,
        positions: np.ndarray,
        connectivity: np.ndarray,
        element_neighbors: np.ndarray,
        element_to_block: np.ndarray,
        domain_bounds: np.ndarray,
        grid_size: Tuple[int, int, int]
    ):
        """
        Initialize GPU particle tracker.

        Args:
            positions: Node positions [N_nodes, 3]
            connectivity: Element connectivity [N_elements, 4]
            element_neighbors: Neighbor array [N_elements, max_neighbors]
            element_to_block: Element-to-block mapping [N_elements]
            domain_bounds: Domain bounds [6] (xmin, xmax, ymin, ymax, zmin, zmax)
            grid_size: Grid dimensions (nx, ny, nz)
        """
        print("\n🚀 Initializing GPU Particle Tracker...")

        # Store domain info
        self.domain_bounds = jnp.array(domain_bounds, dtype=jnp.float32)
        self.grid_size = grid_size
        self.n_blocks = grid_size[0] * grid_size[1] * grid_size[2]

        # Build block element lists (CPU preprocessing)
        print("  Building block element lists...")
        block_elements, block_counts = build_block_element_lists(
            element_to_block, self.n_blocks
        )

        # Store mesh data (transfer to GPU device)
        print("  Transferring mesh data to GPU...")
        self.positions_gpu = jax.device_put(jnp.array(positions, dtype=jnp.float32))
        self.connectivity_gpu = jax.device_put(jnp.array(connectivity, dtype=jnp.int32))
        self.element_neighbors_gpu = jax.device_put(jnp.array(element_neighbors, dtype=jnp.int32))
        self.block_elements_gpu = jax.device_put(jnp.array(block_elements, dtype=jnp.int32))
        self.block_counts_gpu = jax.device_put(jnp.array(block_counts, dtype=jnp.int32))

        # Statistics
        self.stats = SearchStatistics()
        self.total_update_time = 0.0
        self.n_updates = 0

        # Memory usage
        mesh_memory = (
            positions.nbytes +
            connectivity.nbytes +
            element_neighbors.nbytes +
            block_elements.nbytes +
            block_counts.nbytes
        ) / 1024**2

        print(f"  ✅ GPU initialization complete")
        print(f"  Mesh memory on GPU: {mesh_memory:.1f} MB")
        print(f"  Block element lists: {block_elements.shape}")
        print(f"  Grid size: {grid_size[0]}×{grid_size[1]}×{grid_size[2]} = {self.n_blocks} blocks")

    def update_block_ids(self, particles: ParticleData) -> ParticleData:
        """
        Update block IDs for all active particles.

        Uses fast O(1) position → block_id mapping on GPU.

        Args:
            particles: Particle data with positions

        Returns:
            Updated particle data with new block_ids
        """
        # Transfer positions to GPU
        positions_gpu = jax.device_put(jnp.array(particles.positions, dtype=jnp.float32))

        # Compute block IDs on GPU
        block_ids_gpu = positions_to_block_ids_batch(
            positions_gpu,
            self.domain_bounds,
            self.grid_size
        )

        # Transfer back to CPU
        block_ids = np.array(block_ids_gpu, dtype=np.int32)

        # Update particles
        particles_updated = particles.copy()
        particles_updated.block_ids[:] = block_ids

        return particles_updated

    def update_particle_elements(
        self,
        particles: ParticleData,
        batch_size: Optional[int] = None
    ) -> ParticleData:
        """
        Update element IDs for all active particles using GPU.

        Uses three-tier search with block-level batching for parallelism.

        Args:
            particles: Particle data with cached element/block IDs
            batch_size: Optional batch size for large particle counts.
                       If None, processes all particles at once.

        Returns:
            Updated particle data with new element_ids

        Note:
            For >100K particles, consider using batch_size to avoid GPU memory issues.
        """
        start_time = time.time()

        # Update block IDs first
        particles_with_blocks = self.update_block_ids(particles)

        # Get active particles
        active_mask = particles_with_blocks.active_mask
        n_active = np.sum(active_mask)

        if n_active == 0:
            return particles_with_blocks

        # Transfer particle data to GPU
        positions_active = particles_with_blocks.positions[active_mask]
        element_ids_active = particles_with_blocks.element_ids[active_mask]
        block_ids_active = particles_with_blocks.block_ids[active_mask]

        positions_gpu = jax.device_put(jnp.array(positions_active, dtype=jnp.float32))
        element_ids_gpu = jax.device_put(jnp.array(element_ids_active, dtype=jnp.int32))
        block_ids_gpu = jax.device_put(jnp.array(block_ids_active, dtype=jnp.int32))

        # Process in batches if requested
        if batch_size is None:
            batch_size = n_active

        n_batches = (n_active + batch_size - 1) // batch_size
        new_element_ids = np.zeros(n_active, dtype=np.int32)

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_active)

            # Get batch
            pos_batch = positions_gpu[start_idx:end_idx]
            elem_batch = element_ids_gpu[start_idx:end_idx]
            block_batch = block_ids_gpu[start_idx:end_idx]

            # Search on GPU
            new_elem_batch = find_containing_elements_batch(
                pos_batch,
                elem_batch,
                block_batch,
                self.element_neighbors_gpu,
                self.block_elements_gpu,
                self.block_counts_gpu,
                self.positions_gpu,
                self.connectivity_gpu
            )

            # Transfer back to CPU
            new_element_ids[start_idx:end_idx] = np.array(new_elem_batch, dtype=np.int32)

        # Update statistics (compare with cached IDs)
        for i in range(n_active):
            cached_id = element_ids_active[i]
            new_id = new_element_ids[i]

            if new_id < 0:
                self.stats.failures += 1
            elif cached_id == new_id and cached_id >= 0:
                self.stats.level0_hits += 1
            else:
                # For simplicity, count non-cached hits as level2
                # (more detailed tracking would require kernel modification)
                self.stats.level2_hits += 1

        # Update particles
        particles_updated = particles_with_blocks.copy()
        particles_updated.element_ids[active_mask] = new_element_ids

        # Deactivate particles that left domain
        left_domain = (new_element_ids < 0)
        active_indices = np.where(active_mask)[0]
        particles_updated.active_mask[active_indices[left_domain]] = False

        # Track timing
        elapsed = time.time() - start_time
        self.total_update_time += elapsed
        self.n_updates += 1

        return particles_updated

    def update_particle_elements_by_block(
        self,
        particles: ParticleData
    ) -> ParticleData:
        """
        Update element IDs with block-level batching.

        This version partitions particles by block and processes each block
        separately for better memory locality.

        Args:
            particles: Particle data

        Returns:
            Updated particle data

        Note:
            This is the recommended method for large particle counts.
        """
        start_time = time.time()

        # Update block IDs
        particles_with_blocks = self.update_block_ids(particles)

        # Partition by block
        partition = partition_particles_by_block(particles_with_blocks, self.n_blocks)

        # Process each block
        particles_updated = particles_with_blocks.copy()

        for block_id in range(self.n_blocks):
            particle_indices = partition[block_id]
            if len(particle_indices) == 0:
                continue

            # Get particles in this block
            positions_block = particles_with_blocks.positions[particle_indices]
            element_ids_block = particles_with_blocks.element_ids[particle_indices]
            block_ids_block = particles_with_blocks.block_ids[particle_indices]

            # Transfer to GPU
            positions_gpu = jax.device_put(jnp.array(positions_block, dtype=jnp.float32))
            element_ids_gpu = jax.device_put(jnp.array(element_ids_block, dtype=jnp.int32))
            block_ids_gpu = jax.device_put(jnp.array(block_ids_block, dtype=jnp.int32))

            # Search on GPU
            new_element_ids_gpu = find_containing_elements_batch(
                positions_gpu,
                element_ids_gpu,
                block_ids_gpu,
                self.element_neighbors_gpu,
                self.element_to_block_gpu,
                self.positions_gpu,
                self.connectivity_gpu
            )

            # Transfer back
            new_element_ids = np.array(new_element_ids_gpu, dtype=np.int32)

            # Update
            particles_updated.element_ids[particle_indices] = new_element_ids

            # Update statistics
            for i in range(len(particle_indices)):
                cached_id = element_ids_block[i]
                new_id = new_element_ids[i]

                if new_id < 0:
                    self.stats.failures += 1
                elif cached_id == new_id and cached_id >= 0:
                    self.stats.level0_hits += 1
                else:
                    self.stats.level2_hits += 1

            # Deactivate particles that left domain
            left_domain = new_element_ids < 0
            particles_updated.active_mask[particle_indices[left_domain]] = False

        # Track timing
        elapsed = time.time() - start_time
        self.total_update_time += elapsed
        self.n_updates += 1

        return particles_updated

    def print_statistics(self):
        """Print tracker statistics."""
        print("\n📊 GPU Particle Tracker Statistics:")

        if self.n_updates > 0:
            avg_time = self.total_update_time / self.n_updates
            print(f"  Total updates: {self.n_updates}")
            print(f"  Total time: {self.total_update_time:.3f} s")
            print(f"  Average time per update: {avg_time:.3f} s")

        # Search statistics
        self.stats.print_statistics()

    def reset_statistics(self):
        """Reset all statistics."""
        self.stats.reset()
        self.total_update_time = 0.0
        self.n_updates = 0
