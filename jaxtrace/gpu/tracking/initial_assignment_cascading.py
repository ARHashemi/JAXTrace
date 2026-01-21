#!/usr/bin/env python3
"""
Cascading Initial Assignment for Refined Mesh Regions

Uses progressive search radius expansion to assign particles, starting with small radius
for most particles and increasing only for those that fail. This is memory-efficient
and ensures particles in refined regions get assigned correctly.

Strategy:
1. Search all particles with radius=10 (fast, catches most particles)
2. For unassigned particles, search with radius=50
3. For still unassigned, search with radius=100
4. For still unassigned, search with radius=200 (last resort)

This avoids the memory overhead of radius=500 for all particles while ensuring
that particles in challenging locations (refined regions, boundaries) eventually get assigned.
"""
import time
import jax
import jax.numpy as jnp
from jaxtrace.gpu.search.morton_global_search import MeshGPUGlobalMorton
from .initial_assignment_extended import search_L2_extended_single



def initial_assignment_cascading(
    positions_gpu: jax.Array,
    mesh_gpu_global_morton: MeshGPUGlobalMorton,
    radii: list[int] = [10, 50, 100, 200],
    verbose: bool = True
) -> jax.Array:
    """
    Cascading initial assignment with progressive radius expansion.

    Uses multiple passes with increasing search radii, only searching unassigned
    particles in each subsequent pass. This is much more memory-efficient than
    using large radius for all particles.

    Args:
        positions_gpu: (N, 3) float32 - particle positions
        mesh_gpu_global_morton: GPU-resident Morton structure
        radii: List of search radii to try (default: [10, 50, 100, 200])
        verbose: Print progress information

    Returns:
        element_ids: (N,) int32 - assigned elements, -1 if not found
    """
    n_particles = positions_gpu.shape[0]

    # Initialize all as unassigned
    element_ids = jnp.full(n_particles, -1, dtype=jnp.int32)

    if verbose:
        print(f"\n  Cascading search with radii: {radii}")

    for radius in radii:
        # Find unassigned particles
        unassigned_mask = element_ids < 0
        n_unassigned = jnp.sum(unassigned_mask)

        if n_unassigned == 0:
            if verbose:
                print(f"    radius={radius:3d}: All particles assigned, skipping")
            break

        # Extract positions of unassigned particles
        # NOTE: We create a dense array of unassigned positions (not masked array)
        # This is more efficient for JAX and avoids ragged arrays
        unassigned_indices = jnp.where(unassigned_mask, jnp.arange(n_particles), -1)
        # Get actual indices (filter out -1)
        valid_unassigned = jnp.where(unassigned_mask)[0]
        unassigned_positions = positions_gpu[valid_unassigned]

        if verbose:
            print(f"    radius={radius:3d}: Searching {int(n_unassigned):,} unassigned particles...")

        # Search only unassigned particles
        @jax.jit
        def search_unassigned_batch(positions):
            return jax.vmap(
                lambda pos: search_L2_extended_single(pos, mesh_gpu_global_morton, radius)
            )(positions)

        found_elements = search_unassigned_batch(unassigned_positions)

        # Update element_ids for particles that were found
        # Use scatter update to avoid Python loop
        element_ids = element_ids.at[valid_unassigned].set(found_elements)

        # Count successes in this round
        newly_assigned = jnp.sum((found_elements >= 0).astype(jnp.int32))

        if verbose:
            total_assigned = jnp.sum((element_ids >= 0).astype(jnp.int32))
            print(f"              Found: {int(newly_assigned):,} particles "
                  f"(total: {int(total_assigned):,}/{n_particles:,}, "
                  f"{100*int(total_assigned)/n_particles:.1f}%)")

    # Final statistics
    n_assigned = jnp.sum((element_ids >= 0).astype(jnp.int32))

    if verbose:
        print(f"  Final assignment: {int(n_assigned):,}/{n_particles:,} "
              f"({100*int(n_assigned)/n_particles:.2f}%)")
        if n_assigned < n_particles:
            n_failed = n_particles - int(n_assigned)
            print(f"  ⚠️  {n_failed:,} particles could not be assigned")

    return element_ids


def initial_assignment_cascading_fallback(
    positions_gpu: jax.Array,
    mesh_gpu_global_morton: MeshGPUGlobalMorton,
    initial_radius: int = 100,
    fallback_radii: list[int] = [200, 500, 1000],
    verbose: bool = True
) -> jax.Array:
    """
    Two-stage cascading: Fast initial pass, then cascading fallback for failures.

    This is optimized for the common case where most particles assign quickly,
    but some (in refined regions or boundaries) need larger search radius.

    Args:
        positions_gpu: (N, 3) float32 - particle positions
        mesh_gpu_global_morton: GPU-resident Morton structure
        initial_radius: Radius for first pass on ALL particles (default: 100)
        fallback_radii: Radii for cascading search on failures (default: [200, 500, 1000])
        verbose: Print progress information

    Returns:
        element_ids: (N,) int32 - assigned elements, -1 if not found
    """
    n_particles = positions_gpu.shape[0]

    if verbose:
        print(f"\n  Initial search (radius={initial_radius}) for all particles...")
        t_start = time.time()

    # First pass: Search all particles with initial radius
    @jax.jit
    def search_all_batch(positions):
        return jax.vmap(
            lambda pos: search_L2_extended_single(pos, mesh_gpu_global_morton, initial_radius)
        )(positions)

    element_ids = search_all_batch(positions_gpu)

    n_assigned = jnp.sum((element_ids >= 0).astype(jnp.int32))

    if verbose:
        t_elapsed = time.time() - t_start
        print(f"    Assigned: {int(n_assigned):,}/{n_particles:,} "
              f"({100*int(n_assigned)/n_particles:.2f}%)")
        print(f"    Search Time: {t_elapsed:.2f} s")

    # Cascading fallback for unassigned particles
    if n_assigned < n_particles and fallback_radii:
        if verbose:
            print(f"\n  Cascading fallback search for {n_particles - int(n_assigned):,} unassigned particles...")

        for radius in fallback_radii:
            unassigned_mask = element_ids < 0
            n_unassigned = jnp.sum(unassigned_mask)

            if n_unassigned == 0:
                if verbose:
                    print(f"    radius={radius:3d}: All particles assigned, stopping")
                break

            valid_unassigned = jnp.where(unassigned_mask)[0]
            unassigned_positions = positions_gpu[valid_unassigned]

            if verbose:
                print(f"    radius={radius:4d}: Searching {int(n_unassigned):,} particles...")
                t_start = time.time()

            @jax.jit
            def search_unassigned_batch(positions):
                return jax.vmap(
                    lambda pos: search_L2_extended_single(pos, mesh_gpu_global_morton, radius)
                )(positions)

            found_elements = search_unassigned_batch(unassigned_positions)
            element_ids = element_ids.at[valid_unassigned].set(found_elements)

            newly_assigned = jnp.sum((found_elements >= 0).astype(jnp.int32))
            total_assigned = jnp.sum((element_ids >= 0).astype(jnp.int32))

            if verbose:
                t_elapsed = time.time() - t_start
                print(f"               Search Time: {t_elapsed:.2f} s")
                print(f"               Found: {int(newly_assigned):,} "
                      f"(total: {int(total_assigned):,}/{n_particles:,}, "
                      f"{100*int(total_assigned)/n_particles:.2f}%)")

    # Final report
    n_assigned_final = jnp.sum((element_ids >= 0).astype(jnp.int32))

    if verbose:
        print(f"\n  Final assignment: {int(n_assigned_final):,}/{n_particles:,} "
              f"({100*int(n_assigned_final)/n_particles:.2f}%)")
        if n_assigned_final < n_particles:
            n_failed = n_particles - int(n_assigned_final)
            print(f"  ⚠️  {n_failed:,} particles could not be assigned")
            print(f"     These particles are likely outside the mesh domain")

    return element_ids
