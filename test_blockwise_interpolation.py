"""
Test block-wise velocity interpolation with optimized transfers.

This test validates the CORRECT approach:
- Process one block at a time (avoid GPU OOM)
- Upload block data ONCE per block
- Process on GPU
- Download results
- No redundant transfers within each block

This is the approach that should be integrated into ParticleTimeMarcher.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

from jaxtrace.gpu.mesh_loader import load_mesh_complete
from jaxtrace.gpu.search.initial_assignment import assign_particles_to_mesh_optimized
from jaxtrace.gpu.particles import ParticleData
from jaxtrace.gpu.tracking import batch_interpolate_velocities
from jaxtrace.gpu.batching.block_grouping import group_particles_by_block


def create_constant_velocity_field(padded_arrays, velocity=(1.0, 0.0, 0.0)):
    """Create constant velocity field for all blocks."""
    n_blocks = len(padded_arrays.block_sizes)
    max_nodes = padded_arrays.node_positions.shape[1]

    velocity_field = np.zeros((n_blocks, max_nodes, 3), dtype=np.float32)

    for block_id in range(n_blocks):
        n_nodes = padded_arrays.block_sizes[block_id]
        velocity_field[block_id, :n_nodes, :] = velocity

    return velocity_field


def interpolate_velocities_blockwise_optimized(
    particle_data: ParticleData,
    velocity_field_all_blocks: np.ndarray,
    connectivity_gpu: jnp.ndarray,  # Pre-uploaded, persistent
    node_positions_gpu: jnp.ndarray,  # Pre-uploaded, persistent
    padded_arrays,
    verbose: bool = False
) -> tuple[np.ndarray, dict]:
    """
    Interpolate velocities using block-wise processing with optimized transfers.

    CORRECT APPROACH (per user's feedback):
    - Process one block at a time (avoid GPU OOM)
    - Upload each block's data ONCE
    - Keep connectivity and node_positions on GPU (persistent)
    - No redundant transfers within each block

    This is the pattern that should be used in ParticleTimeMarcher.
    """
    t_start = time.time()

    n_particles = len(particle_data.positions)
    velocities = np.zeros((n_particles, 3), dtype=np.float32)

    # Group particles by block (CPU-side, only computes indices)
    grouping = group_particles_by_block(
        particle_data.block_ids,
        padded_arrays.block_sizes
    )

    stats = {
        'n_particles': n_particles,
        'n_blocks_active': len(grouping.groups),
        'time_per_block': [],
        'particles_per_block': [],
    }

    # Process each block
    for block_id, particle_indices in grouping.groups.items():
        if len(particle_indices) == 0:
            continue

        t_block_start = time.time()
        n_block_particles = len(particle_indices)

        # Extract data for this block (CPU)
        block_positions = particle_data.positions[particle_indices]
        block_element_ids = particle_data.element_ids[particle_indices]

        # Upload block data to GPU (ONCE per block)
        block_positions_gpu = jax.device_put(block_positions)
        block_element_ids_gpu = jax.device_put(block_element_ids)
        block_velocity_field_gpu = jax.device_put(velocity_field_all_blocks[block_id])

        # Interpolate on GPU (using persistent connectivity and node_positions)
        block_velocities = batch_interpolate_velocities(
            block_positions_gpu,
            block_element_ids_gpu,
            connectivity_gpu,
            node_positions_gpu,
            block_velocity_field_gpu
        )

        # Download results (ONCE per block)
        velocities[particle_indices] = np.array(block_velocities)

        t_block_end = time.time()
        block_time = t_block_end - t_block_start

        stats['time_per_block'].append(block_time)
        stats['particles_per_block'].append(n_block_particles)

        if verbose and n_block_particles > 0:
            throughput = n_block_particles / block_time if block_time > 0 else 0
            print(f"  Block {block_id}: {n_block_particles} particles, "
                  f"{block_time*1000:.1f} ms, {throughput:.1f} p/s")

    t_end = time.time()
    stats['time_total'] = t_end - t_start
    stats['throughput'] = n_particles / stats['time_total'] if stats['time_total'] > 0 else 0

    return velocities, stats


def test_blockwise_interpolation():
    """Test block-wise velocity interpolation with various particle counts."""

    print("=" * 80)
    print("Block-Wise Velocity Interpolation Test")
    print("=" * 80)
    print()

    # Load mesh
    print("Loading ThreadedA mesh...")
    exodus_file = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/ThreadedA.e"
    mesh_data = load_mesh_complete(exodus_file, step=0)

    print(f"  Mesh: {len(mesh_data.connectivity)} elements, "
          f"{len(mesh_data.node_positions)} nodes, "
          f"{len(mesh_data.padded_arrays.block_sizes)} blocks")
    print()

    # Pre-upload mesh to GPU (persistent)
    print("Uploading mesh to GPU (one-time)...")
    connectivity_gpu = jax.device_put(mesh_data.connectivity)
    node_positions_gpu = jax.device_put(mesh_data.node_positions)
    print("  ✓ Mesh on GPU")
    print()

    # Create constant velocity field
    velocity_field = create_constant_velocity_field(
        mesh_data.padded_arrays,
        velocity=(1.0, 0.0, 0.0)
    )

    # Test with different particle counts
    particle_counts = [100, 1000, 10000]

    for n_particles in particle_counts:
        print(f"Test with {n_particles} particles")
        print("-" * 80)

        # Create initial particles
        bbox = mesh_data.bbox
        np.random.seed(42)
        initial_positions = np.random.uniform(
            low=[bbox.min_x, bbox.min_y, bbox.min_z],
            high=[bbox.max_x, bbox.max_y, bbox.max_z],
            size=(n_particles, 3)
        ).astype(np.float32)

        # Assign to mesh
        print(f"  Assigning {n_particles} particles to mesh...")
        particle_data = assign_particles_to_mesh_optimized(
            initial_positions,
            mesh_data.connectivity,
            mesh_data.node_positions,
            mesh_data.padded_arrays
        )

        n_active = np.sum(particle_data.active)
        print(f"  Active particles: {n_active}/{n_particles}")

        # Filter to active particles
        active_mask = particle_data.active
        particle_data_active = ParticleData(
            positions=particle_data.positions[active_mask],
            velocities=particle_data.velocities[active_mask],
            element_ids=particle_data.element_ids[active_mask],
            block_ids=particle_data.block_ids[active_mask],
            active=particle_data.active[active_mask]
        )

        # Interpolate velocities
        print(f"  Interpolating velocities (block-wise)...")
        velocities, stats = interpolate_velocities_blockwise_optimized(
            particle_data_active,
            velocity_field,
            connectivity_gpu,
            node_positions_gpu,
            mesh_data.padded_arrays,
            verbose=(n_particles <= 1000)
        )

        # Verify results
        print()
        print(f"  Results:")
        print(f"    Particles processed: {stats['n_particles']}")
        print(f"    Active blocks: {stats['n_blocks_active']}")
        print(f"    Total time: {stats['time_total']*1000:.1f} ms")
        print(f"    Throughput: {stats['throughput']:.1f} particles/second")

        # Check velocity correctness
        expected_velocity = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        velocity_errors = np.linalg.norm(velocities - expected_velocity, axis=1)
        max_error = np.max(velocity_errors)
        mean_error = np.mean(velocity_errors)

        print(f"    Velocity accuracy:")
        print(f"      Max error: {max_error:.6e}")
        print(f"      Mean error: {mean_error:.6e}")

        if max_error < 1e-5:
            print("    ✓ Velocity interpolation CORRECT")
        else:
            print(f"    ✗ Velocity interpolation FAILED (max error: {max_error:.6e})")

        print()

    print("=" * 80)
    print("Block-Wise Interpolation Test Complete")
    print("=" * 80)
    print()
    print("Key Insights:")
    print("  1. Block-wise processing avoids GPU OOM")
    print("  2. Each block's data uploaded ONCE")
    print("  3. Mesh stays on GPU (persistent)")
    print("  4. No redundant transfers within each block")
    print("  5. This pattern should be used in ParticleTimeMarcher")


if __name__ == "__main__":
    test_blockwise_interpolation()
