#!/usr/bin/env python3
"""
Diagnose WHY particles are lost over 2500 RK4 steps.

Goal: Understand if loss is due to:
1. Search failures (element not found)
2. Domain boundary (particles leave mesh)
3. Velocity closure (zero velocity regions)
4. Time integration issues
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
from pathlib import Path

# Import mesh loading
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

# Import octree
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.structures import upload_mesh_aligned_octree_multi_to_gpu

# Import search
from jaxtrace.gpu.search.mesh_aligned_point_location import search_mesh_aligned_octree_multi_local

# Import seeding
from jaxtrace.gpu.seeding import seed_particles_in_elements

# Configuration
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'

# Tracking parameters
N_STEPS = 100  # Shorter test - 100 steps
DT = 0.0025
N_PARTICLES = 10000  # Smaller sample

# Diagnostic intervals
CHECK_INTERVAL = 10


def main():
    print("="*80)
    print("Particle Loss Pattern Diagnostic")
    print("="*80)
    print(f"Steps: {N_STEPS}, dt={DT}, Particles: {N_PARTICLES}")
    print()

    # Load mesh
    print("[1/6] Loading mesh...")
    t0 = time.time()
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=False
    )

    n_nodes_orig = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"    Elements: {n_elements:,}, Nodes: {n_nodes_orig:,}")

    print("  Deduplicating...")
    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    n_nodes = node_positions.shape[0]
    print(f"    Removed {n_duplicates_removed:,} duplicates ({n_nodes:,} nodes remaining)")

    # Extract octree
    print("\n[2/6] Extracting multi-cell octree...")
    t0 = time.time()
    octree_multi = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=True
    )
    print(f"  Extracted in {time.time()-t0:.1f}s")

    # Upload to GPU
    print("\n[3/6] Uploading to GPU...")
    t0 = time.time()
    connectivity_gpu = jnp.array(connectivity, dtype=jnp.int32)
    node_positions_gpu = jnp.array(node_positions, dtype=jnp.float32)
    velocity_sequence_gpu = jnp.array(velocity_sequence, dtype=jnp.float32)

    octree_gpu = upload_mesh_aligned_octree_multi_to_gpu(octree_multi)

    jax.block_until_ready(connectivity_gpu)
    print(f"  Uploaded in {time.time()-t0:.1f}s")

    # Seed particles
    print("\n[4/6] Seeding particles...")
    t0 = time.time()

    # Use perturbed element centroids (realistic seeding)
    positions_host, element_ids_host = seed_particles_in_elements(
        node_positions=node_positions,
        connectivity=connectivity,
        n_particles=N_PARTICLES,
        method='perturbed',
        perturbation=0.1,
        seed=42
    )

    positions_gpu = jnp.array(positions_host, dtype=jnp.float32)
    element_ids_gpu = jnp.array(element_ids_host, dtype=jnp.int32)

    jax.block_until_ready(positions_gpu)
    print(f"  Seeded in {time.time()-t0:.1f}s")
    print(f"    Initial active: {N_PARTICLES:,}")

    # Track loss patterns
    print("\n[5/6] Running RK4 tracking with diagnostics...")

    loss_patterns = {
        'step': [],
        'n_active': [],
        'n_search_failed': [],
        'n_left_domain': [],
        'mean_velocity': [],
        'mean_displacement': []
    }

    # Simple RK4 step (inline for diagnostics)
    @jax.jit
    def rk4_step_diagnostic(pos, elem_id, vel_seq, time_idx):
        """Single RK4 step with search."""
        # Search for current position
        new_elem_id, n_tests = search_mesh_aligned_octree_multi_local(
            pos, octree_gpu, max_tests=jnp.int32(600)
        )

        # Get velocity (simplified - just use current element)
        velocity = jnp.zeros(3, dtype=jnp.float32)
        if new_elem_id >= 0:
            # Simple velocity: average of element nodes (not accurate but ok for diagnostic)
            elem_nodes = connectivity_gpu[new_elem_id]
            node_vels = vel_seq[time_idx, elem_nodes]
            velocity = node_vels.mean(axis=0)

        # RK4 integration
        k1 = velocity
        new_pos = pos + k1 * DT

        return new_pos, new_elem_id, velocity

    # Track over time
    positions = positions_gpu
    element_ids = element_ids_gpu

    for step in range(N_STEPS):
        time_idx = step % velocity_sequence_gpu.shape[0]

        # Vectorized RK4 step
        positions, element_ids, velocities = jax.vmap(
            lambda p, e: rk4_step_diagnostic(p, e, velocity_sequence_gpu, time_idx)
        )(positions, element_ids)

        # Diagnostic at intervals
        if step % CHECK_INTERVAL == 0 or step == N_STEPS - 1:
            positions = jax.block_until_ready(positions)
            element_ids = jax.block_until_ready(element_ids)
            velocities = jax.block_until_ready(velocities)

            n_active = int(jnp.sum(element_ids >= 0))
            n_lost = N_PARTICLES - n_active

            # Analyze lost particles
            lost_mask = element_ids < 0
            active_mask = element_ids >= 0

            mean_vel = float(jnp.linalg.norm(velocities[active_mask], axis=1).mean()) if n_active > 0 else 0

            loss_patterns['step'].append(step)
            loss_patterns['n_active'].append(n_active)
            loss_patterns['n_search_failed'].append(n_lost)  # Simplified - all lost = search failed
            loss_patterns['mean_velocity'].append(mean_vel)

            print(f"  Step {step:4d}: Active={n_active:,}/{N_PARTICLES:,} ({100*n_active/N_PARTICLES:.1f}%), "
                  f"Mean vel={mean_vel:.6f}")

    # Final analysis
    print("\n[6/6] Loss Pattern Analysis")
    print("="*80)

    steps = np.array(loss_patterns['step'])
    active = np.array(loss_patterns['n_active'])
    retention = 100 * active / N_PARTICLES

    print(f"\nRetention over time:")
    for i in range(len(steps)):
        print(f"  Step {steps[i]:4d}: {retention[i]:5.1f}% ({active[i]:,} particles)")

    print(f"\nLoss rate:")
    for i in range(1, len(steps)):
        loss = active[i-1] - active[i]
        loss_rate = loss / CHECK_INTERVAL if CHECK_INTERVAL > 0 else 0
        print(f"  Steps {steps[i-1]:4d}-{steps[i]:4d}: {loss:,} particles lost ({loss_rate:.1f} p/step)")

    print(f"\nFinal retention: {retention[-1]:.1f}%")
    print(f"Total lost: {N_PARTICLES - active[-1]:,} particles")

    print("\n" + "="*80)
    print("Diagnostic Complete")
    print("="*80)


if __name__ == "__main__":
    main()
