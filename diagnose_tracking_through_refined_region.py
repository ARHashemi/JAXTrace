#!/usr/bin/env python3
"""
Diagnose Element Assignment During Tracking Through Refined Region
===================================================================

Track a small number of particles that pass through the refined region
and monitor their element assignments at each step.

Goal: Verify if particles are assigned to FINE or COARSE elements when
they enter the refined region (rotating tool area).
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.tracking.rk4_fully_fused_timedep import create_rk4_fully_fused_timedep
from jaxtrace.gpu.tracking.initial_assignment_extended import initial_assignment_extended_batch

print("="*80)
print("TRACKING DIAGNOSTIC: ELEMENT ASSIGNMENT THROUGH REFINED REGION")
print("="*80)

# Configuration
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 159)
VELOCITY_FIELD_NAME = 'Displacement'

N_HOPS = 3
L2_SEARCH_RADIUS = 10
DT = 0.0025
N_STEPS = 500  # Track for 500 steps to ensure particles pass through refined region

# Load mesh
print("\n[1/4] Loading mesh...")
t0 = time.time()
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    MESH_BASE_PATH,
    MESH_FILE_PATTERN,
    VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=False
)
t_load = time.time() - t0
print(f"  Loaded in {t_load:.2f}s")
print(f"  Nodes: {len(node_positions):,}")
print(f"  Elements: {len(connectivity):,}")

# Identify refined region
print("\n[2/4] Identifying refined region...")
element_sizes = np.zeros(len(connectivity))
element_centroids = np.zeros((len(connectivity), 3))

for i, elem_nodes_idx in enumerate(connectivity):
    elem_nodes = node_positions[elem_nodes_idx]
    element_centroids[i] = elem_nodes.mean(axis=0)
    max_edge = max([np.linalg.norm(elem_nodes[j] - elem_nodes[k])
                    for j in range(4) for k in range(j+1, 4)])
    element_sizes[i] = max_edge

fine_threshold = np.percentile(element_sizes, 10)
fine_mask = element_sizes < fine_threshold
fine_centroids = element_centroids[fine_mask]

refined_bbox_min = fine_centroids.min(axis=0)
refined_bbox_max = fine_centroids.max(axis=0)

print(f"  Refined region bounding box:")
print(f"    X: [{refined_bbox_min[0]*1000:.2f}, {refined_bbox_max[0]*1000:.2f}] mm")
print(f"    Y: [{refined_bbox_min[1]*1000:.2f}, {refined_bbox_max[1]*1000:.2f}] mm")
print(f"    Z: [{refined_bbox_min[2]*1000:.2f}, {refined_bbox_max[2]*1000:.2f}] mm")
print(f"  Fine elements: {fine_mask.sum():,} ({100*fine_mask.sum()/len(connectivity):.1f}%)")

# Setup GPU structures
print("\n[3/4] Setting up GPU structures...")
mesh_gpu_connectivity = jax.device_put(connectivity)
mesh_gpu_node_positions = jax.device_put(node_positions)
velocity_fields_gpu = jax.device_put(velocity_sequence)

element_neighbors = build_element_neighbors_array(connectivity)
mesh_gpu_element_neighbors = jax.device_put(element_neighbors)

morton_cpu = build_global_morton_octree(node_positions, connectivity, leaf_capacity=256, verbose=False)
mesh_gpu_morton = upload_global_morton_to_gpu(morton_cpu, connectivity, node_positions)

# Create RK4 integrator
rk4_step = create_rk4_fully_fused_timedep(
    mesh_gpu_connectivity,
    mesh_gpu_node_positions,
    mesh_gpu_element_neighbors,
    mesh_gpu_morton,
    n_hops=N_HOPS,
    l2_search_radius=L2_SEARCH_RADIUS
)

# Seed test particles
print("\n[4/4] Seeding test particles...")

# Seed particles on a line from entrance to exit
# Choose Y=0, Z=-2mm (middle of refined region in YZ)
# X from -20mm to -10mm (will advect through refined region)

n_test_particles = 5
test_x = np.linspace(-0.020, -0.010, n_test_particles)  # -20mm to -10mm in meters
test_positions = np.array([[x, 0.0, -0.002] for x in test_x], dtype=np.float32)

print(f"  Seeded {n_test_particles} particles along X-axis:")
for i, pos in enumerate(test_positions):
    print(f"    Particle {i}: ({pos[0]*1000:.2f}, {pos[1]*1000:.2f}, {pos[2]*1000:.2f}) mm")

# Initial assignment
positions_gpu = jax.device_put(test_positions)
element_ids_gpu = initial_assignment_extended_batch(positions_gpu, mesh_gpu_morton, max_radius=100)

element_ids_cpu = np.array(element_ids_gpu)
n_assigned = (element_ids_cpu >= 0).sum()
print(f"\n  Initial assignment: {n_assigned}/{n_test_particles} particles assigned")

if n_assigned < n_test_particles:
    print(f"  ⚠️  WARNING: {n_test_particles - n_assigned} particles not assigned!")
    for i, elem_id in enumerate(element_ids_cpu):
        if elem_id < 0:
            print(f"    Particle {i} at ({test_positions[i]*1000}) NOT ASSIGNED")

# Track particles and record element assignments
print(f"\n{'='*80}")
print(f"TRACKING {n_test_particles} PARTICLES FOR {N_STEPS} STEPS")
print(f"{'='*80}\n")

# Storage for tracking history
particle_history = {
    i: {
        'positions': [],
        'element_ids': [],
        'element_types': [],  # 'fine', 'coarse', or 'lost'
        'in_refined_region': []
    }
    for i in range(n_test_particles)
}

# Track particles
for step in range(N_STEPS):
    positions_cpu = np.array(positions_gpu)
    element_ids_cpu = np.array(element_ids_gpu)

    # Record current state
    for i in range(n_test_particles):
        pos = positions_cpu[i]
        elem_id = element_ids_cpu[i]

        # Check if in refined region
        in_refined = (elem_id >= 0 and
                     np.all(pos >= refined_bbox_min) and
                     np.all(pos <= refined_bbox_max))

        # Determine element type
        if elem_id < 0:
            elem_type = 'lost'
        elif fine_mask[elem_id]:
            elem_type = 'fine'
        else:
            elem_type = 'coarse'

        particle_history[i]['positions'].append(pos.copy())
        particle_history[i]['element_ids'].append(elem_id)
        particle_history[i]['element_types'].append(elem_type)
        particle_history[i]['in_refined_region'].append(in_refined)

    # Integrate one step
    positions_gpu, element_ids_gpu = rk4_step(
        positions_gpu,
        element_ids_gpu,
        DT,
        velocity_fields_gpu,
        step % len(velocity_sequence)
    )

    # Print progress every 100 steps
    if (step + 1) % 100 == 0:
        n_active = (np.array(element_ids_gpu) >= 0).sum()
        print(f"  Step {step+1:4d}: {n_active}/{n_test_particles} particles active")

# Analysis
print(f"\n{'='*80}")
print("ANALYSIS")
print(f"{'='*80}\n")

for i in range(n_test_particles):
    history = particle_history[i]

    print(f"Particle {i}:")
    print(f"  Initial position: ({test_positions[i][0]*1000:.2f}, {test_positions[i][1]*1000:.2f}, {test_positions[i][2]*1000:.2f}) mm")

    # Find when particle entered refined region
    entered_refined = None
    exited_refined = None

    for step_idx, in_refined in enumerate(history['in_refined_region']):
        if in_refined and entered_refined is None:
            entered_refined = step_idx
        if not in_refined and entered_refined is not None and exited_refined is None:
            exited_refined = step_idx

    if entered_refined is not None:
        print(f"  ✅ Entered refined region at step {entered_refined}")

        # Analyze element types while in refined region
        steps_in_refined = []
        if exited_refined is not None:
            steps_in_refined = range(entered_refined, exited_refined)
            print(f"  ✅ Exited refined region at step {exited_refined} (spent {exited_refined - entered_refined} steps)")
        else:
            steps_in_refined = range(entered_refined, len(history['in_refined_region']))
            print(f"  Still in refined region (from step {entered_refined} onwards)")

        # Count element types in refined region
        elem_types_in_refined = [history['element_types'][s] for s in steps_in_refined]
        n_fine = elem_types_in_refined.count('fine')
        n_coarse = elem_types_in_refined.count('coarse')
        n_lost = elem_types_in_refined.count('lost')

        print(f"  Element types while in refined region:")
        print(f"    Fine elements: {n_fine}/{len(elem_types_in_refined)} ({100*n_fine/len(elem_types_in_refined):.1f}%)")
        print(f"    Coarse elements: {n_coarse}/{len(elem_types_in_refined)} ({100*n_coarse/len(elem_types_in_refined):.1f}%)")
        print(f"    Lost: {n_lost}/{len(elem_types_in_refined)} ({100*n_lost/len(elem_types_in_refined):.1f}%)")

        if n_coarse > n_fine:
            print(f"  ❌ PROBLEM: Mostly assigned to COARSE elements in refined region!")
        elif n_fine > n_coarse:
            print(f"  ✅ Good: Mostly assigned to FINE elements in refined region")

        # Show first few steps in refined region
        print(f"  First 5 steps in refined region:")
        for idx, step_idx in enumerate(list(steps_in_refined)[:5]):
            pos = history['positions'][step_idx]
            elem_id = history['element_ids'][step_idx]
            elem_type = history['element_types'][step_idx]
            print(f"    Step {step_idx}: elem {elem_id:7d} ({elem_type:6s}), pos=({pos[0]*1000:.2f}, {pos[1]*1000:.2f}, {pos[2]*1000:.2f}) mm")

    else:
        print(f"  ❌ Never entered refined region")
        # Show final position
        final_pos = history['positions'][-1]
        print(f"  Final position: ({final_pos[0]*1000:.2f}, {final_pos[1]*1000:.2f}, {final_pos[2]*1000:.2f}) mm")

    print()

print("="*80)
