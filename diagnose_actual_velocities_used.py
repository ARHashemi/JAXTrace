#!/usr/bin/env python3
"""
Diagnose Actual Velocities Being Used by Particles
===================================================

Sample particles across mesh and check:
1. Which element they're assigned to
2. What velocity that element has
3. Compare refined vs coarse regions
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.tracking.initial_assignment_extended import initial_assignment_extended_batch
from jaxtrace.gpu.tracking.rk4_fully_fused_timedep import create_rk4_fully_fused_timedep
from jaxtrace.gpu.forest import build_element_neighbors_array

print("="*80)
print("DIAGNOSING ACTUAL VELOCITIES USED BY PARTICLES")
print("="*80)

# Load mesh
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule"),
    "featurelessAvtk_{timestep}.pvtu",
    (120, 122),
    field_name='Displacement',
    verbose=False
)

velocity_field = velocity_sequence[0]

# Identify refined vs coarse regions
print("\nClassifying mesh regions...")
element_sizes = np.zeros(len(connectivity))
element_velocities = np.zeros(len(connectivity))

for i in range(len(connectivity)):
    elem_nodes = node_positions[connectivity[i]]
    elem_vels = velocity_field[connectivity[i]]

    max_edge = max([np.linalg.norm(elem_nodes[j] - elem_nodes[k])
                    for j in range(4) for k in range(j+1, 4)])
    element_sizes[i] = max_edge
    element_velocities[i] = np.linalg.norm(elem_vels, axis=1).mean()

fine_threshold = np.percentile(element_sizes, 10)
coarse_threshold = np.percentile(element_sizes, 90)

fine_mask = element_sizes < fine_threshold
coarse_mask = element_sizes > coarse_threshold

print(f"  Fine elements (<{fine_threshold*1000:.4f}mm): {fine_mask.sum():,}")
print(f"  Coarse elements (>{coarse_threshold*1000:.4f}mm): {coarse_mask.sum():,}")

fine_vel_mean = element_velocities[fine_mask].mean()
coarse_vel_mean = element_velocities[coarse_mask].mean()

print(f"\n  Fine region mean |vel|: {fine_vel_mean:.6e} m/s")
print(f"  Coarse region mean |vel|: {coarse_vel_mean:.6e} m/s")
print(f"  Ratio: {fine_vel_mean/coarse_vel_mean:.2f}x")

# Setup tracking
print("\nSetting up tracking structures...")
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
    n_hops=3,
    l2_search_radius=2
)

# Sample particles from fine and coarse regions
n_sample = 100

fine_elem_indices = np.where(fine_mask)[0]
coarse_elem_indices = np.where(coarse_mask)[0]

fine_sample_indices = np.random.choice(fine_elem_indices, n_sample, replace=False)
coarse_sample_indices = np.random.choice(coarse_elem_indices, n_sample, replace=False)

# Place particles at element centroids
fine_positions = np.array([node_positions[connectivity[i]].mean(axis=0)
                          for i in fine_sample_indices], dtype=np.float32)
coarse_positions = np.array([node_positions[connectivity[i]].mean(axis=0)
                            for i in coarse_sample_indices], dtype=np.float32)

# Initial assignment
print("\nPerforming initial assignment...")
fine_pos_gpu = jax.device_put(fine_positions)
coarse_pos_gpu = jax.device_put(coarse_positions)

fine_assigned = np.array(initial_assignment_extended_batch(fine_pos_gpu, mesh_gpu_morton, max_radius=100))
coarse_assigned = np.array(initial_assignment_extended_batch(coarse_pos_gpu, mesh_gpu_morton, max_radius=100))

fine_assigned_success = (fine_assigned >= 0).sum()
coarse_assigned_success = (coarse_assigned >= 0).sum()

print(f"  Fine region: {fine_assigned_success}/{n_sample} assigned ({100*fine_assigned_success/n_sample:.1f}%)")
print(f"  Coarse region: {coarse_assigned_success}/{n_sample} assigned ({100*coarse_assigned_success/n_sample:.1f}%)")

# Check assigned velocities
fine_assigned_velocities = []
for i, elem_id in enumerate(fine_assigned):
    if elem_id >= 0:
        actual_elem = fine_sample_indices[i]
        assigned_vel = element_velocities[elem_id]
        expected_vel = element_velocities[actual_elem]
        fine_assigned_velocities.append({
            'expected_elem': actual_elem,
            'assigned_elem': elem_id,
            'expected_vel': expected_vel,
            'assigned_vel': assigned_vel,
            'correct': (elem_id == actual_elem)
        })

coarse_assigned_velocities = []
for i, elem_id in enumerate(coarse_assigned):
    if elem_id >= 0:
        actual_elem = coarse_sample_indices[i]
        assigned_vel = element_velocities[elem_id]
        expected_vel = element_velocities[actual_elem]
        coarse_assigned_velocities.append({
            'expected_elem': actual_elem,
            'assigned_elem': elem_id,
            'expected_vel': expected_vel,
            'assigned_vel': assigned_vel,
            'correct': (elem_id == actual_elem)
        })

# Analyze assignment correctness
fine_correct = sum(1 for v in fine_assigned_velocities if v['correct'])
coarse_correct = sum(1 for v in coarse_assigned_velocities if v['correct'])

print(f"\nAssignment correctness:")
print(f"  Fine region: {fine_correct}/{len(fine_assigned_velocities)} correct ({100*fine_correct/len(fine_assigned_velocities):.1f}%)")
print(f"  Coarse region: {coarse_correct}/{len(coarse_assigned_velocities)} correct ({100*coarse_correct/len(coarse_assigned_velocities):.1f}%)")

# Analyze velocity errors for incorrect assignments
fine_incorrect = [v for v in fine_assigned_velocities if not v['correct']]
if fine_incorrect:
    vel_errors = [abs(v['assigned_vel'] - v['expected_vel']) for v in fine_incorrect]
    print(f"\nFine region incorrect assignments ({len(fine_incorrect)}):")
    print(f"  Mean velocity error: {np.mean(vel_errors):.6e} m/s")
    print(f"  Max velocity error: {np.max(vel_errors):.6e} m/s")
    print(f"  Sample:")
    for i, v in enumerate(fine_incorrect[:5]):
        print(f"    {i+1}. Expected elem {v['expected_elem']} (vel={v['expected_vel']:.6e})")
        print(f"       Got elem {v['assigned_elem']} (vel={v['assigned_vel']:.6e})")

# Track one particle for 10 steps
print(f"\n" + "="*80)
print("TRACKING ONE PARTICLE FOR 10 STEPS")
print("="*80)

test_elem = fine_sample_indices[0]
test_pos = fine_positions[0:1]
test_elem_id = fine_assigned[0:1]

print(f"\nStarting in fine region:")
print(f"  Element: {test_elem}")
print(f"  Element size: {element_sizes[test_elem]*1000:.4f} mm")
print(f"  Element velocity: {element_velocities[test_elem]:.6e} m/s")

DT = 0.0025
pos_gpu = jax.device_put(test_pos)
elem_gpu = jax.device_put(test_elem_id.astype(np.int32))

for step in range(10):
    pos_before = np.array(pos_gpu[0])
    elem_before = int(elem_gpu[0])

    pos_gpu, elem_gpu = rk4_step(pos_gpu, elem_gpu, DT, velocity_fields_gpu, step % 3)

    pos_after = np.array(pos_gpu[0])
    elem_after = int(elem_gpu[0])

    displacement = np.linalg.norm(pos_after - pos_before)

    if elem_after >= 0:
        elem_vel = element_velocities[elem_after]
        expected_disp = elem_vel * DT
        ratio = displacement / expected_disp if expected_disp > 0 else 0

        print(f"  Step {step}: elem {elem_after:7d}, |v|={elem_vel:.6e}, "
              f"disp={displacement*1000:.6f}mm, ratio={ratio:.3f}")
    else:
        print(f"  Step {step}: LOST")
        break

print("="*80)

EOF
