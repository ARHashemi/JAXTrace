#!/usr/bin/env python3
"""
Test Cascading Initial Assignment
==================================

Run the cascading initial assignment and verify:
1. Assignment success rate
2. How many particles found at each cascade stage
3. Memory efficiency vs single large radius
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
from jaxtrace.gpu.tracking.initial_assignment_cascading import initial_assignment_cascading_fallback
from jaxtrace.tracking.seeding import uniform_grid_seeds

print("="*80)
print("TESTING CASCADING INITIAL ASSIGNMENT")
print("="*80)

# Configuration (from production script)
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 159)
VELOCITY_FIELD_NAME = 'Displacement'

# Particle seeding (from production script)
PARTICLE_GRID_RESOLUTION = (50, 90, 50)
PARTICLE_BOUNDS_FRACTION = {
    'x': (0.1, 0.3),
    'y': (0.2, 0.8),
    'z': (0.3, 1.0),
}
N_X = PARTICLE_GRID_RESOLUTION[0]
N_Y = PARTICLE_GRID_RESOLUTION[1]
N_Z = PARTICLE_GRID_RESOLUTION[2]
SEED = 42

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
print(f"  Velocity timesteps: {len(velocity_sequence)}")

# Analyze mesh refinement
element_sizes = np.zeros(len(connectivity))
for i, elem_nodes_idx in enumerate(connectivity):
    elem_nodes = node_positions[elem_nodes_idx]
    max_edge = max([np.linalg.norm(elem_nodes[j] - elem_nodes[k])
                    for j in range(4) for k in range(j+1, 4)])
    element_sizes[i] = max_edge

fine_threshold = np.percentile(element_sizes, 10)
fine_mask = element_sizes < fine_threshold

print(f"\n  Element size distribution:")
print(f"    Min: {element_sizes.min()*1000:.4f} mm")
print(f"    10th percentile (refined): {fine_threshold*1000:.4f} mm")
print(f"    Median: {np.median(element_sizes)*1000:.4f} mm")
print(f"    90th percentile (coarse): {np.percentile(element_sizes, 90)*1000:.4f} mm")
print(f"    Max: {element_sizes.max()*1000:.4f} mm")
print(f"  Fine elements (<{fine_threshold*1000:.4f}mm): {fine_mask.sum():,} ({100*fine_mask.sum()/len(connectivity):.1f}%)")

# Setup GPU structures
print("\n[2/4] Setting up GPU structures...")
mesh_gpu_connectivity = jax.device_put(connectivity)
mesh_gpu_node_positions = jax.device_put(node_positions)

element_neighbors = build_element_neighbors_array(connectivity)
mesh_gpu_element_neighbors = jax.device_put(element_neighbors)

t0 = time.time()
morton_cpu = build_global_morton_octree(node_positions, connectivity, leaf_capacity=256, verbose=False)
mesh_gpu_morton = upload_global_morton_to_gpu(morton_cpu, connectivity, node_positions)
t_morton = time.time() - t0
print(f"  Morton octree built in {t_morton:.2f}s")

# Generate particles
print("\n[3/4] Generating particles...")
bbox_min = node_positions.min(axis=0)
bbox_max = node_positions.max(axis=0)

# Apply bounds fraction
domain_size = bbox_max - bbox_min
x_bounds = (bbox_min[0] + PARTICLE_BOUNDS_FRACTION['x'][0] * domain_size[0],
            bbox_min[0] + PARTICLE_BOUNDS_FRACTION['x'][1] * domain_size[0])
y_bounds = (bbox_min[1] + PARTICLE_BOUNDS_FRACTION['y'][0] * domain_size[1],
            bbox_min[1] + PARTICLE_BOUNDS_FRACTION['y'][1] * domain_size[1])
z_bounds = (bbox_min[2] + PARTICLE_BOUNDS_FRACTION['z'][0] * domain_size[2],
            bbox_min[2] + PARTICLE_BOUNDS_FRACTION['z'][1] * domain_size[2])

seeding_bbox_min = np.array([x_bounds[0], y_bounds[0], z_bounds[0]], dtype=np.float32)
seeding_bbox_max = np.array([x_bounds[1], y_bounds[1], z_bounds[1]], dtype=np.float32)

np.random.seed(SEED)
seeding_bounds = np.array([seeding_bbox_min, seeding_bbox_max], dtype=np.float32)
particle_positions = uniform_grid_seeds((N_X, N_Y, N_Z), seeding_bounds)

print(f"  Generated {len(particle_positions):,} particles ({N_X}×{N_Y}×{N_Z})")
print(f"  Seeding region: [{seeding_bbox_min[0]*1000:.2f}, {seeding_bbox_max[0]*1000:.2f}] × "
      f"[{seeding_bbox_min[1]*1000:.2f}, {seeding_bbox_max[1]*1000:.2f}] × "
      f"[{seeding_bbox_min[2]*1000:.2f}, {seeding_bbox_max[2]*1000:.2f}] mm")

# Check particles in refined region
fine_centroids = np.array([node_positions[connectivity[i]].mean(axis=0)
                          for i in np.where(fine_mask)[0]])
refined_bbox_min = fine_centroids.min(axis=0)
refined_bbox_max = fine_centroids.max(axis=0)

particles_in_refined = sum(1 for pos in particle_positions
                          if np.all(pos >= refined_bbox_min) and np.all(pos <= refined_bbox_max))

print(f"\n  Refined region bounding box:")
print(f"    X: [{refined_bbox_min[0]*1000:.2f}, {refined_bbox_max[0]*1000:.2f}] mm")
print(f"    Y: [{refined_bbox_min[1]*1000:.2f}, {refined_bbox_max[1]*1000:.2f}] mm")
print(f"    Z: [{refined_bbox_min[2]*1000:.2f}, {refined_bbox_max[2]*1000:.2f}] mm")
print(f"  Particles in refined region: {particles_in_refined:,}/{len(particle_positions):,} "
      f"({100*particles_in_refined/len(particle_positions):.2f}%)")

if particles_in_refined < 100:
    print(f"\n  ⚠️  WARNING: Only {particles_in_refined} particles in refined region!")
    print(f"      This may be insufficient to capture rotating velocities.")
else:
    print(f"  ✅ Sufficient particles in refined region for rotation capture")

# Run cascading initial assignment
print("\n[4/4] Running cascading initial assignment...")
positions_gpu = jax.device_put(particle_positions)

INITIAL_RADIUS = 100
FALLBACK_RADII = [200, 500, 1000]

print(f"  Initial radius: {INITIAL_RADIUS} (all particles)")
print(f"  Fallback radii: {FALLBACK_RADII} (only unassigned particles)")

t0 = time.time()
element_ids = initial_assignment_cascading_fallback(
    positions_gpu,
    mesh_gpu_morton,
    initial_radius=INITIAL_RADIUS,
    fallback_radii=FALLBACK_RADII,
    verbose=True
)
t_assign = time.time() - t0

element_ids_cpu = np.array(element_ids)

# Analyze results
n_assigned = (element_ids_cpu >= 0).sum()
n_unassigned = (element_ids_cpu < 0).sum()
success_rate = 100 * n_assigned / len(element_ids_cpu)

print(f"\n  Assignment completed in {t_assign:.2f}s")
print(f"  Assigned: {n_assigned:,}/{len(element_ids_cpu):,} ({success_rate:.2f}%)")
print(f"  Unassigned: {n_unassigned:,}")

# Check assignment accuracy for particles in refined region
particles_in_refined_indices = [i for i, pos in enumerate(particle_positions)
                                if np.all(pos >= refined_bbox_min) and np.all(pos <= refined_bbox_max)]

if len(particles_in_refined_indices) > 0:
    refined_assigned = sum(1 for i in particles_in_refined_indices if element_ids_cpu[i] >= 0)
    refined_success_rate = 100 * refined_assigned / len(particles_in_refined_indices)

    print(f"\n  Refined region assignment:")
    print(f"    Particles: {len(particles_in_refined_indices):,}")
    print(f"    Assigned: {refined_assigned:,} ({refined_success_rate:.2f}%)")

    # Check if assigned to fine elements
    assigned_to_fine = 0
    assigned_to_coarse = 0
    for i in particles_in_refined_indices:
        elem_id = element_ids_cpu[i]
        if elem_id >= 0:
            if fine_mask[elem_id]:
                assigned_to_fine += 1
            else:
                assigned_to_coarse += 1

    print(f"    Assigned to fine elements: {assigned_to_fine:,} ({100*assigned_to_fine/refined_assigned:.1f}%)")
    print(f"    Assigned to coarse elements: {assigned_to_coarse:,} ({100*assigned_to_coarse/refined_assigned:.1f}%)")

    if assigned_to_coarse > assigned_to_fine:
        print(f"\n  ❌ PROBLEM: Most particles in refined region assigned to COARSE elements!")
        print(f"     This explains why rotation is not captured.")
    else:
        print(f"  ✅ Good: Most particles in refined region assigned to FINE elements")

print("="*80)
