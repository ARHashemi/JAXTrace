#!/usr/bin/env python3
"""
Diagnose Particle Seeding Distribution
========================================

Check if particles are being seeded in the refined region (rotating tool area).
For friction stir welding, particles MUST be seeded in the refined region to
capture rotating velocities.
"""

import numpy as np
from pathlib import Path

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.tracking.seeding import uniform_grid_seeds

print("="*80)
print("PARTICLE SEEDING DISTRIBUTION ANALYSIS")
print("="*80)

# Load mesh to identify refined region
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule"),
    "featurelessAvtk_{timestep}.pvtu",
    (120, 120),
    field_name='Displacement',
    verbose=False
)

velocity_field = velocity_sequence[0]

# Identify refined region (smallest 10% of elements)
print("\nIdentifying refined region...")
element_sizes = np.zeros(len(connectivity))
element_centroids = np.zeros((len(connectivity), 3))
element_velocities = np.zeros(len(connectivity))

for i in range(len(connectivity)):
    elem_nodes = node_positions[connectivity[i]]
    elem_vels = velocity_field[connectivity[i]]

    centroid = elem_nodes.mean(axis=0)
    element_centroids[i] = centroid

    max_edge = max([np.linalg.norm(elem_nodes[j] - elem_nodes[k])
                    for j in range(4) for k in range(j+1, 4)])
    element_sizes[i] = max_edge
    element_velocities[i] = np.linalg.norm(elem_vels, axis=1).mean()

fine_threshold = np.percentile(element_sizes, 10)
fine_mask = element_sizes < fine_threshold
fine_centroids = element_centroids[fine_mask]

print(f"  Fine elements: {fine_mask.sum():,} (<{fine_threshold*1000:.4f}mm)")

# Find bounding box of refined region
refined_bbox_min = fine_centroids.min(axis=0)
refined_bbox_max = fine_centroids.max(axis=0)

print(f"\nRefined region bounding box:")
print(f"  X: [{refined_bbox_min[0]*1000:.2f}, {refined_bbox_max[0]*1000:.2f}] mm")
print(f"  Y: [{refined_bbox_min[1]*1000:.2f}, {refined_bbox_max[1]*1000:.2f}] mm")
print(f"  Z: [{refined_bbox_min[2]*1000:.2f}, {refined_bbox_max[2]*1000:.2f}] mm")

refined_size = refined_bbox_max - refined_bbox_min
print(f"  Size: {refined_size[0]*1000:.2f} x {refined_size[1]*1000:.2f} x {refined_size[2]*1000:.2f} mm")

# Generate particles using production config
bbox_min = node_positions.min(axis=0)
bbox_max = node_positions.max(axis=0)

# Production seeding parameters
N_X, N_Y, N_Z = 50, 70, 30
SEED = 42

np.random.seed(SEED)
particle_positions = uniform_grid_seeds(
    bbox_min, bbox_max,
    N_X, N_Y, N_Z
)

print(f"\nParticle seeding:")
print(f"  Grid: {N_X} × {N_Y} × {N_Z} = {len(particle_positions):,} particles")
print(f"  Domain: [{bbox_min[0]*1000:.2f}, {bbox_max[0]*1000:.2f}] × "
      f"[{bbox_min[1]*1000:.2f}, {bbox_max[1]*1000:.2f}] × "
      f"[{bbox_min[2]*1000:.2f}, {bbox_max[2]*1000:.2f}] mm")

# Check how many particles are in refined region
particles_in_refined = 0
for pos in particle_positions:
    in_refined = np.all(pos >= refined_bbox_min) and np.all(pos <= refined_bbox_max)
    if in_refined:
        particles_in_refined += 1

particles_in_refined_pct = 100 * particles_in_refined / len(particle_positions)

print(f"\nParticles in refined region:")
print(f"  Count: {particles_in_refined:,}/{len(particle_positions):,} ({particles_in_refined_pct:.2f}%)")

# Compute refined region volume vs total volume
total_volume = np.prod(bbox_max - bbox_min)
refined_volume = np.prod(refined_size)
refined_volume_pct = 100 * refined_volume / total_volume

print(f"\nVolume analysis:")
print(f"  Total domain volume: {total_volume*1e9:.2e} mm³")
print(f"  Refined region volume: {refined_volume*1e9:.2e} mm³ ({refined_volume_pct:.2f}%)")

print(f"\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if particles_in_refined < 100:
    print(f"\n❌ CRITICAL: Only {particles_in_refined} particles in refined region!")
    print(f"   For friction stir welding, you need particles in the rotating tool area.")
    print(f"\n   RECOMMENDATIONS:")
    print(f"   1. Add localized particle seeding in refined region")
    print(f"   2. Increase particle density (current: {N_X}×{N_Y}×{N_Z})")
    print(f"   3. Use adaptive seeding based on element size")

    # Suggest better seeding density
    cells_per_refined_dim = [int(np.ceil(refined_size[i] / (bbox_max[i]-bbox_min[i]) * N_dim))
                              for i, N_dim in enumerate([N_X, N_Y, N_Z])]
    print(f"\n   Suggested refined region grid: {cells_per_refined_dim[0]} × {cells_per_refined_dim[1]} × {cells_per_refined_dim[2]}")

    # Suggest additional localized particles
    suggested_additional = cells_per_refined_dim[0] * cells_per_refined_dim[1] * cells_per_refined_dim[2]
    print(f"   Suggested additional particles in refined region: ~{suggested_additional}")

elif particles_in_refined < 500:
    print(f"\n⚠️  WARNING: Only {particles_in_refined} particles in refined region")
    print(f"   This may be insufficient to capture rotating velocities.")
    print(f"   Consider adding more particles in the tool area.")

else:
    print(f"\n✅ Good: {particles_in_refined} particles in refined region")
    print(f"   Should be sufficient to capture rotating velocities.")

# Check velocity characteristics in refined vs coarse regions
fine_vel_mean = element_velocities[fine_mask].mean()
coarse_mask = element_sizes > np.percentile(element_sizes, 90)
coarse_vel_mean = element_velocities[coarse_mask].mean()

print(f"\nVelocity characteristics:")
print(f"  Fine region (rotating tool): {fine_vel_mean:.4f} m/s")
print(f"  Coarse region (advancing): {coarse_vel_mean:.4f} m/s")
print(f"  Ratio: {fine_vel_mean/coarse_vel_mean:.2f}x")

if fine_vel_mean > coarse_vel_mean * 2:
    print(f"\n✅ Refined region has significantly higher velocities (rotating motion)")
    print(f"   Particles MUST be assigned to these elements to show rotation!")

print("="*80)
