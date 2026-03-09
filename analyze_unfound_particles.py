#!/usr/bin/env python3
"""
Analyze unfound particles to determine if they're in void regions.

Hypothesis: The 25.4% unfound particles are in bbox regions with no tetrahedra.
"""

import numpy as np
from pathlib import Path

# Import JAX
import jax
import jax.numpy as jnp

from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.search.mesh_aligned_point_location import search_mesh_aligned_octree_batch
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

print(f"{'='*80}")
print("Analysis: Are Unfound Particles in Void Regions?")
print(f"{'='*80}\n")

# Load mesh
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'

print("Loading mesh...")
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    base_path=MESH_BASE_PATH,
    file_pattern=MESH_FILE_PATTERN,
    timestep_range=VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=False
)

node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
    node_positions,
    connectivity,
    velocity_sequence=velocity_sequence,
    verbose=False
)

print(f"  Loaded {connectivity.shape[0]:,} elements, {node_positions.shape[0]:,} nodes\n")

# Extract octree cells
print("Extracting octree cells...")
cells = extract_octree_cells_single(
    node_positions,
    connectivity,
    tolerance=1e-6,
    verbose=False
)
print(f"  Unique cells: {cells.n_cells:,}\n")

# Upload to GPU
print("Uploading to GPU...")
octree_gpu = upload_mesh_aligned_octree_to_gpu(
    node_positions=node_positions,
    connectivity=connectivity,
    octree_cells=cells,
    verbose=False
)
print("  Done\n")

# Generate test particles
bbox_min = node_positions.min(axis=0)
bbox_max = node_positions.max(axis=0)

print(f"Generating 10,000 random test particles...")
np.random.seed(42)
n_particles = 10000
particle_positions_cpu = np.random.uniform(
    low=bbox_min,
    high=bbox_max,
    size=(n_particles, 3)
).astype(np.float32)

particle_positions_gpu = jnp.array(particle_positions_cpu)

# Search
print(f"Searching for {n_particles:,} particles...")
found_elements, n_tests = search_mesh_aligned_octree_batch(
    particle_positions_gpu,
    octree_gpu,
    max_tests=100
)
jax.block_until_ready((found_elements, n_tests))

found_elements_cpu = np.array(found_elements)
n_tests_cpu = np.array(n_tests)

n_found = np.sum(found_elements_cpu >= 0)
n_unfound = np.sum(found_elements_cpu == -1)

print(f"  Found: {n_found:,} / {n_particles:,} ({100.0 * n_found / n_particles:.1f}%)")
print(f"  Unfound: {n_unfound:,} / {n_particles:,} ({100.0 * n_unfound / n_particles:.1f}%)\n")

# Brute force check: Are unfound particles actually inside ANY element?
print(f"{'='*80}")
print(f"Brute Force Verification: Are Unfound Particles in Void Regions?")
print(f"{'='*80}\n")

def point_in_tet_cpu(pos, v0, v1, v2, v3, tolerance=1e-6):
    """CPU point-in-tet test using barycentric coordinates."""
    v0p = pos - v0
    v01 = v1 - v0
    v02 = v2 - v0
    v03 = v3 - v0

    # Build matrix [v01, v02, v03]
    mat = np.stack([v01, v02, v03], axis=1)  # 3x3

    try:
        bary = np.linalg.solve(mat, v0p)  # (b1, b2, b3)
    except np.linalg.LinAlgError:
        return False

    b0 = 1.0 - bary[0] - bary[1] - bary[2]

    # Check if all barycentric coords are in [0, 1]
    return (b0 >= -tolerance and b0 <= 1.0 + tolerance and
            bary[0] >= -tolerance and bary[0] <= 1.0 + tolerance and
            bary[1] >= -tolerance and bary[1] <= 1.0 + tolerance and
            bary[2] >= -tolerance and bary[2] <= 1.0 + tolerance)

# Sample unfound particles for brute force check
unfound_mask = found_elements_cpu == -1
unfound_indices = np.where(unfound_mask)[0]

n_sample = min(100, len(unfound_indices))
sample_indices = unfound_indices[:n_sample]

print(f"Brute-force checking {n_sample} unfound particles against ALL {connectivity.shape[0]:,} elements...")
print(f"(This will take a while...)\n")

actually_inside = 0
actually_outside = 0

for idx_count, idx in enumerate(sample_indices):
    pos = particle_positions_cpu[idx]

    # Check all elements
    found_in_any = False
    for elem_id in range(connectivity.shape[0]):
        node_ids = connectivity[elem_id]
        vertices = node_positions[node_ids]

        if point_in_tet_cpu(pos, vertices[0], vertices[1], vertices[2], vertices[3]):
            found_in_any = True
            break

    if found_in_any:
        actually_inside += 1
    else:
        actually_outside += 1

    if (idx_count + 1) % 10 == 0:
        print(f"  Checked {idx_count + 1}/{n_sample}...")

print(f"\n{'='*80}")
print(f"RESULTS")
print(f"{'='*80}\n")

print(f"Sampled {n_sample} unfound particles:")
print(f"  Actually INSIDE some element: {actually_inside} ({100.0 * actually_inside / n_sample:.1f}%)")
print(f"  Actually OUTSIDE all elements (void): {actually_outside} ({100.0 * actually_outside / n_sample:.1f}%)")
print()

if actually_outside > 0.9 * n_sample:
    print("✅ HYPOTHESIS CONFIRMED!")
    print(f"   {100.0 * actually_outside / n_sample:.1f}% of unfound particles are in VOID REGIONS.")
    print("   The mesh does NOT fill the entire bounding box.")
    print("   The current 74.6% searchability is CORRECT and represents actual mesh coverage.")
    print()
    print("   Estimated mesh volume coverage: ~74.6% of bbox")
    print("   Estimated void volume: ~25.4% of bbox")
elif actually_inside > 0.9 * n_sample:
    print("❌ ALGORITHM STILL HAS BUGS!")
    print(f"   {100.0 * actually_inside / n_sample:.1f}% of unfound particles are INSIDE elements.")
    print("   The search algorithm is missing particles that should be found.")
    print("   Possible causes:")
    print("   - Elements spanning multiple cells (need neighbor search)")
    print("   - Precision issues in grid computation")
    print("   - Morton encoding issues")
else:
    print("⚠️  MIXED RESULTS")
    print(f"   {100.0 * actually_inside / n_sample:.1f}% inside, {100.0 * actually_outside / n_sample:.1f}% outside")
    print("   Some unfound particles are in voids, some are algorithm misses.")
    print("   Need further investigation.")

print(f"\n{'='*80}\n")
