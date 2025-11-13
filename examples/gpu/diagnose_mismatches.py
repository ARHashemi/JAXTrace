"""
Diagnostic script to analyze CPU/GPU mismatches.

Checks if mismatches are due to boundary particles (expected) or bugs (unexpected).
"""

import numpy as np
from jaxtrace.gpu import ParticleData, update_particle_element_ids, GPUParticleTracker
from jaxtrace.gpu.forest import (
    create_regular_forest_grid,
    assign_elements_to_blocks,
    build_element_adjacency,
    position_to_block_id,
)
from jaxtrace.gpu.search import point_in_element
from pathlib import Path
import vtk
from vtk.util import numpy_support


def check_mismatch_particles(
    particles_cpu,
    particles_gpu,
    positions_mesh,
    connectivity,
    tolerance=1e-6
):
    """
    Analyze particles where CPU and GPU disagree.

    Checks if both element IDs are valid (boundary case) or if one is wrong (bug).
    """
    mismatches = particles_cpu.element_ids != particles_gpu.element_ids
    mismatch_indices = np.where(mismatches)[0]

    print(f"\n🔍 Analyzing {len(mismatch_indices)} mismatches...\n")

    for idx in mismatch_indices:
        point = particles_cpu.positions[idx]
        cpu_elem = particles_cpu.element_ids[idx]
        gpu_elem = particles_gpu.element_ids[idx]

        print(f"Particle {idx}:")
        print(f"  Position: [{point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}]")
        print(f"  CPU element: {cpu_elem}")
        print(f"  GPU element: {gpu_elem}")

        # Check if point is in CPU element
        if cpu_elem >= 0:
            cpu_nodes = connectivity[cpu_elem]
            cpu_vertices = positions_mesh[cpu_nodes]
            in_cpu = point_in_element(point, cpu_vertices)
            print(f"  Point in CPU element: {in_cpu}")

        # Check if point is in GPU element
        if gpu_elem >= 0:
            gpu_nodes = connectivity[gpu_elem]
            gpu_vertices = positions_mesh[gpu_nodes]
            in_gpu = point_in_element(point, gpu_vertices)
            print(f"  Point in GPU element: {in_gpu}")

        # Check if elements are neighbors (boundary case)
        if cpu_elem >= 0 and gpu_elem >= 0:
            # Compute distance between element centroids
            cpu_centroid = np.mean(positions_mesh[connectivity[cpu_elem]], axis=0)
            gpu_centroid = np.mean(positions_mesh[connectivity[gpu_elem]], axis=0)
            dist = np.linalg.norm(cpu_centroid - gpu_centroid)
            print(f"  Element centroid distance: {dist:.6f} m")

            if dist < 0.01:  # Elements are close
                print(f"  ✅ Likely boundary particle (elements are neighbors)")
            else:
                print(f"  ⚠️  Elements far apart - possible bug!")

        print()


if __name__ == "__main__":
    # This script can be imported and used in the notebook
    # Or run standalone with mesh data
    print("Mismatch diagnostic tool ready.")
    print("Import and call check_mismatch_particles() to analyze mismatches.")
