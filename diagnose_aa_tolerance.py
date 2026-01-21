#!/usr/bin/env python3
"""
Diagnose why AA detection finds only 0.06% axis-aligned elements.

Check a sample of elements to see how close they are to axis-aligned.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

# Load mesh
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"

print("Loading mesh...")
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    base_path=MESH_BASE_PATH,
    file_pattern=MESH_FILE_PATTERN,
    timestep_range=(120, 121),
    field_name='Displacement',
    verbose=False
)

print("Deduplicating...")
node_positions, connectivity, _, _ = deduplicate_nodes(
    node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
)

print(f"\nMesh: {connectivity.shape[0]:,} elements\n")

# Sample elements
n_sample = 20
sample_ids = np.random.choice(connectivity.shape[0], n_sample, replace=False)

print("Checking sample elements for axis-alignment:\n")
print("=" * 80)

def check_edge_alignment(edge, tol_abs):
    """Check if edge is aligned to X, Y, or Z axis."""
    dx, dy, dz = np.abs(edge)
    edge_len = np.sqrt(dx**2 + dy**2 + dz**2)

    # Check each axis
    if dy < tol_abs and dz < tol_abs and dx > tol_abs:
        return 'X', dy, dz, edge_len
    elif dx < tol_abs and dz < tol_abs and dy > tol_abs:
        return 'Y', dx, dz, edge_len
    elif dx < tol_abs and dy < tol_abs and dz > tol_abs:
        return 'Z', dx, dy, edge_len
    else:
        return None, dy, dz, edge_len

for elem_id in sample_ids[:5]:  # Show first 5 in detail
    nodes = node_positions[connectivity[elem_id]]
    p0, p1, p2, p3 = nodes[0], nodes[1], nodes[2], nodes[3]

    print(f"\nElement {elem_id}:")
    print(f"  p0: {p0}")
    print(f"  p1: {p1}")
    print(f"  p2: {p2}")
    print(f"  p3: {p3}")

    # Check all edges
    edges = [
        ('p0→p1', p1 - p0),
        ('p0→p2', p2 - p0),
        ('p0→p3', p3 - p0),
        ('p1→p2', p2 - p1),
        ('p1→p3', p3 - p1),
        ('p2→p3', p3 - p2),
    ]

    print(f"\n  Edge analysis:")
    for edge_name, edge in edges:
        edge_len = np.linalg.norm(edge)

        # Try different tolerances
        for tol_name, tol_abs in [('1e-10×L', 1e-10 * edge_len),
                                   ('1e-8×L', 1e-8 * edge_len),
                                   ('1e-6×L', 1e-6 * edge_len),
                                   ('1e-12', 1e-12)]:
            axis, perp1, perp2, _ = check_edge_alignment(edge, tol_abs)
            if axis:
                print(f"    {edge_name:8s} L={edge_len:.6e}  →  {axis}-aligned (tol={tol_name}, perp=[{perp1:.2e}, {perp2:.2e}])")
                break
        else:
            # Not aligned with any tolerance
            dx, dy, dz = np.abs(edge)
            print(f"    {edge_name:8s} L={edge_len:.6e}  →  NOT aligned |Δx|={dx:.2e} |Δy|={dy:.2e} |Δz|={dz:.2e}")

    print()

print("=" * 80)

# Test with different tolerances
print("\nAA Detection with Different Tolerances:\n")
print("=" * 80)

def count_aa_elements(connectivity, node_positions, tol_base):
    """Count how many elements are AA with given tolerance."""
    n_aa = 0

    for elem_id in range(min(10000, connectivity.shape[0])):  # Sample first 10K
        nodes = node_positions[connectivity[elem_id]]
        p0, p1, p2, p3 = nodes[0], nodes[1], nodes[2], nodes[3]

        # Check each vertex as potential right-angle corner
        vertices = [p0, p1, p2, p3]
        found_aa = False

        for v_idx in range(4):
            p_base = vertices[v_idx]
            other_indices = [i for i in range(4) if i != v_idx]
            edges = [vertices[i] - p_base for i in other_indices]

            aligned_axes = []
            edge_lengths = []

            for edge in edges:
                dx, dy, dz = np.abs(edge)
                edge_len = np.sqrt(dx**2 + dy**2 + dz**2)
                tol_abs = tol_base * edge_len

                if dy < tol_abs and dz < tol_abs and dx > tol_abs:
                    aligned_axes.append(0)  # X
                    edge_lengths.append(edge_len)
                elif dx < tol_abs and dz < tol_abs and dy > tol_abs:
                    aligned_axes.append(1)  # Y
                    edge_lengths.append(edge_len)
                elif dx < tol_abs and dy < tol_abs and dz > tol_abs:
                    aligned_axes.append(2)  # Z
                    edge_lengths.append(edge_len)

            # Found 3 orthogonal aligned edges?
            if len(aligned_axes) == 3 and len(set(aligned_axes)) == 3:
                found_aa = True
                break

        if found_aa:
            n_aa += 1

    return n_aa

sample_size = min(10000, connectivity.shape[0])
print(f"Testing on {sample_size:,} elements:\n")

for tol_base in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
    n_aa = count_aa_elements(connectivity, node_positions, tol_base)
    percentage = (n_aa / sample_size) * 100
    print(f"  tol = {tol_base:.0e} (relative):  {n_aa:,}/{sample_size:,} = {percentage:.2f}% AA")

print("\n" + "=" * 80)
print("Recommendation: Adjust tol_base in aa_detection.py if needed")
print("=" * 80)
