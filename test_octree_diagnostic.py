#!/usr/bin/env python3
"""
Focused Octree Diagnostic Test

This test systematically diagnoses octree construction and search issues by:
1. Loading mesh and building octree
2. Generating test particles at element centroids (with tiny perturbations)
3. Checking if elements are assigned to correct octree leaves
4. Checking if particles navigate to the same leaves as their containing elements
5. Analyzing mismatches to identify root cause
"""

import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Tuple

# Force CPU-GPU memory management
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# JAXTrace imports
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.search.octree_builder import build_octree_for_level, flatten_octree_to_arrays
from jaxtrace.gpu.search.octree_search_gpu import search_level2_octree_scan
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.search.level0_cached import point_in_tet_jax
from jaxtrace.gpu.forest import build_element_neighbors_array

jax.config.update("jax_enable_x64", True)


def compute_element_centroids(node_positions: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    """Compute element centroids."""
    n_elements = len(connectivity)
    centroids = np.zeros((n_elements, 3), dtype=np.float32)

    for i in range(n_elements):
        node_ids = connectivity[i]
        centroids[i] = node_positions[node_ids].mean(axis=0)

    return centroids


def compute_element_level_field(node_positions: np.ndarray, connectivity: np.ndarray, mesh_path: Path) -> np.ndarray:
    """Compute per-element LEVEL field from mesh."""
    import vtk
    from vtk.util import numpy_support

    # Read mesh to get LEVEL field
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(mesh_path))
    reader.Update()
    mesh_vtk = reader.GetOutput()

    # Get LEVEL from point data
    if mesh_vtk.GetPointData().HasArray('LEVEL'):
        level_field = numpy_support.vtk_to_numpy(mesh_vtk.GetPointData().GetArray('LEVEL'))
        print(f"  ✓ Found LEVEL in point data: {len(level_field):,} nodes")
        print(f"    Node levelset range: [{level_field.min():.6f}, {level_field.max():.6f}]")
    else:
        print("  ✗ LEVEL field not found, using zeros")
        level_field = np.zeros(len(node_positions), dtype=np.float32)

    # Compute per-element levelset (max of element's nodes)
    n_elements = len(connectivity)
    element_levels = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_elements):
        node_ids = connectivity[i]
        element_levels[i] = level_field[node_ids].max()

    print(f"  ✓ Computed element levelset: {n_elements:,} elements")
    print(f"    Element levelset range: [{element_levels.min():.6f}, {element_levels.max():.6f}]")

    return element_levels


def navigate_to_leaf_cpu(position: np.ndarray, octree_metadata: np.ndarray) -> Tuple[int, int]:
    """
    Navigate particle to leaf using same logic as GPU search.

    Returns:
        (leaf_node_id, depth)
    """
    node_id = 0  # Start at root
    depth = 0

    while True:
        # Get node metadata
        is_leaf = octree_metadata[node_id, 0]
        bbox_min = octree_metadata[node_id, 1:4]
        bbox_max = octree_metadata[node_id, 4:7]
        first_child = int(octree_metadata[node_id, 7])

        # If leaf, return
        if is_leaf == 1:
            return node_id, depth

        # Compute octant (same as GPU search)
        bbox_mid = (bbox_min + bbox_max) / 2.0
        octant = (
            int(position[0] >= bbox_mid[0]) +
            (int(position[1] >= bbox_mid[1]) << 1) +
            (int(position[2] >= bbox_mid[2]) << 2)
        )

        # Move to child
        node_id = first_child + octant
        depth += 1

        # Safety check
        if depth > 20 or node_id >= len(octree_metadata):
            print(f"    WARNING: Navigation exceeded depth 20 or invalid node_id")
            return node_id, depth


def find_element_assigned_leaves(element_id: int, octree_elements: np.ndarray) -> list:
    """
    Find ALL leaves an element was assigned to during construction.

    With bbox-based assignment, elements can be in multiple leaves.

    Returns:
        List of (leaf_node_id, element_index_in_leaf) tuples
    """
    n_leaves = len(octree_elements)
    max_elements_per_leaf = octree_elements.shape[1]

    assigned_leaves = []

    for leaf_id in range(n_leaves):
        for idx in range(max_elements_per_leaf):
            if octree_elements[leaf_id, idx] == element_id:
                assigned_leaves.append((leaf_id, idx))
                break  # Element appears at most once per leaf

    return assigned_leaves


def check_octree_consistency(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    element_centroids: np.ndarray,
    octree_metadata: np.ndarray,
    octree_elements: np.ndarray,
    n_test_elements: int = 1000
):
    """
    Check if elements are assigned to leaves that their centroids navigate to.

    This is the core diagnostic that identifies the octree bug.
    """
    print("\n" + "="*80)
    print("OCTREE CONSISTENCY CHECK")
    print("="*80)
    print(f"Testing {n_test_elements:,} random elements...")
    print()

    # Sample random elements
    n_elements = len(connectivity)
    test_element_ids = np.random.choice(n_elements, size=min(n_test_elements, n_elements), replace=False)

    match_count = 0
    mismatch_count = 0
    not_found_count = 0

    mismatch_examples = []

    for elem_id in test_element_ids:
        # Get element centroid
        centroid = element_centroids[elem_id]

        # Find ALL leaves this element was assigned to
        assigned_leaves = find_element_assigned_leaves(elem_id, octree_elements)

        if len(assigned_leaves) == 0:
            not_found_count += 1
            continue

        # Navigate centroid to leaf (using same logic as search)
        navigated_leaf, depth = navigate_to_leaf_cpu(centroid, octree_metadata)

        # Check if navigated leaf is ONE OF the assigned leaves
        assigned_leaf_ids = [leaf_id for leaf_id, _ in assigned_leaves]

        if navigated_leaf in assigned_leaf_ids:
            match_count += 1
        else:
            mismatch_count += 1

            # Store example for detailed analysis
            if len(mismatch_examples) < 5:
                # Use first assigned leaf for comparison
                primary_assigned_leaf = assigned_leaves[0][0]

                assigned_bbox_min = octree_metadata[primary_assigned_leaf, 1:4]
                assigned_bbox_max = octree_metadata[primary_assigned_leaf, 4:7]
                assigned_depth = int(octree_metadata[primary_assigned_leaf, 8])

                navigated_bbox_min = octree_metadata[navigated_leaf, 1:4]
                navigated_bbox_max = octree_metadata[navigated_leaf, 4:7]
                navigated_depth = int(octree_metadata[navigated_leaf, 8])

                # Check if centroid is inside assigned bbox
                inside_assigned = (
                    (centroid[0] >= assigned_bbox_min[0]) and (centroid[0] <= assigned_bbox_max[0]) and
                    (centroid[1] >= assigned_bbox_min[1]) and (centroid[1] <= assigned_bbox_max[1]) and
                    (centroid[2] >= assigned_bbox_min[2]) and (centroid[2] <= assigned_bbox_max[2])
                )

                # Check if centroid is inside navigated bbox
                inside_navigated = (
                    (centroid[0] >= navigated_bbox_min[0]) and (centroid[0] <= navigated_bbox_max[0]) and
                    (centroid[1] >= navigated_bbox_min[1]) and (centroid[1] <= navigated_bbox_max[1]) and
                    (centroid[2] >= navigated_bbox_min[2]) and (centroid[2] <= navigated_bbox_max[2])
                )

                mismatch_examples.append({
                    'elem_id': elem_id,
                    'centroid': centroid,
                    'assigned_leaves': assigned_leaf_ids,  # List of all assigned leaf IDs
                    'primary_assigned_leaf': primary_assigned_leaf,
                    'assigned_depth': assigned_depth,
                    'assigned_bbox_min': assigned_bbox_min,
                    'assigned_bbox_max': assigned_bbox_max,
                    'inside_assigned': inside_assigned,
                    'navigated_leaf': navigated_leaf,
                    'navigated_depth': navigated_depth,
                    'navigated_bbox_min': navigated_bbox_min,
                    'navigated_bbox_max': navigated_bbox_max,
                    'inside_navigated': inside_navigated
                })

    # Print summary
    total = match_count + mismatch_count
    print(f"Results:")
    print(f"  Assigned leaf == Navigated leaf: {match_count}/{total} ({100*match_count/total:.2f}%)")
    print(f"  Assigned leaf != Navigated leaf: {mismatch_count}/{total} ({100*mismatch_count/total:.2f}%)")
    if not_found_count > 0:
        print(f"  Elements not found in octree: {not_found_count}")
    print()

    # Print detailed examples of mismatches
    if mismatch_examples:
        print("-"*80)
        print("MISMATCH EXAMPLES (First 5)")
        print("-"*80)

        for i, example in enumerate(mismatch_examples, 1):
            print(f"\nExample {i}:")
            print(f"  Element ID: {example['elem_id']}")
            print(f"  Centroid: [{example['centroid'][0]:.8f}, {example['centroid'][1]:.8f}, {example['centroid'][2]:.8f}]")
            print()
            print(f"  DURING CONSTRUCTION:")
            print(f"    Assigned to {len(example['assigned_leaves'])} leaves: {example['assigned_leaves'][:5]}{'...' if len(example['assigned_leaves']) > 5 else ''}")
            print(f"    Primary leaf: {example['primary_assigned_leaf']} (depth {example['assigned_depth']})")
            print(f"    Leaf bbox: min=[{example['assigned_bbox_min'][0]:.8f}, {example['assigned_bbox_min'][1]:.8f}, {example['assigned_bbox_min'][2]:.8f}]")
            print(f"               max=[{example['assigned_bbox_max'][0]:.8f}, {example['assigned_bbox_max'][1]:.8f}, {example['assigned_bbox_max'][2]:.8f}]")
            print(f"    Centroid inside assigned bbox: {example['inside_assigned']}")
            print()
            print(f"  DURING SEARCH:")
            print(f"    Navigated to leaf: {example['navigated_leaf']} (depth {example['navigated_depth']})")
            print(f"    Leaf bbox: min=[{example['navigated_bbox_min'][0]:.8f}, {example['navigated_bbox_min'][1]:.8f}, {example['navigated_bbox_min'][2]:.8f}]")
            print(f"               max=[{example['navigated_bbox_max'][0]:.8f}, {example['navigated_bbox_max'][1]:.8f}, {example['navigated_bbox_max'][2]:.8f}]")
            print(f"    Centroid inside navigated bbox: {example['inside_navigated']}")
            print()
            if example['navigated_leaf'] in example['assigned_leaves']:
                print(f"  CONCLUSION: ✓ Navigated leaf {example['navigated_leaf']} IS in assigned leaves (should have matched)")
            else:
                print(f"  CONCLUSION: ✗ Navigated leaf {example['navigated_leaf']} NOT in assigned leaves {example['assigned_leaves']}")
            print(f"              This means particles inside this element will search the WRONG leaf!")

    return match_count, mismatch_count, not_found_count


def test_particle_search_accuracy(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    element_centroids: np.ndarray,
    mesh_gpu,
    octree_metadata_gpu: jax.Array,
    octree_elements_gpu: jax.Array,
    n_test_particles: int = 1000
):
    """
    Test search accuracy by generating particles at element centroids and searching.
    """
    print("\n" + "="*80)
    print("PARTICLE SEARCH ACCURACY TEST")
    print("="*80)
    print(f"Testing {n_test_particles:,} particles at element centroids...")
    print()

    # Sample random elements
    n_elements = len(connectivity)
    test_element_ids = np.random.choice(n_elements, size=min(n_test_particles, n_elements), replace=False)

    # Generate particles at centroids with tiny perturbations
    perturbation_scale = 1e-6  # Very small perturbation
    test_positions = np.zeros((len(test_element_ids), 3), dtype=np.float32)

    for i, elem_id in enumerate(test_element_ids):
        centroid = element_centroids[elem_id]
        # Add tiny random perturbation to ensure particle is strictly inside
        perturbation = np.random.uniform(-perturbation_scale, perturbation_scale, size=3)
        test_positions[i] = centroid + perturbation

    # Upload particles to GPU
    test_positions_gpu = jax.device_put(test_positions)

    # JIT compile search
    print("JIT compiling octree search...")
    _ = search_level2_octree_scan(
        test_positions_gpu[0],
        octree_metadata_gpu,
        octree_elements_gpu,
        mesh_gpu.node_positions,
        mesh_gpu.connectivity
    )
    print("✓ Search compiled")
    print()

    # Search particles
    print("Searching particles...")
    t0 = time.time()

    found_elements = np.zeros(len(test_element_ids), dtype=np.int32)
    for i in range(len(test_element_ids)):
        elem_id = search_level2_octree_scan(
            test_positions_gpu[i],
            octree_metadata_gpu,
            octree_elements_gpu,
            mesh_gpu.node_positions,
            mesh_gpu.connectivity
        )
        found_elements[i] = int(elem_id)

    t_search = time.time() - t0

    # Analyze results
    correct = 0
    wrong = 0
    not_found = 0

    for i, true_elem in enumerate(test_element_ids):
        found_elem = found_elements[i]

        if found_elem == -1:
            not_found += 1
        elif found_elem == true_elem:
            correct += 1
        else:
            wrong += 1

    total = len(test_element_ids)
    found_rate = 100 * (correct + wrong) / total
    accuracy = 100 * correct / total if total > 0 else 0

    print(f"Results:")
    print(f"  Particles tested: {total:,}")
    print(f"  Found (any element): {correct + wrong:,} ({found_rate:.2f}%)")
    print(f"  Not found: {not_found:,} ({100*not_found/total:.2f}%)")
    print(f"  Correct element: {correct:,} ({accuracy:.2f}%)")
    print(f"  Wrong element: {wrong:,} ({100*wrong/total:.2f}%)")
    print(f"  Search time: {t_search:.3f} s ({total/t_search:.1f} particles/s)")
    print()

    return correct, wrong, not_found


def main():
    """Main diagnostic test."""
    print("="*80)
    print("OCTREE DIAGNOSTIC TEST")
    print("="*80)
    print()

    # Configuration
    mesh_path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_120.pvtu")
    level_threshold = 1.1
    octree_max_depth = 15
    octree_max_leaf_size = 50
    n_test_elements = 2000
    n_test_particles = 1000

    print(f"Configuration:")
    print(f"  Mesh: {mesh_path}")
    print(f"  Level threshold: {level_threshold}")
    print(f"  Octree max depth: {octree_max_depth}")
    print(f"  Octree max leaf size: {octree_max_leaf_size}")
    print(f"  Test elements: {n_test_elements:,}")
    print(f"  Test particles: {n_test_particles:,}")
    print()

    # ========================================================================
    # LOAD MESH
    # ========================================================================
    print("="*80)
    print("LOADING MESH")
    print("="*80)
    print()

    if not mesh_path.exists():
        print(f"✗ Mesh not found: {mesh_path}")
        return

    t0 = time.time()
    node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
        mesh_path,
        field_name='Displacement'
    )
    print(f"✓ Loaded mesh ({time.time()-t0:.2f} s):")
    print(f"  Nodes: {len(node_positions):,}")
    print(f"  Elements: {len(connectivity):,}")
    print()

    # Compute element centroids
    print("Computing element centroids...")
    t0 = time.time()
    element_centroids = compute_element_centroids(node_positions, connectivity)
    print(f"✓ Computed centroids ({time.time()-t0:.2f} s)")
    print()

    # Compute element LEVEL field
    print("Computing element LEVEL field...")
    t0 = time.time()
    element_levels = compute_element_level_field(node_positions, connectivity, mesh_path)
    print(f"✓ Computed element levels ({time.time()-t0:.2f} s)")
    print()

    # ========================================================================
    # BUILD OCTREE
    # ========================================================================
    print("="*80)
    print("BUILDING OCTREE")
    print("="*80)
    print()

    print(f"Building octree with LEVEL < {level_threshold}...")
    t0 = time.time()

    # Prepare element IDs for octree
    element_ids = np.arange(len(connectivity), dtype=np.int32)

    # Build octree (with bbox-based element assignment)
    octree_root, octree_stats = build_octree_for_level(
        element_centroids=element_centroids,
        element_ids=element_ids,
        node_positions=node_positions,
        connectivity=connectivity,
        level_field=element_levels,
        level_threshold=level_threshold,
        max_depth=octree_max_depth,
        max_leaf_size=octree_max_leaf_size,
        use_levelset=True
    )
    print(f"✓ Octree built ({time.time()-t0:.2f} s)")
    print()

    # Flatten octree
    print("Flattening octree to arrays...")
    t0 = time.time()
    octree_metadata, octree_elements = flatten_octree_to_arrays(octree_root, octree_max_leaf_size)
    print(f"✓ Octree flattened ({time.time()-t0:.2f} s)")
    print(f"  Octree nodes: {len(octree_metadata):,}")
    print(f"  Max elements per leaf: {octree_elements.shape[1]}")
    print(f"  Filtered elements: {np.sum(octree_elements >= 0):,}")
    print()

    # ========================================================================
    # DIAGNOSTIC 1: OCTREE CONSISTENCY CHECK
    # ========================================================================
    match_count, mismatch_count, not_found = check_octree_consistency(
        node_positions,
        connectivity,
        element_centroids,
        octree_metadata,
        octree_elements,
        n_test_elements=n_test_elements
    )

    # ========================================================================
    # BUILD ELEMENT NEIGHBORS (for mesh upload)
    # ========================================================================
    print("="*80)
    print("BUILDING ELEMENT NEIGHBORS")
    print("="*80)
    print()

    print("Building element neighbors...")
    t0 = time.time()
    element_neighbors = build_element_neighbors_array(connectivity, verbose=False)
    print(f"✓ Element neighbors built ({time.time()-t0:.2f} s)")
    print()

    # ========================================================================
    # UPLOAD TO GPU
    # ========================================================================
    print("="*80)
    print("UPLOADING TO GPU")
    print("="*80)
    print()

    print("Uploading mesh to GPU...")
    t0 = time.time()
    mesh_gpu = upload_mesh_to_gpu(node_positions, connectivity, element_neighbors)
    print(f"✓ Mesh uploaded ({time.time()-t0:.2f} s)")
    print()

    print("Uploading octree to GPU...")
    t0 = time.time()
    octree_metadata_gpu = jax.device_put(octree_metadata)
    octree_elements_gpu = jax.device_put(octree_elements)
    print(f"✓ Octree uploaded ({time.time()-t0:.2f} s)")
    print()

    # ========================================================================
    # DIAGNOSTIC 2: PARTICLE SEARCH ACCURACY
    # ========================================================================
    correct, wrong, not_found_particles = test_particle_search_accuracy(
        node_positions,
        connectivity,
        element_centroids,
        mesh_gpu,
        octree_metadata_gpu,
        octree_elements_gpu,
        n_test_particles=n_test_particles
    )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print()

    total_tested = match_count + mismatch_count
    consistency_rate = 100 * match_count / total_tested if total_tested > 0 else 0

    total_particles = correct + wrong + not_found_particles
    accuracy = 100 * correct / total_particles if total_particles > 0 else 0

    print(f"Octree Consistency Check:")
    print(f"  Elements where assigned_leaf == navigated_leaf: {match_count}/{total_tested} ({consistency_rate:.2f}%)")
    print(f"  Elements where assigned_leaf != navigated_leaf: {mismatch_count}/{total_tested} ({100-consistency_rate:.2f}%)")
    print()

    print(f"Particle Search Accuracy:")
    print(f"  Correct assignments: {correct}/{total_particles} ({accuracy:.2f}%)")
    print(f"  Wrong assignments: {wrong}/{total_particles} ({100*wrong/total_particles:.2f}%)")
    print(f"  Not found: {not_found_particles}/{total_particles} ({100*not_found_particles/total_particles:.2f}%)")
    print()

    if mismatch_count > 0:
        print("✗ OCTREE BUG CONFIRMED:")
        print(f"  {100-consistency_rate:.1f}% of elements are assigned to leaves that their centroids don't navigate to!")
        print("  This directly causes the low search accuracy.")
        print()
        print("RECOMMENDED FIX:")
        print("  Update octree_builder.py to use the same octant computation logic as octree_search_gpu.py")
        print("  See: jaxtrace/gpu/search/octree_builder.py")
    else:
        print("✓ Octree consistency check passed")
        if accuracy < 95:
            print("  But search accuracy is still low - investigate other issues")
    print()

    print("="*80)
    print("DIAGNOSTIC TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
