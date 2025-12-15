#!/usr/bin/env python3
"""
Systematic Octree Search Failure Diagnosis

Investigates why octree search has 99.47% found rate but only 5.66% correct matches.

Hypotheses to test:
1. LEVEL-based filtering excludes valid elements near interfaces
2. Octree bounding box approximation fails for small refined elements
3. Spatial indexing errors in octree construction
4. Parallel search race conditions (unlikely but check)

Approach:
- Build alternative octree without LEVEL filtering (pure geometry)
- Compare LEVEL-filtered vs geometry-only octrees
- Analyze failure patterns spatially
- Identify root cause systematically
"""

import numpy as np
import time
import jax
import jax.numpy as jnp
from pathlib import Path

# Import mesh and octree tools
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.search.octree_builder import build_octree_for_level, flatten_octree_to_arrays
from jaxtrace.gpu.search.octree_search_gpu import search_level2_octree_scan
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.search.level0_cached import point_in_tet_jax

jax.config.update("jax_enable_x64", True)


def compute_element_centroids(mesh):
    """Compute centroids for all elements."""
    n_elements = len(mesh.connectivity)
    centroids = np.zeros((n_elements, 3), dtype=np.float32)

    for i in range(n_elements):
        node_ids = mesh.connectivity[i]
        centroids[i] = mesh.node_positions[node_ids].mean(axis=0)

    return centroids


def compute_element_level_field(mesh):
    """Compute element-wise level set values (average of nodes)."""
    n_elements = len(mesh.connectivity)
    elem_levels = np.zeros(n_elements, dtype=np.float32)

    # Get LEVEL field from mesh
    if hasattr(mesh, 'point_data') and 'LEVEL' in mesh.point_data:
        level_field = mesh.point_data['LEVEL']
    else:
        print("⚠️  No LEVEL field found, using zeros")
        return elem_levels

    for i in range(n_elements):
        node_ids = mesh.connectivity[i]
        elem_levels[i] = level_field[node_ids].mean()

    return elem_levels


def build_geometry_only_octree(mesh, element_centroids, max_depth=15, max_leaf_size=50):
    """
    Build octree using ONLY geometry (all elements), no LEVEL filtering.

    This is the alternative construction method to test hypothesis 1.
    """
    print("\n" + "="*80)
    print("BUILDING GEOMETRY-ONLY OCTREE (NO LEVEL FILTERING)")
    print("="*80)

    n_elements = len(mesh.connectivity)
    all_element_ids = np.arange(n_elements, dtype=np.int32)

    print(f"  Total elements: {n_elements:,}")
    print(f"  Max depth: {max_depth}")
    print(f"  Max leaf size: {max_leaf_size}")

    t0 = time.time()

    # Compute bounding box
    bbox_min = mesh.node_positions.min(axis=0).astype(np.float32)
    bbox_max = mesh.node_positions.max(axis=0).astype(np.float32)

    # Build octree recursively
    class OctreeNode:
        def __init__(self, bbox_min, bbox_max, element_ids, depth):
            self.bbox_min = bbox_min
            self.bbox_max = bbox_max
            self.element_ids = element_ids
            self.depth = depth
            self.children = None
            self.is_leaf = True

    def build_node(bbox_min, bbox_max, elem_ids, depth):
        """Recursively build octree node."""
        n_elems = len(elem_ids)

        # Leaf condition
        if depth >= max_depth or n_elems <= max_leaf_size:
            return OctreeNode(bbox_min, bbox_max, elem_ids, depth)

        # Split into 8 children
        mid = (bbox_min + bbox_max) / 2
        node = OctreeNode(bbox_min, bbox_max, elem_ids, depth)
        node.is_leaf = False
        node.children = []

        for i in range(8):
            # Child bounding box
            child_min = bbox_min.copy()
            child_max = bbox_max.copy()

            if i & 1:
                child_min[0] = mid[0]
            else:
                child_max[0] = mid[0]

            if i & 2:
                child_min[1] = mid[1]
            else:
                child_max[1] = mid[1]

            if i & 4:
                child_min[2] = mid[2]
            else:
                child_max[2] = mid[2]

            # Find elements in child
            child_centroids = element_centroids[elem_ids]
            in_child = np.all((child_centroids >= child_min) & (child_centroids < child_max), axis=1)
            child_elem_ids = elem_ids[in_child]

            if len(child_elem_ids) > 0:
                child_node = build_node(child_min, child_max, child_elem_ids, depth + 1)
                node.children.append(child_node)

        return node

    root = build_node(bbox_min, bbox_max, all_element_ids, 0)

    # Flatten to arrays (compatible with existing search)
    def count_nodes(node):
        if node.is_leaf:
            return 1
        return 1 + sum(count_nodes(child) for child in node.children)

    total_nodes = count_nodes(root)

    # Allocate arrays (same format as current octree)
    node_bbox_min = np.zeros((total_nodes, 3), dtype=np.float32)
    node_bbox_max = np.zeros((total_nodes, 3), dtype=np.float32)
    node_children = np.full((total_nodes, 8), -1, dtype=np.int32)
    node_elem_start = np.zeros(total_nodes, dtype=np.int32)
    node_elem_count = np.zeros(total_nodes, dtype=np.int32)
    node_is_leaf = np.zeros(total_nodes, dtype=bool)

    # Collect all leaf elements
    all_leaf_elements = []

    def flatten_recursive(node, node_id):
        node_bbox_min[node_id] = node.bbox_min
        node_bbox_max[node_id] = node.bbox_max
        node_is_leaf[node_id] = node.is_leaf

        if node.is_leaf:
            node_elem_start[node_id] = len(all_leaf_elements)
            node_elem_count[node_id] = len(node.element_ids)
            all_leaf_elements.extend(node.element_ids)
            return node_id + 1
        else:
            # Reserve space for children
            first_child_id = node_id + 1
            next_id = first_child_id + len(node.children)

            # Flatten children
            for i, child in enumerate(node.children):
                node_children[node_id, i] = first_child_id + i
                next_id = flatten_recursive(child, first_child_id + i)

            return next_id

    flatten_recursive(root, 0)

    leaf_elements = np.array(all_leaf_elements, dtype=np.int32)

    t1 = time.time()

    print(f"✓ Built geometry-only octree ({t1-t0:.2f} s)")
    print(f"  Total nodes: {total_nodes:,}")
    print(f"  Leaf elements: {len(leaf_elements):,}")
    print(f"  Coverage: {100*len(leaf_elements)/n_elements:.1f}% of mesh")

    # Return in same format as LEVEL-filtered octree
    class GeometryOctree:
        def __init__(self):
            self.node_bbox_min = node_bbox_min
            self.node_bbox_max = node_bbox_max
            self.node_children = node_children
            self.node_elem_start = node_elem_start
            self.node_elem_count = node_elem_count
            self.node_is_leaf = node_is_leaf
            self.leaf_elements = leaf_elements
            self.root_bbox_min = bbox_min
            self.root_bbox_max = bbox_max

    return GeometryOctree()


def find_true_element_bruteforce(position, mesh):
    """
    Brute-force ground truth: check ALL elements.

    This is the definitive correct answer.
    """
    pos_jax = jnp.array(position, dtype=jnp.float32)

    for elem_id in range(len(mesh.connectivity)):
        node_ids = mesh.connectivity[elem_id]
        tet_nodes = mesh.node_positions[node_ids]
        tet_nodes_jax = jnp.array(tet_nodes, dtype=jnp.float32)

        if point_in_tet_jax(pos_jax, tet_nodes_jax):
            return elem_id

    return -1


def compare_octree_implementations(mesh_path, n_test_particles=1000, level_threshold=1.1):
    """
    Compare LEVEL-filtered octree vs geometry-only octree.

    This is the main diagnostic function.
    """
    print("\n" + "="*80)
    print("OCTREE SEARCH FAILURE DIAGNOSIS")
    print("="*80)
    print(f"Mesh: {mesh_path}")
    print(f"Test particles: {n_test_particles:,}")
    print(f"Level threshold: {level_threshold}")

    # Load mesh
    print("\nLoading mesh...")
    t0 = time.time()
    node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
        mesh_path,
        field_name='Displacement'
    )
    print(f"✓ Loaded: {len(node_positions):,} nodes, {len(connectivity):,} elements ({time.time()-t0:.2f} s)")

    # Compute element data
    print("\nComputing element data...")
    element_centroids = compute_element_centroids(mesh)
    element_levels = compute_element_level_field(mesh)
    print(f"✓ Element centroids and levels computed")
    print(f"  Level range: [{element_levels.min():.2f}, {element_levels.max():.2f}]")

    # Build LEVEL-filtered octree (current method)
    print("\n" + "-"*80)
    print("METHOD 1: LEVEL-FILTERED OCTREE (CURRENT)")
    print("-"*80)
    t0 = time.time()
    octree_level = build_octree_for_level(
        mesh.node_positions,
        mesh.connectivity,
        element_levels,
        element_centroids,
        level_threshold=level_threshold,
        max_depth=15,
        max_leaf_size=50
    )
    octree_level_arrays = flatten_octree_to_arrays(octree_level)
    t1 = time.time()

    n_filtered = len(octree_level_arrays.leaf_elements)
    print(f"✓ LEVEL-filtered octree built ({t1-t0:.2f} s)")
    print(f"  Filtered elements: {n_filtered:,} / {len(mesh.connectivity):,} ({100*n_filtered/len(mesh.connectivity):.1f}%)")

    # Build geometry-only octree (alternative method)
    print("\n" + "-"*80)
    print("METHOD 2: GEOMETRY-ONLY OCTREE (ALTERNATIVE)")
    print("-"*80)
    octree_geom = build_geometry_only_octree(mesh, element_centroids, max_depth=15, max_leaf_size=50)

    # Upload to GPU
    print("\nUploading to GPU...")
    mesh_gpu = upload_mesh_to_gpu(mesh.node_positions, mesh.connectivity, mesh.element_neighbors)

    # Upload octrees
    octree_level_gpu = {
        'node_bbox_min': jnp.array(octree_level_arrays.node_bbox_min),
        'node_bbox_max': jnp.array(octree_level_arrays.node_bbox_max),
        'node_children': jnp.array(octree_level_arrays.node_children),
        'node_elem_start': jnp.array(octree_level_arrays.node_elem_start),
        'node_elem_count': jnp.array(octree_level_arrays.node_elem_count),
        'node_is_leaf': jnp.array(octree_level_arrays.node_is_leaf),
        'leaf_elements': jnp.array(octree_level_arrays.leaf_elements),
        'root_bbox_min': jnp.array(octree_level_arrays.root_bbox_min),
        'root_bbox_max': jnp.array(octree_level_arrays.root_bbox_max),
    }

    octree_geom_gpu = {
        'node_bbox_min': jnp.array(octree_geom.node_bbox_min),
        'node_bbox_max': jnp.array(octree_geom.node_bbox_max),
        'node_children': jnp.array(octree_geom.node_children),
        'node_elem_start': jnp.array(octree_geom.node_elem_start),
        'node_elem_count': jnp.array(octree_geom.node_elem_count),
        'node_is_leaf': jnp.array(octree_geom.node_is_leaf),
        'leaf_elements': jnp.array(octree_geom.leaf_elements),
        'root_bbox_min': jnp.array(octree_geom.root_bbox_min),
        'root_bbox_max': jnp.array(octree_geom.root_bbox_max),
    }

    print("✓ Data uploaded to GPU")

    # Generate test particles at element centroids with small perturbations
    print(f"\nGenerating {n_test_particles:,} test particles...")
    np.random.seed(42)

    # Sample from elements in LEVEL-filtered set (where octree should work)
    filtered_elem_ids = octree_level_arrays.leaf_elements
    test_elem_ids = np.random.choice(filtered_elem_ids, size=n_test_particles, replace=True)

    # Get centroids and add small perturbations
    test_positions = element_centroids[test_elem_ids].copy()

    # Compute minimum element size for perturbation scale
    min_elem_size = np.inf
    for elem_id in test_elem_ids[:100]:  # Sample for efficiency
        node_ids = mesh.connectivity[elem_id]
        nodes = mesh.node_positions[node_ids]
        edges = [
            np.linalg.norm(nodes[1] - nodes[0]),
            np.linalg.norm(nodes[2] - nodes[0]),
            np.linalg.norm(nodes[3] - nodes[0]),
        ]
        min_elem_size = min(min_elem_size, min(edges))

    perturbation_scale = min_elem_size * 0.01  # 1% of minimum edge
    test_positions += np.random.normal(0, perturbation_scale, test_positions.shape)

    print(f"✓ Test particles generated")
    print(f"  Perturbation scale: {perturbation_scale:.6f}")

    # Test 1: LEVEL-filtered octree search
    print("\n" + "-"*80)
    print("TEST 1: LEVEL-FILTERED OCTREE SEARCH")
    print("-"*80)

    test_positions_gpu = jnp.array(test_positions, dtype=jnp.float32)

    print("Warming up JIT...")
    _ = search_level2_octree_scan(
        test_positions_gpu[:10],
        octree_level_gpu['node_bbox_min'],
        octree_level_gpu['node_bbox_max'],
        octree_level_gpu['node_children'],
        octree_level_gpu['node_elem_start'],
        octree_level_gpu['node_elem_count'],
        octree_level_gpu['node_is_leaf'],
        octree_level_gpu['leaf_elements'],
        octree_level_gpu['root_bbox_min'],
        octree_level_gpu['root_bbox_max'],
        mesh_gpu.node_positions,
        mesh_gpu.connectivity
    )

    print("Running search...")
    t0 = time.time()
    found_elem_ids_level = search_level2_octree_scan(
        test_positions_gpu,
        octree_level_gpu['node_bbox_min'],
        octree_level_gpu['node_bbox_max'],
        octree_level_gpu['node_children'],
        octree_level_gpu['node_elem_start'],
        octree_level_gpu['node_elem_count'],
        octree_level_gpu['node_is_leaf'],
        octree_level_gpu['leaf_elements'],
        octree_level_gpu['root_bbox_min'],
        octree_level_gpu['root_bbox_max'],
        mesh_gpu.node_positions,
        mesh_gpu.connectivity
    )
    t1 = time.time()

    found_elem_ids_level = np.array(found_elem_ids_level, dtype=np.int32)

    found_count_level = np.sum(found_elem_ids_level >= 0)
    correct_count_level = np.sum(found_elem_ids_level == test_elem_ids)

    print(f"✓ LEVEL-filtered search complete ({t1-t0:.4f} s)")
    print(f"  Found: {found_count_level}/{n_test_particles} ({100*found_count_level/n_test_particles:.2f}%)")
    print(f"  Correct: {correct_count_level}/{n_test_particles} ({100*correct_count_level/n_test_particles:.2f}%)")

    # Test 2: Geometry-only octree search
    print("\n" + "-"*80)
    print("TEST 2: GEOMETRY-ONLY OCTREE SEARCH")
    print("-"*80)

    print("Running search...")
    t0 = time.time()
    found_elem_ids_geom = search_level2_octree_scan(
        test_positions_gpu,
        octree_geom_gpu['node_bbox_min'],
        octree_geom_gpu['node_bbox_max'],
        octree_geom_gpu['node_children'],
        octree_geom_gpu['node_elem_start'],
        octree_geom_gpu['node_elem_count'],
        octree_geom_gpu['node_is_leaf'],
        octree_geom_gpu['leaf_elements'],
        octree_geom_gpu['root_bbox_min'],
        octree_geom_gpu['root_bbox_max'],
        mesh_gpu.node_positions,
        mesh_gpu.connectivity
    )
    t1 = time.time()

    found_elem_ids_geom = np.array(found_elem_ids_geom, dtype=np.int32)

    found_count_geom = np.sum(found_elem_ids_geom >= 0)
    correct_count_geom = np.sum(found_elem_ids_geom == test_elem_ids)

    print(f"✓ Geometry-only search complete ({t1-t0:.4f} s)")
    print(f"  Found: {found_count_geom}/{n_test_particles} ({100*found_count_geom/n_test_particles:.2f}%)")
    print(f"  Correct: {correct_count_geom}/{n_test_particles} ({100*correct_count_geom/n_test_particles:.2f}%)")

    # Detailed failure analysis
    print("\n" + "="*80)
    print("FAILURE ANALYSIS")
    print("="*80)

    # Categorize failures
    level_wrong = (found_elem_ids_level >= 0) & (found_elem_ids_level != test_elem_ids)
    level_notfound = found_elem_ids_level < 0

    geom_wrong = (found_elem_ids_geom >= 0) & (found_elem_ids_geom != test_elem_ids)
    geom_notfound = found_elem_ids_geom < 0

    print("\nLEVEL-FILTERED OCTREE:")
    print(f"  Correct: {correct_count_level} ({100*correct_count_level/n_test_particles:.2f}%)")
    print(f"  Wrong element: {np.sum(level_wrong)} ({100*np.sum(level_wrong)/n_test_particles:.2f}%)")
    print(f"  Not found: {np.sum(level_notfound)} ({100*np.sum(level_notfound)/n_test_particles:.2f}%)")

    print("\nGEOMETRY-ONLY OCTREE:")
    print(f"  Correct: {correct_count_geom} ({100*correct_count_geom/n_test_particles:.2f}%)")
    print(f"  Wrong element: {np.sum(geom_wrong)} ({100*np.sum(geom_wrong)/n_test_particles:.2f}%)")
    print(f"  Not found: {np.sum(geom_notfound)} ({100*np.sum(geom_notfound)/n_test_particles:.2f}%)")

    print("\nCOMPARISON:")
    improvement = correct_count_geom - correct_count_level
    print(f"  Improvement: {improvement} particles ({100*improvement/n_test_particles:+.2f}%)")

    if improvement > 0:
        print(f"  ✓ Geometry-only octree is MORE accurate")
    elif improvement < 0:
        print(f"  ✗ Geometry-only octree is LESS accurate")
    else:
        print(f"  = Both octrees have same accuracy")

    # Sample detailed analysis of failures
    print("\n" + "-"*80)
    print("SAMPLE FAILURE CASES (first 10)")
    print("-"*80)

    level_failures = np.where(level_wrong | level_notfound)[0][:10]

    for idx in level_failures:
        pos = test_positions[idx]
        true_elem = test_elem_ids[idx]
        level_found = found_elem_ids_level[idx]
        geom_found = found_elem_ids_geom[idx]

        print(f"\nParticle {idx}:")
        print(f"  Position: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
        print(f"  True element: {true_elem}")
        print(f"  LEVEL-filtered found: {level_found} {'✓' if level_found == true_elem else '✗'}")
        print(f"  Geometry-only found: {geom_found} {'✓' if geom_found == true_elem else '✗'}")

        # Check if true element is in LEVEL-filtered set
        if true_elem in octree_level_arrays.leaf_elements:
            print(f"  True element IN LEVEL-filtered octree ✓")
        else:
            print(f"  True element NOT in LEVEL-filtered octree ✗ (LEVEL filtering excluded it!)")

    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)

    # Save results
    results = {
        'n_test_particles': n_test_particles,
        'level_threshold': level_threshold,
        'level_filtered_elements': n_filtered,
        'geometry_all_elements': len(mesh.connectivity),
        'level_correct': correct_count_level,
        'level_wrong': np.sum(level_wrong),
        'level_notfound': np.sum(level_notfound),
        'geom_correct': correct_count_geom,
        'geom_wrong': np.sum(geom_wrong),
        'geom_notfound': np.sum(geom_notfound),
        'improvement': improvement,
    }

    return results


if __name__ == "__main__":
    mesh_path = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_120.pvtu"

    results = compare_octree_implementations(
        mesh_path=mesh_path,
        n_test_particles=1000,
        level_threshold=1.1
    )

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nHypothesis 1: LEVEL filtering excludes valid elements")
    print(f"  Elements excluded: {results['geometry_all_elements'] - results['level_filtered_elements']:,}")
    print(f"  Percentage excluded: {100*(results['geometry_all_elements'] - results['level_filtered_elements'])/results['geometry_all_elements']:.1f}%")

    print(f"\nAccuracy comparison:")
    print(f"  LEVEL-filtered: {100*results['level_correct']/results['n_test_particles']:.2f}% correct")
    print(f"  Geometry-only: {100*results['geom_correct']/results['n_test_particles']:.2f}% correct")
    print(f"  Improvement: {100*results['improvement']/results['n_test_particles']:+.2f}%")

    if results['improvement'] > results['n_test_particles'] * 0.05:
        print(f"\n✓ CONFIRMED: LEVEL filtering is the primary cause of failures!")
        print(f"  Recommendation: Use geometry-only octree for initial assignment")
    else:
        print(f"\n✗ LEVEL filtering is NOT the primary cause")
        print(f"  Failures likely due to octree spatial indexing errors")
        print(f"  Need further investigation of octree search algorithm")
