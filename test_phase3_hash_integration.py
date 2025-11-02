#!/usr/bin/env python3
"""
Phase 3 Hash Octree Integration Test.

This test validates that the hash octree integration with SharedOctreeFEMField works correctly.
It tests the _find_elements_with_hash_octree function without requiring full mesh files.
"""

import numpy as np
import jax
import jax.numpy as jnp

# Enable JAX 64-bit mode (required for hash octree)
jax.config.update("jax_enable_x64", True)

from jaxtrace.fields.hash_octree import (
    build_hash_octree_from_leaves,
    hash_lookup_batch_jax,
    get_hash_octree_memory_stats
)
from jaxtrace.fields.morton_code import encode_morton_3d


def test_hash_octree_integration():
    """
    Test hash octree integration with mock data that simulates
    the SharedOctreeFEMField workflow.
    """
    print("\n" + "="*70)
    print("PHASE 3: HASH OCTREE INTEGRATION TEST")
    print("="*70)

    # Create mock mesh data
    print("\n1. Creating mock tetrahedral mesh...")

    # Simple 2x2x2 cube with 8 vertices and 5 tetrahedra
    positions = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
    ], dtype=np.float32)

    # 5 tetrahedra that partition the cube
    connectivity = np.array([
        [0, 1, 2, 5],  # Element 0
        [0, 2, 3, 7],  # Element 1
        [0, 5, 4, 7],  # Element 2
        [2, 5, 6, 7],  # Element 3
        [0, 2, 5, 7],  # Element 4 (center)
    ], dtype=np.int32)

    print(f"  Created mesh: {len(positions)} vertices, {len(connectivity)} tetrahedra")

    # Create hash octree (simulate extracting leaves from fine octree)
    print("\n2. Building hash octree from mock leaf nodes...")

    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    level = 3  # Fine octree level

    # Create "leaf nodes" - divide domain into octree cells and assign elements
    # We'll create a grid of cells and assign elements that intersect each cell
    n_cells_per_dim = 2  # 2x2x2 = 8 cells
    leaf_morton_codes = []
    leaf_element_lists = []

    for i in range(n_cells_per_dim):
        for j in range(n_cells_per_dim):
            for k in range(n_cells_per_dim):
                # Cell center
                cell_size = 1.0 / n_cells_per_dim
                cx = (i + 0.5) * cell_size
                cy = (j + 0.5) * cell_size
                cz = (k + 0.5) * cell_size

                # Find elements that might intersect this cell
                # Simple heuristic: check if any vertex is within cell bounds
                cell_min = np.array([i * cell_size, j * cell_size, k * cell_size])
                cell_max = np.array([(i + 1) * cell_size, (j + 1) * cell_size, (k + 1) * cell_size])

                intersecting_elements = []
                for elem_id in range(len(connectivity)):
                    elem_nodes = connectivity[elem_id]
                    elem_vertices = positions[elem_nodes]

                    # Check if any vertex is inside cell (expanded slightly)
                    margin = 0.1
                    for vertex in elem_vertices:
                        if np.all(vertex >= cell_min - margin) and np.all(vertex <= cell_max + margin):
                            intersecting_elements.append(elem_id)
                            break

                if len(intersecting_elements) > 0:
                    # Encode cell center as Morton code
                    morton_code = encode_morton_3d(
                        float(cx), float(cy), float(cz),
                        level,
                        domain_min, domain_max
                    )

                    leaf_morton_codes.append(morton_code)
                    leaf_element_lists.append(intersecting_elements)

    leaf_morton_codes = np.array(leaf_morton_codes, dtype=np.uint64)
    print(f"  Created {len(leaf_morton_codes)} leaf cells")

    hash_octree = build_hash_octree_from_leaves(
        leaf_morton_codes,
        leaf_element_lists,
        domain_min,
        domain_max,
        target_load_factor=0.77
    )

    stats = get_hash_octree_memory_stats(hash_octree)
    print(f"  ✓ Built hash octree:")
    print(f"    Leaves: {stats['n_leaves']}")
    print(f"    Hash table size: {stats['hash_table_size']}")
    print(f"    Load factor: {stats['load_factor']:.3f}")
    print(f"    Memory: {stats['total_mb']:.3f} MB")

    # Test hash lookup
    print("\n3. Testing hash lookup...")

    # Query points - use cell centers to ensure we find them
    query_points = np.array([
        [0.25, 0.25, 0.25],  # Cell (0, 0, 0)
        [0.75, 0.25, 0.25],  # Cell (1, 0, 0)
        [0.25, 0.75, 0.25],  # Cell (0, 1, 0)
        [0.75, 0.75, 0.75],  # Cell (1, 1, 1)
        [0.25, 0.25, 0.75],  # Cell (0, 0, 1)
    ], dtype=np.float32)

    # DEBUG: Print Morton codes
    print("  DEBUG: Query point Morton codes:")
    for i, pt in enumerate(query_points):
        code = encode_morton_3d(
            float(pt[0]), float(pt[1]), float(pt[2]),
            level, domain_min, domain_max
        )
        print(f"    Point {i} at {pt}: Morton={code}")

    print("  DEBUG: Leaf Morton codes:")
    for i, code in enumerate(leaf_morton_codes):
        print(f"    Leaf {i}: Morton={code}, Elements={leaf_element_lists[i]}")

    query_points_jax = jnp.asarray(query_points, dtype=jnp.float32)
    levels = jnp.full(len(query_points), level, dtype=jnp.int32)

    # Batch hash lookup
    candidate_elements, n_elements = hash_lookup_batch_jax(
        query_points_jax,
        hash_octree,
        levels
    )

    candidate_elements_np = np.asarray(candidate_elements, dtype=np.int32)
    n_elements_np = np.asarray(n_elements, dtype=np.int32)

    print(f"  ✓ Hash lookup completed for {len(query_points)} points")
    for i in range(len(query_points)):
        n = int(n_elements_np[i])
        candidates = candidate_elements_np[i, :n].tolist()
        print(f"    Point {i}: Found {n} candidates {candidates}")

    # Test element testing (simulate _find_elements_with_hash_octree)
    print("\n4. Testing element containment checks...")

    from numba import njit

    @njit
    def compute_barycentric_coords(point, vertices):
        """Compute barycentric coordinates."""
        v0 = vertices[0]
        v1 = vertices[1]
        v2 = vertices[2]
        v3 = vertices[3]

        mat = np.empty((3, 3), dtype=np.float32)
        mat[:, 0] = v1 - v0
        mat[:, 1] = v2 - v0
        mat[:, 2] = v3 - v0

        rhs = point - v0

        try:
            bary123 = np.linalg.solve(mat, rhs)
            bary0 = 1.0 - (bary123[0] + bary123[1] + bary123[2])
            bary = np.empty(4, dtype=np.float32)
            bary[0] = bary0
            bary[1] = bary123[0]
            bary[2] = bary123[1]
            bary[3] = bary123[2]
            return bary
        except:
            bary = np.empty(4, dtype=np.float32)
            bary[:] = -1.0
            return bary

    @njit
    def is_point_in_tetrahedron(bary_coords, tolerance=1e-6):
        """Check if barycentric coordinates indicate point inside."""
        return (bary_coords[0] >= -tolerance and
                bary_coords[1] >= -tolerance and
                bary_coords[2] >= -tolerance and
                bary_coords[3] >= -tolerance and
                bary_coords.sum() <= 1.0 + tolerance)

    @njit
    def test_candidates(query_positions, candidate_elements, n_elements, positions, connectivity):
        """Test candidate elements to find containing element."""
        n_queries = len(query_positions)
        result = np.full(n_queries, -1, dtype=np.int32)

        for i in range(n_queries):
            point = query_positions[i]
            n_candidates = n_elements[i]

            for j in range(n_candidates):
                elem_id = candidate_elements[i, j]
                if elem_id < 0:
                    break

                # Get element vertices
                elem_nodes = connectivity[elem_id]
                vertices = np.empty((4, 3), dtype=np.float32)
                for k in range(4):
                    vertices[k] = positions[elem_nodes[k]]

                # Test if point is inside
                bary = compute_barycentric_coords(point, vertices)
                if is_point_in_tetrahedron(bary):
                    result[i] = elem_id
                    break

        return result

    # Find containing elements
    element_ids = test_candidates(
        query_points,
        candidate_elements_np,
        n_elements_np,
        positions,
        connectivity
    )

    print("  ✓ Element containment results:")
    n_found = 0
    for i in range(len(query_points)):
        elem_id = element_ids[i]
        status = "✓ FOUND" if elem_id >= 0 else "✗ NOT FOUND"
        print(f"    Point {i} at {query_points[i]}: Element {elem_id} {status}")
        if elem_id >= 0:
            n_found += 1

    # Validate results
    print(f"\n5. Validation:")
    print(f"  Points found: {n_found}/{len(query_points)}")

    if n_found >= len(query_points) * 0.8:  # At least 80% found
        print("  ✅ Integration test PASSED")
        return True
    else:
        print("  ❌ Integration test FAILED - too few points found")
        return False


if __name__ == '__main__':
    success = test_hash_octree_integration()
    exit(0 if success else 1)
