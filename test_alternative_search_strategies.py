#!/usr/bin/env python3
"""
Alternative Search Strategy Comparison

Tests different point location approaches suggested in:
"Critical_Evaluation_of_A_Point_Location_Algorithm.md"

Strategies tested:
1. Current Implementation: Morton-based with pre-computed neighbors (Option B)
2. Direct Hash Table: (level, i, j, k) tuple as key
3. Hierarchical Descent: Walk down octree from root
4. Hybrid: Primary + neighbor fallback without Morton encoding

Performance metrics:
- Searchability (% found)
- Accuracy (% correct for ground truth)
- Throughput (particles/sec)
- Memory usage
- Tests per particle
"""

import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Dict

import jax
import jax.numpy as jnp

from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single
from jaxtrace.gpu.search.mesh_aligned_octree_with_neighbor_table import (
    add_neighbor_table_to_octree,
    upload_octree_with_neighbors_to_gpu
)
from jaxtrace.gpu.search.mesh_aligned_search_with_neighbors import (
    search_batch_with_precomputed_neighbors
)
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

# For point-in-tet testing
from jaxtrace.gpu.search.point_in_tet_methods import point_in_tet_gpu
import jaxtrace.config as config


# ============================================================================
# Strategy 1: Current Morton-based with Neighbors (Option B)
# ============================================================================

def strategy_morton_precomputed_neighbors(positions, octree_gpu, levels_to_try):
    """
    Current implementation: Pre-computed neighbor table with Morton encoding.

    Pros: Stable execution, no JAX memory issues
    Cons: 8× slower than baseline, Morton encoding overhead
    """
    return search_batch_with_precomputed_neighbors(
        positions, octree_gpu, levels_to_try=levels_to_try, max_tests_per_cell=20
    )


# ============================================================================
# Strategy 2: Direct Hash Table (Recommended by Critical Evaluation)
# ============================================================================

class DirectHashTableSearch:
    """
    Direct octree key lookup without Morton encoding.

    Key: (level, grid_i, grid_j, grid_k)
    Lookup: O(1) hash table

    Theory: Simpler and faster than Morton approach for point queries.
    """

    def __init__(self, octree_cells, connectivity, node_positions):
        self.connectivity_cpu = connectivity
        self.node_positions_cpu = node_positions

        # Build hash table: (level, i, j, k) -> list of element IDs
        self.cell_map = defaultdict(list)

        for cell_idx in range(octree_cells.n_cells):
            level = octree_cells.cell_levels[cell_idx]
            grid = octree_cells.cell_grid_indices[cell_idx]
            key = (int(level), int(grid[0]), int(grid[1]), int(grid[2]))

            # Get elements in this cell
            start = octree_cells.cell_to_elements_offsets[cell_idx]
            end = octree_cells.cell_to_elements_offsets[cell_idx + 1]

            elem_ids = octree_cells.cell_to_elements_data[start:end].tolist()
            self.cell_map[key] = elem_ids

        # Compute cell sizes per level
        unique_levels = np.unique(octree_cells.cell_levels)
        self.level_cell_sizes = {}

        for level in unique_levels:
            level_mask = octree_cells.cell_levels == level
            level_sizes = octree_cells.cell_sizes[level_mask]
            self.level_cell_sizes[int(level)] = level_sizes[0]

        # Bounding box (for normalization if needed)
        self.bbox_min = node_positions.min(axis=0)
        self.bbox_max = node_positions.max(axis=0)

        # Debug: Check grid index range
        all_grid_indices = octree_cells.cell_grid_indices
        grid_min = all_grid_indices.min(axis=0)
        grid_max = all_grid_indices.max(axis=0)

        available_levels = sorted(self.level_cell_sizes.keys())

        print(f"  Direct hash table built:")
        print(f"    Cells: {len(self.cell_map):,}")
        print(f"    Available levels: {available_levels}")
        print(f"    Grid index range: [{grid_min[0]}, {grid_max[0]}] × [{grid_min[1]}, {grid_max[1]}] × [{grid_min[2]}, {grid_max[2]}]")

    def point_to_key(self, point, level):
        """Convert point to octree key at specified level.

        MUST match find_parent_cube() logic in mesh_aligned_octree_single_cell.py:
            i = int(np.floor(centroid[0] / cell_size[0]))
            j = int(np.floor(centroid[1] / cell_size[1]))
            k = int(np.floor(centroid[2] / cell_size[2]))

        Grid indices are computed directly from world coordinates, NOT bbox-relative.
        """
        cell_size = self.level_cell_sizes[level]

        i = int(np.floor(point[0] / cell_size[0]))
        j = int(np.floor(point[1] / cell_size[1]))
        k = int(np.floor(point[2] / cell_size[2]))

        return (level, i, j, k)

    def get_neighbors(self, key):
        """Get 26 spatial neighbors of a cell."""
        level, i, j, k = key
        neighbors = []

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    neighbors.append((level, i + di, j + dj, k + dk))

        return neighbors

    def test_point_in_elements(self, point, element_ids):
        """Test point against list of elements."""
        for elem_id in element_ids:
            elem_nodes = self.connectivity_cpu[elem_id]
            elem_positions = self.node_positions_cpu[elem_nodes]

            # Simple point-in-tet check (barycentric)
            if self.point_in_tet_cpu(point, elem_positions):
                return elem_id

        return -1

    def point_in_tet_cpu(self, point, tet_vertices):
        """Simple barycentric point-in-tet test."""
        v0, v1, v2, v3 = tet_vertices

        # Compute barycentric coordinates
        mat = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
        try:
            coords = np.linalg.solve(mat, point - v0)
        except np.linalg.LinAlgError:
            return False

        # Check if inside
        if np.all(coords >= -1e-6) and np.sum(coords) <= 1.0 + 1e-6:
            return True

        return False

    def search_single(self, point, levels_to_try=(14, 13, 12, 11, 10, 9, 8, 7)):
        """Search for single point using direct hash table."""
        n_tests = 0

        for level in levels_to_try:
            # Skip levels not in octree
            if level not in self.level_cell_sizes:
                continue
            # Primary cell
            key = self.point_to_key(point, level)

            if key in self.cell_map:
                elem_id = self.test_point_in_elements(point, self.cell_map[key])
                n_tests += len(self.cell_map[key])

                if elem_id >= 0:
                    return elem_id, n_tests

                # Try neighbors
                for neighbor_key in self.get_neighbors(key):
                    if neighbor_key in self.cell_map:
                        elem_id = self.test_point_in_elements(point, self.cell_map[neighbor_key])
                        n_tests += len(self.cell_map[neighbor_key])

                        if elem_id >= 0:
                            return elem_id, n_tests

        return -1, n_tests

    def search_batch(self, positions_cpu, levels_to_try=(14, 13, 12, 11, 10, 9, 8, 7)):
        """Search batch of positions (CPU implementation)."""
        n_positions = len(positions_cpu)
        results = np.full(n_positions, -1, dtype=np.int32)
        tests = np.zeros(n_positions, dtype=np.int32)

        for i, point in enumerate(positions_cpu):
            results[i], tests[i] = self.search_single(point, levels_to_try)

        return results, tests


def strategy_direct_hash_table(positions_cpu, octree_cells, connectivity, node_positions, levels_to_try):
    """
    Strategy 2: Direct hash table lookup.

    No Morton encoding, pure (level, i, j, k) keys.
    Expected: Simpler, potentially faster.
    """
    searcher = DirectHashTableSearch(octree_cells, connectivity, node_positions)
    return searcher.search_batch(positions_cpu, levels_to_try)


# ============================================================================
# Strategy 3: Hierarchical Descent (Tree Walk)
# ============================================================================

# Note: This would require building full parent-child tree structure
# Skipped for now as our current implementation only stores leaf cells


# ============================================================================
# Test Harness
# ============================================================================

def run_strategy_comparison(
    positions_cpu,
    strategy_name,
    ground_truth,
    octree_cells,
    octree_gpu,
    connectivity,
    node_positions,
    levels_to_try
):
    """Run one search strategy and measure performance."""
    n_particles = len(positions_cpu)

    print(f"\n{'='*80}")
    print(f"Strategy: {strategy_name}")
    print("="*80)

    # Run search based on strategy
    t0 = time.time()

    if strategy_name == "Morton with Pre-Computed Neighbors (Option B)":
        positions_gpu = jnp.array(positions_cpu)
        found, tests = strategy_morton_precomputed_neighbors(
            positions_gpu, octree_gpu, levels_to_try
        )
        jax.block_until_ready(found)
        found_cpu = np.array(found)
        tests_cpu = np.array(tests)

    elif strategy_name == "Direct Hash Table (No Morton)":
        found_cpu, tests_cpu = strategy_direct_hash_table(
            positions_cpu, octree_cells, connectivity, node_positions, levels_to_try
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    t1 = time.time()
    elapsed = t1 - t0

    # Compute statistics
    n_found = np.sum(found_cpu >= 0)
    searchability = 100.0 * n_found / n_particles

    found_mask = found_cpu >= 0
    if np.any(found_mask):
        mean_tests = tests_cpu[found_mask].mean()
        median_tests = np.median(tests_cpu[found_mask])
        max_tests = tests_cpu.max()
    else:
        mean_tests = median_tests = max_tests = 0

    throughput = n_particles / elapsed

    # Ground truth accuracy
    accuracy = None
    if ground_truth is not None:
        n_correct = np.sum(found_cpu == ground_truth)
        accuracy = 100.0 * n_correct / n_particles

    # Print results
    print(f"  Particles: {n_particles:,}")
    print(f"  Found: {n_found:,} ({searchability:.2f}%)")
    if accuracy is not None:
        print(f"  Correct (ground truth): {n_correct:,} ({accuracy:.2f}%)")
    print(f"  Tests/particle:")
    print(f"    Mean: {mean_tests:.1f}")
    print(f"    Median: {median_tests:.0f}")
    print(f"    Max: {max_tests}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {throughput:,.0f} particles/sec")

    return {
        'name': strategy_name,
        'searchability': searchability,
        'accuracy': accuracy,
        'mean_tests': mean_tests,
        'median_tests': median_tests,
        'max_tests': max_tests,
        'time': elapsed,
        'throughput': throughput
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ALTERNATIVE SEARCH STRATEGY COMPARISON")
    print("="*80 + "\n")

    # Load mesh
    MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")

    print("Loading mesh...")
    node_positions, connectivity, _ = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern="featurelessAvtk_{timestep}.pvtu",
        timestep_range=(158, 159),
        field_name='Displacement',
        verbose=False
    )

    node_positions, connectivity, _, _ = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=None, verbose=False
    )
    print(f"  Mesh: {connectivity.shape[0]:,} elements, {node_positions.shape[0]:,} nodes\n")

    # Extract octree
    print("Extracting octree...")
    octree_cells = extract_octree_cells_single(node_positions, connectivity, verbose=False)
    print(f"  Cells: {octree_cells.n_cells:,}\n")

    # Build neighbor table for Option B
    print("Building neighbor table...")
    octree_with_neighbors = add_neighbor_table_to_octree(octree_cells, verbose=False)
    octree_gpu = upload_octree_with_neighbors_to_gpu(
        connectivity, node_positions, octree_with_neighbors, verbose=False
    )
    print(f"  ✅ Neighbor table built\n")

    # Generate test positions (element centroids for ground truth)
    n_particles = 10000
    np.random.seed(42)

    print(f"Generating {n_particles:,} test positions (element centroids)...")
    n_elements = connectivity.shape[0]
    selected_elements = np.random.choice(n_elements, n_particles, replace=True)

    positions_cpu = np.zeros((n_particles, 3), dtype=np.float32)
    for i, elem_idx in enumerate(selected_elements):
        elem_nodes = connectivity[elem_idx]
        elem_positions = node_positions[elem_nodes]
        positions_cpu[i] = elem_positions.mean(axis=0)

    ground_truth = selected_elements.copy()
    print(f"  ✅ Generated centroids (ground truth available)\n")

    # Test levels
    levels_to_try = (14, 13, 12, 11, 10, 9, 8, 7)

    # Run strategies
    results = []

    # Strategy 1: Morton with neighbors (Option B)
    r1 = run_strategy_comparison(
        positions_cpu,
        "Morton with Pre-Computed Neighbors (Option B)",
        ground_truth,
        octree_cells,
        octree_gpu,
        connectivity,
        node_positions,
        levels_to_try
    )
    results.append(r1)

    # Strategy 2: Direct hash table
    r2 = run_strategy_comparison(
        positions_cpu,
        "Direct Hash Table (No Morton)",
        ground_truth,
        octree_cells,
        octree_gpu,
        connectivity,
        node_positions,
        levels_to_try
    )
    results.append(r2)

    # Comparison table
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print("="*80)
    print()
    print(f"{'Strategy':<45} {'Found':<10} {'Correct':<10} {'Tests':<8} {'Throughput':<12}")
    print(f"{'':<45} {'%':<10} {'%':<10} {'(mean)':<8} {'(p/s)':<12}")
    print("-"*80)

    for r in results:
        acc_str = f"{r['accuracy']:.2f}" if r['accuracy'] is not None else "N/A"
        print(f"{r['name']:<45} {r['searchability']:>6.2f}    {acc_str:>6}    "
              f"{r['mean_tests']:>6.1f}  {r['throughput']:>10,.0f}")

    print()
    print("="*80)

    # Analysis
    print("\nANALYSIS")
    print("="*80)

    # Find best performer in each category
    best_searchability = max(results, key=lambda x: x['searchability'])
    best_throughput = max(results, key=lambda x: x['throughput'])
    best_efficiency = min(results, key=lambda x: x['mean_tests'])

    print(f"Best searchability: {best_searchability['name']} ({best_searchability['searchability']:.2f}%)")
    print(f"Best throughput: {best_throughput['name']} ({best_throughput['throughput']:,.0f} p/s)")
    print(f"Most efficient: {best_efficiency['name']} ({best_efficiency['mean_tests']:.1f} tests/particle)")

    print("\n" + "="*80 + "\n")
