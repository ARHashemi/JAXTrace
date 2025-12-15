#!/usr/bin/env python3
"""
Test Global Morton Builder - Phase 1 Validation

Quick test to validate Morton encoding, sorting, and leaf segmentation
on the ThreadedA mesh (3.5M elements).
"""

import os
import sys
import time
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# JAXTrace imports
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.search.morton_global_builder import build_global_morton_structure


# Configuration
MESH_PATH = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_159.pvtu"
LEAF_CAPACITY = 256
MAX_DEPTH = 21


def main():
    logger.info("=" * 80)
    logger.info("Global Morton Builder - Phase 1 Validation Test")
    logger.info("=" * 80)

    # ========================================================================
    # 1. Load Mesh
    # ========================================================================

    logger.info("\n[1/3] Loading mesh...")
    t_load = time.time()
    node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
        Path(MESH_PATH),
        field_name='Displacement'
    )
    t_load = time.time() - t_load

    n_nodes = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    logger.info(f"  Mesh: {n_elements:,} elements, {n_nodes:,} nodes")
    logger.info(f"  Load time: {t_load:.2f}s")

    # ========================================================================
    # 2. Build Global Morton Structure
    # ========================================================================

    logger.info("\n[2/3] Building global Morton structure...")
    t_build = time.time()

    morton_struct = build_global_morton_structure(
        node_positions=node_positions,
        connectivity=connectivity,
        leaf_capacity=LEAF_CAPACITY,
        max_depth=MAX_DEPTH,
        verbose=True
    )

    t_build = time.time() - t_build
    logger.info(f"\n  Build time: {t_build:.2f}s")

    # ========================================================================
    # 3. Validate Structure
    # ========================================================================

    logger.info("\n[3/3] Validating Morton structure...")

    # Test 1: Check all elements are present
    unique_elem_ids = np.unique(morton_struct.elem_ids_sorted)
    assert len(unique_elem_ids) == n_elements, "Not all elements in sorted list"
    assert unique_elem_ids[0] == 0, "Missing element 0"
    assert unique_elem_ids[-1] == n_elements - 1, "Missing last element"
    logger.info("  ✅ All elements present in sorted list")

    # Test 2: Check Morton codes are sorted
    morton_sorted = morton_struct.morton_sorted
    assert np.all(morton_sorted[:-1] <= morton_sorted[1:]), "Morton codes not sorted"
    logger.info("  ✅ Morton codes are sorted")

    # Test 3: Check leaf coverage
    total_elems_in_leaves = np.sum(morton_struct.leaf_length)
    assert total_elems_in_leaves == n_elements, "Leaves don't cover all elements"
    logger.info(f"  ✅ Leaves cover all {n_elements:,} elements")

    # Test 4: Check no leaf exceeds capacity
    max_leaf_len = np.max(morton_struct.leaf_length)
    assert max_leaf_len <= LEAF_CAPACITY, f"Leaf exceeds capacity: {max_leaf_len}"
    logger.info(f"  ✅ No leaf exceeds capacity ({max_leaf_len} ≤ {LEAF_CAPACITY})")

    # Test 5: Check leaf start/length consistency
    for i in range(morton_struct.n_leaves):
        start = morton_struct.leaf_start[i]
        length = morton_struct.leaf_length[i]
        expected_start = i * LEAF_CAPACITY
        assert start == expected_start, f"Leaf {i} start mismatch"
        assert length > 0, f"Leaf {i} has zero elements"
        if i < morton_struct.n_leaves - 1:
            # Not last leaf
            assert length == LEAF_CAPACITY, f"Non-last leaf {i} not full"
    logger.info(f"  ✅ All {morton_struct.n_leaves:,} leaves have consistent start/length")

    # Test 6: Check Morton range
    assert morton_struct.morton_min == morton_sorted[0], "Morton min mismatch"
    assert morton_struct.morton_max == morton_sorted[-1], "Morton max mismatch"
    logger.info(f"  ✅ Morton range: [{morton_struct.morton_min}, {morton_struct.morton_max}]")

    # Test 7: Check bounding box
    bbox_min = node_positions.min(axis=0)
    bbox_max = node_positions.max(axis=0)
    assert np.allclose(morton_struct.bbox_min, bbox_min), "Bbox min mismatch"
    assert np.allclose(morton_struct.bbox_max, bbox_max), "Bbox max mismatch"
    logger.info(f"  ✅ Bounding box correct")

    # ========================================================================
    # Summary
    # ========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✅ All validation tests passed")
    logger.info(f"✅ Mesh: {n_elements:,} elements, {n_nodes:,} nodes")
    logger.info(f"✅ Leaves: {morton_struct.n_leaves:,} (capacity: {LEAF_CAPACITY})")
    logger.info(f"✅ Memory: {(morton_struct.elem_ids_sorted.nbytes + morton_struct.leaf_start.nbytes + morton_struct.leaf_length.nbytes) / (1024**2):.2f} MB")
    logger.info(f"✅ Build time: {t_build:.2f}s")
    logger.info("=" * 80)
    logger.info("\n🎉 Phase 1 (CPU Preprocessing) COMPLETE and VALIDATED!")
    logger.info("    Ready to proceed with Phase 2 (GPU Search Kernel)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
