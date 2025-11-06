"""
Forest-of-Octrees Block Management.

Part of Phase 1 & 2: Forest Structure, Block Partitioning, and Padded Arrays

This module handles forest block creation, spatial partitioning,
element neighbors, and padded array storage for GPU-native particle tracking.
"""

from .block_grid import (
    Block,
    create_regular_grid,
    compute_6_neighbors,
    compute_26_neighbors,
    position_to_block_id,
    find_block_containing_point,
    infer_grid_size,
)
from .block_mapper import (
    BlockAssignmentStats,
    compute_element_centroids,
    assign_elements_to_blocks,
    assign_elements_to_block_list,
    validate_assignment,
)
from .element_adjacency import (
    AdjacencyStats,
    get_tet_faces,
    build_face_to_element_map,
    extract_element_neighbors,
    validate_neighbor_symmetry,
)
from .padded_arrays import (
    PaddedArrays,
    build_padded_block_arrays,
    validate_padded_arrays,
    get_block_element_list,
    print_memory_comparison,
)

__all__ = [
    # Block grid (Phase 1)
    "Block",
    "create_regular_grid",
    "compute_6_neighbors",
    "compute_26_neighbors",
    "position_to_block_id",
    "find_block_containing_point",
    "infer_grid_size",
    # Element mapping (Phase 1)
    "BlockAssignmentStats",
    "compute_element_centroids",
    "assign_elements_to_blocks",
    "assign_elements_to_block_list",
    "validate_assignment",
    # Element adjacency (Phase 2)
    "AdjacencyStats",
    "get_tet_faces",
    "build_face_to_element_map",
    "extract_element_neighbors",
    "validate_neighbor_symmetry",
    # Padded arrays (Phase 2)
    "PaddedArrays",
    "build_padded_block_arrays",
    "validate_padded_arrays",
    "get_block_element_list",
    "print_memory_comparison",
]
