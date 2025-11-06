"""
Forest-of-Octrees Block Management.

Part of Phase 1: Forest Structure & Block Partitioning

This module handles forest block creation, spatial partitioning, and
block metadata management for GPU-native particle tracking.
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

__all__ = [
    # Block grid
    "Block",
    "create_regular_grid",
    "compute_6_neighbors",
    "compute_26_neighbors",
    "position_to_block_id",
    "find_block_containing_point",
    "infer_grid_size",
    # Element mapping
    "BlockAssignmentStats",
    "compute_element_centroids",
    "assign_elements_to_blocks",
    "assign_elements_to_block_list",
    "validate_assignment",
]
