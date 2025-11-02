"""
Forest-of-Octrees Block Management.

This module handles forest block creation, spatial partitioning, and
block metadata management for GPU-native particle tracking.
"""

from .block_builder import (
    BlockMetadata,
    create_regular_forest_grid,
    find_block_containing_point,
)
from .visualize import (
    visualize_forest_blocks,
    visualize_forest_with_mesh_pieces,
)

__all__ = [
    "BlockMetadata",
    "create_regular_forest_grid",
    "find_block_containing_point",
    "visualize_forest_blocks",
    "visualize_forest_with_mesh_pieces",
]
