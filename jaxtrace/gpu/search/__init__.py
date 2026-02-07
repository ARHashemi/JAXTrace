"""
GPU Multi-Level Search Module - Phase 2-4

Implements hierarchical particle-to-element search with hash bucket subdivision
for heavy blocks.

Search Levels:
    L0: Cached element (last known position)
    L1: Neighbor elements (face-adjacent)
    L2a: Light block direct search (<10K elements)
    L2b: Heavy block hash bucket search (>10K elements)
    L3: Neighbor blocks (26-adjacent)

Phase 2 Block-Wise Search (NEW):
    Batched block-wise GPU kernels for use with the batching architecture.
    - search_particles_in_block(): JAX-native 3-level search
    - search_particles_in_block_with_hash(): Hash bucket optimization for heavy blocks
    - batch_search_light_blocks(): Combined light block search

Key Innovation:
    Hash bucket subdivision using Morton codes reduces heavy block search
    from O(900K) to O(200) elements per particle.
"""

from .block_classifier import (
    BlockClassification,
    classify_blocks,
    print_classification_summary,
)
from .hash_bucket import (
    HashBucketArrays,
    compute_morton_codes,
    build_hash_bucket_arrays,
)
from .hash_bucket_csr import (
    HashBucketArraysCSR,
    build_hash_bucket_arrays_csr,
)
from .level0_cached import search_level0_cached
from .level1_neighbors import search_level1_neighbors
from .level2a_light import search_level2a_light_block
from .level2b_heavy import search_level2b_hash_bucket
from .level2b_heavy_csr import search_level2b_hash_bucket_csr
from .level3_neighbor_blocks import search_level3_neighbor_blocks
from .multi_level_search import (
    SearchStats,
    multi_level_search_batch,
    multi_level_search_batch_vectorized,
)
from .multi_level_search_optimized import (
    multi_level_search_batch_optimized,
)
from .monitoring import (
    print_performance_report,
    save_performance_log,
)
from .initial_assignment import (
    InitialSearchStats,
    find_containing_block_jax,
    initial_search_single,
    initial_search_batch,
)
from .incremental_search import (
    IncrementalSearchStats,
    incremental_search_batch,
)

# Phase 2: Batched block-wise search kernels
from .block_search import (
    BlockSearchResult,
    search_particles_in_block,
    search_particles_in_block_with_hash,
    batch_search_light_blocks,
    compute_morton_code,
    lookup_hash_bucket,
    create_block_search_kernel,
)

# Global Morton search (Phase 3A - NEW HOT-like L2)
from .morton_global_builder import (
    GlobalMortonStructure,
    build_global_morton_structure,
)
from .morton_global_search import (
    MeshGPUGlobalMorton,
    morton_encode_position_jax,
    morton_encode_positions_batch,
    position_to_leaf_id_linear,
    search_L2_global_morton_single,
    upload_global_morton_to_gpu,
)

# Mesh-aligned octree (Phase 2: Cell Extraction)
from .mesh_aligned_octree import (
    OctreeCellData,
    extract_octree_cells_multi_insert,
    validate_searchability,
    encode_morton_3d,
    find_axis_aligned_edges,
    infer_cell_size_from_edges,
    find_all_overlapping_cells,
)
# Fast optimized version (assumes 8-cell pattern)
from .mesh_aligned_octree_fast import (
    extract_octree_cells_fast,
    encode_morton_3d_vectorized,
    compute_8cell_pattern,
)

# Mesh-aligned octree (Phase 3: GPU Structure)
from .mesh_aligned_octree_gpu import (
    MeshAlignedOctreeGPU,
    upload_mesh_aligned_octree_to_gpu,
    encode_morton_3d_jax,
    position_to_grid_indices,
    position_to_morton_code,
    find_cell_by_morton,
    find_cell_by_morton_and_level,
    get_cell_elements,
)

# Mesh-aligned octree (Phase 4: Point Location)
from .mesh_aligned_point_location import (
    search_mesh_aligned_octree_single,
    search_mesh_aligned_octree_batch,
    search_mesh_aligned_octree_multi_local,
    search_mesh_aligned_octree_multi_local_batch,
    search_mesh_aligned_multi_level,
    compute_search_statistics,
    print_search_statistics,
    search_mesh_aligned_octree_single_jit,
    search_mesh_aligned_octree_batch_jit,
    search_mesh_aligned_octree_multi_local_jit,
    search_mesh_aligned_octree_multi_local_batch_jit,
)

# Mesh-aligned Morton (Phase 5: Hybrid Approach)
from .mesh_aligned_morton_builder import (
    MeshAlignedMortonStructure,
    build_mesh_aligned_morton_structure,
    validate_mesh_aligned_morton_structure,
)
from .mesh_aligned_morton_search import (
    MeshAlignedMortonGPU,
    upload_mesh_aligned_morton_to_gpu,
    search_L2_mesh_aligned_morton_single,
    search_L2_mesh_aligned_morton_incremental_single,
    search_L2_mesh_aligned_morton_batch,
    search_L2_mesh_aligned_grid_neighbors_single,
    search_L2_mesh_aligned_grid_neighbors_batch,
)

# KD-tree node search (Phase 6: Simple and Direct)
from .kdtree_node_search import (
    NodeKDTreeStructure,
    NodeKDTreeGPU,
    build_kdtree_structure,
    upload_kdtree_to_gpu,
    search_L2_kdtree_single,
    search_L2_kdtree_batch,
    JAXKD_AVAILABLE,
)

# Mesh-aligned octree: Single-cell registration (Phase 2: Corrected)
from .mesh_aligned_octree_single_cell import (
    OctreeCellDataSingle,
    extract_octree_cells_single,
    encode_morton_3d_single,
    find_axis_aligned_edges_single,
    find_parent_cube,
)

# Mesh-aligned octree: Multi-cell vertex registration (Phase 2: Retention Fix)
from .mesh_aligned_octree_vertex_multi import (
    OctreeCellDataVertexMulti,
    extract_octree_cells_vertex_multi,
)

__all__ = [
    # Block classification
    'BlockClassification',
    'classify_blocks',
    'print_classification_summary',
    # Hash buckets
    'HashBucketArrays',
    'compute_morton_codes',
    'build_hash_bucket_arrays',
    # CSR hash buckets (Phase 1)
    'HashBucketArraysCSR',
    'build_hash_bucket_arrays_csr',
    # Level search (original)
    'search_level0_cached',
    'search_level1_neighbors',
    'search_level2a_light_block',
    'search_level2b_hash_bucket',
    'search_level2b_hash_bucket_csr',
    'search_level3_neighbor_blocks',
    # Multi-level search
    'SearchStats',
    'multi_level_search_batch',
    'multi_level_search_batch_vectorized',
    'multi_level_search_batch_optimized',
    # Monitoring
    'print_performance_report',
    'save_performance_log',
    # Initial assignment
    'InitialSearchStats',
    'find_containing_block_jax',
    'initial_search_single',
    'initial_search_batch',
    # Incremental search (L0+L1 optimized)
    'IncrementalSearchStats',
    'incremental_search_batch',
    # Phase 2: Batched block-wise search
    'BlockSearchResult',
    'search_particles_in_block',
    'search_particles_in_block_with_hash',
    'batch_search_light_blocks',
    'compute_morton_code',
    'lookup_hash_bucket',
    'create_block_search_kernel',
    # Global Morton search (Phase 3A)
    'GlobalMortonStructure',
    'build_global_morton_structure',
    'MeshGPUGlobalMorton',
    'morton_encode_position_jax',
    'morton_encode_positions_batch',
    'position_to_leaf_id_linear',
    'search_L2_global_morton_single',
    'upload_global_morton_to_gpu',
    # Mesh-aligned octree (Phase 2)
    'OctreeCellData',
    'extract_octree_cells_multi_insert',
    'validate_searchability',
    'encode_morton_3d',
    'find_axis_aligned_edges',
    'infer_cell_size_from_edges',
    'find_all_overlapping_cells',
    # Fast optimized version
    'extract_octree_cells_fast',
    'encode_morton_3d_vectorized',
    'compute_8cell_pattern',
    # Mesh-aligned octree (Phase 3)
    'MeshAlignedOctreeGPU',
    'upload_mesh_aligned_octree_to_gpu',
    'encode_morton_3d_jax',
    'position_to_grid_indices',
    'position_to_morton_code',
    'find_cell_by_morton',
    'get_cell_elements',
    # Mesh-aligned octree (Phase 4)
    'search_mesh_aligned_octree_single',
    'search_mesh_aligned_octree_batch',
    'search_mesh_aligned_octree_multi_local',
    'search_mesh_aligned_octree_multi_local_batch',
    'search_mesh_aligned_multi_level',
    'compute_search_statistics',
    'print_search_statistics',
    'search_mesh_aligned_octree_single_jit',
    'search_mesh_aligned_octree_batch_jit',
    'search_mesh_aligned_octree_multi_local_jit',
    'search_mesh_aligned_octree_multi_local_batch_jit',
    # Mesh-aligned Morton (Phase 5 - Hybrid)
    'MeshAlignedMortonStructure',
    'build_mesh_aligned_morton_structure',
    'validate_mesh_aligned_morton_structure',
    'MeshAlignedMortonGPU',
    'upload_mesh_aligned_morton_to_gpu',
    'search_L2_mesh_aligned_morton_single',
    'search_L2_mesh_aligned_morton_incremental_single',
    'search_L2_mesh_aligned_morton_batch',
    'search_L2_mesh_aligned_grid_neighbors_single',
    'search_L2_mesh_aligned_grid_neighbors_batch',
    # KD-tree node search (Phase 6)
    'NodeKDTreeStructure',
    'NodeKDTreeGPU',
    'build_kdtree_structure',
    'upload_kdtree_to_gpu',
    'search_L2_kdtree_single',
    'search_L2_kdtree_batch',
    'JAXKD_AVAILABLE',
    # Mesh-aligned octree: Single-cell (corrected)
    'OctreeCellDataSingle',
    'extract_octree_cells_single',
    'encode_morton_3d_single',
    'find_axis_aligned_edges_single',
    'find_parent_cube',
    # Mesh-aligned octree: Multi-cell vertex registration
    'OctreeCellDataVertexMulti',
    'extract_octree_cells_vertex_multi',
]
