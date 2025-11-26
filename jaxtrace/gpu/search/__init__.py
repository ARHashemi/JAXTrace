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
from .level0_cached import search_level0_cached
from .level1_neighbors import search_level1_neighbors
from .level2a_light import search_level2a_light_block
from .level2b_heavy import search_level2b_hash_bucket
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

__all__ = [
    # Block classification
    'BlockClassification',
    'classify_blocks',
    'print_classification_summary',
    # Hash buckets
    'HashBucketArrays',
    'compute_morton_codes',
    'build_hash_bucket_arrays',
    # Level search (original)
    'search_level0_cached',
    'search_level1_neighbors',
    'search_level2a_light_block',
    'search_level2b_hash_bucket',
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
]
