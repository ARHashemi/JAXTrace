"""
GPU Multi-Level Search Module - Phase 4

Implements hierarchical particle-to-element search with hash bucket subdivision
for heavy blocks.

Search Levels:
    L0: Cached element (last known position)
    L1: Neighbor elements (face-adjacent)
    L2a: Light block direct search (<10K elements)
    L2b: Heavy block hash bucket search (>10K elements)
    L3: Neighbor blocks (26-adjacent)

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
)
from .monitoring import (
    print_performance_report,
    save_performance_log,
)

__all__ = [
    'BlockClassification',
    'classify_blocks',
    'print_classification_summary',
    'HashBucketArrays',
    'compute_morton_codes',
    'build_hash_bucket_arrays',
    'search_level0_cached',
    'search_level1_neighbors',
    'search_level2a_light_block',
    'search_level2b_hash_bucket',
    'search_level3_neighbor_blocks',
    'SearchStats',
    'multi_level_search_batch',
    'print_performance_report',
    'save_performance_log',
]
