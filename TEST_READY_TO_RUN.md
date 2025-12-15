# Octree vs Blockwise Test - READY TO RUN

## Status: ✓ VALIDATED

The comparison test has been created and validated. All imports successful.

## Quick Start

```bash
source .venv/bin/activate
python test_octree_vs_blockwise_initialization.py
```

## What Was Fixed

### Initial Issues
1. **Wrong import**: `search_octree_batch` → `search_level2_octree_scan`
2. **Wrong blockwise initialization**: Fixed to match production pattern from [production_tracking_3hop_l2_octree.py](production_tracking_3hop_l2_octree.py)

### Corrections Applied

#### 1. Octree Search Function
```python
# NOW USES:
from jaxtrace.gpu.search.octree_search_gpu import search_level2_octree_scan

# With cached_element_ids initialized to -1 (all particles need search)
cached_element_ids = jnp.full(n_particles, -1, dtype=jnp.int32)
```

#### 2. Blockwise Initialization Sequence
```python
# CORRECTED WORKFLOW (from production_tracking_3hop_l2_octree.py):

# 1. Create block grid (list of Block objects)
blocks = create_regular_grid(bbox, grid_size)

# 2. Assign elements to blocks
element_to_block, stats = assign_elements_to_blocks(
    node_positions, connectivity, bbox, grid_size
)

# 3. Build padded arrays (requires element_neighbors)
padded_arrays = build_padded_block_arrays(
    element_to_block, stats,
    node_positions=node_positions,
    connectivity=connectivity,
    element_neighbors=element_neighbors
)

# 4. Classify blocks (light vs heavy)
classification = classify_blocks(padded_arrays, threshold=10000)

# 5. Build block 26-connectivity
block_neighbors_26 = np.array([b.neighbors_26 for b in blocks], dtype=np.int32)

# 6. Build hash buckets for heavy blocks only
hash_bucket_data = {}
for block_id in classification.heavy_blocks:
    # ... build hash bucket for this block
    hash_bucket_data[block_id] = hash_arrays
```

## Test Configuration

- **Mesh**: threadedAvtk_120.pvtu (real production mesh)
- **Grid**: 40×40×40 (64,000 blocks)
- **Octree**: max_depth=15, max_leaf_size=50, level_threshold=1.1
- **Particles**: 50,000 (at element centroids + 1% perturbation)

## Expected Runtime

- **Total**: ~30-60 seconds
  - Mesh loading: ~5s
  - Octree init: ~5-10s
  - Blockwise init: ~10-20s
  - Particle generation: ~5s
  - Octree search: <1s
  - Blockwise search: ~1-3s

## Output Format

```
================================================================================
FINAL COMPARISON: OCTREE vs BLOCKWISE
================================================================================

Method              Found           Accuracy        Time (s)     Throughput (p/s)
----------------------------------------------------------------------------------
Octree              100.00%         99.95%          0.1234       405,184.3
Blockwise           99.98%          99.92%          0.5678       88,028.7

✓ Octree is 4.60× FASTER than blockwise
✓ Both methods have similar accuracy

RECOMMENDATIONS:
✓ Both methods achieve >99.9% accuracy - suitable for production
✓ Recommend OCTREE for initial assignment (faster by 4.60×)
```

## Files

- **Test**: [test_octree_vs_blockwise_initialization.py](test_octree_vs_blockwise_initialization.py)
- **Docs**: [OCTREE_VS_BLOCKWISE_TEST.md](OCTREE_VS_BLOCKWISE_TEST.md)
- **Summary**: [TEST_OCTREE_VS_BLOCKWISE_READY.md](TEST_OCTREE_VS_BLOCKWISE_READY.md)

## Validation

✓ Test imports successfully
✓ All dependencies available
✓ Mesh path verified
✓ Function signatures correct
✓ Blockwise initialization matches production pattern
✓ Octree search uses correct function

## Next Steps After Running

Based on results, determine:

1. **If octree is fast (>200k p/s)**:
   - Octree is NOT the bottleneck
   - Debug Scenario #2 time-stepping loop
   - Check for GPU syncs, export overhead

2. **If blockwise is faster**:
   - Consider switching L2 from octree to blockwise
   - Measure end-to-end improvement

3. **If both are slow (<50k p/s)**:
   - General GPU issue
   - Profile with JAX profiler
   - Check GPU utilization

4. **If accuracy is low (<99%)**:
   - Inspect point-in-tet implementation
   - Check coordinate systems
   - Adjust tolerances

## The Test Is Ready

No additional configuration needed. Just run:

```bash
python test_octree_vs_blockwise_initialization.py
```
