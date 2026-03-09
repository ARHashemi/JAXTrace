# Mesh-Aligned Morton Hybrid Approach - Implementation

**Date**: 2026-01-27
**Status**: Implemented, ready for testing

## Summary

Implemented hybrid approach combining:
1. **Intrinsic mesh octree structure** (from mesh-aligned cells)
2. **Proven Morton radius search algorithm** (93-98% retention)

This addresses the 74.6% retention issue in direct mesh-aligned octree by using radius search to handle elements spanning multiple cells.

## Key Insight from Critical Reviews

All three critical reviews (GPT5.2, OPUS, SUNNET) unanimously recommended:
- **Use Morton radius search OVER mesh-aligned cell centers** (not element centroids)
- Treat mesh-aligned octree as spatial index
- Leverage both intrinsic mesh structure + proven search algorithm
- Avoids nested control flow → no OOM issues

## Implementation

### New Modules

1. **[mesh_aligned_morton_builder.py](jaxtrace/gpu/search/mesh_aligned_morton_builder.py)**
   - `MeshAlignedMortonStructure`: Data structure for cell-based Morton octree
   - `build_mesh_aligned_morton_structure()`: Build Morton structure from cell centers
   - `validate_mesh_aligned_morton_structure()`: Validation checks

2. **[mesh_aligned_morton_search.py](jaxtrace/gpu/search/mesh_aligned_morton_search.py)**
   - `MeshAlignedMortonGPU`: GPU-resident structure
   - `upload_mesh_aligned_morton_to_gpu()`: Upload to GPU
   - `search_L2_mesh_aligned_morton_single()`: Single-particle search (radius)
   - `search_L2_mesh_aligned_morton_incremental_single()`: Incremental radius search
   - `search_L2_mesh_aligned_morton_batch()`: Batch search (for testing)

### Updated Modules

3. **[jaxtrace/config.py](jaxtrace/config.py)**
   - Added `"mesh_aligned_morton"` option to `L2_SEARCH_METHOD`
   - Updated documentation with performance comparison table

4. **[rk4_fully_fused_timedep.py](jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py)**
   - Added `mesh_aligned_morton` parameter
   - Updated `search_l2_single()` to support new method
   - Respects single-position constraint (no nested vmap)

5. **[production_tracking_fully_fused_timedep.py](production_tracking_fully_fused_timedep.py)**
   - Build and upload mesh-aligned Morton structure
   - Pass to RK4 integrator
   - Conditional based on `config.L2_SEARCH_METHOD`

6. **[benchmark_l2_search_methods.py](benchmark_l2_search_methods.py)**
   - Added two test configurations:
     - `"Mesh-Aligned Morton r=2"`: Fixed radius=2
     - `"Mesh-Aligned Morton (2,5,10)"`: Incremental radius

7. **[jaxtrace/gpu/search/__init__.py](jaxtrace/gpu/search/__init__.py)**
   - Export new functions and classes

### Test Script

8. **[test_mesh_aligned_morton.py](test_mesh_aligned_morton.py)**
   - Quick validation test
   - Loads mesh, builds structure, runs search
   - Reports success rate

## Architecture Comparison

### Original Morton (Element Centroids)
```
Position → Morton code (from query position)
         → Binary search in sorted element list
         → Find leaf (107 elements/leaf avg)
         → Search ±radius leaves (2R+1 leaves)
         → Test elements in leaves
```
- **Element centroids** → Morton codes
- 3,048,900 Morton codes (one per element)
- ~107 elements/leaf
- radius=2 → 5 leaves × 107 elem/leaf = ~536 tests

### Direct Mesh-Aligned Octree (Current)
```
Position → Compute grid indices (i, j, k)
         → Morton code
         → Binary search in sorted cell list
         → Find cell
         → Test elements in cell (center cell only)
```
- **Element centroids** → Assign to ONE cell
- 517,069 cells extracted from mesh
- ~5.9 elements/cell
- **Problem**: Elements span multiple cells → 74.6% retention

### Mesh-Aligned Morton (HYBRID - NEW)
```
Position → Morton code (from query position)
         → Binary search in sorted cell list
         → Find cell
         → Search ±radius cells (2R+1 cells)
         → Test elements in cells
```
- **Cell centers** → Morton codes
- 517,069 Morton codes (one per cell, not per element)
- ~5.9 elements/cell
- radius=2 → 5 cells × 5.9 elem/cell = ~30 tests
- **Handles elements spanning cells** via radius search

## Expected Performance

### Comparison Table

| Method                    | Morton Codes | Tests/Particle | Expected Retention |
|---------------------------|--------------|----------------|-------------------|
| Original Morton (r=2)     | 3M elements  | ~536           | 93-98%            |
| Direct mesh-aligned       | 517K cells   | ~5.9           | 74.6%             |
| **Hybrid mesh-aligned (r=2)** | **517K cells** | **~30**        | **~98%**          |

### Key Benefits

1. **18× fewer tests** than original Morton (30 vs 536)
2. **Handles spanning elements** via radius search (vs 74.6% direct)
3. **Better data locality**: Cell-based grouping vs element-based
4. **No nested control flow**: Single vmap over particles
5. **Proven algorithm**: Uses same radius search as production Morton

## Configuration

### Using the New Method

Set in [jaxtrace/config.py](jaxtrace/config.py):

```python
L2_SEARCH_METHOD = "mesh_aligned_morton"
```

### Available Options

1. **`"morton"`** (default)
   - Original: Morton codes from element centroids
   - Works with any mesh
   - ~536 tests, 93-98% retention

2. **`"mesh_aligned_octree"`**
   - Direct cell lookup (center cell only)
   - ~5.9 tests, 74.6% retention
   - Requires Kuhn mesh

3. **`"mesh_aligned_morton"`** (NEW)
   - Hybrid: Morton radius over cell centers
   - ~30 tests (r=2), ~98% expected retention
   - Requires Kuhn mesh

### L2 Search Sub-Methods

When using `"mesh_aligned_morton"`, you can choose:

1. **Fixed radius** (set `L2_SEARCH_RADIUS` in production script)
   - radius=2: 5 cells, ~30 tests
   - radius=5: 11 cells, ~65 tests
   - radius=10: 21 cells, ~124 tests

2. **Incremental** (set `L2_SEARCH_METHOD = 'incremental'` in RK4 call)
   - Cascading radii: (2, 5, 10)
   - Adaptive search depth
   - Expected ~11.5 cells avg (60/30/10 distribution)

## Testing

### Quick Test

```bash
python test_mesh_aligned_morton.py
```

Expected output:
- Builds structure: ~517K cells
- Tests 1,000 random particles
- Reports success rate (30-50% due to void regions)
- For particles inside mesh: expect ~98%

### Benchmark All Methods

```bash
python benchmark_l2_search_methods.py
```

Tests 9 L2 methods including:
- Original Morton (radius, incremental)
- Direct mesh-aligned octree
- **NEW: Mesh-aligned Morton (radius=2)**
- **NEW: Mesh-aligned Morton (incremental 2,5,10)**

### Production Test

```bash
python production_tracking_fully_fused_timedep.py
```

Set `config.L2_SEARCH_METHOD = "mesh_aligned_morton"` first.

## Design Decisions

### Why Morton Codes from Cell Centers (Not Element Centroids)?

1. **Fewer Morton codes**: 517K cells vs 3M elements
2. **Natural grouping**: Elements pre-grouped by parent cell
3. **Better locality**: Cells are spatially coherent
4. **Intrinsic structure**: Leverages mesh generator's octree

### Why Radius Search (Not Neighbor Arithmetic)?

1. **Proven algorithm**: 93-98% retention in production
2. **No nested control flow**: Avoids 631 GB OOM
3. **Simple implementation**: Reuses existing Morton radius search
4. **Handles spanning elements**: Searches multiple cells systematically

### Why Not 27-Neighbor Stencil?

1. **Memory explosion**: Nested lax.cond caused 631 GB allocation
2. **Overkill for most cases**: Radius search covers spanning elements
3. **Complexity**: Harder to implement without nested control flow

## Memory and Performance

### CPU Memory

- Cell extraction: ~50 MB (517K cells)
- Morton structure: ~60 MB
- Total: ~110 MB (vs ~200 MB for original Morton)

### GPU Memory

- Cell structure: Same as direct mesh-aligned octree
- No additional overhead vs direct approach
- Less than original Morton (fewer codes)

### Compilation

- Single vmap over particles (same as all methods)
- No nested control flow
- No memory explosion
- Expected compilation time: ~60s (first run)

## References

### Critical Reviews

1. [MESH_ALIGNED_OCTREE_COMPREHENSIVE_ANALYSIS_Critical_Review_GPT5.2.md](MESH_ALIGNED_OCTREE_COMPREHENSIVE_ANALYSIS_Critical_Review_GPT5.2.md)
   - "Keep your fast 'position → leaf' mapping, then add a deterministic escalation policy"
   - Recommended Morton radius over mesh structure

2. [MESH_ALIGNED_OCTREE_COMPREHENSIVE_ANALYSIS_Critical_Review_OPUS.md](MESH_ALIGNED_OCTREE_COMPREHENSIVE_ANALYSIS_Critical_Review_OPUS.md)
   - "Use the aligned octree cell centers or centroids to generate Morton codes"
   - Treat octree as spatial index

3. [MESH_ALIGNED_OCTREE_COMPREHENSIVE_ANALYSIS_Critical_Review_SUNNET.md](MESH_ALIGNED_OCTREE_COMPREHENSIVE_ANALYSIS_Critical_Review_SUNNET.md)
   - "Morton codes over cell centers + radius search"
   - Clear explanation of Morton as hash, not structure

### Previous Work

- [MESH_ALIGNED_OCTREE_COMPREHENSIVE_ANALYSIS.md](MESH_ALIGNED_OCTREE_COMPREHENSIVE_ANALYSIS.md) - Original implementation
- [MESH_ALIGNED_OCTREE_NEIGHBOR_SEARCH_FIX.md](MESH_ALIGNED_OCTREE_NEIGHBOR_SEARCH_FIX.md) - 74.6% diagnosis
- [logs/analyze_unfound_particles.log](logs/analyze_unfound_particles.log) - 100% of unfound ARE inside elements

## Next Steps

1. **Run quick test**: `python test_mesh_aligned_morton.py`
2. **Run benchmark**: `python benchmark_l2_search_methods.py`
3. **Compare results**: Check retention vs original Morton
4. **Production test**: Full 225K particles, 1000 steps
5. **Tune radius**: Try radius=1, 2, 3, 5, 10
6. **Tune incremental**: Try different tier configurations

## Expected Outcome

Based on analysis and reviews:
- **Retention**: ~98% (vs 93-98% original Morton, 74.6% direct)
- **Tests/particle**: ~30 (vs ~536 original Morton, ~5.9 direct)
- **Speedup**: ~18× fewer tests than original Morton
- **Accuracy**: Better than original (intrinsic mesh structure)

If radius=2 insufficient, fallback tiers (2,5,10) should catch remainder.

---

**Status**: Implementation complete. Ready for user testing.
