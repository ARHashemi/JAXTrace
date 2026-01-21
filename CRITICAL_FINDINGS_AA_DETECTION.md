# CRITICAL FINDING: FLA Mesh is NOT Axis-Aligned!

## Summary

The corrected AA detection algorithm revealed that the FLA mesh is **only 0.06% axis-aligned** (1,820 out of 3,048,900 elements), NOT 100% as expected.

## Test Results (Partial)

From [logs/test_point_in_tet_production_benchmark_corrected.log](logs/test_point_in_tet_production_benchmark_corrected.log):

```
[3/9] Precomputing corrected AA metadata...

================================================================================
Precomputing Axis-Aligned Metadata (Corrected Algorithm)
================================================================================
  Elements: 3,048,900
  Sampling edge lengths to determine adaptive tolerance...
  Edge length range: 7.81e-05 to 8.72e-03
  Dynamic range: 111.7×
  Base tolerance: 1.00e-10 (relative)

  Processing 3,048,900 elements...
    Progress: 100.0% (3,048,900/3,048,900)

  ✅ Detection complete!
  Axis-aligned elements: 1,820/3,048,900 (0.06%)
  ❌ Only 0.1% AA → Skip AA optimization
================================================================================
```

## Implications

### 1. Original Assumption Was Wrong

**Expected**: FLA mesh generated with Kuhn decomposition + octree refinement → 100% axis-aligned tetrahedra

**Reality**: Only 0.06% axis-aligned (1,820 / 3,048,900 elements)

**Why?**: Possible reasons:
- Kuhn decomposition not strictly axis-aligned (or mesh was rotated)
- Mesh includes surface/boundary elements with arbitrary orientation
- Mesh deformation/warping from simulation
- Octree refinement introduces non-axis-aligned transitions

### 2. Pure AA Method Will NOT Help

The `pure_aa` method (11 FLOPs) was designed for 100% axis-aligned meshes. With only 0.06% AA:
- **pure_aa**: Will fail on 99.94% of elements → particles will be lost
- **branchless_hybrid**: Will execute both branches for every element → overhead with no benefit
- **skala_memory_opt**: The ONLY viable corrected method (works on all elements)

### 3. Expected Performance

**Best case (skala_memory_opt only)**:
- Computational speedup: 3× (145/48 FLOPs)
- Memory speedup: 4× (coalesced access)
- **Real speedup: ~2.3×** (Amdahl's Law with 50% memory-bound)
- Throughput: ~260 p/s (vs 112 p/s baseline)

**Comparison to OLD methods**:
- OLD skala: 100 p/s (0.90× slower) - memory-bound, random access
- OLD axis_aligned: 50 p/s (0.45× slower) - BROKEN (lax.cond overhead)
- **NEW skala_memory_opt: ~260 p/s (2.3× faster)** - memory-optimized

### 4. Test Modifications Needed

The test currently tries to benchmark all 6 methods, but:

1. **pure_aa** will fail (only works on 0.06% of elements)
2. **branchless_hybrid** will be slower than skala_memory_opt (unnecessary branching overhead)
3. **Only skala_memory_opt is viable** for this mesh

**Recommendation**:
- Skip `pure_aa` and `branchless_hybrid` for FLA mesh
- Focus on comparing OLD skala vs NEW skala_memory_opt
- This validates the memory optimization alone (not the AA optimization)

## Test Status

**Fixes Applied**:
1. ✅ Fixed `jax.tree_map` deprecation → `jax.tree_util.tree_map`
2. ✅ Fixed dataclass upload - manually upload each field instead of tree_map on dataclass

**Ready to run**: ✅ Yes (user will run manually)

**Expected outcome**:
```
Method               Throughput    Speedup    Notes
─────────────────────────────────────────────────────────────
current              112 p/s       1.00×      Baseline
skala (OLD)          100 p/s       0.90×      Memory-bound
axis_aligned (OLD)    50 p/s       0.45×      BROKEN
pure_aa (NEW)        FAIL          N/A        Mesh not AA
skala_memory_opt     ~260 p/s      ~2.3×      Memory-optimized ✅
branchless_hybrid    ~200 p/s?     ~1.8×?     Overhead from branching
```

## Root Cause: Why Was Mesh Assumed to be AA?

Need to investigate:
1. Check mesh generation code - is Kuhn decomposition actually axis-aligned?
2. Check if mesh was rotated or transformed during simulation
3. Sample a few elements and visualize their geometry
4. Check if octree refinement preserves axis-alignment

## Next Steps

1. **User runs test** to confirm skala_memory_opt achieves ~2× speedup
2. **Investigate mesh geometry** to understand why it's not axis-aligned
3. **Document findings** - pure AA optimization only applicable to truly axis-aligned meshes
4. **Integrate skala_memory_opt** into production (still 2× speedup from memory optimization alone)

## Files Modified

1. [test_point_in_tet_production_benchmark.py](test_point_in_tet_production_benchmark.py) - Fixed `jax.tree_map` → `jax.tree_util.tree_map`
2. [jaxtrace/gpu/search/aa_detection.py](jaxtrace/gpu/search/aa_detection.py) - Complete corrected implementation
3. [jaxtrace/gpu/search/point_in_tet_methods.py](jaxtrace/gpu/search/point_in_tet_methods.py) - Dispatcher with new methods

---

**Status**: ✅ READY FOR USER TO RUN

User will run test manually. Expected result: skala_memory_opt achieves ~2× speedup (not 3-4× due to mesh not being AA).
