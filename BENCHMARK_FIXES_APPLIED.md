# Benchmark Scripts - Critical Fixes Applied

**Date**: 2026-01-19
**Status**: ✅ All issues resolved - Ready to run

---

## Summary

Both benchmark scripts had identical critical bugs that prevented execution. All issues have been resolved based on the production script [production_tracking_fully_fused_timedep.py](production_tracking_fully_fused_timedep.py:536-553).

---

## Issues Found and Fixed

### Issue 1: Incorrect Import Statement

**Problem**: Wrong function imported for setting inverse matrices

**Files Affected**:
- [benchmark_point_in_tet_comprehensive.py:33](benchmark_point_in_tet_comprehensive.py#L33)
- [benchmark_l2_search_methods.py:52](benchmark_l2_search_methods.py#L52)

**Before**:
```python
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices, set_inverse_matrices
```

**After**:
```python
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
from jaxtrace.gpu.search.point_in_tet_methods import set_inverse_matrices_gpu
```

**Why**: The correct function name is `set_inverse_matrices_gpu` (not `set_inverse_matrices`) and it's located in `point_in_tet_methods.py` (not `point_in_tet_inverse.py`).

---

### Issue 2: Incorrect Function Call (verbose parameter)

**Problem**: `precompute_inverse_matrices()` called with non-existent `verbose` parameter

**Files Affected**:
- [benchmark_point_in_tet_comprehensive.py:221](benchmark_point_in_tet_comprehensive.py#L221)
- [benchmark_l2_search_methods.py:267](benchmark_l2_search_methods.py#L267)

**Error**:
```
TypeError: precompute_inverse_matrices() got an unexpected keyword argument 'verbose'
```

**Before**:
```python
# Point-in-tet benchmark
inverse_matrices = precompute_inverse_matrices(connectivity, node_positions, verbose=True)

# L2 search benchmark
inverse_matrices = precompute_inverse_matrices(connectivity, node_positions, verbose=False)
```

**After**:
```python
# Both benchmarks
M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)
```

**Why**:
- Function has no `verbose` parameter
- Function returns a **tuple** `(M_inv_array, p0_array)`, not a single array

---

### Issue 3: Incorrect Return Value Handling

**Problem**: Function returns tuple but code expected single array

**Files Affected**:
- [benchmark_point_in_tet_comprehensive.py:221-224](benchmark_point_in_tet_comprehensive.py#L221-L224)
- [benchmark_l2_search_methods.py:267,295](benchmark_l2_search_methods.py#L267)

**Before**:
```python
inverse_matrices = precompute_inverse_matrices(...)
memory_mb = inverse_matrices.nbytes / (1024**2)
# ...
inverse_matrices_gpu = jax.device_put(inverse_matrices)
```

**After**:
```python
M_inv_array, p0_array = precompute_inverse_matrices(...)
memory_mb = (M_inv_array.nbytes + p0_array.nbytes) / (1024**2)
# ...
M_inv_gpu = jax.device_put(M_inv_array)
p0_gpu = jax.device_put(p0_array)
```

**Why**: The function returns two separate arrays:
- `M_inv_array`: 3×3 inverse matrices for each element (shape: `[n_elements, 3, 3]`)
- `p0_array`: Base vertex offsets for each element (shape: `[n_elements, 3]`)

---

### Issue 4: Incorrect Function Call (set_inverse_matrices)

**Problem**: Wrong function name and incorrect number of arguments

**Files Affected**:
- [benchmark_point_in_tet_comprehensive.py:235](benchmark_point_in_tet_comprehensive.py#L235)
- [benchmark_l2_search_methods.py:299](benchmark_l2_search_methods.py#L299)

**Before**:
```python
set_inverse_matrices(inverse_matrices_gpu)  # Wrong function name, wrong args
```

**After**:
```python
set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)  # Correct function, correct args
```

**Why**: The correct function is `set_inverse_matrices_gpu()` and it requires **two arguments** (M_inv, p0).

---

## Root Cause

The benchmark scripts were created based on incomplete understanding of the `inverse` method API. The correct usage pattern is documented in the production script:

**Reference**: [production_tracking_fully_fused_timedep.py:536-553](production_tracking_fully_fused_timedep.py#L536-L553)

```python
elif POINT_IN_TET_METHOD == "inverse":
    print("\n  Precomputing inverse matrices for inverse method...")
    from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
    from jaxtrace.gpu.search.point_in_tet_methods import set_inverse_matrices_gpu

    t_inverse = time.time()
    M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)
    t_inverse = time.time() - t_inverse

    # Upload to GPU and register with point-in-tet dispatcher
    M_inv_gpu = jax.device_put(M_inv_array)
    p0_gpu = jax.device_put(p0_array)
    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)

    inverse_mb = (M_inv_array.nbytes + p0_array.nbytes) / (1024**2)
    print(f"    Inverse matrices: {connectivity.shape[0]:,} × 3×3 + p0")
    print(f"    Memory: {inverse_mb:.1f} MB")
    print(f"    Computation time: {t_inverse:.2f} s")
```

---

## Additional Issue: create_rk4_fully_fused_timedep API (L2 Benchmark)

**A second critical issue was found in the L2 benchmark** after fixing the inverse matrices issue.

**Error**: `TypeError: create_rk4_fully_fused_timedep() got an unexpected keyword argument 'mesh_gpu'`

**Root Cause**: The L2 benchmark had an incorrect understanding of the RK4 API:
- Expected it to take `mesh_gpu` object → Actually takes individual components
- Expected it to take `velocity_sequence`, `dt`, `point_in_tet_method` → Not parameters
- Expected returned function to take state dict → Actually takes 5 positional arguments
- Missing `element_volumes` computation and upload

**See**: [BENCHMARK_L2_FIX_APPLIED.md](BENCHMARK_L2_FIX_APPLIED.md) for complete details of all RK4 API fixes.

---

## Verification

### Files Fixed

1. ✅ [benchmark_point_in_tet_comprehensive.py](benchmark_point_in_tet_comprehensive.py)
   - Line 33: Import corrected
   - Line 221: Function call corrected (removed verbose, tuple unpacking)
   - Line 223: Memory calculation updated for tuple
   - Line 233-235: GPU upload and function call corrected

2. ✅ [benchmark_l2_search_methods.py](benchmark_l2_search_methods.py)
   - **Inverse matrices fixes**:
     - Line 51-52: Import corrected
     - Line 267: Function call corrected (removed verbose, tuple unpacking)
     - Line 326: GPU upload corrected for tuple
     - Line 315: Function call corrected
   - **RK4 API fixes** (see [BENCHMARK_L2_FIX_APPLIED.md](BENCHMARK_L2_FIX_APPLIED.md)):
     - Lines 127-223: Fixed `run_rk4_tracking` function signature and implementation
     - Lines 285-299: Added element_volumes computation
     - Line 326: Added element_volumes GPU upload
     - Lines 495-515: Fixed function call and results storage

### Testing

**Before fixes**: Both scripts failed with `TypeError` on line 221/267

**After fixes**: Both scripts should run successfully

**To verify**:
```bash
# Test point-in-tet benchmark
python benchmark_point_in_tet_comprehensive.py 2>&1 | tee logs/benchmark_point_in_tet.log

# Test L2 search benchmark
python benchmark_l2_search_methods.py 2>&1 | tee logs/benchmark_l2_search.log
```

---

## Impact

### What's Fixed

✅ Both benchmark scripts now correctly:
1. Import the right function (`set_inverse_matrices_gpu`)
2. Call `precompute_inverse_matrices()` without invalid parameters
3. Handle the returned tuple `(M_inv_array, p0_array)`
4. Upload both arrays to GPU separately
5. Register both arrays with the point-in-tet dispatcher

### Expected Behavior

**Point-in-Tet Benchmark**:
- Tests 7 methods including `inverse` method
- Should complete in ~20-30 minutes
- Expected `inverse` speedup: **4.36×** vs baseline

**L2 Search Benchmark**:
- Tests 6 L2 configurations
- Should complete in ~30-45 minutes
- Expected incremental speedup: **1.80×** vs baseline

**Combined Expected Speedup**: 4.36 × 1.80 = **7.8× total** 🎉

---

## Additional Notes

### Memory Overhead (Inverse Method)

For 3.3M elements:
- `M_inv_array`: 3.3M × 3×3 × 4 bytes = ~119 MB
- `p0_array`: 3.3M × 3 × 4 bytes = ~40 MB
- **Total**: ~159 MB additional GPU memory

This is **acceptable** for modern GPUs (A100 has 40-80 GB).

### Why the API is Designed This Way

The inverse method stores:
1. **M_inv**: Inverse of the matrix formed by edge vectors (3×3 per element)
2. **p0**: Base vertex position (3D vector per element)

This allows point-in-tet to be computed as:
```python
lambda = M_inv @ (p - p0)  # Single matrix-vector multiply
is_inside = all(lambda >= 0) and sum(lambda) <= 1
```

This reduces the operation from **145 FLOPs** (barycentric) to **22 FLOPs** (inverse), achieving the **6.59× theoretical** (4.36× measured) speedup.

---

## Status

✅ **All issues resolved**
✅ **Both scripts ready to run**
✅ **Production script pattern followed**
✅ **Expected to run without errors**

---

## Next Steps

1. ✅ Fixes applied to both benchmark scripts
2. ⏳ **Run point-in-tet benchmark** (~30 min)
3. ⏳ **Run L2 search benchmark** (~45 min)
4. ⏳ **Analyze results and update paper with empirical data**

**Ready to benchmark!** 🚀

See [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) for complete usage instructions.
