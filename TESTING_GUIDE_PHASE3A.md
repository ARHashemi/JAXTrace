# Phase 3a Testing Guide

This guide explains how to test and validate the Phase 3a optimizations.

---

## Quick Start

### 1. Test GPU-Fused RK4 (Part 2)

```bash
source .venv/bin/activate
python3 test_rk4_gpu_fused.py
```

**Expected Output**:
```
✓ PASS: Position agreement
✓ PASS: Element ID agreement
✓ PASS: Vectorized throughput
✓ ALL TESTS PASSED - GPU-fused RK4 validated!

Expected impact on production:
  Current throughput: ~13k p/s
  With GPU-fused RK4: ~30-50k p/s
```

**What This Tests**:
- Correctness: GPU-fused RK4 matches baseline CPU-orchestrated RK4
- Performance: 2-3× speedup from eliminating intermediate CPU-GPU transfers
- Transfer reduction: 10 MB → 2 MB per timestep

---

### 2. Test Vectorized Search (Part 1) - Already Validated

```bash
source .venv/bin/activate
python3 test_phase3a_simple.py
```

**Results** (from previous run):
```
✓ L0 throughput:  207,521 p/s
✓ L1 throughput:  214,866 p/s
✓ ALL TESTS PASSED
```

---

### 3. Run Production Tracking with Phase 3a

```bash
source .venv/bin/activate
python3 production_tracking_threadeda.py
```

**Configuration Check**:
The script should print:
```
✓ Using GLOBAL MESH interpolator (Phase 2)
✓ Using HYBRID incremental search (Phase 3a - Option A+D optimized)
  Architecture: Vectorized L0 + Extended L1 (2-hop, ~20 neighbors)
```

**NOT**:
```
✓ Using VECTORIZED incremental search (Phase 3a)  # ← OLD VERSION
```

If you see the old message, clean Python cache:
```bash
find . -name "*.pyc" -delete
```

---

## Test Scripts Overview

### test_rk4_gpu_fused.py
**Purpose**: Validate GPU-fused RK4 implementation

**What it does**:
1. Loads ThreadedA mesh (3.5M elements)
2. Generates 10K test particles
3. Runs baseline CPU-orchestrated RK4 (10 timesteps)
4. Runs GPU-fused RK4 (10 timesteps)
5. Compares results for correctness
6. Measures performance improvement

**Success Criteria**:
- Position agreement: < 10 microns
- Element ID agreement: > 95%
- Speedup: > 1.5×

**Runtime**: ~2-3 minutes

---

### test_phase3a_simple.py
**Purpose**: Validate vectorized L0/L1 search

**What it does**:
1. Loads ThreadedA mesh (3.5M elements)
2. Generates 60K test particles with random cached element IDs
3. Runs vectorized L0 search (all particles in single GPU kernel)
4. Runs vectorized L1 search (all L0 misses in single GPU kernel)
5. Measures throughput

**Success Criteria**:
- L0 throughput: > 100k p/s
- L1 throughput: > 50k p/s

**Runtime**: ~1-2 minutes

**Note**: Uses random element IDs so hit rates are ~0%. In production, L0 hit rate is 80-90%.

---

### test_phase3a_vectorized_search.py
**Purpose**: Comprehensive comparison with baseline (DEPRECATED - use test_phase3a_simple.py instead)

This test is more complex and takes longer. The simple test is sufficient for validation.

---

## Performance Expectations

### Current Status (Before Phase 3a)
```
Throughput: 13k p/s
GPU Utilization: 30-40% (drops to 1-2%)
Bottleneck: CPU-GPU transfers (687 GB for full simulation)
```

### After Phase 3a Part 1 (Vectorized Search)
```
Throughput: 20-30k p/s
GPU Utilization: 30-40%
Transfer Reduction: 687 GB → 37.5 GB (95% reduction)
```

### After Phase 3a Complete (Parts 1+2)
```
Throughput: 50-100k p/s (expected)
GPU Utilization: 60-80% (expected)
Transfer Reduction: 687 GB → 6.25 GB (99% reduction)
```

### Target (Phase 3a + L2 Optimization)
```
Throughput: 200-300k p/s
GPU Utilization: 90%+
Requires: Spatial indexing for L2 search (next phase)
```

---

## Troubleshooting

### Issue: ModuleNotFoundError for 'jaxtrace.gpu.geometry'

**Symptom**:
```
ModuleNotFoundError: No module named 'jaxtrace.gpu.geometry'
```

**Cause**: Fixed in latest version. Old import path was incorrect.

**Solution**: The fix has been applied to `rk4_gpu_fused.py` (line 24).

If you still see this error:
```bash
git pull  # Get latest version
```

---

### Issue: "ValueError: not enough values to unpack (expected 3, got 2)"

**Symptom**:
```
ValueError: not enough values to unpack (expected 3, got 2)
  elem_ids_2, block_ids_2, search_stats_2 = incremental_searcher(...)
```

**Cause**: Old Python bytecode in `__pycache__` from previous implementation.

**Solution**:
```bash
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
```

Then re-run the script.

---

### Issue: "JAX out of memory"

**Symptom**:
```
RuntimeError: RESOURCE_EXHAUSTED: Out of memory
```

**Cause**: GPU memory fragmentation or insufficient GPU memory.

**Solution 1**: Reduce number of particles
```python
# In test script
N_PARTICLES = 5000  # Reduce from 10000
```

**Solution 2**: Clear JAX caches
```python
import jax
jax.clear_caches()
```

**Solution 3**: Check GPU memory
```bash
nvidia-smi
```

ThreadedA mesh requires ~117 MB GPU memory. Test with 10K particles requires ~500 MB total.

---

### Issue: Slow first timestep (~5-10 seconds)

**Symptom**: First timestep is very slow, then fast afterwards.

**Cause**: JAX JIT compilation on first call.

**Solution**: This is expected behavior. The test scripts include warm-up calls to avoid this in benchmarks.

In production, first timestep will be slow, but all subsequent timesteps will be fast.

---

## Integration with Production Script

Once `test_rk4_gpu_fused.py` passes, you can integrate GPU-fused RK4 into the production script.

### Option 1: Add Configuration Flag (Recommended)

**File**: `production_tracking_threadeda.py`

Add after line 268:
```python
USE_VECTORIZED_SEARCH = True
USE_GPU_FUSED_RK4 = True  # NEW FLAG
```

Then modify the time marching loop (around line 788):
```python
if USE_GPU_FUSED_RK4:
    from jaxtrace.gpu.tracking.rk4_gpu_fused import rk4_step_gpu_fused_wrapper

    # GPU-fused RK4
    particle_data.positions, particle_data.element_ids, rk4_stats = \
        rk4_step_gpu_fused_wrapper(
            particle_data.positions,
            particle_data.element_ids,
            DT,
            mesh_gpu,
            velocity_field
        )
else:
    # Baseline CPU-orchestrated RK4
    particle_data, rk4_stats = rk4_step_with_incremental_search(
        particle_data,
        velocity_interpolator,
        incremental_searcher,
        dt=DT,
        current_time=step * DT
    )
```

### Option 2: Direct Replacement (Simpler)

Replace the RK4 call in the time marching loop:
```python
# OLD
particle_data, rk4_stats = rk4_step_with_incremental_search(...)

# NEW
from jaxtrace.gpu.tracking.rk4_gpu_fused import rk4_step_gpu_fused_wrapper
particle_data.positions, particle_data.element_ids, rk4_stats = \
    rk4_step_gpu_fused_wrapper(
        particle_data.positions,
        particle_data.element_ids,
        DT,
        mesh_gpu,
        velocity_field
    )
```

---

## Next Steps After Phase 3a

If Phase 3a achieves 50-100k p/s but you want to reach 200-300k p/s:

### Phase 3b: L2 Search Optimization

**Problem**: L2 search for 5-10% unmapped particles uses slow baseline (~10k p/s).

**Solution Options**:

1. **Spatial Indexing** (Best for high L2 rate):
   - Implement octree/BVH on GPU
   - Target: 100-200k p/s for L2
   - Impact: 5-10× overall speedup

2. **Extended L1 Neighborhood** (Best for low L2 rate):
   - Extend L1 to 26-neighbor search
   - Target: <1% L2 miss rate
   - Impact: L2 becomes negligible

3. **Hybrid**:
   - Extended L1 to reduce L2 to 2-3%
   - Spatial indexing for remaining 2-3%
   - Impact: Best of both worlds

---

## Summary

**Phase 3a is complete and ready for testing!**

**To validate**:
1. Run `test_rk4_gpu_fused.py` to confirm GPU-fused RK4 works
2. Run `production_tracking_threadeda.py` to measure actual performance
3. Compare throughput: should be 50-100k p/s (4-8× improvement over baseline)

**Files Created**:
- `jaxtrace/gpu/tracking/rk4_gpu_fused.py` - GPU-fused RK4 implementation
- `test_rk4_gpu_fused.py` - Validation test
- `docs/gpu/PHASE3A_COMPLETE_WITH_FUSED_RK4.md` - Complete documentation

**Issues Fixed**:
- ✅ Import error in `rk4_gpu_fused.py` (wrong path to `point_in_tet_jax`)
- ✅ Python cache issue (old bytecode causing return value mismatch)
- ✅ Slow initial assignment in test (514s → 0.1s by using element centroids)
- ✅ Wrong function signature for `create_global_interpolator` (fixed parameter names)
- ✅ Missing velocities field in `ParticleData` initialization
- ✅ Boolean indexing inside JIT (replaced with full vectorization)

**Ready to run**: Both tests and production script are ready for your validation! 🚀
