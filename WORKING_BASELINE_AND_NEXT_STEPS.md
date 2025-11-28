# Working Baseline: GPU-Fused RK4 + Next Steps

## Current Status: ✅ Working Baseline Committed

**Commit**: `9f74afa` - "Phase 3a Part 2: GPU-fused RK4 with argument order fix and 2-hop L1 search"

This establishes a stable, working baseline with:
- ✅ GPU-fused RK4 (all 4 stages on GPU, no intermediate transfers)
- ✅ 2-hop L1 neighbor search (~20 neighbors per element)
- ✅ Stable execution (no errors, 88% GPU utilization)
- ✅ Good performance (640k p/s initial, 117k p/s final)
- ⚠️ **Particle loss issue**: 83.8% loss (10,016/61,819 final particles)

---

## Performance Metrics (Current Baseline)

| Metric | Value | Status |
|--------|-------|--------|
| Initial throughput | 640,000 p/s | ✅ Excellent |
| Final throughput | 117,000 p/s | ✅ Good |
| GPU utilization | 88% | ✅ Excellent |
| Particle retention | 16.2% | ❌ **Major issue** |
| Memory usage | ~500 MB GPU | ✅ Low |
| Stability | No errors | ✅ Perfect |

---

## Root Cause of Particle Loss

**Problem**: 2-hop L1 search creates neighborhoods of ~20 elements, giving 95-98% hit rate per timestep.

**Math**:
```
Hit rate per step: 97%
Over 2,500 steps: 0.97^2500 = 0.0% (essentially all particles lost)
Observed retention: 16.2%
```

The compounding effect of 2-3% misses per timestep causes catastrophic particle loss.

---

## Options to Fix Particle Loss

### Option 1: Increase L1 Hop Count (Pure GPU)

**Approach**: Increase `RK4_L1_HOP_COUNT` from 2 to 3 or 4

| Hop Count | Neighbors | Hit Rate | Expected Retention | Throughput | GPU Memory |
|-----------|-----------|----------|-------------------|------------|------------|
| 2 (current) | ~20 | 95-98% | 16% ✗ | 117k p/s | 500 MB |
| 3 | ~84 | 98-99.5% | 60-80% | 80-120k p/s | ~2 GB |
| 4 | ~340 | 99.5-99.9% | 90-98% ✓ | 60-80k p/s | **3.5+ GB ✗** |

**Issue with 4-hop**: Out of memory during JIT compilation (~3.8 GB required)

```
W1126 13:51:56.981430 bfc_allocator.cc:512] Allocator (GPU_0_bfc) ran out of memory
trying to allocate 3.55GiB
```

**Pros**:
- Pure GPU implementation (no CPU-GPU transfers)
- Simpler code (no fallback logic)
- Best performance when it fits in memory

**Cons**:
- 4-hop doesn't fit in GPU memory
- 3-hop might be marginal (60-80% retention)

**Implementation**: Change one line in `production_tracking_threadeda.py`:
```python
RK4_L1_HOP_COUNT = 3  # Try 3-hop first
```

---

### Option 2: Add CPU L2/L3 Fallback

**Approach**: Keep 2-hop L1 GPU search, add CPU fallback for misses

**Architecture**:
1. Run 2-hop L1 search on GPU (~97% hit rate, fast)
2. For particles that miss (3%), transfer to CPU
3. Run expensive L2/L3 search on CPU (~99.9% hit rate)
4. Transfer back to GPU

**Expected Performance**:
```
Per timestep:
- 97% particles: GPU-only (very fast)
- 3% particles: CPU fallback (slow, but rare)
- Overall throughput: ~150-200k p/s (estimated)
- Particle retention: 98-99% ✓
```

**Pros**:
- High particle retention (98-99%)
- Doesn't require more GPU memory
- Falls back gracefully for difficult cases

**Cons**:
- More complex code (CPU-GPU synchronization)
- Small performance penalty (3% of particles need CPU transfer)
- Slightly lower throughput than pure GPU

**Implementation Complexity**: Medium
- Modify `rk4_gpu_fused.py` to detect misses
- Add CPU search fallback in `production_tracking_threadeda.py`
- Requires careful synchronization

---

### Option 3: Memory Optimization for 4-Hop GPU

**Approach**: Optimize GPU memory usage to fit 4-hop L1 search

**Strategies**:
1. **Gradient checkpointing**: Recompute intermediate values instead of storing
2. **Reduce JIT compilation memory**: Use `jax.jit(donate_argnums=...)` to reuse buffers
3. **Optimize neighbor storage**: Use sparse representation instead of dense arrays
4. **Stream processing**: Process particles in smaller batches

**Expected Result**:
- 4-hop L1 fits in GPU memory
- 90-98% particle retention ✓
- 60-80k p/s throughput
- Pure GPU implementation ✓

**Pros**:
- Best retention (90-98%)
- Pure GPU (no CPU fallback)
- Clean architecture

**Cons**:
- Requires significant code changes
- May be challenging to optimize sufficiently
- Lower throughput than 2-hop

**Implementation Complexity**: High
- Requires deep JAX/XLA knowledge
- May need to restructure search algorithm
- Iterative optimization process

---

### Option 4: Hybrid 3-Hop + CPU Fallback

**Approach**: Use 3-hop L1 (fits in memory) + CPU fallback for remaining misses

**Architecture**:
1. Run 3-hop L1 search on GPU (~99% hit rate)
2. CPU fallback for remaining 1% misses

**Expected Performance**:
```
Per timestep:
- 99% particles: GPU-only with 3-hop
- 1% particles: CPU fallback
- Overall throughput: ~100-120k p/s
- Particle retention: 98-99% ✓
```

**Pros**:
- High retention (98-99%)
- Lower CPU overhead than 2-hop + fallback
- More robust than pure 3-hop

**Cons**:
- Still requires CPU fallback code
- Slightly more complex than pure GPU

**Implementation Complexity**: Medium
- Same as Option 2, but less frequent fallback

---

## Recommended Approach

### **Recommendation: Option 4 (Hybrid 3-Hop + CPU Fallback)**

**Rationale**:
1. **Fits in GPU memory**: 3-hop uses ~2 GB (safe)
2. **High hit rate**: 99% on GPU means minimal CPU overhead
3. **Robust**: CPU fallback catches edge cases
4. **Good performance**: 100-120k p/s throughput
5. **High retention**: 98-99% particles retained

**Implementation Steps**:

1. **Step 1**: Change hop count to 3
   ```python
   RK4_L1_HOP_COUNT = 3
   ```

2. **Step 2**: Test 3-hop alone
   - Run production script
   - Check retention (expect 60-80%)
   - Check memory usage (expect ~2 GB)

3. **Step 3**: Add CPU L2 fallback
   - Detect L1 misses in RK4
   - Transfer missed particles to CPU
   - Run L2 search on CPU
   - Transfer back to GPU

4. **Step 4**: Validate
   - Run full 2,500 timestep simulation
   - Target: 98-99% retention (55k-60k final particles)
   - Target: 100-120k p/s throughput

---

## Alternative Quick Test: 3-Hop Alone

**Before implementing fallback**, try 3-hop alone:

```bash
# Edit production_tracking_threadeda.py
RK4_L1_HOP_COUNT = 3

# Run
python3 production_tracking_threadeda.py 2>&1 | tee logs/production_3hop_test.log
```

**Expected results**:
- Memory usage: ~2 GB (should fit)
- Retention: 60-80% (better than 16%, but not great)
- Throughput: 80-120k p/s

If 3-hop gives >80% retention, you might not need CPU fallback at all!

---

## Summary

| Option | Retention | Throughput | GPU Memory | Complexity | Status |
|--------|-----------|------------|------------|------------|--------|
| 1. 4-hop GPU | 90-98% ✓ | 60-80k | **>3.5 GB ✗** | Low | **Out of memory** |
| 2. 2-hop + CPU | 98-99% ✓ | 150-200k | 500 MB | Medium | Viable |
| 3. Optimize 4-hop | 90-98% ✓ | 60-80k | 2-3 GB? | **High** | Uncertain |
| **4. 3-hop + CPU** | **98-99% ✓** | **100-120k** | **~2 GB ✓** | **Medium** | **Recommended** |

---

## Next Session Action Items

1. **Test 3-hop alone** (5 minutes):
   - Change `RK4_L1_HOP_COUNT = 3`
   - Run production script
   - Check memory usage and retention

2. **If 3-hop retention < 80%**: Implement CPU L2 fallback

3. **If 3-hop retention > 80%**: Consider it good enough, or add fallback for robustness

---

## Files to Modify (for CPU fallback)

1. **jaxtrace/gpu/tracking/rk4_gpu_fused.py**:
   - Add miss detection in search functions
   - Return mask of successful/failed searches

2. **production_tracking_threadeda.py**:
   - Add CPU L2 search import
   - Implement fallback logic after GPU RK4
   - Transfer missed particles, search on CPU, merge back

---

## Current Baseline is Stable ✓

The commit `9f74afa` provides a solid foundation:
- Clean, working GPU-fused RK4 implementation
- Correct argument order in all 4 stages
- Configurable hop count
- Good performance and stability

All future work can branch from this known-good state.
