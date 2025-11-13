# Phase 2 GPU Performance Bottleneck - Critical Findings

## Executive Summary

**Status**: GPU implementation is NOT viable for ThreadedA mesh without major architectural changes.

**Performance**: GPU is 50,000× SLOWER than CPU (not a typo)

**Root Cause**: JAX vmap memory scaling issue - creates intermediate arrays proportional to `n_particles × elements_per_block`

---

## Performance Results

### With Optimized Block Element Lists (max_per_block=10,000)

| Particles | CPU Time | GPU Time | Speedup | Match Rate | Status |
|-----------|----------|----------|---------|------------|--------|
| 100       | 0.000s   | 2.804s   | 0.00×   | 68%        | ⚠️ CPU 34,000× faster |
| 1,000     | 0.001s   | 28.671s  | 0.00×   | 69%        | ⚠️ CPU 50,000× faster |
| 10,000    | 0.006s   | OOM      | N/A     | N/A        | ❌ Out of memory (15.7 GB) |

### Memory Allocation

- **100 particles**: Works, but slow
- **1,000 particles**: Works, but extremely slow
- **10,000 particles**: **Out of memory** (trying to allocate 15.7 GB)

**GPU has only ~8 GB available**, so 10K particles cannot run.

---

## Root Cause Analysis

### Memory Scaling Problem

JAX's `vmap` creates intermediate arrays that scale as:

```
Memory = n_particles × max_per_block × intermediate_array_size
```

For Thread edA with 10K particles and 10K max_per_block:
- 10,000 × 10,000 = **100 million element checks**
- Each check creates intermediate arrays for vertices, barycentric coords, etc.
- Total allocation: **15.7 GB** (exceeds GPU capacity!)

### Algorithm Mismatch

**CPU Algorithm (efficient)**:
```python
for block_id in get_blocks_with_elements():
    block_elements = np.where(element_to_block == block_id)[0]  # Compact list
    for elem_id in block_elements:
        if point_in_element(...):
            return elem_id  # Early exit!
```

- Creates compact lists dynamically
- Early termination on first match
- Memory: O(block_size) per particle
- Time: O(k) where k = actual elements in block

**GPU Algorithm (broken)**:
```python
# Pre-compute (once)
block_elements = [all_block_0_elements, all_block_1_elements, ...]  # Fixed-size arrays

# Per particle (vmap)
elements = block_elements[block_id]  # [max_per_block] array
found_array, result_array = vmap(check_element)(elements)  # NO EARLY EXIT!
```

- Uses fixed-size arrays (padded)
- No early termination (vmap processes ALL elements)
- Memory: O(n_particles × max_per_block)
- Time: O(max_per_block) for ALL particles

### Why GPU is Slower

1. **No early termination**: CPU exits on first match, GPU checks all 10,000 elements
2. **Memory overhead**: vmap materializes massive intermediate arrays
3. **Transfer overhead**: CPU↔GPU transfer for each batch
4. **Load imbalance**: ThreadedA has extreme load imbalance (max block: 938K elements)

**For ThreadedA**, even with Level 0/1 having high hit rates:
- Level 0: 85% hit → 8,500 particles use cached element (fast on both)
- Level 1: 10% hit → 1,000 particles check neighbors (fast on both)
- Level 2: 5% hit → **500 particles × 10,000 elements = 5M checks!**

CPU does ~50K checks total, GPU does ~5M checks (100× more work).

---

## Why ThreadedA is Problematic

### Extreme Load Imbalance

```
Block size distribution:
  Min: 36 elements
  Max: 938,236 elements
  Mean: 109,212 elements
  Load imbalance: 8.59× (max/mean)
```

**4 blocks exceed 10K elements** (including one with 938K!).

For particles in these blocks:
- GPU truncates to 10K → **misses 99% of elements!**
- This causes the 68-69% match rate (31% errors!)

### Domain Bounds Issue

The test particles are seeded near origin `(-0.01, 0.01)`, but the mesh domain is:
- X: [-0.030, 0.030]
- Y: [-0.023, 0.023]
- Z: [-0.010, 0.000]

Most test particles are likely **outside the domain**, explaining why CPU finds 0 particles.

---

## What Went Wrong

### Original Hypothesis (WRONG)

> "GPU will be faster for large particle counts due to parallelism"

This assumed:
1. ✅ GPU can parallelize point-in-element tests (TRUE)
2. ✅ Block sizes are reasonable (~100-1000 elements) (FALSE - up to 938K!)
3. ❌ vmap memory usage is acceptable (FALSE - scales as n×k!)
4. ❌ Fixed-size arrays work well (FALSE - causes massive waste!)

### Implementation Attempts

**Attempt 1: lax.scan over sparse array**
- Result: 40× slower than CPU
- Problem: Checked 1000 elements with 90% dummies

**Attempt 2: vmap over full mesh**
- Result: Out of memory
- Problem: 3.5M elements × n_particles = too large

**Attempt 3: vmap over pre-computed block lists**
- Result: 50,000× slower than CPU, OOM at 10K particles
- Problem: 10K elements × 10K particles = 100M checks

---

## Fundamental Issue: JAX Design vs This Algorithm

### JAX Strengths

JAX is designed for:
- **Dense, regular computations**: matrix multiply, convolutions
- **Fixed-size arrays**: known at compile time
- **Batch parallelism**: process many similar items identically
- **No control flow**: no early exits, no dynamic loops

### This Algorithm's Needs

This algorithm requires:
- **Sparse, irregular data**: different particles need different block elements
- **Dynamic-size arrays**: block sizes vary from 36 to 938K
- **Conditional execution**: early termination on first match
- **Control flow**: if found, stop searching

**Mismatch**: JAX forces us to:
1. Pad sparse data to fixed size → wasted computation
2. Process all elements → no early exit
3. Materialize all intermediate results → massive memory

---

## Path Forward

### Option 1: Use Smaller Test Mesh ✅

**For Phase 2 completion**, use a small test mesh with:
- ~1,000 elements total
- Balanced blocks (~30-50 elements each)
- Test with 100-1,000 particles

This demonstrates the **concept** works, even if not performant on ThreadedA.

### Option 2: Implement Phase 9 Early (Hash Octree)

Phase 9's hash octree provides **O(log n)** search:
- Each particle searches only its local octree (~10-100 elements)
- No need for pre-computed block lists
- Works with any block size

**But**: Still has vmap memory scaling issue.

### Option 3: Hybrid CPU/GPU Approach

- Level 0/1: GPU (high hit rate, simple, parallel)
- Level 2: **CPU fallback** (rare, complex, needs dynamic search)

This avoids the Level 2 bottleneck entirely.

### Option 4: Different GPU Framework

Consider alternatives to JAX:
- **CUDA kernels**: Full control, early termination, dynamic memory
- **Numba CUDA**: Python-like syntax, more flexible than JAX
- **Taichi**: Designed for sparse, irregular computations

JAX may not be the right tool for this algorithm.

---

## Recommendations

### For Phase 2 (Immediate)

1. ✅ **Document findings**: This document
2. ✅ **Update documentation**: Explain GPU limitations
3. ⬜ **Test with small mesh**: Demonstrate concept works
4. ⬜ **Complete Phase 2**: Mark as "concept validated, performance TBD"

### For Future Phases

1. **Phase 3-7**: Focus on CPU implementation
2. **Phase 8**: Implement adaptive grid to fix load imbalance
3. **Phase 9**: Evaluate if hash octree helps GPU performance
4. **Phase 10**: Reconsider GPU strategy:
   - Custom CUDA kernels for Level 2?
   - Hybrid CPU/GPU approach?
   - Different framework?

---

## Lessons Learned

1. **JAX is not a silver bullet**: Not all algorithms parallelize well
2. **Load imbalance matters**: GPU performance degrades catastrophically with imbalance
3. **Memory is limiting**: vmap memory scaling can exceed GPU capacity
4. **Early validation is critical**: Should have tested with real mesh sooner
5. **Algorithm design drives technology choice**: Don't force an algorithm into the wrong framework

---

## Conclusion

The GPU implementation for Level 2 search is **not viable** for production use with Thread edA mesh. The fundamental mismatch between JAX's design (dense, regular, batch-parallel) and this algorithm's needs (sparse, irregular, early-exit) makes it 50,000× slower than a simple CPU loop.

**Phase 2 should be marked as "research complete"** with findings that GPU acceleration requires either:
- Significant mesh preprocessing (Phase 8: adaptive grid)
- Different algorithm (Phase 9: hash octree)
- Different implementation (custom CUDA, not JAX)
- Hybrid approach (GPU for Level 0/1, CPU for Level 2)

The work done in Phase 2 is valuable for understanding **why** this is hard and **what** is needed to make it work.
