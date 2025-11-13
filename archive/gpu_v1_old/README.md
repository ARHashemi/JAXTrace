# Old GPU Implementation (Phase 2) - Archived

**Date Archived**: 2025-11-03
**Status**: Failed performance requirements - 50,000× slower than CPU

## What's in this archive

This directory contains the Phase 2 GPU implementation that was found to be incompatible with JAX design principles and catastrophically slow on production meshes.

### Files:
- `kernels_v2.py` - Attempted GPU kernels using hierarchical search
- `tracker.py` - GPUParticleTracker class with block element lists
- `search.py` - Multi-level search implementation
- `test_kernels.py` - Unit tests (all passing but not performant)
- `test_search.py` - Search tests

## Why it failed

### Performance Results (ThreadedA mesh, 3.5M elements)

| Particles | CPU Time | GPU Time | Speedup | Status |
|-----------|----------|----------|---------|--------|
| 100       | 0.08 ms  | 2,804 ms | 0.00003× | CPU 34,000× faster |
| 1,000     | 0.56 ms  | 28,671 ms | 0.00002× | CPU 50,000× faster |
| 10,000    | 6.1 ms   | **OOM**  | N/A     | Out of memory (15.7 GB) |

### Root Causes

1. **Dynamic array creation incompatible with JAX**
   - Used `np.where(element_to_block == block_id)[0]` to get block elements
   - JAX requires static shapes, forced pre-computed padded arrays
   - Result: Massive memory overhead

2. **No early termination**
   - CPU uses `for` loop with `return` on first match
   - JAX's `vmap` processes ALL elements (no early exit)
   - Result: 10× more computation than CPU

3. **Load imbalance**
   - ThreadedA blocks: 36 to 938,236 elements (8.59× imbalance)
   - GPU forced to truncate to 10K max → 31% error rate
   - CPU handles naturally with dynamic arrays

4. **Memory scaling**
   - vmap memory: O(n_particles × max_per_block)
   - 10K particles × 10K elements = 100M checks = 15.7 GB
   - GPU only has 8 GB available

### Design Issues

**Used hierarchical structures:**
```python
@dataclass
class BlockMetadata:
    block_id: int
    elements: List[int]  # Dynamic size - JAX incompatible!
```

**Used filtering instead of indexing:**
```python
# OLD (JAX-incompatible):
block_elements = np.where(element_to_block == block_id)[0]
for elem_id in block_elements:  # Dynamic iteration
    if point_in_element(...):
        return elem_id  # Early exit - JAX incompatible!
```

## What replaced it

**V3 implementation using flat arrays** (see `docs/gpu/GPU_NATIVE_IMPLEMENTATION_PLAN_V3_COMPREHENSIVE.md`)

**Key changes:**
1. All flat, static-size arrays
2. Use indexing: `element_block_IDs[elem_id]` (not filtering)
3. Use masking: `element_block_IDs == block_id` (vectorized)
4. Minimal scan carry (29 bytes/particle vs 57 bytes)

**Expected performance:** 10-100× faster than CPU (not 50,000× slower!)

## Lessons Learned

1. **JAX is not a silver bullet** - Not all algorithms parallelize well
2. **Design for the framework** - Don't force CPU patterns onto GPU
3. **Validate early with production data** - Small test meshes hide issues
4. **Memory is the real bottleneck** - Not just computation
5. **Choose the right abstraction** - Flat arrays > hierarchical objects for JAX

## References

- [CPU_vs_GPU_ALGORITHMS.md](../../docs/gpu/CPU_vs_GPU_ALGORITHMS.md) - Side-by-side comparison
- [PHASE2_GPU_BOTTLENECK_FINDINGS.md](../../docs/gpu/PHASE2_GPU_BOTTLENECK_FINDINGS.md) - Detailed analysis
- [PHASE_2_GPU_ANALYSIS_COMPLETE.md](../../docs/gpu/PHASE_2_GPU_ANALYSIS_COMPLETE.md) - Complete findings
- [GPU_NATIVE_IMPLEMENTATION_PLAN_V3_COMPREHENSIVE.md](../../docs/gpu/GPU_NATIVE_IMPLEMENTATION_PLAN_V3_COMPREHENSIVE.md) - New design

## Do NOT use this code

This implementation is archived for historical reference only. Use the V3 implementation being developed from Phase 0.
