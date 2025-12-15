# CRITICAL: L2 Block Morton Memory Issue

**Status**: BLOCKED - Cannot proceed with current architecture
**Date**: 2025-12-11

## Problem

The L2 block Morton search causes a 4.88 TiB memory allocation during JIT compilation, resulting in OOM error.

```
W1211 13:14:22.745451 3596744 bfc_allocator.cc:501] Allocator (GPU_0_bfc) ran out of memory
trying to allocate 4.88TiB (rounded to 5361560685824)requested by op
```

## Root Cause

The L2 Morton search function accesses large mesh arrays inside a vmapped function:

```python
# In search_block_morton_single_particle() - called inside vmap
node_ids = connectivity[safe_elem_id].astype(jnp.int32)  # connectivity: (3.5M, 4)
tet_nodes = node_positions[node_ids]                      # node_positions: (900k, 3)
```

When this is vmapped over all particles (81k particles), JAX's tracer tries to materialize all possible array accesses, leading to massive memory allocation.

## Why This Happens

1. **Dynamic indexing in vmap**: `connectivity[safe_elem_id]` where `safe_elem_id` is computed dynamically
2. **Large arrays**: 3.5M elements × 4 nodes × 4 bytes = 53 MB connectivity array
3. **JAX tracer conservativeness**: JAX assumes worst-case and tries to allocate space for all possible accesses
4. **Nested indexing**: Double indirection (element → nodes → positions) amplifies the problem

## Attempted Fixes

### Fix 1: Closure-based mesh capture (FAILED)
- Captured mesh arrays in closure instead of passing as arguments
- **Result**: Same OOM error - JAX still materializes arrays during vmap

### Fix 2: Move JIT outside wrapper (FAILED)
- Moved `@jax.jit` decorator outside the wrapper function
- **Result**: Same OOM error - problem is in vmap, not JIT placement

## The Fundamental Issue

The per-block Morton architecture requires accessing **global mesh arrays** with **particle-dependent indices** inside a vmap. This is fundamentally incompatible with JAX's static shape requirements.

## Alternative Approaches

### Option 1: Disable L2, Use Only L0+L1 (3-hop)
- **Pros**: Works reliably, 40-48k p/s, no memory issues
- **Cons**: Lower retention (~60% vs target 80%)
- **Status**: ✅ Tested and working

### Option 2: Use Smaller Mesh Subset Per Block
- Idea: Pre-compute per-block mesh subsets (block-local connectivity, node_positions)
- **Problem**: Would need to remap node IDs → massive preprocessing overhead
- **Memory**: Would still be large (~256 blocks × ~50 elements × 4 nodes × 3 coords)

### Option 3: Move L2 Search to CPU
- Idea: Do L2 search on CPU for particles that miss L0+L1
- **Problem**: CPU-GPU sync kills throughput (currently 40-48k p/s)
- **Not viable** for production

### Option 4: Hierarchical 5-hop Search (No L2)
- Use deeper L1 neighbor search (5 hops instead of 3)
- **Tested**: Achieved 91% retention (vs 60% for 3-hop)
- **Throughput**: Same 40-48k p/s (L1 is cheap)
- **Status**: ✅ Already implemented and working

### Option 5: Per-Block Local Coordinate System (Complex)
- Transform all mesh data to block-local coordinates
- Pad each block with fixed-size arrays
- **Problem**: Requires extensive preprocessing and memory overhead
- **Estimate**: 256 blocks × 50 elements × (connectivity + positions) ≈ 200 MB
- **Complexity**: High - needs complete mesh transformation

## Recommended Path Forward

**Immediate**: Use hierarchical 5-hop L1 search (no L2)
- Already tested: 91% retention at 2,500 steps
- No memory issues
- Same 40-48k p/s throughput
- Production-ready

**Future**: Investigate JAX compilation flags or alternative libraries
- Try `jax.checkpoint` to control materialization
- Investigate XLA compilation options
- Consider alternative JAX-compatible approaches

## Status: Phase 2 Assessment

**Phase 2 L2 Block Morton**: ❌ BLOCKED by JAX memory issue

**Phase 3 L3 Neighbor Fallback**: ⏸️  ON HOLD (depends on L2 working)

**Alternative - Hierarchical 5-hop**: ✅ WORKING (91% retention)

## Conclusion

The L2 block Morton architecture is theoretically sound but practically blocked by JAX's vmap memory behavior. The hierarchical 5-hop search is the pragmatic solution that achieves good retention without L2.

**Action**: Abandon L2 Morton, document as "theoretically correct but JAX-incompatible", proceed with hierarchical 5-hop production testing.
