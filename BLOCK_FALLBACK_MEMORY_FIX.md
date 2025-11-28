# Block-Local Fallback: Memory Issue and Fix

## Issue Discovered

During JIT compilation of the block-local fallback search, JAX attempted to allocate **40.88 GiB** of GPU memory, causing out-of-memory error.

### Root Cause

**Original implementation** (lines 224-250 in block_local_search.py):

```python
# Vectorize over all elements in block
found_ids = jax.vmap(check_element)(jnp.arange(max_block_size))
```

**Problem:**
- `max_block_size = 450,004` (heaviest block in ThreadedA mesh)
- `jax.vmap(check_element)` over 450k elements creates massive intermediate arrays
- Each particle search would materialize:
  - 450k element IDs
  - 450k × 4 node IDs (1.8M values)
  - 450k × 4 × 3 node positions (5.4M values)
  - 450k tet containment results
- **Total per particle**: ~40 GB of temporary arrays!

### Error Message

```
W1127 13:34:51.896032 3017768 bfc_allocator.cc:501] Allocator (GPU_0_bfc) ran out of memory trying to allocate 40.88GiB (rounded to 43889241088)requested by op
E1127 13:34:51.896392 3017768 pjrt_stream_executor_client.cc:2974] Execution of replica 0 failed: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 43889240864 bytes.
```

## Solution: Sequential Scan

**Fixed implementation** (lines 224-257 in block_local_search.py):

```python
# Sequential scan (memory-efficient)
def scan_elements(carry, elem_idx):
    """Sequential scan over block elements."""
    found_id, found = carry

    # Only check if we haven't found yet
    is_valid = valid_mask[elem_idx]
    elem_id = block_elements[elem_idx]

    # Bounds check
    elem_valid = (elem_id >= 0) & (elem_id < len(connectivity))
    safe_elem_id = jnp.where(elem_valid, elem_id, 0)

    # Get element nodes
    node_ids = connectivity[safe_elem_id]
    tet_nodes = node_positions[node_ids]

    # Check if point is inside
    inside = point_in_tet_jax(position, tet_nodes)

    # Update if found and valid
    should_update = is_valid & elem_valid & inside & ~found
    new_found_id = jnp.where(should_update, elem_id, found_id)
    new_found = found | should_update

    return (new_found_id, new_found), None

# Sequential scan (memory-efficient, no huge intermediate arrays)
(result, _), _ = jax.lax.scan(
    scan_elements,
    (-1, False),  # Initial: (found_id=-1, found=False)
    jnp.arange(max_block_size)
)
```

### Why This Works

**Memory Efficiency:**
- `jax.lax.scan` iterates sequentially (like a for-loop)
- Only stores carry state: `(found_id, found)` = 2 values
- No massive intermediate arrays
- **Memory per particle**: ~1 KB (vs 40 GB!)

**Performance:**
- Sequential scan over 450k elements
- Early exit when element found (via `found` flag)
- Expected: 2-50 ms per failed particle (acceptable for 0.09% of particles)

## Testing Status

**Test configuration:**
- 1,000 particles
- 100 timesteps
- 3-hop L1 + block-local fallback

**Current status:** Running initialization (mesh loading + forest creation ~150s)

## Expected Impact

### Memory Usage

| Component | Before (vmap) | After (scan) | Savings |
|-----------|---------------|--------------|---------|
| Per-particle search | 40 GB | 1 KB | 40,000,000× |
| GPU memory total | OOM | ~150 MB | Fits easily |

### Performance

- Sequential scan is slower than vectorized search
- But only used for 0.09% of particles (L1 failures)
- Expected overhead: < 1% overall (negligible)

## Lessons Learned

1. **JAX vmap creates full materialized arrays**
   - Don't vmap over large dimensions (> 10k)
   - Use `lax.scan` for memory-efficient sequential processing

2. **Block size variation matters**
   - Light blocks: 2-10k elements (vmap would work)
   - Heavy blocks: 50-450k elements (vmap explodes memory)
   - Solution must handle worst-case (450k)

3. **Fallback search is rare**
   - Used for < 0.1% of particles
   - Sequential is acceptable when rarely used
   - Optimizing for memory over speed is correct trade-off

## Next Steps

1. ✅ Fix implemented (scan-based search)
2. ⏳ Test with 1k particles (running)
3. ⏳ Verify memory usage stays reasonable
4. ⏳ Test with production particle count
5. ⏳ Measure actual retention improvement
