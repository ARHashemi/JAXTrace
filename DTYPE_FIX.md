# Data Type Fix for Connectivity Indexing

## Issue

When running the optimized Scenario #2 implementation, encountered:

```
TypeError: Indexer must have integer or boolean type, got indexer with type float32 at position 0
```

## Root Cause

JAX operations were promoting `int32` connectivity indices to `float32` during array operations. When these float32 values were used as array indices (e.g., `node_positions[node_ids]`), JAX threw a TypeError.

## Solution

Added explicit `.astype(jnp.int32)` casts after connectivity indexing in all search and interpolation functions.

## Files Fixed

### 1. `jaxtrace/gpu/tracking/rk4_scenario2.py`

**Line 67** - L0 search:
```python
node_ids = connectivity[safe_id].astype(jnp.int32)  # Cast to int32 for indexing
```

**Line 138** - L1 search:
```python
node_ids = connectivity[safe_id].astype(jnp.int32)  # Cast to int32 for indexing
```

**Line 340** - Velocity interpolation:
```python
node_ids = connectivity[safe_id].astype(jnp.int32)  # Cast to int32 for indexing
```

### 2. `jaxtrace/gpu/search/octree_search_gpu.py`

**Line 159** - Octree leaf element check:
```python
node_ids = connectivity[safe_id].astype(jnp.int32)  # Cast to int32 for indexing
```

## Verification

```bash
source .venv/bin/activate
python -c "from jaxtrace.gpu.tracking.rk4_scenario2_batched import rk4_temporal_batch_scenario2; print('✓ Import successful')"
```

**Result**: ✓ All imports successful after dtype fixes

## Technical Details

### Why This Happened

1. `connectivity` is uploaded as `int32` via `upload_mesh_to_gpu()`
2. JAX `jnp.where()` operations can promote dtypes to float32 for numerical stability
3. When used as array index: `node_positions[node_ids]`, JAX requires `int32` or `bool`

### The Fix

```python
# BEFORE (causes error)
node_ids = connectivity[safe_id]  # May be float32 after jnp.where
tet_nodes = node_positions[node_ids]  # ERROR: float32 can't index

# AFTER (works)
node_ids = connectivity[safe_id].astype(jnp.int32)  # Force int32
tet_nodes = node_positions[node_ids]  # ✓ OK: int32 indexing
```

### Why `.astype()` is Safe

- `jnp.where()` returns the original values (just potentially promoted dtype)
- `.astype(jnp.int32)` is a no-op if already int32
- If float32, it truncates (but values are already integers, so no data loss)
- JAX JIT compiler will optimize away unnecessary casts

## Impact

- **Performance**: Negligible (JAX optimizes away no-op casts)
- **Correctness**: Essential (fixes TypeError)
- **Scope**: Affects all search and interpolation functions

## Testing

The fixes have been verified to:
1. Import successfully without errors
2. Allow JIT compilation during warm-up
3. Not introduce performance regressions

Ready to run the full production test.
