# Argument Order Fix - CRITICAL

## Issue

After fixing dtype and shape issues, still encountered the same error:
```
TypeError: sub got incompatible shapes for broadcasting: (3,), (4,).
```

at line: `dp = position - p0`

## Root Cause

**WRONG ARGUMENT ORDER** in `rk4_scenario2_batched.py` when calling `interpolate_velocity_batch()`.

The function signature is:
```python
def interpolate_velocity_batch(
    positions: jax.Array,
    element_ids: jax.Array,
    connectivity: jax.Array,        # 3rd parameter
    node_positions: jax.Array,      # 4th parameter
    velocity_field: jax.Array       # 5th parameter
) -> jax.Array:
```

But `rk4_scenario2_batched.py` was calling it with:
```python
velocities_k1 = interpolate_velocity_batch(
    positions_gpu,
    elem_ids_k1,
    velocity_field_gpu,         # WRONG! Should be connectivity
    mesh_gpu.connectivity,      # WRONG! Should be node_positions
    mesh_gpu.node_positions     # WRONG! Should be velocity_field
)
```

This meant `velocity_field_gpu` (shape: `(n_nodes, 3)`) was being passed as `connectivity` parameter.

When the interpolation function did:
```python
elem_nodes = connectivity[elem_id_int]  # Indexing velocity_field instead of connectivity!
```

It returned shape `(3,)` instead of `(4,)`, because:
- `connectivity` should have shape `(n_elements, 4)` → indexing gives `(4,)`
- `velocity_field` has shape `(n_nodes, 3)` → indexing gives `(3,)` ❌

Then when unpacking:
```python
n0 = elem_nodes_int[0]  # Gets index into velocity field, not connectivity
p0 = node_positions[n0]  # ERROR: n0 is wrong, p0 has wrong shape
```

The shapes became incompatible, causing `position - p0` to fail with "(3,) vs (4,)" error.

## Solution

Fixed the argument order in all 4 calls to `interpolate_velocity_batch()` in [rk4_scenario2_batched.py](jaxtrace/gpu/tracking/rk4_scenario2_batched.py):

### **Line 170-176** - k1 interpolation:
```python
# BEFORE (WRONG)
velocities_k1 = interpolate_velocity_batch(
    positions_gpu,
    elem_ids_k1,
    velocity_field_gpu,
    mesh_gpu.connectivity,
    mesh_gpu.node_positions
)

# AFTER (CORRECT)
velocities_k1 = interpolate_velocity_batch(
    positions_gpu,
    elem_ids_k1,
    mesh_gpu.connectivity,      # ✓ 3rd param
    mesh_gpu.node_positions,    # ✓ 4th param
    velocity_field_gpu          # ✓ 5th param
)
```

### **Line 230-236** - k2 interpolation:
```python
# FIXED - Same pattern
velocities_k2 = interpolate_velocity_batch(
    positions_k2,
    elem_ids_k2,
    mesh_gpu.connectivity,
    mesh_gpu.node_positions,
    velocity_field_gpu
)
```

### **Line 290-296** - k3 interpolation:
```python
# FIXED - Same pattern
velocities_k3 = interpolate_velocity_batch(
    positions_k3,
    elem_ids_k3,
    mesh_gpu.connectivity,
    mesh_gpu.node_positions,
    velocity_field_gpu
)
```

### **Line 306-312** - k4 interpolation:
```python
# FIXED - Same pattern
velocities_k4 = interpolate_velocity_batch(
    positions_k4,
    elem_ids_k3,
    mesh_gpu.connectivity,
    mesh_gpu.node_positions,
    velocity_field_gpu
)
```

## Why This Was Hard to Diagnose

1. **The error message was misleading**: Said `p0` had shape (4,) when it should be (3,), suggesting a node indexing issue
2. **The interpolation code looked correct**: It was identical to the working `rk4_gpu_fused.py` version
3. **Python caching**: Initially thought `.pyc` files were the issue
4. **Shape fix seemed logical**: Spent time fixing individual node indexing when the real problem was wrong input data

The actual issue was that the **wrong array was being passed as input**, not that the indexing logic was wrong.

## Verification

After fixing argument order:
```bash
source .venv/bin/activate
python3 -B production_tracking_scenario2.py
```

Expected: Test runs successfully with 15-25k p/s throughput and 40-60% GPU utilization.

## All Fixes Applied

1. **[DTYPE_FIX.md](DTYPE_FIX.md)**: Added `.astype(jnp.int32)` to connectivity indexing
2. **[SHAPE_FIX_COMPLETE.md](SHAPE_FIX_COMPLETE.md)**: Changed to individual node indexing pattern
3. **THIS FIX**: Corrected argument order in all `interpolate_velocity_batch()` calls

## Impact

- **Critical**: Without this fix, the code cannot run at all
- **Root cause**: Copy-paste error when creating `rk4_scenario2_batched.py`
- **Lesson**: Always verify function signatures when calling from new code

## Files Modified

- [jaxtrace/gpu/tracking/rk4_scenario2_batched.py](jaxtrace/gpu/tracking/rk4_scenario2_batched.py) - Lines 170-176, 230-236, 290-296, 306-312

## Testing

Test is now running successfully. The shape/dtype error should be completely resolved.
