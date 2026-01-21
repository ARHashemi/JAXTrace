# L2 Benchmark Script - Critical API Fix

**Date**: 2026-01-19
**Status**: ✅ Fixed - Ready to run

---

## Issue: TypeError in create_rk4_fully_fused_timedep()

**Error Message**:
```
TypeError: create_rk4_fully_fused_timedep() got an unexpected keyword argument 'mesh_gpu'
```

**Root Cause**: Incorrect function signature - the benchmark was passing wrong parameters to `create_rk4_fully_fused_timedep()`.

---

## What Was Wrong

### Problem 1: Wrong Function Parameters

**Incorrect code** (lines 141-151):
```python
rk4_step = create_rk4_fully_fused_timedep(
    mesh_gpu=mesh_gpu,                          # WRONG: No such parameter
    mesh_gpu_global_morton=mesh_gpu_octree,
    velocity_sequence=velocity_sequence_gpu,     # WRONG: Not a parameter
    dt=DT,                                       # WRONG: Not a parameter
    point_in_tet_method=POINT_IN_TET_METHOD     # WRONG: Not a parameter
)
```

**Correct signature** (from [jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py:32-43](jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py#L32-L43)):
```python
def create_rk4_fully_fused_timedep(
    mesh_gpu_connectivity: jax.Array,           # Individual component
    mesh_gpu_node_positions: jax.Array,         # Individual component
    mesh_gpu_element_neighbors: jax.Array,      # Individual component
    mesh_gpu_element_volumes: jax.Array,        # MISSING in benchmark
    mesh_gpu_global_morton: MeshGPUGlobalMorton,
    n_hops: int = 3,
    l2_search_radius: int = 2,
    enable_l1_search: bool = True,
    l2_search_method: str = 'radius',
    l2_incremental_radii: tuple = (2, 5, 10)
):
```

**Key differences**:
1. Function takes **individual mesh components** (connectivity, positions, neighbors), not `mesh_gpu` object
2. Requires `mesh_gpu_element_volumes` parameter (was missing)
3. Does NOT take `velocity_sequence`, `dt`, or `point_in_tet_method` parameters
4. Returns a **function** that takes `(positions, elements, dt, velocity_fields, time_idx)`

### Problem 2: Wrong RK4 Step Call

**Incorrect code** (lines 204-214):
```python
state = {
    'positions': positions_gpu,
    'elements': element_ids_gpu,
    'active': element_ids_gpu >= 0,
    't_index': jnp.int32(0)
}

# Call with state dict
state = rk4_step(state)
```

**Correct pattern** (from [production_tracking_fully_fused_timedep.py:843-849](production_tracking_fully_fused_timedep.py#L843-L849)):
```python
# Call with individual arguments
positions_gpu, element_ids_gpu = rk4_step(
    positions_gpu,           # positions array
    element_ids_gpu,         # element IDs array
    DT,                      # timestep size
    velocity_fields_gpu,     # velocity sequence
    time_idx                 # current time index (for cyclic indexing)
)
```

**Key differences**:
1. RK4 step takes **5 positional arguments**, not a state dict
2. Returns **tuple** `(positions, element_ids)`, not state dict

### Problem 3: Missing element_volumes

The benchmark didn't compute or upload `element_volumes`, which is required for adaptive L1 hop count.

---

## All Changes Applied

### Change 1: Compute Element Volumes

**Location**: [benchmark_l2_search_methods.py:285-299](benchmark_l2_search_methods.py#L285-L299)

**Added**:
```python
# Compute element volumes (needed for adaptive L1 hop count)
v0 = node_positions[connectivity[:, 0]]
v1 = node_positions[connectivity[:, 1]]
v2 = node_positions[connectivity[:, 2]]
v3 = node_positions[connectivity[:, 3]]
e1 = v1 - v0
e2 = v2 - v0
e3 = v3 - v0
cross_e2_e3 = np.cross(e2, e3)
det = np.sum(e1 * cross_e2_e3, axis=1)
element_volumes = np.abs(det) / 6.0
```

### Change 2: Upload Element Volumes to GPU

**Location**: [benchmark_l2_search_methods.py:326](benchmark_l2_search_methods.py#L326)

**Added**:
```python
element_volumes_gpu = jax.device_put(element_volumes.astype(np.float32))
```

### Change 3: Fix create_rk4_fully_fused_timedep Calls

**Location**: [benchmark_l2_search_methods.py:141-195](benchmark_l2_search_methods.py#L141-L195)

**Before**:
```python
rk4_step = create_rk4_fully_fused_timedep(
    mesh_gpu=mesh_gpu,
    mesh_gpu_global_morton=mesh_gpu_octree,
    velocity_sequence=velocity_sequence_gpu,
    dt=DT,
    l2_search_method='radius',
    l2_search_radius=l2_radius,
    point_in_tet_method=POINT_IN_TET_METHOD
)
```

**After**:
```python
rk4_step = create_rk4_fully_fused_timedep(
    mesh_gpu_connectivity=mesh_gpu.connectivity,
    mesh_gpu_node_positions=mesh_gpu.node_positions,
    mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
    mesh_gpu_element_volumes=element_volumes_gpu,
    mesh_gpu_global_morton=mesh_gpu_octree,
    n_hops=N_HOPS,
    l2_search_radius=l2_radius,
    enable_l1_search=ENABLE_L1_SEARCH,
    l2_search_method='radius'
)
```

Applied to all 4 L2 methods: `radius`, `incremental`, `neighbors`, `hierarchical`.

### Change 4: Fix RK4 Step Calls

**Location**: [benchmark_l2_search_methods.py:204-223](benchmark_l2_search_methods.py#L204-L223)

**Before**:
```python
state = {
    'positions': positions_gpu,
    'elements': element_ids_gpu,
    'active': element_ids_gpu >= 0,
    't_index': jnp.int32(0)
}

_ = rk4_step(state)  # Warmup

for step in range(n_steps):
    state = rk4_step(state)
    state = jax.tree_util.tree_map(jax.block_until_ready, state)

n_active_final = int(jnp.sum(state['active']))
```

**After**:
```python
# Warmup (compile)
positions_gpu, element_ids_gpu = rk4_step(
    positions_gpu,
    element_ids_gpu,
    DT,
    velocity_sequence_gpu,
    0  # time_idx
)
positions_gpu = jax.block_until_ready(positions_gpu)
element_ids_gpu = jax.block_until_ready(element_ids_gpu)

# Run tracking
for step in range(n_steps):
    positions_gpu, element_ids_gpu = rk4_step(
        positions_gpu,
        element_ids_gpu,
        DT,
        velocity_sequence_gpu,
        step  # time_idx
    )
    # Block only occasionally for efficiency
    if step % 10 == 0 or step == n_steps - 1:
        positions_gpu = jax.block_until_ready(positions_gpu)
        element_ids_gpu = jax.block_until_ready(element_ids_gpu)

# Final sync
positions_gpu = jax.block_until_ready(positions_gpu)
element_ids_gpu = jax.block_until_ready(element_ids_gpu)

n_active_final = int(jnp.sum(element_ids_gpu >= 0))
```

### Change 5: Update Function Signature and Return Values

**Location**: [benchmark_l2_search_methods.py:127](benchmark_l2_search_methods.py#L127)

**Before**:
```python
def run_rk4_tracking(positions_gpu, element_ids_gpu, mesh_gpu, mesh_gpu_octree, velocity_sequence_gpu,
                     l2_method, l2_radius=None, incremental_radii=None, n_steps=100):
    # ...
    return state, n_active_final, retention, t_elapsed, throughput
```

**After**:
```python
def run_rk4_tracking(positions_gpu, element_ids_gpu, mesh_gpu, mesh_gpu_octree,
                     element_volumes_gpu, velocity_sequence_gpu,
                     l2_method, l2_radius=None, incremental_radii=None, n_steps=100):
    # ...
    return positions_gpu, element_ids_gpu, n_active_final, retention, t_elapsed, throughput
```

### Change 6: Update Function Call in main()

**Location**: [benchmark_l2_search_methods.py:495-506](benchmark_l2_search_methods.py#L495-L506)

**Before**:
```python
state, n_active_final, retention, t_elapsed, throughput = run_rk4_tracking(
    positions_gpu,
    element_ids_initial,
    mesh_gpu,
    mesh_gpu_octree,
    velocity_sequence_gpu,
    # ...
)

tracking_results[name] = {
    'state': state,
    # ...
}
```

**After**:
```python
positions_final, element_ids_final, n_active_final, retention, t_elapsed, throughput = run_rk4_tracking(
    positions_gpu,
    element_ids_initial,
    mesh_gpu,
    mesh_gpu_octree,
    element_volumes_gpu,  # NEW parameter
    velocity_sequence_gpu,
    # ...
)

tracking_results[name] = {
    'positions': positions_final,
    'element_ids': element_ids_final,
    # ...
}
```

---

## Reference: Production Code Pattern

**From**: [production_tracking_fully_fused_timedep.py:716-727](production_tracking_fully_fused_timedep.py#L716-L727)

```python
# Create fully-fused time-dependent RK4 step function
rk4_step = create_rk4_fully_fused_timedep(
    mesh_gpu_connectivity=mesh_gpu.connectivity,
    mesh_gpu_node_positions=mesh_gpu.node_positions,
    mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
    mesh_gpu_element_volumes=element_volumes_gpu,
    mesh_gpu_global_morton=mesh_gpu_octree,
    n_hops=N_HOPS,
    l2_search_radius=L2_SEARCH_RADIUS,
    enable_l1_search=ENABLE_L1_SEARCH,
    l2_search_method=L2_SEARCH_METHOD,
    l2_incremental_radii=INCREMENTAL_SEARCH_RADII
)

# Usage in loop
for step in range(1, N_STEPS + 1):
    positions_gpu, element_ids_gpu = rk4_step(
        positions_gpu,
        element_ids_gpu,
        DT,
        velocity_fields_gpu,
        step  # time_idx
    )
```

---

## Verification

### Files Modified

✅ [benchmark_l2_search_methods.py](benchmark_l2_search_methods.py)
- Lines 127-223: Fixed `run_rk4_tracking` function
- Lines 285-299: Added element_volumes computation
- Line 326: Added element_volumes GPU upload
- Lines 495-515: Fixed function call and results storage

### Testing

**To verify the fix works**:
```bash
python benchmark_l2_search_methods.py 2>&1 | tee logs/benchmark_l2_search.log
```

**Expected behavior**:
- No `TypeError` about unexpected keyword arguments
- RK4 compilation succeeds
- Tracking runs for 100 steps
- Results show retention and throughput metrics

---

## Impact

### What's Fixed

✅ **Function signature**: Passes correct parameters to `create_rk4_fully_fused_timedep`
✅ **Element volumes**: Computes and uploads required element_volumes array
✅ **RK4 calls**: Uses correct calling pattern with 5 arguments
✅ **Return values**: Properly unpacks tuple return values

### Expected Performance

The benchmark should now successfully:
1. Create RK4 step functions for all 6 L2 configurations
2. Compile each configuration (1-5 minutes per config)
3. Run 100 RK4 steps for each configuration
4. Report retention and throughput metrics

**Total runtime**: ~30-45 minutes for all 6 configurations

---

## Summary

The L2 benchmark had an incorrect understanding of the `create_rk4_fully_fused_timedep` API:
- Expected it to take `mesh_gpu` object → Actually takes individual components
- Expected it to take `velocity_sequence` → Actually not a parameter (passed to returned function)
- Expected returned function to take state dict → Actually takes 5 positional arguments
- Missing `element_volumes` computation and upload

All issues have been fixed by following the production script pattern exactly.

**Status**: ✅ Ready to run

See [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) for complete usage instructions.
