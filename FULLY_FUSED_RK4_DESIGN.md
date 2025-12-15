# Fully-Fused RK4 Design Document

## Current Architecture Analysis

### Current Implementation Issues

**Problem 1: Separate vmaps for each subprocess**
```python
# Current: rk4_global_morton.py lines 281-361
# Stage 1
element_ids_k1 = search_l0_l1_l2_global_morton(positions_gpu, ...)  # vmap inside
velocities_k1 = interpolate_velocity_batch_gpu(positions_gpu, ...)  # vmap inside

# Stage 2
element_ids_k2 = search_l0_l1_l2_global_morton(positions_k1, ...)   # vmap inside
velocities_k2 = interpolate_velocity_batch_gpu(positions_k1, ...)   # vmap inside
# ... stages 3, 4, final
```

Each function contains its own vmap:
- `search_l0_l1_l2_global_morton`: vmaps over L0, L1, L2 separately
- `interpolate_velocity_batch_gpu`: vmaps over interpolation

This creates **multiple kernel launches per RK4 stage** (5 stages × 2 operations = 10+ kernel launches).

**Problem 2: CPU-GPU transfers at every timestep**
```python
# production_tracking_global_morton.py lines 260-265
positions_gpu = jax.device_put(positions)          # Upload positions
element_ids_gpu = jax.device_put(element_ids)      # Upload element IDs
velocity_field_gpu = jax.device_put(velocity_field) # Upload velocity field

# Lines 379-396: After EVERY timestep
particle_data, stats = rk4_step(...)

# Lines 375-377: Download results EVERY timestep
positions_final = jax.block_until_ready(positions_final)
element_ids_final = jax.block_until_ready(element_ids_final)
```

Transfers happen at **every timestep** even though:
- Velocity field is static (uploaded once, never changes)
- Positions/element_ids stay on GPU between timesteps
- Only need to download for export (every 10 steps, not every step)

## Proposed Fully-Fused Architecture

### Design Goal
**Single vmap over particles** that fuses all RK4 stages into one kernel per timestep.

### New Structure

```python
# Time marching loop (production script)
for step in range(N_STEPS):
    # All computation on GPU, NO transfers
    positions_gpu, element_ids_gpu = rk4_fused_step(
        positions_gpu, element_ids_gpu, dt, ...
    )

    # Download ONLY when needed for export
    if step % EXPORT_FREQUENCY == 0:
        positions_cpu = np.array(positions_gpu)  # Single download
        exporter.enqueue_export(step, positions_cpu, element_ids_gpu)

# Single vmap over particles for entire RK4
@jax.jit
def rk4_fused_step(positions_gpu, element_ids_gpu, dt, ...):
    def rk4_single_particle(pos, elem_id):
        # Stage 1: k1 = f(t, y)
        elem_k1 = search_l0_l1_l2_single(pos, elem_id, ...)
        vel_k1 = interpolate_velocity_single(pos, elem_k1, ...)
        pos_k1 = pos + 0.5 * dt * vel_k1

        # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
        elem_k2 = search_l0_l1_l2_single(pos_k1, elem_k1, ...)
        vel_k2 = interpolate_velocity_single(pos_k1, elem_k2, ...)
        pos_k2 = pos + 0.5 * dt * vel_k2

        # Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
        elem_k3 = search_l0_l1_l2_single(pos_k2, elem_k2, ...)
        vel_k3 = interpolate_velocity_single(pos_k2, elem_k3, ...)
        pos_k3 = pos + dt * vel_k3

        # Stage 4: k4 = f(t + dt, y + dt * k3)
        elem_k4 = search_l0_l1_l2_single(pos_k3, elem_k3, ...)
        vel_k4 = interpolate_velocity_single(pos_k3, elem_k4, ...)

        # Final position
        pos_final = pos + (dt / 6.0) * (vel_k1 + 2*vel_k2 + 2*vel_k3 + vel_k4)

        # Search at final position
        elem_final = search_l0_l1_l2_single(pos_final, elem_k4, ...)

        return pos_final, elem_final

    # SINGLE vmap over all particles
    return jax.vmap(rk4_single_particle)(positions_gpu, element_ids_gpu)
```

### Key Improvements

1. **Single Kernel Launch Per Timestep**
   - All 5 RK4 stages + 5 searches + 4 interpolations fused into one kernel
   - Expected speedup: 2-3× from reduced kernel launch overhead

2. **Minimal CPU-GPU Transfers**
   - Upload velocity field: **once** at initialization
   - Upload positions/element_ids: **once** at initialization
   - Download positions: **only at export frequency** (every 10 steps)
   - Expected speedup: 1.5-2× from eliminated transfer overhead

3. **Persistent GPU Data**
   - Positions and element IDs stay on GPU between timesteps
   - No unnecessary copies or transfers
   - Direct GPU-to-GPU data flow

## Implementation Plan

### Phase 1: Create Single-Particle Functions

**File**: `jaxtrace/gpu/tracking/rk4_fully_fused.py`

1. Extract single-particle search from current vectorized functions:
   ```python
   def search_l0_l1_l2_single_particle(pos, cached_elem_id, ...):
       # L0: point-in-tet test
       # L1: multi-hop neighbor search
       # L2: global Morton search
       return element_id
   ```

2. Extract single-particle interpolation:
   ```python
   def interpolate_velocity_single_particle(pos, elem_id, ...):
       # Barycentric interpolation
       return velocity
   ```

### Phase 2: Create Fully-Fused RK4

1. Implement fused RK4 function:
   ```python
   def create_rk4_fully_fused_global_morton(...):
       @jax.jit
       def rk4_single_particle(pos, elem_id):
           # All 5 stages inline
           return pos_final, elem_final

       def rk4_step_impl(particle_data, velocity_field_gpu, dt, ...):
           # NO uploads (data already on GPU)
           positions_final, element_ids_final = jax.vmap(rk4_single_particle)(
               particle_data.positions, particle_data.element_ids
           )
           return positions_final, element_ids_final

       return rk4_step_impl
   ```

### Phase 3: Update Production Script

**File**: `production_tracking_global_morton.py`

1. Upload data once at initialization:
   ```python
   # Upload mesh data (lines 230-265)
   # Upload velocity field ONCE
   velocity_field_gpu = jax.device_put(velocity_field)

   # Upload particle data ONCE
   positions_gpu = jax.device_put(particle_data.positions)
   element_ids_gpu = jax.device_put(particle_data.element_ids)
   ```

2. Modify time integration loop:
   ```python
   for step in range(1, N_STEPS + 1):
       # All computation on GPU
       positions_gpu, element_ids_gpu = rk4_step(
           positions_gpu, element_ids_gpu, dt, velocity_field_gpu, ...
       )

       # Download ONLY for export
       if step % EXPORT_FREQUENCY == 0:
           positions_cpu = np.array(positions_gpu)
           element_ids_cpu = np.array(element_ids_gpu)
           particle_data = ParticleData(positions_cpu, element_ids_cpu)
           exporter.enqueue_export(step, particle_data)

       # Count active (NO download needed for this)
       n_active = jnp.sum(element_ids_gpu >= 0)
       n_active = int(n_active)  # Single scalar download
   ```

3. Remove per-timestep transfers:
   - Delete upload code inside loop
   - Delete download code (except for export)

## Expected Performance Gains

### Current Performance
- Throughput: ~30-50k particles/second
- Per-timestep overhead: ~2-3ms (kernel launches + transfers)
- Export overhead: Minimal (async queue)

### Expected Performance
- Throughput: **60-120k particles/second** (2-3× improvement)
- Per-timestep overhead: **<1ms** (single kernel launch, no transfers)
- Export overhead: Unchanged (async queue)

### Breakdown
1. **Kernel fusion**: 2-3× speedup from reducing 10+ kernel launches to 1
2. **Transfer elimination**: 1.5-2× speedup from removing per-timestep CPU-GPU copies
3. **Combined**: 3-6× theoretical speedup

## Validation Criteria

1. **Correctness**: Match current implementation results exactly
2. **Performance**: Achieve >2× throughput improvement
3. **Memory**: No increase in GPU memory usage
4. **Retention**: Maintain >95% particle retention at 2,500 steps

## Files to Modify

1. `jaxtrace/gpu/tracking/rk4_fully_fused.py` (NEW)
2. `production_tracking_global_morton.py` (MODIFY)
3. `production_tracking_threadeda.py` (REFERENCE for comparison)

## Testing Strategy

1. Create `test_rk4_fully_fused.py` to validate correctness
2. Run `production_tracking_global_morton.py` with both implementations
3. Compare:
   - Final positions (should match within numerical precision)
   - Particle retention (should be identical)
   - Throughput (should be >2× faster)
   - GPU memory usage (should be similar)

---

**Status**: Design complete, ready for implementation
**Date**: 2025-01-XX
**Next**: Implement Phase 1 - Single-particle functions
