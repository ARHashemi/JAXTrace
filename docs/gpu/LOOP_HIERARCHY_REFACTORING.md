# Loop Hierarchy Refactoring: CPU-GPU Transfer Optimization

## Current Status

### Priority 1: L0+L1 RK4 Optimization ✅ COMPLETED
- **Achievement**: 4.5× speedup over RK4 Full baseline
- **Performance**: 3 p/s → 13 p/s
- **Hit Rates**: L0: 68%, L1: 19%, L2: 4.5%
- **Status**: Successfully tested and validated

### Priority 2: Fix Tracking Loop Hierarchy 🔄 IN PROGRESS
**Goal**: Restructure loops to match user's specified hierarchy and eliminate block-level CPU-GPU transfers

### Priority 3: Async Data Prefetching ⏳ PENDING

---

## Problem Statement

### Current Architecture (INEFFICIENT)

**Loop Hierarchy:**
```
time_marching.py: march_forward_euler() or rk4_step_with_incremental_search()
  └─> for timestep in timesteps:
        └─> interpolate_velocities(particle_data, velocity_field)
              └─> group_particles_by_block()
              └─> for block_id in blocks:
                    └─> 🔴 CPU→GPU: jax.device_put(block_positions)
                    └─> 🔴 CPU→GPU: jax.device_put(block_element_ids)
                    └─> 🔴 CPU→GPU: jax.device_put(velocity_field[block_id])
                    └─> ✅ GPU: batch_interpolate_velocities()
                    └─> 🔴 GPU→CPU: np.array(velocities)
        └─> integrate_positions(velocities, dt)
        └─> search_new_elements(new_positions)
```

**Issues:**
1. ❌ CPU-GPU transfers occur at block level (inside block loop)
2. ❌ Data returns to CPU after each block
3. ❌ No batching of particles before GPU processing
4. ❌ Excessive data movement overhead (~5-10% of total time)

### Desired Architecture (USER SPECIFIED)

**Loop Hierarchy:**
```
1. Time marching loop (outer)
   2. Particle batches loop (200K particles per batch)
      3. Blocks loop (process blocks on GPU)
```

**User's Requirements (from conversation):**
> "The hierarchy of loops in tracking is as follows:
> 1. time marching:
>   2. Particle batches:
>     3. Blocks
> So, all the transfers should take place after each batch of particles, not within subprocesses."

**Correct Flow:**
```
for timestep in timesteps:
  for batch in particle_batches:
    # 🟢 BATCH LEVEL: Single CPU→GPU transfer for all batch data
    batch_positions_gpu = jax.device_put(batch.positions)
    batch_element_ids_gpu = jax.device_put(batch.element_ids)
    batch_block_ids_gpu = jax.device_put(batch.block_ids)
    velocity_fields_gpu = jax.device_put(velocity_fields[relevant_blocks])

    for block_id in active_blocks:
      # ✅ GPU ONLY: Filter batch data on GPU for this block
      block_mask = (batch_block_ids_gpu == block_id)
      block_velocities_gpu = batch_interpolate_velocities_gpu(
        batch_positions_gpu[block_mask],
        batch_element_ids_gpu[block_mask],
        ...
      )
      # ✅ Keep results on GPU, accumulate in batch_velocities_gpu

    # 🟢 BATCH LEVEL: Single GPU→CPU transfer for all batch results
    batch.velocities = np.array(batch_velocities_gpu)
```

---

## Implementation Plan

### Step 1: Create Batch-Level Velocity Interpolator

**File:** `jaxtrace/gpu/tracking/batch_velocity_interpolation.py` (NEW)

**Purpose:**
- Process entire particle batch with single CPU→GPU transfer
- Filter particles by block on GPU (not CPU)
- Keep data on GPU throughout block loop
- Single GPU→CPU transfer at end

**Key Functions:**
```python
def interpolate_velocities_batched(
    batch_particle_data: ParticleData,
    velocity_field_all_blocks: np.ndarray,
    connectivity_gpu: jnp.ndarray,  # Pre-uploaded, persistent
    node_positions_gpu: jnp.ndarray,  # Pre-uploaded, persistent
    padded_arrays: PaddedArrays,
) -> np.ndarray:
    """
    Interpolate velocities for entire particle batch.

    CPU→GPU transfer: Once at start
    GPU processing: Loop over blocks
    GPU→CPU transfer: Once at end
    """
```

### Step 2: Create GPU-Native Block Filtering

**File:** `jaxtrace/gpu/tracking/gpu_block_filtering.py` (NEW)

**Purpose:**
- Filter particle data by block_id on GPU (not CPU)
- Avoid downloading data to CPU for grouping

**Key Functions:**
```python
@jax.jit
def filter_particles_by_block_gpu(
    all_positions: jnp.ndarray,
    all_element_ids: jnp.ndarray,
    all_block_ids: jnp.ndarray,
    target_block_id: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Filter particles belonging to target_block_id entirely on GPU.

    Returns:
    - block_positions: (n_block_particles, 3)
    - block_element_ids: (n_block_particles,)
    - block_indices: (n_block_particles,) - indices in original array
    """
```

### Step 3: Update RK4 Incremental Search

**File:** `jaxtrace/gpu/tracking/time_integration.py`

**Changes:**
- Modify `rk4_step_with_incremental_search()` to accept batch-level velocity interpolator
- Ensure velocity interpolator is called once per RK4 stage, not block-by-block

### Step 4: Update ParticleTimeMarcher

**File:** `jaxtrace/gpu/tracking/time_marching.py`

**Changes:**
- Replace `interpolate_velocities()` method with batch-level version
- Remove `_interpolate_velocities_for_block()` method (becomes internal to batch processor)
- Add batch size parameter to configuration

### Step 5: Integration with Existing Batch Processor

**File:** `jaxtrace/gpu/batching/batch_processor.py`

**Changes:**
- Integrate batch-level velocity interpolation into `process_batch()`
- Ensure search operations also follow batch-level transfer pattern

---

## Performance Benefits (Expected)

### Current Performance
- Forward Euler: 10 p/s
- RK4 L0+L1 Optimized: 13 p/s
- **Bottleneck**: Block-level CPU-GPU transfers (~5-10% overhead)

### Expected Performance After Refactoring
- **Eliminate transfer overhead**: +5-10% throughput improvement
- **Better GPU utilization**: Data stays on GPU longer
- **Reduced memory fragmentation**: Fewer small transfers
- **Foundation for async prefetching**: Enables Priority 3

**Estimated improvement:**
- Forward Euler: 10 p/s → 11-12 p/s
- RK4 L0+L1: 13 p/s → 14-15 p/s

---

## Testing Strategy

### Test 1: Single Batch Performance
- Process 1,000 particles in single batch
- Measure transfer overhead vs computation time
- Compare old vs new interpolator

### Test 2: Multi-Batch Performance
- Process 10,000 particles in 10 batches
- Verify correct data transfer at batch boundaries
- Confirm no regressions in throughput

### Test 3: RK4 Integration
- Run RK4 L0+L1 test with new batch-level interpolator
- Verify 4 RK4 stages work correctly
- Confirm no degradation in hit rates

### Test 4: Correctness Validation
- Verify particle displacements match expected values
- Check velocity interpolation accuracy
- Validate element search results

---

## Implementation Order

1. ✅ **COMPLETED**: L0+L1 RK4 optimization (Priority 1)
2. 🔄 **IN PROGRESS**: Create batch-level velocity interpolator
3. ⏳ **NEXT**: Create GPU block filtering functions
4. ⏳ **NEXT**: Integrate with RK4 incremental search
5. ⏳ **NEXT**: Update ParticleTimeMarcher
6. ⏳ **NEXT**: Testing and validation
7. ⏳ **FUTURE**: Async data prefetching (Priority 3)

---

## Files to Modify

### New Files
- `jaxtrace/gpu/tracking/batch_velocity_interpolation.py`
- `jaxtrace/gpu/tracking/gpu_block_filtering.py`

### Modified Files
- `jaxtrace/gpu/tracking/time_marching.py` (refactor interpolate_velocities)
- `jaxtrace/gpu/tracking/time_integration.py` (update RK4 to use batch interpolator)
- `jaxtrace/gpu/batching/batch_processor.py` (integrate batch transfers)
- `jaxtrace/gpu/tracking/__init__.py` (add new exports)
- `test_time_marching_integrated.py` (update test to use new API)

---

## Risk Mitigation

### Risks
1. **Breaking existing tests**: Many tests use current API
2. **GPU memory pressure**: Larger batch transfers
3. **Debugging complexity**: Harder to inspect GPU-side data
4. **JIT compilation overhead**: New JAX functions need compilation

### Mitigations
1. **Maintain backward compatibility**: Keep old API temporarily
2. **Monitor VRAM usage**: Add batch memory tracking
3. **Add logging**: Instrument batch-level operations
4. **Pre-compile kernels**: JIT compile during initialization

---

## Success Criteria

✅ All CPU-GPU transfers happen at batch level
✅ No transfers inside block loop
✅ Throughput improves by 5-10%
✅ All existing tests pass
✅ RK4 L0+L1 maintains 4.5× speedup
✅ Memory usage within acceptable limits
✅ Foundation ready for async prefetching (Priority 3)

---

## Notes

- User explicitly specified loop hierarchy: time → batches → blocks
- User explicitly requested batch-level transfers, not subprocess-level
- This refactoring enables Priority 3 (async prefetching)
- Must maintain compatibility with existing batch_processor.py architecture
