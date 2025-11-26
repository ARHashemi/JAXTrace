# GPU Transfer Bottleneck Analysis

## Performance Gap

**Current Performance:**
- Throughput: 40k p/s (step 100)
- GPU utilization: 0-11% (low!)
- Time per timestep: ~1.5 seconds

**Expected Performance (from working log):**
- Throughput: 644k p/s (step 100)
- GPU utilization: High (expected 80%+)
- Time per timestep: ~0.1 seconds

**Performance loss: 16× slower than expected!**

---

## Root Cause: Repeated CPU-GPU Transfers

### Transfer Pattern Per Timestep

The current implementation uploads/downloads particle data on EVERY timestep:

**File: `jaxtrace/gpu/tracking/rk4_gpu_fused.py`**

```python
def rk4_step_gpu_fused_wrapper(positions, element_ids, dt, mesh_gpu, velocity_field, n_hops):
    # LINE 505-506: UPLOAD TO GPU (every timestep!)
    positions_gpu = jax.device_put(positions.astype(np.float32))      # ~750 KB
    element_ids_gpu = jax.device_put(element_ids.astype(np.int32))   # ~250 KB

    # LINE 508-513: Check if velocity_field needs upload
    if isinstance(velocity_field, np.ndarray):
        velocity_field_gpu = jax.device_put(velocity_field)           # ~10 MB (FIXED!)

    # ... GPU computation ...

    # LINE 526-527: DOWNLOAD FROM GPU (every timestep!)
    positions_final = np.array(positions_final_gpu)                    # ~750 KB
    element_ids_final = np.array(element_ids_final_gpu)               # ~250 KB
```

**File: `jaxtrace/gpu/tracking/rk4_gpu_fused.py`**

```python
def rk4_step_gpu_fused_for_production(particle_data, velocity_field, dt, mesh_gpu, ...):
    # LINE 597-598: particle_data contains NUMPY ARRAYS
    positions_new, element_ids_new, stats = rk4_step_gpu_fused_wrapper(
        particle_data.positions,    # numpy array → triggers upload
        particle_data.element_ids,  # numpy array → triggers upload
        ...
    )

    # LINE 606-609: Create new ParticleData with NUMPY ARRAYS
    new_particle_data = replace(
        particle_data,
        positions=positions_new,       # numpy array (downloaded from GPU)
        element_ids=element_ids_new    # numpy array (downloaded from GPU)
    )
```

### Transfer Volume

**Per timestep (62,500 particles at step 100):**
- Upload positions: 62,500 × 3 × 4 bytes = 750 KB
- Upload element_ids: 62,500 × 4 bytes = 250 KB
- Download positions: 750 KB
- Download element_ids: 250 KB
- **Total: 2 MB per timestep**

**Total for 2,500 timesteps:**
- **5 GB of unnecessary transfers!**

**Transfer time estimate:**
- PCIe 3.0 bandwidth: ~12 GB/s theoretical
- Actual bandwidth: ~6 GB/s (with overhead)
- Transfer time per timestep: 2 MB / 6 GB/s = **0.33 ms**
- But each transfer has latency: ~10-50 μs × 4 transfers = **40-200 μs overhead**
- Plus synchronization overhead

**Total overhead per timestep: ~500 μs to 1 ms**

This explains the low GPU utilization (0-11%): The GPU is idle waiting for data transfers!

---

## Why Velocity Field Fix Wasn't Enough

We successfully fixed the velocity field upload (lines 508-513), eliminating **10 MB × 2,500 = 25 GB** of transfers.

But we still have **particle data** transfers (positions + element_ids) happening every timestep because:

1. **ParticleData uses numpy arrays** (`jaxtrace/gpu/particles.py:55-59`)
2. **RK4 wrapper always converts to numpy** (line 526-527)
3. **Production script passes numpy arrays** (line 597-598)

---

## Proposed Solutions

### **Option 1: GPU-Resident ParticleData (RECOMMENDED)**

**Approach:** Keep particle data on GPU throughout entire simulation

**Changes Required:**

1. **Create `ParticleDataGPU` class:**
```python
@dataclass
class ParticleDataGPU:
    """GPU-resident particle data (JAX arrays)"""
    positions: jax.Array      # [N, 3] float32 - ON GPU
    velocities: jax.Array     # [N, 3] float32 - ON GPU
    element_ids: jax.Array    # [N] int32 - ON GPU
    block_ids: jax.Array      # [N] int32 - ON GPU
    active_mask: jax.Array    # [N] bool - ON GPU
```

2. **Modify `rk4_step_gpu_fused_for_production` to accept/return `ParticleDataGPU`:**
```python
def rk4_step_gpu_fused_for_production(
    particle_data_gpu: ParticleDataGPU,  # Already on GPU!
    velocity_field_gpu: jax.Array,        # Already on GPU!
    ...
) -> ParticleDataGPU:
    # No uploads! Work directly with GPU arrays
    positions_new_gpu, element_ids_new_gpu = rk4_fused_with_search(
        particle_data_gpu.positions,    # Already on GPU
        particle_data_gpu.element_ids,  # Already on GPU
        ...
    )

    # Return GPU-resident data
    return ParticleDataGPU(
        positions=positions_new_gpu,    # Stay on GPU
        element_ids=element_ids_new_gpu,
        ...
    )
```

3. **Upload particle data ONCE at initialization:**
```python
# After initial assignment
particle_data_cpu = ParticleData(...)  # Numpy arrays
particle_data_gpu = ParticleDataGPU(   # JAX arrays
    positions=jax.device_put(particle_data_cpu.positions),
    element_ids=jax.device_put(particle_data_cpu.element_ids),
    ...
)
```

4. **Download ONLY when exporting VTK:**
```python
if step % EXPORT_FREQUENCY == 0:
    # Download only when needed for export
    positions_cpu = np.array(particle_data_gpu.positions)
    exporter.queue_export(step, positions_cpu, ...)
```

**Benefits:**
- ✅ Eliminates 2 MB × 2,500 = **5 GB transfers**
- ✅ Particle data stays on GPU throughout simulation
- ✅ Only download for VTK export (every 10 steps)
- ✅ Expected speedup: **10-16×** (reaches 400-640k p/s)

**Drawbacks:**
- Requires creating new `ParticleDataGPU` class
- Requires modifying production script initialization
- Need to handle boundary deactivation on GPU

**Effort:** Medium (2-3 hours)

---

### **Option 2: Batch Multiple Timesteps (PARTIAL FIX)**

**Approach:** Upload once, run N timesteps on GPU, download once

**Changes Required:**

```python
def rk4_multi_step_gpu_fused(
    particle_data,
    velocity_field_gpu,
    dt,
    n_steps: int,  # Run N timesteps on GPU
    mesh_gpu,
    ...
):
    # Upload ONCE
    positions_gpu = jax.device_put(particle_data.positions)
    element_ids_gpu = jax.device_put(particle_data.element_ids)

    # Run N timesteps on GPU
    for i in range(n_steps):
        positions_gpu, element_ids_gpu = rk4_fused_with_search(
            positions_gpu, element_ids_gpu, ...
        )

    # Download ONCE
    return np.array(positions_gpu), np.array(element_ids_gpu)
```

**Benefits:**
- ✅ Reduces transfers by factor of N
- ✅ Less intrusive than Option 1
- ✅ Amortizes transfer cost

**Drawbacks:**
- ❌ Still has transfers (just less frequent)
- ❌ Complicates VTK export (need to export mid-batch)
- ❌ Boundary deactivation happens after N steps (not every step)
- ❌ Only 2-4× speedup (not 16×)

**Effort:** Low (1 hour)

---

### **Option 3: Hybrid - GPU-Resident Until Export (RECOMMENDED)**

**Approach:** Combine Option 1 simplicity with minimal changes

**Changes Required:**

1. **Modify `rk4_step_gpu_fused_wrapper` to return JAX arrays:**
```python
def rk4_step_gpu_fused_wrapper(..., return_gpu_arrays=False):
    ...
    if return_gpu_arrays:
        # Return GPU arrays (no download)
        return positions_final_gpu, element_ids_final_gpu, stats
    else:
        # Return numpy arrays (download for compatibility)
        return np.array(positions_final_gpu), np.array(element_ids_final_gpu), stats
```

2. **Keep particle data on GPU in production script:**
```python
# Upload once at initialization
positions_gpu = jax.device_put(particle_data.positions)
element_ids_gpu = jax.device_put(particle_data.element_ids)

for step in range(N_TIMESTEPS):
    # Run RK4 on GPU (no transfers)
    positions_gpu, element_ids_gpu, stats = rk4_step_gpu_fused_wrapper(
        positions_gpu,      # JAX array (already on GPU)
        element_ids_gpu,    # JAX array (already on GPU)
        ...,
        return_gpu_arrays=True  # Don't download
    )

    # Download only for export
    if step % EXPORT_FREQUENCY == 0:
        positions_cpu = np.array(positions_gpu)
        element_ids_cpu = np.array(element_ids_gpu)

        # Boundary deactivation
        active_mask = ...

        # Update ParticleData for export
        particle_data = replace(particle_data,
            positions=positions_cpu[active_mask],
            element_ids=element_ids_cpu[active_mask],
            ...
        )
        exporter.queue_export(...)
```

**Benefits:**
- ✅ Eliminates most transfers (only download for export)
- ✅ Minimal code changes
- ✅ No new data structures needed
- ✅ Expected speedup: **10-16×**
- ✅ Simplest to implement

**Drawbacks:**
- Need to handle boundary deactivation separately
- Production script logic becomes slightly more complex

**Effort:** Low-Medium (1-2 hours)

---

## Recommendation

**I recommend Option 3: Hybrid approach**

**Rationale:**
1. **Minimal changes:** No new data structures, just keep JAX arrays in production loop
2. **Maximum speedup:** Eliminates 99% of transfers (only export downloads)
3. **Low risk:** Easy to revert if issues arise
4. **Quick to implement:** 1-2 hours

**Expected Performance:**
- Throughput: 400-640k p/s (10-16× improvement)
- GPU utilization: 80-90% (high)
- Transfer volume: 5 GB → 50 MB (100× reduction)

---

## Implementation Steps (Option 3)

### Step 1: Modify RK4 wrapper (5 min)
```python
# File: jaxtrace/gpu/tracking/rk4_gpu_fused.py
def rk4_step_gpu_fused_wrapper(..., return_gpu_arrays=False):
    ...
    # Check if inputs are already on GPU
    if isinstance(positions, jax.Array):
        positions_gpu = positions
    else:
        positions_gpu = jax.device_put(positions.astype(np.float32))

    if isinstance(element_ids, jax.Array):
        element_ids_gpu = element_ids
    else:
        element_ids_gpu = jax.device_put(element_ids.astype(np.int32))

    ...

    # Return GPU or CPU arrays based on flag
    if return_gpu_arrays:
        return positions_final_gpu, element_ids_final_gpu, stats
    else:
        return np.array(positions_final_gpu), np.array(element_ids_final_gpu), stats
```

### Step 2: Update production script (30 min)
```python
# Upload particle data once after initial assignment
positions_gpu = jax.device_put(particle_data.positions)
element_ids_gpu = jax.device_put(particle_data.element_ids)

for step in range(N_TIMESTEPS):
    # RK4 on GPU (no transfers)
    positions_gpu, element_ids_gpu, stats = rk4_step_gpu_fused_wrapper(
        positions_gpu,
        element_ids_gpu,
        dt=DT,
        mesh_gpu=mesh_gpu,
        velocity_field=velocity_field_gpu,
        n_hops=RK4_L1_HOP_COUNT,
        return_gpu_arrays=True
    )

    # Export and boundary check every N steps
    if step % EXPORT_FREQUENCY == 0:
        # Download for processing
        positions_cpu = np.array(positions_gpu)
        element_ids_cpu = np.array(element_ids_gpu)

        # Boundary deactivation
        out_of_bounds = (
            (positions_cpu[:, 0] < bbox[0]) | (positions_cpu[:, 0] > bbox[1]) |
            (positions_cpu[:, 1] < bbox[2]) | (positions_cpu[:, 1] > bbox[3]) |
            (positions_cpu[:, 2] < bbox[4]) | (positions_cpu[:, 2] > bbox[5])
        )
        active_mask = ~out_of_bounds

        # Keep only active particles on GPU
        positions_gpu = positions_gpu[active_mask]
        element_ids_gpu = element_ids_gpu[active_mask]

        # Update ParticleData for export
        particle_data = replace(particle_data,
            positions=positions_cpu[active_mask],
            element_ids=element_ids_cpu[active_mask],
            n_active=active_mask.sum()
        )

        # Queue export
        exporter.queue_export(step, particle_data)
```

### Step 3: Test (30 min)
- Run with small timestep count (100 steps)
- Verify GPU utilization increases
- Verify throughput increases
- Verify VTK export still works

---

## Alternative: Just Test Option 1 First

If you prefer the cleaner architecture, I can implement Option 1 (GPU-resident ParticleData) first. It's more work but cleaner design.

Your choice:
- **A) Option 3 (Hybrid)** - Fast, minimal changes, 1-2 hour implementation
- **B) Option 1 (ParticleDataGPU)** - Clean, more work, 2-3 hour implementation
- **C) Something else** - Your preference
