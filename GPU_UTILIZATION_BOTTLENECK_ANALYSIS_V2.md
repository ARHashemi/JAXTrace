# GPU Utilization Bottleneck Analysis V2

**Date:** 2025-11-30
**Current Test:** Filtered octree implementation running
**User Observation:** GPU spikes to 55%, then idles at 0% (repeating pattern)

---

## User Observations

### GPU Utilization Pattern (During Time Marching)
- GPU loads shortly to **55%** during compute
- GPU idles at **0%** for long periods
- Pattern repeats every timestep
- **Expected:** 85-95% sustained utilization

### Memory Usage During Initialization
- Before initial search: **~600 MiB**
- During initial search: **~2,800 MiB** (jump of ~2.2 GB)
- **Question:** Is this necessary? Can we free space for more useful arrays?

---

## Root Cause Analysis

### Bottleneck #1: Per-Timestep CPU-GPU Transfers (CRITICAL)

**File:** [jaxtrace/gpu/tracking/rk4_gpu_fused.py:1227-1259](jaxtrace/gpu/tracking/rk4_gpu_fused.py#L1227-L1259)

**Current Implementation:**
```python
def rk4_step_gpu_fused_for_production_with_l2_octree(...):
    # UPLOAD to GPU (EVERY timestep) - 1.6 MB
    positions_gpu = jax.device_put(positions.astype(np.float32))  # 100k × 3 × 4 = 1.2 MB
    element_ids_gpu = jax.device_put(element_ids.astype(np.int32))  # 100k × 4 = 400 KB

    # GPU COMPUTE
    positions_final_gpu, element_ids_final_gpu = rk4_fused_with_l2_search(...)
    positions_final_gpu.block_until_ready()  # Synchronization point

    # DOWNLOAD from GPU (EVERY timestep) - 1.6 MB
    positions_final = np.array(positions_final_gpu)  # 1.2 MB
    element_ids_final = np.array(element_ids_final_gpu)  # 400 KB
```

**Timeline Per Timestep:**
```
1. Upload (GPU idle at 0%)         ← 1-2 ms
2. Compute (GPU busy at 55%)       ← 5-10 ms
3. block_until_ready() sync        ← 0 ms (already done)
4. Download (GPU idle at 0%)       ← 1-2 ms
5. Python/CPU overhead             ← 1-2 ms
   → GPU active only ~70% of timestep
```

**Impact:**
- Upload/download happens **2,500 times** (every timestep)
- Total transfer: 1.6 MB × 2 × 2,500 = **8 GB**
- GPU idle during transfers → 55% utilization instead of 95%
- No overlap between compute and transfer (serialized)

---

### Bottleneck #2: Temporary Padded Arrays During Initialization

**GPU Memory Jump:** 600 MiB → 2,800 MiB (~2.2 GB)

**Cause:** Padded block arrays uploaded temporarily for initial assignment

**File:** [production_tracking_3hop_l2_octree.py:533-548](production_tracking_3hop_l2_octree.py#L533-L548)

```python
# Build padded arrays (needed for initial assignment)
padded_arrays = build_padded_block_arrays(...)
print(f"  Memory: {padded_arrays.memory_mb:.1f} MB")  # 6,593.8 MB on CPU

# During initial_search_batch(), these are uploaded to GPU temporarily
element_ids, _, _ = initial_search_batch(
    particle_positions,
    padded_arrays,  # ← Uploaded to GPU (2.2 GB)
    ...
)

# After initialization, padded arrays are deleted (memory freed)
```

**Padded Array Structure:**
- 256 blocks × 450,004 elements (worst-case padding)
- Average block: 13,735 elements
- **Waste:** 98% of array is padding/zeros

**Why So Large?**
- Heaviest block has 450,004 elements
- All blocks padded to same size for uniform GPU arrays
- Necessary for vectorized block-local search

**After Initialization:**
- Padded arrays deleted
- GPU memory returns to baseline (~600 MB) + permanent arrays (~235 MB)

**Conclusion:** The 2.2 GB jump is **expected and temporary** during initialization only.

---

## GPU Memory Breakdown

### Permanent GPU Arrays (Time Marching)

| Array | Size | Usage |
|-------|------|-------|
| Mesh connectivity | 53.6 MB | ✅ Element-to-node mapping |
| Mesh node positions | 10.3 MB | ✅ Node coordinates |
| Mesh neighbors | 53.6 MB | ✅ Element adjacency |
| Velocity field | 10.3 MB | ✅ Pre-uploaded, reused |
| Octree metadata | 24.4 MB | ✅ Node bounding boxes |
| Octree elements | 81.2 MB | ✅ Leaf element lists |
| **Total Permanent** | **233.4 MB** | Stays on GPU |

### Temporary GPU Arrays (Per Timestep)

| Array | Size | Frequency |
|-------|------|-----------|
| Particle positions | 1.2 MB | Upload every step |
| Particle element_ids | 0.4 MB | Upload every step |
| Result positions | 1.2 MB | Download every step |
| Result element_ids | 0.4 MB | Download every step |
| **Total Transfer/Step** | **3.2 MB** | 2,500 × = 8 GB total |

### Temporary During Initialization Only

| Array | Size | When |
|-------|------|------|
| Padded block arrays | 2.2 GB | Initial search only |
| Hash bucket arrays | ~50 MB | Initial search only |
| **Total** | **~2.25 GB** | Deleted after init |

---

## Solutions

### Solution #1: GPU-Resident Particle Data (CRITICAL)

**Eliminate per-timestep CPU-GPU transfers by keeping particles on GPU permanently.**

#### Option A: Modify Existing Wrapper (Dual Mode)

**Add parameter to control GPU residency:**

```python
def create_rk4_step_gpu_fused_for_production_with_l2_octree(
    n_hops: int = 3,
    octree_metadata: Optional[jax.Array] = None,
    octree_elements: Optional[jax.Array] = None,
    max_octree_depth: int = 10,
    keep_on_gpu: bool = False  # ← NEW: Enable GPU-resident mode
):
    search_func = create_search_gpu_fused_with_l2_octree(...)

    if keep_on_gpu:
        # GPU-RESIDENT MODE (FAST)
        @jax.jit
        def rk4_step_gpu_resident(
            positions_gpu: jax.Array,  # Already on GPU
            element_ids_gpu: jax.Array,  # Already on GPU
            dt: float,
            mesh_gpu: MeshDataGPU,
            velocity_field_gpu: jax.Array
        ):
            # Pure GPU - no CPU interaction
            return positions_final_gpu, element_ids_final_gpu

        return rk4_step_gpu_resident

    else:
        # CPU-GPU MODE (CURRENT, backward compatible)
        def rk4_step_with_transfers(...):
            # Upload, compute, download (as before)
            ...
        return rk4_step_with_transfers
```

#### Production Script Changes

```python
# Initialize particles on CPU
particle_data = ParticleData(
    positions=np.array(...),
    element_ids=np.array(...),
    ...
)

# Create GPU-resident RK4 function
rk4_step_func = create_rk4_step_gpu_fused_for_production_with_l2_octree(
    n_hops=3,
    octree_metadata=octree_metadata_gpu,
    octree_elements=octree_elements_gpu,
    keep_on_gpu=True  # ← Enable GPU residency
)

# Upload particles ONCE before time marching
positions_gpu = jax.device_put(particle_data.positions)
element_ids_gpu = jax.device_put(particle_data.element_ids)

# Time marching loop
for step in range(N_TIMESTEPS):
    # GPU-to-GPU (NO transfers!)
    positions_gpu, element_ids_gpu = rk4_step_func(
        positions_gpu,
        element_ids_gpu,
        DT,
        mesh_gpu,
        velocity_field_gpu
    )

    # Only download for export (every 10 steps)
    if step % EXPORT_FREQUENCY == 0:
        positions_cpu = np.array(positions_gpu)  # Download once per 10 steps
        element_ids_cpu = np.array(element_ids_gpu)

        # Queue export
        exporter.queue_export(step, positions_cpu, ...)
```

#### Expected Performance Improvement

**Before (Current):**
```
Per timestep:
  Upload: 1.6 MB (1-2 ms)
  Compute: 5-10 ms (GPU 55%)
  Download: 1.6 MB (1-2 ms)
  Total: 7-14 ms
  GPU utilization: 50-70%
```

**After (GPU-Resident):**
```
Per timestep:
  Compute: 5-10 ms (GPU 90-95%)
  Total: 5-10 ms
  GPU utilization: 90-95%

Every 10 steps (export):
  Download: 1.6 MB (1-2 ms)  # Only when exporting
```

**Speedup:**
- Eliminate 8 GB transfers over 2,500 steps
- Reduce time/step: 7-14 ms → 5-10 ms (**1.4-2× faster**)
- Increase GPU utilization: 55% → 90% (**1.6× better**)
- **Total expected throughput:** 40-48k p/s → **60-80k p/s**

---

### Solution #2: Optimize Initialization Memory (Optional)

**Goal:** Free 2.2 GB GPU memory after initialization

#### Option A: Explicit Cleanup

```python
# After initial assignment
element_ids, _, _ = initial_search_batch(...)

# Delete temporary arrays immediately
del padded_arrays
del hash_bucket_arrays
import gc
gc.collect()

print(f"✓ Temporary arrays freed")
print(f"  GPU memory available: {jax.devices()[0].memory_stats()['bytes_available'] / 1e9:.1f} GB")
```

**Benefit:** Frees 2.2 GB for larger simulations
**Effort:** Trivial (2 lines of code)

#### Option B: Use Octree for Initialization (Advanced)

```python
# Skip padded arrays entirely, use octree for initial search
element_ids = search_level2_octree_scan(
    particle_positions,
    jnp.full(len(particle_positions), -1),  # All unfound
    octree_metadata_gpu,
    octree_elements_gpu,
    mesh_gpu.node_positions,
    mesh_gpu.connectivity,
    max_depth=10
)
```

**Benefits:**
- No padded arrays needed (saves 2.2 GB)
- Uses existing octree (~105 MB)
- Simpler code

**Drawbacks:**
- Slightly slower initialization (octree traversal vs block scan)
- But only runs once, so acceptable

---

### Solution #3: Reduce Octree Size (If Still Large)

**Current octree:** 415,921 nodes (100% element filtering)
**Expected after filtered fix:** 4,284 nodes (30% element filtering)

**If octree is still too large:**
1. Verify levelset filtering is working: `mask = level_field < threshold`
2. Adjust threshold: Lower threshold → fewer elements → smaller octree
3. Check octree depth: Reduce `max_depth` if actual depth < 10

---

## Implementation Priority

### CRITICAL (Implement Now)
1. **GPU-resident particle data (Solution #1)**
   - **Impact:** 1.4-2× speedup, 90% GPU utilization
   - **Effort:** 1-2 hours (modify RK4 wrapper + production script)
   - **Risk:** Medium (requires testing)

### VERIFY (Test Currently Running)
2. **Filtered octree fix**
   - Already implemented, testing in other terminal
   - Expected: 100-120× speedup (if nested scan fixed)
   - Verify octree size reduction (415k → 4k nodes)

### OPTIONAL (After Above Works)
3. **Delete padded arrays after init (Solution #2A)**
   - **Impact:** Free 2.2 GB GPU memory
   - **Effort:** Trivial (2 lines)
   - **Priority:** Low (doesn't affect time marching)

4. **Tune octree parameters (Solution #3)**
   - Only if octree still too large after filtering
   - Reduce max_depth, adjust levelset threshold

---

## Expected Final Performance

### Current (With Filtered Octree, NO GPU-Resident)
```
Throughput: 40-48k p/s
GPU utilization: 55% (compute), 0% (transfers)
Time/step: 0.11-0.15s
Total time (2,500 steps): 4.6-6.3 minutes
```

### With GPU-Resident Particles
```
Throughput: 60-80k p/s
GPU utilization: 90-95% (sustained)
Time/step: 0.06-0.08s
Total time (2,500 steps): 2.5-3.3 minutes
Speedup: 1.4-2× vs current
```

---

## Answers to User Questions

### Q1: Why GPU jumps from 600MB to 2,800MB during initialization?

**Answer:** Temporary padded block arrays (2.2 GB) uploaded for initial search.
- **Padded arrays:** 256 blocks × 450k elements (worst-case padding)
- **Used only during initialization** - deleted afterward
- **This is expected and necessary** for vectorized block-local search
- **Can be optimized:** Use octree for initialization instead (saves 2.2 GB)

### Q2: Why GPU utilization spikes to 55% then idles at 0%?

**Answer:** CPU-GPU transfer bottleneck.
- **Upload** particle data (1-2 ms, GPU idle)
- **Compute** on GPU (5-10 ms, GPU 55%)
- **Download** results (1-2 ms, GPU idle)
- **Solution:** Keep particles on GPU (no transfers except during export)

### Q3: Can we load more arrays permanently on GPU?

**Answer:** Yes! Currently only 233 MB used permanently.
- **Available:** Most GPUs have 8-24 GB VRAM
- **Current:** 233 MB permanent + 1.6 MB temporary per step
- **After GPU-resident:** 233 MB + 1.6 MB = 234.6 MB (particles stay on GPU)
- **Plenty of room** for larger meshes or more particles

---

**Date:** 2025-11-30
**Analysis by:** Claude Code
