# 41.70 TiB Memory Error - Investigation Summary

## What I Found

After systematic comparison between `benchmark_l2_search_methods.py` (WORKS) and `production_tracking_fully_fused_timedep.py` (FAILS), I identified **ONE CRITICAL DIFFERENCE**:

### Velocity Timesteps

| Script | Timestep Range | # Timesteps | Velocity Size | Status |
|--------|---------------|-------------|---------------|---------|
| **Benchmark** | `(158, 159)` | **2** | 13.7 MB | ✅ **WORKS** |
| **Production** | `(120, 159)` | **40** | 274 MB | ❌ **FAILS with 41.70 TiB error** |

**Everything else is identical or BETTER in production**:
- Particles: 225k (production) vs 324k (benchmark) → production has **FEWER**
- Mesh: 3.05M elements, 571k nodes → **SAME**
- RK4 function: **SAME** (`create_rk4_fully_fused_timedep`)
- Octree structure: **SAME** (multi-cell vertex registration)
- Parameter pattern: **IDENTICAL**

## The Error

```
The byte size of input/output arguments (24696099900024) exceeds the base limit (27028357120)
Can't reduce memory use below 22.40GiB by rematerialization; only reduced to 41.70TiB
```

**Translation**: JAX is trying to create a **22.46 TiB** intermediate array during compilation. This is NOT the input arrays (which are only ~281 MB total) but something being created internally.

## Why This Happens

The RK4 function creates a **closure** over `velocity_fields_gpu` with shape `(40, 571173, 3)`. During JAX compilation/tracing, something is causing this to expand into a huge intermediate array. Possible causes:

1. **XLA compiler unrolling**: XLA might be trying to unroll/expand operations involving the velocity sequence
2. **Broadcast during vmap**: The `jax.vmap` over particles might be incorrectly broadcasting the velocity array
3. **Interaction with octree structure**: The mesh-aligned octree search might be interacting badly with the larger velocity sequence

## What I've Done

### 1. Created Diagnostic Documentation

- [PRODUCTION_41TIB_ERROR_ANALYSIS.md](PRODUCTION_41TIB_ERROR_ANALYSIS.md) - Detailed analysis

### 2. Added Debug Output

Modified [production_tracking_fully_fused_timedep.py](production_tracking_fully_fused_timedep.py) at line ~1084 to print all array shapes before compilation:

```python
print(f"\n  DEBUG: Array shapes before compilation:")
print(f"    positions_gpu: {positions_gpu.shape}")
print(f"    element_ids_gpu: {element_ids_gpu.shape}")
print(f"    velocity_fields_gpu: {velocity_fields_gpu.shape}")
... (all mesh and octree arrays)
```

This will help verify shapes are correct.

### 3. Created Test Script

[test_production_2timesteps.sh](test_production_2timesteps.sh) - Automatically tests production script with only 2 timesteps.

## What You Should Do Next

### IMMEDIATE TEST (5 minutes):

Run the 2-timestep test:
```bash
./test_production_2timesteps.sh
```

**If it WORKS**:
- ✅ Confirms 40 timesteps is the trigger
- → Solution: Need to restructure velocity handling (see below)

**If it FAILS**:
- ❌ Something else is wrong
- → Need to investigate octree structure or JAX environment

### If 2-Timestep Test Works

The problem is confirmed to be the 40 timesteps. Solutions (in order of preference):

#### Option A: Restructure Velocity as External Parameter (RECOMMENDED)

Instead of closing over `velocity_fields_gpu`, pass it explicitly to each search level:

```python
# Current (problematic):
def search_l0_l1_l2_single(pos, elem_id):
    # ... searches ...
    velocity_field = velocity_fields_gpu[vel_idx]  # Closure
    # ... interpolation ...

# Fixed:
def search_l0_l1_l2_single(pos, elem_id, velocity_field):
    # velocity_field passed explicitly (n_nodes, 3)
    # ... searches and interpolation ...
```

This requires modifying `rk4_fully_fused_timedep.py` but is the cleanest solution.

#### Option B: Use JAX Checkpointing

Add `jax.checkpoint` to control memory:

```python
@jax.checkpoint
def rk4_single_particle(pos, elem_id):
    ...
```

This tells JAX to rematerialize instead of caching, which might help with the huge intermediate.

#### Option C: Load Velocity On-Demand (WORKAROUND)

Instead of uploading all 40 timesteps, only upload 2-3 at a time and cycle through them during tracking:

```python
# Upload only 3 timesteps to GPU
velocity_window_gpu = jax.device_put(velocity_sequence[timestep:timestep+3])

# Update window every few steps
if step % 10 == 0:
    velocity_window_gpu = jax.device_put(velocity_sequence[new_timestep:new_timestep+3])
```

This avoids the 40-timestep closure but adds transfer overhead.

#### Option D: XLA Compiler Flags

Try running with different XLA settings:

```bash
XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1" python3 production_tracking_fully_fused_timedep.py
```

or

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache python3 production_tracking_fully_fused_timedep.py
```

### If 2-Timestep Test Still Fails

Then the issue is NOT velocity timesteps. Check:

1. **Octree structure difference**:
   - Print `mesh_aligned_octree_gpu.n_cells` in both scripts
   - Compare `cell_to_elements_offsets.shape` and `cell_to_elements_data.shape`

2. **JAX environment**:
   ```bash
   python3 -c "import jax; print(f'JAX: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
   ```

3. **Mesh data corruption**:
   - Check if `velocity_sequence` has unexpected shape after deduplication
   - Verify `connectivity` and `node_positions` shapes match benchmark

## Recommendation

**START WITH**: Run `./test_production_2timesteps.sh` now. This 5-minute test will definitively tell us if velocity timesteps are the problem.

Once we know the result, I can provide the exact fix needed.

---

## Technical Details (For Reference)

### Expected Array Sizes

**Production with 40 timesteps**:
- Total inputs: ~281 MB
- Expected compilation: ~500 MB peak
- **Actual error**: 22.46 TiB (45,000× larger!)

### Error Size Factorization

24,696,099,900,024 bytes doesn't cleanly factor as any known array combination:
- ❌ 40 × 225k × 3M = 27B floats (≠ 6.17 trillion)
- ❌ 40 × 571k × 3M = 68.5B floats (≠ 6.17 trillion)

This suggests JAX is creating some complex intermediate that's larger than simple products of input dimensions.

### Velocity Handling in RK4

Current implementation (`rk4_fully_fused_timedep.py:479-483`):
```python
n_timesteps = velocity_fields_gpu.shape[0]
vel_idx = time_idx % n_timesteps
velocity_field = velocity_fields_gpu[vel_idx]  # Dynamic indexing
```

This **should** work (dynamic indexing is supported), but something in the compilation is causing issues with 40 timesteps that doesn't happen with 2.
