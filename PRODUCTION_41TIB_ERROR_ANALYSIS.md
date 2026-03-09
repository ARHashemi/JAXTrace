# Production Script 41.70 TiB Memory Error - Analysis

## Problem Statement

Production script (`production_tracking_fully_fused_timedep.py`) fails with:
```
E0207 18:44:44.481770 1957515 gpu_hlo_schedule.cc:815] The byte size of input/output arguments (24696099900024) exceeds the base limit (27028357120). This indicates an error in the calculation!
W0207 18:44:45.327753 1957194 hlo_rematerialization.cc:3204] Can't reduce memory use below 22.40GiB (24055867801 bytes) by rematerialization; only reduced to 41.70TiB (45845471998700 bytes)
```

**Context**: Benchmark script works perfectly with SAME RK4 function (300k particles, 100 RK4 steps, 2 velocity timesteps).

## Key Differences: Production vs Benchmark

| Parameter | Production | Benchmark | Notes |
|-----------|-----------|-----------|-------|
| **Particles** | 225,000 (50×90×50) | 324,000 (60×90×60) | Production has FEWER |
| **Velocity timesteps** | 40 (range 120-159) | 2 (range 158-159) | **20× difference** |
| **RK4 steps** | Variable | 100 | Both use time cycling |
| **Octree** | Multi-cell (665k cells) | Both single+multi tested | Production only builds multi |
| **Mesh elements** | 3,048,900 | 3,048,900 | SAME |
| **Mesh nodes** | 571,173 | 571,173 | SAME (after dedup) |

## Analysis

### Expected Memory Footprint

**Benchmark (works)**:
- `positions_gpu`: (300k, 3) × 4 = 3.6 MB
- `element_ids_gpu`: (300k,) × 4 = 1.2 MB
- `velocity_fields_gpu`: **(2, 571k, 3) × 4 = 13.7 MB**
- Outputs: 4.8 MB
- **Total**: ~23 MB

**Production (fails)**:
- `positions_gpu`: (225k, 3) × 4 = 2.7 MB
- `element_ids_gpu`: (225k,) × 4 = 0.9 MB
- `velocity_fields_gpu`: **(40, 571k, 3) × 4 = 274 MB**
- Outputs: 3.6 MB
- **Total**: ~281 MB (should work fine!)

**But error reports**: 24,696,099,900,024 bytes = **22.46 TiB**

→ **This is NOT the input arrays themselves, but an INTERMEDIATE array created during JAX compilation/tracing!**

### Error Size Analysis

```
24,696,099,900,024 bytes / 4 bytes = 6,174,024,975,006 float32 elements
```

This doesn't cleanly factor as:
- ❌ `n_timesteps × n_particles × n_elements` = 40 × 225k × 3M = 27B floats (different)
- ❌ `n_timesteps × n_nodes × n_elements` = 40 × 571k × 3M = 68.5B floats (different)
- ❌ Simple multiple of known arrays

**Hypothesis**: JAX is creating a huge intermediate array during compilation, possibly due to:
1. Closure capture issue with `velocity_fields_gpu`
2. Broadcast/expansion during vmap tracing
3. Interaction between velocity sequence and octree structure
4. Something in the mesh-aligned octree search that scales with velocity timesteps

## Critical Observations

### 1. Velocity Sequence Handling

In `rk4_fully_fused_timedep.py` line 479-483:
```python
n_timesteps = velocity_fields_gpu.shape[0]

# Cyclic indexing for velocity
vel_idx = time_idx % n_timesteps
velocity_field = velocity_fields_gpu[vel_idx]  # Extract single (n_nodes, 3)
```

This **should** work fine - JAX should trace this as dynamic indexing, not unroll the entire sequence.

### 2. Function Creation Pattern

**Benchmark** (lines 295-314):
```python
elif l2_method == 'mesh_aligned_octree_multi_local':
    config.L2_SEARCH_METHOD = 'mesh_aligned_octree'

    rk4_step = create_rk4_fully_fused_timedep(
        mesh_gpu_connectivity=mesh_gpu.connectivity,
        mesh_gpu_node_positions=mesh_gpu.node_positions,
        mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
        mesh_gpu_element_volumes=element_volumes_gpu,
        mesh_gpu_global_morton=mesh_gpu_octree,
        n_hops=N_HOPS,
        enable_l1_search=ENABLE_L1_SEARCH,
        l2_search_method='radius',  # Fallback
        mesh_aligned_octree=mesh_aligned_octree_multi_gpu,  # Multi-cell octree
        mesh_aligned_octree_use_multi_local=True
    )
```

**Production** (lines 901-914):
```python
if L2_SEARCH_METHOD == 'mesh_aligned_octree':
    rk4_step = create_rk4_fully_fused_timedep(
        mesh_gpu_connectivity=mesh_gpu.connectivity,
        mesh_gpu_node_positions=mesh_gpu.node_positions,
        mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
        mesh_gpu_element_volumes=element_volumes_gpu,
        mesh_gpu_global_morton=mesh_gpu_octree,
        n_hops=N_HOPS,
        enable_l1_search=ENABLE_L1_SEARCH,
        l2_search_method='radius',  # Fallback
        mesh_aligned_octree=mesh_aligned_octree_gpu,  # Multi-cell octree
        mesh_aligned_octree_use_multi_local=True
    )
```

**Pattern looks IDENTICAL** (just different variable names for octree: `mesh_aligned_octree_multi_gpu` vs `mesh_aligned_octree_gpu`)

### 3. Compilation Call

**Both scripts** call RK4 the same way:
```python
positions_gpu, element_ids_gpu = rk4_step(
    positions_gpu,
    element_ids_gpu,
    DT,
    velocity_fields_gpu,  # or velocity_sequence_gpu (same thing)
    0  # time_idx
)
```

## Diagnostic Strategy

### Immediate Actions

1. **Print array shapes before compilation** in production script (line ~1083):
```python
print(f"\n  DEBUG: Array shapes before compilation:")
print(f"    positions_gpu: {positions_gpu.shape}")
print(f"    element_ids_gpu: {element_ids_gpu.shape}")
print(f"    velocity_fields_gpu: {velocity_fields_gpu.shape}")
print(f"    DT: {DT}")
print(f"    mesh_gpu.connectivity: {mesh_gpu.connectivity.shape}")
print(f"    mesh_gpu.node_positions: {mesh_gpu.node_positions.shape}")
print(f"    mesh_gpu.element_neighbors: {mesh_gpu.element_neighbors.shape}")
print(f"    element_volumes_gpu: {element_volumes_gpu.shape}")
print(f"    mesh_aligned_octree_gpu.cell_to_elements_offsets: {mesh_aligned_octree_gpu.cell_to_elements_offsets.shape}")
print(f"    mesh_aligned_octree_gpu.cell_to_elements_data: {mesh_aligned_octree_gpu.cell_to_elements_data.shape}")
```

2. **Test with 2 velocity timesteps** (same as benchmark):
   - Change line 88: `VELOCITY_TIMESTEP_RANGE = (158, 159)`
   - If this WORKS → confirms 40 timesteps is the trigger
   - If this FAILS → something else is wrong

3. **Simplify RK4 creation** to absolute minimum:
```python
# Test with ONLY required parameters
rk4_step = create_rk4_fully_fused_timedep(
    mesh_gpu_connectivity=mesh_gpu.connectivity,
    mesh_gpu_node_positions=mesh_gpu.node_positions,
    mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
    mesh_gpu_element_volumes=element_volumes_gpu,
    mesh_gpu_global_morton=mesh_gpu_octree
    # NO optional parameters at all!
)
```

4. **Check if velocity sequence shape is corrupted**:
```python
print(f"\n  Velocity sequence validation:")
print(f"    Type: {type(velocity_sequence)}")
print(f"    Shape: {velocity_sequence.shape}")
print(f"    Dtype: {velocity_sequence.dtype}")
print(f"    Size: {velocity_sequence.nbytes / (1024**2):.1f} MB")
print(f"    Min/max: [{velocity_sequence.min():.6f}, {velocity_sequence.max():.6f}]")

print(f"\n  After GPU upload:")
print(f"    Type: {type(velocity_fields_gpu)}")
print(f"    Shape: {velocity_fields_gpu.shape}")
print(f"    Dtype: {velocity_fields_gpu.dtype}")
```

### Secondary Investigation

5. **Compare octree structures** between benchmark and production:
   - Are the multi-cell octrees actually identical?
   - Check cell counts, elements per cell, etc.

6. **JAX configuration differences**:
   - Check `JAX_PLATFORM_NAME`
   - Check `jax.config` settings
   - Compare JAX versions if different environments

7. **Memory fragmentation**:
   - Try running with `XLA_PYTHON_CLIENT_PREALLOCATE=false`
   - Try with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.7`

## Next Steps

**IMMEDIATE**: Add shape printing and test with 2 timesteps. This will definitively show if the 40 timesteps is the problem or if there's something else.

**IF 2 timesteps works**: The issue is with how JAX handles the larger velocity sequence in closures. Solutions:
- Restructure to pass velocity as explicit parameter to each search level
- Use `jax.checkpoint` to control memory
- Investigate XLA compilation flags

**IF 2 timesteps still fails**: The issue is unrelated to velocity timesteps. Look at:
- Octree structure differences
- Mesh data corruption
- JAX/XLA environment issues
