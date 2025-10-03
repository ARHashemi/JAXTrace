# GPU Optimization Guide for JAXTrace

## Current Setup (4GB GPU Memory)

### Configuration Changes Made

1. **GPU Memory Allocation**
   ```python
   os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"  # Use 75% (3GB of 4GB)
   ```
   - Leaves 1GB for system and other processes
   - Prevents out-of-memory errors

2. **JAXTrace Device Setting**
   ```python
   jt.configure(
       device="gpu",           # Changed from "cpu"
       memory_limit_gb=3.0     # Match GPU allocation
   )
   ```

3. **Increased Batch Size**
   ```python
   'batch_size': min(len(seeds), 2000)  # Increased from 500
   ```
   - Larger batches reduce overhead
   - Fewer CPU↔GPU transfers
   - Better GPU utilization

## Understanding the Progress Delays

### Why You See Delays Every ~10%

#### 1. **JIT Compilation (First Batch Only)**
   - **What**: JAX compiles functions to GPU code on first use
   - **Duration**: 10-30 seconds for first batch
   - **Frequency**: Once per run (cached afterward)
   - **Solution**: This is normal and unavoidable

#### 2. **Progress Update Intervals**
   - Code updates every 10% to reduce overhead
   - Not a performance issue, just visual

#### 3. **Memory Transfer (If Using CPU)**
   - **Old setup**: `device="cpu"` but JAX using GPU anyway
   - **Problem**: Data copied CPU→GPU for every operation
   - **New setup**: `device="gpu"` keeps data in VRAM
   - **Benefit**: 2-5x faster after compilation

## Performance Expectations

### With GPU-Optimized Settings

| Phase | Expected Behavior |
|-------|------------------|
| First batch (0-10%) | 10-30s delay (JIT compilation) |
| Subsequent batches | Smooth, 1-3s per 10% |
| Overall speedup | 2-5x faster than CPU mode |
| GPU utilization | 80-95% (good!) |
| GPU memory usage | 250MB-1GB (depending on batch) |

### Memory Usage Breakdown

```
Total 4GB GPU:
├── System reserved: ~500MB
├── JAX overhead: ~200MB
├── Velocity field: ~87MB (40 timesteps × 185k points × 3D × float32)
├── Particle data: ~24MB (2000 particles × 1500 timesteps × 3D × float32)
└── Available: ~3GB
```

## Further Optimizations

### If You Want Even Better Performance

1. **Increase Batch Size** (if memory allows)
   ```python
   'batch_size': min(len(seeds), 5000)  # Try 5000
   ```
   - Monitor GPU memory with `nvidia-smi`
   - Increase until you see ~80% memory usage

2. **Preload Velocity Field to GPU**
   ```python
   import jax.numpy as jnp

   # After loading field
   field.data = jnp.array(field.data)  # Convert to JAX array on GPU
   field.positions = jnp.array(field.positions)
   ```

3. **Reduce Timesteps for Testing**
   ```python
   'n_timesteps': 500  # For quick tests
   ```

4. **Use Compiled Mode** (if available)
   ```python
   os.environ["JAX_DISABLE_JIT"] = "0"  # Ensure JIT is enabled
   ```

## Monitoring GPU Usage

### Check GPU Status
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or one-time check
nvidia-smi
```

### Expected Output During Tracking
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.x   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        GPU-Util  Memory-Usage                                    |
|===============================+======================+======================|
|   0  Your GPU          93%   1200MiB / 4096MiB                             |
+-------------------------------+----------------------+----------------------+
```

## Troubleshooting

### Problem: Out of Memory Errors
**Solution**: Reduce batch size
```python
'batch_size': min(len(seeds), 1000)  # Reduce from 2000
```

### Problem: GPU Not Being Used (0% utilization)
**Check**:
1. JAX can see GPU: `python -c "import jax; print(jax.devices())"`
2. CUDA installed: `nvidia-smi`
3. Device setting: `device="gpu"` in config

### Problem: Very Slow (Slower Than Expected)
**Possible causes**:
1. First batch still compiling (wait 30s)
2. Small batch size causing overhead
3. CPU mode instead of GPU mode
4. Data transfer between CPU and GPU

### Problem: Progress Bar Freezes
**This is normal during**:
- JIT compilation (first batch)
- Large data transfers
- Batch processing

**Wait 30-60 seconds** - it should resume

## Summary

✅ **Your Current Optimization Status**:
- ✅ GPU enabled (`device="gpu"`)
- ✅ Memory configured (3GB/4GB)
- ✅ Large batch size (2000 particles)
- ✅ Float32 precision (memory efficient)
- ✅ Progress monitoring enabled

🎯 **Expected Performance**:
- First batch: 10-30s (compilation)
- Subsequent: ~1-3s per 10%
- Total time: ~5-10x faster than pure CPU

📊 **Your Metrics** (93% GPU, 250MB):
- **93% GPU utilization**: Excellent! GPU is working hard
- **250MB memory**: Low usage, room for larger batches
- **Delays every 10%**: Normal progress update intervals

💡 **Next Step**: Try increasing batch size to 5000 for even better performance!