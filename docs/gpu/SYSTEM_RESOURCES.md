# System Resources Profile

**Date**: 2025-11-03
**Purpose**: Document available hardware for GPU implementation planning

---

## GPU Resources

### NVIDIA T1000

**Specifications**:
- **Model**: NVIDIA T1000 (Turing architecture)
- **VRAM**: 4,096 MiB (4 GB)
- **Driver Version**: 580.95.05
- **CUDA Version**: 13.0
- **TDP**: 50W
- **Persistence Mode**: On ✅

**Current Status**:
- Temperature: 37°C (idle)
- Memory Usage: 10 MiB / 4096 MiB (0.2% used)
- GPU Utilization: 0%
- Available Memory: **~4.0 GB** ✅

**Key Characteristics**:
- **Professional GPU** (Quadro line, rebranded T-series)
- Designed for CAD/visualization workloads
- Lower memory vs gaming GPUs but high reliability
- Excellent for development/testing

**Implications for JAXTrace V3**:
- ⚠️ **Memory constraint**: 4 GB (not 8 GB as assumed in Phase 0!)
- Need to reduce memory estimates by 50%
- ThreadedA mesh analysis needs revision
- May need to reduce max particles or use gradient checkpointing

---

## CPU Resources

### Intel Core i7-12700 (12th Gen Alder Lake)

**Specifications**:
- **Cores**: 12 (8 P-cores + 4 E-cores)
- **Threads**: 20 (with Hyper-Threading on P-cores)
- **Architecture**: x86_64 (Alder Lake hybrid)
- **Current Clock**: ~2.9 GHz (56% scaling)
- **NUMA Nodes**: 1

**Performance Characteristics**:
- **High-performance P-cores**: 8 cores for latency-sensitive tasks
- **Efficient E-cores**: 4 cores for throughput tasks
- Excellent single-thread performance (IPC improvements vs 11th gen)
- Good multi-threaded performance (20 threads)

**Implications for JAXTrace**:
- Strong CPU baseline for comparison
- Good for preprocessing (mesh loading, analysis)
- Can parallelize CPU tracker across 20 threads
- Hybrid architecture may affect thread pinning

---

## System Memory

### RAM

**Specifications**:
- **Total**: 31 GiB (32 GB installed, ~1 GB reserved)
- **Available**: 18 GiB
- **Used**: 12 GiB
- **Buffers/Cache**: 16 GiB
- **Swap**: 511 MiB (426 MiB used)

**Implications for JAXTrace**:
- Plenty of RAM for mesh loading/preprocessing
- Can load ThreadedA mesh (898K nodes, 3.5M elements) easily
- No memory constraints for CPU operations
- Can cache multiple meshes simultaneously

---

## Memory Budget Revisions (4 GB GPU)

### Original Estimates (8 GB GPU):
```
ThreadedA mesh (3.5M elements, 898K nodes):
  Mesh data: 140.5 MB
  Particles (1M): 27.7 MB
  Total: 168.2 MB
  Overhead: 7.8 GB free
```

### Revised Estimates (4 GB GPU):

**Available**: ~3.9 GB (after driver overhead)

**ThreadedA Mesh**:
- Mesh data (static): 140.5 MB
- JAX compilation cache: ~500 MB (conservative)
- **Remaining for particles**: 3.9 GB - 140.5 MB - 500 MB = **3.26 GB**

**Maximum Particle Count**:
```
Particle memory per 1M: 27.7 MB
Max particles: 3260 MB / 27.7 MB = 117M particles ✅
```

**Realistic Target** (with safety margin):
- Conservative: 50M particles (1.4 GB) → 2.1 GB free for intermediates
- Aggressive: 100M particles (2.8 GB) → 400 MB free for intermediates

**Conclusion**: 4 GB is sufficient for production workloads, but requires careful memory management.

---

## JAX Device Configuration

Let me check JAX's view of the devices:

```python
import jax
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
```

**Expected Output**:
- GPU device: `CudaDevice(id=0)` with 4 GB
- CPU fallback available

---

## Performance Expectations

### GPU (NVIDIA T1000)

**Compute Capability**: 7.5 (Turing)
- **FP32 Performance**: ~2.5 TFLOPS
- **Memory Bandwidth**: ~160 GB/s
- **Tensor Cores**: No (not RTX variant)

**Expected Speedup vs CPU**:
- Embarrassingly parallel: 10-20× (limited by memory bandwidth)
- With memory transfers: 5-10×
- With vmap overhead: 3-5×

**Bottlenecks**:
- Memory bandwidth (not compute)
- Host-device transfers (PCIe)
- Small batch sizes (underutilization)

### CPU (i7-12700)

**Serial Performance**: Excellent (modern Alder Lake)
- Single-thread: ~4.5 GHz boost
- Cache: 25 MB L3

**Parallel Performance**: Good (20 threads)
- Multi-threaded element search: 20× single-thread
- NumPy operations: Multi-threaded BLAS

**Expected Baseline**:
- Element search: ~1M particles/second/thread
- 20 threads: ~20M particles/second
- GPU needs >20× to beat this!

---

## Disk/Storage

**Not explicitly checked**, but from workspace path:
- Working directory: `/home/arhashemi/Workspace/welding/JAXTrace`
- Mesh data: `/home/arhashemi/Workspace/welding/Edgar/ThreadedA/`

**Assumptions**:
- SSD (reasonable I/O for mesh loading)
- Sufficient space for outputs

---

## Network (for distributed computing)

**Not relevant** for current single-GPU implementation.

**Future considerations** (Phase 9+):
- Multi-GPU via JAX pmap
- Would require additional GPUs or cloud instances

---

## Updated Phase 1+ Constraints

### Memory Constraints (4 GB GPU, not 8 GB)

**Phase 1** (Flat Arrays):
- Target: Load ThreadedA mesh (140 MB) ✅
- Validation: Memory usage < 200 MB ✅

**Phase 4** (Multi-Level Search):
- Block element lists: (4 blocks, 10K max) = 160 KB ✅
- Test with 10K particles (277 KB) ✅

**Phase 6** (Time Integration):
- Test with 100K particles (2.8 MB) ✅
- Production: 1M particles (28 MB) ✅

**Phase 7** (Block Batching):
- May need to batch particles in groups of 10M (not 50M)
- Batch size: 10M × 28 MB = 280 MB per batch

### Updated Success Criteria

**Phase 6 Target**:
- 1M particles, 100 timesteps: < 1 minute on GPU ✅ Achievable
- 10M particles, 100 timesteps: < 10 minutes on GPU ✅ Achievable

**Phase 7 Target**:
- 50M particles, 100 timesteps: < 30 minutes (5 batches × 6 min)
- 100M particles, 100 timesteps: < 60 minutes (10 batches × 6 min)

---

## Recommendations

### Immediate (Phase 1-4)

1. **Monitor GPU memory carefully**
   - Use `nvidia-smi dmon` during tests
   - Add memory tracking to tests
   - Fail fast if approaching 3.5 GB

2. **Reduce max_elements_per_block if needed**
   - Current: 10K elements
   - If memory tight: 5K elements (80 KB vs 160 KB)

3. **Start with small particle counts**
   - Phase 4: Test with 1K particles
   - Phase 6: Test with 10K particles
   - Phase 7: Scale to 1M, then 10M

### Medium-term (Phase 5-7)

4. **Use gradient checkpointing**
   - JAX's `jax.checkpoint` for RK4 stages
   - Trade compute for memory (acceptable on T1000)

5. **Consider mixed precision**
   - Use FP32 for positions (accuracy critical)
   - Use FP16 for intermediate calculations (save memory)

6. **Batch processing**
   - Process particles in chunks of 10M (not 50M)
   - Keep each batch < 500 MB

### Long-term (Phase 8+)

7. **Adaptive grid refinement is CRITICAL**
   - T1000's 4 GB makes static grids limiting
   - Adaptive grid reduces memory by concentrating blocks

8. **Profile memory, not just time**
   - Memory bandwidth is bottleneck on T1000
   - Optimize for fewer memory transfers
   - Cache reuse is critical

---

## Testing Strategy

### Development (Phases 1-3)

**Target**: Correctness on tiny meshes
- Tiny mesh (162 elements)
- 100 particles
- GPU memory: < 10 MB

### Validation (Phases 4-6)

**Target**: Functionality on small meshes
- Small mesh (~6K elements)
- 10K particles
- GPU memory: < 100 MB

### Performance (Phase 7)

**Target**: Speed on ThreadedA
- ThreadedA mesh (3.5M elements)
- 1M particles
- GPU memory: ~170 MB

### Stress Testing (Phase 8+)

**Target**: Scale limits
- ThreadedA mesh
- 10M-100M particles (batched)
- GPU memory: < 3.5 GB

---

## Conclusion

**Hardware Profile**:
- ✅ GPU: NVIDIA T1000 (4 GB VRAM) - Sufficient with care
- ✅ CPU: Intel i7-12700 (20 threads) - Strong baseline
- ✅ RAM: 32 GB - No constraints
- ⚠️ GPU Memory: 50% less than assumed (4 GB not 8 GB)

**Impact on V3 Plan**:
- **Minor adjustments needed** (Phase 0 estimates were 168 MB → fits in 4 GB)
- More careful memory management required
- May limit max particles to 10M per batch (not 50M)
- All phases remain achievable

**Recommendation**: Proceed with Phase 1 as planned, with added GPU memory monitoring.
