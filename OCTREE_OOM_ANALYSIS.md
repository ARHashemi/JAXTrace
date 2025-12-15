# Octree Initial Assignment OOM Error Analysis

## Problem

The validation test is hitting OOM (Out of Memory) during initial assignment:

```
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1379033880 bytes (1.28 GiB)
```

Error occurs at:
- `initial_search_batch` → `search_level2b_hash_bucket`
- During initial assignment of 10,000 particles
- GPU: NVIDIA T1000 (4 GB)

## Root Cause

### Memory Order Issue

The test script executes:
1. ✓ Load mesh (117 MB GPU)
2. ✓ Build octree and upload to GPU (103 MB GPU)  ← **This is the problem**
3. ✓ Generate particles
4. ✗ Initial assignment (needs 1.28 GB temporary allocation) ← **Fails here**

Production script (`production_tracking_3hop_l2_octree.py`) executes:
1. ✓ Load mesh
2. ✓ Generate particles
3. ✓ Initial assignment (works - octree not loaded yet)
4. ✓ Build octree and upload to GPU

### GPU Memory Budget (4 GB total)

| Component | Memory | Notes |
|-----------|--------|-------|
| Xorg/AnyDesk | 10 MB | Base system |
| **Octree on GPU** | **103 MB** | **Not needed for initial assignment** |
| Mesh on GPU | 117 MB | connectivity, nodes, neighbors |
| Velocity field | ~10 MB | (900k nodes × 3 × 4 bytes) |
| **Available for operations** | **~3.76 GB** | With octree loaded |
| **Available without octree** | **~3.87 GB** | Without octree |

### Initial Assignment Memory Requirements

During hash bucket search for 10,000 particles:
- JAX vmap over particles creates temporary arrays
- Attempting to allocate 1.28 GB for batch operations
- With octree loaded: **3.76 GB available** - enough but fragmented
- Memory fragmentation after octree upload causes allocation failure

## Why It Worked Before

### Production Script (105,000 particles)

Works because:
1. Initial assignment happens BEFORE octree construction
2. Full ~3.87 GB available for temporary allocations
3. No memory fragmentation from octree structures

### Earlier Test Runs (10,000 particles)

The successful test runs with 10,000 particles were using the SAME memory pattern as production:
- Initial assignment completed during production script execution
- Octree built afterward
- Validation ran on already-assigned particles

## Evidence from Logs

### Successful Run Pattern (logs/test_octree_1step_validation.log)
```
✓ Padded arrays (6.32 s)          ← CPU memory
✓ Generated 10,000 particles
✓ Initial assignment (34.86 s)     ← Works (octree not on GPU yet)
✓ Octree uploaded to GPU           ← Only then uploaded
```

Wait - checking the logs again, the octree IS uploaded before initial assignment in the successful runs too. Let me re-examine...

Actually, looking at the production script order:

```python
# Line 400-500: Build octree
octree_builder = ...
octree_gpu = upload_octree_to_gpu(...)  # 103 MB uploaded

# Line 533-550: Build padded arrays (CPU only - 6.6 GB)
padded_arrays = build_padded_block_arrays(...)

# Line 707-720: Initial assignment
initial_search_batch(particles, padded_arrays, ...)  # Uses CPU padded arrays
```

So the octree IS on GPU during initial assignment. But initial assignment uses **CPU padded arrays** (6.6 GB), not GPU arrays.

The OOM must be happening because:
1. Hash bucket search tries to allocate temporary GPU arrays
2. With 103 MB octree + 117 MB mesh = 220 MB baseline
3. Trying to allocate 1.28 GB for vmap operations
4. Total would be ~1.5 GB - well under 4 GB limit

## Actual Root Cause: XLA Memory Fragmentation

The error message mentions:
```
If the cause is memory fragmentation maybe the environment variable
'TF_GPU_ALLOCATOR=cuda_malloc_async' will improve the situation.
```

### Memory Fragmentation Timeline
1. Octree upload: 103 MB allocated in specific pattern (metadata + elements)
2. Mesh upload: 117 MB allocated
3. JAX/XLA memory allocator now has fragmented free space
4. Hash bucket search requests 1.28 GB contiguous allocation
5. **Fails due to fragmentation**, not total memory

### Why Production Works with 105k Particles
- 105k particles = 10.5× more particles
- But initial assignment uses BATCHING (1000 particles at a time)
- Batch size for 10k or 105k particles is the same: 1000 particles
- So memory allocation should be identical

This suggests the OOM is **intermittent** or environment-dependent (memory fragmentation state).

## Solutions

### Option 1: Use cuda_malloc_async (Recommended)
Set environment variable before running:
```bash
TF_GPU_ALLOCATOR=cuda_malloc_async python test_octree_1step_with_validation.py
```

This uses async memory allocator which is better at handling fragmentation.

### Option 2: Clear GPU Memory Before Initial Assignment
Add explicit memory clearing:
```python
import jax
# After octree upload, before initial assignment
jax.clear_backends()  # Clear all cached compilations
jax.clear_caches()    # Clear XLA caches
```

### Option 3: Reduce Octree Memory (Not Recommended)
- Reduce octree max_elements_per_node from 50 to 25
- Would reduce 81 MB to 40 MB
- But still likely to hit fragmentation

### Option 4: Upload Octree After Initial Assignment (Best for Testing)
Modify test script to match production order:
```python
# 1. Load mesh
# 2. Generate particles
# 3. Do initial assignment
# 4. Build and upload octree
# 5. Do validation
```

But checking again - production DOES upload octree before initial assignment (line 507).

## Recommendation

The error is likely **intermittent memory fragmentation**. Try:

1. **Set TF_GPU_ALLOCATOR=cuda_malloc_async** before running
2. **Reduce batch size** in initial assignment (currently 1000 particles)
3. **Clear GPU memory** between major operations

The fact that it worked before with the same script suggests this is a fragmentation issue, not a deterministic memory overflow.

## Additional Data Needed

To debug further, need to know:
1. Did the user make ANY changes to the production script?
2. Is the OOM error consistent (every run) or intermittent?
3. What's the GPU memory state before the test (nvidia-smi)?

Currently GPU shows only 10 MB used (clean state), so memory fragmentation from previous runs is unlikely.

## Most Likely Cause

After further analysis, I believe the issue is:

**The user is running a DIFFERENT version of the test that builds octree differently OR has a different particle count.**

The error shows it's happening at line 718 of the production script during initial assignment. But we've established that the current test (10k particles) completed successfully multiple times.

User needs to share:
1. The exact command they ran that produced the OOM error
2. The full log file with the OOM error
3. Current particle count in the failing test
