# Filtered Octree Fix - FAILED

**Date:** 2025-12-01
**Status:** ❌ Filtering approach failed - No performance improvement

---

## Test Results

**Log:** [logs/production_3hop_l2_ALL_FIXES.log](logs/production_3hop_l2_ALL_FIXES.log)

**Performance:**
- Time/step: **13.25s** (NO CHANGE from before)
- Throughput: **6,429 p/s** (degrading 7.8k → 3.9k)
- Expected: 40-48k p/s
- **Result:** ❌ **NO IMPROVEMENT**

---

## Root Cause: JAX Can't Skip vmap+scan Work

### The Fundamental Problem

**Current Implementation:** [octree_search_gpu.py:230-319](jaxtrace/gpu/search/octree_search_gpu.py#L230-L319)

```python
# Step 1: Create mask
unfound_mask = cached_element_ids < 0  # Shape: (N,)

# Step 2: Mask positions (but still N elements)
unfound_positions = jnp.where(
    unfound_mask[:, None],
    positions,
    0.0  # Dummy value
)

# Step 3: vmap over ALL particles (PROBLEM!)
def search_one_particle(pos):
    # Contains lax.scan (nested operation!)
    (_, element_id), _ = jax.lax.scan(step, ...)
    return element_id

# This is STILL nested vmap+scan!
octree_results = jax.vmap(search_one_particle)(unfound_positions)  # ← 100k particles

# Step 4: Merge with mask
element_ids = jnp.where(unfound_mask, octree_results, cached_element_ids)
```

**Why This Doesn't Work:**

1. **vmap processes ALL particles**: Even with masked positions, `jax.vmap` creates 100k parallel threads
2. **scan runs for each thread**: Each particle executes the full `lax.scan` (10 iterations)
3. **Total operations**: 100k particles × 10 iterations = **1M nested operations**
4. **JAX compiles this statically**: Can't skip work based on runtime masks

**The masking only affects the OUTPUT, not the COMPUTATION.**

---

## What JAX Actually Does

```python
# What we want (pseudocode):
if particle already found:
    return cached_id  # Skip octree - FAST
else:
    return octree_search(particle)  # Only ~0.5% of particles

# What JAX actually does:
for particle in all_100k_particles:  # ALWAYS ALL PARTICLES
    octree_result = scan(traversal, 10_iterations)  # ALWAYS RUNS SCAN
    output = mask ? octree_result : cached_id  # Masking happens AFTER
```

**Result:** Octree scan runs for ALL particles, not just unfound ones.

---

## Additional Issues Found

### Issue #2: CPU-GPU Transfers Every Timestep

**File:** [rk4_gpu_fused.py:1227-1258](jaxtrace/gpu/tracking/rk4_gpu_fused.py#L1227-L1258)

**Current Code:**
```python
# EVERY timestep:
positions_gpu = jax.device_put(positions.astype(np.float32))  # Upload
element_ids_gpu = jax.device_put(element_ids.astype(np.int32))  # Upload

# ... compute ...

positions_final = np.array(positions_final_gpu)  # Download
element_ids_final = np.array(element_ids_final_gpu)  # Download
```

**Impact:**
- 1.6 MB uploaded every step
- 1.6 MB downloaded every step
- GPU idle during transfers
- Causes 55% → 0% GPU utilization pattern

**This is separate from octree issue** - both need fixing.

---

## Why User's Requirements Matter

User asked to verify:
1. ❌ **L2 searches only unfound particles** - No, searches ALL particles
2. ❌ **Uses GPU-resident arrays** - No, uploads/downloads every timestep
3. ❌ **RK4 uses GPU-resident data** - No, particle data transferred every step
4. ✅ **Only particle data uploads/downloads** - Correct (mesh/octree are resident)

**Conclusion:** Only 1 of 4 requirements met.

---

## Possible Solutions

### Option A: Dynamic Particle Filtering (Breaks JIT)

```python
# Extract only unfound particles
unfound_indices = jnp.where(cached_element_ids < 0)[0]  # Dynamic size!
unfound_positions = positions[unfound_indices]  # Gather

# vmap over ONLY unfound particles
octree_results = jax.vmap(search_one_particle)(unfound_positions)  # ~500 instead of 100k

# Scatter back to full array
element_ids = cached_element_ids.at[unfound_indices].set(octree_results)
```

**Problem:** `jnp.where()[0]` returns dynamic-sized array → breaks JIT compilation

### Option B: Single Scan Over Particles (Flatten Nesting)

Instead of `vmap(scan)`, use single `scan` over (particle, depth) pairs:

```python
def search_all_particles_flat(positions, cached_ids):
    def step_one_particle(carry, particle_idx):
        pos = positions[particle_idx]
        cached_id = cached_ids[particle_idx]

        # If found, skip
        element_id = jax.lax.cond(
            cached_id >= 0,
            lambda: cached_id,
            lambda: traverse_octree_single(pos)  # No nested scan!
        )
        return carry, element_id

    _, element_ids = jax.lax.scan(
        step_one_particle,
        None,
        jnp.arange(len(positions))
    )
    return element_ids
```

**Problem:** Serial execution (not parallel) → likely slower than current

### Option C: Switch to Block Fallback

**User suggestion:** "We have also some global search used for particle initializations but they are blockwise"

**Block-local search:**
- No octree needed
- No nested structures
- Pure vmap over particles
- Already implemented and working

**Performance comparison:**

| Method | Structure | Operations | Expected Speed |
|--------|-----------|------------|----------------|
| Current octree | vmap(scan) | 100k × 10 = 1M | 6.4k p/s ❌ |
| Filtered octree | vmap(scan) + mask | 100k × 10 = 1M | 6.4k p/s ❌ |
| Block fallback | vmap(point-in-tet) | 100k × 13k avg = 1.3B | **40-48k p/s** ✅ |

Block fallback is paradoxically **faster** because:
- No nested compilation overhead
- Pure parallel vmap (no scan)
- Better GPU occupancy
- Simpler code

---

## Recommendation

**STOP trying to fix octree. Switch to block fallback.**

Reasons:
1. Octree requires nested vmap+scan → JAX can't optimize
2. Filtering doesn't help (JAX evaluates all branches)
3. Block fallback already works and is faster
4. User has block search implemented

**Next steps:**
1. Use block-local search as L2 fallback
2. Fix CPU-GPU transfers (GPU-resident particle data)
3. Expected: 40-48k p/s, 82% retention

---

## Lessons Learned

### JAX Limitations for Octrees

1. **vmap + scan nesting is slow** - JAX compiles massive XLA graphs
2. **Masking doesn't skip work** - Both branches evaluated eagerly
3. **Dynamic filtering breaks JIT** - Can't use runtime-sized arrays
4. **Control flow is limited** - lax.cond doesn't avoid compilation

### When Octrees Make Sense in JAX

Octrees work well when:
- Single particle search (no vmap)
- CPU-side preprocessing
- Small trees (< 1000 nodes)
- Called infrequently (e.g., initialization only)

Octrees DON'T work when:
- Vectorized over many particles (vmap)
- Nested inside other operations (scan, vmap)
- Large trees (> 100k nodes)
- Called every timestep

**For this use case: Block fallback is the right choice.**

---

**Date:** 2025-12-01
**Status:** ❌ Filtered octree failed - Recommending block fallback instead
