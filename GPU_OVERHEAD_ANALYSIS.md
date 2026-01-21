# GPU Overhead Analysis - Fully Fused RK4 Time-Dependent

**Date**: 2026-01-08
**Status**: 🔍 **PERFORMANCE ANALYSIS** - Identifying GPU overhead sources
**Symptom**: Production test running slow with 100% GPU utilization

---

## Executive Summary

### Findings

**CRITICAL GPU OVERHEAD SOURCES IDENTIFIED:**

1. **Nested vmap inside vmap** (Line 182 in rk4_fully_fused_timedep.py)
   - Outer vmap: Over all particles (line 386)
   - Inner vmap: Over 4 neighbors per particle (line 182)
   - Impact: **4× unnecessary parallelization** for L1 neighbor checks

2. **Multiple lax.fori_loop calls per particle**
   - L1 adaptive hop loop: 6 iterations × N particles
   - L2 neighbor search: 27 octants × N particles (or 125 for enhanced)
   - Per-leaf element search: leaf_capacity iterations × leaves × particles
   - Impact: **Thousands of loop iterations per timestep**

3. **Repeated Morton neighbor computation**
   - Hierarchical search: 27 octants at depth-7 + 27 octants at depth-6 = 54 octant searches
   - Each octant: up to 8 leaves to search
   - Each leaf: up to leaf_capacity elements to check
   - Impact: **~400-500 element checks per particle that needs L2**

4. **Data-independent execution overhead**
   - jnp.where evaluates BOTH branches even when condition is known
   - Hierarchical search compiles depth-6 search for ALL particles (even if depth-7 found)
   - Impact: **50% wasted compute** in hierarchical mode

---

## Detailed Analysis

### 1. Nested vmap in L1 Search (CRITICAL)

**Location**: `jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py:182`

```python
def search_l1_single(pos: jax.Array, start_elem_id: jax.Array) -> jax.Array:
    # ... adaptive hop logic ...

    for hop_idx in range(6):  # UNROLLED LOOP - GOOD
        # ...

        def check_neighbor(elem_id):
            valid = elem_id >= 0
            inside = jnp.where(
                valid,
                point_in_tet_gpu(pos, elem_id, connectivity, node_positions),
                False
            )
            return jnp.where(inside, elem_id, jnp.int32(-1))

        # NESTED VMAP (line 182)
        found_in_neighbors = jax.vmap(check_neighbor)(neighbors)  # ← OVERHEAD!
```

**Why this is overhead:**
- **Outer context**: `search_l1_single` is ALREADY inside `vmap(rk4_single_particle)` (line 386)
- **Inner vmap**: Checks 4 neighbors per particle
- **Total parallelization**: N_particles × 4 neighbors = 4N parallel operations
- **Problem**: GPU threads are already saturated with N particles, adding 4× dimension doesn't help

**Expected behavior:**
- Sequential neighbor checking would be FASTER (no vmap overhead)
- Early-exit optimization impossible with vmap (all 4 neighbors checked even if first one contains particle)

**Impact estimation:**
- With 225,000 particles: 900,000 parallel point-in-tet checks per hop
- With 6 hops max: 5.4 million checks per RK4 stage
- With 5 RK4 stages: **27 million checks per timestep** (L1 only!)

---

### 2. lax.fori_loop Overhead in L2 Search

**Location**: Multiple L2 search functions in `morton_global_search.py`

#### 2.1 Standard Neighbor Search (27 octants)

```python
def search_L2_morton_neighbors_single(pos, mesh_gpu):
    # ...

    # Search all 27 neighbor octants (line 739-744)
    final_elem_id, final_found = lax.fori_loop(
        0,
        27,  # 27 iterations per particle
        search_neighbor_octant,
        init_state
    )
```

**Each octant iteration:**
- Decodes Morton prefix
- Looks up prefix table
- Searches up to 3 leaves (unrolled)
- Each leaf: searches up to `leaf_capacity` elements (another fori_loop!)

**Nested loop structure:**
```
For each particle:
  For each of 27 octants:                    ← lax.fori_loop
    For each of 3 leaves:                    ← unrolled
      For each of leaf_capacity elements:    ← lax.fori_loop (in search_in_leaf_global)
        point_in_tet check
```

**Cost per particle needing L2:**
- 27 octants × 3 leaves × ~200 elements/leaf = **~16,200 operations**

#### 2.2 Enhanced 5×5×5 Search (125 octants)

```python
def search_5x5x5_outer_shell(pos, mesh_gpu, current_elem, already_found):
    # ...

    # Search all 125 octants (line 861-866)
    final_elem_id, final_found = lax.fori_loop(
        0,
        125,  # 125 iterations per particle!
        search_neighbor_octant,
        init_state
    )
```

**Impact:**
- 98 outer shell octants (skips inner 27)
- But loop bounds are 125 (masks 27 with conditionals)
- **Worst case**: 125 × 3 leaves × 200 elements = **75,000 operations per particle**

#### 2.3 Hierarchical Search (Multi-depth)

```python
def search_L2_morton_hierarchical_single(pos, mesh_gpu):
    # Search at depth 7 (fine-grained)
    result_depth_7 = search_at_depth(7)  # 27 octants

    # CRITICAL: jnp.where evaluates BOTH branches!
    result_final = jnp.where(
        result_depth_7 >= 0,
        result_depth_7,
        search_at_depth(6)  # Another 27 octants (ALWAYS executed!)
    )
```

**Data-independent execution overhead:**
- `jnp.where` compiles AND executes both branches
- Even if depth-7 finds element, depth-6 search STILL runs
- **Wasted compute**: 50% of all depth-6 searches

**Inner structure:**
```python
def search_at_depth(depth):
    # ...
    final_elem_id, final_found = lax.fori_loop(
        0, 27, search_neighbor_octant, init_state  # ← Loop 1
    )

    # Inside search_neighbor_octant:
    elem_neighbor, _ = lax.fori_loop(
        0, 8,  # Up to 8 leaves per prefix
        search_multi_leaf,  # ← Loop 2 (nested!)
        init
    )

    # Inside search_multi_leaf:
    # Calls search_in_leaf_global which has ANOTHER fori_loop!
    # ← Loop 3 (triple nesting!)
```

**Total nested loops:**
```
For each particle:
  Depth 7:
    For 27 octants:                          ← Loop 1
      For 8 leaves:                          ← Loop 2
        For leaf_capacity elements:          ← Loop 3
          point_in_tet
  Depth 6 (ALWAYS executed):                 ← 50% waste
    For 27 octants:                          ← Loop 1
      For 8 leaves:                          ← Loop 2
        For leaf_capacity elements:          ← Loop 3
          point_in_tet
```

**Worst-case operations per particle:**
- Depth-7: 27 × 8 × 200 = 43,200
- Depth-6: 27 × 8 × 200 = 43,200
- **Total**: 86,400 operations per particle needing L2

---

### 3. Per-Leaf Element Search Loop

**Location**: `morton_global_search.py:455-513`

```python
def search_in_leaf_global(pos, leaf_id, mesh_gpu):
    """Search within a single leaf."""
    start = mesh_gpu.leaf_start[leaf_id]
    length = mesh_gpu.leaf_length[leaf_id]

    def body(j, found_elem):
        active = (found_elem == -1) & (j < length)
        # ...
        inside = jnp.where(
            active,
            point_in_tet_gpu(pos, elem_id, connectivity, node_positions),
            False
        )
        return jnp.where(inside & active, elem_id, found_elem)

    # Bounded loop with fixed upper bound
    found_elem = lax.fori_loop(0, mesh_gpu.leaf_capacity, body, init)
    return found_elem
```

**Why this is overhead:**
- Loop ALWAYS runs `leaf_capacity` iterations (e.g., 200)
- Even if leaf only has 10 elements, runs 200 iterations
- Uses masking (`active = j < length`) instead of early exit
- **Cannot early-exit** when element found (JAX limitation)

**Impact:**
- Each leaf search: 200 iterations (most are masked no-ops)
- With 27 octants × 3 leaves: 600 × 200 = **120,000 iterations per particle**
- Only ~600 are actual element checks, rest are masked overhead

---

### 4. RK4 Stage Multiplication

**All above overhead × 5 stages per timestep:**

```python
def rk4_single_particle(pos, elem_id):
    # Stage 1: k1
    elem_k1 = search_l0_l1_l2_single(pos, elem_id)          # ← Full hierarchy
    vel_k1 = interpolate_velocity_single(pos, elem_k1, ...)

    # Stage 2: k2
    elem_k2 = search_l0_l1_l2_single(pos_k1, elem_k1)       # ← Full hierarchy
    vel_k2 = interpolate_velocity_single(pos_k1, elem_k2, ...)

    # Stage 3: k3
    elem_k3 = search_l0_l1_l2_single(pos_k2, elem_k2)       # ← Full hierarchy
    vel_k3 = interpolate_velocity_single(pos_k2, elem_k3, ...)

    # Stage 4: k4
    elem_k4 = search_l0_l1_l2_single(pos_k3, elem_k3)       # ← Full hierarchy
    vel_k4 = interpolate_velocity_single(pos_k3, elem_k4, ...)

    # Stage 5: Final position
    elem_final = search_l0_l1_l2_single(pos_final, elem_k4) # ← Full hierarchy
```

**Multiplier**: All L1/L2 overhead × 5 per timestep

---

## Overhead Quantification

### Per-Particle Cost Estimation

**Assumptions:**
- 100% particles hit L1 at some stage (velocity crossing)
- 30% particles hit L2 at some stage (mesh boundary/refinement)
- L2 method: 'neighbors' (standard 27 octants)
- Leaf capacity: 200
- Average leaf length: 100

**L0 (cached element):**
- 1 point-in-tet check
- Cost: ~50 FLOPs

**L1 (multi-hop neighbors):**
- Nested vmap over 4 neighbors: 4 parallel point-in-tet
- 6 hops max (adaptive)
- Cost per hop: 4 × 50 = 200 FLOPs
- Total: 6 × 200 = **1,200 FLOPs per particle hitting L1**

**L2 (Morton neighbors):**
- 27 octants
- 3 leaves per octant avg
- 200 iterations per leaf (leaf_capacity)
- ~100 active iterations per leaf (actual length)
- Cost: 27 × 3 × 100 × 50 = **405,000 FLOPs per particle hitting L2**

**Per RK4 timestep:**
- L0: 5 stages × 225k particles × 50 FLOPs = 56M FLOPs
- L1: 5 stages × 225k particles × 1,200 FLOPs = **1.35B FLOPs**
- L2: 5 stages × 67.5k particles (30%) × 405k FLOPs = **137B FLOPs**

**Total per timestep: ~138 billion FLOPs** (L2 dominates)

### GPU Throughput Reality Check

**T4 GPU specs:**
- Peak FP32: 8.1 TFLOPS
- Memory bandwidth: 320 GB/s
- Memory latency: ~100-300 cycles

**Expected time per timestep:**
- Compute-bound: 138 GFLOPS / 8100 GFLOPS = **17 ms**
- Memory-bound: Likely 10-100× slower due to random access pattern

**Actual observation:**
- Production running "too slow" with 100% GPU
- Likely: **Memory-bound** (random element access kills cache)

---

## Root Causes

### 1. vmap Overhead (CRITICAL)

**Problem**: Nested vmap in L1 neighbor checking
- vmap over particles (outer)
- vmap over 4 neighbors (inner)
- Creates 4× parallelization that saturates GPU

**Why it's slow:**
- GPU thread block size typically 256-1024
- With 225k particles, already have thousands of blocks
- Adding 4× neighbor dimension: warp scheduling overhead
- **Cannot early-exit** when neighbor found

### 2. lax.fori_loop Overhead

**Problem**: Fixed-bound loops with masking
- Leaf search: Always 200 iterations (even for 10 elements)
- Octant search: Always 27/125 iterations (even if found early)
- Multi-leaf: Always 8 iterations (even if prefix has 1 leaf)

**Why it's slow:**
- **No early exit** in JAX compiled loops
- Masked iterations still consume GPU cycles (branch divergence)
- Loop overhead compounds in nested structures

### 3. Data-Independent Execution

**Problem**: jnp.where always evaluates both branches
- Hierarchical search runs depth-6 even if depth-7 succeeds
- 5×5×5 enhanced search evaluates all tiers

**Why it's slow:**
- 50% wasted compute in hierarchical mode
- Cannot use lax.cond in vmap context (OOM due to compilation)

### 4. Memory Access Pattern

**Problem**: Random element access kills cache locality
- Morton curve provides some locality
- But L2 search jumps across 27+ octants
- Element connectivity lookups scattered across memory

**Why it's slow:**
- GPU caches are small (L1: 64KB, L2: 4MB on T4)
- 3M elements × 4 nodes × 4 bytes = 48 MB connectivity
- 571k nodes × 3 coords × 4 bytes = 6.8 MB positions
- **Cache misses dominate latency**

---

## Performance Bottleneck Identification

### Primary Bottleneck: L2 Search (97% of compute)

**Evidence:**
- L2 cost: 137 GFLOPS (99.3% of total)
- L1 cost: 1.35 GFLOPS (1.0%)
- L0 cost: 0.056 GFLOPS (0.04%)

**Why L2 dominates:**
1. 27-125 octant searches per particle
2. 3 leaves per octant
3. 200 iterations per leaf (even if only 100 active)
4. All this × 5 RK4 stages

### Secondary Bottleneck: Memory Bandwidth

**Random access pattern:**
- Element i → connectivity[i] → 4 node IDs → 4 node positions
- Each particle touches different elements across RK4 stages
- **No cache reuse** between particles

**Bandwidth calculation:**
- Per point-in-tet: 4 nodes × 3 coords × 4 bytes = 48 bytes read
- Per L2 search: 27 × 3 × 100 × 48 bytes = **388 KB per particle**
- 225k particles × 5 stages × 388 KB = **437 GB per timestep**
- At 320 GB/s: **1.37 seconds per timestep** (memory-bound!)

---

## Configuration Issues

### Current Production Settings

```python
# From production_tracking_fully_fused_timedep.py

L2_SEARCH_METHOD = 'neighbors'  # vs 'radius' or 'hierarchical'
L2_SEARCH_RADIUS = 2            # Only used if method='radius'
N_HOPS = 3                      # Base hop count (adaptive up to 6)
ENABLE_L1_SEARCH = True
```

**'neighbors' method overhead:**
- 27 octants (vs ~4-6 leaves for radius=2)
- Prefix table lookups (extra indirection)
- 3 leaves per octant (wasteful if only 1 leaf)

**Adaptive L1 overhead:**
- Volume ratio check every particle
- Median computation over 4 neighbors
- 6 hops for boundary particles (vs 3 normal)

### Comparison with 'radius' Method

**'radius' method (L2_SEARCH_RADIUS=2):**
- Linear search: center ± 2 leaves = 5 leaves total
- Each leaf: 200 iterations
- Total: 5 × 200 = **1,000 iterations per particle**

**'neighbors' method:**
- 27 octants × 3 leaves = 81 leaf searches
- Each leaf: 200 iterations
- Total: 81 × 200 = **16,200 iterations per particle**

**Overhead ratio: 16× more iterations!**

---

## Recommendations (For User Approval)

### HIGH IMPACT (Expected 5-10× speedup)

1. **Switch L2 method to 'radius'**
   ```python
   L2_SEARCH_METHOD = 'radius'
   L2_SEARCH_RADIUS = 3  # Slightly larger to compensate
   ```
   - Reduces L2 iterations: 16,200 → 1,400 (11× reduction)
   - Simpler code path (no prefix tables)
   - Expected: **5-10× faster L2 search**

2. **Remove nested vmap in L1**
   - Replace `jax.vmap(check_neighbor)(neighbors)` with sequential loop
   - Use jnp.where to mask results
   - Allows early-exit logic
   - Expected: **2-3× faster L1 search**

### MEDIUM IMPACT (Expected 1.5-2× speedup)

3. **Reduce adaptive hop count**
   ```python
   N_HOPS = 2  # Down from 3
   # Adaptive: 4 hops for boundaries (down from 6)
   ```
   - Less aggressive multi-hop search
   - Rely more on L2 for difficult cases
   - Expected: **1.5× faster L1**

4. **Optimize leaf search loop bounds**
   - Use actual leaf_length instead of leaf_capacity
   - Requires dynamic loop bounds (harder in JAX)
   - Expected: **1.5-2× faster per-leaf search**

### LOW IMPACT (Expected <1.5× speedup)

5. **Disable L1 for first few stages**
   - L0 likely succeeds for k1, k2 (small position change)
   - Only enable L1 for k3, k4, final
   - Expected: **1.2-1.3× overall speedup**

6. **Reduce particle count for testing**
   ```python
   PARTICLE_GRID_RESOLUTION = (30, 60, 30)  # 54k vs 225k
   ```
   - 4× fewer particles
   - Same per-particle cost, but faster iteration
   - Expected: **4× faster (linear scaling)**

---

## Critical Questions for User

Before proceeding with any modifications:

1. **Which L2 search method are you currently using?**
   - Check production_tracking_fully_fused_timedep.py line ~180
   - 'radius', 'neighbors', or 'hierarchical'?

2. **What is the actual runtime per timestep?**
   - Seconds per RK4 step?
   - GPU utilization pattern (steady 100% or spiky)?

3. **Where is most time spent?**
   - Initial assignment?
   - RK4 stepping loop?
   - VTK export?

4. **Is retention still low even with node deduplication?**
   - If retention is now good (>60%), slowness may be acceptable
   - If retention is still poor (<50%), need algorithmic fix not optimization

---

## Analysis Summary

**GPU is 100% utilized because:**
1. Nested vmap creates 4× unnecessary parallelization (L1)
2. Multiple nested lax.fori_loop iterations (L2: 16k-86k per particle)
3. Data-independent execution (50% wasted compute in hierarchical)
4. Memory-bound random access pattern (cache thrashing)

**But slow because:**
1. L2 search method 'neighbors' is 16× more iterations than 'radius'
2. Cannot early-exit from JAX loops (masked overhead)
3. Memory bandwidth saturated (437 GB/timestep, 320 GB/s limit)

**Recommended fix (no code changes yet):**
- Switch to L2_SEARCH_METHOD = 'radius' (config change only)
- Expected: 5-10× speedup in L2, ~3-5× overall speedup

**Awaiting user confirmation before any code modifications.**
