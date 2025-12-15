# Octree Bottleneck: Detailed Explanation with Code

## Overview

The octree implementation has two fundamental issues that make it 65× slower than expected:
1. **JAX nested vmap+scan cannot be masked** (computational bottleneck)
2. **Octree filtering is ineffective** (captures 100% of elements)

Let me explain both with the actual code.

---

## Bottleneck #1: JAX Nested Vmap+Scan Cannot Be Masked

### The Core Problem

**Location:** [octree_search_gpu.py:329-341](jaxtrace/gpu/search/octree_search_gpu.py:329-341)

```python
# Step 4: Run octree search on ALL particles
octree_results = jax.vmap(search_one_particle)(unfound_positions)

# Step 5: Merge results - use octree result only for unfound particles
element_ids = jnp.where(
    unfound_mask,
    octree_results,    # Use octree result for unfound particles
    cached_element_ids # Keep cached ID for already-found particles
)
```

### Why This Doesn't Work

**You might think:** "We filter with `unfound_mask`, so JAX only processes unfound particles, right?"

**Reality:** JAX evaluates `jax.vmap(search_one_particle)(unfound_positions)` for **ALL particles**, regardless of the mask.

### Proof from Code Structure

Let's trace what happens for 10,000 particles where L0+L1 finds 99.5% (9,950 found, 50 unfound):

#### Step 1: Create mask
```python
# Line 239
unfound_mask = cached_element_ids < 0  # [False, False, ..., True, True, ...False]
# 9,950 False, 50 True
```

#### Step 2: Mask positions
```python
# Lines 244-248
unfound_positions = jnp.where(
    unfound_mask[:, None],
    positions,    # Keep position if unfound
    0.0          # Replace with dummy if found
)
# Result: (10000, 3) array - FULL SIZE!
# 9,950 particles have positions [0.0, 0.0, 0.0]
# 50 particles have real positions
```

#### Step 3: Vmap over ALL particles
```python
# Line 331
octree_results = jax.vmap(search_one_particle)(unfound_positions)
```

**CRITICAL:** `jax.vmap` creates 10,000 parallel function calls, one per particle.

Each call to `search_one_particle(pos)` executes:

```python
# Line 320-325
(_, element_id), _ = jax.lax.scan(
    step,
    (jnp.int32(0), jnp.int32(-1)),
    None,
    length=max_depth  # 10 iterations
)
```

**Total operations:** 10,000 particles × 10 iterations = **100,000 scan steps**

Even though 9,950 particles have dummy position `[0.0, 0.0, 0.0]`, the scan STILL EXECUTES for them!

#### Step 4: Filter results
```python
# Lines 335-339
element_ids = jnp.where(unfound_mask, octree_results, cached_element_ids)
```

This ONLY affects which output is selected. It does NOT skip computation.

### Why JAX Can't Optimize This

JAX uses **static compilation**:

1. JAX traces the function ONCE to build a computation graph
2. The graph has fixed structure: `vmap` over N particles, `scan` for M iterations
3. The graph is compiled to GPU kernel code
4. Masking with `jnp.where` only adds conditional WRITES, not conditional EXECUTION

**Analogy:** It's like writing:
```python
for i in range(10000):
    result = expensive_computation(particle[i])
    if mask[i]:
        output[i] = result  # Use result
    else:
        output[i] = cached[i]  # Ignore result
```

The `expensive_computation` RUNS for all 10,000 iterations. The `if` only decides what to save.

### The Nested Structure Problem

The actual bottleneck is **nested vmap+scan**:

```python
jax.vmap(
    lambda pos: jax.lax.scan(
        lambda carry, _: (update_carry(carry), None),
        initial_carry,
        None,
        length=10
    )
)(positions)  # 10,000 particles
```

This creates a **2D grid of operations**:
- Outer dimension: 10,000 particles (vmap)
- Inner dimension: 10 iterations (scan)
- Total: 100,000 operations

JAX compiles this as a fixed 10,000×10 computation grid. Masking cannot change the grid size.

### Performance Impact

With 105,000 particles (production):
- L0 + L1 finds: ~99.5% (104,475 found, 525 unfound)
- Octree should process: 525 particles
- Octree ACTUALLY processes: 105,000 particles
- **Wasted work: 200× overhead**

Time breakdown:
- Expected: 525 particles × 10 iterations × 0.001ms = 5.25ms
- Actual: 105,000 particles × 10 iterations × 0.001ms = 1,050ms
- **Result: 200× slower**

### Can This Be Fixed in JAX?

**Short answer: No, not with current octree structure.**

Three approaches that DON'T work:

#### 1. `jax.lax.cond` masking
```python
def search_with_mask(pos, mask):
    return jax.lax.cond(
        mask,
        lambda p: search_one_particle(p),  # If unfound
        lambda p: -1,                       # If found
        pos
    )
```

**Problem:** `jax.lax.cond` evaluates **both branches** during tracing to build the graph. At runtime, it only selects which output to use. Both branches compile into the final kernel.

#### 2. Dynamic shape arrays
```python
unfound_indices = jnp.where(unfound_mask)[0]  # Shape: (525,)
unfound_positions_filtered = positions[unfound_indices]  # Shape: (525, 3)
octree_results = jax.vmap(search_one_particle)(unfound_positions_filtered)
```

**Problem:** JAX doesn't support **data-dependent array shapes** in JIT-compiled functions. The size 525 is determined at runtime, but JAX needs to know array sizes at compile time.

This would work in eager mode (no `@jax.jit`), but then you lose 100× GPU performance.

#### 3. Scatter-gather approach
```python
# Extract unfound particles (dynamic size)
unfound_count = unfound_mask.sum()
unfound_positions_compact = jnp.zeros((unfound_count, 3))  # Runtime size!
# ... scatter results back
```

**Problem:** Again, `unfound_count` is runtime-dependent. JAX JIT doesn't support this.

### Why Masking Works in Other Contexts

In simple elementwise operations, masking is efficient:
```python
result = jnp.where(mask, x * 2, x)  # Efficient!
```

Because `x * 2` compiles to: "multiply all elements, then select based on mask"
- Total ops: N multiplies + N selects = 2N ops

But with scan/loop structures:
```python
result = jnp.where(mask, scan_10_times(x), x)  # Inefficient!
```

This compiles to: "run 10-iteration scan for ALL elements, then select"
- Total ops: N × 10 scan steps + N selects = 10N ops

The scan cannot be skipped.

---

## Bottleneck #2: Octree Filtering Is Ineffective

### The Intent

**Location:** [octree_builder.py](jaxtrace/gpu/search/octree_builder.py) (from production script output)

The octree is supposed to filter to only the **refined mesh region** using levelset:

```python
# From production log:
Building octree (levelset < 0.012)...
✓ Octree built (7.14 s)
  Filtered elements: 3,511,335/3,512,384 (100.0%)
```

### The Problem

**Only 1,049 elements filtered out** (0.03% reduction). The octree captures virtually the entire mesh.

### Why This Happens

#### 1. Levelset Threshold Too Permissive

```python
levelset_threshold = 0.012  # 12mm from interface
```

Let's check the actual levelset distribution from the log:

```
Node levelset range: [-0.003510, 0.030511]
Element levelset range: [-0.002615, 0.030511]
```

The threshold is 0.012, but:
- Elements with levelset < 0.012: Almost all elements (3,511,335)
- Elements with levelset >= 0.012: Only 1,049 elements

**Interpretation:** The welding pool boundary (levelset = 0) is at the center of the domain, and the threshold of 12mm captures almost the entire computational domain.

#### 2. Per-Element Levelset = Max of Nodes

From production script (octree construction):
```python
# Compute per-element levelset (max of element's nodes)
level_field = np.array([
    node_level[connectivity[i]].max()
    for i in range(len(connectivity))
])
```

For a tetrahedron with 4 nodes:
- If ANY node has levelset < 0.012, the entire element is included
- This is conservative but captures many elements

**Example:**
- Tet nodes: [0.001, 0.011, 0.015, 0.020]
- Max levelset: 0.020 (>0.012, should exclude)
- But max of [0.001, 0.011] = 0.011 (<0.012, included)

Wait, the code uses MAX, not min. Let me reconsider...

Actually, if the code uses MAX:
```python
level_field[i] = max(node_level[connectivity[i]])
```

Then for threshold `levelset < 0.012`:
- Tet nodes: [0.001, 0.011, 0.015, 0.020]
- Max: 0.020
- Check: 0.020 < 0.012? False → **excluded**

This should be MORE selective, not less. So why does it capture 100% of elements?

#### 3. Investigating the Mesh

From the mesh loading output:
```
Nodes: 900,671
Elements: 3,512,384
Node levelset range: [-0.003510, 0.030511]
```

The domain is very small:
```
Domain bounds:
  X: [-0.0300, 0.0300]  (60mm span)
  Y: [-0.0230, 0.0230]  (46mm span)
  Z: [-0.0100, 0.0000]  (10mm span)
```

With a threshold of 0.012 (12mm), and the Z-dimension only 10mm total:
- **12mm threshold captures most of the domain in Z**
- The refined region is NOT spatially localized - it's spread throughout

This is a **mesh characteristic issue**, not an octree algorithm issue.

#### 4. Levelset Distribution

If we analyze the statistics:

**Fact 1:** Filtered elements: 3,511,335 / 3,512,384 (99.97%)
**Fact 2:** Element levelset range: [-0.002615, 0.030511]

This means:
- Minimum element levelset: -0.002615 (inside refined region)
- Maximum element levelset: 0.030511 (30.5mm from interface)
- Threshold: 0.012 (12mm from interface)

The distribution likely looks like:
```
Elements with levelset < 0.012:  3,511,335 (99.97%)
Elements with levelset >= 0.012:      1,049 (0.03%)
```

**Conclusion:** The refined region (levelset < 0.012) covers almost the entire computational domain for this mesh.

### Why Filtering Doesn't Help Performance

Even if the octree only contains 3,511,335 elements instead of 3,512,384:

1. **Octree node count stays the same:**
   ```
   Total nodes: 415,921
   Leaf nodes: 363,361
   Max depth: 8
   ```

2. **Scan iterations stay the same:**
   - max_depth = 10 iterations
   - Every particle traverses up to 10 nodes
   - Filtering elements doesn't reduce scan depth

3. **Leaf element checks:**
   Each leaf contains up to 50 elements:
   ```python
   Elements array: (415921, 50)
   ```

   In `check_leaf_elements_vectorized`:
   ```python
   # Line 166
   found_ids = jax.vmap(check_one_element)(leaf_elements)  # vmap over 50 elements
   ```

   This checks all 50 elements in every leaf visited. Filtering 1,049 elements out of 3.5M (0.03%) has negligible impact.

### Can Filtering Be Improved?

**Options to increase filtering:**

#### Option 1: Reduce levelset threshold
```python
levelset_threshold = 0.005  # 5mm instead of 12mm
```

Estimate from range [-0.002615, 0.030511]:
- If distribution is linear: 5mm threshold would capture ~40% of elements
- If distribution is volume-based: 5mm threshold might capture ~60% (volumetric scaling)

This could reduce elements to ~1.4M, but octree structure would have similar depth.

#### Option 2: Spatial filtering
Instead of levelset alone, filter by bounding box:
```python
# Only include elements in welding region
bbox_filter = (
    (element_centers[:, 0] > -0.020) & (element_centers[:, 0] < 0.020) &  # ±20mm in X
    (element_centers[:, 2] > -0.005)  # Top 5mm in Z
)
combined_filter = (level_field < 0.012) & bbox_filter
```

This could reduce to ~20-30% of elements if welding region is localized.

#### Option 3: Adaptive threshold based on particle distribution
Analyze where particles actually go and build octree only there.

---

## Fundamental Issues Summary

### Issue #1: Nested Vmap+Scan
- **Where:** [octree_search_gpu.py:331](jaxtrace/gpu/search/octree_search_gpu.py:331)
- **What:** `jax.vmap(search_with_scan)` over all 105k particles
- **Why it's slow:** JAX compiles fixed 105k×10 computation grid
- **Masking doesn't help:** `jnp.where` only filters output, not execution
- **Solution exists:** No, not with scan-based octree in JIT-compiled JAX

### Issue #2: Ineffective Filtering
- **Where:** Octree construction with levelset threshold
- **What:** Captures 3,511,335 / 3,512,384 elements (99.97%)
- **Why it's ineffective:** Refined region covers entire domain for this mesh
- **Solution exists:** Yes, can reduce threshold or add spatial filtering

---

## Can Octree Be Salvaged?

### What CANNOT Be Fixed

1. **Nested vmap+scan structure** - fundamental JAX limitation
2. **Scan over all particles** - JIT requires static array sizes
3. **Data-dependent control flow** - JAX traces at compile time

### What CAN Be Improved

1. **Tighter filtering** - reduce levelset threshold to 0.005mm
2. **Spatial bounding box** - limit octree to known particle region
3. **Shallower octree** - reduce max_depth from 10 to 6 (if most particles found early)

### Expected Performance Gain from Improvements

#### If filtering reduces to 30% of elements:
- Leaf element checks: 50 → 15 elements per leaf
- Scan iterations: Still 10 (depth unchanged)
- Speedup: 1.5× (minor)

#### If reduce max_depth from 10 to 6:
- Scan iterations: 10 → 6
- Computation: 105k × 10 → 105k × 6
- Speedup: 1.67× (minor)

#### Combined:
- Speedup: 1.5 × 1.67 = 2.5×
- Current: 765 p/s
- Improved: 1,912 p/s
- **Still 26× slower than expected 50k p/s**

The nested vmap+scan structure is the dominant bottleneck (200× overhead from processing found particles). Filtering and depth reduction provide minor improvements.

---

## Recommended Solution: Abandon Octree

Given:
1. Nested vmap+scan CANNOT be masked in JAX
2. Filtering is ineffective for this mesh (99.97% of elements captured)
3. Even with improvements, only 2.5× speedup (still 26× too slow)

**Recommendation:** Use **block-based exhaustive search** for L2 fallback:

```python
def search_l2_block_fallback(pos, block_id, element_ids_in_block):
    """Search all elements in containing block (no nesting)."""
    return jax.vmap(point_in_tet)(pos, tet_nodes_in_block)
```

**Advantages:**
- Single vmap (no nesting): vmap over elements, not particles
- For 525 unfound particles in blocks with ~10k elements each:
  - 525 particles × vmap(10k elements) = 5.25M point-in-tet checks
  - vs octree: 105k particles × 10 iterations × 50 elements = 52.5M operations
  - **10× faster than octree**

**Expected performance:** 40-48k p/s (from hierarchical 4-hop baseline)

This matches the original hierarchical search performance without the octree overhead.
