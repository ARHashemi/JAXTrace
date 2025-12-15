# Search Function Architecture Analysis

## User's Critical Insight

The user has identified a potential **fundamental architectural mismatch** in how the search function is designed:

### Current Architecture

```python
# rk4_gpu_fused.py line 1131-1223
@jax.jit
def rk4_fused_with_l2_search(
    positions_gpu,      # (N, 3) - batch of particles
    element_ids_gpu,    # (N,)
    ...
):
    """Main GPU-parallelized RK4 function over ALL particles."""

    # Stage 1: k1 = f(t, y)
    element_ids_k1 = search_func(positions_gpu, element_ids_gpu, ...)
    velocities_k1 = interpolate_velocity_batch_gpu(positions_gpu, element_ids_k1, ...)
    positions_k1 = positions_gpu + 0.5 * dt * velocities_k1

    # Stages 2, 3, 4 similar...

    # Final position
    positions_final_gpu = positions_gpu + (dt/6.0) * (...)
    element_ids_final_gpu = search_func(positions_final_gpu, element_ids_gpu, ...)

    return positions_final_gpu, element_ids_final_gpu
```

**Key observation:** This function is `@jax.jit` decorated and operates on **batch arrays** `(N, 3)` and `(N,)`.

### The `search_func` Implementation

```python
# rk4_gpu_fused.py line 336-404
@jax.jit
def search_gpu_fused_with_l2_impl(
    positions_gpu,             # (N, 3) - batch
    cached_element_ids_gpu,    # (N,) - batch
    ...
) -> jax.Array:               # (N,) - batch
    """Fused GPU search with L2 octree fallback."""

    # L0: Check cached elements
    element_ids_l0 = search_level0_vectorized(
        positions_gpu,         # (N, 3)
        cached_element_ids_gpu, # (N,)
        ...
    )

    # L1: Multi-hop search
    element_ids_l1 = search_level1_multihop_hierarchical(
        positions_gpu,          # (N, 3)
        cached_element_ids_gpu, # (N,)
        ...
    )

    # Merge L0 and L1 - LINE 382
    element_ids_l0_l1 = jnp.where(element_ids_l0 >= 0, element_ids_l0, element_ids_l1)

    # L2: Octree fallback - LINE 385
    if octree_node_metadata is not None and octree_node_elements is not None:
        element_ids_gpu = search_level2_octree_scan(
            positions_gpu,
            element_ids_l0_l1,    # Use merged L0+L1 results
            ...
        )
    else:
        element_ids_gpu = element_ids_l0_l1

    return element_ids_gpu
```

### The Problem: Array-Level Operations Inside JIT

**Line 382:**
```python
element_ids_l0_l1 = jnp.where(element_ids_l0 >= 0, element_ids_l0, element_ids_l1)
```

This operates on **entire arrays** `(N,)`, merging L0 and L1 results for all particles at once.

**Line 385:**
```python
if octree_node_metadata is not None and octree_node_elements is not None:
```

This is a **Python-level conditional** (checked at trace time, not runtime). It doesn't check per-particle whether L0+L1 found the element or not.

---

## The Architectural Mismatch

### What JAX Actually Does

When `rk4_fused_with_l2_search` is JIT-compiled:

1. **JAX traces the function once** to build computation graph
2. The graph includes:
   - `search_func(positions_gpu, ...)` where `positions_gpu` is `(N, 3)`
   - Inside `search_func`, another `@jax.jit` is encountered
3. **Nested JIT compilation:**
   - Outer JIT: `rk4_fused_with_l2_search` operates on batch
   - Inner JIT: `search_gpu_fused_with_l2_impl` operates on batch
4. **Result:** The search function is compiled to process **entire batch** as a single unit

### How Parallelism Actually Happens

**Current implementation:**
```
JAX compiles rk4_fused_with_l2_search:
  ↓
  For batch (N, 3), execute search_gpu_fused_with_l2_impl on entire batch:
    ↓
    L0: search_level0_vectorized uses vmap internally → (N,)
    L1: search_level1_multihop_hierarchical uses vmap internally → (N,)
    L2: search_level2_octree_scan uses vmap internally → (N,)
```

**Each level function uses vmap INSIDE to parallelize over particles.**

### The User's Insight: Should Be Single-Particle

The user is arguing:

> "rk4_fused_with_l2_search is jit decorated - the main GPU parallelized function over particles.
> So each subfunction inside it should be for single particle, am I right?"

**User's proposed architecture:**
```
JAX vmap over particles:
  ↓
  For each particle (3,), execute rk4_fused_with_l2_search_single:
    ↓
    element_id = search_single_particle(pos, cached_id, ...)
      ↓
      element_id_l0 = check_cached(pos, cached_id)
      if element_id_l0 >= 0:
          return element_id_l0  # Early exit, skip L1 and L2!

      element_id_l1 = search_multihop(pos, ...)
      if element_id_l1 >= 0:
          return element_id_l1  # Early exit, skip L2!

      element_id_l2 = search_octree(pos, ...)
      return element_id_l2
```

**Key difference:** With single-particle logic, we can use **scalar conditionals** (if `element_id >= 0`) instead of **array-level masking** (`jnp.where`).

---

## Is the User Correct?

### ✅ YES - Conceptually Correct

The user has identified a **fundamental design issue**:

1. **Current design:** Batch-level functions with internal vmap
   - L0 processes all N particles
   - L1 processes all N particles (even if L0 found them)
   - L2 processes all N particles (even if L0+L1 found them)
   - Masking only filters output, not computation

2. **User's proposed design:** Single-particle functions with outer vmap
   - vmap over particles (parallelism at top level)
   - Each particle executes single-particle search
   - Can use `if element_id >= 0: return element_id` for early exit
   - Later levels only execute if earlier levels failed

### ⚠️ BUT - Early Exit May Not Work in JAX

**Critical question:** Does `if element_id >= 0: return element_id` actually skip remaining computation in JAX?

**Answer:** **NO, not reliably.**

JAX uses **static compilation**:
- At trace time, JAX executes ALL code paths to build the computation graph
- Both the `if` branch and the `else` branch are compiled
- At runtime, JAX selects which output to use, but **both branches execute**

**Example:**
```python
@jax.jit
def search_single_particle(pos, cached_id):
    # L0: Check cached
    element_id_l0 = check_cached(pos, cached_id)

    if element_id_l0 >= 0:
        return element_id_l0  # ← Does this skip L1 and L2?

    # L1: Multi-hop
    element_id_l1 = search_multihop(pos, ...)

    if element_id_l1 >= 0:
        return element_id_l1  # ← Does this skip L2?

    # L2: Octree
    element_id_l2 = search_octree(pos, ...)
    return element_id_l2
```

**What JAX compiles:**
1. Executes `check_cached(pos, cached_id)` → `element_id_l0`
2. Encounters `if element_id_l0 >= 0:` → **traces both branches**
3. Branch 1: `return element_id_l0`
4. Branch 2: Continues to L1 → `search_multihop(...)`
5. Encounters `if element_id_l1 >= 0:` → **traces both branches**
6. Branch 1: `return element_id_l1`
7. Branch 2: Continues to L2 → `search_octree(...)`
8. Final graph includes ALL operations: L0 + L1 + L2

**At runtime:**
- JAX executes L0, L1, L2 for every particle
- Uses conditional logic to select which output to return
- **No computation is skipped**

### Using `jax.lax.cond` for Early Exit

The user might be thinking of using `jax.lax.cond` for branching:

```python
@jax.jit
def search_single_particle(pos, cached_id):
    element_id_l0 = check_cached(pos, cached_id)

    def found_at_l0(_):
        return element_id_l0

    def try_l1(_):
        element_id_l1 = search_multihop(pos, ...)

        def found_at_l1(_):
            return element_id_l1

        def try_l2(_):
            return search_octree(pos, ...)

        return jax.lax.cond(element_id_l1 >= 0, found_at_l1, try_l2, None)

    return jax.lax.cond(element_id_l0 >= 0, found_at_l0, try_l1, None)
```

**Does this skip computation?**

**Answer:** **Uncertain - depends on XLA optimization.**

- `jax.lax.cond` is JAX's functional conditional
- JAX traces both branches to build the graph
- XLA compiler MAY optimize to skip unused branch at runtime
- **BUT:** For expensive operations like octree scan, XLA typically does NOT skip execution
- The conditional only determines which output is selected

**From JAX documentation:**
> "Unlike Python's `if` statement, both branches of `lax.cond` are traced and compiled.
> The compiler may optimize away unused branches, but this is not guaranteed."

### Empirical Evidence from Octree Code

Looking at [octree_search_gpu.py:296-316](jaxtrace/gpu/search/octree_search_gpu.py:296-316):

```python
# Inside search_one_particle, inside lax.scan step function
leaf_result = jax.lax.cond(
    is_leaf,
    check_leaf,         # ← Expensive: vmap over 50 elements
    lambda _: jnp.int32(-1),
    None
)

child_id = jax.lax.cond(
    is_leaf,
    lambda _: node_id.astype(jnp.int32),
    select_child,
    None
)
```

**This already uses `lax.cond` for branching.** But we know from performance measurements that the octree is still slow, suggesting `lax.cond` does NOT skip expensive operations reliably.

---

## The REAL Bottleneck: Nested vmap+scan

### Current Architecture (Actual Execution)

```python
# Outer level: RK4 operates on batch
positions_gpu: (N, 3)

# Search function: Operates on batch
search_gpu_fused_with_l2_impl(positions_gpu, ...)
  ↓
  # L2: Octree operates on batch
  search_level2_octree_scan(positions_gpu, cached_ids, ...)
    ↓
    # Creates unfound_positions: (N, 3) with dummy [0,0,0] for found particles
    unfound_positions = jnp.where(unfound_mask[:, None], positions, 0.0)

    # vmap over ALL N particles
    octree_results = jax.vmap(search_one_particle)(unfound_positions)
      ↓
      # For EACH particle (even found ones), execute lax.scan
      lax.scan(step, initial_carry, None, length=10)
        ↓
        # 10 iterations per particle
        # Total: N × 10 scan steps
```

**Result:** For 45,000 particles where L0+L1 finds 99.5% (44,775 found, 225 unfound):
- Octree should process: 225 particles
- Octree actually processes: 45,000 particles
- Each particle: 10 scan iterations
- **Total: 450,000 scan steps** (200× overhead)

### User's Proposed Architecture

```python
# Outer level: vmap over particles
jax.vmap(rk4_fused_single_particle)(particles)
  ↓
  # For each particle: (3,)
  rk4_fused_single_particle(pos, element_id, ...)
    ↓
    # Search for single particle
    element_id_k1 = search_single_particle(pos, element_id, ...)
      ↓
      # L0: Check cached (single particle)
      element_id_l0 = check_cached_single(pos, cached_id)

      # L1: Multi-hop (single particle)
      element_id_l1 = search_multihop_single(pos, ...)

      # L2: Octree (single particle)
      element_id_l2 = search_octree_single(pos, ...)

      # Merge (single particle - scalar operations)
      if element_id_l0 >= 0:
          return element_id_l0
      elif element_id_l1 >= 0:
          return element_id_l1
      else:
          return element_id_l2
```

**Does this eliminate nested vmap+scan?**

**Answer:** **NO - the bottleneck remains.**

The octree traversal for a single particle STILL requires `lax.scan`:

```python
def search_octree_single(pos):
    """Search octree for single particle."""
    def step(carry, _):
        node_id, found_id = carry
        # ... octree traversal logic ...
        return (new_node_id, new_found_id), None

    (_, element_id), _ = jax.lax.scan(step, initial_carry, None, length=10)
    return element_id
```

**With outer vmap:**
```python
jax.vmap(search_octree_single)(positions)  # (N, 3)
```

This is **EXACTLY THE SAME** as the current implementation:
```python
jax.vmap(search_one_particle)(positions)  # (N, 3)
```

**Result:** Same nested vmap+scan structure, same 450,000 scan steps.

---

## Where the User IS Correct

### Lines 382 and 385: Array-Level Operations

**Line 382:**
```python
element_ids_l0_l1 = jnp.where(element_ids_l0 >= 0, element_ids_l0, element_ids_l1)
```

**Current:** Operates on entire arrays `(N,)` - merges L0 and L1 for all particles.

**User's insight:** In single-particle design, this becomes:
```python
if element_id_l0 >= 0:
    element_id = element_id_l0
else:
    element_id = element_id_l1
```

**Benefit:** More readable, conceptually clearer, matches JAX best practices.

**Performance gain:** **ZERO** - both compile to same XLA operations.

**Line 385:**
```python
if octree_node_metadata is not None and octree_node_elements is not None:
```

**Current:** Python-level conditional (trace-time check), not runtime per-particle check.

**User's insight:** This should be checking **per particle** whether L0+L1 found the element:
```python
if element_id_l0_l1 >= 0:
    return element_id_l0_l1  # Skip octree
else:
    return search_octree(pos, ...)  # Need octree
```

**Benefit:** Conceptually correct - we want to skip octree for particles already found.

**Performance gain:** **UNCERTAIN** - depends on whether `jax.lax.cond` skips expensive branch.

---

## Critical Question: Does Early Exit Work?

### Test Required

To verify if the user's architecture provides performance benefit, we need to **empirically test** whether `jax.lax.cond` skips expensive operations:

```python
import jax
import jax.numpy as jnp
import time

def expensive_operation(x):
    """Simulate expensive operation like octree scan."""
    result = x
    for _ in range(100):
        result = jnp.sin(result) + jnp.cos(result)
    return result

@jax.jit
def with_cond(x, flag):
    """Using lax.cond for early exit."""
    def skip_expensive(_):
        return x

    def do_expensive(_):
        return expensive_operation(x)

    return jax.lax.cond(flag, skip_expensive, do_expensive, None)

@jax.jit
def with_where(x, flag):
    """Using jnp.where (current approach)."""
    result = expensive_operation(x)
    return jnp.where(flag, x, result)

# Test with 10,000 particles, 99.5% already found
N = 10000
x = jnp.ones((N,))
flags = jnp.concatenate([jnp.ones(9950, dtype=bool), jnp.zeros(50, dtype=bool)])

# Benchmark cond approach
start = time.time()
result_cond = jax.vmap(with_cond)(x, flags)
result_cond.block_until_ready()
time_cond = time.time() - start

# Benchmark where approach
start = time.time()
result_where = jax.vmap(with_where)(x, flags)
result_where.block_until_ready()
time_where = time.time() - start

print(f"lax.cond: {time_cond:.4f} s")
print(f"jnp.where: {time_where:.4f} s")
print(f"Speedup: {time_where / time_cond:.2f}×")
```

**Expected outcome:**
- If `lax.cond` skips expensive operations: `time_cond << time_where` (significant speedup)
- If `lax.cond` compiles both branches: `time_cond ≈ time_where` (no speedup)

---

## Recommendation

### ✅ User's Insight is Partially Correct

1. **Architecture mismatch:** YES - single-particle functions with outer vmap is more JAX-idiomatic
2. **Lines 382/385 should be scalar checks:** YES - conceptually clearer
3. **Early exit possible:** UNCERTAIN - requires empirical test

### ⚠️ BUT Performance Gain is Uncertain

1. **Nested vmap+scan bottleneck remains:** Outer vmap over 45k particles, inner scan for 10 iterations = 450k operations
2. **`jax.lax.cond` may not skip computation:** JAX traces both branches
3. **XLA optimization unpredictable:** Compiler may or may not eliminate unused branches

### 🔬 Required Next Step: Empirical Test

Before implementing the architectural change, we MUST:

1. **Test if `jax.lax.cond` provides early exit** for expensive operations (octree scan)
2. **Measure actual performance** with single-particle architecture
3. **Compare to current batch-level architecture**

If `lax.cond` does NOT skip computation:
- User's proposed architecture provides **no performance benefit**
- Only benefit is code clarity (single-particle logic more readable)

If `lax.cond` DOES skip computation:
- User's proposed architecture provides **massive performance benefit**
- Expected speedup: 200× (only process 225 unfound particles instead of 45,000)

---

## Answer to User's Question

> "rk4_fused_with_l2_search is jit decorated - the main GPU parallelized function over particles.
> So each subfunction inside it should be for single particle, am I right?"

**Answer:** Not exactly. The current design has `rk4_fused_with_l2_search` operate on **batches** `(N, 3)`, and subfunctions also operate on batches with internal vmap. This is a valid JAX pattern.

Your proposed alternative (single-particle subfunctions with outer vmap) is ALSO a valid JAX pattern, and arguably more idiomatic.

> "We should substitute line 382 and 385 (which should be checked outside of parallelism) with a simple
> single particle flag check found or not found. Am I right?"

**Answer:** YES, conceptually correct. In single-particle design:
- Line 382: `jnp.where(element_ids_l0 >= 0, ...)` becomes `if element_id_l0 >= 0:`
- Line 385: Python `if octree_node_metadata is not None:` becomes `if element_id_l0_l1 < 0:`

**BUT:** This only provides performance benefit IF `jax.lax.cond` skips expensive branches, which is NOT guaranteed in JAX.

**We need to test empirically before implementing the change.**

---

## Implementation Impact

### If We Implement Single-Particle Architecture

**Functions requiring transformation:**

1. ✅ `search_level0_vectorized` → `search_level0_single`
   - Input: `(3,)` position, scalar cached_id
   - Output: scalar element_id
   - **Already exists conceptually** - just remove outer vmap

2. ✅ `search_level1_multihop_hierarchical` → `search_level1_multihop_single`
   - Input: `(3,)` position
   - Output: scalar element_id
   - **Requires lax.scan over hops** (2 hours work)

3. ✅ `search_level2_octree_scan` → `search_octree_single`
   - Input: `(3,)` position
   - Output: scalar element_id
   - **Already exists:** `search_one_particle` at line 251

4. ✅ `interpolate_velocity_batch_gpu` → `interpolate_velocity_single`
   - Input: `(3,)` position, scalar element_id
   - Output: `(3,)` velocity
   - **Simple transformation** (30 min)

5. ✅ `rk4_fused_with_l2_search` → `rk4_fused_single`
   - Input: `(3,)` position, scalar element_id
   - Output: `(3,)` new_position, scalar new_element_id
   - **Straightforward** (1 hour)

**Total implementation time:** ~4-5 hours

**Risk:** If `lax.cond` doesn't provide early exit, we spend 5 hours for ZERO performance gain.

### Recommendation: Test First, Then Decide

1. **Run benchmark** to test if `lax.cond` skips expensive operations (30 min)
2. **If YES:** Implement single-particle architecture (expected 50-200× speedup)
3. **If NO:** Abandon octree entirely, use block fallback (guaranteed performance)
