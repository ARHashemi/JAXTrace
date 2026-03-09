# KD-Tree Vmappable Implementation - Feasibility Analysis

**Date**: 2026-01-28

---

## Executive Summary

**Your assessment is correct**: KD-tree node-based search is the only method that can achieve **100% retention with minimal tests (~64 per particle)**, but the current jaxkd library implementation is not vmappable due to Python control flow in tree traversal.

This document analyzes:
1. **Why jaxkd cannot be vmapped** (detailed code analysis)
2. **Feasibility of implementing vmappable KD-tree** (custom implementation)
3. **Alternative: Morton node-based search** (nodes instead of elements)
4. **Comprehensive comparison** with pros/cons and recommendations

---

## Table of Contents

1. [Why jaxkd Cannot Be Vmapped](#why-jaxkd-cannot-be-vmapped)
2. [Feasibility of Vmappable KD-Tree](#feasibility-of-vmappable-kdtree)
3. [Alternative: Morton Node-Based Search](#alternative-morton-node-based-search)
4. [Comprehensive Comparison](#comprehensive-comparison)
5. [Recommendations](#recommendations)

---

## Why jaxkd Cannot Be Vmapped

### 1.1 Current Usage Pattern (Fails)

```python
# Current implementation in kdtree_node_search.py
def search_L2_kdtree_single(pos, kdtree_gpu, k_nearest=3, max_tests=256):
    """Single particle search - NOT vmappable"""

    # This line is the problem ❌
    nearest_node_ids, distances = kdtree_gpu.kdtree.query(pos, k=k_nearest)

    # Rest is JAX-traceable ✅
    for i in range(k_nearest):
        node_id = nearest_node_ids[i]
        start = kdtree_gpu.node_to_elements_offsets[node_id]
        end = kdtree_gpu.node_to_elements_offsets[node_id + 1]
        for elem_idx in range(start, end):
            elem = kdtree_gpu.node_to_elements_data[elem_idx]
            if point_in_tet(pos, elem):
                return elem
    return -1

# When vmapped in RK4:
@jax.jit
def rk4_step(positions, ...):
    # This fails! ❌
    search_fn = jax.vmap(search_L2_kdtree_single, in_axes=(0, None, ...))
    elem_ids = search_fn(positions, kdtree_gpu, ...)
    # Error: TracerIntegerConversionError
```

### 1.2 Root Cause: `lax.while_loop` in Tree Traversal

From jaxkd source ([tree.py:339-391](file:///home/arhashemi/Workspace/jaxkd/jaxkd/tree.py#L339-L391)):

```python
def _traverse_tree(tree, query, update_func, initial_state, initial_square_radius):
    """Base k-d tree traversal logic"""

    points, indices, split_dims = tree.points, tree.indices, tree.split_dims
    n_points = len(points)

    def step(carry):
        current, previous, state, square_radius = carry
        parent = (current - 1) // 2

        # Update state at current node
        state, square_radius = lax.cond(
            previous == parent, update_func, lambda _, s, r: (s, r),
            current, state, square_radius
        )

        # Determine next node (complex tree traversal logic)
        level = jnp.frexp(current + 1)[1] - 1
        split_dim = jnp.mod(level, points.shape[-1]) if split_dims is None else split_dims[current]
        split_distance = query[split_dim] - points[indices[current], split_dim]
        near_side = jnp.asarray(split_distance > 0, dtype=indices.dtype)
        near_child = 2 * current + 1 + near_side
        far_child = 2 * current + 2 - near_side
        far_in_range = jnp.square(split_distance) <= square_radius

        # Next node selection (data-dependent)
        next = lax.select(
            (previous == near_child) | ((previous == parent) & (near_child >= n_points)),
            lax.select((far_child < n_points) & far_in_range, far_child, parent),
            lax.select(previous == parent, near_child, parent),
        )
        return next, current, state, square_radius

    # THE PROBLEM: while_loop with data-dependent condition ❌
    current = jnp.asarray(0, dtype=indices.dtype)
    previous = (current - 1) // 2
    _, _, state, _ = lax.while_loop(
        lambda carry: carry[0] != (current - 1) // 2,  # Loop until return to root
        step,
        (current, previous, initial_state, initial_square_radius),
    )
    return state
```

**Why this fails when vmapped:**

```
Vmap transformation:
  vmap(lambda pos: kdtree.query(pos, k=3))(positions)
    ↓
  For each position, JAX needs to trace through:
    1. _traverse_tree with lax.while_loop
    2. while_loop condition: carry[0] != (current - 1) // 2
    3. This condition depends on DATA (which nodes visited)
    4. Number of loop iterations varies per particle!

JAX vmap requirements:
  ✅ Fixed control flow (same operations for all inputs)
  ❌ Data-dependent loops (different iterations per input)

Result: TracerIntegerConversionError
  "Cannot convert traced value to int for loop bound"
```

### 1.3 What Actually Happens

```python
# JAX tries to compile:
def vmapped_query(positions):  # (N, 3) positions
    # For position[0]: Tree traversal visits nodes [0, 1, 3, 7, 15, 30, ...]
    #   → 12 iterations
    # For position[1]: Tree traversal visits nodes [0, 2, 5, 11, 23, ...]
    #   → 8 iterations
    # For position[2]: Tree traversal visits nodes [0, 1, 2, 4, 8, 16, 32, ...]
    #   → 15 iterations

    # JAX vmap needs SAME control flow for all inputs
    # But tree traversal has DIFFERENT paths!
    # → Cannot trace → Error
```

### 1.4 Why Batch Queries Work

```python
# This works! ✅
@jax.jit
def batch_query_before_vmap(positions, kdtree):
    # Query KD-tree for ALL positions at once (outside vmap)
    # jaxkd internally vmaps over positions using its own vmap
    nearest_nodes, distances = jk.query_neighbors(kdtree, positions, k=3)
    # nearest_nodes shape: (N, 3)

    # Now vmap the traceable part only
    def search_with_prequeried(pos, nodes):
        # No KD-tree query here! Just use pre-computed nodes
        for node in nodes:
            for elem in node_to_elements[node]:
                if point_in_tet(pos, elem):
                    return elem
        return -1

    # This vmap works because no tree traversal inside ✅
    results = jax.vmap(search_with_prequeried)(positions, nearest_nodes)
    return results
```

**Why this works:**
1. KD-tree query happens **once** for all positions (jaxkd's internal vmap)
2. Results are materialized arrays (no tracers)
3. Then we vmap over the traceable search logic
4. No `lax.while_loop` inside the user's vmap

**Why this doesn't work for RK4:**
```python
# RK4 needs per-particle queries at unpredictable positions
@jax.jit
def rk4_step_vmapped(pos, vel, elem_cache, kdtree):
    # Compute k1, k2, k3, k4 (data-dependent positions)
    k1 = vel
    pos_k2 = pos + 0.5 * dt * k1  # Position depends on DATA

    # Need to query KD-tree at pos_k2 (inside vmap)
    # But pos_k2 is different for each particle!
    # Cannot pre-query because we don't know pos_k2 until runtime!

    elem_k2 = search(pos_k2, kdtree)  # ❌ Requires tree query inside vmap
    # → FAILS
```

---

## Feasibility of Vmappable KD-Tree

### 2.1 The Challenge

**To make KD-tree vmappable, we need to eliminate data-dependent control flow.**

Two approaches:
1. **Fixed-depth traversal** (traverse all nodes up to max depth)
2. **Bounded traversal** (traverse at most N nodes)

### 2.2 Approach 1: Fixed-Depth Traversal

**Idea**: Visit all nodes up to depth D, regardless of distance.

```python
def vmappable_kdtree_query_fixed_depth(query, tree, k=3, max_depth=10):
    """
    Vmappable KD-tree query using fixed-depth traversal.

    Instead of early-exit tree traversal (data-dependent),
    visit ALL nodes up to max_depth (fixed control flow).
    """
    points, indices = tree.points, tree.indices
    n_nodes_to_visit = 2**max_depth - 1  # Full binary tree up to depth D

    # Initialize K-nearest tracking
    nearest_indices = jnp.full(k, -1, dtype=jnp.int32)
    nearest_sq_dists = jnp.full(k, jnp.inf, dtype=jnp.float32)

    def check_node(i, state):
        """Check node i and update nearest neighbors"""
        nearest_idx, nearest_dist = state

        # Node i's position in tree
        node_pos = points[indices[i]]
        sq_dist = jnp.sum(jnp.square(query - node_pos))

        # Find worst of current K nearest
        max_idx = jnp.argmax(nearest_dist)

        # Replace if closer
        is_closer = sq_dist < nearest_dist[max_idx]
        nearest_idx = jnp.where(
            is_closer,
            nearest_idx.at[max_idx].set(indices[i]),
            nearest_idx
        )
        nearest_dist = jnp.where(
            is_closer,
            nearest_dist.at[max_idx].set(sq_dist),
            nearest_dist
        )

        return nearest_idx, nearest_dist

    # Visit all nodes up to max_depth (FIXED iteration count)
    nearest_indices, nearest_sq_dists = lax.fori_loop(
        0,
        min(n_nodes_to_visit, len(points)),  # Fixed upper bound
        check_node,
        (nearest_indices, nearest_sq_dists)
    )

    return nearest_indices, jnp.sqrt(nearest_sq_dists)
```

**Pros**:
- ✅ **Vmappable**: Fixed iteration count (no data-dependent loops)
- ✅ **Correct results**: Eventually finds K nearest (if within max_depth)
- ✅ **Simple implementation**: ~100 lines of code

**Cons**:
- ❌ **EXTREMELY SLOW**: Visits 2^D - 1 nodes regardless of need
  - D=10: 1,023 nodes
  - D=15: 32,767 nodes
  - D=20: 1,048,575 nodes (mesh has 571K nodes!)
- ❌ **No spatial pruning**: Visits nodes that are provably too far
- ❌ **Defeats purpose of KD-tree**: Linear search would be faster!

**Performance estimate**:
```
FLA mesh: 571,173 nodes
Max depth: log2(571,173) ≈ 19.1 → D=20

Fixed-depth traversal:
  Nodes visited: min(1,048,575, 571,173) = 571,173
  → Visit EVERY node!
  → O(N) complexity (same as linear search)
  → ~1000× slower than smart traversal

Verdict: NOT VIABLE ❌
```

### 2.3 Approach 2: Bounded Traversal with Fixed Iterations

**Idea**: Traverse tree with early exit, but pad with no-ops to fixed iteration count.

```python
def vmappable_kdtree_query_bounded(query, tree, k=3, max_iterations=100):
    """
    Vmappable KD-tree query with bounded iterations.

    Use standard tree traversal, but limit to max_iterations.
    Pad with no-op iterations if early exit occurs.
    """
    points, indices, split_dims = tree.points, tree.indices, tree.split_dims
    n_points = len(points)

    # Initialize state
    current = jnp.int32(0)
    previous = jnp.int32(-1)
    nearest_indices = jnp.full(k, -1, dtype=jnp.int32)
    nearest_sq_dists = jnp.full(k, jnp.inf, dtype=jnp.float32)
    finished = jnp.bool_(False)  # Early exit flag

    def traversal_step(i, state):
        """Single tree traversal step"""
        current, previous, nearest_idx, nearest_dist, finished = state

        # If finished, do nothing (no-op padding)
        # This ensures fixed iteration count for vmap

        # Check if current node is closer
        node_pos = points[indices[current]]
        sq_dist = jnp.sum(jnp.square(query - node_pos))
        max_idx = jnp.argmax(nearest_dist)
        is_closer = sq_dist < nearest_dist[max_idx]

        nearest_idx = lax.cond(
            finished | ~is_closer,
            lambda: nearest_idx,
            lambda: nearest_idx.at[max_idx].set(indices[current])
        )
        nearest_dist = lax.cond(
            finished | ~is_closer,
            lambda: nearest_dist,
            lambda: nearest_dist.at[max_idx].set(sq_dist)
        )

        # Compute next node (standard KD-tree traversal)
        parent = (current - 1) // 2
        level = jnp.frexp(current + 1)[1] - 1
        split_dim = split_dims[current] if split_dims is not None else jnp.mod(level, 3)
        split_distance = query[split_dim] - points[indices[current], split_dim]
        near_side = split_distance > 0
        near_child = 2 * current + 1 + near_side
        far_child = 2 * current + 2 - near_side
        far_in_range = jnp.square(split_distance) <= jnp.max(nearest_dist)

        # Next node selection (complex logic, but fixed operations)
        next_node = lax.select(
            (previous == near_child) | ((previous == parent) & (near_child >= n_points)),
            lax.select((far_child < n_points) & far_in_range, far_child, parent),
            lax.select(previous == parent, near_child, parent),
        )

        # Check if finished (returned to root)
        finished_new = (next_node == -1) | finished

        # Update state
        previous_new = lax.select(finished, previous, current)
        current_new = lax.select(finished, current, next_node)

        return current_new, previous_new, nearest_idx, nearest_dist, finished_new

    # Fixed iteration count (REQUIRED for vmap)
    _, _, nearest_indices, nearest_sq_dists, _ = lax.fori_loop(
        0,
        max_iterations,  # FIXED upper bound
        traversal_step,
        (current, previous, nearest_indices, nearest_sq_dists, finished)
    )

    return nearest_indices, jnp.sqrt(nearest_sq_dists)
```

**Pros**:
- ✅ **Vmappable**: Fixed iteration count
- ✅ **Faster than fixed-depth**: Only visits nodes along traversal path
- ✅ **Correct (mostly)**: Finds K nearest if traversal completes within max_iterations

**Cons**:
- ❌ **Still slower than smart traversal**: Average case ~50-100 iterations vs ~10-20 for data-dependent
- ❌ **Incomplete results risk**: If max_iterations too small, may not reach K nearest
- ❌ **Wasted iterations**: Padding with no-ops after early exit
- ⚠️ **Complex implementation**: ~300-400 lines, tricky to debug

**Performance estimate**:
```
Smart tree traversal (jaxkd):
  Average nodes visited: ~15-25 (depends on tree balance)
  Time per query: ~5 μs

Bounded traversal (max_iterations=100):
  Nodes visited: 100 (always, due to padding)
  Time per query: ~20 μs (4× slower)

Bounded traversal (max_iterations=50):
  Nodes visited: 50 (always)
  Time per query: ~10 μs (2× slower)
  Risk: May not reach K nearest for deep tree positions

Verdict: MARGINALLY VIABLE ⚠️
  - 2-4× slower than smart traversal
  - Complex implementation
  - Risk of incomplete results
```

### 2.4 Implementation Complexity

**Estimated effort for bounded traversal:**

```
Implementation:
  - Tree traversal logic: ~200 lines (port from jaxkd)
  - Fixed iteration wrapper: ~50 lines
  - Testing and validation: ~200 lines
  - Total: ~450 lines of non-trivial JAX code

Debugging challenges:
  - Tree traversal has 10+ branches (near/far child, in-range checks, etc.)
  - Each branch must be tested for correctness
  - Edge cases: tree not balanced, particles at boundaries, etc.
  - Numerical precision issues (distance comparisons)

Testing requirements:
  - Unit tests for tree construction
  - Unit tests for traversal logic
  - Validation against jaxkd results (100% agreement required)
  - Performance benchmarks
  - Edge case testing (particles outside mesh, degenerate positions, etc.)

Estimated time: 1-2 weeks of focused development + testing
```

### 2.5 Expected Performance

**Best-case scenario** (bounded traversal with max_iterations=50):

```python
# Per RK4 step (4 stages × 225K particles)
def rk4_step_with_bounded_kdtree(positions, velocities, kdtree):
    # Each particle needs 4 KD-tree queries (k1, k2, k3, k4)
    # Each query: ~10 μs (bounded traversal)
    # Total: 4 × 225K × 10 μs = 9 seconds per step

    # Compare to current incremental search:
    # 225K particles × 2,000 element tests × 0.1 μs = 45 seconds per step

    # Speedup: 5× faster!
    pass

# BUT: Still 2× slower than smart KD-tree
# Smart traversal: ~5 μs/query → 4.5 seconds per step
# Bounded: ~10 μs/query → 9 seconds per step
```

**Risk**:
- If max_iterations=50 is insufficient for deep positions, retention drops
- Need max_iterations=100 → 18 seconds per step (only 2.5× speedup)

---

## Alternative: Morton Node-Based Search

### 3.1 The Idea

**Instead of indexing elements by centroid, index NODES and search nodes.**

```
Current approach (fails):
  Elements → Morton codes (by centroid) → Search elements
  Problem: Elements span cells, particle in tail not found

New approach:
  Nodes → Morton codes (nodes are points!) → Search nodes → Test connected elements
  Advantage: Nodes are point-like, no spanning issue!
```

### 3.2 Algorithm

```python
# Phase 1: Build node-based Morton structure (one-time, CPU)
def build_morton_node_structure(node_positions, connectivity):
    """Build Morton curve over mesh nodes"""

    # 1. Encode node positions to Morton codes
    morton_codes = encode_morton_batch(node_positions)  # (n_nodes,)

    # 2. Sort nodes by Morton code
    sorted_indices = np.argsort(morton_codes)
    sorted_codes = morton_codes[sorted_indices]

    # 3. Build Morton leaves (e.g., 1000 nodes per leaf)
    n_leaves = (len(node_positions) + 999) // 1000
    leaf_node_offsets = np.arange(0, len(node_positions), 1000)

    # 4. Build node → elements mapping (inverted connectivity)
    node_to_elements = build_node_to_elements_mapping(connectivity)

    return MortonNodeStructure(
        sorted_node_indices=sorted_indices,
        sorted_morton_codes=sorted_codes,
        leaf_node_offsets=leaf_node_offsets,
        node_to_elements_offsets=node_to_elements_offsets,
        node_to_elements_data=node_to_elements_data,
    )

# Phase 2: Query (per particle, GPU, vmappable)
@jax.jit
def search_L2_morton_nodes(pos, morton_node_gpu, radius=10):
    """
    Search using Morton curve over nodes (vmappable).

    1. Encode particle position to Morton code
    2. Binary search to find leaf
    3. Test ±radius leaves
    4. For each node in those leaves, test connected elements
    """

    # 1. Encode position
    morton_code = encode_morton_jax(pos)

    # 2. Binary search leaves
    leaf_id = binary_search_morton(morton_code, morton_node_gpu.sorted_morton_codes)

    # 3. Expand to radius
    for r in range(-radius, radius+1):
        test_leaf = leaf_id + r
        if test_leaf < 0 or test_leaf >= morton_node_gpu.n_leaves:
            continue

        # 4. For each node in this leaf
        start_node = morton_node_gpu.leaf_node_offsets[test_leaf]
        end_node = morton_node_gpu.leaf_node_offsets[test_leaf + 1]

        for node_idx in range(start_node, end_node):
            node_id = morton_node_gpu.sorted_node_indices[node_idx]

            # 5. For each element connected to this node
            elem_start = morton_node_gpu.node_to_elements_offsets[node_id]
            elem_end = morton_node_gpu.node_to_elements_offsets[node_id + 1]

            for elem_idx in range(elem_start, elem_end):
                elem_id = morton_node_gpu.node_to_elements_data[elem_idx]

                # 6. Test point-in-tet
                if point_in_tet(pos, elem_id, ...):
                    return elem_id

    return -1  # Not found
```

### 3.3 Key Differences from Element-Based Morton

| Aspect | Element-Based (Current) | Node-Based (Proposed) |
|--------|-------------------------|----------------------|
| **Indexed entities** | Elements (centroids) | Nodes (positions) |
| **Morton codes** | Element centroids | Node positions |
| **Spanning problem** | ❌ Yes (25% loss) | ✅ No (nodes are points) |
| **Tests per particle** | ~536 elements (R=2) | ~1,000 nodes → ~21K elements |
| **Memory** | 3M codes (elements) | 571K codes (nodes) |
| **Vmappable** | ✅ Yes | ✅ Yes |
| **Implementation** | ✅ Exists | ⚠️ Needs coding (~500 lines) |

### 3.4 Performance Analysis

**Tests per particle:**

```
Setup:
  - 571,173 nodes
  - ~21.4 elements per node (mean)
  - Morton leaves: 1,000 nodes/leaf → 571 leaves
  - Radius R: test 2R+1 leaves

Tests breakdown:
  R=2:  5 leaves × 1,000 nodes × 21.4 elem/node = ~107,000 element tests ❌ (200× more than element-based!)
  R=5:  11 leaves × 1,000 nodes × 21.4 elem/node = ~235,400 element tests ❌
  R=10: 21 leaves × 1,000 nodes × 21.4 elem/node = ~449,400 element tests ❌

This is MUCH WORSE than element-based Morton!
```

**Why so many tests?**
```
Element-based Morton:
  - 3M elements / 43,691 leaves = ~69 elements/leaf
  - R=2: 5 leaves × 69 = ~345 tests

Node-based Morton:
  - 571K nodes / 571 leaves = 1,000 nodes/leaf
  - Each node connects to 21.4 elements
  - R=2: 5 leaves × 1,000 × 21.4 = ~107,000 tests

Ratio: 310× MORE tests!
```

### 3.5 Optimization: Smaller Node Leaves

**Idea**: Use more leaves to reduce nodes per leaf.

```python
# Instead of 1,000 nodes/leaf, use 50 nodes/leaf
n_leaves = 571,173 // 50 = 11,423 leaves

Tests per particle:
  R=2:  5 leaves × 50 nodes × 21.4 elem/node = ~5,350 tests (10× element-based)
  R=5:  11 leaves × 50 nodes × 21.4 elem/node = ~11,770 tests (22× element-based)
  R=10: 21 leaves × 50 nodes × 21.4 elem/node = ~22,470 tests (42× element-based)

Still MUCH worse than element-based! ❌
```

**Fundamental problem:**
```
Elements connect to 4 nodes
Nodes connect to ~21.4 elements

Element-based: Test element directly
Node-based: Test element's 4 nodes → 4× redundancy minimum

Plus: Nodes are denser than elements
  → More nodes per Morton leaf
  → Even more tests

Verdict: Node-based Morton is SLOWER than element-based ❌
```

### 3.6 Optimization: Deduplicate Elements

**Idea**: Track tested elements to avoid duplicates.

```python
def search_L2_morton_nodes_dedup(pos, morton_node_gpu, radius=10):
    """Node-based Morton with element deduplication"""

    tested_elements = jnp.full(MAX_TESTS, -1, dtype=jnp.int32)
    n_tested = 0

    for r in range(-radius, radius+1):
        leaf_id = ...
        for node_idx in range(start_node, end_node):
            node_id = morton_node_gpu.sorted_node_indices[node_idx]

            for elem_idx in range(elem_start, elem_end):
                elem_id = morton_node_gpu.node_to_elements_data[elem_idx]

                # Check if already tested
                already_tested = jnp.any(tested_elements == elem_id)
                if already_tested:
                    continue  # Skip duplicate

                # Test element
                if point_in_tet(pos, elem_id, ...):
                    return elem_id

                # Mark as tested
                tested_elements = tested_elements.at[n_tested].set(elem_id)
                n_tested += 1

    return -1
```

**Problem with deduplication:**
```
❌ Requires array scan (jnp.any) for EVERY element candidate
   - 5,350 candidates × 5,350 scans = 28M comparisons
   - Far slower than just testing duplicates!

❌ Alternative: Use hash set
   - Not efficiently implementable in JAX
   - Would require custom GPU kernel

Verdict: Deduplication is SLOWER than testing duplicates ❌
```

---

## Comprehensive Comparison

### 4.1 Retention Comparison

| Method | Retention | Tests/Particle | Vmappable | Status |
|--------|-----------|----------------|-----------|--------|
| Element Morton R=10 | 96.96% | ~2,247 | ✅ | ✅ **Production** |
| Element Morton incremental | 98.21% | ~22.5 | ✅ | ✅ Production (slow) |
| **KD-tree (jaxkd)** | **95-100%** | **~64** | ❌ | ⚠️ **Batch only** |
| KD-tree (bounded traversal) | 95-100% | ~64 | ✅ | ⚠️ Feasible (complex) |
| **Node-based Morton R=2** | **~100%** | **~5,350** | ✅ | ❌ **Too slow** |
| Node-based Morton R=5 | ~100% | ~11,770 | ✅ | ❌ Too slow |

### 4.2 Speed Comparison (Estimated)

| Method | Tests/Particle | Time/Particle | Throughput | Speedup |
|--------|----------------|---------------|------------|---------|
| Element Morton R=10 | 2,247 | 19 μs | 52K p/s | 1.0× (baseline) |
| **KD-tree (jaxkd)** | **64** | **~1 μs** | **~1M p/s** | **~50×** (batch only) |
| KD-tree (bounded, max_iter=50) | 64 | ~5 μs | ~200K p/s | ~4× |
| KD-tree (bounded, max_iter=100) | 64 | ~10 μs | ~100K p/s | ~2× |
| Node Morton R=2 (no dedup) | 5,350 | ~45 μs | ~22K p/s | ~0.4× |
| Node Morton R=2 (with dedup) | ~2,000 | ~120 μs | ~8K p/s | ~0.15× |

### 4.3 Implementation Effort

| Method | Lines of Code | Complexity | Debugging | Testing | Total Effort |
|--------|---------------|------------|-----------|---------|--------------|
| Element Morton (current) | ✅ Exists | Low | Easy | ✅ Done | 0 days |
| KD-tree (jaxkd) | ✅ Exists | Low | Easy | ✅ Done | 0 days |
| **KD-tree (bounded)** | **~450** | **High** | **Hard** | **1 week** | **1-2 weeks** |
| Node Morton | ~500 | Medium | Medium | 3 days | 1 week |

### 4.4 Risk Assessment

| Method | Performance Risk | Correctness Risk | Maintenance Risk | Overall Risk |
|--------|-----------------|------------------|------------------|--------------|
| Element Morton R=10 | ✅ Low (proven) | ✅ Low (proven) | ✅ Low (simple) | ✅ **LOW** |
| KD-tree (jaxkd) | ✅ Low (fast) | ✅ Low (tested) | ⚠️ Medium (external dep) | ⚠️ Medium |
| **KD-tree (bounded)** | ⚠️ **Medium (2-4× slower)** | ⚠️ **Medium (complex)** | 🔴 **High (custom impl)** | 🔴 **HIGH** |
| Node Morton | 🔴 High (40× slower) | ✅ Low (simple) | ✅ Low (simple) | 🔴 **HIGH** |

---

## Recommendations

### 5.1 For Production RK4 Tracking: Keep Element-Based Morton

**Recommendation**: **Do NOT implement vmappable KD-tree or node-based Morton.**

**Rationale**:

1. **Element Morton R=10 is good enough**:
   - ✅ 97% retention (acceptable for particle tracking)
   - ✅ 52K p/s throughput (fast)
   - ✅ Simple, proven, low-risk
   - ✅ Already implemented and tested

2. **Vmappable KD-tree is complex and risky**:
   - ⚠️ 2-4× slower than smart KD-tree (defeats the purpose)
   - ⚠️ 1-2 weeks implementation + testing effort
   - 🔴 High debugging complexity (10+ branches in traversal)
   - 🔴 High maintenance burden (custom implementation)
   - 🔴 Only 2× speedup over current Element Morton (not worth it!)

3. **Node-based Morton is prohibitively slow**:
   - ❌ 40× more tests than element-based
   - ❌ 0.4× throughput (2.5× SLOWER than current!)
   - ❌ Defeats the entire purpose of spatial indexing

**Verdict**: **The juice is not worth the squeeze.** ❌

### 5.2 For Initial Assignment: Use KD-tree (jaxkd)

**Recommendation**: **Use jaxkd for initial assignment** (already implemented).

**Rationale**:
- ✅ 100% retention with cascading K
- ✅ Fast (batch query before vmap)
- ✅ Already implemented and tested
- ✅ No vmappable requirement (batch operation)

**Configuration**:
```python
# Initial assignment with KD-tree cascading
if JAXKD_AVAILABLE:
    kdtree_gpu = upload_kdtree_to_gpu(kdtree_struct)
    elem_ids = search_L2_kdtree_batch(
        positions,
        kdtree_gpu,
        k_cascading=[3, 5, 10, 20, 50]
    )
    # Result: 100% retention
else:
    # Fallback to Morton cascading
    elem_ids = search_L2_morton_cascading(
        positions,
        morton_gpu,
        radii=[500, 1000, 2000, 5000, 10000, 100000]
    )
    # Result: 100% retention
```

### 5.3 For Maximum Performance: Optimize Incremental Morton

**Recommendation**: **Investigate and fix incremental Morton slowness.**

**Current anomaly**:
```
Incremental (2,4,8,15,30):
  - Retention: 98.21% (excellent!)
  - Tests: ~22.5 (adaptive, excellent!)
  - Throughput: 9,136 p/s (SLOW - 5× slower than expected!)

Expected: ~30-40K p/s (early exits should be fast)
Actual: 9K p/s (only 3× faster than graph traversal!)

Something is wrong! ⚠️
```

**Possible causes**:
1. JIT compilation overhead (recompilation per radius tier?)
2. `jnp.where` branching overhead (GPU divergence?)
3. Memory access patterns (non-coalesced reads?)
4. Control flow overhead (lax.cond vs jnp.where?)

**Action items**:
1. Profile incremental search (JAX profiler, nsys)
2. Compare to fixed radius R=10 (why is R=10 5× faster?)
3. Try alternative implementations:
   - Replace `jnp.where` with `lax.cond`
   - Unroll cascading loop (explicit tiers vs for-loop)
   - Vectorize radius search (test all tiers in parallel)

**Potential outcome**:
- If optimized incremental → 30-40K p/s: **Best of both worlds!**
  - 98% retention + 30K p/s = Perfect solution ✅
- If still slow → stick with fixed R=10 (52K p/s, 97% retention)

### 5.4 Summary Table

| Approach | Effort | Risk | Speedup | Retention | Recommendation |
|----------|--------|------|---------|-----------|----------------|
| **Keep Element Morton R=10** | **0 days** | **Low** | **1.0×** | **97%** | ✅ **DO THIS** |
| Optimize Incremental Morton | 2-3 days | Low | 2-4× | 98% | ✅ **TRY THIS** |
| Use KD-tree for init only | 0 days | Low | N/A | 100% | ✅ **ALREADY DONE** |
| Implement bounded KD-tree | 1-2 weeks | High | 2× | 95-100% | ❌ **NOT WORTH IT** |
| Implement node Morton | 1 week | Medium | 0.4× | 100% | ❌ **SLOWER** |

---

## Conclusion

### Final Verdict

**DO NOT implement vmappable KD-tree or node-based Morton.**

**Why:**

1. **Element-based Morton is already good enough** (97%, 52K p/s)
2. **Vmappable KD-tree is too complex** (1-2 weeks, high risk, only 2× speedup)
3. **Node-based Morton is slower** (40× more tests, 0.4× throughput)
4. **Better ROI: optimize incremental Morton** (2-3 days, low risk, 2-4× speedup potential)

### Recommended Actions

1. ✅ **Production**: Use Element Morton R=10 (current)
2. ✅ **Initial assignment**: Use KD-tree batch (already implemented)
3. ✅ **Optimization effort**: Profile and fix incremental Morton (2-3 days)
4. ❌ **Do NOT pursue**: Vmappable KD-tree or node-based Morton

### The Fundamental Insight

> **There is no perfect solution to the multi-cell element spanning problem.**

The three options are:
1. **Accept 2-4% loss** with fast search (Element Morton R=10) ✅
2. **Pay 4-27× overhead** for perfect coverage (Multi-cell octree) ❌
3. **Implement complex vmappable KD-tree** for 2× speedup (not worth it) ❌

**We choose option 1: fast search with acceptable loss.**

This is the right engineering trade-off for production particle tracking.

---

## References

- [jaxkd source code](file:///home/arhashemi/Workspace/jaxkd/jaxkd/tree.py)
- [kdtree_node_search.py](jaxtrace/gpu/search/kdtree_node_search.py)
- [ELEMENT_SPANNING_PROBLEM_ANALYSIS.md](ELEMENT_SPANNING_PROBLEM_ANALYSIS.md)
- [METHODS_PERFORMANCE_REPORT.md](METHODS_PERFORMANCE_REPORT.md)
- [KDTREE_IMPLEMENTATION_SUMMARY.md](KDTREE_IMPLEMENTATION_SUMMARY.md)

---

**Document Status: Complete**
**Analysis depth: Comprehensive with code-level feasibility assessment**
**Recommendation: Clear and actionable**
