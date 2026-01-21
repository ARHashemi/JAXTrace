# Hierarchical Octree Search Design for Fully-Fused RK4

**Date**: 2025-12-25
**Goal**: Fix Morton neighbor search to use octree hierarchy without breaking fully-fused architecture

---

## Current Bug Analysis

### Critical Bug in morton_global_search.py

**Line 686** double-shifts the neighbor prefix:

```python
# get_26_neighbor_prefixes_jax returns FULL 64-bit Morton codes (left-aligned):
neighbor_prefixes = get_26_neighbor_prefixes_jax(...)
# Returns: [0x7A3B000000000000, 0x7A3C000000000000, ...]

# But then we shift AGAIN:
shift_amount = 63 - (table_depth * 3)  # 63 - 21 = 42 for depth 7
prefix_idx = lax.shift_right_logical(neighbor_prefix, jnp.uint64(42))
# Result: 0x7A3B >> 42 = completely wrong value!
```

**This is why 67% particle loss occurs** - we're looking up the wrong prefixes!

### What Should Happen

```python
# Neighbor prefix is already left-aligned 64-bit Morton code
neighbor_prefix_full = 0x7A3B5C0000000000

# Extract top (depth*3) bits to get index:
shift_amount = 63 - (depth * 3)  # 63 - 21 = 42 for depth 7
prefix_as_index = neighbor_prefix_full >> shift_amount
# Result: 0x7A3B5C = 8,010,588 (valid index in 2M table)
```

**Fix**: The code is actually correct! The bug is subtle - `get_26_neighbor_prefixes_jax` returns prefixes encoded at the QUERY depth (depth 7), but the table might have been built at a different depth.

Let me check what `encode_morton_prefix_jax` actually returns...

Actually, looking more carefully:

```python
# encode_morton_prefix_jax(nx, ny, nz, depth=7)
# Encodes 7-level coordinates into Morton code
# Stores in TOP bits of uint64

# The returned value IS a full 64-bit left-aligned Morton code
# So shifting by (63 - depth*3) extracts the prefix index
```

The code looks correct structurally. The problem must be elsewhere.

---

## Root Cause: Mismatch Between Leaf Depth and Table Depth

After analysis, the real problem is:

**Leaves are at variable depths (6-7), but neighbor search assumes all at table depth 7.**

```
Particle position maps to:
  Query Morton code: 0x7A3B5C... at depth 7

get_26_neighbor_prefixes_jax:
  Returns neighbors at depth 7: [0x7A3B5D, 0x7A3C5C, ...]

prefix_table lookup:
  Table built at depth 7 (2M entries)
  BUT: Some leaves are at depth 6!

When we look up depth-7 prefix for a depth-6 leaf:
  Depth-6 leaf covers 0x7A3000 - 0x7A3FFF (all 8 depth-7 children)
  Query for 0x7A3B5C → not found! (wrong granularity)
```

**The issue**: Variable-depth leaves (6-7) don't align with fixed-depth neighbor search (7).

---

## Solution Strategy: Hierarchical Fallback Search

We need to search at MULTIPLE depths if fine-grained search fails:

```python
def search_L2_hierarchical(pos, mesh_gpu):
    """Hierarchical octree search with multi-depth fallback."""

    # 1. Try depth 7 (finest) first
    result = search_at_depth(pos, mesh_gpu, depth=7)
    if result >= 0:
        return result

    # 2. If failed, try depth 6 (coarser)
    result = search_at_depth(pos, mesh_gpu, depth=6)
    if result >= 0:
        return result

    # 3. Final fallback: radius search
    return search_with_radius(pos, mesh_gpu, radius=2)
```

**Challenge**: This has branching (if-else), which JAX doesn't like in JIT-compiled code.

---

## JAX-Compatible Solution: Parallel Multi-Depth Search

Instead of sequential fallback, **search all depths in parallel**:

```python
def search_L2_multi_depth_single(pos, mesh_gpu):
    """Search at multiple octree depths in parallel (JAX-compatible)."""

    # Search at depth 7 (27 neighbors)
    result_d7 = search_morton_neighbors_at_depth(pos, mesh_gpu, depth=7)

    # Search at depth 6 (27 neighbors, coarser)
    result_d6 = search_morton_neighbors_at_depth(pos, mesh_gpu, depth=6)

    # Combine results: prefer finer depth if found
    final_result = jnp.where(
        result_d7 >= 0,
        result_d7,  # Found at depth 7 (prefer this)
        result_d6   # Fallback to depth 6
    )

    return final_result
```

**Cost**: 2× searches (27+27=54 octants), but no branching.

**Benefit**: Covers both fine and coarse leaves, catches particles at depth boundaries.

---

## Optimized Solution: Smart Depth Selection

Use the actual leaf depth to decide search depth:

```python
def search_L2_adaptive_depth(pos, mesh_gpu):
    """Adaptively choose search depth based on leaf structure."""

    # 1. Get Morton code for position
    morton_code = morton_encode_position_jax(pos, mesh_gpu.bbox_min, ...)

    # 2. Look up leaf at finest table depth (7)
    leaf_id_fine = position_to_leaf_id_octree(pos, mesh_gpu)

    # 3. Check leaf's actual depth (stored in leaf metadata)
    leaf_depth = mesh_gpu.leaf_depths[leaf_id_fine]  # NEW: need to store this!

    # 4. Search at leaf's native depth
    result = jnp.where(
        leaf_depth == 7,
        search_morton_neighbors_at_depth(pos, mesh_gpu, depth=7),
        search_morton_neighbors_at_depth(pos, mesh_gpu, depth=6)
    )

    return result
```

**Problem**: Requires storing leaf depth for each leaf (extra metadata).

**Benefit**: Only searches at correct depth, no wasted work.

---

## RECOMMENDED: Hybrid Approach (Best Performance/Accuracy Trade-off)

Combine Morton neighbors with small radius fallback:

```python
def search_L2_hybrid_single(pos, mesh_gpu):
    """
    Hybrid search: Morton neighbors at depth 7, then small radius fallback.

    This is JAX-compatible and handles depth mismatches gracefully.
    """

    # 1. Search 27 spatial neighbors at depth 7
    result_neighbors = search_morton_neighbors_at_depth(pos, mesh_gpu, depth=7)

    # 2. If not found, search small radius (±2 leaves in sorted order)
    #    This catches depth-6 leaves that weren't in depth-7 neighbor set
    center_leaf = position_to_leaf_id_octree(pos, mesh_gpu)

    def search_radius_leaf(offset):
        leaf_id = center_leaf + offset
        valid = (leaf_id >= 0) & (leaf_id < mesh_gpu.n_leaves)
        return jnp.where(
            valid,
            search_in_leaf_global(pos, leaf_id, mesh_gpu),
            jnp.int32(-1)
        )

    # Search ±2 leaves
    offsets = jnp.array([-2, -1, 0, 1, 2], dtype=jnp.int32)
    radius_results = jax.vmap(search_radius_leaf)(offsets)

    # Find first valid result from radius search
    radius_mask = radius_results >= 0
    result_radius = jnp.where(
        jnp.any(radius_mask),
        radius_results[jnp.argmax(radius_mask)],
        jnp.int32(-1)
    )

    # Combine: prefer neighbor result, fallback to radius
    final_result = jnp.where(
        result_neighbors >= 0,
        result_neighbors,
        result_radius
    )

    return final_result
```

**Cost**: 27 neighbor searches + 5 radius searches = 32 total (if neighbor fails)

**Benefit**:
- Geometrically correct (uses Morton neighbors)
- Handles depth mismatches (radius fallback)
- Fully JAX-compatible (no branching)
- Expected to catch 95%+ of particles

---

## Implementation Plan

### Phase 1: Fix the Current Bug (IMMEDIATE)

**File**: `jaxtrace/gpu/search/morton_global_search.py` line 686

**Current** (WRONG):
```python
prefix_idx = lax.shift_right_logical(neighbor_prefix, jnp.uint64(shift_amount))
```

Wait, let me trace through the actual values to find the real bug...

Actually, I need to verify what `encode_morton_prefix_jax` returns. Let me check:

```python
# encode_morton_prefix_jax(x=61, y=35, z=27, depth=7)
# Returns: uint64 with top 21 bits set

# Format: bits [60-62,57-59,54-56,...,3-5,0-2] = Morton code
# For depth 7: uses top 21 bits (63-21=42 to 62)
# Shift: bits positioned at [60-62] to [0-2] for level 0

# Return value: prefix << (63 - depth*3)
# So for depth 7: prefix << (63-21) = prefix << 42
# Result: left-aligned in uint64
```

So `encode_morton_prefix_jax` DOES return left-aligned codes.

Then `morton_global_search.py` line 686 is CORRECT - it shifts right to extract the index.

**Where's the bug then?**

Let me trace through with actual numbers:

```
center_prefix (input to get_26_neighbor_prefixes_jax):
  = morton_query >> (63 - 7*3)
  = morton_query >> 42
  = 0x7A3B5C (just the top 21 bits, as uint64 = 0x00000000007A3B5C)

decode_morton_prefix_jax(0x00000000007A3B5C, depth=7):
  Expects prefix in TOP bits!
  But we passed it as a small number!
  BUG HERE!!!
```

**FOUND IT!** Line 661 extracts the prefix as a **small integer** (0x7A3B5C), but `get_26_neighbor_prefixes_jax` → `decode_morton_prefix_jax` **expects it left-aligned**!

---

## THE ACTUAL BUG

**File**: `jaxtrace/gpu/search/morton_global_search.py` lines 658-665

```python
# Extract prefix at table depth
table_depth_int = int(mesh_gpu.table_depth)
prefix_bits = table_depth_int * 3
shift_amount = 63 - prefix_bits
center_prefix = lax.shift_right_logical(morton_query, jnp.uint64(shift_amount))
# BUG: center_prefix is now a SMALL INTEGER (0x7A3B5C)
# But get_26_neighbor_prefixes_jax expects LEFT-ALIGNED uint64!

neighbor_prefixes = get_26_neighbor_prefixes_jax(
    center_prefix,  # WRONG: should be left-aligned!
    table_depth_int,
    max_coord
)
```

**Fix**:
```python
# DON'T shift! Keep center_prefix left-aligned
center_prefix = morton_query  # Keep full 64-bit Morton code

neighbor_prefixes = get_26_neighbor_prefixes_jax(
    center_prefix,  # Now correct: left-aligned
    table_depth_int,
    max_coord
)
```

But wait, `decode_morton_prefix_jax` expects the prefix in the TOP bits... let me check its implementation again.

Looking at line 81 of `morton_neighbors.py`:
```python
bit_pos = (63 - 3) - i * 3  # Start at bit 60 for level 0
```

So it DOES expect left-aligned! The bug is definitely in line 661 of `morton_global_search.py`.

---

## FINAL FIX

**File**: `jaxtrace/gpu/search/morton_global_search.py`

**Line 661** (WRONG):
```python
center_prefix = lax.shift_right_logical(morton_query, jnp.uint64(shift_amount))
```

**Should be**:
```python
# Keep Morton code left-aligned for decode_morton_prefix_jax
center_prefix = morton_query
```

**Line 686** (ALSO WRONG - becomes redundant shift):

Current code then double-shifts when looking up. After fixing line 661, line 686 logic stays the same (it correctly extracts index from the returned neighbor prefixes).

Actually wait, let me re-examine the full flow once more to be absolutely sure...

No, I think I'm confusing myself. Let me write out the correct architecture from scratch.

