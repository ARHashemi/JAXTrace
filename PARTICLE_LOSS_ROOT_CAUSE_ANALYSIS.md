# Particle Loss Root Cause Analysis - Hierarchical Search

**Date**: 2025-12-25
**Current Performance**: 83.74% initial → 78.60% @ step 400, 10,230 p/s
**Expected**: 85-90% retention, 18-20K p/s

---

## Test Results Analysis

### Current Performance
```
Initial assignment: 40,194/48,000 (83.74%)
Step 100: 40,158 (83.66% retention) - 10,959 p/s
Step 200: 38,996 (81.24% retention) - 10,755 p/s
Step 400: 37,726 (78.60% retention) - 10,230 p/s
```

### Issues Identified

1. **Performance 50% slower than expected** (10K p/s vs 18-20K expected)
2. **Retention degrading over time** (83% → 78%)
3. **Initial assignment missing 16% of particles** (83.74% vs expected >95%)

---

## Root Causes of Particle Loss

### 1. Initial Assignment Failures (16% loss before tracking starts)

**Source**: Line 83 shows 7,806 particles couldn't be assigned even with radius=500

**Possible causes**:
- **Particles outside mesh bounds**: Check if particles seeded beyond mesh domain
- **Velocity scaling mismatch**: Seeding may use wrong coordinate system
- **Boundary elements missing**: Mesh may not cover full seeding volume
- **Morton encoding overflow**: Positions outside bbox → invalid Morton codes

**Evidence**: Cascading search found only 483 additional particles (from 39,554 to 40,194) despite searching radius 100→500

**Diagnosis needed**:
```python
# Check if unassigned particles are outside bbox
unassigned_positions = positions_gpu[element_ids_gpu == -1]
outside_bbox = jnp.any(
    (unassigned_positions < mesh_gpu.bbox_min) |
    (unassigned_positions > mesh_gpu.bbox_max),
    axis=1
)
print(f"Particles outside bbox: {jnp.sum(outside_bbox)}/{len(unassigned_positions)}")
```

### 2. Prefix Table Index Mismatch (Critical Bug in Hierarchical Search)

**Source**: Line 782-787 in `search_L2_morton_hierarchical_single()`

**Problem**: When searching at depth 6, we generate depth-6 neighbor prefixes, but then look them up in a **depth-7 table**!

```python
def search_at_depth(depth: int) -> jnp.int32:
    """Search 27 neighbors at specified octree depth."""
    # Generate neighbors at depth 6 or 7
    neighbor_prefixes = get_26_neighbor_prefixes_jax(
        center_prefix,
        depth,  # Could be 6!
        max_coord
    )

    # BUG: Always looks up in depth-7 table!
    table_depth_int = int(mesh_gpu.table_depth)  # = 7
    shift_amount = 63 - (table_depth_int * 3)    # = 63 - 21 = 42
    prefix_idx = lax.shift_right_logical(neighbor_prefix, jnp.uint64(shift_amount))

    # If neighbor_prefix is depth-6 (top 18 bits):
    #   Shifting by 42 extracts bits [60:42] (18 bits) → WRONG!
    #   Should shift by 63 - (6*3) = 45 for depth-6
```

**Result**: Depth-6 searches look up wrong prefixes → miss elements at coarse leaves

**Fix needed**:
```python
# Use QUERY depth for shift, not TABLE depth
shift_amount = 63 - (depth * 3)  # Use depth parameter, not table_depth
```

### 3. Single-Leaf Search per Prefix (Missing Multi-Leaf Elements)

**Source**: Line 790 in `search_at_depth()`

```python
# Look up leaf
first_leaf = mesh_gpu.prefix_start[prefix_idx]
has_leaves = mesh_gpu.prefix_length[prefix_idx] > 0
valid_leaf = first_leaf >= 0

# BUG: Only searches FIRST leaf!
elem_neighbor = search_in_leaf_global(pos, first_leaf, mesh_gpu)
```

**Problem**: Some prefixes map to MULTIPLE leaves (2-8 leaves for depth-6 prefixes)

**Example**:
```
Depth-6 prefix 0x7A3000... may contain:
  - Leaf 1234 (depth 7): elements 10000-10049
  - Leaf 1235 (depth 7): elements 10050-10099
  - Leaf 1236 (depth 6): elements 10100-10399  ← Particle here, but we only check leaf 1234!
```

**Fix needed**: Search ALL leaves in the prefix range
```python
# Search all leaves in prefix (not just first)
def search_all_leaves_in_prefix(start, count):
    def check_leaf(i, state):
        elem_id, found = state
        leaf_id = start + i
        valid = (i < count) & (leaf_id >= 0) & (leaf_id < mesh_gpu.n_leaves)
        result = jnp.where(
            valid & (~found),
            search_in_leaf_global(pos, leaf_id, mesh_gpu),
            jnp.int32(-1)
        )
        improved = result >= 0
        return (jnp.where(improved, result, elem_id), found | improved)

    return lax.fori_loop(0, 8, check_leaf, (jnp.int32(-1), False))[0]
```

### 4. Degrading Retention Over Time (83% → 78%)

**Source**: Particles lost during RK4 integration steps

**Possible causes**:

#### 4a. L0 Cache Misses at Refined Region Boundaries
- Particle exits small refined element
- L0 check fails (not in cached element)
- L1 search fails (neighbors are in different refinement level)
- L2 must succeed, but hierarchical search may fail due to bugs above

#### 4b. L1 Multi-Hop Fails Across Refinement Levels
- Particle moves from coarse → fine region
- Cached element is coarse (large neighbors)
- L1 tries to hop through coarse neighbors
- Never reaches fine-grained neighbors containing particle

**Evidence**:
- Step 100 → 200: Lost 1,162 particles (1.16% loss in 100 steps)
- Step 200 → 400: Lost 1,270 particles (1.27% loss in 200 steps)
- Consistent ~0.6% loss per 100 steps suggests systematic search failure

#### 4c. Velocity Extrapolation Errors
- RK4 intermediate positions (k1, k2, k3, k4) may exit mesh
- Position extrapolates beyond mesh boundary
- Search fails → particle marked lost

#### 4d. Degenerate Elements
- Point-in-tet test fails for highly distorted elements
- False negative → particle not found even when inside
- Threshold: `DEGENERACY_THRESHOLD = 1e-6` (relative)

### 5. Performance Bottleneck (50% slower than expected)

**Expected**: ~18-20K p/s
**Actual**: ~10K p/s

**Cause**: Hierarchical search doing BOTH depth-7 AND depth-6 searches for EVERY particle

**Evidence from code** (line 816-825):
```python
# This is NOT conditional execution - JAX evaluates BOTH branches!
result_depth_7 = search_at_depth(7)  # Always executes: 27 octants

result_depth_6 = jnp.where(
    result_depth_7 >= 0,
    result_depth_7,
    search_at_depth(6)  # JAX evaluates this REGARDLESS of condition!
)
```

**JAX behavior**: `jnp.where(cond, a, b)` evaluates **both** `a` AND `b`, then selects based on `cond`

**Result**: Every particle searches 54 octants (27 @ depth-7 + 27 @ depth-6) even if found at depth-7

**Expected cost**:
- Best case (found @ depth-7): 27 octants
- Worst case: 54 octants
- **Actual**: 54 octants for ALL particles (JAX evaluates both branches)

**Impact**: 2× work → 50% slower (20K → 10K p/s)

---

## Summary of Bugs Found

| Bug | Impact | Priority | Fix Complexity |
|-----|--------|----------|----------------|
| **1. Depth-6 prefix → depth-7 table lookup** | 🔴 Critical | P0 | Easy (1 line) |
| **2. Single-leaf search (missing multi-leaf)** | 🔴 Critical | P0 | Medium (add loop) |
| **3. JAX evaluates both depth branches** | 🟡 Major | P1 | Hard (need lax.cond) |
| **4. Initial assignment <95%** | 🟡 Major | P1 | Diagnosis needed |
| **5. L1 fails across refinement levels** | 🟠 Moderate | P2 | Complex (adaptive L1) |

---

## Recommended Fixes (Priority Order)

### P0 - Fix Critical Bugs (Expected: 85%+ retention, 15K+ p/s)

#### Fix 1: Correct depth-dependent shift in hierarchical search
```python
# Line 784: Use query depth, not table depth
shift_amount = 63 - (depth * 3)  # Use depth parameter!
```

#### Fix 2: Search ALL leaves in prefix range (not just first)
```python
# Replace single-leaf search with multi-leaf loop
first_leaf = mesh_gpu.prefix_start[prefix_idx]
num_leaves = mesh_gpu.prefix_length[prefix_idx]

# Search all leaves (max 8 for depth-6 prefix)
def search_multi_leaf(i, state):
    elem_id, found = state
    leaf_id = first_leaf + i
    valid = (i < num_leaves) & (leaf_id >= 0) & (leaf_id < mesh_gpu.n_leaves) & (~found)
    result = jnp.where(valid, search_in_leaf_global(pos, leaf_id, mesh_gpu), jnp.int32(-1))
    improved = result >= 0
    return (jnp.where(improved, result, elem_id), found | improved)

elem_neighbor, _ = lax.fori_loop(0, 8, search_multi_leaf, (jnp.int32(-1), False))
```

### P1 - Optimize Performance

#### Fix 3: Use lax.cond for conditional depth-6 search
```python
# Replace jnp.where (evaluates both) with lax.cond (only evaluates one)
def search_depth_6(_):
    return search_at_depth(6)

result = lax.cond(
    result_depth_7 >= 0,
    lambda _: result_depth_7,  # Found at depth-7, return it
    search_depth_6,             # Not found, search depth-6
    None
)
```

**Expected speedup**: 2× for particles found at depth-7 (most particles)

### P2 - Diagnose Initial Assignment

#### Fix 4: Add diagnostic for unassigned particles
```python
# After initial assignment, analyze failures
unassigned_mask = element_ids_gpu == -1
unassigned_pos = positions_gpu[unassigned_mask]

# Check bounds
outside_x = (unassigned_pos[:, 0] < bbox_min[0]) | (unassigned_pos[:, 0] > bbox_max[0])
outside_y = (unassigned_pos[:, 1] < bbox_min[1]) | (unassigned_pos[:, 1] > bbox_max[1])
outside_z = (unassigned_pos[:, 2] < bbox_min[2]) | (unassigned_pos[:, 2] > bbox_max[2])
outside_bbox = outside_x | outside_y | outside_z

print(f"Unassigned particles outside bbox: {jnp.sum(outside_bbox)}/{jnp.sum(unassigned_mask)}")
print(f"Unassigned inside bbox: {jnp.sum(~outside_bbox)}/{jnp.sum(unassigned_mask)}")
```

---

## Expected Results After P0 Fixes

| Metric | Before | After P0 Fixes | After P1 Optimization |
|--------|--------|----------------|------------------------|
| **Initial assignment** | 83.74% | **90-95%** (multi-leaf) | 90-95% |
| **Retention @ step 100** | 83.66% | **88-92%** | 88-92% |
| **Retention @ step 400** | 78.60% | **85-90%** | 85-90% |
| **Throughput** | 10,230 p/s | **12-15K p/s** | **18-22K p/s** (lax.cond) |

---

## Next Steps

1. ✅ **Implement Fix 1** (depth-dependent shift) - CRITICAL
2. ✅ **Implement Fix 2** (multi-leaf search) - CRITICAL
3. ⏳ **Test with P0 fixes** - expect 85-90% retention @ 12-15K p/s
4. ⏳ **Implement Fix 3** (lax.cond optimization) - if retention OK
5. ⏳ **Diagnose initial assignment** - if retention still <90%

---

## Why Particles Get Lost

### Root Cause Categories

1. **Wrong table lookup** (depth-6 prefix → depth-7 table) ← **FIX 1**
2. **Missing leaves** (only search first leaf, not all) ← **FIX 2**
3. **Outside mesh bounds** (seeding error or velocity extrapolation)
4. **L1 can't cross refinement boundaries** (structural limitation)
5. **Degenerate elements** (point-in-tet false negatives)

**Most impactful**: Fixes 1 and 2 address systematic search failures affecting 10-15% of particles.
