# L1 Search Disable Configuration Guide

## Overview

The production tracking script now includes a configuration parameter `ENABLE_L1_SEARCH` that allows you to disable L1 neighbor search and test the L0→L2 search hierarchy directly.

## Configuration

In [production_tracking_fully_fused_timedep.py](production_tracking_fully_fused_timedep.py):

```python
# Search Hierarchy Configuration
N_HOPS = 3                     # Number of hops for L1 neighbor search
L2_SEARCH_RADIUS = 10          # L2 search radius during integration
ENABLE_L1_SEARCH = True        # Enable L1 neighbor search (set False to test L0→L2 only)
INITIAL_SEARCH_RADIUS = 100    # Extended radius for initial assignment
```

## Usage

### Test 1: L0→L2 Only (No L1)

This tests if Morton global search alone can handle the graded refinement.

```python
ENABLE_L1_SEARCH = False       # Disable L1 neighbor search
L2_SEARCH_RADIUS = 50          # Increase radius for better coverage
```

**Search hierarchy**: L0 (cached element) → L2 (Morton global search)

**Expected behavior**:
- L1 completely bypassed
- Every failed L0 goes directly to L2 Morton search
- Should find fine elements if Morton search radius is sufficient
- Performance: Slower than L0+L1+L2 (more Morton searches)

### Test 2: Baseline with L1 (Default)

Standard configuration for comparison.

```python
ENABLE_L1_SEARCH = True        # Enable L1 neighbor search
N_HOPS = 3
L2_SEARCH_RADIUS = 10
```

**Search hierarchy**: L0 (cached element) → L1 (3 hops) → L2 (Morton)

**Expected behavior**:
- Current behavior (particles stuck in coarse/medium elements)
- Faster performance but incorrect results in graded refinement

### Test 3: Increased N_HOPS with L1

Test if more hops can traverse graded refinement.

```python
ENABLE_L1_SEARCH = True        # Enable L1 neighbor search
N_HOPS = 10                    # Increase hops
L2_SEARCH_RADIUS = 50          # Also increase L2 radius
```

**Search hierarchy**: L0 (cached element) → L1 (10 hops) → L2 (Morton)

**Expected behavior**:
- L1 has more chances to find fine elements
- May traverse coarse→medium→fine successfully
- Performance: Moderate cost (~3x more neighbor checks in L1)

## Performance Comparison

### Metrics to Monitor

From script output, compare:
1. **Throughput**: particles/second
2. **Retention**: % particles active at end
3. **Memory**: GPU memory usage
4. **Correctness**: Do particles show rotation in refined region?

### Expected Performance

| Configuration | Throughput | Memory | Correctness |
|---------------|------------|--------|-------------|
| L0+L1(3)+L2 (baseline) | 50-120K/s | ~900MB | ❌ No rotation |
| L0+L2 only (L1 disabled) | 30-80K/s | ~900MB | ✅ Should work |
| L0+L1(10)+L2 | 40-100K/s | ~900MB | ? To test |

### Why L0→L2 Might Work

- Morton global search is **spatially aware** (not topology-based)
- Directly finds elements containing a position
- Doesn't rely on neighbor connectivity
- Should find fine elements if `L2_SEARCH_RADIUS` is large enough

## Recommended Testing Sequence

### Step 1: Test L0→L2 Only

```python
ENABLE_L1_SEARCH = False
L2_SEARCH_RADIUS = 50
N_STEPS = 500  # Short test run
```

**Goal**: Verify Morton search can find fine elements

**Success criteria**:
- Run `diagnose_tracking_through_refined_region.py` modified to use these settings
- Particles should be assigned to fine elements (>90%) in refined region
- Rotation should be visible

### Step 2: Performance Test

If Step 1 works, run full production:

```python
ENABLE_L1_SEARCH = False
L2_SEARCH_RADIUS = 50
N_STEPS = 2_500  # Full run
```

**Monitor**:
- Throughput (should be 30-80K particles/s)
- GPU memory (should be ~900MB, unchanged)
- Output VTK shows rotation

### Step 3: Optimize L2_SEARCH_RADIUS

If performance is acceptable but some particles still miss fine elements:

```python
ENABLE_L1_SEARCH = False
L2_SEARCH_RADIUS = 100  # Increase radius
```

**Trade-off**: Larger radius = more Morton leaves searched = slower but more robust

## Reverting Changes

To restore original behavior:

```python
ENABLE_L1_SEARCH = True
N_HOPS = 3
L2_SEARCH_RADIUS = 10
```

## Implementation Details

### Code Changes

Modified files:
1. [jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py](jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py)
   - Added `enable_l1_search` parameter to `create_rk4_fully_fused_timedep()`
   - Conditional L1 search in `search_l0_l1_l2_single()`

2. [production_tracking_fully_fused_timedep.py](production_tracking_fully_fused_timedep.py)
   - Added `ENABLE_L1_SEARCH` configuration parameter
   - Passed parameter to RK4 integrator
   - Added search hierarchy logging

### How It Works

When `enable_l1_search=False`:

```python
def search_l0_l1_l2_single(pos, cached_elem_id):
    # L0: Cached element
    elem_l0 = search_l0_single(pos, cached_elem_id)
    found_l0 = elem_l0 >= 0

    if enable_l1_search:
        # Normal L0→L1→L2 hierarchy
        elem_l1 = jnp.where(found_l0, elem_l0, search_l1_single(pos, cached_elem_id))
        found_l1 = elem_l1 >= 0
        elem_final = jnp.where(found_l1, elem_l1, search_l2_single(pos))
    else:
        # L1 disabled: L0→L2 hierarchy
        elem_final = jnp.where(found_l0, elem_l0, search_l2_single(pos))

    return elem_final
```

The `if enable_l1_search` is evaluated at **JIT compile time** (not runtime), so there's no performance penalty - the disabled path is completely eliminated from the compiled code.

## Diagnostic Script

To verify the configuration works, run the tracking diagnostic:

```bash
# Edit diagnose_tracking_through_refined_region.py to set:
# N_HOPS = 3  # or 10
# L2_SEARCH_RADIUS = 50
# And add enable_l1_search parameter to create_rk4_fully_fused_timedep call

python diagnose_tracking_through_refined_region.py
```

Expected output with `ENABLE_L1_SEARCH=False`:
```
Particle 3:
  Element types while in refined region:
    Fine elements: 260/289 (90.0%)   ✅
    Medium elements: 25/289 (8.7%)   ✅
    Coarse elements: 4/289 (1.4%)   ✅
```

## Conclusion

The `ENABLE_L1_SEARCH` parameter provides a simple way to test if L0→L2 hierarchy works for your graded mesh without L1 neighbor search.

**Key advantage**: Morton global search is spatially aware and doesn't depend on topology, so it should find fine elements even when L1 neighbor hops can't reach them.

**Trade-off**: Slightly lower performance (more L2 searches), but if results are correct, the performance cost may be acceptable.
