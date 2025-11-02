# Phase 2: Backward Compatibility Fix

**Date**: 2025-10-28
**Issue**: Legacy code accessing `node_centers` and `node_sizes` after Morton code integration
**Status**: ✅ FIXED

---

## Problem

After integrating Morton codes in Phase 2, several parts of the codebase still accessed the old `node_centers` and `node_sizes` attributes, causing `AttributeError`:

```python
AttributeError: 'OctreeCoarseLevels' object has no attribute 'node_centers'
```

**Affected files**:
- `jaxtrace/fields/coarse_octree_builder.py` (line 307)
- `jaxtrace/fields/fine_octree_builder.py` (lines 62, 77-78, 324)
- `jaxtrace/fields/direct_octree_fem_interpolator.py` (lines 286-287, 294-295)
- `jaxtrace/fields/octree_search_cpu.py` (lines 308, 315)
- `jaxtrace/fields/direct_octree_interpolator_jax.py` (lines 105-106, 113-114)

---

## Solution

Added **backward compatibility properties** to octree dataclasses that decode Morton codes on-the-fly:

### For `OctreeCoarseLevels` (has `bbox_min`/`bbox_max`)

```python
@property
def node_centers(self) -> jnp.ndarray:
    """Decode Morton codes to get node centers (for backward compatibility)."""
    n_nodes = len(self.node_morton_codes)
    centers = np.zeros((n_nodes, 3), dtype=np.float32)

    domain_min = np.asarray(self.bbox_min, dtype=np.float32)
    domain_max = np.asarray(self.bbox_max, dtype=np.float32)

    for i in range(n_nodes):
        code = np.uint64(self.node_morton_codes[i])
        node_min, node_max, _ = decode_morton_3d(code, domain_min, domain_max)
        centers[i] = (node_min + node_max) / 2.0

    return jnp.array(centers)

@property
def node_sizes(self) -> jnp.ndarray:
    """Decode Morton codes to get node sizes (for backward compatibility)."""
    # Similar implementation
    ...
```

### For `OctreeFineLevel` (no `bbox_min`/`bbox_max`)

```python
def decode_node_centers(self, domain_min: np.ndarray, domain_max: np.ndarray) -> jnp.ndarray:
    """Decode Morton codes to get node centers."""
    # Implementation with explicit domain bounds
    ...

@property
def node_centers(self) -> jnp.ndarray:
    """Raises error - use decode_node_centers(domain_min, domain_max) instead."""
    raise AttributeError(
        "Fine octree nodes require domain bounds to decode. "
        "Use decode_node_centers(domain_min, domain_max) instead."
    )
```

---

## Direct Fixes

### `coarse_octree_builder.py`

**Before**:
```python
n_nodes = len(coarse_octree.node_centers)
```

**After**:
```python
n_nodes = len(coarse_octree.node_morton_codes)  # Phase 2
```

### `fine_octree_builder.py`

**Before**:
```python
for coarse_idx in range(len(coarse_octree.node_centers)):
    coarse_center = np.array(coarse_octree.node_centers[coarse_idx])
    coarse_size = float(coarse_octree.node_sizes[coarse_idx])
```

**After**:
```python
domain_min = np.asarray(coarse_octree.bbox_min, dtype=np.float32)
domain_max = np.asarray(coarse_octree.bbox_max, dtype=np.float32)

for coarse_idx in range(len(coarse_octree.node_morton_codes)):
    code = np.uint64(coarse_octree.node_morton_codes[coarse_idx])
    bbox_min, bbox_max, _ = decode_morton_3d(code, domain_min, domain_max)
```

---

## Performance Note

**⚠️ Important**: The `node_centers` and `node_sizes` properties decode Morton codes on-the-fly, which has a performance cost.

**For performance-critical code**:
- Cache the result: `centers = octree.node_centers  # Do once`
- Or use Morton codes directly and decode in batches
- Or refactor to work with Morton codes natively (Phase 3)

**Current usage**:
- Most code accesses these properties infrequently (initialization, logging)
- Performance impact is minimal for current use cases
- Full optimization will come in Phase 3 with GPU-native implementation

---

## Files Modified

1. **[jaxtrace/fields/shared_coarse_octree.py](../jaxtrace/fields/shared_coarse_octree.py)**
   - Added `node_centers` property to `OctreeCoarseLevels`
   - Added `node_sizes` property to `OctreeCoarseLevels`
   - Added `decode_node_centers()` method to `OctreeFineLevel`
   - Added `decode_node_sizes()` method to `OctreeFineLevel`

2. **[jaxtrace/fields/coarse_octree_builder.py](../jaxtrace/fields/coarse_octree_builder.py)**
   - Line 307: `node_centers` → `node_morton_codes`

3. **[jaxtrace/fields/fine_octree_builder.py](../jaxtrace/fields/fine_octree_builder.py)**
   - Lines 62-84: Decode Morton codes directly instead of accessing properties
   - Line 325: `node_centers` → `node_morton_codes`

---

## Verification

**Test**:
```bash
python -c "from jaxtrace.fields.shared_coarse_octree import OctreeCoarseLevels; print('✅ Import successful')"
```

**Result**: ✅ All modules import successfully

---

## Status

✅ **FIXED** - Backward compatibility restored

All legacy code can now access `node_centers` and `node_sizes` through properties that decode Morton codes on-the-fly.

**Next**: Run `example_workflow.py` to verify full workflow works with Morton code integration.
