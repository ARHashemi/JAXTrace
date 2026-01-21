# JIT Dataclass Fix - Array-Based Wrappers

## Problem

JAX's JIT compiler cannot handle dataclass objects passed as arguments to functions. The error was:

```
TypeError: Cannot interpret value of type <class 'jaxtrace.gpu.search.aa_detection.AxisAlignedMetadata'>
as an abstract array; it does not have a dtype attribute

This typically means that a jit-wrapped function was called with a non-array argument, and this argument
was not marked as static using the static_argnums or static_argnames parameters of jax.jit.
```

## Root Cause

The corrected methods (`point_in_tet_pure_aa`, `point_in_tet_branchless_hybrid`) accepted `AxisAlignedMetadata` dataclass as a parameter. When these methods were called from within JIT-compiled code (L2 Morton search), JAX couldn't trace through the dataclass.

## Solution

### 1. Store Individual Arrays (Not Dataclass)

Modified [point_in_tet_methods.py](jaxtrace/gpu/search/point_in_tet_methods.py:304) to store individual arrays:

**Before**:
```python
_aa_metadata_gpu = None  # AxisAlignedMetadata dataclass
_element_vertices_gpu = None

def set_corrected_metadata(aa_metadata_gpu, element_vertices_gpu):
    global _aa_metadata_gpu, _element_vertices_gpu
    _aa_metadata_gpu = aa_metadata_gpu
    _element_vertices_gpu = element_vertices_gpu
```

**After**:
```python
# Store individual arrays (not dataclass) to avoid JIT compilation issues
_aa_base_vertices_gpu = None
_aa_inv_edge_lengths_gpu = None
_aa_axis_indices_gpu = None
_aa_is_axis_aligned_gpu = None
_element_vertices_gpu = None

def set_corrected_metadata(aa_metadata_gpu, element_vertices_gpu):
    global _aa_base_vertices_gpu, _aa_inv_edge_lengths_gpu, _aa_axis_indices_gpu
    global _aa_is_axis_aligned_gpu, _element_vertices_gpu

    # Extract individual arrays from dataclass
    _aa_base_vertices_gpu = aa_metadata_gpu.base_vertices
    _aa_inv_edge_lengths_gpu = aa_metadata_gpu.inv_edge_lengths
    _aa_axis_indices_gpu = aa_metadata_gpu.axis_indices
    _aa_is_axis_aligned_gpu = aa_metadata_gpu.is_axis_aligned
    _element_vertices_gpu = element_vertices_gpu
```

### 2. Add Array-Based Wrapper Functions

Added to [aa_detection.py](jaxtrace/gpu/search/aa_detection.py:490):

```python
@jax.jit
def point_in_tet_pure_aa_arrays(
    pos: jax.Array,
    elem_id: jnp.int32,
    base_vertices: jax.Array,          # Individual arrays instead of dataclass
    inv_edge_lengths: jax.Array,
    axis_indices: jax.Array
) -> jnp.bool_:
    """Pure AA method - array-based version for JIT compatibility."""
    # Extract precomputed metadata (same algorithm as original)
    p_base = base_vertices[elem_id]
    inv_len = inv_edge_lengths[elem_id]
    axes = axis_indices[elem_id]

    # ... (same computation as point_in_tet_pure_aa)
```

Similarly for `point_in_tet_branchless_hybrid_arrays`.

### 3. Update Dispatcher

Modified [point_in_tet_methods.py](jaxtrace/gpu/search/point_in_tet_methods.py:385) dispatcher:

**Before**:
```python
elif method == "pure_aa":
    from jaxtrace.gpu.search.aa_detection import point_in_tet_pure_aa
    return point_in_tet_pure_aa(pos, elem_id, _aa_metadata_gpu)  # ❌ Dataclass
```

**After**:
```python
elif method == "pure_aa":
    from jaxtrace.gpu.search.aa_detection import point_in_tet_pure_aa_arrays
    return point_in_tet_pure_aa_arrays(
        pos, elem_id,
        _aa_base_vertices_gpu,      # ✅ Individual arrays
        _aa_inv_edge_lengths_gpu,
        _aa_axis_indices_gpu
    )
```

## Pattern Used by OLD Implementation

The old `axis_aligned` method already followed this pattern - it accepted individual arrays (`connectivity`, `node_positions`) instead of a dataclass:

```python
def point_in_tet_axis_aligned(
    pos: jax.Array,
    elem_id: jnp.int32,
    connectivity: jax.Array,      # ✅ Arrays, not dataclass
    node_positions: jax.Array
) -> jnp.bool_:
```

## Why This Works

JAX's JIT compiler can trace through:
- ✅ JAX arrays (`jax.Array`, `jnp.ndarray`)
- ✅ Python primitives (int, float, bool)
- ❌ **NOT** custom Python objects (dataclasses, classes)

By extracting the arrays from the dataclass in `set_corrected_metadata()` and passing them individually to the JIT-compiled functions, we avoid the JIT compilation issue.

## Files Modified

1. [jaxtrace/gpu/search/point_in_tet_methods.py](jaxtrace/gpu/search/point_in_tet_methods.py)
   - Changed module-level storage from dataclass to individual arrays
   - Updated `set_corrected_metadata()` to extract arrays
   - Updated dispatcher to call array-based wrappers

2. [jaxtrace/gpu/search/aa_detection.py](jaxtrace/gpu/search/aa_detection.py)
   - Added `point_in_tet_pure_aa_arrays()` - array-based wrapper
   - Added `point_in_tet_branchless_hybrid_arrays()` - array-based wrapper
   - Original dataclass versions kept for completeness

## Status

✅ **FIXED** - Test should now run successfully

The corrected methods can now be called from within JIT-compiled search functions without dataclass serialization issues.

---

**Next**: User will run the production benchmark test manually to validate performance improvements.
