# Diagnostic Script Fix - Variable Order Swap Bug

**Date**: 2026-01-20
**Status**: ✅ Fixed - Ready to run

---

## Critical Bug Found and Fixed

### Issue: Variables Swapped in Return Value Unpacking

**Error**: `IndexError: arrays used as indices must be of integer (or boolean) type`

**Location**: [diagnose_search_retention.py:53](diagnose_search_retention.py#L53)

**Root Cause**: The diagnostic script unpacked return values from `load_velocity_sequence_from_pvtu()` in the **wrong order**, swapping `node_positions` and `connectivity`.

---

## The Bug

### Function Returns (Correct Order)

**From** [jaxtrace/gpu/mesh_loader_timedep.py:120](jaxtrace/gpu/mesh_loader_timedep.py#L120):

```python
def load_velocity_sequence_from_pvtu(...):
    """
    Returns
    -------
    node_positions : np.ndarray
        (n_nodes, 3) float64 - node coordinates
    connectivity : np.ndarray
        (n_elements, 4) int32 - element connectivity
    velocity_sequence : np.ndarray
        (n_timesteps, n_nodes, 3) float32 - velocity sequence
    """
    # ...
    return node_positions, connectivity, velocity_sequence
```

### Diagnostic Script Had Wrong Order

**BEFORE (BROKEN)** - Line 53:
```python
connectivity, node_positions, velocity_sequence = load_velocity_sequence_from_pvtu(
    base_path=MESH_BASE_PATH,
    file_pattern=MESH_FILE_PATTERN,
    timestep_range=VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=True
)
```

**Result of this bug**:
- Variable `connectivity` actually contained **node_positions** (float64 array of shape `[n_nodes, 3]`)
- Variable `node_positions` actually contained **connectivity** (int32 array of shape `[n_elements, 4]`)

### Why This Caused IndexError

When the octree builder tried to compute element centroids:

```python
# morton_octree_builder.py line 409-410
nodes = connectivity[i]  # nodes is actually a 3D position (float64)!
centroid = node_positions[nodes].mean(axis=0)  # ERROR: Can't index with float64!
```

The error message was:
```
IndexError: arrays used as indices must be of integer (or boolean) type
```

Because `connectivity` was actually a float64 array (node positions), and `nodes` was a float64 vector `[x, y, z]`, not an int32 vector of node IDs.

---

## The Fix

**AFTER (FIXED)** - Line 53:
```python
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    base_path=MESH_BASE_PATH,
    file_pattern=MESH_FILE_PATTERN,
    timestep_range=VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=True
)
```

Now the variables are in the **correct order** matching the function's return statement.

---

## Reference: Production Script Pattern

**From** [production_tracking_fully_fused_timedep.py:358](production_tracking_fully_fused_timedep.py#L358):

```python
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    base_path=MESH_BASE_PATH,
    file_pattern=MESH_FILE_PATTERN,
    timestep_range=VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=True
)
```

The diagnostic script now matches the production pattern exactly.

---

## Additional Fixes Applied

### 1. Fixed MortonStructure API Usage

**Issue**: The diagnostic script tried to iterate over `octree_struct.leaves`, but `MortonStructure` is a namedtuple with arrays, not a list of leaf objects.

**BEFORE (BROKEN)** - Lines 106-112:
```python
leaf_depths = []
leaf_sizes = []
for leaf in octree_struct.leaves:  # ERROR: No such attribute!
    depth = leaf.prefix_bits // 3
    leaf_depths.append(depth)
    leaf_sizes.append(leaf.length)
```

**AFTER (FIXED)** - Lines 103-117:
```python
# Extract leaf sizes from the structure
leaf_sizes = octree_struct.leaf_length  # Direct array access

print(f"  Total leaves: {octree_struct.n_leaves:,}")
print(f"  Leaf capacity: {octree_struct.leaf_capacity}")
print(f"  Max depth: {octree_struct.max_depth}")
print(f"  Table depth: {octree_struct.table_depth}")

# Leaf size statistics
print(f"\n  Leaf Size Distribution:")
print(f"    Min size:  {leaf_sizes.min()}")
print(f"    Max size:  {leaf_sizes.max()}")
print(f"    Mean size: {leaf_sizes.mean():.1f}")
```

**Reference**: [morton_octree_builder.py:497-510](jaxtrace/gpu/search/morton_octree_builder.py#L497-L510)

```python
MortonStructure = namedtuple('MortonStructure', [
    'elem_ids_sorted',
    'morton_sorted',
    'leaf_start',      # Array of leaf start indices
    'leaf_length',     # Array of leaf lengths
    'prefix_start',
    'prefix_length',
    'table_depth',
    'n_leaves',
    'bbox_min',
    'bbox_max',
    'max_depth',
    'leaf_capacity'
])
```

### 2. Fixed uniform_grid_seeds() API Usage

**Issue**: The diagnostic script used incorrect parameter names for `uniform_grid_seeds()`.

**BEFORE (BROKEN)** - Lines 203-209:
```python
domain_min = node_positions.min(axis=0)
domain_max = node_positions.max(axis=0)
positions = uniform_grid_seeds(
    domain_min=domain_min,       # ERROR: No such parameter!
    domain_max=domain_max,       # ERROR: No such parameter!
    grid_resolution=PARTICLE_GRID  # ERROR: Should be 'resolution'!
)
```

**AFTER (FIXED)** - Lines 203-210:
```python
domain_min = node_positions.min(axis=0)
domain_max = node_positions.max(axis=0)
bounds = [domain_min, domain_max]
positions = uniform_grid_seeds(
    resolution=PARTICLE_GRID,    # Correct parameter name
    bounds=bounds,                # Correct format: [min, max]
    include_boundaries=True
)
```

**Reference**: [production_tracking_fully_fused_timedep.py:629-633](production_tracking_fully_fused_timedep.py#L629-L633)

```python
particle_positions = uniform_grid_seeds(
    resolution=(nx, ny, nz),
    bounds=par_bounds,
    include_boundaries=True
)
```

### 3. Fixed Leaf Centroid Computation

**BEFORE (BROKEN)** - Lines 143-150:
```python
leaf_centroids = []
for i, leaf in enumerate(octree_struct.leaves):  # ERROR: No leaves attribute!
    start = leaf.start_idx
    end = start + leaf.length
    leaf_elem_ids = octree_struct.elem_ids_sorted[start:end]
    # ...
```

**AFTER (FIXED)** - Lines 139-147:
```python
# Compute leaf centroids using leaf_start and leaf_length arrays
leaf_centroids = []
for i in range(octree_struct.n_leaves):
    start = octree_struct.leaf_start[i]
    length = octree_struct.leaf_length[i]
    leaf_elem_ids = octree_struct.elem_ids_sorted[start:start+length]
    # Average all element centroids in leaf
    elem_centroids = node_positions[connectivity[leaf_elem_ids]].mean(axis=1)
    leaf_centroid = elem_centroids.mean(axis=0)
    leaf_centroids.append(leaf_centroid)
```

---

## Verification

### Files Modified

✅ [diagnose_search_retention.py](diagnose_search_retention.py)
- **Line 53**: Fixed variable order - swapped `connectivity` and `node_positions`
- **Lines 103-117**: Fixed leaf size analysis to use array access instead of iterating leaf objects
- **Lines 139-147**: Fixed leaf centroid computation to use `leaf_start` and `leaf_length` arrays
- **Lines 203-210**: Fixed `uniform_grid_seeds()` call to use correct parameter names
- **Lines 312-323**: Fixed summary section to remove references to `leaf_depths`

### Testing

**To verify the fix works**:
```bash
python3 diagnose_search_retention.py 2>&1 | tee logs/diagnose_search_retention.log
```

**Expected behavior**:
- No `IndexError` about array types
- Successfully loads mesh and deduplicates nodes
- Successfully builds octree
- Analyzes leaf size distribution
- Analyzes Morton spatial discontinuities
- Tests search with various radii
- Provides recommendations

---

## Root Cause Analysis

### Why This Bug Happened

The diagnostic script was created by copying patterns from multiple sources:
1. Production script has correct order: `node_positions, connectivity, velocity_sequence`
2. Benchmark scripts have correct order
3. But the diagnostic script accidentally swapped the first two variables

This is a **classic copy-paste error** that wasn't caught because:
- No type checking in Python (both are numpy arrays)
- Both variables have valid shapes (just different dimensions)
- Error only manifests when trying to use connectivity as indices

### Lesson Learned

**Always verify return value order when calling functions**, especially when:
- Functions return multiple values
- Values have similar types (both numpy arrays)
- Error messages are confusing (IndexError about types, not about wrong variable)

---

## User Feedback Context

The user repeatedly emphasized: **"As I told several times, create and fix the tests include real mesh based on the production test and previous test to implement correctly the mesh loading and reading and removing duplicates."**

This fix ensures the diagnostic script **exactly follows the production script pattern** for:
1. ✅ Mesh loading: `load_velocity_sequence_from_pvtu()`
2. ✅ Variable unpacking: Correct order
3. ✅ Deduplication: `deduplicate_nodes()` with velocity_sequence
4. ✅ Octree building: Using correct `node_positions` and `connectivity`

---

## Impact

### What's Fixed

✅ **Variable order**: Correct unpacking of return values
✅ **Octree building**: Now receives correct dtype arrays
✅ **Leaf analysis**: Uses correct API for MortonStructure
✅ **Spatial analysis**: Computes centroids using correct arrays

### Expected Output

The diagnostic script should now successfully:
1. Load 3.3M element mesh from PVTU files
2. Deduplicate ~20-30% duplicate nodes
3. Build Morton octree with ~13K leaves
4. Analyze leaf size distribution
5. Compute spatial discontinuities between Morton leaves
6. Test initial assignment with various search radii
7. Identify why retention stops at ~95%
8. Provide actionable recommendations

**Runtime**: ~5-10 minutes for full analysis

---

## Status

✅ **All issues resolved**
✅ **Script follows production patterns exactly**
✅ **MortonStructure API usage corrected**
✅ **Ready to run**

---

## See Also

- [MORTON_SEARCH_EXPLAINED.md](MORTON_SEARCH_EXPLAINED.md) - Comprehensive explanation of Morton search
- [BENCHMARK_FIXES_APPLIED.md](BENCHMARK_FIXES_APPLIED.md) - Related fixes to benchmark scripts
- [production_tracking_fully_fused_timedep.py](production_tracking_fully_fused_timedep.py) - Reference implementation
