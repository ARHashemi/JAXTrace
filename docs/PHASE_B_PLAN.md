# Phase B: Per-Timestep Data Loading for True AMR Support

**Status**: 🚧 **REQUIRED** - Current filtering approach is a temporary workaround
**Priority**: **HIGH** - Needed for production use with real AMR data

---

## Problem Statement

### Current Situation (Phase A)

The workflow **pre-loads** all velocity data into a single NumPy array:
```python
velocity_data = np.array(velocity_data, dtype=np.float32)  # Shape: (T, N, 3)
```

This requires **uniform array shape** - all timesteps must have the same number of nodes (N).

### Real AMR Data

Real AMR data has **varying node counts** across timesteps:

**Example: ThreadedA Dataset** (40 timesteps, 120-159):
```
- 39 different mesh sizes (almost every timestep different!)
- Range: 898,509 to 907,623 nodes (~1% variation)
- Octree reuse: 97.5% (structure is stable despite node count changes)
```

**Example: FLA Dataset** (40 timesteps, 120-159):
```
- 4 different mesh sizes (more stable)
- 780,922 nodes: 37 timesteps
- 780,933, 781,466, 790,285: 1 timestep each
- Octree reuse: 97.5%
```

### Current Workaround (Wrong!)

Phase A filters to the "most common mesh size":
- ThreadedA: Would keep only 2/40 timesteps! (each size appears 1-2 times)
- FLA: Keeps 37/40 timesteps (acceptable but not ideal)

**This violates the design**: We should use **ALL** timesteps, not filter them.

---

## Why Shared Octree Still Works

**Key Insight**: The octree structure is based on **spatial layout**, not node count!

Even with 39 different mesh sizes, the octree shows 97.5% reuse because:
1. AMR adds/removes nodes **locally** (refines specific regions)
2. Overall **spatial structure** remains stable
3. Coarse octree captures the **general layout**
4. Fine octree differences are detected by hash comparison

**This proves the shared octree strategy is correct!** We just need to fix the data loading.

---

## Solution: Per-Timestep Data Loading

### Design

Instead of pre-loading velocity data, **load it on-demand** during interpolation:

**Current (Phase A)**:
```python
# Pre-load ALL velocity data
velocity_data = []
for file in files:
    vel = load_velocity(file)
    velocity_data.append(vel)
velocity_data = np.array(velocity_data)  # ❌ Requires uniform shape

field = SharedOctreeFEMTimeSeriesField(
    data=velocity_data,  # Pass pre-loaded data
    times=times,
    positions=points,     # From ONE mesh
    connectivity=connectivity  # From ONE mesh
)
```

**Proposed (Phase B)**:
```python
# Store file paths, load per-timestep
field = SharedOctreeFEMTimeSeriesField(
    mesh_files=files,     # Pass file paths
    times=times,
    # No pre-loaded data!
)

# In field.sample_at_positions(query_pos, t):
#   1. Find timestep index for time t
#   2. Load velocity, positions, connectivity from mesh_files[timestep_idx]
#   3. Use cached octree structure (already built!)
#   4. Interpolate using loaded data
#   5. Cache recently used timesteps
```

### Architecture Changes

**Modified Files**:
1. `jaxtrace/fields/shared_octree_fem_field.py`
   - Change constructor to accept `mesh_files` instead of `data`
   - Implement `_load_timestep_data(timestep_idx)` method
   - Add LRU cache for recently loaded timesteps

2. `jaxtrace/fields/octree_fem_time_series_optimized.py` (base class)
   - May need to check if per-timestep loading is compatible
   - Possibly create a new base class or modify interpolation method

3. `example_workflow.py`
   - Remove velocity data pre-loading loop
   - Remove mesh size filtering
   - Pass mesh file paths to field constructor

### Implementation Steps

#### Step 1: Modify SharedOctreeFEMTimeSeriesField

```python
class SharedOctreeFEMTimeSeriesField:
    def __init__(
        self,
        mesh_files: List[str],
        times: np.ndarray,
        shared_octree_config: Optional[Dict[str, Any]] = None,
        cache_size: int = 3,  # Cache last 3 timesteps
        **kwargs
    ):
        # Build shared octree (already working!)
        config = SharedOctreeConfig(**shared_octree_config)
        factory = SharedOctreeFactory(config)
        self.shared_octree = factory.build_from_files(mesh_files)

        # Store file paths, not data
        self.mesh_files = mesh_files
        self.times = times

        # Initialize cache
        self._timestep_cache = {}  # {timestep_idx: (data, positions, connectivity)}
        self._cache_order = []     # LRU tracking
        self._cache_size = cache_size

    def _load_timestep_data(self, timestep_idx: int):
        """Load velocity, positions, connectivity for a specific timestep."""
        if timestep_idx in self._timestep_cache:
            return self._timestep_cache[timestep_idx]

        # Load from file
        file_path = self.mesh_files[timestep_idx]
        reader = vtk.vtkXMLPUnstructuredGridReader()
        reader.SetFileName(file_path)
        reader.Update()
        mesh = reader.GetOutput()

        # Extract data
        positions = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float32)

        # Extract connectivity
        connectivity = []
        for i in range(mesh.GetNumberOfCells()):
            cell = mesh.GetCell(i)
            if cell.GetCellType() == vtk.VTK_TETRA:
                point_ids = cell.GetPointIds()
                connectivity.append([point_ids.GetId(j) for j in range(4)])
        connectivity = np.array(connectivity, dtype=np.int32)

        # Extract velocity
        point_data = mesh.GetPointData()
        for name in ['Displacement', 'displacement', 'Velocity', 'velocity']:
            if point_data.HasArray(name):
                velocity = vtk_to_numpy(point_data.GetArray(name)).astype(np.float32)
                break

        # Cache it
        self._timestep_cache[timestep_idx] = (velocity, positions, connectivity)
        self._cache_order.append(timestep_idx)

        # Evict old entries
        if len(self._cache_order) > self._cache_size:
            old_idx = self._cache_order.pop(0)
            del self._timestep_cache[old_idx]

        return velocity, positions, connectivity

    def sample_at_positions(self, query_positions: np.ndarray, t: float) -> np.ndarray:
        """Sample field at positions using per-timestep data."""
        # Find timestep
        timestep_idx = self._find_timestep_for_time(t)

        # Load data for this timestep
        velocity, positions, connectivity = self._load_timestep_data(timestep_idx)

        # Use shared octree structure (already built!)
        # Interpolate using octree + per-timestep mesh data
        # ... (implement interpolation logic)
```

#### Step 2: Update Workflow

```python
# Remove this entire pre-loading loop:
# velocity_data = []
# for file in files:
#     vel = load_velocity(file)
#     velocity_data.append(vel)
# velocity_data = np.array(velocity_data)

# Remove mesh size filtering

# Simply pass file paths:
field = create_shared_octree_fem_field(
    mesh_files=all_files,    # Just file paths!
    times=times,             # Time array
    user_config=config
)
```

### Challenges

1. **Base Class Compatibility**: The base `OctreeFEMTimeSeriesFieldOptimized` expects pre-loaded data
   - **Solution**: Override `sample_at_positions()` in `SharedOctreeFEMTimeSeriesField`
   - Or create a new base class for per-timestep loading

2. **Performance**: Loading from disk per interpolation could be slow
   - **Solution**: LRU cache keeps recent timesteps in memory
   - Particle tracking typically accesses sequential timesteps (good for caching)

3. **Memory Management**: Need to balance cache size vs memory usage
   - **Solution**: Configurable cache size (default: 3 timesteps)
   - Monitor memory usage during testing

4. **Octree Mesh Mismatch**: Octree built once, but mesh varies per timestep
   - **Solution**: Octree is spatial structure (elements in space), works with any mesh that fits the same spatial domain
   - The octree tells us "which elements to check", then we use per-timestep connectivity to find actual element

---

## Expected Results

### Performance

| Metric | Phase A (filtered) | Phase B (per-timestep) | Change |
|--------|-------------------|------------------------|--------|
| Timesteps used | 37/40 (FLA) or 2/40 (ThreadedA) | 40/40 | +8% to +1900% |
| Data coverage | 92.5% or 5% | 100% | Complete |
| Memory (data) | 331 MB (all pre-loaded) | ~100 MB (cached) | -70% |
| Build time | 5.5 min | 5.5 min | Same |
| Tracking time | 103s | ~110s (+loading) | +7% |

### Benefits

✅ **Complete data coverage** - No filtering, all timesteps used

✅ **Lower memory usage** - Only cache 2-3 timesteps instead of all 40

✅ **True AMR support** - Handles any mesh variation

✅ **Matches design** - Implements original plan correctly

### Risks

⚠️ **Loading overhead** - Disk I/O during particle tracking
- Mitigated by caching
- Sequential access pattern is cache-friendly

⚠️ **Implementation complexity** - Need to override base class behavior
- Well-defined interface
- Can test incrementally

---

## Testing Plan

### Unit Tests

1. **Per-Timestep Loading**:
   - Test `_load_timestep_data()` for single timestep
   - Verify positions, connectivity, velocity loaded correctly
   - Test with different mesh sizes

2. **Cache Management**:
   - Test LRU eviction
   - Verify cache hit/miss behavior
   - Test memory usage

3. **Interpolation**:
   - Test `sample_at_positions()` with cached data
   - Test with uncached data (triggers loading)
   - Verify results match pre-loaded approach

### Integration Tests

1. **FLA Dataset** (4 mesh sizes):
   - Run full workflow with 40 timesteps
   - Verify all 40 timesteps used
   - Compare results to Phase A (should match for 37 common timesteps)

2. **ThreadedA Dataset** (39 mesh sizes):
   - Run full workflow with 40 timesteps
   - Verify all 40 timesteps used
   - Confirm 97.5% octree reuse maintained

3. **Performance Test**:
   - Measure build time
   - Measure tracking time (with cache hits/misses)
   - Measure memory usage (peak and steady-state)

### Regression Tests

- Ensure FLA 37-timestep results still work
- Verify backward compatibility
- Test with uniform mesh data (no AMR)

---

## Estimated Effort

| Task | Time | Notes |
|------|------|-------|
| Modify `SharedOctreeFEMTimeSeriesField` | 2-3 hours | Core implementation |
| Update workflow | 30 min | Remove pre-loading |
| Test with FLA | 30 min | Known dataset |
| Test with ThreadedA | 1 hour | New, complex dataset |
| Documentation | 30 min | Update docs |
| **Total** | **4-5 hours** | Can be done in one session |

---

## Alternative Approaches (Not Recommended)

### Alternative 1: Pad Arrays to Max Size

```python
max_size = max(mesh_sizes)
padded_data = [np.pad(vel, ...) for vel in velocity_data]
```

**Problems**:
- Wastes memory (padding can be significant)
- Need to pad positions and connectivity too
- Padding indices would break connectivity
- Complex and error-prone

### Alternative 2: Multiple Field Objects

```python
fields = [create_field(data[i], ...) for i in range(40)]
```

**Problems**:
- Defeats purpose of shared octree
- 40× memory usage for octree structures
- No reuse benefits
- Back to old approach

### Alternative 3: Interpolate to Common Grid

```python
# Interpolate all timesteps onto a fixed grid
common_grid_data = [interpolate_to_grid(vel, ...) for vel in velocity_data]
```

**Problems**:
- Loses FEM accuracy
- Extra interpolation step (slow)
- Requires choosing grid resolution
- Not true AMR support

---

## Recommendation

**Proceed with Phase B: Per-Timestep Data Loading**

This is the correct solution that:
- ✅ Matches original design intent
- ✅ Provides true AMR support
- ✅ Uses all timesteps (no filtering)
- ✅ Reduces memory usage
- ✅ Maintains octree reuse benefits
- ✅ Can be implemented in 4-5 hours

**Next Steps**:
1. Commit current Phase A state with documentation of limitation
2. Create Phase B branch (or continue on dynamic_octree)
3. Implement per-timestep loading
4. Test with both FLA and ThreadedA datasets
5. Document and commit Phase B

---

**Author**: ARHashemi + Claude
**Date**: October 15, 2025
**Branch**: dynamic_octree
**Status**: 📋 Plan ready for implementation
