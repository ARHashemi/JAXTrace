# Timestep Mapping Fix for Direct Interpolation

## Problem

When implementing the direct interpolation mode, an `IndexError` occurred:

```python
IndexError: list index out of range
# at shared_octree.get_fine_level_for_timestep(timestep_idx)
```

## Root Cause

**Mismatch between global timestep indices and revolution cycle indices:**

- **Global mesh files**: 160 files total (timesteps 0-159)
  - Refinement phase: timesteps 0-119
  - Revolution cycle: timesteps 120-159 (40 timesteps)

- **SharedOctreeStructure storage**: Only stores fine octrees for revolution cycle
  - `fine_levels_per_timestep`: List with 40 entries (indices 0-39)
  - Maps to revolution cycle timesteps 120-159

- **Direct interpolator creation**: Was passing global timestep index
  - Called `create_direct_octree_fem_interpolator(..., timestep_idx=120)`
  - Then called `shared_octree.get_fine_level_for_timestep(120)`
  - But `fine_levels_per_timestep` only has indices 0-39 → **IndexError!**

## Solution

**Added timestep mapping from global indices to revolution cycle indices:**

### 1. Store Revolution Cycle Offset

In `SharedOctreeFEMTimeSeriesField.__init__()`:

```python
# Revolution cycle is the LAST N timesteps
revolution_timesteps = shared_octree_config.get('revolution_timesteps', 40)
reference_timestep = max(0, len(mesh_files) - revolution_timesteps)

# Store revolution cycle offset for timestep mapping
# Global timestep -> Revolution cycle index = global_idx - revolution_start_idx
self.revolution_start_idx = reference_timestep  # 120
self.revolution_end_idx = len(mesh_files) - 1   # 159
```

### 2. Map Indices Before Interpolator Creation

In `_sample_with_direct_interpolation()`:

```python
# Get or create direct interpolator for this timestep
if left_idx not in self._direct_interpolator_cache:
    # Map global timestep index to revolution cycle index
    revolution_idx = left_idx - self.revolution_start_idx
    # Example: left_idx=120 -> revolution_idx=0
    #          left_idx=159 -> revolution_idx=39

    self._direct_interpolator_cache[left_idx] = create_direct_octree_fem_interpolator(
        self.shared_octree,
        self.reference_positions,
        self.reference_connectivity,
        revolution_idx  # Use revolution cycle index (0-39), not global index (120-159)
    )
```

### 3. Cache Using Global Index

**Important design decision**: Cache interpolators using **global timestep index** as key, but pass **revolution cycle index** to the factory.

This allows the field to correctly handle temporal interpolation across the entire time range while internally mapping to the correct fine octree.

## Example Mapping

```
Global Index → Revolution Index → Fine Octree
─────────────────────────────────────────────
   0-119     → N/A (refinement)  → Error (topology mismatch)
   120       → 0                 → fine_levels_per_timestep[0]
   121       → 1                 → fine_levels_per_timestep[1]
   ...
   159       → 39                → fine_levels_per_timestep[39]
```

## Files Modified

### `jaxtrace/fields/shared_octree_fem_field.py`

**Lines 105-108**: Store revolution cycle range
```python
self.revolution_start_idx = reference_timestep
self.revolution_end_idx = len(mesh_files) - 1
```

**Lines 347-354**: Map indices for single timestep interpolation
```python
revolution_idx = left_idx - self.revolution_start_idx
self._direct_interpolator_cache[left_idx] = create_direct_octree_fem_interpolator(
    self.shared_octree,
    self.reference_positions,
    self.reference_connectivity,
    revolution_idx  # Use revolution cycle index
)
```

**Lines 389-405**: Map indices for temporal interpolation
```python
revolution_idx_left = left_idx - self.revolution_start_idx
revolution_idx_right = right_idx - self.revolution_start_idx
# ... create interpolators using revolution indices ...
```

**Lines 337, 379**: Updated error messages to show actual revolution cycle range
```python
f"Revolution cycle: timesteps {self.revolution_start_idx}-{self.revolution_end_idx} (constant topology)\n\n"
f"   config['time_span'] = ({float(self.revolution_start_idx)}, {float(self.revolution_end_idx)})"
```

## Testing

To test the fix:

```python
# In example_workflow.py config:
user_config = {
    # ... other config ...
    'use_direct_interpolation': True,  # Enable direct mode
    'time_span': (120.0, 159.0),       # Match revolution cycle range
}
```

Expected behavior:
- ✅ Field initializes successfully
- ✅ Tracking starts at t=120.0
- ✅ Direct interpolators created with correct indices (0-39)
- ✅ No IndexError
- ✅ Memory usage: ~1 MB (vs 5-8 GB for legacy mode)

## Summary

The fix ensures that when the direct interpolation mode accesses fine octrees from the `SharedOctreeStructure`, it correctly maps global timestep indices (120-159) to revolution cycle indices (0-39), preventing the `IndexError` and enabling the memory-efficient direct interpolation path.
