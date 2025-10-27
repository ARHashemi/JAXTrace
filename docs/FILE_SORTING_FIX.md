# File Sorting Fix - Non-Monotonic Times Warning

## Issue

**Warning Message**:
```
/home/arhashemi/Workspace/welding/JAXTrace/jaxtrace/fields/time_series.py:135: UserWarning: Times are not monotonically non-decreasing; interpolation may be unreliable
```

## Root Cause

**Location**: `example_workflow.py:460`

**Problem**: Files sorted **lexicographically** (as strings) instead of **numerically**:

```python
# OLD CODE (WRONG):
files = sorted(glob(vtk_pattern))
```

**Why this fails**:
- Lexicographic sorting: `"0", "1", "10", "100", "101", ..., "2", "20", ...`
- Numeric sorting: `0, 1, 2, ..., 10, ..., 100, 101, ...`

**Example**:
```
Lexicographic order: 0, 1, 10, 100, 101, ..., 109, 11, 110, ..., 2, 20, ...
                      ↑  ↑  ↑   ↑                    ↑              ↑
                      OK OK ❌ WRONG - 10 < 100 but 100 comes before 11!

Actual time sequence: 0, 1, 10, 100, 101, ..., 109, 11, 110, ...
                           ✓  ❌ JUMP BACK!        ❌ JUMP BACK!
```

## Impact

### On Phase A (Pre-loading):
- Times array: `[0, 1, 10, 100, ..., 11, ...]`
- Velocity data loaded in same wrong order
- Temporal interpolation searches for wrong timesteps
- **Result**: Wrong velocities at certain simulation times

### On Phase B (Per-timestep loading):
- Times array: `[0, 1, 10, 100, ..., 11, ...]`
- When sampling at time `t`, finds wrong bracketing indices
- Loads wrong timestep files
- **Result**: Wrong velocities everywhere that crosses these boundaries

### Example Scenario:

```python
times = [0, 1, 10, 100, 101, ..., 109, 11, 110, ...]
                    ↑                   ↑
                    100                 11

# Query at t=50 (halfway between timesteps 10 and 100)
# Searchsorted finds: left_idx=2 (time=10), right_idx=3 (time=100)
# Alpha = (50-10)/(100-10) = 0.44
# Interpolates between timestep 10 and 100 ✅ CORRECT

# Query at t=105 (halfway between timesteps 100 and 110)
# Searchsorted finds: left_idx=7 (time=109), right_idx=8 (time=11) ❌ WRONG!
# Alpha = (105-109)/(11-109) = 0.04 ❌ NEGATIVE DENOMINATOR!
# Interpolates between timestep 109 and 11 ❌ COMPLETELY WRONG
```

## The Fix

**Location**: `example_workflow.py:462-470`

```python
# CRITICAL FIX: Sort files NUMERICALLY by timestep number, not lexicographically
def extract_timestep(filename):
    match = re.search(r'_(\d+)\.pvtu$', filename)
    return int(match.group(1)) if match else 0

files = sorted(glob(vtk_pattern), key=extract_timestep)
```

## Verification

**Before fix**:
```python
files = sorted(glob("featurelessAvtk_*.pvtu"))
times = [0, 1, 10, 100, 101, ..., 109, 11, 110, ...]
monotonic = False  # ❌ WARNING TRIGGERED
```

**After fix**:
```python
files = sorted(glob("featurelessAvtk_*.pvtu"), key=extract_timestep)
times = [0, 1, 2, 3, ..., 10, 11, ..., 100, 101, ..., 159]
monotonic = True   # ✅ NO WARNING
```

## Impact Assessment

### Accuracy:
- **Before**: Temporal interpolation completely wrong at certain times
- **After**: Correct temporal interpolation at all times ✅

### Performance:
- No performance impact (sorting is done once at startup)

### Scope:
- Affects **ALL** datasets with filenames like `name_X.pvtu` where X > 9
- FLA dataset: 160 files (0-159) → AFFECTED
- Any dataset with timesteps 0-9 only → NOT affected (already sorted correctly)

## Why This Wasn't Caught Earlier

1. **Warning was there** but may have been overlooked in logs
2. **Errors subtle**: Only affects certain query times (where interpolation crosses the non-monotonic boundary)
3. **May have contributed** to reported velocity errors (along with octree element assignment issue)

## Related Issues

This file sorting bug is **independent** from:
1. ✅ Octree element assignment bug (fixed in `octree_fem_interpolator_optimized.py`)
2. ✅ Phase B mesh mismatch bug (fixed - use revolution cycle mesh)
3. ✅ Cheap fallback issue (pre-existing in base interpolator)

All four issues could contribute to velocity errors:
- **File sorting**: Wrong temporal interpolation at certain times
- **Octree assignment**: Wrong spatial interpolation at boundaries
- **Mesh mismatch**: Completely wrong octree for Phase B
- **Cheap fallback**: Wrong interpolation when element search fails

## Files Modified

1. `example_workflow.py:462-470` - Numeric sorting of files
2. `docs/FILE_SORTING_FIX.md` - This document

## Testing

```bash
# Run workflow and verify no warning
python example_workflow.py 2>&1 | grep "monotonically"
# Should return nothing (no warning)

# Check times are actually sorted
# Add temporary print in workflow after loading:
# print(f"Times: {times[:20]}")
# Should print: [0. 1. 2. 3. ... 19.]
```

## Conclusion

**Critical fix** that ensures temporal interpolation works correctly.

**Combined with octree element assignment fix**, this should significantly improve velocity accuracy throughout the simulation.

Both fixes are **independent** and **complementary**:
- File sorting: Fixes temporal interpolation
- Octree assignment: Fixes spatial interpolation
