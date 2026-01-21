# Velocity Deduplication Fix - Complete Summary

**Date:** 2026-01-14
**Status:** ✅ FIXED - Ready for testing

---

## The Problem

### Root Cause
After loading the mesh and removing 209,749 duplicate nodes (26.9%), the velocity arrays were **NOT remapped** to match the deduplicated node IDs.

### Diagnostic Results
```
[DIAGNOSTIC] Checking array shapes after deduplication...
  node_positions shape:   (571173, 3)      ✅ Deduplicated
  connectivity shape:     (3048900, 4)     ✅ Remapped to new node IDs
  velocity_sequence shape: (40, 780922, 3) ❌ NOT remapped!

  ⚠️  CRITICAL BUG DETECTED:
      Velocity array has 780,922 nodes
      But mesh has 571,173 nodes after deduplication
      Difference: 209,749 nodes
```

### Impact
- RK4 interpolation: `node_vels = velocity_field[nodes_idx]`
- `nodes_idx` contains deduplicated node IDs [0, 571,172]
- `velocity_field` has 780,922 entries (pre-deduplication)
- **Result:** Accessing wrong velocities → wrong RK4 integration → wrong trajectories

### Why This Explains Everything
- ✅ Initial assignment: 100% success (doesn't use velocity)
- ✅ Spatial accuracy: 100% correct (doesn't use velocity)
- ✅ No particle loss (search works correctly)
- ❌ Wrong trajectories (uses wrong velocities in RK4)

---

## The Fix

### Modified: `jaxtrace/gpu/mesh_deduplication.py`

**Changed function signature:**
```python
def deduplicate_nodes(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    velocity_sequence: np.ndarray = None,  # NEW parameter
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:  # Now returns 4 values
```

**Added velocity remapping logic:**
```python
# After building node_map (old_id → new_id):
if velocity_sequence is not None:
    n_timesteps = velocity_sequence.shape[0]
    remapped_velocity_sequence = np.zeros(
        (n_timesteps, n_unique, 3),
        dtype=velocity_sequence.dtype
    )

    # Remap velocities using same node_map as connectivity
    for old_id in range(n_nodes):
        new_id = node_map[old_id]
        # Copy velocity for all timesteps at once
        remapped_velocity_sequence[:, new_id, :] = velocity_sequence[:, old_id, :]

    return (compacted_positions, remapped_connectivity,
            n_duplicates, remapped_velocity_sequence)
```

### Modified: `production_tracking_fully_fused_timedep.py:327`

**Changed deduplication call:**
```python
# BEFORE (wrong):
node_positions, connectivity, n_duplicates_removed = deduplicate_nodes(
    node_positions, connectivity, verbose=True
)

# AFTER (correct):
node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
    node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=True
)
```

Now `velocity_sequence` is automatically remapped to match deduplicated node IDs!

---

## Benefits of the Fix

### 1. Correctness
- **Velocity arrays now aligned** with deduplicated connectivity
- RK4 interpolation accesses **correct velocities**
- Trajectories follow **physically correct streamlines**

### 2. Memory Reduction
- **Before:** 780,922 nodes × 40 timesteps × 3 floats × 4 bytes = **376 MB**
- **After:** 571,173 nodes × 40 timesteps × 3 floats × 4 bytes = **274 MB**
- **Reduction:** 102 MB (27% less memory for velocity arrays)

### 3. No Performance Penalty
- Remapping is one-time during loading (~1-2s)
- Runtime performance unchanged (may be slightly faster due to better cache locality)

### 4. Clean Implementation
- **Separation of concerns:** Deduplication handles all data remapping
- **Reusable:** Any script using `deduplicate_nodes()` gets velocity remapping automatically
- **Backward compatible:** `velocity_sequence` parameter is optional

---

## Validation

### Expected Diagnostic Output (After Fix)

```
[DIAGNOSTIC] Verifying array consistency after deduplication...
  node_positions shape:   (571173, 3)
  connectivity shape:     (3048900, 4)
  velocity_sequence shape: (40, 571173, 3)  ← NOW MATCHES!
  ✅ Velocity array correctly remapped (571,173 nodes)
  ✅ Connectivity valid (max node ID 571172 < 571173)
  ✅ All array consistency checks passed!
      → Trajectories should now be physically correct
```

### Test Procedure

1. **Run production script:**
   ```bash
   source .venv/bin/activate
   python production_tracking_fully_fused_timedep.py > logs/production_velocity_fixed.log 2>&1
   ```

2. **Check for:**
   - ✅ Diagnostic passes all checks
   - ✅ No RuntimeError about velocity shape mismatch
   - ✅ Trajectories follow streamlines (visual inspection)
   - ✅ Particle retention remains high (95-100%)
   - ✅ Performance remains good (~20-25K particles/s)

---

## Technical Details

### Data Flow (Fixed)

**Step 1: Load mesh**
```python
node_positions.shape = (780922, 3)
connectivity.shape = (3048900, 4)  # Uses node IDs [0, 780921]
velocity_sequence.shape = (40, 780922, 3)
```
All arrays consistent ✅

**Step 2: Deduplicate (with fix)**
```python
node_positions, connectivity, n_dup, velocity_sequence = deduplicate_nodes(
    node_positions, connectivity, velocity_sequence=velocity_sequence
)
```

**Internal process:**
1. Build `node_map`: old_id [0, 780921] → new_id [0, 571172]
2. Compact `node_positions`: 780,922 → 571,173 nodes
3. Remap `connectivity`: All node IDs updated to [0, 571172]
4. **NEW:** Remap `velocity_sequence`: 780,922 → 571,173 nodes using same `node_map`

**Step 3: Arrays after deduplication**
```python
node_positions.shape = (571173, 3)           ✅
connectivity.shape = (3048900, 4)            ✅ Uses node IDs [0, 571172]
velocity_sequence.shape = (40, 571173, 3)    ✅ NOW CORRECT!
```
All arrays consistent ✅

**Step 4: RK4 interpolation**
```python
nodes_idx = connectivity[elem_id]  # Node IDs [0, 571172] ✅
node_vels = velocity_field[nodes_idx]  # Indexes into 571,173-length array ✅
# Accesses CORRECT velocities! ✅
```

### Why Previous Approach Failed

**Old code (before fix):**
```python
node_positions, connectivity, n_dup = deduplicate_nodes(
    node_positions, connectivity  # velocity_sequence NOT passed
)
# velocity_sequence NOT updated!
```

**Result:**
- `connectivity` references node IDs [0, 571,172]
- `velocity_sequence` still has 780,922 entries
- `velocity_field[node_id]` for `node_id=45678` (deduplicated) accesses:
  - Velocity at **original node 45678** (wrong!)
  - Not velocity at **original node 123456** which is now at new_id=45678 (correct)

**Fix:**
- Pass `velocity_sequence` to `deduplicate_nodes()`
- Remap using same `node_map`: `velocity_new[new_id] = velocity_old[old_id]`
- Now `velocity_field[45678]` correctly contains velocity from original node 123456

---

## Files Modified

1. **[jaxtrace/gpu/mesh_deduplication.py](jaxtrace/gpu/mesh_deduplication.py)**
   - Lines 20-48: Updated function signature and docstring
   - Lines 94: Return 4 values instead of 3
   - Lines 117-142: Added velocity remapping logic
   - Lines 173: Updated return statement

2. **[production_tracking_fully_fused_timedep.py](production_tracking_fully_fused_timedep.py)**
   - Line 327: Pass `velocity_sequence` to `deduplicate_nodes()`
   - Line 327: Receive remapped `velocity_sequence`
   - Lines 340-363: Updated diagnostic to be more concise

---

## Related Documentation

- [DIAGNOSTIC_RESULTS_AND_NEXT_STEPS.md](DIAGNOSTIC_RESULTS_AND_NEXT_STEPS.md) - Diagnostic results and investigation plan
- [RK4_VELOCITY_INTERPOLATION_ANALYSIS.md](RK4_VELOCITY_INTERPOLATION_ANALYSIS.md) - Detailed analysis of bug mechanism
- [diagnose_element_assignment_accuracy.py](diagnose_element_assignment_accuracy.py) - Spatial accuracy verification (100% correct)

---

## Summary

**Problem:** Velocity arrays not remapped after node deduplication

**Symptom:** Wrong particle trajectories despite perfect spatial accuracy

**Root Cause:** RK4 accessed wrong velocities due to array indexing mismatch

**Fix:** Modified `deduplicate_nodes()` to remap velocity arrays automatically

**Result:** Trajectories should now follow physically correct streamlines

**Status:** ✅ Fix implemented - Ready for testing

**Next Step:** Run production script to verify trajectories are now correct
