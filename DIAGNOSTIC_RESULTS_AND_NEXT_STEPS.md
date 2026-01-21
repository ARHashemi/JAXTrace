# Diagnostic Results and Next Steps

**Date:** 2026-01-14
**Status:** Initial assignment working, spatial accuracy perfect, investigating wrong trajectories

---

## Diagnostic Test Results Summary

### Test Configuration
- Mesh: featurelessAvtk_120.pvtu (3,048,900 elements, 571,173 nodes after deduplication)
- Test particles: 10,000 in refined region
- Search method: Cascading radius (500 → 1000 → 2000 → 5000 → 10000 → 100000)

### Results

```
Initial assignment: 100.00% (10,000/10,000)
  - radius=500:    93.05% (9,305/10,000)
  - radius=1000:   +3.14% (314 more → 96.19%)
  - radius=2000:   +1.86% (186 more → 98.05%)
  - radius=5000:   +1.53% (153 more → 99.58%)
  - radius=10000:  +0.08% (8 more → 99.66%)
  - radius=100000: +0.34% (34 more → 100.00%)

Spatial accuracy verification:
  Method 1 (Barycentric):   100.00% correct (10,000/10,000)
  Method 2 (Signed Volume): 100.00% correct (10,000/10,000)
  Disagreements: 0 (0.00%)
```

---

## Key Findings

### ✅ What's Working Correctly

1. **Initial Assignment**: 100% success with cascading radii
   - Most particles (93%) found with small radius (500 leaves ≈ 2% of mesh)
   - Remaining 7% found with larger radii
   - This shows the system CAN find particles when given sufficient radius

2. **Spatial Accuracy**: Perfect (100%)
   - Both barycentric and signed volume methods agree
   - NO spatial inaccuracy bugs in `point_in_tet_gpu()`
   - NO Morton range spatial mismatch issues
   - All assigned particles are genuinely inside their elements

3. **Search Algorithm**: Fundamentally sound
   - `position_to_leaf_id_octree()` works (eventually finds correct leaf)
   - `search_in_leaf_global()` works (finds correct element in leaf)
   - `point_in_tet_gpu()` works (correctly verifies containment)

### ❓ What Remains to Investigate

**Primary Suspect: Velocity array mismatch after deduplication**

The diagnostic does NOT test velocity interpolation. Initial assignment accuracy doesn't guarantee trajectory accuracy if velocities are wrong.

**Hypothesis:**
```python
# After loading (line 300-306 in production script):
velocity_sequence.shape = (40, 780922, 3)  # 40 timesteps, 780,922 nodes (WITH duplicates)

# After deduplication (line 327-329):
node_positions.shape = (571173, 3)  # 571,173 nodes (AFTER removing 209,749 duplicates)

# ⚠️ CRITICAL: velocity_sequence NOT updated!
# velocity_sequence still has 780,922 node entries
# But connectivity references node IDs in range [0, 571,173]
```

**Impact:**
- During RK4, when interpolating velocity at element nodes
- Connectivity returns node IDs in range [0, 571,173] (deduplicated)
- Velocity array indexed with these IDs accesses WRONG velocities
- Wrong velocity → Wrong RK4 integration → Wrong trajectories

---

## Investigation Plan

### Priority 1: Verify Velocity Array Mismatch ✅ COMPLETED

**Status:** Bug CONFIRMED and FIXED!

**Diagnostic Results:**
```
[DIAGNOSTIC] Checking array shapes after deduplication...
  node_positions shape:   (571173, 3)
  connectivity shape:     (3048900, 4)
  velocity_sequence shape: (40, 780922, 3)  ← MISMATCH!
  Nodes removed:          209,749

  ⚠️  CRITICAL BUG DETECTED:
      Velocity array has 780,922 nodes
      But mesh has 571,173 nodes after deduplication
      Difference: 209,749 nodes
```

**Fix Implemented:**
Modified `deduplicate_nodes()` in [jaxtrace/gpu/mesh_deduplication.py](jaxtrace/gpu/mesh_deduplication.py) to:
1. Accept optional `velocity_sequence` parameter
2. Remap velocity arrays using same `node_map` as connectivity
3. Return remapped velocity arrays with correct shape

**Production script updated:**
```python
node_positions, connectivity, n_duplicates, velocity_sequence = deduplicate_nodes(
    node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=True
)
```

Now velocity arrays are automatically remapped when nodes are deduplicated.

### Priority 2: Check Velocity Interpolation in RK4 ✅ COMPLETED

**Analysis Complete:** See [RK4_VELOCITY_INTERPOLATION_ANALYSIS.md](RK4_VELOCITY_INTERPOLATION_ANALYSIS.md)

**Confirmed:** RK4 uses `velocity_field[nodes_idx]` where `nodes_idx` comes from deduplicated connectivity. Before the fix, this accessed wrong velocities causing wrong trajectories.

**After Fix:** Velocity arrays now correctly aligned with deduplicated node IDs.

### Priority 3: Test Fix and Verify Correct Trajectories ⏳ PENDING

**Next Step:** Run production script to verify:
1. Arrays pass consistency checks
2. Trajectories follow physically correct streamlines
3. No particle loss
4. Performance remains good (~20-25K particles/s)

### Priority 4: Implement Binary Search (Optional - Future)

To improve performance (reduce radius from 500 to ~10), but NOT required for accuracy.
This can be done after verifying the trajectory fix works correctly.

---

## Detailed Analysis: Velocity Loading and Interpolation

### Step 1: How velocities are loaded

**File:** `jaxtrace/gpu/mesh_loader_timedep.py`

Expected behavior:
```python
def load_velocity_sequence_from_pvtu(base_path, file_pattern, timestep_range, field_name):
    # Loads PVTU files for each timestep
    # Returns:
    #   node_positions: (n_nodes_with_duplicates, 3)
    #   connectivity: (n_elements, 4)
    #   velocity_sequence: (n_timesteps, n_nodes_with_duplicates, 3)
```

### Step 2: Deduplication in production script

**File:** `production_tracking_fully_fused_timedep.py`

```python
# Line 300-306: Load mesh
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    base_path=MESH_BASE_PATH,
    file_pattern=MESH_FILE_PATTERN,
    timestep_range=VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=True
)

# Line 327-329: Deduplicate
node_positions, connectivity, n_duplicates = deduplicate_nodes(
    node_positions, connectivity, verbose=True
)

# ⚠️ Problem: velocity_sequence NOT passed to deduplicate_nodes!
```

### Step 3: How velocity is used in RK4

**File:** `jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py`

Expected RK4 velocity interpolation:
```python
def interpolate_velocity_at_position(pos, elem_id, velocity_field):
    # 1. Get element nodes
    nodes = connectivity[elem_id]  # (4,) node IDs

    # 2. Get node positions
    p0 = node_positions[nodes[0]]
    p1 = node_positions[nodes[1]]
    p2 = node_positions[nodes[2]]
    p3 = node_positions[nodes[3]]

    # 3. Get velocities at nodes
    v0 = velocity_field[nodes[0]]  # ← POTENTIAL BUG HERE!
    v1 = velocity_field[nodes[1]]
    v2 = velocity_field[nodes[2]]
    v3 = velocity_field[nodes[3]]

    # 4. Compute barycentric coordinates
    b0, b1, b2, b3 = compute_barycentric(pos, p0, p1, p2, p3)

    # 5. Interpolate
    v_interp = b0*v0 + b1*v1 + b2*v2 + b3*v3
    return v_interp
```

**If velocity_field has wrong shape:**
```python
# nodes[0] = 123456 (valid deduplicated node ID, in range [0, 571,173])
# velocity_field.shape = (780922, 3)  # WRONG - not deduplicated!

# v0 = velocity_field[123456]  # Accesses WRONG node's velocity!
# The velocity at index 123456 in the original array belongs to a DIFFERENT node
# that may now be merged or re-indexed
```

---

## Diagnostic to Add to Production Script

Add after deduplication (around line 339):

```python
# ========================================================================
# CRITICAL DIAGNOSTIC: Verify array consistency after deduplication
# ========================================================================
print(f"\n[DIAGNOSTIC] Checking array shapes after deduplication...")
print(f"  node_positions shape:   {node_positions.shape}")
print(f"  connectivity shape:     {connectivity.shape}")
print(f"  velocity_sequence shape: {velocity_sequence.shape}")
print(f"  Nodes removed:          {n_duplicates:,}")

# Check if velocity_sequence matches deduplicated node count
n_nodes_current = node_positions.shape[0]
n_nodes_velocity = velocity_sequence.shape[1]

if n_nodes_velocity != n_nodes_current:
    print(f"\n  ⚠️  CRITICAL BUG DETECTED:")
    print(f"      Velocity array has {n_nodes_velocity:,} nodes")
    print(f"      But mesh has {n_nodes_current:,} nodes after deduplication")
    print(f"      Difference: {n_nodes_velocity - n_nodes_current:,} nodes")
    print(f"\n  IMPACT:")
    print(f"      - Connectivity references node IDs in [0, {n_nodes_current-1}]")
    print(f"      - Velocity array has {n_nodes_velocity} entries")
    print(f"      - Indexing velocity_sequence[t, node_id] will access WRONG velocities!")
    print(f"      - This causes WRONG trajectories (even with correct element assignment)")
    print(f"\n  REQUIRED FIX:")
    print(f"      Velocity array must be remapped to deduplicated node indices")
    print(f"      OR deduplication must update velocity_sequence")
    raise RuntimeError("Velocity array shape mismatch - cannot continue safely")
else:
    print(f"  ✅ Array shapes consistent (velocity has correct node count)")

# Additional check: Verify connectivity references valid node IDs
max_node_id = np.max(connectivity)
if max_node_id >= n_nodes_current:
    print(f"\n  ⚠️  CONNECTIVITY BUG:")
    print(f"      Max node ID in connectivity: {max_node_id}")
    print(f"      But only {n_nodes_current} nodes exist!")
    raise RuntimeError("Connectivity references non-existent nodes")
else:
    print(f"  ✅ Connectivity valid (max node ID {max_node_id} < {n_nodes_current})")
```

---

## Expected Diagnostic Output

### If Bug Exists:
```
[DIAGNOSTIC] Checking array shapes after deduplication...
  node_positions shape:   (571173, 3)
  connectivity shape:     (3048900, 4)
  velocity_sequence shape: (40, 780922, 3)
  Nodes removed:          209,749

  ⚠️  CRITICAL BUG DETECTED:
      Velocity array has 780,922 nodes
      But mesh has 571,173 nodes after deduplication
      Difference: 209,749 nodes

  IMPACT:
      - Connectivity references node IDs in [0, 571172]
      - Velocity array has 780922 entries
      - Indexing velocity_sequence[t, node_id] will access WRONG velocities!
      - This causes WRONG trajectories (even with correct element assignment)

  REQUIRED FIX:
      Velocity array must be remapped to deduplicated node indices
      OR deduplication must update velocity_sequence
```

### If No Bug:
```
[DIAGNOSTIC] Checking array shapes after deduplication...
  node_positions shape:   (571173, 3)
  connectivity shape:     (3048900, 4)
  velocity_sequence shape: (40, 571173, 3)
  Nodes removed:          209,749

  ✅ Array shapes consistent (velocity has correct node count)
  ✅ Connectivity valid (max node ID 571172 < 571173)
```

---

## Fix Strategy (If Bug Confirmed)

### Option A: Update deduplicate_nodes() to remap velocities

**File:** `jaxtrace/gpu/mesh_deduplication.py`

Modify to accept and remap velocity arrays:
```python
def deduplicate_nodes(node_positions, connectivity, velocity_sequence=None, verbose=True):
    # ... existing deduplication logic ...
    # Build node_mapping: old_id → new_id

    if velocity_sequence is not None:
        # Remap velocity sequence
        n_timesteps = velocity_sequence.shape[0]
        n_new_nodes = len(unique_nodes)
        velocity_dedup = np.zeros((n_timesteps, n_new_nodes, 3), dtype=velocity_sequence.dtype)

        for old_id, new_id in node_mapping.items():
            velocity_dedup[:, new_id, :] = velocity_sequence[:, old_id, :]

        return node_positions_dedup, connectivity_dedup, n_duplicates, velocity_dedup
    else:
        return node_positions_dedup, connectivity_dedup, n_duplicates
```

### Option B: Remap in production script

**File:** `production_tracking_fully_fused_timedep.py`

After deduplication:
```python
# Get node mapping from deduplication
node_positions, connectivity, n_duplicates, node_mapping = deduplicate_nodes(
    node_positions, connectivity, verbose=True, return_mapping=True
)

# Remap velocity sequence if duplicates were removed
if n_duplicates > 0:
    print(f"  Remapping velocity fields to deduplicated nodes...")
    n_timesteps = velocity_sequence.shape[0]
    n_new_nodes = node_positions.shape[0]
    velocity_sequence_dedup = np.zeros((n_timesteps, n_new_nodes, 3), dtype=np.float32)

    for old_id, new_id in node_mapping.items():
        velocity_sequence_dedup[:, new_id, :] = velocity_sequence[:, old_id, :]

    velocity_sequence = velocity_sequence_dedup
    print(f"  Remapped velocity: {velocity_sequence.shape}")
```

---

## Summary

**Root Cause:** Velocity array not remapped after node deduplication ✅ CONFIRMED

**Evidence:** Diagnostic confirmed velocity_sequence has 780,922 nodes while mesh has 571,173 nodes

**Impact:** Wrong velocity interpolation → Wrong RK4 integration → Wrong trajectories

**Fix Applied:** Modified `deduplicate_nodes()` to automatically remap velocity arrays

**Status:** ✅ FIXED - Ready for testing

---

## Fix Implementation Details

### Modified Files

1. **[jaxtrace/gpu/mesh_deduplication.py](jaxtrace/gpu/mesh_deduplication.py)**
   - Added `velocity_sequence` parameter (optional)
   - Remap velocity arrays using existing `node_map`
   - Return remapped arrays with correct shape
   - Memory reduction: 780,922 → 571,173 nodes per timestep (27% less memory)

2. **[production_tracking_fully_fused_timedep.py](production_tracking_fully_fused_timedep.py):327**
   - Pass `velocity_sequence` to `deduplicate_nodes()`
   - Receive remapped `velocity_sequence` back
   - Diagnostic now verifies array consistency

### Fix Algorithm

```python
# In deduplicate_nodes():
for old_id in range(n_nodes):
    new_id = node_map[old_id]
    # Copy velocity for all timesteps at once
    remapped_velocity_sequence[:, new_id, :] = velocity_sequence[:, old_id, :]
```

**Key insight:** Use the same `node_map` that remaps connectivity to remap velocity arrays.

### Expected Results After Fix

**Before:**
- Velocity shape: (40, 780922, 3) - WRONG
- Trajectories: Wrong physics
- Memory usage: 376 MB for velocity arrays

**After:**
- Velocity shape: (40, 571173, 3) - ✅ CORRECT
- Trajectories: Correct physical streamlines
- Memory usage: 274 MB for velocity arrays (27% reduction)

### Validation

Run production script - should see:
```
[DIAGNOSTIC] Verifying array consistency after deduplication...
  node_positions shape:   (571173, 3)
  connectivity shape:     (3048900, 4)
  velocity_sequence shape: (40, 571173, 3)  ← NOW CORRECT!
  ✅ Velocity array correctly remapped (571,173 nodes)
  ✅ Connectivity valid (max node ID 571172 < 571173)
  ✅ All array consistency checks passed!
      → Trajectories should now be physically correct
```

---

## Next Action

**Run production script to verify the fix:**
```bash
source .venv/bin/activate
python production_tracking_fully_fused_timedep.py > logs/production_velocity_fixed.log 2>&1
```

**Expected outcome:**
- Arrays pass consistency checks ✅
- Particles track correctly along streamlines ✅
- No wrong trajectories ✅
- Performance remains good (~20-25K particles/s) ✅
