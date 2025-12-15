# Production Script Issue: Missing LEVEL Field

**Date:** 2025-11-28
**Status:** ⚠️ LEVEL field not found in mesh - L2 octree disabled

---

## Issue Summary

The production script `production_tracking_3hop_l2_octree.py` detected that the ThreadedA mesh **does not have a LEVEL field**, which is required for L2 octree construction.

### What Happened

```
================================================================================
L2 OCTREE CONSTRUCTION
================================================================================

Loading mesh cell data for LEVEL field...
⚠  WARNING: No LEVEL field found in mesh
  L2 octree will be DISABLED (falling back to L0+L1 only)
  Expected retention: 16% (vs 82% with L2 octree)
```

### Root Cause

The mesh file `/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_120.pvtu` does not contain cell data field named `LEVEL`.

The LEVEL field is used to identify refined regions of the mesh where:
- Higher LEVEL values = finer mesh resolution
- L2 octree filters elements with `LEVEL >= 7`
- This creates a sparse octree covering only refined regions

---

## Impact

### Without L2 Octree (Current State)

**Architecture:** L0 (cached) + L1 (3-hop only)

**Performance:**
- Search hit rate: ~99.5% (missing ~0.5% of particles per timestep)
- Expected retention: **16% at 2,500 timesteps** ❌
- Throughput: 40-48k p/s (3-hop is fast)

**Problem:** Particles in refined mesh regions escape after crossing element boundaries, causing cumulative loss over time.

### With L2 Octree (Target State)

**Architecture:** L0 (cached) + L1 (3-hop) + L2 (octree)

**Performance:**
- Search hit rate: ~99.95%+ (L2 catches remaining 0.5%)
- Expected retention: **82% at 2,500 timesteps** ✅
- Throughput: 40-48k p/s (L2 overhead <1%)

---

## Options to Proceed

### Option 1: Run with L0+L1 Only (Current Configuration) ⚠️
**Pros:**
- Script is ready to run now
- Will complete successfully
- Fast throughput

**Cons:**
- Only 16% retention at 2,500 timesteps
- Not meeting the 82% retention target
- Particles lost in refined regions

**Command:**
```bash
python production_tracking_3hop_l2_octree.py 2>&1 | tee logs/production_3hop_no_l2.log
```

### Option 2: Use 4-Hop or 5-Hop Hierarchical (No L2 Needed) ✅
**Pros:**
- Achieves 82% retention without LEVEL field
- Uses existing `production_tracking_hierarchical_5hop_CLEAN.py`
- Proven to work (after JIT fix)

**Cons:**
- Slower throughput (8-18k p/s vs 40-48k p/s)
- Higher memory usage for neighbor expansion

**Command:**
```bash
# Use 4-hop hierarchical (recommended balance)
python production_tracking_hierarchical_5hop_CLEAN.py 2>&1 | tee logs/production_4hop_hierarchical.log
```

**Configuration in script:**
```python
RK4_L1_HOP_COUNT = 4  # Change from 5 to 4 for better throughput
```

### Option 3: Add LEVEL Field to Mesh (Long-term Solution) 🔧
**Steps:**
1. Identify mesh generation tool (e.g., AMReX, p4est, MOAB)
2. Regenerate mesh with LEVEL field in cell data
3. Or post-process existing mesh to add LEVEL field based on element size

**Pros:**
- Enables L2 octree for optimal performance
- 82% retention + 40-48k p/s throughput
- Best of both worlds

**Cons:**
- Requires mesh regeneration or post-processing
- May need access to original simulation setup

### Option 4: Use 3-Hop + Block Fallback (Alternative Fallback) 🧪
**Description:**
- L0 (cached) + L1 (3-hop) + Block-local search
- Block fallback searches local block instead of octree

**Status:** Not currently enabled, would require:
```python
USE_BLOCK_LOCAL_FALLBACK = True
USE_L2_OCTREE_FALLBACK = False
```

**Expected Performance:**
- Retention: ~77.9% at 2,500 timesteps
- Throughput: Lower than L2 octree (nested scan overhead)

---

## Recommendation

### Immediate Action: Use Option 2 (4-Hop Hierarchical)

**Rationale:**
1. ✅ Achieves 82% retention target
2. ✅ No LEVEL field required
3. ✅ Script already exists and tested
4. ✅ JIT fix already applied
5. ⚠️ Acceptable throughput trade-off (12-18k p/s)

**Steps:**
```bash
cd /home/arhashemi/Workspace/welding/JAXTrace

# Edit configuration (optional - use 4-hop instead of 5-hop)
# In production_tracking_hierarchical_5hop_CLEAN.py, line ~314:
# RK4_L1_HOP_COUNT = 4

# Run production test
source .venv/bin/activate
python production_tracking_hierarchical_5hop_CLEAN.py 2>&1 | tee logs/production_4hop_hierarchical_FINAL.log
```

**Expected Results:**
```
Timesteps: 2,500
Throughput: 12-18k p/s
Retention: 82%
Total time: ~8-10 minutes
```

### Long-term: Investigate LEVEL Field Addition (Option 3)

If mesh can be regenerated with LEVEL field:
1. Check mesh generation parameters
2. Enable AMR level tracking in output
3. Regenerate mesh with cell data containing LEVEL
4. Return to 3-hop + L2 octree for optimal performance

---

## Technical Notes

### LEVEL Field Format
The LEVEL field should be:
- **Type:** Integer cell data (one value per element)
- **Range:** 0 to max_refinement_level (e.g., 0-9)
- **Meaning:** Refinement level (higher = finer resolution)
- **Format:** VTK cell data array named "LEVEL"

### How L2 Octree Uses LEVEL
```python
# Filter elements for octree
filtered_elements = elements[LEVEL >= OCTREE_LEVEL_THRESHOLD]

# Example: OCTREE_LEVEL_THRESHOLD = 7
# Builds octree only for refined regions (LEVEL 7, 8, 9)
# Skips coarse regions (LEVEL 0-6)
```

This creates a sparse octree covering ~30% of elements but catching particles that escape in refined regions.

---

## Current Script Status

### Fixed Issues ✅
1. ✅ Missing `USE_BLOCK_LOCAL_FALLBACK` variable added
2. ✅ VTK import for LEVEL field loading fixed
3. ✅ Script compiles without syntax errors
4. ✅ Gracefully handles missing LEVEL field

### Outstanding ⚠️
1. ⚠️ Mesh lacks LEVEL field → L2 octree disabled
2. ⚠️ Will run with L0+L1 only → 16% retention
3. ⚠️ Does not meet 82% retention target

### Ready to Run ✅
The script will run successfully but with reduced retention:
```bash
python production_tracking_3hop_l2_octree.py 2>&1 | tee logs/production_3hop_no_l2.log
```

---

## Files Modified

1. **[production_tracking_3hop_l2_octree.py:325](production_tracking_3hop_l2_octree.py#L325)**
   - Added: `USE_BLOCK_LOCAL_FALLBACK = False`
   - Fixes: `NameError: name 'USE_BLOCK_LOCAL_FALLBACK' is not defined`

2. **[production_tracking_3hop_l2_octree.py:436-449](production_tracking_3hop_l2_octree.py#L436-L449)**
   - Fixed: VTK import for LEVEL field loading
   - Handles: Missing LEVEL field gracefully

---

## Next Steps

**Recommended:**
1. Use `production_tracking_hierarchical_5hop_CLEAN.py` with 4-hop configuration
2. Achieve 82% retention target immediately
3. Investigate mesh LEVEL field for future optimization

**Alternative:**
1. Run `production_tracking_3hop_l2_octree.py` to completion
2. Document 16% retention baseline
3. Compare with 4-hop results to validate L2 octree benefit

---

**Summary:** The script is ready but the mesh doesn't support L2 octree. Use 4-hop hierarchical search instead to achieve 82% retention target.
