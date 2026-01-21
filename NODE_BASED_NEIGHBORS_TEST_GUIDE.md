# Node-Based Neighbors Testing Guide

**Date**: 2025-12-19
**Status**: Ready for testing
**File Modified**: `production_tracking_fully_fused_timedep.py`

---

## Changes Made

### 1. Switched to Node-Based Neighbors

**File**: `production_tracking_fully_fused_timedep.py`

**Line 297**:
```python
# BEFORE (Face-based):
element_neighbors = build_element_neighbors_array(connectivity)

# AFTER (Node-based):
element_neighbors = build_element_neighbors_array(connectivity, method='node', verbose=True)
```

**Lines 300-303**: Added diagnostic output:
```python
neighbor_memory_mb = element_neighbors.nbytes / (1024**2)
print(f"    Neighbor memory: {neighbor_memory_mb:.1f} MB")
print(f"    Neighbor array shape: {element_neighbors.shape}")
print(f"    Max neighbors per element: {element_neighbors.shape[1]}")
```

### 2. Added Configuration Note

**Lines 79-81**: Documented the change:
```python
# NOTE: Using NODE-BASED neighbors (line ~297) to support 1:2 octree refinement
#       Face-based neighbors don't cross refinement levels (coarse/fine share edges, not faces)
#       Node-based: ~1.1GB memory, 20-100 neighbors/element (vs face: 48MB, 4 neighbors/element)
```

---

## Quick Test (Recommended First)

Test with reduced particle count to verify correctness quickly:

### Step 1: Modify Particle Count

**Edit** `production_tracking_fully_fused_timedep.py` **lines 62-63**:

```python
# Original (225,000 particles):
PARTICLE_GRID_RESOLUTION = (50, 90, 50)

# Quick test (12,000 particles):
PARTICLE_GRID_RESOLUTION = (20, 30, 20)
```

**Edit line 76**:

```python
# Original (2,500 steps):
N_STEPS = 2_500

# Quick test (500 steps):
N_STEPS = 500
```

### Step 2: Run Quick Test

```bash
cd /home/arhashemi/Workspace/welding/JAXTrace
source .venv/bin/activate

python production_tracking_fully_fused_timedep.py 2>&1 | tee logs/production_node_based_quick_test.log
```

**Expected time**: ~5-10 minutes (vs ~2 hours for full run)

### Step 3: Check Output

**Look for in log**:

```
[3/6] Uploading mesh and Morton structure to GPU...
  Computing element neighbors (NODE-BASED for 1:2 refinement support)...

Extracting NODE-BASED element neighbors for 3,512,384 elements...
  (Elements sharing ANY node are considered neighbors)
Building node-to-element map...
  ...
  Total unique nodes: 900,671
Building neighbor lists from node connectivity...
  ...
    Neighbor computation: XX.XXs
    Neighbor memory: ~1100 MB  ← Should be around 1.1 GB
    Neighbor array shape: (3512384, YY)  ← YY = max neighbors per element (20-100)
    Max neighbors per element: YY
```

**Success indicators during tracking**:

```
Step 100/500: 12000/12000 (100.0%), retention=100.0%, throughput=XXXX particles/step
  ↑ Should maintain 100% retention
  ↑ Throughput: expect 20-50K particles/s (slower than L2-only but CORRECT)
```

---

## Full Production Test

If quick test shows correct results, run full production:

### Step 1: Restore Full Configuration

**Edit** `production_tracking_fully_fused_timedep.py`:

```python
# Restore original values:
PARTICLE_GRID_RESOLUTION = (50, 90, 50)  # 225,000 particles
N_STEPS = 2_500
```

### Step 2: Run Full Test

```bash
python production_tracking_fully_fused_timedep.py 2>&1 | tee logs/production_node_based_full_test.log
```

**Expected time**: ~1-2 hours (depends on L1 hit rate)

---

## Verification: Check Trajectories

### Method 1: Visual Inspection (ParaView)

1. **Load VTK files**:
   ```bash
   # In ParaView, open:
   output/global_morton_timedep/particles_step_*.vtu
   ```

2. **Check refined region motion**:
   - Apply "Glyph" filter with velocity arrows
   - Focus on refined region: X=30, Y=15, Z=0.3
   - **Expected**: Particles show **ROTATING** motion (circular/helical paths)
   - **Bug symptom**: Particles show **LINEAR** motion (straight lines)

3. **Animate**:
   - Play animation through timesteps
   - Particles near tool (refined region) should rotate
   - Particles far from tool should move slower/straighter

### Method 2: Log Analysis

**Check element assignment rates**:

```bash
# Extract retention stats
grep "retention=" logs/production_node_based_full_test.log | tail -20

# Expected: retention stays >95% throughout
# Bug symptom: retention drops significantly (<80%)
```

**Check particle statistics** (if added to script):

```python
# If you add diagnostic output like:
# "Fine elements: XX%, Medium: XX%, Coarse: XX%"

grep "Fine elements" logs/production_node_based_full_test.log

# Expected in refined region:
#   Fine: 60-85%
#   Medium: 10-30%
#   Coarse: 0-10%

# Bug symptom:
#   Fine: 0-5%
#   Medium: 10-20%
#   Coarse: 75-90%
```

### Method 3: Trajectory Analysis Script

Create diagnostic script to quantify rotation:

```python
# analyze_trajectories.py
import numpy as np
import pyvista as pv

# Load particle VTK files
particles_t0 = pv.read("output/global_morton_timedep/particles_step_0000.vtu")
particles_t1 = pv.read("output/global_morton_timedep/particles_step_0100.vtu")
particles_t2 = pv.read("output/global_morton_timedep/particles_step_0200.vtu")

# Get positions
pos_t0 = particles_t0.points
pos_t1 = particles_t1.points
pos_t2 = particles_t2.points

# Select particles in refined region
tool_center = np.array([30.0, 15.0, 0.3])
dist_t0 = np.linalg.norm(pos_t0 - tool_center, axis=1)
in_refined = dist_t0 < 2.0  # Within 2mm of tool

# Compute angular displacement
def compute_angular_change(p0, p1, p2, center):
    """Compute angle swept by particle trajectory."""
    v0 = p0 - center
    v1 = p1 - center
    v2 = p2 - center

    # Project to XY plane (rotation plane)
    v0_xy = v0[:2]
    v1_xy = v1[:2]
    v2_xy = v2[:2]

    # Angle from t0 to t1 to t2
    angle_01 = np.arctan2(v1_xy[1], v1_xy[0]) - np.arctan2(v0_xy[1], v0_xy[0])
    angle_12 = np.arctan2(v2_xy[1], v2_xy[0]) - np.arctan2(v1_xy[1], v1_xy[0])

    return angle_01, angle_12

# Compute for particles in refined region
angles_01 = []
angles_12 = []
for i in np.where(in_refined)[0]:
    a01, a12 = compute_angular_change(pos_t0[i], pos_t1[i], pos_t2[i], tool_center)
    angles_01.append(a01)
    angles_12.append(a12)

angles_01 = np.array(angles_01)
angles_12 = np.array(angles_12)

# Check for rotation
mean_rotation = np.mean(np.abs(angles_01)) + np.mean(np.abs(angles_12))
print(f"Mean angular displacement: {np.rad2deg(mean_rotation):.1f} degrees")

if mean_rotation > np.deg2rad(30):  # >30 degrees total
    print("✓ ROTATING motion detected (correct)")
else:
    print("✗ LINEAR motion detected (bug)")
```

---

## Expected Results

### Success Criteria

1. ✅ **Neighbor construction completes** (~30-60 seconds)
2. ✅ **Neighbor memory: ~1.1 GB** (vs 48 MB for face-based)
3. ✅ **Max neighbors: 50-100** per element (vs 4 for face-based)
4. ✅ **Retention: >95%** throughout tracking
5. ✅ **Trajectories: ROTATING** in refined region (visual check)
6. ✅ **Performance: 20-50K particles/s** (slower but correct)

### If ALL criteria met → SUCCESS!

Node-based neighbors fix the refinement issue. You can then:
- **Option A**: Use node-based for production (correct, acceptable performance)
- **Option B**: Optimize with hybrid neighbors (Phase 1B)
- **Option C**: Proceed to Phase 2 (octree-aligned leaves, best performance)

### If criteria NOT met → Need investigation

Possible issues:
1. **Memory error**: Too much GPU memory for node-based
   - **Solution**: Implement hybrid neighbors OR use smaller mesh region
2. **Still linear trajectories**: Different issue (not neighbor connectivity)
   - **Solution**: Debug velocity interpolation or RK4 stepping
3. **Low retention**: Particles lost/invalid
   - **Solution**: Check L2 search radius, Morton structure

---

## Performance Comparison

### Expected Performance Differences

**Face-Based (WRONG)**:
- Neighbor memory: 48 MB
- L1 hit rate: 0% (can't find fine neighbors)
- L2 fallback: 100%
- Throughput: ~30K particles/s
- **Trajectories: LINEAR** ❌

**Node-Based (CORRECT)**:
- Neighbor memory: ~1.1 GB
- L1 hit rate: 60-80% (finds fine neighbors)
- L2 fallback: 20-40%
- Throughput: 20-50K particles/s
- **Trajectories: ROTATING** ✓

**Trade-off**: ~30-50% slower but CORRECT results

---

## Optimization Path (If Node-Based Works)

### Short-term: Tune L1 Parameters

Since node-based has more neighbors, you may benefit from:

**Reduce hops** (neighbors found faster):
```python
N_HOPS = 2  # Changed from 3
```

**Increase L2 radius** (catch edge cases):
```python
L2_SEARCH_RADIUS = 20  # Changed from 10
```

### Medium-term: Hybrid Neighbors (Phase 1B)

Implement selective node-based neighbors:
- Face-based for interior (4 neighbors)
- Node-based for boundary (20-100 neighbors)
- **Memory**: ~110 MB (vs 1.1 GB)
- **Performance**: 30-60K particles/s

### Long-term: Octree-Aligned Leaves (Phase 2)

Replace fixed-capacity leaves with octree cells:
- Better spatial coherence
- O(1) prefix lookup
- **Performance**: 100-150K particles/s
- **Time**: ~1 week implementation

---

## Troubleshooting

### Issue 1: Out of GPU Memory

**Error**: `RuntimeError: CUDA out of memory`

**Cause**: Node-based neighbors + velocity fields + mesh > available GPU RAM

**Solutions**:
1. **Reduce particle count**: Use smaller grid (e.g., 30×50×30 = 45K particles)
2. **Use CPU for neighbors**: Keep on CPU, transfer on-demand (slower but works)
3. **Implement hybrid neighbors**: Only boundary elements need node-based
4. **Use smaller mesh region**: Load subset of domain

### Issue 2: Neighbor Construction Hangs

**Symptom**: Script stuck at "Building neighbor lists from node connectivity..."

**Cause**: 3.5M elements × 20-100 neighbors takes time

**Expected**: 30-60 seconds (be patient!)

**If >5 minutes**: May be memory thrashing, reduce particle count

### Issue 3: Still Linear Trajectories

**Symptom**: Node-based neighbors loaded, but particles still move linearly

**Possible causes**:
1. **L1 still disabled**: Check `ENABLE_L1_SEARCH = True`
2. **Velocity issue**: Wrong velocity field or scaling
3. **RK4 issue**: Bug in integration (unlikely if L2-only worked)

**Debug steps**:
1. Check log: "L1 neighbor search" should show in search stats
2. Add diagnostic: Print element IDs during tracking
3. Verify particles assigned to fine elements

### Issue 4: Low Performance (<10K particles/s)

**Symptom**: Correct trajectories but very slow

**Possible causes**:
1. **Too many neighbors**: Max neighbors >100 (check log)
2. **L1 not helping**: Hit rate <50% (L2 still dominant)
3. **L2 radius too large**: Searching too many leaves

**Solutions**:
1. **Reduce N_HOPS**: Try `N_HOPS = 2`
2. **Implement hybrid neighbors**: Only boundary elements need many neighbors
3. **Proceed to Phase 2**: Octree-aligned leaves will be faster

---

## Next Steps After Testing

### If Test Succeeds (Rotating Trajectories)

1. **Document results**:
   - Log file: `logs/production_node_based_full_test.log`
   - Screenshots: VTK visualization showing rotation
   - Performance metrics: Throughput, retention, L1 hit rate

2. **Choose optimization path**:
   - **Path A**: Use node-based (good enough)
   - **Path B**: Implement hybrid neighbors (better memory/performance)
   - **Path C**: Phase 2 - octree leaves (best long-term)

3. **Update documentation**:
   - Mark L1 issue as RESOLVED (needed node-based neighbors)
   - Document performance characteristics
   - Plan for Phase 2 (if pursuing)

### If Test Fails (Still Linear)

1. **Run diagnostic script**: `diagnose_neighbor_connectivity_refinement.py`
   - Confirms face vs node connectivity
   - Shows coarse→fine connection counts

2. **Add trajectory analysis**: Quantify rotation vs linear motion

3. **Debug search hierarchy**:
   - Add logging to L1 search
   - Track element ID changes
   - Verify containment checks

---

## Summary

**Changes Made**:
- ✅ Switched to node-based neighbors: `method='node'`
- ✅ Added diagnostic output: memory, shape, max neighbors
- ✅ Added configuration note: documented trade-offs

**Ready to Test**:
```bash
# Quick test (12K particles, 500 steps):
python production_tracking_fully_fused_timedep.py

# Check: Rotating trajectories in refined region
# Check: Neighbor memory ~1.1 GB
# Check: Retention >95%
```

**Success Criteria**:
- ✅ Rotating trajectories (visual check)
- ✅ High retention (>95%)
- ✅ Acceptable performance (20-50K particles/s)

**If successful**: Problem solved! Node-based neighbors support 1:2 refinement.

---

**Ready for your testing!**
