# Refinement Boundary Diagnostic - Fixed Version

**Date**: 2026-01-03
**Status**: Ready to run
**Purpose**: Identify WHY particles fail at refinement boundaries

---

## What Was Fixed

### Original Error
```
ValueError: zero-size array to reduction operation minimum which has no identity
```

**Cause**: Found 0 boundary elements → empty array → numpy min/max failed

### Root Cause of 0 Boundary Elements

**Two possibilities**:

1. **Face-based neighbors don't capture size transitions** (MOST LIKELY)
   - Small elements only have small face-neighbors
   - Large elements only have large face-neighbors
   - **No direct face-connection** between small and large elements at boundaries!
   - This would CONFIRM our hypothesis

2. **All elements are similar size** (UNLIKELY)
   - Mesh has no refinement
   - Contradicts volume range [8.12e-14, 2.13e-08] = 262K× variation

---

## What the Fixed Diagnostic Does

### Speed Improvements
- **Samples 10,000 elements** instead of checking all 3M elements
- Progress indicators every 2,000 elements
- Estimated total from sample

### Better Error Handling
- Checks if boundary list is empty
- Reports diagnostic information before crashing
- Explains what empty results mean

### New Debug Output

#### 1.1 Element Volume Distribution
```
Volume range: [8.12e-14, 2.13e-08]
Volume median: 6.50e-13
Unique volumes: ~500,000
```

**What to look for**:
- Wide range → confirms refinement exists
- Many unique volumes → graded mesh refinement

#### 1.2 Boundary Element Search
```
Sampling 10,000 elements for speed...
  Checked 0/10000 elements, found 0 boundaries...
  Checked 2000/10000 elements, found 0 boundaries...
  ...
Found 0 boundary elements in 10,000 sampled
```

**What this means if 0 found**:
- ❌ Face-based neighbors DON'T capture refinement transitions
- ✅ CONFIRMS hypothesis: small and large elements don't share faces
- ✅ Explains particle loss: L1 search can't cross boundaries

#### 1.3 Small→Large Transition Search
```
Sampling 10,000 elements for speed...
  Checked 0/10000 elements, found 0 transitions...
Found 0 small→large transitions in 10,000 sampled
```

**What this means if 0 found**:
- Face-neighbor graph is DISCONNECTED at size boundaries
- Small elements isolated from large elements
- **Particles can NEVER cross** using face-neighbor hops!

---

## Expected Results

### Scenario 1: Face Neighbors Don't Capture Transitions (90% probability)

**Output**:
```
[1.1] Checking element volume distribution...
  Volume range: [8.12e-14, 2.13e-08]
  Volume median: 6.50e-13
  Unique volumes: 482,345

[1.2] Identifying refinement boundary elements...
  Sampling 10,000 elements for speed...
    Checked 0/10000 elements, found 0 boundaries...
    Checked 2000/10000 elements, found 0 boundaries...
    ...
  Found 0 boundary elements in 10,000 sampled
  ⚠️  No refinement boundary elements found!
  This suggests either:
    1. All elements are similar size (no refinement)
    2. Face-based neighbors don't capture size transitions
```

**Conclusion**: **Face-based neighbor construction is BROKEN** for refined meshes

**Solution**: Switch to node-based neighbors OR increase L1 hops to 100+

---

### Scenario 2: Face Neighbors Partially Capture Transitions (8% probability)

**Output**:
```
Found 1,234 boundary elements in 10,000 sampled
Estimated total: 376,554 (12.35%)

[1.3] Neighbor count distribution at boundaries...
  Mean neighbors: 2.8
  Min neighbors: 1
  Max neighbors: 4
  Elements with <4 neighbors: 987 (80.06%)
```

**Analysis**:
- Some boundaries found, but **incomplete** (mean 2.8 vs expected 4)
- 80% have <4 neighbors → **missing connections**

**Conclusion**: Face neighbors capture SOME transitions but miss most

**Solution**: Node-based neighbors OR 30-hop L1

---

### Scenario 3: Face Neighbors Work Fine (2% probability)

**Output**:
```
Found 4,567 boundary elements in 10,000 sampled
Estimated total: 1,392,234 (45.67%)

[1.3] Neighbor count distribution at boundaries...
  Mean neighbors: 3.9
  Min neighbors: 3
  Max neighbors: 4

[2.2] Testing L1 3-hop search across transitions...
  L1 failures: 87/100 tested transitions
  ❌ PROBLEM: L1 cannot reach large neighbors in 3 hops!
```

**Analysis**:
- Boundaries found correctly
- Neighbors complete
- **But L1 3-hops insufficient**

**Conclusion**: Neighbor construction OK, but hop count too low

**Solution**: Increase L1 hops to 30-50

---

## What to Do Based on Results

### If Scenario 1 (0 boundaries found):

**Root cause**: Face-neighbor graph doesn't connect small→large elements

**Fix options** (ranked):
1. **Node-based neighbors** (8 hours, 95% fix probability)
   - Change neighbor definition: share ≥1 node (not full face)
   - Captures all geometric adjacencies
   - 12-20 neighbors per element (vs 4)

2. **Huge L1 hop count** (1 hour, 50% fix probability)
   - Increase to 100 hops (brute force)
   - May still fail if graph is disconnected
   - Very slow (100× more point-in-tet tests)

3. **Spatial search** (16 hours, 99% fix probability)
   - Replace L1 topological search with spatial radius search
   - Guaranteed to work
   - Complex implementation (BVH or grid)

---

### If Scenario 2 (partial boundaries found):

**Fix options**:
1. **30-hop L1** (15 minutes, 80% fix probability)
2. **Node-based neighbors** (8 hours, 95% fix probability)

---

### If Scenario 3 (boundaries found, L1 fails):

**Fix options**:
1. **30-hop L1** (15 minutes, 90% fix probability)
2. **50-hop L1 if 30 insufficient** (5 minutes more)

---

## Run the Diagnostic

```bash
python diagnose_refinement_boundary_crossing.py > logs/diagnose_boundary.log 2>&1
```

**Expected runtime**: 2-3 minutes (was hanging before due to 3M element loop)

---

## Next Steps After Diagnostic

1. **Share the log file** - I'll analyze which scenario occurred
2. **Implement recommended fix** - Based on which scenario
3. **Test and iterate** - May need to try multiple approaches

---

**Status**: Ready to run. No more crashes expected.
