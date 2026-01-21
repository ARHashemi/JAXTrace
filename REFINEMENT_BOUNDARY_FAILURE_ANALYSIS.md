# Refinement Boundary Crossing Failure - Root Cause Analysis

**Date**: 2026-01-03
**Status**: 🔴 CRITICAL - Enhanced Morton FAILED, investigating root cause
**Problem**: Particles crossing refined→coarse boundaries are lost

---

## Test Results: Enhanced Morton Search

### Performance Catastrophe
- **Retention @ step 100**: 85.67% (vs 82.45% baseline) - Only +3.2% improvement
- **Throughput**: 1,246 p/s (vs 6,500 p/s baseline) - **5× slower!**
- **Final retention**: 31.27% @ step 2500 (massive particle loss)

### Why Enhanced Morton Failed
1. **Wrong diagnosis**: Assumed Morton locality was the issue
2. **Actual problem**: Particles crossing refinement boundaries fail ALL search levels
3. **Performance cost**: 5×5×5 search (125 octants) is too expensive for minimal gain

---

## Visual Evidence

From [Screenshot_20260103_140627.png](Screenshot_20260103_140627.png):
- Particle loss concentrated around **node 88456**
- Clear boundary between refined (small elements) and coarse (large elements)
- Particles "disappear" when crossing from refined to coarse region

---

## Hypothesis: L1 Face-Based Neighbor Search is Incomplete

### Current L1 Search Algorithm
```python
# L1: 3-hop face-based neighbor search
start_elem = cached_element
for hop in range(3):
    for face_neighbor in element_neighbors[current_elem]:
        if point_in_tet(pos, face_neighbor):
            return face_neighbor
    current_elem = best_neighbor  # Greedy step
```

### Problem: Face Neighbors Insufficient at 10× Size Jumps

**Face-based neighbor definition**:
- Two tetrahedra share a face (3 nodes)
- Works well for uniform mesh
- **Fails at refinement boundaries**

**Why it fails**:

1. **Small element** (43 µm characteristic length)
   - Has 4 faces
   - Each face shared with another small element
   - Face neighbors: all small (≈same size)

2. **Particle moves 100 µm** (2.3× element size)
   - Exits small element
   - Should enter adjacent large element (2.77 mm)
   - **But**: Large element doesn't share a face with small element!

3. **Large element** (2.77 mm characteristic length)
   - Spans dozens of small elements
   - Only shares faces with other large elements OR boundary small elements
   - Interior small elements: NOT face neighbors!

**Geometric visualization**:
```
Large element (2.77 mm)
┌─────────────────────────────────┐
│  ┌──┬──┬──┐  ┌──┬──┬──┐        │
│  │s1│s2│s3│  │s4│s5│s6│  ...   │  s1-s24: Small elements (43 µm)
│  ├──┼──┼──┤  ├──┼──┼──┤        │
│  │s7│s8│s9│  │..│..│..│        │  Only s1, s3, s7, s9, ... are
│  └──┴──┴──┘  └──┴──┴──┘        │  face neighbors of large element
└─────────────────────────────────┘
       ↑
       Particle in s5: 3 hops from edge
       Can't reach large element in 3 face-neighbor hops!
```

**Calculation**:
- Large element: 2.77 mm = 2770 µm
- Small element: 43.3 µm
- Ratio: 2770 / 43.3 ≈ **64 small elements** span one large element (linear dimension)
- Interior small element: **32 hops away from large element edge**!
- L1 search: Only **3 hops**

**Conclusion**: **L1 cannot reach large element from interior of refined region**

---

## Hypothesis Validation

### Test 1: L1 Hop Count Insufficient

**Expected finding**:
- Particles in center of refined region need >10 hops to reach coarse element
- Current L1: 3 hops (even with Phase 1.3 adaptive 6 hops)
- **Gap**: 3-6 hops vs 10-30 hops needed

**Diagnostic**: Run `diagnose_refinement_boundary_crossing.py`
- Check: Can L1 3-hop search reach large neighbors from small elements?
- Expected: **NO** for interior small elements

### Test 2: Neighbor Construction Completeness

**Expected finding**:
- Face-based neighbors: Incomplete at boundaries
- Small elements have 0-4 face neighbors (all small)
- Large elements have 0-4 face neighbors (mix of large + boundary small)
- **Missing**: Interior small ↔ large connections

**Diagnostic**: Check neighbor symmetry and coverage
- Are face-neighbor relations symmetric?
- Do boundary small elements have large neighbors?
- Do interior small elements have ANY path to large elements?

### Test 3: Morton Octree Alignment

**Expected finding**:
- Morton octree partitions space geometrically
- Refinement boundaries may NOT align with octree boundaries
- Particles at boundaries may be in "wrong" leaf for their containing element

**Diagnostic**: Check leaf assignments vs element locations
- Are small and large neighbors in same/adjacent leaves?
- Do leaves span large size variations?

---

## Root Cause Candidates (Ranked by Likelihood)

### 1. L1 Hop Count Insufficient (90% confidence)

**Evidence**:
- 262K× size variation
- 64× linear dimension ratio
- Particles need 10-30 hops to cross refined region
- Current: 3 hops (or 6 with adaptive)

**Fix**:
- Increase L1 hop count to 30 (1 hour)
- Use node-based neighbors instead of face-based (8 hours)
- Use spatial search instead of topological hops (4 hours)

**Expected gain**: +10-15% retention (to 95%+)

---

### 2. Face-Based Neighbors Incomplete (80% confidence)

**Evidence**:
- Screenshot shows loss at refined/coarse boundary
- Face neighbors don't capture interior→boundary connections

**Fix**:
- Switch to node-based neighbors (share 1+ nodes, not full face)
- Captures more connections (12-20 neighbors vs 4 face neighbors)
- Implementation: 8 hours

**Expected gain**: +5-10% retention (node neighbors capture interior→boundary)

---

### 3. Morton Octree Misalignment (30% confidence)

**Evidence**:
- Previous diagnostic showed 32% Morton search failures
- BUT: Enhanced Morton (5×5×5) didn't help much

**Analysis**:
- Morton is final fallback (L2)
- If L1 fails, particle already far from cached element
- Morton search may be too late

**Fix**: Not priority - L1 is the bottleneck

---

### 4. Point-in-Tet Tolerance Issues (5% confidence)

**Evidence**:
- Diagnostic showed 0% precision losses
- Current tolerance is adequate

**Analysis**: **NOT the issue** (proven by diagnostic)

---

## Recommended Investigation Plan

### Phase 1: Diagnostic (1 hour)

Run comprehensive boundary diagnostic:
```bash
python diagnose_refinement_boundary_crossing.py > logs/diagnose_boundary.log 2>&1
```

**Expected findings**:
1. L1 3-hop cannot reach 50%+ of large neighbors from small elements
2. Face-based neighbors miss 80%+ of geometric neighbors at boundaries
3. Some Morton leaves span 1000× size variation

---

### Phase 2: Quick Fix - Increase L1 Hop Count (1 hour)

**Implementation**:
```python
# In rk4_fully_fused_timedep.py
n_hops_adaptive = jnp.where(
    size_ratio < 0.1,
    jnp.int32(30),  # Increase from 6 to 30
    jnp.int32(3)
)
```

**Test**:
```bash
python production_tracking_fully_fused_timedep.py > logs/production_30hops.log 2>&1
```

**Expected**:
- Retention: 90-95% @ step 100
- Throughput: 4,000-5,000 p/s (30 hops = 10× slower L1, but only 10% particles affected)

**If this works**: Problem solved! (90% confidence)

---

### Phase 3: Robust Fix - Node-Based Neighbors (8 hours, if Phase 2 fails)

**Implementation**:
1. Modify `build_element_neighbors_array` to use node-based definition
2. Element A and B are neighbors if they share ≥1 node
3. Results in 12-20 neighbors per element (vs 4 face neighbors)

**Benefits**:
- Captures all geometric adjacencies
- Interior small elements connect to boundary large elements
- May reduce hops needed (denser graph)

**Drawbacks**:
- More neighbors → slower L1 search
- Memory: 4×-5× larger neighbor array

---

### Phase 4: Spatial Search (4 hours, if Phases 2-3 fail)

**Implementation**:
1. Instead of topological hops, use spatial radius search
2. Find all elements within R × char_length of particle
3. Bypasses neighbor graph entirely

**Benefits**:
- Guaranteed to find element if within radius
- Works for any mesh topology

**Drawbacks**:
- Requires spatial indexing (bounding volume hierarchy)
- More complex implementation

---

## Performance Impact Analysis

### Current (Failed) Enhanced Morton
- Throughput: 1,246 p/s (**5× slower**)
- Cause: Every particle searches 125 octants (expensive!)
- Lesson: Cannot afford expensive L2 search for all particles

### Proposed: Increase L1 Hops to 30
- **Best case** (L1 succeeds more often):
  - L1 slower: 30 hops vs 3 hops = 10× slower
  - But only 10% particles at boundaries need this
  - Average: 0.9 × 1.0 + 0.1 × 10.0 = **1.9× slower**
  - Throughput: 3,400 p/s (acceptable if retention fixes)

- **Worst case** (L1 still fails):
  - All particles do 30 hops → 10× slower L1
  - Throughput: 650 p/s (unacceptable!)
  - Need to proceed to Phase 3 (node neighbors)

### Proposed: Node-Based Neighbors
- More neighbors: 12-20 vs 4 → **3×-5× more point-in-tet tests**
- But fewer hops needed: 3-5 hops vs 30 hops
- Net: ~2× slower than current (acceptable)
- Throughput: ~3,000 p/s

---

## Critical Next Steps

1. **Revert enhanced Morton** (restore baseline performance):
   ```python
   # In rk4_fully_fused_timedep.py, line 226
   return search_L2_morton_neighbors_single(pos, mesh_gpu_global_morton)  # Restore
   ```

2. **Run diagnostic** to confirm root cause:
   ```bash
   python diagnose_refinement_boundary_crossing.py > logs/diagnose_boundary.log 2>&1
   ```

3. **Try 30-hop L1 first** (1 hour, 90% chance of success)

4. **If fails**: Implement node-based neighbors (8 hours)

---

## Summary

**What we learned**:
- ✅ Precision is NOT the issue (0% losses)
- ❌ Enhanced Morton doesn't help (wrong problem)
- 🔴 **Particles fail at refined→coarse crossings**

**Root cause hypothesis**:
- L1 face-based neighbors can't traverse 64× size jumps in 3 hops
- Need 10-30 hops OR denser neighbor graph

**Next action**:
1. Run diagnostic to confirm hypothesis
2. Try 30-hop L1 (quick fix, likely to work)
3. Fall back to node-based neighbors if needed

---

**Status**: Awaiting user to run `diagnose_refinement_boundary_crossing.py`
