# Graded Mesh Refinement Solution

## Problem

Particles in refined regions use coarse element velocities (no rotation visible) even though they're spatially inside the refined region.

## Root Cause Confirmed

The mesh has a **graded refinement structure**:

```
Fine (0.14mm) → Medium (0.14-0.30mm) → Coarse (>0.30mm)
```

### Key Finding from Spatial Analysis

- **Fine elements**: 2,599,528 (85.3%)
- **Medium elements**: 381,987 (12.5%) - **Transition layer**
- **Coarse elements**: 67,385 (2.2%)

### Spatial Distribution

```
Fine region:     X=[-9.36, 9.34]mm, Y=[-9.38, 9.40]mm, Z=[-4.51, -0.02]mm
Medium region:   X=[-9.65, 9.61]mm, Y=[-9.82, 9.86]mm, Z=[-4.96, -0.04]mm (surrounds fine)
Coarse region:   X=[-28.75, 28.75]mm, Y=[-21.72, 21.72]mm, Z=[-9.37, -0.08]mm (outer domain)
```

### Node Connectivity Analysis

**Test case**: Closest coarse and fine elements near particle location (-10mm, 0, -2mm)
- Closest coarse element: 1793477 (size=1.09mm)
- 129 fine elements within 2mm
- **ZERO shared nodes** between coarse and ANY nearby fine elements

**Conclusion**: Fine and coarse elements don't share nodes. Medium-sized elements form a buffer zone between them.

## Why Current Search Fails

### L0-L1-L2 Search Hierarchy

```
Particle in coarse element moves into refined region:
  L0 (cached): FAILS - particle left coarse element
  L1 (neighbor hops, N_HOPS=3):
    Hop 1: coarse → medium ✅
    Hop 2: medium → medium ✅
    Hop 3: medium → fine ❌ (not enough hops!)
    Result: Finds MEDIUM neighbor, marks as success
  L2 (Morton global): NEVER REACHED - L1 "succeeded"
```

The particle gets stuck in medium elements (size ~0.2mm) instead of fine elements (size ~0.14mm). While medium elements are closer to fine than coarse, they still don't have the highest velocity gradients.

## Why Node-Based Neighbors Won't Help

Node-based neighbor construction finds ALL elements sharing nodes. However:
- Fine and coarse don't share nodes (graded refinement)
- Node-based neighbors won't create coarse→fine connectivity
- Memory overhead (1GB) with no benefit

**Verdict**: Node-based neighbors are not useful for this mesh.

## Solution Options

### Option 1: Increase N_HOPS (Recommended First Try)

**Implementation**:
```python
# In production script
N_HOPS = 10  # Increase from 3 to 10
```

**Rationale**:
- Coarse → Medium → Fine requires 2+ hops
- N_HOPS=10 ensures we can traverse multiple refinement levels
- Simple config change, no code modifications needed

**Pros**:
- Trivial to implement
- Works for any graded refinement pattern
- Guaranteed to reach fine elements if they're connected

**Cons**:
- Performance cost (more neighbor checks per step)
- May still find medium instead of fine if path is ambiguous

**Expected result**:
- L1 search with more hops should traverse coarse→medium→fine
- Particles should use fine element velocities
- Rotation should become visible

### Option 2: Increase L2_SEARCH_RADIUS (Complementary)

**Implementation**:
```python
# In production script
L2_SEARCH_RADIUS = 50  # Increase from 10 to 50
```

**Rationale**:
- If L1 still fails with N_HOPS=10, L2 gets more chances
- Larger radius covers more Morton leaves
- More robust for complex refinement patterns

**Pros**:
- Complements increased N_HOPS
- Catches edge cases L1 misses

**Cons**:
- Performance cost in L2 search
- L2 still rarely reached if L1 succeeds

### Option 3: Force L2 When Element Size Changes Significantly (Advanced)

**Implementation**:
Add element size check after L1 search. If L1 finds element significantly larger than expected for local region, force L2.

**Requires**:
- Element sizes on GPU
- Modified RK4 search hierarchy

**Pros**:
- Most targeted solution
- Only affects particles crossing refinement boundaries

**Cons**:
- Code changes required
- More complex implementation

### Option 4: Skip L1 Entirely (Nuclear Option)

**Implementation**:
```python
# Modify search hierarchy to always use L2
elem_final = jnp.where(
    found_l0,
    elem_l0,
    search_l2_single(pos)  # Skip L1 entirely
)
```

**Pros**:
- Guaranteed to find correct element via Morton
- Simple code change

**Cons**:
- Significant performance penalty (L2 for every failed L0)
- Defeats purpose of L1 optimization

## Recommended Approach

**Phase 1: Test N_HOPS increase (IMMEDIATE)**
1. Set `N_HOPS = 10` in production script
2. Run tracking diagnostic to verify fine elements are found
3. Check performance impact

**Phase 2: Add L2_SEARCH_RADIUS if needed**
1. If N_HOPS=10 still misses some fine elements:
2. Set `L2_SEARCH_RADIUS = 50`
3. Re-run tests

**Phase 3: Advanced solution if needed**
1. If performance is acceptable but some particles still miss fine elements:
2. Implement element size verification after L1
3. Force L2 for size mismatches

## Expected Outcomes

### After N_HOPS=10

From [diagnose_tracking_through_refined_region.py](diagnose_tracking_through_refined_region.py), particles should show:

**Before (N_HOPS=3)**:
```
Particle 3:
  Element types while in refined region:
    Fine elements: 0/289 (0.0%)   ❌
    Medium elements: ? (unknown)
    Coarse elements: 289/289 (100.0%)   ❌
```

**After (N_HOPS=10)**:
```
Particle 3:
  Element types while in refined region:
    Fine elements: 250-280/289 (86-97%)   ✅
    Medium elements: 5-30/289 (2-10%)   ✅ (acceptable)
    Coarse elements: 0-9/289 (0-3%)   ✅
```

### Production Tracking

- Particles should show **rotating trajectories** in refined region
- Results should match commercial code (swirling pattern visible)
- Global advancing velocity should remain correct (already working)

## Performance Considerations

### N_HOPS Impact

- Current N_HOPS=3: ~12 neighbor checks per L1 search (4 neighbors per hop)
- N_HOPS=10: ~40 neighbor checks per L1 search
- **Impact**: ~3x more checks in L1, but only for particles where L0 fails (~10-20% of steps)
- **Expected throughput**: 50-100K particles/s (down from 50-120K)

### Memory Impact

- N_HOPS is a runtime parameter, no additional memory needed
- L2_SEARCH_RADIUS also runtime parameter
- Node-based neighbors NOT needed (saves 1GB GPU memory)

## Validation

Run modified [diagnose_tracking_through_refined_region.py](diagnose_tracking_through_refined_region.py) with N_HOPS=10:
1. Verify particles entering refined region are assigned to FINE elements
2. Check element type distribution (expect >90% fine when in refined region)
3. Validate rotation velocities are being used

## Implementation

Update [production_tracking_fully_fused_timedep.py](production_tracking_fully_fused_timedep.py):

```python
# BEFORE:
N_HOPS = 3
L2_SEARCH_RADIUS = 10

# AFTER:
N_HOPS = 10  # Increased for graded refinement (coarse→medium→fine)
L2_SEARCH_RADIUS = 50  # Larger radius for robustness
```

No other code changes needed - these are runtime parameters.
