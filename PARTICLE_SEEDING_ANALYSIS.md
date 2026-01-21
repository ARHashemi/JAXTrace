# Particle Seeding and Refined Region Analysis

## Critical Finding

**ZERO particles are initially seeded in the refined region (rotating tool area).**

## Spatial Configuration

### Mesh Domain
- **X**: [-60, 0] mm (60 mm total length)
- **Y**: [-23, 23] mm (46 mm total width)
- **Z**: [-10, 0] mm (10 mm total height)

### Refined Region (Rotating Tool)
- **X**: [-9.04, +9.02] mm (18.06 mm wide)
- **Y**: [-9.16, +9.16] mm (18.32 mm wide)
- **Z**: [-4.51, -0.02] mm (4.49 mm high)
- **Elements**: 288,749 elements (9.5% of total)
- **Element size**: 0.1363 mm (smallest in mesh)

### Particle Seeding Region
- **X**: [-24.00, -12.00] mm (20% of domain length, entrance region)
- **Y**: [-13.80, +13.80] mm (60% of domain width)
- **Z**: [-7.00, 0.00] mm (70% of domain height)
- **Total particles**: 225,000 (50×90×50 grid)

## Configuration (from production_tracking_fully_fused_timedep.py)

```python
PARTICLE_BOUNDS_FRACTION = {
    'x': (0.1, 0.3),  # Use first 20% of domain in X (entrance region)
    'y': (0.2, 0.8),  # 60% domain in Y
    'z': (0.3, 1.0),  # 70% domain in Z
}
```

## Problem Analysis

### Seeding Strategy
The current seeding strategy seeds particles at the **entrance region** (left side, negative X) of the domain. This is intentional for friction stir welding, where:
1. Material enters from the left (-X direction)
2. Tool rotates in the center (around X=0, Y=0)
3. Material exits to the right (+X direction)

### Expected Behavior
Particles should:
1. Start at X ∈ [-24, -12] mm (entrance)
2. Advect with advancing velocity (positive X direction)
3. **Pass through refined region** at X ∈ [-9, +9] mm
4. Experience rotating velocities from fine elements
5. Show swirling trajectories

### Actual Behavior
Particles are NOT showing rotation, which means one of the following:
1. **Particles never reach the refined region** (unlikely - commercial code shows they do)
2. **Particles pass through but are assigned to COARSE neighboring elements** (most likely)
3. **Search fails in refined region during tracking** (possible - needs verification)

## Initial Assignment Results (Cascading Search)

### Overall Assignment
- **Initial search (radius=100)**: 78.32% success (176,221/225,000)
- **Fallback (radius=200)**: +437 particles → 78.51%
- **Fallback (radius=500)**: +367 particles → 78.68%
- **Fallback (radius=1000)**: OOM error (out of GPU memory)

### Refined Region Assignment
- **Particles in refined region at t=0**: 0 (none seeded there)
- **Cannot verify fine vs coarse assignment** without tracking particles into refined region

## Root Cause Hypothesis

The lack of rotation in tracking results suggests:

### Most Likely: Incorrect Element Assignment During Tracking

When particles advect into the refined region during tracking:
1. RK4 step moves particle into refined region
2. L0 (cached element) search fails (particle left previous element)
3. L1 (neighbor hop) search finds a COARSE neighbor instead of fine element
4. L2 (Morton) search may also fail to find correct fine element
5. Particle continues with COARSE element velocity (no rotation)

**Evidence supporting this**:
- Degeneracy threshold fix (1e-17) only affects point-in-tet test
- It doesn't affect the L1 neighbor search strategy
- Fine elements may not be in the neighbor list of coarse elements
- Morton search with small radius (L2_SEARCH_RADIUS=10) may miss fine elements

### Alternative: Insufficient Search During Tracking

Current tracking configuration:
- **N_HOPS**: 3 (searches up to 3 hops in neighbor graph)
- **L2_SEARCH_RADIUS**: 10 (searches ±10 leaves in Morton curve)

For highly refined meshes, these parameters may be insufficient to find fine neighbors from coarse elements.

## Recommended Next Steps

### 1. Track Particle Element Assignments Through Refined Region

Create a diagnostic that:
- Seeds a small number of particles in entrance region
- Tracks them through the domain
- Records element ID at each step
- Checks if element is fine or coarse
- Visualizes when particles enter/exit refined region

### 2. Verify Search Parameters Are Sufficient

Test if increasing search parameters helps:
- Increase `N_HOPS` from 3 to 5 or 10
- Increase `L2_SEARCH_RADIUS` from 10 to 50 or 100
- Check if this improves fine element assignment in refined region

### 3. Add Localized Particle Seeding in Refined Region (Verification Test)

As a VERIFICATION test:
- Seed 1,000 particles directly in refined region centroids
- Run tracking for 10 steps
- Check if these particles:
  - Stay assigned to fine elements
  - Show rotating velocities
  - Match commercial code behavior

### 4. Implement Adaptive Search Based on Element Size

If the issue is L1/L2 search not finding fine elements:
- Use element size to determine search radius
- For fine elements: use larger search radius in L2
- For coarse elements: use standard radius

## Memory Issue

The cascading search with radius=1000 caused OOM error:
```
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 2400720440 bytes
```

This suggests we need to:
1. Use smaller fallback radii (e.g., [150, 200, 300] instead of [200, 500, 1000])
2. Process unassigned particles in smaller batches
3. Accept some unassigned particles rather than OOM

## Conclusion

**The core issue is NOT particle seeding** - particles ARE seeded in the entrance region as intended.

**The core issue is likely element assignment DURING TRACKING** - when particles advect into the refined region, they're being assigned to coarse elements instead of fine elements, causing them to use wrong (non-rotating) velocities.

Next step: Create a tracking diagnostic to verify this hypothesis by monitoring element assignments as particles move through the refined region.
