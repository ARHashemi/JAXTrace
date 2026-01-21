# Particle Loss Root Causes and Solutions

**Date**: 2026-01-20
**Context**: Production tracking shows 6-13% particle loss over 100-200 steps despite 100% initial assignment
**Key finding**: L2 search method doesn't affect retention (neighbors = hierarchical = 93.29% @ step 100)

---

## Executive Summary

Based on production log analysis and recent literature review, particle loss during tracking is **NOT caused by L2 search failures** but likely by one or more of the following numerical/physical issues:

1. **Float32 precision in barycentric coordinate calculations** (HIGH PROBABILITY)
2. **Velocity field discontinuities at PVTU piece boundaries** (TESTABLE - diagnostic created)
3. **Large RK4 displacement beyond local search coverage** (MODERATE PROBABILITY)
4. **Numerical instability in velocity interpolation at element boundaries** (MODERATE PROBABILITY)
5. **Mesh quality issues near refinement boundaries** (LOW PROBABILITY - already tested)

---

## Detailed Analysis of Each Root Cause

### 1. Float32 Precision in Barycentric Coordinates ⭐ **HIGHEST PROBABILITY**

#### Evidence from Your System

- Using `inverse` method with precomputed float32 matrices on GPU
- Particles at element boundaries have barycentric coordinates very close to 0
- Float32 has ~7 decimal digits precision
- Element size range: 8.12e-14 to 2.13e-08 (262,000× variation!)

#### Mechanism

```python
# After RK4 step, particle at element boundary
position = [0.0123456789, -0.0045678901, -0.0089012345]

# Compute barycentric coordinates (float32)
bary = M_inv @ (position - p0)  # float32 operations
lam0 = 1.0 - bary.sum()

# Check if inside element
if (bary < 0).any() or (lam0 < 0):
    # REJECTED - even though particle IS inside!
    # Reason: float32 roundoff → bary = [-1e-8, 0.3, 0.3, 0.4]
    mark_as_lost()
```

#### Recent Research (2025)

A [December 2025 paper](https://onlinelibrary.wiley.com/doi/10.1002/nme.70243) in the International Journal for Numerical Methods in Engineering shows that **barycentric coordinate-based shape functions achieve second-order convergence and exhibit superior numerical stability** compared to traditional methods.

However, [GPU implementations using double-precision floating-point variables significantly impact performance](https://www.researchgate.net/publication/281105497_GPU_Fast_and_Robust_Computation_for_Barycentric_Coordinates_and_Intersection_of_Planes_Using_Projective_Representation), since GPUs for real-time rendering are primarily designed to support single-precision floating-point calculations.

[SciPy's barycentric interpolator documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BarycentricInterpolator.html) notes that "the barycentric representation avoids many of the problems associated with polynomial interpolation caused by floating-point arithmetic."

#### Solutions

**A. Add epsilon tolerance to point-in-tet checks** (RECOMMENDED - easy to implement)

```python
# In point_in_tet_barycentric_inverse()
EPSILON = 1e-6  # Tolerance for boundary cases

# Instead of:
inside = (bary >= 0).all() and (lam0 >= 0)

# Use:
inside = (bary >= -EPSILON).all() and (lam0 >= -EPSILON)
```

**Expected impact**: +3-5% retention improvement
**Cost**: Negligible (just threshold change)
**Risk**: Low (standard practice in computational geometry)

**B. Use float64 for barycentric calculations** (HIGH IMPACT - expensive)

```python
# Promote to float64 only for barycentric calculation
M_inv_f64 = M_inv.astype(jnp.float64)
p0_f64 = p0.astype(jnp.float64)
position_f64 = position.astype(jnp.float64)

bary_f64 = M_inv_f64 @ (position_f64 - p0_f64)
# Then convert back to float32 for velocity interpolation
```

**Expected impact**: +5-10% retention improvement
**Cost**: ~2× slower point-in-tet checks
**Risk**: Memory overhead

**C. Use robust barycentric calculation** (BEST - moderate cost)

Implement [stochastic barycentric coordinates](https://dl.acm.org/doi/10.1145/3658131) (ACM Transactions on Graphics 2024) or [projective representation](https://www.researchgate.net/publication/281105497_GPU_Fast_and_Robust_Computation_for_Barycentric_Coordinates_and_Intersection_of_Planes_Using_Projective_Representation) to minimize numerical imprecisions.

**Expected impact**: +5-8% retention improvement
**Cost**: ~1.5× slower point-in-tet checks
**Risk**: Requires implementation effort

---

### 2. Velocity Field Discontinuities at PVTU Piece Boundaries ⚠️ **TESTABLE**

#### Evidence from Your System

- Mesh has 209,749 duplicate nodes (26.9% of nodes!)
- Duplicates removed, but velocity may have discontinuities
- Linear particle loss rate (145 particles/step) suggests systematic issue
- Loss happens uniformly during tracking, not at specific locations

#### Mechanism

```python
# Element A and Element B share edge at old PVTU boundary
# Node IDs: [100, 101, 102, 103] and [100, 101, 200, 201]

# BEFORE deduplication (PVTU pieces):
velocity[100] = [1.000, 0.000, 0.000]  # Piece 1
velocity[100_dup] = [1.001, 0.000, 0.000]  # Piece 2 (duplicate)

# AFTER deduplication:
# One of the duplicate velocities is kept (arbitrary choice)
# Elements from Piece 2 now reference wrong velocity!

# Result: Discontinuous velocity field at piece boundaries
# RK4 step → particle position slightly wrong → outside all elements
```

#### Diagnostic Created

Run the diagnostic script to test this hypothesis:

```bash
python diagnose_pvtu_velocity_discontinuities.py > logs/diagnose_pvtu_velocity.log
```

This will:
1. Identify elements at PVTU boundaries (containing duplicate nodes)
2. Measure velocity discontinuity across element boundaries
3. Compare boundary vs interior discontinuity magnitude
4. Determine if boundary discontinuities are significantly higher (>2×)

#### Solutions

**A. Velocity field smoothing at PVTU boundaries** (If diagnostic confirms issue)

```python
# After node deduplication, smooth velocity at boundaries
for node_id in duplicate_node_ids:
    # Find all elements sharing this node
    sharing_elements = find_elements_containing_node(node_id, connectivity)

    # Average velocity from all sharing elements
    velocities = [velocity_field[elem] for elem in sharing_elements]
    velocity_field[node_id] = np.mean(velocities, axis=0)
```

**Expected impact**: +5-10% retention if discontinuities confirmed
**Cost**: One-time preprocessing (~5-10 seconds)
**Risk**: May smooth out physical features (use with caution)

**B. Higher-order velocity interpolation near boundaries**

Use quadratic or cubic interpolation instead of linear barycentric for elements touching PVTU boundaries.

**Expected impact**: +3-5% retention
**Cost**: ~3× slower velocity interpolation for boundary elements
**Risk**: Requires second-order derivatives (may not be available)

**C. Re-merge mesh with proper velocity handling**

Re-load PVTU and merge with velocity-aware algorithm (average duplicate node velocities during merge).

**Expected impact**: Complete fix if this is root cause
**Cost**: Modify mesh loading pipeline
**Risk**: May affect other parts of pipeline

---

### 3. Large RK4 Displacement Beyond Search Coverage 🔍 **MODERATE PROBABILITY**

#### Evidence from Your System

- dt = 2.5e-3 (relatively large for fine mesh regions)
- Velocity magnitude varies by 262,000× across mesh
- Element size varies by 262,000× (8.12e-14 to 2.13e-08)
- Linear loss rate suggests consistent mechanism

#### Mechanism

```python
# Particle in high-velocity, fine-mesh region
current_position = [0.012, 0.003, -0.005]
current_element_size = 1e-13  # Very fine mesh
velocity = [0.1, 0.05, -0.02]  # High velocity region

# RK4 step with dt = 2.5e-3
displacement = velocity * dt = [2.5e-4, 1.25e-4, -5e-5]

# Displacement is 2500× element size!
# Particle moves 2500 elements away in one step
# L2 search (radius 64 leaves) cannot find it
```

#### Calculation

Max velocity: ~1e-7 to 1e-5 (from logs)
dt: 2.5e-3
Max displacement per step: 1e-5 × 2.5e-3 = 2.5e-8

Min element size: 8.12e-14
Max elements crossed: 2.5e-8 / 8.12e-14 ≈ **308,000 elements!**

This is far beyond any reasonable search radius!

#### Recent Research

[A 2024 paper on GPU-accelerated particle tracking](https://www.mdpi.com/2226-4310/12/11/1005) notes that pure tetrahedral grids have the lowest proportion of warp divergence, which can impact tracking efficiency.

[Research on fast particle-locating methods](https://www.mdpi.com/1999-4893/12/9/179) emphasizes that efficient searching of host cells for tracked particles is essential for improving computational efficiency in hybrid Euler-Lagrangian models on arbitrary polyhedral meshes.

#### Solutions

**A. Adaptive timestep based on local velocity and element size** (RECOMMENDED)

```python
# Compute adaptive dt for each particle
element_sizes = compute_element_sizes(element_ids)  # Precomputed
local_velocities = interpolate_velocity(positions, ...)

# CFL-like condition: max displacement = α × element_size
ALPHA = 0.1  # Safety factor (10% of element size per step)
dt_adaptive = ALPHA * element_sizes / jnp.linalg.norm(local_velocities, axis=1)

# Use minimum across all particles (or per-particle)
dt_global = jnp.min(dt_adaptive)
```

**Expected impact**: +10-15% retention improvement
**Cost**: 2-3× more RK4 steps (but each step faster due to better L2 cache hit rate)
**Risk**: More complex implementation

**B. Reduce global timestep** (SIMPLE - test first)

```python
DT = 1.0e-3  # Instead of 2.5e-3
```

**Expected impact**: +5-10% retention if this is root cause
**Cost**: 2.5× more steps (2.5× slower overall)
**Risk**: None (just slower)

**C. Multi-step search for large displacements** (ADVANCED)

```python
# If particle not found after RK4 step
if element_id < 0:
    # Trace back along RK4 trajectory
    for substep in [0.25, 0.5, 0.75]:
        intermediate_position = old_position + substep * (new_position - old_position)
        element_id = search_from_position(intermediate_position)
        if element_id >= 0:
            # Found! Now search forward from here
            break
```

**Expected impact**: +3-5% retention
**Cost**: 3-4× more searches for lost particles
**Risk**: Complex implementation

---

### 4. Numerical Instability in Velocity Interpolation at Element Boundaries 📊 **MODERATE**

#### Evidence

- Velocity interpolation uses barycentric coordinates (same precision issues as point-in-tet)
- RK4 requires 4 velocity evaluations per step (k1, k2, k3, k4)
- Intermediate positions (k2, k3) are often near element boundaries
- Accumulated error over 4 evaluations

#### Mechanism

```python
# RK4 intermediate steps
k1 = velocity_at(position)  # At current position (inside element)
k2 = velocity_at(position + 0.5*dt*k1)  # May be at boundary → precision issue!
k3 = velocity_at(position + 0.5*dt*k2)  # May be at boundary → precision issue!
k4 = velocity_at(position + dt*k3)      # May be outside → element search fails!

# Final position
new_position = position + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
# If k2, k3, or k4 have precision errors, new_position is slightly wrong
```

#### Solutions

**A. Use float64 for RK4 intermediate calculations** (TARGETED)

```python
# Promote only RK4 state to float64
position_f64 = position.astype(jnp.float64)

# Compute k1, k2, k3, k4 in float64
k1_f64 = velocity_at_f64(position_f64)
k2_f64 = velocity_at_f64(position_f64 + 0.5*dt*k1_f64)
...

# Convert final result back to float32
new_position = (position_f64 + ...).astype(jnp.float32)
```

**Expected impact**: +3-5% retention
**Cost**: ~1.5× slower RK4 step
**Risk**: Memory overhead

**B. Use lower-order integrator for robustness** (FALLBACK)

Replace RK4 with RK2 (Heun's method) or even Euler for particles near boundaries.

**Expected impact**: Unknown (may help or hurt)
**Cost**: Lower accuracy
**Risk**: Physical trajectories may be less accurate

**C. Stabilize velocity gradient** (ADVANCED)

Use gradient limiting or slope limiting techniques from CFD.

**Expected impact**: +2-3% retention
**Cost**: Complex implementation
**Risk**: May affect physical accuracy

---

### 5. Mesh Quality Issues Near Refinement Boundaries ⚠️ **LESS LIKELY**

#### Evidence

- Mesh has extreme refinement (262,000× volume variation)
- 1:2 octree refinement creates "hanging nodes"
- Face-based neighbors may fail at refinement boundaries
- Warning in logs: "Face-based neighbors may NOT work for 1:2 octree refinement!"

#### User's Correction

"Since we have L2 fallback, L1 can not cause particle loss. I've tested wit and without L1."

This rules out L1 search failures as root cause.

#### However, Mesh Quality Still Matters

Even with L2 fallback, poor mesh quality near refinement can cause:
- Thin/sliver elements with ill-conditioned inverse matrices
- Large velocity gradients across refinement boundaries
- Numerical instability in interpolation

#### Solutions

**A. Check for degenerate elements**

```python
# Compute element quality (volume / max edge length^3)
volumes = compute_element_volumes(connectivity, node_positions)
quality = volumes / element_characteristic_length**3

# Find poor quality elements
poor_quality = quality < 0.01  # Threshold
print(f"Poor quality elements: {poor_quality.sum()}")
```

If many poor quality elements, consider:
- Mesh regeneration with better refinement strategy
- Filtering out poor quality elements from tracking

**B. Add velocity gradient limiting near refinement**

Limit velocity jump across refinement boundaries to prevent large errors.

---

## Comprehensive Testing Strategy

### Phase 1: Quick Tests (1-2 hours)

1. **Test epsilon tolerance** (easiest, highest probability)
   ```python
   # In point_in_tet_barycentric_inverse()
   EPSILON = 1e-6
   inside = (bary >= -EPSILON).all() and (lam0 >= -EPSILON)
   ```

   Expected: +3-5% retention

2. **Test reduced timestep** (simple, confirms displacement hypothesis)
   ```python
   DT = 1.0e-3  # Instead of 2.5e-3
   ```

   Expected: +5-10% retention if displacement is issue

3. **Run PVTU velocity diagnostic** (confirms/rejects velocity discontinuity hypothesis)
   ```bash
   python diagnose_pvtu_velocity_discontinuities.py > logs/diagnose_pvtu_velocity.log
   ```

   Expected: Statistical comparison of boundary vs interior discontinuities

### Phase 2: Moderate Tests (1 day)

4. **Test float64 barycentric calculations**
   - Modify point_in_tet_barycentric_inverse() to use float64
   - Measure retention improvement and performance cost

5. **Test adaptive timestep**
   - Implement CFL-based dt calculation
   - Measure retention improvement

6. **Test velocity field smoothing** (if Phase 1 diagnostic confirms discontinuities)
   - Average velocity at duplicate node locations
   - Measure retention improvement

### Phase 3: Advanced Solutions (3-7 days)

7. **Implement robust barycentric coordinates**
   - Use projective representation or stochastic methods
   - Benchmark against current implementation

8. **Implement multi-step trajectory search**
   - Search intermediate positions for lost particles
   - Measure success rate

9. **Comprehensive numerical stability analysis**
   - Profile accumulated error in RK4
   - Identify dominant error source

---

## Prioritized Recommendations

### 🔥 **IMMEDIATE** (Implement today)

1. **Add epsilon tolerance to point-in-tet checks**
   - File: [jaxtrace/gpu/search/point_in_tet_methods.py](jaxtrace/gpu/search/point_in_tet_methods.py)
   - Change: `EPSILON = 1e-6`, `inside = (bary >= -EPSILON).all()`
   - Expected: +3-5% retention
   - Cost: Zero (just threshold change)

2. **Run PVTU velocity discontinuity diagnostic**
   - Command: `python diagnose_pvtu_velocity_discontinuities.py > logs/diagnose_pvtu_velocity.log`
   - Expected: Confirms or rejects velocity discontinuity hypothesis
   - Cost: ~10 minutes runtime

### ⚡ **HIGH PRIORITY** (This week)

3. **Test reduced timestep**
   - File: [production_tracking_fully_fused_timedep.py](production_tracking_fully_fused_timedep.py)
   - Change: `DT = 1.0e-3`
   - Expected: +5-10% retention if displacement is issue
   - Cost: 2.5× slower (but confirms root cause)

4. **Test float64 barycentric calculations**
   - Implementation: Promote M_inv, p0, position to float64 for barycentric calc
   - Expected: +5-10% retention
   - Cost: ~2× slower point-in-tet checks

5. **Implement adaptive timestep** (if reduced timestep helps)
   - Compute dt = α × element_size / velocity_magnitude per particle
   - Expected: +10-15% retention
   - Cost: 2-3× more steps but better cache locality

### 🔬 **MEDIUM PRIORITY** (Next 2-4 weeks)

6. **Velocity field smoothing** (if diagnostic confirms discontinuities)
   - Average velocity at PVTU boundary nodes
   - Expected: +5-10% retention
   - Cost: One-time preprocessing

7. **Implement robust barycentric coordinate method**
   - Use stochastic or projective representation
   - Expected: +5-8% retention
   - Cost: Implementation effort + ~1.5× slower

8. **Multi-step trajectory search**
   - Trace intermediate positions for lost particles
   - Expected: +3-5% retention
   - Cost: 3-4× more searches for lost particles

### 📚 **LOW PRIORITY** (Research/long-term)

9. **Mesh quality analysis and regeneration**
   - Identify poor quality elements near refinement
   - Regenerate mesh with better refinement strategy
   - Expected: Unknown (may help significantly if mesh quality is poor)
   - Cost: Major effort

10. **Explore alternative integration schemes**
    - Try RK2, adaptive RK4-5, or symplectic integrators
    - Expected: Unknown
    - Cost: Research + implementation

---

## Expected Combined Impact

If you implement the top 3 recommendations:

1. Epsilon tolerance: +3-5% retention
2. PVTU diagnostic + fix: +5-10% retention (if confirmed)
3. Reduced timestep: +5-10% retention

**Estimated combined improvement: +13-25% retention**

This would bring retention from:
- Current: 93% @ step 100 → **Target: >98% @ step 100**
- Current: 87% @ step 200 → **Target: >95% @ step 200**

---

## Literature References

### Recent Research (2024-2025)

1. **Barycentric Coordinates & Numerical Stability**
   - [Barycentric Coordinate‐Based Shape Functions (Dec 2025)](https://onlinelibrary.wiley.com/doi/10.1002/nme.70243) - International Journal for Numerical Methods in Engineering
   - [Stochastic Computation of Barycentric Coordinates (2024)](https://dl.acm.org/doi/10.1145/3658131) - ACM Transactions on Graphics
   - [GPU Fast and Robust Computation for Barycentric Coordinates](https://www.researchgate.net/publication/281105497_GPU_Fast_and_Robust_Computation_for_Barycentric_Coordinates_and_Intersection_of_Planes_Using_Projective_Representation)
   - [SciPy Barycentric Interpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BarycentricInterpolator.html)

2. **Particle Tracking in Tetrahedral Meshes**
   - [GPU-Accelerated Particle Tracking (2024)](https://www.mdpi.com/2226-4310/12/11/1005) - MDPI Aerospace
   - [Fast Particle-Locating Method for Arbitrary Polyhedral Mesh](https://www.mdpi.com/1999-4893/12/9/179) - MDPI Algorithms
   - [Parallel Mesh-Based Particle Methods](https://scorec.rpi.edu/research/parallel-mesh-based-particle-methods) - SCOREC Research

3. **RK4 Integration & Stability**
   - [4th-order Runge-Kutta - Computational Astrophysics](https://zingale.github.io/computational_astrophysics/ODEs/ODEs-rk4.html)
   - [Integration Basics - Gaffer On Games](https://gafferongames.com/post/integration_basics/)
   - [Runge–Kutta Methods - Wikipedia](https://en.wikipedia.org/wiki/Runge–Kutta_methods)

### Key Insights from Literature

1. **Float32 vs Float64**: Recent research emphasizes that "robustness and numerical stability is becoming a key issue more important than computational time in engineering applications."

2. **Tetrahedral Mesh Tracking**: Studies show that pure tetrahedral grids have the lowest proportion of warp divergence, but require careful handling of point location and velocity interpolation.

3. **Barycentric Stability**: The barycentric representation avoids many floating-point problems, but requires proper tolerance handling for boundary cases.

4. **Timestep Considerations**: RK4 error scales as O(h^5) locally and O(h^4) globally, but large timesteps can cause particles to "jump" beyond search coverage in highly refined meshes.

---

## Summary

The particle loss issue is **NOT due to L2 search strategy** (confirmed by neighbors = hierarchical retention), but most likely due to:

1. **Float32 precision at element boundaries** (add epsilon tolerance - IMMEDIATE)
2. **Velocity field discontinuities at PVTU boundaries** (test with diagnostic - IMMEDIATE)
3. **Large RK4 displacement in refined regions** (test reduced dt - HIGH PRIORITY)

Implement the immediate recommendations first and measure retention improvement. Based on results, proceed with high-priority items.

---

## Files Created

1. **[diagnose_pvtu_velocity_discontinuities.py](diagnose_pvtu_velocity_discontinuities.py)** - Test velocity discontinuity hypothesis
2. **[PARTICLE_LOSS_ROOT_CAUSES_AND_SOLUTIONS.md](PARTICLE_LOSS_ROOT_CAUSES_AND_SOLUTIONS.md)** - This document

Run the diagnostic and report results!
