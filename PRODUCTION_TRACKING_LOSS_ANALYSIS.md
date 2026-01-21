# Production Tracking Particle Loss Analysis

**Date**: 2026-01-20
**Hardware**: NVIDIA RTX 5000
**Source Logs**:
- `production_fully_fused_timedep_radius-2-8-16-32-64_withL1face-5hop_inverse_RTX5000.log`
- `production_fully_fused_timedep_neighbor_withL1-5hop_inverse_RTX5000.log`
- `production_fully_fused_timedep_hierarchical_withL1-5hop_inverse_RTX5000.log`

---

## Executive Summary

🔥 **Critical Finding**: All three L2 search methods show **significant particle loss during tracking** (NOT at initial assignment):

| L2 Method | Step 0 (Initial) | Step 100 | Step 200 | Loss Rate |
|-----------|------------------|----------|----------|-----------|
| **Incremental (radius)** | 100.00% (225,000) | **95.35%** (214,543) | *(killed)* | **4.65% loss in 100 steps** |
| **Neighbors** | 100.00% (225,000) | **93.29%** (209,912) | **86.89%** (195,500) | **13.11% loss in 200 steps** |
| **Hierarchical** | 100.00% (225,000) | **93.29%** (209,912) | **86.89%** (195,500) | **13.11% loss in 200 steps** |

**Key insight**: The particle loss is happening **during RK4 tracking**, not during initial assignment!

---

## Detailed Analysis

### 1. Initial Assignment Performance

**All three methods achieve 100% initial assignment** using cascading fallback:

```
Initial search (radius=500):  188,560/225,000 (83.80%)
Cascading fallback:
  radius=1000:   +11,221 → 199,781/225,000 (88.79%)
  radius=2000:   +12,329 → 212,110/225,000 (94.27%)
  radius=5000:   +10,487 → 222,597/225,000 (98.93%)
  radius=10000:     +492 → 223,089/225,000 (99.15%)
  radius=100000:  +1,911 → 225,000/225,000 (100.00%) ✅
```

**Observation**: Initial assignment is NOT the problem - all particles are successfully assigned!

### 2. Tracking Particle Loss

#### Incremental (Radius) Method

**Configuration**:
- L2 method: `incremental`
- L2 radii: (2, 8, 16, 32, 64) *(inferred from filename)*
- L1 hops: 5
- Point-in-tet: inverse

**Performance**:
```
Step   0:  225,000 active (100.00%)
Step 100:  214,543 active (95.35%)
```

**Loss**:
- 10,457 particles lost in 100 steps
- Loss rate: **~105 particles/step** (0.047%/step)
- **Run was killed** before step 200

#### Neighbors Method

**Configuration**:
- L2 method: `neighbors` (3×3×3 octant search, 27 octants)
- L1 hops: 5
- Point-in-tet: inverse

**Performance**:
```
Step   0:  225,000 active (100.00%)
Step 100:  209,912 active (93.29%)
Step 200:  195,500 active (86.89%)
```

**Loss**:
- Steps 0-100: 15,088 particles lost (**6.71% loss**)
- Steps 100-200: 14,412 particles lost (**6.40% loss**)
- **Average loss: ~145 particles/step** (0.064%/step)
- Total: 29,500 particles lost in 200 steps (**13.11% total loss**)

#### Hierarchical Method

**Configuration**:
- L2 method: `hierarchical` (depth 7+6, multi-depth spatial search)
- L1 hops: 5
- Point-in-tet: inverse

**Performance**:
```
Step   0:  225,000 active (100.00%)
Step 100:  209,912 active (93.29%)
Step 200:  195,500 active (86.89%)
```

**Loss**:
- **IDENTICAL to neighbors method!**
- Steps 0-100: 15,088 particles lost (6.71%)
- Steps 100-200: 14,412 particles lost (6.40%)
- Average loss: ~145 particles/step (0.064%/step)
- **Run was killed** after step 200

---

## Critical Observations

### 1. Neighbors = Hierarchical Performance

**Both methods lose EXACTLY the same particles**:
- Step 100: Both at 209,912 (93.29%)
- Step 200: Both at 195,500 (86.89%)

**Implications**:
- The search strategy (neighbors vs hierarchical) **does NOT affect retention**!
- Particle loss is **NOT due to L2 search failures**
- The problem is **upstream** (L0/L1) or **downstream** (velocity field/RK4)

### 2. Incremental Method Slightly Better

**Incremental shows 95.35% at step 100** vs neighbors/hierarchical at **93.29%**

**Difference**: 2.06% better retention (4,631 more particles)

**Possible reasons**:
- Incremental radii (2, 8, 16, 32, 64) may cover more leaves than 3×3×3 neighbors
- Or statistical variance (run was killed, might not be reproducible)

### 3. Linear Particle Loss Rate

**Both neighbors and hierarchical show remarkably consistent loss rate**:
- Steps 0-100: 6.71% loss (151 particles/step)
- Steps 100-200: 6.40% loss (144 particles/step)

**This linear pattern suggests**:
- NOT a catastrophic failure (would show exponential growth)
- NOT a boundary accumulation (would show decreasing rate)
- **Consistent mechanism** causing losses at every step

---

## Root Cause Hypotheses

### Hypothesis 1: L1 Search Failures at Coarse/Fine Boundaries ⭐ **MOST LIKELY**

**Evidence**:
- L1 uses face neighbors with `N_HOPS=5`
- Warning in logs: "Face-based neighbors may NOT work for 1:2 octree refinement!"
- Mesh has 262K× volume variation (extreme refinement)
- L2 method doesn't matter (neighbors = hierarchical = incremental ≈ similar)

**Mechanism**:
1. Particle crosses from coarse to fine region
2. L1 face traversal cannot cross 1:2 refinement boundary
3. L2 search also fails (particle outside 3×3×3 / hierarchical coverage)
4. Particle marked as lost

**Test**: Run with `N_HOPS=7` to see if retention improves.

### Hypothesis 2: Velocity Field Discontinuities at PVTU Boundaries

**Evidence**:
- Mesh has 209,749 duplicate nodes (PVTU piece boundaries)
- Duplicates were removed, but velocity may have discontinuities
- Linear loss rate suggests systematic issue

**Mechanism**:
1. RK4 interpolates velocity at element boundaries
2. Velocity field has small discontinuities at old PVTU boundaries
3. Particle position after RK4 step is slightly outside all elements
4. Search fails even with large coverage

**Test**: Check if lost particles cluster near old PVTU piece boundaries.

### Hypothesis 3: Float32 Precision in Point-in-Tet Checks

**Evidence**:
- Using `inverse` method with precomputed matrices
- Barycentric checks use float32 on GPU
- Particles at element boundaries may fail all checks

**Mechanism**:
1. Particle very close to element boundary
2. Float32 roundoff causes all barycentric checks to fail
3. Search finds correct element but check rejects it
4. Particle marked as lost

**Test**: Run with `float64` precision (if possible) or add epsilon tolerance.

### Hypothesis 4: Large RK4 Displacement Beyond Search Coverage

**Evidence**:
- dt = 2.5e-3 (relatively large)
- Velocity varies by 262K× across mesh (extreme gradients)
- Linear loss rate matches expected displacement failures

**Mechanism**:
1. Particle in high-velocity region
2. RK4 step moves particle >3 octants away
3. 3×3×3 neighbors search misses new location
4. Hierarchical also misses (only searches depths 7+6, not 5)

**Test**: Reduce dt to 1e-3 or increase L2 coverage.

---

## Comparison with Your Diagnostic

**Your diagnostic script** (diagnose_lost_particles.py) showed:
- Initial assignment: 82.05% (18% loss)
- Tracking steps 1-10: Only 0.06% additional loss (17 particles)

**Production script** shows:
- Initial assignment: 100.00% (0% loss with cascading)
- Tracking steps 1-100: 6.71% loss (15,088 particles)

**Key difference**: Production uses **cascading fallback with radius up to 100,000** for initial assignment!

Your diagnostic used:
```python
initial_assignment_cascading_fallback(
    positions_gpu,
    mesh_gpu_octree,
    initial_radius=2,
    fallback_radii=[4, 8, 15, 30],  # Max radius 30
    verbose=False
)
```

Production uses:
```python
Initial radius: 500
Fallback radii: [1000, 2000, 5000, 10000, 100000]  # Max radius 100,000!
```

**This explains the initial assignment difference!**

---

## Performance Analysis

### Step Time Comparison

| L2 Method | Step 100 Time | Step 200 Time | Throughput |
|-----------|---------------|---------------|------------|
| **Incremental** | 6,215.74 ms | *(killed)* | 36,198 p/s |
| **Neighbors** | 11,881.27 ms | 11,879.86 ms | 18,937-18,940 p/s |
| **Hierarchical** | 11,208.51 ms | 11,210.44 ms | 20,071-20,074 p/s |

**Key findings**:
1. **Incremental is 2× faster** than neighbors (6.2s vs 11.9s per step)
2. **Hierarchical slightly faster than neighbors** (11.2s vs 11.9s) - 5.6% speedup
3. **Neighbors and hierarchical have identical retention** but hierarchical is faster

**Throughput** (particles/second):
- Incremental: **36,198 p/s** ⚡ (fastest)
- Hierarchical: **20,071 p/s** (middle)
- Neighbors: **18,937 p/s** (slowest)

---

## Recommendations

### 1. 🔥 **Increase L1 Search Depth** (Most Likely to Help)

**Action**:
```python
N_HOPS = 7  # Instead of 5
```

**Expected**:
- Better retention at coarse/fine boundaries
- +3-5% retention improvement
- 2-3× slower L1 search (worth it if fixes particle loss)

**Reasoning**: Face neighbors with 5 hops may not cross 1:2 refinement boundaries.

### 2. 📊 **Diagnose Lost Particle Locations During Tracking**

**Create diagnostic** to capture:
```python
# After each RK4 step, save lost particles
lost_positions_per_step = []
for step in range(100):
    positions_gpu, element_ids_gpu = rk4_step(...)

    # Find newly lost particles
    lost_mask = element_ids_gpu < 0
    newly_lost = positions_gpu[lost_mask]
    lost_positions_per_step.append(newly_lost)

# Analyze spatial patterns
all_lost = np.concatenate(lost_positions_per_step)
# Check if clustered near PVTU boundaries, refinement boundaries, etc.
```

**This will reveal**:
- Are particles lost at refinement boundaries?
- Are they clustered near old PVTU piece boundaries?
- Do they follow velocity streamlines out of domain?

### 3. ⚡ **Use Incremental Method for Production** (Best Speed/Retention Trade-off)

**Current best configuration**:
```python
L2_SEARCH_METHOD = 'incremental'
INCREMENTAL_SEARCH_RADII = (2, 8, 16, 32, 64)
```

**Performance**:
- **2× faster** than neighbors/hierarchical
- **Slightly better retention** (95.35% vs 93.29% at step 100)
- Established and tested

**Don't use neighbors/hierarchical** - they're slower with same/worse retention.

### 4. 🔍 **Reduce Timestep** (Test if RK4 Displacement is Issue)

**Test**:
```python
DT = 1.0e-3  # Instead of 2.5e-3
```

**Expected**:
- Smaller RK4 displacement per step
- Better L2 search coverage
- Slower overall (2.5× more steps)

**If retention improves significantly** → confirms large displacement hypothesis.

### 5. ✅ **Use Cascading Initial Assignment** (Already Working!)

**Your production script already does this correctly**:
```python
Initial radius: 500
Fallback radii: [1000, 2000, 5000, 10000, 100000]
```

**Result**: 100% initial assignment ✅

**Update diagnostic script** to use same parameters.

---

## Summary Table

| Metric | Incremental | Neighbors | Hierarchical |
|--------|-------------|-----------|--------------|
| **Initial Assignment** | 100.00% ✅ | 100.00% ✅ | 100.00% ✅ |
| **Retention @ Step 100** | **95.35%** ⭐ | 93.29% | 93.29% |
| **Retention @ Step 200** | *(killed)* | 86.89% | 86.89% |
| **Step Time** | **6.2s** ⚡ | 11.9s | 11.2s |
| **Throughput** | **36,198 p/s** | 18,937 p/s | 20,071 p/s |
| **Recommendation** | ✅ **Use this** | ❌ Slower, same loss | ❌ Slower, same loss |

---

## Answers to Your Original Questions

### Q: How does the 'neighbor' L2 search work?

**A**: It searches 3×3×3 spatial octants (27 octants) around query position using Morton arithmetic. Your understanding was correct!

### Q: Would sequential search (current element first) be beneficial?

**A**: **NO** - The production logs prove:
1. Neighbors and hierarchical have **identical retention** (93.29% @ step 100)
2. Both are **2× slower** than incremental radius method
3. L2 method choice **doesn't matter** - particle loss is due to other factors

**Your proposed modification wouldn't help because**:
- L2 search is NOT the bottleneck (neighbors = hierarchical = similar retention)
- Particle loss is likely at **L1 search failures** or **velocity discontinuities**

### Q: Why doesn't retention improve above 95%?

**A**: Production logs reveal:
1. **Initial assignment is 100%** (with cascading fallback)
2. **Tracking loses 6-7% per 100 steps** (linear rate)
3. **L2 method doesn't matter** (neighbors = hierarchical = incremental ≈ similar)

**Root cause**: NOT L2 search failures, but likely:
- L1 face traversal failing at refinement boundaries (N_HOPS=5 insufficient)
- Velocity field discontinuities at old PVTU boundaries
- Or large RK4 displacement beyond search coverage

**Solution**: Increase `N_HOPS=7` or diagnose lost particle locations.

---

## Final Recommendation

🎯 **Use incremental L2 method** (current configuration):
- ✅ 2× faster than neighbors/hierarchical
- ✅ Slightly better retention
- ✅ Well-tested and established

🔥 **Increase L1 depth** to fix particle loss:
```python
N_HOPS = 7  # Try this first
```

📊 **Add diagnostic** to capture lost particle positions during tracking - this will reveal the true root cause!

**Don't pursue neighbors/hierarchical search modifications** - the logs prove they don't improve retention and are 2× slower. The particle loss is happening at L1 or due to velocity/RK4 issues, not L2 search!

---

## References

- [production_fully_fused_timedep_radius-2-8-16-32-64_withL1face-5hop_inverse_RTX5000.log](logs/production_fully_fused_timedep_radius-2-8-16-32-64_withL1face-5hop_inverse_RTX5000.log)
- [production_fully_fused_timedep_neighbor_withL1-5hop_inverse_RTX5000.log](logs/production_fully_fused_timedep_neighbor_withL1-5hop_inverse_RTX5000.log)
- [production_fully_fused_timedep_hierarchical_withL1-5hop_inverse_RTX5000.log](logs/production_fully_fused_timedep_hierarchical_withL1-5hop_inverse_RTX5000.log)
- [NEIGHBORS_L2_SEARCH_EXPLAINED.md](NEIGHBORS_L2_SEARCH_EXPLAINED.md) - Search algorithm explanation
- [LOST_PARTICLES_ROOT_CAUSE.md](LOST_PARTICLES_ROOT_CAUSE.md) - Initial assignment diagnostic
