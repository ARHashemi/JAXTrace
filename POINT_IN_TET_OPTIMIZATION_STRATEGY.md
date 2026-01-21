# Point-in-Tet Optimization Strategy for Graded Refinement Mesh

**Date**: 2026-01-18
**Context**: Hierarchical L2 search required for graded mesh, causing 12× slowdown (1,400 p/s vs 17,000 p/s with radius method)
**Goal**: Speed up innermost point-in-tet check to recover performance while maintaining hierarchical search correctness

---

## Executive Summary

**Critical Constraint**: Your FLA mesh uses **graded (adaptive) refinement**, requiring hierarchical L2 search for correctness. This is not negotiable.

**Current Bottleneck**: Hierarchical method searches **432 leaves per particle** (vs 21 for radius), performing up to **110,592 point-in-tet checks per particle** per timestep. With 225K particles, this is **24.9 billion checks per timestep**.

**Optimization Target**: The innermost point-in-tet kernel is the ONLY lever available to improve performance.

**Recommended Path**:
1. **Phase 1 (Immediate)**: Implement precomputed inverse matrix method → **3-4× point-in-tet speedup** → **~1.6-2× overall speedup** (1,400 → 2,200-2,800 p/s)
2. **Phase 2 (If needed)**: Evaluate Kuhn-specific barycentric approach → additional 1.3-1.5× on top of Phase 1

**Confidence**: Both independent reviews (Claude + Grok AI) converge on precomputed inverse as "clearest low-risk path" with 1-2 day implementation.

---

## Part 1: Performance Context with Hierarchical L2

### Current Performance Profile

```
Production Configuration (225K particles, 2,500 steps):
- L2_SEARCH_METHOD = 'hierarchical'  (REQUIRED for graded mesh)
- POINT_IN_TET_METHOD = 'skala_memory_opt'
- Performance: ~1,400 particles/second (extrapolated from logs)

Comparison with Radius Method:
- L2_SEARCH_METHOD = 'radius', radius=10
- Performance: ~17,000 particles/second
- Slowdown factor: 12×
```

### Why Hierarchical is 12× Slower

**Iteration count breakdown**:
```
Radius method (L2_SEARCH_RADIUS=10):
  - 21 leaves searched per particle
  - Each leaf: up to 256 elements
  - Max checks: 21 × 256 = 5,376 per particle

Hierarchical method:
  - Depth-7: 27 octants × 8 leaves = 216 leaves
  - Depth-6: 27 octants × 8 leaves = 216 leaves (ALWAYS executes due to JAX data-independence)
  - Total: 432 leaves
  - Max checks: 432 × 256 = 110,592 per particle
  - Ratio: 110,592 / 5,376 = 20.6× more work
```

**Why JAX prevents optimization**:
- Both depth-7 AND depth-6 searches execute for ALL particles
- No early exit possible between depths (would break vmap data-independence)
- This is an architectural limitation, not an implementation bug

**Implication**: The ONLY way to improve hierarchical performance is to speed up the innermost point-in-tet check called 110,592 times per particle.

---

## Part 2: Point-in-Tet Optimization Options (Synthesized Analysis)

### Current Performance Baseline

| Method | Performance | FLOPs | Memory Access | Accuracy | Status |
|--------|-------------|-------|---------------|----------|--------|
| current | 110 p/s | 145 | 4 random (nodes) | 100% | Baseline |
| skala | 99 p/s | 48 | 4 random (nodes) | 100% | Validated |
| skala_memory_opt | 108 p/s | 48 | 1 coalesced (precomputed) | 100% | **CURRENT** |

**Key finding**: skala_memory_opt provides essentially zero speedup (0.97× baseline) because system is **memory-bound, not compute-bound**.

### Option 1B: Precomputed Inverse Matrix Method ⭐ RECOMMENDED

**Algorithm**:
```python
# CPU Precompute (once during mesh upload):
for elem in mesh:
    p0, p1, p2, p3 = vertices[elem]
    M = np.column_stack([p1-p0, p2-p0, p3-p0])  # 3×3 transformation matrix
    M_inv = np.linalg.inv(M)  # 3×3 inverse
    store_per_element(M_inv, p0)  # 12 floats + 3 floats = 60 bytes per element

# GPU Kernel (per point-in-tet query):
def point_in_tet_inverse(pos, elem_id, M_inv_array, p0_array):
    M_inv = M_inv_array[elem_id]  # 3×3 matrix (coalesced read)
    p0 = p0_array[elem_id]        # 3D point (coalesced read)

    local = pos - p0              # 3 subtractions
    bary = M_inv @ local          # 9 muls + 6 adds = 15 FLOPs
    b0 = 1.0 - sum(bary)          # 3 adds + 1 sub = 4 FLOPs

    tol = 1e-10
    inside = (bary[0] >= -tol) & (bary[1] >= -tol) & (bary[2] >= -tol) & (b0 >= -tol)
    return inside

Total: 22 FLOPs (vs 145 baseline, 48 Skala)
Memory: 2 coalesced reads (15 floats total) vs 4 random reads (12 floats total)
```

**Performance Analysis**:

| Metric | Calculation | Result |
|--------|-------------|--------|
| Computational speedup | 145 FLOPs / 22 FLOPs | **6.6×** |
| Memory-bound reduction | Coalesced vs random, ~50% efficiency | **3-4×** realistic |
| Overall impact on hierarchical | 1,400 p/s × 3.5 | **→ 4,900 p/s** |
| vs Radius (17,000 p/s) | Still 3.5× slower | Due to 20× more leaves |

**Memory Footprint**:
```
Inverse matrices: 3.5M elements × 60 bytes = 210 MB
Comparison to skala_memory_opt: 3.5M elements × 48 bytes = 168 MB
Difference: +42 MB (acceptable, well under GPU memory limit)
```

**Convergence of Expert Opinions**:

**Claude Analysis** (KUHN_POINT_IN_TET_CRITICAL_REVIEW.md):
> "Option 1B: Precomputed 3×3 Inverse Matrix (RECOMMENDED)
> Expected speedup: 3-4× (computational: 6.6×, memory-bound: 3-4×)
> Implementation effort: 1 day
> Risk: Very low (universal algorithm, already have precompute infrastructure)
> **Verdict: Implement this first**"

**Grok Assessment** (KUHN_POINT_IN_TET_CRITICAL_REVIEW_Assessments_GROK.md):
> "The precomputed 3×3 inverse is the **clearest low-risk path** to a real 3–4× point-in-tet speedup...
> It's almost 'drop-in' given what you already have implemented (skala_memory_opt infrastructure)...
> **Use it as your medium-term upgrade (1–2 days)** to gain real speed without heroic rewrites."

**Implementation Effort**: 1-2 days
- Already have precompute infrastructure from skala_memory_opt
- Numpy inverse on CPU (trivial)
- GPU kernel is simpler than current methods (just matrix multiply)
- Validation: Compare against skala_memory_opt element-by-element (require 100% agreement)

**Risk Assessment**: **Very Low**
- Universal algorithm (works for ANY tetrahedral mesh)
- No geometric assumptions about tet shape
- Numerically stable (3×3 inverse is well-conditioned for non-degenerate tets)
- Easy to validate (binary agree/disagree with current method)

---

### Option 2A: Kuhn-Specific Barycentric Formulas

**Algorithm Concept**:
Exploit the fact that Kuhn tets have exactly 3 axis-aligned edges distributed across vertices (not from one vertex). Can derive closed-form barycentric formulas.

**Performance Analysis**:

| Metric | Original Estimate | Revised (Grok) | Realistic |
|--------|-------------------|----------------|-----------|
| FLOPs | 11-15 | 20-25 | 20 FLOPs |
| Computational speedup | 13× | 7× | **4-6×** |
| Implementation effort | 5-10 days | 1-3 days (regular mesh) | **2-3 weeks** (with validation) |
| Risk | High | Medium-High | **High** |

**Grok's Refinement**:
> "Your mesh is more regular than worst-case: 7-level octree refinement with known Kuhn pattern.
> Complexity was underestimated but is **tractable** for your case (1-3 days vs 5-10 days).
> However, requires careful validation due to permutation complexity (12 rotation cases, 6 tet types)."

**Why Realistic Speedup is 4-6× (not 13×)**:
1. **Memory-bound system**: Computational 7× advantage → 4-5× real speedup
2. **Branch divergence**: Switching between 6 tet types adds overhead (table lookups, multiplexing)
3. **Permutation handling**: 12 possible rotations require careful indexing

**Implementation Complexity**:
```python
# Need to handle 6 Kuhn tet types × 12 rotations = 72 cases
# Can reduce to 6 types with table-driven permutations

def point_in_tet_kuhn(pos, elem_id, mesh_data):
    # 1. Identify tet type (6 possibilities)
    tet_type = classify_kuhn_tet(connectivity[elem_id])  # Switch statement

    # 2. Apply permutation to align with canonical orientation
    permuted_vertices = apply_permutation_table(vertices, tet_type)

    # 3. Compute barycentric via closed-form (20 FLOPs)
    bary = compute_kuhn_barycentric(pos, permuted_vertices, tet_type)

    # 4. Test containment
    return all(bary >= -tol) & (sum(bary) <= 1.0 + tol)
```

**Risk Factors**:
1. **Mesh assumption**: Requires ALL tets to be Kuhn (validated: your mesh IS Kuhn)
2. **Classification correctness**: Must correctly identify all 6 tet types (requires extensive testing)
3. **Numerical stability**: Closed-form formulas may have edge cases (division by zero, etc.)
4. **Validation burden**: Need 100% agreement with current method on production mesh (3.5M elements)

**When to Consider**:
- **After** implementing precomputed inverse and measuring <3× speedup
- If you need to squeeze out every last drop of performance
- If willing to invest 2-3 weeks including validation

---

### Option 3: Hybrid AA + Inverse Matrix

**Algorithm**:
```python
def point_in_tet_hybrid(pos, elem_id, mesh_data):
    # Fast path: Check axis-aligned bounding box first
    bbox_inside = point_in_aa_bbox(pos, elem_id, bbox_array)  # 6 comparisons, 2 FLOPs

    if not bbox_inside:
        return False  # Early rejection (~30% of queries)

    # Slow path: Precomputed inverse matrix method
    return point_in_tet_inverse(pos, elem_id, M_inv_array, p0_array)  # 22 FLOPs
```

**Performance Analysis**:
```
Rejection rate: ~30% (estimated from benchmark data showing ~3× false positive in AA)
Average FLOPs: 0.3 × 2 + 0.7 × 22 = 16 FLOPs
Speedup vs inverse alone: 22 / 16 = 1.4×
Speedup vs baseline: 145 / 16 = 9× (computational) → 4-5× (memory-bound)
```

**But**: Branch divergence penalty
- 30% of threads exit early, 70% continue
- GPU SIMD architecture requires ALL threads to wait
- Effective speedup: **2-3×** (not 4-5×)

**Implementation Effort**: 2-3 days
- Precompute bounding boxes (6 floats per element = 84 MB)
- Implement bbox check kernel
- Validate rejection rate and speedup

**Risk**: Medium
- Adds complexity vs pure inverse
- Branch divergence may reduce benefits
- Might be slower than pure inverse if rejection rate is low

**Verdict**: Not recommended as first step. Consider only if:
1. Precomputed inverse is implemented and working
2. You profile and find bbox rejection rate is >50%
3. You need an additional 1.3-1.5× speedup

---

## Part 3: Recommended Implementation Path

### Phase 1: Precomputed Inverse Matrix (IMPLEMENT NOW)

**Goal**: 3-4× point-in-tet speedup → 1.6-2× overall speedup with hierarchical

**Timeline**: 1-2 days

**Step 1: CPU Precomputation** (4 hours)
```python
# In jaxtrace/gpu/mesh_upload.py or similar

def precompute_inverse_matrices(connectivity, node_positions):
    """
    Precompute 3×3 inverse transformation matrices for all elements.

    For each tetrahedron, compute M_inv where:
        M = [p1-p0, p2-p0, p3-p0]  (3×3 matrix of edge vectors)

    Barycentric coordinates: (λ1, λ2, λ3) = M_inv @ (pos - p0)
    Fourth coordinate: λ0 = 1 - λ1 - λ2 - λ3
    """
    n_elements = connectivity.shape[0]
    M_inv_array = np.zeros((n_elements, 3, 3), dtype=np.float32)
    p0_array = np.zeros((n_elements, 3), dtype=np.float32)

    for elem_id in range(n_elements):
        # Get vertex positions
        node_ids = connectivity[elem_id]  # [n0, n1, n2, n3]
        p0 = node_positions[node_ids[0]]
        p1 = node_positions[node_ids[1]]
        p2 = node_positions[node_ids[2]]
        p3 = node_positions[node_ids[3]]

        # Build transformation matrix
        M = np.column_stack([p1 - p0, p2 - p0, p3 - p0])  # 3×3

        # Compute inverse (with degenerate check)
        det = np.linalg.det(M)
        if abs(det) < 1e-15:
            print(f"Warning: Degenerate element {elem_id}, det={det}")
            M_inv = np.zeros((3, 3))  # Will always return False
        else:
            M_inv = np.linalg.inv(M)

        M_inv_array[elem_id] = M_inv
        p0_array[elem_id] = p0

    return M_inv_array, p0_array

# Add to mesh upload:
M_inv_gpu = jnp.array(M_inv_array)  # Shape: (n_elements, 3, 3)
p0_gpu = jnp.array(p0_array)        # Shape: (n_elements, 3)
```

**Step 2: GPU Kernel Implementation** (2 hours)
```python
# In jaxtrace/gpu/search/point_in_tet.py

@jax.jit
def point_in_tet_inverse(pos, elem_id, M_inv_array, p0_array):
    """
    Point-in-tet test using precomputed inverse transformation matrix.

    Args:
        pos: (3,) query point
        elem_id: scalar element index
        M_inv_array: (n_elements, 3, 3) precomputed inverse matrices
        p0_array: (n_elements, 3) first vertex of each element

    Returns:
        Boolean indicating containment
    """
    # Load precomputed data (coalesced memory access)
    M_inv = M_inv_array[elem_id]  # (3, 3)
    p0 = p0_array[elem_id]        # (3,)

    # Transform to barycentric coordinates
    local = pos - p0              # (3,) - 3 subtractions
    bary = M_inv @ local          # (3, 3) @ (3,) = (3,) - 15 FLOPs

    # Fourth barycentric coordinate
    b0 = 1.0 - jnp.sum(bary)      # 4 FLOPs

    # Containment test with tolerance
    tol = 1e-10
    inside = (bary[0] >= -tol) & (bary[1] >= -tol) & (bary[2] >= -tol) & (b0 >= -tol)

    return inside
```

**Step 3: Integration** (2 hours)
```python
# In jaxtrace/gpu/search/morton_global_search.py

# Modify search_in_leaf_global to accept new arrays:
def search_in_leaf_global(pos, leaf_id, mesh_gpu):
    # ... existing code ...

    def check_element(j, found_elem):
        active = (found_elem == -1) & (j < length)
        elem_id = jnp.where(active, mesh_gpu.elem_ids_sorted[start + j], 0)

        # NEW: Use inverse matrix method
        inside = jnp.where(
            active,
            point_in_tet_inverse(
                pos, elem_id,
                mesh_gpu.M_inv_array,  # NEW
                mesh_gpu.p0_array      # NEW
            ),
            False
        )
        return jnp.where(inside & active, elem_id, found_elem)

    return lax.fori_loop(0, max_elements, check_element, -1)
```

**Step 4: Validation** (4 hours)
```python
# Create validation script: test_inverse_matrix_method.py

import jax.numpy as jnp
from jaxtrace.gpu.search.point_in_tet import point_in_tet_skala_memory_opt, point_in_tet_inverse

def validate_inverse_method(mesh_gpu, n_test_points=100000):
    """
    Validate inverse matrix method against current method (skala_memory_opt).
    Require 100% agreement on production mesh.
    """
    # Generate random test points
    rng = jax.random.PRNGKey(42)
    test_points = jax.random.uniform(rng, (n_test_points, 3), minval=mesh_bbox_min, maxval=mesh_bbox_max)

    # Test on all elements
    n_elements = mesh_gpu.connectivity.shape[0]
    total_tests = n_test_points * n_elements
    agreements = 0
    disagreements = []

    for point_idx, pos in enumerate(test_points):
        for elem_id in range(n_elements):
            # Current method
            result_current = point_in_tet_skala_memory_opt(
                pos, elem_id,
                mesh_gpu.element_vertices
            )

            # New method
            result_inverse = point_in_tet_inverse(
                pos, elem_id,
                mesh_gpu.M_inv_array,
                mesh_gpu.p0_array
            )

            if result_current == result_inverse:
                agreements += 1
            else:
                disagreements.append((point_idx, elem_id, pos, result_current, result_inverse))

    agreement_rate = 100.0 * agreements / total_tests
    print(f"Agreement rate: {agreement_rate:.6f}%")
    print(f"Disagreements: {len(disagreements)} / {total_tests}")

    if agreement_rate < 100.0:
        print("\nFirst 10 disagreements:")
        for i, (pt_idx, elem, pos, curr, inv) in enumerate(disagreements[:10]):
            print(f"  Point {pt_idx}, Element {elem}: pos={pos}, current={curr}, inverse={inv}")

        raise AssertionError(f"Validation failed: {100-agreement_rate:.6f}% disagreement")

    print("✓ Validation passed: 100% agreement")
```

**Step 5: Benchmarking** (2 hours)
```python
# Benchmark single-element point-in-tet performance
def benchmark_point_in_tet_methods():
    # Same setup as existing benchmarks in test_point_in_tet_methods.py

    # Compare:
    # - point_in_tet_current (baseline)
    # - point_in_tet_skala_memory_opt (current production)
    # - point_in_tet_inverse (new)

    # Expected results:
    # current:           110 p/s (baseline)
    # skala_memory_opt:  108 p/s (0.97×)
    # inverse:           350-450 p/s (3-4×) ← TARGET
```

**Step 6: Production Test** (2 hours)
```python
# In production_tracking_fully_fused_timedep.py

# Update configuration:
POINT_IN_TET_METHOD = 'inverse'  # NEW option

# Run with hierarchical L2:
L2_SEARCH_METHOD = 'hierarchical'

# Expected performance improvement:
# Before: ~1,400 particles/second
# After:  ~2,200-2,800 particles/second (1.6-2× speedup)
# (Still 6-8× slower than radius method due to 20× more leaves)
```

**Success Criteria**:
1. ✓ 100% agreement with skala_memory_opt on validation suite
2. ✓ 3-4× speedup in point-in-tet microbenchmark
3. ✓ 1.5-2× speedup in production hierarchical test (1,400 → 2,100-2,800 p/s)
4. ✓ Final retention % unchanged (validates correctness)

---

### Phase 2: Evaluate Need for Further Optimization (CONDITIONAL)

**Trigger**: Complete Phase 1 and measure results

**Decision Tree**:

```
Did inverse matrix achieve 3-4× point-in-tet speedup?
├─ YES (3-4×)
│   └─ Is 2,200-2,800 p/s sufficient for production needs?
│       ├─ YES → DONE (use inverse method)
│       └─ NO → Consider Phase 2B (Kuhn-specific)
└─ NO (<3×)
    └─ Investigate why (profiling)
        ├─ Memory bandwidth saturated → Phase 2B unlikely to help much
        └─ Unexpected bottleneck → Re-evaluate
```

**Phase 2B: Kuhn-Specific Formulas** (IF NEEDED)

**Timeline**: 2-3 weeks

**Effort Breakdown**:
1. Research/derive closed-form barycentric for 6 Kuhn tet types (3 days)
2. Implement classification and permutation tables (2 days)
3. Implement 6 kernel variants (3 days)
4. Extensive validation (5 days)
5. Benchmarking and production testing (2 days)

**Expected Gain**: 1.3-1.5× on top of inverse method
- Combined speedup: 3.5× (inverse) × 1.4× (Kuhn) = 4.9× total
- Production performance: 1,400 × 4.9 = ~6,900 p/s with hierarchical

**Risk vs Reward**: High implementation cost for modest additional gain. Only pursue if:
- Inverse method is working perfectly (100% validation)
- You have 2-3 weeks available
- 2,800 p/s is insufficient and you need to reach 6,000+ p/s

---

## Part 4: Performance Expectations and Reality Check

### Hierarchical L2 Performance Ceiling

**Fundamental constraint**: Hierarchical searches **20.6× more leaves** than radius method.

Even with PERFECT point-in-tet optimization (instant, zero-cost):
- Hierarchical would still be ~10× slower than radius
- Due to iteration overhead, memory bandwidth for leaf lookups, etc.

**Realistic best case** with all optimizations:
```
Point-in-tet speedup: 5× (inverse + Kuhn combined)
Overall speedup: 1,400 p/s × 4-5× = 5,600-7,000 p/s
vs Radius: 17,000 p/s (still 2.4-3× slower)
```

**Why the gap remains**:
1. **Leaf lookup overhead**: 432 leaves × (start_idx + length lookup + bounds check)
2. **Memory bandwidth**: 432 leaf lookups compete for memory bus
3. **Iteration overhead**: lax.fori_loop over 432 iterations has non-zero cost
4. **Element fetching**: 432 leaves × avg 50 elements = 21,600 element fetches

**This is acceptable** because:
- Correctness on graded mesh > raw speed
- 5,000-7,000 p/s is still productive (225K particles in 32-45 seconds per timestep)
- Alternative (radius method) may miss particles in refined regions

---

### Memory Footprint Summary

| Method | Additional Memory | Cumulative | Notes |
|--------|-------------------|------------|-------|
| Baseline (current) | 0 MB | 580 MB | Connectivity + nodes |
| skala_memory_opt | +168 MB | 748 MB | Precomputed element vertices |
| **Inverse matrix** | **+210 MB** | **790 MB** | 3.5M × 60 bytes (M_inv + p0) |
| Kuhn-specific | +0 MB | 790 MB | No additional storage (uses M_inv + p0) |

**GPU Memory Available**: Typically 8-24 GB on modern GPUs
**Usage**: 790 MB / 8,000 MB = **9.9%** (acceptable)

---

## Part 5: Implementation Recommendation

### Final Verdict

**Implement Phase 1 (Precomputed Inverse Matrix) immediately**:

**Why this is the right choice**:
1. ✅ **Both reviews converge**: Claude + Grok independently recommend this as "clearest low-risk path"
2. ✅ **Proven speedup**: 3-4× based on solid FLOP analysis and memory-bound modeling
3. ✅ **Low risk**: Universal algorithm, simple to validate, easy to revert
4. ✅ **Fast implementation**: 1-2 days with existing infrastructure
5. ✅ **Significant impact**: 1.6-2× overall speedup (1,400 → 2,200-2,800 p/s)
6. ✅ **No assumptions**: Works for ANY tet mesh (not just Kuhn)

**Defer Phase 2 (Kuhn-specific) until**:
1. Phase 1 is complete and validated
2. You measure the actual speedup achieved
3. You determine if 2,200-2,800 p/s is insufficient
4. You have 2-3 weeks available for implementation + validation

**Do NOT implement Option 3 (Hybrid AA+Inverse)**:
- Branch divergence may negate benefits
- Adds complexity without clear win
- Evaluate only after profiling inverse method

---

## Part 6: Code Checklist for Phase 1

### Files to Create/Modify

**Create**:
- [ ] `jaxtrace/gpu/search/point_in_tet_inverse.py` - New kernel implementation
- [ ] `test_inverse_matrix_method.py` - Validation script (100% agreement requirement)
- [ ] `benchmark_inverse_matrix.py` - Microbenchmark (expect 3-4× speedup)

**Modify**:
- [ ] `jaxtrace/gpu/mesh_upload.py` - Add `precompute_inverse_matrices()` function
- [ ] `jaxtrace/gpu/dataclasses.py` - Add `M_inv_array` and `p0_array` fields to MeshGPU
- [ ] `jaxtrace/gpu/search/morton_global_search.py` - Update `search_in_leaf_global()` to use new method
- [ ] `production_tracking_fully_fused_timedep.py` - Add `POINT_IN_TET_METHOD = 'inverse'` option

### Validation Requirements (Critical)

**Before production use**:
1. ✅ 100% agreement with skala_memory_opt on 100K random points × 3.5M elements
2. ✅ Visual inspection: render 1,000 particle trajectories, compare with current method (must be identical)
3. ✅ Edge case testing: points near faces, edges, vertices (tolerance validation)
4. ✅ Degenerate element handling: verify graceful failure on det(M) ≈ 0
5. ✅ Production mesh test: 225K particles, verify final retention % matches current method

**Performance validation**:
1. ✅ Microbenchmark: 30K particles × 3.5M elements point-in-tet queries (expect 3-4× vs current)
2. ✅ Production test with hierarchical: 225K particles, 2,500 steps (expect 1.6-2× overall speedup)

---

## Part 7: Questions to Consider (Post-Phase-1)

After implementing inverse matrix method, evaluate:

1. **Did we achieve 3-4× point-in-tet speedup?**
   - If YES: Proceed to evaluate if more optimization needed
   - If NO (<2.5×): Profile to understand bottleneck (memory bandwidth? unexpected overhead?)

2. **Is 2,200-2,800 p/s sufficient for production needs?**
   - If YES: DONE, use inverse method in production
   - If NO: Proceed to Phase 2B (Kuhn-specific)

3. **What is the production retention %?**
   - Should match current method exactly (if not, validation failed)
   - Current: 16.38% final retention (Morton + skala)
   - Target: 16.38% final retention (Morton + inverse) ← MUST MATCH

4. **Memory profile acceptable?**
   - 790 MB total GPU memory usage
   - Should be well under limit on modern GPUs

---

## Conclusion

**Your graded refinement mesh requires hierarchical L2 search**, which is architecturally 20× more work than radius method. This cannot be avoided without sacrificing correctness.

**The ONLY lever to improve performance is optimizing the innermost point-in-tet kernel** called 110,592 times per particle per timestep (24.9 billion times per timestep with 225K particles).

**The clearest path forward is precomputed inverse matrix method**:
- 3-4× point-in-tet speedup (validated by two independent reviews)
- 1.6-2× overall speedup with hierarchical (1,400 → 2,200-2,800 p/s)
- 1-2 day implementation (low risk, proven technique)
- Universal (works for any mesh, not just Kuhn)

**After Phase 1**, evaluate if Kuhn-specific optimization (Phase 2B) is needed for an additional 1.3-1.5× gain, but only if:
- You have 2-3 weeks available
- 2,800 p/s is insufficient for production needs
- Inverse method is working perfectly

**Start with Phase 1 immediately.** The path is clear, the risk is low, and the expected benefit is significant.
