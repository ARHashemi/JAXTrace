# Retention Loss: Diagnosis and Recommendations

**Date**: 2025-12-31
**Current Status**: 82.45% retention @ step 100 (0.8% loss per 100 steps)
**Goal**: >95% retention @ step 100

---

## Current Implementation Analysis

### Point-in-Tet Tolerance
- **Fixed tolerance**: `-1e-6` ([morton_global_search.py:445](jaxtrace/gpu/search/morton_global_search.py#L445))
- **Smallest element**: 43.3 µm → tolerance is **2.3% of element size** (too large!)
- **Largest element**: 2.77 mm → tolerance is **0.036% of element size** (appropriate)
- **Size ratio**: 262,146× (extreme!)

### Already Implemented ✅
- Relative degeneracy check (lines 412-420) - Good!
- Adaptive L1 hop count (Phase 1.3) - Working!

---

## Recommended Approach

### Step 1: Run Quick Diagnostic (1 hour)

**Purpose**: Identify which factor is dominant

**Action**:
```bash
python diagnose_retention_loss_factors.py
```

**Expected output**:
```
Precision losses:  ~50 particles (~5.0%)
Morton losses:     ~10 particles (~1.0%)

⚠️  PRECISION IS THE DOMINANT FACTOR
```

**Decision tree**:
- If precision losses > 5% → Implement adaptive tolerance
- If Morton losses > 5% → Investigate Morton vs Hilbert
- If both < 2% → Look for other factors (time-dependent mesh issues)

---

### Step 2A: Adaptive Tolerance (if precision is dominant)

**Implementation time**: 4 hours
**Expected gain**: +5-10% retention

#### Modify `point_in_tet_gpu`

**File**: [`jaxtrace/gpu/search/morton_global_search.py`](jaxtrace/gpu/search/morton_global_search.py#L365)

**Changes**:

1. Add `element_volumes` parameter (already available on GPU!)

2. Compute adaptive tolerance:
```python
def point_in_tet_gpu(
    pos: jax.Array,
    elem_id: jnp.int32,
    connectivity: jax.Array,
    node_positions: jax.Array,
    element_volumes: jax.Array  # NEW
) -> jnp.bool_:
    # ... existing code ...

    # ADAPTIVE TOLERANCE (replace line 445)
    # Scale tolerance by element characteristic length
    vol = element_volumes[elem_id]
    char_length = vol ** (1.0/3.0)

    # Tolerance: 0.1% of characteristic length, normalized by edge length
    edge_length = jnp.sqrt(jnp.sum(v1 * v1))  # First edge length
    tol = -(char_length * 1e-3) / edge_length  # Relative to normalized coords

    # Clamp to reasonable range
    tol = jnp.clip(tol, -1e-4, -1e-8)  # For small: -1e-4, for large: -1e-8

    inside = (b0 >= tol) & (b1 >= tol) & (b2 >= tol) & (b3 >= tol) & (~is_degenerate)
    return inside
```

3. Update all call sites to pass `element_volumes`:
   - `search_in_leaf_global` (line 502)
   - All L1 neighbor search calls
   - Initial assignment functions

**Expected results**:
- Small elements (43 µm): `tol ≈ -4.3e-7` (1% of element, vs current 2.3%)
- Large elements (2.77 mm): `tol ≈ -2.77e-8` (0.001% of element, vs current 0.036%)
- Better boundary detection for small elements
- No change for large elements

---

### Step 2B: Morton Locality Fix (if Morton is dominant)

**Implementation time**: 8 hours
**Expected gain**: +5-10% retention

#### Option B1: Enhanced Morton Neighbor Search (Recommended)

**Simpler than Hilbert, targets specific failure case**

**Problem**: Morton neighbors at octree boundaries miss spatial neighbors

**Solution**: Multi-radius search at boundaries

```python
def search_L2_morton_neighbors_enhanced(pos, mesh_gpu_morton):
    """Enhanced Morton neighbor search with boundary handling."""

    # Standard 27-neighbor search
    elem_id = search_L2_morton_neighbors_single(pos, mesh_gpu_morton)

    if elem_id >= 0:
        return elem_id  # Success

    # FALLBACK: Particle near octree boundary?
    # Test with larger radius (54 neighbors: ±1 and ±2 cells)
    leaf_id = position_to_leaf_id_octree(pos, mesh_gpu_morton)

    # Search leaves in 5×5×5 neighborhood (125 cells)
    # Only activated when 27-neighbor search fails
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            for dz in range(-2, 3):
                # Skip already-tested 3×3×3 core
                if abs(dx) <= 1 and abs(dy) <= 1 and abs(dz) <= 1:
                    continue

                neighbor_leaf = get_neighbor_leaf_offset(leaf_id, dx, dy, dz)
                elem_id = search_in_leaf_global(pos, neighbor_leaf, mesh_gpu_morton)

                if elem_id >= 0:
                    return elem_id

    return jnp.int32(-1)  # Still not found
```

**Cost**: ~4× slower ONLY for particles that fail 27-neighbor search (~10-20%)
**Overall impact**: ~1.5× slower (acceptable)

---

#### Option B2: Hilbert Curve (High effort, uncertain benefit)

**Implementation time**: 40 hours
**Expected gain**: +5-15% retention (uncertain)
**Performance cost**: 2× slower neighbor arithmetic

**Only pursue if**:
- Diagnostic confirms Morton is the dominant issue
- Enhanced Morton search (B1) is insufficient

**Why not recommended**:
- 40 hours vs 8 hours for enhanced Morton
- Complex algorithm (high bug risk)
- Performance penalty
- Unclear if better than enhanced Morton for your case

---

### Step 2C: Coordinate Normalization (lowest risk, quick test)

**Implementation time**: 2 hours
**Expected gain**: +2-5% retention (if precision is the issue)

**Purpose**: Reduce catastrophic cancellation in `p - v0` subtractions

**Implementation**:

1. Normalize mesh coordinates to [0, 1]:
```python
# In mesh loader
mesh_min = node_positions.min(axis=0)
mesh_max = node_positions.max(axis=0)
mesh_extent = mesh_max - mesh_min

node_positions_normalized = (node_positions - mesh_min) / mesh_extent

# Scale velocities
velocities_normalized = velocities_physical / mesh_extent[np.newaxis, np.newaxis, :]
```

2. Update particle seeding to use normalized coordinates

3. Run test

**Pros**:
- 2 hours implementation
- Guaranteed precision improvement
- Easy to revert

**Cons**:
- Need to track normalization transform
- May not be sufficient alone (combine with adaptive tolerance)

---

## Recommended Implementation Order

### Priority 1: Quick Diagnostic (1 hour) ✅ DO FIRST
```bash
python diagnose_retention_loss_factors.py
```

Identifies which factor to target.

---

### Priority 2: Based on Diagnostic Results

#### If Precision > 5%:
1. **Coordinate Normalization** (2 hours) - Quick win
2. **Adaptive Tolerance** (4 hours) - Main fix
3. **Test**: Expect +7-15% retention

#### If Morton > 5%:
1. **Enhanced Morton Search** (8 hours) - Boundary handling
2. **Test**: Expect +5-10% retention
3. **If insufficient**: Consider Hilbert (40 hours)

#### If Both < 2%:
- Investigate other factors:
  - Time-dependent mesh topology changes
  - Velocity interpolation errors
  - RK4 intermediate stage overshooting

---

## Expected Final Results

### Conservative Estimate (Precision fix only):
```
Current:  82.45% @ step 100
+Norm:    85.00% @ step 100 (+2.55%)
+Adaptive: 90.00% @ step 100 (+5.00%)
Total:     90.00% @ step 100
```

### Optimistic Estimate (Precision + Morton):
```
Current:   82.45% @ step 100
+Norm:     85.00% @ step 100 (+2.55%)
+Adaptive: 91.00% @ step 100 (+6.00%)
+Morton:   95.00% @ step 100 (+4.00%)
Total:     95.00% @ step 100 ✅ TARGET
```

---

## Implementation Checklist

### Phase 1: Diagnostic
- [ ] Run `diagnose_retention_loss_factors.py`
- [ ] Analyze precision loss rate
- [ ] Analyze Morton loss rate
- [ ] Choose implementation path

### Phase 2A: Precision Fix (if needed)
- [ ] Implement coordinate normalization
- [ ] Test with production script (500 steps)
- [ ] If retention < 90%: Implement adaptive tolerance
- [ ] Modify `point_in_tet_gpu` to accept `element_volumes`
- [ ] Add adaptive tolerance computation
- [ ] Update all call sites
- [ ] Test with production script (500 steps)

### Phase 2B: Morton Fix (if needed)
- [ ] Implement enhanced Morton neighbor search
- [ ] Add 5×5×5 fallback for boundary particles
- [ ] Test with production script (500 steps)
- [ ] If retention < 95%: Consider Hilbert (schedule 40 hours)

### Phase 3: Validation
- [ ] Run full 2,500-step test
- [ ] Verify retention > 95% @ step 100
- [ ] Verify throughput > 6,000 p/s
- [ ] Document final results

---

## Hilbert vs Morton: Detailed Comparison

### When Hilbert is Worth It
✅ Neighbor locality critical (proven by diagnostic Morton losses > 5%)
✅ Willing to accept 2× performance penalty
✅ Have 40 hours for implementation + debugging

### When to Stick with Morton
✅ Enhanced Morton (5×5×5 search) is sufficient
✅ Precision is the dominant issue (not locality)
✅ Performance is critical (can't afford 2× slowdown)

**My recommendation**: Enhanced Morton is 80% of Hilbert's benefit for 20% of the effort.

---

## Next Steps

1. **Run diagnostic** (do this first!):
   ```bash
   python diagnose_retention_loss_factors.py
   ```

2. **Share results** - I'll provide specific implementation guidance

3. **Implement highest-impact solution** based on diagnostic

---

**Status**: Analysis complete. Diagnostic script ready. Awaiting your test results to confirm which factor is dominant.
