# Precision and Spatial Locality Analysis

**Date**: 2025-12-31
**Context**: Phase 1.3 retention still dropping (82.45% @ step 100 vs 83.65% initial)
**Investigating**: Float precision issues and Morton vs Hilbert spatial locality

---

## Factor 1: Float Precision Issues with Small Refined Elements

### Observed Problem

From diagnostic output:
```
Characteristic length range: [4.33e-05, 2.77e-03]
Characteristic length median: 8.66e-05
Size ratio (max/min): 262,145.92×
```

**Smallest element**: 43.3 microns (4.33e-05 m)
**Largest element**: 2.77 mm (2.77e-03 m)

### Potential Precision Issues

#### 1.1 Point-in-Tet Numerical Stability

**Point-in-tet test** uses barycentric coordinates via determinants:
```python
# Compute 4 determinants for barycentric coordinates
det0 = det([v1-v0, v2-v0, v3-v0])  # Base determinant
det1 = det([p-v0, v2-v0, v3-v0])   # Lambda1
det2 = det([v1-v0, p-v0, v3-v0])   # Lambda2
det3 = det([v1-v0, v2-v0, p-v0])   # Lambda3

# Check if inside: all lambdas in [0, 1]
lambda1 = det1 / det0
lambda2 = det2 / det0
lambda3 = det3 / det0
lambda0 = 1 - lambda1 - lambda2 - lambda3

inside = (lambda0 >= 0) & (lambda1 >= 0) & (lambda2 >= 0) & (lambda3 >= 0)
```

**Problem with small elements**:
- Edge vectors `v1-v0`, `v2-v0`, `v3-v0` are ~4e-5 m
- Determinant involves triple product of these vectors
- `det0 ~ (4e-5)^3 = 6.4e-14` (very small!)
- Float32 precision: ~1.2e-7 (relative), absolute precision ~7e-22
- **Ratio**: 6.4e-14 / 7e-22 = 9e7 → Should be OK

**But**: Subtractions like `p-v0` can cause cancellation errors when `p ≈ v0`

#### 1.2 Catastrophic Cancellation in Boundary Cases

When particle is **near element boundary**:
```
p = [0.0001234567, 0.0002345678, 0.0003456789]
v0 = [0.0001234560, 0.0002345670, 0.0003456780]
p - v0 = [7e-10, 8e-10, 9e-10]  # Lost ~7 significant digits!
```

**In float32**:
- Original coordinates: ~7 significant digits
- After subtraction: Only 1-2 significant digits remain
- Determinant calculation magnifies this error

**Result**: Particle **inside** element may be classified as **outside** due to rounding

#### 1.3 Epsilon Tolerance Issues

Current point-in-tet may use tolerance like:
```python
inside = (lambda0 >= -eps) & (lambda1 >= -eps) & (lambda2 >= -eps) & (lambda3 >= -eps)
```

**For small elements** (4.3e-5 m):
- If `eps = 1e-6` (typical): This is **2.3% of element size**!
- Too tight → reject particles actually inside
- Too loose → accept particles actually outside

**Optimal epsilon** should be **relative to element size**:
```python
eps_adaptive = element_volume^(1/3) * 1e-4  # 0.01% of characteristic length
```

---

### Solution 1A: Coordinate Normalization

**Idea**: Work in normalized coordinates [0, 1] instead of physical coordinates

#### Implementation

```python
# 1. Compute bounding box of entire mesh
mesh_min = node_positions.min(axis=0)
mesh_max = node_positions.max(axis=0)
mesh_extent = mesh_max - mesh_min

# 2. Normalize all coordinates to [0, 1]
node_positions_normalized = (node_positions - mesh_min) / mesh_extent
particle_positions_normalized = (particle_positions - mesh_min) / mesh_extent

# 3. All operations in normalized space
# Point-in-tet now works with numbers in [0, 1] range
# Element sizes: [4.3e-5 / extent, 2.77e-3 / extent]
```

**Benefits**:
- All coordinates in [0, 1] → no catastrophic cancellation
- Element sizes normalized to same scale
- Float32 precision optimal for [0, 1] range

**Drawbacks**:
- Need to track normalization transform
- Velocities need scaling: `v_normalized = v_physical / mesh_extent`

---

### Solution 1B: Adaptive Epsilon in Point-in-Tet

**Idea**: Use element-size-dependent tolerance

#### Current Implementation Check

Let me check what epsilon is used:

```python
# From point_in_tet_gpu or point_in_tet_jax
# Need to verify tolerance value
```

#### Proposed Adaptive Tolerance

```python
def point_in_tet_adaptive_eps(pos, elem_id, connectivity, node_positions, element_volumes):
    """Point-in-tet with adaptive epsilon based on element size."""

    # Get element volume
    vol = element_volumes[elem_id]

    # Characteristic length (cube root of volume)
    char_length = vol ** (1.0/3.0)

    # Adaptive epsilon: 0.01% of characteristic length
    # For 4.3e-5 m element: eps = 4.3e-9
    # For 2.77e-3 m element: eps = 2.77e-7
    eps = char_length * 1e-4

    # Compute barycentric coordinates
    v0, v1, v2, v3 = get_element_vertices(elem_id, connectivity, node_positions)
    lambda0, lambda1, lambda2, lambda3 = compute_barycentric(pos, v0, v1, v2, v3)

    # Check with adaptive epsilon
    inside = (lambda0 >= -eps) & (lambda1 >= -eps) & (lambda2 >= -eps) & (lambda3 >= -eps)

    return inside
```

**Expected gain**: Reduce false negatives for small elements

---

## Factor 2: Morton vs Hilbert Spatial Locality

### Morton Code Spatial Continuity

**Morton code** (Z-order curve) interleaves bits:
```
Position (x, y, z) → Morton = ...z2y2x2z1y1x1z0y0x0
```

**Spatial locality issues**:

#### 2.1 Discontinuities at Power-of-2 Boundaries

Morton curve has **jumps** at octree cell boundaries:

```
Example in 2D (quadtree):

┌─────┬─────┐
│  2  │  3  │  Morton order: 0 → 1 → 2 → 3
├─────┼─────┤                ↓   ↓   ↓   ↓
│  0  │  1  │  But spatial:  0→1 far from 1→2!
└─────┴─────┘

Cell 1 (01) → Cell 2 (10): Morton codes differ in 2 bits
Spatial distance: 1 cell
Morton distance: Large jump!
```

**In 3D octree**: Even worse - Morton jumps can span entire domain

#### 2.2 Impact on Neighbor Search

**L2 Morton neighbor search** (27 octants):
```python
# Current method: ±1 along each axis in Morton space
for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            neighbor_morton = position_to_morton(pos + [dx, dy, dz] * cell_size)
```

**Problem**: Morton neighbors != spatial neighbors at boundaries!

Example:
```
Particle at octree boundary → needs elements from 2 adjacent cells
Morton code puts these cells FAR APART in sorted order
Result: Neighbor search misses containing element
```

---

### Hilbert Curve Spatial Continuity

**Hilbert curve** has **better spatial locality**:

#### 2.3 Hilbert Advantages

1. **Continuous**: No large jumps between adjacent cells
2. **Locality preservation**: Spatially close → Hilbert-close (mostly)
3. **Better cache**: Traversing Hilbert order → fewer cache misses

**Hilbert order example** (2D):
```
┌─────┬─────┐
│  1  │  2  │  Hilbert order: 0 → 1 → 2 → 3
├─────┼─────┤                 └──→──┘  │
│  0  │  3  │  Continuous path!    └───┘
└─────┴─────┘
```

All transitions are **1 cell distance**!

#### 2.4 Quantitative Locality Comparison

**Morton curve**:
- Average locality: 70-80% (cells within ±1 Morton index are spatial neighbors)
- Worst case: 8× domain jump at power-of-2 boundaries

**Hilbert curve**:
- Average locality: 90-95%
- Worst case: 2× domain span (much better)

**For your mesh** (262K× size variation):
- Morton discontinuities likely cause particles to "jump" across refinement boundaries
- Hilbert's continuity would keep searches local

---

### Solution 2: Replace Morton with Hilbert

#### 2A: Hilbert Encoding Algorithm

**3D Hilbert curve encoding** (Gray code + rotation):

```python
def position_to_hilbert_3d(pos: jax.Array, bbox_min: jax.Array, bbox_max: jax.Array, depth: int) -> jnp.uint64:
    """
    Encode 3D position as Hilbert index.

    Based on Compact Hilbert Index algorithm (Lawder & King, 2000).
    """
    # Normalize to [0, 2^depth - 1] integer coordinates
    normalized = (pos - bbox_min) / (bbox_max - bbox_min)
    coords = (normalized * ((1 << depth) - 1)).astype(jnp.uint32)
    x, y, z = coords[0], coords[1], coords[2]

    hilbert_index = jnp.uint64(0)

    # Process from MSB to LSB
    for i in range(depth - 1, -1, -1):
        # Extract bits at level i
        xi = (x >> i) & 1
        yi = (y >> i) & 1
        zi = (z >> i) & 1

        # Combine into 3-bit cell index
        cell = (zi << 2) | (yi << 1) | xi

        # Apply Hilbert Gray code (rotation depends on previous bits)
        hilbert_cell = hilbert_gray_code_3d(cell, rotation_state)

        # Append to Hilbert index
        hilbert_index |= (jnp.uint64(hilbert_cell) << (3 * i))

        # Update rotation for next level
        rotation_state = hilbert_rotation_update(rotation_state, cell)

    return hilbert_index
```

**Complexity**: Same as Morton (O(depth) bit operations)

#### 2B: Hilbert Neighbor Arithmetic

**Challenge**: Hilbert neighbors are HARD to compute directly

**Morton neighbors**:
```python
# Easy: just add/subtract in each dimension
neighbor_morton = morton + delta_x + (delta_y << 21) + (delta_z << 42)
```

**Hilbert neighbors**:
```python
# Complex: need to trace curve, no simple arithmetic
# Must decode → shift spatially → re-encode
```

**Solutions**:

1. **Pre-compute neighbor table** (memory intensive):
   ```python
   # For depth=7: 8^7 = 2,097,152 entries
   # Each entry: 27 neighbors × 8 bytes = 216 bytes
   # Total: 452 MB (too large!)
   ```

2. **Decode-shift-encode approach**:
   ```python
   def hilbert_neighbors_27(hilbert_idx, depth):
       # 1. Decode Hilbert → (x, y, z)
       x, y, z = hilbert_decode_3d(hilbert_idx, depth)

       # 2. Enumerate 27 spatial neighbors
       neighbors = []
       for dx in [-1, 0, 1]:
           for dy in [-1, 0, 1]:
               for dz in [-1, 0, 1]:
                   nx, ny, nz = x + dx, y + dy, z + dz
                   # 3. Re-encode neighbor as Hilbert
                   neighbor_hilbert = hilbert_encode_3d(nx, ny, nz, depth)
                   neighbors.append(neighbor_hilbert)

       return neighbors
   ```

   **Cost**: 27 × (decode + encode) = ~54 depth iterations
   **vs Morton**: 27 × (simple arithmetic) = ~27 additions
   **Overhead**: ~2× slower

---

## Experimental Plan

### Experiment 1: Coordinate Normalization (Quick Test)

**Goal**: Check if float precision is the issue

**Implementation** (2 hours):
1. Add normalization to mesh loader
2. Scale velocities accordingly
3. Run same 500-step test
4. Compare retention

**Expected result**:
- If precision is the issue: +5-10% retention
- If not: <1% change

---

### Experiment 2: Adaptive Epsilon (Medium Test)

**Goal**: Improve point-in-tet robustness for small elements

**Implementation** (4 hours):
1. Modify `point_in_tet_gpu` to accept element_volumes
2. Compute adaptive epsilon per element
3. Update all point-in-tet calls
4. Run test

**Expected result**:
- +2-5% retention (reduce false negatives)

---

### Experiment 3: Hilbert Curve (Large Effort)

**Goal**: Replace Morton with Hilbert for better spatial locality

**Implementation** (40 hours):
1. Implement `hilbert_encode_3d` in JAX (8 hours)
2. Implement `hilbert_decode_3d` in JAX (8 hours)
3. Update Morton octree builder to use Hilbert (8 hours)
4. Update neighbor search to decode-shift-encode (8 hours)
5. Test and debug (8 hours)

**Expected result**:
- +5-15% retention (better neighbor locality)
- -20% throughput (slower neighbor arithmetic)

**Alternative**: Use Hilbert ONLY for refined region
- Hybrid Morton (coarse) + Hilbert (refined)
- Best of both worlds

---

## Quick Diagnostic: Which Factor is Dominant?

### Test 1: Check Point-in-Tet False Negatives

Add diagnostic to RK4:
```python
# After each point-in-tet test that fails
if not inside:
    # Compute distance from particle to element centroid
    centroid = node_positions[connectivity[elem_id]].mean(axis=0)
    dist = jnp.linalg.norm(pos - centroid)
    char_length = element_volumes[elem_id] ** (1.0/3.0)

    # If distance << char_length, likely false negative!
    if dist < 0.5 * char_length:
        false_negative_count += 1
```

**If false_negative_count is high**: Precision is the issue

---

### Test 2: Check Morton Discontinuity Losses

Add diagnostic to L2 search:
```python
# After L2 Morton neighbor search fails
if elem_id_L2 == -1:
    # Check if particle is near octree cell boundary
    morton_pos = position_to_morton(pos)

    # Test if neighbors in SPATIAL coords would find it
    for spatial_neighbor in get_spatial_neighbors_27(pos):
        elem_id_spatial = search_in_cell(spatial_neighbor)
        if elem_id_spatial >= 0:
            # Found with spatial search but NOT Morton search
            morton_discontinuity_loss += 1
```

**If morton_discontinuity_loss is high**: Morton locality is the issue

---

## Recommendations

### Priority 1: Coordinate Normalization (Fast, Low Risk)

**Pros**:
- 2 hours implementation
- Guaranteed to improve precision
- No performance penalty
- Easy to revert if ineffective

**Cons**:
- Need to track transform
- Velocity scaling complexity

**Verdict**: **DO THIS FIRST**

---

### Priority 2: Adaptive Epsilon (Medium, Medium Risk)

**Pros**:
- 4 hours implementation
- Addresses known issue (small elements)
- Likely +2-5% retention

**Cons**:
- Need to pass element_volumes everywhere
- Slight performance cost (extra computation per test)

**Verdict**: **DO IF NORMALIZATION HELPS**

---

### Priority 3: Hilbert Curve (Slow, High Risk)

**Pros**:
- Theoretically best spatial locality
- May be necessary for 262K× size variation

**Cons**:
- 40 hours implementation
- Complex algorithm (bugs likely)
- 2× slower neighbor arithmetic
- Unproven benefit for your case

**Verdict**: **ONLY IF DIAGNOSTICS CONFIRM MORTON IS THE ISSUE**

**Better alternative**: Fix Morton neighbor search for boundaries
- Simpler than Hilbert
- Target specific failure cases
- ~8 hours implementation

---

## Next Steps

1. **Add diagnostics** to identify which factor dominates (2 hours)
2. **Run 500-step test with diagnostics** (1 hour)
3. **Analyze results**:
   - If high false negatives → Normalization + Adaptive eps
   - If high Morton losses → Fix Morton boundary search or Hilbert
4. **Implement highest-impact solution first**

---

**Status**: Analysis complete. Ready to implement diagnostics or proceed with normalization experiment.
