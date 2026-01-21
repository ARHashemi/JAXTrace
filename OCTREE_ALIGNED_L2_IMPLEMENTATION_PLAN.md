# Octree-Aligned L2 Implementation Plan

**Date**: 2025-12-22
**Goal**: Replace fixed-capacity Morton leaves with proper octree-aligned leaves for improved particle retention

---

## Executive Summary

### Current Problem (Fixed-Capacity Leaves)

**Current Implementation**:
- Leaves are arbitrary segments: Leaf i = elements [i×256, (i+1)×256] in Morton-sorted array
- No spatial coherence: Elements in same leaf can be far apart spatially
- Binary search inefficiency: Must search Morton array to find leaf
- **Result**: 60-70% particle retention, high L2 failure rate

**Root Cause**:
```
Example: Element centroids at positions with similar Morton codes may be in
different fixed-capacity leaves, causing L2 to miss them even with radius search.
```

### Proposed Solution (Octree-Aligned Leaves)

**New Implementation**:
- Leaves align with spatial octants: Leaf defined by Morton code prefix
- Perfect spatial coherence: Elements in same leaf are spatially close
- O(1) prefix lookup: Position → Morton code → Extract prefix bits → Leaf ID
- **Expected**: 90-95% particle retention, minimal L2 failures

**Key Insight**:
Morton codes encode octree hierarchy in their bits. A k-bit prefix defines a unique octant at depth k/3.

---

## Background: Morton Codes and Octrees

### Morton Code Structure

A 63-bit Morton code encodes 3D position as interleaved (x, y, z) bits:

```
Depth 0 (root):     [empty prefix]          → 1 cell (entire domain)
Depth 1:            [xyz]                    → 8 cells (3 bits)
Depth 2:            [xyz][xyz]               → 64 cells (6 bits)
Depth 3:            [xyz][xyz][xyz]          → 512 cells (9 bits)
...
Depth 21:           [xyz] × 21               → 2^63 cells (63 bits)
```

**Example Morton Code**:
```
Position: (0.25, 0.75, 0.5) in [0, 1]³
Binary grid (21 bits): x=010..., y=110..., z=100...
Interleaved: [xyz][xyz][xyz]... = [011][110][000]...
Morton code: 0b011110000... (leading bits define octant hierarchy)
```

### Octree Hierarchy in Morton Codes

**Key Property**: Morton code prefixes naturally define octree cells.

```
Morton Code:     [001][101][011][xyz...]
                  └─┬─┘ └─┬─┘ └─┬─┘
                 Depth1 Depth2 Depth3
                 Cell 1 Cell 5 Cell 3 in parent
```

**Spatial Meaning**:
- Prefix `[001]`: Octant 1 at depth 1 (binary 001 → bottom-left-back)
- Prefix `[001][101]`: Octant 5 at depth 2 within parent octant 1
- Elements with same k-bit prefix are in same depth-k octant

---

## Current Implementation Analysis

### File: `jaxtrace/gpu/search/morton_octree_builder.py`

**Status**: Already implements adaptive octree builder! ✅

**Key Functions**:

1. **`build_adaptive_octree_leaves()`** (lines 113-150+)
   - Recursively subdivides Morton-sorted array into octants
   - Creates leaves with ≤256 elements aligned to spatial octants
   - Returns `List[OctreeLeaf]` with prefix information

2. **`compute_octant_ranges()`** (lines 40-110)
   - Partitions Morton range into 8 child octants using binary search
   - Uses bit-shifting to match Morton prefixes

3. **`OctreeLeaf` dataclass** (lines 28-37)
   ```python
   @dataclass
   class OctreeLeaf:
       start_idx: int        # Index in morton_sorted
       length: int           # Number of elements
       morton_prefix: int    # Prefix defining this octant
       prefix_bits: int      # Number of prefix bits (depth × 3)
   ```

**Current Status**: Builder exists but **NOT USED** in production code!

### File: `jaxtrace/gpu/search/morton_global_search.py`

**Current L2 Search** (uses fixed-capacity leaves):

1. **`position_to_leaf_id_linear()`** (lines 324-360)
   - Linear approximation: `leaf_id = (morton - morton_min) / span × n_leaves`
   - **WRONG**: Assumes uniform Morton distribution

2. **`position_to_leaf_id_octree()`** (lines 207-282) ✅
   - Uses `prefix_start` and `prefix_length` lookup tables
   - Extracts prefix from Morton code
   - Searches candidate leaves for exact match
   - **CORRECT**: But prefix tables not built properly in current code

3. **`search_in_leaf_global()`** (lines 456+)
   - Searches elements within a leaf using point-in-tet tests
   - Works with both fixed-capacity and octree leaves

**Problem**: Production code uses `table_depth=0` fallback to linear method!

---

## Implementation Plan

### Phase 1: Enable Existing Octree Code ✅ (1-2 hours)

**Goal**: Use existing `build_adaptive_octree_leaves()` in production

**Files to Modify**:
1. `jaxtrace/gpu/mesh_loader.py` (or wherever Morton structure is built)
   - Replace fixed-capacity leaf building with `build_adaptive_octree_leaves()`
   - Build prefix lookup tables from octree leaves

2. `production_tracking_fully_fused_timedep.py`
   - Ensure `position_to_leaf_id_octree()` is used (check `table_depth > 0`)

**Steps**:
1. Locate where `MeshGPUGlobalMorton` is created
2. Replace fixed-capacity logic with octree builder
3. Build prefix tables (depth 6-7 recommended for 3M elements)
4. Test position→leaf accuracy

**Expected Outcome**:
- Leaves aligned with octants
- Improved L2 hit rate (85-90%)

---

### Phase 2: Add Configuration Switch (30 minutes)

**Goal**: Allow easy A/B testing between methods

**File**: `production_tracking_fully_fused_timedep.py`

**Add Configuration**:
```python
# L2 Search Method Selection:
#   'fixed': Fixed-capacity leaves (current, ~60% retention)
#            - Leaf i = elements [i*256, (i+1)*256]
#            - Fast to build, but poor spatial coherence
#   'octree': Octree-aligned leaves (new, expected 90-95% retention)
#            - Leaves align with Morton code prefixes (spatial octants)
#            - Slower to build (~10s), but much better accuracy
L2_SEARCH_METHOD = 'octree'  # 'fixed' or 'octree'
```

**Implementation**:
```python
if L2_SEARCH_METHOD == 'octree':
    # Use adaptive octree builder
    leaves, prefix_tables = build_adaptive_octree_leaves(...)
    table_depth = 7  # Use prefix table for fast lookup
else:
    # Use fixed-capacity leaves (current)
    n_leaves = (n_elements + 255) // 256
    table_depth = 0  # Disable prefix table (fallback to linear)
```

---

### Phase 3: Optimize Octree Leaf Building (if needed)

**Current Builder Issues** (to check):
1. Prefix table size for depth D: `8^D` entries
   - Depth 6: 262K entries (1 MB)
   - Depth 7: 2M entries (8 MB)
   - Depth 8: 16M entries (64 MB) ← Too large?

2. Build time: Recursive subdivision can be slow for 3M elements

**Optimizations** (if needed):
1. **Sparse prefix table**: Only store valid prefixes (save memory)
2. **Parallel building**: Use NumPy vectorization for octant partitioning
3. **Depth selection**: Auto-select depth based on element count
   - Target: 100-300 elements per leaf
   - Formula: `depth ≈ log₈(n_elements / target_leaf_size)`

---

### Phase 4: Neighbor Search Optimization (future)

**Current**: L2 searches ±radius leaves linearly

**With Octree Leaves**: Can use Morton arithmetic to find neighbor octants

**Morton Neighbor Finding**:
```python
def get_neighbor_octants(morton_prefix, prefix_bits):
    """Get 26 neighbor octants at same depth."""
    # Decode prefix to (x, y, z) octant coordinates
    x, y, z = decode_morton_prefix(morton_prefix, prefix_bits)

    # Generate 26 neighbors: (x±1, y±1, z±1) excluding self
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0:
                    continue
                nx, ny, nz = x + dx, y + dy, z + dz
                neighbor_prefix = encode_morton_prefix(nx, ny, nz, prefix_bits)
                neighbors.append(neighbor_prefix)

    return neighbors
```

**Benefit**: Search only spatially adjacent leaves (not arbitrary ±radius)

---

## Technical Details

### Prefix Table Construction

**Goal**: Fast O(1) mapping from Morton code to leaf ID

**Current Approach** (in `morton_octree_builder.py`):
```python
# For each possible prefix at depth D:
for prefix in range(8**D):
    # Find which leaf contains this prefix
    # Store: prefix_start[prefix] = first_leaf_id
    #        prefix_length[prefix] = num_leaves_with_this_prefix
```

**Problem**: Variable-depth leaves mean multiple leaves can share a prefix

**Example**:
```
Prefix at depth 6: [001][101][011][110][000][111]
  Leaf A (depth 7): [001][101][011][110][000][111][000] → 128 elements
  Leaf B (depth 7): [001][101][011][110][000][111][001] → 64 elements
  ...
  (Subdivided because parent had >256 elements)
```

**Solution**: `prefix_length` stores count of matching leaves, then linear search

### GPU Memory Layout

**Fixed-Capacity (Current)**:
```python
MeshGPUGlobalMorton:
    leaf_start:  [0, 256, 512, 768, ...]  # Evenly spaced
    leaf_length: [256, 256, 256, ...]     # All same capacity
    n_leaves: n_elements // 256
    table_depth: 0                         # No prefix table
```

**Octree-Aligned (New)**:
```python
MeshGPUGlobalMorton:
    leaf_start:  [0, 128, 256, 384, 640, ...]  # Variable spacing
    leaf_length: [128, 128, 128, 256, 192, ...]  # Variable capacity
    n_leaves: ~n_elements / 200              # Fewer, larger leaves
    table_depth: 7                           # Use prefix table
    prefix_start: (2M entries, 8 MB)         # Prefix → first_leaf_id
    prefix_length: (2M entries, 8 MB)        # Prefix → num_leaves
```

**Memory Impact**: +16 MB for prefix tables (negligible)

---

## Expected Performance Improvements

### Particle Retention

| Metric | Fixed-Capacity | Octree-Aligned | Improvement |
|--------|----------------|----------------|-------------|
| Initial assignment | 83.74% | 83.74% | Same (L2 not used) |
| Step 100 | 79.39% | **~85-88%** | +6-9% |
| Step 500 | 70.27% | **~80-85%** | +10-15% |
| Step 2500 | ~60% | **~75-85%** | +15-25% |

**Reason**: Particles that move between elements stay within same octant → L2 finds them

### Throughput

| Metric | Fixed-Capacity | Octree-Aligned | Change |
|--------|----------------|----------------|--------|
| Initial build | <1s | ~5-10s | Slower (one-time) |
| Step time | 3.7s | **3.0-3.5s** | 10-20% faster |
| Throughput | 13K p/s | **14-16K p/s** | +1-3K p/s |

**Reason**: Fewer L2 searches needed (more L0/L1 hits), fewer leaves to search

### L2 Search Effectiveness

| Metric | Fixed-Capacity | Octree-Aligned |
|--------|----------------|----------------|
| Leaf hit rate | ~12-15% | **~95-98%** |
| Avg leaves searched | 21 (radius=10) | **1-3** (radius=1) |
| Point-in-tet tests | 21 × 256 = 5376 | **200-600** |

**Definition**: "Leaf hit rate" = % of particles where correct element is in center leaf

---

## Testing Plan

### Test 1: Octree Leaf Building (CPU-side)

**File**: Create `test_octree_leaves.py`

```python
# 1. Load mesh and build Morton structure
# 2. Build octree leaves with build_adaptive_octree_leaves()
# 3. Verify:
#    - All elements covered (sum of leaf_length == n_elements)
#    - No overlaps (leaf ranges are contiguous)
#    - Capacity constraint (all leaves ≤ 256 elements)
#    - Prefix correctness (elements in leaf match prefix)
# 4. Build prefix tables
# 5. Test position→leaf mapping accuracy
```

**Success Criteria**:
- 100% element coverage
- 95%+ leaf hit rate for element centroids
- Prefix table build time <10s

### Test 2: Production Run with Octree Leaves

**Configuration**:
```python
L2_SEARCH_METHOD = 'octree'
L2_SEARCH_RADIUS = 2  # Start with radius=2 (was 10)
NEIGHBOR_METHOD = 'node'
ENABLE_L1_SEARCH = True
```

**Run**:
```bash
python production_tracking_fully_fused_timedep.py 2>&1 | tee logs/test_octree_leaves.log
```

**Expected Results**:
- Higher retention: 75-85% at step 2500 (vs 60% current)
- Faster steps: 3.0-3.5s (vs 3.7s current)
- Correct trajectories: Rotating motion maintained

### Test 3: A/B Comparison

**Run Both Methods**:
1. Fixed-capacity: `L2_SEARCH_METHOD = 'fixed'`
2. Octree-aligned: `L2_SEARCH_METHOD = 'octree'`

**Compare**:
- Retention curves (plot active particles vs step)
- Throughput (particles/s)
- Compilation time
- Memory usage

---

## Implementation Priority

### Must-Have (Phase 1)
1. ✅ Enable `build_adaptive_octree_leaves()` in production
2. ✅ Build and upload prefix tables to GPU
3. ✅ Ensure `position_to_leaf_id_octree()` is used

### Should-Have (Phase 2)
4. ✅ Add L2_SEARCH_METHOD configuration switch
5. ✅ Reduce L2_SEARCH_RADIUS to 1-2 (octree needs less)
6. ✅ Add warnings if fixed-capacity selected

### Nice-to-Have (Phase 3+)
7. Optimize prefix table size (sparse representation)
8. Morton neighbor arithmetic for smarter radius search
9. Auto-select octree depth based on element count
10. Parallel octree building (if slow)

---

## Known Issues and Mitigations

### Issue 1: Variable-Depth Complexity

**Problem**: Leaves at different depths complicate neighbor finding

**Example**:
```
Depth 6 leaf (coarse): 256 elements in large octant
Depth 8 leaf (fine):   64 elements in small octant
```

**Mitigation**:
- Prefix table handles this automatically (maps to all matching leaves)
- Linear search through candidate leaves (typically <8)

### Issue 2: Prefix Table Memory

**Problem**: Depth 8 table = 16M entries (64 MB) per GPU

**Mitigation**:
- Use depth 7 (2M entries, 8 MB)
- For 3M elements: depth 7 gives ~300 elements/leaf on average
- Sparse table: Only store valid prefixes (future optimization)

### Issue 3: Build Time

**Problem**: Recursive octree building can be slow (up to 30s for 3M elements)

**Mitigation**:
- One-time cost at startup (acceptable)
- Can be pre-built and cached to disk
- Parallel building (NumPy vectorization) if needed

---

## Code Organization

### New Files (Optional)
- `jaxtrace/gpu/search/morton_octree_search.py` - Octree-specific search functions
- `test_octree_leaves.py` - Unit tests for octree builder

### Modified Files
1. **`jaxtrace/gpu/mesh_loader.py`** (or wherever Morton is built)
   - Add octree leaf builder
   - Build prefix tables
   - Add method selection logic

2. **`production_tracking_fully_fused_timedep.py`**
   - Add L2_SEARCH_METHOD configuration
   - Pass method to mesh loader

3. **`jaxtrace/gpu/search/morton_global_search.py`** (minimal changes)
   - Already has `position_to_leaf_id_octree()` ✅
   - Just ensure it's used when `table_depth > 0`

---

## References

Based on web research (2025-12-22):

1. **[Linear representation of the octree using the Morton code](https://sudonull.com/post/121448-Linear-representation-of-the-octree-using-the-Morton-code)**
   - "Morton order is perfect for an Octree, as the 8 children of a node are guaranteed to be in contiguous memory"
   - Describes variable-depth linear octrees

2. **[Z-order curve - Wikipedia](https://en.wikipedia.org/wiki/Z-order_curve)**
   - "Z-ordering can be used to efficiently build a quadtree or octree by sorting the input set according to Z-order"
   - Standard reference for Morton codes

3. **[GPU Octrees and Optimized Search](http://profs.ic.uff.br/~esteban/files/papers/SBGames09_Madeira.pdf)**
   - GPU-optimized octree traversal techniques
   - Discusses linear octree representations

4. **[Binarized octree generation for Cartesian adaptive](https://arxiv.org/pdf/1712.00408)**
   - Academic paper on adaptive octree generation
   - Describes capacity-constrained subdivision

---

## Summary

**Current State**:
- Octree builder exists but unused
- Fixed-capacity leaves cause low retention (~60%)
- Binary search inefficient

**Proposed State**:
- Use existing `build_adaptive_octree_leaves()`
- Octree-aligned leaves with prefix tables
- Expected 90-95% retention

**Effort**:
- Phase 1: 1-2 hours (enable existing code)
- Phase 2: 30 minutes (add switch)
- Phase 3: Optional (optimizations)

**Risk**: Low - can easily switch back to fixed-capacity if issues

**Next Steps**: Locate mesh loading code and implement Phase 1

---

**Ready to implement! Start with Phase 1: Find where `MeshGPUGlobalMorton` is built.**
