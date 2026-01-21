# Node-Based Neighbors - Test Results & Analysis

**Date**: 2025-12-19
**Status**: ✅ SUCCESS - Trajectories Correct, Memory Constraint Identified

---

## Executive Summary

**SUCCESS**: Node-based neighbors + L1 fix produce **correct rotating trajectories**!

**Key Findings**:
1. ✅ **Trajectories**: Completely correct (rotating motion in refined region)
2. ✅ **Performance**: 28-29K particles/s (acceptable for 48K particles)
3. ✅ **Retention**: 86.66% (stable throughout tracking)
4. ⚠️ **Memory Constraint**: OOM during compilation for >50K particles (GPU limit: 4GB T1000)

**Recommendation**: Implement **hybrid neighbors** to reduce memory footprint from 1GB to ~150MB, enabling 200K+ particles.

---

## Test Results

### Configuration
- **Mesh**: FLA (3,048,900 elements, 780,922 nodes)
- **Particles**: 48,000 (20×80×30 grid)
- **Timesteps**: 2,500
- **dt**: 0.0025 s
- **Search**: L0 → L1 (3 hops) → L2 (radius=10)

### Node-Based Neighbor Statistics

**Construction**:
```
Method: NODE-BASED
Elements: 3,048,900
Neighbors per element: min=1, max=90, avg=58.97
Memory: 1046.8 MB (1.02 GB)
Array shape: (3,048,900, 90)
Construction time: 28.43s
```

**Key Metrics**:
- **Max neighbors**: 90 (vs 4 for face-based)
- **Avg neighbors**: 58.97 (vs 4 for face-based)
- **Memory**: 1.02 GB (vs 48 MB for face-based)
- **Multiplier**: 14.7× more neighbors than face-based

### Performance Results

**Initial Assignment**:
```
Radius 100: 39,908/48,000 (83.14%)
Radius 200:    157 more    (83.47%)
Radius 500:    129 more    (83.74%)
Final:      40,194/48,000 (83.74%)
```

**Tracking Performance**:
```
Step 100: 41,597 active (86.66%), 29,290 particles/s
Step 200: 41,597 active (86.66%), 28,790 particles/s

Avg throughput: ~29,000 particles/s
Retention: 86.66% (stable)
Compilation time: 25.38s (first step)
```

**Memory Usage** (during tracking):
```
GPU: NVIDIA T1000 (4GB)
  Total: 4096 MB
  Used:  3711 MB (90.6%)
  Free:     6 MB (0.1%)
  Utilization: 100% (GPU busy)

CPU: 32GB
  Used: 17 GB (53%)
  Available: 13 GB
```

### Correctness Verification

✅ **Trajectories**: User confirms "completely correct" rotating motion in refined region
✅ **Stability**: Retention stable at 86.66% (no particle loss during tracking)
✅ **Performance**: Acceptable speed (~29K particles/s for 48K particles)

---

## Memory Breakdown

### GPU Memory Allocation (Total: ~3.7 GB used)

**1. Mesh Data** (~450 MB):
- Node positions: (780,922, 3) × 4 bytes = 9.4 MB
- Connectivity: (3,048,900, 4) × 4 bytes = 46.4 MB
- **Element neighbors: (3,048,900, 90) × 4 bytes = 1,046.8 MB** ← LARGEST
- Morton codes: 34.9 MB

**2. Velocity Fields** (357.5 MB):
- 40 timesteps × (780,922, 3) × 4 bytes = 357.5 MB

**3. Particle Data** (~0.6 MB):
- Positions: (48,000, 3) × 4 bytes = 0.576 MB
- Element IDs: 48,000 × 4 bytes = 0.192 MB

**4. JIT Compilation Overhead** (~2 GB):
- Intermediate buffers during RK4 compilation
- Neighbor indexing arrays (90 neighbors × 48K particles)
- **This is where OOM occurs for >50K particles**

### Critical Bottleneck: Element Neighbors Array

**Current**: `(3,048,900, 90)` = **1.02 GB**
- All elements have 90-neighbor slots (max)
- Most elements use only ~59 neighbors (avg)
- Wasted memory: ~30% padding

**Impact on Compilation**:
- JAX creates neighbor lookup tables during compilation
- For N particles × 90 neighbors: Large intermediate arrays
- **OOM threshold**: ~50K particles on 4GB GPU

---

## Comparison: Face vs Node-Based

### Face-Based (WRONG)

**Neighbors**:
- Max: 4 neighbors/element
- Avg: ~3.5 neighbors/element
- Memory: 48 MB

**Results**:
- ❌ Linear trajectories (WRONG)
- ❌ 0% fine element assignment in refined region
- ✅ Can handle 200K+ particles (low memory)

**L1 Search**:
- ❌ Cannot cross refinement levels
- ❌ 0% hit rate
- ❌ Falls to L2 100% of time

### Node-Based (CORRECT)

**Neighbors**:
- Max: 90 neighbors/element
- Avg: 58.97 neighbors/element
- Memory: 1,046.8 MB

**Results**:
- ✅ Rotating trajectories (CORRECT)
- ✅ 60-85% fine element assignment (estimated)
- ⚠️ Can handle ~48K particles (memory limited)

**L1 Search**:
- ✅ Crosses refinement levels
- ✅ 60-80% hit rate (estimated)
- ✅ Reduces L2 fallback

---

## Root Cause: Why OOM During Compilation?

### JIT Compilation Memory Pattern

JAX's JIT compiler creates **intermediate arrays** during first execution:

**For L1 neighbor search**:
```python
# For each particle, access neighbors of current element
neighbors = element_neighbors[current_elem]  # (90,) per particle

# During compilation, JAX allocates:
# - Input buffer: (N_particles, 90) neighbor IDs
# - Mask buffer: (N_particles, 90) validity checks
# - Result buffer: (N_particles, 90) containment tests
# - Reduction buffer: (N_particles,) first found

# For 50K particles × 90 neighbors:
# 50,000 × 90 × 4 bytes = 18 MB per buffer
# 4 buffers × 18 MB = 72 MB intermediate memory

# Plus: vmap overhead, gradient buffers, etc.
# Total compilation overhead: ~200-500 MB
```

**With node-based**: 1.02 GB neighbors + 357 MB velocities + 500 MB compilation = **1.88 GB base**
- **Remaining for particles**: 4 GB - 1.88 GB = ~2.1 GB
- **Max particles**: 2100 MB / (90 neighbors × 4 bytes × 4 buffers) ≈ **58K particles**

**With face-based**: 48 MB neighbors + 357 MB velocities + 200 MB compilation = **605 MB base**
- **Remaining for particles**: 4 GB - 605 MB = ~3.4 GB
- **Max particles**: Much higher (200K+)

---

## Solution: Hybrid Neighbors

### Concept

**Selective node-based neighbors**:
- **Interior elements** (95%): Face-based (4 neighbors)
- **Boundary elements** (5%): Node-based (20-100 neighbors)

**Boundary identification**:
- Elements near refinement transitions (coarse/fine interface)
- Detected via element size gradient or Morton level change

### Expected Memory Reduction

**Current (Full Node-Based)**:
```
3,048,900 elements × 90 neighbors × 4 bytes = 1,046.8 MB
```

**Hybrid (5% Boundary)**:
```
Interior (95%):  2,896,455 × 4 × 4 bytes = 44.3 MB
Boundary (5%):     152,445 × 90 × 4 bytes = 52.4 MB
Total:                                       96.7 MB
```

**Savings**: 1,046.8 MB → 96.7 MB = **90.8% reduction**

### Implementation Strategy

**Option A: Ragged Array (Best Memory)**
```python
def build_hybrid_neighbors_ragged(connectivity, element_sizes):
    """
    Build hybrid neighbors with variable-length arrays.

    Interior: 4 neighbors (face-based)
    Boundary: ~60 neighbors (node-based)

    Returns:
        neighbors: List[np.ndarray] - variable length per element
        max_neighbors: int - for padding if needed
    """
    # Classify elements
    is_fine = element_sizes <= 0.15
    is_coarse = element_sizes >= 0.30

    # Build face-based for all
    face_neighbors = extract_element_neighbors(connectivity)

    # Build node map for boundary detection
    node_to_elements = build_node_to_elements_map(connectivity)

    neighbors = []
    for elem_id in range(len(connectivity)):
        # Start with face neighbors
        neighs = set(face_neighbors[elem_id])

        # Check if boundary (shares nodes with different level)
        is_boundary = False
        for node_id in connectivity[elem_id]:
            node_elems = node_to_elements[node_id]
            # If fine element has coarse neighbor, or vice versa
            if is_fine[elem_id] and np.any(is_coarse[list(node_elems)]):
                is_boundary = True
                break
            if is_coarse[elem_id] and np.any(is_fine[list(node_elems)]):
                is_boundary = True
                break

        # Add node-based neighbors if boundary
        if is_boundary:
            for node_id in connectivity[elem_id]:
                neighs.update(node_to_elements[node_id])
            neighs.discard(elem_id)  # Remove self

        neighbors.append(np.array(sorted(neighs), dtype=np.int32))

    return neighbors

# Convert to padded array
max_neighbors = max(len(n) for n in neighbors)
element_neighbors = np.full((len(neighbors), max_neighbors), -1, dtype=np.int32)
for i, neighs in enumerate(neighbors):
    element_neighbors[i, :len(neighs)] = neighs
```

**Option B: Fixed-Size Dual Array (Simpler JAX)**
```python
def build_hybrid_neighbors_dual(connectivity, element_sizes):
    """
    Build hybrid with two separate arrays.

    Returns:
        face_neighbors: (N, 4) - all elements
        boundary_mask: (N,) bool - True if needs node-based
        node_neighbors: (N_boundary, 100) - boundary elements only
        boundary_indices: (N_boundary,) - maps boundary to full array
    """
    # Build face-based (all)
    face_neighbors = build_element_neighbors_array(connectivity, method='face')

    # Identify boundary elements
    boundary_mask = identify_boundary_elements(connectivity, element_sizes)
    boundary_indices = np.where(boundary_mask)[0]

    # Build node-based for boundary only
    node_neighbors_dict = extract_element_neighbors_node_based(connectivity)
    max_node_neighbors = 100
    node_neighbors = np.full((len(boundary_indices), max_node_neighbors), -1, dtype=np.int32)

    for i, elem_id in enumerate(boundary_indices):
        neighs = node_neighbors_dict[elem_id]
        n = min(len(neighs), max_node_neighbors)
        node_neighbors[i, :n] = neighs[:n]

    return face_neighbors, boundary_mask, node_neighbors, boundary_indices

# Usage in L1 search:
if boundary_mask[current_elem]:
    # Lookup boundary index
    boundary_idx = jnp.searchsorted(boundary_indices, current_elem)
    neighbors = node_neighbors[boundary_idx]  # 100 neighbors
else:
    neighbors = face_neighbors[current_elem]  # 4 neighbors
```

**Trade-offs**:

| Approach | Memory | JAX Complexity | Performance |
|----------|--------|----------------|-------------|
| **Ragged** | Minimal (~100 MB) | Medium (padding needed) | Best (no wasted lookups) |
| **Dual** | Low (~150 MB) | Medium (conditional lookup) | Good (boundary check overhead) |
| **Full Node** | High (1 GB) | Simple (single array) | Good (but OOM) |

---

## Recommended Next Steps

### Immediate: Document Success ✅

**Current Status**:
- ✅ L1 fix works (algorithm corrected)
- ✅ Node-based neighbors work (trajectories correct)
- ✅ Performance acceptable (~29K particles/s for 48K particles)
- ⚠️ Memory limit: ~50K particles max on 4GB GPU

**Action**: Update documentation with test results

### Short-Term: Hybrid Neighbors (Priority)

**Goal**: Enable 200K+ particles on 4GB GPU

**Implementation** (3-5 days):
1. **Boundary detection** (1 day):
   - Implement element size gradient detection
   - Identify coarse/fine interface elements
   - Validate: 5-10% boundary elements

2. **Hybrid neighbor construction** (1 day):
   - Build dual-array structure (face + boundary node)
   - Pad to max neighbors per category
   - Test: Verify memory ~150 MB

3. **L1 search modification** (1 day):
   - Modify L1 to use conditional lookup
   - Test: Verify same trajectories as full node-based
   - Benchmark: Measure performance impact

4. **Production testing** (1-2 days):
   - Test with 100K particles
   - Test with 200K particles
   - Verify: Correct trajectories + no OOM

**Expected Results**:
- Memory: 1,046 MB → 150 MB (86% reduction)
- Max particles: 50K → 200K (4× increase)
- Performance: ~29K p/s (similar to current)

### Medium-Term: Octree-Aligned Leaves (Best Long-Term)

**Goal**: Improve L2 search performance + reduce memory further

**Why Better Than Optimizing L1**:
1. **Memory**: No neighbor array needed (L2-only)
2. **Performance**: 100-150K particles/s (vs 29K)
3. **Scalability**: Works for any mesh size
4. **Maintainability**: Simpler than hybrid neighbors

**Implementation** (1-2 weeks):
1. **Octree leaf builder**:
   - Replace fixed 256-element segments
   - Build leaves = octree cells at depth 7
   - Implement 1:1 prefix→leaf mapping

2. **L2 search optimization**:
   - O(1) prefix lookup (vs binary search)
   - Radius search in octree space
   - Expected: 3-5× faster than current L2

3. **Disable L1** (optional):
   - If L2 fast enough, skip L1 entirely
   - Saves 150 MB neighbor memory
   - Simplifies code

**Expected Results**:
- Performance: 100-150K particles/s
- Memory: ~500 MB (no neighbors)
- Max particles: 400K+ on 4GB GPU

### Long-Term: LBVH (Optional)

**Goal**: Maximum performance (200-300K particles/s)

**Only if needed for**:
- >500K particles
- Real-time tracking
- Multi-physics coupling

---

## Performance Analysis

### Current Performance: 29K particles/s

**Breakdown** (estimated from 1.64s/step):
- L0 check: ~100 ms (cached element)
- L1 search: ~800 ms (neighbor lookups, 60-80% hit)
  - 90 neighbors × point-in-tet tests
  - 3 hops max
- L2 search: ~400 ms (20-40% fallback)
  - Prefix lookup
  - Leaf scan
- RK4 integration: ~300 ms (4 stages)
- Export overhead: ~40 ms

**L1 Cost Analysis**:
```
Avg neighbors checked per particle: 90 × 3 hops × 60% = ~162 tet tests
Point-in-tet cost: ~5 microseconds
Total L1 cost: 162 × 5 µs = 810 µs per particle
For 48K particles: 810 µs × 48,000 = 38.9 seconds

Actual L1 time: ~800 ms (with GPU parallelization)
Speedup: 48× from parallelization ✓
```

### Hybrid Neighbors Impact

**Boundary elements**: 5% (152,445 / 3,048,900)
**Interior elements**: 95% (2,896,455)

**L1 search**:
- Interior: 4 neighbors × 3 hops = 12 tet tests (fast)
- Boundary: 90 neighbors × 3 hops = 270 tet tests (same as current)

**Expected performance**:
- If 50% particles in boundary region: ~29K p/s (same)
- If 20% particles in boundary region: ~35K p/s (20% faster)
- If 5% particles in boundary region: ~40K p/s (38% faster)

**For typical friction stir welding**:
- Refined region: ~5mm diameter
- Domain: ~60mm × 30mm × 7mm
- Volume fraction: (π × 2.5² × 7) / (60 × 30 × 7) ≈ 1%
- Expected: **~40K particles/s** with hybrid neighbors

### Octree-Aligned Leaves Impact

**L2 search**:
- Current: Binary search (log N) + leaf scan
- Octree: O(1) prefix lookup + neighbor leaves

**Expected performance**:
- L2 speedup: 3-5× faster
- If L2 = 40% of time: Overall 1.5-2× speedup
- **Expected: 50-60K particles/s** (L1 + L2 optimized)

**If disable L1** (L2-only):
- L2 must handle 100% of searches (vs 20-40%)
- But L2 is 3-5× faster
- **Expected: 100-150K particles/s** (L2-only, optimized)

---

## Recommendations Summary

### For Production Use Now

**Option 1: Use Current Node-Based** (Immediate)
- ✅ Works: Correct trajectories
- ✅ Performance: 29K p/s (acceptable)
- ⚠️ Limit: 48K particles max
- **Best for**: Immediate production use with small particle counts

### For Scaling to 200K Particles

**Option 2: Implement Hybrid Neighbors** (3-5 days)
- ✅ Memory: 150 MB (vs 1 GB)
- ✅ Particles: 200K max
- ✅ Performance: ~35-40K p/s
- ⚠️ Complexity: Medium implementation
- **Best for**: Scaling while keeping L1 search

### For Best Performance

**Option 3: Octree-Aligned Leaves** (1-2 weeks)
- ✅ Memory: 500 MB (no neighbors)
- ✅ Particles: 400K max
- ✅ Performance: 100-150K p/s
- ⚠️ Time: 1-2 weeks
- **Best for**: Long-term optimal solution

---

## Conclusion

**SUCCESS**: Node-based neighbors + L1 fix solve the trajectory problem completely!

**Key Achievement**: Particles now follow correct rotating flow in refined region.

**Next Decision**: Choose optimization path based on requirements:
- **Small runs (<50K particles)**: Use current node-based ✓
- **Large runs (200K particles)**: Implement hybrid neighbors
- **Maximum performance**: Implement octree-aligned leaves

**Immediate Action**: Document this success and decide on optimization priority.

---

## Appendix: System Specifications

### GPU
```
Model: NVIDIA T1000
Total Memory: 4096 MB
Memory Used: 3711 MB (90.6%)
Memory Free: 6 MB (0.1%)
Utilization: 100%
```

### CPU
```
Total: 32 GB
Used: 17 GB (53%)
Available: 13 GB
Swap: 512 MB (22 MB used)
```

### Mesh
```
Elements: 3,048,900 (FLA mesh)
Nodes: 780,922
Velocity timesteps: 40
Total mesh data: ~400 MB
```

### Neighbor Array
```
Shape: (3,048,900, 90)
Size: 1,046.8 MB
Construction time: 28.43s
Avg neighbors: 58.97
Max neighbors: 90
```

---

**End of Report**
