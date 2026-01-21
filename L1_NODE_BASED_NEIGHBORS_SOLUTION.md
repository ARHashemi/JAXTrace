# L1 Node-Based Neighbors Solution

**Date**: 2025-12-19
**Issue**: L1 fix confirmed - face-based neighbors don't cross 1:2 refinement levels
**Solution**: Switch to node-based neighbor construction

---

## Problem Confirmed

After implementing the L1 algorithm fix, production tests still show **linear trajectories** in refined region. This definitively proves:

✅ **L1 algorithm fix is correct** (neighbor search now executes)
❌ **Face-based neighbors don't cross refinement levels** (coarse and fine elements don't share faces)

---

## Root Cause: Face-Based vs Node-Based Neighbors

### Current Implementation (Face-Based)

**Definition**: Two elements are neighbors if they share a **face (3 nodes)**

```python
# In production_tracking_fully_fused_timedep.py:
element_neighbors = build_element_neighbors_array(connectivity)
# Default: method='face'
# Shape: (N_elements, 4) - at most 4 face neighbors per element
```

**Why It Fails in 1:2 Octree Refinement**:

In octree refinement:
- Coarse cube (side length L) → 8 refined cubes (side length L/2)
- Each cube contains 4 right-angled tetrahedra

**Coarse-Fine Interface**:
```
Coarse element:           Fine element:
   Vertices: [A, B, C, D]    Vertices: [A, M₁, M₂, M₃]
   Faces:                     Faces:
   - [A, B, C]               - [A, M₁, M₂]
   - [A, B, D]               - [A, M₁, M₃]
   - [A, C, D]               - [A, M₂, M₃]
   - [B, C, D]               - [M₁, M₂, M₃]

where M₁, M₂, M₃ are midpoints on coarse edges
```

**Critical Observation**:
- Coarse and fine elements share **EDGES** (2 nodes like [A, M₁])
- They do NOT share **FACES** (3 nodes)
- **Face-based neighbors = 0 fine neighbors for coarse element** ❌

### Solution: Node-Based Neighbors

**Definition**: Two elements are neighbors if they share **ANY node** (vertex, edge, or face)

**Why It Works**:
```
Coarse element nodes: [A, B, C, D]
Fine element nodes:   [A, M₁, M₂, M₃]

Shared nodes: {A} (vertex) or {A, M₁} (edge)
→ Node-based neighbors detect this! ✓
```

---

## Implementation: Switch to Node-Based Neighbors

### Option 1: Simple Switch (Recommended for Testing)

**File**: `production_tracking_fully_fused_timedep.py` (line ~297)

**Current**:
```python
element_neighbors = build_element_neighbors_array(connectivity)
```

**Change to**:
```python
element_neighbors = build_element_neighbors_array(connectivity, method='node', verbose=True)
```

**Trade-offs**:
- ✅ **Correctness**: Finds all coarse→fine neighbors
- ✅ **Easy**: One-line change
- ❌ **Memory**: 600-1200 MB (vs 48 MB for face-based)
- ❌ **L1 Search Time**: More neighbors to check (20-100 vs 4)

### Option 2: Hybrid Neighbors (Best Performance)

Use **face-based for most elements**, **node-based only for boundary elements** near refined regions.

**Algorithm**:
1. Build face-based neighbors for all elements (fast, 48 MB)
2. Identify boundary elements (coarse elements near fine region)
3. For boundary elements only, add node-based neighbors
4. Result: 4 neighbors for most elements, 20-100 for boundary elements

**Implementation Sketch**:
```python
def build_hybrid_neighbors(connectivity, element_sizes, fine_threshold=0.15, coarse_threshold=0.30):
    """
    Build hybrid neighbor array: face-based + node-based for boundaries.

    Args:
        connectivity: (N, 4) element connectivity
        element_sizes: (N,) element characteristic sizes
        fine_threshold: Elements ≤ this are "fine" (0.15mm)
        coarse_threshold: Elements ≥ this are "coarse" (0.30mm)

    Returns:
        neighbors: (N, MAX_NEIGHBORS) with -1 padding
    """
    n_elements = connectivity.shape[0]

    # 1. Build face-based neighbors (fast, O(N))
    face_neighbors_dict, _ = extract_element_neighbors(connectivity, verbose=True)

    # 2. Build node-to-elements map (for boundary expansion)
    node_to_elements = build_node_to_elements_map(connectivity, verbose=True)

    # 3. Identify boundary elements (coarse near fine, fine near coarse)
    is_fine = element_sizes <= fine_threshold
    is_coarse = element_sizes >= coarse_threshold
    is_medium = ~is_fine & ~is_coarse

    # Boundary elements = coarse/medium elements with fine neighbors, or vice versa
    boundary_elements = set()

    for elem_id in range(n_elements):
        if is_coarse[elem_id] or is_medium[elem_id]:
            # Check if any face neighbor is fine
            face_neighs = face_neighbors_dict.get(elem_id, np.array([]))
            if np.any(is_fine[face_neighs]):
                # This coarse element has fine face-neighbors → Already handled by face-based
                continue

            # Check if any NODE neighbor is fine (edge/vertex sharing)
            nodes = connectivity[elem_id]
            for node_id in nodes:
                node_elements = node_to_elements[node_id]
                if np.any(is_fine[list(node_elements)]):
                    # This coarse element shares a node with fine element → BOUNDARY
                    boundary_elements.add(elem_id)
                    break

    print(f"Identified {len(boundary_elements):,} boundary elements ({100*len(boundary_elements)/n_elements:.2f}%)")

    # 4. Build hybrid neighbor array
    max_neighbors_face = 4
    max_neighbors_node = 100  # Upper bound for node-based
    max_neighbors = max_neighbors_node  # Use max for all (padded)

    neighbors = np.full((n_elements, max_neighbors), -1, dtype=np.int32)

    for elem_id in range(n_elements):
        if elem_id in boundary_elements:
            # Use node-based neighbors for boundary elements
            nodes = connectivity[elem_id]
            neighbor_set = set()
            for node_id in nodes:
                neighbor_set.update(node_to_elements[node_id])
            neighbor_set.discard(elem_id)  # Remove self

            neighbor_list = sorted(neighbor_set)
            n_neighs = min(len(neighbor_list), max_neighbors)
            neighbors[elem_id, :n_neighs] = neighbor_list[:n_neighs]
        else:
            # Use face-based neighbors for interior elements
            face_neighs = face_neighbors_dict.get(elem_id, np.array([]))
            n_neighs = len(face_neighs)
            neighbors[elem_id, :n_neighs] = face_neighs

    return neighbors
```

**Trade-offs**:
- ✅ **Memory Efficient**: Most elements have 4 neighbors, only boundary has 20-100
- ✅ **Fast L1**: Only boundary elements pay node-based search cost
- ❌ **Implementation**: More complex than simple switch
- ❌ **Maintenance**: Need to identify boundaries (element sizes required)

### Option 3: Octree-Aware Neighbors (Most Efficient)

Leverage **octree structure** to directly compute coarse→fine neighbors.

**Key Insight**: In octree refinement, coarse element at level L has **predictable** fine neighbors at level L+1.

**Algorithm**:
```python
def build_octree_aware_neighbors(connectivity, morton_codes, octree_levels):
    """
    Build neighbors using octree parent-child relationships.

    For each coarse element:
    1. Get its Morton code prefix at current level
    2. Find children in octree (8 sub-cubes)
    3. Add elements in children as neighbors

    This directly computes coarse→fine connectivity without node-based search.
    """
    # Implementation requires Morton code per element + octree hierarchy
    # More complex but most efficient for large refinement gaps
    pass
```

**Trade-offs**:
- ✅ **Most Efficient**: Directly computes refinement neighbors
- ✅ **Predictable Memory**: Known neighbor counts from octree structure
- ❌ **Requires Octree Info**: Need Morton codes + level per element
- ❌ **Implementation Complexity**: Higher than other options

---

## Recommendation: Phased Approach

### Phase 1A: Test Node-Based Neighbors (Immediate - 1 hour)

**Goal**: Confirm that node-based neighbors fix the linear trajectory issue

**Implementation**:
```python
# File: production_tracking_fully_fused_timedep.py, line ~297
element_neighbors = build_element_neighbors_array(connectivity, method='node', verbose=True)
```

**Test**:
```bash
# Run production script with node-based neighbors
python production_tracking_fully_fused_timedep.py 2>&1 | tee logs/production_node_based_test.log

# Expected results:
# - Particles show ROTATING trajectories in refined region ✓
# - Fine element assignment: 70-85% (vs 0% with face-based) ✓
# - Performance: 20-50K particles/s (slower due to more neighbors, but CORRECT)
```

**Success Criteria**:
1. ✅ Particles follow rotating flow (visual check of VTK output)
2. ✅ Fine element assignment >50% in refined region
3. ✅ No invalid element IDs (all particles tracked)

**If successful**: Node-based neighbors work! Proceed to optimization.

### Phase 1B: Optimize Memory (If Needed - 3-5 hours)

**If node-based neighbors consume too much memory** (>available GPU RAM):

**Option**: Implement hybrid neighbors (face-based + node-based for boundaries)

**Expected**:
- Memory: 100-200 MB (vs 48 MB face-only, 600-1200 MB node-only)
- Performance: 30-60K particles/s (faster than full node-based)

### Phase 2: Octree-Aligned Leaves (1 week)

**Regardless of Phase 1 outcome**, octree-aligned leaves provide better performance:

**Current**: Fixed 256-element segments (no spatial coherence)
**Phase 2**: Octree cells at depth 7 (1:1 prefix→leaf mapping)

**Benefits**:
- L2 search: O(1) prefix lookup instead of binary search
- Smaller leaf sizes in refined region (better cache locality)
- Expected: 100-150K particles/s

**Implementation**: See `MORTON_OPTIMIZATION_GUIDE.md` Section 5.2

### Phase 3: LBVH (2-3 weeks)

**Final optimization**: Linear Bounding Volume Hierarchy (Karras 2012)

**Expected**: 200-300K particles/s

---

## Detailed Steps for Phase 1A

### 1. Modify Production Script

**File**: `production_tracking_fully_fused_timedep.py`

**Line ~295-299** (in main function, section 3):
```python
# Compute element neighbors
print("  Computing element neighbors...")
t_neighbors = time.time()
element_neighbors = build_element_neighbors_array(connectivity, method='node', verbose=True)
t_neighbors = time.time() - t_neighbors
print(f"    Neighbor computation: {t_neighbors:.2f}s")
neighbor_memory_mb = element_neighbors.nbytes / (1024**2)
print(f"    Neighbor memory: {neighbor_memory_mb:.1f} MB")
print(f"    Neighbor array shape: {element_neighbors.shape}")
print(f"    Max neighbors per element: {element_neighbors.shape[1]}")
```

### 2. Enable L1 Search

**Line ~81**:
```python
ENABLE_L1_SEARCH = True  # Changed from False
```

### 3. Adjust L1 Hops (Optional)

With more neighbors, you may want to reduce hops:

**Line ~79**:
```python
N_HOPS = 2  # Changed from 3 (node-based finds neighbors faster)
```

### 4. Run Test

```bash
cd /home/arhashemi/Workspace/welding/JAXTrace
source .venv/bin/activate

# Run with smaller particle count first (fast test)
# Edit script: N_X = 20, N_Y = 30, N_Z = 20 → 12,000 particles
# N_STEPS = 500 (instead of 2,500)

python production_tracking_fully_fused_timedep.py 2>&1 | tee logs/production_node_based_test.log
```

### 5. Analyze Results

**Check Log**:
```bash
grep -A 5 "Neighbor" logs/production_node_based_test.log
# Look for:
#   Neighbor array shape: (3512384, XX)  ← XX = max neighbors (should be 20-100)
#   Neighbor memory: XXX MB  ← Should be 600-1200 MB

grep "Step.*retention" logs/production_node_based_test.log
# Look for retention >95% throughout
```

**Check VTK Output**:
```python
# Load last timestep in ParaView
# Apply Glyph filter (arrows) for velocities
# Visual check: Particles should show ROTATING motion near tool (X=30, Y=15, Z=0.3)
```

---

## Memory Estimation

### Face-Based (Current)
```
Shape: (3,512,384 elements, 4 neighbors)
Memory: 3,512,384 × 4 × 4 bytes = 56.2 MB
```

### Node-Based (Full)
```
Shape: (3,512,384 elements, MAX_NEIGHBORS)
MAX_NEIGHBORS estimate:
  - Uniform mesh: ~20-30 neighbors
  - Refined mesh: ~50-100 neighbors (near boundaries)
  - Conservative: 80 neighbors

Memory: 3,512,384 × 80 × 4 bytes = 1,122 MB = 1.1 GB
```

### Hybrid (Optimized)
```
Assume 5% boundary elements:
  - 95% elements: 4 neighbors
  - 5% elements: 80 neighbors

Memory:
  0.95 × 3,512,384 × 4 × 4 = 53.4 MB
  0.05 × 3,512,384 × 80 × 4 = 56.2 MB
  Total: ~110 MB
```

---

## Expected Performance Impact

### Face-Based (Current - WRONG)
- L1 hit rate: 0% (can't find fine neighbors)
- L2 fallback: 100%
- Throughput: 30K particles/s
- **Result**: LINEAR trajectories ❌

### Node-Based (Correct)
- L1 hit rate: 60-80% (finds fine neighbors)
- L2 fallback: 20-40%
- Throughput: 20-50K particles/s (slower due to more neighbors, but CORRECT)
- **Result**: ROTATING trajectories ✓

### Why Slower?
- More neighbors to check: 20-100 vs 4
- L1 loop: `for neighbor in neighbors[current_elem]`
- Each iteration: point-in-tet test (expensive)
- **But**: Still faster than L2-only if L1 hit rate >50%

---

## Alternative: Increase L1 Hops with Face-Based?

**Question**: Can we just increase `N_HOPS` with face-based neighbors to eventually reach fine elements?

**Answer**: **NO** ❌

**Reason**: Face-based neighbors form **disconnected components** across refinement levels.

```
Coarse element A (level L) has face-neighbors B, C, D, E (all level L)
Fine element F (level L+1) shares EDGE with A (NOT face)

Multi-hop search from A:
  Hop 1: Check B, C, D, E (all coarse) → Particle not found
  Hop 2: Check neighbors of B, C, D, E (all coarse, same level) → Not found
  Hop 3: Check their neighbors (still coarse) → Not found
  ...
  Hop N: NEVER reaches F (different connected component)

Result: L1 fails, falls to L2 (global search)
```

**Conclusion**: Multi-hop face-based search **CANNOT cross refinement levels**. Must use node-based.

---

## Summary

### Current Status
- ✅ L1 algorithm fixed (neighbor search executes)
- ❌ Face-based neighbors don't cross 1:2 refinement
- ❌ Linear trajectories persist

### Immediate Action (Phase 1A)
1. **Change one line** in production script:
   ```python
   element_neighbors = build_element_neighbors_array(connectivity, method='node', verbose=True)
   ```
2. **Enable L1**: `ENABLE_L1_SEARCH = True`
3. **Run test** with small particle count (12K particles, 500 steps)
4. **Check results**: Rotating trajectories + >50% fine assignment

### If Memory is Concern
- Implement **hybrid neighbors** (Phase 1B)
- Or proceed directly to **Phase 2: Octree-aligned leaves**

### Long-Term Path
- Phase 2: Octree-aligned leaves (100-150K particles/s)
- Phase 3: LBVH (200-300K particles/s)

---

## Code Changes Required

### Minimal Change (Testing)

**File**: `production_tracking_fully_fused_timedep.py`

**Line ~297** (in section `[3/6] Uploading mesh and Morton structure to GPU`):
```python
# BEFORE:
element_neighbors = build_element_neighbors_array(connectivity)

# AFTER:
element_neighbors = build_element_neighbors_array(connectivity, method='node', verbose=True)
```

**Line ~81**:
```python
# BEFORE:
ENABLE_L1_SEARCH = False

# AFTER:
ENABLE_L1_SEARCH = True
```

**That's it!** Two-line change to test node-based neighbors.

---

## Next Steps

1. **Test node-based neighbors** (1 hour)
   - Make two-line change above
   - Run with 12K particles, 500 steps
   - Verify rotating trajectories

2. **Analyze results**
   - Check memory usage (expect ~1.1 GB for neighbors)
   - Check performance (expect 20-50K particles/s)
   - Check correctness (visual + fine assignment %)

3. **If successful**:
   - **Option A**: Use node-based for production (correct but slower)
   - **Option B**: Implement hybrid neighbors (faster, more complex)
   - **Option C**: Proceed to Phase 2 (octree leaves, best long-term)

4. **If memory issue**:
   - Implement hybrid neighbors (Phase 1B)
   - Or go directly to Phase 2 (skip L1, use better L2)

---

**Recommendation**: Start with Phase 1A (test node-based). If it works but is too slow, proceed to Phase 2 (octree-aligned leaves) rather than optimizing L1 further.

---

**End of Document**
