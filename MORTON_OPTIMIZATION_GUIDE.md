# Morton-Based Search Optimization Guide

## Executive Summary

This document explains the particle tracking accuracy issue in the friction stir welding simulation, the root cause analysis, the current Morton code implementation, and detailed optimization strategies to improve L2 search efficiency.

**Problem**: Particles show linear trajectories instead of rotating trajectories in the refined region around the tool.

**Root Cause**: L1 neighbor search fails in graded refinement meshes; L0→L2 works but requires large search radius (trade-off between accuracy and performance).

**Solution Path**: Optimize L2 Morton-based search using spatial hierarchy information encoded in Morton codes.

---

## Table of Contents

1. [Problem Analysis: Particle Trajectory Issue](#1-problem-analysis-particle-trajectory-issue)
2. [L1 Search Failure in Graded Refinement](#2-l1-search-failure-in-graded-refinement)
3. [L0+L2 Results and Shortcomings](#3-l0l2-results-and-shortcomings)
4. [Morton Code Structure and Encoded Information](#4-morton-code-structure-and-encoded-information)
5. [Optimization Option A: Adaptive Search Radius](#5-optimization-option-a-adaptive-search-radius)
6. [Optimization Option B: Multi-Scale Hierarchical Search](#6-optimization-option-b-multi-scale-hierarchical-search)
7. [Optimization Option C: Prefix Table with Spatial Filtering](#7-optimization-option-c-prefix-table-with-spatial-filtering)
8. [Implementation Recommendations](#8-implementation-recommendations)

---

## 1. Problem Analysis: Particle Trajectory Issue

### 1.1 Observed Behavior

**Symptom**: Particles passing through the refined region (rotating tool area) show:
- ✅ Correct global advancing velocity along X-axis (matches commercial code)
- ❌ **Missing rotation** in the refined region (diverges from commercial code)
- ❌ Linear trajectories instead of swirling patterns

**Visual Evidence**: `Screenshot_20251218_010306.png` shows:
- Red spheres (JAXTrace): Linear paths through refined region
- Gray spheres (commercial code): Swirling/rotating paths

### 1.2 Root Cause Diagnosis

**Investigation Timeline**:

1. **Initial Hypothesis**: Morton code resolution insufficient
   - **Result**: REJECTED - 63-bit encoding provides nanometer resolution (far exceeds 0.14mm elements)

2. **Particle Seeding Issue**: Particles not seeded in refined region
   - **Result**: REJECTED - Seeding at entrance is correct; particles advect through refined region

3. **Element Assignment in Refined Region**: (`diagnose_tracking_through_refined_region.py`)
   - **Result**: CONFIRMED ROOT CAUSE
   - Particles in refined region assigned to COARSE elements 100% of time
   - Example: Particle 3 traveled 289 steps through refined region
     - Fine elements (≤0.15mm): **0/289 (0%)**
     - Coarse elements (>0.30mm): **289/289 (100%)**

4. **Neighbor Connectivity Analysis**: (`diagnose_neighbor_connectivity.py`)
   - **Result**: SMOKING GUN
   - 100% of boundary coarse elements have **ZERO fine neighbors**
   - Face-based neighbors only find face-sharing elements
   - L1 hop search cannot reach fine elements from coarse elements

### 1.3 Mesh Structure Discovery

**Graded Refinement Structure**:

```
Fine (≤0.15mm)    →    Medium (0.15-0.30mm)    →    Coarse (>0.30mm)
85.3% (2.6M elems)     12.5% (382K elems)           2.2% (67K elems)
[Rotating tool]        [Transition buffer]          [Outer domain]
```

**Key Finding**: Fine and coarse elements **do not share nodes**. Medium-sized elements form a buffer zone between them.

**Spatial Distribution**:
- Fine region: X=[-9.36, 9.34]mm, Y=[-9.38, 9.40]mm, Z=[-4.51, -0.02]mm
- Medium region: X=[-9.65, 9.61]mm, Y=[-9.82, 9.86]mm, Z=[-4.96, -0.04]mm
- Coarse region: X=[-28.75, 28.75]mm, Y=[-21.72, 21.72]mm, Z=[-9.37, -0.08]mm

---

## 2. L1 Search Failure in Graded Refinement

### 2.1 Why Face-Based Neighbors Fail

**Face-Based Neighbor Definition**: Two tetrahedra are neighbors if they share a face (3 nodes).

**Problem in Graded Refinement**:
```
Coarse Element Boundary:
  Shares EDGE (2 nodes) with fine elements
  Shares FACE (3 nodes) with medium elements only

Face-Based Neighbors:
  Coarse → Medium ✅
  Coarse → Fine   ❌ (edge-sharing not detected)
```

**Test Results** (`test_node_based_neighbors.py`):
- Tested 20 boundary coarse elements
- Face-based neighbors: **0 fine neighbors** for all 20 elements
- Node-based neighbors: Also **0 fine neighbors** (confirmed graded structure)

### 2.2 Why Node-Based Neighbors Are Insufficient

**Node-Based Neighbor Definition**: Two tetrahedra are neighbors if they share ANY node (vertex, edge, or face).

**Mesh Structure** (6-7 Level Octree Refinement):
- Base unit: 4 right-angled tetrahedra form a cube
- Refinement: Each cube subdivides into 8 octant cubes (each with 4 tets)
- **Edge-sharing between levels**: Coarse cube edge splits into 2 edges, shared by 2 refined elements

**Expected from Octree Refinement**:
Node-based neighbors SHOULD find coarse→fine connectivity via edge-sharing:
```
Coarse element edge: Nodes [A, B]
  After refinement: Split into edges [A, M] and [M, B] (M = midpoint)
  Fine elements share nodes A, M, or B with coarse element
  → Node-based neighbors SHOULD detect this
```

**Why Implementation Still Insufficient**:

The issue is **not that node-based neighbors failed conceptually**, but rather:

1. **Memory Constraints**: Node-based neighbor arrays are large
   - Face-based: 4 neighbors → 48 MB for 3M elements
   - Node-based: 20-100 neighbors → **1+ GB for 3M elements**
   - May exceed available GPU memory depending on hardware

2. **L1 Hop Search Limitation** (explained in 2.3): Even with complete neighbor connectivity, the hop search algorithm itself has fundamental issues in graded refinement

3. **Diagnostic Test Results** (`test_node_based_neighbors.py`):
   - Found that specific boundary elements tested had 0 fine neighbors
   - This occurred because **medium elements form complete spatial buffer**
   - With 6-7 levels of gradual refinement, coarse and fine are separated by multiple intermediate levels
   - Direct coarse-to-fine node sharing is rare

**Conclusion**: Node-based neighbors are theoretically correct for single-level refinement (coarse→fine with edge-sharing), but in **multi-level graded refinement** (coarse→level2→level3→...→fine), even node-based connectivity requires multiple hops, bringing us back to the L1 hop search problem.

### 2.3 L1 Hop Search Failure - Detailed Algorithm Analysis

**Current Configuration**: `N_HOPS = 3`, `L2_SEARCH_RADIUS = 10`

**L1 Algorithm** (from `jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py:87-122`):

```python
def search_l1_single(pos, start_elem_id):
    """L1: Multi-hop neighbor search (single particle)."""
    current_elem = start_elem_id  # Start from cached element
    found = current_elem >= 0

    for hop in range(n_hops):  # n_hops = 3
        if found:
            return current_elem  # EARLY EXIT if any element found

        # Get 4 face-based neighbors of current_elem
        neighbors = element_neighbors[current_elem]  # Shape: (4,)

        # Test all 4 neighbors (point-in-tet test)
        for neighbor_id in neighbors:
            if neighbor_id >= 0:
                if point_in_tet(pos, neighbor_id):
                    current_elem = neighbor_id
                    found = True
                    break  # RETURN FIRST FOUND NEIGHBOR

        # Move to found neighbor for next hop
        # (If no neighbor found, current_elem unchanged, next hop tries same neighbors)

    return current_elem  # Return last found element (or -1 if none)
```

**Key Algorithm Properties**:

1. **Early Exit on First Found**: Returns immediately when ANY element is found
2. **No Size/Quality Check**: Accepts first element containing position
3. **Greedy Search**: Doesn't explore all paths, just first successful path
4. **No Backtracking**: Can't revisit previous elements

**Why L1 Fails in Graded Refinement**:

**Scenario**: Particle at position P inside a fine element, but L0 cached a coarse element

```
Hop 0 (Initial):
  current_elem = COARSE (element 1793477, size=1.09mm)
  L0 test: point_in_tet(P, COARSE) = FALSE (particle moved)
  → Proceed to Hop 1

Hop 1:
  neighbors of COARSE: [Medium1, Medium2, Medium3, Medium4]  (sizes ~0.25mm)
  Test Medium1: point_in_tet(P, Medium1) = TRUE  ✅
  → found = TRUE
  → current_elem = Medium1
  → EARLY EXIT (algorithm returns Medium1)

L1 Result: Returns Medium1 (size=0.25mm)
L2: NEVER REACHED (because L1 "succeeded")
```

**Why Medium Element is Wrong**:

The particle position P is actually inside **both**:
- Medium1 (size 0.25mm) - covers large volume
- Fine_k (size 0.14mm) - nested inside Medium1

**Tetrahedral mesh property**: Elements can **spatially overlap** in graded refinement:
```
Coarse element (1mm cube):
  Subdivided into 8 Medium cubes (0.5mm each)
  One Medium cube further subdivided into 8 Fine cubes (0.25mm each)

Position inside Fine cube is also geometrically inside:
  - Its parent Medium cube
  - Its grandparent Coarse cube (if we kept it)

BUT: In the mesh, only the FINEST level elements exist at each location
```

**The Real Problem**:

The L1 algorithm returns Medium1 because:
1. Medium1 **geometrically contains** position P (point-in-tet = TRUE)
2. Medium1 is a **neighbor** of the coarse element
3. Algorithm accepts **first found element**, doesn't check if finer elements exist

But the mesh was refined further at this location:
- Medium1 was subdivided into 8 fine elements
- Medium1 **no longer exists** in the active mesh (or is marked inactive)
- The actual active element is Fine_k

**Why This Causes Wrong Velocities**:

```
Medium1 vertices: Interpolate velocity from 4 nodes at medium spacing
  → Smooth, low-resolution velocity field
  → Misses high gradients in rotating tool region

Fine_k vertices: Interpolate velocity from 4 nodes at fine spacing
  → Captures rotation velocity gradients
  → Correct physics
```

**Fundamental L1 Flaw**:

The algorithm **assumes topological connectivity = spatial accuracy**, but in graded refinement:
- Topology: Coarse connects to Medium (neighbors via face/edge)
- Spatial reality: Position is in Fine element nested inside Medium

L1 follows topology, finds Medium, and stops searching (early exit).

**Why Increasing N_HOPS Doesn't Fully Solve This**:

```
N_HOPS = 10:
  Hop 1: Coarse → Medium
  Hop 2: Medium → Fine  ✅ (if Fine shares edge with Medium)

But:
  - Early exit on Hop 1 if Medium contains position (common case)
  - Only proceeds to Hop 2 if Medium point-in-tet = FALSE
  - For graded refinement, Medium DOES contain position (spatially larger)
  → Still returns Medium, never reaches Fine
```

**Only Case Where More Hops Help**:

If the cached element is **completely wrong** (particle moved far):
```
Cached: Coarse element at (-15mm, 0, 0)
Actual: Fine element at (-10mm, 0, 0)

Hop 1: Coarse → Medium1 (point-in-tet = FALSE, particle outside)
Hop 2: Medium1 → Medium2 (point-in-tet = FALSE)
...
Hop 5: Medium5 → Fine (point-in-tet = TRUE)
→ Returns Fine ✅
```

But this is rare - most particles stay within ~1-2 element distances per timestep.

**Summary of L1 Failure**:

1. **Greedy first-found**: Accepts first element containing position
2. **No refinement awareness**: Can't distinguish parent from child elements
3. **Early exit prevents deeper search**: Stops at medium-sized elements
4. **Spatial containment ≠ correct assignment**: Medium contains position but Fine is the active element

**Why L0→L2 Works**:

L2 Morton search is **spatially-aware**, not topology-based:
- Encodes position → Morton code → Find spatially local elements
- Searches Morton-sorted list directly (bypasses neighbor topology)
- Returns **first element whose point-in-tet succeeds** in spatial order
- Fine elements have Morton codes near the particle's Morton code
- Finds fine elements directly without relying on neighbor hops

---

## 3. L0+L2 Results and Shortcomings

### 3.1 L0+L2 Configuration (L1 Disabled)

**Implementation**: Added `ENABLE_L1_SEARCH = False` parameter

**Search Hierarchy**:
```
L0: Cached element
  ↓ FAILS
L2: Morton global search (DIRECT)
  → Spatially-aware search, not topology-dependent
  → Can find fine elements regardless of neighbor connectivity
```

### 3.2 L0+L2 with Small Radius (INITIAL TEST)

**Configuration**:
```python
ENABLE_L1_SEARCH = False
L2_SEARCH_RADIUS = 10
```

**Results** (`production_fully_fused_timedep_NoL1.log`):
- ✅ **Correct trajectories**: Rotation visible, matches commercial code
- ❌ **Significant particle loss**: Many particles lost each timestep
- ⚠️ **Performance**: Acceptable (~40-60K particles/s)

**Why Particle Loss Occurs**:
```
Particle position: (-10.000mm, 0, -2.000mm)
  → Morton code M_p
  → Maps to Leaf L_center

Fine element centroid: (-10.050mm, 0, -2.030mm)  [50μm away]
  → Morton code M_e (DIFFERENT from M_p!)
  → Maps to Leaf L_target

Leaf distance: |L_target - L_center| = 50 leaves

L2_SEARCH_RADIUS=10:
  Searches leaves [L_center-10, L_center+10]
  → Misses L_target (too far)
  → Returns -1 (particle lost)
```

### 3.3 L0+L2 with Large Radius (WORKING BUT SLOW)

**Configuration**:
```python
ENABLE_L1_SEARCH = False
L2_SEARCH_RADIUS = 100
```

**Results**:
- ✅ **Correct trajectories**: Rotation visible
- ✅ **No particle loss**: All particles tracked successfully
- ❌ **Performance**: **~30 seconds per timestep** (10x slower than baseline)

**Performance Analysis**:
- Baseline (L0+L1+L2, radius=10): 50-120K particles/s, 2-5s/timestep
- L0+L2 (radius=100): ~30K particles/s, ~30s/timestep
- **Cause**: Searching 200 leaves (±100) per failed L0 check
  - ~20-30% of particles fail L0 per timestep
  - Each failure searches 200 leaves × 256 elements = 51,200 point-in-tet tests

### 3.4 The Trade-off

| Configuration | Accuracy | Particle Loss | Performance | Verdict |
|---------------|----------|---------------|-------------|---------|
| L0+L1(3)+L2(10) | ❌ Wrong | ✅ None | ✅ Fast (50-120K/s) | ❌ Incorrect results |
| L0+L2(10) | ✅ Correct | ❌ High | ✅ Fast (40-60K/s) | ❌ Unusable (particle loss) |
| L0+L2(100) | ✅ Correct | ✅ None | ❌ Slow (30K/s) | ⚠️ Works but slow |

**Goal**: Achieve L0+L2(100) accuracy with L0+L2(10) performance.

---

## 4. Morton Code Structure and Encoded Information

### 4.1 63-Bit Morton Code Structure

**Total Bits**: 63 bits (21 bits per dimension)

**Bit Layout** (interleaved Z-order):
```
Bit position:  62 61 60 | 59 58 57 | 56 55 54 | ... | 5 4 3 | 2 1 0
Dimension:      z  y  x |  z  y  x |  z  y  x | ... | z y x | z y x
Octree level:   Level 1  | Level 2  | Level 3  | ... | L20   | L21
```

**Each 3-bit group**: `[z_bit, y_bit, x_bit]` encodes octant 0-7 at that level

**Example**:
```
Morton code: 0x1A3F5C2E8B74D196
Binary (top 18 bits): 000110 100011 111101 ...
                      ^^^^^^ ^^^^^^ ^^^^^^
                      Level1 Level2 Level3
Octree path: Octant 6 → Octant 35 → Octant 61 → ...
```

### 4.2 Encoding Process (Position → Morton Code)

**Implementation**: `jaxtrace/gpu/morton_code.py:119-147`

**Steps**:

1. **Normalize to bounding box**:
   ```python
   normalized = (pos - bbox_min) / (bbox_max - bbox_min)
   # Result: [0, 1]³
   ```

2. **Quantize to integer grid**:
   ```python
   grid_max = 2^21 - 1  # 2,097,151
   u_x = floor(normalized.x * grid_max)  # [0, 2,097,151]
   u_y = floor(normalized.y * grid_max)
   u_z = floor(normalized.z * grid_max)
   ```

3. **Interleave bits**:
   ```python
   morton = interleave_bits_3d(u_x, u_y, u_z)
   # Bit pattern: z[20]y[20]x[20] z[19]y[19]x[19] ... z[0]y[0]x[0]
   ```

**Resolution for Your Domain** (60mm × 46mm × 10mm):
- X resolution: 60mm / 2,097,152 ≈ **29 nanometers**
- Y resolution: 46mm / 2,097,152 ≈ **22 nanometers**
- Z resolution: 10mm / 2,097,152 ≈ **5 nanometers**

**Comparison to Mesh**: Finest elements are 0.14mm = 140,000 nanometers → Morton resolution is **4,800× finer** than mesh.

---

### 4.3 Mathematical Formulation of Morton Encoding/Decoding

This section provides the complete mathematical formulation for encoding positions to Morton codes and decoding Morton codes back to positions.

#### **4.3.1 Encoding: Position → Morton Code**

**Given**:
- Position in world coordinates: **p** = (p_x, p_y, p_z) ∈ ℝ³
- Domain bounding box: [**b**_min, **b**_max] where **b**_min, **b**_max ∈ ℝ³
- Encoding depth: D = 21 bits per dimension (total 63 bits)

**Step 1: Normalization**

Normalize position to unit cube [0, 1]³:

```
p̂ = (p - b_min) / (b_max - b_min)

p̂_x = (p_x - b_min,x) / (b_max,x - b_min,x)  ∈ [0, 1]
p̂_y = (p_y - b_min,y) / (b_max,y - b_min,y)  ∈ [0, 1]
p̂_z = (p_z - b_min,z) / (b_max,z - b_min,z)  ∈ [0, 1]
```

**Step 2: Quantization**

Map to integer grid [0, 2^D - 1]:

```
N = 2^D - 1  (for D=21: N = 2,097,151)

u_x = ⌊p̂_x × N⌋  ∈ [0, N]  (uint32)
u_y = ⌊p̂_y × N⌋  ∈ [0, N]  (uint32)
u_z = ⌊p̂_z × N⌋  ∈ [0, N]  (uint32)
```

where ⌊·⌋ is the floor function.

**Step 3: Bit Interleaving**

Morton code M is constructed by interleaving the bits of (u_x, u_y, u_z):

```
M = ∑_{i=0}^{D-1} [bit_i(u_x) << (3i) | bit_i(u_y) << (3i+1) | bit_i(u_z) << (3i+2)]

where:
  bit_i(u) = (u >> i) & 1  (extract i-th bit from right)
  << denotes left bit shift
  | denotes bitwise OR
```

**Bit Layout of M** (63 bits, uint64):

```
M = [b_{20}^z b_{20}^y b_{20}^x] [b_{19}^z b_{19}^y b_{19}^x] ... [b_0^z b_0^y b_0^x]
     └─────────────┘  └─────────────┘        └─────────┘
       Level 1           Level 2              Level 21

where b_i^x, b_i^y, b_i^z are the i-th bits of u_x, u_y, u_z respectively
```

**Example**:
```
Given: p = (0.3, 0.7, 0.1) in domain [0, 1]³, D = 3 (for illustration)

Step 1: p̂ = (0.3, 0.7, 0.1)  (already normalized)

Step 2: N = 2^3 - 1 = 7
  u_x = ⌊0.3 × 7⌋ = 2 = 0b010
  u_y = ⌊0.7 × 7⌋ = 4 = 0b100
  u_z = ⌊0.1 × 7⌋ = 0 = 0b000

Step 3: Interleave bits (right to left):
  Bit 0: z=0, y=0, x=0 → 000
  Bit 1: z=0, y=0, x=1 → 001
  Bit 2: z=0, y=1, x=0 → 010

  M = 0b010001000 = 72 (decimal)
```

**Efficient Implementation** (using magic numbers):

Instead of bit-by-bit loops, use bit-twiddling with magic masks:

```
expand_bits(u):  # Inserts 00 between each bit of u
  u = (u | (u << 32)) & 0x1f00000000ffff
  u = (u | (u << 16)) & 0x1f0000ff0000ff
  u = (u | (u <<  8)) & 0x100f00f00f00f00f
  u = (u | (u <<  4)) & 0x10c30c30c30c30c3
  u = (u | (u <<  2)) & 0x1249249249249249
  return u

morton_encode(u_x, u_y, u_z):
  xx = expand_bits(u_x)
  yy = expand_bits(u_y)
  zz = expand_bits(u_z)
  return xx | (yy << 1) | (zz << 2)
```

**Implementation**: `jaxtrace/gpu/morton_code.py:28-76`

---

#### **4.3.2 Decoding: Morton Code → Position**

**Given**:
- Morton code: M ∈ [0, 2^63 - 1]  (uint64)
- Domain bounding box: [**b**_min, **b**_max]
- Encoding depth: D = 21

**Step 1: Deinterleave Bits**

Extract the interleaved bits to recover (u_x, u_y, u_z):

```
u_x = ∑_{i=0}^{D-1} [(M >> (3i)) & 1] << i
u_y = ∑_{i=0}^{D-1} [(M >> (3i+1)) & 1] << i
u_z = ∑_{i=0}^{D-1} [(M >> (3i+2)) & 1] << i
```

In words: Extract every 3rd bit starting from positions 0, 1, 2 respectively.

**Efficient Implementation**:

```
compact_bits(m):  # Reverse of expand_bits
  m = (m | (m >> 2)) & 0x10c30c30c30c30c3
  m = (m | (m >> 4)) & 0x100f00f00f00f00f
  m = (m | (m >> 8)) & 0x1f0000ff0000ff
  m = (m | (m >> 16)) & 0x1f00000000ffff
  m = (m | (m >> 32)) & 0x1fffff
  return m

morton_decode(M):
  x_bits = M & 0x1249249249249249  # Extract bits at positions 0,3,6,...
  y_bits = (M >> 1) & 0x1249249249249249  # Positions 1,4,7,...
  z_bits = (M >> 2) & 0x1249249249249249  # Positions 2,5,8,...

  u_x = compact_bits(x_bits)  # uint32 ∈ [0, 2^D-1]
  u_y = compact_bits(y_bits)
  u_z = compact_bits(z_bits)

  return (u_x, u_y, u_z)
```

**Step 2: Denormalization**

Map from integer grid back to [0, 1]³:

```
N = 2^D - 1

p̂_x = u_x / N  ∈ [0, 1]
p̂_y = u_y / N
p̂_z = u_z / N
```

**Step 3: Scale to World Coordinates**

```
p_x = p̂_x × (b_max,x - b_min,x) + b_min,x
p_y = p̂_y × (b_max,y - b_min,y) + b_min,y
p_z = p̂_z × (b_max,z - b_min,z) + b_min,z

p = (p_x, p_y, p_z)  ∈ [b_min, b_max]
```

**Implementation**: `jaxtrace/gpu/morton_code.py:150-191`

---

#### **4.3.3 Element Morton Code**

For mesh elements (tetrahedra), Morton code is computed from the **element centroid**:

**Given**:
- Element k with 4 vertices: **v**_1^k, **v**_2^k, **v**_3^k, **v**_4^k ∈ ℝ³

**Centroid**:
```
c^k = (1/4) ∑_{i=1}^4 v_i^k

c^k = ((x_1 + x_2 + x_3 + x_4)/4,
       (y_1 + y_2 + y_3 + y_4)/4,
       (z_1 + z_2 + z_3 + z_4)/4)
```

**Morton Code for Element k**:
```
M_k = morton_encode(c^k, b_min, b_max, D)
```

**Properties**:
- Element k is assigned a single Morton code based on its centroid
- Elements are sorted globally by M_k (Z-curve order)
- Spatially nearby elements → nearby M_k values
- Element k resides in Morton leaf: L_k = M_k >> (63 - table_depth × 3)

**Implementation**: `jaxtrace/gpu/search/morton_global_builder.py:109-179`

---

#### **4.3.4 Octree Prefix Extraction**

The Morton code directly encodes the octree path. Extract the top d levels:

**Given**:
- Morton code M (63 bits)
- Prefix depth d ∈ [1, 21]

**Prefix at depth d**:
```
P_d = M >> (63 - 3d)

This extracts the top 3d bits of M
```

**Interpretation**:
```
P_d ∈ [0, 8^d - 1]  (there are 8^d possible octants at depth d)

Each 3 bits = one octree level:
  P_d = [bits 62-60][bits 59-57]...[bits (63-3d+2)-(63-3d)]
       └─ Level 1 ─┘└─ Level 2 ─┘    └──── Level d ────┘
```

**Example for Your Mesh** (table_depth = 6):
```
d = 6 → 3d = 18 bits
P_6 = M >> 45

P_6 ∈ [0, 262,143]  (8^6 = 262,144 octants at level 6)

Octree cell size at level 6:
  Domain: 60mm × 46mm × 10mm
  Cell size: (60/64, 46/64, 10/64) ≈ (0.94mm, 0.72mm, 0.16mm)
```

**Octant Index to 3D Coordinates**:

Each octant at level i is identified by 3 bits [z, y, x]:
```
octant = 0b[z][y][x]  ∈ [0, 7]

octant = 0 = 0b000 → (x=0, y=0, z=0)  (bottom-front-left)
octant = 1 = 0b001 → (x=1, y=0, z=0)  (bottom-front-right)
octant = 2 = 0b010 → (x=0, y=1, z=0)  (bottom-back-left)
...
octant = 7 = 0b111 → (x=1, y=1, z=1)  (top-back-right)
```

**Decoding Prefix to Octree Path**:

Given P_d, extract the octree path from root to level d:

```
For level i ∈ [1, d]:
  shift = (d - i) × 3
  octant_i = (P_d >> shift) & 0b111  # Extract 3 bits

  x_i = (octant_i >> 0) & 1  # Bit 0
  y_i = (octant_i >> 1) & 1  # Bit 1
  z_i = (octant_i >> 2) & 1  # Bit 2
```

**Example**:
```
M = 0x1A3F5C2E8B74D196 (63-bit Morton code)
P_6 = M >> 45 = 0x034FBC (18 bits)

Binary: 0x034FBC = 0b 000 000 110 010 111 111
                     └─┘ └─┘ └─┘ └─┘ └─┘ └─┘
                     Lv1 Lv2 Lv3 Lv4 Lv5 Lv6

Octree path:
  Level 1: octant 0 = (x=0, y=0, z=0)
  Level 2: octant 0 = (x=0, y=0, z=0)
  Level 3: octant 6 = (x=0, y=1, z=1)
  Level 4: octant 2 = (x=0, y=1, z=0)
  Level 5: octant 7 = (x=1, y=1, z=1)
  Level 6: octant 7 = (x=1, y=1, z=1)
```

---

#### **4.3.5 Position to Octree Leaf Mapping**

**Problem**: Given particle position **p**, find the Morton leaf containing elements near **p**.

**Method 1: Direct Encoding + Prefix Table** (Current Implementation, O(1)):

```
1. Encode position to Morton code:
   M_p = morton_encode(p, b_min, b_max, D)

2. Extract prefix at table_depth:
   P = M_p >> (63 - 3 × table_depth)

3. Lookup leaf range in prefix table:
   first_leaf = prefix_start[P]  # O(1) array access
   num_leaves = prefix_length[P]

4. Search within leaf range (typically 1-8 leaves):
   for i ∈ [0, num_leaves):
     leaf_id = first_leaf + i
     if leaf_contains_morton(leaf_id, M_p):
       return leaf_id
```

**Method 2: Binary Search** (Fallback, O(log n_leaves)):

```
1. Encode position to Morton code: M_p
2. Binary search in sorted Morton array to find insertion point
3. Compute leaf_id = insertion_index / leaf_capacity
```

**Implementation**:
- Prefix table: `jaxtrace/gpu/search/morton_global_search.py:207-282`
- Binary search: `jaxtrace/gpu/search/morton_global_search.py:148-204`

---

### 4.4 What Information is Encoded

#### **Spatial Position** (Primary)

Morton code encodes the **discrete position** on a 2^21 × 2^21 × 2^21 integer grid.

**Decoding**:
```python
def morton_decode_3d(morton: uint64) -> (uint32, uint32, uint32):
    """Reverse interleaving to get (u_x, u_y, u_z)"""
    # Returns integer coordinates [0, 2^21-1]
```

**Back to world coordinates**:
```python
normalized = (u_x, u_y, u_z) / (2^21 - 1)
position = normalized * (bbox_max - bbox_min) + bbox_min
```

#### **Octree Hierarchy** (Implicit)

Morton code **directly encodes the octree path** from root to leaf.

**Extracting octree prefix** (top N levels):
```python
def get_octree_prefix(morton: uint64, depth: int) -> uint64:
    """Extract top 'depth' octree levels (depth × 3 bits)"""
    shift = 63 - depth * 3
    return morton >> shift

# Example: depth=6 → extract top 18 bits
prefix_6 = morton >> 45
# This prefix identifies which octant at level 6 the point is in
```

**Current Implementation**: `table_depth = 6` in your mesh

- **Prefix bits**: 6 levels × 3 bits/level = **18 bits**
- **Prefix space**: 2^18 = **262,144 unique prefixes**
- **Interpretation**: Domain divided into 8^6 = 262,144 octree cells at level 6

#### **Spatial Locality** (Z-Order Curve Property)

Morton codes preserve **spatial proximity** along the Z-order space-filling curve:

- Points close in 3D space → close Morton codes (usually)
- Morton codes form continuous path through 3D space
- **Not perfect**: Discontinuities at octant boundaries

**Sorting by Morton**:
```python
sorted_morton = np.sort(morton_codes)
# Elements are now in Z-curve traversal order
# Spatially nearby elements tend to be adjacent in sorted array
```

**Why This Matters for Search**:
- Elements in same octree cell → similar Morton codes
- Elements in adjacent octree cells → nearby Morton codes
- **Search radius**: ±R leaves searches spatially local region

### 4.4 What is NOT Encoded

| Information | Encoded? | Notes |
|-------------|----------|-------|
| Position | ✅ Yes | Quantized to 2^21 grid per dimension |
| Octree path | ✅ Yes (implicit) | Top N×3 bits encode path to level N |
| Spatial neighbors | ⚠️ Approximate | Z-curve has discontinuities |
| Element size | ❌ No | Must be stored separately |
| Refinement level | ⚠️ Indirect | Element density at prefix gives hint |
| Velocity | ❌ No | Looked up via element ID |
| Topology | ❌ No | No neighbor connectivity info |

### 4.5 Morton Leaf Structure

**Current Implementation**: `jaxtrace/gpu/search/morton_global_builder.py`

**Fixed-Capacity Leaves**:
```python
leaf_capacity = 256  # Max elements per leaf
n_leaves = (n_elements + 255) // 256  # 3,048,900 elems → 11,910 leaves
```

**Leaf Structure**:
- `elem_ids_sorted`: Global Morton-sorted element list (3M elements)
- `leaf_start[i]`: Start index of leaf i in sorted list
- `leaf_length[i]`: Number of elements in leaf i (≤256)

**Leaf Assignment**:
- Leaves are **consecutive segments** of sorted Morton array
- Leaf 0: elements [0, 256)
- Leaf 1: elements [256, 512)
- Leaf L: elements [L×256, min(L×256+256, 3M))

**Not Octree-Aligned**: Leaves are **arbitrary segments**, not geometric octants. Multiple leaves can span a single octant, or one leaf can contain multiple octants.

### 4.6 Octree Prefix Table (table_depth=6)

**Purpose**: O(1) position→leaf mapping using octree hierarchy

**Structure**:
```python
table_depth = 6  # Octree levels in prefix table
n_prefixes = 8^6 = 262,144  # Number of octree cells at level 6

prefix_start:  int32[262,144]  # First leaf containing this prefix
prefix_length: int32[262,144]  # Number of leaves with this prefix
```

**How It Works**:
```python
def position_to_leaf_id_octree(pos):
    # 1. Encode position to Morton code
    morton = encode(pos)

    # 2. Extract top 6 levels (18 bits)
    prefix = morton >> 45

    # 3. Lookup leaf range O(1)
    first_leaf = prefix_start[prefix]
    num_leaves = prefix_length[prefix]

    # 4. Search within leaf range (typically 1-8 leaves)
    for i in range(num_leaves):
        leaf_id = first_leaf + i
        if leaf_contains_morton(leaf_id, morton):
            return leaf_id
```

**Why It's Fast**:
- No binary search (O(log N) → O(1))
- Prefix table lookup: Single array access
- Refinement search: Only 1-8 leaves (unrolled loop)

**Memory**: 262,144 × 2 arrays × 4 bytes = **2 MB** (negligible)

---

## 5. Optimization Option A: Adaptive Search Radius

### 5.1 Concept

**Key Insight**: Graded refinement causes **variable Morton distances** between particle position and containing element centroid.

- **Coarse region**: Element size ~1mm → Morton distance ~10 leaves
- **Fine region**: Element size ~0.14mm → Morton distance ~50+ leaves (due to graded buffer)

**Solution**: Use **larger search radius in refined regions**, smaller radius in coarse regions.

### 5.2 Implementation Strategy

**Method 1: Bounding Box Detection**

```python
# In production_tracking_fully_fused_timedep.py
REFINED_REGION_BBOX_MIN = np.array([-0.010, -0.010, -0.0046], dtype=np.float32)  # -10mm, -10mm, -4.6mm
REFINED_REGION_BBOX_MAX = np.array([0.010, 0.010, -0.0002], dtype=np.float32)   # +10mm, +10mm, -0.2mm

L2_SEARCH_RADIUS_COARSE = 10   # Fast for coarse regions
L2_SEARCH_RADIUS_REFINED = 50  # Accurate for refined regions
```

**Modify RK4 integrator**:
```python
# In jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py

def is_in_refined_region(pos: jax.Array) -> jnp.bool_:
    """Check if position is in refined region bounding box."""
    in_x = (pos[0] >= refined_bbox_min[0]) & (pos[0] <= refined_bbox_max[0])
    in_y = (pos[1] >= refined_bbox_min[1]) & (pos[1] <= refined_bbox_max[1])
    in_z = (pos[2] >= refined_bbox_min[2]) & (pos[2] <= refined_bbox_max[2])
    return in_x & in_y & in_z

def search_l2_single_adaptive(pos: jax.Array) -> jax.Array:
    """L2 search with adaptive radius based on position."""
    in_refined = is_in_refined_region(pos)
    radius = jnp.where(in_refined,
                       jnp.int32(L2_SEARCH_RADIUS_REFINED),
                       jnp.int32(L2_SEARCH_RADIUS_COARSE))
    return search_L2_global_morton_single(pos, mesh_gpu_morton, radius)
```

**Advantages**:
- ✅ Simple to implement (10 lines of code)
- ✅ Minimal performance overhead (single bbox check)
- ✅ Predictable: Refined region is known a priori

**Disadvantages**:
- ❌ Requires manual bounding box specification
- ❌ Not adaptive to unknown refinement patterns
- ❌ Wastes searches at refined region edges

**Expected Performance**:
```
Baseline L0+L2(100): 30K particles/s, 30s/timestep

Adaptive L0+L2(10/50):
  Coarse region (80% of domain): radius=10 (fast)
  Refined region (20% of domain): radius=50 (accurate)

  Weighted average: 0.8 × fast + 0.2 × medium

  Estimated: 60-80K particles/s, 10-15s/timestep

  Speedup: 2-3× over uniform radius=100
```

---

### 5.3 Method 2: Density-Based Adaptive Radius

**Concept**: Use local element density as proxy for refinement level.

**Implementation**:
```python
def compute_local_density(pos: jax.Array, mesh_gpu: MeshGPUGlobalMorton) -> jnp.int32:
    """Estimate element density at position using prefix table."""
    # 1. Encode position to Morton code
    morton = morton_encode_position_jax(pos, mesh_gpu.bbox_min,
                                        mesh_gpu.bbox_max, mesh_gpu.max_depth)

    # 2. Extract coarse prefix (e.g., level 4 = 12 bits)
    coarse_prefix = morton >> (63 - 12)

    # 3. Count elements in this coarse octant
    # (Requires precomputed density map on GPU)
    density = density_map[coarse_prefix]

    return density

def adaptive_search_radius_density(pos: jax.Array, mesh_gpu: MeshGPUGlobalMorton) -> jnp.int32:
    """Adaptive radius based on local element density."""
    density = compute_local_density(pos, mesh_gpu)

    # High density = highly refined = need large radius
    radius = jnp.where(density > 5000, jnp.int32(50),   # Very refined
             jnp.where(density > 1000, jnp.int32(25),   # Medium refined
                                        jnp.int32(10)))  # Coarse
    return radius
```

**Precomputation** (CPU, during mesh loading):
```python
def build_density_map(morton_struct: GlobalMortonStructure, depth: int = 4):
    """Build element density map at octree level 'depth'."""
    n_prefixes = 8 ** depth  # e.g., 8^4 = 4096 for depth=4
    density_map = np.zeros(n_prefixes, dtype=np.int32)

    for elem_morton in morton_struct.morton_sorted:
        prefix = elem_morton >> (63 - depth * 3)
        density_map[prefix] += 1

    return density_map
```

**Advantages**:
- ✅ Fully adaptive (no manual bbox)
- ✅ Works for arbitrary refinement patterns
- ✅ Small memory overhead (4096 × 4 bytes = 16 KB for depth=4)

**Disadvantages**:
- ❌ More complex implementation
- ❌ Slight overhead (density lookup per search)
- ❌ Coarse density estimation (level 4-5 octants are large)

---

### 5.4 Method 3: Element Size Check (Hybrid L1+L2)

**Concept**: Use L1 search, but **verify element size** before accepting result.

**Implementation**:
```python
def search_l0_l1_l2_with_size_check(pos: jax.Array, cached_elem_id: jax.Array) -> jax.Array:
    """L1 search with element size verification, fallback to L2 with large radius."""
    # L0: Cached element
    elem_l0 = search_l0_single(pos, cached_elem_id)
    found_l0 = elem_l0 >= 0

    # L1: Neighbor hops
    elem_l1 = jnp.where(found_l0, elem_l0, search_l1_single(pos, cached_elem_id))
    found_l1 = elem_l1 >= 0

    # Size verification: Check if found element is suspiciously large
    elem_size = compute_element_size(elem_l1, mesh_gpu.connectivity, mesh_gpu.node_positions)
    expected_max_size = 0.0005  # 0.5mm threshold (fine/medium boundary)
    size_ok = elem_size <= expected_max_size

    # If L1 found element but it's too large, force L2 with large radius
    needs_l2 = found_l1 & (~size_ok)
    elem_l2 = jnp.where(needs_l2,
                        search_L2_global_morton_single(pos, mesh_gpu_morton, radius=50),
                        elem_l1)

    # Final fallback: L2 if L1 failed entirely
    elem_final = jnp.where(~found_l1,
                           search_L2_global_morton_single(pos, mesh_gpu_morton, radius=10),
                           elem_l2)

    return elem_final
```

**Element Size Computation**:
```python
@jax.jit
def compute_element_size(elem_id: jnp.int32,
                         connectivity: jax.Array,
                         node_positions: jax.Array) -> jnp.float32:
    """Compute max edge length of element (tet diameter proxy)."""
    nodes = connectivity[elem_id]
    p0, p1, p2, p3 = node_positions[nodes[0]], node_positions[nodes[1]], \
                     node_positions[nodes[2]], node_positions[nodes[3]]

    # Compute all 6 edge lengths
    edges = [
        jnp.linalg.norm(p1 - p0),
        jnp.linalg.norm(p2 - p0),
        jnp.linalg.norm(p3 - p0),
        jnp.linalg.norm(p2 - p1),
        jnp.linalg.norm(p3 - p1),
        jnp.linalg.norm(p3 - p2),
    ]
    return jnp.max(jnp.array(edges))
```

**Advantages**:
- ✅ Best of both worlds: Fast L1 + accurate L2 fallback
- ✅ Automatic detection of graded refinement boundaries
- ✅ No manual configuration needed

**Disadvantages**:
- ❌ Overhead: Element size computation for every L1 success
- ❌ Requires element size threshold tuning (0.5mm in example)
- ❌ More complex code

---

### 5.5 Recommendation: Method 1 (Bounding Box)

**For your specific case** (friction stir welding with known tool location):

**Pros**:
- Refined region location is **fixed and known** (rotating tool at origin)
- Simplest implementation (5 minutes to code)
- Zero performance overhead outside refined region
- Predictable and debuggable

**Implementation Steps**:

1. Add bbox constants to `production_tracking_fully_fused_timedep.py`
2. Pass bbox to RK4 integrator
3. Modify `search_l2_single` to check bbox and select radius
4. Test with `L2_SEARCH_RADIUS_COARSE=10`, `L2_SEARCH_RADIUS_REFINED=50`

**Expected Result**:
- Correct trajectories (rotation visible)
- No particle loss
- **2-3× faster** than uniform radius=100 (10-15s/timestep vs 30s/timestep)

---

## 6. Optimization Option B: Multi-Scale Hierarchical Search

### 6.1 Concept

**Key Insight**: Morton code **encodes octree hierarchy**. We can exploit this by searching **coarse octree levels first**, then refining.

**Strategy**:
1. Search at coarse octree level (e.g., level 4) with small radius
2. If found, refine search within that octant's children at finer level
3. Repeat until reaching leaf level

**Analogy**: Binary search on spatial hierarchy instead of linear search on leaves.

---

### 6.2 Implementation: Two-Level Search

**Level 1: Coarse Search** (Octree level 4-5)

```python
def search_coarse_octree(pos: jax.Array, mesh_gpu: MeshGPUGlobalMorton) -> jnp.int32:
    """Search at coarse octree level to find approximate region."""
    # 1. Encode position to Morton code
    morton = morton_encode_position_jax(pos, mesh_gpu.bbox_min,
                                        mesh_gpu.bbox_max, mesh_gpu.max_depth)

    # 2. Extract coarse prefix (level 4 = 12 bits)
    coarse_depth = 4
    coarse_prefix = morton >> (63 - coarse_depth * 3)

    # 3. Map prefix to leaf range
    # Requires precomputed coarse_prefix_table
    first_leaf = coarse_prefix_start[coarse_prefix]
    num_leaves = coarse_prefix_length[coarse_prefix]

    # 4. Search within coarse region (typically 100-500 leaves)
    # Use radius=5 (only search local neighborhood within coarse octant)
    center_leaf = first_leaf + num_leaves // 2

    elem_id = search_in_leaf_and_neighbors(pos, center_leaf, mesh_gpu, radius=5)
    return elem_id
```

**Level 2: Fine Refinement** (If coarse search found element)

```python
def search_fine_refinement(pos: jax.Array,
                           coarse_elem_id: jnp.int32,
                           mesh_gpu: MeshGPUGlobalMorton) -> jnp.int32:
    """Refine search around coarse element to find finest containing element."""
    # 1. Get coarse element's Morton code
    coarse_elem_idx = jnp.searchsorted(mesh_gpu.elem_ids_sorted, coarse_elem_id)
    coarse_morton = mesh_gpu.morton_sorted[coarse_elem_idx]

    # 2. Extract fine prefix around coarse element (level 7 = 21 bits)
    fine_prefix_bits = 21
    coarse_prefix_bits = 12

    # Focus search on 8 child octants of coarse cell
    child_octant_mask = 0b111 << (63 - fine_prefix_bits)

    # 3. Search leaves within child octants (radius=2, very local)
    coarse_leaf = coarse_elem_idx // mesh_gpu.leaf_capacity
    elem_fine = search_in_leaf_and_neighbors(pos, coarse_leaf, mesh_gpu, radius=2)

    # 4. Return finest element found
    return jnp.where(elem_fine >= 0, elem_fine, coarse_elem_id)
```

**Combined Multi-Scale Search**:

```python
def search_l2_multiscale(pos: jax.Array, mesh_gpu: MeshGPUGlobalMorton) -> jnp.int32:
    """Two-level hierarchical search: coarse → fine refinement."""
    # Level 1: Coarse search (octree level 4, large region, small radius)
    elem_coarse = search_coarse_octree(pos, mesh_gpu)
    found_coarse = elem_coarse >= 0

    # Level 2: Fine refinement (octree level 7, small region, tiny radius)
    elem_fine = jnp.where(found_coarse,
                          search_fine_refinement(pos, elem_coarse, mesh_gpu),
                          jnp.int32(-1))

    # Fallback: Full search if both failed
    elem_fallback = jnp.where(elem_fine >= 0,
                              elem_fine,
                              search_L2_global_morton_single(pos, mesh_gpu, radius=50))

    return elem_fallback
```

---

### 6.3 Precomputation: Coarse Prefix Table

**Build during mesh loading**:

```python
def build_coarse_prefix_table(morton_struct: GlobalMortonStructure, coarse_depth: int = 4):
    """Build prefix table at coarse octree level for multi-scale search."""
    n_coarse_prefixes = 8 ** coarse_depth  # 4096 for depth=4

    coarse_prefix_start = np.full(n_coarse_prefixes, -1, dtype=np.int32)
    coarse_prefix_length = np.zeros(n_coarse_prefixes, dtype=np.int32)

    # Group leaves by their coarse prefix
    leaf_prefixes = morton_struct.morton_sorted[morton_struct.leaf_start] >> (63 - coarse_depth * 3)

    for prefix in range(n_coarse_prefixes):
        leaves_in_prefix = np.where(leaf_prefixes == prefix)[0]
        if len(leaves_in_prefix) > 0:
            coarse_prefix_start[prefix] = leaves_in_prefix[0]
            coarse_prefix_length[prefix] = len(leaves_in_prefix)

    return coarse_prefix_start, coarse_prefix_length
```

**Upload to GPU**:
```python
# Add to MeshGPUGlobalMorton dataclass
coarse_prefix_start: jax.Array   # (4096,) int32
coarse_prefix_length: jax.Array  # (4096,) int32
coarse_depth: jnp.int32          # 4
```

**Memory Overhead**: 4096 × 2 × 4 bytes = **32 KB** (negligible)

---

### 6.4 Performance Analysis

**Coarse Search** (Level 4):
- Octree cells at level 4: 8^4 = **4096 cells**
- Your domain: 60mm × 46mm × 10mm
- Cell size at level 4: ~15mm × 12mm × 2.5mm (large cells)
- Elements per coarse cell: 3M / 4096 ≈ **750 elements**
- Leaves per coarse cell: 750 / 256 ≈ **3 leaves**
- Search radius=5: **15 leaves** searched (3 center + 12 neighbors)
- Point-in-tet tests: 15 × 256 = **3,840 tests** (vs 51,200 for radius=100)

**Fine Refinement** (Level 7):
- Child octants: 8 children per coarse cell
- Leaves per child: ~0.4 leaves
- Search radius=2: **5 leaves** searched
- Point-in-tet tests: 5 × 256 = **1,280 tests**

**Total**: 3,840 + 1,280 = **5,120 tests** (10× fewer than radius=100!)

**Expected Performance**:
```
Baseline L0+L2(100): 30K particles/s (51,200 tests per search)

Multi-scale L2:
  Coarse + fine: 5,120 tests per search (10× reduction)

  Estimated: 150-200K particles/s, 4-6s/timestep

  Speedup: 5-6× over uniform radius=100
```

---

### 6.5 Advantages and Disadvantages

**Advantages**:
- ✅ **Best performance**: 10× fewer point-in-tet tests
- ✅ Exploits Morton octree hierarchy directly
- ✅ Fully adaptive (no manual configuration)
- ✅ Scales to arbitrary mesh sizes and refinement patterns

**Disadvantages**:
- ❌ **Complex implementation**: Requires coarse prefix table, two-level search logic
- ❌ Precomputation overhead (negligible, but adds code)
- ❌ **Fallback risk**: If coarse search misses, fallback is expensive
  - Mitigation: Make coarse radius=10 instead of 5 (safety margin)

---

### 6.6 Recommendation for Multi-Scale

**When to Use**:
- Large-scale production runs (amortize implementation effort)
- Unknown or complex refinement patterns
- Need maximum performance (target: <5s/timestep)

**Implementation Complexity**: **Medium-High** (2-3 hours of coding + testing)

**Alternative**: Start with Option A (bbox-based adaptive radius) for quick wins, then implement multi-scale if more performance is needed.

---

## 7. Optimization Option C: Prefix Table with Spatial Filtering

### 7.1 Current Prefix Table Implementation

**Your mesh already has** `table_depth = 6`:

```python
# In MeshGPUGlobalMorton
prefix_start:  jax.Array  # (8^6 = 262,144,) int32
prefix_length: jax.Array  # (262,144,) int32
table_depth:   jnp.int32  # 6
```

**How it works** (`jaxtrace/gpu/search/morton_global_search.py:207-282`):

```python
def position_to_leaf_id_octree(pos, mesh_gpu):
    """O(1) leaf lookup using prefix table."""
    # 1. Encode position to Morton code
    morton = morton_encode_position_jax(pos, bbox_min, bbox_max, max_depth)

    # 2. Extract prefix (top 18 bits for depth=6)
    prefix = morton >> (63 - 6*3)  # Right shift by 45 bits

    # 3. Get leaf range for this prefix (O(1) array lookup)
    first_leaf = prefix_start[prefix]
    num_leaves = prefix_length[prefix]

    # 4. Linear search within leaf range (typically 1-8 leaves)
    for offset in range(8):  # Unrolled JAX loop
        leaf_id = first_leaf + offset
        if offset < num_leaves and leaf_contains_morton(leaf_id, morton):
            return leaf_id
```

**Current Performance**: Already using O(1) prefix lookup instead of O(log N) binary search.

---

### 7.2 Problem: Prefix Table Alone Doesn't Solve Graded Refinement

**Why particle loss still occurs**:

```
Particle at (-10.000mm, 0, -2.000mm):
  → Morton M_p
  → Prefix P_p = M_p >> 45 (top 18 bits)
  → Leaf L_p via prefix_start[P_p]

Fine element centroid at (-10.050mm, 0, -2.030mm):
  → Morton M_e (DIFFERENT from M_p)
  → Prefix P_e = M_e >> 45
  → P_e ≠ P_p (graded refinement causes different level-6 octants)
  → Leaf L_e not searched (outside radius)
```

**Issue**: Prefix table maps position to **its own leaf**, not to **containing element's leaf**.

---

### 7.3 Optimization: Spatial Filtering with Prefix Table

**Concept**: Use prefix table to **quickly identify candidate octants** at level 6, then search all leaves within those octants.

**Implementation**:

```python
def get_neighbor_prefixes_3d(prefix: jnp.int32, depth: int) -> List[jnp.int32]:
    """Get 27 neighboring octant prefixes at given depth (3×3×3 cube)."""
    # Decode prefix to 3D octree coordinates
    mask_per_level = 0b111  # 3 bits
    x_coord, y_coord, z_coord = 0, 0, 0

    for level in range(depth):
        shift = (depth - 1 - level) * 3
        octant = (prefix >> shift) & mask_per_level
        x_coord = (x_coord << 1) | ((octant >> 0) & 1)
        y_coord = (y_coord << 1) | ((octant >> 1) & 1)
        z_coord = (z_coord << 1) | ((octant >> 2) & 1)

    # Get 27 neighbors (3×3×3 cube)
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                nx = x_coord + dx
                ny = y_coord + dy
                nz = z_coord + dz

                # Clamp to valid range [0, 2^depth)
                if 0 <= nx < (1 << depth) and \
                   0 <= ny < (1 << depth) and \
                   0 <= nz < (1 << depth):
                    # Encode back to prefix
                    neighbor_prefix = 0
                    for level in range(depth):
                        shift = depth - 1 - level
                        bit_x = (nx >> shift) & 1
                        bit_y = (ny >> shift) & 1
                        bit_z = (nz >> shift) & 1
                        octant = (bit_z << 2) | (bit_y << 1) | bit_x
                        neighbor_prefix = (neighbor_prefix << 3) | octant

                    neighbors.append(neighbor_prefix)

    return neighbors

def search_L2_with_prefix_filtering(
    pos: jax.Array,
    mesh_gpu: MeshGPUGlobalMorton,
    spatial_radius: int = 1  # Octant radius (1 = 3×3×3 = 27 neighbors)
) -> jnp.int32:
    """L2 search using prefix table with spatial octant neighborhood."""
    # 1. Get center prefix
    morton = morton_encode_position_jax(pos, mesh_gpu.bbox_min,
                                        mesh_gpu.bbox_max, mesh_gpu.max_depth)
    center_prefix = (morton >> (63 - mesh_gpu.table_depth * 3)).astype(jnp.int32)

    # 2. Get neighboring prefixes (27 octants for radius=1)
    neighbor_prefixes = get_neighbor_prefixes_3d(center_prefix, int(mesh_gpu.table_depth))

    # 3. Collect all leaves in these octants
    candidate_leaves = []
    for prefix in neighbor_prefixes:
        first_leaf = mesh_gpu.prefix_start[prefix]
        num_leaves = mesh_gpu.prefix_length[prefix]
        for offset in range(num_leaves):
            candidate_leaves.append(first_leaf + offset)

    # 4. Search all candidate leaves
    found_elem = jnp.int32(-1)
    for leaf_id in candidate_leaves:
        if found_elem < 0:
            found_elem = search_in_leaf_global(pos, leaf_id, mesh_gpu)

    return found_elem
```

---

### 7.4 JAX-Compatible Implementation

**Problem**: `get_neighbor_prefixes_3d` uses Python loops and lists (not JAX-compatible).

**Solution**: Precompute neighbor table on CPU, upload to GPU.

```python
def build_prefix_neighbor_table(table_depth: int):
    """Precompute 27-neighbor lookup table for all prefixes."""
    n_prefixes = 8 ** table_depth
    max_neighbors = 27  # 3×3×3 cube

    neighbor_table = np.full((n_prefixes, max_neighbors), -1, dtype=np.int32)
    neighbor_count = np.zeros(n_prefixes, dtype=np.int32)

    for prefix in range(n_prefixes):
        neighbors = get_neighbor_prefixes_3d(prefix, table_depth)
        neighbor_count[prefix] = len(neighbors)
        neighbor_table[prefix, :len(neighbors)] = neighbors

    return neighbor_table, neighbor_count

# Upload to GPU
mesh_gpu.prefix_neighbors = jax.device_put(neighbor_table)  # (262144, 27) int32
mesh_gpu.prefix_neighbor_count = jax.device_put(neighbor_count)  # (262144,) int32
```

**Memory**: 262,144 × 27 × 4 bytes = **28 MB** (acceptable)

**JAX Search Function**:

```python
def search_L2_prefix_spatial(pos: jax.Array, mesh_gpu: MeshGPUGlobalMorton) -> jnp.int32:
    """L2 search using precomputed prefix neighbor table."""
    # 1. Get center prefix
    morton = morton_encode_position_jax(pos, mesh_gpu.bbox_min,
                                        mesh_gpu.bbox_max, mesh_gpu.max_depth)
    center_prefix = (morton >> (63 - mesh_gpu.table_depth * 3)).astype(jnp.int32)

    # 2. Get neighbor prefixes (O(1) lookup)
    neighbor_prefixes = mesh_gpu.prefix_neighbors[center_prefix]  # (27,)
    num_neighbors = mesh_gpu.prefix_neighbor_count[center_prefix]

    # 3. Collect candidate leaves from all neighbor octants
    def collect_leaves_from_prefix(i, candidate_leaves):
        """Add leaves from i-th neighbor prefix."""
        active = i < num_neighbors
        prefix = jnp.where(active, neighbor_prefixes[i], jnp.int32(0))

        first_leaf = mesh_gpu.prefix_start[prefix]
        num_leaves = mesh_gpu.prefix_length[prefix]

        # Search each leaf in this prefix (max 8 leaves per prefix)
        for j in range(8):
            leaf_active = active & (j < num_leaves)
            leaf_id = jnp.where(leaf_active, first_leaf + j, jnp.int32(-1))

            # Search in leaf if active
            elem = jnp.where(leaf_active & (candidate_leaves[0] < 0),
                             search_in_leaf_global(pos, leaf_id, mesh_gpu),
                             jnp.int32(-1))

            # Update candidate if found
            candidate_leaves = jnp.where(elem >= 0,
                                         jnp.array([elem]),
                                         candidate_leaves)

        return candidate_leaves

    # 4. Search all 27 neighbor prefixes
    init_candidates = jnp.array([-1], dtype=jnp.int32)
    final_elem = lax.fori_loop(0, 27, collect_leaves_from_prefix, init_candidates)

    return final_elem[0]
```

---

### 7.5 Performance Analysis

**Spatial Radius = 1** (3×3×3 = 27 octant neighbors):

- Level-6 octant size: Domain / 2^6 ≈ 60mm / 64 ≈ **1mm per octant**
- Graded refinement: Fine element at (-10.05mm, 0, -2.03mm) vs particle at (-10.00mm, 0, -2.00mm)
  - Distance: ~50μm
  - Octant distance: 50μm / 1mm = **0.05 octants** → same or adjacent octant
- **27 neighbor octants cover ±1 octant in each dimension** = ±1mm range
- **Sufficient** for graded refinement (medium buffer is <0.5mm)

**Leaves Searched**:
- 27 octants × ~3 leaves/octant = **~80 leaves**
- Point-in-tet tests: 80 × 256 = **20,480 tests**

**Comparison**:
- L2 radius=100: 200 leaves, 51,200 tests
- Prefix spatial radius=1: 80 leaves, 20,480 tests
- **Reduction**: 2.5× fewer tests

**Expected Performance**:
```
Baseline L0+L2(100): 30K particles/s

Prefix spatial:
  Estimated: 70-90K particles/s, 8-12s/timestep

  Speedup: 2.5-3× over uniform radius=100
```

---

### 7.6 Advantages and Disadvantages

**Advantages**:
- ✅ Exploits existing prefix table (table_depth=6 already available)
- ✅ Spatially-aware: Searches **geometric neighborhood**, not arbitrary Morton range
- ✅ Moderate complexity (precompute neighbor table, modify search)
- ✅ Tunable: Can increase spatial_radius if needed (27 neighbors → 125 for radius=2)

**Disadvantages**:
- ❌ Precomputation required (neighbor table)
- ❌ Memory overhead: 28 MB for neighbor table
- ❌ Less performant than multi-scale (Option B)

**Recommendation**:
- **Good middle ground** between Option A (simple bbox) and Option B (complex multi-scale)
- **Best if** you want automatic adaptation without multi-scale complexity

---

### 7.7 Tuning Spatial Radius

**Spatial Radius = 1** (27 neighbors):
- Coverage: ±1mm in each dimension
- Sufficient for most cases

**Spatial Radius = 2** (125 neighbors):
- Coverage: ±2mm in each dimension
- Handles extreme graded refinement or large coarse elements
- ~3× more leaves searched

**Adaptive Spatial Radius**:
```python
def adaptive_spatial_radius(pos: jax.Array, mesh_gpu: MeshGPUGlobalMorton) -> jnp.int32:
    """Use larger spatial radius in refined regions."""
    in_refined = is_in_refined_region(pos)
    return jnp.where(in_refined, jnp.int32(2), jnp.int32(1))
```

---

## 8. Implementation Recommendations

### 8.1 Quick Win: Option A (Adaptive Radius by Bounding Box)

**Timeline**: 30 minutes

**Implementation**:

1. **Edit `production_tracking_fully_fused_timedep.py`**:
   ```python
   # After line 47 (search configuration)

   # Refined region bounding box (rotating tool area)
   REFINED_REGION_BBOX = {
       'min': np.array([-0.010, -0.010, -0.0046], dtype=np.float32),  # -10mm
       'max': np.array([0.010, 0.010, -0.0002], dtype=np.float32)     # +10mm
   }

   # Adaptive search radius
   L2_SEARCH_RADIUS_COARSE = 10
   L2_SEARCH_RADIUS_REFINED = 50
   ```

2. **Edit `jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py`**:

   Add after line 150 (before search functions):
   ```python
   # Adaptive L2 search radius
   refined_bbox_min = jnp.array(refined_region_bbox['min'])
   refined_bbox_max = jnp.array(refined_region_bbox['max'])
   l2_radius_coarse = jnp.int32(l2_search_radius_coarse)
   l2_radius_refined = jnp.int32(l2_search_radius_refined)

   def is_in_refined_region(pos: jax.Array) -> jnp.bool_:
       """Check if position is in refined region."""
       in_x = (pos[0] >= refined_bbox_min[0]) & (pos[0] <= refined_bbox_max[0])
       in_y = (pos[1] >= refined_bbox_min[1]) & (pos[1] <= refined_bbox_max[1])
       in_z = (pos[2] >= refined_bbox_min[2]) & (pos[2] <= refined_bbox_max[2])
       return in_x & in_y & in_z
   ```

   Modify `search_l2_single` (around line 200):
   ```python
   def search_l2_single(pos: jax.Array) -> jax.Array:
       """L2 search with adaptive radius."""
       # Check if in refined region
       in_refined = is_in_refined_region(pos)
       radius = jnp.where(in_refined, l2_radius_refined, l2_radius_coarse)

       # Search with adaptive radius
       return search_L2_global_morton_single(
           pos,
           mesh_gpu_global_morton,
           search_radius=radius
       )
   ```

3. **Test**:
   ```bash
   python production_tracking_fully_fused_timedep.py
   ```

**Expected Results**:
- Throughput: 60-80K particles/s (2-3× improvement)
- Time per timestep: 10-15s (vs 30s baseline)
- Particle retention: >99%
- Rotation visible: ✅

---

### 8.2 Medium-Term: Option C (Prefix Spatial Filtering)

**Timeline**: 3-4 hours

**When**: After Option A proves concept, if more performance needed

**Implementation Steps**:

1. **Add neighbor table builder** to `jaxtrace/gpu/search/morton_global_builder.py`

2. **Modify `GlobalMortonStructure`** to include neighbor table

3. **Update upload function** in `morton_global_search.py`

4. **Implement `search_L2_prefix_spatial`** function

5. **Add config parameter** `USE_PREFIX_SPATIAL = True` in production script

**Expected Results**:
- Throughput: 70-90K particles/s
- Time per timestep: 8-12s
- Fully adaptive (no bbox needed)

---

### 8.3 Long-Term: Option B (Multi-Scale Hierarchical)

**Timeline**: 1-2 days

**When**: Production runs with strict performance requirements (<5s/timestep)

**Implementation Steps**:

1. **Build coarse prefix table** (depth=4) during mesh preprocessing

2. **Implement two-level search**: coarse → fine refinement

3. **Add fallback logic** for missed searches

4. **Extensive testing** on various mesh regions

**Expected Results**:
- Throughput: 150-200K particles/s
- Time per timestep: 4-6s
- Best performance, highest complexity

---

### 8.4 Testing and Validation

**For each implementation**:

1. **Diagnostic Test**:
   ```bash
   python diagnose_tracking_through_refined_region.py
   ```
   - Verify >90% fine element assignment in refined region
   - Check rotation velocities are captured

2. **Production Test**:
   ```bash
   python production_tracking_fully_fused_timedep.py
   ```
   - Monitor throughput (particles/s)
   - Check particle retention (>99%)
   - Verify GPU memory usage (<2GB)

3. **Visual Validation**:
   - Export VTK, load in ParaView
   - Compare with commercial code screenshots
   - Check for swirling trajectories in refined region

4. **Performance Profiling**:
   ```python
   # Add timing instrumentation
   import time

   t0 = time.time()
   state = rk4_step(state, dt)
   t1 = time.time()

   print(f"RK4 step: {(t1-t0)*1000:.2f}ms, {n_active/(t1-t0):.0f} particles/s")
   ```

---

### 8.5 Summary Table

| Option | Complexity | Performance | Memory | When to Use |
|--------|------------|-------------|--------|-------------|
| **A: Bbox Adaptive** | ⭐ Low | ⭐⭐ 2-3× faster | ✅ None | **Start here** - Quick win |
| **C: Prefix Spatial** | ⭐⭐ Medium | ⭐⭐⭐ 3-4× faster | 28 MB | Unknown refinement patterns |
| **B: Multi-Scale** | ⭐⭐⭐ High | ⭐⭐⭐⭐ 5-6× faster | 32 KB | Production at scale |

---

### 8.6 Recommended Implementation Sequence

**Week 1: Option A** (30 min implementation + testing)
- Implement bbox-based adaptive radius
- Validate correctness (rotation visible, no particle loss)
- Measure performance improvement

**If performance insufficient** → **Week 2: Option C** (half-day implementation)
- Build prefix neighbor table
- Implement spatial filtering search
- Compare performance to Option A

**If still insufficient** → **Week 3: Option B** (2-day implementation)
- Build coarse prefix table (depth=4)
- Implement two-level hierarchical search
- Optimize for production workloads

---

## Appendix: Morton Code Bit Layout Diagram

```
63-bit Morton Code (21 bits per dimension):

Bit position:    62 61 60 | 59 58 57 | 56 55 54 | ... | 5 4 3 | 2 1 0
Dimension:        z  y  x |  z  y  x |  z  y  x | ... | z y x | z y x
Octree level:    ├─Level 1─┤├─Level 2─┤├─Level 3─┤ ... ├─L20─┤├─L21─┤

Example:
  Level 1 (bits 62-60): [z=0, y=1, x=1] → Octant 6 (binary 110)
  Level 2 (bits 59-57): [z=1, y=0, x=0] → Octant 4 (binary 100)
  Level 3 (bits 56-54): [z=1, y=1, x=1] → Octant 7 (binary 111)
  ...

Octree Path Interpretation:
  Top 3 bits  → Which of 8 root octants
  Next 3 bits → Which of 8 sub-octants within parent
  Next 3 bits → Which of 8 sub-sub-octants
  ...
  Bottom 3 bits → Finest level (21st subdivision)

Prefix Table (table_depth=6):
  Top 18 bits (6 levels × 3 bits) → 2^18 = 262,144 octree cells
  Each cell → List of Morton leaves intersecting that cell

  Example prefix: 0b000110100011111101 (18 bits)
                   └──┘└──┘└──┘└──┘└──┘└──┘
                   Lv1  Lv2  Lv3  Lv4  Lv5  Lv6
```

---

## Appendix: Element Size Distribution in Your Mesh

From spatial analysis (`diagnose_coarse_fine_spatial_relationship.py`):

```
Element Count by Size:
  Fine (≤0.15mm):     2,599,528 elements (85.3%)
    - Rotating tool region: X=[-9.36, 9.34]mm
    - Captures high-velocity gradients
    - Critical for rotation tracking

  Medium (0.15-0.30mm): 381,987 elements (12.5%)
    - Transition buffer: X=[-9.65, 9.61]mm
    - Surrounds fine region
    - Prevents direct coarse-fine contact

  Coarse (>0.30mm):      67,385 elements (2.2%)
    - Outer domain: X=[-28.75, 28.75]mm
    - Lower velocity gradients
    - Global advection region

Morton Leaf Distribution:
  Total leaves: ~11,910 (256 elements/leaf)

  Leaves in fine region: ~10,150 (85%)
  Leaves in medium region: ~1,490 (12.5%)
  Leaves in coarse region: ~270 (2.5%)

  → Fine region is DENSE in Morton space (many leaves)
  → Coarse region is SPARSE (few leaves)
  → Graded refinement creates Morton GAPS (medium buffer)
```

---

**End of Document**
