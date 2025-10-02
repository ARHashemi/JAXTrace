# FEM Interpolation Workflow - Detailed Explanation

## Overview

This document explains how FEM (Finite Element Method) interpolation works in JAXTrace, focusing on what gets computed once vs what happens per velocity evaluation.

---

## The Problem We're Solving

**Goal**: Given a query point `(x, y, z)` and time `t`, compute the velocity at that point.

**Challenge**:
- We have velocity data at **mesh nodes** (185,865 nodes in your case)
- Query point is usually **NOT** at a node location
- Need to **interpolate** velocity from nearby nodes

**Why Not Nearest-Neighbor?**
- ❌ **Inaccurate**: Finds closest node, which may be in a different element
- ❌ **Discontinuous**: Jumps between node values
- ❌ **Wrong for refined meshes**: With 6 refinement levels, nearest node may belong to wrong element

**FEM Interpolation Solution**:
- ✅ Find the **element** (tetrahedron) containing the query point
- ✅ Use **barycentric coordinates** to interpolate from the 4 nodes of that element
- ✅ **Physically correct**: This is exactly how FEM represents the solution

---

## Data Structures

### 1. Mesh Data (Computed Once, Never Changes)

#### a) **Node Positions** - `points: (N, 3)`
```python
# Example:
points[0] = [0.01, 0.005, -0.002]  # Position of node 0
points[1] = [0.012, 0.006, -0.0021]  # Position of node 1
# ... 185,865 nodes total
```
- **What**: 3D coordinates of every node in the mesh
- **When**: Loaded once from VTK file
- **Used**:
  - Once: Build spatial index (octree/grid)
  - Per query: Get node coordinates for interpolation

#### b) **Connectivity** - `connectivity: (M, 4)`
```python
# Example:
connectivity[0] = [142, 573, 891, 1024]  # Element 0 uses nodes 142, 573, 891, 1024
connectivity[1] = [573, 891, 1024, 1099]  # Element 1 uses nodes 573, 891, 1024, 1099
# ... 750,773 elements total
```
- **What**: Defines which 4 nodes form each tetrahedral element
- **When**: Loaded once from VTK file
- **Used**: Per query to get the 4 nodes of the containing element

#### c) **Element Bounds** - `element_bounds: (M, 2, 3)`
```python
# Example:
element_bounds[0, 0] = [0.01, 0.005, -0.002]  # Min corner of element 0
element_bounds[0, 1] = [0.015, 0.008, -0.0015]  # Max corner of element 0
```
- **What**: Bounding box (min/max corners) for each element
- **When**: Computed once during octree/grid building
- **Used**: Per query for quick rejection test (is point possibly in this element?)

---

### 2. Octree Structure (Computed Once)

The octree is a **hierarchical spatial index** that subdivides space recursively.

#### Why Octree Instead of Uniform Grid?

**Your mesh has 6 refinement levels**, meaning element sizes vary dramatically:
- Coarse elements: ~1mm in some regions
- Fine elements: ~0.01mm in refined regions (100x smaller!)

**Uniform Grid Problems**:
- Too coarse → Many elements per cell (39,933 in your case!)
- Too fine → Memory explosion

**Octree Solution**: Adapts to mesh density
- **Coarse cells** where elements are large
- **Fine cells** where elements are small
- **Result**: 7.1 elements per leaf on average (vs 39,933!)

#### Octree Components:

##### a) **Octree Nodes** - `nodes_min, nodes_max: (num_nodes, 3)`
```python
# Example root node (covers entire domain):
nodes_min[0] = [-0.03, -0.02, -0.008]  # Domain minimum
nodes_max[0] = [0.07, 0.02, 0.0]       # Domain maximum

# Example leaf node (small refined region):
nodes_min[52341] = [0.01, 0.005, -0.002]   # Small box in refined area
nodes_max[52341] = [0.011, 0.006, -0.0015]  # Only 1mm x 1mm x 0.5mm
```
- **What**: Bounding boxes for each octree node
- **Count**: 120,873 total nodes (105,261 leaves)
- **When**: Built once during initialization (~60 seconds)
- **Used**: Per query for tree traversal

##### b) **Octree Children** - `nodes_children: (num_nodes, 8)`
```python
# Example internal node:
nodes_children[0] = [1, 2, 3, 4, 5, 6, 7, 8]  # Root has 8 children

# Example leaf node:
nodes_children[52341] = [-1, -1, -1, -1, -1, -1, -1, -1]  # No children (leaf)
```
- **What**: Indices of 8 child nodes (or -1 if no child in that octant)
- **When**: Built once
- **Used**: Per query for tree traversal

##### c) **Octree Elements** - `nodes_elements, nodes_elem_counts: (num_nodes, max_elems)`
```python
# Example leaf node containing 7 elements:
nodes_elements[52341] = [142573, 142574, 142575, 142576, 142577, 142578, 142579, -1, -1, ...]
nodes_elem_counts[52341] = 7  # Only first 7 are valid

# Example empty internal node:
nodes_elements[0] = [-1, -1, -1, ...]  # Internal nodes store no elements
nodes_elem_counts[0] = 0
```
- **What**: Element indices stored in each leaf node
- **When**: Built once
- **Used**: Per query to get candidate elements

---

### 3. Field Data (Per Timestep)

#### **Velocity Data** - `data: (T, N, 3)`
```python
# Example:
data[0, :, :] = velocities at time t=0 for all N nodes
data[1, :, :] = velocities at time t=1 for all N nodes
# ...
data[39, :, :] = velocities at time t=39 for all N nodes
```
- **What**: Velocity (stored as "Displacement") at each node for each timestep
- **Shape**: (40 timesteps, 185,865 nodes, 3 components)
- **When**: Loaded once from all VTK files
- **Used**: Per query after temporal interpolation

---

## Workflow

### Phase 1: One-Time Setup (Happens Once at Start)

```
1. Load VTK Files (40 timesteps)
   ├─> Extract node positions: points (185,865, 3)
   ├─> Extract connectivity: connectivity (750,773, 4)
   └─> Extract velocity data: data (40, 185,865, 3)
       Time: ~30 seconds

2. Build Octree
   ├─> Compute element bounds for all 750,773 elements
   ├─> Recursively subdivide space:
   │   ├─> Split node if > 32 elements
   │   ├─> Stop at depth 12
   │   └─> Result: 120,873 nodes, 105,261 leaves
   └─> Convert to JAX arrays for GPU
       Time: ~60 seconds

3. Upload to GPU
   ├─> points → GPU
   ├─> connectivity → GPU
   ├─> element_bounds → GPU
   ├─> octree structure → GPU
   └─> velocity data → GPU
       GPU Memory: ~100-200 MB

4. JIT Compile Interpolation Function
   └─> Compile octree traversal + barycentric interpolation
       Time: ~2 seconds (first query only)
```

**Total One-Time Cost**: ~90 seconds

---

### Phase 2: Per Velocity Evaluation (Happens Millions of Times)

For each query point `(x, y, z)` at time `t`:

```
┌─────────────────────────────────────────────┐
│ INPUT: query_point = (x, y, z), time = t   │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ STEP 1: Temporal Interpolation              │
│                                             │
│ - Find timesteps surrounding t             │
│ - Interpolate velocity at ALL nodes        │
│ - Result: field_at_nodes (185,865, 3)     │
│                                             │
│ Cost: ~0.001ms (linear interpolation)      │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ STEP 2: Octree Traversal                    │
│                                             │
│ Start at root (node 0)                      │
│ While not at leaf:                          │
│   - Compare query_point to node center     │
│   - Determine octant (0-7)                  │
│   - Go to child in that octant             │
│                                             │
│ Result: leaf_node containing query point    │
│                                             │
│ Cost: ~0.01ms (traverse ~8 levels)         │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ STEP 3: Get Candidate Elements              │
│                                             │
│ - Read nodes_elements[leaf_node]           │
│ - Typically 7 elements (avg)               │
│ - Max 32 elements (by design)              │
│                                             │
│ Cost: negligible (array lookup)            │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ STEP 4: Find Containing Element             │
│                                             │
│ For each candidate element:                │
│   1. Quick bounds check:                    │
│      - Is point in element bounding box?   │
│   2. If yes, precise check:                 │
│      - Get 4 node positions                │
│      - Compute barycentric coordinates      │
│      - Check if all coords in [0, 1]       │
│   3. If inside, STOP (found it!)           │
│                                             │
│ Cost: ~0.05ms (check ~7 elements)          │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ STEP 5: Barycentric Interpolation           │
│                                             │
│ - Have: 4 barycentric coords [b0,b1,b2,b3] │
│ - Get velocity at 4 nodes:                 │
│   v0 = field_at_nodes[node0]               │
│   v1 = field_at_nodes[node1]               │
│   v2 = field_at_nodes[node2]               │
│   v3 = field_at_nodes[node3]               │
│                                             │
│ - Interpolate:                              │
│   velocity = b0*v0 + b1*v1 + b2*v2 + b3*v3 │
│                                             │
│ Cost: ~0.001ms (4 dot products)            │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ OUTPUT: velocity = (vx, vy, vz)            │
└─────────────────────────────────────────────┘

TOTAL PER-QUERY COST: ~0.1-0.5ms
```

---

## Mathematical Details

### Barycentric Coordinates

For a tetrahedron with vertices `v0, v1, v2, v3`:

Any point `p` inside can be expressed as:
```
p = b0·v0 + b1·v1 + b2·v2 + b3·v3
```

Where:
- `b0 + b1 + b2 + b3 = 1`
- `bᵢ ∈ [0, 1]` if point is inside

**Computation** (using Cramer's rule):
```python
# Build matrix from edge vectors
M = [v1-v0, v2-v0, v3-v0]  # 3x3 matrix
det_M = det(M)              # Tetrahedron volume × 6

# Solve for each coordinate
b1 = det([p-v0, v2-v0, v3-v0]) / det_M
b2 = det([v1-v0, p-v0, v3-v0]) / det_M
b3 = det([v1-v0, v2-v0, p-v0]) / det_M
b0 = 1 - b1 - b2 - b3
```

**Interpolation**:
```python
# FEM uses same shape functions as barycentric coordinates
velocity(p) = b0·v0 + b1·v1 + b2·v2 + b3·v3
```

This is **exactly** how FEM represents the solution inside an element!

---

## Performance Summary

### What's Computed Once (Setup):

| Operation | Time | Memory |
|-----------|------|--------|
| Load VTK files | ~30s | ~85 MB |
| Build octree | ~60s | ~50 MB |
| Upload to GPU | ~1s | ~150 MB |
| JIT compile | ~2s | - |
| **Total** | **~93s** | **~150 MB** |

### What's Computed Per Query:

| Operation | Time per Query | Operations |
|-----------|---------------|------------|
| Temporal interpolation | ~0.001ms | 2 array lookups + lerp |
| Octree traversal | ~0.01ms | ~8 node visits |
| Candidate filtering | ~0.05ms | ~7 bound checks + containment tests |
| Barycentric interpolation | ~0.001ms | 4 dot products |
| **Total** | **~0.1ms** | **All on GPU** |

### Batch Performance:

For 7,500 particles:
- **Per timestep**: 7,500 queries × 0.1ms = 0.75s
- **For 1,500 timesteps**: 1,500 × 0.75s = **~1,125s** (~19 minutes)

**Note**: Current implementation is slower due to non-optimized octree traversal.

---

## Why Current Implementation is Slow

**Problem**: The `while_loop` for octree traversal doesn't vectorize well across many queries.

**Solution** (to be implemented):
1. Pre-compute traversal paths
2. Use fixed-depth traversal (no loops)
3. Or switch to simpler KD-tree with distance-based filtering

---

## Comparison: Octree vs Uniform Grid

| Aspect | Uniform Grid (256³) | Octree |
|--------|---------------------|--------|
| Total cells/nodes | 16,777,216 cells | 120,873 nodes |
| Elements per cell/leaf | 393 (avg) | 7.1 (avg) |
| Memory | ~100 MB | ~50 MB |
| Build time | ~30s | ~60s |
| Query time (theoretical) | ~0.05ms | ~0.1ms |
| **Best for** | **Uniform meshes** | **Refined meshes** ✅ |

**For your 6-level refined mesh, octree is the right choice.**

---

## Next Steps

1. ✅ Octree implementation complete
2. ⏳ Optimize octree traversal for batch queries
3. ⏳ Test full workflow
4. ⏳ Benchmark vs nearest-neighbor

Your mesh with 6 refinement levels **requires** FEM interpolation with spatial indexing. Octree is the correct approach, but needs optimization for batch queries.
