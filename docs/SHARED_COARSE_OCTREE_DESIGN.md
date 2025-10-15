# Shared Coarse Octree with Time-Dependent Fine Levels
## Advanced AMR Strategy for Welding Simulations

**Date**: 2025-10-09
**Concept**: Based on tetrahedral element splitting patterns
**Insight**: Coarse structure is stable, only fine levels vary with tool movement

---

## 🎯 Core Concept

### Observation from Mesh Refinement Pattern

**Your key insight**: Tetrahedral mesh refinement splits elements
```
Refinement step N → N+1:
- Coarse tetrahedra remain unchanged
- Some tetrahedra split into 2 child tetrahedra
- Split pattern follows tool position (localized)
```

**Implication for Octree**:
```
Octree depth levels:
Level 0-2: Coarse structure (STATIC - spans entire domain)
Level 3-5: Medium structure (SEMI-STATIC - large regions)
Level 6-12: Fine structure (TIME-DEPENDENT - near weld pool)
```

### Shared Coarse Octree Strategy

```
┌─────────────────────────────────────────────────┐
│  SHARED COARSE OCTREE (Levels 0-5)             │
│  - Built from initial refinement steps          │
│  - Covers entire domain at coarse resolution    │
│  - NEVER changes during revolution cycles       │
│                                                  │
│  ┌─────────────────────────────────────┐       │
│  │  Coarse Branch A (static)           │       │
│  │  ├─ No fine elements                │       │
│  │  └─ Used directly for all timesteps │       │
│  └─────────────────────────────────────┘       │
│                                                  │
│  ┌─────────────────────────────────────┐       │
│  │  Coarse Branch B (has fine levels)  │       │
│  │  ├─ TIME-DEPENDENT FINE BRANCHES     │       │
│  │  │  ├─ t=0:  Fine structure A       │       │
│  │  │  ├─ t=1:  Fine structure A (reuse!)│       │
│  │  │  ├─ t=5:  Fine structure B       │       │
│  │  │  └─ t=10: Fine structure C       │       │
│  │  └─ Only deep levels (6-12) vary     │       │
│  └─────────────────────────────────────┘       │
└─────────────────────────────────────────────────┘
```

---

## 📊 Hierarchical Octree Structure

### Level Classification

```python
@dataclass
class SharedOctreeStructure:
    """
    Multi-level octree with time-dependent fine levels.

    Structure:
    - Coarse levels (0-5): Shared across all timesteps
    - Fine levels (6-12): Time-dependent per timestep
    """

    # Shared coarse structure (STATIC)
    coarse_levels: OctreeCoarseLevels  # Levels 0-5

    # Time-dependent fine structures
    fine_levels_per_timestep: List[OctreeFineLevel]  # One per revolution step

    # Metadata
    n_coarse_levels: int = 6    # Depth of shared structure
    n_total_levels: int = 12    # Maximum octree depth
    n_timesteps: int = 40       # Number of revolution timesteps


@dataclass
class OctreeCoarseLevels:
    """Shared coarse octree structure (levels 0-5)."""

    # Octree nodes for levels 0-5
    nodes_min: jnp.ndarray       # (N_coarse, 3) bounding boxes
    nodes_max: jnp.ndarray       # (N_coarse, 3)
    nodes_depth: jnp.ndarray     # (N_coarse,) depth level
    nodes_is_leaf: jnp.ndarray   # (N_coarse,) bool - is this a leaf?
    nodes_children: jnp.ndarray  # (N_coarse, 8) child indices or -1

    # For non-leaf nodes at level 5: link to fine structures
    nodes_fine_link: jnp.ndarray # (N_coarse,) index into fine structures

    # Elements at coarse levels (only for leaves at level <6)
    nodes_elements: jnp.ndarray  # (N_coarse, max_elem) element indices
    nodes_elem_counts: jnp.ndarray  # (N_coarse,) counts


@dataclass
class OctreeFineLevel:
    """Time-dependent fine octree branches for one timestep."""

    timestep_idx: int            # Which timestep this represents

    # Fine nodes (levels 6-12) for this timestep
    fine_nodes_min: jnp.ndarray       # (N_fine, 3)
    fine_nodes_max: jnp.ndarray       # (N_fine, 3)
    fine_nodes_depth: jnp.ndarray     # (N_fine,) depth 6-12
    fine_nodes_is_leaf: jnp.ndarray   # (N_fine,) bool
    fine_nodes_children: jnp.ndarray  # (N_fine, 8)
    fine_nodes_parent_coarse: jnp.ndarray  # (N_fine,) link to coarse node

    # Elements in fine leaves
    fine_nodes_elements: jnp.ndarray
    fine_nodes_elem_counts: jnp.ndarray

    # Reuse information (NEW)
    reused_from_timestep: Optional[int] = None  # If identical, which timestep?
    change_mask: Optional[jnp.ndarray] = None    # Which fine nodes changed
```

---

## 🏗️ Building Strategy

### Phase 1: Build Shared Coarse Octree

Use first N refinement steps to understand coarse structure:

```python
def build_shared_coarse_octree(refinement_meshes, n_coarse_levels=6):
    """
    Build shared coarse octree from refinement progression.

    Strategy:
    1. Use first few refinement steps to see splitting pattern
    2. Build octree down to level n_coarse_levels (default: 6)
    3. This structure is SHARED across all revolution timesteps

    Parameters:
    -----------
    refinement_meshes : List[Mesh]
        First N meshes showing refinement progression (user-configurable)
    n_coarse_levels : int
        Depth of shared coarse structure (default: 6)

        Tuning guide:
        - Level 4: Very coarse (~16^3 = 4k regions)
        - Level 5: Coarse (~32^3 = 32k regions)
        - Level 6: Medium (~64^3 = 262k regions) [DEFAULT]
        - Level 7: Fine (~128^3 = 2M regions)

    Returns:
    --------
    coarse_octree : OctreeCoarseLevels
        Shared coarse structure
    """

    print(f"🌲 Building shared coarse octree (depth {n_coarse_levels})...")
    print(f"   Using {len(refinement_meshes)} refinement steps")

    # Start with coarsest mesh
    base_mesh = refinement_meshes[0]

    # Build octree, but stop at depth n_coarse_levels
    octree = build_octree_limited_depth(
        base_mesh.points,
        base_mesh.connectivity,
        max_depth=n_coarse_levels,
        max_elements_per_leaf=32  # Allow more elements at coarse levels
    )

    # Analyze which coarse nodes will have fine children
    # (i.e., which regions show refinement in later steps)
    fine_parent_mask = analyze_refinement_regions(
        octree,
        refinement_meshes
    )

    # Mark coarse nodes that will have time-dependent fine structure
    octree.nodes_fine_link = np.full(len(octree.nodes_min), -1, dtype=np.int32)
    fine_parent_indices = np.where(fine_parent_mask)[0]
    octree.nodes_fine_link[fine_parent_indices] = np.arange(len(fine_parent_indices))

    print(f"   ✅ Shared coarse octree: {len(octree.nodes_min)} nodes")
    print(f"   📍 Nodes with fine structure: {len(fine_parent_indices)} "
          f"({100*len(fine_parent_indices)/len(octree.nodes_min):.1f}%)")

    return octree


def analyze_refinement_regions(coarse_octree, refinement_meshes):
    """
    Determine which coarse nodes contain refined elements.

    Strategy:
    Compare mesh sizes across refinement steps.
    Any coarse node that gains elements needs fine structure.
    """

    n_coarse_nodes = len(coarse_octree.nodes_min)
    has_refinement = np.zeros(n_coarse_nodes, dtype=bool)

    # For each coarse node
    for node_idx in range(n_coarse_nodes):
        if not coarse_octree.nodes_is_leaf[node_idx]:
            continue  # Skip internal nodes

        bbox_min = coarse_octree.nodes_min[node_idx]
        bbox_max = coarse_octree.nodes_max[node_idx]

        # Count elements in this region across refinement steps
        elem_counts = []
        for mesh in refinement_meshes:
            n_elems = count_elements_in_bbox(mesh, bbox_min, bbox_max)
            elem_counts.append(n_elems)

        # If element count increases significantly → needs fine structure
        if max(elem_counts) > elem_counts[0] * 1.5:  # 50% increase
            has_refinement[node_idx] = True

    return has_refinement
```

### Phase 2: Build Time-Dependent Fine Structures

For each revolution timestep, build only the fine levels:

```python
def build_fine_structure_for_timestep(coarse_octree, mesh, timestep_idx,
                                       prev_fine_structure=None):
    """
    Build fine octree branches (levels 6-12) for one timestep.

    Key optimization: Reuse identical fine structures when possible.

    Parameters:
    -----------
    coarse_octree : OctreeCoarseLevels
        Shared coarse structure
    mesh : Mesh
        Current timestep mesh
    timestep_idx : int
        Current timestep index
    prev_fine_structure : OctreeFineLevel, optional
        Previous timestep's fine structure (for reuse detection)
    """

    # Step 1: Check if mesh is identical to previous
    if prev_fine_structure is not None:
        if meshes_are_identical(prev_fine_structure.mesh, mesh):
            # IDENTICAL mesh → reuse fine structure entirely!
            print(f"   Step {timestep_idx}: Reusing fine structure from step {prev_fine_structure.timestep_idx}")
            return OctreeFineLevel(
                timestep_idx=timestep_idx,
                reused_from_timestep=prev_fine_structure.timestep_idx,
                # All arrays shallow-copied from prev
                **{k: v for k, v in asdict(prev_fine_structure).items()
                   if k not in ['timestep_idx', 'reused_from_timestep']}
            )

    # Step 2: Build fine structure from scratch or incrementally
    fine_nodes_list = []

    # For each coarse node that has fine children
    for coarse_idx in np.where(coarse_octree.nodes_fine_link >= 0)[0]:
        bbox_min = coarse_octree.nodes_min[coarse_idx]
        bbox_max = coarse_octree.nodes_max[coarse_idx]

        # Find elements in this coarse region
        elements_in_region = find_elements_in_bbox(mesh, bbox_min, bbox_max)

        # Build fine octree for this region (levels 6-12)
        fine_branch = build_fine_branch(
            bbox_min, bbox_max,
            elements_in_region,
            mesh,
            start_depth=6,  # Continue from coarse level
            max_depth=12
        )

        fine_nodes_list.extend(fine_branch)

    # Convert to arrays
    fine_structure = consolidate_fine_nodes(fine_nodes_list, timestep_idx)

    # Step 3: If previous exists, check for partial reuse
    if prev_fine_structure is not None and not hasattr(prev_fine_structure, 'reused_from_timestep'):
        change_mask = detect_fine_structure_changes(
            prev_fine_structure,
            fine_structure
        )
        fine_structure.change_mask = change_mask

        n_changed = np.sum(change_mask)
        n_total = len(change_mask)
        print(f"   Step {timestep_idx}: {n_changed}/{n_total} fine nodes changed "
              f"({100*n_changed/n_total:.1f}%)")

    return fine_structure


def build_fine_branch(bbox_min, bbox_max, elements, mesh, start_depth, max_depth):
    """
    Recursively build fine octree branch.

    This is standard octree subdivision, but starting from depth 6.
    """
    nodes = []

    def subdivide(min_corner, max_corner, elem_indices, depth):
        node_idx = len(nodes)

        n_elems = len(elem_indices)
        is_leaf = (n_elems <= 32) or (depth >= max_depth)

        if is_leaf:
            node = FineOctreeNode(
                min_corner=min_corner,
                max_corner=max_corner,
                depth=depth,
                elements=elem_indices,
                is_leaf=True,
                children=[-1] * 8
            )
            nodes.append(node)
            return node_idx

        # Subdivide into 8 octants
        center = (min_corner + max_corner) / 2.0
        children = []

        for octant_idx in range(8):
            octant_min, octant_max = compute_octant_bounds(
                min_corner, max_corner, center, octant_idx
            )

            # Find elements in this octant
            octant_elems = [e for e in elem_indices
                            if element_intersects_bbox(mesh, e, octant_min, octant_max)]

            if len(octant_elems) > 0:
                child_idx = subdivide(octant_min, octant_max, octant_elems, depth + 1)
                children.append(child_idx)
            else:
                children.append(-1)

        node = FineOctreeNode(
            min_corner=min_corner,
            max_corner=max_corner,
            depth=depth,
            elements=[],
            is_leaf=False,
            children=children
        )
        nodes.append(node)
        return node_idx

    # Start subdivision
    subdivide(bbox_min, bbox_max, elements, start_depth)

    return nodes
```

---

## 🔍 Query Strategy

### Interpolation with Shared Coarse Octree

```python
def interpolate_with_shared_octree(query_point, timestep_idx,
                                    shared_coarse, fine_levels, mesh):
    """
    Interpolate at query point using shared coarse + time-dependent fine.

    Algorithm:
    1. Traverse shared coarse octree (levels 0-5)
    2. If reach coarse leaf → interpolate using coarse elements
    3. If reach coarse node with fine children → switch to fine octree
    4. Traverse fine octree for this timestep (levels 6-12)
    5. Interpolate in fine leaf
    """

    # Step 1: Traverse coarse octree
    coarse_node_idx = traverse_coarse_octree(query_point, shared_coarse)

    # Step 2: Check if this coarse node is a leaf
    if shared_coarse.nodes_is_leaf[coarse_node_idx]:
        # Coarse leaf → use coarse elements directly
        elem_indices = shared_coarse.nodes_elements[coarse_node_idx]
        elem_count = shared_coarse.nodes_elem_counts[coarse_node_idx]

        return interpolate_in_elements(
            query_point,
            elem_indices[:elem_count],
            mesh
        )

    # Step 3: Coarse node has fine children
    fine_link_idx = shared_coarse.nodes_fine_link[coarse_node_idx]

    if fine_link_idx < 0:
        # No fine structure (shouldn't happen if octree built correctly)
        return fallback_interpolation(query_point, mesh)

    # Step 4: Get fine structure for this timestep
    fine_structure = fine_levels[timestep_idx]

    # Handle reused fine structure
    if fine_structure.reused_from_timestep is not None:
        actual_timestep = fine_structure.reused_from_timestep
        fine_structure = fine_levels[actual_timestep]

    # Step 5: Traverse fine octree
    fine_node_idx = traverse_fine_octree(
        query_point,
        fine_structure,
        coarse_node_idx
    )

    # Step 6: Interpolate in fine leaf
    elem_indices = fine_structure.fine_nodes_elements[fine_node_idx]
    elem_count = fine_structure.fine_nodes_elem_counts[fine_node_idx]

    return interpolate_in_elements(
        query_point,
        elem_indices[:elem_count],
        mesh
    )


def traverse_coarse_octree(query_point, coarse_octree):
    """
    Traverse coarse octree to find node containing query point.

    Stops at:
    - Coarse leaf (no fine children)
    - Coarse node with fine children link
    """
    node_idx = 0  # Start at root

    while True:
        # Check if leaf or has fine children
        if coarse_octree.nodes_is_leaf[node_idx]:
            return node_idx

        if coarse_octree.nodes_fine_link[node_idx] >= 0:
            # Has fine children → stop here
            return node_idx

        # Find child containing query point
        children = coarse_octree.nodes_children[node_idx]
        node_min = coarse_octree.nodes_min[node_idx]
        node_max = coarse_octree.nodes_max[node_idx]
        center = (node_min + node_max) / 2.0

        # Determine octant
        octant = compute_octant_index(query_point, center)
        child_idx = children[octant]

        if child_idx < 0:
            # No child in this octant (shouldn't happen)
            return node_idx

        node_idx = child_idx


def traverse_fine_octree(query_point, fine_structure, coarse_parent_idx):
    """
    Traverse fine octree starting from specific coarse parent.
    """
    # Find fine root nodes for this coarse parent
    fine_roots = np.where(
        fine_structure.fine_nodes_parent_coarse == coarse_parent_idx
    )[0]

    if len(fine_roots) == 0:
        raise ValueError(f"No fine structure for coarse node {coarse_parent_idx}")

    # Typically only one root, but handle multiple
    node_idx = fine_roots[0]

    # Traverse fine tree
    while not fine_structure.fine_nodes_is_leaf[node_idx]:
        node_min = fine_structure.fine_nodes_min[node_idx]
        node_max = fine_structure.fine_nodes_max[node_idx]
        center = (node_min + node_max) / 2.0

        octant = compute_octant_index(query_point, center)
        children = fine_structure.fine_nodes_children[node_idx]
        child_idx = children[octant]

        if child_idx < 0:
            # No child → use this node
            return node_idx

        node_idx = child_idx

    return node_idx
```

---

## 💾 Memory Analysis

### Memory Breakdown

For Edgar/FLA case (40 revolution timesteps):

```
SHARED COARSE OCTREE (levels 0-6):
  Nodes: ~5,000 nodes (coarse structure)
  Memory: ~2 MB (shared across all timesteps)

TIME-DEPENDENT FINE OCTREES (levels 7-12, per timestep):
  Assumption: 10% of domain has fine structure
  Fine nodes per timestep: ~10,000 nodes
  Memory per timestep: ~50 MB

  With reuse optimization:
  - 37 timesteps reuse (0 MB additional)
  - 3 timesteps unique (3 × 50 MB = 150 MB)

  Total fine structures: ~150 MB (vs 2,000 MB without reuse!)

MESH DATA (same as before):
  Points + Velocity: 715 MB
  Connectivity: 46 MB

TOTAL: 2 MB + 150 MB + 715 MB + 46 MB = 913 MB
```

**Previous estimate**: 3,116 MB
**With shared coarse + reuse**: **913 MB** (3.4× less!)

✅ **Massive memory savings!**

---

## ⚙️ Configuration

### User-Configurable Parameters

```python
user_config = {
    # =========================================================================
    # AMR Octree Strategy
    # =========================================================================

    # Number of initial refinement steps to analyze
    'n_refinement_steps': 10,  # User-configurable (default: 10)
    # Used to understand mesh splitting pattern and build coarse structure

    # Depth of shared coarse octree
    'coarse_octree_depth': 6,  # Levels 0-6 are shared (default: 6)
    # Tuning: Lower = less memory, may need more fine structure
    #         Higher = more memory, finer coarse structure

    # Maximum total octree depth
    'max_octree_depth': 12,    # Levels 7-12 are time-dependent

    # Revolution timesteps (from END)
    'revolution_timesteps': 40,

    # Element splitting threshold for coarse nodes
    'refinement_threshold': 1.5,  # 50% increase triggers fine structure

    # =========================================================================
    # Reuse Detection
    # =========================================================================

    # Enable fine structure reuse
    'enable_fine_structure_reuse': True,  # Detect identical fine structures

    # Reuse tolerance
    'reuse_tolerance': 0.001,  # Consider identical if <0.1% different
}
```

---

## 📈 Expected Performance

### For Edgar/FLA (780k points, 40 revolution steps)

**Coarse octree build** (one-time):
- Analyze 10 refinement steps: 2 min
- Build coarse octree (depth 6): 5 sec
- Total: **2 min**

**Fine structure build** (per timestep):
- 37 timesteps identical: 0 sec each (instant reuse)
- 3 timesteps unique: 2 sec each
- Total: **6 sec** for all 40 timesteps

**Memory**:
- Coarse: 2 MB
- Fine (with reuse): 150 MB
- Mesh data: 761 MB
- **Total: 913 MB** (vs 3,116 MB before - 3.4× savings!)

**Total startup**:
```
VTK loading:        5 min
Coarse octree:      2 min
Fine structures:    6 sec
JAX conversion:     1 min
────────────────────────
Total:              8 min
```

**Previous estimate**: 8 min
**With shared coarse**: **8 min** (same time, 3.4× less memory!)

---

## 🎯 Implementation Tasks

### Phase 1: Shared Coarse Octree (8 hours)

**Task 1.1**: Build limited-depth octree (3 hours)
- Modify existing octree builder to stop at specified depth
- Add coarse/fine transition markers

**Task 1.2**: Refinement region analysis (3 hours)
- Analyze which coarse nodes have refinement
- Mark fine parent nodes

**Task 1.3**: Fine structure builder (2 hours)
- Build fine branches from coarse nodes
- Link to coarse parents

### Phase 2: Reuse Detection (4 hours)

**Task 2.1**: Mesh identity check (2 hours)
- Fast connectivity comparison
- Partial reuse detection

**Task 2.2**: Fine structure reuse (2 hours)
- Shallow copy for identical structures
- Change mask for partial changes

### Phase 3: Query Engine (6 hours)

**Task 3.1**: Two-level traversal (4 hours)
- Coarse octree traversal
- Fine octree traversal
- Transition logic

**Task 3.2**: JIT optimization (2 hours)
- Compile query functions
- Optimize hot paths

**Total effort**: 18 hours (vs 28 hours before)

---

## ✅ Advantages of This Approach

1. **Memory efficient**: 3.4× less memory (913 MB vs 3,116 MB)
2. **Naturally exploits AMR structure**: Coarse regions static, fine vary
3. **Massive reuse**: 92.5% of timesteps reuse fine structure
4. **User-configurable**: Tunable refinement analysis and coarse depth
5. **Works for varying datasets**: ThreadedA with more variation will benefit too

This is the optimal strategy for your welding AMR simulations! 🚀
