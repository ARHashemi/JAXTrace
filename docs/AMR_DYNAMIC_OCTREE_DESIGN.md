# Dynamic Octree for AMR Welding Simulations
## Redesigned Strategy for Revolution Cycles

**Date**: 2025-10-09
**Context**: Welding tool rotation with AMR mesh evolution
**Goal**: Efficient octree management for 1-2 revolution cycles (~40 timesteps)

---

## Table of Contents
1. [Problem Reformulation](#problem)
2. [Use Case Analysis](#use-case)
3. [Hierarchical Mesh Strategy](#hierarchical)
4. [Incremental Octree Updates](#incremental)
5. [Implementation Design](#implementation)
6. [Performance Expectations](#performance)

---

## 1. Problem Reformulation {#problem}

### 1.1 Your Actual Use Case

**Welding simulation characteristics**:
- **Tool rotation**: 1-2 complete revolutions
- **Velocity timesteps needed**: ~40 (covers revolution cycles)
- **Tracking timesteps**: 1000+ (fine temporal resolution for particles)
- **Mesh evolution**:
  - First 10 steps: Refinement progression (coarse → fine)
  - Remaining steps: Localized refinement near weld pool
  - Revolution pattern: Mesh refinement follows tool position

**Key insight**: Load all 40 velocity timesteps to GPU, wrap for tracking!

### 1.2 AMR Evolution Pattern

```
Timesteps 0-9: Progressive refinement (coarsest → finest)
├─ Step 0:  500k points (coarse)
├─ Step 1:  550k points
├─ Step 2:  600k points
├─ ...
└─ Step 9:  780k points (fully refined base mesh)

Timesteps 10-49: Revolution cycles (localized refinement)
├─ Step 10: 780k points (tool at 0°)
├─ Step 15: 790k points (refinement at 90°)
├─ Step 20: 785k points (tool at 180°)
├─ Step 25: 795k points (refinement at 270°)
└─ Step 40: 780k points (tool back at 0°)
```

**Refinement characteristics**:
- **Spatial**: Follows tool position (10-20% of domain)
- **Temporal**: Most mesh nodes stable across revolution
- **Hierarchical**: Coarse levels (0-9) build finer levels (10-49)

### 1.3 Rejected Strategies (and why)

❌ **Lazy loading**: Too slow for tracking (need fast velocity lookup)
❌ **Disk caching**: Not useful (don't repeat same tracking run)
❌ **Parallel VTK loading**: Unknown benefit on your PC
❌ **Skip all mesh detection**: Still need it as optional feature

### 1.4 Accepted Strategies (your insights)

✅ **Skip mesh detection by default** (make it optional)
✅ **Load all 40 timesteps to GPU** (fits in memory, fast tracking)
✅ **Hierarchical mesh from steps 0-9** (coarse → fine progression)
✅ **Incremental octree for steps 10-49** (localized changes)
✅ **Disk caching as option** (user choice, not default)

---

## 2. Use Case Analysis {#use-case}

### 2.1 Memory Budget

**Your GPU**: Likely 4-24 GB (T1000 or better)
**Required for 40 timesteps**:

| Component | Size per step | Total (40 steps) |
|-----------|---------------|------------------|
| Points | 780k × 3 × 4B = 9.4 MB | 376 MB |
| Velocity | 780k × 3 × 4B = 9.4 MB | 376 MB |
| Connectivity | 3.5M × 4 × 4B = 56 MB | 56 MB (shared) |
| Element bounds | 3.5M × 6 × 4B = 84 MB | 84 MB (shared) |
| Octrees (incremental) | ~50 MB average | 2 GB |
| **Total** | - | **~3 GB** |

**Conclusion**: ✅ Fits comfortably in 4+ GB GPU

### 2.2 Performance Requirements

**Target performance** (for 1000 tracking steps, 18k particles):
- Mesh loading: < 5 minutes (40 files × 7 sec = 4.7 min)
- Octree building: < 3 minutes (hierarchical + incremental)
- Tracking: ~10-30 minutes (depends on complexity)
- **Total**: < 40 minutes end-to-end

**Bottleneck priority**:
1. **Octree building** (biggest gain from hierarchical + incremental)
2. Mesh loading (acceptable at ~5 min for 40 files)
3. Tracking (already fast with spatial batching)

### 2.3 Workflow Pattern

```
Typical user workflow:
1. Set up simulation parameters
2. Run tracking (40 minutes)
3. Visualize results
4. Adjust parameters
5. Re-run with different config → NEW RUN (no cache benefit)

Conclusion: Disk caching is optional, not priority
```

---

## 3. Hierarchical Mesh Strategy {#hierarchical}

### 3.1 Concept: Build Coarse-to-Fine Hierarchy

**Use first 10 timesteps as refinement levels**:

```
Level 0 (Step 0): 500k points - Coarsest mesh
Level 1 (Step 2): 550k points
Level 2 (Step 4): 600k points
Level 3 (Step 6): 700k points
Level 4 (Step 9): 780k points - Finest base mesh
```

**Key insight**: Each level is a refinement of previous level
- Level i+1 contains all nodes from Level i + new refined nodes
- Can build octree incrementally: Octree(i+1) = Refine(Octree(i))

### 3.2 Hierarchical Octree Structure

```python
@dataclass
class HierarchicalOctree:
    """
    Multi-level octree for AMR mesh evolution.

    Levels built from coarse (0) to fine (4).
    Each level reuses structure from previous level.
    """

    levels: List[OctreeLevel]  # 5 levels (steps 0, 2, 4, 6, 9)
    base_level: int = 4         # Finest base level (step 9)

@dataclass
class OctreeLevel:
    """Single octree level."""

    timestep_idx: int           # Which timestep this represents
    n_points: int               # Mesh size

    # Octree structure
    nodes_min: jnp.ndarray
    nodes_max: jnp.ndarray
    nodes_elements: jnp.ndarray
    nodes_children: jnp.ndarray
    nodes_is_leaf: jnp.ndarray

    # Incremental info
    parent_level: Optional[int] = None  # Previous level index
    new_nodes_mask: Optional[jnp.ndarray] = None  # Which octree nodes are new
```

### 3.3 Building Strategy

#### Phase 1: Build Level 0 (Coarsest)

```python
# Load step 0 (500k points, 2M elements)
mesh_0 = load_vtk_file(files[0])

# Build full octree
octree_level_0 = build_octree_mesh_optimized(
    mesh_0.points,
    mesh_0.connectivity,
    max_elements_per_leaf=32,
    max_depth=12
)

# Expected: ~15 seconds (smaller mesh)
```

#### Phase 2: Incremental Build for Levels 1-4

```python
for level in [1, 2, 3, 4]:
    timestep = [2, 4, 6, 9][level - 1]

    mesh_new = load_vtk_file(files[timestep])
    mesh_old = mesh_from_level[level - 1]

    # Detect mesh changes
    changes = detect_amr_changes(mesh_old, mesh_new)

    # Incremental octree update
    octree_new = incremental_octree_build(
        octree_old=octree_levels[level - 1],
        mesh_old=mesh_old,
        mesh_new=mesh_new,
        changes=changes
    )

    octree_levels[level] = octree_new

# Expected per level: 3-5 seconds (only rebuild changed branches)
# Total: 12-20 seconds for all 4 refinements
```

**Total hierarchical build time**: 15s + 20s = **35 seconds** (vs 4 minutes for 5 separate builds)

### 3.4 AMR Change Detection

```python
def detect_amr_changes(mesh_old, mesh_new):
    """
    Detect AMR refinement/coarsening changes.

    Returns:
    --------
    changes : AMRChanges
        - added_points: New node indices
        - removed_points: Deleted node indices
        - added_elements: New element indices
        - removed_elements: Deleted element indices
        - affected_regions: Bounding boxes of changed areas
    """

    n_old = len(mesh_old.points)
    n_new = len(mesh_new.points)

    if n_new > n_old:
        # Refinement: new points added
        # Assumption: Old points preserved, new points at end
        added_points = range(n_old, n_new)
        removed_points = []

    elif n_new < n_old:
        # Coarsening: points removed
        # Need to detect which points are missing
        added_points = []
        removed_points = detect_removed_points(mesh_old, mesh_new)

    else:
        # Same size, check if points moved
        added_points = []
        removed_points = []

    # Detect element changes
    old_elem_set = set(map(tuple, mesh_old.connectivity))
    new_elem_set = set(map(tuple, mesh_new.connectivity))

    added_elements = list(new_elem_set - old_elem_set)
    removed_elements = list(old_elem_set - new_elem_set)

    # Compute affected regions
    affected_regions = compute_affected_bboxes(
        added_points, removed_points,
        added_elements, removed_elements,
        mesh_new
    )

    return AMRChanges(
        added_points=added_points,
        removed_points=removed_points,
        added_elements=added_elements,
        removed_elements=removed_elements,
        affected_regions=affected_regions
    )
```

---

## 4. Incremental Octree Updates {#incremental}

### 4.1 Concept: Reuse Unchanged Branches

For revolution steps (10-49), mesh changes are **localized**:
- Tool position changes → refinement moves
- 80-90% of mesh unchanged
- Only rebuild octree branches in affected regions

### 4.2 Algorithm: Incremental Update

```python
def incremental_octree_build(octree_old, mesh_old, mesh_new, changes):
    """
    Incrementally update octree when mesh changes.

    Algorithm:
    1. Identify affected octree nodes (intersect changed regions)
    2. Mark branches for rebuild
    3. Copy unchanged branches from old octree
    4. Rebuild only affected branches
    5. Merge into new octree
    """

    # Step 1: Find affected octree nodes
    affected_nodes = find_affected_octree_nodes(
        octree_old,
        changes.affected_regions
    )

    # Step 2: Determine rebuild strategy
    n_affected = len(affected_nodes)
    n_total = len(octree_old.nodes_min)

    if n_affected > 0.5 * n_total:
        # Too many changes (>50%) - full rebuild faster
        print(f"   Large mesh change ({n_affected}/{n_total} nodes)")
        print(f"   Performing full octree rebuild...")
        return build_octree_mesh_optimized(
            mesh_new.points,
            mesh_new.connectivity
        )

    # Step 3: Incremental update
    print(f"   Incremental update: rebuilding {n_affected}/{n_total} nodes")

    # Copy old octree structure
    octree_new = copy_octree_structure(octree_old)

    # For each affected node, rebuild its subtree
    for node_idx in affected_nodes:
        # Get spatial region for this node
        bbox_min = octree_old.nodes_min[node_idx]
        bbox_max = octree_old.nodes_max[node_idx]

        # Find all elements in this region (from new mesh)
        elements_in_region = find_elements_in_bbox(
            mesh_new,
            bbox_min,
            bbox_max
        )

        # Rebuild this branch
        rebuild_octree_branch(
            octree_new,
            node_idx,
            bbox_min,
            bbox_max,
            elements_in_region,
            mesh_new
        )

    return octree_new
```

### 4.3 Affected Node Detection

```python
def find_affected_octree_nodes(octree, affected_regions):
    """
    Find octree nodes that intersect with affected regions.

    Parameters:
    -----------
    octree : OctreeMesh
        Existing octree structure
    affected_regions : List[Tuple[bbox_min, bbox_max]]
        Bounding boxes of changed mesh regions

    Returns:
    --------
    affected_nodes : List[int]
        Indices of octree nodes needing rebuild
    """

    affected_nodes = []

    for node_idx in range(len(octree.nodes_min)):
        node_min = octree.nodes_min[node_idx]
        node_max = octree.nodes_max[node_idx]

        # Check if this node intersects any affected region
        for region_min, region_max in affected_regions:
            if bbox_intersects(node_min, node_max, region_min, region_max):
                affected_nodes.append(node_idx)
                break

    return affected_nodes


def bbox_intersects(min1, max1, min2, max2):
    """AABB intersection test."""
    return (
        (min1[0] <= max2[0]) and (max1[0] >= min2[0]) and
        (min1[1] <= max2[1]) and (max1[1] >= min2[1]) and
        (min1[2] <= max2[2]) and (max1[2] >= min2[2])
    )
```

### 4.4 Branch Rebuilding

```python
def rebuild_octree_branch(octree, node_idx, bbox_min, bbox_max,
                          elements, mesh):
    """
    Rebuild a single octree branch with new elements.

    Strategy:
    1. Clear old children of this node
    2. Recursively subdivide with new elements
    3. Update parent's child pointer
    """

    n_elems = len(elements)
    depth = compute_node_depth(octree, node_idx)

    # Check if leaf
    is_leaf = (n_elems <= octree.max_elements_per_leaf) or \
              (depth >= octree.max_depth)

    if is_leaf:
        # Update as leaf node
        octree.nodes_elements[node_idx, :n_elems] = elements
        octree.nodes_elem_counts[node_idx] = n_elems
        octree.nodes_is_leaf[node_idx] = True
        octree.nodes_children[node_idx] = -1  # No children

    else:
        # Subdivide into 8 octants
        center = (bbox_min + bbox_max) / 2.0

        # Distribute elements to octants
        octant_elements = [[] for _ in range(8)]
        for elem_idx in elements:
            elem_bbox = compute_element_bbox(mesh, elem_idx)

            for octant_idx in range(8):
                oct_min, oct_max = get_octant_bounds(
                    bbox_min, bbox_max, center, octant_idx
                )

                if bbox_intersects(elem_bbox[0], elem_bbox[1],
                                   oct_min, oct_max):
                    octant_elements[octant_idx].append(elem_idx)

        # Create child nodes
        for octant_idx in range(8):
            if len(octant_elements[octant_idx]) > 0:
                oct_min, oct_max = get_octant_bounds(
                    bbox_min, bbox_max, center, octant_idx
                )

                # Allocate new node in octree
                child_idx = allocate_octree_node(octree)
                octree.nodes_children[node_idx, octant_idx] = child_idx

                # Recursively build child
                rebuild_octree_branch(
                    octree, child_idx,
                    oct_min, oct_max,
                    octant_elements[octant_idx],
                    mesh
                )
```

### 4.5 Expected Performance

**Incremental update for revolution steps (10-49)**:

| Mesh change | Affected nodes | Build time | Speedup |
|-------------|----------------|------------|---------|
| 5% changed | 10% nodes | 5 sec | 6× |
| 10% changed | 15% nodes | 8 sec | 4× |
| 20% changed | 25% nodes | 12 sec | 2.5× |
| >50% changed | Full rebuild | 30 sec | 1× |

**Total time for 40 revolution steps** (steps 10-49):
- Without incremental: 40 × 30 sec = **20 minutes**
- With incremental: 40 × 8 sec = **5.3 minutes**
- **Speedup: 3.8×**

---

## 5. Implementation Design {#implementation}

### 5.1 New Classes

#### HierarchicalAMROctreeField

```python
# File: jaxtrace/fields/hierarchical_amr_octree.py

class HierarchicalAMROctreeField:
    """
    Time series field with hierarchical octree for AMR data.

    Strategy:
    1. Build hierarchy from first N timesteps (coarse → fine)
    2. Use incremental updates for remaining timesteps
    3. Load all velocity data to GPU
    """

    def __init__(self,
                 file_pattern: str,
                 max_timesteps: int = 40,
                 hierarchy_steps: List[int] = [0, 2, 4, 6, 9],
                 skip_mesh_detection: bool = True,
                 enable_disk_cache: bool = False,
                 cache_dir: str = ".cache/octrees"):
        """
        Initialize hierarchical AMR octree field.

        Parameters:
        -----------
        file_pattern : str
            VTK file glob pattern
        max_timesteps : int
            Number of timesteps to load (e.g., 40 for 1-2 revolutions)
        hierarchy_steps : List[int]
            Timestep indices for building hierarchy levels
        skip_mesh_detection : bool
            If True, skip scanning all files for mesh sizes (default: True)
        enable_disk_cache : bool
            If True, cache octrees to disk (default: False)
        cache_dir : str
            Directory for octree cache files
        """

        self.file_pattern = file_pattern
        self.max_timesteps = max_timesteps
        self.hierarchy_steps = hierarchy_steps
        self.skip_mesh_detection = skip_mesh_detection
        self.enable_disk_cache = enable_disk_cache
        self.cache_dir = cache_dir

        # Load data
        self._load_mesh_data()

        # Build hierarchical octrees
        self._build_hierarchical_octrees()

        # Build octrees for revolution steps
        self._build_revolution_octrees()

    def _load_mesh_data(self):
        """Load all velocity timesteps."""
        files = sorted(glob(self.file_pattern))

        if not self.skip_mesh_detection:
            # Optional: detect stable mesh
            files = self._detect_stable_mesh(files)

        files_to_load = files[:self.max_timesteps]

        print(f"📁 Loading {len(files_to_load)} velocity timesteps...")

        # Load all meshes and velocities
        self.meshes = []
        self.velocities = []
        self.times = []

        for idx, filepath in enumerate(files_to_load):
            mesh_data = self._load_vtk_with_connectivity(filepath)
            self.meshes.append(mesh_data)
            self.velocities.append(mesh_data.velocity)
            self.times.append(mesh_data.time)

            if (idx + 1) % 10 == 0:
                print(f"   Loaded {idx + 1}/{len(files_to_load)} timesteps")

        print(f"✅ Loaded {len(self.meshes)} timesteps")

    def _build_hierarchical_octrees(self):
        """Build hierarchy from coarse to fine."""
        print(f"🌲 Building hierarchical octree...")
        print(f"   Hierarchy levels: {self.hierarchy_steps}")

        self.hierarchy_octrees = []

        for level_idx, timestep_idx in enumerate(self.hierarchy_steps):
            mesh = self.meshes[timestep_idx]

            if level_idx == 0:
                # Base level: full build
                print(f"   Level 0: Building base octree (step {timestep_idx}, {mesh.n_points} points)...")
                octree = build_octree_mesh_optimized(
                    mesh.points,
                    mesh.connectivity
                )

            else:
                # Incremental from previous level
                prev_mesh = self.meshes[self.hierarchy_steps[level_idx - 1]]
                prev_octree = self.hierarchy_octrees[level_idx - 1]

                print(f"   Level {level_idx}: Incremental build (step {timestep_idx}, {mesh.n_points} points)...")

                changes = detect_amr_changes(prev_mesh, mesh)
                octree = incremental_octree_build(
                    prev_octree, prev_mesh, mesh, changes
                )

            self.hierarchy_octrees.append(octree)

        print(f"✅ Hierarchical octree built: {len(self.hierarchy_octrees)} levels")

    def _build_revolution_octrees(self):
        """Build octrees for revolution steps using incremental updates."""
        n_hierarchy = len(self.hierarchy_steps)
        n_total = len(self.meshes)

        if n_total <= n_hierarchy:
            # No revolution steps
            self.revolution_octrees = []
            return

        print(f"🔄 Building octrees for revolution steps...")
        print(f"   Steps {n_hierarchy} to {n_total - 1} ({n_total - n_hierarchy} steps)")

        # Start from finest hierarchy level
        base_octree = self.hierarchy_octrees[-1]
        base_mesh = self.meshes[self.hierarchy_steps[-1]]

        self.revolution_octrees = []

        for timestep_idx in range(n_hierarchy, n_total):
            mesh = self.meshes[timestep_idx]

            # Incremental update from previous
            if timestep_idx == n_hierarchy:
                prev_octree = base_octree
                prev_mesh = base_mesh
            else:
                prev_octree = self.revolution_octrees[-1]
                prev_mesh = self.meshes[timestep_idx - 1]

            changes = detect_amr_changes(prev_mesh, mesh)

            octree = incremental_octree_build(
                prev_octree, prev_mesh, mesh, changes
            )

            self.revolution_octrees.append(octree)

            if (timestep_idx + 1) % 10 == 0:
                print(f"   Built {timestep_idx + 1 - n_hierarchy}/{n_total - n_hierarchy} revolution octrees")

        print(f"✅ Revolution octrees built: {len(self.revolution_octrees)} steps")

    def get_octree(self, timestep_idx):
        """Get octree for specific timestep."""
        n_hierarchy = len(self.hierarchy_steps)

        if timestep_idx in self.hierarchy_steps:
            # From hierarchy
            level_idx = self.hierarchy_steps.index(timestep_idx)
            return self.hierarchy_octrees[level_idx]

        elif timestep_idx >= n_hierarchy:
            # From revolution steps
            rev_idx = timestep_idx - n_hierarchy
            return self.revolution_octrees[rev_idx]

        else:
            # Between hierarchy steps - use previous hierarchy level
            for i in range(len(self.hierarchy_steps) - 1, -1, -1):
                if timestep_idx > self.hierarchy_steps[i]:
                    return self.hierarchy_octrees[i]

            return self.hierarchy_octrees[0]

    def sample_at_positions(self, query_positions, t):
        """Sample velocity at positions and time."""
        # Find bracketing timesteps
        t_idx_left, t_idx_right, alpha = self._find_bracketing_timesteps(t)

        # Get velocities from both timesteps
        octree_left = self.get_octree(t_idx_left)
        octree_right = self.get_octree(t_idx_right)

        v_left = self._interpolate_spatial(query_positions, t_idx_left, octree_left)
        v_right = self._interpolate_spatial(query_positions, t_idx_right, octree_right)

        # Temporal interpolation
        v = v_left + alpha * (v_right - v_left)

        return v
```

### 5.2 Configuration Options

```python
# File: example_workflow.py

user_config = {
    # -------------------------------------------------------------------------
    # AMR Octree Configuration
    # -------------------------------------------------------------------------

    # Mesh detection (optional, off by default)
    'detect_stable_mesh': False,  # Set True to scan all files and filter

    # Hierarchical octree
    'use_hierarchical_octree': True,  # Enable hierarchical AMR strategy
    'hierarchy_timesteps': [0, 2, 4, 6, 9],  # Steps for building hierarchy

    # Revolution cycles
    'max_timesteps_to_load': 40,  # Velocity timesteps (1-2 revolutions)

    # Incremental updates
    'use_incremental_octree': True,  # Enable incremental updates
    'incremental_rebuild_threshold': 0.5,  # Rebuild fully if >50% changed

    # Disk caching (optional)
    'enable_octree_cache': False,  # Set True to cache octrees to disk
    'octree_cache_dir': '.cache/octrees',

    # Octree parameters
    'max_elements_per_leaf': 32,
    'max_octree_depth': 12,
    'use_advanced_element_search': True,
}
```

### 5.3 File Structure

```
jaxtrace/
├── fields/
│   ├── hierarchical_amr_octree.py          # NEW: Main hierarchical field
│   ├── incremental_octree_builder.py       # NEW: Incremental update logic
│   ├── amr_mesh_utils.py                   # NEW: AMR change detection
│   ├── octree_fem_interpolator_optimized.py  # EXISTING: Base octree
│   └── octree_fem_time_series_optimized.py   # EXISTING: Time series base
│
├── io/
│   └── vtk_loader.py                       # EXISTING: VTK reading
│
└── utils/
    └── octree_cache.py                     # NEW: Optional disk caching
```

---

## 6. Performance Expectations {#performance}

### 6.1 Time Breakdown (40 timesteps, 780k points, 3.5M elements)

#### Current Implementation (No Optimization)
```
Mesh detection:        10 min (scan 160 files - OPTIONAL NOW)
VTK loading:           5 min  (40 × 7 sec)
Octree building:       20 min (40 × 30 sec, full rebuild each)
JAX conversion:        2 min
──────────────────────────────
Total:                 37 min (with mesh detection)
Total:                 27 min (skip mesh detection)
```

#### With Hierarchical + Incremental (Optimized)
```
Mesh detection:        0 min  (SKIPPED by default)
VTK loading:           5 min  (40 × 7 sec, unavoidable)
Hierarchical octree:   0.5 min (5 levels × 6 sec average)
Revolution octrees:    5 min  (35 × 8 sec incremental)
JAX conversion:        2 min
──────────────────────────────
Total:                 12.5 min
```

**Speedup**: 27 min → 12.5 min = **2.2× faster** (skipping mesh detection)

### 6.2 Component-wise Performance

| Component | Current | Optimized | Speedup |
|-----------|---------|-----------|---------|
| **Mesh detection** | 10 min | 0 min (skipped) | ∞× |
| **VTK loading** | 5 min | 5 min (same) | 1× |
| **Octree: First 10 steps** | 5 min | 30 sec (hierarchical) | 10× |
| **Octree: Remaining 30 steps** | 15 min | 4 min (incremental) | 3.8× |
| **JAX conversion** | 2 min | 2 min (same) | 1× |
| **TOTAL** | 37 min | 11.5 min | **3.2×** |
| **TOTAL (no detection)** | 27 min | 11.5 min | **2.3×** |

### 6.3 Memory Usage

**All 40 timesteps loaded to GPU**:
- Meshes: 376 MB (points)
- Velocities: 376 MB
- Connectivity: 56 MB (shared)
- Octrees: ~2 GB (40 octrees, incremental reduces size)
- **Total: ~3 GB**

✅ Fits in 4GB GPU (T1000)
✅ Comfortable in 8GB+ GPU

### 6.4 Scaling with Parameters

**Effect of max_timesteps_to_load**:

| Timesteps | Octree build time | Total time | Memory |
|-----------|-------------------|------------|--------|
| 20 (1 rev) | 3 min | 7.5 min | 1.5 GB |
| 40 (2 rev) | 5 min | 11.5 min | 3 GB |
| 80 (4 rev) | 10 min | 19 min | 6 GB |

**Effect of mesh size**:

| Mesh size | Octree build time | Memory |
|-----------|-------------------|--------|
| 500k points | 2.5 min | 2 GB |
| 780k points | 5 min | 3 GB |
| 1M points | 8 min | 4 GB |

---

## 7. Implementation Roadmap {#roadmap}

### Phase 1: Core Infrastructure (Week 1, ~16 hours)

#### Task 1.1: AMR Change Detection (4 hours)
**File**: `jaxtrace/fields/amr_mesh_utils.py`

```python
def detect_amr_changes(mesh_old, mesh_new):
    """Detect added/removed points and elements."""
    # Implementation as shown above
    pass

def compute_affected_bboxes(changes, mesh):
    """Compute bounding boxes of changed regions."""
    pass
```

**Test**: Create unit tests with synthetic AMR data

#### Task 1.2: Incremental Octree Builder (8 hours)
**File**: `jaxtrace/fields/incremental_octree_builder.py`

```python
def incremental_octree_build(octree_old, mesh_old, mesh_new, changes):
    """Main incremental update function."""
    pass

def find_affected_octree_nodes(octree, affected_regions):
    """Find nodes intersecting changed regions."""
    pass

def rebuild_octree_branch(octree, node_idx, elements, mesh):
    """Rebuild single branch."""
    pass
```

**Test**: Test with progressive refinement (500k → 780k points)

#### Task 1.3: Hierarchical Octree Field (4 hours)
**File**: `jaxtrace/fields/hierarchical_amr_octree.py`

```python
class HierarchicalAMROctreeField:
    """Main field class with hierarchical + incremental."""
    pass
```

**Test**: Load 10 timesteps, build hierarchy, verify speedup

### Phase 2: Integration (Week 2, ~12 hours)

#### Task 2.1: Update example_workflow.py (3 hours)
- Add configuration options
- Integrate HierarchicalAMROctreeField
- Make mesh detection optional

#### Task 2.2: Optional Disk Caching (4 hours)
**File**: `jaxtrace/utils/octree_cache.py`

```python
def save_octree_to_cache(octree, filepath):
    """Save octree to .npz file."""
    pass

def load_octree_from_cache(filepath):
    """Load octree from cache."""
    pass

def check_cache_validity(cache_file, mesh_file):
    """Check if cache is newer than mesh."""
    pass
```

#### Task 2.3: Testing and Validation (5 hours)
- Test with your Edgar/FLA data
- Verify performance improvements
- Check memory usage
- Validate tracking results

### Phase 3: Optimization and Polish (Week 3, optional)

#### Task 3.1: Performance Profiling
- Identify remaining bottlenecks
- Optimize critical paths

#### Task 3.2: Documentation
- Usage examples
- Configuration guide
- Performance tuning tips

#### Task 3.3: Error Handling
- Graceful degradation for large changes
- Better progress reporting
- Memory overflow handling

---

## 8. Usage Example {#usage}

### 8.1 Basic Usage

```python
from jaxtrace import configure
from jaxtrace.fields.hierarchical_amr_octree import HierarchicalAMROctreeField
from jaxtrace.tracking import SpatialBatchingTracker

# Configure
configure(device='gpu', memory_limit_gb=4.0)

# Create hierarchical AMR field
field = HierarchicalAMROctreeField(
    file_pattern="/path/to/data/welding_*.pvtu",
    max_timesteps=40,  # Load 40 timesteps (1-2 revolutions)
    hierarchy_steps=[0, 2, 4, 6, 9],  # Build hierarchy from these
    skip_mesh_detection=True,  # Don't scan all files
    enable_disk_cache=False  # Don't cache (new runs each time)
)

# Track particles
tracker = SpatialBatchingTracker(
    field=field,
    dt=0.0025,
    integrator='rk4'
)

# Generate particles
initial_positions = generate_uniform_particles(
    field.domain_bounds,
    resolution=(30, 40, 15)
)

# Run tracking
trajectories = tracker.track(
    initial_positions,
    n_steps=1000,
    boundary_conditions='reflective'
)

# Save results
trajectories.save('welding_particles.h5')
```

### 8.2 With Mesh Detection (Optional)

```python
# When you want to auto-filter mesh sizes
field = HierarchicalAMROctreeField(
    file_pattern="/path/to/data/welding_*.pvtu",
    max_timesteps=40,
    skip_mesh_detection=False,  # ENABLE scanning
    detect_most_common=True,  # Filter to most common size
    hierarchy_steps=[0, 2, 4, 6, 9],
)

# Will print:
# "📁 Scanning 160 files for mesh sizes..."
# "   Found 3 mesh sizes: 780k (120 files), 790k (25 files), 800k (15 files)"
# "   Using most common: 780k points (120 files)"
# "   Filtered out 40 files with different mesh sizes"
```

### 8.3 With Disk Caching (Optional)

```python
# When you'll run same dataset multiple times
field = HierarchicalAMROctreeField(
    file_pattern="/path/to/data/welding_*.pvtu",
    max_timesteps=40,
    enable_disk_cache=True,  # ENABLE caching
    cache_dir=".cache/octrees",
    hierarchy_steps=[0, 2, 4, 6, 9],
)

# First run: Builds octrees, saves to cache (~12 min)
# Second run: Loads from cache (~2 min)
```

---

## 9. Summary and Next Steps {#summary}

### 9.1 Strategy Summary

✅ **Skip mesh detection by default** (make optional)
✅ **Load all 40 timesteps to GPU** (fits in memory, enables fast tracking)
✅ **Hierarchical octree from steps 0-9** (10× speedup for base mesh)
✅ **Incremental updates for steps 10-49** (3.8× speedup for revolution)
✅ **Optional disk caching** (user choice, not default)

### 9.2 Expected Benefits

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Startup time** | 27 min | 11.5 min | **2.3× faster** |
| **Octree build** | 20 min | 5.5 min | **3.6× faster** |
| **Memory** | 3 GB | 3 GB | Same (acceptable) |
| **Tracking speed** | Fast | Fast | Same (already good) |

### 9.3 Implementation Effort

**Total: ~28 hours over 2-3 weeks**
- Week 1: Core infrastructure (16 hours)
- Week 2: Integration and testing (12 hours)
- Week 3: Polish (optional)

### 9.4 Recommended Next Steps

**Option A: Implement now** (recommended if bottleneck is critical)
1. Start with Task 1.1: AMR change detection
2. Then Task 1.2: Incremental octree builder
3. Then Task 1.3: Hierarchical field class
4. Test with your data

**Option B: Profile first** (recommended to confirm bottleneck)
1. Let current spatial batching test finish
2. Measure actual time breakdown
3. Confirm octree building is bottleneck
4. Then implement optimization

**Option C: Partial implementation** (quickest win)
1. Just implement "skip mesh detection" (30 min)
2. Measure if remaining time is acceptable
3. Defer hierarchical/incremental if not critical

### 9.5 Questions to Resolve

Before implementation, please clarify:

1. **Mesh evolution pattern**: Does refinement actually follow tool rotation?
2. **Hierarchy assumption**: Are steps 0-9 actually progressive refinement?
3. **Performance target**: Is 11.5 min startup acceptable, or need faster?
4. **Memory constraints**: Confirm GPU has 4+ GB available
5. **Reusability**: Will you ever re-run same dataset? (affects caching priority)

---

## Appendix: Code Snippets

### A. Simple Mesh Change Detection

```python
def detect_amr_changes_simple(mesh_old, mesh_new):
    """
    Simple change detection assuming:
    - Refinement adds points at end
    - Old points preserved
    """

    n_old = len(mesh_old.points)
    n_new = len(mesh_new.points)

    if n_new == n_old:
        # No topological change
        return AMRChanges(
            added_points=[],
            removed_points=[],
            added_elements=[],
            removed_elements=[],
            affected_regions=[]
        )

    elif n_new > n_old:
        # Refinement: new points added at end
        added_points = list(range(n_old, n_new))

        # Find new elements
        old_elem_set = set(map(tuple, mesh_old.connectivity))
        new_elem_set = set(map(tuple, mesh_new.connectivity))
        added_elements = list(new_elem_set - old_elem_set)

        # Compute affected region
        new_points_positions = mesh_new.points[added_points]
        bbox_min = new_points_positions.min(axis=0)
        bbox_max = new_points_positions.max(axis=0)

        # Expand slightly for safety
        margin = 0.01 * (bbox_max - bbox_min)
        bbox_min -= margin
        bbox_max += margin

        return AMRChanges(
            added_points=added_points,
            removed_points=[],
            added_elements=added_elements,
            removed_elements=[],
            affected_regions=[(bbox_min, bbox_max)]
        )

    else:
        # Coarsening - more complex
        # Fallback to full rebuild
        return None  # Signals full rebuild needed
```

### B. Fast AABB Intersection

```python
@jax.jit
def bbox_intersects_jax(min1, max1, min2, max2):
    """JIT-compiled AABB intersection test."""
    return jnp.all(
        (min1 <= max2) & (max1 >= min2)
    )

def find_affected_nodes_vectorized(octree, affected_regions):
    """Vectorized version for speed."""
    nodes_min = octree.nodes_min  # (N, 3)
    nodes_max = octree.nodes_max  # (N, 3)

    affected_mask = np.zeros(len(nodes_min), dtype=bool)

    for region_min, region_max in affected_regions:
        # Vectorized intersection test
        intersects = (
            (nodes_min[:, 0] <= region_max[0]) & (nodes_max[:, 0] >= region_min[0]) &
            (nodes_min[:, 1] <= region_max[1]) & (nodes_max[:, 1] >= region_min[1]) &
            (nodes_min[:, 2] <= region_max[2]) & (nodes_max[:, 2] >= region_min[2])
        )
        affected_mask |= intersects

    return np.where(affected_mask)[0].tolist()
```

### C. Progress Reporting

```python
class ProgressReporter:
    """Nice progress reporting for octree building."""

    def __init__(self, total_steps, description):
        self.total = total_steps
        self.desc = description
        self.start_time = time.time()

    def update(self, step, extra_info=""):
        elapsed = time.time() - self.start_time
        pct = 100 * step / self.total
        eta = elapsed / step * (self.total - step) if step > 0 else 0

        print(f"   {self.desc}: {step}/{self.total} ({pct:.1f}%) "
              f"| Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s {extra_info}")

# Usage:
progress = ProgressReporter(40, "Building octrees")
for i in range(40):
    # Build octree...
    progress.update(i + 1, f"| Incremental: {is_incremental}")
```
