# Octree and Mesh Loading Optimization for AMR Data
## Bottleneck Analysis and Optimization Strategy

**Date**: 2025-10-09
**Context**: AMR (Adaptive Mesh Refinement) simulation data with variable mesh topology
**Problem**: Mesh loading and octree building are too time-consuming and memory-intensive

---

## Table of Contents
1. [Current Implementation Analysis](#current-implementation)
2. [Bottleneck Identification](#bottlenecks)
3. [Optimization Opportunities](#opportunities)
4. [Proposed Solutions](#solutions)
5. [Implementation Plan](#implementation)

---

## 1. Current Implementation Analysis {#current-implementation}

### 1.1 Workflow Overview

```
User starts workflow
    ↓
Step 1: Detect stable mesh (SLOW - scans ALL 160 files)
    ├─ Open each VTK file
    ├─ Read only metadata (number of points)
    ├─ Find most common mesh size
    └─ Filter files → ~5-10 minutes for 160 files
    ↓
Step 2: Load mesh and velocity data (SLOW)
    ├─ Read first file for connectivity
    ├─ Load max_timesteps_to_load files (default: 20)
    ├─ Extract velocity for each timestep
    └─ ~2-5 minutes
    ↓
Step 3: Build single octree (VERY SLOW)
    ├─ Compute element bounds (all 3.5M elements)
    ├─ Recursively subdivide space
    ├─ Assign elements to octree nodes
    └─ ~30-60 seconds for 780k points, 3.5M elements
    ↓
Step 4: Convert to JAX arrays
    ↓
Step 5: Track particles
```

### 1.2 Code Locations

#### Mesh Detection (example_workflow.py:560-610)
```python
# Current: Scans ALL files to detect stable mesh
for i, file in enumerate(files):  # 160 files!
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(file)
    reader.Update()
    mesh = reader.GetOutput()
    n_points = mesh.GetNumberOfPoints()
    # ...
```

**Issue**: Opens every single file just to read metadata.

#### VTK Loading (example_workflow.py:618-683)
```python
# Load first file for mesh structure
reader.SetFileName(files_to_load[0])
reader.Update()
mesh = reader.GetOutput()
points = vtk_to_numpy(mesh.GetPoints().GetData())

# Extract connectivity
for i in range(mesh.GetNumberOfCells()):
    cell = mesh.GetCell(i)
    if cell.GetCellType() == vtk.VTK_TETRA:
        # Extract tetrahedral nodes
        connectivity.append(...)

# Load velocity for each timestep
for filename in files_to_load:
    reader.SetFileName(filename)
    reader.Update()
    velocity = vtk_to_numpy(...)
    velocity_data.append(velocity)
```

**Issue**: Loads all timesteps upfront, even though only a few are needed initially.

#### Octree Building (octree_fem_interpolator_optimized.py:41-224)
```python
def build_octree_mesh_optimized(points, connectivity):
    # 1. Compute element bounds - O(M)
    for elem_idx in range(n_elements):  # 3.5M elements
        node_indices = connectivity[elem_idx]
        elem_points = points[node_indices]
        elem_min = elem_points.min(axis=0)
        elem_max = elem_points.max(axis=0)
        element_bounds[elem_idx] = (elem_min, elem_max)

    # 2. Recursive subdivision - O(M × log(M))
    def subdivide_node(min_corner, max_corner, elem_indices, depth):
        # For each element, test intersection with 8 octants
        for elem_idx in elem_indices:
            for octant_idx in range(8):
                if bbox_intersects(element_bounds[elem_idx], octant_bounds[octant_idx]):
                    octant_elements[octant_idx].append(elem_idx)

        # Recursively subdivide children
        for octant_idx in range(8):
            subdivide_node(octant_bounds[octant_idx], ...)
```

**Issue**: Builds entire octree from scratch for each timestep, even though mesh is mostly stable.

### 1.3 Current Performance

Based on your AMR data (Edgar/FLA):

| Operation | Time | Notes |
|-----------|------|-------|
| **Mesh detection** | 5-10 min | Scans 160 files |
| **VTK loading** | 2-5 min | Loads 20 timesteps |
| **Octree building** | 30-60 sec | 780k points, 3.5M elements |
| **JAX conversion** | 10-20 sec | Large arrays |
| **Total startup** | **8-15 min** | Before any tracking! |

For comparison:
- Particle tracking: ~0.1-1 sec per timestep (fast)
- **Bottleneck ratio**: Setup time is 500-900× longer than tracking

---

## 2. Bottleneck Identification {#bottlenecks}

### 2.1 Primary Bottlenecks

#### Bottleneck 1: Redundant Mesh Detection (CRITICAL)
**What**: Scans all 160 files to detect stable mesh
**Why slow**: Opens and reads VTK metadata 160 times
**Time**: 5-10 minutes
**Why necessary**: Assumption is that mesh varies significantly

**Reality check**:
- Your AMR data has most files with same mesh size (780922 points)
- Only a few files have different sizes (refinement events)
- **Don't need to scan all files** - can sample or cache

#### Bottleneck 2: Upfront Loading (MAJOR)
**What**: Loads max_timesteps_to_load (20) timesteps at start
**Why slow**: VTK I/O is expensive (~5-15 sec per file)
**Time**: 2-5 minutes
**Why unnecessary**: Spatial batching only needs a few timesteps active at once

**Better approach**: Lazy loading (load on-demand)

#### Bottleneck 3: Complete Octree Rebuild (MAJOR)
**What**: Builds octree from scratch for each new mesh
**Why slow**: O(M × log M) complexity with M = 3.5M elements
**Time**: 30-60 seconds per mesh
**Why unnecessary**: For AMR, most of the mesh stays the same!

**Key insight**: AMR refinement is **localized**
- Most mesh regions unchanged between timesteps
- Only small regions get refined/coarsened
- Could reuse 80-90% of octree structure

#### Bottleneck 4: No Caching (MODERATE)
**What**: Recomputes everything on each run
**Why slow**: Octree build + JAX conversion repeated
**Time**: 1-2 minutes
**Why unnecessary**: For same dataset, octree structure is deterministic

**Better approach**: Cache octree to disk

### 2.2 Memory Consumption

**Current memory footprint** (for 20 timesteps, 780k points, 3.5M elements):

| Component | Size per timestep | Total (20 timesteps) |
|-----------|-------------------|---------------------|
| Points | 780k × 3 × 4B = 9.4 MB | 188 MB |
| Connectivity | 3.5M × 4 × 4B = 56 MB | 56 MB (shared) |
| Velocity | 780k × 3 × 4B = 9.4 MB | 188 MB |
| Octree structure | ~100 MB | 100 MB (shared) |
| Element bounds | 3.5M × 6 × 4B = 84 MB | 84 MB |
| **Total** | - | **~600 MB** |

**Issue**: Loads all 20 timesteps into memory, but spatial batching only uses 1-2 at a time.

### 2.3 AMR Characteristics Analysis

Based on typical AMR behavior and your data:

**Mesh evolution pattern**:
```
Timestep 0-50:   Initial mesh (780k points) - stable
Timestep 51-52:  Refinement event → 950k points
Timestep 53-100: Refined mesh (950k points) - stable
Timestep 101-102: Coarsening → back to 780k points
...
```

**Spatial locality of refinement**:
- Refinement typically happens in high-gradient regions (e.g., near weld pool)
- ~80-90% of domain remains unchanged
- Only 10-20% of elements are added/removed/modified

**Implications**:
1. Don't need to rebuild entire octree
2. Can detect mesh changes incrementally
3. Can reuse octree branches for unchanged regions

---

## 3. Optimization Opportunities {#opportunities}

### 3.1 Quick Wins (Low Effort, High Impact)

#### Opportunity 1: Skip Mesh Detection (1 hour work, 5-10 min saved)
**Current**: Scans all 160 files
**Better**: Use first file's mesh size, load adaptively

```python
# Instead of scanning all files:
if use_stable_mesh_only:
    first_mesh_size = read_mesh_size(files[0])

    # Only check subsequent files when loading
    # If mesh size differs, handle gracefully
```

**Savings**: 5-10 minutes → 0 seconds

#### Opportunity 2: Lazy Timestep Loading (2 hours, major impact)
**Current**: Load all 20 timesteps upfront
**Better**: Load on-demand when needed for tracking

```python
class LazyTimeSeriesField:
    def __init__(self, file_pattern):
        self.files = glob(file_pattern)
        self.cache = {}  # LRU cache

    def get_timestep(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self._load_timestep(idx)
            # Evict oldest if cache full
        return self.cache[idx]
```

**Savings**: Load time: 2-5 min → ~10 sec (only load first timestep initially)

#### Opportunity 3: Octree Disk Caching (3 hours, 8× speedup on reruns)
**Current**: Rebuild octree every run
**Better**: Save octree to disk, load if unchanged

```python
def load_or_build_octree(mesh_file, cache_dir):
    cache_file = f"{cache_dir}/{hash(mesh_file)}_octree.npz"

    if os.path.exists(cache_file):
        # Check if mesh file is newer than cache
        if os.path.getmtime(mesh_file) < os.path.getmtime(cache_file):
            return load_octree_from_cache(cache_file)  # 2-5 sec

    # Build and cache
    octree = build_octree(mesh_file)  # 30-60 sec
    save_octree_to_cache(octree, cache_file)
    return octree
```

**Savings**: 30-60 sec → 2-5 sec (8-12× faster on subsequent runs)

### 3.2 Medium Effort Optimizations (1-3 days work)

#### Opportunity 4: Parallel VTK Loading (1 day, 3-4× speedup)
**Current**: Sequential loading
**Better**: Parallel loading with ThreadPoolExecutor

```python
from concurrent.futures import ThreadPoolExecutor

def load_timesteps_parallel(files, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_vtk_file, f): i
                   for i, f in enumerate(files)}

        results = {}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return results
```

**Savings**: 2-5 min → 30-75 sec (3-4× faster)

**Caveat**: Only helps if I/O bound (not CPU bound)

#### Opportunity 5: Incremental Mesh Detection (1-2 days)
**Current**: Scan all files
**Better**: Build index file once, reuse

```python
# Run once, create index
def build_mesh_index(file_pattern):
    files = glob(file_pattern)
    index = {}

    for i, f in enumerate(files):
        index[i] = {
            'file': f,
            'n_points': read_mesh_size(f),
            'timestamp': extract_timestamp(f)
        }

    # Save to JSON
    json.dump(index, open('mesh_index.json', 'w'))

# Subsequent runs: load index
index = json.load(open('mesh_index.json'))
stable_files = [f for f in index if f['n_points'] == target_size]
```

**Savings**: 5-10 min → 0.1 sec (50-100× faster)

### 3.3 Advanced Optimizations (1-2 weeks work)

#### Opportunity 6: Incremental Octree Updates (THE BIG ONE)
**Concept**: Reuse octree structure for unchanged mesh regions

**Algorithm**:
```python
def incremental_octree_update(old_octree, old_mesh, new_mesh):
    """
    Update octree incrementally when mesh changes.

    Strategy:
    1. Detect which mesh regions changed
    2. Invalidate only affected octree nodes
    3. Rebuild only those branches
    4. Reuse ~80-90% of old octree
    """

    # Step 1: Detect changed elements
    changed_elements = detect_mesh_changes(old_mesh, new_mesh)

    if len(changed_elements) == 0:
        # Mesh unchanged - reuse octree entirely!
        return old_octree

    if len(changed_elements) > 0.3 * len(new_mesh):
        # Too many changes (>30%) - full rebuild faster
        return build_octree_from_scratch(new_mesh)

    # Step 2: Find affected octree nodes
    affected_nodes = set()
    for elem_idx in changed_elements:
        elem_bbox = compute_element_bbox(new_mesh, elem_idx)
        nodes = find_octree_nodes_intersecting_bbox(old_octree, elem_bbox)
        affected_nodes.update(nodes)

    # Step 3: Rebuild only affected branches
    new_octree = copy_octree(old_octree)
    for node_idx in affected_nodes:
        # Collect all elements in this node's region
        node_bbox = new_octree.nodes[node_idx].bbox
        elements_in_region = find_elements_in_bbox(new_mesh, node_bbox)

        # Rebuild this branch
        rebuild_octree_branch(new_octree, node_idx, elements_in_region)

    return new_octree
```

**Key functions**:

```python
def detect_mesh_changes(old_mesh, new_mesh):
    """
    Detect which elements changed between meshes.

    For AMR, typical changes:
    - Element added (new node IDs)
    - Element removed (missing from new mesh)
    - Element moved (nodes displaced)
    """

    if old_mesh.connectivity.shape != new_mesh.connectivity.shape:
        # Different topology - find differences
        old_elems = set(map(tuple, old_mesh.connectivity))
        new_elems = set(map(tuple, new_mesh.connectivity))

        added = new_elems - old_elems
        removed = old_elems - new_elems

        return list(added) + list(removed)

    else:
        # Same topology - check if nodes moved significantly
        changed = []
        for elem_idx in range(len(old_mesh.connectivity)):
            old_centroid = compute_element_centroid(old_mesh, elem_idx)
            new_centroid = compute_element_centroid(new_mesh, elem_idx)

            if np.linalg.norm(old_centroid - new_centroid) > threshold:
                changed.append(elem_idx)

        return changed
```

**Expected speedup**:
- Unchanged mesh: Instant (0 sec vs 30 sec) - **∞× faster!**
- 10% changed: ~5 sec (vs 30 sec) - **6× faster**
- 30% changed: ~15 sec (vs 30 sec) - **2× faster**
- >50% changed: Fall back to full rebuild - same speed

#### Opportunity 7: Hierarchical Mesh Representation
**Concept**: Store mesh at multiple resolutions

```python
class HierarchicalMesh:
    """
    Multi-resolution mesh representation.

    Levels:
    0: Coarse (10k elements) - for fast preview
    1: Medium (500k elements) - for particle initialization
    2: Fine (3.5M elements) - for accurate tracking
    """

    def __init__(self, full_mesh):
        self.levels = [
            coarsen_mesh(full_mesh, factor=350),  # Level 0
            coarsen_mesh(full_mesh, factor=7),     # Level 1
            full_mesh                               # Level 2
        ]

    def get_appropriate_level(self, query_density):
        """Select mesh resolution based on query needs."""
        if query_density < 100:
            return self.levels[0]  # Coarse
        elif query_density < 10000:
            return self.levels[1]  # Medium
        else:
            return self.levels[2]  # Fine
```

**Use case**: During particle initialization, use coarse mesh (fast octree), then switch to fine for tracking.

---

## 4. Proposed Solutions {#solutions}

### 4.1 Solution 1: Lazy Loading + Disk Caching (RECOMMENDED)

**Changes required**:

#### A. Implement lazy timestep loading
```python
# File: jaxtrace/fields/lazy_time_series.py

class LazyOctreeFEMTimeSeries:
    def __init__(self, file_pattern, max_cache_size=3):
        self.files = sorted(glob(file_pattern))
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.octree_cache_dir = ".cache/octrees"

    def load_timestep(self, idx):
        """Load timestep on-demand with caching."""
        if idx in self.cache:
            return self.cache[idx]

        # Load mesh
        mesh = self._load_vtk_file(self.files[idx])

        # Load or build octree
        octree = self._load_or_build_octree(mesh, idx)

        # Cache
        self.cache[idx] = (mesh, octree)

        # Evict oldest if over limit
        if len(self.cache) > self.max_cache_size:
            oldest = min(self.cache.keys())
            del self.cache[oldest]

        return self.cache[idx]

    def _load_or_build_octree(self, mesh, idx):
        """Load cached octree or build new one."""
        cache_file = f"{self.octree_cache_dir}/octree_{idx:04d}.npz"

        if os.path.exists(cache_file):
            # Check freshness
            mesh_mtime = os.path.getmtime(self.files[idx])
            cache_mtime = os.path.getmtime(cache_file)

            if cache_mtime > mesh_mtime:
                # Cache is fresh
                return self._load_octree_from_cache(cache_file)

        # Build and cache
        octree = build_octree_mesh_optimized(mesh.points, mesh.connectivity)
        self._save_octree_to_cache(octree, cache_file)
        return octree
```

#### B. Skip mesh detection
```python
# File: example_workflow.py

# REMOVE this entire section (lines 559-610):
if use_stable_mesh_only and len(files) > 3:
    print(f"   Detecting stable mesh size...")
    # ... scan all files ...

# REPLACE with:
print(f"   Using lazy loading (will detect mesh changes on-demand)")
```

#### C. Modify workflow to use lazy loading
```python
# File: example_workflow.py

# OLD:
velocity_data = []
for filename in files_to_load:
    # Load all upfront

# NEW:
field = LazyOctreeFEMTimeSeries(
    file_pattern=data_pattern,
    max_cache_size=3
)
```

**Expected results**:
- Startup time: 8-15 min → **10-20 sec** (50-70× faster)
- Subsequent runs: 8-15 min → **2-5 sec** (with cached octrees)
- Memory: 600 MB → **100-200 MB** (only 2-3 timesteps in memory)

### 4.2 Solution 2: Incremental Octree Updates (ADVANCED)

**For when mesh changes between timesteps**:

```python
class IncrementalOctreeField:
    def __init__(self):
        self.current_mesh = None
        self.current_octree = None

    def update_to_timestep(self, idx):
        new_mesh = self._load_mesh(idx)

        if self.current_mesh is None:
            # First timestep - build from scratch
            self.current_octree = build_octree(new_mesh)
        else:
            # Incremental update
            self.current_octree = incremental_octree_update(
                self.current_octree,
                self.current_mesh,
                new_mesh
            )

        self.current_mesh = new_mesh
```

**When to use**:
- Mesh topology changes significantly between timesteps
- Still want fast updates for localized refinement
- Need to track through entire AMR evolution

### 4.3 Solution 3: Mesh Index File (SIMPLE)

**One-time preprocessing**:

```python
# Script: tools/build_mesh_index.py

def build_mesh_index(data_pattern, output_file):
    """Build searchable index of all mesh files."""
    files = sorted(glob(data_pattern))

    index = {
        'files': [],
        'metadata': {}
    }

    for i, f in enumerate(files):
        # Read only metadata (fast)
        n_points = read_mesh_size_fast(f)
        timestamp = extract_timestamp(f)

        index['files'].append({
            'idx': i,
            'path': f,
            'n_points': n_points,
            'timestamp': timestamp
        })

        if (i + 1) % 20 == 0:
            print(f"Indexed {i+1}/{len(files)} files...")

    # Save index
    with open(output_file, 'w') as fp:
        json.dump(index, fp, indent=2)

    print(f"✅ Index saved to {output_file}")
```

**Use in workflow**:
```python
# Load pre-built index
with open('mesh_index.json') as fp:
    index = json.load(fp)

# Find files with target mesh size
target_size = 780922
matching_files = [f for f in index['files']
                  if f['n_points'] == target_size]

print(f"Found {len(matching_files)} files with {target_size} points")
```

**Benefit**: 5-10 min mesh detection → 0.1 sec index load

---

## 5. Implementation Plan {#implementation}

### Phase 1: Quick Wins (Week 1)

**Goal**: Reduce startup time from 8-15 min to <1 min

#### Task 1.1: Implement octree disk caching
- **File**: `jaxtrace/fields/octree_fem_interpolator_optimized.py`
- **Add**: Save/load functions for octree
- **Effort**: 2-3 hours
- **Speedup**: 8× on reruns

#### Task 1.2: Skip mesh detection
- **File**: `example_workflow.py`
- **Change**: Remove scan-all-files loop
- **Effort**: 30 minutes
- **Speedup**: 5-10 min saved

#### Task 1.3: Lazy loading for octree field
- **File**: New `jaxtrace/fields/lazy_octree_time_series.py`
- **Change**: Load timesteps on-demand
- **Effort**: 4-6 hours
- **Speedup**: Load only what's needed (3-5× faster startup)

**Expected result after Phase 1**:
- First run: 8-15 min → 30-60 sec
- Subsequent runs: 8-15 min → 5-10 sec
- **Total: 15-90× faster**

### Phase 2: Medium Optimizations (Week 2-3)

#### Task 2.1: Parallel VTK loading
- **File**: `jaxtrace/io/`
- **Add**: ThreadPoolExecutor for parallel loads
- **Effort**: 1 day
- **Speedup**: 3-4× faster loading

#### Task 2.2: Mesh index file
- **File**: New `tools/build_mesh_index.py`
- **Add**: Index builder + loader
- **Effort**: Half day
- **Speedup**: Instant mesh detection

#### Task 2.3: Optimized mesh change detection
- **File**: `jaxtrace/fields/mesh_utils.py`
- **Add**: Fast mesh comparison
- **Effort**: 1-2 days
- **Benefit**: Foundation for incremental updates

**Expected result after Phase 2**:
- First run: 30-60 sec → 10-20 sec
- Subsequent runs: 5-10 sec → 2-3 sec

### Phase 3: Advanced Features (Week 4-6)

#### Task 3.1: Incremental octree updates
- **File**: `jaxtrace/fields/incremental_octree.py`
- **Add**: Diff-based octree reconstruction
- **Effort**: 1-2 weeks
- **Speedup**: 5-10× for changed meshes

#### Task 3.2: Adaptive resolution
- **File**: `jaxtrace/fields/hierarchical_mesh.py`
- **Add**: Multi-resolution mesh representation
- **Effort**: 1 week
- **Benefit**: Faster initialization, adaptive accuracy

**Expected result after Phase 3**:
- Handle mesh changes: 30 sec → 3-5 sec
- Support larger meshes: 10M+ elements

---

## 6. Summary and Recommendations

### Current Bottlenecks (Ranked by Impact)

1. **Mesh detection** - 5-10 min (CRITICAL) → Skip entirely
2. **Upfront loading** - 2-5 min (MAJOR) → Lazy loading
3. **Octree rebuild** - 30-60 sec (MAJOR) → Disk caching
4. **No parallelization** - 50% waste (MODERATE) → Parallel I/O

### Recommended Immediate Actions

**Priority 1 (Do first - highest ROI)**:
1. Implement octree disk caching (3 hours) → 8× speedup on reruns
2. Skip mesh detection (30 min) → Save 5-10 min
3. Implement lazy loading (6 hours) → Save 2-5 min, reduce memory

**Combined effect**: 8-15 min → 10-30 sec (20-90× faster)

**Priority 2 (Do next - good ROI)**:
4. Build mesh index file (4 hours) → Instant mesh info lookup
5. Parallel VTK loading (1 day) → 3-4× faster I/O

**Priority 3 (Future - diminishing returns)**:
6. Incremental octree updates (2 weeks) → Handle AMR changes efficiently
7. Hierarchical mesh (1 week) → Advanced use cases

### Key Insights for AMR Data

1. **Most files have same mesh** → Don't scan all files
2. **Only need 2-3 timesteps active** → Lazy loading
3. **Octree is deterministic** → Cache to disk
4. **Mesh changes are localized** → Incremental updates

### Expected Performance After Optimizations

| Scenario | Current | After Phase 1 | After Phase 2 |
|----------|---------|---------------|---------------|
| **First run** | 8-15 min | 30-60 sec | 10-20 sec |
| **Subsequent runs** | 8-15 min | 5-10 sec | 2-3 sec |
| **Changed mesh** | 8-15 min | 30-60 sec | 3-5 sec |
| **Memory usage** | 600 MB | 150 MB | 100 MB |

**Target achieved**: <10 sec startup for subsequent runs (50-90× improvement)

---

## Appendix: Code Examples

### A. Octree Disk Caching

```python
def save_octree_to_cache(octree: OctreeMeshOptimized, filepath: str):
    """Save octree to compressed numpy format."""
    np.savez_compressed(
        filepath,
        points=octree.points,
        connectivity=octree.connectivity,
        element_bounds=octree.element_bounds,
        element_centroids=octree.element_centroids,
        nodes_min=octree.nodes_min,
        nodes_max=octree.nodes_max,
        nodes_elements=octree.nodes_elements,
        nodes_elem_counts=octree.nodes_elem_counts,
        nodes_children=octree.nodes_children,
        nodes_is_leaf=octree.nodes_is_leaf,
        max_elements_per_leaf=octree.max_elements_per_leaf,
        max_depth=octree.max_depth
    )

def load_octree_from_cache(filepath: str) -> OctreeMeshOptimized:
    """Load octree from cache."""
    data = np.load(filepath)
    return OctreeMeshOptimized(
        points=jnp.array(data['points']),
        connectivity=jnp.array(data['connectivity']),
        element_bounds=jnp.array(data['element_bounds']),
        element_centroids=jnp.array(data['element_centroids']),
        nodes_min=jnp.array(data['nodes_min']),
        nodes_max=jnp.array(data['nodes_max']),
        nodes_elements=jnp.array(data['nodes_elements']),
        nodes_elem_counts=jnp.array(data['nodes_elem_counts']),
        nodes_children=jnp.array(data['nodes_children']),
        nodes_is_leaf=jnp.array(data['nodes_is_leaf']),
        max_elements_per_leaf=int(data['max_elements_per_leaf']),
        max_depth=int(data['max_depth'])
    )
```

### B. Fast Mesh Size Reading

```python
def read_mesh_size_fast(vtk_file: str) -> int:
    """Read only mesh size without loading full data."""
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(vtk_file)
    reader.UpdateInformation()  # Don't load data, just metadata
    reader.Update()
    return reader.GetOutput().GetNumberOfPoints()
```

### C. Lazy Loading Pattern

```python
from functools import lru_cache

class LazyLoader:
    @lru_cache(maxsize=3)
    def load_timestep(self, idx):
        """Load and cache timestep."""
        return self._load_from_disk(idx)
```
