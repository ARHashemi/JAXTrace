# Dynamic Octree for AMR Welding Simulations
## UPDATED Design Based on User Clarifications

**Date**: 2025-10-09 (Updated)
**Context**: Welding tool rotation with AMR mesh evolution
**Data**: Edgar/FLA case - 160 timesteps total

---

## 🎯 Clarified Requirements

### User-Specified Parameters

**From user clarifications**:

1. **Progressive refinement steps**: User-configurable (e.g., first ~10 steps for Edgar/FLA)
   - NOT fixed - varies by dataset
   - Points may not increase monotonically (elements can split)
   - Need to detect automatically or user-specify

2. **Revolution cycle timesteps**: LAST N steps (e.g., last 40 or 80)
   - **For Edgar/FLA**: Steps 120-159 (last 40 steps)
   - This is where velocity field is used for tracking
   - User configures: `revolution_timesteps=40` or `revolution_timesteps=80`

3. **GPU memory**: 4096 MB total, use 90% = **3.6 GB safe limit**
   - NVIDIA T1000: 4GB GDDR6
   - Safety factor: 90% = 3,686 MB available

4. **Optimization goals**: Current approach is acceptable
   - Hierarchical octree from refinement phase
   - Incremental updates for revolution cycles
   - Skip mesh detection by default (optional)

### Critical Insight: Revolution Steps Are at END

**Previous assumption** (WRONG):
```
Steps 0-9:   Refinement (coarse → fine)
Steps 10-49: Revolution cycles  ❌ INCORRECT
```

**Actual pattern** (CORRECT):
```
Steps 0-N:     Progressive refinement (N ~ 10-20, user-specified)
Steps N to M:  Intermediate/transient (may not be used)
Steps (160-40) to 159: Last 40 steps = REVOLUTION CYCLES ✅
```

**For Edgar/FLA data (160 total steps)**:
- Refinement: Steps 0-10 (approximately)
- Revolution: Steps 120-159 (last 40 steps)
- Load ONLY revolution steps for tracking

---

## 📊 GPU Memory Analysis

### Available Memory
```
GPU: NVIDIA T1000
Total: 4,096 MB
Free: 3,884 MB
Safe limit (90%): 3,686 MB
```

### Memory Requirements (40 revolution timesteps)

Assuming ~780k points per timestep (from Edgar/FLA):

| Component | Size Calculation | Total |
|-----------|------------------|-------|
| **Points (40 steps)** | 40 × 780k × 3 × 4B | 375 MB |
| **Velocity (40 steps)** | 40 × 780k × 3 × 4B | 375 MB |
| **Connectivity (shared)** | 1 × 3.5M × 4 × 4B | 56 MB |
| **Element bounds (shared)** | 1 × 3.5M × 6 × 4B | 84 MB |
| **Octrees (40 steps)** | 40 × 50 MB (incremental) | 2,000 MB |
| **JAX overhead** | ~10% | 300 MB |
| **TOTAL** | | **3,190 MB** |

✅ **Fits within 3,686 MB limit** (86% usage)

### For 80 Revolution Timesteps

| Component | Total (80 steps) |
|-----------|------------------|
| Points + Velocity | 750 MB × 2 = 1,500 MB |
| Connectivity + Bounds | 140 MB |
| Octrees | 80 × 50 MB = 4,000 MB |
| **TOTAL** | **5,640 MB** |

❌ **Exceeds 3,686 MB limit** - would need optimization or smaller batch

**Recommendation**:
- 40 timesteps: ✅ Safe
- 60 timesteps: ~4.4 GB (marginal, test needed)
- 80 timesteps: ❌ Too large, need disk streaming or reduce

---

## 🏗️ Redesigned Architecture

### Configuration Structure

```python
user_config = {
    # =========================================================================
    # Data Selection
    # =========================================================================
    'data_pattern': "/path/to/data/*.pvtu",

    # Number of timesteps for tracking (from END of dataset)
    'revolution_timesteps': 40,  # Use LAST 40 timesteps

    # Number of initial refinement timesteps (auto-detect or manual)
    'refinement_timesteps': 'auto',  # Or specific: [0, 2, 5, 8, 10]

    # =========================================================================
    # Mesh Detection (Optional)
    # =========================================================================
    'detect_stable_mesh': False,  # Skip by default
    'detect_refinement_pattern': True,  # Auto-detect refinement steps

    # =========================================================================
    # Hierarchical Octree
    # =========================================================================
    'use_hierarchical_octree': True,
    'incremental_octree_threshold': 0.5,  # Rebuild if >50% changed

    # =========================================================================
    # Octree Parameters
    # =========================================================================
    'max_elements_per_leaf': 32,
    'max_octree_depth': 12,
    'use_advanced_element_search': True,

    # =========================================================================
    # Optional Features
    # =========================================================================
    'enable_octree_cache': False,  # Disk caching (optional)
    'cache_dir': '.cache/octrees',
}
```

### Workflow Logic

```python
class HierarchicalAMROctreeField:
    def __init__(self, data_pattern, revolution_timesteps=40,
                 refinement_timesteps='auto', ...):
        """
        Initialize field for welding simulation tracking.

        Parameters:
        -----------
        data_pattern : str
            Glob pattern for VTK files
        revolution_timesteps : int
            Number of LAST timesteps to load for tracking (default: 40)
        refinement_timesteps : 'auto' or List[int]
            Either 'auto' to detect, or explicit list [0, 2, 5, 10]
        """

        # Step 1: Find all files
        all_files = sorted(glob(data_pattern))
        n_total = len(all_files)

        # Step 2: Select revolution cycle files (LAST N timesteps)
        revolution_start_idx = n_total - revolution_timesteps
        self.revolution_files = all_files[revolution_start_idx:]

        print(f"📁 Total files: {n_total}")
        print(f"📁 Revolution cycle: steps {revolution_start_idx}-{n_total-1} ({revolution_timesteps} timesteps)")

        # Step 3: Detect or use refinement timesteps
        if refinement_timesteps == 'auto':
            self.refinement_steps = self._detect_refinement_pattern(all_files)
        else:
            self.refinement_steps = refinement_timesteps

        print(f"🌲 Refinement hierarchy: steps {self.refinement_steps}")

        # Step 4: Load refinement meshes (for hierarchy building)
        self.refinement_meshes = []
        for step_idx in self.refinement_steps:
            mesh = self._load_vtk_file(all_files[step_idx])
            self.refinement_meshes.append(mesh)

        # Step 5: Build hierarchical octree from refinement steps
        self.refinement_octrees = self._build_hierarchical_octrees()

        # Step 6: Load revolution cycle meshes
        self.revolution_meshes = []
        for filepath in self.revolution_files:
            mesh = self._load_vtk_with_velocity(filepath)
            self.revolution_meshes.append(mesh)
            self.velocities.append(mesh.velocity)

        print(f"✅ Loaded {len(self.revolution_meshes)} revolution timesteps")

        # Step 7: Build octrees for revolution steps (incremental)
        self.revolution_octrees = self._build_revolution_octrees()

    def _detect_refinement_pattern(self, files):
        """
        Auto-detect refinement pattern from initial timesteps.

        Strategy:
        1. Check first 15-20 files for mesh size
        2. Find where mesh size stabilizes (< 0.5% change)
        3. Select ~5 representative steps for hierarchy

        Returns:
        --------
        refinement_steps : List[int]
            Indices like [0, 3, 6, 9, 12]
        """

        print(f"🔍 Auto-detecting refinement pattern...")

        mesh_sizes = []
        for idx in range(min(20, len(files))):
            n_points = self._read_mesh_size_fast(files[idx])
            mesh_sizes.append((idx, n_points))
            print(f"   Step {idx}: {n_points:,} points")

        # Find stabilization point
        stable_idx = None
        for i in range(1, len(mesh_sizes)):
            prev_size = mesh_sizes[i-1][1]
            curr_size = mesh_sizes[i][1]
            change_pct = abs((curr_size - prev_size) / prev_size) * 100

            if change_pct < 0.5:  # < 0.5% change
                stable_idx = i
                break

        if stable_idx is None:
            stable_idx = min(10, len(mesh_sizes) - 1)

        print(f"   Mesh stabilizes around step {stable_idx} ({mesh_sizes[stable_idx][1]:,} points)")

        # Select ~5 representative steps
        if stable_idx <= 5:
            refinement_steps = list(range(stable_idx + 1))
        else:
            # Spread evenly: 0, stable/4, stable/2, 3*stable/4, stable
            refinement_steps = [
                0,
                stable_idx // 4,
                stable_idx // 2,
                3 * stable_idx // 4,
                stable_idx
            ]

        print(f"   Selected refinement steps: {refinement_steps}")
        return refinement_steps

    def _build_revolution_octrees(self):
        """
        Build octrees for revolution cycle using incremental updates.

        Strategy:
        1. Start from finest refinement level as base
        2. For each revolution step, incremental update from previous
        3. Fall back to full rebuild if >50% changed
        """

        print(f"🔄 Building octrees for {len(self.revolution_meshes)} revolution steps...")

        base_octree = self.refinement_octrees[-1]  # Finest refinement level
        base_mesh = self.refinement_meshes[-1]

        revolution_octrees = []

        for idx, mesh in enumerate(self.revolution_meshes):
            if idx == 0:
                # First revolution step: update from base refinement
                prev_octree = base_octree
                prev_mesh = base_mesh
            else:
                # Subsequent steps: update from previous revolution step
                prev_octree = revolution_octrees[-1]
                prev_mesh = self.revolution_meshes[idx - 1]

            # Detect changes
            changes = detect_amr_changes(prev_mesh, mesh)

            # Incremental update
            octree = incremental_octree_build(
                prev_octree, prev_mesh, mesh, changes
            )

            revolution_octrees.append(octree)

            if (idx + 1) % 10 == 0:
                print(f"   Built {idx + 1}/{len(self.revolution_meshes)} octrees")

        print(f"✅ Revolution octrees complete")
        return revolution_octrees

    def get_octree_for_tracking(self, tracking_step_idx):
        """
        Get octree for tracking timestep.

        Note: tracking_step_idx is relative to revolution cycle (0-39 for 40 steps)
        """
        return self.revolution_octrees[tracking_step_idx]

    def sample_at_positions(self, positions, t):
        """
        Sample velocity at positions and time.

        Note: t is tracking time, mapped to revolution cycle indices
        """

        # Map tracking time to revolution step indices
        t_idx_left, t_idx_right, alpha = self._map_time_to_revolution_indices(t)

        # Get octrees and velocities
        octree_left = self.revolution_octrees[t_idx_left]
        octree_right = self.revolution_octrees[t_idx_right]

        v_left = self._interpolate_spatial(positions, t_idx_left, octree_left)
        v_right = self._interpolate_spatial(positions, t_idx_right, octree_right)

        # Temporal interpolation
        v = v_left + alpha * (v_right - v_left)

        return v
```

---

## 🔍 AMR Change Detection Strategy

### Challenge: Non-Monotonic Point Count

**Issue**: Element splitting doesn't always increase point count
- Splitting 1 element → 2 elements may reuse existing nodes
- Point count can stay same or even decrease (if coarsening elsewhere)

**Solution**: Multi-criteria change detection

```python
def detect_amr_changes(mesh_old, mesh_new):
    """
    Detect AMR changes using multiple criteria.

    Criteria:
    1. Point count change
    2. Connectivity changes (elements added/removed/modified)
    3. Point position changes (nodes moved)
    4. Topology changes (connectivity matrix differs)
    """

    changes = AMRChanges()

    # Criterion 1: Point count
    n_old = len(mesh_old.points)
    n_new = len(mesh_new.points)

    if n_new != n_old:
        # Points added or removed
        if n_new > n_old:
            # Assume new points at end (common AMR pattern)
            changes.added_points = list(range(n_old, n_new))
        else:
            # Points removed - need to detect which
            changes.removed_points = detect_removed_points(mesh_old, mesh_new)

    # Criterion 2: Connectivity changes
    old_conn_set = set(map(tuple, mesh_old.connectivity))
    new_conn_set = set(map(tuple, mesh_new.connectivity))

    changes.added_elements = list(new_conn_set - old_conn_set)
    changes.removed_elements = list(old_conn_set - new_conn_set)

    # Criterion 3: Point position changes (if same number of points)
    if n_new == n_old:
        # Check if points moved significantly
        displacements = np.linalg.norm(mesh_new.points - mesh_old.points, axis=1)
        moved_threshold = 0.001  # 0.1% of domain size
        domain_size = np.max(mesh_new.points) - np.min(mesh_new.points)
        moved_points = np.where(displacements > moved_threshold * domain_size)[0]

        if len(moved_points) > 0:
            changes.moved_points = moved_points.tolist()

    # Compute affected regions
    changes.affected_regions = compute_affected_bboxes(changes, mesh_new)

    # Estimate change percentage
    total_elements = len(mesh_new.connectivity)
    changed_elements = len(changes.added_elements) + len(changes.removed_elements)
    changes.change_percentage = (changed_elements / total_elements) * 100

    return changes


@dataclass
class AMRChanges:
    """Container for AMR mesh changes."""
    added_points: List[int] = field(default_factory=list)
    removed_points: List[int] = field(default_factory=list)
    moved_points: List[int] = field(default_factory=list)
    added_elements: List[tuple] = field(default_factory=list)
    removed_elements: List[tuple] = field(default_factory=list)
    affected_regions: List[Tuple] = field(default_factory=list)
    change_percentage: float = 0.0
```

---

## 📈 Expected Performance (Updated)

### For Edgar/FLA Case (160 total, 40 revolution steps)

**Current implementation** (no optimization):
```
Mesh detection:        10 min  (scan all 160 files)
VTK loading:           5 min   (load 40 revolution timesteps)
Refinement octrees:    1 min   (5 steps × 12 sec, small meshes)
Revolution octrees:    20 min  (40 × 30 sec, full rebuild each)
JAX conversion:        2 min
──────────────────────────────
Total:                 38 min
```

**With hierarchical + incremental**:
```
Mesh detection:        0 min   (SKIPPED)
Refinement detection:  1 min   (scan first 15 files)
VTK loading:           5 min   (load 40 revolution timesteps)
Hierarchical octree:   30 sec  (5 levels, incremental)
Revolution octrees:    5 min   (40 × 7.5 sec, incremental ~75% reuse)
JAX conversion:        2 min
──────────────────────────────
Total:                 13.5 min
```

**Speedup**: 38 min → 13.5 min = **2.8× faster**

### Performance Breakdown

| Component | Current | Optimized | Speedup |
|-----------|---------|-----------|---------|
| Mesh detection | 10 min | 0 min | ∞× |
| Refinement detection | 0 min | 1 min | - |
| VTK loading | 5 min | 5 min | 1× |
| Refinement octrees | 1 min | 30 sec | 2× |
| Revolution octrees | 20 min | 5 min | **4×** |
| JAX conversion | 2 min | 2 min | 1× |
| **TOTAL** | 38 min | 13.5 min | **2.8×** |

---

## 📋 Implementation Tasks (Updated)

### Phase 1: Core Infrastructure (16 hours)

#### Task 1.1: AMR Change Detection (4 hours)
**File**: `jaxtrace/fields/amr_mesh_utils.py`

**Functions to implement**:
```python
def detect_amr_changes(mesh_old, mesh_new) -> AMRChanges
def compute_affected_bboxes(changes, mesh) -> List[Tuple]
def detect_removed_points(mesh_old, mesh_new) -> List[int]
```

**Test cases**:
- Same mesh → no changes
- Refinement → added elements
- Coarsening → removed elements
- Point movement → moved points

#### Task 1.2: Refinement Pattern Detection (3 hours)
**File**: `jaxtrace/fields/refinement_detector.py`

**Functions**:
```python
def detect_refinement_pattern(files, max_check=20) -> List[int]
def read_mesh_size_fast(filename) -> Tuple[int, int]
def find_stabilization_point(mesh_sizes) -> int
```

#### Task 1.3: Incremental Octree Builder (6 hours)
**File**: `jaxtrace/fields/incremental_octree_builder.py`

**Functions**:
```python
def incremental_octree_build(octree_old, mesh_old, mesh_new, changes)
def find_affected_octree_nodes(octree, affected_regions)
def rebuild_octree_branch(octree, node_idx, elements, mesh)
```

#### Task 1.4: Hierarchical AMR Field (3 hours)
**File**: `jaxtrace/fields/hierarchical_amr_octree.py`

**Main class with updated logic for revolution cycle loading**

### Phase 2: Integration (12 hours)

#### Task 2.1: Configuration Updates (2 hours)
- Add `revolution_timesteps` parameter
- Add `refinement_timesteps` parameter ('auto' or list)
- Update example_workflow.py

#### Task 2.2: Testing with Edgar/FLA Data (6 hours)
- Test refinement detection
- Test revolution octree building
- Measure performance
- Verify tracking results

#### Task 2.3: Optional Features (4 hours)
- Disk caching (if user wants it)
- Progress reporting
- Memory monitoring

---

## ✅ Validation Checklist

Before implementation:
- [x] Confirmed revolution cycles are LAST 40 steps (120-159)
- [x] Confirmed GPU memory: 4GB, use 90% = 3.6GB safe
- [ ] Measure actual mesh sizes from files (running in background)
- [ ] Confirm refinement pattern (first ~10 steps)
- [ ] Check percentage of mesh change per revolution step

After implementation:
- [ ] Verify refinement auto-detection works
- [ ] Confirm revolution octrees build correctly
- [ ] Measure total startup time (target: <15 min)
- [ ] Check GPU memory usage (target: <3.6 GB)
- [ ] Validate tracking results match original

---

## 🎓 Summary of Changes from Original Design

### What Changed:

1. **Revolution timesteps location**: Last N steps, not steps 10-49
   - **Before**: Steps 10-49 assumed middle of dataset
   - **After**: Steps (160-40) to 159 = steps 120-159

2. **Refinement timesteps**: User-configurable or auto-detect
   - **Before**: Fixed [0, 2, 4, 6, 9]
   - **After**: Auto-detect from first ~15 files or user-specify

3. **Point count**: May not be monotonic during refinement
   - **Before**: Assumed monotonic increase
   - **After**: Use multi-criteria change detection

4. **GPU memory**: 90% safe limit
   - **Before**: Assumed full 4GB
   - **After**: 3.6GB safe limit (90% of 4GB)

5. **Memory budget**: More precise for 40 revolution steps
   - **Total**: 3.2 GB (within 3.6 GB safe limit)
   - **Can support**: Up to ~55 timesteps before exceeding limit

### What Stayed the Same:

- ✅ Skip mesh detection by default
- ✅ Load all revolution timesteps to GPU
- ✅ Hierarchical octree from refinement steps
- ✅ Incremental updates for revolution cycles
- ✅ Disk caching optional (not default)

---

## 🚀 Ready for Implementation

**Next steps**:
1. Wait for mesh analysis to complete (running in background)
2. Confirm refinement pattern and revolution changes
3. Begin Phase 1 implementation

The design is now updated with your clarifications and ready to implement!
