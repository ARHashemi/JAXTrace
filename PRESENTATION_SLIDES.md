# Slide 1: JAXTrace AMR Development - Key Achievements

## 🎯 Major Improvements (Oct 2-9, 2025)

### ✅ Adaptive Mesh Refinement (AMR) Support
- **Flexible Octree**: AABB intersection-based element assignment (was centroid-based)
- **Critical Fix**: Elements now assigned to ALL overlapping octants
- **Impact**: Supports 7 levels of refinement, finer octrees = better accuracy
- **Auto-detection**: Filters inconsistent meshes, skips AMR warmup automatically

### ✅ Velocity Accuracy Breakthrough
- **Root Cause Found**: Octree bug causing particles to miss containing elements
- **Solution**: Advanced element search checking ALL candidates, selecting best match
- **Verification**: RK4 + FEM math confirmed correct, smooth trajectories achieved

### ✅ Dual Tracking Architecture
| Mode | Build Time | Memory | GPU Use | AMR Support | Status |
|------|-----------|--------|---------|-------------|--------|
| **Octree FEM** | 10s | 400-800 MB | 15-30% | Fixed mesh | ✅ Ready |
| **Temporal Batching** | 0.1s | 50-100 MB | 80-95% | Variable mesh | ⚠️ Large mesh opt. |

### ✅ Memory & Performance
- Optional velocity recording: **50% memory savings**
- Safe VTK export: Prevents crashes
- Clean progress reporting

---

# Slide 2: User-Friendly Configuration

## 📋 Complete Control via `user_config`

### Example Configuration (18 Ready-to-Use Examples Available)
```python
user_config = {
    # ========== AMR DATA ==========
    'data_pattern': "/path/to/data_*.pvtu",
    'skip_initial_timesteps': 20,        # Skip AMR warmup
    'use_stable_mesh_only': True,        # Auto-filter

    # ========== TRACKING METHOD (ONE LINE SWITCH!) ==========
    'use_temporal_batching': False,      # Octree FEM (fixed mesh)
    # OR
    'use_temporal_batching': True,       # Grid hash (AMR)

    # ========== OCTREE FEM ==========
    'max_elements_per_leaf': 32,         # Subdivision threshold
    'max_octree_depth': 12,              # 7 AMR levels supported
    'use_advanced_element_search': True, # Check all elements (accurate!)

    # ========== TEMPORAL BATCHING ==========
    'temporal_window_size': 20,          # GPU window size
    'grid_resolution': 32,               # 32³ grid cells

    # ========== PARTICLES ==========
    'particle_concentrations': {'x': 60, 'y': 50, 'z': 15},
    'particle_bounds_fraction': {'x': (0.0, 0.2), ...},  # Inlet only
    'record_velocities': False,          # Save 50% memory!

    # ========== TRACKING ==========
    'n_timesteps': 2500,
    'use_data_dt': True,                 # Auto-extract from VTK
    'integrator': 'rk4',

    # ========== BOUNDARIES ==========
    'boundary_inlet': 'continuous',      # Particle injection
    'boundary_outlet': 'absorbing',      # Exit domain

    # ========== PERFORMANCE ==========
    'device': 'gpu',
    'memory_limit_gb': 3.0,
}
```

## 🔧 The Fix: Octree Element Assignment

**BEFORE** (Bug):
```python
# Centroid-based → Large elements MISSED!
elem_centroid = compute_centroid(element)
assign_to_single_octant(elem_centroid)
```
❌ **Problem**: Large tetrahedra spanning multiple octants only in ONE octant
❌ **Effect**: Finer octrees made accuracy WORSE!

**AFTER** (Fixed):
```python
# AABB intersection → All overlaps detected!
elem_min, elem_max = element_bounds[elem_idx]
for each of 8 child octants:
    if AABB_intersects(elem_bounds, octant_bounds):
        assign_to_octant(octant)  # Can be in MULTIPLE
```
✅ **Result**: Finer octrees now MORE accurate (as expected!)
✅ **Impact**: Smooth particle trajectories, proper velocity interpolation

## 🎓 Key Results

✅ **7 AMR refinement levels** fully supported
✅ **18 configuration examples** for common scenarios
✅ **100× faster** spatial index build (temporal batching)
✅ **50% memory savings** with optional velocity recording
✅ **One-line switch** between tracking modes
✅ **Auto-detection** of mesh stability
✅ **Production-ready** octree FEM, temporal batching optimization in progress
