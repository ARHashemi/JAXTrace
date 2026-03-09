# JAXTrace Methods Quick Reference

**Date**: 2026-01-28

---

## Point-in-Tet Methods (Choose ONE)

| Method | Speed | Accuracy | Status | Use? |
|--------|-------|----------|--------|------|
| **inverse** ⭐ | **3-4× faster** | **100%** | ✅ Validated | **YES - Production** |
| current | 1.0× (baseline) | 100% | ✅ Reference | Only for validation |
| skala_memory_opt | 0.97× | 100% | ✅ Works | No advantage |
| skala | 0.90× | 100% | ✅ Works | No advantage |
| branchless_hybrid | 0.62× | 93.7% | ❌ Low accuracy | **NO** |
| axis_aligned | 0.45× | 99.4% | ❌ Broken | **NO** |
| pure_aa | 27.49× | 0% | ❌ False positives | **NO** |

**Recommendation:**
```python
config.POINT_IN_TET_METHOD = 'inverse'  # MANDATORY for production
```

---

## L2 Element Search Methods (Choose ONE for RK4)

### For RK4 Tracking (must be vmappable)

| Method | Speed | Retention | Vmappable | Status | Use? |
|--------|-------|-----------|-----------|--------|------|
| **radius=10** ⭐ | **51,894 p/s** | **96.96%** | ✅ | ✅ Fastest | **YES - Speed priority** |
| **incremental (2,4,8,15,30)** | 9,136 p/s | **98.21%** | ✅ | ✅ Works | **YES - Retention priority** |
| incremental (2,5,10) | 31,077 p/s | 96.96% | ✅ | ✅ Works | Alternative |
| radius=30 | 17,895 p/s | 98.21% | ✅ | ✅ Works | Too slow |
| neighbors | 2,378 p/s | 98.21% | ✅ | ⚠️ Very slow | **NO** |
| hierarchical | 2,529 p/s | 98.14% | ✅ | ⚠️ Very slow | **NO** |
| mesh_aligned_octree | Fast (~6 tests) | **74.6%** | ✅ | ❌ Low retention | **NO** |
| mesh_aligned_morton | TBD (~30 tests) | ~98% (expected) | ✅ | ⚠️ Not validated | **Maybe** |
| **kdtree** | 64 tests/particle | **95-100%** | ❌ | ⚠️ Not vmappable | **NO for RK4** |

**Recommendation for RK4:**
```python
# Fast (recommended):
config.L2_SEARCH_METHOD = 'radius'
L2_SEARCH_RADIUS = 10

# OR better retention:
config.L2_SEARCH_METHOD = 'incremental'
INCREMENTAL_SEARCH_RADII = (2, 4, 8, 15, 30)
```

### For Initial Assignment (batch search)

| Method | Speed | Retention | Use? |
|--------|-------|-----------|------|
| Morton cascading | 144 p/s | **100%** | **YES** |
| KD-tree cascading | Varies | **100%** | **YES** |

**Recommendation for Initial Assignment:**
```python
# Use large cascading radii (works for both Morton and KD-tree)
INITIAL_SEARCH_RADIUS = 500
INITIAL_SEARCH_FALLBACK_RADII = [1000, 2000, 5000, 10000, 100000]
# Result: 100% retention
```

---

## Production Configuration Template

```python
import jaxtrace.config as config
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
from jaxtrace.gpu.search.point_in_tet_methods import set_inverse_matrices_gpu

# ============================================================================
# Point-in-Tet: INVERSE (mandatory for performance)
# ============================================================================
config.POINT_IN_TET_METHOD = 'inverse'

# Precompute inverse matrices (one-time, ~29s for 3M elements)
M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)
M_inv_gpu = jax.device_put(M_inv_array)
p0_gpu = jax.device_put(p0_array)
set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)

# ============================================================================
# L2 Search Method: RADIUS (fastest) or INCREMENTAL (better retention)
# ============================================================================
# Option 1: Fixed radius (fastest, 96.96% retention)
config.L2_SEARCH_METHOD = 'radius'
L2_SEARCH_RADIUS = 10

# Option 2: Incremental (better retention 98.21%, but slower)
# config.L2_SEARCH_METHOD = 'incremental'
# INCREMENTAL_SEARCH_RADII = (2, 4, 8, 15, 30)

# ============================================================================
# Initial Assignment: Large cascading radii (100% retention)
# ============================================================================
INITIAL_SEARCH_RADIUS = 500
INITIAL_SEARCH_FALLBACK_RADII = [1000, 2000, 5000, 10000, 100000]

# ============================================================================
# L1 Neighbor Search
# ============================================================================
ENABLE_L1_SEARCH = True
N_HOPS = 5  # Adaptive 5-6 hops at refinement boundaries

# ============================================================================
# Tracking Parameters
# ============================================================================
DT = 0.0005  # Timestep
N_STEPS = 2500  # Number of RK4 steps
```

---

## Expected Performance

**With recommended configuration:**
- **Initial assignment**: 100% (with cascading radii)
- **Final retention** (2,500 steps): 95-98%
- **Throughput**: 50,000-120,000 particles/s
- **Speedup vs baseline**: 2.4× overall (from 4× point-in-tet speedup)

---

## Method Details

### Point-in-Tet: inverse

- **How**: Precomputed inverse matrices transform world→barycentric in single multiply
- **Precomputation**: 28.9s for 3M elements (one-time)
- **Memory**: 139.6 MB
- **Runtime**: Single matrix-vector multiply + 4 comparisons
- **Speedup**: 3-4×
- **Accuracy**: 100% (mathematically equivalent to barycentric)

### L2 Search: radius

- **How**: Binary search Morton curve, test ±R leaves
- **Tests**: 2R+1 leaves (radius=10 → 21 leaves → ~536 element tests)
- **Retention**: 96.96%
- **Throughput**: 51,894 p/s (fastest)

### L2 Search: incremental

- **How**: Cascading radii, start small, expand if not found
- **Tiers**: (2, 4, 8, 15, 30) → 5 adaptive tiers
- **Tests**: ~22.5 leaves average (adaptive)
- **Retention**: 98.21% (better than radius=10)
- **Throughput**: 9,136 p/s (slower than expected - needs investigation)

### L2 Search: kdtree (batch only)

- **How**: Find K nearest mesh nodes, test connected elements
- **Tests**: K×21.4 elements (K=3 → ~64 tests)
- **Retention**: 95-100%
- **Limitation**: NOT vmappable (Python control flow in tree query)
- **Use**: Initial assignment, offline analysis, validation

---

## Do NOT Use

❌ **Point-in-Tet:**
- `pure_aa` - 27× fast but gives **wrong results** (false positives)
- `branchless_hybrid` - Low accuracy (93.7%)
- `axis_aligned` - Broken (99.4%)

❌ **L2 Search for RK4:**
- `neighbors` - 20× slower than radius
- `hierarchical` - 20× slower than radius
- `mesh_aligned_octree` - Only 74.6% retention
- `kdtree` - Not vmappable (use for batch searches only)

---

## Quick Decision Tree

**What are you doing?**

1. **RK4 particle tracking** (per-step search)
   - Point-in-tet: `inverse` ✅
   - L2 search: `radius` (fast) or `incremental` (retention) ✅
   - NOT: `kdtree` (can't vmap) ❌

2. **Initial assignment** (batch search)
   - Morton with cascading radii ✅
   - OR KD-tree with cascading radii ✅
   - Both achieve 100% retention

3. **Offline analysis** (batch particle location)
   - KD-tree works great ✅
   - Morton also works ✅

4. **Maximum speed, acceptable retention (~97%)**
   - Point-in-tet: `inverse` ✅
   - L2: `radius=10` ✅
   - Throughput: ~52K p/s

5. **Maximum retention (~98%)**
   - Point-in-tet: `inverse` ✅
   - L2: `incremental (2,4,8,15,30)` ✅
   - Throughput: ~9K p/s (slower)

---

**For detailed explanations, see [METHODS_PERFORMANCE_REPORT.md](METHODS_PERFORMANCE_REPORT.md)**
