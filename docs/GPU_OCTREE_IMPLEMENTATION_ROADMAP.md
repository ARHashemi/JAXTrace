# GPU-Friendly JAX Octree Implementation Roadmap v2.0

**Research-Backed Phased Optimization Strategy**

**Based On**:
- `SUGGESTION_FOR_GPU_FRIENDLY_OCTREE.md` (original GPU-native proposal)
- `Comparison_Current_JAXTrace_Implementation_vs_Optimized_GPU_Approaches.md` (2023-2025 research analysis)

**Date**: 2025-10-27  
**Version**: 2.0 - REVISED with incremental optimization approach

---

## Executive Summary

### Key Finding: Your Implementation is Strong

**What JAXTrace does WELL** (validated by research):
- ✅ Structure reuse (97.5%) - matches AMR best practices
- ✅ Two-stage architecture - pragmatic solution to JAX limitations
- ✅ Flat array storage - aligns with linear octree standards  
- ✅ Element assignment - standard center-based approach

**Critical optimization opportunities** identified:
1. **Element ID caching** → 5-10× search speedup
2. **JAX io_callback integration** → 5× integration speedup  
3. **Morton code encoding** → 3× memory reduction, 2-3× traversal
4. **Hash-based fine octree** → 3-5× fine query speedup

### Current Baseline (500 particles, 40 timesteps)

```
Total per step: 695 ms
  - CPU search: 120 ms (17.3%)
  - GPU interpolation: 80 ms (11.5%)
  - Integration overhead: 495 ms (71.2%) ← PRIMARY BOTTLENECK

Memory: 1.24 GB
  - Octrees: 1.05 MB
  - Cache: 368 MB
  - GPU: 184 MB
```

**Root cause**: RK4 loop cannot compile (Numba callbacks block JAX tracing)

### Phased Strategy Overview

| Phase | Time | Target | Speedup | Memory | Risk | Status |
|-------|------|--------|---------|--------|------|--------|
| **1: Quick Wins** | 1 week | 100-150 ms | 5-7× | Same | Very Low | **RECOMMENDED** |
| **2: Memory Opt** | 2-3 weeks | 50-80 ms | 9-14× | -67% octree | Low | **RECOMMENDED** |
| **3: GPU-Native** | 2-3 weeks | 5-10 ms | 70-140× | +50% | Medium | **RECOMMENDED** |
| **4: Full Rewrite** | 6-8 weeks | <5 ms | 100+× | +100% | High | Only if >100K particles |

**Recommendation**: Phase 1 + 2 (4-5 weeks) provides **9-14× speedup** with minimal risk.

---

## Phase 1: Quick Wins (1 week) → 5-7× Speedup

**Goal**: Eliminate the 71% integration overhead bottleneck  
**Risk**: Very Low | **Memory**: No change

### Task 1: Element ID Caching (1-2 days)

**Problem**: Re-searching octree for every particle every step, despite slow particle motion.

**Solution**: Cache last-known element, only re-search if displacement > threshold.

**Implementation**: See comparison doc lines 54-92 for complete `ElementCache` class.

**Key code**:
```python
class ElementCache:
    threshold: float = 0.001  # 1mm
    
    def get_elements(self, positions, timestep, search_fn):
        for i, pos in enumerate(positions):
            if i in cache:
                displacement = ||pos - cached_pos||
                if displacement < threshold:
                    return cached_elem  # HIT
            # MISS: call search_fn
```

**Expected**: 120 ms → 15-25 ms (hit rate 85-95%)

### Task 2: JAX io_callback Integration (3-5 days)

**Problem**: RK4 loop can't compile due to Numba callbacks.

**Solution**: Use `jax.experimental.io_callback` to make Numba traceable.

**Implementation**: See comparison doc lines 96-135 for complete example.

**Key code**:
```python
from jax.experimental import io_callback

@jax.jit  # NOW COMPILES!
def rk4_step_compiled(x, dt):
    def get_velocities(positions):
        elem_ids = io_callback(
            numba_search_cpu,
            jax.ShapeDtypeStruct(positions.shape[0], jnp.int32),
            positions,
            ordered=False
        )
        return interpolate_jax(positions, elem_ids, ...)
    
    k1 = get_velocities(x)
    k2 = get_velocities(x + dt/2 * k1)
    k3 = get_velocities(x + dt/2 * k2)
    k4 = get_velocities(x + dt * k3)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

**Expected**: 495 ms → ~100 ms integration overhead

### Phase 1 Result

```
Component               Before    After     Speedup
────────────────────────────────────────────────────
CPU Search              120 ms    15-25 ms  5-8×
Integration Overhead    495 ms    ~100 ms   5×
────────────────────────────────────────────────────
TOTAL                   695 ms    100-150ms 5-7×
```

---

## Phase 2: Memory Optimization (2-3 weeks) → 9-14× Cumulative

**Goal**: Reduce octree memory and improve traversal  
**Risk**: Low | **Memory**: -67% octree size

### Morton Code Background

**Morton code** (Z-order curve) maps 3D → 1D while preserving spatial locality.

**Benefits**:
- Implicit hierarchy: parent = `morton >> 3`, child = `(morton << 3) | octant`
- Spatial locality: sequential codes = nearby in 3D
- Memory: **12B vs 56B per node** (4.7× savings)

**Current storage**:
```
center (12B) + half_size (12B) + children (32B) = 56B/node
```

**Morton storage**:
```
morton_code (8B) + child_offset (4B) = 12B/node
```

### Implementation

See comparison doc lines 139-196 for complete Morton utilities:
- `compute_morton_code()` - encode 3D point to Morton
- `decode_morton_code()` - recover center/bounds from Morton
- `morton_parent()`, `morton_child()` - implicit hierarchy

### Morton Octree Builder

See comparison doc lines 199-269 for complete builder implementation.

**Algorithm**:
1. Compute Morton code for each element
2. Sort elements by Morton (spatial locality!)
3. Build octree top-down on sorted ranges
4. Store in breadth-first order

### Phase 2 Result

```
Component               Phase 1    Phase 2    Speedup
─────────────────────────────────────────────────────
Octree Memory           1.05 MB    0.30 MB    3.5×
CPU Search (cached)     15-25 ms   10-15 ms   1.5-2×
Traversal (locality)    -          -          2-3×
─────────────────────────────────────────────────────
TOTAL                   100-150ms  50-80ms    2-3×

Cumulative from baseline: 695 ms → 50-80 ms (9-14×)
```

---

## Phase 3: GPU-Native Search (2-3 weeks) → 70-140× Cumulative

**Goal**: Move search to GPU for full pipeline compilation  
**Risk**: Medium | **Memory**: +50% runtime

### Architecture

```
FULLY COMPILED JAX (no CPU callbacks!):

@jax.jit
def track_particles(positions, times):
    def rk4_step(x, t):
        k1 = search_and_interpolate_gpu(x)  # ALL ON GPU
        k2 = search_and_interpolate_gpu(x + dt/2 * k1)
        k3 = search_and_interpolate_gpu(x + dt/2 * k2)
        k4 = search_and_interpolate_gpu(x + dt * k3)
        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return jax.lax.scan(rk4_step, positions, times)
```

### Task 1: Hash-Based Fine Octree (1-2 weeks)

**Problem**: O(log n) tree traversal for fine octree.

**Solution**: O(1) hash table lookup.

**Implementation**: See comparison doc lines 272-310 for complete hash octree.

**Key concepts**:
- Prime hash size (~2× nodes)
- Open addressing with linear probing
- Max probe length: 10

```python
@jax.jit
def hash_lookup_gpu(morton_query, hash_table, morton_codes):
    h = morton_query % hash_size
    for probe in range(10):  # Fixed iterations for JAX
        idx = hash_table[(h + probe) % hash_size]
        if idx == -1: return NOT_FOUND
        if morton_codes[idx] == morton_query: return idx
    return NOT_FOUND
```

### Task 2: Flatten Element Lists (1-2 weeks)

**Problem**: Variable-length element lists prevent GPU compilation.

**Solution**: Pre-flatten to fixed padded arrays with offsets.

**Implementation**: See comparison doc lines 199-251 for flattened octree structure.

```python
@dataclass
class FlattenedOctree:
    morton_codes: jnp.ndarray        # (N,) uint64
    child_offsets: jnp.ndarray       # (N,) int32
    element_offsets: jnp.ndarray     # (N,) int32
    element_counts: jnp.ndarray      # (N,) int32
    elements_flat: jnp.ndarray       # (total_elems,) int32 - FLATTENED

@jax.jit
def search_octree_gpu(point, octree):
    # ... stackless traversal ...
    elem_start = octree.element_offsets[leaf_idx]
    elem_count = octree.element_counts[leaf_idx]
    elem_slice = jax.lax.dynamic_slice(
        octree.elements_flat, (elem_start,), (elem_count,)
    )
    # ... vectorized element testing ...
```

### Phase 3 Result

```
Component               Phase 2    Phase 3    Speedup
─────────────────────────────────────────────────────
Search (GPU hash)       10-15 ms   1-2 ms     5-10×
Interpolation (GPU)     70-80 ms   2-3 ms     25-35×
Integration (compiled)  ~100 ms    2-5 ms     20-50×
─────────────────────────────────────────────────────
TOTAL                   50-80 ms   5-10 ms    5-10×

Cumulative from baseline: 695 ms → 5-10 ms (70-140×)
```

**Memory impact**:
```
Phase 2:                    Phase 3:
  Octrees (CPU): 0.3 MB      Octrees (GPU): 1.05 MB (moved to GPU)
  Mesh (GPU): 184 MB         Mesh (GPU): 184 MB
  Cache: 368 MB              Hash table: +5 MB
  Total: 1.24 GB             Runtime buffers: +15-25 MB
                             Total: 1.8-2.5 GB (+50%)
```

---

## Phase 4: Full Rewrite (6-8 weeks, optional)

**Only needed for >100K particles**

**Architecture**: Forest of octrees (partition domain into tiles, 1 octree/tile)

**When needed**:
```
<1K particles:     Phase 3 sufficient (5-10 ms/step)
1K-10K:            Phase 3 sufficient (10-50 ms/step)
10K-100K:          Phase 4 beneficial
>100K:             Phase 4 required (multi-GPU)
```

**Recommendation**: Defer until scaling requirements change.

---

## Testing Strategy

### Phase 1 Tests
```python
tests/test_element_cache.py:
  - test_cache_hit()
  - test_cache_invalidation()
  - test_hit_rate()

tests/test_jax_io_callback.py:
  - test_io_callback_compilation()
  - test_rk4_compiled()
```

### Phase 2 Tests
```python
tests/test_morton_utils.py:
  - test_morton_encoding_decoding()
  - test_morton_hierarchy()

tests/test_morton_octree_builder.py:
  - test_build_octree()
  - test_memory_usage()
```

### Phase 3 Tests
```python
tests/test_morton_hash_octree.py:
  - test_hash_build()
  - test_hash_lookup()

tests/test_octree_search_gpu.py:
  - test_end_to_end_gpu()
  - test_vs_cpu_correctness()
```

### Profiling
```bash
# Baseline
python -m cProfile -o baseline.prof test_reduced_particles.py

# JAX profiling
JAX_PROFILER_PORT=9999 python test_reduced_particles.py

# GPU profiling
nvprof python test_reduced_particles.py
nsys profile -o phase3.qdrep python test_reduced_particles.py
```

---

## Risk Assessment

### Phase 1: Very Low Risk
- io_callback overhead: mitigate with `ordered=False` tuning
- Cache thrashing: tune threshold based on dt
- JAX compatibility: test on JAX 0.4.23+

### Phase 2: Low Risk
- Morton precision: 21 bits/dimension sufficient
- Octree imbalance: validate depth distribution
- Implementation bugs: extensive unit tests

### Phase 3: Medium Risk
- Hash collisions: prime size, validate max_probe < 10
- JAX compile memory: use chunking (50-200 particles/batch)
- Numerical precision: tune barycentric tolerance
- Debugging: unit test each component separately

---

## Decision Matrix

| Factor | Weight | Phase 1 | Phase 2 | Phase 3 | Current |
|--------|--------|---------|---------|---------|---------|
| **Speed** | 40% | 7/10 | 9/10 | 10/10 | 3/10 |
| **Memory** | 20% | 10/10 | 10/10 | 6/10 | 9/10 |
| **Complexity** | 15% | 9/10 | 7/10 | 4/10 | 8/10 |
| **Stability** | 15% | 9/10 | 8/10 | 6/10 | 10/10 |
| **Maintainability** | 10% | 9/10 | 8/10 | 6/10 | 8/10 |
| **Weighted Total** | - | **8.4/10** | **8.6/10** | **7.6/10** | **6.5/10** |

### Recommendations

✅ **MUST DO**: Phase 1 (1 week, 5-7× speedup, very low risk)  
✅ **SHOULD DO**: Phase 2 (2-3 weeks, 9-14× cumulative, low risk)  
✅ **SHOULD DO**: Phase 3 (2-3 weeks, 70-140× cumulative, medium risk, if >1K particles)  
⚠️ **CONSIDER**: Phase 4 (6-8 weeks, high risk, only if >100K particles)

---

## Implementation Roadmap

### Week 1: Phase 1
- Days 1-2: Element caching + integration + tests
- Days 3-5: JAX io_callback + integration + tests
- Days 6-7: Profiling and validation

**Decision point**: If speedup >= 5×, proceed to Phase 2

### Weeks 2-4: Phase 2
- Week 2: Morton utilities + tests
- Week 3: Morton octree builder + integration
- Week 4: Validation and profiling

**Decision point**: If memory reduced and speedup >= 9×, proceed to Phase 3

### Weeks 5-7: Phase 3
- Week 5: Hash-based fine octree
- Week 6: GPU search implementation
- Week 7: Integration and optimization

**Decision point**: If speedup >= 70×, production-ready for 5K+ particles

---

## References

Based on research from:

1. Wang, Z., et al. (2024) - High-Performance AMR on GPUs
2. Karras, T., & Aila, T. (2023) - Fast Parallel BVH Construction
3. Madeira, D., et al. (2009) - Hash-Based Spatial Data Structures
4. Yerry & Shephard (1984) - Automatic 3D Mesh Generation
5. Morton/Z-order curve literature (Sietstra 2019, Wikipedia)

See `Comparison_Current_JAXTrace_Implementation_vs_Optimized_GPU_Approaches.md` for complete citations and detailed analysis.

---

## Quick Start

```bash
# 1. Backup
git checkout -b optimization-phase1
git commit -am "Checkpoint before Phase 1"

# 2. Baseline benchmark
python test_reduced_particles.py --profile baseline

# 3. Implement Phase 1 (see comparison doc for complete code)
cp jaxtrace/fields/element_cache.py.template jaxtrace/fields/element_cache.py
# Edit shared_octree_fem_field.py per comparison doc lines 96-135

# 4. Test
pytest tests/test_element_cache.py
pytest tests/test_jax_io_callback.py

# 5. Profile
python test_reduced_particles.py --profile phase1

# 6. Validate 5-7× speedup, then proceed to Phase 2
```

---

**END OF ROADMAP v2.0**
