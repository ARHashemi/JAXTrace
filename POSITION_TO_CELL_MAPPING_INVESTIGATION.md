# Position-to-Cell Mapping Investigation

## Problem Statement

All mesh-aligned search methods show **~10% lower retention** compared to baseline Morton during RK4 tracking:

```
From logs/benchmark_l2_search_methods_with-mesh-aligned-multi-cell-local.log:
- Baseline Morton radius=10:                92.07% retention
- Mesh-Aligned Single-Cell (direct):        80.23% retention
- Mesh-Aligned Multi-Cell + 2×2×2 Local:    80.23% retention
- Mesh-Aligned Morton hybrid:               ~82% retention
- Mesh-Aligned Neighbors (Option B):        ~81% retention
```

**Key observation**: The retention loss is **consistent across all mesh-aligned methods**, suggesting a systematic issue with the position-to-cell mapping algorithm, NOT with neighbor search strategies.

---

## Position-to-Cell Mapping Algorithm Analysis

### Current Implementation

The position-to-cell mapping uses **Morton encoding with direct grid index calculation**:

**File**: [jaxtrace/gpu/search/mesh_aligned_point_location.py:228-263](jaxtrace/gpu/search/mesh_aligned_point_location.py#L228-L263)

```python
# Step 1: Get cell size for this refinement level
level = 14 - level_idx  # Maps level_idx 0-7 → levels 14-7
cell_size = octree_gpu.level_cell_sizes[level]  # (3,) float32

# Step 2: Compute grid indices via floor division
i_base = jnp.floor(pos[0] / cell_size[0]).astype(jnp.int32)
j_base = jnp.floor(pos[1] / cell_size[1]).astype(jnp.int32)
k_base = jnp.floor(pos[2] / cell_size[2]).astype(jnp.int32)

# Step 3: Add neighbor offset (e.g., [-1, -1, -1])
i = i_base + di
j = j_base + dj
k = k_base + dk

# Step 4: Apply Morton offset and clipping
i_offset = jnp.clip(i + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)
j_offset = jnp.clip(j + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)
k_offset = jnp.clip(k + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)

# Step 5: Encode to Morton code
morton_code = encode_morton_3d_jax(i_offset, j_offset, k_offset)

# Step 6: Binary search for (morton, level) pair
cell_idx = find_cell_by_morton_and_level(
    morton_code, level,
    octree_gpu.cell_morton_codes,
    octree_gpu.cell_levels
)
# Returns: cell_idx ≥ 0 if found, -1 if not found
```

### Binary Search Implementation

**File**: [jaxtrace/gpu/search/mesh_aligned_octree_gpu.py:211-279](jaxtrace/gpu/search/mesh_aligned_octree_gpu.py#L211-L279)

```python
def find_cell_by_morton_and_level(
    morton_code: jnp.uint64,
    level: jnp.uint8,
    cell_morton_codes: jax.Array,
    cell_levels: jax.Array,
) -> jnp.int32:
    """
    Find cell index by (Morton code, level) pair using binary search.

    CRITICAL: Cells are sorted by (morton, level) tuples. We must find
    a cell where BOTH morton AND level match.

    Returns: Cell index, or -1 if not found
    """
    max_iters = 25  # log2(665k) ≈ 20, use 25 for safety

    def search_step(i, carry):
        left, right, found_idx = carry
        is_active = left < right

        mid = (left + right) // 2
        mid_morton = jnp.where(is_active, cell_morton_codes[mid], jnp.uint64(0))
        mid_level = jnp.where(is_active, cell_levels[mid], jnp.uint8(0))

        # Lexicographic comparison of (morton, level) tuples
        mid_less = jnp.logical_or(
            mid_morton < morton_code,
            jnp.logical_and(mid_morton == morton_code, mid_level < level)
        )
        mid_greater = jnp.logical_or(
            mid_morton > morton_code,
            jnp.logical_and(mid_morton == morton_code, mid_level > level)
        )

        # Update bounds
        new_left = jnp.where(jnp.logical_and(is_active, mid_less), mid + 1, left)
        new_right = jnp.where(jnp.logical_and(is_active, mid_greater), mid, right)

        # Check if found (both morton AND level match)
        is_found = jnp.logical_and(
            jnp.logical_and(is_active, mid_morton == morton_code),
            mid_level == level
        )
        new_found_idx = jnp.where(is_found, mid, found_idx)

        return (new_left, new_right, new_found_idx)

    n_cells = cell_morton_codes.shape[0]
    init_state = (jnp.int32(0), jnp.int32(n_cells), jnp.int32(-1))
    final_state = lax.fori_loop(0, max_iters, search_step, init_state)

    _, _, found_idx = final_state
    return found_idx  # -1 if not found
```

---

## Potential Issues to Investigate

### 1. **Grid Origin Mismatch**

**Hypothesis**: The grid used during octree extraction (CPU) may have a different origin than the grid used during search (GPU).

**Evidence needed**:
- What is `octree_gpu.morton_offset`? (Expected: 2^19 = 524,288)
- Are cell grid indices stored during extraction?
- Do stored grid indices match computed grid indices during search?

**Test**: For a known particle position that fails to find its element:
```python
# During extraction (CPU)
i_extract = int(np.floor(vertex[0] / cell_size[0]))
i_morton_extract = np.clip(i_extract + offset, 0, max_coord - 1)

# During search (GPU)
i_search = jnp.floor(pos[0] / cell_size[0]).astype(jnp.int32)
i_morton_search = jnp.clip(i_search + morton_offset, 0, morton_max_coord - 1)

# Compare: Should be identical for same position
assert i_morton_extract == i_morton_search
```

**File to check**: [jaxtrace/gpu/search/mesh_aligned_octree_vertex_multi.py:138-150](jaxtrace/gpu/search/mesh_aligned_octree_vertex_multi.py#L138-L150)

```python
# During extraction
for vertex in vertices:
    # Compute grid indices for this vertex
    i = int(np.floor(vertex[0] / cell_size[0]))
    j = int(np.floor(vertex[1] / cell_size[1]))
    k = int(np.floor(vertex[2] / cell_size[2]))

    # Encode to Morton (with offset for negative coordinates)
    offset = (1 << 19)  # 2^19 = 524,288
    max_coord = (1 << 20)  # 2^20 = 1,048,576

    i_morton = np.clip(i + offset, 0, max_coord - 1)
    j_morton = np.clip(j + offset, 0, max_coord - 1)
    k_morton = np.clip(k + offset, 0, max_coord - 1)

    morton = encode_morton_3d_single(i_morton, j_morton, k_morton, max_depth=21)
```

**Comparison needed**: Does `octree_gpu.morton_offset` match `offset` used during extraction?

---

### 2. **Cell Size Precision Mismatch**

**Hypothesis**: Cell sizes might have floating-point precision issues when stored and retrieved.

**Evidence needed**:
- Are cell sizes stored with sufficient precision? (float32 vs float64)
- Do `octree_gpu.level_cell_sizes[level]` match the cell sizes used during extraction?

**Test**: For each refinement level:
```python
# During extraction
cell_size_extract = find_axis_aligned_edges_single(vertices, tolerance=1e-6)

# During search
cell_size_search = octree_gpu.level_cell_sizes[level]

# Compare
rel_error = np.abs(cell_size_extract - cell_size_search) / cell_size_extract
print(f"Level {level}: relative error = {rel_error}")
# Should be < 1e-6 for float32
```

**File to check**: [jaxtrace/gpu/search/mesh_aligned_octree_gpu.py:45-78](jaxtrace/gpu/search/mesh_aligned_octree_gpu.py#L45-L78) (upload function)

---

### 3. **Level Mapping Inconsistency**

**Hypothesis**: The mapping between `level_idx` (0-7) and actual `level` (14-7) might be inconsistent.

**Current mapping**:
```python
level = 14 - level_idx  # level_idx ∈ [0,7] → level ∈ [14,7]
```

**Evidence needed**:
- During extraction, what level values are assigned to cells?
- Are cells sorted by `(morton, level)` correctly?
- Does the binary search handle level comparison correctly?

**Test**: Print level distribution:
```python
# From octree structure
unique_levels = np.unique(octree_cells.cell_levels)
print(f"Levels in octree: {unique_levels}")

# Expected: [7, 8, 9, 10, 11, 12, 13, 14]
# If different, level mapping is wrong
```

---

### 4. **Morton Encoding Consistency**

**Hypothesis**: The Morton encoding might differ between CPU (extraction) and GPU (search).

**Files to compare**:
- CPU: [jaxtrace/gpu/search/mesh_aligned_octree_single_cell.py](jaxtrace/gpu/search/mesh_aligned_octree_single_cell.py) - `encode_morton_3d_single()`
- GPU: [jaxtrace/gpu/search/mesh_aligned_octree_gpu.py](jaxtrace/gpu/search/mesh_aligned_octree_gpu.py) - `encode_morton_3d_jax()`

**Test**: For same grid indices `(i, j, k)`:
```python
morton_cpu = encode_morton_3d_single(i, j, k, max_depth=21)
morton_gpu = encode_morton_3d_jax(i, j, k)

assert morton_cpu == morton_gpu
```

---

### 5. **Binary Search Correctness**

**Hypothesis**: The binary search might fail to find cells that exist due to:
- Insufficient iterations (max_iters=25 might be too few)
- Incorrect lexicographic comparison
- Early termination issues

**Test**: Verify binary search on known cells:
```python
# Pick a random cell from octree
cell_idx_known = 12345
morton_known = octree_gpu.cell_morton_codes[cell_idx_known]
level_known = octree_gpu.cell_levels[cell_idx_known]

# Search for it
cell_idx_found = find_cell_by_morton_and_level(
    morton_known, level_known,
    octree_gpu.cell_morton_codes,
    octree_gpu.cell_levels
)

assert cell_idx_found == cell_idx_known, f"Failed to find known cell: expected {cell_idx_known}, got {cell_idx_found}"
```

---

### 6. **Particle Position Drift During RK4**

**Hypothesis**: RK4 integration might push particles outside the octree domain due to:
- Velocity field discontinuities at mesh boundaries
- Incorrect boundary handling
- Cell boundaries not aligned with actual element boundaries

**Evidence from benchmarks**:
- Initial assignment: 100% searchability (all methods)
- After 100 RK4 steps: 80% retention (mesh-aligned) vs 92% (Morton)

**Difference**: Morton baseline searches a **radius of 10 leaves**, testing ~536 elements. This provides a "safety margin" that catches particles that drift slightly outside their expected cells.

Mesh-aligned methods search only **local cells** (~5.9-18.31 elements), so if a particle drifts outside the expected cell due to grid misalignment, it's immediately lost.

**Test**: Track particle positions that fail to find elements:
```python
# During RK4
if elem_id < 0:  # Not found
    print(f"Lost particle at step {step}:")
    print(f"  Position: {pos}")
    print(f"  Cell indices: ({i_base}, {j_base}, {k_base})")
    print(f"  Morton code: {morton_code}")
    print(f"  Level: {level}")
    print(f"  Binary search result: {cell_idx}")
```

---

## Investigation Strategy

### Phase 1: Verify Grid Consistency (CRITICAL)

1. **Extract a single test element** with known vertex positions
2. **Compute grid indices** during extraction and during search
3. **Compare**:
   - Grid indices `(i, j, k)`
   - Morton offsets
   - Morton codes
   - Cell lookup results

**Diagnostic script**: `diagnose_position_to_cell_mapping.py`

```python
#!/usr/bin/env python3
"""
Diagnose position-to-cell mapping for a single test element.
"""
import numpy as np
import jax.numpy as jnp

# Select test element
elem_id = 1234567
vertices = node_positions[connectivity[elem_id]]

# CPU: Extraction
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import find_axis_aligned_edges_single, encode_morton_3d_single
cell_size, level = find_axis_aligned_edges_single(vertices, tolerance=1e-6)

print(f"Element {elem_id}:")
print(f"  Vertices: {vertices}")
print(f"  Cell size: {cell_size}")
print(f"  Level: {level}")

offset = (1 << 19)
max_coord = (1 << 20)

for v_idx, vertex in enumerate(vertices):
    i_cpu = int(np.floor(vertex[0] / cell_size[0]))
    j_cpu = int(np.floor(vertex[1] / cell_size[1]))
    k_cpu = int(np.floor(vertex[2] / cell_size[2]))

    i_morton = np.clip(i_cpu + offset, 0, max_coord - 1)
    j_morton = np.clip(j_cpu + offset, 0, max_coord - 1)
    k_morton = np.clip(k_cpu + offset, 0, max_coord - 1)

    morton_cpu = encode_morton_3d_single(i_morton, j_morton, k_morton, max_depth=21)

    print(f"\n  Vertex {v_idx}: {vertex}")
    print(f"    Grid indices (CPU): ({i_cpu}, {j_cpu}, {k_cpu})")
    print(f"    Morton offset: ({i_morton}, {j_morton}, {k_morton})")
    print(f"    Morton code: {morton_cpu}")

    # GPU: Search
    from jaxtrace.gpu.search.mesh_aligned_octree_gpu import encode_morton_3d_jax
    vertex_gpu = jnp.array(vertex)
    cell_size_gpu = jnp.array(cell_size)

    i_gpu = jnp.floor(vertex_gpu[0] / cell_size_gpu[0]).astype(jnp.int32)
    j_gpu = jnp.floor(vertex_gpu[1] / cell_size_gpu[1]).astype(jnp.int32)
    k_gpu = jnp.floor(vertex_gpu[2] / cell_size_gpu[2]).astype(jnp.int32)

    i_morton_gpu = jnp.clip(i_gpu + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)
    j_morton_gpu = jnp.clip(j_gpu + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)
    k_morton_gpu = jnp.clip(k_gpu + octree_gpu.morton_offset, 0, octree_gpu.morton_max_coord - 1)

    morton_gpu = encode_morton_3d_jax(i_morton_gpu, j_morton_gpu, k_morton_gpu)

    print(f"    Grid indices (GPU): ({i_gpu}, {j_gpu}, {k_gpu})")
    print(f"    Morton offset (GPU): ({i_morton_gpu}, {j_morton_gpu}, {k_morton_gpu})")
    print(f"    Morton code (GPU): {morton_gpu}")

    # Compare
    match = (i_cpu == i_gpu) and (j_cpu == j_gpu) and (k_cpu == k_gpu) and (morton_cpu == morton_gpu)
    print(f"    ✅ MATCH" if match else f"    ❌ MISMATCH")

    # Binary search
    from jaxtrace.gpu.search.mesh_aligned_octree_gpu import find_cell_by_morton_and_level
    cell_idx = find_cell_by_morton_and_level(
        morton_gpu, jnp.uint8(level),
        octree_gpu.cell_morton_codes,
        octree_gpu.cell_levels
    )

    print(f"    Cell lookup: {cell_idx} ({'found' if cell_idx >= 0 else 'NOT FOUND'})")
```

### Phase 2: Cell Size Verification

```python
# Compare cell sizes across all levels
for level_idx in range(8):
    level = 14 - level_idx
    cell_size_gpu = octree_gpu.level_cell_sizes[level]

    # Find elements at this level
    elements_at_level = np.where(octree_cells.cell_levels == level)[0]
    if len(elements_at_level) > 0:
        cell_idx_sample = elements_at_level[0]
        cell_size_stored = octree_cells.cell_sizes[cell_idx_sample]

        rel_error = np.abs(cell_size_gpu - cell_size_stored) / cell_size_stored
        print(f"Level {level}: GPU={cell_size_gpu}, Stored={cell_size_stored}, Error={rel_error}")
```

### Phase 3: Binary Search Validation

```python
# Test binary search on ALL cells
n_cells = len(octree_gpu.cell_morton_codes)
n_failures = 0

for cell_idx in range(n_cells):
    morton = octree_gpu.cell_morton_codes[cell_idx]
    level = octree_gpu.cell_levels[cell_idx]

    found_idx = find_cell_by_morton_and_level(
        morton, level,
        octree_gpu.cell_morton_codes,
        octree_gpu.cell_levels
    )

    if found_idx != cell_idx:
        n_failures += 1
        print(f"Failed to find cell {cell_idx}: morton={morton}, level={level}, found={found_idx}")

print(f"\nBinary search validation: {n_failures}/{n_cells} failures ({100*n_failures/n_cells:.2f}%)")
```

---

## Expected Outcomes

If the investigation reveals:

1. **Grid origin mismatch** → Fix `morton_offset` consistency
2. **Cell size precision** → Use float64 or store exact rational cell sizes
3. **Level mapping error** → Fix level indexing
4. **Morton encoding difference** → Unify CPU/GPU encoding
5. **Binary search bug** → Fix comparison logic or increase iterations
6. **Particle drift** → Add boundary clamping or expand search radius

**Target**: Achieve **92%+ retention** for mesh-aligned methods, matching baseline Morton.

---

## Next Steps

1. ✅ **Commit current implementation** (done)
2. ✅ **Update production script** to use multi-cell + 2×2×2 local search (done)
3. **Create diagnostic script** `diagnose_position_to_cell_mapping.py`
4. **Run diagnostics** on failed particles from benchmark logs
5. **Identify root cause** of ~10% retention loss
6. **Implement fix** based on findings
7. **Re-run benchmarks** to verify fix

---

## Reference Files

- Position-to-cell search: [mesh_aligned_point_location.py](jaxtrace/gpu/search/mesh_aligned_point_location.py)
- Binary search: [mesh_aligned_octree_gpu.py](jaxtrace/gpu/search/mesh_aligned_octree_gpu.py)
- CPU extraction: [mesh_aligned_octree_vertex_multi.py](jaxtrace/gpu/search/mesh_aligned_octree_vertex_multi.py)
- Benchmark results: [logs/benchmark_l2_search_methods_with-mesh-aligned-multi-cell-local.log](logs/benchmark_l2_search_methods_with-mesh-aligned-multi-cell-local.log)
