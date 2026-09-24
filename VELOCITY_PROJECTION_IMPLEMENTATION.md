# Field Projection onto a Union-Octree Reference Mesh: JAXTrace Implementation Guide

> Revised 2026-05-07 to reflect the LUMI 18232497 envelope-scan results
> and the user's design constraints
> (see `TIME_DEPENDENT_MESH_FEASIBILITY.md` §15).

This document specifies how to implement the **field projection
preprocessor** for JAXTrace: convert a periodic, r-adaptive source
mesh sequence (150 distinct meshes per cycle, ~8.4 M tetrahedra each,
FEMUSS output) into a **fixed union-octree reference mesh** plus a
**time-dependent stack of every field present in the source PVTUs**.
The runtime tracker then runs unchanged against the destination mesh,
indexing the cycle stack by `t mod N_seq`.

**Design constraints (from user, 2026-05-07):**

1. The destination mesh must have **the same multi-level refinement as
   the source**, not a 2-level fine/coarse approximation. Wherever any
   cycle step subdivides a parent cube into octants, the destination
   must too. ⇒ **Union-octree construction** (Algorithm U-A in
   `TIME_DEPENDENT_MESH_FEASIBILITY.md` §15.5).
2. Every variable in the source PVTU is mapped 1:1 by name to the
   destination. No transformations during preprocessing
   (Displacement→velocity, pin reconstruction, etc.).
3. Pin-velocity reconstruction is a **runtime opt-in flag**, not a
   preprocessing step. Default off.
4. Output format: **VTKHDF** (single-file, static-mesh time series) as
   the canonical archive, plus a JAXTrace-native NPZ bundle for
   runtime GPU loading. Both written from the same in-memory result.

---

## 1. Inputs and Outputs

### 1.1 Inputs

- A directory of source PVTU files following pattern
  `{stem}_{k}.pvtu` for `k = 0, 1, ..., N_seq - 1`
- Cycle period `N_seq` (auto-detected from the file list, or supplied)
- Field-projection list (default: **all** point-data fields in the
  source PVTU; specify via `--fields` to subset).
  Verified field set for the FEMUSS C3 dataset (LUMI 18232497):

  | Field | Components | Type |
  |---|---:|:---|
  | `Displacement` | 3 | double |
  | `Pressure`     | 1 | double |
  | `Reactions`    | 3 | double |
  | `Temperature`  | 1 | double |
  | `LEVEL`        | 1 | double |

  Cell-data (`Stress` 6-comp, `Strain` 6-comp): not handled by this
  preprocessor (different transfer operator needed for cell-located
  data; out of scope).

### 1.2 Outputs (two formats, both written from the same run)

**Format A — VTKHDF static-mesh archive** (canonical / interop).
A single `.vtkhdf` file holding the destination mesh once and one
time-step group per cycle index, each containing all projected
point-data fields. Paraview reads it natively (recent versions);
HDF5-backed for parallel I/O.
[Kitware 2025 status update](https://www.kitware.com/vtkhdf-file-format-2025-status-update/)
documents that VTKHDF "added static mesh support in both the reader
and the writer, which allows caching geometry between time steps when
scalar fields are changing but the unstructured mesh geometry is
static" — exactly our case.

**Format B — JAXTrace native NPZ bundle** (runtime / fastest GPU upload).
- `ref_mesh.npz` — `node_positions (N_dest, 3)`, `connectivity
  (N_tet_dest, 4)`. Octree, AA metadata, inverse matrices are
  **recomputed at runtime** from positions + connectivity (this is what
  the existing `benchmark_femuss_comparison.py` already does for the
  fixed-mesh case; no need to cache derived structures).
- `field_<name>.npz` — one file per projected field, e.g.
  `field_Displacement.npz` containing `(N_seq, N_dest, 3)` float32.
  One file per field keeps each dataset independently mmap-able and
  partial-update friendly.
- `manifest.json` — field list, dtypes, shapes, cycle length, source
  pattern, preprocessor version.

**Optional Format C — per-step VTU + .pvd collection**, behind
`--write-vtu-collection` flag. Compatible with older Paraview versions
that lack the VTKHDF reader. Disk size penalty: 150× mesh duplication
(~22 GB). Off by default.

---

---

## 2. Established Facts (from LUMI runs)

These are the **load-bearing measurements** that the design relies on.

| Fact | Value | Source |
|------|------:|:------|
| Cycle period | 150 (step 150 ≈ step 0) | `analyze_mesh_18020158/mesh_variation_report.csv` |
| Bounding box | `[6.000e-2, 3.000e-2, 6.000e-3]` m, identical across C3/C4/A2 | All three diff reports |
| Edge length range | `[4.687e-5, 5.196e-3]` m, **invariant to 7 sig figs** across all 152 timesteps | C3 CSV |
| Per-level element-count drift across cycle | ≤13% at coarse, **0.08% at finest level** | C3 CSV refinement_dist column |
| Source-mesh element count | ~8.44 M (C3, C4) / ~5.97 M (A2) | All three diffs |
| Source-mesh node count | ~2.20 M (C3, C4) / ~1.66 M (A2) | All three diffs |
| Refinement levels | 7 (coarse=0, finest=6) | Edge ratio 110.85× ≈ 2^6.79 |
| Dynamic-element fraction (step 0 vs 50) | 98–99.99% across cases | Diffs |

Additional facts from LUMI run 18232497 (C3, 16 envelope samples spanning
the full cycle):

| Fact | Value | Source |
|------|------:|:------|
| Fine-zone envelope (20-pct edge threshold) | `[-7.227e-3, -7.148e-3, -5.988e-3]` to `[+7.117e-3, +7.148e-3, -1.172e-5]` | `fine_zone_envelope.json` |
| Drift in fine envelope across cycle | **0** (identical bytes at all 16 samples) | Same |
| Fine-envelope volume fraction | **11.35%** of full bbox | Same |
| Fine-element count drift | **0.19%** (1,703,962 to 1,707,219) | Same |
| Edge length range (envelope sample) | `[5.66e-5, 3.99e-3]` m, ratio 70.5× | Same |
| Octave levels of refinement | **7** (level 0 to level 6) | Same |

Key consequences:

1. The **refinement budget** (per-level cell count) is fixed across the
   cycle to within 0.2% at the finest level; only the spatial location
   of each refined element moves.
2. The **fine-zone envelope is spatially fixed** to the byte — the
   refinement template is invariant. This guarantees the union octree
   construction is well-defined and bounded.
3. The destination mesh must therefore be the union of all source
   octrees' (level, cell) pairs across the cycle (§3) — *not* a
   two-zone uniform-fine + uniform-coarse mesh, which would discard
   the multi-level structure.

---

## 3. Destination Mesh Design (Union-Octree)

### 3.1 Why a union octree, not a two-zone mesh

The two-zone strategy of the prior draft (uniform fine + uniform coarse)
**discards the source's multi-level refinement** and imposes a single
edge length within each zone. With 7 octave levels in the source mesh,
this would either waste 8× memory (if uniform-fine throughout) or lose
detail (if uniform-coarse anywhere).

The correct design — and the one the user's intuition demanded — is to
build the destination as the **forest-of-octrees union** of all 150
source meshes' parent-cube octrees:

> For every (level ℓ, parent-cube position c) appearing as a leaf in any
> source step's octree, the destination octree has a leaf at exactly
> (ℓ, c). Each destination leaf is then split into 6 Kuhn tetrahedra.

The resulting mesh has **at least the resolution of any source step at
every spatial location**. There is no information loss anywhere across
the cycle. Concretely:

- Inside the fixed fine envelope (the swept volume of the tool), the
  destination has the union of every source's level-6 refinement —
  i.e., maximally refined.
- In the transition band, the destination preserves the natural
  level-graded structure (levels 1–4) that source meshes already have.
- Far from the tool, the destination has the same coarse cells as the
  source.

Because the source's per-level element count drifts by only 0.08–13%
across the cycle (`TIME_DEPENDENT_MESH_FEASIBILITY.md` §15.2), the union
octree is at most ~1.2× the size of any single source octree. Total
destination tet count: **~10–12 M** (vs ~8.4 M source) — comfortable.

### 3.2 Algorithm U-A (recommended)

```
build_union_octree(source_pvtu_pattern, N_seq):
    # Phase 1: collect octree leaves from every source step
    union_leaves = empty set                         # set of (level, cell_idx)
    for k in 0 ... N_seq - 1:
        src_pos, src_conn, _ = load_mesh_from_pvtu(<source pvtu k>)
        src_pos, src_conn = deduplicate_nodes(...)
        src_octree = extract_octree_cells_parent_cube(src_pos, src_conn, ...)
        for cell in src_octree.cells:
            union_leaves.add((cell.level, cell.cell_idx))
        free src_octree, src_pos, src_conn

    # Phase 2: enforce 2:1 balance (optional — required only if MALMO
    # search assumes balanced trees; current MALMO does not)
    union_leaves = balance_octree_leaves(union_leaves)

    # Phase 3: Kuhn tetrahedralisation of each leaf cube
    dest_node_positions = []
    dest_connectivity   = []
    node_dedup_map      = {}
    for (lvl, cell_idx) in union_leaves:
        cube_corners = compute_cube_corners(lvl, cell_idx, bbox)
        for tet in kuhn_split(cube_corners):
            for vertex in tet:
                if vertex not in node_dedup_map:
                    node_dedup_map[vertex] = len(dest_node_positions)
                    dest_node_positions.append(vertex)
            dest_connectivity.append(
                [node_dedup_map[v] for v in tet]
            )

    return dest_node_positions, dest_connectivity
```

**Phase 1** reuses the existing `extract_octree_cells_parent_cube`
without modification. We discard the per-cell tetrahedra after
collection — only the cube-level (level, idx) pairs are kept.

**Phase 2** (balancing) is optional. The 2:1 balance condition
("neighbouring leaves differ by at most one level") is required by some
AMR search algorithms but **not** by the current MALMO search kernels —
which traverse cells via Morton codes, not pointer-chasing. Skip
unless we observe correctness issues.

**Phase 3** generates positions and connectivity. The Kuhn
tetrahedralisation of a cube into 6 tets is a standard construction:
the 8 corners are labelled by their binary coordinate; the 6 tets are
defined by the 6 monotone paths from `000` to `111` along the cube
edges. JAXTrace already has Kuhn helpers (used implicitly by
`extract_octree_cells_parent_cube`), but the union-construction needs a
fresh pass that emits tets at the union leaf level, not at source
leaves.

The deduplication map ensures shared faces between adjacent cubes use
the same node IDs. Boundary nodes between cubes at *different* levels
require a hanging-node treatment: the simpler cube must be subdivided
along its face that touches the more-refined cube. This is where 2:1
balance helps — with balance, only one type of subdivision is needed.

### 3.3 Memory cost during construction

**Phase 1 dominates wall time** because each source PVTU load is ~50 s
(I/O bound on LUMI scratch) and each octree build is ~150 s. For
N_seq = 150: ~5 hours, embarrassingly parallel across cycle indices
(each k is independent until the union-merge step at the end).

**Memory** during Phase 1: peak ~2 GB (one source mesh + its octree at
a time; each is freed before the next). The accumulated `union_leaves`
set is ~1.2× source_leaf_count × ~24 B ≈ 30 MB — negligible.

### 3.4 Bootstrapping with a smaller union

For initial validation we don't need all 150 cycle indices in the
union. Run Phase 1 with stride 10 (16 indices) — this captures the
spatial extent of the refinement template (which §15.2 confirmed is
fixed across the cycle), missing only the per-step micro-fluctuations.
**Fast bootstrap = ~30 minutes**, then gives a usable destination mesh
to validate the rest of the pipeline. Run the full 150 union later
when ready.

### 3.5 Validation

Before running the full projection over 150 indices, verify the
destination mesh:

- Build its MALMO octree (reusing existing
  `extract_octree_cells_parent_cube` →
  `upload_mesh_aligned_octree_to_gpu`)
- For each sampled cycle index k: project source field 0 onto the
  destination, then back-project to source — error should be at
  P1-interpolation level, not larger
- Spot-check that the destination contains all source leaves: for a
  random sample of source elements, locate their centroid in the
  destination and confirm host element exists
- Confirm Kuhn-tet quality: aspect ratio, volume positivity

---

## 4. Projection Pipeline

### 4.0 No field transformations during preprocessing

**Per user constraint 2 (2026-05-07):** the projection step copies every
field by name with **P1 interpolation only**. Specifically:

- `Displacement` (source) → `Displacement` (destination)
- `Pressure` → `Pressure`
- `Temperature` → `Temperature`
- `LEVEL` → `LEVEL`
- `Reactions` → `Reactions`

No frame differencing, no pin-velocity reconstruction, no derived
quantities are computed in the preprocessor. The projected
`Displacement` field is the same cumulative-displacement stack the
existing tracker already consumes; the runtime tracker continues to
convert displacement → velocity via frame differencing, exactly as it
does today against source meshes.

**Pin-velocity reconstruction (user constraint 3):** the existing
`reconstruct_pin_velocity` function in
`benchmark_femuss_comparison.py:390` is preserved as a **runtime
opt-in flag** (`--reconstruct-pin`, default off). When enabled, it
runs at runtime on the *destination mesh's* projected `Displacement`
field, using the same pin geometry parameters as today. The
preprocessor itself never invokes it.

Justification for moving pin reconstruction to runtime:
- The pin region is defined by analytic geometry (cylinder + threads),
  not by mesh topology, so it operates correctly on either source or
  destination nodes.
- It runs on N_dest nodes per cycle step (cheaper than N_source × N_seq
  batched at preprocessing).
- It stays a clean opt-in toggle for users who don't model an embedded
  tool.

### 4.1 Overall structure

> **Important.** The current `load_velocity_sequence_from_pvtu`
> (`jaxtrace.gpu.mesh_loader_timedep`) assumes a *fixed* topology
> across the timestep range — it loads geometry from the first PVTU
> only and expects all subsequent files to share the same connectivity.
> For r-adaptive datasets like FEMUSS, this assumption is violated.
> The projection preprocessor must therefore **load each PVTU
> independently using `load_mesh_from_pvtu` (the single-file loader)**,
> recompute geometry per cycle index, and build a fresh octree at each
> step. This is the entire reason the projection step is expensive
> upfront — but it only runs once.

```
PREPROCESSING (offline, runs once per dataset)
─────────────────────────────────────────────
# Phase A — Union-octree destination mesh
union_leaves = empty set
for k in 0, 1, ..., N_seq - 1:                                       [§3.2]
    src_pos, src_conn, _ = load_mesh_from_pvtu(<source pvtu k>)
    src_pos, src_conn = deduplicate_nodes(src_pos, src_conn)
    src_octree_cells = extract_octree_cells_parent_cube(src_pos, src_conn, ...)
    union_leaves |= {(cell.level, cell.cell_idx) for cell in src_octree_cells}
    free src_pos, src_conn, src_octree_cells

dest_pos, dest_conn = kuhn_tetrahedralise_union(union_leaves, bbox)
dest_pos, dest_conn = deduplicate_nodes(dest_pos, dest_conn)         # safety

# Phase B — Per-cycle field projection
upload dest_pos                  → dest_nodes_gpu (N_dest × 3)
allocate field stacks            → fields[name][k] for each (name, k)

for k in 0, 1, ..., N_seq - 1:
    src_pos, src_conn, src_fields = load_mesh_from_pvtu(             [§4.2]
        <source pvtu k>, field_names=ALL_POINT_DATA_FIELDS
    )
    src_pos, src_conn = deduplicate_nodes(src_pos, src_conn)
    src_octree_gpu, src_mesh_gpu = build_source_structures(src_pos, src_conn)
    host_elem_ids = vmap(search_l2_vectorized)(dest_nodes_gpu, src_octree_gpu)
    for fname, src_field in src_fields.items():
        fields[fname][k] = project_p1(dest_nodes_gpu, host_elem_ids,
                                       src_conn, src_pos, src_field)
    free src_octree_gpu, src_mesh_gpu, src_pos, src_conn, src_fields
    flush_to_disk(k)             # incremental: a partial run leaves usable output

# Phase C — Storage (both formats from same in-memory state)
write_jaxtrace_npz_bundle(dest_pos, dest_conn, fields, manifest)
write_vtkhdf_archive    (dest_pos, dest_conn, fields, manifest)

RUNTIME (every tracking run)
────────────────────────────
load NPZ bundle (mesh + per-field cycle stacks)
build dest MALMO octree once  (existing path: extract_octree_cells_parent_cube
                                + upload_mesh_aligned_octree_to_gpu)
displacement_at_step(t) = fields['Displacement'][t mod N_seq]   # array slice
if --reconstruct-pin: apply pin reconstruction to displacement at step
run RK4 fused tracking via create_rk4_comparison(...) — unchanged from
the fixed-mesh path
```

The runtime tracker continues to do its existing displacement → velocity
frame differencing internally; the destination mesh just looks like a
fixed-topology FEMUSS mesh to it.

### 4.2 Step-by-step, mapped to **verified** existing modules

The exact module / function names below are taken from the working
imports in `run_tracking.py` and `benchmark_femuss_comparison.py`
(verified 2026-05-04). The `jaxtrace/gpu/search/` directory contains
many experimental variants; the production path uses the specific
modules listed here.

**Production-grade module set (do not substitute alternatives without
testing):**

| Concern | Module | Function |
|---------|--------|----------|
| PVTU mesh loading | `jaxtrace.gpu.mesh_loader` | `load_mesh_from_pvtu` |
| Time-dep mesh loading + velocity stack | `jaxtrace.gpu.mesh_loader_timedep` | `load_velocity_sequence_from_pvtu` |
| Node deduplication | `jaxtrace.gpu.mesh_deduplication` | `deduplicate_nodes` |
| Element-neighbour build | `jaxtrace.gpu.forest` | `build_element_neighbors_array` |
| Mesh upload to GPU | `jaxtrace.gpu.tracking.mesh_data_gpu` | `upload_mesh_to_gpu` |
| Global Morton octree (build) | `jaxtrace.gpu.search.morton_octree_builder` | `build_global_morton_octree` |
| Global Morton octree (upload) | `jaxtrace.gpu.search.morton_global_search` | `upload_global_morton_to_gpu` |
| Centroid-cell extraction (paper §4.2) | `jaxtrace.gpu.search.mesh_aligned_octree_parent_cube` | `extract_octree_cells_parent_cube` |
| Vertex-multi extraction (alternative) | `jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi` | `extract_octree_cells_vertex_multi` |
| Mesh-aligned octree upload | `jaxtrace.gpu.search.mesh_aligned_octree_gpu` | `upload_mesh_aligned_octree_to_gpu` |
| AA-detection metadata | `jaxtrace.gpu.search.aa_detection` | `precompute_aa_metadata`, `precompute_element_vertices`, class `AxisAlignedMetadata` |
| Inverse-matrix precompute | `jaxtrace.gpu.search.point_in_tet_inverse` | `precompute_inverse_matrices` |
| AA setters on GPU | `jaxtrace.gpu.search.point_in_tet_methods` | `set_corrected_metadata`, `set_inverse_matrices_gpu` |
| Search kernels (the four `_where` variants) | `jaxtrace.gpu.search.mesh_aligned_point_location` | `search_mesh_aligned_octree_multi_local_where` (3×3×3, MALMO §4.4), `search_mesh_aligned_octree_static_where`, `search_mesh_aligned_octree_5x5x5_where`, `search_l2_vectorized` |
| Initial assignment cascade | `jaxtrace.gpu.tracking.initial_assignment_cascading` | (used to seed L0 cache) |
| Fully-fused RK4 step (time-dep) | `jaxtrace.gpu.tracking.rk4_fully_fused_timedep` | (built via `create_rk4_comparison` in `benchmark_femuss_comparison.py`, line 672) |

**Stage 1 — Reference mesh.**

This is the same construction `benchmark_femuss_comparison.py` does
today (lines 1582–1644) — we just call it once at preprocessing time:

```python
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import (
    extract_octree_cells_parent_cube,
)
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import (
    upload_mesh_aligned_octree_to_gpu,
)
from jaxtrace.gpu.search.aa_detection import (
    precompute_aa_metadata, precompute_element_vertices, AxisAlignedMetadata,
)
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
from jaxtrace.gpu.search.point_in_tet_methods import (
    set_corrected_metadata, set_inverse_matrices_gpu,
)

# Build destination mesh as the union octree of all cycle indices (§3.2).
# Phase 1: collect (level, cell_idx) pairs across all k.
union_leaves = collect_union_leaves(
    source_pvtu_pattern=str(source_dir / f"{stem}_{{k}}.pvtu"),
    cycle_indices=range(N_seq),
)

# Phase 3: emit Kuhn tetrahedra per leaf cube.
dest_pos, dest_conn = kuhn_tetrahedralise_union(union_leaves, bbox)
dest_pos, dest_conn = deduplicate_nodes(dest_pos, dest_conn)
dest_conn = dest_conn.astype(np.int32)

# (We DON'T need a destination-mesh octree for projection — the projection
# loop uses the SOURCE octree at each k. We only need the destination
# octree for the runtime tracker, which the existing pipeline builds anyway.)
```

**Stage 2 — Source octree per cycle index k.**

For each `k`, replicate the lines 1466–1644 of
`benchmark_femuss_comparison.py` against PVTU `k`. This is currently
done **once** in the benchmark; we now do it **N_seq times** in a loop:

```python
def build_source_structures(source_pvtu: Path, field_names: list):
    """Returns everything we need to evaluate any of the listed fields
    at arbitrary query points."""
    pos, conn, fields = load_mesh_from_pvtu(source_pvtu, field_name=...)
    # NOTE: load_mesh_from_pvtu currently accepts a single field_name and
    # returns (pos, conn, vel). To get multiple fields, either:
    #   (a) call it once with field_name=None, then pull additional arrays
    #       directly from the VTK reader's PointData (cheap, already loaded), OR
    #   (b) extend load_mesh_from_pvtu to accept a list (preferred — see §10 Step 1)
    pos, conn = deduplicate_nodes(pos, conn)
    conn = conn.astype(np.int32)

    # Match benchmark_femuss_comparison.py:1517-1571 exactly:
    aa_metadata = precompute_aa_metadata(pos, conn, ...)
    element_vertices = precompute_element_vertices(pos, conn)
    M_inv_array, p0_array = precompute_inverse_matrices(pos, conn, ...)

    # Octree
    octree_struct = build_global_morton_octree(
        node_positions=pos, connectivity=conn,
        leaf_capacity=256, max_depth=21, verbose=False,
    )
    cells = extract_octree_cells_parent_cube(
        pos, conn, tolerance=1e-6, verbose=False,
    )

    # Element neighbours (skip for projection — only needed for L1 search)
    element_neighbors = build_element_neighbors_array(
        conn, method='face', verbose=False,
    )

    # Upload everything
    mesh_gpu = upload_mesh_to_gpu(conn, pos, element_neighbors, verbose=False)
    morton_gpu = upload_global_morton_to_gpu(octree_struct, conn, pos)
    octree_gpu = upload_mesh_aligned_octree_to_gpu(conn, pos, cells, verbose=False)

    # AA metadata + inverse matrices on GPU + setters
    aa_gpu = AxisAlignedMetadata(
        base_vertex_indices=jax.device_put(aa_metadata.base_vertex_indices),
        base_vertices=jax.device_put(aa_metadata.base_vertices),
        inv_edge_lengths=jax.device_put(aa_metadata.inv_edge_lengths),
        axis_indices=jax.device_put(aa_metadata.axis_indices),
        is_axis_aligned=jax.device_put(aa_metadata.is_axis_aligned),
    )
    element_vertices_gpu = jax.device_put(element_vertices)
    M_inv_gpu = jax.device_put(M_inv_array)
    p0_gpu = jax.device_put(p0_array)
    set_corrected_metadata(aa_gpu, element_vertices_gpu)
    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)

    return SourceStructures(
        pos=pos, conn=conn, fields=fields,
        mesh_gpu=mesh_gpu, morton_gpu=morton_gpu, octree_gpu=octree_gpu,
        aa_gpu=aa_gpu, element_vertices_gpu=element_vertices_gpu,
        M_inv_gpu=M_inv_gpu, p0_gpu=p0_gpu,
    )
```

**Stage 3 — Point-locate reference nodes in source mesh (GPU search).**

This is where we use the existing JAXTrace search kernel, but with
**ref-mesh nodes as queries** instead of moving particles.

The benchmark uses `search_mesh_aligned_octree_multi_local_where` for
the inner-loop 3×3×3 search; for projection we use the simpler L2
fallback `search_l2_vectorized` because:

1. There is no L0 cache to seed (this is a one-shot batch call)
2. The L2 path covers any point in the bbox without needing the
   neighbour topology
3. Implementation is **vmap over query points**, exactly the right
   shape for projecting N_ref nodes

```python
from jaxtrace.gpu.search.mesh_aligned_point_location import search_l2_vectorized

# search_l2_vectorized signature (verified from line 136):
#   search_l2_vectorized(pos: jax.Array, octree_gpu) -> jnp.int32
# Returns the host element ID (or -1 if no candidate cell contains pos).

@jax.jit
def locate_ref_nodes(ref_nodes, octree_gpu):
    # vmap over the leading (N_ref,) axis
    return jax.vmap(lambda p: search_l2_vectorized(p, octree_gpu))(ref_nodes)

ref_nodes_jax = jax.device_put(ref_pos)         # (N_ref, 3), uploaded once
host_elem_ids = locate_ref_nodes(ref_nodes_jax, src.octree_gpu)  # (N_ref,)
```

The L2 search is correct but visits all candidate cells in the parent-
cube neighbourhood — slower than the cached-L0 path used at runtime. For
the projection batch (a single point-location call per cycle index over
N_ref ≈ 2 M nodes), this is comfortably fast: from §11.4 of the
feasibility doc, the cost is dominated by source octree build, not by
the search itself.

**Important caveat — handling missing fits.** Although the reference
and source meshes share the same bbox, an interior reference node can
still fall in a region the source mesh under-covers (e.g. concave
surface near the tool). Add a fallback:

```python
host_ids = jax.device_get(host_elem_ids)
n_misses = int(np.sum(host_ids < 0))
if n_misses:
    # Snap missed ref nodes to the closest source-element centroid
    # using a kdtree over source centroids — robust fallback.
    print(f"  {n_misses}/{len(host_ids)} ref nodes missed L2 search; using nearest-centroid fallback")
    ...
```

For the FEMUSS dataset in practice we expect **zero** misses if the
reference mesh = source mesh at index 0 (the trivial case), or
<<0.1% if the reference is a slightly different mesh.

**Stage 4 — P1 interpolation of fields.**

The barycentric P1 evaluator is **already implemented** in
`rk4_fully_fused_timedep.py` lines 498–548 as a closure called
`interpolate_velocity_single(pos, elem_id, velocity_field)`. It is the
exact computation we need for projection. The closure is not currently
exported, so we lift it into a module-level helper:

```python
# New file: jaxtrace/gpu/tracking/p1_interpolation.py
@jax.jit
def interpolate_p1_at_point(pos, elem_id, conn, node_pos, field):
    """
    P1 interpolation of `field` at `pos`, given the host element `elem_id`.
    Vectorisable shape: scalars for pos / elem_id, arrays for conn / node_pos / field.

    field can be (n_nodes,) or (n_nodes, K) — the last axis is the field's
    component count. Returns shape (K,) or scalar.
    """
    # Same arithmetic as rk4_fully_fused_timedep:498-548
    ...

@jax.jit
def project_field(ref_nodes, host_elem_ids, conn, node_pos, field):
    """vmap over ref-nodes."""
    return jax.vmap(
        lambda p, e: interpolate_p1_at_point(p, e, conn, node_pos, field)
    )(ref_nodes, host_elem_ids)
```

Then for each k, iterate over the field set discovered from the
PVTU's PointData (or specified via `--fields`):

```python
projected_k = {}
for fname, src_field in src.fields.items():
    projected_k[fname] = project_field(
        dest_nodes_jax, host_elem_ids,
        src.conn, src.pos, src_field
    )
# projected_k is e.g.:
#   {'Displacement': (N_dest, 3),
#    'Pressure':     (N_dest,),
#    'Reactions':    (N_dest, 3),
#    'Temperature':  (N_dest,),
#    'LEVEL':        (N_dest,)}
```

`project_field` works for both vector and scalar fields because the
underlying arithmetic uses elementwise multiply-add on the field array;
the leading shape `(n_nodes, ...)` is preserved through the vmap.

**Stage 5 — Free source-mesh GPU memory between cycle indices.**

```python
# Drop Python references to all GPU arrays for this k:
del src
import jax
jax.clear_caches()  # discard compiled & cached JIT graphs from this k
```

`jax.clear_caches()` matters: each `build_source_structures()` call
JIT-compiles the search kernel against the new source-mesh shapes, and
the compile cache can grow unbounded over 150 iterations otherwise.

**Stage 6 — Save bundle.**

After the per-k loop completes, write **both** the NPZ runtime bundle
and the VTKHDF archive (§6):

```python
# Stack per-k slices into per-field arrays
field_stacks = {
    fname: np.stack([projected[k][fname] for k in range(N_seq)])
    for fname in field_names
}

save_npz_bundle(
    out_dir,
    dest_pos=dest_pos, dest_conn=dest_conn,
    fields=field_stacks,
    manifest={...}
)
save_vtkhdf_bundle(
    out_dir / 'projected.vtkhdf',
    dest_pos=dest_pos, dest_conn=dest_conn,
    fields=field_stacks,
)
```

The destination mesh's AA metadata, inverse matrices and octree are
**recomputed at runtime** from the saved positions+connectivity (the
existing tracker already does this for fixed meshes; no need to cache
them).

### 4.3 Cost estimate (single GPU)

**Phase A** (union octree construction, one pass over all sources):

| Step | Time per k | Notes |
|------|-----------:|:------|
| PVTU I/O | ~50 s | LUMI scratch, dominant cost |
| Dedup + extract_octree_cells_parent_cube | ~30 s | We don't need full AA / M_inv here — only the (level, cell) set |
| Add to union set | <1 s | Python set / hash |
| **Per-k total** | **~80 s** | Less than full source-structure build |

For N_seq = 150: **~3.3 h** one-time, parallelisable across CPU cores
if multi-process is used.

**Phase B** (per-cycle-index field projection):

| Step | Time per k | Notes |
|------|-----------:|:------|
| PVTU I/O (with all field arrays) | ~80 s | Slightly slower than Phase A due to fields |
| Dedup + AA + inverse matrices | ~120 s | Existing CPU code |
| Octree build + GPU upload | ~150 s | Existing code; skip element-neighbours to save ~150 s |
| Point-locate N_dest dest nodes | ~0.5 s | vmap search_l2_vectorized, N_dest ~ 3 M |
| Interpolate 5 fields | ~1 s | Vectorised P1, summed across all fields |
| Free GPU + jax.clear_caches | ~2 s | |
| **Per-k total** | **~350 s** | Without optimisations |

For N_seq = 150: **~14.5 h baseline**, **~4.5 h with optimisations**
(skip neighbours, GPU-side Morton sort).

**Total preprocessing**: Phase A + Phase B ≈ 7.8 h optimised.

### 4.4 Memory profile (peak per cycle index)

**Phase A** (union construction):
- Source mesh on host: ~600 MB
- Source octree cells: ~50 MB
- Union set (cumulative): ~30 MB at end
- **Peak: ~700 MB** host

**Phase B** (projection):
- Source mesh on GPU: ~600 MB octree + ~600 MB inverse matrices ≈ 1.2 GB
- Source field arrays on GPU: ~24 MB per scalar / 72 MB per vector
  → ~250 MB total for the 5-field set
- Destination nodes on GPU: ~36 MB (N_dest = 3 M × 3 × float32)
- Field-staging output: ~250 MB
- **Peak: ~1.7 GB** — well within MI250X GCD budget (64 GB)

---

## 5. Module Layout

Proposed new directory: `jaxtrace/preprocessing/`.

```
jaxtrace/preprocessing/
├── __init__.py
├── union_octree.py              # collect_union_leaves(), kuhn_tetrahedralise_union()
├── source_octree.py             # build_source_structures()
├── projection.py                # project_one_step(), project_field_sequence()
└── bundle_io.py                 # save_npz_bundle, save_vtkhdf_bundle, load_npz_bundle
```

Plus one CLI driver:

```
preprocess_field_projection.py
```

at the repo root, mirroring `run_tracking.py`.

### 5.1 union_octree.py

```python
def collect_union_leaves(
    source_pvtu_pattern: str,
    cycle_indices: list[int],
) -> set[tuple[int, int]]:
    """Phase 1 of Algorithm U-A: union of (level, cell_idx) over all
    sampled cycle indices. Reuses extract_octree_cells_parent_cube;
    discards the per-cube tetrahedra after collecting cube identities."""

def kuhn_tetrahedralise_union(
    union_leaves: set,
    bbox: BBox,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase 3 of Algorithm U-A: emit 6 Kuhn tets per leaf cube,
    deduplicate shared nodes. Returns (positions, connectivity)."""
```

### 5.2 source_octree.py

Wraps `benchmark_femuss_comparison.py:1582–1644` into a clean factory:

```python
@dataclass
class SourceStructures:
    pos: np.ndarray; conn: np.ndarray
    fields: dict[str, np.ndarray]   # name → host-side field array
    mesh_gpu: ...; morton_gpu: ...; octree_gpu: ...
    aa_gpu: ...; M_inv_gpu: ...; p0_gpu: ...

def build_source_structures(
    pos, conn, fields,
    skip_neighbours: bool = True,   # 150 s saving per k; safe for projection
) -> SourceStructures: ...
```

### 5.3 projection.py

```python
def project_one_step(
    src_pvtu: Path,
    dest_nodes_gpu: jnp.ndarray,
    field_names: list[str] | str = "*",
) -> dict[str, np.ndarray]:
    """For one cycle index: load source, build structures, locate dest
    nodes, interpolate every requested field. Returns {fname: array}.
    Frees source GPU state at exit."""

def project_field_sequence(
    source_pvtu_pattern: str,
    cycle_indices: list[int],
    dest_nodes: np.ndarray,
    field_names: list[str] | str,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, np.ndarray]:
    """Loop over cycle indices, calling project_one_step. Writes per-k
    NPZ slices to output_dir/_partial/. Returns the final stacked
    field arrays after concatenation."""
```

### 5.4 bundle_io.py

```python
def save_npz_bundle(out_dir, dest_pos, dest_conn, fields, manifest): ...
def save_vtkhdf_bundle(out_path, dest_pos, dest_conn, fields, manifest): ...
def save_vtu_collection(out_dir, dest_pos, dest_conn, fields): ...     # optional
def load_npz_bundle(path) -> tuple[mesh, fields, manifest]: ...
```

---

## 6. Output Formats

Two formats are produced from the same in-memory result. A third is
available behind an opt-in flag.

### 6.1 Primary archive — VTKHDF (single-file, static-mesh time series)

VTKHDF is the modern HDF5-based VTK format with explicit
[static-mesh time-series support added in 2024–2025](https://www.kitware.com/vtkhdf-file-format-2025-status-update/):
the geometry is written once and shared across all time steps, with
per-step point-data groups. Paraview reads it natively in recent
versions.

Why VTKHDF and not XDMF+HDF5 or VTU+PVD:
- **Memory at load time.** The
  [ParaView discourse forum thread](https://discourse.paraview.org/t/substantial-memory-use-differences-between-vtu-and-multi-block-xdmf2-hdf5/4136)
  documents a 6× memory blowup for XDMF2+HDF5 at 640 k elements
  because the reader duplicates the points array per block; for our
  ~10 M-element mesh and 150 time steps this would be infeasible.
- **Single file vs. 150 files.** A `.pvd`+150×`.vtu` collection
  duplicates the destination mesh 150× on disk (~22 GB overhead).
- **Static-mesh support built in.** The VTKHDF writer caches geometry
  between time steps when the mesh is unchanged — exactly our case.

File layout (HDF5 hierarchy):

```
projected.vtkhdf
├── /VTKHDF                              (group with version attribute)
│   ├── /VTKHDF/Type                     "UnstructuredGrid"
│   ├── /VTKHDF/Points                   (N_dest, 3) float64
│   ├── /VTKHDF/Connectivity             (4 * N_tet,) int64
│   ├── /VTKHDF/Offsets                  (N_tet+1,) int64
│   ├── /VTKHDF/Types                    (N_tet,) uint8 = VTK_TETRA
│   ├── /VTKHDF/PointData/
│   │   ├── Displacement                 (N_dest, 3, N_seq) float32
│   │   ├── Pressure                     (N_dest, N_seq) float32
│   │   ├── Reactions                    (N_dest, 3, N_seq) float32
│   │   ├── Temperature                  (N_dest, N_seq) float32
│   │   └── LEVEL                        (N_dest, N_seq) float32
│   └── /VTKHDF/Steps                    (N_seq,) int64 = step indices
└── (HDF5 attributes carry manifest metadata)
```

Writer: use the `vtkHDFWriter` class (available in VTK 9.4+) or write
the hierarchy directly via `h5py` following the
[VTKHDF format specification](https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/index.html).

Estimated size for the C3 dataset (N_dest ≈ 3 M nodes, all 5 fields,
N_seq = 150, float32): mesh ~150 MB, fields ≈ 10 GB. With chunked
storage and gzip compression, ~5–7 GB on disk.

### 6.2 Runtime bundle — JAXTrace NPZ

For the GPU runtime path we keep a separate, minimal NPZ bundle.
Reasons to keep both:
- VTKHDF is a generic visualization-friendly format; loading it via the
  VTK Python bindings adds a runtime dependency we don't need on LUMI.
- NPZ files mmap directly into numpy → JAX device transfer, bypassing
  any VTK indirection.
- Per-field NPZ files keep each field independently mmap-able; we can
  load only `Displacement` if that's all the tracker uses.

Layout:

```
output_dir/
├── manifest.json
├── ref_mesh.npz
│   ├── node_positions: (N_dest, 3) float32
│   └── connectivity:   (N_tet_dest, 4) int32
├── field_Displacement.npz                # (N_seq, N_dest, 3) float32
├── field_Pressure.npz                    # (N_seq, N_dest) float32
├── field_Reactions.npz                   # (N_seq, N_dest, 3) float32
├── field_Temperature.npz                 # (N_seq, N_dest) float32
└── field_LEVEL.npz                       # (N_seq, N_dest) float32
```

`manifest.json`:
```json
{
  "version": 1,
  "source_pattern": "C3_{timestep}.pvtu",
  "cycle_length": 150,
  "n_dest_nodes": 3000000,
  "n_dest_elements": 12000000,
  "bbox_min": [-0.02, -0.015, -0.006],
  "bbox_max": [ 0.04,  0.015,  0.0  ],
  "edge_min": 5.66e-5,
  "edge_max": 3.99e-3,
  "fields": {
    "Displacement": {"shape": [150, 3000000, 3], "dtype": "float32", "components": 3},
    "Pressure":     {"shape": [150, 3000000],    "dtype": "float32", "components": 1},
    "Reactions":    {"shape": [150, 3000000, 3], "dtype": "float32", "components": 3},
    "Temperature":  {"shape": [150, 3000000],    "dtype": "float32", "components": 1},
    "LEVEL":        {"shape": [150, 3000000],    "dtype": "float32", "components": 1}
  },
  "construction": {
    "method": "union_octree_kuhn",
    "n_cycle_indices_used": 150,
    "octree_levels": 7
  },
  "preprocessor_version": "0.1.0"
}
```

Per-field NPZ size (N_seq=150, N_dest=3M, float32): 1.7 GB scalar /
5.4 GB vector. Total ~16 GB uncompressed on disk; compressible to
~50–60% with gzip if disk space matters.

### 6.3 Optional — per-step VTU + .pvd collection

Behind `--write-vtu-collection` flag, for environments that don't
support VTKHDF (older Paraview, third-party tools). One VTU per cycle
index, plus a `.pvd` collection. Disk size penalty as noted (~22 GB
mesh duplication overhead). Off by default.

---

## 7. Runtime Integration

The runtime tracker change is **minimal**.

### 7.1 Loader

Replace the time-dependent loader path in `run_tracking.py`:

```python
# Before (current code):
node_pos, conn, vel_seq = load_velocity_sequence_from_pvtu(
    base_path=source_dir,
    file_pattern="C3_{timestep}.pvtu",
    timestep_range=(0, 149),
    field_name="Displacement",
)
# Limitation: assumes fixed source mesh, fails on r-adaptive meshes.

# After (projected mode):
mesh, fields, manifest = load_npz_bundle(bundle_path)
node_pos    = mesh['node_positions']
conn        = mesh['connectivity']
disp_seq    = fields['Displacement']    # (N_seq, N_dest, 3) float32
# Plus fields['Pressure'], fields['Temperature'], fields['LEVEL'], fields['Reactions']
# as needed by the tracker / output writer.
```

The runtime tracker continues to do its existing displacement →
velocity frame differencing internally; the destination mesh just
looks like a fixed-topology FEMUSS mesh to it.

### 7.2 RK4 step indexing

The fully-fused RK4 step is built by `create_rk4_comparison()` in
`benchmark_femuss_comparison.py` (line 672). It takes the displacement
(or velocity) sequence as a `(T, N_nodes, 3)` array and indexes it
internally by step number. **No code change is needed in the RK4
builder itself**: we pass `disp_seq` of shape `(N_seq, N_dest, 3)` and
the existing time-index logic uses `t % N_seq` cyclic indexing as
already configured by `--velocity-range`.

If `--reconstruct-pin` is enabled, apply
`reconstruct_pin_velocity(node_pos, disp_seq, pin_params, ...)` once
after the bundle load and before passing the sequence to the RK4
builder.

### 7.3 Mode flag in CLI

Add to `run_tracking.py`:

```
--mode {source,projected}
  source: current behaviour, requires fixed-topology source mesh
  projected: load reference-mesh bundle, run as fixed-mesh

--bundle PATH
  path to projected bundle (NPZ dir or HDF5 file); required if --mode projected
```

The two modes use the **same** rest-of-the-pipeline code — octree build,
RK4 step, output writing. Only the loader and the velocity-at-step
function differ.

---

## 8. Validation Strategy

### 8.1 Unit tests

1. **Self-projection identity.** If reference mesh = source mesh at
   cycle index 0, projecting source velocity at index 0 onto the
   reference must yield the source velocity *exactly* (up to FP
   round-off). Tolerance: 1e-7 for float32.

2. **P1 reproduction.** A P1 polynomial velocity field `u(x) = a + B·x`
   must project exactly onto a Kuhn-tetrahedral reference. Tolerance:
   1e-6.

3. **Conservation of bounds.** `min(u_ref_k) >= min(u_src_k)` and
   `max(u_ref_k) <= max(u_src_k)` (P1 interpolation preserves bounds).

### 8.2 End-to-end accuracy

Run a short tracking window (e.g. 100 steps, one third of a cycle) in
both modes:

- **Mode A (direct):** current path with the cycle-cached source
  octrees of §11.5
- **Mode B (projected):** the new bundle

Compare:

- Per-particle position trajectories (max L2 error)
- Final-time histograms of position
- Number of particles lost (out-of-bounds, search misses)

Acceptance: max trajectory L2 error < 10× the source-mesh element
size at the relevant location, after one full cycle (~150 steps).

### 8.3 Destination-mesh resolution sweep

Compare destination meshes built from different `cycle_indices`
subsets:
- `union(k=0)` — single source mesh as destination (lower bound)
- `union(k=0, 75)` — two cycle samples
- `union(k=0, 30, 60, 90, 120)` — five-step bootstrap (cheap)
- `union(k=0..149, stride 10)` — 16-sample bootstrap from §15.4
- `union(k=0..149)` — full 150-step union (production)

For each, measure trajectory error vs the source-mode reference. Plot
error vs N_dest. This justifies the chosen subset and gives a clean
"resolution" knob. Expectation: error converges as the cycle subset
grows, then plateaus; pick the smallest subset on the plateau.

---

## 9. Performance Budget

For the full 2,684-step tracking run:

| Phase | Mode A (cycle cache) | Mode B (projected) |
|-------|:--------------------|:-------------------|
| Preprocessing (one-time) | ~14 h (first cycle on-demand build) | ~13 h (full projection) |
| Steady-state per-step | tracking + cache load + L2 flush at cycle wrap | tracking only |
| Steady-state total | ~3.7 h tracking + ~10 min overhead per cycle wrap | ~3.7 h tracking |
| GPU memory | 1.4 GB (active octree) | 120 MB octree + 3.6 GB sequence |
| Code paths | New: cache I/O, dirty flag, swap | Unchanged from fixed-mesh |
| Risk | Cache corruption / I/O slowdowns | Bigger one-time investment |

Mode B is preferred unless the projection error (Section 8.2) is
unacceptable.

---

## 10. Implementation Order

Suggested sequence, each step independently testable. All file paths
are relative to the repo root. Total estimated new code: **~1100 lines**,
all wrappers over already-validated JAXTrace functions.

**Step 1. Lift the P1 evaluator out of the RK4 closure.**
File: new `jaxtrace/gpu/tracking/p1_interpolation.py`. Copy lines
498–548 of `jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py` and turn
the `interpolate_velocity_single` closure into a top-level
`@jax.jit`-friendly function with explicit array arguments (no closure
captures). Add a `vmap`-batched helper `project_field(query_points,
host_elem_ids, conn, node_pos, field) -> projected_field` that handles
both scalar (1-component) and vector (3+ component) fields. Add unit
tests: identity (project source onto itself) and P1 reproduction
(linear field on a Kuhn cube). *~100 lines.*

**Step 2. Extend `load_mesh_from_pvtu` for multi-field loading.**
File: `jaxtrace/gpu/mesh_loader.py`. New optional kwarg
`field_names: list[str] | None`. When supplied, returns
`(positions, connectivity, dict_of_fields)`. Keep single-name path
backward-compatible. Auto-detect all point-data fields when
`field_names="*"`. *~40 lines.*

**Step 3. `SourceStructures` factory.**
File: new `jaxtrace/preprocessing/source_octree.py`. Factor
`benchmark_femuss_comparison.py:1582–1644` into a function
`build_source_structures(pos, conn) -> SourceStructures` returning a
dataclass holding `mesh_gpu`, `morton_gpu`, `octree_gpu`, `aa_gpu`,
`M_inv_gpu`, `p0_gpu`, plus host `pos`, `conn`. Same code as the
benchmark, just refactored to a callable for reuse in the projection
loop. *~120 lines.*

**Step 4. Union-octree construction (Algorithm U-A from §3.2).**
File: new `jaxtrace/preprocessing/union_octree.py`. Two passes:

```python
def collect_union_leaves(source_pvtu_pattern: str, k_indices: list[int]) -> set:
    """Phase 1: union of (level, cell_idx) across cycle indices.
    Reuses extract_octree_cells_parent_cube; discards per-cell tets."""

def kuhn_tetrahedralise_union(union_leaves, bbox) -> (positions, connectivity):
    """Phase 3: emit 6 Kuhn tets per union leaf cube; dedup nodes."""
```

Add unit tests against simple synthetic 1-cube and 8-octant cases.
The Kuhn split takes a cube's 8 corners and returns 6 tets (the
six monotone paths from corner `000` to corner `111`). *~250 lines.*

**Step 5. Single-k projection.**
File: new `jaxtrace/preprocessing/projection.py`. Function:

```python
def project_one_step(
    src_pvtu: Path,
    dest_nodes_gpu: jnp.ndarray,
    field_names: list[str] | str = "*",
) -> dict[str, np.ndarray]:
```

Calls Step 2 (multi-field load) and Step 3 (build source structures),
runs `vmap(search_l2_vectorized)` to get host_elem_ids, calls Step 1's
`project_field` for each requested field, returns `{field_name:
projected_array}`. Frees source GPU state at exit. Self-projection
test: project source 0 onto itself → exact (within FP). *~180 lines.*

**Step 6. CLI driver with incremental disk output.**
File: new top-level `preprocess_field_projection.py` (mirrors
`run_tracking.py`). Stages:

1. Run `collect_union_leaves` over all cycle indices (Phase 1)
2. Run `kuhn_tetrahedralise_union` once
3. Save destination mesh
4. Loop k = 0..N_seq-1: call `project_one_step(k, ...)`, write
   per-k slice to `out_dir/_partial/step_{k:04d}.npz`
5. After loop: stack slices into per-field NPZ files
6. Write VTKHDF archive (Step 8)

Add `--resume` flag that skips cached k slices. Add `--bootstrap`
flag that uses stride-10 sampling for the union (fast 30-min
validation pass before committing to the full 5-h run). *~200 lines.*

**Step 7. NPZ bundle I/O.**
File: new `jaxtrace/preprocessing/bundle_io.py`. Functions
`save_npz_bundle(out_dir, dest_pos, dest_conn, fields_dict, manifest)`
and `load_npz_bundle(path) -> (mesh, fields_dict, manifest)`. Manifest
JSON with the schema in §6.2. *~100 lines.*

**Step 8. VTKHDF bundle I/O.**
File: extend `bundle_io.py` with `save_vtkhdf_bundle(path, ...)`.
Implementation choice: write the HDF5 hierarchy directly via `h5py`
following the
[VTKHDF spec](https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/index.html)
(works without a VTK 9.4+ runtime dependency). Use chunked datasets
with `(N_dest, 3, 1)` chunking for vector fields so per-step slices
are contiguous reads. Apply gzip-1 compression on the field data
arrays. *~150 lines.*

**Step 9. Wire `run_tracking.py --mode projected`.**
Add `--mode {source,projected}` flag and `--bundle PATH`. When
`projected`, replace the call to `load_velocity_sequence_from_pvtu`
with `load_npz_bundle`; the rest of the pipeline (build dest octree,
build RK4 step, run) is unchanged. The runtime tracker continues to
do its existing displacement → velocity frame differencing internally,
indexing the cycle stack by `t mod N_seq`. *~70 lines.*

**Step 10. Pin-velocity reconstruction as runtime opt-in.**
The function already exists at `benchmark_femuss_comparison.py:390`.
Promote it to a public location: new
`jaxtrace/runtime/pin_velocity.py`. Add `--reconstruct-pin` flag to
`run_tracking.py`. When set: after loading the projected
`Displacement` field, call `reconstruct_pin_velocity` with the
destination mesh nodes and pin parameters. Default off. *~50 lines.*

**Step 11. Optional per-step VTU + .pvd export.**
Behind `--write-vtu-collection` flag. Existing JAXTrace VTU writer can
be reused; loop over cycle indices and emit one VTU each, plus a
`.pvd` collection file. *~80 lines.*

**Step 12. Validation harness.**
New `tests/test_field_projection.py` covering: (a) self-projection
identity, (b) P1 reproduction on synthetic Kuhn mesh, (c) bound
preservation (`min/max(projected) ⊆ min/max(source)`), (d) end-to-end
trajectory comparison: 100-step window in `--mode source` vs `--mode
projected` with the same seed, max trajectory L2 error reported. *~150
lines.*

**Bootstrap milestone.** After Steps 1–7 (NPZ bundle path complete),
running on LUMI with `--bootstrap` should produce a usable
destination mesh + projected fields in ~1 hour. Validate Step 12 (a)
and (b) before scaling to the full 150-index union.

**Production milestone.** After Step 9, `run_tracking.py --mode
projected` should reproduce the existing source-mode trajectory
within tolerance over a 100-step window.

---

## 11. Risks and Mitigations

| Risk | Mitigation |
|------|:----------|
| Destination node lies outside source mesh at some k (host_elem_ids = -1) | Sanity-check at projection time; log and either extrapolate via nearest-centroid fallback or carry forward last-valid value |
| Float32 underflow in barycentric coords near tet faces | Use float64 for the inverse matrices and barycentric solve; cast field result to float32 at the end |
| Disk full during per-k caching (~16 GB final + 150 × ~80 MB partials) | Use compressed `.npz` for partials; `--cleanup-partials` after stack-and-write succeeds |
| Out-of-memory during per-k projection if N_dest > 5 M | Block-wise vmap over dest nodes (existing pattern in `batch_velocity_interpolation`); chunk size ~256 k |
| Hanging-node mismatch at level transitions in the union octree | Algorithm U-A's Kuhn split is conformal *within* a leaf cube; cross-cube faces of differently-sized cubes need explicit hanging-node treatment. Either enforce 2:1 balance in `collect_union_leaves` (Step 4) or add a face-conformity pass when building destination connectivity. |
| Source octree build dominates preprocessing (~150 × 350 s baseline) | Apply §11.4 optimisations: skip element neighbours (saves ~150 s/k), GPU-side Morton sort (saves ~100 s/k). Brings Phase B down from ~14 h to ~4 h. |
| VTKHDF reader compatibility on user's Paraview version | If reader doesn't support static-mesh time series, fall back to `--write-vtu-collection` (Step 11) for visualisation; NPZ bundle is unaffected. |
| Cycle period mismatch between case files (C3, C4, A2) | The CSV from `analyze_mesh_variation.py` reports the period; spot-check step-N_seq ≈ step-0 in the per-case CSV before launching union construction (1-minute check). |
