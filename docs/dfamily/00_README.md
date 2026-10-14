# The D-family lost-particle investigation

**Resolved 2026-10-13.** D2 particle loss went from **4.07% to 0.20%** through
two independent defects, both in the mesh-aligned octree point-location path.

Keywords for future searches: D-family, D-ConcavityTilt, lost particles,
frozen particles, ElementID = -1, MALMO, mesh-aligned octree, cell-overflow
truncation, MAX_ELEMS_PER_CELL, anisotropic pitch, non-Kuhn.

---

## The two defects

### 1. Registration/lookup lattice disagreement ("anisotropic pitch")

Registration and lookup divided coordinates by **different** numbers: Pass 1
used each element's own `cell_size`, Pass 2 a borrowed neighbour's, and the
search the per-level median. On an anisotropic mesh those lattices drift apart,
and because the error is relative it grows with distance from the origin —
reaching 2 cells at a particle position while the 3×3×3 neighbourhood reaches
only ±1, so the host cell was never visited.

Fixed by deriving one per-level pitch **before either pass registers**, so all
three agree by construction. Recovery on previously-lost particles: 58.2% →
**90.4%**, all three mismatch categories to zero, at no cost (max cell
occupancy actually improved 95 → 79).

Account: [`07_root_cause.md`](07_root_cause.md). Guard:
`scripts/dfamily/test_lattice_invariant.py` (fails on the pre-fix code at
66.67% unreachable cells, passes after, no-op on isotropic meshes).

### 2. Cell-overflow truncation (the larger cause)

The production L2 search runs `fori_loop(0, MAX_ELEMS_PER_CELL)` — a **static**
bound so XLA can unroll it. The inner guard stops a short cell early but never
**extends** the loop, so any element at index ≥ the bound in its cell is never
tested. The bound was a fixed **32**; D2 cells hold up to **79**, D1 up to
**115**.

On D2 that left **12,661,126 element entries unreachable** across 13.45% of
cells. Fixed by deriving the bound from the octree that was actually built.

Measured, matched A/B, 80k particles, identical seeds:

| | bound 32 | **bound 79 (derived)** |
|---|---:|---:|
| final lost | 2,341 (2.93%) | **157 (0.20%)** |
| throughput | 2.3 step/s | 1.3 step/s |

**93.3% of the loss recovered** at ~1.7× per-step cost. Account:
[`11_max_elems_per_cell.md`](11_max_elems_per_cell.md).

### Why "cell-overflow truncation" and not "extended elements"

The failing hosts *are* mostly non-Kuhn and z-elongated (93.1% and 93.3%), but
that is a **risk factor, not the mechanism**: AABB registration files each
non-Kuhn element into ~8 cells, which is what pushes a cell past the bound. An
element of any shape is lost if it lands past the bound. The defect is the
truncation; elongation only explains *which* cells overflow.

---

## Why the D family and not A/B/C

Two independent reasons, both measured:

| | A1 | C1 | D1 | D2 |
|---|---:|---:|---:|---:|
| non-Kuhn fraction | 0.84% | ~1% | 50–59% | 50–59% |
| **max cell occupancy** | **31** | **30** | **115** | **79** |
| vs the old bound of 32 | under | under | **over** | **over** |

A/B/C max occupancy sits *just under* 32 by coincidence, so they never
truncated. That is why three families were clean and the fourth was not.

---

## The amplifier: a latched dead state

Neither defect alone explains why a particle stayed lost forever. The substage
trace showed 110,997 failed-substage records from only **3,302 particles** —
~34 each, with **zero** substage displacement:

```
k1 fails → vel_k1 = 0 (k1 has no predecessor, so last_valid_vel cannot help)
         → pos_k1 = pos → k2, k3, k4 fail at the identical point
         → pos_final = pos → elem_final = search(pos) fails again → -1 forever
```

`RK4_CARRY_LAST_VALID_VEL` (implemented, default off) completes the chain
`last_vel → k1 → k2 → k3 → k4` so a particle that misses once keeps moving.
Account: [`10_substage_trace_mechanism.md`](10_substage_trace_mechanism.md).

---

## Two methodological traps, recorded so they are not repeated

**A fixed constant has now failed twice.** 8 was too small for cylA (max 24);
32 too small for the D family. The config docstring already warned that
overflow is "SILENTLY TRUNCATED". Derive it from the mesh.

**Check which search function the configured registration method dispatches
to.** This defect was wrongly "refuted" earlier in the investigation by
checking `max_tests` in `search_mesh_aligned_octree_multi_local_where`, which
production does *not* call on the `parent_cube` path. It calls
`search_mesh_aligned_octree_static_where`, whose bound is real. That error is
also why standalone replays kept reporting 94–95% while the fused kernel found
7.1% — the replays used the dynamically bounded variant.

---

## Also established along the way

- **All 20 PinShapes cases have a static mesh**, certified over every loaded
  timestep with the piece index rotating (LUMI job 22787422, 3,354 reads,
  22 min). One search structure per run is valid; nothing needs rebuilding per
  revolution. B2 is the one case loading 2..201 rather than 34..199.
  See [`08_mesh_is_static_all_cases.md`](08_mesh_is_static_all_cases.md) and
  [`../meshes/PINSHAPES_MESH_FAMILIES.md`](../meshes/PINSHAPES_MESH_FAMILIES.md).
- **Seed-box bug**: `--seed-source box-frac` takes fractions of the mesh
  *bounding box*, which for D extends above the workpiece, so ~18% of seeds
  land in air. Clamp the upper z fraction per case.
- **`zero_vel` does not freeze a particle**, it integrates a partially zeroed
  velocity — so it silently biases trajectories toward stagnation rather than
  stopping them. Only `skip_step` freezes.
- **Ruled out by measurement, do not re-investigate**: mesh motion during a
  revolution, Morton mis-mapping, unregistered hosts, levels outside the walked
  range, the point-in-tet tolerance, float32 precision, the degeneracy guard,
  the level-set/tool mask, and search reach (5×5×5 or neighbour-cell
  registration would recover 0.0%).

---

## Document index

| file | contents |
|---|---|
| [`01_diagnosis.md`](01_diagnosis.md) | how the investigation started (superseded) |
| [`05_root_cause_and_fix.md`](05_root_cause_and_fix.md) | first pitch account (partly superseded) |
| [`06_pending_actions.md`](06_pending_actions.md) | settings to apply when preparing runs |
| [`07_root_cause.md`](07_root_cause.md) | **defect 1**: the lattice disagreement |
| [`08_mesh_is_static_all_cases.md`](08_mesh_is_static_all_cases.md) | the static-mesh certification |
| [`09_d2_full_run_residual.md`](09_d2_full_run_residual.md) | the full run that exposed defect 2 |
| [`10_substage_trace_mechanism.md`](10_substage_trace_mechanism.md) | the latched dead state |
| [`11_max_elems_per_cell.md`](11_max_elems_per_cell.md) | **defect 2**: cell-overflow truncation |
| [`12_occupancy_optimization.md`](12_occupancy_optimization.md) | occupancy experiments in detail |
| [`13_throughput_roadmap.md`](13_throughput_roadmap.md) | **throughput roadmap**: every candidate, measured gain, effort, risk |
| [`14_concepts_explained.md`](14_concepts_explained.md) | **plain-language explanation** of the static trip count and exact tet-box registration, with what is and is not proven |

---

## Open items

1. **Throughput**: the derived bound costs ~1.7× per step on D2. Fully mapped
   in [`13_throughput_roadmap.md`](13_throughput_roadmap.md): splitting cells
   and dynamic trip counts are rejected by measurement, exact tet-box
   registration is viable at a measured 1.41×, and the **level loop is the
   larger untouched prize** (potential ~4×, unmeasured — walks all 8 levels
   every query while 98.2% of cells are at one level).
2. **`RK4_CARRY_LAST_VALID_VEL`** is off by default; enable it for production.
3. **LUMI** has not been updated with any of this.
