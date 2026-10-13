# The production L2 path truncates its per-cell element loop at 32

**Status: ROOT CAUSE, confirmed end to end and fixed.** A matched A/B on the
real case takes the D2 loss from 2.93% to 0.20% — 93.3% of the loss recovered
— at 1.77x the per-step cost (§6).

---

## 1. The finding

`run_tracking.py` builds its RK4 step through `create_rk4_comparison()`. With
`OCTREE_REGISTRATION_METHOD = "parent_cube"` — the production setting — its L2
search dispatches to:

```python
# benchmark_femuss_comparison.py, _search_l2_3x3x3
elif config.OCTREE_REGISTRATION_METHOD == "parent_cube":
    elem_id, _ = search_mesh_aligned_octree_static_where(
        pos, mesh_aligned_octree,
        max_elems_per_cell=config.MAX_ELEMS_PER_CELL)      # 32
```

`search_mesh_aligned_octree_static_where` runs
`fori_loop(0, max_elems_per_cell)` — a **static Python bound**, by design, so
XLA can unroll it. The inner guard `elem_offset < n_elems_in_cell` prevents
reading past the end of a short cell's list; it does **not** extend the loop
for a long one. An element sitting at index 32 or beyond in its cell is never
tested.

| | |
|---|---|
| `config.MAX_ELEMS_PER_CELL` | **32** |
| D2 octree max occupancy | **79** |

## 2. Why this splits A/B/C from D so cleanly

Measured max occupancy, from each case's own build log:

| case | elem/cell | max occupancy | against the bound of 32 |
|---|---:|---:|---|
| A1 | 6.0 | **31** | under — never truncates |
| C1 | 6.1 | **30** | under — never truncates |
| D1 | 13.7 | **115** | **over** |
| D2 | 14.2 | **79** | **over** |

A/B/C sit just *below* 32 and D well above it. That is the sharpest available
explanation of why three families are unaffected and the fourth is not, and it
is consistent with every other measurement:

- **93.1% of failing hosts are non-Kuhn.** AABB registration files each
  non-Kuhn element into ~8 cells (median 8, max 27), which is exactly what
  pushes a cell's occupancy past 32.
- **Index delta median 0, 100% within the 3×3×3.** The element is in the right
  cell; the loop never reaches it.
- **The predicate accepts 234/234** with min-barycentric median 3.9e-02. The
  test would pass if it were run.
- **All four RK4 stages fail together with stride exactly 0.** Every stage
  calls the same truncated path at the same point.
- **98.7% of particles fail on every logged step.** The truncation is
  deterministic, so a particle in a crowded cell fails forever.

## 3. A correction to this investigation's record

Earlier notes state that `MAX_ELEMS_PER_CELL` truncation had been *refuted*,
on the grounds that `max_tests` appears only in the signature and docstring of
`search_mesh_aligned_octree_multi_local_where` and never truncates.

That is true of `multi_local_where` — and irrelevant, because the production
`parent_cube` path does not call it. It calls `static_where`, whose bound is
real. The refutation checked the wrong function.

It is also why every standalone replay in this investigation kept succeeding at
94–95% while the fused kernel found 7.1%: the replays called
`multi_local_where` (dynamic bound 600), not the function production uses.

## 4. The fix, and its cost

Raising `MAX_ELEMS_PER_CELL` to exceed the mesh's actual max occupancy. The
cost is bounded and predictable: the static loop lengthens for *every* visited
cell, so per-query work scales with the bound, not with the occupancy actually
present. On D2 that is 32 → 128, a 4× inner-loop bound.

Cheaper alternatives worth measuring against it:

1. **Derive the bound from the built octree** rather than a fixed constant:
   `max_elems_per_cell = int(cells.max_elements_per_cell)`. Correct by
   construction on any mesh, and no larger than necessary — 79 on D2 rather
   than 128.
2. **Dispatch to `multi_local_where`** on this path, which is dynamically
   bounded and already what the non-`parent_cube` branch uses.
3. **Reduce occupancy at build time**, e.g. splitting over-full cells to a
   finer level, which also lowers the mean query cost.

Option 1 is the smallest correct change and is self-adjusting; option 2 trades
the unrolled loop for a dynamic one. Both need a throughput measurement before
being adopted, since this path is the hot loop of every run.

## 5. Confirmation: the isolated search

Occupancy of the D2 octree, and the same 600 known-failing freeze positions
searched at three bounds:

```
CELL OCCUPANCY: max 79
  cells with >32 elements       : 804,653 of 5,982,836  (13.45%)
  element entries at index >=32 : 12,661,126            <- never tested

static_where, max_elems_per_cell=32  (production) :  55 / 600  ( 9.2%)
static_where, max_elems_per_cell=128              : 563 / 600  (93.8%)
multi_local_where (dynamically bounded)           : 563 / 600  (93.8%)
  recovered by raising the bound alone            : 508
```

Bound 128 and the dynamic variant agree exactly, so the bound accounts for the
entire gap and nothing else is missing from this path.

## 6. Confirmation: the real case, matched A/B

D2, 80,000 particles seeded in the loss band, 600 steps, `last_valid_vel`,
identical seeds in both arms, the bound as the only difference:

| | bound 32 (historical) | **bound 79 (derived)** |
|---|---:|---:|
| final lost | 2,341 (**2.93%**) | **157 (0.20%)** |
| throughput | 2.3 step/s | 1.3 step/s |
| exit | 0 | 0 |

**93.3% of the loss recovered**, at **1.77× the per-step cost**. The cost is
sublinear in the bound — a 2.5× longer loop costs 1.77× — because most cells
hold ~14 elements and the inner guard short-circuits the tail for the
majority.

Loss at matched intermediate steps, showing the effect is immediate rather
than a late divergence:

| step | bound 32 | bound 79 | reduction |
|---:|---:|---:|---:|
| 100 | 1,262 | 19 | 98.5% |
| 200 | 1,577 | 52 | 96.7% |
| 600 | 2,341 | 157 | 93.3% |

The baseline arm reproduced the earlier trace run exactly (`lost=2,165` at
step 500 in both), confirming the defect is fully deterministic.

Extrapolated to the full D2 production configuration (288,000 particles):
**~11,724 lost → ~565**.

The residual 0.20% is near the measured floor of particles that are genuinely
outside the mesh: of the freeze positions, 6.5% had no containing element at
any tolerance, and a further ~10.5% sat above the free surface.

## 7. A corroborating detail

Initial assignment uses `multi_local_where` (dynamically bounded) while the
per-step L2 uses `static_where` (truncating). That is why every run in this
investigation assigned **100% of particles at step 0** and then lost them once
stepping began — the seeding path never truncates and the stepping path does.
That observation was recorded as reassuring earlier in the investigation
before its significance was understood.
