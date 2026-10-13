# The PinShapes mesh families

**What this is.** A reference description of the 20 FSW pin-geometry cases used
for particle tracking: what each family is, how its mesh is structured, which
velocity snapshots each case loads, and whether its geometry moves during a
run. Written to be read on its own — it assumes no prior context from the
D-family investigation.

**Why it exists.** Two questions kept being re-derived from scratch: *"does the
mesh move as the tool rotates?"* (it does not, and that licenses building the
search structure once) and *"what actually makes the D family different?"* (its
tetrahedra, not its refinement). Both are answered here with measurements.

Last updated 2026-10-13.

---

## 1. The four families

All 20 cases model the same friction-stir-welding process and differ only in
the geometry of the tool pin. Each lives in its own GiD case folder,
`<FAMILY>/<CASE>.gid/`, with the mesh and velocity snapshots under `post/`.

| family | varies | cases | n |
|---|---|---|---:|
| **A-FlatsVariations** | number/size of flats milled on the pin | A1, A2, A2old, A3, A3old, A4, A4old | 7 |
| **B-FluteVariations** | flute geometry | B1, B2, B3, B4 | 4 |
| **C-ThreadsVariations** | thread pitch/profile | C1, C2, C3, C4 | 4 |
| **D-ConcavityTilt** | shoulder concavity + tool tilt | D1, D2, D2old, D3, D4 | 5 |

The `*old` cases are superseded variants kept for comparison; they are real
cases with their own meshes, not backups, and the survey covers them.

Mesh scale is comparable across all families: roughly 2.8M nodes and 10.8M
tetrahedra for D2, assembled from 64 parallel pieces of ~13k–23k points each.
These are production meshes, not test fixtures.

---

## 2. The mesh does not move — all 20 cases

**This is the single most useful fact in this document.** The simulation is
**Eulerian**: the mesh is a fixed background grid and the tool rotates
*through* it. Only the velocity field and the level-set field change from
timestep to timestep. The node coordinates never move.

**Certified over every loaded timestep of every case** — LUMI job 22787422,
13 Oct 2026, 22 min, 3,354 piece-reads. This supersedes the earlier 5-sample
screen (job 22775342), which could not see motion confined to one piece at one
step (see [Method](#method)):

| | |
|---|---|
| cases checked | **20** |
| static | **20** |
| moving | **0** |
| inconclusive | **0** |
| timesteps compared | **every loaded step of every case** |
| pieces covered | **128 per case**, index rotating per step |

### Consequence

**One search structure is valid for an entire run.** The octree grid, the
Morton codes *and* the element registration can all be built once, before the
time loop starts. Nothing needs to be rebuilt or re-registered per timestep or
per revolution. Any future design that assumes per-step rebuilding is paying
for something this measurement rules out.

### Per-case results

Each row gives the velocity range read from that case's **own**
`run_jaxtrace.sh`, the size of one representative mesh piece, and how many
timesteps were compared.

| family | case | velocity steps | n steps | comparisons | verdict |
|---|---|---|---:|---:|---|
| A-FlatsVariations | A1 | 34..199 | 166 | 38 | STATIC |
| A-FlatsVariations | A2 | 34..199 | 166 | 38 | STATIC |
| A-FlatsVariations | A2old | 34..199 | 166 | 38 | STATIC |
| A-FlatsVariations | A3 | 34..199 | 166 | 38 | STATIC |
| A-FlatsVariations | A3old | 34..199 | 166 | 38 | STATIC |
| A-FlatsVariations | A4 | 34..199 | 166 | 38 | STATIC |
| A-FlatsVariations | A4old | 34..199 | 166 | 38 | STATIC |
| B-FluteVariations | B1 | 34..199 | 166 | 38 | STATIC |
| B-FluteVariations | **B2** | **2..201** | **200** | 72 | STATIC |
| B-FluteVariations | B3 | 34..199 | 166 | 38 | STATIC |
| B-FluteVariations | B4 | 34..199 | 166 | 38 | STATIC |
| C-ThreadsVariations | C1 | 34..199 | 166 | 38 | STATIC |
| C-ThreadsVariations | C2 | 34..199 | 166 | 38 | STATIC |
| C-ThreadsVariations | C3 | 34..199 | 166 | 38 | STATIC |
| C-ThreadsVariations | C4 | 34..199 | 166 | 38 | STATIC |
| D-ConcavityTilt | D1 | 34..199 | 166 | 38 | STATIC |
| D-ConcavityTilt | D2 | 34..199 | 166 | 38 | STATIC |
| D-ConcavityTilt | D2old | 34..199 | 166 | 38 | STATIC |
| D-ConcavityTilt | D3 | 34..199 | 166 | 38 | STATIC |
| D-ConcavityTilt | D4 | 34..199 | 166 | 38 | STATIC |

Every case reads all 128 pieces across its run, with the piece index advancing
per timestep, and every comparison agreed byte-for-byte. **A1 additionally has
a fully exhaustive result** from job 22776651 before it was cancelled: 21,120
comparisons over all 166 steps and all 128 pieces, 0 differences.

### B2 is the one exception to the step set

**19 of 20 cases load velocity steps 34..199 (166 snapshots). B2 loads
2..201 (200 snapshots).** Always read the range from the case's own
`run_jaxtrace.sh` rather than assuming 34..199 — and remember that snapshot
loading, not GPU work, sets the wall clock on LUMI, so B2 costs ~20% more time
in the loading phase than its siblings.

### Method

For each case the survey reads `VEL_START` and `VEL_END` from the case's own
`run_jaxtrace.sh`, then hashes the `Points` array of a mesh piece at every
timestep in that range, advancing the piece index by one each step. Identical
MD5 hashes for the same piece across time mean the coordinates are
byte-identical. A case is reported STATIC only if some piece was actually read
at two or more timesteps, so an unreadable path cannot masquerade as astatic result.

**Why not every piece at every step.** That mode exists
(`--all-steps --pieces=0`) and was tried first (job 22776651), but LUMI
`/scratch` delivered only 1.77 piece-reads/s, i.e. ~3.3 h per case: a 12 h job
would have certified ~4 of 20 cases, and since the survey walks families
alphabetically all four would have been A-family, leaving the D family — the
one under investigation — unmeasured. The rotating variant covers all 20 cases
in 3,354 reads and 22 minutes.

**What rotating still catches.** Motion confined to one piece at one step. This
was verified rather than assumed: on a synthetic case tree whose mesh displaces
a single node in piece 3 at one timestep, the old 5-sample screen reports
`STATIC (4 comparisons)` — a false negative — while this mode reports
`MOVES at ts17/piece3`.

**What it does not claim.** That every piece was checked at every step. A mesh
moving only in a piece this run never samples at that step would be missed, but
it would have to evade a rotating index across 166 timesteps. For A1 that gap
is closed outright by the exhaustive job's 21,120 comparisons.

Scripts: [`scripts/dfamily/mesh_survey.py`](../../scripts/dfamily/mesh_survey.py),
submitted with
[`sbatch_mesh_survey_rotating.sh`](../../scripts/dfamily/sbatch_mesh_survey_rotating.sh)
(the certification above) or
[`sbatch_mesh_survey_allsteps.sh`](../../scripts/dfamily/sbatch_mesh_survey_allsteps.sh)
(every piece, for one family at a time). Both are read-only — no writes to any
case folder.

### The bug that made the first attempt lie, and the guard against it

**The first submission (job 22775201) reported the same "ALL cases static"
conclusion while reading zero data** — every row showed `piece 0p/0c`. It
derived each piece's filename by string arithmetic on the parent `.pvtu` name:

```python
prefix = name[:-(len(str(vs)) + 6)]      # drops one character too many
```

which lost the trailing underscore, and it assumed pieces are numbered from
`_1` when they actually start at `_0`. Every candidate path missed, nothing was
read, and "no differences found" was reported as STATIC.

Two changes make the current result trustworthy:

1. piece paths are resolved from each PVTU's own `<Piece Source="..."/>`
   entries instead of being reconstructed, and
2. the script **refuses to report STATIC** unless it actually read ≥ 2
   timesteps with a non-zero point count — otherwise it prints
   `NO DATA READ (cannot conclude)` or `ONLY n TIMESTEP READ`.

The per-piece point and cell counts in the table above are the evidence this
run read real geometry. **A `0p/0c` row in any future survey means the loader
missed, not that the mesh is empty.**

---

## 3. What makes the D family structurally different

Not refinement — tetrahedron shape. Two measurements separate D from A/B/C with
no overlap.

### 3.1 Non-Kuhn fraction

A **Kuhn** (Freudenthal) tetrahedron has an axis-aligned edge on every axis, so
it sits inside an identifiable cube of the background grid. A **non-Kuhn**
tetrahedron does not, so the mesh-aligned octree cannot place it by its own
parent cube and must borrow a neighbour's grid. Detected as
`np.any(cell_size == 0)`.

Measured from each case's own Pass 1 summary in the phase4 production logs:

| family | meshes | non-Kuhn fraction |
|---|---:|---|
| A | 7 | 0.74 – 1.05% |
| B | 4 | 0.89 – 1.03% |
| C | 4 | 0.91 – 1.00% |
| **D** | 5 | **50.55 – 59.32%** |

A/B/C average 0.92%; D averages 55.18% — **60× higher, with no overlap between
the groups.**

### 3.2 Cell anisotropy

A/B/C cells are **cubic**: every octree level holds exactly one cell size. D
cells are **cuboid**, taller in z than wide in x/y, with a constant aspect
ratio of 1.1972 — and crucially their sizes form a *continuum* rather than a
few discrete families.

| mesh | distinct size-triples per level | per-cell aspect ratio |
|---|---|---|
| A1 | **1 at every level** | 1.000 exactly |
| D2 | 20 → 345 per level | median **1.197**, max 1.457 |

### 3.3 Cost consequence in the octree

Against the builder's stated target of ~5–6 elements per cell, max ~8:

| case | cells | elem/cell | max occupancy |
|---|---:|---:|---:|
| A1 | 1,221,369 | 6.0 | 31 |
| C1 | 1,324,524 | 6.1 | 30 |
| D1 | 5,649,185 | 13.7 | 115 |
| D2 | 5,982,836 | 14.2 | 79 |

D carries ~2.3× the mean load, which is a real **throughput** cost (D2 initial
assignment 27.6 s vs A1 11.8 s, and ~0.5 vs ~1.1 tracking steps/s). It is a
cost effect only — occupancy alone does not lose particles.

### 3.4 A high non-Kuhn fraction is not itself a failure mode

Worth stating plainly, because it is easy to misread §3.1 as the bug. The MALMO
paper's revised benchmark cohort includes meshes that are **100% non-Kuhn** and
are located perfectly (`aabb` registration: 100% on all 10 meshes). What caused
the D-family particle losses was non-Kuhn geometry *combined with* a
registration lattice that disagreed with the lookup lattice — a defect since
fixed. See
[`docs/dfamily/07_root_cause.md`](../dfamily/07_root_cause.md) for that account.

---

## 4. Practical notes when setting up a run

- **Read the velocity range from the case's own `run_jaxtrace.sh`.** 19 cases
  use 34..199; B2 uses 2..201 (§2).
- **Build the search structure once.** The mesh is static in every case (§2).
- **`MESH_SUBDIR` must be empty for these cases.** The PVTUs sit directly in
  `post/`. A `MESH_SUBDIR="0eule"` left over from the cylA templates silently
  breaks the mesh path.
- **Snapshot loading dominates the wall clock on LUMI**, not GPU work, so trim
  `VEL_END` together with `N_STEPS` when sizing a short diagnostic run.
- **Seeding with `box-frac` over-seeds the D family.** Those fractions are
  taken of the mesh *bounding box*, and the D bbox extends above the workpiece
  (D1: z −6.00…+1.35 mm vs A1: −6.00…0.00), so ~18% of seeds land in empty air
  and fail initial assignment. Clamp the upper z fraction per case, or seed from
  the FEMUSS node set.
- **Case folders are shared.** Write results elsewhere
  (`OUTPUT_TARGET=scratch`), not into `<CASE>.gid/`.

---

## 5. Related documents

| document | covers |
|---|---|
| [`dfamily/07_root_cause.md`](../dfamily/07_root_cause.md) | why D-family particles were lost, and the fix |
| [`dfamily/08_mesh_is_static_all_cases.md`](../dfamily/08_mesh_is_static_all_cases.md) | the static-mesh survey as it bears on the octree build |
| [`dfamily/06_pending_actions.md`](../dfamily/06_pending_actions.md) | settings to apply when preparing full runs |
| [`THREADEDA_MESH_ANALYSIS.md`](../THREADEDA_MESH_ANALYSIS.md) | a different mesh (ThreadedA), unrelated to PinShapes |
