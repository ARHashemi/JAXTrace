# D-family diagnostic tooling

Two scripts for the "particles stop near the tool in D-ConcavityTilt" issue.
Background and measurements: `docs/dfamily/01_diagnosis.md`.

## The question they answer

When a particle stops moving, is it because

- the **search** could not find its host element (`ElementID < 0`), or
- the search worked and the **velocity it was given was zero**
  (`ElementID >= 0`) — e.g. the level set marked it as inside the tool?

The two need opposite fixes, so this must be settled before tuning anything.

## 1. `make_diag_run.sh` — prepare the run

```bash
scripts/dfamily/make_diag_run.sh \
    /scratch/project_465002752/lorenzgl/Cases/PinShapes/D-ConcavityTilt/D2.gid \
    1500 10
```

Copies the case's own `run_jaxtrace.sh` into
`/scratch/<proj>/<user>/dfamily/<case>_diag/` and changes exactly eight
things: `EXPORT_ELEMENT_IDS=1`, `OUTPUT_TARGET=scratch`, `N_STEPS`,
`EXPORT_FREQ`, `AUTO_DETECT_CASE=0`, `INPUT`, `RUN_TAG`, plus a shorter
SLURM time limit and job name.

**The colleague's case folder is never written to.** Review with the `diff`
the script prints, then `sbatch run_jaxtrace_diag.sh`.

1500 steps is enough: in the existing D2 output 18% of particles are frozen
before step 100 and the late-freeze population is well established by ~1000.

## 2. `analyze_frozen_particles.py` — read the result

```bash
python scripts/dfamily/analyze_frozen_particles.py <run_dir> \
    --level-set <case>.gid/post/<MESH>_34.pvtu \
    --out docs/dfamily/02_D2_frozen_report.md
```

Reports, as a markdown table:

1. how many particles are dead-on-arrival vs freeze later,
2. **for each group, the ElementID split** — the decisive test,
3. where they stop, by depth and radius, optionally against the tool region
   read from the `LEVEL` field.

Runs without `--level-set`, and without ElementID in the files (it then says
so rather than guessing). Needs VTK, so on LUMI run it inside the same
singularity image used for tracking.

### Reading the verdict

- **>70% of late freezers have `ElementID < 0`** → search failure. Act on
  `ENHANCED_SEARCH_BAND`, `L0_SKIP_BAND`, `L2_NEIGHBORHOOD`.
- **<30%** → the host is found and the velocity is zero. Search settings will
  not help; look at `LEVELSET_MODE` and the velocity field.
- **in between** → both are active; use the depth/radius table to separate
  them (in D2 the shoulder region and the deep pin region behave differently).

## Note on mesh file prefixes

The mesh PVTU prefix is **not** always the case name: `D2.gid/post` contains
`A2_*.vtu`, and `D1.gid/post` contains `A1_*.vtu`. Check with `ls` before
passing `--level-set`.

## Note on run length vs mesh loading cost

Wall clock for these diagnostic runs is dominated by **mesh loading, not
tracking**. Topology is read once, but the velocity field is re-read from
every snapshot in `[VEL_START, VEL_END]`. For D2 that is 189 GB over 166
snapshots — roughly **1.4 min per snapshot**, so loading the full range costs
about **4 hours** before a single RK4 step runs.

That is wasted work for a short diagnostic. The velocity sequence is
**cyclic**:

```
cycle_period = n_snapshots x velocity_dt
```

so a contiguous subset starting at `VEL_START` is still physically
meaningful — the flow just repeats sooner. With `DT=0.0003629` and
`velocity_dt ~ 0.0175 s`:

| snapshots | load time | cycles seen by 1500 steps |
|---:|---:|---:|
| 166 (full) | ~232 min | 0.19 |
| 40 | ~56 min | 0.78 |
| **30 (default)** | **~42 min** | **1.04** |
| 20 | ~28 min | 1.56 |

30 is the default: particles see one complete flow cycle, and the job fits
comfortably in its time limit. Override with `--vel N` (`--vel 0` keeps the
full range).

`make_diag_run.sh` scales `#SBATCH --time` from the snapshot count, so the
time limit follows the subset automatically. **Do not shorten `N_STEPS`
without also checking `--vel`:** a short run with the full velocity range is
the worst of both worlds — hours of loading for a fraction of one cycle.
This is exactly how the first D2 attempt (job 22613298) hit its 1h30 limit
with zero VTUs written.

## Note on which JAXTrace the job runs (important)

The per-case `run_jaxtrace.sh` sets:

```
JAXTRACE="/projappl/${PROJECT}/hashemia/JAXTrace_stable"
```

but the production array (`sbatch_phase4_rk4_array.sh`) used:

```
REPO="${REPO:-/projappl/${PROJ_ID}/${USERDIR}/JAXTrace}"     # the dev repo
```

**These are different code.** `JAXTrace_stable` predates the octree
`orphan_fallback` fix. In stable, `run_tracking.py` calls
`extract_octree_cells_parent_cube(...)` without an `orphan_fallback` argument
(the parameter does not exist there), so `parent_cube.py` takes an
unconditional drop path:

```python
if nbr_id < 0:
    if verbose: print("... no Kuhn neighbour, skipped")
    continue          # element registered in NO octree cell
```

On D2 that drops **over 1.3 million elements** and floods the log with ~2.7M
warning lines (280 MB). An element in no octree cell can never be returned by
L2 search, so particles whose host is such an element are frozen with
`ElementID = -1` **as an artefact of the build**, not of the physics.

The dev repo registers all of them (`orphan_fallback=True` borrows the median
Kuhn cell) and prints nothing. Production D2 confirms this:

```
Non-Kuhn registered: 5,463,331 / 5,463,331
Non-Kuhn AABB-fallback (orphan): 4,680,600
```

`make_diag_run.sh` therefore **pins `JAXTRACE` to the dev repo** to match
production. Override with `JAXTRACE_REPO=... scripts/dfamily/make_diag_run.sh`.
If a diagnostic log contains `no Kuhn neighbour, skipped`, the run is on the
wrong build and its `ElementID<0` counts are not comparable to production.

## Note on the production velocity range

Production used `--vel-range "$IDXSTEP" "$IDXSTEP"` — a **single** velocity
snapshot (steady field), not a cycled sequence. So `--vel 1` reproduces
production most faithfully *and* costs almost nothing to load. The `--vel 30`
default above is for runs that deliberately want a time-varying field; it is
not what production did.

## The registration sweep (`registration_sweep.sh`)

The question this answers: is the D-family host loss caused by the SEARCH, or
by the octree REGISTRATION it is handed?

The revised paper already measures registration quality. `paper_table_found.md`
(Table T2) reports location correctness over the 10-mesh R2-6 cohort, which
**includes fully non-Kuhn meshes** (Bunny, Porous media and Microfluidics are
100% non-Kuhn; FP-8.7k is 93.8%):

| registration | worst correct% across the cohort |
|---|---:|
| `aabb` | **100.00%** |
| `centroid` (= parent_cube) | 99.64% |
| `vertex_multi` | 96.12% |

So `aabb` is the only coverage-complete option, and `centroid` — which is what
the D-family tracking runs use — already loses queries in the paper's own data.
The D meshes are 50.6–59.3% non-Kuhn, against 0.06% for the FSW mesh, so they
sit squarely in the regime where this choice matters.

`aabb` was benchmark-only until now; `run_tracking.py --registration aabb` was
added so tracking can use it.

```bash
# the decisive pair, which is also the cheapest thing to try first
scripts/dfamily/registration_sweep.sh D2 --only aabb,parent_cube

# isolate each fix in the parent_cube path
scripts/dfamily/registration_sweep.sh D2 --only pc-nohybrid,pc-noorphan

# compare whatever has finished
scripts/dfamily/registration_sweep.sh D2 --compare
```

**Reading it:** `aabb` ≈ 0 lost while `parent_cube` loses thousands means
registration coverage is the root cause, the search kernel is fine, and the
band settings are *not* the lever — widening a search cannot find an element
absent from the cells it visits. If `aabb` still loses particles in the same
r 7–9 mm annulus, the defect is in the search.

### Why it is cheap

The losses are already localised: they happen at r 6.4–9.5 mm, z −4..0 mm, and
the particles that suffer them *start* at r 11.1–14.2 mm, needing ~670 steps to
advect in. The sweep seeds **directly in that annulus**, so losses appear within
a few tens of steps and 400 steps suffice instead of 1500. The D family needs
the real time-dependent velocity sequence, so a sequence is always loaded
(never a single snapshot); `--vel` only bounds how much of the cyclic sequence
is read.

### Intermediate logging — which stage fails

Every variant gets `--hit-stats-log`, which writes `hit_stats.csv`: at each
`LOG_INTERVAL` step (set to 25) it classifies the surviving particles by which
level would find them on a cold lookup — L0 (cached element), L1 (neighbour
hop), L2 (global octree) or miss. A rising `miss` share is the search giving
up; a high L2 share means L0/L1 are not retaining the host. `run_tracking.py`
also always writes `search_stats.csv` with per-step `n_lost` / `new_lost`.
`--compare` reports both.

## The coverage audit (`audit_octree_coverage.py`)

A cross-check that needs no tracking run at all. For each lost particle it
brute-forces the element that *really* contains its final position, then asks
whether that element is registered in any cell the 3×3×3 search would visit:

| outcome | meaning |
|---|---|
| no containing element | genuinely outside the mesh — not a search bug |
| host in a **visited** cell | **search** defect: it should have been found |
| host only in **non-visited** cells | **registration coverage** defect |
| host in **no** cell | element dropped from the octree at build time |

**Run it as a batch job, not on a login node.** The per-user cgroup on a LUMI
login node caps memory at 96 GB, and building the node-to-elements map plus the
octree for a 10.8M-element D-family mesh in NumPy exceeds that
(`MemoryError` in `build_node_to_elements`). Use the wrapper, which asks for
240 GB on a CPU-only `small` node:

```bash
sbatch scripts/dfamily/sbatch_audit_coverage.sh \
    /scratch/<proj>/<user>/dfamily/D2_results \
    /scratch/<proj>/lorenzgl/Cases/PinShapes/D-ConcavityTilt/D2.gid \
    parent_cube 150
```

Small meshes can still go through the login-node path:

```bash
scripts/dfamily/run_analysis_lumi.sh --script scripts/dfamily/audit_octree_coverage.py \
    --run <results_dir> --case <case>.gid --registration parent_cube
```

This distinguishes the four mechanisms directly, rather than inferring them
from loss counts. No GPU is needed — the audit is pure NumPy and JAX is pinned
to the CPU backend (`JAX_PLATFORMS=cpu`), which is also why
`run_analysis_lumi.sh` sets that: importing the octree extractors pulls in JAX,
which otherwise aborts with "No visible GPU devices" on a login node.

## Confirming the wrong-grid-pitch hypothesis

### What is already measured

| finding | evidence |
|---|---|
| D2 has two cell sizes a factor of 2 apart sharing one level, in 99.86% of cells | `check_level_sizes.py`, job 22623893 |
| A1 has 0.00% — only float round-off (max ratio 1.00001) | job 22624436 |
| A1 loses zero particles; D2 loses thousands | production logs |
| Registration coverage is adequate for lost particles (150/150 hosts in visited cells) | `audit_octree_coverage.py`, job 22623843 |
| `aabb` cuts the loss 8,027 → 3,540 (56%) but cannot remove it | sweep jobs 22622780/22622781 |

### Why sizes mix within a level

`level` is derived from the **mean** of a possibly anisotropic `cell_size`
([mesh_aligned_octree_single_cell.py:101-104](../../jaxtrace/gpu/search/mesh_aligned_octree_single_cell.py#L101-L104)):

```python
avg_size = np.mean(cell_size)        # cell_size is [dx, dy, dz]
level    = round(-np.log2(avg_size))
```

so cells with different per-axis sizes collapse onto one level:

| dx, dy, dz | mean | level |
|---|---:|---:|
| 4.688e-05, 4.688e-05, 4.688e-05 | 4.6875e-05 | 14 |
| 9.375e-05, 4.688e-05, 4.688e-05 | 6.2500e-05 | 14 |
| 9.375e-05, 9.375e-05, 4.688e-05 | 7.8125e-05 | 14 |

The GPU upload then collapses each level to the size of the **first** cell it
saw at that level
([mesh_aligned_octree_gpu.py:452-457](../../jaxtrace/gpu/search/mesh_aligned_octree_gpu.py#L452-L457),
"all should be nearly identical per level") and the search indexes every cell
at that level with that one pitch
([mesh_aligned_point_location.py:206-211](../../jaxtrace/gpu/search/mesh_aligned_point_location.py#L206-L211)).
A 2× pitch error puts the computed index far outside the ±1 the 3×3×3
neighbourhood covers — for a position 1.0e-03 m with a 4.6875e-05 pitch the
index is 21, against 10 for the true 2× cell, so |Δ| = 11.

### The two tests

```bash
scripts/dfamily/sbatch_confirm_tests.sh --dry-run   # inspect first
scripts/dfamily/sbatch_confirm_tests.sh             # submit all four
scripts/dfamily/sbatch_confirm_tests.sh --only index  # just the decisive one
```

**Test 1 — `why_factor2.py`** (`aniso_D2`, `aniso_A1`): per level, the distinct
`(dx,dy,dz)` triples present and the per-cell anisotropy. Confirms *why* sizes
mix, which decides which fix is appropriate. If D2's level 14 holds several
anisotropic triples while A1's holds one near-cubic triple, the mean-based
`level` formula is the cause.

**Test 2 — `test_index_mismatch.py`** (`idxmis_pc`, `idxmis_aabb`): **the
decisive test.** Per particle it brute-forces the element that really contains
the final position, then compares the index the search would use,
`floor(pos / level_cell_sizes[level])`, against the stored grid index of the
cell that element is registered in. Lost particles are compared against an
equal-sized **control sample of survivors**, so a high mismatch rate only
counts if the survivors do not share it.

Reading test 2:

| outcome | conclusion |
|---|---|
| lost mostly mismatched, survivors mostly matched | **confirmed** — wrong pitch is the mechanism |
| both mismatch at a similar rate | **refuted** — mismatch is common and harmless; look elsewhere |
| lost mostly matched | **refuted** — suspect point-in-tet tolerance or level ordering |

Both tests are CPU-only, need ~240 GB for a 10.8M-element D mesh, and must run
as batch jobs — the login-node cgroup caps per-user memory at 96 GB.

### Fix options, once confirmed

| option | accuracy | throughput | notes |
|---|---|---|---|
| **A. split mixed levels at build time** | exact | unchanged | search kernel untouched; A/B/C unaffected since they have no mixed levels |
| B. per-cell size lookup in the search | exact | slower | breaks the static-pitch assumption the unrolled loop relies on |
| C. `level` from `max(cell_size)` not the mean | exact if it separates sizes | unchanged | one-line probe, but re-buckets every mesh |
| D. search both pitches at mixed levels | exact | ~2× cells at those levels | simple, costs throughput |

Preference: **A**, with **C** as a quick probe. A fixes the root cause where it
arises and leaves the search kernel and its static unrolled loop alone, so no
mesh loses throughput. It should be **automatic with a warning**, not a user
switch: a user cannot know whether their mesh has mixed levels, and guessing
wrong silently loses particles. `benchmark_l2_accuracy.py:990` already does
detect-correct-warn for `MAX_ELEMS_PER_CELL`; `run_tracking.py` has no
equivalent gate.

## The fix: per-axis median pitch

**One functional line**, in
[mesh_aligned_octree_gpu.py](../../jaxtrace/gpu/search/mesh_aligned_octree_gpu.py):

```python
level_cell_sizes_cpu[level] = np.median(level_sizes, axis=0)   # was: level_sizes[0]
```

### It is a pre-build change only

| | touches cell sizes? |
|---|---|
| `jaxtrace/gpu/tracking/` (RK4 time marching) | **no** — grep for `cell_size` returns nothing |
| `mesh_aligned_point_location.py` (9 search sites) | reads `level_cell_sizes[level]`, already per-axis |

`level_cell_sizes` is already `(max_level+1, 3)` and every search site does
`floor(pos[a] / cell_size[a])` per axis, so **cuboid cells were already
supported**. The bug was only that the wrong 3-vector was chosen. Nothing in
the per-step path changes, so there is **no throughput cost** and no new
branch in any inner loop.

### Measured effect

| | unfindable cells (D2) | A1 |
|---|---:|---:|
| `level_sizes[0]` (old) | 57,369 (0.96%) | 0 |
| `np.median(...)` (new) | **9,266 (0.15%)** | **0** |

6.2× better on D2, and a **provable no-op on A/B/C**: A1 has one cell size per
level (100% dominant share at all 7 levels), so its median *is* its first cell.
Median, dominant-mode and cluster-median all converge on 9,266, which is why
the plain median suffices.

The 9,266 residual sits at r 4.43–7.51 mm, z −2.46…+1.03 mm — a thin shell at
the shoulder *edge*. Particles were being lost at r 2.51–6.86 mm (concentrated
2.9–5.3 mm), so the residual is **not** where the losses occur. No special
handling of the minority is warranted.

### Should it be a user switch?

**No.** A switch would mean a runtime `if` in a per-step path, and more
importantly a user cannot know whether their mesh has anisotropic cells —
guessing wrong silently loses particles. The patch instead **detects and
reports** at build time, once:

```
cell aspect ratio (max/min per level): up to 1.1972 -> ANISOTROPIC (cuboid) cells; per-axis pitch in use
WARNING: these levels are not dominated by a single cell size, ...
```

The warning fires when a level's dominant size family is under 60% (clustered
at 5% relative tolerance so mesh-motion jitter does not fragment a family).
That is the detect-correct-warn pattern `benchmark_l2_accuracy.py:990` already
uses for `MAX_ELEMS_PER_CELL`.

### Verifying it — full production-length run

```bash
scripts/dfamily/verify_pitch_fix.sh D2 --no-submit    # review the diff first
scripts/dfamily/verify_pitch_fix.sh D2                # submit
scripts/dfamily/verify_pitch_fix.sh D2 --analyze      # when it finishes
scripts/dfamily/verify_pitch_fix.sh A1 --family A     # the control
```

Built from the case's own `run_jaxtrace.sh`, keeping `N_STEPS=8000` and
`VEL_START/END=34..199` (the full time-dependent sequence) and all its physics
settings. Only four things change, each necessary:

| change | why |
|---|---|
| `JAXTRACE` → dev repo | the case script points at `JAXTrace_stable`, which predates the fix *and* drops non-Kuhn orphans |
| `OUTPUT_TARGET=scratch` | never write into the colleague's case folder |
| `EXPORT_ELEMENT_IDS=1` | without it the lost-particle count is unmeasurable |
| `+ --hit-stats-log` | free per-stage L0/L1/L2/miss breakdown |

**Prediction:** losses should fall from the 8,027-of-150,620 measured with
`parent_cube` toward the few hundred the residual explains. If they do not,
the pitch was not the whole story and the report should say so.
