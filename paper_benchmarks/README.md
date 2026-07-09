# Section 6 (Validation) re-run harness

Purpose: reproduce every numeric shipped in
`sec6_validation.tex` on the paper's benchmark mesh
(`/flash/users/ali/data/cylA.gid/`), under one consistent
aggregation protocol, so the two documented inconsistencies
(`sec6_inconsistencies.md`, Issues 1 and 2) can be resolved
and the LaTeX text updated to match.

## What each file does

* **`sec6_rerun.sh`** — bash driver. Fixes the protocol
  (`N_p = 10 000`, warm-up 3, timed 7, all six σ levels, all
  three registration strategies, full scalability sweep,
  `float64`) and runs
  `benchmark_l2_accuracy.py` with those settings. Writes
  the raw stdout to `sec6_raw.log` and invokes the
  post-processor.
* **`sec6_postprocess.py`** — Python parser + report
  generator. Reads `sec6_raw.log`, extracts every table's
  measured values, applies the canonical aggregation rule,
  cross-references against the paper values (hard-coded from
  `sec6_validation.tex`), and writes:
    * `sec6_numbers.json` — full machine-readable dump of
      protocol, measured values, paper references, and
      per-table diffs.
    * `sec6_report.md` — human-readable per-table PASS /
      CHANGED report.

## Canonical protocol (single rule across every table)

| Quantity | Rule |
|---|---|
| Wall time | Reported as `min–max` range across the 7 timed runs (matches the paper format). |
| Queries/s | `N_p / mean(times)` — **one** rule, applied to every timing table. |
| Failure decomposition | See "Issue 1" below. |
| Batch size | `N_p = 10 000` for main tables; scalability sweep uses `{1 k, 2 k, 5 k, 10 k, 20 k, 50 k, 100 k, 200 k, 500 k}`. |
| Warm-up / timed runs | 3 / 7 (paper protocol). |
| Precision | IEEE-754 double (`--float64`). |
| Random seed | 42. |

## What causes the two documented inconsistencies

### Issue 1 — Failure-decomposition sum of 20 000

**Root cause**: The failure analyser in `benchmark_l2_accuracy.py`
(`1x1x1 FAILURE ANALYSIS` block) increments the offset histogram
**once per matching (level, offset) hit** across the 3×3×3
neighbour stencil scanned at every active octree level. A single
failed particle can therefore contribute multiple hits (up to
26 × N_active_levels). The "20 000" total in
`tab:failure_decomposition` is thus a **hit-count total**, not a
batch size.

**What the re-run does**:
1. Extracts the raw hit histogram (unchanged from the current
   harness) into `sec6_numbers.json` under
   `measured.failure_decomposition_raw`.
2. Records the intended per-particle interpretation as a note in
   `sec6_report.md`.

**Recommended paper edit** (either is acceptable):
* Add one sentence to the table caption: "*Counts are total
  (level, offset) neighbour-cell matches, aggregated across all
  ~10 000 failed queries and all active octree levels.*" —
  keeps the current numbers.
* OR patch the failure analyser to classify each failed particle
  by its **smallest** recovering offset (face < edge < corner)
  and re-run so the counts sum to `N_missed` instead of ~20 000.
  If you want this variant, tell me and I'll add a
  `--per-particle-decomp` flag to the harness.

### Issue 2 — 5 264 q/s vs 5 457 q/s for the same configuration

**Root cause**: `benchmark_l2_accuracy.py` **always** computes
`queries_per_sec = N_p / mean(times)`. The paper's Table
`tab:timing` shows `5 264 = 10 000 / 1.900` (i.e. `N_p / max`),
while Table `tab:scalability` shows `5 457 ≈ 10 000 / 1.832`
(i.e. `N_p / mean`). Someone manually recomputed Table `tab:timing`'s
q/s off the worst-case wall time when writing the LaTeX.

**What the re-run does**: adopts one canonical rule
(`N_p / mean(times)`) everywhere. Every timing table's q/s in
`sec6_report.md` and `sec6_numbers.json` is computed the same
way. The paper's Tables `tab:timing` and `tab:scalability` should
be updated in the LaTeX to show the mean-based q/s in both.

## How to run

On the workstation (RTX 5090 with the JAXTrace venv):

```bash
cd /flash/shared/jax/JAXTrace       # or wherever this repo is checked out
bash paper_benchmarks/sec6_rerun.sh
```

Or override the mesh directory:

```bash
JAXTRACE_MESH_DIR=/some/other/path/post bash paper_benchmarks/sec6_rerun.sh
```

Expected wall time: **20–40 minutes** on an RTX 5090 depending
on JIT-cache warmth (fresh compile is closer to 40 min; a warm
cache is closer to 20 min).

## What to look at when the re-run finishes

1. **`sec6_report.md`** — Markdown diff, table-by-table, of paper
   vs measured. Each table is flagged `PASS` (within tolerance) or
   `CHANGED` (needs paper-side edit). Tolerances: 2 % relative or
   0.5 unit absolute by default; timing rows use 3 % / 30 ms;
   scalability rows 5 % / 30 ms.
2. **`sec6_numbers.json`** — full machine-readable payload.
   Structure:
    * `protocol` — the fixed protocol used
    * `measured.{tab_name}` — parsed numbers per table
    * `paper_reference.{tab_name}` — the shipped LaTeX values
    * `diffs` — per-table pass/change flags
3. **`sec6_raw.log`** — verbatim stdout from `benchmark_l2_accuracy.py`.

## Which paper tables are covered

All ten Section 6 tables:

* `tab:mesh_properties` — mesh + registration statistics
* `tab:memory` — GPU memory footprint
* `tab:taxonomy` — method list (no numbers to check)
* `tab:found_rate` — found rate under perturbation
* `tab:search_failures` — in-bbox misses
* `tab:intra_found` — intra-element found rate
* `tab:timing` — computational performance @ σ = 0
* `tab:bytes` — bytes per query from HLO cost analysis
* `tab:level_dist` — resolving refinement level
* `tab:scalability` — throughput vs. batch size
* `tab:failure_decomposition` — 1×1×1 miss breakdown

The Discussion section prose (§6.5.2–6.5.4) also quotes several
numbers derived from these tables (e.g. `8.1 × PIT increase`,
`~50 % found rate`, `85 % resolve at level 7`, `1.55 × more
traffic`); these numbers are **computed from** the table entries
above so they are validated automatically by verifying the
underlying tables.

## Feedback loop for future changes

If the tolerance defaults trip on legitimate hardware drift (e.g.
CUDA driver update changing timing by a few percent), edit the
`close_enough` calls near the top of `sec6_postprocess.py` and
re-run just the post-processor against the same log:

```bash
python paper_benchmarks/sec6_postprocess.py \
    --log paper_benchmarks/sec6_raw.log \
    --json paper_benchmarks/sec6_numbers.json \
    --report paper_benchmarks/sec6_report.md
```

No re-run of the harness itself is needed for tolerance tweaks.
