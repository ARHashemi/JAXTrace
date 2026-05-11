# JAXTrace — Release Notes

## stable-2026-05-11

### Highlights since `stable-2026-04-14`

* **New particle-seeding modes** in `run_tracking.py` (via
  `--seed-source`):
  - `grid` — uniform grid in an absolute `--seed-box`; particle count =
    `Nx*Ny*Nz` from `--seed-grid`.
  - `box-frac` — uniform random in a per-axis fractional sub-box of the
    mesh bounding box (`--seed-fraction XLO XHI YLO YHI ZLO ZHI`).
  - `grid-frac` — uniform grid in a fractional sub-box.

  The legacy `box` / `femuss` / `file` modes are unchanged. The shell
  wrappers `scripts/run_lumi.sh` and the new
  `scripts/run_workstation.sh` expose the same six modes via
  `SEED_SOURCE`, `SEED_BOX`, `SEED_FRACTION`, `SEED_GRID` and
  `N_PARTICLES`.

* **New shared helper** `jaxtrace.tracking.seeding.bounds_from_fractions()`
  for converting per-axis domain fractions to absolute `(2, 3)` bounds.

* **Production tracking driver improvements** (`run_tracking.py`,
  `scripts/run_lumi.sh`):
  - `--mesh-dir` and `--femuss-dir` overrides for non-standard data
    layouts.
  - Named scratch output folder
    (`<case>_jaxtrace_<jobid>`, configurable via `SCRATCH_BASE`,
    `SCRATCH_FOLDER`).
  - SLURM logs and monitor log now collected into the run output
    folder at job end.

* **Robustness**: `mesh_loader` now fails fast with a clear error when
  a PVTU file is missing instead of segfaulting downstream.

* **LUMI script**: trimmed `find` / `du` calls from the per-step
  monitor (which dominated I/O); added a benchmark mode.

* **Auto-derive FEMUSS case patterns** from the input folder, removing
  hard-coded case names.

### Branch structure

* `release/stable` — this clean release (no dev notes, no analysis
  scripts, no logs). Suitable for `git clone` on production machines
  (LUMI, workstations).
* `feature/lumi-benchmark` — full development branch with planning
  docs, diagnostic scripts, analysis tools, and run logs.

### Cleanup applied to this release branch

Removed from the source tree (still preserved on
`feature/lumi-benchmark`):
- Top-level development notes (`*.md` except `README.md`)
- Top-level reference PDFs, sidecar `*.meta.json`, notebooks, `.ods`
- `logs/`, `docs/`, `archive/`, `OLD/`
- Top-level one-off scripts: `test_*`, `debug_*`, `analyze_*`,
  `check_*`, `verify_*`, `visualize_*`, `investigate_*`, `morton_*`,
  `simple_*`, `detailed_*`, `production_tracking_*`,
  `example_workflow*`, `config_example.py`,
  `benchmark_l2_search_methods*`, `benchmark_point_in_tet_*`,
  `benchmark_seeding_*`, `benchmark_rk4_diagnostic*`,
  `mesh_aligned_*_visualize`, `morton_*_visualize`, `new_l2_*`,
  `search_methods_*`, `compare_block_vs_multicell.py`
- All `diagnose_*.py` at top level **except**
  `diagnose_femuss_deviation.py` (kept for runtime FEMUSS validation).

Kept: `jaxtrace/`, `scripts/`, `tests/`, `examples/`, `config/`,
`utils/`, `run_tracking.py`, `run.py`, `benchmark_femuss_comparison.py`,
`benchmark_l2_accuracy.py`, `diagnose_femuss_deviation.py`,
`fix_merge_duplicate_nodes.py`, `RUN_PHASE3_TESTS.sh`,
`monitor_integration_test.sh`, `run_all_benchmarks.sh`,
`test_production_2timesteps.sh`, plus `README.md`, `LICENSE`,
`requirements.txt`, `pyproject.toml`, `.gitignore`.

---

## stable-2026-04-14

Initial lean release snapshot — development artifacts stripped from
`feature/lumi-benchmark`.
