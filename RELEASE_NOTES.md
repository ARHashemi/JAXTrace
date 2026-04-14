# JAXTrace — Release Branch

This is the `release/stable` branch, a lean snapshot of JAXTrace meant for
production use on LUMI and other HPC systems. Development notes, large
reference PDFs, historical logs, archived experiments, and one-off debug
scripts have been removed — they are preserved on `feature/lumi-benchmark`
and other development branches on GitHub.

## Contents

- Source library: `jaxtrace/`
- Production driver: `run_tracking.py`
- Benchmark + diagnostic scripts: `benchmark_femuss_comparison.py`,
  `benchmark_l2_accuracy.py`, `diagnose_femuss_deviation.py`
- LUMI SLURM submission: `scripts/run_lumi.sh`
- Tests: `tests/`
- Examples: `examples/`

## Using on LUMI

See the top-level `README.md` and `scripts/run_lumi.sh`.

The recommended workflow for colleagues is:
1. `cd /project/.../JAXTrace_stable`
2. Copy `scripts/run_lumi.sh` to your own working directory.
3. Edit the USER CONFIGURATION block at the top (case folder, seeding,
   n_steps, ...).
4. `sbatch your_run_lumi.sh`

Do **not** edit files inside `JAXTrace_stable/` — the folder is updated
periodically by bumping the `stable-YYYY-MM-DD` tag.

## Release tags

- `stable-2026-04-14` — initial lean release snapshot.
