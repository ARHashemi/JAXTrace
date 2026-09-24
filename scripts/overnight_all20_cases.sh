#!/bin/bash
# =============================================================================
# overnight_all20_cases.sh
#
# Thin wrapper around overnight_rom_vs_fom_sweep.sh that runs the FULL
# 20-case cohort (cylindrical_000 .. cylindrical_019), including the 16
# cases that were NOT part of the initial 4-case ({0, 1, 3, 4}) sweep.
#
# All 16 missing cases (2, 5-19) will get all four tracker variants
# (fom_hct_on, fom_hct_off, rom_hct_on, rom_hct_off).  Total runtime is
# ~27 h at ~25 min per tracker on a workstation-class GPU.  Won't fit
# in one night; this is expected.  The sweep script's sentinel-based
# resume means re-running this wrapper the next morning picks up where
# it left off — no work is redone.
#
# Env vars this wrapper sets (before delegating to the sweep):
#   CASES      = "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
#   SWEEP_ROOT = /flash/users/$USER/overnight_all20_$(date +%Y%m%d_%H%M%S)
#                (unless already set in the environment)
#
# Every other override understood by overnight_rom_vs_fom_sweep.sh is
# forwarded verbatim.  Example usage:
#
#   # First run (starts tonight)
#   bash /flash/shared/jax/JAXTrace/scripts/overnight_all20_cases.sh
#
#   # Second run (resume tomorrow morning against the same sweep root)
#   SWEEP_ROOT=/flash/users/ali/overnight_all20_20260717_190000 \
#       bash /flash/shared/jax/JAXTrace/scripts/overnight_all20_cases.sh
#
#   # Just re-do compare + mixing at the end (using the catch-up wrapper)
#   SWEEP_ROOT=/flash/users/ali/overnight_all20_20260717_190000 \
#       bash /flash/shared/jax/JAXTrace/scripts/overnight_rerun_compare_and_mixing.sh
#
# Prep phase behaviour
# --------------------
# The sweep's Phase 0 will reconstruct ROM PVTUs for every case that
# doesn't already have one on disk.  The BasisCoefficients array in
# cylindrical.som.fswrom.romdata has all 20 cases (indices 0..19), so
# every case is a valid target.  Case 002 in particular was missing at
# sweep time because it was never in the CASES list — Phase 0 fills it
# in.
#
# The three prep tools this triggers all live on-workstation:
#   /scratch/shared/ROM/FOM/reconstruct_rom_velocities.sh
#   /flash/shared/jax/JAXTrace/scripts/generate_jaxtrace_scripts.sh
#   /flash/shared/jax/JAXTrace/scripts/generate_jaxtrace_recon_scripts.sh
# They are already CASES-aware; the wrapper simply exports the extended
# CASES list before delegating.
# =============================================================================

set -uo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CASES="${CASES:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
export SWEEP_ROOT="${SWEEP_ROOT:-/flash/users/${USER:-$(whoami)}/overnight_all20_$(date +%Y%m%d_%H%M%S)}"

# Delegate.  All other env-var overrides (VARIANTS, FORCE_RERUN,
# SKIP_PREP, SKIP_TRACKING, SKIP_COMPARE, SKIP_MIXING, COMPARE_STEP,
# MIXING_STRIDE, DRY_RUN, ...) are inherited from the caller's
# environment.
exec bash "$_HERE/overnight_rom_vs_fom_sweep.sh" "$@"
