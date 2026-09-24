#!/bin/bash
# =============================================================================
# overnight_rerun_compare_and_mixing.sh
#
# Catch-up wrapper for overnight_rom_vs_fom_sweep.sh.  Runs only the
# comparison + mixing phases against an EXISTING sweep root, reusing
# the tracker particles.vtkhdf archives that already exist on disk.
#
# Use when:
#
#   * A prior overnight run finished tracking but its compare / mixing
#     steps failed for a fixable reason (missing venv, module import,
#     path typo, etc.); the tracker outputs are perfectly good.
#
#   * You changed the compare or mixing script and want to re-produce
#     the outputs without redoing the 5+ hours of tracking.
#
# Behaviour: sets SKIP_PREP=1 + SKIP_TRACKING=1 and delegates to
# overnight_rom_vs_fom_sweep.sh.  The compare + mixing steps are always
# rerun (their outputs overwrite in place — the compare tool always
# writes fresh VTUs and the mixing tool always writes fresh CSVs).
#
# The tracker sentinels are respected: if any variant's
# particles.vtkhdf is missing, the compare / mixing step for that
# combination is reported as SKIP rather than crashing.
#
# Usage:
#
#   SWEEP_ROOT=/flash/users/$USER/overnight_rom_vs_fom_20260716_165147 \
#     bash /flash/shared/jax/JAXTrace/scripts/overnight_rerun_compare_and_mixing.sh
#
#   # or run against a fresh sweep root (a new one is created if unset)
#   bash /flash/shared/jax/JAXTrace/scripts/overnight_rerun_compare_and_mixing.sh
#
# Every override understood by overnight_rom_vs_fom_sweep.sh (CASES,
# VARIANTS, ROM_FORMULA, FOM_ROOT, ROM_RECON_ROOT, JAXTRACE,
# COMPARE_STEP, MIXING_STRIDE, DRY_RUN, ...) is forwarded verbatim.
# =============================================================================

set -uo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SKIP_PREP=1
export SKIP_TRACKING=1
exec bash "$_HERE/overnight_rom_vs_fom_sweep.sh" "$@"
