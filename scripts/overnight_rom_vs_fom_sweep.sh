#!/bin/bash
# =============================================================================
# overnight_rom_vs_fom_sweep.sh
#
# One-shot orchestrator for Roadmap § 2 (FOM-vs-ROM PT sweep) + § 3
# (HCT-3D ablation).  For each case in $CASES it runs four tracking
# variants and, once all four are done, four pairwise comparisons and
# two Lagrangian mixing diagnostics.
#
# Variants per case (output paths follow the runner's INPUT tree — FOM
# variants land under FOM_ROOT/<case>.gid/..., ROM variants under
# ROM_RECON_ROOT/<case>.gid/..., because the case runners derive
# _CASE_DIR from their own INPUT):
#
#   fom_hct_on   FOM/<case>.gid/post_pt/fom_hct_on/<run>/particles.vtkhdf
#   fom_hct_off  FOM/<case>.gid/post_pt/fom_hct_off/<run>/particles.vtkhdf
#   rom_hct_on   ROM_recon_<formula>/<case>.gid/post_pt/rom_<formula>_hct_on/<run>/particles.vtkhdf
#   rom_hct_off  ROM_recon_<formula>/<case>.gid/post_pt/rom_<formula>_hct_off/<run>/particles.vtkhdf
#
# Comparisons emitted per case (via scripts/compare_rom_vs_fom_tracking.py):
#
#   rom_vs_fom_hct_on   ROM(HCT-on) vs FOM(HCT-on): the primary § 2 answer
#   rom_vs_fom_hct_off  ROM(HCT-off) vs FOM(HCT-off): sanity check
#   fom_hct_on_vs_off   HCT effect on FOM tracking
#   rom_hct_on_vs_off   HCT effect on ROM tracking
#
# Mixing diagnostics (via scripts/lagrangian_mixing_diagnostics.py):
#
#   rom_vs_fom_hct_on_mixing  residence-time + pair-separation on the
#                             primary comparison archives
#   rom_vs_fom_hct_off_mixing same for the HCT-off pair
#
# Layout guarantees:
#
#   * Every variant's output subfolder is unique -- no collisions
#     between variants of the same case, and re-running a variant
#     overwrites only that variant's own directory.
#   * A sentinel file $VARIANT.done is written after each variant's
#     particles.vtkhdf lands, so a re-invocation of this script
#     picks up where it left off.  Delete the sentinel to force a
#     re-run of a specific variant.
#   * All logs (per-variant + per-comparison + per-mixing) sit under
#     the sweep root directory, one file per artifact.
#
# Usage:
#
#   bash /flash/shared/jax/JAXTrace/scripts/overnight_rom_vs_fom_sweep.sh
#
#   CASES="4 1 3" bash overnight_rom_vs_fom_sweep.sh    # subset
#   FORCE_RERUN=1 ...                                    # ignore sentinels
#   VARIANTS="fom_hct_on rom_hct_on" ...                 # skip the ablation
#   SKIP_TRACKING=1 ...                                  # only run compare + mixing
#   SKIP_COMPARE=1 ...                                   # only tracking
#   DRY_RUN=1 ...                                        # show plan, do nothing
#
# All env-var overrides:
#
#   CASES              (default: "4 1 3 0")   space-separated case indices
#   VARIANTS           (default: "fom_hct_on fom_hct_off rom_hct_on rom_hct_off")
#   ROM_FORMULA        (default: centered)    formula label
#   ROM_RECON_ROOT     (default: /scratch/shared/ROM/ROM_recon_${ROM_FORMULA})
#   FOM_ROOT           (default: /scratch/shared/ROM/FOM)
#   SWEEP_ROOT         (default: /flash/users/$USER/overnight_rom_vs_fom_$(date +%Y%m%d_%H%M%S))
#                                             where logs + summaries land
#   JAXTRACE           (default: /flash/shared/jax/JAXTrace)
#   FORCE_RERUN        (default: 0)           1 = ignore .done sentinels
#   SKIP_TRACKING      (default: 0)           1 = only run compare + mixing
#   SKIP_COMPARE       (default: 0)           1 = only tracking
#   SKIP_MIXING        (default: 0)           1 = skip the Lagrangian mixing step
#   DRY_RUN            (default: 0)           1 = print the plan, do nothing
#   COMPARE_STEP       (default: 500)         --step passed to compare tool
#   MIXING_STRIDE      (default: 20)          --stride passed to mixing tool
#
# =============================================================================

set -uo pipefail

CASES="${CASES:-4 1 3 0}"
VARIANTS="${VARIANTS:-fom_hct_on fom_hct_off rom_hct_on rom_hct_off}"
ROM_FORMULA="${ROM_FORMULA:-centered}"
ROM_RECON_ROOT="${ROM_RECON_ROOT:-/scratch/shared/ROM/ROM_recon_${ROM_FORMULA}}"
FOM_ROOT="${FOM_ROOT:-/scratch/shared/ROM/FOM}"
SWEEP_ROOT="${SWEEP_ROOT:-/flash/users/${USER:-$(whoami)}/overnight_rom_vs_fom_$(date +%Y%m%d_%H%M%S)}"
JAXTRACE="${JAXTRACE:-/flash/shared/jax/JAXTrace}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SKIP_TRACKING="${SKIP_TRACKING:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"
SKIP_MIXING="${SKIP_MIXING:-0}"
DRY_RUN="${DRY_RUN:-0}"
COMPARE_STEP="${COMPARE_STEP:-500}"
MIXING_STRIDE="${MIXING_STRIDE:-20}"

RUN_TAG_RE="run_grid-frac_n360000_s2000"   # deterministic under the current template

mkdir -p "$SWEEP_ROOT"
GLOBAL_LOG="$SWEEP_ROOT/sweep.log"
SUMMARY_LOG="$SWEEP_ROOT/summary.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$GLOBAL_LOG"; }

log "== Overnight ROM-vs-FOM sweep =="
log "  CASES         : $CASES"
log "  VARIANTS      : $VARIANTS"
log "  ROM_FORMULA   : $ROM_FORMULA"
log "  ROM_RECON_ROOT: $ROM_RECON_ROOT"
log "  FOM_ROOT      : $FOM_ROOT"
log "  SWEEP_ROOT    : $SWEEP_ROOT"
log "  JAXTRACE      : $JAXTRACE"
log "  FORCE_RERUN   : $FORCE_RERUN"
log "  SKIP_TRACKING : $SKIP_TRACKING"
log "  SKIP_COMPARE  : $SKIP_COMPARE"
log "  SKIP_MIXING   : $SKIP_MIXING"
log "  DRY_RUN       : $DRY_RUN"
log "  COMPARE_STEP  : $COMPARE_STEP"
log "  MIXING_STRIDE : $MIXING_STRIDE"

# ── Static per-case checks ──────────────────────────────────────────────────
for CASE in $CASES; do
    CASE_ID=$(printf "%03d" "$CASE")
    CASE_DIR="$FOM_ROOT/cylindrical_${CASE_ID}.gid"
    FOM_RUNNER="$CASE_DIR/run_jaxtrace.sh"
    ROM_RUNNER="$CASE_DIR/run_jaxtrace_recon.sh"

    if [[ "$VARIANTS" == *fom* ]] && [ ! -f "$FOM_RUNNER" ]; then
        log "ERROR: case $CASE_ID needs $FOM_RUNNER (run generate_jaxtrace_scripts.sh first)"
        exit 3
    fi
    if [[ "$VARIANTS" == *rom* ]]; then
        if [ ! -f "$ROM_RUNNER" ]; then
            log "ERROR: case $CASE_ID needs $ROM_RUNNER (run generate_jaxtrace_recon_scripts.sh first)"
            exit 3
        fi
        if [ ! -f "$ROM_RECON_ROOT/cylindrical_${CASE_ID}.gid/post/cylindrical_0.pvtu" ]; then
            log "ERROR: case $CASE_ID needs the ROM PVTU at $ROM_RECON_ROOT/cylindrical_${CASE_ID}.gid/post/cylindrical_0.pvtu (run reconstruct_rom_velocities.sh first)"
            exit 3
        fi
    fi
done

# ── Variant metadata ────────────────────────────────────────────────────────
# For each variant define:
#   * SOURCE_RUNNER: run_jaxtrace.sh (FOM) or run_jaxtrace_recon.sh (ROM)
#   * OUTPUT_CASE_SUBFOLDER: unique per variant
#   * RECOVERY_METHOD, GRADIENT_RECOVERY: HCT toggle
variant_source_runner() {
    case "$1" in
        fom_*) echo "run_jaxtrace.sh" ;;
        rom_*) echo "run_jaxtrace_recon.sh" ;;
        *)     echo "" ;;
    esac
}

# The tracker runner writes to ${_CASE_DIR}/${OUTPUT_CASE_SUBFOLDER},
# where _CASE_DIR is derived from the runner's INPUT.  For fom_*
# variants INPUT is the FOM case folder; for rom_* variants INPUT is
# the ROM_recon case folder.  Report where each variant's output tree
# lives so the sentinel/compare/mixing paths line up with reality.
variant_case_root() {
    local CASE_ID="$1"
    case "$2" in
        fom_*) echo "$FOM_ROOT/cylindrical_${CASE_ID}.gid" ;;
        rom_*) echo "$ROM_RECON_ROOT/cylindrical_${CASE_ID}.gid" ;;
    esac
}

# The source runner (run_jaxtrace.sh / run_jaxtrace_recon.sh) always
# sits inside the FOM case folder, regardless of where its output
# lands, because that's where generate_jaxtrace_*_scripts.sh writes
# them.
variant_source_dir() {
    local CASE_ID="$1"
    echo "$FOM_ROOT/cylindrical_${CASE_ID}.gid"
}

variant_output_subfolder() {
    case "$1" in
        fom_hct_on)  echo "post_pt/fom_hct_on" ;;
        fom_hct_off) echo "post_pt/fom_hct_off" ;;
        rom_hct_on)  echo "post_pt/rom_${ROM_FORMULA}_hct_on" ;;
        rom_hct_off) echo "post_pt/rom_${ROM_FORMULA}_hct_off" ;;
    esac
}

variant_gradient_recovery() {
    case "$1" in
        *_hct_on)  echo "1" ;;
        *_hct_off) echo "0" ;;
    esac
}

variant_recovery_method() {
    case "$1" in
        *_hct_on)  echo "hct_cubic" ;;
        *_hct_off) echo "centroid_taylor" ;;  # ignored when gradient-recovery=0 but still valid
    esac
}

# ── Run one variant for one case ────────────────────────────────────────────
# Sed-patches a temp copy of the source runner with the four knobs we need
# to override, runs bash on the patched copy, and writes a .done sentinel
# on success.
run_variant() {
    local CASE_ID="$1"
    local VARIANT="$2"

    local SRC_DIR
    SRC_DIR="$(variant_source_dir "$CASE_ID")"
    local SRC_NAME
    SRC_NAME="$(variant_source_runner "$VARIANT")"
    local SRC="$SRC_DIR/$SRC_NAME"
    local CASE_ROOT
    CASE_ROOT="$(variant_case_root "$CASE_ID" "$VARIANT")"
    local OCS
    OCS="$(variant_output_subfolder "$VARIANT")"
    local GR
    GR="$(variant_gradient_recovery "$VARIANT")"
    local RM
    RM="$(variant_recovery_method "$VARIANT")"

    local OUT_DIR="$CASE_ROOT/$OCS"
    local PARTICLES="$OUT_DIR/$RUN_TAG_RE/particles.vtkhdf"
    local DONE_SENTINEL="$OUT_DIR/.overnight_${VARIANT}.done"
    local LOG="$SWEEP_ROOT/case${CASE_ID}_${VARIANT}.log"

    if [ "$FORCE_RERUN" != "1" ] && [ -f "$DONE_SENTINEL" ] && [ -f "$PARTICLES" ]; then
        log "  case $CASE_ID / $VARIANT: SKIP (sentinel present)"
        return 0
    fi

    if [ "$DRY_RUN" = "1" ]; then
        log "  case $CASE_ID / $VARIANT: DRY-RUN"
        log "    source runner   <- $SRC"
        log "    patched runner  -> $SRC_DIR/.overnight_${VARIANT}.sh"
        log "    output root     -> $CASE_ROOT ($( [ "$VARIANT" = "${VARIANT#fom_}" ] && echo "ROM" || echo "FOM" ) tree)"
        log "    output subdir   -> $OUT_DIR"
        log "    gradient/method -> $GR / $RM"
        log "    particles will be at: $PARTICLES"
        return 0
    fi

    mkdir -p "$OUT_DIR"
    local PATCHED="$SRC_DIR/.overnight_${VARIANT}.sh"
    sed \
        -e "s|^OUTPUT_CASE_SUBFOLDER=.*|OUTPUT_CASE_SUBFOLDER=\"$OCS\"|" \
        -e "s|^GRADIENT_RECOVERY=.*|GRADIENT_RECOVERY=$GR|" \
        -e "s|^RECOVERY_METHOD=.*|RECOVERY_METHOD=\"$RM\"|" \
        -e "s|^ENABLE_UNION=.*|ENABLE_UNION=0|" \
        "$SRC" > "$PATCHED"
    chmod +x "$PATCHED"

    log "  case $CASE_ID / $VARIANT: RUNNING (log $LOG)"
    local T0=$SECONDS
    if ( cd "$SRC_DIR" && bash "$(basename "$PATCHED")" > "$LOG" 2>&1 ); then
        local DT=$(( SECONDS - T0 ))
        if [ -f "$PARTICLES" ]; then
            touch "$DONE_SENTINEL"
            log "  case $CASE_ID / $VARIANT: OK (${DT}s, particles at $PARTICLES)"
        else
            log "  case $CASE_ID / $VARIANT: FAILED (${DT}s, particles.vtkhdf missing at $PARTICLES) — see $LOG"
            return 1
        fi
    else
        local RC=$?
        log "  case $CASE_ID / $VARIANT: FAILED (exit $RC) — see $LOG"
        return $RC
    fi
    rm -f "$PATCHED"
}

# ── Run all tracker variants ────────────────────────────────────────────────
if [ "$SKIP_TRACKING" != "1" ]; then
    log
    log "== [1/3] Tracker variants =="
    for CASE in $CASES; do
        CASE_ID=$(printf "%03d" "$CASE")
        for VARIANT in $VARIANTS; do
            run_variant "$CASE_ID" "$VARIANT" || true
        done
    done
else
    log
    log "== [1/3] SKIP tracker variants (SKIP_TRACKING=1) =="
fi

# ── Comparisons ─────────────────────────────────────────────────────────────
particles_path() {
    local CASE_ID="$1"
    local VARIANT="$2"
    local CASE_ROOT
    CASE_ROOT="$(variant_case_root "$CASE_ID" "$VARIANT")"
    local OCS
    OCS="$(variant_output_subfolder "$VARIANT")"
    echo "$CASE_ROOT/$OCS/$RUN_TAG_RE/particles.vtkhdf"
}

compare_pair() {
    local CASE_ID="$1"
    local LABEL="$2"
    local A_VARIANT="$3"
    local B_VARIANT="$4"

    local A_PART
    A_PART="$(particles_path "$CASE_ID" "$A_VARIANT")"
    local B_PART
    B_PART="$(particles_path "$CASE_ID" "$B_VARIANT")"

    if [ ! -f "$A_PART" ] || [ ! -f "$B_PART" ]; then
        log "  case $CASE_ID / $LABEL: SKIP (missing archive)"
        [ ! -f "$A_PART" ] && log "    missing: $A_PART"
        [ ! -f "$B_PART" ] && log "    missing: $B_PART"
        return 0
    fi

    local OUT_DIR="$SWEEP_ROOT/case${CASE_ID}_compare_${LABEL}"
    mkdir -p "$OUT_DIR"
    local LOG="$OUT_DIR/compare.log"
    local VTU="$OUT_DIR/${LABEL}_step${COMPARE_STEP}.vtu"

    if [ "$DRY_RUN" = "1" ]; then
        log "  case $CASE_ID / $LABEL: DRY-RUN compare -> $OUT_DIR"
        return 0
    fi

    log "  case $CASE_ID / $LABEL: compare (log $LOG, vtu $VTU)"
    python3 "$JAXTRACE/scripts/compare_rom_vs_fom_tracking.py" \
        --fom-vtkhdf "$A_PART" --rom-vtkhdf "$B_PART" \
        --step "$COMPARE_STEP" --out-vtu "$VTU" \
        > "$LOG" 2>&1 || log "    WARN: compare exited nonzero — see $LOG"

    # Also do a last-step comparison with --suggest-alive-step so the
    # ballistic-tail regime is covered too.
    local LAST_LOG="$OUT_DIR/compare_last.log"
    python3 "$JAXTRACE/scripts/compare_rom_vs_fom_tracking.py" \
        --fom-vtkhdf "$A_PART" --rom-vtkhdf "$B_PART" \
        --step last --suggest-alive-step \
        > "$LAST_LOG" 2>&1 || log "    WARN: last-step compare exited nonzero — see $LAST_LOG"
}

if [ "$SKIP_COMPARE" != "1" ]; then
    log
    log "== [2/3] Comparisons =="
    for CASE in $CASES; do
        CASE_ID=$(printf "%03d" "$CASE")
        # Primary: ROM vs FOM at same recovery mode
        compare_pair "$CASE_ID" "rom_vs_fom_hct_on"  "fom_hct_on"  "rom_hct_on"  || true
        compare_pair "$CASE_ID" "rom_vs_fom_hct_off" "fom_hct_off" "rom_hct_off" || true
        # Ablation: HCT effect within a single velocity source
        compare_pair "$CASE_ID" "fom_hct_on_vs_off"  "fom_hct_on"  "fom_hct_off" || true
        compare_pair "$CASE_ID" "rom_hct_on_vs_off"  "rom_hct_on"  "rom_hct_off" || true
    done
else
    log
    log "== [2/3] SKIP comparisons (SKIP_COMPARE=1) =="
fi

# ── Mixing diagnostics ──────────────────────────────────────────────────────
mixing_pair() {
    local CASE_ID="$1"
    local LABEL="$2"
    local FOM_VARIANT="$3"
    local ROM_VARIANT="$4"

    local FOM_PART
    FOM_PART="$(particles_path "$CASE_ID" "$FOM_VARIANT")"
    local ROM_PART
    ROM_PART="$(particles_path "$CASE_ID" "$ROM_VARIANT")"

    if [ ! -f "$FOM_PART" ] || [ ! -f "$ROM_PART" ]; then
        log "  case $CASE_ID / $LABEL: SKIP mixing (missing archive)"
        return 0
    fi

    local OUT_DIR="$SWEEP_ROOT/case${CASE_ID}_mixing_${LABEL}"
    mkdir -p "$OUT_DIR"
    local LOG="$OUT_DIR/mixing.log"

    if [ "$DRY_RUN" = "1" ]; then
        log "  case $CASE_ID / $LABEL: DRY-RUN mixing -> $OUT_DIR"
        return 0
    fi

    log "  case $CASE_ID / $LABEL: mixing (log $LOG)"
    python3 "$JAXTRACE/scripts/lagrangian_mixing_diagnostics.py" \
        --fom-vtkhdf "$FOM_PART" --rom-vtkhdf "$ROM_PART" \
        --out-dir "$OUT_DIR" --stride "$MIXING_STRIDE" --plot \
        > "$LOG" 2>&1 || log "    WARN: mixing exited nonzero — see $LOG"
}

if [ "$SKIP_MIXING" != "1" ]; then
    log
    log "== [3/3] Mixing diagnostics =="
    for CASE in $CASES; do
        CASE_ID=$(printf "%03d" "$CASE")
        mixing_pair "$CASE_ID" "hct_on"  "fom_hct_on"  "rom_hct_on"  || true
        mixing_pair "$CASE_ID" "hct_off" "fom_hct_off" "rom_hct_off" || true
    done
else
    log
    log "== [3/3] SKIP mixing (SKIP_MIXING=1) =="
fi

# ── Summary ─────────────────────────────────────────────────────────────────
log
log "== Sweep complete =="
log "  Sweep root : $SWEEP_ROOT"
log "  Global log : $GLOBAL_LOG"

# One-line summary per (case, variant)
{
    echo "case,variant,tree,particles_exists,sentinel_exists"
    for CASE in $CASES; do
        CASE_ID=$(printf "%03d" "$CASE")
        for VARIANT in $VARIANTS; do
            CASE_ROOT="$(variant_case_root "$CASE_ID" "$VARIANT")"
            OCS="$(variant_output_subfolder "$VARIANT")"
            OUT_DIR="$CASE_ROOT/$OCS"
            PARTICLES="$OUT_DIR/$RUN_TAG_RE/particles.vtkhdf"
            SENTINEL="$OUT_DIR/.overnight_${VARIANT}.done"
            TREE="$( [ "$VARIANT" = "${VARIANT#fom_}" ] && echo "ROM" || echo "FOM" )"
            printf "%s,%s,%s,%s,%s\n" "$CASE_ID" "$VARIANT" "$TREE" \
                "$( [ -f "$PARTICLES" ] && echo yes || echo no )" \
                "$( [ -f "$SENTINEL" ] && echo yes || echo no )"
        done
    done
} > "$SUMMARY_LOG"
log "  Summary CSV: $SUMMARY_LOG"

exit 0
