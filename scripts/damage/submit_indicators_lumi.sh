#!/usr/bin/env bash
# =============================================================================
# Overnight indicator run on LUMI — all four families, 16 cases.
#
#   bash submit_indicators_lumi.sh                 # print the plan, submit nothing
#   SUBMIT=1 bash submit_indicators_lumi.sh        # submit all 16
#   SUBMIT=1 CASES="D2 B2" bash submit_indicators_lumi.sh
#   SUBMIT=1 MODE=short bash submit_indicators_lumi.sh   # 200-step sanity pass
#   SUBMIT=1 DETACHMENT=1 bash submit_indicators_lumi.sh # + det F and Pi_1_min
#
# HOW THIS WORKS, AND WHY IT DOES NOT WRITE A NEW RUN SCRIPT
# -----------------------------------------------------------
# Each case already has a correct, verified run script under
#   scripts/dfamily/lumi/percase/[abc/]run_jaxtrace_<CASE>.sh
# and those encode per-case facts that CANNOT be generalised: B2 uses
# dt=0.0003 and VEL_END=201 while every other case uses dt=0.0003629 and 199,
# and each case's mesh file stem differs from its folder name (D2.gid holds
# A2_*.pvtu). Regenerating them from a template is how those details get lost.
#
# ⚠️ So this wrapper does NOT rewrite them. It copies each script to a scratch
# staging directory, appends the damage/indicator flags to its ARGS array, and
# submits the copy. The originals under scripts/ are read-only inputs here, and
# the authoritative copies beside the cases in lorenzgl/ are never touched.
#
# The injected flags are exactly what the workstation validation run used:
#   --damage --damage-driver models --damage-order 1
#   --damage-indicators --residence-edot-threshold <thr>
# and, with DETACHMENT=1, additionally
#   --damage-detachment --damage-tau-ref-edot <edot>
# See docs/damage/NEXT_STEPS_detachment_criterion.md Part A and B.6f/B.6g.
# =============================================================================
set -uo pipefail

REPO="${REPO:-/projappl/project_465002752/hashemia/JAXTrace}"
PERCASE="${PERCASE:-${REPO}/scripts/dfamily/lumi/percase}"
STAGE="${STAGE:-/scratch/project_465002752/hashemia/damage_indicators/staged}"
# ⚠️ LOGS LIVE WITH THE RESULTS.
# SLURM's stdout/stderr go into the per-case output directory, not a central
# logs/ dir, so <OUT_ROOT>/<case>/ is self-contained: the npz, the exports, the
# job log and a copy of the exact script that produced them. A result whose log
# sits somewhere else is unreproducible six months later. A central copy is still
# kept, because the output dir does not exist until the job's own resolver runs.
LOGDIR="${LOGDIR:-/scratch/project_465002752/hashemia/logs}"

MODE="${MODE:-full}"
RESIDENCE_THR="${RESIDENCE_THR:-1.0}"
DAMAGE_ORDER="${DAMAGE_ORDER:-1}"
SUBMIT="${SUBMIT:-0}"

# ── Detachment groundwork (Part B) ───────────────────────────────────────────
# DETACHMENT=1 appends --damage-detachment, adding two accumulator columns:
#   ln_J    = integral(div(u) dt)  -> det F = exp(ln_J)   [the step-1 GATE]
#   tau_min = running min of Pi_1 = sigma_eq / tau_yield_ref
#
# ⚠️ COST, measured on D2 (166 snapshots, 1.92M nodes): the field stack grows
# 3 -> 5 slots, i.e. 3650 -> 6083 MB, +2433 MB on BOTH host and device. On
# LUMI's MI250X (64 GB) that is comfortable; it was verified on a 32.6 GB
# RTX 5090 at 12.2 GB total resident, 63 % free. Build time scales with the
# slot count in the damage stage.
#
# ✅ tau_min is recorded as a MINIMUM, never thresholded in-kernel, so every
# Pi_1_crit is a post-process on the npz and the sensitivity sweep costs no
# extra jobs. Do NOT add a threshold knob here.
DETACHMENT="${DETACHMENT:-0}"
TAU_REF_EDOT="${TAU_REF_EDOT:-1.0}"

# ⚠️ OUTPUT MUST NOT GO INTO THE CASE FOLDER.
# The per-case scripts ship with OUTPUT_TARGET=case-flash, which writes results
# into <case>.gid/<subfolder> -- and the cases live under lorenzgl/, which is
# READ-ONLY for us. A staged run left as-is would either fail on write or try to
# modify someone else's tree. The injection therefore forces OUTPUT_TARGET=scratch,
# which writes to /scratch/$PROJECT/$USER/outputs/<case>_jaxtrace_<jobid>.
OUT_ROOT="${OUT_ROOT:-/scratch/project_465002752/hashemia/damage_indicators}"

# ── Octree cache: the ONE thing allowed in the case folder on LUMI ───────────
# ⚠️ Note the asymmetry, and keep it. RESULTS must stay out of the case folders
# under lorenzgl/ (hence OUTPUT_TARGET=scratch above), but the OCTREE CACHE is
# explicitly permitted to live beside the case as <case>.gid/octree_cache/.
# That is the right place for it: the cache is a pure function of the mesh and
# the build options, so one copy beside the case is reused by every run of that
# case by anyone, instead of each user rebuilding 963s of octree in their own
# scratch.
#
# The key hashes the mesh arrays, every build option and the build modules'
# source, so a code or option change rebuilds rather than reusing a stale tree.
# hdf5 gives one self-contained .h5 per entry, which keeps the case folder tidy
# and travels with the case; it is verified working on LUMI (h5py 3.16.0 /
# HDF5 2.0.0 in the container via PYTHONPATH=$PKGS) and falls back to a .npy
# directory if h5py is ever missing, rather than losing the build.
OCTREE_CACHE_IN_CASE="${OCTREE_CACHE_IN_CASE:-1}"
OCTREE_CACHE_FORMAT="${OCTREE_CACHE_FORMAT:-hdf5}"

ALL_CASES="D1 D2 D3 D4 A1 A2 A3 A4 B1 B2 B3 B4 C1 C2 C3 C4"
CASES="${CASES:-$ALL_CASES}"

script_for() {
    case "$1" in
        D[1-4]) echo "${PERCASE}/run_jaxtrace_$1.sh" ;;
        [ABC][1-4]) echo "${PERCASE}/abc/run_jaxtrace_$1.sh" ;;
        *) echo "" ;;
    esac
}

mkdir -p "$STAGE" "$LOGDIR" || { echo "ERROR: cannot create $STAGE" >&2; exit 1; }

echo "============================================================"
echo " LUMI indicator submission"
echo "   mode       : $MODE"
echo "   cases      : $CASES"
echo "   driver     : models   order $DAMAGE_ORDER"
echo "   t_res thr  : $RESIDENCE_THR 1/s"
echo "   detachment : $([ "$DETACHMENT" = "1" ] && echo "ON  (ln_J + tau_min, tau_ref_edot $TAU_REF_EDOT 1/s, stack 3->5 slots)" || echo "off")"
echo "   out root   : $OUT_ROOT  (OUTPUT_TARGET=scratch)"
echo "   octree\$   : $([ "$OCTREE_CACHE_IN_CASE" = 1 ] && echo "beside each case (<case>.gid/octree_cache)" || echo "scratch")  format $OCTREE_CACHE_FORMAT"
echo "   staged to  : $STAGE"
echo "   submit     : $([ "$SUBMIT" = "1" ] && echo YES || echo "no (dry run)")"
echo "============================================================"
echo

n_ok=0 n_skip=0
for c in $CASES; do
    src="$(script_for "$c")"
    if [ -z "$src" ] || [ ! -f "$src" ]; then
        echo "  SKIP $c — no per-case script at ${src:-<unknown>}"
        n_skip=$((n_skip + 1))
        continue
    fi

    dst="${STAGE}/run_indicators_${c}.sh"
    cp "$src" "$dst" || { echo "  SKIP $c — copy failed"; n_skip=$((n_skip+1)); continue; }

    # ── Inject the damage flags into the ARGS array ──────────────────────
    # ⚠️ Appended as a new block AFTER the closing paren of ARGS=( ... ), so
    # nothing in the per-case configuration is edited or reordered. The
    # per-case conditional flags that follow ARGS=() keep working because
    # they also use ARGS+=().
    python3 - "$dst" "$DAMAGE_ORDER" "$RESIDENCE_THR" "$OUT_ROOT" "$c" \
             "$OCTREE_CACHE_IN_CASE" "$OCTREE_CACHE_FORMAT" \
             "$DETACHMENT" "$TAU_REF_EDOT" <<'PYEOF'
import re, sys
(path, order, thr, out_root, case, oc_in_case, oc_fmt,
 detach, tau_ref) = sys.argv[1:10]
s = open(path).read()

# 1. Append the indicator flags right after the ARGS array literal closes.
m = re.search(r'^ARGS=\(\n(?:.*\n)*?\)\n', s, flags=re.M)
if not m:
    sys.exit(f"ARGS=( ... ) block not found in {path}")
inject = (
    "\n# ── Damage indicators (injected by submit_indicators_lumi.sh) ───────\n"
    "# models driver is required: the Piwnik-Plata integrand is p/sigma_bar\n"
    "# = -eta, and eta exists only in the 3-slot [edot, eta, s1_ratio] stack.\n"
    "ARGS+=(\n"
    "  --damage\n"
    "  --damage-driver models\n"
    f"  --damage-order {int(order)}\n"
    "  --damage-indicators\n"
    f"  --residence-edot-threshold {float(thr)}\n"
    ")\n"
)
# ⚠️ Detachment appends SEPARATELY, so a run without it stages a script that is
# byte-identical to the previous indicator runs -- the 16-case results stay
# comparable rather than silently acquiring two columns.
if detach == "1":
    inject += (
        "\n# ── Detachment groundwork (injected) ────────────────────────────\n"
        "# ln_J = int(div u dt) -> det F; tau_min = running min of Pi_1.\n"
        "# Pi_1 is a ratio to the Norton flow stress at TAU_REF_EDOT with the\n"
        "# LOCAL temperature -- not the local flow stress, which on the models\n"
        "# route IS sigma_eq and would make Pi_1 identically 1 everywhere.\n"
        "ARGS+=(\n"
        "  --damage-detachment\n"
        f"  --damage-tau-ref-edot {float(tau_ref)}\n"
        ")\n"
    )
s = s[:m.end()] + inject + s[m.end():]

# 2. ⚠️ Force the output OUT of the case folder. The cases live under
#    lorenzgl/, which we may only READ, so case / case-flash must not be used.
s, n = re.subn(r'^OUTPUT_TARGET=\S+',
               'OUTPUT_TARGET=scratch', s, count=1, flags=re.M)
if n == 0:
    sys.exit(f"OUTPUT_TARGET not found in {path} -- refusing to stage a run "
             f"that might write into the read-only case folder")
# Pin the destination explicitly rather than relying on the script's default
# /scratch/$PROJECT/$USER/outputs, so every indicator run lands together.
s, n2 = re.subn(r'^(\s*)SCRATCH_BASE="/scratch/\$\{PROJECT\}/\$\{USER\}/outputs"',
                rf'\1SCRATCH_BASE="{out_root}"', s, count=1, flags=re.M)
if n2 == 0:
    sys.exit(f"SCRATCH_BASE line not found in {path}")
s, n3 = re.subn(r'^(\s*)SCRATCH_FOLDER="\${_CASE}_jaxtrace_\${SLURM_JOB_ID}"',
                rf'\1SCRATCH_FOLDER="{case}"', s, count=1, flags=re.M)
# Not fatal: the auto name still lands under OUT_ROOT, just with a job id.
if n3 == 0:
    print(f"    [note] {case}: kept the auto SCRATCH_FOLDER name")

# 3. Octree cache. ⚠️ Deliberately the OPPOSITE of the output rule above: the
#    cache IS allowed beside the case on LUMI, and belongs there so every run of
#    that case shares one build instead of each user rebuilding it.
s, n4 = re.subn(r'^OCTREE_CACHE_IN_CASE=\S+',
                f'OCTREE_CACHE_IN_CASE={int(oc_in_case)}', s, count=1, flags=re.M)
if n4 == 0:
    print(f"    [note] {case}: no OCTREE_CACHE_IN_CASE key; cache left as the "
          f"script had it")
s, n5 = re.subn(r'^OCTREE_CACHE_FORMAT=\S+',
                f'OCTREE_CACHE_FORMAT={oc_fmt}', s, count=1, flags=re.M)
if n5 == 0:
    print(f"    [note] {case}: no OCTREE_CACHE_FORMAT key; format left as the "
          f"script had it")

# 4. Distinguish the job name in squeue / log filenames.
s = re.sub(r'^#SBATCH --job-name=\S+', '#SBATCH --job-name=dmgind',
           s, count=1, flags=re.M)

open(path, "w").write(s)
PYEOF
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "  SKIP $c — injection failed (rc=$rc)"
        n_skip=$((n_skip + 1))
        continue
    fi

    # Shorten for a sanity pass without touching the per-case physics.
    if [ "$MODE" = "short" ]; then
        sed -i -E 's/^N_STEPS=[0-9]+/N_STEPS=200/' "$dst"
        sed -i -E 's/^EXPORT_FREQ=[0-9]+/EXPORT_FREQ=100/' "$dst"
    fi

    chmod +x "$dst"
    if ! bash -n "$dst"; then
        echo "  SKIP $c — staged script has a syntax error"
        n_skip=$((n_skip + 1))
        continue
    fi

    # Confirm the injection actually landed, rather than trusting the regex.
    if ! grep -q -- "--damage-indicators" "$dst"; then
        echo "  SKIP $c — flags not present after injection"
        n_skip=$((n_skip + 1))
        continue
    fi

    _ns=$(grep -oE '^N_STEPS=[0-9]+' "$dst" | head -1 | cut -d= -f2)
    _dt=$(grep -oE '^DT=[0-9.e-]+' "$dst" | head -1 | cut -d= -f2)
    if [ "$SUBMIT" = "1" ]; then
        # The run's own output dir, matching the SCRATCH_FOLDER pinned above.
        _case_out="${OUT_ROOT}/${c}"
        mkdir -p "$_case_out" 2>/dev/null
        jid=$(sbatch --parsable \
                --job-name="dmgind_${c}" \
                --output="${_case_out}/slurm_%j.out" \
                --error="${_case_out}/slurm_%j.err" \
                "$dst" 2>&1)
        if [[ "$jid" =~ ^[0-9]+$ ]]; then
            # Keep the exact script and the submission metadata beside the
            # results, so the run can be reproduced from its own directory.
            cp "$dst" "${_case_out}/run_script_${jid}.sh" 2>/dev/null
            {
                echo "case            : $c"
                echo "job_id          : $jid"
                echo "submitted       : $(date -Is)"
                echo "submitted_by    : ${USER}@$(hostname)"
                echo "reference       : $src"
                echo "n_steps         : $_ns"
                echo "dt              : $_dt"
                echo "damage_order    : $DAMAGE_ORDER"
                echo "residence_thr   : $RESIDENCE_THR"
                echo "repo_commit     : $(cd "$REPO" 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
                echo "log             : ${_case_out}/slurm_${jid}.out"
            } > "${_case_out}/RUN_INFO_${jid}.txt"
            echo "  SUBMIT $c  job $jid   steps=$_ns dt=$_dt"
            echo "         log: ${_case_out}/slurm_${jid}.out"
            n_ok=$((n_ok + 1))
        else
            echo "  FAIL   $c  sbatch: $jid"
            n_skip=$((n_skip + 1))
        fi
    else
        echo "  READY  $c  steps=$_ns dt=$_dt   -> $dst"
        n_ok=$((n_ok + 1))
    fi
done

echo
echo "============================================================"
echo "  ready/submitted: $n_ok    skipped: $n_skip"
if [ "$SUBMIT" != "1" ]; then
    echo
    echo "  Dry run — nothing was submitted. Re-run with SUBMIT=1 to launch."
    echo "  Inspect a staged script first, e.g.:"
    echo "    diff ${PERCASE}/run_jaxtrace_D2.sh ${STAGE}/run_indicators_D2.sh"
fi
echo
echo "  After the runs finish, build the indicator table with:"
echo "    python3 ${REPO}/scripts/damage/indicator_table.py \\"
echo "        --src ${OUT_ROOT} --out indicators.csv"
echo "============================================================"
