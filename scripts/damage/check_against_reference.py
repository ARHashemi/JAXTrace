"""Diff a damage-run script's config against the case's ORIGINAL run_jaxtrace.sh.

    python3 check_against_reference.py --reference <orig run_jaxtrace.sh> \
                                       --candidate <our run script>

WHY THIS EXISTS
---------------
The authoritative configuration for every case is the run_jaxtrace.sh that ships
beside it under
    /scratch/project_465002752/lorenzgl/Cases/PinShapes/<family>/<case>.gid/
Those files encode per-case facts that do NOT generalise, and every one of them
has already cost a failed run when hardcoded from memory instead of read from the
reference:

  * LEVELSET_FIELD=LEVEL, not 'LevelSet'. ⚠️ The worst failure mode, because
    run_tracking.py only WARNS on a missing level-set and then DISABLES the mask:
    the tool region stops being enforced and particles pass through the pin. A
    silent physics change that still exits 0.
  * SEED_SOURCE=grid needs --seed-box AND --seed-grid. --seed-fraction belongs to
    'grid-frac'. The mismatch fails at the seeding stage, 13 minutes into the
    mesh load.
  * CASE_STEM / MESH_SUBDIR: D2.gid contains A2_*.pvtu, and auto-detection
    guessed 'post/0eule/D2_*.pvtu'.
  * B2 uses DT=0.0003 and VEL_END=201; every other case uses 0.0003629 and 199.
  * OUTPUT_TARGET=case-flash writes INTO the case folder, which is read-only for
    us. Must be overridden to 'scratch'.

So: never retype these. Derive them, then run this check.

WHAT COUNTS AS AN ACCEPTABLE DIFFERENCE
---------------------------------------
Run SIZE is expected to differ (a smoke test is not a production run), as are the
output destination and anything added for the indicators. Those keys are listed in
EXPECTED_DIFFERENT and reported separately rather than as failures. Everything
else — the physics, the search policy, the field names, the seeding geometry — must
match, and a mismatch there is a bug.
"""
from __future__ import annotations

import argparse
import re
import sys

# Must match the reference exactly: physics, search policy, fields, seed geometry.
MUST_MATCH = [
    "PRECISION", "VEL_START", "VELOCITY_FIELD", "LEVELSET_FIELD", "DT",
    "SEED_SOURCE", "SEED_BOX", "SEED", "RK4_MODE", "L1_METHOD",
    "L2_NEIGHBORHOOD", "L0_SKIP_BAND", "ENHANCED_SEARCH_BAND",
    "BOUNDARY_PROJ_TOL", "POINT_IN_TET_TOL", "LEVELSET_MODE",
    "FAILED_SUBSTAGE", "INTERPOLATION_METHOD", "PIN_RPM", "PIN_CENTER",
    "PIN_AXIS", "PIN_TILT", "N_GROUPS", "BOUNDARY_WALLS", "MESH_SUBDIR",
    "FEMUSS_SUBDIR", "REGISTRATION", "EXACT_TET_BOX", "MAX_ELEMS_PER_CELL",
]

# Allowed to differ, with the reason. Reported, never a failure.
EXPECTED_DIFFERENT = {
    "N_STEPS": "run size (smoke vs full)",
    "VEL_END": "run size (fewer snapshots in a smoke test)",
    "SEED_GRID": "run size (coarser grid in a smoke test)",
    "EXPORT_FREQ": "export cadence",
    "LOG_INTERVAL": "log verbosity",
    "MESH_LOAD_WORKERS": "host-dependent",
    "OUTPUT_TARGET": "⚠️ MUST be overridden: case-flash writes into the "
                     "read-only case folder",
    "OUTPUT_CASE_SUBFOLDER": "output destination",
    "CASE_STEM": "⚠️ pinned deliberately where auto-detection fails "
                 "(e.g. D2.gid holds A2_*.pvtu)",
    "OCTREE_CACHE": "cache location is host-dependent",
    "OCTREE_CACHE_IN_CASE": "⚠️ turned ON deliberately: the octree cache is the "
                            "one artefact allowed in the case folder, and "
                            "belongs there so every run of that case shares "
                            "one 963s build",
    "OCTREE_CACHE_FORMAT": "hdf5 chosen deliberately: one self-contained .h5 "
                           "per entry keeps the case folder tidy; falls back "
                           "to a .npy dir if h5py is missing",
    "XLA_PREALLOC": "⚠️ HOST-DEPENDENT, and the reference's value is WRONG for a "
                    "32 GB card. PREALLOCATE=true claims ~75 % of HBM "
                    "regardless of need (measured: 24.65 GB claimed for 7.65 GB "
                    "of arrays), leaving too little for the RK4 kernel's "
                    "compilation -- two D2 runs died at stage 6/7. LUMI's "
                    "MI250X has 64 GB so 75 % still leaves ~48 GB. Use 0 "
                    "(on-demand, measured 7.84 GB) on the workstation.",
    "SIF": "container, not used off LUMI",
    "JAXTRACE": "repo path differs per host",
    "PKGS": "package path differs per host",
    "PROJECT": "not meaningful off LUMI",
    "N_PARTICLES": "ignored by the grid seed sources",
    "SEED_FRACTION": "unused when SEED_SOURCE=grid",
}


def value(text: str, key: str) -> str | None:
    """Right-hand side of `KEY=...`, with a ${KEY:-default} unwrapped."""
    m = re.search(rf'^{re.escape(key)}=("[^"]*"|\'[^\']*\'|\S*)', text, flags=re.M)
    if not m:
        return None
    v = m.group(1).strip('"').strip("'")
    # ${VAR:-default} / ${VAR-default} -> default, which is what a run uses
    # when the variable is not exported.
    m2 = re.match(r'^\$\{[A-Za-z_][A-Za-z0-9_]*:?-(.*)\}$', v)
    if m2:
        v = m2.group(1)
    v = re.sub(r'\s+#.*$', '', v).strip()
    return v


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True,
                    help="the case's own run_jaxtrace.sh (authoritative)")
    ap.add_argument("--candidate", required=True, help="our run script")
    ap.add_argument("--strict", action="store_true",
                    help="also fail on keys present in the reference but absent "
                         "from the candidate")
    args = ap.parse_args()

    try:
        ref = open(args.reference).read()
        cand = open(args.candidate).read()
    except OSError as exc:
        print(f"  cannot read: {exc}")
        return 2

    fails, notes, missing = [], [], []
    for k in MUST_MATCH:
        r, c = value(ref, k), value(cand, k)
        if r is None:
            continue
        if c is None:
            missing.append((k, r))
        elif r != c:
            fails.append((k, r, c))

    for k, why in EXPECTED_DIFFERENT.items():
        r, c = value(ref, k), value(cand, k)
        if r is not None and c is not None and r != c:
            notes.append((k, r, c, why))

    w = max([len(k) for k in MUST_MATCH] + [12])
    if fails:
        print(f"  ❌ {len(fails)} CONFIG MISMATCH(ES) — these must match the "
              f"reference:\n")
        for k, r, c in fails:
            print(f"    {k:<{w}} reference: {r!r}")
            print(f"    {'':<{w}} candidate: {c!r}")
        print()
    else:
        print("  ✅ every must-match key agrees with the reference\n")

    if missing:
        print(f"  ⚠️  {len(missing)} key(s) in the reference but not in the "
              f"candidate:")
        for k, r in missing:
            print(f"    {k:<{w}} reference: {r!r}")
        print("    (fine if the candidate relies on run_tracking.py's default; "
              "a problem if\n     the reference set it deliberately — check "
              "each one)\n")

    if notes:
        print(f"  ℹ️  {len(notes)} expected difference(s):")
        for k, r, c, why in notes:
            print(f"    {k:<{w}} {r!r} -> {c!r}   ({why})")
        print()

    rc = 1 if fails else 0
    if args.strict and missing:
        rc = 1
    print("  RESULT:", "FAIL" if rc else "PASS")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
