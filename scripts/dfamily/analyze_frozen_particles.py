#!/usr/bin/env python3
"""
D-family frozen-particle diagnostic.

Answers one question: when a particle stops moving, does it still have a
valid host element?

  ElementID >= 0 while frozen  ->  the SEARCH is fine; the velocity it was
                                   given was zero (level set, or a genuinely
                                   stagnant velocity field).
  ElementID  < 0 while frozen  ->  the SEARCH failed; no host was found.

Run this on a tracking output directory that contains per-step particle
VTU/VTKHDF files. ElementID is only present when the run used
--export-element-ids; without it the script still reports the freeze
statistics and tells you what is missing.

Usage
-----
  python analyze_frozen_particles.py <run_dir> [--level-set <mesh.pvtu>]
                                     [--out <report.md>] [--max-steps N]

Example
-------
  python scripts/dfamily/analyze_frozen_particles.py \
      /scratch/project_465002752/hashemia/dfamily/D2_diag \
      --level-set /scratch/.../D2.gid/post/A2_34.pvtu \
      --out docs/dfamily/02_D2_frozen_report.md
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

FROZEN_TOL_MM = 1e-6   # displacement below this counts as "not moving"


# ----------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------
def _require_vtk():
    try:
        import vtk  # noqa: F401
        from vtk.util.numpy_support import vtk_to_numpy  # noqa: F401
    except ImportError:
        sys.exit("ERROR: this script needs VTK. On LUMI run it inside the "
                 "same singularity image used for tracking.")
    return sys.modules["vtk"], sys.modules["vtk.util.numpy_support"].vtk_to_numpy


def list_steps(run_dir: Path):
    """Return sorted (step, path) for every per-step particle file."""
    out = []
    for f in os.listdir(run_dir):
        m = re.match(r"particles_step_(\d+)\.vtu$", f)
        if m:
            out.append((int(m.group(1)), run_dir / f))
    return sorted(out)


def read_step(path: Path):
    """Return (positions_mm, element_ids_or_None)."""
    vtk, vtk_to_numpy = _require_vtk()
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData()) * 1000.0      # m -> mm
    arr = g.GetPointData().GetArray("ElementID")
    eid = vtk_to_numpy(arr).astype(np.int64) if arr is not None else None
    return pts, eid


def read_levelset(pvtu: Path, sample_pieces: int | None = None):
    """Return (points_mm, LEVEL) for a mesh file, or (None, None)."""
    vtk, vtk_to_numpy = _require_vtk()
    import glob as _glob
    pieces = sorted(_glob.glob(str(pvtu).replace(".pvtu", "_*.vtu")))
    if not pieces:                                   # single-piece fallback
        pieces = [str(pvtu)]
    if sample_pieces:
        pieces = pieces[:sample_pieces]
    P, L = [], []
    for fn in pieces:
        rd = (vtk.vtkXMLUnstructuredGridReader() if fn.endswith(".vtu")
              else vtk.vtkXMLPUnstructuredGridReader())
        rd.SetFileName(fn)
        rd.Update()
        g = rd.GetOutput()
        a = g.GetPointData().GetArray("LEVEL")
        if a is None:
            continue
        P.append(vtk_to_numpy(g.GetPoints().GetData()) * 1000.0)
        L.append(vtk_to_numpy(a))
    if not P:
        return None, None
    return np.vstack(P), np.concatenate(L)


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------
def tool_radius_by_depth(pts, lev, bands):
    """Max radius of the LEVEL<0 (tool) region within each depth band."""
    inside = lev < 0
    r = np.hypot(pts[:, 0], pts[:, 1])
    out = {}
    for lo, hi in bands:
        m = inside & (pts[:, 2] >= lo) & (pts[:, 2] < hi)
        out[(lo, hi)] = (float(r[m].max()), int(m.sum())) if m.any() else (np.nan, 0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="tracking output directory")
    ap.add_argument("--level-set", type=Path, default=None,
                    help="mesh .pvtu carrying the LEVEL field (optional)")
    ap.add_argument("--level-set-pieces", type=int, default=8,
                    help="how many mesh pieces to sample for LEVEL (default 8)")
    ap.add_argument("--out", type=Path, default=None, help="write a markdown report here")
    ap.add_argument("--early-step", type=int, default=100,
                    help="step used to detect dead-on-arrival (default 100)")
    args = ap.parse_args()

    steps = list_steps(args.run_dir)
    if len(steps) < 3:
        sys.exit(f"ERROR: found {len(steps)} particle files in {args.run_dir}; need >= 3.")

    s0, f0 = steps[0]
    sE, fE = min(steps, key=lambda t: abs(t[0] - args.early_step))
    sF, fF = steps[-1]
    sP, fP = steps[-2]

    L = []
    def say(line=""):
        print(line)
        L.append(line)

    say(f"# Frozen-particle diagnostic — `{args.run_dir.name}`")
    say()
    say(f"Steps found: {len(steps)}  (first {s0}, last {sF})")
    say()

    p0, e0 = read_step(f0)
    pE, eE = read_step(fE)
    pF, eF = read_step(fF)
    pP, _ = read_step(fP)
    n = len(p0)

    doa = np.linalg.norm(pE - p0, axis=1) < FROZEN_TOL_MM
    fin = np.linalg.norm(pF - pP, axis=1) < FROZEN_TOL_MM
    late = fin & ~doa

    say("## 1. How many stop, and when")
    say()
    say("| population | count | % |")
    say("|---|---:|---:|")
    say(f"| dead on arrival (step {s0}->{sE}) | {doa.sum():,} | {100*doa.mean():.2f} |")
    say(f"| froze later | {late.sum():,} | {100*late.mean():.2f} |")
    say(f"| frozen at final step | {fin.sum():,} | {100*fin.mean():.2f} |")
    say(f"| total particles | {n:,} | 100.00 |")
    say()

    # ---- the decisive test -------------------------------------------------
    say("## 2. Do frozen particles still have a host element?")
    say()
    if eF is None:
        say("**ElementID is NOT present in these files.**")
        say()
        say("Re-run the case with `EXPORT_ELEMENT_IDS=1` in `run_jaxtrace.sh`.")
        say("Without it this question cannot be answered: the freeze statistics")
        say("above are valid, but search-failure and zero-velocity are")
        say("indistinguishable.")
        say()
    else:
        lost = eF < 0
        say("| group | n | ElementID < 0 (search failed) | ElementID >= 0 (velocity was zero) |")
        say("|---|---:|---:|---:|")
        for name, m in (("dead on arrival", doa), ("froze later", late),
                        ("frozen (all)", fin), ("still moving", ~fin)):
            if m.sum() == 0:
                continue
            nl = int((m & lost).sum())
            say(f"| {name} | {m.sum():,} | {nl:,} ({100*nl/m.sum():.1f}%) "
                f"| {m.sum()-nl:,} ({100*(m.sum()-nl)/m.sum():.1f}%) |")
        say()
        nl_late = int((late & lost).sum())
        if late.sum():
            frac = 100 * nl_late / late.sum()
            say(f"**Verdict for the late freezers:** {frac:.1f}% have no host element.")
            if frac > 70:
                say("Dominated by **search failure** — MALMO/L0-L1-L2 is losing them.")
                say("Act on `ENHANCED_SEARCH_BAND`, `L0_SKIP_BAND`, `L2_NEIGHBORHOOD`.")
            elif frac < 30:
                say("Dominated by **zero velocity** — the host is found, the")
                say("velocity given to it is zero. Search settings will NOT help;")
                say("look at the level set (`LEVELSET_MODE`) and the velocity field.")
            else:
                say("**Mixed** — both mechanisms are active; treat them separately")
                say("by depth and radius using section 3.")
        say()

    # ---- geometry ----------------------------------------------------------
    say("## 3. Where they stop")
    say()
    z = pF[:, 2]
    r = np.hypot(pF[:, 0], pF[:, 1])
    bands = [(-6, -4), (-4, -2), (-2, 0), (0, 2)]

    tool = None
    if args.level_set:
        pts, lev = read_levelset(args.level_set, args.level_set_pieces)
        if pts is not None:
            tool = tool_radius_by_depth(pts, lev, bands)
            say(f"Tool region (`LEVEL < 0`) from `{args.level_set.name}` "
                f"({args.level_set_pieces} pieces sampled):")
            say()
            say("| depth (mm) | tool radius (mm) | nodes |")
            say("|---|---:|---:|")
            for b in bands:
                rr, cnt = tool[b]
                say(f"| {b[0]} .. {b[1]} | {rr:.2f} | {cnt:,} |")
            say()
        else:
            say(f"(no LEVEL field found in {args.level_set})")
            say()

    say("Late freezers by depth:")
    say()
    hdr = "| depth (mm) | n | median r (mm) |"
    sep = "|---|---:|---:|"
    if tool:
        hdr += " inside tool |"
        sep += "---:|"
    say(hdr); say(sep)
    for b in bands:
        m = late & (z >= b[0]) & (z < b[1])
        if not m.sum():
            continue
        row = f"| {b[0]} .. {b[1]} | {m.sum():,} | {np.median(r[m]):.2f} |"
        if tool:
            rr = tool[b][0]
            ins = int((r[m] <= rr).sum()) if np.isfinite(rr) else 0
            row += f" {ins:,} ({100*ins/m.sum():.1f}%) |"
        say(row)
    say()

    hist, edges = np.histogram(r[late], bins=np.arange(0, 21, 1))
    say("Radial histogram of late freezers (1 mm bins):")
    say()
    say("```")
    mx = max(hist.max(), 1)
    for i, c in enumerate(hist):
        if c:
            say(f"  {edges[i]:4.0f}-{edges[i+1]:2.0f} mm {c:7,d} {'#'*int(50*c/mx)}")
    say("```")
    say()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"\n[report written to {args.out}]")


if __name__ == "__main__":
    main()
