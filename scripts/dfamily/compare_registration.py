#!/usr/bin/env python3
"""
Compare registration variants of the D-family host-loss sweep.

Answers, per variant, the one question that separates a REGISTRATION defect
from a SEARCH defect:

    aabb ~0 frozen, parent_cube loses thousands
        -> registration coverage is the root cause. The search kernel is
           fine; it was being handed an octree that did not list the element.
    aabb still loses particles in the same r 7-9 mm annulus
        -> the defect is in the search, and the band settings matter.

Context from the revised paper (paper_table_found.md, Table T2), measured over
the 10-mesh R2-6 cohort INCLUDING fully non-Kuhn meshes:

    MALMO aabb          100.00% correct on all 10
    MALMO centroid       99.64% worst case
    MALMO vertex_multi   96.12% worst case

Usage
-----
  compare_registration.py --run aabb=/path/aabb_results \
                          --run parent_cube=/path/parent_cube_results \
                          [--level-set mesh.pvtu] [--out report.md]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

FROZEN_TOL_MM = 1e-6


def _vtk():
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        sys.exit("ERROR: needs VTK. On LUMI run inside the tracking singularity image.")
    return vtk, vtk_to_numpy


def list_steps(run_dir: Path):
    """Particle files, searching one level down for the RUN_TAG subdirectory."""
    pat = re.compile(r"particles_step_(\d+)\.vtu$")

    def scan(d: Path):
        try:
            names = os.listdir(d)
        except OSError:
            return []
        return [(int(m.group(1)), d / n) for n in names if (m := pat.match(n))]

    out = scan(run_dir)
    if not out:
        for sub in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            out = scan(sub)
            if out:
                break
    return sorted(out)


def read_step(path: Path):
    vtk, vtk_to_numpy = _vtk()
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData()) * 1000.0      # m -> mm
    a = g.GetPointData().GetArray("ElementID")
    eid = vtk_to_numpy(a).astype(np.int64) if a is not None else None
    return pts, eid


def read_levelset(pvtu: Path, pieces: int = 8):
    """Return (points_mm, LEVEL) sampled from the first `pieces` sub-files."""
    vtk, vtk_to_numpy = _vtk()
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(pvtu).getroot()
        srcs = [p.get("Source") for p in root.iter("Piece")][:pieces]
    except Exception:
        return None, None
    P, L = [], []
    for s in srcs:
        f = pvtu.parent / s
        if not f.exists():
            continue
        r = vtk.vtkXMLUnstructuredGridReader()
        r.SetFileName(str(f))
        r.Update()
        g = r.GetOutput()
        arr = g.GetPointData().GetArray("LEVEL")
        if arr is None:
            continue
        P.append(vtk_to_numpy(g.GetPoints().GetData()) * 1000.0)
        L.append(vtk_to_numpy(arr))
    if not P:
        return None, None
    return np.vstack(P), np.concatenate(L)



def read_stage_logs(run_dir: Path):
    """Read hit_stats.csv / search_stats.csv, searching one level down too.

    hit_stats.csv classifies the surviving particles at each log step by which
    search level would find them: L0 (cached element), L1 (neighbour hop),
    L2 (global octree) or miss. That is what identifies WHICH stage fails.
    search_stats.csv carries per-step n_lost / new_lost.
    """
    def find(stem):
        for d in (run_dir, *(p for p in sorted(run_dir.iterdir()) if p.is_dir())):
            c = d / stem
            if c.is_file():
                return c
        return None

    out = {}
    hs = find("hit_stats.csv")
    if hs:
        rows = [l.strip().split(",") for l in hs.read_text().splitlines() if l.strip()]
        if len(rows) > 1:
            hdr, body = rows[0], rows[1:]
            idx = {k: i for i, k in enumerate(hdr)}
            def col(name):
                i = idx.get(name)
                return [float(r[i]) for r in body if i is not None and i < len(r)]
            out["hit"] = dict(
                steps=col("step"), l0=col("l0_hits"), l1=col("l1_hits"),
                l2=col("l2_hits"), miss=col("miss"), n_active=col("n_active"),
            )
    ss = find("search_stats.csv")
    if ss:
        rows = [l.strip().split(",") for l in ss.read_text().splitlines() if l.strip()]
        if len(rows) > 1:
            out["search"] = [(int(r[0]), int(r[2]), int(r[3]))
                             for r in rows[1:] if len(r) >= 4 and r[0].isdigit()]
    return out


def analyse(run_dir: Path):
    """Per-variant summary. Returns None when the run produced nothing usable."""
    steps = list_steps(run_dir)
    if len(steps) < 2:
        return None

    (_, f0), (sF, fF) = steps[0], steps[-1]
    p0, e0 = read_step(f0)
    pF, eF = read_step(fF)
    pP, _ = read_step(steps[-2][1])

    n = len(p0)
    moved_last = np.linalg.norm(pF - pP, axis=1)
    frozen = moved_last < FROZEN_TOL_MM
    lost = (eF < 0) if eF is not None else np.zeros(n, bool)
    seeded_ok = (e0 >= 0) if e0 is not None else np.ones(n, bool)

    # Only particles that HAD a host at step 0 are meaningful here: anything
    # seeded inside the tool or outside the mesh never entered the experiment.
    valid = seeded_ok
    r = np.hypot(pF[:, 0], pF[:, 1])
    z = pF[:, 2]

    # When was each lost particle lost? The first step at which its id goes
    # negative tells us whether loss is immediate or develops with travel.
    # A production run with EXPORT_FREQ=1 and N_STEPS=8000 leaves 8,000 files
    # of ~10 MB each. Reading every one costs ~80 GB of Lustre traffic and
    # hours, so subsample to at most MAX_TIMING_READS evenly spaced steps.
    # The onset step is then known only to within one sampling interval, which
    # is reported so the resolution is not mistaken for exact.
    MAX_TIMING_READS = 60
    first_loss = None
    timing_stride = 1
    if eF is not None and lost.any():
        if len(steps) > MAX_TIMING_READS:
            timing_stride = (len(steps) + MAX_TIMING_READS - 1) // MAX_TIMING_READS
            sel = steps[::timing_stride]
            if sel[-1] != steps[-1]:
                sel = sel + [steps[-1]]
        else:
            sel = steps
        prev = None
        hits = []
        for st, f in sel:
            _, e = read_step(f)
            if e is None:
                continue
            cur = e < 0
            if prev is not None:
                new = cur & ~prev
                if new.sum():
                    hits.append((st, int(new.sum())))
            prev = cur
        first_loss = hits

    stage = read_stage_logs(run_dir)

    return dict(
        stage=stage,
        n=n, n_valid=int(valid.sum()), last_step=sF, n_files=len(steps),
        frozen=int((frozen & valid).sum()),
        lost=int((lost & valid).sum()),
        frozen_hosted=int((frozen & ~lost & valid).sum()),
        lost_moving=int((lost & ~frozen & valid).sum()),
        r_lost=r[lost & valid], z_lost=z[lost & valid],
        first_loss=first_loss,
        timing_stride=timing_stride,
        has_eid=eF is not None,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR",
                    help="variant name and its results dir (repeatable)")
    ap.add_argument("--level-set", type=Path, default=None,
                    help="mesh .pvtu carrying LEVEL, for the tool radius")
    ap.add_argument("--level-set-pieces", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    L = []
    def say(line=""):
        print(line)
        L.append(line)

    runs = []
    for spec in args.run:
        if "=" not in spec:
            sys.exit(f"ERROR: --run expects NAME=DIR, got '{spec}'")
        name, d = spec.split("=", 1)
        runs.append((name, Path(d)))

    say("# Registration sweep — does coverage or search explain the host loss?")
    say()
    say("Reference, revised paper Table T2 (10-mesh R2-6 cohort, including the")
    say("fully non-Kuhn Bunny / Porous media / Microfluidics meshes):")
    say()
    say("| registration | worst correct% in cohort |")
    say("|---|---:|")
    say("| `aabb` | **100.00%** |")
    say("| `centroid` (parent_cube) | 99.64% |")
    say("| `vertex_multi` | 96.12% |")
    say()
    say("The D-family meshes are 50.6–59.3% non-Kuhn, against 0.06% for the")
    say("FSW mesh the earlier benchmark used — squarely in the regime where")
    say("the registration choice is expected to matter.")
    say()

    results = {}
    for name, d in runs:
        if not d.is_dir():
            say(f"- `{name}`: **no results** at `{d}`")
            continue
        res = analyse(d)
        if res is None:
            say(f"- `{name}`: fewer than 2 particle files in `{d}` — run incomplete")
            continue
        results[name] = res

    if not results:
        say()
        say("Nothing to compare.")
        _write(args.out, L)
        return

    say("## 1. Host loss by registration")
    say()
    say("| variant | seeded w/ host | frozen | no host (`ElementID<0`) | frozen BUT hosted | lost but moving |")
    say("|---|---:|---:|---:|---:|---:|")
    for name, r in results.items():
        pct = 100 * r["lost"] / r["n_valid"] if r["n_valid"] else 0.0
        say(f"| `{name}` | {r['n_valid']:,} | {r['frozen']:,} | "
            f"{r['lost']:,} ({pct:.2f}%) | {r['frozen_hosted']:,} | {r['lost_moving']:,} |")
    say()
    say("`frozen BUT hosted` is the control column: a particle that stopped")
    say("while still holding a valid host would mean zero velocity, not a")
    say("search failure. It should stay at or near zero.")
    say()

    # ---- the verdict ----------------------------------------------------
    say("## 2. Verdict")
    say()
    if "aabb" in results:
        a = results["aabb"]
        a_pct = 100 * a["lost"] / a["n_valid"] if a["n_valid"] else 0.0
        others = {k: v for k, v in results.items() if k != "aabb"}
        worst = max((100 * v["lost"] / v["n_valid"] if v["n_valid"] else 0.0)
                    for v in others.values()) if others else 0.0
        say(f"`aabb` loses {a['lost']:,} of {a['n_valid']:,} ({a_pct:.2f}%).")
        if others:
            say(f"Worst other variant loses {worst:.2f}%.")
        say()
        if a_pct < 0.01 and worst > 0.1:
            say("**Registration coverage is the root cause.** AABB-overlap")
            say("registration eliminates the loss, so the search kernel was")
            say("correct all along — it was being given an octree that did not")
            say("list the element containing the particle.")
            say()
            say("Fix: run the D family with `--registration aabb`. The search")
            say("band settings (`ENHANCED_SEARCH_BAND`, `L0_SKIP_BAND`,")
            say("`L2_NEIGHBORHOOD`) are **not** the lever — widening a search")
            say("cannot find an element that is absent from the cells visited.")
        elif a_pct >= 0.1:
            say("**Not explained by registration alone.** `aabb` is")
            say("coverage-complete by construction, so a residual loss here")
            say("points at the SEARCH or at the velocity/geometry handling.")
            say("Compare the annulus in section 3: if the surviving losses sit")
            say("in the same r 7–9 mm ring, investigate the search; if they")
            say("have moved, suspect the boundary or level-set handling.")
        else:
            say("**Partially explained.** `aabb` reduces but does not")
            say("eliminate the loss. Treat the residue separately, using the")
            say("per-variant geometry below.")
    else:
        say("`aabb` was not run, so the decisive comparison is missing.")
        say("Re-run with `--only aabb,parent_cube` at minimum.")
    say()

    # ---- geometry -------------------------------------------------------
    say("## 3. Where the surviving losses are")
    say()
    tool_r = None
    if args.level_set:
        pts, lev = read_levelset(args.level_set, args.level_set_pieces)
        if pts is not None:
            m = (lev < 0) & (pts[:, 2] >= -2) & (pts[:, 2] < 0)
            if m.any():
                tool_r = float(np.hypot(pts[m, 0], pts[m, 1]).max())
                say(f"Tool radius in z ∈ [-2,0] mm: **{tool_r:.2f} mm** "
                    f"(from `{args.level_set.name}`).")
                say()

    say("| variant | n lost | r median | r 5–95% | z median | inside tool |")
    say("|---|---:|---:|---|---:|---:|")
    for name, r in results.items():
        if not len(r["r_lost"]):
            say(f"| `{name}` | 0 | — | — | — | — |")
            continue
        rl, zl = r["r_lost"], r["z_lost"]
        lo, hi = np.percentile(rl, [5, 95])
        ins = int((rl <= tool_r).sum()) if tool_r else 0
        ins_s = f"{ins:,} ({100*ins/len(rl):.1f}%)" if tool_r else "n/a"
        say(f"| `{name}` | {len(rl):,} | {np.median(rl):.2f} | "
            f"{lo:.2f}–{hi:.2f} | {np.median(zl):.2f} | {ins_s} |")
    say()
    if tool_r:
        say("A loss ring sitting OUTSIDE the tool radius cannot be the level")
        say("set zeroing the velocity — that only acts where `LEVEL < 0`.")
        say()

    # ---- timing ---------------------------------------------------------
    say("## 4. Which search stage fails (hit_stats.csv)")
    say()
    say("At each log step every surviving particle is classified by which")
    say("level would find it on a cold lookup: L0 = cached element, L1 =")
    say("neighbour hop, L2 = global octree, miss = none of them. A rising")
    say("`miss` share is the search giving up; a high L2 share means L0/L1")
    say("are not retaining the host and every step pays for a global search.")
    say()
    any_hit = False
    for name, r in results.items():
        h = r.get("stage", {}).get("hit")
        if not h or not h["steps"]:
            continue
        any_hit = True
        say(f"**`{name}`** — L0/L1/L2/miss share at first, middle and last log step:")
        say()
        say("| step | n_active | L0 | L1 | L2 | miss |")
        say("|---|---:|---:|---:|---:|---:|")
        idxs = sorted({0, len(h["steps"]) // 2, len(h["steps"]) - 1})
        for i in idxs:
            tot = max(h["l0"][i] + h["l1"][i] + h["l2"][i] + h["miss"][i], 1)
            say(f"| {int(h['steps'][i])} | {int(h['n_active'][i]):,} | "
                f"{100*h['l0'][i]/tot:.1f}% | {100*h['l1'][i]/tot:.1f}% | "
                f"{100*h['l2'][i]/tot:.1f}% | **{100*h['miss'][i]/tot:.2f}%** |")
        say()
        peak = max(range(len(h["steps"])), key=lambda i: h["miss"][i])
        say(f"Peak miss: {int(h['miss'][peak]):,} at step {int(h['steps'][peak])}.")
        say()
    if not any_hit:
        say("_No hit_stats.csv found. Re-run the sweep with `--hit-stats-log`")
        say("(the sweep adds it automatically) to get the per-stage breakdown._")
        say()

    say("## 5. When the losses happen")
    say()
    say("With seeding concentrated in the annulus, loss should begin almost")
    say("immediately. A late onset would mean the particles still have to")
    say("travel to reach the failure region.")
    say()
    for name, r in results.items():
        fl = r["first_loss"]
        if not fl:
            say(f"- `{name}`: no losses recorded.")
            continue
        first, n_first = fl[0]
        total = sum(c for _, c in fl)
        stride = r.get("timing_stride", 1)
        res = "" if stride == 1 else (f" (sampled every {stride} exported "
                                      f"steps, so the onset is known only to "
                                      f"within that interval)")
        say(f"- `{name}`: first loss at or before step **{first}** "
            f"(+{n_first:,}); {len(fl)} sampled points saw new losses, "
            f"{total:,} in total{res}.")
    say()

    _write(args.out, L)


def _write(out: Path | None, L: list[str]):
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"\n[report written to {out}]")


if __name__ == "__main__":
    main()
