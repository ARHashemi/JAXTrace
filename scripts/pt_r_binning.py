"""
pt_r_binning.py — per-r-bin displacement statistics from compare VTUs.

Consumes one or more VTU files produced by compare_rom_vs_fom_tracking.py
(--out-vtu) and reports displacement statistics binned by radial
position r = sqrt(x**2 + y**2) computed from the FOM positions at the
reporting step stored as the VTU point coordinates.  This closes the
"spatial breakdown" deliverable of the ROM PT roadmap sections 2 and 3
(near-pin r <= 0.010 m vs outer-domain r > 0.010 m); the compare tool
itself only emits per-particle displacement and aggregate statistics.

Subset selection (--subset):

    all       (default) every particle in the archive.  Because the
              trackers apply ballistic + inlet extensions when a
              particle leaves the mesh interior, an "escaped" particle
              is not gone from the physics -- its trajectory continues
              on the extension.  Restricting statistics to both-alive
              hides that ballistic-tail residual and biases the near-pin
              bin (particles that reached the outlet are dropped from
              the outer bin).  So "all" is now the default; use
              "both_alive" only to reproduce older numbers.
    both_alive  particles with fom_escaped == 0 AND rom_escaped == 0.

Trapped-particle count: for the near-pin bin (r <= 0.010 m) we
additionally report the number of particles for which BOTH FOM and ROM
final positions are still inside the near-pin cylinder.  For a
divergence-free flow with proper inlet/outlet handling this should be
approximately zero -- every seeded streamline should pass around the
tool and reach the exit.  Trapping is therefore a Lagrangian proxy for
how div-free the numerical tracker's velocity field behaves.

CSV columns emitted (one row per bin per VTU):

    vtu, subset, r_lo, r_hi, n_total, n_used, n_bin,
    mean, median, rms, p95, p99, max,
    rms_x, rms_y, rms_z, rms_rel_fom_diag,
    n_trapped_fom, n_trapped_rom, n_trapped_both

`rms_rel_fom_diag` normalises against the FOM bounding-box diagonal,
matching the "rms / FOM diagonal" line the compare tool prints so the
percentages are directly comparable.  Trapped columns are non-zero only
on the near-pin (r_lo == 0) row.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def _read_vtu(path: Path):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    ug = r.GetOutput()
    if ug is None or ug.GetNumberOfPoints() == 0:
        raise RuntimeError(f"{path}: empty or unreadable")
    fom_pos = vtk_to_numpy(ug.GetPoints().GetData()).astype(np.float64)
    pd = ug.GetPointData()

    def _get(name):
        arr = pd.GetArray(name)
        if arr is None:
            raise KeyError(f"{path}: array '{name}' missing")
        return vtk_to_numpy(arr)

    disp_vec = _get("displacement_vec").astype(np.float64)
    return {
        "fom_pos":          fom_pos,
        "rom_pos":          fom_pos + disp_vec,       # rom = fom + (rom-fom)
        "displacement_vec": disp_vec,
        "displacement_mag": _get("displacement_mag").astype(np.float64),
        "fom_escaped":      _get("fom_escaped").astype(np.float32),
        "rom_escaped":      _get("rom_escaped").astype(np.float32),
    }


def _bin_stats(disp_vec, disp_mag, mask, fom_span):
    n = int(mask.sum())
    if n == 0:
        return {"n_bin": 0, "mean": np.nan, "median": np.nan, "rms": np.nan,
                "p95": np.nan, "p99": np.nan, "max": np.nan,
                "rms_x": np.nan, "rms_y": np.nan, "rms_z": np.nan,
                "rms_rel_fom_diag": np.nan}
    dm = disp_mag[mask]
    d = disp_vec[mask]
    rms = float(np.sqrt((dm ** 2).mean()))
    return {
        "n_bin": n,
        "mean":   float(dm.mean()),
        "median": float(np.median(dm)),
        "rms":    rms,
        "p95":    float(np.percentile(dm, 95)),
        "p99":    float(np.percentile(dm, 99)),
        "max":    float(dm.max()),
        "rms_x":  float(np.sqrt((d[:, 0] ** 2).mean())),
        "rms_y":  float(np.sqrt((d[:, 1] ** 2).mean())),
        "rms_z":  float(np.sqrt((d[:, 2] ** 2).mean())),
        "rms_rel_fom_diag": 100.0 * rms / max(fom_span, 1e-30),
    }


def _process(path: Path, edges: np.ndarray, subset: str,
             r_near_pin: float) -> list[dict]:
    d = _read_vtu(path)
    both_alive = (d["fom_escaped"] == 0) & (d["rom_escaped"] == 0)

    if subset == "both_alive":
        subset_mask = both_alive
    elif subset == "all":
        subset_mask = np.ones_like(both_alive)
    else:
        raise ValueError(f"unknown subset: {subset}")

    n_total = int(both_alive.size)
    n_used = int(subset_mask.sum())

    # Bin by FOM final r.  We use FOM position for binning even when the
    # subset is "all" — that keeps the binning consistent across
    # subsets, and using ROM position would put ROM-drift artefacts
    # into the wrong bin.
    r_fom = np.sqrt(d["fom_pos"][:, 0] ** 2 + d["fom_pos"][:, 1] ** 2)
    r_rom = np.sqrt(d["rom_pos"][:, 0] ** 2 + d["rom_pos"][:, 1] ** 2)
    fom_span = float(
        np.linalg.norm(d["fom_pos"].max(0) - d["fom_pos"].min(0)))

    # Trapping — evaluated at the reporting step directly on the FOM
    # (resp. ROM) final position.  Since the seeding is upstream of the
    # tool and the run is long enough for particles to advect through
    # the tool region, a particle whose final r is still <= r_near_pin
    # under a truly incompressible extension of the field must have
    # failed to advect past the tool.
    n_trapped_fom = int((r_fom <= r_near_pin).sum())
    n_trapped_rom = int((r_rom <= r_near_pin).sum())
    n_trapped_both = int(((r_fom <= r_near_pin) &
                          (r_rom <= r_near_pin)).sum())

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (r_fom >= lo) & (r_fom < hi)
        row_mask = subset_mask & in_bin
        stats = _bin_stats(d["displacement_vec"], d["displacement_mag"],
                           row_mask, fom_span)
        row = {"vtu": str(path), "subset": subset,
               "r_lo": float(lo), "r_hi": float(hi),
               "n_total": n_total, "n_used": n_used}
        row.update(stats)
        # Trapped counts belong on the near-pin (r_lo == 0) row only.
        if lo == 0.0:
            row["n_trapped_fom"] = n_trapped_fom
            row["n_trapped_rom"] = n_trapped_rom
            row["n_trapped_both"] = n_trapped_both
        else:
            row["n_trapped_fom"] = -1
            row["n_trapped_rom"] = -1
            row["n_trapped_both"] = -1
        rows.append(row)
    return rows


def _fmt_edges(edges: np.ndarray) -> str:
    parts = []
    for e in edges:
        if not np.isfinite(e):
            parts.append("inf")
        else:
            parts.append(f"{e:.4g}")
    return "[" + ", ".join(parts) + "]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vtus", type=Path, nargs="+",
                    help="One or more compare VTU files.")
    ap.add_argument("--edges", type=float, nargs="+",
                    default=[0.0, 0.010, float("inf")],
                    help="r-bin edges in metres; N+1 values define N bins. "
                         "Default: [0.0, 0.010, inf] -> near-pin / outer.")
    ap.add_argument("--subset", choices=["all", "both_alive"], default="all",
                    help="Which particles enter the stats. 'all' (default) "
                         "keeps escaped particles too; they continue on "
                         "the ballistic/inlet extension of the tracker and "
                         "still carry meaningful ROM-vs-FOM residual. Use "
                         "'both_alive' to reproduce older numbers.")
    ap.add_argument("--r-near-pin", type=float, default=0.010,
                    help="Radius (m) defining the 'trapped near the tool' "
                         "region for the trapped-particle count. Default "
                         "0.010 m matches the near-pin bin edge.")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Also write the full table to this CSV.")
    args = ap.parse_args()

    edges = np.asarray(args.edges, dtype=np.float64)
    if edges.size < 2 or not np.all(np.diff(edges) > 0):
        print("ERROR: --edges must be a strictly increasing list of >= 2 "
              "values.", file=sys.stderr)
        return 2

    print(f"[pt_r_bin] edges: {_fmt_edges(edges)}  subset: {args.subset}  "
          f"r_near_pin: {args.r_near_pin:g} m")
    all_rows: list[dict] = []
    for p in args.vtus:
        if not p.exists():
            print(f"WARN: skipping missing {p}", file=sys.stderr)
            continue
        rows = _process(p, edges, args.subset, args.r_near_pin)
        all_rows.extend(rows)

    if not all_rows:
        print("ERROR: no VTUs processed.", file=sys.stderr)
        return 3

    fields = ["vtu", "subset", "r_lo", "r_hi", "n_total", "n_used", "n_bin",
              "mean", "median", "rms", "p95", "p99", "max",
              "rms_x", "rms_y", "rms_z", "rms_rel_fom_diag",
              "n_trapped_fom", "n_trapped_rom", "n_trapped_both"]

    # Stdout: short-form pretty table (one line per row).  Use the
    # grandparent directory name (e.g. rom_vs_fom_hct_on) plus the case
    # id extracted from the path so cohort tables are readable at a
    # glance -- the filename alone is often identical across cases.
    def _label(p: Path) -> str:
        parts = p.parts
        # ".../cylindrical_<case>.gid/post_pt_compare/<label>/<file>.vtu"
        gid = next((x for x in parts if x.endswith(".gid")), "?")
        label = p.parent.name
        return f"{gid[:-4]}/{label}"

    print()
    print(f"{'case/label':<50} {'r_lo':>7} {'r_hi':>7} {'n_bin':>8} "
          f"{'rms':>10} {'rms%diag':>9} {'trapFOM':>8} {'trapROM':>8}")
    for row in all_rows:
        lbl = _label(Path(row["vtu"]))
        r_hi = row["r_hi"]
        r_hi_s = "inf" if not np.isfinite(r_hi) else f"{r_hi:.4f}"
        trap_fom = "-" if row["n_trapped_fom"] < 0 else str(row["n_trapped_fom"])
        trap_rom = "-" if row["n_trapped_rom"] < 0 else str(row["n_trapped_rom"])
        print(f"{lbl:<50} {row['r_lo']:>7.4f} {r_hi_s:>7} "
              f"{row['n_bin']:>8} {row['rms']:>10.4e} "
              f"{row['rms_rel_fom_diag']:>8.3f}% "
              f"{trap_fom:>8} {trap_rom:>8}")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in all_rows:
                w.writerow(row)
        print(f"\n[pt_r_bin] wrote {args.out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
