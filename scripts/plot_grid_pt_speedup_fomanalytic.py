"""Grid PT vs mesh PT — wall-clock + trajectory-error comparison.

Reads:
    /scratch/shared/ROM/FOM_analytic/<case>/summary.json      (per-variant PT error)
    /scratch/shared/ROM/FOM_analytic/<case>/out_*/run_*.log   (per-variant wall time)

Writes:
    paper_figs/rom_pt_analytic/grid_vs_mesh_pt.{png,svg}
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASE_ROOTS = {
    "recirc_2026": Path("/home/arhashemi/fsw-gpu/scratch/shared/ROM/FOM_analytic/recirc_2026"),
    "rot_cyl_2026": Path("/home/arhashemi/fsw-gpu/scratch/shared/ROM/FOM_analytic/rot_cyl_2026"),
}
OUT_DIR = Path("/home/arhashemi/Workspace/welding/JAXTrace/paper_figs/rom_pt_analytic")


_WALL_RE = re.compile(r"Wall time:\s+([\d.]+)s")


def _extract_wall_from_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    txt = log_path.read_text()
    m = _WALL_RE.findall(txt)
    return float(m[-1]) if m else None


def _walltime_for(case_root: Path, variant: str) -> float | None:
    """Guess the wall time for a variant by scanning its out_*/ log."""
    # Grid variants have 'grid_' prefix in summary; strip for the fs.
    if variant.startswith("grid_"):
        d = case_root / f"out_grid_{variant[len('grid_'):]}"
        candidates = list(d.glob("run_grid.log")) + list(d.glob("*.log"))
    else:
        d = case_root / f"out_mesh_{variant}"
        candidates = list(d.glob("run_mesh_*.log"))
    for c in candidates:
        w = _extract_wall_from_log(c)
        if w is not None:
            return w
    return None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(CASE_ROOTS), figsize=(13, 5))

    for ax, (case, root) in zip(axes, CASE_ROOTS.items()):
        summ = root / "summary.json"
        if not summ.exists():
            ax.text(0.5, 0.5, f"missing summary.json\nfor {case}",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        data = json.load(summ.open())
        variants = data.get("variants", {})
        # Final-step RMS trajectory error per variant.
        rows = []
        for name, v in variants.items():
            rms = v.get("rms_err")
            if rms is None:
                continue
            r_final = rms[-1] if isinstance(rms, list) else rms
            wall = _walltime_for(root, name)
            rows.append((name, r_final, wall))
        rows.sort(key=lambda r: (r[2] if r[2] else 1e9))

        names = [r[0] for r in rows]
        rmss  = [r[1] for r in rows]
        walls = [r[2] if r[2] is not None else np.nan for r in rows]

        # colour: grid → green, mesh_hct → blue, other mesh → grey
        colours = []
        for n in names:
            if n.startswith("grid_"):
                colours.append("#3f8f3f")
            elif "hct" in n:
                colours.append("#1f77b4")
            else:
                colours.append("#888888")

        # Scatter wall time vs error
        for n, r, w, c in zip(names, rmss, walls, colours):
            if w is None or np.isnan(w):
                continue
            ax.scatter(w, r, s=160, color=c, edgecolor="black", zorder=3)
            ax.annotate(n, (w, r), xytext=(6, 4),
                        textcoords="offset points", fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("wall time (s), log")
        ax.set_ylabel("final-step RMS trajectory error (m), log")
        ax.set_title(case)
        ax.grid(which="both", alpha=0.3)

    fig.suptitle("Grid-based PT vs mesh-based PT — accuracy and speed "
                 "on the analytic reference cases",
                 y=1.02)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"grid_vs_mesh_pt.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"wrote {OUT_DIR / 'grid_vs_mesh_pt.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
