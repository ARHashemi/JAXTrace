"""
Lagrangian mixing diagnostics for §4 / §7 of the ROM PT roadmap.

Reads two JAXTrace VTKHDF particle archives (FOM and ROM), computes
two lightweight mixing-relevant diagnostics, and writes:

  * a CSV with the residence-time-in-annulus curve for each archive,
  * a CSV with the geometric-mean pairwise separation vs time for
    each archive plus the fitted exponential slope (proxy for the
    top FTLE eigenvalue),
  * an optional PNG summarising both curves if matplotlib is
    available.

Both diagnostics are post-hoc numpy on the exported particles.vtkhdf;
no new solver code required.

Usage:

    python3 scripts/lagrangian_mixing_diagnostics.py \\
        --fom-vtkhdf .../post_pt/fom_hct_on/<run>/particles.vtkhdf \\
        --rom-vtkhdf .../post_pt/rom_centered_hct_on/<run>/particles.vtkhdf \\
        --out-dir .../mixing_diag/

Reference: rom_pt_roadmap.md §§ 4, 7.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np


def _iter_positions_vtkhdf(vtkhdf_path: Path,
                           stride: int = 10,
                           n_particle_cap: int | None = None):
    """Yield (step, positions[stride-strided subset], escaped mask).

    Reads NumberOfPoints, PointOffsets, and (if present) the Escaped
    PointDataOffsets from the archive, then walks every `stride`-th
    step and yields the (positions, escaped) pair at that step.

    n_particle_cap: if not None, only the first N particles of each
    step are returned.  Useful for the pairwise-separation diagnostic
    which only needs a few hundred particles.
    """
    import h5py
    with h5py.File(str(vtkhdf_path), "r") as f:
        vtkhdf = f["VTKHDF"]
        n_pts_per_step = vtkhdf["NumberOfPoints"][:]
        pts_offsets = vtkhdf["Steps"]["PointOffsets"][:]
        n_steps = n_pts_per_step.shape[0]

        pd_offsets_esc = None
        if ("PointData" in vtkhdf
                and "Escaped" in vtkhdf["PointData"]):
            pd_offsets_esc = vtkhdf["Steps"]["PointDataOffsets"]["Escaped"][:]

        for step in range(0, n_steps, stride):
            start = int(pts_offsets[step])
            count = int(n_pts_per_step[step])
            if n_particle_cap is not None:
                count = min(count, n_particle_cap)
            pts = np.asarray(
                vtkhdf["Points"][start:start + count], dtype=np.float32,
            )
            if pd_offsets_esc is not None:
                esc_start = int(pd_offsets_esc[step])
                esc = np.asarray(
                    vtkhdf["PointData"]["Escaped"][
                        esc_start:esc_start + count
                    ],
                    dtype=np.uint8,
                )
            else:
                esc = np.zeros(count, dtype=np.uint8)
            yield step, pts, esc


def residence_time(vtkhdf_path: Path, r_min: float, r_max: float,
                   z_min: float, z_max: float, stride: int) -> np.ndarray:
    """Fraction of alive particles inside the annulus at each sampled step.

    Returns (n_samples, 3) with columns (step, n_alive, n_inside).
    """
    rows: list[tuple[int, int, int]] = []
    for step, pts, esc in _iter_positions_vtkhdf(vtkhdf_path, stride=stride):
        alive = esc == 0
        r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        z = pts[:, 2]
        inside = alive & (r >= r_min) & (r < r_max) & (z >= z_min) & (z < z_max)
        rows.append((step, int(alive.sum()), int(inside.sum())))
    return np.array(rows, dtype=np.int64)


def pairwise_separation(vtkhdf_path: Path, pair_idx: np.ndarray,
                        stride: int) -> np.ndarray:
    """Geometric-mean pairwise separation vs time for a fixed pair set.

    pair_idx: (n_pairs, 2) int array — indices into the particle list.
              Sample the seed step, pick random pairs among alive
              particles within a small neighbourhood.

    Returns (n_samples, 2) with columns (step, geometric_mean_sep).
    """
    max_needed = int(pair_idx.max()) + 1
    rows: list[tuple[int, float]] = []
    for step, pts, _esc in _iter_positions_vtkhdf(
        vtkhdf_path, stride=stride, n_particle_cap=max_needed,
    ):
        a = pts[pair_idx[:, 0]]
        b = pts[pair_idx[:, 1]]
        sep = np.linalg.norm(a - b, axis=1)
        # Guard against exact-zero separations (would break log)
        sep = np.maximum(sep, 1e-30)
        gmean = float(np.exp(np.log(sep).mean()))
        rows.append((step, gmean))
    return np.array(rows, dtype=np.float64)


def sample_pair_indices(vtkhdf_path: Path, n_pairs: int,
                        radius: float, rng_seed: int) -> np.ndarray:
    """Pick n_pairs random pairs of alive-at-seed particles whose seed
    positions are within `radius` of each other.

    Uses only the step-0 snapshot to avoid loading the whole trajectory.
    """
    import h5py
    with h5py.File(str(vtkhdf_path), "r") as f:
        vtkhdf = f["VTKHDF"]
        n_pts_per_step = vtkhdf["NumberOfPoints"][:]
        pts_offsets = vtkhdf["Steps"]["PointOffsets"][:]
        step0_start = int(pts_offsets[0])
        step0_count = int(n_pts_per_step[0])
        seeds = np.asarray(
            vtkhdf["Points"][step0_start:step0_start + step0_count],
            dtype=np.float32,
        )

    rng = np.random.default_rng(rng_seed)
    n = seeds.shape[0]
    # Pick random anchors, then find any alive-at-seed particle within
    # `radius`.  Using scipy KDTree would be cleaner but avoiding the
    # dependency.
    picked: list[tuple[int, int]] = []
    max_tries = n_pairs * 10
    for _ in range(max_tries):
        if len(picked) >= n_pairs:
            break
        i = int(rng.integers(0, n))
        # 128-particle random probe around i
        j_pool = rng.integers(0, n, size=128)
        d = np.linalg.norm(seeds[j_pool] - seeds[i], axis=1)
        d[j_pool == i] = np.inf
        mask = d <= radius
        if not mask.any():
            continue
        j = int(j_pool[mask][0])
        picked.append((i, j))
    return np.array(picked, dtype=np.int64)


def _write_csv(path: Path, header: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def _fit_exponential_slope(t_arr: np.ndarray, sep_arr: np.ndarray) -> float:
    """Fit log(sep) = a + b * t, return b.  Positive b == chaotic
    exponential separation; b close to 0 == no exponential growth."""
    if len(t_arr) < 3:
        return float('nan')
    logs = np.log(sep_arr)
    b, _ = np.polyfit(t_arr, logs, 1)
    return float(b)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--fom-vtkhdf", type=Path, required=True)
    ap.add_argument("--rom-vtkhdf", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory to write CSVs (created if missing).")

    # Annular probe (defaults match the r-binning from
    # rom_reconstruction_findings.md § 5).
    ap.add_argument("--annulus-r-min", type=float, default=0.005)
    ap.add_argument("--annulus-r-max", type=float, default=0.010)
    ap.add_argument("--annulus-z-min", type=float, default=-1.0)
    ap.add_argument("--annulus-z-max", type=float, default=1.0)

    # Pairwise separation
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--pair-radius", type=float, default=1.5e-3,
                    help="Seed-time neighbour radius for building pairs.")
    ap.add_argument("--rng-seed", type=int, default=42)

    ap.add_argument("--stride", type=int, default=10,
                    help="Sample every N-th tracking step (both diagnostics).")
    ap.add_argument("--plot", action="store_true",
                    help="Also write a PNG summary if matplotlib is available.")
    args = ap.parse_args()

    for label, path in [("FOM", args.fom_vtkhdf), ("ROM", args.rom_vtkhdf)]:
        if not path.exists():
            print(f"ERROR: {label} archive not found: {path}", file=sys.stderr)
            return 3

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Diagnostic 1: residence time in annulus
    # -----------------------------------------------------------------
    print(f"[mixing] Residence-time in annulus "
          f"r ∈ [{args.annulus_r_min}, {args.annulus_r_max}]")
    fom_res = residence_time(args.fom_vtkhdf,
                             args.annulus_r_min, args.annulus_r_max,
                             args.annulus_z_min, args.annulus_z_max,
                             stride=args.stride)
    rom_res = residence_time(args.rom_vtkhdf,
                             args.annulus_r_min, args.annulus_r_max,
                             args.annulus_z_min, args.annulus_z_max,
                             stride=args.stride)
    print(f"  FOM: step 0 -> {int(fom_res[0, 2]):,} inside; "
          f"step {int(fom_res[-1, 0])} -> {int(fom_res[-1, 2]):,} inside")
    print(f"  ROM: step 0 -> {int(rom_res[0, 2]):,} inside; "
          f"step {int(rom_res[-1, 0])} -> {int(rom_res[-1, 2]):,} inside")

    _write_csv(args.out_dir / "residence_time.csv",
               ["step", "fom_n_alive", "fom_n_inside_annulus",
                "rom_n_alive", "rom_n_inside_annulus"],
               ((int(fom_res[i, 0]),
                 int(fom_res[i, 1]), int(fom_res[i, 2]),
                 int(rom_res[i, 1]), int(rom_res[i, 2]))
                for i in range(min(len(fom_res), len(rom_res)))))

    # -----------------------------------------------------------------
    # Diagnostic 2: pairwise separation
    # -----------------------------------------------------------------
    print(f"\n[mixing] Pairwise separation (n_pairs={args.n_pairs}, "
          f"pair_radius={args.pair_radius})")
    pair_idx = sample_pair_indices(args.fom_vtkhdf, args.n_pairs,
                                   args.pair_radius, args.rng_seed)
    print(f"  built {len(pair_idx):,} pair(s) from FOM seed positions "
          f"(reused for ROM to keep the pair set constant)")
    fom_sep = pairwise_separation(args.fom_vtkhdf, pair_idx,
                                  stride=args.stride)
    rom_sep = pairwise_separation(args.rom_vtkhdf, pair_idx,
                                  stride=args.stride)

    fom_slope = _fit_exponential_slope(fom_sep[:, 0], fom_sep[:, 1])
    rom_slope = _fit_exponential_slope(rom_sep[:, 0], rom_sep[:, 1])
    print(f"  FOM d(log sep)/d(step) slope: {fom_slope:.4e}")
    print(f"  ROM d(log sep)/d(step) slope: {rom_slope:.4e}")
    if abs(fom_slope) > 1e-12:
        print(f"  ratio ROM / FOM             : {rom_slope / fom_slope:.4f}")

    _write_csv(args.out_dir / "pairwise_separation.csv",
               ["step", "fom_gmean_sep", "rom_gmean_sep"],
               ((int(fom_sep[i, 0]), float(fom_sep[i, 1]),
                 float(rom_sep[i, 1]))
                for i in range(min(len(fom_sep), len(rom_sep)))))

    # Manifest summarising the two diagnostics
    with (args.out_dir / "manifest.txt").open("w") as fh:
        fh.write(f"fom_vtkhdf         : {args.fom_vtkhdf}\n")
        fh.write(f"rom_vtkhdf         : {args.rom_vtkhdf}\n")
        fh.write(f"annulus r range    : "
                 f"[{args.annulus_r_min}, {args.annulus_r_max}]\n")
        fh.write(f"annulus z range    : "
                 f"[{args.annulus_z_min}, {args.annulus_z_max}]\n")
        fh.write(f"n_pairs            : {args.n_pairs}\n")
        fh.write(f"pair_radius        : {args.pair_radius}\n")
        fh.write(f"stride             : {args.stride}\n")
        fh.write(f"fom_seed_slope     : {fom_slope:.6e}\n")
        fh.write(f"rom_seed_slope     : {rom_slope:.6e}\n")
        if abs(fom_slope) > 1e-12:
            fh.write(f"rom_over_fom_slope : {rom_slope / fom_slope:.4f}\n")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  (skip PNG summary: matplotlib not available)")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(fom_res[:, 0], fom_res[:, 2], label="FOM")
            axes[0].plot(rom_res[:, 0], rom_res[:, 2], label="ROM")
            axes[0].set_xlabel("step")
            axes[0].set_ylabel("n particles inside annulus")
            axes[0].set_title("Residence time")
            axes[0].legend()
            axes[1].semilogy(fom_sep[:, 0], fom_sep[:, 1], label="FOM")
            axes[1].semilogy(rom_sep[:, 0], rom_sep[:, 1], label="ROM")
            axes[1].set_xlabel("step")
            axes[1].set_ylabel("gmean pair separation (m)")
            axes[1].set_title("Pairwise separation")
            axes[1].legend()
            fig.tight_layout()
            fig.savefig(args.out_dir / "mixing_summary.png", dpi=120)
            print(f"  wrote {args.out_dir / 'mixing_summary.png'}")

    print(f"\n[mixing] Wrote CSVs and manifest under {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
