"""Is an apparent void REAL, or just where the flow is fast?

⚠️ THE AMBIGUITY THIS RESOLVES. For steady incompressible flow, mass
conservation along a stream tube gives |u| A = const, so the tube NARROWS
where the flow is fast and an advected particle cloud has

    n(x) ~ 1 / A(x) ~ |u(x)|

Low particle density therefore means EITHER the flow is slow there (an
artefact of advection) OR material genuinely left (a real void). A raw
density map cannot tell them apart.

THE FIX. Form the FLUX-NORMALISED density

    rho_hat(x) = n(x) / |u(x)|

which is CONSTANT for a purely advected incompressible cloud. Any DEVIATION
from constant is a genuine material deficit, independent of flow speed.

⚠️ VALIDATE ON THE ANALYTIC CASE FIRST. rot_cyl_2026's field is
divergence-free BY CONSTRUCTION, so rho_hat there must come out FLAT. Whatever
spread it shows is the method's own noise floor -- binning, kernel width,
finite sampling -- and sets the threshold below which a 'void' on a real case
means nothing. Running it on cylA without that calibration would invite
reading noise as physics.

Usage:
    python3 flux_normalised_density.py --run DIR --case-post DIR \
        --pattern 'cylA_{timestep}.pvtu' --timestep 159 [--res 48]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np


def load_final(run: Path):
    """Final particle positions, alive only.

    ⚠️ Reads final_state.npz when present and falls back to the LAST STEP of
    particles.vtkhdf otherwise: the rot_cyl_2026 runs export only the vtkhdf,
    and the analytic calibration needs them.
    """
    fs = sorted(run.glob("*/final_state.npz"))
    if fs:
        z = np.load(fs[0])
        P = np.asarray(z["positions"], dtype=np.float64)
        eid = np.asarray(z["element_ids"])
        # ⚠️ Escaped particles sit frozen at their last position and would
        # pile up a spurious density spike. Drop them.
        return P[eid >= 0]

    hs = sorted(run.glob("*/particles.vtkhdf"))
    if not hs:
        raise SystemExit(f"no final_state.npz or particles.vtkhdf under {run}")
    import h5py
    with h5py.File(hs[0], "r") as h:
        g = h["VTKHDF"]
        npts = g["NumberOfPoints"][:]
        i = len(npts) - 1
        off = int(np.sum(npts[:i])); m = int(npts[i])
        P = np.asarray(g["Points"][off:off + m], dtype=np.float64)
        if "Escaped" in g["PointData"]:
            esc = np.asarray(g["PointData"]["Escaped"][off:off + m])
            P = P[esc == 0]
    return P


def speed_on_grid(post: Path, pattern: str, ts: int, centres: np.ndarray,
                  field: str = "Displacement") -> np.ndarray:
    """|u| sampled at voxel centres by nearest mesh node.

    ⚠️ Nearest-node, not barycentric: the voxel grid is much coarser than the
    mesh, so a higher-order sample would be false precision. What matters is
    the LOCAL MEAN speed in the voxel, and nearest-node over a fine mesh
    approximates that well.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from jaxtrace.gpu.mesh_loader_timedep import load_field_sequences_from_pvtu
    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
    from scipy.spatial import cKDTree
    nodes, conn, seq = load_field_sequences_from_pvtu(
        post, pattern, (ts, ts), [field], verbose=False, n_workers=8)
    o = deduplicate_nodes(nodes, conn, seq[field], verbose=False)
    nodes, vel = o[0], o[3][0]
    spd = np.linalg.norm(np.asarray(vel, dtype=np.float64), axis=1)
    _, idx = cKDTree(nodes).query(centres, k=1)
    return spd[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--case-post", required=True, type=Path)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--timestep", type=int, required=True)
    ap.add_argument("--field", default="Displacement")
    ap.add_argument("--res", type=int, default=48)
    ap.add_argument("--min-count", type=int, default=20,
                    help="voxels with fewer particles are masked: a handful "
                         "of particles is noise, not a density")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    P = load_final(a.run)
    print(f"  {len(P):,} alive particles")

    lo, hi = P.min(axis=0), P.max(axis=0)
    pad = 0.02 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    # ⚠️ ANISOTROPIC BINNING. The cylA cloud is a thin slab (extent ratios
    # ~6 : 4.6 : 1), so a CUBIC grid spends most of its voxels on empty space
    # and leaves only a few particles in each occupied one -- measured 0.7 %
    # occupancy and 2.7 particles/voxel at res 48, which is sampling noise,
    # not a density field. Size the bins so each voxel is roughly CUBIC IN
    # SPACE: --res sets the count along the LONGEST axis, the others follow.
    ext = hi - lo
    bins = np.maximum(np.round(a.res * ext / ext.max()).astype(int), 4)
    print(f"  grid {tuple(bins)}  (anisotropic: ~cubic voxels, "
          f"{np.prod(bins):,} total)")
    H, edges = np.histogramdd(P, bins=list(bins), range=list(zip(lo, hi)))
    centres = np.stack(np.meshgrid(
        *[0.5 * (e[1:] + e[:-1]) for e in edges], indexing="ij"), axis=-1
    ).reshape(-1, 3)

    spd = speed_on_grid(a.case_post, a.pattern, a.timestep, centres, a.field)
    n = H.ravel()

    occupied = n >= a.min_count
    print(f"  voxels {n.size:,}, occupied (>= {a.min_count} particles) "
          f"{int(occupied.sum()):,} ({100*occupied.mean():.1f} %)")
    if occupied.sum() < 50:
        raise SystemExit("too few occupied voxels; lower --res or --min-count")

    rho_raw = n[occupied].astype(np.float64)
    u = np.maximum(spd[occupied], 1e-30)
    rho_hat = rho_raw / u

    def stats(v, nm):
        v = v / np.median(v)          # normalise to the bulk
        print(f"    {nm:22s} p01 {np.percentile(v,1):7.4f}  "
              f"p50 {np.median(v):7.4f}  p99 {np.percentile(v,99):7.4f}  "
              f"CV {v.std()/v.mean():7.4f}")
        return v

    print("\n  ── relative to the median voxel ──")
    r = stats(rho_raw, "raw density n")
    rh = stats(rho_hat, "flux-normalised n/|u|")

    # ⚠️ The deficit map is what a reader should act on, and only AFTER the
    # analytic calibration says the floor is below the reported level.
    for thr in (0.5, 0.3, 0.1):
        f_raw = float((r < thr).mean())
        f_hat = float((rh < thr).mean())
        print(f"    voxels below {thr:.0%} of median:  raw {100*f_raw:6.2f} %"
              f"   flux-normalised {100*f_hat:6.2f} %")

    if a.out:
        np.savez_compressed(
            a.out, edges_x=edges[0], edges_y=edges[1], edges_z=edges[2],
            count=H, speed=spd.reshape(H.shape),
            occupied=occupied.reshape(H.shape))
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
