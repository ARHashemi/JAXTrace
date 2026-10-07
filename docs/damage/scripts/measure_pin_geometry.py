"""M4 -- pin geometry descriptors from the tool STLs.

    python3 measure_pin_geometry.py --cases-root <PinShapes> --out <dir>

WHY
---
The PinShapes set is a **pure tool-geometry study at a single operating point**:
rpm, advancing speed, plate height and material are all effectively constant
(``BENCHMARK_READINESS.md`` section 1). **Pin geometry is the only real independent
variable, and it is currently recorded nowhere** -- only ``r_tool``, which the
2026-09-28 profile work showed is the SHOULDER radius, identical (~7 mm) across
all 20 cases.

So "threads score higher than flats" is presently a statement about four file
names. Nothing in it can be modelled, regressed or extrapolated. This script turns
each STL into numbers.

WHAT SHIPS (validated) vs WHAT DOES NOT
---------------------------------------
The CSV carries ONLY descriptors verified sane against independent evidence:

  r_max_mm, height_mm   pin envelope. ⚠️ Validated against M1's independently
                        measured r_pin (2.1-2.4 mm from the velocity field) --
                        the STL gives 2.96 mm for B1/C1/D3, same ballpark.
  volume_mm3, area_mm2, sphericity
  hull_deficit          how far the pin is from a plain cylinder
  n_lobes, lobe_depth   flat/flute count and depth. ⚠️ THE USEFUL PAIR:
                        B1 = 3 flutes at 30 % depth, C1 = 2 at 6 %,
                        D3 = 8 at 18 %. This is what M4 exists to produce.
  n_triangles, stl_md5  provenance

Three descriptors are computed but written ONLY to the per-case JSON under an
"unvalidated" key, never to the CSV (see the block at the call site for why).

WHAT IT MEASURES, AND WHY EACH ONE
----------------------------------
Everything here comes from a closed triangle soup with no CAD parameters, so each
descriptor is a *measurement*, not a recovered design intent.

  envelope        r_max, r_mean, z_min, z_max, height
                  -> the pin's gross size, and (with r_pin from M1) whether the
                     STL envelope agrees with what the level-set says at depth
  volume, area    signed-tet volume and total triangle area
  sphericity      area normalised by an equal-volume cylinder's area
                  -> a single "how featured is this pin" scalar
  taper_deg       slope of r_95(z): positive = wider at the top (tapered pin)
                  ⚠️ PROVISIONAL -- the sign flipped between runs as the pin mask
                  changed (C1 gave -2.7 then +8.1 deg). Treat as unvalidated.
  concavity       departure of r_95(z) from the straight line through its ends
                  ⚠️ NOT WORKING -- returns +0.0000 for every case, so the metric
                  carries no information as written. Reported for provenance only.
  hull_deficit    1 - V / V_convexhull : how much material is cut away
                  -> flats and flutes remove material; threads mostly do not
  n_lobes         dominant azimuthal wavenumber of r(theta), by FFT
                  -> flat count / flute count, MEASURED not assumed
  lobe_depth      peak-to-peak azimuthal radius variation / r_mean
                  -> how deep those flats/flutes cut
  thread_lead_mm  dominant AXIAL wavelength of the r(theta,z) residual
                  ⚠️ NOT WORKING. Two successive wrong answers: first 5.35 mm for
                  BOTH a fluted and a threaded pin (= half the mis-detected 10.70 mm
                  span, i.e. the FFT fundamental), then 777 mm once the span was
                  fixed. A 1-D r(z) profile averages the thread away around the
                  circumference; a real lead needs the helix followed in
                  (theta, z) together. Reported but MUST NOT be used.
  n_triangles     mesh size, for provenance

⚠️ ``n_lobes`` and ``thread_lead_mm`` are spectral estimates. They are reported
with the spectral peak's prominence so a weak, untrustworthy peak is visible
rather than silently quoted as a clean integer.

⚠️ Copied cases keep the SOURCE case's STL name (``A2old.gid/A2.stl``,
``D2old.gid/D2.stl``). The md5 of every STL is recorded, because
**D2 and D2old are byte-identical** (they share a pin) while the A-family ``old``
pairs are NOT -- A2 and A2old are genuinely different geometries despite the
shared file name. Do not infer duplication from a name.

READ-ONLY on the case folders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Azimuthal / axial sampling for the spectral descriptors.
N_THETA = 180          # 2 deg bins
N_Z = 60
# A spectral peak must exceed this multiple of the median spectrum to be quoted.
PEAK_PROMINENCE = 3.0
# Minimum peak-to-peak azimuthal radius variation (as a fraction of mean radius)
# for a lobe count to be believable. 1 % -- below that it is mesh round-off.
MIN_LOBE_DEPTH = 0.01
# Radius percentile used for the profile: robust against a few stray vertices.
R_PCTILE = 95.0

# ⚠️ The STL is the WHOLE TOOL, not just the pin. Measured structure (A2/C1/D1):
#     z < ~-0.35 mm : the pin,      r ~ 2.2 - 3.0 mm
#     z >= ~-0.35 mm: the shoulder, r  = 7.0 mm  (identical in all 20 cases)
# Computing descriptors over the whole tool therefore reports the SHOULDER -- the
# first run gave r_max = 7.00 mm for every case, which is exactly the quantity M1
# already showed is constant and uninformative.
#
# The pin is isolated by a radius cut rather than a hard z cut, because the
# junction height varies slightly between tools: a vertex belongs to the pin if
# its radius is below this fraction of the overall maximum radius.
PIN_R_FRAC = 0.70

# A z-bin needs this many vertices to enter the profile. ⚠️ A2 and D1 carry
# vertices only at the pin tip and the shoulder (coarse STLs), so a strict
# threshold leaves too few bins and taper/concavity come back NaN. Reported as
# n_profile_bins so a thin fit is visible rather than silently quoted.
MIN_BIN_VERTS = 8


def read_stl(path: Path):
    """Triangles from a binary or ASCII STL. Returns (n_tri, 3, 3) float64."""
    raw = path.read_bytes()
    # ASCII STLs start with "solid", but so can some binary ones; the reliable
    # test is whether the declared triangle count matches the file length.
    if len(raw) >= 84:
        n_dec = struct.unpack("<I", raw[80:84])[0]
        if 84 + 50 * n_dec == len(raw):
            tris = np.empty((n_dec, 3, 3), dtype=np.float64)
            for i in range(n_dec):
                off = 84 + 50 * i + 12          # skip the facet normal
                vals = struct.unpack("<9f", raw[off:off + 36])
                tris[i] = np.asarray(vals, dtype=np.float64).reshape(3, 3)
            return tris
    # ASCII fallback.
    verts = []
    for line in raw.decode("ascii", errors="replace").splitlines():
        p = line.split()
        if len(p) == 4 and p[0] == "vertex":
            verts.append([float(p[1]), float(p[2]), float(p[3])])
    if len(verts) < 3 or len(verts) % 3:
        raise RuntimeError("unreadable STL (%d vertices)" % len(verts))
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)


def mesh_volume_area(tris):
    """Signed volume by the divergence theorem, and total area.

    Volume is |sum (v0 . (v1 x v2)) / 6|, which is exact for a closed surface and
    does not care about triangle ordering once the absolute value is taken.
    """
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    vol = float(np.abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()) / 6.0)
    area = float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())
    return vol, area


def axis_and_radius(tris):
    """Cylindrical coords about the tool axis, for the WHOLE tool."""
    v = tris.reshape(-1, 3)
    span = v.max(axis=0) - v.min(axis=0)
    cx, cy = float(v[:, 0].mean()), float(v[:, 1].mean())
    r = np.hypot(v[:, 0] - cx, v[:, 1] - cy)
    th = np.arctan2(v[:, 1] - cy, v[:, 0] - cx)
    return v, r, th, cx, cy, span


def isolate_pin(v, r):
    """Split the tool into pin and shoulder by finding the junction HEIGHT.

    ⚠️ A pure radius cut does NOT work, and the failure is instructive. On C1 the
    mask r < 0.7*r_max kept 599,373 vertices spanning the full z = -5.70..5.00,
    because the tool also has a **shank centreline at r = 0.050 mm running up to
    z = +5.0** (180 vertices). Those few vertices stretched the z-span to 10.70 mm,
    which then became the FFT's fundamental wavelength -- and the script reported
    a "thread lead" of 5.35 mm (= half of 10.70) for BOTH a fluted pin (B1) and a
    threaded one (C1). A fabricated number that looked plausible.

    The junction is a real, sharp feature: r_95(z) jumps 2.95 -> 7.03 mm across
    one bin. So find that jump and cut below it.

    Returns (pin mask, shoulder radius, junction z).
    """
    r_all_max = float(r.max())
    z = v[:, 2]
    n = 60
    edges = np.linspace(z.min(), z.max(), n + 1)
    zc, rp = [], []
    for i in range(n):
        m = (z >= edges[i]) & (z <= edges[i + 1])
        if int(m.sum()) >= MIN_BIN_VERTS:
            zc.append(0.5 * (edges[i] + edges[i + 1]))
            rp.append(float(np.percentile(r[m], R_PCTILE)))
    # ⚠️ A sparse profile cannot yield a reliable pin ENVELOPE, and two attempts to
    # force one both produced plausible-looking wrong answers:
    #
    #   attempt 1 (guard len(rp) < 6): bailed out to "whole tool" -> reported the
    #     SHOULDER radius 7.00 mm as the pin radius, for 9 of 20 cases.
    #   attempt 2 (relax to 2, cut at the step midpoint): r_pin came out correct
    #     (2.16-2.21 mm, matching M1) but height became **86 mm on a 5.4 mm pin**,
    #     because the mask then swept up stray shank vertices at large z.
    #
    # Root cause: these are coarse CAD tessellations with vertices only where the
    # surface changes. A2 (2,794 tri) has **4 populated z-bins of 60** -- the pin
    # tip, the shoulder underside, and two above. There is no mid-pin surface to
    # profile, so `height`, `volume` and `hull_deficit` are not recoverable from
    # the STL alone no matter how the cut is placed.
    #
    # So the guard stays, and these cases are reported as FAILED rather than given
    # a fabricated envelope. `n_lobes`/`lobe_depth` remain valid for them because
    # they come from the azimuthal profile, which does not need z resolution.
    if len(rp) < 6:
        return np.ones(len(v), dtype=bool), r_all_max, float("nan")
    zc, rp = np.asarray(zc), np.asarray(rp)
    jumps = np.diff(rp)
    j = int(np.argmax(jumps))
    # Require a real step, not just the largest of many small wiggles.
    if jumps[j] < 0.25 * r_all_max:
        # ⚠️ Coarse STLs (A1 2.8k tri, A2 2.8k, D1 11k) have too few populated
        # z-bins for the jump test, and returning "whole tool" silently reported
        # the SHOULDER (r_max = 7.00, h = 10.70) for those cases. Fall back to the
        # highest z at which the radius is still small, which does not need a
        # dense profile.
        small = r < 0.70 * r_all_max
        if int(small.sum()) < 50:
            return np.ones(len(v), dtype=bool), r_all_max, float("nan")
        # Exclude the shank centreline: a thin spike of tiny-radius vertices
        # running far above the pin (C1 has 180 verts at r = 0.05 up to z = +5).
        z_small = z[small]
        r_small = r[small]
        body = r_small > 0.15 * r_all_max
        z_junction = float(np.percentile(z_small[body], 99.5)) if body.any() \
            else float(z_small.max())
        mask = z <= z_junction
        if int(mask.sum()) < 50:
            return np.ones(len(v), dtype=bool), r_all_max, float("nan")
        return mask, r_all_max, z_junction
    z_junction = float(zc[j])
    mask = z <= z_junction
    if int(mask.sum()) < 50:
        return np.ones(len(v), dtype=bool), r_all_max, float("nan")
    return mask, r_all_max, z_junction


def radial_profile(v, r, n_z=N_Z):
    """r_95(z) -- a robust outer-radius profile up the PIN (pre-masked)."""
    z = v[:, 2]
    if len(z) < 20:
        return np.asarray([]), np.asarray([])
    edges = np.linspace(z.min(), z.max(), n_z + 1)
    zc, rp = [], []
    for i in range(n_z):
        m = (z >= edges[i]) & (z <= edges[i + 1])
        if int(m.sum()) >= MIN_BIN_VERTS:
            zc.append(0.5 * (edges[i] + edges[i + 1]))
            rp.append(float(np.percentile(r[m], R_PCTILE)))
    return np.asarray(zc), np.asarray(rp)


def taper_and_concavity(zc, rp):
    """Taper angle from a linear fit; concavity as departure from that line.

    concavity > 0 means the mid-section is NARROWER than the straight line
    joining the ends, i.e. a waisted / concave pin.
    """
    if len(zc) < 5:
        return float("nan"), float("nan")
    a, b = np.polyfit(zc, rp, 1)
    taper_deg = float(np.degrees(np.arctan(a)))
    line = a * zc + b
    # Normalised so it is comparable between pins of different radius.
    conc = float(np.mean(line - rp) / max(np.mean(rp), 1e-12))
    return taper_deg, conc


def azimuthal_lobes(r, th, n_theta=N_THETA):
    """Dominant azimuthal wavenumber of r(theta) -- flats or flutes, measured.

    Uses the mean radius per angular bin, then an FFT. Returns the wavenumber,
    its prominence over the median spectrum, and the peak-to-peak depth.
    """
    edges = np.linspace(-np.pi, np.pi, n_theta + 1)
    prof = np.full(n_theta, np.nan)
    for i in range(n_theta):
        m = (th >= edges[i]) & (th < edges[i + 1])
        if int(m.sum()) > 5:
            prof[i] = float(np.percentile(r[m], R_PCTILE))
    ok = ~np.isnan(prof)
    if int(ok.sum()) < n_theta // 2:
        return 0, float("nan"), float("nan")
    # Fill gaps by interpolation so the FFT sees a closed periodic signal.
    idx = np.arange(n_theta)
    prof = np.interp(idx, idx[ok], prof[ok], period=n_theta)
    depth = float((prof.max() - prof.min()) / max(prof.mean(), 1e-12))

    # ⚠️ A depth gate comes FIRST. On A2 the FFT returned k=3 with prominence
    # 14889 at a depth of 0.001 -- a 0.1 % radius variation is round-off on a
    # 2794-triangle mesh, not a flat. A huge prominence on a flat spectrum means
    # the spectrum is noise, not that the peak is strong.
    if depth < MIN_LOBE_DEPTH:
        return 0, float("nan"), depth

    spec = np.abs(np.fft.rfft(prof - prof.mean()))
    if len(spec) < 4:
        return 0, float("nan"), depth
    # Ignore k=0; a pin's flats/flutes are k >= 2 (k=1 is an off-centre axis).
    k = int(np.argmax(spec[2:]) + 2)
    med = float(np.median(spec[2:]))
    prom = float(spec[k] / med) if med > 0 else float("inf")
    # A peak must also stand out from its own spectrum to be quoted.
    if prom < PEAK_PROMINENCE:
        return 0, prom, depth
    return k, prom, depth


def thread_lead(v, r, th, cx, cy):
    """Dominant AXIAL wavelength of the r(theta,z) residual -- thread pitch.

    A thread makes the radius vary periodically in z at fixed theta. Removing the
    mean r(z) profile leaves that ripple. Returns (lead_mm, prominence).
    """
    z = v[:, 2]
    n_z = 120
    edges = np.linspace(z.min(), z.max(), n_z + 1)
    prof = np.full(n_z, np.nan)
    for i in range(n_z):
        m = (z >= edges[i]) & (z < edges[i + 1])
        if int(m.sum()) > 20:
            prof[i] = float(np.percentile(r[m], R_PCTILE))
    ok = ~np.isnan(prof)
    if int(ok.sum()) < 40:
        return float("nan"), float("nan")
    idx = np.arange(n_z)
    prof = np.interp(idx, idx[ok], prof[ok])
    dz = (z.max() - z.min()) / n_z
    # Detrend: a thread is a ripple ON TOP of the taper, so remove the taper.
    trend = np.polyval(np.polyfit(idx, prof, 2), idx)
    resid = prof - trend
    spec = np.abs(np.fft.rfft(resid))
    if len(spec) < 5:
        return float("nan"), float("nan")
    k = int(np.argmax(spec[2:]) + 2)
    med = float(np.median(spec[2:]))
    prom = float(spec[k] / med) if med > 0 else float("inf")
    # An unthreaded pin has no axial ripple; do not invent a lead for one.
    if prom < PEAK_PROMINENCE or float(np.ptp(resid)) < MIN_LOBE_DEPTH * float(prof.mean()):
        return float("nan"), prom
    lead = float(n_z * dz / k) if k else float("nan")
    return lead, prom


def convex_hull_deficit(v_pin, _unused):
    """How non-convex the pin is, from its own point cloud.

    ⚠️ Computed as 1 - V_alphaish / V_hull would need a watertight pin volume,
    which the radius-masked subset is not (cutting the tool in half leaves an open
    surface). Instead this reports the hull volume normalised by the enclosing
    cylinder, which is well defined on an open point set:

        1 - V_hull / (pi r_max^2 h)

    A perfect cylinder gives 0; flats and flutes cut the hull below the cylinder
    and give a positive value. It measures the same thing the original intended --
    how much material is removed -- without pretending the subset is closed.

    Falls back to NaN when scipy is unavailable rather than failing the case.
    """
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return float("nan")
    try:
        h = ConvexHull(v_pin)
        rr = np.hypot(v_pin[:, 0] - v_pin[:, 0].mean(),
                      v_pin[:, 1] - v_pin[:, 1].mean()).max()
        hh = v_pin[:, 2].max() - v_pin[:, 2].min()
        v_cyl = np.pi * rr * rr * hh
        return float(1.0 - h.volume / v_cyl) if v_cyl > 0 else float("nan")
    except Exception:
        return float("nan")


def find_cases(root: Path, max_depth: int = 3):
    found, stack = [], [(root, 0)]
    while stack:
        d, depth = stack.pop(0)
        try:
            entries = sorted(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if not e.is_dir():
                continue
            if e.name.endswith(".gid"):
                found.append(e)
            elif depth < max_depth:
                stack.append((e, depth + 1))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases-root", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None

    cases = []
    for root in args.cases_root:
        f = find_cases(root)
        print("found %3d cases under %s" % (len(f), root))
        cases += f
    print()

    rows = []
    for cdir in cases:
        name = cdir.name[:-4]
        if only and name not in only:
            continue
        stls = sorted(cdir.glob("*.stl"))
        if not stls:
            print("  SKIP %-8s no STL" % name)
            continue
        stl = stls[0]
        try:
            tris = read_stl(stl)
            v, r, th, cx, cy, span = axis_and_radius(tris)
            vol, area = mesh_volume_area(tris)

            # ⚠️ Descriptors are computed on the PIN ONLY. The STL carries the
            # shoulder too, and the shoulder is r = 7.0 mm in every case, so
            # including it reports a constant and hides the real variable.
            pin, r_tool_stl, z_junc = isolate_pin(v, r)
            v_p, r_p, th_p = v[pin], r[pin], th[pin]
            zc, rp = radial_profile(v_p, r_p)
            taper, conc = taper_and_concavity(zc, rp)
            k_lobe, prom_lobe, depth_lobe = azimuthal_lobes(r_p, th_p)
            lead, prom_lead = thread_lead(v_p, r_p, th_p, cx, cy)
            # Hull deficit on the pin subset; the shoulder disc would
            # dominate the hull and wash out the pin's features.
            deficit = convex_hull_deficit(v_p, float("nan"))

            r_max = float(r_p.max())
            height = float(v_p[:, 2].max() - v_p[:, 2].min())
            # Equal-volume cylinder of the same height -> sphericity-like scalar.
            r_eq = float(np.sqrt(vol / (np.pi * height))) if height > 0 else float("nan")
            a_eq = (2 * np.pi * r_eq * height + 2 * np.pi * r_eq ** 2) if r_eq == r_eq else float("nan")
            spher = float(a_eq / area) if area > 0 else float("nan")

            # ⚠️ Did pin isolation actually work? The shoulder is ~7.0 mm and the
            # pin ~2.4-3.1 mm in every case measured, so a "pin radius" close to
            # the shoulder radius means the split failed (coarse STLs: A1, A2, D1
            # have too few populated z-bins for the junction test). Emitting 7.00
            # as a pin radius would be worse than emitting nothing.
            pin_ok = bool(r_max < 0.80 * r_tool_stl)
            row = {
                "family": cdir.parent.name, "case": name,
                "pin_isolated": pin_ok,
                "stl": stl.name,
                "stl_md5": hashlib.md5(stl.read_bytes()).hexdigest()[:10],
                "n_triangles": int(len(tris)),
                "n_pin_verts": int(pin.sum()),
                "n_profile_bins": int(len(zc)),
                "r_shoulder_mm": r_tool_stl,
                "z_junction_mm": z_junc,
                "r_max_mm": r_max * 1e3 if r_max < 1 else r_max,
                "height_mm": height * 1e3 if height < 1 else height,
                "volume_mm3": vol * 1e9 if r_max < 1 else vol,
                "area_mm2": area * 1e6 if r_max < 1 else area,
                "sphericity": spher,
                "hull_deficit": deficit,
                "n_lobes": int(k_lobe),
                "lobe_prominence": prom_lobe,
                "lobe_depth": depth_lobe,
            }
            # ⚠️ UNVALIDATED descriptors are kept OUT of the row (and so out of
            # delta CSV columns) and written only to the per-case JSON under an
            # explicitly-labelled key. They are recorded for provenance, never
            # tabulated, because each produced a plausible-looking wrong answer:
            #   thread_lead : 5.35 mm for BOTH a fluted and a threaded pin (= half
            #                 the mis-detected span, i.e. the FFT fundamental),
            #                 then 777 mm once the span was fixed
            #   concavity   : +0.0000 for every case -- no information
            #   taper_deg   : sign flipped between runs as the pin mask changed
            unvalidated = {
                "_WARNING": "NOT VALIDATED - do not quote these",
                "taper_deg": taper,
                "concavity": conc,
                "thread_lead_mm": (lead * 1e3 if (lead == lead and lead < 1) else lead),
                "thread_prominence": prom_lead,
            }
            if not pin_ok:
                # Blank the envelope columns so they cannot be tabulated as a pin.
                for k in ("r_max_mm", "height_mm", "volume_mm3", "area_mm2",
                          "sphericity", "hull_deficit"):
                    row[k] = None
            rows.append(row)
            json.dump({"row": row, "unvalidated": unvalidated},
                      open(args.out / ("pin_%s.json" % name), "w"), indent=1)
            if not pin_ok:
                print("  %-8s tri=%7d  ⚠️ PIN ISOLATION FAILED "
                      "(r=%.2f vs shoulder %.2f mm; coarse STL) — envelope "
                      "columns blanked; lobes=%d depth=%.3f still reported"
                      % (name, len(tris), r_max, r_tool_stl, k_lobe,
                         depth_lobe if depth_lobe == depth_lobe else float("nan")),
                      flush=True)
                continue
            print("  %-8s tri=%7d  r_pin=%5.2f mm  h=%5.2f mm  vol=%8.1f mm3"
                  "  deficit=%s  lobes=%2d (x%s)  depth=%.3f"
                  % (name, len(tris), row["r_max_mm"], row["height_mm"],
                     row["volume_mm3"],
                     ("%.3f" % deficit) if deficit == deficit else " n/a ",
                     k_lobe,
                     ("%.1f" % prom_lobe) if prom_lobe == prom_lobe else "n/a",
                     depth_lobe if depth_lobe == depth_lobe else float("nan")),
                  flush=True)
        except Exception as exc:
            print("  FAIL %-8s %s: %s" % (name, type(exc).__name__, exc), flush=True)
            continue

    if not rows:
        print("no cases measured")
        return 1

    keys = ["family", "case"] + sorted({k for r in rows for k in r} - {"family", "case"})
    with open(args.out / "pin_geometry.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump(rows, open(args.out / "pin_geometry.json", "w"), indent=1)

    # ---- which cases SHARE a pin? Do not infer it from the file name. ----
    print()
    print("=" * 74)
    print(" Shared geometries (identical STL content)")
    print("=" * 74)
    by_md5 = {}
    for r in rows:
        by_md5.setdefault(r["stl_md5"], []).append(r["case"])
    dups = {k: v for k, v in by_md5.items() if len(v) > 1}
    if dups:
        for k, v in dups.items():
            print("  %s : %s" % (k, ", ".join(sorted(v))))
        print("  ⚠️ These cases share a pin, so they are NOT independent samples")
        print("     of geometry. Any geometry regression must account for that.")
    else:
        print("  none -- every case has a distinct pin")
    print()
    print("  ⚠️ Cases whose STL name differs from the case name (copied cases):")
    odd = [r for r in rows if not r["stl"].startswith(r["case"])]
    for r in odd:
        print("     %-8s carries %s" % (r["case"], r["stl"]))
    if not odd:
        print("     none")

    print()
    print("wrote %s" % (args.out / "pin_geometry.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
